"""Complete the bounded correctness screen, then time both operand adapters.

Run serially on an idle XPU after xpu_w4a16_grouped_probe. A new process/output
directory is required for each start. Operator gains do not qualify a service.
"""

import argparse
import json
import shutil
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

from scripts.xpu_w4a16_capture import digest
from scripts.xpu_w4a16_grouped_probe import (
    adapt,
    difference,
    native,
    reference,
    validate_plan,
)


def measure(call):
    torch.xpu.synchronize()
    start = time.perf_counter()
    output = call()
    submitted = time.perf_counter()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    del output
    return {"wall_ms": elapsed * 1000, "host_ms": (submitted - start) * 1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in (
        Path(__file__),
        Path(__file__).with_name("xpu_w4a16_grouped_probe.py"),
    ):
        shutil.copy2(path, args.output / path.name)
    screen = json.loads((args.screen / "report.json").read_text())
    manifest = json.loads((args.capture / "manifest.json").read_text())
    plan = json.loads((args.screen / "plan.json").read_text())
    validate_plan(plan)
    import vllm_xpu_kernels._xpu_C as kernels

    if (
        screen["status"] != "initial_screen_passed_requires_remaining_gates"
        or digest(kernels.__file__) != screen["kernel_sha256"]
        or digest(args.capture / "manifest.json") != screen["capture_manifest_sha256"]
        or digest(args.screen / "plan.json") != screen["plan_sha256"]
        or manifest["torch"] != torch.__version__
    ):
        raise ValueError("requires passing screen with identical runtime and operands")
    report = {
        "argv": sys.argv,
        "screen_sha256": digest(args.screen / "report.json"),
        "cases": [],
        "status": "running",
        "service_qualified": False,
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    def require(row, name, result):
        row[name] = result
        save()
        if not result["pass"]:
            raise ValueError(f"correctness gate failed: {name}")

    try:
        for prefix in (
            plan["projections"][::-1] if args.reverse else plan["projections"]
        ):
            path = args.capture / f"{prefix}.weights.pt"
            if digest(path) != manifest["weights_sha256"][path.name]:
                raise ValueError("weight hash changed")
            weights = torch.load(path, weights_only=True)
            packed_cpu, scales_cpu = adapt(**weights)
            before = torch.xpu.memory_allocated()
            torch.xpu.reset_peak_memory_stats()
            load_start = time.perf_counter()
            q_nt = weights["q_nt"].to("xpu")
            original_scales = weights["scales"].to("xpu")
            zero = weights["zero"].to("xpu")
            packed, scales = packed_cpu.to("xpu"), scales_cpu.to("xpu")
            torch.xpu.synchronize()
            memory = {
                "both_formats_xpu_bytes": torch.xpu.memory_allocated() - before,
                "peak_xpu_bytes": torch.xpu.max_memory_allocated(),
                "load_seconds": time.perf_counter() - load_start,
            }
            for m in plan["M"][::-1] if args.reverse else plan["M"]:
                path = args.capture / f"{prefix}.m{m}.pt"
                meta = next(c for c in manifest["captures"] if c["file"] == path.name)
                if digest(path) != meta["sha256"]:
                    raise ValueError("activation hash changed")
                operands = torch.load(path, weights_only=True)
                x = operands["x"].to("xpu")
                if operands["bias"] is not None:
                    raise ValueError("this screen expects the captured bias-free MLP")
                row = {"prefix": prefix, "m": m, "memory": memory}
                report["cases"].append(row)
                tolerance = {
                    k: plan["gates"]["batch_separability"][k] for k in ("atol", "rtol")
                }

                control = partial(
                    torch.ops._xpu_C.int4_gemm_w4a16,
                    x,
                    q_nt,
                    None,
                    original_scales,
                    zero,
                    128,
                    None,
                )

                require(
                    row,
                    "control_replay",
                    difference(control().cpu(), operands["output"], 0, 0),
                )
                # The existing M=4 vs 4xM=1 contract plus rows spanning the
                # beginning, interior and final partial tile of real prefill.
                sample = x[[0, 1, m // 2, m - 1]].contiguous()
                independent = torch.cat(
                    [
                        native(sample[i : i + 1], packed, scales, canaries=True)
                        for i in range(4)
                    ]
                )
                require(
                    row,
                    "batch_4_vs_1",
                    difference(
                        native(sample, packed, scales, canaries=True),
                        independent,
                        **tolerance,
                    ),
                )
                full = native(x, packed, scales, canaries=True)
                require(
                    row,
                    "prefill_vs_1",
                    difference(full[[0, 1, m // 2, m - 1]], independent, **tolerance),
                )
                # Bias is absent in the checkpoint's selected layers; exercise
                # the optional ABI with a bounded deterministic synthetic bias.
                bias = torch.linspace(
                    -0.1, 0.1, packed.shape[1], device="xpu"
                ).bfloat16()
                biased = native(x, packed, scales, bias, canaries=True)
                ref = reference(x, weights["q_nt"], weights["scales"], bias)
                require(
                    row,
                    "bias_reference",
                    difference(
                        biased.cpu(),
                        ref,
                        **{
                            k: plan["gates"]["native_reference"][k]
                            for k in ("atol", "rtol")
                        },
                    ),
                )
                # Change all data pointers and values, shape and allocator
                # placement, then require exact restoration of the target.
                decoy_b, decoy_s = packed.roll(1, 1), scales.mul(1.5)
                for repeat in range(4):
                    decoy = native(sample.neg(), decoy_b, decoy_s, bias, canaries=True)
                    pressure = torch.full(
                        (1048573 * (repeat + 1),),
                        repeat,
                        dtype=torch.uint8,
                        device="xpu",
                    )
                    again = native(x, packed, scales, canaries=True)
                    require(row, f"repeat_{repeat}", difference(again, full, 0, 0))
                    del decoy, pressure, again
                unchanged = (
                    torch.equal(x.cpu(), operands["x"])
                    and torch.equal(packed.cpu(), packed_cpu)
                    and torch.equal(scales.cpu(), scales_cpu)
                    and torch.equal(q_nt.cpu(), weights["q_nt"])
                    and torch.equal(original_scales.cpu(), weights["scales"])
                )
                require(row, "noninterference", {"pass": unchanged})
                del sample, independent, full, bias, biased, decoy_b, decoy_s

                calls = {
                    "onednn": control,
                    "native": partial(native, x, packed, scales),
                }
                for call in calls.values():
                    for _ in range(3):
                        measure(call)
                row["pairs"] = []
                for repeat in range(20):
                    order = ["onednn", "native"]
                    if (repeat + args.reverse) % 2:
                        order.reverse()
                    pair = {"order": order}
                    for name in order:
                        pair[name] = measure(calls[name])
                    row["pairs"].append(pair)
                row["median_ms"] = {
                    name: statistics.median(p[name]["wall_ms"] for p in row["pairs"])
                    for name in calls
                }
                row["native_speedup"] = (
                    row["median_ms"]["onednn"] / row["median_ms"]["native"]
                )
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in ("prefix", "m", "median_ms", "native_speedup")
                        }
                    ),
                    flush=True,
                )
                save()
                del x, calls, call, control
            del packed, scales, q_nt, original_scales, zero
        report["status"] = "operator_comparison_complete"
        return 0
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
