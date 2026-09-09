"""Screen and compare experimental dense INT4 policies on captured operands.

Run serially on an idle XPU, with a fresh output directory and an explicit
candidate DSO. This reports operator evidence, never service qualification.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

POLICY_ENV = "VLLM_XPU_INT4_DENSE_POLICY"
ARMS = (
    "onednn",
    "original_uncached",
    "original_cached",
    "dense_128x128",
    "dense_256x128",
)
POLICIES = {"dense_128x128": "128x128", "dense_256x128": "256x128"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def block_order(index, reverse=False, arms=ARMS):
    """Each arm occurs twice at symmetric positions; rotate edge positions."""
    shift = index % len(arms)
    order = list(arms[shift:] + arms[:shift])
    if (index // len(arms) + reverse) % 2:
        order.reverse()
    return order + order[::-1]


def window_stable(windows, elapsed_seconds, minimum_seconds=1.0, limit=0.01, arms=ARMS):
    """Require three complete windows and independent stability of every arm."""
    if elapsed_seconds < minimum_seconds or len(windows) < 3:
        return False
    recent = windows[-3:]
    if any(w["seconds"] < 0.1 for w in recent):
        return False
    for arm in arms:
        values = [w["median_ms"][arm] for w in recent]
        if min(values) <= 0 or max(values) / min(values) - 1 > limit:
            return False
    return True


def bootstrap_interval(values, seed=0):
    """Percentile interval resampling complete balanced blocks, not calls."""
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(values, k=len(values))) for _ in range(2000)
    )
    return [medians[49], medians[1949]]


def summarize_blocks(blocks, arms=ARMS):
    totals = []
    for block in blocks:
        by_arm = {arm: [s for s in block["samples"] if s["arm"] == arm] for arm in arms}
        if any(len(samples) != 2 for samples in by_arm.values()):
            raise ValueError("each balanced block must contain two calls per arm")
        totals.append(
            {arm: sum(s["wall_ms"] for s in samples) for arm, samples in by_arm.items()}
        )
    if len(totals) < 3:
        raise ValueError("at least three complete blocks are required")
    third = max(1, len(totals) // 3)
    result = {}
    for arm in arms:
        changes = [100 * (b[arm] / b["onednn"] - 1) for b in totals]
        first = statistics.median(b[arm] / 2 for b in totals[:third])
        last = statistics.median(b[arm] / 2 for b in totals[-third:])
        result[arm] = {
            "median_block_mean_ms": statistics.median(b[arm] / 2 for b in totals),
            "median_latency_change_percent": statistics.median(changes),
            "block_bootstrap_95_percent": bootstrap_interval(changes),
            "paired_latency_changes_percent": changes,
            "first_third_ms": first,
            "last_third_ms": last,
            "first_to_last_change_percent": 100 * (last / first - 1),
            "drift_flag": max(first, last) / min(first, last) - 1 > 0.01,
        }
    return result


def set_policy(arm):
    value = POLICIES.get(arm)
    if value is None:
        os.environ.pop(POLICY_ENV, None)
    else:
        os.environ[POLICY_ENV] = value


def measure(torch, call):
    torch.xpu.synchronize()
    start = time.perf_counter()
    output = call()
    submitted = time.perf_counter()
    torch.xpu.synchronize()
    elapsed = time.perf_counter() - start
    del output
    return {"wall_ms": elapsed * 1000, "host_ms": (submitted - start) * 1000}


def run_block(torch, calls, index, reverse):
    samples = []
    for arm in block_order(index, reverse, tuple(calls)):
        # This setting is read dynamically by the C++ dispatcher. Its mutation
        # is outside measurement, after the preceding call has completed.
        set_policy(arm)
        samples.append({"arm": arm, **measure(torch, calls[arm])})
    return {"index": index, "samples": samples}


def warm(torch, calls, reverse):
    windows = []
    start = time.perf_counter()
    index = 0
    while time.perf_counter() - start < 10:
        window_start = time.perf_counter()
        blocks = []
        while time.perf_counter() - window_start < 0.1:
            blocks.append(run_block(torch, calls, index, reverse))
            index += 1
        windows.append(
            {
                "seconds": time.perf_counter() - window_start,
                "blocks": blocks,
                "median_ms": {
                    arm: statistics.median(
                        s["wall_ms"]
                        for b in blocks
                        for s in b["samples"]
                        if s["arm"] == arm
                    )
                    for arm in calls
                },
            }
        )
        elapsed = time.perf_counter() - start
        if window_stable(windows, elapsed, arms=tuple(calls)):
            return {"stable": True, "seconds": elapsed, "windows": windows}
    return {
        "stable": False,
        "seconds": time.perf_counter() - start,
        "windows": windows,
    }


def mapped_libraries():
    return sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if ".so" in line and line.split()[-1].startswith("/")
        }
    )


def prepare(torch, capture, manifest, case):
    from scripts.xpu_w4a16_grouped_probe import adapt

    weights_path = capture / f"{case['prefix']}.weights.pt"
    operand_path = capture / case["file"]
    if digest(weights_path) != manifest["weights_sha256"][weights_path.name]:
        raise ValueError("captured weights changed")
    if digest(operand_path) != case["sha256"]:
        raise ValueError("captured activation/output changed")
    weights = torch.load(weights_path, weights_only=True)
    operands = torch.load(operand_path, weights_only=True)
    packed, scales = adapt(**weights)
    if operands["bias"] is not None:
        raise ValueError("this screen requires the captured bias-free projections")
    return {
        "cpu_operands": operands,
        "cpu_packed": packed,
        "cpu_scales": scales,
        "x": operands["x"].to("xpu"),
        "packed": packed.to("xpu"),
        "scales": scales.to("xpu"),
        "q_nt": weights["q_nt"].to("xpu"),
        "original_scales": weights["scales"].to("xpu"),
        "zero": weights["zero"].to("xpu"),
        "rows": torch.tensor([case["m"]], dtype=torch.int32, device="xpu"),
    }


def native(torch, x, packed, scales, rows=None, canaries=False):
    m, k = x.shape
    n = packed.shape[1]
    if rows is None:
        rows = torch.tensor([m], dtype=torch.int32, device=x.device)
    storage = torch.empty((m + 2 if canaries else m, n), dtype=x.dtype, device=x.device)
    output = storage[1:-1] if canaries else storage
    if canaries:
        storage.fill_(-42)
        output.fill_(float("nan"))
    result = torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        x, None, packed, scales, None, output, rows, n, k, 1
    )
    if result.data_ptr() != output.data_ptr():
        raise ValueError("native did not return caller-owned output")
    if canaries:
        torch.xpu.synchronize()
        if not torch.all(storage[[0, -1]] == -42) or not torch.isfinite(output).all():
            raise ValueError("native output/canary gate failed")
    return output


def calls_for(torch, data, arms=ARMS):
    from functools import partial

    control = partial(
        torch.ops._xpu_C.int4_gemm_w4a16,
        data["x"],
        data["q_nt"],
        None,
        data["original_scales"],
        data["zero"],
        128,
        None,
    )
    uncached = partial(native, torch, data["x"], data["packed"], data["scales"])
    cached = partial(uncached, rows=data["rows"])
    return {
        arm: control
        if arm == "onednn"
        else uncached
        if arm == "original_uncached"
        else cached
        for arm in arms
    }


def screen_case(torch, data, case, reference_screen, output_directory, result, arms):
    from scripts.xpu_w4a16_grouped_probe import difference

    outputs = {}
    for name in ("native", "reference"):
        path = reference_screen / f"{case['prefix']}.m{case['m']}.{name}.pt"
        outputs[name] = torch.load(path, weights_only=True)
    result.update({"arms": {}, "reference_hashes": {}})
    for name in outputs:
        path = reference_screen / f"{case['prefix']}.m{case['m']}.{name}.pt"
        result["reference_hashes"][name] = digest(path)

    def require(gates, name, actual, expected, tolerance):
        gates[name] = difference(actual, expected, tolerance, tolerance)
        if not gates[name]["pass"]:
            raise ValueError(f"{case['prefix']} M={case['m']} {arm}: failed {name}")

    calls = calls_for(torch, data, arms)
    for arm in arms:
        set_policy(arm)
        gates = result["arms"][arm] = {}
        if arm == "onednn":
            actual = calls[arm]().cpu()
            require(gates, "captured_exact", actual, data["cpu_operands"]["output"], 0)
            continue
        rows = None if arm == "original_uncached" else data["rows"]
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.XPU,
            ]
        ) as prof:
            actual = native(
                torch, data["x"], data["packed"], data["scales"], rows, True
            )
            torch.xpu.synchronize()
        trace_path = (
            output_directory / f"{case['prefix']}.m{case['m']}.{arm}.trace.json"
        )
        prof.export_chrome_trace(str(trace_path))
        trace = json.loads(trace_path.read_text())
        policy = (
            "MoE::w4a16_policy,"
            if arm.startswith("original")
            else f"MoE::w4a16_dense_policy_{POLICIES[arm]},"
        )
        dispatch = [
            e["name"]
            for e in trace["traceEvents"]
            if e.get("cat") == "kernel" and "MoE::GemmCuteName" in e["name"]
        ]
        gates["dispatch"] = {
            "expected_policy": policy,
            "kernels": dispatch,
            "trace": trace_path.name,
            "sha256": digest(trace_path),
        }
        if len(dispatch) != 1 or policy not in dispatch[0]:
            raise ValueError(f"{arm}: trace does not prove requested native policy")
        actual_cpu = actual.cpu()
        require(gates, "fp32_reference", actual_cpu, outputs["reference"], 0.01)
        require(
            gates,
            "captured_compatibility",
            actual_cpu,
            data["cpu_operands"]["output"],
            0.015625,
        )
        require(
            gates,
            "frozen_native",
            actual_cpu,
            outputs["native"],
            0 if arm.startswith("original") else 0.015625,
        )
        sample_indices = [0, 1, case["m"] // 2, case["m"] - 1]
        sample = data["x"][sample_indices].contiguous()
        independent = torch.cat(
            [
                native(
                    torch,
                    sample[i : i + 1],
                    data["packed"],
                    data["scales"],
                    canaries=True,
                )
                for i in range(4)
            ]
        )
        require(
            gates, "prefill_rows_vs_m1", actual[sample_indices], independent, 0.015625
        )
        grouped = native(torch, sample, data["packed"], data["scales"], canaries=True)
        require(gates, "m4_vs_m1", grouped, independent, 0.015625)
        for repeat in range(2):
            decoy = native(
                torch,
                sample.neg(),
                data["packed"].roll(1, 1),
                data["scales"] * 1.5,
                canaries=True,
            )
            del decoy
            again = native(torch, data["x"], data["packed"], data["scales"], rows, True)
            require(gates, f"repeat_{repeat}", again.cpu(), actual_cpu, 0)
            del again
        gates["canaries"] = True
        del actual, sample, independent, grouped
    result["noninterference"] = all(
        torch.equal(data[gpu].cpu(), data[cpu])
        for gpu, cpu in (("packed", "cpu_packed"), ("scales", "cpu_scales"))
    ) and torch.equal(data["x"].cpu(), data["cpu_operands"]["x"])
    if not result["noninterference"]:
        raise ValueError("native changed its inputs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--reference-screen", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--baseline-info", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(ARMS))
    parser.add_argument("--m", type=int, choices=(1535, 2047, 2048), nargs="+")
    args = parser.parse_args()
    if args.blocks < 30:
        parser.error("at least 30 complete balanced blocks are required")
    if "onednn" not in args.arms or len(set(args.arms)) != len(args.arms):
        parser.error("arms must be unique and include onednn")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    library = args.library.resolve(strict=True)
    manifest_path = args.capture / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    old_screen_path = args.reference_screen / "report.json"
    old_screen = json.loads(old_screen_path.read_text())
    if (
        manifest["status"] != "captured"
        or old_screen["status"] != "initial_screen_passed_requires_remaining_gates"
        or old_screen["capture_manifest_sha256"] != digest(manifest_path)
        or old_screen["kernel_sha256"] != manifest["kernel_sha256"]
    ):
        raise ValueError("requires passing frozen screen for these captured operands")
    report = {
        "argv": sys.argv,
        "source_sha256": digest(__file__),
        "capture_manifest_sha256": digest(manifest_path),
        "reference_screen_sha256": digest(old_screen_path),
        "baseline_info": json.loads(args.baseline_info.read_text()),
        "baseline_info_sha256": digest(args.baseline_info),
        "library_requested": str(args.library),
        "library_resolved": str(library),
        "library_sha256": digest(library),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if (
                k.startswith(("VLLM_", "SYCL_", "ZE_", "ONEAPI_", "ONEDNN_", "DNNL_"))
                or k in ("LD_LIBRARY_PATH", "OCL_ICD_VENDORS", "PYTHONPATH")
            )
            and not any(
                word in k
                for word in ("API_KEY", "AUTH", "PASSWORD", "SECRET", "CREDENTIAL")
            )
            and not k.endswith("_TOKEN")
        },
        "arms": {
            arm: {
                "policy": POLICIES.get(arm),
                "cached_rows": arm not in ("onednn", "original_uncached"),
            }
            for arm in args.arms
        },
        "warmup_rule": {
            "minimum_seconds": 1,
            "maximum_seconds": 10,
            "window_minimum_seconds": 0.1,
            "consecutive_windows": 3,
            "per_arm_median_relative_spread": 0.01,
        },
        "cases": [],
        "status": "starting",
        "service_qualified": False,
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    previous_policy = os.environ.pop(POLICY_ENV, None)
    try:
        import torch

        if torch.__version__ != manifest["torch"]:
            raise ValueError("candidate must use the frozen capture's Torch version")
        # LD_LIBRARY_PATH must be supplied before Python starts. The unchanged
        # frozen torch binding loads its grouped-GEMM dependency by SONAME.
        import vllm_xpu_kernels._xpu_C as kernels

        if digest(kernels.__file__) != manifest["kernel_sha256"]:
            raise ValueError("torch binding DSO differs from frozen capture")
        mappings = mapped_libraries()
        if str(library) not in mappings:
            raise ValueError("requested candidate DSO not found in process maps")
        grouped = [p for p in mappings if Path(p).name == "libgrouped_gemm_xe_2.so"]
        if grouped != [str(library)]:
            raise ValueError(f"unexpected grouped GEMM DSOs: {grouped}")
        report["runtime"] = {
            "torch": torch.__version__,
            "torch_file": torch.__file__,
            "torch_binding": kernels.__file__,
            "torch_binding_sha256": digest(kernels.__file__),
            "device": str(torch.xpu.get_device_properties(0)),
            "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
            "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "vllm_package_version": importlib.metadata.version("vllm"),
            "mapped_libraries": mappings,
            "loaded_candidate_sha256": digest(library),
        }
        cases = sorted(
            [c for c in manifest["captures"] if args.m is None or c["m"] in args.m],
            key=lambda c: (c["prefix"], c["m"]),
            reverse=args.reverse,
        )
        report["status"] = "screening"
        save()
        with torch.inference_mode():
            for case in cases:
                row = {"prefix": case["prefix"], "m": case["m"]}
                report["cases"].append(row)
                data = prepare(torch, args.capture, manifest, case)
                row["screen"] = {}
                screen_case(
                    torch,
                    data,
                    case,
                    args.reference_screen,
                    args.output,
                    row["screen"],
                    args.arms,
                )
                del data
                save()
            report["status"] = "screen_passed"
            save()
            if args.screen_only:
                return 0
            for case, row in zip(cases, report["cases"], strict=True):
                data = prepare(torch, args.capture, manifest, case)
                calls = calls_for(torch, data, args.arms)
                # No references, disk I/O or progress printing between warmup
                # and the end of measurement for a case.
                row["warmup"] = warm(torch, calls, args.reverse)
                row["blocks"] = [
                    run_block(torch, calls, index, args.reverse)
                    for index in range(args.blocks)
                ]
                row["summary"] = summarize_blocks(row["blocks"], args.arms)
                row["stable"] = row["warmup"]["stable"] and not any(
                    s["drift_flag"] for s in row["summary"].values()
                )
                del calls, data
                print(
                    json.dumps(
                        {
                            "prefix": row["prefix"],
                            "m": row["m"],
                            "stable": row["stable"],
                            "summary": row["summary"],
                        }
                    ),
                    flush=True,
                )
                save()
        report["status"] = (
            "operator_comparison_complete"
            if all(c["stable"] for c in report["cases"])
            else "operator_comparison_unstable"
        )
        return 0
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = repr(error)
        raise
    finally:
        if previous_policy is None:
            os.environ.pop(POLICY_ENV, None)
        else:
            os.environ[POLICY_ENV] = previous_policy
        save()


if __name__ == "__main__":
    raise SystemExit(main())
