"""Gate E=1 native INT4 GEMM against real, captured oneDNN W4A16 operands.

This is an offline experiment, with no serving dispatch or kernel-policy changes.
The format adapter rejects unsupported operands rather than requantizing them.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

from scripts.xpu_w4a16_capture import describe, digest


def validate_plan(plan):
    """Keep the frozen numerical profile; results never select tolerances."""
    for name, tolerance in (
        ("native_reference", 0.01),
        ("onednn_compatibility", 0.015625),
        ("batch_separability", 0.015625),
    ):
        if any(plan["gates"][name][key] != tolerance for key in ("atol", "rtol")):
            raise ValueError(f"changed frozen numerical threshold: {name}")
    if plan["M"] != [2048, 1535, 2047] or len(set(plan["projections"])) != 2:
        raise ValueError("requires the two-projection prefill/tail screen")


def validate_format(q_nt, scales, zero, group_size, g_idx):
    if q_nt.dtype != torch.int32 or q_nt.ndim != 2:
        raise ValueError("requires int32 NT [K/8,N] weights")
    k, n = q_nt.shape[0] * 8, q_nt.shape[1]
    if k == 0 or n == 0 or k % 128 or n % 8 or q_nt.stride() != (1, k // 8):
        raise ValueError("requires K divisible by 128, N by 8 and NT strides")
    if group_size != 128 or g_idx is not None:
        raise ValueError("requires G128 without an activation-order permutation")
    if (
        zero is None
        or zero.dtype != torch.int8
        or zero.shape != (1,)
        or zero.item() != 8
    ):
        raise ValueError("requires verified scalar int8 zero point 8")
    if (
        scales.dtype != torch.bfloat16
        or scales.shape != (k // 128, n)
        or not scales.is_contiguous()
    ):
        raise ValueError("requires contiguous BF16 [K/128,N] scales")
    if not torch.isfinite(scales).all():
        raise ValueError("requires finite scales")
    if not q_nt.device == scales.device == zero.device:
        raise ValueError("weight operands must share a device")
    return k, n


def adapt(q_nt, scales, zero, group_size, g_idx):
    k, n = validate_format(q_nt, scales, zero, group_size, g_idx)
    # Transposition exposes the original physical bytes, low nibble first.
    # XOR flips each nibble's sign bit: signed(u XOR 8) == u - 8.
    packed = q_nt.t().view(torch.uint8).bitwise_xor(0x88)
    return (
        packed.view(torch.int8).reshape(1, n, k // 2),
        scales.t().contiguous().unsqueeze(0),
    )


def unpack_original(q_nt, scales, start, stop):
    words = q_nt[:, start:stop].t().to(torch.int64)
    shift = torch.arange(8, device=words.device) * 4
    integers = ((words[..., None] >> shift) & 15).reshape(stop - start, -1) - 8
    values = (
        integers.float() * scales[:, start:stop].t().repeat_interleave(128, 1).float()
    )
    return integers, values


def verify_adapter(q_nt, scales, packed, native_scales):
    """Exhaustively compare independent word and signed-byte unpacking on CPU."""
    k, n = q_nt.shape[0] * 8, q_nt.shape[1]
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        original, values = unpack_original(q_nt, scales, start, stop)
        raw = packed[0, start:stop].to(torch.int16)
        nibble = torch.stack((raw & 15, (raw >> 4) & 15), dim=-1).reshape(
            stop - start, k
        )
        signed = (nibble ^ 8) - 8
        if not torch.equal(original, signed):
            raise ValueError("adapted integer weights differ")
        reconstructed = (
            signed.float()
            * native_scales[0, start:stop].repeat_interleave(128, 1).float()
        )
        if not torch.equal(values, reconstructed):
            raise ValueError("adapted dequantized weights differ")
    return {
        "logical_weights_checked": k * n,
        "integer_exact": True,
        "dequantized_exact": True,
    }


def difference(actual, expected, atol, rtol):
    actual, expected = actual.float(), expected.float()
    delta = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    failures = int((delta > atol + rtol * expected.abs()).sum())
    return {
        "pass": finite and failures == 0,
        "finite": finite,
        "failed_elements": failures,
        "elements": actual.numel(),
        "exact_mismatches": int((actual != expected).sum()),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "atol": atol,
        "rtol": rtol,
    }


def native(x, packed, scales, bias=None, *, canaries=False):
    if x.dtype != torch.bfloat16 or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("requires contiguous BF16 [M,K] activations")
    m, k = x.shape
    n = packed.shape[1]
    if bias is not None and (
        bias.dtype != x.dtype or bias.shape != (n,) or not bias.is_contiguous()
    ):
        raise ValueError("requires contiguous BF16 [N] bias")
    # Include metadata, output allocation and the operator's own scratch in
    # the callable; a future timing stage must measure this whole adapter.
    rows = torch.tensor([m], device=x.device, dtype=torch.int32)
    storage = torch.empty((m + 2 if canaries else m, n), device=x.device, dtype=x.dtype)
    output = storage[1:-1] if canaries else storage
    if canaries:
        storage.fill_(-42)
        output.fill_(float("nan"))
    result = torch.ops._xpu_C.cutlass_grouped_gemm_interface(
        x,
        None,
        packed,
        scales,
        None if bias is None else bias.unsqueeze(0),
        output,
        rows,
        n,
        k,
        1,
    )
    if result.data_ptr() != output.data_ptr():
        raise ValueError("E=1 operator did not return the caller-owned output")
    if canaries:
        torch.xpu.synchronize()
        if not torch.all(storage[[0, -1]] == -42) or not torch.isfinite(output).all():
            raise ValueError("output ownership/canary gate failed")
    return output


def reference(x, q_nt, scales, bias):
    n = q_nt.shape[1]
    output = torch.empty((x.shape[0], n), dtype=torch.bfloat16)
    # Match the frozen grouped INT4 test: original weights dequantized to BF16,
    # FP32 accumulation, BF16 output. Never derive the oracle from adapted B.
    a = x.float()
    for start in range(0, n, 256):
        stop = min(start + 256, n)
        _, values = unpack_original(q_nt, scales, start, stop)
        b = values.to(torch.bfloat16).to(x.device).float()
        result = a @ b.t()
        if bias is not None:
            result += bias[start:stop].float()
        output[:, start:stop] = result.to(torch.bfloat16).cpu()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "probe-source.py")
    shutil.copy2(args.plan, args.output / "plan.json")
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    manifest = json.loads((args.capture / "manifest.json").read_text())
    if (
        manifest.get("status") != "captured"
        or manifest["revision"] != plan["checkpoint_revision"]
        or manifest["torch"] != torch.__version__
    ):
        raise ValueError(
            "requires completed capture with matching checkpoint and Torch"
        )
    import vllm_xpu_kernels._xpu_C as kernels

    if digest(kernels.__file__) != manifest["kernel_sha256"]:
        raise ValueError("capture and replay kernel binaries differ")
    report = {
        "argv": sys.argv,
        "plan_sha256": digest(args.plan),
        "capture_manifest_sha256": digest(args.capture / "manifest.json"),
        "kernel_binary": kernels.__file__,
        "kernel_sha256": digest(kernels.__file__),
        "device": str(torch.xpu.get_device_properties(0)),
        "schema": str(torch.ops._xpu_C.cutlass_grouped_gemm_interface.default._schema),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "formats": [],
        "cases": [],
        "status": "running",
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for prefix in plan["projections"]:
            path = args.capture / f"{prefix}.weights.pt"
            if digest(path) != manifest["weights_sha256"][path.name]:
                raise ValueError("captured weight hash changed")
            weights = torch.load(path, weights_only=True)
            start = time.perf_counter()
            packed_cpu, scales_cpu = adapt(**weights)
            prep = time.perf_counter() - start
            exact = verify_adapter(
                weights["q_nt"], weights["scales"], packed_cpu, scales_cpu
            )
            torch.xpu.reset_peak_memory_stats()
            before = torch.xpu.memory_allocated()
            start = time.perf_counter()
            packed = packed_cpu.to("xpu")
            scales = scales_cpu.to("xpu")
            torch.xpu.synchronize()
            report["formats"].append(
                {
                    "prefix": prefix,
                    **exact,
                    "cpu_prepack_seconds": prep,
                    "upload_seconds": time.perf_counter() - start,
                    "retained_xpu_bytes": torch.xpu.memory_allocated() - before,
                    "peak_xpu_bytes": torch.xpu.max_memory_allocated(),
                    "packed": describe(packed),
                    "scales": describe(scales),
                }
            )
            save()
            for m in plan["M"]:
                path = args.capture / f"{prefix}.m{m}.pt"
                meta = next(c for c in manifest["captures"] if c["file"] == path.name)
                if digest(path) != meta["sha256"]:
                    raise ValueError("captured activation hash changed")
                operands = torch.load(path, weights_only=True)
                x = operands["x"].to("xpu")
                bias = None if operands["bias"] is None else operands["bias"].to("xpu")
                row = {"prefix": prefix, "m": m, "bias": describe(bias)}
                report["cases"].append(row)
                # Separate profiling proves dispatch; this call has no timing claim.
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.XPU,
                    ]
                ) as prof:
                    actual = native(x, packed, scales, bias, canaries=True)
                    torch.xpu.synchronize()
                trace_path = args.output / f"{prefix}.m{m}.trace.json"
                prof.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text())
                row["native_dispatch"] = [
                    e["name"]
                    for e in trace["traceEvents"]
                    if e.get("cat") == "kernel"
                    and "MoE::GemmCuteName" in e["name"]
                    and "MoE::w4a16_policy," in e["name"]
                ]
                if len(row["native_dispatch"]) != 1:
                    raise ValueError("trace did not prove native prefill INT4 dispatch")
                actual_cpu = actual.cpu()
                torch.save(actual_cpu, args.output / f"{prefix}.m{m}.native.pt")
                row["ownership"] = "pass"
                row["onednn_compatibility"] = difference(
                    actual_cpu,
                    operands["output"],
                    **{
                        k: plan["gates"]["onednn_compatibility"][k]
                        for k in ("atol", "rtol")
                    },
                )
                expected = reference(x, weights["q_nt"], weights["scales"], bias)
                torch.save(expected, args.output / f"{prefix}.m{m}.reference.pt")
                row["native_reference"] = difference(
                    actual_cpu,
                    expected,
                    **{
                        k: plan["gates"]["native_reference"][k]
                        for k in ("atol", "rtol")
                    },
                )
                save()
                print(json.dumps(row), flush=True)
                if (
                    not row["native_reference"]["pass"]
                    or not row["onednn_compatibility"]["pass"]
                ):
                    report["status"] = "rejected_numerical"
                    return 2
                for repeat in range(4):
                    decoy = native(x.neg(), packed, scales, bias)
                    pressure = torch.full(
                        (1048573 * (repeat + 1),),
                        repeat,
                        dtype=torch.uint8,
                        device="xpu",
                    )
                    repeated = native(x, packed, scales, bias, canaries=True)
                    torch.xpu.synchronize()
                    if not torch.equal(repeated.cpu(), actual_cpu):
                        row["repeatability"] = "fail"
                        report["status"] = "rejected_repeatability"
                        return 2
                    del decoy, pressure, repeated
                row["repeatability"] = "pass"
                save()
                del x, actual
            del packed, scales
        report["status"] = "initial_screen_passed_requires_remaining_gates"
        return 0
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        save()


if __name__ == "__main__":
    with torch.inference_mode():
        raise SystemExit(main())
