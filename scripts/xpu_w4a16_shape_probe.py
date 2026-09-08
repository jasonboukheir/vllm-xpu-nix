#!/usr/bin/env python3
"""Measure first-seen versus reused oneDNN W4A16 token-count shapes.

Run with the serving package's Python environment and optionally
ONEDNN_VERBOSE=profile_create to attribute primitive creation separately.
Uses synthetic tensors and does not load or contact a model server.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import vllm_xpu_kernels._xpu_C  # noqa: F401

# Qwen3.5 27B projection dimensions, also exercised by the kernels repository's
# tests/test_int4_gemm_determinism.py. Order is K, N.
PROJECTIONS = {
    "attention": (5120, 8192),
    "gdn_qkvz": (5120, 16384),
    "gdn_ba": (5120, 96),
    "gdn_out": (6144, 5120),
    "mlp_gate_up": (5120, 34816),
    "mlp_down": (17408, 5120),
}


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 2048, 107, 108, 109, 511, 512, 513, 2047, 107],
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--projection", choices=list(PROJECTIONS), action="append")
    args = parser.parse_args()
    if min(args.lengths) <= 0 or args.repeats < 2:
        parser.error("positive lengths and at least two repeats are required")

    torch.xpu.set_device(args.device)
    torch.manual_seed(20260908)
    device = torch.device("xpu", args.device)
    dtype = torch.bfloat16
    # Initialize allocation and submission before measuring any matmul.
    torch.zeros(1, device=device).add_(1)
    torch.xpu.synchronize()
    report = {
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(args.device),
        "op": "torch.ops._xpu_C.int4_gemm_w4a16",
        "dtype": str(dtype),
        "group_size": 128,
        "lengths": args.lengths,
        "repeats": args.repeats,
        "rows": [],
    }
    for name in args.projection or PROJECTIONS:
        k, n = PROJECTIONS[name]
        # Match the production packed NT layout, BF16 scales and scalar zero
        # point. Allocate and initialize outside the timed region.
        weight = (
            torch.randint(-128, 128, (n, k // 2), dtype=torch.int8, device=device)
            .view(torch.int32)
            .t()
        )
        scales = torch.full((k // 128, n), 0.01, dtype=dtype, device=device)
        zero = torch.tensor([8], dtype=torch.int8, device=device)
        inputs = torch.randn((max(args.lengths), k), dtype=dtype, device=device)
        seen = set()
        for m in args.lengths:
            x = inputs[:m]
            torch.xpu.synchronize()
            row = {
                "projection": name,
                "m": m,
                "k": k,
                "n": n,
                "seen_before": m in seen,
                "host_ms": [],
                "wall_ms": [],
            }
            print(
                "SHAPE "
                + json.dumps(
                    {k: v for k, v in row.items() if k not in ("host_ms", "wall_ms")}
                ),
                flush=True,
            )
            for _ in range(args.repeats):
                start = time.perf_counter()
                output = torch.ops._xpu_C.int4_gemm_w4a16(
                    x, weight, None, scales, zero, 128, None
                )
                submitted = time.perf_counter()
                torch.xpu.synchronize()
                end = time.perf_counter()
                row["host_ms"].append((submitted - start) * 1000)
                row["wall_ms"].append((end - start) * 1000)
                del output
            row["repeat_median_ms"] = statistics.median(row["wall_ms"][1:])
            row["first_minus_repeat_ms"] = row["wall_ms"][0] - row["repeat_median_ms"]
            report["rows"].append(row)
            seen.add(m)
            print("RESULT " + json.dumps(row), flush=True)
        del weight, scales, zero, inputs, x
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
