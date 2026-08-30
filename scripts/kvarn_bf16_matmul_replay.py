#!/usr/bin/env python3
"""Check repeatability of the BF16 matmuls used by the Kvarn W4 control."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class IterationResult:
    iteration: int
    input_data_ptr: int
    weight_data_ptr: int
    output_data_ptr: int
    mismatch_count: int
    max_abs: float


def parse_shape(value: str) -> tuple[int, int, int]:
    try:
        m, k, n = (int(part) for part in value.split("x"))
    except ValueError as error:
        raise argparse.ArgumentTypeError("shape must be MxKxN") from error
    if min(m, k, n) <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return m, k, n


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mkldnn-deterministic", action="store_true")
    args = parser.parse_args()

    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.mkldnn.deterministic = args.mkldnn_deterministic
    torch.manual_seed(args.seed)
    device = torch.device("xpu")
    dtype = getattr(torch, args.dtype)
    report: dict[str, object] = {
        "device": torch.xpu.get_device_name(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        "dtype": args.dtype,
        "iterations": args.iterations,
        "seed": args.seed,
        "shapes": {},
    }

    for m, k, n in args.shape:
        base_input = torch.randn((m, k), dtype=dtype, device=device)
        base_weight = torch.randn((k, n), dtype=dtype, device=device)
        torch.xpu.synchronize()

        reference: torch.Tensor | None = None
        results: list[IterationResult] = []
        for iteration in range(args.iterations):
            # Keep a differently sized allocation alive so successive clones do
            # not all receive the same cached-allocation addresses.
            guard = torch.empty(
                (iteration * 2 * 1024 * 1024,), dtype=torch.uint8, device=device
            )
            input_value = base_input.clone()
            weight_value = base_weight.clone()
            output = torch.matmul(input_value, weight_value)
            torch.xpu.synchronize()

            if reference is None:
                reference = output.clone()
                mismatch_count = 0
                max_abs = 0.0
            else:
                mismatch_count = int(torch.count_nonzero(output != reference).item())
                max_abs = float(
                    torch.max(torch.abs(output.float() - reference.float())).item()
                )
            results.append(
                IterationResult(
                    iteration=iteration,
                    input_data_ptr=input_value.data_ptr(),
                    weight_data_ptr=weight_value.data_ptr(),
                    output_data_ptr=output.data_ptr(),
                    mismatch_count=mismatch_count,
                    max_abs=max_abs,
                )
            )
            del guard, input_value, weight_value, output

        assert reference is not None
        report["shapes"][f"{m}x{k}x{n}"] = {
            "exact": all(result.mismatch_count == 0 for result in results),
            "results": [asdict(result) for result in results],
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
