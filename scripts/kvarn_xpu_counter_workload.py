"""Fixed, warmed GPU regions for unitrace counter qualification, not tuning."""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

from scripts.kvarn_factory_run import (
    ensure_durable_output,
    preflight_xpu,
    sha256_file,
    write_json_atomic,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("gemm", "sinkhorn"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if min(args.iterations, args.warmup, args.pages) < 1:
        parser.error("iterations, warmup, and pages must be positive")
    return args


def main():
    args = parse_args()
    output = ensure_durable_output(args.output, allow_tmp=False)
    if output.exists():
        raise ValueError("output exists; choose a fresh path")
    source_hash = sha256_file(Path(__file__))
    # unitrace --start-paused honors this documented in-application control.
    # Import, initialization, warmup, and correctness checks stay outside it.
    os.environ["PTI_ENABLE_COLLECTION"] = "0"
    import torch

    hardware = preflight_xpu(torch)
    module_record = None
    if args.workload == "gemm":
        size = 2048
        left = torch.ones((size, size), device="xpu", dtype=torch.bfloat16)
        right = torch.ones_like(left)
        result = torch.empty_like(left)

        def operation():
            torch.mm(left, right, out=result)
            return result

        def validate(value):
            torch.testing.assert_close(
                value, torch.full_like(value, size), rtol=0, atol=0
            )

        contract = {
            "shape": [size, size, size],
            "dtype": "bfloat16",
            "expected_value": size,
        }
    else:
        module = importlib.import_module("vllm.v1.attention.ops.triton_kvarn_sinkhorn")
        module_path = Path(module.__file__).resolve()
        if not str(module_path).startswith("/nix/store/"):
            raise ValueError(
                "counter qualification requires an immutable packaged Sinkhorn module"
            )
        module_record = {"path": str(module_path), "sha256": sha256_file(module_path)}
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        shape = (args.pages, 128, 4, 256)
        key = (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .half()
            .to("xpu")
        )
        value = (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .half()
            .to("xpu")
        )
        slots = torch.arange(args.pages, device="xpu", dtype=torch.int64)

        def operation():
            return module.kvarn_sinkhorn_fused_pool_kv_triton(
                key, value, slots, iterations=8
            )

        def validate(outputs):
            tiles = module._materialize_sinkhorn_pool_kv(key, value, slots)
            for original, (balanced, col, row) in zip(
                tiles, (outputs[:3], outputs[3:])
            ):
                assert bool(torch.isfinite(balanced).all())
                assert bool((col > 0).all()) and bool((row > 0).all())
                torch.testing.assert_close(
                    balanced * col[:, None, :] * row[:, :, None],
                    original,
                    rtol=2e-5,
                    atol=2e-5,
                )

        contract = {
            "pool_shape": list(shape),
            "dtype": "float16",
            "sinkhorn_iterations": 8,
            "key_tile_shape": [args.pages * 4, 256, 128],
            "value_tile_shape": [args.pages * 4, 128, 256],
            "scope": "production fused pool materialization plus K/V Sinkhorn; excludes cache writer and model",
        }
    for _ in range(args.warmup):
        result = operation()
    torch.xpu.synchronize()
    validate(result)
    torch.xpu.synchronize()
    os.environ["PTI_ENABLE_COLLECTION"] = "1"
    start = time.perf_counter_ns()
    try:
        for _ in range(args.iterations):
            result = operation()
        torch.xpu.synchronize()
    finally:
        end = time.perf_counter_ns()
        os.environ["PTI_ENABLE_COLLECTION"] = "0"
    validate(result)
    torch.xpu.synchronize()
    if sha256_file(Path(__file__)) != source_hash:
        raise ValueError("workload source changed during execution")
    if (
        module_record
        and sha256_file(Path(module_record["path"])) != module_record["sha256"]
    ):
        raise ValueError("Sinkhorn module changed during execution")
    write_json_atomic(
        output,
        {
            "artifact_kind": "xpu_counter_workload",
            "schema_version": 1,
            "diagnostic_only": True,
            "collection_validated": False,
            "workload": args.workload,
            "contract": contract,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "seed": args.seed,
            "hardware": hardware,
            "window_elapsed_ms": (end - start) / 1e6,
            "mean_iteration_ms": (end - start) / 1e6 / args.iterations,
            "correctness_passed": True,
            "sinkhorn_module": module_record,
            "workload_source_sha256": source_hash,
            "torch_version": torch.__version__,
            "argv": sys.argv,
            "collection_control": "PTI_ENABLE_COLLECTION with unitrace --start-paused",
            "limitations": [
                "Workload success alone does not validate hardware counter samples.",
                "Host window includes dispatch and final synchronization, not process startup/export.",
            ],
        },
    )
    print(
        f"{args.workload}: {(end - start) / 1e6:.3f} ms; correctness passed; {output}"
    )


if __name__ == "__main__":
    main()
