"""Run one serial off/on/off unitrace counter experiment and preserve evidence."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from scripts.kvarn_factory_run import (
    ensure_durable_output,
    sha256_file,
    write_json_atomic,
)
from scripts.kvarn_xpu_profile_sources import snapshot_sources, verify_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unitrace", type=Path, required=True)
    parser.add_argument("--workload", choices=("gemm", "sinkhorn"), required=True)
    parser.add_argument("--group", default="ComputeBasic")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--sampling-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if min(args.iterations, args.sampling_interval) < 1:
        parser.error("iterations and sampling interval must be positive")
    root = Path(__file__).resolve().parents[1]
    output = ensure_durable_output(
        args.output_dir / "capture.json", allow_tmp=False
    ).parent
    output.mkdir(parents=True, exist_ok=False)
    unitrace = args.unitrace.resolve(strict=True)
    environment = dict(os.environ, ZET_ENABLE_METRICS="1")
    injection_library = unitrace.parent.parent / "lib" / "libunitrace_tool.so"
    if not injection_library.is_file():
        raise ValueError("unitrace injection library is missing")
    environment["LD_LIBRARY_PATH"] = (
        str(injection_library.parent) + ":" + environment.get("LD_LIBRARY_PATH", "")
    )
    snapshot_sources(root, output / "source-snapshot")
    record = {
        "schema_version": 1,
        "artifact_kind": "xpu_counter_capture",
        "collection_validated": False,
        "diagnostic_only": True,
        "unitrace": {"path": str(unitrace), "sha256": sha256_file(unitrace)},
        "injection_library": {
            "path": str(injection_library),
            "sha256": sha256_file(injection_library),
        },
        "python": sys.executable,
        "environment": {
            k: v
            for k, v in environment.items()
            if k.startswith(
                ("ZE", "ZET", "SYCL", "ONEAPI", "PTI", "UNITRACE", "LD_LIBRARY_PATH")
            )
        },
        "observation_paranoid": (
            Path("/proc/sys/dev/xe/observation_paranoid").read_text().strip()
            if Path("/proc/sys/dev/xe/observation_paranoid").is_file()
            else None
        ),
        "group": args.group,
        "sampling_interval_us_requested": args.sampling_interval,
        "commands": [],
        "limitations": [
            "Capture success is not sample validation; inspect numeric metric rows independently.",
            "One bracket is descriptive, not a statistical performance qualification.",
        ],
    }
    write_json_atomic(output / "capture.json", record)

    def run(label, command):
        command = [str(item) for item in command]
        entry = {
            "label": label,
            "argv": command,
            "shell": shlex.join(command),
            "cwd": str(root),
        }
        record["commands"].append(entry)
        write_json_atomic(output / "capture.json", record)
        start = time.monotonic()
        with (
            (output / f"{label}.stdout").open("xb") as stdout,
            (output / f"{label}.stderr").open("xb") as stderr,
        ):
            result = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                timeout=600,
                check=False,
            )
        entry.update(
            returncode=result.returncode, process_elapsed_s=time.monotonic() - start
        )
        write_json_atomic(output / "capture.json", record)
        if result.returncode:
            raise RuntimeError(f"{label} failed: inspect {output}")

    run("version", [unitrace, "--version"])
    run("metric-list", [unitrace, "--metric-list"])
    for label in ("before", "profiled", "after"):
        workload = [
            sys.executable,
            "-m",
            "scripts.kvarn_xpu_counter_workload",
            "--workload",
            args.workload,
            "--iterations",
            str(args.iterations),
            "--output",
            str(output / f"{label}.json"),
        ]
        command = workload
        if label == "profiled":
            command = [
                unitrace,
                "--start-paused",
                "--metric-sampling",
                "--group",
                args.group,
                "--sampling-interval",
                str(args.sampling_interval),
                "--chrome-kernel-logging",
                "--result-dir",
                str(output / "collector"),
                *workload,
            ]
        run(label, command)
    values = [
        json.loads((output / f"{label}.json").read_text())
        for label in ("before", "profiled", "after")
    ]
    for key in (
        "workload",
        "contract",
        "iterations",
        "warmup",
        "seed",
        "hardware",
        "sinkhorn_module",
        "workload_source_sha256",
        "torch_version",
    ):
        if any(value[key] != values[0][key] for value in values):
            raise ValueError(f"bracket identity mismatch: {key}")
    if not all(value["correctness_passed"] for value in values):
        raise ValueError("workload correctness failed")
    before, profiled, after = [value["window_elapsed_ms"] for value in values]
    baseline = (before + after) / 2
    record["overhead"] = {
        "before_ms": before,
        "profiled_ms": profiled,
        "after_ms": after,
        "baseline_ms": baseline,
        "profiled_over_baseline": profiled / baseline,
        "after_over_before": after / before,
    }
    verify_sources(output / "source-snapshot", source_root=root)
    if sha256_file(unitrace) != record["unitrace"]["sha256"]:
        raise ValueError("unitrace binary changed")
    record["files"] = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "capture.json"
    }
    record["capture_completed"] = True
    write_json_atomic(output / "capture.json", record)
    print(output / "capture.json")


if __name__ == "__main__":
    main()
