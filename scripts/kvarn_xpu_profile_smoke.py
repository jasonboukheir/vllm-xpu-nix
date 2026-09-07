"""Fail-fast CPU/XPU profiler preflight with a known asynchronous workload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.kvarn_factory_run import (
    ensure_durable_output,
    preflight_xpu,
    write_json_atomic,
)
from scripts.kvarn_xpu_trace import analyze_attribution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()
    output = ensure_durable_output(args.output, allow_tmp=False)
    report_path = ensure_durable_output(
        output.with_suffix(".summary.json"), allow_tmp=False
    )
    import torch

    hardware = preflight_xpu(torch)
    host = torch.randn((512, 512))
    x = host.xpu()
    for _ in range(4):
        result = torch.relu(x @ x)
    torch.xpu.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.XPU,
        ],
        record_shapes=True,
        with_stack=args.with_stack,
        profile_memory=False,
    ) as prof:
        for step in range(2):
            with torch.profiler.record_function(f"known_step_{step}"):
                with torch.profiler.record_function("known_h2d_copy"):
                    x.copy_(host, non_blocking=True)
                with torch.profiler.record_function("known_compute"):
                    for _ in range(4):
                        result = torch.relu(x @ x)
        with torch.profiler.record_function("known_sync"):
            torch.xpu.synchronize()
    assert result.device.type == "xpu"
    prof.export_chrome_trace(str(output))
    document = json.loads(output.read_text())
    analysis = analyze_attribution(document, step_prefix="known_step_")
    python_events = [
        event
        for event in document["traceEvents"]
        if "python_function" in str(event.get("cat", "")).split(",")
    ]
    if args.with_stack and not python_events:
        raise RuntimeError(
            "stack capture requested but trace contains no Python frames"
        )
    if (
        analysis["coverage"]["assigned_fraction"] != 1
        or any(
            step["kernel_count"] != 8 or step["copy_count"] != 1
            for step in analysis["steps"]
        )
        or len(analysis["steps"]) != 2
    ):
        raise RuntimeError(
            "profiler did not capture/correlate the expected kernels and copies"
        )
    write_json_atomic(
        report_path,
        {
            "passed": True,
            "hardware": hardware,
            "analysis": analysis,
            "with_stack": args.with_stack,
            "python_frame_count": len(python_events),
            "python_frame_examples": list(
                dict.fromkeys(str(event.get("name")) for event in python_events)
            )[:20],
        },
    )
    print(f"Profiler preflight passed: {report_path}")


if __name__ == "__main__":
    main()
