"""Locate recorded Python callers of long CPU runtime/driver events.

CPU containment uses the same process and thread only. This does not assign
GPU execution by CPU timestamps or infer a hardware bottleneck from a wait.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from scripts.kvarn_factory_run import (
    ensure_durable_output,
    sha256_file,
    write_json_atomic,
)
from scripts.kvarn_xpu_profile import load_trace
from scripts.kvarn_xpu_trace import categories, interval


def analyze_stacks(document: dict, *, event_name: str, limit: int = 10) -> dict:
    if limit < 1:
        raise ValueError("event limit must be positive")
    events = [
        event
        for event in document["traceEvents"]
        if event.get("ph") == "X"
        and all(
            isinstance(event.get(key), (int, float)) and math.isfinite(event[key])
            for key in ("ts", "dur")
        )
        and event["dur"] > 0
    ]
    targets = sorted(
        (
            event
            for event in events
            if event.get("name") == event_name
            and categories(event) & {"xpu_runtime", "xpu_driver", "cpu_op"}
        ),
        key=lambda event: event["dur"],
        reverse=True,
    )
    if not targets:
        raise ValueError(f"trace has no matching CPU event: {event_name}")
    rows = []
    for target in targets[:limit]:
        start, end = interval(target)
        enclosing = [
            event
            for event in events
            if event.get("pid") == target.get("pid")
            and event.get("tid") == target.get("tid")
            and categories(event) & {"python_function", "cpu_op", "user_annotation"}
            and event["ts"] <= start
            and interval(event)[1] >= end
        ]
        enclosing.sort(key=lambda event: (event["ts"], -event["dur"]))
        frames = [
            event["name"]
            for event in enclosing
            if "python_function" in categories(event)
        ]
        operators = [
            {
                "name": event["name"],
                "duration_us": event["dur"],
                "input_shapes": event.get("args", {}).get("Input Dims"),
                "input_types": event.get("args", {}).get("Input type"),
            }
            for event in enclosing
            if "cpu_op" in categories(event)
        ]
        rows.append(
            {
                "event": target["name"],
                "category": target["cat"],
                "pid": target.get("pid"),
                "tid": target.get("tid"),
                "timestamp_us": target["ts"],
                "cpu_inclusive_us": target["dur"],
                "event_arguments": target.get("args", {}),
                "python_frames_outer_to_inner": frames,
                "cpu_operators_outer_to_inner": operators,
                "annotations_outer_to_inner": [
                    event["name"]
                    for event in enclosing
                    if "user_annotation" in categories(event)
                ],
                "source_stack_present": bool(frames),
            }
        )
    return {
        "schema_version": 1,
        "artifact_kind": "xpu_cpu_source_stack_diagnostic",
        "diagnostic_only": True,
        "acceptance_eligible": False,
        "selected_event_name": event_name,
        "matching_event_count": len(targets),
        "reported_event_count": len(rows),
        "events": rows,
        "limitations": [
            "Recorded Python frames are enclosing calls, not sampled CPU instruction stacks.",
            "Frame line numbers identify recorded Python functions, not necessarily the exact call expression.",
            "No Python frames means source attribution is unavailable, not that Python caused no cost.",
            "Inclusive CPU waits can overlap GPU work and nested CPU events; do not sum as independent overhead.",
            "This source report makes no GPU ownership or hardware bottleneck inference.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--event", default="zeEventHostSynchronize")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    output = ensure_durable_output(args.output, allow_tmp=False)
    paths = [
        args.trace.resolve(strict=True),
        Path(__file__).resolve(),
        Path(__file__).with_name("kvarn_xpu_trace.py").resolve(),
        Path(__file__).with_name("kvarn_xpu_profile.py").resolve(),
    ]
    hashes = {str(path): sha256_file(path) for path in paths}
    report = analyze_stacks(
        load_trace(args.trace), event_name=args.event, limit=args.limit
    )
    if hashes != {str(path): sha256_file(path) for path in paths}:
        raise ValueError("input changed during analysis")
    report["source_hashes"] = hashes
    write_json_atomic(output, report)
    for row in report["events"]:
        print(f"{row['event']}: {row['cpu_inclusive_us'] / 1000:.3f} ms")
        for frame in row["python_frames_outer_to_inner"][-8:]:
            print(f"  {frame}")


if __name__ == "__main__":
    main()
