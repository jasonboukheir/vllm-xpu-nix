"""Correlation-aware CPU/XPU attribution for Kineto Chrome traces.

Durations are diagnostic, not performance acceptance evidence. GPU operations
are assigned to their submitting CPU scope, never to CPU timestamp windows.
Unresolved or ambiguous associations stay explicitly unassigned.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.kvarn_factory_run import (
        ensure_durable_output,
        sha256_file,
        write_json_atomic,
    )
except ModuleNotFoundError:
    from kvarn_factory_run import (
        ensure_durable_output,
        sha256_file,
        write_json_atomic,
    )

HOST_CATEGORIES = {"cpu_op", "user_annotation", "xpu_runtime", "xpu_driver"}
DEVICE_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def categories(event: dict[str, Any]) -> set[str]:
    return {part.strip() for part in str(event.get("cat", "")).split(",")}


def interval(event: dict[str, Any]) -> tuple[float, float]:
    start = float(event["ts"])
    return start, start + float(event["dur"])


def union_us(intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    end = -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def identifier(value: Any) -> str | None:
    return None if value in (None, 0, "0", "") else str(value)


def analyze_attribution(
    document: dict[str, Any],
    *,
    step_prefix: str = "execute_",
    first_step: int | None = None,
    last_step: int | None = None,
) -> dict[str, Any]:
    events = [
        event
        for event in document["traceEvents"]
        if event.get("ph") == "X"
        and isinstance(event.get("ts"), (float, int))
        and isinstance(event.get("dur"), (float, int))
        and math.isfinite(event["ts"])
        and math.isfinite(event["dur"])
        and event["dur"] > 0
    ]
    host = [event for event in events if categories(event) & HOST_CATEGORIES]
    device = [event for event in events if categories(event) & DEVICE_CATEGORIES]
    if not device:
        raise ValueError(
            "trace has no device operations; CPU timing is not a substitute"
        )
    by_thread: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    by_external: dict[str, list[int]] = defaultdict(list)
    by_correlation: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(host):
        by_thread[(event.get("pid"), event.get("tid"))].append(index)
        args = event.get("args", {})
        external = identifier(args.get("External id"))
        correlation = identifier(args.get("correlation"))
        if external and categories(event) & {"cpu_op", "user_annotation"}:
            by_external[external].append(index)
        if correlation and categories(event) & {"xpu_runtime", "xpu_driver"}:
            by_correlation[correlation].append(index)

    parent: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    owner: dict[int, int | None] = {}
    steps: list[int] = []
    for indices in by_thread.values():
        stack: list[int] = []
        for index in sorted(indices, key=lambda i: (host[i]["ts"], -host[i]["dur"])):
            start, end = interval(host[index])
            while stack and not (
                host[stack[-1]]["ts"] <= start and interval(host[stack[-1]])[1] >= end
            ):
                stack.pop()
            if stack:
                parent[index] = stack[-1]
                children[stack[-1]].append(index)
            if str(host[index].get("name", "")).startswith(step_prefix):
                steps.append(index)
                owner[index] = index
            else:
                owner[index] = owner.get(stack[-1]) if stack else None
            stack.append(index)
    steps.sort(key=lambda i: host[i]["ts"])
    if not steps:
        raise ValueError(f"trace lacks CPU step scopes starting with {step_prefix!r}")
    step_number = {index: number + 1 for number, index in enumerate(steps)}
    for value in (first_step, last_step):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int)
        ):
            raise ValueError("selected step range must use integer indices")
    first = 1 if first_step is None else first_step
    last = len(steps) if last_step is None else last_step
    if first < 1 or last < first or last > len(steps):
        raise ValueError("selected step range is invalid or empty")
    selected_steps = set(steps[first - 1 : last])

    def path(index: int) -> list[int]:
        result = [index]
        while index in parent:
            index = parent[index]
            result.append(index)
        return list(reversed(result))

    assigned: dict[int, list[dict[str, Any]]] = defaultdict(list)
    unmatched: list[dict[str, Any]] = []
    operation_table: dict[tuple[str, str, str], dict[str, Any]] = {}
    correlation_methods: dict[str, int] = defaultdict(int)
    outside_operations: dict[tuple[str, str], int] = defaultdict(int)
    excluded_operations: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0, "duration_us": 0.0}
    )
    for event in device:
        kinds = categories(event) & DEVICE_CATEGORIES
        if len(kinds) != 1:
            raise ValueError("device event has conflicting operation categories")
        kind = next(iter(kinds))
        args = event.get("args", {})
        external = identifier(args.get("External id"))
        correlation = identifier(args.get("correlation"))
        launches = by_correlation.get(correlation, [])
        # PTI can attach the same correlation ID to the enclosing SYCL call
        # and its nested Level Zero call. The unique runtime call is the
        # submitting host scope; this is not an ambiguous cross-process ID.
        runtime_launches = [
            index for index in launches if "xpu_runtime" in categories(host[index])
        ]
        if len(runtime_launches) == 1:
            launches = runtime_launches
        candidates = by_external.get(external, [])
        if launches:
            pids = {host[index].get("pid") for index in launches}
            candidates = [
                index for index in candidates if host[index].get("pid") in pids
            ]
        if len(candidates) == 1:
            origin = candidates[0]
            method = "external_id"
        elif len(launches) == 1:
            origin = launches[0]
            method = "runtime_correlation"
        else:
            unmatched.append(
                {
                    "name": event.get("name"),
                    "reason": "missing_or_ambiguous_host_origin",
                }
            )
            continue
        step = owner.get(origin)
        if step is None:
            outside_operations[(kind, str(event.get("name", "<unnamed>")))] += 1
            unmatched.append(
                {"name": event.get("name"), "reason": "outside_step_scopes"}
            )
            continue
        if step not in selected_steps:
            key = (kind, str(event.get("name", "<unnamed>")))
            excluded_operations[key]["count"] += 1
            excluded_operations[key]["duration_us"] += event["dur"]
            continue
        correlation_methods[method] += 1
        assigned[step].append(event)
        ancestors = path(origin)
        ops = [i for i in ancestors if "cpu_op" in categories(host[i])]
        scopes = [i for i in ancestors if "user_annotation" in categories(host[i])]
        operator = str(host[ops[-1]]["name"]) if ops else "<no_cpu_operator>"
        scope = str(host[scopes[-1]]["name"]) if scopes else "<no_annotation>"
        key = (kind, str(event.get("name", "<unnamed>")), operator)
        row = operation_table.setdefault(
            key,
            {
                "kind": kind,
                "device_operation": key[1],
                "cpu_operator": operator,
                "count": 0,
                "device_duration_sum_us": 0.0,
                "bytes": 0,
                "annotation_examples": [],
                "submitted_to_start_us": [],
            },
        )
        row["count"] += 1
        row["device_duration_sum_us"] += event["dur"]
        row["bytes"] += int(args.get("bytes", 0))
        if (
            scope not in row["annotation_examples"]
            and len(row["annotation_examples"]) < 4
        ):
            row["annotation_examples"].append(scope)
        try:
            submitted = float(args["submitted"])
        except (KeyError, ValueError, TypeError):
            submitted = math.nan
        if math.isfinite(submitted) and submitted <= event["ts"]:
            row["submitted_to_start_us"].append(event["ts"] - submitted)

    host_table: dict[tuple[str, str], dict[str, Any]] = {}
    host_callers: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, event in enumerate(host):
        if owner[index] is None or owner[index] not in selected_steps:
            continue
        key = (str(event.get("cat")), str(event.get("name")))
        row = host_table.setdefault(
            key,
            {
                "category": key[0],
                "name": key[1],
                "count": 0,
                "cpu_inclusive_us": 0.0,
                "cpu_self_us": 0.0,
            },
        )
        row["count"] += 1
        row["cpu_inclusive_us"] += event["dur"]
        self_us = max(
            0.0, event["dur"] - union_us([interval(host[i]) for i in children[index]])
        )
        row["cpu_self_us"] += self_us
        if categories(event) & {"xpu_runtime", "xpu_driver"}:
            ancestors = [i for i in path(index) if "cpu_op" in categories(host[i])]
            caller = host[ancestors[-1]]["name"] if ancestors else "<no_cpu_operator>"
            call_key = (*key, caller)
            call = host_callers.setdefault(
                call_key,
                {
                    "category": key[0],
                    "name": key[1],
                    "cpu_caller": caller,
                    "count": 0,
                    "cpu_self_us": 0.0,
                },
            )
            call["count"] += 1
            call["cpu_self_us"] += self_us

    per_step = []
    for step in steps:
        if step not in selected_steps:
            continue
        operations = assigned[step]
        start, cpu_end = interval(host[step])
        end = max([cpu_end] + [interval(event)[1] for event in operations])
        busy = union_us([interval(event) for event in operations])
        per_step.append(
            {
                "step": step_number[step],
                "annotation": host[step]["name"],
                "cpu_scope_us": host[step]["dur"],
                "host_start_through_last_completion_us": end - start,
                "device_operation_count": len(operations),
                "kernel_count": sum("kernel" in categories(e) for e in operations),
                "copy_count": sum("gpu_memcpy" in categories(e) for e in operations),
                "device_duration_sum_us": sum(e["dur"] for e in operations),
                "device_busy_union_us": busy,
                "uncovered_by_assigned_device_us": max(0.0, end - start - busy),
            }
        )
    selected = [event for operations in assigned.values() for event in operations]
    selected_host_steps = [index for index in steps if index in selected_steps]
    start = min(host[index]["ts"] for index in selected_host_steps)
    end = max(
        [interval(host[index])[1] for index in selected_host_steps]
        + [interval(event)[1] for event in selected]
    )
    busy = union_us([interval(event) for event in selected])
    excluded_count = sum(
        int(values["count"]) for values in excluded_operations.values()
    )
    excluded_duration = sum(
        values["duration_us"] for values in excluded_operations.values()
    )
    full_busy = union_us(
        [
            (
                max(start, interval(event)[0]),
                min(end, interval(event)[1]),
            )
            for event in device
            if interval(event)[1] > start and interval(event)[0] < end
        ]
    )
    first_device = min((event["ts"] for event in selected), default=end)
    last_device = max((interval(event)[1] for event in selected), default=end)
    return {
        "schema_version": 1,
        "artifact_kind": "correlated_xpu_trace_attribution",
        "diagnostic_only": True,
        "acceptance_eligible": False,
        "association": "external IDs and runtime correlation IDs, then CPU scope ancestry",
        "coverage": {
            "total_device_operations": len(device),
            "assigned_device_operations": len(selected),
            "unassigned_device_operations": len(unmatched),
            "assigned_fraction": len(selected) / len(device),
            "assigned_fraction_definition": "selected scope fraction of all device operations",
            "resolved_host_origin_fraction": (
                (
                    len(device)
                    - sum(
                        entry["reason"] == "missing_or_ambiguous_host_origin"
                        for entry in unmatched
                    )
                )
                / len(device)
            ),
            "unresolved_host_origins": sum(
                entry["reason"] == "missing_or_ambiguous_host_origin"
                for entry in unmatched
            ),
            "resolved_outside_step_scopes": sum(
                entry["reason"] == "outside_step_scopes" for entry in unmatched
            ),
            "methods": dict(correlation_methods),
            "outside_step_operation_counts": [
                {"kind": kind, "name": name, "count": count}
                for (kind, name), count in sorted(outside_operations.items())
            ],
            "captured_step_count": len(steps),
            "selected_step_indices": list(range(first, last + 1)),
            "excluded_guard_operation_counts": [
                {"kind": kind, "name": name, **values}
                for (kind, name), values in sorted(excluded_operations.items())
            ],
            "excluded_guard_device_operations": excluded_count,
            "excluded_guard_device_duration_sum_us": excluded_duration,
            "unassigned_examples": unmatched[:30],
        },
        "window": {
            "host_start_through_last_completion_us": end - start,
            "device_busy_union_us": busy,
            "full_trace_device_busy_union_us": full_busy,
            "uncovered_by_assigned_device_us": max(0.0, end - start - busy),
            "recorded_device_span_us": last_device - first_device,
            "internal_recorded_device_gap_us": max(
                0.0, last_device - first_device - busy
            ),
            "leading_capture_gap_us": max(0.0, first_device - start),
            "trailing_capture_gap_us": max(0.0, end - last_device),
        },
        "steps": per_step,
        "device_operations": sorted(
            operation_table.values(),
            key=lambda row: row["device_duration_sum_us"],
            reverse=True,
        ),
        "cpu_operations": sorted(
            host_table.values(), key=lambda row: row["cpu_self_us"], reverse=True
        ),
        "runtime_driver_by_cpu_caller": sorted(
            host_callers.values(), key=lambda row: row["cpu_self_us"], reverse=True
        ),
        "limitations": [
            "Unassigned operations are excluded from step and window budgets; inspect coverage.",
            "Device duration sums may overlap; device busy uses interval union across queues.",
            "CPU inclusive scopes overlap; self time subtracts immediate nested recorded scopes.",
            "CPU self time includes uninstrumented descendants and is not necessarily Python work.",
            "Idle gaps alone do not prove CPU starvation or a device stall.",
            "Capture-boundary gaps can contain untraced work submitted before profiling started.",
            "Uncovered per-step time can contain work owned by another asynchronously queued step.",
            "Submission-to-start delay is measured queue delay, not its causal explanation.",
            "Per-step completion windows may overlap and must not be summed as wall time.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-prefix", default="execute_")
    parser.add_argument("--first-step", type=int)
    parser.add_argument("--last-step", type=int)
    args = parser.parse_args()
    output = ensure_durable_output(args.output, allow_tmp=False)
    source_files = [args.trace.resolve(strict=True), Path(__file__).resolve()]
    before = {str(path): sha256_file(path) for path in source_files}
    opener = gzip.open if args.trace.name.endswith(".gz") else open
    with opener(args.trace, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    result = analyze_attribution(
        document,
        step_prefix=args.step_prefix,
        first_step=args.first_step,
        last_step=args.last_step,
    )
    after = {str(path): sha256_file(path) for path in source_files}
    result["provenance"] = {
        "before": before,
        "after": after,
        "stable": before == after,
        "step_prefix": args.step_prefix,
        "first_step": args.first_step,
        "last_step": args.last_step,
    }
    if before != after:
        raise ValueError("trace or analyzer changed during analysis")
    write_json_atomic(output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result["coverage"].items()
                if key != "unassigned_examples"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
