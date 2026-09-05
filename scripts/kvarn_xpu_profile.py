#!/usr/bin/env python3
"""Capture a diagnostic Kineto XPU timeline for one Brutus decode workload.

This runner deliberately does not implement a performance gate.  Profiling
changes execution timing, so its output may explain an auto/Kvarn gap but may
never be used as throughput, latency, or parity acceptance evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import math
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    from scripts import kvarn_perf_run as perf
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import kvarn_perf_run as perf


DIAGNOSTIC_WARNING = (
    "Profiled timings include Kineto overhead and are diagnostic only; do not "
    "use them for throughput, latency, parity, promotion, or acceptance claims."
)
STEP_PATTERN = re.compile(
    r"^execute_(?P<total>\d+)_context_(?P<context_requests>\d+)"
    r"\(sq(?P<context_tokens>\d+)sk(?P<context_sk>\d+)"
    r"sqsq(?P<context_sqsq>\d+)sqsk(?P<context_sqsk>\d+)\)"
    r"_generation_(?P<generation_requests>\d+)"
    r"\(sq(?P<generation_tokens>\d+)sk(?P<generation_sk>\d+)"
    r"sqsq(?P<generation_sqsq>\d+)sqsk(?P<generation_sqsk>\d+)\)$"
)
TRACE_SUFFIXES = (".pt.trace.json", ".pt.trace.json.gz")
MIN_PROFILE_STEPS = 20
MAX_PROFILE_STEPS = 50
DECODE_SETTLE_STEPS = 4
VARIANT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,191}$")


def profile_delay_iterations(
    *, context: int, batch: int, max_num_batched_tokens: int
) -> int:
    """Conservatively skip all chunked prefill plus a few decode iterations."""
    if min(context, batch, max_num_batched_tokens) < 1:
        raise perf.RunnerError("profile delay inputs must be positive")
    return batch * math.ceil(context / max_num_batched_tokens) + DECODE_SETTLE_STEPS


def profiler_config(
    trace_dir: Path, *, delay_iterations: int, profile_steps: int
) -> dict[str, Any]:
    if not trace_dir.is_absolute():
        raise perf.RunnerError("Kineto trace directory must be absolute")
    if delay_iterations < 1:
        raise perf.RunnerError("profile delay must be positive")
    if not MIN_PROFILE_STEPS <= profile_steps <= MAX_PROFILE_STEPS:
        raise perf.RunnerError(
            f"profile steps must be in [{MIN_PROFILE_STEPS}, {MAX_PROFILE_STEPS}]"
        )
    return {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir),
        "torch_profiler_with_stack": False,
        "torch_profiler_with_flops": False,
        "torch_profiler_use_gzip": True,
        "torch_profiler_dump_cuda_time_total": False,
        "torch_profiler_record_shapes": False,
        "torch_profiler_with_memory": False,
        "detailed_trace_annotation": True,
        "ignore_frontend": True,
        "delay_iterations": delay_iterations,
        "max_iterations": profile_steps,
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_xpu_kernel_event(event: Mapping[str, Any]) -> bool:
    categories = {part.strip().lower() for part in str(event.get("cat", "")).split(",")}
    return (
        event.get("ph") == "X"
        and "kernel" in categories
        and (_numeric(event.get("ts")) is not None)
        and ((_numeric(event.get("dur")) or 0.0) > 0)
    )


def load_trace(path: Path) -> dict[str, Any]:
    try:
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                document = json.load(stream)
        else:
            document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise perf.RunnerError(f"cannot load Kineto trace {path}: {exc}") from exc
    if isinstance(document, list):
        document = {"traceEvents": document}
    if not isinstance(document, dict) or not isinstance(
        document.get("traceEvents"), list
    ):
        raise perf.RunnerError(f"Kineto trace has no traceEvents array: {path}")
    return document


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise perf.RunnerError("cannot summarize an empty duration series")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _merge_intervals(
    intervals: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _interval_summary(intervals: Sequence[tuple[float, float]]) -> dict[str, Any]:
    if not intervals:
        raise perf.RunnerError("cannot summarize an empty XPU interval series")
    merged = _merge_intervals(intervals)
    gaps = [
        max(0.0, current[0] - previous[1]) for previous, current in pairwise(merged)
    ]
    span = merged[-1][1] - merged[0][0]
    busy = sum(end - start for start, end in merged)
    idle = max(0.0, span - busy)
    return {
        "first_kernel_ts_us": merged[0][0],
        "last_kernel_end_ts_us": merged[-1][1],
        "device_span_us": span,
        "device_busy_union_us": busy,
        "device_idle_gap_us": idle,
        "device_busy_fraction": busy / span,
        "device_idle_fraction": idle / span,
        "idle_gap_count": len(gaps),
        "idle_gap_mean_us": statistics.mean(gaps) if gaps else 0.0,
        "idle_gap_p95_us": _percentile(gaps, 95) if gaps else 0.0,
        "idle_gap_max_us": max(gaps, default=0.0),
    }


def _event_interval(event: Mapping[str, Any]) -> tuple[float, float]:
    start = float(event["ts"])
    return start, start + float(event["dur"])


def _queue_key(event: Mapping[str, Any]) -> str:
    args = event.get("args")
    if isinstance(args, dict) and "device" in args and "stream" in args:
        return f"device={args['device']},stream={args['stream']}"
    return f"pid={event.get('pid')},tid={event.get('tid')}"


def _duration_stats(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "device_total_us": sum(values),
        "device_mean_us": statistics.mean(values),
        "device_median_us": statistics.median(values),
        "device_p95_us": _percentile(values, 95),
        "device_max_us": max(values),
    }


def _parse_steps(
    events: Sequence[Mapping[str, Any]], *, batch: int, context: int, expected: int
) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for event in events:
        if event.get("ph") != "X" or not isinstance(event.get("name"), str):
            continue
        match = STEP_PATTERN.fullmatch(event["name"])
        if match is None:
            continue
        start = _numeric(event.get("ts"))
        duration = _numeric(event.get("dur"))
        if start is None or duration is None or duration <= 0:
            raise perf.RunnerError("decode-step annotation has invalid timestamp")
        fields = {name: int(value) for name, value in match.groupdict().items()}
        expected_fields = {
            "total": batch,
            "context_requests": 0,
            "context_tokens": 0,
            "context_sk": 0,
            "context_sqsq": 0,
            "context_sqsk": 0,
            "generation_requests": batch,
            "generation_tokens": batch,
            "generation_sqsq": batch,
        }
        mismatches = {
            name: {"actual": fields[name], "expected": value}
            for name, value in expected_fields.items()
            if fields[name] != value
        }
        if fields["generation_sk"] < batch * context:
            mismatches["generation_sk"] = {
                "actual": fields["generation_sk"],
                "expected_minimum": batch * context,
            }
        if fields["generation_sqsk"] != fields["generation_sk"]:
            mismatches["generation_sqsk"] = {
                "actual": fields["generation_sqsk"],
                "expected": fields["generation_sk"],
            }
        if mismatches:
            raise perf.RunnerError(
                "profile contains a non-steady-decode annotated step: "
                + json.dumps(mismatches, sort_keys=True)
            )
        steps.append({"ts": start, "end": start + duration, **fields})
    steps.sort(key=lambda step: step["ts"])
    if len(steps) != expected:
        raise perf.RunnerError(
            f"expected exactly {expected} steady decode steps, found {len(steps)}"
        )
    sequence_lengths = [step["generation_sk"] for step in steps]
    for previous, current in pairwise(sequence_lengths):
        if current - previous != batch:
            raise perf.RunnerError(
                "profiled decode sequence length did not advance by exactly "
                f"B{batch}: {previous} -> {current}"
            )
    return steps


def trace_device_names(document: Mapping[str, Any]) -> list[str]:
    properties = document.get("deviceProperties", [])
    if not isinstance(properties, list):
        return []
    return [
        value
        for item in properties
        if isinstance(item, dict)
        and isinstance((value := item.get("name")), str)
        and value
    ]


def analyze_trace(
    document: Mapping[str, Any],
    *,
    arm: str,
    context: int,
    batch: int,
    profile_steps: int,
    hardware_preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and summarize GPU events without producing a perf conclusion."""
    perf.validate_xpu_preflight(hardware_preflight)
    raw_events = document.get("traceEvents")
    if not isinstance(raw_events, list):
        raise perf.RunnerError("Kineto trace has no traceEvents array")
    events = [event for event in raw_events if isinstance(event, dict)]
    kernels = [event for event in events if _is_xpu_kernel_event(event)]
    if not kernels:
        raise perf.RunnerError("Kineto trace has no positive-duration XPU kernels")

    device_names = trace_device_names(document)
    if device_names and perf.EXPECTED_XPU_DEVICE_NAME not in device_names:
        raise perf.RunnerError(
            "Kineto trace device properties do not name the preflight B70: "
            + json.dumps(device_names)
        )
    steps = _parse_steps(events, batch=batch, context=context, expected=profile_steps)
    kernels = [event for event in kernels if float(event["ts"]) >= steps[0]["ts"]]
    native_events = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("name") == "kvarn_native_xpu_decode"
        and _numeric(event.get("ts")) is not None
    ]
    native_annotations = len(native_events)
    if arm == "candidate" and native_annotations < profile_steps:
        raise perf.RunnerError(
            "native profile lacks Kvarn decoder annotations for every decode step"
        )
    if arm == "reference" and native_annotations:
        raise perf.RunnerError("auto profile unexpectedly contains native Kvarn decode")

    per_step: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        upper = steps[index + 1]["ts"] if index + 1 < len(steps) else math.inf
        selected = [
            event for event in kernels if step["ts"] <= float(event["ts"]) < upper
        ]
        if not selected:
            raise perf.RunnerError(f"decode step {index + 1} has no XPU kernel events")
        step_native_annotations = sum(
            step["ts"] <= float(event["ts"]) < upper for event in native_events
        )
        if arm == "candidate" and step_native_annotations == 0:
            raise perf.RunnerError(
                f"decode step {index + 1} has no native Kvarn annotation"
            )
        intervals = [_event_interval(event) for event in selected]
        per_step.append(
            {
                "step": index + 1,
                "generation_sequence_length_sum": step["generation_sk"],
                "xpu_kernel_count": len(selected),
                "xpu_kernel_duration_sum_us": sum(float(e["dur"]) for e in selected),
                "native_decode_annotation_count": step_native_annotations,
                **_interval_summary(intervals),
            }
        )

    by_name: dict[str, list[float]] = defaultdict(list)
    by_queue: dict[str, list[tuple[float, float]]] = defaultdict(list)
    all_intervals: list[tuple[float, float]] = []
    for event in kernels:
        name = str(event.get("name", "<unnamed>"))
        by_name[name].append(float(event["dur"]))
        interval = _event_interval(event)
        by_queue[_queue_key(event)].append(interval)
        all_intervals.append(interval)
    kernel_table = [
        {"kernel": name, **_duration_stats(durations)}
        for name, durations in by_name.items()
    ]
    kernel_table.sort(key=lambda item: item["device_total_us"], reverse=True)

    device_timeline = _interval_summary(all_intervals)
    kernel_duration_sum = sum(float(event["dur"]) for event in kernels)
    return {
        "schema_version": 1,
        "artifact_kind": "gpu_diagnostic_profile",
        "status": "valid_diagnostic",
        "diagnostic_only": True,
        "promotable": False,
        "acceptance_eligible": False,
        "parity_conclusion": None,
        "warning": DIAGNOSTIC_WARNING,
        "timing_source": "Kineto XPU device kernel events",
        "accelerator": "xpu",
        "device_name": perf.EXPECTED_XPU_DEVICE_NAME,
        "trace_device_names": device_names,
        "arm": arm,
        "context": context,
        "batch": batch,
        "steady_decode_steps": len(steps),
        "native_decode_annotation_count": native_annotations,
        "xpu_kernel_count": len(kernels),
        "xpu_kernel_duration_sum_us": kernel_duration_sum,
        "xpu_device_timeline": device_timeline,
        "xpu_queue_timelines": {
            queue: _interval_summary(intervals)
            for queue, intervals in sorted(by_queue.items())
        },
        "decode_steps": per_step,
        "distinct_xpu_kernel_names": len(kernel_table),
        "xpu_kernels_by_device_time": kernel_table[:100],
        "gpu_leaderboard_metrics": {
            "diagnostic_only": True,
            "timing_source": "Kineto XPU device kernel events",
            "profiled_decode_steps": len(steps),
            "device_kernel_time_us": kernel_duration_sum,
            "device_kernel_time_per_step_us": kernel_duration_sum / len(steps),
            "device_busy_union_us": device_timeline["device_busy_union_us"],
            "device_span_us": device_timeline["device_span_us"],
            "device_idle_fraction": device_timeline["device_idle_fraction"],
            "kernel_launch_count": len(kernels),
            "kernel_launches_per_step": len(kernels) / len(steps),
        },
    }


def variant_provenance(
    run: perf.PlannedRun, args: argparse.Namespace
) -> dict[str, Any]:
    """Describe the small, currently supported first-round variant space."""
    layout = perf.native_layout_for_run(run, args)
    splits = perf.native_splits_for_run(run, args)
    native_frontend = perf.native_frontend_for_run(run, args)
    flush_indices = perf.flush_index_materialization_environment(args)
    flush_writer = perf.flush_writer_for_run(run, args)
    prefill_store = perf.prefill_store_for_run(run, args)
    forward_pool_ensure = perf.forward_pool_ensure_for_run(run, args)
    qlen1_inline_plan = perf.qlen1_inline_plan_for_run(run, args)
    decode_window = perf.decode_fp16_window_blocks_for_run(run, args)
    decode_low_water = perf.decode_fp16_low_water_blocks_for_run(run, args)
    decode_flush_scope = perf.decode_flush_scope_for_run(run, args)
    if run.arm == "reference":
        kernel_strategy = "auto_vllm_backend"
        fusion_selection = "backend_default"
        scheduling_selection = "backend_default"
        generated_id = f"auto-natural-b{run.workload.batch}"
    else:
        kernel_strategy = (
            f"native_xe2_decode_{perf.native_kernel_variant_for_run(run, args)}"
        )
        fusion_selection = (
            f"fused_attention_decode_{flush_indices}_flush_"
            f"{flush_writer}_writer_{prefill_store}_prefill_store_"
            f"{native_frontend}_frontend_"
            f"{forward_pool_ensure}_forward_pool_ensure_"
            f"{qlen1_inline_plan}_qlen1_inline_plan_"
            f"decode_fp16_window_{decode_window}_low_water_{decode_low_water}_"
            f"flush_scope_{decode_flush_scope}"
        )
        scheduling_selection = "split_k"
        generated_id = (
            f"native-xe2-{layout}-{perf.native_kernel_variant_for_run(run, args)}-"
            f"split{splits}-{flush_indices}-flush-{flush_writer}-writer-"
            f"{prefill_store}-prefill-store-"
            f"{native_frontend}-frontend-"
            f"{forward_pool_ensure}-forward-pool-ensure-"
            f"{perf.QLEN1_INLINE_PLAN_IDS[qlen1_inline_plan]}-"
            f"dw{decode_window}-lw{decode_low_water}-"
            f"{perf.DECODE_FLUSH_SCOPE_IDS[decode_flush_scope]}-"
            f"b{run.workload.batch}"
        )
    variant_id = args.variant_id or generated_id
    if VARIANT_ID_PATTERN.fullmatch(variant_id) is None:
        raise perf.RunnerError(
            "variant id must be a lowercase slug of at most 192 characters"
        )
    return {
        "variant_id": variant_id,
        "layout": layout,
        "kernel_strategy": kernel_strategy,
        "split_count": splits,
        "max_split_count": perf.native_max_splits_for_run(run, args),
        "split_policy": perf.native_split_policy_for_run(run, args),
        "split_policy_selector": perf.native_split_policy_name_for_run(run, args),
        "native_frontend": native_frontend,
        "forward_pool_ensure": forward_pool_ensure,
        "qlen1_inline_plan": qlen1_inline_plan,
        "decode_flush_scope": decode_flush_scope,
        "decode_fp16_low_water_blocks": decode_low_water,
        "decode_fp16_window_blocks": decode_window,
        "request_stable_projection_rows": (
            perf.request_stable_projection_rows_environment(args)
        ),
        "request_stable_rmsnorm": perf.request_stable_rmsnorm_environment(args),
        "request_stability_qualification": (
            "qualified-default"
            if perf.request_stable_projection_rows_environment(args) == "1"
            and perf.request_stable_rmsnorm_environment(args) == "1"
            else "diagnostic-unqualified"
        ),
        "flush_index_materialization": flush_indices,
        "flush_writer": flush_writer,
        "prefill_store": prefill_store,
        "fusion_selection": fusion_selection,
        "scheduling_selection": scheduling_selection,
        "scheduler_max_num_batched_tokens": args.max_num_batched_tokens,
        "scheduler_max_num_seqs": run.workload.batch,
    }


def _replace_option(command: list[str], name: str, value: str) -> None:
    try:
        command[command.index(name) + 1] = value
    except (ValueError, IndexError) as exc:
        raise perf.RunnerError(f"generated benchmark command lacks {name}") from exc


def profile_benchmark_command(
    run: perf.PlannedRun, args: argparse.Namespace, raw_result: Path
) -> list[str]:
    command = perf.benchmark_command(run, args, raw_result)
    _replace_option(command, "--num-warmups", str(run.workload.batch))
    command.append("--profile")
    return command


def verify_profiler_process_config(
    argv: Sequence[str], expected: Mapping[str, Any]
) -> None:
    raw_config = perf._arg_after(argv, "--profiler-config")
    try:
        captured = json.loads(raw_config) if raw_config is not None else None
    except (json.JSONDecodeError, TypeError) as exc:
        raise perf.RunnerError(
            "captured service process has an invalid profiler config"
        ) from exc
    if captured != dict(expected):
        raise perf.RunnerError(
            "captured service process lost the requested profiler config"
        )


def start_profile_service(
    run: perf.PlannedRun,
    run_dir: Path,
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> perf.ServiceProcess:
    command = [
        *perf.service_command(
            run,
            args,
            extra_arguments=(
                "--profiler-config",
                json.dumps(config, sort_keys=True, separators=(",", ":")),
            ),
        ),
    ]
    perf.write_json_atomic(run_dir / "service-command.json", command)
    for attempt in range(1, args.startup_attempts + 1):
        perf.assert_port_unused(args.base_url)
        log_path = run_dir / "engine.log"
        log_stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=args.config_repo,
            env=perf.service_environment(run, args),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_group = os.getpgid(process.pid)
        args.supervisor.register(process_group, f"profile-service:{run.arm}")
        provisional = perf.ServiceProcess(
            process=process,
            process_group=process_group,
            log_stream=log_stream,
            log_path=log_path,
            engine_pid=process.pid,
            argv=[],
            environment={},
            supervisor=args.supervisor,
        )
        try:
            perf.wait_for_ready(process, args)
            engine_pid, argv, environment = perf.capture_engine_process(process)
            perf.verify_service_profile(argv, environment, run, args)
            verify_profiler_process_config(argv, config)
            provisional.engine_pid = engine_pid
            provisional.argv = argv
            provisional.environment = environment
            return provisional
        except BaseException as exc:
            perf.stop_service(provisional, args.shutdown_timeout)
            if isinstance(exc, (perf.RunnerInterrupted, KeyboardInterrupt)):
                raise
            if attempt == args.startup_attempts:
                raise
            log_path.replace(run_dir / f"engine-failed-startup-{attempt}.log")
            deadline = time.monotonic() + args.shutdown_timeout
            while time.monotonic() < deadline:
                try:
                    perf.assert_port_unused(args.base_url)
                    break
                except perf.RunnerError:
                    time.sleep(0.25)
            else:
                raise perf.RunnerError(
                    "service port remained occupied after failed startup"
                )
    raise AssertionError("unreachable")


def find_kineto_trace(trace_dir: Path) -> Path:
    paths = [
        path
        for path in trace_dir.rglob("*")
        if path.is_file() and path.name.endswith(TRACE_SUFFIXES)
    ]
    if len(paths) != 1:
        raise perf.RunnerError(
            f"expected exactly one Kineto XPU trace, found {[str(p) for p in paths]}"
        )
    return paths[0]


def _run_directory(args: argparse.Namespace) -> Path:
    return (
        args.output_dir
        / f"{args.arm}-{args.native_layout}-b{args.batch}-context-{args.context}"
    )


def execute(args: argparse.Namespace) -> dict[str, Any]:
    workload = perf.Workload(
        context=args.context,
        batch=args.batch,
        output_tokens=args.output_tokens,
        num_prompts=args.batch,
        seed=args.seed,
    )
    run = perf.PlannedRun(workload=workload, arm=args.arm, order=1)
    native_frontend = perf.native_frontend_for_run(run, args)
    forward_pool_ensure = perf.forward_pool_ensure_for_run(run, args)
    qlen1_inline_plan = perf.qlen1_inline_plan_for_run(run, args)
    expected_launcher = perf.launcher_name(run, args)
    if args.launcher is not None and args.launcher != expected_launcher:
        raise perf.RunnerError(
            f"--launcher {args.launcher!r} conflicts with {args.arm}/"
            f"{args.native_layout}/B{args.batch} ({expected_launcher!r})"
        )
    args.resolved_launchers = perf.resolve_launchers([run], args)
    variant = variant_provenance(run, args)
    run_dir = _run_directory(args)
    run_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = (run_dir / "kineto").resolve()
    trace_dir.mkdir()
    delay = profile_delay_iterations(
        context=args.context,
        batch=args.batch,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    config = profiler_config(
        trace_dir, delay_iterations=delay, profile_steps=args.profile_steps
    )
    hardware = perf.probe_xpu_hardware(args)
    hardware_path = run_dir / "hardware-preflight.json"
    perf.write_json_atomic(hardware_path, hardware)

    run_uuid = str(uuid.uuid4())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "gpu_diagnostic_profile_run",
        "status": "running",
        "diagnostic_only": True,
        "promotable": False,
        "acceptance_eligible": False,
        "parity_conclusion": None,
        "warning": DIAGNOSTIC_WARNING,
        "run_uuid": run_uuid,
        "started_at": perf.utc_timestamp(),
        "workload": dataclasses.asdict(workload),
        "arm": args.arm,
        "native_layout": perf.native_layout_for_run(run, args),
        "native_layout_environment": perf.NATIVE_LAYOUT_ENV[
            perf.native_layout_for_run(run, args)
        ],
        "native_cache_layout_environment": perf.native_layout_for_run(run, args),
        "native_nominal_splits": perf.native_splits_for_run(run, args),
        "native_effective_splits": perf.native_splits_for_run(run, args),
        "native_split_policy_contract": perf.native_split_policy_contract(args),
        "native_splits_environment": perf.native_splits_environment_for_run(run, args),
        "native_split_policy": perf.native_split_policy_for_run(run, args),
        "native_split_policy_selector": perf.native_split_policy_name_for_run(
            run, args
        ),
        "native_max_splits": perf.native_max_splits_for_run(run, args),
        "native_kernel_variant": perf.native_kernel_variant_for_run(run, args),
        "native_kernel_variant_id": perf.NATIVE_KERNEL_VARIANTS[
            perf.native_kernel_variant_for_run(run, args)
        ],
        "native_frontend": native_frontend,
        "forward_pool_ensure": forward_pool_ensure,
        "qlen1_inline_plan": qlen1_inline_plan,
        "decode_flush_scope": perf.decode_flush_scope_for_run(run, args),
        "decode_fp16_low_water_blocks": (
            perf.decode_fp16_low_water_blocks_for_run(run, args)
        ),
        "decode_fp16_window_blocks": perf.decode_fp16_window_blocks_for_run(run, args),
        "flush_writer": perf.flush_writer_for_run(run, args),
        "prefill_store": perf.prefill_store_for_run(run, args),
        "native_factory_marker": (
            perf.kvarn_factory_marker(
                cache_layout=perf.native_layout_for_run(run, args),
                kernel_variant=perf.native_kernel_variant_for_run(run, args),
                max_decode_splits=perf.native_max_splits_for_run(run, args),
                split_policy=perf.native_split_policy_name_for_run(run, args),
            )
            if args.arm == "candidate"
            else "unavailable"
        ),
        "variant_id": variant["variant_id"],
        "variant_parameters": variant,
        "logical_launcher": expected_launcher,
        "resolved_launcher": args.resolved_launchers[expected_launcher],
        "launcher_binding": perf.launcher_binding_for_run(run, args),
        "repositories": [
            perf.repository_state("vllm-xpu-nix", args.packaging_repo),
            perf.repository_state("nix-config", args.config_repo),
        ],
        "profiler_config": config,
        "hardware_preflight": str(hardware_path),
        "hardware_preflight_sha256": perf.sha256_file(hardware_path),
    }
    perf.write_json_atomic(run_dir / "run.json", manifest)
    service: perf.ServiceProcess | None = None
    try:
        service = start_profile_service(run, run_dir, args, config)
        service_profile = perf.service_profile_evidence(
            service.argv,
            service.environment,
            variant_provenance=perf.variant_provenance_for_run(run, args),
        )
        perf.write_json_atomic(
            run_dir / "service-argv.json", service_profile["redacted_argv"]
        )
        perf.write_json_atomic(
            run_dir / "service-environment.json",
            service_profile["redacted_environment"],
        )
        perf.write_json_atomic(
            run_dir / "diagnostic-service-profile.json", service_profile
        )
        identity = perf.verify_candidate_identity(service.argv, args.candidate_env)
        identity["decode_fp16_window_blocks"] = perf.decode_fp16_window_blocks_for_run(
            run, args
        )
        identity["decode_fp16_low_water_blocks"] = (
            perf.decode_fp16_low_water_blocks_for_run(run, args)
        )
        identity["decode_flush_scope"] = perf.decode_flush_scope_for_run(run, args)
        identity["qlen1_inline_plan"] = qlen1_inline_plan
        perf.write_json_atomic(run_dir / "candidate-identity.json", identity)

        raw_result = run_dir / "profiled-workload.raw.json"
        command = profile_benchmark_command(run, args, raw_result)
        perf.write_json_atomic(run_dir / "profiled-workload-argv.json", command)
        samples: list[dict[str, Any]] = []
        metric_errors: list[str] = []
        stop_sampling = threading.Event()
        sampler = threading.Thread(
            target=perf.sample_scheduler,
            kwargs={
                "args": args,
                "required_running": args.batch,
                "stop": stop_sampling,
                "samples": samples,
                "errors": metric_errors,
            },
            daemon=True,
        )
        sampler.start()
        with (run_dir / "profiled-workload.stdout.log").open(
            "w", encoding="utf-8"
        ) as output:
            try:
                returncode = perf.run_managed_process(
                    command,
                    cwd=args.packaging_repo,
                    environment=perf.runner_environment(args),
                    output=output,
                    timeout=args.benchmark_timeout,
                    supervisor=args.supervisor,
                    label="profiled diagnostic workload",
                )
            finally:
                stop_sampling.set()
                sampler.join(timeout=5.0)
        if sampler.is_alive():
            raise perf.RunnerError("scheduler sampler did not stop")
        scheduler = perf.scheduler_summary(samples, metric_errors, args.batch)
        perf.write_json_atomic(run_dir / "scheduler-metrics.json", scheduler)
        if returncode != 0:
            raise perf.RunnerError(f"profiled benchmark exited {returncode}")
        perf.load_and_validate_benchmark_result(raw_result, workload)
        if scheduler["peak_running"] < args.batch:
            raise perf.RunnerError(
                f"scheduler peak {scheduler['peak_running']:g} did not reach B{args.batch}"
            )

        engine_pid = service.engine_pid
        perf.stop_service(service, args.shutdown_timeout)
        service = None
        log_scan = perf.validate_engine_log(
            run_dir / "engine.log",
            native=args.arm == "candidate",
            expected_layout=perf.native_layout_for_run(run, args),
            expected_kernel_variant=(
                perf.native_kernel_variant_for_run(run, args)
                if args.arm == "candidate"
                else None
            ),
            expected_max_splits=(
                perf.native_max_splits_for_run(run, args)
                if args.arm == "candidate"
                else None
            ),
            expected_split_policy=(
                perf.native_split_policy_name_for_run(run, args)
                if args.arm == "candidate"
                else None
            ),
            expected_frontend=native_frontend,
            expected_forward_pool_ensure=forward_pool_ensure,
            expected_qlen1_inline_plan=qlen1_inline_plan,
            expected_decode_flush_scope=perf.decode_flush_scope_for_run(run, args),
            expected_decode_fp16_low_water_blocks=(
                perf.decode_fp16_low_water_blocks_for_run(run, args)
            ),
            expected_decode_fp16_window_blocks=(
                perf.decode_fp16_window_blocks_for_run(run, args)
            ),
            require_decode_flush_batch_execution=args.output_tokens >= 768,
        )
        perf.write_json_atomic(run_dir / "engine-log-scan.json", log_scan)
        trace_path = find_kineto_trace(trace_dir)
        trace = load_trace(trace_path)
        summary = analyze_trace(
            trace,
            arm=args.arm,
            context=args.context,
            batch=args.batch,
            profile_steps=args.profile_steps,
            hardware_preflight=hardware,
        )
        summary.update(
            {
                "created_at": perf.utc_timestamp(),
                "run_uuid": run_uuid,
                "native_layout": perf.native_layout_for_run(run, args),
                "native_layout_environment": perf.NATIVE_LAYOUT_ENV[
                    perf.native_layout_for_run(run, args)
                ],
                "native_cache_layout_environment": service_profile[
                    "native_cache_layout_environment"
                ],
                "native_nominal_splits": perf.native_splits_for_run(run, args),
                "native_effective_splits": perf.native_splits_for_run(run, args),
                "native_split_policy_contract": perf.native_split_policy_contract(args),
                "native_split_policy": perf.native_split_policy_for_run(run, args),
                "native_split_policy_selector": perf.native_split_policy_name_for_run(
                    run, args
                ),
                "native_max_splits": perf.native_max_splits_for_run(run, args),
                "native_kernel_variant": perf.native_kernel_variant_for_run(run, args),
                "native_kernel_variant_id": perf.NATIVE_KERNEL_VARIANTS[
                    perf.native_kernel_variant_for_run(run, args)
                ],
                "native_kernel_variant_environment": service_profile[
                    "native_kernel_variant_environment"
                ],
                "native_max_splits_environment": service_profile[
                    "native_max_splits_environment"
                ],
                "native_split_policy_environment": service_profile[
                    "native_split_policy_environment"
                ],
                "native_factory_marker": log_scan["native_layout_log_marker"],
                "native_factory_selection_verified": log_scan[
                    "native_factory_selection_verified"
                ],
                "native_frontend": native_frontend,
                "forward_pool_ensure": forward_pool_ensure,
                "qlen1_inline_plan": qlen1_inline_plan,
                "qlen1_inline_plan_environment": service_profile[
                    "qlen1_inline_plan_environment"
                ],
                "forward_pool_ensure_environment": service_profile[
                    "forward_pool_ensure_environment"
                ],
                "decode_flush_scope": perf.decode_flush_scope_for_run(run, args),
                "decode_flush_scope_environment": service_profile[
                    "decode_flush_scope_environment"
                ],
                "decode_fp16_low_water_blocks": (
                    perf.decode_fp16_low_water_blocks_for_run(run, args)
                ),
                "decode_fp16_low_water_blocks_environment": service_profile[
                    "decode_fp16_low_water_blocks_environment"
                ],
                "decode_fp16_window_blocks": (
                    perf.decode_fp16_window_blocks_for_run(run, args)
                ),
                "decode_fp16_window_blocks_environment": service_profile[
                    "decode_fp16_window_blocks_environment"
                ],
                "decode_flush_batch_active_verified": log_scan[
                    "decode_flush_batch_active_verified"
                ],
                "decode_flush_batch_execution_required": log_scan[
                    "decode_flush_batch_execution_required"
                ],
                "decode_flush_batch_execution_status": log_scan[
                    "decode_flush_batch_execution_status"
                ],
                "decode_flush_batch_events": log_scan["decode_flush_batch_events"],
                "flush_writer": perf.flush_writer_for_run(run, args),
                "prefill_store": perf.prefill_store_for_run(run, args),
                "native_frontend_active_verified": log_scan[
                    "native_frontend_active_verified"
                ],
                "native_frontend_log_marker": log_scan["native_frontend_log_marker"],
                "native_frontend_inline_active_verified": log_scan[
                    "native_frontend_inline_active_verified"
                ],
                "native_frontend_inline_log_marker": log_scan[
                    "native_frontend_inline_log_marker"
                ],
                "forward_pool_ensure_active_verified": log_scan[
                    "forward_pool_ensure_active_verified"
                ],
                "forward_pool_ensure_log_marker": log_scan[
                    "forward_pool_ensure_log_marker"
                ],
                "qlen1_inline_plan_selection_verified": log_scan[
                    "qlen1_inline_plan_selection_verified"
                ],
                "qlen1_inline_plan_selection_log_marker": log_scan[
                    "qlen1_inline_plan_selection_log_marker"
                ],
                "qlen1_inline_plan_active_verified": log_scan[
                    "qlen1_inline_plan_active_verified"
                ],
                "qlen1_inline_plan_active_log_marker": log_scan[
                    "qlen1_inline_plan_active_log_marker"
                ],
                "variant_id": variant["variant_id"],
                "variant_parameters": variant,
                "logical_launcher": expected_launcher,
                "resolved_launcher": args.resolved_launchers[expected_launcher],
                "launcher_binding": perf.launcher_binding_for_run(run, args),
                "process_package": identity["process_package"],
                "process_closure_sha256": identity["process_closure_sha256"],
                "candidate_closure_sha256": identity["candidate_closure_sha256"],
                "engine_log_sha256": perf.sha256_file(run_dir / "engine.log"),
                "xpu_runtime": log_scan["xpu_runtime"],
                "native_dispatch_marker_required": args.arm == "candidate",
                "native_dispatch_marker": (
                    perf.NATIVE_DISPATCH if args.arm == "candidate" else None
                ),
                "hardware_preflight_sha256": perf.sha256_file(hardware_path),
                "kineto_trace": str(trace_path),
                "kineto_trace_sha256": perf.sha256_file(trace_path),
                "profile_delay_iterations": delay,
                "profile_steps_requested": args.profile_steps,
            }
        )
        summary_path = run_dir / "profile-summary.json"
        perf.write_json_atomic(summary_path, summary)
        manifest.update(
            status="valid_diagnostic",
            finished_at=perf.utc_timestamp(),
            service_pid=engine_pid,
            profile_summary=str(summary_path),
            profile_summary_sha256=perf.sha256_file(summary_path),
            service_profile=str(run_dir / "diagnostic-service-profile.json"),
            service_profile_sha256=perf.sha256_file(
                run_dir / "diagnostic-service-profile.json"
            ),
            candidate_identity=str(run_dir / "candidate-identity.json"),
            candidate_identity_sha256=perf.sha256_file(
                run_dir / "candidate-identity.json"
            ),
            engine_log=str(run_dir / "engine.log"),
            engine_log_sha256=perf.sha256_file(run_dir / "engine.log"),
            engine_log_scan=str(run_dir / "engine-log-scan.json"),
            engine_log_scan_sha256=perf.sha256_file(run_dir / "engine-log-scan.json"),
            kineto_trace=str(trace_path),
            kineto_trace_sha256=perf.sha256_file(trace_path),
            native_factory_marker=log_scan["native_layout_log_marker"],
            native_factory_selection_verified=log_scan[
                "native_factory_selection_verified"
            ],
            native_frontend=native_frontend,
            forward_pool_ensure=forward_pool_ensure,
            qlen1_inline_plan=qlen1_inline_plan,
            decode_flush_scope=perf.decode_flush_scope_for_run(run, args),
            decode_fp16_low_water_blocks=(
                perf.decode_fp16_low_water_blocks_for_run(run, args)
            ),
            decode_fp16_window_blocks=perf.decode_fp16_window_blocks_for_run(run, args),
            decode_flush_batch_active_verified=log_scan[
                "decode_flush_batch_active_verified"
            ],
            decode_flush_batch_execution_required=log_scan[
                "decode_flush_batch_execution_required"
            ],
            decode_flush_batch_execution_status=log_scan[
                "decode_flush_batch_execution_status"
            ],
            decode_flush_batch_events=log_scan["decode_flush_batch_events"],
            flush_writer=perf.flush_writer_for_run(run, args),
            prefill_store=perf.prefill_store_for_run(run, args),
            native_frontend_active_verified=log_scan["native_frontend_active_verified"],
            native_frontend_log_marker=log_scan["native_frontend_log_marker"],
            native_frontend_inline_active_verified=log_scan[
                "native_frontend_inline_active_verified"
            ],
            native_frontend_inline_log_marker=log_scan[
                "native_frontend_inline_log_marker"
            ],
            forward_pool_ensure_active_verified=log_scan[
                "forward_pool_ensure_active_verified"
            ],
            forward_pool_ensure_log_marker=log_scan["forward_pool_ensure_log_marker"],
            qlen1_inline_plan_selection_verified=log_scan[
                "qlen1_inline_plan_selection_verified"
            ],
            qlen1_inline_plan_selection_log_marker=log_scan[
                "qlen1_inline_plan_selection_log_marker"
            ],
            qlen1_inline_plan_active_verified=log_scan[
                "qlen1_inline_plan_active_verified"
            ],
            qlen1_inline_plan_active_log_marker=log_scan[
                "qlen1_inline_plan_active_log_marker"
            ],
        )
        perf.write_json_atomic(run_dir / "run.json", manifest)
        perf.write_checksums(args.output_dir)
        return manifest
    except BaseException as exc:
        if service is not None:
            try:
                perf.stop_service(service, args.shutdown_timeout)
            except (
                OSError,
                subprocess.SubprocessError,
                perf.RunnerError,
            ) as stop_error:
                manifest["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        manifest.update(
            status="failed",
            finished_at=perf.utc_timestamp(),
            error=f"{type(exc).__name__}: {exc}",
        )
        perf.write_json_atomic(run_dir / "run.json", manifest)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=("reference", "candidate"), default="candidate"
    )
    parser.add_argument("--context", type=int, default=65023)
    parser.add_argument("--batch", type=int, choices=(1, 4), default=1)
    parser.add_argument("--output-tokens", type=int, default=96)
    parser.add_argument("--profile-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--variant-id",
        help=(
            "lowercase factory-candidate slug; defaults to a deterministic "
            "arm/layout/split/batch identifier"
        ),
    )
    parser.add_argument("--native-layout", choices=perf.NATIVE_LAYOUTS)
    parser.add_argument(
        "--native-kernel-variant",
        choices=tuple(perf.NATIVE_KERNEL_VARIANTS),
    )
    parser.add_argument(
        "--native-split-policy",
        choices=perf.NATIVE_SPLIT_POLICIES,
    )
    parser.add_argument(
        "--native-frontend",
        choices=perf.NATIVE_FRONTEND_VARIANTS,
        default="reference",
    )
    parser.add_argument(
        "--forward-pool-ensure",
        choices=perf.FORWARD_POOL_ENSURE_VARIANTS,
        default="always",
        help=(
            "engine-lifetime forward pool guard for the native profile; "
            "the auto reference always uses always"
        ),
    )
    parser.add_argument(
        "--qlen1-inline-plan",
        choices=perf.QLEN1_INLINE_PLAN_VARIANTS,
        default=perf.DEFAULT_QLEN1_INLINE_PLAN,
        help=(
            "engine-lifetime qlen=1 orchestration plan; trusted_native "
            "requires qkv_scatter_inline"
        ),
    )
    parser.add_argument(
        "--decode-fp16-window-blocks",
        type=int,
        default=perf.DEFAULT_DECODE_FP16_WINDOW_BLOCKS,
        help=(
            "bounded decode FP16 history window for the candidate; "
            "the auto reference is pinned to 0"
        ),
    )
    parser.add_argument(
        "--decode-fp16-low-water-blocks",
        type=int,
        default=perf.DEFAULT_DECODE_FP16_LOW_WATER_BLOCKS,
        help=(
            "decode FP16 low-water mark for the candidate; must be 0 when the "
            "window is 0 and no greater than a nonzero window; the auto "
            "reference is pinned to 0"
        ),
    )
    parser.add_argument(
        "--decode-flush-scope",
        choices=perf.DECODE_FLUSH_SCOPES,
        default=perf.DEFAULT_DECODE_FLUSH_SCOPE,
        help=(
            "qlen=1 decode flush coordination scope for the candidate; "
            "the auto reference is pinned to per_row"
        ),
    )
    parser.add_argument(
        "--flush-index-materialization",
        choices=perf.FLUSH_INDEX_MATERIALIZATION_VARIANTS,
        default="per_layer",
    )
    parser.add_argument(
        "--flush-writer",
        choices=perf.FLUSH_WRITER_VARIANTS,
        default="reference",
    )
    parser.add_argument(
        "--prefill-store",
        choices=perf.PREFILL_STORE_VARIANTS,
        default="reference",
    )
    parser.add_argument(
        "--onednn-deterministic",
        type=int,
        choices=(0, 1),
        default=1,
    )
    parser.add_argument(
        "--request-stable-projection-rows",
        type=int,
        choices=(0, 1),
        default=1,
        help="diagnostic projection-row policy; 0 requires runtime-factory mode",
    )
    parser.add_argument(
        "--request-stable-rmsnorm",
        type=int,
        choices=(0, 1),
        default=1,
        help="diagnostic Gemma RMSNorm policy; 0 requires runtime-factory mode",
    )
    parser.add_argument("--native-splits", type=int)
    parser.add_argument(
        "--launcher",
        help="explicit logical launcher name; must match arm/layout/batch",
    )
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=perf.DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--model", default=perf.DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=perf.DEFAULT_MODEL_REVISION)
    parser.add_argument("--served-model", default="sunny-chat")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--config-ref", default="path:/home/jasonbk/.config/nix")
    parser.add_argument(
        "--launcher-mode",
        choices=perf.LAUNCHER_MODES,
        default="immutable",
        help=(
            "immutable keeps variant-specific config apps; runtime-factory "
            "resolves one package-free app and supplies all engine selectors"
        ),
    )
    parser.add_argument(
        "--runtime-cache",
        type=Path,
        default=Path("benchmark-results/kvarn-runtime-cache"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("/var/cache/huggingface"))
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--startup-attempts", type=int, default=2)
    parser.add_argument("--readiness-poll-interval", type=float, default=2.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    parser.add_argument("--benchmark-timeout", type=float, default=7200.0)
    parser.add_argument("--metrics-poll-interval", type=float, default=0.1)
    parser.add_argument(
        "--packaging-repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--config-repo", type=Path, default=Path("/home/jasonbk/.config/nix")
    )
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args(argv)
    try:
        selectors = (
            args.native_layout,
            args.native_kernel_variant,
            args.native_split_policy,
        )
        if args.arm == "candidate" and any(value is None for value in selectors):
            raise perf.RunnerError(
                "candidate profiles require explicit --native-layout, "
                "--native-kernel-variant, and --native-split-policy"
            )
        args.native_layout = args.native_layout or "natural"
        args.native_kernel_variant = (
            args.native_kernel_variant or perf.REFERENCE_NATIVE_KERNEL_VARIANT
        )
        args.native_split_policy = args.native_split_policy or "fixed"
        if (
            args.native_kernel_variant in perf.RUNTIME_FACTORY_ONLY_KERNEL_VARIANTS
            and perf.launcher_mode(args) != "runtime-factory"
        ):
            raise perf.RunnerError(
                f"{args.native_kernel_variant} requires --launcher-mode runtime-factory"
            )
        if args.flush_writer != "reference" and args.native_layout != "xe2_dpas":
            raise perf.RunnerError(
                "native --flush-writer requires --native-layout xe2_dpas"
            )
        args.onednn_deterministic = bool(args.onednn_deterministic)
        args.request_stable_projection_rows = bool(args.request_stable_projection_rows)
        args.request_stable_rmsnorm = bool(args.request_stable_rmsnorm)
        perf.forward_pool_ensure_environment(args)
        perf.qlen1_inline_plan_environment(args)
        perf.decode_fp16_low_water_blocks_environment(args)
        perf.decode_flush_scope_environment(args)
        if perf.launcher_mode(args) != "runtime-factory" and (
            not args.request_stable_projection_rows or not args.request_stable_rmsnorm
        ):
            raise perf.RunnerError(
                "request-stability opt-outs require --launcher-mode runtime-factory"
            )
        args.output_dir = perf.ensure_durable(args.output_dir, allow_tmp=args.allow_tmp)
        args.candidate_env = args.candidate_env.expanduser().resolve()
        args.runtime_cache = args.runtime_cache.expanduser().resolve()
        args.hf_home = args.hf_home.expanduser().resolve()
        args.packaging_repo = args.packaging_repo.expanduser().resolve()
        args.config_repo = args.config_repo.expanduser().resolve()
        if not args.config_ref.startswith("path:"):
            raise perf.RunnerError("--config-ref must be a local path: reference")
        config_ref_path = Path(args.config_ref.removeprefix("path:")).expanduser()
        if config_ref_path.resolve() != args.config_repo:
            raise perf.RunnerError(
                "--config-ref and --config-repo must identify one tree"
            )
        args.config_ref = f"path:{args.config_repo}"
        if not (args.candidate_env / "bin" / "vllm").is_file():
            raise perf.RunnerError("--candidate-env must contain bin/vllm")
        if not (args.candidate_env / "bin" / "python").is_file():
            raise perf.RunnerError(
                "--candidate-env must contain bin/python for XPU proof"
            )
        if args.max_model_len not in {65536, 262144}:
            raise perf.RunnerError("max model length must be 65,536 or 262,144")
        if (
            args.max_model_len == 262144
            and perf.launcher_mode(args) != "runtime-factory"
        ):
            raise perf.RunnerError(
                "262K profiling requires --launcher-mode runtime-factory"
            )
        if args.context < 1 or args.context + args.output_tokens > args.max_model_len:
            raise perf.RunnerError("context plus output tokens exceeds model length")
        if (
            args.output_tokens
            < args.profile_steps + DECODE_SETTLE_STEPS + args.batch + 1
        ):
            raise perf.RunnerError(
                "output tokens must leave room for settled profiled decode steps"
            )
        if not MIN_PROFILE_STEPS <= args.profile_steps <= MAX_PROFILE_STEPS:
            raise perf.RunnerError(
                f"profile steps must be in [{MIN_PROFILE_STEPS}, {MAX_PROFILE_STEPS}]"
            )
        if args.max_num_batched_tokens < 1:
            raise perf.RunnerError("max num batched tokens must be positive")
        if args.arm == "reference" and args.native_layout != "natural":
            raise perf.RunnerError("the auto reference layout must be natural")
        if (
            args.arm == "candidate"
            and perf.launcher_mode(args) == "immutable"
            and (
                args.native_layout != "xe2_dpas"
                or args.native_kernel_variant
                not in perf.IMMUTABLE_QUALIFIED_KERNEL_VARIANTS
                or args.native_split_policy != "b70_q6"
            )
        ):
            raise perf.RunnerError(
                "immutable Round-2 profiling launchers require xe2_dpas, a Q6 "
                "kernel variant, and b70_q6; use --launcher-mode runtime-factory "
                "for other compatible compiled variants"
            )
        if (
            args.native_kernel_variant != perf.REFERENCE_NATIVE_KERNEL_VARIANT
            and args.native_layout != "xe2_dpas"
        ):
            raise perf.RunnerError(
                "non-baseline native kernel variants require --native-layout xe2_dpas"
            )
        if (
            args.arm == "reference"
            and args.native_kernel_variant != perf.REFERENCE_NATIVE_KERNEL_VARIANT
        ):
            raise perf.RunnerError("the auto reference kernel variant must be baseline")
        if (
            args.variant_id is not None
            and VARIANT_ID_PATTERN.fullmatch(args.variant_id) is None
        ):
            raise perf.RunnerError(
                "variant id must be a lowercase slug of at most 192 characters"
            )
        if perf.split_policy.owns_runtime_selection(args.native_split_policy):
            if args.arm != "candidate":
                raise perf.RunnerError("the auto reference split policy must be fixed")
            if args.native_splits is not None:
                raise perf.RunnerError(
                    "--native-splits must be absent with named split policies"
                )
            try:
                perf.split_policy.validate_kernel_compatibility(
                    args.native_split_policy,
                    args.native_kernel_variant,
                    q6_variants=perf.B70_Q6_KERNEL_VARIANTS,
                )
                selected_splits = perf.split_policy.effective_splits(
                    args.native_split_policy,
                    batch=args.batch,
                    context_tokens=args.context,
                )
            except ValueError as exc:
                raise perf.RunnerError(str(exc)) from exc
        else:
            default_splits = (
                perf.REFERENCE_NATIVE_SPLITS
                if args.arm == "reference"
                else perf.DEFAULT_NATIVE_SPLITS[args.batch]
            )
            selected_splits = (
                default_splits if args.native_splits is None else args.native_splits
            )
        if args.arm == "reference" and selected_splits != perf.REFERENCE_NATIVE_SPLITS:
            raise perf.RunnerError("the auto reference native split value must be 1")
        if (
            args.arm == "candidate"
            and selected_splits not in perf.SUPPORTED_NATIVE_SPLITS
        ):
            raise perf.RunnerError(f"unsupported native split count {selected_splits}")
        args.native_splits = (
            {}
            if args.native_split_policy == "b70_q6_v2"
            else {args.batch: selected_splits}
        )
        if (
            args.startup_attempts < 1
            or min(
                args.startup_timeout,
                args.readiness_poll_interval,
                args.shutdown_timeout,
                args.benchmark_timeout,
                args.metrics_poll_interval,
            )
            <= 0
        ):
            raise perf.RunnerError("attempt counts and timeouts must be positive")
    except perf.RunnerError as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.supervisor = perf.ProcessSupervisor()
    args.supervisor.install_signal_handlers()
    try:
        try:
            result = execute(args)
        except (perf.RunnerError, OSError, subprocess.SubprocessError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    finally:
        args.supervisor.signal_all(perf.signal.SIGTERM)
        args.supervisor.restore_signal_handlers()
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
