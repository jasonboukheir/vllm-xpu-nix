"""Validate attribution with known asynchronous and overlapping timelines."""

import pytest

from scripts.kvarn_prefill_trace import validate_steps
from scripts.kvarn_xpu_trace import analyze_attribution


def test_full_prefill_qualification_rejects_missing_first_chunk_and_wrong_extent():
    def annotation(q, k, prefill=True):
        ctx = f"1(sq{q}sk{k}sqsq0sqsk0)" if prefill else "0(sq0sk0sqsq0sqsk0)"
        gen = "0(sq0sk0sqsq0sqsk0)" if prefill else f"1(sq1sk{k}sqsq1sqsk{k})"
        return {"annotation": f"execute_{q}_context_{ctx}_generation_{gen}"}

    steps = [
        annotation(2048, 2048),
        annotation(2047, 4095),
        annotation(1, 4096, False),
        annotation(1, 4097, False),
    ]
    assert validate_steps(steps, 4095) == 2
    with pytest.raises(ValueError, match="full prefill"):
        validate_steps(steps[1:], 4095)
    with pytest.raises(ValueError, match="chunk lengths"):
        validate_steps(steps, 4096)


def event(name, cat, start, duration, *, pid=1, tid=1, **args):
    return {
        "ph": "X",
        "name": name,
        "cat": cat,
        "ts": start,
        "dur": duration,
        "pid": pid,
        "tid": tid,
        "args": args,
    }


def test_gpu_work_is_owned_by_submitter_not_overlapping_cpu_step():
    trace = {
        "traceEvents": [
            event("execute_first", "user_annotation", 0, 10),
            event("aten::mm", "cpu_op", 1, 5, **{"External id": 7}),
            event("execute_second", "user_annotation", 10, 10),
            event("aten::relu", "cpu_op", 11, 5, **{"External id": 8}),
            event("gemm", "kernel", 15, 10, pid=0, tid=5, **{"External id": 7}),
            event("relu", "kernel", 26, 2, pid=0, tid=5, **{"External id": 8}),
        ]
    }
    result = analyze_attribution(trace)
    assert result["coverage"]["assigned_fraction"] == 1
    assert [step["device_duration_sum_us"] for step in result["steps"]] == [10, 2]
    assert result["steps"][0]["host_start_through_last_completion_us"] == 25


def test_overlap_uses_device_union_and_nested_cpu_self_time():
    trace = {
        "traceEvents": [
            event("execute_first", "user_annotation", 0, 20),
            event("aten::mm", "cpu_op", 1, 10, **{"External id": 7}),
            event("urEnqueue", "xpu_runtime", 2, 3, correlation=17),
            event("zeLaunch", "xpu_driver", 3, 1, correlation=17),
            event("gemm", "kernel", 12, 10, pid=0, tid=5, **{"External id": 7}),
            event(
                "copy", "gpu_memcpy", 16, 10, pid=0, tid=6, correlation=17, bytes=1024
            ),
        ]
    }
    result = analyze_attribution(trace)
    assert result["window"]["device_busy_union_us"] == 14
    assert result["steps"][0]["device_duration_sum_us"] == 20
    assert result["coverage"]["methods"] == {"external_id": 1, "runtime_correlation": 1}
    rows = {row["name"]: row for row in result["cpu_operations"]}
    assert rows["aten::mm"]["cpu_self_us"] == 7
    assert rows["execute_first"]["cpu_self_us"] == 10


def test_missing_and_ambiguous_ids_remain_explicitly_unassigned():
    trace = {
        "traceEvents": [
            event("execute_first", "user_annotation", 0, 10),
            event("aten::mm", "cpu_op", 1, 5, **{"External id": 7}),
            event("aten::mm", "cpu_op", 1, 5, pid=2, **{"External id": 7}),
            event("ambiguous", "kernel", 15, 10, pid=0, tid=5, **{"External id": 7}),
            event("unlinked", "kernel", 26, 2, pid=0, tid=5),
        ]
    }
    result = analyze_attribution(trace)
    assert result["coverage"]["assigned_device_operations"] == 0
    assert result["coverage"]["unassigned_device_operations"] == 2
    assert result["steps"][0]["device_operation_count"] == 0


def test_outside_scope_operations_are_counted_separately():
    trace = {
        "traceEvents": [
            event("execute_first", "user_annotation", 0, 10),
            event("aten::argmax", "cpu_op", 11, 5, **{"External id": 9}),
            event("argmax", "kernel", 20, 2, pid=0, tid=5, **{"External id": 9}),
        ]
    }
    result = analyze_attribution(trace)
    assert result["coverage"]["unresolved_host_origins"] == 0
    assert result["coverage"]["resolved_outside_step_scopes"] == 1
    assert result["coverage"]["outside_step_operation_counts"] == [
        {"kind": "kernel", "name": "argmax", "count": 1}
    ]


def test_conflicting_device_categories_are_rejected():
    trace = {
        "traceEvents": [
            event("execute_first", "user_annotation", 0, 10),
            event("invalid", "kernel, gpu_memcpy", 20, 2),
        ]
    }
    with pytest.raises(ValueError, match="conflicting operation categories"):
        analyze_attribution(trace)


def test_guard_steps_are_excluded_but_full_busy_window_keeps_overlap():
    trace = {
        "traceEvents": [
            event("execute_first", "user_annotation", 0, 10),
            event("aten::first", "cpu_op", 1, 5, **{"External id": 7}),
            event("execute_second", "user_annotation", 10, 10),
            event("aten::second", "cpu_op", 11, 5, **{"External id": 8}),
            event("guard_kernel", "kernel", 15, 10, pid=0, **{"External id": 7}),
            event("selected_kernel", "kernel", 16, 2, pid=0, **{"External id": 8}),
        ]
    }
    result = analyze_attribution(trace, first_step=2, last_step=2)
    assert result["coverage"]["captured_step_count"] == 2
    assert result["coverage"]["selected_step_indices"] == [2]
    assert result["coverage"]["excluded_guard_operation_counts"][0]["count"] == 1
    assert result["coverage"]["excluded_guard_device_operations"] == 1
    assert result["coverage"]["excluded_guard_device_duration_sum_us"] == 10
    assert result["coverage"]["total_device_operations"] == (
        result["coverage"]["assigned_device_operations"]
        + result["coverage"]["unassigned_device_operations"]
        + result["coverage"]["excluded_guard_device_operations"]
    )
    assert result["coverage"]["resolved_host_origin_fraction"] == 1
    assert result["steps"][0]["device_operation_count"] == 1
    assert result["window"]["full_trace_device_busy_union_us"] == 5


def test_guard_step_range_rejects_empty_or_out_of_bounds_selection():
    trace = {
        "traceEvents": [
            event("execute_only", "user_annotation", 0, 10),
            event("kernel", "kernel", 1, 1),
        ]
    }
    with pytest.raises(ValueError, match="invalid or empty"):
        analyze_attribution(trace, first_step=2)

    for first, last in ((True, 1), (1.0, 1), (-1, 1), (2, 1), (1, 2)):
        with pytest.raises(ValueError):
            analyze_attribution(trace, first_step=first, last_step=last)
