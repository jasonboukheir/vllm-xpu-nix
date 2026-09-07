import pytest
from test_kvarn_xpu_trace import event

from scripts.kvarn_xpu_trace_stacks import analyze_stacks


def test_wait_source_frames_require_same_thread_and_full_cpu_containment():
    document = {
        "traceEvents": [
            event("model.py(12): forward", "python_function", 0, 100),
            event("wrong_thread.py(2): work", "python_function", 0, 100, tid=9),
            event("wrong_process.py(2): work", "python_function", 0, 100, pid=2),
            event("finished.py(2): work", "python_function", 0, 4),
            event("aten::copy_", "cpu_op", 4, 80, **{"Input Dims": [[16], [16]]}),
            event("zeEventHostSynchronize", "xpu_driver", 5, 70),
        ]
    }
    report = analyze_stacks(document, event_name="zeEventHostSynchronize")
    row = report["events"][0]
    assert row["python_frames_outer_to_inner"] == ["model.py(12): forward"]
    assert row["cpu_operators_outer_to_inner"][0]["input_shapes"] == [[16], [16]]


def test_missing_python_stack_is_explicit_and_longest_events_come_first():
    document = {
        "traceEvents": [
            event("zeEventHostSynchronize", "xpu_driver", 1, 2),
            event("zeEventHostSynchronize", "xpu_driver", 4, 7),
        ]
    }
    report = analyze_stacks(document, event_name="zeEventHostSynchronize", limit=1)
    assert report["matching_event_count"] == 2
    assert report["events"][0]["cpu_inclusive_us"] == 7
    assert report["events"][0]["source_stack_present"] is False


def test_missing_event_or_invalid_limit_rejected():
    with pytest.raises(ValueError, match="no matching CPU event"):
        analyze_stacks({"traceEvents": []}, event_name="missing")
    with pytest.raises(ValueError, match="limit must be positive"):
        analyze_stacks({"traceEvents": []}, event_name="missing", limit=0)
