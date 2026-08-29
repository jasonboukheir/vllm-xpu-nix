import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "kvarn_service_gate.py"
SPEC = importlib.util.spec_from_file_location("kvarn_service_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_metrics_aggregate_requests_and_take_peak_cache_usage():
    text = """
vllm:num_requests_running{engine="0"} 2
vllm:num_requests_running{engine="1"} 1
vllm:num_requests_waiting{engine="0"} 0
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:kv_cache_usage_perc{engine="1"} 0.5
"""
    assert MODULE.parse_metrics(text) == {
        "vllm:num_requests_running": 3,
        "vllm:num_requests_waiting": 0,
        "vllm:kv_cache_usage_perc": 0.5,
    }


def test_metrics_reject_missing_idle_gauges():
    with pytest.raises(ValueError, match="kv_cache_usage_perc"):
        MODULE.parse_metrics(
            "vllm:num_requests_running 0\n"
            "vllm:num_requests_waiting 0\n"
        )


def test_repeated_span_gate_detects_three_consecutive_copies():
    span = list(range(16))
    assert MODULE.repeated_span_three_times([99, *span, *span, *span, 100])
    assert not MODULE.repeated_span_three_times(span * 2)


def test_short_period_gate_detects_128_token_collapse():
    assert MODULE.short_period_run([1, 2, 3, 4] * 32)
    assert not MODULE.short_period_run(list(range(128)))
