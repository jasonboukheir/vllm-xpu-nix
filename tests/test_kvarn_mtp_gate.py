import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "kvarn_mtp_gate.py"
SPEC = importlib.util.spec_from_file_location("kvarn_mtp_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


METRICS = """
vllm:spec_decode_num_drafts_total{model_name="m"} 10
vllm:spec_decode_num_draft_tokens_total{model_name="m"} 20
vllm:spec_decode_num_accepted_tokens_total{model_name="m"} 12
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="0"} 8
vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="1"} 5
"""


def test_parse_and_infer_all_three_qlen3_acceptance_lengths():
    metrics = gate.parse_metrics(METRICS)
    assert metrics == gate.Metrics(10, 20, 12, (8, 5))
    assert gate.acceptance_lengths(metrics) == {1: 2, 2: 3, 3: 5}


def test_delta_handles_cumulative_prometheus_counters():
    before = gate.Metrics(10, 20, 12, (8, 5))
    after = gate.Metrics(14, 28, 17, (11, 7))
    assert gate.delta(before, after) == gate.Metrics(4, 8, 5, (3, 2))


def test_run_requires_two_drafts_and_every_acceptance_length(monkeypatch):
    snapshots = iter([gate.Metrics(), gate.Metrics(9, 18, 9, (6, 3))])
    monkeypatch.setattr(gate, "parse_metrics", lambda _text: next(snapshots))
    monkeypatch.setattr(gate, "request", lambda _url, payload=None: "metrics")
    monkeypatch.setattr(gate, "completion", lambda *_args: [[1, 2, 3]])
    monkeypatch.setattr(gate, "stream_step_lengths", lambda *_args: [1, 1, 2])
    result = gate.run("http://server", "model", [[7]])
    assert result["qlen"] == 3
    assert result["completion_token_ids"] == [[1, 2, 3]]
    assert len(result["completion_token_ids_sha256"]) == 64
    assert result["acceptance_lengths"] == {1: 3, 2: 3, 3: 3}
    assert result["consecutive_rejection_recovered"] is True


def test_run_rejects_missing_intermediate_acceptance_length(monkeypatch):
    snapshots = iter([gate.Metrics(), gate.Metrics(4, 8, 4, (2, 2))])
    monkeypatch.setattr(gate, "parse_metrics", lambda _text: next(snapshots))
    monkeypatch.setattr(gate, "request", lambda _url, payload=None: "metrics")
    monkeypatch.setattr(gate, "completion", lambda *_args: [[1]])
    monkeypatch.setattr(gate, "stream_step_lengths", lambda *_args: [1, 1, 3])
    with pytest.raises(AssertionError, match=r"acceptance lengths \[2\]"):
        gate.run("http://server", "model", [[7]])


@pytest.mark.parametrize("lengths", [[1, 2, 3], [1, 1], [3, 2, 1, 1]])
def test_rejection_trace_requires_adjacent_rejection_then_recovery(lengths):
    with pytest.raises(AssertionError):
        gate.validate_rejection_trace(lengths)


def test_rejection_trace_accepts_consecutive_rejection_and_recovery():
    gate.validate_rejection_trace([3, 1, 1, 2, 3])
