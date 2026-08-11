import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "kvarn_prefix_gate.py"
SPEC = importlib.util.spec_from_file_location("kvarn_prefix_gate", MODULE_PATH)
assert SPEC and SPEC.loader
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_repeat_to_length_covers_boundaries():
    for length in (0, 1, 3, 4, 127, 128, 129, 4096):
        result = gate.repeat_to_length([10, 11, 12], length)
        assert len(result) == length
        assert result == ([10, 11, 12] * ((length + 2) // 3))[:length]


def test_comparable_discards_request_metadata():
    response = {
        "id": "different-each-call",
        "usage": {"prompt_tokens": 9},
        "choices": [
            {
                "token_ids": [1, 2],
                "text": "ok",
                "finish_reason": "length",
                "logprobs": {"token_logprobs": [-0.1, -0.2]},
            }
        ],
    }
    assert gate.comparable(response) == {
        "token_ids": [1, 2],
        "text": "ok",
        "finish_reason": "length",
        "logprobs": {"token_logprobs": [-0.1, -0.2]},
    }


def test_run_case_compares_uncached_and_cached_target(monkeypatch):
    calls = []

    def fake_completion(base_url, model, token_ids):
        calls.append(token_ids)
        return {
            "choices": [
                {
                    "token_ids": [7],
                    "text": "x",
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": ["x"],
                        "text_offset": [0],
                        "token_logprobs": [-0.5],
                        "top_logprobs": [{"x": -0.5}],
                    },
                }
            ]
        }

    monkeypatch.setattr(gate, "completion", fake_completion)
    assert gate.run_case("http://server", "model", [10, 11, 12], 4, 0.125) == {
        "shared_prefix_tokens": 4,
        "decoded_output_identical": True,
        "top_logprob_support_identical": True,
        "max_logprob_delta": 0.0,
    }
    assert len(calls) == 3
    assert calls[0] == calls[2]
    assert calls[0][:-1] == calls[1][:-1]
    assert calls[0][-1] != calls[1][-1]


def test_compare_results_accepts_bounded_logprob_drift():
    left = gate.comparable(
        {
            "choices": [{
                "token_ids": [7], "text": "x", "finish_reason": "length",
                "logprobs": {"tokens": ["x"], "text_offset": [0],
                    "token_logprobs": [-0.5], "top_logprobs": [{"x": -0.5}]},
            }]
        }
    )
    right = {**left, "logprobs": {**left["logprobs"],
        "token_logprobs": [-0.55], "top_logprobs": [{"x": -0.55}]}}
    maximum, support_identical = gate.compare_results(left, right, 0.1)
    assert maximum == pytest.approx(0.05)
    assert support_identical is True


def test_compare_results_rejects_changed_tokens():
    left = {"token_ids": [1], "text": "a", "finish_reason": "length",
        "logprobs": {"tokens": ["a"], "text_offset": [0],
            "token_logprobs": [-0.1], "top_logprobs": [{"a": -0.1}]}}
    with pytest.raises(AssertionError, match="token_ids"):
        gate.compare_results(left, {**left, "token_ids": [2]}, 0.1)
