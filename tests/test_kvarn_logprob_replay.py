import copy
import json
from types import SimpleNamespace

import pytest

from scripts import kvarn_logprob_replay as MODULE


def _record(
    fixture_id: str,
    *,
    token_ids: list[int] | None = None,
    token_logprobs: list[float] | None = None,
    top_logprobs: list[dict[str, float]] | None = None,
):
    token_ids = token_ids or [10, 20, 30]
    token_logprobs = token_logprobs or [-0.1, -0.2, -0.3]
    top_logprobs = top_logprobs or [
        {"a": -0.1},
        {"b": -0.2},
        {"c": -0.3},
    ]
    return {
        "id": fixture_id,
        "token_ids": token_ids,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "token_ids_sha256": f"tokens-{fixture_id}-{token_ids}",
        "logprobs_sha256": f"logprobs-{fixture_id}-{token_logprobs}",
    }


def test_completion_requests_greedy_top5_logprobs_and_normalizes(monkeypatch):
    observed = {}

    def fake_http_request(url, payload, timeout):
        observed.update(url=url, payload=payload, timeout=timeout)
        return json.dumps(
            {
                "id": "completion-id",
                "model": "served-model",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                "choices": [
                    {
                        "finish_reason": "length",
                        "text": "ab",
                        "prompt_token_ids": [1, 2, 3],
                        "token_ids": [11, 12],
                        "logprobs": {
                            "tokens": ["token_id:11", "token_id:12"],
                            "token_logprobs": [-0.125, -0.25],
                            "top_logprobs": [
                                {"token_id:11": -0.125, "token_id:99": -2.0},
                                {"token_id:12": -0.25, "token_id:98": -3.0},
                            ],
                            "text_offset": [0, 1],
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr(MODULE, "http_request", fake_http_request)
    result = MODULE.completion_with_logprobs(
        "http://service",
        "served-model",
        {"id": "fixture", "prompt": "prompt", "max_tokens": 2},
        timeout=42,
    )

    assert observed == {
        "url": "http://service/v1/completions",
        "payload": {
            "model": "served-model",
            "prompt": "prompt",
            "max_tokens": 2,
            "temperature": 0,
            "ignore_eos": True,
            "echo": False,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": 5,
        },
        "timeout": 42,
    }
    assert result["token_ids"] == [11, 12]
    assert result["prompt_token_ids"] == [1, 2, 3]
    assert result["tokens"] == ["token_id:11", "token_id:12"]
    assert result["token_logprobs"] == [-0.125, -0.25]
    assert result["top_logprobs"] == [
        {"token_id:11": -0.125, "token_id:99": -2.0},
        {"token_id:12": -0.25, "token_id:98": -3.0},
    ]
    assert result["response_metadata"]["usage"]["completion_tokens"] == 2


def test_comparison_finds_float_divergence_before_token_ids_split():
    expected = _record(
        "math",
        top_logprobs=[{"a": -0.1}, {"b": -0.2}, {"c": -0.3}],
    )
    actual = _record(
        "math",
        token_logprobs=[-0.1, -0.2001, -0.3],
        top_logprobs=[{"a": -0.1}, {"b": -0.2}, {"c": -0.301}],
    )

    comparison = MODULE.compare_fixture_replay(expected, actual)

    assert comparison["first_divergence_index"] == 1
    assert comparison["first_logprob_divergence_index"] == 1
    assert comparison["logprob_diverges_before_token_ids"]
    assert comparison["first_token_id_divergence"] is None
    assert comparison["first_selected_logprob_divergence"] == {
        "index": 1,
        "expected": -0.2,
        "actual": -0.2001,
    }
    assert comparison["first_top_logprobs_divergence"] == {
        "index": 2,
        "expected": {"c": -0.3},
        "actual": {"c": -0.301},
    }


def test_max_tokens_override_wins_and_omission_preserves_fixture(tmp_path):
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        json.dumps(
            [
                {"id": "fixture-default", "prompt": "a", "max_tokens": 17},
                {"id": "script-default", "prompt": "b"},
            ]
        ),
        encoding="utf-8",
    )

    unchanged = MODULE.load_fixtures(fixtures_path, None)
    overridden = MODULE.load_fixtures(fixtures_path, 384)

    assert [fixture["max_tokens"] for fixture in unchanged] == [17, 2048]
    assert [fixture["max_tokens"] for fixture in overridden] == [384, 384]
    with pytest.raises(ValueError, match="max_tokens override must be positive"):
        MODULE.load_fixtures(fixtures_path, 0)


def test_run_checkpoints_each_response_and_reports_divergence(monkeypatch, tmp_path):
    fixtures_path = tmp_path / "fixtures.json"
    fixtures_path.write_text(
        json.dumps([{"id": "math", "prompt": "2+2", "max_tokens": 3}]),
        encoding="utf-8",
    )
    responses = [
        _record("math"),
        _record("math", token_logprobs=[-0.1, -0.2001, -0.3]),
    ]
    checkpoints = []

    def fake_completion(*_args, **_kwargs):
        return responses.pop(0)

    def capture_checkpoint(_path, document):
        checkpoints.append(copy.deepcopy(document))

    monkeypatch.setattr(MODULE, "completion_with_logprobs", fake_completion)
    monkeypatch.setattr(MODULE, "write_json_atomic", capture_checkpoint)
    args = SimpleNamespace(
        fixtures=fixtures_path,
        max_tokens=None,
        fixture_id=None,
        base_url="http://service",
        model="served-model",
        timeout=42,
        output=tmp_path / "diagnostic.json",
    )

    result = MODULE.run(args)

    assert [
        (item["phase"], len(item["first"]), len(item["replay"])) for item in checkpoints
    ] == [
        ("first", 0, 0),
        ("first", 1, 0),
        ("replay", 1, 0),
        ("replay", 1, 1),
        ("complete", 1, 1),
    ]
    assert result["status"] == "diverged"
    assert result["divergences"][0]["first_logprob_divergence_index"] == 1
