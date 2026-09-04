import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace

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
            "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
        )


def test_repeated_span_gate_detects_three_consecutive_copies():
    span = list(range(16))
    assert MODULE.repeated_span_three_times([99, *span, *span, *span, 100])
    assert not MODULE.repeated_span_three_times(span * 2)


def test_short_period_gate_detects_128_token_collapse():
    assert MODULE.short_period_run([1, 2, 3, 4] * 32)
    assert not MODULE.short_period_run(list(range(128)))


def test_completion_retains_raw_repetition_response_and_finding(monkeypatch):
    span = list(range(16))
    raw_response = {
        "id": "completion-id",
        "choices": [
            {
                "finish_reason": "length",
                "text": "raw repeated output",
                "token_ids": span * 3,
            }
        ],
    }
    monkeypatch.setattr(
        MODULE, "http_request", lambda *_args, **_kwargs: json.dumps(raw_response)
    )

    result = MODULE.completion(
        "http://service",
        "model",
        {"id": "repeated", "prompt": "prompt", "max_tokens": 48},
        48,
        30,
    )

    assert result["raw_response"] == raw_response
    assert result["token_ids"] == span * 3
    assert result["quality_findings"] == [
        {
            "kind": "repeated_span",
            "span_tokens": 16,
            "consecutive_copies": 3,
        }
    ]


@pytest.mark.parametrize(
    ("token_ids", "kind"),
    [([], "empty_output"), ([1], "short_output"), ([1, 2, 3], "excess_output")],
)
def test_completion_quality_findings_classify_wrong_output_length(token_ids, kind):
    assert MODULE.completion_quality_findings(token_ids, 2)[0] == {
        "kind": kind,
        "expected_tokens": 2,
        "actual_tokens": len(token_ids),
    }


def test_token_prompt_metadata_records_count_and_hash_without_embedding_ids():
    metadata = MODULE.prompt_metadata([10, 20, 30])
    assert metadata == {
        "prompt_token_count": 3,
        "prompt_token_ids_sha256": (
            "7784d2c72a24031b6f93509c2163a212b3d7388f1ac7513692bd69e742e8b976"
        ),
    }


def test_fixture_files_combine_in_command_line_order_and_preserve_single(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = [
        {"id": "dialogue-127", "prompt": [1], "max_tokens": 1024},
        {"id": "code-4095", "prompt": [2], "max_tokens": 768},
    ]
    second = [{"id": "math-16383", "prompt": [3], "max_tokens": 768}]
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    assert MODULE.load_fixture_files(first_path) == first
    assert MODULE.load_fixture_files([first_path, second_path]) == [*first, *second]


def test_cli_collects_repeated_fixture_paths_in_order(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    args = MODULE.parse_args(
        [
            "--model",
            "model",
            "--fixtures",
            str(first_path),
            "--fixtures",
            str(second_path),
            "--max-tokens",
            "512",
            "--override-max-tokens",
            "--require-duplicate-prompt-isolation",
            "--minimum-output-tokens",
            "512",
            "--output",
            str(tmp_path / "output.json"),
            "--allow-tmp",
        ]
    )

    assert args.fixtures == [first_path, second_path]
    assert args.max_tokens == 512
    assert args.override_max_tokens is True
    assert args.require_duplicate_prompt_isolation is True


def test_combined_fixture_files_reject_duplicate_ids(tmp_path):
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(
        json.dumps([{"id": "duplicate", "prompt": "first"}]),
        encoding="utf-8",
    )
    second_path.write_text(
        json.dumps([{"id": "duplicate", "prompt": "second"}]),
        encoding="utf-8",
    )
    fixtures = MODULE.load_fixture_files([first_path, second_path])

    with pytest.raises(ValueError, match="duplicate fixture ID: duplicate"):
        MODULE.validate_fixtures(
            fixtures,
            concurrency=1,
            default_max_tokens=2048,
            minimum_output_tokens=512,
        )


def test_max_tokens_override_forces_all_mixed_context_fixtures_to_512(tmp_path):
    paths = []
    for fixture_id, prompt_length, max_tokens in (
        ("dialogue-127", 127, 1024),
        ("code-4095", 4095, 768),
        ("math-16383", 16383, 768),
        ("reasoning-65023", 65023, 512),
    ):
        path = tmp_path / f"{fixture_id}.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "id": fixture_id,
                        "prompt": [1] * prompt_length,
                        "max_tokens": max_tokens,
                    }
                ]
            ),
            encoding="utf-8",
        )
        paths.append(path)

    loaded = MODULE.load_fixture_files(paths)
    overridden = MODULE.override_fixture_max_tokens(loaded, 512)
    validated = MODULE.validate_fixtures(
        overridden,
        concurrency=4,
        default_max_tokens=512,
        minimum_output_tokens=512,
    )

    assert [fixture["id"] for fixture in validated] == [
        "dialogue-127",
        "code-4095",
        "math-16383",
        "reasoning-65023",
    ]
    assert [fixture["max_tokens"] for fixture in validated] == [512] * 4
    assert [len(fixture["prompt"]) for fixture in validated] == [
        127,
        4095,
        16383,
        65023,
    ]
    assert [fixture["max_tokens"] for fixture in loaded] == [1024, 768, 768, 512]


def test_duplicate_prompt_isolation_accepts_same_wave_abba_groups():
    fixtures = [
        {"id": "a1", "prompt": [1, 2], "isolation_group": "a"},
        {"id": "b1", "prompt": [3, 4], "isolation_group": "b"},
        {"id": "b2", "prompt": [3, 4], "isolation_group": "b"},
        {"id": "a2", "prompt": [1, 2], "isolation_group": "a"},
    ]

    plan = MODULE.duplicate_prompt_isolation_plan(fixtures, 4, required=True)

    assert plan == [
        {"isolation_group": "a", "fixture_ids": ["a1", "a2"], "wave_index": 0},
        {"isolation_group": "b", "fixture_ids": ["b1", "b2"], "wave_index": 0},
    ]


def test_duplicate_prompt_isolation_rejects_nonidentical_prompt_ids():
    fixtures = [
        {"id": "a1", "prompt": [1, 2], "isolation_group": "a"},
        {"id": "a2", "prompt": [1, 3], "isolation_group": "a"},
    ]

    with pytest.raises(ValueError, match="identical prompt IDs"):
        MODULE.duplicate_prompt_isolation_plan(fixtures, 2, required=True)


def test_duplicate_prompt_isolation_requires_declarations_when_requested():
    fixtures = [
        {"id": "first", "prompt": [1, 2]},
        {"id": "second", "prompt": [3, 4]},
    ]

    with pytest.raises(ValueError, match="needs declared isolation groups"):
        MODULE.duplicate_prompt_isolation_plan(fixtures, 2, required=True)


def test_duplicate_prompt_isolation_rejects_group_split_across_waves():
    fixtures = [
        {"id": "a1", "prompt": [1, 2], "isolation_group": "a"},
        {"id": "unrelated", "prompt": [3, 4]},
        {"id": "a2", "prompt": [1, 2], "isolation_group": "a"},
        {"id": "another", "prompt": [5, 6]},
    ]

    with pytest.raises(ValueError, match="share one concurrent wave"):
        MODULE.duplicate_prompt_isolation_plan(fixtures, 2, required=True)


def test_replay_mismatch_reports_first_divergence_and_total():
    first = [
        {
            "id": "dialogue",
            "token_ids": [10, 20, 30, 40],
            "token_ids_sha256": "expected",
        }
    ]
    replay = [
        {
            "id": "dialogue",
            "token_ids": [10, 20, 31, 41],
            "token_ids_sha256": "actual",
        }
    ]

    assert MODULE.replay_mismatches(first, replay) == [
        {
            "id": "dialogue",
            "expected_sha256": "expected",
            "actual_sha256": "actual",
            "common_prefix_tokens": 2,
            "expected_token_at_divergence": 30,
            "actual_token_at_divergence": 31,
            "differing_positions": 2,
        }
    ]
    assert MODULE.replay_mismatches(first, first) == []


class StreamingResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.lines)


def test_cancellation_keeps_full_generation_budget(monkeypatch):
    captured = {}

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        chunk = b'data: {"choices":[{"token_ids":[7]}]}\n'
        return StreamingResponse([chunk] * 257)

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", urlopen)

    observed = MODULE.cancel_stream(
        "http://service",
        "model",
        {"id": "fixture", "prompt": "prompt", "max_tokens": 2048},
        2048,
        30,
        257,
    )

    assert observed == 257
    assert captured["payload"]["max_tokens"] == 2048
    assert captured["payload"]["return_token_ids"] is True
    assert captured["timeout"] == 30


def test_cancellation_checkpoint_must_leave_generation_work():
    with pytest.raises(ValueError, match="leave tokens remaining"):
        MODULE.cancel_stream(
            "http://service",
            "model",
            {"id": "fixture", "prompt": "prompt", "max_tokens": 2048},
            2048,
            30,
            2048,
        )


def test_b4_waves_cover_every_fixture_but_b1_keeps_one_probe():
    fixtures = [{"id": str(index)} for index in range(5)]

    assert [
        [fixture["id"] for fixture in wave]
        for wave in MODULE.concurrent_fixture_waves(fixtures, 4)
    ] == [["0", "1", "2", "3"], ["4"]]
    assert MODULE.concurrent_fixture_waves(fixtures, 1) == [[fixtures[0]]]


def test_concurrent_wave_records_observed_overlap(monkeypatch):
    finish = threading.Event()

    def completion(
        _base_url,
        _model,
        fixture,
        _default_max_tokens,
        _timeout,
    ):
        assert finish.wait(timeout=1)
        return {"id": fixture["id"], "token_ids": [int(fixture["id"])]}

    def http_request(url, timeout):
        assert url == "http://service/metrics"
        assert timeout == 10.0
        finish.set()
        return """
vllm:num_requests_running 2
vllm:num_requests_waiting 0
vllm:kv_cache_usage_perc 0.25
"""

    monkeypatch.setattr(MODULE, "completion", completion)
    monkeypatch.setattr(MODULE, "http_request", http_request)

    results, evidence = MODULE.run_concurrent_wave(
        "http://service",
        "model",
        [{"id": "0"}, {"id": "1"}],
        2048,
        4,
        30,
        0.001,
    )

    assert [result["id"] for result in results] == ["0", "1"]
    assert evidence["fixture_ids"] == ["0", "1"]
    assert evidence["required_running"] == 2
    assert evidence["peak_running"] == 2
    assert evidence["metrics_samples"] == 1
    assert evidence["required_overlap_observed"] is True
    assert evidence["quality_failures"] == []


def _gate_args(tmp_path, fixtures):
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(json.dumps(fixtures), encoding="utf-8")
    return SimpleNamespace(
        base_url="http://service",
        model="model",
        fixtures=[fixture_path],
        output=tmp_path / "output.json",
        provenance=None,
        max_tokens=48,
        override_max_tokens=False,
        minimum_output_tokens=48,
        concurrency=len(fixtures),
        cancel_after_events=0,
        metrics_poll_interval=0.001,
        timeout=30,
        idle_timeout=30,
    )


def _completion_result(fixture_id, *, findings=None, raw_marker="isolated"):
    token_ids = list(range(48))
    return {
        "id": fixture_id,
        "prompt": "prompt",
        "max_tokens": 48,
        "elapsed_seconds": 0.1,
        "finish_reason": "length",
        "text": raw_marker,
        "token_ids": token_ids,
        "token_ids_sha256": f"sha-{fixture_id}",
        "quality_findings": findings or [],
        "raw_response": {"marker": raw_marker, "fixture_id": fixture_id},
    }


def test_concurrent_quality_failure_persists_all_peer_results_and_wave(
    monkeypatch, tmp_path
):
    args = _gate_args(
        tmp_path,
        [
            {"id": "good", "prompt": "prompt", "max_tokens": 48},
            {"id": "repeated", "prompt": "prompt", "max_tokens": 48},
        ],
    )
    monkeypatch.setattr(MODULE, "wait_for_idle", lambda *_args: {})
    monkeypatch.setattr(
        MODULE,
        "completion",
        lambda _base, _model, fixture, _max, _timeout: _completion_result(
            fixture["id"]
        ),
    )
    repeated_finding = {
        "kind": "repeated_span",
        "span_tokens": 16,
        "consecutive_copies": 3,
    }
    peer_results = [
        _completion_result("good", raw_marker="concurrent-good"),
        _completion_result(
            "repeated",
            findings=[repeated_finding],
            raw_marker="concurrent-repeated",
        ),
    ]
    quality_failures = MODULE.result_quality_failures(peer_results)
    wave_evidence = {
        "fixture_ids": ["good", "repeated"],
        "required_running": 2,
        "peak_running": 2,
        "peak_metrics": {"vllm:num_requests_running": 2},
        "metrics_samples": 1,
        "required_overlap_observed": True,
        "elapsed_seconds": 0.2,
        "quality_failures": quality_failures,
    }
    monkeypatch.setattr(
        MODULE,
        "run_concurrent_wave",
        lambda *_args: (peer_results, wave_evidence),
    )

    with pytest.raises(AssertionError, match="concurrent.*repeated: repeated_span"):
        MODULE.run(args)

    document = json.loads(args.output.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["phase"] == "concurrent_quality"
    assert document["concurrent"] == peer_results
    assert document["concurrent_waves"] == [wave_evidence]
    assert document["quality_failures"] == quality_failures
    assert [result["raw_response"]["marker"] for result in document["concurrent"]] == [
        "concurrent-good",
        "concurrent-repeated",
    ]


def test_isolated_quality_failure_persists_offending_result(monkeypatch, tmp_path):
    args = _gate_args(
        tmp_path,
        [{"id": "empty", "prompt": "prompt", "max_tokens": 48}],
    )
    empty_finding = {
        "kind": "empty_output",
        "expected_tokens": 48,
        "actual_tokens": 0,
    }
    result = _completion_result(
        "empty", findings=[empty_finding], raw_marker="isolated-empty"
    )
    result["token_ids"] = []
    monkeypatch.setattr(MODULE, "wait_for_idle", lambda *_args: {})
    monkeypatch.setattr(MODULE, "completion", lambda *_args: result)

    with pytest.raises(AssertionError, match="isolated_first.*empty_output"):
        MODULE.run(args)

    document = json.loads(args.output.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["phase"] == "isolated_first_quality"
    assert document["isolated_first"] == [result]
    assert document["quality_failures"] == [
        {"id": "empty", "findings": [empty_finding]}
    ]


def test_duplicate_prompt_divergence_fails_separate_with_raw_wave_evidence(
    monkeypatch, tmp_path
):
    fixtures = [
        {
            "id": "a1",
            "prompt": [1, 2],
            "max_tokens": 48,
            "isolation_group": "a",
        },
        {
            "id": "b1",
            "prompt": [3, 4],
            "max_tokens": 48,
            "isolation_group": "b",
        },
        {
            "id": "b2",
            "prompt": [3, 4],
            "max_tokens": 48,
            "isolation_group": "b",
        },
        {
            "id": "a2",
            "prompt": [1, 2],
            "max_tokens": 48,
            "isolation_group": "a",
        },
    ]
    args = _gate_args(tmp_path, fixtures)
    args.require_duplicate_prompt_isolation = True
    monkeypatch.setattr(MODULE, "wait_for_idle", lambda *_args: {})
    monkeypatch.setattr(
        MODULE,
        "completion",
        lambda _base, _model, fixture, _max, _timeout: _completion_result(
            fixture["id"]
        ),
    )
    peer_results = [
        _completion_result("a1", raw_marker="wave-a1"),
        _completion_result("b1", raw_marker="wave-b1"),
        _completion_result("b2", raw_marker="wave-b2"),
        _completion_result("a2", raw_marker="wave-a2"),
    ]
    peer_results[-1]["token_ids"] = [999, *list(range(1, 48))]
    peer_results[-1]["token_ids_sha256"] = "sha-a2-diverged"
    wave_evidence = {
        "fixture_ids": ["a1", "b1", "b2", "a2"],
        "required_running": 4,
        "peak_running": 4,
        "peak_metrics": {"vllm:num_requests_running": 4},
        "metrics_samples": 1,
        "required_overlap_observed": True,
        "elapsed_seconds": 0.2,
        "quality_failures": [],
    }
    monkeypatch.setattr(
        MODULE,
        "run_concurrent_wave",
        lambda *_args: (peer_results, wave_evidence),
    )

    with pytest.raises(
        AssertionError, match="within-wave duplicate-prompt isolation failed: a"
    ):
        MODULE.run(args)

    document = json.loads(args.output.read_text(encoding="utf-8"))
    isolation = document["duplicate_prompt_isolation"]
    assert document["schema_version"] == 3
    assert document["status"] == "failed"
    assert document["phase"] == "duplicate_prompt_isolation"
    assert isolation["status"] == "failed"
    assert isolation["within_wave_only"] is True
    assert [group["bit_identical"] for group in isolation["groups"]] == [
        False,
        True,
    ]
    assert isolation["failures"] == [
        {
            "kind": "duplicate_prompt_output_mismatch",
            "isolation_group": "a",
            "fixture_ids": ["a1", "a2"],
            "wave_index": 0,
        }
    ]
    assert document["concurrent"] == peer_results
    assert [result["raw_response"]["marker"] for result in document["concurrent"]] == [
        "wave-a1",
        "wave-b1",
        "wave-b2",
        "wave-a2",
    ]
    assert all(not result["quality_findings"] for result in document["concurrent"])
