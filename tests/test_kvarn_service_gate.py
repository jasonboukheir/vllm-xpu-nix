import importlib.util
import json
import threading
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
            "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n"
        )


def test_repeated_span_gate_detects_three_consecutive_copies():
    span = list(range(16))
    assert MODULE.repeated_span_three_times([99, *span, *span, *span, 100])
    assert not MODULE.repeated_span_three_times(span * 2)


def test_short_period_gate_detects_128_token_collapse():
    assert MODULE.short_period_run([1, 2, 3, 4] * 32)
    assert not MODULE.short_period_run(list(range(128)))


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
        return StreamingResponse([b"data: {}\n"] * 257)

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
