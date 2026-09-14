"""Protect serving measurements from incomplete streams and MTP chunk bias."""

import io
import json
import threading

import pytest

from scripts import kvarn_lowbit_service as runner


def test_compilation_detection_covers_jit_hooks_and_autotuner_stdout():
    startup = "[jit_monitor.py:85] Kernel JIT monitor activated; mode=warn"
    compile_line = (
        "[jit_monitor.py:146] Triton JIT compilation during inference: kernel."
    )
    autotune_line = "Triton autotuning for function kernel,"
    assert runner.compilation_events(startup) == []
    assert runner.compilation_events(f"{startup}\n{compile_line}\n{autotune_line}") == [
        compile_line,
        autotune_line,
    ]


@pytest.mark.parametrize("draft_tokens", [0, 31, 60])
@pytest.mark.parametrize("k4v4_draft", [False, True])
def test_mtp_activity_requires_two_token_work_but_allows_budget_trimming(
    draft_tokens, k4v4_draft
):
    from scripts.kvarn_lowbit_compare import audit_mtp_activity

    counters = {
        "vllm:spec_decode_num_drafts_total": 31,
        "vllm:spec_decode_num_draft_tokens_total": draft_tokens,
    }
    config = {"method": "mtp", "num_speculative_tokens": 2}
    if k4v4_draft:
        config["kv_cache_dtype"] = "kvarn_k4v4_g128_compact"
    args = ["--speculative-config", json.dumps(config)]
    if draft_tokens == 60:
        audit_mtp_activity(counters, args)
    else:
        with pytest.raises(ValueError, match="missing active two-token MTP"):
            audit_mtp_activity(counters, args)


def _wire(*, complete=True, token_ids=True):
    events = [
        {
            "choices": [
                {"text": "hello", **({"token_ids": [1, 2]} if token_ids else {})}
            ]
        },
        {
            "choices": [
                {"text": " world", "token_ids": [3, 4, 5], "finish_reason": "length"}
            ]
        },
        {
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        },
    ]
    raw = "".join("data: " + json.dumps(e) + "\n" for e in events)
    if complete:
        raw += "data: [DONE]\n"
    return raw.encode()


def _request(tmp_path, monkeypatch, wire):
    monkeypatch.setattr(
        runner.urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(wire)
    )
    clock = iter([0.0, 0.2, 1.0, 1.1, 1.2, 1.3])
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    release = threading.Event()
    release.set()
    return runner.stream_request(
        "http://127.0.0.1:18000",
        {"model": "test", "prompt": [9, 8, 7], "max_tokens": 5},
        tmp_path,
        "case",
        release,
    )


def test_decode_rate_excludes_all_tokens_in_first_mtp_chunk(tmp_path, monkeypatch):
    result = _request(tmp_path, monkeypatch, _wire())
    assert result["token_ids"] == [1, 2, 3, 4, 5]
    assert result["first_chunk_tokens"] == 2
    assert result["ttft_seconds"] == 0.2
    assert result["decode_tokens_per_second"] == pytest.approx(3.75)
    assert result["elapsed_seconds"] == 1.3
    assert (
        runner.perf.sha256_file(tmp_path / "case-sse.jsonl")
        == result["raw_stream_sha256"]
    )


@pytest.mark.parametrize("options", [{"complete": False}, {"token_ids": False}])
def test_incomplete_stream_retains_raw_evidence_and_cannot_pass(
    tmp_path, monkeypatch, options
):
    with pytest.raises(RuntimeError, match="incomplete token or stream evidence"):
        _request(tmp_path, monkeypatch, _wire(**options))
    assert (tmp_path / "case-sse.jsonl").read_text()
    assert not (tmp_path / "case-response.json").exists()


def test_comparer_reconstructs_timing_from_raw_stream(tmp_path, monkeypatch):
    from scripts.kvarn_lowbit_compare import audit_stream

    result = _request(tmp_path, monkeypatch, _wire())
    body = {"model": "test", "prompt": [9, 8, 7], "max_tokens": 5}
    assert audit_stream(tmp_path, "case", body) == result
    result["decode_tokens_per_second"] = 1e10
    (tmp_path / "case-response.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="measurement disagrees"):
        audit_stream(tmp_path, "case", body)


def test_comparer_rejects_changed_stream_bytes(tmp_path, monkeypatch):
    from scripts.kvarn_lowbit_compare import audit_stream

    _request(tmp_path, monkeypatch, _wire())
    with (tmp_path / "case-sse.jsonl").open("a") as raw:
        raw.write("data: [DONE]\n")
    with pytest.raises(ValueError, match="changed artifact"):
        audit_stream(
            tmp_path, "case", {"model": "test", "prompt": [9, 8, 7], "max_tokens": 5}
        )
