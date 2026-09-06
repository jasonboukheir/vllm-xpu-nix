"""CPU checks for fixture and evidence integrity, not model accuracy."""

import io
import json

import pytest

from scripts import kvarn_vision_compare as comparison
from scripts import kvarn_vision_run as vision


def test_vision_harness_does_not_override_dtype_selected_kvarn_defaults():
    env = vision.runtime_environment()
    assert env == {
        "CCL_ATL_TRANSPORT": "ofi",
        "CCL_LOG_LEVEL": "warn",
        "CCL_PROCESS_LAUNCHER": "none",
        "CCL_ZE_IPC_EXCHANGE": "sockets",
    }
    assert all(not key.startswith(("KVARN_", "VLLM_")) for key in env)


def test_changed_image_control_has_identical_question_and_distinct_pixels(tmp_path):
    cases = vision.fixtures(tmp_path / "images")
    first = cases[0]["messages"][0]["content"]
    changed = cases[1]["messages"][0]["content"]
    assert first[1] == changed[1]
    assert first[0] != changed[0]
    assert (tmp_path / "images/a.png").read_bytes() != (
        tmp_path / "images/b.png"
    ).read_bytes()
    repeated = vision.fixtures(tmp_path / "repeat")
    assert repeated == cases


@pytest.mark.parametrize(
    "done,usage,valid", [(True, True, True), (False, True, False), (True, False, False)]
)
def test_stream_requires_terminal_marker_and_token_usage(
    tmp_path, monkeypatch, done, usage, valid
):
    events = [
        {"choices": [{"delta": {"content": "42"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    if usage:
        events.append(
            {"choices": [], "usage": {"prompt_tokens": 26, "completion_tokens": 3}}
        )
    payload = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    if done:
        payload += "data: [DONE]\n\n"
    monkeypatch.setattr(
        vision.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(payload.encode())
    )
    case = {"id": "text", "messages": [], "expected_terms": ["42"]}
    if valid:
        result = vision.request_case("http://127.0.0.1:8017", case, tmp_path)
        assert result["stream_valid"] and result["term_check"]
    else:
        with pytest.raises(ValueError, match="incomplete or invalid stream"):
            vision.request_case("http://127.0.0.1:8017", case, tmp_path)
    assert (tmp_path / "text-sse.jsonl").is_file()


def test_long_context_rejects_image_inside_uncompressed_sink(tmp_path, monkeypatch):
    cases = vision.fixtures(tmp_path / "images")
    tokens = [0] * 6143
    tokens[20] = 248056
    monkeypatch.setattr(
        vision, "tokenize_case", lambda *a: {"count": 6143, "tokens": tokens}
    )
    with pytest.raises(ValueError, match="beyond sink"):
        vision.long_case("http://127.0.0.1:8017", cases[0], tmp_path)


def test_comparison_rejects_changed_workload_bytes(tmp_path):
    path = tmp_path / "workload.json"
    path.write_text('{"image":"a"}')
    digest = vision.perf.sha256_file(path)
    path.write_text('{"image":"b"}')
    with pytest.raises(ValueError, match="changed artifact"):
        comparison.check_hash(path, digest)


def test_argument_matching_ignores_only_cache_dtype():
    auto = ["vllm", "--kv-cache-dtype", "auto", "--max-model-len", "8192"]
    kvarn = [
        "vllm",
        "--kv-cache-dtype",
        vision.perf.COMPACT_DTYPE,
        "--max-model-len",
        "8192",
    ]
    assert comparison.canonical_argv(auto) == comparison.canonical_argv(kvarn)
    kvarn[-1] = "16384"
    assert comparison.canonical_argv(auto) != comparison.canonical_argv(kvarn)


def test_memory_summary_rejects_shared_buffers_that_could_double_count(tmp_path):
    sample = {
        "seconds": 1,
        "clients": {
            "device:1": {
                "fields": {
                    "drm-pdev": "device",
                    "drm-resident-vram0": "1024 KiB",
                    "drm-shared-vram0": "512 KiB",
                }
            }
        },
        "errors": [],
    }
    (tmp_path / "memory-fdinfo.jsonl").write_text(json.dumps(sample) + "\n")
    with pytest.raises(ValueError, match="de-duplicated"):
        comparison.memory_summary(tmp_path)
