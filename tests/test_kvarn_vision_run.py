"""CPU checks for fixture and evidence integrity, not model accuracy."""

import io
import json
import threading
import time
from pathlib import Path

import pytest

from scripts import kvarn_mtp_compare as mtp
from scripts import kvarn_prefill_compare as prefill
from scripts import kvarn_vision_compare as comparison
from scripts import kvarn_vision_run as vision


@pytest.mark.parametrize("operation", ["check", "finish"])
def test_memory_sampler_finish_reports_worker_failure(tmp_path, monkeypatch, operation):
    failure = OSError("unexpected proc read failure")

    def failed_scan(_group):
        raise failure

    monkeypatch.setattr(vision.perf, "_process_group_members", failed_scan)
    monkeypatch.setattr(threading, "excepthook", lambda _args: None)
    sampler = vision.MemorySampler(77, tmp_path / "memory.jsonl")
    sampler.start()
    sampler.thread.join(timeout=2)
    assert not sampler.thread.is_alive()
    with pytest.raises(RuntimeError, match="memory sampler failed") as error:
        getattr(sampler, operation)()
    assert error.value.__cause__ is failure


def test_memory_sampler_normal_stop_preserves_collected_sample(tmp_path, monkeypatch):
    sampled = threading.Event()

    def empty_group(_group):
        sampled.set()
        return []

    monkeypatch.setattr(vision.perf, "_process_group_members", empty_group)
    output = tmp_path / "memory.jsonl"
    sampler = vision.MemorySampler(77, output)
    started = time.time()
    sampler.start()
    assert sampled.wait(timeout=2)
    sampler.check()
    sampler.finish()
    finished = time.time()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows and rows[0]["clients"] == {} and rows[0]["errors"] == []
    assert all(started <= row["sampled_unix"] <= finished for row in rows)
    assert not sampler.thread.is_alive()


@pytest.mark.parametrize(
    "timestamps,window,error",
    [
        ([99, 100, 101, 102], (100, 102), None),
        ([99, 100, 101], (100, 120), "does not cover"),
        ([101, 102], (100, 102), "does not cover"),
        ([99, 100, 110, 111], (100, 111), "does not cover"),
        ([99, None, 101], (100, 101), "lack finite"),
        ([99, float("nan"), 101], (100, 101), "lack finite"),
        ([99, 101, 100], (100, 101), "non-monotonic"),
    ],
)
def test_memory_review_requires_complete_time_coverage(
    tmp_path, timestamps, window, error
):
    samples = [
        {
            "sampled_unix": timestamp,
            "seconds": index * 0.5,
            "errors": [],
            "clients": {
                "device:1": {
                    "pid": 77,
                    "fields": {
                        "drm-pdev": "device",
                        "drm-resident-vram0": "1024 KiB",
                        "drm-total-vram0": "1024 KiB",
                        "drm-shared-vram0": "0",
                    },
                },
            },
        }
        for index, timestamp in enumerate(timestamps)
    ]
    (tmp_path / "memory-fdinfo.jsonl").write_text(
        "".join(json.dumps(sample) + "\n" for sample in samples)
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            comparison.memory_summary(tmp_path, required_window=window)
    else:
        result = comparison.memory_summary(tmp_path, required_window=window)
        assert (
            result["peak_bytes_by_device_and_field"]["device/drm-resident-vram0"]
            == 1024**2
        )
        assert result["temporal_coverage"]["required_window_unix"] == list(window)
        assert result["temporal_coverage"]["largest_sample_gap_seconds"] == 1


def test_profile_scope_environment_is_captured_without_unrelated_values(monkeypatch):
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (
            b"VLLM_CUSTOM_SCOPES_FOR_PROFILING=1\0UNRELATED_PRIVATE_VALUE=hidden\0"
        ),
    )
    captured = vision.perf._process_environment(123)
    assert captured["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] == "1"
    assert "UNRELATED_PRIVATE_VALUE" not in captured


def test_mtp_uses_one_bundled_draft_token_and_off_preserves_baseline():
    assert vision.speculative_config(False) is None
    assert vision.speculative_config(True) == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }
    assert vision.speculative_config(2) == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }
    with pytest.raises(ValueError):
        vision.speculative_config(3)


def test_performance_suite_fixes_context_caps_repetitions_and_input_identity(
    tmp_path, monkeypatch
):
    image = vision.fixtures(tmp_path / "images")[0]
    long = {**image, "id": "long-image-a", "coverage": {"prompt_tokens": 6143}}
    (tmp_path / "long-image-a-tokenize.json").write_text('{"count":6143}')
    monkeypatch.setattr(vision, "long_case", lambda *args, **kwargs: long)
    monkeypatch.setattr(vision, "tokenize_case", lambda *args: {"count": 4096})
    cases = vision.performance_cases("http://unused", image, tmp_path)
    warmups = [case for case in cases if case["phase"] == "warmup"]
    measured = [case for case in cases if case["phase"] != "warmup"]
    assert warmups and measured
    for c in cases:
        assert c["generation"] == {
            "max_tokens": 256,
            "ignore_eos": True,
            "return_token_ids": True,
        }
    for warmup in warmups:
        assert any(case["messages"] == warmup["messages"] for case in measured)
    for case in measured:
        assert any(case["messages"] == warmup["messages"] for warmup in warmups)


@pytest.mark.parametrize("profiled", [False, True])
@pytest.mark.parametrize("budget", [2048, 4096, 8192])
def test_prefill_exact_lengths_and_warmup_bracket(
    tmp_path, monkeypatch, profiled, budget
):
    def tokenize(_url, case):
        count = 64 + case["messages"][0]["content"].count(" x")
        return {"count": count, "tokens": [42] * count}

    monkeypatch.setattr(vision, "tokenize_case", tokenize)
    cases = vision.prefill_cases(
        "unused", tmp_path, 65023, 512, 3, profiled=profiled, chunk_budget=budget
    )
    assert len(cases) == 4
    assert [c["phase"] for c in cases] == [
        "warmup",
        "performance",
        "profiled-diagnostic" if profiled else "performance",
        "performance",
    ]
    for case in cases:
        assert case["messages"] == cases[0]["messages"]
        assert case["coverage"]["max_num_batched_tokens"] == budget
        assert (
            case["coverage"]["expected_chunk_count"] == (65023 + budget - 1) // budget
        )
        assert case["generation"]["max_tokens"] == 512
        assert case["generation"]["ignore_eos"]
        evidence = json.loads((tmp_path / f"{case['id']}-tokenize.json").read_text())
        assert evidence["count"] == len(evidence["tokens"]) == 65023


def test_prefill_rejects_tokenizer_count_disagreement(tmp_path, monkeypatch):
    monkeypatch.setattr(vision, "tokenize_case", lambda *a: {"count": 64, "tokens": []})
    with pytest.raises(ValueError, match="disagrees"):
        vision.prefill_cases("unused", tmp_path, 64, 512, 3)


@pytest.mark.parametrize("wrong_dtype", [True, False])
def test_prefill_service_comparison_rejects_wrong_arm_or_profiled_run(
    monkeypatch, wrong_dtype
):
    def audit(path):
        dtype = (
            "auto" if wrong_dtype or path.name == "auto" else vision.perf.COMPACT_DTYPE
        )
        return {
            "actual_argv": ["vllm", "--kv-cache-dtype", dtype],
            "suite": "text-prefill-profile-v1",
            "profiler_config": {},
        }, []

    monkeypatch.setattr(prefill.vision, "audit", audit)
    with pytest.raises(ValueError, match="auto then|profiler-off"):
        prefill.compare([(Path("auto"), Path("kvarn"))])


@pytest.mark.parametrize("bundled", [False, True])
def test_prefill_tail_latency_requires_individual_token_arrivals(tmp_path, bundled):
    arrivals = (
        [(1.0, [1]), (1.1, [2, 3])] if bundled else [(1.0, [1]), (1.1, [2]), (1.3, [3])]
    )
    (tmp_path / "case-sse.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "seconds": ts,
                    "line": "data: " + json.dumps({"choices": [{"token_ids": ids}]}),
                }
            )
            for ts, ids in arrivals
        )
    )
    result = {"id": "case", "usage": {"completion_tokens": 3}}
    if bundled:
        with pytest.raises(ValueError, match="bundled"):
            prefill.token_timing(tmp_path, result)
    else:
        timing = prefill.token_timing(tmp_path, result)
        assert timing["inter_token_seconds"]["p99"] == pytest.approx(0.2)
        assert timing["decode_tokens_per_second"] == pytest.approx(2 / 0.3)


def test_mtp_comparison_normalizes_only_one_token_speculation():
    base = ["vllm", "--kv-cache-dtype", "auto"]
    on = base + ["--speculative-config", json.dumps(vision.speculative_config(True))]
    assert mtp.without_mtp(on, enabled=True) == base
    assert mtp.without_mtp(base, enabled=False) == base
    with pytest.raises(ValueError):
        mtp.without_mtp(on, enabled=False)
    on[-1] = '{"method":"mtp","num_speculative_tokens":2}'
    with pytest.raises(ValueError):
        mtp.without_mtp(on, enabled=True)
    assert mtp.without_mtp(on, enabled=2) == base


@pytest.mark.parametrize("clipped", [0, 2])
def test_overhead_model_uses_unconditional_acceptance_and_separates_cap(clipped):
    delivered = 23 - clipped
    base = [
        {"delivered_decode_tokens": delivered, "generation_seconds": delivered / 10}
    ]
    on = [
        {
            "delivered_decode_tokens": delivered,
            "generation_seconds": 1.5,
            "speculative_counter_delta": {
                "vllm:spec_decode_num_drafts_total": 10,
                "vllm:spec_decode_num_draft_tokens_total": 20,
                "vllm:spec_decode_num_accepted_tokens_total": 13,
                "vllm:spec_decode_num_accepted_tokens_per_pos_total/position=0": 8,
                "vllm:spec_decode_num_accepted_tokens_per_pos_total/position=1": 5,
            },
        }
    ]
    result = mtp.overhead_model(base, on, 2)
    assert result["acceptance_by_position_unconditional"] == [0.8, 0.5]
    assert result["raw_ideal_decode_tokens_per_second"] == pytest.approx(23)
    assert result["cap_adjusted_ideal_decode_tokens_per_second"] == pytest.approx(
        delivered
    )
    assert result["output_cap_clipped_tokens"] == clipped
    assert result["efficiency"] == pytest.approx(2 / 3)
    assert result["effective_extra_ms_per_verification_step"] == pytest.approx(50)
    on[0]["speculative_counter_delta"][
        "vllm:spec_decode_num_accepted_tokens_total"
    ] += 1
    with pytest.raises(ValueError, match="accounting"):
        mtp.overhead_model(base, on, 2)


def test_speculative_counter_parser_retains_positions_and_ignores_created(tmp_path):
    path = tmp_path / "metrics.txt"
    path.write_text(
        'vllm:spec_decode_num_drafts_total{engine="0"} 10.0\n'
        'vllm:spec_decode_num_drafts_created{engine="0"} 123.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 8.0\n'
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 5.0\n'
    )
    assert mtp.speculative_counters(path) == {
        "vllm:spec_decode_num_drafts_total": 10,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total/position=0": 8,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total/position=1": 5,
    }


@pytest.mark.parametrize("ids", [[42, 248046], [42], []])
def test_mtp_token_evidence_requires_every_generated_token_including_eos(tmp_path, ids):
    event = {"prompt_token_ids": [1], "choices": [{"token_ids": ids}]}
    (tmp_path / "case-sse.jsonl").write_text(
        json.dumps({"line": "data: " + json.dumps(event)}) + "\n"
    )
    (tmp_path / "case-response.json").write_text(
        json.dumps({"usage": {"prompt_tokens": 1, "completion_tokens": 2}})
    )
    if len(ids) == 2:
        assert mtp.tokens(tmp_path, "case") == ([1], ids)
    else:
        with pytest.raises(ValueError, match="incomplete generated"):
            mtp.tokens(tmp_path, "case")


def test_vision_harness_does_not_override_dtype_selected_kvarn_defaults():
    env = vision.runtime_environment()
    assert all(not key.startswith(("KVARN_", "VLLM_")) for key in env)


def test_mtp_logprobs_stop_after_first_divergent_token():
    def row(token):
        return {
            "token": token,
            "logprob": -0.5,
            "top_logprobs": [
                {"token": "token_id:1", "logprob": -0.5},
                {"token": "token_id:2", "logprob": -0.625},
            ],
        }

    left = [row("token_id:1"), row("token_id:1"), row("token_id:1")]
    right = [row("token_id:1"), row("token_id:2"), row("token_id:1")]
    result = mtp.shared_prefix_logprobs(left, right)
    assert result["positions_compared"] == 2
    assert result["max_common_logprob_delta"] == 0
    assert result["positions"][1]["off_top_margin"] == 0.125


@pytest.mark.parametrize("token", ["token_id:42", "token_id:43"])
def test_mtp_logprobs_require_generated_token_alignment(tmp_path, token):
    (tmp_path / "case-request.json").write_text('{"logprobs":true}')
    entry = {
        "token": token,
        "logprob": -0.5,
        "top_logprobs": [{"token": token, "logprob": -0.5}],
    }
    event = {"choices": [{"logprobs": {"content": [entry]}}]}
    (tmp_path / "case-sse.jsonl").write_text(
        json.dumps({"line": "data: " + json.dumps(event)}) + "\n"
    )
    if token == "token_id:42":
        assert mtp.logprobs(tmp_path, "case", [42]) == [entry]
    else:
        with pytest.raises(ValueError, match="misaligned"):
            mtp.logprobs(tmp_path, "case", [42])


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
    context = 6143
    cases = vision.fixtures(tmp_path / "images")
    tokens = [0] * context
    tokens[20] = 248056
    monkeypatch.setattr(
        vision, "tokenize_case", lambda *a: {"count": context, "tokens": tokens}
    )
    with pytest.raises(ValueError, match="beyond sink"):
        vision.long_case("http://127.0.0.1:8017", cases[0], tmp_path, context)


@pytest.mark.parametrize("budget", [2048, 4096, 8192])
def test_long_context_coverage_tracks_actual_128k_target(tmp_path, monkeypatch, budget):
    cases = vision.fixtures(tmp_path / "images")
    tokens = [0] * 130943
    tokens[266:462] = [248056] * 196
    monkeypatch.setattr(
        vision, "tokenize_case", lambda *a: {"count": len(tokens), "tokens": tokens}
    )
    case = vision.long_case(
        "http://localhost", cases[0], tmp_path, len(tokens), chunk_budget=budget
    )
    assert case["coverage"]["prompt_tokens"] == 130943
    assert (
        case["coverage"]["expected_chunk_count"] == (len(tokens) + budget - 1) // budget
    )
    assert case["coverage"]["decode_crosses_page_after_tokens"] == 1


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


@pytest.mark.parametrize("raw", ["1024 KiB", "1 MiB", "0.0009765625 GiB"])
def test_memory_summary_parses_integer_drm_units_and_rejects_fractional(tmp_path, raw):
    sample = {
        "seconds": 1,
        "clients": {
            "device:1": {
                "fields": {
                    "drm-pdev": "device",
                    "drm-resident-vram0": raw,
                    "drm-shared-vram0": "0",
                }
            }
        },
        "errors": [],
    }
    (tmp_path / "memory-fdinfo.jsonl").write_text(json.dumps(sample) + "\n")
    if "." in raw:
        with pytest.raises(ValueError):
            comparison.memory_summary(tmp_path)
    else:
        result = comparison.memory_summary(tmp_path)
        assert (
            result["peak_bytes_by_device_and_field"]["device/drm-resident-vram0"]
            == 1024**2
        )


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
