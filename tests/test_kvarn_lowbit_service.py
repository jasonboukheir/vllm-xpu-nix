"""Protect serving measurements from incomplete streams and MTP chunk bias."""

import io
import json
import threading
from copy import deepcopy

import pytest

from scripts import kvarn_lowbit_service as runner


def _history_fixture():
    # Three complete pairs: both match, only first matches, first rejects.
    # A correct second guess after a wrong first guess is not an acceptance.
    inputs = [[11, 12], [20], [21], [22], [23]]
    positions = [[0, 1], [2], [3], [4], [5]]
    output = [20, 21, 22, 23, 24]
    drafts = [[21, 22], [22, 99], [99, 24], [24, 25], [25, 26]]
    rows = [
        {
            "step": i,
            "suppress": True,
            "target_tokens": tokens,
            "query_start_loc": [0, len(tokens)],
            "seq_lens": [positions[i][-1] + 1],
            "target_positions": [positions[i]] * 3,
            "num_rejected_tokens": None,
            "target_hidden_states": {
                "shape": [len(tokens), 5120],
                "dtype": "torch.bfloat16",
                "sha256": str(i) * 64,
            },
            "discard": [False],
            "next_token_ids": [output[i]],
            "draft_token_ids": drafts[i],
        }
        for i, tokens in enumerate(inputs)
    ]
    return rows, [11, 12], output


def test_history_scoring_aligns_shift_and_counts_only_on_policy_second():
    from scripts.kvarn_mtp_history_compare import audit_history

    result = audit_history(*_history_fixture())
    assert result["unique_prefixes"] == 3
    assert result["first_agreements"] == 2
    assert result["both_agreements"] == 1
    assert result["conditional_second_agreement_rate"] == 0.5
    assert result["counterfactual_accepted_tokens_per_pair"] == 1
    assert result["matches"] == [[True, True], [True, False], [False, False]]


@pytest.mark.parametrize("failure", ["index", "position", "history", "target", "tail"])
def test_history_scoring_rejects_incomplete_or_misaligned_inputs(failure):
    from scripts.kvarn_mtp_history_compare import audit_history

    rows, prompt, output = _history_fixture()
    if failure == "index":
        rows[1]["step"] = 0
    elif failure == "position":
        rows[1]["target_positions"] = [[3]] * 3
    elif failure == "history":
        rows[1]["target_tokens"] = [99]
    elif failure == "target":
        rows[1]["next_token_ids"] = [99]
    else:
        rows.pop()
    with pytest.raises(ValueError):
        audit_history(rows, prompt, output)


def test_matched_histories_allow_only_draft_prediction_differences():
    from scripts.kvarn_mtp_history_compare import require_matched_states

    rows, _, _ = _history_fixture()
    changed = deepcopy(rows)
    changed[0]["draft_token_ids"] = [99, 98]
    require_matched_states(rows, changed)
    changed[1]["target_hidden_states"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="target state differs at invocation 1"):
        require_matched_states(rows, changed)


def _acceptance_counts(*, rounds=5, proposed=10, first=4, second=2):
    return {
        "vllm:spec_decode_num_drafts_total": rounds,
        "vllm:spec_decode_num_draft_tokens_total": proposed,
        "vllm:spec_decode_num_accepted_tokens_total": first + second,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total/position=0": first,
        "vllm:spec_decode_num_accepted_tokens_per_pos_total/position=1": second,
    }


def test_acceptance_counts_verify_work_and_exact_second_denominator():
    from scripts.kvarn_mtp_draft_compare import acceptance

    result = acceptance(_acceptance_counts(), output_tokens=12, requests=1)
    assert result["first_acceptance"] == 4 / 5
    assert result["conditional_second_acceptance_bounds"] == [0.5, 0.5]
    assert result["both_acceptance_per_two_token_round"] == 2 / 5
    assert result["accepted_tokens_per_round"] == 6 / 5
    assert result["output_budget_clipped_tokens"] == 0


def test_trimmed_drafts_require_second_denominator_bounds():
    from scripts.kvarn_mtp_draft_compare import acceptance

    result = acceptance(_acceptance_counts(proposed=8), output_tokens=11, requests=1)
    assert result["two_token_rounds"] == 3
    assert result["one_token_rounds"] == 2
    assert result["conditional_second_denominator_bounds"] == [2, 3]
    assert result["conditional_second_acceptance_bounds"] == [2 / 3, 1]
    assert result["output_budget_clipped_tokens"] == 1


@pytest.mark.parametrize(
    "changes,output",
    [
        ({"rounds": 0}, 12),
        ({"first": 6}, 14),
        ({"second": 5}, 15),
        ({"proposed": 6}, 12),
        ({"first": float("nan")}, 12),
        ({}, 8),
        ({}, 13),
    ],
)
def test_acceptance_rejects_impossible_counters_or_emitted_work(changes, output):
    from scripts.kvarn_mtp_draft_compare import acceptance

    with pytest.raises(ValueError):
        acceptance(_acceptance_counts(**changes), output, 1)


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
@pytest.mark.parametrize(
    "draft_dtype",
    [None, "kvarn_k4v4_g128_compact", "kvarn_k4v2_g128_compact", "bfloat16"],
)
def test_mtp_activity_requires_two_token_work_but_allows_budget_trimming(
    draft_tokens, draft_dtype
):
    from scripts.kvarn_lowbit_compare import audit_mtp_activity

    counters = {
        "vllm:spec_decode_num_drafts_total": 31,
        "vllm:spec_decode_num_draft_tokens_total": draft_tokens,
    }
    config = {"method": "mtp", "num_speculative_tokens": 2}
    if draft_dtype:
        config["kv_cache_dtype"] = draft_dtype
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
