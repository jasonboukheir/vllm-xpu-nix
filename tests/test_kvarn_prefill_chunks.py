"""Reject stale 2K assumptions and incomplete full-prefill traces."""

import pytest

from scripts import kvarn_chunk_compare as comparison
from scripts.kvarn_chunk_compare import canonical_arguments, preemptions
from scripts.kvarn_prefill_chunks import chunk_plan, recorded_budget
from scripts.kvarn_prefill_trace import packed_flush_sequence, validate_steps


def step(query, end, decode=False):
    context = (
        "0(sq0sk0sqsq0sqsk0)"
        if decode
        else f"1(sq{query}sk{end}sqsq{query**2}sqsk{query * end})"
    )
    generation = f"1(sq1sk{end}sqsq1sqsk{end})" if decode else "0(sq0sk0sqsq0sqsk0)"
    return {
        "annotation": f"execute_{1 if decode else query}_context_{context}_generation_{generation}"
    }


@pytest.mark.parametrize("budget", [2048, 4096, 8192])
@pytest.mark.parametrize("tokens", [16383, 65023])
def test_full_prefill_and_final_partial_chunk(tokens, budget):
    plan = chunk_plan(tokens, budget)
    steps = [
        step(e["end"] - e["start"], e["end"]) for e in plan["expected_chunk_extents"]
    ]
    steps += [step(1, tokens + i, True) for i in (1, 2)]
    assert validate_steps(steps, tokens, budget) == plan["expected_chunk_count"]
    assert len(steps) == plan["expected_profile_steps"]
    assert plan["expected_chunk_extents"][-1]["end"] == tokens
    with pytest.raises(ValueError, match="full prefill"):
        validate_steps(steps[:-1], tokens, budget)
    steps[-3] = step(budget, tokens)
    with pytest.raises(ValueError, match="chunk lengths"):
        validate_steps(steps, tokens, budget)


def test_trace_rejects_stale_2k_budget():
    steps = [
        step(4096, 4096),
        step(4095, 8191),
        step(1, 8192, True),
        step(1, 8193, True),
    ]
    with pytest.raises(ValueError):
        validate_steps(steps, 8191, 2048)


@pytest.mark.parametrize("budget", [0, -1])
def test_invalid_budget(budget):
    with pytest.raises(ValueError, match="positive"):
        chunk_plan(65023, budget)


def test_actual_argv_is_authoritative_and_must_agree():
    manifest = {"actual_argv": ["vllm", "--max-num-batched-tokens", "8192"]}
    assert recorded_budget(manifest) == 8192
    manifest["max_num_batched_tokens"] = 2048
    with pytest.raises(ValueError, match="disagree"):
        recorded_budget(manifest)
    with pytest.raises(ValueError, match="ambiguous"):
        recorded_budget({"actual_argv": []})


def test_comparison_preserves_dtype_and_other_scheduler_settings():
    argv = [
        "vllm",
        "--kv-cache-dtype",
        "auto",
        "--max-num-batched-tokens",
        "4096",
        "--max-model-len",
        "65536",
    ]
    canonical = canonical_arguments({"actual_argv": argv})
    assert canonical == argv[:4] + ["<CHUNK_BUDGET>"] + argv[5:]
    assert argv[4] == "4096"


@pytest.mark.parametrize("value", ["nan", "inf", "-1", None])
def test_preemption_evidence_fails_closed(tmp_path, value):
    (tmp_path / "metrics.txt").write_text(
        "" if value is None else f'vllm:num_preemptions_total{{engine="0"}} {value}\n'
    )
    with pytest.raises(ValueError, match="preemption"):
        preemptions(tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("--kv-cache-dtype", "kvarn_k4v4_g128_compact"),
        ("--max-model-len", "131072"),
        ("--gpu-memory-utilization", "0.95"),
    ],
)
def test_pair_rejects_non_budget_service_changes(tmp_path, monkeypatch, field, value):
    def audit(path):
        argv = [
            "vllm",
            "--max-num-batched-tokens",
            "2048",
            "--kv-cache-dtype",
            "auto",
            "--max-model-len",
            "65536",
            "--gpu-memory-utilization",
            "0.90",
        ]
        if path.name == "candidate":
            argv[2] = "4096"
            argv[argv.index(field) + 1] = value
        return {
            "actual_argv": argv,
            "suite": "text-prefill-performance-v1",
            "profiler_config": None,
            "speculative_config": None,
            "runtime_identity": {},
            "harness_sha256": {},
            "image_sha256": {},
            "actual_environment": {},
        }, []

    monkeypatch.setattr(comparison.vision, "audit", audit)
    with pytest.raises(ValueError, match="non-budget"):
        comparison.compare([(tmp_path / "control", tmp_path / "candidate")])


def test_packing_sequence_requires_same_thread_and_full_scope():
    scope = {
        "name": step(4096, 4096)["annotation"],
        "ts": 0,
        "dur": 100,
        "pid": 1,
        "tid": 2,
    }
    pack = {
        "name": "_vllm_fa2_C::kvarn_pack_balanced_kv",
        "ts": 10,
        "dur": 5,
        "pid": 1,
        "tid": 2,
        "args": {"Input Dims": [[16, 128]]},
    }
    document = {"traceEvents": [scope, pack]}
    assert packed_flush_sequence(document) == [
        {"step": 1, "timestamp_us": 10, "inputs": {"Input Dims": [[16, 128]]}}
    ]
    pack["tid"] = 3
    with pytest.raises(ValueError, match="unique execution owner"):
        packed_flush_sequence(document)
