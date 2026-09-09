"""Model-reference gates allow bounded score drift while retaining fixed history."""

import json

import numpy as np
import pytest

from scripts import kvarn_chunk_accuracy as accuracy


def artifact(path, *, drift=False, budget=2048, forced=None):
    values = np.tile([3.0, 2.99, 2.0, 1.0, 0.0], (100, 1))
    if drift:
        values[0, 1] = 3.01  # One different greedy choice, with identical history.
    np.savez(
        path,
        artifact_schema_version=2,
        model=accuracy.service.MODEL,
        engine_kwargs_json=json.dumps(accuracy.engine_options(budget, "auto")),
        prompt_token_ids=[1, 2],
        forced_token_ids=[0] * 100 if forced is None else forced,
        logit_token_ids=np.tile(np.arange(5), (100, 1)),
        raw_logits=values,
        full_logits=False,
    )


def test_bounded_model_error_is_not_rejected_for_one_different_greedy_choice(tmp_path):
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    artifact(reference)
    artifact(candidate, drift=True)
    report = accuracy.compare_pair(
        reference,
        candidate,
        {
            "min_top1_agreement_rate": 0.98,
            "max_matched_logit_rmse": 0.75,
        },
        1e-5,
    )
    assert report["top1_agreement_rate"] == 0.99
    assert report["status"] == "passed"


def test_changed_teacher_sequence_is_rejected(tmp_path):
    reference, candidate = tmp_path / "reference.npz", tmp_path / "candidate.npz"
    artifact(reference)
    artifact(candidate, forced=[1] * 100)
    with pytest.raises(ValueError, match="forced token IDs differ"):
        accuracy.compare_pair(reference, candidate, {}, 1e-5)


def test_recorded_engine_budget_must_match_plan(tmp_path):
    path = tmp_path / "result.npz"
    artifact(path, budget=8192)
    with pytest.raises(ValueError, match="identity"):
        accuracy.validate_artifact(
            path, accuracy.engine_options(2048, "auto"), [1, 2], [0] * 100
        )


def test_schedule_prioritizes_8k_and_reverses_order_on_second_start():
    runs = accuracy.schedule([16383, 65023], [2048, 8192, 4096], 2)
    assert len(runs) == 24
    assert len({tuple(r.values()) for r in runs}) == 24
    assert all(r["budget"] != 4096 for r in runs[:16])
    assert [(r["budget"], r["dtype"]) for r in runs[:4]] == [
        (2048, "auto"),
        (2048, accuracy.perf.COMPACT_DTYPE),
        (8192, "auto"),
        (8192, accuracy.perf.COMPACT_DTYPE),
    ]
    assert [(r["budget"], r["dtype"]) for r in runs[8:12]] == [
        (8192, accuracy.perf.COMPACT_DTYPE),
        (8192, "auto"),
        (2048, accuracy.perf.COMPACT_DTYPE),
        (2048, "auto"),
    ]


def test_only_dtype_and_budget_change_in_engine_options():
    left = accuracy.engine_options(2048, "auto")
    right = accuracy.engine_options(8192, accuracy.perf.COMPACT_DTYPE)
    assert {k for k in left if left[k] != right[k]} == {
        "kv_cache_dtype",
        "max_num_batched_tokens",
    }
    assert left["enable_prefix_caching"] is False
    assert left["max_model_len"] == 65536
