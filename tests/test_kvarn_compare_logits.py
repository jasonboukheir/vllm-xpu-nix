# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import numpy as np
import pytest

from scripts.kvarn_compare_logits import (
    compare,
    evaluate_acceptance,
    load_artifact,
    main,
)


def _write_artifact(path, *, prompt, forced, rows, full_logits=False):
    width = max(len(row) for row in rows)
    token_ids = np.full((len(rows), width), -1, dtype=np.int32)
    logits = np.full((len(rows), width), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
        for column, (token_id, value) in enumerate(row.items()):
            token_ids[index, column] = token_id
            logits[index, column] = value
    np.savez(
        path,
        prompt_token_ids=np.asarray(prompt, dtype=np.int32),
        forced_token_ids=np.asarray(forced, dtype=np.int32),
        logit_token_ids=token_ids,
        raw_logits=logits,
        full_logits=np.asarray(full_logits),
    )


def test_compare_reports_agreement_errors_and_context_drift(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    _write_artifact(
        reference_path,
        prompt=[1, 2, 3],
        forced=[7, 8],
        rows=[{7: 2.0, 9: 2.0, 4: 0.0}, {8: 3.0, 5: 2.0, 6: 1.0}],
    )
    _write_artifact(
        candidate_path,
        prompt=[1, 2, 3],
        forced=[7, 8],
        rows=[{9: 2.0, 7: 1.9995, 4: 0.1}, {8: 2.5, 6: 2.0, 5: 1.0}],
    )

    result = compare(
        load_artifact(reference_path),
        load_artifact(candidate_path),
        tie_tolerance=0.001,
        boundaries=(4,),
    )

    assert result["decode_steps"] == 2
    assert result["top1_agreement_rate"] == 0.5
    assert result["tie_aware_top1_agreement_rate"] == 1.0
    assert result["top5_exact_agreement_rate"] == 1.0
    assert result["top5_mean_jaccard"] == 1.0
    assert result["mean_intersection_coverage"] == 1.0
    assert result["selected_token_delta"]["count"] == 2
    assert result["selected_token_delta"]["max_abs"] == pytest.approx(0.5)
    assert set(result["drift_by_context"]) == {"1-4", "5+"}


def test_compare_rejects_different_forced_sequence(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_artifact(first, prompt=[1], forced=[2], rows=[{2: 1.0}])
    _write_artifact(second, prompt=[1], forced=[3], rows=[{3: 1.0}])

    with pytest.raises(ValueError, match="forced token IDs differ"):
        compare(
            load_artifact(first),
            load_artifact(second),
            tie_tolerance=0.0,
            boundaries=(4096,),
        )


def test_compare_requires_selected_token_logits(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_artifact(first, prompt=[1], forced=[2], rows=[{2: 1.0, 4: 0.0}])
    _write_artifact(second, prompt=[1], forced=[2], rows=[{4: 0.0, 5: -1.0}])

    with pytest.raises(ValueError, match="forced token 2 is absent"):
        compare(
            load_artifact(first),
            load_artifact(second),
            tie_tolerance=0.0,
            boundaries=(4096,),
        )


@pytest.mark.parametrize("non_finite", [np.nan, np.inf])
def test_load_artifact_rejects_non_finite_logit_for_real_token(tmp_path, non_finite):
    artifact = tmp_path / "non-finite.npz"
    _write_artifact(
        artifact,
        prompt=[1],
        forced=[2],
        rows=[{2: 1.0, 4: non_finite}],
    )

    with pytest.raises(
        ValueError,
        match=r"NaN or positive-infinite logit at decode step 0, column 1, token ID 4",
    ):
        load_artifact(artifact)


def test_compare_allows_matching_negative_infinity_masks(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    rows = [{2: 1.0, 4: -np.inf}]
    _write_artifact(reference_path, prompt=[1], forced=[2], rows=rows)
    _write_artifact(candidate_path, prompt=[1], forced=[2], rows=rows)

    report = compare(
        load_artifact(reference_path),
        load_artifact(candidate_path),
        tie_tolerance=0.0,
        boundaries=(4096,),
    )

    assert report["matched_negative_infinity_count"] == 1
    assert report["matched_logit_delta"]["count"] == 1


def test_compare_rejects_asymmetric_negative_infinity_masks(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    _write_artifact(reference_path, prompt=[1], forced=[2], rows=[{2: 1.0, 4: -np.inf}])
    _write_artifact(candidate_path, prompt=[1], forced=[2], rows=[{2: 1.0, 4: -2.0}])

    with pytest.raises(ValueError, match="asymmetric negative-infinity mask"):
        compare(
            load_artifact(reference_path),
            load_artifact(candidate_path),
            tie_tolerance=0.0,
            boundaries=(4096,),
        )


def test_load_artifact_allows_non_finite_padding(tmp_path):
    artifact = tmp_path / "padded.npz"
    _write_artifact(
        artifact,
        prompt=[1],
        forced=[2, 3],
        rows=[{2: 1.0, 4: 0.0}, {3: 2.0}],
    )

    result = load_artifact(artifact)

    assert result["rows"] == [{2: 1.0, 4: 0.0}, {3: 2.0}]


def test_acceptance_reports_inclusive_passes_and_failures(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    _write_artifact(
        reference_path,
        prompt=[1],
        forced=[2, 3],
        rows=[{2: 2.0, 4: 1.0}, {3: 3.0, 5: 1.0}],
    )
    _write_artifact(
        candidate_path,
        prompt=[1],
        forced=[2, 3],
        rows=[{2: 1.5, 4: 1.0}, {5: 3.0, 3: 2.5}],
    )
    report = compare(
        load_artifact(reference_path),
        load_artifact(candidate_path),
        tie_tolerance=0.0,
        boundaries=(4096,),
    )

    acceptance = evaluate_acceptance(
        report,
        {
            "min_top1_agreement_rate": 0.5,
            "max_selected_token_mae": 0.5,
            "max_matched_logit_rmse": 1.1,
        },
    )

    assert acceptance["status"] == "passed"
    assert all(check["passed"] for check in acceptance["checks"])

    failed = evaluate_acceptance(
        report,
        {
            "min_top1_agreement_rate": 0.75,
            "max_selected_token_mae": 0.49,
        },
    )
    assert failed["status"] == "failed"
    assert [failure["name"] for failure in failed["failures"]] == [
        "min_top1_agreement_rate",
        "max_selected_token_mae",
    ]


def test_cli_is_report_only_without_thresholds(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    output = tmp_path / "comparison.json"
    _write_artifact(reference_path, prompt=[1], forced=[2], rows=[{2: 2.0}])
    _write_artifact(candidate_path, prompt=[1], forced=[2], rows=[{2: 1.0}])

    exit_code = main(
        [
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output),
        ]
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert document["status"] == "report_only"
    assert document["acceptance"]["status"] == "not_evaluated"


def test_cli_writes_failed_acceptance_and_returns_nonzero(tmp_path):
    reference_path = tmp_path / "bf16.npz"
    candidate_path = tmp_path / "kvarn.npz"
    output = tmp_path / "comparison.json"
    _write_artifact(
        reference_path,
        prompt=[1],
        forced=[2],
        rows=[{2: 2.0, 3: 1.0}],
    )
    _write_artifact(
        candidate_path,
        prompt=[1],
        forced=[2],
        rows=[{3: 2.0, 2: 1.0}],
    )

    exit_code = main(
        [
            "--reference",
            str(reference_path),
            "--candidate",
            str(candidate_path),
            "--output",
            str(output),
            "--min-top1-agreement-rate",
            "1.0",
        ]
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert document["status"] == "failed"
    assert document["acceptance"]["failures"] == [
        {
            "actual": 0.0,
            "comparison": ">=",
            "metric": "top1_agreement_rate",
            "name": "min_top1_agreement_rate",
            "passed": False,
            "threshold": 1.0,
        }
    ]
