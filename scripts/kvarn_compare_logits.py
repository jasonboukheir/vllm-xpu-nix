# SPDX-License-Identifier: Apache-2.0
"""Compare paired BF16 and KVarN persistent forced-decode artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

ACCEPTANCE_THRESHOLD_SPECS = {
    "min_top1_agreement_rate": ("top1_agreement_rate", ">="),
    "min_tie_aware_top1_agreement_rate": (
        "tie_aware_top1_agreement_rate",
        ">=",
    ),
    "min_top5_exact_agreement_rate": ("top5_exact_agreement_rate", ">="),
    "min_top5_mean_jaccard": ("top5_mean_jaccard", ">="),
    "min_mean_intersection_coverage": ("mean_intersection_coverage", ">="),
    "max_selected_token_mae": ("selected_token_delta.mae", "<="),
    "max_selected_token_p99_abs": ("selected_token_delta.p99_abs", "<="),
    "max_matched_logit_rmse": ("matched_logit_delta.rmse", "<="),
    "max_matched_logit_p99_abs": ("matched_logit_delta.p99_abs", "<="),
}
REPORT_SCHEMA_VERSION = 2
ACCEPTANCE_PROFILE_VERSION = 1


def _rows(token_ids: np.ndarray, logits: np.ndarray) -> list[dict[int, float]]:
    if token_ids.ndim != 2 or logits.shape != token_ids.shape:
        raise ValueError("logit_token_ids and raw_logits must be equal-size matrices")
    rows: list[dict[int, float]] = []
    for step, (ids, values) in enumerate(zip(token_ids, logits)):
        row: dict[int, float] = {}
        for column, (token_id, value) in enumerate(zip(ids, values)):
            token_id = int(token_id)
            if token_id < 0:
                continue
            value = float(value)
            if math.isnan(value) or value == math.inf:
                raise ValueError(
                    "NaN or positive-infinite logit at decode step "
                    f"{step}, column {column}, token ID {token_id}"
                )
            row[token_id] = value
        if not any(math.isfinite(value) for value in row.values()):
            raise ValueError("every decode step must contain at least one finite logit")
        rows.append(row)
    return rows


def load_artifact(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "prompt_token_ids",
            "forced_token_ids",
            "logit_token_ids",
            "raw_logits",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        prompt = np.asarray(data["prompt_token_ids"], dtype=np.int64)
        forced = np.asarray(data["forced_token_ids"], dtype=np.int64)
        rows = _rows(data["logit_token_ids"], data["raw_logits"])
        full_logits = bool(data["full_logits"]) if "full_logits" in data else False
        schema_version = (
            int(data["artifact_schema_version"])
            if "artifact_schema_version" in data
            else 1
        )
    if prompt.ndim != 1 or forced.ndim != 1:
        raise ValueError("prompt_token_ids and forced_token_ids must be vectors")
    if len(rows) != forced.size:
        raise ValueError("the number of logit rows must match forced_token_ids")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "schema_version": schema_version,
        "prompt": prompt,
        "forced": forced,
        "rows": rows,
        "full_logits": full_logits,
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "mae": float(np.abs(array).mean()),
        "rmse": float(np.sqrt(np.square(array).mean())),
        "p50_abs": float(np.percentile(np.abs(array), 50)),
        "p95_abs": float(np.percentile(np.abs(array), 95)),
        "p99_abs": float(np.percentile(np.abs(array), 99)),
        "max_abs": float(np.abs(array).max()),
    }


def _top_ids(row: dict[int, float], count: int) -> list[int]:
    return [
        token_id
        for token_id, _ in sorted(row.items(), key=lambda item: (-item[1], item[0]))[
            :count
        ]
    ]


def _tie_set(row: dict[int, float], tolerance: float) -> set[int]:
    maximum = max(row.values())
    return {token_id for token_id, value in row.items() if maximum - value <= tolerance}


def _bucket_name(context_position: int, boundaries: tuple[int, ...]) -> str:
    lower = 0
    for upper in boundaries:
        if context_position <= upper:
            return f"{lower + 1}-{upper}"
        lower = upper
    return f"{lower + 1}+"


def compare(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    tie_tolerance: float,
    boundaries: tuple[int, ...],
) -> dict[str, Any]:
    if not np.array_equal(reference["prompt"], candidate["prompt"]):
        raise ValueError("prompt token IDs differ")
    if not np.array_equal(reference["forced"], candidate["forced"]):
        raise ValueError("forced token IDs differ")

    forced = reference["forced"]
    prompt_length = int(reference["prompt"].size)
    row_deltas: list[float] = []
    selected_deltas: list[float] = []
    top1_matches = 0
    tie_matches = 0
    top5_exact_matches = 0
    top5_overlap: list[float] = []
    intersection_coverage: list[float] = []
    bucket_values: dict[str, dict[str, list[float] | int]] = {}
    matched_negative_infinity = 0

    for step, (reference_row, candidate_row) in enumerate(
        zip(reference["rows"], candidate["rows"])
    ):
        common = reference_row.keys() & candidate_row.keys()
        if not common:
            raise ValueError(f"decode step {step} has no token IDs in common")
        finite_common = []
        for token in common:
            reference_masked = reference_row[token] == -math.inf
            candidate_masked = candidate_row[token] == -math.inf
            if reference_masked != candidate_masked:
                raise ValueError(
                    f"decode step {step}, token ID {token} has an asymmetric "
                    "negative-infinity mask"
                )
            if reference_masked:
                matched_negative_infinity += 1
            else:
                finite_common.append(token)
        if not finite_common:
            raise ValueError(f"decode step {step} has no finite token IDs in common")
        differences = [
            candidate_row[token] - reference_row[token] for token in finite_common
        ]
        row_deltas.extend(differences)
        intersection_coverage.append(
            len(common) / max(len(reference_row), len(candidate_row))
        )

        forced_token = int(forced[step])
        if forced_token not in common:
            raise ValueError(
                f"forced token {forced_token} is absent from step {step} logits"
            )
        if reference_row[forced_token] == -math.inf:
            raise ValueError(f"forced token {forced_token} is masked at step {step}")
        selected_delta = candidate_row[forced_token] - reference_row[forced_token]
        selected_deltas.append(selected_delta)

        reference_top1 = _top_ids(reference_row, 1)[0]
        candidate_top1 = _top_ids(candidate_row, 1)[0]
        top1_matches += reference_top1 == candidate_top1
        tie_matches += candidate_top1 in _tie_set(
            reference_row, tie_tolerance
        ) and reference_top1 in _tie_set(candidate_row, tie_tolerance)

        reference_top5 = set(_top_ids(reference_row, 5))
        candidate_top5 = set(_top_ids(candidate_row, 5))
        top5_exact_matches += reference_top5 == candidate_top5
        top5_overlap.append(
            len(reference_top5 & candidate_top5)
            / max(len(reference_top5 | candidate_top5), 1)
        )

        context_position = prompt_length + step + 1
        bucket = bucket_values.setdefault(
            _bucket_name(context_position, boundaries),
            {"steps": 0, "selected_deltas": [], "row_deltas": []},
        )
        bucket["steps"] = int(bucket["steps"]) + 1
        assert isinstance(bucket["selected_deltas"], list)
        assert isinstance(bucket["row_deltas"], list)
        bucket["selected_deltas"].append(selected_delta)
        bucket["row_deltas"].extend(differences)

    steps = len(reference["rows"])
    drift = {
        name: {
            "steps": values["steps"],
            "selected_token_delta": _summary(values["selected_deltas"]),
            "matched_logit_delta": _summary(values["row_deltas"]),
        }
        for name, values in bucket_values.items()
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "reference": {
            "path": reference["path"],
            "sha256": reference["sha256"],
            "artifact_schema_version": reference["schema_version"],
        },
        "candidate": {
            "path": candidate["path"],
            "sha256": candidate["sha256"],
            "artifact_schema_version": candidate["schema_version"],
        },
        "prompt_tokens": prompt_length,
        "decode_steps": steps,
        "context_end": prompt_length + steps,
        "full_logits": {
            "reference": reference["full_logits"],
            "candidate": candidate["full_logits"],
        },
        "top1_agreement_rate": top1_matches / steps,
        "tie_aware_top1_agreement_rate": tie_matches / steps,
        "tie_tolerance": tie_tolerance,
        "top5_exact_agreement_rate": top5_exact_matches / steps,
        "top5_mean_jaccard": float(np.mean(top5_overlap)),
        "mean_intersection_coverage": float(np.mean(intersection_coverage)),
        "matched_negative_infinity_count": matched_negative_infinity,
        "selected_token_delta": _summary(selected_deltas),
        "matched_logit_delta": _summary(row_deltas),
        "drift_by_context": drift,
    }


def _metric_value(report: dict[str, Any], path: str) -> float:
    value: Any = report
    for component in path.split("."):
        if not isinstance(value, dict) or component not in value:
            raise ValueError(f"comparison report is missing metric {path}")
        value = value[component]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"comparison metric {path} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"comparison metric {path} must be finite")
    return normalized


def evaluate_acceptance(
    report: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    """Evaluate inclusive quality bounds against a comparison report."""
    unknown = thresholds.keys() - ACCEPTANCE_THRESHOLD_SPECS.keys()
    if unknown:
        raise ValueError("unknown acceptance thresholds: " + ", ".join(sorted(unknown)))

    checks: list[dict[str, Any]] = []
    for name, threshold in thresholds.items():
        metric, comparison = ACCEPTANCE_THRESHOLD_SPECS[name]
        actual = _metric_value(report, metric)
        passed = actual >= threshold if comparison == ">=" else actual <= threshold
        checks.append(
            {
                "name": name,
                "metric": metric,
                "comparison": comparison,
                "threshold": threshold,
                "actual": actual,
                "passed": passed,
            }
        )

    failures = [check for check in checks if not check["passed"]]
    return {
        "profile_version": ACCEPTANCE_PROFILE_VERSION,
        "status": "failed" if failures else "passed",
        "thresholds": dict(thresholds),
        "checks": checks,
        "failures": failures,
    }


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {
        name: value
        for name in ACCEPTANCE_THRESHOLD_SPECS
        if (value := getattr(args, name)) is not None
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tie-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--context-boundary",
        type=int,
        action="append",
        default=[],
        help="upper context-position boundary; repeat as needed",
    )
    parser.add_argument(
        "--min-top1-agreement-rate",
        type=float,
        help="require top-1 agreement at or above this inclusive rate",
    )
    parser.add_argument(
        "--min-tie-aware-top1-agreement-rate",
        type=float,
        help="require tie-aware top-1 agreement at or above this inclusive rate",
    )
    parser.add_argument(
        "--min-top5-exact-agreement-rate",
        type=float,
        help="require exact top-5-set agreement at or above this inclusive rate",
    )
    parser.add_argument(
        "--min-top5-mean-jaccard",
        type=float,
        help="require mean top-5 Jaccard similarity at or above this value",
    )
    parser.add_argument(
        "--min-mean-intersection-coverage",
        type=float,
        help="require mean captured-logit intersection coverage at or above this value",
    )
    parser.add_argument(
        "--max-selected-token-mae",
        type=float,
        help="require selected-token logit MAE at or below this value",
    )
    parser.add_argument(
        "--max-selected-token-p99-abs",
        type=float,
        help="require selected-token absolute p99 error at or below this value",
    )
    parser.add_argument(
        "--max-matched-logit-rmse",
        type=float,
        help="require matched-logit RMSE at or below this value",
    )
    parser.add_argument(
        "--max-matched-logit-p99-abs",
        type=float,
        help="require matched-logit absolute p99 error at or below this value",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.tie_tolerance) or args.tie_tolerance < 0:
        parser.error("--tie-tolerance must be finite and non-negative")
    if any(boundary <= 0 for boundary in args.context_boundary):
        parser.error("--context-boundary values must be positive")
    for name, value in _thresholds_from_args(args).items():
        option = "--" + name.replace("_", "-")
        if not math.isfinite(value):
            parser.error(f"{option} must be finite")
        if name.startswith("min_") and not 0 <= value <= 1:
            parser.error(f"{option} must be between zero and one")
        if name.startswith("max_") and value < 0:
            parser.error(f"{option} must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    boundaries = tuple(sorted(set(args.context_boundary or [4096, 16384, 32768])))
    try:
        document = compare(
            load_artifact(args.reference),
            load_artifact(args.candidate),
            tie_tolerance=args.tie_tolerance,
            boundaries=boundaries,
        )
        thresholds = _thresholds_from_args(args)
        if thresholds:
            acceptance = evaluate_acceptance(document, thresholds)
            document["status"] = acceptance["status"]
        else:
            acceptance = {
                "profile_version": ACCEPTANCE_PROFILE_VERSION,
                "status": "not_evaluated",
                "thresholds": {},
                "checks": [],
                "failures": [],
            }
            document["status"] = "report_only"
        document["acceptance"] = acceptance
    except (OSError, TypeError, ValueError) as exc:
        document = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "status": "invalid",
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if document["status"] == "invalid":
        return 2
    return int(document["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
