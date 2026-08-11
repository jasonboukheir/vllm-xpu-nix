"""Compare a KVarN serving benchmark with its matched BF16 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MIN_THROUGHPUT_RATIO = 0.95
DEFAULT_MAX_TPOT_RATIO = 1.10


def _load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    required = ("completed", "failed", "output_throughput", "mean_tpot_ms")
    missing = [key for key in required if key not in result]
    if missing:
        raise ValueError(f"{path}: missing benchmark keys: {', '.join(missing)}")
    return result


def compare_results(
    baseline_path: Path,
    candidate_path: Path,
    min_throughput_ratio: float = DEFAULT_MIN_THROUGHPUT_RATIO,
    max_tpot_ratio: float = DEFAULT_MAX_TPOT_RATIO,
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    throughput_ratio = (
        candidate["output_throughput"] / baseline["output_throughput"]
    )
    tpot_ratio = candidate["mean_tpot_ms"] / baseline["mean_tpot_ms"]
    eligible = (
        baseline["failed"] == 0
        and candidate["failed"] == 0
        and baseline["completed"] == candidate["completed"]
        and candidate["completed"] > 0
    )
    throughput_passed = throughput_ratio >= min_throughput_ratio
    tpot_passed = tpot_ratio <= max_tpot_ratio
    return {
        "passed": eligible and throughput_passed and tpot_passed,
        "eligible": eligible,
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "throughput": {
            "baseline": baseline["output_throughput"],
            "candidate": candidate["output_throughput"],
            "ratio": throughput_ratio,
            "minimum_ratio": min_throughput_ratio,
            "passed": throughput_passed,
        },
        "tpot": {
            "baseline_ms": baseline["mean_tpot_ms"],
            "candidate_ms": candidate["mean_tpot_ms"],
            "ratio": tpot_ratio,
            "maximum_ratio": max_tpot_ratio,
            "passed": tpot_passed,
        },
        "requests": {
            "baseline_completed": baseline["completed"],
            "candidate_completed": candidate["completed"],
            "baseline_failed": baseline["failed"],
            "candidate_failed": candidate["failed"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--min-throughput-ratio",
        type=float,
        default=DEFAULT_MIN_THROUGHPUT_RATIO,
    )
    parser.add_argument(
        "--max-tpot-ratio", type=float, default=DEFAULT_MAX_TPOT_RATIO
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = compare_results(
        args.baseline,
        args.candidate,
        args.min_throughput_ratio,
        args.max_tpot_ratio,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
