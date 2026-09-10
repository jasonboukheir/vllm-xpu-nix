"""Bootstrap weighted pure-prefill costs while retaining within-pair covariance.

Independently resample paired timing blocks at each contributing anchor. Keep
observed call counts and six-anchor interpolation coefficients fixed. Intervals
do not model cross-anchor covariance, interpolation error, or service overlap.
Fresh starts and cache conditions remain separate estimates.
"""

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

from scripts.xpu_attention_weight import anchor_quality, predict


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def coefficients(calls, anchors):
    """Obtain linear weights from the same interpolation used by the report."""
    return [
        predict(
            calls,
            [
                {"m": a["m"], "k": a["k"], "basis_ms": float(i == j)}
                for j, a in enumerate(anchors)
            ],
            "basis",
        )["ms"]
        for i in range(len(anchors))
    ]


def quantile(values, fraction):
    position = (len(values) - 1) * fraction
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (position - lo) * (values[hi] - values[lo])


def estimate(calls, anchors, *, draws, seed):
    if draws < 100:
        raise ValueError("at least 100 bootstrap draws required")
    weights = coefficients(calls, anchors)
    paired = []
    for anchor in anchors:
        blocks = []
        for block in anchor["blocks"]:
            if len(block) != 2 or sorted(s["arm"] for s in block) != [
                "candidate",
                "control",
            ]:
                raise ValueError("requires complete control/candidate pairs")
            values = {s["arm"]: s["ms"] for s in block}
            if min(values.values()) <= 0:
                raise ValueError("timings must be positive")
            blocks.append((values["control"], values["candidate"]))
        if not blocks:
            raise ValueError("anchor has no timing pairs")
        for arm, column in (("control", 0), ("candidate", 1)):
            mean = statistics.mean(b[column] for b in blocks)
            if abs(mean - anchor[f"{arm}_ms"]) > 1e-9:
                raise ValueError("anchor summary disagrees with raw timing pairs")
        paired.append(blocks)
    control = predict(calls, anchors, "control")
    candidate = predict(calls, anchors, "candidate")
    for column, prediction in ((0, control), (1, candidate)):
        weighted = sum(
            weight * statistics.mean(block[column] for block in blocks)
            for weight, blocks in zip(weights, paired, strict=True)
        )
        if abs(weighted - prediction["ms"]) > 1e-6:
            raise ValueError("linear coefficients disagree with existing prediction")
    rng = random.Random(seed)
    distribution = []
    for _ in range(draws):
        totals = [0.0, 0.0]
        for weight, blocks in zip(weights, paired, strict=True):
            if weight == 0:
                continue
            # Select each control/candidate tuple together, never separately.
            selected = rng.choices(blocks, k=len(blocks))
            for column in (0, 1):
                totals[column] += weight * statistics.mean(
                    pair[column] for pair in selected
                )
        distribution.append(100 * (totals[1] / totals[0] - 1))
    distribution.sort()
    return {
        "control": control,
        "candidate": candidate,
        "latency_change_percent": 100 * (candidate["ms"] / control["ms"] - 1),
        "latency_change_percent_ci95": [
            quantile(distribution, 0.025),
            quantile(distribution, 0.975),
        ],
        "saved_ms": control["ms"] - candidate["ms"],
        "anchor_coefficients": [
            {"m": a["m"], "k": a["k"], "coefficient": w}
            for a, w in zip(anchors, weights, strict=True)
        ],
        "draws": draws,
        "seed": seed,
        **anchor_quality(control, anchors),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=3000)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["status"] != "measured":
        raise ValueError("both timing starts must be complete")
    groups = sorted(
        {(r["manifest"], r["start"], r["condition"]) for r in report["timings"]}
    )
    if {r["start"] for r in report["timings"]} != {0, 1}:
        raise ValueError("requires both fresh starts")
    results = []
    for group_index, (manifest, start, condition) in enumerate(groups):
        capture = json.loads(Path(manifest).read_text())
        rows = sorted(
            (
                r
                for r in report["timings"]
                if (r["manifest"], r["start"], r["condition"])
                == (manifest, start, condition)
            ),
            key=lambda r: r["index"],
        )
        if [r["index"] for r in rows] != list(range(6)):
            raise ValueError("requires each of the six anchors exactly once")
        anchors = [
            {
                **row,
                "m": capture["captures"][row["index"]]["m"],
                "k": capture["captures"][row["index"]]["k"],
            }
            for row in rows
        ]
        for request_index, request in enumerate(capture["requests"]):
            calls = [c for c in capture["calls"] if c["request"] == request_index]
            results.append(
                {
                    "manifest": manifest,
                    "manifest_sha256": digest(manifest),
                    "start": start,
                    "condition": condition,
                    "request": request_index,
                    "prompt_tokens": len(request["prompt_token_ids"]),
                    **estimate(
                        calls,
                        anchors,
                        draws=args.draws,
                        seed=991100 + group_index * 10 + request_index,
                    ),
                }
            )
    args.output.write_text(
        json.dumps(
            {
                "method": __doc__,
                "source_report": str(args.report),
                "source_report_sha256": digest(args.report),
                "script_sha256": digest(__file__),
                "weighting_script_sha256": digest(
                    Path(__file__).with_name("xpu_attention_weight.py")
                ),
                "predictions": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
