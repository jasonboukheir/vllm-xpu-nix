"""Predict full attention-call cost from six anchors and observed call extents.

This is an operator prediction, not a model-service timing. Interpolate latency
linearly in K for full query chunks; normalize the 2047-row anchor to 2048 rows.
Use the two ragged final chunks exactly. Every observed layer call contributes.
"""

import argparse
import itertools
import json
from pathlib import Path


def predict(calls, anchors, arm):
    exact = {(a["m"], a["k"]): a[f"{arm}_ms"] for a in anchors}
    full = sorted(
        (a["k"], a[f"{arm}_ms"] * 2048 / a["m"], (a["m"], a["k"]))
        for a in anchors
        if a["m"] in (2047, 2048)
    )
    total = 0.0
    interpolated = 0
    contributing = set()
    for call in calls:
        key = (call["m"], call["k"])
        if key in exact:
            total += exact[key]
            contributing.add(key)
            continue
        if call["m"] != 2048:
            raise ValueError(f"unmeasured ragged extent: {key}")
        for (lo_k, lo_ms, lo_key), (hi_k, hi_ms, hi_key) in itertools.pairwise(full):
            if lo_k <= call["k"] <= hi_k:
                fraction = (call["k"] - lo_k) / (hi_k - lo_k)
                total += lo_ms + fraction * (hi_ms - lo_ms)
                if fraction < 1:
                    contributing.add(lo_key)
                if fraction > 0:
                    contributing.add(hi_key)
                interpolated += 1
                break
        else:
            raise ValueError(f"extrapolation forbidden: {key}")
    return {
        "ms": total,
        "calls": len(calls),
        "interpolated_calls": interpolated,
        "contributing_anchors": [
            {"m": m, "k": k} for m, k in sorted(contributing, key=lambda key: key[1])
        ],
    }


def anchor_quality(prediction, anchors):
    """Report stability only for anchors with nonzero prediction weight."""
    lookup = {(a["m"], a["k"]): a for a in anchors}
    used = [lookup[(a["m"], a["k"])] for a in prediction["contributing_anchors"]]
    if not used:
        raise ValueError("prediction has no contributing anchors")
    return {
        "all_anchors_stable": all(a["warmup_stable"] for a in used),
        "largest_control_cv": max(a["control_cv"] for a in used),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if report["status"] != "measured":
        raise ValueError("timing run is incomplete")
    predictions = []
    manifests = sorted({row["manifest"] for row in report["timings"]})
    for manifest in manifests:
        capture = json.loads(Path(manifest).read_text())
        for start in (0, 1):
            for condition in ("warm", "cold", "interleaved"):
                rows = [
                    r
                    for r in report["timings"]
                    if r["manifest"] == manifest
                    and r["start"] == start
                    and r["condition"] == condition
                ]
                if len(rows) != 6:
                    raise ValueError("requires all six screen anchors")
                anchors = [
                    {
                        **row,
                        "m": capture["captures"][row["index"]]["m"],
                        "k": capture["captures"][row["index"]]["k"],
                    }
                    for row in rows
                ]
                for request in (0, 1):
                    calls = [c for c in capture["calls"] if c["request"] == request]
                    control = predict(calls, anchors, "control")
                    candidate = predict(calls, anchors, "candidate")
                    predictions.append(
                        {
                            "cache_dtype": capture["cache_dtype"],
                            "start": start,
                            "condition": condition,
                            "request": request,
                            "prompt_tokens": len(
                                capture["requests"][request]["prompt_token_ids"]
                            ),
                            "control": control,
                            "candidate": candidate,
                            "latency_change_percent": 100
                            * (candidate["ms"] / control["ms"] - 1),
                            **anchor_quality(control, anchors),
                        }
                    )
    args.output.write_text(
        json.dumps(
            {
                "method": __doc__,
                "source_report": str(args.report),
                "predictions": predictions,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
