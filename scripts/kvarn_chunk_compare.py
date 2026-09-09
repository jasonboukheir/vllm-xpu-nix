"""Audit chunk-budget trials within one cache dtype; never certify model state."""

import argparse
import math
import re
from pathlib import Path

from scripts import kvarn_mtp_compare as mtp
from scripts import kvarn_prefill_compare as prefill
from scripts import kvarn_vision_compare as vision
from scripts.kvarn_perf_run import write_json_atomic
from scripts.kvarn_prefill_chunks import chunk_plan, recorded_budget
from scripts.kvarn_scan_engine_log import scan


def canonical_arguments(manifest):
    argv = list(manifest["actual_argv"])
    recorded_budget(manifest)
    argv[argv.index("--max-num-batched-tokens") + 1] = "<CHUNK_BUDGET>"
    return argv


def preemptions(directory):
    samples = re.findall(
        r"^vllm:num_preemptions_total(?:\{[^\n]*\})? ([^\s]+)$",
        (directory / "metrics.txt").read_text(),
        re.MULTILINE,
    )
    if not samples or any(not math.isfinite(float(s)) or float(s) < 0 for s in samples):
        raise ValueError("missing or invalid preemption evidence")
    return sum(float(s) for s in samples)


def compare(pairs):
    rows = []
    services = []
    baseline = None
    budgets = None
    seen = set()
    for control, candidate in pairs:
        control, candidate = control.resolve(), candidate.resolve()
        if control in seen or candidate in seen or control == candidate:
            raise ValueError("each service start must have a distinct capture")
        seen.update((control, candidate))
        left, lr = vision.audit(control)
        right, rr = vision.audit(candidate)
        current = (recorded_budget(left), recorded_budget(right))
        if current[0] != 2048 or current[1] not in (4096, 8192):
            raise ValueError("expected 2K control and 4K or 8K candidate")
        if budgets is not None and budgets != current:
            raise ValueError("candidate budget changed between pairs")
        budgets = current
        for manifest in (left, right):
            if (
                manifest.get("suite") != "text-prefill-performance-v1"
                or manifest["profiler_config"] is not None
                or manifest["speculative_config"] is not None
            ):
                raise ValueError("requires profiler-off, MTP-off text-prefill trials")
            identity = {
                "runtime": manifest["runtime_identity"],
                "harness": manifest["harness_sha256"],
                "images": manifest["image_sha256"],
                "arguments": canonical_arguments(manifest),
                "environment": {
                    k: v
                    for k, v in manifest["actual_environment"].items()
                    if k != "VLLM_CACHE_ROOT"
                },
            }
            if baseline is not None and baseline != identity:
                raise ValueError(
                    "runtime, dtype, harness, or non-budget settings changed"
                )
            baseline = identity
        for directory, manifest in ((control, left), (candidate, right)):
            outputs = []
            for case in vision.load(directory / "workload.json"):
                coverage = case["coverage"]
                expected = chunk_plan(
                    coverage["prompt_tokens"], recorded_budget(manifest)
                )
                if any(coverage.get(k) != v for k, v in expected.items()):
                    raise ValueError(
                        "workload coverage disagrees with actual chunk budget"
                    )
                outputs.append(mtp.tokens(directory, case["id"])[1])
            services.append(
                {
                    "path": str(directory),
                    "budget": recorded_budget(manifest),
                    "within_service_generated_tokens_equal": all(
                        output == outputs[0] for output in outputs
                    ),
                    "preemptions": preemptions(directory),
                    "memory": vision.memory_summary(directory),
                    "engine_log": scan(
                        (directory / "service.log").read_text().splitlines()
                    ),
                }
            )
        for a, b in zip(lr, rr, strict=True):
            name = a["id"]
            if name != b["id"] or a["phase"] != b["phase"]:
                raise ValueError("mismatched request order")
            if vision.load(control / f"{name}-request.json") != vision.load(
                candidate / f"{name}-request.json"
            ):
                raise ValueError("mismatched requests")
            ap, at = mtp.tokens(control, name)
            bp, bt = mtp.tokens(candidate, name)
            if ap != bp:
                raise ValueError("processed prompt IDs changed")
            for result in (a, b):
                if (
                    result["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                ):
                    raise ValueError("prefix reuse invalidates comparison")
            rows.append(
                {
                    "id": name,
                    "phase": a["phase"],
                    "prompt_tokens": len(ap),
                    "control_path": str(control),
                    "candidate_path": str(candidate),
                    "same_generated_tokens": at == bt,
                    "first_divergent_token": next(
                        (i for i, (x, y) in enumerate(zip(at, bt)) if x != y), None
                    ),
                    **{
                        label: {
                            "ttft_seconds": result["ttft_seconds"],
                            "total_seconds": result["total_seconds"],
                            **prefill.token_timing(directory, result),
                        }
                        for label, directory, result in (
                            ("control", control, a),
                            ("candidate", candidate, b),
                        )
                    },
                }
            )
    if not rows:
        raise ValueError("no comparison samples")
    by_length = {}
    for length in sorted({r["prompt_tokens"] for r in rows}):
        selected = [
            r
            for r in rows
            if r["prompt_tokens"] == length and r["phase"] == "performance"
        ]
        by_length[str(length)] = {
            arm: {
                metric: prefill.distribution([r[arm][metric] for r in selected])
                for metric in (
                    "ttft_seconds",
                    "total_seconds",
                    "decode_tokens_per_second",
                )
            }
            for arm in ("control", "candidate")
        }
    return {
        "schema": "kvarn-chunk-comparison-v1",
        "budgets": budgets,
        "matched_identity": baseline,
        "by_length": by_length,
        "services": services,
        "requests": rows,
        "all_generated_tokens_equal": all(r["same_generated_tokens"] for r in rows),
        "zero_preemptions": all(s["preemptions"] == 0 for s in services),
        "clean_engine_logs": all(
            s["engine_log"]["status"] == "passed" for s in services
        ),
        "qualified": False,
        "limitations": [
            "Descriptive screen; apply the predeclared paired-start performance gates separately.",
            "Token agreement does not certify ordinary attention, GDN, or packed history/tail state.",
            "DRM samples measure allocation/residency, not instantaneous torch allocator peaks.",
            "Full numerical/state/replay gates and then MTP2/images are required before recommendation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, compare(args.pair))


if __name__ == "__main__":
    main()
