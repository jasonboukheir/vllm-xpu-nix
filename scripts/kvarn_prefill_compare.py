"""Audit matched profiler-off text-prefill trials and token arrival evidence."""

import argparse
import json
import math
import statistics
from itertools import pairwise
from pathlib import Path

from scripts import kvarn_mtp_compare as mtp
from scripts import kvarn_vision_compare as vision
from scripts.kvarn_perf_run import write_json_atomic


def distribution(values):
    values = sorted(values)
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("missing or non-finite timing samples")
    return {
        "samples": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "p99": values[math.ceil(0.99 * len(values)) - 1],
    }


def token_timing(directory, result):
    arrivals = []
    for line in (directory / f"{result['id']}-sse.jsonl").read_text().splitlines():
        record = json.loads(line)
        raw = record["line"]
        if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
            continue
        for choice in json.loads(raw[6:]).get("choices", []):
            ids = choice.get("token_ids") or []
            if ids:
                arrivals.append((record["seconds"], len(ids)))
    if sum(n for _, n in arrivals) != result["usage"]["completion_tokens"]:
        raise ValueError("incomplete token timing evidence")
    if any(n != 1 for _, n in arrivals):
        raise ValueError(
            "bundled token events cannot establish individual inter-token latency"
        )
    gaps = [b[0] - a[0] for a, b in pairwise(arrivals)]
    if any(gap < 0 for gap in gaps):
        raise ValueError("nonmonotonic token arrivals")
    return {
        "first_token_seconds": arrivals[0][0],
        "decode_tokens_per_second": (len(arrivals) - 1)
        / (arrivals[-1][0] - arrivals[0][0]),
        "inter_token_seconds": distribution(gaps),
    }


def compare(pairs):
    rows = []
    identity = None
    baseline_arguments = None
    for auto, kvarn in pairs:
        left, ar = vision.audit(auto)
        right, kr = vision.audit(kvarn)
        for manifest, dtype in ((left, "auto"), (right, "kvarn_k4v4_g128_compact")):
            argv = manifest["actual_argv"]
            if argv[argv.index("--kv-cache-dtype") + 1] != dtype:
                raise ValueError("pair must contain auto then compact KVarN")
        for manifest in (left, right):
            if (
                manifest.get("suite") != "text-prefill-performance-v1"
                or manifest["profiler_config"] is not None
            ):
                raise ValueError(
                    "service comparison requires profiler-off text-prefill captures"
                )
            if manifest["speculative_config"] is not None:
                raise ValueError("prefill comparison requires MTP off")
            if identity is None:
                identity = manifest["runtime_identity"]
            elif identity != manifest["runtime_identity"]:
                raise ValueError("runtime identity changed between trials")
        for key in ("harness_sha256", "workload_sha256", "image_sha256"):
            if left[key] != right[key]:
                raise ValueError(f"mismatched {key}")
        if vision.canonical_argv(left["actual_argv"]) != vision.canonical_argv(
            right["actual_argv"]
        ):
            raise ValueError("mismatched runtime arguments")
        arguments = vision.canonical_argv(left["actual_argv"])
        if baseline_arguments is None:
            baseline_arguments = arguments
        elif arguments != baseline_arguments:
            raise ValueError("runtime arguments changed between trials")
        for key in (
            left["actual_environment"].keys() | right["actual_environment"].keys()
        ):
            if key != "VLLM_CACHE_ROOT" and left["actual_environment"].get(
                key
            ) != right["actual_environment"].get(key):
                raise ValueError(f"mismatched environment: {key}")
        for a, k in zip(ar, kr, strict=True):
            if a["phase"] == "warmup":
                continue
            name = a["id"]
            if (
                name != k["id"]
                or a["phase"] != "performance"
                or k["phase"] != "performance"
            ):
                raise ValueError("mismatched measured request order")
            if vision.load(auto / f"{name}-request.json") != vision.load(
                kvarn / f"{name}-request.json"
            ):
                raise ValueError("mismatched request")
            ap, at = mtp.tokens(auto, name)
            kp, kt = mtp.tokens(kvarn, name)
            if ap != kp:
                raise ValueError("mismatched processed prompt IDs")
            for result in (a, k):
                if (
                    result["usage"]
                    .get("prompt_tokens_details", {})
                    .get("cached_tokens", 0)
                ):
                    raise ValueError("prefix reuse invalidates full-prefill comparison")
            rows.append(
                {
                    "auto_path": str(auto),
                    "kvarn_path": str(kvarn),
                    "id": name,
                    "prompt_tokens": len(ap),
                    "same_generated_tokens": at == kt,
                    **{
                        label: {
                            "ttft_seconds": result["ttft_seconds"],
                            "total_seconds": result["total_seconds"],
                            **token_timing(directory, result),
                        }
                        for label, directory, result in (
                            ("auto", auto, a),
                            ("kvarn", kvarn, k),
                        )
                    },
                }
            )
    by_length = {}
    for length in sorted({row["prompt_tokens"] for row in rows}):
        selected = [r for r in rows if r["prompt_tokens"] == length]
        by_length[str(length)] = {
            arm: {
                key: distribution([r[arm][key] for r in selected])
                for key in ("ttft_seconds", "total_seconds", "decode_tokens_per_second")
            }
            for arm in ("auto", "kvarn")
        }
        by_length[str(length)]["paired_ttft_delta_seconds"] = distribution(
            [r["kvarn"]["ttft_seconds"] - r["auto"]["ttft_seconds"] for r in selected]
        )
    return {
        "schema": "kvarn-prefill-comparison-v1",
        "runtime_identity": identity,
        "by_length": by_length,
        "requests": rows,
        "limitations": [
            "Descriptive matched samples, not an independence or significance claim.",
            "Inter-token latency measures client arrivals, including transport buffering.",
            "Generated-token equality is reported, not required between different KV dtypes.",
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
