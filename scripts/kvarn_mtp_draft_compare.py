"""Audit fixed-target MTP draft-cache comparisons without assuming a winner."""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import kvarn_lowbit_compare as lowbit
from scripts import kvarn_perf_run as perf
from scripts.kvarn_mtp_compare import speculative_counters

DTYPES = {
    "k4v2": "kvarn_k4v2_g128_compact",
    "k4v4": "kvarn_k4v4_g128_compact",
    "bf16": "bfloat16",
}


def acceptance(counters, output_tokens, requests):
    prefix = "vllm:spec_decode_"
    fields = (
        "num_drafts_total",
        "num_draft_tokens_total",
        "num_accepted_tokens_total",
        "num_accepted_tokens_per_pos_total/position=0",
        "num_accepted_tokens_per_pos_total/position=1",
    )
    if set(counters) != {prefix + name for name in fields}:
        raise ValueError("missing or unexpected speculative counters")
    values = [counters[prefix + name] for name in fields]
    if any(not math.isfinite(v) or v < 0 or int(v) != v for v in values):
        raise ValueError("invalid/reset speculative counters")
    rounds, proposed, accepted, first, second = map(int, values)
    if not (0 < rounds <= proposed <= 2 * rounds):
        raise ValueError("invalid MTP2 proposal denominators")
    full = proposed - rounds
    trimmed = rounds - full
    if accepted != first + second or not 0 <= second <= min(first, full) <= rounds:
        raise ValueError("accepted position counts disagree")
    if first > rounds:
        raise ValueError("more first acceptances than rounds")
    # The first token of each request comes from prefill. Each verification
    # emits accepted drafts plus one target token, capped by the output budget.
    clipped = requests + rounds + accepted - output_tokens
    if not 0 <= clipped <= 2 * requests:
        raise ValueError("verification work does not explain emitted output")
    denominator_low = max(second, first - trimmed)
    denominator_high = min(first, full)
    conditional = None
    if denominator_high:
        conditional = [
            second / denominator_high,
            second / denominator_low if denominator_low else 1.0,
        ]
    return {
        "rounds": rounds,
        "proposed_tokens": proposed,
        "accepted_tokens": accepted,
        "accepted_first": first,
        "accepted_second": second,
        "two_token_rounds": full,
        "one_token_rounds": trimmed,
        "output_budget_clipped_tokens": clipped,
        "first_acceptance": first / rounds,
        "both_acceptance_per_two_token_round": second / full if full else None,
        "conditional_second_acceptance_bounds": conditional,
        "conditional_second_denominator_bounds": [denominator_low, denominator_high],
        "accepted_tokens_per_round": accepted / rounds,
        "accepted_fraction_of_proposals": accepted / proposed,
        "note": "Conditional-second bounds account for first acceptances on budget-trimmed one-token rounds; equal bounds identify the exact rate.",
    }


def pools(directory, manifest):
    rows = []
    for line in (directory / "service.log").read_text().splitlines():
        match = re.search(r"KV cache pool (\d+):", line)
        if not match:
            continue
        row = {"id": int(match[1])}
        for key in ("types", "page_sizes", "block_sizes"):
            row[key] = ast.literal_eval(
                re.search(r"\b" + key + r"=(\([^)]*\))", line)[1]
            )
        for key in (
            "layers",
            "physical_blocks",
            "null_blocks",
            "usable_blocks",
            "allocated",
            "max_request_blocks",
        ):
            row[key] = int(
                re.search(r"\b" + key + r"=([0-9,]+)", line)[1].replace(",", "")
            )
        if row["physical_blocks"] != row["null_blocks"] + row["usable_blocks"]:
            raise ValueError("cache pool null/usable accounting differs")
        if len(row["page_sizes"]) != 1 or len(row["block_sizes"]) != 1:
            raise ValueError("unexpected heterogeneous cache pool")
        if (
            row["allocated"]
            != row["physical_blocks"] * row["layers"] * row["page_sizes"][0]
        ):
            raise ValueError("cache byte accounting differs")
        rows.append(row)
    attention = [p for p in rows if p["types"] == ("FullAttentionSpec",)]
    recurrent = [p for p in rows if p["types"] == ("MambaSpec",)]
    actual_pages = {}
    for p in attention:
        page = p["page_sizes"][0]
        actual_pages[page] = actual_pages.get(page, 0) + p["layers"]
    expected_pages = {107520: 16}
    spec = manifest["plan"]["draft_cache"]
    if spec:
        page = {DTYPES["k4v2"]: 107520, DTYPES["k4v4"]: 140288, DTYPES["bf16"]: 524288}[
            spec
        ]
        expected_pages[page] = expected_pages.get(page, 0) + 1
    if actual_pages != expected_pages:
        raise ValueError("target/draft physical page or layer count differs")
    if len(recurrent) != 1 or recurrent[0]["layers"] != 48:
        raise ValueError("unexpected recurrent layer layout")
    if recurrent[0]["usable_blocks"] != recurrent[0]["max_request_blocks"] * 4:
        raise ValueError("four-slot recurrent reservation lost")
    capacities = {p["usable_blocks"] * p["block_sizes"][0] for p in attention}
    if len(capacities) != 1:
        raise ValueError("target/draft capacities differ")
    return {
        "pools": rows,
        "usable_tokens": capacities.pop(),
        "paged_allocated_bytes": sum(p["allocated"] for p in attention),
        "recurrent_allocated_bytes": recurrent[0]["allocated"],
    }


def audit_arm(directory, plan_path):
    plan = json.loads(plan_path.read_text())
    manifest, groups, memory = lowbit.audit(
        directory, plan, perf.sha256_file(plan_path)
    )
    if manifest["cache_dtype"] != DTYPES["k4v2"]:
        raise ValueError("target cache changed")
    previous = None
    streams = {}
    for wave in manifest["waves"]:
        name = wave["id"]
        before = speculative_counters(directory / f"{name}-metrics-before.txt")
        after = speculative_counters(directory / f"{name}-metrics-after.txt")
        if previous is not None and before != previous:
            raise ValueError("speculative counters changed between isolated waves")
        previous = after
        counts = {k: after[k] - before[k] for k in before}
        measured = (
            acceptance(
                counts, wave["concurrency"] * plan["output_tokens"], wave["concurrency"]
            )
            if plan["draft_cache"]
            else None
        )
        streams[name] = {
            "acceptance": measured,
            "tokens": [
                json.loads((directory / f"{request}-response.json").read_text())[
                    "token_ids"
                ]
                for request in wave["request_ids"]
            ],
        }
    allocation = pools(directory, manifest)
    if allocation["usable_tokens"] != plan["common_usable_attention_tokens"]:
        raise ValueError("matched capacity differs")
    return {
        "directory": str(directory),
        "manifest_sha256": perf.sha256_file(directory / "manifest.json"),
        "manifest": manifest,
        "groups": groups,
        "memory": memory,
        "allocation": allocation,
        "streams": streams,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--captures", required=True, nargs=6, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    root = args.protocol.parent
    arms = []
    previous = None
    for label, directory in zip(protocol["arm_order"], args.captures, strict=True):
        plan = root / f"serving-{label}-plan.json"
        if perf.sha256_file(plan) != protocol["plans"][plan.name]:
            raise ValueError("frozen plan changed")
        arm = audit_arm(directory, plan)
        m = arm["manifest"]
        if (
            m["plan"]["draft_cache"] != DTYPES[label]
            or m["runtime"] != protocol["runtime"]
        ):
            raise ValueError("draft format or runtime differs")
        if not m["started_unix"] < m["finished_unix"] or (
            previous and previous["finished_unix"] >= m["started_unix"]
        ):
            raise ValueError("overlapping/reordered fresh starts")
        if previous and any(
            previous[k] != m[k] for k in ("runtime_identity", "harness_sha256")
        ):
            raise ValueError("runtime/harness changed between formats")
        previous = m
        arm["label"] = label
        del arm["manifest"]
        arms.append(arm)
    statistics_by_format = {}
    for label in DTYPES:
        samples = [arm for arm in arms if arm["label"] == label]
        if len(samples) != 2:
            raise ValueError("requires two fresh starts per format")
        statistics_by_format[label] = {
            trial: {
                field: {
                    "median": statistics.median(
                        values := [
                            row[field]
                            for arm in samples
                            for row in arm["groups"][trial]
                        ]
                    ),
                    "min": min(values),
                    "max": max(values),
                    "fresh_start_medians": [
                        statistics.median(row[field] for row in arm["groups"][trial])
                        for arm in samples
                    ],
                }
                for field in ("decode", "throughput", "ttft", "latency")
            }
            for trial in samples[0]["groups"]
        }
    report = {
        "status": "audited-captures-require-quality-and-trajectory-review",
        "protocol_sha256": perf.sha256_file(args.protocol),
        "arms": arms,
        "statistics": statistics_by_format,
        "scope": "Real free-running MTP acceptance and profiler-off serving timings. Matched-history prediction agreement is a separate diagnostic. No winner selected by this script.",
    }
    perf.write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
