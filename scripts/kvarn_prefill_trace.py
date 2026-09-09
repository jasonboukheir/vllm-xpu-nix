"""Qualify full-prefill captures from the current service runner and retain slices."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts import kvarn_vision_compare as vision
from scripts.kvarn_perf_run import write_json_atomic
from scripts.kvarn_xpu_profile import STEP_PATTERN, load_trace
from scripts.kvarn_xpu_profile_compare import family
from scripts.kvarn_xpu_trace import analyze_attribution


def validate_steps(steps, prompt_tokens, chunk_size=2048):
    count = math.ceil(prompt_tokens / chunk_size)
    if len(steps) != count + 2:
        raise ValueError("trace lacks the full prefill plus two boundary guard steps")
    for index, step in enumerate(steps[:count]):
        match = STEP_PATTERN.fullmatch(step["annotation"])
        if match is None:
            raise ValueError("unrecognized execution annotation")
        values = {k: int(v) for k, v in match.groupdict().items()}
        expected_query = min(chunk_size, prompt_tokens - index * chunk_size)
        if (
            values["context_requests"] != 1
            or values["generation_requests"] != 0
            or values["context_tokens"] != expected_query
            or values["context_sk"] != min((index + 1) * chunk_size, prompt_tokens)
        ):
            raise ValueError("prefill chunk lengths do not cover the exact request")
    for step in steps[count:]:
        match = STEP_PATTERN.fullmatch(step["annotation"])
        if (
            match is None
            or int(match["generation_requests"]) != 1
            or int(match["context_requests"]) != 0
        ):
            raise ValueError("missing prefill-to-decode boundary")
    return count


def analyze(directory):
    manifest, requests = vision.audit(directory)
    if manifest.get("suite") != "text-prefill-profile-v1":
        raise ValueError("expected a full text-prefill diagnostic capture")
    traces = manifest.get("trace_sha256", {})
    if len(traces) != 1:
        raise ValueError("expected exactly one hashed worker trace")
    name, digest = next(iter(traces.items()))
    vision.check_hash(directory / name, digest)
    document = load_trace(directory / name)
    inputs = {}
    for event in document["traceEvents"]:
        if event.get("name") not in (
            "_vllm_fa2_C::varlen_fwd",
            "_vllm_fa2_C::kvarn_materialize_packed_kv",
            "_vllm_fa2_C::kvarn_pack_balanced_kv",
        ):
            continue
        args = {
            k: v
            for k, v in event.get("args", {}).items()
            if k.startswith("Input") or k == "Concrete Inputs"
        }
        key = (event["name"], json.dumps(args, sort_keys=True))
        row = inputs.setdefault(
            key, {"operator": event["name"], "inputs": args, "count": 0}
        )
        row["count"] += 1
    full = analyze_attribution(document)
    first_flush = None
    sinkhorn = [
        e
        for e in document["traceEvents"]
        if e.get("cat") == "kernel" and "_sinkhorn_log_kernel" in e.get("name", "")
    ]
    if sinkhorn:
        kernel = min(sinkhorn, key=lambda e: e["ts"])
        origins = [
            e
            for e in document["traceEvents"]
            if e.get("cat") == "xpu_runtime"
            and e.get("args", {}).get("correlation") == kernel["args"]["correlation"]
        ]
        if len(origins) != 1:
            raise ValueError("first Sinkhorn launch lacks a unique runtime origin")
        origin = origins[0]
        owners = [
            e
            for e in document["traceEvents"]
            if e.get("name", "").startswith("execute_")
            and (e.get("pid"), e.get("tid")) == (origin.get("pid"), origin.get("tid"))
            and e["ts"] <= origin["ts"]
            and e["ts"] + e["dur"] >= origin["ts"] + origin["dur"]
        ]
        if len(owners) != 1:
            raise ValueError("first Sinkhorn launch lacks a unique execution owner")
        first_flush = next(
            s["step"] for s in full["steps"] if s["annotation"] == owners[0]["name"]
        )
    tokens = requests[0]["usage"]["prompt_tokens"]
    count = validate_steps(full["steps"], tokens)
    if full["coverage"]["unresolved_host_origins"]:
        raise ValueError("trace has unresolved submitting CPU origins")
    result = {
        "schema": "kvarn-full-prefill-attribution-v1",
        "diagnostic_only": True,
        "trace_sha256": digest,
        "runtime_identity": manifest["runtime_identity"],
        "prompt_tokens": tokens,
        "output_tokens": requests[0]["usage"]["completion_tokens"],
        "coverage": full["coverage"],
        "operator_inputs": list(inputs.values()),
        "first_sinkhorn_step": first_flush,
        "slices": {},
        "profiler_bracket": {
            "before": requests[1]["ttft_seconds"],
            "profiled": requests[2]["ttft_seconds"],
            "after": requests[3]["ttft_seconds"],
            "profiled_over_bracket_mean": requests[2]["ttft_seconds"]
            / ((requests[1]["ttft_seconds"] + requests[3]["ttft_seconds"]) / 2),
        },
        "limitations": [
            "Device family sums, GPU unions and CPU waits are overlapping diagnostics, not additive savings.",
            "The trace covers the worker, not frontend or scheduler time outside worker execution.",
            "Uncovered time and CPU self time do not establish removable host overhead.",
        ],
    }
    slices = [
        ("all-prefill", 1, count),
        ("initial", 1, min(2, count)),
        ("early", 1, min(4, count)),
        ("middle", max(1, count // 2 - 1), min(count, count // 2 + 2)),
        ("late", max(1, count - 3), count),
        ("boundary", count, count + 2),
    ]
    if first_flush is not None:
        slices.append(("first-flush", first_flush, first_flush))
    for label, first, last in slices:
        attribution = analyze_attribution(document, first_step=first, last_step=last)
        groups = defaultdict(float)
        for row in attribution["device_operations"]:
            groups[family(row["device_operation"], row["kind"])] += (
                row["device_duration_sum_us"] / 1000
            )
        result["slices"][label] = {
            "family_duration_sum_ms": dict(groups),
            "attribution": attribution,
        }
    vision.check_hash(directory / name, digest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json_atomic(args.output, analyze(args.run))


if __name__ == "__main__":
    main()
