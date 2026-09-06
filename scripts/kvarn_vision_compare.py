#!/usr/bin/env python3
"""Verify matched vision captures and summarize their bounded evidence."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import kvarn_perf_run as perf


def load(path: Path):
    return json.loads(path.read_text())


def check_hash(path: Path, expected: str) -> None:
    if perf.sha256_file(path) != expected:
        raise ValueError(f"changed artifact: {path}")


def audit(directory: Path) -> tuple[dict, list[dict]]:
    manifest = load(directory / "manifest.json")
    if manifest["status"] != "requests-completed-not-yet-qualified":
        raise ValueError(f"capture did not finish: {directory}")
    check_hash(directory / "service.log", manifest["service_log_sha256"])
    check_hash(directory / "workload.json", manifest["workload_sha256"])
    for name, digest in manifest["harness_sha256"].items():
        check_hash(directory / "harness-source" / name, digest)
    for name, digest in manifest["image_sha256"].items():
        check_hash(directory / "images" / name, digest)
    if any(
        k.startswith("KVARN_") and v is not None
        for k, v in manifest["actual_environment"].items()
    ):
        raise ValueError("capture overrode dtype-selected KVarN defaults")
    cases = load(directory / "workload.json")
    results = []
    for case in cases:
        result = load(directory / f"{case['id']}-response.json")
        request = load(directory / f"{case['id']}-request.json")
        expected = {
            "model": "sunny-vision",
            "messages": case["messages"],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        expected.update(case.get("generation", {}))
        if request != expected:
            raise ValueError(f"request does not match workload: {case['id']}")
        if not result["stream_valid"] or not result["stream_done"]:
            raise ValueError(f"invalid stream: {case['id']}")
        if result["phase"] == "correctness":
            terms_ok = all(
                term.lower() in result["content"].lower()
                for term in case["expected_terms"]
            )
            regex_ok = not case.get("expected_regex") or re.search(
                case["expected_regex"], result["content"]
            )
            if not terms_ok or not regex_ok:
                raise ValueError(f"failed semantic term check: {case['id']}")
        content = ""
        usage = None
        done = False
        for line in (directory / f"{case['id']}-sse.jsonl").read_text().splitlines():
            raw = json.loads(line)["line"]
            if raw.strip() == "data: [DONE]":
                done = True
            elif raw.startswith("data: "):
                event = json.loads(raw[6:])
                if "error" in event:
                    raise ValueError("error in raw stream")
                usage = event.get("usage") or usage
                content += "".join(
                    choice.get("delta", {}).get("content") or ""
                    for choice in event.get("choices", [])
                )
        if not done or content != result["content"] or usage != result["usage"]:
            raise ValueError(f"response disagrees with raw stream: {case['id']}")
        if case.get("coverage"):
            tokenized = load(directory / f"{case['id']}-tokenize.json")
            positions = [
                i for i, token in enumerate(tokenized["tokens"]) if token == 248056
            ]
            if (
                not result["coverage_valid"]
                or tokenized["count"] != 6143
                or len(tokenized["tokens"]) != 6143
                or min(positions) <= 128
                or max(positions) >= 2048
            ):
                raise ValueError(f"invalid compression/page coverage: {case['id']}")
        if (
            result["phase"] == "performance"
            and result["usage"]["completion_tokens"] != 96
        ):
            raise ValueError("performance capture did not produce 96 tokens")
        results.append(result)
    return manifest, results


def memory_summary(directory: Path) -> dict:
    samples = [
        json.loads(line)
        for line in (directory / "memory-fdinfo.jsonl").read_text().splitlines()
    ]
    peaks = {}
    shared = {}
    errors = []
    for sample in samples:
        totals = {}
        for client in sample["clients"].values():
            fields = client["fields"]
            device = fields["drm-pdev"]
            for name in ("drm-resident-vram0", "drm-total-vram0", "drm-shared-vram0"):
                raw = fields.get(name)
                if raw is None:
                    continue
                parts = raw.split()
                if len(parts) == 2 and parts[1] == "KiB":
                    value = int(parts[0]) * 1024
                elif raw == "0":
                    value = 0
                else:
                    raise ValueError(f"unsupported memory unit: {raw}")
                key = device + "/" + name
                totals[key] = totals.get(key, 0) + value
        for key, value in totals.items():
            peaks[key] = max(peaks.get(key, 0), value)
            if "shared" in key and value:
                shared[key] = value
        errors.extend(sample["errors"])
    if not peaks:
        raise ValueError("no actual DRM memory samples")
    if shared:
        raise ValueError("shared GPU buffers require de-duplicated memory accounting")
    return {
        "peak_bytes_by_device_and_field": peaks,
        "sample_count": len(samples),
        "poll_interval_seconds": 0.5,
        "errors": sorted(set(errors)),
        "limitation": "sampled peak of owned unique DRM clients, not an instantaneous allocator high-water mark",
    }


def canonical_argv(argv: list[str]) -> list[str]:
    result = list(argv)
    result[result.index("--kv-cache-dtype") + 1] = "<CACHE_DTYPE>"
    return result


def compare(auto: Path, kvarn: Path) -> dict:
    left, ar = audit(auto)
    right, kr = audit(kvarn)
    for key in (
        "harness_sha256",
        "workload_sha256",
        "image_sha256",
        "runtime_identity",
    ):
        if left[key] != right[key]:
            raise ValueError(f"mismatched {key}")
    if canonical_argv(left["actual_argv"]) != canonical_argv(right["actual_argv"]):
        raise ValueError("mismatched runtime arguments")
    for key in left["actual_environment"].keys() | right["actual_environment"].keys():
        if key == "VLLM_CACHE_ROOT":
            continue
        if left["actual_environment"].get(key) != right["actual_environment"].get(key):
            raise ValueError(f"mismatched runtime environment: {key}")
    rows = []
    for a, k in zip(ar, kr, strict=True):
        if a["id"] != k["id"]:
            raise ValueError("mismatched request order")
        name = a["id"]
        if load(auto / f"{name}-request.json") != load(kvarn / f"{name}-request.json"):
            raise ValueError(f"mismatched request: {name}")
        if a["usage"]["prompt_tokens"] != k["usage"]["prompt_tokens"]:
            raise ValueError(f"mismatched processed tokens: {name}")
        rows.append(
            {
                "id": name,
                "phase": a["phase"],
                "prompt_tokens": a["usage"]["prompt_tokens"],
                "auto_content": a["content"],
                "kvarn_content": k["content"],
                "same_text": a["content"] == k["content"],
                "auto_ttft_seconds": a["ttft_seconds"],
                "kvarn_ttft_seconds": k["ttft_seconds"],
            }
        )
    performance = {}
    for kind in ("image", "text"):
        arms = {}
        for label, results in (("auto", ar), ("kvarn", kr)):
            selected = [
                r
                for r in results
                if r["phase"] == "performance" and r["id"].startswith("perf-" + kind)
            ]
            if len(selected) != 3:
                raise ValueError(
                    "expected three warmed performance samples per workload"
                )
            arms[label] = {
                key: {
                    "median": statistics.median(r[key] for r in selected),
                    "samples": [r[key] for r in selected],
                }
                for key in ("ttft_seconds", "decode_tokens_per_second")
            }
        performance[kind] = arms
    startup = {}
    for label, directory in (("auto", auto), ("kvarn", kvarn)):
        lines = (directory / "service.log").read_text().splitlines()
        startup[label] = [
            line
            for line in lines
            if re.search(
                r"Model loading took|Available KV cache memory|GPU KV cache size|selected_.*|protected .*queued|fused_materialized",
                line,
            )
        ]
    return {
        "schema": "kvarn-vision-comparison-v1",
        "auto": str(auto),
        "kvarn": str(kvarn),
        "matched_inputs_verified": True,
        "rows": rows,
        "performance": performance,
        "memory": {"auto": memory_summary(auto), "kvarn": memory_summary(kvarn)},
        "startup_and_selectors": startup,
        "timing_scope": "three serial warmed requests per arm, 96 output tokens, profiler off; CPU fdinfo sampling on both arms; no statistical parity claim",
        "semantic_scope": "term checks plus manual review required; token equality is not required for lossy KV",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", type=Path, required=True)
    parser.add_argument("--kvarn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.auto.resolve(), args.kvarn.resolve())
    perf.write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("rows", "startup_and_selectors")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
