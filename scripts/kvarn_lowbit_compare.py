#!/usr/bin/env python3
"""Audit a frozen native K4V4/K4V2 ABBA serving capture and report each cell."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import kvarn_perf_run as perf
from scripts import kvarn_service_gate as gate
from scripts.kvarn_lowbit_service import compilation_events
from scripts.kvarn_mtp_compare import speculative_counters
from scripts.kvarn_vision_compare import check_hash, load, memory_summary


def close(actual, expected):
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9):
        raise ValueError(f"derived measurement disagrees: {actual} != {expected}")


def audit_mtp_activity(counters, server_args):
    if any(not math.isfinite(v) or v < 0 or int(v) != v for v in counters.values()):
        raise ValueError("speculative counters invalid or reset")
    if "--speculative-config" in server_args:
        index = server_args.index("--speculative-config")
        config = json.loads(server_args[index + 1])
        matching = {"method": "mtp", "num_speculative_tokens": 2}
        configurations = [matching] + [
            {**matching, "kv_cache_dtype": dtype}
            for dtype in (
                "kvarn_k4v4_g128_compact",
                "kvarn_k4v2_g128_compact",
                "bfloat16",
            )
        ]
        prefix = "vllm:spec_decode_"
        drafts = counters.get(prefix + "num_drafts_total", 0)
        tokens = counters.get(prefix + "num_draft_tokens_total", 0)
        # Scheduler budgets can trim the last draft. Require actual two-token
        # activity while permitting those bounded one-token verification steps.
        if (
            config not in configurations
            or drafts <= 0
            or not drafts < tokens <= 2 * drafts
        ):
            raise ValueError("missing active two-token MTP evidence")
    elif any(counters.values()):
        raise ValueError("speculation active in an MTP-off capture")


def audit_stream(directory, name, expected):
    request = load(directory / f"{name}-request.json")
    result = load(directory / f"{name}-response.json")
    if request != expected:
        raise ValueError(f"request differs from frozen trial: {name}")
    check_hash(directory / f"{name}-request.json", result["request_sha256"])
    check_hash(directory / f"{name}-sse.jsonl", result["raw_stream_sha256"])
    tokens, content = [], ""
    first = last = None
    first_count = 0
    done = False
    finish = usage = None
    previous = 0
    for line in (directory / f"{name}-sse.jsonl").read_text().splitlines():
        raw = json.loads(line)
        seconds = raw["seconds"]
        if not math.isfinite(seconds) or seconds < previous:
            raise ValueError("non-monotonic stream timestamps")
        previous = seconds
        if raw["line"].strip() == "data: [DONE]":
            if done:
                raise ValueError("duplicate stream terminator")
            done = True
        elif raw["line"].startswith("data: "):
            if done:
                raise ValueError("payload after stream terminator")
            event = json.loads(raw["line"][6:])
            if "error" in event:
                raise ValueError("error in raw stream")
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                chunk = choice.get("token_ids") or []
                if chunk:
                    if first is None:
                        first, first_count = seconds, len(chunk)
                    last = seconds
                tokens.extend(chunk)
                content += choice.get("text") or ""
                finish = choice.get("finish_reason") or finish
    if (
        not done
        or not result["stream_done"]
        or finish != "length"
        or finish != result["finish_reason"]
        or tokens != result["token_ids"]
        or content != result["content"]
        or usage != result["usage"]
        or len(tokens) != expected["max_tokens"]
        or first is None
        or last <= first
        or usage["prompt_tokens"] != len(expected["prompt"])
        or usage["completion_tokens"] != len(tokens)
        or usage["total_tokens"] != len(tokens) + len(expected["prompt"])
        or first_count != result["first_chunk_tokens"]
        or not math.isfinite(result["elapsed_seconds"])
        or result["elapsed_seconds"] < previous
    ):
        raise ValueError(f"invalid or incomplete stream: {name}")
    close(result["ttft_seconds"], first)
    close(result["last_token_seconds"], last)
    close(
        result["decode_tokens_per_second"], (len(tokens) - first_count) / (last - first)
    )
    findings = gate.completion_quality_findings(tokens, len(tokens))
    if findings != result["quality_findings"]:
        raise ValueError("changed quality findings")
    return result


def audit(directory, plan, plan_hash):
    manifest = load(directory / "manifest.json")
    if manifest["status"] != "captured-not-qualified":
        raise ValueError(f"capture incomplete: {directory}")
    if manifest["plan"] != plan or manifest["plan_sha256"] != plan_hash:
        raise ValueError("capture does not match frozen plan")
    check_hash(directory / "service.log", manifest["service_log_sha256"])
    service_log = (directory / "service.log").read_bytes()
    if (
        "--jit-monitor-verbose" not in plan["server_args"]
        or b"Kernel JIT monitor activated" not in service_log
    ):
        raise ValueError("missing complete runtime compilation monitoring")
    check_hash(directory / "memory-fdinfo.jsonl", manifest["memory_sha256"])
    for name, digest in manifest["harness_sha256"].items():
        check_hash(directory / "harness-source" / name, digest)
    argv = manifest["argv"]
    actual = manifest["actual_argv"]
    if actual[actual.index("serve") :] != argv[1:]:
        raise ValueError("running service arguments differ from planned launch")
    expected_tail = [
        "--served-model-name",
        "sunny-chat",
        "--kv-cache-dtype",
        manifest["cache_dtype"],
        *plan["server_args"],
    ]
    if argv[2:6] != [plan["model"], "--revision", plan["revision"], "--host"]:
        raise ValueError("service model identity differs")
    if argv[6] != "127.0.0.1" or argv[9:] != expected_tail:
        raise ValueError("unexpected service flags")
    environment = manifest["actual_environment"]
    if any(k.startswith("KVARN_") and v is not None for k, v in environment.items()):
        raise ValueError("capture overrode dtype-selected KVarN defaults")
    for key, value in manifest["selected_environment"].items():
        if key in environment and environment[key] != value:
            raise ValueError(f"running environment differs: {key}")
    if manifest["runtime_identity"]["candidate_env"] != manifest["runtime"]:
        raise ValueError("candidate closure identity differs")
    expected_waves = [
        (t, r) for t in plan["trials"] for r in range(plan["repeats"] + 1)
    ]
    if len(manifest["waves"]) != len(expected_waves):
        raise ValueError("incomplete trial matrix")
    groups = {t["id"]: [] for t in plan["trials"]}
    previous_log_end = 0
    for wave, (trial, repeat) in zip(manifest["waves"], expected_waves):
        name = f"{trial['id']}-r{repeat}"
        b = trial["concurrency"]
        if (
            wave["id"] != name
            or wave["trial"] != trial["id"]
            or wave["phase"] != ("performance" if repeat else "warmup")
            or wave["concurrency"] != b
            or wave["request_ids"] != [f"{name}-q{i}" for i in range(b)]
        ):
            raise ValueError("mismatched trial identity")
        log_start, log_end = wave["service_log_range"]
        if not 0 <= previous_log_end <= log_start <= log_end <= len(service_log):
            raise ValueError("invalid service log interval")
        previous_log_end = log_end
        events = compilation_events(
            service_log[log_start:log_end].decode(errors="replace")
        )
        if events != wave["compilation_events"] or (repeat and events):
            raise ValueError("unaccounted or measured runtime compilation")
        paths = [
            f"{name}-metrics-before.txt",
            f"{name}-metrics-after.txt",
            f"{name}-scheduler.json",
        ]
        if set(paths) != set(wave["evidence_sha256"]):
            raise ValueError("missing scheduler evidence hashes")
        for filename in paths:
            check_hash(directory / filename, wave["evidence_sha256"][filename])
        samples = load(directory / paths[2])
        peak = max((s["running"] for s in samples), default=0)
        if peak < b or peak != wave["peak_running"]:
            raise ValueError("missing measured concurrency")
        before, after = [
            perf.parse_scheduler_metrics((directory / p).read_text()) for p in paths[:2]
        ]
        if after["vllm:num_preemptions_total"] != before["vllm:num_preemptions_total"]:
            raise ValueError("preemption invalidates matched performance cell")
        expected = {
            "model": "sunny-chat",
            "prompt": trial["prompt_token_ids"],
            "temperature": 0,
            "seed": 42,
            "max_tokens": plan["output_tokens"],
            "ignore_eos": True,
            "return_token_ids": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        responses = [audit_stream(directory, n, expected) for n in wave["request_ids"]]
        if wave["seconds"] < max(r["elapsed_seconds"] for r in responses):
            raise ValueError("wave interval shorter than a contained request")
        close(
            wave["output_tokens_per_second"],
            b * plan["output_tokens"] / wave["seconds"],
        )
        pre, post = [speculative_counters(directory / p) for p in paths[:2]]
        if pre.keys() != post.keys():
            raise ValueError("speculative counter set changed")
        counters = {k: post[k] - pre[k] for k in pre}
        audit_mtp_activity(counters, plan["server_args"])
        if repeat:
            groups[trial["id"]].append(
                {
                    "throughput": wave["output_tokens_per_second"],
                    "wave_seconds": wave["seconds"],
                    "decode": statistics.median(
                        r["decode_tokens_per_second"] for r in responses
                    ),
                    "ttft": statistics.median(r["ttft_seconds"] for r in responses),
                    "latency": statistics.median(
                        r["elapsed_seconds"] for r in responses
                    ),
                    "mtp_counters": counters,
                    "quality_findings": {
                        r["id"]: r["quality_findings"]
                        for r in responses
                        if r["quality_findings"]
                    },
                }
            )
    memory_window = None
    if manifest.get("memory_sampling_schema") is not None:
        if manifest["memory_sampling_schema"] != "owned-drm-timestamped-v1":
            raise ValueError("unknown memory sampling schema")
        memory_window = (
            min(wave["started_unix"] for wave in manifest["waves"]),
            max(wave["started_unix"] + wave["seconds"] for wave in manifest["waves"]),
        )
    memory = memory_summary(directory, required_window=memory_window)
    if memory["errors"]:
        raise ValueError(f"memory sampling errors: {memory['errors']}")
    return manifest, groups, memory


def compare(plan_path, directories):
    plan = load(plan_path)
    if len(directories) != 4 or plan["arm_order"] != ["k4v4", "k4v2", "k4v2", "k4v4"]:
        raise ValueError("requires four independently started ABBA arms")
    captures = [audit(p, plan, perf.sha256_file(plan_path)) for p in directories]
    previous = None
    for label, (manifest, _, _) in zip(plan["arm_order"], captures):
        if manifest["cache_dtype"] != f"kvarn_{label}_g128_compact":
            raise ValueError("incorrect arm order")
        if not manifest["started_unix"] < manifest["finished_unix"]:
            raise ValueError("invalid service lifetime")
        if previous is not None and previous > manifest["started_unix"]:
            raise ValueError("capture services overlap or are out of order")
        previous = manifest["finished_unix"]
        for key in ("runtime_identity", "harness_sha256"):
            if manifest[key] != captures[0][0][key]:
                raise ValueError(f"arm identities differ: {key}")
        left, right = [
            dict(m["actual_environment"]) for m in (manifest, captures[0][0])
        ]
        left.pop("VLLM_CACHE_ROOT", None)
        right.pop("VLLM_CACHE_ROOT", None)
        if left != right:
            raise ValueError("arm environments differ")
    rows = []
    for trial in plan["trials"]:
        name = trial["id"]
        a = captures[0][1][name] + captures[3][1][name]
        b = captures[1][1][name] + captures[2][1][name]
        stats = {}
        for metric in ("throughput", "decode", "ttft", "latency", "wave_seconds"):
            av, bv = [[r[metric] for r in arm] for arm in (a, b)]
            am, bm = statistics.median(av), statistics.median(bv)
            stats[metric] = {
                "k4v4": av,
                "k4v2": bv,
                "ratio": bm / am,
                "k4v4_median": am,
                "k4v2_median": bm,
            }
        passed = (
            stats["throughput"]["ratio"] >= plan["min_throughput_ratio"]
            and stats["decode"]["ratio"] >= plan["min_request_decode_ratio"]
            and stats["latency"]["ratio"] <= plan["max_latency_ratio"]
            and stats["ttft"]["ratio"] <= plan["max_latency_ratio"]
        )
        target = all(
            stats[m]["ratio"] >= plan["tuning_target_min_ratio"]
            for m in ("throughput", "decode")
        )
        rows.append(
            {
                "trial": name,
                "metrics": stats,
                "outer_floor_passed": passed,
                "tuning_target_passed": target,
                "k4v4_samples": a,
                "k4v2_samples": b,
            }
        )
    return {
        "schema": "kvarn-lowbit-performance-comparison-v1",
        "plan_sha256": perf.sha256_file(plan_path),
        "comparer_sha256": perf.sha256_file(Path(__file__)),
        "captures": [str(p.resolve()) for p in directories],
        "outer_floor_passed": all(r["outer_floor_passed"] for r in rows),
        "tuning_target_passed": all(r["tuning_target_passed"] for r in rows),
        "rows": rows,
        "memory": [c[2] for c in captures],
        "scope": "Matched service performance with observed concurrency. Quality findings need review; this is not model quality or complete release qualification. Timings include MTP acceptance effects and SSE chunks, not individual inter-token latency.",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("captures", nargs=4, type=Path)
    args = p.parse_args()
    report = compare(args.plan, args.captures)
    perf.write_json_atomic(args.output, report)
    print(
        json.dumps(
            {k: report[k] for k in ("outer_floor_passed", "tuning_target_passed")}
        )
    )


if __name__ == "__main__":
    main()
