"""Audit paired Python-dispatch service changes with identical native dependencies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

from scripts import kvarn_chunk_compare as chunks
from scripts import kvarn_mtp_compare as mtp
from scripts import kvarn_prefill_compare as prefill
from scripts import kvarn_vision_compare as vision
from scripts.kvarn_perf_run import sha256_file, write_json_atomic
from scripts.kvarn_prefill_chunks import chunk_plan, recorded_budget
from scripts.kvarn_scan_engine_log import scan, xpu_runtime_evidence

ARMS = ("control", "candidate")
PREFILL = "text-prefill-performance-v1"
QUALIFY = "vision-correctness"
METRICS = (
    "ttft_seconds",
    "total_seconds",
    "first_token_seconds",
    "decode_tokens_per_second",
    "inter_token_p95_seconds",
    "inter_token_p99_seconds",
)


def closure_digest(paths: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(set(paths))) + "\n").encode()).hexdigest()


def runtime_spec(environment: Path) -> dict:
    environment = environment.resolve(strict=True)
    executable = (environment / "bin/vllm").resolve(strict=True)
    package = executable.parent.parent
    if not re.fullmatch(r"[^/]+-python3\.[0-9]+-vllm-xpu-.+", package.name):
        # withPackages can insert a binary wrapper instead of a direct symlink.
        packages = set(
            re.findall(
                rb"(/nix/store/[a-z0-9]+-python3\.[0-9]+-vllm-xpu-[^/\x00\s\"']+)/bin/(?:vllm|\.vllm-wrapped)",
                executable.read_bytes(),
            )
        )
        if len(packages) == 1:
            package = Path(packages.pop().decode())
    if not re.fullmatch(r"[^/]+-python3\.[0-9]+-vllm-xpu-.+", package.name):
        raise ValueError("expected a normal vLLM Python package bin/vllm wrapper")
    return {"candidate_env": str(environment), "process_package": str(package)}


def validate_identity(manifest: dict, expected: dict) -> dict:
    identity = manifest["runtime_identity"]
    for key, value in expected.items():
        if identity[key] != value:
            raise ValueError(f"unexpected runtime identity: {key}")
    if manifest["service_env"] != expected["candidate_env"]:
        raise ValueError("service environment disagrees with runtime identity")
    executable = expected["process_package"] + "/bin/.vllm-wrapped"
    if identity["process_executable"] != executable:
        raise ValueError("unexpected process executable")
    if manifest["actual_argv"].count(executable) != 1:
        raise ValueError("actual argv does not identify the expected vLLM package")
    for prefix, root in (
        ("candidate", expected["candidate_env"]),
        ("process", expected["process_package"]),
    ):
        paths = identity[prefix + "_closure_paths"]
        if (
            paths != sorted(set(paths))
            or root not in paths
            or not all(re.fullmatch(r"/nix/store/[^/]+", p) for p in paths)
            or closure_digest(paths) != identity[prefix + "_closure_sha256"]
        ):
            raise ValueError(f"invalid {prefix} closure evidence")
    if not set(identity["process_closure_paths"]) <= set(
        identity["candidate_closure_paths"]
    ):
        raise ValueError("process closure is absent from the service environment")
    return identity


def compare_identities(left: dict, right: dict, allowed_wrappers=()) -> dict:
    """Only the two specified Python packages may differ inside process closures."""
    dependencies = []
    for identity in (left, right):
        dependencies.append(
            set(identity["process_closure_paths"]) - {identity["process_package"]}
        )
    if dependencies[0] != dependencies[1]:
        raise ValueError("native or other process dependencies changed")
    allowed = [
        (left["candidate_env"], right["candidate_env"]),
        (left["process_package"], right["process_package"]),
    ]
    seen = set()
    for pair in allowed_wrappers:
        if len(pair) != 2:
            raise ValueError("wrapper exceptions must be control/candidate pairs")
        for path, identity in zip(pair, (left, right), strict=True):
            if (
                path in seen
                or path not in identity["candidate_closure_paths"]
                or path in identity["process_closure_paths"]
                or not re.fullmatch(r"/nix/store/[^/]+-env", path)
            ):
                raise ValueError("only explicit outer environment wrappers may differ")
            seen.add(path)
        allowed.append(tuple(pair))
    residual = [
        set(identity["candidate_closure_paths"]) - {p[index] for p in allowed}
        for index, identity in enumerate((left, right))
    ]
    if residual[0] != residual[1]:
        raise ValueError(
            "environment closure changed beyond declared packages/wrappers; "
            "declare an outer wrapper with --allow-env-path-pair"
        )
    return {
        "allowed_path_pairs": [dict(zip(ARMS, pair, strict=True)) for pair in allowed],
        "identical_process_dependencies": sorted(dependencies[0]),
        "identical_process_dependencies_sha256": closure_digest(list(dependencies[0])),
        "scope": "Immutable recorded closures; not a measurement of every loaded DSO.",
    }


def normalized_manifest(manifest: dict, directory: Path) -> dict:
    argv = list(manifest["actual_argv"])
    argv[argv.index(manifest["runtime_identity"]["process_executable"])] = (
        "<VLLM_PYTHON_EXECUTABLE>"
    )
    environments = {}
    for name in ("actual_environment", "selected_environment"):
        env = dict(manifest[name])
        expected_cache = str(directory / "runtime-cache")
        if env.get("VLLM_CACHE_ROOT") != expected_cache:
            raise ValueError(
                "runtime cache must be private to this fresh service start"
            )
        env["VLLM_CACHE_ROOT"] = "<PER_START_RUNTIME_CACHE>"
        environments[name] = env
    return {
        "actual_argv": argv,
        **environments,
        "transport_environment": manifest["transport_environment"],
        "speculative_config": manifest["speculative_config"],
        "suite": manifest["suite"],
        "harness_sha256": manifest["harness_sha256"],
    }


def argument(manifest: dict, flag: str) -> str:
    argv = manifest["actual_argv"]
    if argv.count(flag) != 1:
        raise ValueError(f"missing or ambiguous argument: {flag}")
    return argv[argv.index(flag) + 1]


def service_evidence(directory: Path, manifest: dict) -> dict:
    lines = (directory / "service.log").read_text().splitlines()
    counters = mtp.speculative_counters(directory / "metrics.txt")
    if any(not math.isfinite(value) or value < 0 for value in counters.values()):
        raise ValueError("invalid speculative counters")
    if manifest["speculative_config"] and not (
        counters.get("vllm:spec_decode_num_drafts_total", 0) > 0
    ):
        raise ValueError("MTP qualification did not record any speculative drafts")
    return {
        "path": str(directory),
        "manifest_sha256": sha256_file(directory / "manifest.json"),
        "preemptions": chunks.preemptions(directory),
        "memory": vision.memory_summary(directory),
        "engine_log": scan(lines),
        "xpu_runtime_evidence": xpu_runtime_evidence(lines),
        "speculative_counters": counters,
        "evidence_sha256": {
            name: sha256_file(directory / name)
            for name in ("metrics.txt", "memory-fdinfo.jsonl")
        },
    }


def numerical_timing(directory: Path, result: dict) -> dict:
    timed = prefill.token_timing(directory, result)
    result = {**result, **timed}
    values = {
        key: result[key]
        for key in (
            "ttft_seconds",
            "total_seconds",
            "first_token_seconds",
            "decode_tokens_per_second",
        )
    }
    values.update(
        inter_token_p95_seconds=timed["inter_token_seconds"]["p95"],
        inter_token_p99_seconds=timed["inter_token_seconds"]["p99"],
    )
    if (
        any(not math.isfinite(v) or v < 0 for v in values.values())
        or values["ttft_seconds"] <= 0
        or values["total_seconds"] < values["ttft_seconds"]
        or values["decode_tokens_per_second"] <= 0
    ):
        raise ValueError("invalid request timing")
    return {**values, "inter_token_seconds": timed["inter_token_seconds"]}


def compare_request(control, candidate, left, right, *, timed: bool) -> dict:
    name = left["id"]
    if name != right["id"] or left["phase"] != right["phase"]:
        raise ValueError("mismatched request order or phase")
    requests = [
        vision.load(directory / f"{name}-request.json")
        for directory in (control, candidate)
    ]
    if requests[0] != requests[1]:
        raise ValueError(f"mismatched request: {name}")
    if timed and requests[0].get("logprobs"):
        raise ValueError("logprob collection invalidates performance timing")
    tokens = [mtp.tokens(path, name) for path in (control, candidate)]
    if tokens[0][0] != tokens[1][0]:
        raise ValueError(f"mismatched processed prompt IDs: {name}")
    if timed and any(len(ids[1]) != 512 for ids in tokens):
        raise ValueError(
            "every prefill request, including warmup, must emit 512 tokens"
        )
    for result in (left, right):
        if result["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0):
            raise ValueError("prefix reuse invalidates the fresh-prefill envelope")
    logprobs = [
        mtp.logprobs(path, name, ids[1])
        for path, ids in zip((control, candidate), tokens, strict=True)
    ]
    equal = tokens[0][1] == tokens[1][1]
    mismatch = next(
        (i for i, (a, b) in enumerate(zip(tokens[0][1], tokens[1][1])) if a != b),
        None,
    )
    if mismatch is None and not equal:
        mismatch = min(len(ids[1]) for ids in tokens)
    logprob_summary = None
    if logprobs[0] is not None and logprobs[1] is not None:
        summary = mtp.shared_prefix_logprobs(*logprobs)
        logprob_summary = {k: v for k, v in summary.items() if k != "positions"}
    return {
        "id": name,
        "phase": left["phase"],
        "prompt_tokens": len(tokens[0][0]),
        "generated_tokens": [len(ids[1]) for ids in tokens],
        "token_id_sha256": {
            arm: {
                kind: hashlib.sha256(json.dumps(values).encode()).hexdigest()
                for kind, values in zip(("prompt", "output"), ids, strict=True)
            }
            for arm, ids in zip(ARMS, tokens, strict=True)
        },
        "same_generated_tokens": equal,
        "first_token_mismatch": mismatch,
        "same_content": left["content"] == right["content"],
        "same_usage": left["usage"] == right["usage"],
        "same_finish_reason": left.get("finish_reason") == right.get("finish_reason"),
        "same_logprobs": logprobs[0] == logprobs[1],
        "logprob_evidence_collected": logprobs[0] is not None,
        "shared_prefix_logprobs": logprob_summary,
        "artifact_sha256": {
            arm: {
                suffix: sha256_file(path / f"{name}-{suffix}")
                for suffix in ("request.json", "response.json", "sse.jsonl")
            }
            for arm, path in zip(ARMS, (control, candidate), strict=True)
        },
        "timing": {
            arm: numerical_timing(path, result)
            for arm, path, result in zip(
                ARMS, (control, candidate), (left, right), strict=True
            )
        }
        if timed
        else None,
    }


def timing_summary(rows: list[dict]) -> dict:
    measured = [r for r in rows if r["phase"] == "performance"]
    if not measured:
        raise ValueError("missing measured requests")
    result = {
        arm: {
            metric: prefill.distribution([r["timing"][arm][metric] for r in measured])
            for metric in METRICS
        }
        for arm in ARMS
    }
    result["candidate_minus_control"] = {
        metric: {
            "paired_absolute_delta": prefill.distribution(
                [
                    r["timing"]["candidate"][metric] - r["timing"]["control"][metric]
                    for r in measured
                ]
            ),
            "median_relative_percent": 100
            * (
                result["candidate"][metric]["median"]
                / result["control"][metric]["median"]
                - 1
            )
            if result["control"][metric]["median"]
            else None,
        }
        for metric in METRICS
    }
    return result


def compare(pairs, *, control_env: Path, candidate_env: Path, allowed_wrappers=()):
    specs = [runtime_spec(path) for path in (control_env, candidate_env)]
    if specs[0]["process_package"] == specs[1]["process_package"]:
        raise ValueError("control and candidate must identify distinct Python packages")
    seen = set()
    identities = None
    identity_report = None
    baseline_harness = None
    configurations = {}
    workloads = {}
    reports = []
    failures = []
    for index, pair in enumerate(pairs):
        directories = [Path(p).resolve(strict=True) for p in pair]
        if directories[0] == directories[1] or any(p in seen for p in directories):
            raise ValueError("each service start must have a distinct capture")
        seen.update(directories)
        audits = [vision.audit(p) for p in directories]
        manifests = [item[0] for item in audits]
        current = [
            validate_identity(manifest, spec)
            for manifest, spec in zip(manifests, specs, strict=True)
        ]
        if identities is not None and current != identities:
            raise ValueError(
                "an arm's immutable runtime changed between service starts"
            )
        identities = current
        identity_report = compare_identities(*current, allowed_wrappers)
        normalized = [
            normalized_manifest(m, p)
            for m, p in zip(manifests, directories, strict=True)
        ]
        if normalized[0] != normalized[1]:
            raise ValueError("arguments, environment, configuration or harness changed")
        if (
            baseline_harness is not None
            and manifests[0]["harness_sha256"] != baseline_harness
        ):
            raise ValueError("harness changed between service starts")
        baseline_harness = manifests[0]["harness_sha256"]
        for key in ("workload_sha256", "image_sha256"):
            if manifests[0][key] != manifests[1][key]:
                raise ValueError(f"mismatched {key}")
        manifest = manifests[0]
        suite = manifest["suite"]
        if suite not in (PREFILL, QUALIFY) or any(
            m.get("profiler_config") for m in manifests
        ):
            raise ValueError("requires profiler-off prefill or vision qualification")
        timed = suite == PREFILL
        if timed and manifest["speculative_config"] is not None:
            raise ValueError("prefill timing requires MTP off")
        speculative = manifest["speculative_config"]
        if speculative is None:
            if "--speculative-config" in manifest["actual_argv"]:
                raise ValueError("actual speculative arguments disagree with manifest")
        elif json.loads(argument(manifest, "--speculative-config")) != speculative:
            raise ValueError("actual speculative arguments disagree with manifest")
        if (
            argument(manifest, "--max-num-seqs") != "1"
            or recorded_budget(manifest) != 2048
        ):
            raise ValueError("requires the bounded B1, 2048-token chunk envelope")
        if "--no-enable-prefix-caching" not in manifest["actual_argv"]:
            raise ValueError("requires disabled prefix caching")
        dtype = argument(manifest, "--kv-cache-dtype")
        configuration = (
            suite,
            dtype,
            json.dumps(manifest["speculative_config"], sort_keys=True),
        )
        if (
            configuration in configurations
            and configurations[configuration] != normalized[0]
        ):
            raise ValueError("settings changed across repeated workload configurations")
        configurations[configuration] = normalized[0]
        rows = [
            compare_request(*directories, left, right, timed=timed)
            for left, right in zip(audits[0][1], audits[1][1], strict=True)
        ]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("duplicate workload request IDs")
        workload_key = (*configuration, tuple(r["prompt_tokens"] for r in rows))
        workload_identity = {
            "workload_sha256": manifest["workload_sha256"],
            "image_sha256": manifest["image_sha256"],
            "prompt_id_sha256": [
                r["token_id_sha256"]["control"]["prompt"] for r in rows
            ],
        }
        if workload_key in workloads and workloads[workload_key] != workload_identity:
            raise ValueError("repeated service starts changed the matched workload")
        workloads[workload_key] = workload_identity
        if timed:
            if [r["phase"] for r in rows] != ["warmup", *(["performance"] * 3)]:
                raise ValueError(
                    "expected one identical warmup and three measured requests"
                )
            reference_request = None
            for case in vision.load(directories[0] / "workload.json"):
                coverage = case["coverage"]
                expected = chunk_plan(coverage["prompt_tokens"], 2048)
                if any(coverage.get(k) != v for k, v in expected.items()):
                    raise ValueError("coverage disagrees with actual chunk budget")
                request = vision.load(directories[0] / f"{case['id']}-request.json")
                if (
                    request.get("max_tokens") != 512
                    or request.get("ignore_eos") is not True
                ):
                    raise ValueError(
                        "requires exactly 512 requested output tokens and ignore_eos"
                    )
                if reference_request is not None and request != reference_request:
                    raise ValueError("warmup and measured requests differ")
                reference_request = request
        services = {
            arm: service_evidence(p, m)
            for arm, p, m in zip(ARMS, directories, manifests, strict=True)
        }
        for arm, evidence in services.items():
            if (
                evidence["preemptions"] != 0
                or evidence["engine_log"]["status"] != "passed"
            ):
                failures.append(
                    {
                        "pair": index,
                        "arm": arm,
                        "reason": "preemption or fatal engine log",
                    }
                )
        for row in rows:
            for key in (
                "same_generated_tokens",
                "same_content",
                "same_usage",
                "same_finish_reason",
                "same_logprobs",
            ):
                if not row[key]:
                    failures.append({"pair": index, "id": row["id"], "reason": key})
            if not timed and not row["logprob_evidence_collected"]:
                raise ValueError(
                    "qualification comparison requires target logprob evidence"
                )
        repeat_equality = None
        if timed:
            repeat_equality = {
                arm: all(
                    row["token_id_sha256"][arm] == rows[0]["token_id_sha256"][arm]
                    for row in rows
                )
                for arm in ARMS
            }
            for arm, equal in repeat_equality.items():
                if not equal:
                    failures.append(
                        {
                            "pair": index,
                            "arm": arm,
                            "reason": "identical requests differ within service",
                        }
                    )
        reports.append(
            {
                "pair_index": index,
                "suite": suite,
                "cache_dtype": dtype,
                "speculative_config": manifest["speculative_config"],
                "allowed_environment_difference": {
                    "VLLM_CACHE_ROOT": [str(p / "runtime-cache") for p in directories]
                },
                "services": services,
                "same_speculative_counters": services["control"]["speculative_counters"]
                == services["candidate"]["speculative_counters"],
                "within_service_tokens_equal": repeat_equality,
                "requests": rows,
                "timing": timing_summary(rows) if timed else None,
            }
        )
    if not reports:
        raise ValueError("at least one control/candidate pair is required")
    by_length = {}
    for report in reports:
        if report["timing"] is None:
            continue
        key = f"{report['cache_dtype']}/{report['requests'][0]['prompt_tokens']}"
        by_length.setdefault(key, []).append(report)
    variation = {}
    for key, group in by_length.items():
        variation[key] = {
            "fresh_start_pairs": len(group),
            "per_start_relative_percent": {
                metric: [
                    g["timing"]["candidate_minus_control"][metric][
                        "median_relative_percent"
                    ]
                    for g in group
                ]
                for metric in METRICS
            },
            "control_start_median_range": {
                metric: prefill.distribution(
                    [g["timing"]["control"][metric]["median"] for g in group]
                )
                for metric in METRICS
            },
        }
    return {
        "schema": "xpu-prefill-source-comparison-v1",
        "status": "failed" if failures else "passed-correctness-and-resource-checks",
        "failures": failures,
        "runtime_identities": dict(zip(ARMS, identities, strict=True)),
        "runtime_comparison": identity_report,
        "pairs": reports,
        "timing_variation": variation,
        "performance_acceptance": "No minimum speedup or numerical non-inferiority threshold; inspect per-start results and repeat a suspicious regression.",
        "confidence": "Descriptive paired requests within separately started services; three serial requests and two starts do not establish statistical equivalence. No formal confidence interval is claimed.",
        "limitations": [
            "Source-only attribution also requires review of the exact Python source diff; closure equality alone cannot prove that diff's scope.",
            "Only captured environment variables are audited; recorded closure membership is not a loaded-library trace.",
            "Client token arrival times include transport buffering; qualification logprob runs are excluded from performance summaries.",
            "Exact observed tokens/top-k scores do not prove arbitrary cache or recurrent-state equivalence.",
            "A pass denotes correctness/resource evidence, not automatic acceptance of any measured latency regression.",
        ],
        "comparer_sha256": sha256_file(Path(__file__)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True)
    parser.add_argument("--control-env", type=Path, required=True)
    parser.add_argument("--candidate-env", type=Path, required=True)
    parser.add_argument("--allow-env-path-pair", nargs=2, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        args.pair,
        control_env=args.control_env,
        candidate_env=args.candidate_env,
        allowed_wrappers=args.allow_env_path_pair,
    )
    write_json_atomic(args.output, report)
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
