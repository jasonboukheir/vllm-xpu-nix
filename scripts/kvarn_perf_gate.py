#!/usr/bin/env python3
"""Gate repeated, provenance-matched Kvarn serving benchmarks."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import statistics
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts.kvarn_scan_engine_log import scan, xpu_runtime_evidence
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from kvarn_scan_engine_log import scan, xpu_runtime_evidence

NATIVE_DISPATCH = "Using the native Xe2 KVarN qlen=1 decoder"
FALLBACK_PATTERN = re.compile(
    r"(?i)(?:\bkvarn\b[^\n]{0,120}\b(?:fallback|falling back)\b|"
    r"\b(?:fallback|falling back)\b[^\n]{0,120}\bkvarn\b)"
)
COMPACT_DTYPE = "kvarn_k4v4_g128_compact"
EXPECTED_XPU_DEVICE_NAME = "Intel(R) Arc(TM) Pro B70 Graphics"
FORMAL_CONTEXTS = frozenset({4096, 16384, 32768, 65023})
REQUIRED_GATES = (
    "native_decode_short",
    "native_decode_262k",
    "b1_replay",
    "b1_restart",
    "cancel_reuse",
    "b4_isolation",
    "near_262k_reference_equivalence",
    "near_262k_restart",
)
CORRECTNESS_FIXTURE_LENGTHS = {
    "dialogue-127": 127,
    "code-4095": 4095,
    "math-16383": 16383,
    "reasoning-65023": 65023,
    "reasoning-261631": 261631,
}
CORRECTNESS_SHORT_FIXTURES = tuple(list(CORRECTNESS_FIXTURE_LENGTHS)[:4])
CORRECTNESS_PHASE_SPECS = {
    "native-65k-b1-first": {
        "name": "native-65k-b1-first",
        "launcher": "vllm-xpu-brutus-kvarn-native-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 65536,
        "splits": 24,
    },
    "native-65k-b1-restart": {
        "name": "native-65k-b1-restart",
        "launcher": "vllm-xpu-brutus-kvarn-native-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 65536,
        "splits": 24,
    },
    "native-65k-b4": {
        "name": "native-65k-b4",
        "launcher": "vllm-xpu-brutus-kvarn-native-b4",
        "native": True,
        "batch": 4,
        "max_model_len": 65536,
        "splits": 16,
    },
    "reference-262k-b1": {
        "name": "reference-262k-b1",
        "launcher": "vllm-xpu-brutus-kvarn-262k-b1",
        "native": False,
        "batch": 1,
        "max_model_len": 262144,
        "splits": 1,
    },
    "native-262k-b1-first": {
        "name": "native-262k-b1-first",
        "launcher": "vllm-xpu-brutus-kvarn-native-262k-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 262144,
        "splits": 24,
    },
    "native-262k-b1-restart": {
        "name": "native-262k-b1-restart",
        "launcher": "vllm-xpu-brutus-kvarn-native-262k-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 262144,
        "splits": 24,
    },
}
CORRECTNESS_RUNNER_SOURCES = {
    "scripts/kvarn_correctness_run.py",
    "scripts/kvarn_perf_gate.py",
    "scripts/kvarn_perf_run.py",
    "scripts/kvarn_scan_engine_log.py",
    "scripts/kvarn_service_gate.py",
}
CORRECTNESS_NATIVE_SOURCES = {
    "tests/flash_attn/test_kvarn_decode_xpu.py",
    "benchmark/check_kvarn_decode.py",
    "benchmark/kvarn_utils.py",
}
COMMON_PROVENANCE_FIELDS = (
    "backend",
    "model_id",
    "tokenizer_id",
    "num_prompts",
    "request_rate",
    "max_concurrency",
    "kvarn_candidate_id",
    "kvarn_model_revision",
    "kvarn_service_profile",
    "kvarn_workload_id",
    "kvarn_seed",
    "kvarn_max_model_len",
    "kvarn_max_num_seqs",
    "kvarn_enforce_eager",
    "kvarn_prefix_caching",
    "kvarn_mtp",
    "kvarn_xpu_graph",
    "kvarn_scheduler_peak_running",
    "kvarn_correctness_sha256",
    "kvarn_process_package",
    "kvarn_process_closure_sha256",
    "kvarn_candidate_closure_sha256",
    "kvarn_max_num_batched_tokens",
    "kvarn_matched_profile_sha256",
    "kvarn_accelerator",
    "kvarn_xpu_available",
    "kvarn_xpu_device_count",
    "kvarn_xpu_device_name",
    "kvarn_xpu_compute_probe",
    "kvarn_hardware_preflight_path",
    "kvarn_hardware_preflight_sha256",
    "kvarn_evidence_mode",
)
ARM_PROVENANCE_FIELDS = (
    "kvarn_arm",
    "kvarn_kv_cache_dtype",
    "kvarn_native_xpu",
    "kvarn_native_splits",
)


class GateError(ValueError):
    """Raised when inputs do not form a valid matched comparison."""


def _artifact_reference(value: Any, *, name: str, owner: Path) -> tuple[Path, str]:
    if not isinstance(value, dict):
        raise GateError(f"{owner}: {name} must be an artifact reference")
    raw_path, digest = value.get("path"), value.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise GateError(f"{owner}: {name} has no artifact path")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise GateError(f"{owner}: {name} has invalid SHA-256")
    path = Path(raw_path).expanduser().resolve()
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise GateError(f"{owner}: cannot read {name} artifact {path}: {exc}") from exc
    if actual != digest:
        raise GateError(f"{owner}: {name} artifact SHA differs: {path}")
    return path, digest


def _json_artifact(
    value: Any, *, name: str, owner: Path
) -> tuple[Path, dict[str, Any]]:
    path, _digest = _artifact_reference(value, name=name, owner=owner)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{owner}: cannot parse {name} artifact {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise GateError(f"{owner}: {name} artifact must be an object")
    return path, document


def _validate_correctness_result(value: Any, fixture_id: str, *, owner: Path) -> None:
    if not isinstance(value, dict):
        raise GateError(f"{owner}: result for {fixture_id} must be an object")
    prompt_tokens = CORRECTNESS_FIXTURE_LENGTHS[fixture_id]
    expected_usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 512,
        "total_tokens": prompt_tokens + 512,
    }
    token_ids = value.get("token_ids")
    if (
        value.get("id") != fixture_id
        or value.get("prompt_token_count") != prompt_tokens
        or not isinstance(value.get("prompt_token_ids_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", value["prompt_token_ids_sha256"])
        or value.get("max_tokens") != 512
        or value.get("finish_reason") != "length"
        or value.get("usage") != expected_usage
        or value.get("quality_findings") != []
        or not isinstance(token_ids, list)
        or len(token_ids) != 512
        or any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in token_ids
        )
    ):
        raise GateError(f"{owner}: result for {fixture_id} has invalid token evidence")
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    if value.get("token_ids_sha256") != hashlib.sha256(encoded).hexdigest():
        raise GateError(f"{owner}: result for {fixture_id} has a false token digest")


def _validate_correctness_comparison(
    value: Any, fixture_id: str, *, owner: Path
) -> None:
    if not isinstance(value, dict) or any(
        value.get(name) is not True
        for name in ("same_fixture", "same_prompt_token_ids", "token_ids_identical")
    ):
        raise GateError(f"{owner}: comparison for {fixture_id} is not exact")
    if value.get("status") != "passed" or value.get("fixture_id") != fixture_id:
        raise GateError(f"{owner}: comparison for {fixture_id} is not passed")
    for name in ("expected_token_ids_sha256", "actual_token_ids_sha256"):
        digest = value.get(name)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GateError(f"{owner}: comparison for {fixture_id} lacks {name}")
    if value["expected_token_ids_sha256"] != value["actual_token_ids_sha256"]:
        raise GateError(f"{owner}: comparison for {fixture_id} has unequal digests")


def _validate_correctness_comparisons(
    value: Any, fixture_ids: tuple[str, ...], *, owner: Path
) -> None:
    if not isinstance(value, list) or len(value) != len(fixture_ids):
        raise GateError(f"{owner}: comparison set has the wrong size")
    for comparison, fixture_id in zip(value, fixture_ids, strict=True):
        _validate_correctness_comparison(comparison, fixture_id, owner=owner)


def _validate_correctness_phase(
    value: Any,
    phase_name: str,
    candidate_id: str,
    process_package: str,
    *,
    owner: Path,
) -> dict[str, Any]:
    path, phase = _json_artifact(value, name=f"{phase_name} phase", owner=owner)
    expected_spec = CORRECTNESS_PHASE_SPECS[phase_name]
    if (
        phase.get("status") != "passed"
        or phase.get("spec") != expected_spec
        or phase.get("native_dispatch_verified") is not expected_spec["native"]
        or not isinstance(phase.get("workload"), dict)
    ):
        raise GateError(f"{owner}: invalid {phase_name} phase evidence")
    for name in ("profile", "identity", "engine_log", "engine_log_scan"):
        _artifact_reference(phase.get(name), name=f"{phase_name}.{name}", owner=path)
    _identity_path, identity = _json_artifact(
        phase["identity"], name=f"{phase_name}.identity", owner=path
    )
    if (
        identity.get("candidate_env") != candidate_id
        or identity.get("process_package") != process_package
    ):
        raise GateError(f"{owner}: {phase_name} identity differs from the candidate")
    log_path, _digest = _artifact_reference(
        phase["engine_log"], name=f"{phase_name}.engine_log", owner=path
    )
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    dispatched = NATIVE_DISPATCH in log_text
    if dispatched is not expected_spec["native"]:
        raise GateError(f"{owner}: {phase_name} native dispatch evidence differs")
    if expected_spec["native"] and FALLBACK_PATTERN.search(log_text):
        raise GateError(f"{owner}: {phase_name} reports a native fallback")
    _scan_path, log_scan = _json_artifact(
        phase["engine_log_scan"], name=f"{phase_name}.engine_log_scan", owner=path
    )
    if log_scan.get("status") != "passed" or log_scan.get("fatal_findings") != []:
        raise GateError(f"{owner}: {phase_name} log scan is not clean")
    return phase["workload"]


def validate_correctness_gate_evidence(
    name: str,
    evidence: Any,
    *,
    path: Path,
    candidate_id: str,
    process_package: str,
) -> None:
    """Validate the meaning of one hashed correctness-gate artifact."""
    if name not in REQUIRED_GATES:
        raise GateError(f"{path}: unknown correctness gate {name}")
    if (
        not isinstance(evidence, dict)
        or evidence.get("status") != "passed"
        or evidence.get("gate") != name
    ):
        raise GateError(f"{path}: gate identity/status differs from {name}")

    if name.startswith("native_decode_"):
        counts = evidence.get("junit_counts")
        if (
            not isinstance(counts, dict)
            or isinstance(counts.get("tests"), bool)
            or not isinstance(counts.get("tests"), int)
            or counts["tests"] < 1
            or any(
                counts.get(field) != 0 for field in ("failures", "errors", "skipped")
            )
            or evidence.get("candidate_id") != candidate_id
        ):
            raise GateError(f"{path}: primitive gate lacks an all-pass JUnit result")
        command = evidence.get("command")
        expression = {
            "native_decode_short": "not long_context_ragged_b4_matches_structured_oracle",
            "native_decode_262k": "long_context_ragged_b4_matches_structured_oracle",
        }[name]
        if (
            not isinstance(command, list)
            or not all(isinstance(argument, str) for argument in command)
            or not any(
                command[index : index + 2] == ["-m", "pytest"]
                for index in range(len(command) - 1)
            )
            or "-k" not in command
            or command.index("-k") + 1 >= len(command)
            or command[command.index("-k") + 1] != expression
        ):
            raise GateError(f"{path}: primitive gate has the wrong pytest command")
        _artifact_reference(
            evidence.get("native_library"), name="native_library", owner=path
        )
        test_source, _digest = _artifact_reference(
            evidence.get("test_source"), name="test_source", owner=path
        )
        if str(test_source) not in command:
            raise GateError(
                f"{path}: pytest command did not execute the hashed test source"
            )
        _artifact_reference(evidence.get("pytest_log"), name="pytest_log", owner=path)
        junit, _digest = _artifact_reference(
            evidence.get("junit"), name="junit", owner=path
        )
        try:
            junit_root = ET.parse(junit).getroot()
            suites = (
                [junit_root]
                if junit_root.tag == "testsuite"
                else list(junit_root.findall("testsuite"))
            )
            junit_counts = {
                field: sum(int(suite.get(field, "0")) for suite in suites)
                for field in ("tests", "failures", "errors", "skipped")
            }
        except (OSError, ET.ParseError, ValueError) as exc:
            raise GateError(f"{path}: primitive JUnit is invalid: {exc}") from exc
        if junit_counts != counts:
            raise GateError(f"{path}: primitive JUnit counts differ from evidence")
        helpers = evidence.get("helper_sources")
        if not isinstance(helpers, dict) or set(helpers) != {
            "benchmark/check_kvarn_decode.py",
            "benchmark/kvarn_utils.py",
        }:
            raise GateError(f"{path}: primitive helper-source evidence is incomplete")
        for helper, reference in helpers.items():
            _artifact_reference(reference, name=helper, owner=path)
        return

    def phase(field: str, phase_name: str) -> dict[str, Any]:
        return _validate_correctness_phase(
            evidence.get(field),
            phase_name,
            candidate_id,
            process_package,
            owner=path,
        )

    if name == "b1_replay":
        workload = phase("service_phase", "native-65k-b1-first")
        comparisons = evidence.get("comparisons")
        if comparisons != workload.get("replay_comparisons"):
            raise GateError(f"{path}: replay evidence differs from its service phase")
        _validate_correctness_comparisons(
            comparisons, CORRECTNESS_SHORT_FIXTURES, owner=path
        )
        for field in ("first", "replay"):
            results = workload.get(field)
            if not isinstance(results, list) or len(results) != 4:
                raise GateError(f"{path}: {field} result set has the wrong size")
            for result, fixture_id in zip(
                results, CORRECTNESS_SHORT_FIXTURES, strict=True
            ):
                _validate_correctness_result(result, fixture_id, owner=path)
        return
    if name == "cancel_reuse":
        workload = phase("service_phase", "native-65k-b1-first")
        cancellation = workload.get("cancellation")
        if not isinstance(cancellation, dict) or any(
            evidence.get(field) != cancellation.get(field)
            for field in (
                "requested_generated_token_checkpoint",
                "generated_token_ids_before_close",
                "idle_metrics_before_replacement",
                "replacement",
                "comparison",
            )
        ):
            raise GateError(f"{path}: cancellation evidence differs from its phase")
        if (
            cancellation.get("requested_generated_token_checkpoint") != 257
            or cancellation.get("generated_token_ids_before_close") != 257
        ):
            raise GateError(f"{path}: cancellation did not close at token 257")
        metrics = cancellation.get("idle_metrics_before_replacement")
        if (
            not isinstance(metrics, dict)
            or set(metrics)
            != {
                "vllm:num_requests_running",
                "vllm:num_requests_waiting",
                "vllm:kv_cache_usage_perc",
            }
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value != 0
                for value in metrics.values()
            )
        ):
            raise GateError(f"{path}: cancellation reuse lacks exact idle metrics")
        _validate_correctness_result(
            cancellation.get("replacement"), "reasoning-65023", owner=path
        )
        _validate_correctness_comparison(
            cancellation.get("comparison"), "reasoning-65023", owner=path
        )
        return
    if name == "b1_restart":
        phase("original_service_phase", "native-65k-b1-first")
        workload = phase("restarted_service_phase", "native-65k-b1-restart")
        if evidence.get("comparisons") != workload.get("comparisons"):
            raise GateError(f"{path}: restart evidence differs from its service phase")
        _validate_correctness_comparisons(
            evidence["comparisons"], CORRECTNESS_SHORT_FIXTURES, owner=path
        )
        results = workload.get("results")
        if not isinstance(results, list) or len(results) != 4:
            raise GateError(f"{path}: restart result set has the wrong size")
        for result, fixture_id in zip(results, CORRECTNESS_SHORT_FIXTURES, strict=True):
            _validate_correctness_result(result, fixture_id, owner=path)
        return
    if name == "b4_isolation":
        phase("b1_service_phase", "native-65k-b1-first")
        workload = phase("b4_service_phase", "native-65k-b4")
        if any(
            evidence.get(field) != workload.get(field)
            for field in ("comparisons", "overlap")
        ):
            raise GateError(f"{path}: B4 evidence differs from its service phase")
        _validate_correctness_comparisons(
            evidence["comparisons"], CORRECTNESS_SHORT_FIXTURES, owner=path
        )
        results = workload.get("results")
        if not isinstance(results, list) or len(results) != 4:
            raise GateError(f"{path}: B4 result set has the wrong size")
        for result, fixture_id in zip(results, CORRECTNESS_SHORT_FIXTURES, strict=True):
            _validate_correctness_result(result, fixture_id, owner=path)
        overlap = evidence.get("overlap")
        if (
            not isinstance(overlap, dict)
            or overlap.get("required_running") != 4
            or overlap.get("required_overlap_observed") is not True
            or isinstance(overlap.get("peak_running"), bool)
            or not isinstance(overlap.get("peak_running"), (int, float))
            or overlap["peak_running"] < 4
        ):
            raise GateError(f"{path}: B4 did not prove four-way overlap")
        return
    if name == "near_262k_reference_equivalence":
        reference = phase("reference_service_phase", "reference-262k-b1")
        native = phase("native_service_phase", "native-262k-b1-first")
        _validate_correctness_result(
            reference.get("result"), "reasoning-261631", owner=path
        )
        _validate_correctness_result(
            native.get("result"), "reasoning-261631", owner=path
        )
        comparison = evidence.get("comparison")
        if comparison != native.get("reference_comparison"):
            raise GateError(f"{path}: 262K comparison differs from its service phase")
        _validate_correctness_comparison(comparison, "reasoning-261631", owner=path)
        return
    if name == "near_262k_restart":
        phase("reference_service_phase", "reference-262k-b1")
        phase("first_native_service_phase", "native-262k-b1-first")
        restarted = phase("restarted_native_service_phase", "native-262k-b1-restart")
        _validate_correctness_result(
            restarted.get("result"), "reasoning-261631", owner=path
        )
        for field in ("reference_comparison", "native_restart_comparison"):
            if evidence.get(field) != restarted.get(field):
                raise GateError(
                    f"{path}: 262K restart comparison differs from its phase"
                )
            _validate_correctness_comparison(
                evidence[field], "reasoning-261631", owner=path
            )
        return
    raise AssertionError("unreachable")


def _validate_correctness_manifest_identity(
    document: dict[str, Any], path: Path
) -> str:
    identity = document.get("candidate_identity")
    if not isinstance(identity, dict):
        raise GateError(f"{path}: candidate_identity must be an object")
    process_package = identity.get("process_package")
    if not isinstance(process_package, str) or not process_package.startswith(
        "/nix/store/"
    ):
        raise GateError(f"{path}: candidate process package is invalid")
    for field in ("candidate_closure_sha256", "process_closure_sha256"):
        digest = identity.get(field)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise GateError(f"{path}: candidate identity lacks {field}")

    source = document.get("source_identity")
    if not isinstance(source, dict):
        raise GateError(f"{path}: source_identity must be an object")
    revisions = source.get("revisions")
    if (
        not isinstance(revisions, dict)
        or set(revisions)
        != {
            "vllm-xpu-release",
            "vllm-xpu-unstable-src",
            "vllm-xpu-kernels-unstable-src",
        }
        or any(
            not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            for revision in revisions.values()
        )
    ):
        raise GateError(f"{path}: candidate lock revisions are invalid")

    for field in ("config_checkout", "runner_checkout", "kernel_tracked_checkout"):
        checkout = source.get(field)
        if (
            not isinstance(checkout, dict)
            or not isinstance(checkout.get("head"), str)
            or not re.fullmatch(r"[0-9a-f]{40}", checkout["head"])
            or isinstance(checkout.get("files"), bool)
            or not isinstance(checkout.get("files"), int)
            or checkout["files"] < 1
            or not isinstance(checkout.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", checkout["sha256"])
            or checkout.get("unexpected_changes") != []
        ):
            raise GateError(f"{path}: {field} is not a clean identified checkout")
    if (
        source["kernel_tracked_checkout"]["head"]
        != revisions["vllm-xpu-kernels-unstable-src"]
    ):
        raise GateError(f"{path}: kernel checkout differs from the candidate lock")

    lock_path = source.get("lock_path")
    lock_sha256 = source.get("lock_sha256")
    if (
        not isinstance(lock_path, str)
        or not isinstance(lock_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", lock_sha256)
    ):
        raise GateError(f"{path}: lock-file identity is invalid")
    try:
        lock_bytes = Path(lock_path).expanduser().resolve().read_bytes()
        lock = json.loads(lock_bytes)
        locked = lock["nodes"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise GateError(f"{path}: cannot re-read candidate lock: {exc}") from exc
    if hashlib.sha256(lock_bytes).hexdigest() != lock_sha256 or any(
        locked.get(name, {}).get("locked", {}).get("rev") != revision
        for name, revision in revisions.items()
    ):
        raise GateError(f"{path}: candidate lock content differs from source identity")

    runner_sources = source.get("runner_sources")
    if not isinstance(runner_sources, dict) or set(runner_sources) != (
        CORRECTNESS_RUNNER_SOURCES
    ):
        raise GateError(f"{path}: correctness-runner source evidence is incomplete")
    for name, reference in runner_sources.items():
        _artifact_reference(reference, name=name, owner=path)
    native_sources = source.get("native_source_sha256")
    if (
        not isinstance(native_sources, dict)
        or set(native_sources) != (CORRECTNESS_NATIVE_SOURCES)
        or any(
            not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in native_sources.values()
        )
    ):
        raise GateError(f"{path}: native primitive source identity is incomplete")
    return process_package


@dataclass(frozen=True)
class Run:
    path: str
    completed: int
    output_throughput: float
    request_throughput: float
    total_token_throughput: float
    request_decode_throughputs: tuple[float, ...]
    ttft_ms: tuple[float, ...]
    itl_ms: tuple[float, ...]
    input_lens: tuple[int, ...]
    output_lens: tuple[int, ...]
    max_concurrent_requests: int
    provenance: dict[str, Any]
    run_order: int
    run_uuid: str
    run_started_at: dt.datetime
    engine_log_sha256: str

    def metrics(self) -> dict[str, float]:
        request_rates = list(self.request_decode_throughputs)
        ttft = list(self.ttft_ms)
        itl = list(self.itl_ms)
        return {
            "output_throughput": self.output_throughput,
            "request_throughput": self.request_throughput,
            "total_token_throughput": self.total_token_throughput,
            "min_request_decode_throughput": min(request_rates),
            "p10_request_decode_throughput": _percentile(request_rates, 10),
            "median_request_decode_throughput": statistics.median(request_rates),
            "p50_ttft_ms": _percentile(ttft, 50),
            "p90_ttft_ms": _percentile(ttft, 90),
            "p99_ttft_ms": _percentile(ttft, 99),
            "p50_itl_ms": _percentile(itl, 50),
            "p90_itl_ms": _percentile(itl, 90),
            "p99_itl_ms": _percentile(itl, 99),
        }


def _positive_number(value: Any, *, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateError(f"{path}: {name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise GateError(f"{path}: {name} must be finite and positive")
    return result


def _positive_integer(value: Any, *, name: str, path: Path) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise GateError(f"{path}: {name} must be a positive integer")
    return value


def _integer_list(value: Any, *, name: str, path: Path) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise GateError(f"{path}: {name} must be a non-empty list")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in value
    ):
        raise GateError(f"{path}: {name} must contain positive integers")
    return tuple(value)


def _required_text(document: dict[str, Any], name: str, path: Path) -> str:
    value = document.get(name)
    if not isinstance(value, str) or not value.strip():
        raise GateError(f"{path}: {name} must be a non-empty metadata string")
    return value.strip()


def _required_sha256(document: dict[str, Any], name: str, path: Path) -> str:
    value = _required_text(document, name, path)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise GateError(f"{path}: {name} must be lowercase SHA-256")
    return value


def _validate_hardware_preflight(raw_path: str, digest: str, *, owner: Path) -> None:
    path = Path(raw_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{owner}: cannot read XPU hardware preflight: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise GateError(f"{owner}: XPU hardware preflight SHA-256 differs")
    expected = {
        "xpu_available": True,
        "xpu_device_count": 1,
        "xpu_device_names": [EXPECTED_XPU_DEVICE_NAME],
        "probe_device": "xpu:0",
        "probe_value": 6.0,
    }
    if (
        not isinstance(document, dict)
        or any(document.get(name) != value for name, value in expected.items())
        or document.get("xpu_available") is not True
        or isinstance(document.get("xpu_device_count"), bool)
        or isinstance(document.get("probe_value"), bool)
    ):
        raise GateError(f"{owner}: XPU hardware preflight is not an exact B70 proof")


def _validate_warmup(
    raw_path: str,
    digest: str,
    *,
    owner: Path,
    batch: int,
    context: int,
    output_tokens: int,
    seed: str,
    arm: str,
    run_uuid: str,
    process_package: str,
    process_closure_sha256: str,
    candidate_closure_sha256: str,
    matched_profile_sha256: str,
) -> None:
    path = Path(raw_path).expanduser().resolve()
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{owner}: cannot read warmup evidence: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != digest:
        raise GateError(f"{owner}: warmup evidence SHA-256 differs")
    if path.parent != owner.expanduser().resolve().parent:
        raise GateError(
            f"{owner}: warmup evidence is not in the measured run directory"
        )
    expected_identity = {
        "schema_version": 1,
        "arm": arm,
        "run_uuid": run_uuid,
        "process_package": process_package,
        "process_closure_sha256": process_closure_sha256,
        "candidate_closure_sha256": candidate_closure_sha256,
        "matched_profile_sha256": matched_profile_sha256,
    }
    if (
        not isinstance(document, dict)
        or any(document.get(name) != value for name, value in expected_identity.items())
        or document.get("status") != "passed"
        or document.get("failed") != 0
        or isinstance(document.get("completed"), bool)
        or not isinstance(document.get("completed"), int)
        or document["completed"] < batch
        or isinstance(document.get("max_concurrent_requests"), bool)
        or not isinstance(document.get("max_concurrent_requests"), int)
        or document["max_concurrent_requests"] < batch
    ):
        raise GateError(f"{owner}: warmup evidence is not full-width and passed")

    try:
        numeric_seed = int(seed)
    except ValueError as exc:
        raise GateError(f"{owner}: benchmark seed is not an integer") from exc
    workload = document.get("workload")
    if (
        not isinstance(workload, dict)
        or workload.get("context") != context
        or workload.get("batch") != batch
        or workload.get("output_tokens") != output_tokens
        or workload.get("seed") != numeric_seed
        or workload.get("num_prompts") != document["completed"]
    ):
        raise GateError(f"{owner}: warmup workload differs from the measured run")

    raw_result_text = document.get("raw_result")
    raw_result_sha256 = document.get("raw_result_sha256")
    if not isinstance(raw_result_text, str) or not re.fullmatch(
        r"[0-9a-f]{64}", str(raw_result_sha256)
    ):
        raise GateError(f"{owner}: warmup raw-result identity is invalid")
    raw_result = Path(raw_result_text).expanduser().resolve()
    if raw_result.parent != path.parent:
        raise GateError(
            f"{owner}: warmup raw result is not in the measured run directory"
        )
    try:
        raw_result_bytes = raw_result.read_bytes()
        raw_document = json.loads(raw_result_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{owner}: cannot read warmup raw result: {exc}") from exc
    if hashlib.sha256(raw_result_bytes).hexdigest() != raw_result_sha256:
        raise GateError(f"{owner}: warmup raw-result SHA-256 differs")
    completed = document["completed"]
    raw_max_concurrent = (
        raw_document.get("max_concurrent_requests")
        if isinstance(raw_document, dict)
        else None
    )
    if (
        not isinstance(raw_document, dict)
        or raw_document.get("completed") != completed
        or raw_document.get("num_prompts") != completed
        or raw_document.get("failed") != 0
        or raw_document.get("max_concurrency") != batch
        or isinstance(raw_max_concurrent, bool)
        or not isinstance(raw_max_concurrent, int)
        or raw_max_concurrent < batch
        or raw_document.get("input_lens") != [context] * completed
        or raw_document.get("output_lens") != [output_tokens] * completed
    ):
        raise GateError(f"{owner}: warmup raw-result workload is invalid")

    argv = document.get("argv")
    if not isinstance(argv, list) or any(not isinstance(item, str) for item in argv):
        raise GateError(f"{owner}: warmup argv is invalid")

    def argument(name: str) -> str | None:
        try:
            index = argv.index(name)
        except ValueError:
            return None
        return argv[index + 1] if index + 1 < len(argv) else None

    expected_arguments = {
        "--random-input-len": str(context),
        "--random-output-len": str(output_tokens),
        "--num-prompts": str(completed),
        "--num-warmups": "0",
        "--max-concurrency": str(batch),
        "--seed": seed,
    }
    if any(argument(name) != value for name, value in expected_arguments.items()):
        raise GateError(f"{owner}: warmup argv differs from the measured run")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise GateError("cannot compute a percentile from an empty sample")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_run(path: Path) -> Run:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: cannot read benchmark JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise GateError(f"{path}: benchmark JSON must be an object")
    if document.get("kvarn_promotable") is not True:
        raise GateError(f"{path}: formal performance evidence must be promotable")

    completed = _positive_integer(
        document.get("completed"), name="completed", path=path
    )
    num_prompts = _positive_integer(
        document.get("num_prompts"), name="num_prompts", path=path
    )
    failed = document.get("failed")
    if failed != 0:
        raise GateError(f"{path}: failed must be zero, got {failed!r}")
    if completed != num_prompts:
        raise GateError(
            f"{path}: completed={completed} must equal num_prompts={num_prompts}"
        )

    duration = _positive_number(document.get("duration"), name="duration", path=path)

    max_concurrency = _positive_integer(
        document.get("max_concurrency"), name="max_concurrency", path=path
    )
    max_concurrent_requests = _positive_integer(
        document.get("max_concurrent_requests"),
        name="max_concurrent_requests",
        path=path,
    )
    if max_concurrent_requests < max_concurrency:
        raise GateError(
            f"{path}: observed concurrency {max_concurrent_requests} is below "
            f"configured max_concurrency={max_concurrency}"
        )
    if completed < 2 * max_concurrency:
        raise GateError(f"{path}: formal benchmark must contain two measured waves")

    input_lens = _integer_list(document.get("input_lens"), name="input_lens", path=path)
    output_lens = _integer_list(
        document.get("output_lens"), name="output_lens", path=path
    )
    if len(set(input_lens)) != 1 or input_lens[0] not in FORMAL_CONTEXTS:
        raise GateError(
            f"{path}: formal benchmark context is not a required matrix cell"
        )
    if len(set(output_lens)) != 1 or output_lens[0] != 512:
        raise GateError(f"{path}: formal benchmark must decode exactly 512 tokens")
    ttfts = document.get("ttfts")
    itls = document.get("itls")
    if not isinstance(ttfts, list) or not isinstance(itls, list):
        raise GateError(f"{path}: --save-detailed ttfts and itls are required")
    lengths = (len(input_lens), len(output_lens), len(ttfts), len(itls))
    if any(length != completed for length in lengths):
        raise GateError(
            f"{path}: detailed arrays {lengths} must all match completed={completed}"
        )
    total_input = _positive_integer(
        document.get("total_input_tokens"), name="total_input_tokens", path=path
    )
    total_output = _positive_integer(
        document.get("total_output_tokens"), name="total_output_tokens", path=path
    )
    if total_input != sum(input_lens) or total_output != sum(output_lens):
        raise GateError(f"{path}: aggregate token counts differ from detailed lengths")

    claimed_metrics = {
        "output_throughput": _positive_number(
            document.get("output_throughput"), name="output_throughput", path=path
        ),
        "request_throughput": _positive_number(
            document.get("request_throughput"), name="request_throughput", path=path
        ),
        "total_token_throughput": _positive_number(
            document.get("total_token_throughput"),
            name="total_token_throughput",
            path=path,
        ),
    }
    recomputed_metrics = {
        "output_throughput": total_output / duration,
        "request_throughput": completed / duration,
        "total_token_throughput": (total_input + total_output) / duration,
    }
    for name, claimed in claimed_metrics.items():
        if not math.isclose(claimed, recomputed_metrics[name], rel_tol=1e-6):
            raise GateError(f"{path}: {name} is inconsistent with duration and counts")

    ttft_values = [
        _positive_number(value, name="ttft", path=path) * 1000.0 for value in ttfts
    ]
    flat_itls: list[float] = []
    request_rates: list[float] = []
    for request, (intervals, output_len) in enumerate(zip(itls, output_lens)):
        if not isinstance(intervals, list):
            raise GateError(f"{path}: itls[{request}] must be a list")
        parsed = [
            _positive_number(value, name=f"itls[{request}]", path=path)
            for value in intervals
        ]
        expected_intervals = output_len - 1
        if len(parsed) != expected_intervals:
            raise GateError(
                f"{path}: itls[{request}] has {len(parsed)} values, expected "
                f"{expected_intervals} for output_len={output_len}"
            )
        if parsed:
            flat_itls.extend(value * 1000.0 for value in parsed)
            request_rates.append(expected_intervals / sum(parsed))
    if not flat_itls or len(request_rates) != completed:
        raise GateError(f"{path}: every completion must contain multiple tokens")

    provenance: dict[str, Any] = {}
    for name in COMMON_PROVENANCE_FIELDS:
        if name in {"num_prompts", "max_concurrency"}:
            provenance[name] = document[name]
        elif name == "request_rate":
            value = document.get(name)
            if value != "inf" and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise GateError(f"{path}: request_rate must be numeric or 'inf'")
            provenance[name] = value
        else:
            provenance[name] = _required_text(document, name, path)
    for name in ARM_PROVENANCE_FIELDS:
        provenance[name] = _required_text(document, name, path)

    for name in (
        "kvarn_process_closure_sha256",
        "kvarn_candidate_closure_sha256",
        "kvarn_matched_profile_sha256",
        "kvarn_hardware_preflight_sha256",
    ):
        provenance[name] = _required_sha256(document, name, path)
    _validate_hardware_preflight(
        provenance["kvarn_hardware_preflight_path"],
        provenance["kvarn_hardware_preflight_sha256"],
        owner=path,
    )
    warmup_path = str(
        Path(_required_text(document, "kvarn_warmup_path", path)).expanduser().resolve()
    )
    warmup_sha256 = _required_sha256(document, "kvarn_warmup_sha256", path)
    run_uuid = _required_text(document, "kvarn_run_uuid", path)
    _validate_warmup(
        warmup_path,
        warmup_sha256,
        owner=path,
        batch=max_concurrency,
        context=input_lens[0],
        output_tokens=output_lens[0],
        seed=provenance["kvarn_seed"],
        arm=provenance["kvarn_arm"],
        run_uuid=run_uuid,
        process_package=provenance["kvarn_process_package"],
        process_closure_sha256=provenance["kvarn_process_closure_sha256"],
        candidate_closure_sha256=provenance["kvarn_candidate_closure_sha256"],
        matched_profile_sha256=provenance["kvarn_matched_profile_sha256"],
    )
    provenance["kvarn_warmup_path"] = warmup_path
    provenance["kvarn_warmup_sha256"] = warmup_sha256
    provenance["kvarn_xpu_consumed_memory_gib"] = _positive_number(
        document.get("kvarn_xpu_consumed_memory_gib"),
        name="kvarn_xpu_consumed_memory_gib",
        path=path,
    )
    provenance["kvarn_xpu_kv_cache_memory_gib"] = _positive_number(
        document.get("kvarn_xpu_kv_cache_memory_gib"),
        name="kvarn_xpu_kv_cache_memory_gib",
        path=path,
    )

    try:
        run_order = int(_required_text(document, "kvarn_run_order", path))
    except ValueError as exc:
        raise GateError(f"{path}: kvarn_run_order must be an integer") from exc
    if run_order < 1:
        raise GateError(f"{path}: kvarn_run_order must be positive")
    engine_log_sha256 = _required_text(document, "kvarn_engine_log_sha256", path)
    if not re.fullmatch(r"[0-9a-f]{64}", engine_log_sha256):
        raise GateError(f"{path}: kvarn_engine_log_sha256 must be lowercase SHA-256")
    started_text = _required_text(document, "kvarn_run_started_at", path)
    try:
        run_started_at = dt.datetime.fromisoformat(started_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GateError(f"{path}: kvarn_run_started_at must be ISO-8601") from exc
    if run_started_at.tzinfo is None:
        raise GateError(f"{path}: kvarn_run_started_at must include a timezone")

    return Run(
        path=str(path),
        completed=completed,
        output_throughput=claimed_metrics["output_throughput"],
        request_throughput=claimed_metrics["request_throughput"],
        total_token_throughput=claimed_metrics["total_token_throughput"],
        request_decode_throughputs=tuple(request_rates),
        ttft_ms=tuple(ttft_values),
        itl_ms=tuple(flat_itls),
        input_lens=input_lens,
        output_lens=output_lens,
        max_concurrent_requests=max_concurrent_requests,
        provenance=provenance,
        run_order=run_order,
        run_uuid=run_uuid,
        run_started_at=run_started_at,
        engine_log_sha256=engine_log_sha256,
    )


def _load_arm(paths: list[Path], name: str) -> list[Run]:
    if len(paths) < 8:
        raise GateError(
            f"at least eight {name} repeats are required for four ABBA pairs"
        )
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise GateError(f"{name} result paths must be unique")
    runs = [_load_run(path) for path in paths]
    first = runs[0]
    for run in runs[1:]:
        if (run.input_lens, run.output_lens) != (
            first.input_lens,
            first.output_lens,
        ):
            raise GateError(f"{run.path}: {name} workload shape differs across repeats")
        for field in (*COMMON_PROVENANCE_FIELDS, *ARM_PROVENANCE_FIELDS):
            if run.provenance[field] != first.provenance[field]:
                raise GateError(f"{run.path}: {name} provenance field {field} differs")
    if first.provenance["kvarn_arm"] != name:
        raise GateError(
            f"{first.path}: kvarn_arm must be {name!r}, got "
            f"{first.provenance['kvarn_arm']!r}"
        )
    warmup_paths = [run.provenance["kvarn_warmup_path"] for run in runs]
    if len(set(warmup_paths)) != len(warmup_paths):
        raise GateError(f"{name} runs must have distinct warmup evidence")
    return runs


def _load_correctness(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: cannot read correctness artifact: {exc}") from exc
    if not isinstance(document, dict) or document.get("status") != "passed":
        raise GateError(f"{path}: correctness status must be passed")
    if document.get("native_dispatch_verified") is not True:
        raise GateError(f"{path}: native_dispatch_verified must be true")
    candidate_id = document.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise GateError(f"{path}: candidate_id must be a non-empty string")
    process_package = _validate_correctness_manifest_identity(document, path)
    gates = document.get("gates")
    if not isinstance(gates, dict) or set(gates) != set(REQUIRED_GATES):
        raise GateError(f"{path}: gates must contain exactly the required gate set")
    for name in REQUIRED_GATES:
        gate = gates.get(name)
        if not isinstance(gate, dict) or gate.get("status") != "passed":
            raise GateError(f"{path}: correctness gate {name} is not passed")
        artifact_path = gate.get("path")
        artifact_sha256 = gate.get("sha256")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise GateError(f"{path}: correctness gate {name} has no artifact path")
        if not isinstance(artifact_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", artifact_sha256
        ):
            raise GateError(f"{path}: correctness gate {name} has invalid SHA-256")
        evidence = Path(artifact_path).expanduser().resolve()
        try:
            actual_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
        except OSError as exc:
            raise GateError(
                f"{path}: cannot read correctness evidence for {name}: {exc}"
            ) from exc
        if actual_sha256 != artifact_sha256:
            raise GateError(f"{path}: correctness evidence SHA differs for {name}")
        try:
            evidence_document = json.loads(evidence.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GateError(
                f"{path}: cannot parse correctness evidence for {name}: {exc}"
            ) from exc
        validate_correctness_gate_evidence(
            name,
            evidence_document,
            path=evidence,
            candidate_id=candidate_id,
            process_package=process_package,
        )
    resolved = path.expanduser().resolve()
    _verify_artifact_references(resolved, owner=path, seen={resolved})
    return document, hashlib.sha256(raw).hexdigest()


def _verify_artifact_references(path: Path, *, owner: Path, seen: set[Path]) -> None:
    if path.suffix != ".json":
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{owner}: cannot read nested evidence {path}: {exc}") from exc

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
            return
        if not isinstance(value, dict):
            return
        nested_path = value.get("path")
        nested_sha256 = value.get("sha256")
        if isinstance(nested_path, str) and isinstance(nested_sha256, str):
            nested = Path(nested_path).expanduser().resolve()
            if not re.fullmatch(r"[0-9a-f]{64}", nested_sha256):
                raise GateError(f"{owner}: nested evidence has invalid SHA-256")
            try:
                actual = hashlib.sha256(nested.read_bytes()).hexdigest()
            except OSError as exc:
                raise GateError(
                    f"{owner}: cannot read nested evidence {nested}: {exc}"
                ) from exc
            if actual != nested_sha256:
                raise GateError(f"{owner}: nested evidence SHA differs: {nested}")
            if nested not in seen:
                seen.add(nested)
                _verify_artifact_references(nested, owner=owner, seen=seen)
        for item in value.values():
            visit(item)

    visit(document)


def _validate_logs(
    paths: list[Path], arm: str, *, expect_native: bool
) -> list[dict[str, Any]]:
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise GateError(f"{arm} engine-log paths must be unique")
    evidence: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise GateError(f"{path}: cannot read engine log: {exc}") from exc
        text = raw.decode("utf-8", errors="replace")
        if scan(text.splitlines())["status"] != "passed":
            raise GateError(f"{path}: engine log contains fatal findings")
        xpu = xpu_runtime_evidence(text.splitlines())
        if not xpu["device_config_xpu"]:
            raise GateError(f"{path}: {arm} log does not report device_config=xpu")
        if not xpu["positive_residency"]:
            raise GateError(f"{path}: {arm} log lacks positive XPU model/KV residency")
        dispatched = NATIVE_DISPATCH in text
        if dispatched != expect_native:
            expectation = "contain" if expect_native else "not contain"
            raise GateError(f"{path}: {arm} log must {expectation} native dispatch")
        if expect_native and FALLBACK_PATTERN.search(text):
            raise GateError(f"{path}: native candidate log reports fallback")
        evidence.append({"sha256": hashlib.sha256(raw).hexdigest(), **xpu})
    return evidence


def _aggregate(runs: list[Run]) -> dict[str, float]:
    return {
        metric: statistics.median(run.metrics()[metric] for run in runs)
        for metric in runs[0].metrics()
    }


def _validate_balanced_order(reference: list[Run], candidate: list[Run]) -> None:
    repeats = len(reference)
    if repeats != len(candidate):
        raise GateError("reference and candidate must have the same number of repeats")
    if repeats % 2:
        raise GateError("repeat count per arm must be even for balanced ABBA order")
    ordered = sorted(
        [(run.run_order, "reference") for run in reference]
        + [(run.run_order, "candidate") for run in candidate]
    )
    if [order for order, _ in ordered] != list(range(1, 2 * repeats + 1)):
        raise GateError(
            "kvarn_run_order must be unique and contiguous across both arms"
        )
    expected = ["reference", "candidate", "candidate", "reference"] * (repeats // 2)
    if [arm for _, arm in ordered] != expected:
        raise GateError(
            "runs must use repeated reference/candidate/candidate/reference order"
        )
    all_runs = reference + candidate
    if len({run.run_uuid for run in all_runs}) != len(all_runs):
        raise GateError("kvarn_run_uuid must be unique across both arms")
    chronological = sorted(all_runs, key=lambda run: run.run_started_at)
    if [run.run_order for run in chronological] != list(range(1, 2 * repeats + 1)):
        raise GateError("chronological run timestamps must match kvarn_run_order")


def _validate_ratio(value: float, *, name: str, lower: float, upper: float) -> None:
    if not math.isfinite(value) or not lower <= value <= upper:
        raise GateError(f"{name} must be finite and in [{lower}, {upper}]")


def compare(
    reference_paths: list[Path],
    candidate_paths: list[Path],
    *,
    reference_logs: list[Path],
    candidate_logs: list[Path],
    correctness_path: Path,
    comparison_kind: str,
    mode: str,
    min_throughput_ratio: float,
    min_request_decode_ratio: float,
    max_latency_ratio: float,
) -> dict[str, Any]:
    if mode not in {"match", "win"}:
        raise GateError(f"unsupported mode {mode!r}")
    _validate_ratio(
        min_throughput_ratio, name="min_throughput_ratio", lower=0.5, upper=1.0
    )
    _validate_ratio(
        min_request_decode_ratio,
        name="min_request_decode_ratio",
        lower=0.5,
        upper=1.0,
    )
    _validate_ratio(max_latency_ratio, name="max_latency_ratio", lower=1.0, upper=2.0)
    if (
        min_throughput_ratio < 0.95
        or min_request_decode_ratio < 0.95
        or max_latency_ratio > 1.10
    ):
        raise GateError(
            "formal comparison thresholds must be at least 0.95 throughput/decode "
            "and no more than 1.10 latency"
        )
    reference = _load_arm(reference_paths, "reference")
    candidate = _load_arm(candidate_paths, "candidate")
    warmup_paths = [
        run.provenance["kvarn_warmup_path"] for run in (*reference, *candidate)
    ]
    if len(set(warmup_paths)) != len(warmup_paths):
        raise GateError("every reference/candidate run needs distinct warmup evidence")
    _validate_balanced_order(reference, candidate)
    if len(reference_logs) != len(reference) or len(candidate_logs) != len(candidate):
        raise GateError("each result must have one engine log in the same arm order")

    first_ref = reference[0]
    first_cand = candidate[0]
    if (first_ref.input_lens, first_ref.output_lens) != (
        first_cand.input_lens,
        first_cand.output_lens,
    ):
        raise GateError("reference and candidate workload shapes differ")
    for field in COMMON_PROVENANCE_FIELDS:
        if first_ref.provenance[field] != first_cand.provenance[field]:
            raise GateError(f"reference and candidate provenance field {field} differs")

    expected_profile = {
        "backend": "openai",
        "model_id": "sunny-chat",
        "tokenizer_id": "jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound",
        "request_rate": "inf",
        "kvarn_model_revision": "6b0622f4354481d5d04577d48ba0db844efc1330",
        "kvarn_max_model_len": "65536",
        "kvarn_enforce_eager": "1",
        "kvarn_prefix_caching": "0",
        "kvarn_mtp": "0",
        "kvarn_xpu_graph": "0",
        "kvarn_accelerator": "xpu",
        "kvarn_xpu_available": "1",
        "kvarn_xpu_device_count": "1",
        "kvarn_xpu_device_name": EXPECTED_XPU_DEVICE_NAME,
        "kvarn_xpu_compute_probe": "passed",
        "kvarn_evidence_mode": "formal",
    }
    for field, expected_value in expected_profile.items():
        if first_ref.provenance[field] != expected_value:
            raise GateError(
                f"performance profile requires {field}={expected_value!r}, got "
                f"{first_ref.provenance[field]!r}"
            )
    concurrency = first_ref.provenance["max_concurrency"]
    if concurrency not in {1, 4}:
        raise GateError("performance max_concurrency must be 1 or 4")
    try:
        max_num_seqs = int(first_ref.provenance["kvarn_max_num_seqs"])
        scheduler_peak = int(first_ref.provenance["kvarn_scheduler_peak_running"])
    except ValueError as exc:
        raise GateError(
            "kvarn_max_num_seqs and kvarn_scheduler_peak_running must be integers"
        ) from exc
    if max_num_seqs != concurrency:
        raise GateError("kvarn_max_num_seqs must equal benchmark max_concurrency")
    if scheduler_peak < concurrency:
        raise GateError("scheduler evidence did not reach the requested concurrency")

    correctness, correctness_sha256 = _load_correctness(correctness_path)
    candidate_id = first_cand.provenance["kvarn_candidate_id"]
    if correctness["candidate_id"] != candidate_id:
        raise GateError(
            "correctness artifact candidate_id differs from benchmark candidate"
        )
    if first_cand.provenance["kvarn_correctness_sha256"] != correctness_sha256:
        raise GateError(
            "benchmark correctness SHA-256 does not match the supplied artifact"
        )
    correctness_identity = {
        field: correctness["candidate_identity"][field]
        for field in (
            "process_package",
            "candidate_closure_sha256",
            "process_closure_sha256",
        )
    }
    benchmark_identity = {
        "process_package": first_cand.provenance["kvarn_process_package"],
        "candidate_closure_sha256": first_cand.provenance[
            "kvarn_candidate_closure_sha256"
        ],
        "process_closure_sha256": first_cand.provenance["kvarn_process_closure_sha256"],
    }
    if correctness_identity != benchmark_identity:
        raise GateError(
            "correctness and performance evidence identify different candidate builds"
        )

    ref_dtype = first_ref.provenance["kvarn_kv_cache_dtype"]
    cand_dtype = first_cand.provenance["kvarn_kv_cache_dtype"]
    ref_native = first_ref.provenance["kvarn_native_xpu"]
    cand_native = first_cand.provenance["kvarn_native_xpu"]
    if (ref_native, cand_native) != ("0", "1"):
        raise GateError("reference must disable and candidate must enable native XPU")
    try:
        ref_splits = int(first_ref.provenance["kvarn_native_splits"])
        cand_splits = int(first_cand.provenance["kvarn_native_splits"])
    except ValueError as exc:
        raise GateError("kvarn_native_splits must be an integer") from exc
    if ref_splits != 1:
        raise GateError("non-native reference must declare the neutral split count 1")
    if cand_splits not in {1, 2, 4, 8, 16, 17, 24, 32}:
        raise GateError("candidate must declare a supported native split count")
    if comparison_kind == "kernel":
        if (ref_dtype, cand_dtype) != (COMPACT_DTYPE, COMPACT_DTYPE):
            raise GateError("kernel comparison requires compact Kvarn in both arms")
    elif comparison_kind == "end-to-end":
        if (ref_dtype, cand_dtype) != ("auto", COMPACT_DTYPE):
            raise GateError("end-to-end comparison requires auto versus compact Kvarn")
    else:
        raise GateError(f"unsupported comparison kind {comparison_kind!r}")

    reference_log_evidence = _validate_logs(
        reference_logs, "reference", expect_native=False
    )
    candidate_log_evidence = _validate_logs(
        candidate_logs, "candidate", expect_native=True
    )
    for run, evidence in zip(reference, reference_log_evidence):
        if run.engine_log_sha256 != evidence["sha256"]:
            raise GateError(f"{run.path}: reference engine-log SHA-256 differs")
        if not math.isclose(
            run.provenance["kvarn_xpu_consumed_memory_gib"],
            evidence["consumed_memory_gib"],
        ) or not math.isclose(
            run.provenance["kvarn_xpu_kv_cache_memory_gib"],
            evidence["kv_cache_memory_gib"],
        ):
            raise GateError(f"{run.path}: reference XPU residency evidence differs")
    for run, evidence in zip(candidate, candidate_log_evidence):
        if run.engine_log_sha256 != evidence["sha256"]:
            raise GateError(f"{run.path}: candidate engine-log SHA-256 differs")
        if not math.isclose(
            run.provenance["kvarn_xpu_consumed_memory_gib"],
            evidence["consumed_memory_gib"],
        ) or not math.isclose(
            run.provenance["kvarn_xpu_kv_cache_memory_gib"],
            evidence["kv_cache_memory_gib"],
        ):
            raise GateError(f"{run.path}: candidate XPU residency evidence differs")

    reference_log_sha256 = [item["sha256"] for item in reference_log_evidence]
    candidate_log_sha256 = [item["sha256"] for item in candidate_log_evidence]

    ref = _aggregate(reference)
    cand = _aggregate(candidate)
    ratios = {metric: cand[metric] / ref[metric] for metric in ref}
    effective_throughput = 1.0 if mode == "win" else min_throughput_ratio
    effective_request = 1.0 if mode == "win" else min_request_decode_ratio
    effective_latency = 1.0 if mode == "win" else max_latency_ratio

    checks = {
        "output_throughput": ratios["output_throughput"] >= effective_throughput,
        "request_throughput": ratios["request_throughput"] >= effective_throughput,
        "total_token_throughput": (
            ratios["total_token_throughput"] >= effective_throughput
        ),
        "median_request_decode_throughput": (
            ratios["median_request_decode_throughput"] >= effective_request
        ),
        "p10_request_decode_throughput": (
            ratios["p10_request_decode_throughput"] >= effective_request
        ),
        "p99_ttft": ratios["p99_ttft_ms"] <= effective_latency,
        "p99_itl": ratios["p99_itl_ms"] <= effective_latency,
    }
    if mode == "win":
        checks["meaningful_throughput_gain"] = (
            max(
                ratios["output_throughput"],
                ratios["median_request_decode_throughput"],
            )
            >= 1.05
        )

    return {
        "status": "passed" if all(checks.values()) else "failed",
        "mode": mode,
        "comparison_kind": comparison_kind,
        "thresholds": {
            "min_throughput_ratio": effective_throughput,
            "min_request_decode_ratio": effective_request,
            "max_latency_ratio": effective_latency,
            "meaningful_win_ratio": 1.05 if mode == "win" else None,
        },
        "workload": {
            "input_lens": list(first_ref.input_lens),
            "output_lens": list(first_ref.output_lens),
            "max_concurrency": first_ref.provenance["max_concurrency"],
            "num_prompts": first_ref.provenance["num_prompts"],
        },
        "provenance": {
            field: first_ref.provenance[field] for field in COMMON_PROVENANCE_FIELDS
        },
        "correctness_artifact": {
            "path": str(correctness_path),
            "sha256": correctness_sha256,
        },
        "reference": {
            "runs": [run.path for run in reference],
            "engine_log_sha256": reference_log_sha256,
            "arm": {
                field: first_ref.provenance[field] for field in ARM_PROVENANCE_FIELDS
            },
            "median": ref,
        },
        "candidate": {
            "runs": [run.path for run in candidate],
            "engine_log_sha256": candidate_log_sha256,
            "arm": {
                field: first_cand.provenance[field] for field in ARM_PROVENANCE_FIELDS
            },
            "median": cand,
        },
        "candidate_over_reference": ratios,
        "checks": checks,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="append", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--reference-log", action="append", type=Path, required=True)
    parser.add_argument("--candidate-log", action="append", type=Path, required=True)
    parser.add_argument("--correctness", type=Path, required=True)
    parser.add_argument(
        "--comparison-kind", choices=("kernel", "end-to-end"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    parser.add_argument("--mode", choices=("match", "win"), default="match")
    parser.add_argument("--min-throughput-ratio", type=float, default=0.95)
    parser.add_argument("--min-request-decode-ratio", type=float, default=0.95)
    parser.add_argument("--max-latency-ratio", type=float, default=1.10)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    inputs = {
        path.expanduser().resolve()
        for path in (
            args.reference
            + args.candidate
            + args.reference_log
            + args.candidate_log
            + [args.correctness]
        )
    }
    if output in inputs:
        parser.error("--output must not overwrite an input artifact")
    if not args.allow_tmp and output.is_relative_to(Path("/tmp")):
        parser.error("--output must be durable (outside /tmp)")
    args.output = output
    return args


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = _parse_args()
    try:
        result = compare(
            args.reference,
            args.candidate,
            reference_logs=args.reference_log,
            candidate_logs=args.candidate_log,
            correctness_path=args.correctness,
            comparison_kind=args.comparison_kind,
            mode=args.mode,
            min_throughput_ratio=args.min_throughput_ratio,
            min_request_decode_ratio=args.min_request_decode_ratio,
            max_latency_ratio=args.max_latency_ratio,
        )
    except GateError as exc:
        result = {"status": "invalid", "error": str(exc)}
    _write_json_atomic(args.output, result)
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
