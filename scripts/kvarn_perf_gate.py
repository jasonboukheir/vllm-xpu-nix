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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from scripts import kvarn_split_policy as split_policy
    from scripts.kvarn_scan_engine_log import scan, xpu_runtime_evidence
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import kvarn_split_policy as split_policy
    from kvarn_scan_engine_log import scan, xpu_runtime_evidence

NATIVE_DISPATCH = "Using the native Xe2 KVarN qlen=1 decoder"
NATIVE_DIRECT_BF16_MARKER = "direct bf16 output=True"
NATIVE_DIRECT_BF16_DISABLED_MARKER = "direct bf16 output=False"
FALLBACK_PATTERN = re.compile(
    r"(?i)(?:\bkvarn\b[^\n]{0,120}\b(?:fallback|falling back)\b|"
    r"\b(?:fallback|falling back)\b[^\n]{0,120}\bkvarn\b)"
)
COMPACT_DTYPE = "kvarn_k4v4_g128_compact"
NATIVE_LAYOUTS = ("natural", "xe2_dpas")
NATIVE_LAYOUT_ENV = {"natural": "0", "xe2_dpas": "1"}
NATIVE_KERNEL_VARIANTS = {
    "baseline": 0,
    "qk_i8u4": 1,
    "q6_scalar": 2,
    "q8_vector": 3,
    "q6_vector": 4,
    "q6_cached_weights": 6,
    "q6_exact_rows": 7,
    "q6_cached_weights_exact_rows": 8,
    "q6_page_pair": 9,
    "q6_main_grf128": 10,
    "q6_split_reducer_specialized": 11,
    "q6_next_page_prefetch": 12,
    "q6_next_page_prefetch_split_reducer": 13,
    "q6_simd_unpack": 14,
    "q6_block_output_store": 15,
    "q6_current_half_v_prefetch": 16,
    "q6_page_record_cursor": 17,
    "q6_prefetch_record_cursor": 18,
    "q6_page_metadata_cursor": 20,
    "q6_paired_nibble_half2": 21,
}
NATIVE_SPLIT_POLICIES = split_policy.NATIVE_SPLIT_POLICIES
FLUSH_INDEX_MATERIALIZATION_VARIANTS = ("per_layer", "shared")
FLUSH_WRITER_VARIANTS = ("reference", "native_xe2", "sinkhorn_pack_xe2")
PREFILL_STORE_VARIANTS = ("reference", "hadamard_scatter")
NATIVE_FRONTEND_VARIANTS = ("reference", "qkv_scatter", "qkv_scatter_inline")
NATIVE_FRONTEND_ACTIVE_MARKER = "[KVARN_FRONTEND] active=qkv_scatter;"
NATIVE_FRONTEND_INLINE_ACTIVE_MARKER = (
    "[KVARN_FRONTEND_INLINE] active=qkv_scatter_inline; "
    "wrapper=unified_qkv_attention_with_output;"
)
FORWARD_POOL_ENSURE_VARIANTS = ("always", "fused_qkv_proof")
FORWARD_POOL_ENSURE_ACTIVE_MARKER = (
    "[KVARN_FORWARD_POOL_ENSURE] active=fused_qkv_proof; "
    "action=elide_ensure_pool;"
)
FILTERED_SOURCE_SCHEME = "nix-filtered-source-store-hash-v1"
NIX_STORE_HASH = re.compile(r"^[0-9abcdfghijklmnpqrsvwxyz]{32}$")
DEFAULT_NATIVE_SPLITS = {1: 24, 4: 16}
B70_Q6_SPLITS = split_policy.B70_Q6_SPLITS
B70_Q6_MAX_SPLITS = split_policy.B70_Q6_MAX_SPLITS
B70_Q6_KERNEL_VARIANTS = frozenset(
    {
        "q6_scalar",
        "q6_vector",
        "q6_cached_weights",
        "q6_exact_rows",
        "q6_cached_weights_exact_rows",
        "q6_page_pair",
        "q6_main_grf128",
        "q6_split_reducer_specialized",
        "q6_next_page_prefetch",
        "q6_next_page_prefetch_split_reducer",
        "q6_simd_unpack",
        "q6_block_output_store",
        "q6_current_half_v_prefetch",
        "q6_page_record_cursor",
        "q6_prefetch_record_cursor",
        "q6_page_metadata_cursor",
        "q6_paired_nibble_half2",
    }
)
COMBINED_LIBRARY_VARIANT_MATRIX = [
    {
        "cache_layout": "xe2_dpas",
        "kernel_variant": kernel_variant,
        "kernel_variant_id": NATIVE_KERNEL_VARIANTS[kernel_variant],
    }
    for kernel_variant in (
        "q6_scalar",
        "q6_vector",
        "q6_cached_weights",
        "q6_exact_rows",
        "q6_cached_weights_exact_rows",
        "q6_page_pair",
        "q6_main_grf128",
        "q6_split_reducer_specialized",
        "q6_next_page_prefetch",
        "q6_next_page_prefetch_split_reducer",
        "q6_simd_unpack",
        "q6_block_output_store",
        "q6_current_half_v_prefetch",
        "q6_page_record_cursor",
        "q6_prefetch_record_cursor",
        "q6_page_metadata_cursor",
        "q6_paired_nibble_half2",
    )
]
VARIANT_FIELDS = (
    "kernel_strategy",
    "split_policy",
    "fusion_strategy",
    "scheduling_variant",
    "variant_id",
)
PRIMITIVE_SERVICE_ONLY_FIELDS = (
    "native_frontend",
    "kvarn_native_frontend",
    "native_frontend_active_verified",
    "native_frontend_log_marker",
    "native_frontend_inline_active_verified",
    "native_frontend_inline_log_marker",
    "forward_pool_ensure",
    "kvarn_forward_pool_ensure",
    "forward_pool_ensure_active_verified",
    "forward_pool_ensure_log_marker",
)
EXPECTED_XPU_DEVICE_NAME = "Intel(R) Arc(TM) Pro B70 Graphics"
FORMAL_CONTEXTS = frozenset({4096, 16384, 32768, 65023})
FACTORY_QUALIFICATION_CONTEXTS = (4096, 16384, 65023)
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
    },
    "native-65k-b1-restart": {
        "name": "native-65k-b1-restart",
        "launcher": "vllm-xpu-brutus-kvarn-native-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 65536,
    },
    "native-65k-b4": {
        "name": "native-65k-b4",
        "launcher": "vllm-xpu-brutus-kvarn-native-b4",
        "native": True,
        "batch": 4,
        "max_model_len": 65536,
    },
    "reference-262k-b1": {
        "name": "reference-262k-b1",
        "launcher": "vllm-xpu-brutus-kvarn-262k-b1",
        "native": False,
        "batch": 1,
        "max_model_len": 262144,
    },
    "native-262k-b1-first": {
        "name": "native-262k-b1-first",
        "launcher": "vllm-xpu-brutus-kvarn-native-262k-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 262144,
    },
    "native-262k-b1-restart": {
        "name": "native-262k-b1-restart",
        "launcher": "vllm-xpu-brutus-kvarn-native-262k-b1",
        "native": True,
        "batch": 1,
        "max_model_len": 262144,
    },
}


def _candidate_variant_provenance(
    native_layout: str,
    native_frontend: str,
    flush_index_materialization: str,
    native_kernel_variant: str = "baseline",
    native_split_policy: str = "fixed",
    native_splits: Mapping[int, int] | None = None,
    flush_writer: str = "reference",
    prefill_store: str = "reference",
    forward_pool_ensure: str = "always",
) -> dict[str, str]:
    selected_splits = DEFAULT_NATIVE_SPLITS if native_splits is None else native_splits
    split_policy = native_split_policy
    if split_policy == "fixed":
        split_policy += "_" + "_".join(
            f"b{batch}s{splits}" for batch, splits in sorted(selected_splits.items())
        )
    scheduling = "eager_mnbt2048"
    return {
        "kernel_strategy": f"native_xe2_qlen1_{native_kernel_variant}",
        "split_policy": split_policy,
        "fusion_strategy": (
            "native_materializer_persistent_scratch_"
            f"{flush_index_materialization}_indices_{flush_writer}_writer_"
            f"{prefill_store}_prefill_store_{native_frontend}_frontend_"
            f"{forward_pool_ensure}_forward_pool_ensure"
        ),
        "scheduling_variant": scheduling,
        "variant_id": (
            f"native-xe2-{native_layout}-{native_kernel_variant}-"
            f"{split_policy}-{flush_index_materialization}-indices-"
            f"{flush_writer}-writer-{prefill_store}-prefill-store-"
            f"{native_frontend}-frontend-"
            f"{forward_pool_ensure}-forward-pool-ensure-{scheduling}"
        ),
    }


def _correctness_reference_variant_provenance() -> dict[str, str]:
    return {
        "kernel_strategy": "kvarn_non_native",
        "split_policy": "neutral_1",
        "fusion_strategy": "none",
        "scheduling_variant": "eager_mnbt2048",
        "variant_id": "natural-kvarn-correctness-reference-eager_mnbt2048",
    }


def _performance_reference_variant_provenance() -> dict[str, str]:
    return {
        "kernel_strategy": "vllm_auto",
        "split_policy": "neutral_1",
        "fusion_strategy": "vllm_auto",
        "scheduling_variant": "eager_mnbt2048",
        "variant_id": "auto-control-eager_mnbt2048",
    }


def _correctness_phase_spec(
    phase_name: str,
    native_layout: str,
    native_frontend: str,
    flush_index_materialization: str,
    native_kernel_variant: str = "baseline",
    native_split_policy: str = "fixed",
    native_splits: Mapping[int, int] | None = None,
    request_stable_projection_rows: str = "1",
    request_stable_rmsnorm: str = "1",
    flush_writer: str = "reference",
    prefill_store: str = "reference",
    forward_pool_ensure: str = "always",
) -> dict[str, Any]:
    spec = dict(CORRECTNESS_PHASE_SPECS[phase_name])
    selected_splits = DEFAULT_NATIVE_SPLITS if native_splits is None else native_splits
    effective_layout = native_layout if spec["native"] else "natural"
    effective_frontend = native_frontend if spec["native"] else "reference"
    effective_flush_writer = flush_writer if spec["native"] else "reference"
    effective_prefill_store = prefill_store if spec["native"] else "reference"
    effective_forward_pool_ensure = forward_pool_ensure if spec["native"] else "always"
    effective_projection_rows = (
        request_stable_projection_rows if spec["native"] else "1"
    )
    effective_rmsnorm = request_stable_rmsnorm if spec["native"] else "1"
    if spec["native"] and native_layout == "xe2_dpas":
        if native_split_policy == "b70_q6_v2":
            spec["launcher"] = "vllm-xpu-brutus-kvarn-factory-runtime"
        else:
            suffix = "-262k" if spec["max_model_len"] == 262144 else ""
            spec["launcher"] = (
                "vllm-xpu-brutus-kvarn-native-dpas-"
                f"{native_kernel_variant}{suffix}-b{spec['batch']}"
            )
    spec["native_layout"] = effective_layout
    selected_kernel = native_kernel_variant if spec["native"] else "baseline"
    selected_policy = native_split_policy if spec["native"] else "fixed"
    context_dependent = selected_policy == "b70_q6_v2"
    effective_splits = (
        None
        if spec["native"] and context_dependent
        else selected_splits[spec["batch"]]
        if spec["native"]
        else 1
    )
    policy_contract = (
        split_policy.split_policy_contract(
            selected_policy, selected_splits or None
        )
        if spec["native"]
        else None
    )
    max_splits = (
        int(policy_contract["scratch_max_splits"])
        if policy_contract is not None
        else effective_splits
    )
    spec.update(
        native_kernel_variant=selected_kernel,
        native_kernel_variant_id=NATIVE_KERNEL_VARIANTS[selected_kernel],
        native_frontend=effective_frontend,
        flush_writer=effective_flush_writer,
        prefill_store=effective_prefill_store,
        forward_pool_ensure=effective_forward_pool_ensure,
        native_split_policy=selected_policy,
        native_split_policy_contract=policy_contract,
        max_decode_splits=max_splits,
        nominal_decode_splits=effective_splits,
        request_stable_projection_rows=effective_projection_rows,
        request_stable_rmsnorm=effective_rmsnorm,
    )
    spec.update(
        _candidate_variant_provenance(
            native_layout,
            native_frontend,
            flush_index_materialization,
            native_kernel_variant,
            native_split_policy,
            selected_splits,
            flush_writer,
            prefill_store,
            forward_pool_ensure,
        )
        if spec["native"]
        else _correctness_reference_variant_provenance()
    )
    return spec


def _factory_marker(
    cache_layout: str,
    kernel_variant: str,
    max_decode_splits: int,
    split_policy: str,
) -> str:
    try:
        kernel_variant_id = NATIVE_KERNEL_VARIANTS[kernel_variant]
    except KeyError as exc:
        raise GateError(f"unknown native kernel variant {kernel_variant!r}") from exc
    return (
        f"[KVARN_FACTORY] selected_cache_layout={cache_layout}; "
        f"selected_kernel_variant={kernel_variant}({kernel_variant_id}); "
        f"max_decode_splits={max_decode_splits}; "
        f"selected_split_policy={split_policy}; immutable for engine lifetime"
    )


CORRECTNESS_RUNNER_SOURCES = {
    "scripts/kvarn_correctness_run.py",
    "scripts/kvarn_perf_gate.py",
    "scripts/kvarn_perf_run.py",
    "scripts/kvarn_scan_engine_log.py",
    "scripts/kvarn_service_gate.py",
    "scripts/kvarn_split_policy.py",
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
    "kvarn_onednn_deterministic",
    "kvarn_request_stable_projection_rows",
    "kvarn_request_stable_rmsnorm",
    "kvarn_request_stability_qualification",
    "kvarn_vllm_use_v2_model_runner",
)
ARM_PROVENANCE_FIELDS = (
    "kvarn_arm",
    "kvarn_kv_cache_dtype",
    "kvarn_native_xpu",
    "kvarn_native_layout",
    "kvarn_native_layout_environment",
    "kvarn_native_cache_layout_environment",
    "kvarn_native_kernel_variant",
    "kvarn_native_kernel_variant_id",
    "kvarn_native_output_dtype",
    "kvarn_native_direct_bf16_verified",
    "kvarn_native_direct_bf16_log_marker",
    "kvarn_native_max_splits",
    "kvarn_native_nominal_splits",
    "kvarn_native_split_policy",
    "kvarn_flush_index_materialization",
    "kvarn_flush_writer",
    "kvarn_prefill_store",
    "kvarn_native_frontend",
    "kvarn_forward_pool_ensure",
    "kvarn_native_frontend_active_verified",
    "kvarn_native_frontend_log_marker",
    "kvarn_native_frontend_inline_active_verified",
    "kvarn_native_frontend_inline_log_marker",
    "kvarn_forward_pool_ensure_active_verified",
    "kvarn_forward_pool_ensure_log_marker",
    "kvarn_native_layout_log_marker",
    "kvarn_native_layout_evidence",
    "kvarn_kernel_strategy",
    "kvarn_split_policy",
    "kvarn_fusion_strategy",
    "kvarn_scheduling_variant",
    "kvarn_variant_id",
)


class GateError(ValueError):
    """Raised when inputs do not form a valid matched comparison."""


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_factory_qualification(
    factory_path: Path,
    *,
    native_layout: str,
    native_kernel_variant: str,
    native_split_policy: str,
    native_splits: Mapping[int, int],
    output_dtype: str,
    expected_revisions: Mapping[str, str],
    expected_package: str,
    expected_native_library: str | None = None,
    expected_native_library_sha256: str | None = None,
    flush_writer: str = "reference",
    prefill_store: str = "reference",
) -> dict[str, Any]:
    """Validate and bind the selected per-ID primitive factory matrix."""
    path = factory_path.expanduser().resolve()
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: cannot read factory qualification: {exc}") from exc
    if not isinstance(document, dict):
        raise GateError(f"{path}: factory qualification must be an object")
    if (
        document.get("schema_version") != 3
        or document.get("artifact_kind") != "kvarn_b70_primitive_factory_run"
        or document.get("status") != "completed_primitive_diagnostic"
        or document.get("identity_stable_through_sweep") is not True
        or document.get("evidence_identity_sha256")
        != document.get("ending_evidence_identity_sha256")
        or not isinstance(document.get("evidence_identity_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", document["evidence_identity_sha256"])
    ):
        raise GateError(f"{path}: factory artifact identity/status is invalid")
    if native_layout != "xe2_dpas":
        raise GateError(f"{path}: selected per-ID factory result requires xe2_dpas")
    if native_kernel_variant not in NATIVE_KERNEL_VARIANTS:
        raise GateError(f"{path}: selected factory kernel variant is unsupported")
    if native_split_policy not in NATIVE_SPLIT_POLICIES:
        raise GateError(f"{path}: selected factory split policy is unsupported")
    if output_dtype not in {"fp16", "bf16"}:
        raise GateError(f"{path}: selected factory output dtype is unsupported")
    if flush_writer not in FLUSH_WRITER_VARIANTS:
        raise GateError(f"{path}: selected factory flush writer is unsupported")
    if flush_writer != "reference" and native_layout != "xe2_dpas":
        raise GateError(f"{path}: native Kvarn writer requires xe2_dpas")
    if prefill_store not in PREFILL_STORE_VARIANTS:
        raise GateError(f"{path}: selected factory prefill store is unsupported")
    if native_split_policy == "b70_q6_v2":
        if native_splits:
            raise GateError(
                f"{path}: context-dependent policy must not use a batch-only "
                "split map"
            )
    elif set(native_splits) != {1, 4}:
        raise GateError(f"{path}: selected factory split map must cover B1 and B4")
    try:
        split_policy.validate_kernel_compatibility(
            native_split_policy,
            native_kernel_variant,
            q6_variants=B70_Q6_KERNEL_VARIANTS,
        )
        policy_contract = split_policy.split_policy_contract(
            native_split_policy, native_splits or None
        )
    except ValueError as exc:
        raise GateError(f"{path}: invalid split policy selection: {exc}") from exc

    source_revisions = document.get("source_revisions")
    expected_sources = dict(expected_revisions)
    if (
        not isinstance(source_revisions, dict)
        or source_revisions.get("verified") is not True
        or source_revisions.get("expected") != expected_sources
        or source_revisions.get("actual") != expected_sources
    ):
        raise GateError(f"{path}: factory source revisions differ from correctness")
    repositories = document.get("repositories")
    if not isinstance(repositories, list) or len(repositories) != len(expected_sources):
        raise GateError(f"{path}: factory repository evidence is incomplete")
    repository_map = {
        item.get("name"): item for item in repositories if isinstance(item, dict)
    }
    if set(repository_map) != set(expected_sources) or any(
        repository_map[name].get("head") != revision
        or repository_map[name].get("dirty") is not False
        or repository_map[name].get("status_porcelain") != []
        for name, revision in expected_sources.items()
    ):
        raise GateError(f"{path}: factory repository source evidence differs")

    runtime = document.get("runtime_environment")
    hardware = document.get("hardware_preflight")
    fixture = document.get("fixture_matching")
    kill_suite = document.get("kernel_kill_suite")
    if (
        not isinstance(runtime, dict)
        or runtime.get("prefixed_environment_clean") is not True
        or runtime.get("kvarn_or_vllm_prefixed_variables") != {}
        or not isinstance(hardware, dict)
        or hardware.get("passed") is not True
        or hardware.get("selected_device") != "xpu:0"
        or hardware.get("selected_device_name") != EXPECTED_XPU_DEVICE_NAME
        or not isinstance(fixture, dict)
        or fixture.get("fixture_mode") != "matched-production"
        or fixture.get("validation_status") != "passed"
        or fixture.get("logical_kv_payloads_matched_between_auto_and_kvarn") is not True
        or fixture.get("matched_primitive_fixture_eligible") is not True
        or fixture.get("matched_primitive_ratio_eligible") is not True
        or not isinstance(kill_suite, dict)
        or kill_suite.get("status") != "passed"
        or kill_suite.get("passed") is not True
        or kill_suite.get("returncode") != 0
        or kill_suite.get("skipped_count") != 0
        or kill_suite.get("flush_writer") != flush_writer
        or kill_suite.get("prefill_store") != prefill_store
    ):
        raise GateError(f"{path}: factory XPU correctness preconditions are invalid")

    libraries = document.get("libraries")
    builds = document.get("build_attestations")
    ownership = document.get("source_ownership")
    if (
        not isinstance(libraries, dict)
        or not isinstance(libraries.get("flash"), dict)
        or not isinstance(libraries.get("native_attention"), dict)
        or not isinstance(builds, dict)
        or not isinstance(builds.get("package"), dict)
        or not isinstance(builds.get("flash"), dict)
        or not isinstance(builds.get("native_attention"), dict)
        or not isinstance(ownership, dict)
        or ownership.get("verified") is not True
    ):
        raise GateError(f"{path}: factory build provenance is incomplete")
    package = builds["package"]
    flash_build = builds["flash"]
    native_attention_build = builds["native_attention"]
    flash = libraries["flash"]
    native_attention = libraries["native_attention"]
    owned_artifacts = ownership.get("artifacts")
    if (
        package.get("verified") is not True
        or package.get("output_path") != expected_package
        or flash_build.get("verified") is not True
        or flash_build.get("library_path") != flash.get("path")
        or flash_build.get("output_path") not in package.get("closure_paths", [])
        or native_attention_build.get("verified") is not True
        or native_attention_build.get("library_path") != native_attention.get("path")
        or native_attention_build.get("output_path")
        not in package.get("closure_paths", [])
        or not isinstance(owned_artifacts, dict)
        or any(
            not isinstance(owned_artifacts.get(name), dict)
            or owned_artifacts[name].get("verified") is not True
            or owned_artifacts[name].get("member_of_package_closure") is not True
            for name in ("package", "base", "flash", "native_attention")
        )
    ):
        raise GateError(f"{path}: factory package/library provenance differs")
    native_source_contract = native_attention_build.get("source_contract")
    nix_evaluation_identity = (
        native_source_contract.get("nix_evaluation_identity")
        if isinstance(native_source_contract, dict)
        else None
    )
    native_source_identity = (
        native_source_contract.get("artifact_identity")
        if isinstance(native_source_contract, dict)
        else None
    )
    compatibility = (
        native_source_contract.get("compatibility_provenance")
        if isinstance(native_source_contract, dict)
        else None
    )
    native_owner = owned_artifacts["native_attention"]
    native_derivation = native_attention_build.get("derivation")
    source_hash = (
        native_source_identity.get("filtered_source_store_hash")
        if isinstance(native_source_identity, dict)
        else None
    )
    source_marker = f"+src.{source_hash}"
    if (
        not isinstance(native_source_contract, dict)
        or not isinstance(nix_evaluation_identity, dict)
        or nix_evaluation_identity.get("output_path")
        != native_attention_build.get("output_path")
        or nix_evaluation_identity.get("derivation") != native_derivation
        or not isinstance(native_source_identity, dict)
        or native_source_identity.get("scheme") != FILTERED_SOURCE_SCHEME
        or not isinstance(source_hash, str)
        or not NIX_STORE_HASH.fullmatch(source_hash)
        or not isinstance(native_derivation, str)
        or source_marker not in Path(native_derivation).name
        or native_owner.get("repository") != "vllm-xpu-kernels"
        or native_owner.get("compatible_upstream_revision")
        != expected_sources["vllm-xpu-kernels"]
        or native_owner.get("derivation") != native_derivation
        or native_owner.get("derivation_source_marker") != source_marker
        or native_owner.get("artifact_identity") != native_source_identity
        or native_owner.get("nix_evaluation_identity") != nix_evaluation_identity
        or native_owner.get("compatibility_source") != "factory_nix_evaluation"
        or compatibility
        != {
            "upstream_revision": expected_sources["vllm-xpu-kernels"],
            "asserted_against_expected_repository_revision": True,
        }
    ):
        raise GateError(f"{path}: factory native-attention source identity differs")
    flash_path = flash.get("path")
    flash_sha256 = flash.get("sha256")
    if (
        not isinstance(flash_path, str)
        or not isinstance(flash_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", flash_sha256)
        or expected_native_library is not None
        and str(Path(expected_native_library).expanduser().resolve()) != flash_path
        or expected_native_library_sha256 is not None
        and expected_native_library_sha256 != flash_sha256
    ):
        raise GateError(f"{path}: factory native-library identity differs")
    try:
        actual_flash_sha256 = hashlib.sha256(Path(flash_path).read_bytes()).hexdigest()
    except OSError as exc:
        raise GateError(f"{path}: cannot rehash factory native library: {exc}") from exc
    if actual_flash_sha256 != flash_sha256:
        raise GateError(f"{path}: factory native-library artifact changed")
    native_attention_path = native_attention.get("path")
    native_attention_sha256 = native_attention.get("sha256")
    runtime_binding = document.get("native_attention_runtime_binding")
    if (
        not isinstance(native_attention_path, str)
        or Path(native_attention_path).name != "libattn_kernels_xe_2.so"
        or not isinstance(native_attention_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", native_attention_sha256)
        or not isinstance(runtime_binding, dict)
        or runtime_binding.get("status") != "verified"
        or runtime_binding.get("expected_path") != native_attention_path
        or runtime_binding.get("mapped_path") != native_attention_path
        or runtime_binding.get("basename") != "libattn_kernels_xe_2.so"
        or runtime_binding.get("unique_basename_mapping") is not True
    ):
        raise GateError(f"{path}: factory native-attention identity differs")
    try:
        actual_native_attention_sha256 = hashlib.sha256(
            Path(native_attention_path).read_bytes()
        ).hexdigest()
    except OSError as exc:
        raise GateError(
            f"{path}: cannot rehash factory native attention library: {exc}"
        ) from exc
    if actual_native_attention_sha256 != native_attention_sha256:
        raise GateError(f"{path}: factory native-attention artifact changed")

    results = document.get("results")
    settings = document.get("requested_settings")
    if (
        not isinstance(results, list)
        or document.get("completed_cases") != len(results)
        or not isinstance(settings, dict)
        or not isinstance(settings.get("matrix"), list)
        or len(settings["matrix"]) != len(results)
        or settings.get("fixture_mode") != "matched-production"
        or settings.get("flush_writer") != flush_writer
        or settings.get("prefill_store") != prefill_store
        or not isinstance(settings.get("output_dtypes"), list)
        or output_dtype not in settings["output_dtypes"]
    ):
        raise GateError(f"{path}: factory result matrix is incomplete")
    required_matrix_fields = (
        "batch",
        "context",
        "requested_num_kv_splits",
        "effective_num_kv_splits",
        "kernel_variant",
        "variant_name",
        "dpas_layout",
        "output_dtype",
    )
    factory_axis_fields = (
        "cache_layout",
        "kernel_strategy",
        "split_policy",
        "fusion_strategy",
        "scheduling_variant",
    )
    axis_presence = [
        all(field in matrix for field in factory_axis_fields)
        for matrix in settings["matrix"]
        if isinstance(matrix, dict)
    ]
    if any(
        not isinstance(result, dict)
        or not isinstance(matrix, dict)
        or any(field not in matrix for field in required_matrix_fields)
        or matrix != {field: result.get(field) for field in matrix}
        for matrix, result in zip(settings["matrix"], results, strict=True)
    ) or axis_presence and not (all(axis_presence) or not any(axis_presence)):
        raise GateError(f"{path}: requested factory matrix differs from its results")
    if axis_presence and all(axis_presence) and any(
        not all(
            isinstance(result.get(field), str) and result[field]
            for field in factory_axis_fields
        )
        for result in results
    ):
        raise GateError(f"{path}: requested factory matrix differs from its results")

    variant_id = NATIVE_KERNEL_VARIANTS[native_kernel_variant]
    required_keys = {
        (
            batch,
            context,
            split_policy.effective_splits(
                native_split_policy,
                batch=batch,
                context_tokens=context,
                fixed_splits=native_splits or None,
            ),
        )
        for batch in (1, 4)
        for context in FACTORY_QUALIFICATION_CONTEXTS
    }
    selected: dict[tuple[int, int, int], dict[str, Any]] = {}
    for result in results:
        if (
            result.get("kernel_variant") != variant_id
            or result.get("variant_name") != native_kernel_variant
            or result.get("output_dtype") != output_dtype
            or result.get("dpas_layout") is not True
        ):
            continue
        key = (
            result.get("batch"),
            result.get("context"),
            result.get("requested_num_kv_splits"),
        )
        if key not in required_keys:
            continue
        if key in selected:
            raise GateError(f"{path}: duplicate selected factory result {key}")
        correctness = result.get("correctness")
        explicit = result.get("explicit_native_op_args")
        result_fixture = result.get("fixture")
        timing = result.get("timing")
        expected_case_id = (
            f"b{key[0]}-c{key[1]}-s{key[2]}-"
            f"v{variant_id}-{native_kernel_variant}-{output_dtype}"
        )
        if (
            result.get("case_id") != expected_case_id
            or result.get("effective_num_kv_splits") != key[2]
            or result.get("status") != "correctness_passed_and_timed"
            or result.get("scope") != "xpu_primitive_device_stage"
            or result.get("matched_primitive_ratio_eligible") is not True
            or all(axis_presence)
            and (
                result.get("cache_layout") != native_layout
                or result.get("kernel_strategy")
                != f"native_xe2_qlen1_{native_kernel_variant}"
            )
            or not isinstance(correctness, dict)
            or correctness.get("matched_auto_vs_quantized_natural_passed") is not True
            or any(
                not isinstance(correctness.get(field), dict)
                or correctness[field].get("finite") is not True
                for field in (
                    "structured_candidate_vs_natural",
                    "dense_candidate_vs_natural",
                    "matched_auto_vs_quantized_natural",
                )
            )
            or not isinstance(explicit, dict)
            or explicit.get("num_kv_splits") != key[2]
            or explicit.get("kernel_variant") != variant_id
            or explicit.get("dpas_layout") is not True
            or explicit.get("write_bf16_output") is not (output_dtype == "bf16")
            or not isinstance(result_fixture, dict)
            or result_fixture.get("fixture_mode") != "matched-production"
            or result_fixture.get("matched_primitive_fixture_eligible") is not True
            or result_fixture.get("logical_kv_payloads_matched_between_auto_and_kvarn")
            is not True
            or not isinstance(timing, dict)
            or timing.get("source") != "torch.xpu.Event device elapsed time"
        ):
            raise GateError(f"{path}: selected factory result {key} is invalid")
        selected[key] = result
    if set(selected) != required_keys:
        missing = sorted(required_keys - selected.keys())
        raise GateError(
            f"{path}: selected factory result coverage is incomplete: {missing}"
        )

    cases = [
        {
            "case_id": selected[key]["case_id"],
            "batch": key[0],
            "context": key[1],
            "num_kv_splits": key[2],
            "result_sha256": _json_sha256(selected[key]),
        }
        for key in sorted(selected)
    ]
    return {
        "status": "passed",
        "qualification_scope": "selected_native_variant_primitive_matrix",
        "factory_artifact": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
        },
        "source_revisions": expected_sources,
        "package_output": expected_package,
        "native_library": {
            "path": flash_path,
            "sha256": flash_sha256,
        },
        "native_attention_library": {
            "path": native_attention_path,
            "sha256": native_attention_sha256,
            "source_contract": native_source_contract,
            "runtime_binding_verified": True,
        },
        "selection": {
            "cache_layout": native_layout,
            "kernel_variant": native_kernel_variant,
            "kernel_variant_id": variant_id,
            "split_policy": native_split_policy,
            "split_policy_contract": policy_contract,
            "nominal_splits_by_batch": split_policy.nominal_splits_by_batch(
                native_split_policy, native_splits or None
            ),
            "effective_splits_by_batch": split_policy.nominal_splits_by_batch(
                native_split_policy, native_splits or None
            ),
            "effective_splits_by_context_and_batch": {
                str(context): {
                    str(batch): split_policy.effective_splits(
                        native_split_policy,
                        batch=batch,
                        context_tokens=context,
                        fixed_splits=native_splits or None,
                    )
                    for batch in (1, 4)
                }
                for context in FACTORY_QUALIFICATION_CONTEXTS
            },
            "output_dtype": output_dtype,
            "flush_writer": flush_writer,
            "prefill_store": prefill_store,
        },
        "cases": cases,
    }


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
    native_layout: str,
    native_kernel_variant: str,
    native_split_policy: str,
    native_splits: Mapping[int, int],
    native_output_dtype: str,
    flush_index_materialization: str,
    flush_writer: str,
    prefill_store: str,
    native_frontend: str,
    forward_pool_ensure: str,
    request_stable_projection_rows: str,
    request_stable_rmsnorm: str,
    *,
    owner: Path,
) -> dict[str, Any]:
    path, phase = _json_artifact(value, name=f"{phase_name} phase", owner=owner)
    expected_spec = _correctness_phase_spec(
        phase_name,
        native_layout,
        native_frontend,
        flush_index_materialization,
        native_kernel_variant,
        native_split_policy,
        native_splits,
        request_stable_projection_rows,
        request_stable_rmsnorm,
        flush_writer,
        prefill_store,
        forward_pool_ensure,
    )
    expected_layout = expected_spec["native_layout"]
    expected_kernel = expected_spec["native_kernel_variant"]
    expected_kernel_id = expected_spec["native_kernel_variant_id"]
    expected_policy = expected_spec["native_split_policy"]
    expected_max_splits = expected_spec["max_decode_splits"]
    expected_nominal_splits = expected_spec["nominal_decode_splits"]
    expected_splits_environment = (
        None
        if expected_spec["native"]
        and split_policy.owns_runtime_selection(expected_policy)
        else str(expected_max_splits)
    )
    expected_marker = _factory_marker(
        expected_layout,
        expected_kernel,
        expected_max_splits,
        expected_policy,
    )
    expected_variant = {field: expected_spec[field] for field in VARIANT_FIELDS}
    expected_layout_evidence = (
        "captured-process-environment-plus-factory-marker-plus-native-dispatch"
        if expected_spec["native"]
        else "captured-process-environment-plus-factory-marker"
    )
    expected_direct_bf16_marker = (
        NATIVE_DIRECT_BF16_MARKER if expected_spec["native"] else "not_applicable"
    )
    effective_frontend = expected_spec["native_frontend"]
    effective_flush_writer = expected_spec["flush_writer"]
    effective_prefill_store = expected_spec["prefill_store"]
    effective_forward_pool_ensure = expected_spec["forward_pool_ensure"]
    expected_frontend_active = (
        expected_spec["native"]
        and effective_frontend in {"qkv_scatter", "qkv_scatter_inline"}
    )
    expected_frontend_inline_active = (
        expected_spec["native"] and effective_frontend == "qkv_scatter_inline"
    )
    expected_forward_pool_ensure_active = (
        expected_spec["native"]
        and effective_forward_pool_ensure == "fused_qkv_proof"
    )
    if (
        phase.get("status") != "passed"
        or phase.get("spec") != expected_spec
        or phase.get("native_dispatch_verified") is not expected_spec["native"]
        or phase.get("native_layout") != expected_layout
        or phase.get("native_layout_environment") != NATIVE_LAYOUT_ENV[expected_layout]
        or phase.get("native_cache_layout_environment") != expected_layout
        or phase.get("native_kernel_variant") != expected_kernel
        or phase.get("native_kernel_variant_id") != expected_kernel_id
        or phase.get("native_kernel_variant_environment") != expected_kernel
        or phase.get("native_output_dtype") != native_output_dtype
        or phase.get("native_direct_bf16_verified") is not expected_spec["native"]
        or phase.get("native_direct_bf16_log_marker") != expected_direct_bf16_marker
        or phase.get("native_max_splits") != expected_max_splits
        or phase.get("native_nominal_splits") != expected_nominal_splits
        or phase.get("native_max_splits_environment") != expected_splits_environment
        or phase.get("native_split_policy") != expected_variant["split_policy"]
        or phase.get("native_split_policy_contract")
        != expected_spec["native_split_policy_contract"]
        or phase.get("native_split_policy_environment") != expected_policy
        or phase.get("native_layout_log_marker") != expected_marker
        or phase.get("native_layout_evidence") != expected_layout_evidence
        or phase.get("flush_index_materialization") != flush_index_materialization
        or phase.get("flush_writer") != effective_flush_writer
        or phase.get("prefill_store") != effective_prefill_store
        or phase.get("native_frontend") != effective_frontend
        or phase.get("forward_pool_ensure") != effective_forward_pool_ensure
        or phase.get("request_stable_projection_rows")
        != expected_spec["request_stable_projection_rows"]
        or phase.get("request_stable_rmsnorm")
        != expected_spec["request_stable_rmsnorm"]
        or phase.get("native_frontend_active_verified") is not expected_frontend_active
        or phase.get("native_frontend_log_marker")
        != (
            NATIVE_FRONTEND_ACTIVE_MARKER
            if expected_frontend_active
            else "not_applicable"
        )
        or phase.get("native_frontend_inline_active_verified")
        is not expected_frontend_inline_active
        or phase.get("native_frontend_inline_log_marker")
        != (
            NATIVE_FRONTEND_INLINE_ACTIVE_MARKER
            if expected_frontend_inline_active
            else "not_applicable"
        )
        or phase.get("forward_pool_ensure_active_verified")
        is not expected_forward_pool_ensure_active
        or phase.get("forward_pool_ensure_log_marker")
        != (
            FORWARD_POOL_ENSURE_ACTIVE_MARKER
            if expected_forward_pool_ensure_active
            else "not_applicable"
        )
        or not isinstance(phase.get("workload"), dict)
    ):
        raise GateError(f"{owner}: invalid {phase_name} phase evidence")
    for name in ("profile", "identity", "engine_log", "engine_log_scan"):
        _artifact_reference(phase.get(name), name=f"{phase_name}.{name}", owner=path)
    _profile_path, profile = _json_artifact(
        phase["profile"], name=f"{phase_name}.profile", owner=path
    )
    captured_environment = profile.get("redacted_environment")
    if (
        profile.get("native_layout") != expected_layout
        or profile.get("native_layout_environment")
        != NATIVE_LAYOUT_ENV[expected_layout]
        or profile.get("native_cache_layout_environment") != expected_layout
        or profile.get("native_kernel_variant_environment") != expected_kernel
        or profile.get("native_max_splits_environment") != expected_splits_environment
        or profile.get("native_split_policy_environment") != expected_policy
        or profile.get("flush_index_materialization_environment")
        != flush_index_materialization
        or profile.get("flush_writer_environment") != effective_flush_writer
        or profile.get("prefill_store_environment") != effective_prefill_store
        or profile.get("native_frontend_environment") != effective_frontend
        or profile.get("forward_pool_ensure_environment")
        != effective_forward_pool_ensure
        or profile.get("request_stable_projection_rows_environment")
        not in (
            {"1", None}
            if expected_spec["request_stable_projection_rows"] == "1"
            else {"0"}
        )
        or profile.get("request_stable_rmsnorm_environment")
        not in (
            {"1", None}
            if expected_spec["request_stable_rmsnorm"] == "1"
            else {"0"}
        )
        or not isinstance(captured_environment, dict)
        or captured_environment.get("KVARN_NATIVE_XPU_DPAS_LAYOUT")
        != NATIVE_LAYOUT_ENV[expected_layout]
        or captured_environment.get("KVARN_NATIVE_XPU_CACHE_LAYOUT") != expected_layout
        or captured_environment.get("KVARN_NATIVE_XPU_KERNEL_VARIANT")
        != expected_kernel
        or captured_environment.get("KVARN_NATIVE_XPU_SPLITS")
        != expected_splits_environment
        or captured_environment.get("KVARN_NATIVE_XPU_SPLIT_POLICY") != expected_policy
        or captured_environment.get("KVARN_FLUSH_INDEX_MATERIALIZATION")
        != flush_index_materialization
        or captured_environment.get("KVARN_FLUSH_WRITER") != effective_flush_writer
        or captured_environment.get("KVARN_NATIVE_XPU_PREFILL_STORE")
        != effective_prefill_store
        or captured_environment.get("KVARN_NATIVE_XPU_FRONTEND") != effective_frontend
        or captured_environment.get("KVARN_FORWARD_POOL_ENSURE")
        != effective_forward_pool_ensure
        or captured_environment.get("KVARN_ONEDNN_DETERMINISTIC") != "1"
        or captured_environment.get("KVARN_REQUEST_STABLE_PROJECTION_ROWS")
        != profile.get("request_stable_projection_rows_environment")
        or captured_environment.get("KVARN_REQUEST_STABLE_RMSNORM")
        != profile.get("request_stable_rmsnorm_environment")
        or captured_environment.get("VLLM_USE_V2_MODEL_RUNNER") != "0"
        or profile.get("variant_provenance") != expected_variant
    ):
        raise GateError(f"{owner}: {phase_name} layout profile differs")
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
    native_decoder_lines = [
        line for line in log_text.splitlines() if NATIVE_DISPATCH in line
    ]
    final_decoder_line = native_decoder_lines[-1] if native_decoder_lines else ""
    direct_bf16_verified = NATIVE_DIRECT_BF16_MARKER in final_decoder_line
    if expected_spec["native"] and not direct_bf16_verified:
        raise GateError(f"{owner}: {phase_name} lacks direct BF16 runtime proof")
    frontend_active = NATIVE_FRONTEND_ACTIVE_MARKER in log_text
    if frontend_active != expected_frontend_active:
        raise GateError(f"{owner}: {phase_name} frontend runtime proof differs")
    frontend_inline_active = NATIVE_FRONTEND_INLINE_ACTIVE_MARKER in log_text
    if frontend_inline_active != expected_frontend_inline_active:
        raise GateError(f"{owner}: {phase_name} inline frontend runtime proof differs")
    forward_pool_ensure_active = FORWARD_POOL_ENSURE_ACTIVE_MARKER in log_text
    if forward_pool_ensure_active != expected_forward_pool_ensure_active:
        raise GateError(
            f"{owner}: {phase_name} forward-pool runtime proof differs"
        )
    if expected_marker not in log_text:
        raise GateError(f"{owner}: {phase_name} lacks the exact factory marker")
    _scan_path, log_scan = _json_artifact(
        phase["engine_log_scan"], name=f"{phase_name}.engine_log_scan", owner=path
    )
    if (
        log_scan.get("status") != "passed"
        or log_scan.get("fatal_findings") != []
        or log_scan.get("native_direct_bf16_verified") is not expected_spec["native"]
        or log_scan.get("native_direct_bf16_log_marker") != expected_direct_bf16_marker
        or log_scan.get("native_frontend_expected") != effective_frontend
        or log_scan.get("native_frontend_active_verified")
        is not expected_frontend_active
        or log_scan.get("native_frontend_log_marker")
        != (
            NATIVE_FRONTEND_ACTIVE_MARKER
            if expected_frontend_active
            else "not_applicable"
        )
        or log_scan.get("native_frontend_inline_active_verified")
        is not expected_frontend_inline_active
        or log_scan.get("native_frontend_inline_log_marker")
        != (
            NATIVE_FRONTEND_INLINE_ACTIVE_MARKER
            if expected_frontend_inline_active
            else "not_applicable"
        )
        or log_scan.get("forward_pool_ensure_expected")
        != effective_forward_pool_ensure
        or log_scan.get("forward_pool_ensure_active_verified")
        is not expected_forward_pool_ensure_active
        or log_scan.get("forward_pool_ensure_log_marker")
        != (
            FORWARD_POOL_ENSURE_ACTIVE_MARKER
            if expected_forward_pool_ensure_active
            else "not_applicable"
        )
    ):
        raise GateError(f"{owner}: {phase_name} log scan is not clean")
    return phase["workload"]


def validate_correctness_gate_evidence(
    name: str,
    evidence: Any,
    *,
    path: Path,
    candidate_id: str,
    process_package: str,
    native_layout: str,
    native_kernel_variant: str = "baseline",
    native_split_policy: str = "fixed",
    native_splits: Mapping[int, int] | None = None,
    native_output_dtype: str = "bf16",
    flush_index_materialization: str = "per_layer",
    flush_writer: str = "reference",
    prefill_store: str = "reference",
    native_frontend: str = "reference",
    forward_pool_ensure: str = "always",
    request_stable_projection_rows: str = "1",
    request_stable_rmsnorm: str = "1",
) -> None:
    """Validate the meaning of one hashed correctness-gate artifact."""
    selected_splits = DEFAULT_NATIVE_SPLITS if native_splits is None else native_splits
    if flush_writer not in FLUSH_WRITER_VARIANTS:
        raise GateError(f"{path}: flush writer is unsupported")
    if flush_writer != "reference" and native_layout != "xe2_dpas":
        raise GateError(f"{path}: native Kvarn writer requires xe2_dpas layout")
    if prefill_store not in PREFILL_STORE_VARIANTS:
        raise GateError(f"{path}: prefill store is unsupported")
    if forward_pool_ensure not in FORWARD_POOL_ENSURE_VARIANTS:
        raise GateError(f"{path}: forward pool ensure is unsupported")
    if forward_pool_ensure == "fused_qkv_proof" and native_frontend not in {
        "qkv_scatter",
        "qkv_scatter_inline",
    }:
        raise GateError(f"{path}: fused pool proof requires a fused QKV frontend")
    if request_stable_projection_rows not in {"0", "1"}:
        raise GateError(f"{path}: projection-row selector is unsupported")
    if request_stable_rmsnorm not in {"0", "1"}:
        raise GateError(f"{path}: RMSNorm selector is unsupported")
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
            or evidence.get("qualification_scope") != "combined_library_variant_matrix"
            or evidence.get("variant_selection") != "explicit_per_op_arguments"
            or evidence.get("factory_variant_matrix") != COMBINED_LIBRARY_VARIANT_MATRIX
            or any(
                field in evidence
                for field in (
                    "native_layout",
                    "native_kernel_variant",
                    "native_kernel_variant_id",
                    "native_nominal_splits_by_batch",
                    "native_split_policy",
                    "native_scratch_max_splits",
                    *PRIMITIVE_SERVICE_ONLY_FIELDS,
                    *VARIANT_FIELDS,
                )
            )
        ):
            raise GateError(
                f"{path}: combined-library primitive gate lacks valid matrix evidence"
            )
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
            native_layout,
            native_kernel_variant,
            native_split_policy,
            selected_splits,
            native_output_dtype,
            flush_index_materialization,
            flush_writer,
            prefill_store,
            native_frontend,
            forward_pool_ensure,
            request_stable_projection_rows,
            request_stable_rmsnorm,
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
    engine_log_scan_path: str
    engine_log_scan_sha256: str

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
    native_layout: str,
    native_layout_environment: str,
    variant_provenance: dict[str, str],
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
        "native_layout": native_layout,
        "native_layout_environment": native_layout_environment,
        "variant_provenance": variant_provenance,
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
    boolean_arm_fields = {
        "kvarn_native_direct_bf16_verified",
        "kvarn_native_frontend_active_verified",
        "kvarn_native_frontend_inline_active_verified",
        "kvarn_forward_pool_ensure_active_verified",
    }
    for name in ARM_PROVENANCE_FIELDS:
        if name in boolean_arm_fields:
            value = document.get(name)
            if not isinstance(value, bool):
                raise GateError(f"{path}: {name} must be boolean")
            provenance[name] = value
        else:
            provenance[name] = _required_text(document, name, path)
    native = provenance["kvarn_native_xpu"] == "1"
    if provenance["kvarn_native_xpu"] not in {"0", "1"}:
        raise GateError(f"{path}: kvarn_native_xpu must be 0 or 1")
    frontend = provenance["kvarn_native_frontend"]
    forward_pool_ensure = provenance["kvarn_forward_pool_ensure"]
    if frontend not in NATIVE_FRONTEND_VARIANTS:
        raise GateError(f"{path}: native frontend is unsupported")
    if forward_pool_ensure not in FORWARD_POOL_ENSURE_VARIANTS:
        raise GateError(f"{path}: forward pool ensure is unsupported")
    if forward_pool_ensure == "fused_qkv_proof" and frontend not in {
        "qkv_scatter",
        "qkv_scatter_inline",
    }:
        raise GateError(f"{path}: fused pool proof requires a fused QKV frontend")
    frontend_active = native and frontend in {"qkv_scatter", "qkv_scatter_inline"}
    frontend_inline_active = native and frontend == "qkv_scatter_inline"
    forward_pool_ensure_active = native and forward_pool_ensure == "fused_qkv_proof"
    expected_execution_provenance = {
        "kvarn_native_frontend_active_verified": frontend_active,
        "kvarn_native_frontend_log_marker": (
            NATIVE_FRONTEND_ACTIVE_MARKER if frontend_active else "not_applicable"
        ),
        "kvarn_native_frontend_inline_active_verified": frontend_inline_active,
        "kvarn_native_frontend_inline_log_marker": (
            NATIVE_FRONTEND_INLINE_ACTIVE_MARKER
            if frontend_inline_active
            else "not_applicable"
        ),
        "kvarn_forward_pool_ensure_active_verified": forward_pool_ensure_active,
        "kvarn_forward_pool_ensure_log_marker": (
            FORWARD_POOL_ENSURE_ACTIVE_MARKER
            if forward_pool_ensure_active
            else "not_applicable"
        ),
    }
    if any(
        provenance[name] != expected
        for name, expected in expected_execution_provenance.items()
    ):
        raise GateError(f"{path}: runtime execution provenance is inconsistent")

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
        native_layout=provenance["kvarn_native_layout"],
        native_layout_environment=provenance["kvarn_native_layout_environment"],
        variant_provenance={
            field: provenance[f"kvarn_{field}"] for field in VARIANT_FIELDS
        },
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
    engine_log_scan_path = Path(
        _required_text(document, "kvarn_engine_log_scan_path", path)
    ).expanduser().resolve()
    engine_log_scan_sha256 = _required_sha256(
        document, "kvarn_engine_log_scan_sha256", path
    )
    if engine_log_scan_path.parent != path.resolve().parent:
        raise GateError(f"{path}: engine-log scan is not in the benchmark directory")
    try:
        engine_log_scan_bytes = engine_log_scan_path.read_bytes()
        engine_log_scan = json.loads(engine_log_scan_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"{path}: cannot read engine-log scan: {exc}") from exc
    if hashlib.sha256(engine_log_scan_bytes).hexdigest() != engine_log_scan_sha256:
        raise GateError(f"{path}: engine-log scan SHA-256 differs")
    expected_scan_evidence = {
        "native_frontend_expected": provenance["kvarn_native_frontend"],
        "native_frontend_active_verified": provenance[
            "kvarn_native_frontend_active_verified"
        ],
        "native_frontend_log_marker": provenance[
            "kvarn_native_frontend_log_marker"
        ],
        "native_frontend_inline_active_verified": provenance[
            "kvarn_native_frontend_inline_active_verified"
        ],
        "native_frontend_inline_log_marker": provenance[
            "kvarn_native_frontend_inline_log_marker"
        ],
        "forward_pool_ensure_expected": provenance["kvarn_forward_pool_ensure"],
        "forward_pool_ensure_active_verified": provenance[
            "kvarn_forward_pool_ensure_active_verified"
        ],
        "forward_pool_ensure_log_marker": provenance[
            "kvarn_forward_pool_ensure_log_marker"
        ],
        "engine_log_sha256": engine_log_sha256,
    }
    if (
        not isinstance(engine_log_scan, dict)
        or engine_log_scan.get("status") != "passed"
        or engine_log_scan.get("fatal_findings") != []
        or any(
            engine_log_scan.get(name) != expected
            for name, expected in expected_scan_evidence.items()
        )
    ):
        raise GateError(f"{path}: engine-log scan execution evidence differs")
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
        engine_log_scan_path=str(engine_log_scan_path),
        engine_log_scan_sha256=engine_log_scan_sha256,
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
    scan_paths = [run.engine_log_scan_path for run in runs]
    if len(set(scan_paths)) != len(scan_paths):
        raise GateError(f"{name} runs must have distinct engine-log scans")
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
    if (
        document.get("native_direct_bf16_verified") is not True
        or document.get("native_direct_bf16_log_marker") != NATIVE_DIRECT_BF16_MARKER
    ):
        raise GateError(f"{path}: direct BF16 runtime proof is missing")
    native_layout = document.get("native_layout")
    if native_layout not in NATIVE_LAYOUTS:
        raise GateError(f"{path}: native_layout must be natural or xe2_dpas")
    native_kernel_variant = document.get("native_kernel_variant")
    if native_kernel_variant not in NATIVE_KERNEL_VARIANTS:
        raise GateError(f"{path}: native_kernel_variant is unsupported")
    if (
        document.get("native_kernel_variant_id")
        != NATIVE_KERNEL_VARIANTS[native_kernel_variant]
    ):
        raise GateError(f"{path}: native_kernel_variant_id is inconsistent")
    if native_kernel_variant != "baseline" and native_layout != "xe2_dpas":
        raise GateError(
            f"{path}: non-baseline native kernels require the xe2_dpas layout"
        )
    native_split_policy = document.get("native_split_policy")
    if native_split_policy not in NATIVE_SPLIT_POLICIES:
        raise GateError(f"{path}: native_split_policy is unsupported")
    native_output_dtype = document.get("native_output_dtype")
    if native_output_dtype != "bf16":
        raise GateError(
            f"{path}: finalist service qualification requires bf16 native output"
        )
    flush_index_materialization = document.get("flush_index_materialization")
    if flush_index_materialization not in FLUSH_INDEX_MATERIALIZATION_VARIANTS:
        raise GateError(f"{path}: flush-index materialization is unsupported")
    flush_writer = document.get("flush_writer")
    if flush_writer not in FLUSH_WRITER_VARIANTS:
        raise GateError(f"{path}: flush writer is unsupported")
    if flush_writer != "reference" and native_layout != "xe2_dpas":
        raise GateError(f"{path}: native Kvarn writer requires xe2_dpas layout")
    prefill_store = document.get("prefill_store")
    if prefill_store not in PREFILL_STORE_VARIANTS:
        raise GateError(f"{path}: prefill store is unsupported")
    native_frontend = document.get("native_frontend")
    if native_frontend not in NATIVE_FRONTEND_VARIANTS:
        raise GateError(f"{path}: native frontend is unsupported")
    forward_pool_ensure = document.get("forward_pool_ensure")
    if forward_pool_ensure not in FORWARD_POOL_ENSURE_VARIANTS:
        raise GateError(f"{path}: forward pool ensure is unsupported")
    if forward_pool_ensure == "fused_qkv_proof" and native_frontend not in {
        "qkv_scatter",
        "qkv_scatter_inline",
    }:
        raise GateError(f"{path}: fused pool proof requires a fused QKV frontend")
    service_controls = document.get("service_controls")
    correctness_onednn = (
        service_controls.get("kvarn_onednn_deterministic")
        if isinstance(service_controls, dict)
        else None
    )
    correctness_projection_rows = (
        service_controls.get("kvarn_request_stable_projection_rows")
        if isinstance(service_controls, dict)
        else None
    )
    correctness_rmsnorm = (
        service_controls.get("kvarn_request_stable_rmsnorm")
        if isinstance(service_controls, dict)
        else None
    )
    correctness_forward_pool_ensure = (
        service_controls.get("kvarn_forward_pool_ensure")
        if isinstance(service_controls, dict)
        else None
    )
    if correctness_onednn not in {"0", "1"}:
        raise GateError(f"{path}: correctness oneDNN selector is unsupported")
    if correctness_projection_rows not in {"0", "1"}:
        raise GateError(f"{path}: correctness projection-row selector is unsupported")
    if correctness_rmsnorm not in {"0", "1"}:
        raise GateError(f"{path}: correctness RMSNorm selector is unsupported")
    if correctness_forward_pool_ensure != forward_pool_ensure:
        raise GateError(f"{path}: correctness forward-pool selector is inconsistent")
    expected_service_controls = {
        "kvarn_flush_index_materialization": flush_index_materialization,
        "kvarn_flush_writer": flush_writer,
        "kvarn_prefill_store": prefill_store,
        "kvarn_native_frontend": native_frontend,
        "kvarn_forward_pool_ensure": forward_pool_ensure,
        "kvarn_onednn_deterministic": correctness_onednn,
        "kvarn_request_stable_projection_rows": correctness_projection_rows,
        "kvarn_request_stable_rmsnorm": correctness_rmsnorm,
        "vllm_use_v2_model_runner": "0",
    }
    if service_controls != expected_service_controls:
        raise GateError(f"{path}: correctness service controls are inconsistent")
    expected_qualification = (
        "qualified-default"
        if correctness_projection_rows == "1" and correctness_rmsnorm == "1"
        else "replay-qualified"
    )
    if document.get("request_stability_qualification") != expected_qualification:
        raise GateError(f"{path}: request-stability qualification is inconsistent")
    raw_native_splits = document.get("native_nominal_splits_by_batch")
    if native_split_policy == "b70_q6_v2":
        if raw_native_splits is not None:
            raise GateError(
                f"{path}: context-dependent split policy must not declare a "
                "batch-only nominal map"
            )
        native_splits: dict[int, int] = {}
    else:
        if not isinstance(raw_native_splits, dict) or set(raw_native_splits) != {
            "1",
            "4",
        }:
            raise GateError(f"{path}: native_nominal_splits_by_batch is incomplete")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in {1, 2, 4, 8, 16, 17, 24, 32}
            for value in raw_native_splits.values()
        ):
            raise GateError(f"{path}: native nominal split count is unsupported")
        native_splits = {
            int(batch): splits for batch, splits in raw_native_splits.items()
        }
    try:
        split_policy.validate_kernel_compatibility(
            native_split_policy,
            native_kernel_variant,
            q6_variants=B70_Q6_KERNEL_VARIANTS,
        )
        expected_policy_contract = split_policy.split_policy_contract(
            native_split_policy, native_splits or None
        )
    except ValueError as exc:
        raise GateError(f"{path}: invalid split policy selection: {exc}") from exc
    observed_policy_contract = document.get("native_split_policy_contract")
    if observed_policy_contract != expected_policy_contract:
        raise GateError(f"{path}: native split-policy contract differs")
    expected_scratch_max = int(expected_policy_contract["scratch_max_splits"])
    if document.get("native_scratch_max_splits") != expected_scratch_max:
        raise GateError(f"{path}: native scratch split ceiling is inconsistent")
    expected_service_plan = [
        _correctness_phase_spec(
            phase_name,
            native_layout,
            native_frontend,
            flush_index_materialization,
            native_kernel_variant,
            native_split_policy,
            native_splits,
            correctness_projection_rows,
            correctness_rmsnorm,
            flush_writer,
            prefill_store,
            forward_pool_ensure,
        )
        for phase_name in CORRECTNESS_PHASE_SPECS
    ]
    if document.get("service_start_plan") != expected_service_plan:
        raise GateError(f"{path}: service_start_plan differs from native_layout")
    expected_variant = _candidate_variant_provenance(
        native_layout,
        native_frontend,
        flush_index_materialization,
        native_kernel_variant,
        native_split_policy,
        native_splits,
        flush_writer,
        prefill_store,
        forward_pool_ensure,
    )
    if any(document.get(field) != value for field, value in expected_variant.items()):
        raise GateError(f"{path}: correctness variant provenance is inconsistent")
    candidate_id = document.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise GateError(f"{path}: candidate_id must be a non-empty string")
    process_package = _validate_correctness_manifest_identity(document, path)
    source_revisions = document["source_identity"]["revisions"]
    factory_reference = document.get("factory_qualification")
    if not isinstance(factory_reference, dict):
        raise GateError(f"{path}: factory_qualification must be an object")
    factory_artifact = factory_reference.get("factory_artifact")
    factory_path, _factory_sha256 = _artifact_reference(
        factory_artifact, name="factory_qualification", owner=path
    )
    validated_factory = validate_factory_qualification(
        factory_path,
        native_layout=native_layout,
        native_kernel_variant=native_kernel_variant,
        native_split_policy=native_split_policy,
        native_splits=native_splits,
        output_dtype=native_output_dtype,
        flush_writer=flush_writer,
        prefill_store=prefill_store,
        expected_revisions={
            "vllm-xpu-nix": source_revisions["vllm-xpu-release"],
            "vllm": source_revisions["vllm-xpu-unstable-src"],
            "vllm-xpu-kernels": source_revisions["vllm-xpu-kernels-unstable-src"],
        },
        expected_package=process_package,
    )
    if factory_reference != validated_factory:
        raise GateError(f"{path}: factory qualification binding is inconsistent")
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
            native_layout=native_layout,
            native_kernel_variant=native_kernel_variant,
            native_split_policy=native_split_policy,
            native_splits=native_splits,
            native_output_dtype=native_output_dtype,
            flush_index_materialization=flush_index_materialization,
            flush_writer=flush_writer,
            prefill_store=prefill_store,
            native_frontend=native_frontend,
            forward_pool_ensure=forward_pool_ensure,
            request_stable_projection_rows=correctness_projection_rows,
            request_stable_rmsnorm=correctness_rmsnorm,
        )
        if name.startswith("native_decode_"):
            library_path, library_sha256 = _artifact_reference(
                evidence_document.get("native_library"),
                name=f"{name}.native_library",
                owner=evidence,
            )
            if {
                "path": str(library_path),
                "sha256": library_sha256,
            } != validated_factory["native_library"]:
                raise GateError(
                    f"{path}: {name} combined library differs from the selected "
                    "factory artifact"
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
    paths: list[Path],
    arm: str,
    *,
    expect_native: bool,
    expected_layout: str,
    expected_kernel_variant: str | None = None,
    expected_max_splits: int | None = None,
    expected_split_policy: str | None = None,
    expected_frontend: str,
    expected_forward_pool_ensure: str,
) -> list[dict[str, Any]]:
    if expected_layout not in NATIVE_LAYOUTS or (
        not expect_native and expected_layout != "natural"
    ):
        raise GateError(f"{arm} has an invalid native-layout log expectation")
    factory_expectations = (
        expected_kernel_variant,
        expected_max_splits,
        expected_split_policy,
    )
    if expect_native != all(value is not None for value in factory_expectations):
        raise GateError(f"{arm} has incomplete factory-marker expectations")
    expected_marker = (
        _factory_marker(
            expected_layout,
            expected_kernel_variant,
            expected_max_splits,
            expected_split_policy,
        )
        if expect_native
        else "unavailable"
    )
    if expected_frontend not in NATIVE_FRONTEND_VARIANTS:
        raise GateError(f"{arm} has an invalid native-frontend log expectation")
    expected_frontend_active = expect_native and expected_frontend in {
        "qkv_scatter",
        "qkv_scatter_inline",
    }
    expected_frontend_inline_active = (
        expect_native and expected_frontend == "qkv_scatter_inline"
    )
    if expected_forward_pool_ensure not in FORWARD_POOL_ENSURE_VARIANTS:
        raise GateError(f"{arm} has an invalid forward-pool log expectation")
    if expected_forward_pool_ensure == "fused_qkv_proof" and expected_frontend not in {
        "qkv_scatter",
        "qkv_scatter_inline",
    }:
        raise GateError(f"{arm} fused pool proof requires a fused QKV frontend")
    expected_forward_pool_ensure_active = (
        expect_native and expected_forward_pool_ensure == "fused_qkv_proof"
    )
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
        native_decoder_lines = [
            line for line in text.splitlines() if NATIVE_DISPATCH in line
        ]
        direct_bf16_verified = any(
            NATIVE_DIRECT_BF16_MARKER in line for line in native_decoder_lines
        )
        direct_bf16_disabled = any(
            NATIVE_DIRECT_BF16_DISABLED_MARKER in line for line in native_decoder_lines
        )
        if expect_native and (direct_bf16_disabled or not direct_bf16_verified):
            raise GateError(f"{path}: native candidate lacks direct BF16 runtime proof")
        if expect_native and expected_marker not in text:
            raise GateError(f"{path}: native candidate lacks the exact factory marker")
        frontend_active = NATIVE_FRONTEND_ACTIVE_MARKER in text
        if frontend_active != expected_frontend_active:
            raise GateError(f"{path}: {arm} frontend runtime proof differs")
        frontend_inline_active = NATIVE_FRONTEND_INLINE_ACTIVE_MARKER in text
        if frontend_inline_active != expected_frontend_inline_active:
            raise GateError(f"{path}: {arm} inline frontend runtime proof differs")
        forward_pool_ensure_active = FORWARD_POOL_ENSURE_ACTIVE_MARKER in text
        if forward_pool_ensure_active != expected_forward_pool_ensure_active:
            raise GateError(f"{path}: {arm} forward-pool runtime proof differs")
        evidence.append(
            {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "native_layout_expected": expected_layout,
                "native_layout_log_marker": expected_marker,
                "native_direct_bf16_verified": (
                    direct_bf16_verified if expect_native else False
                ),
                "native_direct_bf16_log_marker": (
                    NATIVE_DIRECT_BF16_MARKER if expect_native else "not_applicable"
                ),
                "native_frontend_expected": expected_frontend,
                "native_frontend_active_verified": frontend_active,
                "native_frontend_log_marker": (
                    NATIVE_FRONTEND_ACTIVE_MARKER
                    if expected_frontend_active
                    else "not_applicable"
                ),
                "native_frontend_inline_active_verified": frontend_inline_active,
                "native_frontend_inline_log_marker": (
                    NATIVE_FRONTEND_INLINE_ACTIVE_MARKER
                    if expected_frontend_inline_active
                    else "not_applicable"
                ),
                "forward_pool_ensure_expected": expected_forward_pool_ensure,
                "forward_pool_ensure_active_verified": forward_pool_ensure_active,
                "forward_pool_ensure_log_marker": (
                    FORWARD_POOL_ENSURE_ACTIVE_MARKER
                    if expected_forward_pool_ensure_active
                    else "not_applicable"
                ),
                **xpu,
            }
        )
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
    correctness_controls = correctness["service_controls"]
    benchmark_controls = {
        "kvarn_flush_index_materialization": first_cand.provenance[
            "kvarn_flush_index_materialization"
        ],
        "kvarn_flush_writer": first_cand.provenance["kvarn_flush_writer"],
        "kvarn_prefill_store": first_cand.provenance["kvarn_prefill_store"],
        "kvarn_native_frontend": first_cand.provenance["kvarn_native_frontend"],
        "kvarn_forward_pool_ensure": first_cand.provenance[
            "kvarn_forward_pool_ensure"
        ],
        "kvarn_onednn_deterministic": first_cand.provenance[
            "kvarn_onednn_deterministic"
        ],
        "kvarn_request_stable_projection_rows": first_cand.provenance[
            "kvarn_request_stable_projection_rows"
        ],
        "kvarn_request_stable_rmsnorm": first_cand.provenance[
            "kvarn_request_stable_rmsnorm"
        ],
        "vllm_use_v2_model_runner": first_cand.provenance[
            "kvarn_vllm_use_v2_model_runner"
        ],
    }
    if benchmark_controls != correctness_controls:
        raise GateError(
            "correctness and performance selected different service controls"
        )
    expected_request_stability_qualification = (
        "qualified-default"
        if correctness_controls["kvarn_request_stable_projection_rows"] == "1"
        and correctness_controls["kvarn_request_stable_rmsnorm"] == "1"
        else "replay-qualified"
    )
    if (
        first_cand.provenance["kvarn_request_stability_qualification"]
        != expected_request_stability_qualification
    ):
        raise GateError(
            "performance request-stability policy is not replay-qualified"
        )
    correctness_layout = correctness["native_layout"]
    correctness_kernel_variant = correctness["native_kernel_variant"]
    correctness_kernel_variant_id = correctness["native_kernel_variant_id"]
    correctness_split_policy = correctness["native_split_policy"]
    correctness_nominal_splits = split_policy.effective_splits(
        correctness_split_policy,
        batch=concurrency,
        context_tokens=first_ref.input_lens[0],
        fixed_splits=(
            {
                int(batch): splits
                for batch, splits in correctness[
                    "native_nominal_splits_by_batch"
                ].items()
            }
            if correctness["native_nominal_splits_by_batch"] is not None
            else None
        ),
    )
    correctness_max_splits = (
        correctness["native_scratch_max_splits"]
        if split_policy.owns_runtime_selection(correctness_split_policy)
        else correctness_nominal_splits
    )

    ref_dtype = first_ref.provenance["kvarn_kv_cache_dtype"]
    cand_dtype = first_cand.provenance["kvarn_kv_cache_dtype"]
    ref_native = first_ref.provenance["kvarn_native_xpu"]
    cand_native = first_cand.provenance["kvarn_native_xpu"]
    if (ref_native, cand_native) != ("0", "1"):
        raise GateError("reference must disable and candidate must enable native XPU")
    if first_ref.provenance["kvarn_native_frontend"] != "reference":
        raise GateError("reference must use the unfused reference frontend")
    if first_ref.provenance["kvarn_forward_pool_ensure"] != "always":
        raise GateError("reference must use the conservative forward pool guard")
    if first_ref.provenance["kvarn_flush_writer"] != "reference":
        raise GateError("reference must use the reference flush writer")
    if first_ref.provenance["kvarn_prefill_store"] != "reference":
        raise GateError("reference must use the reference prefill store")
    if first_cand.provenance["kvarn_native_frontend"] != correctness["native_frontend"]:
        raise GateError("candidate frontend must match correctness")
    if (
        first_cand.provenance["kvarn_forward_pool_ensure"]
        != correctness["forward_pool_ensure"]
    ):
        raise GateError("candidate forward pool guard must match correctness")
    if first_cand.provenance["kvarn_flush_writer"] != correctness["flush_writer"]:
        raise GateError("candidate flush writer must match correctness")
    if first_cand.provenance["kvarn_prefill_store"] != correctness["prefill_store"]:
        raise GateError("candidate prefill store must match correctness")
    ref_layout = first_ref.provenance["kvarn_native_layout"]
    cand_layout = first_cand.provenance["kvarn_native_layout"]
    if ref_layout != "natural" or cand_layout != correctness_layout:
        raise GateError(
            "candidate native layout must match correctness and reference must be "
            "natural"
        )
    reference_variant = {
        field: first_ref.provenance[f"kvarn_{field}"] for field in VARIANT_FIELDS
    }
    candidate_variant = {
        field: first_cand.provenance[f"kvarn_{field}"] for field in VARIANT_FIELDS
    }
    correctness_variant = {field: correctness[field] for field in VARIANT_FIELDS}
    if reference_variant != _performance_reference_variant_provenance():
        raise GateError("reference variant provenance is not the auto control")
    if candidate_variant != correctness_variant:
        raise GateError(
            "candidate variant provenance must match the correctness manifest"
        )
    for (
        run,
        layout,
        native,
        kernel,
        kernel_id,
        max_splits,
        nominal_splits,
        policy,
        output_dtype,
    ) in (
        (
            first_ref,
            ref_layout,
            False,
            "baseline",
            0,
            1,
            1,
            "fixed",
            "not_applicable",
        ),
        (
            first_cand,
            cand_layout,
            True,
            correctness_kernel_variant,
            correctness_kernel_variant_id,
            correctness_max_splits,
            correctness_nominal_splits,
            correctness_split_policy,
            correctness["native_output_dtype"],
        ),
    ):
        expected_evidence = (
            "captured-process-environment-plus-factory-marker-plus-native-dispatch"
            if native
            else "captured-process-environment"
        )
        expected_marker = (
            _factory_marker(layout, kernel, max_splits, policy)
            if native
            else "unavailable"
        )
        expected_direct_bf16_marker = (
            NATIVE_DIRECT_BF16_MARKER if native else "not_applicable"
        )
        if (
            run.provenance["kvarn_native_layout_environment"]
            != NATIVE_LAYOUT_ENV[layout]
            or run.provenance["kvarn_native_cache_layout_environment"] != layout
            or run.provenance["kvarn_native_kernel_variant"] != kernel
            or run.provenance["kvarn_native_kernel_variant_id"] != str(kernel_id)
            or run.provenance["kvarn_native_output_dtype"] != output_dtype
            or run.provenance["kvarn_native_direct_bf16_verified"] is not native
            or run.provenance["kvarn_native_direct_bf16_log_marker"]
            != expected_direct_bf16_marker
            or run.provenance["kvarn_native_max_splits"] != str(max_splits)
            or run.provenance["kvarn_native_nominal_splits"] != str(nominal_splits)
            or run.provenance["kvarn_native_split_policy"] != policy
            or run.provenance["kvarn_native_layout_log_marker"] != expected_marker
            or run.provenance["kvarn_native_layout_evidence"] != expected_evidence
        ):
            raise GateError(f"{run.path}: native-layout evidence is inconsistent")
    try:
        ref_splits = int(first_ref.provenance["kvarn_native_nominal_splits"])
        cand_splits = int(first_cand.provenance["kvarn_native_nominal_splits"])
    except ValueError as exc:
        raise GateError("kvarn_native_nominal_splits must be an integer") from exc
    if ref_splits != 1:
        raise GateError("non-native reference must declare the neutral split count 1")
    if cand_splits not in {1, 2, 4, 8, 16, 17, 24, 32}:
        raise GateError("candidate must declare a supported native split count")
    if cand_splits != correctness_nominal_splits:
        raise GateError("candidate nominal split count differs from correctness")
    if comparison_kind == "kernel":
        if (ref_dtype, cand_dtype) != (COMPACT_DTYPE, COMPACT_DTYPE):
            raise GateError("kernel comparison requires compact Kvarn in both arms")
    elif comparison_kind == "end-to-end":
        if (ref_dtype, cand_dtype) != ("auto", COMPACT_DTYPE):
            raise GateError("end-to-end comparison requires auto versus compact Kvarn")
    else:
        raise GateError(f"unsupported comparison kind {comparison_kind!r}")

    reference_log_evidence = _validate_logs(
        reference_logs,
        "reference",
        expect_native=False,
        expected_layout=ref_layout,
        expected_frontend="reference",
        expected_forward_pool_ensure="always",
    )
    candidate_log_evidence = _validate_logs(
        candidate_logs,
        "candidate",
        expect_native=True,
        expected_layout=cand_layout,
        expected_frontend=correctness["native_frontend"],
        expected_forward_pool_ensure=correctness["forward_pool_ensure"],
        expected_kernel_variant=correctness_kernel_variant,
        expected_max_splits=correctness_max_splits,
        expected_split_policy=correctness_split_policy,
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
            "engine_log_scan": [
                {
                    "path": run.engine_log_scan_path,
                    "sha256": run.engine_log_scan_sha256,
                }
                for run in reference
            ],
            "arm": {
                field: first_ref.provenance[field] for field in ARM_PROVENANCE_FIELDS
            },
            "median": ref,
        },
        "candidate": {
            "runs": [run.path for run in candidate],
            "engine_log_sha256": candidate_log_sha256,
            "engine_log_scan": [
                {
                    "path": run.engine_log_scan_path,
                    "sha256": run.engine_log_scan_sha256,
                }
                for run in candidate
            ],
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
