from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import kvarn_perf_gate as gate_module
from scripts.kvarn_perf_gate import GateError, _load_correctness, compare


def test_combined_library_matrix_registers_opt_in_variants_through_id15() -> None:
    assert [
        (item["kernel_variant"], item["kernel_variant_id"])
        for item in gate_module.COMBINED_LIBRARY_VARIANT_MATRIX
    ] == [
        ("q6_scalar", 2),
        ("q6_vector", 4),
        ("q6_cached_weights", 6),
        ("q6_exact_rows", 7),
        ("q6_cached_weights_exact_rows", 8),
        ("q6_page_pair", 9),
        ("q6_main_grf128", 10),
        ("q6_split_reducer_specialized", 11),
        ("q6_next_page_prefetch", 12),
        ("q6_next_page_prefetch_split_reducer", 13),
        ("q6_simd_unpack", 14),
        ("q6_block_output_store", 15),
    ]


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _correctness_result(fixture_id: str) -> dict[str, object]:
    prompt_tokens = gate_module.CORRECTNESS_FIXTURE_LENGTHS[fixture_id]
    token_ids = list(range(512))
    encoded = json.dumps(token_ids, separators=(",", ":")).encode()
    return {
        "id": fixture_id,
        "prompt_token_count": prompt_tokens,
        "prompt_token_ids_sha256": "1" * 64,
        "max_tokens": 512,
        "finish_reason": "length",
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(encoded).hexdigest(),
        "quality_findings": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 512,
            "total_tokens": prompt_tokens + 512,
        },
    }


def _correctness_comparison(fixture_id: str) -> dict[str, object]:
    digest = _correctness_result(fixture_id)["token_ids_sha256"]
    return {
        "status": "passed",
        "fixture_id": fixture_id,
        "same_fixture": True,
        "same_prompt_token_ids": True,
        "token_ids_identical": True,
        "expected_token_ids_sha256": digest,
        "actual_token_ids_sha256": digest,
    }


def _factory_result(
    path: Path,
    *,
    native_library: Path,
    revisions: dict[str, str],
    native_kernel_variant: str,
    native_splits: dict[int, int],
    native_split_policy: str = "b70_q6",
    output_dtype: str = "bf16",
    explicit_factory_axes: bool = False,
) -> Path:
    variant_id = gate_module.NATIVE_KERNEL_VARIANTS[native_kernel_variant]
    cases: list[dict[str, object]] = []
    for batch in (1, 4):
        for context in gate_module.FACTORY_QUALIFICATION_CONTEXTS:
            splits = gate_module.split_policy.effective_splits(
                native_split_policy,
                batch=batch,
                context_tokens=context,
                fixed_splits=native_splits or None,
            )
            case = {
                "case_id": (
                    f"b{batch}-c{context}-s{splits}-v{variant_id}-"
                    f"{native_kernel_variant}-{output_dtype}"
                ),
                "batch": batch,
                "context": context,
                "requested_num_kv_splits": splits,
                "effective_num_kv_splits": splits,
                "kernel_variant": variant_id,
                "variant_name": native_kernel_variant,
                "dpas_layout": True,
                "output_dtype": output_dtype,
                "status": "correctness_passed_and_timed",
                "scope": "xpu_primitive_device_stage",
                "matched_primitive_ratio_eligible": True,
                "correctness": {
                    name: {"finite": True}
                    for name in (
                        "structured_candidate_vs_natural",
                        "dense_candidate_vs_natural",
                        "matched_auto_vs_quantized_natural",
                    )
                },
                "explicit_native_op_args": {
                    "num_kv_splits": splits,
                    "kernel_variant": variant_id,
                    "dpas_layout": True,
                    "natural_oracle": False,
                    "unrotate_output": True,
                    "write_bf16_output": output_dtype == "bf16",
                },
                "fixture": {
                    "fixture_mode": "matched-production",
                    "matched_primitive_fixture_eligible": True,
                    "logical_kv_payloads_matched_between_auto_and_kvarn": True,
                },
                "timing": {"source": "torch.xpu.Event device elapsed time"},
            }
            if explicit_factory_axes:
                case.update(
                    {
                        "cache_layout": "xe2_dpas",
                        "kernel_strategy": (
                            f"native_xe2_qlen1_{native_kernel_variant}"
                        ),
                        "split_policy": "fixed",
                        "fusion_strategy": "standard_split_reduction",
                        "scheduling_variant": "tile64",
                    }
                )
            case["correctness"]["matched_auto_vs_quantized_natural_passed"] = True
            cases.append(case)
    matrix_fields = (
        "batch",
        "context",
        "requested_num_kv_splits",
        "effective_num_kv_splits",
        "kernel_variant",
        "variant_name",
        "dpas_layout",
        "output_dtype",
    )
    if explicit_factory_axes:
        matrix_fields += (
            "cache_layout",
            "kernel_strategy",
            "split_policy",
            "fusion_strategy",
            "scheduling_variant",
        )
    sources = {
        "vllm-xpu-nix": revisions["vllm-xpu-release"],
        "vllm": revisions["vllm-xpu-unstable-src"],
        "vllm-xpu-kernels": revisions["vllm-xpu-kernels-unstable-src"],
    }
    library_sha256 = hashlib.sha256(native_library.read_bytes()).hexdigest()
    factory_output = "/nix/store/factory-kernels"
    native_attention_library = path.parent / "libattn_kernels_xe_2.so"
    native_attention_library.write_bytes(b"native Xe2 attention kernels")
    native_attention_sha256 = hashlib.sha256(
        native_attention_library.read_bytes()
    ).hexdigest()
    native_attention_output = "/nix/store/factory-native-attention"
    native_attention_source_hash = "4" * 32
    native_attention_source_identity = {
        "scheme": gate_module.FILTERED_SOURCE_SCHEME,
        "filtered_source_store_hash": native_attention_source_hash,
    }
    native_attention_derivation = (
        "/nix/store/"
        + "4" * 32
        + "-native-attention-0.1+src."
        + native_attention_source_hash
        + ".drv"
    )
    native_attention_source_contract = {
        "nix_evaluation_identity": {
            "output_path": native_attention_output,
            "derivation": native_attention_derivation,
        },
        "artifact_identity": native_attention_source_identity,
        "compatibility_provenance": {
            "upstream_revision": sources["vllm-xpu-kernels"],
            "asserted_against_expected_repository_revision": True,
        },
    }
    document = {
        "schema_version": 3,
        "artifact_kind": "kvarn_b70_primitive_factory_run",
        "status": "completed_primitive_diagnostic",
        "identity_stable_through_sweep": True,
        "evidence_identity_sha256": "9" * 64,
        "ending_evidence_identity_sha256": "9" * 64,
        "source_revisions": {
            "verified": True,
            "expected": sources,
            "actual": sources,
        },
        "repositories": [
            {
                "name": name,
                "head": revision,
                "dirty": False,
                "status_porcelain": [],
            }
            for name, revision in sources.items()
        ],
        "runtime_environment": {
            "prefixed_environment_clean": True,
            "kvarn_or_vllm_prefixed_variables": {},
        },
        "hardware_preflight": {
            "passed": True,
            "selected_device": "xpu:0",
            "selected_device_name": gate_module.EXPECTED_XPU_DEVICE_NAME,
        },
        "fixture_matching": {
            "fixture_mode": "matched-production",
            "validation_status": "passed",
            "logical_kv_payloads_matched_between_auto_and_kvarn": True,
            "matched_primitive_fixture_eligible": True,
            "matched_primitive_ratio_eligible": True,
        },
        "kernel_kill_suite": {
            "status": "passed",
            "passed": True,
            "returncode": 0,
            "skipped_count": 0,
        },
        "libraries": {
            "flash": {
                "path": str(native_library.resolve()),
                "sha256": library_sha256,
            },
            "native_attention": {
                "path": str(native_attention_library.resolve()),
                "sha256": native_attention_sha256,
            },
        },
        "build_attestations": {
            "package": {
                "verified": True,
                "output_path": "/nix/store/package",
                "closure_paths": [
                    "/nix/store/package",
                    factory_output,
                    native_attention_output,
                ],
            },
            "flash": {
                "verified": True,
                "output_path": factory_output,
                "library_path": str(native_library.resolve()),
            },
            "native_attention": {
                "verified": True,
                "output_path": native_attention_output,
                "library_path": str(native_attention_library.resolve()),
                "derivation": native_attention_derivation,
                "source_contract": native_attention_source_contract,
            },
        },
        "source_ownership": {
            "verified": True,
            "artifacts": {
                "package": {"verified": True, "member_of_package_closure": True},
                "base": {"verified": True, "member_of_package_closure": True},
                "flash": {"verified": True, "member_of_package_closure": True},
                "native_attention": {
                    "verified": True,
                    "member_of_package_closure": True,
                    "repository": "vllm-xpu-kernels",
                    "compatible_upstream_revision": sources["vllm-xpu-kernels"],
                    "derivation": native_attention_derivation,
                    "derivation_source_marker": (
                        "+src." + native_attention_source_hash
                    ),
                    "artifact_identity": native_attention_source_identity,
                    "nix_evaluation_identity": native_attention_source_contract[
                        "nix_evaluation_identity"
                    ],
                    "compatibility_source": "factory_nix_evaluation",
                },
            },
        },
        "native_attention_runtime_binding": {
            "status": "verified",
            "expected_path": str(native_attention_library.resolve()),
            "mapped_path": str(native_attention_library.resolve()),
            "basename": "libattn_kernels_xe_2.so",
            "unique_basename_mapping": True,
        },
        "requested_settings": {
            "fixture_mode": "matched-production",
            "output_dtypes": [output_dtype],
            "matrix": [
                {field: case[field] for field in matrix_fields} for case in cases
            ],
        },
        "completed_cases": len(cases),
        "results": cases,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _correctness(
    path: Path,
    candidate_id: str = "candidate-store-path",
    native_layout: str = "xe2_dpas",
    native_kernel_variant: str = "q6_scalar",
    native_split_policy: str = "b70_q6",
    native_splits: dict[int, int] | None = None,
    flush_index_materialization: str = "per_layer",
    native_frontend: str = "reference",
) -> Path:
    selected_splits = (
        dict(gate_module.B70_Q6_SPLITS) if native_splits is None else native_splits
    )
    root = path.parent / "correctness-evidence"
    root.mkdir()
    short = gate_module.CORRECTNESS_SHORT_FIXTURES
    short_results = [_correctness_result(name) for name in short]
    short_comparisons = [_correctness_comparison(name) for name in short]
    long_result = _correctness_result("reasoning-261631")
    long_comparison = _correctness_comparison("reasoning-261631")
    idle = {
        "vllm:num_requests_running": 0.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:kv_cache_usage_perc": 0.0,
    }
    cancellation = {
        "requested_generated_token_checkpoint": 257,
        "generated_token_ids_before_close": 257,
        "idle_metrics_before_replacement": idle,
        "replacement": _correctness_result("reasoning-65023"),
        "comparison": _correctness_comparison("reasoning-65023"),
    }
    overlap = {
        "required_running": 4,
        "peak_running": 4.0,
        "required_overlap_observed": True,
    }
    workloads = {
        "native-65k-b1-first": {
            "first": short_results,
            "replay": short_results,
            "replay_comparisons": short_comparisons,
            "cancellation": cancellation,
        },
        "native-65k-b1-restart": {
            "results": short_results,
            "comparisons": short_comparisons,
        },
        "native-65k-b4": {
            "results": short_results,
            "comparisons": short_comparisons,
            "overlap": overlap,
        },
        "reference-262k-b1": {"result": long_result},
        "native-262k-b1-first": {
            "result": long_result,
            "reference_comparison": long_comparison,
        },
        "native-262k-b1-restart": {
            "result": long_result,
            "reference_comparison": long_comparison,
            "native_restart_comparison": long_comparison,
        },
    }

    phases: dict[str, dict[str, str]] = {}
    for phase_name in gate_module.CORRECTNESS_PHASE_SPECS:
        spec = gate_module._correctness_phase_spec(
            phase_name,
            native_layout,
            native_frontend,
            flush_index_materialization,
            native_kernel_variant,
            native_split_policy,
            selected_splits,
        )
        effective_layout = spec["native_layout"]
        effective_kernel = spec["native_kernel_variant"]
        effective_policy = spec["native_split_policy"]
        effective_frontend = spec["native_frontend"]
        max_splits = spec["max_decode_splits"]
        splits_environment = (
            None
            if spec["native"]
            and gate_module.split_policy.owns_runtime_selection(effective_policy)
            else str(max_splits)
        )
        marker = gate_module._factory_marker(
            effective_layout, effective_kernel, max_splits, effective_policy
        )
        variant = {field: spec[field] for field in gate_module.VARIANT_FIELDS}
        phase_dir = root / phase_name
        phase_dir.mkdir()
        profile = phase_dir / "profile.json"
        profile.write_text(
            json.dumps(
                {
                    "native_layout": effective_layout,
                    "native_layout_environment": gate_module.NATIVE_LAYOUT_ENV[
                        effective_layout
                    ],
                    "native_cache_layout_environment": effective_layout,
                    "native_kernel_variant_environment": effective_kernel,
                    "native_max_splits_environment": splits_environment,
                    "native_split_policy_environment": effective_policy,
                    "flush_index_materialization_environment": (
                        flush_index_materialization
                    ),
                    "native_frontend_environment": effective_frontend,
                    "redacted_environment": {
                        "KVARN_FLUSH_INDEX_MATERIALIZATION": (
                            flush_index_materialization
                        ),
                        "KVARN_NATIVE_XPU_FRONTEND": effective_frontend,
                        "KVARN_NATIVE_XPU_DPAS_LAYOUT": gate_module.NATIVE_LAYOUT_ENV[
                            effective_layout
                        ],
                        "KVARN_NATIVE_XPU_CACHE_LAYOUT": effective_layout,
                        "KVARN_NATIVE_XPU_KERNEL_VARIANT": effective_kernel,
                        "KVARN_NATIVE_XPU_SPLITS": splits_environment,
                        "KVARN_NATIVE_XPU_SPLIT_POLICY": effective_policy,
                        "KVARN_ONEDNN_DETERMINISTIC": "1",
                        "VLLM_USE_V2_MODEL_RUNNER": "0",
                    },
                    "variant_provenance": variant,
                }
            ),
            encoding="utf-8",
        )
        identity = phase_dir / "identity.json"
        identity.write_text(
            json.dumps(
                {
                    "candidate_env": candidate_id,
                    "process_package": "/nix/store/package",
                }
            ),
            encoding="utf-8",
        )
        engine_log = phase_dir / "engine.log"
        engine_log.write_text(
            f"INFO {marker}\n"
            + (
                "INFO " + gate_module.NATIVE_DISPATCH + " (direct bf16 output=True)\n"
                if spec["native"]
                else "INFO reference reader\n"
            )
            + (
                f"INFO {gate_module.NATIVE_FRONTEND_ACTIVE_MARKER} layer=test\n"
                if spec["native"] and effective_frontend == "qkv_scatter"
                else ""
            ),
            encoding="utf-8",
        )
        log_scan = phase_dir / "log-scan.json"
        log_scan.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "fatal_findings": [],
                    "native_direct_bf16_verified": spec["native"],
                    "native_direct_bf16_log_marker": (
                        gate_module.NATIVE_DIRECT_BF16_MARKER
                        if spec["native"]
                        else "not_applicable"
                    ),
                    "native_frontend_expected": effective_frontend,
                    "native_frontend_active_verified": (
                        spec["native"] and effective_frontend == "qkv_scatter"
                    ),
                    "native_frontend_log_marker": (
                        gate_module.NATIVE_FRONTEND_ACTIVE_MARKER
                        if spec["native"] and effective_frontend == "qkv_scatter"
                        else "not_applicable"
                    ),
                }
            ),
            encoding="utf-8",
        )
        phase = phase_dir / "phase.json"
        phase.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "spec": spec,
                    "native_dispatch_verified": spec["native"],
                    "native_layout": effective_layout,
                    "native_layout_environment": gate_module.NATIVE_LAYOUT_ENV[
                        effective_layout
                    ],
                    "native_cache_layout_environment": effective_layout,
                    "native_kernel_variant": effective_kernel,
                    "native_kernel_variant_id": spec["native_kernel_variant_id"],
                    "native_kernel_variant_environment": effective_kernel,
                    "native_max_splits": max_splits,
                    "native_nominal_splits": spec["nominal_decode_splits"],
                    "native_output_dtype": "bf16",
                    "native_direct_bf16_verified": spec["native"],
                    "native_direct_bf16_log_marker": (
                        gate_module.NATIVE_DIRECT_BF16_MARKER
                        if spec["native"]
                        else "not_applicable"
                    ),
                    "native_max_splits_environment": splits_environment,
                    "native_split_policy": variant["split_policy"],
                    "native_split_policy_contract": spec[
                        "native_split_policy_contract"
                    ],
                    "native_split_policy_environment": effective_policy,
                    "native_layout_log_marker": marker,
                    "native_layout_evidence": (
                        "captured-process-environment-plus-factory-marker-plus-native-dispatch"
                        if spec["native"]
                        else "captured-process-environment-plus-factory-marker"
                    ),
                    "flush_index_materialization": flush_index_materialization,
                    "native_frontend": effective_frontend,
                    "native_frontend_active_verified": (
                        spec["native"] and effective_frontend == "qkv_scatter"
                    ),
                    "native_frontend_log_marker": (
                        gate_module.NATIVE_FRONTEND_ACTIVE_MARKER
                        if spec["native"] and effective_frontend == "qkv_scatter"
                        else "not_applicable"
                    ),
                    "profile": _artifact(profile),
                    "identity": _artifact(identity),
                    "engine_log": _artifact(engine_log),
                    "engine_log_scan": _artifact(log_scan),
                    "workload": workloads[phase_name],
                }
            ),
            encoding="utf-8",
        )
        phases[phase_name] = _artifact(phase)

    primitive_files = {}
    for filename in (
        "native.so",
        "test.py",
        "helper.py",
        "utils.py",
        "pytest.log",
        "pytest.xml",
    ):
        primitive_file = root / filename
        primitive_file.write_text("evidence\n", encoding="utf-8")
        primitive_files[filename] = _artifact(primitive_file)
    junit = root / "pytest.xml"
    junit.write_text(
        '<testsuite tests="1" failures="0" errors="0" skipped="0"/>\n',
        encoding="utf-8",
    )
    primitive_files["pytest.xml"] = _artifact(junit)
    gates: dict[str, dict[str, str]] = {}
    gate_documents: dict[str, dict[str, object]] = {
        name: {
            "status": "passed",
            "gate": name,
            "candidate_id": candidate_id,
            "qualification_scope": "combined_library_variant_matrix",
            "variant_selection": "explicit_per_op_arguments",
            "factory_variant_matrix": gate_module.COMBINED_LIBRARY_VARIANT_MATRIX,
            "command": [
                "/nix/store/python/bin/python",
                "-m",
                "pytest",
                primitive_files["test.py"]["path"],
                "-k",
                (
                    "not long_context_ragged_b4_matches_structured_oracle"
                    if name == "native_decode_short"
                    else "long_context_ragged_b4_matches_structured_oracle"
                ),
            ],
            "junit_counts": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
            "native_library": primitive_files["native.so"],
            "test_source": primitive_files["test.py"],
            "helper_sources": {
                "benchmark/check_kvarn_decode.py": primitive_files["helper.py"],
                "benchmark/kvarn_utils.py": primitive_files["utils.py"],
            },
            "pytest_log": primitive_files["pytest.log"],
            "junit": primitive_files["pytest.xml"],
        }
        for name in ("native_decode_short", "native_decode_262k")
    }
    gate_documents.update(
        {
            "b1_replay": {
                "status": "passed",
                "gate": "b1_replay",
                "service_phase": phases["native-65k-b1-first"],
                "comparisons": short_comparisons,
            },
            "cancel_reuse": {
                "status": "passed",
                "gate": "cancel_reuse",
                "service_phase": phases["native-65k-b1-first"],
                **cancellation,
            },
            "b1_restart": {
                "status": "passed",
                "gate": "b1_restart",
                "original_service_phase": phases["native-65k-b1-first"],
                "restarted_service_phase": phases["native-65k-b1-restart"],
                "comparisons": short_comparisons,
            },
            "b4_isolation": {
                "status": "passed",
                "gate": "b4_isolation",
                "b1_service_phase": phases["native-65k-b1-first"],
                "b4_service_phase": phases["native-65k-b4"],
                "comparisons": short_comparisons,
                "overlap": overlap,
            },
            "near_262k_reference_equivalence": {
                "status": "passed",
                "gate": "near_262k_reference_equivalence",
                "reference_service_phase": phases["reference-262k-b1"],
                "native_service_phase": phases["native-262k-b1-first"],
                "comparison": long_comparison,
            },
            "near_262k_restart": {
                "status": "passed",
                "gate": "near_262k_restart",
                "reference_service_phase": phases["reference-262k-b1"],
                "first_native_service_phase": phases["native-262k-b1-first"],
                "restarted_native_service_phase": phases["native-262k-b1-restart"],
                "reference_comparison": long_comparison,
                "native_restart_comparison": long_comparison,
            },
        }
    )
    for name in gate_module.REQUIRED_GATES:
        evidence = root / f"{name}.json"
        evidence.write_text(json.dumps(gate_documents[name]), encoding="utf-8")
        gates[name] = {"status": "passed", **_artifact(evidence)}
    runner_sources = {}
    for name in gate_module.CORRECTNESS_RUNNER_SOURCES:
        source = root / "runner" / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {name}\n", encoding="utf-8")
        runner_sources[name] = _artifact(source)
    revisions = {
        "vllm-xpu-release": "1" * 40,
        "vllm-xpu-unstable-src": "2" * 40,
        "vllm-xpu-kernels-unstable-src": "3" * 40,
    }
    lock = root / "flake.lock"
    lock.write_text(
        json.dumps(
            {
                "nodes": {
                    name: {"locked": {"rev": revision}}
                    for name, revision in revisions.items()
                }
            }
        ),
        encoding="utf-8",
    )
    clean_checkout = {
        "head": "4" * 40,
        "files": 1,
        "sha256": "5" * 64,
        "unexpected_changes": [],
    }
    factory_path = _factory_result(
        root / "factory-result.json",
        native_library=Path(primitive_files["native.so"]["path"]),
        revisions=revisions,
        native_kernel_variant=native_kernel_variant,
        native_splits=selected_splits,
        native_split_policy=native_split_policy,
    )
    factory_qualification = gate_module.validate_factory_qualification(
        factory_path,
        native_layout=native_layout,
        native_kernel_variant=native_kernel_variant,
        native_split_policy=native_split_policy,
        native_splits=selected_splits,
        output_dtype="bf16",
        expected_revisions={
            "vllm-xpu-nix": revisions["vllm-xpu-release"],
            "vllm": revisions["vllm-xpu-unstable-src"],
            "vllm-xpu-kernels": revisions["vllm-xpu-kernels-unstable-src"],
        },
        expected_package="/nix/store/package",
        expected_native_library=primitive_files["native.so"]["path"],
        expected_native_library_sha256=primitive_files["native.so"]["sha256"],
    )
    document = {
        "status": "passed",
        "candidate_id": candidate_id,
        "native_layout": native_layout,
        "native_kernel_variant": native_kernel_variant,
        "native_kernel_variant_id": gate_module.NATIVE_KERNEL_VARIANTS[
            native_kernel_variant
        ],
        "native_nominal_splits_by_batch": (
            gate_module.split_policy.nominal_splits_by_batch(
                native_split_policy, selected_splits or None
            )
        ),
        "native_split_policy_contract": gate_module.split_policy.split_policy_contract(
            native_split_policy, selected_splits or None
        ),
        "native_output_dtype": "bf16",
        "native_split_policy": native_split_policy,
        "flush_index_materialization": flush_index_materialization,
        "native_frontend": native_frontend,
        "service_controls": {
            "kvarn_flush_index_materialization": flush_index_materialization,
            "kvarn_native_frontend": native_frontend,
            "kvarn_onednn_deterministic": "1",
            "vllm_use_v2_model_runner": "0",
        },
        "native_scratch_max_splits": gate_module.split_policy.split_policy_contract(
            native_split_policy, selected_splits or None
        )["scratch_max_splits"],
        **gate_module._candidate_variant_provenance(
            native_layout,
            native_frontend,
            flush_index_materialization,
            native_kernel_variant,
            native_split_policy,
            selected_splits,
        ),
        "native_dispatch_verified": True,
        "native_direct_bf16_verified": True,
        "native_direct_bf16_log_marker": gate_module.NATIVE_DIRECT_BF16_MARKER,
        "factory_qualification": factory_qualification,
        "service_start_plan": [
            gate_module._correctness_phase_spec(
                name,
                native_layout,
                native_frontend,
                flush_index_materialization,
                native_kernel_variant,
                native_split_policy,
                selected_splits,
            )
            for name in gate_module.CORRECTNESS_PHASE_SPECS
        ],
        "gates": gates,
        "candidate_identity": {
            "process_package": "/nix/store/package",
            "candidate_closure_sha256": "b" * 64,
            "process_closure_sha256": "a" * 64,
        },
        "source_identity": {
            "lock_path": str(lock),
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "revisions": revisions,
            "config_checkout": clean_checkout,
            "runner_checkout": clean_checkout,
            "kernel_tracked_checkout": {
                **clean_checkout,
                "head": revisions["vllm-xpu-kernels-unstable-src"],
            },
            "runner_sources": runner_sources,
            "native_source_sha256": {
                name: "8" * 64 for name in gate_module.CORRECTNESS_NATIVE_SOURCES
            },
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _result(
    path: Path,
    *,
    arm: str,
    run_order: int,
    correctness_sha256: str,
    engine_log_sha256: str,
    output_throughput: float,
    request_throughput: float,
    ttft: float,
    itl: float,
    kv_cache_dtype: str | None = None,
    native_splits: int | None = None,
    native_layout: str = "natural",
    native_kernel_variant: str = "q6_scalar",
    native_split_policy: str = "b70_q6",
    native_split_map: dict[int, int] | None = None,
    flush_index_materialization: str = "per_layer",
    native_frontend: str = "reference",
) -> Path:
    completed = 8
    context = 4096
    output_tokens = 512
    output_lens = [output_tokens] * completed
    input_lens = [context] * completed
    duration = sum(output_lens) / output_throughput
    hardware = path.parent / "hardware-preflight.json"
    hardware.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "torch_version": "test",
                "xpu_available": True,
                "xpu_device_count": 1,
                "xpu_device_names": ["Intel(R) Arc(TM) Pro B70 Graphics"],
                "probe_device": "xpu:0",
                "probe_value": 6.0,
            }
        ),
        encoding="utf-8",
    )
    warmup = path.with_name(f"{path.stem}-warmup.json")
    warmup_raw = path.with_name(f"{path.stem}-warmup.raw.json")
    warmup_raw.write_text(
        json.dumps(
            {
                "completed": 4,
                "num_prompts": 4,
                "failed": 0,
                "max_concurrency": 4,
                "max_concurrent_requests": 4,
                "input_lens": [context] * 4,
                "output_lens": [output_tokens] * 4,
            }
        ),
        encoding="utf-8",
    )
    warmup.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "arm": arm,
                "run_uuid": f"run-{run_order}",
                "workload": {
                    "context": context,
                    "batch": 4,
                    "output_tokens": output_tokens,
                    "num_prompts": 4,
                    "seed": 17,
                },
                "argv": [
                    "vllm",
                    "--random-input-len",
                    str(context),
                    "--random-output-len",
                    str(output_tokens),
                    "--num-prompts",
                    "4",
                    "--num-warmups",
                    "0",
                    "--max-concurrency",
                    "4",
                    "--seed",
                    "17",
                ],
                "raw_result": str(warmup_raw.resolve()),
                "raw_result_sha256": hashlib.sha256(
                    warmup_raw.read_bytes()
                ).hexdigest(),
                "completed": 4,
                "failed": 0,
                "max_concurrent_requests": 4,
                "process_package": "/nix/store/package",
                "process_closure_sha256": "a" * 64,
                "candidate_closure_sha256": "b" * 64,
                "matched_profile_sha256": "c" * 64,
                "native_layout": ("natural" if arm == "reference" else native_layout),
                "native_layout_environment": (
                    "0"
                    if arm == "reference"
                    else gate_module.NATIVE_LAYOUT_ENV[native_layout]
                ),
                "variant_provenance": (
                    gate_module._performance_reference_variant_provenance()
                    if arm == "reference"
                    else gate_module._candidate_variant_provenance(
                        native_layout,
                        native_frontend,
                        flush_index_materialization,
                        native_kernel_variant,
                        native_split_policy,
                        (
                            dict(gate_module.B70_Q6_SPLITS)
                            if native_split_map is None
                            else native_split_map
                        ),
                    )
                ),
            }
        ),
        encoding="utf-8",
    )
    selected_layout = "natural" if arm == "reference" else native_layout
    selected_kernel = "baseline" if arm == "reference" else native_kernel_variant
    selected_policy = "fixed" if arm == "reference" else native_split_policy
    selected_split_map = (
        dict(gate_module.B70_Q6_SPLITS)
        if native_split_map is None
        else native_split_map
    )
    effective_splits = (
        native_splits
        if native_splits is not None
        else (
            1
            if arm == "reference"
            else gate_module.split_policy.effective_splits(
                selected_policy,
                batch=4,
                context_tokens=context,
                fixed_splits=selected_split_map or None,
            )
        )
    )
    max_splits = (
        gate_module.split_policy.split_policy_contract(
            selected_policy, selected_split_map or None
        )["scratch_max_splits"]
        if arm == "candidate"
        and gate_module.split_policy.owns_runtime_selection(selected_policy)
        else effective_splits
    )
    marker = (
        gate_module._factory_marker(
            selected_layout, selected_kernel, max_splits, selected_policy
        )
        if arm == "candidate"
        else "unavailable"
    )
    document = {
        "completed": completed,
        "failed": 0,
        "output_throughput": output_throughput,
        "request_throughput": request_throughput,
        "total_token_throughput": (sum(input_lens) + sum(output_lens)) / duration,
        "duration": duration,
        "total_input_tokens": sum(input_lens),
        "total_output_tokens": sum(output_lens),
        "input_lens": input_lens,
        "output_lens": output_lens,
        "ttfts": [ttft] * completed,
        "itls": [[itl] * (length - 1) for length in output_lens],
        "backend": "openai",
        "model_id": "sunny-chat",
        "tokenizer_id": "jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound",
        "num_prompts": completed,
        "request_rate": "inf",
        "max_concurrency": 4,
        "max_concurrent_requests": 4,
        "kvarn_evidence_mode": "formal",
        "kvarn_promotable": True,
        "kvarn_candidate_id": "candidate-store-path",
        "kvarn_model_revision": "6b0622f4354481d5d04577d48ba0db844efc1330",
        "kvarn_service_profile": "qwen38-64k-b4-eager",
        "kvarn_workload_id": "fixed-b4",
        "kvarn_seed": "17",
        "kvarn_max_model_len": "65536",
        "kvarn_max_num_seqs": "4",
        "kvarn_enforce_eager": "1",
        "kvarn_prefix_caching": "0",
        "kvarn_mtp": "0",
        "kvarn_xpu_graph": "0",
        "kvarn_scheduler_peak_running": "4",
        "kvarn_correctness_sha256": correctness_sha256,
        "kvarn_process_package": "/nix/store/package",
        "kvarn_process_closure_sha256": "a" * 64,
        "kvarn_candidate_closure_sha256": "b" * 64,
        "kvarn_max_num_batched_tokens": "2048",
        "kvarn_flush_index_materialization": flush_index_materialization,
        "kvarn_native_frontend": (
            "reference" if arm == "reference" else native_frontend
        ),
        "kvarn_onednn_deterministic": "1",
        "kvarn_vllm_use_v2_model_runner": "0",
        "kvarn_matched_profile_sha256": "c" * 64,
        "kvarn_accelerator": "xpu",
        "kvarn_xpu_available": "1",
        "kvarn_xpu_device_count": "1",
        "kvarn_xpu_device_name": "Intel(R) Arc(TM) Pro B70 Graphics",
        "kvarn_xpu_compute_probe": "passed",
        "kvarn_hardware_preflight_path": str(hardware.resolve()),
        "kvarn_hardware_preflight_sha256": hashlib.sha256(
            hardware.read_bytes()
        ).hexdigest(),
        "kvarn_warmup_path": str(warmup.resolve()),
        "kvarn_warmup_sha256": hashlib.sha256(warmup.read_bytes()).hexdigest(),
        "kvarn_xpu_consumed_memory_gib": 17.54,
        "kvarn_xpu_kv_cache_memory_gib": 10.92,
        "kvarn_arm": arm,
        "kvarn_kv_cache_dtype": kv_cache_dtype
        or ("auto" if arm == "reference" else "kvarn_k4v4_g128_compact"),
        "kvarn_native_xpu": "0" if arm == "reference" else "1",
        "kvarn_native_layout": selected_layout,
        "kvarn_native_layout_environment": (
            "0" if arm == "reference" else gate_module.NATIVE_LAYOUT_ENV[native_layout]
        ),
        "kvarn_native_cache_layout_environment": selected_layout,
        "kvarn_native_kernel_variant": selected_kernel,
        "kvarn_native_kernel_variant_id": str(
            gate_module.NATIVE_KERNEL_VARIANTS[selected_kernel]
        ),
        "kvarn_native_max_splits": str(max_splits),
        "kvarn_native_nominal_splits": str(effective_splits),
        "kvarn_native_output_dtype": (
            "not_applicable" if arm == "reference" else "bf16"
        ),
        "kvarn_native_direct_bf16_verified": arm == "candidate",
        "kvarn_native_direct_bf16_log_marker": (
            gate_module.NATIVE_DIRECT_BF16_MARKER
            if arm == "candidate"
            else "not_applicable"
        ),
        "kvarn_native_split_policy": selected_policy,
        "kvarn_native_layout_log_marker": marker,
        "kvarn_native_layout_evidence": (
            "captured-process-environment"
            if arm == "reference"
            else "captured-process-environment-plus-factory-marker-plus-native-dispatch"
        ),
        **{
            f"kvarn_{field}": value
            for field, value in (
                gate_module._performance_reference_variant_provenance()
                if arm == "reference"
                else gate_module._candidate_variant_provenance(
                    native_layout,
                    native_frontend,
                    flush_index_materialization,
                    native_kernel_variant,
                    native_split_policy,
                    selected_split_map,
                )
            ).items()
        },
        "kvarn_run_order": str(run_order),
        "kvarn_run_uuid": f"run-{run_order}",
        "kvarn_run_started_at": f"2026-08-31T00:00:{run_order:02d}Z",
        "kvarn_engine_log_sha256": engine_log_sha256,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _log(
    path: Path,
    *,
    native: bool,
    native_layout: str = "xe2_dpas",
    native_kernel_variant: str = "q6_scalar",
    native_split_policy: str = "b70_q6",
    native_max_splits: int = 32,
    native_frontend: str = "reference",
) -> Path:
    lines = [
        "INFO config: device_config=xpu",
        (
            "INFO Actual usage is 17.54 GiB for consumed memory. "
            "Current kv cache memory in use is 10.92 GiB."
        ),
    ]
    if native:
        lines.append(
            "INFO "
            + gate_module._factory_marker(
                native_layout,
                native_kernel_variant,
                native_max_splits,
                native_split_policy,
            )
        )
        lines.append(
            "INFO Using the native Xe2 KVarN qlen=1 decoder (direct bf16 output=True)"
        )
        if native_frontend == "qkv_scatter":
            lines.append(f"INFO {gate_module.NATIVE_FRONTEND_ACTIVE_MARKER} layer=test")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _arms(
    tmp_path: Path,
    *,
    reference_value: float = 100.0,
    candidate_value: float = 97.0,
    reference_ttft: float = 0.100,
    candidate_ttft: float = 0.105,
    reference_itl: float = 0.050,
    candidate_itl: float = 0.052,
    native_layout: str = "xe2_dpas",
    native_kernel_variant: str = "q6_scalar",
    native_split_policy: str = "b70_q6",
    native_splits: dict[int, int] | None = None,
    native_frontend: str = "reference",
    flush_index_materialization: str = "per_layer",
) -> tuple[list[Path], list[Path], list[Path], list[Path], Path]:
    selected_splits = (
        dict(gate_module.B70_Q6_SPLITS) if native_splits is None else native_splits
    )
    correctness = _correctness(
        tmp_path / "correctness.json",
        native_layout=native_layout,
        native_kernel_variant=native_kernel_variant,
        native_split_policy=native_split_policy,
        native_splits=selected_splits,
        native_frontend=native_frontend,
        flush_index_materialization=flush_index_materialization,
    )
    digest = hashlib.sha256(correctness.read_bytes()).hexdigest()
    reference_orders = (1, 4, 5, 8, 9, 12, 13, 16)
    candidate_orders = (2, 3, 6, 7, 10, 11, 14, 15)
    reference_logs = [
        _log(tmp_path / f"reference-{index}.log", native=False)
        for index in range(len(reference_orders))
    ]
    candidate_logs = [
        _log(
            tmp_path / f"candidate-{index}.log",
            native=True,
            native_layout=native_layout,
            native_kernel_variant=native_kernel_variant,
            native_split_policy=native_split_policy,
            native_max_splits=(
                gate_module.split_policy.split_policy_contract(
                    native_split_policy, selected_splits or None
                )["scratch_max_splits"]
                if gate_module.split_policy.owns_runtime_selection(
                    native_split_policy
                )
                else selected_splits[4]
            ),
            native_frontend=native_frontend,
        )
        for index in range(len(candidate_orders))
    ]
    references = [
        _result(
            tmp_path / f"reference-{index}.json",
            arm="reference",
            run_order=order,
            correctness_sha256=digest,
            engine_log_sha256=hashlib.sha256(
                reference_logs[index].read_bytes()
            ).hexdigest(),
            output_throughput=reference_value,
            request_throughput=reference_value / 512,
            ttft=reference_ttft,
            itl=reference_itl,
            flush_index_materialization=flush_index_materialization,
        )
        for index, order in enumerate(reference_orders)
    ]
    candidates = [
        _result(
            tmp_path / f"candidate-{index}.json",
            arm="candidate",
            run_order=order,
            correctness_sha256=digest,
            engine_log_sha256=hashlib.sha256(
                candidate_logs[index].read_bytes()
            ).hexdigest(),
            output_throughput=candidate_value,
            request_throughput=candidate_value / 512,
            ttft=candidate_ttft,
            itl=candidate_itl,
            native_layout=native_layout,
            native_kernel_variant=native_kernel_variant,
            native_split_policy=native_split_policy,
            native_split_map=selected_splits,
            native_frontend=native_frontend,
            flush_index_materialization=flush_index_materialization,
        )
        for index, order in enumerate(candidate_orders)
    ]
    return references, candidates, reference_logs, candidate_logs, correctness


def _compare(
    arms: tuple[list[Path], list[Path], list[Path], list[Path], Path],
    *,
    mode: str = "match",
    comparison_kind: str = "end-to-end",
) -> dict[str, object]:
    references, candidates, reference_logs, candidate_logs, correctness = arms
    return compare(
        references,
        candidates,
        reference_logs=reference_logs,
        candidate_logs=candidate_logs,
        correctness_path=correctness,
        comparison_kind=comparison_kind,
        mode=mode,
        min_throughput_ratio=0.95,
        min_request_decode_ratio=0.95,
        max_latency_ratio=1.10,
    )


def test_match_gate_uses_repeat_medians_and_both_perf_axes(tmp_path: Path) -> None:
    result = _compare(_arms(tmp_path))

    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert result["candidate_over_reference"]["output_throughput"] == pytest.approx(
        0.97
    )
    assert result["candidate_over_reference"][
        "median_request_decode_throughput"
    ] == pytest.approx(0.05 / 0.052)


def test_match_gate_accepts_qkv_candidate_with_unfused_reference(
    tmp_path: Path,
) -> None:
    arms = _arms(
        tmp_path,
        native_frontend="qkv_scatter",
        flush_index_materialization="shared",
    )

    result = _compare(arms)

    assert result["status"] == "passed"
    assert result["reference"]["arm"]["kvarn_native_frontend"] == "reference"
    assert result["candidate"]["arm"]["kvarn_native_frontend"] == "qkv_scatter"
    assert result["candidate"]["arm"]["kvarn_fusion_strategy"] == (
        "native_materializer_persistent_scratch_shared_flush_qkv_scatter_frontend"
    )
    assert (
        "-shared-flush-qkv_scatter-frontend-"
        in result["candidate"]["arm"]["kvarn_variant_id"]
    )
    assert result["reference"]["arm"]["kvarn_variant_id"] == (
        "auto-control-eager_mnbt2048"
    )


def test_match_gate_allows_canonical_auto_flush_with_shared_kvarn_candidate(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path, flush_index_materialization="shared")
    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_flush_index_materialization"] = "per_layer"
        path.write_text(json.dumps(document), encoding="utf-8")

    result = _compare(arms)

    assert result["status"] == "passed"
    assert result["reference"]["arm"]["kvarn_flush_index_materialization"] == (
        "per_layer"
    )
    assert result["candidate"]["arm"]["kvarn_flush_index_materialization"] == ("shared")


def test_gate_rejects_qkv_marker_in_reference_log(tmp_path: Path) -> None:
    arms = _arms(tmp_path, native_frontend="qkv_scatter")
    reference_log = arms[2][0]
    reference_log.write_text(
        reference_log.read_text(encoding="utf-8")
        + f"INFO {gate_module.NATIVE_FRONTEND_ACTIVE_MARKER} layer=test\n",
        encoding="utf-8",
    )

    with pytest.raises(GateError, match="reference frontend runtime proof differs"):
        _compare(arms)


def test_correctness_gate_rejects_qkv_marker_in_natural_reference(
    tmp_path: Path,
) -> None:
    correctness_path = _correctness(
        tmp_path / "correctness.json", native_frontend="qkv_scatter"
    )
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    outer_gate = correctness["gates"]["near_262k_reference_equivalence"]
    gate_path = Path(outer_gate["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    phase_path = Path(gate["reference_service_phase"]["path"])
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    log_path = Path(phase["engine_log"]["path"])
    log_path.write_text(
        log_path.read_text(encoding="utf-8")
        + f"INFO {gate_module.NATIVE_FRONTEND_ACTIVE_MARKER} layer=test\n",
        encoding="utf-8",
    )
    phase["engine_log"] = _artifact(log_path)
    phase_path.write_text(json.dumps(phase), encoding="utf-8")
    gate["reference_service_phase"] = _artifact(phase_path)
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    outer_gate["sha256"] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    correctness_path.write_text(json.dumps(correctness), encoding="utf-8")

    with pytest.raises(GateError, match="reference-262k-b1 frontend runtime proof"):
        _load_correctness(correctness_path)


def test_gate_accepts_dpas_only_with_matching_correctness_layout(
    tmp_path: Path,
) -> None:
    dpas_root = tmp_path / "dpas"
    dpas_root.mkdir()
    arms = _arms(dpas_root, native_layout="xe2_dpas")
    assert _compare(arms)["status"] == "passed"

    natural_root = tmp_path / "natural"
    natural_root.mkdir()
    natural_correctness = _correctness(
        natural_root / "correctness.json", native_kernel_variant="q6_vector"
    )
    natural_digest = hashlib.sha256(natural_correctness.read_bytes()).hexdigest()
    for result_path in (*arms[0], *arms[1]):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["kvarn_correctness_sha256"] = natural_digest
        result_path.write_text(json.dumps(result), encoding="utf-8")
    mismatched = (*arms[:4], natural_correctness)
    with pytest.raises(GateError, match="variant provenance must match"):
        _compare(mismatched)


def test_gate_accepts_b70_q6_with_exact_factory_provenance(tmp_path: Path) -> None:
    arms = _arms(
        tmp_path,
        native_layout="xe2_dpas",
        native_kernel_variant="q6_scalar",
        native_split_policy="b70_q6",
        native_splits={1: 32, 4: 8},
    )

    result = _compare(arms)

    assert result["status"] == "passed"
    assert result["candidate"]["arm"]["kvarn_native_max_splits"] == "32"
    assert result["candidate"]["arm"]["kvarn_native_nominal_splits"] == "8"
    assert result["candidate"]["arm"]["kvarn_native_split_policy"] == "b70_q6"


def test_formal_gate_accepts_exact_b70_q6_v2_contract(tmp_path: Path) -> None:
    arms = _arms(
        tmp_path,
        native_layout="xe2_dpas",
        native_kernel_variant="q6_next_page_prefetch",
        native_split_policy="b70_q6_v2",
        native_splits={},
    )

    result = _compare(arms)

    assert result["status"] == "passed"
    assert result["candidate"]["arm"]["kvarn_native_max_splits"] == "32"
    assert result["candidate"]["arm"]["kvarn_native_nominal_splits"] == "8"
    assert result["candidate"]["arm"]["kvarn_native_split_policy"] == "b70_q6_v2"


def test_factory_qualification_binds_context_dependent_b70_q6_v2(
    tmp_path: Path,
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"selected native library")
    revisions = {
        "vllm-xpu-release": "1" * 40,
        "vllm-xpu-unstable-src": "2" * 40,
        "vllm-xpu-kernels-unstable-src": "3" * 40,
    }
    factory = _factory_result(
        tmp_path / "factory.json",
        native_library=library,
        revisions=revisions,
        native_kernel_variant="q6_next_page_prefetch",
        native_splits={},
        native_split_policy="b70_q6_v2",
    )

    qualification = gate_module.validate_factory_qualification(
        factory,
        native_layout="xe2_dpas",
        native_kernel_variant="q6_next_page_prefetch",
        native_split_policy="b70_q6_v2",
        native_splits={},
        output_dtype="bf16",
        expected_revisions={
            "vllm-xpu-nix": revisions["vllm-xpu-release"],
            "vllm": revisions["vllm-xpu-unstable-src"],
            "vllm-xpu-kernels": revisions["vllm-xpu-kernels-unstable-src"],
        },
        expected_package="/nix/store/package",
        expected_native_library=str(library.resolve()),
        expected_native_library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    )

    selection = qualification["selection"]
    assert selection["nominal_splits_by_batch"] is None
    assert selection["effective_splits_by_context_and_batch"] == {
        "4096": {"1": 32, "4": 8},
        "16384": {"1": 32, "4": 8},
        "65023": {"1": 32, "4": 32},
    }
    assert selection["split_policy_contract"]["kernel_compatibility"] == {
        "kind": "exact_variant",
        "name": "q6_next_page_prefetch",
        "id": 12,
    }


@pytest.mark.parametrize(
    ("override", "expected_error"),
    [
        ({"native_layout": "natural"}, "requires xe2_dpas"),
        ({"native_kernel_variant": "q6_vector"}, "coverage is incomplete"),
        ({"output_dtype": "fp16"}, "result matrix is incomplete"),
        (
            {"expected_revisions": {"vllm-xpu-nix": "a" * 40}},
            "source revisions differ",
        ),
        ({"expected_native_library_sha256": "0" * 64}, "library identity differs"),
    ],
)
def test_factory_qualification_fails_closed_on_selected_identity_mismatch(
    tmp_path: Path, override: dict[str, object], expected_error: str
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"selected native library")
    revisions = {
        "vllm-xpu-release": "1" * 40,
        "vllm-xpu-unstable-src": "2" * 40,
        "vllm-xpu-kernels-unstable-src": "3" * 40,
    }
    factory = _factory_result(
        tmp_path / "factory.json",
        native_library=library,
        revisions=revisions,
        native_kernel_variant="q6_scalar",
        native_splits=dict(gate_module.B70_Q6_SPLITS),
    )
    arguments: dict[str, object] = {
        "native_layout": "xe2_dpas",
        "native_kernel_variant": "q6_scalar",
        "native_split_policy": "b70_q6",
        "native_splits": dict(gate_module.B70_Q6_SPLITS),
        "output_dtype": "bf16",
        "expected_revisions": {
            "vllm-xpu-nix": revisions["vllm-xpu-release"],
            "vllm": revisions["vllm-xpu-unstable-src"],
            "vllm-xpu-kernels": revisions["vllm-xpu-kernels-unstable-src"],
        },
        "expected_package": "/nix/store/package",
        "expected_native_library": str(library.resolve()),
        "expected_native_library_sha256": hashlib.sha256(
            library.read_bytes()
        ).hexdigest(),
    }
    if "expected_revisions" in override:
        requested_revisions = dict(arguments["expected_revisions"])
        requested_revisions.update(override["expected_revisions"])
        arguments["expected_revisions"] = requested_revisions
    else:
        arguments.update(override)

    with pytest.raises(GateError, match=expected_error):
        gate_module.validate_factory_qualification(factory, **arguments)


def test_factory_qualification_accepts_and_checks_explicit_factory_axes(
    tmp_path: Path,
) -> None:
    library = tmp_path / "native.so"
    library.write_bytes(b"selected native library")
    revisions = {
        "vllm-xpu-release": "1" * 40,
        "vllm-xpu-unstable-src": "2" * 40,
        "vllm-xpu-kernels-unstable-src": "3" * 40,
    }
    factory = _factory_result(
        tmp_path / "factory.json",
        native_library=library,
        revisions=revisions,
        native_kernel_variant="q6_scalar",
        native_splits=dict(gate_module.B70_Q6_SPLITS),
        explicit_factory_axes=True,
    )
    arguments = {
        "native_layout": "xe2_dpas",
        "native_kernel_variant": "q6_scalar",
        "native_split_policy": "b70_q6",
        "native_splits": dict(gate_module.B70_Q6_SPLITS),
        "output_dtype": "bf16",
        "expected_revisions": {
            "vllm-xpu-nix": revisions["vllm-xpu-release"],
            "vllm": revisions["vllm-xpu-unstable-src"],
            "vllm-xpu-kernels": revisions["vllm-xpu-kernels-unstable-src"],
        },
        "expected_package": "/nix/store/package",
        "expected_native_library": str(library.resolve()),
        "expected_native_library_sha256": hashlib.sha256(
            library.read_bytes()
        ).hexdigest(),
    }
    assert gate_module.validate_factory_qualification(factory, **arguments)[
        "status"
    ] == "passed"

    document = json.loads(factory.read_text(encoding="utf-8"))
    document["results"][0]["kernel_strategy"] = "wrong"
    factory.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="matrix differs"):
        gate_module.validate_factory_qualification(factory, **arguments)


def test_correctness_rehashes_bound_factory_artifact(tmp_path: Path) -> None:
    correctness = _correctness(tmp_path / "correctness.json")
    document = json.loads(correctness.read_text(encoding="utf-8"))
    factory = Path(document["factory_qualification"]["factory_artifact"]["path"])
    factory_document = json.loads(factory.read_text(encoding="utf-8"))
    factory_document["tampered_after_binding"] = True
    factory.write_text(json.dumps(factory_document), encoding="utf-8")

    with pytest.raises(GateError, match="artifact SHA differs"):
        _load_correctness(correctness)


def test_gate_rejects_incompatible_or_unmarked_factory_selection(
    tmp_path: Path,
) -> None:
    invalid = _correctness(tmp_path / "invalid.json")
    invalid_document = json.loads(invalid.read_text(encoding="utf-8"))
    invalid_document["native_layout"] = "natural"
    invalid.write_text(json.dumps(invalid_document), encoding="utf-8")
    with pytest.raises(GateError, match="require.*xe2_dpas"):
        _load_correctness(invalid)

    marked_root = tmp_path / "marked"
    marked_root.mkdir()
    arms = _arms(
        marked_root,
        native_layout="xe2_dpas",
        native_kernel_variant="q6_scalar",
        native_split_policy="b70_q6",
        native_splits={1: 32, 4: 8},
    )
    arms[3][0].write_text(
        arms[3][0]
        .read_text(encoding="utf-8")
        .replace("max_decode_splits=32", "max_decode_splits=8"),
        encoding="utf-8",
    )
    with pytest.raises(GateError, match="exact factory marker"):
        _compare(arms)


def test_correctness_manifest_binds_dpas_service_plan(tmp_path: Path) -> None:
    correctness_path = _correctness(
        tmp_path / "correctness.json", native_layout="xe2_dpas"
    )
    document = json.loads(correctness_path.read_text(encoding="utf-8"))
    document["service_start_plan"][0]["native_layout"] = "natural"
    correctness_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="service_start_plan differs"):
        _load_correctness(correctness_path)


def test_gate_binds_candidate_variant_provenance_to_correctness(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    for result_path in arms[1]:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["kvarn_variant_id"] = "unqualified-variant"
        warmup_path = Path(result["kvarn_warmup_path"])
        warmup = json.loads(warmup_path.read_text(encoding="utf-8"))
        warmup["variant_provenance"]["variant_id"] = "unqualified-variant"
        warmup_path.write_text(json.dumps(warmup), encoding="utf-8")
        result["kvarn_warmup_sha256"] = hashlib.sha256(
            warmup_path.read_bytes()
        ).hexdigest()
        result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(GateError, match="variant provenance must match"):
        _compare(arms)


def test_win_mode_rejects_a_latency_for_throughput_trade(tmp_path: Path) -> None:
    result = _compare(
        _arms(
            tmp_path,
            candidate_value=110.0,
            candidate_ttft=0.110,
            candidate_itl=0.055,
        ),
        mode="win",
    )

    assert result["status"] == "failed"
    assert result["checks"]["meaningful_throughput_gain"]
    assert not result["checks"]["median_request_decode_throughput"]
    assert not result["checks"]["p99_ttft"]
    assert not result["checks"]["p99_itl"]


def test_win_mode_requires_a_meaningful_gain(tmp_path: Path) -> None:
    result = _compare(
        _arms(
            tmp_path,
            candidate_value=100.1,
            candidate_ttft=0.0999,
            candidate_itl=0.0499,
        ),
        mode="win",
    )

    assert result["status"] == "failed"
    assert not result["checks"]["meaningful_throughput_gain"]


def test_gate_rejects_unmatched_workload_shapes(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    document = json.loads(arms[1][0].read_text(encoding="utf-8"))
    document["input_lens"][0] = 4097
    document["total_input_tokens"] = sum(document["input_lens"])
    document["total_token_throughput"] = (
        document["total_input_tokens"] + document["total_output_tokens"]
    ) / document["duration"]
    arms[1][0].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="context.*matrix"):
        _compare(arms)


def test_gate_rejects_unbalanced_repeat_counts(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    arms[1].pop()
    arms[3].pop()

    with pytest.raises(GateError, match="at least eight candidate repeats"):
        _compare(arms)


def test_kernel_comparison_requires_kvarn_in_both_arms(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    with pytest.raises(GateError, match="compact Kvarn in both arms"):
        _compare(arms, comparison_kind="kernel")


def test_kernel_comparison_passes_with_compact_kvarn_reference(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_kv_cache_dtype"] = "kvarn_k4v4_g128_compact"
        path.write_text(json.dumps(document), encoding="utf-8")

    assert _compare(arms, comparison_kind="kernel")["status"] == "passed"


def test_gate_requires_neutral_reference_and_supported_candidate_splits(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_nominal_splits"] = "16"
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="native-layout evidence is inconsistent"):
        _compare(arms)

    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_nominal_splits"] = "1"
        path.write_text(json.dumps(document), encoding="utf-8")
    for path in arms[1]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_nominal_splits"] = "3"
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="native-layout evidence is inconsistent"):
        _compare(arms)


def test_gate_rejects_missing_native_dispatch(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    text = arms[3][0].read_text(encoding="utf-8")
    arms[3][0].write_text(
        text.replace(
            "INFO Using the native Xe2 KVarN qlen=1 decoder "
            "(direct bf16 output=True)\n",
            "",
        ),
        encoding="utf-8",
    )

    with pytest.raises(GateError, match="must contain native dispatch"):
        _compare(arms)


@pytest.mark.parametrize(
    "replacement",
    ["direct bf16 output=False", "direct output path unavailable"],
    ids=("disabled", "missing"),
)
def test_gate_rejects_disabled_or_missing_direct_bf16(
    tmp_path: Path, replacement: str
) -> None:
    arms = _arms(tmp_path)
    candidate_log = arms[3][0]
    candidate_log.write_text(
        candidate_log.read_text(encoding="utf-8").replace(
            gate_module.NATIVE_DIRECT_BF16_MARKER, replacement
        ),
        encoding="utf-8",
    )

    with pytest.raises(GateError, match="direct BF16 runtime proof"):
        _compare(arms)


def test_gate_allows_unrelated_fallback_but_rejects_kvarn_fallback(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    candidate_log = arms[3][0]
    candidate_log.write_text(
        candidate_log.read_text(encoding="utf-8")
        + "INFO Falling back to the Triton GDN decode path\n"
        + "WARNING sampler is Falling back to PyTorch-native implementation\n",
        encoding="utf-8",
    )
    candidate_result = json.loads(arms[1][0].read_text(encoding="utf-8"))
    candidate_result["kvarn_engine_log_sha256"] = hashlib.sha256(
        candidate_log.read_bytes()
    ).hexdigest()
    arms[1][0].write_text(json.dumps(candidate_result), encoding="utf-8")

    assert _compare(arms)["status"] == "passed"

    candidate_log.write_text(
        candidate_log.read_text(encoding="utf-8")
        + "WARNING Falling back from the Kvarn native decoder\n",
        encoding="utf-8",
    )
    candidate_result["kvarn_engine_log_sha256"] = hashlib.sha256(
        candidate_log.read_bytes()
    ).hexdigest()
    arms[1][0].write_text(json.dumps(candidate_result), encoding="utf-8")
    with pytest.raises(GateError, match="fallback"):
        _compare(arms)


def test_gate_rejects_cpu_or_zero_residency_engine_logs(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    cpu_log = arms[2][0]
    cpu_log.write_text(
        cpu_log.read_text(encoding="utf-8").replace(
            "device_config=xpu", "device_config=cpu"
        ),
        encoding="utf-8",
    )
    with pytest.raises(GateError, match="device_config=xpu"):
        _compare(arms)

    zero_root = tmp_path / "zero"
    zero_root.mkdir()
    arms = _arms(zero_root)
    zero_log = arms[2][0]
    zero_log.write_text(
        zero_log.read_text(encoding="utf-8")
        .replace("17.54 GiB", "0.0 GiB")
        .replace("10.92 GiB", "0.0 GiB"),
        encoding="utf-8",
    )
    with pytest.raises(GateError, match="positive XPU model/KV residency"):
        _compare(arms)


def test_gate_requires_hashed_b70_and_full_width_warmup_evidence(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    document = json.loads(arms[1][0].read_text(encoding="utf-8"))
    hardware = Path(document["kvarn_hardware_preflight_path"])
    probe = json.loads(hardware.read_text(encoding="utf-8"))
    probe["xpu_device_names"] = ["Intel Arc A770"]
    hardware.write_text(json.dumps(probe), encoding="utf-8")
    digest = hashlib.sha256(hardware.read_bytes()).hexdigest()
    for path in [*arms[0], *arms[1]]:
        result = json.loads(path.read_text(encoding="utf-8"))
        result["kvarn_hardware_preflight_sha256"] = digest
        path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(GateError, match="exact B70 proof"):
        _compare(arms)

    warmup_root = tmp_path / "warmup"
    warmup_root.mkdir()
    arms = _arms(warmup_root)
    result = json.loads(arms[1][0].read_text(encoding="utf-8"))
    warmup = Path(result["kvarn_warmup_path"])
    evidence = json.loads(warmup.read_text(encoding="utf-8"))
    evidence["max_concurrent_requests"] = 1
    warmup.write_text(json.dumps(evidence), encoding="utf-8")
    result["kvarn_warmup_sha256"] = hashlib.sha256(warmup.read_bytes()).hexdigest()
    arms[1][0].write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(GateError, match="not full-width"):
        _compare(arms)


def test_gate_rejects_mismatched_build_or_service_profile(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    for path in arms[1]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_process_closure_sha256"] = "d" * 64
        warmup = Path(document["kvarn_warmup_path"])
        warmup_document = json.loads(warmup.read_text(encoding="utf-8"))
        warmup_document["process_closure_sha256"] = "d" * 64
        warmup.write_text(json.dumps(warmup_document), encoding="utf-8")
        document["kvarn_warmup_sha256"] = hashlib.sha256(
            warmup.read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="provenance field.*differs"):
        _compare(arms)


def test_gate_binds_performance_to_correctness_candidate_build(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    for path in [*arms[0], *arms[1]]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_process_package"] = "/nix/store/other-package"
        document["kvarn_process_closure_sha256"] = "d" * 64
        document["kvarn_candidate_closure_sha256"] = "e" * 64
        warmup = Path(document["kvarn_warmup_path"])
        warmup_document = json.loads(warmup.read_text(encoding="utf-8"))
        warmup_document["process_package"] = "/nix/store/other-package"
        warmup_document["process_closure_sha256"] = "d" * 64
        warmup_document["candidate_closure_sha256"] = "e" * 64
        warmup.write_text(json.dumps(warmup_document), encoding="utf-8")
        document["kvarn_warmup_sha256"] = hashlib.sha256(
            warmup.read_bytes()
        ).hexdigest()
        path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="different candidate builds"):
        _compare(arms)


def test_gate_binds_warmup_workload_and_raw_artifact(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    result = json.loads(arms[1][0].read_text(encoding="utf-8"))
    warmup = Path(result["kvarn_warmup_path"])
    warmup_document = json.loads(warmup.read_text(encoding="utf-8"))
    warmup_document["workload"]["context"] = 16384
    warmup.write_text(json.dumps(warmup_document), encoding="utf-8")
    result["kvarn_warmup_sha256"] = hashlib.sha256(warmup.read_bytes()).hexdigest()
    arms[1][0].write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(GateError, match="warmup workload differs"):
        _compare(arms)

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    arms = _arms(raw_root)
    result = json.loads(arms[1][0].read_text(encoding="utf-8"))
    warmup = Path(result["kvarn_warmup_path"])
    warmup_document = json.loads(warmup.read_text(encoding="utf-8"))
    Path(warmup_document["raw_result"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(GateError, match="raw-result SHA-256 differs"):
        _compare(arms)


def test_gate_requires_globally_distinct_warmup_evidence(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    first = json.loads(arms[1][0].read_text(encoding="utf-8"))
    second = json.loads(arms[1][1].read_text(encoding="utf-8"))
    second["kvarn_run_uuid"] = first["kvarn_run_uuid"]
    second["kvarn_warmup_path"] = first["kvarn_warmup_path"]
    second["kvarn_warmup_sha256"] = first["kvarn_warmup_sha256"]
    arms[1][1].write_text(json.dumps(second), encoding="utf-8")

    with pytest.raises(GateError, match="distinct warmup evidence"):
        _compare(arms)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_throughput_ratio", 0.94),
        ("min_request_decode_ratio", 0.94),
        ("max_latency_ratio", 1.11),
    ],
)
def test_gate_rejects_weakened_formal_thresholds(
    tmp_path: Path, field: str, value: float
) -> None:
    arms = _arms(tmp_path)
    kwargs = {
        "min_throughput_ratio": 0.95,
        "min_request_decode_ratio": 0.95,
        "max_latency_ratio": 1.10,
    }
    kwargs[field] = value
    with pytest.raises(GateError, match="formal comparison thresholds"):
        compare(
            arms[0],
            arms[1],
            reference_logs=arms[2],
            candidate_logs=arms[3],
            correctness_path=arms[4],
            comparison_kind="end-to-end",
            mode="match",
            **kwargs,
        )


def test_gate_rejects_non_promotable_exploratory_results(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    document = json.loads(arms[1][0].read_text(encoding="utf-8"))
    document["kvarn_evidence_mode"] = "exploratory"
    document["kvarn_promotable"] = False
    arms[1][0].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="must be promotable"):
        _compare(arms)


def test_gate_rejects_correctness_evidence_hash_mismatch(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    correctness = json.loads(arms[4].read_text(encoding="utf-8"))
    evidence = Path(correctness["gates"]["b4_isolation"]["path"])
    evidence.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    with pytest.raises(GateError, match="evidence SHA differs"):
        _compare(arms)


def test_gate_rejects_mismatched_correctness_evidence_identity(tmp_path: Path) -> None:
    correctness_path = _correctness(tmp_path / "correctness.json")
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    gate = correctness["gates"]["b4_isolation"]
    evidence = Path(gate["path"])
    document = json.loads(evidence.read_text(encoding="utf-8"))
    document["gate"] = "b1_replay"
    evidence.write_text(json.dumps(document), encoding="utf-8")
    gate["sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    correctness_path.write_text(json.dumps(correctness), encoding="utf-8")

    with pytest.raises(GateError, match="gate identity/status differs"):
        _load_correctness(correctness_path)


def test_gate_requires_complete_runner_source_identity(tmp_path: Path) -> None:
    correctness_path = _correctness(tmp_path / "correctness.json")
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    correctness["source_identity"]["runner_sources"].pop(
        "scripts/kvarn_service_gate.py"
    )
    correctness_path.write_text(json.dumps(correctness), encoding="utf-8")

    with pytest.raises(GateError, match="runner source evidence is incomplete"):
        _load_correctness(correctness_path)


def test_gate_binds_service_phase_to_exact_candidate(tmp_path: Path) -> None:
    correctness_path = _correctness(tmp_path / "correctness.json")
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    outer_gate = correctness["gates"]["b4_isolation"]
    gate_path = Path(outer_gate["path"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    phase_path = Path(gate["b4_service_phase"]["path"])
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    identity_path = Path(phase["identity"]["path"])
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["candidate_env"] = "different-candidate"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    phase["identity"]["sha256"] = hashlib.sha256(identity_path.read_bytes()).hexdigest()
    phase_path.write_text(json.dumps(phase), encoding="utf-8")
    gate["b4_service_phase"]["sha256"] = hashlib.sha256(
        phase_path.read_bytes()
    ).hexdigest()
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    outer_gate["sha256"] = hashlib.sha256(gate_path.read_bytes()).hexdigest()
    correctness_path.write_text(json.dumps(correctness), encoding="utf-8")

    with pytest.raises(GateError, match="identity differs from the candidate"):
        _load_correctness(correctness_path)


def test_gate_recursively_rehashes_nested_correctness_evidence(tmp_path: Path) -> None:
    correctness_path = _correctness(tmp_path / "correctness.json")
    correctness = json.loads(correctness_path.read_text(encoding="utf-8"))
    gate = correctness["gates"]["b4_isolation"]
    gate_path = Path(gate["path"])
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    phase_path = Path(gate_document["b4_service_phase"]["path"])
    phase = json.loads(phase_path.read_text(encoding="utf-8"))
    nested = Path(phase["profile"]["path"])
    nested.write_text('{"tampered":true}\n', encoding="utf-8")

    with pytest.raises(GateError, match="artifact SHA differs"):
        _load_correctness(correctness_path)


def test_gate_rejects_claimed_throughput_inconsistent_with_counts(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    document = json.loads(arms[1][0].read_text(encoding="utf-8"))
    document["output_throughput"] *= 2
    arms[1][0].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="inconsistent with duration"):
        _compare(arms)
