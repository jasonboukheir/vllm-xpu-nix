from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import kvarn_perf_gate as gate_module
from scripts.kvarn_perf_gate import GateError, _load_correctness, compare


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


def _correctness(
    path: Path,
    candidate_id: str = "candidate-store-path",
    native_layout: str = "natural",
) -> Path:
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
        spec = gate_module._correctness_phase_spec(phase_name, native_layout)
        effective_layout = spec["native_layout"]
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
                    "redacted_environment": {
                        "KVARN_NATIVE_XPU_DPAS_LAYOUT": gate_module.NATIVE_LAYOUT_ENV[
                            effective_layout
                        ]
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
            "INFO " + gate_module.NATIVE_DISPATCH + "\n"
            if spec["native"]
            else "INFO reference reader\n",
            encoding="utf-8",
        )
        log_scan = phase_dir / "log-scan.json"
        log_scan.write_text(
            json.dumps({"status": "passed", "fatal_findings": []}),
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
                    "native_layout_log_marker": "unavailable",
                    "native_layout_evidence": (
                        "captured-process-environment-plus-native-dispatch"
                        if spec["native"]
                        else "captured-process-environment"
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
            "native_layout": native_layout,
            **gate_module._candidate_variant_provenance(native_layout),
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
    document = {
        "status": "passed",
        "candidate_id": candidate_id,
        "native_layout": native_layout,
        **gate_module._candidate_variant_provenance(native_layout),
        "native_dispatch_verified": True,
        "service_start_plan": [
            gate_module._correctness_phase_spec(name, native_layout)
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
                    else gate_module._candidate_variant_provenance(native_layout)
                ),
            }
        ),
        encoding="utf-8",
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
        "kvarn_native_layout": "natural" if arm == "reference" else native_layout,
        "kvarn_native_layout_environment": (
            "0" if arm == "reference" else gate_module.NATIVE_LAYOUT_ENV[native_layout]
        ),
        "kvarn_native_layout_log_marker": "unavailable",
        "kvarn_native_layout_evidence": (
            "captured-process-environment"
            if arm == "reference"
            else "captured-process-environment-plus-native-dispatch"
        ),
        **{
            f"kvarn_{field}": value
            for field, value in (
                gate_module._performance_reference_variant_provenance()
                if arm == "reference"
                else gate_module._candidate_variant_provenance(native_layout)
            ).items()
        },
        "kvarn_native_splits": str(
            native_splits
            if native_splits is not None
            else (1 if arm == "reference" else 16)
        ),
        "kvarn_run_order": str(run_order),
        "kvarn_run_uuid": f"run-{run_order}",
        "kvarn_run_started_at": f"2026-08-31T00:00:{run_order:02d}Z",
        "kvarn_engine_log_sha256": engine_log_sha256,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _log(path: Path, *, native: bool) -> Path:
    lines = [
        "INFO config: device_config=xpu",
        (
            "INFO Actual usage is 17.54 GiB for consumed memory. "
            "Current kv cache memory in use is 10.92 GiB."
        ),
    ]
    if native:
        lines.append("INFO Using the native Xe2 KVarN qlen=1 decoder")
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
    native_layout: str = "natural",
) -> tuple[list[Path], list[Path], list[Path], list[Path], Path]:
    correctness = _correctness(
        tmp_path / "correctness.json", native_layout=native_layout
    )
    digest = hashlib.sha256(correctness.read_bytes()).hexdigest()
    reference_orders = (1, 4, 5, 8, 9, 12, 13, 16)
    candidate_orders = (2, 3, 6, 7, 10, 11, 14, 15)
    reference_logs = [
        _log(tmp_path / f"reference-{index}.log", native=False)
        for index in range(len(reference_orders))
    ]
    candidate_logs = [
        _log(tmp_path / f"candidate-{index}.log", native=True)
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


def test_gate_accepts_dpas_only_with_matching_correctness_layout(
    tmp_path: Path,
) -> None:
    dpas_root = tmp_path / "dpas"
    dpas_root.mkdir()
    arms = _arms(dpas_root, native_layout="xe2_dpas")
    assert _compare(arms)["status"] == "passed"

    natural_root = tmp_path / "natural"
    natural_root.mkdir()
    natural_correctness = _correctness(natural_root / "correctness.json")
    natural_digest = hashlib.sha256(natural_correctness.read_bytes()).hexdigest()
    for result_path in (*arms[0], *arms[1]):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["kvarn_correctness_sha256"] = natural_digest
        result_path.write_text(json.dumps(result), encoding="utf-8")
    mismatched = (*arms[:4], natural_correctness)
    with pytest.raises(GateError, match="layout must match correctness"):
        _compare(mismatched)


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
        document["kvarn_native_splits"] = "16"
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="neutral split count 1"):
        _compare(arms)

    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_splits"] = "1"
        path.write_text(json.dumps(document), encoding="utf-8")
    for path in arms[1]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_splits"] = "3"
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="supported native split count"):
        _compare(arms)


def test_gate_rejects_missing_native_dispatch(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    text = arms[3][0].read_text(encoding="utf-8")
    arms[3][0].write_text(
        text.replace("INFO Using the native Xe2 KVarN qlen=1 decoder\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(GateError, match="must contain native dispatch"):
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
