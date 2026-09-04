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


def _correctness(path: Path, candidate_id: str = "candidate-store-path") -> Path:
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
    for phase_name, spec in gate_module.CORRECTNESS_PHASE_SPECS.items():
        phase_dir = root / phase_name
        phase_dir.mkdir()
        profile = phase_dir / "profile.json"
        profile.write_text("{}\n", encoding="utf-8")
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
        "native_dispatch_verified": True,
        "gates": gates,
        "candidate_identity": {
            "process_package": "/nix/store/package",
            "candidate_closure_sha256": "6" * 64,
            "process_closure_sha256": "7" * 64,
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
) -> Path:
    output_lens = [4, 4, 4, 4]
    input_lens = [127, 4095, 127, 4095]
    duration = sum(output_lens) / output_throughput
    document = {
        "completed": 4,
        "failed": 0,
        "output_throughput": output_throughput,
        "request_throughput": request_throughput,
        "total_token_throughput": (sum(input_lens) + sum(output_lens)) / duration,
        "duration": duration,
        "total_input_tokens": sum(input_lens),
        "total_output_tokens": sum(output_lens),
        "input_lens": input_lens,
        "output_lens": output_lens,
        "ttfts": [ttft] * 4,
        "itls": [[itl] * (length - 1) for length in output_lens],
        "backend": "openai",
        "model_id": "sunny-chat",
        "tokenizer_id": "model-repo",
        "num_prompts": 4,
        "request_rate": "inf",
        "max_concurrency": 4,
        "max_concurrent_requests": 4,
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
        "kvarn_arm": arm,
        "kvarn_kv_cache_dtype": kv_cache_dtype
        or ("auto" if arm == "reference" else "kvarn_k4v4_g128_compact"),
        "kvarn_native_xpu": "0" if arm == "reference" else "1",
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
    lines = ["INFO engine ready"]
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
) -> tuple[list[Path], list[Path], list[Path], list[Path], Path]:
    correctness = _correctness(tmp_path / "correctness.json")
    digest = hashlib.sha256(correctness.read_bytes()).hexdigest()
    reference_orders = (1, 4, 5, 8)
    candidate_orders = (2, 3, 6, 7)
    reference_logs = [
        _log(tmp_path / f"reference-{index}.log", native=False) for index in range(4)
    ]
    candidate_logs = [
        _log(tmp_path / f"candidate-{index}.log", native=True) for index in range(4)
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
            request_throughput=reference_value / 4,
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
            request_throughput=candidate_value / 4,
            ttft=candidate_ttft,
            itl=candidate_itl,
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
    document["input_lens"] = [128, 4095, 127, 4095]
    document["total_input_tokens"] = sum(document["input_lens"])
    document["total_token_throughput"] = (
        document["total_input_tokens"] + document["total_output_tokens"]
    ) / document["duration"]
    arms[1][0].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="workload shape"):
        _compare(arms)


def test_gate_rejects_unbalanced_repeat_counts(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    arms[1].pop()
    arms[3].pop()

    with pytest.raises(GateError, match="at least four candidate repeats"):
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
    arms[3][0].write_text("INFO engine ready\n", encoding="utf-8")

    with pytest.raises(GateError, match="must contain native dispatch"):
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
