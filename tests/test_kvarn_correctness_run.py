from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.kvarn_correctness_run as correctness


def test_service_plan_has_exact_six_restarts_and_selected_splits() -> None:
    assert [spec.name for spec in correctness.SERVICE_PLAN] == [
        "native-65k-b1-first",
        "native-65k-b1-restart",
        "native-65k-b4",
        "reference-262k-b1",
        "native-262k-b1-first",
        "native-262k-b1-restart",
    ]
    assert correctness.PRIMITIVE_PLAN == (
        (
            "native_decode_short",
            "not long_context_ragged_b4_matches_structured_oracle",
        ),
        (
            "native_decode_262k",
            "long_context_ragged_b4_matches_structured_oracle",
        ),
    )


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [2 + ord(character) % 31 for character in text]


def test_exact_prompt_ids_is_deterministic_and_preserves_trailing_task() -> None:
    tokenizer = FakeTokenizer()
    prompt = "preserve this instruction"
    first = correctness.exact_prompt_ids(
        tokenizer, prompt, "reasoning", 400, trailing_prompt=True
    )
    second = correctness.exact_prompt_ids(
        tokenizer, prompt, "reasoning", 400, trailing_prompt=True
    )
    suffix = tokenizer.encode(
        "\n\nFinal task after reviewing the records:\n" + prompt,
        add_special_tokens=False,
    )

    assert len(first) == 400
    assert first == second
    assert first[-len(suffix) :] == suffix


def test_tokenize_worker_generates_code_fixture_with_trailing_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text(
        json.dumps(
            [
                {"category": category, "prompt": f"{category} prompt"}
                for category in ("dialogue", "code", "math", "reasoning")
            ]
        ),
        encoding="utf-8",
    )
    tokenizer = object()
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: tokenizer
            )
        ),
    )
    calls: dict[str, tuple[str, int, bool]] = {}

    def fake_exact_prompt_ids(
        actual_tokenizer: object,
        prompt: str,
        category: str,
        target: int,
        *,
        trailing_prompt: bool,
    ) -> list[int]:
        assert actual_tokenizer is tokenizer
        calls[category] = (prompt, target, trailing_prompt)
        return [target]

    monkeypatch.setattr(correctness, "exact_prompt_ids", fake_exact_prompt_ids)
    assert (
        correctness.tokenize_worker(
            [
                "--model",
                "model",
                "--revision",
                "1" * 40,
                "--fixtures",
                str(fixtures),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["code-4095"] == [4095]
    assert calls["code"] == ("code prompt", 4095, True)


def test_compact_results_and_exact_comparison_fail_closed() -> None:
    raw = {
        "id": "fixture",
        "prompt_token_ids_sha256": "a" * 64,
        "token_ids": [1, 2],
        "token_ids_sha256": "b" * 64,
        "raw_response": {"choices": [{"prompt_token_ids": [9] * 100}]},
        "text": "discarded",
    }
    compact = correctness.compact_result(raw)
    assert "raw_response" not in compact
    assert "text" not in compact
    assert correctness.compare_results(compact, dict(compact))["status"] == "passed"

    changed = {**compact, "token_ids": [1, 3], "token_ids_sha256": "c" * 64}
    with pytest.raises(correctness.CorrectnessError, match="token IDs differ"):
        correctness.compare_results(compact, changed)


def test_completion_usage_and_length_finish_are_mandatory() -> None:
    fixture = {"prompt": [1, 2, 3]}
    result = {
        "finish_reason": "length",
        "raw_response": {
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 512,
                "total_tokens": 515,
            }
        },
    }

    assert correctness.validate_completion_result(result, fixture, 512) == {
        "prompt_tokens": 3,
        "completion_tokens": 512,
        "total_tokens": 515,
    }
    result["raw_response"]["usage"]["prompt_tokens"] = 2
    with pytest.raises(correctness.CorrectnessError, match="usage mismatch"):
        correctness.validate_completion_result(result, fixture, 512)


def test_token_fixture_artifact_records_hashes_not_prompt_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_lengths = {
        "dialogue-127": 127,
        "code-4095": 4095,
        "math-16383": 16383,
        "reasoning-65023": 65023,
        "reasoning-261631": correctness.NEAR_262K_PROMPT_TOKENS,
    }
    prompts = {
        name: [index % 97 for index in range(length)]
        for name, length in expected_lengths.items()
    }
    completed = subprocess.CompletedProcess(
        ["python"], 0, stdout=json.dumps(prompts), stderr=""
    )
    monkeypatch.setattr(
        correctness.subprocess, "run", lambda *args, **kwargs: completed
    )
    base = tmp_path / "fixtures.json"
    base.write_text("[]\n", encoding="utf-8")
    args = argparse.Namespace(
        primitive_python=tmp_path / "python",
        model="model",
        model_revision="1" * 40,
        fixtures=base,
        packaging_repo=tmp_path,
        output_dir=tmp_path / "output",
        output_tokens=512,
        tokenizer_timeout=10.0,
        runtime_cache=tmp_path / "cache",
        hf_home=tmp_path / "hf",
    )
    args.output_dir.mkdir()

    fixtures = correctness.tokenize_fixtures(args)
    manifest_text = (args.output_dir / "fixture-manifest.json").read_text()
    manifest = json.loads(manifest_text)

    assert len(fixtures["reasoning-261631"]["prompt"]) == 261631
    assert manifest["status"] == "passed"
    assert all("prompt" not in record for record in manifest["fixtures"])
    assert "[0,1,2,3" not in manifest_text


def _service_argv(spec: correctness.ServiceSpec, args: argparse.Namespace) -> list[str]:
    return [
        "/nix/store/package/bin/.vllm-wrapped",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model,
        "--revision",
        args.model_revision,
        "--dtype",
        "bfloat16",
        "--quantization",
        "compressed-tensors",
        "--kv-cache-dtype",
        "kvarn_k4v4_g128_compact",
        "--max-model-len",
        str(spec.max_model_len),
        "--max-num-seqs",
        str(spec.batch),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--language-model-only",
        "--no-enable-prefix-caching",
    ]


def _service_environment(
    spec: correctness.ServiceSpec, args: argparse.Namespace
) -> dict[str, str | None]:
    native = "1" if spec.native else "0"
    return {
        "CCL_ATL_TRANSPORT": "ofi",
        "CCL_LOG_LEVEL": "warn",
        "CCL_PROCESS_LAUNCHER": "none",
        "CCL_ZE_IPC_EXCHANGE": "sockets",
        "HF_HOME": str(args.hf_home),
        "HOME": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "KVARN_NATIVE_XPU": native,
        "KVARN_NATIVE_XPU_CACHE_LAYOUT": correctness.native_layout_for_spec(spec, args),
        "KVARN_NATIVE_XPU_DECODE": native,
        "KVARN_NATIVE_XPU_DPAS_LAYOUT": (
            correctness.perf.NATIVE_LAYOUT_ENV[args.native_layout]
            if spec.native
            else "0"
        ),
        "KVARN_FLUSH_INDEX_MATERIALIZATION": (
            correctness.perf.flush_index_materialization_environment(args)
        ),
        "KVARN_NATIVE_XPU_FRONTEND": correctness.perf.native_frontend_environment(args),
        "KVARN_NATIVE_XPU_KERNEL_VARIANT": (
            correctness.native_kernel_variant_for_spec(spec, args)
        ),
        "KVARN_NATIVE_XPU_MATERIALIZE": native,
        "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH": native,
        "KVARN_NATIVE_XPU_SPLITS": correctness.native_splits_environment_for_spec(
            spec, args
        ),
        "KVARN_NATIVE_XPU_SPLIT_POLICY": correctness.native_split_policy_for_spec(
            spec, args
        ),
        "KVARN_ONEDNN_DETERMINISTIC": "1",
        "KVARN_PREFILL_FP16_WINDOW_BLOCKS": "16",
        "VLLM_CACHE_ROOT": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_KVARN_DEFER_PREFILL_FLUSH": None,
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_XPU_ENABLE_XPU_GRAPH": None,
        "XDG_CACHE_HOME": str(args.runtime_cache),
    }


def test_service_profile_enforces_262k_reference_and_native_settings(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        model="model",
        served_model="sunny-chat",
        model_revision="1" * 40,
        max_num_batched_tokens=2048,
        native_layout="natural",
        native_kernel_variant="baseline",
        native_split_policy="fixed",
        native_splits={1: 24, 4: 16},
        native_output_dtype="bf16",
        hf_home=tmp_path / "hf",
        runtime_cache=tmp_path / "cache",
    )
    for spec in (
        correctness.SERVICE_PLAN[3],
        correctness.SERVICE_PLAN[4],
    ):
        argv = _service_argv(spec, args)
        environment = _service_environment(spec, args)
        correctness.verify_service_profile(argv, environment, spec, args)
        environment["KVARN_NATIVE_XPU_SPLITS"] = "32"
        with pytest.raises(correctness.CorrectnessError, match="profile mismatch"):
            correctness.verify_service_profile(argv, environment, spec, args)


def test_dpas_mode_uses_separate_launchers_and_keeps_reference_natural(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        native_layout="xe2_dpas",
        native_kernel_variant="q6_scalar",
        native_split_policy="fixed",
        native_splits={1: 32, 4: 8},
        native_output_dtype="bf16",
        model="model",
        served_model="sunny-chat",
        model_revision="1" * 40,
        max_num_batched_tokens=2048,
        hf_home=tmp_path / "hf",
        runtime_cache=tmp_path / "cache",
    )
    native = correctness.SERVICE_PLAN[4]
    reference = correctness.SERVICE_PLAN[3]

    assert correctness.launcher_name(native, args) == (
        "vllm-xpu-brutus-kvarn-native-dpas-q6_scalar-262k-b1"
    )
    assert correctness.launcher_name(reference, args) == reference.launcher
    assert correctness.native_layout_for_spec(native, args) == "xe2_dpas"
    assert correctness.native_layout_for_spec(reference, args) == "natural"
    assert correctness.candidate_variant_provenance(args)["variant_id"] == (
        "native-xe2-xe2_dpas-q6_scalar-fixed_b1s32_b4s8-eager_mnbt2048"
    )
    assert correctness.service_variant_provenance(reference, args)["variant_id"] == (
        "natural-kvarn-correctness-reference-eager_mnbt2048"
    )
    correctness.verify_service_profile(
        _service_argv(native, args), _service_environment(native, args), native, args
    )
    correctness.verify_service_profile(
        _service_argv(reference, args),
        _service_environment(reference, args),
        reference,
        args,
    )


@pytest.mark.parametrize(
    ("variant", "variant_id"),
    [
        ("q6_scalar", 2),
        ("q6_vector", 4),
        ("q6_cached_weights", 6),
        ("q6_exact_rows", 7),
        ("q6_cached_weights_exact_rows", 8),
        ("q6_page_pair", 9),
        ("q6_main_grf128", 10),
        ("q6_split_reducer_specialized", 11),
        ("q6_next_page_prefetch", 12),
    ],
)
def test_dpas_launcher_names_bind_variant_for_65k_and_262k(
    variant: str, variant_id: int
) -> None:
    args = argparse.Namespace(native_layout="xe2_dpas", native_kernel_variant=variant)

    assert correctness.launcher_name(correctness.SERVICE_PLAN[0], args) == (
        f"vllm-xpu-brutus-kvarn-native-dpas-{variant}-b1"
    )
    assert correctness.launcher_name(correctness.SERVICE_PLAN[4], args) == (
        f"vllm-xpu-brutus-kvarn-native-dpas-{variant}-262k-b1"
    )
    assert correctness.perf.NATIVE_KERNEL_VARIANTS[variant] == variant_id


def test_service_environment_pins_bounded_window_and_scrubs_full_defer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", "999")
    monkeypatch.setenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", "1")
    args = argparse.Namespace(
        runtime_cache=tmp_path / "cache",
        hf_home=tmp_path / "hf",
    )

    environment = correctness.service_environment(args)

    assert environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] == "16"
    assert environment["KVARN_FACTORY_FLUSH_INDEX_MATERIALIZATION"] == "per_layer"
    assert environment["KVARN_FACTORY_NATIVE_XPU_FRONTEND"] == "reference"
    assert "VLLM_KVARN_DEFER_PREFILL_FLUSH" not in environment


def test_primitive_environment_disables_external_python_and_pytest_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        monkeypatch.setenv(name, "/tmp/untrusted")
    args = argparse.Namespace(
        runtime_cache=tmp_path / "cache",
        hf_home=tmp_path / "hf",
    )

    environment = correctness.primitive_environment(args)

    assert all(
        name not in environment
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS")
    )
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"


def test_manifest_rejects_content_free_gate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    args = argparse.Namespace(
        output_dir=tmp_path,
        candidate_env=candidate,
        expected_package=Path("/nix/store/package"),
        native_layout="natural",
        native_kernel_variant="baseline",
        native_split_policy="fixed",
        native_splits={1: 24, 4: 16},
        native_output_dtype="bf16",
        factory_qualification={"status": "passed"},
        max_num_batched_tokens=2048,
        source_identity={"revisions": {"vllm": "1" * 40}},
    )
    monkeypatch.setattr(correctness, "verify_config_identity", lambda _args: None)
    monkeypatch.setattr(correctness, "verify_packaging_identity", lambda _args: None)
    fixture_manifest = tmp_path / "fixture-manifest.json"
    fixture_manifest.write_text('{"status":"passed"}\n', encoding="utf-8")
    identity = {
        "candidate_env": str(candidate),
        "process_package": "/nix/store/package",
        "candidate_closure_sha256": "1" * 64,
        "process_closure_sha256": "2" * 64,
    }
    for spec in correctness.SERVICE_PLAN:
        phase_dir = tmp_path / "services" / spec.name
        phase_dir.mkdir(parents=True)
        (phase_dir / "candidate-identity.json").write_text(
            json.dumps(identity), encoding="utf-8"
        )
        (phase_dir / "phase.json").write_text(
            json.dumps(
                {
                    "status": "passed",
                    "native_dispatch_verified": spec.native,
                    "native_direct_bf16_verified": spec.native,
                    "native_direct_bf16_log_marker": (
                        correctness.perf.NATIVE_DIRECT_BF16_MARKER
                        if spec.native
                        else "not_applicable"
                    ),
                }
            ),
            encoding="utf-8",
        )
    gate_paths = {}
    for gate in correctness.REQUIRED_GATES:
        path = tmp_path / f"{gate}.json"
        path.write_text('{"status":"passed"}\n', encoding="utf-8")
        gate_paths[gate] = path

    with pytest.raises(correctness.CorrectnessError, match="gate identity/status"):
        correctness.build_manifest(args, gate_paths)


def test_primitive_gate_rejects_an_all_skipped_xpu_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_source = tmp_path / correctness.NATIVE_TEST
    test_source.parent.mkdir(parents=True)
    test_source.write_text("def test_x(): pass\n", encoding="utf-8")
    for relative in correctness.NATIVE_TEST_SOURCES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        if source != test_source:
            source.write_text("# helper\n", encoding="utf-8")
    native_library = tmp_path / "native.so"
    native_library.write_bytes(b"library")
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        primitive_python=tmp_path / "python",
        kernels_repo=tmp_path,
        native_library=native_library,
        candidate_env=tmp_path / "candidate",
        runtime_cache=tmp_path / "cache",
        hf_home=tmp_path / "hf",
        primitive_timeout=10.0,
        supervisor=object(),
        require_inactive_unit=list(correctness.REQUIRED_INACTIVE_UNITS),
        source_identity={
            "native_source_sha256": {
                relative: correctness.sha256_file(tmp_path / relative)
                for relative in correctness.NATIVE_TEST_SOURCES
            },
            "kernel_tracked_checkout": {"files": 3, "sha256": "a" * 64},
        },
    )
    args.output_dir.mkdir()

    def fake_run(command, **kwargs):
        junit = Path(command[command.index("--junitxml") + 1])
        junit.write_text(
            '<testsuites><testsuite tests="2" failures="0" errors="0" '
            'skipped="2"/></testsuites>',
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(correctness.perf, "run_managed_process", fake_run)
    monkeypatch.setattr(
        correctness,
        "assert_units_inactive",
        lambda units: {unit: "inactive" for unit in units},
    )
    monkeypatch.setattr(
        correctness,
        "kernel_checkout_identity",
        lambda _repo: args.source_identity["kernel_tracked_checkout"],
    )
    with pytest.raises(correctness.CorrectnessError, match="all-pass XPU run"):
        correctness.run_primitive_gate("native_decode_short", "anything", args)


def test_primitive_gate_rejects_tracked_checkout_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for relative in correctness.NATIVE_TEST_SOURCES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# source\n", encoding="utf-8")
    native_library = tmp_path / "native.so"
    native_library.write_bytes(b"library")
    expected_tree = {"files": 3, "sha256": "a" * 64}
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        primitive_python=tmp_path / "python",
        kernels_repo=tmp_path,
        native_library=native_library,
        candidate_env=tmp_path / "candidate",
        runtime_cache=tmp_path / "cache",
        hf_home=tmp_path / "hf",
        primitive_timeout=10.0,
        supervisor=object(),
        require_inactive_unit=list(correctness.REQUIRED_INACTIVE_UNITS),
        source_identity={
            "native_source_sha256": {
                relative: correctness.sha256_file(tmp_path / relative)
                for relative in correctness.NATIVE_TEST_SOURCES
            },
            "kernel_tracked_checkout": expected_tree,
        },
    )
    args.output_dir.mkdir()

    def fake_run(command, **kwargs):
        junit = Path(command[command.index("--junitxml") + 1])
        junit.write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" '
            'skipped="0"/></testsuites>',
            encoding="utf-8",
        )
        return 0

    identities = iter([expected_tree, {"files": 3, "sha256": "b" * 64}])
    monkeypatch.setattr(
        correctness, "kernel_checkout_identity", lambda _repo: next(identities)
    )
    monkeypatch.setattr(correctness.perf, "run_managed_process", fake_run)
    monkeypatch.setattr(
        correctness,
        "assert_units_inactive",
        lambda units: {unit: "inactive" for unit in units},
    )

    with pytest.raises(correctness.CorrectnessError, match="changed during pytest"):
        correctness.run_primitive_gate("native_decode_short", "anything", args)


def test_record_failure_preserves_completed_preflight(tmp_path: Path) -> None:
    session_path = tmp_path / "session.json"
    session_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "running",
                "candidate_id": "/nix/store/candidate",
                "resolved_launchers": {"native": "/nix/store/launcher/bin/native"},
            }
        ),
        encoding="utf-8",
    )

    result = correctness.record_failure(tmp_path, correctness.CorrectnessError("bad"))

    assert result["status"] == "failed"
    assert result["candidate_id"] == "/nix/store/candidate"
    assert result["resolved_launchers"]["native"].endswith("/bin/native")
    assert result["error"] == "CorrectnessError: bad"
    assert (tmp_path / "SHA256SUMS").is_file()


def test_cli_binds_config_ref_and_keeps_mandatory_inactive_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin/vllm").write_text("", encoding="utf-8")
    (candidate / "bin/python").write_text("", encoding="utf-8")
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(
        correctness, "DEFAULT_FIXTURE_SHA256", correctness.sha256_file(fixtures)
    )
    config = tmp_path / "config"
    config.mkdir()
    factory = tmp_path / "factory.json"
    factory.write_text("{}\n", encoding="utf-8")
    common = [
        "--candidate-env",
        str(candidate),
        "--fixtures",
        str(fixtures),
        "--runtime-cache",
        str(tmp_path / "cache"),
        "--factory-result",
        str(factory),
        "--native-layout",
        "xe2_dpas",
        "--native-kernel-variant",
        "q6_scalar",
        "--native-split-policy",
        "b70_q6",
        "--config-repo",
        str(config),
        "--allow-tmp",
        "--plan-only",
    ]
    with pytest.raises(SystemExit):
        correctness.parse_args(
            [
                *common,
                "--config-ref",
                f"path:{tmp_path / 'other-config'}",
                "--output-dir",
                str(tmp_path / "mismatch"),
            ]
        )
    with pytest.raises(SystemExit):
        correctness.parse_args(
            [
                *common,
                "--config-ref",
                f"path:{config}",
                "--cancel-after-events",
                "256",
                "--output-dir",
                str(tmp_path / "wrong-cancel-checkpoint"),
            ]
        )

    args = correctness.parse_args(
        [
            *common,
            "--config-ref",
            f"path:{config}",
            "--require-inactive-unit",
            "extra.service",
            "--output-dir",
            str(tmp_path / "valid"),
        ]
    )
    assert args.config_ref == f"path:{config.resolve()}"
    assert set(correctness.REQUIRED_INACTIVE_UNITS) < set(args.require_inactive_unit)
    assert args.native_layout == "xe2_dpas"

    dpas = correctness.parse_args(
        [
            *common,
            "--config-ref",
            f"path:{config}",
            "--native-layout",
            "xe2_dpas",
            "--output-dir",
            str(tmp_path / "valid-dpas"),
        ]
    )
    assert dpas.native_layout == "xe2_dpas"

    b70 = correctness.parse_args(
        [
            *common,
            "--config-ref",
            f"path:{config}",
            "--native-layout",
            "xe2_dpas",
            "--native-kernel-variant",
            "q6_scalar",
            "--native-split-policy",
            "b70_q6",
            "--output-dir",
            str(tmp_path / "valid-b70"),
        ]
    )
    assert b70.native_splits == {1: 32, 4: 8}

    with pytest.raises(SystemExit):
        correctness.parse_args(
            [
                *common,
                "--config-ref",
                f"path:{config}",
                "--native-layout",
                "xe2_dpas",
                "--native-kernel-variant",
                "q6_scalar",
                "--native-split-policy",
                "b70_q6",
                "--native-splits",
                "1=32",
                "--native-splits",
                "4=8",
                "--output-dir",
                str(tmp_path / "invalid-b70-conflict"),
            ]
        )
    with pytest.raises(SystemExit):
        correctness.parse_args(
            [
                *common,
                "--config-ref",
                f"path:{config}",
                "--native-kernel-variant",
                "q6_scalar",
                "--native-layout",
                "natural",
                "--native-split-policy",
                "fixed",
                "--output-dir",
                str(tmp_path / "invalid-natural-q6"),
            ]
        )


def test_tracked_checkout_identity_records_head_digest_and_dirtiness(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    source = repo / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)

    clean = correctness.tracked_checkout_identity(repo)
    source.write_text("value = 2\n", encoding="utf-8")
    dirty = correctness.tracked_checkout_identity(repo)

    assert len(clean["head"]) == 40
    assert len(clean["sha256"]) == 64
    assert clean["unexpected_changes"] == []
    assert dirty["unexpected_changes"] == [" M source.py"]


def test_source_identity_records_clean_runner_checkout_and_source_hashes(
    tmp_path: Path,
) -> None:
    def init_repo(path: Path) -> None:
        path.mkdir()
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(path), "config", "user.name", "Test"], check=True
        )

    def commit_all(path: Path) -> str:
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(["git", "-C", str(path), "commit", "-qm", "fixture"], check=True)
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    packaging = tmp_path / "packaging"
    init_repo(packaging)
    for relative in correctness.RUNNER_SOURCES:
        source = packaging / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {relative}\n", encoding="utf-8")
    runner_commit = commit_all(packaging)

    kernels = tmp_path / "kernels"
    init_repo(kernels)
    for relative in correctness.NATIVE_TEST_SOURCES:
        source = kernels / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(f"# {relative}\n", encoding="utf-8")
    kernels_commit = commit_all(kernels)

    packaging_commit = "1" * 40
    vllm_commit = "2" * 40
    config = tmp_path / "config"
    init_repo(config)
    lock = config / "modules/flake/nixos/server/flake.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(
        json.dumps(
            {
                "nodes": {
                    "vllm-xpu-release": {"locked": {"rev": packaging_commit}},
                    "vllm-xpu-unstable-src": {"locked": {"rev": vllm_commit}},
                    "vllm-xpu-kernels-unstable-src": {
                        "locked": {"rev": kernels_commit}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    commit_all(config)
    args = argparse.Namespace(
        output_dir=tmp_path / "output",
        config_repo=config,
        packaging_repo=packaging,
        kernels_repo=kernels,
        packaging_commit=packaging_commit,
        vllm_commit=vllm_commit,
        kernels_commit=kernels_commit,
    )
    args.output_dir.mkdir()

    identity = correctness.verify_source_identity(args)
    args.source_identity = identity

    assert identity["runner_checkout"]["head"] == runner_commit
    assert identity["runner_checkout"]["head"] != packaging_commit
    assert set(identity["runner_sources"]) == set(correctness.RUNNER_SOURCES)
    for reference in identity["runner_sources"].values():
        assert Path(reference["path"]).is_file()
        assert Path(reference["path"]).is_relative_to(args.output_dir)
        assert len(reference["sha256"]) == 64

    (packaging / correctness.RUNNER_SOURCES[0]).write_text(
        "# changed\n", encoding="utf-8"
    )
    with pytest.raises(correctness.CorrectnessError, match="runner checkout changed"):
        correctness.verify_packaging_identity(args)
