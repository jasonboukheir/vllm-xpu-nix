from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.kvarn_correctness_run as correctness


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
        flush_index_materialization="per_layer",
        native_frontend="reference",
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
