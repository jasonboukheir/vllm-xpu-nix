from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
from pathlib import Path
from typing import NoReturn

import pytest

from scripts import kvarn_factory_host as host

PROJECT_REVISION = "1" * 40
VLLM_REVISION = "2" * 40
KERNELS_REVISION = "3" * 40
ATTENTION_SOURCE_HASH = "4" * 32
ATTENTION_SOURCE_IDENTITY = host.FilteredSourceIdentity(
    scheme=host.FILTERED_SOURCE_SCHEME,
    store_hash=ATTENTION_SOURCE_HASH,
)


def _attention_expectation(output: Path) -> host.NativeAttentionExpectation:
    return host.NativeAttentionExpectation(
        output=output,
        source_identity=ATTENTION_SOURCE_IDENTITY,
        compatible_revision=KERNELS_REVISION,
    )


def test_closure_digest_matches_factory_convention() -> None:
    assert host.closure_digest(["/nix/store/z", "/nix/store/a", "/nix/store/z"]) == (
        hashlib.sha256(b"/nix/store/a\n/nix/store/z\n").hexdigest()
    )


def test_package_must_resolve_to_exact_store_output(tmp_path: Path) -> None:
    with pytest.raises(host.HostLauncherError, match="not in a Nix store"):
        host.resolve_package_output(tmp_path)


def test_discovery_fails_closed_on_absence_and_ambiguity(tmp_path: Path) -> None:
    first = tmp_path / "first/lib/python3.12/site-packages/one"
    second = tmp_path / "second/lib/python3.12/site-packages/two"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    with pytest.raises(host.HostLauncherError, match="found 0"):
        host.discover_library([tmp_path / "first"], host.BASE_LIBRARY)
    (first / host.BASE_LIBRARY).touch()
    (second / host.BASE_LIBRARY).touch()
    with pytest.raises(host.HostLauncherError, match="found 2"):
        host.discover_library(
            [tmp_path / "first", tmp_path / "second"], host.BASE_LIBRARY
        )


def test_native_attention_discovery_is_exact_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "lib").mkdir(parents=True)
    (second / "lib64").mkdir(parents=True)
    native = first / "lib" / host.NATIVE_ATTENTION_LIBRARY
    native.touch()
    monkeypatch.setattr(host, "store_output_from_resolved", lambda path: first)
    assert (
        host.discover_library(
            [first],
            host.NATIVE_ATTENTION_LIBRARY,
            relative_patterns=host.NATIVE_LIBRARY_PATTERNS,
        )
        == native.resolve()
    )
    (second / "lib64" / host.NATIVE_ATTENTION_LIBRARY).touch()
    with pytest.raises(host.HostLauncherError, match="found 2"):
        host.discover_library(
            [first, second],
            host.NATIVE_ATTENTION_LIBRARY,
            relative_patterns=host.NATIVE_LIBRARY_PATTERNS,
        )


def test_attestation_queries_true_deriver_and_hashes_sorted_closure() -> None:
    output = Path("/nix/store/" + "a" * 32 + "-kernels")
    library = output / "lib/python3.12/site-packages/kernels/_C.abi3.so"
    derivation = "/nix/store/" + "d" * 32 + "-kernels.drv"
    dependency = "/nix/store/" + "b" * 32 + "-dependency"
    commands: list[tuple[str, ...]] = []

    def command_runner(command: tuple[str, ...]) -> str:
        commands.append(command)
        if "--deriver" in command:
            return derivation
        return f"{output}\n{dependency}\n{output}\n"

    artifact = host.attest_library(library, command_runner)
    assert artifact.output == output
    assert artifact.derivation == derivation
    assert artifact.closure_sha256 == host.closure_digest(
        [str(output), dependency, str(output)]
    )
    assert commands == [
        ("nix-store", "-q", "--deriver", str(output)),
        ("nix-store", "-qR", str(output)),
    ]


def test_output_attestation_queries_package_deriver_and_closure() -> None:
    output = Path("/nix/store/" + "a" * 32 + "-vllm")
    derivation = "/nix/store/" + "d" * 32 + "-vllm.g" + VLLM_REVISION[:7] + ".drv"
    dependency = "/nix/store/" + "b" * 32 + "-dependency"

    def command_runner(command: tuple[str, ...]) -> str:
        if "--deriver" in command:
            return derivation
        return f"{dependency}\n{output}\n"

    build = host.attest_output(output, command_runner)
    assert build.output == output
    assert build.derivation == derivation
    assert build.closure_sha256 == host.closure_digest([dependency, str(output)])


def test_package_closure_rejects_non_store_entries() -> None:
    output = Path("/nix/store/" + "a" * 32 + "-vllm")
    with pytest.raises(host.HostLauncherError, match="not in a Nix store"):
        host.query_closure(
            output,
            lambda _command: f"{output}\n/usr/lib/not-attested\n",
        )


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["/nix/store/hash-vllm/bin/vllm", "serve", "model"], True),
        (["/nix/store/hash-vllm/bin/.vllm-wrapped", "serve", "model"], True),
        (["python", "-m", "vllm.entrypoints.openai.api_server"], True),
        (["python", "-m", "vllm.entrypoints.cli.main", "serve", "model"], True),
        (["VLLM::APIServer_0"], True),
        (["VLLM::EngineCore"], True),
        (["tenant-prefix::Worker_TP0"], True),
        (["custom::DPCoordinator"], True),
        (["python", "scripts/kvarn_factory_host.py", "result-vllm"], False),
        (["ninja", "vllm-kernel-target"], False),
    ],
)
def test_vllm_service_classification(argv: list[str], expected: bool) -> None:
    assert host.is_vllm_service_argv(argv) is expected


def _proc_entry(root: Path, pid: int, argv: list[str]) -> None:
    entry = root / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(b"\0".join(item.encode() for item in argv) + b"\0")


def test_running_service_is_detected_without_mutation(tmp_path: Path) -> None:
    _proc_entry(tmp_path, 10, ["python", "worker.py"])
    _proc_entry(tmp_path, 20, ["/nix/store/hash/bin/vllm", "serve", "model"])
    assert host.find_vllm_services(tmp_path, self_pid=99) == [20]
    with pytest.raises(host.HostLauncherError, match="needs about 4 GiB"):
        host.require_no_vllm_service(tmp_path)
    assert (tmp_path / "20/cmdline").exists()


def test_repository_check_rejects_dirty_tree(tmp_path: Path) -> None:
    responses = iter((str(tmp_path), "?? generated"))

    def command_runner(_command: tuple[str, ...]) -> str:
        return next(responses)

    with pytest.raises(host.HostLauncherError, match="repository is dirty"):
        host.require_clean_repository("repo", tmp_path, command_runner)


def test_repository_and_derivation_must_match_full_expected_head() -> None:
    head = "abcdef0123456789abcdef0123456789abcdef01"
    artifact = host.NixArtifact(
        library=Path("/nix/store/" + "a" * 32 + "-vllm/lib/_C.abi3.so"),
        output=Path("/nix/store/" + "a" * 32 + "-vllm"),
        derivation=(
            "/nix/store/"
            + "d" * 32
            + "-python3.12-vllm-xpu-0.28.0+unstable.2026.09.04.gabcdef0.drv"
        ),
        closure_sha256="0" * 64,
    )
    host.require_repository_revision(
        "vLLM", Path("/src/vllm"), head, lambda _command: head
    )
    with pytest.raises(host.HostLauncherError, match="source mismatch"):
        host.require_repository_revision(
            "vLLM", Path("/src/vllm"), "1" * 40, lambda _command: head
        )
    host.require_derivation_source("vLLM", artifact, head)
    with pytest.raises(host.HostLauncherError, match="source mismatch"):
        host.require_derivation_source(
            "vLLM",
            dataclasses.replace(
                artifact, derivation=artifact.derivation.replace("gabcdef0", "g1234567")
            ),
            head,
        )


def test_filtered_source_identity_replaces_revision_marker_for_split_library() -> None:
    output = Path("/nix/store/" + "n" * 32 + "-attention")
    derivation = (
        "/nix/store/"
        + "d" * 32
        + "-vllm-xpu-attn-0.1+src."
        + ATTENTION_SOURCE_HASH
        + ".drv"
    )
    expectation = host.validate_native_attention_expectation(
        output=output,
        source_scheme=host.FILTERED_SOURCE_SCHEME,
        source_store_hash=ATTENTION_SOURCE_HASH,
        compatible_revision=KERNELS_REVISION,
        expected_kernels_revision=KERNELS_REVISION,
    )
    artifact = host.NixArtifact(
        library=output / "lib/libattn_kernels_xe_2.so",
        output=output,
        derivation=derivation,
        closure_sha256="0" * 64,
    )
    host.require_native_attention_artifact("attention", artifact, expectation)
    with pytest.raises(host.HostLauncherError, match="source mismatch"):
        host.require_native_attention_artifact(
            "attention",
            dataclasses.replace(
                artifact,
                derivation=artifact.derivation.replace(ATTENTION_SOURCE_HASH, "5" * 32),
            ),
            expectation,
        )
    with pytest.raises(host.HostLauncherError, match="output mismatch"):
        host.require_native_attention_artifact(
            "attention",
            dataclasses.replace(
                artifact,
                output=Path("/nix/store/" + "m" * 32 + "-other-attention"),
            ),
            expectation,
        )
    with pytest.raises(host.HostLauncherError, match="unsupported"):
        host.validate_filtered_source_identity(
            scheme="git-revision", store_hash=ATTENTION_SOURCE_HASH
        )
    with pytest.raises(host.HostLauncherError, match="compatibility revision"):
        host.validate_native_attention_expectation(
            output=output,
            source_scheme=host.FILTERED_SOURCE_SCHEME,
            source_store_hash=ATTENTION_SOURCE_HASH,
            compatible_revision="6" * 40,
            expected_kernels_revision=KERNELS_REVISION,
        )


def test_source_ownership_maps_package_and_all_libraries_to_repositories() -> None:
    package = host.NixOutput(
        output=Path("/nix/store/" + "p" * 32 + "-vllm"),
        derivation=(
            "/nix/store/" + "d" * 32 + "-python-vllm.g" + VLLM_REVISION[:7] + ".drv"
        ),
        closure_sha256="0" * 64,
    )
    base = host.NixArtifact(
        library=Path("/nix/store/" + "b" * 32 + "-kernels/lib/_C.abi3.so"),
        output=Path("/nix/store/" + "b" * 32 + "-kernels"),
        derivation=(
            "/nix/store/"
            + "e" * 32
            + "-python-kernels.g"
            + KERNELS_REVISION[:7]
            + ".drv"
        ),
        closure_sha256="1" * 64,
    )
    flash = dataclasses.replace(
        base,
        library=Path("/nix/store/" + "b" * 32 + "-kernels/lib/_vllm_fa2_C.abi3.so"),
    )
    native_attention = dataclasses.replace(
        base,
        library=Path("/nix/store/" + "b" * 32 + "-kernels/lib/libattn_kernels_xe_2.so"),
        derivation=(
            "/nix/store/"
            + "f" * 32
            + "-vllm-xpu-attn-kernels-xe-2-0.1+src."
            + ATTENTION_SOURCE_HASH
            + ".drv"
        ),
    )
    native_expectation = _attention_expectation(native_attention.output)
    host.require_source_ownership(
        package=package,
        base=base,
        flash=flash,
        native_attention=native_attention,
        native_attention_expectation=native_expectation,
        expected_vllm_revision=VLLM_REVISION,
        expected_kernels_revision=KERNELS_REVISION,
    )
    with pytest.raises(host.HostLauncherError, match="vLLM package source mismatch"):
        host.require_source_ownership(
            package=dataclasses.replace(
                package,
                derivation=package.derivation.replace(
                    VLLM_REVISION[:7], KERNELS_REVISION[:7]
                ),
            ),
            base=base,
            flash=flash,
            native_attention=native_attention,
            native_attention_expectation=native_expectation,
            expected_vllm_revision=VLLM_REVISION,
            expected_kernels_revision=KERNELS_REVISION,
        )
    with pytest.raises(host.HostLauncherError, match="base library source mismatch"):
        host.require_source_ownership(
            package=package,
            base=dataclasses.replace(
                base,
                derivation=base.derivation.replace(
                    KERNELS_REVISION[:7], VLLM_REVISION[:7]
                ),
            ),
            flash=flash,
            native_attention=native_attention,
            native_attention_expectation=native_expectation,
            expected_vllm_revision=VLLM_REVISION,
            expected_kernels_revision=KERNELS_REVISION,
        )
    with pytest.raises(host.HostLauncherError, match="flash extension source mismatch"):
        host.require_source_ownership(
            package=package,
            base=base,
            flash=dataclasses.replace(
                flash,
                derivation=flash.derivation.replace(
                    KERNELS_REVISION[:7], VLLM_REVISION[:7]
                ),
            ),
            native_attention=native_attention,
            native_attention_expectation=native_expectation,
            expected_vllm_revision=VLLM_REVISION,
            expected_kernels_revision=KERNELS_REVISION,
        )
    with pytest.raises(
        host.HostLauncherError, match="native attention library source mismatch"
    ):
        host.require_source_ownership(
            package=package,
            base=base,
            flash=flash,
            native_attention=dataclasses.replace(
                native_attention,
                derivation=native_attention.derivation.replace(
                    ATTENTION_SOURCE_HASH, "5" * 32
                ),
            ),
            native_attention_expectation=native_expectation,
            expected_vllm_revision=VLLM_REVISION,
            expected_kernels_revision=KERNELS_REVISION,
        )


def test_timestamped_output_is_durable_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = dt.datetime(2026, 9, 3, 21, 4, 5, tzinfo=dt.UTC)
    output_dir = Path("/var/lib/kvarn-evidence")
    expected = output_dir / "factory-b70-20260903T210405Z.json"
    monkeypatch.setattr(Path, "exists", lambda path: path == expected)
    with pytest.raises(host.HostLauncherError, match="overwrite"):
        host.timestamped_output(output_dir, now=now)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    assert host.timestamped_output(output_dir, now=now) == expected
    with pytest.raises(host.HostLauncherError, match="outside /tmp"):
        host.timestamped_output(Path("/tmp/evidence"), now=now)


def test_runner_command_forwards_matrix_and_exact_attestations(tmp_path: Path) -> None:
    package = host.NixOutput(
        output=Path("/nix/store/package"),
        derivation="/nix/store/package.drv",
        closure_sha256="c" * 64,
    )
    base = host.NixArtifact(
        library=Path("/nix/store/base/lib/python3.12/site-packages/x/_C.abi3.so"),
        output=Path("/nix/store/base"),
        derivation="/nix/store/base.drv",
        closure_sha256="a" * 64,
    )
    flash = host.NixArtifact(
        library=Path(
            "/nix/store/flash/lib/python3.12/site-packages/x/_vllm_fa2_C.abi3.so"
        ),
        output=Path("/nix/store/flash"),
        derivation="/nix/store/flash.drv",
        closure_sha256="b" * 64,
    )
    native_attention = host.NixArtifact(
        library=Path(
            "/nix/store/" + "n" * 32 + "-attention/lib/libattn_kernels_xe_2.so"
        ),
        output=Path("/nix/store/" + "n" * 32 + "-attention"),
        derivation=(
            "/nix/store/"
            + "d" * 32
            + "-attention-0.1+src."
            + ATTENTION_SOURCE_HASH
            + ".drv"
        ),
        closure_sha256="d" * 64,
    )
    native_expectation = _attention_expectation(native_attention.output)
    repositories = host.RepositoryPaths(
        project=tmp_path / "nix",
        vllm=tmp_path / "vllm",
        kernels=tmp_path / "kernels",
    )
    command = host.build_runner_command(
        runner=repositories.project / "scripts/kvarn_factory_run.py",
        repositories=repositories,
        package=package,
        base=base,
        flash=flash,
        native_attention=native_attention,
        native_attention_expectation=native_expectation,
        output=tmp_path / "evidence.json",
        variants="baseline,q8_vector",
        flush_writer="native_xe2",
        prefill_store="hadamard_scatter",
        splits="auto,24",
        contexts="4096,65023",
        batches="1,4",
        output_dtypes="fp16,bf16",
        warmup_rounds=12,
        sample_rounds=24,
        service_layer_count=16,
        expected_project_revision=PROJECT_REVISION,
        expected_vllm_revision=VLLM_REVISION,
        expected_kernels_revision=KERNELS_REVISION,
    )
    assert command[0] == host.sys.executable
    assert command[command.index("--package-output") + 1] == str(package.output)
    assert command[command.index("--package-derivation") + 1] == package.derivation
    assert command[command.index("--package-closure-sha256") + 1] == "c" * 64
    assert command[command.index("--base-derivation") + 1] == base.derivation
    assert command[command.index("--flash-closure-sha256") + 1] == "b" * 64
    assert command[command.index("--native-attention-library") + 1] == str(
        native_attention.library
    )
    assert command[command.index("--native-attention-derivation") + 1] == (
        native_attention.derivation
    )
    assert command[command.index("--native-attention-closure-sha256") + 1] == "d" * 64
    assert command[command.index("--expected-native-attention-output") + 1] == str(
        native_attention.output
    )
    assert command[command.index("--expected-native-attention-derivation") + 1] == (
        native_attention.derivation
    )
    assert command[command.index("--native-attention-source-scheme") + 1] == (
        host.FILTERED_SOURCE_SCHEME
    )
    assert command[command.index("--native-attention-source-store-hash") + 1] == (
        ATTENTION_SOURCE_HASH
    )
    assert command[command.index("--native-attention-compatible-revision") + 1] == (
        KERNELS_REVISION
    )
    assert command[command.index("--variants") + 1] == "baseline,q8_vector"
    assert command[command.index("--flush-writer") + 1] == "native_xe2"
    assert command[command.index("--prefill-store") + 1] == "hadamard_scatter"
    assert command[command.index("--splits") + 1] == "auto,24"
    assert command[command.index("--contexts") + 1] == "4096,65023"
    assert command[command.index("--batches") + 1] == "1,4"
    assert command[command.index("--output-dtypes") + 1] == "fp16,bf16"
    assert command[command.index("--warmup-rounds") + 1] == "12"
    assert command[command.index("--sample-rounds") + 1] == "24"
    assert command[command.index("--service-layer-count") + 1] == "16"
    assert (
        command[command.index("--expected-vllm-xpu-nix-revision") + 1]
        == PROJECT_REVISION
    )
    assert command[command.index("--expected-vllm-revision") + 1] == VLLM_REVISION
    assert command[command.index("--expected-kernels-revision") + 1] == KERNELS_REVISION
    assert command[command.index("--fixture-mode") + 1] == "matched-production"


def test_default_cli_is_the_matched_b70_factory_matrix() -> None:
    argv = [
        "result-kvarn-factory",
        PROJECT_REVISION,
        VLLM_REVISION,
        KERNELS_REVISION,
        "--expected-native-attention-output",
        "/nix/store/" + "n" * 32 + "-attention",
        "--native-attention-source-scheme",
        host.FILTERED_SOURCE_SCHEME,
        "--native-attention-source-store-hash",
        ATTENTION_SOURCE_HASH,
        "--native-attention-compatible-revision",
        KERNELS_REVISION,
    ]
    args = host.parse_args(argv)
    assert args.variants == host.DEFAULT_VARIANTS
    assert args.variants == "all"
    assert args.flush_writer == host.DEFAULT_FLUSH_WRITER
    assert args.flush_writer == "reference"
    assert args.prefill_store == host.DEFAULT_PREFILL_STORE
    assert args.prefill_store == "reference"
    assert args.splits == host.DEFAULT_SPLITS
    assert args.contexts == host.DEFAULT_CONTEXTS
    assert args.batches == host.DEFAULT_BATCHES
    assert args.output_dtypes == host.DEFAULT_OUTPUT_DTYPES
    assert args.warmup_rounds == host.DEFAULT_WARMUP_ROUNDS
    assert args.sample_rounds == host.DEFAULT_SAMPLE_ROUNDS
    assert args.service_layer_count == host.DEFAULT_SERVICE_LAYER_COUNT

    selected = host.parse_args(
        [
            *argv,
            "--flush-writer",
            "native_xe2",
            "--prefill-store",
            "hadamard_scatter",
            "--service-layer-count",
            "16",
        ]
    )
    assert selected.flush_writer == "native_xe2"
    assert selected.prefill_store == "hadamard_scatter"
    assert selected.service_layer_count == 16

    sinkhorn = host.parse_args(
        [*argv, "--flush-writer", "sinkhorn_pack_xe2"]
    )
    assert sinkhorn.flush_writer == "sinkhorn_pack_xe2"


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--flush-writer", "native"),
        ("--prefill-store", "scatter"),
        ("--service-layer-count", "4"),
    ],
)
def test_cli_rejects_unknown_writer_selectors(option: str, value: str) -> None:
    with pytest.raises(SystemExit, match="2"):
        host.parse_args(
            [
                "result-kvarn-factory",
                PROJECT_REVISION,
                VLLM_REVISION,
                KERNELS_REVISION,
                "--expected-native-attention-output",
                "/nix/store/" + "n" * 32 + "-attention",
                "--native-attention-source-scheme",
                host.FILTERED_SOURCE_SCHEME,
                "--native-attention-source-store-hash",
                ATTENTION_SOURCE_HASH,
                "--native-attention-compatible-revision",
                KERNELS_REVISION,
                option,
                value,
            ]
        )


def test_launch_executes_once_with_resolved_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "scripts/kvarn_factory_run.py").touch()
    vllm = tmp_path / "vllm"
    kernels = tmp_path / "kernels"
    vllm.mkdir()
    kernels.mkdir()
    package = Path("/nix/store/" + "p" * 32 + "-vllm")
    package_build = host.NixOutput(
        package,
        "/nix/store/" + "a" * 32 + "-vllm.g" + VLLM_REVISION[:7] + ".drv",
        "0" * 64,
    )
    base_path = Path("/nix/store/" + "b" * 32 + "-base/lib/_C.abi3.so")
    flash_path = Path("/nix/store/" + "f" * 32 + "-flash/lib/_vllm_fa2_C.abi3.so")
    native_attention_path = Path(
        "/nix/store/" + "n" * 32 + "-attention/lib/libattn_kernels_xe_2.so"
    )
    base = host.NixArtifact(
        base_path,
        host.store_output_from_resolved(base_path),
        "/nix/store/" + "d" * 32 + "-base.drv",
        "1" * 64,
    )
    flash = host.NixArtifact(
        flash_path,
        host.store_output_from_resolved(flash_path),
        "/nix/store/" + "e" * 32 + "-flash.drv",
        "2" * 64,
    )
    native_attention = host.NixArtifact(
        native_attention_path,
        host.store_output_from_resolved(native_attention_path),
        "/nix/store/"
        + "c" * 32
        + "-attention-0.1+src."
        + ATTENTION_SOURCE_HASH
        + ".drv",
        "3" * 64,
    )
    monkeypatch.setattr(host, "require_no_vllm_service", lambda _root: None)
    monkeypatch.setattr(
        host, "require_clean_repository", lambda _label, path, _runner: path.resolve()
    )
    monkeypatch.setattr(host, "require_repository_revision", lambda *_args: None)
    monkeypatch.setattr(host, "resolve_package_output", lambda _path: package)
    monkeypatch.setattr(host, "query_closure", lambda _path, _runner: [package])
    monkeypatch.setattr(host, "attest_output", lambda _path, _runner: package_build)
    libraries = iter((base_path, flash_path, native_attention_path))
    monkeypatch.setattr(
        host,
        "discover_library",
        lambda _closure, _basename, **_kwargs: next(libraries),
    )
    attestations = {
        base_path: base,
        flash_path: flash,
        native_attention_path: native_attention,
    }
    monkeypatch.setattr(
        host, "attest_library", lambda library, _runner: attestations[library]
    )
    ownership_calls: list[dict] = []
    monkeypatch.setattr(
        host,
        "require_source_ownership",
        lambda **kwargs: ownership_calls.append(kwargs),
    )
    monkeypatch.setattr(
        host,
        "timestamped_output",
        lambda _directory, now: tmp_path / "factory-b70-20260903T000000Z.json",
    )
    captured: list[list[str]] = []

    def executor(command: list[str]) -> NoReturn:
        captured.append(command)
        raise SystemExit(0)

    args = argparse.Namespace(
        package=tmp_path / "result",
        vllm_xpu_nix_repo=project,
        vllm_repo=vllm,
        kernels_repo=kernels,
        output_dir=tmp_path / "results",
        variants=host.DEFAULT_VARIANTS,
        flush_writer="native_xe2",
        prefill_store="hadamard_scatter",
        splits=host.DEFAULT_SPLITS,
        contexts=host.DEFAULT_CONTEXTS,
        batches=host.DEFAULT_BATCHES,
        output_dtypes=host.DEFAULT_OUTPUT_DTYPES,
        warmup_rounds=host.DEFAULT_WARMUP_ROUNDS,
        sample_rounds=host.DEFAULT_SAMPLE_ROUNDS,
        service_layer_count=host.DEFAULT_SERVICE_LAYER_COUNT,
        expected_project_revision=PROJECT_REVISION,
        expected_vllm_revision=VLLM_REVISION,
        expected_kernels_revision=KERNELS_REVISION,
        expected_native_attention_output=native_attention.output,
        native_attention_source_scheme=host.FILTERED_SOURCE_SCHEME,
        native_attention_source_store_hash=ATTENTION_SOURCE_HASH,
        native_attention_compatible_revision=KERNELS_REVISION,
    )
    with pytest.raises(SystemExit, match="0"):
        host.launch(
            args,
            command_runner=lambda _command: "",
            executor=executor,
            proc_root=tmp_path,
            now=dt.datetime(2026, 9, 3, tzinfo=dt.UTC),
        )
    assert len(captured) == 1
    assert str(base_path) in captured[0]
    assert str(flash_path) in captured[0]
    assert str(native_attention_path) in captured[0]
    assert captured[0][captured[0].index("--package-output") + 1] == str(package)
    assert captured[0][captured[0].index("--flush-writer") + 1] == "native_xe2"
    assert captured[0][captured[0].index("--prefill-store") + 1] == "hadamard_scatter"
    assert ownership_calls == [
        {
            "package": package_build,
            "base": base,
            "flash": flash,
            "native_attention": native_attention,
            "native_attention_expectation": _attention_expectation(
                native_attention.output
            ),
            "expected_vllm_revision": VLLM_REVISION,
            "expected_kernels_revision": KERNELS_REVISION,
        }
    ]
    assert captured[0][-1].endswith("factory-b70-20260903T000000Z.json")
