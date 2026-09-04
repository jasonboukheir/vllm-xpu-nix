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


def test_source_ownership_maps_package_to_vllm_and_both_libraries_to_kernels() -> None:
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
    host.require_source_ownership(
        package=package,
        base=base,
        flash=flash,
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
            expected_vllm_revision=VLLM_REVISION,
            expected_kernels_revision=KERNELS_REVISION,
        )
    with pytest.raises(
        host.HostLauncherError, match="attention library source mismatch"
    ):
        host.require_source_ownership(
            package=package,
            base=base,
            flash=dataclasses.replace(
                flash,
                derivation=flash.derivation.replace(
                    KERNELS_REVISION[:7], VLLM_REVISION[:7]
                ),
            ),
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
        output=tmp_path / "evidence.json",
        variants="baseline,q8_vector",
        splits="auto,24",
        contexts="4096,65023",
        batches="1,4",
        output_dtypes="fp16,bf16",
        warmup_rounds=12,
        sample_rounds=24,
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
    assert command[command.index("--variants") + 1] == "baseline,q8_vector"
    assert command[command.index("--splits") + 1] == "auto,24"
    assert command[command.index("--contexts") + 1] == "4096,65023"
    assert command[command.index("--batches") + 1] == "1,4"
    assert command[command.index("--output-dtypes") + 1] == "fp16,bf16"
    assert command[command.index("--warmup-rounds") + 1] == "12"
    assert command[command.index("--sample-rounds") + 1] == "24"
    assert (
        command[command.index("--expected-vllm-xpu-nix-revision") + 1]
        == PROJECT_REVISION
    )
    assert command[command.index("--expected-vllm-revision") + 1] == VLLM_REVISION
    assert command[command.index("--expected-kernels-revision") + 1] == KERNELS_REVISION
    assert command[command.index("--fixture-mode") + 1] == "matched-production"


def test_default_cli_is_the_matched_b70_factory_matrix() -> None:
    args = host.parse_args(
        [
            "result-kvarn-factory",
            PROJECT_REVISION,
            VLLM_REVISION,
            KERNELS_REVISION,
        ]
    )
    assert args.variants == host.DEFAULT_VARIANTS
    assert args.variants == (
        "q6_scalar,q6_vector,q6_cached_weights,q6_exact_rows,"
        "q6_cached_weights_exact_rows"
    )
    assert args.splits == host.DEFAULT_SPLITS
    assert args.contexts == host.DEFAULT_CONTEXTS
    assert args.batches == host.DEFAULT_BATCHES
    assert args.output_dtypes == host.DEFAULT_OUTPUT_DTYPES
    assert args.warmup_rounds == host.DEFAULT_WARMUP_ROUNDS
    assert args.sample_rounds == host.DEFAULT_SAMPLE_ROUNDS


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
    monkeypatch.setattr(host, "require_no_vllm_service", lambda _root: None)
    monkeypatch.setattr(
        host, "require_clean_repository", lambda _label, path, _runner: path.resolve()
    )
    monkeypatch.setattr(host, "require_repository_revision", lambda *_args: None)
    monkeypatch.setattr(host, "resolve_package_output", lambda _path: package)
    monkeypatch.setattr(host, "query_closure", lambda _path, _runner: [package])
    monkeypatch.setattr(host, "attest_output", lambda _path, _runner: package_build)
    libraries = iter((base_path, flash_path))
    monkeypatch.setattr(
        host, "discover_library", lambda _closure, _basename: next(libraries)
    )
    attestations = {base_path: base, flash_path: flash}
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
        splits=host.DEFAULT_SPLITS,
        contexts=host.DEFAULT_CONTEXTS,
        batches=host.DEFAULT_BATCHES,
        output_dtypes=host.DEFAULT_OUTPUT_DTYPES,
        warmup_rounds=host.DEFAULT_WARMUP_ROUNDS,
        sample_rounds=host.DEFAULT_SAMPLE_ROUNDS,
        expected_project_revision=PROJECT_REVISION,
        expected_vllm_revision=VLLM_REVISION,
        expected_kernels_revision=KERNELS_REVISION,
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
    assert captured[0][captured[0].index("--package-output") + 1] == str(package)
    assert ownership_calls == [
        {
            "package": package_build,
            "base": base,
            "flash": flash,
            "expected_vllm_revision": VLLM_REVISION,
            "expected_kernels_revision": KERNELS_REVISION,
        }
    ]
    assert captured[0][-1].endswith("factory-b70-20260903T000000Z.json")
