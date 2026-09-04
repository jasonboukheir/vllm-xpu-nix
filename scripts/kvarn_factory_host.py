#!/usr/bin/env python3
"""Launch the Kvarn B70 factory from one realized vLLM Nix output.

The launcher discovers the two Python extensions and the Xe2 attention kernel
library from the package's realized closure, records their true Nix
derivations and closure digests, and then
replaces itself with ``kvarn_factory_run.py``.  It never starts or stops a
service and refuses to compete with an already-running vLLM service for VRAM.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

BASE_LIBRARY = "_C.abi3.so"
FLASH_LIBRARY = "_vllm_fa2_C.abi3.so"
NATIVE_ATTENTION_LIBRARY = "libattn_kernels_xe_2.so"
PYTHON_EXTENSION_PATTERNS = (
    "lib/python*/site-packages/*/{basename}",
    "lib64/python*/site-packages/*/{basename}",
)
NATIVE_LIBRARY_PATTERNS = ("lib/{basename}", "lib64/{basename}")
FILTERED_SOURCE_SCHEME = "nix-filtered-source-store-hash-v1"
NIX_STORE_HASH = re.compile(r"^[0-9abcdfghijklmnpqrsvwxyz]{32}$")
# Resolve ``all`` in the runner so every layout-compatible candidate compiled
# into the shared library automatically participates in the default sweep.
DEFAULT_VARIANTS = "all"
FLUSH_WRITER_VARIANTS = ("reference", "native_xe2", "sinkhorn_pack_xe2")
PREFILL_STORE_VARIANTS = ("reference", "hadamard_scatter")
DEFAULT_FLUSH_WRITER = "reference"
DEFAULT_PREFILL_STORE = "reference"
DEFAULT_SPLITS = "8,32"
DEFAULT_CONTEXTS = "4096,16384,65023"
DEFAULT_BATCHES = "1,4"
DEFAULT_OUTPUT_DTYPES = "bf16"
DEFAULT_WARMUP_ROUNDS = 16
DEFAULT_SAMPLE_ROUNDS = 20
VALID_SERVICE_LAYER_COUNTS = (1, 16)
DEFAULT_SERVICE_LAYER_COUNT = 1
STORE_NAME = re.compile(r"^[a-z0-9]{32}-.+")
DERIVATION = re.compile(r"^/nix/store/[a-z0-9]{32}-.+\.drv$")
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class HostLauncherError(RuntimeError):
    """Raised when host provenance or exclusivity cannot be established."""


@dataclasses.dataclass(frozen=True)
class NixArtifact:
    library: Path
    output: Path
    derivation: str
    closure_sha256: str


@dataclasses.dataclass(frozen=True)
class NixOutput:
    output: Path
    derivation: str
    closure_sha256: str


@dataclasses.dataclass(frozen=True)
class RepositoryPaths:
    project: Path
    vllm: Path
    kernels: Path


@dataclasses.dataclass(frozen=True)
class FilteredSourceIdentity:
    scheme: str
    store_hash: str


@dataclasses.dataclass(frozen=True)
class NativeAttentionExpectation:
    output: Path
    source_identity: FilteredSourceIdentity
    compatible_revision: str


CommandRunner = Callable[[Sequence[str]], str]
Executor = Callable[[Sequence[str]], NoReturn]


def _run(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise HostLauncherError(
            f"command failed: {' '.join(command)}: {error}"
        ) from error
    return completed.stdout.strip()


def closure_digest(paths: Sequence[str]) -> str:
    """Match kvarn_factory_run.py's sorted, unique, newline-terminated hash."""
    canonical = "\n".join(sorted(set(paths))) + "\n"
    return hashlib.sha256(canonical.encode()).hexdigest()


def store_output_from_resolved(path: Path) -> Path:
    parts = path.parts
    if (
        len(parts) < 4
        or parts[:3] != ("/", "nix", "store")
        or not STORE_NAME.fullmatch(parts[3])
    ):
        raise HostLauncherError(f"path is not in a Nix store output: {path}")
    return Path(*parts[:4])


def resolve_package_output(path: Path) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise HostLauncherError(
            f"cannot resolve vLLM package {path}: {error}"
        ) from error
    output = store_output_from_resolved(resolved)
    if resolved != output or not output.is_dir():
        raise HostLauncherError(
            f"vLLM package must resolve to a Nix store output directory: {resolved}"
        )
    return output


def query_closure(output: Path, command_runner: CommandRunner = _run) -> list[Path]:
    raw_paths = command_runner(("nix-store", "-qR", str(output))).splitlines()
    if not raw_paths:
        raise HostLauncherError(f"Nix returned an empty closure for {output}")
    closure: list[Path] = []
    for raw_path in sorted(set(raw_paths)):
        candidate = Path(raw_path)
        if store_output_from_resolved(candidate) != candidate:
            raise HostLauncherError(
                f"Nix closure contains a non-output path: {candidate}"
            )
        closure.append(candidate)
    if output not in closure:
        raise HostLauncherError(
            f"realized package output is absent from its own closure: {output}"
        )
    return closure


def discover_library(
    closure: Sequence[Path],
    basename: str,
    *,
    relative_patterns: Sequence[str] = PYTHON_EXTENSION_PATTERNS,
) -> Path:
    candidates: set[Path] = set()
    try:
        for output in closure:
            for template in relative_patterns:
                pattern = template.format(basename=basename)
                for match in output.glob(pattern):
                    resolved = match.resolve(strict=True)
                    if resolved.is_file():
                        candidates.add(resolved)
    except OSError as error:
        raise HostLauncherError(
            f"cannot inspect the realized Nix closure for {basename}: {error}"
        ) from error
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in sorted(candidates)) or "none"
        raise HostLauncherError(
            f"expected exactly one {basename} in the vLLM closure, found "
            f"{len(candidates)}: {rendered}"
        )
    library = next(iter(candidates))
    library_output = store_output_from_resolved(library)
    if library_output not in closure:
        raise HostLauncherError(
            f"resolved {basename} escapes the attested vLLM closure: {library}"
        )
    return library


def attest_output(output: Path, command_runner: CommandRunner = _run) -> NixOutput:
    if store_output_from_resolved(output) != output:
        raise HostLauncherError(f"path is not an exact Nix store output: {output}")
    derivation = command_runner(("nix-store", "-q", "--deriver", str(output)))
    if not DERIVATION.fullmatch(derivation):
        raise HostLauncherError(
            f"Nix returned no unique derivation for {output}: {derivation!r}"
        )
    closure = command_runner(("nix-store", "-qR", str(output))).splitlines()
    if not closure:
        raise HostLauncherError(f"Nix returned an empty closure for {output}")
    for item in closure:
        candidate = Path(item)
        if store_output_from_resolved(candidate) != candidate:
            raise HostLauncherError(
                f"artifact closure contains a non-output path: {candidate}"
            )
    return NixOutput(
        output=output,
        derivation=derivation,
        closure_sha256=closure_digest(closure),
    )


def attest_library(library: Path, command_runner: CommandRunner = _run) -> NixArtifact:
    build = attest_output(store_output_from_resolved(library), command_runner)
    return NixArtifact(
        library=library,
        output=build.output,
        derivation=build.derivation,
        closure_sha256=build.closure_sha256,
    )


def is_vllm_service_argv(argv: Sequence[str]) -> bool:
    if not argv:
        return False
    process_roles = ("::APIServer", "::EngineCore", "::Worker", "::DPCoordinator")
    if any(role in token for token in argv for role in process_roles):
        return True
    executable_names = {Path(token).name for token in argv}
    if "serve" in argv and executable_names.intersection({"vllm", ".vllm-wrapped"}):
        return True
    modules = {
        "vllm.entrypoints.openai.api_server",
        "vllm.entrypoints.api_server",
        "vllm.entrypoints.cli.main",
    }
    return any(token in modules for token in argv)


def find_vllm_services(
    proc_root: Path = Path("/proc"), *, self_pid: int | None = None
) -> list[int]:
    own_pid = os.getpid() if self_pid is None else self_pid
    matches: list[int] = []
    inaccessible: list[int] = []
    try:
        entries = list(proc_root.iterdir())
    except OSError as error:
        raise HostLauncherError(f"cannot inspect {proc_root}: {error}") from error
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == own_pid:
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except FileNotFoundError:
            continue
        except (OSError, PermissionError):
            inaccessible.append(int(entry.name))
            continue
        argv = [part.decode(errors="replace") for part in raw.split(b"\0") if part]
        if is_vllm_service_argv(argv):
            matches.append(int(entry.name))
    if inaccessible:
        preview = ", ".join(str(pid) for pid in sorted(inaccessible)[:8])
        raise HostLauncherError(
            "cannot prove vLLM is stopped because process command lines are "
            f"inaccessible (PIDs {preview})"
        )
    return sorted(matches)


def require_no_vllm_service(proc_root: Path = Path("/proc")) -> None:
    running = find_vllm_services(proc_root)
    if running:
        rendered = ", ".join(str(pid) for pid in running)
        raise HostLauncherError(
            "refusing to run while a vLLM service is active "
            f"(PIDs {rendered}); the matched corpus needs about 4 GiB of free "
            "VRAM. Stop it yourself, then rerun this launcher."
        )


def require_clean_repository(
    label: str, path: Path, runner: CommandRunner = _run
) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise HostLauncherError(
            f"cannot resolve {label} repository {path}: {error}"
        ) from error
    top_level = Path(
        runner(("git", "-C", str(resolved), "rev-parse", "--show-toplevel"))
    ).resolve(strict=True)
    if top_level != resolved:
        raise HostLauncherError(
            f"{label} repository path is not its Git top level: {resolved}"
        )
    status = runner(
        (
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    )
    if status:
        raise HostLauncherError(f"{label} repository is dirty: {resolved}")
    return resolved


def require_repository_revision(
    label: str,
    repository: Path,
    expected_revision: str,
    runner: CommandRunner = _run,
) -> None:
    if not GIT_COMMIT.fullmatch(expected_revision):
        raise HostLauncherError(
            f"{label} expected revision is not a full Git commit: {expected_revision!r}"
        )
    head = runner(("git", "-C", str(repository), "rev-parse", "HEAD"))
    if not GIT_COMMIT.fullmatch(head):
        raise HostLauncherError(
            f"{label} repository returned an invalid Git commit: {head!r}"
        )
    if head != expected_revision:
        raise HostLauncherError(
            f"{label} source mismatch: expected {expected_revision}, got {head}"
        )


def require_derivation_source(
    label: str,
    artifact: NixArtifact | NixOutput,
    expected_revision: str,
) -> None:
    if not GIT_COMMIT.fullmatch(expected_revision):
        raise HostLauncherError(
            f"{label} expected revision is not a full Git commit: {expected_revision!r}"
        )
    marker = f".g{expected_revision[:7]}"
    if marker not in Path(artifact.derivation).name:
        raise HostLauncherError(
            f"{label} source mismatch: expected revision {expected_revision} "
            "is not stamped "
            f"into artifact derivation {artifact.derivation}"
        )


def validate_filtered_source_identity(
    *, scheme: str, store_hash: str
) -> FilteredSourceIdentity:
    if scheme != FILTERED_SOURCE_SCHEME:
        raise HostLauncherError(
            f"unsupported native attention source identity scheme: {scheme!r}"
        )
    if not NIX_STORE_HASH.fullmatch(store_hash):
        raise HostLauncherError(
            "native attention filtered source store hash must be 32 Nix-base32 "
            "characters"
        )
    return FilteredSourceIdentity(scheme=scheme, store_hash=store_hash)


def validate_native_attention_expectation(
    *,
    output: Path,
    source_scheme: str,
    source_store_hash: str,
    compatible_revision: str,
    expected_kernels_revision: str,
) -> NativeAttentionExpectation:
    if store_output_from_resolved(output) != output:
        raise HostLauncherError(
            f"expected native attention output is not exact: {output}"
        )
    if (
        not GIT_COMMIT.fullmatch(compatible_revision)
        or compatible_revision != expected_kernels_revision
    ):
        raise HostLauncherError(
            "native attention compatibility revision does not match the expected "
            f"kernel checkout: {compatible_revision!r} != {expected_kernels_revision!r}"
        )
    source_identity = validate_filtered_source_identity(
        scheme=source_scheme, store_hash=source_store_hash
    )
    return NativeAttentionExpectation(
        output=output,
        source_identity=source_identity,
        compatible_revision=compatible_revision,
    )


def require_native_attention_artifact(
    label: str,
    artifact: NixArtifact,
    expectation: NativeAttentionExpectation,
) -> None:
    if artifact.output != expectation.output:
        raise HostLauncherError(
            f"{label} output mismatch: expected {expectation.output}, "
            f"found {artifact.output}"
        )
    marker = f"+src.{expectation.source_identity.store_hash}"
    if marker not in Path(artifact.derivation).name:
        raise HostLauncherError(
            f"{label} source mismatch: expected filtered source identity "
            f"{expectation.source_identity.scheme}:"
            f"{expectation.source_identity.store_hash} is not stamped into "
            f"artifact derivation {artifact.derivation}"
        )


def require_source_ownership(
    *,
    package: NixOutput,
    base: NixArtifact,
    flash: NixArtifact,
    native_attention: NixArtifact,
    native_attention_expectation: NativeAttentionExpectation,
    expected_vllm_revision: str,
    expected_kernels_revision: str,
) -> None:
    require_derivation_source("vLLM package", package, expected_vllm_revision)
    require_derivation_source(
        "vllm-xpu-kernels base library", base, expected_kernels_revision
    )
    require_derivation_source(
        "vllm-xpu-kernels flash extension", flash, expected_kernels_revision
    )
    require_native_attention_artifact(
        "vllm-xpu-kernels native attention library",
        native_attention,
        native_attention_expectation,
    )


def timestamped_output(output_dir: Path, *, now: dt.datetime | None = None) -> Path:
    resolved_dir = output_dir.expanduser().resolve()
    if resolved_dir.is_relative_to(Path("/tmp")):
        raise HostLauncherError(
            f"factory output must be durable and outside /tmp: {resolved_dir}"
        )
    instant = now or dt.datetime.now(tz=dt.UTC)
    if instant.tzinfo is None:
        raise HostLauncherError("timestamp must be timezone-aware")
    stamp = instant.astimezone(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    output = resolved_dir / f"factory-b70-{stamp}.json"
    if output.exists():
        raise HostLauncherError(f"refusing to overwrite factory evidence: {output}")
    return output


def build_runner_command(
    *,
    runner: Path,
    repositories: RepositoryPaths,
    package: NixOutput,
    base: NixArtifact,
    flash: NixArtifact,
    native_attention: NixArtifact,
    native_attention_expectation: NativeAttentionExpectation,
    output: Path,
    variants: str,
    splits: str,
    contexts: str,
    batches: str,
    expected_project_revision: str,
    expected_vllm_revision: str,
    expected_kernels_revision: str,
    flush_writer: str = DEFAULT_FLUSH_WRITER,
    prefill_store: str = DEFAULT_PREFILL_STORE,
    output_dtypes: str = DEFAULT_OUTPUT_DTYPES,
    warmup_rounds: int = DEFAULT_WARMUP_ROUNDS,
    sample_rounds: int = DEFAULT_SAMPLE_ROUNDS,
    service_layer_count: int = DEFAULT_SERVICE_LAYER_COUNT,
) -> list[str]:
    return [
        sys.executable,
        str(runner),
        "--package-output",
        str(package.output),
        "--package-derivation",
        package.derivation,
        "--package-closure-sha256",
        package.closure_sha256,
        "--base-library",
        str(base.library),
        "--flash-library",
        str(flash.library),
        "--base-derivation",
        base.derivation,
        "--base-closure-sha256",
        base.closure_sha256,
        "--flash-derivation",
        flash.derivation,
        "--flash-closure-sha256",
        flash.closure_sha256,
        "--native-attention-library",
        str(native_attention.library),
        "--native-attention-derivation",
        native_attention.derivation,
        "--native-attention-closure-sha256",
        native_attention.closure_sha256,
        "--expected-native-attention-output",
        str(native_attention_expectation.output),
        "--expected-native-attention-derivation",
        # Content-addressed derivations have an unresolved path at flake
        # evaluation time and a different resolved path after realization.
        # The expected output above is rewritten to its exact realized store
        # path; attest and forward that output's true runtime deriver here.
        native_attention.derivation,
        "--native-attention-source-scheme",
        native_attention_expectation.source_identity.scheme,
        "--native-attention-source-store-hash",
        native_attention_expectation.source_identity.store_hash,
        "--native-attention-compatible-revision",
        native_attention_expectation.compatible_revision,
        "--vllm-xpu-nix-repo",
        str(repositories.project),
        "--vllm-repo",
        str(repositories.vllm),
        "--kernels-repo",
        str(repositories.kernels),
        "--expected-vllm-xpu-nix-revision",
        expected_project_revision,
        "--expected-vllm-revision",
        expected_vllm_revision,
        "--expected-kernels-revision",
        expected_kernels_revision,
        "--variants",
        variants,
        "--flush-writer",
        flush_writer,
        "--prefill-store",
        prefill_store,
        "--splits",
        splits,
        "--contexts",
        contexts,
        "--batches",
        batches,
        "--output-dtypes",
        output_dtypes,
        "--warmup-rounds",
        str(warmup_rounds),
        "--sample-rounds",
        str(sample_rounds),
        "--service-layer-count",
        str(service_layer_count),
        "--fixture-mode",
        "matched-production",
        "--output",
        str(output),
    ]


def _exec(command: Sequence[str]) -> NoReturn:
    os.execv(command[0], list(command))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # ``nix run .#kvarn-factory`` executes this file from the Nix store while
    # preserving the caller's working directory. The factory is intentionally
    # run from the clean vllm-xpu-nix checkout whose state enters provenance.
    project = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "package",
        type=Path,
        help="realized vLLM package output or a result symlink to it",
    )
    parser.add_argument("expected_project_revision")
    parser.add_argument("expected_vllm_revision")
    parser.add_argument("expected_kernels_revision")
    parser.add_argument("--expected-native-attention-output", type=Path, required=True)
    parser.add_argument("--native-attention-source-scheme", required=True)
    parser.add_argument("--native-attention-source-store-hash", required=True)
    parser.add_argument("--native-attention-compatible-revision", required=True)
    parser.add_argument("--vllm-xpu-nix-repo", type=Path, default=project)
    parser.add_argument("--vllm-repo", type=Path, default=project.parent / "vllm")
    parser.add_argument(
        "--kernels-repo", type=Path, default=project.parent / "vllm-xpu-kernels"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=project / "benchmark-results/kvarn"
    )
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument(
        "--flush-writer",
        choices=FLUSH_WRITER_VARIANTS,
        default=DEFAULT_FLUSH_WRITER,
        help="full-page writer whose direct-op kill suite must pass",
    )
    parser.add_argument(
        "--prefill-store",
        choices=PREFILL_STORE_VARIANTS,
        default=DEFAULT_PREFILL_STORE,
        help="multi-token prefill store whose direct-op kill suite must pass",
    )
    parser.add_argument("--splits", default=DEFAULT_SPLITS)
    parser.add_argument("--contexts", default=DEFAULT_CONTEXTS)
    parser.add_argument("--batches", default=DEFAULT_BATCHES)
    parser.add_argument("--output-dtypes", default=DEFAULT_OUTPUT_DTYPES)
    parser.add_argument("--warmup-rounds", type=int, default=DEFAULT_WARMUP_ROUNDS)
    parser.add_argument("--sample-rounds", type=int, default=DEFAULT_SAMPLE_ROUNDS)
    parser.add_argument(
        "--service-layer-count",
        type=int,
        choices=VALID_SERVICE_LAYER_COUNTS,
        default=DEFAULT_SERVICE_LAYER_COUNT,
    )
    return parser.parse_args(argv)


def launch(
    args: argparse.Namespace,
    *,
    command_runner: CommandRunner = _run,
    executor: Executor = _exec,
    proc_root: Path = Path("/proc"),
    now: dt.datetime | None = None,
) -> NoReturn:
    require_no_vllm_service(proc_root)
    project = require_clean_repository(
        "vllm-xpu-nix", args.vllm_xpu_nix_repo, command_runner
    )
    repositories = RepositoryPaths(
        project=project,
        vllm=require_clean_repository("vLLM", args.vllm_repo, command_runner),
        kernels=require_clean_repository(
            "vllm-xpu-kernels", args.kernels_repo, command_runner
        ),
    )
    require_repository_revision(
        "vllm-xpu-nix",
        repositories.project,
        args.expected_project_revision,
        command_runner,
    )
    require_repository_revision(
        "vLLM", repositories.vllm, args.expected_vllm_revision, command_runner
    )
    require_repository_revision(
        "vllm-xpu-kernels",
        repositories.kernels,
        args.expected_kernels_revision,
        command_runner,
    )
    package_output = resolve_package_output(args.package)
    package_closure = query_closure(package_output, command_runner)
    package = attest_output(package_output, command_runner)
    base = attest_library(
        discover_library(package_closure, BASE_LIBRARY), command_runner
    )
    flash = attest_library(
        discover_library(package_closure, FLASH_LIBRARY), command_runner
    )
    native_attention = attest_library(
        discover_library(
            package_closure,
            NATIVE_ATTENTION_LIBRARY,
            relative_patterns=NATIVE_LIBRARY_PATTERNS,
        ),
        command_runner,
    )
    native_attention_expectation = validate_native_attention_expectation(
        output=args.expected_native_attention_output,
        source_scheme=args.native_attention_source_scheme,
        source_store_hash=args.native_attention_source_store_hash,
        compatible_revision=args.native_attention_compatible_revision,
        expected_kernels_revision=args.expected_kernels_revision,
    )
    require_source_ownership(
        package=package,
        base=base,
        flash=flash,
        native_attention=native_attention,
        native_attention_expectation=native_attention_expectation,
        expected_vllm_revision=args.expected_vllm_revision,
        expected_kernels_revision=args.expected_kernels_revision,
    )
    output = timestamped_output(args.output_dir, now=now)
    runner = project / "scripts/kvarn_factory_run.py"
    if not runner.is_file():
        raise HostLauncherError(f"factory runner is missing: {runner}")
    command = build_runner_command(
        runner=runner,
        repositories=repositories,
        package=package,
        base=base,
        flash=flash,
        native_attention=native_attention,
        native_attention_expectation=native_attention_expectation,
        output=output,
        variants=args.variants,
        flush_writer=args.flush_writer,
        prefill_store=args.prefill_store,
        splits=args.splits,
        contexts=args.contexts,
        batches=args.batches,
        output_dtypes=args.output_dtypes,
        warmup_rounds=args.warmup_rounds,
        sample_rounds=args.sample_rounds,
        service_layer_count=args.service_layer_count,
        expected_project_revision=args.expected_project_revision,
        expected_vllm_revision=args.expected_vllm_revision,
        expected_kernels_revision=args.expected_kernels_revision,
    )
    print(f"Launching matched B70 factory evidence: {output}", flush=True)
    executor(command)
    raise HostLauncherError("factory runner unexpectedly returned from exec")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        launch(parse_args(argv))
    except HostLauncherError as error:
        print(f"kvarn factory launcher: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
