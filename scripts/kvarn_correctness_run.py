#!/usr/bin/env python3
"""Run and seal the native Kvarn correctness qualification on Brutus.

The runner deliberately uses one completion for each near-262K service start.
The shorter service phases exercise replay, restart, cancellation/reuse, and
concurrent B4 isolation.  A passing manifest is written only after every
primitive and service artifact has been finalized and re-hashed.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

try:
    from scripts import kvarn_perf_run as perf
    from scripts.kvarn_perf_gate import (
        COMBINED_LIBRARY_VARIANT_MATRIX,
        GateError,
        validate_correctness_gate_evidence,
        validate_factory_qualification,
    )
    from scripts.kvarn_service_gate import (
        cancel_stream,
        completion,
        result_quality_failures,
        run_concurrent_wave,
        wait_for_idle,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    import kvarn_perf_run as perf
    from kvarn_perf_gate import (
        COMBINED_LIBRARY_VARIANT_MATRIX,
        GateError,
        validate_correctness_gate_evidence,
        validate_factory_qualification,
    )
    from kvarn_service_gate import (
        cancel_stream,
        completion,
        result_quality_failures,
        run_concurrent_wave,
        wait_for_idle,
    )


DEFAULT_MODEL = (
    "jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound"
)
DEFAULT_MODEL_REVISION = "6b0622f4354481d5d04577d48ba0db844efc1330"
DEFAULT_FIXTURE_SHA256 = (
    "51a64c614f11eb0bee363fc2142afd1d03086c0efc165c66a316b6f3b5f3f7bd"
)
DEFAULT_OUTPUT_TOKENS = 512
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_PREFILL_WINDOW_BLOCKS = perf.DEFAULT_PREFILL_WINDOW_BLOCKS
NEAR_262K_PROMPT_TOKENS = 261631
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
NATIVE_LIBRARY_MODULE = "vllm_xpu_kernels._vllm_fa2_C"
NATIVE_TEST = "tests/flash_attn/test_kvarn_decode_xpu.py"
NATIVE_TEST_SOURCES = (
    NATIVE_TEST,
    "benchmark/check_kvarn_decode.py",
    "benchmark/kvarn_utils.py",
)
RUNNER_SOURCES = (
    "scripts/kvarn_correctness_run.py",
    "scripts/kvarn_perf_gate.py",
    "scripts/kvarn_perf_run.py",
    "scripts/kvarn_scan_engine_log.py",
    "scripts/kvarn_service_gate.py",
)
REQUIRED_INACTIVE_UNITS = (
    "vllm-xpu-chat.service",
    "vllm-xpu-embedding.service",
)


class CorrectnessError(RuntimeError):
    """Raised when evidence cannot support a passing correctness manifest."""


@dataclasses.dataclass(frozen=True)
class ServiceSpec:
    name: str
    launcher: str
    native: bool
    batch: int
    max_model_len: int


SERVICE_PLAN = (
    ServiceSpec(
        "native-65k-b1-first",
        "vllm-xpu-brutus-kvarn-native-b1",
        True,
        1,
        65536,
    ),
    ServiceSpec(
        "native-65k-b1-restart",
        "vllm-xpu-brutus-kvarn-native-b1",
        True,
        1,
        65536,
    ),
    ServiceSpec(
        "native-65k-b4",
        "vllm-xpu-brutus-kvarn-native-b4",
        True,
        4,
        65536,
    ),
    ServiceSpec(
        "reference-262k-b1",
        "vllm-xpu-brutus-kvarn-262k-b1",
        False,
        1,
        262144,
    ),
    ServiceSpec(
        "native-262k-b1-first",
        "vllm-xpu-brutus-kvarn-native-262k-b1",
        True,
        1,
        262144,
    ),
    ServiceSpec(
        "native-262k-b1-restart",
        "vllm-xpu-brutus-kvarn-native-262k-b1",
        True,
        1,
        262144,
    ),
)

PRIMITIVE_PLAN = (
    (
        "native_decode_short",
        "not long_context_ragged_b4_matches_structured_oracle",
    ),
    (
        "native_decode_262k",
        "long_context_ragged_b4_matches_structured_oracle",
    ),
)


def native_layout_for_spec(spec: ServiceSpec, args: argparse.Namespace) -> str:
    return args.native_layout if spec.native else "natural"


def native_kernel_variant_for_spec(spec: ServiceSpec, args: argparse.Namespace) -> str:
    return (
        args.native_kernel_variant
        if spec.native
        else perf.REFERENCE_NATIVE_KERNEL_VARIANT
    )


def native_splits_for_spec(spec: ServiceSpec, args: argparse.Namespace) -> int:
    if not spec.native:
        return perf.REFERENCE_NATIVE_SPLITS
    try:
        return int(args.native_splits[spec.batch])
    except (KeyError, TypeError, ValueError) as exc:
        raise CorrectnessError(
            f"no native split count configured for B{spec.batch}"
        ) from exc


def native_split_policy_for_spec(spec: ServiceSpec, args: argparse.Namespace) -> str:
    return "fixed" if not spec.native else args.native_split_policy


def native_max_splits_for_spec(spec: ServiceSpec, args: argparse.Namespace) -> int:
    if native_split_policy_for_spec(spec, args) == "b70_q6":
        return perf.B70_Q6_MAX_SPLITS
    return native_splits_for_spec(spec, args)


def native_splits_environment_for_spec(
    spec: ServiceSpec, args: argparse.Namespace
) -> str | None:
    if native_split_policy_for_spec(spec, args) == "b70_q6":
        return None
    return str(native_max_splits_for_spec(spec, args))


def candidate_variant_provenance(args: argparse.Namespace) -> dict[str, str]:
    split_policy = args.native_split_policy
    if split_policy == "fixed":
        split_policy += "_" + "_".join(
            f"b{batch}s{splits}" for batch, splits in sorted(args.native_splits.items())
        )
    scheduling = f"eager_mnbt{args.max_num_batched_tokens}"
    return {
        "kernel_strategy": f"native_xe2_qlen1_{args.native_kernel_variant}",
        "split_policy": split_policy,
        "fusion_strategy": "native_materializer_persistent_scratch",
        "scheduling_variant": scheduling,
        "variant_id": (
            f"native-xe2-{args.native_layout}-{args.native_kernel_variant}-"
            f"{split_policy}-{scheduling}"
        ),
    }


def service_variant_provenance(
    spec: ServiceSpec, args: argparse.Namespace
) -> dict[str, str]:
    if spec.native:
        return candidate_variant_provenance(args)
    scheduling = f"eager_mnbt{args.max_num_batched_tokens}"
    return {
        "kernel_strategy": "kvarn_non_native",
        "split_policy": "neutral_1",
        "fusion_strategy": "none",
        "scheduling_variant": scheduling,
        "variant_id": f"natural-kvarn-correctness-reference-{scheduling}",
    }


def launcher_name(spec: ServiceSpec, args: argparse.Namespace) -> str:
    if spec.native and args.native_layout == "xe2_dpas":
        suffix = "-262k" if spec.max_model_len == 262144 else ""
        return (
            "vllm-xpu-brutus-kvarn-native-dpas-"
            f"{args.native_kernel_variant}{suffix}-b{spec.batch}"
        )
    return spec.launcher


def service_spec_evidence(
    spec: ServiceSpec, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        **dataclasses.asdict(spec),
        "launcher": launcher_name(spec, args),
        "native_layout": native_layout_for_spec(spec, args),
        "native_kernel_variant": native_kernel_variant_for_spec(spec, args),
        "native_kernel_variant_id": perf.NATIVE_KERNEL_VARIANTS[
            native_kernel_variant_for_spec(spec, args)
        ],
        "native_output_dtype": args.native_output_dtype,
        "max_decode_splits": native_max_splits_for_spec(spec, args),
        "nominal_decode_splits": native_splits_for_spec(spec, args),
        "native_split_policy": native_split_policy_for_spec(spec, args),
        **service_variant_provenance(spec, args),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    perf.write_json_atomic(path, value)


def artifact(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CorrectnessError(f"missing evidence artifact: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def verify_artifact_references(
    value: Any, *, owner: Path, seen: set[Path] | None = None
) -> None:
    visited = set() if seen is None else seen
    if isinstance(value, list):
        for item in value:
            verify_artifact_references(item, owner=owner, seen=visited)
        return
    if not isinstance(value, dict):
        return
    referenced_path = value.get("path")
    referenced_sha256 = value.get("sha256")
    if isinstance(referenced_path, str) and isinstance(referenced_sha256, str):
        path = Path(referenced_path).expanduser().resolve()
        if not re.fullmatch(r"[0-9a-f]{64}", referenced_sha256):
            raise CorrectnessError(f"{owner}: invalid nested artifact SHA-256")
        if not path.is_file() or sha256_file(path) != referenced_sha256:
            raise CorrectnessError(f"{owner}: nested artifact changed: {path}")
        if path.suffix == ".json" and path not in visited:
            visited.add(path)
            try:
                nested = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CorrectnessError(
                    f"{owner}: cannot read nested JSON artifact {path}: {exc}"
                ) from exc
            verify_artifact_references(nested, owner=path, seen=visited)
    for item in value.values():
        verify_artifact_references(item, owner=owner, seen=visited)


def passed_artifact(
    path: Path,
    *,
    gate: str,
    candidate_id: str,
    process_package: str,
    native_layout: str,
    native_kernel_variant: str,
    native_split_policy: str,
    native_splits: Mapping[int, int],
    native_output_dtype: str,
    flush_index_materialization: str,
    native_frontend: str,
) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CorrectnessError(f"cannot load evidence {path}: {exc}") from exc
    if not isinstance(value, dict) or value.get("status") != "passed":
        raise CorrectnessError(f"evidence is not passed: {path}")
    verify_artifact_references(value, owner=path.resolve())
    try:
        validate_correctness_gate_evidence(
            gate,
            value,
            path=path.resolve(),
            candidate_id=candidate_id,
            process_package=process_package,
            native_layout=native_layout,
            native_kernel_variant=native_kernel_variant,
            native_split_policy=native_split_policy,
            native_splits=native_splits,
            native_output_dtype=native_output_dtype,
            flush_index_materialization=flush_index_materialization,
            native_frontend=native_frontend,
        )
    except GateError as exc:
        raise CorrectnessError(str(exc)) from exc
    return {"status": "passed", **artifact(path)}


def store_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    parts = resolved.parts
    if len(parts) < 4 or parts[:3] != ("/", "nix", "store"):
        raise CorrectnessError(f"path is not in the Nix store: {resolved}")
    return Path(*parts[:4])


def nix_closure(path: Path) -> list[str]:
    try:
        output = subprocess.run(
            ["nix-store", "-qR", str(path)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CorrectnessError(f"cannot query Nix closure for {path}: {exc}") from exc
    closure = sorted(set(output.splitlines()))
    if not closure:
        raise CorrectnessError(f"empty Nix closure for {path}")
    return closure


def closure_digest(paths: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(set(paths))) + "\n").encode()).hexdigest()


def runner_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = perf.runner_environment(args)
    environment.pop("VLLM_KVARN_DEFER_PREFILL_FLUSH", None)
    return environment


def primitive_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = runner_environment(args)
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def service_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = runner_environment(args)
    environment["KVARN_FACTORY_FLUSH_INDEX_MATERIALIZATION"] = (
        perf.flush_index_materialization_environment(args)
    )
    environment["KVARN_FACTORY_NATIVE_XPU_FRONTEND"] = perf.native_frontend_environment(
        args
    )
    environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] = str(DEFAULT_PREFILL_WINDOW_BLOCKS)
    return environment


def tracked_checkout_identity(
    repo: Path, *, allowed_untracked_prefixes: Sequence[str] = ()
) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "ls-files", "-z"],
            check=True,
            capture_output=True,
        )
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CorrectnessError(f"cannot enumerate tracked checkout: {exc}") from exc
    relative_paths = [
        Path(raw.decode(errors="surrogateescape"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]
    digest = hashlib.sha256()
    for relative in sorted(relative_paths, key=lambda path: str(path)):
        source = repo / relative
        if not source.is_file():
            raise CorrectnessError(f"tracked kernel source is missing: {source}")
        digest.update(str(relative).encode(errors="surrogateescape"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(source)))
    unexpected = [
        line
        for line in status
        if not (
            line.startswith("?? ")
            and any(
                line.removeprefix("?? ").startswith(prefix)
                for prefix in allowed_untracked_prefixes
            )
        )
    ]
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CorrectnessError(f"cannot identify checkout HEAD: {exc}") from exc
    return {
        "head": head,
        "files": len(relative_paths),
        "sha256": digest.hexdigest(),
        "unexpected_changes": unexpected,
    }


def kernel_checkout_identity(repo: Path) -> dict[str, Any]:
    return tracked_checkout_identity(repo, allowed_untracked_prefixes=(".dev-bin/",))


def verify_config_identity(args: argparse.Namespace) -> None:
    current = tracked_checkout_identity(args.config_repo)
    if current != args.config_identity:
        raise CorrectnessError("configuration checkout changed during the run")


def verify_packaging_identity(args: argparse.Namespace) -> None:
    current = tracked_checkout_identity(args.packaging_repo)
    expected = args.source_identity["runner_checkout"]
    if current != expected:
        raise CorrectnessError("correctness-runner checkout changed during the run")


def read_lock_revisions(path: Path) -> dict[str, str | None]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        nodes = document["nodes"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise CorrectnessError(f"cannot read flake lock {path}: {exc}") from exc
    return {
        name: (
            nodes.get(name, {}).get("locked", {}).get("rev")
            if isinstance(nodes.get(name), dict)
            else None
        )
        for name in (
            "vllm-xpu-release",
            "vllm-xpu-unstable-src",
            "vllm-xpu-kernels-unstable-src",
        )
    }


def verify_source_identity(args: argparse.Namespace) -> dict[str, Any]:
    config_checkout = tracked_checkout_identity(args.config_repo)
    if config_checkout["unexpected_changes"]:
        raise CorrectnessError(
            "configuration checkout must be clean: "
            + json.dumps(config_checkout["unexpected_changes"])
        )
    packaging_checkout = tracked_checkout_identity(args.packaging_repo)
    if packaging_checkout["unexpected_changes"]:
        raise CorrectnessError(
            "correctness-runner checkout must be clean: "
            + json.dumps(
                {
                    "head": packaging_checkout["head"],
                    "changed": packaging_checkout["unexpected_changes"],
                },
                sort_keys=True,
            )
        )
    lock_path = args.config_repo / "modules/flake/nixos/server/flake.lock"
    identity_dir = args.output_dir / "source-identity"
    identity_dir.mkdir(parents=True, exist_ok=True)
    lock_snapshot = identity_dir / "flake.lock"
    lock_snapshot.write_bytes(lock_path.read_bytes())
    revisions = read_lock_revisions(lock_path)
    if any(
        not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
        for revision in revisions.values()
    ):
        raise CorrectnessError(
            "configuration lock lacks complete immutable source revisions: "
            + json.dumps(revisions, sort_keys=True)
        )
    overrides = {
        "vllm-xpu-release": args.packaging_commit,
        "vllm-xpu-unstable-src": args.vllm_commit,
        "vllm-xpu-kernels-unstable-src": args.kernels_commit,
    }
    mismatches = {
        name: {"locked": revisions[name], "requested": requested}
        for name, requested in overrides.items()
        if requested is not None and requested != revisions[name]
    }
    if mismatches:
        raise CorrectnessError(
            "explicit source revisions differ from the verified configuration lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    args.packaging_commit = revisions["vllm-xpu-release"]
    args.vllm_commit = revisions["vllm-xpu-unstable-src"]
    args.kernels_commit = revisions["vllm-xpu-kernels-unstable-src"]

    test_path = (args.kernels_repo / NATIVE_TEST).resolve()
    try:
        kernel_head = subprocess.run(
            ["git", "-C", str(args.kernels_repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        changed = subprocess.run(
            [
                "git",
                "-C",
                str(args.kernels_repo),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CorrectnessError(f"cannot verify kernel test source: {exc}") from exc
    unexpected_changes = [
        line for line in changed if not line.startswith("?? .dev-bin/")
    ]
    if kernel_head != args.kernels_commit or unexpected_changes:
        raise CorrectnessError(
            "kernel checkout does not match the pinned commit: "
            + json.dumps({"head": kernel_head, "changed": unexpected_changes})
        )
    if not test_path.is_file():
        raise CorrectnessError(f"native primitive test is missing: {test_path}")
    source_sha256: dict[str, str] = {}
    for relative in NATIVE_TEST_SOURCES:
        source = args.kernels_repo / relative
        if not source.is_file():
            raise CorrectnessError(f"native primitive source is missing: {source}")
        source_sha256[relative] = sha256_file(source)
    runner_sources: dict[str, dict[str, str]] = {}
    for relative in RUNNER_SOURCES:
        source = args.packaging_repo / relative
        if not source.is_file():
            raise CorrectnessError(f"correctness runner source is missing: {source}")
        snapshot = identity_dir / "runner" / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        runner_sources[relative] = artifact(snapshot)
    return {
        "lock_path": str(lock_snapshot.resolve()),
        "lock_origin_path": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_snapshot),
        "revisions": revisions,
        "native_test_path": str(test_path),
        "native_test_sha256": sha256_file(test_path),
        "native_source_sha256": source_sha256,
        "kernel_tracked_checkout": kernel_checkout_identity(args.kernels_repo),
        "allowed_untracked_prefixes": [".dev-bin/"],
        "config_checkout": config_checkout,
        "config_checkout_path": str(args.config_repo),
        "runner_checkout": packaging_checkout,
        "runner_checkout_path": str(args.packaging_repo),
        "runner_sources": runner_sources,
    }


def resolve_native_library(
    primitive_python: Path, explicit: Path | None, args: argparse.Namespace
) -> Path:
    if explicit is not None:
        library = explicit.expanduser().resolve()
    else:
        code = (
            "import importlib.util; "
            f"s=importlib.util.find_spec({NATIVE_LIBRARY_MODULE!r}); "
            "assert s is not None and s.origin; print(s.origin)"
        )
        try:
            result = subprocess.run(
                [str(primitive_python), "-c", code],
                cwd=args.candidate_env,
                env=primitive_environment(args),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise CorrectnessError(
                f"cannot locate native XPU extension: {exc}"
            ) from exc
        library = Path(result.stdout.strip()).resolve()
    if not library.is_file() or library.suffix != ".so":
        raise CorrectnessError(
            f"native XPU extension is not a shared object: {library}"
        )
    return library


def probe_primitive_imports(args: argparse.Namespace) -> dict[str, Any]:
    code = """
import importlib
import importlib.util
import json
import pathlib
import sys

modules = ("pytest", "torch", "transformers", "vllm", "vllm_xpu_kernels")
origins = {}
for name in modules:
    module = importlib.import_module(name)
    origin = getattr(module, "__file__", None)
    if origin is None:
        raise RuntimeError(f"cannot resolve {name}")
    origins[name] = str(pathlib.Path(origin).resolve())
native = importlib.util.find_spec("vllm_xpu_kernels._vllm_fa2_C")
if native is None or native.origin is None:
    raise RuntimeError("cannot resolve native extension")
print(json.dumps({
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "origins": origins,
    "native_library": str(pathlib.Path(native.origin).resolve()),
}))
"""
    try:
        result = subprocess.run(
            [str(args.primitive_python), "-c", code],
            cwd=args.candidate_env,
            env=primitive_environment(args),
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise CorrectnessError(f"primitive import probe failed: {exc}") from exc
    origins = probe.get("origins") if isinstance(probe, dict) else None
    if not isinstance(origins, dict) or set(origins) != {
        "pytest",
        "torch",
        "transformers",
        "vllm",
        "vllm_xpu_kernels",
    }:
        raise CorrectnessError("primitive import probe returned invalid origins")
    if Path(probe.get("native_library", "")).resolve() != args.native_library:
        raise CorrectnessError("primitive import probe resolved a different extension")
    expected_closure = set(nix_closure(args.expected_package))
    python_closure = set(nix_closure(store_root(args.primitive_python)))
    for module, raw_origin in origins.items():
        origin = Path(raw_origin).resolve()
        origin_root = str(store_root(origin))
        if origin_root not in python_closure:
            raise CorrectnessError(
                f"primitive import {module} is outside the Python closure"
            )
        if (
            module in {"vllm", "vllm_xpu_kernels"}
            and origin_root not in expected_closure
        ):
            raise CorrectnessError(
                f"primitive import {module} is outside the pinned package closure"
            )
    if store_root(Path(origins["vllm"])) != args.expected_package:
        raise CorrectnessError("primitive Python imported a different vLLM package")
    return probe


def verify_candidate_preflight(
    candidate_env: Path,
    primitive_python: Path,
    native_library: Path,
    expected_package: Path,
) -> dict[str, Any]:
    if store_root(candidate_env) != candidate_env:
        raise CorrectnessError("--candidate-env must resolve to a Nix store root")
    candidate_closure = nix_closure(candidate_env)
    python_closure = nix_closure(store_root(primitive_python))
    expected_package_closure = nix_closure(expected_package)
    library_root = str(store_root(native_library))
    expected_package_text = str(expected_package)
    if expected_package_text not in candidate_closure:
        raise CorrectnessError(
            "candidate environment does not close over the package built from "
            "the pinned configuration"
        )
    if library_root not in candidate_closure:
        raise CorrectnessError("native extension is not in the candidate closure")
    if library_root not in python_closure:
        raise CorrectnessError(
            "primitive Python does not close over the native extension"
        )
    if library_root not in expected_package_closure:
        raise CorrectnessError(
            "native extension is not in the pinned service package closure"
        )
    return {
        "candidate_env": str(candidate_env),
        "candidate_closure_paths": candidate_closure,
        "candidate_closure_sha256": closure_digest(candidate_closure),
        "expected_process_package": expected_package_text,
        "expected_process_package_closure_sha256": closure_digest(
            expected_package_closure
        ),
        "primitive_python": str(primitive_python),
        "primitive_python_closure_sha256": closure_digest(python_closure),
        "native_library": str(native_library),
        "native_library_store_root": library_root,
        "native_library_sha256": sha256_file(native_library),
    }


def resolve_expected_package(args: argparse.Namespace) -> Path:
    package_name = (
        "vllm-xpu-kvarn-factory"
        if args.native_layout == "xe2_dpas"
        else "vllm-xpu-brutus"
    )
    installable = f"{args.config_ref}#{package_name}"
    verify_config_identity(args)
    try:
        result = subprocess.run(
            [
                "nix",
                "build",
                "--store",
                "daemon",
                "--no-link",
                "--json",
                installable,
            ],
            cwd=args.config_repo,
            env=runner_environment(args),
            check=True,
            capture_output=True,
            text=True,
        )
        document = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise CorrectnessError(
            f"cannot resolve pinned service package {installable}: {exc}"
        ) from exc
    if not isinstance(document, list) or len(document) != 1:
        raise CorrectnessError("service package build returned an invalid result")
    outputs = document[0].get("outputs") if isinstance(document[0], dict) else None
    if not isinstance(outputs, dict) or len(outputs) != 1:
        raise CorrectnessError("service package must have exactly one output")
    package = Path(next(iter(outputs.values()))).resolve()
    if store_root(package) != package or not (package / "bin/vllm").is_file():
        raise CorrectnessError(
            f"resolved service package is not a realized vLLM package: {package}"
        )
    verify_config_identity(args)
    return package


def resolve_launcher(launcher: str, args: argparse.Namespace) -> str:
    installable = f"{args.config_ref}#{launcher}"
    app_installable = f"{args.config_ref}#apps.x86_64-linux.{launcher}"
    environment = runner_environment(args)
    verify_config_identity(args)
    try:
        build = subprocess.run(
            [
                "nix",
                "build",
                "--store",
                "daemon",
                "--no-link",
                "--json",
                installable,
            ],
            cwd=args.config_repo,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        app = subprocess.run(
            [
                "nix",
                "eval",
                "--store",
                "daemon",
                "--json",
                app_installable,
                "--apply",
                "app: { program = app.program; context = builtins.getContext app.program; }",
            ],
            cwd=args.config_repo,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        build_result = json.loads(build.stdout)
        app_result = json.loads(app.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        hint = (
            "; xe2_dpas requires dedicated Brutus native-dpas launcher outputs "
            "that export KVARN_NATIVE_XPU_DPAS_LAYOUT=1"
            if "native-dpas" in launcher
            else ""
        )
        raise CorrectnessError(
            f"cannot resolve launcher {installable}{hint}: {exc}"
        ) from exc

    if not isinstance(build_result, list) or len(build_result) != 1:
        raise CorrectnessError(
            f"launcher build returned invalid result: {build_result!r}"
        )
    entry = build_result[0]
    if not isinstance(entry, dict):
        raise CorrectnessError("launcher build result is not an object")
    drv_path, outputs = entry.get("drvPath"), entry.get("outputs")
    program = app_result.get("program") if isinstance(app_result, dict) else None
    context = app_result.get("context") if isinstance(app_result, dict) else None
    if not isinstance(drv_path, str) or not isinstance(outputs, dict):
        raise CorrectnessError("launcher build result lacks drvPath/outputs")
    if not isinstance(program, str) or not isinstance(context, dict):
        raise CorrectnessError("launcher app result lacks program/context")
    context_entry = context.get(drv_path)
    if len(context) != 1 or not isinstance(context_entry, dict):
        raise CorrectnessError("launcher build and app derivations differ")
    output_names = context_entry.get("outputs")
    if not isinstance(output_names, list) or len(output_names) != 1:
        raise CorrectnessError("launcher app must reference exactly one output")
    physical_package = outputs.get(output_names[0])
    logical_program = Path(program)
    if (
        not isinstance(physical_package, str)
        or not physical_package.startswith("/nix/store/")
        or not logical_program.is_absolute()
        or logical_program.parent.name != "bin"
    ):
        raise CorrectnessError("launcher did not resolve to a physical store program")
    physical_program = Path(physical_package) / "bin" / logical_program.name
    if not physical_program.is_file():
        raise CorrectnessError(f"launcher program is not realized: {physical_program}")
    verify_config_identity(args)
    return str(physical_program)


def _arg_after(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def verify_service_profile(
    argv: Sequence[str],
    environment: Mapping[str, str | None],
    spec: ServiceSpec,
    args: argparse.Namespace,
) -> None:
    try:
        serve_index = argv.index("serve")
        served_model_path = argv[serve_index + 1]
    except (ValueError, IndexError) as exc:
        raise CorrectnessError(
            "captured service argv has no model after serve"
        ) from exc
    expected_arguments = {
        "model": args.model,
        "--served-model-name": args.served_model,
        "--revision": args.model_revision,
        "--dtype": "bfloat16",
        "--quantization": "compressed-tensors",
        "--kv-cache-dtype": perf.COMPACT_DTYPE,
        "--max-model-len": str(spec.max_model_len),
        "--max-num-seqs": str(spec.batch),
        "--max-num-batched-tokens": str(args.max_num_batched_tokens),
        "--gpu-memory-utilization": "0.95",
    }
    actual_arguments = {
        name: served_model_path if name == "model" else _arg_after(argv, name)
        for name in expected_arguments
    }
    argument_mismatches = {
        name: {"actual": actual_arguments[name], "expected": expected}
        for name, expected in expected_arguments.items()
        if actual_arguments[name] != expected
    }
    required_flags = {
        "--enforce-eager",
        "--language-model-only",
        "--no-enable-prefix-caching",
    }
    missing_flags = sorted(required_flags - set(argv))
    forbidden_flags = sorted(
        flag
        for flag in ("--speculative-config", "--compilation-config")
        if flag in argv
    )
    native = "1" if spec.native else "0"
    expected_environment = {
        "CCL_ATL_TRANSPORT": "ofi",
        "CCL_LOG_LEVEL": "warn",
        "CCL_PROCESS_LAUNCHER": "none",
        "CCL_ZE_IPC_EXCHANGE": "sockets",
        "HF_HOME": str(args.hf_home),
        "HOME": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "KVARN_NATIVE_XPU": native,
        "KVARN_NATIVE_XPU_CACHE_LAYOUT": native_layout_for_spec(spec, args),
        "KVARN_NATIVE_XPU_DECODE": native,
        "KVARN_NATIVE_XPU_DPAS_LAYOUT": perf.NATIVE_LAYOUT_ENV[
            native_layout_for_spec(spec, args)
        ],
        "KVARN_FLUSH_INDEX_MATERIALIZATION": (
            perf.flush_index_materialization_environment(args)
        ),
        "KVARN_NATIVE_XPU_FRONTEND": perf.native_frontend_environment(args),
        "KVARN_NATIVE_XPU_KERNEL_VARIANT": native_kernel_variant_for_spec(spec, args),
        "KVARN_NATIVE_XPU_MATERIALIZE": native,
        "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH": native,
        "KVARN_NATIVE_XPU_SPLITS": native_splits_environment_for_spec(spec, args),
        "KVARN_NATIVE_XPU_SPLIT_POLICY": native_split_policy_for_spec(spec, args),
        "KVARN_ONEDNN_DETERMINISTIC": "1",
        "KVARN_PREFILL_FP16_WINDOW_BLOCKS": str(DEFAULT_PREFILL_WINDOW_BLOCKS),
        "VLLM_CACHE_ROOT": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_KVARN_DEFER_PREFILL_FLUSH": None,
        "VLLM_USE_V2_MODEL_RUNNER": perf.VLLM_USE_V2_MODEL_RUNNER,
        "XDG_CACHE_HOME": str(args.runtime_cache),
    }
    environment_mismatches = {
        name: {"actual": environment.get(name), "expected": expected}
        for name, expected in expected_environment.items()
        if environment.get(name) != expected
    }
    if environment.get("VLLM_XPU_ENABLE_XPU_GRAPH") not in {None, "", "0"}:
        environment_mismatches["VLLM_XPU_ENABLE_XPU_GRAPH"] = {
            "actual": environment.get("VLLM_XPU_ENABLE_XPU_GRAPH"),
            "expected": "unset or 0",
        }
    if (
        argument_mismatches
        or missing_flags
        or forbidden_flags
        or environment_mismatches
    ):
        raise CorrectnessError(
            "foreground service profile mismatch: "
            + json.dumps(
                {
                    "arguments": argument_mismatches,
                    "missing_flags": missing_flags,
                    "forbidden_flags": forbidden_flags,
                    "environment": environment_mismatches,
                },
                sort_keys=True,
            )
        )


def service_command(spec: ServiceSpec, args: argparse.Namespace) -> list[str]:
    return [
        args.resolved_launchers[launcher_name(spec, args)],
        str(args.candidate_env),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]


def start_service(
    spec: ServiceSpec, phase_dir: Path, args: argparse.Namespace
) -> perf.ServiceProcess:
    command = service_command(spec, args)
    write_json_atomic(phase_dir / "service-command.json", command)
    for attempt in range(1, args.startup_attempts + 1):
        assert_units_inactive(args.require_inactive_unit)
        perf.assert_port_unused(args.base_url)
        log_path = phase_dir / "engine.log"
        log_stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=args.config_repo,
            env=service_environment(args),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_group = os.getpgid(process.pid)
        args.supervisor.register(process_group, f"service:{spec.name}")
        service = perf.ServiceProcess(
            process=process,
            process_group=process_group,
            log_stream=log_stream,
            log_path=log_path,
            engine_pid=process.pid,
            argv=[],
            environment={},
            supervisor=args.supervisor,
        )
        try:
            perf.wait_for_ready(process, args)
            engine_pid, argv, environment = perf.capture_engine_process(process)
            verify_service_profile(argv, environment, spec, args)
            service.engine_pid = engine_pid
            service.argv = argv
            service.environment = environment
            return service
        except BaseException as exc:
            perf.stop_service(service, args.shutdown_timeout)
            if isinstance(exc, (perf.RunnerInterrupted, KeyboardInterrupt)):
                raise
            if attempt == args.startup_attempts:
                raise
            log_path.replace(phase_dir / f"engine-failed-startup-{attempt}.log")
    raise AssertionError("unreachable")


def identity_key(identity: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        identity.get(name)
        for name in (
            "candidate_env",
            "process_package",
            "candidate_closure_sha256",
            "process_closure_sha256",
        )
    )


def run_service_phase(
    spec: ServiceSpec,
    args: argparse.Namespace,
    workload: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    phase_dir = args.output_dir / "services" / spec.name
    phase_dir.mkdir(parents=True)
    phase_path = phase_dir / "phase.json"
    state: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "spec": service_spec_evidence(spec, args),
        "started_at": perf.utc_timestamp(),
    }
    write_json_atomic(phase_path, state)
    service: perf.ServiceProcess | None = None
    try:
        service = start_service(spec, phase_dir, args)
        profile = perf.service_profile_evidence(
            service.argv,
            service.environment,
            variant_provenance=service_variant_provenance(spec, args),
        )
        identity = perf.verify_candidate_identity(service.argv, args.candidate_env)
        captured_layout_environment = service.environment.get(
            "KVARN_NATIVE_XPU_DPAS_LAYOUT"
        )
        captured_cache_layout_environment = service.environment.get(
            "KVARN_NATIVE_XPU_CACHE_LAYOUT"
        )
        captured_kernel_variant_environment = service.environment.get(
            "KVARN_NATIVE_XPU_KERNEL_VARIANT"
        )
        captured_max_splits_environment = service.environment.get(
            "KVARN_NATIVE_XPU_SPLITS"
        )
        captured_split_policy_environment = service.environment.get(
            "KVARN_NATIVE_XPU_SPLIT_POLICY"
        )
        captured_flush_index_materialization = service.environment.get(
            "KVARN_FLUSH_INDEX_MATERIALIZATION"
        )
        captured_native_frontend = service.environment.get("KVARN_NATIVE_XPU_FRONTEND")
        write_json_atomic(phase_dir / "service-profile.json", profile)
        write_json_atomic(phase_dir / "candidate-identity.json", identity)
        engine_pid = service.engine_pid
        payload = workload()
        perf.stop_service(service, args.shutdown_timeout)
        service = None
        log_scan = perf.validate_engine_log(
            phase_dir / "engine.log",
            native=spec.native,
            expected_layout=native_layout_for_spec(spec, args),
            expected_kernel_variant=native_kernel_variant_for_spec(spec, args),
            expected_max_splits=native_max_splits_for_spec(spec, args),
            expected_split_policy=native_split_policy_for_spec(spec, args),
            expected_frontend=perf.native_frontend_environment(args),
        )
        native_dispatch_verified = spec.native and perf.NATIVE_DISPATCH in (
            phase_dir / "engine.log"
        ).read_text(encoding="utf-8", errors="replace")
        write_json_atomic(phase_dir / "engine-log-scan.json", log_scan)
        state.update(
            status="passed",
            finished_at=perf.utc_timestamp(),
            service_pid=engine_pid,
            profile=artifact(phase_dir / "service-profile.json"),
            identity=artifact(phase_dir / "candidate-identity.json"),
            engine_log=artifact(phase_dir / "engine.log"),
            engine_log_scan=artifact(phase_dir / "engine-log-scan.json"),
            native_dispatch_verified=native_dispatch_verified,
            native_layout=native_layout_for_spec(spec, args),
            native_layout_environment=captured_layout_environment,
            native_cache_layout_environment=captured_cache_layout_environment,
            native_kernel_variant=native_kernel_variant_for_spec(spec, args),
            native_kernel_variant_id=perf.NATIVE_KERNEL_VARIANTS[
                native_kernel_variant_for_spec(spec, args)
            ],
            native_output_dtype=args.native_output_dtype,
            native_direct_bf16_verified=log_scan["native_direct_bf16_verified"],
            native_direct_bf16_log_marker=log_scan["native_direct_bf16_log_marker"],
            native_kernel_variant_environment=captured_kernel_variant_environment,
            native_max_splits=native_max_splits_for_spec(spec, args),
            native_nominal_splits=native_splits_for_spec(spec, args),
            native_max_splits_environment=captured_max_splits_environment,
            native_split_policy=service_variant_provenance(spec, args)["split_policy"],
            native_split_policy_environment=captured_split_policy_environment,
            flush_index_materialization=captured_flush_index_materialization,
            native_frontend=captured_native_frontend,
            native_frontend_active_verified=log_scan["native_frontend_active_verified"],
            native_frontend_log_marker=log_scan["native_frontend_log_marker"],
            native_layout_log_marker=perf.kvarn_factory_marker(
                cache_layout=native_layout_for_spec(spec, args),
                kernel_variant=native_kernel_variant_for_spec(spec, args),
                max_decode_splits=native_max_splits_for_spec(spec, args),
                split_policy=native_split_policy_for_spec(spec, args),
            ),
            native_layout_evidence=(
                "captured-process-environment-plus-factory-marker-plus-native-dispatch"
                if spec.native
                else "captured-process-environment-plus-factory-marker"
            ),
            workload=payload,
        )
        write_json_atomic(phase_path, state)
        return state
    except BaseException as exc:
        if service is not None:
            try:
                perf.stop_service(service, args.shutdown_timeout)
            except (
                OSError,
                subprocess.SubprocessError,
                perf.RunnerError,
            ) as stop_error:  # Preserve the primary failure.
                state["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        state.update(
            status="failed",
            finished_at=perf.utc_timestamp(),
            error=f"{type(exc).__name__}: {exc}",
        )
        if (phase_dir / "engine.log").is_file():
            state["engine_log"] = artifact(phase_dir / "engine.log")
        write_json_atomic(phase_path, state)
        raise


def _fixture_texts(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, list):
        raise CorrectnessError("base fixtures must be a JSON list")
    result: dict[str, str] = {}
    for item in document:
        if isinstance(item, dict):
            category, prompt = item.get("category"), item.get("prompt")
            if isinstance(category, str) and isinstance(prompt, str):
                result[category] = prompt
    missing = {"dialogue", "code", "math", "reasoning"} - result.keys()
    if missing:
        raise CorrectnessError(f"base fixtures lack categories: {sorted(missing)}")
    return result


def exact_prompt_ids(
    tokenizer: Any,
    prompt: str,
    category: str,
    target: int,
    *,
    trailing_prompt: bool,
) -> list[int]:
    suffix: list[int] = []
    fill_target = target
    if trailing_prompt:
        suffix = tokenizer.encode(
            "\n\nFinal task after reviewing the records:\n" + prompt,
            add_special_tokens=False,
        )
        fill_target -= len(suffix)
    if fill_target < 1:
        raise CorrectnessError("target is too short for the trailing instruction")
    token_ids = list(tokenizer.encode(prompt))
    counter = 0
    while len(token_ids) < fill_target:
        digest = hashlib.sha256(f"{category}:{counter}".encode()).hexdigest()
        record = (
            f"\nCategory {category} evidence record {counter}; "
            f"stable digest {digest}; retain its distinct facts and order."
        )
        token_ids.extend(tokenizer.encode(record, add_special_tokens=False))
        counter += 1
    result = token_ids[:fill_target] + suffix
    if len(result) != target or any(
        isinstance(token_id, bool) or not isinstance(token_id, int)
        for token_id in result
    ):
        raise CorrectnessError("tokenizer produced an invalid exact-length prompt")
    return result


def tokenize_worker(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    args = parser.parse_args(arguments)
    from transformers import AutoTokenizer

    texts = _fixture_texts(args.fixtures)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    specs = (
        ("dialogue-127", "dialogue", 127, False),
        ("code-4095", "code", 4095, True),
        ("math-16383", "math", 16383, False),
        ("reasoning-65023", "reasoning", 65023, True),
        ("reasoning-261631", "reasoning", NEAR_262K_PROMPT_TOKENS, True),
    )
    prompts = {
        name: exact_prompt_ids(
            tokenizer,
            texts[category],
            category,
            length,
            trailing_prompt=trailing,
        )
        for name, category, length, trailing in specs
    }
    json.dump(prompts, sys.stdout, separators=(",", ":"))
    return 0


def tokenize_fixtures(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    command = [
        str(args.primitive_python),
        str(Path(__file__).resolve()),
        "_tokenize_worker",
        "--model",
        args.model,
        "--revision",
        args.model_revision,
        "--fixtures",
        str(args.fixtures),
    ]
    try:
        result = subprocess.run(
            command,
            cwd=args.packaging_repo,
            env=primitive_environment(args),
            check=True,
            capture_output=True,
            text=True,
            timeout=args.tokenizer_timeout,
        )
        prompts = json.loads(result.stdout)
    except (
        OSError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ) as exc:
        raise CorrectnessError(
            f"cannot prepare deterministic token fixtures: {exc}"
        ) from exc
    expected_lengths = {
        "dialogue-127": 127,
        "code-4095": 4095,
        "math-16383": 16383,
        "reasoning-65023": 65023,
        "reasoning-261631": NEAR_262K_PROMPT_TOKENS,
    }
    if not isinstance(prompts, dict) or set(prompts) != set(expected_lengths):
        raise CorrectnessError("tokenizer worker returned the wrong fixture set")
    fixtures: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for name, expected_length in expected_lengths.items():
        prompt = prompts[name]
        if (
            not isinstance(prompt, list)
            or len(prompt) != expected_length
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in prompt
            )
        ):
            raise CorrectnessError(f"invalid generated fixture: {name}")
        fixtures[name] = {
            "id": name,
            "prompt": prompt,
            "max_tokens": args.output_tokens,
        }
        records.append(
            {
                "id": name,
                "prompt_tokens": expected_length,
                "max_tokens": args.output_tokens,
                "prompt_token_ids_sha256": sha256_json(prompt),
            }
        )
    write_json_atomic(
        args.output_dir / "fixture-manifest.json",
        {
            "schema_version": 1,
            "status": "passed",
            "generator": "category-record-v1",
            "base_fixtures": artifact(args.fixtures),
            "model": args.model,
            "model_revision": args.model_revision,
            "fixtures": records,
            "note": "Token arrays are generated in memory and are not persisted.",
        },
    )
    return fixtures


def compact_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in result.items()
        if name not in {"raw_response", "text", "prompt", "prompt_token_ids"}
    }


def validate_completion_result(
    result: Mapping[str, Any], fixture: Mapping[str, Any], output_tokens: int
) -> dict[str, int]:
    prompt = fixture.get("prompt")
    expected_prompt_tokens = len(prompt) if isinstance(prompt, list) else None
    response = result.get("raw_response")
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict) or expected_prompt_tokens is None:
        raise CorrectnessError("completion response has no verifiable token usage")
    expected_usage = {
        "prompt_tokens": expected_prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": expected_prompt_tokens + output_tokens,
    }
    actual_usage = {name: usage.get(name) for name in expected_usage}
    if actual_usage != expected_usage:
        raise CorrectnessError(
            "completion token usage mismatch: "
            + json.dumps({"actual": actual_usage, "expected": expected_usage})
        )
    if result.get("finish_reason") != "length":
        raise CorrectnessError(
            f"completion finish_reason must be 'length': {result.get('finish_reason')!r}"
        )
    return expected_usage


def checked_completion(
    args: argparse.Namespace, fixture: dict[str, Any]
) -> dict[str, Any]:
    result = completion(
        args.base_url,
        args.served_model,
        fixture,
        args.output_tokens,
        args.request_timeout,
    )
    failures = result_quality_failures([result])
    if failures:
        raise CorrectnessError(f"completion quality failed: {failures}")
    result["usage"] = validate_completion_result(result, fixture, args.output_tokens)
    return compact_result(result)


def compare_results(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> dict[str, Any]:
    same_fixture = expected.get("id") == actual.get("id")
    same_prompt = expected.get("prompt_token_ids_sha256") == actual.get(
        "prompt_token_ids_sha256"
    )
    same_tokens = expected.get("token_ids") == actual.get("token_ids")
    passed = same_fixture and same_prompt and same_tokens
    result = {
        "status": "passed" if passed else "failed",
        "fixture_id": actual.get("id"),
        "same_fixture": same_fixture,
        "same_prompt_token_ids": same_prompt,
        "token_ids_identical": same_tokens,
        "expected_token_ids_sha256": expected.get("token_ids_sha256"),
        "actual_token_ids_sha256": actual.get("token_ids_sha256"),
    }
    if not passed:
        raise CorrectnessError("completion token IDs differ: " + json.dumps(result))
    return result


def run_primitive_gate(gate: str, expression: str, args: argparse.Namespace) -> Path:
    gate_dir = args.output_dir / "primitives" / gate
    gate_dir.mkdir(parents=True)
    stdout_path = gate_dir / "pytest.log"
    junit_path = gate_dir / "pytest.xml"
    evidence_path = gate_dir / "evidence.json"
    expected_source_sha256 = args.source_identity["native_source_sha256"]
    expected_tree = args.source_identity["kernel_tracked_checkout"]

    def current_source_sha256() -> dict[str, str]:
        return {
            relative: sha256_file(args.kernels_repo / relative)
            for relative in NATIVE_TEST_SOURCES
        }

    if (
        current_source_sha256() != expected_source_sha256
        or kernel_checkout_identity(args.kernels_repo) != expected_tree
    ):
        raise CorrectnessError(
            f"primitive gate {gate} source changed after candidate preflight"
        )
    command = [
        str(args.primitive_python),
        "-m",
        "pytest",
        str((args.kernels_repo / NATIVE_TEST).resolve()),
        "-v",
        "-p",
        "no:cacheprovider",
        "-k",
        expression,
        "--junitxml",
        str(junit_path),
    ]
    write_json_atomic(gate_dir / "command.json", command)
    environment = primitive_environment(args)
    environment["VLLM_XPU_KERNELS_LIBRARY"] = str(args.native_library)
    assert_units_inactive(args.require_inactive_unit)
    with stdout_path.open("w", encoding="utf-8") as output:
        returncode = perf.run_managed_process(
            command,
            cwd=args.kernels_repo,
            environment=environment,
            output=output,
            timeout=args.primitive_timeout,
            supervisor=args.supervisor,
            label=f"primitive:{gate}",
        )
    if returncode != 0 or not junit_path.is_file():
        raise CorrectnessError(f"primitive gate {gate} exited {returncode}")
    if (
        current_source_sha256() != expected_source_sha256
        or kernel_checkout_identity(args.kernels_repo) != expected_tree
    ):
        raise CorrectnessError(f"primitive gate {gate} source changed during pytest")
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise CorrectnessError(
            f"primitive gate {gate} has invalid JUnit: {exc}"
        ) from exc
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    counts = {
        name: sum(int(suite.get(name, "0")) for suite in suites)
        for name in ("tests", "failures", "errors", "skipped")
    }
    if counts["tests"] < 1:
        raise CorrectnessError(f"primitive gate {gate} collected no tests")
    if any(counts[name] for name in ("failures", "errors", "skipped")):
        raise CorrectnessError(
            f"primitive gate {gate} was not an all-pass XPU run: {counts}"
        )
    write_json_atomic(
        evidence_path,
        {
            "schema_version": 1,
            "status": "passed",
            "gate": gate,
            "candidate_id": str(args.candidate_env),
            "qualification_scope": "combined_library_variant_matrix",
            "variant_selection": "explicit_per_op_arguments",
            "factory_variant_matrix": COMBINED_LIBRARY_VARIANT_MATRIX,
            "command": command,
            "native_library": artifact(args.native_library),
            "test_source": artifact(args.kernels_repo / NATIVE_TEST),
            "helper_sources": {
                relative: artifact(args.kernels_repo / relative)
                for relative in NATIVE_TEST_SOURCES
                if relative != NATIVE_TEST
            },
            "junit_counts": counts,
            "pytest_log": artifact(stdout_path),
            "junit": artifact(junit_path),
        },
    )
    return evidence_path


def assert_units_inactive(units: Sequence[str]) -> dict[str, str]:
    states: dict[str, str] = {}
    for unit in units:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", unit],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            raise CorrectnessError(f"cannot inspect {unit}: {exc}") from exc
        state = result.stdout.strip()
        states[unit] = state
        if state not in {"inactive", "failed"}:
            raise CorrectnessError(
                f"required service is not inactive: {unit}={state!r}"
            )
    return states


def _phase_path(args: argparse.Namespace, spec: ServiceSpec) -> Path:
    return args.output_dir / "services" / spec.name / "phase.json"


def write_gate(path: Path, gate: str, **evidence: Any) -> Path:
    write_json_atomic(
        path,
        {"schema_version": 1, "status": "passed", "gate": gate, **evidence},
    )
    return path


def run_service_gates(
    fixtures: dict[str, dict[str, Any]], args: argparse.Namespace
) -> dict[str, Path]:
    by_name = {spec.name: spec for spec in SERVICE_PLAN}
    service_evidence = args.output_dir / "gates"
    service_evidence.mkdir(parents=True)
    short_fixtures = [
        fixtures[name]
        for name in ("dialogue-127", "code-4095", "math-16383", "reasoning-65023")
    ]

    first_results: list[dict[str, Any]] = []
    replay_results: list[dict[str, Any]] = []
    cancel_evidence: dict[str, Any] = {}

    def b1_first_workload() -> dict[str, Any]:
        nonlocal first_results, replay_results, cancel_evidence
        first_results = [
            checked_completion(args, fixture) for fixture in short_fixtures
        ]
        replay_results = [
            checked_completion(args, fixture) for fixture in short_fixtures
        ]
        comparisons = [
            compare_results(expected, actual)
            for expected, actual in zip(first_results, replay_results, strict=True)
        ]
        boundary = short_fixtures[-1]
        events = cancel_stream(
            args.base_url,
            args.served_model,
            boundary,
            args.output_tokens,
            args.request_timeout,
            args.cancel_after_events,
        )
        idle_metrics = wait_for_idle(
            args.base_url,
            timeout=min(args.request_timeout, args.cancellation_idle_timeout),
        )
        replacement = checked_completion(args, boundary)
        cancel_comparison = compare_results(first_results[-1], replacement)
        cancel_evidence = {
            "requested_generated_token_checkpoint": args.cancel_after_events,
            "generated_token_ids_before_close": events,
            "idle_metrics_before_replacement": idle_metrics,
            "replacement": replacement,
            "comparison": cancel_comparison,
        }
        return {
            "first": first_results,
            "replay": replay_results,
            "replay_comparisons": comparisons,
            "cancellation": cancel_evidence,
        }

    first_spec = by_name["native-65k-b1-first"]
    first_phase = run_service_phase(first_spec, args, b1_first_workload)
    b1_phase_ref = artifact(_phase_path(args, first_spec))
    gate_paths = {
        "b1_replay": write_gate(
            service_evidence / "b1-replay.json",
            "b1_replay",
            service_phase=b1_phase_ref,
            comparisons=first_phase["workload"]["replay_comparisons"],
        ),
        "cancel_reuse": write_gate(
            service_evidence / "cancel-reuse.json",
            "cancel_reuse",
            service_phase=b1_phase_ref,
            **cancel_evidence,
        ),
    }

    def b1_restart_workload() -> dict[str, Any]:
        results = [checked_completion(args, fixture) for fixture in short_fixtures]
        comparisons = [
            compare_results(expected, actual)
            for expected, actual in zip(first_results, results, strict=True)
        ]
        return {"results": results, "comparisons": comparisons}

    restart_spec = by_name["native-65k-b1-restart"]
    restart_phase = run_service_phase(restart_spec, args, b1_restart_workload)
    gate_paths["b1_restart"] = write_gate(
        service_evidence / "b1-restart.json",
        "b1_restart",
        original_service_phase=b1_phase_ref,
        restarted_service_phase=artifact(_phase_path(args, restart_spec)),
        comparisons=restart_phase["workload"]["comparisons"],
    )

    def b4_workload() -> dict[str, Any]:
        raw_results, overlap = run_concurrent_wave(
            args.base_url,
            args.served_model,
            short_fixtures,
            args.output_tokens,
            4,
            args.request_timeout,
            args.metrics_poll_interval,
        )
        failures = result_quality_failures(raw_results)
        for fixture, result in zip(short_fixtures, raw_results, strict=True):
            result["usage"] = validate_completion_result(
                result, fixture, args.output_tokens
            )
        if failures or not overlap["required_overlap_observed"]:
            raise CorrectnessError(
                "B4 service did not prove clean full-width overlap: "
                + json.dumps({"quality": failures, "overlap": overlap})
            )
        results = [compact_result(result) for result in raw_results]
        comparisons = [
            compare_results(expected, actual)
            for expected, actual in zip(first_results, results, strict=True)
        ]
        return {"results": results, "comparisons": comparisons, "overlap": overlap}

    b4_spec = by_name["native-65k-b4"]
    b4_phase = run_service_phase(b4_spec, args, b4_workload)
    gate_paths["b4_isolation"] = write_gate(
        service_evidence / "b4-isolation.json",
        "b4_isolation",
        b1_service_phase=b1_phase_ref,
        b4_service_phase=artifact(_phase_path(args, b4_spec)),
        comparisons=b4_phase["workload"]["comparisons"],
        overlap=b4_phase["workload"]["overlap"],
    )

    near_fixture = fixtures["reasoning-261631"]
    reference_result: dict[str, Any] = {}

    def reference_workload() -> dict[str, Any]:
        nonlocal reference_result
        reference_result = checked_completion(args, near_fixture)
        return {"result": reference_result}

    reference_spec = by_name["reference-262k-b1"]
    run_service_phase(reference_spec, args, reference_workload)

    first_native_262: dict[str, Any] = {}

    def native_262_workload() -> dict[str, Any]:
        nonlocal first_native_262
        first_native_262 = checked_completion(args, near_fixture)
        comparison = compare_results(reference_result, first_native_262)
        return {"result": first_native_262, "reference_comparison": comparison}

    native_262_spec = by_name["native-262k-b1-first"]
    native_262_phase = run_service_phase(native_262_spec, args, native_262_workload)
    gate_paths["near_262k_reference_equivalence"] = write_gate(
        service_evidence / "near-262k-reference-equivalence.json",
        "near_262k_reference_equivalence",
        reference_service_phase=artifact(_phase_path(args, reference_spec)),
        native_service_phase=artifact(_phase_path(args, native_262_spec)),
        comparison=native_262_phase["workload"]["reference_comparison"],
    )

    def native_262_restart_workload() -> dict[str, Any]:
        result = checked_completion(args, near_fixture)
        return {
            "result": result,
            "reference_comparison": compare_results(reference_result, result),
            "native_restart_comparison": compare_results(first_native_262, result),
        }

    native_262_restart_spec = by_name["native-262k-b1-restart"]
    native_262_restart_phase = run_service_phase(
        native_262_restart_spec, args, native_262_restart_workload
    )
    gate_paths["near_262k_restart"] = write_gate(
        service_evidence / "near-262k-restart.json",
        "near_262k_restart",
        reference_service_phase=artifact(_phase_path(args, reference_spec)),
        first_native_service_phase=artifact(_phase_path(args, native_262_spec)),
        restarted_native_service_phase=artifact(
            _phase_path(args, native_262_restart_spec)
        ),
        reference_comparison=native_262_restart_phase["workload"][
            "reference_comparison"
        ],
        native_restart_comparison=native_262_restart_phase["workload"][
            "native_restart_comparison"
        ],
    )
    return gate_paths


def verify_phase_identities(args: argparse.Namespace) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for spec in SERVICE_PLAN:
        path = args.output_dir / "services" / spec.name / "candidate-identity.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CorrectnessError(f"invalid service identity: {path}")
        if value.get("process_package") != str(args.expected_package):
            raise CorrectnessError(
                f"service phase did not execute the pinned package: {path}"
            )
        identities.append(value)
    keys = {identity_key(identity) for identity in identities}
    if len(keys) != 1:
        raise CorrectnessError(
            "service starts did not use one candidate/package closure"
        )
    return identities


def build_manifest(
    args: argparse.Namespace, gate_paths: Mapping[str, Path]
) -> dict[str, Any]:
    verify_config_identity(args)
    verify_packaging_identity(args)
    missing = set(REQUIRED_GATES) - gate_paths.keys()
    extra = gate_paths.keys() - set(REQUIRED_GATES)
    if missing or extra:
        raise CorrectnessError(
            f"incorrect gate set: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    identities = verify_phase_identities(args)
    native_phases = [spec for spec in SERVICE_PLAN if spec.native]
    native_phase_documents = [
        json.loads(_phase_path(args, spec).read_text(encoding="utf-8"))
        for spec in native_phases
    ]
    if not all(
        phase.get("native_dispatch_verified") is True
        for phase in native_phase_documents
    ):
        raise CorrectnessError("not every native service phase verified dispatch")
    if not all(
        phase.get("native_direct_bf16_verified") is True
        and phase.get("native_direct_bf16_log_marker") == perf.NATIVE_DIRECT_BF16_MARKER
        for phase in native_phase_documents
    ):
        raise CorrectnessError(
            "not every native service phase verified direct BF16 output"
        )
    gates = {
        name: passed_artifact(
            gate_paths[name],
            gate=name,
            candidate_id=str(args.candidate_env),
            process_package=str(args.expected_package),
            native_layout=args.native_layout,
            native_kernel_variant=args.native_kernel_variant,
            native_split_policy=args.native_split_policy,
            native_splits=args.native_splits,
            native_output_dtype=args.native_output_dtype,
            flush_index_materialization=(
                perf.flush_index_materialization_environment(args)
            ),
            native_frontend=perf.native_frontend_environment(args),
        )
        for name in REQUIRED_GATES
    }
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "candidate_id": str(args.candidate_env),
        "native_dispatch_verified": True,
        "native_direct_bf16_verified": True,
        "native_direct_bf16_log_marker": perf.NATIVE_DIRECT_BF16_MARKER,
        "gates": gates,
        "candidate_identity": {
            name: identities[0][name]
            for name in (
                "process_package",
                "candidate_closure_sha256",
                "process_closure_sha256",
            )
        },
        "source_identity": args.source_identity,
        "factory_qualification": args.factory_qualification,
        "native_layout": args.native_layout,
        "native_kernel_variant": args.native_kernel_variant,
        "native_kernel_variant_id": perf.NATIVE_KERNEL_VARIANTS[
            args.native_kernel_variant
        ],
        "native_nominal_splits_by_batch": {
            str(batch): splits for batch, splits in sorted(args.native_splits.items())
        },
        "native_output_dtype": args.native_output_dtype,
        "native_split_policy": args.native_split_policy,
        "flush_index_materialization": (
            perf.flush_index_materialization_environment(args)
        ),
        "native_frontend": perf.native_frontend_environment(args),
        "service_controls": {
            "kvarn_flush_index_materialization": (
                perf.flush_index_materialization_environment(args)
            ),
            "kvarn_onednn_deterministic": "1",
            "kvarn_native_frontend": perf.native_frontend_environment(args),
            "vllm_use_v2_model_runner": perf.VLLM_USE_V2_MODEL_RUNNER,
        },
        "native_scratch_max_splits": (
            perf.B70_Q6_MAX_SPLITS
            if args.native_split_policy == "b70_q6"
            else max(args.native_splits.values())
        ),
        **candidate_variant_provenance(args),
        "service_start_plan": [
            service_spec_evidence(spec, args) for spec in SERVICE_PLAN
        ],
        "fixture_manifest": artifact(args.output_dir / "fixture-manifest.json"),
        "created_at": perf.utc_timestamp(),
    }
    verify_artifact_references(
        manifest, owner=args.output_dir / "native-correctness.json"
    )
    return manifest


def write_checksums(root: Path) -> None:
    output = root / "SHA256SUMS"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root)}"
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path != output
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    args.source_identity = verify_source_identity(args)
    args.config_identity = args.source_identity["config_checkout"]
    args.native_library = resolve_native_library(
        args.primitive_python, args.native_library, args
    )
    args.expected_package = resolve_expected_package(args)
    preflight = verify_candidate_preflight(
        args.candidate_env,
        args.primitive_python,
        args.native_library,
        args.expected_package,
    )
    try:
        args.factory_qualification = validate_factory_qualification(
            args.factory_result,
            native_layout=args.native_layout,
            native_kernel_variant=args.native_kernel_variant,
            native_split_policy=args.native_split_policy,
            native_splits=args.native_splits,
            output_dtype=args.native_output_dtype,
            expected_revisions={
                "vllm-xpu-nix": args.packaging_commit,
                "vllm": args.vllm_commit,
                "vllm-xpu-kernels": args.kernels_commit,
            },
            expected_package=str(args.expected_package),
            expected_native_library=str(args.native_library),
            expected_native_library_sha256=preflight["native_library_sha256"],
        )
    except GateError as exc:
        raise CorrectnessError(str(exc)) from exc
    preflight["primitive_imports"] = probe_primitive_imports(args)
    args.resolved_launchers = {
        launcher: resolve_launcher(launcher, args)
        for launcher in dict.fromkeys(
            launcher_name(spec, args) for spec in SERVICE_PLAN
        )
    }
    session = {
        "schema_version": 1,
        "status": "planned" if args.plan_only else "running",
        "created_at": perf.utc_timestamp(),
        "candidate_id": str(args.candidate_env),
        "preflight": preflight,
        "source_identity": args.source_identity,
        "factory_qualification": args.factory_qualification,
        "primitive_plan": [
            {"gate": gate, "pytest_expression": expression}
            for gate, expression in PRIMITIVE_PLAN
        ],
        "native_layout": args.native_layout,
        "native_kernel_variant": args.native_kernel_variant,
        "native_kernel_variant_id": perf.NATIVE_KERNEL_VARIANTS[
            args.native_kernel_variant
        ],
        "native_nominal_splits_by_batch": {
            str(batch): splits for batch, splits in sorted(args.native_splits.items())
        },
        "native_output_dtype": args.native_output_dtype,
        "native_split_policy": args.native_split_policy,
        "flush_index_materialization": (
            perf.flush_index_materialization_environment(args)
        ),
        "native_frontend": perf.native_frontend_environment(args),
        "service_controls": {
            "kvarn_flush_index_materialization": (
                perf.flush_index_materialization_environment(args)
            ),
            "kvarn_onednn_deterministic": "1",
            "kvarn_native_frontend": perf.native_frontend_environment(args),
            "vllm_use_v2_model_runner": perf.VLLM_USE_V2_MODEL_RUNNER,
        },
        "native_scratch_max_splits": (
            perf.B70_Q6_MAX_SPLITS
            if args.native_split_policy == "b70_q6"
            else max(args.native_splits.values())
        ),
        **candidate_variant_provenance(args),
        "service_start_plan": [
            service_spec_evidence(spec, args) for spec in SERVICE_PLAN
        ],
        "resolved_launchers": args.resolved_launchers,
    }
    session_path = args.output_dir / "session.json"
    write_json_atomic(session_path, session)
    if args.plan_only:
        write_checksums(args.output_dir)
        return session

    states = assert_units_inactive(args.require_inactive_unit)
    perf.assert_port_unused(args.base_url)
    session["inactive_units"] = states
    write_json_atomic(session_path, session)
    fixtures = tokenize_fixtures(args)
    gate_paths = {
        gate: run_primitive_gate(gate, expression, args)
        for gate, expression in PRIMITIVE_PLAN
    }
    gate_paths.update(run_service_gates(fixtures, args))
    manifest = build_manifest(args, gate_paths)
    manifest_path = args.output_dir / "native-correctness.json"
    write_json_atomic(manifest_path, manifest)
    session.update(
        status="passed",
        finished_at=perf.utc_timestamp(),
        correctness_manifest=artifact(manifest_path),
    )
    write_json_atomic(session_path, session)
    write_checksums(args.output_dir)
    return session


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-env", type=Path, required=True)
    parser.add_argument(
        "--factory-result",
        type=Path,
        required=True,
        help="completed direct XPU factory artifact for the selected kernel ID",
    )
    parser.add_argument("--primitive-python", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--served-model", default="sunny-chat")
    parser.add_argument(
        "--native-layout",
        choices=perf.NATIVE_LAYOUTS,
        required=True,
        help=(
            "native service cache layout; xe2_dpas requires dedicated Brutus "
            "variant-specific native-dpas launcher outputs"
        ),
    )
    parser.add_argument(
        "--native-kernel-variant",
        choices=tuple(perf.NATIVE_KERNEL_VARIANTS),
        required=True,
        help="engine-lifetime native decoder specialization",
    )
    parser.add_argument(
        "--native-split-policy",
        choices=perf.NATIVE_SPLIT_POLICIES,
        required=True,
        help="engine-lifetime split policy",
    )
    parser.add_argument(
        "--native-output-dtype",
        choices=("fp16", "bf16"),
        default="bf16",
        help="direct native output path that must be qualified (default: bf16)",
    )
    parser.add_argument(
        "--flush-index-materialization",
        choices=perf.FLUSH_INDEX_MATERIALIZATION_VARIANTS,
        default="per_layer",
        help=(
            "engine-lifetime flush-index strategy for all Kvarn correctness "
            "services (default: per_layer)"
        ),
    )
    parser.add_argument(
        "--native-frontend",
        choices=perf.NATIVE_FRONTEND_VARIANTS,
        default="reference",
        help="engine-lifetime Q/K/V frontend for native correctness services",
    )
    parser.add_argument(
        "--native-splits",
        action="append",
        metavar="SPLITS|BATCH=SPLITS",
        help="fixed maximum split count; repeat BATCH=SPLITS for B1/B4",
    )
    parser.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    parser.add_argument("--cancel-after-events", type=int, default=257)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--config-ref", default="path:/home/jasonbk/.config/nix")
    parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "fixtures/kvarn-long-generation.json",
    )
    parser.add_argument(
        "--runtime-cache",
        type=Path,
        default=Path("benchmark-results/kvarn-runtime-cache"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("/var/cache/huggingface"))
    parser.add_argument("--packaging-commit")
    parser.add_argument("--vllm-commit")
    parser.add_argument("--kernels-commit")
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--startup-attempts", type=int, default=1)
    parser.add_argument("--readiness-poll-interval", type=float, default=2.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--cancellation-idle-timeout", type=float, default=300.0)
    parser.add_argument("--primitive-timeout", type=float, default=1800.0)
    parser.add_argument("--tokenizer-timeout", type=float, default=600.0)
    parser.add_argument("--metrics-poll-interval", type=float, default=0.1)
    parser.add_argument(
        "--require-inactive-unit",
        action="append",
        default=None,
        help="systemd unit that must be inactive; repeat as needed",
    )
    parser.add_argument(
        "--packaging-repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--kernels-repo",
        type=Path,
        default=Path("/home/jasonbk/Projects/vllm-xpu-kernels"),
    )
    parser.add_argument(
        "--config-repo", type=Path, default=Path("/home/jasonbk/.config/nix")
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.native_output_dtype != "bf16":
            raise CorrectnessError(
                "finalist service qualification requires --native-output-dtype bf16"
            )
        if (
            args.native_layout != "xe2_dpas"
            or args.native_kernel_variant not in perf.B70_Q6_KERNEL_VARIANTS
            or args.native_split_policy != "b70_q6"
        ):
            raise CorrectnessError(
                "Round-2 finalist qualification requires xe2_dpas, a Q6 kernel "
                "variant, and b70_q6; natural/fixed is reference-only"
            )
        if args.native_split_policy == "b70_q6":
            if args.native_splits:
                raise CorrectnessError(
                    "--native-splits must be absent with --native-split-policy b70_q6"
                )
            if args.native_kernel_variant not in perf.B70_Q6_KERNEL_VARIANTS:
                raise CorrectnessError(
                    "b70_q6 split policy requires a q6 native kernel variant"
                )
            args.native_splits = dict(perf.B70_Q6_SPLITS)
        else:
            args.native_splits = perf._parse_native_splits(args.native_splits, (1, 4))
        args.candidate_env = args.candidate_env.expanduser().resolve()
        args.factory_result = args.factory_result.expanduser().resolve()
        args.primitive_python = (
            args.primitive_python.expanduser().resolve()
            if args.primitive_python is not None
            else args.candidate_env / "bin/python"
        )
        args.output_dir = perf.ensure_durable(args.output_dir, allow_tmp=args.allow_tmp)
        args.fixtures = args.fixtures.expanduser().resolve()
        args.runtime_cache = args.runtime_cache.expanduser().resolve()
        args.hf_home = args.hf_home.expanduser().resolve()
        args.packaging_repo = args.packaging_repo.expanduser().resolve()
        args.kernels_repo = args.kernels_repo.expanduser().resolve()
        args.config_repo = args.config_repo.expanduser().resolve()
        if not args.config_ref.startswith("path:"):
            raise CorrectnessError("--config-ref must be a local path: reference")
        config_ref_path = Path(args.config_ref.removeprefix("path:")).expanduser()
        if config_ref_path.resolve() != args.config_repo:
            raise CorrectnessError(
                "--config-ref and --config-repo must identify one tree"
            )
        args.config_ref = f"path:{args.config_repo}"
        args.require_inactive_unit = list(
            dict.fromkeys(
                (*REQUIRED_INACTIVE_UNITS, *(args.require_inactive_unit or ()))
            )
        )
        if not (args.candidate_env / "bin/vllm").is_file():
            raise CorrectnessError("--candidate-env must contain bin/vllm")
        if not args.primitive_python.is_file():
            raise CorrectnessError(
                "--primitive-python must be a wrapped Python containing pytest, "
                "transformers, torch XPU, and the candidate kernel package"
            )
        if not args.factory_result.is_file():
            raise CorrectnessError("--factory-result must be a readable JSON file")
        if not args.fixtures.is_file():
            raise CorrectnessError("--fixtures must be a readable JSON file")
        if sha256_file(args.fixtures) != DEFAULT_FIXTURE_SHA256:
            raise CorrectnessError("the deterministic base fixture SHA-256 differs")
        if args.output_tokens != 512:
            raise CorrectnessError("the near-262K gate requires exactly 512 outputs")
        if args.max_num_batched_tokens != 2048:
            raise CorrectnessError("the beta correctness profile requires MNBT=2048")
        if args.cancel_after_events != 257:
            raise CorrectnessError(
                "the cancellation gate requires the exact 257-token checkpoint"
            )
        if (
            args.startup_attempts != 1
            or min(
                args.startup_timeout,
                args.readiness_poll_interval,
                args.shutdown_timeout,
                args.request_timeout,
                args.cancellation_idle_timeout,
                args.primitive_timeout,
                args.tokenizer_timeout,
                args.metrics_poll_interval,
            )
            <= 0
        ):
            raise CorrectnessError(
                "startup-attempts must be 1 and all timeouts must be positive"
            )
        for commit in (
            args.packaging_commit,
            args.vllm_commit,
            args.kernels_commit,
        ):
            if commit is not None and not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise CorrectnessError(
                    "explicit source revisions must be 40 lowercase hex digits"
                )
        if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
            raise CorrectnessError("model revision must be 40 lowercase hex digits")
    except (CorrectnessError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    return args


def record_failure(output_dir: Path, exc: BaseException) -> dict[str, Any]:
    session_path = output_dir / "session.json"
    session: dict[str, Any] = {"schema_version": 1}
    if session_path.is_file():
        try:
            existing = json.loads(session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict):
            session.update(existing)
    session.update(
        status="failed",
        finished_at=perf.utc_timestamp(),
        error=f"{type(exc).__name__}: {exc}",
    )
    write_json_atomic(session_path, session)
    write_checksums(output_dir)
    return session


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.supervisor = perf.ProcessSupervisor()
    args.supervisor.install_signal_handlers()
    session: dict[str, Any]
    try:
        try:
            session = execute(args)
        except (KeyboardInterrupt, perf.RunnerInterrupted) as exc:
            session = record_failure(args.output_dir, exc)
            return 130
        except Exception as exc:  # noqa: BLE001 - preserve fail-closed evidence.
            session = record_failure(args.output_dir, exc)
            print(f"error: {exc}", file=sys.stderr)
            return 2
    finally:
        args.supervisor.signal_all(signal.SIGTERM)
        args.supervisor.restore_signal_handlers()
    json.dump(session, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "_tokenize_worker":
        raise SystemExit(tokenize_worker(sys.argv[2:]))
    raise SystemExit(main())
