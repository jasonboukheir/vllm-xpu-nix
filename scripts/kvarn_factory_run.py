#!/usr/bin/env python3
"""Run Kvarn kernel-factory candidates on one verified Intel B70.

This is a primitive/device-stage diagnostic. It invokes registered Torch XPU
operators directly and cannot establish vLLM service throughput, latency, or
parity. Natural-layout Kvarn is the correctness reference and FA2 over the
auto BF16 cache is the performance control.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import gc
import hashlib
import importlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

EXPECTED_DEVICE_NAME = "Intel(R) Arc(TM) Pro B70 Graphics"
H_Q = 24
H_KV = 4
HEAD_DIM = 256
KVARN_PAGE = 128
KVARN_RECORD_STRIDE = 35_072
SOFTMAX_SCALE = 1.0 / 16.0
VALID_SPLITS = (1, 2, 4, 8, 16, 17, 24, 32)
MATCHED_FIXTURE_MODE = "matched-production"
UNMATCHED_FIXTURE_MODE = "unmatched-diagnostic"
CORPUS_SEED = 20_260_903
CORPUS_PACK_TILE_BATCH = 256
CORPUS_ROTATE_ROW_BATCH = 8_192
SCOPE_WARNING = (
    "Primitive/device-stage diagnostic only: these measurements exclude the "
    "vLLM scheduler, model layers, service transport, and Kvarn page flushes. "
    "They are not service-performance or parity evidence."
)
GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
RUNTIME_ENVIRONMENT_KEYS = (
    "CC",
    "CMPLR_ROOT",
    "CXX",
    "LD_LIBRARY_PATH",
    "LEVEL_ZERO_V1_SDK_PATH",
    "LIBRARY_PATH",
    "ONEAPI_ROOT",
    "PATH",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD",
    "SYCL_HOME",
)


class FactoryError(RuntimeError):
    """Raised when the factory run cannot produce valid device evidence."""


@dataclasses.dataclass(frozen=True)
class VariantSpec:
    variant_id: int
    name: str
    description: str
    dpas_layout: bool = True


VARIANTS = {
    spec.name: spec
    for spec in (
        VariantSpec(0, "baseline", "q8 scalar packed-word loads"),
        VariantSpec(1, "qk_i8u4", "integer q/k dot-product candidate"),
        VariantSpec(2, "q6_scalar", "q6 scalar packed-word loads"),
        VariantSpec(3, "q8_vector", "q8 vector packed-word loads"),
        VariantSpec(4, "q6_vector", "q6 vector packed-word loads"),
        VariantSpec(5, "page128", "128-token decode-tile candidate"),
    )
}
VARIANTS_BY_ID = {spec.variant_id: spec for spec in VARIANTS.values()}
DEFAULT_VARIANT_NAMES = (
    "baseline",
    "qk_i8u4",
    "q6_scalar",
    "q8_vector",
    "q6_vector",
)
FOCUSED_XPU_TESTS = (
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_structured_permuted_pages",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_nonuniform_kvarn_factors_across_page_boundary",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_round1_dpas_variants_match_canonical_ragged_and_hybrid",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_q6_multisplit_lse_owns_all_six_distinct_query_rows",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_full_precision_tail_and_packed_history_share_softmax",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[natural-split24]",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[dpas-split1]",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[dpas-split24]",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[dpas-qk-i8u4-split24]",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[r1-p2-dpas-q6]",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[r1-p5-dpas-vector-load]",
    "tests/flash_attn/test_kvarn_decode_xpu.py::test_long_context_ragged_b4_matches_structured_oracle[r1-p2-p5-dpas-q6-vector-load]",
    "tests/flash_attn/test_kvarn_hadamard_scatter_xpu.py::test_kvarn_fused_qkv_hadamard_scatter_matches_separate_ops",
)
FOCUSED_XPU_MIN_PASSED = 41
UNMATCHED_FIXTURE_WARNING = (
    "The auto BF16 cache and Kvarn packed cache have identical model shapes, "
    "batch/context, query seed, and warmed execution, but do not yet originate "
    "from one logical K/V corpus. Kvarn also repeats two packed records while "
    "auto uses unique Gaussian storage. Ratios are candidate-ranking diagnostics "
    "only and are not matched-parity evidence. A production Kvarn packer fixture "
    "is required before promoting these ratios to a parity gate."
)


@dataclasses.dataclass(frozen=True)
class TailFixture:
    block_to_slot: Any
    tail_key: Any
    tail_value: Any
    provenance: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class MatchedCorpus:
    auto_base: Any
    auto_key_cache: Any
    auto_value_cache: Any
    auto_block_table: Any
    natural_cache: Any
    dpas_cache: Any
    native_block_table: Any
    tail_fixtures: dict[int, TailFixture]
    frontier_key: dict[int, Any]
    frontier_value: dict[int, Any]
    max_batch: int
    max_context: int
    auto_block_size: int
    provenance: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class MatrixCase:
    batch: int
    context: int
    requested_splits: int
    effective_splits: int
    variant: VariantSpec

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch": self.batch,
            "context": self.context,
            "requested_num_kv_splits": self.requested_splits,
            "effective_num_kv_splits": self.effective_splits,
            "kernel_variant": self.variant.variant_id,
            "variant_name": self.variant.name,
            "dpas_layout": self.variant.dpas_layout,
        }


def utc_now() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FactoryError(f"shared library is not a regular file: {resolved}")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable):
        raise FactoryError(f"shared library changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": digest,
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
    }


def ensure_durable_output(path: Path, *, allow_tmp: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.suffix != ".json":
        raise FactoryError("--output must name a JSON file")
    if not allow_tmp and resolved.is_relative_to(Path("/tmp")):
        raise FactoryError(f"--output must be durable (outside /tmp): {resolved}")
    return resolved


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def closure_digest(paths: Sequence[str]) -> str:
    canonical = "\n".join(sorted(set(paths))) + "\n"
    return sha256_text(canonical)


def nix_store_root(path: Path) -> Path:
    parts = path.parts
    if len(parts) < 4 or parts[:3] != ("/", "nix", "store"):
        raise FactoryError(f"shared library is not in the Nix store: {path}")
    return Path(*parts[:4])


def verify_nix_artifact(
    *,
    label: str,
    library: Path,
    attestation: dict[str, str],
    command_runner: Callable[[Sequence[str]], str] = _run,
) -> dict[str, Any]:
    # execute() has already opened and hashed this path with strict resolution.
    # Keeping this helper syntactic also makes its Nix query contract testable
    # without creating files under /nix/store.
    resolved = library.expanduser().resolve()
    output_path = nix_store_root(resolved)
    derivation = attestation["derivation"]
    try:
        outputs = sorted(
            set(
                command_runner(
                    ("nix-store", "-q", "--outputs", derivation)
                ).splitlines()
            )
        )
        actual_deriver = command_runner(
            ("nix-store", "-q", "--deriver", str(output_path))
        )
        closure = sorted(
            set(command_runner(("nix-store", "-qR", str(output_path))).splitlines())
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise FactoryError(f"cannot verify {label} Nix artifact: {error}") from error
    if str(output_path) not in outputs:
        raise FactoryError(
            f"{label} library output {output_path} is not produced by {derivation}"
        )
    if actual_deriver != derivation:
        raise FactoryError(
            f"{label} output deriver mismatch: expected {derivation}, "
            f"got {actual_deriver!r}"
        )
    if not closure:
        raise FactoryError(f"{label} Nix output has an empty closure")
    actual_digest = closure_digest(closure)
    if actual_digest != attestation["closure_sha256"]:
        raise FactoryError(
            f"{label} closure digest mismatch: expected "
            f"{attestation['closure_sha256']}, got {actual_digest}"
        )
    return {
        **attestation,
        "attestation_source": "verified_nix_store",
        "library_path": str(resolved),
        "output_path": str(output_path),
        "actual_deriver": actual_deriver,
        "closure_paths": closure,
        "closure_sha256": actual_digest,
        "verified": True,
    }


def repository_state(name: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    status = _run(
        (
            "git",
            "-C",
            str(resolved),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    ).splitlines()
    return {
        "name": name,
        "path": str(resolved),
        "head": _run(("git", "-C", str(resolved), "rev-parse", "HEAD")),
        "branch": _run(("git", "-C", str(resolved), "branch", "--show-current"))
        or None,
        "status_porcelain": status,
        "status_sha256": sha256_text("\n".join(status)),
        "dirty": bool(status),
    }


def require_clean_repositories(repositories: Sequence[dict[str, Any]]) -> None:
    dirty = [value["name"] for value in repositories if value["dirty"]]
    if dirty:
        raise FactoryError(
            "factory evidence requires clean source repositories; dirty: "
            + ", ".join(dirty)
        )


def require_expected_repository_revisions(
    repositories: Sequence[dict[str, Any]], expected: dict[str, str]
) -> dict[str, str]:
    actual = {value["name"]: value["head"] for value in repositories}
    if set(actual) != set(expected):
        raise FactoryError(
            "source revision contract does not match the repository set: "
            f"expected {sorted(expected)}, got {sorted(actual)}"
        )
    for name, revision in expected.items():
        if not GIT_COMMIT.fullmatch(revision):
            raise FactoryError(f"expected {name} revision is not a full Git commit")
        if actual[name] != revision:
            raise FactoryError(
                f"{name} revision mismatch: expected {revision}, got {actual[name]}"
            )
    return actual


def runtime_environment_contract() -> dict[str, Any]:
    prefixed = {
        name: value
        for name, value in sorted(os.environ.items())
        if name.startswith(("KVARN_", "VLLM_"))
    }
    return {
        "kvarn_or_vllm_prefixed_variables": prefixed,
        "prefixed_environment_clean": not prefixed,
        "pinned_runtime_environment": {
            name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT_KEYS
        },
    }


def evidence_identity(
    libraries: dict[str, dict[str, Any]], repositories: Sequence[dict[str, Any]]
) -> str:
    stable = {
        "libraries": {
            name: {
                key: value[key] for key in ("path", "sha256", "size_bytes", "mtime_ns")
            }
            for name, value in sorted(libraries.items())
        },
        "repositories": [
            {
                "name": value["name"],
                "path": value["path"],
                "head": value["head"],
                "branch": value["branch"],
                "status_sha256": value["status_sha256"],
            }
            for value in repositories
        ],
    }
    return sha256_text(json.dumps(stable, sort_keys=True, separators=(",", ":")))


def validate_build_attestation(
    *, label: str, derivation: str, closure_sha256: str
) -> dict[str, str]:
    if not re.fullmatch(r"/nix/store/[a-z0-9]{32}-[^/]+\.drv", derivation):
        raise FactoryError(f"--{label}-derivation must be an absolute Nix .drv path")
    if not re.fullmatch(r"[0-9a-f]{64}", closure_sha256):
        raise FactoryError(f"--{label}-closure-sha256 must be 64 lowercase hex")
    return {
        "derivation": derivation,
        "closure_sha256": closure_sha256,
        "attestation_source": "explicit_cli",
    }


def parse_int_list(value: str, *, label: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise FactoryError(f"{label} must be a comma-separated integer list") from error
    if not parsed or len(parsed) != len(value.split(",")):
        raise FactoryError(f"{label} must not contain empty entries")
    return list(dict.fromkeys(parsed))


def parse_variants(value: str) -> list[VariantSpec]:
    names = list(DEFAULT_VARIANT_NAMES) if value == "all" else value.split(",")
    if not names or any(not name for name in names):
        raise FactoryError("--variants must contain named candidate IDs")
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise FactoryError(
            f"unknown variant {unknown[0]!r}; choose from {', '.join(VARIANTS)} or all"
        )
    return [VARIANTS[name] for name in dict.fromkeys(names)]


def parse_split_tokens(value: str) -> list[int | None]:
    tokens = value.split(",")
    if not tokens or any(not token for token in tokens):
        raise FactoryError("--splits must not contain empty entries")
    result: list[int | None] = []
    for token in tokens:
        if token == "auto":
            split = None
        else:
            try:
                split = int(token)
            except ValueError as error:
                raise FactoryError(
                    "--splits accepts auto or comma-separated integers"
                ) from error
            if split not in VALID_SPLITS:
                choices = ", ".join(str(item) for item in VALID_SPLITS)
                raise FactoryError(f"unsupported split {split}; choose from {choices}")
        if split not in result:
            result.append(split)
    return result


def effective_split_count(context: int, requested: int) -> int:
    kv_tiles = math.ceil(context / 64)
    return 1 if requested > 1 and kv_tiles < requested else requested


def build_matrix(
    *,
    batches: Sequence[int],
    contexts: Sequence[int],
    splits: Sequence[int | None],
    variants: Sequence[VariantSpec],
) -> list[MatrixCase]:
    if any(batch not in (1, 4) for batch in batches):
        raise FactoryError("this factory runner supports only B1 and B4")
    if any(context <= 0 for context in contexts):
        raise FactoryError("contexts must be positive")
    cases: list[MatrixCase] = []
    seen: set[tuple[int, int, int, int]] = set()
    for context in contexts:
        for batch in batches:
            for requested_value in splits:
                requested = (
                    (24 if batch == 1 else 16)
                    if requested_value is None
                    else requested_value
                )
                for variant in variants:
                    key = (batch, context, requested, variant.variant_id)
                    if key in seen:
                        continue
                    seen.add(key)
                    cases.append(
                        MatrixCase(
                            batch=batch,
                            context=context,
                            requested_splits=requested,
                            effective_splits=effective_split_count(context, requested),
                            variant=variant,
                        )
                    )
    return cases


def interleaved_order(names: Sequence[str], round_index: int) -> tuple[str, ...]:
    if not names:
        raise FactoryError("at least one timing arm is required")
    offset = round_index % len(names)
    rotated = [*names[offset:], *names[:offset]]
    return (*rotated, *reversed(rotated))


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise FactoryError("cannot summarize an empty timing sample")
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def timing_summary(device_us: list[float], wall_us: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(device_us),
        "device_us": device_us,
        "wall_us": wall_us,
        "device_median_us": statistics.median(device_us),
        "device_p10_us": percentile(device_us, 0.10),
        "device_p90_us": percentile(device_us, 0.90),
        "wall_median_us": statistics.median(wall_us),
    }


def measure_interleaved(
    torch_module: Any,
    arms: dict[str, Callable[[], None]],
    *,
    warmup_rounds: int,
    sample_rounds: int,
) -> dict[str, Any]:
    names = list(arms)
    for round_index in range(warmup_rounds):
        for name in interleaved_order(names, round_index):
            arms[name]()
    torch_module.xpu.synchronize()

    samples = {name: {"device": [], "wall": []} for name in names}
    sample_orders: list[list[str]] = []
    for round_index in range(sample_rounds):
        order = interleaved_order(names, round_index)
        sample_orders.append(list(order))
        for name in order:
            start = torch_module.xpu.Event(enable_timing=True)
            end = torch_module.xpu.Event(enable_timing=True)
            torch_module.xpu.synchronize()
            wall_start = time.perf_counter_ns()
            start.record()
            arms[name]()
            end.record()
            torch_module.xpu.synchronize()
            samples[name]["device"].append(start.elapsed_time(end) * 1000.0)
            samples[name]["wall"].append((time.perf_counter_ns() - wall_start) / 1000.0)
    return {
        "order_policy": "rotating palindromic ABBA/interleaved",
        "warmup_rounds": warmup_rounds,
        "sample_rounds": sample_rounds,
        "samples_per_arm": sample_rounds * 2,
        "sample_orders": sample_orders,
        "arms": {
            name: timing_summary(values["device"], values["wall"])
            for name, values in samples.items()
        },
    }


def validate_device_identity(
    *, available: bool, count: int, selected_name: str
) -> None:
    if not available or count < 1:
        raise FactoryError("a real XPU device is required")
    if selected_name != EXPECTED_DEVICE_NAME:
        raise FactoryError(
            f"expected exact device {EXPECTED_DEVICE_NAME!r}, got {selected_name!r}"
        )


def preflight_xpu(torch_module: Any) -> dict[str, Any]:
    available = bool(torch_module.xpu.is_available())
    count = int(torch_module.xpu.device_count()) if available else 0
    name = torch_module.xpu.get_device_name(0) if count else ""
    validate_device_identity(available=available, count=count, selected_name=name)
    try:
        probe = torch_module.arange(1, 9, dtype=torch_module.float32, device="xpu:0")
        probe_result = (probe.square() + 1).sum()
        torch_module.xpu.synchronize()
        value = float(probe_result.cpu().item())
    except Exception as error:
        raise FactoryError(f"real XPU tensor-op probe failed: {error}") from error
    if probe.device.type != "xpu" or value != 212.0:
        raise FactoryError(
            f"real XPU tensor-op probe returned invalid evidence: {value}"
        )
    return {
        "xpu_available": available,
        "xpu_device_count": count,
        "selected_device": "xpu:0",
        "selected_device_name": name,
        "tensor_op": "sum(square(arange(1, 9, xpu:0)) + 1)",
        "tensor_op_result": value,
        "passed": True,
    }


def focused_xpu_test_command(kernels_repo: Path) -> list[str]:
    resolved = kernels_repo.expanduser().resolve(strict=True)
    node_ids: list[str] = []
    for selection in FOCUSED_XPU_TESTS:
        relative, separator, test_name = selection.partition("::")
        if not separator:
            raise FactoryError(f"invalid focused XPU test selection: {selection}")
        source = (resolved / relative).resolve(strict=True)
        if not source.is_relative_to(resolved):
            raise FactoryError(f"focused XPU test escapes kernel repository: {source}")
        node_ids.append(f"{source}::{test_name}")
    return [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "-rA",
        *node_ids,
    ]


def run_focused_xpu_kill_suite(
    *, kernels_repo: Path, flash_library: Path
) -> dict[str, Any]:
    resolved_repo = kernels_repo.expanduser().resolve(strict=True)
    resolved_library = flash_library.expanduser().resolve(strict=True)
    command = focused_xpu_test_command(resolved_repo)
    environment = os.environ.copy()
    environment["VLLM_XPU_KERNELS_LIBRARY"] = str(resolved_library)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    started_at = utc_now()
    monotonic_start = time.monotonic_ns()
    try:
        completed = subprocess.run(
            command,
            cwd=resolved_repo,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return {
            "status": "failed",
            "passed": False,
            "started_at": started_at,
            "finished_at": utc_now(),
            "duration_seconds": (time.monotonic_ns() - monotonic_start) / 1e9,
            "command": command,
            "cwd": str(resolved_repo),
            "library": str(resolved_library),
            "error": str(error),
        }
    combined_output = f"{completed.stdout}\n{completed.stderr}"
    passed_matches = re.findall(r"(?m)(\d+) passed(?:[, ]|$)", combined_output)
    passed_count = max((int(value) for value in passed_matches), default=0)
    skipped_matches = re.findall(r"(?m)(\d+) skipped(?:[, ]|$)", combined_output)
    skipped_count = max((int(value) for value in skipped_matches), default=0)
    passed = (
        completed.returncode == 0
        and passed_count >= FOCUSED_XPU_MIN_PASSED
        and skipped_count == 0
    )
    return {
        "status": "passed" if passed else "failed",
        "passed": passed,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": (time.monotonic_ns() - monotonic_start) / 1e9,
        "command": command,
        "cwd": str(resolved_repo),
        "library": str(resolved_library),
        "library_environment": "VLLM_XPU_KERNELS_LIBRARY",
        "minimum_passed_required": FOCUSED_XPU_MIN_PASSED,
        "passed_count": passed_count,
        "skipped_count": skipped_count,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def require_focused_xpu_kill_suite(result: dict[str, Any]) -> None:
    if result.get("passed") is not True:
        raise FactoryError(
            "mandatory focused B70 XPU kernel kill suite failed; "
            f"returncode={result.get('returncode')}, "
            f"passed={result.get('passed_count')}, "
            f"skipped={result.get('skipped_count')}"
        )


def _operator(torch_module: Any, namespace: str, name: str) -> tuple[Any, str]:
    try:
        packet = getattr(getattr(torch_module.ops, namespace), name)
        schema = str(packet.default._schema)
    except (AttributeError, RuntimeError) as error:
        raise FactoryError(
            f"required operator is missing: {namespace}::{name}"
        ) from error
    return packet, schema


def _optional_operator(
    torch_module: Any, namespace: str, name: str
) -> tuple[Any | None, str | None]:
    try:
        packet = getattr(getattr(torch_module.ops, namespace), name)
        return packet, str(packet.default._schema)
    except (AttributeError, RuntimeError):
        return None, None


def load_operators(
    torch_module: Any, *, base_library: Path, flash_library: Path
) -> tuple[SimpleNamespace, dict[str, str | None]]:
    if base_library.resolve() == flash_library.resolve():
        raise FactoryError("base and flash libraries must be distinct files")
    torch_module.ops.load_library(str(base_library))
    torch_module.ops.load_library(str(flash_library))
    cache_store, cache_schema = _operator(
        torch_module, "_C_cache_ops", "reshape_and_cache_flash"
    )
    auto_decode, auto_schema = _operator(torch_module, "_vllm_fa2_C", "varlen_fwd")
    native_decode, native_schema = _operator(
        torch_module, "_vllm_fa2_C", "kvarn_decode_with_scratch"
    )
    hadamard, hadamard_schema = _operator(torch_module, "_vllm_fa2_C", "kvarn_hadamard")
    scatter, scatter_schema = _operator(
        torch_module, "_vllm_fa2_C", "kvarn_hadamard_scatter"
    )
    fused, fused_schema = _optional_operator(
        torch_module, "_vllm_fa2_C", "kvarn_hadamard_qkv_scatter"
    )
    for argument in ("num_kv_splits", "kernel_variant", "dpas_layout"):
        if argument not in native_schema:
            raise FactoryError(
                "native decode library lacks the explicit factory ABI argument "
                f"{argument!r}"
            )
    if "dpas_layout" not in scatter_schema:
        raise FactoryError("Kvarn scatter library lacks explicit dpas_layout ABI")
    if fused_schema is not None and "dpas_layout" not in fused_schema:
        raise FactoryError("fused QKV scatter lacks explicit dpas_layout ABI")
    return (
        SimpleNamespace(
            auto_store=cache_store,
            auto_decode=auto_decode,
            native_decode=native_decode,
            hadamard=hadamard,
            scatter=scatter,
            fused_qkv_scatter=fused,
        ),
        {
            "reshape_and_cache_flash": cache_schema,
            "varlen_fwd": auto_schema,
            "kvarn_decode_with_scratch": native_schema,
            "kvarn_hadamard": hadamard_schema,
            "kvarn_hadamard_scatter": scatter_schema,
            "kvarn_hadamard_qkv_scatter": fused_schema,
        },
    )


def load_fixture_helpers(kernels_repo: Path) -> SimpleNamespace:
    resolved = kernels_repo.expanduser().resolve(strict=True)
    sys.path.insert(0, str(resolved))
    try:
        checks = importlib.import_module("benchmark.check_kvarn_decode")
        layouts = importlib.import_module("benchmark.kvarn_utils")
    finally:
        sys.path.pop(0)
    _require_module_from_repo(checks, resolved, Path("benchmark/check_kvarn_decode.py"))
    _require_module_from_repo(layouts, resolved, Path("benchmark/kvarn_utils.py"))
    return SimpleNamespace(
        make_cache=checks.make_cache,
        make_random_cache=checks.make_random_cache,
        swizzle_record=layouts.swizzle_record_dpas_k4v4,
    )


def load_production_packers(vllm_repo: Path) -> SimpleNamespace:
    """Load the attested production Sinkhorn and batched RTN packers.

    Importing these from the source repository (and verifying every module's
    path) prevents a stale site-packages copy from silently defining the
    benchmark corpus ABI.
    """
    resolved = vllm_repo.expanduser().resolve(strict=True)
    sys.path.insert(0, str(resolved))
    try:
        config_module = importlib.import_module(
            "vllm.model_executor.layers.quantization.kvarn.config"
        )
        sinkhorn_module = importlib.import_module(
            "vllm.v1.attention.ops.triton_kvarn_sinkhorn"
        )
        store_module = importlib.import_module("vllm.v1.attention.ops.kvarn_store")
    finally:
        sys.path.pop(0)
    module_paths = {
        "config": Path("vllm/model_executor/layers/quantization/kvarn/config.py"),
        "sinkhorn": Path("vllm/v1/attention/ops/triton_kvarn_sinkhorn.py"),
        "store": Path("vllm/v1/attention/ops/kvarn_store.py"),
    }
    modules = {
        "config": config_module,
        "sinkhorn": sinkhorn_module,
        "store": store_module,
    }
    for name, module in modules.items():
        _require_module_from_repo(module, resolved, module_paths[name])
    sources = {
        name: stable_file_record((resolved / relative).resolve(strict=True))
        for name, relative in module_paths.items()
    }
    return SimpleNamespace(
        make_config=config_module.KVarNConfig.from_cache_dtype,
        sinkhorn=sinkhorn_module.kvarn_sinkhorn_triton,
        pack_k=store_module.kvarn_store_tile_k_batch_from_sinkhorn,
        pack_v=store_module.kvarn_store_tile_v_batch_from_sinkhorn,
        sources=sources,
    )


def _require_module_from_repo(module: Any, repo: Path, relative_path: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        raise FactoryError(f"fixture helper {module.__name__!r} has no source file")
    loaded = Path(module_file).resolve(strict=True)
    expected = (repo / relative_path).resolve(strict=True)
    if loaded != expected:
        raise FactoryError(
            f"fixture helper {module.__name__!r} came from {loaded}, expected {expected}"
        )


def tail_page_assignments(
    *, batch: int, context: int, pages_per_request: int
) -> list[dict[str, Any]]:
    """Describe the live fp16 sink/tail ownership for a decode snapshot."""
    if batch <= 0 or context <= 0 or pages_per_request <= 0:
        raise FactoryError("tail assignment dimensions must be positive")
    used_pages = math.ceil(context / KVARN_PAGE)
    if used_pages > pages_per_request:
        raise FactoryError("tail assignment exceeds the corpus page table")
    result: list[dict[str, Any]] = []
    slot = 0
    for request in range(batch):
        local_pages = [0]
        if used_pages > 1:
            local_pages.append(used_pages - 1)
        for local_page in local_pages:
            valid_tokens = min(
                KVARN_PAGE,
                max(context - local_page * KVARN_PAGE, 0),
            )
            roles = ["sink"] if local_page == 0 else []
            if local_page == used_pages - 1:
                roles.append("current_tail")
            result.append(
                {
                    "request": request,
                    "local_page": local_page,
                    "physical_page": request * pages_per_request + local_page,
                    "pool_slot": slot,
                    "roles": roles,
                    "valid_tokens": valid_tokens,
                }
            )
            slot += 1
    return result


def _update_cpu_tensor_digest(torch_module: Any, digest: Any, tensor: Any) -> int:
    """Hash every byte of one contiguous CPU tensor with bounded scratch."""
    contiguous = tensor.detach().contiguous()
    if contiguous.device.type != "cpu":
        raise FactoryError("logical corpus hashing requires a CPU tensor")
    # Reinterpret through uint8 without requiring NumPy to understand BF16.
    raw = contiguous.view(torch_module.uint8).reshape(-1)
    chunk_bytes = 64 * 1024 * 1024
    for start in range(0, raw.numel(), chunk_bytes):
        digest.update(raw[start : start + chunk_bytes].numpy().tobytes())
    return raw.numel()


def _sha256_device_tensor(torch_module: Any, tensor: Any) -> str:
    """Hash a device tensor in bounded first-axis chunks, outside timing."""
    digest = hashlib.sha256()
    if tensor.ndim == 0:
        chunks = (tensor.reshape(1),)
    else:
        row_bytes = max(1, tensor[0].numel() * tensor.element_size())
        rows = max(1, (64 * 1024 * 1024) // row_bytes)
        chunks = (
            tensor[index : index + rows] for index in range(0, tensor.size(0), rows)
        )
    for chunk in chunks:
        cpu = chunk.detach().contiguous().cpu()
        raw = cpu.view(torch_module.uint8).reshape(-1)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _corpus_identity(provenance: dict[str, Any]) -> str:
    identity = {
        "generator": provenance["generator"],
        "logical_shape": provenance["logical_shape"],
        "logical_dtype": provenance["logical_dtype"],
        "key_sha256": provenance["key_sha256"],
        "value_sha256": provenance["value_sha256"],
        "logical_bytes_hashed": provenance["logical_bytes_hashed"],
    }
    return sha256_text(json.dumps(identity, sort_keys=True, separators=(",", ":")))


def validate_matched_corpus_manifest(manifest: dict[str, Any]) -> None:
    """Fail closed before any matched-corpus result becomes parity eligible."""
    required_hashes = (
        "key_sha256",
        "value_sha256",
        "logical_corpus_sha256",
        "auto_key_cache_sha256",
        "auto_value_cache_sha256",
        "natural_packed_cache_sha256",
        "dpas_packed_cache_sha256",
    )
    if manifest.get("fixture_mode") != MATCHED_FIXTURE_MODE:
        raise FactoryError("matched corpus manifest has the wrong fixture mode")
    for name in required_hashes:
        if not re.fullmatch(r"[0-9a-f]{64}", str(manifest.get(name, ""))):
            raise FactoryError(f"matched corpus manifest lacks a valid {name}")
    if manifest["logical_corpus_sha256"] != _corpus_identity(manifest):
        raise FactoryError("matched logical corpus identity does not verify")
    shape = manifest.get("logical_shape", [])
    if len(shape) != 4 or any(
        not isinstance(value, int) or value <= 0 for value in shape
    ):
        raise FactoryError("matched corpus logical shape is invalid")
    expected_bytes = math.prod(shape) * 2
    byte_counts = manifest.get("logical_bytes_hashed", {})
    if byte_counts != {"key": expected_bytes, "value": expected_bytes}:
        raise FactoryError("matched corpus hashes do not cover every logical BF16 byte")
    invariants = manifest.get("invariants", {})
    required_invariants = (
        "auto_populated_by_reshape_and_cache_flash",
        "natural_populated_by_production_packers",
        "dpas_populated_by_production_packers",
        "natural_and_dpas_share_sinkhorn_results",
        "sink_page_mapped_to_fp16_pool",
        "current_tail_mapped_to_fp16_pool",
        "partial_tail_valid_token_counts_verified",
        "all_setup_and_hashing_outside_timing",
    )
    failed = [name for name in required_invariants if invariants.get(name) is not True]
    if failed:
        raise FactoryError("matched corpus invariants failed: " + ", ".join(failed))
    sources = manifest.get("production_packer_sources", {})
    if set(sources) != {"config", "sinkhorn", "store"}:
        raise FactoryError("matched corpus production packer provenance is incomplete")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))
        for record in sources.values()
    ):
        raise FactoryError("matched corpus production packer source hash is invalid")
    contexts = manifest.get("tail_mapping_by_context", {})
    if not contexts or any(not value.get("validated") for value in contexts.values()):
        raise FactoryError("matched corpus tail mappings are not fully validated")
    for context, value in contexts.items():
        for field in ("block_to_slot_sha256", "tail_key_sha256", "tail_value_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(value.get(field, ""))):
                raise FactoryError(
                    f"matched corpus tail mapping {context} lacks {field}"
                )


def _layout_pattern(pattern: Any, layout: Any, helpers: Any, dpas: bool) -> Any:
    result = pattern.clone()
    if dpas:
        for page in range(result.size(0)):
            for head in range(result.size(1)):
                result[page, head] = helpers.swizzle_record(pattern[page, head], layout)
    return result


def _allocate_packed_cache(pattern: Any, *, total_pages: int, torch_module: Any) -> Any:
    device_pattern = pattern.to(device="xpu:0")
    repeats = math.ceil(total_pages / pattern.size(0))
    return device_pattern.repeat(repeats, 1, 1)[:total_pages].contiguous()


def _assemble_packed_records(
    torch_module: Any,
    config: Any,
    packed_key: dict[str, Any],
    packed_value: dict[str, Any],
) -> Any:
    count = packed_key["q_packed_uint8"].size(0)
    parts = (
        packed_key["q_packed_uint8"].reshape(count, config.k_packed_bytes),
        packed_key["s_col_K"].contiguous().view(torch_module.uint8),
        packed_key["zp_K"].contiguous().view(torch_module.uint8),
        packed_key["s_row_K"].contiguous().view(torch_module.uint8),
        packed_value["q_packed_uint8"].reshape(count, config.v_packed_bytes),
        packed_value["s_col_V"].contiguous().view(torch_module.uint8),
        packed_value["s_row_V"].contiguous().view(torch_module.uint8),
        packed_value["zp_V"].contiguous().view(torch_module.uint8),
    )
    active = torch_module.cat(parts, dim=1)
    if active.size(1) != config.tile_bytes:
        raise FactoryError(
            "production packers produced an invalid active record size: "
            f"{active.size(1)} != {config.tile_bytes}"
        )
    records = torch_module.zeros(
        (count, config.record_bytes),
        dtype=torch_module.uint8,
        device=active.device,
    )
    records[:, : config.tile_bytes] = active
    return records


def _require_same_quantization_metadata(
    torch_module: Any,
    natural_key: dict[str, Any],
    natural_value: dict[str, Any],
    dpas_key: dict[str, Any],
    dpas_value: dict[str, Any],
) -> None:
    for field in ("s_col_K", "zp_K", "s_row_K"):
        if not bool(torch_module.equal(natural_key[field], dpas_key[field])):
            raise FactoryError(f"natural/DPAS K metadata diverged at {field}")
    for field in ("s_col_V", "s_row_V", "zp_V"):
        if not bool(torch_module.equal(natural_value[field], dpas_value[field])):
            raise FactoryError(f"natural/DPAS V metadata diverged at {field}")


def _build_tail_fixture(
    torch_module: Any,
    *,
    rotated_key: Any,
    rotated_value: Any,
    context: int,
    pages_per_request: int,
) -> TailFixture:
    batch = rotated_key.size(0)
    assignments = tail_page_assignments(
        batch=batch,
        context=context,
        pages_per_request=pages_per_request,
    )
    block_to_slot = torch_module.full(
        (batch * pages_per_request,),
        -1,
        dtype=torch_module.int32,
        device="xpu:0",
    )
    tail_key = torch_module.empty(
        (len(assignments), KVARN_PAGE, H_KV, HEAD_DIM),
        dtype=torch_module.float16,
        device="xpu:0",
    )
    tail_value = torch_module.empty_like(tail_key)
    for assignment in assignments:
        request = assignment["request"]
        local_page = assignment["local_page"]
        pool_slot = assignment["pool_slot"]
        physical_page = assignment["physical_page"]
        start = local_page * KVARN_PAGE
        stop = start + KVARN_PAGE
        block_to_slot[physical_page] = pool_slot
        tail_key[pool_slot].copy_(rotated_key[request, start:stop])
        tail_value[pool_slot].copy_(rotated_value[request, start:stop])
    sink_ok = all(
        int(block_to_slot[item["physical_page"]].item()) == item["pool_slot"]
        for item in assignments
        if "sink" in item["roles"]
    )
    tail_ok = all(
        int(block_to_slot[item["physical_page"]].item()) == item["pool_slot"]
        for item in assignments
        if "current_tail" in item["roles"]
    )
    expected_tail_tokens = context % KVARN_PAGE or KVARN_PAGE
    tail_count_ok = all(
        item["valid_tokens"] == expected_tail_tokens
        for item in assignments
        if "current_tail" in item["roles"]
    )
    provenance = {
        "context": context,
        "used_pages_per_request": math.ceil(context / KVARN_PAGE),
        "assignments": assignments,
        "sink_mapping_verified": sink_ok,
        "current_tail_mapping_verified": tail_ok,
        "tail_valid_token_count_verified": tail_count_ok,
        "partial_tail_tokens": expected_tail_tokens,
        "block_to_slot_sha256": _sha256_device_tensor(torch_module, block_to_slot),
        "tail_key_sha256": _sha256_device_tensor(torch_module, tail_key),
        "tail_value_sha256": _sha256_device_tensor(torch_module, tail_value),
        "validated": sink_ok and tail_ok and tail_count_ok,
    }
    if not provenance["validated"]:
        raise FactoryError(
            f"failed to validate fp16 tail mapping for context {context}"
        )
    return TailFixture(block_to_slot, tail_key, tail_value, provenance)


def build_matched_corpus(
    torch_module: Any,
    operations: Any,
    production: Any,
    *,
    batches: Sequence[int],
    contexts: Sequence[int],
    auto_block_size: int,
) -> MatchedCorpus:
    """Materialize one logical BF16 corpus into every timed cache layout."""
    max_batch = max(batches)
    max_context = max(contexts)
    native_pages = math.ceil(max_context / KVARN_PAGE)
    padded_context = native_pages * KVARN_PAGE
    total_native_pages = max_batch * native_pages
    config = production.make_config("kvarn_k4v4_g128_compact", HEAD_DIM)
    actual_config = (
        config.head_dim,
        config.group,
        config.key_bits,
        config.value_bits,
        config.record_bytes,
        config.sink_tokens,
    )
    expected_config = (
        HEAD_DIM,
        KVARN_PAGE,
        4,
        4,
        KVARN_RECORD_STRIDE,
        KVARN_PAGE,
    )
    if actual_config != expected_config:
        raise FactoryError(
            "production compact Kvarn config does not match the factory ABI: "
            f"{actual_config} != {expected_config}"
        )

    logical_shape = (max_batch, max_context, H_KV, HEAD_DIM)
    key_rotated_input = torch_module.zeros(
        (max_batch, padded_context, H_KV, HEAD_DIM),
        dtype=torch_module.bfloat16,
        device="xpu:0",
    )
    value_rotated_input = torch_module.zeros_like(key_rotated_input)
    key_digest = hashlib.sha256()
    value_digest = hashlib.sha256()
    key_bytes = 0
    value_bytes = 0
    key_generator = torch_module.Generator(device="cpu").manual_seed(CORPUS_SEED)
    value_generator = torch_module.Generator(device="cpu").manual_seed(CORPUS_SEED + 1)
    for request in range(max_batch):
        key_cpu = torch_module.randn(
            (max_context, H_KV, HEAD_DIM),
            generator=key_generator,
            dtype=torch_module.bfloat16,
        )
        key_bytes += _update_cpu_tensor_digest(torch_module, key_digest, key_cpu)
        key_rotated_input[request, :max_context].copy_(key_cpu)
        del key_cpu
    for request in range(max_batch):
        value_cpu = torch_module.randn(
            (max_context, H_KV, HEAD_DIM),
            generator=value_generator,
            dtype=torch_module.bfloat16,
        )
        value_bytes += _update_cpu_tensor_digest(torch_module, value_digest, value_cpu)
        value_rotated_input[request, :max_context].copy_(value_cpu)
        del value_cpu

    auto_pages = math.ceil(max_context / auto_block_size)
    auto_base = torch_module.zeros(
        (max_batch * auto_pages, H_KV, auto_block_size, 2 * HEAD_DIM),
        dtype=torch_module.bfloat16,
        device="xpu:0",
    )
    auto_key_cache, auto_value_cache = auto_base.transpose(1, 2).split(HEAD_DIM, dim=-1)
    auto_block_table = torch_module.arange(
        max_batch * auto_pages,
        dtype=torch_module.int32,
        device="xpu:0",
    ).view(max_batch, auto_pages)
    k_scale = torch_module.ones((), dtype=torch_module.float32, device="xpu:0")
    v_scale = torch_module.ones((), dtype=torch_module.float32, device="xpu:0")
    for request in range(max_batch):
        slots = (
            torch_module.arange(max_context, dtype=torch_module.int64, device="xpu:0")
            + request * auto_pages * auto_block_size
        )
        operations.auto_store(
            key_rotated_input[request, :max_context],
            value_rotated_input[request, :max_context],
            auto_key_cache,
            auto_value_cache,
            slots,
            "auto",
            k_scale,
            v_scale,
        )
    torch_module.xpu.synchronize()
    sample_positions = sorted(
        {
            0,
            min(127, max_context - 1),
            min(128, max_context - 1),
            max_context - 1,
        }
    )
    for request in range(max_batch):
        for position in sample_positions:
            page, offset = divmod(position, auto_block_size)
            physical = request * auto_pages + page
            torch_module.testing.assert_close(
                auto_key_cache[physical, offset],
                key_rotated_input[request, position],
                atol=0,
                rtol=0,
            )
            torch_module.testing.assert_close(
                auto_value_cache[physical, offset],
                value_rotated_input[request, position],
                atol=0,
                rtol=0,
            )

    key_rotated = torch_module.empty(
        key_rotated_input.shape,
        dtype=torch_module.float16,
        device="xpu:0",
    )
    value_rotated = torch_module.empty_like(key_rotated)
    flat_key_input = key_rotated_input.reshape(-1, HEAD_DIM)
    flat_value_input = value_rotated_input.reshape(-1, HEAD_DIM)
    flat_key_rotated = key_rotated.reshape(-1, HEAD_DIM)
    flat_value_rotated = value_rotated.reshape(-1, HEAD_DIM)
    for start in range(0, flat_key_input.size(0), CORPUS_ROTATE_ROW_BATCH):
        stop = min(start + CORPUS_ROTATE_ROW_BATCH, flat_key_input.size(0))
        operations.hadamard(flat_key_input[start:stop], flat_key_rotated[start:stop])
        operations.hadamard(
            flat_value_input[start:stop], flat_value_rotated[start:stop]
        )
    torch_module.xpu.synchronize()

    natural_cache = torch_module.empty(
        (total_native_pages, H_KV, config.record_bytes),
        dtype=torch_module.uint8,
        device="xpu:0",
    )
    dpas_cache = torch_module.empty_like(natural_cache)
    key_pages = key_rotated.view(
        max_batch, native_pages, KVARN_PAGE, H_KV, HEAD_DIM
    ).reshape(total_native_pages, KVARN_PAGE, H_KV, HEAD_DIM)
    value_pages = value_rotated.view(
        max_batch, native_pages, KVARN_PAGE, H_KV, HEAD_DIM
    ).reshape(total_native_pages, KVARN_PAGE, H_KV, HEAD_DIM)
    pages_per_pack = max(1, CORPUS_PACK_TILE_BATCH // H_KV)
    for page_start in range(0, total_native_pages, pages_per_pack):
        page_stop = min(page_start + pages_per_pack, total_native_pages)
        page_count = page_stop - page_start
        key_tiles = (
            key_pages[page_start:page_stop]
            .permute(0, 2, 3, 1)
            .reshape(page_count * H_KV, HEAD_DIM, KVARN_PAGE)
            .contiguous()
        )
        value_tiles = (
            value_pages[page_start:page_stop]
            .permute(0, 2, 1, 3)
            .reshape(page_count * H_KV, KVARN_PAGE, HEAD_DIM)
            .contiguous()
        )
        key_balanced, key_s_col, key_s_row = production.sinkhorn(
            key_tiles, iterations=config.sinkhorn_iters
        )
        value_balanced, value_s_col, value_s_row = production.sinkhorn(
            value_tiles, iterations=config.sinkhorn_iters
        )
        natural_key = production.pack_k(
            key_balanced,
            key_s_col,
            key_s_row,
            bits=config.key_bits,
            dpas_layout=False,
        )
        natural_value = production.pack_v(
            value_balanced,
            value_s_col,
            value_s_row,
            bits=config.value_bits,
            dpas_layout=False,
        )
        dpas_key = production.pack_k(
            key_balanced,
            key_s_col,
            key_s_row,
            bits=config.key_bits,
            dpas_layout=True,
        )
        dpas_value = production.pack_v(
            value_balanced,
            value_s_col,
            value_s_row,
            bits=config.value_bits,
            dpas_layout=True,
        )
        _require_same_quantization_metadata(
            torch_module, natural_key, natural_value, dpas_key, dpas_value
        )
        natural_cache[page_start:page_stop] = _assemble_packed_records(
            torch_module, config, natural_key, natural_value
        ).view(page_count, H_KV, config.record_bytes)
        dpas_cache[page_start:page_stop] = _assemble_packed_records(
            torch_module, config, dpas_key, dpas_value
        ).view(page_count, H_KV, config.record_bytes)
    torch_module.xpu.synchronize()

    native_block_table = torch_module.arange(
        total_native_pages,
        dtype=torch_module.int32,
        device="xpu:0",
    ).view(max_batch, native_pages)
    tail_fixtures = {
        context: _build_tail_fixture(
            torch_module,
            rotated_key=key_rotated,
            rotated_value=value_rotated,
            context=context,
            pages_per_request=native_pages,
        )
        for context in sorted(set(contexts))
    }
    frontier_key = {
        context: key_rotated_input[:, context - 1].clone()
        for context in sorted(set(contexts))
    }
    frontier_value = {
        context: value_rotated_input[:, context - 1].clone()
        for context in sorted(set(contexts))
    }
    generator_provenance = {
        "algorithm": "torch CPU Generator.randn request-major",
        "key_seed": CORPUS_SEED,
        "value_seed": CORPUS_SEED + 1,
        "generation_dtype": "torch.bfloat16",
    }
    provenance: dict[str, Any] = {
        "fixture_mode": MATCHED_FIXTURE_MODE,
        "logical_shape": list(logical_shape),
        "logical_dtype": "torch.bfloat16",
        "logical_tokens": max_batch * max_context,
        "padded_context_for_kvarn": padded_context,
        "padding_policy": "zero; never addressable beyond each case seq_len",
        "generator": generator_provenance,
        "key_sha256": key_digest.hexdigest(),
        "value_sha256": value_digest.hexdigest(),
        "logical_bytes_hashed": {"key": key_bytes, "value": value_bytes},
        "production_packer_sources": production.sources,
        "production_config": {
            "cache_dtype": "kvarn_k4v4_g128_compact",
            "head_dim": config.head_dim,
            "group": config.group,
            "key_bits": config.key_bits,
            "value_bits": config.value_bits,
            "sinkhorn_iters": config.sinkhorn_iters,
            "record_bytes": config.record_bytes,
            "sink_tokens": config.sink_tokens,
            "rtn_quantile_environment": os.environ.get("KVARN_RTN_QUANTILE"),
        },
        "materialization_batches": {
            "rotation_rows": CORPUS_ROTATE_ROW_BATCH,
            "sinkhorn_rtn_tiles": CORPUS_PACK_TILE_BATCH,
        },
        "auto_store_operator": "_C_cache_ops::reshape_and_cache_flash",
        "auto_block_size": auto_block_size,
        "auto_store_sample_positions": sample_positions,
        "auto_key_cache_sha256": _sha256_device_tensor(torch_module, auto_key_cache),
        "auto_value_cache_sha256": _sha256_device_tensor(
            torch_module, auto_value_cache
        ),
        "natural_packed_cache_sha256": _sha256_device_tensor(
            torch_module, natural_cache
        ),
        "dpas_packed_cache_sha256": _sha256_device_tensor(torch_module, dpas_cache),
        "tail_mapping_by_context": {
            str(context): fixture.provenance
            for context, fixture in tail_fixtures.items()
        },
        "invariants": {
            "auto_populated_by_reshape_and_cache_flash": True,
            "natural_populated_by_production_packers": True,
            "dpas_populated_by_production_packers": True,
            "natural_and_dpas_share_sinkhorn_results": True,
            "sink_page_mapped_to_fp16_pool": all(
                fixture.provenance["sink_mapping_verified"]
                for fixture in tail_fixtures.values()
            ),
            "current_tail_mapped_to_fp16_pool": all(
                fixture.provenance["current_tail_mapping_verified"]
                for fixture in tail_fixtures.values()
            ),
            "partial_tail_valid_token_counts_verified": all(
                fixture.provenance["tail_valid_token_count_verified"]
                for fixture in tail_fixtures.values()
            ),
            "all_setup_and_hashing_outside_timing": True,
        },
        "logical_kv_payloads_matched_between_auto_and_kvarn": True,
        "matched_primitive_fixture_eligible": False,
        "matched_parity_eligible": False,
        "validation_status": "pending",
    }
    provenance["logical_corpus_sha256"] = _corpus_identity(provenance)
    validate_matched_corpus_manifest(provenance)
    provenance["matched_primitive_fixture_eligible"] = True
    provenance["validation_status"] = "passed"
    del key_rotated_input, value_rotated_input, key_rotated, value_rotated
    gc.collect()
    torch_module.xpu.empty_cache()
    return MatchedCorpus(
        auto_base=auto_base,
        auto_key_cache=auto_key_cache,
        auto_value_cache=auto_value_cache,
        auto_block_table=auto_block_table,
        natural_cache=natural_cache,
        dpas_cache=dpas_cache,
        native_block_table=native_block_table,
        tail_fixtures=tail_fixtures,
        frontier_key=frontier_key,
        frontier_value=frontier_value,
        max_batch=max_batch,
        max_context=max_context,
        auto_block_size=auto_block_size,
        provenance=provenance,
    )


def ensure_distinct_storage(reference: Any, candidate: Any) -> None:
    if reference.data_ptr() == candidate.data_ptr():
        raise FactoryError("natural and candidate layout caches alias")


def _make_interleaved_auto_cache(
    torch_module: Any, *, batch: int, context: int, block_size: int
) -> tuple[Any, Any, Any, Any]:
    pages = math.ceil(context / block_size)
    total_pages = batch * pages
    base = torch_module.zeros(
        (total_pages, H_KV, block_size, 2 * HEAD_DIM),
        dtype=torch_module.bfloat16,
        device="xpu:0",
    )
    key_cache, value_cache = base.transpose(1, 2).split(HEAD_DIM, dim=-1)
    block_table = torch_module.arange(
        total_pages, dtype=torch_module.int32, device="xpu:0"
    ).view(batch, pages)
    native_pages = math.ceil(context / KVARN_PAGE)
    logical = torch_module.arange(context, dtype=torch_module.int64, device="xpu:0")
    native_local_page = torch_module.div(logical, KVARN_PAGE, rounding_mode="floor")
    native_offset = torch_module.remainder(logical, KVARN_PAGE)
    auto_local_page = torch_module.div(logical, block_size, rounding_mode="floor")
    auto_offset = torch_module.remainder(logical, block_size)
    for request in range(batch):
        native_physical_page = request * native_pages + native_local_page
        rotated_value = (
            torch_module.remainder(native_physical_page, 2).to(torch_module.float32)
            * 0.25
            + native_offset.to(torch_module.float32) / 1024.0
        )
        auto_physical_page = request * pages + auto_local_page
        value_cache[auto_physical_page, auto_offset, :, 0] = (
            16.0 * rotated_value[:, None]
        ).to(torch_module.bfloat16)
    return base, key_cache, value_cache, block_table


def _structured_expected(torch_module: Any, batch: int, context: int) -> Any:
    pages = math.ceil(context / KVARN_PAGE)
    logical = torch_module.arange(context, dtype=torch_module.int64)
    local_page = torch_module.div(logical, KVARN_PAGE, rounding_mode="floor")
    offset = torch_module.remainder(logical, KVARN_PAGE)
    result = torch_module.zeros((batch, H_Q, HEAD_DIM), dtype=torch_module.float32)
    for request in range(batch):
        physical_page = request * pages + local_page
        value = (
            torch_module.remainder(physical_page, 2).to(torch_module.float32) * 0.25
            + offset.to(torch_module.float32) / 1024.0
        )
        result[request, :, 0] = 16.0 * value.mean()
    return result


def invoke_native_decode(
    operation: Callable[..., None],
    *,
    query: Any,
    cache: Any,
    block_table: Any,
    seq_lens: Any,
    block_to_slot: Any,
    tail_key: Any,
    tail_value: Any,
    temp_output: Any,
    exp_sums: Any,
    max_logits: Any,
    output: Any,
    context: int,
    unrotate_output: bool,
    num_kv_splits: int,
    kernel_variant: int,
    dpas_layout: bool,
) -> None:
    operation(
        query,
        cache,
        block_table,
        seq_lens,
        block_to_slot,
        tail_key,
        tail_value,
        temp_output,
        exp_sums,
        max_logits,
        output,
        context,
        SOFTMAX_SCALE,
        unrotate_output,
        False,
        num_kv_splits,
        kernel_variant,
        dpas_layout,
    )


def _difference_metrics(torch_module: Any, candidate: Any, reference: Any) -> dict:
    candidate_cpu = candidate.float().cpu()
    reference_cpu = reference.float().cpu()
    absolute = (candidate_cpu - reference_cpu).abs()
    denominator = reference_cpu.abs().clamp_min(1e-6)
    return {
        "max_abs": float(absolute.max().item()),
        "mean_abs": float(absolute.mean().item()),
        "max_rel_above_floor": float((absolute / denominator).max().item()),
        "finite": bool(torch_module.isfinite(candidate).all().item()),
    }


def run_case(
    torch_module: Any,
    helpers: Any,
    operations: Any,
    case: MatrixCase,
    *,
    corpus: MatchedCorpus | None,
    auto_block_size: int,
    warmup_rounds: int,
    sample_rounds: int,
    correctness_atol: float,
    correctness_rtol: float,
) -> dict[str, Any]:
    batch = case.batch
    context = case.context
    splits = case.requested_splits
    dpas_layout = case.variant.dpas_layout
    native_pages = math.ceil(context / KVARN_PAGE)
    native_total_pages = batch * native_pages
    seq_lens = torch_module.full(
        (batch,), context, dtype=torch_module.int32, device="xpu:0"
    )
    if corpus is None:
        native_table = torch_module.arange(
            native_total_pages, dtype=torch_module.int32, device="xpu:0"
        ).view(batch, native_pages)
        block_to_slot = torch_module.full(
            (native_total_pages,), -1, dtype=torch_module.int32, device="xpu:0"
        )
        tail_key = torch_module.zeros(
            (1, KVARN_PAGE, H_KV, HEAD_DIM),
            dtype=torch_module.float16,
            device="xpu:0",
        )
        tail_value = torch_module.zeros_like(tail_key)
        fixture_mode = UNMATCHED_FIXTURE_MODE
        tail_provenance = None
    else:
        if (
            batch > corpus.max_batch
            or context > corpus.max_context
            or auto_block_size != corpus.auto_block_size
        ):
            raise FactoryError("case exceeds the validated matched corpus")
        native_table = corpus.native_block_table[:batch, :native_pages].contiguous()
        tail_fixture = corpus.tail_fixtures[context]
        block_to_slot = tail_fixture.block_to_slot
        tail_key = tail_fixture.tail_key
        tail_value = tail_fixture.tail_value
        fixture_mode = MATCHED_FIXTURE_MODE
        tail_provenance = tail_fixture.provenance

    query_seed = CORPUS_SEED + batch + context
    generator = torch_module.Generator().manual_seed(query_seed)
    qkv_cpu = torch_module.randn(
        (batch, H_Q * HEAD_DIM + 2 * H_KV * HEAD_DIM), generator=generator
    ).to(torch_module.bfloat16)
    qkv = qkv_cpu.to(device="xpu:0")
    query = qkv[:, : H_Q * HEAD_DIM].view(batch, H_Q, HEAD_DIM)
    if corpus is None:
        key = qkv[:, H_Q * HEAD_DIM : (H_Q + H_KV) * HEAD_DIM].view(
            batch, H_KV, HEAD_DIM
        )
        value = qkv[:, (H_Q + H_KV) * HEAD_DIM :].view(batch, H_KV, HEAD_DIM)
    else:
        key = corpus.frontier_key[context][:batch]
        value = corpus.frontier_value[context][:batch]
    query_rotated = torch_module.empty(
        (batch * H_Q, HEAD_DIM), dtype=torch_module.float16, device="xpu:0"
    )
    operations.hadamard(query.reshape(-1, HEAD_DIM), query_rotated)
    query_rotated = query_rotated.view(batch, H_Q, HEAD_DIM)

    if corpus is None:
        auto_base, auto_key_cache, auto_value_cache, auto_table = (
            _make_interleaved_auto_cache(
                torch_module,
                batch=batch,
                context=context,
                block_size=auto_block_size,
            )
        )
    else:
        auto_pages = math.ceil(context / auto_block_size)
        auto_base = corpus.auto_base
        auto_key_cache = corpus.auto_key_cache
        auto_value_cache = corpus.auto_value_cache
        auto_table = corpus.auto_block_table[:batch, :auto_pages].contiguous()
    cu_q = torch_module.arange(batch + 1, dtype=torch_module.int32, device="xpu:0")
    dummy_cu_k = torch_module.empty_like(cu_q)
    auto_output = torch_module.empty_like(query)

    def auto_launch(
        key_cache: Any, value_cache: Any, block_table: Any, output: Any
    ) -> None:
        operations.auto_decode(
            query,
            key_cache,
            value_cache,
            output,
            cu_q,
            dummy_cu_k,
            seq_lens,
            None,
            block_table,
            None,
            1,
            context,
            0.0,
            None,
            None,
            SOFTMAX_SCALE,
            None,
            False,
            True,
            -1,
            -1,
            0.0,
            False,
            None,
            None,
            True,
            None,
            None,
        )

    def auto_decode() -> None:
        auto_launch(auto_key_cache, auto_value_cache, auto_table, auto_output)

    (
        structured_auto_base,
        structured_auto_key,
        structured_auto_value,
        structured_auto_table,
    ) = _make_interleaved_auto_cache(
        torch_module,
        batch=batch,
        context=context,
        block_size=auto_block_size,
    )
    structured_auto_output = torch_module.empty_like(query)
    structured_native_table = torch_module.arange(
        native_total_pages, dtype=torch_module.int32, device="xpu:0"
    ).view(batch, native_pages)
    structured_block_to_slot = torch_module.full(
        (native_total_pages,), -1, dtype=torch_module.int32, device="xpu:0"
    )
    structured_tail_key = torch_module.zeros(
        (1, KVARN_PAGE, H_KV, HEAD_DIM),
        dtype=torch_module.float16,
        device="xpu:0",
    )
    structured_tail_value = torch_module.zeros_like(structured_tail_key)

    def make_buffers() -> SimpleNamespace:
        return SimpleNamespace(
            temp=torch_module.empty(
                (batch, H_Q * splits, HEAD_DIM),
                dtype=torch_module.float16,
                device="xpu:0",
            ),
            lse=torch_module.empty(
                (batch, H_Q, splits),
                dtype=torch_module.float32,
                device="xpu:0",
            ),
            legacy=torch_module.empty(
                (batch, H_Q, splits),
                dtype=torch_module.float32,
                device="xpu:0",
            ),
            output=torch_module.empty(
                (batch, H_Q, HEAD_DIM),
                dtype=torch_module.float16,
                device="xpu:0",
            ),
        )

    unrotate_output = case.effective_splits > 1

    def native_launch(
        cache: Any,
        buffers: SimpleNamespace,
        variant: int,
        dpas: bool,
        *,
        launch_table: Any | None = None,
        launch_block_to_slot: Any | None = None,
        launch_tail_key: Any | None = None,
        launch_tail_value: Any | None = None,
    ) -> None:
        invoke_native_decode(
            operations.native_decode,
            query=query_rotated,
            cache=cache,
            block_table=native_table if launch_table is None else launch_table,
            seq_lens=seq_lens,
            block_to_slot=(
                block_to_slot if launch_block_to_slot is None else launch_block_to_slot
            ),
            tail_key=tail_key if launch_tail_key is None else launch_tail_key,
            tail_value=tail_value if launch_tail_value is None else launch_tail_value,
            temp_output=buffers.temp,
            exp_sums=buffers.lse,
            max_logits=buffers.legacy,
            output=buffers.output,
            context=context,
            unrotate_output=unrotate_output,
            num_kv_splits=splits,
            kernel_variant=variant,
            dpas_layout=dpas,
        )

    def normalized_output(buffers: SimpleNamespace) -> Any:
        if unrotate_output:
            return buffers.output
        normalized = torch_module.empty_like(buffers.output)
        operations.hadamard(
            buffers.output.reshape(-1, HEAD_DIM),
            normalized.reshape(-1, HEAD_DIM),
        )
        return normalized

    structured_cpu, structured_layout = helpers.make_cache(2, KVARN_RECORD_STRIDE)
    structured_candidate_cpu = _layout_pattern(
        structured_cpu, structured_layout, helpers, dpas_layout
    )
    structured_natural = _allocate_packed_cache(
        structured_cpu, total_pages=native_total_pages, torch_module=torch_module
    )
    structured_candidate = _allocate_packed_cache(
        structured_candidate_cpu,
        total_pages=native_total_pages,
        torch_module=torch_module,
    )
    ensure_distinct_storage(structured_natural, structured_candidate)
    structured_natural_buffers = make_buffers()
    structured_candidate_buffers = make_buffers()
    native_launch(
        structured_natural,
        structured_natural_buffers,
        0,
        False,
        launch_table=structured_native_table,
        launch_block_to_slot=structured_block_to_slot,
        launch_tail_key=structured_tail_key,
        launch_tail_value=structured_tail_value,
    )
    native_launch(
        structured_candidate,
        structured_candidate_buffers,
        case.variant.variant_id,
        dpas_layout,
        launch_table=structured_native_table,
        launch_block_to_slot=structured_block_to_slot,
        launch_tail_key=structured_tail_key,
        launch_tail_value=structured_tail_value,
    )
    auto_launch(
        structured_auto_key,
        structured_auto_value,
        structured_auto_table,
        structured_auto_output,
    )
    torch_module.xpu.synchronize()
    structured_natural_output = normalized_output(structured_natural_buffers)
    structured_candidate_output = normalized_output(structured_candidate_buffers)
    expected = _structured_expected(torch_module, batch, context)
    torch_module.testing.assert_close(
        structured_natural_output.float().cpu(),
        expected,
        atol=correctness_atol,
        rtol=correctness_rtol,
    )
    torch_module.testing.assert_close(
        structured_candidate_output.float().cpu(),
        expected,
        atol=correctness_atol,
        rtol=correctness_rtol,
    )
    torch_module.testing.assert_close(
        structured_auto_output.float().cpu(),
        expected,
        atol=correctness_atol,
        rtol=correctness_rtol,
    )
    structured_candidate_metrics = _difference_metrics(
        torch_module, structured_candidate_output, structured_natural_output
    )
    auto_metrics = _difference_metrics(
        torch_module, structured_auto_output, structured_natural_output
    )
    del (
        structured_natural,
        structured_candidate,
        structured_natural_buffers,
        structured_candidate_buffers,
        structured_natural_output,
        structured_candidate_output,
        structured_auto_base,
        structured_auto_key,
        structured_auto_value,
        structured_auto_table,
        structured_auto_output,
    )
    gc.collect()
    torch_module.xpu.empty_cache()

    if corpus is None:
        dense_cpu, dense_layout = helpers.make_random_cache(2, KVARN_RECORD_STRIDE)
        dense_candidate_cpu = _layout_pattern(
            dense_cpu, dense_layout, helpers, dpas_layout
        )
        natural_cache = _allocate_packed_cache(
            dense_cpu, total_pages=native_total_pages, torch_module=torch_module
        )
        candidate_cache = _allocate_packed_cache(
            dense_candidate_cpu,
            total_pages=native_total_pages,
            torch_module=torch_module,
        )
    else:
        natural_cache = corpus.natural_cache
        candidate_cache = corpus.dpas_cache
    ensure_distinct_storage(natural_cache, candidate_cache)
    natural_buffers = make_buffers()
    candidate_buffers = make_buffers()

    def natural_decode() -> None:
        native_launch(natural_cache, natural_buffers, 0, False)

    def candidate_decode() -> None:
        native_launch(
            candidate_cache,
            candidate_buffers,
            case.variant.variant_id,
            dpas_layout,
        )

    natural_decode()
    candidate_decode()
    auto_decode()
    torch_module.xpu.synchronize()
    natural_normalized = normalized_output(natural_buffers)
    candidate_normalized = normalized_output(candidate_buffers)
    torch_module.testing.assert_close(
        candidate_normalized.float().cpu(),
        natural_normalized.float().cpu(),
        atol=correctness_atol,
        rtol=correctness_rtol,
    )
    dense_metrics = _difference_metrics(
        torch_module, candidate_normalized, natural_normalized
    )
    matched_auto_metrics = (
        _difference_metrics(torch_module, auto_output, natural_normalized)
        if corpus is not None
        else None
    )
    matched_auto_correctness_passed = False
    if matched_auto_metrics is not None:
        if not matched_auto_metrics["finite"]:
            raise FactoryError("matched auto-control output is non-finite")
        torch_module.testing.assert_close(
            natural_normalized.float().cpu(),
            auto_output.float().cpu(),
            atol=correctness_atol,
            rtol=correctness_rtol,
        )
        matched_auto_correctness_passed = True
    del natural_normalized, candidate_normalized

    frontier_offset = (context - 1) % KVARN_PAGE
    scatter_slots = (
        torch_module.arange(batch, dtype=torch_module.int64, device="xpu:0")
        * KVARN_PAGE
        + frontier_offset
    )
    scatter_lookup = torch_module.arange(
        batch, dtype=torch_module.int32, device="xpu:0"
    )
    separate_query = torch_module.empty(
        (batch, H_Q, HEAD_DIM), dtype=torch_module.float16, device="xpu:0"
    )
    fused_query = torch_module.empty_like(separate_query)
    separate_key = torch_module.full(
        (batch, KVARN_PAGE, H_KV, HEAD_DIM),
        -123.0,
        dtype=torch_module.float16,
        device="xpu:0",
    )
    separate_value = torch_module.full_like(separate_key, -123.0)
    fused_key = separate_key.clone()
    fused_value = separate_value.clone()

    def separate_qkv_frontend() -> None:
        operations.hadamard(
            query.reshape(-1, HEAD_DIM), separate_query.reshape(-1, HEAD_DIM)
        )
        operations.scatter(
            key,
            value,
            scatter_slots,
            scatter_lookup,
            separate_key,
            separate_value,
            KVARN_PAGE,
            dpas_layout,
        )

    def fused_qkv_frontend() -> None:
        if operations.fused_qkv_scatter is None:
            raise FactoryError("fused QKV operation is unavailable")
        operations.fused_qkv_scatter(
            query,
            key,
            value,
            scatter_slots,
            scatter_lookup,
            fused_query,
            fused_key,
            fused_value,
            KVARN_PAGE,
            dpas_layout,
        )

    store_base = torch_module.empty(
        (batch, H_KV, auto_block_size, 2 * HEAD_DIM),
        dtype=torch_module.bfloat16,
        device="xpu:0",
    )
    store_key_cache, store_value_cache = store_base.transpose(1, 2).split(
        HEAD_DIM, dim=-1
    )
    auto_slots = (
        torch_module.arange(batch, dtype=torch_module.int64, device="xpu:0")
        * auto_block_size
        + auto_block_size
        - 1
    )
    k_scale = torch_module.ones((), dtype=torch_module.float32, device="xpu:0")
    v_scale = torch_module.ones((), dtype=torch_module.float32, device="xpu:0")

    def auto_store() -> None:
        operations.auto_store(
            key,
            value,
            store_key_cache,
            store_value_cache,
            auto_slots,
            "auto",
            k_scale,
            v_scale,
        )

    separate_qkv_frontend()
    auto_store()
    for token in range(batch):
        torch_module.testing.assert_close(
            store_key_cache[token, auto_block_size - 1],
            key[token],
            atol=0,
            rtol=0,
        )
        torch_module.testing.assert_close(
            store_value_cache[token, auto_block_size - 1],
            value[token],
            atol=0,
            rtol=0,
        )
    fused_correctness: dict[str, Any]
    if operations.fused_qkv_scatter is not None:
        fused_qkv_frontend()
        torch_module.xpu.synchronize()
        for actual, reference in (
            (fused_query, separate_query),
            (fused_key, separate_key),
            (fused_value, separate_value),
        ):
            torch_module.testing.assert_close(actual, reference, atol=0, rtol=0)
        fused_correctness = {"available": True, "separate_matches_fused": True}
    else:
        torch_module.xpu.synchronize()
        fused_correctness = {
            "available": False,
            "separate_matches_fused": None,
            "reason": "kvarn_hadamard_qkv_scatter symbol absent",
        }

    if corpus is None:
        torch_module.xpu.manual_seed_all(20260903 + batch + context)
        auto_base.normal_()
    torch_module.xpu.synchronize()
    decode_timing = measure_interleaved(
        torch_module,
        {
            "natural_oracle": natural_decode,
            "candidate": candidate_decode,
            "auto_control": auto_decode,
        },
        warmup_rounds=warmup_rounds,
        sample_rounds=sample_rounds,
    )
    frontend_arms = {
        "kvarn_separate_q_plus_kv": separate_qkv_frontend,
        "auto_cache_store": auto_store,
    }
    if operations.fused_qkv_scatter is not None:
        frontend_arms["kvarn_fused_qkv"] = fused_qkv_frontend
    frontend_timing = measure_interleaved(
        torch_module,
        frontend_arms,
        warmup_rounds=warmup_rounds,
        sample_rounds=sample_rounds,
    )
    output_postprocess_timing: dict[str, Any] | None = None
    candidate_postprocess_us = 0.0
    if not unrotate_output:
        normalized_candidate_output = torch_module.empty_like(candidate_buffers.output)

        def candidate_output_hadamard() -> None:
            operations.hadamard(
                candidate_buffers.output.reshape(-1, HEAD_DIM),
                normalized_candidate_output.reshape(-1, HEAD_DIM),
            )

        output_postprocess_timing = measure_interleaved(
            torch_module,
            {"candidate_output_hadamard": candidate_output_hadamard},
            warmup_rounds=warmup_rounds,
            sample_rounds=sample_rounds,
        )
        candidate_postprocess_us = output_postprocess_timing["arms"][
            "candidate_output_hadamard"
        ]["device_median_us"]
    decode_medians = {
        name: metrics["device_median_us"]
        for name, metrics in decode_timing["arms"].items()
    }
    frontend_medians = {
        name: metrics["device_median_us"]
        for name, metrics in frontend_timing["arms"].items()
    }
    candidate_separate_stage = (
        decode_medians["candidate"]
        + frontend_medians["kvarn_separate_q_plus_kv"]
        + candidate_postprocess_us
    )
    candidate_fused_stage = (
        decode_medians["candidate"]
        + frontend_medians["kvarn_fused_qkv"]
        + candidate_postprocess_us
        if operations.fused_qkv_scatter is not None
        else None
    )
    auto_stage = decode_medians["auto_control"] + frontend_medians["auto_cache_store"]
    result = {
        **case.as_dict(),
        "case_id": (
            f"b{batch}-c{context}-s{splits}-"
            f"v{case.variant.variant_id}-{case.variant.name}"
        ),
        "status": "correctness_passed_and_timed",
        "scope": "xpu_primitive_device_stage",
        "matched_primitive_ratio_eligible": (
            corpus is not None
            and corpus.provenance["matched_primitive_fixture_eligible"] is True
            and matched_auto_correctness_passed
        ),
        "explicit_native_op_args": {
            "num_kv_splits": splits,
            "kernel_variant": case.variant.variant_id,
            "dpas_layout": dpas_layout,
            "natural_oracle": {
                "kernel_variant": 0,
                "dpas_layout": False,
            },
            "unrotate_output": unrotate_output,
            "write_bf16_output": False,
        },
        "allocation_evidence": {
            "natural_and_candidate_cache_distinct": True,
            "natural_cache_data_ptr": natural_cache.data_ptr(),
            "candidate_cache_data_ptr": candidate_cache.data_ptr(),
            "auto_cache_data_ptr": auto_base.data_ptr(),
        },
        "correctness": {
            "thresholds": {
                "atol": correctness_atol,
                "rtol": correctness_rtol,
            },
            "structured_candidate_vs_natural": structured_candidate_metrics,
            "structured_auto_vs_natural": auto_metrics,
            "dense_candidate_vs_natural": dense_metrics,
            "matched_auto_vs_quantized_natural": matched_auto_metrics,
            "matched_auto_vs_quantized_natural_passed": (
                matched_auto_correctness_passed if corpus is not None else None
            ),
            "fused_qkv": fused_correctness,
        },
        "timing": {
            "source": "torch.xpu.Event device elapsed time",
            "decode": decode_timing,
            "frontend": frontend_timing,
            "candidate_output_postprocess": (
                output_postprocess_timing
                if output_postprocess_timing is not None
                else "inverse H256 fused into multi-split decode"
            ),
        },
        "diagnostic_ratios": {
            "candidate_decode_over_natural": (
                decode_medians["candidate"] / decode_medians["natural_oracle"]
            ),
            "candidate_decode_over_auto": (
                decode_medians["candidate"] / decode_medians["auto_control"]
            ),
            "candidate_separate_device_stage_over_auto": (
                candidate_separate_stage / auto_stage
            ),
            "candidate_fused_device_stage_over_auto": (
                candidate_fused_stage / auto_stage
                if candidate_fused_stage is not None
                else None
            ),
            "candidate_separate_device_stage_us": candidate_separate_stage,
            "candidate_fused_device_stage_us": candidate_fused_stage,
            "auto_device_stage_us": auto_stage,
            "candidate_output_postprocess_us": candidate_postprocess_us,
            "fusion_selection": "reported_as_independent_axes",
            "fused_over_separate_frontend": (
                frontend_medians["kvarn_fused_qkv"]
                / frontend_medians["kvarn_separate_q_plus_kv"]
                if operations.fused_qkv_scatter is not None
                else None
            ),
        },
        "fixture": {
            "fixture_mode": fixture_mode,
            "auto_timing_cache": (
                "shared_deterministic_logical_bfloat16_corpus"
                if corpus is not None
                else "dense_random_bfloat16"
            ),
            "kvarn_timing_cache": (
                "production_sinkhorn_rtn_from_shared_logical_corpus"
                if corpus is not None
                else "dense_random_packed_k4v4"
            ),
            "structured_correctness_cache_timed": False,
            "kvarn_page_size": KVARN_PAGE,
            "kvarn_record_stride": KVARN_RECORD_STRIDE,
            "auto_block_size": auto_block_size,
            "natural_and_dpas_caches_allocated_separately": True,
            "logical_kv_payloads_matched_between_auto_and_kvarn": corpus is not None,
            "matched_primitive_fixture_eligible": (
                corpus.provenance["matched_primitive_fixture_eligible"]
                if corpus is not None
                else False
            ),
            "matched_parity_eligible": False,
            "logical_corpus_sha256": (
                corpus.provenance["logical_corpus_sha256"]
                if corpus is not None
                else None
            ),
            "query_seed": query_seed,
            "frontend_kv_logical_position": context - 1 if corpus is not None else None,
            "frontend_kv_page_offset": frontier_offset,
            "tail_mapping": tail_provenance,
            "limitation": UNMATCHED_FIXTURE_WARNING if corpus is None else None,
        },
    }
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-library", type=Path, required=True)
    parser.add_argument("--flash-library", type=Path, required=True)
    parser.add_argument("--base-derivation", required=True)
    parser.add_argument("--base-closure-sha256", required=True)
    parser.add_argument("--flash-derivation", required=True)
    parser.add_argument("--flash-closure-sha256", required=True)
    parser.add_argument("--vllm-xpu-nix-repo", type=Path, default=project)
    parser.add_argument("--vllm-repo", type=Path, default=project.parent / "vllm")
    parser.add_argument(
        "--kernels-repo", type=Path, default=project.parent / "vllm-xpu-kernels"
    )
    parser.add_argument("--expected-vllm-xpu-nix-revision", required=True)
    parser.add_argument("--expected-vllm-revision", required=True)
    parser.add_argument("--expected-kernels-revision", required=True)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANT_NAMES))
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--contexts", default="4096,16384,65023")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--auto-block-size", type=int, default=64)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--sample-rounds", type=int, default=10)
    parser.add_argument("--correctness-atol", type=float, default=0.08)
    parser.add_argument("--correctness-rtol", type=float, default=0.03)
    parser.add_argument(
        "--fixture-mode",
        choices=(MATCHED_FIXTURE_MODE, UNMATCHED_FIXTURE_MODE),
        default=MATCHED_FIXTURE_MODE,
        help=(
            "matched-production is the fail-closed default; unmatched-diagnostic "
            "retains the older candidate-ranking fixture"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.variant_specs = parse_variants(args.variants)
        args.split_values = parse_split_tokens(args.splits)
        args.context_values = parse_int_list(args.contexts, label="--contexts")
        args.batch_values = parse_int_list(args.batches, label="--batches")
        args.matrix = build_matrix(
            batches=args.batch_values,
            contexts=args.context_values,
            splits=args.split_values,
            variants=args.variant_specs,
        )
        args.output = ensure_durable_output(args.output, allow_tmp=args.allow_tmp)
        if args.auto_block_size <= 0:
            raise FactoryError("--auto-block-size must be positive")
        if args.fixture_mode == MATCHED_FIXTURE_MODE and args.auto_block_size != 64:
            raise FactoryError(
                "matched-production requires Brutus's effective auto block size 64; "
                "other sizes require an explicitly broader build and runner contract"
            )
        if args.warmup_rounds < 1 or args.sample_rounds < 1:
            raise FactoryError("warmup and sample rounds must be positive")
        if args.correctness_atol < 0 or args.correctness_rtol < 0:
            raise FactoryError("correctness tolerances must be non-negative")
        args.base_build = validate_build_attestation(
            label="base",
            derivation=args.base_derivation,
            closure_sha256=args.base_closure_sha256,
        )
        args.flash_build = validate_build_attestation(
            label="flash",
            derivation=args.flash_derivation,
            closure_sha256=args.flash_closure_sha256,
        )
        args.expected_revisions = {
            "vllm-xpu-nix": args.expected_vllm_xpu_nix_revision,
            "vllm": args.expected_vllm_revision,
            "vllm-xpu-kernels": args.expected_kernels_revision,
        }
        for label, revision in args.expected_revisions.items():
            if not GIT_COMMIT.fullmatch(revision):
                raise FactoryError(
                    f"--expected-{label}-revision must be a full Git commit"
                )
    except FactoryError as error:
        parser.error(str(error))
    return args


def initial_document(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "artifact_kind": "kvarn_b70_primitive_factory_run",
        "status": "running",
        "started_at": utc_now(),
        "scope": {
            "level": "primitive_and_per_layer_device_stage",
            "warning": SCOPE_WARNING,
            "promotable": False,
            "acceptance_eligible": False,
            "service_performance_conclusion": None,
            "parity_conclusion": None,
            "included": [
                "registered Torch XPU operators",
                "Kvarn decode primitive",
                "FA2 auto-cache decode control",
                "Kvarn and auto cache-write/query-transform device stages",
            ],
            "excluded": [
                "vLLM service and scheduler",
                "model projections, MLP, and non-attention layers",
                "transport and request queueing",
                "Kvarn periodic packed-page flush",
            ],
        },
        "fixture_matching": {
            "requested_mode": args.fixture_mode,
            "logical_kv_payloads_matched_between_auto_and_kvarn": False,
            "matched_primitive_fixture_eligible": False,
            "matched_parity_eligible": False,
            "validation_status": (
                "pending"
                if args.fixture_mode == MATCHED_FIXTURE_MODE
                else "not_applicable"
            ),
            "warning": (
                UNMATCHED_FIXTURE_WARNING
                if args.fixture_mode == UNMATCHED_FIXTURE_MODE
                else None
            ),
        },
        "command": {"argv": sys.argv, "cwd": os.getcwd()},
        "source_revisions": {
            "expected": args.expected_revisions,
            "actual": None,
            "verified": False,
        },
        "runtime_environment": runtime_environment_contract(),
        "requested_settings": {
            "variants": [dataclasses.asdict(item) for item in args.variant_specs],
            "splits": args.splits,
            "contexts": args.context_values,
            "batches": args.batch_values,
            "auto_block_size": args.auto_block_size,
            "warmup_rounds": args.warmup_rounds,
            "sample_rounds": args.sample_rounds,
            "samples_per_arm": args.sample_rounds * 2,
            "correctness_atol": args.correctness_atol,
            "correctness_rtol": args.correctness_rtol,
            "fixture_mode": args.fixture_mode,
            "matrix": [case.as_dict() for case in args.matrix],
        },
        "build_attestations": {
            "base": args.base_build,
            "flash": args.flash_build,
        },
        "results": [],
    }


def execute(args: argparse.Namespace) -> int:
    document = initial_document(args)
    write_json_atomic(args.output, document)
    try:
        base_record = stable_file_record(args.base_library)
        flash_record = stable_file_record(args.flash_library)
        document["libraries"] = {"base": base_record, "flash": flash_record}
        document["build_attestations"] = {
            "base": verify_nix_artifact(
                label="base",
                library=Path(base_record["path"]),
                attestation=args.base_build,
            ),
            "flash": verify_nix_artifact(
                label="flash",
                library=Path(flash_record["path"]),
                attestation=args.flash_build,
            ),
        }
        repositories = [
            repository_state("vllm-xpu-nix", args.vllm_xpu_nix_repo),
            repository_state("vllm", args.vllm_repo),
            repository_state("vllm-xpu-kernels", args.kernels_repo),
        ]
        document["repositories"] = repositories
        require_clean_repositories(repositories)
        if not document["runtime_environment"]["prefixed_environment_clean"]:
            names = sorted(
                document["runtime_environment"]["kvarn_or_vllm_prefixed_variables"]
            )
            raise FactoryError(
                "factory environment contains inherited Kvarn/vLLM variables: "
                + ", ".join(names)
            )
        document["source_revisions"]["actual"] = require_expected_repository_revisions(
            repositories, args.expected_revisions
        )
        document["source_revisions"]["verified"] = True
        starting_identity = evidence_identity(
            document["libraries"], document["repositories"]
        )
        document["evidence_identity_sha256"] = starting_identity
        write_json_atomic(args.output, document)

        import torch

        document["software"] = {"python": sys.version, "torch": torch.__version__}
        document["hardware_preflight"] = preflight_xpu(torch)
        document["kernel_kill_suite"] = run_focused_xpu_kill_suite(
            kernels_repo=args.kernels_repo,
            flash_library=Path(flash_record["path"]),
        )
        write_json_atomic(args.output, document)
        require_focused_xpu_kill_suite(document["kernel_kill_suite"])
        operations, schemas = load_operators(
            torch,
            base_library=Path(base_record["path"]),
            flash_library=Path(flash_record["path"]),
        )
        document["operator_schemas"] = schemas
        document["optional_features"] = {
            "fused_qkv_scatter": operations.fused_qkv_scatter is not None
        }
        helpers = load_fixture_helpers(args.kernels_repo)
        corpus: MatchedCorpus | None = None
        if args.fixture_mode == MATCHED_FIXTURE_MODE:
            production = load_production_packers(args.vllm_repo)
            corpus = build_matched_corpus(
                torch,
                operations,
                production,
                batches=args.batch_values,
                contexts=args.context_values,
                auto_block_size=args.auto_block_size,
            )
            document["fixture_matching"] = corpus.provenance
        else:
            document["fixture_matching"] = {
                "fixture_mode": UNMATCHED_FIXTURE_MODE,
                "logical_kv_payloads_matched_between_auto_and_kvarn": False,
                "matched_primitive_fixture_eligible": False,
                "matched_parity_eligible": False,
                "validation_status": "not_applicable",
                "warning": UNMATCHED_FIXTURE_WARNING,
            }
        write_json_atomic(args.output, document)

        for index, case in enumerate(args.matrix, start=1):
            print(
                f"[{index}/{len(args.matrix)}] {case.as_dict()}",
                flush=True,
            )
            result = run_case(
                torch,
                helpers,
                operations,
                case,
                corpus=corpus,
                auto_block_size=args.auto_block_size,
                warmup_rounds=args.warmup_rounds,
                sample_rounds=args.sample_rounds,
                correctness_atol=args.correctness_atol,
                correctness_rtol=args.correctness_rtol,
            )
            document["results"].append(result)
            document["completed_cases"] = index
            write_json_atomic(args.output, document)
            gc.collect()
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
        document["fixture_matching"]["matched_primitive_ratio_eligible"] = bool(
            corpus is not None
            and document["results"]
            and all(
                result["matched_primitive_ratio_eligible"]
                for result in document["results"]
            )
        )
        ending_libraries = {
            "base": stable_file_record(args.base_library),
            "flash": stable_file_record(args.flash_library),
        }
        ending_repositories = [
            repository_state("vllm-xpu-nix", args.vllm_xpu_nix_repo),
            repository_state("vllm", args.vllm_repo),
            repository_state("vllm-xpu-kernels", args.kernels_repo),
        ]
        ending_identity = evidence_identity(ending_libraries, ending_repositories)
        document["ending_evidence_identity_sha256"] = ending_identity
        if ending_identity != starting_identity:
            raise FactoryError(
                "shared-library or source Git identity changed during the sweep"
            )
        document["identity_stable_through_sweep"] = True
        document["status"] = "completed_primitive_diagnostic"
        document["finished_at"] = utc_now()
        document["completed_cases"] = len(args.matrix)
        write_json_atomic(args.output, document)
        print(f"durable primitive diagnostic: {args.output}", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001
        document["status"] = "failed"
        document["finished_at"] = utc_now()
        document["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_json_atomic(args.output, document)
        print(f"factory run failed; durable state: {args.output}", file=sys.stderr)
        if isinstance(error, KeyboardInterrupt):
            return 130
        return 2


def main() -> None:
    raise SystemExit(execute(parse_args()))


if __name__ == "__main__":
    main()
