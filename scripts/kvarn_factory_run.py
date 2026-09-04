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
SCOPE_WARNING = (
    "Primitive/device-stage diagnostic only: these measurements exclude the "
    "vLLM scheduler, model layers, service transport, and Kvarn page flushes. "
    "They are not service-performance or parity evidence."
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
    native_table = torch_module.arange(
        native_total_pages, dtype=torch_module.int32, device="xpu:0"
    ).view(batch, native_pages)
    seq_lens = torch_module.full(
        (batch,), context, dtype=torch_module.int32, device="xpu:0"
    )
    block_to_slot = torch_module.full(
        (native_total_pages,), -1, dtype=torch_module.int32, device="xpu:0"
    )
    tail_key = torch_module.zeros(
        (1, KVARN_PAGE, H_KV, HEAD_DIM),
        dtype=torch_module.float16,
        device="xpu:0",
    )
    tail_value = torch_module.zeros_like(tail_key)

    generator = torch_module.Generator().manual_seed(20260903 + batch + context)
    qkv_cpu = torch_module.randn(
        (batch, H_Q * HEAD_DIM + 2 * H_KV * HEAD_DIM), generator=generator
    ).to(torch_module.bfloat16)
    qkv = qkv_cpu.to(device="xpu:0")
    query = qkv[:, : H_Q * HEAD_DIM].view(batch, H_Q, HEAD_DIM)
    key = qkv[:, H_Q * HEAD_DIM : (H_Q + H_KV) * HEAD_DIM].view(batch, H_KV, HEAD_DIM)
    value = qkv[:, (H_Q + H_KV) * HEAD_DIM :].view(batch, H_KV, HEAD_DIM)
    query_rotated = torch_module.empty(
        (batch * H_Q, HEAD_DIM), dtype=torch_module.float16, device="xpu:0"
    )
    operations.hadamard(query.reshape(-1, HEAD_DIM), query_rotated)
    query_rotated = query_rotated.view(batch, H_Q, HEAD_DIM)

    auto_base, auto_key_cache, auto_value_cache, auto_table = (
        _make_interleaved_auto_cache(
            torch_module,
            batch=batch,
            context=context,
            block_size=auto_block_size,
        )
    )
    cu_q = torch_module.arange(batch + 1, dtype=torch_module.int32, device="xpu:0")
    dummy_cu_k = torch_module.empty_like(cu_q)
    auto_output = torch_module.empty_like(query)

    def auto_decode() -> None:
        operations.auto_decode(
            query,
            auto_key_cache,
            auto_value_cache,
            auto_output,
            cu_q,
            dummy_cu_k,
            seq_lens,
            None,
            auto_table,
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
        cache: Any, buffers: SimpleNamespace, variant: int, dpas: bool
    ) -> None:
        invoke_native_decode(
            operations.native_decode,
            query=query_rotated,
            cache=cache,
            block_table=native_table,
            seq_lens=seq_lens,
            block_to_slot=block_to_slot,
            tail_key=tail_key,
            tail_value=tail_value,
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
    native_launch(structured_natural, structured_natural_buffers, 0, False)
    native_launch(
        structured_candidate,
        structured_candidate_buffers,
        case.variant.variant_id,
        dpas_layout,
    )
    auto_decode()
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
        auto_output.float().cpu(),
        expected,
        atol=correctness_atol,
        rtol=correctness_rtol,
    )
    structured_candidate_metrics = _difference_metrics(
        torch_module, structured_candidate_output, structured_natural_output
    )
    auto_metrics = _difference_metrics(
        torch_module, auto_output, structured_natural_output
    )
    del (
        structured_natural,
        structured_candidate,
        structured_natural_buffers,
        structured_candidate_buffers,
        structured_natural_output,
        structured_candidate_output,
    )
    gc.collect()
    torch_module.xpu.empty_cache()

    dense_cpu, dense_layout = helpers.make_random_cache(2, KVARN_RECORD_STRIDE)
    dense_candidate_cpu = _layout_pattern(dense_cpu, dense_layout, helpers, dpas_layout)
    natural_cache = _allocate_packed_cache(
        dense_cpu, total_pages=native_total_pages, torch_module=torch_module
    )
    candidate_cache = _allocate_packed_cache(
        dense_candidate_cpu,
        total_pages=native_total_pages,
        torch_module=torch_module,
    )
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
    del natural_normalized, candidate_normalized

    scatter_slots = (
        torch_module.arange(batch, dtype=torch_module.int64, device="xpu:0")
        * KVARN_PAGE
        + KVARN_PAGE
        - 1
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
            "auto_timing_cache": "dense_random_bfloat16",
            "kvarn_timing_cache": "dense_random_packed_k4v4",
            "structured_correctness_cache_timed": False,
            "kvarn_page_size": KVARN_PAGE,
            "kvarn_record_stride": KVARN_RECORD_STRIDE,
            "auto_block_size": auto_block_size,
            "natural_and_dpas_caches_allocated_separately": True,
            "logical_kv_payloads_matched_between_auto_and_kvarn": False,
            "matched_parity_eligible": False,
            "limitation": UNMATCHED_FIXTURE_WARNING,
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
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANT_NAMES))
    parser.add_argument("--splits", default="auto")
    parser.add_argument("--contexts", default="4096,16384,65023")
    parser.add_argument("--batches", default="1,4")
    parser.add_argument("--auto-block-size", type=int, default=832)
    parser.add_argument("--warmup-rounds", type=int, default=4)
    parser.add_argument("--sample-rounds", type=int, default=10)
    parser.add_argument("--correctness-atol", type=float, default=0.08)
    parser.add_argument("--correctness-rtol", type=float, default=0.03)
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
    except FactoryError as error:
        parser.error(str(error))
    return args


def initial_document(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": 1,
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
            "logical_kv_payloads_matched_between_auto_and_kvarn": False,
            "matched_parity_eligible": False,
            "warning": UNMATCHED_FIXTURE_WARNING,
        },
        "command": {"argv": sys.argv, "cwd": os.getcwd()},
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
