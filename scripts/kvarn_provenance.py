#!/usr/bin/env python3
"""Write a durable provenance manifest for Kvarn validation artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

DEFAULT_ENV_ALLOWLIST = (
    "CCL_ATL_TRANSPORT",
    "CCL_LOG_LEVEL",
    "CCL_PROCESS_LAUNCHER",
    "CCL_ZE_IPC_EXCHANGE",
    "HF_HOME",
    "HOME",
    "KVARN_DBG_LAYERS",
    "KVARN_DUMP_TILES",
    "KVARN_FAST_FLUSH",
    "KVARN_FUSED_DECODE",
    "KVARN_FUSED_VERIFY",
    "KVARN_FUSED_VERIFY_MAXQ",
    "KVARN_FUSED_VERIFY_MIN_BLOCKS",
    "KVARN_NATIVE_XPU",
    "KVARN_NATIVE_XPU_CHUNK_PREFILL",
    "KVARN_NATIVE_XPU_DECODE",
    "KVARN_NATIVE_XPU_DPAS_LAYOUT",
    "KVARN_NATIVE_XPU_HADAMARD_SCATTER",
    "KVARN_NATIVE_XPU_LAYER",
    "KVARN_NATIVE_XPU_MATERIALIZE",
    "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH",
    "KVARN_NATIVE_XPU_SPLITS",
    "KVARN_NUM_KV_SPLITS",
    "KVARN_POOL_MEM_FRAC",
    "KVARN_POOL_SLOTS",
    "KVARN_QUANT_SLIDING",
    "KVARN_RTN_QUANTILE",
    "KVARN_SHARED_VERIFY",
    "KVARN_SINKHORN_ITERS",
    "KVARN_SINK_TOKENS",
    "KVARN_SPLIT_K",
    "VLLM_CACHE_ROOT",
    "VLLM_TARGET_DEVICE",
    "VLLM_XPU_ENABLE_XPU_GRAPH",
    "XDG_CACHE_HOME",
)


def utc_timestamp(timestamp: float | None = None) -> str:
    instant = dt.datetime.fromtimestamp(
        time.time() if timestamp is None else timestamp,
        tz=dt.UTC,
    )
    return instant.isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def ensure_durable_output(path: Path, allow_tmp: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not allow_tmp and resolved.is_relative_to(Path("/tmp")):
        raise ValueError(f"output directory must be durable (outside /tmp): {resolved}")
    return resolved


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def repository_state(name: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    head = git(resolved, "rev-parse", "HEAD")
    status = git(
        resolved,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    ).splitlines()
    return {
        "name": name,
        "path": str(resolved),
        "head": head,
        "branch": git(resolved, "branch", "--show-current") or None,
        "commit_timestamp": git(resolved, "show", "-s", "--format=%cI", "HEAD"),
        "dirty": bool(status),
        "status_porcelain": status,
        "status_sha256": sha256_text("\n".join(status)),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_argv(path: Path) -> list[str]:
    value = load_json(path)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) for item in value)
    ):
        raise ValueError("argv file must contain a non-empty JSON list of strings")
    return value


def load_prompt_fixtures(path: Path) -> list[dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError("fixtures must contain a non-empty JSON list")
    fixtures: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            raise TypeError("each fixture must be a JSON object")
        fixture_id = item.get("id")
        prompt = item.get("prompt")
        max_tokens = item.get("max_tokens")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise TypeError("each fixture requires a non-empty string id")
        if fixture_id in seen:
            raise ValueError(f"duplicate fixture id: {fixture_id}")
        if not isinstance(prompt, str) or not prompt:
            raise TypeError(f"{fixture_id}: prompt must be a non-empty string")
        if not isinstance(max_tokens, int) or max_tokens < 2048:
            raise ValueError(f"{fixture_id}: max_tokens must be at least 2048")
        seen.add(fixture_id)
        fixtures.append(
            {
                "id": fixture_id,
                "category": item.get("category"),
                "max_tokens": max_tokens,
                "prompt_utf8_bytes": len(prompt.encode()),
                "prompt_sha256": sha256_text(prompt),
            }
        )
    return fixtures


def load_environment(path: Path | None) -> tuple[Mapping[str, Any], str]:
    if path is None:
        return os.environ, "process"
    value = load_json(path)
    if not isinstance(value, dict):
        raise TypeError("environment file must contain a JSON object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError("environment keys must be strings")
    return value, str(path.expanduser().resolve())


def select_environment(
    environment: Mapping[str, Any], allowlist: Sequence[str]
) -> dict[str, str | None]:
    selected: dict[str, str | None] = {}
    for name in sorted(set(allowlist)):
        value = environment.get(name)
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"environment value for {name} is not scalar")
        selected[name] = None if value is None else str(value)
    return selected


def artifact_paths(
    output_dir: Path,
    manifest_path: Path,
    explicit: Sequence[Path],
) -> list[Path]:
    if explicit:
        paths = [path.expanduser().resolve() for path in explicit]
    else:
        paths = [path.resolve() for path in output_dir.rglob("*") if path.is_file()]
    unique = {
        path
        for path in paths
        if path != manifest_path and not path.name.startswith(f".{manifest_path.name}.")
    }
    missing = [path for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"artifact is not a regular file: {missing[0]}")
    return sorted(unique)


def artifact_record(path: Path, output_dir: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"artifact changed while it was hashed: {path}")
    try:
        name = str(path.relative_to(output_dir))
    except ValueError:
        name = str(path)
    return {
        "path": name,
        "size_bytes": after.st_size,
        "modified_at": utc_timestamp(after.st_mtime),
        "sha256": digest,
    }


def collect_manifest(args: argparse.Namespace) -> dict[str, Any]:
    started_ns = time.time_ns()
    output_dir = ensure_durable_output(args.output_dir, args.allow_tmp)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / args.manifest_name

    argv = load_argv(args.argv_file)
    fixtures = load_prompt_fixtures(args.fixtures)
    environment, environment_source = load_environment(args.environment_file)
    allowlist = [*DEFAULT_ENV_ALLOWLIST, *args.env]
    repositories = [
        repository_state("vllm-xpu-nix", args.vllm_xpu_nix),
        repository_state("vllm", args.vllm),
        repository_state("vllm-xpu-kernels", args.kernels),
    ]
    artifacts = [
        artifact_record(path, output_dir)
        for path in artifact_paths(output_dir, manifest_path, args.artifact)
    ]
    finished_ns = time.time_ns()
    return {
        "schema_version": 1,
        "collection": {
            "started_at": utc_timestamp(started_ns / 1_000_000_000),
            "finished_at": utc_timestamp(finished_ns / 1_000_000_000),
            "duration_nanoseconds": finished_ns - started_ns,
        },
        "model": {
            "id": args.model,
            "revision": args.model_revision,
        },
        "repositories": repositories,
        "command": {
            "argv": argv,
            "shell_rendered": shlex.join(argv),
        },
        "environment": {
            "source": environment_source,
            "allowlist": sorted(set(allowlist)),
            "values": select_environment(environment, allowlist),
        },
        "fixtures": {
            "path": str(args.fixtures.expanduser().resolve()),
            "sha256": sha256_file(args.fixtures),
            "prompts": fixtures,
        },
        "artifacts": artifacts,
    }


def write_manifest(args: argparse.Namespace) -> Path:
    document = collect_manifest(args)
    output_dir = ensure_durable_output(args.output_dir, args.allow_tmp)
    manifest_path = output_dir / args.manifest_name
    temporary = output_dir / f".{args.manifest_name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return manifest_path


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "When --artifact is omitted, every regular file already beneath "
            "--output-dir is hashed. The manifest itself is always excluded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="durable result directory; /tmp is rejected by default",
    )
    parser.add_argument("--manifest-name", default="provenance.json")
    parser.add_argument("--model", required=True, help="Hugging Face model ID")
    parser.add_argument(
        "--model-revision",
        required=True,
        help="immutable 40-character Hugging Face commit",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="service-gate fixture JSON whose prompt hashes are recorded",
    )
    parser.add_argument(
        "--argv-file",
        type=Path,
        required=True,
        help="JSON list containing the exact rendered vLLM argv",
    )
    parser.add_argument(
        "--environment-file",
        type=Path,
        help="optional rendered environment JSON; defaults to this process",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        help="additional non-secret environment name to record",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=Path,
        help="artifact to hash; repeat as needed, or omit to scan output-dir",
    )
    parser.add_argument("--vllm-xpu-nix", type=Path, default=project)
    parser.add_argument("--vllm", type=Path, default=project.parent / "vllm")
    parser.add_argument(
        "--kernels",
        type=Path,
        default=project.parent / "vllm-xpu-kernels",
    )
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.model_revision):
        parser.error("--model-revision must be a 40-character lowercase hex commit")
    if Path(args.manifest_name).name != args.manifest_name:
        parser.error("--manifest-name must be a filename, not a path")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) for name in args.env):
        parser.error("--env names must be valid environment identifiers")
    try:
        ensure_durable_output(args.output_dir, args.allow_tmp)
    except ValueError as error:
        parser.error(str(error))
    return args


def main() -> None:
    path = write_manifest(parse_args())
    print(path)


if __name__ == "__main__":
    main()
