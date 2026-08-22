#!/usr/bin/env python3
"""Workspace-oriented AutoRound runner and Hugging Face publisher."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from urllib.parse import urlparse

from huggingface_hub import HfApi


SCHEMA = 1
FORMATS = {
    "w4a16": ("W4A16", "compressed-tensors", False),
    "w8a16": ("W8A16", "compressed-tensors", False),
    "w3a16": ("W3A16", "compressed-tensors", False),
    "w2a16": ("W2A16", "compressed-tensors", False),
    "mxfp4": ("MXFP4", "compressed-tensors", True),
    "mxfp8": ("MXFP8", "compressed-tensors", True),
    "fp8": ("FP8_STATIC", "compressed-tensors", True),
}
ALIASES = {"int4": "w4a16", "int8": "w8a16", "int3": "w3a16", "int2": "w2a16"}
RECIPES = {
    "light": {"iters": 50, "samples": 128, "batch_size": 4, "sequence_length": 2048},
    "default": {"iters": 200, "samples": 128, "batch_size": 4, "sequence_length": 2048},
    "overnight": {"iters": 400, "samples": 256, "batch_size": 4, "sequence_length": 2048},
    # 1,000 iterations retains AutoRound's highest-accuracy tuning schedule.
    "best": {"iters": 1000, "samples": 512, "batch_size": 4, "sequence_length": 2048},
}
DEFAULT_IGNORE = ["lm_head"]
DEFAULT_STORAGE = {
    "mode": "disk", "path": None, "limit_gib": None, "min_free_gib": 200,
    "min_free_percent": 10, "extent_mib": "auto", "pageable_lru_gib": "auto",
    "pinned_staging_mib": 512, "pinned_slots": 2, "strict_pinned": False,
    "reader_workers": 1, "writer_workers": 1, "prefetch_batches": 2,
    "read_queue_mib": 512, "write_queue_mib": 512, "reorder_queue_mib": 512,
    "cleanup": "success",
}
DEFAULT_RESOURCES = {
    "host_mem_available_floor_gib": 24, "host_mem_abort_floor_gib": 12,
    "host_mem_resume_gib": 28, "pressure_grace_seconds": 120,
    "pressure_poll_ms": 500, "pause_timeout_seconds": 900,
    "resume_settle_seconds": 30, "memory_high_gib": 68, "memory_max_gib": 76,
    "memory_swap_max_gib": 24, "oom_score_adjust": 500, "cpu_weight": 25,
    "io_weight": 25, "nice": 10, "io_scheduling_class": "best-effort",
    "io_scheduling_priority": 7, "read_bandwidth_mib_s": 300,
    "write_bandwidth_mib_s": 200, "read_iops": 4000, "write_iops": 2000,
}


def die(message: str) -> "NoReturn":
    raise SystemExit(f"quantize: {message}")


def atomic_json(path: Path, value: dict) -> None:
    """Publish JSON only after its complete contents reach stable storage."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def normalize_model(value: str) -> tuple[str, str | None]:
    """Return (owner/name, URL tree revision)."""
    if value.startswith(("http://", "https://")):
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            die("only https://huggingface.co model URLs are accepted")
        if parsed.query or parsed.fragment:
            die("Hugging Face model URLs must not contain a query or fragment")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) == 2:
            repo, revision = "/".join(parts), None
        elif len(parts) == 4 and parts[2] == "tree":
            repo, revision = "/".join(parts[:2]), parts[3]
        else:
            die("expected https://huggingface.co/OWNER/MODEL[/tree/REVISION]")
    else:
        parts = value.strip("/").split("/")
        if len(parts) != 2:
            die("model must be OWNER/MODEL or a Hugging Face model URL")
        repo, revision = "/".join(parts), None
    if not all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) for part in repo.split("/")):
        die(f"invalid Hugging Face repository id: {repo}")
    return repo, revision


def workspace_config(path: Path) -> dict:
    config_path = path / "quantization.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def ignore_rules(config: dict) -> list[str]:
    """Return validated llm-compressor selectors kept at source precision."""
    value = config.get("quantization", {}).get("ignore", DEFAULT_IGNORE)
    if not isinstance(value, list):
        die("quantization.ignore must be a JSON array of module-name strings")
    result = []
    for index, rule in enumerate(value):
        if not isinstance(rule, str) or not rule.strip():
            die(f"quantization.ignore[{index}] must be a non-empty string")
        rule = rule.strip()
        if "," in rule:
            die(f"quantization.ignore[{index}] must not contain a comma: {rule!r}")
        if rule not in result:
            result.append(rule)
    return result


def file_manifest(root: Path) -> list[dict]:
    result = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return result


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_phase_commands(
    base_command: list[str],
    *,
    workspace: Path,
    run_dir: Path,
    eval_dir: Path,
    kv_cache: str,
    checkpoint_enabled: bool,
) -> tuple[list[str], list[str] | None, Path]:
    """Build phase commands with one shared, immutable workspace identity."""
    command = base_command.copy()
    lock_path = workspace / "flake.lock"
    if lock_path.exists():
        command.extend(["--workspace-lock-sha256", sha256_path(lock_path)])

    reference_command = None
    reference_info = eval_dir / "bf16-reference.json"
    if kv_cache == "fp8":
        reference_command = command.copy()
        reference_command.extend([
            "--phase", "bf16-reference",
            "--reference-root", str(workspace / "references"),
            "--reference-info-output", str(reference_info),
            "--no-save",
        ])
        reference_command[reference_command.index("--diagnostics-dir") + 1] = str(
            eval_dir / "bf16-reference-capture"
        )
        command.extend(["--phase", "quantize", "--bf16-reference", str(reference_info)])

    if checkpoint_enabled:
        command.extend([
            "--checkpoint-root", str(workspace / "checkpoints"),
            "--checkpoint-info-output", str(run_dir / "checkpoint.json"),
        ])
    return command, reference_command, reference_info


def ensure_model_card(root: Path, repo: str, sha: str, fmt: str, recipe: str, kv_cache: str) -> None:
    card = root / "README.md"
    if card.exists():
        return
    kv_note = "Static FP8 KV-cache scales were dataset-calibrated during quantization." if kv_cache == "fp8" else "No KV-cache scales are bundled."
    serve_flags = " --quantization compressed-tensors --kv-cache-dtype fp8" if kv_cache == "fp8" else ""
    card.write_text(f"""---
base_model: {repo}
library_name: transformers
tags:
- vllm
- autoround
- quantized
- intel-xpu
---

# {repo.split('/')[-1]} {fmt.upper()} AutoRound

Quantized from [`{repo}`](https://huggingface.co/{repo}) at commit `{sha}`
using the `{recipe}` AutoRound recipe. {kv_note}

```bash
vllm serve .{serve_flags}
```

Review the source model's license and add evaluation and Intel XPU benchmark
results before making this repository public.
""")


def cmd_init(args: argparse.Namespace) -> None:
    repo, url_revision = normalize_model(args.model)
    revision = args.revision or url_revision
    root = Path(args.workspace).expanduser().resolve()
    target = root.joinpath(*repo.split("/"))
    if target.exists() and any(target.iterdir()):
        die(f"workspace already exists and is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    hf_owner = os.environ.get("HF_QUANTIZATION_OWNER", "jasonboukheir")
    hf_repo = args.hf_repo or f"{hf_owner}/{repo.split('/')[1]}-W4A16-AutoRound"
    config = {
        "schema": SCHEMA,
        "source": {"repo": repo, "revision": revision},
        "quantization": {
            "format": "w4a16",
            "recipe": "default",
            "kv_cache": "none",
            "ignore": DEFAULT_IGNORE,
            "calibration": {
                "storage": DEFAULT_STORAGE,
                "resources": DEFAULT_RESOURCES,
            },
        },
        "publish": {"repo": hf_repo, "private": False},
    }
    (target / "quantization.json").write_text(json.dumps(config, indent=2) + "\n")
    flake_url = args.flake_url
    (target / "flake.nix").write_text(f'''{{
  description = "Quantization workspace for {repo}";
  inputs.vllm-xpu-nix.url = "{flake_url}";
  outputs = {{ self, vllm-xpu-nix }}:
    vllm-xpu-nix.lib.x86_64-linux.mkQuantizationWorkspace {{ workspace = self; }};
}}
''')
    (target / ".gitignore").write_text(
        "/artifacts/\n/checkpoints/\n/runs/\n/.direnv/\n/.envrc.local\n"
    )
    (target / "README.md").write_text(f"# {repo} — W4A16 AutoRound\n\nQuantization workspace. See `quantization.json` for the pinned recipe.\n")
    print(target)


def resolve_run_config(args: argparse.Namespace) -> tuple[Path, dict, str, str, str]:
    workspace = Path(args.workspace or os.getcwd()).resolve()
    config = workspace_config(workspace)
    model_value = args.model or config.get("source", {}).get("repo")
    if not model_value:
        die("no model given and quantization.json has no source.repo")
    repo, url_revision = normalize_model(model_value)
    revision = args.revision or url_revision or config.get("source", {}).get("revision")
    fmt = (args.format or config.get("quantization", {}).get("format") or "w4a16").lower()
    fmt = ALIASES.get(fmt, fmt)
    if fmt not in FORMATS:
        die(f"unsupported format {fmt!r}; choose: {', '.join(FORMATS)}")
    recipe = args.recipe or config.get("quantization", {}).get("recipe") or "default"
    if recipe not in RECIPES:
        die(f"unsupported recipe {recipe!r}; choose: {', '.join(RECIPES)}")
    return workspace, config, repo, revision, fmt


def execute_run(args: argparse.Namespace, *, test_mode: bool = False) -> None:
    workspace, config, repo, revision, fmt = resolve_run_config(args)
    recipe = args.recipe or config.get("quantization", {}).get("recipe") or "default"
    args.kv_cache = args.kv_cache or config.get("quantization", {}).get("kv_cache") or "none"
    ignored = ignore_rules(config)
    calibration = config.get("quantization", {}).get("calibration", {})
    if not isinstance(calibration, dict):
        die("quantization.calibration must be a JSON object")
    profile = RECIPES[recipe]
    if test_mode and (
        len(args.test_iters) != 2
        or args.test_iters[0] <= 0
        or args.test_iters[0] >= args.test_iters[1]
    ):
        die("--test-iters requires two positive increasing values, such as 5 20")
    if test_mode and args.test_calibration_samples <= 0:
        die("--test-calibration-samples must be positive")
    args.batch_size = args.batch_size or calibration.get("batch_size", profile["batch_size"])
    args.seqlen = args.seqlen or calibration.get("sequence_length", profile["sequence_length"])
    target_calibration_samples = args.calibration_samples or calibration.get("samples", profile["samples"])
    if test_mode and not args.full_calibration:
        args.calibration_samples = min(target_calibration_samples, args.test_calibration_samples)
    else:
        args.calibration_samples = target_calibration_samples
    args.dataset = args.dataset or calibration.get("dataset", "NeelNanda/pile-10k")
    args.seed = args.seed if args.seed is not None else calibration.get("seed", 42)
    storage = {**DEFAULT_STORAGE, **calibration.get("storage", {})}
    resources = {**DEFAULT_RESOURCES, **calibration.get("resources", {})}
    if storage["mode"] not in {"disk", "memory"}:
        die("quantization.calibration.storage.mode must be disk or memory")
    if args.isolate and os.environ.get("QUANTIZE_SYSTEMD_SCOPE") != "1":
        properties = [
            f"MemoryHigh={resources['memory_high_gib']}G",
            f"MemoryMax={resources['memory_max_gib']}G",
            f"MemorySwapMax={resources['memory_swap_max_gib']}G",
            f"CPUWeight={resources['cpu_weight']}",
            f"IOWeight={resources['io_weight']}",
        ]
        command = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                   "--setenv=QUANTIZE_SYSTEMD_SCOPE=1"]
        for prop in properties:
            command.extend(["--property", prop])
        command.extend([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
        raise SystemExit(subprocess.run(command).returncode)
    args.torch_compile = (
        args.torch_compile
        if args.torch_compile is not None
        else bool(config.get("quantization", {}).get("torch_compile", False))
    )
    checkpoint_config = config.get("quantization", {}).get("checkpoint", {})
    if not isinstance(checkpoint_config, dict):
        die("quantization.checkpoint must be a JSON object")
    checkpoint_enabled = (
        args.checkpoint
        if args.checkpoint is not None
        else False
        if test_mode
        else bool(checkpoint_config.get("enabled", True))
    )
    if args.resume and not checkpoint_enabled:
        die("--resume requires checkpointing; remove --no-checkpoint")
    scheme, export_format, experimental = FORMATS[fmt]
    if experimental and not args.allow_experimental:
        die(f"{fmt} is experimental on Intel XPU; pass --allow-experimental after validating the vLLM kernel path")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo, revision=revision)
    resolved_sha = info.sha
    variant = f"{repo.split('/')[-1]}-{fmt.upper()}-AutoRound"
    if args.variant_suffix:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.variant_suffix):
            die("--variant-suffix must contain only letters, digits, dot, underscore, or hyphen")
        variant = f"{variant}-{args.variant_suffix}"
    final_dir = workspace / "artifacts" / variant
    if not test_mode and final_dir.exists():
        die(f"artifact already exists: {final_dir}; move it aside before rerunning")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = workspace / "runs" / run_id
    output_dir = run_dir / ("test-output" if test_mode else "output")
    output_dir.mkdir(parents=True)
    eval_dir = run_dir / "eval"
    eval_dir.mkdir()
    if storage["path"] is None:
        storage["path"] = str(workspace / ".cache" / "autoround")
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest = {
        "schema": SCHEMA,
        "status": "running",
        "source": {"repo": repo, "requested_revision": revision, "resolved_sha": resolved_sha},
        "quantization": {
            "format": fmt,
            "scheme": scheme,
            "export_format": export_format,
            "recipe": recipe,
            "kv_cache": args.kv_cache,
            "calibration_samples": args.calibration_samples,
            "sequence_length": args.seqlen,
            "dataset": args.dataset,
            "seed": args.seed,
            "ignore": ignored,
            "target_calibration_samples": target_calibration_samples,
            "checkpoint": {
                "enabled": checkpoint_enabled,
                "resume": args.resume,
            },
            "low_gpu_mem_usage": True,
            "storage": storage,
            "resources": resources,
            "variant": variant,
        },
        "reports": {"eval": str(eval_dir.relative_to(workspace))},
        "started_at": started,
        "mode": "test" if test_mode else "run",
    }
    manifest_path = run_dir / "manifest.json"
    atomic_json(manifest_path, manifest)
    if args.kv_cache == "fp8" and fmt != "w4a16":
        die("the calibrated FP8 KV path is currently validated only with W4A16")
    script = os.environ.get("QUANTIZE_LLMCOMPRESSOR_SCRIPT")
    if not script:
        die("llm-compressor runner is missing from this package")
    recipe_iters = args.test_iters[-1] if test_mode else profile["iters"]
    command = [sys.executable, script, "--model", repo, "--revision", resolved_sha,
               "--scheme", scheme, "--output-dir", str(output_dir), "--iters", str(recipe_iters),
               "--samples", str(args.calibration_samples), "--seqlen", str(args.seqlen),
               "--batch-size", str(args.batch_size), "--dataset", args.dataset, "--seed", str(args.seed),
               "--kv-cache", args.kv_cache, "--ignore-json", json.dumps(ignored),
               "--resolved-ignore-output", str(run_dir / "ignore-matches.json"),
               "--activation-store-config", json.dumps(storage),
               "--resource-config", json.dumps(resources),
               "--diagnostics-dir", str(eval_dir)]
    if args.kv_cache == "fp8":
        (eval_dir / "bf16-reference-capture").mkdir()
    command, reference_command, reference_info = build_phase_commands(
        command,
        workspace=workspace,
        run_dir=run_dir,
        eval_dir=eval_dir,
        kv_cache=args.kv_cache,
        checkpoint_enabled=checkpoint_enabled,
    )
    if args.resume:
        command.append("--resume")
    if args.torch_compile:
        command.append("--enable-torch-compile")
    manifest["command"] = command
    if reference_command is not None:
        manifest["phases"] = {
            "bf16_reference": {"status": "pending", "command": reference_command},
            "w4_post_kv": {"status": "pending", "command": command},
            "selective_kv": {"status": "pending"},
        }
    atomic_json(manifest_path, manifest)
    print(f">>> {repo}@{resolved_sha} -> {fmt} ({recipe})")
    elapsed = None

    def run_phase(phase_command: list[str], log_name: str) -> None:
        """Stream a phase while preserving its complete crash log under eval/."""
        log_path = eval_dir / log_name
        with log_path.open("w", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                phase_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            returncode = process.wait()
        if returncode:
            raise subprocess.CalledProcessError(returncode, phase_command)

    try:
        low_elapsed = None
        if reference_command is not None:
            run_phase(reference_command, "bf16-reference.log")
            reference = json.loads(reference_info.read_text())
            if reference.get("status") != "complete":
                raise RuntimeError("BF16 reference phase did not publish a complete manifest")
            manifest["phases"]["bf16_reference"] = {
                "status": "complete",
                "identity_sha256": reference["identity_sha256"],
                "report": str(reference_info.relative_to(workspace)),
            }
            manifest["phases"]["w4_post_kv"]["status"] = "running"
            atomic_json(manifest_path, manifest)
        if test_mode:
            low_iters, high_iters = args.test_iters
            low_command = command.copy()
            low_command[low_command.index("--iters") + 1] = str(low_iters)
            low_command.append("--no-save")
            low_started = time.monotonic()
            run_phase(low_command, "w4-post-kv-low.log")
            low_elapsed = time.monotonic() - low_started
            manifest["test_timing"] = {
                "low_iters": low_iters,
                "low_elapsed_seconds": low_elapsed,
                "high_iters": high_iters,
            }
            atomic_json(manifest_path, manifest)
        run_started = time.monotonic()
        run_phase(command, "w4-post-kv.log")
        if reference_command is not None:
            manifest["phases"]["w4_post_kv"]["status"] = "complete"
        elapsed = time.monotonic() - run_started
        matches_path = run_dir / "ignore-matches.json"
        if matches_path.exists():
            manifest["quantization"]["resolved_ignore_matches"] = json.loads(matches_path.read_text())
        checkpoint_path = run_dir / "checkpoint.json"
        if checkpoint_path.exists():
            manifest["checkpoint"] = json.loads(checkpoint_path.read_text())
        ensure_model_card(output_dir, repo, resolved_sha, fmt, recipe, args.kv_cache)
        if test_mode:
            target_iters = profile["iters"]
            assert low_elapsed is not None
            low_iters, high_iters = args.test_iters
            seconds_per_iteration = max(
                0.0, (elapsed - low_elapsed) / (high_iters - low_iters)
            )
            measured_fixed_seconds = max(
                0.0, low_elapsed - seconds_per_iteration * low_iters
            )
            sample_scale = target_calibration_samples / args.calibration_samples
            estimated_fixed_seconds = measured_fixed_seconds * sample_scale
            estimate = estimated_fixed_seconds + seconds_per_iteration * target_iters
            manifest.update(
                status="complete",
                completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                elapsed_seconds=elapsed,
                estimated_full_run_seconds=estimate,
                estimated_seconds_per_iteration=seconds_per_iteration,
                measured_fixed_seconds=measured_fixed_seconds,
                estimated_full_sample_fixed_seconds=estimated_fixed_seconds,
                test_output=str(output_dir.relative_to(workspace)),
                files=file_manifest(output_dir),
            )
            print(
                f">>> {low_iters}/{high_iters}-iteration tests completed; "
                f"fitted {recipe!r} estimate: {estimate / 3600:.1f} hours"
            )
            print(
                ">>> estimate is directional; the full run also uses "
                f"{target_calibration_samples} rather than {args.calibration_samples} samples"
            )
        else:
            shutil.copytree(eval_dir, output_dir / "eval", dirs_exist_ok=True)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            output_dir.rename(final_dir)
            manifest.update(status="complete", completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                            elapsed_seconds=elapsed, artifact=str(final_dir.relative_to(workspace)),
                            files=file_manifest(final_dir))
    except BaseException as exc:
        manifest.update(status="failed", completed_at=dt.datetime.now(dt.timezone.utc).isoformat(), error=str(exc))
        raise
    finally:
        atomic_json(manifest_path, manifest)


def cmd_run(args: argparse.Namespace) -> None:
    execute_run(args)


def cmd_test(args: argparse.Namespace) -> None:
    execute_run(args, test_mode=True)


def cmd_doctor(_args: argparse.Namespace) -> None:
    """Fail fast unless the packaged llm-compressor stack can see an Intel XPU."""
    probe = """
import json
import torch
from llmcompressor.modifiers.autoround import AutoRoundModifier

available = torch.xpu.is_available()
sdpa = "not-run"
if available:
    query = torch.randn((1, 8, 16, 64), device="xpu", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(query, query, query)
    torch.xpu.synchronize()
    sdpa = "ok"
result = {
    "torch": torch.__version__,
    "xpu_available": available,
    "xpu_count": torch.xpu.device_count(),
    "devices": [torch.xpu.get_device_name(i) for i in range(torch.xpu.device_count())],
    "llmcompressor_autoround": AutoRoundModifier.__name__,
    "bf16_sdpa": sdpa,
}
print(json.dumps(result, indent=2))
raise SystemExit(0 if available else 1)
"""
    subprocess.run([sys.executable, "-c", probe], check=True)


def cmd_export(args: argparse.Namespace) -> None:
    workspace = Path(args.workspace or os.getcwd()).resolve()
    config = workspace_config(workspace)
    repo = args.repo or config.get("publish", {}).get("repo")
    if not repo:
        die("pass --repo or set publish.repo in quantization.json")
    artifacts = workspace / "artifacts"
    choices = [p for p in artifacts.iterdir() if p.is_dir()] if artifacts.exists() else []
    artifact = Path(args.artifact).resolve() if args.artifact else (choices[0] if len(choices) == 1 else None)
    if artifact is None or not artifact.is_dir():
        die("pass --artifact when the workspace does not contain exactly one artifact")
    gates_path = artifact / "eval" / "export-gates.json"
    gates = json.loads(gates_path.read_text()) if gates_path.exists() else {
        "exportable": False, "gates": {"evaluation_manifest": {"status": "missing"}}, "waivers": []
    }
    if not gates.get("exportable"):
        if not args.waive_gates:
            failed = [name for name, gate in gates.get("gates", {}).items() if gate.get("status") != "pass"]
            die(f"artifact is not exportable; incomplete gates: {', '.join(failed)}; use --waive-gates with --waiver-reason only after review")
        if not args.waiver_reason:
            die("--waive-gates requires --waiver-reason")
        gates.setdefault("waivers", []).append({
            "at": dt.datetime.now(dt.timezone.utc).isoformat(), "reason": args.waiver_reason,
            "failed_gates": [name for name, gate in gates.get("gates", {}).items() if gate.get("status") != "pass"],
        })
        gates["exportable"] = True
        gates["exported_with_waiver"] = True
        gates_path.parent.mkdir(parents=True, exist_ok=True)
        gates_path.write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n")
    configured_private = bool(config.get("publish", {}).get("private", False))
    private = args.private or (configured_private and not args.public)
    if args.dry_run:
        print(json.dumps({"repo": repo, "artifact": str(artifact), "private": private}, indent=2))
        return
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    commit = api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(artifact), commit_message=args.message)
    if args.public:
        # create_repo(exist_ok=True) does not promise to change an existing
        # repository's visibility. Publication must therefore be explicit.
        api.update_repo_settings(repo_id=repo, repo_type="model", private=False)
    print(commit)


def parser() -> argparse.ArgumentParser:
    formatter = argparse.RawDescriptionHelpFormatter
    p = argparse.ArgumentParser(
        prog="quantize",
        description="Create reproducible Intel XPU model quantizations and publish them to Hugging Face.",
        epilog="""Typical flow:
  quantize init OWNER/MODEL
  cd $QUANTIZED_MODELS_ROOT/OWNER/MODEL
  nix run .#quantize -- --kv-cache fp8
  nix run .#export -- --dry-run
  nix run .#export

Run `quantize COMMAND --help` for command-specific choices and safety behavior.""",
        formatter_class=formatter,
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    doctor = sub.add_parser(
        "doctor",
        help="verify that the packaged Torch, llm-compressor, and Intel XPU runtime work",
        description="""Probe the exact packaged quantization closure before loading a model.

The command prints the Torch version, discovered Intel XPU devices, and the
llm-compressor AutoRound modifier. It exits unsuccessfully when no XPU is
visible, preventing an accidental CPU quantization run.""",
        formatter_class=formatter,
    )
    doctor.set_defaults(func=cmd_doctor)
    init = sub.add_parser(
        "init",
        help="create a reproducible per-model flake workspace",
        description="""Create the small Git/Nix workspace that owns one model's recipe, manifests, and outputs.

No model weights are downloaded by this command. The generated workspace pins
vllm-xpu-nix as a flake input, ignores large artifacts in Git, and configures a
public Hugging Face destination under jasonboukheir by default.""",
        formatter_class=formatter,
    )
    init.add_argument(
        "model",
        metavar="MODEL",
        help="source as OWNER/MODEL or https://huggingface.co/OWNER/MODEL[/tree/REVISION]",
    )
    init.add_argument(
        "--revision",
        metavar="REVISION",
        help="source branch, tag, or commit; use a commit SHA when the workspace must be reproducible before its first run",
    )
    init.add_argument(
        "--hf-repo",
        metavar="OWNER/NAME",
        help="publication destination; defaults to jasonboukheir/<model>-W4A16-AutoRound",
    )
    init.add_argument(
        "--workspace",
        metavar="DIR",
        default=os.environ.get("QUANTIZED_MODELS_ROOT", "/home/jasonbk/Projects/quantized_models"),
        help="parent directory for OWNER/MODEL workspaces (default: $QUANTIZED_MODELS_ROOT or %(default)s)",
    )
    init.add_argument(
        "--flake-url",
        metavar="URL",
        default="git+ssh://forgejo@git.sunnycareboo.com:2222/jasonbk/vllm-xpu-nix.git",
        help="vllm-xpu-nix source embedded in the generated flake; override only when testing a fork",
    )
    init.set_defaults(func=cmd_init)
    run = sub.add_parser(
        "run",
        help="run AutoRound and create a hashed model artifact",
        description="""Quantize the configured source model and atomically publish the result inside this workspace.

The source revision is resolved to an immutable Hub commit. Downloads use
HF_HOME; outputs go to artifacts/ and provenance to runs/. Existing artifacts
are never overwritten. W4A16 is the conservative Intel XPU default.""",
        epilog="""Examples:
  quantize run
  quantize run OWNER/MODEL --recipe light
  quantize run --kv-cache fp8 --calibration-samples 128
  quantize run --format mxfp4 --allow-experimental""",
        formatter_class=formatter,
    )
    run.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="source repo/URL override; normally read from quantization.json",
    )
    run.add_argument(
        "--workspace",
        metavar="DIR",
        help="workspace containing quantization.json (default: current directory)",
    )
    run.add_argument(
        "--revision",
        metavar="REVISION",
        help="source branch, tag, or commit override; always resolved and recorded as an immutable SHA",
    )
    run.add_argument(
        "--format",
        choices=sorted(FORMATS),
        help="weight format (default: w4a16); MX/FP formats require --allow-experimental on Intel XPU",
    )
    run.add_argument(
        "--recipe",
        choices=sorted(RECIPES),
        help="AutoRound quality/time preset: light=50, default=200, overnight=400, best=1000 iterations",
    )
    run.add_argument(
        "--kv-cache",
        choices=["none", "fp8"],
        help="fp8 calibrates static KV scales with llm-compressor during W4A16 quantization; serve with --kv-cache-dtype fp8",
    )
    run.add_argument(
        "--batch-size",
        type=int,
        help="samples processed per XPU optimization step; defaults to the recipe profile (best: 4)",
    )
    run.add_argument(
        "--seqlen",
        type=int,
        help="calibration token length; defaults to quantization.calibration.sequence_length or 2048",
    )
    run.add_argument(
        "--calibration-samples",
        type=int,
        help="samples used to tune weights and KV scales; defaults to the recipe profile (best: 128)",
    )
    run.add_argument(
        "--dataset",
        help="calibration dataset override; defaults to quantization.calibration.dataset or NeelNanda/pile-10k",
    )
    run.add_argument(
        "--seed",
        type=int,
        help="calibration seed override; defaults to quantization.calibration.seed or 42",
    )
    run.add_argument(
        "--variant-suffix",
        help="append a version label to the artifact directory without replacing an older artifact",
    )
    run.add_argument(
        "--allow-experimental",
        action="store_true",
        help="acknowledge that the selected non-W4A16 format lacks a fully validated Intel vLLM kernel path",
    )
    run.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable AutoRound torch.compile acceleration; defaults to quantization.torch_compile or false",
    )
    run.add_argument(
        "--checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="commit a content-addressed resume point after every decoder block (default: enabled for run, disabled for test)",
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help="continue the matching checkpoint DAG; refuses checkpoints produced by different inputs or settings",
    )
    run.add_argument(
        "--isolate", action=argparse.BooleanOptionalAction, default=True,
        help="run in a transient user systemd scope with configured memory/CPU/I/O protection (default: enabled)",
    )
    run.set_defaults(func=cmd_run)
    test = sub.add_parser(
        "test",
        parents=[run],
        add_help=False,
        help="run a small end-to-end quantization preflight and estimate full runtime",
        description="""Exercise the real model, backend, ignore rules, calibration, packing, and save path with reduced work.

The test uses the full recipe sample count, batch size, and sequence length so
it can expose memory failures. Only AutoRound iterations are reduced. Output
remains under runs/<timestamp>/test-output and is never promoted or exported;
the two-point fitted estimate is directional rather than a guarantee.""",
        epilog="""Examples:
  quantize test
  quantize test --kv-cache fp8
  quantize test --test-iters 5 20""",
        formatter_class=formatter,
    )
    test.add_argument(
        "--test-iters",
        type=int,
        nargs=2,
        default=[5, 20],
        metavar=("LOW", "HIGH"),
        help="two iteration counts used to fit runtime after fixed costs (default: 5 20)",
    )
    test.add_argument(
        "--test-calibration-samples",
        type=int,
        default=32,
        metavar="COUNT",
        help="sample count for the timing passes while retaining full batch/sequence pressure (default: 32)",
    )
    test.add_argument(
        "--full-calibration",
        action="store_true",
        help="use the recipe's full calibration sample count in both timing passes; much slower but tests host-cache capacity",
    )
    test.set_defaults(func=cmd_test)
    export = sub.add_parser(
        "export",
        help="create a Hugging Face repo and upload through Xet",
        description="""Create the configured Hugging Face model repository and upload one completed artifact through hf-xet.

Repositories are public by default. Use --private for gated review. The upload
uses the Hub API rather than a second Git/LFS checkout and is safe to retry.""",
        epilog="""Examples:
  quantize export --dry-run
  quantize export
  quantize export --private
  quantize export --repo organization/custom-name --artifact artifacts/candidate""",
        formatter_class=formatter,
    )
    export.add_argument(
        "--workspace",
        metavar="DIR",
        help="workspace containing quantization.json and artifacts/ (default: current directory)",
    )
    export.add_argument(
        "--waive-gates", action="store_true",
        help="permit export with failed/pending evaluation gates and record the waiver",
    )
    export.add_argument(
        "--waiver-reason", help="required audit reason when --waive-gates is used",
    )
    export.add_argument(
        "--repo",
        metavar="OWNER/NAME",
        help="destination override; normally read from publish.repo in quantization.json",
    )
    export.add_argument(
        "--artifact",
        metavar="DIR",
        help="artifact directory to upload; required when artifacts/ contains zero or multiple candidates",
    )
    visibility = export.add_mutually_exclusive_group()
    visibility.add_argument(
        "--public",
        action="store_true",
        help="publish publicly, overriding publish.private=true from an older workspace (this is the default)",
    )
    visibility.add_argument(
        "--private",
        action="store_true",
        help="create or retain a private Hub repository for review before publication",
    )
    export.add_argument(
        "--dry-run",
        action="store_true",
        help="show resolved repository, artifact, and visibility without creating or uploading anything",
    )
    export.add_argument(
        "--message",
        default="Upload quantized model",
        help="Hub commit message attached to this upload (default: %(default)s)",
    )
    export.set_defaults(func=cmd_export)
    return p


def main() -> None:
    p = parser()
    argv = sys.argv[1:]
    if argv and argv[0] not in {"doctor", "init", "run", "test", "export", "-h", "--help"}:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
