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
import subprocess
import sys
from urllib.parse import urlparse

from huggingface_hub import HfApi, snapshot_download


SCHEMA = 1
FORMATS = {
    "w4a16": ("W4A16", "auto_round", False),
    "w8a16": ("W8A16", "auto_round", False),
    "w3a16": ("W3A16", "auto_round", False),
    "w2a16": ("W2A16", "auto_round", False),
    "mxfp4": ("MXFP4", "auto_round", True),
    "mxfp8": ("MXFP8", "auto_round", True),
    "fp8": ("FP8_STATIC", "auto_round", True),
}
ALIASES = {"int4": "w4a16", "int8": "w8a16", "int3": "w3a16", "int2": "w2a16"}
RECIPES = {
    "default": ("auto-round", []),
    "light": ("auto-round-light", []),
    "overnight": ("auto-round", ["--iters", "400", "--nsamples", "256", "--dynamic_max_gap", "100"]),
    "best": ("auto-round-best", []),
}


def die(message: str) -> "NoReturn":
    raise SystemExit(f"quantize: {message}")


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


def file_manifest(root: Path) -> list[dict]:
    result = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        result.append({"path": str(path.relative_to(root)), "size": path.stat().st_size, "sha256": digest.hexdigest()})
    return result


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
        "quantization": {"format": "w4a16", "recipe": "default", "kv_cache": "none"},
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
    (target / ".gitignore").write_text("/artifacts/\n/runs/\n/.direnv/\n/.envrc.local\n")
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


def cmd_run(args: argparse.Namespace) -> None:
    workspace, config, repo, revision, fmt = resolve_run_config(args)
    recipe = args.recipe or config.get("quantization", {}).get("recipe") or "default"
    args.kv_cache = args.kv_cache or config.get("quantization", {}).get("kv_cache") or "none"
    scheme, export_format, experimental = FORMATS[fmt]
    if args.kv_cache == "fp8":
        export_format = "compressed-tensors"
    if experimental and not args.allow_experimental:
        die(f"{fmt} is experimental on Intel XPU; pass --allow-experimental after validating the vLLM kernel path")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = api.model_info(repo, revision=revision)
    resolved_sha = info.sha
    variant = f"{repo.split('/')[-1]}-{fmt.upper()}-AutoRound"
    final_dir = workspace / "artifacts" / variant
    if final_dir.exists():
        die(f"artifact already exists: {final_dir}; move it aside before rerunning")
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = workspace / "runs" / run_id
    output_dir = run_dir / "output"
    output_dir.mkdir(parents=True)
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
        },
        "started_at": started,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    binary, recipe_flags = RECIPES[recipe]
    if args.kv_cache == "fp8":
        if fmt != "w4a16":
            die("the calibrated FP8 KV path is currently validated only with W4A16")
        script = os.environ.get("QUANTIZE_LLMCOMPRESSOR_SCRIPT")
        if not script:
            die("llm-compressor runner is missing from this package")
        recipe_iters = {"light": 50, "default": 200, "overnight": 400, "best": 1000}[recipe]
        command = [sys.executable, script, "--model", repo, "--revision", resolved_sha,
                   "--scheme", scheme, "--output-dir", str(output_dir), "--iters", str(recipe_iters),
                   "--samples", str(args.calibration_samples), "--seqlen", str(args.seqlen),
                   "--batch-size", str(args.batch_size), "--dataset", args.dataset, "--seed", str(args.seed)]
    else:
        source_path = snapshot_download(repo_id=repo, revision=resolved_sha)
        extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
        command = [binary, "--model", source_path, "--scheme", scheme,
                   "--format", export_format, "--device", "0", "--batch_size", str(args.batch_size),
                   "--gradient_accumulate_steps", str(args.gradient_accumulate), "--seqlen", str(args.seqlen),
                   "--nsamples", str(args.calibration_samples), "--dataset", args.dataset, "--seed", str(args.seed),
                   "--output_dir", str(output_dir), *recipe_flags, *extra]
    manifest["command"] = command
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f">>> {repo}@{resolved_sha} -> {fmt} ({recipe})")
    try:
        subprocess.run(command, check=True)
        ensure_model_card(output_dir, repo, resolved_sha, fmt, recipe, args.kv_cache)
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        output_dir.rename(final_dir)
        manifest.update(status="complete", completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                        artifact=str(final_dir.relative_to(workspace)), files=file_manifest(final_dir))
    except BaseException as exc:
        manifest.update(status="failed", completed_at=dt.datetime.now(dt.timezone.utc).isoformat(), error=str(exc))
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


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
        default=int(os.environ.get("AUTOROUND_QUANTIZE_BS", "4")),
        help="samples processed per XPU step; raise for speed only when VRAM allows (default: %(default)s)",
    )
    run.add_argument(
        "--gradient-accumulate",
        type=int,
        default=int(os.environ.get("AUTOROUND_QUANTIZE_GA", "2")),
        help="native AutoRound accumulation steps; trades memory for effective batch size (default: %(default)s)",
    )
    run.add_argument(
        "--seqlen",
        type=int,
        default=int(os.environ.get("AUTOROUND_QUANTIZE_SEQLEN", "2048")),
        help="calibration token length; longer context costs more memory and time (default: %(default)s)",
    )
    run.add_argument(
        "--calibration-samples",
        type=int,
        default=128,
        help="number of dataset samples used to tune weights and static KV scales (default: %(default)s)",
    )
    run.add_argument(
        "--dataset",
        default="NeelNanda/pile-10k",
        help="Hugging Face calibration dataset; use representative data for better scales (default: %(default)s)",
    )
    run.add_argument(
        "--seed",
        type=int,
        default=42,
        help="calibration sampling seed recorded for reproducibility (default: %(default)s)",
    )
    run.add_argument(
        "--allow-experimental",
        action="store_true",
        help="acknowledge that the selected non-W4A16 format lacks a fully validated Intel vLLM kernel path",
    )
    run.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        metavar="-- AUTO_ROUND_ARGS",
        help="advanced native AutoRound arguments after --; not used by the llm-compressor FP8-KV path",
    )
    run.set_defaults(func=cmd_run)
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
    if argv and argv[0] not in {"init", "run", "export", "-h", "--help"}:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    if not hasattr(args, "func"):
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
