#!/usr/bin/env python3
"""KL-divergence eval: AutoRound-quantized model vs. its BF16 reference.

Two-pass design — never both models in memory at once:
  pass 1: load BF16 reference (accelerate offload to system RAM), forward
          on N sequences, write log-softmax(logits) to per-sample .npy under
          --cache-dir. Resumable: existing files are skipped.
  pass 2: unload BF16, load quant model on-device, forward on the same
          sequences, accumulate KL(P_bf16 || P_quant) and top-1 agreement
          against the cached reference.

Cache key is (bf16-model, dataset, seqlen, num-samples, seed) so reruns
against a different quantized model reuse the BF16 pass for free.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _empty_cache() -> None:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cache_key(bf16_model: str, dataset: str, seqlen: int, num_samples: int, seed: int) -> str:
    # Bumped 'v2' tag invalidates pre-arch-aware caches (pre-2026-05-08): those
    # were generated with AutoModelForCausalLM on a VLM checkpoint, which silently
    # picked Qwen3_5ForCausalLM (text-only) and produced random-init logits.
    raw = f"v2|{bf16_model}|{dataset}|{seqlen}|{num_samples}|{seed}"
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _resolve_model_class(model_path: str):
    """Pick the right model class by reading the saved architectures field.

    AutoModelForCausalLM dispatches by model_type, which for VLMs like Qwen3_5
    resolves to the text-only class (Qwen3_5ForCausalLM) — but the saved
    weights live under `model.language_model.X.*`, so the LM-only class can't
    match them. We honor the architectures list instead, falling back to
    AutoModelForCausalLM only when the explicit class isn't importable.
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    arch_list = getattr(config, "architectures", None) or []
    for arch_name in arch_list:
        cls = getattr(transformers, arch_name, None)
        if cls is not None:
            return arch_name, cls
    return "AutoModelForCausalLM", AutoModelForCausalLM


def load_eval_sequences(tokenizer, dataset: str, num_samples: int, seqlen: int, seed: int) -> list[torch.Tensor]:
    if "/" in dataset:
        ds_name, ds_config = dataset.split("/", 1)
    else:
        ds_name, ds_config = dataset, None
    split = "test" if ds_name == "wikitext" else "train"
    raw = load_dataset(ds_name, ds_config, split=split)
    text = "\n\n".join(t for t in raw["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]

    needed = num_samples * seqlen
    if ids.numel() < needed + 1:
        raise ValueError(
            f"dataset has {ids.numel()} tokens, need {needed + 1} for "
            f"{num_samples} samples of length {seqlen}"
        )
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, ids.numel() - seqlen - 1, size=num_samples)
    return [ids[s : s + seqlen].unsqueeze(0) for s in starts]


def pass1_bf16(args, eval_seqs: list[torch.Tensor], cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if all((cache_dir / f"{i}.npy").exists() for i in range(len(eval_seqs))):
        print(f"[pass1] cache hit ({cache_dir}); skipping BF16 forward")
        return

    print(f"[pass1] loading BF16 reference: {args.bf16_model}")
    print(f"[pass1] device_map=auto  gpu={args.gpu_mem}  cpu={args.cpu_mem}")
    arch_name, model_cls = _resolve_model_class(args.bf16_model)
    print(f"[pass1] using class: {arch_name}")
    model = model_cls.from_pretrained(
        args.bf16_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: args.gpu_mem, "cpu": args.cpu_mem},
        trust_remote_code=True,
    )
    model.eval()
    input_device = next(model.parameters()).device

    n = len(eval_seqs)
    for i, seq in enumerate(eval_seqs):
        out_path = cache_dir / f"{i}.npy"
        if out_path.exists():
            continue
        t0 = time.time()
        with torch.no_grad():
            logits = model(input_ids=seq.to(input_device)).logits
        logp = torch.log_softmax(logits.float(), dim=-1).half().cpu().numpy()
        np.save(out_path, logp)
        print(f"[pass1] {i + 1}/{n}  {time.time() - t0:5.1f}s  shape={tuple(logp.shape)}")

    del model
    gc.collect()
    _empty_cache()


def pass2_quant(args, eval_seqs: list[torch.Tensor], cache_dir: Path) -> dict:
    device = args.quant_device or _device()
    print(f"[pass2] loading quant model: {args.quant_model}  device={device}")
    # device_map="auto" with explicit max_memory lets accelerate spill peak
    # transient buffers to CPU during auto-round's QuantLinear conversion,
    # even though the converted model itself fits on XPU. Plain device_map=xpu
    # OOMs during _initialize_missing_keys on the GDN A_log path because
    # peak load memory > steady-state.
    if device == "cpu":
        device_map = "cpu"
    else:
        device_map = "auto"
    arch_name, model_cls = _resolve_model_class(args.quant_model)
    print(f"[pass2] using class: {arch_name}")
    model = model_cls.from_pretrained(
        args.quant_model,
        device_map=device_map,
        max_memory=({0: args.quant_gpu_mem, "cpu": args.cpu_mem} if device != "cpu" else None),
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device

    total_kl = 0.0
    total_tok = 0
    top1_match = 0
    n = len(eval_seqs)
    for i, seq in enumerate(eval_seqs):
        ref_logp = torch.from_numpy(np.load(cache_dir / f"{i}.npy")).float().to(device)
        with torch.no_grad():
            logits = model(input_ids=seq.to(device)).logits
        logp_q = torch.log_softmax(logits.float(), dim=-1)
        kl = (ref_logp.exp() * (ref_logp - logp_q)).sum(-1)
        total_kl += kl.sum().item()
        total_tok += kl.numel()
        top1_match += (ref_logp.argmax(-1) == logp_q.argmax(-1)).sum().item()
        print(
            f"[pass2] {i + 1}/{n}  "
            f"mean_kl={total_kl / total_tok:.4f}  top1={top1_match / total_tok:.3%}"
        )

    return {
        "mean_kl_per_token_nats": total_kl / total_tok,
        "top1_agreement": top1_match / total_tok,
        "num_samples": n,
        "seqlen": args.seqlen,
        "tokens_evaluated": total_tok,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bf16-model", required=True, help="HF id or local path to BF16 reference")
    p.add_argument("--quant-model", required=True, help="local path to AutoRound-quantized model dir")
    p.add_argument("--dataset", default="wikitext/wikitext-2-raw-v1", help="HF dataset, name or name/config")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache-dir", default=None, help="default: ~/.cache/intellm/kl-eval/<key>")
    p.add_argument("--gpu-mem", default="26GiB", help="accelerate max_memory for the BF16 reference (device-0 slot)")
    p.add_argument("--quant-gpu-mem", default="22GiB", help="accelerate max_memory for the quant model (device-0 slot)")
    p.add_argument("--cpu-mem", default="120GiB", help="accelerate max_memory for the cpu slot")
    p.add_argument("--quant-device", default=None, help="override device for the quant model (xpu|cuda|cpu)")
    p.add_argument("--out", default=None, help="optional JSON output path")
    p.add_argument("--skip-pass1", action="store_true", help="assume cache is populated; only run pass 2")
    p.add_argument("--no-subprocess", action="store_true", help="run both passes in this process (default: pass1 in subprocess for clean VRAM release)")
    p.add_argument("--_phase", choices=["pass1", "pass2"], default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.bf16_model, trust_remote_code=True)
    eval_seqs = load_eval_sequences(tokenizer, args.dataset, args.num_samples, args.seqlen, args.seed)

    key = cache_key(args.bf16_model, args.dataset, args.seqlen, args.num_samples, args.seed)
    cache_dir = (
        Path(args.cache_dir).expanduser()
        if args.cache_dir
        else Path.home() / ".cache" / "intellm" / "kl-eval" / key
    )
    print(f"[kl-eval] cache_dir = {cache_dir} phase={args._phase or 'all'}")

    # Internal recursion: run pass 1 in a child process so XPU buffers (held
    # by accelerate hooks even after del+gc.collect+empty_cache) are released
    # by OS-level process exit before we load the quant model.
    if args._phase == "pass1":
        pass1_bf16(args, eval_seqs, cache_dir)
        return
    if args._phase == "pass2":
        metrics = pass2_quant(args, eval_seqs, cache_dir)
        _emit(args, metrics)
        return

    if not args.skip_pass1:
        if args.no_subprocess:
            pass1_bf16(args, eval_seqs, cache_dir)
        else:
            print("[kl-eval] forking pass 1 as subprocess for clean VRAM release")
            child_argv = [sys.executable, str(Path(__file__).resolve()), *_unparsed_args(), "--_phase", "pass1"]
            rc = subprocess.run(child_argv).returncode
            if rc != 0:
                print(f"[kl-eval] pass 1 subprocess exited {rc}", file=sys.stderr)
                sys.exit(rc)
    metrics = pass2_quant(args, eval_seqs, cache_dir)
    _emit(args, metrics)


def _emit(args, metrics: dict) -> None:
    print("\n=== KL eval results ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if args.out:
        Path(args.out).expanduser().write_text(json.dumps(metrics, indent=2))
        print(f"\nwrote {args.out}")


def _unparsed_args() -> list[str]:
    """Forward original argv to a child invocation, stripping --_phase."""
    out: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--_phase":
            skip_next = True
            continue
        if arg.startswith("--_phase="):
            continue
        out.append(arg)
    return out


if __name__ == "__main__":
    main()
