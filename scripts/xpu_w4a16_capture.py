"""Capture loaded W4A16 operands without changing the model's GEMM dispatch.

Run in the immutable serving Python environment on an idle XPU. The in-process
engine is diagnostic: capture overhead makes its timings unsuitable for service
qualification. Only selected projections are copied, to CPU/disk.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from scripts.kvarn_vision_run import MODEL, REVISION


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "contiguous": tensor.is_contiguous(),
        "bytes": tensor.numel() * tensor.element_size(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenize", type=Path, action="append", required=True)
    parser.add_argument(
        "--projection", action="append", required=True, help="Exact loaded layer prefix"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "capture-source.py")
    # Install hooks in the worker's process before any model loading.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import vllm
    import vllm_xpu_kernels._xpu_C as kernels
    from vllm import LLM, SamplingParams
    from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
        XPUwNa16LinearKernel,
    )

    report = {
        "argv": sys.argv,
        "model": MODEL,
        "revision": REVISION,
        "torch": torch.__version__,
        "vllm": vllm.__file__,
        "kernels": kernels.__file__,
        "kernel_sha256": digest(kernels.__file__),
        "device": str(torch.xpu.get_device_properties(0)),
        "loaded": {},
        "captures": [],
        "requests": [],
        "diagnostic_only": True,
    }

    def save_report():
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    original_load = XPUwNa16LinearKernel.process_weights_after_loading
    original_apply = XPUwNa16LinearKernel.apply_weights
    active = False
    captured = set()
    weights_saved = set()

    def loaded(self, layer):
        original_load(self, layer)
        q, scales, zero, gidx = self._get_weight_params(layer)
        report["loaded"][layer.prefix] = {
            "q_nt": describe(q.t()),
            "scales": describe(scales),
            "zero": describe(zero),
            "zero_values": zero.unique().cpu().tolist(),
            "g_idx": describe(gidx),
            "group_size": self.config.group_size,
            "has_g_idx": self.config.has_g_idx,
        }

    def apply(self, layer, x, bias=None):
        output = original_apply(self, layer, x, bias)
        m = x.reshape(-1, x.shape[-1]).shape[0]
        key = (layer.prefix, m)
        if (
            active
            and layer.prefix in args.projection
            and m in (2048, 1535, 2047)
            and key not in captured
        ):
            q, scales, zero, gidx = self._get_weight_params(layer)
            if layer.prefix not in weights_saved:
                path = args.output / f"{layer.prefix}.weights.pt"
                torch.save(
                    {
                        "q_nt": q.t().cpu(),
                        "scales": scales.cpu(),
                        "zero": zero.cpu(),
                        "g_idx": None if gidx is None else gidx.cpu(),
                        "group_size": self.config.group_size,
                    },
                    path,
                )
                weights_saved.add(layer.prefix)
            path = args.output / f"{layer.prefix}.m{m}.pt"
            torch.save(
                {
                    "x": x.cpu(),
                    "output": output.cpu(),
                    "bias": None if bias is None else bias.cpu(),
                },
                path,
            )
            report["captures"].append(
                {
                    "prefix": layer.prefix,
                    "m": m,
                    "x": describe(x),
                    "bias": describe(bias),
                    "output": describe(output),
                    "file": path.name,
                    "sha256": digest(path),
                }
            )
            captured.add(key)
            save_report()
            print(f"Captured {key}", flush=True)
        return output

    XPUwNa16LinearKernel.process_weights_after_loading = loaded
    XPUwNa16LinearKernel.apply_weights = apply
    report["engine_kwargs"] = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": "bfloat16",
        "quantization": "compressed-tensors",
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": 0.90,
        "max_model_len": 65536,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 2048,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "mm_processor_kwargs": {"min_pixels": 200704, "max_pixels": 200704},
        "seed": 0,
    }
    save_report()
    start = time.perf_counter()
    try:
        llm = LLM(**report["engine_kwargs"])
        report["model_load_seconds"] = time.perf_counter() - start
        active = True
        for index, path in enumerate(args.tokenize):
            raw = json.loads(path.read_text())
            tokens = raw["tokens"]
            if len(tokens) not in (16383, 65023) or raw["count"] != len(tokens):
                raise ValueError("expected retained exact 16K/65K tokenization")
            shutil.copy2(path, args.output / f"request-{index}-tokenize.json")
            result = llm.generate(
                [{"prompt_token_ids": tokens}],
                SamplingParams(
                    temperature=0,
                    seed=0,
                    max_tokens=512,
                    min_tokens=512,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )[0]
            report["requests"].append(
                {
                    "prompt_token_ids": result.prompt_token_ids,
                    "output_token_ids": list(result.outputs[0].token_ids),
                    "text": result.outputs[0].text,
                }
            )
            save_report()
        expected = {(p, m) for p in args.projection for m in (2048, 1535, 2047)}
        if captured != expected:
            raise ValueError(f"missing operand captures: {expected - captured}")
        report["weights_sha256"] = {
            p.name: digest(p) for p in args.output.glob("*.weights.pt")
        }
        report["status"] = "captured"
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        save_report()
        XPUwNa16LinearKernel.process_weights_after_loading = original_load
        XPUwNa16LinearKernel.apply_weights = original_apply


if __name__ == "__main__":
    main()
