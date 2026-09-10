"""Capture actual prefill attention operands in the frozen diagnostic engine."""

import argparse
import inspect
import json
import os
import shutil
import sys
from pathlib import Path

from scripts.kvarn_vision_run import MODEL, REVISION
from scripts.xpu_w4a16_capture import describe, digest

EXTENTS = {
    (2048, 2048),
    (2048, 8192),
    (2047, 16383),
    (2048, 32768),
    (2048, 63488),
    (1535, 65023),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dtype", choices=("auto", "kvarn_k4v4_g128_compact"), required=True
    )
    parser.add_argument("--tokenize", type=Path, action="append", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output / "capture-source.py")
    plan_path = (
        Path(__file__).resolve().parents[1] / "fixtures/xpu-attention-d256-plan.json"
    )
    plan = json.loads(plan_path.read_text())
    shutil.copy2(plan_path, args.output / "plan.json")
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    import torch
    import vllm
    import vllm._xpu_ops as xpu_module
    import vllm_xpu_kernels._vllm_fa2_C as binding
    from vllm import LLM, SamplingParams

    original = xpu_module.flash_attn_varlen_func
    signature = inspect.signature(original)
    library = Path(plan["control_library"])
    mappings = Path("/proc/self/maps").read_text()
    if str(library) not in mappings:
        raise ValueError("capture must use the original immutable attention library")
    report = {
        "argv": sys.argv,
        "torch": torch.__version__,
        "vllm": vllm.__file__,
        "model": MODEL,
        "revision": REVISION,
        "cache_dtype": args.cache_dtype,
        "binding": binding.__file__,
        "binding_sha256": digest(binding.__file__),
        "device": str(torch.xpu.get_device_properties(0)),
        "calls": [],
        "captures": [],
        "requests": [],
        "status": "starting",
        "diagnostic_only": True,
        "plan_sha256": digest(plan_path),
        "attention_library": str(library),
        "attention_library_sha256": digest(library),
    }
    active = False
    request_index = -1
    captured = set()

    def save():
        (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")

    def capture(*positional, **keywords):
        bound = signature.bind(*positional, **keywords)
        bound.apply_defaults()
        values = bound.arguments
        result = original(*positional, **keywords)
        q, k, v = (values[name] for name in ("q", "k", "v"))
        m, length = int(values["max_seqlen_q"]), int(values["max_seqlen_k"])
        if not active or m <= 1 or tuple(q.shape[1:]) != (24, 256):
            return result
        paged = values["block_table"] is not None
        call = {
            "request": request_index,
            "m": m,
            "k": length,
            "paged": paged,
            "q": describe(q),
            "key": describe(k),
            "value": describe(v),
            "key_value_shared_storage": k.untyped_storage().data_ptr()
            == v.untyped_storage().data_ptr(),
            "key_value_offset_delta": v.storage_offset() - k.storage_offset(),
            "out": describe(result[0] if isinstance(result, tuple) else result),
            "causal": values["causal"],
            "softmax_scale": values["softmax_scale"],
            "window_size": values["window_size"],
        }
        report["calls"].append(call)
        if (m, length) not in EXTENTS or (m, length) in captured:
            return result
        if not values["causal"] or q.shape[0] != m:
            raise ValueError("capture requires B1 bottom-right causal prefill")
        tensors = {
            "q": q.detach().cpu().contiguous(),
            "output": (result[0] if isinstance(result, tuple) else result)
            .detach()
            .cpu()
            .contiguous(),
        }
        for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_k"):
            tensors[name] = None if values[name] is None else values[name].cpu()
        if paged:
            pages = (length + k.shape[1] - 1) // k.shape[1]
            physical = values["block_table"][0, :pages].to(torch.long)
            tensors["key"] = k.index_select(0, physical).cpu()
            tensors["value"] = v.index_select(0, physical).cpu()
            tensors["physical_pages"] = physical.cpu()
            tensors["block_table"] = torch.arange(pages, dtype=torch.int32)[None]
            call["block_table"] = describe(values["block_table"])
            call["page_snapshot"] = (
                "Used pages in logical order; original physical page IDs and tensor strides retained"
            )
        else:
            tensors["key"] = k.detach().cpu().contiguous()
            tensors["value"] = v.detach().cpu().contiguous()
            tensors["block_table"] = None
        name = f"q{m}-k{length}.pt"
        torch.save(tensors, args.output / name)
        captured.add((m, length))
        report["captures"].append(
            {**call, "file": name, "sha256": digest(args.output / name)}
        )
        save()
        print(
            f"Captured {args.cache_dtype} Q={m} K={length} paged={paged} dtype={q.dtype}",
            flush=True,
        )
        return result

    xpu_module.flash_attn_varlen_func = capture
    report["engine_kwargs"] = {
        "model": MODEL,
        "revision": REVISION,
        "dtype": "bfloat16",
        "quantization": "compressed-tensors",
        "kv_cache_dtype": args.cache_dtype,
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
    save()
    try:
        llm = LLM(**report["engine_kwargs"])
        active = True
        for request_index, path in enumerate(args.tokenize):
            raw = json.loads(path.read_text())
            if raw["count"] != len(raw["tokens"]) or raw["count"] not in (16383, 65023):
                raise ValueError("requires exact retained16K/65K tokenizations")
            shutil.copy2(path, args.output / f"request-{request_index}-tokenize.json")
            response = llm.generate(
                [{"prompt_token_ids": raw["tokens"]}],
                SamplingParams(
                    temperature=0,
                    seed=0,
                    min_tokens=512,
                    max_tokens=512,
                    ignore_eos=True,
                ),
                use_tqdm=False,
            )[0]
            if (
                response.prompt_token_ids != raw["tokens"]
                or len(response.outputs[0].token_ids) != 512
            ):
                raise ValueError(
                    "diagnostic request did not preserve exact prompt/output extents"
                )
            report["requests"].append(
                {
                    "prompt_token_ids": response.prompt_token_ids,
                    "output_token_ids": list(response.outputs[0].token_ids),
                    "text": response.outputs[0].text,
                }
            )
            save()
        if captured != EXTENTS:
            raise ValueError(f"missing actual attention extents: {EXTENTS - captured}")
        report["status"] = "captured"
    except Exception as error:
        report["status"] = "stopped"
        report["error"] = repr(error)
        raise
    finally:
        xpu_module.flash_attn_varlen_func = original
        save()


if __name__ == "__main__":
    main()
