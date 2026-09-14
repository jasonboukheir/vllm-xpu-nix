#!/usr/bin/env python3
"""Replay a frozen case set through one persistent engine per cache format.

Uses the existing forced-token processor and raw-logit artifact format. Run
serially with the production services stopped; timings here include diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from scripts import kvarn_forced_decode as forced
from scripts.kvarn_perf_run import write_json_atomic

CACHE_DTYPES = (
    "bfloat16",
    "kvarn_k4v4_g128_compact",
    "kvarn_k4v2_g128_compact",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args) -> None:
    plan = json.loads(args.plan.read_text())
    cases = plan["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("the plan requires uniquely named cases")
    for case in cases:
        if Path(case["id"]).name != case["id"] or case["id"] in (".", ".."):
            raise ValueError("case IDs must be plain filenames")
        prompt = case["prompt_token_ids"]
        if not prompt or any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError("prompt tokens must be nonnegative integers")
        if type(case["output_tokens"]) is not int or case["output_tokens"] < 1:
            raise ValueError("output_tokens must be a positive integer")
    if args.teacher is None and args.cache_dtype != "bfloat16":
        raise ValueError("teacher generation requires explicit BF16 KV")
    teacher = json.loads(args.teacher.read_text()) if args.teacher else None
    if teacher and teacher["plan_sha256"] != digest(args.plan):
        raise ValueError("teacher tokens do not belong to this frozen plan")
    options = dict(plan["engine_options"])
    options.update(revision=plan["revision"], kv_cache_dtype=args.cache_dtype)
    if options.get("speculative_config") or options.get("enable_prefix_caching"):
        raise ValueError("numerical replay requires MTP and prefix caching off")
    if any(
        len(c["prompt_token_ids"]) + c["output_tokens"] > options["max_model_len"]
        for c in cases
    ):
        raise ValueError("a case exceeds the frozen engine capacity")
    args.output.mkdir(parents=True, exist_ok=False)
    import vllm
    import vllm_xpu_kernels._vllm_fa2_C as fa

    identity = {
        "model": plan["model"],
        "revision": plan["revision"],
        "engine_options": options,
        "plan_sha256": digest(args.plan),
        "teacher_sha256": digest(args.teacher) if args.teacher else None,
        "vllm_file": vllm.__file__,
        "vllm_version": vllm.__version__,
        "native_library": fa.__file__,
        "native_library_sha256": digest(Path(fa.__file__)),
        "torch": torch.__version__,
        "gpu": torch.xpu.get_device_name(0),
        "worker_pid": os.getpid(),
        "started_unix": time.time(),
        "completed": [],
        "note": "Top-50 plus selected-token raw logits; no full-distribution KL or performance claim.",
    }
    write_json_atomic(args.output / "manifest.json", identity)
    llm = LLM(
        model=plan["model"],
        logits_processors=[forced.ForcedTokenSequenceLogitsProcessor],
        logprobs_mode="raw_logits",
        max_logprobs=50,
        **options,
    )
    teachers = {"plan_sha256": digest(args.plan), "cases": {}}
    for case in cases:
        name, prompt, count = (
            case["id"],
            case["prompt_token_ids"],
            case["output_tokens"],
        )
        started = time.monotonic()
        if teacher is None:
            generated = llm.generate(
                [TokensPrompt(prompt_token_ids=prompt)],
                SamplingParams(
                    temperature=0,
                    max_tokens=count,
                    min_tokens=count,
                    ignore_eos=True,
                    detokenize=False,
                ),
                use_tqdm=False,
            )[0].outputs[0]
            token_ids = list(generated.token_ids)
            if len(token_ids) != count:
                raise RuntimeError("teacher did not produce the frozen output length")
        else:
            token_ids = teacher["cases"][name]
            if len(token_ids) != count or any(
                type(t) is not int or t < 0 for t in token_ids
            ):
                raise ValueError("invalid frozen continuation")
        teachers["cases"][name] = token_ids
        write_json_atomic(args.output / "teacher.json", teachers)
        params = SamplingParams(
            temperature=0,
            max_tokens=count,
            min_tokens=count,
            ignore_eos=True,
            detokenize=False,
            logprobs=50,
            flat_logprobs=True,
            extra_args={"forced_token_ids": token_ids},
        )
        result = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False
        )[0].outputs[0]
        if list(result.token_ids) != token_ids or result.logprobs is None:
            raise RuntimeError("engine did not return the frozen tokens and logits")
        ids, raw = forced._logit_rows(result.logprobs, expected_steps=count)
        np.savez_compressed(
            args.output / f"{name}.npz",
            artifact_schema_version=np.asarray(2, dtype=np.int32),
            model=np.asarray(plan["model"]),
            engine_kwargs_json=np.asarray(json.dumps(options, sort_keys=True)),
            prompt_token_ids=np.asarray(prompt, dtype=np.int32),
            forced_token_ids=np.asarray(token_ids, dtype=np.int32),
            logit_token_ids=ids,
            raw_logits=raw,
            full_logits=np.asarray(False),
        )
        from scripts.kvarn_compare_logits import load_artifact

        load_artifact(args.output / f"{name}.npz")
        identity["completed"].append(
            {
                "id": name,
                "seconds": time.monotonic() - started,
                "artifact_sha256": digest(args.output / f"{name}.npz"),
            }
        )
        write_json_atomic(args.output / "manifest.json", identity)
        print(json.dumps(identity["completed"][-1]), flush=True)
    identity["finished_unix"] = time.time()
    identity["status"] = "captured"
    write_json_atomic(args.output / "manifest.json", identity)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--cache-dtype", choices=CACHE_DTYPES, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
