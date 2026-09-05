# SPDX-License-Identifier: Apache-2.0
"""Record raw logits from a persistent, teacher-forced vLLM request.

This utility is intended for paired KV-cache accuracy experiments.  Run it
twice with identical prompt and forced token IDs (for example, once with BF16
KV and once with KVarN K4V4), then compare the resulting ``.npz`` files.

Unlike scoring a successively longer prompt, this advances one engine request
one token at a time.  The custom logits processor forces the next reference
token *after* the sampler has copied the unmodified model logits for output.
Consequently the artifact measures decode with the request's persistent cache.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

if kernel_library := os.environ.get("VLLM_XPU_KERNELS_LIBRARY"):
    # Accuracy iteration commonly validates a locally built kernel before its
    # package input is updated.  Load that exact ABI in both the driver and the
    # spawned engine process instead of silently exercising an older closure.
    torch.ops.load_library(kernel_library)

from vllm import LLM, SamplingParams, TokensPrompt
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import FlatLogprobs, SampleLogprobs
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)


class _ForceTokenSequence:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if os.environ.get("KVARN_FORCED_VALIDATE_FINITE", "0") == "1":
            nan_count = int(torch.isnan(logits).sum().item())
            posinf_count = int(torch.isposinf(logits).sum().item())
            # Some model vocabularies intentionally mask invalid/reserved
            # entries with -inf.  Those are valid sampling logits; NaN and
            # +inf are not.
            if nan_count or posinf_count:
                neginf_count = int(torch.isneginf(logits).sum().item())
                raise RuntimeError(
                    "forced-decode received invalid model logits before "
                    f"forcing step {len(output_ids)}: nan={nan_count} "
                    f"posinf={posinf_count} neginf={neginf_count} "
                    f"total={logits.numel()}"
                )
        token_id = self.token_ids[len(output_ids)]
        value = logits[token_id].clone()
        logits.fill_(float("-inf"))
        logits[token_id] = value
        return logits


class ForcedTokenSequenceLogitsProcessor(AdapterLogitsProcessor):
    """Force ``extra_args['forced_token_ids']`` in order for each request."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        token_ids = params.extra_args and params.extra_args.get("forced_token_ids")
        if token_ids is None:
            return
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                not isinstance(token_id, int) or token_id < 0 for token_id in token_ids
            )
        ):
            raise VLLMValidationError(
                "forced_token_ids must be a non-empty list of non-negative integers"
            )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self, params: SamplingParams
    ) -> RequestLogitsProcessor | None:
        self.validate_params(params)
        token_ids = params.extra_args and params.extra_args.get("forced_token_ids")
        return None if token_ids is None else _ForceTokenSequence(token_ids)


def _read_token_ids(path: Path) -> list[int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"{path} must contain a JSON list of integer token IDs")
    return value


def _logit_rows(
    logprobs: SampleLogprobs, *, expected_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return padded [step, rank] token-id and raw-logit arrays."""
    if len(logprobs) != expected_steps:
        raise RuntimeError(
            f"expected {expected_steps} decode rows, received {len(logprobs)}"
        )

    if isinstance(logprobs, FlatLogprobs):
        starts, ends = logprobs.start_indices, logprobs.end_indices
        row_ids = [logprobs.token_ids[start:end] for start, end in zip(starts, ends)]
        row_values = [logprobs.logprobs[start:end] for start, end in zip(starts, ends)]
    else:
        row_ids = [list(row) for row in logprobs]
        row_values = [
            [row[token_id].logprob for token_id in token_ids]
            for row, token_ids in zip(logprobs, row_ids)
        ]

    width = max(map(len, row_ids))
    token_ids = np.full((expected_steps, width), -1, dtype=np.int32)
    raw_logits = np.full((expected_steps, width), np.nan, dtype=np.float32)
    for step, (ids, values) in enumerate(zip(row_ids, row_values)):
        token_ids[step, : len(ids)] = ids
        raw_logits[step, : len(values)] = values
    return token_ids, raw_logits


def run(args: argparse.Namespace) -> None:
    prompt_ids = _read_token_ids(args.prompt_token_ids)
    forced_ids = _read_token_ids(args.forced_token_ids)
    engine_kwargs: dict[str, Any] = {}
    if args.engine_kwargs is not None:
        engine_kwargs = json.loads(args.engine_kwargs.read_text(encoding="utf-8"))
        if not isinstance(engine_kwargs, dict):
            raise ValueError("--engine-kwargs must contain a JSON object")

    # raw_logits is essential: probabilities after forcing are intentionally
    # degenerate, while raw logits are copied before the processor runs.
    llm = LLM(
        model=args.model,
        logits_processors=[ForcedTokenSequenceLogitsProcessor],
        logprobs_mode="raw_logits",
        max_logprobs=-1 if args.full_logits else args.top_k,
        **engine_kwargs,
    )
    params = SamplingParams(
        temperature=0.0,
        max_tokens=len(forced_ids),
        min_tokens=len(forced_ids),
        ignore_eos=True,
        detokenize=False,
        logprobs=-1 if args.full_logits else args.top_k,
        flat_logprobs=True,
        extra_args={"forced_token_ids": forced_ids},
    )
    request = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False
    )[0]
    output = request.outputs[0]
    if list(output.token_ids) != forced_ids:
        raise RuntimeError("engine output did not match the forced token sequence")
    if output.logprobs is None:
        raise RuntimeError("engine returned no logits")
    token_ids, raw_logits = _logit_rows(output.logprobs, expected_steps=len(forced_ids))
    np.savez_compressed(
        args.output,
        artifact_schema_version=np.asarray(2, dtype=np.int32),
        model=np.asarray(args.model),
        engine_kwargs_json=np.asarray(json.dumps(engine_kwargs, sort_keys=True)),
        prompt_token_ids=np.asarray(prompt_ids, dtype=np.int32),
        forced_token_ids=np.asarray(forced_ids, dtype=np.int32),
        logit_token_ids=token_ids,
        raw_logits=raw_logits,
        full_logits=np.asarray(args.full_logits),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-token-ids", type=Path, required=True)
    parser.add_argument("--forced-token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-kwargs", type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full-logits", action="store_true")
    group.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
