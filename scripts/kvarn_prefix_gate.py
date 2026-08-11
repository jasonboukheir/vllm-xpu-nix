#!/usr/bin/env python3
"""Validate prefix-cache reuse against an uncached completion."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def completion(base_url: str, model: str, token_ids: list[int]) -> dict[str, Any]:
    return request_json(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": token_ids,
            "max_tokens": 8,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": 5,
            "return_token_ids": True,
        },
    )


def comparable(response: dict[str, Any]) -> dict[str, Any]:
    choice = response["choices"][0]
    return {
        "token_ids": choice.get("token_ids"),
        "text": choice.get("text"),
        "finish_reason": choice.get("finish_reason"),
        "logprobs": choice.get("logprobs"),
    }


def compare_results(
    uncached: dict[str, Any], cached: dict[str, Any], tolerance: float
) -> tuple[float, bool]:
    """Require identical decoding and bound comparable logprob drift.

    The fifth top-logprob entry can change across the fp16-sink-to-int4 prefix
    lifecycle when candidates straddle the top-k cutoff. That is not a decode
    mismatch and the missing cross-probability is unknowable, so compare the
    selected-token probabilities and the intersection of each top-k support,
    while reporting whether the complete support was identical.
    """
    for key in ("token_ids", "text", "finish_reason"):
        if cached[key] != uncached[key]:
            raise AssertionError(f"cached {key} differs from uncached result")

    left = uncached["logprobs"]
    right = cached["logprobs"]
    for key in ("tokens", "text_offset"):
        if left[key] != right[key]:
            raise AssertionError(f"cached logprobs.{key} differs")

    deltas = [
        abs(a - b)
        for a, b in zip(left["token_logprobs"], right["token_logprobs"], strict=True)
    ]
    support_identical = True
    for left_top, right_top in zip(
        left["top_logprobs"], right["top_logprobs"], strict=True
    ):
        support_identical &= left_top.keys() == right_top.keys()
        deltas.extend(
            abs(left_top[token] - right_top[token])
            for token in left_top.keys() & right_top.keys()
        )
    maximum = max(deltas, default=0.0)
    if maximum > tolerance:
        raise AssertionError(
            f"cached logprob drift {maximum:.6g} exceeds tolerance {tolerance:.6g}"
        )
    return maximum, support_identical


def repeat_to_length(seed: list[int], length: int) -> list[int]:
    return (seed * ((length + len(seed) - 1) // len(seed)))[:length]


def run_case(
    base_url: str, model: str, seed: list[int], length: int, tolerance: float
) -> dict[str, Any]:
    # Give every case a distinct first token so shorter cases cannot prime a
    # longer case accidentally. Both prompts then share exactly ``length``
    # tokens and differ at the final token.
    markers = [seed[(length // len(seed)) % len(seed)], seed[length % len(seed)]]
    common = (markers + repeat_to_length(seed, max(0, length - 2)))[:length]
    prime = common + [seed[-2]]
    target = common + [seed[-1]]

    # First target is necessarily uncached for this case. The distinct-suffix
    # request then exercises reuse of its common prefix, and the final target
    # exercises the same cached history while preserving identical inputs for
    # the content comparison.
    uncached = comparable(completion(base_url, model, target))
    completion(base_url, model, prime)
    cached = comparable(completion(base_url, model, target))
    maximum, support_identical = compare_results(uncached, cached, tolerance)
    return {
        "shared_prefix_tokens": length,
        "decoded_output_identical": True,
        "top_logprob_support_identical": support_identical,
        "max_logprob_delta": maximum,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed-token-ids", type=Path, required=True)
    parser.add_argument("--lengths", default="127,128,129,4096")
    # BF16 MTP controls reach 0.437 at the 129-token boundary while producing
    # identical tokens.  Use a rounded 0.5 envelope as the numerical gate;
    # exact decoded tokens remain the hard cache-correctness requirement.
    parser.add_argument("--logprob-tolerance", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    seed = json.loads(args.seed_token_ids.read_text())
    if (
        not isinstance(seed, list)
        or len(seed) < 2
        or not all(isinstance(token, int) and token >= 0 for token in seed)
    ):
        raise ValueError("seed-token-ids must be a JSON list of at least two IDs")
    results = [
        run_case(
            args.base_url.rstrip("/"),
            args.model,
            seed,
            int(length),
            args.logprob_tolerance,
        )
        for length in args.lengths.split(",")
    ]
    document = {"model": args.model, "cases": results}
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
