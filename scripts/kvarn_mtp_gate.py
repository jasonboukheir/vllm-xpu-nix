#!/usr/bin/env python3
"""Validate two-token MTP acceptance and rejection/cache lifecycle behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metrics:
    drafts: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    accepted_by_position: tuple[int, int] = (0, 0)


def request(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=600) as response:
        body = response.read().decode()
    return body if payload is None else json.loads(body)


def parse_metrics(text: str) -> Metrics:
    values = {"drafts": 0, "draft_tokens": 0, "accepted_tokens": 0}
    by_position = [0, 0]
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode") or not line.rstrip().endswith(tuple("0123456789")):
            continue
        name = line.split("{", 1)[0].split()[0]
        if not name.endswith("_total"):
            continue
        try:
            value = int(float(line.rsplit(None, 1)[1]))
        except (IndexError, ValueError):
            continue
        if "num_accepted_tokens_per_pos" in name:
            match = re.search(r'position="(\d+)"', line)
            if match and int(match.group(1)) < 2:
                by_position[int(match.group(1))] += value
        elif "num_draft_tokens" in name:
            values["draft_tokens"] += value
        elif "num_accepted_tokens" in name:
            values["accepted_tokens"] += value
        elif "num_drafts" in name:
            values["drafts"] += value
    return Metrics(**values, accepted_by_position=tuple(by_position))


def delta(before: Metrics, after: Metrics) -> Metrics:
    return Metrics(
        drafts=after.drafts - before.drafts,
        draft_tokens=after.draft_tokens - before.draft_tokens,
        accepted_tokens=after.accepted_tokens - before.accepted_tokens,
        accepted_by_position=tuple(
            right - left
            for left, right in zip(
                before.accepted_by_position, after.accepted_by_position, strict=True
            )
        ),
    )


def acceptance_lengths(metrics: Metrics) -> dict[int, int]:
    """Infer qlen=3 output lengths from two draft-position counters."""
    first, second = metrics.accepted_by_position
    return {1: metrics.drafts - first, 2: first - second, 3: second}


def completion(base_url: str, model: str, prompts: list[list[int]]) -> list[list[int]]:
    response = request(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompts,
            "max_tokens": 64,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
    )
    return [choice["token_ids"] for choice in response["choices"]]


def stream_step_lengths(
    base_url: str, model: str, prompt: list[int]
) -> list[int]:
    """Return target-approved token counts for each qlen=3 verification step."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
        "stream": True,
    }
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    lengths = []
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = json.loads(line[6:])["choices"][0]
            token_ids = choice.get("token_ids") or []
            if token_ids:
                lengths.append(len(token_ids))
    if not lengths or any(length not in (1, 2, 3) for length in lengths):
        raise AssertionError(f"invalid qlen=3 streaming step lengths: {lengths}")
    return lengths


def validate_rejection_trace(lengths: list[int]) -> None:
    consecutive = next(
        (index for index in range(len(lengths) - 1) if lengths[index:index + 2] == [1, 1]),
        None,
    )
    if consecutive is None:
        raise AssertionError("MTP trace did not exercise consecutive rejection")
    if not any(length > 1 for length in lengths[consecutive + 2:]):
        raise AssertionError("MTP trace did not recover after consecutive rejection")


def run(base_url: str, model: str, prompts: list[list[int]]) -> dict[str, Any]:
    before = parse_metrics(request(f"{base_url}/metrics"))
    step_lengths = stream_step_lengths(base_url, model, prompts[0])
    validate_rejection_trace(step_lengths)
    first = completion(base_url, model, prompts)
    second = completion(base_url, model, prompts)
    after = parse_metrics(request(f"{base_url}/metrics"))
    if first != second:
        raise AssertionError("replayed MTP requests produced different token IDs")
    completion_json = json.dumps(first, separators=(",", ":")).encode()
    observed = delta(before, after)
    if observed.draft_tokens != 2 * observed.drafts:
        raise AssertionError(
            "server did not draft exactly two tokens per MTP verification step"
        )
    lengths = acceptance_lengths(observed)
    missing = [length for length, count in lengths.items() if count <= 0]
    if missing:
        raise AssertionError(f"MTP run did not exercise acceptance lengths {missing}")
    return {
        "qlen": 3,
        "speculative_tokens": 2,
        "replay_token_ids_identical": True,
        "completion_token_ids": first,
        "completion_token_ids_sha256": hashlib.sha256(completion_json).hexdigest(),
        "stream_step_lengths": step_lengths,
        "consecutive_rejection_recovered": True,
        "drafts": observed.drafts,
        "draft_tokens": observed.draft_tokens,
        "accepted_tokens": observed.accepted_tokens,
        "draft_acceptance_rate": (
            observed.accepted_tokens / observed.draft_tokens
            if observed.draft_tokens
            else 0.0
        ),
        "accepted_by_position": list(observed.accepted_by_position),
        "acceptance_lengths": lengths,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prompts = json.loads(args.prompt_token_ids.read_text())
    if not isinstance(prompts, list) or not prompts or not all(
        isinstance(prompt, list)
        and prompt
        and all(isinstance(token, int) and token >= 0 for token in prompt)
        for prompt in prompts
    ):
        raise ValueError("prompt-token-ids must contain a non-empty list of token lists")
    document = run(args.base_url.rstrip("/"), args.model, prompts)
    rendered = json.dumps(document, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
