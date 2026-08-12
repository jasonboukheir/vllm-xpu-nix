#!/usr/bin/env python3
"""Warm MTP2 inference through vLLM's real HTTP scheduler lifecycle."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any


def request(url: str, payload: dict[str, Any] | None = None, timeout: float = 10) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode()
    return json.loads(body)


def wait_for_model(base_url: str, model: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            response = request(f"{base_url}/v1/models", timeout=2)
            if model in {item["id"] for item in response.get("data", [])}:
                return
            last_error = RuntimeError(f"model {model!r} is not advertised")
        except (OSError, ValueError, urllib.error.URLError) as error:
            last_error = error
        time.sleep(1)
    raise TimeoutError(f"vLLM did not become ready within {timeout}s: {last_error}")


def completion(
    base_url: str,
    model: str,
    prompts: list[list[int]],
    max_tokens: int,
) -> None:
    response = request(
        f"{base_url}/v1/completions",
        {
            "model": model,
            "prompt": prompts,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
        timeout=600,
    )
    if len(response.get("choices", [])) != len(prompts):
        raise RuntimeError(f"warmup returned an invalid response: {response}")


def run(base_url: str, model: str, ready_timeout: float, max_tokens: int) -> None:
    wait_for_model(base_url, model, ready_timeout)
    # Cross one 128-token compact KVarN tile in B1, then compile the ragged B4
    # qlen=3 scheduler/rejection shapes used by concurrent short requests.
    completion(base_url, model, [list(range(1, 135))], max_tokens)
    completion(
        base_url,
        model,
        [list(range(1, length + 1)) for length in (14, 15, 16, 17)],
        max_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ready-timeout", type=float, default=900)
    parser.add_argument("--max-tokens", type=int, default=8)
    args = parser.parse_args()
    if args.max_tokens < 3:
        raise ValueError("--max-tokens must cover at least one full qlen=3 step")
    run(args.base_url.rstrip("/"), args.model, args.ready_timeout, args.max_tokens)


if __name__ == "__main__":
    main()
