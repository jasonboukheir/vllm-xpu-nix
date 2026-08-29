#!/usr/bin/env python3
"""Run deterministic Kvarn service corruption and lifecycle gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

METRIC_NAMES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def http_request(
    url: str, payload: dict[str, Any] | None = None, timeout: float = 600
) -> str:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def parse_metrics(text: str) -> dict[str, float]:
    values: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
    pattern = re.compile(
        r"^(vllm:(?:num_requests_running|num_requests_waiting|"
        r"kv_cache_usage_perc))(?:\{[^}]*\})?\s+([-+0-9.eE]+)$"
    )
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            values[match.group(1)].append(float(match.group(2)))
    missing = [name for name, found in values.items() if not found]
    if missing:
        raise ValueError(
            "vLLM metrics response is missing required gauges: "
            + ", ".join(missing)
        )
    return {
        name: (
            max(found)
            if name.endswith("kv_cache_usage_perc")
            else sum(found)
        )
        for name, found in values.items()
    }


def wait_for_idle(base_url: str, timeout: float) -> dict[str, float]:
    deadline = time.monotonic() + timeout
    last: dict[str, float] = {}
    while time.monotonic() < deadline:
        last = parse_metrics(http_request(f"{base_url}/metrics", timeout=timeout))
        if all(last.get(name, math.inf) == 0 for name in METRIC_NAMES):
            return last
        time.sleep(1)
    raise TimeoutError(f"vLLM metrics did not return to idle: {last}")


def repeated_span_three_times(token_ids: list[int], width: int = 16) -> bool:
    return any(
        token_ids[start : start + width]
        == token_ids[start + width : start + 2 * width]
        == token_ids[start + 2 * width : start + 3 * width]
        for start in range(max(0, len(token_ids) - 3 * width + 1))
    )


def short_period_run(
    token_ids: list[int], max_period: int = 32, min_run: int = 128
) -> bool:
    for period in range(1, max_period + 1):
        matched = 0
        for index in range(period, len(token_ids)):
            if token_ids[index] == token_ids[index - period]:
                matched += 1
                if matched + period >= min_run:
                    return True
            else:
                matched = 0
    return False


def completion(
    base_url: str,
    model: str,
    fixture: dict[str, Any],
    default_max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    max_tokens = int(fixture.get("max_tokens", default_max_tokens))
    payload = {
        "model": model,
        "prompt": fixture["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "return_token_ids": True,
    }
    started = time.monotonic()
    response = json.loads(
        http_request(f"{base_url}/v1/completions", payload, timeout=timeout)
    )
    elapsed = time.monotonic() - started
    choice = response["choices"][0]
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise AssertionError("completion did not return integer token_ids")
    if len(token_ids) != max_tokens:
        raise AssertionError(
            f"{fixture['id']} returned {len(token_ids)} of {max_tokens} tokens"
        )
    if repeated_span_three_times(token_ids):
        raise AssertionError(f"{fixture['id']} repeated a 16-token span three times")
    if short_period_run(token_ids):
        raise AssertionError(f"{fixture['id']} collapsed into a short-period loop")
    rendered = json.dumps(token_ids, separators=(",", ":")).encode()
    return {
        "id": fixture["id"],
        "prompt": fixture["prompt"],
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text", ""),
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(rendered).hexdigest(),
    }


def cancel_stream(
    base_url: str,
    model: str,
    fixture: dict[str, Any],
    timeout: float,
    after_events: int,
) -> int:
    payload = {
        "model": model,
        "prompt": fixture["prompt"],
        "max_tokens": max(256, after_events + 1),
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    observed = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith("data: ") and line != "data: [DONE]":
                observed += 1
                if observed >= after_events:
                    break
    if observed < after_events:
        raise AssertionError("stream ended before the cancellation checkpoint")
    return observed


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or len(fixtures) < args.concurrency:
        raise ValueError("fixtures must contain at least --concurrency entries")
    for fixture in fixtures:
        if (
            not isinstance(fixture, dict)
            or not isinstance(fixture.get("id"), str)
            or not isinstance(fixture.get("prompt"), str)
        ):
            raise TypeError("each fixture requires string id and prompt fields")
        if int(fixture.get("max_tokens", args.max_tokens)) < 2048:
            raise ValueError("each fixture must request at least 2048 output tokens")

    before = wait_for_idle(args.base_url, args.idle_timeout)
    first = [
        completion(
            args.base_url,
            args.model,
            fixture,
            args.max_tokens,
            args.timeout,
        )
        for fixture in fixtures
    ]
    second = [
        completion(
            args.base_url,
            args.model,
            fixture,
            args.max_tokens,
            args.timeout,
        )
        for fixture in fixtures
    ]
    if [item["token_ids"] for item in first] != [item["token_ids"] for item in second]:
        raise AssertionError("same-process replay produced different token IDs")

    batch_fixtures = fixtures[: args.concurrency]
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        batch = list(
            executor.map(
                lambda fixture: completion(
                    args.base_url,
                    args.model,
                    fixture,
                    args.max_tokens,
                    args.timeout,
                ),
                batch_fixtures,
            )
        )
    expected = {item["id"]: item["token_ids"] for item in first}
    if any(item["token_ids"] != expected[item["id"]] for item in batch):
        raise AssertionError("concurrent output diverged from isolated output")

    cancellation: dict[str, Any] | None = None
    if args.cancel_after_events:
        events = cancel_stream(
            args.base_url,
            args.model,
            fixtures[0],
            args.timeout,
            args.cancel_after_events,
        )
        replacement = completion(
            args.base_url,
            args.model,
            fixtures[0],
            args.max_tokens,
            args.timeout,
        )
        if replacement["token_ids"] != expected[fixtures[0]["id"]]:
            raise AssertionError("replacement after cancellation was not isolated")
        cancellation = {
            "stream_events_before_close": events,
            "replacement": replacement,
        }

    after = wait_for_idle(args.base_url, args.idle_timeout)
    provenance = (
        json.loads(args.provenance.read_text(encoding="utf-8"))
        if args.provenance
        else {}
    )
    return {
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "metrics_before": before,
        "metrics_after": after,
        "same_process_replay_identical": True,
        "concurrent_isolation_identical": True,
        "isolated_first": first,
        "isolated_replay": second,
        "concurrent": batch,
        "cancellation": cancellation,
        "provenance": provenance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--cancel-after-events", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=120)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.max_tokens < 2048:
        parser.error("--max-tokens must be at least 2048")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    output = args.output.resolve()
    if not args.allow_tmp and output.is_relative_to(Path("/tmp")):
        parser.error("--output must be durable (outside /tmp)")
    return args


def main() -> None:
    args = parse_args()
    document = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(document, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
