#!/usr/bin/env python3
"""Run deterministic Kvarn service corruption and lifecycle gates."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

METRIC_NAMES = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    """Persist a checkpoint without exposing a partially written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def replay_mismatches(
    first: list[dict[str, Any]], replay: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for expected, actual in zip(first, replay, strict=True):
        expected_ids = expected["token_ids"]
        actual_ids = actual["token_ids"]
        if expected_ids == actual_ids:
            continue
        common_prefix = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(expected_ids, actual_ids, strict=False)
                )
                if left != right
            ),
            min(len(expected_ids), len(actual_ids)),
        )
        mismatches.append(
            {
                "id": expected["id"],
                "expected_sha256": expected["token_ids_sha256"],
                "actual_sha256": actual["token_ids_sha256"],
                "common_prefix_tokens": common_prefix,
                "expected_token_at_divergence": (
                    expected_ids[common_prefix]
                    if common_prefix < len(expected_ids)
                    else None
                ),
                "actual_token_at_divergence": (
                    actual_ids[common_prefix]
                    if common_prefix < len(actual_ids)
                    else None
                ),
                "differing_positions": sum(
                    left != right
                    for left, right in zip(expected_ids, actual_ids, strict=False)
                )
                + abs(len(expected_ids) - len(actual_ids)),
            }
        )
    return mismatches


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
            "vLLM metrics response is missing required gauges: " + ", ".join(missing)
        )
    return {
        name: (max(found) if name.endswith("kv_cache_usage_perc") else sum(found))
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
    default_max_tokens: int,
    timeout: float,
    after_events: int,
) -> int:
    max_tokens = int(fixture.get("max_tokens", default_max_tokens))
    if after_events >= max_tokens:
        raise ValueError(
            "cancellation checkpoint must leave tokens remaining in the request"
        )
    payload = {
        "model": model,
        "prompt": fixture["prompt"],
        "max_tokens": max_tokens,
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


def concurrent_fixture_waves(
    fixtures: list[dict[str, Any]], concurrency: int
) -> list[list[dict[str, Any]]]:
    """Return full-width B4 waves without making B1 repeat every fixture again."""
    selected = fixtures if concurrency > 1 else fixtures[:1]
    return [
        selected[start : start + concurrency]
        for start in range(0, len(selected), concurrency)
    ]


def run_concurrent_wave(
    base_url: str,
    model: str,
    fixtures: list[dict[str, Any]],
    default_max_tokens: int,
    concurrency: int,
    timeout: float,
    metrics_poll_interval: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one wave and sample metrics until the required overlap is observed."""
    required_running = min(concurrency, len(fixtures))
    release = threading.Event()

    def after_release(fixture: dict[str, Any]) -> dict[str, Any]:
        release.wait()
        return completion(
            base_url,
            model,
            fixture,
            default_max_tokens,
            timeout,
        )

    started = time.monotonic()
    peak_running = 0.0
    peak_metrics: dict[str, float] | None = None
    samples = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fixtures)) as executor:
        futures = [executor.submit(after_release, fixture) for fixture in fixtures]
        release.set()
        deadline = started + timeout
        while not all(future.done() for future in futures):
            if time.monotonic() >= deadline:
                break
            metrics = parse_metrics(
                http_request(
                    f"{base_url}/metrics",
                    timeout=min(timeout, 10.0),
                )
            )
            samples += 1
            running = metrics["vllm:num_requests_running"]
            if peak_metrics is None or running > peak_running:
                peak_running = running
                peak_metrics = metrics
            if peak_running >= required_running:
                break
            time.sleep(metrics_poll_interval)
        results = [future.result() for future in futures]

    return results, {
        "fixture_ids": [fixture["id"] for fixture in fixtures],
        "required_running": required_running,
        "peak_running": peak_running,
        "peak_metrics": peak_metrics,
        "metrics_samples": samples,
        "required_overlap_observed": peak_running >= required_running,
        "elapsed_seconds": time.monotonic() - started,
    }


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
    progress: dict[str, Any] = {
        "schema_version": 2,
        "status": "running",
        "phase": "isolated_first",
        "created_unix_seconds": time.time(),
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "metrics_before": before,
        "isolated_first": [],
        "isolated_replay": [],
        "concurrent": [],
        "concurrent_waves": [],
        "cancellation": None,
    }
    write_json_atomic(args.output, progress)

    first: list[dict[str, Any]] = []
    for fixture in fixtures:
        first.append(
            completion(
                args.base_url,
                args.model,
                fixture,
                args.max_tokens,
                args.timeout,
            )
        )
        progress["isolated_first"] = first
        write_json_atomic(args.output, progress)

    progress["phase"] = "isolated_replay"
    write_json_atomic(args.output, progress)
    second: list[dict[str, Any]] = []
    for fixture in fixtures:
        second.append(
            completion(
                args.base_url,
                args.model,
                fixture,
                args.max_tokens,
                args.timeout,
            )
        )
        progress["isolated_replay"] = second
        write_json_atomic(args.output, progress)

    mismatches = replay_mismatches(first, second)
    if mismatches:
        progress.update(
            status="failed",
            phase="same_process_replay_comparison",
            replay_mismatches=mismatches,
        )
        write_json_atomic(args.output, progress)
        raise AssertionError("same-process replay produced different token IDs")

    progress.update(
        phase="concurrent_isolation",
        same_process_replay_identical=True,
    )
    write_json_atomic(args.output, progress)

    batch: list[dict[str, Any]] = []
    waves: list[dict[str, Any]] = []
    for wave_fixtures in concurrent_fixture_waves(fixtures, args.concurrency):
        wave_results, wave_evidence = run_concurrent_wave(
            args.base_url,
            args.model,
            wave_fixtures,
            args.max_tokens,
            args.concurrency,
            args.timeout,
            args.metrics_poll_interval,
        )
        batch.extend(wave_results)
        waves.append(wave_evidence)
        progress.update(concurrent=batch, concurrent_waves=waves)
        write_json_atomic(args.output, progress)

    expected = {item["id"]: item["token_ids"] for item in first}
    if any(item["token_ids"] != expected[item["id"]] for item in batch):
        progress.update(status="failed", phase="concurrent_isolation_comparison")
        write_json_atomic(args.output, progress)
        raise AssertionError("concurrent output diverged from isolated output")
    missed_overlap = [wave for wave in waves if not wave["required_overlap_observed"]]
    if missed_overlap:
        progress.update(
            status="failed",
            phase="concurrent_overlap",
            missed_overlap=missed_overlap,
        )
        write_json_atomic(args.output, progress)
        raise AssertionError("concurrent requests did not reach required overlap")

    progress.update(
        phase="cancellation",
        concurrent_isolation_identical=True,
        concurrent_overlap_observed=True,
    )
    write_json_atomic(args.output, progress)

    cancellation: dict[str, Any] | None = None
    if args.cancel_after_events:
        events = cancel_stream(
            args.base_url,
            args.model,
            fixtures[0],
            args.max_tokens,
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
            progress.update(status="failed", phase="cancellation_replacement")
            write_json_atomic(args.output, progress)
            raise AssertionError("replacement after cancellation was not isolated")
        cancellation = {
            "requested_max_tokens": int(fixtures[0].get("max_tokens", args.max_tokens)),
            "stream_events_before_close": events,
            "replacement": replacement,
        }
        progress["cancellation"] = cancellation
        write_json_atomic(args.output, progress)

    after = wait_for_idle(args.base_url, args.idle_timeout)
    provenance = (
        json.loads(args.provenance.read_text(encoding="utf-8"))
        if args.provenance
        else {}
    )
    return {
        "schema_version": 2,
        "status": "passed",
        "phase": "complete",
        "created_unix_seconds": time.time(),
        "base_url": args.base_url,
        "model": args.model,
        "concurrency": args.concurrency,
        "metrics_before": before,
        "metrics_after": after,
        "same_process_replay_identical": True,
        "concurrent_isolation_identical": True,
        "concurrent_overlap_observed": True,
        "isolated_first": first,
        "isolated_replay": second,
        "concurrent": batch,
        "concurrent_waves": waves,
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
    parser.add_argument("--cancel-after-events", type=int, default=257)
    parser.add_argument("--metrics-poll-interval", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--idle-timeout", type=float, default=120)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.max_tokens < 2048:
        parser.error("--max-tokens must be at least 2048")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if args.cancel_after_events < 0:
        parser.error("--cancel-after-events must be non-negative")
    if args.cancel_after_events >= args.max_tokens:
        parser.error("--cancel-after-events must be less than --max-tokens")
    if args.metrics_poll_interval <= 0:
        parser.error("--metrics-poll-interval must be positive")
    output = args.output.resolve()
    if not args.allow_tmp and output.is_relative_to(Path("/tmp")):
        parser.error("--output must be durable (outside /tmp)")
    return args


def main() -> None:
    args = parse_args()
    try:
        document = run(args)
    except Exception as error:
        if args.output.exists():
            document = json.loads(args.output.read_text(encoding="utf-8"))
            document.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                failed_unix_seconds=time.time(),
            )
            write_json_atomic(args.output, document)
        raise
    write_json_atomic(args.output, document)
    rendered = json.dumps(document, indent=2) + "\n"
    print(rendered, end="")


if __name__ == "__main__":
    main()
