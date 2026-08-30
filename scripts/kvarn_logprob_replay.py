#!/usr/bin/env python3
"""Compare greedy completion logprobs across two same-process passes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

if __package__:
    from .kvarn_service_gate import http_request, write_json_atomic
else:
    from kvarn_service_gate import http_request, write_json_atomic


def load_fixtures(
    path: Path,
    default_max_tokens: int,
    fixture_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load and validate completion fixtures, optionally selecting IDs."""
    fixtures = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError("fixtures must be a non-empty JSON list")

    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fixture in fixtures:
        if (
            not isinstance(fixture, dict)
            or not isinstance(fixture.get("id"), str)
            or not isinstance(fixture.get("prompt"), str)
        ):
            raise TypeError("each fixture requires string id and prompt fields")
        fixture_id = fixture["id"]
        if fixture_id in seen:
            raise ValueError(f"duplicate fixture id: {fixture_id}")
        seen.add(fixture_id)
        max_tokens = int(fixture.get("max_tokens", default_max_tokens))
        if max_tokens < 1:
            raise ValueError(f"{fixture_id} must request at least one output token")
        validated.append(
            {
                "id": fixture_id,
                "prompt": fixture["prompt"],
                "max_tokens": max_tokens,
            }
        )

    if fixture_ids:
        requested = set(fixture_ids)
        missing = requested - seen
        if missing:
            raise ValueError("unknown fixture ids: " + ", ".join(sorted(missing)))
        validated = [item for item in validated if item["id"] in requested]
    return validated


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must contain numeric values")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise AssertionError(f"{field} must contain finite values")
    return normalized


def _normalize_logprobs(
    choice: dict[str, Any], expected_tokens: int
) -> tuple[
    list[str],
    list[float | None],
    list[dict[str, float] | None],
    list[int],
]:
    raw = choice.get("logprobs")
    if not isinstance(raw, dict):
        raise TypeError("completion did not return a logprobs object")

    tokens = raw.get("tokens")
    token_logprobs = raw.get("token_logprobs")
    top_logprobs = raw.get("top_logprobs")
    text_offsets = raw.get("text_offset")
    fields = (tokens, token_logprobs, top_logprobs, text_offsets)
    if not all(isinstance(field, list) for field in fields):
        raise AssertionError("completion returned malformed logprob arrays")
    if not all(len(field) == expected_tokens for field in fields):
        raise AssertionError("completion logprob arrays do not match token_ids")
    if not all(isinstance(token, str) for token in tokens):
        raise AssertionError("completion logprob tokens must be strings")
    if not all(isinstance(offset, int) for offset in text_offsets):
        raise AssertionError("completion text offsets must be integers")

    selected = [
        None if value is None else _finite_float(value, "token_logprobs")
        for value in token_logprobs
    ]
    alternatives: list[dict[str, float] | None] = []
    for entry in top_logprobs:
        if entry is None:
            alternatives.append(None)
            continue
        if not isinstance(entry, dict) or not all(
            isinstance(token, str) for token in entry
        ):
            raise AssertionError("completion top_logprobs must be token-score objects")
        alternatives.append(
            {
                token: _finite_float(score, "top_logprobs")
                for token, score in entry.items()
            }
        )
    return tokens, selected, alternatives, text_offsets


def completion_with_logprobs(
    base_url: str,
    model: str,
    fixture: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    """Request one greedy completion and retain its token-level evidence."""
    max_tokens = int(fixture["max_tokens"])
    payload = {
        "model": model,
        "prompt": fixture["prompt"],
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "echo": False,
        "return_token_ids": True,
        "return_tokens_as_token_ids": True,
        "logprobs": 5,
    }
    started = time.monotonic()
    response = json.loads(
        http_request(f"{base_url}/v1/completions", payload, timeout=timeout)
    )
    elapsed = time.monotonic() - started
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise AssertionError("completion did not return a choice object")
    choice = choices[0]
    token_ids = choice.get("token_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in token_ids
    ):
        raise AssertionError("completion did not return integer token_ids")
    if len(token_ids) != max_tokens:
        raise AssertionError(
            f"{fixture['id']} returned {len(token_ids)} of {max_tokens} tokens"
        )

    prompt_token_ids = choice.get("prompt_token_ids")
    if not isinstance(prompt_token_ids, list) or not all(
        isinstance(token_id, int) and not isinstance(token_id, bool)
        for token_id in prompt_token_ids
    ):
        raise AssertionError("completion did not return integer prompt_token_ids")

    tokens, selected, alternatives, text_offsets = _normalize_logprobs(
        choice, len(token_ids)
    )
    prompt_ids_rendered = json.dumps(prompt_token_ids, separators=(",", ":")).encode()
    token_ids_rendered = json.dumps(token_ids, separators=(",", ":")).encode()
    logprobs_rendered = json.dumps(
        [selected, alternatives], separators=(",", ":"), sort_keys=True
    ).encode()
    response_metadata = {
        key: response[key]
        for key in (
            "id",
            "object",
            "created",
            "model",
            "system_fingerprint",
            "usage",
        )
        if key in response
    }
    return {
        "id": fixture["id"],
        "prompt": fixture["prompt"],
        "max_tokens": max_tokens,
        "elapsed_seconds": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "text": choice.get("text", ""),
        "stop_reason": choice.get("stop_reason"),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_ids_sha256": hashlib.sha256(prompt_ids_rendered).hexdigest(),
        "token_ids": token_ids,
        "token_ids_sha256": hashlib.sha256(token_ids_rendered).hexdigest(),
        "tokens": tokens,
        "token_logprobs": selected,
        "top_logprobs": alternatives,
        "text_offsets": text_offsets,
        "logprobs_sha256": hashlib.sha256(logprobs_rendered).hexdigest(),
        "response_metadata": response_metadata,
    }


def _first_sequence_divergence(
    expected: list[Any], actual: list[Any]
) -> dict[str, Any] | None:
    for index in range(max(len(expected), len(actual))):
        expected_value = expected[index] if index < len(expected) else None
        actual_value = actual[index] if index < len(actual) else None
        if expected_value != actual_value:
            return {
                "index": index,
                "expected": expected_value,
                "actual": actual_value,
            }
    return None


def compare_fixture_replay(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, Any]:
    """Locate token and floating-point replay divergence independently."""
    if expected["id"] != actual["id"]:
        raise ValueError(f"fixture order mismatch: {expected['id']} != {actual['id']}")
    token_id = _first_sequence_divergence(expected["token_ids"], actual["token_ids"])
    selected = _first_sequence_divergence(
        expected["token_logprobs"], actual["token_logprobs"]
    )
    alternatives = _first_sequence_divergence(
        expected["top_logprobs"], actual["top_logprobs"]
    )
    logprob_indices = [
        divergence["index"]
        for divergence in (selected, alternatives)
        if divergence is not None
    ]
    first_logprob_index = min(logprob_indices, default=None)
    any_indices = [
        divergence["index"]
        for divergence in (token_id, selected, alternatives)
        if divergence is not None
    ]
    first_any_index = min(any_indices, default=None)
    token_id_index = token_id["index"] if token_id is not None else None
    return {
        "id": expected["id"],
        "identical": first_any_index is None,
        "first_divergence_index": first_any_index,
        "first_logprob_divergence_index": first_logprob_index,
        "logprob_diverges_before_token_ids": first_logprob_index is not None
        and (token_id_index is None or first_logprob_index < token_id_index),
        "first_token_id_divergence": token_id,
        "first_selected_logprob_divergence": selected,
        "first_top_logprobs_divergence": alternatives,
        "expected_token_ids_sha256": expected["token_ids_sha256"],
        "actual_token_ids_sha256": actual["token_ids_sha256"],
        "expected_logprobs_sha256": expected["logprobs_sha256"],
        "actual_logprobs_sha256": actual["logprobs_sha256"],
    }


def compare_replays(
    first: list[dict[str, Any]], replay: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        compare_fixture_replay(expected, actual)
        for expected, actual in zip(first, replay, strict=True)
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    fixtures = load_fixtures(args.fixtures, args.max_tokens, args.fixture_id)
    progress: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "phase": "first",
        "created_unix_seconds": time.time(),
        "base_url": args.base_url,
        "model": args.model,
        "request_parameters": {
            "temperature": 0,
            "ignore_eos": True,
            "echo": False,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": 5,
        },
        "first": [],
        "replay": [],
        "comparisons": [],
    }
    write_json_atomic(args.output, progress)

    first: list[dict[str, Any]] = []
    for fixture in fixtures:
        first.append(
            completion_with_logprobs(args.base_url, args.model, fixture, args.timeout)
        )
        progress["first"] = first
        write_json_atomic(args.output, progress)

    progress["phase"] = "replay"
    write_json_atomic(args.output, progress)
    replay: list[dict[str, Any]] = []
    for fixture in fixtures:
        replay.append(
            completion_with_logprobs(args.base_url, args.model, fixture, args.timeout)
        )
        progress["replay"] = replay
        write_json_atomic(args.output, progress)

    comparisons = compare_replays(first, replay)
    diverged = [comparison for comparison in comparisons if not comparison["identical"]]
    progress.update(
        status="diverged" if diverged else "identical",
        phase="complete",
        completed_unix_seconds=time.time(),
        comparisons=comparisons,
        divergences=diverged,
    )
    write_json_atomic(args.output, progress)
    return progress


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--fixture-id", action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
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
    print(json.dumps(document, indent=2) + "\n", end="")


if __name__ == "__main__":
    main()
