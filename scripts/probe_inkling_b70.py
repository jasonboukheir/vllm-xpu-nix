#!/usr/bin/env python3
"""Run the same synthetic single-client chat workloads on either local model."""

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from kvarn_vision_run import post_json, request_case

PROMPT = (
    "Explain to a junior engineer how to test a cache with time-based expiry. "
    "Give three test cases, including equality at expiry and an entry rewritten "
    "before expiry. Explain how a fake clock keeps each test deterministic."
)
METRICS = (
    "vllm:spec_decode_num_drafts_total",
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:generation_tokens_total",
)


def get(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def counters(metrics):
    counts = {}
    for name in METRICS:
        values = re.findall(
            rf"^{name}(?:\{{[^\n]*?\}})? ([\d.eE+-]+)", metrics, re.MULTILINE
        )
        counts[name] = sum(map(float, values)) if values else None
    return counts


def grades(content, expected):
    if content.strip().startswith("```"):
        content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content)
    try:
        return json.loads(content) == expected
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("output", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:18001")
    parser.add_argument("--expect-draft", action="store_true")
    parser.add_argument("--only-long", type=int, metavar="MIN_PROMPT_TOKENS")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    ids = [entry["id"] for entry in json.loads(get(args.url + "/v1/models"))["data"]]
    if args.model not in ids:
        raise ValueError(f"Expected {args.model}, server lists {ids}")
    before = get(args.url + "/metrics")
    (out / "metrics-before.txt").write_text(before)
    examples = [
        (
            "expiry",
            (
                "A cache stores (value, expires_at). Reads hit only when now < expires_at. "
                "At t=3 set k=A with TTL 7. At t=8 overwrite k=B with TTL 5. "
                "Return only a JSON array of the values read at t=9, t=10, t=13, t=14. Use null for a miss."
            ),
            ["B", "B", None, None],
        ),
        (
            "ledger",
            (
                "Use only this ledger: invoice P-17 has owner Bea and 3 cartons of 7 coils. "
                "Invoice H-04 has owner Ari and 11 bolts. Return only a JSON object for P-17 with "
                "keys owner, total_coils, tracking_code. Use null for unrecorded fields."
            ),
            {"owner": "Bea", "total_coils": 21, "tracking_code": None},
        ),
        ("chat-first", PROMPT, None),
        ("chat-repeat", PROMPT, None),
    ]
    if args.only_long:
        # Calibrate with the active tokenizer and count chat-template overhead.
        needle = "The archived catalogue has two authoritative updates. First tag is MANGO-793.\n"
        tail = "\nFinal tag is IVORY-284. Return only a JSON array with the first and final tags."
        unit = " Archive contains unchanged entries."
        n = max(1, args.only_long // 4)
        for _ in range(6):
            prompt = needle + unit * n + tail
            tokenized = post_json(
                args.url,
                "/tokenize",
                {
                    "model": args.model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            count = tokenized["count"]
            if args.only_long <= count <= args.only_long + 32:
                break
            n = max(1, round(n * (args.only_long + 8) / count))
        if not args.only_long <= count <= args.only_long + 32:
            raise ValueError(f"Calibration failed, got {count}")
        examples = [("long-retrieval", prompt, ["MANGO-793", "IVORY-284"])]
    results = []
    for label, prompt, expected in examples:
        generation = {
            "model": args.model,
            "chat_template_kwargs": {},
            "max_tokens": 1024,
        }
        result = request_case(
            args.url,
            {
                "id": label,
                "messages": [{"role": "user", "content": prompt}],
                "generation": generation,
                "expected_terms": [],
            },
            out,
        )
        row = {
            k: result[k]
            for k in (
                "id",
                "usage",
                "ttft_seconds",
                "total_seconds",
                "finish_reason",
                "decode_tokens_per_second",
            )
        }
        row["answer_chars"] = len(result["content"])
        row["grade"] = (
            grades(result["content"], expected) if expected is not None else None
        )
        results.append(row)
    if not args.only_long:
        tool = {
            "type": "function",
            "function": {
                "name": "lookup_ticket",
                "description": "Look up a ticket by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            },
        }
        body = {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": "Please look up ticket P-17 with the provided tool.",
                }
            ],
            "tools": [tool],
            "tool_choice": "auto",
            "temperature": 0,
            "seed": 42,
            "max_tokens": 1024,
        }
        reply = post_json(args.url, "/v1/chat/completions", body)
        (out / "tool-response.json").write_text(json.dumps(reply, indent=2))
        calls = reply["choices"][0]["message"].get("tool_calls") or []
        valid = len(calls) == 1 and calls[0]["function"]["name"] == "lookup_ticket"
        valid = valid and grades(calls[0]["function"]["arguments"], {"id": "P-17"})
        results.append({"id": "tool", "grade": valid})
    start = counters(before)
    for _ in range(16):
        after = get(args.url + "/metrics")
        end = counters(after)
        total = end[METRICS[-1]] or 0
        generated = sum(
            row.get("usage", {}).get("completion_tokens", 0) for row in results
        )
        if total >= (start[METRICS[-1]] or 0) + generated:
            break
        time.sleep(1)
    (out / "metrics-after.txt").write_text(after)
    delta = {
        name: None if val is None else val - (start[name] or 0)
        for name, val in end.items()
    }
    draft_active = all((delta[key] or 0) > 0 for key in METRICS[:3])
    summary = {
        "model": args.model,
        "results": results,
        "counter_delta": delta,
        "expect_draft": args.expect_draft,
        "draft_active": draft_active,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    failed = any(row["grade"] is False for row in results)
    return int(failed or draft_active != args.expect_draft)


if __name__ == "__main__":
    raise SystemExit(main())
