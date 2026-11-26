#!/usr/bin/env python3
"""Check four independent in-flight conversations on a local Inkling trial."""

import argparse
import concurrent.futures
import json
import re
import threading
import time
import urllib.request
from pathlib import Path

KEYS = ["QUARTZ-741", "EMBER-382", "COBALT-614", "PINE-957"]


def query(url, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read().decode()


def messages(index, rows):
    # Distinct prefixes force independent physical histories. Different prompt
    # lengths also allow requests to leave the batch in a different order.
    key = KEYS[index]
    data = [
        f"Desk {index}, item {i:04d}, ordinary receipt S{(i * 113 + 17) % 65537}."
        for i in range(rows + 13 * index)
    ]
    data.insert(rows // 5, f"The approved archive code for this desk is {key}.")
    data.append("What is the approved archive code? Respond with only a JSON string.")
    return [{"role": "user", "content": "\n".join(data)}]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("model")
    cli.add_argument("output", type=Path)
    cli.add_argument("--url", default="http://127.0.0.1:18001")
    cli.add_argument("--rows", type=int, default=160)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    ready = threading.Barrier(len(KEYS) + 1)

    def request(index):
        body = {
            "model": args.model,
            "messages": messages(index, args.rows),
            "temperature": 0,
            "seed": 42,
            "max_tokens": 1024,
            "reasoning_effort": "low",
        }
        ready.wait()
        started = time.monotonic()
        response = json.loads(query(args.url + "/v1/chat/completions", body))
        elapsed = time.monotonic() - started
        (args.output / f"desk-{index}.json").write_text(json.dumps(response, indent=2))
        choice = response["choices"][0]
        answer = choice["message"].get("content") or ""
        try:
            correct = json.loads(answer) == KEYS[index]
        except ValueError:
            correct = False
        return {
            "desk": index,
            "answer": answer,
            "correct": correct,
            "stop": choice["finish_reason"],
            "seconds": elapsed,
            "usage": response["usage"],
        }

    peaks = []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(KEYS)) as pool:
        jobs = [pool.submit(request, index) for index in range(len(KEYS))]
        started = time.monotonic()
        ready.wait()
        while any(not job.done() for job in jobs):
            metrics = query(args.url + "/metrics")
            gauges = re.findall(
                r"^vllm:num_requests_running(?:\{[^}]*\})? ([.\d]+)",
                metrics,
                re.MULTILINE,
            )
            peaks.append(sum(map(float, gauges)))
            time.sleep(0.1)
        elapsed = time.monotonic() - started
        for index, job in enumerate(jobs):
            try:
                results.append(job.result())
            except (ValueError, OSError, KeyError) as exc:
                results.append({"desk": index, "correct": False, "error": str(exc)})
    total = sum(row.get("usage", {}).get("completion_tokens", 0) for row in results)
    receipt = {
        "rows": args.rows,
        "results": results,
        "peak_running": max(peaks, default=0),
        "wave_seconds": elapsed,
        "aggregate_output_tok_s": total / elapsed,
    }
    (args.output / "summary.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    return int(
        receipt["peak_running"] < 4
        or not all(row["correct"] and row["stop"] == "stop" for row in results)
    )


if __name__ == "__main__":
    raise SystemExit(main())
