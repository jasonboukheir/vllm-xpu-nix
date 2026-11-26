"""Check Inkling tool-result continuation and stream abort while four B70 slots are active."""

import argparse
import hashlib
import http.client
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from offload_replay import Trial, diff, parse_metrics

TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_ticket",
        "description": "Look up a shipping ticket by its exact ticket ID.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
}
EXPECTED = {"owner": "Tala", "tracking": "COPPER-581"}
BUDGET = 512
KEYS = ("TEAK246", "ALDER738", "BEECH591", "FIR824")


def checkpoint(trial, name):
    raw = trial.request("/metrics", text=True)
    if name:
        (trial.output / f"{name}.prom").write_text(raw)
    metrics, _ = parse_metrics(raw)
    for reason in ("abort", "error", "length", "stop"):
        values = re.findall(
            rf'^vllm:request_success_total\{{[^}}]*finished_reason="{reason}"[^}}]*\}} (\S+)',
            raw,
            re.MULTILINE,
        )
        metrics[reason] = sum(map(float, values)) if values else None
    return metrics


def parsed_json(text):
    try:
        return json.loads(text or "")
    except ValueError:
        return None


def tool_result(trial):
    payload = {
        "model": trial.args.model,
        "tools": [TOOL],
        "messages": [
            {
                "role": "user",
                "content": "Look up shipping ticket P-17 with the provided tool. After the result arrives, return only a JSON object with its owner and tracking fields.",
            }
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 1024,
        "reasoning_effort": "low",
        "tool_choice": "auto",
    }
    first = trial.request("/v1/chat/completions", payload)
    (trial.output / "tool-request-1.json").write_text(json.dumps(payload, indent=2))
    (trial.output / "tool-response-1.json").write_text(json.dumps(first, indent=2))
    choice = first["choices"][0]
    returned = choice["message"]
    calls = returned.get("tool_calls") or []
    valid = (
        choice["finish_reason"] == "tool_calls"
        and len(calls) == 1
        and bool(calls[0].get("id"))
        and calls[0].get("type") == "function"
        and calls[0]["function"]["name"] == TOOL["function"]["name"]
        and parsed_json(calls[0]["function"]["arguments"]) == {"id": "P-17"}
    )
    trial.check("requested-tool-and-arguments", valid)
    if not valid:
        return
    assistant_message = {
        k: returned[k]
        for k in ("role", "content", "reasoning", "tool_calls")
        if k in returned
    }
    payload["messages"].extend(
        [
            assistant_message,
            {
                "role": "tool",
                "tool_call_id": calls[0]["id"],
                "content": json.dumps({"id": "P-17", **EXPECTED}),
            },
        ]
    )
    second = trial.request("/v1/chat/completions", payload)
    (trial.output / "tool-request-2.json").write_text(json.dumps(payload, indent=2))
    (trial.output / "tool-response-2.json").write_text(json.dumps(second, indent=2))
    final = second["choices"][0]
    trial.check("tool-result-final-stop", final["finish_reason"] == "stop")
    trial.check(
        "tool-result-returned-fields",
        parsed_json(final["message"].get("content")) == EXPECTED,
    )
    trial.check(
        "tool-result-no-extra-invocation", not final["message"].get("tool_calls")
    )


def consume(trial, fixture, i, ready, cancel, start_barrier):
    """Read until EOS/length or close the victim's socket after observed generation."""
    connection = http.client.HTTPConnection("127.0.0.1", trial.args.port, timeout=180)
    payload = {
        "model": trial.args.model,
        "prompt": fixture["prompt_ids"],
        "max_tokens": BUDGET,
        "ignore_eos": True,
        "min_tokens": BUDGET,
        "temperature": 0,
        "seed": 42,
        "return_token_ids": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start_barrier.wait(timeout=15)
    started = time.monotonic()
    ids = []
    words = []
    reason = None
    done = False
    closed = False
    usage = None
    response_id = None
    try:
        connection.request(
            "POST",
            "/v1/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Connection": "close"},
        )
        with (
            connection.getresponse() as response,
            (trial.output / f"stream-{i}.sse").open("wb") as raw,
        ):
            if response.status != 200:
                raise ValueError(f"HTTP {response.status}: {response.read(2000)!r}")
            for line in response:
                raw.write(line)
                if not line.startswith(b"data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == b"[DONE]":
                    done = True
                    break
                msg = json.loads(chunk)
                if msg.get("error"):
                    raise RuntimeError(msg["error"])
                response_id = msg.get("id") or response_id
                usage = msg.get("usage") or usage
                for choice in msg["choices"]:
                    ids.extend(choice.get("token_ids") or [])
                    words.append(choice.get("text") or "")
                    reason = choice.get("finish_reason") or reason
                if len(ids) >= 4:
                    ready.set()
                if i == 0 and cancel.is_set() and len(ids) >= 32 and reason is None:
                    closed = True
                    break
    finally:
        connection.close()
    return {
        "request_id": response_id,
        "started": started,
        "ended": time.monotonic(),
        "count": len(ids),
        "leading_output": "".join(words)[:240],
        "client_closed": closed,
        "finish_reason": reason,
        "stream_done": done,
        "usage": usage,
    }


def abort_and_reuse(trial):
    trial.reset(True)
    replacement = trial.fixture(1024, "ASH275", varied=True)
    reference = trial.case("replacement-reference", replacement, "cold")
    fixtures = [
        trial.fixture(1024 + 128 * i, code, varied=True) for i, code in enumerate(KEYS)
    ]
    (trial.output / "wave-inputs.json").write_text(json.dumps([*fixtures, replacement]))
    before = checkpoint(trial, "abort-before")
    barrier = threading.Barrier(5)
    ready = [threading.Event() for _ in fixtures]
    cancel = threading.Event()
    peak = 0
    with ThreadPoolExecutor(4) as pool:
        jobs = [
            pool.submit(consume, trial, f, i, ready[i], cancel, barrier)
            for i, f in enumerate(fixtures)
        ]
        barrier.wait(timeout=15)
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            live = checkpoint(trial, None)
            peak = max(peak, live["running"] or 0)
            if peak >= 4 and all(e.is_set() for e in ready):
                break
            if any(job.done() for job in jobs):
                break
            time.sleep(0.04)
        trial.check(
            "four-live-and-each-generating",
            peak >= 4 and all(e.is_set() for e in ready),
        )
        trial.doc["scheduler_running"] = [
            {
                "phase": "before-disconnect",
                "running": live["running"],
                "at": time.monotonic(),
            }
        ]
        cancel.set()
        victim = jobs[0].result(timeout=45)
        trial.doc["victim"] = victim
        trial.check(
            "closed-active-stream",
            victim["client_closed"]
            and 32 <= victim["count"] < BUDGET
            and not victim["stream_done"]
            and victim["finish_reason"] is None,
        )
        # The remaining three requests hold their slots while this warm request
        # is admitted. Confirm their overlap using both the scheduler and HTTP.
        # A disconnected async HTTP generator drops its output-processor state
        # before the ordinary finish-reason metrics are recorded. Observe the
        # actual scheduler slot while all three remaining streams are active.
        admissions = trial.doc["scheduler_running"]

        def observe(phase):
            metrics = checkpoint(trial, None)
            count = metrics["running"]
            if not admissions or (
                admissions[-1]["phase"],
                admissions[-1]["running"],
            ) != (phase, count):
                admissions.append(
                    {"phase": phase, "running": count, "at": time.monotonic()}
                )
            return count

        deadline = time.monotonic() + 3
        while observe("after-disconnect") != 3 and time.monotonic() < deadline:
            if any(peer.done() for peer in jobs[1:]):
                break
            time.sleep(0.01)
        trial.check(
            "scheduler-released-one-slot",
            admissions[-1]["running"] == 3
            and all(not peer.done() for peer in jobs[1:]),
        )
        before_reuse = checkpoint(trial, "replacement-before")
        reuse_started = time.monotonic()
        reuse = pool.submit(trial.complete, replacement)
        while not reuse.done():
            observe("replacement")
            time.sleep(0.005)
        actual = reuse.result()
        reuse_ended = time.monotonic()
        trial.check(
            "scheduler-reoccupied-fourth-slot",
            any(
                item["phase"] == "replacement" and item["running"] == 4
                for item in admissions
            ),
        )
        trial.evidence("replacement", replacement, actual)
        trial.same("replacement-cold-exact", reference, actual)
        trial.doc["replacement"] = {
            "started": reuse_started,
            "ended": reuse_ended,
            "result": actual,
        }
        survivors = [job.result(timeout=180) for job in jobs[1:]]
    trial.doc["survivors"] = survivors
    trial.doc["peak_running"] = peak
    request_ids = [r.get("request_id") for r in [victim, *survivors, actual]]
    trial.check(
        "five-distinct-request-ids", all(request_ids) and len(set(request_ids)) == 5
    )
    for i, survivor in enumerate(survivors, 1):
        trial.check(
            f"survivor-{i}-overlapped-replacement",
            survivor["started"] <= reuse_started < reuse_ended <= survivor["ended"],
        )
        trial.check(
            f"survivor-{i}-identity",
            survivor["leading_output"].lstrip().startswith(KEYS[i]),
        )
        trial.check(
            f"survivor-{i}-finished",
            survivor["finish_reason"] == "length"
            and survivor["stream_done"]
            and survivor["count"] == BUDGET
            and (survivor["usage"] or {}).get("completion_tokens") == BUDGET,
        )
    trial.check(
        "victim-had-correct-identity",
        victim["leading_output"].lstrip().startswith(KEYS[0]),
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        after = checkpoint(trial, None)
        if (
            after["running"] == after["waiting"] == 0
            and (diff(after, before_reuse, "host_hit_tokens") or 0) > 0
        ):
            break
        time.sleep(0.1)
    after = checkpoint(trial, "abort-after")
    trial.doc["counter_delta"] = {k: diff(after, before, k) for k in before}
    trial.doc["replacement_delta"] = {
        k: diff(after, before_reuse, k)
        for k in ("load_bytes", "host_hit_tokens", "gpu_hit_tokens")
    }
    trial.check(
        "replacement-from-ram",
        (diff(after, before_reuse, "load_bytes") or 0) > 0
        and (diff(after, before_reuse, "host_hit_tokens") or 0)
        >= len(replacement["prompt_ids"]) - 128,
    )
    trial.check("idle-after-abort-and-reuse", after["running"] == after["waiting"] == 0)
    # The only completed requests in this interval must be the three forced-
    # length peers and one naturally finished replacement. Keep the raw abort
    # counter as evidence even when this frontend does not increment it.
    trial.check(
        "only-three-peer-finishes-and-one-replacement",
        diff(after, before, "length") == 3 and diff(after, before, "stop") == 1,
    )
    trial.check("no-engine-errors", diff(after, before, "error") == 0)
    trial.check("DSpark-active", (diff(after, before, "accepted") or 0) > 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18001)
    parser.add_argument("--model", default="inkling-trial")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.depth = None
    trial = Trial(args)
    trial.doc["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (trial.output / "tool-abort-client.py").write_bytes(Path(__file__).read_bytes())
    try:
        trial.verify_model()
        if checkpoint(trial, None)["running"] != 0:
            raise ValueError("Other requests are already using this test server")
        tool_result(trial)
        abort_and_reuse(trial)
        trial.doc["status"] = (
            "passed" if all(trial.doc["checks"].values()) else "failed"
        )
    except Exception as exc:
        trial.doc.update(status="error", error=repr(exc))
        raise
    finally:
        trial.save()
    print(trial.doc["status"], [k for k, v in trial.doc["checks"].items() if not v])
    return int(trial.doc["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
