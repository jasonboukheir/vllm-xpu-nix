"""Compare completed answers on the same six technical prompts via local OpenAI-compatible APIs."""

import argparse
import hashlib
import http.client
import ipaddress
import json
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from itertools import pairwise, permutations
from pathlib import Path
from urllib.parse import urlsplit

CASES = [
    (
        "python_aliases",
        """In Python 3.12 the following code runs to completion.

rows = [[2], [3], [5]]
shadow = rows[:]
for i, row in enumerate(rows):
    row.append(sum(item[0] for item in shadow[:i + 1]))
shadow[1] = [11]
rows[2].extend(shadow[0][-1:])
result = [sum(row) for row in rows] + [sum(row) for row in shadow]

Return just the JSON array represented by result.""",
        [4, 8, 17, 4, 11, 17],
    ),
    (
        "cache_capacity",
        """A service has a 436 MiB device budget. It must reserve 40 MiB for scratch, 92 MiB for fixed state, and exactly 87 physical sliding-window blocks in each of 3 layers. Each sliding-window block costs 128 KiB per layer. The rest can buy only whole full-attention blocks. One full-attention block allocates 96 KiB in each of 16 layers and covers 128 tokens per request. Exactly one physical full-attention block is reserved as a null sentinel. Nothing else is allocated. MiB and KiB are binary units.

An incoming wave contains an unlimited supply of unrelated 4,097-token requests. Each request occupies its own integer number of full-attention blocks with no sharing. Find (1) the number of physical full-attention blocks including the sentinel, (2) total usable token capacity before admissions, (3) maximum simultaneous complete requests, and (4) usable full-attention blocks still free after admitting that many requests. Return just one JSON array containing those four integers in order.""",
        [180, 22912, 5, 14],
    ),
    (
        "constrained_schedule",
        """A maintenance job starts at t=0. There is one CPU and one GPU. Each can run at most one task at a time, and they can operate simultaneously. Tasks run without interruption, use only their assigned resource, and start at nonnegative integer times. A successor can start at exactly its predecessor's finish time. The entire job finishes when all tasks finish.

Task  Resource  Duration  Must finish before this task starts
A     CPU       4         none
B     GPU       3         A
C     GPU       5         none
D     CPU       2         C
E     GPU       4         D
F     CPU       3         B and E

Minimize the finish time. Among schedules with that minimum, select the lexicographically earliest sequence of start times in alphabetical task order A,B,C,D,E,F. Return only a JSON object with keys "finish" (integer) and "starts" (that six-integer array).""",
        {"finish": 15, "starts": [0, 5, 0, 5, 8, 12]},
    ),
    (
        "sql_nulls",
        """A SQL table jobs(team TEXT, result TEXT, units INTEGER) contains exactly these rows. NULL is SQL NULL.

('A','ok',4), ('A',NULL,7), ('A','skip',9), ('A','error',8),
('B','skip',NULL), ('B','ok',3), ('B','error',5),
('C','ok',NULL), ('C','ok',-2), ('C',NULL,6), ('C','skip',0)

SELECT team, COUNT(*), COUNT(units),
       SUM(CASE WHEN result <> 'skip' THEN COALESCE(units, 0) ELSE 0 END)
FROM jobs
WHERE result IS NULL OR result <> 'error'
GROUP BY team
HAVING COUNT(units) >= 2
ORDER BY team;

Return just the result rows as a JSON array of arrays, in query order.""",
        [["A", 3, 3, 4], ["C", 4, 3, -2]],
    ),
    (
        "rate_limiter",
        """A token bucket starts at time 0 with exactly 2 tokens. Its capacity is 5 tokens. Before each arrival it refills continuously at exactly 2 tokens per second, preserving fractional tokens and saturating at 5. An arrival is accepted iff the bucket contains at least its cost; on acceptance deduct the cost. Rejection consumes nothing. There is no queue. Process equal timestamps in the listed order.

ID  Arrival milliseconds  Cost
A   0                     3
B   250                   2
C   500                   1
D   750                   2
E   2000                  3
F   4000                  5
G   6000                  4
H   6000                  2
I   6750                  2

Return only a JSON object with "accepted" (accepted IDs as an array of strings in order) and "remaining_millitokens" (integer residual immediately after arrival I; 1 token = 1000 millitokens).""",
        {"accepted": ["B", "C", "E", "G", "I"], "remaining_millitokens": 500},
    ),
    (
        "grounded_incident",
        """The following is the entire surviving incident log for request R9. The service contract says ACCEPT means only that a region accepted a submission. PREPARED is also provisional. A COMMIT entry with durable=true is the only evidence that identifies a durable commit region. The log is partial and time order alone establishes no commit.

10:00 west  R9  ACCEPT  attempt=1
10:01 west  R9  PREPARED
10:02 south R8  COMMIT  durable=true
10:03 east  R9  ACCEPT  attempt=2
10:04 east  R9  COMMIT  durable=false
10:05 west  R9  CONNECTION_LOST
10:06 north R9  FORWARDED target=east

Using just the log, return one JSON object with "accepted_regions" (alphabetically sorted distinct regions that explicitly accepted R9) and "durable_commit_region" (a region string if identified by the contract, JSON null if it cannot be determined). Return no additional text.""",
        {"accepted_regions": ["east", "west"], "durable_commit_region": None},
    ),
]
SETTINGS = {
    "temperature": 0,
    "top_p": 1,
    "seed": 42,
    "max_tokens": 4096,
    "reasoning_effort": "low",
    "stream": True,
    "stream_options": {"include_usage": True},
}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def answer_check(answer, expected):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        decoded = json.loads(answer.strip(), object_pairs_hook=unique)
        return canonical(decoded) == canonical(expected)
    except (ValueError, TypeError):
        return False


def verify_keys():
    # Solve the scheduling case by enumerating both resource orderings and their earliest feasible starts.
    durations = dict(zip("ABCDEF", (4, 3, 5, 2, 4, 3), strict=True))
    schedules = []
    for cpu in permutations("ADF"):
        for gpu in permutations("BCE"):
            preds = defaultdict(
                set, {"B": {"A"}, "D": {"C"}, "E": {"D"}, "F": {"B", "E"}}
            )
            for order in (cpu, gpu):
                for a, b in pairwise(order):
                    preds[b].add(a)
            starts = {}
            for _ in durations:
                for task in durations:
                    if task not in starts and preds[task] <= starts.keys():
                        starts[task] = max(
                            (starts[p] + durations[p] for p in preds[task]), default=0
                        )
            if len(starts) == len(durations):
                schedules.append(
                    (
                        max(starts[t] + durations[t] for t in durations),
                        [starts[t] for t in durations],
                    )
                )
    finish, optimal = min(schedules)
    rows = [[2], [3], [5]]
    shared = list(rows)
    for i, row in enumerate(rows):
        row.append(sum(item[0] for item in shared[: i + 1]))
    shared[1] = [11]
    rows[2].extend(shared[0][-1:])
    alias_answer = list(map(sum, rows + shared))

    available = (436 - 40 - 92) * 1024 - 87 * 3 * 128
    physical = available // (16 * 96)
    admitted, residual = divmod(physical - 1, (4097 + 127) // 128)

    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE jobs(team TEXT,result TEXT,units INTEGER)")
        db.executemany(
            "INSERT INTO jobs VALUES(?,?,?)",
            [
                ("A", "ok", 4),
                ("A", None, 7),
                ("A", "skip", 9),
                ("A", "error", 8),
                ("B", "skip", None),
                ("B", "ok", 3),
                ("B", "error", 5),
                ("C", "ok", None),
                ("C", "ok", -2),
                ("C", None, 6),
                ("C", "skip", 0),
            ],
        )
        sql_answer = [
            list(row)
            for row in db.execute(
                """SELECT team,COUNT(*),COUNT(units),SUM(CASE WHEN result <> 'skip' THEN COALESCE(units,0) ELSE 0 END) FROM jobs WHERE result IS NULL OR result <> 'error' GROUP BY team HAVING COUNT(units)>=2 ORDER BY team"""
            )
        ]

    accepted = []
    tokens = Fraction(2)
    last = 0
    for name, ms, cost in zip(
        "ABCDEFGHI",
        (0, 250, 500, 750, 2000, 4000, 6000, 6000, 6750),
        (3, 2, 1, 2, 3, 5, 4, 2, 2),
        strict=True,
    ):
        tokens = min(5, tokens + Fraction(ms - last, 500))
        last = ms
        if tokens >= cost:
            tokens -= cost
            accepted.append(name)
    log_rows = [
        line.split() for line in CASES[-1][1].splitlines() if line.startswith("10:")
    ]
    usable = [row for row in log_rows if row[2] == "R9"]
    commits = [row[1] for row in usable if row[3:] == ["COMMIT", "durable=true"]]
    computed = [
        alias_answer,
        [physical, (physical - 1) * 128, admitted, residual],
        {"finish": finish, "starts": optimal},
        sql_answer,
        {"accepted": accepted, "remaining_millitokens": int(tokens * 1000)},
        {
            "accepted_regions": sorted(
                {row[1] for row in usable if row[3] == "ACCEPT"}
            ),
            "durable_commit_region": commits[0] if commits else None,
        },
    ]
    for (name, _prompt, expected), reference in zip(CASES, computed, strict=True):
        if not answer_check(canonical(reference), expected):
            raise ValueError(
                f"Answer key differs from independent computation for {name}: {reference}"
            )
    for bad, expected in [
        ('{"a":true}', {"a": 1}),
        ('{"a":2,"a":1}', {"a": 1}),
        ("answer [1]", [1]),
        ('{"a":1.0}', {"a": 1}),
    ]:
        if answer_check(bad, expected):
            raise ValueError(
                "Strict JSON scorer accepted a wrong type, duplicate, or prose"
            )


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args):
        return None


def local_url(option):
    parsed = urlsplit(option.rstrip("/"))
    if (
        parsed.scheme not in ("http", "https")
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/v1")
    ):
        raise ValueError("Supply the loopback API base URL, optionally ending in /v1")
    if (
        parsed.hostname != "localhost"
        and not ipaddress.ip_address(parsed.hostname or "").is_loopback
    ):
        raise ValueError("Only loopback model servers are supported")
    return f"{parsed.scheme}://{parsed.netloc}/v1/chat/completions"


def request(opener, endpoint, model, case, timeout, *, settings=None):
    name, prompt, expected = case
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **(SETTINGS if settings is None else settings),
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    started = time.monotonic()
    first_final = first_generated = None
    content = []
    reasoning_chars = 0
    finish = usage = None
    done = False
    status = None
    error = None
    try:
        with opener.open(req, timeout=timeout) as response:
            status = response.status
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data:"):
                    continue
                packet = line[5:].strip()
                if packet == "[DONE]":
                    done = True
                    break
                row = json.loads(packet)
                if row.get("error"):
                    raise ValueError(str(row["error"]))
                if row.get("usage") is not None:
                    usage = row["usage"]
                for choice in row.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = (
                        delta.get("reasoning") or delta.get("reasoning_content") or ""
                    )
                    visible = delta.get("content") or ""
                    now = time.monotonic() - started
                    if first_generated is None and (
                        reasoning.strip() or visible.strip()
                    ):
                        first_generated = now
                    if first_final is None and visible.strip():
                        first_final = now
                    reasoning_chars += len(reasoning)
                    content.append(visible)
                    finish = choice.get("finish_reason") or finish
    except (OSError, ValueError, KeyError, TypeError, http.client.HTTPException) as exc:
        detail = (
            exc.read(4096).decode(errors="replace")
            if isinstance(exc, urllib.error.HTTPError)
            else ""
        )
        status = getattr(exc, "code", status)
        error = f"{type(exc).__name__}: {exc} {detail}".strip()
    elapsed = time.monotonic() - started
    answer = "".join(content)
    match = answer_check(answer, expected)
    if not error and (
        not done
        or not usage
        or not isinstance(usage.get("completion_tokens"), int)
        or not finish
    ):
        error = "Incomplete SSE stream, finish reason, or token usage"
    return {
        "id": name,
        "messages_sha256": sha(body["messages"]),
        "answer": answer,
        "expected": expected,
        "strict_answer_match": match,
        "scored_pass": error is None and finish == "stop" and match,
        "finish_reason": finish,
        "natural_stop": finish == "stop",
        "capped": finish == "length",
        "stream_done": done,
        "http_status": status,
        "usage": usage,
        "reasoning_characters": reasoning_chars,
        "first_generated_seconds": first_generated,
        "first_visible_answer_seconds": first_final,
        "wall_seconds": elapsed,
        "error": error,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--base-url")
    p.add_argument("--model")
    p.add_argument("--output", type=Path)
    p.add_argument("--timeout", type=float, default=360)
    args = p.parse_args()
    verify_keys()
    if args.self_test:
        print("All six independent answers and strict JSON parsing checks passed.")
        return 0
    if not args.base_url or not args.model or not args.output:
        p.error("--base-url, --model and --output are required")
    endpoint = local_url(args.base_url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    results = []
    doc = {
        "schema": 1,
        "model": args.model,
        "endpoint": endpoint,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "settings": SETTINGS,
        "client_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixtures_sha256": sha(
            [
                {
                    "id": c[0],
                    "messages": [{"role": "user", "content": c[1]}],
                    "expected": c[2],
                }
                for c in CASES
            ]
        ),
        "grading": "Exact parsed JSON data and types, with a completed SSE stream, token usage and natural stop. Whitespace and object key ordering are ignored.",
        "scope": "Fixed six-case technical screen. First visible answer excludes reasoning. Wall time includes reasoning, transfer and stop. No repetitions or broad quality inference.",
        "complete": False,
        "results": results,
    }
    try:
        with args.output.open("x") as f:
            json.dump(doc, f, indent=2)
    except FileExistsError:
        p.error(f"Output already exists: {args.output}")
    temp = args.output.with_name(args.output.name + ".partial")

    def save():
        temp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)

    for case in CASES:
        row = request(opener, endpoint, args.model, case, args.timeout)
        results.append(row)
        save()
        print(
            f"{row['id']}  pass={row['scored_pass']}  finish={row['finish_reason']}  seconds={row['wall_seconds']:.2f}",
            flush=True,
        )
        if row["error"]:
            print(row["error"], file=sys.stderr)
            return 2
    durations = [row["wall_seconds"] for row in results]
    doc.update(
        complete=True,
        summary={
            "passed": sum(r["scored_pass"] for r in results),
            "cases": len(results),
            "naturally_stopped": sum(r["natural_stop"] for r in results),
            "capped": sum(r["capped"] for r in results),
            "median_case_wall_seconds": statistics.median(durations),
            "total_case_wall_seconds": sum(durations),
        },
    )
    save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
