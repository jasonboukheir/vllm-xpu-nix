"""Score the same pinned GSM8K and CRUXEval-O items on a local chat endpoint."""

import argparse
import ast
import hashlib
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from quality import SETTINGS, NoRedirect, local_url, request, sha

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "standard_quality_sample.json"
SEED = "brutus-inkling-20261126"
EFFORTS = ("low", "medium")
SOURCES = {
    "gsm8k": {
        "repo": "openai/grade-school-math",
        "revision": "3101c7d5072418e28b9008a6636bde82a006892c",
        "path": "grade_school_math/data/test.jsonl",
        "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
        "rows": 1319,
        "count": 20,
        "license": "MIT",
        "copyright": "2021 OpenAI",
    },
    "cruxeval": {
        "repo": "facebookresearch/cruxeval",
        "revision": "190faf16d175b5847b0af05d937872b1fb395942",
        "path": "data/cruxeval.jsonl",
        "sha256": "8368b81047dc5014e4caf5a2f97604eff7644e0ecd7415e3ceeb184bbc2e0c96",
        "rows": 800,
        "count": 12,
        "license": "MIT",
        "copyright": "2023 Meta",
    },
}
NUMBER = re.compile(r"(?<![\w.])[+−-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w.])")
FENCE = re.compile(
    r"^```(?:python|py|json|text)?\s*\n?(.*?)\n?```$", re.IGNORECASE | re.DOTALL
)
LABEL = re.compile(
    r"^(?:final\s+)?(?:answer|output|result)\s*(?:is\b|[:=])\s*", re.IGNORECASE
)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def select(data_dir):
    """Read the two official JSONL files as text. Never import or execute a task."""
    cases = []
    for dataset, meta in SOURCES.items():
        raw = (data_dir / (dataset + ".jsonl")).read_bytes()
        if digest(raw) != meta["sha256"]:
            raise ValueError(f"{dataset} raw file SHA-256 mismatch")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) != meta["rows"]:
            raise ValueError(f"Unexpected row count for {dataset}")
        indices = sorted(range(len(rows)), key=lambda i: sha([SEED, dataset, i]))
        for i in indices[: meta["count"]]:
            r = rows[i]
            item = {"dataset": dataset, "source_row": i, "source_row_sha256": sha(r)}
            if dataset == "gsm8k":
                item.update(
                    id=f"gsm8k-test-{i}",
                    question=r["question"],
                    expected=r["answer"].rsplit("#### ", 1)[1].strip(),
                )
            else:
                item.update(
                    id="cruxeval-o-" + r["id"],
                    code=r["code"],
                    arguments=r["input"],
                    expected=r["output"],
                )
            cases.append(item)
    # Interleave domains deterministically, identically on every server.
    cases.sort(key=lambda x: sha([SEED, "order", x["id"]]))
    return {"selection_seed": SEED, "sources": SOURCES, "items": cases}


def prompt(item):
    if item["dataset"] == "gsm8k":
        return (
            item["question"]
            + "\n\nSolve this problem. Return the final numerical answer by itself, without units."
        )
    return (
        "Predict the return value of this Python function for the call below. "
        "Return just the value as a Python literal, with quotes around strings."
        "\n\n" + item["code"] + "\n\nCall: f(" + item["arguments"] + ")"
    )


def unwrap(text):
    text = LABEL.sub("", text.strip()).strip()
    match = FENCE.fullmatch(text)
    if match:
        text = match[1].strip()
    if text.startswith("`") and text.endswith("`") and len(text) >= 2:
        text = text[1:-1].strip()
    return LABEL.sub("", text).strip()


def typed_literal(value):
    # Python equates True with 1 even inside containers; the printed return type matters.
    if isinstance(value, dict):
        return (
            dict,
            frozenset((typed_literal(k), typed_literal(v)) for k, v in value.items()),
        )
    if isinstance(value, (tuple, list)):
        return (type(value), tuple(map(typed_literal, value)))
    if isinstance(value, (set, frozenset)):
        return (type(value), frozenset(map(typed_literal, value)))
    return (type(value), value)


def score(item, answer):
    """Only grade final-channel text. Never execute generated code or source snippets."""
    value = unwrap(answer)
    # A clearly separated answer may follow prose; an entire multiline literal stays first.
    candidates = [value]
    if "\n" in value:
        last = value.splitlines()[-1]
        if last.strip():
            candidates.append(unwrap(last))
        suffix = re.search(
            r"```(?:python|py|json|text)?\s*\n(.*?)\n```\s*\Z",
            value,
            re.DOTALL | re.IGNORECASE,
        )
        if suffix:
            candidates.append(suffix[1].strip())
    if item["dataset"] == "gsm8k":
        expected = Decimal(item["expected"].replace(",", ""))
        for s in candidates:
            s = s.replace("**", "").replace("$", "").removeprefix("####").strip()
            s = s.removesuffix(".").strip()
            # A final equality explicitly identifies the result even when the
            # same line contains intermediate operands (e.g. "Total: 27+18=45").
            if "=" in s and "\n" not in s:
                s = s.rsplit("=", 1)[1].strip()
            if s.startswith(r"\boxed{") and s.endswith("}"):
                s = s[len(r"\boxed{") : -1].strip()
            s = re.sub(
                r"^(?:therefore,? +)?(?:the +)?(?:final +)?answer +is +",
                "",
                s,
                flags=re.IGNORECASE,
            )
            match = NUMBER.fullmatch(s)
            if (
                match
                and Decimal(match[0].replace(",", "").replace("−", "-")) == expected
            ):
                return True
        return False
    expected = ast.literal_eval(item["expected"])
    for s in candidates:
        try:
            actual = ast.literal_eval(s)
        except (ValueError, SyntaxError, RecursionError):
            if isinstance(expected, str) and s == expected:
                return True
            continue
        if typed_literal(actual) == typed_literal(expected):
            return True
    return False


def summarize(rows):
    return {
        "cases": len(rows),
        "passed": sum(r["scored_pass"] for r in rows),
        "finish_reasons": dict(Counter(r["finish_reason"] for r in rows)),
        "naturally_stopped": sum(r["natural_stop"] for r in rows),
        "capped": sum(r["capped"] for r in rows),
        "elapsed_seconds": sum(r["wall_seconds"] for r in rows),
        "completion_tokens": sum(
            (r["usage"] or {}).get("completion_tokens", 0) for r in rows
        ),
    }


def grade_row(row, item):
    row.pop("strict_answer_match", None)
    row["dataset"] = item["dataset"]
    row["answer_match"] = score(item, row["answer"])
    row["scored_pass"] = (
        row["error"] is None and row["natural_stop"] and row["answer_match"]
    )


def summarize_domains(rows):
    return {
        **{d: summarize([r for r in rows if r["dataset"] == d]) for d in SOURCES},
        "all": summarize(rows),
    }


def regrade(source, output, corpus, reasoning_effort=None):
    raw = source.read_bytes()
    doc = json.loads(raw)
    recorded_effort = doc["settings"].get("reasoning_effort")
    if (
        recorded_effort not in EFFORTS
        or (reasoning_effort is not None and recorded_effort != reasoning_effort)
        or not doc["complete"]
        or doc["sample_sha256"] != digest(SAMPLE.read_bytes())
        or doc["settings"] != {**SETTINGS, "reasoning_effort": recorded_effort}
    ):
        raise ValueError(
            "Regrading requires a complete run of this sample and generation configuration"
        )
    for row, item in zip(doc["results"], corpus["items"], strict=True):
        messages = [{"role": "user", "content": prompt(item)}]
        if (row["id"], row["expected"], row["messages_sha256"]) != (
            item["id"],
            item["expected"],
            sha(messages),
        ):
            raise ValueError(
                f"Source row does not match the frozen sample: {row['id']}"
            )
        grade_row(row, item)
    doc["summary"] = summarize_domains(doc["results"])
    doc["original_grader_sha256"] = doc["grader_sha256"]
    doc["grader_sha256"] = digest(Path(__file__).read_bytes())
    doc["original_results_sha256"] = digest(raw)
    with output.open("x") as f:
        f.write(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(json.dumps(doc["summary"], indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument(
        "--reasoning-effort",
        choices=EFFORTS,
        help="Request reasoning effort (default low). Regrading uses the recorded effort unless specified.",
    )
    parser.add_argument(
        "--regrade",
        type=Path,
        help="Rescore a saved run offline into a new --output file",
    )
    parser.add_argument(
        "--verify-data-dir",
        type=Path,
        help="Compare the frozen selection to verified gsm8k.jsonl and cruxeval.jsonl",
    )
    args = parser.parse_args()
    corpus = json.loads(SAMPLE.read_text())
    if args.verify_data_dir:
        if select(args.verify_data_dir) != corpus:
            raise SystemExit("Pinned sample and the original dataset disagree")
        print("Verified all 32 selected rows against both pinned official files.")
        return
    if args.regrade:
        if args.output is None:
            parser.error("--regrade requires --output")
        regrade(args.regrade, args.output, corpus, args.reasoning_effort)
        return 0
    if not args.base_url or not args.model or not args.output:
        parser.error("supply --base-url, --model and --output")
    endpoint = local_url(args.base_url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    settings = {
        **SETTINGS,
        "reasoning_effort": args.reasoning_effort or SETTINGS["reasoning_effort"],
    }
    rows = []
    doc = {
        "schema": 1,
        "model": args.model,
        "endpoint": endpoint,
        "settings": settings,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "sample_sha256": digest(SAMPLE.read_bytes()),
        "grader_sha256": digest(Path(__file__).read_bytes()),
        "client_sha256": digest((HERE / "quality.py").read_bytes()),
        "complete": False,
        "results": rows,
    }
    with args.output.open("x") as f:
        json.dump(doc, f)
    temp = args.output.with_suffix(args.output.suffix + ".partial")
    for item in corpus["items"]:
        row = request(
            opener,
            endpoint,
            args.model,
            (item["id"], prompt(item), item["expected"]),
            args.timeout,
            settings=settings,
        )
        grade_row(row, item)
        rows.append(row)
        doc["summary"] = summarize_domains(rows)
        doc["complete"] = len(rows) == len(corpus["items"]) and not row["error"]
        temp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        temp.replace(args.output)
        print(
            f"{len(rows):2}/32 {row['id']} pass={row['scored_pass']} finish={row['finish_reason']} seconds={row['wall_seconds']:.2f}",
            flush=True,
        )
        if row["error"]:
            print(row["error"], file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
