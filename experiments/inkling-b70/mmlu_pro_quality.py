"""Score a frozen, zero-shot technical subset of the official MMLU-Pro test split."""

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from quality import SETTINGS, NoRedirect, local_url, request, sha
from standard_quality import summarize

SEED = "brutus-inkling-20261126"
SOURCE = {
    "repo": "TIGER-Lab/MMLU-Pro",
    "revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
    "split": "test",
    "path": "data/test-00000-of-00001.parquet",
    "parquet_sha256": "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
    "card_sha256": "4bd710f67da3fa359a33edce1b4b5816b3de416c823c2624ba5e89c2557d2a47",
    "card_license": "mit",
    "rows": 12032,
}
# First question IDs per subject in SHA-256 rank order. Answers, question
# text and options are absent from the selection key.
SELECTED = {
    "computer science": [10721, 10551, 10545, 10632, 10462, 10408],
    "engineering": [11343, 11922, 11989, 11612, 11635],
    "math": [8744, 8122, 8065, 8178, 8834],
}
LETTERS = "ABCDEFGHIJ"
GENERATION = {**SETTINGS, "reasoning_effort": "medium", "max_tokens": 4096}
MANIFEST_SHA256 = "8e152fc5312fe1d2b45af4af9e7a574648db5e5bf562ce56b1e2177cb908a8e7"


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cases(directory):
    import pyarrow.parquet as pq

    parquet = directory / "test.parquet"
    card = directory / "README.md"
    if (
        file_sha(parquet) != SOURCE["parquet_sha256"]
        or file_sha(card) != SOURCE["card_sha256"]
    ):
        raise ValueError("MMLU-Pro test split or dataset card checksum mismatch")
    if "license: mit" not in card.read_text().splitlines():
        raise ValueError("Missing pinned license declaration")
    rows = pq.read_table(
        parquet,
        columns=[
            "category",
            "question_id",
            "question",
            "options",
            "answer",
            "answer_index",
        ],
    ).to_pylist()
    if len(rows) != SOURCE["rows"] or len({r["question_id"] for r in rows}) != len(
        rows
    ):
        raise ValueError("Unexpected split length or duplicate question IDs")
    cases = []
    for category, ids in SELECTED.items():
        ranked = sorted(
            (r for r in rows if r["category"] == category),
            key=lambda r: sha([SEED, "mmlu-pro/test", category, r["question_id"]]),
        )[: len(ids)]
        if [row["question_id"] for row in ranked] != ids:
            raise ValueError("Frozen selection diverged from the original ranking")
        for row in ranked:
            options = [s for s in row["options"] if s != "N/A"]
            gold = row["answer"]
            if not 2 <= len(options) <= len(LETTERS) or not 0 <= row[
                "answer_index"
            ] < len(options):
                raise ValueError("Invalid choice list")
            if (
                LETTERS[row["answer_index"]] != gold
                or options[row["answer_index"]] != row["options"][row["answer_index"]]
            ):
                raise ValueError("Official letter, option and answer index disagree")
            choices = "\n".join(
                f"{letter}. {option}" for letter, option in zip(LETTERS, options)
            )
            text = (
                row["question"]
                + "\n\n"
                + choices
                + "\n\nGive your final answer as a single option letter."
            )
            cases.append(
                {
                    "question_id": row["question_id"],
                    "category": category,
                    "prompt": text,
                    "options": options,
                    "expected": gold,
                    "messages_sha256": sha([{"role": "user", "content": text}]),
                }
            )
    cases.sort(key=lambda item: sha([SEED, "mmlu-pro/order", item["question_id"]]))
    return cases


def parse_letter(text, options):
    """Read a final choice without guessing from a stray letter in reasoning."""
    cleaned = re.sub(r"[\u0024`*]", "", text).replace(r"\(", "").replace(r"\)", "")
    cleaned = re.sub(r"\\boxed\{\s*([A-J])\s*\}", r"\1", cleaned, flags=re.IGNORECASE)
    lines = [s.strip() for s in cleaned.splitlines() if s.strip()]
    if not lines:
        return None
    label = (
        r"(?:the\s+)?(?:final\s+answer|correct\s+(?:answer|option|choice)|answer|my\s+choice)"
        r"\s*(?::|=|\bis\b)\s*|I\s+(?:choose|select|pick)\s+"
    )
    choice = r"(?:option\s+|choice\s+)?[\[(]?([A-J])\b[\])]?"
    direct = re.compile(r"(?:(?:" + label + r"))?" + choice + r"[.!]?", re.IGNORECASE)
    candidates = set()
    if any(
        re.fullmatch(
            choice + r"\s*(?:/|or|and)\s*" + choice + r"[.!]?", line, re.IGNORECASE
        )
        for line in lines
    ):
        return None
    # Either edge may contain the answer with explanatory prose on the other.
    # Collect all explicit selections before accepting any of them.
    for line in (lines[0], lines[-1]):
        if match := direct.fullmatch(line):
            candidates.add(match[1].upper())
        elif echo := re.fullmatch(
            r"[\[(]?([A-J])[\])]?\s*[.):-]\s*(.+)", line, re.IGNORECASE
        ):
            index = LETTERS.index(echo[1].upper())
            if (
                index < len(options)
                and echo[2].strip().rstrip(".").casefold()
                == options[index].strip().rstrip(".").casefold()
            ):
                candidates.add(LETTERS[index])

    for match in re.finditer(r"\b(?:" + label + r")" + choice, cleaned, re.IGNORECASE):
        suffix = cleaned[match.end() :]
        if re.match(
            r"^\s*(?:/|,\s*[A-J]\b|(?:or|and)\s+(?:option\s+)?[\[(]?[A-J]\b)",
            suffix,
            re.IGNORECASE,
        ):
            return None
        # The lower-case article in "the answer is a positive number" is prose.
        if match[1] == "a" and re.match(r"^[ \t]+[a-z]", suffix):
            continue
        candidates.add(match[1].upper())
    for match in re.finditer(
        r"\b(?:[Oo]ption\s+)?([A-J])\)?\s+is\s+(?:the\s+)?(?:correct|right)(?:[.!]|\s+(?:choice|option|answer)\b)",
        cleaned,
    ):
        candidates.add(match[1])
    if len(candidates) != 1:
        return None
    candidate = candidates.pop()
    return candidate if candidate in LETTERS[: len(options)] else None


def parser_self_test():
    for letter in LETTERS:
        for final in (
            letter,
            f"({letter})",
            f"**Final answer: ({letter})**.",
            f"Thinking.\nThe answer is {letter}.",
            f"**{letter}**\n\nA calculation gives the stated result.",
            rf"\boxed{{{letter}}}",
        ):
            assert parse_letter(final, list(LETTERS)) == letter
    for final in (
        "",
        "A or B",
        "The answer is (A) or B.",
        "Answer: C. Answer: D.",
        "**C**\n\nThe final answer is D.",
        "**C**\n\nI choose D.",
        "C\nD",
        "C\nOption D is correct.",
        "C\nMy choice is D or E.",
        "C\nC or D.",
        "I considered F for some time.",
        "K",
    ):
        assert parse_letter(final, list(LETTERS)) is None
    assert parse_letter("J", ["one", "two"]) is None
    assert parse_letter("B. two", ["one", "two"]) == "B"


def grade(row, item):
    row["parsed_answer"] = parse_letter(row["answer"], item["options"])
    row["scored_pass"] = (
        row["natural_stop"]
        and row["error"] is None
        and row["parsed_answer"] == item["expected"]
    )


def summary(rows):
    return {
        "all": summarize(rows),
        **{c: summarize([r for r in rows if r["category"] == c]) for c in SELECTED},
    }


def regrade(source, output, cases):
    raw = source.read_bytes()
    doc = json.loads(raw)
    if (
        not doc["complete"]
        or doc["source"] != SOURCE
        or doc["seed"] != SEED
        or doc["manifest_sha256"] != MANIFEST_SHA256
        or doc["settings"] != GENERATION
    ):
        raise ValueError(
            "Regrading requires a complete run with the frozen data and generation settings"
        )
    changes = []
    for row, item in zip(doc["results"], cases, strict=True):
        if (
            any(
                row[k] != item[k]
                for k in ("question_id", "category", "messages_sha256", "expected")
            )
            or row["id"] != f"mmlu-pro-test-{item['question_id']}"
            or row["error"] is not None
            or not row["stream_done"]
            or not row["usage"]
            or row["natural_stop"] != (row["finish_reason"] == "stop")
            or row["capped"] != (row["finish_reason"] == "length")
        ):
            raise ValueError(
                f"Source result does not match the frozen item or streaming receipt: {row['id']}"
            )
        original = {
            "parsed_answer": row["parsed_answer"],
            "scored_pass": row["scored_pass"],
        }
        grade(row, item)
        if any(row[k] != v for k, v in original.items()):
            changes.append(
                {
                    "question_id": row["question_id"],
                    "original": original,
                    "regraded": {k: row[k] for k in original},
                }
            )
    doc["regrade"] = {
        "source_file": source.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "source_grader_sha256": doc["grader_sha256"],
        "changes": changes,
    }
    doc["grader_sha256"] = file_sha(Path(__file__))
    doc["summary"] = summary(doc["results"])
    with output.open("x") as f:
        f.write(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"regrade": doc["regrade"], "summary": doc["summary"]["all"]}, indent=2
        )
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory holding pinned test.parquet and README.md",
    )
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument(
        "--regrade",
        type=Path,
        help="Grade a completed local receipt offline into a new --output file",
    )
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--output", type=Path)
    ap.add_argument("--timeout", type=float, default=360)
    args = ap.parse_args()
    parser_self_test()
    cases = load_cases(args.data_dir)
    manifest = [
        {k: row[k] for k in ("question_id", "category", "messages_sha256", "expected")}
        for row in cases
    ]
    if sha(manifest) != MANIFEST_SHA256:
        raise ValueError(f"Frozen manifest differs: {sha(manifest)}")
    if args.verify_only:
        print(
            f"Verified official split, card license, {len(manifest)} ranked IDs, prompt hashes, answer indices and scorer. Manifest {sha(manifest)}"
        )
        return 0
    if args.regrade:
        if args.output is None:
            ap.error("--regrade requires a new --output file")
        regrade(args.regrade, args.output, cases)
        return 0
    if not args.base_url or not args.model or not args.output or args.timeout <= 0:
        ap.error("Supply --base-url, --model, --output and a positive timeout")
    endpoint = local_url(args.base_url)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    rows = []
    doc = {
        "schema": 1,
        "source": SOURCE,
        "seed": SEED,
        "manifest_sha256": sha(manifest),
        "model": args.model,
        "endpoint": endpoint,
        "settings": GENERATION,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "results": rows,
        "grader_sha256": file_sha(Path(__file__)),
        "client_sha256": file_sha(Path(__file__).with_name("quality.py")),
        "complete": False,
    }
    with args.output.open("x") as f:
        json.dump(doc, f)
    for item in cases:
        row = request(
            opener,
            endpoint,
            args.model,
            (f"mmlu-pro-test-{item['question_id']}", item["prompt"], item["expected"]),
            args.timeout,
            settings=GENERATION,
        )
        if row["messages_sha256"] != item["messages_sha256"]:
            raise ValueError("Client sent a different prompt than the frozen item")
        row.pop("strict_answer_match")
        grade(row, item)
        row.update(question_id=item["question_id"], category=item["category"])
        rows.append(row)
        doc["summary"] = summary(rows)
        doc["complete"] = len(rows) == len(cases) and row["error"] is None
        tmp = args.output.with_suffix(args.output.suffix + ".partial")
        tmp.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        tmp.replace(args.output)
        print(
            f"{len(rows):2}/16 {row['id']} {row['parsed_answer']} correct={row['scored_pass']} finish={row['finish_reason']} seconds={row['wall_seconds']:.1f}",
            flush=True,
        )
        if row["error"]:
            print(row["error"], flush=True)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
