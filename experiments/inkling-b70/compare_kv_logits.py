#!/usr/bin/env python3
"""Compare next-token predictions on identical token prefixes across Inkling KV formats."""

import argparse
import hashlib
import json
import statistics
import urllib.request
from pathlib import Path

SECTIONS = (
    (
        "inventory",
        "Bay {idx:03d} lists {qty} gaskets for shipment {code}.",
        "The shipment manifest records quantities by bay, so a verifier should check",
    ),
    (
        "code",
        "cache[{idx}] = ({qty}, 'rack-{code}')  # process the next observation",
        "def read_latest(cache, key):\n    if key not in cache:\n        return None\n    return",
    ),
    (
        "incident",
        "[10:{qty:02d}] runner-{idx}: acknowledged checkpoint {code}, verification pending.",
        "Acknowledgment alone does not establish a durable write. To confirm durability,",
    ),
)


def post(base, route, body):
    req = urllib.request.Request(
        base + route, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def capture(args):
    rows = []
    for label, pattern, tail in SECTIONS:
        history = "\n".join(
            pattern.format(idx=i, qty=(i * 17) % 59, code=1000 + (i * 23) % 997)
            for i in range(190)
        )
        prompt = post(
            args.url,
            "/tokenize",
            {"model": args.model, "prompt": history + "\n\n" + tail},
        )
        tokens = prompt["tokens"]
        for i in range(12):
            ids = tokens[: len(tokens) - 11 + i]
            response = post(
                args.url,
                "/v1/completions",
                {
                    "model": args.model,
                    "prompt": ids,
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": 1,
                    "ignore_eos": True,
                    "logprobs": 5,
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                },
            )
            choice = response["choices"][0]
            top = choice["logprobs"]["top_logprobs"][0]
            rows.append(
                {
                    "case": label,
                    "position": i,
                    "n": len(ids),
                    "input_sha256": hashlib.sha256(
                        json.dumps(ids).encode()
                    ).hexdigest(),
                    "token": choice["token_ids"][0],
                    "top": top,
                }
            )
    args.output.write_text(json.dumps(rows, indent=2) + "\n")


def compare(args):
    left, right = (
        json.loads(path.read_text()) for path in (args.reference, args.candidate)
    )
    if len(left) != len(right) or not left:
        raise ValueError("Different sample counts")
    matches = 0
    drops = []
    mismatch = []
    absent = 0
    for a, b in zip(left, right, strict=True):
        if (a["input_sha256"], a["n"]) != (b["input_sha256"], b["n"]):
            raise ValueError("Compared inputs differ")
        token = max(a["top"], key=a["top"].get)
        if a["token"] == b["token"]:
            matches += 1
        else:
            ordered = sorted(a["top"].values(), reverse=True)
            mismatch.append(
                {
                    "case": a["case"],
                    "position": a["position"],
                    "ref_token": a["token"],
                    "new_token": b["token"],
                    "ref_margin": ordered[0] - ordered[1],
                }
            )
        if token in b["top"]:
            drops.append(abs(b["top"][token] - a["top"][token]))
        else:
            absent += 1
    print(
        json.dumps(
            {
                "positions": len(left),
                "top1_matches": matches,
                "ref_top1_missing_from_candidate_top5": absent,
                "median_abs_logprob_change_among_returned": statistics.median(drops)
                if drops
                else None,
                "mismatches": mismatch,
            },
            indent=2,
        )
    )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("model")
    cap.add_argument("output", type=Path)
    cap.add_argument("--url", default="http://127.0.0.1:18001")
    cmp = commands.add_parser("compare")
    cmp.add_argument("reference", type=Path)
    cmp.add_argument("candidate", type=Path)
    args = cli.parse_args()
    if args.command == "capture":
        capture(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
