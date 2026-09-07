"""Compare matched MTP-off/on captures within one KV dtype, without hiding drift."""

import argparse
import json
import math
import re
import shutil
import statistics
from pathlib import Path

from scripts import kvarn_vision_compare as vision
from scripts import kvarn_vision_run as runner


def without_mtp(argv: list[str], *, enabled: bool | int) -> list[str]:
    result = list(argv)
    count = result.count("--speculative-config")
    if count != int(bool(enabled)):
        raise ValueError("unexpected speculative configuration")
    if enabled:
        index = result.index("--speculative-config")
        if json.loads(result[index + 1]) != runner.speculative_config(enabled):
            raise ValueError("outside requested bundled MTP envelope")
        del result[index : index + 2]
    return result


def tokens(directory: Path, name: str) -> tuple[list[int], list[int]]:
    prompt = None
    output = []
    for line in (directory / f"{name}-sse.jsonl").read_text().splitlines():
        raw = json.loads(line)["line"]
        if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
            continue
        event = json.loads(raw[6:])
        if event.get("prompt_token_ids") is not None:
            if prompt is not None:
                raise ValueError("duplicate prompt token evidence")
            prompt = event["prompt_token_ids"]
        for choice in event.get("choices", []):
            output.extend(choice.get("token_ids") or [])
    usage = vision.load(directory / f"{name}-response.json")["usage"]
    if prompt is None or len(prompt) != usage["prompt_tokens"]:
        raise ValueError("missing or incomplete prompt token evidence")
    if len(output) != usage["completion_tokens"]:
        raise ValueError("missing or incomplete generated token evidence")
    if any(type(value) is not int for value in prompt + output):
        raise ValueError("invalid token ID")
    return prompt, output


def logprobs(directory: Path, name: str, output: list[int]) -> list[dict] | None:
    """Read target logprobs, rejecting missing or misaligned generated tokens."""
    requested = vision.load(directory / f"{name}-request.json").get("logprobs")
    if not requested:
        return None
    rows = []
    for line in (directory / f"{name}-sse.jsonl").read_text().splitlines():
        raw = json.loads(line)["line"]
        if not raw.startswith("data: ") or raw.strip() == "data: [DONE]":
            continue
        for choice in json.loads(raw[6:]).get("choices", []):
            rows.extend((choice.get("logprobs") or {}).get("content") or [])
    if [r["token"] for r in rows] != [f"token_id:{t}" for t in output]:
        raise ValueError("missing or misaligned logprob evidence")
    for row in rows:
        entries = [row, *row["top_logprobs"]]
        if not row["top_logprobs"] or any(
            not math.isfinite(entry["logprob"]) for entry in entries
        ):
            raise ValueError("invalid logprob evidence")
    return rows


def shared_prefix_logprobs(left: list[dict], right: list[dict]) -> dict:
    """Compare only distributions with identical preceding generated tokens."""
    rows = []
    for i, (a, b) in enumerate(zip(left, right)):
        at = {e["token"]: e["logprob"] for e in a["top_logprobs"]}
        bt = {e["token"]: e["logprob"] for e in b["top_logprobs"]}
        common = at.keys() & bt.keys()
        av, bv = sorted(at.values(), reverse=True), sorted(bt.values(), reverse=True)
        rows.append(
            {
                "index": i,
                "off": a,
                "on": b,
                "common_top_tokens": len(common),
                "max_common_logprob_delta": max(
                    (abs(at[t] - bt[t]) for t in common), default=None
                ),
                "off_top_margin": av[0] - av[1] if len(av) >= 2 else None,
                "on_top_margin": bv[0] - bv[1] if len(bv) >= 2 else None,
            }
        )
        if a["token"] != b["token"]:
            break
    return {
        "positions_compared": len(rows),
        "max_common_logprob_delta": max(
            (
                r["max_common_logprob_delta"]
                for r in rows
                if r["max_common_logprob_delta"] is not None
            ),
            default=None,
        ),
        "positions": rows,
        "scope": "Target top-k logprobs through first divergence only; not full-logit or recurrent-state equivalence proof.",
    }


def compare(off: Path, on: Path, *, draft_tokens: int = 1) -> dict:
    if draft_tokens not in (1, 2):
        raise ValueError("requires one or two draft tokens")
    left, lr = vision.audit(off)
    right, rr = vision.audit(on)
    for key in (
        "harness_sha256",
        "workload_sha256",
        "image_sha256",
        "runtime_identity",
    ):
        if left[key] != right[key]:
            raise ValueError(f"mismatched {key}")
    if without_mtp(left["actual_argv"], enabled=False) != without_mtp(
        right["actual_argv"], enabled=draft_tokens
    ):
        raise ValueError("runtime arguments differ beyond MTP")
    for key in left["actual_environment"].keys() | right["actual_environment"].keys():
        if key != "VLLM_CACHE_ROOT" and left["actual_environment"].get(key) != right[
            "actual_environment"
        ].get(key):
            raise ValueError(f"mismatched environment: {key}")
    rows = []
    for a, b in zip(lr, rr, strict=True):
        name = a["id"]
        if name != b["id"] or vision.load(off / f"{name}-request.json") != vision.load(
            on / f"{name}-request.json"
        ):
            raise ValueError("mismatched request")
        ap, at = tokens(off, name)
        bp, bt = tokens(on, name)
        if ap != bp:
            raise ValueError(f"mismatched processed prompt tokens: {name}")
        mismatch = next(
            (i for i, pair in enumerate(zip(at, bt)) if pair[0] != pair[1]), None
        )
        if mismatch is None and len(at) != len(bt):
            mismatch = min(len(at), len(bt))
        al, bl = logprobs(off, name, at), logprobs(on, name, bt)
        rows.append(
            {
                "id": name,
                "phase": a["phase"],
                "same_tokens": at == bt,
                "first_mismatch": mismatch,
                "off_tokens": at,
                "on_tokens": bt,
                "off_content": a["content"],
                "on_content": b["content"],
                "shared_prefix_logprobs": shared_prefix_logprobs(al, bl)
                if al is not None and bl is not None
                else None,
            }
        )
    return {
        "schema": "kvarn-mtp-token-comparison-v1",
        "comparer_sha256": runner.perf.sha256_file(Path(__file__)),
        "off": str(off),
        "on": str(on),
        "matched_inputs_verified": True,
        "all_tokens_equal": all(r["same_tokens"] for r in rows),
        "rows": rows,
        "scope": "Token comparison only; unequal outputs require investigation, and equal outputs alone do not prove cache lifecycle correctness.",
    }


def speculative_counters(path: Path) -> dict[str, float]:
    result = {}
    for line in path.read_text().splitlines():
        if not line.startswith("vllm:spec_decode_") or "_total{" not in line:
            continue
        metric, value = line.rsplit(" ", 1)
        name = metric.split("{", 1)[0]
        position = re.search(r'position="(\d+)"', metric)
        if position:
            name += "/position=" + position[1]
        if name in result:
            raise ValueError("performance screen requires one engine")
        result[name] = float(value)
    return result


def performance_summary(directory: Path) -> dict:
    manifest, rows = vision.audit(directory)
    if manifest.get("suite") != "mtp-performance-256-v1" or len(rows) != 12:
        raise ValueError("not the fixed MTP performance screen")
    groups = {}
    previous = None
    for row in rows:
        name = row["id"]
        prompt, output = tokens(directory, name)
        if len(output) != 256:
            raise ValueError("performance screen requires 256 generated tokens")
        if "text-4k" in name and len(prompt) != 4096:
            raise ValueError("incorrect text context length")
        before = speculative_counters(directory / f"{name}-metrics-before.txt")
        after = speculative_counters(directory / f"{name}-metrics-after.txt")
        if previous is not None and before != previous:
            raise ValueError("speculative counters changed between requests")
        previous = after
        delta = {
            k: after.get(k, 0) - before.get(k, 0) for k in before.keys() | after.keys()
        }
        if any(not math.isfinite(v) or v < 0 for v in delta.values()):
            raise ValueError("invalid or reset speculative counters")
        if row["phase"] == "warmup":
            continue
        key = name.rsplit("-", 1)[0]
        groups.setdefault(key, []).append(
            {
                "id": name,
                "prompt_tokens": len(prompt),
                "delivered_decode_tokens": len(output) - 1,
                "ttft_seconds": row["ttft_seconds"],
                "total_seconds": row["total_seconds"],
                "generation_seconds": row["total_seconds"] - row["ttft_seconds"],
                "decode_tokens_per_second": row["decode_tokens_per_second"],
                "speculative_counter_delta": delta,
            }
        )
    if previous != speculative_counters(directory / "metrics.txt"):
        raise ValueError("final speculative counters disagree with request boundaries")
    for group in groups.values():
        if len(group) != 3:
            raise ValueError("requires three measured repetitions per workload")
    return {
        "directory": str(directory),
        "speculative_config": manifest["speculative_config"],
        "workloads": {
            key: {
                "samples": group,
                "statistics": {
                    field: {
                        "median": statistics.median(row[field] for row in group),
                        "min": min(row[field] for row in group),
                        "max": max(row[field] for row in group),
                    }
                    for field in (
                        "ttft_seconds",
                        "total_seconds",
                        "generation_seconds",
                        "decode_tokens_per_second",
                    )
                },
            }
            for key, group in groups.items()
        },
        "memory": vision.memory_summary(directory),
        "scope": "Unprofiled SSE service timings; warmups excluded. Counter continuity and final publication checked; per-request output accounting is checked by the overhead model.",
    }


def overhead_model(baseline: list[dict], speculative: list[dict], drafts: int) -> dict:
    """Estimate extra cycle cost, not its cause, using pooled decode timing."""
    if drafts not in (1, 2) or not baseline or len(baseline) != len(speculative):
        raise ValueError("requires matched nonempty samples and one or two drafts")
    prefix = "vllm:spec_decode_"
    rounds = accepted = delivered = clipped = 0
    positions = [0.0] * drafts
    for base, row in zip(baseline, speculative, strict=True):
        if base["delivered_decode_tokens"] != row["delivered_decode_tokens"]:
            raise ValueError("mismatched delivered token count")
        for sample in (base, row):
            if (
                not math.isfinite(sample["generation_seconds"])
                or sample["generation_seconds"] <= 0
            ):
                raise ValueError("invalid generation duration")
        c = row["speculative_counter_delta"]
        n = c[prefix + "num_drafts_total"]
        a = c[prefix + "num_accepted_tokens_total"]
        p = [
            c[prefix + f"num_accepted_tokens_per_pos_total/position={i}"]
            for i in range(drafts)
        ]
        values = [
            n,
            a,
            *p,
            c[prefix + "num_draft_tokens_total"],
            row["delivered_decode_tokens"],
        ]
        if any(not math.isfinite(v) or v < 0 or int(v) != v for v in values) or n <= 0:
            raise ValueError("invalid speculative counts")
        excess = n + a - row["delivered_decode_tokens"]
        if (
            sum(p) != a
            or p != sorted(p, reverse=True)
            or p[0] > n
            or c[prefix + "num_draft_tokens_total"] != drafts * n
            or not 0 <= excess <= drafts
        ):
            raise ValueError(
                "counter/output accounting does not match fixed-draft capped decoding"
            )
        rounds += n
        accepted += a
        delivered += row["delivered_decode_tokens"]
        clipped += excess
        positions = [x + y for x, y in zip(positions, p, strict=True)]
    baseline_rate = sum(r["delivered_decode_tokens"] for r in baseline) / sum(
        r["generation_seconds"] for r in baseline
    )
    generation_seconds = sum(r["generation_seconds"] for r in speculative)
    actual_rate = delivered / generation_seconds
    raw_factor = 1 + accepted / rounds
    capped_factor = delivered / rounds
    ideal = baseline_rate * capped_factor
    return {
        "verification_steps": rounds,
        "accepted_draft_tokens": accepted,
        "acceptance_by_position_unconditional": [v / rounds for v in positions],
        "accepted_drafts_per_step": accepted / rounds,
        "delivered_decode_tokens": delivered,
        "output_cap_clipped_tokens": clipped,
        "raw_ideal_factor": raw_factor,
        "cap_adjusted_ideal_factor": capped_factor,
        "baseline_pooled_decode_tokens_per_second": baseline_rate,
        "actual_pooled_decode_tokens_per_second": actual_rate,
        "raw_ideal_decode_tokens_per_second": baseline_rate * raw_factor,
        "cap_adjusted_ideal_decode_tokens_per_second": ideal,
        "efficiency": actual_rate / ideal,
        "fraction_of_ideal_speed_lost": 1 - actual_rate / ideal,
        "effective_extra_ms_per_verification_step": 1000
        * (generation_seconds / rounds - 1 / baseline_rate),
        "scope": "Decode after first token, pooled across measured repetitions. Counterfactual assumes each verification cycle costs one baseline decode step and drafting is free. Extra cost includes all service, draft and verification overhead; it is not profiler attribution. Cap clipping is separated from overhead.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off", type=Path, required=True)
    parser.add_argument("--on", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, choices=(1, 2), default=1)
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    result = compare(args.off, args.on, draft_tokens=args.draft_tokens)
    if args.performance:
        off = performance_summary(args.off)
        on = performance_summary(args.on)
        result["performance"] = {"off": off, "on": on}
        result["overhead_model"] = {
            key: overhead_model(
                off["workloads"][key]["samples"], value["samples"], args.draft_tokens
            )
            for key, value in on["workloads"].items()
        }
    shutil.copyfile(Path(__file__), args.output.with_suffix(".source.py"))
    runner.perf.write_json_atomic(args.output, result)
    print(
        json.dumps(
            {
                "all_tokens_equal": result["all_tokens_equal"],
                "cases": len(result["rows"]),
            }
        )
    )
    return 0 if result["all_tokens_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
