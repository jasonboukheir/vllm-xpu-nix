#!/usr/bin/env python3
"""Paired, teacher-forced KV-cache drift evaluation through vLLM endpoints.

Both endpoints receive identical token prefixes.  Each JSONL input row must have
``token_ids`` (or ``prompt_token_ids`` plus ``continuation_token_ids``) and may
have an ``id``.  The completions API's echoed prompt logprobs provide the
teacher-forced target probability at each requested context checkpoint.

KL/JS are emitted only when both endpoints return the same token support. With
truncated top-k output, differing supports have unknown cross-probabilities and
cannot yield a defensible KL/JS; the other drift signals remain available.

This endpoint protocol replays each prefix as a fresh prompt. It is a prefill
and quantized-store smoke test, not the paper's accumulated pseudo-decode test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def coarsened_divergences(
    ref_logprobs: dict[str, float], mode_logprobs: dict[str, float]
) -> tuple[float, float] | None:
    """Return KL/JS on common support plus residual, or None if unknowable.

    If top-k supports differ, the probability assigned by one endpoint to a
    token returned only by the other is unknown. Folding those tokens into an
    endpoint-specific residual can make completely disjoint predictions report
    KL=0, so fail closed instead of manufacturing a reassuring metric.
    """
    if set(ref_logprobs) != set(mode_logprobs):
        return None
    p_known = {k: math.exp(value) for k, value in ref_logprobs.items()}
    q_known = {k: math.exp(value) for k, value in mode_logprobs.items()}
    p_known["__other__"] = max(0.0, 1.0 - sum(p_known.values()))
    q_known["__other__"] = max(0.0, 1.0 - sum(q_known.values()))

    def norm(d: dict[str, float]) -> dict[str, float]:
        total = sum(d.values())
        if total <= 0:
            raise ValueError("logprob distribution has no positive mass")
        return {k: v / total for k, v in d.items()}

    p, q = norm(p_known), norm(q_known)
    eps = 1e-300
    kl = sum(v * math.log(v / max(q[k], eps)) for k, v in p.items() if v)
    midpoint = {k: (p[k] + q[k]) / 2 for k in p}
    js = 0.5 * sum(v * math.log(v / midpoint[k]) for k, v in p.items() if v)
    js += 0.5 * sum(v * math.log(v / midpoint[k]) for k, v in q.items() if v)
    return kl, js


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    kls = [r["kl_ref_mode_nats"] for r in rows if r["kl_ref_mode_nats"] is not None]
    js = [r["js_nats"] for r in rows if r["js_nats"] is not None]
    deltas = [r["target_logprob_delta"] for r in rows]
    top1 = [float(r["top1_agreement"]) for r in rows]
    top5 = [float(r["top5_agreement"]) for r in rows]
    by_checkpoint: dict[str, dict[str, Any]] = {}
    for checkpoint in sorted({r["checkpoint"] for r in rows}):
        subset = [r for r in rows if r["checkpoint"] == checkpoint]
        by_checkpoint[str(checkpoint)] = aggregate(subset) | {"by_checkpoint": {}} if len(subset) < len(rows) else {}
    slope = None
    slope_rows = [r for r in rows if r["kl_ref_mode_nats"] is not None]
    if len(slope_rows) > 1:
        xs = [math.log2(r["checkpoint"]) for r in slope_rows]
        x_mean, y_mean = statistics.fmean(xs), statistics.fmean(kls)
        denominator = sum((x - x_mean) ** 2 for x in xs)
        if denominator:
            slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, kls)) / denominator
    return {
        "count": len(rows),
        "divergence_count": len(kls),
        "kl_nats": {"mean": statistics.fmean(kls) if kls else None,
                    "p50": _percentile(kls, .50), "p95": _percentile(kls, .95),
                    "p99": _percentile(kls, .99)},
        "js_nats": {"mean": statistics.fmean(js) if js else None,
                    "p50": _percentile(js, .50), "p95": _percentile(js, .95),
                    "p99": _percentile(js, .99)},
        "target_logprob_delta": {"mean": statistics.fmean(deltas) if deltas else None,
                                 "p50": _percentile(deltas, .50),
                                 "p95_abs": _percentile([abs(x) for x in deltas], .95)},
        "top1_agreement": statistics.fmean(top1) if top1 else None,
        "top5_agreement": statistics.fmean(top5) if top5 else None,
        "kl_slope_per_log2_context": slope,
        "by_checkpoint": by_checkpoint,
    }


def _get_json(url: str, timeout: float) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def _post_json(url: str, payload: dict[str, Any], timeout: float) -> Any:
    request = urllib.request.Request(url, json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{url}: HTTP {exc.code}: {exc.read().decode()}") from exc


def _token_key(token: str) -> str:
    # --return-tokens-as-token-ids makes vLLM return token_id:<integer>.
    if not token.startswith("token_id:"):
        raise ValueError(f"endpoint returned non-token-id logprob key {token!r}")
    return token


def _checkpoint(endpoint: str, model: str, ids: list[int], top_k: int,
                timeout: float) -> tuple[float, dict[str, float]]:
    result = _post_json(endpoint.rstrip("/") + "/v1/completions", {
        "model": model, "prompt": ids, "max_tokens": 0, "echo": True,
        "logprobs": top_k, "temperature": 0, "seed": 0,
        "return_tokens_as_token_ids": True,
    }, timeout)
    logprobs = result["choices"][0]["logprobs"]
    target_lp = logprobs["token_logprobs"][-1]
    top = logprobs["top_logprobs"][-1]
    if target_lp is None or top is None:
        raise ValueError("endpoint omitted the final prompt token logprobs")
    return float(target_lp), {_token_key(k): float(v) for k, v in top.items()}


def _load_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.token_ids:
        ids = [int(x) for x in args.token_ids.split(",") if x.strip()]
        return [{"id": "cli", "token_ids": ids}]
    samples = []
    with Path(args.dataset).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            ids = item.get("token_ids")
            if ids is None:
                ids = item.get("prompt_token_ids", []) + item.get("continuation_token_ids", [])
            if len(ids) < 2 or not all(isinstance(x, int) for x in ids):
                raise ValueError(f"line {line_no}: need at least two integer token IDs")
            samples.append({"id": str(item.get("id", line_no)), "token_ids": ids})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--token-ids", help="comma-separated exact teacher-forced tokens")
    source.add_argument("--dataset", help="JSONL containing exact token sequences")
    parser.add_argument("--bf16-url", required=True)
    parser.add_argument("--kvarn-url", required=True)
    parser.add_argument("--model", help="shared served model alias (overrides endpoint defaults)")
    parser.add_argument("--bf16-model", help="BF16 served model; default: endpoint's first model")
    parser.add_argument("--kvarn-model", help="KVarN served model; default: endpoint's first model")
    parser.add_argument("--checkpoints", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", required=True, help="per-checkpoint JSONL path")
    parser.add_argument("--summary", required=True, help="summary JSON path")
    parser.add_argument("--metadata-json", help="engine/build metadata merged into run metadata")
    args = parser.parse_args()

    samples = _load_samples(args)
    checkpoints = sorted({int(x) for x in args.checkpoints.split(",") if x})
    bf16_models = _get_json(args.bf16_url.rstrip("/") + "/v1/models", args.timeout)
    kvarn_models = _get_json(args.kvarn_url.rstrip("/") + "/v1/models", args.timeout)
    bf16_model = args.model or args.bf16_model or bf16_models["data"][0]["id"]
    kvarn_model = args.model or args.kvarn_model or kvarn_models["data"][0]["id"]
    extra_metadata = json.loads(Path(args.metadata_json).read_text()) if args.metadata_json else {}
    metadata = {
        "schema_version": 1, "created_unix": time.time(), "python": sys.version,
        "platform": platform.platform(), "seed": 0, "temperature": 0,
        "top_logprobs": args.top_logprobs, "checkpoints": checkpoints,
        "bf16_model": bf16_model, "kvarn_model": kvarn_model,
        "bf16_url": args.bf16_url, "kvarn_url": args.kvarn_url,
        "bf16_models_response": bf16_models, "kvarn_models_response": kvarn_models,
        "input_sha256": _json_hash(samples),
        "evaluation_path": "fresh-prompt-prefill-proxy",
        "metric_distribution": "equal-returned-support-plus-residual",
        "extra": extra_metadata,
    }
    # Stable across reruns of the same inputs/configuration; timestamps and
    # endpoint model responses remain in metadata but do not perturb identity.
    run_id = _json_hash({
        "schema_version": metadata["schema_version"], "seed": 0,
        "top_logprobs": args.top_logprobs, "checkpoints": checkpoints,
        "bf16_model": bf16_model, "kvarn_model": kvarn_model,
        "input_sha256": metadata["input_sha256"], "extra": extra_metadata,
    })
    rows = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            ids = sample["token_ids"]
            for checkpoint in checkpoints:
                # Context length c predicts token c; include it to obtain its
                # teacher-forced echoed logprob.
                if checkpoint >= len(ids):
                    continue
                prefix = ids[:checkpoint + 1]
                ref_lp, ref_top = _checkpoint(args.bf16_url, bf16_model, prefix,
                                               args.top_logprobs, args.timeout)
                mode_lp, mode_top = _checkpoint(args.kvarn_url, kvarn_model, prefix,
                                                 args.top_logprobs, args.timeout)
                divergences = coarsened_divergences(ref_top, mode_top)
                kl, js = divergences if divergences is not None else (None, None)
                ref_order = sorted(ref_top, key=ref_top.get, reverse=True)
                mode_order = sorted(mode_top, key=mode_top.get, reverse=True)
                row = {
                    "run_id": run_id, "sample_id": sample["id"],
                    "sample_sha256": _json_hash(ids), "checkpoint": checkpoint,
                    "target_token_id": ids[checkpoint], "bf16_target_logprob": ref_lp,
                    "kvarn_target_logprob": mode_lp,
                    "target_logprob_delta": mode_lp - ref_lp,
                    "kl_ref_mode_nats": kl, "js_nats": js,
                    "divergence_support_complete": divergences is not None,
                    "top1_agreement": ref_order[:1] == mode_order[:1],
                    # Treat BF16's top-1 as the reference prediction and ask
                    # whether KVarN retains it in its top five.
                    "top5_agreement": bool(ref_order and ref_order[0] in mode_order[:5]),
                    "bf16_top_logprobs": ref_top, "kvarn_top_logprobs": mode_top,
                }
                rows.append(row)
                handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
    summary = {"run_id": run_id, "metadata": metadata, "metrics": aggregate(rows)}
    Path(args.summary).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
