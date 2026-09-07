"""Descriptive off/on/off profiler perturbation checks, never parity evidence."""

from __future__ import annotations

import math
import statistics


def summarize_overhead(before: dict, profiled: dict, after: dict) -> dict:
    documents = {
        "unprofiled_before": before,
        "profiled": profiled,
        "unprofiled_after": after,
    }
    for key in ("input_lens", "output_lens", "num_prompts", "max_concurrency"):
        if key not in before or any(
            document.get(key) != before[key] for document in documents.values()
        ):
            raise ValueError(f"unmatched overhead workload: {key}")
    measurements = {}
    for label, document in documents.items():
        if (
            document.get("failed") != 0
            or document.get("completed") != before["num_prompts"]
        ):
            raise ValueError(f"incomplete overhead benchmark: {label}")
        arrays = [
            document.get(key, [])
            for key in ("latencies", "ttfts", "itls", "generated_texts")
        ]
        if any(len(array) != before["num_prompts"] for array in arrays):
            raise ValueError(f"missing per-request overhead evidence: {label}")
        latencies, ttfts, nested_itls, _ = arrays
        itls = [value for request in nested_itls for value in request]
        if not itls or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in [*latencies, *ttfts, *itls]
        ):
            raise ValueError(f"invalid overhead timings: {label}")
        measurements[label] = {
            "median_request_latency_ms": statistics.median(latencies) * 1000,
            "median_ttft_ms": statistics.median(ttfts) * 1000,
            "mean_inter_chunk_latency_ms": statistics.mean(itls) * 1000,
            "median_inter_chunk_latency_ms": statistics.median(itls) * 1000,
            "max_inter_chunk_latency_ms": max(itls) * 1000,
        }
    deltas = {}
    for metric, active in measurements["profiled"].items():
        low = measurements["unprofiled_before"][metric]
        high = measurements["unprofiled_after"][metric]
        baseline = (low + high) / 2
        if baseline <= 0 or low <= 0:
            raise ValueError(f"nonpositive overhead baseline: {metric}")
        deltas[metric] = {
            "bracketing_baseline_mean": baseline,
            "profiled_minus_baseline": active - baseline,
            "profiled_over_baseline": active / baseline,
            "after_over_before": high / low,
        }
    same_text = (
        before["generated_texts"]
        == profiled["generated_texts"]
        == after["generated_texts"]
    )
    return {
        "artifact_kind": "profiler_overhead_diagnostic",
        "schema_version": 1,
        "diagnostic_only": True,
        "acceptance_eligible": False,
        "order": list(documents),
        "same_service_process": True,
        "matched_generated_texts": same_text,
        "interpretation_status": "matched_descriptive_comparison"
        if same_text
        else "output_mismatch_do_not_attribute_to_profiler_only",
        "measurements": measurements,
        "deltas": deltas,
        "limitations": [
            "One off/on/off bracket, not a statistical profiler-overhead estimate.",
            "The baseline retains profiler configuration but collection is inactive; it is not a profiler-free binary.",
            "Each benchmark has the same unprofiled warmup and deterministic workload configuration.",
            "Client request latency includes in-request trace stopping/export, but excludes profile activation before requests.",
            "Median inter-chunk latency mixes captured and uncaptured decode; it does not isolate active capture overhead.",
            "Compare before/after drift and output matching before interpreting ratios.",
        ],
    }
