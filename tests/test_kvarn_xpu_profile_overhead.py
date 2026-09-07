import pytest

from scripts.kvarn_xpu_profile_overhead import summarize_overhead


def result(scale=1):
    return {
        "input_lens": [4096],
        "output_lens": [4],
        "num_prompts": 1,
        "max_concurrency": 1,
        "completed": 1,
        "failed": 0,
        "latencies": [4 * scale],
        "ttfts": [scale],
        "itls": [[scale, scale, scale]],
        "generated_texts": ["same"],
    }


def test_overhead_uses_bracketing_baseline_and_reports_drift():
    report = summarize_overhead(result(), result(3), result(2))
    row = report["deltas"]["median_request_latency_ms"]
    assert row["bracketing_baseline_mean"] == 6000
    assert row["profiled_over_baseline"] == 2
    assert row["after_over_before"] == 2
    assert report["acceptance_eligible"] is False


def test_output_changes_are_flagged_not_called_profiler_overhead():
    changed = {**result(), "generated_texts": ["different"]}
    report = summarize_overhead(result(), changed, result())
    assert not report["matched_generated_texts"]
    assert report["interpretation_status"].startswith("output_mismatch")


@pytest.mark.parametrize(
    "change",
    [
        {"input_lens": [8192]},
        {"failed": 1},
        {"ttfts": []},
        {"latencies": [float("nan")]},
        {"itls": [[]]},
    ],
)
def test_missing_or_unmatched_evidence_rejected(change):
    with pytest.raises(ValueError):
        summarize_overhead(result(), {**result(), **change}, result())
