"""CPU checks for the dense-policy benchmark's experimental design."""

import collections

import pytest

from scripts.xpu_w4a16_dense_bench import (
    ARMS,
    block_order,
    summarize_blocks,
    window_stable,
)


def test_schedule_balances_position_and_cancels_linear_drift():
    positions = collections.defaultdict(list)
    for index in range(len(ARMS) * 2):
        order = block_order(index)
        assert order == order[::-1]
        assert collections.Counter(order) == {arm: 2 for arm in ARMS}
        for position, arm in enumerate(order):
            positions[arm].append(position)
    assert all(
        collections.Counter(p) == {i: 2 for i in range(10)} for p in positions.values()
    )
    assert block_order(0, True) != block_order(0)


def windows(values, seconds=0.1):
    return [
        {"seconds": seconds, "median_ms": dict.fromkeys(ARMS, value)}
        for value in values
    ]


def test_warmup_requires_duration_three_windows_and_every_arm_stable():
    stable = windows([5, 5.01, 5.02])
    assert window_stable(stable, 1)
    assert not window_stable(stable, 0.99)
    assert not window_stable(stable[:2], 1)
    assert not window_stable(windows([5, 5, 5], seconds=0.09), 1)
    stable[-1]["median_ms"][ARMS[-1]] = 5.2
    assert not window_stable(stable, 1)


def test_warmup_can_stabilize_after_earlier_drift():
    assert window_stable(windows([6, 7, 8, 8.01, 8.02]), 2)
    assert not window_stable(windows([6, 7, 8, 8.2, 8.4]), 2)


def blocks_with_latency(factor=1.0, drift=0):
    result = []
    for index in range(30):
        samples = []
        for position, arm in enumerate(block_order(index)):
            base = 10 + index * drift
            value = base * (1 if arm == "onednn" else factor)
            samples.append({"arm": arm, "wall_ms": value, "host_ms": 0.1})
        result.append({"index": index, "samples": samples})
    return result


def test_summary_reports_paired_gain_and_flags_drift_separately():
    summary = summarize_blocks(blocks_with_latency(0.8))
    assert summary["original_cached"]["median_latency_change_percent"] == pytest.approx(
        -20
    )
    assert summary["original_cached"]["block_bootstrap_95_percent"] == pytest.approx(
        [-20, -20]
    )
    assert not summary["original_cached"]["drift_flag"]
    drifting = summarize_blocks(blocks_with_latency(0.8, drift=0.1))
    assert drifting["original_cached"][
        "median_latency_change_percent"
    ] == pytest.approx(-20)
    assert drifting["original_cached"]["drift_flag"]
    assert drifting["onednn"]["drift_flag"]


def test_summary_rejects_missing_or_unbalanced_samples():
    blocks = blocks_with_latency()
    blocks[0]["samples"].pop()
    with pytest.raises(ValueError, match="two calls"):
        summarize_blocks(blocks)
    with pytest.raises(ValueError, match="three complete"):
        summarize_blocks(blocks_with_latency()[:2])


def test_original_library_subset_keeps_paired_summary_and_stability_rules():
    arms = ARMS[:3]
    blocks = blocks_with_latency(0.9)
    for block in blocks:
        block["samples"] = [s for s in block["samples"] if s["arm"] in arms]
    summary = summarize_blocks(blocks, arms)
    assert set(summary) == set(arms)
    assert summary["original_cached"]["median_latency_change_percent"] == pytest.approx(
        -10
    )
    warmup = windows([5, 5, 5])
    for window in warmup:
        window["median_ms"] = {arm: window["median_ms"][arm] for arm in arms}
    assert window_stable(warmup, 1, arms=arms)
