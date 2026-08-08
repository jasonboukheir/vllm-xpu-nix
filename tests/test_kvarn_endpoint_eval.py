import importlib.util
import math
from argparse import Namespace
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "kvarn_endpoint_eval.py"
SPEC = importlib.util.spec_from_file_location("kvarn_endpoint_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_coarsened_divergence_identical_is_zero():
    logprobs = {"token_id:1": math.log(.6), "token_id:2": math.log(.3)}
    result = MODULE.coarsened_divergences(logprobs, logprobs)
    assert result is not None
    kl, js = result
    assert math.isclose(kl, 0, abs_tol=1e-14)
    assert math.isclose(js, 0, abs_tol=1e-14)


def test_coarsened_divergence_includes_residual_mass():
    ref = {"token_id:1": math.log(.6)}
    mode = {"token_id:1": math.log(.3)}
    result = MODULE.coarsened_divergences(ref, mode)
    assert result is not None
    kl, js = result
    expected = .6 * math.log(.6 / .3) + .4 * math.log(.4 / .7)
    assert math.isclose(kl, expected)
    assert 0 < js < kl


def test_differing_topk_support_does_not_report_false_zero_kl():
    # Collapsing endpoint-specific tokens into "other" makes these maximally
    # different top-1 predictions look identical. Their cross-probabilities
    # are unknown, so the evaluator must withhold KL/JS.
    ref = {"token_id:1": math.log(.99)}
    mode = {"token_id:2": math.log(.99)}
    assert MODULE.coarsened_divergences(ref, mode) is None


def test_aggregate_percentiles_and_agreement():
    rows = [{
        "checkpoint": checkpoint, "kl_ref_mode_nats": value,
        "js_nats": value / 2, "target_logprob_delta": -value,
        "top1_agreement": value < 2, "top5_agreement": True,
    } for checkpoint, value in enumerate((1., 2., 3., 4.), 1)]
    result = MODULE.aggregate(rows)
    assert result["count"] == 4
    assert result["divergence_count"] == 4
    assert result["kl_nats"]["p50"] == 2.5
    assert math.isclose(result["kl_nats"]["p95"], 3.85)
    assert result["top1_agreement"] == .25
    assert set(result["by_checkpoint"]) == {"1", "2", "3", "4"}
    assert result["kl_slope_per_log2_context"] > 0


def test_aggregate_omits_unavailable_divergences():
    row = {
        "checkpoint": 128, "kl_ref_mode_nats": None, "js_nats": None,
        "target_logprob_delta": .1, "top1_agreement": False,
        "top5_agreement": True,
    }
    result = MODULE.aggregate([row])
    assert result["count"] == 1
    assert result["divergence_count"] == 0
    assert result["kl_nats"]["mean"] is None


def test_pair_order_alternates_deterministically():
    assert MODULE._pair_order("alternating", 0) == ("bf16", "kvarn")
    assert MODULE._pair_order("alternating", 1) == ("kvarn", "bf16")
    assert MODULE._pair_order("alternating", 2) == ("bf16", "kvarn")
    assert MODULE._pair_order("bf16-first", 9) == ("bf16", "kvarn")
    assert MODULE._pair_order("kvarn-first", 0) == ("kvarn", "bf16")


def test_paired_checkpoint_uses_identical_prefix_and_records_order(monkeypatch):
    calls = []

    def checkpoint(endpoint, model, ids, top_k, timeout):
        calls.append((endpoint, model, list(ids), top_k, timeout))
        if endpoint == "kvarn":
            return -1.25, {"token_id:7": -0.1}
        return -1.0, {"token_id:7": -0.2}

    monkeypatch.setattr(MODULE, "_checkpoint", checkpoint)
    result = MODULE._paired_checkpoint(
        ("kvarn", "bf16"),
        {"bf16": "bf16", "kvarn": "kvarn"},
        {"bf16": "reference", "kvarn": "candidate"},
        [1, 2, 3],
        20,
        10,
    )

    assert [call[0] for call in calls] == ["kvarn", "bf16"]
    assert calls[0][2] == calls[1][2] == [1, 2, 3]
    assert result["request_order"] == ["kvarn", "bf16"]
    assert result["bf16"][0] == -1.0
    assert result["kvarn"][0] == -1.25


def test_load_samples_rejects_duplicate_ids(tmp_path):
    dataset = tmp_path / "samples.jsonl"
    dataset.write_text(
        '{"id":"same","token_ids":[1,2]}\n'
        '{"id":"same","token_ids":[3,4]}\n'
    )
    args = Namespace(token_ids=None, dataset=str(dataset))
    try:
        MODULE._load_samples(args)
    except ValueError as exc:
        assert "duplicate sample id" in str(exc)
    else:
        raise AssertionError("duplicate sample id was accepted")
