import importlib.util
import math
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "kvarn_endpoint_eval.py"
SPEC = importlib.util.spec_from_file_location("kvarn_endpoint_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_coarsened_divergence_identical_is_zero():
    logprobs = {"token_id:1": math.log(.6), "token_id:2": math.log(.3)}
    kl, js = MODULE.coarsened_divergences(logprobs, logprobs)
    assert math.isclose(kl, 0, abs_tol=1e-14)
    assert math.isclose(js, 0, abs_tol=1e-14)


def test_coarsened_divergence_includes_residual_mass():
    ref = {"token_id:1": math.log(.6)}
    mode = {"token_id:1": math.log(.3)}
    kl, js = MODULE.coarsened_divergences(ref, mode)
    expected = .6 * math.log(.6 / .3) + .4 * math.log(.4 / .7)
    assert math.isclose(kl, expected)
    assert 0 < js < kl


def test_aggregate_percentiles_and_agreement():
    rows = [{
        "checkpoint": checkpoint, "kl_ref_mode_nats": value,
        "js_nats": value / 2, "target_logprob_delta": -value,
        "top1_agreement": value < 2, "top5_agreement": True,
    } for checkpoint, value in enumerate((1., 2., 3., 4.), 1)]
    result = MODULE.aggregate(rows)
    assert result["count"] == 4
    assert result["kl_nats"]["p50"] == 2.5
    assert math.isclose(result["kl_nats"]["p95"], 3.85)
    assert result["top1_agreement"] == .25
    assert set(result["by_checkpoint"]) == {"1", "2", "3", "4"}
    assert result["kl_slope_per_log2_context"] > 0
