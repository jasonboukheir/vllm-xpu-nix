import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "kv_kernel_ab.py"
SPEC = importlib.util.spec_from_file_location("kv_kernel_ab", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_all_bf16_control_skips_exact_full_attention_layers():
    reference = MODULE.default_plan()[0]
    assert [int(value) for value in reference["skip_layers"]] == list(range(3, 64, 4))
    assert "--enforce-eager" in reference["extra_args"]
    assert not any("speculative" in value for value in reference["extra_args"])


def test_trace_metrics_reports_first_token_divergence():
    def entry(token, a, b):
        return {"token": token, "top_logprobs": [
            {"token": token, "logprob": -0.1}, {"token": a, "logprob": -2.4},
            {"token": b, "logprob": -3.0},
        ]}

    reference = [entry("a", "b", "c"), entry("b", "a", "c")]
    candidate = [entry("a", "b", "c"), entry("c", "a", "b")]
    metrics = MODULE.trace_metrics(reference, candidate)
    assert metrics["first_divergence_token"] == 1
    assert metrics["top_token_agreement"] == 0.5
    assert metrics["mean_truncated_top5_kl"] > 0


def test_repetition_metrics_detects_decode_collapse():
    text = " ".join(["bad state loops forever"] * 20)
    assert MODULE.repetition_metrics(text)["repeated_fourgram_fraction"] > 0.5
