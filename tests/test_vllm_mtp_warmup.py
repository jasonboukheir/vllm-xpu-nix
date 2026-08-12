import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "vllm_mtp_warmup.py"
SPEC = importlib.util.spec_from_file_location("vllm_mtp_warmup", MODULE_PATH)
assert SPEC and SPEC.loader
warmup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(warmup)


def test_run_uses_real_b1_tile_and_ragged_b4_requests(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(warmup, "wait_for_model", lambda *args: None)
    monkeypatch.setattr(
        warmup,
        "completion",
        lambda base_url, model, prompts, max_tokens: observed.append(
            (base_url, model, [len(prompt) for prompt in prompts], max_tokens)
        ),
    )

    warmup.run("http://127.0.0.1:8000", "model", 10, 8)

    assert observed == [
        ("http://127.0.0.1:8000", "model", [134], 8),
        ("http://127.0.0.1:8000", "model", [134], 8),
        ("http://127.0.0.1:8000", "model", [4097], 3),
        ("http://127.0.0.1:8000", "model", [14, 15, 16, 17], 8),
    ]


def test_wait_for_model_retries_until_advertised(monkeypatch) -> None:
    responses = iter([OSError("not ready"), {"data": [{"id": "model"}]}])

    def fake_request(*args, **kwargs):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(warmup, "request", fake_request)
    monkeypatch.setattr(warmup.time, "sleep", lambda _: None)

    warmup.wait_for_model("http://127.0.0.1:8000", "model", 10)
