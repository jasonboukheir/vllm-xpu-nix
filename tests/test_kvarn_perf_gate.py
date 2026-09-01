from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.kvarn_perf_gate import GateError, compare


def _correctness(path: Path, candidate_id: str = "candidate-store-path") -> Path:
    gate_names = (
        "native_decode_short",
        "native_decode_262k",
        "b1_replay",
        "b1_restart",
        "cancel_reuse",
        "b4_isolation",
        "near_262k_reference_equivalence",
        "near_262k_restart",
    )
    gates = {}
    for name in gate_names:
        evidence = path.parent / f"{name}.json"
        evidence.write_text(json.dumps({"status": "passed"}), encoding="utf-8")
        gates[name] = {
            "status": "passed",
            "path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    document = {
        "status": "passed",
        "candidate_id": candidate_id,
        "native_dispatch_verified": True,
        "gates": gates,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _result(
    path: Path,
    *,
    arm: str,
    run_order: int,
    correctness_sha256: str,
    engine_log_sha256: str,
    output_throughput: float,
    request_throughput: float,
    ttft: float,
    itl: float,
    kv_cache_dtype: str | None = None,
    native_splits: int | None = None,
) -> Path:
    output_lens = [4, 4, 4, 4]
    input_lens = [127, 4095, 127, 4095]
    duration = sum(output_lens) / output_throughput
    document = {
        "completed": 4,
        "failed": 0,
        "output_throughput": output_throughput,
        "request_throughput": request_throughput,
        "total_token_throughput": (sum(input_lens) + sum(output_lens)) / duration,
        "duration": duration,
        "total_input_tokens": sum(input_lens),
        "total_output_tokens": sum(output_lens),
        "input_lens": input_lens,
        "output_lens": output_lens,
        "ttfts": [ttft] * 4,
        "itls": [[itl] * (length - 1) for length in output_lens],
        "backend": "openai",
        "model_id": "sunny-chat",
        "tokenizer_id": "model-repo",
        "num_prompts": 4,
        "request_rate": "inf",
        "max_concurrency": 4,
        "max_concurrent_requests": 4,
        "kvarn_candidate_id": "candidate-store-path",
        "kvarn_model_revision": "6b0622f4354481d5d04577d48ba0db844efc1330",
        "kvarn_service_profile": "qwen38-64k-b4-eager",
        "kvarn_workload_id": "fixed-b4",
        "kvarn_seed": "17",
        "kvarn_max_model_len": "65536",
        "kvarn_max_num_seqs": "4",
        "kvarn_enforce_eager": "1",
        "kvarn_prefix_caching": "0",
        "kvarn_mtp": "0",
        "kvarn_xpu_graph": "0",
        "kvarn_scheduler_peak_running": "4",
        "kvarn_correctness_sha256": correctness_sha256,
        "kvarn_arm": arm,
        "kvarn_kv_cache_dtype": kv_cache_dtype
        or ("auto" if arm == "reference" else "kvarn_k4v4_g128_compact"),
        "kvarn_native_xpu": "0" if arm == "reference" else "1",
        "kvarn_native_splits": str(
            native_splits
            if native_splits is not None
            else (1 if arm == "reference" else 16)
        ),
        "kvarn_run_order": str(run_order),
        "kvarn_run_uuid": f"run-{run_order}",
        "kvarn_run_started_at": f"2026-08-31T00:00:{run_order:02d}Z",
        "kvarn_engine_log_sha256": engine_log_sha256,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _log(path: Path, *, native: bool) -> Path:
    lines = ["INFO engine ready"]
    if native:
        lines.append("INFO Using the native Xe2 KVarN qlen=1 decoder")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _arms(
    tmp_path: Path,
    *,
    reference_value: float = 100.0,
    candidate_value: float = 97.0,
    reference_ttft: float = 0.100,
    candidate_ttft: float = 0.105,
    reference_itl: float = 0.050,
    candidate_itl: float = 0.052,
) -> tuple[list[Path], list[Path], list[Path], list[Path], Path]:
    correctness = _correctness(tmp_path / "correctness.json")
    digest = hashlib.sha256(correctness.read_bytes()).hexdigest()
    reference_orders = (1, 4, 5, 8)
    candidate_orders = (2, 3, 6, 7)
    reference_logs = [
        _log(tmp_path / f"reference-{index}.log", native=False) for index in range(4)
    ]
    candidate_logs = [
        _log(tmp_path / f"candidate-{index}.log", native=True) for index in range(4)
    ]
    references = [
        _result(
            tmp_path / f"reference-{index}.json",
            arm="reference",
            run_order=order,
            correctness_sha256=digest,
            engine_log_sha256=hashlib.sha256(
                reference_logs[index].read_bytes()
            ).hexdigest(),
            output_throughput=reference_value,
            request_throughput=reference_value / 4,
            ttft=reference_ttft,
            itl=reference_itl,
        )
        for index, order in enumerate(reference_orders)
    ]
    candidates = [
        _result(
            tmp_path / f"candidate-{index}.json",
            arm="candidate",
            run_order=order,
            correctness_sha256=digest,
            engine_log_sha256=hashlib.sha256(
                candidate_logs[index].read_bytes()
            ).hexdigest(),
            output_throughput=candidate_value,
            request_throughput=candidate_value / 4,
            ttft=candidate_ttft,
            itl=candidate_itl,
        )
        for index, order in enumerate(candidate_orders)
    ]
    return references, candidates, reference_logs, candidate_logs, correctness


def _compare(
    arms: tuple[list[Path], list[Path], list[Path], list[Path], Path],
    *,
    mode: str = "match",
    comparison_kind: str = "end-to-end",
) -> dict[str, object]:
    references, candidates, reference_logs, candidate_logs, correctness = arms
    return compare(
        references,
        candidates,
        reference_logs=reference_logs,
        candidate_logs=candidate_logs,
        correctness_path=correctness,
        comparison_kind=comparison_kind,
        mode=mode,
        min_throughput_ratio=0.95,
        min_request_decode_ratio=0.95,
        max_latency_ratio=1.10,
    )


def test_match_gate_uses_repeat_medians_and_both_perf_axes(tmp_path: Path) -> None:
    result = _compare(_arms(tmp_path))

    assert result["status"] == "passed"
    assert all(result["checks"].values())
    assert result["candidate_over_reference"]["output_throughput"] == pytest.approx(
        0.97
    )
    assert result["candidate_over_reference"][
        "median_request_decode_throughput"
    ] == pytest.approx(0.05 / 0.052)


def test_win_mode_rejects_a_latency_for_throughput_trade(tmp_path: Path) -> None:
    result = _compare(
        _arms(
            tmp_path,
            candidate_value=110.0,
            candidate_ttft=0.110,
            candidate_itl=0.055,
        ),
        mode="win",
    )

    assert result["status"] == "failed"
    assert result["checks"]["meaningful_throughput_gain"]
    assert not result["checks"]["median_request_decode_throughput"]
    assert not result["checks"]["p99_ttft"]
    assert not result["checks"]["p99_itl"]


def test_win_mode_requires_a_meaningful_gain(tmp_path: Path) -> None:
    result = _compare(
        _arms(
            tmp_path,
            candidate_value=100.1,
            candidate_ttft=0.0999,
            candidate_itl=0.0499,
        ),
        mode="win",
    )

    assert result["status"] == "failed"
    assert not result["checks"]["meaningful_throughput_gain"]


def test_gate_rejects_unmatched_workload_shapes(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    document = json.loads(arms[1][0].read_text(encoding="utf-8"))
    document["input_lens"] = [128, 4095, 127, 4095]
    document["total_input_tokens"] = sum(document["input_lens"])
    document["total_token_throughput"] = (
        document["total_input_tokens"] + document["total_output_tokens"]
    ) / document["duration"]
    arms[1][0].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="workload shape"):
        _compare(arms)


def test_gate_rejects_unbalanced_repeat_counts(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    arms[1].pop()
    arms[3].pop()

    with pytest.raises(GateError, match="at least four candidate repeats"):
        _compare(arms)


def test_kernel_comparison_requires_kvarn_in_both_arms(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    with pytest.raises(GateError, match="compact Kvarn in both arms"):
        _compare(arms, comparison_kind="kernel")


def test_kernel_comparison_passes_with_compact_kvarn_reference(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_kv_cache_dtype"] = "kvarn_k4v4_g128_compact"
        path.write_text(json.dumps(document), encoding="utf-8")

    assert _compare(arms, comparison_kind="kernel")["status"] == "passed"


def test_gate_requires_neutral_reference_and_supported_candidate_splits(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_splits"] = "16"
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="neutral split count 1"):
        _compare(arms)

    for path in arms[0]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_splits"] = "1"
        path.write_text(json.dumps(document), encoding="utf-8")
    for path in arms[1]:
        document = json.loads(path.read_text(encoding="utf-8"))
        document["kvarn_native_splits"] = "3"
        path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(GateError, match="supported native split count"):
        _compare(arms)


def test_gate_rejects_missing_native_dispatch(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    arms[3][0].write_text("INFO engine ready\n", encoding="utf-8")

    with pytest.raises(GateError, match="must contain native dispatch"):
        _compare(arms)


def test_gate_rejects_correctness_evidence_hash_mismatch(tmp_path: Path) -> None:
    arms = _arms(tmp_path)
    correctness = json.loads(arms[4].read_text(encoding="utf-8"))
    evidence = Path(correctness["gates"]["b4_isolation"]["path"])
    evidence.write_text(json.dumps({"status": "failed"}), encoding="utf-8")

    with pytest.raises(GateError, match="evidence SHA differs"):
        _compare(arms)


def test_gate_rejects_claimed_throughput_inconsistent_with_counts(
    tmp_path: Path,
) -> None:
    arms = _arms(tmp_path)
    document = json.loads(arms[1][0].read_text(encoding="utf-8"))
    document["output_throughput"] *= 2
    arms[1][0].write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(GateError, match="inconsistent with duration"):
        _compare(arms)
