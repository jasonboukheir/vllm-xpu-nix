import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "kvarn_perf_gate.py"
SPEC = importlib.util.spec_from_file_location("kvarn_perf_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_result(path: Path, throughput: float, tpot: float) -> None:
    path.write_text(
        json.dumps({
            "completed": 32,
            "failed": 0,
            "output_throughput": throughput,
            "mean_tpot_ms": tpot,
        }),
        encoding="utf-8",
    )


def test_candidate_passes_both_gates(tmp_path: Path) -> None:
    baseline = tmp_path / "bf16.json"
    candidate = tmp_path / "native.json"
    _write_result(baseline, 20.0, 100.0)
    _write_result(candidate, 19.1, 109.0)

    report = MODULE.compare_results(baseline, candidate)

    assert report["passed"] is True
    assert report["throughput"]["ratio"] == pytest.approx(0.955)
    assert report["tpot"]["ratio"] == pytest.approx(1.09)


def test_candidate_fails_if_either_gate_misses(tmp_path: Path) -> None:
    baseline = tmp_path / "bf16.json"
    slow_throughput = tmp_path / "slow-throughput.json"
    slow_tpot = tmp_path / "slow-tpot.json"
    _write_result(baseline, 20.0, 100.0)
    _write_result(slow_throughput, 18.9, 100.0)
    _write_result(slow_tpot, 20.0, 110.1)

    assert MODULE.compare_results(baseline, slow_throughput)["passed"] is False
    assert MODULE.compare_results(baseline, slow_tpot)["passed"] is False


def test_failed_requests_make_result_ineligible(tmp_path: Path) -> None:
    baseline = tmp_path / "bf16.json"
    candidate = tmp_path / "native.json"
    _write_result(baseline, 20.0, 100.0)
    _write_result(candidate, 20.0, 100.0)
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["failed"] = 1
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    report = MODULE.compare_results(baseline, candidate)

    assert report["passed"] is False
    assert report["eligible"] is False
