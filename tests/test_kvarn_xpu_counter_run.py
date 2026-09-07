from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import kvarn_xpu_counter_run as counter


def invoke(monkeypatch, tmp_path: Path, *, mismatch: bool = False, **extra):
    unitrace_root = tmp_path / "unitrace"
    unitrace_root.joinpath("bin").mkdir(parents=True)
    unitrace_root.joinpath("lib").mkdir()
    unitrace = unitrace_root / "bin" / "unitrace"
    unitrace.write_bytes(b"unitrace")
    unitrace_root.joinpath("lib", "libunitrace_tool.so").write_bytes(b"tool")
    output = tmp_path / "capture"
    argv = [
        "counter_run",
        "--unitrace",
        str(unitrace),
        "--workload",
        "gemm",
        "--output-dir",
        str(output),
    ]
    for key, value in extra.items():
        argv.extend([f"--{key.replace('_', '-')}", str(value)])
    monkeypatch.setattr(counter.sys, "argv", argv)
    monkeypatch.setattr(counter, "ensure_durable_output", lambda path, **_: path)
    monkeypatch.setattr(counter, "snapshot_sources", lambda *args: None)
    monkeypatch.setattr(counter, "verify_sources", lambda *args, **kwargs: None)

    def run(command, **kwargs):
        if "--output" in command:
            path = Path(command[command.index("--output") + 1])
            value = {
                "workload": "gemm",
                "contract": "test",
                "iterations": 4,
                "warmup": 1,
                "seed": 17,
                "hardware": "cpu",
                "sinkhorn_module": "none",
                "workload_source_sha256": "source",
                "torch_version": "test",
                "correctness_passed": True,
                "window_elapsed_ms": 20.0,
            }
            if mismatch and path.name == "profiled.json":
                value["seed"] = 18
            path.write_text(json.dumps(value))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(counter.subprocess, "run", run)
    return output


def test_failed_command_retains_noncompleted_manifest(monkeypatch, tmp_path):
    output = invoke(monkeypatch, tmp_path)
    calls = {"count": 0}

    def fail(command, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(returncode=1 if calls["count"] == 2 else 0)

    monkeypatch.setattr(counter.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="metric-list failed"):
        counter.main()
    record = json.loads((output / "capture.json").read_text())
    assert record.get("capture_completed", False) is False
    assert len(record["commands"]) == 2


def test_successful_bracket_records_overhead_and_hashes(monkeypatch, tmp_path):
    output = invoke(monkeypatch, tmp_path)
    counter.main()
    record = json.loads((output / "capture.json").read_text())
    assert record["capture_completed"] is True
    assert record["injection_library"]["path"].endswith("lib/libunitrace_tool.so")
    assert record["injection_library"]["sha256"]
    assert record["overhead"]["profiled_over_baseline"] == 1.0
    assert record["files"]["before.json"]
    assert len(record["commands"]) == 5


def test_mismatched_workload_is_rejected(monkeypatch, tmp_path):
    output = invoke(monkeypatch, tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="bracket identity mismatch: seed"):
        counter.main()
    record = json.loads((output / "capture.json").read_text())
    assert "capture_completed" not in record


@pytest.mark.parametrize("option", ["iterations", "sampling_interval"])
def test_positive_argument_validation(monkeypatch, tmp_path, option):
    with pytest.raises(SystemExit):
        invoke(monkeypatch, tmp_path, **{option: 0})
        counter.main()
