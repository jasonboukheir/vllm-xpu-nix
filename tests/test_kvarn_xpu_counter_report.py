from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts import kvarn_xpu_counter_report as report
from scripts.kvarn_factory_run import sha256_file


def definitions(path: Path) -> None:
    metrics = [
        {"name": "GPU_BUSY", "units": "percent", "properties_status": 0},
        {"name": "XVE_ACTIVE", "units": "percent", "properties_status": 0},
        {"name": "XVE_STALL", "units": "percent", "properties_status": 0},
        {"name": "GPU_MEMORY_BYTE_READ", "units": "Byte", "properties_status": 0},
    ]
    path.write_text(
        json.dumps(
            {
                "returncode": 0,
                "capabilities": {
                    "devices": [
                        {
                            "is_target": True,
                            "metric_groups": [
                                {
                                    "name": "ComputeBasic",
                                    "sampling_type_flags": 1,
                                    "status": 0,
                                    "metrics_count_status": 0,
                                    "metrics_get_status": 0,
                                    "metrics": metrics,
                                }
                            ],
                        }
                    ]
                },
            }
        )
    )


def samples(path: Path, *, duplicate: bool = False, bad: str | None = None) -> None:
    fields = [
        "Kernel",
        "GlobalInstanceId",
        "GpuTime[ns]",
        "QueryBeginTime[ns]",
        "GPU_BUSY[%]",
        "XVE_ACTIVE[%]",
        "XVE_STALL[%]",
        "GPU_MEMORY_BYTE_READ[Byte]",
    ]
    rows = [
        {
            "Kernel": "gemm_kernel",
            "GlobalInstanceId": "1",
            "GpuTime[ns]": "10",
            "QueryBeginTime[ns]": "100",
            "GPU_BUSY[%]": "50",
            "XVE_ACTIVE[%]": "40",
            "XVE_STALL[%]": "5",
            "GPU_MEMORY_BYTE_READ[Byte]": "12",
        },
        {
            "Kernel": "gemm_kernel",
            "GlobalInstanceId": "2",
            "GpuTime[ns]": "10",
            "QueryBeginTime[ns]": "100",
            "GPU_BUSY[%]": "60",
            "XVE_ACTIVE[%]": "40",
            "XVE_STALL[%]": "5",
            "GPU_MEMORY_BYTE_READ[Byte]": "13",
        },
    ]
    if duplicate:
        rows[1]["GlobalInstanceId"] = "1"
    if bad == "nonfinite":
        rows[0]["GPU_BUSY[%]"] = "nan"
    if bad == "duration":
        rows[0]["GpuTime[ns]"] = "0"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_read_samples_rejects_empty_nonfinite_duplicate_and_bad_duration(
    tmp_path: Path,
):
    empty = tmp_path / "empty.csv"
    empty.write_text("Kernel,GlobalInstanceId\n")
    with pytest.raises(ValueError, match="no actual"):
        report.read_samples(empty)
    for kind, message in (("nonfinite", "nonfinite"), ("duration", "nonpositive")):
        path = tmp_path / f"{kind}.csv"
        samples(path, bad=kind)
        with pytest.raises(ValueError, match=message):
            report.read_samples(path)
    path = tmp_path / "duplicate.csv"
    samples(path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate"):
        report.read_samples(path)


def test_summarize_units_identity_unsampled_and_shared_timestamp(tmp_path: Path):
    path = tmp_path / "samples.csv"
    samples(path)
    rows = report.read_samples(path)
    defs = {
        name: {"units": unit}
        for name, unit in (
            ("GPU_BUSY", "percent"),
            ("XVE_ACTIVE", "percent"),
            ("XVE_STALL", "percent"),
            ("GPU_MEMORY_BYTE_READ", "Byte"),
        )
    }
    events = [
        {"ph": "X", "cat": "gpu_op", "name": "gemm_kernel", "args": {"id": "1"}},
        {"ph": "X", "cat": "gpu_op", "name": "gemm_kernel", "args": {"id": "2"}},
        {"ph": "X", "cat": "gpu_op", "name": "short", "args": {"id": "3"}},
    ]
    result = report.summarize(rows, events, defs)
    assert result["timestamps_assigned_to_multiple_kernels"] == 1
    assert result["unsampled_instance_ids"] == ["3"]
    assert (
        result["kernels"][0]["metrics"]["GPU_BUSY"]["duration_weighted_sample_mean"]
        == 55
    )
    bad_defs = dict(defs)
    bad_defs["GPU_BUSY"] = {"units": "GBpS"}
    with pytest.raises(ValueError, match="unit mismatch"):
        report.summarize(rows, events, bad_defs)
    rows[0]["Kernel"] = "other"
    with pytest.raises(ValueError, match="identity mismatch"):
        report.summarize(rows, events, defs)


def make_capture(
    tmp_path: Path,
    monkeypatch,
    *,
    mismatch: bool = False,
    bracket_mismatch: bool = False,
    auxiliary: str | None = None,
):
    capture = tmp_path / "capture"
    (capture / "collector" / "run" / "metrics").mkdir(parents=True)
    definitions_path = tmp_path / "definitions.json"
    definitions(definitions_path)
    samples_path = capture / "collector" / "run" / "metrics" / "samples.csv"
    samples(samples_path)
    trace_path = capture / "collector" / "run" / "chrome_trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "gpu_op",
                        "name": "gemm_kernel",
                        "args": {"id": "1"},
                    },
                    {
                        "ph": "X",
                        "cat": "gpu_op",
                        "name": "gemm_kernel",
                        "args": {"id": "2"},
                    },
                ]
            }
        )
    )
    if auxiliary is not None:
        aux_dir = capture / "collector" / "aux"
        aux_dir.mkdir()
        (aux_dir / "chrome_trace.json").write_text(auxiliary)
    workload = {
        "workload": "gemm",
        "contract": "test",
        "iterations": 2,
        "warmup": 1,
        "seed": 17,
        "hardware": "cpu",
        "sinkhorn_module": "none",
        "workload_source_sha256": "source",
        "torch_version": "test",
        "correctness_passed": True,
    }
    if mismatch:
        workload["iterations"] = 3
    (capture / "profiled.json").write_text(json.dumps(workload))
    for label in ("before", "after"):
        bracket = dict(workload)
        if bracket_mismatch and label == "after":
            bracket["seed"] = 18
        (capture / f"{label}.json").write_text(json.dumps(bracket))
    tool = tmp_path / "unitrace"
    tool.write_bytes(b"unitrace")
    injection = tmp_path / "libunitrace_tool.so"
    injection.write_bytes(b"tool")
    files = {
        str(p.relative_to(capture)): sha256_file(p)
        for p in (
            samples_path,
            trace_path,
            capture / "profiled.json",
            capture / "before.json",
            capture / "after.json",
        )
    }
    (capture / "source-snapshot").mkdir()
    (capture / "capture.json").write_text(
        json.dumps(
            {
                "capture_completed": True,
                "files": files,
                "group": "ComputeBasic",
                "overhead": {},
                "commands": [{"returncode": 0}],
                "unitrace": {"path": str(tool), "sha256": sha256_file(tool)},
                "injection_library": {
                    "path": str(injection),
                    "sha256": sha256_file(injection),
                },
                "environment": {},
            }
        )
    )
    monkeypatch.setattr(report, "verify_sources", lambda *args, **kwargs: None)
    return capture, definitions_path


def test_validate_capture_rejects_full_envelope_mismatch(tmp_path, monkeypatch):
    capture, defs = make_capture(tmp_path, monkeypatch, mismatch=True)
    with pytest.raises(ValueError, match="kernel envelope"):
        report.validate_capture(capture, defs)


def test_validate_capture_rejects_mutated_hashed_input(tmp_path, monkeypatch):
    capture, defs = make_capture(tmp_path, monkeypatch)
    (capture / "profiled.json").write_text("mutated")
    with pytest.raises(ValueError, match="capture input changed"):
        report.validate_capture(capture, defs)


def test_validate_capture_successfully_reports_fixture(tmp_path, monkeypatch):
    capture, defs = make_capture(tmp_path, monkeypatch)
    result = report.validate_capture(capture, defs)
    assert result["collection_validated"] is True
    assert result["sample_rows"] == 2
    assert result["traced_instances"] == 2


def test_validate_capture_rejects_changed_tool_and_bracket_identity(
    tmp_path, monkeypatch
):
    capture, defs = make_capture(tmp_path, monkeypatch)
    manifest = json.loads((capture / "capture.json").read_text())
    Path(manifest["unitrace"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="counter tool changed"):
        report.validate_capture(capture, defs)

    capture, defs = make_capture(
        tmp_path / "bracket", monkeypatch, bracket_mismatch=True
    )
    with pytest.raises(ValueError, match="counter bracket mismatch"):
        report.validate_capture(capture, defs)


def test_validate_capture_accepts_metadata_only_auxiliary_trace(tmp_path, monkeypatch):
    capture, defs = make_capture(
        tmp_path, monkeypatch, auxiliary='{"traceEvents":[{"ph":"M"}'
    )
    result = report.validate_capture(capture, defs)
    assert result["excluded_auxiliary_traces"][0]["reason"].startswith(
        "truncated metadata-only"
    )


def test_validate_capture_rejects_truncated_operational_trace(tmp_path, monkeypatch):
    capture, defs = make_capture(
        tmp_path, monkeypatch, auxiliary='{"traceEvents":[{"ph":"X"}'
    )
    with pytest.raises(ValueError, match="malformed operational trace"):
        report.validate_capture(capture, defs)
