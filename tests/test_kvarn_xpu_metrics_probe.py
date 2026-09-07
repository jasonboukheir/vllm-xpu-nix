import json
import os
import subprocess
import sys

import pytest

from scripts import kvarn_xpu_metrics_probe as probe


@pytest.mark.parametrize(
    "disable,stream", [(False, None), (True, None), (False, "ComputeBasic")]
)
def test_probe_records_negative_capability_without_claiming_collection(
    tmp_path, monkeypatch, disable, stream
):
    binary = tmp_path / "probe"
    binary.write_bytes(b"test fixture, never executed")
    output = tmp_path / "result.json"
    observed = {}
    original_env = dict(os.environ)

    def run(argv, **kwargs):
        observed.update(kwargs)
        observed["argv"] = argv
        return subprocess.CompletedProcess(
            argv, 1, stdout="", stderr="missing metric dependency"
        )

    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe, "ensure_durable_output", lambda path, **_: path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--probe-binary", str(binary), "--output", str(output)]
        + (["--disable-metrics"] if disable else [])
        + (["--stream-group", stream] if stream else []),
    )
    probe.main()
    report = json.loads(output.read_text())
    assert report["capabilities"] is None
    assert report["collection_validated"] is False
    assert report["returncode"] == 1
    assert report["stderr"] == "missing metric dependency"
    assert report["requested_stream_group"] == stream
    if stream:
        assert observed["argv"][-1] == stream
    assert observed["env"]["ZET_ENABLE_METRICS"] == ("0" if disable else "1")
    assert dict(os.environ) == original_env


@pytest.mark.parametrize("failure", ["timeout", "launch"])
def test_probe_preserves_execution_failure(tmp_path, monkeypatch, failure):
    binary = tmp_path / "probe"
    binary.write_bytes(b"not executed")
    output = tmp_path / "result.json"

    def run(argv, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(
                argv, 60, output=b"partial JSON", stderr=b"loader diagnostic\xff"
            )
        raise PermissionError("probe not executable")

    monkeypatch.setattr(probe.subprocess, "run", run)
    monkeypatch.setattr(probe, "ensure_durable_output", lambda path, **_: path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--probe-binary", str(binary), "--output", str(output)],
    )
    probe.main()
    report = json.loads(output.read_text())
    assert report["returncode"] is None
    assert report["collection_validated"] is False
    assert report["capabilities"] is None
    assert report["launch_error"]
    assert report["timeout_seconds"] == 60
    if failure == "timeout":
        assert report["probe_status"] == "timed_out"
        assert report["stdout"] == "partial JSON"
        assert report["stderr"] == "loader diagnostic\ufffd"
    else:
        assert report["probe_status"] == "launch_failed"


def test_disabled_metrics_rejects_stream_check(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "probe",
            "--probe-binary",
            "/unused",
            "--output",
            "/unused",
            "--disable-metrics",
            "--stream-group",
            "ComputeBasic",
        ],
    )
    with pytest.raises(SystemExit):
        probe.main()
