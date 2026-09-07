import json
import sys

import pytest
from test_kvarn_xpu_trace import event

from scripts import kvarn_xpu_profile_compare as compare
from scripts.kvarn_factory_run import sha256_file, write_json_atomic
from scripts.kvarn_xpu_profile_compare import normalized_service_profile
from scripts.kvarn_xpu_profile_sources import snapshot_sources


def profile(directory, *, model="model-a", shapes=False):
    return {
        "canonical_matched_profile": {
            "argv": [
                "vllm",
                "serve",
                model,
                "--profiler-config",
                json.dumps(
                    {
                        "torch_profiler_dir": directory,
                        "torch_profiler_record_shapes": shapes,
                        "delay_iterations": 17,
                        "max_iterations": 4,
                    }
                ),
            ],
            "environment": {"KVARN_NATIVE_XPU": "<ARM_VALUE>"},
        }
    }


def test_only_output_path_is_ignored_in_profile_comparison():
    original = profile("/first")
    normalized = normalized_service_profile(original)
    assert normalized == normalized_service_profile(profile("/second"))
    assert normalized != normalized_service_profile(profile("/second", model="model-b"))
    assert normalized != normalized_service_profile(profile("/second", shapes=True))
    assert (
        json.loads(original["canonical_matched_profile"]["argv"][-1])[
            "torch_profiler_dir"
        ]
        == "/first"
    )


def capture(directory, arm, *, model="model-a", annotation="execute_one"):
    source = directory / "fixture-source"
    (source / "scripts").mkdir(parents=True)
    (source / "scripts" / "capture.py").write_text("# identical fixture source\n")
    snapshot = directory / "source-snapshot"
    snapshot_sources(source, snapshot)
    trace = directory / "trace.json"
    summary = directory / "profile-summary.json"
    service = directory / "diagnostic-service-profile.json"
    write_json_atomic(
        trace,
        {
            "traceEvents": [
                event(annotation, "user_annotation", 0, 10),
                event("aten::mm", "cpu_op", 1, 3, **{"External id": 7}),
                event("gemm_kernel", "kernel", 15, 8, pid=0, **{"External id": 7}),
            ]
        },
    )
    write_json_atomic(
        summary,
        {
            "process_package": "/immutable/package",
            "candidate_closure_sha256": "abc",
            "device_name": "test-device",
        },
    )
    write_json_atomic(service, profile(str(directory), model=model))
    write_json_atomic(
        directory / "run.json",
        {
            "arm": arm,
            "status": "valid_diagnostic",
            "source_provenance_qualified": True,
            "source_snapshot": str(snapshot),
            "source_snapshot_sha256": sha256_file(snapshot / "manifest.json"),
            "kineto_trace": str(trace),
            "kineto_trace_sha256": sha256_file(trace),
            "profile_summary_sha256": sha256_file(summary),
            "service_profile_sha256": sha256_file(service),
            "profiler_config": {"torch_profiler_dir": str(directory)},
            "workload": {"batch": 1, "context": 8},
            "profile_phase": "prefill",
            "profile_start_step": 1,
        },
    )


@pytest.mark.parametrize(
    "mismatch",
    [None, "model", "annotation", "hash", "source", "snapshot", "missing-provenance"],
)
def test_cli_compares_matched_captures_and_rejects_mismatches(
    tmp_path, monkeypatch, mismatch
):
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    capture(reference, "reference")
    capture(
        candidate,
        "candidate",
        model="wrong" if mismatch == "model" else "model-a",
        annotation="execute_wrong" if mismatch == "annotation" else "execute_one",
    )
    if mismatch == "hash":
        (candidate / "trace.json").write_text("{}")
    if mismatch == "source":
        (candidate / "source-snapshot/files/scripts/capture.py").write_text("changed")
    if mismatch == "snapshot":
        (candidate / "source-snapshot/manifest.json").write_text("{}")
    if mismatch == "missing-provenance":
        path = candidate / "run.json"
        manifest = json.loads(path.read_text())
        manifest.pop("source_provenance_qualified")
        write_json_atomic(path, manifest)
    output, markdown = tmp_path / "result.json", tmp_path / "result.md"
    monkeypatch.setattr(compare, "ensure_durable_output", lambda path, **_: path)
    monkeypatch.setattr(compare, "ensure_durable", lambda path, **_: path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare",
            "--reference-run",
            str(reference),
            "--candidate-run",
            str(candidate),
            "--output",
            str(output),
            "--markdown-output",
            str(markdown),
        ],
    )
    if mismatch:
        with pytest.raises(ValueError):
            compare.main()
        assert not output.exists()
    else:
        compare.main()
        report = json.loads(output.read_text())
        assert report["source_provenance_qualified"] is True
        assert report["analyses"]["reference"]["window"]["device_busy_union_us"] == 8
        assert "CPU origins resolved: 100.00%" in markdown.read_text()
        assert "not request wall time" in markdown.read_text()
        with pytest.raises(ValueError, match="already exists"):
            compare.main()
