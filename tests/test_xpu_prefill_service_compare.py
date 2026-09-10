"""Reject attribution and output-evidence errors in Python dispatch comparisons."""

import copy
import json
from pathlib import Path

import pytest

from scripts import xpu_prefill_service_compare as comparison


def identity(arm):
    package = f"/nix/store/{arm}-python3.12-vllm-xpu-test"
    environment = f"/nix/store/{arm}-service-env"
    process = sorted([package, "/nix/store/same-torch", "/nix/store/same-kernels"])
    candidate = sorted([*process, environment, f"/nix/store/{arm}-python-env"])
    return {
        "process_package": package,
        "process_executable": package + "/bin/.vllm-wrapped",
        "candidate_env": environment,
        "process_closure_paths": process,
        "process_closure_sha256": comparison.closure_digest(process),
        "candidate_closure_paths": candidate,
        "candidate_closure_sha256": comparison.closure_digest(candidate),
    }


WRAPPERS = [("/nix/store/control-python-env", "/nix/store/candidate-python-env")]


def test_records_only_declared_wrapper_and_python_package_differences():
    result = comparison.compare_identities(
        identity("control"), identity("candidate"), WRAPPERS
    )
    assert result["identical_process_dependencies"] == [
        "/nix/store/same-kernels",
        "/nix/store/same-torch",
    ]
    assert len(result["allowed_path_pairs"]) == 3


def test_rejects_native_dependency_drift_even_with_wrapper_exception():
    left, right = identity("control"), identity("candidate")
    right["process_closure_paths"].remove("/nix/store/same-kernels")
    right["process_closure_paths"].append("/nix/store/changed-kernels")
    with pytest.raises(ValueError, match="process dependencies changed"):
        comparison.compare_identities(left, right, WRAPPERS)


def test_requires_explicit_outer_wrapper_exception():
    with pytest.raises(ValueError, match="environment closure changed"):
        comparison.compare_identities(identity("control"), identity("candidate"))


def test_cannot_allow_a_process_dependency_as_an_environment_wrapper():
    bad = [("/nix/store/same-kernels", "/nix/store/same-kernels")]
    with pytest.raises(ValueError, match="outer environment wrappers"):
        comparison.compare_identities(identity("control"), identity("candidate"), bad)


def test_rejects_corrupt_recorded_closure_digest():
    recorded = identity("control")
    expected = {k: recorded[k] for k in ("candidate_env", "process_package")}
    manifest = {
        "runtime_identity": recorded,
        "service_env": recorded["candidate_env"],
        "actual_argv": [recorded["process_executable"], "serve"],
    }
    comparison.validate_identity(manifest, expected)
    manifest = copy.deepcopy(manifest)
    manifest["runtime_identity"]["candidate_closure_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="invalid candidate closure"):
        comparison.validate_identity(manifest, expected)


def test_reads_withpackages_binary_wrapper_without_executing_it(tmp_path):
    (tmp_path / "bin").mkdir()
    package = "/nix/store/example-python3.12-vllm-xpu-test"
    (tmp_path / "bin/vllm").write_bytes(
        b"\x7fELF\x00" + (package + "/bin/vllm").encode() + b"\x00"
    )
    assert comparison.runtime_spec(tmp_path)["process_package"] == package


def request_capture(path: Path, *, prompt=(10, 20), output=(30, 40), bundled=False):
    path.mkdir()
    name = "warmup"
    request = {"return_token_ids": True}
    result = {
        "id": name,
        "phase": "warmup",
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(output)},
        "content": "same decoded text",
        "ttft_seconds": 0.1,
        "total_seconds": 0.2,
    }
    for suffix, value in (("request.json", request), ("response.json", result)):
        (path / f"{name}-{suffix}").write_text(json.dumps(value))
    events = [{"prompt_token_ids": list(prompt), "choices": []}]
    for group in [list(output)] if bundled else [[t] for t in output]:
        events.append({"choices": [{"token_ids": group}]})
    records = [
        {"seconds": i / 10, "line": "data: " + json.dumps(event)}
        for i, event in enumerate(events)
    ]
    records.append({"seconds": 0.3, "line": "data: [DONE]"})
    (path / f"{name}-sse.jsonl").write_text("\n".join(map(json.dumps, records)))
    return result


def test_warmup_output_mismatch_is_not_hidden_by_equal_text(tmp_path):
    left, right = tmp_path / "control", tmp_path / "candidate"
    a = request_capture(left)
    b = request_capture(right, output=(30, 41))
    report = comparison.compare_request(left, right, a, b, timed=False)
    assert report["same_content"]
    assert not report["same_generated_tokens"]
    assert report["first_token_mismatch"] == 1


def test_rejects_mismatched_processed_prompt_even_with_identical_request_json(tmp_path):
    left, right = tmp_path / "control", tmp_path / "candidate"
    a = request_capture(left)
    b = request_capture(right, prompt=(10, 21))
    with pytest.raises(ValueError, match="processed prompt IDs"):
        comparison.compare_request(left, right, a, b, timed=False)


def test_bundled_mtp_events_cannot_establish_individual_token_latency(tmp_path):
    directory = tmp_path / "bundled"
    result = request_capture(directory, bundled=True)
    with pytest.raises(ValueError, match="bundled token events"):
        comparison.numerical_timing(directory, result)


def service_capture(path, arm):
    """A complete small service artifact, audited through the real shared helpers."""
    path.mkdir()
    runtime = identity(arm)
    workload = []
    for index in range(4):
        name = f"prefill-64-{index}"
        phase = "warmup" if index == 0 else "performance"
        messages = [{"role": "user", "content": "synthetic prompt"}]
        generation = {"max_tokens": 512, "ignore_eos": True, "return_token_ids": True}
        workload.append(
            {
                "id": name,
                "phase": phase,
                "messages": messages,
                "generation": generation,
                "expected_terms": [],
                "coverage": {
                    "kind": "text-prefill",
                    "prompt_tokens": 64,
                    **comparison.chunk_plan(64, 2048),
                },
            }
        )
        request = {
            "model": "sunny-vision",
            "messages": messages,
            "temperature": 0,
            "seed": 42,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
            **generation,
        }
        usage = {"prompt_tokens": 64, "completion_tokens": 512}
        response = {
            "id": name,
            "phase": phase,
            "content": "synthetic output",
            "usage": usage,
            "stream_valid": True,
            "stream_done": True,
            "coverage_valid": True,
            "ttft_seconds": 0.01,
            "total_seconds": 6.0,
        }
        events = [{"prompt_token_ids": [10] * 64, "choices": []}]
        events += [
            {
                "choices": [
                    {
                        "token_ids": [40],
                        "delta": {"content": "synthetic output" if i == 0 else ""},
                    }
                ]
            }
            for i in range(512)
        ]
        events.append({"choices": [], "usage": usage})
        records = [
            {"seconds": i * 0.01, "line": "data: " + json.dumps(event)}
            for i, event in enumerate(events)
        ]
        records.append({"seconds": 6.0, "line": "data: [DONE]"})
        for suffix, document in (
            ("request.json", request),
            ("response.json", response),
            ("tokenize.json", {"count": 64, "tokens": [10] * 64}),
        ):
            (path / f"{name}-{suffix}").write_text(json.dumps(document))
        (path / f"{name}-sse.jsonl").write_text("\n".join(map(json.dumps, records)))
    (path / "workload.json").write_text(json.dumps(workload))
    (path / "service.log").write_text("Synthetic engine started and stopped.\n")
    (path / "metrics.txt").write_text('vllm:num_preemptions_total{engine="0"} 0\n')
    (path / "memory-fdinfo.jsonl").write_text(
        json.dumps(
            {
                "clients": {
                    "1": {"fields": {"drm-pdev": "gpu0", "drm-resident-vram0": "1 GiB"}}
                },
                "errors": [],
            }
        )
        + "\n"
    )
    manifest = {
        "status": "requests-completed-not-yet-qualified",
        "suite": comparison.PREFILL,
        "runtime_identity": runtime,
        "service_env": runtime["candidate_env"],
        "actual_argv": [
            "python",
            runtime["process_executable"],
            "serve",
            "model",
            "--max-num-seqs",
            "1",
            "--max-num-batched-tokens",
            "2048",
            "--kv-cache-dtype",
            "auto",
            "--no-enable-prefix-caching",
        ],
        "actual_environment": {"VLLM_CACHE_ROOT": str(path / "runtime-cache")},
        "selected_environment": {"VLLM_CACHE_ROOT": str(path / "runtime-cache")},
        "transport_environment": {},
        "speculative_config": None,
        "profiler_config": None,
        "harness_sha256": {},
        "image_sha256": {},
        "service_log_sha256": comparison.sha256_file(path / "service.log"),
        "workload_sha256": comparison.sha256_file(path / "workload.json"),
    }
    (path / "manifest.json").write_text(json.dumps(manifest))


def test_complete_capture_audit_and_warmup_failure_reporting(tmp_path, monkeypatch):
    paths = [tmp_path / arm for arm in comparison.ARMS]
    for path, arm in zip(paths, comparison.ARMS, strict=True):
        service_capture(path, arm)

    def expected_runtime(path):
        recorded = identity(path.name)
        return {key: recorded[key] for key in ("candidate_env", "process_package")}

    monkeypatch.setattr(comparison, "runtime_spec", expected_runtime)
    kwargs = {
        "control_env": paths[0],
        "candidate_env": paths[1],
        "allowed_wrappers": WRAPPERS,
    }
    report = comparison.compare([paths], **kwargs)
    assert report["status"] == "passed-correctness-and-resource-checks"
    assert report["timing_variation"]["auto/64"]["fresh_start_pairs"] == 1
    assert len(report["pairs"][0]["requests"]) == 4

    raw_path = paths[1] / "prefill-64-0-sse.jsonl"
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    event = json.loads(records[1]["line"][6:])
    event["choices"][0]["token_ids"] = [41]
    records[1]["line"] = "data: " + json.dumps(event)
    raw_path.write_text("\n".join(map(json.dumps, records)))
    failed = comparison.compare([paths], **kwargs)
    assert failed["status"] == "failed"
    assert any(f.get("id") == "prefill-64-0" for f in failed["failures"])
