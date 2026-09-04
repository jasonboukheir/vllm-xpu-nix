from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pytest

from scripts import kvarn_perf_run as perf
from scripts import kvarn_xpu_profile as profile
from scripts.kvarn_xpu_profile import (
    DIAGNOSTIC_WARNING,
    analyze_trace,
    load_trace,
    profile_benchmark_command,
    profile_delay_iterations,
    profiler_config,
    variant_provenance,
    verify_profiler_process_config,
)

HARDWARE = {
    "schema_version": 1,
    "torch_version": "test",
    "xpu_available": True,
    "xpu_device_count": 1,
    "xpu_device_names": [perf.EXPECTED_XPU_DEVICE_NAME],
    "probe_device": "xpu:0",
    "probe_value": 6.0,
}


def _step_name(*, context: int, batch: int, index: int) -> str:
    sk = context * batch + index * batch
    return (
        f"execute_{batch}_context_0(sq0sk0sqsq0sqsk0)"
        f"_generation_{batch}(sq{batch}sk{sk}sqsq{batch}sqsk{sk})"
    )


def _trace(
    *,
    arm: str = "candidate",
    context: int = 4096,
    batch: int = 1,
    steps: int = 20,
) -> dict[str, object]:
    events: list[dict[str, object]] = []
    for index in range(steps):
        timestamp = float(1_000 + index * 100)
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": _step_name(context=context, batch=batch, index=index),
                "ts": timestamp,
                "dur": 90.0,
                "pid": 1,
                "tid": 1,
            }
        )
        if arm == "candidate":
            events.append(
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "kvarn_native_xpu_decode",
                    "ts": timestamp + 10,
                    "dur": 50.0,
                    "pid": 1,
                    "tid": 1,
                }
            )
        events.extend(
            [
                {
                    "ph": "X",
                    "cat": "kernel",
                    "name": "kvarn_decode_kernel" if arm == "candidate" else "fmha",
                    "ts": timestamp + 20,
                    "dur": 12.0,
                    "pid": 0,
                    "tid": 9,
                    "args": {"device": 0, "stream": 7},
                },
                {
                    "ph": "X",
                    "cat": "kernel,device",
                    "name": "linear_kernel",
                    "ts": timestamp + 38,
                    "dur": 20.0,
                    "pid": 0,
                    "tid": 9,
                    "args": {"device": 0, "stream": 7},
                },
            ]
        )
    return {
        "deviceProperties": [{"name": perf.EXPECTED_XPU_DEVICE_NAME}],
        "traceEvents": events,
    }


def test_delay_skips_conservative_chunked_prefill() -> None:
    assert (
        profile_delay_iterations(context=4096, batch=1, max_num_batched_tokens=2048)
        == 6
    )
    assert (
        profile_delay_iterations(context=65023, batch=4, max_num_batched_tokens=2048)
        == 132
    )


def test_profiler_config_is_low_overhead_and_bounded(tmp_path: Path) -> None:
    trace_dir = (tmp_path / "trace").resolve()
    config = profiler_config(trace_dir, delay_iterations=36, profile_steps=32)

    assert config["profiler"] == "torch"
    assert config["torch_profiler_dir"] == str(trace_dir)
    assert config["delay_iterations"] == 36
    assert config["max_iterations"] == 32
    assert config["detailed_trace_annotation"] is True
    assert config["ignore_frontend"] is True
    assert config["torch_profiler_with_stack"] is False
    assert config["torch_profiler_with_memory"] is False
    assert config["torch_profiler_record_shapes"] is False
    verify_profiler_process_config(
        ["vllm", "serve", "model", "--profiler-config", json.dumps(config)],
        config,
    )

    with pytest.raises(perf.RunnerError, match="invalid profiler config"):
        verify_profiler_process_config(
            ["vllm", "serve", "model", "--profiler-config", "{"], config
        )


def test_candidate_trace_yields_gpu_only_diagnostic_summary() -> None:
    summary = analyze_trace(
        _trace(),
        arm="candidate",
        context=4096,
        batch=1,
        profile_steps=20,
        hardware_preflight=HARDWARE,
    )

    assert summary["status"] == "valid_diagnostic"
    assert summary["artifact_kind"] == "gpu_diagnostic_profile"
    assert summary["diagnostic_only"] is True
    assert summary["promotable"] is False
    assert summary["acceptance_eligible"] is False
    assert summary["parity_conclusion"] is None
    assert summary["warning"] == DIAGNOSTIC_WARNING
    assert summary["timing_source"] == "Kineto XPU device kernel events"
    assert summary["device_name"] == perf.EXPECTED_XPU_DEVICE_NAME
    assert summary["steady_decode_steps"] == 20
    assert summary["native_decode_annotation_count"] == 20
    assert summary["xpu_kernel_count"] == 40
    assert summary["xpu_kernel_duration_sum_us"] > 0
    assert len(summary["decode_steps"]) == 20
    assert list(summary["xpu_queue_timelines"]) == ["device=0,stream=7"]
    leaderboard = summary["gpu_leaderboard_metrics"]
    assert leaderboard["diagnostic_only"] is True
    assert leaderboard["device_kernel_time_us"] == 640.0
    assert leaderboard["kernel_launch_count"] == 40
    assert 0 <= leaderboard["device_idle_fraction"] < 1
    assert "throughput" not in summary
    assert "latency" not in summary
    assert "candidate_over_reference" not in summary


def test_auto_trace_is_valid_without_native_annotation() -> None:
    summary = analyze_trace(
        _trace(arm="reference", batch=4),
        arm="reference",
        context=4096,
        batch=4,
        profile_steps=20,
        hardware_preflight=HARDWARE,
    )

    assert summary["arm"] == "reference"
    assert summary["native_decode_annotation_count"] == 0
    assert summary["xpu_kernel_duration_sum_us"] > 0


def test_trace_rejects_cpu_only_timing() -> None:
    document = _trace()
    for event in document["traceEvents"]:  # type: ignore[index]
        if event["cat"] in {"kernel", "kernel,device"}:
            event["cat"] = "cpu_op"

    with pytest.raises(perf.RunnerError, match="no positive-duration XPU kernels"):
        analyze_trace(
            document,
            arm="candidate",
            context=4096,
            batch=1,
            profile_steps=20,
            hardware_preflight=HARDWARE,
        )


def test_candidate_trace_requires_native_annotations() -> None:
    with pytest.raises(perf.RunnerError, match="lacks Kvarn decoder annotations"):
        analyze_trace(
            _trace(arm="reference"),
            arm="candidate",
            context=4096,
            batch=1,
            profile_steps=20,
            hardware_preflight=HARDWARE,
        )


def test_trace_rejects_nonsteady_or_incomplete_decode_steps() -> None:
    document = _trace()
    first = document["traceEvents"][0]  # type: ignore[index]
    first["name"] = first["name"].replace("generation_1", "generation_0")
    with pytest.raises(perf.RunnerError, match="non-steady-decode"):
        analyze_trace(
            document,
            arm="candidate",
            context=4096,
            batch=1,
            profile_steps=20,
            hardware_preflight=HARDWARE,
        )

    incomplete = _trace()
    incomplete["traceEvents"] = incomplete["traceEvents"][:-4]  # type: ignore[index]
    with pytest.raises(perf.RunnerError, match="expected exactly 20"):
        analyze_trace(
            incomplete,
            arm="candidate",
            context=4096,
            batch=1,
            profile_steps=20,
            hardware_preflight=HARDWARE,
        )


def test_trace_requires_exact_b70_compute_preflight() -> None:
    wrong = {**HARDWARE, "xpu_device_names": ["Intel Other XPU"]}
    with pytest.raises(perf.RunnerError, match="Intel Arc Pro B70"):
        analyze_trace(
            _trace(),
            arm="candidate",
            context=4096,
            batch=1,
            profile_steps=20,
            hardware_preflight=wrong,
        )


def test_gzip_trace_loading(tmp_path: Path) -> None:
    path = tmp_path / "worker.pt.trace.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(_trace(), stream)

    loaded = load_trace(path)
    assert len(loaded["traceEvents"]) == 80


@pytest.mark.parametrize(
    ("variant", "variant_id"),
    [
        ("q6_scalar", 2),
        ("q6_vector", 4),
        ("q6_cached_weights", 6),
        ("q6_exact_rows", 7),
        ("q6_cached_weights_exact_rows", 8),
        ("q6_page_pair", 9),
        ("q6_main_grf128", 10),
        ("q6_split_reducer_specialized", 11),
        ("q6_next_page_prefetch", 12),
    ],
)
def test_profile_command_and_dpas_launcher_provenance(
    tmp_path: Path, variant: str, variant_id: int
) -> None:
    workload = perf.Workload(
        context=4096, batch=4, output_tokens=96, num_prompts=4, seed=17
    )
    run = perf.PlannedRun(workload=workload, arm="candidate", order=1)
    args = argparse.Namespace(
        candidate_env=tmp_path / "candidate",
        base_url="http://127.0.0.1:8000",
        served_model="sunny-chat",
        model=perf.DEFAULT_MODEL,
        config_ref="path:/config",
        max_model_len=65536,
        max_num_batched_tokens=2048,
        native_layout="xe2_dpas",
        native_kernel_variant=variant,
        native_split_policy="fixed",
        native_splits={4: 16},
        onednn_deterministic=True,
        variant_id="factory-dpas-001",
        resolved_launchers={
            f"vllm-xpu-brutus-kvarn-native-dpas-{variant}-b4": (
                "/nix/store/app/bin/launch"
            )
        },
    )

    assert perf.launcher_name(run, args) == (
        f"vllm-xpu-brutus-kvarn-native-dpas-{variant}-b4"
    )
    assert perf.NATIVE_KERNEL_VARIANTS[variant] == variant_id
    assert perf.service_command(run, args)[0] == "/nix/store/app/bin/launch"
    command = profile_benchmark_command(run, args, tmp_path / "raw.json")
    assert command[command.index("--num-warmups") + 1] == "4"
    assert command[-1] == "--profile"
    assert variant_provenance(run, args) == {
        "variant_id": "factory-dpas-001",
        "layout": "xe2_dpas",
        "kernel_strategy": f"native_xe2_decode_{variant}",
        "split_count": 16,
        "max_split_count": 16,
        "split_policy": "fixed_b4s16",
        "split_policy_selector": "fixed",
        "fusion_selection": "fused_attention_decode",
        "scheduling_selection": "split_k",
        "scheduler_max_num_batched_tokens": 2048,
        "scheduler_max_num_seqs": 4,
    }
    args.native_split_policy = "b70_q6"
    args.native_splits = {4: 8}
    args.variant_id = "factory-dpas-b70-q6"
    assert variant_provenance(run, args) == {
        "variant_id": "factory-dpas-b70-q6",
        "layout": "xe2_dpas",
        "kernel_strategy": f"native_xe2_decode_{variant}",
        "split_count": 8,
        "max_split_count": 32,
        "split_policy": "b70_q6",
        "split_policy_selector": "b70_q6",
        "fusion_selection": "fused_attention_decode",
        "scheduling_selection": "split_k",
        "scheduler_max_num_batched_tokens": 2048,
        "scheduler_max_num_seqs": 4,
    }


def test_candidate_profile_cli_rejects_fixed_round2_launcher_contract(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    for executable in ("vllm", "python"):
        (candidate / "bin" / executable).write_text("", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    common = [
        "--candidate-env",
        str(candidate),
        "--allow-tmp",
        "--runtime-cache",
        str(tmp_path / "runtime-cache"),
        "--config-repo",
        str(config),
        "--config-ref",
        f"path:{config}",
        "--native-layout",
        "xe2_dpas",
        "--native-kernel-variant",
        "q6_scalar",
    ]

    with pytest.raises(SystemExit):
        profile.parse_args(
            [
                *common,
                "--native-split-policy",
                "fixed",
                "--output-dir",
                str(tmp_path / "fixed"),
            ]
        )

    args = profile.parse_args(
        [
            *common,
            "--native-split-policy",
            "b70_q6",
            "--output-dir",
            str(tmp_path / "b70"),
        ]
    )
    assert args.native_splits == {1: 32}
