from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import scripts.kvarn_perf_run as runner
from scripts.kvarn_perf_gate import compare
from scripts.kvarn_perf_run import (
    ARM_ORDER,
    PlannedRun,
    ProcessSupervisor,
    RunnerError,
    Workload,
    abba_arms,
    benchmark_command,
    build_plan,
    gate_workload,
    native_splits_for_run,
    paired_noninferiority,
    parse_running_metric,
    persist_warmup_result,
    pooled_tail_latency,
    probe_xpu_hardware,
    resolve_launchers,
    result_exit_code,
    run_managed_process,
    seal_benchmark_result,
    service_command,
    service_profile_evidence,
    statistical_parity,
    summarize_exploratory_workload,
    validate_matched_results,
    verify_candidate_identity,
    verify_correctness_candidate_identity,
    verify_service_profile,
    warmup_command,
)
from tests.test_kvarn_perf_gate import _correctness as _valid_correctness

MODEL = "jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound"
REVISION = "6b0622f4354481d5d04577d48ba0db844efc1330"
IDENTITY = {
    "process_executable": "/nix/store/package/bin/.vllm-wrapped",
    "process_package": "/nix/store/package",
    "process_closure_sha256": "a" * 64,
    "candidate_closure_sha256": "b" * 64,
}
PROFILE = {
    "max_num_batched_tokens": "2048",
    "canonical_matched_profile_sha256": "3" * 64,
}
HARDWARE_PREFLIGHT = {
    "schema_version": 1,
    "torch_version": "test",
    "xpu_available": True,
    "xpu_device_count": 1,
    "xpu_device_names": ["Intel(R) Arc(TM) Pro B70 Graphics"],
    "probe_device": "xpu:0",
    "probe_value": 6.0,
}


def _args(tmp_path: Path) -> argparse.Namespace:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "vllm").write_text("", encoding="utf-8")
    (candidate / "bin" / "python").write_text("", encoding="utf-8")
    hardware_preflight = tmp_path / "hardware-preflight.json"
    hardware_preflight.write_text(json.dumps(HARDWARE_PREFLIGHT), encoding="utf-8")
    return argparse.Namespace(
        base_url="http://127.0.0.1:8000",
        candidate_env=candidate,
        config_ref="path:/config",
        config_repo=tmp_path / "config",
        packaging_repo=tmp_path,
        exploratory=False,
        max_model_len=65536,
        max_num_batched_tokens=2048,
        model=MODEL,
        model_revision=REVISION,
        native_splits={1: 24, 4: 16},
        num_warmups=None,
        hf_home=Path("/var/cache/huggingface"),
        runtime_cache=tmp_path / "runtime-cache",
        served_model="sunny-chat",
        hardware_preflight_path=hardware_preflight,
    )


def _correctness(path: Path, candidate_id: str) -> Path:
    return _valid_correctness(path, candidate_id)


def _raw_result(path: Path, *, throughput: float, workload: Workload) -> Path:
    input_lens = [workload.context] * workload.num_prompts
    output_lens = [workload.output_tokens] * workload.num_prompts
    duration = sum(output_lens) / throughput
    path.write_text(
        json.dumps(
            {
                "completed": workload.num_prompts,
                "failed": 0,
                "duration": duration,
                "total_input_tokens": sum(input_lens),
                "total_output_tokens": sum(output_lens),
                "output_throughput": throughput,
                "request_throughput": workload.num_prompts / duration,
                "total_token_throughput": (sum(input_lens) + sum(output_lens))
                / duration,
                "input_lens": input_lens,
                "output_lens": output_lens,
                "ttfts": [0.1] * workload.num_prompts,
                "itls": [
                    [0.05] * (workload.output_tokens - 1)
                    for _ in range(workload.num_prompts)
                ],
                "backend": "openai",
                "model_id": "sunny-chat",
                "tokenizer_id": MODEL,
                "num_prompts": workload.num_prompts,
                "request_rate": "inf",
                "max_concurrency": workload.batch,
                "max_concurrent_requests": workload.batch,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_plan_repeats_abba_independently_for_every_cell() -> None:
    plan = build_plan(
        contexts=[4096, 65023],
        batches=[1, 4],
        output_tokens=512,
        waves_per_run=2,
        repeats=4,
        seed=17,
        max_model_len=65536,
    )

    assert len(plan) == 32
    for start in range(0, len(plan), 8):
        cell = plan[start : start + 8]
        assert [run.arm for run in cell] == list(ARM_ORDER * 2)
        assert [run.order for run in cell] == list(range(1, 9))
        assert cell[0].workload.num_prompts == cell[0].workload.batch * 2


def test_plan_rejects_odd_repeats_and_context_overflow() -> None:
    with pytest.raises(RunnerError, match="even integer"):
        abba_arms(5)
    with pytest.raises(RunnerError, match="exceeds max model length"):
        build_plan(
            contexts=[65025],
            batches=[1],
            output_tokens=512,
            waves_per_run=1,
            repeats=4,
            seed=17,
            max_model_len=65536,
        )


def test_exploratory_plan_allows_one_abba_block_but_formal_does_not() -> None:
    with pytest.raises(RunnerError, match="at least 4"):
        abba_arms(2)
    assert abba_arms(2, minimum_repeats=2) == ARM_ORDER
    plan = build_plan(
        contexts=[4096],
        batches=[1],
        output_tokens=512,
        waves_per_run=2,
        repeats=2,
        seed=17,
        max_model_len=65536,
        minimum_repeats=2,
    )
    assert [run.arm for run in plan] == list(ARM_ORDER)


def test_correctness_is_optional_only_for_exploratory_cli(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "vllm").write_text("", encoding="utf-8")
    (candidate / "bin" / "python").write_text("", encoding="utf-8")
    common = [
        "--candidate-env",
        str(candidate),
        "--allow-tmp",
        "--plan-only",
        "--runtime-cache",
        str(tmp_path / "runtime-cache"),
        "--context",
        "4096",
        "--batch",
        "1",
        "--repeats",
        "2",
    ]

    exploratory = runner.parse_args(
        [
            *common,
            "--exploratory",
            "--output-dir",
            str(tmp_path / "exploratory-output"),
        ]
    )
    assert exploratory.exploratory is True
    assert exploratory.correctness is None
    assert exploratory.repeats == 2

    with pytest.raises(SystemExit):
        runner.parse_args([*common, "--output-dir", str(tmp_path / "formal-output")])


def test_exploratory_plan_session_has_no_formal_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "vllm").write_text("", encoding="utf-8")
    (candidate / "bin" / "python").write_text("", encoding="utf-8")
    args = runner.parse_args(
        [
            "--candidate-env",
            str(candidate),
            "--exploratory",
            "--plan-only",
            "--context",
            "4096",
            "--batch",
            "1",
            "--repeats",
            "2",
            "--allow-tmp",
            "--runtime-cache",
            str(tmp_path / "runtime-cache"),
            "--output-dir",
            str(tmp_path / "plan-output"),
        ]
    )
    monkeypatch.setattr(
        runner,
        "resolve_launchers",
        lambda _plan, _args: {
            "vllm-xpu-brutus-auto-b1": "/nix/store/auto/bin/auto",
            "vllm-xpu-brutus-kvarn-native-b1": "/nix/store/native/bin/native",
        },
    )
    monkeypatch.setattr(
        runner,
        "repository_state",
        lambda name, path: {"name": name, "path": str(path)},
    )

    session = runner.execute(args)

    assert session["evidence_mode"] == "exploratory"
    assert session["promotable"] is False
    assert session["formal_gates_skipped"] is True
    assert len(session["plan"]) == 4
    assert "correctness_artifact" not in session
    assert "correctness_sha256" not in session
    assert "acceptance" not in session
    assert "performance_status" not in session
    assert "statistical_parity_status" not in session


def test_commands_pin_launcher_and_deterministic_workload(tmp_path: Path) -> None:
    args = _args(tmp_path)
    run = PlannedRun(Workload(16384, 4, 512, 8, 17), "candidate", 2)
    raw = tmp_path / "run" / "benchmark.raw.json"

    assert service_command(run, args) == [
        "nix",
        "run",
        "--impure",
        "path:/config#vllm-xpu-brutus-kvarn-native-b4",
        "--",
        str(args.candidate_env),
        "--max-num-batched-tokens",
        "2048",
    ]
    command = benchmark_command(run, args, raw)
    assert command[0] == str(args.candidate_env / "bin" / "vllm")
    assert command[command.index("--random-input-len") + 1] == "16384"
    assert command[command.index("--random-output-len") + 1] == "512"
    assert command[command.index("--random-range-ratio") + 1] == "0"
    assert command[command.index("--max-concurrency") + 1] == "4"
    assert command[command.index("--num-prompts") + 1] == "8"
    assert command[command.index("--num-warmups") + 1] == "0"
    assert "--ignore-eos" in command
    assert "--save-detailed" in command
    warmup_raw = tmp_path / "run" / "warmup.raw.json"
    warmup = warmup_command(run, args, warmup_raw)
    assert warmup is not None
    assert warmup[warmup.index("--num-prompts") + 1] == "4"
    assert "--save-result" in warmup
    assert warmup.count("--result-dir") == 1
    assert warmup[warmup.index("--result-dir") + 1] == str(warmup_raw.parent)
    assert warmup[warmup.index("--result-filename") + 1] == warmup_raw.name


@pytest.mark.parametrize("arm", ["reference", "candidate"])
@pytest.mark.parametrize("batch", [1, 4])
def test_service_command_pins_scheduler_budget_for_every_arm(
    tmp_path: Path, arm: str, batch: int
) -> None:
    args = _args(tmp_path)
    args.max_num_batched_tokens = 4096
    run = PlannedRun(Workload(4096, batch, 512, batch, 17), arm, 1)

    command = service_command(run, args)

    assert command[-2:] == ["--max-num-batched-tokens", "4096"]


def test_scheduler_budget_cli_defaults_overrides_and_rejects_zero(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "vllm").write_text("", encoding="utf-8")
    (candidate / "bin" / "python").write_text("", encoding="utf-8")
    correctness = tmp_path / "correctness.json"
    correctness.write_text("{}\n", encoding="utf-8")
    common = [
        "--candidate-env",
        str(candidate),
        "--correctness",
        str(correctness),
        "--allow-tmp",
        "--plan-only",
        "--runtime-cache",
        str(tmp_path / "runtime-cache"),
    ]

    default = runner.parse_args(
        [*common, "--output-dir", str(tmp_path / "default-output")]
    )
    explicit = runner.parse_args(
        [
            *common,
            "--output-dir",
            str(tmp_path / "explicit-output"),
            "--max-num-batched-tokens",
            "4096",
        ]
    )

    assert default.max_num_batched_tokens == 2048
    assert explicit.max_num_batched_tokens == 4096
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *common,
                "--output-dir",
                str(tmp_path / "invalid-output"),
                "--max-num-batched-tokens",
                "0",
            ]
        )


def test_formal_cli_requires_complete_matrix_and_full_width_warmup(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    for name in ("vllm", "python"):
        (candidate / "bin" / name).write_text("", encoding="utf-8")
    correctness = tmp_path / "correctness.json"
    correctness.write_text("{}\n", encoding="utf-8")
    common = [
        "--candidate-env",
        str(candidate),
        "--correctness",
        str(correctness),
        "--allow-tmp",
        "--plan-only",
        "--runtime-cache",
        str(tmp_path / "runtime-cache"),
    ]

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *common,
                "--context",
                "4096",
                "--output-dir",
                str(tmp_path / "subset-output"),
            ]
        )
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                *common,
                "--num-warmups",
                "0",
                "--output-dir",
                str(tmp_path / "cold-output"),
            ]
        )


@pytest.mark.parametrize(
    "extra",
    [
        ["--min-throughput-ratio", "0.94"],
        ["--min-request-decode-ratio", "0.94"],
        ["--max-latency-ratio", "1.11"],
        ["--parity-ratio", "0.97"],
        ["--min-parity-pairs", "3"],
        ["--repeats", "6"],
        ["--output-tokens", "2"],
        ["--waves-per-run", "1"],
    ],
)
def test_formal_cli_cannot_weaken_acceptance_or_decode_shape(
    tmp_path: Path, extra: list[str]
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    for name in ("vllm", "python"):
        (candidate / "bin" / name).write_text("", encoding="utf-8")
    correctness = tmp_path / "correctness.json"
    correctness.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--candidate-env",
                str(candidate),
                "--correctness",
                str(correctness),
                "--allow-tmp",
                "--plan-only",
                "--runtime-cache",
                str(tmp_path / "runtime-cache"),
                "--output-dir",
                str(tmp_path / "output"),
                *extra,
            ]
        )


def test_exploratory_cli_retains_flexible_non_promotable_shape(
    tmp_path: Path,
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    for name in ("vllm", "python"):
        (candidate / "bin" / name).write_text("", encoding="utf-8")

    args = runner.parse_args(
        [
            "--candidate-env",
            str(candidate),
            "--exploratory",
            "--allow-tmp",
            "--plan-only",
            "--runtime-cache",
            str(tmp_path / "runtime-cache"),
            "--output-dir",
            str(tmp_path / "output"),
            "--context",
            "4096",
            "--batch",
            "1",
            "--repeats",
            "2",
            "--output-tokens",
            "2",
            "--waves-per-run",
            "1",
            "--parity-ratio",
            "0.50",
            "--min-parity-pairs",
            "2",
        ]
    )

    assert args.exploratory is True
    assert args.output_tokens == 2
    assert args.waves_per_run == 1


def test_xpu_preflight_requires_an_exact_b70_compute_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)

    def completed(probe: dict[str, object]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=json.dumps(probe), stderr="")

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(HARDWARE_PREFLIGHT),
    )
    assert probe_xpu_hardware(args)["xpu_device_names"] == [
        "Intel(R) Arc(TM) Pro B70 Graphics"
    ]

    wrong_device = {**HARDWARE_PREFLIGHT, "xpu_device_names": ["Intel Arc A770"]}
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(wrong_device),
    )
    with pytest.raises(RunnerError, match="requires one Intel Arc Pro B70"):
        probe_xpu_hardware(args)

    unavailable = {
        **HARDWARE_PREFLIGHT,
        "xpu_available": False,
        "xpu_device_count": 0,
        "xpu_device_names": [],
        "probe_device": None,
        "probe_value": None,
    }
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: completed(unavailable),
    )
    with pytest.raises(RunnerError, match="requires one Intel Arc Pro B70"):
        probe_xpu_hardware(args)


def test_scheduler_metric_sums_labeled_workers() -> None:
    assert (
        parse_running_metric(
            "# HELP ignored\n"
            'vllm:num_requests_running{engine="0"} 3\n'
            'vllm:num_requests_running{engine="1"} 1\n'
        )
        == 4
    )
    with pytest.raises(RunnerError, match="missing"):
        parse_running_metric("vllm:num_requests_waiting 0\n")


def test_profile_verification_uses_actual_argv_and_environment(tmp_path: Path) -> None:
    args = _args(tmp_path)
    run = PlannedRun(Workload(4096, 4, 512, 8, 17), "candidate", 2)
    argv = [
        "/nix/store/package/bin/.vllm-wrapped",
        "serve",
        MODEL,
        "--served-model-name",
        "sunny-chat",
        "--revision",
        REVISION,
        "--dtype",
        "bfloat16",
        "--quantization",
        "compressed-tensors",
        "--kv-cache-dtype",
        "kvarn_k4v4_g128_compact",
        "--max-model-len",
        "65536",
        "--max-num-seqs",
        "4",
        "--max-num-batched-tokens",
        "2048",
        "--gpu-memory-utilization",
        "0.95",
        "--enforce-eager",
        "--language-model-only",
        "--no-enable-prefix-caching",
    ]
    environment = {
        "CCL_ATL_TRANSPORT": "ofi",
        "CCL_LOG_LEVEL": "warn",
        "CCL_PROCESS_LAUNCHER": "none",
        "CCL_ZE_IPC_EXCHANGE": "sockets",
        "HF_HOME": "/var/cache/huggingface",
        "HOME": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "KVARN_NATIVE_XPU": "1",
        "KVARN_NATIVE_XPU_DECODE": "1",
        "KVARN_NATIVE_XPU_DPAS_LAYOUT": "0",
        "KVARN_NATIVE_XPU_MATERIALIZE": "1",
        "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH": "1",
        "KVARN_NATIVE_XPU_SPLITS": "16",
        "KVARN_PREFILL_FP16_WINDOW_BLOCKS": "16",
        "VLLM_CACHE_ROOT": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_KVARN_DEFER_PREFILL_FLUSH": None,
        "VLLM_XPU_ENABLE_XPU_GRAPH": None,
        "XDG_CACHE_HOME": str(args.runtime_cache),
    }

    verify_service_profile(argv, environment, run, args)
    assert native_splits_for_run(run, args) == 16
    environment["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "1"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, run, args)
    environment["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "0"
    argv[argv.index("--max-num-batched-tokens") + 1] = "8192"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, run, args)
    argv[argv.index("--max-num-batched-tokens") + 1] = "2048"
    environment["KVARN_NATIVE_XPU_SPLITS"] = "1"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, run, args)
    environment["KVARN_NATIVE_XPU_SPLITS"] = "16"
    environment["KVARN_NATIVE_XPU_DECODE"] = "0"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, run, args)
    environment["KVARN_NATIVE_XPU_DECODE"] = "1"
    environment["VLLM_KVARN_DEFER_PREFILL_FLUSH"] = "1"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, run, args)
    environment["VLLM_KVARN_DEFER_PREFILL_FLUSH"] = None
    environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] = "0"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, run, args)
    environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] = "16"

    for offload in (
        ["--cpu-offload-gb", "4"],
        ["--cpu-offload-gb=4"],
        ["--kv-offloading-size", "4"],
        ['--kv-transfer-config={"kv_connector":"LMCacheConnectorV1"}'],
        ["--offload-group-size", "8"],
    ):
        with pytest.raises(RunnerError, match="profile mismatch"):
            verify_service_profile([*argv, *offload], environment, run, args)

    reference = PlannedRun(run.workload, "reference", 3)
    argv[argv.index("--kv-cache-dtype") + 1] = "auto"
    for name in (
        "KVARN_NATIVE_XPU",
        "KVARN_NATIVE_XPU_DECODE",
        "KVARN_NATIVE_XPU_DPAS_LAYOUT",
        "KVARN_NATIVE_XPU_MATERIALIZE",
        "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH",
    ):
        environment[name] = "0"
    environment["KVARN_NATIVE_XPU_SPLITS"] = "1"
    verify_service_profile(argv, environment, reference, args)
    environment["KVARN_NATIVE_XPU_SPLITS"] = "16"
    with pytest.raises(RunnerError, match="profile mismatch"):
        verify_service_profile(argv, environment, reference, args)


def test_process_capture_includes_bounded_window_and_full_defer_state() -> None:
    environment = dict(os.environ)
    environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] = "16"
    environment["VLLM_KVARN_DEFER_PREFILL_FLUSH"] = "0"
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env=environment,
    )
    try:
        for _attempt in range(100):
            captured = runner._process_environment(process.pid)
            if captured["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] == "16":
                break
            time.sleep(0.01)
    finally:
        process.terminate()
        process.wait(timeout=5)

    assert captured["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] == "16"
    assert captured["VLLM_KVARN_DEFER_PREFILL_FLUSH"] == "0"


def test_service_environment_pins_window_and_scrubs_full_defer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", "999")
    monkeypatch.setenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", "1")
    monkeypatch.setenv("VLLM_UNPINNED_BEHAVIOR", "1")
    for name in (
        "LD_PRELOAD",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        monkeypatch.setenv(name, "/tmp/injected")

    environment = runner.service_environment(_args(tmp_path))

    assert environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] == "16"
    assert "VLLM_KVARN_DEFER_PREFILL_FLUSH" not in environment
    assert "VLLM_UNPINNED_BEHAVIOR" not in environment
    assert all(
        name not in environment
        for name in (
            "LD_PRELOAD",
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
        )
    )
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_native_log_allows_unrelated_fallback_but_rejects_kvarn_fallback(
    tmp_path: Path,
) -> None:
    engine_log = tmp_path / "engine.log"
    base = (
        "INFO config: device_config=xpu\n"
        "INFO Actual usage is 17.54 GiB for consumed memory (weights + non-torch), "
        "0.31 GiB for peak activation, and 0.0 GiB for CUDAGraph memory. "
        "Current kv cache memory in use is 10.92 GiB.\n"
        "INFO Falling back to the Triton GDN decode path\n"
        "WARNING sampler is Falling back to PyTorch-native implementation\n"
        "INFO Using the native Xe2 KVarN qlen=1 decoder\n"
    )
    engine_log.write_text(base, encoding="utf-8")

    assert runner.validate_engine_log(engine_log, native=True)["status"] == "passed"

    for kvarn_fallback in (
        "WARNING Kvarn decoder used a fallback path\n",
        "WARNING Falling back from Kvarn to a generic decoder\n",
    ):
        engine_log.write_text(base + kvarn_fallback, encoding="utf-8")
        with pytest.raises(RunnerError, match="Kvarn fallback"):
            runner.validate_engine_log(engine_log, native=True)


def test_execute_rejects_correctness_from_another_candidate(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.candidate_id = None
    args.correctness = _correctness(
        tmp_path / "correctness.json", "/nix/store/another-candidate"
    )

    with pytest.raises(RunnerError, match="differs from --candidate-env"):
        runner.execute(args)


def test_runner_binds_correctness_to_actual_service_closures() -> None:
    expected = {
        field: IDENTITY[field]
        for field in (
            "process_package",
            "candidate_closure_sha256",
            "process_closure_sha256",
        )
    }
    verify_correctness_candidate_identity(IDENTITY, expected)

    stale = {**expected, "process_closure_sha256": "c" * 64}
    with pytest.raises(RunnerError, match="different candidate builds"):
        verify_correctness_candidate_identity(IDENTITY, stale)


def test_sealed_results_are_directly_perf_gate_compatible(tmp_path: Path) -> None:
    args = _args(tmp_path)
    candidate_id = "candidate-store-path"
    correctness = _correctness(tmp_path / "correctness.json", candidate_id)
    correctness_sha256 = hashlib.sha256(correctness.read_bytes()).hexdigest()
    workload = Workload(4096, 4, 512, 8, 17)
    arms = ARM_ORDER * 4
    references: list[Path] = []
    candidates: list[Path] = []
    reference_logs: list[Path] = []
    candidate_logs: list[Path] = []
    records: list[dict[str, object]] = []

    for order, arm in enumerate(arms, start=1):
        run_dir = tmp_path / f"run-{order}"
        run_dir.mkdir()
        native = arm == "candidate"
        engine_log = run_dir / "engine.log"
        engine_log.write_text(
            "INFO config: device_config=xpu\n"
            "INFO Actual usage is 17.54 GiB for consumed memory. "
            "Current kv cache memory in use is 10.92 GiB.\n"
            + ("INFO Using the native Xe2 KVarN qlen=1 decoder\n" if native else ""),
            encoding="utf-8",
        )
        raw = _raw_result(
            run_dir / "benchmark.raw.json",
            throughput=99.0 if native else 100.0,
            workload=workload,
        )
        warmup_workload = Workload(4096, 4, 512, 4, 17)
        warmup_raw = _raw_result(
            run_dir / "warmup.raw.json",
            throughput=100.0,
            workload=warmup_workload,
        )
        warmup_result = run_dir / "warmup.json"
        persist_warmup_result(
            raw_result=warmup_raw,
            output=warmup_result,
            workload=warmup_workload,
            argv=[
                "vllm",
                "--random-input-len",
                "4096",
                "--random-output-len",
                "512",
                "--num-prompts",
                "4",
                "--num-warmups",
                "0",
                "--max-concurrency",
                "4",
                "--seed",
                "17",
            ],
            arm=arm,
            run_uuid=f"run-{order}",
            identity=IDENTITY,
            profile=PROFILE,
        )
        output = run_dir / "benchmark.json"
        sealed = seal_benchmark_result(
            raw_result=raw,
            output=output,
            engine_log=engine_log,
            run=PlannedRun(workload, arm, order),
            args=args,
            candidate_id=candidate_id,
            correctness_sha256=correctness_sha256,
            scheduler={"peak_running": 4},
            run_uuid=f"run-{order}",
            started_at=f"2026-08-31T00:00:{order:02d}Z",
            identity=IDENTITY,
            profile=PROFILE,
            warmup_result=warmup_result,
        )
        assert (
            sealed["kvarn_engine_log_sha256"]
            == hashlib.sha256(engine_log.read_bytes()).hexdigest()
        )
        assert sealed["kvarn_max_num_batched_tokens"] == "2048"
        assert sealed["kvarn_evidence_mode"] == "formal"
        if native:
            candidates.append(output)
            candidate_logs.append(engine_log)
        else:
            references.append(output)
            reference_logs.append(engine_log)
        records.append(
            {
                "arm": arm,
                "order": order,
                "result": str(output),
                "engine_log": str(engine_log),
            }
        )

    result = compare(
        references,
        candidates,
        reference_logs=reference_logs,
        candidate_logs=candidate_logs,
        correctness_path=correctness,
        comparison_kind="end-to-end",
        mode="match",
        min_throughput_ratio=0.95,
        min_request_decode_ratio=0.95,
        max_latency_ratio=1.10,
    )

    assert result["status"] == "passed"
    assert result["candidate_over_reference"]["output_throughput"] == pytest.approx(
        0.99
    )
    args.correctness = correctness
    args.min_throughput_ratio = 0.95
    args.min_request_decode_ratio = 0.95
    args.max_latency_ratio = 1.10
    args.parity_ratio = 0.98
    args.min_parity_pairs = 4
    hardened = gate_workload(records, args)
    assert hardened["hard_floor"]["status"] == "passed"
    assert hardened["statistical_parity"]["status"] == "passed"


def test_exploratory_seal_omits_formal_correctness_provenance(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.exploratory = True
    workload = Workload(4096, 1, 4, 1, 17)
    raw = _raw_result(
        tmp_path / "benchmark.raw.json", throughput=100.0, workload=workload
    )
    engine_log = tmp_path / "engine.log"
    engine_log.write_text(
        "INFO config: device_config=xpu\n"
        "INFO Actual usage is 17.54 GiB for consumed memory. "
        "Current kv cache memory in use is 10.92 GiB.\n",
        encoding="utf-8",
    )

    sealed = seal_benchmark_result(
        raw_result=raw,
        output=tmp_path / "benchmark.json",
        engine_log=engine_log,
        run=PlannedRun(workload, "reference", 1),
        args=args,
        candidate_id=None,
        correctness_sha256=None,
        scheduler={"peak_running": 1},
        run_uuid="exploratory-run",
        started_at="2026-09-03T00:00:00Z",
        identity=IDENTITY,
        profile=PROFILE,
        warmup_result=None,
    )

    assert sealed["kvarn_evidence_mode"] == "exploratory"
    assert sealed["kvarn_promotable"] is False
    assert "kvarn_candidate_id" not in sealed
    assert "kvarn_correctness_sha256" not in sealed


def test_exploratory_summary_is_descriptive_and_non_promotable(
    tmp_path: Path,
) -> None:
    records: list[dict[str, object]] = []
    for order, arm in enumerate(ARM_ORDER, start=1):
        document = _parity_document(arm, 90.0 if arm == "candidate" else 100.0)
        document["kvarn_evidence_mode"] = "exploratory"
        document["kvarn_promotable"] = False
        path = tmp_path / f"{order}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        records.append({"arm": arm, "order": order, "result": str(path)})

    summary = summarize_exploratory_workload(records)

    assert summary["evidence_mode"] == "exploratory"
    assert summary["promotable"] is False
    assert summary["formal_performance_gate_run"] is False
    assert summary["formal_statistical_parity_run"] is False
    assert summary["repeats_per_arm"] == 2
    assert summary["abba_blocks"] == 1
    assert summary["descriptive_metrics"]["output_throughput"][
        "candidate_over_reference"
    ] == pytest.approx(0.9)
    assert (
        result_exit_code(
            {
                "evidence_mode": "exploratory",
                "performance_status": "failed",
                "statistical_parity_status": "failed",
            }
        )
        == 0
    )
    assert (
        result_exit_code(
            {
                "evidence_mode": "formal",
                "performance_status": "failed",
                "statistical_parity_status": "failed",
            }
        )
        == 1
    )


def _parity_document(
    arm: str,
    output_throughput: float,
    request_rate: float | None = None,
    *,
    ttft: float = 0.1,
    itl: float = 0.05,
) -> dict[str, object]:
    interval = 1.0 / (request_rate or output_throughput)
    return {
        "kvarn_arm": arm,
        "output_throughput": output_throughput,
        "itls": [[interval] * 3],
        "ttfts": [ttft],
        "kvarn_process_package": "/nix/store/package",
        "kvarn_process_closure_sha256": "1" * 64,
        "kvarn_candidate_closure_sha256": "2" * 64,
        "kvarn_max_num_batched_tokens": "2048",
        "kvarn_matched_profile_sha256": "3" * 64,
        "kvarn_accelerator": "xpu",
        "kvarn_xpu_available": "1",
        "kvarn_xpu_device_count": "1",
        "kvarn_xpu_device_name": "Intel(R) Arc(TM) Pro B70 Graphics",
        "kvarn_xpu_compute_probe": "passed",
        "kvarn_hardware_preflight_path": "/evidence/hardware.json",
        "kvarn_hardware_preflight_sha256": "4" * 64,
    }


def test_paired_parity_is_one_sided_and_requires_enough_abba_pairs() -> None:
    insufficient = paired_noninferiority([1.02, 1.03], threshold=0.98, minimum_pairs=4)
    assert insufficient["status"] == "insufficient_evidence"
    assert insufficient["lower_confidence_bound"] is None

    winning_documents: list[dict[str, object]] = []
    for candidate in (1.03, 1.04, 1.05, 1.06):
        winning_documents.extend(
            [
                _parity_document("reference", 100.0, 100.0),
                _parity_document("candidate", candidate * 100.0, candidate * 100.0),
                _parity_document("candidate", candidate * 100.0, candidate * 100.0),
                _parity_document("reference", 100.0, 100.0),
            ]
        )
    first = statistical_parity(winning_documents, threshold=0.98, minimum_pairs=4)
    second = statistical_parity(winning_documents, threshold=0.98, minimum_pairs=4)
    assert first == second
    assert first["status"] == "passed"
    assert first["metrics"]["output_throughput"]["lower_confidence_bound"] > 1.0

    regressed = [dict(document) for document in winning_documents]
    for document in regressed:
        if document["kvarn_arm"] == "candidate":
            document["output_throughput"] = 96.0
            document["itls"] = [[1.0 / 96.0] * 3]
    assert (
        statistical_parity(regressed, threshold=0.98, minimum_pairs=4)["status"]
        == "failed"
    )


def test_pooled_p99_does_not_hide_one_bad_repeat() -> None:
    reference = [_parity_document("reference", 100.0) for _ in range(4)]
    candidate = [_parity_document("candidate", 100.0) for _ in range(4)]
    for document in reference:
        document["ttfts"] = [0.1] * 100
        document["itls"] = [[0.05] * 100]
    for document in candidate:
        document["ttfts"] = [0.1] * 100
        document["itls"] = [[0.05] * 100]
    candidate[-1]["ttfts"] = [0.1] * 50 + [1.0] * 50
    candidate[-1]["itls"] = [[0.05] * 50 + [0.5] * 50]

    pooled = pooled_tail_latency(reference, candidate)

    assert pooled["ttft"]["candidate_over_reference"] == pytest.approx(10.0)
    assert pooled["itl"]["candidate_over_reference"] == pytest.approx(10.0)


def test_matched_profile_normalizes_only_declared_arm_differences(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    argv = [
        "/nix/store/package/bin/.vllm-wrapped",
        "serve",
        MODEL,
        "--kv-cache-dtype",
        "auto",
        "--max-num-batched-tokens",
        "2048",
        "--api-key",
        "do-not-persist",
    ]
    environment = {
        "KVARN_NATIVE_XPU": "0",
        "KVARN_NATIVE_XPU_DPAS_LAYOUT": "0",
        "KVARN_NATIVE_XPU_SPLITS": "1",
        "HF_HOME": str(args.hf_home),
    }
    reference = service_profile_evidence(argv, environment)
    argv[argv.index("--kv-cache-dtype") + 1] = "kvarn_k4v4_g128_compact"
    environment["KVARN_NATIVE_XPU"] = "1"
    environment["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "0"
    environment["KVARN_NATIVE_XPU_SPLITS"] = "24"
    candidate = service_profile_evidence(argv, environment)

    assert (
        reference["canonical_matched_profile_sha256"]
        == candidate["canonical_matched_profile_sha256"]
    )
    assert "do-not-persist" not in json.dumps(reference)
    assert reference["max_num_batched_tokens"] == "2048"
    assert "2048" in reference["redacted_argv"]
    assert "KVARN_NATIVE_XPU_SPLITS" in reference["allowed_arm_environment_differences"]
    assert (
        "KVARN_NATIVE_XPU_DPAS_LAYOUT"
        in reference["allowed_arm_environment_differences"]
    )

    argv.extend(["--block-size", "64"])
    changed = service_profile_evidence(argv, environment)
    assert (
        changed["canonical_matched_profile_sha256"]
        != candidate["canonical_matched_profile_sha256"]
    )
    del argv[-2:]
    argv[argv.index("--max-num-batched-tokens") + 1] = "8192"
    changed = service_profile_evidence(argv, environment)
    assert (
        changed["canonical_matched_profile_sha256"]
        != candidate["canonical_matched_profile_sha256"]
    )
    argv[argv.index("--max-num-batched-tokens") + 1] = "2048"
    environment["HF_HOME"] = "/different-cache"
    changed = service_profile_evidence(argv, environment)
    assert (
        changed["canonical_matched_profile_sha256"]
        != candidate["canonical_matched_profile_sha256"]
    )


def test_native_split_parser_defaults_and_requires_exact_batch_mapping() -> None:
    assert runner._parse_native_splits(None, [1, 4]) == {1: 24, 4: 16}
    assert runner._parse_native_splits(["8"], [1, 4]) == {1: 8, 4: 8}
    assert runner._parse_native_splits(["1=24", "4=16"], [1, 4]) == {
        1: 24,
        4: 16,
    }
    with pytest.raises(argparse.ArgumentTypeError, match="exactly match"):
        runner._parse_native_splits(["1=24"], [1, 4])
    with pytest.raises(argparse.ArgumentTypeError, match="cannot mix"):
        runner._parse_native_splits(["8", "4=16"], [1, 4])
    with pytest.raises(argparse.ArgumentTypeError, match="unsupported"):
        runner._parse_native_splits(["3"], [1])


def test_reference_keeps_neutral_split_while_candidate_uses_tuned_split(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    workload = Workload(4096, 1, 512, 2, 17)
    reference = PlannedRun(workload, "reference", 1)
    candidate = PlannedRun(workload, "candidate", 2)

    assert native_splits_for_run(reference, args) == 1
    assert native_splits_for_run(candidate, args) == 24


def test_matched_result_digest_mismatch_is_rejected() -> None:
    documents = [
        _parity_document("reference", 100.0),
        _parity_document("candidate", 100.0),
    ]
    validate_matched_results(documents)
    documents[1]["kvarn_process_closure_sha256"] = "4" * 64
    with pytest.raises(RunnerError, match="effective provenance"):
        validate_matched_results(documents)
    documents[1]["kvarn_process_closure_sha256"] = "1" * 64
    documents[1]["kvarn_max_num_batched_tokens"] = "8192"
    with pytest.raises(RunnerError, match="effective provenance"):
        validate_matched_results(documents)


def test_candidate_identity_hashes_sorted_actual_process_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    process_package = "/nix/store/process-package"

    def fake_run(command: list[str], **_kwargs: object) -> SimpleCompleted:
        queried = command[-1]
        if queried == str(candidate):
            return SimpleCompleted(f"/nix/store/z\n{process_package}\n/nix/store/a\n")
        assert queried == process_package
        return SimpleCompleted("/nix/store/z\n/nix/store/a\n/nix/store/z\n")

    class SimpleCompleted:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(subprocess, "run", fake_run)
    identity = verify_candidate_identity(
        [f"{process_package}/bin/.vllm-wrapped", "serve"], candidate
    )

    expected = hashlib.sha256(b"/nix/store/a\n/nix/store/z\n").hexdigest()
    assert identity["process_package"] == process_package
    assert identity["process_closure_paths"] == ["/nix/store/a", "/nix/store/z"]
    assert identity["process_closure_sha256"] == expected


def test_launchers_are_resolved_once_to_immutable_programs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.config_repo.mkdir()
    plan = [
        PlannedRun(Workload(4096, 1, 4, 1, 17), "reference", 1),
        PlannedRun(Workload(4096, 1, 4, 1, 17), "reference", 4),
    ]
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> SimpleCompleted:
        commands.append(command)
        stdout = json.dumps(
            {
                "program": "/logical-launcher/bin/run",
                "context": {
                    "/nix/store/launcher.drv": {"outputs": ["out"]},
                },
            }
            if "eval" in command
            else [
                {
                    "drvPath": "/nix/store/launcher.drv",
                    "outputs": {"out": "/nix/store/launcher"},
                }
            ]
        )
        return SimpleCompleted(stdout)

    class SimpleCompleted:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "is_file", lambda _self: True)

    resolved = resolve_launchers(plan, args)
    args.resolved_launchers = resolved

    assert len(commands) == 2
    assert all(command[2:4] == ["--store", "daemon"] for command in commands)
    assert "--json" in commands[0]
    assert "builtins.getContext" in commands[1][-1]
    assert service_command(plan[0], args) == [
        "/nix/store/launcher/bin/run",
        str(args.candidate_env),
        "--max-num-batched-tokens",
        "2048",
    ]


def test_launcher_resolution_rejects_mismatched_app_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    args.config_repo.mkdir()
    plan = [PlannedRun(Workload(4096, 1, 4, 1, 17), "reference", 1)]

    def fake_run(command: list[str], **_kwargs: object) -> SimpleCompleted:
        result: object
        if "eval" in command:
            result = {
                "program": "/logical-launcher/bin/run",
                "context": {
                    "/nix/store/different.drv": {"outputs": ["out"]},
                },
            }
        else:
            result = [
                {
                    "drvPath": "/nix/store/launcher.drv",
                    "outputs": {"out": "/nix/store/launcher"},
                }
            ]
        return SimpleCompleted(json.dumps(result))

    class SimpleCompleted:
        def __init__(self, stdout: str) -> None:
            self.stdout = stdout

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RunnerError, match="package and app program derivations differ"):
        resolve_launchers(plan, args)


def test_warmup_result_is_validated_and_persisted(tmp_path: Path) -> None:
    workload = Workload(4096, 1, 4, 1, 17)
    raw = _raw_result(tmp_path / "warmup.raw.json", throughput=100.0, workload=workload)
    output = tmp_path / "warmup.json"

    result = persist_warmup_result(
        raw_result=raw,
        output=output,
        workload=workload,
        argv=["vllm", "--api-key", "secret"],
        arm="reference",
        run_uuid="warmup-run",
        identity=IDENTITY,
        profile=PROFILE,
    )

    assert result["status"] == "passed"
    assert result["raw_result_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    assert result["process_closure_sha256"] == IDENTITY["process_closure_sha256"]
    assert "secret" not in output.read_text(encoding="utf-8")


def test_signal_supervisor_forwards_signal_to_all_owned_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        runner,
        "_signal_process_group",
        lambda group, selected_signal: calls.append((group, selected_signal)),
    )
    supervisor = ProcessSupervisor()
    supervisor.register(101, "service")
    supervisor.register(202, "benchmark")

    with pytest.raises(runner.RunnerInterrupted, match="SIGTERM"):
        supervisor._handle_signal(signal.SIGTERM, None)

    assert calls == [(101, signal.SIGTERM), (202, signal.SIGTERM)]


def test_managed_process_uses_new_group_and_unregisters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class FakeProcess:
        pid = 123

        def wait(self, timeout: float) -> int:
            observed["timeout"] = timeout
            return 0

    def fake_popen(command: object, **kwargs: object) -> FakeProcess:
        observed["command"] = command
        observed.update(kwargs)
        return FakeProcess()

    supervisor = ProcessSupervisor()
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: 456)
    monkeypatch.setattr(
        runner, "_wait_for_process_group", lambda _group, _timeout: True
    )
    with (tmp_path / "stdout.log").open("w", encoding="utf-8") as output:
        returncode = run_managed_process(
            ["fake-benchmark"],
            cwd=tmp_path,
            environment={},
            output=output,
            timeout=12.0,
            supervisor=supervisor,
            label="fake",
        )

    assert returncode == 0
    assert observed["start_new_session"] is True
    assert supervisor._groups == {}
