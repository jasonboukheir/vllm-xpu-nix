from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
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
    resolve_launchers,
    run_managed_process,
    seal_benchmark_result,
    service_command,
    service_profile_evidence,
    statistical_parity,
    validate_matched_results,
    verify_candidate_identity,
    verify_service_profile,
    warmup_command,
)

MODEL = "model-repo"
REVISION = "6b0622f4354481d5d04577d48ba0db844efc1330"
IDENTITY = {
    "process_executable": "/nix/store/package/bin/.vllm-wrapped",
    "process_package": "/nix/store/package",
    "process_closure_sha256": "1" * 64,
    "candidate_closure_sha256": "2" * 64,
}
PROFILE = {
    "max_num_batched_tokens": "2048",
    "canonical_matched_profile_sha256": "3" * 64,
}


def _args(tmp_path: Path) -> argparse.Namespace:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    (candidate / "bin" / "vllm").write_text("", encoding="utf-8")
    return argparse.Namespace(
        base_url="http://127.0.0.1:8000",
        candidate_env=candidate,
        config_ref="path:/config",
        config_repo=tmp_path / "config",
        max_model_len=65536,
        max_num_batched_tokens=2048,
        model=MODEL,
        model_revision=REVISION,
        native_splits={1: 24, 4: 16},
        num_warmups=None,
        hf_home=Path("/var/cache/huggingface"),
        runtime_cache=tmp_path / "runtime-cache",
        served_model="sunny-chat",
    )


def _correctness(path: Path, candidate_id: str) -> Path:
    names = (
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
    for name in names:
        evidence = path.parent / f"{name}.json"
        evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
        gates[name] = {
            "status": "passed",
            "path": str(evidence),
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "candidate_id": candidate_id,
                "native_dispatch_verified": True,
                "gates": gates,
            }
        ),
        encoding="utf-8",
    )
    return path


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
    correctness = tmp_path / "correctness.json"
    correctness.write_text("{}\n", encoding="utf-8")
    common = [
        "--candidate-env",
        str(candidate),
        "--correctness",
        str(correctness),
        "--allow-tmp",
        "--plan-only",
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
        "KVARN_NATIVE_XPU_DPAS_LAYOUT": "1",
        "KVARN_NATIVE_XPU_MATERIALIZE": "1",
        "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH": "1",
        "KVARN_NATIVE_XPU_SPLITS": "16",
        "VLLM_CACHE_ROOT": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_XPU_ENABLE_XPU_GRAPH": None,
        "XDG_CACHE_HOME": str(args.runtime_cache),
    }

    verify_service_profile(argv, environment, run, args)
    assert native_splits_for_run(run, args) == 16
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


def test_sealed_results_are_directly_perf_gate_compatible(tmp_path: Path) -> None:
    args = _args(tmp_path)
    candidate_id = "candidate-store-path"
    correctness = _correctness(tmp_path / "correctness.json", candidate_id)
    correctness_sha256 = hashlib.sha256(correctness.read_bytes()).hexdigest()
    workload = Workload(4096, 4, 4, 4, 17)
    arms = ARM_ORDER * 2
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
            "INFO engine ready\n"
            + ("INFO Using the native Xe2 KVarN qlen=1 decoder\n" if native else ""),
            encoding="utf-8",
        )
        raw = _raw_result(
            run_dir / "benchmark.raw.json",
            throughput=97.0 if native else 100.0,
            workload=workload,
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
        )
        assert (
            sealed["kvarn_engine_log_sha256"]
            == hashlib.sha256(engine_log.read_bytes()).hexdigest()
        )
        assert sealed["kvarn_max_num_batched_tokens"] == "2048"
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
        0.97
    )
    args.correctness = correctness
    args.min_throughput_ratio = 0.95
    args.min_request_decode_ratio = 0.95
    args.max_latency_ratio = 1.10
    args.parity_ratio = 0.98
    args.min_parity_pairs = 4
    hardened = gate_workload(records, args)
    assert hardened["hard_floor"]["status"] == "passed"
    assert hardened["statistical_parity"]["status"] == "insufficient_evidence"


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
    environment["KVARN_NATIVE_XPU_DPAS_LAYOUT"] = "1"
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
    )

    assert result["status"] == "passed"
    assert result["raw_result_sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
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
