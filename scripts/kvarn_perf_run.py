#!/usr/bin/env python3
"""Run and seal matched foreground auto/native Kvarn performance trials.

The service is restarted for every recorded trial.  A vLLM result is only
annotated after the service has stopped and its engine log has a final digest.
This makes every emitted ``benchmark.json`` directly consumable by
``kvarn_perf_gate.py`` without claiming a digest of a live log.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

try:
    from scripts.kvarn_perf_gate import GateError, compare
    from scripts.kvarn_scan_engine_log import scan, xpu_runtime_evidence
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from kvarn_perf_gate import GateError, compare
    from kvarn_scan_engine_log import scan, xpu_runtime_evidence


NATIVE_DISPATCH = "Using the native Xe2 KVarN qlen=1 decoder"
COMPACT_DTYPE = "kvarn_k4v4_g128_compact"
NATIVE_LAYOUTS = ("natural", "xe2_dpas")
NATIVE_LAYOUT_ENV = {"natural": "0", "xe2_dpas": "1"}
VARIANT_FIELDS = (
    "kernel_strategy",
    "split_policy",
    "fusion_strategy",
    "scheduling_variant",
    "variant_id",
)
DEFAULT_CONTEXTS = (4096, 16384, 32768, 65023)
DEFAULT_BATCHES = (1, 4)
DEFAULT_NATIVE_SPLITS = {1: 24, 4: 16}
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_PREFILL_WINDOW_BLOCKS = 16
EXPECTED_XPU_DEVICE_NAME = "Intel(R) Arc(TM) Pro B70 Graphics"
DEFAULT_MODEL = (
    "jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound"
)
DEFAULT_MODEL_REVISION = "6b0622f4354481d5d04577d48ba0db844efc1330"
REFERENCE_NATIVE_SPLITS = 1
SUPPORTED_NATIVE_SPLITS = frozenset({1, 2, 4, 8, 16, 17, 24, 32})
ARM_ORDER = ("reference", "candidate", "candidate", "reference")
ARM_SETTINGS = {
    "reference": {
        "launcher": "vllm-xpu-brutus-auto-b{batch}",
        "kv_cache_dtype": "auto",
        "native_xpu": "0",
    },
    "candidate": {
        "launcher": "vllm-xpu-brutus-kvarn-native-b{batch}",
        "kv_cache_dtype": COMPACT_DTYPE,
        "native_xpu": "1",
    },
}
CAPTURED_ENVIRONMENT = (
    "CCL_ATL_TRANSPORT",
    "CCL_LOG_LEVEL",
    "CCL_PROCESS_LAUNCHER",
    "CCL_ZE_IPC_EXCHANGE",
    "HF_HOME",
    "HOME",
    "KVARN_NATIVE_XPU",
    "KVARN_NATIVE_XPU_DECODE",
    "KVARN_NATIVE_XPU_DPAS_LAYOUT",
    "KVARN_NATIVE_XPU_MATERIALIZE",
    "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH",
    "KVARN_NATIVE_XPU_SPLITS",
    "KVARN_PREFILL_FP16_WINDOW_BLOCKS",
    "VLLM_CACHE_ROOT",
    "VLLM_TARGET_DEVICE",
    "VLLM_KVARN_DEFER_PREFILL_FLUSH",
    "VLLM_XPU_ENABLE_XPU_GRAPH",
    "XDG_CACHE_HOME",
)
SCRUBBED_ENVIRONMENT = (
    "CC",
    "CMPLR_ROOT",
    "CXX",
    "LD_LIBRARY_PATH",
    "LEVEL_ZERO_V1_SDK_PATH",
    "LIBRARY_PATH",
    "LD_PRELOAD",
    "ONEAPI_ROOT",
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "SYCL_HOME",
    "VLLM_KVARN_DEFER_PREFILL_FLUSH",
    "VLLM_XPU_ENABLE_XPU_GRAPH",
)
FALLBACK_PATTERN = re.compile(
    r"(?i)(?:\bkvarn\b[^\n]{0,120}\b(?:fallback|falling back)\b|"
    r"\b(?:fallback|falling back)\b[^\n]{0,120}\bkvarn\b)"
)
RUNNING_METRIC = "vllm:num_requests_running"
PARITY_METRICS = (
    "output_throughput",
    "median_request_decode_throughput",
)
ARM_ARGUMENTS = {"--kv-cache-dtype"}
ARM_ENVIRONMENT = {
    "KVARN_NATIVE_XPU",
    "KVARN_NATIVE_XPU_DECODE",
    "KVARN_NATIVE_XPU_DPAS_LAYOUT",
    "KVARN_NATIVE_XPU_MATERIALIZE",
    "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH",
    "KVARN_NATIVE_XPU_SPLITS",
}
SENSITIVE_NAME = re.compile(
    r"(?i)(?:authorization|bearer|credential|password|secret|token|api[_-]?key)"
)
PUBLIC_TOKEN_ARGUMENTS = {"--max-num-batched-tokens"}
# 95th percentiles for a one-sided Student t interval, indexed by degrees of
# freedom.  A one-sided lower interval is the appropriate test here: a result
# above auto is a success, not an equivalence failure at an arbitrary upper
# bound.
ONE_SIDED_T_95 = {
    1: 6.314,
    2: 2.920,
    3: 2.353,
    4: 2.132,
    5: 2.015,
    6: 1.943,
    7: 1.895,
    8: 1.860,
    9: 1.833,
    10: 1.812,
    11: 1.796,
    12: 1.782,
    13: 1.771,
    14: 1.761,
    15: 1.753,
    16: 1.746,
    17: 1.740,
    18: 1.734,
    19: 1.729,
    20: 1.725,
    21: 1.721,
    22: 1.717,
    23: 1.714,
    24: 1.711,
    25: 1.708,
    26: 1.706,
    27: 1.703,
    28: 1.701,
    29: 1.699,
    30: 1.697,
}


class RunnerError(RuntimeError):
    """Raised when a trial cannot be made valid and reproducible."""


class RunnerInterrupted(RunnerError):
    """Raised after forwarding a controlling signal to every owned child."""


class ProcessSupervisor:
    """Track independent child process groups and fail closed on interruption."""

    def __init__(self) -> None:
        self._groups: dict[int, str] = {}
        self._lock = threading.Lock()
        self._previous_handlers: dict[signal.Signals, Any] = {}

    def register(self, process_group: int, label: str) -> None:
        with self._lock:
            self._groups[process_group] = label

    def unregister(self, process_group: int) -> None:
        with self._lock:
            self._groups.pop(process_group, None)

    def signal_all(self, selected_signal: signal.Signals) -> None:
        with self._lock:
            groups = tuple(self._groups)
        for process_group in groups:
            _signal_process_group(process_group, selected_signal)

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        selected_signal = signal.Signals(signum)
        self.signal_all(selected_signal)
        raise RunnerInterrupted(f"received {selected_signal.name}")

    def install_signal_handlers(self) -> None:
        for selected_signal in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            self._previous_handlers[selected_signal] = signal.getsignal(selected_signal)
            signal.signal(selected_signal, self._handle_signal)

    def restore_signal_handlers(self) -> None:
        for selected_signal, handler in self._previous_handlers.items():
            signal.signal(selected_signal, handler)
        self._previous_handlers.clear()


@dataclasses.dataclass(frozen=True)
class Workload:
    context: int
    batch: int
    output_tokens: int
    num_prompts: int
    seed: int

    @property
    def workload_id(self) -> str:
        return (
            f"random-in{self.context}-out{self.output_tokens}"
            f"-b{self.batch}-n{self.num_prompts}"
        )

    @property
    def service_profile(self) -> str:
        return f"qwen38-65k-b{self.batch}-eager"


@dataclasses.dataclass(frozen=True)
class PlannedRun:
    workload: Workload
    arm: str
    order: int


@dataclasses.dataclass
class ServiceProcess:
    process: subprocess.Popen[bytes]
    process_group: int
    log_stream: TextIO
    log_path: Path
    engine_pid: int
    argv: list[str]
    environment: dict[str, str | None]
    supervisor: ProcessSupervisor


XPU_PREFLIGHT_CODE = """
import json
import torch

available = bool(torch.xpu.is_available())
count = int(torch.xpu.device_count()) if available else 0
names = [torch.xpu.get_device_name(index) for index in range(count)]
probe_device = None
probe_value = None
if available and count:
    value = torch.arange(4, dtype=torch.float32, device="xpu:0")
    torch.xpu.synchronize()
    probe_device = str(value.device)
    probe_value = float(value.sum().cpu().item())
print(json.dumps({
    "schema_version": 1,
    "torch_version": torch.__version__,
    "xpu_available": available,
    "xpu_device_count": count,
    "xpu_device_names": names,
    "probe_device": probe_device,
    "probe_value": probe_value,
}, sort_keys=True))
"""


def utc_timestamp() -> str:
    return dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def ensure_durable(path: Path, *, allow_tmp: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not allow_tmp and resolved.is_relative_to(Path("/tmp")):
        raise RunnerError(f"result root must be durable (outside /tmp): {resolved}")
    return resolved


def abba_arms(repeats: int, *, minimum_repeats: int = 4) -> tuple[str, ...]:
    if repeats < minimum_repeats or repeats % 2:
        raise RunnerError(
            f"--repeats must be an even integer of at least {minimum_repeats}"
        )
    return ARM_ORDER * (repeats // 2)


def build_plan(
    *,
    contexts: Sequence[int],
    batches: Sequence[int],
    output_tokens: int,
    waves_per_run: int,
    repeats: int,
    seed: int,
    max_model_len: int,
    minimum_repeats: int = 4,
) -> list[PlannedRun]:
    if not contexts or not batches:
        raise RunnerError("at least one context and batch size are required")
    if output_tokens < 2 or waves_per_run < 1:
        raise RunnerError("output tokens must be >=2 and waves per run must be >=1")
    invalid_batches = sorted(set(batches) - {1, 4})
    if invalid_batches:
        raise RunnerError(f"only B1 and B4 are supported, got {invalid_batches}")
    for context in contexts:
        if context < 1 or context + output_tokens > max_model_len:
            raise RunnerError(
                f"context {context} plus {output_tokens} output tokens exceeds "
                f"max model length {max_model_len}"
            )

    arms = abba_arms(repeats, minimum_repeats=minimum_repeats)
    plan: list[PlannedRun] = []
    for batch in batches:
        for context in contexts:
            workload = Workload(
                context=context,
                batch=batch,
                output_tokens=output_tokens,
                num_prompts=batch * waves_per_run,
                seed=seed,
            )
            plan.extend(
                PlannedRun(workload=workload, arm=arm, order=order)
                for order, arm in enumerate(arms, start=1)
            )
    return plan


def native_splits_for_run(run: PlannedRun, args: argparse.Namespace) -> int:
    """Return the verified effective split count for one service arm."""
    if run.arm == "reference":
        return REFERENCE_NATIVE_SPLITS
    try:
        return int(args.native_splits[run.workload.batch])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError(
            f"no native split count configured for B{run.workload.batch}"
        ) from exc


def native_layout_for_run(run: PlannedRun, args: argparse.Namespace) -> str:
    """The auto reference is always natural; only the native candidate varies."""
    return "natural" if run.arm == "reference" else args.native_layout


def variant_provenance_for_run(
    run: PlannedRun, args: argparse.Namespace
) -> dict[str, str]:
    scheduling = f"eager_mnbt{args.max_num_batched_tokens}"
    if run.arm == "reference":
        return {
            "kernel_strategy": "vllm_auto",
            "split_policy": "neutral_1",
            "fusion_strategy": "vllm_auto",
            "scheduling_variant": scheduling,
            "variant_id": f"auto-control-{scheduling}",
        }
    split_policy = "fixed_" + "_".join(
        f"b{batch}s{splits}" for batch, splits in sorted(args.native_splits.items())
    )
    return {
        "kernel_strategy": "native_xe2_qlen1",
        "split_policy": split_policy,
        "fusion_strategy": "native_materializer_persistent_scratch",
        "scheduling_variant": scheduling,
        "variant_id": f"native-xe2-{args.native_layout}-{split_policy}-{scheduling}",
    }


def load_correctness(
    path: Path, explicit_candidate_id: str | None
) -> tuple[str, str, dict[str, str], str, dict[str, str]]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot load correctness artifact {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("status") != "passed":
        raise RunnerError("correctness artifact status must be passed")
    candidate_id = document.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise RunnerError("correctness artifact candidate_id must be non-empty")
    if explicit_candidate_id is not None and explicit_candidate_id != candidate_id:
        raise RunnerError("--candidate-id differs from the correctness artifact")
    raw_identity = document.get("candidate_identity")
    if not isinstance(raw_identity, dict):
        raise RunnerError("correctness artifact candidate_identity must be an object")
    identity: dict[str, str] = {}
    for field in (
        "process_package",
        "candidate_closure_sha256",
        "process_closure_sha256",
    ):
        value = raw_identity.get(field)
        if not isinstance(value, str) or not value:
            raise RunnerError(f"correctness artifact candidate_identity lacks {field}")
        if field.endswith("_sha256") and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RunnerError(
                f"correctness artifact candidate_identity {field} is not SHA-256"
            )
        identity[field] = value
    native_layout = document.get("native_layout")
    if native_layout not in NATIVE_LAYOUTS:
        raise RunnerError(
            "correctness artifact native_layout must be natural or xe2_dpas"
        )
    variant_provenance: dict[str, str] = {}
    for field in VARIANT_FIELDS:
        value = document.get(field)
        if not isinstance(value, str) or not value:
            raise RunnerError(f"correctness artifact lacks {field}")
        variant_provenance[field] = value
    return (
        candidate_id,
        hashlib.sha256(raw).hexdigest(),
        identity,
        native_layout,
        variant_provenance,
    )


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def repository_state(name: str, path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    try:
        status = _git(
            resolved, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        return {
            "name": name,
            "path": str(resolved),
            "head": _git(resolved, "rev-parse", "HEAD"),
            "branch": _git(resolved, "branch", "--show-current") or None,
            "dirty": bool(status),
            "status_porcelain": status,
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"name": name, "path": str(resolved), "error": str(exc)}


def runner_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(("KVARN_", "VLLM_")) or name in SCRUBBED_ENVIRONMENT:
            environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["XDG_CACHE_HOME"] = str(args.runtime_cache)
    environment["HF_HOME"] = str(args.hf_home)
    return environment


def service_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = runner_environment(args)
    environment["KVARN_PREFILL_FP16_WINDOW_BLOCKS"] = str(DEFAULT_PREFILL_WINDOW_BLOCKS)
    return environment


def validate_xpu_preflight(probe: Mapping[str, Any]) -> None:
    expected = {
        "xpu_available": True,
        "xpu_device_count": 1,
        "xpu_device_names": [EXPECTED_XPU_DEVICE_NAME],
        "probe_device": "xpu:0",
        "probe_value": 6.0,
    }
    mismatches = {
        name: {"actual": probe.get(name), "expected": value}
        for name, value in expected.items()
        if probe.get(name) != value
    }
    if probe.get("xpu_available") is not True:
        mismatches["xpu_available"] = {
            "actual": probe.get("xpu_available"),
            "expected": True,
        }
    if isinstance(probe.get("xpu_device_count"), bool):
        mismatches["xpu_device_count"] = {
            "actual": probe.get("xpu_device_count"),
            "expected": 1,
        }
    if isinstance(probe.get("probe_value"), bool):
        mismatches["probe_value"] = {
            "actual": probe.get("probe_value"),
            "expected": 6.0,
        }
    if mismatches:
        raise RunnerError(
            "performance qualification requires one Intel Arc Pro B70 XPU: "
            + json.dumps(mismatches, sort_keys=True)
        )


def probe_xpu_hardware(args: argparse.Namespace) -> dict[str, Any]:
    """Run a real XPU tensor operation using the candidate's pinned Torch."""
    python = args.candidate_env / "bin" / "python"
    try:
        completed = subprocess.run(
            [str(python), "-c", XPU_PREFLIGHT_CODE],
            cwd=args.packaging_repo,
            env=runner_environment(args),
            check=True,
            capture_output=True,
            text=True,
        )
        probe = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Intel XPU compute preflight failed: {exc}") from exc
    if not isinstance(probe, dict):
        raise RunnerError("Intel XPU compute preflight returned a non-object")
    validate_xpu_preflight(probe)
    return probe


def launcher_name(run: PlannedRun, args: argparse.Namespace) -> str:
    if run.arm == "candidate" and args.native_layout == "xe2_dpas":
        return f"vllm-xpu-brutus-kvarn-native-dpas-b{run.workload.batch}"
    return ARM_SETTINGS[run.arm]["launcher"].format(batch=run.workload.batch)


def resolve_launchers(
    plan: Sequence[PlannedRun], args: argparse.Namespace
) -> dict[str, str]:
    """Resolve mutable flake apps to verified physical daemon-store programs."""
    resolved: dict[str, str] = {}
    for launcher in dict.fromkeys(launcher_name(run, args) for run in plan):
        installable = f"{args.config_ref}#{launcher}"
        app_installable = f"{args.config_ref}#apps.x86_64-linux.{launcher}"
        try:
            build = subprocess.run(
                [
                    "nix",
                    "build",
                    "--store",
                    "daemon",
                    "--no-link",
                    "--json",
                    installable,
                ],
                cwd=args.config_repo,
                env=runner_environment(args),
                check=True,
                capture_output=True,
                text=True,
            )
            app = subprocess.run(
                [
                    "nix",
                    "eval",
                    "--store",
                    "daemon",
                    "--json",
                    app_installable,
                    "--apply",
                    (
                        "app: { program = app.program; "
                        "context = builtins.getContext app.program; }"
                    ),
                ],
                cwd=args.config_repo,
                env=runner_environment(args),
                check=True,
                capture_output=True,
                text=True,
            )
            build_result = json.loads(build.stdout)
            app_result = json.loads(app.stdout)
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
            hint = (
                "; xe2_dpas requires dedicated Brutus native-dpas launcher "
                "outputs that export KVARN_NATIVE_XPU_DPAS_LAYOUT=1"
                if "native-dpas" in launcher
                else ""
            )
            raise RunnerError(
                f"cannot resolve launcher {installable}{hint}: {exc}"
            ) from exc

        if not isinstance(build_result, list) or len(build_result) != 1:
            raise RunnerError(
                f"launcher build must return one derivation: {build_result!r}"
            )
        build_entry = build_result[0]
        if not isinstance(build_entry, dict):
            raise RunnerError(f"invalid launcher build result: {build_entry!r}")
        drv_path = build_entry.get("drvPath")
        outputs = build_entry.get("outputs")
        if not isinstance(drv_path, str) or not isinstance(outputs, dict):
            raise RunnerError(f"incomplete launcher build result: {build_entry!r}")

        if not isinstance(app_result, dict):
            raise RunnerError(f"invalid launcher app metadata: {app_result!r}")
        program = app_result.get("program")
        context = app_result.get("context")
        if not isinstance(program, str) or not isinstance(context, dict):
            raise RunnerError(f"incomplete launcher app metadata: {app_result!r}")
        context_entry = context.get(drv_path)
        if len(context) != 1 or not isinstance(context_entry, dict):
            raise RunnerError(
                "launcher package and app program derivations differ: "
                f"build={drv_path!r}, context={context!r}"
            )
        context_outputs = context_entry.get("outputs")
        if not isinstance(context_outputs, list) or len(context_outputs) != 1:
            raise RunnerError(
                f"launcher app must reference one package output: {context_entry!r}"
            )
        output_name = context_outputs[0]
        physical_package = outputs.get(output_name)
        logical_program = Path(program)
        if (
            not isinstance(physical_package, str)
            or not physical_package.startswith("/nix/store/")
            or not logical_program.is_absolute()
            or logical_program.parent.name != "bin"
            or not logical_program.name
        ):
            raise RunnerError(
                "launcher resolution did not produce one physical package output "
                f"and an absolute bin program: outputs={outputs!r}, "
                f"program={program!r}"
            )
        program_path = Path(physical_package) / "bin" / logical_program.name
        if not program_path.is_file():
            raise RunnerError(
                f"resolved launcher is not a realized program: {program_path}"
            )
        resolved[launcher] = str(program_path)
    return resolved


def service_command(run: PlannedRun, args: argparse.Namespace) -> list[str]:
    launcher = launcher_name(run, args)
    immutable = getattr(args, "resolved_launchers", {}).get(launcher)
    scheduler_arguments = [
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    if immutable is not None:
        return [immutable, str(args.candidate_env), *scheduler_arguments]
    return [
        "nix",
        "run",
        "--impure",
        f"{args.config_ref}#{launcher}",
        "--",
        str(args.candidate_env),
        *scheduler_arguments,
    ]


def benchmark_command(
    run: PlannedRun, args: argparse.Namespace, raw_result: Path
) -> list[str]:
    workload = run.workload
    return [
        str(args.candidate_env / "bin" / "vllm"),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        args.base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.served_model,
        "--tokenizer",
        args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.context),
        "--random-output-len",
        str(workload.output_tokens),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(workload.num_prompts),
        "--num-warmups",
        "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(workload.batch),
        "--seed",
        str(workload.seed),
        "--temperature",
        "0",
        "--ignore-eos",
        "--save-result",
        "--save-detailed",
        "--disable-tqdm",
        "--result-dir",
        str(raw_result.parent),
        "--result-filename",
        raw_result.name,
    ]


def warmup_command(
    run: PlannedRun, args: argparse.Namespace, raw_result: Path
) -> list[str] | None:
    warmups = args.num_warmups if args.num_warmups is not None else run.workload.batch
    if warmups == 0:
        return None
    command = benchmark_command(run, args, raw_result)
    command[command.index("--num-prompts") + 1] = str(warmups)
    return command


def parse_running_metric(text: str) -> float:
    pattern = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")
    found = [
        float(match.group(1))
        for line in text.splitlines()
        if (match := pattern.match(line.strip()))
    ]
    if not found:
        raise RunnerError(f"metrics response is missing {RUNNING_METRIC}")
    return sum(found)


def http_text(url: str, *, timeout: float) -> str:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode()


def base_endpoint(args: argparse.Namespace, path: str) -> str:
    return args.base_url.rstrip("/") + path


def assert_port_unused(base_url: str) -> None:
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise RunnerError("--base-url must be a loopback HTTP endpoint")
    port = parsed.port or 80
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((parsed.hostname, port)) == 0:
            raise RunnerError(
                f"refusing to launch: {parsed.hostname}:{port} already has a listener"
            )


def wait_for_ready(process: subprocess.Popen[bytes], args: argparse.Namespace) -> None:
    deadline = time.monotonic() + args.startup_timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RunnerError(f"service exited before readiness ({process.returncode})")
        try:
            http_text(base_endpoint(args, "/health"), timeout=2.0)
            models = json.loads(
                http_text(base_endpoint(args, "/v1/models"), timeout=5.0)
            )
            ids = [item.get("id") for item in models.get("data", [])]
            if args.served_model not in ids:
                raise RunnerError(
                    f"ready service does not expose {args.served_model!r}: {ids}"
                )
            return
        except (OSError, ValueError, urllib.error.URLError, RunnerError) as exc:
            last_error = exc
            time.sleep(args.readiness_poll_interval)
    raise RunnerError(f"service readiness timed out: {last_error}")


def _process_group_members(process_group: int) -> list[int]:
    members: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat.rsplit(")", 1)[1].split()
            if fields[0] != "Z" and int(fields[2]) == process_group:
                members.append(int(entry.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return members


def _process_argv(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [part.decode(errors="replace") for part in raw.split(b"\0") if part]


def _process_environment(pid: int) -> dict[str, str | None]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    parsed: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" in item:
            key, value = item.split(b"=", 1)
            parsed[key.decode(errors="replace")] = value.decode(errors="replace")
    return {name: parsed.get(name) for name in CAPTURED_ENVIRONMENT}


def capture_engine_process(
    process: subprocess.Popen[bytes],
) -> tuple[int, list[str], dict[str, str | None]]:
    process_group = os.getpgid(process.pid)
    candidates: list[tuple[int, list[str]]] = []
    for pid in _process_group_members(process_group):
        try:
            argv = _process_argv(pid)
        except (FileNotFoundError, PermissionError):
            continue
        if "serve" in argv and "--served-model-name" in argv:
            candidates.append((pid, argv))
    if len(candidates) != 1:
        rendered = [(pid, shlex.join(_redact_argv(argv))) for pid, argv in candidates]
        raise RunnerError(f"expected one vLLM API process, found {rendered}")
    pid, argv = candidates[0]
    return pid, argv, _process_environment(pid)


def _arg_after(argv: Sequence[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def _argument_present(argv: Sequence[str], name: str) -> bool:
    """Recognize both argparse's ``--flag value`` and ``--flag=value`` forms."""
    return any(argument == name or argument.startswith(f"{name}=") for argument in argv)


def _redact_argv(argv: Sequence[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for argument in argv:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if argument.startswith("--") and "=" in argument:
            name, _value = argument.split("=", 1)
            redacted.append(
                f"{name}=<redacted>"
                if name not in PUBLIC_TOKEN_ARGUMENTS and SENSITIVE_NAME.search(name)
                else argument
            )
            continue
        redacted.append(argument)
        if (
            argument.startswith("--")
            and argument not in PUBLIC_TOKEN_ARGUMENTS
            and SENSITIVE_NAME.search(argument)
        ):
            redact_next = True
    return redacted


def _canonical_argv(argv: Sequence[str]) -> list[str]:
    canonical = _redact_argv(argv)
    if canonical and canonical[0].endswith("/bin/.vllm-wrapped"):
        canonical[0] = "<PROCESS_PACKAGE>/bin/.vllm-wrapped"
    for name in ARM_ARGUMENTS:
        try:
            index = canonical.index(name)
        except ValueError:
            continue
        if index + 1 < len(canonical):
            canonical[index + 1] = "<ARM_VALUE>"
    return canonical


def service_profile_evidence(
    argv: Sequence[str],
    environment: Mapping[str, str | None],
    *,
    variant_provenance: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    redacted_environment = {
        name: "<redacted>" if SENSITIVE_NAME.search(name) and value else value
        for name, value in sorted(environment.items())
    }
    canonical_environment = {
        name: "<ARM_VALUE>" if name in ARM_ENVIRONMENT else value
        for name, value in redacted_environment.items()
    }
    canonical = {
        "argv": _canonical_argv(argv),
        "environment": canonical_environment,
        "variant_provenance": {field: "<ARM_VALUE>" for field in VARIANT_FIELDS},
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    raw_layout = environment.get("KVARN_NATIVE_XPU_DPAS_LAYOUT")
    native_layout = {value: name for name, value in NATIVE_LAYOUT_ENV.items()}.get(
        raw_layout
    )
    return {
        "redacted_argv": _redact_argv(argv),
        "redacted_environment": redacted_environment,
        "max_num_batched_tokens": _arg_after(argv, "--max-num-batched-tokens"),
        "native_layout": native_layout,
        "native_layout_environment": raw_layout,
        "variant_provenance": dict(variant_provenance or {}),
        "canonical_matched_profile": canonical,
        "canonical_matched_profile_sha256": hashlib.sha256(encoded).hexdigest(),
        "allowed_arm_argument_differences": sorted(ARM_ARGUMENTS),
        "allowed_arm_environment_differences": sorted(ARM_ENVIRONMENT),
    }


def verify_service_profile(
    argv: Sequence[str],
    environment: Mapping[str, str | None],
    run: PlannedRun,
    args: argparse.Namespace,
) -> None:
    try:
        serve_index = argv.index("serve")
    except ValueError as exc:
        raise RunnerError("captured service argv has no serve subcommand") from exc
    expected_values = {
        "model": (argv[serve_index + 1], args.model),
        "--served-model-name": (
            _arg_after(argv, "--served-model-name"),
            args.served_model,
        ),
        "--revision": (_arg_after(argv, "--revision"), args.model_revision),
        "--dtype": (_arg_after(argv, "--dtype"), "bfloat16"),
        "--quantization": (_arg_after(argv, "--quantization"), "compressed-tensors"),
        "--kv-cache-dtype": (
            _arg_after(argv, "--kv-cache-dtype"),
            ARM_SETTINGS[run.arm]["kv_cache_dtype"],
        ),
        "--max-model-len": (
            _arg_after(argv, "--max-model-len"),
            str(args.max_model_len),
        ),
        "--max-num-seqs": (_arg_after(argv, "--max-num-seqs"), str(run.workload.batch)),
        "--max-num-batched-tokens": (
            _arg_after(argv, "--max-num-batched-tokens"),
            str(args.max_num_batched_tokens),
        ),
        "--gpu-memory-utilization": (
            _arg_after(argv, "--gpu-memory-utilization"),
            "0.95",
        ),
    }
    mismatches = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in expected_values.items()
        if actual != expected
    }
    required_flags = {
        "--enforce-eager",
        "--language-model-only",
        "--no-enable-prefix-caching",
    }
    missing_flags = sorted(required_flags - set(argv))
    forbidden_flags = sorted(
        flag
        for flag in (
            "--compilation-config",
            "--cpu-offload-gb",
            "--cpu-offload-params",
            "--kv-offloading-backend",
            "--kv-offloading-size",
            "--kv-transfer-config",
            "--offload-backend",
            "--offload-group-size",
            "--offload-num-in-group",
            "--offload-params",
            "--offload-prefetch-step",
            "--speculative-config",
        )
        if _argument_present(argv, flag)
    )
    native = ARM_SETTINGS[run.arm]["native_xpu"]
    expected_environment = {
        "CCL_ATL_TRANSPORT": "ofi",
        "CCL_LOG_LEVEL": "warn",
        "CCL_PROCESS_LAUNCHER": "none",
        "CCL_ZE_IPC_EXCHANGE": "sockets",
        "HF_HOME": str(args.hf_home),
        "HOME": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "KVARN_NATIVE_XPU": native,
        "KVARN_NATIVE_XPU_DECODE": native,
        "KVARN_NATIVE_XPU_DPAS_LAYOUT": NATIVE_LAYOUT_ENV[
            native_layout_for_run(run, args)
        ],
        "KVARN_NATIVE_XPU_MATERIALIZE": native,
        "KVARN_NATIVE_XPU_PERSISTENT_SCRATCH": native,
        "KVARN_NATIVE_XPU_SPLITS": str(native_splits_for_run(run, args)),
        "KVARN_PREFILL_FP16_WINDOW_BLOCKS": str(DEFAULT_PREFILL_WINDOW_BLOCKS),
        "VLLM_CACHE_ROOT": str(args.runtime_cache / "vllm-xpu-brutus-kvarn"),
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_KVARN_DEFER_PREFILL_FLUSH": None,
        "XDG_CACHE_HOME": str(args.runtime_cache),
    }
    environment_mismatches = {
        name: {"actual": environment.get(name), "expected": expected}
        for name, expected in expected_environment.items()
        if environment.get(name) != expected
    }
    if environment.get("VLLM_XPU_ENABLE_XPU_GRAPH") not in {None, "", "0"}:
        environment_mismatches["VLLM_XPU_ENABLE_XPU_GRAPH"] = {
            "actual": environment.get("VLLM_XPU_ENABLE_XPU_GRAPH"),
            "expected": "unset or 0",
        }
    if mismatches or missing_flags or forbidden_flags or environment_mismatches:
        raise RunnerError(
            "foreground service profile mismatch: "
            + json.dumps(
                {
                    "arguments": mismatches,
                    "missing_flags": missing_flags,
                    "forbidden_flags": forbidden_flags,
                    "environment": environment_mismatches,
                },
                sort_keys=True,
            )
        )


def verify_candidate_identity(
    argv: Sequence[str], candidate_env: Path
) -> dict[str, Any]:
    process_packages = {
        argument.removesuffix("/bin/.vllm-wrapped")
        for argument in argv
        if argument.endswith("/bin/.vllm-wrapped")
    }
    if len(process_packages) != 1:
        raise RunnerError(
            "captured service argv must identify exactly one .vllm-wrapped package"
        )
    process_package = process_packages.pop()
    executable = next(
        argument for argument in argv if argument.endswith("/bin/.vllm-wrapped")
    )
    try:
        closure = subprocess.run(
            ["nix-store", "-qR", str(candidate_env)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        process_closure = subprocess.run(
            ["nix-store", "-qR", process_package],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"cannot query candidate Nix closure: {exc}") from exc
    if process_package not in closure:
        raise RunnerError(
            f"service package {process_package} is not in {candidate_env}'s closure"
        )

    def closure_digest(paths: Sequence[str]) -> str:
        canonical = "\n".join(sorted(set(paths))) + "\n"
        return hashlib.sha256(canonical.encode()).hexdigest()

    return {
        "candidate_env": str(candidate_env),
        "process_executable": executable,
        "process_package": process_package,
        "candidate_closure_paths": sorted(set(closure)),
        "candidate_closure_sha256": closure_digest(closure),
        "process_closure_paths": sorted(set(process_closure)),
        "process_closure_sha256": closure_digest(process_closure),
    }


def verify_correctness_candidate_identity(
    actual_identity: Mapping[str, Any], expected_identity: Mapping[str, str]
) -> None:
    actual = {
        field: actual_identity.get(field)
        for field in (
            "process_package",
            "candidate_closure_sha256",
            "process_closure_sha256",
        )
    }
    if actual != expected_identity:
        raise RunnerError(
            "correctness artifact and running service identify different candidate "
            "builds"
        )


def _signal_process_group(process_group: int, selected_signal: signal.Signals) -> None:
    try:
        os.killpg(process_group, selected_signal)
    except ProcessLookupError:
        pass


def _wait_for_process_group(process_group: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_group_members(process_group):
            return True
        time.sleep(0.1)
    return not _process_group_members(process_group)


def stop_service(service: ServiceProcess, timeout: float) -> None:
    process = service.process
    process_group = service.process_group
    try:
        if _process_group_members(process_group):
            _signal_process_group(process_group, signal.SIGINT)
        if process.poll() is None:
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _signal_process_group(process_group, signal.SIGTERM)
                try:
                    process.wait(timeout=min(timeout, 30.0))
                except subprocess.TimeoutExpired:
                    _signal_process_group(process_group, signal.SIGKILL)
                    process.wait(timeout=30.0)
        if not _wait_for_process_group(process_group, timeout):
            _signal_process_group(process_group, signal.SIGTERM)
            if not _wait_for_process_group(process_group, min(timeout, 30.0)):
                _signal_process_group(process_group, signal.SIGKILL)
                if not _wait_for_process_group(process_group, 30.0):
                    raise RunnerError("foreground service process group did not exit")
    finally:
        service.supervisor.unregister(process_group)
        if not service.log_stream.closed:
            service.log_stream.flush()
            os.fsync(service.log_stream.fileno())
            service.log_stream.close()


def run_managed_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    output: TextIO,
    timeout: float,
    supervisor: ProcessSupervisor,
    label: str,
) -> int:
    """Run one benchmark in its own process group and reap all descendants."""
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=output,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    process_group = os.getpgid(process.pid)
    supervisor.register(process_group, label)
    try:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _signal_process_group(process_group, signal.SIGTERM)
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                _signal_process_group(process_group, signal.SIGKILL)
                process.wait(timeout=30.0)
            raise RunnerError(f"{label} timed out after {timeout:g}s") from exc
    finally:
        if not _wait_for_process_group(process_group, 5.0):
            _signal_process_group(process_group, signal.SIGTERM)
            if not _wait_for_process_group(process_group, 5.0):
                _signal_process_group(process_group, signal.SIGKILL)
                _wait_for_process_group(process_group, 5.0)
        supervisor.unregister(process_group)


def start_service(
    run: PlannedRun, run_dir: Path, args: argparse.Namespace
) -> ServiceProcess:
    command = service_command(run, args)
    write_json_atomic(run_dir / "service-command.json", command)
    for attempt in range(1, args.startup_attempts + 1):
        assert_port_unused(args.base_url)
        log_path = run_dir / "engine.log"
        log_stream = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=args.config_repo,
            env=service_environment(args),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        process_group = os.getpgid(process.pid)
        args.supervisor.register(process_group, f"service:{run.arm}")
        provisional = ServiceProcess(
            process=process,
            process_group=process_group,
            log_stream=log_stream,
            log_path=log_path,
            engine_pid=process.pid,
            argv=[],
            environment={},
            supervisor=args.supervisor,
        )
        try:
            wait_for_ready(process, args)
            engine_pid, argv, environment = capture_engine_process(process)
            verify_service_profile(argv, environment, run, args)
            provisional.engine_pid = engine_pid
            provisional.argv = argv
            provisional.environment = environment
            return provisional
        except BaseException as exc:
            stop_service(provisional, args.shutdown_timeout)
            if isinstance(exc, (RunnerInterrupted, KeyboardInterrupt)):
                raise
            if attempt == args.startup_attempts:
                raise
            log_path.replace(run_dir / f"engine-failed-startup-{attempt}.log")
            deadline = time.monotonic() + args.shutdown_timeout
            while time.monotonic() < deadline:
                try:
                    assert_port_unused(args.base_url)
                    break
                except RunnerError:
                    time.sleep(0.25)
            else:
                raise RunnerError("service port remained occupied after failed startup")
    raise AssertionError("unreachable")


def sample_scheduler(
    *,
    args: argparse.Namespace,
    required_running: int,
    stop: threading.Event,
    samples: list[dict[str, Any]],
    errors: list[str],
) -> None:
    while not stop.is_set():
        try:
            text = http_text(base_endpoint(args, "/metrics"), timeout=2.0)
            running = parse_running_metric(text)
            samples.append({"at": utc_timestamp(), "running": running})
            if running >= required_running:
                stop.set()
                return
        except (OSError, ValueError, RunnerError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        stop.wait(args.metrics_poll_interval)


def wait_for_scheduler_idle(args: argparse.Namespace, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_running: float | None = None
    while time.monotonic() < deadline:
        try:
            last_running = parse_running_metric(
                http_text(base_endpoint(args, "/metrics"), timeout=2.0)
            )
            if last_running == 0:
                return
        except (OSError, ValueError, RunnerError):
            pass
        time.sleep(args.metrics_poll_interval)
    raise RunnerError(f"scheduler did not become idle after warmup: {last_running}")


def scheduler_summary(
    samples: Sequence[Mapping[str, Any]], errors: Sequence[str], required: int
) -> dict[str, Any]:
    peak = max((float(sample["running"]) for sample in samples), default=0.0)
    return {
        "metric": RUNNING_METRIC,
        "required_running": required,
        "peak_running": peak,
        "required_overlap_observed": peak >= required,
        "samples": list(samples),
        "errors": list(errors),
    }


def validate_engine_log(
    path: Path, *, native: bool, expected_layout: str = "natural"
) -> dict[str, Any]:
    if (
        expected_layout not in NATIVE_LAYOUTS
        or not native
        and expected_layout != "natural"
    ):
        raise RunnerError("invalid native-layout engine-log expectation")
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    result = scan(lines)
    if result["status"] != "passed":
        raise RunnerError("engine log contains fatal findings")
    xpu = xpu_runtime_evidence(lines)
    if not xpu["device_config_xpu"]:
        raise RunnerError("engine log does not report device_config=xpu")
    if not xpu["positive_residency"]:
        raise RunnerError("engine log does not report positive XPU model/KV residency")
    dispatched = NATIVE_DISPATCH in text
    if dispatched != native:
        expectation = "contain" if native else "not contain"
        raise RunnerError(f"engine log must {expectation} native dispatch evidence")
    if native and FALLBACK_PATTERN.search(text):
        raise RunnerError("native engine log reports a Kvarn fallback")
    result["xpu_runtime"] = xpu
    result["native_layout_expected"] = expected_layout
    result["native_layout_log_marker"] = "unavailable"
    result["native_layout_evidence"] = (
        "captured-process-environment-plus-native-dispatch"
        if native
        else "captured-process-environment"
    )
    return result


def load_and_validate_benchmark_result(
    raw_result: Path, workload: Workload
) -> dict[str, Any]:
    try:
        document = json.loads(raw_result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot load raw benchmark result: {exc}") from exc
    if not isinstance(document, dict):
        raise RunnerError("raw benchmark result must be a JSON object")
    expected_lengths = {
        "completed": workload.num_prompts,
        "num_prompts": workload.num_prompts,
        "failed": 0,
        "max_concurrency": workload.batch,
    }
    mismatches = {
        key: {"actual": document.get(key), "expected": expected}
        for key, expected in expected_lengths.items()
        if document.get(key) != expected
    }
    if any(value != workload.context for value in document.get("input_lens", [])):
        mismatches["input_lens"] = {
            "actual": document.get("input_lens"),
            "expected": [workload.context] * workload.num_prompts,
        }
    if any(
        value != workload.output_tokens for value in document.get("output_lens", [])
    ):
        mismatches["output_lens"] = {
            "actual": document.get("output_lens"),
            "expected": [workload.output_tokens] * workload.num_prompts,
        }
    if len(document.get("input_lens", [])) != workload.num_prompts:
        mismatches["input_lens_count"] = {
            "actual": len(document.get("input_lens", [])),
            "expected": workload.num_prompts,
        }
    if len(document.get("output_lens", [])) != workload.num_prompts:
        mismatches["output_lens_count"] = {
            "actual": len(document.get("output_lens", [])),
            "expected": workload.num_prompts,
        }
    if mismatches:
        raise RunnerError("raw benchmark shape mismatch: " + json.dumps(mismatches))
    return document


def persist_warmup_result(
    *,
    raw_result: Path,
    output: Path,
    workload: Workload,
    argv: Sequence[str],
    arm: str,
    run_uuid: str,
    identity: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    document = load_and_validate_benchmark_result(raw_result, workload)
    observed_concurrency = document.get("max_concurrent_requests")
    if (
        isinstance(observed_concurrency, bool)
        or not isinstance(observed_concurrency, int)
        or observed_concurrency < workload.batch
    ):
        raise RunnerError(
            "warmup did not reach full width: "
            f"observed={observed_concurrency!r}, required={workload.batch}"
        )
    result = {
        "schema_version": 1,
        "status": "passed",
        "validated_at": utc_timestamp(),
        "arm": arm,
        "run_uuid": run_uuid,
        "workload": dataclasses.asdict(workload),
        "argv": _redact_argv(argv),
        "raw_result": str(raw_result.resolve()),
        "raw_result_sha256": sha256_file(raw_result),
        "completed": document["completed"],
        "failed": document["failed"],
        "max_concurrent_requests": observed_concurrency,
        "process_package": identity["process_package"],
        "process_closure_sha256": identity["process_closure_sha256"],
        "candidate_closure_sha256": identity["candidate_closure_sha256"],
        "matched_profile_sha256": profile["canonical_matched_profile_sha256"],
        "native_layout": profile["native_layout"],
        "native_layout_environment": profile["native_layout_environment"],
        "variant_provenance": profile["variant_provenance"],
    }
    write_json_atomic(output, result)
    return result


def seal_benchmark_result(
    *,
    raw_result: Path,
    output: Path,
    engine_log: Path,
    run: PlannedRun,
    args: argparse.Namespace,
    candidate_id: str | None,
    correctness_sha256: str | None,
    scheduler: Mapping[str, Any],
    run_uuid: str,
    started_at: str,
    identity: Mapping[str, Any],
    profile: Mapping[str, Any],
    warmup_result: Path | None,
) -> dict[str, Any]:
    workload = run.workload
    document = load_and_validate_benchmark_result(raw_result, workload)
    peak = float(scheduler.get("peak_running", 0))
    if peak < workload.batch:
        raise RunnerError(f"scheduler peak {peak:g} did not reach B{workload.batch}")
    max_num_batched_tokens = profile.get("max_num_batched_tokens")
    if max_num_batched_tokens != str(args.max_num_batched_tokens):
        raise RunnerError(
            "verified service profile lost max_num_batched_tokens: "
            f"expected {args.max_num_batched_tokens}, got {max_num_batched_tokens!r}"
        )
    expected_layout = native_layout_for_run(run, args)
    expected_variant = variant_provenance_for_run(run, args)
    if (
        profile.get("native_layout") != expected_layout
        or profile.get("native_layout_environment")
        != NATIVE_LAYOUT_ENV[expected_layout]
        or profile.get("variant_provenance") != expected_variant
    ):
        raise RunnerError(
            "verified service profile lost the selected native cache layout"
        )

    hardware_path = Path(args.hardware_preflight_path).resolve()
    try:
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read XPU hardware preflight: {exc}") from exc
    if not isinstance(hardware, dict):
        raise RunnerError("XPU hardware preflight must be an object")
    validate_xpu_preflight(hardware)
    xpu = xpu_runtime_evidence(
        engine_log.read_text(encoding="utf-8", errors="replace").splitlines()
    )
    if not xpu["device_config_xpu"] or not xpu["positive_residency"]:
        raise RunnerError("engine log lacks positive XPU runtime evidence")
    if not args.exploratory and warmup_result is None:
        raise RunnerError("formal performance evidence requires a full-width warmup")

    metadata: dict[str, Any] = {
        "kvarn_evidence_mode": "exploratory" if args.exploratory else "formal",
        "kvarn_promotable": not args.exploratory,
        "kvarn_model_revision": args.model_revision,
        "kvarn_service_profile": workload.service_profile,
        "kvarn_workload_id": workload.workload_id,
        "kvarn_seed": str(workload.seed),
        "kvarn_max_model_len": str(args.max_model_len),
        "kvarn_max_num_seqs": str(workload.batch),
        "kvarn_max_num_batched_tokens": max_num_batched_tokens,
        "kvarn_enforce_eager": "1",
        "kvarn_prefix_caching": "0",
        "kvarn_mtp": "0",
        "kvarn_xpu_graph": "0",
        "kvarn_scheduler_peak_running": str(int(peak)),
        "kvarn_arm": run.arm,
        "kvarn_kv_cache_dtype": ARM_SETTINGS[run.arm]["kv_cache_dtype"],
        "kvarn_native_xpu": ARM_SETTINGS[run.arm]["native_xpu"],
        "kvarn_native_layout": expected_layout,
        "kvarn_native_layout_environment": NATIVE_LAYOUT_ENV[expected_layout],
        "kvarn_native_layout_log_marker": "unavailable",
        "kvarn_native_layout_evidence": (
            "captured-process-environment-plus-native-dispatch"
            if run.arm == "candidate"
            else "captured-process-environment"
        ),
        **{f"kvarn_{field}": value for field, value in expected_variant.items()},
        "kvarn_native_splits": str(native_splits_for_run(run, args)),
        "kvarn_run_order": str(run.order),
        "kvarn_run_uuid": run_uuid,
        "kvarn_run_started_at": started_at,
        "kvarn_engine_log_sha256": sha256_file(engine_log),
        "kvarn_process_executable": identity["process_executable"],
        "kvarn_process_package": identity["process_package"],
        "kvarn_process_closure_sha256": identity["process_closure_sha256"],
        "kvarn_candidate_closure_sha256": identity["candidate_closure_sha256"],
        "kvarn_matched_profile_sha256": profile["canonical_matched_profile_sha256"],
        "kvarn_accelerator": "xpu",
        "kvarn_xpu_available": "1",
        "kvarn_xpu_device_count": "1",
        "kvarn_xpu_device_name": EXPECTED_XPU_DEVICE_NAME,
        "kvarn_xpu_compute_probe": "passed",
        "kvarn_hardware_preflight_path": str(hardware_path),
        "kvarn_hardware_preflight_sha256": sha256_file(hardware_path),
        "kvarn_xpu_consumed_memory_gib": xpu["consumed_memory_gib"],
        "kvarn_xpu_kv_cache_memory_gib": xpu["kv_cache_memory_gib"],
    }
    if warmup_result is not None:
        warmup_path = warmup_result.resolve()
        metadata["kvarn_warmup_path"] = str(warmup_path)
        metadata["kvarn_warmup_sha256"] = sha256_file(warmup_path)
    if candidate_id is not None:
        metadata["kvarn_candidate_id"] = candidate_id
    if correctness_sha256 is not None:
        metadata["kvarn_correctness_sha256"] = correctness_sha256
    collisions = sorted(key for key in metadata if key in document)
    if collisions:
        raise RunnerError(f"raw result already contains sealed metadata: {collisions}")
    document.update(metadata)
    write_json_atomic(output, document)
    return document


def _run_directory(root: Path, run: PlannedRun, run_uuid: str) -> Path:
    return (
        root
        / f"b{run.workload.batch}"
        / f"context-{run.workload.context}"
        / f"{run.order:02d}-{run.arm}-{run_uuid}"
    )


def run_one(
    run: PlannedRun,
    *,
    args: argparse.Namespace,
    candidate_id: str | None,
    correctness_sha256: str | None,
    correctness_identity: Mapping[str, str] | None,
) -> dict[str, Any]:
    run_uuid = str(uuid.uuid4())
    started_at = utc_timestamp()
    run_dir = _run_directory(args.output_dir, run, run_uuid)
    run_dir.mkdir(parents=True)
    raw_result = run_dir / "benchmark.raw.json"
    warmup_raw_result = run_dir / "warmup.raw.json"
    sealed_result = run_dir / "benchmark.json"
    benchmark_stdout = run_dir / "benchmark.stdout.log"
    warmup_result: Path | None = None
    service: ServiceProcess | None = None
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "run_uuid": run_uuid,
        "run_started_at": started_at,
        "arm": run.arm,
        "order": run.order,
        "workload": dataclasses.asdict(run.workload),
    }
    write_json_atomic(run_dir / "run.json", manifest)
    try:
        service = start_service(run, run_dir, args)
        profile = service_profile_evidence(
            service.argv,
            service.environment,
            variant_provenance=variant_provenance_for_run(run, args),
        )
        write_json_atomic(run_dir / "service-argv.json", profile["redacted_argv"])
        write_json_atomic(
            run_dir / "service-environment.json", profile["redacted_environment"]
        )
        identity = verify_candidate_identity(service.argv, args.candidate_env)
        if correctness_identity is not None:
            verify_correctness_candidate_identity(identity, correctness_identity)
        write_json_atomic(run_dir / "candidate-identity.json", identity)
        write_json_atomic(run_dir / "matched-service-profile.json", profile)
        benchmark_argv = benchmark_command(run, args, raw_result)
        write_json_atomic(run_dir / "benchmark-argv.json", benchmark_argv)
        warmup_argv = warmup_command(run, args, warmup_raw_result)
        write_json_atomic(run_dir / "warmup-argv.json", warmup_argv)
        if warmup_argv is not None:
            with (run_dir / "warmup.stdout.log").open("w", encoding="utf-8") as output:
                warmup_returncode = run_managed_process(
                    warmup_argv,
                    cwd=args.packaging_repo,
                    environment=runner_environment(args),
                    output=output,
                    timeout=args.benchmark_timeout,
                    supervisor=args.supervisor,
                    label="warmup benchmark",
                )
            if warmup_returncode != 0:
                raise RunnerError(f"warmup benchmark exited {warmup_returncode}")
            warmup_prompts = (
                args.num_warmups if args.num_warmups is not None else run.workload.batch
            )
            warmup_workload = dataclasses.replace(
                run.workload, num_prompts=warmup_prompts
            )
            warmup_result = run_dir / "warmup.json"
            persist_warmup_result(
                raw_result=warmup_raw_result,
                output=warmup_result,
                workload=warmup_workload,
                argv=warmup_argv,
                arm=run.arm,
                run_uuid=run_uuid,
                identity=identity,
                profile=profile,
            )
            wait_for_scheduler_idle(args)

        samples: list[dict[str, Any]] = []
        metric_errors: list[str] = []
        stop_sampling = threading.Event()
        sampler = threading.Thread(
            target=sample_scheduler,
            kwargs={
                "args": args,
                "required_running": run.workload.batch,
                "stop": stop_sampling,
                "samples": samples,
                "errors": metric_errors,
            },
            daemon=True,
        )
        sampler.start()
        with benchmark_stdout.open("w", encoding="utf-8") as output:
            try:
                benchmark_returncode = run_managed_process(
                    benchmark_argv,
                    cwd=args.packaging_repo,
                    environment=runner_environment(args),
                    output=output,
                    timeout=args.benchmark_timeout,
                    supervisor=args.supervisor,
                    label="measured benchmark",
                )
            finally:
                stop_sampling.set()
                sampler.join(timeout=5.0)
        if sampler.is_alive():
            raise RunnerError("scheduler sampler did not stop")
        scheduler = scheduler_summary(samples, metric_errors, run.workload.batch)
        write_json_atomic(run_dir / "scheduler-metrics.json", scheduler)
        if benchmark_returncode != 0:
            raise RunnerError(f"vllm bench serve exited {benchmark_returncode}")

        engine_pid = service.engine_pid
        stop_service(service, args.shutdown_timeout)
        service = None
        log_scan = validate_engine_log(
            run_dir / "engine.log",
            native=run.arm == "candidate",
            expected_layout=native_layout_for_run(run, args),
        )
        write_json_atomic(run_dir / "engine-log-scan.json", log_scan)
        sealed = seal_benchmark_result(
            raw_result=raw_result,
            output=sealed_result,
            engine_log=run_dir / "engine.log",
            run=run,
            args=args,
            candidate_id=candidate_id,
            correctness_sha256=correctness_sha256,
            scheduler=scheduler,
            run_uuid=run_uuid,
            started_at=started_at,
            identity=identity,
            profile=profile,
            warmup_result=warmup_result,
        )
        manifest.update(
            status="passed",
            run_finished_at=utc_timestamp(),
            service_pid=engine_pid,
            result=str(sealed_result),
            engine_log=str(run_dir / "engine.log"),
            engine_log_sha256=sealed["kvarn_engine_log_sha256"],
            scheduler_peak_running=scheduler["peak_running"],
        )
        write_json_atomic(run_dir / "run.json", manifest)
        return {
            "workload_id": run.workload.workload_id,
            "batch": run.workload.batch,
            "context": run.workload.context,
            "arm": run.arm,
            "order": run.order,
            "result": str(sealed_result),
            "engine_log": str(run_dir / "engine.log"),
            "run_manifest": str(run_dir / "run.json"),
        }
    except BaseException as exc:
        if service is not None:
            try:
                stop_service(service, args.shutdown_timeout)
            except (OSError, subprocess.SubprocessError, RunnerError) as stop_error:
                manifest["stop_error"] = f"{type(stop_error).__name__}: {stop_error}"
        manifest.update(
            status="failed",
            run_finished_at=utc_timestamp(),
            error=f"{type(exc).__name__}: {exc}",
        )
        engine_log = run_dir / "engine.log"
        if engine_log.is_file():
            manifest["engine_log"] = str(engine_log)
            manifest["engine_log_sha256"] = sha256_file(engine_log)
        write_json_atomic(run_dir / "run.json", manifest)
        raise


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise RunnerError("cannot compute a percentile from an empty sample")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _load_detailed_result(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot load sealed benchmark {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise RunnerError(f"sealed benchmark must be an object: {path}")
    return document


def _request_decode_rates(document: Mapping[str, Any]) -> list[float]:
    rates: list[float] = []
    for intervals in document.get("itls", []):
        if not isinstance(intervals, list) or not intervals:
            raise RunnerError("sealed benchmark has incomplete ITL evidence")
        try:
            values = [float(value) for value in intervals]
        except (TypeError, ValueError) as exc:
            raise RunnerError("sealed benchmark has non-numeric ITL evidence") from exc
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise RunnerError("sealed benchmark has non-positive ITL evidence")
        rates.append(len(values) / sum(values))
    if not rates:
        raise RunnerError("sealed benchmark has no request decode evidence")
    return rates


def _run_parity_metrics(document: Mapping[str, Any]) -> dict[str, float]:
    try:
        output_throughput = float(document["output_throughput"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError("sealed benchmark has invalid output throughput") from exc
    if not math.isfinite(output_throughput) or output_throughput <= 0:
        raise RunnerError("sealed benchmark has non-positive output throughput")
    result = {
        "output_throughput": output_throughput,
        "median_request_decode_throughput": statistics.median(
            _request_decode_rates(document)
        ),
    }
    if any(not math.isfinite(value) or value <= 0 for value in result.values()):
        raise RunnerError("sealed benchmark has non-positive descriptive metrics")
    return result


def _t_critical_95(sample_count: int) -> float:
    degrees_of_freedom = sample_count - 1
    if degrees_of_freedom <= 0:
        raise RunnerError("a confidence interval requires at least two pairs")
    return ONE_SIDED_T_95.get(degrees_of_freedom, 1.645)


def paired_noninferiority(
    paired_ratios: Sequence[float], *, threshold: float, minimum_pairs: int
) -> dict[str, Any]:
    if any(not math.isfinite(value) or value <= 0 for value in paired_ratios):
        raise RunnerError("paired ratios must be finite and positive")
    sample_count = len(paired_ratios)
    estimate = math.exp(statistics.mean(math.log(value) for value in paired_ratios))
    result: dict[str, Any] = {
        "method": "one-sided-95%-student-t-on-log-ABBA-block-ratios",
        "threshold": threshold,
        "minimum_pairs": minimum_pairs,
        "paired_ratios": list(paired_ratios),
        "paired_geometric_mean": estimate,
        "sample_count": sample_count,
    }
    if sample_count < minimum_pairs:
        result.update(status="insufficient_evidence", lower_confidence_bound=None)
        return result
    logs = [math.log(value) for value in paired_ratios]
    standard_error = statistics.stdev(logs) / math.sqrt(sample_count)
    lower_bound = math.exp(
        statistics.mean(logs) - _t_critical_95(sample_count) * standard_error
    )
    result.update(
        status="passed" if lower_bound >= threshold else "failed",
        lower_confidence_bound=lower_bound,
    )
    return result


def statistical_parity(
    ordered_documents: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    minimum_pairs: int,
) -> dict[str, Any]:
    if len(ordered_documents) % 4:
        raise RunnerError("paired parity requires complete four-run ABBA blocks")
    metrics: dict[str, dict[str, Any]] = {}
    for metric in PARITY_METRICS:
        block_ratios: list[float] = []
        for start in range(0, len(ordered_documents), 4):
            block = ordered_documents[start : start + 4]
            arms = [str(document["kvarn_arm"]) for document in block]
            if arms != list(ARM_ORDER):
                raise RunnerError(f"invalid ABBA block for parity: {arms}")
            values = [_run_parity_metrics(document)[metric] for document in block]
            reference = statistics.mean((values[0], values[3]))
            candidate = statistics.mean((values[1], values[2]))
            block_ratios.append(candidate / reference)
        metrics[metric] = paired_noninferiority(
            block_ratios, threshold=threshold, minimum_pairs=minimum_pairs
        )
    statuses = {metric["status"] for metric in metrics.values()}
    if "failed" in statuses:
        status = "failed"
    elif "insufficient_evidence" in statuses:
        status = "insufficient_evidence"
    else:
        status = "passed"
    return {
        "status": status,
        "interpretation": (
            "one-sided non-inferiority; faster-than-reference candidates are allowed"
        ),
        "metrics": metrics,
    }


def pooled_tail_latency(
    reference: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def values(documents: Sequence[Mapping[str, Any]], name: str) -> list[float]:
        if name == "ttft":
            return [
                float(value) * 1000.0
                for document in documents
                for value in document.get("ttfts", [])
            ]
        return [
            float(value) * 1000.0
            for document in documents
            for intervals in document.get("itls", [])
            for value in intervals
        ]

    result: dict[str, Any] = {"method": "p99-of-pooled-detailed-samples"}
    for name in ("ttft", "itl"):
        reference_p99 = _percentile(values(reference, name), 99)
        candidate_p99 = _percentile(values(candidate, name), 99)
        result[name] = {
            "reference_p99_ms": reference_p99,
            "candidate_p99_ms": candidate_p99,
            "candidate_over_reference": candidate_p99 / reference_p99,
        }
    return result


def validate_matched_results(documents: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "kvarn_process_package",
        "kvarn_process_closure_sha256",
        "kvarn_candidate_closure_sha256",
        "kvarn_max_num_batched_tokens",
        "kvarn_matched_profile_sha256",
        "kvarn_accelerator",
        "kvarn_xpu_available",
        "kvarn_xpu_device_count",
        "kvarn_xpu_device_name",
        "kvarn_xpu_compute_probe",
        "kvarn_hardware_preflight_path",
        "kvarn_hardware_preflight_sha256",
    )
    for field in fields:
        values = {document.get(field) for document in documents}
        if None in values or len(values) != 1:
            raise RunnerError(
                f"reference/candidate effective provenance differs for {field}"
            )


def summarize_exploratory_workload(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate and describe one non-promotable matched ABBA collection."""
    ordered_records = sorted(records, key=lambda record: int(record["order"]))
    documents = [
        _load_detailed_result(Path(str(record["result"]))) for record in ordered_records
    ]
    arms = [str(document.get("kvarn_arm")) for document in documents]
    if not documents or len(documents) % len(ARM_ORDER):
        raise RunnerError("exploratory collection requires complete ABBA blocks")
    if arms != list(ARM_ORDER) * (len(documents) // len(ARM_ORDER)):
        raise RunnerError(f"invalid exploratory ABBA order: {arms}")
    if any(
        document.get("kvarn_evidence_mode") != "exploratory" for document in documents
    ):
        raise RunnerError("exploratory collection contains non-exploratory evidence")
    if any(document.get("kvarn_promotable") is not False for document in documents):
        raise RunnerError("exploratory collection contains promotable evidence")
    validate_matched_results(documents)

    references = [
        document for document in documents if document["kvarn_arm"] == "reference"
    ]
    candidates = [
        document for document in documents if document["kvarn_arm"] == "candidate"
    ]
    descriptive_metrics: dict[str, dict[str, float]] = {}
    for metric in PARITY_METRICS:
        reference = statistics.median(
            _run_parity_metrics(document)[metric] for document in references
        )
        candidate = statistics.median(
            _run_parity_metrics(document)[metric] for document in candidates
        )
        descriptive_metrics[metric] = {
            "reference_median": reference,
            "candidate_median": candidate,
            "candidate_over_reference": candidate / reference,
        }
    return {
        "schema_version": 1,
        "evidence_mode": "exploratory",
        "promotable": False,
        "formal_performance_gate_run": False,
        "formal_statistical_parity_run": False,
        "collection": "complete",
        "repeats_per_arm": len(references),
        "abba_blocks": len(documents) // len(ARM_ORDER),
        "descriptive_metrics": descriptive_metrics,
        "pooled_tail_latency": pooled_tail_latency(references, candidates),
    }


def gate_workload(
    records: Sequence[Mapping[str, Any]], args: argparse.Namespace
) -> dict[str, Any]:
    references = sorted(
        (record for record in records if record["arm"] == "reference"),
        key=lambda record: int(record["order"]),
    )
    candidates = sorted(
        (record for record in records if record["arm"] == "candidate"),
        key=lambda record: int(record["order"]),
    )
    result = compare(
        [Path(str(record["result"])) for record in references],
        [Path(str(record["result"])) for record in candidates],
        reference_logs=[Path(str(record["engine_log"])) for record in references],
        candidate_logs=[Path(str(record["engine_log"])) for record in candidates],
        correctness_path=args.correctness,
        comparison_kind="end-to-end",
        mode="match",
        min_throughput_ratio=args.min_throughput_ratio,
        min_request_decode_ratio=args.min_request_decode_ratio,
        max_latency_ratio=args.max_latency_ratio,
    )
    ordered_records = sorted(records, key=lambda record: int(record["order"]))
    documents = [
        _load_detailed_result(Path(str(record["result"]))) for record in ordered_records
    ]
    validate_matched_results(documents)
    reference_documents = [
        document for document in documents if document["kvarn_arm"] == "reference"
    ]
    candidate_documents = [
        document for document in documents if document["kvarn_arm"] == "candidate"
    ]
    pooled = pooled_tail_latency(reference_documents, candidate_documents)
    result["candidate_over_reference"]["p99_ttft_ms"] = pooled["ttft"][
        "candidate_over_reference"
    ]
    result["candidate_over_reference"]["p99_itl_ms"] = pooled["itl"][
        "candidate_over_reference"
    ]
    result["checks"]["p99_ttft"] = (
        pooled["ttft"]["candidate_over_reference"] <= args.max_latency_ratio
    )
    result["checks"]["p99_itl"] = (
        pooled["itl"]["candidate_over_reference"] <= args.max_latency_ratio
    )
    result["status"] = "passed" if all(result["checks"].values()) else "failed"
    result["hard_floor"] = {
        "status": result["status"],
        "min_throughput_ratio": args.min_throughput_ratio,
        "min_request_decode_ratio": args.min_request_decode_ratio,
        "max_latency_ratio": args.max_latency_ratio,
        "pooled_tail_latency": pooled,
    }
    result["statistical_parity"] = statistical_parity(
        documents,
        threshold=args.parity_ratio,
        minimum_pairs=args.min_parity_pairs,
    )
    return result


def write_checksums(root: Path) -> None:
    output = root / "SHA256SUMS"
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == output:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(args: argparse.Namespace) -> dict[str, Any]:
    candidate_id = args.candidate_id
    correctness_sha256: str | None = None
    correctness_identity: dict[str, str] | None = None
    correctness_variant: dict[str, str] | None = None
    if args.correctness is not None:
        (
            candidate_id,
            correctness_sha256,
            correctness_identity,
            correctness_layout,
            correctness_variant,
        ) = load_correctness(args.correctness, args.candidate_id)
        if candidate_id != str(args.candidate_env):
            raise RunnerError(
                "correctness artifact candidate_id differs from --candidate-env"
            )
        if correctness_layout != args.native_layout:
            raise RunnerError(
                "correctness artifact native_layout differs from --native-layout"
            )
    plan = build_plan(
        contexts=args.context,
        batches=args.batch,
        output_tokens=args.output_tokens,
        waves_per_run=args.waves_per_run,
        repeats=args.repeats,
        seed=args.seed,
        max_model_len=args.max_model_len,
        minimum_repeats=2 if args.exploratory else 4,
    )
    selected_variant = variant_provenance_for_run(
        next(run for run in plan if run.arm == "candidate"), args
    )
    if correctness_variant is not None and correctness_variant != selected_variant:
        raise RunnerError(
            "correctness artifact variant provenance differs from the selected "
            "performance candidate"
        )
    args.resolved_launchers = resolve_launchers(plan, args)
    repositories = [
        repository_state("vllm-xpu-nix", args.packaging_repo),
        repository_state("vllm", args.vllm_repo),
        repository_state("vllm-xpu-kernels", args.kernels_repo),
        repository_state("nix-config", args.config_repo),
    ]
    session: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if args.plan_only else "running",
        "evidence_mode": "exploratory" if args.exploratory else "formal",
        "promotable": not args.exploratory,
        "created_at": utc_timestamp(),
        "candidate_env": str(args.candidate_env),
        "candidate_native_splits_by_batch": {
            str(batch): splits for batch, splits in sorted(args.native_splits.items())
        },
        "native_layout": args.native_layout,
        **selected_variant,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "resolved_launchers": args.resolved_launchers,
        "repositories": repositories,
        "plan": [
            {
                **dataclasses.asdict(run),
                "expected_native_splits": native_splits_for_run(run, args),
                "service_command": service_command(run, args),
            }
            for run in plan
        ],
    }
    if candidate_id is not None:
        session["candidate_id"] = candidate_id
    if args.correctness is not None:
        session["correctness_artifact"] = str(args.correctness)
        session["correctness_sha256"] = correctness_sha256
    if args.exploratory:
        session["formal_gates_skipped"] = True
    else:
        session["acceptance"] = {
            "hard_floor": {
                "min_throughput_ratio": args.min_throughput_ratio,
                "min_request_decode_ratio": args.min_request_decode_ratio,
                "max_pooled_p99_latency_ratio": args.max_latency_ratio,
            },
            "statistical_parity_target": {
                "paired_ratio": args.parity_ratio,
                "minimum_abba_pairs": args.min_parity_pairs,
                "confidence": "one-sided 95%",
            },
        }
    write_json_atomic(args.output_dir / "session.json", session)
    if args.plan_only:
        return session

    hardware_preflight = probe_xpu_hardware(args)
    hardware_preflight_path = args.output_dir / "hardware-preflight.json"
    write_json_atomic(hardware_preflight_path, hardware_preflight)
    args.hardware_preflight_path = hardware_preflight_path
    session["hardware_preflight"] = {
        "path": str(hardware_preflight_path.resolve()),
        "sha256": sha256_file(hardware_preflight_path),
    }
    write_json_atomic(args.output_dir / "session.json", session)

    records: list[dict[str, Any]] = []
    try:
        for run in plan:
            records.append(
                run_one(
                    run,
                    args=args,
                    candidate_id=candidate_id,
                    correctness_sha256=correctness_sha256,
                    correctness_identity=correctness_identity,
                )
            )
    except BaseException as exc:
        session.update(
            status="failed",
            finished_at=utc_timestamp(),
            completed_runs=records,
            error=f"{type(exc).__name__}: {exc}",
        )
        write_json_atomic(args.output_dir / "session.json", session)
        raise

    if args.exploratory:
        summaries: dict[str, Any] = {}
        try:
            for workload_id in dict.fromkeys(
                record["workload_id"] for record in records
            ):
                selected = [
                    record for record in records if record["workload_id"] == workload_id
                ]
                summary = summarize_exploratory_workload(selected)
                summary_path = (
                    Path(str(selected[0]["result"])).parents[1]
                    / "exploratory-summary.json"
                )
                write_json_atomic(summary_path, summary)
                summaries[workload_id] = {"path": str(summary_path)}
        except BaseException as exc:
            session.update(
                status="failed",
                finished_at=utc_timestamp(),
                completed_runs=records,
                error=f"{type(exc).__name__}: {exc}",
            )
            write_json_atomic(args.output_dir / "session.json", session)
            raise
        session.update(
            status="completed",
            finished_at=utc_timestamp(),
            completed_runs=records,
            exploratory_summaries=summaries,
        )
        write_json_atomic(args.output_dir / "session.json", session)
        write_checksums(args.output_dir)
        return session

    gate_results: dict[str, Any] = {}
    for workload_id in dict.fromkeys(record["workload_id"] for record in records):
        selected = [
            record for record in records if record["workload_id"] == workload_id
        ]
        try:
            gate = gate_workload(selected, args)
        except GateError as exc:
            gate = {"status": "invalid", "error": str(exc)}
        gate_path = Path(str(selected[0]["result"])).parents[1] / "gate.json"
        write_json_atomic(gate_path, gate)
        gate_results[workload_id] = {
            "path": str(gate_path),
            "hard_floor_status": gate["status"],
            "statistical_parity_status": gate.get("statistical_parity", {}).get(
                "status", "invalid"
            ),
        }

    performance_status = (
        "passed"
        if all(item["hard_floor_status"] == "passed" for item in gate_results.values())
        else "failed"
    )
    parity_statuses = {
        item["statistical_parity_status"] for item in gate_results.values()
    }
    if "failed" in parity_statuses or "invalid" in parity_statuses:
        parity_status = "failed"
    elif "insufficient_evidence" in parity_statuses:
        parity_status = "insufficient_evidence"
    else:
        parity_status = "passed"
    session.update(
        status="completed",
        performance_status=performance_status,
        statistical_parity_status=parity_status,
        finished_at=utc_timestamp(),
        completed_runs=records,
        gates=gate_results,
    )
    write_json_atomic(args.output_dir / "session.json", session)
    write_checksums(args.output_dir)
    return session


def _parse_int_list(values: Sequence[str] | None, defaults: Sequence[int]) -> list[int]:
    if not values:
        return list(defaults)
    parsed: list[int] = []
    for value in values:
        try:
            parsed.extend(
                int(item.strip()) for item in value.split(",") if item.strip()
            )
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer list: {value}") from exc
    return list(dict.fromkeys(parsed))


def _parse_native_splits(
    values: Sequence[str] | None, batches: Sequence[int]
) -> dict[int, int]:
    """Parse either one split count or explicit ``BATCH=SPLITS`` assignments."""
    selected_batches = list(dict.fromkeys(batches))
    if not values:
        missing = sorted(set(selected_batches) - DEFAULT_NATIVE_SPLITS.keys())
        if missing:
            raise argparse.ArgumentTypeError(
                f"no default native split count for batches {missing}"
            )
        return {batch: DEFAULT_NATIVE_SPLITS[batch] for batch in selected_batches}

    items = [item.strip() for value in values for item in value.split(",")]
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError("--native-splits contains an empty value")
    assigned = ["=" in item for item in items]
    if not any(assigned):
        if len(items) != 1:
            raise argparse.ArgumentTypeError(
                "use one global --native-splits value or BATCH=SPLITS assignments"
            )
        try:
            splits = int(items[0])
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid native split count: {items[0]}"
            ) from exc
        if splits not in SUPPORTED_NATIVE_SPLITS:
            raise argparse.ArgumentTypeError(
                f"unsupported native split count {splits}; expected one of "
                f"{sorted(SUPPORTED_NATIVE_SPLITS)}"
            )
        return {batch: splits for batch in selected_batches}
    if not all(assigned):
        raise argparse.ArgumentTypeError(
            "cannot mix a global native split count with BATCH=SPLITS assignments"
        )

    result: dict[int, int] = {}
    for item in items:
        batch_text, splits_text = item.split("=", 1)
        try:
            batch = int(batch_text)
            splits = int(splits_text)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid native split assignment: {item}"
            ) from exc
        if batch in result:
            raise argparse.ArgumentTypeError(
                f"duplicate native split assignment for B{batch}"
            )
        if splits not in SUPPORTED_NATIVE_SPLITS:
            raise argparse.ArgumentTypeError(
                f"unsupported native split count {splits}; expected one of "
                f"{sorted(SUPPORTED_NATIVE_SPLITS)}"
            )
        result[batch] = splits
    selected = set(selected_batches)
    if set(result) != selected:
        raise argparse.ArgumentTypeError(
            "native split assignments must exactly match selected batches: "
            f"expected {sorted(selected)}, got {sorted(result)}"
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-env", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--correctness", type=Path)
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help=(
            "collect non-promotable matched measurements without requiring or "
            "running the formal correctness/performance gates"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context", action="append")
    parser.add_argument("--batch", action="append")
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--waves-per-run", type=int, default=2)
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help=(
            "recorded repeats per arm; eight provide four ABBA confidence pairs; "
            "exploratory mode permits two for one non-promotable ABBA block"
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-warmups", type=int)
    parser.add_argument(
        "--native-layout",
        choices=NATIVE_LAYOUTS,
        default="natural",
        help=(
            "native candidate cache layout; xe2_dpas requires dedicated Brutus "
            "native-dpas launcher outputs (default: natural)"
        ),
    )
    parser.add_argument(
        "--native-splits",
        action="append",
        metavar="SPLITS|BATCH=SPLITS",
        help=(
            "expected native-launcher split count; repeat BATCH=SPLITS for "
            "per-batch tuning (default: 1=24, 4=16)"
        ),
    )
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help=(
            "scheduler token budget pinned identically for every auto/native "
            f"service arm (default: {DEFAULT_MAX_NUM_BATCHED_TOKENS})"
        ),
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--served-model", default="sunny-chat")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--config-ref", default="path:/home/jasonbk/.config/nix")
    parser.add_argument(
        "--runtime-cache",
        type=Path,
        default=Path("benchmark-results/kvarn-runtime-cache"),
    )
    parser.add_argument("--hf-home", type=Path, default=Path("/var/cache/huggingface"))
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    parser.add_argument("--startup-attempts", type=int, default=2)
    parser.add_argument("--readiness-poll-interval", type=float, default=2.0)
    parser.add_argument("--shutdown-timeout", type=float, default=180.0)
    parser.add_argument("--benchmark-timeout", type=float, default=7200.0)
    parser.add_argument("--metrics-poll-interval", type=float, default=0.1)
    parser.add_argument("--min-throughput-ratio", type=float, default=0.95)
    parser.add_argument("--min-request-decode-ratio", type=float, default=0.95)
    parser.add_argument("--max-latency-ratio", type=float, default=1.10)
    parser.add_argument("--parity-ratio", type=float, default=0.98)
    parser.add_argument("--min-parity-pairs", type=int, default=4)
    parser.add_argument(
        "--packaging-repo", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--vllm-repo", type=Path, default=Path("/home/jasonbk/Projects/vllm")
    )
    parser.add_argument(
        "--kernels-repo",
        type=Path,
        default=Path("/home/jasonbk/Projects/vllm-xpu-kernels"),
    )
    parser.add_argument(
        "--config-repo", type=Path, default=Path("/home/jasonbk/.config/nix")
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.context = _parse_int_list(args.context, DEFAULT_CONTEXTS)
        args.batch = _parse_int_list(args.batch, DEFAULT_BATCHES)
        args.native_splits = _parse_native_splits(args.native_splits, args.batch)
        args.output_dir = ensure_durable(args.output_dir, allow_tmp=args.allow_tmp)
        args.candidate_env = args.candidate_env.expanduser().resolve()
        if args.correctness is not None:
            args.correctness = args.correctness.expanduser().resolve()
        args.runtime_cache = args.runtime_cache.expanduser().resolve()
        args.hf_home = args.hf_home.expanduser().resolve()
        args.packaging_repo = args.packaging_repo.expanduser().resolve()
        args.vllm_repo = args.vllm_repo.expanduser().resolve()
        args.kernels_repo = args.kernels_repo.expanduser().resolve()
        args.config_repo = args.config_repo.expanduser().resolve()
        if not (args.candidate_env / "bin" / "vllm").is_file():
            raise RunnerError("--candidate-env must contain bin/vllm")
        if not (args.candidate_env / "bin" / "python").is_file():
            raise RunnerError("--candidate-env must contain bin/python for XPU proof")
        if not args.exploratory and args.correctness is None:
            raise RunnerError("--correctness is required unless --exploratory is set")
        if args.correctness is not None and not args.correctness.is_file():
            raise RunnerError("--correctness must be a readable file")
        if args.max_model_len != 65536:
            raise RunnerError("the current auto/native foreground launchers are 65,536")
        if args.max_num_batched_tokens < 1:
            raise RunnerError("max num batched tokens must be positive")
        if not args.exploratory and (
            set(args.context) != set(DEFAULT_CONTEXTS)
            or set(args.batch) != set(DEFAULT_BATCHES)
        ):
            raise RunnerError(
                "formal performance qualification requires the complete B1/B4 "
                "4K/16K/32K/65K matrix"
            )
        if not args.exploratory and (
            args.model != DEFAULT_MODEL
            or args.model_revision != DEFAULT_MODEL_REVISION
            or args.served_model != "sunny-chat"
        ):
            raise RunnerError(
                "formal performance qualification requires the pinned Brutus model"
            )
        if not args.exploratory and (
            args.output_tokens != 512 or args.waves_per_run < 2
        ):
            raise RunnerError(
                "formal performance qualification requires 512 output tokens and "
                "at least two measured waves"
            )
        if (
            not args.exploratory
            and args.num_warmups is not None
            and args.num_warmups < max(args.batch)
        ):
            raise RunnerError(
                "formal performance qualification requires a full-width warmup"
            )
        if (
            args.startup_attempts < 1
            or args.num_warmups is not None
            and args.num_warmups < 0
        ):
            raise RunnerError(
                "startup attempts must be positive and warmups non-negative"
            )
        if not 0.5 <= args.parity_ratio <= 1.0:
            raise RunnerError("parity ratio must be in [0.5, 1.0]")
        if args.min_parity_pairs < 2:
            raise RunnerError("minimum parity pairs must be at least two")
        if not args.exploratory and (
            args.min_throughput_ratio < 0.95
            or args.min_request_decode_ratio < 0.95
            or args.max_latency_ratio > 1.10
            or args.parity_ratio < 0.98
            or args.min_parity_pairs < 4
        ):
            raise RunnerError(
                "formal thresholds must be at least 0.95 throughput/decode, "
                "0.98 parity with four pairs, and no more than 1.10 latency"
            )
        if not args.exploratory and args.repeats < 2 * args.min_parity_pairs:
            raise RunnerError(
                "formal repeats must provide at least the requested ABBA pairs"
            )
        if (
            min(
                args.startup_timeout,
                args.readiness_poll_interval,
                args.shutdown_timeout,
                args.benchmark_timeout,
                args.metrics_poll_interval,
            )
            <= 0
        ):
            raise RunnerError("timeouts and polling intervals must be positive")
        build_plan(
            contexts=args.context,
            batches=args.batch,
            output_tokens=args.output_tokens,
            waves_per_run=args.waves_per_run,
            repeats=args.repeats,
            seed=args.seed,
            max_model_len=args.max_model_len,
            minimum_repeats=2 if args.exploratory else 4,
        )
    except (RunnerError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    args.runtime_cache.mkdir(parents=True, exist_ok=True)
    return args


def result_exit_code(result: Mapping[str, Any]) -> int:
    if result.get("evidence_mode") == "exploratory":
        return 0
    if result.get("performance_status") == "failed":
        return 1
    if result.get("statistical_parity_status") != "passed":
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.supervisor = ProcessSupervisor()
    args.supervisor.install_signal_handlers()
    try:
        try:
            result = execute(args)
        except (RunnerError, OSError, subprocess.SubprocessError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    finally:
        args.supervisor.signal_all(signal.SIGTERM)
        args.supervisor.restore_signal_handlers()
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return result_exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
