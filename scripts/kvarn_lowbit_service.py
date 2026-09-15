#!/usr/bin/env python3
"""Capture serial or concurrent low-bit serving trials with the retained harness.

The plan freezes the model, server flags, token inputs, repetitions and output
length. Capture arms serially in ABBA order, then audit and compare their raw
artifacts before making a performance claim.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import kvarn_perf_run as perf
from scripts import kvarn_service_gate as service_gate
from scripts.kvarn_vision_run import MemorySampler, runtime_environment


def compilation_events(text):
    return [
        line
        for line in text.splitlines()
        if ("jit_monitor.py:" in line and "during inference:" in line)
        or "Triton autotuning for function" in line
        or "Autotuning kernel " in line
        or "Autotuning failed with " in line
    ]


def stream_request(base, body, output, name, release):
    request = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    perf.write_json_atomic(output / f"{name}-request.json", body)
    release.wait()
    started = time.monotonic()
    first = last = None
    first_count = 0
    tokens, content = [], ""
    usage, finish, done = None, None, False
    raw_path = output / f"{name}-sse.jsonl"
    with (
        raw_path.open("w") as raw,
        urllib.request.urlopen(request, timeout=1200) as response,
    ):
        for line in response:
            elapsed = time.monotonic() - started
            raw.write(json.dumps({"seconds": elapsed, "line": line.decode()}) + "\n")
            if line.strip() == b"data: [DONE]":
                done = True
                continue
            if not line.startswith(b"data: "):
                continue
            event = json.loads(line[6:])
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                chunk = choice.get("token_ids") or []
                piece = choice.get("text") or ""
                if chunk:
                    last = elapsed
                    if first is None:
                        first, first_count = elapsed, len(chunk)
                tokens.extend(chunk)
                content += piece
                finish = choice.get("finish_reason") or finish
    expected = {
        "prompt_tokens": len(body["prompt"]),
        "completion_tokens": body["max_tokens"],
        "total_tokens": len(body["prompt"]) + body["max_tokens"],
    }
    if (
        not done
        or finish != "length"
        or usage is None
        or any(usage.get(k) != v for k, v in expected.items())
        or len(tokens) != body["max_tokens"]
        or first is None
        or last <= first
    ):
        raise RuntimeError(f"incomplete token or stream evidence: {name}")
    result = {
        "id": name,
        "usage": usage,
        "token_ids": tokens,
        "content": content,
        "stream_done": done,
        "finish_reason": finish,
        "ttft_seconds": first,
        "last_token_seconds": last,
        "elapsed_seconds": time.monotonic() - started,
        "first_chunk_tokens": first_count,
        "decode_tokens_per_second": (len(tokens) - first_count) / (last - first),
        "quality_findings": service_gate.completion_quality_findings(
            tokens, len(tokens)
        ),
        "raw_stream_sha256": perf.sha256_file(raw_path),
        "request_sha256": perf.sha256_file(output / f"{name}-request.json"),
    }
    perf.write_json_atomic(output / f"{name}-response.json", result)
    return result


def run(args):
    plan = json.loads(args.plan.read_text())
    if not plan["trials"] or plan["repeats"] < 3 or plan["output_tokens"] < 128:
        raise ValueError(
            "requires trials, at least three repeats and 128 output tokens"
        )
    if "--jit-monitor-verbose" not in plan["server_args"]:
        raise ValueError("serving captures require complete JIT event logging")
    for trial in plan["trials"]:
        if Path(trial["id"]).name != trial["id"] or trial["id"] in (".", ".."):
            raise ValueError("trial IDs must be plain filenames")
        if trial["concurrency"] not in (1, 4) or not trial["prompt_token_ids"]:
            raise ValueError("requires B1/B4 trials with frozen prompt tokens")
    reserved = {
        "--kv-cache-dtype",
        "--host",
        "--port",
        "--served-model-name",
        "--revision",
    }
    if any(arg.split("=", 1)[0] in reserved for arg in plan["server_args"]):
        raise ValueError("server_args override an arm identity or owned endpoint")
    for unit in ("vllm-xpu-chat.service", "vllm-xpu-embedding.service"):
        state = subprocess.check_output(
            ["systemctl", "show", unit, "-p", "ActiveState", "--value"], text=True
        ).strip()
        if state != "inactive":
            raise RuntimeError(f"{unit} must be inactive")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = args.service_env.resolve(strict=True)
    argv = [
        str(runtime / "bin/vllm"),
        "serve",
        plan["model"],
        "--revision",
        plan["revision"],
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "sunny-chat",
        "--kv-cache-dtype",
        args.cache_dtype,
        *plan["server_args"],
    ]
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("KVARN_", "VLLM_")) and k != "PYTHONPATH"
    }
    selected = {
        **runtime_environment(),
        "VLLM_TARGET_DEVICE": "xpu",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "HF_HOME": "/var/cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "VLLM_CACHE_ROOT": str(output / "runtime-cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.update(selected)
    snapshot = output / "harness-source"
    snapshot.mkdir()
    for source in (ROOT / "scripts").glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    manifest = {
        "schema": "kvarn-lowbit-serving-v1",
        "memory_sampling_schema": "owned-drm-timestamped-v1",
        "plan_sha256": perf.sha256_file(args.plan),
        "plan": plan,
        "argv": argv,
        "cache_dtype": args.cache_dtype,
        "runtime": str(runtime),
        "selected_environment": selected,
        "started_unix": time.time(),
        "status": "starting",
        "waves": [],
        "harness_sha256": {p.name: perf.sha256_file(p) for p in snapshot.iterdir()},
    }
    perf.write_json_atomic(output / "manifest.json", manifest)
    base = f"http://127.0.0.1:{args.port}"
    control = SimpleNamespace(
        base_url=base,
        served_model="sunny-chat",
        startup_timeout=600,
        readiness_poll_interval=2,
        metrics_poll_interval=0.05,
    )
    perf.assert_port_unused(base)
    supervisor = perf.ProcessSupervisor()
    supervisor.install_signal_handlers()
    log = (output / "service.log").open("w")
    service = memory = None
    try:
        process = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        supervisor.register(process.pid, "low-bit serving qualification")
        service = perf.ServiceProcess(
            process,
            process.pid,
            log,
            output / "service.log",
            0,
            argv,
            selected,
            supervisor,
        )
        memory = MemorySampler(process.pid, output / "memory-fdinfo.jsonl")
        memory.start()
        perf.wait_for_ready(process, control)
        pid, actual, actual_env = perf.capture_engine_process(process)
        manifest.update(
            api_pid=pid,
            actual_argv=actual,
            actual_environment=actual_env,
            runtime_identity=perf.verify_candidate_identity(actual, runtime),
            status="ready",
        )
        for trial in plan["trials"]:
            b = trial["concurrency"]
            for repeat in range(plan["repeats"] + 1):
                name = f"{trial['id']}-r{repeat}"
                phase = "warmup" if repeat == 0 else "performance"
                perf.wait_for_scheduler_idle(control)
                memory.check()
                before = perf.http_text(base + "/metrics", timeout=10)
                (output / f"{name}-metrics-before.txt").write_text(before)
                release, stop = threading.Event(), threading.Event()
                samples, errors = [], []
                sampler = threading.Thread(
                    target=perf.sample_scheduler,
                    kwargs={
                        "args": control,
                        "stop": stop,
                        "samples": samples,
                        "errors": errors,
                    },
                    daemon=True,
                )
                sampler.start()
                body = {
                    "model": "sunny-chat",
                    "prompt": trial["prompt_token_ids"],
                    "temperature": 0,
                    "seed": 42,
                    "max_tokens": plan["output_tokens"],
                    "ignore_eos": True,
                    "return_token_ids": True,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                }
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=b
                    ) as executor:
                        futures = [
                            executor.submit(
                                stream_request,
                                base,
                                body,
                                output,
                                f"{name}-q{q}",
                                release,
                            )
                            for q in range(b)
                        ]
                        log_start = (output / "service.log").stat().st_size
                        started_unix = time.time()
                        started = time.monotonic()
                        release.set()
                        results = [f.result() for f in futures]
                        elapsed = time.monotonic() - started
                finally:
                    stop.set()
                    sampler.join(timeout=5)
                if sampler.is_alive() or errors:
                    raise RuntimeError(f"scheduler sampling failed: {errors}")
                memory.check()
                peak = max((s["running"] for s in samples), default=0)
                if peak < b:
                    raise RuntimeError(f"missing live B{b} overlap: {name}")
                perf.wait_for_scheduler_idle(control)
                after = perf.http_text(base + "/metrics", timeout=10)
                (output / f"{name}-metrics-after.txt").write_text(after)
                perf.write_json_atomic(output / f"{name}-scheduler.json", samples)
                pre = perf.parse_scheduler_metrics(before)
                post = perf.parse_scheduler_metrics(after)
                if (
                    post["vllm:num_preemptions_total"]
                    != pre["vllm:num_preemptions_total"]
                ):
                    raise RuntimeError(
                        f"preemption during matched performance trial: {name}"
                    )
                log_end = (output / "service.log").stat().st_size
                with (output / "service.log").open("rb") as service_log:
                    service_log.seek(log_start)
                    log_text = service_log.read(log_end - log_start).decode(
                        errors="replace"
                    )
                compile_events = compilation_events(log_text)
                wave = {
                    "id": name,
                    "trial": trial["id"],
                    "phase": phase,
                    "concurrency": b,
                    "peak_running": peak,
                    "request_ids": [r["id"] for r in results],
                    "seconds": elapsed,
                    "started_unix": started_unix,
                    "service_log_range": [log_start, log_end],
                    "compilation_events": compile_events,
                    "output_tokens_per_second": b * plan["output_tokens"] / elapsed,
                    "evidence_sha256": {
                        filename: perf.sha256_file(output / filename)
                        for filename in (
                            f"{name}-metrics-before.txt",
                            f"{name}-metrics-after.txt",
                            f"{name}-scheduler.json",
                        )
                    },
                }
                manifest["waves"].append(wave)
                perf.write_json_atomic(output / "manifest.json", manifest)
                print(json.dumps(wave), flush=True)
                if repeat and compile_events:
                    raise RuntimeError(
                        f"compilation during measured trial: {name}; warm and recapture"
                    )
        manifest["status"] = "captured-not-qualified"
    except BaseException as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        if service is not None:
            perf.stop_service(service, 30)
        else:
            log.close()
        try:
            if memory is not None:
                memory.finish()
        except BaseException as error:
            manifest.update(status="failed", error=str(error))
            raise
        finally:
            supervisor.restore_signal_handlers()
            manifest["finished_unix"] = time.time()
            manifest["service_log_sha256"] = perf.sha256_file(output / "service.log")
            memory_path = output / "memory-fdinfo.jsonl"
            if memory_path.exists():
                manifest["memory_sha256"] = perf.sha256_file(memory_path)
            perf.write_json_atomic(output / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--service-env", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dtype",
        choices=("kvarn_k4v4_g128_compact", "kvarn_k4v2_g128_compact"),
        required=True,
    )
    parser.add_argument("--port", type=int, default=18000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
