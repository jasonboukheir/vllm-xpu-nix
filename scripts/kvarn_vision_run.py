#!/usr/bin/env python3
"""Isolated vision qualification; does not relax the text-only perf contract."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import kvarn_perf_run as perf

MODEL = "jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound"
REVISION = "6b0622f4354481d5d04577d48ba0db844efc1330"
TRANSPORT_ENVIRONMENT = {
    "CCL_ATL_TRANSPORT": "ofi",
    "CCL_LOG_LEVEL": "warn",
    "CCL_PROCESS_LAUNCHER": "none",
    "CCL_ZE_IPC_EXCHANGE": "sockets",
}


def runtime_environment() -> dict[str, str]:
    """Return qualified transport settings; the dtype alone selects KVarN."""
    return dict(TRANSPORT_ENVIRONMENT)


def fixtures(directory: Path) -> list[dict]:
    """Deterministic image-grounded inputs with explicit expected answers."""
    from PIL import Image, ImageDraw, ImageFont

    directory.mkdir()
    font = ImageFont.load_default(size=40)
    for name, left, right, label in (
        ("a", "red", "blue", "SUN 482"),
        ("b", "green", "yellow", "MOON 715"),
    ):
        image = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((35, 70, 185, 220), fill=left)
        draw.ellipse((260, 70, 410, 220), fill=right)
        draw.text((40, 310), label, font=font, fill="black")
        image.save(directory / f"{name}.png")

    def image_part(name: str) -> dict:
        data = base64.b64encode((directory / f"{name}.png").read_bytes()).decode()
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{data}"},
        }

    question = (
        "Describe the two colored shapes and their left/right positions. "
        "Transcribe the printed text exactly. Be brief."
    )
    cases = []
    for name, expected in (
        ("a", ["red", "square", "blue", "circle", "SUN 482"]),
        ("b", ["green", "square", "yellow", "circle", "MOON 715"]),
    ):
        cases.append(
            {
                "id": f"image-{name}",
                "expected_terms": expected,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_part(name),
                            {"type": "text", "text": question},
                        ],
                    }
                ],
            }
        )
    cases.append(
        {
            "id": "two-images",
            "expected_terms": ["SUN 482", "MOON 715"],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_part("a"),
                        image_part("b"),
                        {
                            "type": "text",
                            "text": "Read the printed text in image 1, then image 2. Be brief.",
                        },
                    ],
                }
            ],
        }
    )
    cases.append(
        {
            "id": "multi-turn",
            "expected_terms": ["blue"],
            "messages": [
                cases[0]["messages"][0],
                {
                    "role": "assistant",
                    "content": "There are two shapes and some printed text.",
                },
                {
                    "role": "user",
                    "content": "What color is the shape on the right in that image? Answer with just the color.",
                },
            ],
        }
    )
    cases.append(
        {
            "id": "text-reuse",
            "expected_terms": ["42"],
            "messages": [
                {
                    "role": "user",
                    "content": "What is 6 times 7? Answer with just the number.",
                }
            ],
        }
    )
    cases.append(
        {
            "id": "count-shapes",
            "expected_terms": [],
            "expected_regex": r"(?i)\b(2|two)\b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        image_part("b"),
                        {
                            "type": "text",
                            "text": "How many colored shapes are visible? Answer with just the number.",
                        },
                    ],
                }
            ],
        }
    )
    return cases


def post_json(base_url: str, route: str, body: dict) -> dict:
    request = urllib.request.Request(
        base_url + route,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


def tokenize_case(base_url: str, case: dict) -> dict:
    return post_json(
        base_url,
        "/tokenize",
        {
            "model": "sunny-vision",
            "messages": case["messages"],
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )


def long_case(base_url: str, source: dict, output: Path) -> dict:
    """Put image beyond sink, then age it; cross a 128-token page in decode."""
    case = copy.deepcopy(source)
    case["id"] = "long-" + source["id"]
    parts = case["messages"][0]["content"]
    parts.insert(0, {"type": "text", "text": "Unrelated context:\n" + "context " * 256})
    parts.insert(2, {"type": "text", "text": "\nIgnore this padding:\n"})
    count = 0
    for _ in range(6):
        parts[2]["text"] = (
            "\nIgnore this padding:\n"
            + " x" * count
            + "\nNow answer about the image:\n"
        )
        tokenized = tokenize_case(base_url, case)
        delta = 6143 - tokenized["count"]
        if delta == 0:
            tokens = tokenized["tokens"]
            image_positions = [i for i, token in enumerate(tokens) if token == 248056]
            if (
                not image_positions
                or min(image_positions) <= 128
                or max(image_positions) >= 2048
            ):
                raise ValueError(
                    "long image must be beyond sink and in first prefill chunk"
                )
            case["coverage"] = {
                "prompt_tokens": 6143,
                "page_size": 128,
                "image_token_start": min(image_positions),
                "image_token_end_inclusive": max(image_positions),
                "expected_chunk_count": 3,
                "decode_crosses_page_after_tokens": 1,
            }
            perf.write_json_atomic(output / f"{case['id']}-tokenize.json", tokenized)
            return case
        count += delta
        if count < 0:
            raise ValueError("long-context base prompt exceeds target length")
    raise ValueError("failed to construct exact 6143-token vision prompt")


class MemorySampler:
    """Read kernel fdinfo for owned service clients; never submit GPU work."""

    def __init__(self, process_group: int, output: Path):
        self.group = process_group
        self.output = output
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        started = time.monotonic()
        with self.output.open("w") as stream:
            while not self.stop.is_set():
                clients = {}
                errors = []
                for pid in perf._process_group_members(self.group):
                    try:
                        paths = list(Path(f"/proc/{pid}/fdinfo").iterdir())
                    except (FileNotFoundError, PermissionError) as error:
                        errors.append(str(error))
                        continue
                    for path in paths:
                        try:
                            fields = dict(
                                line.split(":", 1)
                                for line in path.read_text().splitlines()
                                if line.startswith("drm-")
                            )
                        except (FileNotFoundError, PermissionError):
                            continue
                        fields = {key: value.strip() for key, value in fields.items()}
                        if "drm-client-id" in fields:
                            key = (
                                fields.get("drm-pdev", "")
                                + ":"
                                + fields["drm-client-id"]
                            )
                            clients[key] = {"pid": pid, "fields": fields}
                stream.write(
                    json.dumps(
                        {
                            "seconds": time.monotonic() - started,
                            "clients": clients,
                            "errors": errors,
                        }
                    )
                    + "\n"
                )
                stream.flush()
                self.stop.wait(0.5)

    def start(self):
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("memory sampler failed to stop")


def request_case(base_url: str, case: dict, output: Path) -> dict:
    body = {
        "model": "sunny-vision",
        "messages": case["messages"],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body.update(case.get("generation", {}))
    perf.write_json_atomic(output / f"{case['id']}-request.json", body)
    request = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    first = None
    content = ""
    usage = None
    events = []
    done = False
    finish_reason = None
    raw = (output / f"{case['id']}-sse.jsonl").open("w")
    with raw, urllib.request.urlopen(request, timeout=600) as response:
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
                raise ValueError(f"stream error: {event['error']}")
            events.append({"seconds": elapsed, "event": event})
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta", {})
                piece = delta.get("content") or ""
                if first is None and (
                    piece or delta.get("reasoning_content") or delta.get("reasoning")
                ):
                    first = elapsed
                content += piece
    result = {
        "id": case["id"],
        "content": content,
        "usage": usage,
        "ttft_seconds": first,
        "total_seconds": time.monotonic() - started,
        "expected_terms": case["expected_terms"],
        "term_check": all(
            term.lower() in content.lower() for term in case["expected_terms"]
        ),
        "finish_reason": finish_reason,
        "stream_done": done,
        "coverage": case.get("coverage"),
        "phase": case.get("phase", "correctness"),
        "events": events,
    }
    import re

    if case.get("expected_regex"):
        result["term_check"] &= bool(re.search(case["expected_regex"], content))
    result["stream_valid"] = bool(
        done
        and usage
        and usage.get("completion_tokens", 0) > 0
        and first is not None
        and finish_reason in ("stop", "length")
    )
    if result["stream_valid"]:
        result["decode_tokens_per_second"] = (usage["completion_tokens"] - 1) / (
            result["total_seconds"] - first
        )
    if case.get("coverage"):
        result["coverage_valid"] = (
            usage is not None
            and usage["prompt_tokens"] == 6143
            and usage["completion_tokens"] > 1
        )
    perf.write_json_atomic(output / f"{case['id']}-response.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "events"}), flush=True)
    if not result["stream_valid"]:
        raise ValueError(f"incomplete or invalid stream for {case['id']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--service-env",
        type=Path,
        required=True,
        help="built vLLM environment containing bin/vllm",
    )
    parser.add_argument(
        "--cache-dtype", choices=["auto", perf.COMPACT_DTYPE], default="auto"
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "harness-source"
    snapshot.mkdir()
    for name in (
        "kvarn_vision_run.py",
        "kvarn_perf_run.py",
        "kvarn_perf_gate.py",
        "kvarn_split_policy.py",
        "kvarn_scan_engine_log.py",
    ):
        shutil.copy2(ROOT / "scripts" / name, snapshot / name)
    cases = fixtures(output / "images")
    perf.write_json_atomic(output / "workload.json", cases)
    selected_env = runtime_environment()
    selected_env.update(
        {
            "VLLM_TARGET_DEVICE": "xpu",
            "VLLM_USE_V2_MODEL_RUNNER": "0",
            "HF_HOME": "/var/cache/huggingface",
            "HF_HUB_OFFLINE": "1",
            "VLLM_CACHE_ROOT": str(output / "runtime-cache"),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("KVARN_", "VLLM_")) and k != "PYTHONPATH"
    }
    env.update(selected_env)
    service_env = args.service_env.resolve(strict=True)
    argv = [
        str(service_env / "bin/vllm"),
        "serve",
        MODEL,
        "--revision",
        REVISION,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        "sunny-vision",
        "--dtype",
        "bfloat16",
        "--quantization",
        "compressed-tensors",
        "--kv-cache-dtype",
        args.cache_dtype,
        "--gpu-memory-utilization",
        "0.90",
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "2048",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt",
        '{"image":2,"video":0}',
        "--mm-processor-kwargs",
        '{"min_pixels":200704,"max_pixels":200704}',
        "--reasoning-parser",
        "qwen3",
        "--enable-prompt-tokens-details",
    ]
    manifest = {
        "schema": "kvarn-vision-experiment-v1",
        "argv": argv,
        "selected_environment": selected_env,
        "service_env": str(service_env),
        "transport_environment": TRANSPORT_ENVIRONMENT,
        "kvarn_selection": "dtype-defaults-no-environment-overrides",
        "harness_sha256": {p.name: perf.sha256_file(p) for p in snapshot.iterdir()},
        "workload_sha256": perf.sha256_file(output / "workload.json"),
        "image_sha256": {
            p.name: perf.sha256_file(p) for p in (output / "images").iterdir()
        },
        "status": "starting",
    }
    perf.write_json_atomic(output / "manifest.json", manifest)
    args.base_url = f"http://127.0.0.1:{args.port}"
    args.served_model = "sunny-vision"
    args.startup_timeout = 600
    args.readiness_poll_interval = 2
    perf.assert_port_unused(args.base_url)
    supervisor = perf.ProcessSupervisor()
    supervisor.install_signal_handlers()
    log = (output / "service.log").open("w")
    service = None
    memory = None
    try:
        process = subprocess.Popen(
            argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        supervisor.register(process.pid, "vision service")
        service = perf.ServiceProcess(
            process,
            process.pid,
            log,
            output / "service.log",
            0,
            argv,
            selected_env,
            supervisor,
        )
        memory = MemorySampler(process.pid, output / "memory-fdinfo.jsonl")
        memory.start()
        print(f"Service PID {process.pid}; log: {output / 'service.log'}", flush=True)
        perf.wait_for_ready(process, args)
        pid, actual_argv, actual_env = perf.capture_engine_process(process)
        manifest.update(
            {
                "api_pid": pid,
                "actual_argv": actual_argv,
                "actual_environment": actual_env,
                "status": "ready",
                "runtime_identity": perf.verify_candidate_identity(
                    actual_argv, service_env
                ),
            }
        )
        if args.qualify:
            cases.extend(
                long_case(args.base_url, source, output) for source in cases[:2]
            )
            image_perf = copy.deepcopy(cases[0])
            image_perf["messages"][0]["content"][-1]["text"] = (
                "Describe this image in detail in 200 words."
            )
            text_perf = {
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain how rainbows form in 200 words.",
                    }
                ],
                "expected_terms": [],
            }
            for index in range(4):
                for kind, source in (("image", image_perf), ("text", text_perf)):
                    case = copy.deepcopy(source)
                    case.update(
                        {
                            "id": f"perf-{kind}-{index}",
                            "expected_terms": [],
                            "phase": "warmup" if index == 0 else "performance",
                            "generation": {"max_tokens": 96, "ignore_eos": True},
                        }
                    )
                    cases.append(case)
            perf.write_json_atomic(output / "workload.json", cases)
            manifest["workload_sha256"] = perf.sha256_file(output / "workload.json")
        perf.write_json_atomic(output / "manifest.json", manifest)
        results = []
        for case in cases:
            results.append(request_case(args.base_url, case, output))
        (output / "metrics.txt").write_text(
            perf.http_text(args.base_url + "/metrics", timeout=10)
        )
        manifest["status"] = "requests-completed-not-yet-qualified"
        manifest["all_term_checks"] = all(result["term_check"] for result in results)
    except BaseException as error:
        manifest.update({"status": "failed", "error": str(error)})
        raise
    finally:
        if service is not None:
            perf.stop_service(service, 30)
        else:
            log.close()
        if memory is not None:
            memory.finish()
        supervisor.restore_signal_handlers()
        manifest["service_log_sha256"] = perf.sha256_file(output / "service.log")
        perf.write_json_atomic(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
