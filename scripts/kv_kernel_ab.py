#!/usr/bin/env python3
"""Crash-safe, sequential vLLM/XPU cache-kernel A/B harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


FULL_ATTENTION_LAYERS = [3 + 4 * index for index in range(16)]


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def request_json(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    body = canonical(payload) if payload is not None else None
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def default_plan() -> list[dict]:
    skip = [str(layer) for layer in FULL_ATTENTION_LAYERS]
    mtp = ["--speculative-config", '{"method":"mtp","num_speculative_tokens":2}']
    return [
        {"name": "reference-bf16-kv-eager-no-mtp", "skip_layers": skip,
         "extra_args": ["--enforce-eager"]},
        {"name": "fp8-kv-eager-no-mtp", "skip_layers": [],
         "extra_args": ["--enforce-eager"]},
        {"name": "bf16-kv-eager-mtp", "skip_layers": skip,
         "extra_args": ["--enforce-eager", *mtp]},
        {"name": "fp8-kv-eager-mtp", "skip_layers": [],
         "extra_args": ["--enforce-eager", *mtp]},
        {"name": "bf16-kv-graph-mtp", "skip_layers": skip, "extra_args": mtp},
        {"name": "fp8-kv-graph-mtp", "skip_layers": [], "extra_args": mtp},
    ]


def default_prompts() -> list[dict]:
    return [
        {"name": "decode-768", "messages": [
            {"role": "system", "content": "Be precise. Produce only valid Python."},
            {"role": "user", "content":
             "Implement a dependency-free Python red-black tree with insert, delete, lookup, "
             "an invariant checker, and a deterministic randomized self-test."},
        ], "max_tokens": 768},
        {"name": "turn-refresh-768", "messages": [
            {"role": "system", "content": "Be precise. Produce only valid Python."},
            {"role": "user", "content": "Implement a dependency-free Python red-black tree."},
            {"role": "assistant", "content": "I will provide the implementation."},
            {"role": "user", "content":
             "Continue now with insert, delete, invariant checks, and a deterministic self-test."},
        ], "max_tokens": 768},
    ]


def model_identity(model: str) -> dict:
    root = Path(model)
    identity = {"model": model}
    if root.is_dir():
        for name in ("config.json", "model.safetensors.index.json", "tokenizer_config.json"):
            path = root / name
            if path.is_file():
                identity[name] = sha256_file(path)
    return identity


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_candidate(args, root: Path, candidate: dict, prompts: list[dict]) -> dict:
    candidate_root = root / "candidates" / candidate["name"]
    result_path = candidate_root / "result.json"
    identity = {
        "schema": 1,
        "model": model_identity(args.model),
        "candidate": candidate,
        "prompts_sha256": hashlib.sha256(canonical(prompts)).hexdigest(),
        "vllm": str(Path(args.vllm).resolve()),
        "vllm_sha256": sha256_file(Path(args.vllm).resolve()),
    }
    identity_hash = hashlib.sha256(canonical(identity)).hexdigest()
    if result_path.is_file():
        previous = json.loads(result_path.read_text())
        if previous.get("status") == "complete" and previous.get("identity_sha256") == identity_hash:
            return previous

    candidate_root.mkdir(parents=True, exist_ok=True)
    # Triton's cached XPU driver extension is linked to the active oneAPI SYCL
    # ABI.  Reusing ~/.triton across Nix package revisions can load an old
    # spirv_utils.so (for example libsycl.so.8 with a libsycl.so.9 closure) and
    # make model inspection fail before the artifact is touched.  Keep runtime
    # compile caches package-keyed and shared only by candidates using the exact
    # same vLLM wrapper.
    runtime_root = root / "runtime-cache" / identity["vllm_sha256"][:16]
    environment = os.environ.copy()
    environment.update({
        "HOME": str(runtime_root / "home"),
        "XDG_CACHE_HOME": str(runtime_root / "xdg-cache"),
        "VLLM_CACHE_ROOT": str(runtime_root / "vllm-cache"),
    })
    for path in environment["HOME"], environment["XDG_CACHE_HOME"], environment["VLLM_CACHE_ROOT"]:
        Path(path).mkdir(parents=True, exist_ok=True)
    port = args.port
    command = [
        args.vllm, "serve", args.model, "--host", "127.0.0.1", "--port", str(port),
        "--served-model-name", args.served_model_name,
        "--dtype", "bfloat16", "--quantization", "compressed-tensors",
        "--kv-cache-dtype", "auto", "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", "1", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
    ]
    skip_layers = candidate.get("skip_layers", [])
    if skip_layers:
        command.extend(["--kv-cache-dtype-skip-layers", *skip_layers])
    command.extend(candidate.get("extra_args", []))
    command.extend(args.extra_vllm_arg)
    running = {"schema": 1, "status": "running", "identity": identity,
               "identity_sha256": identity_hash, "command": command,
               "started_at": time.time()}
    atomic_json(result_path, running)
    log_path = candidate_root / "server.log"
    with log_path.open("wb", buffering=0) as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env=environment,
        )
        try:
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"vLLM exited during startup with {process.returncode}")
                try:
                    request_json(f"http://127.0.0.1:{port}/v1/models", timeout=2)
                    break
                except (OSError, urllib.error.URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError("vLLM startup timed out")

            outputs = []
            for prompt in prompts:
                payload = {
                    "model": args.served_model_name,
                    "messages": prompt["messages"],
                    "temperature": 0,
                    "seed": args.seed,
                    "max_tokens": prompt["max_tokens"],
                    "logprobs": True,
                    "top_logprobs": 5,
                }
                started = time.monotonic()
                response = request_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    payload, timeout=args.request_timeout,
                )
                outputs.append({"prompt": prompt["name"], "request": payload,
                                "response": response,
                                "elapsed_seconds": time.monotonic() - started})
                atomic_json(candidate_root / f"{prompt['name']}.json", outputs[-1])
            result = {**running, "status": "complete", "completed_at": time.time(),
                      "outputs": outputs, "server_log": "server.log"}
        except BaseException as error:
            result = {**running, "status": "failed", "completed_at": time.time(),
                      "error": repr(error), "server_returncode": process.poll(),
                      "server_log": "server.log"}
            atomic_json(result_path, result)
            raise
        finally:
            terminate(process)
    atomic_json(result_path, result)
    return result


def content(result: dict, prompt_index: int) -> str:
    return result["outputs"][prompt_index]["response"]["choices"][0]["message"]["content"]


def token_trace(result: dict, prompt_index: int) -> list[dict]:
    choice = result["outputs"][prompt_index]["response"]["choices"][0]
    return (choice.get("logprobs") or {}).get("content") or []


def normalized_top(entry: dict) -> dict[str, float]:
    values = {
        item["token"]: math.exp(float(item["logprob"]))
        for item in (entry.get("top_logprobs") or [])
    }
    total = sum(values.values())
    return {token: probability / total for token, probability in values.items()} if total else {}


def trace_metrics(reference: list[dict], candidate: list[dict]) -> dict:
    count = min(len(reference), len(candidate))
    first_divergence = None
    agreement = 0
    kls = []
    for index in range(count):
        if reference[index].get("token") == candidate[index].get("token"):
            agreement += 1
        elif first_divergence is None:
            first_divergence = index
        left, right = normalized_top(reference[index]), normalized_top(candidate[index])
        if left and right:
            epsilon = 1e-12
            vocabulary = set(left) | set(right)
            kls.append(sum(
                left.get(token, epsilon)
                * math.log(left.get(token, epsilon) / right.get(token, epsilon))
                for token in vocabulary
            ))
    return {
        "reference_tokens": len(reference), "candidate_tokens": len(candidate),
        "compared_tokens": count, "top_token_agreement": agreement / count if count else None,
        "first_divergence_token": first_divergence,
        "mean_truncated_top5_kl": sum(kls) / len(kls) if kls else None,
    }


def repetition_metrics(text: str) -> dict:
    words = text.split()
    grams = [tuple(words[index:index + 4]) for index in range(max(0, len(words) - 3))]
    repeated = len(grams) - len(set(grams))
    return {
        "characters": len(text), "words": len(words),
        "repeated_fourgram_fraction": repeated / len(grams) if grams else 0.0,
        "unicode_replacement_count": text.count("\ufffd"),
        "nonprinting_count": sum(not character.isprintable() and not character.isspace() for character in text),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--vllm", required=True)
    parser.add_argument("--served-model-name", default="kv-kernel-ab")
    parser.add_argument("--plan-json")
    parser.add_argument("--prompts-json")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--seed", type=int, default=739391)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--extra-vllm-arg", action="append", default=[])
    args = parser.parse_args()

    root = Path(args.report_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(args.plan_json).read_text()) if args.plan_json else default_plan()
    prompts = json.loads(Path(args.prompts_json).read_text()) if args.prompts_json else default_prompts()
    plan_manifest = {"schema": 1, "status": "running", "plan": plan,
                     "prompts": prompts, "started_at": time.time()}
    atomic_json(root / "manifest.json", plan_manifest)
    results = []
    try:
        for candidate in plan:
            results.append(run_candidate(args, root, candidate, prompts))
        reference = results[0]
        comparisons = []
        for result in results[1:]:
            per_prompt = []
            for index, prompt in enumerate(prompts):
                left, right = content(reference, index), content(result, index)
                common = 0
                for a, b in zip(left, right):
                    if a != b:
                        break
                    common += 1
                per_prompt.append({"prompt": prompt["name"], "exact_match": left == right,
                                   "common_prefix_chars": common,
                                   "reference": repetition_metrics(left),
                                   "candidate": repetition_metrics(right),
                                   "tokens": trace_metrics(
                                       token_trace(reference, index), token_trace(result, index)
                                   )})
            comparisons.append({"candidate": result["identity"]["candidate"]["name"],
                                "prompts": per_prompt})
        plan_manifest.update(status="complete", completed_at=time.time(), results=results,
                             comparisons=comparisons)
    except BaseException as error:
        plan_manifest.update(status="failed", completed_at=time.time(), error=repr(error),
                             results=results)
        raise
    finally:
        atomic_json(root / "manifest.json", plan_manifest)


if __name__ == "__main__":
    main()
