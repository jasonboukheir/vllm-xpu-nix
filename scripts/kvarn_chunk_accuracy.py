"""Serial teacher-forced chunk study against the model's unquantized KV cache."""

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts import kvarn_compare_logits as logits
from scripts import kvarn_mtp_compare as mtp
from scripts import kvarn_perf_run as perf
from scripts import kvarn_vision_compare as vision
from scripts import kvarn_vision_run as service
from scripts.kvarn_scan_engine_log import scan


def engine_options(budget, dtype):
    return {
        "revision": service.REVISION,
        "dtype": "bfloat16",
        "quantization": "compressed-tensors",
        "kv_cache_dtype": dtype,
        "max_model_len": 65536,
        "max_num_batched_tokens": budget,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.90,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "seed": 42,
    }


def schedule(lengths, budgets, starts):
    # Complete the 2K/8K screen at both lengths before the secondary 4K arm.
    priority = [b for b in (2048, 8192) if b in budgets]
    secondary = [b for b in budgets if b not in priority]
    return [
        {"length": length, "budget": budget, "dtype": dtype, "start": start}
        for group in (priority, secondary)
        for start in range(1, starts + 1)
        for length in lengths
        for budget in (group if start % 2 else list(reversed(group)))
        for dtype in (
            ("auto", perf.COMPACT_DTYPE) if start % 2 else (perf.COMPACT_DTYPE, "auto")
        )
    ]


def compare_pair(reference, candidate, thresholds, tolerance):
    report = logits.compare(
        logits.load_artifact(reference),
        logits.load_artifact(candidate),
        tie_tolerance=tolerance,
        boundaries=(4096, 16384, 32768),
    )
    report["acceptance"] = logits.evaluate_acceptance(report, thresholds)
    report["status"] = report["acceptance"]["status"]
    return report


def validate_artifact(path, expected_options, prompt, forced):
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["model"]) != service.MODEL
            or json.loads(str(data["engine_kwargs_json"])) != expected_options
            or not np.array_equal(data["prompt_token_ids"], prompt)
            or not np.array_equal(data["forced_token_ids"], forced)
            or bool(data["full_logits"])
        ):
            raise ValueError("replay identity or frozen tokens disagree with plan")
    logits.load_artifact(path)  # Validate all captured rows and finite scores.


def summarize(output, plan, completed):
    pairs = []
    sensitivities = []
    for length in plan["lengths"]:
        for start in range(1, plan["starts"] + 1):
            for budget in plan["budgets"]:
                auto = f"{length}-{budget}-auto-r{start}"
                kvarn = f"{length}-{budget}-kvarn-r{start}"
                if auto not in completed or kvarn not in completed:
                    continue
                report = compare_pair(
                    output / auto / "logits.npz",
                    output / kvarn / "logits.npz",
                    plan["thresholds"],
                    plan["tie_tolerance"],
                )
                name = f"compression-{length}-{budget}-r{start}.json"
                perf.write_json_atomic(output / name, report)
                pairs.append(
                    {
                        "length": length,
                        "start": start,
                        "budget": budget,
                        "report": name,
                        "status": report["status"],
                        "top1_agreement_rate": report["top1_agreement_rate"],
                        "selected_token_mae": report["selected_token_delta"]["mae"],
                        "matched_logit_rmse": report["matched_logit_delta"]["rmse"],
                    }
                )
                control = f"{length}-2048-auto-r{start}"
                if budget != 2048 and control in completed:
                    diagnostic = compare_pair(
                        output / control / "logits.npz",
                        output / auto / "logits.npz",
                        {},
                        plan["tie_tolerance"],
                    )
                    diagnostic["status"] = "diagnostic_only"
                    name = (
                        f"unquantized-chunk-sensitivity-{length}-{budget}-r{start}.json"
                    )
                    perf.write_json_atomic(output / name, diagnostic)
                    sensitivities.append(name)
    for row in pairs:
        control = next(
            (
                p
                for p in pairs
                if p["length"] == row["length"]
                and p["start"] == row["start"]
                and p["budget"] == 2048
            ),
            None,
        )
        if control:
            row["change_from_2k_compression_error"] = {
                metric: row[metric] - control[metric]
                for metric in (
                    "top1_agreement_rate",
                    "selected_token_mae",
                    "matched_logit_rmse",
                )
            }
    return {
        "schema": "kvarn-chunk-model-reference-v1",
        "comparisons": pairs,
        "unquantized_chunk_sensitivity": sensitivities,
        "qualified": False,
        "limitations": plan["limitations"],
    }


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime = args.service_env.resolve(strict=True)
    threshold_report = vision.load(args.threshold_report)
    thresholds = threshold_report["acceptance"]["thresholds"]
    if set(thresholds) != set(logits.ACCEPTANCE_THRESHOLD_SPECS):
        raise ValueError("requires the complete retained numerical threshold profile")
    shutil.copy2(args.threshold_report, output / "threshold-source.json")
    source = output / "source"
    shutil.copytree(
        Path(__file__).resolve().parent,
        source / "scripts",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    references = {}
    identity = None
    for directory in args.reference_run:
        directory = directory.resolve()
        manifest, responses = vision.audit(directory)
        argv = manifest["actual_argv"]
        if argv[argv.index("--kv-cache-dtype") + 1] != "auto":
            raise ValueError("teacher tokens must come from unquantized KV")
        if manifest.get("suite") != "text-prefill-performance-v1":
            raise ValueError("requires a retained text-prefill reference")
        actual = perf.verify_candidate_identity(argv, runtime)
        if actual != manifest["runtime_identity"] or (identity and identity != actual):
            raise ValueError("frozen service and replay runtime identities differ")
        identity = actual
        measured = next(r for r in responses if r["phase"] == "performance")
        prompt, forced = mtp.tokens(directory, measured["id"])
        if len(prompt) not in (16383, 65023) or len(forced) != 512:
            raise ValueError(
                "requires 16K/65K references with exactly 512 output tokens"
            )
        if len(prompt) in references:
            raise ValueError("duplicate reference length")
        inputs = output / f"inputs-{len(prompt)}"
        inputs.mkdir()
        perf.write_json_atomic(inputs / "prompt.json", prompt)
        perf.write_json_atomic(inputs / "forced.json", forced)
        references[len(prompt)] = {
            "prompt": prompt,
            "forced": forced,
            "path": str(directory),
        }
        shutil.copy2(directory / "manifest.json", inputs / "reference-manifest.json")
    plan = {
        "schema": "kvarn-chunk-model-reference-plan-v1",
        "created_unix": time.time(),
        "runtime_identity": identity,
        "lengths": sorted(references),
        "budgets": args.budgets,
        "starts": args.starts,
        "top_k": 50,
        "thresholds": thresholds,
        "tie_tolerance": threshold_report["tie_tolerance"],
        "source_sha256": {
            str(p.relative_to(source)): perf.sha256_file(p)
            for p in source.rglob("*.py")
        },
        "input_sha256": {
            str(p.relative_to(output)): perf.sha256_file(p)
            for p in output.glob("inputs-*/*.json")
        },
        "reference_captures": {str(k): v["path"] for k, v in references.items()},
        "reference_definition": "Same W4A16 model weights; auto BF16 KV at each budget.",
        "stop_rule": "Stop on process, invalid-logit, or artifact failure. Numerical failures are retained for diagnosis; never auto-promote. Free-generation wording is not a rejection gate.",
        "limitations": [
            "Top-50 raw logits plus forced token; no full-distribution KL claim.",
            "Synthetic two-length screen is not the full six-fixture model gate.",
            "Replay timing includes diagnostics and does not measure service performance.",
            "Packed state/GDN/attention gates, paired service timing, MTP2/images remain required.",
        ],
    }
    plan["runs"] = schedule(plan["lengths"], args.budgets, args.starts)
    perf.write_json_atomic(output / "plan.json", plan)
    status = {"status": "planned", "completed": [], "current": None}
    perf.write_json_atomic(output / "status.json", status)
    if args.plan_only:
        return
    supervisor = perf.ProcessSupervisor()
    supervisor.install_signal_handlers()
    try:
        for item in plan["runs"]:
            host = subprocess.run(
                ["systemctl", "is-active", "vllm-xpu-chat.service"],
                capture_output=True,
                text=True,
                check=False,
            )
            if host.stdout.strip() != "inactive":
                raise RuntimeError(
                    "production chat service must be inactive; no service was stopped"
                )
            length, budget, dtype, start = (
                item[k] for k in ("length", "budget", "dtype", "start")
            )
            label = "auto" if dtype == "auto" else "kvarn"
            name = f"{length}-{budget}-{label}-r{start}"
            directory = output / name
            directory.mkdir()
            options = engine_options(budget, dtype)
            perf.write_json_atomic(directory / "engine.json", options)
            command = [
                str(runtime / "bin/python"),
                "-m",
                "scripts.kvarn_forced_decode",
                "--model",
                service.MODEL,
                "--engine-kwargs",
                str(directory / "engine.json"),
                "--prompt-token-ids",
                str(output / f"inputs-{length}/prompt.json"),
                "--forced-token-ids",
                str(output / f"inputs-{length}/forced.json"),
                "--top-k",
                "50",
                "--output",
                str(directory / "logits.npz"),
            ]
            perf.write_json_atomic(directory / "command.json", command)
            env = {
                k: v
                for k, v in os.environ.items()
                if not k.startswith(("KVARN_", "VLLM_")) and k != "PYTHONPATH"
            }
            env.update(service.runtime_environment())
            env.update(
                {
                    "VLLM_TARGET_DEVICE": "xpu",
                    "VLLM_USE_V2_MODEL_RUNNER": "0",
                    "HF_HOME": "/var/cache/huggingface",
                    "HF_HUB_OFFLINE": "1",
                    "VLLM_CACHE_ROOT": str(directory / "runtime-cache"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "KVARN_FORCED_VALIDATE_FINITE": "1",
                }
            )
            status.update(status="running", current=name)
            perf.write_json_atomic(output / "status.json", status)
            print(f"START {name}", flush=True)
            with (directory / "run.log").open("x") as log:
                code = perf.run_managed_process(
                    command,
                    cwd=source,
                    environment=env,
                    output=log,
                    timeout=1800,
                    supervisor=supervisor,
                    label=name,
                )
            if code:
                raise RuntimeError(f"{name} exited {code}; inspect run.log")
            validate_artifact(
                directory / "logits.npz",
                options,
                references[length]["prompt"],
                references[length]["forced"],
            )
            log_scan = scan((directory / "run.log").read_text().splitlines())
            perf.write_json_atomic(directory / "engine-log-scan.json", log_scan)
            if log_scan["status"] != "passed":
                raise RuntimeError(f"{name} has fatal engine log findings")
            perf.write_json_atomic(
                directory / "hashes.json",
                {
                    p.name: perf.sha256_file(p)
                    for p in directory.iterdir()
                    if p.is_file()
                },
            )
            status["completed"].append(name)
            perf.write_json_atomic(
                output / "comparison-summary.json",
                summarize(output, plan, status["completed"]),
            )
            perf.write_json_atomic(output / "status.json", status)
            print(f"DONE {name}", flush=True)
        status.update(status="numerical-screen-completed-not-qualified", current=None)
    except BaseException as error:
        status.update(status="failed", error=str(error))
        raise
    finally:
        supervisor.restore_signal_handlers()
        perf.write_json_atomic(output / "status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-env", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, action="append", required=True)
    parser.add_argument("--threshold-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[2048, 8192, 4096])
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if (
        args.starts < 1
        or 2048 not in args.budgets
        or not set(args.budgets) <= {2048, 4096, 8192}
        or len(args.budgets) != len(set(args.budgets))
    ):
        parser.error("use distinct 2K/4K/8K budgets including 2K, and positive starts")
    run(args)


if __name__ == "__main__":
    main()
