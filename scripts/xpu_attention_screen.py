"""Gate and compare separately loaded attention policies on captured operands."""

import argparse
import json
import os
import random
import shutil
import statistics
import subprocess
import time
from pathlib import Path


class Worker:
    def __init__(self, python, library, output, label):
        self.log = (output / f"{label}.stderr.log").open("w")
        self.stdout_log = (output / f"{label}.stdout.log").open("w")
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (
            str(library.resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        self.process = subprocess.Popen(
            [
                str(python),
                "-m",
                "scripts.xpu_attention_worker",
                "--library",
                str(library),
            ],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.identity = self.read()
        if not self.identity.get("ready"):
            raise RuntimeError(self.identity)

    def read(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"worker exited: {self.process.poll()}")
            self.stdout_log.write(line)
            self.stdout_log.flush()
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in value:
                raise RuntimeError(value["error"])
            return value

    def call(self, **command):
        self.process.stdin.write(json.dumps(command) + "\n")
        self.process.stdin.flush()
        return self.read()

    def close(self):
        if self.process.poll() is None:
            self.call(action="quit")
            self.process.wait(timeout=30)
        self.log.close()
        self.stdout_log.close()


def summarize(blocks):
    controls = []
    candidates = []
    for block in blocks:
        c = statistics.mean(s["ms"] for s in block if s["arm"] == "control")
        q = statistics.mean(s["ms"] for s in block if s["arm"] == "candidate")
        controls.append(c)
        candidates.append(q)
    rng = random.Random(11)
    draws = []
    for _ in range(3000):
        indices = rng.choices(range(len(blocks)), k=len(blocks))
        draws.append(
            sum(controls[i] for i in indices) / sum(candidates[i] for i in indices)
        )
    draws.sort()
    third = max(1, len(blocks) // 3)
    return {
        "control_ms": statistics.mean(controls),
        "candidate_ms": statistics.mean(candidates),
        "speedup": statistics.mean(controls) / statistics.mean(candidates),
        "speedup_ci95": [draws[75], draws[2924]],
        "control_cv": statistics.stdev(controls) / statistics.mean(controls),
        "control_last_first_ratio": statistics.mean(controls[-third:])
        / statistics.mean(controls[:third]),
        "candidate_last_first_ratio": statistics.mean(candidates[-third:])
        / statistics.mean(candidates[:third]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--capture", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates-only", action="store_true")
    parser.add_argument("--gates-from", type=Path)
    parser.add_argument(
        "--control-gate-arm", default="control", choices=("original", "control")
    )
    parser.add_argument(
        "--candidate-gate-arm", default="candidate", choices=("control", "candidate")
    )
    parser.add_argument("--blocks", type=int, default=30)
    parser.add_argument("--indices", type=int, nargs="+", default=list(range(6)))
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("warm", "cold", "interleaved"),
        default=("warm", "cold", "interleaved"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("xpu_attention_screen.py", "xpu_attention_worker.py"):
        shutil.copy2(Path(__file__).with_name(name), args.output / name)
    if len(set(args.indices)) != len(args.indices) or any(
        i not in range(6) for i in args.indices
    ):
        raise ValueError("indices must select distinct captured cases from 0 through 5")
    cases = [(path.resolve(), i) for path in args.capture for i in args.indices]
    report = {
        "status": "running",
        "gates": [],
        "timings": [],
        "identities": [],
        "args": {k: str(v) for k, v in vars(args).items()},
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    workers = {}
    try:
        if args.gates_from:
            prior = json.loads(args.gates_from.read_text())
            if prior["status"] != "gates-passed":
                raise ValueError("prior gate run did not pass")
            report["gates"] = prior["gates"]
            report["gate_identities"] = prior["identities"]
            for arm in (args.control_gate_arm, args.candidate_gate_arm):
                passed = {
                    (g["manifest"], g["index"]): g
                    for g in prior["gates"]
                    if g["arm"] == arm
                }
                required = {(str(path), index) for path, index in cases}
                if not required <= passed.keys() or not all(
                    passed[key]["pass"] for key in required
                ):
                    raise ValueError("gate cases do not match requested timing cases")
        else:
            # Original full-coverage DSO also proves that narrowing the build and
            # replaying physical pages did not silently change the baseline.
            for arm in ("original", "control", "candidate"):
                worker = Worker(
                    args.python, getattr(args, arm), args.output, f"gate-{arm}"
                )
                workers[arm] = worker
                report["identities"].append(
                    {"phase": "gate", "arm": arm, **worker.identity}
                )
                for case_id, (manifest, index) in enumerate(cases):
                    worker.call(action="prepare", manifest=str(manifest), index=index)
                    result = worker.call(
                        action="validate",
                        reference=str(args.output / f"reference-{case_id}.pt"),
                        make_reference=arm == "original",
                    )
                    entry = {
                        "case": case_id,
                        "manifest": str(manifest),
                        "index": index,
                        "arm": arm,
                        **result,
                    }
                    report["gates"].append(entry)
                    save()
                    print(f"gate {arm} case {case_id}: {result}", flush=True)
                    if not result["pass"] or (
                        arm in ("original", "control")
                        and not result["capture_bit_exact"]
                    ):
                        report["status"] = "stopped-correctness"
                        return
                    worker.call(
                        action="trace",
                        path=str(
                            (args.output / f"trace-{arm}-{case_id}.json").resolve()
                        ),
                    )
                worker.close()
                del workers[arm]
            report["status"] = "gates-passed"
            save()
        if args.gates_only:
            return
        for start in range(2):
            arms = ["control", "candidate"] if start == 0 else ["candidate", "control"]
            for arm in arms:
                workers[arm] = Worker(
                    args.python, getattr(args, arm), args.output, f"start-{start}-{arm}"
                )
                report["identities"].append(
                    {
                        "phase": "timing",
                        "start": start,
                        "arm": arm,
                        **workers[arm].identity,
                    }
                )
                expected = next(
                    i
                    for i in report.get("gate_identities", report["identities"])
                    if i["phase"] == "gate"
                    and i["arm"] == getattr(args, f"{arm}_gate_arm")
                )
                for key in ("sha256", "binding_sha256", "torch", "device"):
                    if workers[arm].identity[key] != expected[key]:
                        raise ValueError(f"identity changed after gates: {arm} {key}")
            order = list(enumerate(cases))
            if start:
                order.reverse()
            for case_id, (manifest, index) in order:
                for arm in arms:
                    workers[arm].call(
                        action="prepare", manifest=str(manifest), index=index
                    )
                windows = {arm: [] for arm in arms}
                warm_seconds = {arm: 0.0 for arm in arms}
                warm_begin = time.monotonic()
                stable = False
                while time.monotonic() - warm_begin < 10:
                    for arm in arms:
                        samples = []
                        window_begin = time.monotonic()
                        while time.monotonic() - window_begin < 0.1:
                            samples.append(
                                workers[arm].call(action="sample", condition="warm")[
                                    "ms"
                                ]
                            )
                        windows[arm].append(statistics.median(samples))
                        warm_seconds[arm] += time.monotonic() - window_begin
                    stable = all(
                        len(v) >= 3 and max(v[-3:]) / min(v[-3:]) <= 1.01
                        for v in windows.values()
                    )
                    if stable and min(warm_seconds.values()) >= 1:
                        break
                stable = stable and min(warm_seconds.values()) >= 1
                for condition in args.conditions:
                    blocks = []
                    for block in range(args.blocks):
                        rotation = arms if block % 2 == 0 else arms[::-1]
                        sequence = rotation + rotation[::-1]
                        samples = []
                        for arm in sequence:
                            sample = workers[arm].call(
                                action="sample", condition=condition
                            )
                            samples.append({"arm": arm, **sample})
                        blocks.append(samples)
                    row = {
                        "start": start,
                        "case": case_id,
                        "manifest": str(manifest),
                        "index": index,
                        "condition": condition,
                        "warmup_stable": stable,
                        "warmup_windows": windows,
                        "warmup_seconds": warm_seconds,
                        "blocks": blocks,
                        **summarize(blocks),
                    }
                    report["timings"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("blocks", "warmup_windows")
                            }
                        ),
                        flush=True,
                    )
            for worker in workers.values():
                worker.close()
            workers.clear()
        report["status"] = "measured"
    except Exception as error:
        report["status"] = "error"
        report["error"] = repr(error)
        raise
    finally:
        for worker in workers.values():
            worker.close()
        save()


if __name__ == "__main__":
    main()
