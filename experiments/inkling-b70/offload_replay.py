"""Exercise RAM prefix replay on an isolated loopback Inkling test engine."""

import argparse
import hashlib
import http.client
import json
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SERIES = {
    "store_bytes": ("kv_offload_store_bytes_total", None),
    "load_bytes": ("kv_offload_load_bytes_total", None),
    "external_hit_tokens": ("external_prefix_cache_hits_total", None),
    "gpu_hit_tokens": ("prefix_cache_hits_total", None),
    "host_hit_tokens": ("prompt_tokens_cached_by_source_total", "host"),
    "draft": ("spec_decode_num_draft_tokens_total", None),
    "accepted": ("spec_decode_num_accepted_tokens_total", None),
    "running": ("num_requests_running", None),
    "waiting": ("num_requests_waiting", None),
    "kv_usage": ("kv_cache_usage_perc", None),
}
CODE = "ORCHID482"
BUDGET = 8 * 1024**3


def parse_metrics(body):
    values = dict.fromkeys(SERIES)
    configs = []
    for line in body.splitlines():
        m = re.match(r"^(vllm:[^\s{]+)(?:\{(.*?)\})?\s+([0-9.eE+-]+)(?:\s|$)", line)
        if not m:
            continue
        metric, labels, value = m.groups()
        if metric == "vllm:kv_offload_config_info":
            configs.append(dict(re.findall(r'(\w+)="([^"]*)"', labels or "")))
        for name, (series, source) in SERIES.items():
            if metric == "vllm:" + series and (
                source is None or f'source="{source}"' in (labels or "")
            ):
                values[name] = (values[name] or 0) + float(value)
    return values, configs


def diff(after, before, key):
    return None if after[key] is None else after[key] - (before[key] or 0)


class Trial:
    def __init__(self, args):
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(__file__, self.output / "client-source.py")
        self.doc = {
            "status": "running",
            "model": args.model,
            "port": args.port,
            "page_size": 128,
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "checks": {},
            "cases": {},
        }

    def save(self):
        target = self.output / "summary.json"
        tmp = target.with_suffix(".partial")
        tmp.write_text(json.dumps(self.doc, indent=2) + "\n")
        tmp.replace(target)

    def check(self, name, value):
        self.doc["checks"][name] = bool(value)
        self.save()

    def request(self, route, payload=None, text=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.args.port, timeout=240)
        try:
            data = json.dumps(payload).encode() if payload is not None else None
            conn.request(
                "POST" if data is not None else "GET",
                route,
                data,
                {"Content-Type": "application/json"},
            )
            res = conn.getresponse()
            body = res.read().decode()
            if res.status != 200:
                raise ValueError(f"{route} got HTTP {res.status}: {body[:1000]}")
            return body if text else json.loads(body)
        finally:
            conn.close()

    def stats(self, label=None):
        body = self.request("/metrics", text=True)
        if label:
            (self.output / (label + ".prom")).write_text(body)
        vals, configs = parse_metrics(body)
        if configs:
            self.doc["offload_configuration"] = configs
        return vals

    def reset(self, external):
        deadline = time.monotonic() + 25
        while True:
            reset = self.request(
                f"/reset_prefix_cache?reset_external={str(external).lower()}", {}
            )
            if reset.get("success") is True:
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Could not reset isolated engine cache: {reset}")
            time.sleep(0.25)

    def fixture(self, n, code=CODE):
        pad = n - 80
        for _ in range(10):
            text = (
                f"The station identifier in this shipping record is {code}. The following are bulk filler notes.\n"
                + " neutral" * pad
                + "\nIdentify the station. Finish the answer with its exact identifier and nothing else."
            )
            messages = [
                {"role": "user", "content": text},
                {"role": "assistant", "content": "The station identifier is"},
            ]
            result = self.request(
                "/tokenize",
                {
                    "model": self.args.model,
                    "messages": messages,
                    "chat_template_kwargs": {"reasoning_effort": "none"},
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                },
            )
            tokens = result["tokens"]
            if len(tokens) == n:
                return {"code": code, "messages": messages, "prompt_ids": tokens}
            pad += n - len(tokens)
        raise ValueError(f"Could not produce {n} native prompt tokens")

    def complete(self, f):
        start = time.monotonic()
        response = self.request(
            "/v1/completions",
            {
                "model": self.args.model,
                "prompt": f["prompt_ids"],
                "temperature": 0,
                "seed": 42,
                "max_tokens": 32,
                "return_token_ids": True,
            },
        )
        c = response["choices"][0]
        return {
            "seconds": time.monotonic() - start,
            "text": c.get("text"),
            "ids": c.get("token_ids"),
            "usage": response["usage"],
            "actual_prompt_ids_equal": c.get(
                "prompt_token_ids", response.get("prompt_token_ids")
            )
            == f["prompt_ids"],
            "finish_reason": c["finish_reason"],
        }

    def evidence(self, label, f, r):
        self.check(
            label + "-answer",
            (r.get("text") or "").strip().removesuffix(".") == f["code"],
        )
        self.check(label + "-natural-stop", r["finish_reason"] == "stop")
        self.check(label + "-output-ids", bool(r["ids"]))
        self.check(
            label + "-exact-input",
            r["actual_prompt_ids_equal"]
            and r["usage"]["prompt_tokens"] == len(f["prompt_ids"]),
        )

    def case(self, label, f, mode):
        before = self.stats(label + ".before")
        r = self.complete(f)
        deadline = time.monotonic() + 14
        while True:
            after = self.stats()
            needs = (
                ["store_bytes"]
                if mode == "cold"
                else ["load_bytes", "host_hit_tokens", "external_hit_tokens"]
            )
            if (
                all((diff(after, before, k) or 0) > 0 for k in needs)
                and after["running"] == after["waiting"] == 0
            ) or time.monotonic() > deadline:
                break
            time.sleep(0.2)
        after = self.stats(label + ".after")
        deltas = {
            k: diff(after, before, k)
            for k in SERIES
            if k not in ("running", "waiting", "kv_usage")
        }
        self.doc["cases"][label] = {
            "mode": mode,
            "result": r,
            "before": before,
            "after": after,
            "delta": deltas,
        }
        self.evidence(label, f, r)
        self.check(label + "-idle", after["running"] == after["waiting"] == 0)
        self.check(label + "-no-gpu-hit", deltas["gpu_hit_tokens"] == 0)
        if mode == "cold":
            self.check(label + "-wrote-to-host", (deltas["store_bytes"] or 0) > 0)
            self.check(label + "-cold", deltas["external_hit_tokens"] == 0)
        else:
            for k in needs:
                self.check(label + "-" + k, (deltas[k] or 0) > 0)
        print(label, r["text"], r["finish_reason"], deltas, flush=True)
        return r

    def same(self, label, a, b):
        self.check(
            label, bool(a["ids"]) and a["ids"] == b["ids"] and a["text"] == b["text"]
        )

    def wave(self):
        fixtures = [
            self.fixture(769, c)
            for c in ("MAPLE173", "FERN651", "PEACH972", "CEDAR835")
        ]
        cold = [
            self.case(f"client-{i}-seed", f, "cold") for i, f in enumerate(fixtures)
        ]
        self.reset(False)
        before = self.stats("four.before")
        barrier = threading.Barrier(4)

        def run(f):
            barrier.wait(timeout=10)
            start = time.monotonic()
            row = self.complete(f)
            return [start, time.monotonic(), row]

        with ThreadPoolExecutor(4) as pool:
            rows = list(pool.map(run, fixtures))
        self.doc["four_clients"] = {
            "responses": rows,
            "wall_seconds": max(r[1] for r in rows) - min(r[0] for r in rows),
        }
        self.check(
            "four-http-requests-overlapped",
            max(r[0] for r in rows) < min(r[1] for r in rows),
        )
        for i, (f, seed, (_, _, row)) in enumerate(
            zip(fixtures, cold, rows, strict=True)
        ):
            self.evidence(f"client-{i}-warm", f, row)
            self.same(f"client-{i}-deterministic", seed, row)
        deadline = time.monotonic() + 14
        while time.monotonic() < deadline:
            current = self.stats()
            if (diff(current, before, "host_hit_tokens") or 0) > 0 and current[
                "running"
            ] == current["waiting"] == 0:
                break
            time.sleep(0.2)
        after = self.stats("four.after")
        self.doc["four_clients"]["deltas"] = {k: diff(after, before, k) for k in SERIES}
        self.check(
            "four-host-load",
            (diff(after, before, "load_bytes") or 0) > 0
            and (diff(after, before, "host_hit_tokens") or 0) > 0,
        )

    def run(self):
        models = self.request("/v1/models")["data"]
        if (
            [m["id"] for m in models] != [self.args.model]
            or "inkling" not in self.args.model.lower()
            or "trial" not in self.args.model.lower()
        ):
            raise ValueError(f"Dedicated Inkling trial identity mismatch: {models}")
        self.doc["models"] = models
        start = self.stats("before")
        self.reset(True)
        aligned = self.fixture(1024)
        short = self.fixture(1023, "JUNIPER739")
        growing = self.fixture(1537)
        fixtures = [aligned, short, growing]
        (self.output / "fixtures.json").write_text(json.dumps(fixtures) + "\n")
        common = next(
            (
                i
                for i, (a, b) in enumerate(
                    zip(aligned["prompt_ids"], growing["prompt_ids"])
                )
                if a != b
            ),
            1024,
        )
        self.doc["growing_common_prefix_tokens"] = common
        self.check("growth-common-prefix", common >= 896)
        cold = self.case("aligned-seed", aligned, "cold")
        self.same(
            "active-only-idle-identical",
            cold,
            self.case("aligned-idle-replay", aligned, "warm"),
        )
        # Reuse the sole staging slot with different data before the device-only reset.
        other = self.fixture(1024, "WILLOW286")
        self.case("overwrite-staging-slot", other, "cold")
        self.reset(False)
        self.same(
            "device-reset-identical",
            cold,
            self.case("aligned-device-reset", aligned, "warm"),
        )
        self.check(
            "aligned-resume-at-last-token",
            self.doc["cases"]["aligned-device-reset"]["delta"]["host_hit_tokens"]
            == len(aligned["prompt_ids"]) - 1,
        )
        grow = self.case("growing-prefix", growing, "warm")
        self.same(
            "growing-identical", grow, self.case("growing-replay", growing, "warm")
        )
        self.reset(True)
        self.same(
            "growing-cold-reference-identical",
            grow,
            self.case("growing-cold-reference", growing, "cold"),
        )
        self.reset(True)
        cold = self.case("unaligned-seed", short, "cold")
        self.same(
            "unaligned-identical", cold, self.case("unaligned-replay", short, "warm")
        )
        if self.args.four:
            self.reset(True)
            self.wave()
        end = self.stats("after")
        self.check(
            "DSpark-acceptance",
            (diff(end, start, "draft") or 0) > 0
            and (diff(end, start, "accepted") or 0) > 0,
        )
        configs = self.doc.get("offload_configuration", [])
        capacities = [
            int(c["cpu_kv_cache_size_bytes"])
            for c in configs
            if c.get("cpu_kv_cache_size_bytes")
        ]
        self.check(
            "host-pool-within-one-percent-of-eight-GiB",
            len(capacities) == 1 and 0.99 * BUDGET <= capacities[0] <= BUDGET,
        )
        self.doc["status"] = "passed" if all(self.doc["checks"].values()) else "failed"
        self.save()
        return self.doc["status"] == "passed"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=18002)
    ap.add_argument("--model", default="inkling-offload-trial")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--four", action="store_true")
    trial = Trial(ap.parse_args())
    try:
        ok = trial.run()
    except Exception as exc:
        trial.doc.update(status="error", error=repr(exc))
        trial.save()
        raise
    raise SystemExit(0 if ok else 1)
