"""Compare the existing pure-prefill flag in one process on captured paged B1.

Uses an unchanged full attention DSO. This is a bounded operator experiment;
the CPU eligibility check is not a proposed serving-path tensor download.
"""

import argparse
import gc
import json
import random
import shutil
import statistics
import time
from pathlib import Path

from scripts.xpu_attention_prefill_trace import validate_pair
from scripts.xpu_attention_screen import summarize


def eligible(*, paged, cu_q, kv_lengths, q_tokens, max_q, table_rows):
    return (
        paged
        and max_q > 16
        and q_tokens == max_q
        and cu_q == [0, max_q]
        and table_rows == 1
        and len(kv_lengths) == 1
        and kv_lengths[0] >= max_q
    )


def main():
    import torch
    import vllm_xpu_kernels._vllm_fa2_C as binding
    from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

    from scripts.xpu_attention_worker import Replay, comparison
    from scripts.xpu_w4a16_capture import digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gates-from", type=Path)
    parser.add_argument("--start", type=int, choices=(0, 1), default=0)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output)
    for name in (
        "xpu_attention_prefill_trace.py",
        "xpu_attention_worker.py",
        "xpu_attention_screen.py",
    ):
        shutil.copy2(Path(__file__).with_name(name), args.output)
    library = args.library.resolve(strict=True)
    mapped = sorted(
        {
            line.split()[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if "libattn_kernels_xe_2.so" in line
        }
    )
    if mapped != [str(library)]:
        raise ValueError(f"wrong mapped attention DSO: {mapped}")
    identity = {
        "library": str(library),
        "sha256": digest(library),
        "binding": binding.__file__,
        "binding_sha256": digest(binding.__file__),
        "torch": torch.__version__,
        "device": torch.xpu.get_device_name(),
    }
    report = {
        "status": "running",
        "identity": identity,
        "manifest": str(args.capture.resolve()),
        "manifest_sha256": digest(args.capture),
        "start": args.start,
        "gates": [],
        "timings": [],
        "queued": [],
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    if args.gates_from:
        prior = json.loads(args.gates_from.read_text())
        if (
            prior["status"] != "gates-passed"
            or prior["identity"] != identity
            or prior["manifest_sha256"] != report["manifest_sha256"]
            or len(prior["gates"]) != 6
            or {g["index"] for g in prior["gates"]} != set(range(6))
            or not all(g["pass"] for g in prior["gates"])
            or not all(g.get("dispatch", {}).get("pass") for g in prior["gates"])
        ):
            raise ValueError("incorrect prior gate provenance")
        report["gates_from"] = str(args.gates_from.resolve())
        report["gates_sha256"] = digest(args.gates_from)

    def call(replay, arm):
        replay.kwargs["is_mix_batch"] = arm == "control"
        return replay.call()

    def sample(replay, arm):
        start = time.perf_counter_ns()
        call(replay, arm)
        submitted = time.perf_counter_ns()
        torch.xpu.synchronize()
        completed = time.perf_counter_ns()
        return {
            "arm": arm,
            "ms": (completed - start) / 1e6,
            "submit_ms": (submitted - start) / 1e6,
        }

    def warmup(replay):
        begin = time.monotonic()
        for arm in ("control", "candidate"):
            call(replay, arm)
        torch.xpu.synchronize()
        windows = []
        seconds = {"control": 0.0, "candidate": 0.0}
        stable = False
        while time.monotonic() - begin < 10:
            samples = {"control": [], "candidate": []}
            for arm, values in samples.items():
                window_start = time.monotonic()
                while time.monotonic() - window_start < 0.1:
                    call(replay, "control")
                    torch.xpu.synchronize()
                    values.append(sample(replay, arm)["ms"])
                seconds[arm] += time.monotonic() - window_start
            windows.append({a: statistics.median(v) for a, v in samples.items()})
            if (
                min(seconds.values()) >= 1
                and len(windows) >= 3
                and all(
                    max(w[a] for w in windows[-3:]) / min(w[a] for w in windows[-3:])
                    <= 1.01
                    for a in samples
                )
            ):
                stable = True
                break
        return {
            "stable": stable,
            "seconds": time.monotonic() - begin,
            "arm_seconds": seconds,
            "windows": windows,
        }

    indices = list(range(6))
    if args.start:
        indices.reverse()
    try:
        for index in indices:
            replay = Replay(args.capture, index)
            data, meta = replay.data, replay.meta
            if not eligible(
                paged=meta["paged"],
                cu_q=data["cu_seqlens_q"].tolist(),
                kv_lengths=data["seqused_k"].tolist(),
                q_tokens=replay.q.shape[0],
                max_q=meta["m"],
                table_rows=replay.kwargs["block_table"].shape[0],
            ):
                raise ValueError("capture is not confirmed packed B1 prefill")
            if not args.gates_from:
                reference = args.references / f"reference-{index}.pt"
                reference_report = json.loads(
                    (args.references / "report.json").read_text()
                )
                reference_gate = next(
                    g
                    for g in reference_report["gates"]
                    if g["arm"] == "original"
                    and g["index"] == index
                    and g["manifest"] == report["manifest"]
                )
                if (
                    not reference_gate["pass"]
                    or digest(reference) != reference_gate["reference_sha256"]
                ):
                    raise ValueError("reference does not match original passing gate")
                metadata_before = {
                    name: tensor.cpu().clone()
                    for name, tensor in replay.kwargs.items()
                    if name
                    in ("cu_seqlens_q", "cu_seqlens_k", "seqused_k", "block_table")
                    and tensor is not None
                }
                original = call(replay, "control").clone()
                replay.kwargs["is_mix_batch"] = False
                gate = replay.validate(reference, False)
                gate["index"] = index
                gate["paired_output_exact"] = torch.equal(original, replay.out)
                lse_results = {}
                for arm in ("control", "candidate"):
                    replay.kwargs.update(
                        is_mix_batch=arm == "control", return_softmax_lse=True
                    )
                    replay.out.fill_(float("nan"))
                    output, lse = flash_attn_varlen_func(**replay.kwargs)
                    if output.data_ptr() != replay.out.data_ptr():
                        raise ValueError("LSE variant violated output ownership")
                    output, lse = output.clone(), lse.clone()
                    repeat_exact = []
                    for _ in range(3):
                        replay.q.neg_()
                        flash_attn_varlen_func(**replay.kwargs)
                        replay.q.copy_(data["q"])
                        dirt = torch.full((16777216,), 0.25, device="xpu")
                        torch.xpu.synchronize()
                        del dirt
                        replay.out.fill_(float("nan"))
                        repeated_out, repeated_lse = flash_attn_varlen_func(
                            **replay.kwargs
                        )
                        repeat_exact.append(
                            torch.equal(output, repeated_out)
                            and torch.equal(lse, repeated_lse)
                        )
                    lse_results[arm] = (output, lse)
                    gate[f"lse_{arm}"] = {
                        "reference": comparison(
                            output, torch.load(reference, weights_only=True).to("xpu")
                        ),
                        "finite": bool(torch.isfinite(lse).all()),
                        "shape": list(lse.shape),
                        "stride": list(lse.stride()),
                        "dtype": str(lse.dtype),
                        "repeat_exact": repeat_exact,
                    }
                gate["lse_output_exact"] = torch.equal(
                    lse_results["control"][0], lse_results["candidate"][0]
                )
                gate["lse_exact"] = torch.equal(
                    lse_results["control"][1], lse_results["candidate"][1]
                )
                gate["lse_canaries_intact"] = bool(
                    (replay.output_storage[:256] == 7).all()
                    and (replay.output_storage[-256:] == 7).all()
                )
                pages = data["physical_pages"].to("xpu")
                gate["lse_inputs_unchanged"] = (
                    torch.equal(replay.q.cpu(), data["q"])
                    and torch.equal(
                        replay.key.index_select(0, pages).cpu(), data["key"]
                    )
                    and torch.equal(
                        replay.value.index_select(0, pages).cpu(), data["value"]
                    )
                    and all(
                        torch.equal(replay.kwargs[name].cpu(), before)
                        for name, before in metadata_before.items()
                    )
                )
                gate["pass"] &= (
                    gate["paired_output_exact"]
                    and gate["lse_output_exact"]
                    and gate["lse_exact"]
                    and gate["lse_canaries_intact"]
                    and gate["lse_inputs_unchanged"]
                )
                for arm in lse_results:
                    details = gate[f"lse_{arm}"]
                    gate["pass"] &= (
                        details["finite"]
                        and details["reference"]["pass"]
                        and all(details["repeat_exact"])
                        and details["shape"] == [24, meta["m"]]
                        and details["stride"] == [meta["m"], 1]
                        and details["dtype"] == "torch.float32"
                    )
                del lse_results
                replay.kwargs.pop("return_softmax_lse")
                report["gates"].append(gate)
                save()
                if not gate["pass"]:
                    raise ValueError(f"correctness gate failed: {index}")
                for arm in ("control", "candidate"):
                    call(replay, arm)
                    torch.xpu.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.XPU,
                        ]
                    ) as profile:
                        call(replay, arm)
                        torch.xpu.synchronize()
                    profile.export_chrome_trace(
                        str(args.output / f"trace-{index}-{arm}.json")
                    )
                gate["dispatch"] = validate_pair(
                    args.output / f"trace-{index}-control.json",
                    args.output / f"trace-{index}-candidate.json",
                )
                save()
                print(json.dumps({"gate": index, "pass": gate["pass"]}), flush=True)
            else:
                decoy = Replay(args.capture, (index + 1) % 6)
                conditioner = torch.ones((128 * 1024 * 1024,), device="xpu")
                warm = warmup(replay)
                rng = random.Random(9911 + args.start * 100 + index)
                for condition in ("warm", "cold", "interleaved"):
                    orders = [["control", "candidate"], ["candidate", "control"]] * 15
                    rng.shuffle(orders)
                    blocks = []
                    for order in orders:
                        block = []
                        for arm in order:
                            if condition == "warm":
                                call(replay, "control")
                            elif condition == "cold":
                                conditioner.add_(1)
                            else:
                                call(decoy, "control")
                            torch.xpu.synchronize()
                            block.append(sample(replay, arm))
                        blocks.append(block)
                    row = {
                        "manifest": report["manifest"],
                        "index": index,
                        "start": args.start,
                        "condition": condition,
                        "blocks": blocks,
                        "warmup": warm,
                        "warmup_stable": warm["stable"],
                        **summarize(blocks),
                    }
                    for arm in ("control", "candidate"):
                        row[f"{arm}_submit_ms"] = statistics.mean(
                            s["submit_ms"] for b in blocks for s in b if s["arm"] == arm
                        )
                    report["timings"].append(row)
                    save()
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ("blocks", "warmup", "manifest")
                            }
                        ),
                        flush=True,
                    )
                if index in (0, 5):
                    orders = [["control", "candidate"], ["candidate", "control"]] * 10
                    rng.shuffle(orders)
                    blocks = []
                    for order in orders:
                        block = []
                        for arm in order:
                            call(replay, "control")
                            torch.xpu.synchronize()
                            begin = time.perf_counter_ns()
                            for _ in range(8):
                                call(replay, arm)
                            submitted = time.perf_counter_ns()
                            torch.xpu.synchronize()
                            end = time.perf_counter_ns()
                            block.append(
                                {
                                    "arm": arm,
                                    "ms": (end - begin) / 8e6,
                                    "submit_ms": (submitted - begin) / 8e6,
                                }
                            )
                        blocks.append(block)
                    report["queued"].append(
                        {
                            "index": index,
                            "calls_per_window": 8,
                            "blocks": blocks,
                            **summarize(blocks),
                        }
                    )
                del decoy, conditioner
            del replay
            gc.collect()
            torch.xpu.empty_cache()
        report["status"] = "measured" if args.gates_from else "gates-passed"
        save()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        save()
        raise


if __name__ == "__main__":
    main()
