"""Serial JSON worker for replaying captured attention with one immutable DSO.

Launch with LD_LIBRARY_PATH set before Python starts. Timing excludes IPC,
conditioning and synchronization before the call; it includes the native wrapper,
workspace allocation and completion wait. Reference gates use FP32 arithmetic.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

from scripts.xpu_w4a16_capture import digest


def fp32_reference(q, key, value, scale):
    """Independent bottom-right causal GQA, with bounded score storage."""
    m, heads, dim = q.shape
    length, kv_heads, _ = key.shape
    result = torch.empty((m, heads, dim), device=q.device, dtype=torch.float32)
    columns = torch.arange(length, device=q.device)
    group = heads // kv_heads
    for head in range(kv_heads):
        k = key[:, head].float().T.contiguous()
        v = value[:, head].float().contiguous()
        for start in range(0, m, 64):
            stop = min(m, start + 64)
            query = q[start:stop, head * group : (head + 1) * group]
            scores = torch.matmul(query.float().transpose(0, 1), k) * scale
            rows = torch.arange(start, stop, device=q.device) + length - m
            scores.masked_fill_(
                columns[None, None, :] > rows[None, :, None], -torch.inf
            )
            result[start:stop, head * group : (head + 1) * group] = torch.matmul(
                scores.softmax(-1), v
            ).transpose(0, 1)
    return result


def comparison(actual, expected):
    delta = (actual.float() - expected.float()).abs()
    limit = 0.02 + 0.01 * expected.float().abs()
    return {
        "pass": bool(torch.isfinite(actual).all() and (delta <= limit).all()),
        "max_abs": delta.max().item(),
        "rms": delta.square().mean().sqrt().item(),
        "outside_tolerance": int((delta > limit).sum()),
        "atol": 0.02,
        "rtol": 0.01,
    }


class Replay:
    def __init__(self, manifest, index):
        path = Path(manifest)
        report = json.loads(path.read_text())
        self.meta = meta = report["captures"][index]
        snapshot = path.parent / meta["file"]
        if digest(snapshot) != meta["sha256"]:
            raise ValueError("capture checksum mismatch")
        data = torch.load(snapshot, map_location="cpu", weights_only=True)
        self.data = data
        self.q = torch.empty_strided(
            meta["q"]["shape"], meta["q"]["stride"], dtype=data["q"].dtype, device="xpu"
        )
        self.q.copy_(data["q"])
        self.storages = []
        if meta["paged"]:
            shape, stride = meta["key"]["shape"], meta["key"]["stride"]
            if "key_value_shared_storage" not in meta:
                raise ValueError("recapture required: missing K/V alias metadata")
            delta = meta["key_value_offset_delta"]
            shared = meta["key_value_shared_storage"]
            if shared and delta < 0:
                raise ValueError("unsupported negative K/V storage offset")
            size = 1 + sum((n - 1) * s for n, s in zip(shape, stride))
            backing = torch.full(
                (size + (delta if shared else 0),),
                0.375,
                dtype=data["key"].dtype,
                device="xpu",
            )
            self.storages.append(backing)
            self.key = backing.as_strided(shape, stride)
            if shared:
                self.value = backing.as_strided(shape, stride, delta)
            else:
                self.value = torch.empty_strided(
                    shape,
                    meta["value"]["stride"],
                    dtype=data["value"].dtype,
                    device="xpu",
                )
                self.value.fill_(-0.375)
            pages = data["physical_pages"].to("xpu")
            self.key.index_copy_(0, pages, data["key"].to("xpu"))
            self.value.index_copy_(0, pages, data["value"].to("xpu"))
            table_meta = meta["block_table"]
            table = torch.zeros(table_meta["shape"], dtype=torch.int32, device="xpu")
            table[0, : len(pages)] = pages.to(torch.int32)
        else:
            self.key = torch.empty_strided(
                meta["key"]["shape"],
                meta["key"]["stride"],
                dtype=data["key"].dtype,
                device="xpu",
            )
            self.value = torch.empty_strided(
                meta["value"]["shape"],
                meta["value"]["stride"],
                dtype=data["value"].dtype,
                device="xpu",
            )
            self.key.copy_(data["key"])
            self.value.copy_(data["value"])
            table = None
        # A contiguous output matches the captured service layout. Canary regions
        # surround the entire owned view and survive NaN initialization of it.
        if meta["out"]["stride"] != list(self.q.stride()):
            raise ValueError("unsupported output layout")
        self.output_storage = torch.full(
            (self.q.numel() + 512,), 7.0, dtype=self.q.dtype, device="xpu"
        )
        self.out = self.output_storage[256:-256].view_as(self.q)
        self.kwargs = {
            "q": self.q,
            "k": self.key,
            "v": self.value,
            "max_seqlen_q": meta["m"],
            "max_seqlen_k": meta["k"],
            "causal": meta["causal"],
            "softmax_scale": meta["softmax_scale"],
            "window_size": meta["window_size"],
            "block_table": table,
            "out": self.out,
        }
        for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_k"):
            self.kwargs[name] = None if data[name] is None else data[name].to("xpu")

    def call(self):
        result = flash_attn_varlen_func(**self.kwargs)
        if result.data_ptr() != self.out.data_ptr():
            raise ValueError("wrapper did not honor output ownership")
        return result

    def validate(self, reference_path, make_reference):
        metadata_before = {
            name: tensor.clone()
            for name, tensor in self.kwargs.items()
            if name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_k", "block_table")
            and tensor is not None
        }
        self.out.fill_(float("nan"))
        actual = self.call().clone()
        captured = self.data["output"].to("xpu")
        report = {
            "capture_comparison": comparison(actual, captured),
            "capture_bit_exact": torch.equal(actual, captured),
        }
        if make_reference:
            key = self.data["key"].reshape(-1, 4, 256)[: self.meta["k"]].to("xpu")
            value = self.data["value"].reshape(-1, 4, 256)[: self.meta["k"]].to("xpu")
            reference = fp32_reference(
                self.q, key, value, self.meta["softmax_scale"] or 256**-0.5
            )
            torch.save(reference.cpu(), reference_path)
        else:
            reference = torch.load(reference_path, weights_only=True).to("xpu")
        report["reference"] = comparison(actual, reference)
        report["reference_sha256"] = digest(reference_path)
        report["repeat_exact"] = []
        # Change operands and dirty freshly allocated workspace-sized blocks,
        # then restore the exact input and demand exact output, not allclose.
        for repeat in range(3):
            self.q.neg_()
            self.out.fill_(float("nan"))
            self.call()
            self.q.copy_(self.data["q"])
            scratch = [
                torch.full((n,), 0.125 * (repeat + 1), device="xpu")
                for n in (4096, 65536, 1048576, 16777216)
            ]
            torch.xpu.synchronize()
            del scratch
            self.out.fill_(float("nan"))
            report["repeat_exact"].append(torch.equal(self.call(), actual))
        report["canaries_intact"] = bool(
            (self.output_storage[:256] == 7).all()
            and (self.output_storage[-256:] == 7).all()
        )
        report["inputs_unchanged"] = torch.equal(self.q.cpu(), self.data["q"])
        if self.meta["paged"]:
            pages = self.data["physical_pages"].to("xpu")
            key, value = (
                self.key.index_select(0, pages),
                self.value.index_select(0, pages),
            )
        else:
            key, value = self.key, self.value
        report["inputs_unchanged"] &= torch.equal(
            key.cpu(), self.data["key"]
        ) and torch.equal(value.cpu(), self.data["value"])
        report["metadata_unchanged"] = all(
            torch.equal(self.kwargs[name], before)
            for name, before in metadata_before.items()
        )
        report["pass"] = (
            report["reference"]["pass"]
            and report["capture_comparison"]["pass"]
            and all(report["repeat_exact"])
            and report["canaries_intact"]
            and report["inputs_unchanged"]
            and report["metadata_unchanged"]
        )
        torch.xpu.synchronize()
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    library = args.library.resolve(strict=True)
    mappings = Path("/proc/self/maps").read_text()
    mapped = sorted(
        {
            line.split()[-1]
            for line in mappings.splitlines()
            if "libattn_kernels_xe_2.so" in line
        }
    )
    if mapped != [str(library)]:
        raise ValueError(f"wrong mapped attention library: {mapped}")
    import vllm_xpu_kernels._vllm_fa2_C as binding

    def emit(value):
        print(json.dumps(value), flush=True)

    emit(
        {
            "ready": True,
            "library": str(library),
            "sha256": digest(library),
            "binding": binding.__file__,
            "binding_sha256": digest(binding.__file__),
            "torch": torch.__version__,
            "device": str(torch.xpu.get_device_properties(0)),
        }
    )
    current = decoy = conditioner = None
    for line in sys.stdin:
        command = json.loads(line)
        action = command["action"]
        try:
            if action == "prepare":
                current = decoy = None
                gc.collect()
                torch.xpu.empty_cache()
                current = Replay(command["manifest"], command["index"])
                # Separate real captured operands, even when the shape matches.
                decoy = Replay(command["manifest"], (command["index"] + 1) % 6)
                conditioner = torch.zeros(512 * 1024 * 1024 // 4, device="xpu")
                torch.xpu.synchronize()
                response = {"prepared": current.meta}
            elif action == "validate":
                response = current.validate(
                    command["reference"], command["make_reference"]
                )
            elif action == "trace":
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.XPU,
                    ]
                ) as prof:
                    current.call()
                    torch.xpu.synchronize()
                prof.export_chrome_trace(command["path"])
                response = {"trace": command["path"]}
            elif action == "sample":
                condition = command["condition"]
                if condition == "warm":
                    current.call()
                elif condition == "cold":
                    conditioner.add_(1)
                elif condition == "interleaved":
                    decoy.call()
                else:
                    raise ValueError("unknown cache condition")
                torch.xpu.synchronize()
                start = time.perf_counter_ns()
                current.call()
                torch.xpu.synchronize()
                response = {"ms": (time.perf_counter_ns() - start) / 1e6}
            elif action == "quit":
                emit({"quit": True})
                break
            else:
                raise ValueError("unknown action")
            emit(response)
        except Exception as error:
            emit({"error": repr(error)})
            raise


if __name__ == "__main__":
    main()
