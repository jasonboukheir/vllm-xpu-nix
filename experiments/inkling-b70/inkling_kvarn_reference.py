#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU numerical experiment for old, bias-free Inkling 2 Mini target KV tiles.

Use the experimental package's Python to load its pinned plugin, and --fork to
load the reference codec from the KVarN fork. Each case compares a partitioned
read to dense relative attention on the reconstructed cache, then measures
the synthetic output error introduced by compression against the FP8 cache.
All inputs and intermediates stay on the CPU.
"""

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--fork",
    type=Path,
    required=True,
    help="Source root of the KVarN-enabled vLLM fork",
)
args = parser.parse_args()
os.environ["ONEAPI_DEVICE_SELECTOR"] = "level_zero:99"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_LOGGING_LEVEL"] = "CRITICAL"

import torch
from vllm.model_executor.layers import quantization
from vllm.v1.attention import ops
from vllm_inkling2_xpu.attention import _merge_lse

# Keep vLLM and the plugin from the trial package. Extend only the two package
# paths that hold the reference codec, which otherwise is absent from this core.
for package in (quantization, ops):
    subtree = args.fork.resolve().joinpath(*package.__name__.split("."))
    if not subtree.is_dir():
        parser.error(f"--fork is missing {subtree}")
    package.__path__.insert(0, str(subtree))
from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
    variance_normalize_batched,
)
from vllm.v1.attention.ops.kvarn_decode import (
    kvarn_dequant_tile_k,
    kvarn_dequant_tile_v,
)
from vllm.v1.attention.ops.kvarn_store import (
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v_batch_from_sinkhorn,
)

PAGE = 64
GROUP = 128
HEADS = 8
DIM = 128
REL_EXTENT = 1024


def normalized_hadamard():
    factor = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    matrix = factor
    while matrix.shape[0] < DIM:
        matrix = torch.kron(matrix, factor)
    return matrix / math.sqrt(DIM)


HADAMARD = normalized_hadamard()


def dense_attention(queries, keys, values, rel, table, length):
    """Include the 1024-step relative bias, causal mask and 3:1 query/KV heads."""
    qlen = len(queries)
    key_positions = torch.arange(length)
    query_positions = torch.arange(length - qlen, length)
    distance = query_positions[:, None] - key_positions
    rel_index = distance.clamp(0, REL_EXTENT - 1)[:, None].expand(
        -1, queries.shape[1], -1
    )
    bias = rel.gather(2, rel_index).masked_fill(distance[:, None] >= REL_EXTENT, 0)
    kv = [
        cache[table[0]].flatten(0, 1)[:length].repeat_interleave(3, dim=1)
        for cache in (keys, values)
    ]
    scores = torch.einsum("qhd,khd->qhk", queries, kv[0]) * DIM**-0.5 + bias
    scores.masked_fill_(distance[:, None] < 0, -torch.inf)
    return torch.einsum("qhk,khd->qhd", scores.softmax(-1), kv[1]), scores.logsumexp(-1)


def round_trip(keys, values, cfg):
    # One record per KV head. Input K,V shape (128 tokens, 8 heads, 128 dims).
    kt = (keys.float() @ HADAMARD).permute(1, 2, 0).contiguous()
    vt = (values.float() @ HADAMARD).permute(1, 0, 2).contiguous()
    bk, kc, kr = variance_normalize_batched(kt, cfg.sinkhorn_iters)
    bv, vc, vr = variance_normalize_batched(vt, cfg.sinkhorn_iters)
    kd = kvarn_store_tile_k_batch_from_sinkhorn(
        bk, kc.squeeze(1), kr.squeeze(2), cfg.key_bits
    )
    vd = kvarn_store_tile_v_batch_from_sinkhorn(
        bv, vc.squeeze(1), vr.squeeze(2), cfg.value_bits
    )
    record = torch.empty((HEADS, cfg.record_bytes), dtype=torch.uint8)
    fields = [
        (cfg.k_packed_offset, kd, "q_packed_uint8"),
        (cfg.k_s_col_offset, kd, "s_col_K"),
        (cfg.k_zp_offset, kd, "zp_K"),
        (cfg.k_s_row_offset, kd, "s_row_K"),
        (cfg.v_packed_offset, vd, "q_packed_uint8"),
        (cfg.v_s_col_offset, vd, "s_col_V"),
        (cfg.v_s_row_offset, vd, "s_row_V"),
        (cfg.v_zp_offset, vd, "zp_V"),
    ]
    end = 0
    for offset, fields_dict, name in fields:
        value = fields_dict[name].contiguous()
        assert offset == end
        buf = value.view(torch.uint8).reshape(HEADS, -1)
        end = offset + buf.shape[-1]
        record[:, offset:end].copy_(buf)
        fields_dict[name] = record[:, offset:end].view(value.dtype).reshape(value.shape)
    assert end == cfg.record_bytes
    keys_out = (
        torch.stack(
            [
                kvarn_dequant_tile_k(**{n: t[h] for n, t in kd.items()}, group=GROUP).T
                for h in range(HEADS)
            ],
            dim=1,
        )
        @ HADAMARD
    )
    values_out = (
        torch.stack(
            [
                kvarn_dequant_tile_v(
                    **{n: t[h] for n, t in vd.items()},
                    head_dim=DIM,
                    bits=cfg.value_bits,
                )
                for h in range(HEADS)
            ],
            dim=1,
        )
        @ HADAMARD
    )
    return keys_out, values_out


def gate_sink(out, lse, gate, sinks):
    return out * gate.sigmoid() * (lse - sinks).sigmoid().unsqueeze(-1)


def run_case(bits, qlen):
    # The 1024-token bias range is true for all q rows, including the earliest
    # proposal. Only accepted, full groups outside that range are compressed.
    rng = torch.Generator().manual_seed(211 + qlen)
    seq = 1412
    pages = math.ceil(seq / PAGE)
    n_old = max(0, seq - qlen - REL_EXTENT + 1) // GROUP * GROUP
    assert n_old == GROUP * (3 if qlen == 1 else 2)
    table = torch.randperm(pages, generator=rng).view(1, pages)
    shape = (pages, PAGE, HEADS, DIM)
    # The existing community baseline keeps E4M3 pages with unit scale.
    keys, values = [
        torch.randn(shape, generator=rng).to(torch.float8_e4m3fn).float()
        for _ in range(2)
    ]
    q = torch.randn((qlen, 24, DIM), generator=rng)
    rel = torch.randn((qlen, 24, REL_EXTENT), generator=rng) * 0.4
    gate = torch.randn(q.shape, generator=rng)
    sinks = torch.randn(q.shape[:-1], generator=rng) + math.log(seq)
    cfg = KVarNConfig.from_cache_dtype(f"kvarn_k4v{bits}_g128_compact", DIM)
    full = dense_attention(q, keys, values, rel, table, seq)
    mixed_k, mixed_v = keys.clone(), values.clone()
    for start in range(0, n_old, GROUP):
        blocks = table[0, start // PAGE : (start + GROUP) // PAGE]
        k, v = round_trip(keys[blocks].flatten(0, 1), values[blocks].flatten(0, 1), cfg)
        mixed_k[blocks], mixed_v[blocks] = (
            k.view(-1, PAGE, HEADS, DIM),
            v.view(-1, PAGE, HEADS, DIM),
        )
    # Monolithic expectation on the dequantized old prefix plus dense FP8 tail.
    expected, expected_lse = dense_attention(q, mixed_k, mixed_v, rel, table, seq)
    tail = table[:, n_old // PAGE :]
    out, lse = dense_attention(q, keys, values, rel, tail, seq - n_old)
    lse = lse.T.contiguous()
    # Independently attend each old 128-token chunk with zero relative bias:
    # every key in these chunks is visible to every query and older than 1024.
    for start in range(0, n_old, GROUP):
        blocks = table[0, start // PAGE : (start + GROUP) // PAGE]
        k = mixed_k[blocks].flatten(0, 1).repeat_interleave(3, dim=1)
        v = mixed_v[blocks].flatten(0, 1).repeat_interleave(3, dim=1)
        scores = torch.einsum("qhd,khd->hqk", q, k) * DIM**-0.5
        chunk_out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        _merge_lse(out, lse, chunk_out, scores.logsumexp(-1), qlen)
    actual = gate_sink(out, lse.T, gate, sinks)
    wanted = gate_sink(expected, expected_lse, gate, sinks)
    torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(lse.T, expected_lse, rtol=1e-6, atol=2e-6)
    original = gate_sink(*full, gate, sinks)
    return {
        "value_bits": bits,
        "query_tokens": qlen,
        "old_tokens": n_old,
        "bytes_per_record": cfg.record_bytes,
        "split_merge_max_abs": (actual - wanted).abs().max().item(),
        "synthetic_fp8_vs_kvarn_relative_l2": (
            (wanted - original).norm() / original.norm()
        ).item(),
    }


if __name__ == "__main__":
    torch.set_num_threads(2)
    inputs = (
        KVarNConfig,
        variance_normalize_batched,
        kvarn_store_tile_v_batch_from_sinkhorn,
        kvarn_dequant_tile_v,
        _merge_lse,
    )
    hashes = {
        obj.__module__: hashlib.sha256(
            Path(inspect.getfile(obj)).read_bytes()
        ).hexdigest()
        for obj in inputs
    }
    print(
        json.dumps(
            {
                "source_sha256": hashes,
                "torch": torch.__version__,
                "cases": [
                    run_case(bits, qlen) for bits in (2, 4) for qlen in (1, 8, 37)
                ],
            },
            indent=2,
        )
    )
