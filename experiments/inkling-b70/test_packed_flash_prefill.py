# SPDX-License-Identifier: Apache-2.0
"""Compare the bounded flash prefill with direct compact-page attention on Xe2."""

from types import SimpleNamespace
from unittest import SkipTest

import torch


def check_packed_flash_prefill(past, nt, seed):
    if not torch.xpu.is_available():
        raise SkipTest("Requires Xe2 flash attention")
    from vllm_inkling2_xpu.ops.attention import rel_decode_attention
    from vllm_inkling2_xpu.ops.kvarn_prefill import packed_flash_prefill

    gen = torch.Generator().manual_seed(seed)
    had = torch.ones(1, 1)
    for _ in range(7):
        had = torch.kron(had, torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
    had = (had / 128**0.5).to("xpu", torch.bfloat16)
    seq = past + nt
    np = (seq + 127) // 128
    storage = torch.randint(
        0, 256, (np + 1, 8, 1, 17952), dtype=torch.uint8, generator=gen
    )
    for start, base in [
        (8192, 0.23),
        (8448, -1.6),
        (8704, 0.8),
        (17152, 0.9),
        (17408, 0.25),
        (17664, -1.75),
    ]:
        scales = (base + torch.rand((np + 1, 8, 128), generator=gen) * 0.1).half()
        storage[:, :, 0, start : start + 256] = scales.view(torch.uint8)
    packed = storage.to("xpu")[..., :17920]

    def rand(*shape):
        return torch.randn(*shape, generator=gen).to("xpu", torch.bfloat16)

    stage = rand(2, 4, 128, 8, 256)
    q = rand(nt, 24, 128)
    gate = rand(nt, 24 * 128)
    sinks = rand(nt, 24) + 4
    rel = rand(nt, 24, 1024 + 256) * 0.2
    md = SimpleNamespace(
        single_prefill_seq_len=seq,
        num_actual_tokens=nt,
        max_query_len=nt,
        block_table=(torch.randperm(np, generator=gen).to("xpu", torch.int32) + 1)[
            None, :
        ],
        seq_lens=torch.tensor([seq], device="xpu", dtype=torch.int32),
        query_start_loc=torch.tensor([0, nt], device="xpu", dtype=torch.int32),
        kvarn_req_slots=torch.tensor([1], device="xpu", dtype=torch.int32),
    )
    ref = torch.empty_like(gate)
    k, v = stage.view(-1, 128, 8, 256).split(128, dim=-1)
    rel_decode_attention(
        q,
        k,
        v,
        rel,
        gate,
        sinks,
        ref,
        block_table=md.block_table,
        cache_seqlens=md.seq_lens,
        cu_seqlens_q=md.query_start_loc,
        max_query_len=nt,
        num_actual_tokens=nt,
        softmax_scale=128**-0.5,
        is_local=False,
        num_splits=1,
        packed_cache=packed,
        hadamard=had,
        req_slots=md.kvarn_req_slots,
        ring_pages=4,
    )
    out = torch.empty_like(gate)
    assert packed_flash_prefill(
        md, stage, packed, q, rel, gate, sinks, out, 128**-0.5, 1024
    )
    diff = (out.float() - ref.float()).cpu()
    ratio = float(diff.norm() / ref.float().cpu().norm())
    print(
        f"Hybrid/direct relative L2 {ratio:.6f}, maximum absolute {diff.abs().max():.6f}."
    )
    # Both readers output BF16. Bound per-element error in relative units for
    # high-amplitude channels, and absolute units around zero.
    epsilon = torch.finfo(torch.bfloat16).eps
    assert ratio < epsilon
    torch.testing.assert_close(
        out.float().cpu(), ref.float().cpu(), rtol=2 * epsilon, atol=epsilon
    )


def test_packed_flash_prefill():
    # Relative tail only, one flash prefix, and multiple merged flash windows.
    for past, nt, seed in (
        (387, 143, 883),
        (128 * 11 + 9, 143, 3317),
        (128 * 49 + 3, 137, 1891),
    ):
        check_packed_flash_prefill(past, nt, seed)


if __name__ == "__main__":
    test_packed_flash_prefill()
