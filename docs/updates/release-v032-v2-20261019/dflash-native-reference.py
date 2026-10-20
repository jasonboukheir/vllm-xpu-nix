"""Exact drafter geometry: native noncausal local attention vs CPU FP32."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch
import vllm_xpu_kernels.flash_attn_interface as fa

root = Path(__file__).resolve().parent
print(json.dumps({'kernel_interface': fa.__file__, 'torch': torch.__file__}), flush=True)
sys.path.insert(0, str(root / 'vllm-xpu-kernels'))
from tests.flash_attn.test_flash_attn_varlen_func import ref_paged_attn


def reject_fallback(*args, **kwargs):
    raise AssertionError('DFlash reference test requires native XPU attention')


@torch.inference_mode()
def run_case(lengths, page_size):
    torch.manual_seed(0xDFA200 + page_size + sum(lengths))
    pages = [(n + page_size - 1) // page_size for n in lengths]
    total_pages = sum(pages)
    q = torch.randn(8 * len(lengths), 32, 128, dtype=torch.bfloat16, device='xpu')
    k = torch.randn(total_pages, page_size, 8, 128, dtype=torch.bfloat16, device='xpu')
    v = torch.randn_like(k)
    permutation = torch.randperm(total_pages, device='cpu')
    table = torch.zeros(len(lengths), max(pages), dtype=torch.int32)
    offset = 0
    for row, count in enumerate(pages):
        table[row, :count] = permutation[offset:offset + count]
        offset += count
    table = table.to('xpu')
    cu_q = torch.arange(len(lengths) + 1, dtype=torch.int32, device='xpu') * 8
    seq_k = torch.tensor(lengths, dtype=torch.int32, device='xpu')

    def native(query, key, value):
        return fa.flash_attn_varlen_func(query, key, value, 8, cu_q, max(lengths),
            seqused_k=seq_k, softmax_scale=128**-0.5, causal=False,
            block_table=table, window_size=(2047, 2047), out=torch.empty_like(query))

    expected = native(q, k, v).cpu()
    reference = ref_paged_attn(q.cpu().float(), k.cpu().float(), v.cpu().float(),
        query_lens=[8] * len(lengths), kv_lens=lengths, block_tables=table.cpu(),
        scale=128**-0.5, casual=False, is_paged=True,
        window_size_left=2047, window_size_right=2047, dtype=torch.float32)
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(expected.float(), reference, rtol=1.5e-2, atol=1.5e-2)
    decoy = (torch.randn_like(q), torch.randn_like(k), torch.randn_like(v))
    for repeat in range(8):
        native(*decoy)
        pressure = torch.empty(2 * 1024 * 1024 + repeat * 4099, dtype=torch.uint8, device='xpu')
        pressure.fill_(repeat)
        actual = native(q, k, v).cpu()
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (lengths, page_size, repeat)
        del pressure
    result = {'kv_lengths': lengths, 'query_length': 8, 'batch': len(lengths),
        'query_heads': 32, 'kv_heads': 8, 'head_dim': 128, 'page_size': page_size,
        'window': [2047, 2047], 'causal': False, 'reference': 'CPU FP32',
        'max_abs_error': (expected.float() - reference).abs().max().item(),
        'bitwise_replays': 8, 'native_only': True, 'passed': True}
    print(json.dumps(result), flush=True)
    return result


with patch.object(fa, '_fallback_varlen_attn', reject_fallback):
    lengths = [104, 2048, 2113, 4097]
    results = [run_case(case, page_size) for page_size in (64, 128)
               for case in [[n] for n in lengths] + [lengths]]
(root / 'dflash-native-reference.json').write_text(json.dumps(results, indent=2) + '\n')
