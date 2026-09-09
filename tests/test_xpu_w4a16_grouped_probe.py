"""The bounded adapter must preserve each integer/group and fail closed."""

import json
from pathlib import Path

import pytest
import torch

from scripts.xpu_w4a16_grouped_probe import (
    adapt,
    difference,
    validate_plan,
    verify_adapter,
)


def operands():
    n, k = 16, 256
    logical = (torch.arange(n * k).reshape(n, k) * 7 + torch.arange(n)[:, None]) % 16
    words = (logical.reshape(n, k // 8, 8) << (torch.arange(8) * 4)).sum(-1)
    scales = (torch.arange(2 * n).reshape(2, n).float() + 1).div(128).bfloat16()
    scales[0] *= -1
    scales[1, 0] = 0
    return {
        "q_nt": words.int().t(),
        "scales": scales,
        "zero": torch.tensor([8], dtype=torch.int8),
        "group_size": 128,
        "g_idx": None,
    }, logical


def test_every_nibble_and_scale_group_is_preserved():
    source, logical = operands()
    packed, scales = adapt(**source)
    # Read bytes independently of the adapter, including sign extension.
    raw = packed[0].to(torch.int16)
    low, high = raw & 15, (raw >> 4) & 15
    signed = torch.stack(
        (torch.where(low >= 8, low - 16, low), torch.where(high >= 8, high - 16, high)),
        -1,
    ).flatten(1)
    assert torch.equal(signed, logical - 8)
    assert torch.equal(scales[0], source["scales"].t())
    assert packed.dtype == torch.int8 and packed.is_contiguous()
    assert scales.dtype == torch.bfloat16 and scales.is_contiguous()
    assert verify_adapter(source["q_nt"], source["scales"], packed, scales) == {
        "logical_weights_checked": logical.numel(),
        "integer_exact": True,
        "dequantized_exact": True,
    }


@pytest.mark.parametrize(
    "change",
    [
        "zero",
        "packed_zero",
        "permutation",
        "group_size",
        "nt_stride",
        "scale_dtype",
        "scale_stride",
        "nan_scale",
        "weight_dtype",
    ],
)
def test_incompatible_loaded_formats_are_rejected(change):
    source, _ = operands()
    if change == "zero":
        source["zero"].fill_(7)
    elif change == "packed_zero":
        source["zero"] = torch.full((2, 2), -2004318072, dtype=torch.int32)
    elif change == "permutation":
        source["g_idx"] = torch.arange(256, dtype=torch.int32)
    elif change == "group_size":
        source["group_size"] = 64
    elif change == "nt_stride":
        source["q_nt"] = source["q_nt"].contiguous()
    elif change == "scale_dtype":
        source["scales"] = source["scales"].float()
    elif change == "scale_stride":
        source["scales"] = source["scales"].t().contiguous().t()
    elif change == "nan_scale":
        source["scales"][0, 0] = float("nan")
    elif change == "weight_dtype":
        source["q_nt"] = source["q_nt"].long()
    with pytest.raises(ValueError):
        adapt(**source)


def test_exhaustive_check_detects_packed_or_group_corruption():
    source, _ = operands()
    packed, scales = adapt(**source)
    packed[0, -1, -1] ^= 1
    with pytest.raises(ValueError, match="integer weights"):
        verify_adapter(source["q_nt"], source["scales"], packed, scales)
    packed, scales = adapt(**source)
    scales[0, -1, -1] *= 2
    with pytest.raises(ValueError, match="dequantized weights"):
        verify_adapter(source["q_nt"], source["scales"], packed, scales)


def test_nonfinite_output_fails_even_when_nan_comparison_is_false():
    result = difference(torch.tensor([float("nan")]), torch.zeros(1), 0.01, 0.01)
    assert not result["pass"]
    assert not result["finite"]


def test_retained_plan_passes_but_changed_numerical_gate_is_rejected():
    plan = json.loads(
        (Path(__file__).parents[1] / "fixtures/xpu-w4a16-grouped-plan.json").read_text()
    )
    validate_plan(plan)
    plan["gates"]["native_reference"]["rtol"] = 0.02
    with pytest.raises(ValueError, match="threshold"):
        validate_plan(plan)
