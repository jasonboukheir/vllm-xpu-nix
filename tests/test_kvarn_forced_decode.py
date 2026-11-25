# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm import SamplingParams
from vllm.exceptions import VLLMValidationError
from vllm.logprobs import FlatLogprobs

from scripts.kvarn_forced_decode import (
    ForcedTokenSequenceLogitsProcessor,
    _ForceTokenSequence,
    _logit_rows,
)


def test_force_sequence_advances_from_persistent_output_state():
    processor = _ForceTokenSequence([2, 0])

    first = processor(0, torch.tensor([1.0, 2.0, 3.0]))
    second = processor(1, torch.tensor([4.0, 5.0, 6.0]))

    torch.testing.assert_close(first, torch.tensor([-torch.inf, -torch.inf, 3.0]))
    torch.testing.assert_close(second, torch.tensor([4.0, -torch.inf, -torch.inf]))


def test_force_sequence_tracks_mrv2_slots_positions_and_recycling():
    state = SimpleNamespace(prompt_len=SimpleNamespace(np=[4, 2]), vocab_size=3)
    proc = ForcedTokenSequenceLogitsProcessor(None, state)
    assert proc.add_request(0, SamplingParams(extra_args={"forced_token_ids": [1, 2]}))
    assert proc.add_request(1, SamplingParams(extra_args={"forced_token_ids": [0]}))
    ctx = SimpleNamespace(
        idx_mapping_np=np.array([1, 0]),
        expanded_idx_mapping=torch.tensor([1, 0, 0, 0]),
        pos=torch.tensor([1, 2, 3, 4]),
    )
    logits = torch.ones((4, 3))
    actual = proc.apply(logits, ctx)
    assert torch.isfinite(actual).tolist() == [
        [True, False, False],
        [True, True, True],
        [False, True, False],
        [False, False, True],
    ]
    assert not proc.add_request(1, SamplingParams())
    actual = proc.apply(torch.ones((4, 3)), ctx)
    assert torch.all(actual[0] == 1)


def test_force_sequence_validation_allows_masking_negative_infinity(monkeypatch):
    monkeypatch.setenv("KVARN_FORCED_VALIDATE_FINITE", "1")
    processor = _ForceTokenSequence([0])

    result = processor(0, torch.tensor([1.0, -torch.inf]))

    torch.testing.assert_close(result, torch.tensor([1.0, -torch.inf]))


@pytest.mark.parametrize("invalid", [torch.nan, torch.inf])
def test_force_sequence_validation_rejects_nan_and_positive_infinity(
    monkeypatch, invalid
):
    monkeypatch.setenv("KVARN_FORCED_VALIDATE_FINITE", "1")
    processor = _ForceTokenSequence([0])

    with pytest.raises(RuntimeError, match="invalid model logits"):
        processor(0, torch.tensor([1.0, invalid]))


@pytest.mark.parametrize("value", [[], [1, -1], [1, "2"], "1,2"])
def test_force_sequence_rejects_invalid_token_ids(value):
    params = SamplingParams(extra_args={"forced_token_ids": value})
    with pytest.raises(VLLMValidationError, match="forced_token_ids"):
        ForcedTokenSequenceLogitsProcessor.validate_params(params)


def test_logit_rows_preserve_raw_flat_values_and_pad_topk():
    rows = FlatLogprobs(
        start_indices=[0, 2],
        end_indices=[2, 3],
        token_ids=[7, 3, 4],
        logprobs=[2.5, -1.0, 8.0],
        ranks=[None, None, None],
        decoded_tokens=[None, None, None],
    )

    token_ids, logits = _logit_rows(rows, expected_steps=2)

    np.testing.assert_array_equal(token_ids, [[7, 3], [4, -1]])
    np.testing.assert_allclose(logits[0], [2.5, -1.0])
    assert logits[1, 0] == 8.0
    assert np.isnan(logits[1, 1])
