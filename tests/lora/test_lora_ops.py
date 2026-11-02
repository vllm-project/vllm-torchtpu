# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The one-hot LoRA shrink and expand against a per-token reference."""

import pytest
import torch

from vllm_torchtpu.lora.lora_ops import bgmv_expand, bgmv_shrink, one_hot_slots

NUM_SLOTS, RANK, HIDDEN, OUT = 3, 4, 16, 24
# Every slot, -1 for no adapter, and NUM_SLOTS, which is out of range.
IDX = torch.tensor([0, 1, 2, -1, 1, NUM_SLOTS, 0], dtype=torch.int32)
NO_SLOT = (IDX < 0) | (IDX >= NUM_SLOTS)


def _stacked(out_features, in_features, dtype):
    """Stacked LoRA weights in vLLM's layout [slots, 1, out, in]."""
    gen = torch.Generator().manual_seed(out_features)
    return torch.randn(NUM_SLOTS, 1, out_features, in_features, generator=gen).to(dtype)


def _per_token(inputs, stacked, idx):
    """Each row times its own slot's weights in fp32; rows without a slot get
    zeros."""
    rows = [
        stacked[slot, 0].float() @ row
        if 0 <= slot < NUM_SLOTS
        else torch.zeros(stacked.shape[2])
        for row, slot in zip(inputs.float(), idx.tolist())
    ]
    return torch.stack(rows)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shrink_and_expand_match_per_token_reference(dtype):
    x = torch.randn(len(IDX), HIDDEN, generator=torch.Generator().manual_seed(0))
    x = x.to(dtype)
    lora_a = _stacked(RANK, HIDDEN, dtype)
    lora_b = _stacked(OUT, RANK, dtype)
    tol = {} if dtype == torch.float32 else {"rtol": 2e-2, "atol": 2e-2}
    onehot = one_hot_slots(IDX, NUM_SLOTS)

    shrunk = bgmv_shrink(x, lora_a, onehot, 0.5)
    expanded = bgmv_expand(shrunk, lora_b, onehot)

    assert shrunk.dtype == dtype and shrunk.shape == (len(IDX), RANK)
    assert expanded.dtype == dtype and expanded.shape == (len(IDX), OUT)
    torch.testing.assert_close(shrunk.float(), 0.5 * _per_token(x, lora_a, IDX), **tol)
    torch.testing.assert_close(expanded.float(), _per_token(shrunk, lora_b, IDX), **tol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rows_without_a_slot_are_exactly_zero(dtype):
    """-1 and out-of-range slots add nothing, instead of wrapping around to
    the last slot the way a gather would."""
    x = torch.ones(len(IDX), HIDDEN, dtype=dtype)
    buf = torch.ones(len(IDX), RANK, dtype=dtype)
    onehot = one_hot_slots(IDX, NUM_SLOTS)

    shrunk = bgmv_shrink(x, _stacked(RANK, HIDDEN, dtype), onehot, 1.0)
    expanded = bgmv_expand(buf, _stacked(OUT, RANK, dtype), onehot)

    assert not shrunk[NO_SLOT].any()
    assert not expanded[NO_SLOT].any()
    assert expanded[~NO_SLOT].any()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nan_or_inf_in_one_slot_stays_out_of_other_tokens(bad):
    """A NaN or Inf in one slot's A reaches only the tokens of that slot; a
    multiply by the mask would spread it to every token, since 0 * NaN is
    NaN."""
    x = torch.randn(len(IDX), HIDDEN, generator=torch.Generator().manual_seed(0))
    lora_a = _stacked(RANK, HIDDEN, torch.float32)
    lora_a[1, 0, 0, 0] = bad

    shrunk = bgmv_shrink(x, lora_a, one_hot_slots(IDX, NUM_SLOTS), 1.0)

    in_slot = IDX == 1
    assert not shrunk[in_slot].isfinite().all()
    assert shrunk[~in_slot].isfinite().all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_rows_without_a_slot_stay_zero_with_a_nan_or_inf_input(bad):
    """The embedding layer hands expand LoRA A rows for tokens without an
    adapter too, looked up in slot 0; a NaN or Inf there must not reach
    them."""
    buf = torch.randn(len(IDX), RANK, generator=torch.Generator().manual_seed(0))
    buf[NO_SLOT] = bad

    expanded = bgmv_expand(
        buf, _stacked(OUT, RANK, torch.float32), one_hot_slots(IDX, NUM_SLOTS)
    )

    assert not expanded[NO_SLOT].any()
    assert expanded[~NO_SLOT].isfinite().all()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "fullgraph_compile"])
def test_dtype_mismatch_points_at_lora_dtype(compiled):
    """The ops also run inside the compiled backbone, where a failed plain
    assert surfaces as Dynamo Unsupported without this message."""
    lora_a = _stacked(RANK, HIDDEN, torch.bfloat16)
    lora_b = _stacked(OUT, RANK, torch.bfloat16)
    x = torch.randn(len(IDX), HIDDEN)
    buf = torch.randn(len(IDX), RANK)
    onehot = one_hot_slots(IDX, NUM_SLOTS)

    shrink, expand = bgmv_shrink, bgmv_expand
    if compiled:
        shrink = torch.compile(bgmv_shrink, backend="eager", fullgraph=True)
        expand = torch.compile(bgmv_expand, backend="eager", fullgraph=True)
    with pytest.raises(AssertionError, match="lora_dtype must match the model dtype"):
        shrink(x, lora_a, onehot, 1.0)
    with pytest.raises(AssertionError, match="lora_dtype must match the model dtype"):
        expand(buf, lora_b, onehot)
