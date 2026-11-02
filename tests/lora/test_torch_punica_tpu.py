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
"""PunicaWrapperTPU: the id-to-slot mapping, and the add_* methods that the
vLLM LoRA layers call."""

import pytest
import torch
import torch.nn.functional as F
from vllm.lora.layers import LoRAMapping

from vllm_torchtpu.lora.torch_punica_tpu import PunicaWrapperTPU

VOCAB, HIDDEN, RANK = 50, 16, 4
# Adapter id 9 sits in slot 0, id 7 in slot 1, and slot 2 is free.
LORA_INDEX_TO_ID = [9, 7, None]
NUM_SLOTS = len(LORA_INDEX_TO_ID)
# One step as the runner hands it over: padded with id 0, no adapter.
TOKEN_IDS = (7, 7, 0, 9, 0, 0, 0, 0)
REQUEST_IDS = (7, 0, 9, 0, 0, 0, 0, 0)
TOKEN_SLOTS = [1, 1, -1, 0, -1, -1, -1, -1]
REQUEST_SLOTS = [1, -1, 0, -1, -1, -1, -1, -1]


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def _wrapper(token_ids=TOKEN_IDS, request_ids=REQUEST_IDS, capacity=64):
    wrapper = PunicaWrapperTPU(capacity, 8, "cpu")
    wrapper.update_metadata(
        LoRAMapping(token_ids, request_ids), LORA_INDEX_TO_ID, NUM_SLOTS + 1, VOCAB
    )
    return wrapper


def _stacked(out_features, in_features):
    """Stacked LoRA weights in vLLM's layout [slots, 1, out, in]."""
    return torch.randn(NUM_SLOTS, 1, out_features, in_features)


def _per_token(inputs, stacked, slots):
    """Each row times its own slot's weights; rows without a slot get zeros."""
    rows = [
        stacked[slot, 0] @ row if slot >= 0 else torch.zeros(stacked.shape[2])
        for row, slot in zip(inputs, slots)
    ]
    return torch.stack(rows)


def test_update_metadata_translates_ids_to_slots():
    wrapper = _wrapper()

    assert wrapper._token_lora_indices.dtype == torch.int32
    assert wrapper._token_lora_indices[:8].tolist() == TOKEN_SLOTS
    assert wrapper._sampler_indices[:8].tolist() == REQUEST_SLOTS
    # Row offsets into the stacked embedding LoRA A; tokens without an
    # adapter read slot 0.
    assert wrapper._embeddings_indices[:8].tolist() == [VOCAB, VOCAB] + [0] * 6
    # PunicaWrapperBase's properties return only this step's entries.
    assert wrapper.token_lora_indices.tolist() == TOKEN_SLOTS
    assert wrapper.sampler_indices.tolist() == REQUEST_SLOTS
    assert wrapper.indices_len[3] == len(TOKEN_IDS)


@pytest.mark.parametrize("token_ids,request_ids", [((5,), (0,)), ((0,), (5,))])
def test_unloaded_adapter_id_raises(token_ids, request_ids):
    """An id without a slot fails instead of silently running the base model."""
    with pytest.raises(KeyError):
        _wrapper(token_ids, request_ids)


def test_mapping_longer_than_the_buffers_is_rejected():
    with pytest.raises(AssertionError, match="largest compile size"):
        _wrapper(token_ids=(0,) * 65, request_ids=(0,), capacity=64)


def test_add_shrink_returns_a_new_buffer_in_the_input_dtype():
    """The fp32 buffer that vLLM's layers pass in is left alone; the result
    stays in the activation dtype for the expand."""
    x = torch.randn(8, HIDDEN).bfloat16()
    lora_a = tuple(_stacked(RANK, HIDDEN).bfloat16() for _ in range(3))
    buffer = torch.zeros(3, 8, RANK)

    shrunk = _wrapper().add_shrink(buffer, x, lora_a, 1.0)

    assert shrunk.shape == (3, 8, RANK) and shrunk.dtype == torch.bfloat16
    assert not buffer.any()


def test_add_lora_linear_adds_each_tokens_adapter_per_slice():
    # The q, k and v slices of a merged QKV projection.
    slices = (12, 4, 4)
    x = torch.randn(8, HIDDEN)
    lora_a = tuple(_stacked(RANK, HIDDEN) for _ in slices)
    lora_b = tuple(_stacked(width, RANK) for width in slices)
    y = torch.randn(8, sum(slices))
    y_before = y.clone()

    out = _wrapper().add_lora_linear(y, x, lora_a, lora_b, 0.5, slices)

    delta = torch.cat(
        [
            _per_token(0.5 * _per_token(x, a, TOKEN_SLOTS), b, TOKEN_SLOTS)
            for a, b in zip(lora_a, lora_b)
        ],
        dim=-1,
    )
    torch.testing.assert_close(out, y + delta)
    no_adapter = torch.tensor(TOKEN_SLOTS) < 0
    assert torch.equal(out[no_adapter], y[no_adapter])
    assert torch.equal(y, y_before)


@pytest.mark.parametrize("add_inputs", [True, False])
def test_add_expand_places_the_delta_at_offset_start(add_inputs):
    shrunk = torch.randn(1, 8, RANK)
    lora_b = _stacked(12, RANK)
    y = torch.randn(8, 20)

    out = _wrapper().add_expand(
        y, shrunk, (lora_b,), (12,), offset_start=5, add_inputs=add_inputs
    )

    delta = torch.zeros(8, 20)
    delta[:, 5:17] = _per_token(shrunk[0], lora_b, TOKEN_SLOTS)
    torch.testing.assert_close(out, y + delta if add_inputs else delta)


@pytest.mark.parametrize("offset_start", [-1, 9])
def test_add_expand_rejects_a_delta_outside_y(offset_start):
    """The 12 delta columns must fit in y's 20 columns; a padding of the wrong
    sign would crop them silently."""
    shrunk = torch.randn(1, 8, RANK)
    lora_b = _stacked(12, RANK)
    y = torch.randn(8, 20)

    with pytest.raises(AssertionError, match="does not fit"):
        _wrapper().add_expand(y, shrunk, (lora_b,), (12,), offset_start=offset_start)


@pytest.mark.parametrize(
    "flag,adds_y",
    [
        ({}, True),
        ({"add_inputs": False}, False),
        ({"add_input": False}, False),
        ({"add_input": True}, True),
    ],
)
@pytest.mark.parametrize("method", ["add_expand", "add_lora_embedding"])
def test_add_input_and_add_inputs_set_the_same_flag(method, flag, adds_y):
    """vLLM's layers pass add_input while the base class names it add_inputs;
    either one decides whether y is added."""
    y = torch.randn(8, 12)
    lora_b = _stacked(12, RANK)
    if method == "add_expand":
        args = (y, torch.randn(1, 8, RANK), (lora_b,), (12,))
    else:
        args = (y, torch.randn(8, RANK), lora_b)
    run = getattr(_wrapper(), method)

    out = run(*args, **flag)

    delta = run(*args, add_inputs=False)
    torch.testing.assert_close(out, y + delta if adds_y else delta)


@pytest.mark.parametrize("method", ["add_expand", "add_lora_embedding"])
def test_add_input_and_add_inputs_together_are_rejected(method):
    y = torch.randn(8, 12)
    lora_b = _stacked(12, RANK)
    if method == "add_expand":
        args = (y, torch.randn(1, 8, RANK), (lora_b,), (12,))
    else:
        args = (y, torch.randn(8, RANK), lora_b)

    with pytest.raises(AssertionError, match="got both"):
        getattr(_wrapper(), method)(*args, add_inputs=True, add_input=True)


def test_add_lora_embedding_expands_the_layers_lookup():
    """Follows VocabParallelEmbeddingWithLoRA.forward: look the tokens up in
    the stacked LoRA A at the wrapper's offsets, then expand."""
    wrapper = _wrapper()
    lora_a = torch.randn(NUM_SLOTS, VOCAB, RANK)
    lora_b = _stacked(HIDDEN, RANK)
    token_ids = torch.randint(0, VOCAB, (8,), dtype=torch.int32)
    y = torch.randn(8, HIDDEN)

    lookup = F.embedding(
        token_ids + wrapper._embeddings_indices[:8], lora_a.view(-1, RANK)
    )
    out = wrapper.add_lora_embedding(y, lookup, lora_b, add_input=True)

    delta = _per_token(lora_a[TOKEN_SLOTS, token_ids], lora_b, TOKEN_SLOTS)
    torch.testing.assert_close(out, y + delta)


def test_add_lora_logits_uses_each_requests_adapter():
    """Logits rows are requests, so the slots come from the request mapping,
    which here differs from the token mapping. Vocab padding gets no delta."""
    wrapper = _wrapper(token_ids=(0,) * 8)
    hidden_states = torch.randn(8, HIDDEN)
    lora_a = _stacked(RANK, HIDDEN)
    lora_b = _stacked(VOCAB, RANK)
    logits = torch.randn(8, VOCAB + 14)

    out = wrapper.add_lora_logits(logits, hidden_states, lora_a, lora_b, 1.0)

    shrunk = _per_token(hidden_states, lora_a, REQUEST_SLOTS)
    delta = _per_token(shrunk, lora_b, REQUEST_SLOTS)
    torch.testing.assert_close(out[:, :VOCAB], logits[:, :VOCAB] + delta)
    assert torch.equal(out[:, VOCAB:], logits[:, VOCAB:])
