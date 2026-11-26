# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_VOCAB_BOUNDS_INT32: the patched VocabParallelEmbedding's shard bounds as
int32 buffers mask exactly the ids the int64 buffers mask, on every TP rank, for int32
and int64 token ids, including added-vocab and padding rows.

Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest --noconftest <this file>
"""

from __future__ import annotations

import pytest
import torch
import vllm.model_executor.layers.vocab_parallel_embedding as vpe

import vllm_torchtpu
from vllm_torchtpu import envs

NAMES = (
    "org_vocab_start_index",
    "org_vocab_end_index",
    "num_org_vocab_padding",
    "added_vocab_start_index",
    "added_vocab_end_index",
)

# (vocab_size, org_vocab_size, padding_size, tp_size): the Qwen3.8 vocab at TP8, a
# vocab with added tokens (two padded ranges), and a vocab not divisible by the padding.
CASES = [
    (151936, 151936, 64, 8),
    (151936 + 21, 151936, 64, 8),
    (1000 + 7, 1000, 64, 4),
    (50257, 50257, 64, 2),
]


@pytest.fixture(scope="module", autouse=True)
def _patched():
    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()


def _shards(vocab_size, org_vocab_size, padding_size, tp_size):
    org_padded = vpe.pad_vocab_size(org_vocab_size, padding_size)
    padded = vpe.pad_vocab_size(org_padded + vocab_size - org_vocab_size, padding_size)
    for rank in range(tp_size):
        yield vpe.VocabParallelEmbedding._get_indices(
            padded, org_padded, vocab_size, org_vocab_size, rank, tp_size
        )


def _ids(vocab_size, dtype):
    # Every id of the vocab, each shard boundary +-1, and out-of-vocab ids past the end.
    ids = torch.arange(vocab_size + 200, dtype=torch.int64)
    gen = torch.Generator().manual_seed(0)
    extra = torch.randint(0, vocab_size, (4096,), generator=gen)
    return torch.cat([ids, extra]).to(dtype)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_int32_bounds_mask_the_same_ids(case, id_dtype):
    ids = _ids(case[0], id_dtype)
    for si in _shards(*case):
        b64 = [torch.tensor(getattr(si, n), dtype=torch.int64) for n in NAMES]
        b32 = [torch.tensor(getattr(si, n), dtype=torch.int32) for n in NAMES]
        m64, k64 = vpe.get_masked_input_and_mask(ids, *b64)
        m32, k32 = vpe.get_masked_input_and_mask(ids, *b32)
        # The embedding reads `masked.long()`; the mask zeroes the rows of other ranks.
        assert torch.equal(m64.long(), m32.long())
        assert torch.equal(k64, k32)
        # Rows owned by this rank index the local table only.
        own = m32.long()[~k32]
        assert own.numel() == 0 or (0 <= own.min() and own.max() < si.num_elements_padded)


@pytest.mark.parametrize("case", CASES)
def test_int32_bounds_equal_the_python_int_bounds(case):
    """The int32 buffers also match the unpatched form (per-rank Python ints)."""
    ids = _ids(case[0], torch.int64)
    for si in _shards(*case):
        ref, ref_mask = vpe.get_masked_input_and_mask(ids, *(getattr(si, n) for n in NAMES))
        b32 = [torch.tensor(getattr(si, n), dtype=torch.int32) for n in NAMES]
        got, got_mask = vpe.get_masked_input_and_mask(ids, *b32)
        assert torch.equal(ref.long(), got.long())
        assert torch.equal(ref_mask, got_mask)


@pytest.mark.parametrize("flag,dtype", [(False, torch.int64), (True, torch.int32)])
def test_flag_selects_the_buffer_dtype(monkeypatch, flag, dtype):
    monkeypatch.setattr(vpe, "get_tensor_model_parallel_rank", lambda: 3)
    monkeypatch.setattr(vpe, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(envs, "VLLM_TPU_VOCAB_BOUNDS_INT32", flag)
    emb = vpe.VocabParallelEmbedding(151936 + 21, 16)
    bufs = dict(emb.named_buffers())
    assert list(bufs) == list(NAMES)  # same buffer order with the flag on or off
    for name in NAMES:
        assert bufs[name].dtype == dtype
        assert bufs[name].item() == getattr(emb.shard_indices, name)
