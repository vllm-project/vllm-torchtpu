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
"""The indexer's donated top-k row update leaves the buffer and the rows
attention reads exactly as the -1 fill plus partial write did."""

import types

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUSparseAttnIndexer,
    _topk_rows,
    _topk_rows_update_jax,
)
from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer
from vllm_torchtpu.utils import synchronize_tensors


def _reference(buf, topk_tokens, num_tokens, topk_indices):
    """The -1 fill plus partial write `forward_oot` used to do, verbatim."""
    buf[:num_tokens] = -1
    num_out = min(topk_indices.shape[0], buf.shape[0])
    buf[:num_out, :topk_tokens] = topk_indices[:num_out]
    return buf


def _inputs(rows, width, topk, num_tokens, out_rows, n=1):
    g = torch.Generator().manual_seed(7 * rows + width + num_tokens + out_rows)
    stale = torch.randint(0, 1000, (rows, width), dtype=torch.int32, generator=g)
    topks = [
        torch.randint(-1, 5000, (out_rows, topk), dtype=torch.int32, generator=g)
        for _ in range(n)
    ]
    return stale, *topks


@pytest.mark.parametrize(
    "rows,width,topk,num_tokens",
    [
        (64, 32, 32, 16),  # decode bucket, full-width top-k
        (64, 48, 32, 16),  # buffer wider than top-k
        (64, 32, 32, 64),  # the whole buffer
    ],
)
def test_topk_rows_update_matches_fill_then_write(rows, width, topk, num_tokens):
    stale, topk_indices = _inputs(rows, width, topk, num_tokens, num_tokens)
    want = _reference(stale.clone(), topk, num_tokens, topk_indices)

    got_rows = _topk_rows(stale, topk, num_tokens, topk_indices)
    got = _topk_rows_update_jax(
        jnp.asarray(stale.numpy()), jnp.asarray(got_rows.numpy())
    )

    torch.testing.assert_close(torch.tensor(np.asarray(got)), want, rtol=0, atol=0)
    # What attention slices with `[:num_tokens]`.
    torch.testing.assert_close(got_rows, want[:num_tokens], rtol=0, atol=0)


class _TwoIndexedLayers(torch.nn.Module):
    """Two indexed layers, each followed by a layer that reuses its top-k.

    Mirrors the model: the table is a plain tensor attribute shared by every
    layer, an indexed layer writes it through `_write_topk_rows`, and a
    `skip_topk` layer reads `topk_indices_buffer[:num_tokens]`.
    """

    def __init__(self, buf: torch.Tensor, topk_tokens: int):
        super().__init__()
        self.topk_indices_buffer = buf
        self.topk_tokens = topk_tokens

    def forward(self, topk_a: torch.Tensor, topk_b: torch.Tensor, num_tokens: int):
        rows_a = VllmTPUSparseAttnIndexer._write_topk_rows(self, num_tokens, topk_a)
        reuse_a = self.topk_indices_buffer[:num_tokens] + 0
        rows_b = VllmTPUSparseAttnIndexer._write_topk_rows(self, num_tokens, topk_b)
        reuse_b = self.topk_indices_buffer[:num_tokens] + 0
        return rows_a, reuse_a, rows_b, reuse_b


def _compile(layers):
    torch._dynamo.reset()
    return torch.compile(layers, backend="tpu", fullgraph=True, dynamic=False)


def _check_two_layers(stale, topk, num_tokens, topk_a, topk_b, *, compiled):
    """Run `_TwoIndexedLayers` on TPU; compare to `_reference` bit for bit."""
    want_a = _reference(stale.clone(), topk, num_tokens, topk_a)
    want_rows_a = want_a[:num_tokens].clone()
    want_b = _reference(want_a, topk, num_tokens, topk_b)
    want_rows_b = want_b[:num_tokens].clone()

    layers = _TwoIndexedLayers(stale.clone().to("tpu"), topk)
    fwd = _compile(layers) if compiled else layers
    rows_a, reuse_a, rows_b, reuse_b = fwd(
        topk_a.to("tpu"), topk_b.to("tpu"), num_tokens
    )

    eq = dict(rtol=0, atol=0)
    torch.testing.assert_close(rows_a.cpu(), want_rows_a, **eq)
    torch.testing.assert_close(reuse_a.cpu(), want_rows_a, **eq)
    torch.testing.assert_close(rows_b.cpu(), want_rows_b, **eq)
    torch.testing.assert_close(reuse_b.cpu(), want_rows_b, **eq)
    torch.testing.assert_close(layers.topk_indices_buffer.cpu(), want_b, **eq)
    return layers, fwd


@pytest.mark.tpu_test
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize(
    "rows,width,topk,num_tokens",
    [
        (64, 32, 32, 16),
        (4096, 2048, 2048, 16),  # GLM-5.2 AgentX decode: the real table
    ],
)
def test_reuse_layers_read_the_donated_rows(rows, width, topk, num_tokens, compiled):
    stale, topk_a, topk_b = _inputs(rows, width, topk, num_tokens, num_tokens, n=2)
    _check_two_layers(stale, topk, num_tokens, topk_a, topk_b, compiled=compiled)


class _FakeMtpModel:
    """The draft's `compact_topk_indices` over one table, written as vLLM's
    `DeepSeekMultiTokenPredictor` writes it."""

    def __init__(self, buf: torch.Tensor):
        self.buf = buf

    def compact_topk_indices(self, slot_ids: torch.Tensor) -> None:
        self.buf[: slot_ids.numel()] = self.buf[slot_ids]


_ROWS, _WIDTH = 64, 32  # the table: max_num_batched_tokens x index_topk


def _mtp_table(compiled):
    """A table the two-layer forward has written (donating it twice), the
    forward, a stand-in proposer holding the table, and the table's contents."""
    stale, topk_a, topk_b = _inputs(_ROWS, _WIDTH, _WIDTH, _ROWS, _ROWS, n=2)
    layers, fwd = _check_two_layers(
        stale, _WIDTH, _ROWS, topk_a, topk_b, compiled=compiled
    )
    buf = layers.topk_indices_buffer
    proposer = types.SimpleNamespace(
        _mtp_topk_buffers=[buf],
        draft_model=types.SimpleNamespace(model=_FakeMtpModel(buf)),
    )
    return buf, fwd, proposer, buf.cpu(), topk_a, topk_b


def _donated_write(fwd, table, first, second):
    """Another step-0 forward over the whole chunk: two donated row updates.
    Returns the table contents it leaves."""
    fwd(first.to("tpu"), second.to("tpu"), _ROWS)
    after = _reference(table.clone(), _WIDTH, _ROWS, first)
    return _reference(after, _WIDTH, _ROWS, second)


@pytest.mark.tpu_test
@pytest.mark.parametrize("clobber", ["next_chunk", "dp_padding_chunk"])
@pytest.mark.parametrize("num_rows", [_ROWS, 2], ids=["full_table", "sub_slice"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_mtp_restore_after_donated_write(compiled, num_rows, clobber):
    """`Eagle3Proposer` holds the table and writes it outside the donated op.
    IndexShare compacts and snapshots step 0's rows; another step-0 forward --
    the next chunk's, or an EP-DP padding chunk's under DP lockstep -- donates
    the table; `_mtp_restore_topk` then puts the rows back. torch_tpu may
    satisfy a clone of a view that covers the whole table by aliasing the
    table's buffer, so a full-table snapshot must be a buffer of its own or
    the restore reads a donated buffer. The table must stay a valid, writable
    tensor across all of it."""
    buf, fwd, proposer, table, topk_a, topk_b = _mtp_table(compiled)
    g = torch.Generator().manual_seed(num_rows)

    # Chunk 0, step 0: its requests' last-token rows to the front, snapshot.
    gather0 = torch.randperm(_ROWS, generator=g)[:num_rows]
    want0 = table[gather0]
    saved0 = Eagle3Proposer._mtp_compact_and_save(proposer, gather0.to("tpu"))
    table[:num_rows] = want0
    # Submitted before the next forward, as the compiled carry and lm-head
    # programs that run in between would.
    synchronize_tensors(saved0, wait=False)

    # The other step-0 forward donates the table.
    table = _donated_write(fwd, table, topk_b, topk_a)
    saved1 = want1 = None
    if clobber == "next_chunk":
        # Chunk 1 compacts and snapshots its own rows from the same table.
        gather1 = torch.randperm(_ROWS, generator=g)[:num_rows]
        want1 = table[gather1]
        saved1 = Eagle3Proposer._mtp_compact_and_save(proposer, gather1.to("tpu"))
    assert not torch.equal(buf[:num_rows].cpu(), want0)

    eq = dict(rtol=0, atol=0)
    # Loop step 1 is step-major: each chunk's rows go back before its forward.
    Eagle3Proposer._mtp_restore_topk(proposer, saved0)
    torch.testing.assert_close(buf[:num_rows].cpu(), want0, **eq)
    if saved1 is not None:
        Eagle3Proposer._mtp_restore_topk(proposer, saved1)
        torch.testing.assert_close(buf[:num_rows].cpu(), want1, **eq)
        # Loop step 2: chunk 0's snapshot is still whole.
        Eagle3Proposer._mtp_restore_topk(proposer, saved0)
        torch.testing.assert_close(buf[:num_rows].cpu(), want0, **eq)
    # ... and the table still accepts the next donated write.
    rows_a, reuse_a, _, _ = fwd(topk_a.to("tpu"), topk_a.to("tpu"), _ROWS)
    torch.testing.assert_close(reuse_a.cpu(), rows_a.cpu(), **eq)
