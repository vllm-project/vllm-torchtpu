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
    "rows,width,topk,num_tokens,out_rows",
    [
        (64, 32, 32, 16, 16),  # decode bucket, full-width top-k
        (64, 48, 32, 16, 16),  # buffer wider than top-k
        (64, 32, 32, 64, 64),  # the whole buffer
        (64, 32, 32, 16, 8),  # kernel returned fewer rows than tokens
    ],
)
def test_topk_rows_update_matches_fill_then_write(
    rows, width, topk, num_tokens, out_rows
):
    stale, topk_indices = _inputs(rows, width, topk, num_tokens, out_rows)
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


@pytest.mark.tpu_test
def test_mtp_restore_after_donated_write():
    """`Eagle3Proposer` holds the table and writes it outside the donated op:
    IndexShare snapshots step 0's rows, a later donated write clobbers them,
    and `_mtp_restore_topk` puts them back. The table must stay a valid,
    writable tensor across all of it."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    stale, topk_a, topk_b = _inputs(64, 32, 32, 16, 16, n=2)
    layers, fwd = _check_two_layers(stale, 32, 16, topk_a, topk_b, compiled=True)
    buf = layers.topk_indices_buffer
    proposer = types.SimpleNamespace(_mtp_topk_buffers=[buf])

    # Step 0 compaction: last-token rows of two requests to the front, snapshot.
    gather = torch.tensor([3, 11], device="tpu")
    buf[: gather.shape[0]] = buf.index_select(0, gather)
    saved = [buf[: gather.shape[0]].clone()]
    want_saved = saved[0].cpu()

    # The next chunk's donated write clobbers rows [0:16].
    fwd(topk_b.to("tpu"), topk_a.to("tpu"), 16)
    assert not torch.equal(buf[:2].cpu(), want_saved)

    Eagle3Proposer._mtp_restore_topk(proposer, saved)
    torch.testing.assert_close(buf[:2].cpu(), want_saved, rtol=0, atol=0)
    # ... and the table still accepts the next donated write.
    rows_a, reuse_a, _, _ = fwd(topk_a.to("tpu"), topk_a.to("tpu"), 16)
    torch.testing.assert_close(reuse_a.cpu(), rows_a.cpu(), rtol=0, atol=0)
