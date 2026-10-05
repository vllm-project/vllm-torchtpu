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
"""`TPU_INDEXER_TOPK_SINGLE_WRITE` leaves the buffer and the rows attention
reads exactly as the -1 fill plus partial write did."""

import types

import pytest
import torch

from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
    VllmTPUSparseAttnIndexer,
)


def _reference(buf, topk_tokens, num_tokens, topk_indices):
    """The default path of `forward_oot`, verbatim."""
    buf[:num_tokens] = -1
    num_out = min(topk_indices.shape[0], buf.shape[0])
    buf[:num_out, :topk_tokens] = topk_indices[:num_out]
    return buf


@pytest.mark.parametrize(
    "rows,width,topk,num_tokens,out_rows",
    [
        (64, 32, 32, 16, 16),  # decode bucket, full-width top-k
        (64, 48, 32, 16, 16),  # buffer wider than top-k
        (64, 32, 32, 64, 64),  # the whole buffer
        (64, 32, 32, 16, 8),  # kernel returned fewer rows than tokens
    ],
)
def test_single_write_matches_fill_then_write(rows, width, topk, num_tokens, out_rows):
    g = torch.Generator().manual_seed(rows + width + num_tokens + out_rows)
    stale = torch.randint(0, 1000, (rows, width), dtype=torch.int32, generator=g)
    topk_indices = torch.randint(
        -1, 5000, (out_rows, topk), dtype=torch.int32, generator=g
    )

    want_buf = _reference(stale.clone(), topk, num_tokens, topk_indices)
    want_rows = want_buf[:num_tokens]

    layer = types.SimpleNamespace(topk_indices_buffer=stale.clone(), topk_tokens=topk)
    got = VllmTPUSparseAttnIndexer._write_topk_rows(layer, num_tokens, topk_indices)

    torch.testing.assert_close(layer.topk_indices_buffer, want_buf, rtol=0, atol=0)
    # What attention slices with `[:num_tokens]`.
    torch.testing.assert_close(got[:num_tokens], want_rows, rtol=0, atol=0)


# --- TPU_INDEXER_TOPK_NOCOPY: the same rows, written by a donating op ---------
#
# The flag only changes *how* the row update reaches XLA (a donated jax op plus
# a `copy_` writeback instead of a view write), so everything below checks the
# contents against `_reference` bit for bit, on TPU, eager and compiled.

from unittest.mock import patch  # noqa: E402

import jax  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op  # noqa: E402


def _has_tpu() -> bool:
    try:
        torch.empty(1, device="tpu")
    except Exception:  # noqa: BLE001 - any failure means no TPU here
        return False
    return True


requires_tpu = pytest.mark.skipif(not _has_tpu(), reason="needs a TPU device")


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


def _check_two_layers(stale, topk, num_tokens, topk_a, topk_b, *, compiled):
    """Run `_TwoIndexedLayers` on TPU with NOCOPY on; compare to `_reference`."""
    want_a = _reference(stale.clone(), topk, num_tokens, topk_a)
    want_rows_a = want_a[:num_tokens].clone()
    want_b = _reference(want_a, topk, num_tokens, topk_b)
    want_rows_b = want_b[:num_tokens].clone()

    layers = _TwoIndexedLayers(stale.clone().to("tpu"), topk)
    fwd = layers
    if compiled:
        torch._dynamo.reset()
        fwd = torch.compile(layers, backend="tpu", fullgraph=True, dynamic=False)
    with patch.object(envs, "TPU_INDEXER_TOPK_NOCOPY", True):
        rows_a, reuse_a, rows_b, reuse_b = fwd(
            topk_a.to("tpu"), topk_b.to("tpu"), num_tokens
        )

    eq = dict(rtol=0, atol=0)
    torch.testing.assert_close(rows_a.cpu()[:num_tokens], want_rows_a, **eq)
    torch.testing.assert_close(reuse_a.cpu(), want_rows_a, **eq)  # reuse layer 1
    torch.testing.assert_close(rows_b.cpu()[:num_tokens], want_rows_b, **eq)
    torch.testing.assert_close(reuse_b.cpu(), want_rows_b, **eq)  # reuse layer 2
    torch.testing.assert_close(layers.topk_indices_buffer.cpu(), want_b, **eq)
    return layers


def _inputs(rows, width, topk, num_tokens, out_rows):
    g = torch.Generator().manual_seed(7 * rows + width + num_tokens + out_rows)
    stale = torch.randint(0, 1000, (rows, width), dtype=torch.int32, generator=g)
    topk_a, topk_b = (
        torch.randint(-1, 5000, (out_rows, topk), dtype=torch.int32, generator=g)
        for _ in range(2)
    )
    return stale, topk_a, topk_b


_CASES = [
    (64, 32, 32, 16, 16),  # decode bucket, full-width top-k
    (64, 48, 32, 16, 16),  # buffer wider than top-k
    (64, 32, 32, 64, 64),  # the whole buffer
    (64, 32, 32, 16, 8),  # kernel returned fewer rows than tokens
    (4096, 2048, 2048, 16, 16),  # GLM-5.2 AgentX decode: the real table
]


@requires_tpu
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("rows,width,topk,num_tokens,out_rows", _CASES)
def test_nocopy_reuse_layers_read_identical_indices(
    rows, width, topk, num_tokens, out_rows, compiled
):
    stale, topk_a, topk_b = _inputs(rows, width, topk, num_tokens, out_rows)
    _check_two_layers(stale, topk, num_tokens, topk_a, topk_b, compiled=compiled)


@requires_tpu
def test_nocopy_mtp_index_share_restore():
    """IndexShare snapshots step 0's rows, a later write clobbers them, and
    `_mtp_restore_topk` puts them back -- on a table the donating op owns. The
    table must stay a valid, writable tensor across all of it."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    rows, width, topk, num_tokens = 64, 32, 32, 16
    stale, topk_a, topk_b = _inputs(rows, width, topk, num_tokens, num_tokens)
    layers = _check_two_layers(stale, topk, num_tokens, topk_a, topk_b, compiled=True)
    buf = layers.topk_indices_buffer
    proposer = types.SimpleNamespace(_mtp_topk_buffers=[buf])

    # Step 0 compaction: last-token rows of two requests to the front, snapshot.
    gather = torch.tensor([3, 11], device="tpu")
    buf[: gather.shape[0]] = buf.index_select(0, gather)
    saved = [buf[: gather.shape[0]].clone()]
    want_saved = saved[0].cpu()

    # The next chunk's step-0 write (NOCOPY, compiled) clobbers rows [0:16].
    torch._dynamo.reset()
    fwd = torch.compile(layers, backend="tpu", fullgraph=True, dynamic=False)
    with patch.object(envs, "TPU_INDEXER_TOPK_NOCOPY", True):
        fwd(topk_b.to("tpu"), topk_a.to("tpu"), num_tokens)
    assert not torch.equal(buf[:2].cpu(), want_saved)

    Eagle3Proposer._mtp_restore_topk(proposer, saved)
    # A skip_topk loop-step layer reads exactly the restored rows.
    torch.testing.assert_close(buf[:2].cpu(), want_saved, rtol=0, atol=0)
    # ... and the table still accepts the next donated write.
    with patch.object(envs, "TPU_INDEXER_TOPK_NOCOPY", True):
        rows_a, reuse_a, _, _ = fwd(topk_a.to("tpu"), topk_a.to("tpu"), num_tokens)
    torch.testing.assert_close(reuse_a.cpu(), rows_a.cpu()[:num_tokens], rtol=0, atol=0)


@requires_tpu
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_nocopy_positive_control_catches_a_misplaced_write(compiled):
    """An update that lands one row low must fail the same comparison."""
    from torch_tpu._internal import pallas

    def _off_by_one(buf: jax.Array, rows: jax.Array) -> jax.Array:
        return jax.lax.dynamic_update_slice(
            buf, rows.astype(buf.dtype), (1, 0)
        )

    faulty = pallas.jax_op(
        f"pallas::indexer_topk_rows_update_fault_{int(compiled)}",
        _off_by_one,
        donate_argnums=(0,),
    )
    faulty.register_fake(lambda buf, rows: torch.empty_like(buf))

    stale, topk_a, topk_b = _inputs(64, 32, 32, 16, 16)
    with (
        patch.object(mla_attention_op, "_TOPK_ROWS_UPDATE_OP", faulty),
        pytest.raises(AssertionError),
    ):
        _check_two_layers(stale, 32, 16, topk_a, topk_b, compiled=compiled)
