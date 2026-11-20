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
"""SP FFN row pieces (TPU_MINIMAX_M3_SP_FFN_PIECES) on CPU: the per-piece
gather / FFN / reduce-scatter equals the whole-shard one, and the MoE's
row-indexed padding mask follows the piece's rows, also under dynamo."""

import pytest
import torch

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.models.minimax_m3.model import _piece_major_rows

TP, SHARD = 4, 16


def _ffn(x: torch.Tensor) -> torch.Tensor:
    """Row-local stand-in for the FFN."""
    return torch.tanh(x) * 3 + x.sum(-1, keepdim=True)


@pytest.mark.parametrize("pieces", [2, 4])
def test_pieces_reduce_scatter_like_the_whole_shard(pieces: int) -> None:
    full = torch.randn(TP * SHARD, 8)
    shards = full.view(TP, SHARD, 8)
    tokens = TP * SHARD
    # the all-gathers of every rank's piece i, in piece order
    gathered = torch.cat(
        [torch.cat([s.chunk(pieces)[i] for s in shards]) for i in range(pieces)]
    )
    torch.testing.assert_close(gathered, _piece_major_rows(tokens, TP, pieces, full))
    out = _ffn(gathered)
    want = _ffn(full).view(TP, SHARD, 8)
    for rank in range(TP):
        # the reduce-scatter of output piece i returns rank `rank` its block
        got = torch.cat([o.view(TP, -1, 8)[rank] for o in out.chunk(pieces)])
        torch.testing.assert_close(got, want[rank])


def _masked_rows(x: torch.Tensor, pieces: int) -> torch.Tensor:
    rows = lambda m: _piece_major_rows(x.shape[0], TP, pieces, m)  # noqa: E731
    with token_padding.padding_of_rows(rows):
        mask = token_padding._padding_state.get_local_padding_mask(x.shape[0])
        return torch.where(mask[:, None], 0.0, rows(x))


@pytest.mark.parametrize("compiled", [False, True])
def test_padding_mask_follows_the_piece(compiled: bool) -> None:
    state = token_padding.TokenPaddingState.create(TP * SHARD + 8, "cpu")
    valid = TP * SHARD - 5
    state.update(valid, TP * SHARD)
    token_padding.set_padding_state(state)
    try:
        x = torch.randn(TP * SHARD, 2)
        fn = (
            torch.compile(_masked_rows, backend="eager", fullgraph=True)
            if compiled
            else _masked_rows
        )
        got = fn(x, 2)
        assert token_padding._padding_state is state
        want = torch.where(state.local_padding_mask[: TP * SHARD, None], 0.0, x)
        torch.testing.assert_close(got, _piece_major_rows(TP * SHARD, TP, 2, want))
        assert int(got.eq(0).all(-1).sum()) == 5
    finally:
        token_padding.set_padding_state(None)


def test_attention_pieces_back_in_step_order(monkeypatch) -> None:
    from vllm_torchtpu.models.minimax_m3 import attention

    monkeypatch.setattr(attention, "get_tensor_model_parallel_world_size", lambda: TP)
    full = torch.randn(TP * SHARD, 8)
    pieces = list(_piece_major_rows(TP * SHARD, TP, 2, full).chunk(2))
    torch.testing.assert_close(attention._step_rows(pieces), full)
