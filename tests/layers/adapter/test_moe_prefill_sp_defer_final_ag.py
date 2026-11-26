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

from __future__ import annotations

import types

import pytest
import torch
from vllm.compilation import backends as vllm_compilation_backends

from vllm_torchtpu.layers.adapter import moe_prefill_sp


def _u16_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a.view(torch.uint16), b.view(torch.uint16)))


@pytest.mark.parametrize(
    "num_tokens,tp,zigzag_bands",
    [
        (32, 4, 1),
        (64, 4, 2),
        (128, 4, 4),
        (256, 16, 1),
        (512, 16, 2),
        (1024, 16, 4),
        (16384, 16, 4),
    ],
)
def test_should_invert_own_rows_zigzag_exhaustively_for_all_positions(
    num_tokens: int, tp: int, zigzag_bands: int
) -> None:
    # Unique per-token sentinel ids arranged in non-monotonic bit patterns.
    global_ids = torch.arange(num_tokens, dtype=torch.int32)
    shards = [
        moe_prefill_sp.own_rows_zigzag(
            global_ids, torch.tensor([[r]], dtype=torch.int32), tp, zigzag_bands
        )
        for r in range(tp)
    ]

    all_idx = torch.arange(num_tokens, dtype=torch.int32)
    owners, local_rows = moe_prefill_sp.zigzag_global_to_owner_and_local(
        all_idx, num_tokens, tp, zigzag_bands
    )
    assert owners.dtype == torch.int32
    assert local_rows.dtype == torch.int32
    assert int(owners.min()) == 0 and int(owners.max()) == tp - 1
    assert int(local_rows.min()) == 0 and int(local_rows.max()) == num_tokens // tp - 1

    # Gather every global row from its unique owning shard and verify exact identity.
    stacked = torch.stack(shards, dim=0)
    recovered = stacked[owners.to(torch.int64), local_rows.to(torch.int64)]
    assert torch.equal(recovered, global_ids)


def test_should_select_bitwise_exact_bf16_rows_via_masked_all_reduce() -> None:
    torch.manual_seed(7)
    num_tokens, tp, zigzag_bands, hidden = 1024, 16, 4, 12
    full_hidden = (torch.randn(num_tokens, hidden) * 7.5).to(torch.bfloat16)
    # Replace exact -0.0 with distinct finite non-zero sentinel values so
    # uint16 equality catches any masking/sign/addition error.
    full_hidden[full_hidden == 0] = torch.tensor(0.375, dtype=torch.bfloat16)

    shards = [
        moe_prefill_sp.own_rows_zigzag(
            full_hidden, torch.tensor([[r]], dtype=torch.int32), tp, zigzag_bands
        )
        for r in range(tp)
    ]

    band = num_tokens // (tp * zigzag_bands)
    indices = torch.tensor(
        [
            0,
            band - 1,
            band,
            tp * band - 1,
            tp * band,
            tp * band + band - 1,
            2 * tp * band + 3,
            num_tokens - 1,
            band,  # intentional duplicate
            0,  # out-of-order wrap
        ],
        dtype=torch.int32,
    )

    masked_per_rank: list[torch.Tensor] = []

    def _collect(t: torch.Tensor) -> torch.Tensor:
        masked_per_rank.append(t)
        return t

    for r in range(tp):
        moe_prefill_sp.select_sp_hidden_states(
            shards[r],
            indices,
            num_tokens=num_tokens,
            tp=tp,
            zigzag_bands=zigzag_bands,
            rank_tensor=torch.tensor([[r]], dtype=torch.int32),
            all_reduce_fn=_collect,
        )

    stacked = torch.stack(masked_per_rank, dim=0)
    # Verify anti-cancellation: each selected row is owned by strictly one rank.
    nonzero_ranks_per_row = (stacked.view(torch.uint16) != 0).any(dim=-1).sum(dim=0)
    assert torch.equal(nonzero_ranks_per_row, torch.ones_like(nonzero_ranks_per_row))

    reduced = stacked.sum(dim=0, dtype=torch.bfloat16)
    expected = torch.index_select(full_hidden, 0, indices)
    assert _u16_equal(reduced, expected)

    # Positive control: disabling odd-band inversion breaks zigzag=4 parity.
    wrong_parts = [
        moe_prefill_sp.select_sp_hidden_states(
            shards[r],
            indices,
            num_tokens=num_tokens,
            tp=tp,
            zigzag_bands=1,
            rank_tensor=torch.tensor([[r]], dtype=torch.int32),
            all_reduce_fn=lambda t: t,
        )
        for r in range(tp)
    ]
    wrong_reduced = torch.stack(wrong_parts, dim=0).sum(dim=0, dtype=torch.bfloat16)
    assert not _u16_equal(wrong_reduced, expected)


def test_should_fallback_to_full_gather_when_indices_cover_local_shard(
    monkeypatch,
) -> None:
    torch.manual_seed(11)
    num_tokens, tp, zigzag_bands, hidden = 128, 4, 2, 8
    full_hidden = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    shards = [
        moe_prefill_sp.own_rows_zigzag(
            full_hidden, torch.tensor([[r]], dtype=torch.int32), tp, zigzag_bands
        )
        for r in range(tp)
    ]
    gathered = torch.cat(shards, dim=0)
    gather_calls: list[int] = []

    def _fake_all_gather(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        assert dim == 0
        gather_calls.append(x.shape[0])
        return gathered

    monkeypatch.setattr(
        moe_prefill_sp, "tensor_model_parallel_all_gather", _fake_all_gather
    )

    indices = torch.arange(0, num_tokens, 2, dtype=torch.int32)
    assert indices.shape[0] >= num_tokens // tp
    out = moe_prefill_sp.select_sp_hidden_states(
        shards[0],
        indices,
        num_tokens=num_tokens,
        tp=tp,
        zigzag_bands=zigzag_bands,
        rank_tensor=torch.tensor([[0]], dtype=torch.int32),
    )
    assert gather_calls == [num_tokens // tp]
    assert _u16_equal(out, torch.index_select(full_hidden, 0, indices))


def test_should_trace_fullgraph_with_torch_compile_and_replay_distinct_inputs() -> None:
    num_tokens, tp, zigzag_bands, hidden = 256, 4, 4, 16
    rank_t = torch.tensor([[2]], dtype=torch.int32)

    def _step(local_h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return moe_prefill_sp.select_sp_hidden_states(
            local_h,
            idx,
            num_tokens=num_tokens,
            tp=tp,
            zigzag_bands=zigzag_bands,
            rank_tensor=rank_t,
            all_reduce_fn=lambda x: x,
        )

    compiled = torch.compile(_step, fullgraph=True, backend="aot_eager")
    h0 = torch.randn(num_tokens // tp, hidden, dtype=torch.bfloat16)
    i0 = torch.tensor([0, 32, 65, 255], dtype=torch.int32)
    assert _u16_equal(compiled(h0, i0), _step(h0, i0))

    h1 = torch.randn(num_tokens // tp, hidden, dtype=torch.bfloat16) * 3.0
    i1 = torch.tensor([17, 100, 144, 211], dtype=torch.int32)
    assert _u16_equal(compiled(h1, i1), _step(h1, i1))


def test_should_report_bands_and_deferred_flag_outside_backbone_model_tag(
    monkeypatch,
) -> None:
    class _FakeLayout:
        dcp_size = 16

        @staticmethod
        def effective_zigzag_bands(num_tokens: int) -> int:
            return 4 if num_tokens >= 1024 else 1

    layer = types.SimpleNamespace(
        _tpu_sp_keep=[torch.empty(1)],
        self_attn=types.SimpleNamespace(
            mla_attn=types.SimpleNamespace(
                mla_attn=types.SimpleNamespace(
                    sparse_mla_tp_op=types.SimpleNamespace(layout=_FakeLayout())
                )
            )
        ),
    )
    backbone = types.SimpleNamespace(
        layers=[layer],
        start_layer=0,
        end_layer=1,
        _tpu_sp_keep=True,
        _tpu_sp_defer_final_ag=True,
    )
    model = types.SimpleNamespace(model=backbone)
    monkeypatch.setattr(
        moe_prefill_sp, "get_tensor_model_parallel_world_size", lambda: 16
    )
    monkeypatch.setattr(vllm_compilation_backends, "model_tag", "draft_not_backbone")
    monkeypatch.setattr(moe_prefill_sp, "_SP_ENABLED", True)
    monkeypatch.setattr(moe_prefill_sp.dcp_token_parallel_op._state, "enabled", True)
    monkeypatch.setattr(
        moe_prefill_sp.dcp_token_parallel_op._state, "over_capacity", False
    )

    assert moe_prefill_sp.is_final_ag_deferred_for_bucket(model, 16384) is True
    assert moe_prefill_sp.is_final_ag_deferred_for_bucket(model, 64) is False
    assert moe_prefill_sp.step_zigzag_bands_for_model(model, 16384) == 4

    # Non-DeepSeek / un-armed model returns False cleanly without raising.
    plain = types.SimpleNamespace(model=types.SimpleNamespace())
    assert moe_prefill_sp.is_final_ag_deferred_for_bucket(plain, 16384) is False
    assert moe_prefill_sp.step_zigzag_bands_for_model(plain, 16384) == 1
