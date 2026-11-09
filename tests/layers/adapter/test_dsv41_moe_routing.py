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
"""100% Diff coverage test suite for partitioned SP MoE routing.

Covers all lines and branches in:
- vllm_torchtpu.layers.adapter.custom_ops.moe_runner
  (moe_runner.py:134-154, VllmMoERunner._maybe_dispatch)
- vllm_torchtpu.layers.adapter.moe_routing
  (moe_routing.py:136-225, _gather_partitioned_topk and route)
- vllm_torchtpu.models.vllm.deepseek_v41.v2.attention
  (attention.py:456-475, 617-621, _prepare_tpu & forward SP gather assertions)
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.custom_ops import moe_runner as oot_moe_runner
from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.models.vllm.deepseek_v41.v2 import attention as v2_attn
from vllm_torchtpu.models.vllm.deepseek_v41.v2.attention import DeepseekV41Attention

# ==============================================================================
# Helpers & Mocks
# ==============================================================================


class _MockMoERunner(VllmMoERunner):
    """Subclass of VllmMoERunner that bypasses full upstream __init__."""

    def __init__(
        self,
        *,
        do_naive_dispatch_combine: bool = True,
        pcp_size: int = 1,
        is_sequence_parallel: bool = True,
        is_monolithic: bool = True,
        custom_routing_function=None,
        fused_ep_owns_pcp: bool = False,
    ):
        nn.Module.__init__(self)
        self._do_naive = do_naive_dispatch_combine
        self.moe_config = SimpleNamespace(
            pcp_size=pcp_size,
            is_sequence_parallel=is_sequence_parallel,
        )
        self.routed_experts = SimpleNamespace(
            quant_method=SimpleNamespace(is_monolithic=is_monolithic),
            custom_routing_function=custom_routing_function,
        )
        self._fused_ep_owns_pcp = fused_ep_owns_pcp

    @property
    def do_naive_dispatch_combine(self) -> bool:
        return self._do_naive

    def _fused_ep_owns_pcp_collectives(self) -> bool:
        return self._fused_ep_owns_pcp


class _FakeDistributedGroup:
    """Mock collective communication group for all_gather testing."""

    def __init__(self, world_size: int = 4, rank: int = 0):
        self.world_size = world_size
        self.rank_in_group = rank
        self.all_gather_calls: list[tuple[torch.Tensor, int]] = []

    def all_gather(self, tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        self.all_gather_calls.append((tensor.clone(), dim))
        repeats = [1] * tensor.ndim
        repeats[dim] = self.world_size
        return tensor.repeat(*repeats)


# ==============================================================================
# Part 1: VllmMoERunner._maybe_dispatch (moe_runner.py:134-154)
# ==============================================================================


def test_moe_runner_maybe_dispatch_fused_ep_owns_pcp():
    """Lines 132-136: If _fused_ep_owns_pcp_collectives is True, returns directly."""
    runner = _MockMoERunner(fused_ep_owns_pcp=True)
    hidden = torch.randn(8, 16)
    logits = torch.randn(8, 32)
    h_out, r_out = runner._maybe_dispatch(hidden, logits)
    assert h_out is hidden
    assert r_out is logits


def test_moe_runner_maybe_dispatch_partitioned_sp_world_size_gt_1(monkeypatch):
    """Lines 137-152: SP=True, world_size > 1 uses get_ep_group to gather hidden."""
    fake_group = _FakeDistributedGroup(world_size=4, rank=0)
    monkeypatch.setattr(oot_moe_runner, "get_ep_group", lambda: fake_group)
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

    runner = _MockMoERunner(
        do_naive_dispatch_combine=True,
        pcp_size=1,
        is_sequence_parallel=True,
        is_monolithic=True,
        custom_routing_function=None,
    )
    hidden = torch.randn(8, 16)
    logits = torch.randn(8, 32)  # router_logits is not hidden_states

    h_out, r_out = runner._maybe_dispatch(hidden, logits)
    assert len(fake_group.all_gather_calls) == 1
    assert fake_group.all_gather_calls[0][0].shape == hidden.shape
    assert h_out.shape == (32, 16)  # 8 * 4
    assert r_out is logits


def test_moe_runner_maybe_dispatch_partitioned_dp_world_size_gt_1(monkeypatch):
    """Lines 137-152: SP=False, world_size > 1 uses get_dp_group to gather hidden."""
    fake_group = _FakeDistributedGroup(world_size=2, rank=1)
    monkeypatch.setattr(oot_moe_runner, "get_dp_group", lambda: fake_group)
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

    runner = _MockMoERunner(
        do_naive_dispatch_combine=True,
        pcp_size=1,
        is_sequence_parallel=False,
        is_monolithic=True,
        custom_routing_function=None,
    )
    hidden = torch.randn(10, 16)
    logits = torch.randn(10, 32)

    h_out, r_out = runner._maybe_dispatch(hidden, logits)
    assert len(fake_group.all_gather_calls) == 1
    assert h_out.shape == (20, 16)  # 10 * 2
    assert r_out is logits


def test_moe_runner_maybe_dispatch_partitioned_world_size_1(monkeypatch):
    """Lines 150-151: When group.world_size <= 1, all_gather is skipped."""
    fake_group = _FakeDistributedGroup(world_size=1, rank=0)
    monkeypatch.setattr(oot_moe_runner, "get_ep_group", lambda: fake_group)
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

    runner = _MockMoERunner(
        do_naive_dispatch_combine=True,
        pcp_size=1,
        is_sequence_parallel=True,
        is_monolithic=True,
    )
    hidden = torch.randn(8, 16)
    logits = torch.randn(8, 32)

    h_out, r_out = runner._maybe_dispatch(hidden, logits)
    assert len(fake_group.all_gather_calls) == 0
    assert h_out is hidden
    assert r_out is logits


@pytest.mark.parametrize(
    "neg_case",
    [
        "not_naive",
        "pcp_gt_1",
        "not_monolithic",
        "has_custom_routing",
        "has_simulation",
        "router_is_hidden",
    ],
)
def test_moe_runner_maybe_dispatch_fallback_branches(monkeypatch, neg_case):
    """Line 153: If any of the 6 conditions is False, falls back to super()."""
    super_mock = MagicMock(return_value=("super_hidden", "super_logits"))
    monkeypatch.setattr(MoERunner, "_maybe_dispatch", super_mock)
    monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", None)

    runner = _MockMoERunner(
        do_naive_dispatch_combine=(neg_case != "not_naive"),
        pcp_size=(2 if neg_case == "pcp_gt_1" else 1),
        is_monolithic=(neg_case != "not_monolithic"),
        custom_routing_function=(
            (lambda **kw: None) if neg_case == "has_custom_routing" else None
        ),
    )
    if neg_case == "has_simulation":
        sim_mock = SimpleNamespace(route_tokens=lambda **kw: (None, None))
        monkeypatch.setattr(moe_routing, "_SIMULATION_STRATEGY", sim_mock)

    hidden = torch.randn(8, 16)
    logits = hidden if neg_case == "router_is_hidden" else torch.randn(8, 32)

    h_out, r_out = runner._maybe_dispatch(hidden, logits)
    assert super_mock.called
    assert (h_out, r_out) == ("super_hidden", "super_logits")


# ==============================================================================
# Part 2: moe_routing._gather_partitioned_topk & route (moe_routing.py:136-225)
# ==============================================================================


def test_gather_partitioned_topk_bitpacked_bf16():
    """Lines 150-167: bfloat16 + even topk + int32 ids -> packed int32 all_gather."""
    group = _FakeDistributedGroup(world_size=2, rank=0)
    tokens, topk = 4, 6
    weights = torch.randn(tokens, topk, dtype=torch.bfloat16)
    ids = torch.randint(0, 100, (tokens, topk), dtype=torch.int32)

    out_w, out_i = moe_routing._gather_partitioned_topk(group, weights, ids, topk)

    assert len(group.all_gather_calls) == 1
    packed_sent, dim = group.all_gather_calls[0]
    assert dim == 0
    assert packed_sent.shape == (tokens, topk // 2 + topk)
    assert packed_sent.dtype == torch.int32
    assert out_w.shape == (tokens * 2, topk)
    assert out_w.dtype == torch.bfloat16
    assert out_i.shape == (tokens * 2, topk)
    assert out_i.dtype == torch.int32
    # Check rank 0 reconstruction
    assert torch.equal(out_w[:tokens], weights)
    assert torch.equal(out_i[:tokens], ids)


def test_gather_partitioned_topk_bitpacked_fp16():
    """Lines 150-167: float16 + even topk + int32 ids -> packed int32 all_gather."""
    group = _FakeDistributedGroup(world_size=2, rank=0)
    tokens, topk = 4, 4
    weights = torch.randn(tokens, topk, dtype=torch.float16)
    ids = torch.randint(0, 50, (tokens, topk), dtype=torch.int32)

    out_w, out_i = moe_routing._gather_partitioned_topk(group, weights, ids, topk)

    assert len(group.all_gather_calls) == 1
    assert out_w.dtype == torch.float16
    assert out_i.dtype == torch.int32
    assert torch.equal(out_w[:tokens], weights)
    assert torch.equal(out_i[:tokens], ids)


@pytest.mark.parametrize(
    "case,weights_dtype,topk,ids_dtype",
    [
        ("float32_weights", torch.float32, 6, torch.int32),
        ("odd_topk", torch.bfloat16, 5, torch.int32),
        ("int64_ids", torch.bfloat16, 6, torch.int64),
    ],
)
def test_gather_partitioned_topk_fallback_separate_gathers(
    case, weights_dtype, topk, ids_dtype
):
    """Lines 168-171: Fallback separate all_gather for non-packable configurations."""
    group = _FakeDistributedGroup(world_size=2, rank=0)
    tokens = 4
    weights = torch.randn(tokens, topk, dtype=weights_dtype)
    ids = torch.randint(0, 50, (tokens, topk), dtype=ids_dtype)

    out_w, out_i = moe_routing._gather_partitioned_topk(group, weights, ids, topk)

    assert len(group.all_gather_calls) == 2
    assert group.all_gather_calls[0][0].dtype == weights_dtype
    assert group.all_gather_calls[1][0].dtype == ids_dtype
    assert out_w.shape == (tokens * 2, topk)
    assert out_i.shape == (tokens * 2, topk)
    assert torch.equal(out_w[:tokens], weights)
    assert torch.equal(out_i[:tokens], ids)


# ==============================================================================
# Part 3: v2/attention.py (_prepare_tpu lines 617-621 & forward lines 456-475)
# ==============================================================================


def test_forward_sp_sharded_with_indexer_weights(monkeypatch):
    """Lines 457-470: SP sharded forward with indexer_weights gathers components."""
    gathered_calls = []

    def fake_sp_all_gather(tensor):
        gathered_calls.append(tensor)
        return tensor

    monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_all_gather)

    num_tokens = 4
    q_lora_rank = 8
    head_dim = 16
    n_head = 4

    qr = torch.randn(num_tokens, q_lora_rank, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16)
    indexer_w = torch.randn(num_tokens, n_head, dtype=torch.bfloat16)
    kv_score = torch.randn(num_tokens, 2, dtype=torch.float32)

    attn_fn_mock = MagicMock()
    self_obj = SimpleNamespace(
        num_local_heads=2,
        n_local_heads=2,
        head_dim=head_dim,
        padded_heads=2,
        _sp_sharded_input=True,
        _fused_preprocess=False,
        q_lora_rank=q_lora_rank,
        indexer=SimpleNamespace(n_head=n_head),
        _run_parallel_input_projections=lambda hs: (None, kv_score, indexer_w),
        _split_qkv_and_norm=lambda qk: (qr, "scale", kv),
        _prepare_and_attn_fn=attn_fn_mock,
        _o_proj=lambda o, pos: o,
    )

    hidden = torch.randn(num_tokens, 32, dtype=torch.bfloat16)
    positions = torch.arange(num_tokens, dtype=torch.int64)

    DeepseekV41Attention.forward(self_obj, positions, hidden)
    assert len(gathered_calls) == 2  # packed (qr, kv, indexer) + kv_score
    assert gathered_calls[0].shape == (num_tokens, q_lora_rank + head_dim + n_head)
    assert gathered_calls[1].shape == kv_score.shape
    assert attn_fn_mock.called


def test_forward_sp_sharded_indexer_weights_dtype_mismatch_raises():
    """Line 459: assert qr.dtype == kv.dtype == indexer_weights.dtype."""
    num_tokens = 4
    qr = torch.randn(num_tokens, 8, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, 16, dtype=torch.bfloat16)
    indexer_w = torch.randn(num_tokens, 4, dtype=torch.float32)  # mismatched!

    self_obj = SimpleNamespace(
        num_local_heads=2,
        n_local_heads=2,
        head_dim=16,
        padded_heads=2,
        _sp_sharded_input=True,
        _fused_preprocess=False,
        q_lora_rank=8,
        indexer=SimpleNamespace(n_head=4),
        _run_parallel_input_projections=lambda hs: (None, None, indexer_w),
        _split_qkv_and_norm=lambda qk: (qr, None, kv),
        _o_proj=lambda o, pos: o,
    )
    with pytest.raises(AssertionError):
        DeepseekV41Attention.forward(
            self_obj, torch.arange(4, dtype=torch.int64), torch.randn(4, 32)
        )


def test_forward_sp_sharded_without_indexer_weights(monkeypatch):
    """Lines 471-481: SP sharded forward without indexer_weights gathers (qr, kv)."""
    gathered_calls = []

    def fake_sp_all_gather(tensor):
        gathered_calls.append(tensor)
        return tensor

    monkeypatch.setattr(v2_attn, "sp_all_gather", fake_sp_all_gather)

    num_tokens = 4
    q_lora_rank = 8
    head_dim = 16

    qr = torch.randn(num_tokens, q_lora_rank, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, head_dim, dtype=torch.bfloat16)

    attn_fn_mock = MagicMock()
    self_obj = SimpleNamespace(
        num_local_heads=2,
        n_local_heads=2,
        head_dim=head_dim,
        padded_heads=2,
        _sp_sharded_input=True,
        _fused_preprocess=False,
        q_lora_rank=q_lora_rank,
        _run_parallel_input_projections=lambda hs: (None, None, None),
        _split_qkv_and_norm=lambda qk: (qr, None, kv),
        _prepare_and_attn_fn=attn_fn_mock,
        _o_proj=lambda o, pos: o,
    )

    DeepseekV41Attention.forward(
        self_obj,
        torch.arange(num_tokens, dtype=torch.int64),
        torch.randn(num_tokens, 32),
    )
    assert len(gathered_calls) == 1  # packed (qr, kv) only, no kv_score
    assert gathered_calls[0].shape == (num_tokens, q_lora_rank + head_dim)
    assert attn_fn_mock.called


def test_forward_sp_sharded_qr_kv_dtype_mismatch_raises():
    """Line 471: assert qr.dtype == kv.dtype."""
    num_tokens = 4
    qr = torch.randn(num_tokens, 8, dtype=torch.bfloat16)
    kv = torch.randn(num_tokens, 16, dtype=torch.float32)  # mismatched!

    self_obj = SimpleNamespace(
        num_local_heads=2,
        n_local_heads=2,
        head_dim=16,
        padded_heads=2,
        _sp_sharded_input=True,
        _fused_preprocess=False,
        q_lora_rank=8,
        _run_parallel_input_projections=lambda hs: (None, None, None),
        _split_qkv_and_norm=lambda qk: (qr, None, kv),
        _o_proj=lambda o, pos: o,
    )
    with pytest.raises(AssertionError):
        DeepseekV41Attention.forward(
            self_obj, torch.arange(4, dtype=torch.int64), torch.randn(4, 32)
        )


def test_forward_unsharded_bypasses_sp_gathers(monkeypatch):
    """Lines 457: _sp_sharded_input=False skips sp_all_gather completely."""
    gather_mock = MagicMock()
    monkeypatch.setattr(v2_attn, "sp_all_gather", gather_mock)

    num_tokens = 4
    qr = torch.randn(num_tokens, 8)
    kv = torch.randn(num_tokens, 16)

    attn_fn_mock = MagicMock()
    self_obj = SimpleNamespace(
        num_local_heads=2,
        n_local_heads=2,
        head_dim=16,
        padded_heads=2,
        _sp_sharded_input=False,
        _fused_preprocess=False,
        _run_parallel_input_projections=lambda hs: (None, None, None),
        _split_qkv_and_norm=lambda qk: (qr, None, kv),
        _prepare_and_attn_fn=attn_fn_mock,
        _o_proj=lambda o, pos: o,
    )
    DeepseekV41Attention.forward(
        self_obj,
        torch.arange(num_tokens, dtype=torch.int64),
        torch.randn(num_tokens, 32),
    )
    assert not gather_mock.called
    assert attn_fn_mock.called
