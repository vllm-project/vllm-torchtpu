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
"""TPU_MOE_PREFILL_SEQUENCE_PARALLEL: the bucket gate, the refusals, the
re-sharded shared expert and the fused EP op it builds. No TPU."""

from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp


def _shards(full: torch.Tensor, tp: int, dim: int) -> list[torch.Tensor]:
    return list(full.chunk(tp, dim=dim))


@pytest.mark.parametrize("tp", [2, 16])
def test_merge_gate_up_undoes_the_merged_column_sharding(tp):
    k, inter = 8, 4 * tp
    gate, up = torch.randn(k, inter), torch.randn(k, inter)
    g_s, u_s = torch.rand(inter), torch.rand(inter)
    # Rank r holds [gate_r | up_r], as MergedColumnParallelLinear loads it.
    w = torch.cat(
        [torch.cat(p, 1) for p in zip(_shards(gate, tp, 1), _shards(up, tp, 1))], 1
    )
    s = torch.cat([torch.cat(p) for p in zip(_shards(g_s, tp, 0), _shards(u_s, tp, 0))])
    w_rep, s_rep = sp.merge_gate_up(w, s, tp)
    torch.testing.assert_close(w_rep, torch.cat([gate, up], 1), rtol=0, atol=0)
    torch.testing.assert_close(s_rep, torch.cat([g_s, u_s]), rtol=0, atol=0)


def _silu_mul(x):
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[:, :d]) * x[:, d:]


@pytest.mark.parametrize("group", [1, 2, 4])
def test_member_slices_sum_to_the_tp_sharded_expert(group):
    tp, k, i, n, rows = 4, 8, 3, 5, 2
    w13 = torch.randn(k, tp * 2 * i)  # shard r: [gate_r | up_r]
    s13 = torch.rand(tp * 2 * i)
    w2, s2 = torch.randn(tp * i, n), torch.rand(tp, n)
    x = torch.randn(rows, k)
    # Today: each TP shard's partial, summed (the all-reduce).
    want = 0
    for r in range(tp):
        cols = slice(r * 2 * i, (r + 1) * 2 * i)
        act = _silu_mul(x @ (w13[:, cols] * s13[cols]))
        want = want + act @ (w2[r * i : (r + 1) * i] * s2[r])
    got = 0
    for m in range(group):
        gu, gu_s, d, d_s = sp.member_slice(
            w13, s13, w2, s2, tp=tp, group=group, member=m
        )
        act = _silu_mul(x @ (gu * gu_s))
        per = tp // group
        assert d_s.shape == (per, n)
        deq = (d.view(per, i, n) * d_s[:, None, :]).reshape(per * i, n)
        got = got + act @ deq
    torch.testing.assert_close(got, want)


def test_member_slice_refuses_an_uneven_split():
    with pytest.raises(ValueError):
        sp.member_slice(
            torch.zeros(4, 12),
            torch.zeros(12),
            torch.zeros(6, 4),
            torch.zeros(3, 4),
            tp=3,
            group=2,
            member=0,
        )


@pytest.mark.parametrize("rank", [0, 3])
def test_own_rows_are_the_rank_s_contiguous_slice(rank):
    tp, t, h = 4, 2, 3
    x = torch.arange(tp * t * h, dtype=torch.float32).view(tp * t, h)
    got = sp.own_rows(x, torch.tensor([[rank]], dtype=torch.int32), tp)
    torch.testing.assert_close(got, x[rank * t : (rank + 1) * t])


@pytest.mark.parametrize(
    "flag,tag,tokens,want",
    [
        (True, "backbone", 1024, True),
        (True, "backbone", 4096, True),
        (True, "backbone", 512, False),
        (True, "backbone", 64, False),
        (True, "eagle_head", 4096, False),
        (False, "backbone", 4096, False),
    ],
)
def test_applies_only_to_large_backbone_buckets(flag, tag, tokens, want):
    with (
        patch.object(envs, "TPU_MOE_PREFILL_SEQUENCE_PARALLEL", flag),
        patch.object(sp.vllm_compilation_backends, "model_tag", tag),
    ):
        assert sp.applies(tokens) is want


def _config(**parallel):
    p = dict(
        enable_expert_parallel=True,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=16,
    )
    p.update(parallel)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**p), speculative_config=None
    )


_OFF = dict(
    TPU_MOE_SEQUENCE_PARALLEL=False,
    USE_MOE_FUSED_EP_KERNEL=False,
    USE_MOE_TP_DECODE_KERNEL=False,
    TPU_FUSED_ADD_RMSNORM=False,
    TPU_DCP_MLA_OPROJ_ALL_REDUCE=False,
    TPU_MOE_SKIP_PADDED_TOKENS=False,
)


def _with_envs(**values):
    patches = [patch.object(envs, k, v) for k, v in (_OFF | values).items()]
    for p in patches:
        p.start()
    return patches


def test_the_served_config_is_admitted():
    patches = _with_envs()
    try:
        sp.check_config(_config())
    finally:
        for p in patches:
            p.stop()


@pytest.mark.parametrize(
    "parallel,flags,match",
    [
        (dict(enable_expert_parallel=False), {}, "expert-parallel"),
        (dict(data_parallel_size=2), {}, "data_parallel"),
        (dict(pipeline_parallel_size=2), {}, "pipeline"),
        ({}, dict(USE_MOE_FUSED_EP_KERNEL=True), "USE_MOE_FUSED_EP_KERNEL"),
        ({}, dict(TPU_FUSED_ADD_RMSNORM=True), "TPU_FUSED_ADD_RMSNORM"),
        ({}, dict(TPU_MOE_SKIP_PADDED_TOKENS=True), "SKIP_PADDED"),
        ({}, dict(TPU_DCP_MLA_OPROJ_ALL_REDUCE=True), "OPROJ_ALL_REDUCE"),
    ],
)
def test_unserved_configs_are_refused(parallel, flags, match):
    patches = _with_envs(**flags)
    try:
        with pytest.raises(NotImplementedError, match=match):
            sp.check_config(_config(**parallel))
    finally:
        for p in patches:
            p.stop()


def test_speculative_decoding_is_refused():
    cfg = _config()
    cfg.speculative_config = object()
    patches = _with_envs()
    try:
        with pytest.raises(NotImplementedError, match="speculative"):
            sp.check_config(cfg)
    finally:
        for p in patches:
            p.stop()


def _experts(local=2, ep=2, hidden=8, inter=16):
    return SimpleNamespace(
        w13_weight=torch.zeros(local, hidden, 2 * inter, dtype=torch.float8_e4m3fn),
        w2_weight=torch.zeros(local, inter, hidden, dtype=torch.float8_e4m3fn),
        w13_weight_scale_inv=torch.ones(local, 1, 1, 2 * inter),
        w2_weight_scale_inv=torch.ones(local, 1, 1, hidden),
        global_num_experts=local * ep,
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=torch.zeros(local * ep),
        use_grouped_topk=True,
        num_expert_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        moe_config=SimpleNamespace(
            experts_per_token=2,
            moe_parallel_config=SimpleNamespace(
                use_ep=True, pcp_size=1, is_sequence_parallel=False
            ),
        ),
    )


def _build(layer, **stubs):
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: 2})
    seen = []
    with (
        patch.object(bridge, "build_ep_mesh", return_value=stubs.get("mesh", mesh)),
        patch.object(bridge, "ep_mesh_index", return_value=0),
        patch.object(bridge, "ep_rank_order", return_value=None),
        patch.object(
            bridge,
            "ep_token_replica_groups",
            side_effect=lambda *, is_sequence_parallel: (
                ((0,), (1,)) if is_sequence_parallel else ((0, 1),)
            ),
        ),
        patch.object(bridge, "_smem_capacity_bytes", return_value=1024 * 1024),
        patch.object(
            bridge,
            "_build_op",
            side_effect=lambda *a, **kw: seen.append((a, kw)) or "op",
        ),
        patch.object(bridge, "_RANK_BUFFER", None),
        patch.object(envs, "USE_MOE_FUSED_EP_KERNEL", False),
    ):
        op = bridge.build_sequence_parallel_op(
            layer, topk=2, renormalize=True, activation="silu"
        )
    return op, seen


def test_op_takes_own_tokens_and_a_bf16_wire_without_arming_the_layer():
    layer = _experts()
    op, seen = _build(layer)
    assert op == "op"
    args, kw = seen[0]
    assert kw["token_replica_groups"] == ((0,), (1,))
    assert kw["wire_dtype"] == jnp.bfloat16
    assert args[8:11] == ("sigmoid", True, 1.0)  # scoring, bias, routed scale
    # The other buckets keep the grouped-matmul path and its collectives.
    assert not bridge.fused_moe_ep_supported(layer)


@pytest.mark.parametrize(
    "change,match",
    [
        (dict(global_num_experts=5), "equal blocks"),
        (dict(scoring_func="weird"), "scoring_func"),
        (dict(w13_weight_scale_inv=torch.ones(2, 3, 1, 32)), "per-output-channel"),
    ],
)
def test_op_refusals_raise(change, match):
    layer = _experts()
    for k, v in change.items():
        setattr(layer, k, v)
    with pytest.raises(ValueError, match=match):
        _build(layer)


def test_op_without_an_ep_group_raises():
    with pytest.raises(ValueError, match="EP group"):
        _build(_experts(), mesh=None)
