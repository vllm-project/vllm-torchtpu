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

import dataclasses
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import jax.numpy as jnp
import pytest
import torch
from torch import fx, nn
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.schemas import SerializableAOTDispatchCompiler
from torch._functorch.aot_autograd import aot_module_simplified
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._inductor.output_code import OutputCode
from torch._subclasses.fake_tensor import FakeTensor
from torch_tpu._internal.compile._backend import (
    _TPU_DECOMPOSITIONS,
    _serialization_context,
)
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.piecewise_backend import create_concrete_args
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    DeviceConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.models import deepseek_v2
from vllm.platforms import current_platform

import vllm_torchtpu
import vllm_torchtpu.envs as envs
from vllm_torchtpu.compilation import shape_variants, tpu_compiler
from vllm_torchtpu.layers.adapter import fused_moe_ep as bridge
from vllm_torchtpu.layers.adapter import moe_prefill_sp as sp
from vllm_torchtpu.layers.adapter import rmsnorm
from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op
from vllm_torchtpu.layers.adapter.linear_common import PREQUANT_ATTR, PrequantSlot


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


def _config(dtype=torch.bfloat16, **parallel):
    p = dict(
        enable_expert_parallel=True,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=16,
    )
    p.update(parallel)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**p),
        speculative_config=None,
        model_config=SimpleNamespace(dtype=dtype),
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
    patches = _with_envs(
        USE_MOE_TP_DECODE_KERNEL=True,
        TPU_FUSED_ADD_RMSNORM=True,
        TPU_MOE_SKIP_PADDED_TOKENS=True,
    )
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


def test_non_bf16_models_are_refused():
    patches = _with_envs()
    try:
        with pytest.raises(NotImplementedError, match="bfloat16"):
            sp.check_config(_config(dtype=torch.float16))
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


def test_op_takes_own_tokens_and_a_bf16_wire_without_arming_the_layer(monkeypatch):
    monkeypatch.setenv("TPU_MOE_PREFILL_SP_FP8_WIRE", "0")
    layer = _experts()
    op, seen = _build(layer)
    assert op == "op"
    args, kw = seen[0]
    assert kw["token_replica_groups"] == ((0,), (1,))
    assert kw["wire_dtype"] == jnp.bfloat16
    assert args[5] is True  # sharded routing plan
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


def test_op_forwards_ep_tp_size_and_validates_expert_blocks():
    layer = _experts(local=4, ep=2)
    layer.moe_config.moe_parallel_config.tp_size = 2
    mesh = SimpleNamespace(shape={bridge.EP_AXIS_NAME: 4})
    op, seen = _build(layer, mesh=mesh)
    assert op == "op"
    _, kw = seen[0]
    assert kw["ep_tp_size"] == 2


def test_op_refuses_unaligned_or_permuted_ep_tp_mesh():
    layer = _experts(local=4, ep=2)
    layer.moe_config.moe_parallel_config.tp_size = 3
    mesh4 = SimpleNamespace(shape={bridge.EP_AXIS_NAME: 4})
    with pytest.raises(ValueError, match="not divisible by MoE tp_size"):
        _build(layer, mesh=mesh4)

    layer.moe_config.moe_parallel_config.tp_size = 2
    with (
        patch.object(bridge, "build_ep_mesh", return_value=mesh4),
        patch.object(bridge, "ep_rank_order", return_value=(0, 2, 1, 3)),
        patch.object(bridge, "_smem_capacity_bytes", return_value=1024 * 1024),
        pytest.raises(ValueError, match="rank-ordered EP mesh"),
    ):
        bridge.build_sequence_parallel_op(
            layer, topk=2, renormalize=True, activation="silu"
        )


def test_shared_slices_are_stacked_into_four_model_wide_tensors():
    tp, group, k, i, n, layers = 4, 2, 8, 3, 5, 3
    backbone = torch.nn.Module()
    mlps = [SimpleNamespace() for _ in range(layers)]
    parts = [
        sp.member_slice(
            torch.randn(k, tp * 2 * i),
            torch.rand(tp * 2 * i),
            torch.randn(tp * i, n),
            torch.rand(tp, n),
            tp=tp,
            group=group,
            member=1,
        )
        for _ in range(layers)
    ]
    sp._stack_shared_experts(backbone, mlps, parts)
    assert sorted(name for name, _ in backbone.named_buffers()) == sorted(
        sp.SHARED_STACK
    )
    for idx, mlp in enumerate(mlps):
        stacked = getattr(mlp, sp.SHARED_ATTR)
        assert getattr(mlp, sp.SHARED_INDEX) == idx
        # One set of tensors for every layer: one lifted argument each.
        assert all(a is b for a, b in zip(stacked, getattr(mlps[0], sp.SHARED_ATTR)))
        w13, s13, w2, s2 = (t[idx] for t in stacked)
        assert w13.dtype == w2.dtype == torch.float8_e4m3fn
        torch.testing.assert_close(s13, parts[idx][1], rtol=0, atol=0)
        torch.testing.assert_close(s2, parts[idx][3], rtol=0, atol=0)
        torch.testing.assert_close(
            w2.float(), parts[idx][2].to(torch.float8_e4m3fn).float(), rtol=0, atol=0
        )


@pytest.mark.parametrize("tp,num_tokens,split_rows", [(4, 128, 16), (16, 8192, 64)])
def test_gather_projections_slim_qrows_matches_full_gather(tp, num_tokens, split_rows):
    from vllm_torchtpu.kernels.mla.dcp_token_parallel import TokenParallelLayout
    from vllm_torchtpu.layers.adapter import dcp_token_parallel_op as tp_op_mod

    rows_local = num_tokens // tp
    q_lora, kv_dim, kw_dim, hidden = 1536, 576, 192, 32
    full_x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    w_qkv = torch.randn(hidden, q_lora + kv_dim, dtype=torch.bfloat16)
    w_kw = torch.randn(hidden, kw_dim, dtype=torch.bfloat16)
    full_qkv = full_x @ w_qkv
    full_qc, full_kv = full_qkv.split([q_lora, kv_dim], dim=-1)
    full_kw = full_x @ w_kw
    layout = TokenParallelLayout(tp, 4, 256, split_rows=split_rows, max_pages=8)

    # Mock all-gather concatenating every rank's local tensor on dim 0.
    local_parts: dict[int, torch.Tensor] = {}

    for r in range(tp):
        x_r = full_x[r * rows_local : (r + 1) * rows_local]
        qkv_r = x_r @ w_qkv
        qc_r, kv_r = qkv_r.split([q_lora, kv_dim], dim=-1)
        cols_qc = (split_rows * q_lora) // rows_local
        local_parts[r] = torch.cat(
            [kv_r, qc_r[:split_rows].reshape(rows_local, cols_qc), x_r @ w_kw], dim=-1
        )
    gathered_all = torch.cat([local_parts[r] for r in range(tp)], dim=0)
    assert gathered_all.shape == (
        num_tokens,
        kv_dim + (split_rows * q_lora) // rows_local + kw_dim,
    )

    for r in range(tp):
        rank_t = torch.tensor([r], dtype=torch.int32)
        tp_op = SimpleNamespace(layout=layout, q_replicated=True, rank=rank_t)
        indexer = SimpleNamespace(
            wk_weights_proj=lambda z: (z @ w_kw, None),
            indexer_op=SimpleNamespace(topk_op_tp=SimpleNamespace(q_own_rows=True)),
        )
        mla = SimpleNamespace(
            fused_qkv_a_proj=lambda z: (z @ w_qkv, None),
            indexer=indexer,
            skip_topk=False,
            mla_attn=SimpleNamespace(sparse_mla_tp_op=tp_op),
            gate_is_fused=False,
            kimi_preprocess=None,
            qb_overlap=False,
            q_lora_rank=q_lora,
        )
        with (
            patch.object(tp_op_mod, "applies", return_value=True),
            patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
            patch.object(
                sp, "tensor_model_parallel_all_gather", return_value=gathered_all
            ),
        ):
            pre_qkv, pre_kw = sp._GatherProjections.project_gather(
                mla, full_x[r * rows_local : (r + 1) * rows_local], num_tokens
            )
        assert isinstance(pre_qkv, tuple)
        got_qc, got_kv = pre_qkv
        q_idx = tp_op_mod.q_rows(tp_op, num_tokens)
        want_qc = full_qc.index_select(0, q_idx)
        torch.testing.assert_close(got_qc, want_qc, rtol=0, atol=0)
        torch.testing.assert_close(got_kv, full_kv, rtol=0, atol=0)
        assert pre_kw is not None
        torch.testing.assert_close(pre_kw, full_kw, rtol=0, atol=0)


def test_should_select_intra_chip_dense_group_when_tp_is_even():
    assert sp._dense_group_size(16, 4) == 2  # noqa: SLF001 -- policy unit test
    assert sp._dense_group_size(2, 2) == 2  # noqa: SLF001 -- policy unit test
    assert sp._dense_group_size(16, 1) == 1  # noqa: SLF001 -- policy unit test
    assert sp._dense_group_size(3, 3) == 1  # noqa: SLF001 -- policy unit test


def test_should_match_vocab_parallel_all_reduce_slice_when_embed_reduce_scatters():
    vllm_torchtpu._patch_vllm_vocab_parallel_embedding()  # noqa: SLF001 -- installs TPU-patched mask helper used at runtime
    tp, num_tokens, vocab_per_rank, hidden = 4, 16, 8, 6
    rows_local = num_tokens // tp
    full_table = torch.randn(tp * vocab_per_rank, hidden, dtype=torch.bfloat16)
    input_ids = torch.randint(0, tp * vocab_per_rank, (num_tokens,), dtype=torch.int32)
    expected_full = full_table[input_ids.long()]

    partials = []
    embeds = []
    for r in range(tp):
        start = r * vocab_per_rank
        end = (r + 1) * vocab_per_rank
        shard_w = full_table[start:end]

        class _FakeQuant:
            def embedding(self, layer, idx):
                return layer.weight[idx]

        added_bound = tp * vocab_per_rank
        embed = SimpleNamespace(
            weight=shard_w,
            tp_size=tp,
            use_fused_embedding=False,
            org_vocab_start_index=torch.tensor(start, dtype=torch.int32),
            org_vocab_end_index=torch.tensor(end, dtype=torch.int32),
            num_org_vocab_padding=torch.tensor(0, dtype=torch.int32),
            added_vocab_start_index=torch.tensor(added_bound, dtype=torch.int32),
            added_vocab_end_index=torch.tensor(added_bound, dtype=torch.int32),
            shard_indices=SimpleNamespace(
                org_vocab_start_index=start,
                org_vocab_end_index=end,
                num_org_vocab_padding=0,
                added_vocab_start_index=added_bound,
                added_vocab_end_index=added_bound,
            ),
            quant_method=_FakeQuant(),
        )
        embeds.append(embed)
        masked, mask = sp.vpe.get_masked_input_and_mask(
            input_ids,
            embed.org_vocab_start_index,
            embed.org_vocab_end_index,
            embed.num_org_vocab_padding,
            embed.added_vocab_start_index,
            embed.added_vocab_end_index,
        )
        out = shard_w[masked.long()]
        out.masked_fill_(mask.unsqueeze(-1), 0)
        partials.append(out)

    summed = sum(partials)
    for r, embed in enumerate(embeds):
        with patch.object(
            sp,
            "tensor_model_parallel_reduce_scatter",
            side_effect=lambda t, dim, rank=r: summed[
                rank * rows_local : (rank + 1) * rows_local
            ],
        ) as rs_mock:
            got = sp._embed_reduce_scatter(embed, input_ids)  # noqa: SLF001 -- verifying embed reduce-scatter parity directly
        assert rs_mock.call_count == 1
        assert rs_mock.call_args.args[1] == 0
        torch.testing.assert_close(
            got,
            expected_full[r * rows_local : (r + 1) * rows_local],
            rtol=0,
            atol=0,
        )


def _make_fp8_linear(k: int, n: int, *, fp8: bool = True) -> torch.nn.Module:
    mod = torch.nn.Module()
    mod.weight = torch.nn.Parameter(
        torch.zeros(k, n, dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16),
        requires_grad=False,
    )
    mod.weight_scale = torch.nn.Parameter(torch.ones(n), requires_grad=False)
    mod.bias = None
    setattr(mod, sp.WEIGHT_FLIPPED_ATTR, True)
    return mod


def _make_fake_mla():
    class _FakeMLA:
        kimi_preprocess = None
        q_lora_rank = 1536
        g_proj = None

        def forward(
            self,
            positions,
            hidden_states,
            llama_4_scaling=None,
            prefill_group=None,
        ):
            return hidden_states

    return _FakeMLA()


def test_should_arm_dense_layers_and_embed_rs_when_dense_mlps_are_fp8():
    class _FakeDenseMLP(deepseek_v2.DeepseekV2MLP):
        def __init__(self, *, fp8: bool = True):
            torch.nn.Module.__init__(self)
            self.gate_up_proj = _make_fp8_linear(8, 12, fp8=fp8)
            self.down_proj = _make_fp8_linear(6, 8, fp8=fp8)

    class _FakeMoEMLP(deepseek_v2.DeepseekV2MoE):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.routed_scaling_factor = 1.0
            self.shared_experts = _FakeDenseMLP(fp8=True)
            self.experts = SimpleNamespace(
                routed_experts=SimpleNamespace(
                    routed_scaling_factor=1.0,
                    moe_config=SimpleNamespace(experts_per_token=2),
                    renormalize=True,
                    quant_method=SimpleNamespace(_tpu_activation_str="silu"),
                )
            )

    class _FakeEmbed(sp.vpe.VocabParallelEmbedding):
        def __init__(self, tp_size: int):
            torch.nn.Module.__init__(self)
            self.tp_size = tp_size
            self.use_fused_embedding = False
            self.quant_method = SimpleNamespace()

    class _FakeBackbone(deepseek_v2.DeepseekV2Model):
        def __init__(self, layers, tp_size: int):
            torch.nn.Module.__init__(self)
            self.config = SimpleNamespace(llama_4_scaling=None)
            self.aux_hidden_state_layers = ()
            self.start_layer = 0
            self.end_layer = len(layers)
            self.layers = torch.nn.ModuleList(layers)
            self.embed_tokens = _FakeEmbed(tp_size)

    def _layer(mlp, *, pair_prequant: bool = False):
        m = torch.nn.Module()
        m.mlp = mlp
        m.self_attn = SimpleNamespace(mla_attn=_make_fake_mla())
        if pair_prequant:
            with patch.object(
                type(current_platform),
                "check_and_update_config",
                classmethod(lambda cls, cfg: None),
            ):
                cfg = VllmConfig(device_config=DeviceConfig(device="cpu"))
            with set_current_vllm_config(cfg):
                ln_post = rmsnorm.TpuRMSNorm(8, eps=1e-6)
            slot = PrequantSlot()
            ln_post._prequant_slot = slot  # noqa: SLF001 -- seeding prequant slot as rmsnorm._pair does at load
            setattr(mlp.gate_up_proj, PREQUANT_ATTR, slot)
            m.post_attention_layernorm = ln_post
        return m

    tp = 4
    backbone = _FakeBackbone(
        [_layer(_FakeDenseMLP(fp8=True), pair_prequant=True), _layer(_FakeMoEMLP())],
        tp_size=tp,
    )
    assert backbone.layers[0].post_attention_layernorm.is_prequant_armed() is True
    assert hasattr(backbone.layers[0].mlp.gate_up_proj, PREQUANT_ATTR)
    model = SimpleNamespace(model=backbone)
    fake_rank = torch.zeros((1, 1), dtype=torch.int32)
    fake_tp_group = SimpleNamespace(
        rank_in_group=1,
        all_gather=lambda t, dim: torch.cat([t] * tp, dim=dim),
    )

    def _run_arm(target_model):
        vllm_torchtpu._patch_vllm_vocab_parallel_embedding()  # noqa: SLF001 -- runtime vocab patch gate
        extra = [
            patch.object(
                deepseek_v2.DeepseekV2Model, "forward", new=sp.patched_model_forward
            ),
            patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
            patch.object(sp, "get_tp_group", return_value=fake_tp_group),
            patch.object(torch.distributed, "get_world_size", return_value=tp),
            patch.object(sp, "ep_mesh_index", return_value=1),
            patch.object(
                sp.fused_moe_ep, "build_sequence_parallel_op", return_value="moe_op"
            ),
            patch.object(sp.fused_moe_ep, "rank_operand", return_value=fake_rank),
            patch.object(sp.fused_moe_decode, "rank_operand", return_value=fake_rank),
            patch.object(sp, "_build_shared_op", side_effect=lambda g: f"shared_g{g}"),
        ]
        for p in extra:
            p.start()
        patches = [*_with_envs(TPU_MOE_PREFILL_SP_SHARED_GROUP=4), *extra]
        try:
            return sp.arm(target_model, _config(tensor_parallel_size=tp))
        finally:
            for p in patches:
                p.stop()

    armed = _run_arm(model)
    assert armed == 1
    assert getattr(backbone, sp.EMBED_RS_ATTR) is True
    assert hasattr(backbone.layers[0], sp.KEEP_ATTR)
    assert getattr(backbone.layers[0].mlp, sp.DENSE_OP) == "shared_g2"
    assert getattr(backbone.layers[0].mlp, sp.DENSE_INDEX) == 0
    assert backbone.layers[0].post_attention_layernorm.is_prequant_armed() is False
    assert not hasattr(backbone.layers[0].mlp.gate_up_proj, PREQUANT_ATTR)
    for buf_name in sp.DENSE_STACK:
        assert hasattr(backbone, buf_name)

    backbone_bf16 = _FakeBackbone(
        [_layer(_FakeDenseMLP(fp8=False), pair_prequant=True), _layer(_FakeMoEMLP())],
        tp_size=tp,
    )
    _run_arm(SimpleNamespace(model=backbone_bf16))
    assert getattr(backbone_bf16, sp.EMBED_RS_ATTR) is False
    assert not hasattr(backbone_bf16.layers[0], sp.KEEP_ATTR)
    assert not hasattr(backbone_bf16.layers[0].mlp, sp.DENSE_OP)
    assert backbone_bf16.layers[0].post_attention_layernorm.is_prequant_armed() is True
    assert hasattr(backbone_bf16.layers[0].mlp.gate_up_proj, PREQUANT_ATTR)


def test_should_dispatch_dense_sp_layer_and_embed_rs_in_forward():
    num_tokens, tp, hidden = 16, 4, 8
    rows = num_tokens // tp
    positions = torch.arange(num_tokens, dtype=torch.int32)
    x_sp = torch.randn(rows, hidden, dtype=torch.bfloat16)
    seen_groups = []
    seen_dense = []

    class _FakeNorm:
        def __call__(self, x, residual=None):
            if residual is None:
                return x
            return x + residual, residual

    class _FakeAttn:
        def __call__(self, pos, x, scaling, *, prefill_group):
            seen_groups.append(prefill_group)
            return x

    def _run_dense_op(x, *w):
        seen_dense.append((x.shape, len(w)))
        return x

    dense_stack = (
        torch.zeros((1, hidden, 12), dtype=torch.float8_e4m3fn),
        torch.ones((1, 12)),
        torch.zeros((1, 6, hidden), dtype=torch.float8_e4m3fn),
        torch.ones((1, 2, hidden)),
    )
    mlp = SimpleNamespace()
    setattr(mlp, sp.DENSE_ATTR, dense_stack)
    setattr(mlp, sp.DENSE_INDEX, 0)
    setattr(mlp, sp.DENSE_OP, _run_dense_op)
    layer = SimpleNamespace(
        input_layernorm=_FakeNorm(),
        post_attention_layernorm=_FakeNorm(),
        self_attn=SimpleNamespace(mla_attn=_FakeAttn()),
        mlp=mlp,
    )
    with patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp):
        out, res = sp._sequence_parallel_layer_forward(  # noqa: SLF001 -- verifying dense SP dispatch directly
            layer, positions, x_sp, None, None
        )
    assert seen_groups == [sp._GATHER_PROJECTIONS]  # noqa: SLF001 -- verifying gather-projections hook selected on sharded input
    assert seen_dense == [((rows, hidden), 4)]
    assert tuple(out.shape) == (rows, hidden)
    assert tuple(res.shape) == (rows, hidden)


def test_should_retain_small_all_reduce_buffers_when_embed_rs_skips_all_reduce():
    num_tokens, tp, hidden = 16, 4, 8
    rows = num_tokens // tp
    input_ids = torch.arange(num_tokens, dtype=torch.int64)
    positions = torch.arange(num_tokens, dtype=torch.int32)
    embed_out = torch.randn(rows, hidden, dtype=torch.bfloat16)
    retain_calls = []

    def _fake_retain(x: torch.Tensor) -> torch.Tensor:
        retain_calls.append(tuple(x.shape))
        return x

    def _fake_layer(pos, x, res, scaling):
        return x, x if res is None else res

    def _fake_norm(x, res):
        return x, res

    model = SimpleNamespace(
        embed_tokens=SimpleNamespace(),
        layers=[_fake_layer],
        start_layer=0,
        end_layer=1,
        norm=_fake_norm,
    )
    setattr(model, sp.KEEP_ATTR, True)
    setattr(model, sp.EMBED_RS_ATTR, True)
    with (
        patch.object(sp, "applies", return_value=True),
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
        patch.object(sp, "_embed_reduce_scatter", return_value=embed_out),
        patch.object(sp.small_all_reduce_op, "retain", side_effect=_fake_retain),
        patch.object(
            sp, "_all_rows", side_effect=lambda x, n: x.repeat(n // x.shape[0], 1)
        ),
    ):
        out = sp.patched_model_forward(model, input_ids, positions)
    assert retain_calls == [(rows, hidden)]
    assert tuple(out.shape) == (num_tokens, hidden)


class _CpuOutputCode(OutputCode):
    def __init__(self, out_metas):
        self.out_metas = out_metas

    def prepare_for_serialization(self) -> None:
        pass

    def post_compile(self, example_inputs, constants, graph_kwargs) -> None:
        pass

    def __call__(self, args):
        args.clear()
        return [torch.zeros(shape, dtype=dtype) for shape, dtype in self.out_metas]


class _CpuAotBackend:
    def __call__(self, graph_module: fx.GraphModule, example_inputs):
        def _fw(gm: fx.GraphModule, _inps):
            outs = gm.graph.output_node().args[0]
            metas = [
                None
                if o is None
                else (
                    tuple(
                        int(d)
                        for d in o.meta.get("val", o.meta.get("example_value")).shape
                    ),
                    o.meta.get("val", o.meta.get("example_value")).dtype,
                )
                for o in outs
            ]
            code = _CpuOutputCode(metas)
            code._boxed_call = True  # noqa: SLF001 -- torch aot_autograd boxed call protocol
            return code

        fw_compiler = SerializableAOTDispatchCompiler(
            output_code_ty=_CpuOutputCode, compiler_fn=_fw
        )
        with _serialization_context(example_inputs, enable=True) as captured:
            fn = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=_fw,
                partition_fn=min_cut_rematerialization_partition,
                decompositions=_TPU_DECOMPOSITIONS,
                keep_inference_input_mutations=False,
            )(graph_module, example_inputs)
            if captured is not None and captured[0] is not None:
                entry = captured[0]
                fn.serialize = lambda: entry
            return fn


def _run_handover_prequant_case(disarm: bool) -> list[tuple[int, ...]]:
    @support_torch_compile(dynamic_arg_dims={"input_ids": 0, "positions": 0})
    class _HandoverModel(nn.Module):
        def __init__(self, *, vllm_config: VllmConfig) -> None:
            super().__init__()
            self.embed_w = nn.Parameter(torch.ones(32, 8, dtype=torch.bfloat16))
            self.dense_w = nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16))
            self.register_buffer(
                "dense_stack_w",
                torch.ones(1, 8, 8, dtype=torch.bfloat16),
                persistent=False,
            )
            self.post_attention_layernorm = rmsnorm.TpuRMSNorm(8, eps=1e-6)
            slot = PrequantSlot()
            self.post_attention_layernorm._prequant_slot = slot  # noqa: SLF001 -- seeding prequant slot as rmsnorm._pair does
            self.gate_up_proj = nn.Module()
            setattr(self.gate_up_proj, PREQUANT_ATTR, slot)
            self.mlp = SimpleNamespace(gate_up_proj=self.gate_up_proj)

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            n = positions.shape[0]
            slot = getattr(self.gate_up_proj, PREQUANT_ATTR, None)
            if n <= 8:
                out = torch.nn.functional.embedding(input_ids, self.embed_w)
                z = self.dense_stack_w.reshape(-1)[0]
                out = out + (z - z)
                if slot is not None and slot.value is not None:
                    slot.value = None
                return out @ self.dense_w
            torch._check(n % 4 == 0)  # noqa: SLF001 -- torch symbolic divisibility guard matching moe_prefill_sp._check_divides
            out = torch.nn.functional.embedding(input_ids, self.embed_w)
            out = out.view(4, n // 4, 8)[0]
            z = self.dense_w.reshape(-1)[0]
            out = out + (z - z)
            if self.post_attention_layernorm.is_prequant_armed() and slot is not None:
                slot.value = (
                    out.to(torch.float8_e4m3fn),
                    torch.ones((out.shape[0], 1), dtype=torch.float32),
                )
            out = out @ self.dense_stack_w[0]
            return out.repeat(4, 1).narrow(0, 0, n)

    with tempfile.TemporaryDirectory() as cache_root:
        env_patch = {
            "VLLM_CACHE_ROOT": cache_root,
            "VLLM_DISABLE_COMPILE_CACHE": "0",
            "VLLM_USE_AOT_COMPILE": "1",
            "VLLM_USE_MEGA_AOT_ARTIFACT": "0",
            "TPU_PARALLEL_PRECOMPILE": "1",
            "LOCAL_RANK": "0",
        }
        buckets = [4, 8, 64]
        with patch.dict(os.environ, env_patch):
            with patch.object(
                type(current_platform),
                "check_and_update_config",
                classmethod(lambda cls, cfg: None),
            ):
                config = VllmConfig(
                    device_config=DeviceConfig(device="cpu"),
                    scheduler_config=SchedulerConfig(
                        max_num_batched_tokens=16384,
                        max_model_len=16384,
                        is_encoder_decoder=False,
                    ),
                    compilation_config=CompilationConfig(
                        mode=CompilationMode.VLLM_COMPILE,
                        backend="",
                        compile_sizes=list(buckets),
                        splitting_ops=[],
                        cudagraph_capture_sizes=[],
                    ),
                )
            for field in dataclasses.fields(config.compilation_config.pass_config):
                val = getattr(config.compilation_config.pass_config, field.name)
                if isinstance(val, bool):
                    setattr(config.compilation_config.pass_config, field.name, False)

            vllm_torchtpu._patch_vllm_aot_compile_cache_key()  # noqa: SLF001 -- runtime compile hook setup
            vllm_torchtpu._patch_vllm_compile_all_ranges()  # noqa: SLF001 -- runtime compile hook setup
            vllm_torchtpu._patch_vllm_piecewise_backend()  # noqa: SLF001 -- runtime compile hook setup

            with (
                patch.object(
                    type(config.compilation_config),
                    "get_compile_ranges",
                    lambda self: [],
                ),
                patch.object(
                    tpu_compiler,
                    "compute_tpu_compilation_hash",
                    lambda _cfg: f"n37_{disarm}",
                ),
                patch.object(tpu_compiler, "_tpu_backend", _CpuAotBackend()),
                set_current_vllm_config(config),
            ):
                model = _HandoverModel(vllm_config=config)
                if disarm:
                    rmsnorm.disarm_post_attention_prequant(model)

                with shape_variants.warmup() as tracked:
                    with shape_variants.trace_only() as deferred:
                        for n in buckets:
                            if tracked.needs_retrace(n):
                                shape_variants.retrace(model, config)
                            model(
                                torch.zeros(n, dtype=torch.int64),
                                torch.arange(n, dtype=torch.int32),
                            )
                    deferred.compile()

                return [
                    tuple(
                        model(
                            torch.zeros(n, dtype=torch.int64),
                            torch.arange(n, dtype=torch.int32),
                        ).shape
                    )
                    for n in buckets
                ]


def test_should_keep_single_submod_output_when_dense_sp_disarms_post_norm_prequant():
    with pytest.raises(IndexError, match="list index out of range"):
        _run_handover_prequant_case(disarm=False)

    shapes = _run_handover_prequant_case(disarm=True)
    assert shapes == [(4, 8), (8, 8), (64, 8)]


@pytest.mark.parametrize(
    "first_val,dtype",
    [
        (float("nan"), torch.bfloat16),
        (float("inf"), torch.bfloat16),
        (float("-inf"), torch.bfloat16),
        (-0.0, torch.bfloat16),
        (float("nan"), torch.float32),
    ],
)
def test_should_order_shared_before_routed_and_preserve_rank_for_edge_bit_patterns(
    first_val, dtype
):
    events: list[str] = []
    seen_ranks: list[int] = []
    rank = torch.tensor([[7]], dtype=torch.int32)
    tp = 4
    num_tokens = 1024
    rows = num_tokens // tp
    positions = torch.arange(num_tokens, dtype=torch.int32)
    hidden = torch.randn((rows, 8), dtype=dtype)
    residual = torch.randn((rows, 8), dtype=dtype)
    shared_out = torch.full((rows, 8), 2.5, dtype=dtype)
    shared_out[0, 0] = first_val
    routed_out = torch.full((rows, 8), 1.5, dtype=dtype)

    def fake_shared_op(_x, *_weights):
        events.append("shared")
        return shared_out

    def fake_routed_op(_op, _x, _w13, _w2, _s13, _s2, _logits, r, _bias):
        events.append("routed")
        seen_ranks.append(int(r.item()))
        assert r is not rank
        assert r.shape == rank.shape
        assert r.dtype == rank.dtype
        return routed_out

    routed_experts = SimpleNamespace(
        w13_weight=torch.zeros((1,)),
        w2_weight=torch.zeros((1,)),
        w13_weight_scale_inv=torch.zeros((1,)),
        w2_weight_scale_inv=torch.zeros((1,)),
        e_score_correction_bias=None,
    )
    mlp = SimpleNamespace(
        experts=SimpleNamespace(routed_experts=routed_experts),
        gate=lambda z: (torch.zeros((z.shape[0], 2)), None),
    )
    setattr(mlp, sp.OP_ATTR, object())
    setattr(mlp, sp.SCALE_ATTR, 2.0)
    setattr(mlp, sp.SHARED_OP, fake_shared_op)
    setattr(mlp, sp.SHARED_ATTR, (torch.zeros((1,)),))
    setattr(mlp, sp.SHARED_INDEX, 0)

    layer = SimpleNamespace(
        input_layernorm=lambda h, res=None: (h, res if res is not None else h),
        post_attention_layernorm=lambda h, res: (h, res),
        self_attn=SimpleNamespace(
            mla_attn=lambda _pos, h, _scale, prefill_group=None: h
        ),
        mlp=mlp,
    )
    setattr(layer, sp.KEEP_ATTR, ())

    with (
        patch.object(sp, "get_tensor_model_parallel_world_size", return_value=tp),
        patch.object(sp.fused_moe_ep, "rank_operand", return_value=rank),
        patch.object(
            sp.fused_moe_ep, "run_sequence_parallel_op", side_effect=fake_routed_op
        ),
        patch.object(sp.envs, "TPU_MOE_SKIP_PADDED_TOKENS", False),
    ):
        out, out_res = sp.patched_decoder_layer_forward(
            layer, positions, hidden, residual
        )

    assert events == ["shared", "routed"]
    assert seen_ranks == [7]
    torch.testing.assert_close(out_res, residual)
    torch.testing.assert_close(out[1:], (shared_out + routed_out * 2.0)[1:])
    if torch.isnan(torch.tensor(first_val)):
        assert torch.isnan(out[0, 0])
    else:
        torch.testing.assert_close(out[0, 0], shared_out[0, 0] + routed_out[0, 0] * 2.0)


@pytest.mark.parametrize("with_residual", [False, True])
def test_should_fuse_decode_post_attn_norm_when_eligible(with_residual):
    rows, hidden = 8, 6144
    positions = torch.arange(rows, dtype=torch.int32)
    hidden_in = torch.ones((rows, hidden), dtype=torch.bfloat16) * 2.0
    residual_in = (
        torch.ones((rows, hidden), dtype=torch.bfloat16) * 3.0
        if with_residual
        else None
    )
    norm_w = torch.ones((hidden,), dtype=torch.bfloat16) * 0.5
    seen_post_attn = []

    def fake_mla_attn(_pos, h, _scale, prefill_group=None, post_attn_norm=None):
        assert prefill_group is None and post_attn_norm is not None
        res_arg, w_arg, eps_arg = post_attn_norm
        seen_post_attn.append((res_arg, w_arg, eps_arg))
        return h + res_arg, res_arg * 2.0

    fake_mla_attn.o_proj = object()
    fake_mla_attn.mla_attn = SimpleNamespace(W_O_dcp_qrep=None)

    def input_ln(h, res=None):
        if res is None:
            return h * 1.5
        return h * 1.5, res + h

    post_ln = SimpleNamespace(
        weight=norm_w,
        variance_epsilon=1e-6,
        has_weight=True,
        variance_size_override=None,
    )
    layer = SimpleNamespace(
        use_sequence_parallel_moe=False,
        use_mha=False,
        input_layernorm=input_ln,
        post_attention_layernorm=post_ln,
        self_attn=SimpleNamespace(mla_attn=fake_mla_attn),
        mlp=lambda h: h * 4.0,
    )
    setattr(layer, sp.KEEP_ATTR, ())

    with (
        patch.object(sp, "applies", return_value=False),
        patch.object(sp.envs, "TPU_FUSED_OPROJ_AR", True),
        patch.object(sp.small_all_reduce_op, "can_fuse_rmsnorm", return_value=True),
    ):
        out, out_res = sp.patched_decoder_layer_forward(
            layer, positions, hidden_in, residual_in
        )

    assert len(seen_post_attn) == 1
    res_arg, w_arg, eps_arg = seen_post_attn[0]
    expected_res = hidden_in if not with_residual else residual_in + hidden_in
    torch.testing.assert_close(res_arg, expected_res)
    assert w_arg is norm_w and eps_arg == 1e-6
    torch.testing.assert_close(out, (hidden_in * 1.5 + expected_res) * 4.0)
    torch.testing.assert_close(out_res, expected_res * 2.0)


def test_should_fall_back_to_upstream_decode_when_post_attn_norm_ineligible():
    rows, hidden = 4, 6144
    positions = torch.arange(rows, dtype=torch.int32)
    hidden_in = torch.ones((rows, hidden), dtype=torch.bfloat16)
    residual_in = torch.zeros((rows, hidden), dtype=torch.bfloat16)
    layer = SimpleNamespace(
        use_sequence_parallel_moe=False,
        use_mha=False,
        self_attn=SimpleNamespace(mla_attn=None),
    )
    setattr(layer, sp.KEEP_ATTR, ())
    sentinel = (hidden_in * 2.0, residual_in)
    with (
        patch.object(sp, "applies", return_value=False),
        patch.object(sp.envs, "TPU_FUSED_OPROJ_AR", True),
        patch.object(sp, "UPSTREAM_DECODER_LAYER_FORWARD", return_value=sentinel) as up,
    ):
        got = sp.patched_decoder_layer_forward(layer, positions, hidden_in, residual_in)
    assert got is sentinel and up.call_count == 1


def test_should_route_post_attn_norm_to_fused_oproj_ar():
    cls = mla_op.VllmTPUMultiHeadLatentAttentionWrapper
    w = cls.__new__(cls)
    torch.nn.Module.__init__(w)
    w.kimi_preprocess = None
    w.q_lora_rank = None
    w.kv_lora_rank = 8
    w.qk_nope_head_dim = 4
    w.qk_rope_head_dim = 4
    w.qk_head_dim = 8
    w.num_heads = 2
    w.v_head_dim = 4
    w.dcp_q_replicate = False
    w.dcp_group_size = 1
    w.fused_qkv_a_proj = None
    w.q_b_proj = None
    w.kv_a_proj_with_mqa = lambda x: (
        torch.zeros((x.shape[0], 12), dtype=x.dtype),
        None,
    )
    w.q_proj = lambda x: (torch.zeros((x.shape[0], 16), dtype=x.dtype), None)
    w.q_a_layernorm = lambda x: x
    w.kv_a_layernorm = lambda x: x
    w.rotary_emb = None
    w.is_sparse = False
    w.indexer = None
    w.skip_topk = True
    w.g_proj = None
    w.gate_is_fused = False
    attn_out = torch.ones((8, 8), dtype=torch.bfloat16)

    def _fake_mla(*_a, **_kw):
        return attn_out

    _fake_mla.W_O_dcp_qrep = None
    w.mla_attn = _fake_mla
    w.o_proj = object()
    res_in = torch.ones((8, 16), dtype=torch.bfloat16) * 2.0
    norm_w = torch.ones((16,), dtype=torch.bfloat16) * 0.5
    sentinel = (res_in * 3.0, res_in * 4.0)
    with patch.object(
        mla_op.small_all_reduce_op, "row_parallel_or_rmsnorm", return_value=sentinel
    ) as fused:
        got = cls.forward(
            w,
            positions=torch.arange(8, dtype=torch.int32),
            hidden_states=torch.zeros((8, 16), dtype=torch.bfloat16),
            post_attn_norm=(res_in, norm_w, 1e-6),
        )
    assert got is sentinel
    fused.assert_called_once_with(attn_out, w.o_proj, res_in, norm_w, 1e-6)


@torch.library.custom_op("test_moe_prefill_sp::rs_inplace", mutates_args=("out",))
def _test_rs_inplace(out: torch.Tensor, inp: torch.Tensor) -> None:
    del out, inp


@_test_rs_inplace.register_fake
def _(out: torch.Tensor, inp: torch.Tensor) -> None:
    del out, inp


@torch.library.custom_op("test_moe_prefill_sp::ag_inplace", mutates_args=("out",))
def _test_ag_inplace(out: torch.Tensor, inp: torch.Tensor) -> None:
    del out, inp


@_test_ag_inplace.register_fake
def _(out: torch.Tensor, inp: torch.Tensor) -> None:
    del out, inp


def test_should_concretize_embed_reduce_scatter_empty_shape_under_aot_concrete_args():
    tp, num_tokens, hidden = 4, 1024, 16
    rank = torch.zeros((1, 1), dtype=torch.int32)
    embed_weight = torch.randn(32, hidden, dtype=torch.bfloat16)

    def fake_reduce_scatter(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        input_tensor = tensor.movedim(0, dim).contiguous()
        chunk_size = input_tensor.shape[0] // tp
        out = torch.empty(
            (chunk_size,) + input_tensor.shape[1:],
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        _test_rs_inplace(out, input_tensor)
        return out.movedim(0, dim).contiguous()

    def fake_all_gather(tensor: torch.Tensor, dim: int = 0) -> torch.Tensor:
        sz = tensor.size()
        out = torch.empty(
            (sz[0] * tp,) + sz[1:], dtype=tensor.dtype, device=tensor.device
        )
        _test_ag_inplace(out, tensor)
        return (
            out.reshape((tp,) + sz)
            .movedim(0, dim)
            .reshape(sz[:dim] + (tp * sz[dim],) + sz[dim + 1 :])
        )

    shard_indices = SimpleNamespace(
        org_vocab_start_index=0,
        org_vocab_end_index=32,
        num_org_vocab_padding=0,
        added_vocab_start_index=32,
        added_vocab_end_index=32,
    )
    embed = SimpleNamespace(
        shard_indices=shard_indices,
        org_vocab_start_index=0,
        org_vocab_end_index=32,
        num_org_vocab_padding=0,
        added_vocab_start_index=32,
        added_vocab_end_index=32,
        quant_method=SimpleNamespace(embedding=lambda _e, idx: embed_weight[idx]),
    )

    class _ShardedOnlyLayer(nn.Module):
        def forward(
            self, positions, hidden_states, residual, _scale, gather_group=None
        ):
            pos_local = sp.own_rows(positions, rank, tp)
            return hidden_states + pos_local.to(hidden_states.dtype).unsqueeze(-1), (
                hidden_states if residual is None else residual
            )

    layer = _ShardedOnlyLayer()
    setattr(layer, sp.KEEP_ATTR, ())

    model = SimpleNamespace(
        embed_tokens=embed,
        layers=[layer],
        start_layer=0,
        end_layer=1,
        norm=lambda h, r: (h + r, r),
    )
    setattr(model, sp.KEEP_ATTR, True)
    setattr(model, sp.EMBED_RS_ATTR, True)

    captured: list[fx.GraphModule] = []

    def capture_backend(gm: fx.GraphModule, _example_inputs):
        captured.append(gm)
        return gm

    def run_model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return sp.patched_model_forward(model, input_ids, positions)

    with (
        patch.object(sp, "applies", new=lambda _n: True),
        patch.object(sp, "get_tensor_model_parallel_world_size", new=lambda: tp),
        patch.object(
            sp, "tensor_model_parallel_reduce_scatter", new=fake_reduce_scatter
        ),
        patch.object(sp, "tensor_model_parallel_all_gather", new=fake_all_gather),
        patch.object(sp.small_all_reduce_op, "retain", new=lambda h: h),
        patch.object(sp.envs, "TPU_DCP_TP_SHARDED_KV_WRITE", False),
    ):
        compiled = torch.compile(run_model, backend=capture_backend, fullgraph=True)
        input_ids = torch.zeros(num_tokens, dtype=torch.long)
        positions = torch.zeros(num_tokens, dtype=torch.long)
        torch._dynamo.mark_dynamic(input_ids, 0)  # noqa: SLF001 -- mirrors vLLM @support_torch_compile dynamic dim mark
        torch._dynamo.mark_dynamic(positions, 0)  # noqa: SLF001 -- mirrors vLLM @support_torch_compile dynamic dim mark
        compiled(input_ids, positions)

    assert len(captured) == 1
    gm = captured[0]
    concrete_args = create_concrete_args(gm, num_tokens)
    fake_mode = next(
        (t.fake_mode for t in concrete_args if isinstance(t, FakeTensor)), None
    )
    assert fake_mode is not None
    tracing_ctx = torch._guards.TracingContext(fake_mode)  # noqa: SLF001 -- mirrors TpuCompilerAdaptor._run_backend

    empty_shape_args: list[list[object]] = []

    def inspect_fw(fw_gm: fx.GraphModule, _fw_args):
        for node in fw_gm.graph.nodes:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.empty.memory_format
            ):
                empty_shape_args.append(list(node.args[0]))
        return fw_gm

    with torch._guards.tracing(tracing_ctx):  # noqa: SLF001 -- mirrors TpuCompilerAdaptor._run_backend
        aot_module_simplified(gm, concrete_args, fw_compiler=inspect_fw)

    assert empty_shape_args, "Expected aten.empty.memory_format nodes in AOT graph"
    for shape in empty_shape_args:
        assert all(isinstance(dim, int) for dim in shape), (
            f"Symbolic dim leaked into aten.empty shape: {shape!r}"
        )
