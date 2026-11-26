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
"""Unit tests for GLM-5 DSA MoE dispatch, EPxTP split, and SP prefill."""

import dataclasses
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from vllm.model_executor.models import deepseek_v2

import vllm_torchtpu.envs as envs
from vllm_torchtpu import tpu_info
from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import (
    grouped_shared_expert_sharded,
)
from vllm_torchtpu.layers.adapter import (
    fused_moe_ep,
    moe_hierarchical,
    moe_prefill_sp,
    moe_routing,
    moe_tp_decode,
    token_padding,
)
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.models.vllm import glm_dsa_dispatch, glm_dsa_patch


class _NoopLog:
    info = info_once = warning = warning_once = debug = staticmethod(
        lambda *_a, **_kw: None
    )


@pytest.fixture(autouse=True)
def _reset_glm_dispatch_state(monkeypatch):
    for mod, attr, val in (
        (glm_dsa_dispatch, "_ACTIVE_ARCH", None),
        (glm_dsa_dispatch, "_ACTIVE_MAX_MODEL_LEN", 0),
        (tpu_info, "_v7x_model_fastpaths", False),
        (moe_routing, "_ROUTER_TOPK_BIAS", False),
        (moe_tp_decode, "_SPECULATIVE_ACTIVE", False),
        (envs, "TPU_MOE_HIERARCHICAL_EP", False),
        (envs, "MOE_FUSED_EP_SORTED_GATHER", False),
        (moe_hierarchical, "logger", _NoopLog()),
        (moe_prefill_sp, "logger", _NoopLog()),
    ):
        monkeypatch.setattr(mod, attr, val)


def _vllm_cfg(arch: str = "GlmMoeDsaForCausalLM", **ov):
    d = dict(tp=16, dcp=16, ep=True, dp=1, pp=1, pcp=1, eplb=False, sp_moe=False)
    d |= dict(
        max_model_len=163_840, speculative_config=None, gpu_memory_utilization=0.9
    )
    d |= ov
    mcfg = SimpleNamespace(
        architectures=[arch],
        max_model_len=d["max_model_len"],
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(architectures=[arch]),
    )
    pcfg = SimpleNamespace(
        tensor_parallel_size=d["tp"],
        decode_context_parallel_size=d["dcp"],
        enable_expert_parallel=d["ep"],
        data_parallel_size=d["dp"],
        pipeline_parallel_size=d["pp"],
        prefill_context_parallel_size=d["pcp"],
        enable_eplb=d["eplb"],
        use_sequence_parallel_moe=d["sp_moe"],
    )
    ccfg = SimpleNamespace(gpu_memory_utilization=d["gpu_memory_utilization"])
    return SimpleNamespace(
        model_config=mcfg,
        parallel_config=pcfg,
        speculative_config=d["speculative_config"],
        cache_config=ccfg,
    )


def test_ep_tp_size_activates_strictly_for_rank_ordered_glm_dsa_on_v7x(monkeypatch):
    monkeypatch.setattr(glm_dsa_dispatch, "is_tpu_v7x", lambda: True)
    assert moe_hierarchical.ep_tp_size(_vllm_cfg(dcp=16)) == 2

    dcp1_cfg = _vllm_cfg(dcp=1)
    assert moe_hierarchical.ep_tp_size(dcp1_cfg) == 1
    assert moe_prefill_sp._config_refusal_reasons(dcp1_cfg)

    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", True)
    assert moe_hierarchical.ep_tp_size(
        dcp1_cfg
    ) == 2 and not moe_prefill_sp._config_refusal_reasons(dcp1_cfg)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", False)

    for arch in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Qwen3MoeForCausalLM",
    ):
        assert moe_hierarchical.ep_tp_size(_vllm_cfg(arch)) == 1
    for kw in (
        dict(tp=2),
        dict(ep=False),
        dict(dp=2),
        dict(pcp=2),
        dict(eplb=True),
        dict(sp_moe=True),
    ):
        assert moe_hierarchical.ep_tp_size(_vllm_cfg(**kw)) == 1

    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", True)
    assert moe_hierarchical.ep_tp_size(_vllm_cfg()) == 1
    monkeypatch.setattr(envs, "TPU_MOE_HIERARCHICAL_EP", False)
    monkeypatch.setattr(glm_dsa_dispatch, "is_tpu_v7x", lambda: False)
    assert moe_hierarchical.ep_tp_size(_vllm_cfg()) == 1


def test_ep_tp_split_tiles_all_256_glm5_experts_across_8_pairs():
    world, factor, total_experts = 16, 2, 256
    seen_slices = set()
    for r in range(world):
        split = moe_hierarchical.ep_tp_split(world, r, factor)
        assert split == (8, r // 2, 2, r % 2)
        seen_slices.update(
            (e, r % 2) for e in range((r // 2) * 32, ((r // 2) + 1) * 32)
        )
    assert len(seen_slices) == total_experts * factor
    assert (
        moe_hierarchical.ep_tp_split(16, 0, 1) is None
        and moe_hierarchical.ep_tp_split(16, 0, 6) is None
    )


def test_ep_tp_config_and_split_parallel_config():
    @dataclasses.dataclass(frozen=True)
    class _FakeMoEPar:
        use_ep: bool = True
        dp_size: int = 1
        pcp_size: int = 1
        sp_size: int = 1
        tp_size: int = 1
        ep_size: int = 16
        tp_rank: int = 0
        ep_rank: int = 7
        enable_eplb: bool = False
        is_sequence_parallel: bool = False

    good = _FakeMoEPar()
    moe_hierarchical.check_ep_tp_config(good, (8, 3, 2, 1))
    rw = moe_hierarchical._split_parallel_config(good, (8, 3, 2, 1))
    assert (rw.ep_size, rw.ep_rank, rw.tp_size, rw.tp_rank) == (8, 3, 2, 1)
    no_ep = dataclasses.replace(good, use_ep=False)
    assert moe_hierarchical._split_parallel_config(no_ep, (8, 3, 2, 1)) == no_ep
    for kw, match in (
        (dict(dp_size=2), "supports DP=1"),
        (dict(enable_eplb=True), "does not support EPLB"),
        (dict(tp_size=2, ep_size=8), "expects flat EP"),
    ):
        with pytest.raises(ValueError, match=match):
            moe_hierarchical.check_ep_tp_config(
                dataclasses.replace(good, **kw), (8, 0, 2, 0)
            )


def test_glm_dsa_patch_safe_at_platform_activation_and_model_load(monkeypatch):
    import vllm.config as vllm_config_mod
    from jax.experimental.pallas import tpu as pltpu

    def _boom():
        raise AssertionError("unexpected get_tpu_info")

    monkeypatch.setattr(pltpu, "get_tpu_info", _boom)
    monkeypatch.setattr(tpu_info, "is_tpu_v7x", lambda: False)
    assert glm_dsa_patch.maybe_patch_glm_dsa(None) is False
    assert not glm_dsa_dispatch.is_glm_dsa_model()
    for arch in (
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "KimiK25ForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
    ):
        cfg = SimpleNamespace(architectures=[arch], max_model_len=131_072)
        assert glm_dsa_patch.maybe_patch_glm_dsa(cfg) is False
        assert glm_dsa_dispatch._ACTIVE_ARCH is None

    hf = SimpleNamespace(model_type="glm_moe_dsa")
    glm_by_type = SimpleNamespace(architectures=[], max_model_len=202_752, hf_config=hf)
    assert glm_dsa_patch.maybe_patch_glm_dsa(glm_by_type) is False
    assert not tpu_info._v7x_model_fastpaths

    monkeypatch.setattr(tpu_info, "is_tpu_v7x", lambda: True)
    monkeypatch.setattr(
        vllm_config_mod, "get_current_vllm_config_or_none", lambda: None
    )
    assert glm_dsa_patch.maybe_patch_glm_dsa(glm_by_type) is None
    assert (
        tpu_info.v7x_model_fastpaths_enabled() and moe_routing.is_rowmax_bias_enabled()
    )
    assert (
        glm_dsa_dispatch.is_glm_dsa_model()
        and glm_dsa_dispatch._ACTIVE_MAX_MODEL_LEN == 202_752
    )


def test_select_sp_shared_group_hbm_and_context_autosizing():
    gib = 1024**3
    for args, expected in (
        ((16, 163_840, 40 * gib), 1),
        ((16, 262_144, 30 * gib), 1),
        ((16, 163_840, None), 1),
        ((16, 524_288, 40 * gib), 4),
        ((16, 1_048_576, 40 * gib), 4),
        ((16, 1_048_576, None), 4),
        ((16, 163_840, 20 * gib), 4),
        ((6, 1_048_576, 20 * gib), 2),
        ((1, 1_048_576, 10 * gib), 1),
    ):
        assert glm_dsa_dispatch.select_sp_shared_group(*args) == expected


def test_grouped_shared_expert_sharded_rejects_non_divisor_group():
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("tp",))
    for bad in (2, 0):
        with pytest.raises(ValueError, match="do not split into groups"):
            grouped_shared_expert_sharded(mesh, "tp", bad)


@pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
def test_moe_prefill_sp_merge_and_member_slice_equivalence(group):
    torch.manual_seed(0)
    tp, hidden, fs = 16, 64, 8
    sh_w13 = [torch.randn(hidden, 2 * fs) for _ in range(tp)]
    sh_s13 = [torch.rand(2 * fs) + 0.25 for _ in range(tp)]
    sh_w2 = [torch.randn(fs, hidden) for _ in range(tp)]
    sh_s2 = [torch.rand(hidden) + 0.25 for _ in range(tp)]

    w13, s13 = torch.cat(sh_w13, dim=1), torch.cat(sh_s13, dim=0)
    merged_w13, merged_s13 = moe_prefill_sp.merge_gate_up(w13, s13, tp)
    assert merged_w13.shape == w13.shape and merged_s13.shape == s13.shape
    torch.testing.assert_close(
        merged_w13[:, : tp * fs], torch.cat([w[:, :fs] for w in sh_w13], dim=1)
    )
    torch.testing.assert_close(
        merged_w13[:, tp * fs :], torch.cat([w[:, fs:] for w in sh_w13], dim=1)
    )
    w2, s2 = torch.cat(sh_w2, dim=0), torch.stack(sh_s2, dim=0)

    x = torch.randn(12, hidden)
    ref, reconstructed = torch.zeros(12, hidden), torch.zeros(12, hidden)
    silu = torch.nn.functional.silu
    for r in range(tp):
        gu = (x @ sh_w13[r]) * sh_s13[r]
        ref = ref + ((silu(gu[:, :fs]) * gu[:, fs:]) @ sh_w2[r]) * sh_s2[r]
    for member in range(group):
        mw13, ms13, mw2, ms2 = moe_prefill_sp.member_slice(
            w13, s13, w2, s2, tp=tp, group=group, member=member
        )
        half, blocks = mw13.shape[1] // 2, ms2.shape[0]
        assert blocks == tp // group
        gu = (x @ mw13) * ms13
        act, kb = silu(gu[:, :half]) * gu[:, half:], mw2.shape[0] // blocks
        for b in range(blocks):
            sl = slice(b * kb, (b + 1) * kb)
            reconstructed = reconstructed + (act[:, sl] @ mw2[sl]) * ms2[b]
    torch.testing.assert_close(reconstructed, ref, rtol=1e-5, atol=1e-5)


def test_query_min_free_hbm_never_returns_uncoordinated_local_free(monkeypatch):
    gib, mp, query = (
        1024**3,
        monkeypatch.setattr,
        moe_prefill_sp._query_min_free_hbm_per_core,
    )
    mp(torch.accelerator, "get_memory_info", lambda: (40 * gib, 80 * gib))
    mp(
        moe_prefill_sp,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=1, cpu_group=None),
    )
    assert query(_vllm_cfg(gpu_memory_utilization=0.0)) == 0

    cfg = _vllm_cfg(gpu_memory_utilization=1.0)
    mp(
        moe_prefill_sp,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, cpu_group=None),
    )
    assert query(cfg) == 0

    def _timeout(*_a, **_kw):
        raise RuntimeError("gloo timeout")

    mp(
        moe_prefill_sp,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=4, cpu_group=object()),
    )
    mp(torch.distributed, "all_gather_object", _timeout)
    assert query(cfg) == 0

    def _gather_with_none(seen, local_free, group=None):
        del local_free, group
        seen[:] = [40 * gib, None, 35 * gib, 38 * gib]

    mp(torch.distributed, "all_gather_object", _gather_with_none)
    assert query(cfg) == 0


def test_sp_streaming_buffers_padded_mask_and_instance_forwards(monkeypatch):
    mp, sn = monkeypatch.setattr, SimpleNamespace
    assert deepseek_v2.DeepseekV2Model.forward is moe_prefill_sp.UPSTREAM_MODEL_FORWARD
    assert (
        deepseek_v2.DeepseekV2DecoderLayer.forward
        is moe_prefill_sp.UPSTREAM_DECODER_LAYER_FORWARD
    )
    backbone, tp, group, hidden, fs = torch.nn.Module(), 4, 2, 32, 8

    def _lin(rows: int, cols: int):
        w = torch.randn(rows, cols).to(torch.float8_e4m3fn)
        ns = sn(weight=w, weight_scale=torch.rand(cols) + 0.1, bias=None)
        setattr(ns, WEIGHT_FLIPPED_ATTR, True)
        return ns

    mlps = [
        sn(
            shared_experts=sn(
                gate_up_proj=_lin(hidden, 2 * fs), down_proj=_lin(fs, hidden)
            )
        )
        for _ in range(2)
    ]
    mp(
        moe_prefill_sp,
        "_gather",
        lambda t, dim: torch.cat([t.to(torch.bfloat16)] * tp, dim=dim),
    )
    stacked = moe_prefill_sp._allocate_mlp_buffers(
        backbone,
        mlps[0].shared_experts,
        len(mlps),
        tp,
        group,
        moe_prefill_sp.SHARED_STACK,
    )
    assert stacked[0].dtype == stacked[2].dtype == torch.float8_e4m3fn
    for idx, mlp in enumerate(mlps):
        moe_prefill_sp._populate_mlp_layer(
            stacked,
            idx,
            mlp,
            mlp.shared_experts,
            tp,
            1,
            group,
            attr=moe_prefill_sp.SHARED_ATTR,
            index_attr=moe_prefill_sp.SHARED_INDEX,
        )
        assert getattr(mlp, moe_prefill_sp.SHARED_INDEX) == idx

    pad = token_padding.TokenPaddingState.create(8, torch.device("cpu"))
    pad.update(5, 8)
    seen_logits: list[torch.Tensor] = []

    def _fake_sp(_o, x, *a):
        seen_logits.append(a[4])
        return torch.zeros_like(x)

    for mod, attr, val in (
        (envs, "TPU_MOE_SKIP_PADDED_TOKENS", False),
        (glm_dsa_dispatch, "_ACTIVE_ARCH", "GlmMoeDsaForCausalLM"),
        (glm_dsa_dispatch, "is_tpu_v7x", lambda: True),
        (token_padding, "_padding_state", pad),
        (fused_moe_ep, "rank_operand", lambda: torch.tensor([[1]], dtype=torch.int32)),
        (fused_moe_ep, "run_sequence_parallel_op", _fake_sp),
        (moe_prefill_sp, "_shared_expert", lambda _m, x: torch.zeros_like(x)),
    ):
        mp(mod, attr, val)
    z = torch.zeros(1)
    rexp = sn(
        w13_weight=z,
        w2_weight=z,
        w13_weight_scale_inv=z,
        w2_weight_scale_inv=z,
        e_score_correction_bias=None,
    )
    fake_mlp = sn(
        experts=sn(routed_experts=rexp),
        gate=lambda x: (torch.ones(x.shape[0], 4), None),
        _tpu_sp_moe_op="op",
        _tpu_sp_routed_scale=1.0,
    )
    moe_prefill_sp._moe(fake_mlp, torch.ones(4, 8), num_tokens=8, tp=2)
    assert (
        not torch.isnan(seen_logits[0][0]).any()
        and torch.isnan(seen_logits[0][1:]).all()
    )

    layer_a, layer_b = torch.nn.Module(), torch.nn.Module()
    moe_prefill_sp._bind_instance_forwards(backbone, [layer_a], _vllm_cfg())
    assert backbone.forward.__func__ is moe_prefill_sp.patched_model_forward
    assert layer_a.forward.__func__ is moe_prefill_sp.patched_decoder_layer_forward
    assert "forward" not in layer_b.__dict__
    assert deepseek_v2.DeepseekV2Model.forward is moe_prefill_sp.UPSTREAM_MODEL_FORWARD


def test_prebuild_fused_moe_ep_accounts_for_ep_tp_size_and_mesh_order(monkeypatch):
    mp, captured = monkeypatch.setattr, []
    ep_mesh = SimpleNamespace(shape={fused_moe_ep.EP_AXIS_NAME: 16})
    rep_grps = tuple((i,) for i in range(16))
    for mod, attr, val in (
        (envs, "USE_MOE_FUSED_EP_KERNEL", True),
        (envs, "MOE_FUSED_EP_KERNEL_MIN_TOKENS", 64),
        (fused_moe_ep, "logger", _NoopLog()),
        (fused_moe_ep, "build_ep_mesh", lambda: ep_mesh),
        (fused_moe_ep, "ep_mesh_index", lambda: 0),
        (fused_moe_ep, "_max_node_tokens", lambda *_a: 4096),
        (fused_moe_ep, "ep_token_replica_groups", lambda **_kw: rep_grps),
        (
            fused_moe_ep,
            "_build_op",
            lambda *_a, **kw: captured.append(kw["ep_tp_size"]) or "built_op",
        ),
    ):
        mp(mod, attr, val)
    par = SimpleNamespace(
        use_ep=True, tp_size=2, pcp_size=1, is_sequence_parallel=False
    )
    fp8, f32 = torch.float8_e4m3fn, torch.float32
    layer_kw = dict(
        custom_routing_function=None, scoring_func="sigmoid", routed_scaling_factor=1.0
    )
    layer_kw |= dict(use_grouped_topk=False, num_expert_group=1, topk_group=1)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(moe_parallel_config=par, experts_per_token=8),
        global_num_experts=256,
        w13_weight=torch.zeros((32, 128, 256), dtype=fp8),
        w2_weight=torch.zeros((32, 128, 128), dtype=fp8),
        w13_weight_scale_inv=torch.ones((32, 1, 1, 256), dtype=f32),
        w2_weight_scale_inv=torch.ones((32, 1, 1, 128), dtype=f32),
        e_score_correction_bias=torch.zeros(256, dtype=f32),
        **layer_kw,
    )
    bad_order = (0, 1, 4, 5, 6, 7, 2, 3, 8, 9, 12, 13, 14, 15, 10, 11)
    mp(fused_moe_ep, "ep_rank_order", lambda: bad_order)
    assert fused_moe_ep.prebuild_fused_moe_ep(layer, 8, True, "silu") is None
    mp(fused_moe_ep, "ep_rank_order", lambda: tuple(range(16)))
    assert fused_moe_ep.prebuild_fused_moe_ep(layer, 8, True, "silu") == "built_op"
    assert captured == [2]
