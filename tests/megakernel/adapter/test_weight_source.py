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
"""Model-free tests of `megakernel.adapter.weight_source` / `weight_collector`.

Every vLLM weight load runs on `SynthHybridVLLM`, a synthetic hybrid
(`GDN` + `GQA` + `MoE`) `ModelDef` built from library module bindings only,
and on a synthetic vLLM-layout PyTorch backbone that follows those bindings'
`VLLMAlias` sources. No model folder is imported.
"""

from __future__ import annotations

import ast
import dataclasses
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import pytest
import torch

import vllm_torchtpu.megakernel.adapter.exported_bridge as bridge_mod
import vllm_torchtpu.megakernel.adapter.weight_collector as wc_mod
import vllm_torchtpu.megakernel.adapter.weight_source as ws_mod
import vllm_torchtpu.megakernel.core as mk
from tests.megakernel.core.megakernel_fakes import TINY_TILES
from vllm_torchtpu.megakernel.adapter.weight_collector import (
    WEIGHT_FLIPPED_ATTR,
    _orient_2d_kn,
    _own,
    _stack,
    find_backbone,
    load_vllm_weights,
    reshard_layer_experts_ep_to_tp,
)
from vllm_torchtpu.megakernel.adapter.weight_source import VLLMWeightSource
from vllm_torchtpu.megakernel.core.decls import ResolvedModuleGeometry
from vllm_torchtpu.megakernel.core.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.core.geometry import ModelGeometry
from vllm_torchtpu.megakernel.core.model_def import (
    Every,
    FromHF,
    LMHead,
    ModelDef,
    Schedule,
)
from vllm_torchtpu.megakernel.core.modules import GDN, GQA, Embedding, MoE, Norm
from vllm_torchtpu.megakernel.core.modules.gdn import GDNModule
from vllm_torchtpu.megakernel.core.modules.norm import EmbeddingModule
from vllm_torchtpu.megakernel.core.pool.config import CacheExec


class SynthHybridVLLM(ModelDef):
    """Synthetic hybrid `ModelDef`: `GDN`/`GQA` alternating (`att_every=2`) + `MoE`."""

    default_hf_config = {
        "model_type": "synth_hybrid",
        "architectures": ["SynthHybridForCausalLM"],
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "vocab_size": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "num_experts": 128,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
    }
    geometry = FromHF(
        hidden="hidden_size",
        layers="num_hidden_layers",
        vocab="vocab_size",
        eps="rms_norm_eps",
        rope_theta="rope_theta",
        rope_partial=("partial_rotary_factor", 1.0),
    )
    attn = GQA.bind(
        prefix="self_attn",
        heads="num_attention_heads",
        kv_heads="num_key_value_heads",
        head_dim="head_dim",
        qk_norm=True,
        output_gate=True,
    )
    linear = GDN.bind(
        prefix="linear_attn",
        k_heads="linear_num_key_heads",
        v_heads="linear_num_value_heads",
        k_dim="linear_key_head_dim",
        v_dim="linear_value_head_dim",
        conv_kernel="linear_conv_kernel_dim",
    )
    moe = MoE.bind(
        prefix="mlp",
        experts="num_experts",
        top_k="num_experts_per_tok",
        inter="moe_intermediate_size",
        shared_inter="shared_expert_intermediate_size",
        shared_gate=True,
    )
    schedule = Schedule(
        mixer=Every(n=2, last=attn, rest=linear),
        ffn=moe,
        pre_norm=Norm.bind("input_layernorm", zero_centered=True),
        post_norm=Norm.bind("post_attention_layernorm", zero_centered=True),
    )
    embed = Embedding("model.embed_tokens")
    final_norm = Norm.bind("model.norm", zero_centered=True)
    lm_head = LMHead("lm_head", tied_to=None)


def _ep_geom() -> ModelGeometry:
    """`SynthHybridVLLM` geometry with EP experts and `e_slots=68` (`ER=64`)."""
    return SynthHybridVLLM.geometry(moe_tp=False, e_slots=68)


def _exec(*, stack_e: bool = False) -> ExecSpec:
    """`R=2` paged-KV `ExecSpec` on tiny tiles (512 KiB banks, `kv_blk == ctx`)."""
    return ExecSpec(
        R=2,
        B=4,
        T=1,
        ctx=256,
        stack_e=stack_e,
        tiles=replace(TINY_TILES, bank=512 << 10).with_modules(
            {"gqa": {"kv_blk": 256}}
        ),
        interpret=True,
        cache=CacheExec(paged_kv=True, num_pages=64, max_pages_per_seq=2),
    )


def _param(t: torch.Tensor) -> torch.nn.Parameter:
    return torch.nn.Parameter(t, requires_grad=False)


def _fp8(*shape: int) -> torch.Tensor:
    return torch.randint(1, 120, shape, dtype=torch.uint8).view(torch.float8_e4m3fn)


def _vec(n: int, dtype: torch.dtype = torch.bfloat16) -> SimpleNamespace:
    return SimpleNamespace(weight=_param(torch.randn(n, dtype=dtype)))


def _fp8_linear(in_dim: int, out_dim: int, *, flipped: bool = True) -> SimpleNamespace:
    """A vLLM FP8 linear: `[K, N]` when `flipped`, else `[N, K]`; `[N, 1]` scale."""
    w = _fp8(in_dim, out_dim) if flipped else _fp8(out_dim, in_dim)
    s = torch.rand((out_dim, 1), dtype=torch.float32) + 0.1
    ns = SimpleNamespace(weight=_param(w), weight_scale=_param(s))
    setattr(ns, WEIGHT_FLIPPED_ATTR, flipped)
    return ns


def _synth_backbone(
    g: ModelGeometry,
    ex: ExecSpec,
    *,
    fused_mlp: bool = True,
    fused_qkv: bool = False,
    e_cores: list[list[list[int]]] | None = None,
) -> SimpleNamespace:
    """A vLLM-layout decoder backbone for `SynthHybridVLLM` at geometry `g`.

    Attribute paths are the `VLLMAlias` sources of the library `GQA`, `GDN`,
    `MoE` and `Norm` modules (fused or split QKV, fused or split router and
    shared expert); routed experts are FP8 `[EW, H, 2*IR]` / `[EW, IR, H]`.
    """
    R = ex.R
    gdn_lay = GDNModule.lin_in_layout(g, ex)
    nh, nai = g.nh(R), g.NAI(R)
    q_dim = 2 * nh * g.HD
    kv_dim = (nai - q_dim) // 2
    ew, ir, f_r = g.EW(R), g.IR(R), g.ISR(R)
    layers = []
    for l_idx in range(g.L):
        is_att = (l_idx % g.att_every) == (g.att_every - 1)
        self_attn = linear_attn = None
        if is_att:
            norms = dict(q_norm=_vec(g.HD), k_norm=_vec(g.HD))
            o_proj = _fp8_linear(nh * g.HD, g.H)
            if fused_qkv:
                qkv = dict(qkv_proj=_fp8_linear(g.H, nai))
            else:
                qkv = dict(
                    q_proj=_fp8_linear(g.H, q_dim),
                    k_proj=_fp8_linear(g.H, kv_dim, flipped=False),
                    v_proj=_fp8_linear(g.H, kv_dim),
                )
            self_attn = SimpleNamespace(**qkv, o_proj=o_proj, **norms)
        else:
            ba = torch.randn((2 * gdn_lay.vh, g.H), dtype=torch.bfloat16) * 0.05
            linear_attn = SimpleNamespace(
                in_proj_qkvz=_fp8_linear(g.H, gdn_lay.QKV + gdn_lay.Z),
                in_proj_ba=SimpleNamespace(weight=_param(ba)),
                out_proj=_fp8_linear(gdn_lay.Z, g.H),
                conv1d=SimpleNamespace(
                    weight=_param(
                        torch.randn((gdn_lay.QKV, 1, 4), dtype=torch.bfloat16)
                    )
                ),
                A_log=_param(torch.randn(gdn_lay.vh, dtype=torch.float32)),
                dt_bias=_param(torch.randn(gdn_lay.vh, dtype=torch.bfloat16)),
                norm=_vec(gdn_lay.VD, torch.float32),
            )

        experts = SimpleNamespace(
            w13_weight=_param(_fp8(ew, g.H, 2 * ir)),
            w2_weight=_param(_fp8(ew, ir, g.H)),
            w13_weight_scale=_param(torch.rand((ew, 1, 2 * ir)) + 0.1),
            w2_weight_scale=_param(torch.rand((ew, 1, g.H)) + 0.1),
        )
        if e_cores is not None:
            experts._ecopy_cores = e_cores[l_idx]
        if fused_mlp:
            gu_cols = g.E + 2 * g.IS
            mlp = SimpleNamespace(
                experts=experts,
                _fused_gate_gu_fp8_weight=_fp8(g.H, gu_cols),
                _fused_gate_gu_fp8_scale=torch.rand((1, gu_cols)) + 0.1,
                _fused_se_down_fp8_weight=_fp8(g.IS, g.H),
                _fused_se_down_fp8_scale=torch.rand((1, g.H)) + 0.1,
                _fused_se_gate_weight=torch.randn((1, g.H), dtype=torch.bfloat16),
            )
        else:
            mlp = SimpleNamespace(
                experts=experts,
                gate=SimpleNamespace(
                    weight=_param(torch.randn((g.E, g.H), dtype=torch.bfloat16))
                ),
                shared_expert=SimpleNamespace(
                    gate_up_proj=_fp8_linear(g.H, 2 * f_r),
                    down_proj=_fp8_linear(f_r, g.H, flipped=False),
                ),
                shared_expert_gate=SimpleNamespace(
                    weight=_param(torch.randn((1, g.H), dtype=torch.bfloat16))
                ),
            )
        layers.append(
            SimpleNamespace(
                layer_type="full_attention" if is_att else "linear_attention",
                input_layernorm=_vec(g.H),
                post_attention_layernorm=_vec(g.H),
                linear_attn=linear_attn,
                self_attn=self_attn,
                mlp=mlp,
            )
        )
    embed = torch.nn.Embedding(64, g.H, dtype=torch.bfloat16)
    return SimpleNamespace(embed_tokens=embed, layers=layers, norm=_vec(g.H))


def test_synth_backbone_matches_model_def_schedule() -> None:
    """The synthetic geometry is a 2-attention / 2-`GDN` hybrid with EP slots."""
    g = _ep_geom()
    assert (g.L, g.att_every, g.NATT, g.NGDN) == (4, 2, 2, 2)
    assert (g.ER(2), g.EW(2), g.moe_tp) == (64, 68, False)
    bb = _synth_backbone(g, _exec())
    assert [lyr.self_attn is not None for lyr in bb.layers] == [
        False,
        True,
        False,
        True,
    ]


def test_own_and_stack_allocate_base_buffers() -> None:
    """Verify `_own` and `_stack` return non-view base buffers without extra copy."""
    base = torch.arange(32, dtype=torch.float32).reshape(4, 8).clone()
    assert _own(base).data_ptr() == base.data_ptr()

    sliced = base[1:3, 2:6]
    assert sliced._is_view() or sliced.storage_offset() != 0

    owned = _own(sliced)
    assert not owned._is_view()
    assert owned.storage_offset() == 0
    assert torch.equal(owned, sliced)

    stacked = _stack([sliced, sliced])
    assert not stacked._is_view()
    assert stacked.storage_offset() == 0
    assert stacked.shape == (2, 2, 4)


def test_orient_2d_kn_square_orientation() -> None:
    """Verify square `[D, D]` weights respect `WEIGHT_FLIPPED_ATTR`."""
    d = 64
    w_raw = (
        torch.arange(d * d, dtype=torch.int32)
        .remainder(120)
        .add(1)
        .to(torch.uint8)
        .reshape(d, d)
        .view(torch.float8_e4m3fn)
    )
    w_fl = _orient_2d_kn(w_raw.clone(), True, "sq", {d}, {d})
    assert torch.equal(w_fl.view(torch.uint8), w_raw.view(torch.uint8))

    w_unfl = _orient_2d_kn(w_raw.clone(), False, "sq", {d}, {d})
    assert torch.equal(
        w_unfl.view(torch.uint8), w_raw.transpose(0, 1).contiguous().view(torch.uint8)
    )

    with pytest.raises(ValueError, match="Ambiguous square weight shape"):
        _orient_2d_kn(w_raw.clone(), None, "sq", {d}, {d})


def test_find_backbone_wrappers() -> None:
    """Verify `find_backbone` locates `(layers, norm)` across nested vLLM wrappers."""
    inner = SimpleNamespace(layers=[1], norm=object())
    for wrapped in (
        inner,
        SimpleNamespace(model=inner),
        SimpleNamespace(model=SimpleNamespace(model=inner)),
        SimpleNamespace(language_model=SimpleNamespace(model=inner)),
        SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(model=inner))
        ),
    ):
        assert find_backbone(wrapped) is inner

    with pytest.raises(ValueError, match="Cannot locate decoder backbone"):
        find_backbone(SimpleNamespace(foo=1))


def test_reshard_layer_experts_ep_to_tp() -> None:
    """Verify `reshard_layer_experts_ep_to_tp` slices experts and updates MoE flags."""
    g = SynthHybridVLLM.geometry(moe_tp=True, e_slots=0)
    ex_ep = ExecSpec(
        R=2, B=4, T=1, ctx=256, stack_e=False, tiles=TINY_TILES, interpret=True
    )
    g_ep = replace(g, moe_tp=False, e_slots=68)
    backbone = _synth_backbone(g_ep, ex_ep)
    mlp = backbone.layers[0].mlp
    mlp.experts.moe_config = SimpleNamespace(
        use_ep=True,
        is_sequence_parallel=True,
        moe_parallel_config=SimpleNamespace(
            use_ep=True,
            ep_size=2,
            ep_rank=1,
            tp_size=1,
            tp_rank=0,
            sp_size=2,
            is_sequence_parallel=True,
        ),
    )

    with pytest.raises(ValueError, match="all_gather_fn must be provided"):
        reshard_layer_experts_ep_to_tp(mlp, tp_rank=1, g=g, ex=ex_ep)

    # Simulate 2-rank all_gather by repeating primary `ER=64` experts twice -> `E=128`
    reshard_layer_experts_ep_to_tp(
        mlp,
        tp_rank=1,
        g=g,
        ex=ex_ep,
        all_gather_fn=lambda t: torch.cat([t, t], dim=0),
    )
    ir = g.I // ex_ep.R
    assert tuple(mlp.experts.w13_weight.shape) == (g.E, g.H, 2 * ir)
    assert tuple(mlp.experts.w2_weight.shape) == (g.E, ir, g.H)
    assert not mlp.experts.w13_weight.data._is_view()
    assert not mlp.experts.w2_weight.data._is_view()
    assert mlp.experts.moe_config.use_ep is False
    assert mlp.experts.moe_config.moe_parallel_config.tp_size == 2
    assert mlp.experts.moe_config.moe_parallel_config.tp_rank == 1
    assert mlp.experts.moe_config.moe_parallel_config.ep_size == 1


def test_load_vllm_weights_does_not_touch_jax_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`load_vllm_weights` with `tiles=None` never calls `jax.devices()`."""
    g = _ep_geom()
    ex = ExecSpec(R=2, B=4, T=1, ctx=2048, stack_e=False, tiles=None)
    backbone = _synth_backbone(g, ex)
    monkeypatch.setattr(
        jax,
        "devices",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("jax.devices() must not be called during load_vllm_weights")
        ),
    )
    packed = load_vllm_weights(SynthHybridVLLM, backbone, ex, g=g, tp_rank=1)
    assert set(packed["plan"].w_names) <= set(packed)


def test_routed_experts_wrapper_and_scale_inv_fallback() -> None:
    """Verify `_walk_attr` reaches `experts.routed_experts` with `*_scale_inv`."""
    g, ex = _ep_geom(), _exec()
    e_cores = mk.default_eplb_cores(g, ex.R)
    backbone = _synth_backbone(g, ex, e_cores=e_cores)
    for layer in backbone.layers:
        inner = layer.mlp.experts
        routed = SimpleNamespace(
            w13_weight=inner.w13_weight,
            w2_weight=inner.w2_weight,
            w13_weight_scale_inv=inner.w13_weight_scale,
            w2_weight_scale_inv=inner.w2_weight_scale,
            _ecopy_cores=inner._ecopy_cores,
        )
        layer.mlp.experts = SimpleNamespace(routed_experts=routed)

    packed = load_vllm_weights(SynthHybridVLLM, backbone, ex, g=g, tp_rank=0)
    for l_idx in range(g.L):
        routed = backbone.layers[l_idx].mlp.experts.routed_experts
        assert packed["e13"][l_idx].data_ptr() == routed.w13_weight.data_ptr()
        assert packed["e2"][l_idx].data_ptr() == routed.w2_weight.data_ptr()
    ecopy_off, ecopy_w = packed["plan"].smalls_layout["ecopy"]
    assert torch.any(packed["smalls"][:, 0, ecopy_off : ecopy_off + ecopy_w] > 0.0)


def test_load_vllm_weights_validation_errors() -> None:
    """Verify scale shape, expert shape, missing alias, and `g=None` behavior."""
    g, ex = _ep_geom(), _exec()

    # 1. Wrong 2D scale length raises ValueError.
    bb_2d = _synth_backbone(g, ex, fused_mlp=False)
    bb_2d.layers[0].linear_attn.out_proj.weight_scale = _param(
        torch.ones((17, 1), dtype=torch.float32)
    )
    with pytest.raises(ValueError, match="weight_scale for 'linear_attn.out_proj'"):
        load_vllm_weights(SynthHybridVLLM, bb_2d, ex, g=g, tp_rank=0)

    # 2. Wrong 3D expert scale shape raises ValueError.
    bb_3d = _synth_backbone(g, ex, fused_mlp=True)
    bb_3d.layers[0].mlp.experts.w13_weight_scale = _param(
        torch.ones((68, 1, 7), dtype=torch.float32)
    )
    with pytest.raises(ValueError, match="scale for 'mlp.experts.w13_weight'"):
        load_vllm_weights(SynthHybridVLLM, bb_3d, ex, g=g, tp_rank=0)

    # 3. Wrong expert EW (unpadded ER=64 when EW=68) raises ValueError on stack_e=False.
    bb_ew = _synth_backbone(g, ex, fused_mlp=True)
    ew_experts = bb_ew.layers[0].mlp.experts
    ew_experts.w13_weight = _param(ew_experts.w13_weight[:64].contiguous())
    ew_experts.w13_weight_scale = _param(ew_experts.w13_weight_scale[:64].contiguous())
    with pytest.raises(ValueError, match="Layer 0 weight 'e13' has shape"):
        load_vllm_weights(SynthHybridVLLM, bb_ew, ex, g=g, tp_rank=0)

    # 4. Missing alias lists all attempted vllm_aliases sources.
    bb_miss = _synth_backbone(g, ex, fused_mlp=False)
    delattr(bb_miss.layers[0].mlp, "gate")
    with pytest.raises(
        ValueError,
        match=r"\['mlp\._fused_gate_gu_fp8_weight', 'mlp\.gate\.weight'\]",
    ):
        load_vllm_weights(SynthHybridVLLM, bb_miss, ex, g=g, tp_rank=0)

    # 5. `load_vllm_weights(model_def, backbone, ex)` works without passing `g=`.
    g_default = SynthHybridVLLM.geometry()
    bb_def = _synth_backbone(g_default, ex, fused_mlp=True)
    packed_no_g = load_vllm_weights(SynthHybridVLLM, bb_def, ex, tp_rank=0)
    assert len(packed_no_g["e13"]) == 4

    # 6. `VLLMWeightSource.read_model_param` validates `EmbeddingModule` shards
    # via `slice_alias_targets` and declines a padded `VocabParallelEmbedding`
    # shard by name (`model.embed_tokens.weight`).
    vocab = 64
    emb_geom = ResolvedModuleGeometry(
        EmbeddingModule, "model.embed_tokens", {"vocab": vocab, "H": g_default.H}
    )
    emb_decl = EmbeddingModule("model.embed_tokens").hf_params(g=emb_geom)[0]

    def _embed(rows: int, cols: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=_param(torch.ones((rows, cols), dtype=torch.bfloat16))
        )

    bb_def.embed_tokens = _embed(vocab // ex.R, g_default.H)
    src_emb = VLLMWeightSource(bb_def, g_default)
    emb_shard = src_emb.read_model_param(emb_decl, R=ex.R, tp_rank=0)
    assert tuple(emb_shard.shape) == (vocab // ex.R, g_default.H)
    bb_def.embed_tokens = _embed(vocab // ex.R + 8, g_default.H)
    with pytest.raises(
        ValueError,
        match=(
            r"Alias 'model\.embed_tokens\.weight' \(already_sharded=True\) "
            r"axis 0 width 40 != expected 32"
        ),
    ):
        src_emb.read_model_param(emb_decl, R=ex.R, tp_rank=0)
    bb_def.embed_tokens = _embed(vocab // ex.R, g_default.H + 8)
    with pytest.raises(
        ValueError,
        match=(
            r"Model parameter 'embed_tokens' shard has shape \(32, 264\), "
            r"expected \(32, 256\)"
        ),
    ):
        src_emb.read_model_param(emb_decl, R=ex.R, tp_rank=0)


def test_load_vllm_weights_non_qwen_synth_hybrid_model() -> None:
    """Verify `load_vllm_weights` loads a non-Qwen `ModelDef` (`att_every=2`)."""
    g, ex = _ep_geom(), _exec()
    backbone = _synth_backbone(g, ex, fused_mlp=True, fused_qkv=True)
    packed = load_vllm_weights(SynthHybridVLLM, backbone, ex, g=g, tp_rank=0)
    assert packed["att_in"].shape[0] == 2
    assert packed["lin_in"].shape[0] == 2
    for l_idx in range(g.L):
        assert (
            packed["e13"][l_idx].data_ptr()
            == backbone.layers[l_idx].mlp.experts.w13_weight.data_ptr()
        )
        assert (
            packed["e2"][l_idx].data_ptr()
            == backbone.layers[l_idx].mlp.experts.w2_weight.data_ptr()
        )


def _to_bf16_experts(experts: Any) -> None:
    """Dequantize fp8 experts in place (as an unquantized checkpoint loads)."""
    for w in ("w13_weight", "w2_weight"):
        s = getattr(experts, f"{w}_scale").data.float()
        deq = (getattr(experts, w).data.float() * s).to(torch.bfloat16)
        setattr(experts, w, _param(deq))
        delattr(experts, f"{w}_scale")


def test_bf16_experts_load_without_scales() -> None:
    """bf16 routed experts load with no scale (same packed weights as their fp8
    source dequantized by the library); fp8 experts still require a scale."""

    class Bf16Experts(SynthHybridVLLM):
        name = "test_bf16_experts"
        moe = dataclasses.replace(
            SynthHybridVLLM.moe,
            options={
                **SynthHybridVLLM.moe.options,
                "weight_dtypes": {"e13": "bf16", "e2": "bf16"},
            },
        )
        schedule = dataclasses.replace(SynthHybridVLLM.schedule, ffn=moe)

    g, ex = _ep_geom(), _exec()
    torch.manual_seed(0)
    backbone = _synth_backbone(g, ex)
    ref = load_vllm_weights(Bf16Experts, backbone, ex, g=g, tp_rank=1)
    for layer in backbone.layers:
        _to_bf16_experts(layer.mlp.experts)
    got = load_vllm_weights(Bf16Experts, backbone, ex, g=g, tp_rank=1)
    assert len(got["w_flat"]) == len(ref["w_flat"])
    for a, b in zip(got["w_flat"], ref["w_flat"], strict=True):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    # An fp8 expert weight without its scale is a clear error.
    torch.manual_seed(0)
    bb_fp8 = _synth_backbone(g, ex)
    delattr(bb_fp8.layers[0].mlp.experts, "w13_weight_scale")
    with pytest.raises(ValueError, match="w13_weight.* is missing its scale"):
        load_vllm_weights(SynthHybridVLLM, bb_fp8, ex, g=g, tp_rank=1)


def test_reshard_bf16_experts_ep_to_tp_without_scales() -> None:
    """bf16 experts reshard EP -> TP without scales, matching the dequantized
    fp8 reshard; fp8 experts missing a scale raise."""
    g, ex_ep = _ep_geom(), _exec()
    g_tp = replace(g, moe_tp=True, e_slots=0)
    torch.manual_seed(0)
    fp8_mlp = _synth_backbone(g, ex_ep).layers[0].mlp
    torch.manual_seed(0)
    bf16_mlp = _synth_backbone(g, ex_ep).layers[0].mlp
    _to_bf16_experts(bf16_mlp.experts)

    def gather(t: torch.Tensor) -> torch.Tensor:
        return torch.cat([t, t], dim=0)

    kw: dict[str, Any] = dict(tp_rank=1, g=g_tp, ex=ex_ep, all_gather_fn=gather)
    reshard_layer_experts_ep_to_tp(fp8_mlp, **kw)
    reshard_layer_experts_ep_to_tp(bf16_mlp, **kw)
    ir = g.I // ex_ep.R
    bf16_ex, fp8_ex = bf16_mlp.experts, fp8_mlp.experts
    assert tuple(bf16_ex.w13_weight.shape) == (g.E, g.H, 2 * ir)
    assert tuple(bf16_ex.w2_weight.shape) == (g.E, ir, g.H)
    assert bf16_ex.w13_weight.dtype == torch.bfloat16
    assert not hasattr(bf16_ex, "w13_weight_scale")
    assert not hasattr(bf16_ex, "w2_weight_scale")
    # Same slices as the fp8 path, dequantized.
    for w, s in (("w13_weight", "w13_weight_scale"), ("w2_weight", "w2_weight_scale")):
        sc = getattr(fp8_ex, s).data.float().reshape(g.E, 1, -1)
        deq = (getattr(fp8_ex, w).data.float() * sc).to(torch.bfloat16)
        assert torch.equal(getattr(bf16_ex, w).data, deq)

    torch.manual_seed(0)
    bad = _synth_backbone(g, ex_ep).layers[0].mlp
    delattr(bad.experts, "w2_weight_scale")
    with pytest.raises(ValueError, match="missing w2_weight_scale"):
        reshard_layer_experts_ep_to_tp(bad, **kw)


def test_bridge_and_weight_collector_have_no_vllm_imports() -> None:
    """Verify `exported_bridge`, `weight_collector` and `weight_source` are
    framework-portable.

    None may import from `vllm` or `vllm_torchtpu` outside
    `vllm_torchtpu.megakernel.core`, so they can be extracted into a
    standalone library alongside `megakernel`.
    """
    allowed_prefixes = (
        "vllm_torchtpu.megakernel.core",
        "vllm_torchtpu.megakernel.adapter",
    )
    for mod in (bridge_mod, wc_mod, ws_mod):
        path = Path(mod.__file__)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name.startswith("vllm"):
                        assert name.startswith(allowed_prefixes), (
                            f"{path.name} has disallowed import {name!r}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mod_name = node.module or ""
                if mod_name.startswith("vllm"):
                    assert mod_name.startswith(allowed_prefixes), (
                        f"{path.name} has disallowed from-import {mod_name!r}"
                    )
