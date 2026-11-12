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
"""Mock vLLM Qwen3.8-2.4T model tree and shared config stubs for CPU tests."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import torch

from tests.models.qwen38_2_4t.constants import (
    CONV_BF16_BYTES,
    GLOBAL_POOL_NDIM,
    PAGES_PER_MAMBA_SLOT,
    PROD_HEAD_DIM,
    PROD_KV_HEADS,
    PROD_Q_HEADS,
    REC_PAGES_PER_MAMBA_SLOT,
)
from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import (
    NVFP4_GROUP_SIZE,
    PROD_HIDDEN_SIZE,
    PROD_NUM_LAYERS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from tests.models.qwen38_2_4t.oracle_fp64 import SyntheticProblem
    from vllm_torchtpu.models.qwen38_2_4t.megakernel.shapes import Qwen38Config


def pack_rows_fp4(
    w_kn_u8: np.ndarray, s_gk_n_bf16: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert `(K, N//2)` row-packed nibbles into `(N, K//2)` output-first nibbles."""
    k_dim, n_half = w_kn_u8.shape
    n_dim = n_half * 2
    lo = w_kn_u8 & 0x0F
    hi = (w_kn_u8 >> 4) & 0x0F
    nib_kn = np.empty((k_dim, n_dim), dtype=np.uint8)
    nib_kn[:, 0::2] = lo
    nib_kn[:, 1::2] = hi
    nib_nk = np.ascontiguousarray(nib_kn.T)
    pk_nk = (nib_nk[:, 0::2] & 0x0F) | ((nib_nk[:, 1::2] & 0x0F) << 4)
    sc_nk = np.ascontiguousarray(np.asarray(s_gk_n_bf16, dtype=np.float32).T)
    return (
        torch.from_numpy(pk_nk).to(torch.uint8),
        torch.from_numpy(sc_nk).to(torch.bfloat16),
    )


@dataclasses.dataclass
class MockWeightHolder:
    """Mock weight container holding a PyTorch tensor, ndarray, or `None`."""

    weight: torch.Tensor | np.ndarray | None = None


@dataclasses.dataclass
class MockLinearProj:
    """Mock quantized linear projection holding packed weight and scale tensors."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_scale_2: torch.Tensor | None = None


@dataclasses.dataclass
class MockSharedExpert:
    """Mock shared expert MLP holding gate-up and down projections."""

    gate_up_proj: MockLinearProj
    down_proj: MockLinearProj


@dataclasses.dataclass
class MockExperts:
    """Mock routed experts container holding stacked expert weights and scales."""

    w13_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight: torch.Tensor
    w2_weight_scale: torch.Tensor
    down_proj: torch.Tensor | None = None
    down_proj_scale: torch.Tensor | None = None


@dataclasses.dataclass
class MockDottedExperts:
    """Mock routed experts exposing nested `gate_up_proj`/`down_proj` submodules."""

    gate_up_proj: MockLinearProj
    down_proj: MockLinearProj


@dataclasses.dataclass
class MockMlp:
    """Mock MoE MLP block holding router gates, shared expert, and routed experts."""

    gate: MockWeightHolder
    shared_expert_gate: MockWeightHolder | None
    shared_expert: MockSharedExpert
    experts: object


@dataclasses.dataclass
class MockBadMlp:
    """Stub MLP container with missing `shared_expert_gate` for error tests."""

    gate: MockWeightHolder
    shared_expert_gate: MockWeightHolder | None = None


@dataclasses.dataclass
class MockEmptyExperts:
    """Stub experts container with no weight attributes."""


@dataclasses.dataclass
class MockInnerAttn:
    """Mock inner attention module holding layer name and aliased KV cache pool."""

    layer_name: str
    kv_cache: torch.Tensor


@dataclasses.dataclass
class MockSelfAttn:
    """Mock full self-attention sublayer."""

    prefix: str
    qkv_proj: MockLinearProj
    o_proj: MockLinearProj
    q_norm: MockWeightHolder
    k_norm: MockWeightHolder
    attn: MockInnerAttn


@dataclasses.dataclass
class MockLinearAttn:
    """Mock GDN linear-attention sublayer."""

    prefix: str
    kv_cache: torch.Tensor
    in_proj_qkvz: MockLinearProj
    in_proj_ba: MockWeightHolder
    conv1d: MockWeightHolder
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm: MockWeightHolder
    out_proj: MockLinearProj
    layer_name: str = ""


@dataclasses.dataclass
class MockDecoderLayer:
    """Mock decoder layer holding norms, MLP, and either self or linear attention."""

    input_layernorm: MockWeightHolder
    post_attention_layernorm: MockWeightHolder
    mlp: MockMlp
    self_attn: MockSelfAttn | None = None
    linear_attn: MockLinearAttn | None = None


@dataclasses.dataclass
class MockBackbone:
    """Mock single-rank decoder backbone."""

    layers: list[MockDecoderLayer]
    norm: MockWeightHolder
    embed_tokens: Callable[[torch.Tensor], torch.Tensor] | None = None


@dataclasses.dataclass
class MockRankModelHolder:
    """Mock multi-rank model container implementing `MultiRankBackboneHolder`."""

    model: MockBackbone
    _backbones: tuple[MockBackbone, ...]

    @property
    def rank_backbones(self) -> Sequence[MockBackbone]:
        """Return per-rank decoder backbones."""
        return self._backbones


@dataclasses.dataclass
class StubHfConfig:
    """Stub HuggingFace model config for Qwen3.8 eligibility tests."""

    architectures: list[str]
    model_type: str
    num_hidden_layers: int | None = PROD_NUM_LAYERS
    hidden_size: int | None = PROD_HIDDEN_SIZE
    num_attention_heads: int = PROD_Q_HEADS
    num_key_value_heads: int = PROD_KV_HEADS
    head_dim: int = PROD_HEAD_DIM
    linear_num_key_heads: int | None = None
    linear_num_value_heads: int | None = None
    linear_key_head_dim: int | None = None
    linear_value_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    shared_expert_intermediate_size: int | None = None
    quantization_config: dict[str, object] | None = None


def qwen38_hf_config(
    *,
    num_hidden_layers: int | None = PROD_NUM_LAYERS,
    hidden_size: int | None = PROD_HIDDEN_SIZE,
    num_attention_heads: int = PROD_Q_HEADS,
    quantization_config: dict[str, object] | str | None = "default",
) -> StubHfConfig:
    """Return a production Qwen3.8 HuggingFace config stub with selective overrides."""
    resolved_quant: dict[str, object] | None = (
        {"quant_method": "nvfp4", "group_size": NVFP4_GROUP_SIZE}
        if quantization_config == "default"
        else cast("dict[str, object] | None", quantization_config)
    )
    return StubHfConfig(
        architectures=["Qwen3_5MoeForCausalLM"],
        model_type="qwen3_5_moe_text",
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        quantization_config=resolved_quant,
    )


@dataclasses.dataclass
class StubBuilderCtx:
    """Stub attention metadata builder context carrying active request count."""

    num_reqs: int


@dataclasses.dataclass
class StubLayerMetadata:
    """Stub per-layer attention or Mamba state metadata."""

    block_tables: torch.Tensor | None = None
    mamba_state_indices: torch.Tensor | None = None


def _make_embed_tokens_fn(
    x_embed: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a token embedding lookup closure over a rank's input tensor."""

    def _embed_tokens(ids: torch.Tensor) -> torch.Tensor:
        """Index into pre-populated rank embedding rows."""
        return x_embed[ids.long()]

    return _embed_tokens


def _build_mock_layer(
    l_idx: int,
    r: int,
    cfg: Qwen38Config,
    w: dict[str, np.ndarray],
    reg_pool_t: torch.Tensor,
) -> MockDecoderLayer:
    """Build one mock decoder layer with simulated linear and attention blocks."""
    is_att = (l_idx + 1) % cfg.att_every == 0
    in_ln = MockWeightHolder(
        weight=torch.from_numpy(
            np.asarray(w["in_norm"][r, l_idx], dtype=np.float32)
        ).to(torch.bfloat16)
    )
    post_ln = MockWeightHolder(
        weight=torch.from_numpy(
            np.asarray(w["post_norm"][r, l_idx], dtype=np.float32)
        ).to(torch.bfloat16)
    )

    w_gate_t = torch.from_numpy(
        np.asarray(w["w_gate"][r, l_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    w_se_gate_t = torch.from_numpy(
        np.asarray(w["w_se_gate"][r, l_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    se13_w, se13_s = pack_rows_fp4(w["w_se13"][r, l_idx], w["w_se13_s"][r, l_idx])
    se2_w, se2_s = pack_rows_fp4(w["w_se2"][r, l_idx], w["w_se2_s"][r, l_idx])

    e13_pairs = [
        pack_rows_fp4(w["w_e13"][r, l_idx, e_id], w["w_e13_s"][r, l_idx, e_id])
        for e_id in range(cfg.E)
    ]
    e2_pairs = [
        pack_rows_fp4(w["w_e2"][r, l_idx, e_id], w["w_e2_s"][r, l_idx, e_id])
        for e_id in range(cfg.E)
    ]
    mlp = MockMlp(
        gate=MockWeightHolder(weight=w_gate_t),
        shared_expert_gate=MockWeightHolder(weight=w_se_gate_t),
        shared_expert=MockSharedExpert(
            gate_up_proj=MockLinearProj(weight=se13_w, weight_scale=se13_s),
            down_proj=MockLinearProj(weight=se2_w, weight_scale=se2_s),
        ),
        experts=MockExperts(
            w13_weight=torch.stack([p[0] for p in e13_pairs], dim=0),
            w13_weight_scale=torch.stack([p[1] for p in e13_pairs], dim=0),
            w2_weight=torch.stack([p[0] for p in e2_pairs], dim=0),
            w2_weight_scale=torch.stack([p[1] for p in e2_pairs], dim=0),
        ),
    )

    if is_att:
        att_idx = l_idx // cfg.att_every
        qkv_w, qkv_s = pack_rows_fp4(w["wqkv"][r, att_idx], w["wqkv_s"][r, att_idx])
        wo_w, wo_s = pack_rows_fp4(w["wo"][r, att_idx], w["wo_s"][r, att_idx])
        q_norm_t = torch.from_numpy(
            np.asarray(w["q_norm"][r, att_idx], dtype=np.float32)
        ).to(torch.bfloat16)
        k_norm_t = torch.from_numpy(
            np.asarray(w["k_norm"][r, att_idx], dtype=np.float32)
        ).to(torch.bfloat16)
        self_attn = MockSelfAttn(
            prefix=f"model.layers.{l_idx}.self_attn.attn",
            qkv_proj=MockLinearProj(weight=qkv_w, weight_scale=qkv_s),
            o_proj=MockLinearProj(weight=wo_w, weight_scale=wo_s),
            q_norm=MockWeightHolder(weight=q_norm_t),
            k_norm=MockWeightHolder(weight=k_norm_t),
            attn=MockInnerAttn(
                layer_name=f"model.layers.{l_idx}.self_attn.attn",
                kv_cache=reg_pool_t,
            ),
        )
        return MockDecoderLayer(
            input_layernorm=in_ln,
            post_attention_layernorm=post_ln,
            self_attn=self_attn,
            mlp=mlp,
        )

    g_idx = l_idx - (l_idx // cfg.att_every)
    inproj_w, inproj_s = pack_rows_fp4(
        w["w_inproj"][r, g_idx], w["w_inproj_s"][r, g_idx]
    )
    out_w, out_s = pack_rows_fp4(w["w_out"][r, g_idx], w["w_out_s"][r, g_idx])
    ba_t = torch.from_numpy(
        np.asarray(w["w_ba"][r, g_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    conv_t = torch.from_numpy(
        np.asarray(w["w_conv"][r, g_idx], dtype=np.float32).T.copy()
    ).to(torch.bfloat16)
    alog_t = torch.from_numpy(
        np.asarray(w["a_log"][r, g_idx], dtype=np.float32).copy()
    ).to(torch.float32)
    dt_t = torch.from_numpy(
        np.asarray(w["dt_bias"][r, g_idx], dtype=np.float32).copy()
    ).to(torch.float32)
    gnorm_t = torch.from_numpy(
        np.asarray(w["gdn_norm"][r, g_idx], dtype=np.float32).copy()
    ).to(torch.bfloat16)
    linear_attn = MockLinearAttn(
        prefix=f"model.layers.{l_idx}.linear_attn",
        layer_name=f"model.layers.{l_idx}.linear_attn",
        kv_cache=reg_pool_t,
        in_proj_qkvz=MockLinearProj(weight=inproj_w, weight_scale=inproj_s),
        in_proj_ba=MockWeightHolder(weight=ba_t),
        conv1d=MockWeightHolder(weight=conv_t),
        A_log=alog_t,
        dt_bias=dt_t,
        norm=MockWeightHolder(weight=gnorm_t),
        out_proj=MockLinearProj(weight=out_w, weight_scale=out_s),
    )
    return MockDecoderLayer(
        input_layernorm=in_ln,
        post_attention_layernorm=post_ln,
        linear_attn=linear_attn,
        mlp=mlp,
    )


def _gdn_region_index(g_idx: int, cfg: Qwen38Config, mode: str) -> int:
    """Return unified region pool index for GDN sublayer `g_idx` under `mode`."""
    if mode == "strided":
        return g_idx % cfg.NATT
    g_per_reg = max(1, cfg.att_every - 1)
    return min(g_idx // g_per_reg, cfg.NATT - 1)


def build_mock_vllm_qwen38_rank_models(
    problem: SyntheticProblem | Mapping[str, object],
    cfg: Qwen38Config,
    *,
    gdn_region_mode: str = "block",
) -> MockRankModelHolder:
    """Build an `R`-rank mock vLLM Qwen3.8 model tree with aliased KV/GDN pools."""
    weights_map = cast("Mapping[str, jax.Array]", problem["weights"])
    pool_arr = cast("jax.Array", problem["pool"])
    conv_arr = cast("jax.Array", problem["conv"])
    rec_arr = cast("jax.Array", problem["rec"])
    x_arr = cast("jax.Array", problem["x"])
    mamba_slots = cast("np.ndarray", problem["mamba_slots"])

    w = {k: np.asarray(jax.device_get(v)) for k, v in weights_map.items()}
    pool_np = np.asarray(jax.device_get(pool_arr).view(jnp.uint8))
    conv_np = np.asarray(jax.device_get(conv_arr).astype(jnp.float32), dtype=np.float32)
    rec_np = np.asarray(jax.device_get(rec_arr), dtype=np.float32)

    rank_backbones: list[MockBackbone] = []
    conv_bytes = (cfg.conv_kernel - 1) * cfg.QKV * CONV_BF16_BYTES

    for r in range(cfg.R):
        region_pools_t = [
            torch.from_numpy(
                (
                    pool_np[r, reg] if pool_np.ndim == GLOBAL_POOL_NDIM else pool_np[r]
                ).copy()
            ).view(torch.float8_e4m3fn)
            for reg in range(cfg.NATT)
        ]
        for g_idx in range(cfg.NGDN):
            r_idx = _gdn_region_index(g_idx, cfg, gdn_region_mode)
            pool_t = region_pools_t[r_idx]
            for b_idx in range(cfg.B):
                slot = int(mamba_slots[g_idx, b_idx])
                if slot < 0:
                    continue
                p_base = slot * PAGES_PER_MAMBA_SLOT
                rec_bytes = (
                    torch.from_numpy(rec_np[r, g_idx, b_idx].copy())
                    .to(torch.float32)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                pool_t[p_base : p_base + REC_PAGES_PER_MAMBA_SLOT].view(
                    torch.uint8
                ).reshape(-1)[: rec_bytes.numel()].copy_(rec_bytes)
                conv_bf16 = (
                    torch.from_numpy(conv_np[r, g_idx, b_idx].copy())
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                    .reshape(-1)
                )
                pool_t[p_base + REC_PAGES_PER_MAMBA_SLOT].view(torch.uint8).reshape(-1)[
                    :conv_bytes
                ].copy_(conv_bf16)

        rank_layers = [
            _build_mock_layer(
                l_idx,
                r,
                cfg,
                w,
                region_pools_t[
                    (l_idx // cfg.att_every)
                    if (l_idx + 1) % cfg.att_every == 0
                    else _gdn_region_index(
                        l_idx - (l_idx // cfg.att_every), cfg, gdn_region_mode
                    )
                ],
            )
            for l_idx in range(cfg.L)
        ]
        norm_t = torch.from_numpy(
            np.asarray(w["final_norm"][r], dtype=np.float32).copy()
        ).to(torch.bfloat16)
        x_embed_t = torch.from_numpy(
            np.asarray(
                jax.device_get(x_arr)[r].astype(jnp.float32),
                dtype=np.float32,
            ).copy()
        ).to(torch.bfloat16)
        rank_backbones.append(
            MockBackbone(
                layers=rank_layers,
                norm=MockWeightHolder(weight=norm_t),
                embed_tokens=_make_embed_tokens_fn(x_embed_t),
            )
        )

    return MockRankModelHolder(
        model=rank_backbones[0],
        _backbones=tuple(rank_backbones),
    )
