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
"""Qwen3.8-2.4T geometry configuration, derived sharding dimensions, and validators."""

from __future__ import annotations

import dataclasses

from vllm_torchtpu.models.qwen38_2_4t.megakernel.constants import (
    DEFAULT_ATT_EVERY,
    DEFAULT_CONV_KERNEL,
    DEFAULT_EXPERT_INTER,
    DEFAULT_GDN_V_HEADS,
    DEFAULT_HIER_GROUP,
    DEFAULT_Q_HEADS,
    EXPECTED_DCP_SIZE,
    EXPECTED_TP_SIZE,
    MINI_EXPERTS,
    MINI_GDN_K_HEADS,
    MINI_HEAD_DIM,
    MINI_HIDDEN_SIZE,
    MINI_KV_HEADS,
    MINI_MAX_PAGES_PER_SEQ,
    MINI_NUM_LAYERS,
    MINI_NUM_PAGES,
    MINI_ROPE_DIM,
    MINI_TOP_K,
    NVFP4_GROUP_SIZE,
    PAGE_TOKENS,
    PROD_HIDDEN_SIZE,
    PROD_NUM_LAYERS,
)


def _check_positive_divisible(lhs: int, rhs: int, lhs_name: str, rhs_name: str) -> None:
    """Validate both integers are positive and `lhs % rhs == 0`."""
    if lhs <= 0 or rhs <= 0 or lhs % rhs != 0:
        msg = f"Invalid {lhs_name}={lhs}, {rhs_name}={rhs}: must be > 0 and divisible"
        raise ValueError(msg)


def _validate_dimensions(cfg: Qwen38Config) -> None:
    """Validate positive ranges for dimensions and MoE routing bounds."""
    if cfg.R <= 0:
        msg = f"Invalid R={cfg.R}: must be positive"
        raise ValueError(msg)
    if cfg.B <= 0:
        msg = f"Invalid B={cfg.B}: must be positive"
        raise ValueError(msg)
    if cfg.conv_kernel <= 0:
        msg = f"Invalid conv_kernel={cfg.conv_kernel}: must be > 0"
        raise ValueError(msg)
    if cfg.E <= 0 or cfg.K <= 0 or cfg.K > cfg.E:
        msg = f"Invalid MoE routing K={cfg.K}, E={cfg.E}: need 0 < K <= E"
        raise ValueError(msg)
    if cfg.max_pages_per_seq <= 0 or cfg.num_pages < cfg.max_pages_per_seq:
        msg = (
            f"Invalid pages max_pages_per_seq={cfg.max_pages_per_seq}, "
            f"num_pages={cfg.num_pages}"
        )
        raise ValueError(msg)


def _validate_head_counts(cfg: Qwen38Config) -> None:
    """Validate attention and GDN head counts and divisibility."""
    _check_positive_divisible(cfg.Q_H, cfg.R, "Q_H", "R")
    if cfg.Q_H % (cfg.KV_H * cfg.DCP) != 0:
        msg = f"Q_H={cfg.Q_H} must be divisible by KV_H*DCP={cfg.KV_H * cfg.DCP}"
        raise ValueError(msg)
    if (cfg.KV_H * cfg.DCP) % cfg.R != 0:
        msg = f"KV_H*DCP={cfg.KV_H * cfg.DCP} must be divisible by R={cfg.R}"
        raise ValueError(msg)
    if cfg.rope_dim <= 0 or cfg.rope_dim % 2 != 0 or cfg.rope_dim > cfg.D:
        msg = f"rope_dim={cfg.rope_dim} must be positive, even, and <= D={cfg.D}"
        raise ValueError(msg)
    if cfg.KH <= 0 or (cfg.KH % cfg.R != 0 and cfg.R % cfg.KH != 0):
        msg = f"KH={cfg.KH} must divide or be divisible by R={cfg.R}"
        raise ValueError(msg)
    kh_r = max(1, cfg.KH // cfg.R)
    vh_r = cfg.VH // cfg.R
    if vh_r <= 0 or vh_r % kh_r != 0 or cfg.VH % cfg.R != 0:
        msg = f"Incompatible GDN head counts: VH={cfg.VH}, KH={cfg.KH}, R={cfg.R}"
        raise ValueError(msg)


def _validate_config(cfg: Qwen38Config) -> None:
    """Validate structural, divisibility, and TP/DCP sharding invariants."""
    _validate_dimensions(cfg)
    _validate_head_counts(cfg)
    _check_positive_divisible(cfg.L, cfg.att_every, "L", "att_every")
    _check_positive_divisible(cfg.H, cfg.h_tile, "H", "h_tile")
    _check_positive_divisible(cfg.h_tile, cfg.fp4_group, "h_tile", "fp4_group")
    _check_positive_divisible(cfg.I_E, cfg.R, "I_E", "R")
    _check_positive_divisible(cfg.SE_I, cfg.R, "SE_I", "R")


@dataclasses.dataclass(frozen=True, slots=True)
class Qwen38Config:
    """Geometry and sharding configuration for the Qwen3.8-2.4T decode megakernel.

    Attributes:
        R: Total Tensor-Parallel (TP) rank world size.
        DCP: Decode Context-Parallel (DCP) subgroup size.
        B: Decode token batch bucket size.
        H: Model hidden size.
        L: Total decoder layers (`3:1` GDN and full attention).
        att_every: Full attention interval (every Nth layer is full attention).
        Q_H: Number of global query heads.
        KV_H: Number of global KV heads.
        D: Attention head dimension.
        rope_dim: Number of channels receiving NeoX rotary position embeddings.
        KH: Number of global GDN key heads.
        VH: Number of global GDN value heads.
        KD: GDN key head dimension.
        VD: GDN value head dimension.
        conv_kernel: GDN depthwise 1D convolution window width.
        E: Total routed MoE experts per layer.
        K: Active routed experts selected per token.
        I_E: Global routed expert intermediate size.
        SE_I: Global shared expert intermediate size.
        fp4_group: NVFP4 block quantization group size along K.
        page_size: Physical KV cache page token capacity.
        max_pages_per_seq: Maximum local KV pages allocated per sequence.
        num_pages: Total physical KV pages in each region pool.
        hier_group: Intra-node subgroup size for 2-level hierarchical all-reduce.
        h_tile: Horizontal H-tiling factor for streamed NVFP4 matmuls.
        NATT: Derived count of full-attention layers (`L // att_every`).
        NGDN: Derived count of GDN linear-attention layers (`L - NATT`).
        Q_R: Derived query heads per TP rank (`Q_H // R`).
        K_R: Derived KV heads per TP rank (`(KV_H * DCP) // R`).
        gqa_ratio: Derived query heads per KV head (`Q_H // (KV_H * DCP)`).
        ATT_QKV_R: Derived packed attention Q/Q-gate/K/V width per rank.
        VH_R: Derived GDN value heads per TP rank (`VH // R`).
        KH_R: Derived GDN key heads per TP rank (`max(1, KH // R)`).
        gdn_v_per_k: Derived GDN value heads per key head (`VH_R // KH_R`).
        QKV: Derived GDN conv1d channel width (`2 * KH_R * KD + VH_R * VD`).
        Z_R: Derived GDN output gate width (`VH_R * VD`).
        I_R: Derived routed expert intermediate width per rank (`I_E // R`).
        SE_R: Derived shared expert intermediate width per rank (`SE_I // R`).
        max_seq_len: Derived maximum sequence length across `DCP` pages.
    """

    R: int = EXPECTED_TP_SIZE
    DCP: int = EXPECTED_DCP_SIZE
    B: int = 16
    H: int = PROD_HIDDEN_SIZE
    L: int = PROD_NUM_LAYERS
    att_every: int = 4

    Q_H: int = 128
    KV_H: int = 4
    D: int = 256
    rope_dim: int = 64

    KH: int = 16
    VH: int = 64
    KD: int = 128
    VD: int = 128
    conv_kernel: int = 4

    E: int = 512
    K: int = 8
    I_E: int = 2048
    SE_I: int = 2048

    fp4_group: int = NVFP4_GROUP_SIZE
    page_size: int = 128
    max_pages_per_seq: int = 16
    num_pages: int = 256
    hier_group: int = 4
    h_tile: int = 2048

    NATT: int = dataclasses.field(init=False, repr=False, compare=False)
    NGDN: int = dataclasses.field(init=False, repr=False, compare=False)
    Q_R: int = dataclasses.field(init=False, repr=False, compare=False)
    K_R: int = dataclasses.field(init=False, repr=False, compare=False)
    gqa_ratio: int = dataclasses.field(init=False, repr=False, compare=False)
    ATT_QKV_R: int = dataclasses.field(init=False, repr=False, compare=False)
    VH_R: int = dataclasses.field(init=False, repr=False, compare=False)
    KH_R: int = dataclasses.field(init=False, repr=False, compare=False)
    gdn_v_per_k: int = dataclasses.field(init=False, repr=False, compare=False)
    QKV: int = dataclasses.field(init=False, repr=False, compare=False)
    Z_R: int = dataclasses.field(init=False, repr=False, compare=False)
    I_R: int = dataclasses.field(init=False, repr=False, compare=False)
    SE_R: int = dataclasses.field(init=False, repr=False, compare=False)
    max_seq_len: int = dataclasses.field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate configuration invariants and initialize derived shape fields."""
        _validate_config(self)

        natt = self.L // self.att_every
        ngdn = self.L - natt
        q_r = self.Q_H // self.R
        k_r = (self.KV_H * self.DCP) // self.R
        gqa_ratio = self.Q_H // (self.KV_H * self.DCP)
        att_qkv_r = (2 * q_r + 2 * k_r) * self.D
        vh_r = self.VH // self.R
        kh_r = max(1, self.KH // self.R)
        gdn_v_per_k = vh_r // kh_r
        qkv = 2 * kh_r * self.KD + vh_r * self.VD
        z_r = vh_r * self.VD
        i_r = self.I_E // self.R
        se_r = self.SE_I // self.R
        max_seq_len = self.DCP * self.max_pages_per_seq * self.page_size

        object.__setattr__(self, "NATT", natt)
        object.__setattr__(self, "NGDN", ngdn)
        object.__setattr__(self, "Q_R", q_r)
        object.__setattr__(self, "K_R", k_r)
        object.__setattr__(self, "gqa_ratio", gqa_ratio)
        object.__setattr__(self, "ATT_QKV_R", att_qkv_r)
        object.__setattr__(self, "VH_R", vh_r)
        object.__setattr__(self, "KH_R", kh_r)
        object.__setattr__(self, "gdn_v_per_k", gdn_v_per_k)
        object.__setattr__(self, "QKV", qkv)
        object.__setattr__(self, "Z_R", z_r)
        object.__setattr__(self, "I_R", i_r)
        object.__setattr__(self, "SE_R", se_r)
        object.__setattr__(self, "max_seq_len", max_seq_len)


def mini_parity_config(batch_size: int = 4) -> Qwen38Config:
    """Return a compact 4-layer TP32xDCP8 geometry for fast CPU parity tests."""
    return Qwen38Config(
        R=EXPECTED_TP_SIZE,
        DCP=EXPECTED_DCP_SIZE,
        B=batch_size,
        H=MINI_HIDDEN_SIZE,
        L=MINI_NUM_LAYERS,
        att_every=DEFAULT_ATT_EVERY,
        Q_H=DEFAULT_Q_HEADS,
        KV_H=MINI_KV_HEADS,
        D=MINI_HEAD_DIM,
        rope_dim=MINI_ROPE_DIM,
        KH=MINI_GDN_K_HEADS,
        VH=DEFAULT_GDN_V_HEADS,
        KD=MINI_HEAD_DIM,
        VD=MINI_HEAD_DIM,
        conv_kernel=DEFAULT_CONV_KERNEL,
        E=MINI_EXPERTS,
        K=MINI_TOP_K,
        I_E=DEFAULT_EXPERT_INTER,
        SE_I=DEFAULT_EXPERT_INTER,
        fp4_group=NVFP4_GROUP_SIZE,
        page_size=PAGE_TOKENS,
        max_pages_per_seq=MINI_MAX_PAGES_PER_SEQ,
        num_pages=MINI_NUM_PAGES,
        hier_group=DEFAULT_HIER_GROUP,
        h_tile=MINI_HIDDEN_SIZE,
    )
