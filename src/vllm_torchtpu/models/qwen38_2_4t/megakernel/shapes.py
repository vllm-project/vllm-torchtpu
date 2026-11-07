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
"""Frozen geometry configuration for the Qwen3.8-2.4T TP32xDCP8 NVFP4 megakernel."""

from __future__ import annotations

import dataclasses

_PAGE_SIZE_ALIGNMENT: int = 4
PROD_NUM_LAYERS: int = 92
PROD_HIDDEN_SIZE: int = 8192
MINI_HIDDEN_SIZE: int = 128
EXPECTED_TP_SIZE: int = 32
EXPECTED_DCP_SIZE: int = 8
NVFP4_GROUP_SIZE: int = 16


def _validate_gdn_and_tile_config(cfg: Qwen38Config) -> None:
    """Verify GDN head and tile/page divisibility invariants across ranks."""
    if cfg.VH % cfg.R != 0:
        msg = f"VH={cfg.VH} must be divisible by R={cfg.R}"
        raise ValueError(msg)
    if cfg.KH <= 0 or (cfg.KH % cfg.R != 0 and cfg.R % cfg.KH != 0):
        msg = f"KH={cfg.KH} must divide or be divisible by R={cfg.R}"
        raise ValueError(msg)
    vh_r = cfg.VH // cfg.R
    kh_r = max(1, cfg.KH // cfg.R)
    if (
        cfg.VH % cfg.KH != 0
        or vh_r % kh_r != 0
        or (cfg.KH < cfg.R and (cfg.VH // cfg.KH) % vh_r != 0)
    ):
        msg = f"Incompatible GDN head counts KH={cfg.KH}, VH={cfg.VH}, R={cfg.R}"
        raise ValueError(msg)
    if cfg.I_E % cfg.R != 0 or cfg.SE_I % cfg.R != 0:
        msg = f"I_E={cfg.I_E} and SE_I={cfg.SE_I} must be divisible by R={cfg.R}"
        raise ValueError(msg)
    if cfg.H % cfg.fp4_group != 0 or cfg.H % cfg.h_tile != 0:
        msg = (
            f"H={cfg.H} must be divisible by fp4_group={cfg.fp4_group} "
            f"and h_tile={cfg.h_tile}"
        )
        raise ValueError(msg)
    if cfg.h_tile % cfg.fp4_group != 0:
        msg = f"h_tile={cfg.h_tile} must be divisible by fp4_group={cfg.fp4_group}"
        raise ValueError(msg)
    if cfg.page_size % _PAGE_SIZE_ALIGNMENT != 0:
        msg = f"page_size={cfg.page_size} must be divisible by {_PAGE_SIZE_ALIGNMENT}"
        raise ValueError(msg)


def _validate_config(cfg: Qwen38Config) -> None:
    """Validate geometry and divisibility constraints for Qwen38Config."""
    if cfg.R <= 0 or cfg.DCP <= 0 or cfg.R % cfg.DCP != 0:
        msg = f"Invalid R={cfg.R}, DCP={cfg.DCP}"
        raise ValueError(msg)
    if cfg.L <= 0 or cfg.L % cfg.att_every != 0:
        msg = f"L={cfg.L} must be divisible by att_every={cfg.att_every}"
        raise ValueError(msg)
    if cfg.Q_H % cfg.R != 0:
        msg = f"Q_H={cfg.Q_H} must be divisible by R={cfg.R}"
        raise ValueError(msg)
    if (cfg.KV_H * cfg.DCP) % cfg.R != 0:
        msg = f"KV_H*DCP={cfg.KV_H * cfg.DCP} must be divisible by R={cfg.R}"
        raise ValueError(msg)
    if cfg.Q_H % (cfg.KV_H * cfg.DCP) != 0:
        msg = f"Q_H={cfg.Q_H} must be divisible by KV_H*DCP={cfg.KV_H * cfg.DCP}"
        raise ValueError(msg)
    if cfg.rope_dim <= 0 or cfg.rope_dim % 2 != 0 or cfg.rope_dim > cfg.D:
        msg = f"rope_dim={cfg.rope_dim} must be positive, even, and <= D={cfg.D}"
        raise ValueError(msg)
    _validate_gdn_and_tile_config(cfg)


@dataclasses.dataclass(frozen=True)
class Qwen38Config:
    """Configuration for Qwen3.8-2.4T (92L, 3:1 GDN:FullAttn, 512E top-8 + 1 SE).

    Attributes:
        R: Total Tensor-Parallel ranks (32 across 2 hosts).
        DCP: Decode Context Parallelism factor (8 ranks per KV head group).
        B: Decode batch size per step.
        H: Hidden dimension (8192).
        L: Total transformer layers (92 = 23 blocks of 3 GDN + 1 FullAttn).
        att_every: Full attention interval (every 4th layer).
        Q_H: Total query heads (128).
        KV_H: Total key/value heads (4).
        D: Head dimension for full attention (256).
        rope_dim: Partial RoPE dimension (64, rotary_factor=0.25).
        KH: Total GDN key/query heads (16).
        VH: Total GDN value heads (64).
        KD: GDN key head dimension (128).
        VD: GDN value head dimension (128).
        conv_kernel: GDN depthwise conv1d kernel size (4 -> state window 3).
        E: Total routed experts (512).
        K: Active routed experts per token (8).
        I_E: Routed expert intermediate size (2048).
        SE_I: Shared expert intermediate size (2048).
        fp4_group: NVFP4 block-scale group size (16).
        page_size: Tokens per KV cache page per DCP rank (128).
        max_pages_per_seq: Maximum KV pages per sequence per DCP rank.
        num_pages: Total physical pages in the unified per-rank KV/state pool.
        hier_group: First-stage hierarchical all-reduce group size (4).
        h_tile: Hidden-dimension tile size for streaming NVFP4 projections.

    """

    R: int = 32
    DCP: int = 8
    B: int = 16
    H: int = 8192
    L: int = 92
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

    fp4_group: int = 16
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
    """Scaled 32-rank TP32xDCP8 config for fast CPU interpret parity verification."""
    return Qwen38Config(
        R=32,
        DCP=8,
        B=batch_size,
        H=128,
        L=4,
        att_every=4,
        Q_H=128,
        KV_H=4,
        D=128,
        rope_dim=32,
        KH=32,
        VH=64,
        KD=128,
        VD=128,
        conv_kernel=4,
        E=16,
        K=2,
        I_E=2048,
        SE_I=2048,
        fp4_group=16,
        page_size=128,
        max_pages_per_seq=4,
        num_pages=96,
        hier_group=4,
        h_tile=128,
    )
