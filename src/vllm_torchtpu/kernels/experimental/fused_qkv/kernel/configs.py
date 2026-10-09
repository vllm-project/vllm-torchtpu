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

"""Static attention and GDN configs, and the weights they describe.

Shape dims, used with these names across the package:
  T: tokens.
  tile: tokens per tile (one kernel grid step).
  rows: rows of a helper's block (a group of rows within a tile).
  K: hidden size, the projection's contraction dim.
  N: projection columns (all heads' blocks side by side).
  Hq: query heads.
  Hkv: KV heads.
  Hv: GDN value heads.
  D: head dim.
  n_pairs: RoPE rotation pairs per head.
  block: columns per fp8 activation quant block (act_quant_block).
  lanes: TPU vector lanes (128).
"""

from __future__ import annotations

import dataclasses
import enum
from typing import NamedTuple

import jax
import jax.numpy as jnp


class Norm(enum.Enum):
    """Per-head norm applied to q, k or v."""

    NONE = "none"
    RMS = "rms"  # RMSNorm: f32 normalize -> bf16 -> * w in bf16.
    # vLLM GemmaRMSNorm (also Qwen3.5's, as Qwen3NextRMSNorm): w is stored as an
    # offset from 1, and f32 normalize * (1 + w) -> bf16.
    RMS_ONE_PLUS_W = "rms_one_plus_w"
    RMS_NOSCALE = "rms_noscale"  # RMSNorm(has_weight=False).


@dataclasses.dataclass(frozen=True)
class Rope:
    """NeoX rotation of x[:n_pairs] with x[offset:offset + n_pairs]."""

    n_pairs: int
    offset: int

    @classmethod
    def neox(cls, rotary_dim: int) -> Rope:
        return cls(rotary_dim // 2, rotary_dim // 2)


@dataclasses.dataclass(frozen=True)
class AttentionConfig:
    """Attention shapes and options for one chip (head counts after TP)."""

    num_q_heads: int  # Hq.
    num_kv_heads: int  # Hkv.
    head_dim: int  # D.
    gate: bool = False  # Qwen3.5: q_proj also outputs a per-head output gate.
    v_from_k: bool = False  # Gemma 4 full attn: no v_proj; v = raw k projection.
    fp8: bool = True
    # fp8 only: columns per dynamic activation quant block (sub-channel). Must
    # divide the hidden size; larger blocks mean less dequant work in the kernel.
    act_quant_block: int = 1024
    bias: bool = False
    # Decoder input norm; with `residual`, the residual add is fused before it.
    input_norm: Norm = Norm.RMS
    residual: bool = False
    q_norm: Norm = Norm.NONE
    k_norm: Norm = Norm.NONE
    v_norm: Norm = Norm.NONE
    rope: Rope | None = None
    eps: float = 1e-6
    q_dtype: jnp.dtype = jnp.bfloat16
    kv_dtype: jnp.dtype = jnp.float8_e4m3fn
    k_scale: float = 1.0
    v_scale: float = 1.0

    @property
    def q_packing(self) -> int:  # Q values per 32-bit word.
        return 32 // jax.dtypes.itemsize_bits(self.q_dtype)

    @property
    def kv_packing(self) -> int:  # KV values per 32-bit word.
        return 32 // jax.dtypes.itemsize_bits(self.kv_dtype)

    # W is q_proj, k_proj and v_proj side by side; each block is one head:
    #   [q_0 | q_1 | ... | k_0 | k_1 | ... | v_0 | v_1 | ...]
    # With `gate`, each q head is followed by its gate: [q_0 | g_0 | q_1 | ...].
    # With `v_from_k`, there are no v blocks; v reads k's columns.
    # The *_col methods return where a head's block starts.

    @property
    def kv_col(self) -> int:
        return (2 if self.gate else 1) * self.num_q_heads * self.head_dim

    @property
    def num_cols(self) -> int:  # N.
        num_kv = (1 if self.v_from_k else 2) * self.num_kv_heads
        return self.kv_col + num_kv * self.head_dim

    def q_col(self, q_head: int) -> int:
        return (2 * q_head if self.gate else q_head) * self.head_dim

    def gate_col(self, q_head: int) -> int:
        return (2 * q_head + 1) * self.head_dim

    def k_col(self, kv_head: int) -> int:
        return self.kv_col + kv_head * self.head_dim

    def v_col(self, kv_head: int) -> int:
        if self.v_from_k:
            return self.k_col(kv_head)
        return self.kv_col + (self.num_kv_heads + kv_head) * self.head_dim


@dataclasses.dataclass(frozen=True)
class GdnConfig:
    """Qwen3-Next / Qwen3.5 Gated DeltaNet prologue shapes for one chip.

    Head counts are after TP. in_proj_qkvz gives mixed_qkv ([q | k | v]) and z;
    in_proj_ba gives b and a.
    """

    num_k_heads: int  # Heads of q and k.
    num_v_heads: int  # Hv.
    head_dim: int = 128  # D, shared by key and value heads (128 in both models).
    fp8: bool = True
    act_quant_block: int = 1024  # As AttentionConfig.act_quant_block.
    input_norm: Norm = Norm.RMS_ONE_PLUS_W
    residual: bool = True
    eps: float = 1e-6
    # Precision of the input norm's x * w. vLLM uses f32; bf16 is the default
    # because it is ~3-5% faster at T >= 1024.
    norm_dtype: jnp.dtype = jnp.bfloat16

    # W (in_proj_qkvz) is [q | k | v | z] (vLLM gqa_interleaved_layout=False):
    # q and k are num_k_heads heads each, v and z num_v_heads heads each.
    # side_w (in_proj_ba) is [b | a], num_v_heads columns each.

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_dim

    @property
    def qkv_cols(self) -> int:  # mixed_qkv width.
        return 2 * self.key_dim + self.value_dim

    @property
    def z_col(self) -> int:
        return self.qkv_cols

    @property
    def num_cols(self) -> int:  # N.
        return self.qkv_cols + self.value_dim

    @property
    def side_cols(self) -> int:
        return 2 * self.num_v_heads


class QkvWeights(NamedTuple):
    """Projection and norm weights as loaded, before rpa/gdn.prepare_weights."""

    w: jax.Array  # (K, N) fp8 or bf16.
    w_scale: jax.Array | None = None  # (1, N) f32, fp8 only.
    bias: jax.Array | None = None  # (1, N).
    q_norm_w: jax.Array | None = None  # (1, D).
    k_norm_w: jax.Array | None = None  # (1, D).
    input_norm_w: jax.Array | None = None  # (1, K).
    # GDN only: in_proj_ba (K, 2 Hv) bf16, [b | a], an unquantized projection of
    # the same normed x.
    side_w: jax.Array | None = None
