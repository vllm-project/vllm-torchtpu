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
"""TurboQuant VREG unpacking and split-plane FP8 attention for sparse MLA."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.mla import kv_cache_utils
from vllm_torchtpu.layers.core import turboquant as tq

# 256 B per 4-bit latent token = 64 uint32 words (2 tokens per 128-word SC row).
TQ_NOPE_WORDS = tq.MLA_TQ_NOPE_BYTES // kv_cache_utils.WORD_BYTES
# Unroll factor for the TurboQuant per-token attention loop.
TQ_UNROLL = 2


def unpack_tq_nope(bkv_nope_ref: Any, bkv_sz: int) -> tuple[jax.Array, jax.Array]:
    """Unpacks 4-bit NoPE codes in VMEM into `(lo, hi)` `fp8[bkv_sz, 256]` planes."""
    half = tq.MLA_TQ_NOPE_BYTES
    fp8_lo, fp8_hi = tq.unpack_mla_fp8_nibbles_u32(bkv_nope_ref[...])
    return (
        pltpu.bitcast(fp8_lo, jnp.float8_e4m3fn).reshape(bkv_sz, half),
        pltpu.bitcast(fp8_hi, jnp.float8_e4m3fn).reshape(bkv_sz, half),
    )


def unpack_tq_rope_and_norm(
    bkv_rope_ref: Any, bkv_sz: int
) -> tuple[jax.Array, jax.Array]:
    """Extracts `fp8[bkv_sz, 128]` RoPE and `f32[bkv_sz, 1]` norm from a RoPE tile."""
    rope_i32 = pltpu.bitcast(bkv_rope_ref[...], jnp.int32)
    off = tq.MLA_TQ_NORM_BYTE_OFFSET
    lane_i32 = lax.broadcasted_iota(jnp.int32, rope_i32.shape, 1)
    lo_i32 = jnp.sum(jnp.where(lane_i32 == off, rope_i32, 0), axis=1, keepdims=True)
    hi_i32 = jnp.sum(jnp.where(lane_i32 == off + 1, rope_i32, 0), axis=1, keepdims=True)
    lo_u8 = pltpu.bitcast(jnp.broadcast_to(lo_i32, rope_i32.shape), jnp.uint8).reshape(
        bkv_sz, 128
    )
    hi_u8 = pltpu.bitcast(jnp.broadcast_to(hi_i32, rope_i32.shape), jnp.uint8).reshape(
        bkv_sz, 128
    )
    norm_col = tq._bf16_u16_to_f32_vreg(
        lo_u8.astype(jnp.uint32), hi_u8.astype(jnp.uint32)
    )[:, :1]
    rope_pe_i32 = jnp.where(lane_i32 < off, rope_i32, 0)
    rope = pltpu.bitcast(rope_pe_i32, jnp.float8_e4m3fn).reshape(bkv_sz, 128)
    return rope, norm_col


def unpack_tq_bkv(
    bkv_nope_ref: Any,
    bkv_rope_ref: Any,
    get_kv_len: Callable[[], jax.Array],
    *,
    bkv_sz: int,
    rope_scale: float,
    head_dim: int,
) -> jax.Array:
    """Materializes a concatenated `bf16[bkv_sz, head_dim]` KV tile (overflow fixup)."""
    k_lo, k_hi = unpack_tq_nope(bkv_nope_ref, bkv_sz)
    rope, norm_col = unpack_tq_rope_and_norm(bkv_rope_ref, bkv_sz)
    kv_len = get_kv_len()
    k_row = lax.broadcasted_iota(jnp.int32, norm_col.shape, 0)
    norm_col = jnp.where(k_row < kv_len, norm_col, 0.0)
    nope = (
        jnp.concatenate([k_lo, k_hi], axis=-1).astype(jnp.float32) * norm_col
    ).astype(jnp.bfloat16)
    rope_bf16 = (
        (rope.astype(jnp.float32) * rope_scale).astype(jnp.bfloat16)
        if rope_scale != 1.0
        else rope.astype(jnp.bfloat16)
    )
    bkv = jnp.concatenate([nope, rope_bf16], axis=-1)
    if bkv.shape[-1] < head_dim:
        bkv = jnp.pad(bkv, ((0, 0), (0, head_dim - bkv.shape[-1])))
    k_span = lax.broadcasted_iota(jnp.int32, bkv.shape, 0)
    return jnp.where(k_span < kv_len, bkv, 0)


def tq_split_entry(
    bkv_nope_ref: Any,
    bkv_rope_ref: Any,
    q_nope: jax.Array,
    q_pe_128: jax.Array,
    kv_len: jax.Array,
    *,
    bkv_sz: int,
    rope_scale: float,
    sm_scale: float,
    q_dtype: jnp.dtype,
    o_dim: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """TurboQuant attention for one query token without K/V concats.

    The 512-dim latent arrives as two `[bkv_sz, 256]` nibble planes (codes
    `0..255` and `256..511`); scores are the sum of three partial FP8 matmuls
    (lo, hi, rope) and the value product skips the rope tail entirely.
    """
    rope, norm_col = unpack_tq_rope_and_norm(bkv_rope_ref, bkv_sz)
    k_lo, k_hi = unpack_tq_nope(bkv_nope_ref, bkv_sz)
    half = k_lo.shape[1]

    def mm(a, b):
        return jnp.einsum("nd,md->nm", a, b, preferred_element_type=jnp.float32)

    s_rope = mm(q_pe_128, rope)
    if rope_scale != 1.0:
        s_rope = s_rope * rope_scale
    s_nope = mm(q_nope[:, :half], k_lo) + mm(q_nope[:, half : 2 * half], k_hi)
    kv_span = lax.broadcasted_iota(jnp.int32, s_nope.shape, 1)
    norm_row = jnp.where(kv_span[:1, :] < kv_len, jnp.swapaxes(norm_col, 0, 1), 0.0)
    s = (s_nope * norm_row + s_rope) * sm_scale
    s = jnp.where(kv_span < kv_len, s, jnp.finfo(jnp.float32).min)
    m = jnp.max(s, axis=1, keepdims=True)
    p = jnp.exp(s - m)
    l_sum = jnp.sum(p, axis=1, keepdims=True)
    pb = (p * norm_row).astype(jnp.bfloat16)

    def pv(b):
        return jnp.einsum("nm,md->nd", pb, b, preferred_element_type=jnp.float32)

    acc = jnp.concatenate([pv(k_lo), pv(k_hi)], axis=-1)
    out = (
        lax.div(acc, l_sum)
        if q_dtype == jnp.float32
        else (acc * pl.reciprocal(l_sum, approx=True)).astype(q_dtype)
    )
    if o_dim > out.shape[-1]:
        out = jnp.concatenate(
            [out, jnp.zeros((out.shape[0], o_dim - out.shape[-1]), q_dtype)],
            axis=-1,
        )
    return out[:, :o_dim], l_sum, m


def run_tq_batch(
    *,
    batch_size: int,
    batch_start_seq_idx: jax.Array,
    bi_sem_idx: jax.Array,
    bo_sem_idx: jax.Array,
    split_q: bool,
    kv_lens_ref: Any,
    bq_x2_ref: Any,
    bq_pe_x2_ref: Any,
    load_bq: Callable[[jax.Array, jax.Array], jax.Array],
    bkv_nope_x2_ref: Any,
    bkv_rope_x2_ref: Any,
    bo_x2_ref: Any,
    write_blse: Callable[[jax.Array, jax.Array, jax.Array], None],
    bkv_sz: int,
    rope_scale: float,
    sm_scale: float,
    q_dtype: jnp.dtype,
    o_dim: int,
) -> None:
    """Runs `tq_split_entry` across `batch_size` query tokens with bounded unroll."""

    def tq_entry(batch_idx, carry):
        kv_len = kv_lens_ref[batch_start_seq_idx + batch_idx]
        if split_q:
            q_nope = bq_x2_ref.at[bi_sem_idx, batch_idx][...]
            assert bq_pe_x2_ref is not None
            q_pe_raw = bq_pe_x2_ref.at[bi_sem_idx, batch_idx][...]
            q_pe_128 = (
                jnp.pad(
                    q_pe_raw,
                    ((0, 0), (0, 128 - q_pe_raw.shape[-1])),
                )
                if q_pe_raw.shape[-1] < 128
                else q_pe_raw[:, :128]
            )
        else:
            bq = load_bq(bi_sem_idx, batch_idx)
            q_nope = bq[:, :512]
            q_pe_128 = bq[:, 512:640]
        out, l_sum, m = tq_split_entry(
            bkv_nope_x2_ref.at[bi_sem_idx, batch_idx],
            bkv_rope_x2_ref.at[bi_sem_idx, batch_idx],
            q_nope,
            q_pe_128,
            kv_len,
            bkv_sz=bkv_sz,
            rope_scale=rope_scale,
            sm_scale=sm_scale,
            q_dtype=q_dtype,
            o_dim=o_dim,
        )
        bo_x2_ref.at[bo_sem_idx, batch_idx][...] = out
        write_blse(batch_idx, m, l_sum)
        return carry

    lax.fori_loop(0, batch_size, tq_entry, 0, unroll=min(TQ_UNROLL, batch_size))
