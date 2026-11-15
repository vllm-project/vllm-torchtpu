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
"""Pure-JAX helper ops for the GLM-5.3-Flash decode megakernel fast path.

- `quantize_kv_latent_fp8` / `dequantize_kv_latent_fp8`: 2-component FP8
  MLA latent quantization and dequantization.
- `latent_write_local` / `flush_latents_local` / `copy_subpages_local`:
  quantize and scatter MLA latents into the megakernel superpage pool.
- `kda_pool_regions` / `kda_state_from_pool_local`: copy a request's KDA
  conv (`float32` in the unified pool) and recurrent (`float32`) states from
  vLLM's unified block pool into the megakernel's per-slot state buffers.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


def quantize_kv_latent_fp8(
    kv_latent: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Quantize `[T, kv]` MLA latent to 2-component float8_e4m3fn + scale."""
    c_f32 = jnp.nan_to_num(
        kv_latent.astype(jnp.float32), nan=0.0, posinf=448.0, neginf=-448.0
    )
    s_hi = jnp.maximum(jnp.max(jnp.abs(c_f32), axis=-1) / 448.0, jnp.float32(2.0**-14))
    q_hi = jnp.clip(c_f32 / s_hi[:, None], -448.0, 448.0).astype(jnp.float8_e4m3fn)
    rem = c_f32 - q_hi.astype(jnp.float32) * s_hi[:, None]
    s_lo = jnp.maximum(jnp.max(jnp.abs(rem), axis=-1) / 448.0, jnp.float32(2.0**-24))
    q_lo = jnp.clip(rem / s_lo[:, None], -448.0, 448.0).astype(jnp.float8_e4m3fn)
    c_q = jnp.stack([q_hi, q_lo], axis=-2)
    scale = jnp.stack([s_hi, s_lo], axis=-1)
    return c_q, scale


def dequantize_kv_latent_fp8(
    c_q: jax.Array, scale: jax.Array, out_dtype: Any = jnp.bfloat16
) -> jax.Array:
    """Dequantize 2-component (`float8_e4m3fn [..., 2, kv]`, `float32 [..., 2]`)."""
    sf = scale.astype(jnp.float32)
    hi = c_q[..., 0, :].astype(jnp.float32) * sf[..., 0:1]
    lo = c_q[..., 1, :].astype(jnp.float32) * sf[..., 1:2]
    return (hi + lo).astype(out_dtype)


def latent_write_local(
    kvq: jax.Array,
    kvs: jax.Array,
    latent: jax.Array,
    tok_slot: jax.Array,
    tok_pos: jax.Array,
    kpt: jax.Array,
    rank: jax.Array,
    layer: int,
    *,
    page_size: int,
    world: int,
) -> tuple[jax.Array, jax.Array]:
    """Quantize `latent [T, kv]` and write this rank's rows into `layer`.

    Token `t` goes to slot `tok_slot[t]` at position `tok_pos[t]`;
    `tok_slot[t] < 0` skips it. An `int32` word packs 4 rows (byte `k` is row
    `4w + k`, `kernel.kv_words`); each token adds `(new_byte - old_byte) << 8k`
    to its word modulo `2**32`.
    """
    from .kernel import KV_GROUP, kernel_cache_token_loc

    t = latent.shape[0]
    tok_slot, tok_pos = tok_slot[:t], tok_pos[:t]
    kvq_l, kvs_l = kvq[0], kvs[0]
    c_q, c_s = quantize_kv_latent_fp8(latent)
    owner, grp, row = kernel_cache_token_loc(tok_pos, page_size, world, KV_GROUP)
    mine = (tok_slot >= 0) & (owner == rank[0]) & (grp < kpt.shape[1])
    phys = kpt[
        jnp.clip(tok_slot, 0, kpt.shape[0] - 1),
        jnp.clip(grp, 0, kpt.shape[1] - 1),
    ]
    tgt_p = jnp.where(mine, phys, kvq_l.shape[1])
    widx = row // 4
    shift = (8 * (row % 4)).astype(jnp.uint32)[:, None]
    new_b = jax.lax.bitcast_convert_type(c_q.reshape(t, -1), jnp.uint8).astype(
        jnp.uint32
    )
    cur = jax.lax.bitcast_convert_type(
        kvq_l[layer, jnp.where(mine, phys, 0), widx], jnp.uint32
    )
    old_b = (cur >> shift) & jnp.uint32(0xFF)
    delta = jax.lax.bitcast_convert_type((new_b << shift) - (old_b << shift), jnp.int32)
    kvq_l = kvq_l.at[layer, tgt_p, widx].add(delta, mode="drop")
    kvs_l = kvs_l.at[layer, tgt_p, 0, row].set(c_s[:, 0], mode="drop")
    kvs_l = kvs_l.at[layer, tgt_p, 1, row].set(c_s[:, 1], mode="drop")
    return kvq_l[None], kvs_l[None]


def flush_latents_local(
    kvq: jax.Array,
    kvs: jax.Array,
    stash: jax.Array,
    tok_slot: jax.Array,
    tok_pos: jax.Array,
    kpt: jax.Array,
    rank: jax.Array,
    *,
    layers: tuple[int, ...],
    n_tokens: int,
    page_size: int,
    world: int,
    tok_idx: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Write stashed latents `stash [Lm, Tmax, kv]` of `layers` into the pool.

    Token locations are computed once, `quantize_kv_latent_fp8` runs vectorized
    across `layers` (`[nl, T, kv]`), and only the `(nl, T)` touched words are
    gathered and scatter-added directly into `kvq` / `kvs` in place so buffer
    donation aliases `kvq` and `kvs` without slicing `kvq` across layers.
    """
    from .kernel import KV_GROUP, kernel_cache_token_loc

    nl = len(layers)
    if nl == 0 or n_tokens == 0:
        return kvq, kvs
    t = n_tokens
    tok_slot, tok_pos = tok_slot[:t], tok_pos[:t]
    kvq_l = kvq.reshape(kvq.shape[1:])
    kvs_l = kvs.reshape(kvs.shape[1:])
    owner, grp, row = kernel_cache_token_loc(tok_pos, page_size, world, KV_GROUP)
    mine = (tok_slot >= 0) & (owner == rank[0]) & (grp < kpt.shape[1])
    phys = kpt[
        jnp.clip(tok_slot, 0, kpt.shape[0] - 1),
        jnp.clip(grp, 0, kpt.shape[1] - 1),
    ]
    safe_p = jnp.where(mine, phys, 0)
    tgt_p = jnp.where(mine, phys, kvq_l.shape[1])
    widx = row // 4
    shift = (8 * (row % 4)).astype(jnp.uint32)[None, :, None]

    lay_arr = jnp.asarray(layers, dtype=jnp.int32)
    l0 = layers[0]
    if layers == tuple(range(l0, l0 + nl)):
        lat_slice = jax.lax.dynamic_slice_in_dim(stash, l0, nl, axis=0)
    else:
        lat_slice = stash[lay_arr]
    lat_sub = lat_slice[:, tok_idx[:t]] if tok_idx is not None else lat_slice[:, :t]
    c_q, c_s = jax.vmap(quantize_kv_latent_fp8)(lat_sub)
    new_b = jax.lax.bitcast_convert_type(c_q.reshape(nl, t, -1), jnp.uint8).astype(
        jnp.uint32
    )
    lay_col = lay_arr[:, None]
    cur = jax.lax.bitcast_convert_type(
        kvq_l[lay_col, safe_p[None, :], widx[None, :]], jnp.uint32
    )
    old_b = (cur >> shift) & jnp.uint32(0xFF)
    delta = jax.lax.bitcast_convert_type((new_b << shift) - (old_b << shift), jnp.int32)
    b_lay, b_p, b_w = jnp.broadcast_arrays(lay_col, tgt_p[None, :], widx[None, :])
    kvq_l = kvq_l.at[b_lay, b_p, b_w].add(delta, mode="drop")
    _, _, b_row = jnp.broadcast_arrays(lay_col, tgt_p[None, :], row[None, :])
    kvs_l = kvs_l.at[b_lay, b_p, 0, b_row].set(c_s[:, :, 0], mode="drop")
    kvs_l = kvs_l.at[b_lay, b_p, 1, b_row].set(c_s[:, :, 1], mode="drop")
    return kvq_l.reshape(kvq.shape), kvs_l.reshape(kvs.shape)


def copy_subpages_local(
    kvq: jax.Array, kvs: jax.Array, pairs: jax.Array, *, kv_group: int = 8
) -> tuple[jax.Array, jax.Array]:
    """Copy subpages `pairs[0, i] -> pairs[1, i]` across all MLA layers.

    `subpage_id = kpage * kv_group + g`; entries with `pairs[1, i] < 0` are
    dropped (`mode="drop"`).
    """
    nl, nkp, w_tot, kv2 = kvq.shape[1:]
    prow4 = w_tot // kv_group
    prow = kvs.shape[-1] // kv_group
    nsub = nkp * kv_group
    src, dst = pairs[0], pairs[1]
    valid = (src >= 0) & (dst >= 0) & (src < nsub) & (dst < nsub)
    s_idx = jnp.where(valid, src, 0)
    d_idx = jnp.where(valid, dst, nsub)
    q_sub = kvq[0].reshape(nl, nsub, prow4, kv2)
    q_sub = q_sub.at[:, d_idx].set(q_sub[:, s_idx], mode="drop")
    s_sub = jnp.swapaxes(kvs[0].reshape(nl, nkp, 8, kv_group, prow), 2, 3).reshape(
        nl, nsub, 8, prow
    )
    s_sub = s_sub.at[:, d_idx].set(s_sub[:, s_idx], mode="drop")
    s_out = jnp.swapaxes(s_sub.reshape(nl, nkp, kv_group, 8, prow), 2, 3).reshape(
        1, nl, nkp, 8, kv_group * prow
    )
    return q_sub.reshape(1, nl, nkp, w_tot, kv2), s_out


def kda_pool_regions(
    pool_shape: tuple[int, ...],
    pool_dtype: Any,
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: int,
    pool_block_tokens: int,
    conv_elem_bytes: int = 4,
) -> tuple[Any, int]:
    """(`PooledGDNStateLayout`, manager-to-pool block split) for GLM-5.3 KDA.

    Both SSM state and conv state are stored as `float32` (4 bytes/element) in
    the unified block pool (`DEFAULT_POOLED_GDN_CONV_STATE_DTYPE`), regardless
    of the pool tensor dtype (an FP8 KV cache pool holds them as raw bytes).
    """
    import math

    from vllm_torchtpu.gdn_pool_layout import derive_pooled_gdn_state_layout

    tok_bytes = math.prod(pool_shape[2:]) * jnp.dtype(pool_dtype).itemsize
    layout = derive_pooled_gdn_state_layout(
        ssm_bytes=num_heads * head_dim * head_dim * 4,
        conv_bytes=(kernel_size - 1) * 3 * num_heads * head_dim * conv_elem_bytes,
        token_bytes=tok_bytes,
    )
    if pool_block_tokens % pool_shape[1]:
        raise ValueError(
            f"manager block {pool_block_tokens} is not whole pool blocks of "
            f"{pool_shape[1]}"
        )
    return layout, pool_block_tokens // pool_shape[1]


def kda_state_from_pool_local(
    pools: list[jax.Array],
    block_ids: jax.Array,
    conv: jax.Array,
    rec: jax.Array,
    slots: jax.Array,
    *,
    num_heads: int,
    head_dim: int,
    kernel_size: int,
    pool_block_tokens: int,
    conv_pool_dtype: Any = jnp.float32,
) -> tuple[jax.Array, jax.Array]:
    """Copy requests' KDA states from vLLM's unified pool into megakernel slots.

    `pools[i]` is KDA layer `i`'s pool `(num_blocks, block_size, *payload, lanes)`;
    `block_ids [N, L_kda]` holds the manager block for each request's state per
    layer; `slots [N]` holds the megakernel slots. `conv [1, L_kda, B, K-1, 3*H*D]`
    (`bfloat16`) and `rec [1, L_kda, B, H, D, D]` (`float32`) are updated in place.
    """
    from vllm_torchtpu.kernels import pool_adapters

    n = slots.shape[0]
    conv_l, rec_l = conv[0], rec[0]
    conv_elems = (kernel_size - 1) * 3 * num_heads * head_dim
    conv_elem_bytes = jnp.dtype(conv_pool_dtype).itemsize
    for li, pool in enumerate(pools):
        layout, split = kda_pool_regions(
            pool.shape,
            pool.dtype,
            num_heads=num_heads,
            head_dim=head_dim,
            kernel_size=kernel_size,
            pool_block_tokens=pool_block_tokens,
            conv_elem_bytes=conv_elem_bytes,
        )
        ids = block_ids[:, li]
        ssm = pool_adapters.gather_region(
            pool,
            ids,
            tok0=0,
            ntok=layout.ssm_tokens,
            out_dtype=jnp.float32,
            out_lanes=head_dim,
            split=split,
        )
        ssm = ssm[:, : num_heads * head_dim, :].reshape(
            n, num_heads, head_dim, head_dim
        )
        cv = pool_adapters.gather_region(
            pool,
            ids,
            tok0=layout.ssm_tokens,
            ntok=layout.conv_tokens,
            out_dtype=conv_pool_dtype,
            split=split,
        )
        cv = cv.reshape(n, -1)[:, :conv_elems].reshape(
            n, kernel_size - 1, 3 * num_heads * head_dim
        )
        rec_l = rec_l.at[li, slots].set(ssm.astype(rec_l.dtype))
        conv_l = conv_l.at[li, slots].set(cv.astype(conv_l.dtype))
    return conv_l[None], rec_l[None]
