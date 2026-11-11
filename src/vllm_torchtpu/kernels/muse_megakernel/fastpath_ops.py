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
"""Helper JAX/Pallas ops for Muse Spark megakernel KV handoff, KV block writeback,
and shared-store prefill MoE.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import F32, KV_GROUP, PAGE_TOKENS, SUBPAGE_TOKENS, SW_RING, TP_AXIS_NAME


def writeback_blocks_local(
    pool: jax.Array,
    kv_sw: jax.Array,
    kv_gl: jax.Array,
    kv_draft: jax.Array,
    meta: jax.Array,
    lmeta: jax.Array,
) -> jax.Array:
    """Copy whole 128-token blocks of megakernel KV into one vLLM paged pool.

    pool: [num_blocks, 128, ...] with 128 * kvwidth elements per block row, shared
      by the layers in `lmeta`.
    kv_sw / kv_draft: [1, L, slots, SW_RING, kvwidth] rings; kv_gl: [1, L_gl,
      num_kpages, PAGE_TOKENS, kvwidth].
    meta: [4 + G, B] int32 rows (src kv_gl subpage, slot, ring offset of the
      block's first token, draft ready flag), then one vLLM block id per KV cache
      group. A negative block id, source subpage or slot skips the entry.
    lmeta: [L, 3] int32 rows (kind, layer index, group row) for each layer stored
      in `pool`; kind 0 reads kv_gl, 1 kv_sw, 2 kv_draft, -1 is padding.
    """
    _, n_gl, n_kp, _, kvwidth = kv_gl.shape
    total_sub = n_kp * KV_GROUP
    gl_flat = kv_gl.reshape(1, n_gl, total_sub, SUBPAGE_TOKENS, kvwidth)
    num_blocks = pool.shape[0]
    b = meta.shape[1]
    src_sub = meta[0]
    slot = meta[1]
    rows = (
        jnp.clip(meta[2], 0, SW_RING - SUBPAGE_TOKENS)[:, None]
        + jnp.arange(SUBPAGE_TOKENS, dtype=jnp.int32)[None, :]
    )
    src_c = jnp.clip(src_sub, 0, total_sub - 1)

    def ring_rows(ring: jax.Array, idx: jax.Array) -> jax.Array:
        n_l, n_slots = ring.shape[1], ring.shape[2]
        slot_c = jnp.clip(slot, 0, n_slots - 1)
        return ring[0, jnp.clip(idx, 0, n_l - 1), slot_c[:, None], rows]

    branches = (
        lambda idx: gl_flat[0, jnp.clip(idx, 0, n_gl - 1), src_c],
        lambda idx: ring_rows(kv_sw, idx),
        lambda idx: ring_rows(kv_draft, idx),
    )
    for j in range(lmeta.shape[0]):
        kind, idx, g_row = lmeta[j, 0], lmeta[j, 1], lmeta[j, 2]
        vals = jax.lax.switch(jnp.clip(kind, 0, 2), branches, idx)
        blk = meta[4 + jnp.clip(g_row, 0, meta.shape[0] - 5)]
        src_ok = jnp.where(kind == 0, src_sub >= 0, slot >= 0)
        draft_ok = (kind != 2) | (meta[3] != 0)
        valid = (kind >= 0) & (blk >= 0) & (blk < num_blocks) & src_ok & draft_ok
        tgt = jnp.where(valid, blk, num_blocks)
        pool = pool.at[tgt].set(
            vals.reshape(b, *pool.shape[1:]).astype(pool.dtype), mode="drop"
        )
    return pool


def flush_vllm_kv_to_hybrid_local(
    kv_sw: jax.Array,
    kv_gl: jax.Array,
    kpt: jax.Array,
    meta: jax.Array,
    sw_pools: list[jax.Array],
    gl_pools: list[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Copy token KV entries from vLLM's paged KV pools into kv_sw and kv_gl.

    meta: [3, T] int32 with rows (slot, pos, vllm_block_id); slot < 0 is a no-op.
    Each pool in sw_pools / gl_pools has shape [num_blocks, 128, 1, 2, 128] bfloat16.
    """
    n_tok = meta.shape[1]
    max_slots, max_kp = kpt.shape
    num_kpages = kv_gl.shape[2]
    kvwidth = kv_sw.shape[-1]

    sw_stack = jnp.stack(
        [p.reshape(-1, SUBPAGE_TOKENS, kvwidth) for p in sw_pools], axis=0
    )
    gl_stack = jnp.stack(
        [p.reshape(-1, SUBPAGE_TOKENS, kvwidth) for p in gl_pools], axis=0
    )
    num_blocks = sw_stack.shape[1]

    def step(t, carry):
        sw_buf, gl_buf = carry
        slot = meta[0, t]
        pos = meta[1, t]
        blk = meta[2, t]
        valid = (
            (slot >= 0)
            & (slot < max_slots)
            & (pos >= 0)
            & (blk >= 0)
            & (blk < num_blocks)
        )
        slot_c = jnp.clip(slot, 0, max_slots - 1)
        pos_c = jnp.clip(pos, 0, max_kp * PAGE_TOKENS - 1)
        blk_c = jnp.clip(blk, 0, num_blocks - 1)
        blk_off = pos_c % SUBPAGE_TOKENS
        ring_idx = pos_c % SW_RING
        chunk_idx = pos_c // PAGE_TOKENS
        page_off = pos_c % PAGE_TOKENS
        kp = jnp.clip(kpt[slot_c, chunk_idx], 0, num_kpages - 1)

        sw_val = sw_stack[:, blk_c, blk_off]
        cur_sw = sw_buf[0, :, slot_c, ring_idx]
        sw_buf = sw_buf.at[0, :, slot_c, ring_idx].set(jnp.where(valid, sw_val, cur_sw))

        gl_val = gl_stack[:, blk_c, blk_off]
        cur_gl = gl_buf[0, :, kp, page_off]
        gl_buf = gl_buf.at[0, :, kp, page_off].set(jnp.where(valid, gl_val, cur_gl))
        return sw_buf, gl_buf

    return jax.lax.fori_loop(0, n_tok, step, (kv_sw, kv_gl))


def handoff_sw_ring_local(
    kv_sw: jax.Array,
    pools: list[jax.Array],
    blk_ids: jax.Array,
    blk_offs: jax.Array,
    slot_arr: jax.Array,
    *,
    sw_pool_of: tuple[int, ...],
    sw_gid_idx_of: tuple[int, ...],
) -> jax.Array:
    """Gather the 2048-token sliding-window ring for `slot_arr[0]` from vLLM's shared KV
    pools.

    blk_ids: [num_sw_groups, 2048] int32 vLLM block IDs for each ring index 0..2047.
    blk_offs: [2048] int32 block offsets 0..127 (-1 for unwritten ring slots when
    seq_len < 2048).
    """
    kvwidth = kv_sw.shape[-1]
    slot = slot_arr[0]
    valid = (blk_offs >= 0)[:, None]
    offs_c = jnp.clip(blk_offs, 0, SUBPAGE_TOKENS - 1)
    num_blocks = pools[0].shape[0]
    clamped_ids = jnp.clip(blk_ids, 0, num_blocks - 1)
    zero_ring = jnp.zeros((SW_RING, kvwidth), kv_sw.dtype)

    layers = []
    for sw_idx in range(len(sw_pool_of)):
        p = pools[sw_pool_of[sw_idx]]
        b = clamped_ids[sw_gid_idx_of[sw_idx]]
        gathered = p[b, offs_c].reshape(SW_RING, kvwidth)
        layers.append(jnp.where(valid, gathered, zero_ring))
    stacked = jnp.stack(layers, axis=0)  # [46, 2048, 256]
    return kv_sw.at[0, :, slot].set(stacked)


def handoff_gl_subpages_local(
    kv_gl: jax.Array,
    pools: list[jax.Array],
    pairs: jax.Array,
    *,
    gl_pool_of: tuple[int, ...],
) -> jax.Array:
    """Copy 128-token blocks from vLLM's shared KV pools into kv_gl subpages.

    pairs: [2, B] int32 of (vllm_block_id, dst_subpage); negative entries are no-ops.
    """
    _, n_gl, n_kp, _, kvwidth = kv_gl.shape
    total_sub = n_kp * KV_GROUP
    flat = kv_gl.reshape(1, n_gl, total_sub, SUBPAGE_TOKENS, kvwidth)
    num_blocks = pools[0].shape[0]
    b = pairs.shape[1]
    s_all = jnp.clip(pairs[0], 0, num_blocks - 1)
    gathered = jnp.stack(
        [
            pools[gl_pool_of[i]][s_all].reshape(b, SUBPAGE_TOKENS, kvwidth)
            for i in range(n_gl)
        ],
        axis=0,
    )

    def body(i, buf):
        d_c = jnp.clip(pairs[1, i], 0, total_sub - 1)
        return buf.at[0, :, d_c].set(gathered[:, i])

    n_valid = jnp.sum(((pairs[0] >= 0) & (pairs[1] >= 0)).astype(jnp.int32))
    flat = jax.lax.fori_loop(0, n_valid, body, flat)
    return flat.reshape(1, n_gl, n_kp, PAGE_TOKENS, kvwidth)


def sync_seq_kv_all_layers_local(
    kv_sw: jax.Array,
    kv_gl: jax.Array,
    pools: list[jax.Array],
    sw_quads: jax.Array,
    gl_pairs: jax.Array,
    *,
    sw_pool_of: tuple[int, ...],
    sw_gid_idx_of: tuple[int, ...],
    gl_pool_of: tuple[int, ...],
    n_sw_groups: int,
) -> tuple[jax.Array, jax.Array]:
    """Single-dispatch fused prefill-to-decode KV bridge across all 46 SW and 16 Global
    layers.

    sw_quads: [2 * n_sw_groups + 3, B_sw] int32 with rows:
      - [0 .. G-1]: blk_lo per SW KV group
      - [G .. 2G-1]: blk_hi per SW KV group (for wraparound subpage when seq_len > 2048)
      - [2G]: split_off in 0..128 (rows < split_off read blk_lo, >= split_off read
        blk_hi)
      - [2G + 1]: valid_len in 0..128 (rows >= valid_len are zeroed)
      - [2G + 2]: dst_sw_sub in 0..(max_slots * 16 - 1), or < 0 for no-op.
    gl_pairs: [2, B_gl] int32 of (vllm_block_id, dst_gl_subpage) covering both active
    slot
      subpages and prefix-cache ring subpages; negative entries are no-ops.
    """
    kv_sw_out = sync_sw_rings_local(
        kv_sw,
        pools,
        sw_quads,
        sw_pool_of=sw_pool_of,
        sw_gid_idx_of=sw_gid_idx_of,
        n_sw_groups=n_sw_groups,
    )
    kv_gl_out = handoff_gl_subpages_local(kv_gl, pools, gl_pairs, gl_pool_of=gl_pool_of)
    return kv_sw_out, kv_gl_out


def sync_sw_rings_local(
    kv_sw: jax.Array,
    pools: list[jax.Array],
    sw_quads: jax.Array,
    *,
    sw_pool_of: tuple[int, ...],
    sw_gid_idx_of: tuple[int, ...],
    n_sw_groups: int,
) -> jax.Array:
    """Rebuild 2048-token rings [1, L, slots, 2048, w] from vLLM pools (sw_quads as
    above).
    """
    _, n_sw, max_slots, _, kvwidth = kv_sw.shape
    num_blocks = pools[0].shape[0]
    b_sw = sw_quads.shape[1]
    g = n_sw_groups

    lo_ids = jnp.clip(sw_quads[:g], 0, num_blocks - 1)
    hi_ids = jnp.clip(sw_quads[g : 2 * g], 0, num_blocks - 1)
    splits = sw_quads[2 * g]
    valids = sw_quads[2 * g + 1]
    dst_sw = sw_quads[2 * g + 2]

    offs = jnp.arange(SUBPAGE_TOKENS, dtype=jnp.int32)[None, :, None]
    use_lo = offs < splits[:, None, None]
    is_valid_tok = offs < valids[:, None, None]
    zero_sub = jnp.zeros((b_sw, SUBPAGE_TOKENS, kvwidth), kv_sw.dtype)

    sw_gathered_layers = []
    for sw_idx in range(n_sw):
        p = pools[sw_pool_of[sw_idx]]
        gid_i = sw_gid_idx_of[sw_idx]
        g_lo = p[lo_ids[gid_i]].reshape(b_sw, SUBPAGE_TOKENS, kvwidth)
        g_hi = p[hi_ids[gid_i]].reshape(b_sw, SUBPAGE_TOKENS, kvwidth)
        blended = jnp.where(use_lo, g_lo, g_hi)
        sw_gathered_layers.append(jnp.where(is_valid_tok, blended, zero_sub))
    sw_gathered = jnp.stack(sw_gathered_layers, axis=0)

    sw_sub_per_slot = SW_RING // SUBPAGE_TOKENS
    n_slots_b = b_sw // sw_sub_per_slot
    sw_by_slot = sw_gathered.reshape(n_sw, n_slots_b, SW_RING, kvwidth)

    def sw_slot_body(k, buf):
        slot = jnp.clip(
            dst_sw[k * sw_sub_per_slot] // sw_sub_per_slot, 0, max_slots - 1
        )
        return buf.at[0, :, slot].set(sw_by_slot[:, k])

    n_valid_slots = jnp.sum((dst_sw[::sw_sub_per_slot] >= 0).astype(jnp.int32))
    return jax.lax.fori_loop(0, n_valid_slots, sw_slot_body, kv_sw)


def gather_step_kv_local(
    kv_sw: jax.Array,
    kv_gl: jax.Array,
    gl_loc: jax.Array,
    sw_loc: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Gather newly decoded KV rows `[L_gl, B, kvwidth]` and `[L_sw, B, kvwidth]` from
    `kv_gl` and `kv_sw`.

    gl_loc: [L_gl, B, 2] int32 of (kp, page_off)
    sw_loc: [L_sw, B, 2] int32 of (slot, ring_idx)
    """
    max_slots = kv_sw.shape[2]
    num_kpages = kv_gl.shape[2]
    n_gl = gl_loc.shape[0]
    n_sw = sw_loc.shape[0]

    gl_rows = []
    for gl_idx in range(n_gl):
        kp = jnp.clip(gl_loc[gl_idx, :, 0], 0, num_kpages - 1)
        poff = jnp.clip(gl_loc[gl_idx, :, 1], 0, PAGE_TOKENS - 1)
        gl_rows.append(kv_gl[0, gl_idx, kp, poff])

    sw_rows = []
    for sw_idx in range(n_sw):
        slot = jnp.clip(sw_loc[sw_idx, :, 0], 0, max_slots - 1)
        ridx = jnp.clip(sw_loc[sw_idx, :, 1], 0, SW_RING - 1)
        sw_rows.append(kv_sw[0, sw_idx, slot, ridx])

    return jnp.stack(gl_rows, axis=0), jnp.stack(sw_rows, axis=0)


def scatter_pool_kv_local(
    pool: jax.Array,
    gl_vals: jax.Array,
    sw_vals: jax.Array,
    meta: jax.Array,
) -> jax.Array:
    """Scatter up to 4 layers x B batch items of decoded KV rows into a single vLLM
    shared KV pool.

    pool: [num_blocks, 128, 1, 2, 128] bfloat16
    gl_vals: [L_gl, B, kvwidth] bfloat16
    sw_vals: [L_sw, B, kvwidth] bfloat16
    meta: [4, B, 4] int32 of (is_sw, layer_idx, vllm_block_id, blk_off); vllm_block_id <
    0 is a no-op.
    """
    orig_shape = pool.shape
    num_blocks = orig_shape[0]
    elem_shape = orig_shape[2:]
    n_gl = gl_vals.shape[0]
    n_sw = sw_vals.shape[0]
    b_len = gl_vals.shape[1]

    def step_b(j, buf):
        for k in range(4):
            is_sw = meta[k, j, 0] != 0
            lidx = meta[k, j, 1]
            blk = meta[k, j, 2]
            boff = jnp.clip(meta[k, j, 3], 0, SUBPAGE_TOKENS - 1)
            valid = (blk >= 0) & (blk < num_blocks) & (lidx >= 0)
            blk_c = jnp.clip(blk, 0, num_blocks - 1)
            gl_c = jnp.clip(lidx, 0, n_gl - 1)
            sw_c = jnp.clip(lidx, 0, n_sw - 1)
            val = jnp.where(is_sw, sw_vals[sw_c, j], gl_vals[gl_c, j]).reshape(
                elem_shape
            )
            cur = buf[blk_c, boff]
            buf = buf.at[blk_c, boff].set(jnp.where(valid, val, cur))
        return buf

    return jax.lax.fori_loop(0, b_len, step_b, pool)


# -----------------------------------------------------------------------------
# Shared-store TP8 prefill MoE (reads gu_pack / dw_pack / gu_dense / dw_dense in place)
# -----------------------------------------------------------------------------


def _packed_gmm_layer(
    inputs: jax.Array,
    packed: jax.Array,
    gains: jax.Array,
    group_sizes: jax.Array,
    layer_idx: jax.Array,
    *,
    fuse_silu: bool,
    tm: int = 128,
) -> jax.Array:
    """VMEM-dequantized grouped matmul reading layer `layer_idx[0]` from 4D `packed` [L,
    E,
    depth, N] uint8.
    """
    from jax.experimental.pallas.ops.tpu.megablox.gmm import (
        _get_store_mask,
        make_group_metadata,
    )

    original_rows, k = inputs.shape
    _, experts, _, n = packed.shape
    tn = n
    out_cols = n // 2 if fuse_silu else n
    m = pl.cdiv(original_rows, tm) * tm
    if m > original_rows:
        inputs = jnp.pad(inputs, ((0, m - original_rows), (0, 0)))
    metadata, active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=jnp.int32(0),
        num_nonzero_groups=experts,
        visit_empty_groups=False,
    )
    li = layer_idx.reshape(1)
    gains_li = jax.lax.dynamic_index_in_dim(
        gains, li[0], axis=0, keepdims=False
    ).reshape(experts, -1)
    metadata = (*metadata, active_tiles[None], li, gains_li)

    def kernel(meta, x, weights, out, values, scales, cached, sem):
        tile = pl.program_id(1)
        lay = meta[4][0]
        g_ref = meta[5]
        expert = meta[1][tile]
        previous = meta[1][jnp.maximum(tile - 1, 0)]
        m_id = meta[2][tile]
        prev_m_id = meta[2][jnp.maximum(tile - 1, 0)]

        @pl.when((tile == 0) | (m_id != prev_m_id))
        def _zero_out():
            out[...] = jnp.zeros_like(out[...])

        def transfers(e):
            return (
                pltpu.make_async_copy(
                    weights.at[lay, e, pl.ds(0, k // 2), pl.ds(0, tn)],
                    values,
                    sem.at[0],
                ),
                pltpu.make_async_copy(
                    weights.at[lay, e, pl.ds(k // 2, k // 16), pl.ds(0, tn)],
                    scales,
                    sem.at[1],
                ),
            )

        @pl.when((tile == 0) | (previous != expert))
        def _expand():
            v_cp, s_cp = transfers(expert)

            @pl.when(tile == 0)
            def _first():
                v_cp.start()
                s_cp.start()

            v_cp.wait()
            s_cp.wait()
            vals = pltpu.bitcast(values[...], jnp.float4_e2m1fn).astype(jnp.bfloat16)
            scs = jax.lax.bitcast_convert_type(scales[...], jnp.float8_e4m3fn).astype(
                jnp.bfloat16
            )
            cached[...] = (vals.reshape(16, k // 16, tn) * scs[None, :, :]).reshape(
                k, tn
            )

        following = meta[1][jnp.minimum(tile + 1, meta[1].shape[0] - 1)]

        @pl.when((tile + 1 < meta[3][0]) & (following != expert))
        def _prefetch():
            for cp in transfers(following):
                cp.start()

        projected = jnp.dot(x[...], cached[...], preferred_element_type=F32)
        if fuse_silu:
            half = tn // 2
            gate = (
                (projected[:, :half] * g_ref[expert, 0])
                .astype(jnp.bfloat16)
                .astype(F32)
            )
            up = (
                (projected[:, half:] * g_ref[expert, 1])
                .astype(jnp.bfloat16)
                .astype(F32)
            )
            res = (jax.nn.silu(gate).astype(jnp.bfloat16).astype(F32) * up).astype(
                jnp.bfloat16
            )
        else:
            res = (projected * g_ref[expert, 0]).astype(jnp.bfloat16)
        mask = _get_store_mask(
            grid_id=tile, group_metadata=meta[:3], tm=tm, tn=out_cols
        )
        out[...] = jnp.where(mask, res, out[...])

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, out_cols), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(
                pl.BlockSpec((tm, k), lambda n_i, m_i, k_i, meta: (meta[2][m_i], 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(
                (tm, out_cols), lambda n_i, m_i, k_i, meta: (meta[2][m_i], 0)
            ),
            grid=(1, active_tiles, 1),
            scratch_shapes=(
                pltpu.VMEM((k // 2, tn), jnp.uint8),
                pltpu.VMEM((k // 16, tn), jnp.uint8),
                pltpu.VMEM((k, tn), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=32 * 1024**2,
        ),
        name=f"muse_prefill_packed_{'gu' if fuse_silu else 'dw'}",
    )(metadata, inputs, packed)[:original_rows]


def _dense_gmm_layer(
    inputs: jax.Array,
    dense_w: jax.Array,
    group_sizes: jax.Array,
    layer_idx: jax.Array,
    *,
    fuse_silu: bool,
    tm: int = 128,
) -> jax.Array:
    """Grouped matmul reading layer `layer_idx[0]` in place from 4D `dense_w` [L_d, E,
    K,
    N] bfloat16.
    """
    from jax.experimental.pallas.ops.tpu.megablox.gmm import (
        _get_store_mask,
        make_group_metadata,
    )

    original_rows, k = inputs.shape
    _, experts, _, n = dense_w.shape
    tn = n
    out_cols = n // 2 if fuse_silu else n
    m = pl.cdiv(original_rows, tm) * tm
    if m > original_rows:
        inputs = jnp.pad(inputs, ((0, m - original_rows), (0, 0)))
    metadata, active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=jnp.int32(0),
        num_nonzero_groups=experts,
        visit_empty_groups=False,
    )
    li = layer_idx.reshape(1)
    metadata = (*metadata, active_tiles[None], li)

    def kernel(meta, x, weights, out, staging, cached, sem):
        tile = pl.program_id(1)
        lay = meta[4][0]
        expert = meta[1][tile]
        previous = meta[1][jnp.maximum(tile - 1, 0)]
        m_id = meta[2][tile]
        prev_m_id = meta[2][jnp.maximum(tile - 1, 0)]

        @pl.when((tile == 0) | (m_id != prev_m_id))
        def _zero_out():
            out[...] = jnp.zeros_like(out[...])

        # The next expert is prefetched into `staging` while `cached` is still being
        # read.
        def transfer(e):
            return pltpu.make_async_copy(
                weights.at[lay, e, pl.ds(0, k), pl.ds(0, tn)],
                staging,
                sem.at[0],
            )

        @pl.when((tile == 0) | (previous != expert))
        def _load():
            cp = transfer(expert)

            @pl.when(tile == 0)
            def _first():
                cp.start()

            cp.wait()
            cached[...] = staging[...]

        following = meta[1][jnp.minimum(tile + 1, meta[1].shape[0] - 1)]

        @pl.when((tile + 1 < meta[3][0]) & (following != expert))
        def _prefetch():
            transfer(following).start()

        projected = jnp.dot(x[...], cached[...], preferred_element_type=F32)
        if fuse_silu:
            half = tn // 2
            gate = projected[:, :half].astype(jnp.bfloat16).astype(F32)
            up = projected[:, half:].astype(jnp.bfloat16).astype(F32)
            res = (jax.nn.silu(gate).astype(jnp.bfloat16).astype(F32) * up).astype(
                jnp.bfloat16
            )
        else:
            res = projected.astype(jnp.bfloat16)
        mask = _get_store_mask(
            grid_id=tile, group_metadata=meta[:3], tm=tm, tn=out_cols
        )
        out[...] = jnp.where(mask, res, out[...])

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, out_cols), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(
                pl.BlockSpec((tm, k), lambda n_i, m_i, k_i, meta: (meta[2][m_i], 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(
                (tm, out_cols), lambda n_i, m_i, k_i, meta: (meta[2][m_i], 0)
            ),
            grid=(1, active_tiles, 1),
            scratch_shapes=(
                pltpu.VMEM((k, tn), jnp.bfloat16),
                pltpu.VMEM((k, tn), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((1,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            vmem_limit_bytes=48 * 1024**2,
        ),
        name=f"muse_prefill_dense_{'gu' if fuse_silu else 'dw'}",
    )(metadata, inputs, dense_w)[:original_rows]


def tp8_shared_moe_local(
    h: jax.Array,
    z_pre: jax.Array,
    router_w: jax.Array,
    router_bias: jax.Array,
    pre_norm_w: jax.Array,
    post_e_norm_w: jax.Array,
    gu_w: jax.Array,
    dw_w: jax.Array,
    gu_g: jax.Array | None,
    dw_g: jax.Array | None,
    layer_idx: jax.Array,
    rank_arr: jax.Array,
    *,
    is_packed: bool,
    topk: int = 8,
    output_multiplier: float = 0.17677669529663687,
    routed_scaling_factor: float = 1.0,
    rms_norm_eps: float = 1e-5,
    post_norm_eps: float = 1e-8,
    router_eps: float = 1e-15,
    batch_invariant: bool = True,
    s2_opt: bool = True,
) -> jax.Array:
    """Run one Muse Spark MoE layer during prefill over the shared TP8 megakernel expert
    store.

    Returns rank `r`'s weighted expert contribution `[1, T, 4096] float32`
    (for selected expert slot `k = r`),
    which `MuseSparkMoE.forward` then sums across the 8 TP ranks via `det_all_reduce`.
    """
    from vllm_torchtpu.kernels.muse_moe_ops import (
        muse_centered_rms_norm_jax,
        muse_input_scaled_rms_norm_jax,
        muse_route_jax,
    )

    h0, z0 = h[0], z_pre[0]
    rw0, rb0 = router_w[0], router_bias[0]
    pn0, pen0 = pre_norm_w[0], post_e_norm_w[0]
    gu0, dw0 = gu_w[0], dw_w[0]
    li0 = layer_idx[0]
    r0 = rank_arr[0, 0] if rank_arr.ndim > 1 else rank_arr[0]

    num_tokens, moe_hidden = z0.shape
    experts = rb0.shape[-1]

    topk_weights, topk_ids = muse_route_jax(
        h0,
        rw0,
        rb0,
        topk=topk,
        output_multiplier=output_multiplier,
        routed_scaling_factor=routed_scaling_factor,
        router_eps=router_eps,
        batch_invariant=batch_invariant,
    )
    z = muse_centered_rms_norm_jax(
        z0, pn0, rms_norm_eps, batch_invariant=batch_invariant
    )
    z = (
        z.reshape(num_tokens, moe_hidden // 16, 16)
        .swapaxes(1, 2)
        .reshape(num_tokens, moe_hidden)
    )

    m_pairs = num_tokens * topk
    flat_ids = topk_ids.reshape(-1)
    order = jnp.argsort(flat_ids, stable=True)
    group_sizes = jnp.bincount(flat_ids, length=experts).astype(jnp.int32)
    sorted_inputs = z[order // topk]

    if is_packed:
        assert gu_g is not None and dw_g is not None
        act = _packed_gmm_layer(
            sorted_inputs, gu0, gu_g[0], group_sizes, li0, fuse_silu=True
        )
        sorted_down = _packed_gmm_layer(
            act, dw0, dw_g[0], group_sizes, li0, fuse_silu=False
        )
    else:
        act = _dense_gmm_layer(sorted_inputs, gu0, group_sizes, li0, fuse_silu=True)
        sorted_down = _dense_gmm_layer(act, dw0, group_sizes, li0, fuse_silu=False)

    inv_order = (
        jnp.empty_like(order)
        .at[order]
        .set(jnp.arange(m_pairs, dtype=jnp.int32), unique_indices=True)
    )
    # Gather straight into [topk, tokens, H]: folds the transpose into the gather index.
    pair_idx = inv_order.reshape(num_tokens, topk).T.reshape(-1)
    pair_rows_t = sorted_down[pair_idx].reshape(topk, num_tokens, moe_hidden)

    exchanged = jax.lax.all_to_all(
        pair_rows_t, TP_AXIS_NAME, split_axis=0, concat_axis=0, tiled=False
    )
    acc = exchanged[0].astype(F32)
    for p in range(1, topk):
        acc = acc + exchanged[p].astype(F32)
    down_r = acc.astype(jnp.bfloat16)

    normed_r = muse_input_scaled_rms_norm_jax(
        down_r, pen0, post_norm_eps, batch_invariant=batch_invariant
    )
    # all_to_all hands position q the rows of top k slot q; positions follow device
    # order, not
    # rank_in_group, so weight by this rank's collective position.
    ranks = jax.lax.all_gather(jnp.reshape(r0, (1,)), TP_AXIS_NAME, axis=0, tiled=True)
    pos = jnp.argmax(ranks == r0).astype(jnp.int32)
    w_r = jax.lax.dynamic_index_in_dim(topk_weights, pos, axis=1, keepdims=True).astype(
        F32
    )
    out_r = normed_r.astype(F32) * w_r
    return out_r[None]


# -----------------------------------------------------------------------------
# Ring row read / masked write in place (DSpark verify rollback and draft context KV)
# -----------------------------------------------------------------------------

_RING_BLOCK = 16


def ring_rows_read_local(kv: jax.Array, slots: jax.Array, ridx: jax.Array) -> jax.Array:
    """Rows kv[0, :, slots[j], ridx[j], :] as f32 [1, L, n, W] from a [1, L, S, R, W]
    ring.

    DMAs the aligned 16 row block that holds each row (coalescing consecutive hits in
    the same `(slot, block)`), so XLA keeps the ring in the layout the megakernel uses
    (slicing one sublane row in XLA forces a full ring relayout).
    """
    _, n_l, _, _, w = kv.shape
    n = slots.shape[0]

    def kernel(slots_ref, ridx_ref, kv_hbm, out_ref, blk, sem):
        for j in range(n):
            bidx = ridx_ref[j] // _RING_BLOCK
            start = pl.multiple_of(bidx * _RING_BLOCK, _RING_BLOCK)
            if j == 0:
                cp = pltpu.make_async_copy(
                    kv_hbm.at[0, :, slots_ref[0], pl.ds(start, _RING_BLOCK), :],
                    blk,
                    sem.at[0],
                )
                cp.start()
                cp.wait()
            else:
                prev_bidx = ridx_ref[j - 1] // _RING_BLOCK

                @pl.when((slots_ref[j] != slots_ref[j - 1]) | (bidx != prev_bidx))
                def _(j=j, start=start):
                    cp = pltpu.make_async_copy(
                        kv_hbm.at[0, :, slots_ref[j], pl.ds(start, _RING_BLOCK), :],
                        blk,
                        sem.at[0],
                    )
                    cp.start()
                    cp.wait()

            sel = jax.lax.broadcasted_iota(jnp.int32, (n_l, _RING_BLOCK, w), 1) == (
                ridx_ref[j] % _RING_BLOCK
            )
            row = jnp.sum(jnp.where(sel, blk[...].astype(F32), 0.0), axis=1)
            out_ref[0, :, j, :] = row

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((1, n_l, n, w), F32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec((1, n_l, n, w), lambda i, *_: (0, 0, 0, 0)),
            scratch_shapes=[
                pltpu.VMEM((n_l, _RING_BLOCK, w), kv.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
    )(slots, ridx, kv)


def ring_rows_write_local(
    kv: jax.Array,
    vals: jax.Array,
    slots: jax.Array,
    ridx: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """kv[0, :, slots[j], ridx[j], :] = vals[0, :, j, :] where mask[j] != 0, in place.

    vals: f32 [1, L, n, W]. Rows are applied in order, so later rows win on duplicates;
    consecutive masked writes hitting the same `(slot, 16-row block)` are coalesced in
    VMEM into a single HBM read-modify-write DMA pair.
    """
    _, n_l, _, _, w = kv.shape
    n = slots.shape[0]

    def kernel(slots_ref, ridx_ref, mask_ref, kv_in, vals_ref, kv_out, blk, sem, state):
        del kv_in
        state[0] = -1
        state[1] = 0
        for j in range(n):

            @pl.when(mask_ref[j] != 0)
            def _(j=j):
                s = slots_ref[j]
                bidx = ridx_ref[j] // _RING_BLOCK
                start = pl.multiple_of(bidx * _RING_BLOCK, _RING_BLOCK)

                @pl.when((state[0] != s) | (state[1] != bidx))
                def _switch(s=s, bidx=bidx, start=start):
                    @pl.when(state[0] >= 0)
                    def _flush():
                        prev_s = state[0]
                        prev_st = pl.multiple_of(state[1] * _RING_BLOCK, _RING_BLOCK)
                        wr = pltpu.make_async_copy(
                            blk,
                            kv_out.at[0, :, prev_s, pl.ds(prev_st, _RING_BLOCK), :],
                            sem.at[0],
                        )
                        wr.start()
                        wr.wait()

                    rd = pltpu.make_async_copy(
                        kv_out.at[0, :, s, pl.ds(start, _RING_BLOCK), :],
                        blk,
                        sem.at[0],
                    )
                    rd.start()
                    rd.wait()
                    state[0] = s
                    state[1] = bidx

                sel = jax.lax.broadcasted_iota(jnp.int32, (n_l, _RING_BLOCK, w), 1) == (
                    ridx_ref[j] % _RING_BLOCK
                )
                row = vals_ref[0, :, j, :][:, None, :]
                blk[...] = jnp.where(sel, row, blk[...].astype(F32)).astype(blk.dtype)

        @pl.when(state[0] >= 0)
        def _final_flush():
            prev_s = state[0]
            prev_st = pl.multiple_of(state[1] * _RING_BLOCK, _RING_BLOCK)
            wr = pltpu.make_async_copy(
                blk,
                kv_out.at[0, :, prev_s, pl.ds(prev_st, _RING_BLOCK), :],
                sem.at[0],
            )
            wr.start()
            wr.wait()

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(kv.shape, kv.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(1,),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, n_l, n, w), lambda i, *_: (0, 0, 0, 0)),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((n_l, _RING_BLOCK, w), kv.dtype),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SMEM((2,), jnp.int32),
            ],
        ),
        input_output_aliases={3: 0},
    )(slots, ridx, mask, kv, vals)
