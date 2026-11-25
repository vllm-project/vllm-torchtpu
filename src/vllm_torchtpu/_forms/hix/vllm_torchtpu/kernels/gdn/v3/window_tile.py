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
"""GDN verify-window tile, one window position and one value head at a time
(PR 2821 d8340b973 `window_tile`, q38 port behind VLLM_TPU_GDN_WINDOW_TILE).

`wrapper.inner_kernel` computes a whole tile as large arrays: the conv on
the compact [chunk, 1, dim_size] layout (one 128-lane chunk of one token
per vector register) and the recurrence on the [num_v_heads, kq_head_dim,
v_head_dim] FP32 state of every window position. Neither fits in
registers, and the recurrence's MXU weight pushes (two 128 x 128 states per
head and position) leave most of the VPU idle. This tile instead:

* keeps each token's activations dense, [dim_size / 128, 128] (one head
  per row), so the conv touches 8x fewer registers;
* decodes and encodes the pooled conv state one tap (one dense
  [dim_size / 128, 128] block) at a time instead of row by row;
* runs window positions one after the other: position `pos` runs the conv
  of token `pos`, writes conv checkpoint `pos`, and updates the recurrent
  state one value head at a time through a VMEM scratch, storing each
  head's checkpoint as it is produced. The heads of a position are
  independent, which hides each head's MXU round trip, and the conv and
  checkpoint work fills the slots the weight pushes leave free.

Every element sees the same operations in the same order as in
`inner_kernel` (the conv's tap sum, SiLU, per-row masks, L2 norms and
gates, per-head dots, which `recurrent_gdn` batches over heads), so outputs
and every checkpoint byte are bitwise equal.

Not ported from the PR: two-slot checkpoints (TPU_GDN_TWO_SLOT_CKPT, q38
lever 10, closed), paired heads, packed tokens and the single launch.

q38 generalisations (the PR gate admitted only the SEQ_ALONG_LANE pool with
BF16 states and conv rows in whole 8-row tiles, so Flash never engaged):

* Both DMA modes. Head and conv addressing only needs one source block's
  tile shape, which whole-block DMA ([*block]) and row-ranged DMA
  ([nrows, *payload, lanes], Flash's NHD pool) both provide.
* FP32 recurrent and conv state views (Flash) as well as BF16 ones. Stores
  go through the region's view dtype, as `vmem_ldst.store_state_region`.
* Conv regions whose typed rows are wider than 128 lanes (Flash's pool rows
  are 256 lanes) are read and written through a 128-lane view of the same
  bytes (a ref-level lane split): the same flat element stream.
* Conv rows padded to whole 8-row tiles inside the tile: TP8 Flash has
  dim_size / 128 = 10 rows per token, kept as 16-row blocks in the VMEM
  scratches; the pool's conv layout is unchanged. The taps of such a
  layout start at 10-row offsets, so they are loaded and stored one tap at
  a time; this needs a 32-bit conv view (BF16 conv views keep the PR's
  whole-tile 8-row-aligned path or fall back to inner_kernel).
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, memory_ref, vmem_ldst


def _lanes() -> int:
    return pltpu.get_tpu_info().num_lanes


def _rows(cfg: config.GDNConfig) -> int:
    """Dense rows of one token's activations, [rows, 128]."""
    return cfg.dim_size // _lanes()


def _padded(cfg: config.GDNConfig) -> bool:
    """Whether a token's dense rows are padded inside the tile (TP8 Flash)."""
    return cfg.window_tile_rows != _rows(cfg)


def _conv_typed_lanes(cfg: config.GDNConfig, block_lanes: int) -> int:
    """Lanes of one typed row of the pooled conv region."""
    assert cfg.state_plan is not None
    return block_lanes // cfg.state_plan.conv.lane_split


def _stored_rows(cfg: config.GDNConfig, block_lanes: int) -> tuple[int, ...] | None:
    """Logical conv row held by each stored row of one conv tap (None for
    the identity), or () when the pooled row order is not the same
    permutation in every tap or moves value rows."""
    assert cfg.state_plan is not None
    region = cfg.state_plan.conv
    if region.rows_perm is None:
        return None
    if _conv_typed_lanes(cfg, block_lanes) != _lanes():
        # The permutation is in units of wider typed rows.
        return ()
    rows = _rows(cfg)
    perm = region.rows_perm
    tap = perm[:rows]
    for j in range(cfg.prev_kernel_size):
        if tuple(p - j * rows for p in perm[j * rows : (j + 1) * rows]) != tap:
            return ()
    qk_rows = 2 * cfg.num_kq_heads * cfg.kq_head_dim // _lanes()
    if sorted(tap) != list(range(rows)) or tuple(tap[qk_rows:]) != tuple(
        range(qk_rows, rows)
    ):
        return ()
    stored = [0] * rows
    for logical, s in enumerate(tap):
        stored[s] = logical
    return None if tap == tuple(range(rows)) else tuple(stored)


def supported(cfg: config.GDNConfig, block_shape: tuple[int, ...], block_dtype) -> bool:
    """Whether this tile implements `cfg` exactly as inner_kernel does, for
    pooled recurrent-region source blocks of `block_shape` / `block_dtype`
    (one checkpoint tile's per-block shape: the whole block with whole-block
    DMA, else [nrows, *payload, lanes])."""
    lanes = _lanes()
    plan = cfg.state_plan
    if (
        cfg.mode != config.GDNMode.BATCHED
        or cfg.window_size <= 1
        or cfg.is_kda
        or plan is None
        or cfg.kq_head_dim != lanes
        or cfg.v_head_dim != lanes
        or cfg.dim_size % lanes
        or cfg.num_v_heads % cfg.num_kq_heads
    ):
        return False
    conv = plan.conv
    rows = _rows(cfg)
    block_lanes = block_shape[-1]
    typed_lanes = _conv_typed_lanes(cfg, block_lanes)
    if block_lanes % conv.lane_split or typed_lanes % lanes:
        return False
    if (
        conv.view_dtype not in (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float32))
        or conv.nblocks != 1
        # Logical conv rows, counted in 128-lane rows.
        or conv.rows_used * (typed_lanes // lanes) != cfg.prev_kernel_size * rows
        or _stored_rows(cfg, block_lanes) == ()
    ):
        return False
    if rows % 8 and jnp.dtype(conv.view_dtype).itemsize != 4:
        # Taps at unaligned row offsets are moved one 32-bit tap at a time.
        return False
    return plan.recurrent.view_dtype in (
        jnp.dtype(jnp.uint32),
        jnp.dtype(jnp.float32),
    ) and vmem_ldst.supports_head_states(block_shape, block_dtype, cfg)


def _conv_view(slot_ref: jax.Ref, cfg: config.GDNConfig) -> jax.Ref:
    """One checkpoint's conv region tile as typed [rows, 128] rows: the
    region's view dtype, split down to 128 lanes at the ref level, so the
    rows are the same flat element stream `vmem_ldst.store_state_region`
    writes through its wider view."""
    assert cfg.state_plan is not None
    block = slot_ref.at[0]
    return typed_ldst.typed_view(
        block, cfg.state_plan.conv.view_dtype, block.shape[-1] // _lanes()
    )


def _to_stored(block: jax.Array, cfg: config.GDNConfig, order) -> jax.Array:
    """[dim_size / 128, lanes] block from logical to the pool's stored row
    order; only the rows that move are touched."""
    if order is None:
        return block
    head = len(order)
    while head and order[head - 1] == head - 1:
        head -= 1
    moved = jnp.concat([block[order[i] : order[i] + 1] for i in range(head)], axis=0)
    return jnp.concat([moved, block[head:]], axis=0)


def _load_conv_taps(slot_ref: jax.Ref, cfg: config.GDNConfig) -> list[jax.Array]:
    """The conv state as FP32 [dim_size / 128, lanes] taps in the pool's
    stored row order: per element, the values vmem_ldst._load_conv_state
    decodes."""
    rows = _rows(cfg)
    view = _conv_view(slot_ref, cfg)
    if not _padded(cfg):
        # Whole 8-row tiles: one load, then aligned slices (the PR's form).
        used = cfg.prev_kernel_size * rows
        taps = view[pl.ds(0, used)].astype(jnp.float32)
        return [taps[j * rows : (j + 1) * rows] for j in range(cfg.prev_kernel_size)]
    # Taps at unaligned row offsets: one 32-bit load per tap.
    return [
        view[pl.ds(j * rows, rows)].astype(jnp.float32)
        for j in range(cfg.prev_kernel_size)
    ]


def _store_conv_ckpt(slot_ref: jax.Ref, cfg: config.GDNConfig, taps) -> None:
    """Stores one conv checkpoint given as FP32 [dim_size / 128, lanes] taps
    in the stored row order: the bytes vmem_ldst.store_state_region writes,
    zero padding included."""
    view = _conv_view(slot_ref, cfg)
    lanes = _lanes()
    rows = _rows(cfg)
    used = cfg.prev_kernel_size * rows
    dtype = view.dtype
    if not _padded(cfg):
        # Whole packed registers: the data rows, then the zero padding.
        pack = 32 // (8 * jnp.dtype(dtype).itemsize) * 8
        head = min(view.shape[0], -(-used // pack) * pack)
        taps = list(taps)
        if head > used:
            taps.append(jnp.zeros((head - used, lanes), jnp.float32))
        view[pl.ds(0, head)] = jnp.concat(taps, axis=0).astype(dtype)
    else:
        # Taps at unaligned row offsets: one 32-bit store per tap.
        for j, tap in enumerate(taps):
            view[pl.ds(j * rows, rows)] = tap.astype(dtype)
        head = used
    if head < view.shape[0]:
        view[pl.ds(head, view.shape[0] - head)] = jnp.zeros(
            (view.shape[0] - head, view.shape[1]), dtype
        )


def run(
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size] FP32
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, lanes]
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, lanes]
    conv_in_ref: jax.Ref,  # [seq, 1, nblocks, *block tile]
    rec_in_ref: jax.Ref,  # [seq, 1, nblocks, *block tile]
    out_slot_ref: jax.Ref,  # [seq, chunk, num_v_heads, v_head_dim]
    conv_out_ref: jax.Ref,  # [seq, window, nblocks, *block tile]
    rec_out_ref: jax.Ref,  # [seq, window, nblocks, *block tile]
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    state_ref: jax.Ref,  # [seq, num_v_heads, kq_head_dim, v_head_dim] FP32
    lhs_ref: jax.Ref,  # [seq, chunk + prev_kernel_size, window_tile_rows, 128] FP32
    conv_w_ref: jax.Ref,  # [kernel_size + 1, window_tile_rows, 128] FP32
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array,
) -> None:
    lanes = _lanes()
    rows = _rows(cfg)
    padded_rows = cfg.window_tile_rows
    nv, nkq = cfg.num_v_heads, cfg.num_kq_heads
    ks, prev = cfg.kernel_size, cfg.prev_kernel_size
    compute = cfg.dtypes.compute
    q_scale = cfg.kq_head_dim**-0.5
    seqs = range(cfg.seq_tile_size)
    order = _stored_rows(cfg, conv_in_ref.shape[-1])
    assert order != ()

    real_sizes = [metadata_ref.p_id_to_r_size[p_id, idx] for idx in seqs]
    has_initial_state = [
        metadata_ref.s_idx_has_initial_state[metadata_ref.p_id_to_s_idx[p_id, idx]]
        for idx in seqs
    ]

    if padded_rows != rows:
        # Padding rows stay zero; no output or checkpoint ever reads them.
        lhs_ref[...] = jnp.zeros(lhs_ref.shape, jnp.float32)
        conv_w_ref[...] = jnp.zeros(conv_w_ref.shape, jnp.float32)

    # Conv inputs per sequence, dense and in the pool's stored row order: the
    # previous taps (zero for a fresh sequence), then the window's tokens.
    # The conv is per row, so with its weights in the same order it yields
    # the activations in that order and every checkpoint is a plain copy of
    # three input rows.
    for idx in seqs:
        prev_taps = _load_conv_taps(conv_in_ref.at[idx, 0], cfg)
        for tap in range(prev):
            lhs_ref[idx, tap, pl.ds(0, rows)] = jnp.where(
                has_initial_state[idx], prev_taps[tap], 0
            )
        for t in range(cfg.chunk_size):
            token = qkv_slot_ref.at[idx].reshape(-1, lanes)[pl.ds(t * rows, rows)]
            lhs_ref[idx, prev + t, pl.ds(0, rows)] = _to_stored(token, cfg, order)
    for k in range(ks):
        weight = weights_ref.conv.weight.at[k].reshape(-1, lanes)[...]
        conv_w_ref[k, pl.ds(0, rows)] = _to_stored(
            weight.astype(jnp.float32), cfg, order
        )
    if weights_ref.conv.bias is not None:
        # The bias ref is rank 1 ([dim_size]) and Mosaic cannot reshape a
        # rank-1 ref ("Reshape 1D ref is not supported", T44 GDNWT
        # j-bae75c90 at TP8), so it is loaded whole and reshaped as a value.
        bias = weights_ref.conv.bias[...].reshape(-1, lanes)
        conv_w_ref[ks, pl.ds(0, rows)] = _to_stored(
            bias.astype(jnp.float32), cfg, order
        )
    stored_row = list(range(rows)) if order is None else [0] * rows
    if order is not None:
        for s_row, logical in enumerate(order):
            stored_row[logical] = s_row

    padding_size = cfg.aligned_num_v_heads - nv
    a_log = jnp.pad(weights_ref.gdn.a_log[...], (0, padding_size)).reshape(1, -1)
    neg_a = -jnp.exp(a_log.astype(compute))
    dt_bias = jnp.pad(weights_ref.gdn.dt_bias[...], (0, padding_size)).reshape(1, -1)
    dt_bias = dt_bias.astype(compute)
    batched_dims = (((2,), (1,)), ((0,), (0,)))

    def head_scalar(row, head):
        # One head lane of a gate row as [1, 1], picked like
        # compute_gdn.recurrent_gdn's fused_transpose_broadcast does.
        picked = compute_gdn.fused_transpose_broadcast(
            row[:, :nv].reshape(1, 1, nv), src_dim=2, dst_dim=1
        )
        return picked[0, head : head + 1]

    def head_dot(row, mat):
        # A [1, lanes] row times one head's [kq_head_dim, v_head_dim] matrix.
        return jax.lax.dot(
            row[None],
            mat[None],
            dimension_numbers=batched_dims,
            preferred_element_type=jnp.float32,
        )[0].astype(compute)

    def position(pos, first):
        for idx in seqs:
            real_size = real_sizes[idx]
            valid = pos < real_size

            # compute_conv1d.causal_conv1d's tap sum, then SiLU (stored order).
            act = jnp.zeros((padded_rows, lanes), jnp.float32)
            for k in range(ks):
                act += lhs_ref[idx, pos + k] * conv_w_ref[k]
            if weights_ref.conv.bias is not None:
                act += conv_w_ref[ks]
            act = jax.nn.silu(act)

            # Conv checkpoint `pos`: the inputs ending at the window's
            # position `pos`, or at the last real token if that is earlier.
            start = jnp.where(
                jnp.logical_and(real_size >= 1, real_size <= pos),
                real_size,
                pos + 1,
            )
            _store_conv_ckpt(
                conv_out_ref.at[idx, pos],
                cfg,
                [lhs_ref[idx, start + j, pl.ds(0, rows)] for j in range(prev)],
            )

            b_row, a_row = b_slot_ref[idx, pos], a_slot_ref[idx, pos]
            beta = jnp.where(valid, jax.nn.sigmoid(b_row.astype(compute)), 0)
            gating = neg_a * jax.nn.softplus(a_row.astype(compute) + dt_bias)
            gating = jnp.exp(jnp.where(valid, gating, 0))

            out_rows = []
            for kq_head in range(nkq):
                r = stored_row[kq_head]
                q = compute_gdn.l2_norm(
                    jnp.where(valid, act[r : r + 1].astype(compute), 0)
                )
                q = q * q_scale
                r = stored_row[nkq + kq_head]
                k = compute_gdn.l2_norm(
                    jnp.where(valid, act[r : r + 1].astype(compute), 0)
                )
                k_t = compute_gdn.fused_transpose_broadcast(k, src_dim=1, dst_dim=0)
                heads = range(
                    kq_head * cfg.v_per_kq_head, (kq_head + 1) * cfg.v_per_kq_head
                )
                for head in heads:
                    v = act[2 * nkq + head : 2 * nkq + head + 1]
                    v = jnp.where(valid, v.astype(compute), 0)
                    if first:
                        state = vmem_ldst.load_state_head(rec_in_ref, cfg, idx, head)
                        state = jnp.where(has_initial_state[idx], state, 0)
                    else:
                        state = state_ref[idx, head]
                    decayed = state * head_scalar(gating, head)
                    v_new = head_scalar(beta, head) * (v - head_dot(k, decayed))
                    updated = decayed + k_t * v_new
                    out_rows.append(head_dot(q, updated))
                    state_ref[idx, head] = updated
                    vmem_ldst.store_state_head(rec_out_ref, cfg, idx, pos, head, updated)
            out_slot_ref[idx, pos] = jnp.concat(out_rows, axis=0).astype(
                out_slot_ref.dtype
            )

    for pos in range(cfg.chunk_size):
        position(pos, first=pos == 0)
    for idx in seqs:
        for t in range(cfg.window_size):
            vmem_ldst.zero_state_padding(rec_out_ref, cfg, idx, t)
