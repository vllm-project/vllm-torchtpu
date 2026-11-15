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
"""GDN verify-window tile, one window position and one value head at a
time (VLLM_TPU_GDN_V3_WINDOW_VERIFY; port of PR #2821 d8340b973).

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
* loops over window positions: position `pos` runs the conv of token
  `pos`, writes conv checkpoint `pos`, and updates the recurrent state one
  value head at a time through a VMEM scratch, storing each head's
  checkpoint as it is produced. The heads of a position are independent,
  which hides each head's MXU round trip, and the conv and checkpoint work
  fills the slots the weight pushes leave free.

Every element sees the same operations in the same order as in
`inner_kernel` (the conv's tap sum, SiLU, per-row masks, L2 norms and
gates, per-head dots, which `recurrent_gdn` batches over heads), so outputs
and every checkpoint byte are bitwise equal.

Differences from the upstream tile: the pool may be addressed per region
(NHD pools, `whole_block_dma` False) and the conv state may be stored as
FP32 (Qwen3.8's pooled conv state); its typed rows are re-viewed as
128-lane rows. The upstream two-slot checkpoints, paired heads and packed
activations are not ported.
"""

import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.gdn.v3 import compute_gdn, config, memory_ref, vmem_ldst


def _lanes() -> int:
    return pltpu.get_tpu_info().num_lanes


def _conv_dtype(cfg: config.GDNConfig):
    assert cfg.state_plan is not None
    return jnp.dtype(cfg.state_plan.conv.view_dtype)


def _pack_rows(cfg: config.GDNConfig) -> int:
    """Typed rows per whole vector register of the conv storage dtype."""
    return 8 * (4 // _conv_dtype(cfg).itemsize)


def region_block_shape(
    plan: config.StateSourcePlan,
    region: config.StateRegion,
    source_shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Shape of one source block of a region's VMEM tile (the tile holds
    whole blocks with whole-block DMA, else only the region's rows)."""
    if plan.whole_block_dma:
        return tuple(source_shape[1:])
    return (region.nrows, *source_shape[2:])


def _stored_rows(cfg: config.GDNConfig) -> tuple[int, ...] | None:
    """Logical conv row held by each stored row of one conv tap (None for
    the identity), or () when the pooled row order is not the same
    permutation in every tap or moves value rows."""
    assert cfg.state_plan is not None
    region = cfg.state_plan.conv
    rows = cfg.dim_size // _lanes()
    if region.rows_perm is None:
        return None
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


def supported(
    cfg: config.GDNConfig, source_shape: tuple[int, ...], source_dtype
) -> bool:
    """Whether this tile implements `cfg` exactly as inner_kernel does, for
    the pooled state source of `source_shape` / `source_dtype`."""
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
    ):
        return False
    conv = plan.conv
    rows = cfg.dim_size // lanes
    conv_block = region_block_shape(plan, conv, source_shape)
    block_lanes = conv_block[-1]
    if (
        conv.view_dtype not in (jnp.bfloat16, jnp.float32)
        or conv.lane_split != 1
        or conv.nblocks != 1
        or block_lanes % lanes
    ):
        return False
    lane_split = block_lanes // lanes
    if conv.rows_used * lane_split != cfg.prev_kernel_size * rows:
        return False
    if conv.rows_perm is not None and (lane_split != 1 or _stored_rows(cfg) == ()):
        return False
    if (
        _conv_view_rows(conv_block, source_dtype, conv.view_dtype)
        < cfg.prev_kernel_size * rows
    ):
        return False
    if jnp.dtype(conv.view_dtype) == jnp.bfloat16 and rows % 8:
        # BF16 taps are written in whole packed registers (16 rows).
        return False
    return plan.recurrent.view_dtype == jnp.uint32 and vmem_ldst.supports_head_states(
        region_block_shape(plan, plan.recurrent, source_shape), source_dtype, cfg
    )


def _conv_view_rows(block_shape: tuple[int, ...], block_dtype, view_dtype) -> int:
    """Typed 128-lane `view_dtype` rows of one conv tile (one source block)."""
    nbytes = math.prod(block_shape) * jnp.dtype(block_dtype).itemsize
    return nbytes // (_lanes() * jnp.dtype(view_dtype).itemsize)


def _conv_view(slot_ref: jax.Ref, cfg: config.GDNConfig) -> jax.Ref:
    """One checkpoint's conv region tile as typed [rows, 128] rows of the
    conv storage dtype (row-major: each pool typed row splits into
    `lanes / 128` consecutive 128-lane rows)."""
    block = slot_ref.at[0]
    return typed_ldst.typed_view(block, _conv_dtype(cfg), block.shape[-1] // _lanes())


def _to_stored(block: jax.Array, cfg: config.GDNConfig) -> jax.Array:
    """[dim_size / 128, lanes] block from logical to the pool's stored row
    order; only the rows that move are touched."""
    order = _stored_rows(cfg)
    if order is None:
        return block
    head = len(order)
    while head and order[head - 1] == head - 1:
        head -= 1
    moved = jnp.concat([block[order[i] : order[i] + 1] for i in range(head)], axis=0)
    return jnp.concat([moved, block[head:]], axis=0)


def _load_conv_taps(slot_ref: jax.Ref, cfg: config.GDNConfig) -> jax.Array:
    """The conv state as FP32 [prev_kernel_size * dim_size / 128, lanes] in
    the pool's stored row order: per element, the values
    vmem_ldst._load_conv_state decodes."""
    used = cfg.prev_kernel_size * cfg.dim_size // _lanes()
    view = _conv_view(slot_ref, cfg)
    if view.shape[0] <= used + _pack_rows(cfg):
        # A region-sized tile: load it whole.
        return view[...][:used].astype(jnp.float32)
    return view[pl.ds(0, used)].astype(jnp.float32)


def _store_conv_ckpt(slot_ref: jax.Ref, cfg: config.GDNConfig, taps) -> None:
    """Stores one conv checkpoint given as FP32 taps in the stored row
    order: the bytes vmem_ldst.store_state_region writes, zero padding
    included."""
    view = _conv_view(slot_ref, cfg)
    dtype = _conv_dtype(cfg)
    lanes = _lanes()
    used = cfg.prev_kernel_size * cfg.dim_size // lanes
    pack = _pack_rows(cfg)
    # Whole vector registers: the data rows, then the zero padding.
    head = min(view.shape[0], -(-used // pack) * pack)
    taps = list(taps)
    if head > used:
        taps.append(jnp.zeros((head - used, lanes), jnp.float32))
    data = jnp.concat(taps, axis=0).astype(dtype)
    if head == view.shape[0]:
        view[...] = data
        return
    view[pl.ds(0, head)] = data
    if head < view.shape[0]:
        view[pl.ds(head, view.shape[0] - head)] = jnp.zeros(
            (view.shape[0] - head, view.shape[1]), dtype
        )


def run(
    qkv_slot_ref: jax.Ref,  # [seq, chunk, 1, dim_size] FP32
    b_slot_ref: jax.Ref,  # [seq, chunk, 1, lanes] FP32
    a_slot_ref: jax.Ref,  # [seq, chunk, 1, lanes] FP32
    conv_in_ref: jax.Ref,  # [seq, 1, nblocks, *block]
    rec_in_ref: jax.Ref,  # [seq, 1, nblocks, *block]
    out_slot_ref: jax.Ref,  # [seq, chunk, num_v_heads, v_head_dim]
    conv_out_ref: jax.Ref,  # [seq, window, nblocks, *block]
    rec_out_ref: jax.Ref,  # [seq, window, nblocks, *block]
    metadata_ref: memory_ref.MetadataRef,
    weights_ref: memory_ref.WeightRefs,
    state_ref: jax.Ref,  # [seq, num_v_heads, kq_head_dim, v_head_dim] FP32
    lhs_ref: jax.Ref,  # [seq, chunk + prev_kernel_size, dim_size / 128, 128] FP32
    conv_w_ref: jax.Ref,  # [kernel_size + 1, dim_size / 128, 128] FP32
    *,
    cfg: config.GDNConfig,
    p_id: jax.Array,
) -> None:
    lanes = _lanes()
    rows = cfg.dim_size // lanes
    nv, nkq = cfg.num_v_heads, cfg.num_kq_heads
    ks, prev = cfg.kernel_size, cfg.prev_kernel_size
    compute = cfg.dtypes.compute
    q_scale = cfg.kq_head_dim**-0.5
    seqs = range(cfg.seq_tile_size)

    real_sizes = [metadata_ref.p_id_to_r_size[p_id, idx] for idx in seqs]
    has_initial_state = [
        metadata_ref.s_idx_has_initial_state[metadata_ref.p_id_to_s_idx[p_id, idx]]
        for idx in seqs
    ]

    # Conv inputs per sequence, dense and in the pool's stored row order: the
    # previous taps (zero for a fresh sequence), then the window's tokens.
    # The conv is per row, so with its weights in the same order it yields
    # the activations in that order and every checkpoint is a plain copy of
    # three input rows.
    for idx in seqs:
        prev_taps = jnp.where(
            has_initial_state[idx], _load_conv_taps(conv_in_ref.at[idx, 0], cfg), 0
        )
        for tap in range(prev):
            lhs_ref[idx, tap] = prev_taps[tap * rows : (tap + 1) * rows]
        for t in range(cfg.chunk_size):
            token = qkv_slot_ref.at[idx].reshape(-1, lanes)[pl.ds(t * rows, rows)]
            lhs_ref[idx, prev + t] = _to_stored(token, cfg)
    for k in range(ks):
        weight = weights_ref.conv.weight.at[k].reshape(-1, lanes)[...]
        conv_w_ref[k] = _to_stored(weight.astype(jnp.float32), cfg)
    if weights_ref.conv.bias is not None:
        # A 1D ref cannot be reshaped in Mosaic; reshape the loaded value.
        bias = weights_ref.conv.bias[...].reshape(-1, lanes)
        conv_w_ref[ks] = _to_stored(bias.astype(jnp.float32), cfg)
    order = _stored_rows(cfg)
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
        # A [1, lanes] row times one head's [kq_head_dim, v_head_dim] state.
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
            act = jnp.zeros((rows, lanes), jnp.float32)
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
                [lhs_ref[idx, start + j] for j in range(prev)],
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
                for head in range(
                    kq_head * cfg.v_per_kq_head, (kq_head + 1) * cfg.v_per_kq_head
                ):
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
                    vmem_ldst.store_state_head(
                        rec_out_ref, cfg, idx, pos, head, updated
                    )
            out_slot_ref[idx, pos] = jnp.concat(out_rows, axis=0).astype(
                out_slot_ref.dtype
            )

    for pos in range(cfg.chunk_size):
        position(pos, first=pos == 0)
    for idx in seqs:
        for t in range(cfg.window_size):
            vmem_ldst.zero_state_padding(rec_out_ref, cfg, idx, t)
