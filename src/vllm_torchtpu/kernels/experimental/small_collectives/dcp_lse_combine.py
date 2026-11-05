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
"""Single-kernel online-LSE DCP attention combine (flat and chip-pair-first).

At decode with context parallelism (`dcp_size = n`), each DCP rank holds an
attention partial ``bf16[T, H, D]`` and log-sum-exp ``f32[T, H]`` over all
``H = n * local`` query heads of its group, and needs the LSE-weighted combine
``bf16[T, local, D]`` for its own ``local`` heads.

The default reduce-scatter path (`cp_lse_ag_out_rs`) runs three steps: an
all-gather of ``lse`` over ``n`` ranks, ``lse_rescale_head_major`` into HBM,
and an ``n``-rank ``bf16`` reduce-scatter. At ``T = 64..80`` on 16 ranks that
pays a latency-bound collective for ``lse``, forces an HBM round-trip between
VMEM attention outputs and VMEM ``o_proj`` inputs, and sends un-folded ``bf16``
partials across all inter-chip links.

This module implements the combine as one TensorCore Pallas kernel in VMEM with
two exchange topologies:

    flat    Every core packs each destination rank's ``(lse, partial)`` slice
            into ``[16 + T * local, D]`` ``int16`` words (a 16-row header tile
            holding up to ``8 * D`` ``f32`` LSE scalars via sublane-preserving
            ``einshape`` + ``bitcast``, followed by token-major ``bf16``
            partials), pushes ``n - 1`` slices in rotated peer order, and runs
            an ``n``-way ``f32`` max-shifted LSE combine rounded once to
            ``bf16``.

    pairs   Two-stage hierarchical online-LSE combine for meshes whose adjacent
            indices ``(2c, 2c + 1)`` are the two cores of chip ``c``:
              Stage 0 (on-chip): Core ``2c + p`` pushes its opposite-parity
                ``(1 - p)`` head-block slices to its on-chip mate ``2c + (1-p)``
                in rotated chip order ``to = (c + 1 + s) % chips`` (zero
                inter-chip ICI traffic).
              Stage 1 (inter-chip, pipelined per block): As each block ``s``
                lands from the mate, core ``2c + p`` combines ``(own, mate)``
                via 2-way ``f32`` online LSE into normalized ``(L_chip, O_chip)``
                and immediately pushes that block to same-parity peer
                ``2 * to + p`` across the torus (halving inter-chip ICI bytes
                while overlapping intra-chip DMA, VPU combine, and inter-chip
                ICI DMA), then runs a ``chips``-way ``f32`` LSE combine into
                ``bf16[T, local, D]``.

With ``live_tokens`` (``int32[1]`` in SMEM, identical across ranks), trailing
padding tokens ``t >= live_tokens`` sit at the end of every token-major block:
both intra-chip and inter-chip DMAs move only ``16 + align16(live * local)``
rows and dead rows past ``live_tokens`` are zeroed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.moe_tp_decode.all_reduce import (
    check_chip_pairs,
)
from vllm_torchtpu.kernels.experimental.small_collectives.common import (
    LANES,
    MESH,
    wait_bytes_of,
)

# Distinct barrier semaphore group: MoE uses 0, all-reduce / fp8_oproj_ar 5,
# reduce-scatter 7, all-gather 9, small all-to-all 11, sparse MLA send 13,
# rotated collectives 15, prefill row all-gather 17, combine-scatter 19.
COLLECTIVE_ID = 21
HEADER_ROWS = 16
TILE_ROWS = 16
MAX_LIVE_STEPS = 8
DEFAULT_VMEM_LIMIT_BYTES = 32 * 1024 * 1024


def online_lse_combine_supported(
    tokens: int,
    num_heads: int,
    num_cores: int,
    vdim: int,
    mode: str = "pairs",
) -> bool:
    """Whether ``[tokens, num_heads, vdim]`` over ``num_cores`` ranks is valid
    for ``dcp_online_lse_combine`` in ``mode`` (``'pairs'`` or ``'flat'``)."""
    if mode not in ("pairs", "flat"):
        return False
    if (
        tokens < TILE_ROWS
        or tokens % TILE_ROWS != 0
        or num_cores < 2
        or num_heads < num_cores
        or num_heads % num_cores != 0
        or vdim < LANES
        or vdim % LANES != 0
    ):
        return False
    if mode == "pairs" and (num_cores < 4 or num_cores % 2 != 0):
        return False
    local = num_heads // num_cores
    return tokens * local <= 8 * vdim


def mesh_has_chip_pairs(mesh: jax.sharding.Mesh, axis: str) -> bool:
    """Whether every row of ``mesh`` along ``axis`` pairs adjacent indices
    ``(2c, 2c + 1)`` onto the two TensorCores of one TPU chip.

    Returns ``False`` on non-TPU backends (e.g. CPU virtual devices without
    ``coords`` / ``core_on_chip``) or when ``check_chip_pairs`` rejects a row.
    """
    if axis not in mesh.axis_names:
        return False
    n = mesh.shape[axis]
    if n < 4 or n % 2 != 0:
        return False
    axis_pos = mesh.axis_names.index(axis)
    devs = np.moveaxis(mesh.devices, axis_pos, -1).reshape(-1, n)
    for row in devs:
        if not all(hasattr(d, "coords") and hasattr(d, "core_on_chip") for d in row):
            return False
        places = [(tuple(d.coords), int(d.core_on_chip)) for d in row]
        try:
            check_chip_pairs(places)
        except ValueError:
            return False
    return True


def live_part_sizes(
    count: jax.Array, part_rows: int, step_rows: int = TILE_ROWS
) -> list[tuple[int, jax.Array]]:
    """Static ``(part_size, taken)`` pairs for multiples of ``step_rows`` up to
    ``part_rows`` (always including ``part_rows``)."""
    sizes = list(range(step_rows, part_rows, step_rows))
    if not sizes or sizes[-1] != part_rows:
        sizes.append(part_rows)
    return [(size, count == size) for size in sizes]


def reference_float64(
    partials: np.ndarray | jax.Array,
    lses: np.ndarray | jax.Array,
    live_tokens: int | None = None,
) -> np.ndarray:
    """Exact IEEE float64 LSE-weighted combine across ``world`` ranks, rounded
    once to ``bfloat16`` (returned as float32 numpy array of bf16-rounded
    values, shape ``[world, T, local, D]``).

    Args:
      partials: ``[world, T, H, D]`` partial attention outputs.
      lses: ``[world, T, H]`` float32 log-sum-exp values (``-inf`` when empty).
      live_tokens: optional valid token count ``<= T``; rows ``>= live_tokens``
        are zeroed.
    """
    l64 = np.asarray(lses, dtype=np.float64)
    p64 = np.asarray(jax.device_get(partials).astype(jnp.float32), dtype=np.float64)
    world, tokens, heads, vdim = p64.shape
    local = heads // world
    finite = np.isfinite(l64)
    shift = np.max(np.where(finite, l64, -np.inf), axis=0, keepdims=True)
    shift_safe = np.where(np.isfinite(shift), shift, 0.0)
    weights = np.where(finite, np.exp(l64 - shift_safe), 0.0)
    denom = np.sum(weights, axis=0, keepdims=True)
    norm_w = np.where(denom > 0.0, weights / np.where(denom > 0.0, denom, 1.0), 0.0)
    p_clean = np.where(np.isnan(p64) | ~finite[..., None], 0.0, p64)
    merged = np.sum(p_clean * norm_w[..., None], axis=0)  # [T, H, D]
    if live_tokens is not None:
        merged[max(0, int(live_tokens)) :] = 0.0
    bf16 = jnp.asarray(merged, dtype=jnp.float32).astype(jnp.bfloat16)
    scattered = bf16.reshape(tokens, world, local, vdim).transpose(1, 0, 2, 3)
    return np.asarray(scattered.astype(jnp.float32))


def reference_pairs_f32(
    partials: np.ndarray | jax.Array,
    lses: np.ndarray | jax.Array,
    live_tokens: int | None = None,
) -> np.ndarray:
    """Exact two-stage float32/bfloat16 chip-pair oracle matching ``mode='pairs'``:
    Stage 0 merges ``(2c, 2c + 1)`` in ``float32`` to ``(L_chip f32, O_chip bf16)``,
    Stage 1 merges ``chips`` pairs in ``float32`` to ``bf16[world, T, local, D]``."""
    p_f32 = np.asarray(jax.device_get(partials).astype(jnp.float32), dtype=np.float32)
    l_f32 = np.asarray(lses, dtype=np.float32)
    world, tokens, heads, vdim = p_f32.shape
    assert world % 2 == 0
    chips = world // 2
    local = heads // world

    pa = p_f32[0::2]  # [chips, T, H, D]
    pb = p_f32[1::2]
    la = l_f32[0::2]  # [chips, T, H]
    lb = l_f32[1::2]
    fa = np.isfinite(la)
    fb = np.isfinite(lb)
    m = np.maximum(np.where(fa, la, -np.inf), np.where(fb, lb, -np.inf))
    m_safe = np.where(np.isfinite(m), m, 0.0)
    ea = np.where(fa, np.exp(la - m_safe), 0.0)
    eb = np.where(fb, np.exp(lb - m_safe), 0.0)
    denom = ea + eb
    l_chip = np.where(
        denom > 0.0, m_safe + np.log(np.where(denom > 0.0, denom, 1.0)), -np.inf
    )
    safe_d = np.where(denom > 0.0, denom, np.float32(1.0))
    inv_d = np.where(denom > 0.0, np.float32(1.0) / safe_d, np.float32(0.0))
    pa_clean = np.where(np.isnan(pa) | ~fa[..., None], np.float32(0.0), pa)
    pb_clean = np.where(np.isnan(pb) | ~fb[..., None], np.float32(0.0), pb)
    o_chip_f32 = pa_clean * (ea * inv_d)[..., None] + pb_clean * (eb * inv_d)[..., None]
    o_chip = np.asarray(
        jnp.asarray(o_chip_f32, dtype=jnp.bfloat16).astype(jnp.float32),
        dtype=np.float32,
    )

    fc = np.isfinite(l_chip)
    mc = np.max(np.where(fc, l_chip, -np.inf), axis=0, keepdims=True)
    mc_safe = np.where(np.isfinite(mc), mc, 0.0)
    ec = np.where(fc, np.exp(l_chip - mc_safe), 0.0)
    dc = np.sum(ec, axis=0, keepdims=True)
    inv_dc = np.where(dc > 0.0, np.float32(1.0) / np.where(dc > 0.0, dc, 1.0), 0.0)
    acc = np.zeros((tokens, heads, vdim), dtype=np.float32)
    for c_idx in range(chips):
        acc = acc + o_chip[c_idx] * (ec[c_idx] * inv_dc[0])[..., None]
    if live_tokens is not None:
        acc[max(0, int(live_tokens)) :] = 0.0
    bf16 = jnp.asarray(acc, dtype=jnp.bfloat16)
    scattered = bf16.reshape(tokens, world, local, vdim).transpose(1, 0, 2, 3)
    return np.asarray(scattered.astype(jnp.float32))


def _extract_block_lse_col(
    lse_val: jax.Array, h0: int, tokens: int, local: int
) -> jax.Array:
    """Extract token-major ``f32[tokens * local, 1]`` for heads
    ``[h0, h0 + local)`` using only ``(..., 1)`` and ``(..., 128)`` 2D tiles."""
    if local == 1:
        return lse_val[:, h0 : h0 + 1]
    wide = jnp.concatenate(
        [
            jnp.broadcast_to(lse_val[:, h0 + i : h0 + i + 1], (tokens, LANES))
            for i in range(local)
        ],
        axis=1,
    )
    return pltpu.einshape("t(lm)->(tl)m", wide, l=local, m=LANES)[:, :1]


def _pack_lse_header(l_col: jax.Array, part_rows: int, vdim: int) -> jax.Array:
    """Pack 2D column ``f32[part_rows, 1]`` (``part_rows <= 8 * vdim``) into one
    ``int16[16, vdim]`` header tile via native ``(128, 128)`` transposes."""
    n_chunks = -(-part_rows // LANES)
    pad_rows = n_chunks * LANES - part_rows
    l_pad = (
        jnp.pad(l_col, ((0, pad_rows), (0, 0)), constant_values=-jnp.inf)
        if pad_rows > 0
        else l_col
    )
    n_lane_blocks = vdim // LANES
    sub = lax.broadcasted_iota(jnp.int32, (8, LANES), 0)
    lane_blocks = []
    for b in range(n_lane_blocks):
        blk = jnp.full((8, LANES), -jnp.inf, jnp.float32)
        rem = min(8, max(0, n_chunks - b * 8))
        for r in range(rem):
            c = b * 8 + r
            col_c = l_pad[c * LANES : (c + 1) * LANES, :]
            tr_c = jnp.broadcast_to(col_c, (LANES, LANES)).T[:8, :]
            blk = jnp.where(sub == r, tr_c, blk)
        lane_blocks.append(blk)
    hdr_f32 = (
        jnp.concatenate(lane_blocks, axis=1)
        if len(lane_blocks) > 1
        else lane_blocks[0]
    )
    return pltpu.bitcast(hdr_f32, jnp.int16)


def _unpack_lse_header(hdr_i16: jax.Array, part_rows: int) -> jax.Array:
    """Unpack ``int16[16, vdim]`` header tile back to 2D ``f32[part_rows, 1]``."""
    hdr_f32 = pltpu.bitcast(hdr_i16, jnp.float32)
    n_chunks = -(-part_rows // LANES)
    cols = []
    for c in range(n_chunks):
        b, r = divmod(c, 8)
        row_c = hdr_f32[r : r + 1, b * LANES : (b + 1) * LANES]
        col_c = jnp.broadcast_to(row_c, (LANES, LANES)).T[:, :1]
        cols.append(col_c)
    full_col = jnp.concatenate(cols, axis=0) if len(cols) > 1 else cols[0]
    return full_col[:part_rows, :]


def _clean_part(part_i16: jax.Array, live_mask_col: jax.Array | None) -> jax.Array:
    """Bitcast ``int16[rows, vdim]`` to ``bf16``, zero NaNs (and dead rows when
    ``live_mask_col`` is given), and widen to ``float32``."""
    part = pltpu.bitcast(part_i16, jnp.bfloat16)
    valid = ~jnp.isnan(part)
    if live_mask_col is not None:
        valid = valid & live_mask_col
    return jnp.where(valid, part, jnp.bfloat16(0)).astype(jnp.float32)


def _merge_two_blocks(
    own_hdr: jax.Array,
    own_prt: jax.Array,
    mate_hdr: jax.Array,
    mate_prt: jax.Array,
    *,
    part_rows: int,
    vdim: int,
    live_mask_col: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """2-way float32 online-LSE combine of two ``(hdr, part)`` peer blocks."""
    la = _unpack_lse_header(own_hdr, part_rows)
    lb = _unpack_lse_header(mate_hdr, part_rows)
    if live_mask_col is not None:
        la = jnp.where(live_mask_col, la, -jnp.inf)
        lb = jnp.where(live_mask_col, lb, -jnp.inf)
    fa = jnp.isfinite(la)
    fb = jnp.isfinite(lb)
    m = jnp.maximum(jnp.where(fa, la, -jnp.inf), jnp.where(fb, lb, -jnp.inf))
    m_safe = jnp.where(jnp.isfinite(m), m, 0.0)
    ea = jnp.where(fa, jnp.exp(la - m_safe), 0.0)
    eb = jnp.where(fb, jnp.exp(lb - m_safe), 0.0)
    denom = ea + eb
    pos = denom > 0.0
    denom_safe = jnp.where(pos, denom, 1.0)
    l_out = jnp.where(pos, m_safe + jnp.log(denom_safe), -jnp.inf)
    inv_d = jnp.where(pos, 1.0 / denom_safe, 0.0)
    pa = _clean_part(own_prt, live_mask_col)
    pb = _clean_part(mate_prt, live_mask_col)
    o_out = (pa * (ea * inv_d) + pb * (eb * inv_d)).astype(jnp.bfloat16)
    return _pack_lse_header(l_out, part_rows, vdim), pltpu.bitcast(o_out, jnp.int16)


def _merge_n_slots(
    recv_ref,
    num_senders: int,
    *,
    part_rows: int,
    vdim: int,
    live_mask_col: jax.Array | None,
) -> jax.Array:
    """``num_senders``-way float32 max-shifted LSE combine over ``recv_ref``
    slots ``[num_senders, 16 + part_rows, vdim]``, rounded once to ``bf16``."""
    lses = []
    for k in range(num_senders):
        lk = _unpack_lse_header(
            recv_ref[k, pl.ds(0, HEADER_ROWS), :], part_rows
        )
        if live_mask_col is not None:
            lk = jnp.where(live_mask_col, lk, -jnp.inf)
        lses.append(lk)

    l_max = jnp.where(jnp.isfinite(lses[0]), lses[0], -jnp.inf)
    for k in range(1, num_senders):
        l_max = jnp.maximum(l_max, jnp.where(jnp.isfinite(lses[k]), lses[k], -jnp.inf))
    m_safe = jnp.where(jnp.isfinite(l_max), l_max, 0.0)

    exps = [
        jnp.where(jnp.isfinite(lses[k]), jnp.exp(lses[k] - m_safe), 0.0)
        for k in range(num_senders)
    ]
    denom = exps[0]
    for k in range(1, num_senders):
        denom = denom + exps[k]
    inv_d = jnp.where(denom > 0.0, 1.0 / jnp.where(denom > 0.0, denom, 1.0), 0.0)

    acc = jnp.zeros((part_rows, vdim), jnp.float32)
    for k in range(num_senders):
        pk = _clean_part(recv_ref[k, pl.ds(HEADER_ROWS, part_rows), :], live_mask_col)
        acc = acc + pk * (exps[k] * inv_d)
    if live_mask_col is not None:
        acc = jnp.where(live_mask_col, acc, 0.0)
    return acc.astype(jnp.bfloat16)


def dcp_online_lse_combine(
    partial: jax.Array,
    lse: jax.Array,
    *,
    num_cores: int,
    axis: str,
    rank: jax.Array | None = None,
    lane_axis: str | None = None,
    lane: jax.Array | None = None,
    live_tokens: jax.Array | None = None,
    mode: str = "pairs",
    out_vmem: bool = True,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
    collective_id: int = COLLECTIVE_ID,
    interpret: bool | pltpu.InterpretParams = False,
) -> jax.Array:
    """Combine ``partial`` ``bf16[T, H, D]`` and ``lse`` ``f32[T, H]`` across
    ``num_cores`` ranks of ``axis`` into ``bf16[T, H // num_cores, D]`` inside a
    ``shard_map``.

    Args:
      partial: ``bf16[T, H, D]``, ``T`` a multiple of 16, ``D`` a multiple of 128.
      lse: ``f32[T, H]``, per-token per-head log-sum-exp (``-inf`` when empty).
      num_cores: size of mesh axis ``axis``.
      axis: DCP mesh axis name.
      rank: optional ``int32[1]`` core index on ``axis`` (avoids ``partition-id``
        under SPMD export); ``None`` uses ``lax.axis_index(axis)``.
      lane_axis: optional second mesh axis name (e.g. ``'dcp_lane'``).
      lane: optional ``int32[1]`` index on ``lane_axis`` (when ``lane_axis`` has
        size 1, ``lane`` may be ``None`` and constant ``0`` is used when
        ``rank`` is provided).
      live_tokens: optional ``int32[1]`` valid token count (same across ranks).
      mode: ``'pairs'`` (chip-pair-first 2-stage) or ``'flat'`` (1-stage).
      out_vmem: when ``True``, keep the ``[T, local, D]`` output in VMEM so a
        downstream VMEM consumer (e.g. ``fp8_uv_oproj_all_reduce`` /
        ``fp8_oproj_all_reduce``) avoids an HBM round-trip.
      vmem_limit_bytes: scoped VMEM limit for Mosaic compilation.
      collective_id: barrier semaphore ID (default ``13``).
      interpret: Pallas CPU interpret mode for unit testing.
    """
    if partial.ndim != 3 or lse.ndim != 2:
        raise ValueError(
            f"dcp_online_lse_combine expects partial [T, H, D] and lse [T, H], "
            f"got {partial.shape} and {lse.shape}"
        )
    if partial.dtype != jnp.bfloat16 or lse.dtype != jnp.float32:
        raise ValueError(
            f"dcp_online_lse_combine expects bf16 partial and f32 lse, "
            f"got {partial.dtype} and {lse.dtype}"
        )
    tokens, num_heads, vdim = partial.shape
    if lse.shape != (tokens, num_heads):
        raise ValueError(
            f"lse shape {lse.shape} does not match partial {partial.shape[:2]}"
        )
    if not online_lse_combine_supported(tokens, num_heads, num_cores, vdim, mode=mode):
        raise ValueError(
            f"Unsupported dcp_online_lse_combine geometry: tokens={tokens}, "
            f"heads={num_heads}, cores={num_cores}, vdim={vdim}, mode={mode!r}"
        )

    local = num_heads // num_cores
    part_rows = tokens * local
    block_rows = HEADER_ROWS + part_rows
    data_rank = rank is not None
    data_lane = lane_axis is not None and lane is not None
    sized = live_tokens is not None

    def pop_headers(refs_list):
        me = refs_list.pop(0)[0] if data_rank else lax.axis_index(axis)
        if lane_axis is None:
            my_lane = None
        elif data_lane:
            my_lane = refs_list.pop(0)[0]
        elif data_rank:
            # Single-lane mesh (shape[lane_axis] == 1) under torch export:
            # constant 0 avoids emitting lax.axis_index / partition-id.
            my_lane = jnp.int32(0)
        else:
            my_lane = lax.axis_index(lane_axis)
        if sized:
            live = jnp.clip(refs_list.pop(0)[0], 0, tokens)
            live_elems = live * local
            step_tokens = (
                max(1, -(-(tokens // TILE_ROWS) // MAX_LIVE_STEPS)) * TILE_ROWS
            )
            step_rows = step_tokens * local
            if step_rows >= part_rows:
                sizes = None
            else:
                dma_part = jnp.minimum(
                    jnp.maximum(
                        ((live_elems + step_rows - 1) // step_rows) * step_rows,
                        step_rows,
                    ),
                    part_rows,
                )
                sizes = live_part_sizes(dma_part, part_rows, step_rows=step_rows)
            row_idx = lax.broadcasted_iota(jnp.int32, (part_rows, 1), 0)
            live_mask_col = row_idx < live_elems
        else:
            sizes = None
            live_mask_col = None
        return me, my_lane, sizes, live_mask_col

    def make_peer_fn(my_lane):
        def peer_id(j):
            if not data_rank and lane_axis is None:
                return (jnp.int32(j),)
            ids = {axis: jnp.int32(j)}
            if lane_axis is not None:
                ids[lane_axis] = my_lane
            return ids

        return peer_id

    def run_barrier(peer_id):
        sem = pltpu.get_barrier_semaphore()
        for j in range(num_cores):
            pl.semaphore_signal(sem, 1, device_id=peer_id(j), device_id_type=MESH)
        pl.semaphore_wait(sem, num_cores)

    def slice_block(buf_ref, idx, part_sz):
        total_sz = HEADER_ROWS + part_sz
        if total_sz == block_rows:
            return buf_ref.at[idx]
        return buf_ref.at[idx, pl.ds(0, total_sz)]

    def span_blocks(buf_ref, copies: int, part_sz: int):
        total_sz = HEADER_ROWS + part_sz
        if total_sz == block_rows:
            return buf_ref.at[pl.ds(0, copies)]
        return buf_ref.at[pl.ds(0, copies), pl.ds(0, total_sz)]

    if mode == "flat":

        def flat_kernel(*refs):
            refs_list = list(refs)
            me, my_lane, sizes, live_mask_col = pop_headers(refs_list)
            if not out_vmem:
                out_sem = refs_list.pop()
                v_out = refs_list.pop()
            (
                part_ref,
                lse_ref,
                out_ref,
                send_buf,
                recv_buf,
                send_sem,
                recv_sem,
                loc_sem,
            ) = refs_list
            peer_id = make_peer_fn(my_lane)
            run_barrier(peer_id)

            p_wtld = pltpu.einshape(
                "t(wl)d->w(tl)d", part_ref[...], w=num_cores, l=local
            )
            lse_val = lse_ref[...]
            for d in range(num_cores):
                l_col = _extract_block_lse_col(lse_val, d * local, tokens, local)
                send_buf[d, pl.ds(0, HEADER_ROWS), :] = _pack_lse_header(
                    l_col, part_rows, vdim
                )
                send_buf[d, pl.ds(HEADER_ROWS, part_rows), :] = pltpu.bitcast(
                    p_wtld[d], jnp.int16
                )

            def launch_flat(part_sz: int):
                dst_mine = slice_block(recv_buf, me, part_sz)
                for s in range(1, num_cores):
                    to = lax.rem(me + s, num_cores)
                    pltpu.make_async_remote_copy(
                        src_ref=slice_block(send_buf, to, part_sz),
                        dst_ref=dst_mine,
                        send_sem=send_sem,
                        recv_sem=recv_sem,
                        device_id=peer_id(to),
                        device_id_type=MESH,
                    ).start()
                pltpu.make_async_copy(
                    slice_block(send_buf, me, part_sz), dst_mine, loc_sem
                ).start()

            def wait_flat(sem, copies: int):
                if sizes is None:
                    wait_bytes_of(span_blocks(recv_buf, copies, part_rows), sem)
                    return
                for part_sz, taken in sizes:

                    @pl.when(taken)
                    def _(part_sz=part_sz):
                        wait_bytes_of(span_blocks(recv_buf, copies, part_sz), sem)

            if sizes is None:
                launch_flat(part_rows)
            else:
                for part_sz, taken in sizes:

                    @pl.when(taken)
                    def _(part_sz=part_sz):
                        launch_flat(part_sz)

            wait_flat(recv_sem, num_cores - 1)
            wait_flat(loc_sem, 1)

            merged = _merge_n_slots(
                recv_buf,
                num_cores,
                part_rows=part_rows,
                vdim=vdim,
                live_mask_col=live_mask_col,
            )
            relayout = pltpu.einshape("(tl)d->tld", merged, t=tokens, l=local)
            if out_vmem:
                out_ref[...] = relayout
                wait_flat(send_sem, num_cores - 1)
            else:
                v_out[...] = relayout
                out_cp = pltpu.make_async_copy(v_out, out_ref, out_sem)
                out_cp.start()
                wait_flat(send_sem, num_cores - 1)
                out_cp.wait()

        kernel_fn = flat_kernel
        scratch = [
            pltpu.VMEM((num_cores, block_rows, vdim), jnp.int16),
            pltpu.VMEM((num_cores, block_rows, vdim), jnp.int16),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ]
    else:
        chips = num_cores // 2

        def pairs_kernel(*refs):
            refs_list = list(refs)
            me, my_lane, sizes, live_mask_col = pop_headers(refs_list)
            if not out_vmem:
                out_sem = refs_list.pop()
                v_out = refs_list.pop()
            (
                part_ref,
                lse_ref,
                out_ref,
                own_half,
                mate_send,
                mate_recv,
                chip_recv,
                s0,
                r0,
                s1,
                r1,
                loc_sem,
            ) = refs_list
            parity = lax.rem(me, 2)
            chip = lax.div(me, 2)
            mate = lax.bitwise_xor(me, 1)
            peer_id = make_peer_fn(my_lane)
            run_barrier(peer_id)

            # One hardware relayout splits even/odd head-blocks:
            # q=0 holds blocks {0, 2, ..., 2*(chips-1)}, q=1 holds {1, 3, ...}.
            p_qctld = pltpu.einshape(
                "t(cql)d->qc(tl)d", part_ref[...], c=chips, q=2, l=local
            )
            lse_val = lse_ref[...]
            for i in range(chips):
                l_even = _extract_block_lse_col(
                    lse_val, (2 * i) * local, tokens, local
                )
                l_odd = _extract_block_lse_col(
                    lse_val, (2 * i + 1) * local, tokens, local
                )
                h_even = _pack_lse_header(l_even, part_rows, vdim)
                h_odd = _pack_lse_header(l_odd, part_rows, vdim)
                b_even = pltpu.bitcast(p_qctld[0, i], jnp.int16)
                b_odd = pltpu.bitcast(p_qctld[1, i], jnp.int16)
                own_half[i, pl.ds(0, HEADER_ROWS), :] = jnp.where(
                    parity == 0, h_even, h_odd
                )
                own_half[i, pl.ds(HEADER_ROWS, part_rows), :] = jnp.where(
                    parity == 0, b_even, b_odd
                )
                mate_send[i, pl.ds(0, HEADER_ROWS), :] = jnp.where(
                    parity == 0, h_odd, h_even
                )
                mate_send[i, pl.ds(HEADER_ROWS, part_rows), :] = jnp.where(
                    parity == 0, b_odd, b_even
                )

            def push_mate_step(s: int, to: jax.Array, part_sz: int):
                pltpu.make_async_remote_copy(
                    src_ref=slice_block(mate_send, to, part_sz),
                    dst_ref=slice_block(mate_recv, to, part_sz),
                    send_sem=s0,
                    recv_sem=r0.at[s],
                    device_id=peer_id(mate),
                    device_id_type=MESH,
                ).start()

            def push_chip_step(to: jax.Array, is_remote: bool, part_sz: int):
                src = slice_block(own_half, to, part_sz)
                dst = slice_block(chip_recv, chip, part_sz)
                if is_remote:
                    pltpu.make_async_remote_copy(
                        src_ref=src,
                        dst_ref=dst,
                        send_sem=s1,
                        recv_sem=recv_chip_sem,
                        device_id=peer_id(2 * to + parity),
                        device_id_type=MESH,
                    ).start()
                else:
                    pltpu.make_async_copy(src, dst, loc_sem).start()

            recv_chip_sem = r1

            def wait_blocks(buf_ref, sem, copies: int):
                if sizes is None:
                    wait_bytes_of(span_blocks(buf_ref, copies, part_rows), sem)
                    return
                for part_sz, taken in sizes:

                    @pl.when(taken)
                    def _(part_sz=part_sz):
                        wait_bytes_of(span_blocks(buf_ref, copies, part_sz), sem)

            # Stage 0: enqueue all intra-chip mate pushes in rotated chip order
            # (identical rotation `to = (chip + 1 + s) % chips` on both cores
            # of chip `c`).
            for s in range(chips):
                to = lax.rem(chip + 1 + s, chips)
                if sizes is None:
                    push_mate_step(s, to, part_rows)
                else:
                    for part_sz, taken in sizes:

                        @pl.when(taken)
                        def _(s=s, to=to, part_sz=part_sz):
                            push_mate_step(s, to, part_sz)

            # Stage 1 (pipelined): as each mate block lands, combine in f32
            # into `own_half[to]` and immediately push over ICI to `2*to+parity`.
            for s in range(chips):
                to = lax.rem(chip + 1 + s, chips)
                wait_blocks(mate_recv, r0.at[s], 1)
                c_hdr, c_prt = _merge_two_blocks(
                    own_half[to, pl.ds(0, HEADER_ROWS), :],
                    own_half[to, pl.ds(HEADER_ROWS, part_rows), :],
                    mate_recv[to, pl.ds(0, HEADER_ROWS), :],
                    mate_recv[to, pl.ds(HEADER_ROWS, part_rows), :],
                    part_rows=part_rows,
                    vdim=vdim,
                    live_mask_col=live_mask_col,
                )
                own_half[to, pl.ds(0, HEADER_ROWS), :] = c_hdr
                own_half[to, pl.ds(HEADER_ROWS, part_rows), :] = c_prt
                is_remote = s < chips - 1
                if sizes is None:
                    push_chip_step(to, is_remote, part_rows)
                else:
                    for part_sz, taken in sizes:

                        @pl.when(taken)
                        def _(to=to, is_remote=is_remote, part_sz=part_sz):
                            push_chip_step(to, is_remote, part_sz)

            wait_blocks(chip_recv, r1, chips - 1)
            wait_blocks(chip_recv, loc_sem, 1)

            merged = _merge_n_slots(
                chip_recv,
                chips,
                part_rows=part_rows,
                vdim=vdim,
                live_mask_col=live_mask_col,
            )
            relayout = pltpu.einshape("(tl)d->tld", merged, t=tokens, l=local)
            if out_vmem:
                out_ref[...] = relayout
                wait_blocks(mate_recv, s0, chips)
                wait_blocks(chip_recv, s1, chips - 1)
            else:
                v_out[...] = relayout
                out_cp = pltpu.make_async_copy(v_out, out_ref, out_sem)
                out_cp.start()
                wait_blocks(mate_recv, s0, chips)
                wait_blocks(chip_recv, s1, chips - 1)
                out_cp.wait()

        kernel_fn = pairs_kernel
        scratch = [
            pltpu.VMEM((chips, block_rows, vdim), jnp.int16),
            pltpu.VMEM((chips, block_rows, vdim), jnp.int16),
            pltpu.VMEM((chips, block_rows, vdim), jnp.int16),
            pltpu.VMEM((chips, block_rows, vdim), jnp.int16),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA((chips,)),
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
            pltpu.SemaphoreType.DMA,
        ]

    if not out_vmem:
        scratch = [
            *scratch,
            pltpu.VMEM((tokens, local, vdim), jnp.bfloat16),
            pltpu.SemaphoreType.DMA,
        ]

    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)

    scalars = []
    if data_rank:
        scalars.append(jnp.reshape(rank, (-1,))[:1].astype(jnp.int32))
    if data_lane:
        scalars.append(jnp.reshape(lane, (-1,))[:1].astype(jnp.int32))
    if sized:
        scalars.append(jnp.reshape(live_tokens, (-1,))[:1].astype(jnp.int32))

    name = (
        f"dcp_online_lse_combine_{mode}_n{num_cores}_t{tokens}_h{num_heads}_d{vdim}"
        + ("_live" if sized else "")
        + ("_ov" if out_vmem else "")
    )
    return pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((tokens, local, vdim), jnp.bfloat16),
        in_specs=[smem] * len(scalars) + [vmem, vmem],
        out_specs=vmem if out_vmem else hbm,
        scratch_shapes=scratch,
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id,
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        interpret=interpret,
        name=name,
    )(*scalars, partial, lse)
