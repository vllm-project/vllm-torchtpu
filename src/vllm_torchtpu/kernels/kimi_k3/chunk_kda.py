# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked Kimi Delta Attention (KDA) Pallas TPU prefill kernel.

Adated from the implementation from the following tokamax PR:
https://github.com/openxla/tokamax/pull/1103

Two Pallas passes over a shared (sequence, tile) grid:

1. ``pallas_kda_fwd_intra_fused`` -- gate activation, chunk-local prefix sum and
   the intra-chunk solve, producing ``w/u/kg/Aqk/g_cumsum`` per chunk.
2. ``chunk_kda_fwd_h_o_varlen`` -- the inter-chunk recurrence and the output.

Both grid over ``(head block, batch, tile)``, where a tile holds up to
``chunk_size`` tokens of exactly one segment. The tile axis is static at the
worst-case tile count while a device-valued ``num_tiles`` bounds the real work,
so padding the runner's request axis costs skipped tiles rather than real ones.
"""

from __future__ import annotations

import functools
import math
import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import varlen_tiles

__all__ = ["chunk_kda", "restrict_cu_seqlens"]


def exp2(x):
    """Base-2 exponential, matching Triton's tl.exp2."""
    return jnp.exp2(x.astype(jnp.float32))


def get_interpret() -> bool:
    env = os.environ.get("PALLAS_INTERPRET", "")
    return env.strip().lower() in ("1", "true")


def cdiv(x, y: int):
    return (x + y - 1) // y


def align_up(x, align: int):
    return cdiv(x, align) * align


def segment_ids_to_seqlens(
    segment_ids: jax.Array,
    max_segs: int,
    chunk_size: int = 1,
) -> jax.Array:
    """Convert 1-indexed segment IDs with 0 padding to FLA-style cu_seqlens."""
    if segment_ids.ndim == 2:
        rows = [
            segment_ids_to_seqlens(segment_ids[b], max_segs, chunk_size)
            for b in range(segment_ids.shape[0])
        ]
        return jnp.stack(rows, axis=0)
    if segment_ids.ndim != 1:
        raise ValueError(
            f"`segment_ids` must be [T] or [B, T], got {segment_ids.shape}.")

    seg = segment_ids.reshape(-1)
    valid = seg != 0
    is_boundary = jnp.concatenate([
        jnp.ones(1, dtype=jnp.bool_),
        seg[1:] != seg[:-1],
    ]) & valid
    seg_idx = jnp.where(valid, jnp.cumsum(is_boundary.astype(jnp.int32)), 0)
    n_segs = seg_idx.max()
    n_real = jnp.sum(valid).astype(jnp.int32)

    is_end = jnp.concatenate([
        seg[1:] != seg[:-1],
        jnp.ones(1, dtype=jnp.bool_),
    ]) & valid
    prefix_len = jnp.cumsum(valid.astype(jnp.int32))

    drop_idx = jnp.asarray(max_segs + 1, dtype=jnp.int32)
    scatter_idx = jnp.where(is_end, seg_idx, drop_idx)
    scatter_val = jnp.where(is_end, prefix_len, 0)

    cu_seqlens = jnp.zeros((max_segs + 1, ), dtype=jnp.int32)
    cu_seqlens = cu_seqlens.at[scatter_idx].max(scatter_val, mode="drop")
    out_idx = jnp.arange(max_segs + 1, dtype=jnp.int32)
    return jnp.where(out_idx > n_segs, n_real, cu_seqlens)


class _TilePlan(NamedTuple):
    """A (sequence, tile) walk over the ragged token axis.

    Every tile holds up to ``chunk_size`` tokens of exactly one segment, so no
    tile straddles a segment boundary.

    Attributes:
        num_tiles: ``[B]``; how many tiles the batch's real content needs.
        r_base, r_size: ``[B, NT]``; each tile's first row on the token axis and
            how many rows it holds.
        s_idx: ``[B, NT]``; each tile's segment, which is where it reads and
            writes recurrent state.
        is_first, is_last: ``[B, NT]``; where a segment's state is read in and
            written back.
    """
    num_tiles: jax.Array
    r_base: jax.Array
    r_size: jax.Array
    s_idx: jax.Array
    is_first: jax.Array
    is_last: jax.Array


def restrict_cu_seqlens(
    cu_seqlens: jax.Array,  # [..., N + 1]
    start_seq: jax.Array | int,
) -> jax.Array:
    """Collapse segments before ``start_seq`` to zero length.

    ``cu_seqlens`` is non-decreasing, so clamping it from below at its own
    ``start_seq`` entry makes every earlier boundary equal and leaves every later
    one untouched. Segments before ``start_seq`` then hold no tokens and get no
    tiles, while the ones after keep both their absolute token offsets and their
    absolute segment indices -- which is what a caller needs to keep addressing
    recurrent state by request slot.

    This is how a caller restricts the chunked kernel to the prefill part of a
    batch, with the boundary a *device* scalar: a host-side branch on batch
    composition cannot work under vLLM's guard-free single trace.
    """
    floor = jnp.take(cu_seqlens, jnp.asarray(start_seq, jnp.int32),
                     axis=-1)[..., None]
    return jnp.maximum(cu_seqlens, floor)


def _plan_tiles(
    cu_seqlens: jax.Array,  # [B, N + 1]
    *,
    num_tokens: int,
    chunk_size: int,
) -> _TilePlan:
    """Plan one tile per ``chunk_size`` tokens of each segment."""
    batch, num_cu = cu_seqlens.shape
    num_segments = num_cu - 1
    # A tile holds at least one token, so the token count bounds the tile count;
    # and each segment rounds its own token count up to a whole tile, which adds
    # at most one tile per segment.
    max_tiles = min(num_tokens, cdiv(num_tokens, chunk_size) + num_segments)

    query_lens = jnp.diff(cu_seqlens, axis=-1)
    num_tiles, r_base, r_size, s_idx = [], [], [], []
    is_first, is_last = [], []
    for index in range(batch):
        plan = varlen_tiles.plan_per_seq_tiles(
            # Passing `query_lens` as `seq_lens` makes the planner's
            # `s_idx_has_initial_state` uniformly False, and it is unused here:
            # whether a segment carries state in is static for this kernel --
            # `initial_state` is either given for every slot, pre-zeroed by the
            # caller for the fresh ones, or absent altogether.
            seq_lens=query_lens[index],
            query_start_loc=cu_seqlens[index],
            max_tiles=max_tiles,
            chunk_size=chunk_size,
            tile_size=chunk_size,
        )
        valid = jnp.arange(max_tiles) < plan.num_tiles
        num_tiles.append(plan.num_tiles)
        r_base.append(jnp.where(valid, plan.p_id_to_r_base, 0))
        r_size.append(jnp.where(valid, plan.p_id_to_r_size, 0))
        s_idx.append(jnp.where(valid, plan.p_id_to_s_idx, num_segments - 1))
        is_first.append(valid & plan.p_id_is_first_tile)
        is_last.append(valid & plan.p_id_is_last_tile)

    return _TilePlan(
        num_tiles=jnp.stack(num_tiles).astype(jnp.int32),
        r_base=jnp.stack(r_base).astype(jnp.int32),
        r_size=jnp.stack(r_size).astype(jnp.int32),
        s_idx=jnp.stack(s_idx).astype(jnp.int32),
        is_first=jnp.stack(is_first),
        is_last=jnp.stack(is_last),
    )


_VMEM_FRACTION = 0.9

RCP_LN2 = 1.0 / math.log(2)

# =============================================================================
# Mini-batch sizing and shared forward recurrence
# =============================================================================


def estimate_mini_batch(
    per_tile_bytes: int,
    total: int,
    *,
    max_mb: int = 16,
    vmem_budget: int | None = None,
) -> int:
    """Choose the largest mini-batch that fits VMEM and divides ``total``."""
    if vmem_budget is None:
        vmem_budget = int(_VMEM_FRACTION *
                          pltpu.get_tpu_info().vmem_capacity_bytes)

    per_tile_bytes = max(1, per_tile_bytes)
    MB = max(1, vmem_budget // per_tile_bytes)
    MB = max(1, min(MB, total, max_mb))

    while total % MB != 0 and MB > 1:
        MB -= 1

    return MB


# =============================================================================
# Fused gate cumsum and intra-chunk solve
# =============================================================================


def _fused_gate_intra_tile(
    q,  # [MB, BT, K]
    k,  # [MB, BT, K]
    g,  # [MB, BT, K]
    beta,  # [MB, BT, 1]
    v,  # [MB, BT, V]
    A_val,  # [MB] float32
    dt_b,  # [MB, K] float32
    r_size,  # device scalar
    *,
    out_dtype,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    cumsum_scale: float,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    mini_batch: int = 1,
):
    """Gate activation + cumsum + intra-chunk solve for one tile of MB heads.

  Uses Neumann series inversion; bfloat16 only (see the module docstring).

  Only the first ``r_size`` rows hold real tokens. The rest are whatever the
  staging buffer held from the previous tile, so every input is masked to zero
  first -- see the mask comment below for why that is the correct padding value
  and why the gate has to be masked *after* activation.

  Args:
      out_dtype: the caller's activation dtype. The staged inputs arrive in
          float32 (see the staging note in ``pallas_kda_fwd_intra_fused``), so
          the dtype the results belong in has to be passed in rather than read
          off ``q``.
      r_size: rows of this tile that hold real tokens, ``<= chunk_size``.
      mini_batch: number of heads processed per grid point (MB).

  Returns:
      ``(u, w, kg, Aqk, g_cumsum, q_masked)``; the last is ``q`` with its padded
      rows zeroed, which stage 3+4 needs in the same masked form.
  """
    dtype = out_dtype
    BT = chunk_size
    BC = 16
    NC = BT // BC
    K = head_dim
    V = value_dim
    MB = mini_batch

    # Rows past the tile's real length hold whatever the staging buffer had --
    # the previous tile's tokens, or, on the first tile, uninitialised VMEM.
    # Zeroing q/k/v/beta makes them inert: with beta 0 the solve's `L` rows
    # vanish, so `u`/`w` come out zero there, and with k zero those rows add
    # nothing to stage 3's `kg^T @ v_new`.
    #
    # This has to be a select, not a multiply by a 0/1 mask: uninitialised VMEM
    # can hold a NaN bit pattern and `0.0 * NaN` is NaN, which then spreads
    # through the Neumann solve's full-width matmuls into the real rows. Masks
    # are built as 2-D iota comparisons and broadcast on a *leading* axis, which
    # avoids the (BT,) -> (BT, 1) i1 shape casts Mosaic rejects.
    def _row_keep(last_dim):
        rows = jax.lax.broadcasted_iota(jnp.int32, (BT, last_dim), dimension=0)
        return (rows < r_size)[None]  # [1, BT, last_dim]

    keep_k = _row_keep(K)
    keep_v = _row_keep(V)
    keep_1 = _row_keep(1)

    def _mask(x, keep):
        return jnp.where(keep, x, 0.0)

    # --- Gate activation + cumsum ---
    g_f32 = g.astype(jnp.float32)

    if use_gate_in_kernel:
        g_f32 = g_f32 + dt_b[:, None, :]  # [MB, BT, K]
        if lower_bound is None:
            g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
        else:
            g_f32 = lower_bound * jax.nn.sigmoid(
                jnp.exp(A_val)[:, None, None] * g_f32)

    # The gate is masked *after* activation, not before. Both activations map 0
    # to a non-zero decay, so masking the raw gate would leave padded rows
    # decaying; masking the activated value puts `exp2` at 1 there in either
    # mode. That matters beyond the padded rows themselves, because stage 3
    # applies `g_cumsum[BT - 1]` -- the running sum over the *whole* tile -- to
    # the entire recurrent state.
    g_f32 = _mask(g_f32, keep_k)

    tril = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    g_cumsum = jax.lax.dot_general(
        tril,
        g_f32,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2) * cumsum_scale

    q_f32 = _mask(q.astype(jnp.float32), keep_k)
    k_f32 = _mask(k.astype(jnp.float32), keep_k)
    beta_f32 = _mask(beta.astype(jnp.float32), keep_1)
    v_f32 = _mask(v.astype(jnp.float32), keep_v)

    # --- BC=16 sub-block Aqk/L ---
    # The quantity wanted is, for every causal pair (r, t):
    #   Aqk[m,r,t] = sum_k q[m,r,k] * k[m,t,k] * exp2(g_cumsum[m,r,k]
    #                                                 - g_cumsum[m,t,k])
    # KDA's gate is *per channel*, so that decay is a rank-3 tensor over
    # (r, t, k) and the sum over k cannot be folded into a plain matmul. The
    # usual trick is to factor it through a per-sub-block reference row,
    #   exp2(g_r - g_ref) * exp2(g_ref - g_t),
    # which restores the matmul. But those two factors are a tiny*huge pair:
    # their product is exp2(g_r - g_t) <= 1, while individually they saturate
    # float32's exp2 range (~+/-128). Kimi-Linear's gate is unbounded
    # (-exp(A_log) * softplus(.)) and reaches -1736 for a *single* token, so
    # inside one sub-block one factor becomes 0 and the other inf, giving
    # 0 * inf = NaN. Measured on real layer-0 weights: 26 of 32 heads NaN,
    # which reached the sampler as non-finite logits and produced empty output.
    # So the diagonal block is computed *pairwise* instead. For causal pairs
    # r >= t the exponent g_r - g_t is non-positive (g_cumsum is monotonically
    # decreasing, since the activated gate is <= 0), so exp2 cannot overflow and
    # underflow to 0 is the correct answer -- a fully decayed contribution.
    # Columns strictly before the sub-block (t < i_s) keep the fast factored
    # matmul: there both exponents are provably <= 0 as well, because
    # g_cumsum[t] >= gn_ref for t < i_s and g_i[r] <= gn_ref for r >= i_s.
    #
    # This costs the single fused matmul on the diagonal blocks (NC of them, each
    # BC x BC x K) in exchange for being numerically valid at any gate
    # PRECONDITION: the activated gate is <= 0, i.e. `lower_bound` (when given)
    # is negative. `chunk_kda` validates this.
    #
    # Use broadcasted_iota for 2D indices to avoid Mosaic-unsupported i1
    # shape casts (e.g. (BT,) -> (BT, 1) on bool tensors).
    row_iota_bt_k = jax.lax.broadcasted_iota(jnp.int32, (BT, K), dimension=0)
    row_iota_bc = jax.lax.broadcasted_iota(jnp.int32, (BC, BC), dimension=0)
    col_iota_bc = jax.lax.broadcasted_iota(jnp.int32, (BC, BC), dimension=1)
    causal_bc = (row_iota_bc >= col_iota_bc)[None]  # [1, BC, BC]
    strict_bc = (row_iota_bc > col_iota_bc)[None]  # [1, BC, BC]

    # Contract the channel axis: [MB, BC, K] x [MB, K] -> [MB, BC]
    def _mv_k(lhs, rhs):
        return jax.lax.dot_general(
            lhs,
            rhs,
            (((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )

    Aqk_rows = []
    L_rows = []
    for i_sc in range(NC):
        i_s = i_sc * BC
        q_i = q_f32[:, i_s:i_s + BC]  # [MB, BC, K]
        k_i = k_f32[:, i_s:i_s + BC]  # [MB, BC, K]
        g_i = g_cumsum[:, i_s:i_s + BC]  # [MB, BC, K]
        beta_i = beta_f32[:, i_s:i_s + BC]  # [MB, BC, 1]

        # --- columns t < i_s: factored matmul (both exponents <= 0) ---
        gn_ref = g_i[:, 0:1, :]  # [MB, 1, K] — sub-block row 0
        exp_diff_i = jnp.exp2(jnp.minimum(g_i - gn_ref, 0.0))
        q_eg = q_i * exp_diff_i  # [MB, BC, K]
        k_eg = k_i * exp_diff_i  # [MB, BC, K]

        valid_j = (row_iota_bt_k < i_s).astype(jnp.float32)  # [BT, K]
        diff_j = jnp.minimum(gn_ref - g_cumsum,
                             0.0) * valid_j[None]  # [MB, BT, K]
        k_eng_full = k_f32 * jnp.exp2(diff_j) * valid_j[None]  # [MB, BT, K]

        # Stack q_eg / k_eg: [MB, 2*BC, K] x [MB, BT, K] -> [MB, 2*BC, BT]
        qk_eg = jnp.concatenate([q_eg, k_eg], axis=1)  # [MB, 2*BC, K]
        qk_dot = jax.lax.dot_general(
            qk_eg,
            k_eng_full,
            (((2, ), (2, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )  # [MB, 2*BC, BT]
        Aqk_row = qk_dot[:, :BC]  # [MB, BC, BT]
        Akk_row = qk_dot[:, BC:]  # [MB, BC, BT]

        # --- diagonal block t in [i_s, i_s + BC): exact per-channel decay ---
        aqk_diag_rows = []
        akk_diag_rows = []
        for r in range(BC):
            # g_i[r] - g_i[t] <= 0 for t <= r; the clamp is a no-op there and hard
            # bounds the exponent for the masked-away t > r entries.
            kw = k_i * jnp.exp2(jnp.minimum(g_i[:, r:r + 1, :] - g_i, 0.0))
            aqk_diag_rows.append(_mv_k(kw, q_i[:,
                                               r, :])[:,
                                                      None, :])  # [MB, 1, BC]
            akk_diag_rows.append(_mv_k(kw, k_i[:, r, :])[:, None, :])
        Aqk_diag = jnp.concatenate(aqk_diag_rows, axis=1)  # [MB, BC, BC]
        Akk_diag = jnp.concatenate(akk_diag_rows, axis=1)  # [MB, BC, BC]
        Aqk_diag = jnp.where(causal_bc, Aqk_diag, jnp.float32(0.0))
        Akk_diag = jnp.where(strict_bc, Akk_diag, jnp.float32(0.0))

        # Widen the [BC, BC] block to [BC, BT] at column offset i_s. Built by
        # concatenation so every shape stays static; zero-width pads are skipped
        # rather than passed to concatenate.
        def _widen(blk):
            parts = []
            if i_s:
                parts.append(jnp.zeros((MB, BC, i_s), jnp.float32))
            parts.append(blk)
            tail = BT - i_s - BC
            if tail:
                parts.append(jnp.zeros((MB, BC, tail), jnp.float32))
            return parts[0] if len(parts) == 1 else jnp.concatenate(parts,
                                                                    axis=2)

        Aqk_row = (Aqk_row + _widen(Aqk_diag)) * scale
        Akk_row = (Akk_row + _widen(Akk_diag)) * beta_i

        Aqk_rows.append(Aqk_row)
        L_rows.append(Akk_row)

    Aqk = jnp.concatenate(Aqk_rows, axis=1).astype(dtype)  # [MB, BT, BT]
    L = jnp.concatenate(L_rows, axis=1)  # [MB, BT, BT]

    # --- Solve (I + L) x = rhs ---
    v_beta = v_f32 * beta_f32  # [MB, BT, V]
    k_eg_beta = k_f32 * jnp.exp2(g_cumsum) * beta_f32  # [MB, BT, K]
    I_bt = jnp.eye(BT, dtype=jnp.float32)  # [BT, BT]

    # Batched dot helper: [MB, M, K] @ [MB, K, N] → [MB, M, N]
    def _dot_batch(a, b):
        return jax.lax.dot_general(
            a,
            b,
            (((2, ), (1, )), ((0, ), (0, ))),
            preferred_element_type=jnp.float32,
        )

    use_neumann = dtype != jnp.float32

    if use_neumann:
        # --- Neumann series inversion ---
        BC_inv = 8
        NC_inv = BT // BC_inv
        inv_dtype = jnp.float32

        L_inv = L.astype(inv_dtype)  # [MB, BT, BT]

        _idx = jnp.arange(BT, dtype=jnp.int32)
        _block_id = _idx // BC_inv
        _same_block = (_block_id[:,
                                 None] == _block_id[None, :]).astype(inv_dtype)
        L_diag = L_inv * _same_block[None]  # [MB, BT, BT]
        F = L_inv - L_diag  # [MB, BT, BT]

        neg_Ld = -L_diag
        S = I_bt[None] + neg_Ld  # [MB, BT, BT]
        Mk = neg_Ld
        num_diag_steps = {4: 1, 8: 2, 16: 3, 32: 4, 64: 5}[BC_inv]
        for _ in range(num_diag_steps):
            Mk = _dot_batch(Mk, Mk)
            S = _dot_batch(S, I_bt[None] + Mk)
        P = S

        rhs = jnp.concatenate([
            v_beta.astype(inv_dtype),
            k_eg_beta.astype(inv_dtype),
        ],
                              axis=-1)  # [MB, BT, V+K]

        if NC_inv == 1:
            result = _dot_batch(P, rhs)
        else:
            # Fuse `P @ [F | rhs]` to share a matmul; split out G and P_rhs.
            F_and_rhs = jnp.concatenate([F, rhs], axis=-1)  # [MB, BT, BT+V+K]
            P_merged = _dot_batch(P, F_and_rhs)  # [MB, BT, BT+V+K]
            G = P_merged[:, :, :BT]  # [MB, BT, BT]
            P_rhs = P_merged[:, :, BT:]  # [MB, BT, V+K]

            # Compute inv_I_G = (I + G)^{-1} = sum_{k=0}^{NC_inv-1} (-G)^k via
            # Horner doubling. Build the matrix first (small (MB,BT,BT) matmuls),
            # then apply once to P_rhs.
            H_mat = -G
            inv_I_G = I_bt[None] + H_mat  # (I + H)
            Hk = H_mat
            log2_NC_inv = {2: 1, 4: 2, 8: 3, 16: 4, 32: 5}[NC_inv]
            for step in range(log2_NC_inv - 1):
                Hk = _dot_batch(Hk, Hk)  # H^(2^(step+1))
                inv_I_G = inv_I_G + _dot_batch(inv_I_G,
                                               Hk)  # inv_I_G @ (I + Hk)

            result = _dot_batch(inv_I_G, P_rhs)  # [MB, BT, V+K]
    else:
        # The float32 branch used exact forward substitution instead of the Neumann
        # series. Removed with the rest of the fp32 path: Kimi-K3 runs bfloat16, so
        # `use_neumann` is always True here. See the module docstring.
        raise NotImplementedError(
            "float32 inputs are not supported; this kernel is scoped to Kimi-K3's "
            "bfloat16 path (use_neumann). Cast q/k/v/g to bfloat16.")

    u = result[:, :, :V]  # [MB, BT, V]
    w = result[:, :, V:V + K]  # [MB, BT, K]

    # --- kg ---
    g_last = g_cumsum[:, BT - 1:BT, :]  # [MB, 1, K]
    kg = k_f32 * exp2(g_last - g_cumsum)

    return u, w, kg, Aqk, g_cumsum, q_f32.astype(dtype)


def _fused_gate_intra_kernel(
    # Scalar prefetch: the tile plan.
    num_tiles_ref,  # [B]
    r_base_ref,  # [B, NT]
    r_size_ref,  # [B, NT]
    # Inputs, as whole-array HBM refs over the *unaligned* token axis.
    q_hbm,  # [H, B, T, K]
    k_hbm,  # [H, B, T, K]
    g_hbm,  # [H, B, T, K]
    beta_hbm,  # [H, B, T, 1]
    v_hbm,  # [H, B, T, V]
    # Per-head gate parameters.
    A_log_ref,  # [MB, 1, 1, 1, 1]
    dt_bias_ref,  # [MB, 1, 1, 1, K]
    # Outputs, in tile layout.
    u_out_ref,  # [MB, 1, 1, BT, V]
    w_out_ref,  # [MB, 1, 1, BT, K]
    kg_out_ref,  # [MB, 1, 1, BT, K]
    Aqk_out_ref,  # [MB, 1, 1, BT, BT]
    g_cumsum_out_ref,  # [MB, 1, 1, BT, K]
    q_out_ref,  # [MB, 1, 1, BT, K]
    # Scratch: one float32 staging buffer per input, plus a DMA semaphore.
    q_buf,  # [MB, 1, BT, K]
    k_buf,  # [MB, 1, BT, K]
    g_buf,  # [MB, 1, BT, K]
    beta_buf,  # [MB, 1, BT, 1]
    v_buf,  # [MB, 1, BT, V]
    sem,
    *,
    chunk_size: int,
    mini_batch: int,
    **tile_kwargs,
):
    """Stage the tile's rows out of HBM, then run the intra-chunk solve.

  The grid is (head block, batch, tile) and the tile axis is static at
  ``max_tiles``; ``num_tiles`` bounds what the batch actually needs. A tile
  starts at an arbitrary token row, which no ``BlockSpec`` index map can
  address, so the activations arrive as whole-array HBM refs and this kernel
  issues its own DMAs at ``r_base``.
  """
    head_block = pl.program_id(0) * mini_batch
    i_b = pl.program_id(1)
    p_id = pl.program_id(2)
    num_tiles = num_tiles_ref[i_b]

    # Tiles past the batch's real content are skipped, which leaves their slice
    # of every output holding whatever the pipeline last had. Stage 3+4 skips the
    # same tiles, so nothing reads it -- and even if it did not, those tiles
    # carry `r_size` 0 and neither first- nor last-tile flag, so they would
    # write no output rows and touch no state. A stage that starts *depending* on
    # these values, rather than merely reading them, has to zero them here.
    @pl.when(p_id < num_tiles)
    def _():
        r_base = r_base_ref[i_b, p_id]
        r_size = r_size_ref[i_b, p_id]
        # Exactly the rows the tile owns. Rows past `r_size` keep whatever the
        # previous tile left in the buffer, which the solve masks off.
        copies = [
            pltpu.make_async_copy(
                src.at[pl.ds(head_block, mini_batch),
                       pl.ds(i_b, 1),
                       pl.ds(r_base, r_size)],
                dst.at[:, :, pl.ds(0, r_size)],
                sem,
            ) for src, dst in (
                (q_hbm, q_buf),
                (k_hbm, k_buf),
                (g_hbm, g_buf),
                (beta_hbm, beta_buf),
                (v_hbm, v_buf),
            )
        ]
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()

        u, w, kg, Aqk, g_cumsum, q_masked = _fused_gate_intra_tile(
            q_buf[:, 0],
            k_buf[:, 0],
            g_buf[:, 0],
            beta_buf[:, 0],
            v_buf[:, 0],
            A_log_ref[:, 0, 0, 0, 0],
            dt_bias_ref[:, 0, 0, 0],
            r_size,
            chunk_size=chunk_size,
            mini_batch=mini_batch,
            **tile_kwargs,
        )
        u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
        w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
        kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
        Aqk_out_ref[:, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
        g_cumsum_out_ref[:, 0, 0] = g_cumsum
        q_out_ref[:, 0, 0] = q_masked.astype(q_out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=[
        "chunk_size",
        "scale",
        "cumsum_scale",
        "use_gate_in_kernel",
        "lower_bound",
        "mini_batch",
    ],
)
def pallas_kda_fwd_intra_fused(
    q: jax.Array,  # [H, B, T, K]
    k: jax.Array,  # [H, B, T, K]
    v: jax.Array,  # [H, B, T, V]
    g: jax.Array,  # [H, B, T, K]
    beta: jax.Array,  # [H, B, T]
    num_tiles: jax.Array,  # [B]
    r_base: jax.Array,  # [B, NT]
    r_size: jax.Array,  # [B, NT]
    scale: float,
    chunk_size: int = 64,
    cumsum_scale: float = RCP_LN2,
    A_log: jax.Array | None = None,  # [H]
    dt_bias: jax.Array | None = None,  # [H*K]
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    mini_batch: int | None = None,
) -> tuple[
        jax.Array,  # w   [H, B, NT, BT, K]
        jax.Array,  # u   [H, B, NT, BT, V]
        jax.Array,  # kg  [H, B, NT, BT, K]
        jax.Array,  # Aqk [H, B, NT, BT, BT]
        jax.Array,  # g_cumsum [H, B, NT, BT, K]
        jax.Array,  # q   [H, B, NT, BT, K]
]:
    """Fuse gate cumsum with the intra-chunk solve, one tile per grid point.

    Takes the activations on the *unaligned* token axis and returns the derived
    per-chunk quantities in tile layout, indexed ``[H, B, tile, row, channel]``.
    That layout is what stage 3+4 wants anyway -- a tile index is a block index,
    so it needs no gather -- and it is where the padding gather used to go.

    `q` is returned alongside because stage 3+4 needs it with the same rows
    masked; re-deriving it there would mean duplicating the staging DMAs.
    """
    H, B, _, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    NT = r_base.shape[-1]
    out_dtype = q.dtype

    if use_gate_in_kernel:
        assert A_log is not None, "A_log required when use_gate_in_kernel=True"

    if mini_batch is None:
        elem = out_dtype.itemsize
        # Staged inputs (float32: q/k/g at K, v at V, beta lane-padded), the
        # pipelined outputs, and the solve's [BT, BT] float32 temporaries.
        # Under-counting here over-subscribes VMEM silently.
        num_lanes = pltpu.get_tpu_info().num_lanes
        staged = (3 * BT * K + BT * V + BT * num_lanes) * 4
        produced = (3 * BT * K + BT * V + BT * BT) * elem + BT * K * 4
        per_head = staged + 2 * produced + 8 * BT * BT * 4
        MB = estimate_mini_batch(per_head, H, max_mb=16)
    else:
        MB = mini_batch
        assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

    # A per-token scalar on a lane-minor axis cannot be DMA'd at an arbitrary
    # row offset, so beta is carried with the token axis second-minor like the
    # rest. It costs lane padding on a tensor 1/128th the size of the others.
    beta_4d = beta[..., None]

    # The staged tensors are float32, not the caller's bfloat16, because a tile
    # starts at an arbitrary token row and Mosaic will only slice a *packed*
    # memref at tile-aligned offsets and in tile-aligned sizes -- for bfloat16
    # that means multiples of 8, and neither `query_start_loc[s] + t *
    # chunk_size` nor a ragged tile's row count is. Unpacked 32-bit types have no
    # such rule. The kernel casts every input to float32 immediately anyway, so
    # this only moves that cast out of the tile loop; it costs one contiguous
    # pass over the activations -- not the indexed gather over
    # T + N * (chunk_size - 1) rows the alignment needed -- and 2x the DMA bytes.
    q_s, k_s, g_s, v_s, beta_s = (x.astype(jnp.float32)
                                  for x in (q, k, g, v, beta_4d))

    if use_gate_in_kernel:
        A_log_r = A_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
        if dt_bias is not None:
            dt_bias_r = dt_bias.astype(jnp.float32).reshape(H, 1, 1, 1, K)
        else:
            dt_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
    else:
        A_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
        dt_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)

    grid = (H // MB, B, NT)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # Scalar prefetch adds three leading arguments to every index map.
    def _out_spec(last_dim):
        return pl.BlockSpec(
            index_map=lambda i, j, p, *_: (i, j, p, 0, 0),
            block_shape=(MB, 1, 1, BT, last_dim),
        )

    def _per_head_spec(last_dim):
        return pl.BlockSpec(
            index_map=lambda i, j, p, *_: (i, 0, 0, 0, 0),
            block_shape=(MB, 1, 1, 1, last_dim),
        )

    def _tile_shape(last_dim, dtype):
        return jax.ShapeDtypeStruct((H, B, NT, BT, last_dim), dtype)

    u_r, w_r, kg_r, Aqk_r, g_cumsum_r, q_r = pl.pallas_call(
        functools.partial(
            _fused_gate_intra_kernel,
            out_dtype=out_dtype,
            chunk_size=BT,
            head_dim=K,
            value_dim=V,
            scale=scale,
            cumsum_scale=cumsum_scale,
            use_gate_in_kernel=use_gate_in_kernel,
            lower_bound=lower_bound,
            mini_batch=MB,
        ),
        interpret=get_interpret(),
        out_shape=[
            _tile_shape(V, out_dtype),
            _tile_shape(K, out_dtype),
            _tile_shape(K, out_dtype),
            _tile_shape(BT, out_dtype),
            _tile_shape(K, jnp.float32),
            _tile_shape(K, out_dtype),
        ],
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=grid,
            in_specs=[
                hbm_spec,  # q
                hbm_spec,  # k
                hbm_spec,  # g
                hbm_spec,  # beta
                hbm_spec,  # v
                _per_head_spec(1),  # A_log
                _per_head_spec(K),  # dt_bias
            ],
            out_specs=[
                _out_spec(V),
                _out_spec(K),
                _out_spec(K),
                _out_spec(BT),
                _out_spec(K),
                _out_spec(K),
            ],
            scratch_shapes=[
                pltpu.VMEM((MB, 1, BT, K), jnp.float32),
                pltpu.VMEM((MB, 1, BT, K), jnp.float32),
                pltpu.VMEM((MB, 1, BT, K), jnp.float32),
                pltpu.VMEM((MB, 1, BT, 1), jnp.float32),
                pltpu.VMEM((MB, 1, BT, V), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            # The tile axis carries the staging DMAs, so it stays sequential.
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
    )(num_tiles, r_base, r_size, q_s, k_s, g_s, beta_s, v_s, A_log_r,
      dt_bias_r)

    return w_r, u_r, kg_r, Aqk_r, g_cumsum_r, q_r


# =============================================================================
# Fused state propagation and output
# =============================================================================


def _chunk_kda_fwd_h_o_varlen_kernel(
    # Scalar prefetch: the tile plan.
    num_tiles_ref,  # [B]
    r_base_ref,  # [B, NT]
    r_size_ref,  # [B, NT]
    s_idx_ref,  # [B, NT]
    is_first_ref,  # [B, NT]
    is_last_ref,  # [B, NT]
    # Stage 3 inputs
    w_ref,  # [MB, 1, 1, BT, K_PADSIZE]
    u_ref,  # [MB, 1, 1, BT, V_ALIGNED]
    kg_ref,  # [MB, 1, 1, BT, K_PADSIZE]
    gk_ref,  # [MB, 1, 1, BT, K_PADSIZE]  -- g_cumsum
    # Stage 4 inputs
    q_ref,  # [MB, 1, 1, BT, K_PADSIZE]
    A_ref,  # [MB, 1, 1, BT, BT]
    _o_in,  # the zero-filled output, aliased; written through o_ref
    # Optional initial state
    h0_ref,  # [1, 1, MB, K_PADSIZE, V_ALIGNED] or None
    # Outputs
    o_ref,  # [H, B, T, V_ALIGNED] whole-array HBM ref
    ht_ref,  # [1, 1, MB, K_PADSIZE, V_ALIGNED] or None
    # Scratch
    scratch_ref,  # [MB, K_PADSIZE, V_ALIGNED]
    o_buf,  # [MB, 1, BT, V_ALIGNED]
    sem,
    *,
    BT,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    MB,
    OUTPUT_PRECISION,
):
    """Fused Stage 3+4 Pallas kernel body, one tile per grid point.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, p_id):
    - MB heads [h_group*MB .. h_group*MB+MB) are processed per grid point
      via batched matmuls over the MB dimension.
    - p_id indexes the tile axis, static at NT; `num_tiles` bounds what the
      batch actually needs, and tiles past it are skipped outright.
    - A segment's tiles are consecutive p_ids, so `is_first` / `is_last` mark
      where its recurrent state is read in and written back.

  The per-chunk inputs arrive already in tile layout, which is BlockSpec-
  addressable. Only the output is not: a tile's rows land at an arbitrary
  offset on the token axis, so `o_ref` is a whole-array HBM ref this kernel
  DMAs into. Rows no tile covers keep the zeros the caller passed in.
  """
    del _o_in
    head_block = pl.program_id(0) * MB
    i_b = pl.program_id(1)
    p_id = pl.program_id(2)

    K = w_ref.shape[-1]
    V = u_ref.shape[-1]

    def _tile():
        # === Init state (first tile of THIS segment) — uniform across MB heads
        @pl.when(is_first_ref[i_b, p_id])
        def _():
            scratch_ref[:] = jnp.zeros([MB, K, V], dtype=jnp.float32)
            if USE_INITIAL_STATE:
                scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]

        # === Stage 3+4 work — batched over MB heads ===
        # h: pre-update state for all MB heads in this tile.
        b_h = scratch_ref[:]  # [MB, K, V]

        b_w = w_ref[:, 0, 0]  # [MB, BT, K]
        b_u = u_ref[:, 0, 0]  # [MB, BT, V]

        # Stage 3 delta correction: v_new = u - w @ h
        # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
        # HIGHEST precision: v_new feeds the recursive state update directly.
        b_v_new = b_u.astype(jnp.float32) - jnp.matmul(
            b_w.astype(jnp.float32),
            b_h,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )  # [MB, BT, V]

        # Stage 4 inter-chunk output:
        #   scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref))
        # Reference-point stabilization mirrors the production varlen kernel's
        # g_ref = g[0] choice for bit-identical numerics.
        b_q = q_ref[:, 0, 0]  # [MB, BT, K]
        b_g = gk_ref[:, 0, 0].astype(jnp.float32)  # [MB, BT, K]
        b_A = A_ref[:, 0, 0]  # [MB, BT, BT]

        b_g_ref_row = b_g[:, 0:1, :]  # [MB, 1, K]
        b_qg = b_q.astype(jnp.float32) * jnp.exp2(
            jnp.maximum(b_g - b_g_ref_row, -126.0))  # [MB, BT, K]
        b_h_scaled = b_h * jnp.exp2(jnp.maximum(
            b_g_ref_row[:, 0, :], -126.0))[:, :, None]  # [MB, K, V]

        # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
        # Output-only GEMM: bf16 inputs use DEFAULT (no extra precision to
        # preserve), fp32 inputs HIGHEST for full mantissa fidelity.
        b_o = jnp.matmul(
            b_qg,
            b_h_scaled,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        ) * scale  # [MB, BT, V]

        # Stage 4 intra-chunk: A @ v_new
        # Apply lower-triangular mask: Aqk is causal by construction, but masking
        # here matches the original per-head loop behaviour and guards against
        # any tiny upper-triangle fp noise from the Neumann intra-chunk solve.
        m_s = jnp.arange(BT)[:, None] >= jnp.arange(BT)[None, :]  # [BT, BT]
        b_A_f32 = jnp.where(m_s[None, :, :], b_A.astype(jnp.float32), 0.0)
        # [MB, BT, BT] @ [MB, BT, V] -> [MB, BT, V]
        b_o = b_o + jnp.matmul(
            b_A_f32,
            b_v_new,
            precision=OUTPUT_PRECISION,
            preferred_element_type=jnp.float32,
        )  # [MB, BT, V]

        # Scatter this tile's rows to where its tokens live. `r_size` rows, so
        # rows the tile does not own keep the caller's zeros -- and a dynamically
        # sized slice is legal here only because the output is float32 (see the
        # staging note in `pallas_kda_fwd_intra_fused`).
        r_base = r_base_ref[i_b, p_id]
        r_size = r_size_ref[i_b, p_id]
        o_buf[:, 0] = b_o.astype(o_buf.dtype)
        copy = pltpu.make_async_copy(
            o_buf.at[:, :, pl.ds(0, r_size)],
            o_ref.at[pl.ds(head_block, MB),
                     pl.ds(i_b, 1),
                     pl.ds(r_base, r_size)],
            sem,
        )
        copy.start()
        copy.wait()

        # Stage 3 state update: h = decay(h) + kg^T @ v_new
        # HIGHEST precision: accumulates into the recursive hidden state.
        b_gk_last = gk_ref[:, 0, 0][:,
                                    BT - 1, :].astype(jnp.float32)  # [MB, K]
        b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None]  # [MB, K, V] decay

        b_kg = kg_ref[:, 0, 0]  # [MB, BT, K]
        # [MB, K, BT] @ [MB, BT, V] -> [MB, K, V]
        b_h_new = b_h_new + jnp.matmul(
            b_kg.astype(jnp.float32).transpose(0, 2, 1),
            b_v_new,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=jnp.float32,
        )
        scratch_ref[:] = b_h_new

        # === Final state (last tile of THIS segment) ===
        @pl.when(is_last_ref[i_b, p_id])
        def _():
            if STORE_FINAL_STATE:
                ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)

    # Tiles past the batch's real content own no output rows (`r_size` 0) and
    # carry no segment's state (neither first- nor last-tile flag), so skipping
    # them is an optimization rather than a correctness requirement -- running
    # them would only scribble on the state scratch, which the next segment's
    # first tile reinitialises. What *is* load-bearing is that `s_idx` names the
    # final segment for them: that keeps the state window from changing blocks,
    # and so from flushing an uninitialised buffer over a live slot.
    pl.when(p_id < num_tiles_ref[i_b])(_tile)


@functools.partial(
    jax.jit,
    static_argnames=[
        "num_tokens",
        "output_final_state",
        "scale",
        "chunk_size",
        "mini_batch",
    ],
)
def chunk_kda_fwd_h_o_varlen(
    w: jax.Array,  # [H, B, NT, BT, K]
    u: jax.Array,  # [H, B, NT, BT, V]
    kg: jax.Array,  # [H, B, NT, BT, K]
    gk: jax.Array,  # [H, B, NT, BT, K]
    q: jax.Array,  # [H, B, NT, BT, K]
    A: jax.Array,  # [H, B, NT, BT, BT]
    plan: _TilePlan,
    query_lens: jax.Array,  # [B, N]
    num_tokens: int,
    initial_state: jax.Array | None = None,  # [B, N, H, K, V]
    output_final_state: bool = False,
    scale: float = 1.0,
    chunk_size: int = 64,
    mini_batch: int | None = None,
) -> tuple[
        jax.Array,  # [H, B, T, V]
        jax.Array | None,  # [B, N, H, K, V]
]:
    """Fuse variable-length state propagation with output projection.

    Takes the per-chunk quantities in tile layout (what
    ``pallas_kda_fwd_intra_fused`` returns) and scatters the output onto the
    caller's ``num_tokens``-row token axis.
    """
    H, B, NT, BT_in, K = q.shape
    V = u.shape[-1]
    BT = chunk_size
    N = query_lens.shape[-1]

    assert BT_in == BT, f"tile rows {BT_in} must equal chunk_size={BT}"
    assert A.shape[-1] == BT, (
        f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}")
    if initial_state is not None:
        assert initial_state.shape[1] == N, (
            f"initial_state has N={initial_state.shape[1]}, expected {N}")
    assert K <= 256, "current kernel does not support K > 256."

    align_major = pltpu.get_tpu_info().num_lanes
    K_PADSIZE = int(align_up(K, align_major))
    V_ALIGNED = int(align_up(V, align_major))

    # ---- auto-compute mini-batch (MB) to maximise VMEM utilisation ----
    if mini_batch is None:
        elem_size = q.dtype.itemsize
        # scratch per head: h_state[K_PADSIZE*V_ALIGNED] in f32 + i/o buffers
        in_bytes = (BT * K_PADSIZE + BT * V_ALIGNED + BT) * elem_size
        out_bytes = BT * V_ALIGNED * 4  # float32 staging for the output DMA
        scratch_bytes = K_PADSIZE * V_ALIGNED * 4  # float32 accumulator
        per_head = in_bytes + out_bytes + scratch_bytes
        MB = estimate_mini_batch(per_head, H, max_mb=16)
    else:
        MB = mini_batch
        assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

    # Pad the channel axis to the lane count where needed. A no-op at the
    # production head_dim of 128; the tile axis needs no padding at all, since
    # every tile is exactly BT rows by construction.
    def _pad_channels(x, dim_pad):
        if dim_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (0, dim_pad)))
        return x

    w_t = _pad_channels(w, K_PADSIZE - K)
    kg_t = _pad_channels(kg, K_PADSIZE - K)
    gk_t = _pad_channels(gk, K_PADSIZE - K)
    q_t = _pad_channels(q, K_PADSIZE - K)
    u_t = _pad_channels(u, V_ALIGNED - V)
    A_t = A

    # h0: [B, N, H, K, V] -> pad K and V -> [B, N, H, K_PADSIZE, V_ALIGNED]
    if initial_state is not None:
        h0 = initial_state
        if V_ALIGNED > V:
            h0 = jnp.pad(h0,
                         ((0, 0), (0, 0), (0, 0), (0, 0), (0, V_ALIGNED - V)))
        if K_PADSIZE > K:
            h0 = jnp.pad(h0,
                         ((0, 0), (0, 0), (0, 0), (0, K_PADSIZE - K), (0, 0)))
    else:
        h0 = None

    # Six scalar-prefetch arrays, so every index map takes six trailing refs.
    # The tile-layout inputs index by grid position alone; the state windows
    # index by the tile's segment.
    def _tile_index_map(h, b, p, *_):
        return (h, b, p, 0, 0)

    def _state_index_map(h, b, p, _num_tiles, _r_base, _r_size, s_idx, *_):
        return (b, s_idx[b, p], h, 0, 0)

    bspec_k = pl.BlockSpec([MB, 1, 1, BT, K_PADSIZE],
                           index_map=_tile_index_map)
    bspec_v = pl.BlockSpec([MB, 1, 1, BT, V_ALIGNED],
                           index_map=_tile_index_map)
    bspec_a = pl.BlockSpec([MB, 1, 1, BT, BT], index_map=_tile_index_map)
    bspec_state = pl.BlockSpec([1, 1, MB, K_PADSIZE, V_ALIGNED],
                               index_map=_state_index_map)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    # The output is written by DMA at arbitrary token rows, so it is a
    # whole-array HBM ref rather than a pipelined window, and it is passed in
    # zeroed and aliased: rows that no tile owns -- the token axis's padded tail
    # -- are never written, and must still come back as zeros.
    o_shape = jax.ShapeDtypeStruct([H, B, num_tokens, V_ALIGNED], jnp.float32)
    o_zeros = jnp.zeros(o_shape.shape, o_shape.dtype)
    ht_shape = (jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED],
                                     jnp.float32)
                if output_final_state else None)
    grid = (H // MB, B, NT)

    # bf16 inputs: DEFAULT precision is lossless (operands already have ~7-bit
    # mantissa); fp32 inputs: HIGHEST preserves full 23-bit mantissa fidelity.
    # State-update matmuls always use HIGHEST regardless (recursive accumulation).
    _output_prec = (jax.lax.Precision.DEFAULT
                    if q.dtype == jnp.bfloat16 else jax.lax.Precision.HIGHEST)

    o_out, ht_out = pl.pallas_call(
        functools.partial(
            _chunk_kda_fwd_h_o_varlen_kernel,
            BT=BT,
            scale=scale,
            USE_INITIAL_STATE=(h0 is not None),
            STORE_FINAL_STATE=output_final_state,
            MB=MB,
            OUTPUT_PRECISION=_output_prec,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=grid,
            in_specs=[
                bspec_k,  # w
                bspec_v,  # u
                bspec_k,  # kg
                bspec_k,  # gk
                bspec_k,  # q
                bspec_a,  # A
                hbm_spec,  # the zeroed output, aliased
                bspec_state if h0 is not None else None,  # h0
            ],
            out_specs=[
                hbm_spec,
                bspec_state if output_final_state else None,
            ],
            scratch_shapes=[
                pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32),
                pltpu.VMEM((MB, 1, BT, V_ALIGNED), jnp.float32),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        out_shape=[o_shape, ht_shape],
        # The zeroed output rides input slot 6; `h0` sits after it so that
        # index holds whether or not there is an initial state (a None operand
        # contributes no leaf and would otherwise shift it).
        input_output_aliases={6 + 6: 0},
        interpret=get_interpret(),
    )(plan.num_tiles, plan.r_base, plan.r_size, plan.s_idx, plan.is_first,
      plan.is_last, w_t, u_t, kg_t, gk_t, q_t, A_t, o_zeros, h0)

    # Post-process: o is [H, B, num_tokens, V_ALIGNED]
    if V_ALIGNED > V:
        o_out = o_out[..., :V]
    o_out = o_out.astype(u.dtype)

    if output_final_state and ht_out is not None:
        if V_ALIGNED > V:
            ht_out = ht_out[..., :V]
        if K_PADSIZE > K:
            ht_out = ht_out[..., :K, :]

        # Handle empty sequences: sequences with no tiles never execute kernel
        # code, so their final_state is uninitialized -- and the state window of
        # a segment past the last one with tiles may be flushed with a stale
        # buffer. Fill both from initial_state, or zeros.
        empty_mask = (query_lens == 0)  # [B, N]
        if initial_state is not None:
            # For empty sequences, final_state should equal initial_state
            fill_value = initial_state[:, :, :, :K, :V]
        else:
            # For empty sequences without initial_state, final_state should be zeros
            fill_value = jnp.zeros((B, N, H, K, V), dtype=ht_out.dtype)
        # Use where to selectively replace empty sequence states
        ht_out = jnp.where(empty_mask[:, :, None, None, None], fill_value,
                           ht_out)
    else:
        ht_out = None

    return o_out, ht_out


def _run_chunk_kda(
    q: jax.Array,  # [H, B, T, K]
    k: jax.Array,  # [H, B, T, K]
    v: jax.Array,  # [H, B, T, V]
    g: jax.Array,  # [H, B, T, K]
    beta: jax.Array,  # [H, B, T]
    *,
    num_tokens: int,
    cu_seqlens: jax.Array,  # [B, N_CU]
    plan: _TilePlan,
    A_log: jax.Array | None = None,  # [H]
    dt_bias: jax.Array | None = None,  # [H*K]
    scale: float | None = None,
    initial_state: jax.Array | None = None,  # [B, N, H, K, V]
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    chunk_size: int = 64,
) -> tuple[
        jax.Array,  # [H, B, T, V]
        jax.Array | None,  # [B, N, H, K, V]
]:
    """Run the kernel on canonicalized inputs and the batch's tile plan."""
    BT = chunk_size

    # ------------------------------------------------------------------
    # Step 1 + 2 (Fused): Gate cumsum + Intra-chunk solve
    # ------------------------------------------------------------------
    w, u, kg, Aqk, g_cumsum, q_tiled = pallas_kda_fwd_intra_fused(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        num_tiles=plan.num_tiles,
        r_base=plan.r_base,
        r_size=plan.r_size,
        scale=scale,
        chunk_size=BT,
        cumsum_scale=RCP_LN2,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )

    # ------------------------------------------------------------------
    # Step 3 + Step 4: Inter-chunk state + Output
    #
    # Stage 3+4 walks the same tiles, reading the per-chunk quantities straight
    # out of tile layout and scattering the output back to the token rows each
    # tile owns. Nothing is gathered in either direction.
    #
    # Padding semantics (relied on for correctness), produced by the tile
    # kernel's row masking rather than by a zero-filled gather:
    #   - k/w/u/q/Aqk at padded rows = 0. With those zero the state update
    #     reduces to h_pad = h_{t-1} * exp(g_pad), and the output to 0.
    #   - g_cumsum at padded rows = 0 => exp(g_pad) = 1, so the hidden state is
    #     carried through padded rows unchanged.
    # ------------------------------------------------------------------
    output, final_state = chunk_kda_fwd_h_o_varlen(
        w=w,
        u=u,
        kg=kg,
        gk=g_cumsum,
        q=q_tiled,
        A=Aqk,
        plan=plan,
        query_lens=jnp.diff(cu_seqlens, axis=-1),
        num_tokens=num_tokens,
        initial_state=initial_state,
        output_final_state=output_final_state,
        scale=scale,
        chunk_size=BT,
    )
    return output.astype(q.dtype), final_state


# ---------------------------------------------------------------------------
# Input canonicalization
# ---------------------------------------------------------------------------
class _PreparedKdaInputs(NamedTuple):
    q: jax.Array
    k: jax.Array
    v: jax.Array
    g: jax.Array
    beta: jax.Array
    initial_state: jax.Array | None
    cu_seqlens: jax.Array
    plan: _TilePlan


def check_inputs_support(
    q: jax.Array,
    v: jax.Array,
    *,
    chunk_size: int,
) -> None:
    """Checks whether the Pallas TPU backend supports the static inputs."""
    if q.dtype != jnp.bfloat16:
        raise NotImplementedError(
            "The KDA Pallas kernel supports bfloat16 inputs only.")
    heads, batch, seq_len, key_dim = q.shape
    value_dim = v.shape[-1]
    if heads < 1 or batch < 1 or seq_len < 1:
        raise NotImplementedError(
            "`pallas_tpu` requires positive head, batch, and sequence "
            f"dimensions; got H={heads}, B={batch}, T={seq_len}.")
    if key_dim < 1 or value_dim < 1:
        raise NotImplementedError(
            "`pallas_tpu` requires positive key and value dimensions; got "
            f"K={key_dim}, V={value_dim}.")
    if key_dim > 256:
        raise NotImplementedError(
            "`pallas_tpu` currently supports key dimensions up to 256; got "
            f"K={key_dim}.")
    if chunk_size != 64:
        raise NotImplementedError(
            "`pallas_tpu` currently supports chunk_size=64.")


def _preprocess_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    initial_state: jax.Array | None,
    use_qk_l2norm_in_kernel: bool,
    segment_ids: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
    start_seq: jax.Array | int | None,
) -> _PreparedKdaInputs:
    """Canonicalize inputs for the forward kernel and plan its tiles."""
    if N_max is None:
        if initial_state is None:
            raise ValueError(
                "`N_max` is required when `initial_state` is not given: the "
                "segment count is a static shape and cannot be inferred.")
        N_max = initial_state.shape[1]
    cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=N_max)
    if start_seq is not None:
        # Segments before `start_seq` keep their slot in `initial_state` and in
        # the returned `final_state`, but contribute no tiles, so the kernel
        # neither reads nor advances their recurrent state.
        cu_seqlens = restrict_cu_seqlens(cu_seqlens, start_seq)
    plan = _plan_tiles(cu_seqlens,
                       num_tokens=q.shape[2],
                       chunk_size=chunk_size)

    if initial_state is not None and initial_state.shape[1] != N_max:
        raise ValueError(
            f"`initial_state` state count must match segment count {N_max}; "
            f"got {initial_state.shape[1]}.")

    if use_qk_l2norm_in_kernel:
        q_f32 = q.astype(jnp.float32)
        k_f32 = k.astype(jnp.float32)
        q_prepared = (q_f32 * jax.lax.rsqrt(
            jnp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-6)).astype(
                q.dtype)
        k_prepared = (k_f32 * jax.lax.rsqrt(
            jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + 1e-6)).astype(
                k.dtype)
    else:
        q_prepared, k_prepared = q, k

    return _PreparedKdaInputs(
        q=q_prepared,
        k=k_prepared,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        plan=plan,
    )


def chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    *,
    A_log=None,
    dt_bias=None,
    scale: float | None = None,
    initial_state=None,
    output_final_state: bool = False,
    use_gate_in_kernel: bool = False,
    segment_ids=None,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = False,
    N_max: int | None = None,
    start_seq=None,
):
    """Run the KDA forward kernel and return ``(output, final_state)``.

    Args:
      q, k: ``[H, B, T, K]`` queries and keys.
      v: ``[H, B, T, V]`` values.
      g: ``[H, B, T, K]`` per-channel gate. In log space unless
        ``use_gate_in_kernel`` is set, in which case it is raw and is activated
        in-kernel using ``A_log`` and ``dt_bias``.
      beta: ``[H, B, T]`` per-token delta-rule learning rate, already through
        the sigmoid.
      A_log: ``[H]`` gate parameter; required when ``use_gate_in_kernel``.
      dt_bias: ``[H*K]`` optional gate bias.
      scale: query scale, defaults to ``K ** -0.5``.
      initial_state: ``[B, N, H, K, V]`` recurrent state, one per request slot.
      output_final_state: whether to return the final state (else ``None``).
      use_gate_in_kernel: treat ``g`` as raw gate input.
      segment_ids: ``[B, T]`` 1-indexed varlen segment IDs, 0 for padding.
        Required -- every batch is ragged. A fixed-length batch is just one
        segment spanning the token axis.
      safe_gate: deprecated compatibility argument; ignored.
      lower_bound: optional sigmoid-gate lower bound.
      chunk_size: chunk length used by the kernel.
      use_qk_l2norm_in_kernel: normalize queries and keys before the kernel.
      N_max: static segment count. Defaults to ``initial_state``'s.
      start_seq: optional device scalar; restrict the kernel to segments at or
        after this index. Earlier segments get no tiles, so their rows of
        ``final_state`` come back exactly as they went in and their output rows
        are left zero -- which is how a caller hands the leading part of a batch
        to a different kernel without a host-side branch on batch composition.

    Returns:
      ``(output, final_state)`` where output is ``[H, B, T, V]`` and
      ``final_state`` is ``[B, N, H, K, V]`` or ``None``.

    """
    del safe_gate
    if scale is None:
        scale = q.shape[-1]**-0.5

    # Cross-argument shape agreement. jaxtyping used to enforce this via shared
    # dimension names ("H B T K" across q/k/g), but the annotations are now plain
    # comments per the repo convention, so check it explicitly here --
    # check_inputs_support only validates q and v.
    if not (q.shape == k.shape == g.shape):
        raise ValueError(
            f"q, k and g must share [H, B, T, K]; got q={q.shape} "
            f"k={k.shape} g={g.shape}")
    if v.shape[:3] != q.shape[:3]:
        raise ValueError(
            f"v must share [H, B, T] with q; got v={v.shape} q={q.shape}")
    if beta.shape != q.shape[:3]:
        raise ValueError(
            f"beta must be [H, B, T] = {q.shape[:3]}; got {beta.shape}")

    # Ragged batches only. The kernel used to carry a second, fixed-length mode
    # that skipped the alignment gather and took one state per batch item; no
    # caller used it, and keeping both meant every downstream branch had to
    # handle two state ranks and two sets of segment boundaries.
    if segment_ids is None:
        raise ValueError(
            "`segment_ids` is required: a fixed-length batch is expressed as a "
            "single segment spanning the token axis.")
    if initial_state is not None and initial_state.ndim != 5:
        raise ValueError(
            "`initial_state` must be [B, N, H, K, V], one recurrent state per "
            f"request slot; got {initial_state.ndim} dimensions.")

    # The kernel's intra-chunk decay relies on the activated gate being <= 0, so
    # that the cumulative gate decreases along the chunk and every causal
    # exponent g_r - g_t is non-positive (see the Aqk/L section of
    # pallas_tpu_fwd.py). Both activations satisfy this only for a negative
    # bound: -exp(A_log) * softplus(.) is always <= 0, while
    # lower_bound * sigmoid(.) needs lower_bound < 0. A positive bound would
    # make the decay grow along the chunk and silently return wrong values.
    if lower_bound is not None and lower_bound >= 0:
        raise ValueError(
            "lower_bound must be negative (it bounds a log-decay); got "
            f"{lower_bound}. A non-negative bound makes the gate increase "
            "along the chunk, which this kernel does not support.")

    check_inputs_support(q, v, chunk_size=chunk_size)

    prepared = _preprocess_inputs(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        segment_ids=segment_ids,
        chunk_size=chunk_size,
        N_max=N_max,
        start_seq=start_seq,
    )

    value, final_state = _run_chunk_kda(
        prepared.q,
        prepared.k,
        prepared.v,
        prepared.g,
        prepared.beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=prepared.initial_state,
        output_final_state=output_final_state,
        use_gate_in_kernel=use_gate_in_kernel,
        num_tokens=segment_ids.shape[1],
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        cu_seqlens=prepared.cu_seqlens,
        plan=prepared.plan,
    )
    return value, final_state
