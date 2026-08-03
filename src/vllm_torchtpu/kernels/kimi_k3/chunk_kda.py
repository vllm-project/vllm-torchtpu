# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunked Kimi Delta Attention (KDA) Pallas TPU prefill kernel.

Adated from the implementation from the following tokamax PR:
https://github.com/openxla/tokamax/pull/1103
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

__all__ = ["chunk_kda"]


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


def _align_seqs(
    tensors_4d,
    tensors_3d,
    cu_seqlens,
    align,
    aligned_cu_seqlens=None,
):
    """Align (pad) each variable-length sequence to a multiple of ``align``.

  Supports both single-batch (cu_seqlens [N+1]) and batched
  (cu_seqlens [B, N+1]) modes.  In batched mode, each batch element is
  aligned independently and all results are padded to the maximum
  aligned T across batches.
  """
    if cu_seqlens.ndim == 2:
        # Batched: loop over B (values are concrete at trace time).
        B = cu_seqlens.shape[0]
        per_batch_4d = [[] for _ in tensors_4d]
        per_batch_3d = [[] for _ in tensors_3d]
        padded_cus = []
        t_aligned_sizes = []
        for b in range(B):
            t4 = [t[:, b:b + 1, :, :] for t in tensors_4d]
            t3 = [t[:, b:b + 1, :] for t in tensors_3d]
            aligned_cu_b = (None if aligned_cu_seqlens is None else
                            aligned_cu_seqlens[b])
            aligned_4d, aligned_3d, padded_cu_b, _ = _align_seqs(
                t4,
                t3,
                cu_seqlens[b],
                align,
                aligned_cu_seqlens=aligned_cu_b,
            )
            for idx, a in enumerate(aligned_4d):
                per_batch_4d[idx].append(a)
            for idx, a in enumerate(aligned_3d):
                per_batch_3d[idx].append(a)
            padded_cus.append(padded_cu_b)
            t_aligned_sizes.append(aligned_4d[0].shape[2])

        T_max = max(t_aligned_sizes)

        # Pad each batch element to T_max and concatenate along B.
        def _pad_and_cat_4d(tensors_per_batch):
            padded = []
            for t in tensors_per_batch:
                pad_len = T_max - t.shape[2]
                if pad_len > 0:
                    t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len), (0, 0)))
                padded.append(t)
            return jnp.concatenate(padded, axis=1)

        def _pad_and_cat_3d(tensors_per_batch):
            padded = []
            for t in tensors_per_batch:
                pad_len = T_max - t.shape[2]
                if pad_len > 0:
                    t = jnp.pad(t, ((0, 0), (0, 0), (0, pad_len)))
                padded.append(t)
            return jnp.concatenate(padded, axis=1)

        out_4d = [
            _pad_and_cat_4d(per_batch_4d[i]) for i in range(len(tensors_4d))
        ]
        out_3d = [
            _pad_and_cat_3d(per_batch_3d[i]) for i in range(len(tensors_3d))
        ]
        stacked_cu = jnp.stack(padded_cus, axis=0)
        return out_4d, out_3d, stacked_cu, cu_seqlens

    # --- Single-batch path (original) ---
    N = cu_seqlens.shape[0] - 1
    T_old = tensors_4d[0].shape[2]

    seg_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    if aligned_cu_seqlens is None:
        padded_lens = ((seg_lens + align - 1) // align) * align
        padded_cu = jnp.concatenate(
            [jnp.zeros(1, dtype=jnp.int32),
             jnp.cumsum(padded_lens)])
    else:
        padded_cu = aligned_cu_seqlens
    T_new = ((T_old + N * (align - 1) + align - 1) // align) * align

    def _build_gather(i, gather_idx):
        old_start = cu_seqlens[i]
        new_start = padded_cu[i]
        sl = seg_lens[i]
        j = jnp.arange(T_new)
        in_seg = (j >= new_start) & (j < new_start + sl)
        src = old_start + (j - new_start)
        return jnp.where(in_seg, src, gather_idx)

    gather_idx = jnp.full(T_new, T_old, dtype=jnp.int32)
    gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)

    def repack_4d(t):
        # t: [H, B, T, K] — gather along axis 2 (T dimension)
        return jnp.pad(t, ((0, 0), (0, 0), (0, T_new - T_old),
                           (0, 0)))[:, :, gather_idx]

    def repack_3d(t):
        # t: [H, B, T] — gather along axis 2 (T dimension)
        return jnp.pad(t, ((0, 0), (0, 0), (0, T_new - T_old)))[:, :,
                                                                gather_idx]

    return (
        [repack_4d(t) for t in tensors_4d],
        [repack_3d(t) for t in tensors_3d],
        padded_cu,
        cu_seqlens,
    )


def _unalign_output(o, orig_cu_seqlens, aligned_cu_seqlens, T_out):
    """Reverse _align_seqs: scatter aligned output back to original positions.

  Supports batched cu_seqlens [B, N+1] — processes each batch element
  independently.
  """
    if orig_cu_seqlens.ndim == 2:
        B = orig_cu_seqlens.shape[0]
        per_batch = []
        for b in range(B):
            # Use slicing that works for both 3D [H,B,T] and 4D [H,B,T,X]
            ob_slice = jax.lax.dynamic_slice_in_dim(o, b, 1, axis=1)
            ob = _unalign_output(
                ob_slice,
                orig_cu_seqlens[b],
                aligned_cu_seqlens[b],
                T_out,
            )
            per_batch.append(ob)
        return jnp.concatenate(per_batch, axis=1)

    # --- Single-batch path (original) ---
    N = orig_cu_seqlens.shape[0] - 1
    orig_seg_lens = orig_cu_seqlens[1:] - orig_cu_seqlens[:-1]

    def _build_gather(i, gather_idx):
        orig_start = orig_cu_seqlens[i]
        aligned_start = aligned_cu_seqlens[i]
        sl = orig_seg_lens[i]
        j = jnp.arange(T_out)
        in_seg = (j >= orig_start) & (j < orig_start + sl)
        src = aligned_start + (j - orig_start)
        return jnp.where(in_seg, src, gather_idx)

    # Default to aligned_cu_seqlens[-1] — a known-zero padding position.
    # After the _align_seqs fix above, T_aligned > padded_cu[-1], so this
    # index is always valid and always reads padding (zero).
    safe_default = aligned_cu_seqlens[-1]
    gather_idx = jnp.full(T_out, safe_default, dtype=jnp.int32)
    gather_idx = jax.lax.fori_loop(0, N, _build_gather, gather_idx)
    return o[:, :, gather_idx]


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


def prepare_chunk_indices(
    cu_seqlens: jax.Array,
    chunk_size: int,
    max_T: int | None = None,
) -> jax.Array:
    """Compute per-chunk `(seq_id, block_id)` mapping from cu_seqlens."""
    if cu_seqlens.ndim == 2:
        rows = [
            prepare_chunk_indices(cu_seqlens[b], chunk_size, max_T=max_T)
            for b in range(cu_seqlens.shape[0])
        ]
        return jnp.stack(rows, axis=0)
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = cdiv(lens, chunk_size)
    num_seqs = len(lens)
    if max_T is None:
        max_T = cu_seqlens[-1]
    total_nt = max_T // chunk_size
    seq_ids = jnp.repeat(
        jnp.arange(num_seqs, dtype=jnp.int32),
        n_chunks,
        total_repeat_length=total_nt,
    )
    prefix_chunks = jnp.concatenate([
        jnp.zeros(1, dtype=jnp.int32),
        jnp.cumsum(n_chunks),
    ])
    seq_offsets = jnp.repeat(
        prefix_chunks[:-1],
        n_chunks,
        total_repeat_length=total_nt,
    )
    block_ids = jnp.arange(total_nt, dtype=jnp.int32) - seq_offsets
    return jnp.stack([seq_ids, block_ids], axis=1)


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


def _fused_gate_intra_kernel(
    q_ref,
    k_ref,
    g_ref,
    beta_ref,
    v_ref,
    A_log_ref,
    dt_bias_ref,
    u_out_ref,
    w_out_ref,
    kg_out_ref,
    Aqk_out_ref,
    g_cumsum_out_ref,
    *,
    chunk_size: int,
    head_dim: int,
    value_dim: int,
    scale: float,
    cumsum_scale: float,
    use_gate_in_kernel: bool,
    lower_bound: float | None,
    mini_batch: int = 1,
):
    """Fused Pallas kernel: gate activation + cumsum + intra-chunk solve.

  The kernel applies gate activation (if requested) and a chunk-local prefix
  sum before constructing Aqk/L.

  For bfloat16 inputs, uses Neumann series inversion.
  For float32 inputs, falls back to exact forward substitution.

  All refs have leading dims from BlockSpec: [1, MB, 1, BT, D].
  A_log_ref: [1, MB, 1, 1, 1] — per-head scalar.
  dt_bias_ref: [1, MB, 1, 1, K] — per-head bias vector.

  MB heads are processed simultaneously via batch-vectorized ops.

  Args:
      mini_batch: int — number of heads processed per grid point (MB).
  """
    dtype = q_ref.dtype
    BT = chunk_size
    BC = 16
    NC = BT // BC
    K = head_dim
    V = value_dim
    MB = mini_batch

    # Load all MB heads at once
    q = q_ref[:, 0, 0]  # [MB, BT, K]
    k = k_ref[:, 0, 0]  # [MB, BT, K]
    g = g_ref[:, 0, 0]  # [MB, BT, K]
    beta = beta_ref[:, 0, 0]  # [MB, BT, 1]
    v = v_ref[:, 0, 0]  # [MB, BT, V]

    # --- Gate activation + cumsum ---
    g_f32 = g.astype(jnp.float32)

    if use_gate_in_kernel:
        dt_b = dt_bias_ref[:, 0, 0, 0]  # [MB, K]
        g_f32 = g_f32 + dt_b[:, None, :]  # [MB, BT, K]
        A_val = A_log_ref[:, 0, 0, 0, 0]  # [MB]
        if lower_bound is None:
            g_f32 = -jnp.exp(A_val)[:, None, None] * jax.nn.softplus(g_f32)
        else:
            g_f32 = lower_bound * jax.nn.sigmoid(
                jnp.exp(A_val)[:, None, None] * g_f32)

    tril = jnp.tril(jnp.ones((BT, BT), dtype=jnp.float32))
    g_cumsum = jax.lax.dot_general(
        tril,
        g_f32,
        (((1, ), (1, )), ((), ())),
        preferred_element_type=jnp.float32,
    ).transpose(1, 0, 2) * cumsum_scale

    q_f32 = q.astype(jnp.float32)
    k_f32 = k.astype(jnp.float32)
    beta_f32 = beta.astype(jnp.float32)

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
    v_beta = v.astype(jnp.float32) * beta_f32  # [MB, BT, V]
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

    u_out_ref[:, 0, 0] = u.astype(u_out_ref.dtype)
    w_out_ref[:, 0, 0] = w.astype(w_out_ref.dtype)
    kg_out_ref[:, 0, 0] = kg.astype(kg_out_ref.dtype)
    Aqk_out_ref[:, 0, 0] = Aqk.astype(Aqk_out_ref.dtype)
    g_cumsum_out_ref[:, 0, 0] = g_cumsum


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
    scale: float,
    chunk_size: int = 64,
    cumsum_scale: float = RCP_LN2,
    A_log: jax.Array | None = None,  # [H]
    dt_bias: jax.Array | None = None,  # [H*K]
    use_gate_in_kernel: bool = False,
    lower_bound: float | None = None,
    mini_batch: int | None = None,
) -> tuple[
        jax.Array,  # [H, B, T, K]
        jax.Array,  # [H, B, T, V]
        jax.Array,  # [H, B, T, K]
        jax.Array,  # [H, B, T, BT]
        jax.Array,  # [H, B, T, K]
]:
    """Fuse gate cumsum with the fixed-length intra-chunk solve."""
    H, B, T, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    NC = T // BT

    if use_gate_in_kernel:
        assert A_log is not None, "A_log required when use_gate_in_kernel=True"

    if mini_batch is None:
        per_head = (BT * K + BT * V + 2 * BT * BT) * q.dtype.itemsize
        MB = estimate_mini_batch(per_head, H, max_mb=16)
    else:
        MB = mini_batch
        assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

    # [H, B, T, K] -> [H, B, NC, BT, K]
    q_r = q.reshape(H, B, NC, BT, K)
    k_r = k.reshape(H, B, NC, BT, K)
    g_r = g.reshape(H, B, NC, BT, K)
    beta_r = beta.reshape(H, B, NC, BT, 1)
    v_r = v.reshape(H, B, NC, BT, V)

    if use_gate_in_kernel:
        A_log_r = A_log.astype(jnp.float32).reshape(H, 1, 1, 1, 1)
        if dt_bias is not None:
            dt_bias_r = dt_bias.astype(jnp.float32).reshape(H, 1, 1, 1, K)
        else:
            dt_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)
    else:
        A_log_r = jnp.zeros((H, 1, 1, 1, 1), dtype=jnp.float32)
        dt_bias_r = jnp.zeros((H, 1, 1, 1, K), dtype=jnp.float32)

    grid = (H // MB, B, NC)

    # Grid is (head block, batch, chunk); `c` is the chunk index.
    def _make_spec(last_dim):
        return pl.BlockSpec(
            index_map=lambda i, j, c: (i, j, c, 0, 0),
            block_shape=(MB, 1, 1, BT, last_dim),
        )

    def _make_per_head_spec(last_dim):
        return pl.BlockSpec(
            index_map=lambda i, j, c: (i, 0, 0, 0, 0),
            block_shape=(MB, 1, 1, 1, last_dim),
        )

    u_r, w_r, kg_r, Aqk_r, g_cumsum_r = pl.pallas_call(
        functools.partial(
            _fused_gate_intra_kernel,
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
            jax.ShapeDtypeStruct((H, B, NC, BT, V), k.dtype),
            jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
            jax.ShapeDtypeStruct((H, B, NC, BT, K), k.dtype),
            jax.ShapeDtypeStruct((H, B, NC, BT, BT), k.dtype),
            jax.ShapeDtypeStruct((H, B, NC, BT, K), jnp.float32),
        ],
        in_specs=[
            _make_spec(K),
            _make_spec(K),
            _make_spec(K),
            _make_spec(1),
            _make_spec(V),
            _make_per_head_spec(1),
            _make_per_head_spec(K),
        ],
        out_specs=[
            _make_spec(V),
            _make_spec(K),
            _make_spec(K),
            _make_spec(BT),
            _make_spec(K),
        ],
        grid=grid,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "parallel"), ),
    )(q_r, k_r, g_r, beta_r, v_r, A_log_r, dt_bias_r)

    # --- Reshape back to [H, B, T, D] (head-first) ---
    w_out = w_r.reshape(H, B, T, K)
    u_out = u_r.reshape(H, B, T, V)
    kg_out = kg_r.reshape(H, B, T, K)
    Aqk_flat = Aqk_r.reshape(H, B, NC * BT, BT)
    g_cumsum_out = g_cumsum_r.reshape(H, B, T, K)

    return w_out, u_out, kg_out, Aqk_flat, g_cumsum_out


# =============================================================================
# Fused state propagation and output
# =============================================================================


def _chunk_kda_fwd_h_o_varlen_kernel(
    seqlens_ref,  # scalar prefetch: cu_seqlens [N+1]
    chunk_to_seq_ref,  # scalar prefetch: chunk -> seq mapping [NT]
    # Stage 3 inputs
    w_ref,  # [MB, 1, BT, K_PADSIZE]
    u_ref,  # [MB, 1, BT, V_ALIGNED]
    kg_ref,  # [MB, 1, BT, K_PADSIZE]
    gk_ref,  # [MB, 1, BT, K_PADSIZE]  -- g_cumsum
    # Stage 4 inputs
    q_ref,  # [MB, 1, BT, K_PADSIZE]
    A_ref,  # [MB, 1, BT, BT]
    # Optional initial state
    h0_ref,  # [1, MB, K_PADSIZE, V_ALIGNED] or None
    # Outputs
    o_ref,  # [MB, 1, BT, V_ALIGNED]
    ht_ref,  # [1, MB, K_PADSIZE, V_ALIGNED] or None
    # Scratch
    scratch_ref,  # [MB, K_PADSIZE, V_ALIGNED]
    *,
    BT,
    scale,
    USE_INITIAL_STATE,
    STORE_FINAL_STATE,
    MB,
    OUTPUT_PRECISION,
):
    """Fused Stage 3+4 Pallas kernel body for varlen with H-dim mini-batch.

  Grid is (H // MB, B, NT). For each program point (h_group, i_b, i_c):
    - MB heads [h_group*MB .. h_group*MB+MB) are processed per grid point
      via batched matmuls over the MB dimension.
    - i_b is the batch index, i_c is the chunk index within that batch.
    - seq_idx = chunk_to_seq[i_b, i_c] identifies which sequence this chunk
      belongs to within batch i_b.
    - At t0 == bos: init scratch (h0 or zeros) for this sequence
    - At t0 + BT >= eos: store final state for this sequence
  """
    i_b = pl.program_id(1)
    i_c = pl.program_id(2)
    seq_idx = chunk_to_seq_ref[i_b, i_c]

    bos = seqlens_ref[i_b, seq_idx]
    eos = seqlens_ref[i_b, seq_idx + 1]
    t0 = i_c * BT

    K = w_ref.shape[3]
    V = u_ref.shape[3]

    # === Init state (first chunk of THIS sequence) — uniform across all MB heads ===
    @pl.when(t0 == bos)
    def _():
        scratch_ref[:] = jnp.zeros([MB, K, V], dtype=jnp.float32)
        if USE_INITIAL_STATE:
            scratch_ref[:] = h0_ref[0, 0].astype(jnp.float32)  # [MB, K, V]

    # === Stage 3+4 work — batched over MB heads ===
    # h: pre-update state for all MB heads in this tile.
    b_h = scratch_ref[:]  # [MB, K, V]

    b_w = w_ref[:, 0, :]  # [MB, BT, K]
    b_u = u_ref[:, 0, :]  # [MB, BT, V]

    # Stage 3 delta correction: v_new = u - w @ h
    # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
    # HIGHEST precision: v_new feeds directly into the recursive state update.
    b_v_new = b_u.astype(jnp.float32) - jnp.matmul(
        b_w.astype(jnp.float32),
        b_h,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )  # [MB, BT, V]

    # Stage 4 inter-chunk output: scale * (q * exp2(g - g_ref)) @ (h * exp2(g_ref))
    # Reference-point stabilization mirrors the production varlen kernel's
    # g_ref = g[0] choice for bit-identical numerics.
    b_q = q_ref[:, 0, :]  # [MB, BT, K]
    b_g = gk_ref[:, 0, :].astype(jnp.float32)  # [MB, BT, K]
    b_A = A_ref[:, 0, :]  # [MB, BT, BT]

    b_g_ref_row = b_g[:, 0:1, :]  # [MB, 1, K]
    b_qg = b_q.astype(jnp.float32) * jnp.exp2(
        jnp.maximum(b_g - b_g_ref_row, -126.0))  # [MB, BT, K]
    b_h_scaled = b_h * jnp.exp2(jnp.maximum(b_g_ref_row[:, 0, :],
                                            -126.0))[:, :, None]  # [MB, K, V]

    # [MB, BT, K] @ [MB, K, V] -> [MB, BT, V]
    # Output-only GEMM: bf16 inputs use DEFAULT (no extra precision to preserve),
    # fp32 inputs use HIGHEST to maintain full mantissa fidelity.
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

    o_ref[:, 0, :] = b_o.astype(o_ref.dtype)

    # Stage 3 state update: h = decay(h) + kg^T @ v_new
    # HIGHEST precision: accumulates directly into the recursive hidden state.
    b_gk_last = gk_ref[:, 0, :][:, BT - 1, :].astype(jnp.float32)  # [MB, K]
    b_h_new = b_h * jnp.exp2(b_gk_last)[:, :, None]  # [MB, K, V] decay

    b_kg = kg_ref[:, 0, :]  # [MB, BT, K]
    # [MB, K, BT] @ [MB, BT, V] -> [MB, K, V]
    b_h_new = b_h_new + jnp.matmul(
        b_kg.astype(jnp.float32).transpose(0, 2, 1),
        b_v_new,
        precision=jax.lax.Precision.HIGHEST,
        preferred_element_type=jnp.float32,
    )
    scratch_ref[:] = b_h_new

    # === Final state (last chunk of THIS sequence) ===
    @pl.when(t0 + BT >= eos)
    def _():
        if STORE_FINAL_STATE:
            ht_ref[0, 0] = scratch_ref[:].astype(ht_ref.dtype)  # [MB, K, V]


@functools.partial(
    jax.jit,
    static_argnames=[
        "output_final_state",
        "scale",
        "chunk_size",
        "mini_batch",
    ],
)
def chunk_kda_fwd_h_o_varlen(
    w: jax.Array,  # [H, B, T, K]
    u: jax.Array,  # [H, B, T, V]
    kg: jax.Array,  # [H, B, T, K]
    gk: jax.Array,  # [H, B, T, K]
    q: jax.Array,  # [H, B, T, K]
    A: jax.Array,  # [H, B, T, BT]
    cu_seqlens: jax.Array,  # [N_CU] or [B, N_CU]
    chunk_indices: jax.Array | None = None,  # [NT, 2] or [B, NT, 2]
    initial_state: jax.Array | None = None,  # [B, N, H, K, V]
    output_final_state: bool = False,
    scale: float = 1.0,
    chunk_size: int = 64,
    mini_batch: int | None = None,
) -> tuple[
        jax.Array,  # [H, B, T, V]
        jax.Array | None,  # [B, N, H, K, V]
]:
    """Fuse variable-length state propagation with output projection."""
    H, B, T, K = q.shape
    V = u.shape[-1]
    BT = chunk_size

    assert T % BT == 0, f"T={T} must be divisible by chunk_size={BT}"
    # Ensure cu_seqlens is 2D [B, N+1] for kernel block specs
    if cu_seqlens.ndim == 1:
        cu_seqlens = jnp.broadcast_to(cu_seqlens[None, :],
                                      (B, cu_seqlens.shape[0]))
    assert A.shape[-1] == BT, (
        f"A.shape[-1]={A.shape[-1]} must equal chunk_size={BT}")

    N = cu_seqlens.shape[-1] - 1
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
        out_bytes = BT * V_ALIGNED * elem_size
        scratch_bytes = K_PADSIZE * V_ALIGNED * 4  # float32 accumulator
        per_head = in_bytes + out_bytes + scratch_bytes
        MB = estimate_mini_batch(per_head, H, max_mb=16)
    else:
        MB = mini_batch
        assert H % MB == 0, f"H={H} must be divisible by mini_batch={MB}"

    # Generate chunk_to_seq mapping from chunk_indices
    if chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = chunk_indices.shape[-2]
    chunk_to_seq = chunk_indices[..., 0].astype(jnp.int32)  # [NT] or [B, NT]
    # Kernel block specs index with [b, c], so ensure 2D [B, NT].
    if chunk_to_seq.ndim == 1:
        chunk_to_seq = jnp.broadcast_to(chunk_to_seq[None, :], (B, NT))

    # grid = (H // MB, NT) with NT = T // BT, so chunk index c ranges
    # 0..NT-1 and the maximum T offset accessed is (NT-1)*BT+BT = T.
    # No extra trailing chunk is ever touched, so T_alloc == T suffices.
    T_alloc = T

    # Pad K (last dim) to K_PADSIZE if needed, then transpose to
    # [B=1, H, T, K_PADSIZE]. No T-dim padding required (see T_alloc above).
    # Inputs stay in their original dtype (typically bf16); the kernel casts
    # tile-level slices to f32 on the fly in VMEM, avoiding a full-tensor
    # bf16->f32 cast in HBM that doubles memory and DMA bandwidth.
    def _pad_kdim_then_t(x, dim_pad):
        if dim_pad > 0:
            x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, dim_pad)))
        return x

    w_t = _pad_kdim_then_t(w, K_PADSIZE - K)
    kg_t = _pad_kdim_then_t(kg, K_PADSIZE - K)
    gk_t = _pad_kdim_then_t(gk, K_PADSIZE - K)
    q_t = _pad_kdim_then_t(q, K_PADSIZE - K)
    u_t = _pad_kdim_then_t(u, V_ALIGNED - V)

    # A is [B, T, H, BT]; transpose to [B=1, H, T, BT]. No T padding needed.
    A_t = A  # [H, B, T, BT]

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

    # Index maps with MB heads per grid point. Scalar prefetch order:
    # (seqlens_ref, chunk_to_seq_ref).
    # Inputs are [H, B, T_alloc, X]; BlockSpec slices MB heads at h*MB.
    def _t_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
        return (h, b, c, 0)

    def _A_index_map(h, b, c, seqlens_ref, chunk_to_seq_ref):
        return (h, b, c, 0)

    bspec_k = pl.BlockSpec([MB, 1, BT, K_PADSIZE], index_map=_t_index_map)
    bspec_v = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    bspec_a = pl.BlockSpec([MB, 1, BT, BT], index_map=_A_index_map)
    bspec_h0 = (pl.BlockSpec(
        [1, 1, MB, K_PADSIZE, V_ALIGNED],
        index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref:
        (b, chunk_to_seq_ref[b, c], h, 0, 0),
    ) if h0 is not None else None)

    # Output specs.
    o_spec = pl.BlockSpec([MB, 1, BT, V_ALIGNED], index_map=_t_index_map)
    ht_spec = (pl.BlockSpec(
        [1, 1, MB, K_PADSIZE, V_ALIGNED],
        index_map=lambda h, b, c, seqlens_ref, chunk_to_seq_ref:
        (b, chunk_to_seq_ref[b, c], h, 0, 0),
    ) if output_final_state else None)
    o_shape = jax.ShapeDtypeStruct([H, B, T_alloc, V_ALIGNED], jnp.float32)
    ht_shape = (jax.ShapeDtypeStruct([B, N, H, K_PADSIZE, V_ALIGNED],
                                     jnp.float32)
                if output_final_state else None)
    scratch = pltpu.VMEM((MB, K_PADSIZE, V_ALIGNED), jnp.float32)
    grid = (H // MB, B, NT)
    interpret = get_interpret()

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
            num_scalar_prefetch=2,
            grid=grid,
            in_specs=[
                bspec_k,  # w
                bspec_v,  # u
                bspec_k,  # kg
                bspec_k,  # gk
                bspec_k,  # q
                bspec_a,  # A
                bspec_h0,  # h0
            ],
            out_specs=[o_spec, ht_spec],
            scratch_shapes=[scratch],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        out_shape=[o_shape, ht_shape],
        interpret=interpret,
    )(cu_seqlens.astype(jnp.int32), chunk_to_seq, w_t, u_t, kg_t, gk_t, q_t,
      A_t, h0)

    # Post-process: o is [H, 1, T, V_ALIGNED]
    if V_ALIGNED > V:
        o_out = o_out[..., :V]
    o_out = o_out.astype(u.dtype)

    if output_final_state and ht_out is not None:
        if V_ALIGNED > V:
            ht_out = ht_out[..., :V]
        if K_PADSIZE > K:
            ht_out = ht_out[..., :K, :]

        # Handle empty sequences: sequences with no chunks never execute kernel code,
        # so their final_state is uninitialized. Fill them with initial_state or zeros.
        seq_lens = jnp.diff(cu_seqlens, axis=-1)
        empty_mask = (seq_lens == 0)  # [B, N]
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
    q: jax.Array,  # [H, B, T_ALIGNED, K]
    k: jax.Array,  # [H, B, T_ALIGNED, K]
    v: jax.Array,  # [H, B, T_ALIGNED, V]
    g: jax.Array,  # [H, B, T_ALIGNED, K]
    beta: jax.Array,  # [H, B, T_ALIGNED]
    *,
    segment_ids: jax.Array,  # [B, T]
    cu_seqlens: jax.Array,  # [B, N_CU]
    aligned_cu_seqlens: jax.Array,  # [B, N_CU]
    chunk_indices: jax.Array,  # [B, NT, 2]
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
    """Run the kernel on canonicalized, chunk-aligned inputs."""
    BT = chunk_size

    # ------------------------------------------------------------------
    # Step 1 + 2 (Fused): Gate cumsum + Intra-chunk solve
    # ------------------------------------------------------------------
    w, u, kg, Aqk, g_cumsum = pallas_kda_fwd_intra_fused(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        chunk_size=BT,
        cumsum_scale=RCP_LN2,
        A_log=A_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=use_gate_in_kernel,
        lower_bound=lower_bound,
    )

    # ------------------------------------------------------------------
    # Step 3 + Step 4: Inter-chunk state + Output (gather/scatter for varlen)
    #
    # The inter-chunk kernel and output kernel require BT-aligned
    # cu_seqlens (they index blocks via bos // BT).  For non-aligned
    # varlen sequences we reuse the existing _align_seqs / _unalign_output
    # utilities to gather inputs into a chunk-aligned layout, run Stages
    # 3 & 4, then scatter the output back.
    #
    # Padding semantics (relied on for correctness):
    #   - k/w/u/q/Aqk at padded tail positions = 0 (from _align_seqs's
    #     jnp.pad with default fill value 0). With zero k/w/u/q the state
    #     update reduces to h_pad = h_{t-1} * exp(g_pad) + 0, and the
    #     output reduces to q_pad * exp(...) @ h = 0 * ... = 0.
    #   - g_cumsum at padded tail positions = 0 ⇒ exp(g_pad) = 1, so the
    #     hidden state is carried through padded positions unchanged.
    # ------------------------------------------------------------------
    # Stage 1/2 already operate in BT-aligned layout; derived tensors inherit
    # that layout, so Stage 3+4 is fed the aligned segment boundaries.
    o, final_state = chunk_kda_fwd_h_o_varlen(
        w=w,
        u=u,
        kg=kg,
        gk=g_cumsum,
        q=q,
        A=Aqk,
        cu_seqlens=aligned_cu_seqlens,
        chunk_indices=chunk_indices,
        initial_state=initial_state,
        output_final_state=output_final_state,
        scale=scale,
        chunk_size=BT,
    )

    output = _unalign_output(
        o,
        cu_seqlens,
        aligned_cu_seqlens,
        segment_ids.shape[1],
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
    aligned_cu_seqlens: jax.Array
    chunk_indices: jax.Array


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
    use_gate_in_kernel: bool,
    segment_ids: jax.Array | None,
    chunk_size: int,
    N_max: int | None,
) -> _PreparedKdaInputs:
    """Canonicalize inputs for the forward kernel."""
    if N_max is None:
        if initial_state is None:
            raise ValueError(
                "`N_max` is required when `initial_state` is not given: the "
                "segment count is a static shape and cannot be inferred.")
        N_max = initial_state.shape[1]
    cu_seqlens = segment_ids_to_seqlens(segment_ids, max_segs=N_max)

    (
        [q_aligned, k_aligned, v_aligned, g_aligned],
        [beta_aligned],
        aligned_cu_seqlens,
        _,
    ) = _align_seqs(
        [q, k, v, g],
        [beta],
        cu_seqlens,
        align=chunk_size,
    )
    chunk_indices = prepare_chunk_indices(
        aligned_cu_seqlens,
        chunk_size,
        max_T=q_aligned.shape[2],
    )

    if use_gate_in_kernel:
        aligned_seq_len = g_aligned.shape[2]
        original_lengths = jnp.diff(cu_seqlens, axis=-1)
        aligned_starts = aligned_cu_seqlens[..., :-1]
        positions = jnp.arange(aligned_seq_len)
        for batch_index in range(cu_seqlens.shape[0]):
            in_range = (positions[None, :]
                        >= aligned_starts[batch_index, :, None]) & (
                            positions[None, :]
                            < (aligned_starts[batch_index] +
                               original_lengths[batch_index])[:, None])
            valid_mask = in_range.any(axis=0)
            g_aligned = g_aligned.at[:, batch_index].set(
                jnp.where(
                    valid_mask[None, :, None],
                    g_aligned[:, batch_index],
                    -1e4,
                ))

    initial_state_prepared = initial_state
    if initial_state is not None:
        state_count = aligned_cu_seqlens.shape[-1] - 1
        if initial_state.shape[1] < state_count:
            initial_state_prepared = jnp.pad(
                initial_state,
                (
                    (0, 0),
                    (0, state_count - initial_state.shape[1]),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
            )
        if initial_state_prepared.shape[1] != state_count:
            raise ValueError(
                "`initial_state` state count must match aligned segment "
                f"count {state_count}; got {initial_state.shape[1]}.")

    if use_qk_l2norm_in_kernel:
        q_f32 = q_aligned.astype(jnp.float32)
        k_f32 = k_aligned.astype(jnp.float32)
        q_prepared = (q_f32 * jax.lax.rsqrt(
            jnp.sum(q_f32 * q_f32, axis=-1, keepdims=True) + 1e-6)).astype(
                q_aligned.dtype)
        k_prepared = (k_f32 * jax.lax.rsqrt(
            jnp.sum(k_f32 * k_f32, axis=-1, keepdims=True) + 1e-6)).astype(
                k_aligned.dtype)
    else:
        q_prepared, k_prepared = q_aligned, k_aligned

    return _PreparedKdaInputs(
        q=q_prepared,
        k=k_prepared,
        v=v_aligned,
        g=g_aligned,
        beta=beta_aligned,
        initial_state=initial_state_prepared,
        cu_seqlens=cu_seqlens,
        aligned_cu_seqlens=aligned_cu_seqlens,
        chunk_indices=chunk_indices,
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
        use_gate_in_kernel=use_gate_in_kernel,
        segment_ids=segment_ids,
        chunk_size=chunk_size,
        N_max=N_max,
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
        segment_ids=segment_ids,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
        cu_seqlens=prepared.cu_seqlens,
        aligned_cu_seqlens=prepared.aligned_cu_seqlens,
        chunk_indices=prepared.chunk_indices,
    )
    return value, final_state
