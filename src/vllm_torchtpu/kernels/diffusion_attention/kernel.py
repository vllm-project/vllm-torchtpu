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

#
# Adapted from a code-reviewed TPU diffusion kernel implementation.
# Changes from the original:
# - Inlined LOG2_E, FIXED_M_RECENTER, FIXED_M_SAFE_BOUND,
#   max_row_norm_per_head(), and fixed_m_operands().
# - Removed the ring-attention residual path (residual_attention,
#   emit_residuals) and the E4M3 QK path (qk_fp8), which are not used here.
# - flash_attention_bshd() replaces attention(): (B, S, H, D) inputs and
#   outputs, block sizes from autotune_block_sizes(), and fixed-m only
#   when K/V spans more than one block. Lane-aligned heads are DMA'd as
#   column blocks of the (B, S, H * D) activations (column_heads) when the
#   head count is large or K/V fits one block, and transposed to (B, H, S, D)
#   for smaller multi-block head counts (such as Ulysses-sharded attention).
# - q and k reach the kernel unscaled and uncentred: the kernel multiplies
#   its f32 scores by scale * log2(e) (score_scale), and fixed_m_operands()
#   bounds raw-key scores via the centred key norm and key-mean norm so
#   neither q nor k is rounded to bf16 a second time.
# - fixed_m_operands() repeats the key norms over GQA query heads.
# - end() broadcasts 1 / l by slicing: pltpu.repeat is not in JAX 0.11.
# - Lowered FIXED_M_SAFE_BOUND from 213.0 to 200.0 so low-norm query rows
#   stay at least 13 log2 units above the f32 flush threshold.
# - Set _splash_vmem_limit_bytes() to 1x physical VMEM and check
#   head_dim % spec.num_lanes.
# ruff: noqa: E741

"""Custom Pallas flash attention kernel for TPU.

A hand-written Pallas kernel using ``exp2`` fused softmax, ``fori_loop`` KV
tiling, and a hybrid per-head fixed-m / online softmax. The device-local
entry point ``flash_attention_bshd`` wraps the kernel body
(``_flash_attention_kernel``) and returns the normalized output. It knows
nothing about a mesh or a runtime: callers own the ``shard_map`` and the
collectives around it.
"""

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.diffusion_attention.block_sizes import (
    autotune_block_sizes,
    detect_spec,
)

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
NUM_SUBLANES = 8
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))


# ---------------------------------------------------------------------------
# Fixed-m softmax bound helpers.
# ---------------------------------------------------------------------------

# log2(e): the kernels exponentiate with exp2, so natural-log logits are
# scaled by it. One literal, shared by the kernels and the bounds here,
# because the bound must be in exactly the units the kernel forms.
LOG2_E = 1.44269504

# Recenter C in s = exp2(qk - m + C): C=88 maximizes the no-flush window
# (l ~ 2^105, o ~ 2^112 stay in f32). A row flushes (l=0 -> NaN) once
# m - rowmax > C + 126.
FIXED_M_RECENTER = 88.0

# Flat gate: fixed-m only if the head's Cauchy-Schwarz bound <= 200.
# k-smoothing forces rowmax >= 0, and keeping m = ceil(bound) <= 200 leaves at
# least 13 log2 units above the f32 flush threshold (m - (C + 126) <= -13),
# so keys flushed to zero in exp2(qk - (m - C)) sit below 2^-14 of a
# unit-norm row max (avoiding accuracy loss on low-norm query rows near the
# flush ceiling 214). Outlier heads exceed it and keep online softmax.
FIXED_M_SAFE_BOUND = 200.0


def max_row_norm_per_head(x: jax.Array, *, seq_axis: int = 2) -> jax.Array:
    """Max L2 row norm per head over the device-local batch.

    Args:
        x: ``(B, heads, seq, dim)`` (``seq_axis=2``) or ``(B, seq, heads,
            dim)`` (``seq_axis=1``); the d-minor reduce is ~3x faster on
            the seq-major layout.
        seq_axis: which axis is the sequence.

    Returns:
        ``(heads,)`` f32 — max over ``(B, seq)`` of ``|x[..., :]|``.
    """
    # Max over batch too so the prefetch operand stays UNMAPPED under the
    # batch vmap (a mapped operand slows the pallas_call ~1.6x at batch 2).
    return jnp.sqrt((x.astype(jnp.float32) ** 2).sum(axis=-1)).max(axis=(0, seq_axis))


def fixed_m_operands(
    query_scaled: jax.Array, key: jax.Array, *, gate: float, seq_axis: int = 2
) -> jax.Array:
    """Builds the ``(2, heads)`` fixed-m scalar-prefetch operand.

    Splitting ``k = (k - mean(k)) + mean(k)`` bounds raw-key scores without
    rounding ``k - mean(k)`` to bf16 or subtracting a per-row offset:
    ``max_j(q_i . k_j) <= |q_i| (max_j |k_j - mean(k)| + |mean(k)|)`` gives
    the upper bound ``m``, and since ``max_j(q_i . (k_j - mean(k))) >= 0``,
    ``m - max_j(q_i . k_j) <= |q_i| (max_j |k_j - mean(k)| + 2 |mean(k)|)``
    bounds the shift against ``gate``.

    Args:
        query_scaled: query already scaled by ``scale * log2(e)``.
        key: keys, same layout as ``query_scaled``.
        gate: eligibility bound; heads above it fall back to online
            softmax in-kernel.
        seq_axis: sequence axis, per ``max_row_norm_per_head``.

    Returns:
        Row 0 = integral CS bound (the kernel's m), row 1 = flag.
    """
    k32 = key.astype(jnp.float32)
    k_mean = k32.mean(axis=seq_axis, keepdims=True)
    qn_max = max_row_norm_per_head(query_scaled, seq_axis=seq_axis)
    kn_cent_max = max_row_norm_per_head(k32 - k_mean, seq_axis=seq_axis)
    kn_mean_norm = jnp.sqrt((k_mean**2).sum(axis=-1)).max(axis=(0, seq_axis))
    # GQA: each query head takes the norms of the kv head it attends to.
    rep = qn_max.shape[0] // kn_cent_max.shape[0]
    kn_cent_max = jnp.repeat(kn_cent_max, rep)
    kn_mean_norm = jnp.repeat(kn_mean_norm, rep)
    m_bound = qn_max * (kn_cent_max + kn_mean_norm)
    gate_bound = qn_max * (kn_cent_max + 2.0 * kn_mean_norm)
    # Row 0 is the integral bound the kernel uses as m directly; integral
    # so exp2(m) is exact.
    return jnp.stack([jnp.ceil(m_bound), (gate_bound <= gate).astype(jnp.float32)])


@functools.cache
def _splash_vmem_limit_bytes() -> int:
    """Mosaic scoped-VMEM ceiling per pallas_call, adapted to the device.

    Set to physical core VMEM: the block sizes autotune_block_sizes() picks
    may need up to physical VMEM less 1 MiB (see vmem_footprint_bytes), above
    the 32 MiB Mosaic default on libtpu >= 0.0.41; the XLA vmem flag does not
    govern Pallas. Scales across generations.

    Returns:
      The scoped-VMEM limit in bytes.
    """
    return detect_spec().vmem_bytes


def _exp2(x: jax.Array) -> jax.Array:
    return jnp.exp2(x)


# ---------------------------------------------------------------------------
# pylint: disable=too-few-public-methods
class BlockSizes:
    """Pallas grid tiling for one splash-attention call.

    Attributes:
      block_q: Query rows per grid step.
      block_kv: Key/value rows per grid step.
      block_kv_compute: Key/value rows per inner compute step; defaults
        to ``block_kv``, i.e. one compute step per KV tile.
    """

    __slots__ = ("block_q", "block_kv", "block_kv_compute")

    def __init__(
        self, block_q: int, block_kv: int, block_kv_compute: int | None = None
    ):
        self.block_q = block_q
        self.block_kv = block_kv
        self.block_kv_compute = (
            block_kv_compute if block_kv_compute is not None else block_kv
        )


# pylint: disable=too-many-statements
def _flash_attention_kernel(
    mk_ref: Any,  # (2, num_heads) f32 prefetch: [0]=ceil(CS bound), [1]=flag
    q_ref: Any,  # (bq, head_dim_qk) bf16
    k_ref: Any,  # (bkv, head_dim_qk) bf16
    v_ref: Any,  # (bkv, head_dim_v) bf16
    *refs: Any,  # (m, l, o) cross-j scratch, real outputs
    mask_value: float,
    grid_width: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    kv_seq_len: int,
    use_fixed_m: bool,
    score_scale: float,
) -> None:
    """Hybrid fixed-m / online-softmax flash attention over one (h, i) tile.

    With ``use_fixed_m``, per-head flag ``mk_ref[1]`` routes in-window heads
    onto the fixed-m fast path (``m`` = integral CS bound, no reduce_max /
    alpha-rescale); outliers keep online softmax. Matmuls: bf16 x bf16, f32
    accumulate. Scores are ``score_scale * (q . k)`` in f32.
    """
    float32 = jnp.float32
    # Pallas passes inputs then outputs positionally. The first three
    # "outputs" are the cross-j (m, l, o) scratch, declared as outputs with
    # constant index maps so they persist across the j sweep; the rest are
    # the real outputs.
    m_scratch_ref, l_scratch_ref, o_scratch_ref, *out_refs = refs
    _, rem = divmod(head_dim_v, NUM_SUBLANES)
    if rem != 0:
        raise NotImplementedError(
            f"{head_dim_v=} should be a multiple of {NUM_SUBLANES}"
        )
    sv_dims = (((0,), (0,)), ((), ()))  # V . s contracts over the kv sub-block.

    h = pl.program_id(0)
    j = pl.program_id(2)

    # Per-head dispatch via scalar prefetch; branch at body level only
    # (per-step conditionals are far more expensive).
    is_fixed = mk_ref[1, h] > 0.5 if use_fixed_m else False

    @pl.when(j == 0)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)
        l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        if use_fixed_m:

            @pl.when(is_fixed)
            def init_fixed():
                # m = the head's integral Cauchy-Schwarz bound, prefetched as a
                # scalar. Deriving it per q-row in-kernel would expand the whole
                # (bq, head_dim) tile to f32 and reduce it on every (head,
                # q-block) — costly in bf16.
                # The per-head max is a valid m for every row in the tile: the
                # eligibility gate already bounds it inside the no-flush window
                # (see FIXED_M_SAFE_BOUND), and m is a common scale that
                # cancels in o/l. Stored as (m - C) so the hot loop stays
                # exp2(qk - m).
                m_scratch_ref[...] = jnp.full_like(
                    m_scratch_ref, mk_ref[0, h] - FIXED_M_RECENTER
                )

            @pl.when(jnp.logical_not(is_fixed))
            def init_online():
                m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

        else:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)

    def _update(slice_k, fixed: bool):
        """One KV sub-block update; ``fixed`` selects the softmax variant."""
        q = q_ref[...]  # (bq, head_dim) bf16
        k_chunk = k_ref[slice_k, :]  # (bkv_compute, head_dim) bf16
        v_chunk = v_ref[slice_k, :]  # (bkv_compute, head_dim_v) bf16
        # QK matmul: bf16 x bf16 -> f32 accumulate, (bkv_compute, bq).
        qk = lax.dot_general(k_chunk, q, NT_DIM_NUMBERS, preferred_element_type=float32)

        if fixed:
            # Fixed m is a per-head constant: no reduce_max and no alpha
            # rescale — plain accumulate at scale 2^-m. Read it from scalar
            # prefetch, not m_scratch: the scratch tile is (NUM_SUBLANES, bq)
            # f32 and only row 0 carries information.
            s_curr = _exp2(qk * score_scale - (mk_ref[0, h] - FIXED_M_RECENTER))
            s_cast = s_curr.astype(jnp.bfloat16)
            # l comes from an f32 VPU reduce, not the textbook "append a ones
            # row to V and read l out of the PV matmul": s never leaves VREGs,
            # so the reduce is register work rather than a pass over the tile,
            # while the ones row pushes head_dim_v 128 -> 129 into a worse MXU
            # tile. Measured 21.87 ms (this) / 22.61 (matmul) / 23.02 (bf16).
            l_curr = s_curr.sum(axis=0, keepdims=True).astype(float32)
            o_curr = lax.dot_general(
                v_chunk, s_cast, sv_dims, preferred_element_type=float32
            )
            # o_prev read here, not before the matmuls: a longer live range
            # costs VMEM pressure across both dots.
            o_scratch_ref[:] = o_scratch_ref[:] + o_curr
            l_scratch_ref[...] = l_scratch_ref[...] + l_curr
            return

        # Online softmax update (all f32).
        qk = qk * score_scale
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        m_curr = qk.max(axis=0)[None, :]
        m_next = jnp.maximum(m_prev, m_curr)
        s_curr = _exp2(qk - m_next[0:1])
        s_cast = s_curr.astype(jnp.bfloat16)
        l_curr = s_curr.sum(axis=0, keepdims=True)
        alpha = jnp.exp2(m_prev - m_next)
        o_curr = lax.dot_general(
            v_chunk,
            s_cast,
            sv_dims,
            preferred_element_type=float32,
        )
        # alpha[0:1] rescales the running o to the new max before adding.
        o_scratch_ref[:] = alpha[0:1, ...] * o_scratch_ref[:] + o_curr
        m_scratch_ref[...], l_scratch_ref[...] = m_next, l_curr + alpha * l_prev

    def _compute_body(fixed: bool):
        def compute_body(kv_compute_index, _):
            _update(pl.ds(kv_compute_index * bkv_compute, bkv_compute), fixed)

        return compute_body

    compute_online = _compute_body(False)

    def last_compute_body(kv_compute_index):
        # Final partial sub-block when kv_seq_len % bkv_compute != 0.
        slice_len = kv_seq_len % bkv_compute
        _update(pl.ds(kv_compute_index * bkv_compute, slice_len), False)

    assert bkv % bkv_compute == 0

    def _run_main(body_fn):
        lax.fori_loop(0, bkv // bkv_compute, body_fn, None, unroll=True)

    def _run_last():
        # The last KV block always runs the online body (see the note below).
        if kv_seq_len % bkv == 0:
            lax.fori_loop(0, bkv // bkv_compute, compute_online, None, unroll=True)
        else:
            # The last iter may contain padding; handle it separately.
            remain = kv_seq_len % bkv
            iters = (remain + bkv_compute - 1) // bkv_compute
            if remain % bkv_compute == 0:
                lax.fori_loop(0, iters, compute_online, None, unroll=True)
            else:
                lax.fori_loop(0, iters - 1, compute_online, None, unroll=True)
                last_compute_body(iters - 1)

    if use_fixed_m:
        compute_fixed = _compute_body(True)

        @pl.when((j != grid_width - 1) & is_fixed)
        def body_fixed():
            _run_main(compute_fixed)

        @pl.when((j != grid_width - 1) & jnp.logical_not(is_fixed))
        def body_online():
            _run_main(compute_online)

    else:

        @pl.when(j != grid_width - 1)
        def body():
            _run_main(compute_online)

    # Last KV block runs online for EVERY head (exact: fixed heads arrive at
    # scale 2^-m with m_scratch=m, which alpha-rescale absorbs). Must be a
    # single body: a dual fixed+online last branch degrades the Mosaic
    # schedule ~2.5x at most KV edge widths.
    @pl.when(j == grid_width - 1)
    def last_body():
        _run_last()

    @pl.when(j == grid_width - 1)
    def end():
        # Normalize by l; the (1, bq) inverse broadcasts up sublanes (every
        # row of l holds the same sums).
        (o_ref,) = out_refs
        l = l_scratch_ref[...]
        l_inv = 1.0 / l[0:1]
        o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)


def _splash_attention_forward(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    mk: jax.Array,
    block_sizes: BlockSizes,
    *,
    score_scale: float,
    q_seq_len: int | None = None,
    kv_seq_len: int | None = None,
    use_fixed_m: bool = False,
    column_heads: tuple[int, int] | None = None,
):
    """Runs the fused splash kernel over ``(heads, seq, dim)`` q/k/v.

    Args:
        q: ``(num_heads, seq_q, head_dim)`` bf16.
        k, v: ``(num_kv_heads, seq_kv, head_dim)`` bf16.
        mk: ``(2, num_heads)`` f32 fixed-m prefetch operand (row 0 =
            integral CS bound used as m, row 1 = eligibility flag); a
            ``(2, 1)`` zeros placeholder when ``use_fixed_m`` is off.
        block_sizes: kernel block sizes (clamped to seq by the caller).
        score_scale: multiplies the f32 ``q . k`` scores (``scale *
            log2(e)``).
        q_seq_len, kv_seq_len: unpadded lengths (default to the padded
            shapes).
        use_fixed_m: hybrid fixed-m softmax (see the kernel docstring).
        column_heads: ``(num_heads, num_kv_heads)`` when q/k/v are
            ``(seq, heads * dim)`` instead: each head is a lane-aligned
            column block, DMA'd in place with no transposes.

    Returns:
        The normalized output ``(num_heads, head_dim_v, seq_q)`` in
        ``q.dtype``.
    """
    if column_heads is None:
        num_q_heads, padded_q_seq_len, head_dim_qk = q.shape
        head_dim_v = v.shape[-1]
        num_kv_heads = k.shape[0]
        padded_kv_seq_len = k.shape[1]
    else:
        num_q_heads, num_kv_heads = column_heads
        padded_q_seq_len, padded_kv_seq_len = q.shape[0], k.shape[0]
        head_dim_qk = q.shape[1] // num_q_heads
        head_dim_v = v.shape[1] // num_kv_heads
    bq, bkv = block_sizes.block_q, block_sizes.block_kv
    bkv_compute = block_sizes.block_kv_compute
    # Use original (unpadded) lengths for padding handling inside the kernel.
    actual_q_seq_len = q_seq_len if q_seq_len is not None else padded_q_seq_len
    actual_kv_seq_len = kv_seq_len if kv_seq_len is not None else padded_kv_seq_len
    # GQA broadcast: each q-head reads the kv-head it maps to (MHA has
    # q_heads_per_kv_head == 1, so h // 1 == h).
    q_heads_per_kv_head = num_q_heads // num_kv_heads

    def q_index_map(h, i, j, *_):
        return (h, i, 0)

    def kv_index_map(h, i, j, *_):
        return (h // q_heads_per_kv_head, j, 0)

    def out_index_map(h, i, j, *_):
        return (h, 0, i)

    in_specs = [
        pl.BlockSpec((None, bq, head_dim_qk), q_index_map),
        pl.BlockSpec((None, bkv, head_dim_qk), kv_index_map),
        pl.BlockSpec((None, bkv, head_dim_v), kv_index_map),
    ]
    if column_heads is not None:
        # Head h is column block h of the (seq, heads * dim) operands.
        def q_column_map(h, i, j, *_):
            return (i, h)

        def kv_column_map(h, i, j, *_):
            return (j, h // q_heads_per_kv_head)

        in_specs = [
            pl.BlockSpec((bq, head_dim_qk), q_column_map),
            pl.BlockSpec((bkv, head_dim_qk), kv_column_map),
            pl.BlockSpec((bkv, head_dim_v), kv_column_map),
        ]
    # First three outputs are cross-j scratch (a single block reused per
    # head); the rest are the real per-head outputs. Normalized o keeps
    # q.dtype.
    out_shapes = [
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        jax.ShapeDtypeStruct((NUM_SUBLANES, bq), jnp.float32),
        jax.ShapeDtypeStruct((head_dim_v, bq), jnp.float32),
    ]
    out_specs = [
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        pl.BlockSpec((NUM_SUBLANES, bq), lambda *_: (0, 0)),
        pl.BlockSpec((head_dim_v, bq), lambda *_: (0, 0)),
    ]
    # Index of the first REAL output, after the scratch block above.
    first_out = len(out_shapes)
    out_shapes.append(
        jax.ShapeDtypeStruct((num_q_heads, head_dim_v, actual_q_seq_len), q.dtype)
    )
    out_specs.append(pl.BlockSpec((None, head_dim_v, bq), out_index_map))
    # Use actual (unpadded) seq lens with ceiling division.
    grid_width = (actual_kv_seq_len + bkv - 1) // bkv
    grid_height = (actual_q_seq_len + bq - 1) // bq
    grid = (num_q_heads, grid_height, grid_width)

    all_out = pl.pallas_call(
        functools.partial(
            _flash_attention_kernel,
            mask_value=DEFAULT_MASK_VALUE,
            grid_width=grid_width,
            bkv=bkv,
            bkv_compute=bkv_compute,
            head_dim_v=head_dim_v,
            kv_seq_len=actual_kv_seq_len,
            use_fixed_m=use_fixed_m,
            score_scale=score_scale,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=in_specs,
            out_specs=out_specs,
            grid=grid,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            flags={"XLA_TPU_FORCE_LP_LLO_SCHEDULER": True},
            vmem_limit_bytes=_splash_vmem_limit_bytes(),
        ),
        out_shape=out_shapes,
    )(mk, q, k, v)
    return all_out[first_out]


def _local_blocks(
    q_seq_len: int,
    kv_seq_len: int,
    block_q: int,
    block_kv: int,
    block_kv_compute: int,
) -> BlockSizes:
    """Clamps the requested blocks to one device-local call's extents.

    No physical pad: Mosaic masks partial edge blocks and the kernel slices
    the last sub-block itself (padding cost ~20 ms/step). Clamping keeps an
    oversized block from allocating a full tile; a block equal to the array
    dimension is legal regardless of 8/128 alignment.

    Args:
      q_seq_len: Unpadded query length.
      kv_seq_len: Unpadded key/value length.
      block_q: Requested query block.
      block_kv: Requested key/value block.
      block_kv_compute: Requested key/value compute block.

    Returns:
      The clamped block sizes.
    """
    return BlockSizes(
        block_q=min(block_q, q_seq_len),
        block_kv=min(block_kv, kv_seq_len),
        block_kv_compute=min(block_kv_compute, kv_seq_len),
    )


def flash_attention_bshd(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    *,
    scale: float | None,
    use_fixed_m: bool = True,
) -> jax.Array:
    """Device-local custom splash attention, normalized.

    Args:
      query: ``(B, seq_q, heads, dim)`` device-local.
      key: ``(B, seq_kv, kv_heads, dim)`` device-local; ``heads`` is a
        multiple of ``kv_heads``.
      value: ``(B, seq_kv, kv_heads, dim)`` device-local.
      scale: Softmax scale; ``None`` means ``1/sqrt(dim)``.
      use_fixed_m: Hybrid fixed-m softmax, used when K/V spans more than one
        block. ``key`` is passed uncentred; ``fixed_m_operands`` splits
        ``k = (k - mean(k)) + mean(k)`` in f32 to bound raw-key scores
        against ``FIXED_M_SAFE_BOUND`` without a second bf16 rounding.

    Returns:
      ``(B, seq_q, heads, dim)`` in ``query``'s dtype.
    """
    q_orig_len, num_heads, head_dim = query.shape[1:]
    kv_orig_len, num_kv_heads, head_dim_v = value.shape[1:]
    spec = detect_spec()
    fixed_tiles = use_fixed_m and kv_orig_len > spec.safe_kv_block
    bsizes = _local_blocks(
        q_orig_len,
        kv_orig_len,
        **autotune_block_sizes(
            q_orig_len,
            head_dim,
            2,
            spec,
            kv_orig_len,
            "fixedm" if fixed_tiles else "online",
        ),
    )
    use_fixed_m = use_fixed_m and kv_orig_len > bsizes.block_kv
    scale_factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
    # scale and log2(e) (for the exp2-based softmax) multiply the f32 scores
    # in-kernel: pre-scaling q would round it to bf16 a second time.
    score_scale = scale_factor * LOG2_E
    q = query.astype(jnp.bfloat16)
    k = key.astype(jnp.bfloat16)
    v = value.astype(jnp.bfloat16)
    if use_fixed_m:
        mk_arr = fixed_m_operands(
            q.astype(jnp.float32) * score_scale,
            k,
            gate=FIXED_M_SAFE_BOUND,
            seq_axis=1,
        )
    else:
        mk_arr = jnp.zeros((2, 1), jnp.float32)  # unused placeholder
    # When K/V spans multiple blocks and head count is small (e.g. Ulysses
    # sharded heads), transposing to (B, H, S, D) makes every repeated Q/K/V
    # tile DMA contiguous; for large head counts or single-block K/V, column
    # blocks on (B, S, H * D) avoid the transpose.
    if (
        head_dim % spec.num_lanes == 0
        and head_dim_v % spec.num_lanes == 0
        and (not use_fixed_m or num_heads > 16)
    ):
        column_heads = (num_heads, num_kv_heads)
        q, k, v = (x.reshape(*x.shape[:2], -1) for x in (q, k, v))
    else:
        column_heads = None
        q, k, v = (jnp.swapaxes(x, 1, 2) for x in (q, k, v))

    def _kernel_3d(q_3d, k_3d, v_3d):
        # mk_arr is closed over, so it stays unmapped under the vmap —
        # matching the kernel's (2, heads) prefetch operand.
        return _splash_attention_forward(
            q_3d,
            k_3d,
            v_3d,
            mk_arr,
            bsizes,
            score_scale=score_scale,
            q_seq_len=q_orig_len,
            kv_seq_len=kv_orig_len,
            use_fixed_m=use_fixed_m,
            column_heads=column_heads,
        )

    out = jax.vmap(_kernel_3d, in_axes=(0, 0, 0), out_axes=0)(q, k, v)
    # Output is (B, heads, dim, seq) — transpose back to (B, seq, heads, dim).
    return out.transpose(0, 3, 1, 2).astype(query.dtype)
