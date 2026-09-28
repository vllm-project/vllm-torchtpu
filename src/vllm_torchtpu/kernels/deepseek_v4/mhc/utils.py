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

from collections.abc import Callable

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.deepseek_v4.mhc import tuning

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int | None = None,
) -> tuple[int, int]:
    """Grid token block, and the token count padded to a multiple of it.

    ``vmem_need(tb)`` estimates a kernel's VMEM footprint for a block of
    ``tb`` tokens. When supplied, the block halves until it fits within
    ``vmem_limit_bytes`` — degrading to a smaller block instead of a
    compile-time VMEM OOM — with ``SUBLANE`` as the floor. ``None`` resolves
    the budget from the live device (see ``mhc.tuning``), because physical
    VMEM differs per generation: v6e has 128 MiB, v7x has ~64 MiB.

    When the adaptive grid is enabled (the default; see
    ``TPU_DSV4_MHC_ADAPTIVE_GRID``), ``tb`` is capped so that
    ``padded_tokens // tb >= 2`` whenever ``padded_n >= 2 * SUBLANE``,
    preserving Pallas DMA double-buffering across concurrency buckets.

    Returns (token_block, padded_tokens).
    """
    padded_n = round_up(num_tokens, SUBLANE)
    if tuning.adaptive_grid_enabled() and padded_n >= 2 * SUBLANE:
        tb = min(token_block_size, max(SUBLANE, round_up(padded_n // 2, SUBLANE)))
    else:
        tb = min(token_block_size, padded_n)
    if vmem_need is not None:
        if vmem_limit_bytes is None:
            vmem_limit_bytes = tuning.vmem_limit_bytes()
        while tb > SUBLANE and vmem_need(tb) > vmem_limit_bytes:
            tb //= 2
    return tb, round_up(num_tokens, tb)


def pad_to(padded_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Zero-pad the leading (token) axis of each array to ``padded_tokens``."""
    pad = padded_tokens - arrays[0].shape[0]
    if pad == 0:
        return arrays
    return tuple(jnp.pad(a, ((0, pad), (0, 0))) for a in arrays)


def trim_to(num_tokens: int, *arrays: jax.Array) -> tuple[jax.Array, ...]:
    """Inverse of ``pad_to``: drop the rows the padding added."""
    if arrays[0].shape[0] == num_tokens:
        return arrays
    return tuple(a[:num_tokens] for a in arrays)


def split_fn3(fn: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24).

    Computed in XLA outside the kernel; the chunks are what stays
    VMEM-resident. Chunks MUST be built with reduce_precision, not dtype
    round-trips: XLA's excess-precision simplification folds
    f32->bf16->f32 into the identity, which silently zeroes the mid/lo
    chunks.
    """
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    return tuple(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo))


def split_fn3_stacked(fn: jax.Array) -> jax.Array:
    """``split_fn3`` row-stacked into one (3 * hc_mult3, hc_hidden) array.

    The three chunks share an LHS, a contracting dim and an accumulation
    type, so the three GEMMs against them are one GEMM against this stack.
    Stacking here rather than inside the kernel matters: the array is
    VMEM-resident with a constant index_map, so the concatenate is paid
    once at trace time instead of on every grid step.

    Stacking along N leaves each output element's K-accumulation order and
    the K tiling untouched, so summing the three row groups afterwards
    reproduces the unstacked result bit-for-bit.
    """
    return jnp.concatenate(split_fn3(fn), axis=0)


def mhc_pre_gates(
    mixes: jax.Array,
    sqrsum: jax.Array,
    hc_mult: int,
    hidden_size: int,
    hc_scale: jax.Array,
    hc_base: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Gating / softmax / Sinkhorn on the tiny (T, hc_mult3) mix logits.

    Shared by the Pallas ``pre_kernel`` and the fused seam op.
    Returns (pre_mix (T, M), post_mix (T, M), comb_mix (T, M, M)).
    """
    num_tokens = mixes.shape[0]

    mixes = mixes * jax.lax.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = jax.nn.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].reshape(
        num_tokens, hc_mult, hc_mult
    ) * hc_scale[2] + hc_base[2 * hc_mult :].reshape(1, hc_mult, hc_mult)
    comb_mix = jax.nn.softmax(comb_logits, axis=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps)

    # Rolled, not unrolled. `comb_mix` is only (T, hc_mult, hc_mult) -- 16 KB at
    # a decode batch of 256 -- but each normalization is a strict
    # reduce -> broadcast -> divide chain on the previous one, so XLA cannot
    # fuse across iterations. A Python loop therefore emitted
    # 43 layers x 19 iterations x 4 ops = 6,622 separate HLO instructions, each
    # paying a full launch and HBM round-trip on a tensor far too small to
    # amortize either. That measured at 14.15% of decode busy time.
    #
    # The body and its op order are unchanged. The result is bit-identical at
    # the 256- and 2048-token shapes, but differs by 1-2 ULP (<=1.2e-7) at very
    # small token counts, where XLA reassociates the unrolled chain differently
    # than the rolled one. That is orders of magnitude below the bf16 resolution
    # of the activations these gates feed, but it is not bit-exact, so do not
    # assert exact equality in tests.
    def _sinkhorn_step(_, cm: jax.Array) -> jax.Array:
        cm = cm / (jnp.sum(cm, axis=-1, keepdims=True) + hc_sinkhorn_eps)
        cm = cm / (jnp.sum(cm, axis=-2, keepdims=True) + hc_sinkhorn_eps)
        return cm

    comb_mix = jax.lax.fori_loop(0, sinkhorn_repeat - 1, _sinkhorn_step, comb_mix)
    return pre_mix, post_mix, comb_mix


def _softmax_last(row: jax.Array) -> jax.Array:
    """``jax.nn.softmax(row, axis=-1)``, spelled out.

    Written explicitly rather than calling ``jax.nn.softmax`` so the op
    sequence inside a Pallas block is exactly the one the XLA reference
    emits, and so no ``where``/``initial`` machinery has to lower through
    Mosaic.
    """
    row_max = jnp.max(row, axis=-1, keepdims=True)
    unnormalized = jnp.exp(row - jax.lax.stop_gradient(row_max))
    return unnormalized / jnp.sum(unnormalized, axis=-1, keepdims=True)


def mhc_gates_inblock(
    acc: jax.Array,
    sqr: jax.Array,
    hc_mult: int,
    hidden_size: int,
    sc0: jax.Array,
    sc1: jax.Array,
    sc2: jax.Array,
    hb: jax.Array,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre_gates`` computed inside a Pallas block.

    Same gates, same op order, but the ``comb`` matrix is kept in the flat
    ``(tb, hc_mult * hc_mult)`` row-major layout the seam kernel already
    uses for its ``comb`` input, instead of a ``(tb, m, m)`` array. That
    matters for two reasons:

      * A ``(tb, m*m) -> (tb, m, m)`` reshape inside a kernel splits the
        lane dimension into (sublane, lane) — a relayout Mosaic would have
        to materialize, which is exactly the kind of data movement this
        change exists to remove.
      * In flat row-major, the two Sinkhorn reductions are cheap. Row ``i``
        is the contiguous slice ``[i*m:(i+1)*m]``, so the ``axis=-1`` sums
        are per-slice reductions, and the ``axis=-2`` (column) sums fall
        out as a plain elementwise add of the ``m`` slices.

    Running the whole loop here keeps ``comb`` register-resident for all
    ``sinkhorn_repeat`` iterations. In XLA each iteration was a separate
    HBM round-trip on a 16 KB tensor, which measured at 14.15% of decode
    busy time across 6,622 HLO instructions.

    Args:
        acc: (tb, hc_mult * (hc_mult + 2)) f32 — raw, unscaled mixes.
        sqr: (tb, 1) f32 — squared sum for the RMS norm.
        sc0, sc1, sc2: scalars — the pre/post/comb logit scales.
        hb: (1, hc_mult * (hc_mult + 2)) f32 — logit biases.

    Returns:
        (pre_mix (tb, m), post_mix (tb, m), comb_flat (tb, m*m)).
    """
    m, h = hc_mult, hidden_size
    # Scale every column at once, exactly as the XLA reference does before
    # it slices; slicing after an elementwise multiply is equivalent.
    scaled = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)

    pre_mix = jax.nn.sigmoid(scaled[:, :m] * sc0 + hb[:, :m]) + hc_pre_eps

    post_logits = scaled[:, m : 2 * m] * sc1 + hb[:, m : 2 * m]
    post_mix = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = scaled[:, 2 * m :] * sc2 + hb[:, 2 * m :]
    # rows[i] is row i of the m x m matrix, shape (tb, m).
    rows = [
        _softmax_last(comb_logits[:, i * m : (i + 1) * m]) + hc_sinkhorn_eps
        for i in range(m)
    ]

    def _col_sum(rs: list[jax.Array]) -> jax.Array:
        total = rs[0]
        for i in range(1, m):
            total = total + rs[i]
        return total

    # The reference normalizes columns once before entering the loop.
    col = _col_sum(rows)
    rows = [r / (col + hc_sinkhorn_eps) for r in rows]
    for _ in range(sinkhorn_repeat - 1):
        rows = [
            r / (jnp.sum(r, axis=-1, keepdims=True) + hc_sinkhorn_eps) for r in rows
        ]
        col = _col_sum(rows)
        rows = [r / (col + hc_sinkhorn_eps) for r in rows]

    return pre_mix, post_mix, jnp.concatenate(rows, axis=-1)
