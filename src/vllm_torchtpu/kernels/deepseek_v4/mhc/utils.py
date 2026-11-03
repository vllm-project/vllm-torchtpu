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

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16
# Rows in an f32 vreg tile.
F32_SUBLANE = 8

# Explicit scoped-VMEM budget, repo convention (mla/v1, deepseek_v4 and
# fused_moe use the same constant). Never rely on the backend default —
# it varies by Mosaic version.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def round_up(x: int, multiple: int) -> int:
    return (x + multiple - 1) // multiple * multiple


def select_token_block(
    num_tokens: int,
    token_block_size: int,
    *,
    vmem_need: Callable[[int], int] | None = None,
    vmem_limit_bytes: int = DEFAULT_VMEM_LIMIT_BYTES,
) -> tuple[int, int]:
    """Grid token block, and the token count padded to a multiple of it.

    ``vmem_need(tb)`` estimates a kernel's VMEM footprint for a block of
    ``tb`` tokens. When supplied, the block halves until it fits within
    ``vmem_limit_bytes`` — degrading to a smaller block instead of a
    compile-time VMEM OOM — with ``SUBLANE`` as the floor.

    Returns (token_block, padded_tokens).
    """
    tb = min(token_block_size, round_up(num_tokens, SUBLANE))
    if vmem_need is not None:
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


def stack_fn3(fn: jax.Array) -> jax.Array:
    """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24), chunks
    stacked along the output dim: (3 * N, K), hi rows first, then mid, lo.

    Computed in XLA outside the kernel; the stack is what stays
    VMEM-resident. Chunks MUST be built with reduce_precision, not dtype
    round-trips: XLA's excess-precision simplification folds
    f32->bf16->f32 into the identity, which silently zeroes the mid/lo
    chunks. Built elementwise over three row-stacked copies of fn rather
    than as a concatenation of the chunks, which XLA leaves as a second
    fusion.

    The masks pass through an optimization barrier with fn so that each
    layer's stack has operands of its own: with one shared mask, XLA merges
    every layer's stack into a single multi-output fusion at the start of
    the step, which pushes the gates' small copies out of VMEM.
    """
    n = fn.shape[0]
    row = jax.lax.broadcasted_iota(jnp.int32, (3 * n, 1), 0)
    fn, is_hi, is_mid = jax.lax.optimization_barrier((fn, row < n, row < 2 * n))
    x = jnp.tile(fn, (3, 1))
    hi = jax.lax.reduce_precision(x, 8, 7)
    rem = x - hi
    mid = jax.lax.reduce_precision(rem, 8, 7)
    lo = jax.lax.reduce_precision(rem - mid, 8, 7)
    return jnp.where(is_hi, hi, jnp.where(is_mid, mid, lo)).astype(jnp.bfloat16)


def dot_fn_chunks(x: jax.Array, fn_chunks: jax.Array, n: int) -> jax.Array:
    """In-kernel ``x @ fn.T`` (f32) from ``fn``'s bf16 chunks stacked along
    the output dim (``stack_fn3``, or the hi chunk alone).

    At DeepSeek-V4 shapes the chunks' columns fit one MXU tile, so one dot
    streams ``x`` once for all of them; the per-chunk column blocks are
    then added in chunk order.
    """
    assert fn_chunks.shape[0] % n == 0, (fn_chunks.shape, n)
    dn = (((1,), (1,)), ((), ()))
    prod = jax.lax.dot_general(x, fn_chunks, dn, preferred_element_type=jnp.float32)
    acc = prod[:, :n]
    for c in range(1, fn_chunks.shape[0] // n):
        acc = acc + prod[:, c * n : (c + 1) * n]
    return acc


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
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-1, keepdims=True) + hc_sinkhorn_eps
        )
        comb_mix = comb_mix / (
            jnp.sum(comb_mix, axis=-2, keepdims=True) + hc_sinkhorn_eps
        )
    return pre_mix, post_mix, comb_mix


def mhc_post_comb_gates_t(
    mixes_t: jax.Array,
    sqrsum_t: jax.Array,
    post_scale: jax.Array,
    comb_scale: jax.Array,
    base_t: jax.Array,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> tuple[jax.Array, jax.Array]:
    """``mhc_pre_gates``' post and comb gates inside a kernel, tokens on lanes.

    Takes the transposed mixes (hc_mult3, T), squared sums (1, T) and
    logit biases (hc_mult3, 1); returns post_mix (M, T) and comb_mix
    (M * M, T), row-major (i, j). Each comb row is an (M, T) array, so a
    row sum is a sublane reduction and a column sum adds the rows; with
    the tokens on sublanes instead, 16 of 128 lanes would hold data. The
    formulas are ``mhc_pre_gates``', in its order; the sums may associate
    differently.
    """
    m = hc_mult
    eps = hc_sinkhorn_eps
    mixes_t = mixes_t * jax.lax.rsqrt(sqrsum_t / (m * hidden_size) + rms_eps)

    post_logits = mixes_t[m : 2 * m] * post_scale + base_t[m : 2 * m]
    post_t = jax.nn.sigmoid(post_logits) * hc_post_mult_value

    def comb_row(i):
        k = (2 + i) * m
        return mixes_t[k : k + m] * comb_scale + base_t[k : k + m]

    def padded(row):
        # Zero rows up to whole f32 sublane tiles: rows then add without
        # sublane shifts, and the zero entries leave every sum unchanged.
        extra = round_up(m, F32_SUBLANE) - m
        if not extra:
            return row
        return jnp.concatenate([row, jnp.zeros((extra, row.shape[1]), row.dtype)])

    rows = [padded(jax.nn.softmax(comb_row(i), axis=0) + eps) for i in range(m)]

    def normalize_cols(rows):
        col_sum = rows[0]
        for r in rows[1:]:
            col_sum = col_sum + r
        return [r / (col_sum + eps) for r in rows]

    def normalize_rows(rows):
        return [r / (jnp.sum(r, axis=0, keepdims=True) + eps) for r in rows]

    rows = normalize_cols(rows)
    for _ in range(sinkhorn_repeat - 1):
        rows = normalize_cols(normalize_rows(rows))
    return post_t, jnp.concatenate([r[:m] for r in rows], axis=0)
