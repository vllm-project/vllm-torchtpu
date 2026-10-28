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
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp

# bf16 tiles are (16, 128); keep token blocks sublane-aligned.
SUBLANE = 16

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
    whole_small_batch: bool = False,
) -> tuple[int, int]:
    """Grid token block, and the token count padded to a multiple of it.

    ``vmem_need(tb)`` estimates a kernel's VMEM footprint for a block of
    ``tb`` tokens. When supplied, the block halves until it fits within
    ``vmem_limit_bytes`` — degrading to a smaller block instead of a
    compile-time VMEM OOM — with ``SUBLANE`` as the floor.

    ``whole_small_batch``: a batch smaller than one ``SUBLANE`` block is
    not padded; the single block overhangs the arrays instead (rows are
    independent and out-of-range output rows are dropped), which skips the
    pad/trim copies of every operand. Callers size the grid with
    ``pl.cdiv(padded_tokens, token_block)``.

    Returns (token_block, padded_tokens).
    """
    if whole_small_batch and num_tokens < SUBLANE <= token_block_size:
        return SUBLANE, num_tokens
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


class SplitFn(NamedTuple):
    """fn as its three bf16 chunks (see ``split_fn3``), split ahead of time.

    The chunks depend only on the weight, so serving splits each fn once at
    load instead of once per call.
    """

    hi: jax.Array
    mid: jax.Array
    lo: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        return self.hi.shape


def split_fn3(fn: jax.Array | SplitFn) -> SplitFn:
    """3-chunk bf16 split of fn (8 mantissa bits each = f32's 24).

    The chunks are what stays VMEM-resident. Chunks MUST be built with
    reduce_precision, not dtype round-trips: XLA's excess-precision
    simplification folds f32->bf16->f32 into the identity, which silently
    zeroes the mid/lo chunks. An already split fn is returned as is.
    """
    if isinstance(fn, SplitFn):
        assert all(t.dtype == jnp.bfloat16 for t in fn), [t.dtype for t in fn]
        return fn
    assert fn.dtype == jnp.float32, fn.dtype
    fn_hi = jax.lax.reduce_precision(fn, 8, 7)
    rem = fn - fn_hi
    fn_mid = jax.lax.reduce_precision(rem, 8, 7)
    fn_lo = jax.lax.reduce_precision(rem - fn_mid, 8, 7)
    return SplitFn(*(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo)))


class CatFn(NamedTuple):
    """fn's three bf16 chunks stacked along the output axis: (3 * M3, K).

    One MXU pass against the stacked chunks yields the three partial
    products side by side; the MXU pads the 24-wide output to its full
    width anyway, so this costs one pass instead of three. The partials are
    then summed (hi + mid) + lo, the order of the three-pass path.
    """

    cat: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        return (self.cat.shape[0] // 3, self.cat.shape[1])


def cat_fn3(fn: "jax.Array | SplitFn | CatFn") -> CatFn:
    """``split_fn3`` chunks stacked as one (3 * M3, K) bf16 array."""
    if isinstance(fn, CatFn):
        assert fn.cat.dtype == jnp.bfloat16, fn.cat.dtype
        return fn
    return CatFn(jnp.concatenate(tuple(split_fn3(fn)), axis=0))


def fn_operands(fn: "jax.Array | SplitFn | CatFn") -> tuple[jax.Array, ...]:
    """The kernel's fn operands: one stacked array (CatFn) or three chunks."""
    if isinstance(fn, CatFn):
        return (cat_fn3(fn).cat,)
    return tuple(split_fn3(fn))


def mix_gemm(x_bf: jax.Array, fn_refs, hc_mult3: int, gemm_precision: str):
    """x (tb, K) bf16 @ fn.T -> (tb, M3) f32, exact for bf16 x.

    ``fn_refs``: three chunk refs (three MXU passes), or one stacked ref
    (one pass, then the partials are summed in the three-pass order).
    """
    dn = (((1,), (1,)), ((), ()))
    if len(fn_refs) == 1:
        out = jax.lax.dot_general(
            x_bf, fn_refs[0][...], dn, preferred_element_type=jnp.float32
        )
        acc = out[:, :hc_mult3]
        if gemm_precision == "highest":
            acc = acc + out[:, hc_mult3 : 2 * hc_mult3]
            if out.shape[1] > 2 * hc_mult3:  # absent with a 2-chunk read
                acc = acc + out[:, 2 * hc_mult3 :]
        return acc
    fn_hi_ref, fn_mid_ref, fn_lo_ref = fn_refs
    acc = jax.lax.dot_general(
        x_bf, fn_hi_ref[...], dn, preferred_element_type=jnp.float32
    )
    if gemm_precision == "highest":
        acc = acc + jax.lax.dot_general(
            x_bf, fn_mid_ref[...], dn, preferred_element_type=jnp.float32
        )
        acc = acc + jax.lax.dot_general(
            x_bf, fn_lo_ref[...], dn, preferred_element_type=jnp.float32
        )
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


# Above this many tokens the XLA gates epilogue is the faster one: per seam
# at H=7168, in-kernel vs XLA was 76.6 vs 81.9 us at T=512 but 289.8 vs
# 272.3 us at T=2048 (the token-major Sinkhorn fills only tb of 128 lanes).
GATES_IN_KERNEL_MAX_TOKENS = 512


def gate_consts(
    num_tokens: int,
    gates_in_kernel: bool,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    max_tokens: int | None = None,
) -> "GateConsts | None":
    """The in-kernel gate constants, or None for the XLA gates epilogue.

    ``max_tokens``: in-kernel only up to this many tokens (None: the module
    default; 0: no cap -- worth it once the token block fills the 128 lanes
    the token-major Sinkhorn runs on).
    """
    max_tokens = GATES_IN_KERNEL_MAX_TOKENS if max_tokens is None else max_tokens
    if not gates_in_kernel or (max_tokens > 0 and num_tokens > max_tokens):
        return None
    return GateConsts(hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)


class GateConsts(NamedTuple):
    """Static constants of the post/comb gates, for kernels that compute
    them in-block (see ``gates_in_block``)."""

    hc_sinkhorn_eps: float
    hc_post_mult_value: float
    sinkhorn_repeat: int


def _sum4(terms: list[jax.Array]) -> jax.Array:
    if len(terms) == 4:
        return (terms[0] + terms[1]) + (terms[2] + terms[3])
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def gates_in_block(
    acc: jax.Array,
    sqr: jax.Array,
    sc_ref,
    hb_ref,
    *,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    gates: GateConsts,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``mhc_pre_gates`` for one token block, inside a Pallas kernel.

    acc (tb, hc_mult3) raw mixes, sqr (tb, 1) squared sums, sc_ref (1, 3)
    and hb_ref (1, hc_mult3) VMEM refs. Returns pre_mix (tb, M), post_mix
    (tb, M), comb_mix (tb, M*M). Same math as ``mhc_pre_gates``; the
    Mosaic transcendental/division lowerings put post/comb a few f32 ulp
    from the XLA path (pre_mix is bit-identical). The Sinkhorn runs token-major
    (tokens on lanes, one row per (i, j) entry), so each row/column
    reduction is a few full-vreg adds instead of cross-lane work.
    """
    m, h = hc_mult, hidden_size
    eps = gates.hc_sinkhorn_eps
    scaled = acc * jax.lax.rsqrt(sqr / (m * h) + rms_eps)

    pre_mix = jax.nn.sigmoid(scaled[:, :m] * sc_ref[0, 0] + hb_ref[:, :m]) + hc_pre_eps
    post_logits = scaled[:, m : 2 * m] * sc_ref[0, 1] + hb_ref[:, m : 2 * m]
    # Spelled out: Mosaic's native logistic is far less accurate than XLA's
    # (measured 2.9e-3 vs 8.8e-7 max relative error against f64). pre_mix
    # keeps it, to stay bit-identical with the XLA-gates path's collapse.
    post_mix = (1.0 / (1.0 + jnp.exp(-post_logits))) * gates.hc_post_mult_value
    comb_logits = scaled[:, 2 * m :] * sc_ref[0, 2] + hb_ref[:, 2 * m :]

    lt = comb_logits.T  # (M*M, tb), row k = i * M + j
    c = [[lt[i * m + j : i * m + j + 1, :] for j in range(m)] for i in range(m)]
    c = sinkhorn(c, m, eps, gates.sinkhorn_repeat)
    comb_mix = jnp.concatenate([c[i][j] for i in range(m) for j in range(m)], 0).T
    return pre_mix, post_mix, comb_mix


def sinkhorn(c, m: int, eps: float, sinkhorn_repeat: int):
    """Row softmax + Sinkhorn on c[i][j] (same-shape arrays), in the exact
    op order of ``mhc_pre_gates``' token-major in-kernel form."""
    for i in range(m):
        mx = c[i][0]
        for j in range(1, m):
            mx = jnp.maximum(mx, c[i][j])
        e = [jnp.exp(c[i][j] - mx) for j in range(m)]
        s = _sum4(e)
        c[i] = [e[j] / s + eps for j in range(m)]

    def col_norm(c):
        cols = [_sum4([c[i][j] for i in range(m)]) + eps for j in range(m)]
        return [[c[i][j] / cols[j] for j in range(m)] for i in range(m)]

    def row_norm(c):
        rows = [_sum4(c[i]) + eps for i in range(m)]
        return [[c[i][j] / rows[i] for j in range(m)] for i in range(m)]

    c = col_norm(c)
    for _ in range(sinkhorn_repeat - 1):
        c = col_norm(row_norm(c))
    return c


def finish_layer_input(lay: jax.Array, norm_w_ref, norm_eps: float | None) -> jax.Array:
    """The collapsed f32 layer input as the bf16 the sublayer consumes.

    With ``norm_eps`` it is also RMS-normalized (vLLM ``rms_norm``: bf16
    input, f32 statistics), so the sublayer's own input norm can be skipped.
    A bf16 weight follows rms_norm's ``x.to(weight.dtype) * weight`` (round,
    then multiply); an f32 weight multiplies the f32 normalized value.
    """
    out = lay.astype(jnp.bfloat16)
    if norm_eps is None:
        return out
    x = out.astype(jnp.float32)
    var = jnp.mean(x * x, axis=-1, keepdims=True)
    normed = x * jax.lax.rsqrt(var + norm_eps)
    w = norm_w_ref[...]
    if w.dtype == jnp.bfloat16:
        # vLLM rms_norm: ``x.to(weight.dtype) * weight`` -- round to bf16,
        # then a bf16 multiply. The f32 product of two bf16 values is exact,
        # so one more rounding reproduces the bf16 multiply bit for bit.
        # torch_tpu keeps that intermediate rounding (measured: dropping it
        # moves 25% of elements by 1 ulp vs the served XLA rms_norm).
        normed = normed.astype(jnp.bfloat16).astype(jnp.float32)
    return (normed * w.astype(jnp.float32)).astype(jnp.bfloat16)
