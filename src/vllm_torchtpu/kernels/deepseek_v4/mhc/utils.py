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

import functools
from collections.abc import Callable
from typing import cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

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


def add_scaled_bf16(x: jax.Array, addend: jax.Array, scale: float) -> jax.Array:
    """`x + addend * scale` of bf16 operands, the scaled addend and the sum
    each rounded to bf16 as the bf16 expression is."""
    f32, bf16 = jnp.float32, jnp.bfloat16
    scaled = (addend.astype(f32) * scale).astype(bf16)
    return (x.astype(f32) + scaled.astype(f32)).astype(bf16)


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
    return cast(
        tuple[jax.Array, jax.Array, jax.Array],
        tuple(t.astype(jnp.bfloat16) for t in (fn_hi, fn_mid, fn_lo)),
    )


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

    The XLA reference; the kernels run ``post_comb_gates``, its exact twin.
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


# In-kernel twin of mhc_pre_gates' post and comb halves. It reproduces, op
# for op, the program XLA compiles from mhc_pre_gates on TPU, so the gates
# keep that path's exact bits:
# - XLA folds the Sinkhorn's chained divisions x/a/b into x/(a*b), so every
#   step divides the first comb_mix by the running product of the
#   denominators, multiplied in step order;
# - x/y lowers to x * reciprocal(y), the same in Mosaic and XLA;
# - in XLA's (T, M, M) layout a sum over the last axis is a sublane tree,
#   (a0+a2)+(a1+a3) for M=4, and a sum over the middle axis is a left fold;
# - the sigmoid is XLA's expansion 1/(1+exp(-x)), not Mosaic's logistic.


def gate_logits(scaled: jax.Array, sc_ref, hb_ref, hc_mult: int) -> jax.Array:
    """(tb, hc_mult3) gate logits from RMS-scaled mixes: each block of
    columns (pre, post, comb) times its scale, plus hc_base."""
    lane = jax.lax.broadcasted_iota(jnp.int32, (1, scaled.shape[1]), 1)
    scale = jnp.where(
        lane < hc_mult,
        sc_ref[0, 0],
        jnp.where(lane < 2 * hc_mult, sc_ref[0, 1], sc_ref[0, 2]),
    )
    return scaled * scale + hb_ref[...]


def _last_axis_sum(terms: list[jax.Array]) -> jax.Array:
    assert len(terms) == 4, "the reduction tree is spelled out for hc_mult=4"
    return (terms[0] + terms[2]) + (terms[1] + terms[3])


def _middle_axis_sum(rows: jax.Array) -> jax.Array:
    total = rows[0:1]
    for i in range(1, rows.shape[0]):
        total = total + rows[i : i + 1]
    return total


def post_comb_gates(
    logits_t_ref,
    gates_t_ref,
    *,
    hc_mult: int,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """post_mix and comb_mix for every token at once, tokens on lanes.

    logits_t_ref: (hc_mult3, T) f32 gate logits, one row per logit.
    gates_t_ref: (hc_mult + hc_mult**2, T) f32 out: post_mix's hc_mult
        rows, then comb_mix's rows in row-major (i, j) order.
    """
    m = hc_mult
    z = logits_t_ref[pl.ds(m, m), :]
    gates_t_ref[pl.ds(0, m), :] = (1.0 / (1.0 + jnp.exp(-z))) * hc_post_mult_value

    # comb logits as one (i, T) array per column j.
    def rows(first):
        return [logits_t_ref[pl.ds(first + k * m, 1), :] for k in range(m)]

    logits = [jnp.concatenate(rows(2 * m + j), axis=0) for j in range(m)]
    row_max = functools.reduce(jnp.maximum, logits)
    exps = [jnp.exp(lg - row_max) for lg in logits]
    exp_sum = _last_axis_sum(exps)
    comb0 = [e / exp_sum + hc_sinkhorn_eps for e in exps]

    col_den = [_middle_axis_sum(c) + hc_sinkhorn_eps for c in comb0]
    den = [jnp.broadcast_to(d, c.shape) for d, c in zip(col_den, comb0)]
    for _ in range(sinkhorn_repeat - 1):
        comb = [c / d for c, d in zip(comb0, den)]
        row_den = _last_axis_sum(comb) + hc_sinkhorn_eps
        den = [d * row_den for d in den]
        comb = [c / d for c, d in zip(comb0, den)]
        den = [d * (_middle_axis_sum(c) + hc_sinkhorn_eps) for d, c in zip(den, comb)]
    for j in range(m):
        comb = comb0[j] / den[j]
        for i in range(m):
            gates_t_ref[pl.ds(m + i * m + j, 1), :] = comb[i : i + 1]


def post_comb_gates_epilogue(
    scaled: jax.Array,
    sc_ref,
    hb_ref,
    post_ref,
    comb_ref,
    logits_ref,
    logits_t_ref,
    gates_t_ref,
    *,
    hc_mult: int,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
) -> None:
    """Gate tail of a token-blocked mHC kernel on a sequential grid.

    Every block stores its gate logits; the last block runs the gates for
    all tokens at once, so the 2 * sinkhorn_repeat dependent steps are
    paid once per call, with every token's lanes busy.

    scaled: (tb, hc_mult3) f32 RMS-scaled mixes of this block.
    post_ref, comb_ref: (T, hc_mult), (T, hc_mult**2) f32 whole-array outs.
    logits_ref, logits_t_ref, gates_t_ref: f32 VMEM scratch of shapes
        (T, hc_mult3), (hc_mult3, T), (hc_mult + hc_mult**2, T).
    """
    step, tb = pl.program_id(0), scaled.shape[0]
    rows = pl.ds(pl.multiple_of(step * tb, tb), tb)
    logits_ref[rows, :] = gate_logits(scaled, sc_ref, hb_ref, hc_mult)

    @pl.when(step == pl.num_programs(0) - 1)
    def _gates():
        logits_t_ref[...] = logits_ref[...].T
        post_comb_gates(
            logits_t_ref,
            gates_t_ref,
            hc_mult=hc_mult,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )
        gates = gates_t_ref[...].T
        post_ref[...] = gates[:, :hc_mult]
        comb_ref[...] = gates[:, hc_mult:]


def post_comb_gates_specs(
    padded_tokens: int, hc_mult: int
) -> tuple[list[pl.BlockSpec], list[jax.ShapeDtypeStruct], list]:
    """(out_specs, out_shape, scratch_shapes) entries that
    ``post_comb_gates_epilogue`` writes and uses."""
    hc_mult3 = hc_mult * (hc_mult + 2)
    widths = (hc_mult, hc_mult * hc_mult)
    out_specs = [pl.BlockSpec((padded_tokens, w), lambda i: (0, 0)) for w in widths]
    out_shape = [jax.ShapeDtypeStruct((padded_tokens, w), jnp.float32) for w in widths]
    scratch_shapes = [
        pltpu.VMEM((padded_tokens, hc_mult3), jnp.float32),
        pltpu.VMEM((hc_mult3, padded_tokens), jnp.float32),
        pltpu.VMEM((sum(widths), padded_tokens), jnp.float32),
    ]
    return out_specs, out_shape, scratch_shapes
