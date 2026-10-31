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


def without_shared(kernel: Callable) -> Callable:
    """A kernel body taking ``(x_ref, shared_ref, ...)``, for a call that
    passes no shared-expert block: ``shared_ref`` is None."""

    def body(x_ref, *refs):
        return kernel(x_ref, None, *refs)

    return body


def without_ref(kernel: Callable, index: int) -> Callable:
    """A kernel body for a call that omits the optional ref at ``index``
    (counted with every optional ref present): it gets None there."""

    def body(*refs):
        return kernel(*refs[:index], None, *refs[index:])

    return body


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


def rms_norm_rows(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    """RMSNorm of a (tb, hidden) bf16 block inside a kernel, bit for bit the
    XLA fusions torch's RMSNorm compiles to on TPU.

    XLA reduces each row's squares in a fixed order: the 128-lane column
    tiles in sequence, then (after a transpose) the 16 eight-lane groups in
    sequence, then a sublane rotate-add tree by 4, 2, 1; the rsqrt is one
    instruction. The normalized row is rounded to bf16 before the weight
    multiply, as vLLM's rms_norm writes it: torch_tpu compiles with XLA's
    excess precision off, so XLA keeps that rounding. This writes that order
    out explicitly.
    ``weight`` is (hidden,) bf16; tb must be a multiple of 8.
    """
    lanes, sublanes = 128, 8
    xf = x.astype(jnp.float32)
    sq = xf * xf
    acc = sq[:, :lanes]
    for j in range(1, xf.shape[1] // lanes):
        acc = acc + sq[:, j * lanes : (j + 1) * lanes]
    acc_t = acc.T  # (lanes, tb): rows on lanes
    tot = acc_t[:sublanes]
    for k in range(1, lanes // sublanes):
        tot = tot + acc_t[k * sublanes : (k + 1) * sublanes]
    # Mosaic's roll by s rotates the sublanes the other way from XLA's
    # rotate by s, so these shifts emit XLA's rotates by 4, 2, 1.
    for shift in (4, 6, 7):
        tot = pltpu.roll(tot, shift, 0) + tot
    inv = jax.lax.rsqrt(tot[:1] * (1.0 / xf.shape[1]) + eps)  # (1, tb)
    inv_col = jnp.broadcast_to(inv, (sublanes, inv.shape[1])).T[:, :1]  # (tb, 1)
    normed = (xf * inv_col).astype(jnp.bfloat16).astype(jnp.float32)
    return (normed * weight.astype(jnp.float32)).astype(jnp.bfloat16)


def hc_mult_of(hc_mult3: int) -> int:
    """hc_mult from the mix width hc_mult3 = hc_mult * (hc_mult + 2)."""
    return next(m for m in range(1, 64) if m * (m + 2) == hc_mult3)


def fold_fn_streams(fn: jax.Array) -> jax.Array:
    """fn (hc_mult3, hc_mult * hidden) f32 summed over its hc_mult stream
    slices, in stream order: (hc_mult3, hidden), the projection hc_mult
    identical streams see as one (``mhc_pre_single_stream``)."""
    hc_mult = hc_mult_of(fn.shape[0])
    slices = jnp.split(fn, hc_mult, axis=1)
    folded = slices[0]
    for s in slices[1:]:
        folded = folded + s
    return folded


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


def fn_chunk_operands(fn: jax.Array) -> tuple[list[jax.Array], list[pl.BlockSpec]]:
    """fn's three bf16 chunks as VMEM-resident operands of a 1-D grid.

    ``fn`` is the f32 (hc_mult3, K) projection, split here by
    ``split_fn3``, or its chunks already split that way, (3, hc_mult3, K)
    bf16, which are passed as they are. Returns (operands, block specs).
    """
    if fn.ndim == 2:
        assert fn.dtype == jnp.float32, fn.dtype
        resident = pl.BlockSpec(fn.shape, lambda i: (0, 0))
        return list(split_fn3(fn)), [resident] * 3
    assert fn.dtype == jnp.bfloat16 and fn.shape[0] == 3, (fn.dtype, fn.shape)
    specs = [
        pl.BlockSpec((None, *fn.shape[1:]), lambda i, c=c: (c, 0, 0)) for c in range(3)
    ]
    return [fn] * 3, specs


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
    # Keep the consumer's row-major relayout out of the final divide: fused
    # into it, XLA copies every broadcast Sinkhorn normalizer to that layout.
    comb_mix = jax.lax.optimization_barrier(comb_mix)
    return pre_mix, post_mix, comb_mix
