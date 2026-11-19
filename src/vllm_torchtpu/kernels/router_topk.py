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
"""Sort-free top-k expert selection for the MoE router (Pallas).

``torch.topk`` lowers to a full ``stablehlo.sort`` of every expert plus a
slice, so at 512 experts it runs a 45-stage bitonic network over three arrays
to keep 10 values. This kernel selects the same 10 with ``k`` max-and-mask
passes over one VMEM-resident block: ``O(k*N)`` instead of ``O(N*log^2 N)``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

# Clamp floor. Below any real routing score (softmax, sigmoid and sqrtsoftplus
# are all >= 0) and strictly above the dtype minimum used to mask a selected
# column, so a row's maximum is always above the mask.
NEG = -3.0e38

# Rows per grid step, measured fastest at 512 experts. The grid rounds up, so
# this does not have to divide the row count.
MAX_BLOCK_ROWS = 512


def _select_kernel(x_ref, w_ref, i_ref, *, topk: int, n: int):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    Scores at or below ``NEG`` are clamped up to it first, so every row has a
    maximum some column equals and every index written is a real expert. Ties
    go to the lowest id.

    The clamp tests ``x > NEG`` rather than ``isnan(x)`` to catch all three
    spellings of the hazard -- NaN, ``-inf``, ``-FLT_MAX`` -- any of which a
    whole row can be made of. Without it such a row would stay tied with the
    ``finfo.min`` mask and every pass would return column 0.
    """
    x = jnp.where(x_ref[...] > NEG, x_ref[...], NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        w_ref[:, j] = m[:, 0]
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx[:, None], masked, x)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous rather than letting renormalization make it plausible.
    w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def _select_kernel_t(x_ref, w_ref, i_ref, *, topk: int, n: int):
    """`_select_kernel` on the block transposed once in VMEM: experts on
    sublanes, rows on lanes, so each pass's max and min reduce across vregs
    on the VPU (plus one sublane reduce) instead of across lanes on the XLU,
    and each pick is one dense row store of ``[topk, rows]``. Same picks,
    values and ties bit for bit."""
    x = x_ref[...].T  # [n, block]
    x = jnp.where(x > NEG, x, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    ws = []
    for j in range(topk):
        m = jnp.max(x, axis=0, keepdims=True)  # [1, block]
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=0, keepdims=True)
        ws.append(m)
        i_ref[j : j + 1, :] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx, masked, x)
    w = jnp.concatenate(ws, axis=0)
    w_ref[...] = jnp.where(w[:1] <= NEG, jnp.nan, w)


# Rows from which `select` takes the transposed kernel
# (TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS, 0 = never); rows must also be a
# multiple of 128 (whole lanes).
_logged_t = False


def _transposed_min_rows() -> int:
    from vllm_torchtpu import envs

    return int(envs.TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS)


def _takes_transposed(scores: jax.Array) -> bool:
    rows, n = scores.shape
    min_rows = _transposed_min_rows()
    return (
        0 < min_rows <= rows
        and rows % 128 == 0
        and n % 8 == 0
        and scores.dtype == jnp.float32
    )


def _select_t(scores: jax.Array, topk: int, interpret: bool = False):
    global _logged_t
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    if not _logged_t:
        _logged_t = True
        print(
            "TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS=1: router top-k by the "
            f"transposed kernel ({rows} rows, {n} experts)",
            flush=True,
        )
    w, i = pl.pallas_call(
        functools.partial(_select_kernel_t, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, n), lambda b: (b, 0))],
        out_specs=[
            pl.BlockSpec((topk, block), lambda b: (0, b)),
            pl.BlockSpec((topk, block), lambda b: (0, b)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, rows), jnp.float32),
            jax.ShapeDtypeStruct((topk, rows), jnp.int32),
        ],
        interpret=interpret,
    )(scores)
    return w.T, i.T


def _select_biased_kernel_t(x_ref, b_ref, w_ref, i_ref, *, topk: int, n: int):
    """`_select_kernel_t` choosing by ``scores + bias`` and returning the
    unbiased scores of the picks (the router's ``scores.gather(ids)``):
    each pick's score is the one-hot sum of its column, exact."""
    s = x_ref[...].T  # [n, block] unbiased scores
    x = s + b_ref[...]  # [n, 1] bias broadcast along lanes
    x = jnp.where(x > NEG, x, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0)
    for j in range(topk):
        m = jnp.max(x, axis=0, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=0, keepdims=True)
        hit = iota == idx
        # One hit per lane: the sum is the picked score itself (NaN and
        # -inf included, which a max reduction may not propagate).
        w_ref[j : j + 1, :] = jnp.sum(jnp.where(hit, s, 0.0), axis=0, keepdims=True)
        i_ref[j : j + 1, :] = idx.astype(jnp.int32)
        x = jnp.where(hit, masked, x)


def select_biased(
    scores: jax.Array, bias: jax.Array, topk: int, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Top-k of ``scores + bias`` over ``[rows, experts]`` f32 scores, returning
    ``(scores at the picks, picks)`` -- the router's biased choice and its
    gather in one pass (transposed kernel; rows a multiple of 128). Elsewhere
    the same as `select` on the biased scores plus a gather."""
    rows, n = scores.shape
    bias = bias.astype(jnp.float32).reshape(n)
    if rows % 128 or n % 8 or scores.dtype != jnp.float32:
        _, ids = select(scores + bias[None, :], topk, interpret)
        return jnp.take_along_axis(scores, ids, axis=1), ids
    block = min(MAX_BLOCK_ROWS, rows)
    w, i = pl.pallas_call(
        functools.partial(_select_biased_kernel_t, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, n), lambda b: (b, 0)),
            pl.BlockSpec((n, 1), lambda b: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((topk, block), lambda b: (0, b)),
            pl.BlockSpec((topk, block), lambda b: (0, b)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, rows), jnp.float32),
            jax.ShapeDtypeStruct((topk, rows), jnp.int32),
        ],
        interpret=interpret,
    )(scores, bias.reshape(n, 1))
    return w.T, i.T


def router_topk_biased(
    scores: jax.Array, bias: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """``select_biased`` with a jax_op-compatible signature."""
    global _logged_biased
    if not _logged_biased:
        _logged_biased = True
        print(
            "TPU_ROUTER_TOPK_FUSED_BIAS_MIN_ROWS=1: router biased top-k and "
            f"weight gather in one kernel ({scores.shape[0]} rows)",
            flush=True,
        )
    return select_biased(scores, bias, topk)


_logged_biased = False


def select(
    scores: jax.Array, topk: int, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` f32 scores, values descending.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending. ``interpret`` runs the kernel on the host, for tests without a
    TPU.

    The last grid step reads a partial block when ``rows`` is not a multiple
    of the block; Mosaic discards the stores past ``rows``, and every
    reduction is along the expert axis, so surplus rows cannot perturb a real
    one.
    """
    rows, n = scores.shape
    if _takes_transposed(scores):
        return _select_t(scores, topk, interpret)
    block = min(MAX_BLOCK_ROWS, rows)
    return pl.pallas_call(
        functools.partial(_select_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((block, n), lambda i: (i, 0))],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(scores)


def router_topk(scores: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk)


def _select_values_kernel(x_ref, v_ref, w_ref, i_ref, *, topk: int, n: int):
    """`_select_kernel` on ``x``, writing ``v`` at the picked columns instead
    of ``x``'s maxima: one row's ``v`` at the picked column is the only value
    the masked max can return, so the weights are ``v``'s bits. Unlike
    `_select_kernel`, rows with no real maximum are not poisoned: the caller
    used to gather ``v`` with the ids, which never yields NaN either."""
    x = jnp.where(x_ref[...] > NEG, x_ref[...], NEG)
    v = v_ref[...]
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        hit = iota == idx[:, None]
        w_ref[:, j] = jnp.max(jnp.where(hit, v, -jnp.inf), axis=1)
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(hit, masked, x)


def select_values(
    scores: jax.Array, values: jax.Array, topk: int, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Top-k columns of ``[rows, experts]`` f32 ``scores`` (as `select`: values
    descending, ties to the lowest id), returning ``values`` at those columns
    and the column ids: ``(values.gather(-1, ids), ids)`` in one kernel.

    Calls the transposed select qualifies for
    (TPU_ROUTER_TOPK_TRANSPOSED_MIN_ROWS) take it and gather ``values``
    after, so both flags keep their large-call behaviour when combined."""
    if _takes_transposed(scores):
        _, ids = _select_t(scores, topk, interpret)
        return jnp.take_along_axis(values, ids, axis=1), ids
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    spec = pl.BlockSpec((block, n), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(_select_values_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[spec, spec],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
        name="router_topk_values",
    )(scores, values)


def router_topk_values(
    scores: jax.Array, values: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """``select_values`` with a jax_op-compatible signature."""
    return select_values(scores, values, topk)
