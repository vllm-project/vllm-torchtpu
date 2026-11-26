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


def _select_kernel(x_ref, *refs, topk: int, n: int, pick: bool, lanes_are_rows: bool):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    Scores at or below ``NEG`` are clamped up to it first, so every row has a
    maximum some column equals and every index written is a real expert. Ties
    go to the lowest id.

    The clamp tests ``x > NEG`` rather than ``isnan(x)`` to catch all three
    spellings of the hazard -- NaN, ``-inf``, ``-FLT_MAX`` -- any of which a
    whole row can be made of. Without it such a row would stay tied with the
    ``finfo.min`` mask and every pass would return column 0.

    With ``pick`` the first ref holds a second ``[rows, n]`` array, and the
    weights are its entries at the selected columns (``gather``) instead of
    the maxima.

    With ``lanes_are_rows`` the block is transposed first, so each pass
    reduces over sublanes (elementwise across vregs) instead of across the
    lanes of every vreg; every value is unchanged.
    """
    y_ref, w_ref, i_ref = refs if pick else (None, *refs)
    x = x_ref[...]
    y = y_ref[...] if pick else None
    axis = 1
    if lanes_are_rows:
        x = x.T
        y = y.T if pick else None
        axis = 0
    x = jnp.where(x > NEG, x, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, axis)
    picked = []
    for _ in range(topk):
        m = jnp.max(x, axis=axis, keepdims=True)
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=axis, keepdims=True)
        hit = iota == idx
        if pick:
            m = jnp.max(jnp.where(hit, y, -jnp.inf), axis=axis, keepdims=True)
        picked.append((m, idx))
        x = jnp.where(hit, masked, x)
    if lanes_are_rows:
        # Back to [rows, topk]: one lane-dense row per pass, transposed once.
        w_ref[...] = jnp.concatenate([m for m, _ in picked], axis=0).T
        i_ref[...] = jnp.concatenate([idx for _, idx in picked], axis=0).T
    else:
        for j, (m, idx) in enumerate(picked):
            w_ref[:, j] = m[:, 0]
            i_ref[:, j] = idx[:, 0]
    if not pick:
        # No score above the sentinel means no real maximum: keep the row
        # poisonous rather than letting renormalization make it plausible.
        w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def select(
    scores: jax.Array,
    topk: int,
    interpret: bool = False,
    values: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` f32 scores, values descending.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending. With ``values`` (f32, the shape of ``scores``) the weights are
    ``values`` gathered at the indices instead: a router that selects on
    biased scores and weighs by the unbiased ones. ``interpret`` runs the
    kernel on the host, for tests without a TPU.

    The last grid step reads a partial block when ``rows`` is not a multiple
    of the block; Mosaic discards the stores past ``rows``, and every
    reduction is along the expert axis, so surplus rows cannot perturb a real
    one.
    """
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    pick = values is not None
    spec = pl.BlockSpec((block, n), lambda i: (i, 0))
    return pl.pallas_call(
        functools.partial(
            _select_kernel,
            topk=topk,
            n=n,
            pick=pick,
            lanes_are_rows=block % 128 == 0,
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=[spec, spec] if pick else [spec],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(*((scores, values) if pick else (scores,)))


def router_topk(scores: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk)


def router_topk_pick(
    scores: jax.Array, values: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """``select`` with ``values``, jax_op-compatible (arrays, then statics)."""
    return select(scores, topk, values=values)
