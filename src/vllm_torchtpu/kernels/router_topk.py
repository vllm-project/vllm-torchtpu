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


def _select_fused_kernel(
    s_ref, b_ref, w_ref, i_ref, *, topk: int, n: int, renormalize: bool, scale: float
):
    """One block of rows: biased top-k, then the routing-weight epilogue.

    The kernel selects experts on ``scores + bias`` but returns the unbiased
    ``scores`` at the selected columns. This is the DeepSeek-V3 style
    correction bias, which GLM-5.2 uses. The epilogue then applies the same
    float32 operations, in the same order, that ``moe_routing.select_experts``
    applies after an unfused top-k: divide by ``max(sum, 1e-20)`` when
    ``renormalize`` is set, then multiply by ``scale``. The selection rules
    (clamp at ``NEG``, ties to the lowest id) are those of ``_select_kernel``.

    The gather of the unbiased score is a masked row sum: exactly one column
    of ``hit`` is true, and adding the other columns' zeros does not change
    a float32 value, so the result equals the score bit for bit.

    The renormalization row sum is a plain ``jnp.sum``, as in the unfused
    path. Float32 addition is not associative, and the order in which a
    backend adds the ``topk`` values is its own choice: XLA:CPU, for example,
    adds an 8-wide row sequentially for small row counts and in halving order
    for large ones. So the weights equal the unfused path's bit for bit only
    where both sums use the same order, and otherwise differ by at most one
    float32 rounding of the sum. The indices do not depend on the sum.
    """
    s = s_ref[...]
    biased = s + b_ref[...]
    x = jnp.where(biased > NEG, biased, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        hit = iota == idx[:, None]
        w_ref[:, j] = jnp.sum(jnp.where(hit, s, 0.0), axis=1)
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(hit, masked, x)
    w = w_ref[...]
    if renormalize:
        w = w / jnp.maximum(jnp.sum(w, axis=1, keepdims=True), 1e-20)
    if scale != 1.0:
        w = w * scale
    w_ref[...] = w


def select_fused(
    scores: jax.Array,
    bias: jax.Array,
    topk: int,
    renormalize: bool,
    scale: float,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Biased top-k plus the routing-weight epilogue in one kernel call.

    Without this kernel the router runs the top-k kernel on ``scores + bias``
    and then four small XLA ops: the bias add before the kernel, and the
    gather, the renormalization and the scale after it. At a 16-token decode
    step each of those ops costs a fixed launch time that is larger than its
    work, so one kernel call replaces all of them.

    Args:
      scores: ``[rows, experts]`` float32 routing scores after the scoring
        function (for example sigmoid).
      bias: ``[experts]`` float32 expert-selection correction bias. It changes
        which experts are selected, not the returned weights.
      topk: number of experts to select per row.
      renormalize: divide the selected weights by their row sum, clamped
        below at ``1e-20``.
      scale: routed scaling factor that multiplies the final weights. ``1.0``
        skips the multiply.
      interpret: run the kernel on the host, for tests without a TPU.

    Returns:
      ``(weights f32, indices int32)``, both ``[rows, topk]``. The indices are
      in descending order of the biased score. For example, with scores
      ``[0.1, 0.4, 0.3]``, bias ``[0.5, 0.0, 0.0]``, ``topk=2``,
      ``renormalize=True`` and ``scale=2.0``, the biased scores are
      ``[0.6, 0.4, 0.3]``, the indices are ``[0, 1]``, and the weights are
      ``[0.1, 0.4] / 0.5 * 2.0 = [0.4, 1.6]``.
    """
    rows, n = scores.shape
    block = min(MAX_BLOCK_ROWS, rows)
    return pl.pallas_call(
        functools.partial(
            _select_fused_kernel,
            topk=topk,
            n=n,
            renormalize=renormalize,
            scale=scale,
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, n), lambda i: (i, 0)),
            pl.BlockSpec((1, n), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        interpret=interpret,
    )(scores, bias.reshape(1, n))


def router_topk_fused(
    scores: jax.Array, bias: jax.Array, topk: int, renormalize: bool, scale: float
) -> tuple[jax.Array, jax.Array]:
    """``select_fused`` with a jax_op-compatible signature (arrays, then statics)."""
    return select_fused(scores, bias, topk, renormalize, scale)
