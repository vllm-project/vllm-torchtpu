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
import logging
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# A stdlib logger under vLLM's handlers, so the kernel still imports and runs
# where vLLM is not installed.
logger = logging.getLogger(f"vllm.{__name__}")
_traced_shapes: set[tuple[int, int, int]] = set()

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


def _traced_from() -> str:
    """The vllm_torchtpu frames that are tracing this kernel, innermost first."""
    here = Path(__file__).resolve()
    frames = [
        f
        for f in traceback.extract_stack()
        if "vllm_torchtpu" in f.filename and Path(f.filename).resolve() != here
    ]
    return " <- ".join(f"{Path(f.filename).stem}.{f.name}" for f in reversed(frames))


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
    if (rows, n, topk) not in _traced_shapes:
        _traced_shapes.add((rows, n, topk))
        logger.info(
            "router_topk: Pallas top-%d over [%d, %d] scores, traced from %s",
            topk,
            rows,
            n,
            _traced_from(),
        )
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


# Tokens per grid step of the fused kernel; the lane axis of its blocks.
MAX_BLOCK_TOKENS = 512

# Experts per sweep step of the fused kernel: one bf16 sublane tile.
SWEEP_ROWS = 16

# fori_loop unroll of the fused kernel's sweep over the experts.
SWEEP_UNROLL = 4


def _softmax_select_kernel(x_ref, w_ref, i_ref, s_ref, *, topk: int, n: int):
    """One block of tokens, expert-major: ``x_ref`` is ``[experts, tokens]``.

    Each of the ``topk`` passes is one sweep over the experts in steps of
    ``SWEEP_ROWS``, keeping per sublane the running maximum and the first
    step that attained it, so the arg-max needs no second pass and no
    cross-lane reduction; only the last ``SWEEP_ROWS``-row max/min crosses
    sublanes. Strict ``>`` over ascending steps plus the lowest id among tied
    sublanes keeps ``_select_kernel``'s rule: ties go to the lowest expert.
    Pass ``j`` masks pass ``j - 1``'s pick while it sweeps; the clamped f32
    scores live in ``s_ref`` between passes.
    """
    tb = x_ref.shape[1]
    masked = jnp.finfo(jnp.float32).min
    sub = jax.lax.broadcasted_iota(jnp.int32, (SWEEP_ROWS, tb), 0)
    idx_rel = None
    for j in range(topk):

        def step(v, carry, j=j, idx_rel=idx_rel):
            best, bv = carry
            r = pl.multiple_of(v * SWEEP_ROWS, SWEEP_ROWS)
            if j == 0:
                x = x_ref[pl.ds(r, SWEEP_ROWS), :].astype(jnp.float32)
                x = jnp.where(x > NEG, x, NEG)
            else:
                x = s_ref[pl.ds(r, SWEEP_ROWS), :]
                x = jnp.where(idx_rel == r, masked, x)
            if j < topk - 1:
                s_ref[pl.ds(r, SWEEP_ROWS), :] = x
            return jnp.maximum(best, x), jnp.where(x > best, v, bv)

        best, bv = jax.lax.fori_loop(
            0,
            n // SWEEP_ROWS,
            step,
            (
                jnp.full((SWEEP_ROWS, tb), masked, jnp.float32),
                jnp.zeros((SWEEP_ROWS, tb), jnp.int32),
            ),
            unroll=SWEEP_UNROLL,
        )
        m = jnp.max(best, axis=0, keepdims=True)
        idx = jnp.min(
            jnp.where(best == m, bv * SWEEP_ROWS + sub, n), axis=0, keepdims=True
        )
        w_ref[pl.ds(j, 1), :] = m
        i_ref[pl.ds(j, 1), :] = idx
        idx_rel = idx - sub
    # Softmax over the k selected logits: the full-row denominator cancels in
    # the renormalization. A row with no logit above NEG stays NaN.
    w = w_ref[...]
    e = jnp.exp(w - w[:1])
    p = e / jnp.sum(e, axis=0, keepdims=True)
    w_ref[...] = jnp.where(w[:1] <= NEG, jnp.nan, p)


def select_softmax(
    logits: jax.Array, topk: int, interpret: bool = False
) -> tuple[jax.Array, jax.Array]:
    """Top-k experts of softmax(``logits``), weights renormalized over the k.

    Softmax is monotonic, so the experts are selected on the ``[rows,
    experts]`` logits (any float dtype) and only the k selected are
    exponentiated: the full-width softmax and its f32 copy of the scores are
    never materialized. Returns ``(weights f32, indices int32)`` of shape
    ``[rows, topk]``, weights descending and summing to 1.

    The kernel reads the logits expert-major, the layout the router matmul
    already writes them in, so the transpose here is a layout choice for XLA
    rather than a copy.
    """
    rows, n = logits.shape
    if n % SWEEP_ROWS:
        raise ValueError(f"expert count {n} is not a multiple of {SWEEP_ROWS}")
    block = min(MAX_BLOCK_TOKENS, rows)
    w, i = pl.pallas_call(
        functools.partial(_softmax_select_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[pl.BlockSpec((n, block), lambda t: (0, t))],
        out_specs=[
            pl.BlockSpec((topk, block), lambda t: (0, t)),
            pl.BlockSpec((topk, block), lambda t: (0, t)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, rows), jnp.float32),
            jax.ShapeDtypeStruct((topk, rows), jnp.int32),
        ],
        scratch_shapes=[pltpu.VMEM((n, block), jnp.float32)],
        interpret=interpret,
    )(logits.T)
    return w.T, i.T


def router_topk_softmax(logits: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select_softmax`` with a jax_op-compatible signature."""
    return select_softmax(logits, topk)
