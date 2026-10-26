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
"""Top-k expert selection for the fused expert-parallel MoE router:
pallas_select is an in-VMEM max-and-mask pass replacing XLA's lax.top_k.

An all-NaN score row selects the lowest expert on every slot, with NEG as
the weight of each: NaN is flushed to the sentinel before the first max, so
every row has a maximum some column equals and every index returned is a
real expert. What that row's weights MEAN is not decided here. The kernel
layer masks such a row explicitly, on jnp.any(jnp.isfinite(scores)), rather
than relying on the sentinels to sum to something the renormalization turns
into zero -- that outcome was a property of the accumulation dtype and the
association order rather than of the design.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

NEG = -3.0e38  # below any real softmax score (>=0); avoids inf


def _select_kernel(x_ref, w_ref, i_ref, *, topk, n, v_ref=None, l_ref=None):
    """One block of rows: take the top `topk` by repeated max-and-mask."""
    _select(
        x_ref[...],
        None if v_ref is None else v_ref[...],
        None if l_ref is None else l_ref[...],
        w_ref,
        i_ref,
        topk=topk,
        n=n,
    )


def _select(x, v, labels, w_ref, i_ref, *, topk, n):
    """_select_kernel on values: x [R, n] f32, v the weight scores or None,
    labels [1, n] i32 or None."""
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    x = jnp.where(jnp.isnan(x), NEG, x)  # [R, n] f32
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)  # [R,1] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        idx = jnp.min(jnp.where(ismax, iota, n - 1), axis=1)
        picked = iota == idx[:, None]
        # noaux selects with biased scores but emits unbiased weights.
        # A masked sum reads the resident tile without a dynamic lane gather.
        w_ref[:, j] = (
            m[:, 0] if v is None else jnp.sum(jnp.where(picked, v, 0.0), axis=1)
        )
        # Exactly one column is picked, so the masked sum IS its label.
        i_ref[:, j] = (
            idx.astype(jnp.int32)
            if labels is None
            else jnp.sum(jnp.where(picked, labels, 0), axis=1)
        )
        x = jnp.where(picked, NEG, x)


def _select_kernel_split(x_ref, v_ref, w_ref, i_ref, *, topk, n):
    """Select by x_ref and read the selected weights from v_ref.

    Pallas passes input refs before output refs, so the two-input call needs
    this signature. Forward v_ref as a keyword to reuse _select_kernel's
    selection loop while keeping its one-input call unchanged.
    """
    _select_kernel(x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref)


def _select_kernel_labelled(*refs, topk, n, has_weights):
    """Unpack (x, [v], labels, w, i): Pallas passes inputs before outputs."""
    it = iter(refs)
    x_ref = next(it)
    v_ref = next(it) if has_weights else None
    l_ref, w_ref, i_ref = next(it), next(it), next(it)
    _select_kernel(x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref, l_ref=l_ref)


def _select_routed_kernel(*refs, topk, n, has_bias, has_labels):
    """(scores, gate, [bias], [labels]) -> (w, i, routes): the selection on
    scores + bias with weights from the scores, plus the row-routes mask."""
    it = iter(refs)
    s_ref, g_ref = next(it), next(it)
    b_ref = next(it) if has_bias else None
    l_ref = next(it) if has_labels else None
    w_ref, i_ref, r_ref = next(it), next(it), next(it)
    scores = s_ref[...]

    def any_finite(a):
        # isfinite as a compare: NaN and +-inf both fail it.
        finite = (jnp.abs(a) < jnp.inf).astype(jnp.int32)
        return jnp.max(finite, axis=1, keepdims=True)

    r_ref[...] = any_finite(scores) * any_finite(g_ref[...])
    _select(
        scores if b_ref is None else scores + b_ref[...],
        None if b_ref is None else scores,
        None if l_ref is None else l_ref[...],
        w_ref,
        i_ref,
        topk=topk,
        n=n,
    )


def pallas_select_routed(
    scores, gate, *, topk, block_rows, score_bias=None, labels=None, interpret=False
):
    """pallas_select on ``scores + score_bias`` with the weights read from
    ``scores``, which also returns the layer's row-routes mask [R, 1] bool:
    a row routes iff some score and some gate logit of it is finite.

    The same selection, weights and mask the layer otherwise builds from three
    more XLA ops (the bias add and two finiteness reductions); f32 addition and
    the finiteness test are exact, so every output is bit-identical.
    """
    R, N = scores.shape
    assert R % block_rows == 0, (R, block_rows)
    if gate.shape != (R, N):
        raise ValueError(f"gate {gate.shape} must match scores {scores.shape}")
    row_spec = pl.BlockSpec((block_rows, N), lambda i: (i, 0))
    vec_spec = pl.BlockSpec((1, N), lambda i: (0, 0))
    operands, specs = [scores, gate.astype(jnp.float32)], [row_spec, row_spec]
    if score_bias is not None:
        operands.append(score_bias.astype(jnp.float32).reshape(1, N))
        specs.append(vec_spec)
    if labels is not None:
        operands.append(labels.astype(jnp.int32).reshape(1, N))
        specs.append(vec_spec)
    kernel = functools.partial(
        _select_routed_kernel,
        topk=topk,
        n=N,
        has_bias=score_bias is not None,
        has_labels=labels is not None,
    )
    w, i, routes = pl.pallas_call(
        kernel,
        grid=(R // block_rows,),
        in_specs=specs,
        out_specs=[
            pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, 1), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((R, topk), jnp.float32),
            jax.ShapeDtypeStruct((R, topk), jnp.int32),
            jax.ShapeDtypeStruct((R, 1), jnp.int32),
        ],
        interpret=interpret,
    )(*operands)
    return w, i, routes != 0


def pallas_select(
    scores,
    topk=10,
    block_rows=256,
    weight_scores=None,
    labels=None,
    interpret=False,
):
    """Top-k over [R, N] scores -> (weights f32, indices i32) [R, topk].

    With `weight_scores=None`, return the selected scores as weights.
    Otherwise, select experts by `scores` and read their weights from the
    separate [R, N] `weight_scores` table. DeepSeek-V4's `noaux` uses this
    to select with `scores=raw_scores + correction_bias` while returning
    unbiased weights from `weight_scores=raw_scores`. Here `raw_scores` are
    the scores after the scoring function, before adding the bias.
    Normalization and routed scaling are applied by the caller.

    `labels` [N] i32 relabels the returned indices: slot j holds
    labels[selected column]. Selection, tie-breaking and the weights are
    those of the unlabelled call; an expert placement passes its slot table.

    `interpret` runs the kernel body on the host, which is what lets the
    selection rule be tested without a chip; it does not gate Mosaic
    lowering. Serving never sets it.
    """
    R, N = scores.shape
    assert R % block_rows == 0, (R, block_rows)
    if weight_scores is not None and weight_scores.shape != scores.shape:
        raise ValueError(
            f"weight_scores {weight_scores.shape} must match scores {scores.shape}"
        )
    grid = (R // block_rows,)
    out_shapes = (
        jax.ShapeDtypeStruct((R, topk), jnp.float32),
        jax.ShapeDtypeStruct((R, topk), jnp.int32),
    )
    if labels is not None and labels.shape != (N,):
        raise ValueError(f"labels {labels.shape} must be one per column, ({N},)")
    in_spec = pl.BlockSpec((block_rows, N), lambda i: (i, 0))
    label_spec = pl.BlockSpec((1, N), lambda i: (0, 0))
    os = (
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
    )
    operands, bs = [scores], [in_spec]
    if weight_scores is not None:
        operands.append(weight_scores)
        bs.append(in_spec)
    if labels is not None:
        operands.append(labels.astype(jnp.int32).reshape(1, N))
        bs.append(label_spec)
        kernel = functools.partial(
            _select_kernel_labelled, has_weights=weight_scores is not None
        )
    else:
        kernel = _select_kernel if weight_scores is None else _select_kernel_split
    return pl.pallas_call(
        functools.partial(kernel, topk=topk, n=N),
        grid=grid,
        in_specs=bs,
        out_specs=list(os),
        out_shape=list(out_shapes),
        interpret=interpret,
    )(*operands)
