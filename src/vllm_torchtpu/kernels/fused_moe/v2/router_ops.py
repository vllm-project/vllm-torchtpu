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


def _select_kernel(x_ref, w_ref, i_ref, *, topk, n, v_ref=None, m_ref=None):
    """One block of rows: take the top `topk` by repeated max-and-mask.
    With `m_ref` ([1, n] int32, entries < n) each pick emits m_ref[column]
    instead of its column: the lowest-column reduction runs on keys
    (column << lb) | m_ref[column], whose high bits rank the columns as before
    and whose low bits are the entry, so no reduction is added."""
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    x = jnp.where(jnp.isnan(x_ref[...]), NEG, x_ref[...])  # [R, n] f32
    v = None if v_ref is None else v_ref[...]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    lb = max(1, (n - 1).bit_length())
    key = None if m_ref is None else jnp.left_shift(iota, lb) | m_ref[...]
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)  # [R,1] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        if key is None:
            idx = out = jnp.min(jnp.where(ismax, iota, n - 1), axis=1)
        else:
            k = jnp.min(jnp.where(ismax, key, key[:, n - 1 :]), axis=1)
            idx, out = jnp.right_shift(k, lb), jnp.bitwise_and(k, (1 << lb) - 1)
        picked = iota == idx[:, None]
        # noaux selects with biased scores but emits unbiased weights.
        # A masked sum reads the resident tile without a dynamic lane gather.
        w_ref[:, j] = (
            m[:, 0] if v is None else jnp.sum(jnp.where(picked, v, 0.0), axis=1)
        )
        i_ref[:, j] = out.astype(jnp.int32)
        x = jnp.where(picked, NEG, x)


def _select_kernel_split(x_ref, v_ref, w_ref, i_ref, *, topk, n):
    """Select by x_ref and read the selected weights from v_ref.

    Pallas passes input refs before output refs, so the two-input call needs
    this signature. Forward v_ref as a keyword to reuse _select_kernel's
    selection loop while keeping its one-input call unchanged.
    """
    _select_kernel(x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref)


def _select_kernel_mapped(*refs, topk, n, split):
    """Inputs x [, v], index map, then the outputs w, i."""
    x_ref, *rest = refs
    v_ref = rest.pop(0) if split else None
    m_ref, w_ref, i_ref = rest
    _select_kernel(x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref, m_ref=m_ref)


def pallas_select(
    scores,
    topk=10,
    block_rows=256,
    weight_scores=None,
    interpret=False,
    index_map=None,
):
    """Top-k over [R, N] scores -> (weights f32, indices i32) [R, topk].

    With `weight_scores=None`, return the selected scores as weights.
    Otherwise, select experts by `scores` and read their weights from the
    separate [R, N] `weight_scores` table. DeepSeek-V4's `noaux` uses this
    to select with `scores=raw_scores + correction_bias` while returning
    unbiased weights from `weight_scores=raw_scores`. Here `raw_scores` are
    the scores after the scoring function, before adding the bias.
    Normalization and routed scaling are applied by the caller.

    `index_map` [N] int32 relabels the emitted indices: a pick of column c
    returns index_map[c]. Selection, its lowest-column tie rule and the pick
    order are unchanged (a static expert placement's logical -> physical
    table, applied without a separate gather).

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
    in_spec = pl.BlockSpec((block_rows, N), lambda i: (i, 0))
    bs = (in_spec,) if weight_scores is None else (in_spec, in_spec)
    os = (
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0)),
    )
    kernel = _select_kernel if weight_scores is None else _select_kernel_split
    operands = (scores,) if weight_scores is None else (scores, weight_scores)
    if index_map is not None:
        if index_map.shape != (N,):
            raise ValueError(f"index_map must be [{N}]; got {tuple(index_map.shape)}")
        bs = (*bs, pl.BlockSpec((1, N), lambda i: (0, 0)))
        operands = (*operands, index_map.astype(jnp.int32).reshape(1, N))
        kernel = functools.partial(
            _select_kernel_mapped, split=weight_scores is not None
        )
    return pl.pallas_call(
        functools.partial(kernel, topk=topk, n=N),
        grid=grid,
        in_specs=list(bs),
        out_specs=list(os),
        out_shape=list(out_shapes),
        interpret=interpret,
    )(*operands)


def expert_load_counts(counts, logits, bias, valid, *, topk):
    """counts [E] int32 plus the routed rows per expert of `logits` [T, E]:
    each valid row's top `topk` experts by biased sigmoid score (the noaux
    selection; an exact tie may resolve to another expert than the kernel's).
    Rows with valid False (padding) add nothing. A compare-and-sum over the
    picks rather than a scatter-add: a reduction stays on the TensorCore,
    where the scatter is offloaded to the SparseCore."""
    scores = jax.nn.sigmoid(logits.astype(jnp.float32)) + bias.astype(jnp.float32)
    _, ids = jax.lax.top_k(scores, topk)
    experts = jnp.arange(counts.shape[0], dtype=ids.dtype)
    hits = jnp.logical_and(ids[:, :, None] == experts, valid[:, None, None])
    return counts + jnp.sum(hits, axis=(0, 1), dtype=counts.dtype)
