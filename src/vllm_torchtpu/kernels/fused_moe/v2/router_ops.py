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
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v2.tuned_block_sizes import (
    NORMALIZED_ROUTER_BLOCK_ROWS,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

NEG = -3.0e38  # below any real softmax score (>=0); avoids inf


_SELECTOR_ROW_TILES = {(2048, 384, 6, 256): 2048}


def normalized_router_block_rows(
    gate: jax.Array,
    *,
    topk: int,
    scoring_fn: str,
    renormalize: bool,
    has_bias: bool,
    has_hash: bool,
) -> int | None:
    """Return a measured normalized-selector tile, or retain generic routing.

    The serving adapter widens gate logits to FP32 before calling the layer.
    Scoring and finite-row checks support both that form and direct BF16 inputs.
    """
    if (
        scoring_fn != "sqrtsoftplus"
        or not renormalize
        or not has_bias
        or has_hash
        or gate.dtype not in (jnp.bfloat16, jnp.float32)
        or jax.default_backend() != "tpu"
    ):
        return None
    key = (pltpu.get_tpu_info().chip_version, *gate.shape, topk)
    block_rows = NORMALIZED_ROUTER_BLOCK_ROWS.get(key)
    if block_rows is None:
        logger.warning_once(
            "[moe_router] No normalized selector tile for %s; using generic routing",
            key,
        )
    return block_rows


def _select_kernel(x_ref, w_ref, i_ref, *, topk, n, v_ref=None, normalize=False):
    """One block of rows: take the top `topk` by repeated max-and-mask."""
    if normalize:
        _normalized_select_kernel(x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref)
        return
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    raw = x_ref[...].T
    x = jnp.where(jnp.isnan(raw), NEG, raw)  # [n, R] f32
    v = None if v_ref is None else v_ref[...].T
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 0).astype(jnp.float32)
    for j in range(topk):
        m = jnp.max(x, axis=0, keepdims=True)  # [1,R] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        idx = jnp.min(jnp.where(ismax, iota, jnp.float32(n - 1)), axis=0)
        picked = iota == idx[None, :]
        # noaux selects with biased scores but emits unbiased weights.
        # A masked sum reads the resident tile without a dynamic lane gather.
        w_ref[j, :] = (
            m[0, :] if v is None else jnp.sum(jnp.where(picked, v, 0.0), axis=0)
        )
        i_ref[j, :] = idx.astype(jnp.int32)
        x = jnp.where(picked, NEG, x)

def _normalized_select_kernel(x_ref, w_ref, i_ref, *, topk, n, v_ref=None, normalize=True):
    """One block of rows: take the top `topk` by repeated max-and-mask."""
    # NaN is flushed to the sentinel so every row has a maximum some
    # column equals; an all-NaN row selects expert 0 with weight NEG.
    x = jnp.where(jnp.isnan(x_ref[...]), NEG, x_ref[...])  # [R, n] f32
    v = None if v_ref is None else v_ref[...]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    output_columns = packed_weights = pair_first = total = None
    if normalize:
        packed_shape = (x.shape[0], topk)
        output_columns = jax.lax.broadcasted_iota(jnp.int32, packed_shape, 1)
        packed_weights = jnp.zeros(packed_shape, jnp.float32)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)  # [R,1] rowmax
        ismax = x == m
        # Lowest matching column; non-matching columns fill with the last
        # expert so every index returned is a real expert.
        if normalize:
            # IDs 0..383 are exact in FP32; avoid decomposing an INT32 reduction.
            idx = jnp.min(
                jnp.where(ismax, iota.astype(jnp.float32), jnp.float32(n - 1)),
                axis=1,
            ).astype(jnp.int32)
        else:
            idx = jnp.min(jnp.where(ismax, iota, n - 1), axis=1)
        picked = iota == idx[:, None]
        # noaux selects with biased scores but emits unbiased weights.
        # A masked sum reads the resident tile without a dynamic lane gather.
        selected_weight = (
            m[:, 0] if v is None else jnp.sum(jnp.where(picked, v, 0.0), axis=1)
        )
        if normalize:
            assert output_columns is not None and packed_weights is not None
            packed_weights = jnp.where(
                output_columns == j, selected_weight[:, None], packed_weights
            )
            if j % 2 == 0:
                pair_first = selected_weight
            else:
                assert pair_first is not None
                pair_sum = pair_first + selected_weight
                if j == 1:
                    total = pair_sum
                else:
                    assert total is not None
                    total = total + pair_sum
        else:
            w_ref[:, j] = selected_weight
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(picked, NEG, x)
    if normalize:
        assert packed_weights is not None and total is not None
        # Keep ((w0+w1)+(w2+w3))+(w4+w5), then write all six columns together.
        w_ref[...] = packed_weights / jnp.maximum(total, 1e-20)[:, None]


def _select_kernel_split(x_ref, v_ref, w_ref, i_ref, *, topk, n, normalize=False):
    """Select by x_ref and read the selected weights from v_ref.

    Pallas passes input refs before output refs, so the two-input call needs
    this signature. Forward v_ref as a keyword to reuse _select_kernel's
    selection loop while keeping its one-input call unchanged.
    """
    _select_kernel(
        x_ref, w_ref, i_ref, topk=topk, n=n, v_ref=v_ref, normalize=normalize
    )


def pallas_select(
    scores: jax.Array,
    topk: int = 10,
    block_rows: int = 256,
    weight_scores: jax.Array | None = None,
    interpret: bool = False,
    *,
    normalize: bool = False,
) -> list[jax.Array]:
    """Top-k over [R, N] scores -> (weights f32, indices i32) [R, topk].

    With `weight_scores=None`, return the selected scores as weights.
    Otherwise, select experts by `scores` and read their weights from the
    separate [R, N] `weight_scores` table. DeepSeek-V4's `noaux` uses this
    to select with `scores=raw_scores + correction_bias` while returning
    unbiased weights from `weight_scores=raw_scores`. Here `raw_scores` are
    the scores after the scoring function, before adding the bias.
    By default, normalization and routed scaling are applied by the caller.
    `normalize=True` packs and normalizes Top6 weights for the measured
    [288,384] two-table tile. Validity masking and scaling remain in the caller.

    `interpret` runs the kernel body on the host, which is what lets the
    selection rule be tested without a chip; it does not gate Mosaic
    lowering. Serving never sets it.
    """
    R, N = scores.shape
    if normalize and not (
        (R, N, topk, block_rows) == (288, 384, 6, 288)
        and scores.dtype == jnp.float32
        and weight_scores is not None
        and weight_scores.dtype == jnp.float32
    ):
        raise ValueError(
            "Normalized selection requires FP32 [288,384] scores and weights, "
            "topk=6, block_rows=288"
        )
    assert R % block_rows == 0, (R, block_rows)
    if not normalize:
        block_rows = _SELECTOR_ROW_TILES.get((R, N, topk, block_rows), block_rows)
        # The token axis is minor in the transposed output: partial tiles need 128.
        if block_rows != R and block_rows % 128:
            block_rows = 128 if R % 128 == 0 else R
    if weight_scores is not None and weight_scores.shape != scores.shape:
        raise ValueError(
            f"weight_scores {weight_scores.shape} must match scores {scores.shape}"
        )
    grid = (R // block_rows,)
    output_shape = (R, topk) if normalize else (topk, R)
    out_shapes = (
        jax.ShapeDtypeStruct(output_shape, jnp.float32),
        jax.ShapeDtypeStruct(output_shape, jnp.int32),
    )
    in_spec = pl.BlockSpec((block_rows, N), lambda i: (i, 0))
    bs = (in_spec,) if weight_scores is None else (in_spec, in_spec)
    output_spec = (
        pl.BlockSpec((block_rows, topk), lambda i: (i, 0))
        if normalize
        else pl.BlockSpec((topk, block_rows), lambda i: (0, i))
    )
    os = (output_spec, output_spec)
    kernel = _select_kernel if weight_scores is None else _select_kernel_split
    operands = (scores,) if weight_scores is None else (scores, weight_scores)
    result = pl.pallas_call(
        functools.partial(kernel, topk=topk, n=N, normalize=normalize),
        grid=grid,
        in_specs=list(bs),
        out_specs=list(os),
        out_shape=list(out_shapes),
        interpret=interpret,
    )(*operands)

    return result if normalize else [result[0].T, result[1].T]
