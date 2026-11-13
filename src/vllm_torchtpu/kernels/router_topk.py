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
from jax.experimental.pallas import tpu as pltpu

# Clamp floor. Below any real routing score (softmax, sigmoid and sqrtsoftplus
# are all >= 0) and strictly above the dtype minimum used to mask a selected
# column, so a row's maximum is always above the mask.
NEG = -3.0e38

# Rows per grid step, measured fastest at 512 experts. The grid rounds up, so
# this does not have to divide the row count.
MAX_BLOCK_ROWS = 512
MAX_WITH_RAW_BLOCK_ROWS = 1024


SCORING_CODE_TO_NAME = {
    0: "none",
    1: "sqrtsoftplus",
    2: "sigmoid",
    3: "softmax",
}
SCORING_NAME_TO_CODE = {v: k for k, v in SCORING_CODE_TO_NAME.items()}


def _apply_scoring_fn_jax(s: jax.Array, scoring_fn: str) -> jax.Array:
    if scoring_fn == "none":
        return s
    if scoring_fn == "sqrtsoftplus":
        sp = jnp.where(s > 20.0, s, jnp.log(1.0 + jnp.exp(s)))
        return jnp.sqrt(sp)
    if scoring_fn == "sigmoid":
        return jax.nn.sigmoid(s)
    if scoring_fn == "softmax":
        return jax.nn.softmax(s, axis=1)
    raise ValueError(f"Unsupported scoring_fn: {scoring_fn!r}")


def _select_kernel(
    x_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    scoring_fn: str = "none",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
):
    """One block of rows: top ``topk`` by repeated max-and-mask.

    Scores at or below ``NEG`` are clamped up to it first, so every row has a
    maximum some column equals and every index written is a real expert. Ties
    go to the lowest id.

    The clamp tests ``x > NEG`` rather than ``isnan(x)`` to catch all three
    spellings of the hazard -- NaN, ``-inf``, ``-FLT_MAX`` -- any of which a
    whole row can be made of. Without it such a row would stay tied with the
    ``finfo.min`` mask and every pass would return column 0.
    """
    s = (
        x_ref[...]
        if scoring_fn == "none" and x_ref.dtype == jnp.float32
        else _apply_scoring_fn_jax(x_ref[...].astype(jnp.float32), scoring_fn)
    )
    x = jnp.where(s > NEG, s, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    fuse_epilogue = renormalize or routed_scaling_factor != 1.0
    w_cols = []
    w_sum = None
    m0 = None
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        if j == 0:
            m0 = m[:, 0]
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        w_j = m[:, 0]
        if fuse_epilogue:
            w_cols.append(w_j)
            if renormalize:
                w_sum = w_j if w_sum is None else (w_sum + w_j)
        else:
            w_ref[:, j] = w_j
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(iota == idx[:, None], masked, x)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous rather than letting renormalization make it plausible.
    if fuse_epilogue:
        denom = jnp.maximum(w_sum, 1e-20) if renormalize else None
        poison = m0 <= NEG
        for j in range(topk):
            w_j = w_cols[j] / denom if denom is not None else w_cols[j]
            if routed_scaling_factor != 1.0:
                w_j = w_j * routed_scaling_factor
            w_ref[:, j] = jnp.where(poison, jnp.nan, w_j)
    else:
        w_ref[...] = jnp.where(w_ref[:, :1] <= NEG, jnp.nan, w_ref[...])


def _select_with_raw_kernel(x_ref, s_ref, w_ref, i_ref, *, topk: int, n: int):
    """Top ``topk`` selection on ``x_ref`` (biased choice scores), writing
    unbiased weights from ``s_ref`` directly to ``w_ref`` without a post-kernel
    3D gather."""
    x = jnp.where(x_ref[...] > NEG, x_ref[...], NEG)
    s = s_ref[...]
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        sel = iota == idx[:, None]
        w_ref[:, j] = jnp.sum(jnp.where(sel, s, 0.0), axis=1)
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(sel, masked, x)


def _select_with_bias_kernel(
    s_ref,
    b_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    scoring_fn: str = "none",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
):
    """Top ``topk`` selection on ``scoring_fn(s_ref) + b_ref`` in VMEM, writing
    unbiased (and optionally renormalized/scaled) weights from ``s_ref`` without
    materializing intermediate ``[rows, n]`` or ``[rows, topk]`` tensors in HBM."""
    s = (
        s_ref[...]
        if scoring_fn == "none" and s_ref.dtype == jnp.float32
        else _apply_scoring_fn_jax(s_ref[...].astype(jnp.float32), scoring_fn)
    )
    x = s + b_ref[...]
    x = jnp.where(x > NEG, x, NEG)
    masked = jnp.finfo(x.dtype).min
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    fuse_epilogue = renormalize or routed_scaling_factor != 1.0
    w_cols = []
    w_sum = None
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        sel = iota == idx[:, None]
        w_j = jnp.sum(jnp.where(sel, s, 0.0), axis=1)
        if fuse_epilogue:
            w_cols.append(w_j)
            if renormalize:
                w_sum = w_j if w_sum is None else (w_sum + w_j)
        else:
            w_ref[:, j] = w_j
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(sel, masked, x)
    if fuse_epilogue:
        denom = jnp.maximum(w_sum, 1e-20) if renormalize else None
        for j in range(topk):
            w_j = w_cols[j] / denom if denom is not None else w_cols[j]
            if routed_scaling_factor != 1.0:
                w_j = w_j * routed_scaling_factor
            w_ref[:, j] = w_j


def select(
    scores: jax.Array,
    topk: int,
    interpret: bool = False,
    *,
    scoring_fn: str = "none",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``[rows, experts]`` scores or logits, values descending.

    Returns ``(weights f32, indices int32)`` of shape ``[rows, topk]``, values
    descending. ``interpret`` runs the kernel on the host, for tests without a
    TPU.

    The last grid step reads a partial block when ``rows`` is not a multiple
    of the block; Mosaic discards the stores past ``rows``, and every
    reduction is along the expert axis, so surplus rows cannot perturb a real
    one.
    """
    orig_rows, n = scores.shape
    pad_r = (-orig_rows) & 7
    if pad_r:
        scores = jnp.pad(scores, ((0, pad_r), (0, 0)))
    rows = orig_rows + pad_r
    is_fused = scoring_fn != "none" or renormalize or routed_scaling_factor != 1.0
    max_blk = (
        MAX_WITH_RAW_BLOCK_ROWS
        if (is_fused and n > 128 and not interpret)
        else MAX_BLOCK_ROWS
    )
    block = min(max_blk, rows)
    w, idx = pl.pallas_call(
        functools.partial(
            _select_kernel,
            topk=topk,
            n=n,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        ),
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
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(scores)
    return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)


def select_with_raw(
    scores: jax.Array,
    raw_scores: jax.Array,
    topk: int,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Top-k over biased ``scores``, returning unbiased ``raw_scores`` weights."""
    orig_rows, n = scores.shape
    pad_r = (-orig_rows) & 7
    if pad_r:
        scores = jnp.pad(scores, ((0, pad_r), (0, 0)))
        raw_scores = jnp.pad(raw_scores, ((0, pad_r), (0, 0)))
    rows = orig_rows + pad_r
    max_blk = (
        MAX_WITH_RAW_BLOCK_ROWS
        if (n > 128 and not interpret)
        else MAX_BLOCK_ROWS
    )
    block = min(max_blk, rows)
    w, idx = pl.pallas_call(
        functools.partial(_select_with_raw_kernel, topk=topk, n=n),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, n), lambda i: (i, 0)),
            pl.BlockSpec((block, n), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(scores, raw_scores)
    return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)


def select_with_bias(
    raw_scores: jax.Array,
    bias: jax.Array,
    topk: int,
    interpret: bool = False,
    *,
    scoring_fn: str = "none",
    renormalize: bool = False,
    routed_scaling_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Top-k over ``scoring_fn(raw_scores) + bias`` in VMEM, returning unbiased
    (and optionally renormalized/scaled) weights."""
    orig_rows, n = raw_scores.shape
    pad_r = (-orig_rows) & 7
    if pad_r:
        raw_scores = jnp.pad(raw_scores, ((0, pad_r), (0, 0)))
    rows = orig_rows + pad_r
    bias_2d = bias.reshape(1, n)
    max_blk = (
        MAX_WITH_RAW_BLOCK_ROWS
        if (n > 128 and not interpret)
        else MAX_BLOCK_ROWS
    )
    block = min(max_blk, rows)
    w, idx = pl.pallas_call(
        functools.partial(
            _select_with_bias_kernel,
            topk=topk,
            n=n,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
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
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(raw_scores, bias_2d)
    return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)


def router_topk(scores: jax.Array, topk: int) -> tuple[jax.Array, jax.Array]:
    """``select`` with a jax_op-compatible signature (arrays, then statics)."""
    return select(scores, topk)


def router_topk_with_raw(
    scores: jax.Array, raw_scores: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """``select_with_raw`` with a jax_op-compatible signature."""
    return select_with_raw(scores, raw_scores, topk)


def router_topk_with_bias(
    raw_scores: jax.Array, bias: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array]:
    """``select_with_bias`` with a jax_op-compatible signature."""
    return select_with_bias(raw_scores, bias, topk)


def router_topk_fused(
    router_logits: jax.Array,
    topk: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Fused ``scoring_fn + topk + renormalize + scale`` with a jax_op signature."""
    return select(
        router_logits,
        topk,
        scoring_fn=SCORING_CODE_TO_NAME[scoring_code],
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def router_topk_fused_with_bias(
    router_logits: jax.Array,
    bias: jax.Array,
    topk: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """Fused ``scoring_fn + bias + topk + renormalize + scale`` with a jax_op signature."""
    return select_with_bias(
        router_logits,
        bias,
        topk,
        scoring_fn=SCORING_CODE_TO_NAME[scoring_code],
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def _resolve_grouped_block_rows(rows: int, interpret: bool = False) -> int:
    if interpret:
        return min(MAX_BLOCK_ROWS, rows)
    if rows > MAX_WITH_RAW_BLOCK_ROWS and rows % MAX_BLOCK_ROWS == 0 and rows % MAX_WITH_RAW_BLOCK_ROWS != 0:
        return min(MAX_BLOCK_ROWS, rows)
    return min(MAX_WITH_RAW_BLOCK_ROWS, rows)


def _grouped_select_epilogue(
    raw_logits: jax.Array,
    bias_2d: jax.Array | None,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    num_expert_group: int,
    topk_group: int,
    scoring_fn: str,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
):
    """In-VMEM grouped top-k router selection keeping ``[block, n]`` in native
    128-lane VREG layout with zero non-128-aligned reshapes or HBM round-trips.
    """
    s = (
        raw_logits
        if scoring_fn == "none" and raw_logits.dtype == jnp.float32
        else _apply_scoring_fn_jax(raw_logits.astype(jnp.float32), scoring_fn)
    )
    has_bias = bias_2d is not None
    x_choice = (s + bias_2d[:1, :].astype(jnp.float32)) if has_bias else s

    iota = jax.lax.broadcasted_iota(jnp.int32, x_choice.shape, 1)
    per_group = n // num_expert_group

    # 1. Per-group score in registers ([block, 1] per group; preserves exact tie count)
    gs_cols = []
    for g in range(num_expert_group):
        in_g = (iota >= g * per_group) & (iota < (g + 1) * per_group)
        xg = jnp.where(in_g, x_choice, NEG)
        m1 = jnp.max(xg, axis=1, keepdims=True)
        if has_bias:
            idx1 = jnp.min(
                jnp.where(xg == m1, iota, (g + 1) * per_group - 1),
                axis=1,
                keepdims=True,
            )
            m2 = jnp.max(jnp.where(iota == idx1, NEG, xg), axis=1, keepdims=True)
            gs_cols.append(m1 + m2)
        else:
            gs_cols.append(m1)

    # 2. Top ``topk_group`` groups out of ``num_expert_group`` (lowest group id wins ties)
    g_selected = [
        jnp.zeros((x_choice.shape[0], 1), dtype=jnp.bool_)
        for _ in range(num_expert_group)
    ]
    cur_gs = list(gs_cols)
    for _ in range(topk_group):
        best = cur_gs[0]
        for g in range(1, num_expert_group):
            best = jnp.maximum(best, cur_gs[g])
        picked_any = jnp.zeros((x_choice.shape[0], 1), dtype=jnp.bool_)
        for g in range(num_expert_group):
            pick_g = (~picked_any) & (cur_gs[g] == best)
            picked_any = picked_any | pick_g
            g_selected[g] = g_selected[g] | pick_g
            cur_gs[g] = jnp.where(pick_g, NEG, cur_gs[g])

    # 3. Mask unselected groups to NEG across [block, n]
    keep_expert = jnp.zeros(x_choice.shape, dtype=jnp.bool_)
    for g in range(num_expert_group):
        in_g = (iota >= g * per_group) & (iota < (g + 1) * per_group)
        keep_expert = keep_expert | (in_g & g_selected[g])

    x = jnp.where(keep_expert & (x_choice > NEG), x_choice, NEG)
    masked = jnp.finfo(x.dtype).min

    # 4. Sort-free top-k selection on masked biased scores + in-VMEM epilogue
    fuse_epilogue = renormalize or routed_scaling_factor != 1.0
    w_cols = []
    w_sum = None
    for j in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        sel = iota == idx[:, None]
        w_j = jnp.max(jnp.where(sel, s, -1.0), axis=1) if has_bias else m[:, 0]
        if fuse_epilogue:
            w_cols.append(w_j)
            if renormalize:
                w_sum = w_j if w_sum is None else (w_sum + w_j)
        else:
            w_ref[:, j] = w_j
        i_ref[:, j] = idx.astype(jnp.int32)
        x = jnp.where(sel, masked, x)
    if fuse_epilogue:
        denom = jnp.maximum(w_sum, 1e-20) if renormalize else None
        for j in range(topk):
            w_j = w_cols[j] / denom if denom is not None else w_cols[j]
            if routed_scaling_factor != 1.0:
                w_j = w_j * routed_scaling_factor
            w_ref[:, j] = w_j


def _select_grouped_kernel(
    s_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    num_expert_group: int,
    topk_group: int,
    scoring_fn: str,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
):
    _grouped_select_epilogue(
        s_ref[...],
        None,
        w_ref,
        i_ref,
        topk=topk,
        n=n,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_fn=scoring_fn,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def _select_grouped_with_bias_kernel(
    s_ref,
    b_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    num_expert_group: int,
    topk_group: int,
    scoring_fn: str,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
):
    _grouped_select_epilogue(
        s_ref[...],
        b_ref[...],
        w_ref,
        i_ref,
        topk=topk,
        n=n,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_fn=scoring_fn,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def _fused_gate_grouped_with_bias_kernel(
    x_ref,
    w_gate_ref,
    b_ref,
    w_ref,
    i_ref,
    *,
    topk: int,
    n: int,
    num_expert_group: int,
    topk_group: int,
    scoring_fn: str,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
):
    logits = jnp.dot(
        x_ref[...],
        w_gate_ref[...],
        preferred_element_type=jnp.float32,
    )
    _grouped_select_epilogue(
        logits,
        b_ref[...],
        w_ref,
        i_ref,
        topk=topk,
        n=n,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_fn=scoring_fn,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def select_grouped(
    router_logits: jax.Array,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    interpret: bool = False,
    *,
    scoring_fn: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """In-VMEM grouped top-k selection without score correction bias."""
    orig_rows, n = router_logits.shape
    pad_r = (-orig_rows) & 7
    if pad_r:
        router_logits = jnp.pad(router_logits, ((0, pad_r), (0, 0)))
    rows = orig_rows + pad_r
    block = _resolve_grouped_block_rows(rows, interpret=interpret)
    w, idx = pl.pallas_call(
        functools.partial(
            _select_grouped_kernel,
            topk=topk,
            n=n,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        ),
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
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(router_logits)
    return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)


def select_grouped_with_bias(
    router_logits: jax.Array,
    bias: jax.Array,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    interpret: bool = False,
    *,
    scoring_fn: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """In-VMEM grouped top-k selection with score correction bias."""
    orig_rows, n = router_logits.shape
    pad_r = (-orig_rows) & 7
    if pad_r:
        router_logits = jnp.pad(router_logits, ((0, pad_r), (0, 0)))
    rows = orig_rows + pad_r
    bias_2d = jnp.broadcast_to(bias.reshape(1, n), (8, n))
    block = _resolve_grouped_block_rows(rows, interpret=interpret)
    w, idx = pl.pallas_call(
        functools.partial(
            _select_grouped_with_bias_kernel,
            topk=topk,
            n=n,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, n), lambda i: (i, 0)),
            pl.BlockSpec((8, n), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(router_logits, bias_2d)
    return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)


def fused_gate_select_grouped_with_bias(
    hidden_states: jax.Array,
    gate_weight: jax.Array,
    bias: jax.Array,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    interpret: bool = False,
    *,
    scoring_fn: str = "sigmoid",
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
) -> tuple[jax.Array, jax.Array]:
    """Fused in-VMEM ``gate(hidden_states) + scoring_fn + bias + grouped_topk``."""
    orig_rows, h = hidden_states.shape
    pad_r = (-orig_rows) & 7
    if pad_r:
        hidden_states = jnp.pad(hidden_states, ((0, pad_r), (0, 0)))
    rows = orig_rows + pad_r
    if rows > MAX_BLOCK_ROWS and not interpret:
        logits = jnp.dot(
            hidden_states, gate_weight, preferred_element_type=jnp.float32
        )
        w, idx = select_grouped_with_bias(
            logits,
            bias,
            topk,
            num_expert_group,
            topk_group,
            interpret=False,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        )
        return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)
    _, n = gate_weight.shape
    bias_2d = jnp.broadcast_to(bias.reshape(1, n), (8, n))
    block = min(MAX_BLOCK_ROWS, rows)
    w, idx = pl.pallas_call(
        functools.partial(
            _fused_gate_grouped_with_bias_kernel,
            topk=topk,
            n=n,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_fn=scoring_fn,
            renormalize=renormalize,
            routed_scaling_factor=routed_scaling_factor,
        ),
        grid=(pl.cdiv(rows, block),),
        in_specs=[
            pl.BlockSpec((block, h), lambda i: (i, 0)),
            pl.BlockSpec((h, n), lambda i: (0, 0)),
            pl.BlockSpec((8, n), lambda i: (0, 0)),
        ],
        out_specs=[
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
            pl.BlockSpec((block, topk), lambda i: (i, 0)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((rows, topk), jnp.float32),
            jax.ShapeDtypeStruct((rows, topk), jnp.int32),
        ],
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        interpret=interpret,
    )(hidden_states, gate_weight, bias_2d)
    return (w[:orig_rows], idx[:orig_rows]) if pad_r else (w, idx)


def router_grouped_topk(
    router_logits: jax.Array,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """``select_grouped`` with a ``pallas.jax_op`` signature."""
    return select_grouped(
        router_logits,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_fn=SCORING_CODE_TO_NAME[scoring_code],
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def router_grouped_topk_with_bias(
    router_logits: jax.Array,
    bias: jax.Array,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """``select_grouped_with_bias`` with a ``pallas.jax_op`` signature."""
    return select_grouped_with_bias(
        router_logits,
        bias,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_fn=SCORING_CODE_TO_NAME[scoring_code],
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )


def router_fused_gate_grouped_topk_with_bias(
    hidden_states: jax.Array,
    gate_weight: jax.Array,
    bias: jax.Array,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    scoring_code: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> tuple[jax.Array, jax.Array]:
    """``fused_gate_select_grouped_with_bias`` with a ``pallas.jax_op`` signature."""
    return fused_gate_select_grouped_with_bias(
        hidden_states,
        gate_weight,
        bias,
        topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_fn=SCORING_CODE_TO_NAME[scoring_code],
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
    )

