# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""The fused MoE prefill kernel's routing plan (XLA, no TPU needed): its
packed-key sorts give the tables of the two key/value sorts they replace,
kept below as the reference, bit for bit. Routings as fused_moe_prefill hands
them over: random and skewed, with padded rows and sentinel routes, TP and
EP-local ids, packed activation scales of one to six words per token, at the
route counts where each packed sort hands over to its key/value sort."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel as kernel,
)

_TABLES = (
    "row_token",
    "slab_row",
    "sorted_scale",
    "visit_expert",
    "visit_start",
    "visit_rows",
    "routed_visits",
)


def _two_sort_plan(topk_ids, route_valid, experts, row_scale):
    """The routing plan with the two key/value sorts, as before the packed
    keys (verbatim but for this docstring)."""
    tokens, topk = topk_ids.shape
    routes = tokens * topk
    route_expert = jnp.where(
        route_valid.reshape(-1), topk_ids.reshape(-1).astype(jnp.int32), experts
    )
    iota = jnp.arange(routes, dtype=jnp.int32)
    route_scales = [
        jnp.broadcast_to(row_scale[:, k, None], (tokens, topk)).reshape(-1)
        for k in range(row_scale.shape[1])
    ]
    _, order, *sorted_scales = lax.sort(
        (route_expert, iota, *route_scales), num_keys=1, is_stable=True
    )
    _, slab_row = lax.sort((order, iota), num_keys=1, is_stable=False)
    counts = jnp.sum(
        route_expert[None, :] == jnp.arange(experts, dtype=jnp.int32)[:, None],
        axis=1,
        dtype=jnp.int32,
    )
    first = jnp.cumsum(counts) - counts
    active = counts > 0
    (visit_expert,) = jnp.nonzero(
        active, size=experts + kernel._VISIT_SLACK, fill_value=experts
    )
    visit_expert = visit_expert.astype(jnp.int32)
    counts = jnp.concatenate([counts, jnp.zeros((1,), jnp.int32)])
    first = jnp.concatenate([first, jnp.zeros((1,), jnp.int32)])
    return (
        order // topk,
        slab_row.reshape(tokens, topk),
        jnp.stack(sorted_scales),
        visit_expert,
        first[visit_expert],
        counts[visit_expert],
        jnp.sum(active, dtype=jnp.int32),
    )


def _words(h):
    """Packed activation-scale words per token for hidden size h."""
    return -(-(h // kernel._Q_BLOCK) // 2)


def _routing(tokens, topk, experts, seed, *, h, ranks=1, rank=0, skew=0.0, padded=3):
    """(topk_ids, route_valid, row_scale) as fused_moe_prefill passes them to
    the plan. Each token picks topk distinct of ranks * experts global experts
    (expert e with weight (e + 1) ** -skew); the rank holds experts
    [rank * experts, (rank + 1) * experts). Routes with router weight 0 (2% at
    random, and every route of the last `padded` rows) or to another rank's
    expert are not valid; their ids become 0 locally. row_scale [T, P] int32:
    one bf16 scale per row per 512 columns of h, two per word, as
    _quantize_rows packs them."""
    rng = np.random.default_rng(seed)
    n_global = ranks * experts
    logits = -skew * np.log(np.arange(1, n_global + 1))
    # Gumbel top-k: topk distinct experts per token, weighted by the logits.
    keys = logits + rng.gumbel(size=(tokens, n_global))
    ids = np.argsort(-keys, axis=1)[:, :topk].astype(np.int32)
    weights = rng.random((tokens, topk), dtype=np.float32) + 0.01
    weights[rng.random((tokens, topk)) < 0.02] = 0
    weights[tokens - padded :] = 0
    n_q = h // kernel._Q_BLOCK
    scale = rng.random((tokens, n_q), dtype=np.float32) / 448
    scale[rng.random((tokens, n_q)) < 0.05] = 0  # all-zero blocks get scale 0
    scale[::7] = 0  # and all-zero rows
    bits = scale.view(np.uint32)
    bits = (bits + np.uint32(0xFFFF)) & np.uint32(0xFFFF0000)  # up to bf16
    if n_q % 2:
        bits = np.pad(bits, ((0, 0), (0, 1)))
    row_scale = (bits[:, 0::2] | (bits[:, 1::2] >> 16)).view(np.int32)
    # fused_moe_prefill's mapping to the rank's local ids.
    route_valid = weights != 0
    local_ids = ids - rank * experts
    local = (local_ids >= 0) & (local_ids < experts)
    route_valid &= local
    local_ids = np.where(local, local_ids, 0)
    return jnp.asarray(local_ids), jnp.asarray(route_valid), jnp.asarray(row_scale)


_packed_plan = jax.jit(kernel._routing_plan, static_argnums=2)
_reference_plan = jax.jit(_two_sort_plan, static_argnums=2)


def _assert_same_tables(topk_ids, route_valid, experts, row_scale):
    got = _packed_plan(topk_ids, route_valid, experts, row_scale)
    want = _reference_plan(topk_ids, route_valid, experts, row_scale)
    for name, g, w in zip(_TABLES, got, want, strict=True):
        g, w = np.asarray(g), np.asarray(w)
        assert (g.dtype, g.shape) == (w.dtype, w.shape), name
        # Raw bits: the scale words are compared as bytes.
        np.testing.assert_array_equal(
            g.reshape(-1).view(np.uint8), w.reshape(-1).view(np.uint8), err_msg=name
        )
    return got


@pytest.mark.parametrize(
    "tokens,topk,experts,ranks,rank,skew,h",
    [
        (256, 8, 64, 1, 0, 0.0, 6144),  # top-8 over 64 experts, h 6144: 6 words
        (1000, 10, 64, 8, 3, 0.0, 1024),  # the kernel test's EP shapes, rank 3
        (4096, 8, 256, 1, 0, 2.0, 6144),  # GLM-5.2 TP16 (E 256, top-8), skewed
        (4096, 8, 16, 16, 15, 2.0, 6144),  # GLM-5.2 EP16, the last (coldest) rank
        (1000, 10, 512, 1, 0, 0.0, 4096),  # Qwen3.5 TP: E 512, top-10, 4 words
        (512, 8, 64, 1, 0, 1.0, 1536),  # 3 scales: the last word half empty
    ],
)
def test_packed_plan_matches_two_sorts(tokens, topk, experts, ranks, rank, skew, h):
    topk_ids, route_valid, row_scale = _routing(
        tokens, topk, experts, tokens + rank, h=h, ranks=ranks, rank=rank, skew=skew
    )
    row_token, slab_row, sorted_scale, *_ = _assert_same_tables(
        topk_ids, route_valid, experts, row_scale
    )
    # The tables mean what they say: each valid route's slab row holds its
    # token and that token's scale words.
    t, k = np.nonzero(np.asarray(route_valid))
    rows = np.asarray(slab_row)[t, k]
    np.testing.assert_array_equal(np.asarray(row_token)[rows], t)
    np.testing.assert_array_equal(
        np.asarray(sorted_scale)[:, rows], np.asarray(row_scale)[t].T
    )


# Route counts on both sides of each packed sort's bound (65,536 for the
# inverse, 2^20 for the expert order), and experts on both sides of 2^11.
@pytest.mark.parametrize(
    "tokens,topk,experts,h",
    [
        (8192, 8, 256, 6144),  # 65,536 routes: both sorts packed (GLM-5.2 at 8192)
        (65537, 1, 16, 1024),  # 65,537: the inverse falls back
        (131071, 8, 16, 1024),  # 2^20 - 8: the expert order is still packed
        (131072, 8, 16, 1024),  # 2^20: both fall back
        (64, 8, 2047, 1024),  # the sentinel 2047 fills the expert's 11 bits
        (64, 8, 2048, 1024),  # the expert order falls back
    ],
)
def test_packed_plan_matches_two_sorts_at_bounds(tokens, topk, experts, h):
    topk_ids, route_valid, row_scale = _routing(tokens, topk, experts, tokens, h=h)
    _assert_same_tables(topk_ids, route_valid, experts, row_scale)


def test_every_route_padded():
    """No valid route: no routed visit, and the tables still match."""
    topk_ids, route_valid, row_scale = _routing(256, 8, 64, 0, h=6144, padded=256)
    *_, visit_expert, _, visit_rows, routed_visits = _assert_same_tables(
        topk_ids, route_valid, 64, row_scale
    )
    assert int(routed_visits) == 0
    assert (np.asarray(visit_expert) == 64).all()
    assert (np.asarray(visit_rows) == 0).all()


def _sorts(tokens, topk, experts, words):
    """(operand dtypes, is_stable) of each sort in the plan, in order."""
    jaxpr = jax.make_jaxpr(kernel._routing_plan, static_argnums=2)(
        jax.ShapeDtypeStruct((tokens, topk), jnp.int32),
        jax.ShapeDtypeStruct((tokens, topk), jnp.bool_),
        experts,
        jax.ShapeDtypeStruct((tokens, words), jnp.int32),
    )
    return [
        (tuple(str(v.aval.dtype) for v in eqn.invars), eqn.params["is_stable"])
        for eqn in jaxpr.eqns
        if eqn.primitive is lax.sort_p
    ]


def _packed_order(words):
    return ("uint32",) + ("int32",) * words, False


def _order(words):
    return ("int32", "int32") + ("int32",) * words, True


_PACKED_INVERSE = (("uint32",), False)
_INVERSE = (("int32", "int32"), False)


@pytest.mark.parametrize(
    "tokens,topk,experts,h,packed",
    [
        (4096, 8, 256, 6144, (True, True)),
        (8192, 8, 256, 6144, (True, True)),
        (65537, 1, 16, 1024, (True, False)),
        (16384, 8, 256, 6144, (True, False)),
        (131071, 8, 16, 1024, (True, False)),
        (131072, 8, 16, 1024, (False, False)),
        (64, 8, 2047, 1024, (True, True)),
        (64, 8, 2048, 1024, (False, True)),
    ],
)
def test_which_sorts_run(tokens, topk, experts, h, packed):
    """The cases above reach the packed sorts they are meant to test."""
    words = _words(h)
    want = [
        _packed_order(words) if packed[0] else _order(words),
        _PACKED_INVERSE if packed[1] else _INVERSE,
    ]
    assert _sorts(tokens, topk, experts, words) == want
