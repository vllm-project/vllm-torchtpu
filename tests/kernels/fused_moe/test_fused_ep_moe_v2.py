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
"""Device tests for the fused expert-parallel MoE kernel, in pure JAX.

``fused_ep_moe_v2`` is the whole MoE layer: it routes, quantizes, dispatches
the routed rows over the expert-parallel mesh axis, runs the expert FFN and
combines the arrivals back onto the tokens that asked for them. So the only
useful gate is the layer's output against a plain dense reference -- softmax
router, top-k, per-expert dense matmuls, weighted sum -- computed over the
same weights with no fp8 transport anywhere.

The reference and the tolerances are the ones the kernel was originally
validated against: a dense ``ref_moe``, and its two bands -- a batch-wide
relative L2 and a worst-single-token relative L2. The reference here is the
same arithmetic spread over the expert axis rather than replicated, which is
what makes the 512-expert served shape affordable to check at all.

The mesh axis carries the name the kernel body reads
(``vllm_torchtpu.kernels.fused_moe.v2.AXIS``, "d") rather than a name of this
test's choosing; ``test_the_mesh_axis_name_is_the_one_the_kernel_reads`` pins
that constraint, which the layer's own signature does not state.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import (
    AXIS,
    WeightFormat,
    fused_ep_moe_v2,
    host,
    kernel,
    layer,
)

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip the module unless this host really has the eight-way slice.

    A fixture rather than a `skipif`, because a `skipif` is evaluated while
    pytest IMPORTS the module -- during collection, before markers deselect
    anything -- and `jax.device_count()` builds the PJRT client to answer.
    libtpu is process-exclusive, so a client built in the pytest parent locks
    every child out: `tests/conftest.py` puts vLLM workers in SPAWNED
    processes, and they all die with "pjrt client not initialized". Merely
    collecting this file was enough to take down every engine-starting test in
    the same pytest run, whether or not any test here was selected.

    A fixture runs at setup, after `-m` deselection, so on a host that is not
    the eight-chip agent nothing here touches the accelerator at all.
    """
    if jax.device_count() < 8:
        pytest.skip(
            "the fused EP MoE kernel is an eight-way expert-parallel "
            "collective and this host reports fewer than 8 JAX "
            "devices"
        )


# The expert-parallel width, and the tile height the serving adapter passes
# the kernel as its capacity (moe_fused_ep._TILE_M upstream).
EP = 8
CAPACITY = 128

FP8 = jnp.float8_e4m3fn

# The small structurally-complete shape: expert count divisible by the shard
# count, hidden a whole number of 128-lane blocks, intermediate over one.
SMALL = dict(hidden=512, inter=256, e_total=32, topk=4, tokens=256)

# Qwen3.5-397B's MoE block, which is also the served geometry the kernel was
# written for: 512 experts over 8 shards is 64 local experts a shard.
PROD = dict(hidden=4096, inter=1024, e_total=512, topk=10)
PROD_TOKENS = (2048, 16384)

# How far the layer may sit from the dense reference, as a batch-wide
# relative L2 and as the worst single token's. Taken from the kernel's own
# suite, where they are stated against an fp8 wire: the routed rows travel
# between shards quantized to e4m3 with one scale a row, and the intermediate
# is requantized the same way, so the band is the transport's and not the
# expert weights'. Measured on this host, as (batch, worst token): small fp8
# 0.0534/0.0636, T=2048 0.0540/0.0583, T=16384 0.0540/0.0615. The worst token
# clears the batch band on the small shape, which is what the second bound is
# for -- 256 tokens leave one unlucky row nowhere to hide.
FP8_RELATIVE_BOUND = 0.06
FP8_TOKEN_BOUND = 0.075

# The unquantized weight format quantizes nothing -- no fp8 rows, no fp8 wire
# -- so what is left is bf16 rounding in the matmuls and in the intermediate.
# Measured 0.00233/0.00261 on the small shape; the band is an order of
# magnitude under the fp8 one, which is the point of stating it separately.
BF16_RELATIVE_BOUND = 0.01
BF16_TOKEN_BOUND = 0.01

# How much further a reference computed over rotated experts has to sit than
# the reference itself, so that a band this wide still means something: a
# layer that routed to the wrong experts would not clear it. The kernel's own
# suite uses the same control and the same factor.
ROTATED_CONTROL_FACTOR = 10


def _mesh():
    """The single-axis expert-parallel mesh, over the first eight devices."""
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))


def _make_weights(mesh, seed, *, hidden, inter, e_total, weight_format):
    """(w1, w2, w1_scale, w2_scale) as global arrays sharded on the experts.

    Built a shard at a time rather than on one device and handed out: the
    served shape's f32 draw is 17GB where its fp8 form is 6.4GB, and only the
    latter has to exist at once. fp8 scales are per output channel, taken
    along each matmul's contraction axis, which is the layout the kernel
    takes: w1_scale [E, 2 * inter], w2_scale [E, hidden].
    """
    (axis,) = mesh.axis_names
    g_local = e_total // mesh.shape[axis]
    quantized = weight_format == WeightFormat.FP8

    def local(_):
        me = lax.axis_index(axis)
        k1, k2 = jax.random.split(jax.random.fold_in(jax.random.key(seed), me))
        weights = []
        for key, shape in (
            (k1, (g_local, hidden, 2 * inter)),
            (k2, (g_local, inter, hidden)),
        ):
            w = jax.random.normal(key, shape, jnp.float32) / 10
            if not quantized:
                weights.append((w.astype(jnp.bfloat16), None))
                continue
            amax = jnp.max(jnp.abs(w), axis=1, keepdims=True)
            scale = jnp.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
            weights.append(
                (
                    (w / scale).astype(FP8),
                    scale.astype(jnp.float32).reshape(shape[0], shape[2]),
                )
            )
        (w1, w1s), (w2, w2s) = weights
        return (w1, w2) if not quantized else (w1, w2, w1s, w2s)

    out_specs = (P(axis),) * (4 if quantized else 2)
    built = jax.jit(
        jax.shard_map(
            local, mesh=mesh, in_specs=(P(axis),), out_specs=out_specs, check_vma=False
        )
    )(jnp.zeros((EP,), jnp.float32))
    return built if quantized else (built[0], built[1], None, None)


def _make_fp4_weights(mesh, seed, *, hidden, inter, e_total, rhs_qb):
    """((w1, w2), (w1_scale, w2_scale), decoded) as fp4 e2m1 with block scales.

    One scale per `rhs_qb` rows of each matmul's contraction axis, which is
    the layout the kernel takes: w1_scale [E, hidden / qb, 2 * inter] and
    w2_scale [E, inter / qb, hidden]. `decoded` is what those two decode to,
    and is what the dense reference is computed over, so the band between
    them is the kernel's transport rather than the four-bit rounding.
    """
    shard = NamedSharding(mesh, P(AXIS))
    rng = np.random.default_rng(seed)
    weights, scales, decoded = [], [], []
    for shape in ((e_total, hidden, 2 * inter), (e_total, inter, hidden)):
        raw = rng.normal(0, 0.1, shape).astype(np.float32)
        grouped = raw.reshape(shape[0], shape[1] // rhs_qb, rhs_qb, shape[2])
        scale = np.maximum(np.max(np.abs(grouped), axis=2) / 6, 1e-12)
        normalized = (grouped / scale[:, :, None, :]).reshape(shape)
        w = jax.device_put(normalized, shard).astype(jnp.float4_e2m1fn)
        s = jax.device_put(scale, shard)
        weights.append(w)
        scales.append(s)
        decoded.append(
            (w.astype(jnp.float32).reshape(grouped.shape) * s[:, :, None, :]).reshape(
                shape
            )
        )
    return weights, scales, decoded


def _make_inputs(mesh, seed, *, tokens, hidden, e_total):
    """(x, gating) as global arrays sharded on the tokens."""
    (axis,) = mesh.axis_names
    shard = NamedSharding(mesh, P(axis))

    @jax.jit
    def build():
        kx, kg = jax.random.split(jax.random.key(seed))
        x = (jax.random.normal(kx, (tokens, hidden), jnp.float32) / 10).astype(
            jnp.bfloat16
        )
        gating = jax.random.normal(kg, (tokens, e_total), jnp.float32)
        return (
            lax.with_sharding_constraint(x, shard),
            lax.with_sharding_constraint(gating, shard),
        )

    return build()


def _routed_gating(mesh, seed, *, tokens, e_total, topk, expert_rows):
    """Gating whose top-k routes `expert_rows[e]` rows to each listed expert
    and spreads the rest of the tokens * topk routes evenly over the others.

    The routes are laid out column-major over the [tokens, topk] choices, so
    an expert's contiguous run of at most `tokens` of them lands on distinct
    tokens. A chosen expert scores ~6 and every other ~0, with noise in both
    so the router weights still differ.
    """
    (axis,) = mesh.axis_names
    others = [e for e in range(e_total) if e not in expert_rows]
    rest, extra = divmod(tokens * topk - sum(expert_rows.values()), len(others))
    counts = dict(expert_rows) | {e: rest + (i < extra) for i, e in enumerate(others)}
    assert max(counts.values()) <= tokens, counts
    seq = np.repeat(np.asarray(list(counts)), list(counts.values()))
    ids = seq.reshape(topk, tokens).T
    gating = np.random.default_rng(seed).normal(0.0, 0.5, (tokens, e_total))
    gating[np.arange(tokens)[:, None], ids] += 6.0
    return jax.device_put(
        jnp.asarray(gating, jnp.float32), NamedSharding(mesh, P(axis))
    )


def _dense_reference(
    mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk, hash_ids=None
):
    """The plain MoE: softmax, top-k, dense per-expert FFN, weighted sum.

    `hash_ids` [tokens, topk] replaces the top-k selection with ids the caller
    chose, keeping the weights as the scores at those ids -- the reference
    form of DeepSeek-V4's table router.

    Every expert is applied to every token in f32 and weighted by the router
    weight of the slots that chose it, which is zero for a token that did
    not. Nothing is quantized: the weights are dequantized once and the whole
    body runs in f32, so the difference from the kernel is the kernel's
    transport rather than a second implementation of the same shortcuts.

    The loop is spread over the same expert axis the layer shards on -- each
    shard applies its own experts to the whole batch and the results are
    summed across shards -- which is a distribution of the reference, not a
    change of it: the summands and their order are the same.
    """
    (axis,) = mesh.axis_names
    g_local = w1.shape[0] // mesh.shape[axis]
    inter = w1.shape[2] // 2
    scaled = w1_scale is not None

    def local(x_g, gating_g, w1_l, w2_l, *scales_and_ids):
        s1, s2 = scales_and_ids[:2] if scaled else (None, None)
        ids_g = scales_and_ids[-1] if hash_ids is not None else None
        scores = jax.nn.softmax(gating_g.astype(jnp.float32), axis=-1)
        if ids_g is None:
            topk_weights, topk_idx = lax.top_k(scores, topk)
        else:
            topk_idx = ids_g
            topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
        topk_weights = topk_weights / topk_weights.sum(axis=-1, keepdims=True)
        first = lax.axis_index(axis) * g_local
        x32 = x_g.astype(jnp.float32)

        def expert(e, acc):
            w1e = lax.dynamic_index_in_dim(w1_l, e, 0, keepdims=False).astype(
                jnp.float32
            )
            w2e = lax.dynamic_index_in_dim(w2_l, e, 0, keepdims=False).astype(
                jnp.float32
            )
            if scaled:
                w1e = w1e * lax.dynamic_index_in_dim(s1, e, 0, keepdims=True)
                w2e = w2e * lax.dynamic_index_in_dim(s2, e, 0, keepdims=True)
            weight = jnp.sum(
                jnp.where(topk_idx == first + e, topk_weights, 0.0), axis=-1
            )[:, None]
            acc1 = x32 @ w1e
            row = (jax.nn.silu(acc1[:, :inter]) * acc1[:, inter:]) @ w2e
            return acc + weight * row

        mine = lax.fori_loop(0, g_local, expert, jnp.zeros_like(x32))
        return lax.psum(mine, axis)

    in_specs = (
        (P(), P(), P(axis), P(axis))
        + ((P(axis),) * 2 if scaled else ())
        + ((P(),) if hash_ids is not None else ())
    )
    args = (
        (x, gating, w1, w2)
        + ((w1_scale, w2_scale) if scaled else ())
        + ((hash_ids,) if hash_ids is not None else ())
    )
    return jax.jit(
        jax.shard_map(
            local, mesh=mesh, in_specs=in_specs, out_specs=P(), check_vma=False
        )
    )(*args)


def _relative_l2(actual, want):
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    return float(np.linalg.norm(a - w) / np.linalg.norm(w))


def _worst_token_relative_l2(actual, want):
    """The largest per-token relative error. A routing failure is per token,
    and a batch-wide norm divides one bad row by every other row's size."""
    a = np.asarray(actual, np.float64)
    w = np.asarray(want, np.float64)
    per_token = np.linalg.norm(a - w, axis=-1)
    scale = np.linalg.norm(w, axis=-1)
    return float(np.max(per_token / np.where(scale == 0, 1.0, scale)))


def _check_against_the_reference(
    mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk, batch_bound, token_bound
):
    """Run the layer, hold it to the dense reference, and rotate the experts.

    The rotation is the control the bands need: both are wide enough to
    cover an fp8 wire, so without it a layer that sent every token to the
    wrong expert could still pass one of them.
    """
    out = fused_ep_moe_v2(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=topk,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=(WeightFormat.FP8 if w1.dtype == FP8 else WeightFormat.BF16),
    )
    # [tokens, hidden] sharded the way x is: the combine leaves each token's
    # row on the shard that owns the token, so nothing has to be gathered.
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert out.sharding.spec == x.sharding.spec

    want = _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating, topk=topk)
    error = _relative_l2(out, want)
    assert error < batch_bound, f"relative L2 {error:.4f} past the {batch_bound} band"
    worst = _worst_token_relative_l2(out, want)
    assert worst < token_bound, (
        f"worst token's relative error {worst:.4f} past the {token_bound} "
        f"per-token band; the batch norm was {error:.4f}, so this is a few "
        f"tokens rather than the whole batch"
    )

    def roll(a):
        return None if a is None else jnp.roll(a, 1, axis=0)

    rotated = _dense_reference(
        mesh, x, roll(w1), roll(w2), roll(w1_scale), roll(w2_scale), gating, topk=topk
    )
    assert _relative_l2(out, rotated) > ROTATED_CONTROL_FACTOR * error
    return error, worst


@pytest.mark.parametrize("weight_format", [WeightFormat.FP8, WeightFormat.BF16])
def test_small_shape_tracks_a_dense_reference(weight_format):
    """The layer end to end at a small but structurally complete shape."""
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        0,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=weight_format,
    )
    x, gating = _make_inputs(
        mesh,
        1,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    quantized = weight_format == WeightFormat.FP8
    _check_against_the_reference(
        mesh,
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=SMALL["topk"],
        batch_bound=FP8_RELATIVE_BOUND if quantized else BF16_RELATIVE_BOUND,
        token_bound=FP8_TOKEN_BOUND if quantized else BF16_TOKEN_BOUND,
    )


def _make_hash_ids(mesh, seed, *, tokens, topk, e_total):
    """Per-token expert ids, as DeepSeek-V4's table would hand them over.

    Distinct within a row, because the real table is: a repeated id would
    weight one expert twice, which is a routing question rather than a
    kernel one and would make the reference the only thing under test.
    """
    rng = np.random.default_rng(seed)
    ids = np.stack(
        [rng.choice(e_total, size=topk, replace=False) for _ in range(tokens)]
    ).astype(np.int32)
    return jax.device_put(ids, NamedSharding(mesh, P(AXIS)))


def test_hash_routing_serves_the_ids_it_was_given():
    """DeepSeek-V4's bottom layers select by a table indexed on token id, not
    by the gate. The gate is still scored -- the weights are its scores at the
    given ids -- so the only thing that changes is which experts are chosen.

    The control is the same layer with the ids withheld: if the operand were
    ignored the kernel would fall back to its own top-k, and the two runs
    would agree.
    """
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        30,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        31,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    hash_ids = _make_hash_ids(
        mesh, 32, tokens=SMALL["tokens"], topk=SMALL["topk"], e_total=SMALL["e_total"]
    )
    common = dict(
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )
    out = fused_ep_moe_v2(
        x, w1, w2, w1_scale, w2_scale, gating, hash_topk_ids=hash_ids, **common
    )
    assert out.shape == x.shape
    assert out.sharding.spec == x.sharding.spec

    want = _dense_reference(
        mesh,
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=SMALL["topk"],
        hash_ids=hash_ids,
    )
    error = _relative_l2(out, want)
    assert error < FP8_RELATIVE_BOUND, f"relative L2 {error:.4f} past the band"
    worst = _worst_token_relative_l2(out, want)
    assert worst < FP8_TOKEN_BOUND, f"worst token {worst:.4f} past the band"

    selected = fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating, **common)
    assert _relative_l2(selected, want) > ROTATED_CONTROL_FACTOR * error


# Rows each shard's local experts receive, in visit order, and the token
# count that gives a shard their sum. Every expert issues its successor's
# first rows from its own last tile, from an index window started when its
# last group became current. The first order is one tile, two, one, one:
# it crosses one-tile -> multi-tile both ways and ends on a one-tile expert.
# The second is one, one, two, three: a multi-tile -> multi-tile pair, an
# expert of three tiles, and a multi-tile expert with no successor.
MIXED_EXPERT_ROWS = (24, 136, 24, 72)
MIXED_TOKENS = 512
MIXED_CASES = [
    (MIXED_EXPERT_ROWS, MIXED_TOKENS),
    ((56, 24, 136, 296), 1024),
]


def _make_ids_with_expert_rows(mesh, seed, *, rows_per_local, tokens, topk, e_total):
    """Hash ids that give local expert j of every shard rows_per_local[j] rows.

    Each expert's slots are a contiguous run of the flat list, dealt out a
    token at a time, so a run shorter than `tokens` never repeats an id
    within a token.
    """
    g_local = len(rows_per_local)
    counts = np.tile(rows_per_local, e_total // g_local)
    assert counts.sum() == tokens * topk and counts.max() <= tokens
    flat = np.repeat(np.arange(e_total, dtype=np.int32), counts)
    ids = flat.reshape(topk, tokens).T
    ids = ids[np.random.default_rng(seed).permutation(tokens)]
    np.testing.assert_array_equal(np.bincount(ids.ravel(), minlength=e_total), counts)
    assert all(len(set(row)) == topk for row in ids)
    return jax.device_put(ids, NamedSharding(mesh, P(AXIS)))


@pytest.mark.parametrize("rows_per_local,tokens", MIXED_CASES)
def test_mixed_one_tile_and_multi_tile_experts_track_the_reference(
    rows_per_local, tokens
):
    """One-tile and multi-tile experts next to each other match the reference.

    Random routing gives every expert about the same rows, so the ids are
    picked by hand to get these mixes.
    """
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        33,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh, 34, tokens=tokens, hidden=SMALL["hidden"], e_total=SMALL["e_total"]
    )
    hash_ids = _make_ids_with_expert_rows(
        mesh,
        35,
        rows_per_local=rows_per_local,
        tokens=tokens,
        topk=SMALL["topk"],
        e_total=SMALL["e_total"],
    )
    out = fused_ep_moe_v2(
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        hash_topk_ids=hash_ids,
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )

    def reference(roll):
        def r(a):
            return jnp.roll(a, roll, axis=0)

        return _dense_reference(
            mesh,
            x,
            r(w1),
            r(w2),
            r(w1_scale),
            r(w2_scale),
            gating,
            topk=SMALL["topk"],
            hash_ids=hash_ids,
        )

    want = reference(0)
    error = _relative_l2(out, want)
    assert error < FP8_RELATIVE_BOUND, f"relative L2 {error:.4f} past the band"
    worst = _worst_token_relative_l2(out, want)
    assert worst < FP8_TOKEN_BOUND, f"worst token {worst:.4f} past the band"
    assert _relative_l2(out, reference(1)) > ROTATED_CONTROL_FACTOR * error


def test_sharded_routing_plan_is_bit_exact_with_the_replicated_plan():
    """Plan decomposition changes only how the kernel operands are built."""
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        20,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        21,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    common = dict(
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )
    replicated = fused_ep_moe_v2(
        x, w1, w2, w1_scale, w2_scale, gating, sharded_plan=False, **common
    )
    sharded = fused_ep_moe_v2(
        x, w1, w2, w1_scale, w2_scale, gating, sharded_plan=True, **common
    )
    np.testing.assert_array_equal(np.asarray(sharded), np.asarray(replicated))


@pytest.mark.parametrize("sharded_plan", [False, True], ids=["replicated", "sharded"])
@pytest.mark.parametrize(
    "shape,shared_weights",
    [
        pytest.param(dict(SMALL), None, id="small"),
        # Uniform routing gives every expert a few dozen rows, so each tile of
        # the 256-row build runs its shortest rung (or, at prod_t16384, the
        # build has the full height only). These loads put experts on the
        # 128-, 192- and 256-row rungs too (checked below).
        pytest.param(
            dict(SMALL, expert_rows={1: 96, 10: 160, 19: 224}),
            None,
            id="small_every_rung",
        ),
        # The production weights are 6.4GB, so these shapes run against the
        # module's one build of them rather than making their own.
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[0]), "production_weights", id="prod_t2048"
        ),
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[1]), "production_weights", id="prod_t16384"
        ),
    ],
)
def test_256_row_tiles_track_the_128_row_build(
    request, sharded_plan, shape, shared_weights
):
    """A taller tile changes how rows are grouped, not what each row computes.

    Tiles above 128 rows need the scale mirrors stored at 128 lanes
    (host.mirror_rows_per_tile); without that no such build compiles. Every
    row's two matmuls, its scales and its wire quantization are its own, so the layer
    has to stay inside the fp8 bands of the dense reference at both heights
    and match the 128-row build exactly. The lowered programs
    are checked to carry each height's kernel, so the comparison is real.

    A tile shorter than 256 rows stages its scales as whole 128-lane rows and
    part of the next one, so small_every_rung puts experts on each rung of the
    256-row build's ladder, and the test checks its loads still do.
    """
    mesh = _mesh()
    if shared_weights is not None:
        w1, w2, w1_scale, w2_scale = request.getfixturevalue(shared_weights)
    else:
        w1, w2, w1_scale, w2_scale = _make_weights(
            mesh,
            80,
            hidden=shape["hidden"],
            inter=shape["inter"],
            e_total=shape["e_total"],
            weight_format=WeightFormat.FP8,
        )
    x, gating = _make_inputs(
        mesh,
        81,
        tokens=shape["tokens"],
        hidden=shape["hidden"],
        e_total=shape["e_total"],
    )
    if "expert_rows" in shape:
        gating = _routed_gating(
            mesh,
            82,
            tokens=shape["tokens"],
            e_total=shape["e_total"],
            topk=shape["topk"],
            expert_rows=shape["expert_rows"],
        )
        # The 256-row build has its whole ladder at this shape, and some
        # expert's slab rows (each source's run rounded up to ROWBLK) end on
        # each rung.
        rungs = kernel.tile_height_rungs(
            256,
            host.ragged_stride_bound(
                shape["tokens"], shape["topk"], shape["e_total"], 256
            ),
            shape["e_total"] // EP,
            EP,
        )
        assert rungs == (64, 128, 192, 256), rungs
        ids = np.argsort(-np.asarray(gating), axis=1)[:, : shape["topk"]]
        runs = np.stack(
            [
                np.bincount(i.ravel(), minlength=shape["e_total"])
                for i in np.split(ids, EP)
            ]
        )
        rows = (-(-runs // host.ROWBLK) * host.ROWBLK).sum(axis=0)
        assert {min(r for r in rungs if r >= n) for n in rows} == set(rungs), rows
    args = (x, w1, w2, w1_scale, w2_scale, gating)
    outs, names = {}, {}
    for capacity in (128, 256):

        def run(*a, capacity=capacity):
            return fused_ep_moe_v2(
                *a,
                topk=shape["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=capacity,
                weight_format=WeightFormat.FP8,
                sharded_plan=sharded_plan,
            )

        names[capacity] = set(
            re.findall(
                r"fused_ep_moe_v2[A-Za-z0-9_]*?_c(\d+)_nb\d+",
                jax.jit(run).lower(*args).as_text(),
            )
        )
        outs[capacity] = run(*args)
    assert names == {128: {"128"}, 256: {"256"}}, names
    want = _dense_reference(
        mesh, x, w1, w2, w1_scale, w2_scale, gating, topk=shape["topk"]
    )
    for capacity, out in outs.items():
        assert _relative_l2(out, want) < FP8_RELATIVE_BOUND, capacity
        assert _worst_token_relative_l2(out, want) < FP8_TOKEN_BOUND, capacity
    # Each row's matmuls accumulate in the same order at both heights, so the
    # taller tile reproduces the 128-row build bit for bit (measured on v7x at
    # every shape and plan above).
    np.testing.assert_array_equal(np.asarray(outs[256]), np.asarray(outs[128]))


@pytest.mark.parametrize("sharded_plan", [False, True], ids=["replicated", "sharded"])
@pytest.mark.parametrize(
    "shape,shared_weights",
    [
        pytest.param(dict(SMALL), None, id="small"),
        # The production weights are 6.4GB, so this shape runs against the
        # module's one build of them rather than making its own.
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[0]), "production_weights", id="prod_t2048"
        ),
    ],
)
def test_slab_scatter_set_and_add_are_bit_exact(
    request, monkeypatch, sharded_plan, shape, shared_weights
):
    """The slab tables' scatter form changes how they are built, not the layer.

    The chip picks the form (host.slab_tables_on_tensorcore); forcing each one
    here runs both on whatever chip this is, so the form this chip does NOT
    pick is checked as well. Both tables are held on their scatter paths, where
    the form applies, rather than the sort and the lookup the layer prefers.
    The two forced builds must lower to different programs, so the agreement is
    not vacuous: were the form ever dropped from the layer's shard_map cache
    key, the second build would reuse the first one's body and fail here.
    """
    monkeypatch.setattr(layer, "scale_slab_fits", lambda _n: False)
    monkeypatch.setattr(layer, "token_gather_sort_fits", lambda *_a, **_k: False)
    mesh = _mesh()
    if shared_weights is not None:
        w1, w2, w1_scale, w2_scale = request.getfixturevalue(shared_weights)
    else:
        w1, w2, w1_scale, w2_scale = _make_weights(
            mesh,
            40,
            hidden=shape["hidden"],
            inter=shape["inter"],
            e_total=shape["e_total"],
            weight_format=WeightFormat.FP8,
        )
    x, gating = _make_inputs(
        mesh,
        41,
        tokens=shape["tokens"],
        hidden=shape["hidden"],
        e_total=shape["e_total"],
    )

    def make_run():
        # A fresh function per build: jax.jit caches the trace by function, and
        # the form is read while tracing, so reusing one `run` would lower the
        # first build's program again.
        def run(*args):
            return fused_ep_moe_v2(
                *args,
                topk=shape["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=WeightFormat.FP8,
                sharded_plan=sharded_plan,
            )

        return run

    args = (x, w1, w2, w1_scale, w2_scale, gating)
    outs, programs = {}, {}
    for by_set in (False, True):
        monkeypatch.setattr(
            layer, "slab_tables_on_tensorcore", lambda by_set=by_set: by_set
        )
        run = make_run()
        programs[by_set] = jax.jit(run).lower(*args).as_text()
        outs[by_set] = np.asarray(run(*args))
    assert programs[True] != programs[False]
    np.testing.assert_array_equal(outs[True], outs[False])


@pytest.mark.parametrize("sharded_plan", [False, True], ids=["replicated", "sharded"])
@pytest.mark.parametrize(
    "shape,shared_weights",
    [
        pytest.param(dict(SMALL), None, id="small"),
        # The production weights are 6.4GB, so these shapes run against the
        # module's one build of them rather than making their own.
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[0]), "production_weights", id="prod_t2048"
        ),
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[1]), "production_weights", id="prod_t16384"
        ),
    ],
)
def test_scale_slab_lookup_and_scatter_are_bit_exact(
    request, monkeypatch, sharded_plan, shape, shared_weights
):
    """The scale slab built by lookup gives the layer the scatter's output.

    The lookup leaves token 0's scale on the ROWBLK padding rows where the
    scatter leaves zero; the kernel commits neither, so the output is the
    same bits. The scatter form is forced by declaring the scales too large
    for SMEM; the TensorCore builds are forced on, so the lookup runs on a
    chip whose SparseCore would otherwise keep the scatter-add. The lowered
    programs must hold the lookup kernel exactly where it is forced on, so
    the agreement is not vacuous: were the choice ever dropped from the
    layer's shard_map cache key, the second build would reuse the first one's
    body and fail here.
    """
    monkeypatch.setattr(layer, "slab_tables_on_tensorcore", lambda: True)
    # At prefill sizes the run copy builds both tables instead; it is held off
    # so the lookup and the scatter are what is compared.
    monkeypatch.setattr(layer, "slab_run_copy_pays", lambda *_a, **_k: False)
    mesh = _mesh()
    if shared_weights is not None:
        w1, w2, w1_scale, w2_scale = request.getfixturevalue(shared_weights)
    else:
        w1, w2, w1_scale, w2_scale = _make_weights(
            mesh,
            50,
            hidden=shape["hidden"],
            inter=shape["inter"],
            e_total=shape["e_total"],
            weight_format=WeightFormat.FP8,
        )
    x, gating = _make_inputs(
        mesh,
        51,
        tokens=shape["tokens"],
        hidden=shape["hidden"],
        e_total=shape["e_total"],
    )

    def make_run():
        # A fresh function per build: jax.jit caches the trace by function, and
        # the form is read while tracing, so reusing one `run` would lower the
        # first build's program again.
        def run(*args):
            return fused_ep_moe_v2(
                *args,
                topk=shape["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=WeightFormat.FP8,
                sharded_plan=sharded_plan,
            )

        return run

    args = (x, w1, w2, w1_scale, w2_scale, gating)
    outs, has_lookup = {}, {}
    for fits in (True, False):
        monkeypatch.setattr(layer, "scale_slab_fits", lambda _n, fits=fits: fits)
        run = make_run()
        program = jax.jit(run).lower(*args).as_text()
        has_lookup[fits] = "fused_ep_moe_v2_scale_slab" in program
        outs[fits] = np.asarray(run(*args))
    assert has_lookup == {True: True, False: False}, has_lookup
    np.testing.assert_array_equal(outs[True], outs[False])


@pytest.mark.parametrize("sharded_plan", [False, True], ids=["replicated", "sharded"])
@pytest.mark.parametrize(
    "shape,shared_weights",
    [
        pytest.param(dict(SMALL), None, id="small"),
        # The production weights are 6.4GB, so these shapes run against the
        # module's one build of them rather than making their own.
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[0]), "production_weights", id="prod_t2048"
        ),
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[1]), "production_weights", id="prod_t16384"
        ),
    ],
)
def test_token_gather_sort_and_scatter_are_bit_exact(
    request, monkeypatch, sharded_plan, shape, shared_weights
):
    """The token-gather table built by sort gives the layer the scatter's output.

    The scatter form is forced by declaring the sort keys too wide for int32;
    the TensorCore builds are forced on, as in the lookup test, and the run
    copy is held off, so the sort and the scatter are what is compared. The
    two forced builds must lower to different programs, so the agreement is not
    vacuous: were the choice ever dropped from the layer's shard_map cache key,
    the second build would reuse the first one's body and fail here.
    """
    monkeypatch.setattr(layer, "slab_tables_on_tensorcore", lambda: True)
    monkeypatch.setattr(layer, "slab_run_copy_pays", lambda *_a, **_k: False)
    mesh = _mesh()
    if shared_weights is not None:
        w1, w2, w1_scale, w2_scale = request.getfixturevalue(shared_weights)
    else:
        w1, w2, w1_scale, w2_scale = _make_weights(
            mesh,
            60,
            hidden=shape["hidden"],
            inter=shape["inter"],
            e_total=shape["e_total"],
            weight_format=WeightFormat.FP8,
        )
    x, gating = _make_inputs(
        mesh,
        61,
        tokens=shape["tokens"],
        hidden=shape["hidden"],
        e_total=shape["e_total"],
    )

    def make_run():
        # A fresh function per build: jax.jit caches the trace by function, and
        # the form is read while tracing, so reusing one `run` would lower the
        # first build's program again.
        def run(*args):
            return fused_ep_moe_v2(
                *args,
                topk=shape["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=WeightFormat.FP8,
                sharded_plan=sharded_plan,
            )

        return run

    args = (x, w1, w2, w1_scale, w2_scale, gating)
    outs, programs = {}, {}
    for fits in (True, False):
        monkeypatch.setattr(
            layer, "token_gather_sort_fits", lambda *_a, fits=fits, **_k: fits
        )
        run = make_run()
        programs[fits] = jax.jit(run).lower(*args).as_text()
        outs[fits] = np.asarray(run(*args))
    assert programs[True] != programs[False]
    np.testing.assert_array_equal(outs[True], outs[False])


@pytest.mark.parametrize(
    "shape,shared_weights",
    [
        pytest.param(dict(SMALL), None, id="small"),
        # The production weights are 6.4GB, so these shapes run against the
        # module's one build of them rather than making their own.
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[0]), "production_weights", id="prod_t2048"
        ),
        pytest.param(
            dict(PROD, tokens=PROD_TOKENS[1]), "production_weights", id="prod_t16384"
        ),
    ],
)
def test_slab_run_copy_and_sort_lookup_are_bit_exact(
    request, monkeypatch, shape, shared_weights
):
    """Both slab tables by run copy give the layer the sort + lookup's output.

    The run copy changes the routing payload (expert-sorted token numbers and a
    scale per pair) and builds both tables from it; the tables differ from the
    lookup's only on the ROWBLK padding rows, which the kernel never commits.
    The TensorCore builds are forced on and the run copy's gate is forced each
    way so each shape runs both forms; the lowered programs are checked to
    really differ, so the agreement is not vacuous.
    """
    monkeypatch.setattr(layer, "slab_tables_on_tensorcore", lambda: True)
    mesh = _mesh()
    if shared_weights is not None:
        w1, w2, w1_scale, w2_scale = request.getfixturevalue(shared_weights)
    else:
        w1, w2, w1_scale, w2_scale = _make_weights(
            mesh,
            70,
            hidden=shape["hidden"],
            inter=shape["inter"],
            e_total=shape["e_total"],
            weight_format=WeightFormat.FP8,
        )
    x, gating = _make_inputs(
        mesh,
        71,
        tokens=shape["tokens"],
        hidden=shape["hidden"],
        e_total=shape["e_total"],
    )

    def make_run():
        # A fresh function per build: jax.jit caches the trace by function, and
        # the threshold is read while tracing, so reusing one `run` would lower
        # the first build's program again.
        def run(*args):
            return fused_ep_moe_v2(
                *args,
                topk=shape["topk"],
                renormalize=True,
                mesh=mesh,
                capacity=CAPACITY,
                weight_format=WeightFormat.FP8,
                sharded_plan=True,
            )

        return run

    args = (x, w1, w2, w1_scale, w2_scale, gating)
    outs, has_copy = {}, {}
    for on in (False, True):
        monkeypatch.setattr(layer, "slab_run_copy_pays", lambda *_a, on=on, **_k: on)
        layer._LAYER_SM_CACHE.clear()
        jax.clear_caches()
        run = make_run()
        has_copy[on] = (
            "fused_ep_moe_v2_slab_runs" in jax.jit(run).lower(*args).as_text()
        )
        outs[on] = np.asarray(run(*args))
    layer._LAYER_SM_CACHE.clear()
    jax.clear_caches()
    assert has_copy == {False: False, True: True}, has_copy
    np.testing.assert_array_equal(outs[True], outs[False])


@pytest.mark.parametrize("wire_dtype", [FP8, jnp.bfloat16], ids=["fp8", "bf16"])
def test_pallas_combine_kernel_isolates_nan_arrivals_on_unrouted_padded_tokens(
    wire_dtype,
):
    """Pallas `_build_combine_kernel` keeps NaN arrival rows on padded rows.

    In `fused_ep_moe_v2`, a freshly allocated `arrivals` buffer may contain
    zeroes on TPU rather than `NaN`. This single-chip test calls
    `layer._combine_arrivals` directly at `t_local = COMBINE_MIN_TOKENS` (256),
    where `kernel.combine_step_tokens(t_local) > 0` dispatches to the Pallas
    DMA/reduce combine kernel (`_build_combine_kernel`), with `arrivals` and
    `arrival_scales` explicitly poisoned with `NaN` at all sentinel positions,
    followed by the layer's `jnp.where(row_routes, combined, jnp.zeros_like(combined))`.
    """
    t_local = kernel.COMBINE_MIN_TOKENS
    assert kernel.combine_step_tokens(t_local) > 0
    topk, lane_blocks, lanes, tile_m = 4, 2, 128, 128
    hidden = lane_blocks * lanes
    block = host.routing_block(t_local, topk)
    recv_rows = block + 64
    active = t_local // 2
    padded = t_local - active

    with jax.default_device(jax.devices()[0]):
        arrivals_f32 = np.ones((recv_rows, lane_blocks, lanes), dtype=np.float32)
        arrivals_f32[:block] = np.nan
        arrivals = jnp.asarray(arrivals_f32).astype(wire_dtype)
        assert np.isnan(np.asarray(arrivals[:block].astype(jnp.float32))).all()

        scales_f32 = np.ones((4, tile_m), dtype=np.float32)
        scales_f32[: block // tile_m, :] = np.nan
        arrival_scales = jnp.asarray(scales_f32)

        pos = np.full((t_local, topk), block, dtype=np.int32)
        mirror_pos = np.full((t_local, topk), block, dtype=np.int32)
        topk_weights = np.full((t_local, topk), 1.0 / topk, dtype=np.float32)

        # Sentinel pairs receive their rank in `[0, block)` within each routing block.
        sentinel_ranks = (
            np.arange(padded * topk, dtype=np.int32).reshape(padded, topk) % block
        )
        pos[active:] = sentinel_ranks
        mirror_pos[active:] = sentinel_ranks
        topk_weights[active:] = 0.0
        row_routes = jnp.asarray([[True]] * active + [[False]] * padded)

        combined = layer._combine_arrivals(
            arrivals,
            arrival_scales,
            jnp.asarray(pos),
            jnp.asarray(mirror_pos),
            jnp.asarray(topk_weights),
            jnp.bfloat16,
        )
        out = np.asarray(jnp.where(row_routes, combined, jnp.zeros_like(combined)))

    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:active].astype(np.float32), 1.0)
    np.testing.assert_array_equal(
        out[active:].astype(np.float32),
        np.zeros((padded, hidden), dtype=np.float32),
    )


@pytest.mark.parametrize(
    "t_local,uses_kernel",
    [(12288, True), (16384, False)],
    ids=["t12288_kernel", "t16384_xla"],
)
def test_combine_arrivals_at_prefill_buckets_compiles_and_matches_reference(
    t_local, uses_kernel
):
    """The #3694 prefill buckets on the chip (top-10, hidden 4096).

    12288 x 10 (960 KiB of pos and coef tables) stays on the Pallas combine
    kernel and must compile in this chip's SMEM; 16384 x 10 (1,280 KiB) takes
    the XLA epilogue. Both must match a reference on sampled tokens.
    """
    topk, lane_blocks, lanes, tile_m = 10, 32, 128, 128
    hidden = lane_blocks * lanes
    recv_rows, scale_sublanes = 8192, 64
    rng = np.random.default_rng(0)
    arrivals_f32 = rng.standard_normal((recv_rows, lane_blocks, lanes), np.float32)
    scales = rng.uniform(0.5, 1.5, (scale_sublanes, tile_m)).astype(np.float32)
    pos = rng.integers(0, recv_rows, (t_local, topk), dtype=np.int32)
    mirror_pos = rng.integers(0, scale_sublanes * tile_m, (t_local, topk), np.int32)
    weights = rng.uniform(0.0, 1.0, (t_local, topk)).astype(np.float32)

    with jax.default_device(jax.devices()[0]):
        assert kernel.combine_uses_kernel(t_local, topk) is uses_kernel
        arrivals = jnp.asarray(arrivals_f32).astype(FP8)
        combined = layer._combine_arrivals(
            arrivals,
            jnp.asarray(scales),
            jnp.asarray(pos),
            jnp.asarray(mirror_pos),
            jnp.asarray(weights),
            jnp.bfloat16,
        )
        out = np.asarray(combined.astype(jnp.float32))
        rows_f32 = np.asarray(arrivals.astype(jnp.float32)).reshape(recv_rows, hidden)

    assert out.shape == (t_local, hidden)
    sample = rng.choice(t_local, size=64, replace=False)
    coef = weights[sample] * scales.reshape(-1)[mirror_pos[sample]]
    expected = np.einsum("sk,skh->sh", coef, rows_f32[pos[sample]])
    np.testing.assert_allclose(out[sample], expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("sharded_plan", [False, True], ids=["replicated", "sharded"])
@pytest.mark.parametrize("scoring_fn", ["softmax", "sigmoid", "sqrtsoftplus"])
def test_padded_tokens_on_pallas_combine_and_all_padded_warmup(
    sharded_plan, scoring_fn
):
    """Padded rows become exact zero and leave active rows bit-identical.

    Runs at `t_local = 256` (`tokens = EP * 256 = 2048`), where
    `kernel.combine_step_tokens(t_local) > 0` selects the Pallas combine
    kernel (`_build_combine_kernel`) rather than the XLA epilogue. Also gates
    the warmup (`_dummy_run`) case where every row on every shard is NaN-masked.
    """
    t_local = kernel.COMBINE_MIN_TOKENS
    assert kernel.combine_step_tokens(t_local) > 0
    tokens = EP * t_local
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        24,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        25,
        tokens=tokens,
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    common = dict(
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=sharded_plan,
        scoring_fn=scoring_fn,
    )
    unmasked = np.asarray(
        fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating, **common)
    )

    # Mask the second half of each shard's rows (and all of shard 1's rows):
    # use NaN on even shards and -inf on odd shards (matching token_layout.split
    # replica padding) so both non-finite forms are verified under every scoring_fn.
    gating_host = np.asarray(gating).copy()
    valid_mask = np.ones(tokens, dtype=bool)
    for d in range(EP):
        start = d * t_local if d == 1 else d * t_local + t_local // 2
        fill = np.nan if d % 2 == 0 else -np.inf
        gating_host[start : (d + 1) * t_local, :] = fill
        valid_mask[start : (d + 1) * t_local] = False
    token_sharding = NamedSharding(mesh, P(AXIS))
    gating_masked = jax.device_put(gating_host, token_sharding)

    masked = np.asarray(
        fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating_masked, **common)
    )
    assert np.isfinite(masked).all()
    np.testing.assert_array_equal(masked[valid_mask], unmasked[valid_mask])
    np.testing.assert_array_equal(
        masked[~valid_mask],
        np.zeros_like(masked[~valid_mask]),
    )

    # All-padded batch (as in vLLM `_dummy_run` warmup): every row on every
    # shard is non-finite.
    gating_all_nan = jax.device_put(np.full_like(gating_host, np.nan), token_sharding)
    all_padded_out = np.asarray(
        fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating_all_nan, **common)
    )
    np.testing.assert_array_equal(all_padded_out, np.zeros_like(all_padded_out))


def test_nonidentity_mesh_order_relabels_after_topk_bit_exactly():
    """Physical expert placement may follow device order, not EP-rank order.

    Keep the logical router logits untouched, place each logical expert block
    on the mesh index that owns its EP rank, and relabel only selected IDs. The
    result must match the identity placement bit for bit with the sharded plan
    enabled, which is the served configuration.
    """
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        22,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        23,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    order = (0, 1, 6, 7, 2, 3, 4, 5)
    g_local = SMALL["e_total"] // EP
    expert_sharding = NamedSharding(mesh, P(AXIS))

    def place_in_mesh_order(a):
        host = np.asarray(a)
        by_rank = host.reshape((EP, g_local) + host.shape[1:])
        placed = by_rank[np.asarray(order)].reshape(host.shape)
        return jax.device_put(placed, expert_sharding)

    common = dict(
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=True,
    )
    identity = fused_ep_moe_v2(x, w1, w2, w1_scale, w2_scale, gating, **common)
    placed = tuple(place_in_mesh_order(a) for a in (w1, w2, w1_scale, w2_scale))
    remapped = fused_ep_moe_v2(x, *placed, gating, mesh_ep_ranks=order, **common)
    np.testing.assert_array_equal(np.asarray(remapped), np.asarray(identity))


@pytest.mark.parametrize(
    "tensorcore_tables", [False, True], ids=["chip_tables", "tensorcore_tables"]
)
@pytest.mark.parametrize("sharded_plan", [False, True])
def test_split_stages_with_a_reordered_gather_are_bit_exact(
    monkeypatch, sharded_plan, tensorcore_tables
):
    """route, a caller-side gather in another order, compute == fused.

    The caller gathers the wire rows in an order of its own -- block i of q_g
    holds mesh index order[i]'s rows -- and tells compute where each mesh
    index's block landed. Nothing else about the layer may change.

    With the TensorCore table builds forced on, the scale slab is a lookup
    through the token-gather table, and it has to read the table in mesh
    order -- the order the scales were gathered in -- not in q_g's order.
    """
    if tensorcore_tables:
        monkeypatch.setattr(layer, "slab_tables_on_tensorcore", lambda: True)
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        24,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        25,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    common = dict(
        topk=SMALL["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=sharded_plan,
    )
    weights = (w1, w2, w1_scale, w2_scale)
    fused = fused_ep_moe_v2(x, *weights, gating, **common)

    q, router_weights, payload = fused_ep_moe_v2(
        x, *weights, gating, stage="route", **common
    )
    order = (0, 1, 6, 7, 2, 3, 4, 5)
    blocks = np.asarray(q).reshape((EP, -1) + q.shape[1:])
    gathered = blocks[np.asarray(order)].reshape(q.shape)
    # Every shard holds the whole gather, as the caller's collective leaves it.
    q_g = jax.device_put(np.concatenate([gathered] * EP), NamedSharding(mesh, P(AXIS)))
    positions = tuple(order.index(i) for i in range(EP))
    split = fused_ep_moe_v2(
        x,
        *weights,
        gating,
        stage="compute",
        routed=(q_g, router_weights, payload),
        gather_positions=positions,
        **common,
    )
    np.testing.assert_array_equal(np.asarray(split), np.asarray(fused))

    # The control: the same gather with compute told it is in mesh order.
    wrong = fused_ep_moe_v2(
        x,
        *weights,
        gating,
        stage="compute",
        routed=(q_g, router_weights, payload),
        **common,
    )
    assert not np.array_equal(np.asarray(wrong), np.asarray(fused))


@pytest.fixture(scope="module")
def production_weights():
    """The Qwen3.5-397B MoE weights, fp8 e4m3 with per-channel scales.

    6.4GB over the eight devices, so they are built once and both batch
    shapes below run against them.
    """
    mesh = _mesh()
    weights = _make_weights(
        mesh,
        7,
        hidden=PROD["hidden"],
        inter=PROD["inter"],
        e_total=PROD["e_total"],
        weight_format=WeightFormat.FP8,
    )
    jax.block_until_ready(weights)
    yield weights
    del weights


@pytest.mark.parametrize("tokens", PROD_TOKENS)
def test_production_shape_tracks_a_dense_reference(production_weights, tokens):
    """Qwen3.5-397B's MoE block at the decode and prefill batch shapes."""
    mesh = _mesh()
    w1, w2, w1_scale, w2_scale = production_weights
    x, gating = _make_inputs(
        mesh, 2, tokens=tokens, hidden=PROD["hidden"], e_total=PROD["e_total"]
    )
    _check_against_the_reference(
        mesh,
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=PROD["topk"],
        batch_bound=FP8_RELATIVE_BOUND,
        token_bound=FP8_TOKEN_BOUND,
    )


def test_any_mesh_axis_name_works():
    """The kernel no longer cares what the mesh axis is called.

    Upstream reads this shard's index with ``lax.axis_index(kernel.AXIS)``,
    with AXIS a module constant, so a mesh named anything else traced the whole
    kernel and then died on "unbound axis name: d". vllm-torchtpu passes the
    index in as an i32 operand instead -- it has to, because ``lax.axis_index``
    lowers to ``partition-id`` and XLA rejects that when torch_tpu recompiles
    the exported module. The layer's own collectives take their axis from the
    mesh it is handed, so nothing is left that hardcodes a name. Held here so
    the freedom is stated, and so a revert to axis_index fails loudly.
    """
    mesh = Mesh(np.asarray(jax.devices()[:EP]), axis_names=("ep",))
    w1, w2, w1_scale, w2_scale = _make_weights(
        mesh,
        0,
        hidden=SMALL["hidden"],
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=WeightFormat.FP8,
    )
    x, gating = _make_inputs(
        mesh,
        1,
        tokens=SMALL["tokens"],
        hidden=SMALL["hidden"],
        e_total=SMALL["e_total"],
    )
    _check_against_the_reference(
        mesh,
        x,
        w1,
        w2,
        w1_scale,
        w2_scale,
        gating,
        topk=SMALL["topk"],
        batch_bound=FP8_RELATIVE_BOUND,
        token_bound=FP8_TOKEN_BOUND,
    )


def test_fp4_block512_tracks_dense_reference_and_routing_plan():
    """FP4 block scales and FP8 transport, including a single-block W2."""
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(
        mesh, 42, hidden=1024, inter=512, e_total=32, rhs_qb=512
    )
    x, gating = _make_inputs(mesh, 43, tokens=256, hidden=1024, e_total=32)
    common = dict(
        topk=4,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=512,
    )
    out = fused_ep_moe_v2(x, *weights, *scales, gating, sharded_plan=True, **common)
    replicated = fused_ep_moe_v2(
        x, *weights, *scales, gating, sharded_plan=False, **common
    )
    np.testing.assert_array_equal(np.asarray(out), np.asarray(replicated))
    ref = _dense_reference(mesh, x, *decoded, None, None, gating, topk=4)
    error = _relative_l2(out, ref)
    worst = _worst_token_relative_l2(out, ref)
    print(f"FP4 block512 relative_l2={error} worst_token={worst}")
    assert error < FP8_RELATIVE_BOUND
    assert worst < FP8_TOKEN_BOUND
    wrong = _dense_reference(
        mesh,
        x,
        jnp.roll(decoded[0], 1, axis=0),
        jnp.roll(decoded[1], 1, axis=0),
        None,
        None,
        gating,
        topk=4,
    )
    assert _relative_l2(out, wrong) > ROTATED_CONTROL_FACTOR * error


# 4 / 2 blocks over 4 local experts, and DeepSeek-V4's 14 / 6 blocks over 8:
# the served 14-row table is one full sublane tile and a partial one.
SECOND_MINOR_SHAPES = [
    dict(hidden=2048, inter=1024, rhs_qb=512, e_total=32),
    dict(hidden=896, inter=384, rhs_qb=64, e_total=64),
]


@pytest.mark.parametrize("shape", SECOND_MINOR_SHAPES)
@pytest.mark.parametrize(
    "w1_minor,w2_minor", [(True, False), (False, True), (True, True)]
)
def test_fp4_expert_second_minor_scale_reads_are_bit_exact(shape, w1_minor, w2_minor):
    """Reading either scale table expert-second-minor changes only its
    addressing."""
    mesh = _mesh()
    weights, scales, _ = _make_fp4_weights(
        mesh,
        44,
        hidden=shape["hidden"],
        inter=shape["inter"],
        e_total=shape["e_total"],
        rhs_qb=shape["rhs_qb"],
    )
    x, gating = _make_inputs(
        mesh, 45, tokens=256, hidden=shape["hidden"], e_total=shape["e_total"]
    )
    common = dict(
        topk=4,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=shape["rhs_qb"],
        sharded_plan=True,
    )
    default_layout = fused_ep_moe_v2(x, *weights, *scales, gating, **common)
    second_minor = fused_ep_moe_v2(
        x,
        *weights,
        *scales,
        gating,
        w1_scales_expert_second_minor=w1_minor,
        w2_scales_expert_second_minor=w2_minor,
        **common,
    )
    np.testing.assert_array_equal(np.asarray(second_minor), np.asarray(default_layout))


@pytest.mark.parametrize("flag", [True, False])
def test_fp4_expert_second_minor_scales_reach_the_kernel_uncopied(flag):
    """Scale tables staged {2,0,1} reach the kernel with no copy when the
    flags say so. Without them XLA copies both into the default layout, which
    shows the check can see a copy."""
    from jax._src.layout import Format, Layout

    shape = SECOND_MINOR_SHAPES[1]
    hidden, inter, qb, e_total = (
        shape[k] for k in ("hidden", "inter", "rhs_qb", "e_total")
    )
    mesh = _mesh()
    weights, scales, _ = _make_fp4_weights(
        mesh, 46, hidden=hidden, inter=inter, e_total=e_total, rhs_qb=qb
    )
    x, gating = _make_inputs(mesh, 47, tokens=256, hidden=hidden, e_total=e_total)
    common = dict(
        topk=4,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=qb,
        sharded_plan=True,
        w1_scales_expert_second_minor=flag,
        w2_scales_expert_second_minor=flag,
    )
    # [E, blocks, N] with minor-to-major {2,0,1}, as torch_tpu stages it.
    second_minor = Format(
        Layout(major_to_minor=(1, 0, 2)), NamedSharding(mesh, P(AXIS))
    )
    fn = jax.jit(
        lambda x, w1, w2, s1, s2, g: fused_ep_moe_v2(x, w1, w2, s1, s2, g, **common),
        in_shardings=(None, None, None, second_minor, second_minor, None),
    )
    hlo = fn.lower(x, *weights, *scales, gating).compile().as_text()
    # Each shard's tables, in either order of their two major dimensions.
    local = e_total // EP
    tables = [
        (local, hidden // qb, 2 * inter),
        (local, inter // qb, hidden),
    ]
    shapes = "|".join(
        ",".join(map(str, dims))
        for e, b, n in tables
        for dims in ((e, b, n), (b, e, n))
    )
    producers = [
        line.strip()
        for line in hlo.splitlines()
        if re.search(rf"= f32\[({shapes})\]", line)
        and not re.search(r"\b(parameter|bitcast|get-tuple-element)\(", line)
    ]
    assert (producers == []) is flag, producers


# --------------------------------------------------------------------------- #
# The weight-DMA schedule.
#
# How far ahead of the computing expert a weight refill goes out is not an
# argument to the layer: the build asks `host.fit_weight_slots` for the
# deepest runway this shape's VMEM affords, and every shape small enough to
# test here affords the ceiling. So the schedules below are reached by
# capping the fitter, which is the one line of the build that reads it.
# --------------------------------------------------------------------------- #


@pytest.fixture
def at_prefetch_depth(monkeypatch):
    """Build at a named weight-prefetch depth, and report what was built.

    Yields `depth -> observed`, where `observed` fills with the schedule each
    build resolved to. Reporting it is not decoration: a cap that stopped
    biting would leave every case running one schedule and agreeing trivially.

    BOTH caches have to go, and the second one is the trap. `_BUILD_CACHE`
    holds the kernel; `_LAYER_SM_CACHE` holds a shard_map over a `local_fn`
    that CLOSES OVER the kernel built on the call that filled it, and its key
    is the layer's arguments -- which the schedule is not one of, because the
    build derives it from the device. Clearing only the first rebuilds the
    kernel, logs it and fills `observed`, and then runs the cached shard_map's
    older kernel: every depth passes, and a depth-zero-only fault is invisible.
    """
    observed = []
    # Captured once, before the first cap: taken inside `at` it would be the
    # PREVIOUS cap on the second use, and the two depths would compose.
    real = host.fit_weight_slots

    def clear():
        kernel._BUILD_CACHE.clear()
        layer._LAYER_SM_CACHE.clear()

    def at(depth):
        def capped(*args, **kwargs):
            slots = real(*args, **{**kwargs, "max_distance": depth})
            observed.append(slots)
            return slots

        monkeypatch.setattr(host, "fit_weight_slots", capped)
        observed.clear()
        clear()
        return observed

    yield at
    clear()


# An fp4 shape built for the streamed schedule: hidden and inter each span
# several `rhs_qb` contraction blocks, so a slab's wait lands in a block that
# is not also the one that retires it; and the batch puts 2048 * 4 / 32 = 256
# rows on an expert against a tile of CAPACITY, so a visit runs more than one
# tile and the wait has to be the first tile's alone.
STREAM = dict(hidden=1024, inter=512, e_total=32, topk=4, tokens=2048)
STREAM_QB = 256
STREAM_DEPTHS = (2, 1, 0)


def test_fp4_agrees_across_every_weight_schedule(at_prefetch_depth):
    """The three fp4 weight schedules compute the same thing, bit for bit.

    The depth changes only WHERE a weight DMA is started and waited. Depth two
    starts an expert's slabs two visits ahead into a slot of its own; depth
    zero has a single slot, streams the slabs into the blocks that read them,
    and starts the next expert's where this expert's reads of the slot retire.
    Same weights, same order, so the outputs have to agree exactly -- a refill
    landing on a slot still being read is not a small numeric drift, it is
    another expert's weights in the matmul.

    The band against the dense reference is what stops that agreement from
    being vacuous: three identically broken schedules would still match.
    """
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(
        mesh,
        44,
        hidden=STREAM["hidden"],
        inter=STREAM["inter"],
        e_total=STREAM["e_total"],
        rhs_qb=STREAM_QB,
    )
    x, gating = _make_inputs(
        mesh,
        45,
        tokens=STREAM["tokens"],
        hidden=STREAM["hidden"],
        e_total=STREAM["e_total"],
    )
    common = dict(
        topk=STREAM["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=STREAM_QB,
        sharded_plan=True,
    )

    outs = {}
    for depth in STREAM_DEPTHS:
        observed = at_prefetch_depth(depth)
        outs[depth] = np.asarray(
            fused_ep_moe_v2(x, *weights, *scales, gating, **common)
        )
        assert observed, "the build never asked for a weight schedule"
        assert [(s.distance, s.nbuf) for s in observed] == [
            (depth, host.nbuf_for(depth))
        ] * len(observed)

    ref = _dense_reference(mesh, x, *decoded, None, None, gating, topk=STREAM["topk"])
    for depth, out in outs.items():
        error = _relative_l2(out, ref)
        worst = _worst_token_relative_l2(out, ref)
        assert error < FP8_RELATIVE_BOUND, f"depth {depth}: relative L2 {error}"
        assert worst < FP8_TOKEN_BOUND, f"depth {depth}: worst token {worst}"
    deepest = outs[max(STREAM_DEPTHS)]
    for depth in STREAM_DEPTHS:
        np.testing.assert_array_equal(
            outs[depth],
            deepest,
            err_msg=f"depth {depth} disagrees with the deepest schedule",
        )


# The single-slot schedule at a batch whose experts each fit one tile, where
# the slabs stream in column chunks: 256 * 4 / 32 = 32 rows an expert, and
# at least two chunks per slab on each matmul.
CHUNKED = dict(hidden=3584, inter=1024, e_total=32, topk=4, tokens=256)


def test_fp4_chunked_stream_agrees_with_the_prefetch_schedule(at_prefetch_depth):
    """Column-chunk refills compute what whole-slab prefetch computes, bit for
    bit: they move only where each DMA starts and is waited."""
    assert kernel.stream_chunk(2 * CHUNKED["inter"]) > 0
    assert CHUNKED["hidden"] // kernel.stream_chunk(CHUNKED["hidden"]) == 2
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(
        mesh,
        46,
        hidden=CHUNKED["hidden"],
        inter=CHUNKED["inter"],
        e_total=CHUNKED["e_total"],
        rhs_qb=STREAM_QB,
    )
    x, gating = _make_inputs(
        mesh,
        47,
        tokens=CHUNKED["tokens"],
        hidden=CHUNKED["hidden"],
        e_total=CHUNKED["e_total"],
    )
    common = dict(
        topk=CHUNKED["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=STREAM_QB,
        sharded_plan=True,
    )
    outs = {}
    for depth in (2, 0):
        at_prefetch_depth(depth)
        outs[depth] = np.asarray(
            fused_ep_moe_v2(x, *weights, *scales, gating, **common)
        )
    ref = _dense_reference(mesh, x, *decoded, None, None, gating, topk=CHUNKED["topk"])
    assert _relative_l2(outs[0], ref) < FP8_RELATIVE_BOUND
    np.testing.assert_array_equal(outs[0], outs[2])


# The chunk-streamed path is chosen on the EXPECTED rows per expert, so one
# skewed expert can still run several tiles on it. Two chunks per slab on
# both matmuls, and MIXED_EXPERT_ROWS puts a two-tile expert between one-tile
# ones on every shard. Hidden 3584 streams in 1792-column chunks, 4096 in
# 2048-column ones.
CHUNKED_MIXED = dict(inter=2048, e_total=32, topk=4)


@pytest.mark.parametrize(
    "width,chunk",
    [
        (6144, 2048),  # DeepSeek-V4 gate/up
        (7168, 1792),  # DeepSeek-V4 down
        (8192, 2048),  # Qwen3.8 down
        (5120, 1280),
        (1024, 1024),
        (256 * 13, 0),  # no whole chunk from 1024 to 2048 columns
        (768, 0),  # narrower than any chunk
    ],
)
def test_stream_chunk_is_the_widest_dividing_width(width, chunk):
    """The stream chunk is the widest whole number of the reader's output
    chunk that divides the slab, within the chunk bounds, or 0 for none."""
    assert kernel.stream_chunk(width) == chunk


@pytest.mark.parametrize("hidden", [3584, 4096])
@pytest.mark.parametrize("second_scale_slot", [True, False])
def test_fp4_chunked_stream_holds_a_multi_tile_expert(
    at_prefetch_depth, monkeypatch, second_scale_slot, hidden
):
    """A multi-tile expert on the chunk-streamed path reads its slab on every
    tile: chunks are waited on its first tile and refilled after its last, so
    the outputs match whole-slab prefetch bit for bit.

    Without room for a second scale slot, the scales start after the tile
    loop instead of at the visit's head; that schedule must match too.
    """
    assert (2 * CHUNKED_MIXED["inter"]) // kernel.stream_chunk(
        2 * CHUNKED_MIXED["inter"]
    ) == 2
    assert hidden // kernel.stream_chunk(hidden) == 2
    assert max(MIXED_EXPERT_ROWS) > CAPACITY
    # Only the chunk-streamed build sizes a second scale slot, so a call for
    # one shows the depth-zero build took that path rather than the old one.
    scale_slots = []
    real_estimate = host.vmem_estimate_bytes

    def estimate(*args, **kwargs):
        scale_slots.append(kwargs.get("scale_slots"))
        if kwargs.get("scale_slots") == 2 and not second_scale_slot:
            # Over any budget, so the build keeps a single scale slot. Only
            # production shapes run out of room; this small one always fits.
            return float("inf")
        return real_estimate(*args, **kwargs)

    monkeypatch.setattr(host, "vmem_estimate_bytes", estimate)
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(
        mesh,
        49,
        hidden=hidden,
        inter=CHUNKED_MIXED["inter"],
        e_total=CHUNKED_MIXED["e_total"],
        rhs_qb=STREAM_QB,
    )
    x, gating = _make_inputs(
        mesh,
        50,
        tokens=MIXED_TOKENS,
        hidden=hidden,
        e_total=CHUNKED_MIXED["e_total"],
    )
    hash_ids = _make_ids_with_expert_rows(
        mesh,
        51,
        rows_per_local=MIXED_EXPERT_ROWS,
        tokens=MIXED_TOKENS,
        topk=CHUNKED_MIXED["topk"],
        e_total=CHUNKED_MIXED["e_total"],
    )
    common = dict(
        hash_topk_ids=hash_ids,
        topk=CHUNKED_MIXED["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=STREAM_QB,
        sharded_plan=True,
    )
    outs = {}
    for depth in (2, 0):
        at_prefetch_depth(depth)
        scale_slots.clear()
        outs[depth] = np.asarray(
            fused_ep_moe_v2(x, *weights, *scales, gating, **common)
        )
        assert (2 in scale_slots) == (depth == 0), (depth, scale_slots)
    ref = _dense_reference(
        mesh,
        x,
        *decoded,
        None,
        None,
        gating,
        topk=CHUNKED_MIXED["topk"],
        hash_ids=hash_ids,
    )
    assert _relative_l2(outs[0], ref) < FP8_RELATIVE_BOUND
    np.testing.assert_array_equal(outs[0], outs[2])


# --------------------------------------------------------------------------- #
# The tile-height ladder.
#
# Serving runs two rungs (MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE=2), whose short
# rung sits at the expected rows per expert. Every other device test runs the
# default four-rung ladder, so the heights only the two-rung ladder emits are
# reached by setting the override, which the build reads.
# --------------------------------------------------------------------------- #


@pytest.fixture
def at_height_rungs(monkeypatch):
    """Build with the height-rung override at `rungs` (None for the default),
    and report the ladders built.

    Yields `rungs -> observed`, where `observed` fills with each build's
    ladder, so a case cannot pass on a ladder other than the one it names.
    Both caches are cleared for the reason `at_prefetch_depth` gives: the
    override is read at build time and is in neither cache key.
    """
    observed = []
    real = kernel.tile_height_rungs

    def recording(*args):
        ladder = real(*args)
        observed.append(ladder)
        return ladder

    monkeypatch.setattr(kernel, "tile_height_rungs", recording)

    def clear():
        kernel._BUILD_CACHE.clear()
        layer._LAYER_SM_CACHE.clear()

    def at(rungs):
        if rungs is None:
            monkeypatch.delenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", raising=False)
        else:
            monkeypatch.setenv("MOE_FUSED_EP_V2_HEIGHT_RUNGS_OVERRIDE", str(rungs))
        observed.clear()
        clear()
        return observed

    yield at
    clear()


@pytest.mark.parametrize(
    "tokens,ladder,rows_per_local",
    [
        # Expected rows per expert 24 (+2 sd) -> a 48-row short rung. One
        # expert past it runs the full-tile body, three under it the short.
        (128, (48, CAPACITY), (52, 8, 2, 2)),
        # Expected rows 16 (+2 sd) -> 32, the floor.
        (64, (32, CAPACITY), (24, 6, 2, 0)),
    ],
    ids=["short48", "short32"],
)
def test_fp4_two_rung_ladder_matches_the_default_ladder(
    at_height_rungs, tokens, ladder, rows_per_local
):
    """The two-rung ladder's short rung computes what the default ladder does.

    A rung changes only how many rows a tile's body computes, never which, so
    the outputs agree bit for bit; the dense reference keeps that agreement
    from being vacuous.
    """
    mesh = _mesh()
    weights, scales, decoded = _make_fp4_weights(
        mesh,
        46,
        hidden=STREAM["hidden"],
        inter=STREAM["inter"],
        e_total=STREAM["e_total"],
        rhs_qb=STREAM_QB,
    )
    x, gating = _make_inputs(
        mesh, 47, tokens=tokens, hidden=STREAM["hidden"], e_total=STREAM["e_total"]
    )
    hash_ids = _make_ids_with_expert_rows(
        mesh,
        48,
        rows_per_local=rows_per_local,
        tokens=tokens,
        topk=STREAM["topk"],
        e_total=STREAM["e_total"],
    )
    common = dict(
        hash_topk_ids=hash_ids,
        topk=STREAM["topk"],
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=STREAM_QB,
    )

    observed = at_height_rungs(2)
    two_rung = np.asarray(fused_ep_moe_v2(x, *weights, *scales, gating, **common))
    assert observed and all(built == ladder for built in observed), observed

    observed = at_height_rungs(None)
    default = np.asarray(fused_ep_moe_v2(x, *weights, *scales, gating, **common))
    assert observed and all(len(built) == kernel.HEIGHT_RUNGS for built in observed)

    np.testing.assert_array_equal(two_rung, default)
    want = _dense_reference(
        mesh, x, *decoded, None, None, gating, topk=STREAM["topk"], hash_ids=hash_ids
    )
    error = _relative_l2(two_rung, want)
    assert error < FP8_RELATIVE_BOUND, f"relative L2 {error:.4f} past the band"
    worst = _worst_token_relative_l2(two_rung, want)
    assert worst < FP8_TOKEN_BOUND, f"worst token {worst:.4f} past the band"
