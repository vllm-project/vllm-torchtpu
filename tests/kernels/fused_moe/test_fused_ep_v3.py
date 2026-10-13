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
"""The fused EP kernel v3 against the same dense reference, bands and
rotated-expert control the v2 port is held to (test_fused_ep_moe_v2.py),
plus padded rows and the served buckets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, WeightFormat, fused_ep_moe

pytestmark = [pytest.mark.multichip]


@pytest.fixture(autouse=True)
def _eight_devices():
    # Asked at setup, not at import: a device count taken at collection builds
    # the JAX client in the pytest parent and locks every spawned engine test
    # out of the TPU (tests/conftest.py refuses such a collection).
    if jax.device_count() < 8:
        pytest.skip("an eight-way expert-parallel collective; fewer than 8 devices")


# Rows per routing tile, as the serving path calls the kernel.
TILE_ROWS = 128

# Compiled token shapes the kernel is exercised at, as per-rank row counts.
SERVED_LOCAL_BUCKETS = (48, 64, 128, 256, 320)

# The reference and input builders:
EP = 8

FP8 = jnp.float8_e4m3fn

# The small structurally-complete shape: expert count divisible by the shard
# count, hidden a whole number of 128-lane blocks, intermediate over one.
SMALL = dict(hidden=512, inter=256, e_total=32, topk=4, tokens=256)

# Qwen3.5-397B's MoE block, which is also the served geometry the kernel was
# written for: 512 experts over 8 shards is 64 local experts a shard.
PROD = dict(hidden=4096, inter=1024, e_total=512, topk=10)

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


def _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk):
    """The plain MoE: softmax, top-k, dense per-expert FFN, weighted sum.

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

    def local(x_g, gating_g, w1_l, w2_l, *scales):
        s1, s2 = scales if scaled else (None, None)
        scores = jax.nn.softmax(gating_g.astype(jnp.float32), axis=-1)
        topk_weights, topk_idx = lax.top_k(scores, topk)
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

    in_specs = (P(), P(), P(axis), P(axis)) + ((P(axis),) * 2 if scaled else ())
    args = (x, gating, w1, w2) + ((w1_scale, w2_scale) if scaled else ())
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


def _mask_padded_rows(x, gating, padded):
    """Padded rows route nowhere and carry nothing: a row whose `padded`
    entry is nonzero gets NaN logits and a zero input row."""
    valid = (padded == 0)[:, None]
    return jnp.where(valid, x, jnp.zeros_like(x)), jnp.where(valid, gating, jnp.nan)


def _mesh():
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(MESH_AXIS,))


def _run(mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk, fmt, valid_tokens=None):
    """The layer under a jit: the kernel's memory-space constraints need an
    enclosing jit."""
    drop = valid_tokens is not None
    scales = () if w1_scale is None else (w1_scale, w2_scale)
    if drop:
        # The runner's padding mask: each rank's rows past its valid count.
        local = x.shape[0] // EP
        padded = jnp.asarray(
            np.concatenate([np.arange(local) >= n for n in np.asarray(valid_tokens)]),
            jnp.int32,
        )

    @jax.jit
    def run(x, gating, w1, w2, *scales):
        s1, s2 = scales if scales else (None, None)
        gating = gating.astype(jnp.float32)
        if drop:
            x, gating = _mask_padded_rows(x, gating, padded)
        return fused_ep_moe(
            x,
            w1,
            w2,
            s1,
            s2,
            gating,
            top_k=topk,
            renormalize=True,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format=fmt,
            activation="silu",
            drop_nan_rows=drop,
        )

    return run(x, gating, w1, w2, *scales)


def _check(
    mesh, x, w1, w2, w1_scale, w2_scale, gating, *, topk, fmt, batch_bound, token_bound
):
    out = _run(mesh, x, w1, w2, w1_scale, w2_scale, gating, topk=topk, fmt=fmt)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert bool(jnp.isfinite(out.astype(jnp.float32)).all())
    want = _dense_reference(mesh, x, w1, w2, w1_scale, w2_scale, gating, topk=topk)
    error = _relative_l2(out, want)
    worst = _worst_token_relative_l2(out, want)
    assert error < batch_bound, f"relative L2 {error:.4f} past {batch_bound}"
    assert worst < token_bound, f"worst token {worst:.4f} past {token_bound}"

    def roll(a):
        return None if a is None else jnp.roll(a, 1, axis=0)

    rotated = _dense_reference(
        mesh, x, roll(w1), roll(w2), roll(w1_scale), roll(w2_scale), gating, topk=topk
    )
    assert _relative_l2(out, rotated) > ROTATED_CONTROL_FACTOR * error
    return error, worst


# The latest kernel's bf16 weight form still carries the token rows and the
# result rows over the wire as fp8 with one f32 scale per row (Config defaults
# token_rows="fp8", result_rows="fp8"), unlike the v2 port's bf16 form, which
# quantized nothing. So the fp8 bands are the ones that apply to it in every
# weight format; measured 0.0526 on the small bf16 shape, inside the 0.06 band.
@pytest.mark.parametrize("fmt", ["fp8", "bf16"])
def test_small_shape_tracks_a_dense_reference(fmt):
    mesh = _mesh()
    wf = WeightFormat.FP8 if fmt == "fp8" else WeightFormat.BF16
    w1, w2, s1, s2 = _make_weights(
        mesh,
        0,
        hidden=1024,
        inter=SMALL["inter"],
        e_total=SMALL["e_total"],
        weight_format=wf,
    )
    x, gating = _make_inputs(
        mesh, 1, tokens=SMALL["tokens"], hidden=1024, e_total=SMALL["e_total"]
    )
    error, worst = _check(
        mesh,
        x,
        w1,
        w2,
        s1,
        s2,
        gating,
        topk=SMALL["topk"],
        fmt=fmt,
        batch_bound=FP8_RELATIVE_BOUND,
        token_bound=FP8_TOKEN_BOUND,
    )
    print(f"LATEST {fmt} small: rel_l2 {error:.4f} worst token {worst:.4f}")


@pytest.fixture(scope="module")
def production_weights():
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


@pytest.mark.parametrize("local_tokens", SERVED_LOCAL_BUCKETS)
def test_served_buckets_track_a_dense_reference(production_weights, local_tokens):
    """Qwen3.5-397B's MoE block at every bucket the fused path serves."""
    mesh = _mesh()
    w1, w2, s1, s2 = production_weights
    x, gating = _make_inputs(
        mesh,
        2,
        tokens=local_tokens * EP,
        hidden=PROD["hidden"],
        e_total=PROD["e_total"],
    )
    error, worst = _check(
        mesh,
        x,
        w1,
        w2,
        s1,
        s2,
        gating,
        topk=PROD["topk"],
        fmt="fp8",
        batch_bound=FP8_RELATIVE_BOUND,
        token_bound=FP8_TOKEN_BOUND,
    )
    print(f"LATEST fp8 {local_tokens}x{EP}: rel_l2 {error:.4f} worst token {worst:.4f}")


def test_served_bucket_with_a_partially_padded_rank(production_weights):
    """The 320 bucket with one rank half padded and one empty: the valid rows
    match the reference and the padded rows are exactly zero."""
    mesh = _mesh()
    w1, w2, s1, s2 = production_weights
    local = 320
    x, gating = _make_inputs(
        mesh, 3, tokens=local * EP, hidden=PROD["hidden"], e_total=PROD["e_total"]
    )
    valid_tokens = jnp.asarray([320, 320, 160, 320, 33, 320, 320, 0], dtype=jnp.int32)
    out = _run(
        mesh,
        x,
        w1,
        w2,
        s1,
        s2,
        gating,
        topk=PROD["topk"],
        fmt="fp8",
        valid_tokens=valid_tokens,
    )
    want = _dense_reference(mesh, x, w1, w2, s1, s2, gating, topk=PROD["topk"])
    valid = np.concatenate([np.arange(local) < n for n in np.asarray(valid_tokens)])
    actual, reference = np.asarray(out), np.asarray(want)
    assert bool(np.isfinite(actual).all())
    assert not np.count_nonzero(actual[~valid])
    error = _relative_l2(actual[valid], reference[valid])
    assert error < FP8_RELATIVE_BOUND, error
