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
"""Device tests for the push-dispatch fused EP MoE.

`fused_ep_moe_push` is v2's fused stage with only the data movement changed:
the activation all-gather becomes a SparseCore pre-push of the first
`sc_prepush` visit positions plus in-kernel pushes of the rest, the routing
exchange may run as a TensorCore remote-DMA kernel, and the combine folds
the arrival-scale lookup in. So the gate is the layer's output against
`fused_ep_moe_v2` on the same inputs (which tests/kernels/fused_moe holds to
a dense reference), bit for bit, over every dispatch split, both routing
exchanges, both slab builds (push from VMEM / through the contribution slab)
and both served weight formats, with a rotated-expert control.

Every test needs the eight-way expert-parallel slice (8 JAX devices, e.g. a
tpu7x 2x2x1 host); on anything else the module skips at setup.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.push_fused_moe import fused_ep_moe_push
from vllm_torchtpu.kernels.experimental.push_fused_moe.tc_exchange import (
    build_tc_exchange_all_gather,
)
from vllm_torchtpu.kernels.fused_moe.v2 import AXIS, WeightFormat, fused_ep_moe_v2

pytestmark = pytest.mark.multichip


@pytest.fixture(scope="module", autouse=True)
def _require_eight_devices():
    """Skip the module unless this host has the eight-way slice. A fixture,
    not a `skipif`, so collection never builds the PJRT client (see the
    same fixture in tests/kernels/fused_moe/test_fused_ep_moe_v2.py)."""
    if jax.device_count() < 8:
        pytest.skip("the push fused EP MoE is an eight-way collective here")


EP = 8
CAPACITY = 128
FP8 = jnp.float8_e4m3fn
TOPK = 4
TOKENS = 256

# Push and v2 run the same plan, FFN and combine arithmetic on the same
# quantized rows, so the outputs must be bit-identical (measured on tpu7x
# for every case here); the rotated control must sit far from them.
CONTROL_BOUND = 0.1


def _mesh():
    return Mesh(np.asarray(jax.devices()[:EP]), axis_names=(AXIS,))


def _fp8_weights(mesh, seed, *, hidden, inter, e_total):
    """(w1, w2, w1_scale, w2_scale), fp8 with per-output-channel scales."""
    rng = np.random.default_rng(seed)
    shard = NamedSharding(mesh, P(AXIS))
    out = []
    scales = []
    for shape in ((e_total, hidden, 2 * inter), (e_total, inter, hidden)):
        w = rng.normal(0, 0.1, shape).astype(np.float32)
        amax = np.max(np.abs(w), axis=1, keepdims=True)
        scale = np.where(amax == 0, 1.0, amax / float(jnp.finfo(FP8).max))
        out.append(jax.device_put(w / scale, shard).astype(FP8))
        scales.append(jax.device_put(scale.reshape(shape[0], shape[2]), shard))
    return out[0], out[1], scales[0], scales[1]


def _fp4_weights(mesh, seed, *, hidden, inter, e_total, rhs_qb):
    """(w1, w2, w1_scale, w2_scale), fp4 e2m1 with per-`rhs_qb` block scales."""
    rng = np.random.default_rng(seed)
    shard = NamedSharding(mesh, P(AXIS))
    weights, scales = [], []
    for shape in ((e_total, hidden, 2 * inter), (e_total, inter, hidden)):
        raw = rng.normal(0, 0.1, shape).astype(np.float32)
        grouped = raw.reshape(shape[0], shape[1] // rhs_qb, rhs_qb, shape[2])
        scale = np.maximum(np.max(np.abs(grouped), axis=2) / 6, 1e-12)
        normalized = (grouped / scale[:, :, None, :]).reshape(shape)
        weights.append(jax.device_put(normalized, shard).astype(jnp.float4_e2m1fn))
        scales.append(jax.device_put(scale, shard))
    return weights[0], weights[1], scales[0], scales[1]


def _inputs(mesh, seed, *, hidden, e_logical, tokens=TOKENS):
    rng = np.random.default_rng(seed)
    shard = NamedSharding(mesh, P(AXIS))
    x = jax.device_put(rng.normal(0, 0.1, (tokens, hidden)).astype(np.float32), shard)
    gating = rng.normal(0, 1, (tokens, e_logical)).astype(np.float32)
    return x.astype(jnp.bfloat16), jax.device_put(gating, shard)


def _rel(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _roll(weights):
    w1, w2, s1, s2 = weights
    return (
        jnp.roll(w1, 1, axis=0),
        jnp.roll(w2, 1, axis=0),
        jnp.roll(s1, 1, axis=0),
        jnp.roll(s2, 1, axis=0),
    )


SMALL = dict(hidden=512, inter=256, e_total=32)
FP4_SHAPE = dict(hidden=1024, inter=512, e_total=32)
G_LOCAL = SMALL["e_total"] // EP  # 4, the same for FP4_SHAPE
# A ragged stride past ep * g_local * capacity makes the build ship tiles
# through the contribution slab instead of straight from VMEM.
CONTRIB_SLAB_STRIDE = EP * G_LOCAL * CAPACITY + CAPACITY


def _check_parity(weights, common, *, sc_prepush, tc_exchange, ragged_stride):
    mesh = common["mesh"]
    hidden = weights[0].shape[1]
    x, gating = _inputs(mesh, 7, hidden=hidden, e_logical=weights[0].shape[0])
    kw = dict(common, ragged_stride=ragged_stride)
    want = fused_ep_moe_v2(x, *weights, gating, **kw)
    got = fused_ep_moe_push(
        x, *weights, gating, sc_prepush=sc_prepush, tc_exchange=tc_exchange, **kw
    )
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    wrong = fused_ep_moe_v2(x, *_roll(weights), gating, **kw)
    assert _rel(got, wrong) > CONTROL_BOUND


@pytest.mark.parametrize(
    "ragged_stride", [None, CONTRIB_SLAB_STRIDE], ids=["vmem", "slab"]
)
@pytest.mark.parametrize("tc_exchange", [False, True], ids=["xla_ag", "tc_ag"])
@pytest.mark.parametrize("sc_prepush", [1, G_LOCAL], ids=["pre1", "pre_all"])
def test_fp8_push_matches_v2(sc_prepush, tc_exchange, ragged_stride):
    mesh = _mesh()
    weights = _fp8_weights(mesh, 11, **SMALL)
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=True,
    )
    _check_parity(
        weights,
        common,
        sc_prepush=sc_prepush,
        tc_exchange=tc_exchange,
        ragged_stride=ragged_stride,
    )


@pytest.mark.parametrize(
    "ragged_stride", [None, CONTRIB_SLAB_STRIDE], ids=["vmem", "slab"]
)
@pytest.mark.parametrize(
    "sc_prepush,tc_exchange", [(1, False), (G_LOCAL, True)], ids=["pre1_xla", "all_tc"]
)
def test_fp4_push_matches_v2(sc_prepush, tc_exchange, ragged_stride):
    """The served form: FP4 weights with block-512 scales, sqrtsoftplus with
    a score bias, the routed scale and the clamped activation."""
    mesh = _mesh()
    weights = _fp4_weights(mesh, 21, rhs_qb=512, **FP4_SHAPE)
    bias = jax.device_put(
        np.random.default_rng(5)
        .normal(0, 0.1, FP4_SHAPE["e_total"])
        .astype(np.float32),
        NamedSharding(mesh, P()),
    )
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP4,
        rhs_qb=512,
        sharded_plan=True,
        scoring_fn="sqrtsoftplus",
        score_bias=bias,
        routed_scaling_factor=2.5,
        act_fn="silu_and_mul_with_clamp",
    )
    _check_parity(
        weights,
        common,
        sc_prepush=sc_prepush,
        tc_exchange=tc_exchange,
        ragged_stride=ragged_stride,
    )


def test_sc_prepush_below_one_raises_and_above_g_local_clamps():
    mesh = _mesh()
    weights = _fp8_weights(mesh, 31, **SMALL)
    x, gating = _inputs(mesh, 32, hidden=SMALL["hidden"], e_logical=SMALL["e_total"])
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
    )
    with pytest.raises(ValueError, match="sc_prepush must be >= 1; got 0"):
        fused_ep_moe_push(x, *weights, gating, sc_prepush=0, **common)
    clamped = fused_ep_moe_push(x, *weights, gating, sc_prepush=G_LOCAL + 3, **common)
    at_g = fused_ep_moe_push(x, *weights, gating, sc_prepush=G_LOCAL, **common)
    np.testing.assert_array_equal(np.asarray(clamped), np.asarray(at_g))


def _contiguous_plus_one(experts, ep):
    """Rank r holds its contiguous block plus the next rank's first expert:
    per-slot expert ids [ep * (experts / ep + 1)] and the [experts, 2] slot
    table (slots holding each expert in slot order, -1 padded)."""
    per = experts // ep
    held = [
        [r * per + j for j in range(per)] + [((r + 1) * per) % experts]
        for r in range(ep)
    ]
    slot_expert = [e for row in held for e in row]
    slots = [[] for _ in range(experts)]
    for s, e in enumerate(slot_expert):
        slots[e].append(s)
    table = [h + [-1] * (2 - len(h)) for h in slots]
    return np.asarray(slot_expert, np.int32), np.asarray(table, np.int32)


def test_expert_layout_serves_the_same_tokens_and_counts_its_rows():
    """A replicated layout (R = 2): the weights in slot order, gating per
    logical expert. The output must be the unplaced layer's (every copy holds
    the same weights, so only where the rows run changes), and the load
    accumulator must count exactly the rows `_to_slots` sends to each slot:
    expert e of global row t goes to slot table[e, t mod copies(e)]."""
    mesh = _mesh()
    experts = SMALL["e_total"]
    logical = _fp8_weights(mesh, 41, **SMALL)
    slot_expert, table = _contiguous_plus_one(experts, EP)
    shard = NamedSharding(mesh, P(AXIS))
    replicated = NamedSharding(mesh, P())
    physical = tuple(
        jax.device_put(jnp.take(w, jnp.asarray(slot_expert), axis=0), shard)
        for w in logical
    )
    e_total = len(slot_expert)
    x, gating = _inputs(mesh, 42, hidden=SMALL["hidden"], e_logical=experts)
    common = dict(
        topk=TOPK,
        renormalize=True,
        mesh=mesh,
        capacity=CAPACITY,
        weight_format=WeightFormat.FP8,
        sharded_plan=True,
        sc_prepush=2,
    )
    load0 = jnp.full((e_total,), 3, jnp.int32)
    out, load = fused_ep_moe_push(
        x,
        *physical,
        gating,
        expert_slot=jax.device_put(jnp.asarray(table), replicated),
        expert_load=jax.device_put(load0, replicated),
        **common,
    )
    plain = fused_ep_moe_push(x, *logical, gating, **common)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(plain))
    wrong = fused_ep_moe_push(x, *_roll(logical), gating, **common)
    assert _rel(out, wrong) > CONTROL_BOUND
    g = np.asarray(gating)
    selected = np.argsort(-g, axis=1, kind="stable")[:, :TOPK]
    copies = (table >= 0).sum(axis=1)
    rows = np.arange(g.shape[0])[:, None]
    slots = table[selected, rows % copies[selected]]
    want = 3 + np.bincount(slots.ravel(), minlength=e_total)
    np.testing.assert_array_equal(np.asarray(load), want)


@pytest.mark.parametrize("width", [384, 1029])
def test_tc_exchange_all_gather_equals_lax_all_gather(width):
    mesh = _mesh()
    fn = build_tc_exchange_all_gather(ep=EP, width=width)

    def local(p):
        flat = p.reshape(-1)
        tc = fn(flat, lax.axis_index(AXIS))
        xla = lax.all_gather(flat, AXIS, axis=0, tiled=True)
        return tc[None], xla[None]

    data = (jnp.arange(EP * width, dtype=jnp.int32) * 7 - 3).reshape(EP, width)
    tc, xla = jax.jit(
        jax.shard_map(
            local,
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=(P(AXIS), P(AXIS)),
            check_vma=False,
        )
    )(jax.device_put(data, NamedSharding(mesh, P(AXIS))))
    np.testing.assert_array_equal(np.asarray(tc), np.asarray(xla))
    np.testing.assert_array_equal(np.asarray(tc)[0], np.asarray(data).reshape(-1))
