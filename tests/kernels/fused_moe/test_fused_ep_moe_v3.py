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
"""Fused EP MoE v3 (``kernels/fused_moe/v3``) end to end on an eight-core
TPU7x host, against a reference that runs the layer's own arithmetic.

``fused_ep_moe`` runs over an 8-shard expert-parallel mesh. The reference
selects each shard's experts with the layer's own top-k selection, runs every
selected (token, expert) pair through a Pallas program made of the layer's
own per-expert steps (``rowquant.quantize_rows`` for the FP8 token row,
``ffn.expert_ffn_fp8``, and ``ffn.result_rows`` with bf16 result rows) on
tiles of the layer's height, and sums each token's pairs in k order in
float32, as the combine does.

The reference runs those steps as a Pallas program on the same chip rather
than as plain XLA, because the two compilers do not round alike. The
bfloat16 scaling product in front of each FP8 cast
(``Config.elementwise_dtype``, the serving default) leaves its second
rounding to the compiler, and one FP8 step in a token row or in the
intermediate moves every value of its output row by about a percent of the
row's size. A per-token XLA reference put 73-77% of the output values more
than 2^-7 off on TPU, with the same numbers for both row paths and for the
source branch's kernel. Compiled by the same compiler, the reference takes
the same roundings, so the bound below is one bf16 step.

The routing tables, the token exchange, the row gather, the FFN kernel and
the combine are the layer's; only the per-expert arithmetic is shared with
the reference. A row routed to the wrong expert, dropped, swapped with
another token's, or combined with the wrong weight moves most of its 1024
values, and one such row is more than 1 in 1000 of the output. Both row
paths run: the SparseCore row gather (``Config.sorted_rows``, the serving
default) and the Pallas transport, in the serving bridge's shard-index mode,
and the kernel's own mesh-axis mode once.

The layer does not run in the TPU interpreter on CPU: its token exchange
returns DMA semaphores from one Pallas program to the next, which the
interpreter cannot do. The CPU tests here check the reference itself, in
interpret mode, against a by-hand computation and against routing errors.
Small shapes: hidden 1024, intermediate 128, 16 or 32 experts, top-k 4.
"""

import importlib

import numpy as np
import pytest

D, H, INTER, K = 8, 1024, 128, 4
TILE_ROWS = 32
# One bf16 step: a value is off when it is more than this far from the
# reference, relative, with an absolute floor for values near zero.
STEP_REL, STEP_ABS = 2.0**-7, 1e-5
MAX_OFF_FRACTION = 1e-3

# name: (experts, rows per shard, routing, rows dropped on odd shards)
CASES = {
    "spread": (16, 24, "spread", 0),
    "hot": (16, 24, "hot", 0),
    "short_shards": (32, 32, "spread", 5),
}
# The serving bridge passes each shard's index as an operand
# (layout.SHARD_INDEX_OPERAND, no side effects); the kernel's own mode reads
# the mesh axis. mode: (shard index operand, sorted rows, cases)
MODES = {
    "bridge_sorted": (True, True, list(CASES)),
    "bridge_transport": (True, False, list(CASES)),
    "kernel_sorted": (False, True, ["spread"]),
}


def _world(experts, real, routing, seed=0, shards=D, hidden=H, inter=INTER):
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v3.rowquant import FP8

    rng = np.random.default_rng(seed)
    tokens = real * shards
    x = (rng.standard_normal((tokens, hidden)) / 10).astype(np.float32)
    logits = rng.standard_normal((tokens, experts)).astype(np.float32)
    if routing == "hot":
        # Every token picks experts 0..K-1, all on shard 0: one shard receives
        # every row, the others none.
        logits[:, :K] += 20.0
    w1f = rng.standard_normal((experts, hidden, 2 * inter)).astype(np.float32) / 10
    w2f = rng.standard_normal((experts, inter, hidden)).astype(np.float32) / 10
    s1 = np.abs(w1f).max(1) / 448.0
    s2 = np.abs(w2f).max(1) / 448.0
    w1 = jnp.asarray(w1f / s1[:, None, :]).astype(FP8)
    w2 = jnp.asarray(w2f / s2[:, None, :]).astype(FP8)
    return (
        jnp.asarray(x).astype(jnp.bfloat16),
        jnp.asarray(logits),
        w1,
        w2,
        jnp.asarray(s1),
        jnp.asarray(s2),
    )


def _valid_rows(real, valid):
    """The global indices of the rows below each shard's valid count."""
    return np.concatenate(
        [np.arange(d * real, d * real + v) for d, v in enumerate(valid)]
    )


def _routes(logits, real, valid):
    """Each token's selected experts and weights, [tokens, K] each, from the
    layer's top-k selection on each shard's rows."""
    import jax.numpy as jnp

    from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer

    weights = np.zeros((logits.shape[0], K), np.float32)
    experts = np.zeros((logits.shape[0], K), np.int32)
    for d, v in enumerate(valid):
        rows = slice(d * real, (d + 1) * real)
        w, idx = v3_layer._select(
            logits[rows],
            top_k=K,
            tokens_per_shard=real,
            renormalize=True,
            drop_nan_rows=True,
            message_rows=None,
            valid_tokens=jnp.asarray([v], jnp.int32),
        )
        weights[rows], experts[rows] = np.asarray(w), np.asarray(idx)
    return weights, experts


def _expert_rows(rows, tile_expert, w1, w2, s1, s2, config):
    """The result row of every input row: one Pallas program whose grid step
    i takes TILE_ROWS bf16 token rows and runs them through the layer's own
    steps with expert tile_expert[i]'s weights: the FP8 row quantization,
    the expert FFN, and the bf16 result rows. Config.elementwise is the
    scaling products' element type and Config.row_scale_dtype the row
    scales' rounding, as in the layer."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.fused_moe.v3 import ffn, rowquant

    # The forms this reference writes out: one scale per row for the token
    # rows and the intermediate, both FP8, and bf16 result rows.
    assert config.activation_block == 0 and config.token_rows_block == 0, config
    assert config.token_rows == config.intermediate == "fp8", config
    assert config.result_rows == "bf16", config
    hidden, columns = w1.shape[1], w1.shape[2]
    inter = w2.shape[1]

    def body(tile_expert_ref, rows_ref, w1_ref, w2_ref, s1_ref, s2_ref, out_ref):
        del tile_expert_ref  # read by the index maps
        q, q_scales = rowquant.quantize_rows(
            rows_ref[...], config.row_scale_dtype, block=0, dtype=config.elementwise
        )
        acc2, mid_scales = ffn.expert_ffn_fp8(
            q,
            q_scales,
            w1_ref[...],
            w2_ref[...],
            s1_ref[...],
            activation="silu",
            w1_bias=None,
            no_gate=False,
            block=0,
            elementwise=config.elementwise,
            columns_per_pass=config.activation_columns_per_pass,
            token_block=0,
        )
        result, _ = ffn.result_rows(
            acc2,
            mid_scales,
            w2_scales=s2_ref[...],
            w2_bias=None,
            dtype=jnp.bfloat16,
            elementwise=config.elementwise,
        )
        out_ref[...] = result

    def per_expert(shape):
        return pl.BlockSpec((None, *shape), lambda i, experts: (experts[i], 0, 0))

    tile = pl.BlockSpec((TILE_ROWS, hidden), lambda i, experts: (i, 0))
    return pl.pallas_call(
        body,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(rows.shape[0] // TILE_ROWS,),
            in_specs=[
                tile,
                per_expert((hidden, columns)),
                per_expert((inter, hidden)),
                per_expert((1, columns)),
                per_expert((1, hidden)),
            ],
            out_specs=tile,
        ),
        out_shape=jax.ShapeDtypeStruct(rows.shape, jnp.bfloat16),
    )(tile_expert, rows, w1, w2, s1[:, None, :], s2[:, None, :])


def _reference(x, w1, w2, s1, s2, weights, experts, rows, config):
    """The layer's output on `rows` (global token indices), float32 [tokens,
    hidden], zero elsewhere. Each expert's (token, slot) pairs go through
    _expert_rows in whole tiles; then each token sums its K result rows in
    k order in float32, a zero weight contributing nothing, and rounds the
    sum to bf16, as the combine does."""
    import jax.numpy as jnp

    by_expert = [[] for _ in range(w1.shape[0])]
    for t in rows:
        for k in range(K):
            by_expert[experts[t, k]].append((int(t), k))
    order, tile_expert, position = [], [], {}
    for e, pairs in enumerate(by_expert):
        if not pairs:
            continue
        for pair in pairs:
            position[pair] = len(order)
            order.append(pair[0])
        pad = -len(pairs) % TILE_ROWS
        order += [None] * pad
        tile_expert += [e] * ((len(pairs) + pad) // TILE_ROWS)
    xs = np.asarray(x.astype(jnp.float32))
    zero = np.zeros(xs.shape[1], np.float32)
    batch = np.stack([zero if t is None else xs[t] for t in order])
    result = _expert_rows(
        jnp.asarray(batch).astype(jnp.bfloat16),
        jnp.asarray(tile_expert, jnp.int32),
        w1,
        w2,
        s1,
        s2,
        config,
    )
    result = np.asarray(result.astype(jnp.float32))
    out = np.zeros(xs.shape, np.float32)
    for t in rows:
        acc = zero
        for k in range(K):
            c = weights[t, k]
            term = result[position[(int(t), k)]] * c if c != 0 else zero
            acc = term if k == 0 else acc + term
        out[t] = acc.astype(jnp.bfloat16).astype(np.float32)
    return out


def _compare(got, want, rows):
    """How far `got` is from `want` on `rows`. It passes when every value is
    finite and fewer than MAX_OFF_FRACTION of the values are more than one
    bf16 step off. rows_off and worst_row_error (a row's error norm over its
    reference norm) tell a failure on a few rows from one on all of them."""
    g, w = got[rows], want[rows]
    diff = np.abs(g - w)
    off = diff > STEP_REL * np.abs(w) + STEP_ABS
    norms = np.maximum(np.linalg.norm(w, axis=1), np.finfo(np.float32).tiny)
    report = {
        "finite": bool(np.isfinite(g).all()),
        "off_fraction": float(off.mean()),
        "rows_off": int(off.any(axis=1).sum()),
        "rows": len(rows),
        "worst_row_error": float((np.linalg.norm(g - w, axis=1) / norms).max()),
        "max_abs_diff": float(diff.max()),
        "want_rms": float(np.sqrt((w**2).mean())),
    }
    report["pass"] = (
        report["finite"]
        and report["off_fraction"] < MAX_OFF_FRACTION
        and report["want_rms"] > 0
    )
    return report


# ---- the reference on CPU, in TPU interpret mode ----


@pytest.fixture
def interpret(monkeypatch):
    """TPU interpret mode with a TPU7x chip record, on the CPU even on a TPU
    host: on a chip the interpreter's steps would be XLA's TPU arithmetic,
    whose rounding of the bf16 products is the very difference these tests
    are not about."""
    import jax
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    def tpu7x():
        return pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)

    monkeypatch.setattr(pltpu, "get_tpu_info", tpu7x)
    monkeypatch.setattr(tpu_info, "get_tpu_info", tpu7x)
    with (
        jax.default_device(jax.devices("cpu")[0]),
        pltpu.force_tpu_interpret_mode(pltpu.InterpretParams()),
    ):
        yield


# A small world for the CPU tests: two shards of 16 rows, the second shard
# 3 rows short, 8 experts, hidden and intermediate one lane tile each.
SMALL = {"shards": 2, "real": 16, "short": 3, "experts": 8, "hidden": 128}


def _small_world():
    from vllm_torchtpu.kernels.fused_moe.v3 import Config

    s = SMALL
    world = _world(
        s["experts"],
        s["real"],
        "spread",
        seed=3,
        shards=s["shards"],
        hidden=s["hidden"],
        inter=s["hidden"],
    )
    valid = [s["real"] - (s["short"] if d % 2 else 0) for d in range(s["shards"])]
    return world, valid, Config(result_rows="bf16")


def _by_hand(x, w1, w2, s1, s2, weights, experts, rows):
    """The layer's arithmetic written out in numpy, one routed pair at a
    time, with Config's default bfloat16 scaling products: each product of
    a row and its bf16 inverse scale is rounded to bf16, then cast to FP8."""
    import jax.numpy as jnp

    bf16, fp8, f32 = jnp.bfloat16, jnp.float8_e4m3fn, np.float32

    def fp8_row(v):
        """(FP8 values as f32, f32 scale) of one row: scale = amax / 448."""
        scale = f32(np.abs(v).max()) / f32(448.0)
        inverse = (f32(1.0) / scale).astype(bf16).astype(f32)
        product = (v.astype(bf16).astype(f32) * inverse).astype(bf16)
        return product.astype(fp8).astype(f32), scale

    xs = np.asarray(x.astype(jnp.float32))
    w1s = np.asarray(w1.astype(jnp.float32))
    w2s = np.asarray(w2.astype(jnp.float32))
    s1s, s2s = np.asarray(s1), np.asarray(s2)
    inter = w2s.shape[1]
    out = np.zeros(xs.shape, f32)
    for t in rows:
        q, q_scale = fp8_row(xs[t])
        acc = None
        for k in range(K):
            e, c = experts[t, k], weights[t, k]
            h = (q @ w1s[e]).astype(f32) * q_scale * s1s[e]
            gate, up = h[:inter], h[inter:]
            silu = (gate / (f32(1.0) + np.exp(-gate))).astype(f32)
            mid, mid_scale = fp8_row((silu * up).astype(bf16).astype(f32))
            down = (mid @ w2s[e]).astype(f32) * mid_scale * s2s[e]
            result = down.astype(bf16).astype(f32)
            term = result * c if c != 0 else 0 * result
            acc = term if acc is None else acc + term
        out[t] = acc.astype(bf16).astype(f32)
    return out


@pytest.mark.cpu_test
def test_reference_matches_the_layer_arithmetic_by_hand(interpret):
    """The reference's tiles, expert lookup and k-ordered sum, against the
    arithmetic written out in numpy for every routed pair, to the TPU
    test's bound."""
    (x, logits, w1, w2, s1, s2), valid, config = _small_world()
    rows = _valid_rows(SMALL["real"], valid)
    weights, experts = _routes(logits, SMALL["real"], valid)
    assert len(set(experts[rows].ravel())) == SMALL["experts"]  # every expert
    want = _by_hand(x, w1, w2, s1, s2, weights, experts, rows)
    got = _reference(x, w1, w2, s1, s2, weights, experts, rows, config)
    report = _compare(got, want, rows)
    assert report["pass"], report


def _routing_errors(weights, experts, rows):
    """Single-pair routing errors the layer could make, as (name, weights,
    experts): one token's pair sent to an expert it did not select, its
    smallest pair dropped, and its first and last slot weights exchanged."""
    t = int(rows[5])
    sent_elsewhere = experts.copy()
    sent_elsewhere[t, 1] = next(
        e for e in range(SMALL["experts"]) if e not in experts[t]
    )
    dropped = weights.copy()
    dropped[t, int(np.argmin(weights[t]))] = 0.0
    exchanged = weights.copy()
    exchanged[t, [0, K - 1]] = weights[t, [K - 1, 0]]
    return [
        ("pair sent to an unselected expert", weights, sent_elsewhere),
        ("smallest pair dropped", dropped, experts),
        ("first and last slot weights exchanged", exchanged, experts),
    ]


@pytest.mark.cpu_test
def test_one_bf16_step_flags_every_single_row_error(interpret):
    """The bound sees one wrong row: two tokens' rows swapped, a token's row
    zeroed, and each single-pair routing error in _routing_errors, while
    the reference passes against itself."""
    (x, logits, w1, w2, s1, s2), valid, config = _small_world()
    rows = _valid_rows(SMALL["real"], valid)
    weights, experts = _routes(logits, SMALL["real"], valid)
    want = _reference(x, w1, w2, s1, s2, weights, experts, rows, config)
    assert _compare(want, want, rows)["pass"]
    swapped = want.copy()
    swapped[[rows[0], rows[1]]] = swapped[[rows[1], rows[0]]]
    zeroed = want.copy()
    zeroed[rows[2]] = 0.0
    wrong = [("rows swapped", swapped), ("row zeroed", zeroed)]
    for name, bad_weights, bad_experts in _routing_errors(weights, experts, rows):
        bad = _reference(x, w1, w2, s1, s2, bad_weights, bad_experts, rows, config)
        wrong.append((name, bad))
    for name, bad in wrong:
        report = _compare(bad, want, rows)
        assert not report["pass"], (name, report)


# ---- the layer on eight TPU7x cores ----


@pytest.fixture
def eight_tpu7x_cores():
    """Skip unless this host has eight cores of a generation the kernel
    supports. A fixture rather than a `skipif`: a `skipif` builds the PJRT
    client while pytest collects, before markers deselect anything (see
    tests/kernels/fused_moe/test_fused_ep_moe_v2.py)."""
    import jax

    from vllm_torchtpu.kernels.fused_moe.v3 import supported_on

    if jax.device_count() < D or not supported_on(jax.devices()[0]):
        pytest.skip("fused EP MoE v3 needs eight TPU7x cores")


@pytest.fixture
def fresh_builds(monkeypatch):
    """Empty kernel and layer caches for one case. The FFN kernel's build
    reads layout.SHARD_INDEX_OPERAND (the bridge's mode takes one more
    scalar operand) but is cached on its arguments alone, so a build made
    in one mode would be handed to the other. Serving sets the mode once
    per process; this test switches it between cases."""
    ffn_kernel = importlib.import_module(
        "vllm_torchtpu.kernels.fused_moe.v3.ffn_kernel"
    )
    from vllm_torchtpu.kernels.fused_moe.v3 import layer as v3_layer

    monkeypatch.setattr(ffn_kernel, "_BUILD_CACHE", {})
    monkeypatch.setattr(v3_layer, "_LAYER_CACHE", {})


def _run_case(name, mode):
    """(layer output, reference, valid rows) for one case, both float32."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.kernels.fused_moe.v3 import MESH_AXIS, Config, fused_ep_moe
    from vllm_torchtpu.kernels.fused_moe.v3 import layout as v3_layout

    bridge, sorted_rows, _ = MODES[mode]
    experts, real, routing, short = CASES[name]
    x, logits, w1, w2, s1, s2 = _world(experts, real, routing)
    mesh = Mesh(np.asarray(jax.devices()[:D]), (MESH_AXIS,))
    sharded = NamedSharding(mesh, P(MESH_AXIS))
    args = [jax.device_put(a, sharded) for a in (x, w1, w2, s1, s2, logits)]
    valid = [real - (short if d % 2 else 0) for d in range(D)]
    config = Config(result_rows="bf16", sorted_rows=sorted_rows)
    shard_index = jnp.arange(D, dtype=jnp.int32).reshape(D, 1) if bridge else None

    def run(x, w1, w2, s1, s2, logits):
        return fused_ep_moe(
            x,
            w1,
            w2,
            s1,
            s2,
            logits,
            top_k=K,
            renormalize=True,
            mesh=mesh,
            tile_rows=TILE_ROWS,
            weight_format="fp8",
            activation="silu",
            config=config,
            experts_per_shard=experts // D,
            shard_index=shard_index,
            valid_tokens=jnp.asarray(valid, jnp.int32),
            drop_nan_rows=True,
        )

    saved = (v3_layout.SHARD_INDEX_OPERAND, v3_layout.SIDE_EFFECTS)
    v3_layout.SHARD_INDEX_OPERAND, v3_layout.SIDE_EFFECTS = bridge, not bridge
    try:
        got = np.asarray(jax.block_until_ready(jax.jit(run)(*args)))
    finally:
        v3_layout.SHARD_INDEX_OPERAND, v3_layout.SIDE_EFFECTS = saved
    rows = _valid_rows(real, valid)
    weights, chosen = _routes(logits, real, valid)
    want = _reference(x, w1, w2, s1, s2, weights, chosen, rows, config)
    return got.astype(np.float32), want, rows


RUNS = [(mode, name) for mode, (_, _, names) in MODES.items() for name in names]


@pytest.mark.multichip
@pytest.mark.parametrize("mode,name", RUNS)
def test_fused_ep_moe_matches_the_reference(
    eight_tpu7x_cores, fresh_builds, mode, name
):
    """Every value within one bf16 step of the reference, except fewer than
    1 in 1000. The same check must fail on the layer's own output with two
    tokens' rows swapped, and with one row zeroed."""
    got, want, rows = _run_case(name, mode)
    report = _compare(got, want, rows)
    assert report["pass"], report
    swapped = got.copy()
    swapped[[rows[0], rows[1]]] = swapped[[rows[1], rows[0]]]
    zeroed = got.copy()
    zeroed[rows[2]] = 0.0
    assert not _compare(swapped, want, rows)["pass"]
    assert not _compare(zeroed, want, rows)["pass"]
