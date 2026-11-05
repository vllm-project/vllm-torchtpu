# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_PREFILL_SEQ_PARALLEL, no TPU needed: the token split of a prefill step
across the cores of a 16-core mesh.

- The split collectives (fused_moe_prefill_kernel_v2.own_rows, scatter_rows,
  gather_rows) leave each core exactly the rows the fused MoE prefill kernel's
  reduce-scatter leaves it, chunk by combine chunk: reduce-scatter then
  all-gather equals the kernel's sum, and gather undoes own_rows.
- moe_tp_decode's split op body, fed each core's rows, returns bit for bit
  each core's rows of what today's op body returns on every token, with and
  without the norm's fp8 rows, over one and several combine chunks, through
  the slab + combine and the in-kernel sum builds (Pallas interpret mode).
- Positive controls: rows split in contiguous blocks instead of the MoE's
  chunk blocks, or one core's rows swapped, fail the comparison.
- prefill_token_split's three ops (token_fn) and both MoE op bodies, lowered
  for TPU inside an outer jit with SPMD shardings as the model's whole-graph
  compile sees them, hold no partition-id: XLA's SPMD partitioner refuses it
  (`lax.axis_index` lowers to one), so a core reads its mesh index as data.
  The ops fed that index give each core exactly the rows above.

Runs on 16 devices; CPU virtual devices are sufficient, e.g.
XLA_FLAGS=--xla_force_host_platform_device_count=16."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental import rotated_collectives
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel as v1,
)
from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as v2,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host
from vllm_torchtpu.kernels.fused_moe.v2 import kernel as v2_kernel
from vllm_torchtpu.layers.adapter import moe_tp_decode, prefill_token_split

pytestmark = pytest.mark.multichip

RANKS = 16


@pytest.fixture
def mesh():
    if jax.device_count() < RANKS:
        pytest.skip("requires 16 devices; CPU virtual devices are sufficient")
    return Mesh(np.asarray(jax.devices()[:RANKS]), ("d",))


def _bits(x):
    return np.asarray(x).view(np.uint16 if x.dtype == jnp.bfloat16 else np.uint32)


def _per_core(mesh, fn, *args, in_specs=None):
    specs = in_specs or (P("d"),) * len(args)
    return jax.jit(
        jax.shard_map(fn, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False)
    )(*args)


def _own_rows_np(x, index):
    return np.asarray(v2.own_rows(jnp.asarray(x), index, RANKS))


def _contiguous_rows_np(x, index):
    rows = x.shape[0] // RANKS
    return np.asarray(x[index * rows : (index + 1) * rows])


@pytest.mark.parametrize("tokens", [256, 2560, 4096])
def test_in_token_order_puts_core_gathered_rows_in_token_order(mesh, tokens):
    h = 128
    rng = np.random.default_rng(tokens)
    rows = jnp.asarray(rng.integers(-16, 17, (tokens, h)), jnp.bfloat16)
    gathered = np.asarray(
        _per_core(
            mesh,
            lambda y: lax.all_gather(
                v2.own_rows(y, lax.axis_index("d"), RANKS), "d", axis=0, tiled=True
            ),
            jnp.tile(rows, (RANKS, 1)),
        )
    )[:tokens]
    back = np.asarray(v2.in_token_order(jnp.asarray(gathered), RANKS))
    np.testing.assert_array_equal(_bits(back), _bits(rows))
    if v2.token_chunks(tokens) > 1:
        assert not np.array_equal(_bits(gathered), _bits(rows))


def test_gather_routes_returns_every_cores_bits(mesh):
    rng = np.random.default_rng(7)
    weights = jnp.asarray(rng.random((RANKS * 4, 8)), jnp.bfloat16)
    ids = jnp.asarray(rng.integers(0, 256, (RANKS * 4, 8)), jnp.int32)
    scale = jnp.asarray(rng.random((RANKS * 4,)), jnp.float32)

    fn = jax.jit(
        jax.shard_map(
            lambda w, i, s: v2.gather_routes(w, i, s, "d"),
            mesh=mesh,
            in_specs=(P("d"),) * 3,
            out_specs=P(),
            check_vma=False,
        )
    )
    w, i, s = (np.asarray(a) for a in fn(weights, ids, scale))
    np.testing.assert_array_equal(i, np.asarray(ids))
    np.testing.assert_array_equal(_bits(w), _bits(weights))
    np.testing.assert_array_equal(_bits(s), _bits(scale))


# 256-2048: one combine chunk; 2560 and 4096: two chunks (1280 and 2048 rows).
@pytest.mark.parametrize("tokens", [256, 1024, 2560, 4096])
def test_split_collectives_follow_the_moe_reduce_scatter(mesh, tokens):
    h = 128
    rng = np.random.default_rng(tokens)
    # Integers in [-16, 16]: every partial sum is exact in bf16, so any
    # difference is a row in the wrong place, not rounding.
    x = jnp.asarray(rng.integers(-16, 17, (RANKS * tokens, h)), jnp.bfloat16)

    summed = np.asarray(
        _per_core(mesh, lambda p: v1._chunked_psum(tokens, lambda r: p[r], "d"), x)
    )[:tokens]
    mine = np.asarray(_per_core(mesh, lambda p: v2.scatter_rows(p, "d"), x))
    back = np.asarray(
        _per_core(mesh, lambda p: v2.gather_rows(v2.scatter_rows(p, "d"), "d"), x)
    )
    rows = tokens // RANKS
    for core in range(RANKS):
        np.testing.assert_array_equal(
            _bits(mine[core * rows : (core + 1) * rows]),
            _bits(_own_rows_np(summed, core)),
        )
        np.testing.assert_array_equal(
            _bits(back[core * tokens : (core + 1) * tokens]), _bits(summed)
        )

    taken = np.asarray(
        _per_core(
            mesh,
            lambda y: v2.gather_rows(v2.own_rows(y, lax.axis_index("d"), RANKS), "d"),
            jnp.tile(jnp.asarray(summed), (RANKS, 1)),
        )
    )
    np.testing.assert_array_equal(_bits(taken[:tokens]), _bits(summed))
    if v2.token_chunks(tokens) > 1:
        assert not np.array_equal(
            _own_rows_np(summed, 1), _contiguous_rows_np(summed, 1)
        )


# --------------------------------------------------------------------------- #
# The split MoE op body against today's, in Pallas interpret mode
# --------------------------------------------------------------------------- #
H, INTER, EXPERTS, TOPK = 640, 128, 8, 4
F8 = jnp.float8_e4m3fn


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(v2_host, "vmem_limit", lambda info=None: 63 << 20)
    monkeypatch.setattr(
        pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    # No TPU to ask: whole experts (stage A), and no decode kernel.
    monkeypatch.setattr(moe_tp_decode, "plan_blocks", lambda *a, **k: None)
    monkeypatch.setattr(
        moe_tp_decode.kernel, "can_use_moe_tp_decode", lambda *a, **k: False
    )
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL_V2", "1")
    monkeypatch.setattr(
        moe_tp_decode,
        "router_select",
        functools.partial(moe_tp_decode.router_select, interpret=True),
    )
    v2.build_expert_kernel.cache_clear()
    v2.build_combine.cache_clear()
    yield
    v2.build_expert_kernel.cache_clear()
    v2.build_combine.cache_clear()


def _moe_inputs(tokens, seed=0):
    """Replicated x, scores, bias and padding mask; each core's own TP slice
    of the routed and shared experts, stacked core by core."""
    ks = jax.random.split(jax.random.key(seed), 10)
    x = jax.random.normal(ks[0], (tokens, H)).astype(jnp.bfloat16)
    rng = np.random.default_rng(seed)
    scores = rng.random((tokens, EXPERTS)).astype(np.float32)
    scores[:, 0] += 1.0  # one expert in every token: multi-tile visits
    bias = np.zeros((EXPERTS,), np.float32)
    bias[1] = 0.5
    pad = np.zeros((tokens + 64,), bool)
    pad[tokens - 3 : tokens] = True
    shape = {
        "w13": (EXPERTS, H, 2 * INTER),
        "w2": (EXPERTS, INTER, H),
        "s13": (EXPERTS, 2 * INTER),
        "s2": (EXPERTS, H),
        "sw13": (H, 2 * INTER),
        "ss13": (2 * INTER,),
        "sw2": (INTER, H),
        "ss2": (H,),
    }
    weights = []
    for k, (name, dims) in zip(ks[1:], shape.items()):
        if name in ("s13", "s2", "ss13", "ss2"):
            w = jax.random.uniform(k, (RANKS, *dims), minval=0.01, maxval=0.03)
        else:
            w = (jax.random.normal(k, (RANKS, *dims)) * 0.5).astype(F8)
        weights.append(w.reshape(RANKS * dims[0], *dims[1:]))
    return x, jnp.asarray(scores), jnp.asarray(bias), jnp.asarray(pad), weights


def _statics(prefill_above=64):
    return moe_tp_decode._OpStatics(
        topk=TOPK,
        routed_scale=2.5,
        activation="silu",
        use_sparse_core=False,
        onehot_threshold=0,
        prefill_above=prefill_above,
    )


def _run_body(mesh, body, x, scores, bias, pad, weights, prequant):
    """body under shard_map with moe_tp_decode._get_op's operand specs."""
    rank = jnp.arange(RANKS, dtype=jnp.int32).reshape(RANKS, 1)
    e0 = jnp.zeros((RANKS, 1), jnp.int32)
    args = (x, scores, bias, pad, *weights, rank, e0, *prequant)
    specs = (P("d"), P("d"), P(), P()) + (P("d"),) * (len(args) - 4)
    return np.asarray(_per_core(mesh, body, *args, in_specs=specs))


def _today_and_split(mesh, tokens, prequant, split_rows=_own_rows_np):
    """Today's op body on every token of every core, and the split body on
    each core's rows (`split_rows`), as [core, rows, H] arrays."""
    x, scores, bias, pad, weights = _moe_inputs(tokens)
    statics = _statics()
    x_q, x_scale = v2._quantize_rows(x, F8)
    quant = (x_q, x_scale) if prequant else ()

    def tile(a):
        return jnp.tile(a, (RANKS,) + (1,) * (a.ndim - 1))

    today = _run_body(
        mesh,
        moe_tp_decode._op_body(statics, RANKS, "d"),
        tile(x),
        tile(scores),
        bias,
        pad,
        weights,
        tuple(tile(a) for a in quant),
    ).reshape(RANKS, tokens, H)

    def split(a):
        a = np.asarray(a)
        return jnp.asarray(np.concatenate([split_rows(a, c) for c in range(RANKS)]))

    mine = _run_body(
        mesh,
        moe_tp_decode._split_body(statics, RANKS, "d"),
        split(x),
        split(scores),
        bias,
        pad,
        weights,
        tuple(split(a) for a in quant),
    ).reshape(RANKS, tokens // RANKS, H)
    return today, mine


def _assert_rows_match(today, mine):
    for core in range(RANKS):
        np.testing.assert_array_equal(_bits(today[core]), _bits(today[0]))
        np.testing.assert_array_equal(
            _bits(mine[core]), _bits(_own_rows_np(today[0], core))
        )


@pytest.mark.parametrize("gather_once", [False, True])
@pytest.mark.parametrize(
    "tokens,chunk_tokens,sum_in_kernel,prequant",
    [
        (256, 2048, 0, False),  # one chunk, slab + combine
        (384, 128, 0, True),  # three combine chunks, the norm's fp8 rows
        (256, 2048, 256, True),  # the in-kernel sum build
        (384, 128, 384, False),  # the in-kernel sum build over three chunks
    ],
)
def test_split_moe_matches_todays_rows_bit_for_bit(
    mesh,
    interpret,
    monkeypatch,
    tokens,
    chunk_tokens,
    sum_in_kernel,
    prequant,
    gather_once,
):
    monkeypatch.setattr(v2, "_COMBINE_CHUNK_TOKENS", chunk_tokens)
    monkeypatch.setenv("TPU_MOE_PREFILL_SUM_IN_KERNEL_MAX_TOKENS", str(sum_in_kernel))
    monkeypatch.setenv("TPU_MOE_PREFILL_GATHER_ONCE", "1" if gather_once else "0")
    today, mine = _today_and_split(mesh, tokens, prequant)
    _assert_rows_match(today, mine)
    assert np.isfinite(today.astype(np.float32)).all()
    assert np.abs(today[0, : tokens - 3].astype(np.float32)).max() > 0


def _swap_cores(x, index):
    swapped = {3: 4, 4: 3}.get(index, index)
    return _own_rows_np(x, swapped)


@pytest.mark.parametrize("gather_once", [False, True])
@pytest.mark.parametrize("split_rows", [_contiguous_rows_np, _swap_cores])
def test_a_broken_split_fails(mesh, interpret, monkeypatch, split_rows, gather_once):
    monkeypatch.setattr(v2, "_COMBINE_CHUNK_TOKENS", 128)
    monkeypatch.setenv("TPU_MOE_PREFILL_GATHER_ONCE", "1" if gather_once else "0")
    today, mine = _today_and_split(mesh, 384, False, split_rows)
    with pytest.raises(AssertionError):
        _assert_rows_match(today, mine)


# --------------------------------------------------------------------------- #
# No partition-id before SPMD partitioning
# --------------------------------------------------------------------------- #
S = jax.ShapeDtypeStruct


@pytest.fixture
def tpu7x_info(monkeypatch):
    """Lower Mosaic kernels for TPU7x on a CPU host (as in
    tests/kernels/test_idx_topk_sched_groups.py)."""
    from jax._src.pallas.mosaic import tpu_info

    kind = jax.devices()[0].device_kind
    if tpu_info.chip_version_from_device_kind(kind) is None:
        monkeypatch.setitem(
            tpu_info.registry,
            kind,
            lambda: tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_7X, 1),
        )
    tpu_info.get_tpu_info.cache_clear()
    yield
    tpu_info.get_tpu_info.cache_clear()


def _tpu_lowering(mesh, fn, specs, *args):
    """fn lowered for TPU inside an outer jit whose operands and output carry
    their NamedShardings, before XLA's SPMD partitioning: what the torch_tpu
    backend hands XLA for the model's whole graph."""
    jitted = jax.jit(
        fn,
        in_shardings=tuple(NamedSharding(mesh, s) for s in specs),
        out_shardings=NamedSharding(mesh, P("d")),
    )
    return jitted.trace(*args).lower(lowering_platforms=("tpu",)).as_text()


def _assert_no_partition_id(text):
    assert "partition_id" not in text and "partition-id" not in text


@pytest.mark.parametrize("kind", ["own", "gather", "scatter"])
def test_token_ops_lower_without_partition_id(mesh, kind):
    fn, specs, _ = prefill_token_split.token_fn(mesh, kind)
    args = [S((RANKS * 1024, 128), jnp.bfloat16)]
    if kind == "own":
        args.append(S((RANKS, 1), jnp.int32))
    assert len(specs) == len(args)
    _assert_no_partition_id(_tpu_lowering(mesh, fn, specs, *args))


@pytest.mark.parametrize("split", [False, True], ids=["today", "split"])
def test_moe_op_bodies_lower_without_partition_id(mesh, tpu7x_info, monkeypatch, split):
    monkeypatch.setattr(v2_host, "vmem_limit", lambda info=None: 63 << 20)
    monkeypatch.setattr(moe_tp_decode, "plan_blocks", lambda *a, **k: None)
    monkeypatch.setattr(
        moe_tp_decode.kernel, "can_use_moe_tp_decode", lambda *a, **k: False
    )
    monkeypatch.setenv("FUSED_MOE_PREFILL_KERNEL_V2", "1")
    # Kernels the interpret-mode tests built and memoized lower only for CPU.
    monkeypatch.setattr(v2_kernel, "_COMBINE_CACHE", {})
    monkeypatch.setattr(v2_kernel, "_BUILD_CACHE", {})
    tokens = 256
    rows = tokens // RANKS if split else tokens
    weights = [
        S((RANKS * EXPERTS, H, 2 * INTER), F8),
        S((RANKS * EXPERTS, INTER, H), F8),
        S((RANKS * EXPERTS, 2 * INTER), jnp.float32),
        S((RANKS * EXPERTS, H), jnp.float32),
        S((RANKS * H, 2 * INTER), F8),
        S((RANKS * 2 * INTER,), jnp.float32),
        S((RANKS * INTER, H), F8),
        S((RANKS * H,), jnp.float32),
    ]
    args = (
        S((RANKS * rows, H), jnp.bfloat16),
        S((RANKS * rows, EXPERTS), jnp.float32),
        S((EXPERTS,), jnp.float32),
        S((tokens + 64,), jnp.bool_),
        *weights,
        S((RANKS, 1), jnp.int32),
        S((RANKS, 1), jnp.int32),
    )
    specs = (P("d"), P("d"), P(), P()) + (P("d"),) * (len(args) - 4)
    body = (moe_tp_decode._split_body if split else moe_tp_decode._op_body)(
        _statics(), RANKS, "d"
    )
    fn = jax.shard_map(
        body, mesh=mesh, in_specs=specs, out_specs=P("d"), check_vma=False
    )
    v2.build_expert_kernel.cache_clear()
    v2.build_combine.cache_clear()
    try:
        _assert_no_partition_id(_tpu_lowering(mesh, fn, specs, *args))
    finally:
        v2.build_expert_kernel.cache_clear()
        v2.build_combine.cache_clear()


# 1024: one combine chunk; 4096: two chunks.
@pytest.mark.parametrize("tokens", [1024, 4096])
def test_token_ops_give_each_core_its_rows(mesh, tokens):
    """token_fn's ops, each core's index fed as data, against own_rows at
    `lax.axis_index` and the kernel's chunked sum, bit for bit."""
    h = 128
    rng = np.random.default_rng(tokens)
    x = jnp.asarray(rng.integers(-16, 17, (RANKS * tokens, h)), jnp.bfloat16)
    rank = jnp.arange(RANKS, dtype=jnp.int32).reshape(RANKS, 1)
    own, _, _ = prefill_token_split.token_fn(mesh, "own")
    gather, _, _ = prefill_token_split.token_fn(mesh, "gather")
    scatter, _, _ = prefill_token_split.token_fn(mesh, "scatter")

    full = jnp.tile(x[:tokens], (RANKS, 1))
    mine = np.asarray(jax.jit(own)(full, rank))
    by_index = np.asarray(
        _per_core(mesh, lambda y: v2.own_rows(y, lax.axis_index("d"), RANKS), full)
    )
    np.testing.assert_array_equal(_bits(mine), _bits(by_index))
    back = np.asarray(jax.jit(lambda y, r: gather(own(y, r)))(full, rank))
    np.testing.assert_array_equal(_bits(back), _bits(np.asarray(full)))

    summed = np.asarray(
        _per_core(mesh, lambda p: v1._chunked_psum(tokens, lambda r: p[r], "d"), x)
    )[:tokens]
    scattered = np.asarray(jax.jit(scatter)(x))
    rows = tokens // RANKS
    for core in range(RANKS):
        np.testing.assert_array_equal(
            _bits(scattered[core * rows : (core + 1) * rows]),
            _bits(_own_rows_np(summed, core)),
        )
    # Positive control: a wrong index gives other rows.
    shifted = np.asarray(jax.jit(own)(full, (rank + 1) % RANKS))
    assert not np.array_equal(_bits(shifted), _bits(mine))


# 2048 tokens: one combine chunk; 4096: two. Hidden 640 is not 3 parts of whole
# lane tiles, so the kernel does not take it.
@pytest.mark.parametrize(("tokens", "h"), [(2048, 768), (4096, 768), (2048, 640)])
def test_rotated_scatter_rows_gives_each_core_the_rows_of_scatter_rows(
    mesh, interpret, monkeypatch, tokens, h
):
    """v2.rotated_scatter_rows, each core's index fed as data, against
    scatter_rows, bit for bit: integer partials keep every bf16 sum exact in
    any order. One rotated kernel per combine chunk, none for a shape the
    kernel does not take."""
    places = [
        ((x, y, z), c) for x in (0, 1) for y in (0, 1) for z in (0, 1) for c in (0, 1)
    ]
    table = rotated_collectives.peer_table(places, "reduce_scatter")
    chunks = []
    chunked = rotated_collectives.reduce_scatter_chunks
    monkeypatch.setattr(
        rotated_collectives,
        "reduce_scatter_chunks",
        lambda *args, **kwargs: chunks.append(kwargs["chunk"])
        or chunked(*args, **kwargs),
    )
    rng = np.random.default_rng(tokens + h)
    x = jnp.asarray(rng.integers(-16, 17, (RANKS * tokens, h)), jnp.bfloat16)
    rank = jnp.arange(RANKS, dtype=jnp.int32).reshape(RANKS, 1)

    rotated = _per_core(
        mesh, lambda p, r: v2.rotated_scatter_rows(p, r.reshape(1), table, "d"), x, rank
    )
    xla = _per_core(mesh, lambda p: v2.scatter_rows(p, "d"), x)

    np.testing.assert_array_equal(_bits(rotated), _bits(xla))
    assert chunks == ([2048] if h == 768 else [])
