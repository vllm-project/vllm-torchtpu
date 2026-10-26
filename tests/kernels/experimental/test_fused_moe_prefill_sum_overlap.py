# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""TPU_MOE_PREFILL_SUM_IN_KERNEL_OVERLAP_ADDS, no TPU needed: the one-pass stage-A
expert kernel adds each tile's staged expert rows into the per-token sums during the
next tile's gate/up matmuls, and the rows past the next tile's height in a loop
before it. Routed tiles stage their rows in the slot of their parity, the shared
expert's tiles in slot 0.

A replay of the kernel's tile loop over the routing plan of random, hot and skewed
routings checks the row bookkeeping: every live routed row is added exactly once,
while its staging slot still holds it, before the shared expert's tile writes its
token; padding rows only ever go to the spare row past the tokens. The replay is
shown to catch a missing loop, a read of the tile's own slot, a skipped row and a
carry through the shared tiles. In Pallas interpret mode the overlapped build
returns the serial build's output bit for bit.

TPU_MOE_PREFILL_ROW_TOKENS_RESIDENT_MAX_TOKENS, also in interpret mode: the slab
build that reads each route's token from the resident table matches the window
build bit for bit, and beside the overlapped sum it changes nothing."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.fused_moe_prefill import (
    fused_moe_prefill_kernel_v2 as v2,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host as v2_host

TILE_ROWS = 128
RUNG_STEP = TILE_ROWS // 4
K_TILES = 6144 // 256
GARBAGE = -1


def _plan(tokens, experts, topk, routing, seed):
    """The routed visits' row counts and the token of every slab row, from the
    kernel's own routing plan."""
    rng = np.random.default_rng(seed)
    scores = rng.random((tokens, experts))
    if routing == "hot":
        scores[:, 3] = 2.0
    elif routing == "skewed":
        scores *= rng.zipf(1.5, experts).clip(max=50)
    ids = np.argsort(-scores, axis=1)[:, :topk].astype(np.int32)
    valid = np.ones((tokens, topk), bool)
    valid[-3:] = False
    valid[rng.random((tokens, topk)) < 0.02] = False
    row_token, _, _, _, visit_start, visit_rows, routed, _ = v2._routing_plan(
        jnp.asarray(ids),
        jnp.asarray(valid),
        experts,
        jnp.ones((tokens,), jnp.float32),
    )
    routed = int(routed)
    rows = [int(n) for n in np.asarray(visit_rows)[:routed]]
    assert list(np.asarray(visit_start)[:routed]) == list(np.cumsum([0, *rows])[:-1])
    return rows, np.asarray(row_token)


def _replay(visit_rows, tokens, flaw=None):
    """build_expert_kernel's tile loop with overlap_adds as row bookkeeping: in order,
    each add into the sums ("add", the slab row whose token the table names or None
    for the spare row, the slab row the staging slot holds there) and each token the
    shared expert's tile writes ("final", token, slab row held)."""
    routes = sum(visit_rows)
    starts = np.cumsum([0, *visit_rows])
    stage = [[GARBAGE] * TILE_ROWS, [GARBAGE] * TILE_ROWS]
    prev_start = prev_live = 0
    events = []
    g = 0

    def add(r):
        named = prev_start + r if r < prev_live else None
        read = g % 2 if flaw == "reads its own slot" else 1 - g % 2
        events.append(("add", named, stage[read][r]))

    for v, n in enumerate([*visit_rows, tokens]):
        shared = v == len(visit_rows)
        for t in range(-(-n // TILE_ROWS)):
            start = (routes if shared else int(starts[v])) + t * TILE_ROWS
            live = min(n - t * TILE_ROWS, TILE_ROWS)
            rows = -(-live // RUNG_STEP) * RUNG_STEP
            covered = 0 if shared else rows
            if flaw == "covered past the rung":
                covered += 8
            if flaw != "no loop":
                for r in range(covered, prev_live):
                    add(r)
            batches = [] if shared else [range(q, q + 4) for q in range(0, rows, 4)]
            for k in range(K_TILES):
                for j, q in enumerate(batches):
                    if j * K_TILES // len(batches) == k:
                        for r in q:
                            add(r)
            slot = 0 if shared else g % 2
            stage[slot] = [start + r if r < live else GARBAGE for r in range(TILE_ROWS)]
            if shared:
                for r in range(TILE_ROWS):
                    events.append(("final", t * TILE_ROWS + r, stage[slot][r]))
            if not (shared and flaw == "carry through the shared tiles"):
                prev_start, prev_live = (0, 0) if shared else (start, live)
            g += 1
    return events


def _check(events, row_token, routes, tokens):
    added = np.zeros(routes, np.int64)
    written = np.zeros(tokens, bool)
    for kind, named, held in events:
        if kind == "final":
            assert held == routes + named, "the staging slot lost the shared row"
            assert not written[named], "a token written twice"
            written[named] = True
        elif named is not None:
            assert named < routes, "a shared row added as a routed row"
            assert held == named, "the staging slot no longer holds the row"
            assert not written[row_token[named]], "a row added after its token"
            added[named] += 1
    assert (added == 1).all(), f"rows added {np.bincount(added)} times"
    assert written.all(), "a token never written"


def _tile_tokens_distinct(visit_rows, row_token):
    """Each batch of one tile loads all its sums before storing them, which needs the
    tile's rows to belong to distinct tokens."""
    start = 0
    for n in visit_rows:
        for t in range(0, n, TILE_ROWS):
            tile = row_token[start + t : start + min(n, t + TILE_ROWS)]
            assert len(set(tile.tolist())) == len(tile)
        start += n


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize(
    "tokens,experts,topk,routing",
    [
        (1024, 256, 8, "uniform"),
        (1024, 256, 8, "hot"),
        (1024, 256, 8, "skewed"),
        (256, 16, 8, "hot"),
        (128, 8, 2, "skewed"),
    ],
)
def test_every_row_is_added_once_before_its_token_is_written(
    tokens, experts, topk, routing, seed
):
    visit_rows, row_token = _plan(tokens, experts, topk, routing, seed)
    _tile_tokens_distinct(visit_rows, row_token)
    _check(_replay(visit_rows, tokens), row_token, sum(visit_rows), tokens)


@pytest.mark.parametrize(
    "flaw",
    [
        "no loop",
        "reads its own slot",
        "covered past the rung",
        "carry through the shared tiles",
    ],
)
def test_the_replay_check_catches_a_flawed_schedule(flaw):
    visit_rows, row_token = _plan(1024, 256, 8, "skewed", 0)
    assert any(n % TILE_ROWS > RUNG_STEP for n in visit_rows)
    with pytest.raises(AssertionError):
        _check(_replay(visit_rows, 1024, flaw), row_token, sum(visit_rows), 1024)


@pytest.fixture
def interpret(monkeypatch):
    monkeypatch.setattr(v2_host, "vmem_limit", lambda info=None: 63 << 20)
    monkeypatch.setattr(
        pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    v2.build_expert_kernel.cache_clear()
    v2.build_combine.cache_clear()
    yield
    v2.build_expert_kernel.cache_clear()
    v2.build_combine.cache_clear()


def _stage_a_runner(tokens=384, h=640, experts=8, inter=128, topk=4):
    """The op in Pallas interpret mode with down="dot" (no CPU lowering for the
    explicit MXU's pair pack, so h has an odd number of lane blocks), loop issue and
    128-row tiles. One expert in every token and another in the first third:
    multi-tile visits, short tails before full tiles, full tiles before short ones,
    dropped routes and three shared tiles."""
    f8 = jnp.float8_e4m3fn
    ks = jax.random.split(jax.random.key(0), 10)
    x = jax.random.normal(ks[0], (tokens, h)).astype(jnp.bfloat16)
    w1 = (jax.random.normal(ks[1], (experts, h, 2 * inter)) * 0.5).astype(f8)
    w2 = (jax.random.normal(ks[2], (experts, inter, h)) * 0.5).astype(f8)
    w1s = jax.random.uniform(ks[3], (experts, 2 * inter), minval=0.01, maxval=0.03)
    w2s = jax.random.uniform(ks[4], (experts, h), minval=0.01, maxval=0.03)
    sw1 = (jax.random.normal(ks[5], (h, 2 * inter)) * 0.5).astype(f8)
    sw2 = (jax.random.normal(ks[6], (inter, h)) * 0.5).astype(f8)
    sw1s = jax.random.uniform(ks[7], (2 * inter,), minval=0.01, maxval=0.03)
    sw2s = jax.random.uniform(ks[8], (h,), minval=0.01, maxval=0.03)
    gate = (jax.random.normal(ks[9], (h,)) * 0.05).astype(jnp.bfloat16)
    rng = np.random.default_rng(1)
    scores = rng.random((tokens, experts))
    scores[:, 0] = 2.0
    scores[: tokens // 3, 1] = 3.0
    ids = jnp.asarray(np.argsort(-scores, axis=1)[:, :topk].astype(np.int32))
    tw = rng.dirichlet(np.ones(topk), size=tokens).astype(np.float32)
    tw[3, 1] = 0.0
    tw[-1] = 0.0
    tw = jnp.asarray(tw)
    cpu = jax.devices("cpu")[0]
    mesh = Mesh(np.asarray([cpu]), ("d",))

    def run(**kw):
        def f(*args):
            return v2.fused_moe_prefill(
                *args,
                axis_name="d",
                routed_scale=2.5,
                nb=3,
                down="dot",
                issue="loop",
                tile_min=8192,
                flat=True,
                **kw,
            )

        g = jax.shard_map(f, mesh=mesh, in_specs=P(), out_specs=P(), check_vma=False)
        args = (x, w1, w2, w1s, w2s, sw1, sw1s, sw2, sw2s, gate, tw, ids)
        args = jax.device_put(args, cpu)
        return np.asarray(jax.jit(g)(*args).astype(jnp.float32))

    return run


def test_overlapped_adds_match_the_serial_sum_bit_for_bit(interpret):
    tokens = 384
    assert v2.sum_columns(tokens, 640, 128, 128) == 640
    run = _stage_a_runner(tokens)
    serial = run(sum_in_kernel_max_tokens=tokens)
    overlapped = run(sum_in_kernel_max_tokens=tokens, sum_overlap_adds=True)
    np.testing.assert_array_equal(overlapped.view(np.uint32), serial.view(np.uint32))
    slab = run()
    rel = np.linalg.norm(overlapped - slab) / np.linalg.norm(slab)
    assert 0 < rel < 1e-2, rel


def test_resident_tokens_match_token_windows_beside_the_sum(interpret, monkeypatch):
    """TPU_MOE_PREFILL_ROW_TOKENS_RESIDENT_MAX_TOKENS: the slab build that reads each
    route's token from the resident table returns the window build's output bit for
    bit; beside the overlapped in-kernel sum, whose build keeps the table resident
    already, the op builds the same kernel and returns the same output."""
    tokens = 384
    run = _stage_a_runner(tokens)
    names = []
    pallas_call = pl.pallas_call

    def named(*args, **kwargs):
        names.append(kwargs.get("name", ""))
        return pallas_call(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", named)

    def built(**kw):
        v2.build_expert_kernel.cache_clear()
        names.clear()
        out = run(**kw)
        return out, [n for n in names if n.startswith("fused_moe_prefill-")]

    windows, windows_names = built()
    resident, resident_names = built(row_tokens_resident_max_tokens=tokens)
    assert windows_names[0].endswith("-loop") and resident_names[0].endswith("-loop-rt")
    np.testing.assert_array_equal(resident.view(np.uint32), windows.view(np.uint32))

    summed = dict(sum_in_kernel_max_tokens=tokens, sum_overlap_adds=True)
    alone, alone_names = built(**summed)
    both, both_names = built(**summed, row_tokens_resident_max_tokens=tokens)
    assert alone_names == both_names and both_names[0].endswith("-sumk-pipe")
    np.testing.assert_array_equal(both.view(np.uint32), alone.view(np.uint32))
