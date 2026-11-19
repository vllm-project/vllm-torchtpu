# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_WRDEFER (q38 L1 PLE-WRDEFER): the PLE checkpoint writer with K DMAs in
flight (``pool.write_region_ring``) writes exactly the stock writer's pool bytes.

Pool: f8e4m3fn[pages, block, 1, 4, 256] (the served PLE pool's per-token payload),
filled with random bytes, so every byte outside the written regions must survive.
Ids mix live (unique positive) rows with null (0) and pad (-1) rows at the head, in
the middle and at the tail, so the ring's read-ahead, its slot reuse and its drain
all meet skipped rows.

Classic leg (``interpret=True``; the mode pool_adapters documents for its typed
views, it keeps Mosaic's ``ref.bitcast``): the real ring writer with the real typed
store against (a) an order-free reference, each live row's bytes from the same
``typed_ldst.store_typed`` view in an encode-only kernel placed by numpy, and (b) the
l1_ple CPU stand-in of the stock writer (``pool_adapters.scatter_region`` with its
contract). The stock writer itself cannot run in classic interpret in jax 0.11.2
(``pl.program_id`` inside ``pl.run_scoped``). Classic interpret runs a DMA when it
starts, so this leg does not see DMA ordering.

DMA leg (the TPU interpreter, ``pltpu.InterpretParams``; see the DMA-leg block
below): the REAL stock writer against the ring writer, both under an interpreter
that models DMAs and semaphores. It cannot run a bitcast ref, so in this leg both
writers store through the same value-level stand-in of ``typed_ldst.store_typed``.

Composition: ``pooled_short_conv_speculative`` with the flag on vs off (off = the
stand-in), D07-b2 bounded 0 / 1 / 2 and the D14 fold on / off: with the fold on, a
D14 bucket (rows x window tokens) never calls ``write_region`` (the fold writes its
checkpoints itself), every other bucket goes through the ring.
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels import pool_adapters, typed_ldst
from vllm_torchtpu.kernels.qwen4_exp import ple_d14_fold
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows

FLAG = "VLLM_TPU_PLE_WRDEFER"
TAIL = (1, 4, 256)
TOKEN_ELEMS = 512  # BF16 values per pool token
STATE_LEN, DILATION, TAPS, WINDOW = 9, 3, 4, 4
REAL_CALL = pl.pallas_call
REAL_STORE = typed_ldst.store_typed


@pytest.fixture
def classic(monkeypatch):
    def call(*args, **kwargs):
        kwargs["interpret"] = True
        return REAL_CALL(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", call)
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)


def _stand_in(pool, values, ids, *, tok0=0):
    """The stock writer's contract (rows with id > 0 written, the rest left as they
    are) through pool_adapters.scatter_region (the l1_ple CPU stand-in): rows with
    id <= 0 write the null block's own bytes back to it (read before any write)."""
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    lanes = pool.shape[-1]
    vals = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(count, rows, lanes)
    safe = jnp.maximum(ids, 0)
    old = pool_adapters.gather_region(
        pool, safe, tok0=tok0, ntok=ntok, out_dtype=values.dtype, interpret=True
    )
    vals = jnp.where((ids > 0)[:, None, None], vals, old)
    return pool_adapters.scatter_region(pool, vals, safe, tok0=tok0, ntok=ntok, interpret=True)


def _ids(rng, count, pages, pattern):
    """Unique positive ids with null (0) / pad (-1) rows placed by ``pattern``."""
    ids = rng.permutation(np.arange(1, pages))[:count].astype(np.int32)
    if pattern == "mixed":  # head, middle, tail and a run of skipped rows
        skip = {0, count // 3, count // 3 + 1, count - 1} | set(
            rng.choice(count, size=max(1, count // 6), replace=False).tolist()
        )
        for k, r in enumerate(sorted(skip)):
            ids[r] = 0 if k % 2 else -1
    elif pattern == "tailpad":  # a decode bucket: live rows first, padding at the tail
        ids[count - max(1, count // 4) :] = 0
    elif pattern == "headnull":  # the first ring is all skipped
        ids[: min(count, 5)] = 0
    elif pattern == "allnull":
        ids[:] = np.where(np.arange(count) % 2, 0, -1)
    elif pattern != "dense":
        raise ValueError(pattern)
    return ids


def _writer_case(seed, count, *, channels=512, pattern="mixed", tok0=0):
    rng = np.random.default_rng(seed)
    elements = STATE_LEN * channels
    tpp = channels // TOKEN_ELEMS
    block = 2 * STATE_LEN * tpp
    pages = count + 4
    raw = rng.integers(0, 256, size=(pages, block) + TAIL, dtype=np.uint8)
    ids = _ids(rng, count, pages, pattern)
    values = rng.normal(size=(count, elements)).astype(np.float32)
    return (
        jnp.asarray(raw.view(jnp.float8_e4m3fn)),
        jnp.asarray(values, jnp.bfloat16),
        jnp.asarray(ids),
        tok0,
    )


def _order_free_reference(pool, values, ids, tok0, store=None):
    """Each live row's bytes from an encode-only kernel through the same typed store,
    placed by numpy: no DMA, no ring, no order."""
    store = store or REAL_STORE
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    lanes = pool.shape[-1]
    vals = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(count, rows, lanes)

    def encode(v_ref, o_ref):
        for i in range(count):
            store(o_ref.at[i], v_ref[i])

    enc = REAL_CALL(
        encode,
        out_shape=jax.ShapeDtypeStruct((count, ntok) + tuple(pool.shape[2:]), pool.dtype),
        interpret=True,
    )(vals)
    out = np.asarray(pool).view(np.uint8).copy()
    e = np.asarray(enc).view(np.uint8)
    ids = np.asarray(ids)
    for i in range(count):
        if ids[i] > 0:
            out[ids[i], tok0 : tok0 + ntok] = e[i]
    return out


def _bytes(x):
    return np.asarray(jax.block_until_ready(x)).view(np.uint8)


def _fresh_jit(fn):
    """A new function object per arm: the flag is read at trace time, so two arms
    must never share a jit trace cache entry."""
    return jax.jit(lambda *args: fn(*args))


def _ring(pool, values, ids, tok0, depth):
    fn = jax.jit(functools.partial(pool_mod.write_region_ring, tok0=tok0, depth=depth))
    return _bytes(fn(pool, values, ids))


# ---------------------------------------------------------------------------- classic leg
SIZES = [8, 16, 32, 64, 128, 256]


@pytest.mark.parametrize("count", SIZES)
@pytest.mark.parametrize("depth", [2, 4])
def test_ring_writes_the_stock_bytes(classic, count, depth):
    pool, values, ids, tok0 = _writer_case(count * 10 + depth, count)
    got = _ring(pool, values, ids, tok0, depth)
    want = _order_free_reference(pool, values, ids, tok0)
    before = np.asarray(pool).view(np.uint8)
    assert np.any(want != before), "the reference wrote nothing (vacuous case)"
    np.testing.assert_array_equal(got, want, err_msg="ring vs order-free reference")
    stand_in = _bytes(jax.jit(_stand_in, static_argnames="tok0")(pool, values, ids, tok0=tok0))
    np.testing.assert_array_equal(stand_in, want, err_msg="control: stand-in vs order-free reference")


PATTERNS = ["mixed", "tailpad", "headnull", "allnull", "dense"]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("count", [13, 95])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_ring_patterns_and_partial_rings(classic, pattern, count, depth):
    """Odd counts leave a partial last ring (the static tail rows)."""
    pool, values, ids, tok0 = _writer_case(count * 100 + depth * 7 + len(pattern), count, pattern=pattern)
    got = _ring(pool, values, ids, tok0, depth)
    want = _order_free_reference(pool, values, ids, tok0)
    if pattern == "allnull":
        np.testing.assert_array_equal(want, np.asarray(pool).view(np.uint8))
    np.testing.assert_array_equal(got, want, err_msg=f"ring K={depth} n={count} {pattern}")


@pytest.mark.parametrize(
    "count,depth,channels,tok0",
    [(8, 4, 10240, 0), (64, 4, 10240, 0), (16, 8, 512, 9), (5, 8, 512, 0), (1, 4, 512, 0)],
    ids=["served-n8-K4", "served-n64-K4", "tok0-9-K8", "count-below-K", "one-row"],
)
def test_ring_served_width_offsets_and_small_counts(classic, count, depth, channels, tok0):
    pool, values, ids, _ = _writer_case(count + depth + channels, count, channels=channels,
                                        pattern="dense" if count < 3 else "mixed")
    got = _ring(pool, values, ids, tok0, depth)
    want = _order_free_reference(pool, values, ids, tok0)
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------- DMA leg
# The TPU interpreter (pltpu.InterpretParams, dma_execution_mode="on_wait") models DMAs
# and semaphores: a started DMA is a pending task on its semaphore, run only when a wait
# on that semaphore finds the count too low (newest task first), so a DMA that is never
# waited never lands, a buffer reused before its DMA was waited is read after the reuse,
# and a wait that nothing will signal blocks forever (probe j-80694de8). It cannot run a
# bitcast ref, so both writers store through ``_value_store`` here (same bytes in both,
# and in the order-free reference). Each case runs in a subprocess with a timeout, so a
# hang is a failure, not a stuck leg; the subprocess imports the same staged tree.
DMA_TIMEOUT_S = 600


def _value_store(block_ref, values, *, lane_split=1):
    """Value-level stand-in for typed_ldst.store_typed (no ref bitcast): BF16 rows -> bytes."""
    del lane_split
    u = jnp.swapaxes(jax.lax.bitcast_convert_type(values, jnp.uint8), -1, -2)
    block_ref[...] = jax.lax.bitcast_convert_type(u, block_ref.dtype).reshape(block_ref.shape)


_DMA_CHILD = r"""
import json, os, sys
import numpy as np
import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from tests.kernels import test_qwen4_exp_ple_wrdefer as T
from vllm_torchtpu.kernels import typed_ldst
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod

count, depth, pattern, seed = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
params = pltpu.InterpretParams(dma_execution_mode="on_wait")
real = pl.pallas_call
pl.pallas_call = lambda *a, **k: real(*a, **{**k, "interpret": params})
typed_ldst.store_typed = T._value_store
pool, values, ids, tok0 = T._writer_case(seed, count, pattern=pattern)
before = np.asarray(pool).view(np.uint8)
want = T._order_free_reference(pool, values, ids, tok0, store=T._value_store)
stock = T._bytes(pool_mod._write_region_stock(pool, values, ids, tok0=tok0))
ring = T._bytes(pool_mod.write_region_ring(pool, values, ids, tok0=tok0, depth=depth))
print("Q38_DMA_LEG " + json.dumps({
    "origin": os.path.realpath(pool_mod.__file__),
    "reference_wrote": bool(np.any(want != before)),
    "stock_vs_reference": int((stock != want).sum()),
    "ring_vs_stock": int((ring != stock).sum()),
    "ring_bad_pages": sorted({int(p) for p in np.nonzero(ring != stock)[0]})[:8],
}), flush=True)
"""


def _dma_leg(count, depth, pattern, seed):
    import json
    import os
    import subprocess
    import sys

    try:
        r = subprocess.run(
            [sys.executable, "-c", _DMA_CHILD, str(count), str(depth), pattern, str(seed)],
            capture_output=True, text=True, timeout=DMA_TIMEOUT_S, env=dict(os.environ),
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"DMA interpreter run did not finish in {DMA_TIMEOUT_S} s (a wait that nothing signals)"
        ) from None
    lines = [x for x in r.stdout.splitlines() if x.startswith("Q38_DMA_LEG ")]
    assert lines, f"DMA leg child failed (rc {r.returncode}): {(r.stdout + r.stderr)[-3000:]}"
    out = json.loads(lines[-1].split(" ", 1)[1])
    print("Q38 L1-PLEWD DMA leg n=%d K=%d %s: %s" % (count, depth, pattern, out))
    assert out["origin"] == os.path.realpath(pool_mod.__file__), out["origin"]
    return out


def _check_dma(out, pattern):
    assert out["stock_vs_reference"] == 0, "control: the real stock writer vs the order-free reference"
    if pattern != "allnull":
        assert out["reference_wrote"], "vacuous case"
    assert out["ring_vs_stock"] == 0, f"ring vs the real stock writer: bad pages {out['ring_bad_pages']}"


@pytest.mark.parametrize("count", SIZES)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_dma_ring_matches_the_real_stock_writer(count, depth):
    _check_dma(_dma_leg(count, depth, "mixed", count * 10 + depth), "mixed")


@pytest.mark.parametrize("pattern", ["tailpad", "headnull", "allnull"])
@pytest.mark.parametrize("depth", [2, 4])
def test_dma_ring_patterns(pattern, depth):
    _check_dma(_dma_leg(13, depth, pattern, 13 * 100 + depth * 7 + len(pattern)), pattern)


# ---------------------------------------------------------------------------- dispatch + engagement
def test_flag_routes_the_writer(classic, monkeypatch):
    calls = []
    real_ring = pool_mod.write_region_ring
    monkeypatch.setattr(pool_mod, "_write_region_stock", lambda *a, **k: calls.append(("stock", k)) or a[0])
    monkeypatch.setattr(
        pool_mod, "write_region_ring", lambda *a, **k: calls.append(("ring", k)) or real_ring(*a, **k)
    )
    pool, values, ids, tok0 = _writer_case(3, 8)
    pool_mod.write_region(pool, values, ids, tok0=0)
    for value, depth in (("1", 1), ("2", 2), ("4", 4), ("8", 8)):
        monkeypatch.setenv(FLAG, value)
        pool_mod.write_region(pool, values, ids, tok0=0)
    monkeypatch.setenv(FLAG, "0")
    pool_mod.write_region(pool, values, ids, tok0=0)
    assert calls == [("stock", {"tok0": 0})] + [
        ("ring", {"tok0": 0, "depth": d}) for d in (1, 2, 4, 8)
    ] + [("stock", {"tok0": 0})], calls
    monkeypatch.setenv(FLAG, "3")
    with pytest.raises(ValueError):
        pool_mod.write_region(pool, values, ids, tok0=0)


def test_flag_on_is_bitwise_flag_off(classic, monkeypatch):
    """Through the dispatcher: flag off (stock writer -> the CPU stand-in) vs flag 4."""
    monkeypatch.setattr(pool_mod, "_write_region_stock", _stand_in)
    pool, values, ids, tok0 = _writer_case(11, 32)
    off = _bytes(_fresh_jit(pool_mod.write_region)(pool, values, ids))
    monkeypatch.setenv(FLAG, "4")
    on = _bytes(_fresh_jit(pool_mod.write_region)(pool, values, ids))
    assert np.any(off != np.asarray(pool).view(np.uint8))
    np.testing.assert_array_equal(on, off)


def test_ring_logs_its_engagement_once(classic, monkeypatch):
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("vllm.vllm_torchtpu.kernels.qwen4_exp.pool")
    h = _Keep(level=logging.INFO)
    log.addHandler(h)
    old = log.level
    log.setLevel(logging.INFO)
    monkeypatch.setattr(pool_mod, "_WRDEFER_SEEN", set())
    try:
        pool, values, ids, tok0 = _writer_case(5, 13)
        for _ in range(2):  # same shape twice: one line
            pool_mod.write_region_ring(pool, values, ids, depth=2)
    finally:
        log.removeHandler(h)
        log.setLevel(old)
    lines = [m for m in records if "q38 PLE-WRDEFER" in m]
    print("Q38 L1-PLEWD trace engagement:", lines)
    assert lines == [
        "q38 PLE-WRDEFER traced: 13 rows x 9 pool tokens, 2 checkpoint DMAs in flight, one grid step"
    ], records


def _pallas_eqns(jaxpr):
    out = []

    def walk(j):
        for e in j.eqns:
            if e.primitive.name == "pallas_call":
                out.append(e)
            for p in e.params.values():
                for sub in (p if isinstance(p, (list, tuple)) else [p]):
                    if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                        walk(sub.jaxpr)
                    elif hasattr(sub, "eqns"):
                        walk(sub)

    walk(jaxpr.jaxpr)
    return out


def _scratch_shapes(eqn):
    gm = eqn.params["grid_mapping"]
    n = gm.num_scratch_operands
    kernel = eqn.params["jaxpr"]
    avals = [v.aval for v in kernel.invars[len(kernel.invars) - n :]]
    return [tuple(getattr(a, "inner_aval", a).shape) for a in avals], tuple(gm.grid)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_ring_census(depth):
    """One pallas_call named q38_ple_wrdefer, grid (1,), K staging slots, K pool-layout
    slots, K read and K write semaphores (the flag value is the ring depth); the
    stock writer: grid (rows,)."""
    pool, values, ids, tok0 = _writer_case(1, 16)
    ring = _pallas_eqns(jax.make_jaxpr(functools.partial(pool_mod.write_region_ring, depth=depth))(pool, values, ids))
    assert len(ring) == 1
    assert "q38_ple_wrdefer" in str(ring[0].params.get("name_and_src_info", ring[0].params.get("name", "")))
    shapes, grid = _scratch_shapes(ring[0])
    print("Q38 L1-PLEWD census K=%d grid %s scratch %s" % (depth, grid, shapes))
    assert grid == (1,)
    assert shapes == [(depth, 18, 256), (depth, 9) + TAIL, (depth,), (depth,)], shapes
    stock = _pallas_eqns(jax.make_jaxpr(pool_mod._write_region_stock)(pool, values, ids))
    assert len(stock) == 1 and _scratch_shapes(stock[0])[1] == (16,)


# ---------------------------------------------------------------------------- composition
ORDER = ("x", "weight", "pool", "starts", "ids", "initial", "read_ids", "ckpt_ids", "windowed")


def _conv_case(seed, *, count, tokens, live, channels=512, per=WINDOW):
    """A speculative-conv bucket (the l1_ple fold test's layout): ``live`` requests of
    WINDOW tokens first, then padding; random pool bytes with written histories."""
    rng = np.random.default_rng(seed)
    tpp = channels // TOKEN_ELEMS
    block = 2 * STATE_LEN * tpp
    pages = 1 + max(live, 1) * WINDOW + 3
    ckpt = np.zeros((count, WINDOW), np.int32)
    ckpt[:live] = rng.permutation(np.arange(1, pages))[: live * WINDOW].reshape(live, WINDOW)
    offsets = rng.integers(0, WINDOW, size=count)
    rows = np.arange(count)
    read_ids = np.where(rows < live, ckpt[rows, offsets], 0).astype(np.int32)
    ids = ckpt[:, 0].copy()
    starts = np.minimum(np.arange(count + 1) * per, live * per).astype(np.int32)
    initial = rows < live
    windowed = rows < live
    raw = rng.integers(0, 256, size=(pages, block) + TAIL, dtype=np.uint8)
    pool = _stand_in(
        jnp.asarray(raw.view(jnp.float8_e4m3fn)),
        jnp.asarray(rng.normal(size=(pages, STATE_LEN * channels)), jnp.bfloat16),
        jnp.arange(pages, dtype=jnp.int32),
    )
    return dict(
        x=jnp.asarray(rng.normal(size=(tokens, channels)), jnp.bfloat16),
        weight=jnp.asarray(rng.normal(size=(channels, TAPS)), jnp.bfloat16),
        pool=pool,
        starts=jnp.asarray(starts),
        ids=jnp.asarray(ids),
        initial=jnp.asarray(initial),
        read_ids=jnp.asarray(read_ids),
        ckpt_ids=jnp.asarray(ckpt),
        windowed=jnp.asarray(windowed),
    )


# (count = max-num-seqs, tokens, live, what)
BUCKETS = [
    (48, 16, 4, "d14"),  # rows x window: the fold's bucket
    (48, 64, 16, "d14"),
    (48, 18, 4, "fallback"),  # not rows x window: the bounded / default path even with the fold on
    (4, 24, 4, "request"),  # above count x window: request_checkpoints (6-token requests)
]


@pytest.mark.parametrize("bucket", BUCKETS, ids=[f"c{b[0]}-t{b[1]}-n{b[2]}-{b[3]}" for b in BUCKETS])
@pytest.mark.parametrize("bounded,fold", [(0, False), (1, False), (2, False), (1, True), (2, True)])
def test_composes_with_bounded_and_the_d14_fold(classic, monkeypatch, bucket, bounded, fold):
    count, tokens, live, what = bucket
    inputs = _conv_case(count * 1000 + tokens * 10 + live, count=count, tokens=tokens, live=live,
                        per=tokens // live if what == "request" else WINDOW)
    monkeypatch.setattr(pool_mod, "_write_region_stock", _stand_in)
    rings = []
    real_ring = pool_mod.write_region_ring
    monkeypatch.setattr(
        pool_mod, "write_region_ring", lambda *a, **k: rings.append(a[1].shape[0]) or real_ring(*a, **k)
    )
    kw = dict(dilation=DILATION, state_len=STATE_LEN, bounded=bounded)
    if fold:
        kw["fold"] = True
    fn = functools.partial(pool_mod.pooled_short_conv_speculative, **kw)
    pool0, out0 = _fresh_jit(fn)(*(inputs[k] for k in ORDER))
    assert rings == []
    monkeypatch.setenv(FLAG, "4")
    pool1, out1 = _fresh_jit(fn)(*(inputs[k] for k in ORDER))
    takes_fold = fold and what == "d14"
    if takes_fold:
        assert ple_d14_fold.fold_applies(
            inputs["pool"], inputs["x"], inputs["weight"], inputs["ckpt_ids"][: -(-tokens // WINDOW)],
            dilation=DILATION, state_len=STATE_LEN,
        )
        assert rings == [], f"the fold writes its own checkpoints: write_region must not run ({rings})"
    else:
        assert len(rings) == 1, f"one checkpoint write through the ring ({rings})"
    before = np.asarray(inputs["pool"]).view(np.uint8)
    assert np.any(_bytes(pool0) != before), "no checkpoint written (vacuous case)"
    np.testing.assert_array_equal(_bytes(pool1), _bytes(pool0), err_msg="pool: flag 4 vs off")
    np.testing.assert_array_equal(
        np.asarray(out1.view(jnp.uint16)), np.asarray(out0.view(jnp.uint16)), err_msg="output: flag 4 vs off"
    )


def test_prefill_conv_composes(classic, monkeypatch):
    """pooled_short_conv (the non-speculative op) writes its final history through the ring."""
    inputs = _conv_case(77, count=6, tokens=40, live=5, per=8)
    monkeypatch.setattr(pool_mod, "_write_region_stock", _stand_in)
    args = [inputs[k] for k in ("x", "weight", "pool", "starts", "ids", "initial")]
    fn = functools.partial(pool_mod.pooled_short_conv, dilation=DILATION, state_len=STATE_LEN)
    pool0, out0 = _fresh_jit(fn)(*args)
    monkeypatch.setenv(FLAG, "2")
    pool1, out1 = _fresh_jit(fn)(*args)
    assert np.any(_bytes(pool0) != np.asarray(inputs["pool"]).view(np.uint8))
    np.testing.assert_array_equal(_bytes(pool1), _bytes(pool0))
    np.testing.assert_array_equal(np.asarray(out1.view(jnp.uint16)), np.asarray(out0.view(jnp.uint16)))
