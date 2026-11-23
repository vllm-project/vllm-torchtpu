# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_WRDEFER (q38 PLE-WRDEFER on the R form; port of L1's PLE-WRDEFER): the PLE
checkpoint writer with K DMAs in flight (``pool.write_region_ring``) writes exactly the stock
writer's pool bytes, and the pooled short convs return the same outputs.

Pool: f8e4m3fn[pages, block, 1, 4, 256] (the served PLE pool's per-token payload), filled with
random bytes, so every byte outside the written regions must survive. Ids mix live (unique
positive) rows with null (0) and pad (-1) rows at the head, in the middle and at the tail, so the
ring's read-ahead, its slot reuse and its drain all meet skipped rows.

Legs (L1's design, credit L1 PLE-WRDEFER, q38-L1-plewd 2c0da1bd8):
- classic (``interpret=True``, keeps Mosaic's ``ref.bitcast``): the real ring writer with the real
  typed store against an order-free reference (each live row's bytes from the same
  ``typed_ldst.store_typed`` view in an encode-only kernel, placed by numpy) and against the CPU
  stand-in of the stock writer (``pool_adapters.scatter_region``). The stock writer itself cannot
  run in classic interpret (``pl.program_id`` inside ``pl.run_scoped``). Classic interpret runs a
  DMA when it starts, so this leg does not see DMA ordering.
- dma (the TPU interpreter, ``pltpu.InterpretParams(dma_execution_mode="on_wait")``): the REAL
  stock writer against the ring writer, both storing through one value-level stand-in of
  ``typed_ldst.store_typed`` (the interpreter cannot run a bitcast ref). A started DMA lands only
  when a wait runs it, so a missing wait / drain or a slot reused before its DMA was waited gives
  wrong bytes, and a wait nothing signals hangs (caught by the subprocess timeout).
- conv (TPU interpreter, value-level typed load and store in both arms): the whole
  ``pooled_short_conv_speculative`` (token path and request path, history-row kernel on and off)
  and ``pooled_short_conv`` with the flag set against the flag unset, whose writer is the REAL
  stock writer: pool bytes and output bits.
- classic conv: the same ops with the real typed store, flag set vs flag unset with the stock
  writer replaced by the stand-in.
- ledger: the ring kernel's body run on the host with recording stand-ins for the DMA, the store
  and the control flow (``pl.when``, ``fori_loop``): every started DMA is waited exactly once, no
  slot is stored into while its previous write is in flight, every read is waited before its slot
  is stored from, nothing is in flight at the end.
"""

import functools
import importlib
import json
import logging
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pool_adapters, typed_ldst
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows

FLAG = "VLLM_TPU_PLE_WRDEFER"
HRK = "VLLM_TPU_PLE_HISTORY_ROW_KERNEL"
SKIP = "VLLM_TPU_PLE_SKIP_IDLE_ROWS"
TAIL = (1, 4, 256)
TOKEN_ELEMS = 512  # BF16 values per pool token
STATE_LEN, DILATION, TAPS, WINDOW = 9, 3, 4, 4
REAL_CALL = pl.pallas_call
REAL_STORE = typed_ldst.store_typed
SUB_TIMEOUT_S = 900
HERE = os.path.abspath(__file__)


def _clear_env(monkeypatch):
    for name in (FLAG, HRK, SKIP, "VLLM_KV_CACHE_LAYOUT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def classic(monkeypatch):
    def call(*args, **kwargs):
        kwargs["interpret"] = True
        return REAL_CALL(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", call)
    _clear_env(monkeypatch)


def _stand_in(pool, values, ids, *, tok0=0, skip_idle_source=False):
    """The stock writer's contract (rows with id > 0 written, the rest left as they are)
    through pool_adapters.scatter_region (L1's CPU stand-in): rows with id <= 0 write the
    null block's own bytes back to it (read before any write)."""
    del skip_idle_source
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
        skip = {r for r in skip if r < count}  # one row: (count // 3 + 1) is past the end
        for k, r in enumerate(sorted(skip)):
            ids[r] = 0 if k % 2 else -1
    elif pattern == "tailpad":  # a decode bucket: live rows first, padding at the tail
        ids[count - max(1, count // 4) :] = 0
    elif pattern == "headnull":  # the first ring is all skipped
        ids[: min(count, 5)] = 0
    elif pattern == "allnull":
        ids[:] = np.where(np.arange(count) % 2, 0, -1)
    elif pattern == "single":  # exactly one live row, in the middle
        keep = ids[count // 2]
        ids[:] = 0
        ids[count // 2] = keep
    elif pattern != "dense":
        raise ValueError(pattern)
    return ids


def _writer_case(seed, count, *, channels=512, pattern="mixed", tok0=0):
    rng = np.random.default_rng(seed)
    elements = STATE_LEN * channels
    tpp = channels // TOKEN_ELEMS if channels >= TOKEN_ELEMS else 1
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


def _bits16(x):
    return np.asarray(jax.block_until_ready(x)).view(np.uint16)


def _fresh_jit(fn):
    """A new function object per arm: the flags are read at trace time, so two arms
    must never share a jit trace cache entry."""
    return jax.jit(lambda *args: fn(*args))


def _ring(pool, values, ids, tok0, depth):
    fn = jax.jit(functools.partial(pool_mod.write_region_ring, tok0=tok0, depth=depth))
    return _bytes(fn(pool, values, ids))


# ---------------------------------------------------------------------------- classic leg
SIZES = [8, 16, 32, 64, 128, 160, 192, 256]


@pytest.mark.parametrize("count", SIZES)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_ring_writes_the_stock_bytes(classic, count, depth):
    pool, values, ids, tok0 = _writer_case(count * 10 + depth, count)
    got = _ring(pool, values, ids, tok0, depth)
    want = _order_free_reference(pool, values, ids, tok0)
    before = np.asarray(pool).view(np.uint8)
    assert np.any(want != before), "the reference wrote nothing (vacuous case)"
    np.testing.assert_array_equal(got, want, err_msg="ring vs order-free reference")
    stand_in = _bytes(jax.jit(_stand_in, static_argnames="tok0")(pool, values, ids, tok0=tok0))
    np.testing.assert_array_equal(stand_in, want, err_msg="control: stand-in vs order-free reference")


PATTERNS = ["mixed", "tailpad", "headnull", "allnull", "single", "dense"]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("count", [13, 95])
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_ring_patterns_and_partial_rings(classic, pattern, count, depth):
    """Odd counts leave a partial last ring (the static tail rows)."""
    pool, values, ids, tok0 = _writer_case(count * 100 + depth * 7 + len(pattern), count, pattern=pattern)
    got = _ring(pool, values, ids, tok0, depth)
    want = _order_free_reference(pool, values, ids, tok0)
    before = np.asarray(pool).view(np.uint8)
    if pattern == "allnull":
        np.testing.assert_array_equal(want, before)
    else:
        assert np.any(want != before), "vacuous case"
    np.testing.assert_array_equal(got, want, err_msg=f"ring K={depth} n={count} {pattern}")


@pytest.mark.parametrize(
    "count,depth,channels,tok0",
    [(8, 4, 10240, 0), (160, 4, 10240, 0), (16, 8, 512, 9), (5, 8, 512, 0), (1, 4, 512, 0),
     (1, 1, 256, 0)],
    ids=["served-n8-K4", "served-n160-K4", "tok0-9-K8", "count-below-K", "one-row", "one-row-C256-K1"],
)
def test_ring_served_width_offsets_and_small_counts(classic, count, depth, channels, tok0):
    pool, values, ids, _ = _writer_case(count + depth + channels, count, channels=channels,
                                        pattern="dense" if count < 3 else "mixed", tok0=tok0)
    got = _ring(pool, values, ids, tok0, depth)
    want = _order_free_reference(pool, values, ids, tok0)
    assert np.any(want != np.asarray(pool).view(np.uint8)), "vacuous case"
    np.testing.assert_array_equal(got, want)


# ---------------------------------------------------------------------------- subprocess legs
def _value_store(block_ref, values, *, lane_split=1):
    """Value-level stand-in for typed_ldst.store_typed (no ref bitcast): BF16 rows -> bytes."""
    del lane_split
    u = jnp.swapaxes(jax.lax.bitcast_convert_type(values, jnp.uint8), -1, -2)
    block_ref[...] = jax.lax.bitcast_convert_type(u, block_ref.dtype).reshape(block_ref.shape)


def _value_load(block_ref, *, view_dtype, lane_split=1):
    """Inverse of ``_value_store`` (value-level stand-in for typed_ldst.load_typed)."""
    raw = block_ref[...]
    lanes = raw.shape[-1]
    if jnp.dtype(raw.dtype) == jnp.dtype(view_dtype):
        out = raw.reshape(-1, lanes)
    else:
        ratio = jnp.dtype(view_dtype).itemsize // jnp.dtype(raw.dtype).itemsize
        u = jax.lax.bitcast_convert_type(raw, jnp.uint8).reshape(-1, ratio, lanes)
        out = jax.lax.bitcast_convert_type(jnp.swapaxes(u, -1, -2), view_dtype)
    return out.reshape(-1, lanes // lane_split)


def _load_self():
    """This test module, imported from its own file (tests/ is not a package here)."""
    spec = importlib.util.spec_from_file_location("q38_rplew_tests", HERE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_CHILD = r"""
import importlib.util, json, os, sys
spec = importlib.util.spec_from_file_location("q38_rplew_tests", sys.argv[1])
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)
print("Q38_RPLEW_CHILD " + json.dumps(T._child_main(json.loads(sys.argv[2]))), flush=True)
"""


def _sub(what, **kw):
    env = dict(os.environ)
    for name in (FLAG, HRK, SKIP, "VLLM_KV_CACHE_LAYOUT"):
        env.pop(name, None)
    try:
        r = subprocess.run(
            [sys.executable, "-c", _CHILD, HERE, json.dumps(dict(what=what, **kw))],
            capture_output=True, text=True, timeout=SUB_TIMEOUT_S, env=env,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"TPU-interpreter run did not finish in {SUB_TIMEOUT_S} s (a wait that nothing signals)"
        ) from None
    lines = [x for x in r.stdout.splitlines() if x.startswith("Q38_RPLEW_CHILD ")]
    assert lines, f"child failed (rc {r.returncode}): {(r.stdout + r.stderr)[-3000:]}"
    out = json.loads(lines[-1].split(" ", 1)[1])
    print("Q38 RPLEW %s %s: %s" % (what, kw, out))
    assert out["origin"] == os.path.realpath(pool_mod.__file__), out["origin"]
    return out


def _interpret_on_wait():
    params = pltpu.InterpretParams(dma_execution_mode="on_wait")
    pl.pallas_call = lambda *a, **k: REAL_CALL(*a, **{**k, "interpret": params})
    typed_ldst.store_typed = _value_store
    typed_ldst.load_typed = _value_load


def _child_main(a):
    """Runs in a fresh interpreter (no flag in the environment at start)."""
    import vllm_torchtpu.kernels.qwen4_exp.pool as P

    _interpret_on_wait()
    out = {"origin": os.path.realpath(P.__file__)}
    if a["what"] == "dma":
        pool, values, ids, tok0 = _writer_case(a["seed"], a["count"], pattern=a["pattern"],
                                               channels=a.get("channels", 512))
        before = np.asarray(pool).view(np.uint8)
        want = _order_free_reference(pool, values, ids, tok0, store=_value_store)
        stock = _bytes(jax.jit(P._write_region_stock)(pool, values, ids))
        stock_skip = _bytes(jax.jit(functools.partial(P._write_region_stock, skip_idle_source=True))(
            pool, values, ids))
        ring = _bytes(jax.jit(functools.partial(P.write_region_ring, depth=a["depth"]))(pool, values, ids))
        out.update(
            reference_wrote=bool(np.any(want != before)),
            stock_vs_reference=int((stock != want).sum()),
            stock_skip_vs_stock=int((stock_skip != stock).sum()),
            ring_vs_stock=int((ring != stock).sum()),
            ring_bad_pages=sorted({int(p) for p in np.nonzero(ring != stock)[0]})[:8],
        )
        return out
    # conv: flag unset (real stock writer) vs flag set, same inputs, fresh jits.
    case = _conv_case(**a["case"])
    rings = []
    real_ring = P.write_region_ring
    P.write_region_ring = lambda *x, **k: rings.append(int(x[1].shape[0])) or real_ring(*x, **k)
    if a.get("hrk"):
        os.environ[HRK] = "1"
    if a.get("skip"):
        os.environ[SKIP] = "1"
    fn = _conv_fn(P, a["op"])
    args = _conv_args(case, a["op"])
    pool0, out0 = _fresh_jit(fn)(*args)
    ring_off = len(rings)
    os.environ[FLAG] = str(a["depth"])
    pool1, out1 = _fresh_jit(fn)(*args)
    before = np.asarray(case["pool"]).view(np.uint8)
    out.update(
        wrote=int((_bytes(pool0) != before).sum()),
        pool_diff=int((_bytes(pool1) != _bytes(pool0)).sum()),
        out_diff=int((_bits16(out1) != _bits16(out0)).sum()),
        ring_calls_off=ring_off,
        ring_calls_on=len(rings) - ring_off,
        ring_rows=rings,
    )
    return out


def _check_dma(out, pattern):
    assert out["stock_vs_reference"] == 0, "control: the real stock writer vs the order-free reference"
    assert out["stock_skip_vs_stock"] == 0, "control: the stock writer with skip_idle_source vs without"
    if pattern != "allnull":
        assert out["reference_wrote"], "vacuous case"
    assert out["ring_vs_stock"] == 0, f"ring vs the real stock writer: bad pages {out['ring_bad_pages']}"


@pytest.mark.parametrize("count", SIZES)
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_dma_ring_matches_the_real_stock_writer(count, depth):
    _check_dma(_sub("dma", count=count, depth=depth, pattern="mixed", seed=count * 10 + depth), "mixed")


@pytest.mark.parametrize("pattern", ["tailpad", "headnull", "allnull", "single"])
@pytest.mark.parametrize("depth", [2, 4])
def test_dma_ring_patterns(pattern, depth):
    _check_dma(_sub("dma", count=13, depth=depth, pattern=pattern, seed=13 * 100 + depth * 7 + len(pattern)),
               pattern)


# ---------------------------------------------------------------------------- conv cases
ORDER = ("x", "weight", "pool", "starts", "ids", "initial", "read_ids", "ckpt_ids", "windowed")


def _conv_case(seed, count, tokens, lengths, windowed=None, ids_on=None, initial=None, channels=256):
    """A speculative-conv bucket: request ``r`` holds ``lengths[r]`` tokens (padded rows hold
    0); random pool bytes with written histories; unique checkpoint ids per request."""
    rng = np.random.default_rng(seed)

    def pad(v, default, fill):
        v = list(default if v is None else v)
        assert len(v) <= count, (len(v), count)
        return np.asarray(v + [fill] * (count - len(v)), bool)

    lengths = list(lengths) + [0] * (count - len(lengths))
    windowed = pad(windowed, [], 1)
    ids_on = pad(ids_on, [n > 0 for n in lengths], 0)
    initial = pad(initial, [], 1)
    starts = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    assert starts[-1] <= tokens, (starts[-1], tokens)
    tpp = channels // TOKEN_ELEMS if channels >= TOKEN_ELEMS else 1
    block = 2 * STATE_LEN * tpp
    pages = 1 + max(count, 1) * (WINDOW + 1) + 3
    perm = rng.permutation(np.arange(1, pages))
    ckpt = perm[: count * WINDOW].reshape(count, WINDOW).astype(np.int32)
    ckpt = np.where(ids_on[:, None], ckpt, 0).astype(np.int32)
    ids = np.where(ids_on, perm[count * WINDOW : count * WINDOW + count], 0).astype(np.int32)
    offsets = rng.integers(0, WINDOW, size=count)
    read_ids = np.where(ids_on, ckpt[np.arange(count), offsets], 0).astype(np.int32)
    raw = rng.integers(0, 256, size=(pages, block) + TAIL, dtype=np.uint8)
    return dict(
        x=jnp.asarray(rng.normal(size=(tokens, channels)), jnp.bfloat16),
        weight=jnp.asarray(rng.normal(size=(channels, TAPS)), jnp.bfloat16),
        pool=jnp.asarray(raw.view(jnp.float8_e4m3fn)),
        starts=jnp.asarray(starts),
        ids=jnp.asarray(ids),
        initial=jnp.asarray(initial),
        read_ids=jnp.asarray(read_ids),
        ckpt_ids=jnp.asarray(ckpt),
        windowed=jnp.asarray(windowed),
    )


def _conv_fn(P, op):
    if op == "spec":
        return functools.partial(P.pooled_short_conv_speculative, dilation=DILATION, state_len=STATE_LEN)
    return functools.partial(P.pooled_short_conv, dilation=DILATION, state_len=STATE_LEN)


def _conv_args(case, op):
    keys = ORDER if op == "spec" else ORDER[:6]
    return [case[k] for k in keys]


def _w(n, k=WINDOW):
    return [k] * n


# name -> (op, case kwargs, path). Token path: T <= count * window; request path: T > count * window.
CONV_CASES = {
    "tok-T8-c48-mixedlen-idle": ("spec", dict(seed=1, count=48, tokens=8, lengths=[4, 1, 2, 1],
                                              ids_on=[1, 0, 1, 1]), "token"),
    "tok-T16-c48-w4": ("spec", dict(seed=2, count=48, tokens=16, lengths=_w(4)), "token"),
    "tok-T32-c48-w4-short-tail": ("spec", dict(seed=3, count=48, tokens=32, lengths=_w(7) + [3]), "token"),
    "tok-T64-c48-w4-nowin-prefill": ("spec", dict(seed=4, count=48, tokens=64,
                                                  lengths=_w(12) + [9], windowed=[1] * 12 + [0],
                                                  initial=[1] * 12 + [0]), "token"),
    "tok-T128-c48-w4": ("spec", dict(seed=5, count=48, tokens=128, lengths=_w(32)), "token"),
    "tok-T160-c40-w4": ("spec", dict(seed=6, count=40, tokens=160, lengths=_w(40)), "token"),
    "tok-T160-c48-w4-idle": ("spec", dict(seed=7, count=48, tokens=160, lengths=_w(40),
                                          ids_on=[i % 9 != 4 for i in range(40)]), "token"),
    "tok-T192-c48-w4": ("spec", dict(seed=8, count=48, tokens=192, lengths=_w(48)), "token"),
    "tok-T256-c64-w4": ("spec", dict(seed=9, count=64, tokens=256, lengths=_w(64)), "token"),
    "tok-T16-c48-allidle": ("spec", dict(seed=10, count=48, tokens=16, lengths=_w(4), ids_on=[0] * 4), "token"),
    "tok-T8-c48-single": ("spec", dict(seed=11, count=48, tokens=8, lengths=[1]), "token"),
    "req-T192-c40-w4": ("spec", dict(seed=12, count=40, tokens=192, lengths=_w(40)), "request"),
    "req-T256-c48-w4-idle": ("spec", dict(seed=13, count=48, tokens=256, lengths=_w(48),
                                          ids_on=[i % 7 != 3 for i in range(48)]), "request"),
    "req-T24-c4-len6-mixedwin": ("spec", dict(seed=14, count=4, tokens=24, lengths=[6, 6, 6, 6],
                                              windowed=[1, 0, 1, 0], initial=[1, 1, 0, 0]), "request"),
    "req-T40-c4-allidle": ("spec", dict(seed=15, count=4, tokens=40, lengths=[10, 10, 10, 10],
                                        ids_on=[0, 0, 0, 0]), "request"),
    "pre-T40-c6": ("pre", dict(seed=16, count=6, tokens=40, lengths=[8, 8, 8, 8, 8]), "prefill"),
    "pre-T64-c8-idle": ("pre", dict(seed=17, count=8, tokens=64, lengths=[8] * 8,
                                    ids_on=[1, 0, 1, 1, 0, 1, 1, 1]), "prefill"),
}
ALLIDLE = {"tok-T16-c48-allidle", "req-T40-c4-allidle"}


def _check_conv(out, name, *, skip=False):
    if name in ALLIDLE:
        assert out["wrote"] == 0, "all-idle case wrote bytes"
    else:
        assert out["wrote"] > 0, "no checkpoint written (vacuous case)"
    assert out["ring_calls_off"] == 0, "flag unset took the ring"
    assert out["ring_calls_on"] == (0 if skip else 1), out["ring_rows"]
    assert out["pool_diff"] == 0, "pool: flag set vs unset"
    assert out["out_diff"] == 0, "output: flag set vs unset"


@pytest.mark.parametrize("name", sorted(CONV_CASES))
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_conv_flag_on_is_the_stock_writer(name, depth):
    """TPU interpreter, history-row kernel on (the R-form decode flag): flag K vs unset (real stock writer)."""
    op, case, _ = CONV_CASES[name]
    _check_conv(_sub("conv", op=op, case=case, depth=depth, hrk=1), name)


@pytest.mark.parametrize("name", ["tok-T160-c48-w4-idle", "req-T192-c40-w4"])
def test_conv_served_width(name):
    """C 10240 (the R-form PLE width, 180 pool tokens per checkpoint), flag 4 vs unset."""
    op, case, _ = CONV_CASES[name]
    _check_conv(_sub("conv", op=op, case={**case, "channels": 10240}, depth=4, hrk=1), name)


@pytest.mark.parametrize("name", ["tok-T160-c48-w4-idle", "req-T24-c4-len6-mixedwin", "pre-T40-c6"])
def test_conv_without_history_row_kernel(name):
    op, case, _ = CONV_CASES[name]
    _check_conv(_sub("conv", op=op, case=case, depth=4, hrk=0), name)


@pytest.mark.parametrize("name", ["tok-T160-c48-w4-idle", "req-T256-c48-w4-idle", "tok-T8-c48-single"])
def test_conv_skip_idle_rows_keeps_the_stock_writer(name):
    """VLLM_TPU_PLE_SKIP_IDLE_ROWS (skip_idle_source=True): flag 4 makes no ring call, same bytes."""
    op, case, _ = CONV_CASES[name]
    _check_conv(_sub("conv", op=op, case=case, depth=4, hrk=1, skip=1), name, skip=True)


@pytest.mark.parametrize("name", ["tok-T16-c48-w4", "tok-T160-c40-w4", "req-T24-c4-len6-mixedwin",
                                  "tok-T8-c48-mixedlen-idle", "pre-T40-c6"])
def test_classic_conv_flag_on_is_flag_off(classic, monkeypatch, name):
    """Classic interpret, real typed store: flag 4 vs unset with the stock writer replaced by the stand-in."""
    op, kw, _ = CONV_CASES[name]
    # C 512: the stand-in's scatter needs the region (9 pool tokens) to tile the 18-token block in classic interpret.
    case = _conv_case(**{**kw, "channels": 512})
    monkeypatch.setattr(pool_mod, "_write_region_stock", _stand_in)
    rings = []
    real_ring = pool_mod.write_region_ring
    monkeypatch.setattr(pool_mod, "write_region_ring",
                        lambda *a, **k: rings.append(a[1].shape[0]) or real_ring(*a, **k))
    fn = _conv_fn(pool_mod, op)
    args = _conv_args(case, op)
    pool0, out0 = _fresh_jit(fn)(*args)
    assert rings == []
    monkeypatch.setenv(FLAG, "4")
    pool1, out1 = _fresh_jit(fn)(*args)
    assert len(rings) == 1, rings
    assert np.any(_bytes(pool0) != np.asarray(case["pool"]).view(np.uint8)), "vacuous case"
    np.testing.assert_array_equal(_bytes(pool1), _bytes(pool0), err_msg="pool: flag 4 vs off")
    np.testing.assert_array_equal(_bits16(out1), _bits16(out0), err_msg="output: flag 4 vs off")


def test_conv_cases_cover_both_paths():
    """The conv cases hold token-path and request-path buckets (x.shape[0] vs count * window)."""
    paths = {}
    for name, (op, kw, path) in CONV_CASES.items():
        if op == "spec":
            got = "token" if kw["tokens"] <= kw["count"] * WINDOW else "request"
            assert got == path, (name, got, path)
        paths.setdefault(path, []).append(name)
    assert len(paths["token"]) >= 8 and len(paths["request"]) >= 3 and len(paths["prefill"]) >= 2


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
        pool_mod.write_region(pool, values, ids, tok0=0, skip_idle_source=True)
    monkeypatch.setenv(FLAG, "0")
    pool_mod.write_region(pool, values, ids, tok0=0)
    stock = ("stock", {"tok0": 0, "skip_idle_source": False})
    stock_skip = ("stock", {"tok0": 0, "skip_idle_source": True})
    want = [stock]
    for d in (1, 2, 4, 8):
        want += [("ring", {"tok0": 0, "depth": d}), stock_skip]
    assert calls == want + [stock], calls
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


def _capture_pool_log():
    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    names = ("vllm.vllm_torchtpu.kernels.qwen4_exp.pool", "vllm_torchtpu.kernels.qwen4_exp.pool")
    loggers = [logging.getLogger(n) for n in names]
    h = _Keep(level=logging.INFO)
    olds = []
    for log in loggers:
        log.addHandler(h)
        olds.append(log.level)
        log.setLevel(logging.INFO)

    def done():
        for log, old in zip(loggers, olds):
            log.removeHandler(h)
            log.setLevel(old)
        return records

    return done


def test_ring_logs_its_engagement_once(classic, monkeypatch):
    monkeypatch.setattr(pool_mod, "_WRDEFER_SEEN", set())
    done = _capture_pool_log()
    try:
        pool, values, ids, tok0 = _writer_case(5, 13)
        for _ in range(2):  # same shape twice: one line
            pool_mod.write_region_ring(pool, values, ids, depth=2)
    finally:
        records = done()
    lines = [m for m in records if "q38 PLE-WRDEFER" in m]
    print("Q38 RPLEW trace engagement:", lines)
    assert lines == [
        "q38 PLE-WRDEFER traced: 13 rows x 9 pool tokens, 2 checkpoint DMAs in flight, one grid step"
    ], records


@pytest.mark.parametrize("value", [None, "0", "4"])
def test_engagement_line_only_with_the_flag(classic, monkeypatch, value):
    """Through the speculative conv: the trace line appears iff the flag is set."""
    monkeypatch.setattr(pool_mod, "_WRDEFER_SEEN", set())
    monkeypatch.setattr(pool_mod, "_write_region_stock", _stand_in)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    case = _conv_case(**CONV_CASES["tok-T16-c48-w4"][1])
    done = _capture_pool_log()
    try:
        jax.make_jaxpr(_conv_fn(pool_mod, "spec"))(*_conv_args(case, "spec"))
    finally:
        records = done()
    lines = [m for m in records if "q38 PLE-WRDEFER" in m]
    print("Q38 RPLEW conv engagement flag=%s: %s" % (value, lines))
    if value == "4":
        assert lines == [
            "q38 PLE-WRDEFER traced: 16 rows x 5 pool tokens, 4 checkpoint DMAs in flight, one grid step"
        ], records
    else:
        assert lines == [], records


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


def _name(eqn):
    return str(eqn.params.get("name_and_src_info", eqn.params.get("name", "")))


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_ring_census(depth):
    """One pallas_call named q38_ple_wrdefer, grid (1,), K staging slots, K pool-layout
    slots, K read and K write semaphores; the stock writer: grid (rows,)."""
    pool, values, ids, tok0 = _writer_case(1, 16)
    ring = _pallas_eqns(jax.make_jaxpr(functools.partial(pool_mod.write_region_ring, depth=depth))(pool, values, ids))
    assert len(ring) == 1
    assert "q38_ple_wrdefer" in _name(ring[0])
    shapes, grid = _scratch_shapes(ring[0])
    print("Q38 RPLEW census K=%d grid %s scratch %s" % (depth, grid, shapes))
    assert grid == (1,)
    assert shapes == [(depth, 18, 256), (depth, 9) + TAIL, (depth,), (depth,)], shapes
    stock = _pallas_eqns(jax.make_jaxpr(pool_mod._write_region_stock)(pool, values, ids))
    assert len(stock) == 1 and _scratch_shapes(stock[0])[1] == (16,)
    assert "q38_ple_wrdefer" not in _name(stock[0])


@pytest.mark.parametrize("value,skip", [(None, 0), ("4", 0), ("4", 1)])
def test_conv_census(monkeypatch, value, skip):
    """The speculative conv's jaxpr holds a q38_ple_wrdefer pallas_call iff the flag is set
    and SKIP_IDLE_ROWS is off; its pallas_call count is the flag-off count either way."""
    _clear_env(monkeypatch)
    monkeypatch.setenv(HRK, "1")
    if skip:
        monkeypatch.setenv(SKIP, "1")
    case = _conv_case(**CONV_CASES["tok-T160-c40-w4"][1])
    fn = _conv_fn(pool_mod, "spec")
    off = _pallas_eqns(jax.make_jaxpr(fn)(*_conv_args(case, "spec")))
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    got = _pallas_eqns(jax.make_jaxpr(lambda *a: fn(*a))(*_conv_args(case, "spec")))
    ring = [e for e in got if "q38_ple_wrdefer" in _name(e)]
    print("Q38 RPLEW conv census flag=%s skip=%d: %d pallas_calls, %d ring" % (value, skip, len(got), len(ring)))
    assert len(got) == len(off)
    assert len(ring) == (1 if (value == "4" and not skip) else 0)


@pytest.mark.parametrize("value,expected", [(None, 0), ("0", 0), ("1", 1), ("2", 2), ("4", 4), ("8", 8)])
def test_flag_reads(monkeypatch, value, expected):
    from vllm_torchtpu import envs

    envs.__dict__.pop(FLAG, None)
    monkeypatch.delenv(FLAG, raising=False)
    if value is not None:
        monkeypatch.setenv(FLAG, value)
    assert pool_mod.wrdefer_depth() == expected
    assert envs.environment_variables[FLAG]() == (value or "0")


def test_flag_is_a_compile_key(monkeypatch):
    """Unset and "0" give the same compile-env factors; "4" changes them, in this key only."""
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    envs.__dict__.pop(FLAG, None)
    assert FLAG in envs.environment_variables
    assert FLAG not in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv(FLAG, raising=False)
    unset = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "0")
    zero = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv(FLAG, "4")
    four = tpu_compiler._tpu_compile_env_factors()
    assert unset == zero
    assert four != unset and {k for k in four if four[k] != unset.get(k)} == {FLAG}


# ---------------------------------------------------------------------------- DMA ledger
class _Ledger:
    """Host run of the ring kernel's body: records every DMA start / wait and every store."""

    def __init__(self, ids, depth):
        self.ids = np.asarray(ids)
        self.depth = depth
        self.events = []
        self.errors = []
        self.inflight_w = {}  # slot -> row whose write is in flight from encoded[slot]
        self.inflight_r = {}  # slot -> row whose read into staged[slot] is in flight
        self.staged = {}  # slot -> row whose values sit in staged[slot] (read waited)
        self.encoded = {}  # slot -> row stored into encoded[slot]
        self.landed = {}  # pool id -> row whose values landed there

    def err(self, msg):
        self.errors.append(msg)


class _Ref:
    def __init__(self, kind, ledger, index=None):
        self.kind, self.ledger, self.index = kind, ledger, index

    @property
    def at(self):
        return _At(self)

    def __getitem__(self, idx):
        return _Ref(self.kind, self.ledger, idx)


class _At:
    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, idx):
        return _Ref(self.ref.kind, self.ref.ledger, idx)


def _key(idx):
    return int(idx[0]) if isinstance(idx, tuple) else int(idx)


class _Copy:
    def __init__(self, src, dst, sem):
        self.src, self.dst, self.sem = src, dst, sem

    def _what(self):
        if self.src.kind == "source" and self.dst.kind == "staged":
            return "read", _key(self.src.index), _key(self.dst.index), _key(self.sem.index)
        if self.src.kind == "encoded" and self.dst.kind == "target":
            return "write", _key(self.src.index), _key(self.dst.index), _key(self.sem.index)
        raise AssertionError((self.src.kind, self.dst.kind))

    def start(self):
        L = self.src.ledger
        kind, a, b, sem = self._what()
        L.events.append(("start", kind, a, b, sem))
        if kind == "read":
            row, slot = a, b
            if sem != slot:
                L.err(f"read of row {row} into slot {slot} signals semaphore {sem}")
            if slot in L.inflight_r:
                L.err(f"read of row {row} into slot {slot} while row {L.inflight_r[slot]}'s read is in flight")
            if slot in L.staged:
                L.err(f"read of row {row} into slot {slot} before row {L.staged[slot]}'s values were stored")
            L.inflight_r[slot] = row
        else:
            slot, pid = a, b
            if sem != slot:
                L.err(f"write from slot {slot} signals semaphore {sem}")
            if slot in L.inflight_w:
                L.err(f"write from slot {slot} started while row {L.inflight_w[slot]}'s write is in flight")
            row = L.encoded.get(slot)
            if row is None:
                L.err(f"write from slot {slot} with nothing stored")
                return
            if pid != L.ids[row]:
                L.err(f"row {row} written to id {pid}, want {L.ids[row]}")
            L.inflight_w[slot] = row

    def wait(self):
        L = self.src.ledger
        kind, a, b, sem = self._what()
        L.events.append(("wait", kind, a, b, sem))
        if kind == "read":
            row, slot = a, b
            if L.inflight_r.get(sem) != row:
                L.err(f"wait for row {row}'s read on semaphore {sem}, in flight there: {L.inflight_r.get(sem)}")
                return
            del L.inflight_r[sem]
            L.staged[slot] = row
        else:
            slot = a
            if sem not in L.inflight_w:
                L.err(f"wait on write semaphore {sem} with no write in flight (would hang)")
                return
            row = L.inflight_w.pop(sem)
            pid = int(L.ids[row])
            if pid in L.landed:
                L.err(f"id {pid} written twice (rows {L.landed[pid]} and {row})")
            L.landed[pid] = row


def _ledger_run(count, depth, pattern, seed):
    pool, values, ids, tok0 = _writer_case(seed, count, pattern=pattern)
    captured = {}

    def fake_call(kernel, **kw):
        captured["kernel"] = kernel
        return lambda *a: a[-1]

    L = _Ledger(ids, depth)

    def store(dst, val):
        slot = _key(dst.index)
        row = L.staged.get(_key(val.index)) if isinstance(val, _Ref) else None
        if slot in L.inflight_w:
            L.err(f"store into slot {slot} while row {L.inflight_w[slot]}'s write is in flight")
        if row is None:
            L.err(f"store from staged slot {_key(val.index)} with no waited read")
        else:
            del L.staged[_key(val.index)]
        L.encoded[slot] = row
        L.events.append(("store", slot, row))

    def when(cond):
        return (lambda f: f()) if bool(cond) else (lambda f: None)

    def fori(lo, hi, body, carry):
        for c in range(lo, hi):
            carry = body(c, carry)
        return carry

    import unittest.mock as um

    with um.patch.object(pl, "pallas_call", fake_call):
        pool_mod.write_region_ring(pool, values, ids, tok0=tok0, depth=depth)
    kernel = captured["kernel"]
    eff = max(1, min(depth, count))
    L.depth = eff
    with um.patch.object(pl, "when", when), um.patch.object(jax.lax, "fori_loop", fori), \
            um.patch.object(pltpu, "make_async_copy", lambda s, d, m: _Copy(s, d, m)), \
            um.patch.object(typed_ldst, "store_typed", store), \
            um.patch.object(pl, "ds", lambda a, b: (a, b)):
        kernel(np.asarray(ids), _Ref("source", L), None, _Ref("target", L), _Ref("staged", L),
               _Ref("encoded", L), _Ref("rsem", L), _Ref("wsem", L))
    live = {int(i): r for r, i in enumerate(np.asarray(ids)) if i > 0}
    if L.inflight_w:
        L.err(f"writes still in flight at the end: slots {sorted(L.inflight_w)}")
    if L.inflight_r:
        L.err(f"reads still in flight at the end: slots {sorted(L.inflight_r)}")
    if L.landed != live:
        missing = sorted(set(live) - set(L.landed))[:5]
        extra = sorted(set(L.landed) - set(live))[:5]
        L.err(f"landed ids differ from the live ids: missing {missing} extra {extra}")
    starts = sum(1 for e in L.events if e[0] == "start")
    waits = sum(1 for e in L.events if e[0] == "wait")
    return L, starts, waits, len(live)


@pytest.mark.parametrize("pattern", ["mixed", "tailpad", "headnull", "allnull", "single", "dense"])
@pytest.mark.parametrize("count", [1, 5, 13, 64, 95, 160])
@pytest.mark.parametrize("depth", [1, 2, 4, 8])
def test_ring_dma_ledger(pattern, count, depth):
    """Every started DMA is waited exactly once (2 per live row), no slot is reused before its
    previous write was waited, every live id lands once from its own row, nothing is left in flight."""
    L, starts, waits, live = _ledger_run(count, depth, pattern, count * 31 + depth + len(pattern))
    assert not L.errors, L.errors[:5]
    assert starts == waits == 2 * live, (starts, waits, live)
