# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_PLE_D14_FOLD (q38 L1-PLE, PLE-INTO-D14): the D14 PLE conv as one Pallas
program is bitwise today's speculative conv.

Every Pallas kernel runs in Pallas interpret mode on CPU (``pl.pallas_call`` is
wrapped to force ``interpret=True``, the mode pool_adapters documents for its
typed views: it keeps Mosaic's ``ref.bitcast`` semantics). The region read is
the real one (pool_adapters.gather_region); the fold program is the real one,
DMAs included, on a pool in the served PLE geometry
(f8e4m3fn[pages, block, 1, 4, 256]). ``pool.write_region`` cannot run on CPU in
jax 0.11.2 (interpret mode: ``pl.program_id`` inside ``pl.run_scoped`` loses its
grid context; the TPU interpreter, ``InterpretParams``, cannot run a bitcast ref),
so the default / bounded arms write through ``pool_adapters.scatter_region`` with
write_region's contract (only ids > 0 are written; the same ``typed_ldst`` store
view, so the same bytes). Interpret mode runs a DMA at its start, so DMA start /
wait ordering is not checked here (see the fold's module docstring). The pool starts as random bytes; each page's history region
is then written by ``write_region`` from finite BF16 values, so the bytes outside
the regions are random and must survive both arms. Each case compares the conv
output (all T rows) and the whole pool byte for byte between
``pooled_short_conv_speculative(bounded=b, fold=True)`` and ``bounded=0`` (the
default op), ``bounded=1`` and ``bounded=2``.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.qwen4_exp import ple_d14_fold
from vllm_torchtpu.kernels.qwen4_exp import pool as pool_mod
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows

STATE_LEN, DILATION, TAPS, WINDOW = 9, 3, 4, 4
TAIL = (1, 4, 256)  # the served pool's per-token payload (f8e4m3fn[98304, 512, 1, 4, 256])
TOKEN_ELEMS = 512  # BF16 values per pool token


def _write_region_cpu(pool, values, ids, *, tok0=0):
    """pool.write_region's contract (rows with id > 0 written, the rest left as they
    are) through pool_adapters.scatter_region: rows with id <= 0 write the null
    block's own bytes back to it (read before any write; no live row targets it)."""
    count, elements = values.shape
    ntok, rows = region_rows(pool, elements, values.dtype)
    lanes = pool.shape[-1]
    vals = jnp.pad(values, ((0, 0), (0, rows * lanes - elements))).reshape(count, rows, lanes)
    safe = jnp.maximum(ids, 0)
    old = pool_adapters.gather_region(pool, safe, tok0=tok0, ntok=ntok, out_dtype=values.dtype, interpret=True)
    vals = jnp.where((ids > 0)[:, None, None], vals, old)
    return pool_adapters.scatter_region(pool, vals, safe, tok0=tok0, ntok=ntok, interpret=True)


@pytest.fixture(autouse=True)
def _interpret(monkeypatch):
    real = pl.pallas_call

    def call(*args, **kwargs):
        kwargs["interpret"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", call)
    monkeypatch.setattr(pool_mod, "write_region", _write_region_cpu)
    calls = []
    real_fold = ple_d14_fold.uniform_window_conv_fold

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return real_fold(*args, **kwargs)

    monkeypatch.setattr(ple_d14_fold, "uniform_window_conv_fold", spy)
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT", raising=False)
    return calls


def _case(
    seed, *, count, tokens, live, channels=512, offsets=None, no_history=(), no_slot=(),
    poison=False, heavy=False, unwindowed=(),
):
    """A decode bucket: ``live`` requests of WINDOW tokens first, then padding.

    ``count`` is max-num-seqs, ``tokens`` the padded bucket. ``offsets[r]`` picks
    request r's read checkpoint; ``no_history`` live rows have no initial state;
    ``no_slot`` live rows have state id 0 (inactive); ``poison`` fills padded token
    rows with NaN and the null page 0 with 0xFF bytes (NaN in every view);
    ``heavy`` draws half of x and the history as +-2^30 and the weights as +-1, so
    the f32 tap sum absorbs small terms and its order shows; ``unwindowed`` live
    rows are not verify windows (they write only their final window, to column 0).
    """
    rng = np.random.default_rng(seed)
    assert live * WINDOW <= tokens and live <= count and channels % TOKEN_ELEMS == 0
    tpp = channels // TOKEN_ELEMS
    block = 2 * STATE_LEN * tpp  # a multiple of the region (interpret mode pads a partial block)
    pages = 1 + max(live, 1) * WINDOW + 3
    ckpt = np.zeros((count, WINDOW), np.int32)
    ckpt[:live] = rng.permutation(np.arange(1, pages))[: live * WINDOW].reshape(live, WINDOW)
    if offsets is None:
        offsets = rng.integers(0, WINDOW, size=count)
    offsets = np.asarray(offsets, np.int32)
    rows = np.arange(count)
    read_ids = np.where(rows < live, ckpt[rows, offsets], 0).astype(np.int32)
    ids = ckpt[:, 0].copy()
    ids[list(no_slot)] = 0
    starts = np.minimum(np.arange(count + 1) * WINDOW, live * WINDOW).astype(np.int32)
    initial = rows < live
    initial[list(no_history)] = False
    windowed = rows < live
    windowed[list(unwindowed)] = False
    x = rng.normal(size=(tokens, channels)).astype(np.float32)
    history = rng.normal(size=(pages, STATE_LEN * channels)).astype(np.float32)
    weight = rng.normal(size=(channels, TAPS)).astype(np.float32)
    if heavy:
        for a in (x, history):
            big = rng.random(a.shape) < 0.5
            a[big] = np.sign(rng.normal(size=int(big.sum()))) * 2.0**30
        weight = np.sign(rng.normal(size=weight.shape)).astype(np.float32)
    if poison:
        x[live * WINDOW :] = np.nan
    raw = rng.integers(0, 256, size=(pages, block) + TAIL, dtype=np.uint8)
    if poison:
        raw[0] = 0xFF
    pool = jnp.asarray(raw.view(jnp.float8_e4m3fn))
    pool = jax.jit(pool_mod.write_region)(
        pool, jnp.asarray(history, jnp.bfloat16), jnp.arange(pages, dtype=jnp.int32)
    )
    return dict(
        x=jnp.asarray(x, jnp.bfloat16),
        weight=jnp.asarray(weight, jnp.bfloat16),
        pool=pool,
        starts=jnp.asarray(starts),
        ids=jnp.asarray(ids),
        initial=jnp.asarray(initial),
        read_ids=jnp.asarray(read_ids),
        ckpt_ids=jnp.asarray(ckpt),
        windowed=jnp.asarray(windowed),
    )


ORDER = ("x", "weight", "pool", "starts", "ids", "initial", "read_ids", "ckpt_ids", "windowed")


def _run(case, bounded, fold=False):
    kw = dict(dilation=DILATION, state_len=STATE_LEN, bounded=bounded)
    if fold:
        kw["fold"] = True
    fn = functools.partial(pool_mod.pooled_short_conv_speculative, **kw)
    pool, output = jax.jit(fn)(*(case[k] for k in ORDER))
    return np.asarray(pool).view(np.uint8), np.asarray(output.view(jnp.uint16))


# (count = max-num-seqs, tokens = bucket, live requests, extra case options)
CASES = [
    (48, 8, 2, {}),
    (48, 8, 1, {"offsets": [3] * 48}),
    (48, 12, 3, {}),
    (48, 16, 4, {}),
    (48, 16, 1, {}),
    (48, 16, 3, {}),
    (48, 32, 8, {}),
    (48, 32, 5, {"offsets": [0] * 48}),
    (48, 48, 12, {}),
    (48, 64, 16, {}),
    (48, 64, 13, {"no_history": (2, 7)}),
    (48, 64, 16, {"poison": True}),
    (48, 32, 6, {"poison": True, "no_slot": (1,)}),
    (48, 16, 4, {"no_slot": (0, 3), "no_history": (1,)}),
    (48, 16, 4, {"unwindowed": (1, 2)}),
    (16, 64, 16, {}),
    (64, 16, 4, {}),
    (48, 16, 4, {"heavy": True}),
    (48, 32, 7, {"heavy": True}),
    (48, 64, 13, {"heavy": True, "no_history": (5,)}),
    (48, 16, 4, {"channels": 1024}),
    (48, 64, 16, {"channels": 1024, "heavy": True}),
    # The served width (hc 4 x hidden 2560): 20 pool tokens per position.
    (48, 16, 4, {"channels": 10240}),
    (48, 64, 16, {"channels": 10240, "heavy": True, "no_history": (3,)}),
]


def _id(case):
    count, tokens, live, extra = case
    tags = "-".join(f"{k}{v if k == 'channels' else ''}" for k, v in extra.items())
    return f"c{count}-t{tokens}-n{live}" + (f"-{tags}" if tags else "")


@pytest.mark.parametrize("case", CASES, ids=[_id(c) for c in CASES])
@pytest.mark.parametrize("bounded", [1, 2])
def test_fold_is_bitwise_today(_interpret, case, bounded):
    count, tokens, live, extra = case
    inputs = _case(count * 1000 + tokens * 10 + live, count=count, tokens=tokens, live=live, **extra)
    before = np.asarray(inputs["pool"]).view(np.uint8)
    pool0, out0 = _run(inputs, 0)
    poolb, outb = _run(inputs, bounded)
    assert not _interpret, "the bounded path must not take the fold"
    poolf, outf = _run(inputs, bounded, fold=True)
    assert _interpret == [tokens], f"fold not engaged: {_interpret}"

    # Not vacuous: the default op wrote checkpoints and produced live outputs.
    assert np.any(pool0 != before), "the default op wrote no checkpoint"
    live_rows = live * WINDOW
    assert np.any(out0[:live_rows] != 0), "the default op produced no live output"
    assert not np.any(out0[live_rows:]), "padded tokens must output zero"
    # Controls: today's bounded path is the default op (D07-b2).
    np.testing.assert_array_equal(outb, out0, err_msg=f"bounded={bounded} output vs default")
    np.testing.assert_array_equal(poolb, pool0, err_msg=f"bounded={bounded} pool vs default")
    # The fold: bitwise.
    np.testing.assert_array_equal(outf, out0, err_msg="fold output vs default")
    np.testing.assert_array_equal(poolf, pool0, err_msg="fold pool vs default")


@pytest.mark.parametrize(
    "count,tokens,live,channels,why",
    [
        (48, 18, 4, 512, "tokens not rows x window"),
        (4, 24, 4, 512, "above count x window: request path"),
    ],
    ids=["t18-not-rows-x-window", "t24-request-path"],
)
def test_fold_keeps_the_bounded_path_outside_its_preconditions(_interpret, count, tokens, live, channels, why):
    inputs = _case(count * 1000 + tokens, count=count, tokens=tokens, live=live, channels=channels)
    pool0, out0 = _run(inputs, 0)
    poolf, outf = _run(inputs, 1, fold=True)
    assert _interpret == [], f"{why}: the fold must not run"
    np.testing.assert_array_equal(outf, out0)
    np.testing.assert_array_equal(poolf, pool0)


def test_fold_geometry_rules(monkeypatch):
    served = jax.ShapeDtypeStruct((98304, 512) + TAIL, jnp.float8_e4m3fn)
    assert ple_d14_fold.fold_geometry(served, 10240, STATE_LEN, WINDOW) == (20, 40)
    assert ple_d14_fold.fold_geometry(served, 512, STATE_LEN, WINDOW) == (1, 2)
    assert ple_d14_fold.fold_geometry(served, 640, STATE_LEN, WINDOW) is None  # not whole tokens
    small = jax.ShapeDtypeStruct((8, 100) + TAIL, jnp.float8_e4m3fn)
    assert ple_d14_fold.fold_geometry(small, 10240, STATE_LEN, WINDOW) is None  # region > block
    flat = jax.ShapeDtypeStruct((8, 9 * 512), jnp.bfloat16)
    assert ple_d14_fold.fold_geometry(flat, 512, STATE_LEN, WINDOW) is None  # 2-D stand-in pool
    monkeypatch.setenv("VLLM_KV_CACHE_LAYOUT", "HND")
    assert ple_d14_fold.fold_geometry(served, 10240, STATE_LEN, WINDOW) is None
    monkeypatch.delenv("VLLM_KV_CACHE_LAYOUT")
    # VMEM: the D14 buckets fit the budget, with room.
    for t in (8, 16, 32, 48, 64):
        need = ple_d14_fold.vmem_bytes(served, t // WINDOW, WINDOW, TAPS, STATE_LEN, 10240)
        assert need <= ple_d14_fold.VMEM_BUDGET, (t, need)
    assert ple_d14_fold.vmem_bytes(served, 16, WINDOW, TAPS, STATE_LEN, 10240) < 11 * 2**20


def test_fold_traces_one_program_without_channel_gathers(_interpret):
    """The fold path's jaxpr: one pallas_call (the fold) and no gather of a
    channel-wide operand; the bounded path: the region read and the checkpoint
    writer (here the CPU stand-in: a region read + scatter_region) plus the
    channel gathers (taps, windows)."""
    inputs = _case(5, count=48, tokens=32, live=8)
    channels = inputs["x"].shape[-1]

    def census(fold):
        kw = dict(dilation=DILATION, state_len=STATE_LEN, bounded=1)
        if fold:
            kw["fold"] = True
        fn = functools.partial(pool_mod.pooled_short_conv_speculative, **kw)
        jaxpr = jax.make_jaxpr(fn)(*(inputs[k] for k in ORDER))
        prims = []

        def walk(j):
            for e in j.eqns:
                name = e.primitive.name
                if name == "gather" and channels not in e.invars[0].aval.shape:
                    name = "gather_small"  # metadata gathers (ckpt_ids[req, column], ...)
                prims.append(name)
                for p in e.params.values():
                    for sub in (p if isinstance(p, (list, tuple)) else [p]):
                        if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                            walk(sub.jaxpr)
                        elif hasattr(sub, "eqns"):
                            walk(sub)

        walk(jaxpr.jaxpr)
        return prims.count("pallas_call"), prims.count("gather")

    assert census(False)[0] >= 2 and census(False)[1] > 0, census(False)
    assert census(True) == (1, 0), census(True)


def test_fold_logs_its_engagement_once(_interpret):
    import logging

    records = []

    class _Keep(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("vllm.vllm_torchtpu.kernels.qwen4_exp.ple_d14_fold")
    handler = _Keep(level=logging.INFO)
    log.addHandler(handler)
    old = log.level
    log.setLevel(logging.INFO)
    try:
        ple_d14_fold._SEEN.clear()
        for tokens in (16, 32):
            _run(_case(tokens, count=48, tokens=tokens, live=tokens // WINDOW), 1, fold=True)
    finally:
        log.removeHandler(handler)
        log.setLevel(old)
    lines = [m for m in records if "q38 PLE-D14-FOLD traced" in m]
    print("Q38 L1-PLE engagement:", lines)
    assert len(lines) == 1, records
    assert lines[0].startswith("q38 PLE-D14-FOLD traced: 16 tokens, 4 request rows x window 4"), lines
