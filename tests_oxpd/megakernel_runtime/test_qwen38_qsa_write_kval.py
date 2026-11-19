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
"""q38 L2 WFOLD (QSA-WRITE-FOLD stage A): the all-layer writer takes the whole-step kernel's own QSA cache values
(VLLM_TPU_MK_QSA_WRITE_ALL + VLLM_TPU_MK_QSA_WRITE_KVAL) instead of recomputing them (CPU, Pallas interpret).

(a) The kernel's emitted values equal what `_batched_writes` recomputes from the kernel's own projections, bit for
    bit: the D14 whole-step kernel (TINY geometry, R = 4 = one main-KV DCP-4 group sharing a KV head, union attention,
    sharded indexer, HBM tables; and the same with every D08 S1 stage on) at 8 / 16 / 32 / 48 / 64 rows, the last
    request idle; one RoPE table feeds every RoPE row (as `qsa_step_inputs`). Per rank, QSA layer and live row:
    `qsa_kv_new[..., :D]` == `produce_kv(qsa_kr_new, ...)`'s key; per live request: `qsa_ck_new` == the
    `_indexer_updates` key at the request's group-end token, `qsa_ring_new` == its new ring (both settings of
    VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT).
(b) Pool bytes after one step: `batched_writes_all(kernel_values=...)` (KVAL) == `batched_writes_all` (KVW-ALL) ==
    `_batched_writes` (stock), bit for bit:
    - real kernel outputs (the runs of (a)) on 2-layer writer layouts: whole rows, one main pair, a lone paired layer
      (read-merge-write); every DCP-4 rank; through `wholestep_qsa_writes(kernel_values=...)` and directly;
    - 11 layers (KVW-ALL's served / paired / bench layouts, 8-64 rows, DCP-4 ranks, idle last request, 64 / 512-row
      pages, the ring / tag read handed in, two consecutive steps) with the values the kernel emits taken from the
      stock formulas (what (a) proves the kernel equals); idle rows / requests' values NaN-poisoned (never used).
(c) The writer uses the handed-in values (a perturbed key / completed key / ring changes the pool), the KVAL trace
    has no value computation (no `produce_kv` / `_indexer_updates` call, no rsqrt in the lowered text), and the
    whole-step call plumbs the kernel's `qsa_ck_new` / `qsa_ring_new` (stubbed kernel) only with both flags on.
Provenance: with Q38_EXPECT_ROOT set, the modules must import from under it.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel_runtime/test_qwen38_qsa_write_kval.py
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import logging
import os
import sys
import types

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

try:
    import vllm.logger  # noqa: F401
except ImportError:  # CPU venv without vllm: the kernels only need a logger and the KV layout.
    _vllm = types.ModuleType("vllm")
    _envs = types.ModuleType("vllm.envs")
    _envs.VLLM_KV_CACHE_LAYOUT = "NHD"
    _logger = types.ModuleType("vllm.logger")

    def _init_logger(name):
        log = logging.getLogger(name)
        log.warning_once = log.info_once = log.debug_once = log.warning
        return log

    _logger.init_logger = _init_logger
    _vllm.envs, _vllm.logger = _envs, _logger
    sys.modules.update({"vllm": _vllm, "vllm.envs": _envs, "vllm.logger": _logger})

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.qsa import _indexer_updates, produce_kv  # noqa: E402
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import read_ring_and_tags, write_ring_and_tags  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_qsa_write_all as wa  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_cache as wc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache  # noqa: E402

L, WIDTH, HEAD_DIM, INDEX_DIM, RATIO, ROTARY, EPS, T = 11, 256, 256, 128, 4, 64, 1e-6, 4
RING_ROWS = 8
K_SCALE, V_SCALE = 0.375, 2.5
FP8 = jnp.float8_e4m3fn
ROWS = (8, 16, 32, 48, 64)
SHIFT = "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT"
SERVED_QSA_ENV = {SHIFT: True, "VLLM_TPU_QSA_RING_DMA_BATCH": 32}


# ------------------------------------------------------------------------------------------------ helpers
@contextlib.contextmanager
def interpret():
    """The writers' Pallas calls in interpret mode with TPU7x chip info (KVW-ALL's fixture as a context, so a test
    can run the whole-step kernel, which uses the TPU interpreter itself, before it)."""
    with pytest.MonkeyPatch.context() as mp:
        original = pl.pallas_call
        mp.setattr(pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True}))
        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        mp.setattr(pltpu, "get_tpu_info", lambda: info)
        mp.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", False)
        mp.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", False)
        yield mp


def u8(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def bits(x):
    """Bit pattern of a BF16 array (uint16 view)."""
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x, jnp.bfloat16), jnp.uint16))


def finite_fp8(rng, shape):
    """Random FP8 E4M3 bytes without the two NaN codes (0x7F / 0xFF)."""
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    b = np.where((b & 0x7F) == 0x7F, b ^ 0x01, b).astype(np.uint8)
    return jax.lax.bitcast_convert_type(jnp.asarray(b), FP8)


def trig_table(n, theta=10_000_000.0):
    """A RoPE cache `[n, 128]` BF16: cos | sin of the 32 frequencies, zero lanes after (the served layout)."""
    inv = 1.0 / (theta ** (np.arange(0, ROTARY, 2) / ROTARY))
    ang = np.arange(n)[:, None] * inv[None]
    t = np.zeros((n, 128), np.float32)
    t[:, : ROTARY // 2], t[:, ROTARY // 2 : ROTARY] = np.cos(ang), np.sin(ang)
    return jnp.asarray(t, jnp.bfloat16)


def coherent_tags(ctx):
    """Ring tags as earlier steps leave them: lane l holds the newest position = l (mod 8) below the context."""
    base = np.asarray(ctx)[:, None] - RING_ROWS
    p = base + (np.arange(RING_ROWS)[None] - base) % RING_ROWS
    return np.stack([p, np.zeros_like(p)], -1).astype(np.int32)


def layout_spec(name, n):
    comp, comp_group = [o % 8 for o in range(n)], [o // 8 for o in range(n)]
    if name == "served":
        return dict(main=[-1] * n, main_group=list(range(n)), comp=comp, comp_group=comp_group)
    if name == "paired":
        return dict(main=[o % 2 for o in range(n)], main_group=[o // 2 for o in range(n)], comp=comp,
                    comp_group=comp_group)
    if name == "bench":
        return dict(main=[(o + 1) % 2 for o in range(n)], main_group=[o // 2 for o in range(n)],
                    comp=[o % 2 for o in range(n)], comp_group=[o // 2 for o in range(n)])
    if name == "lone":  # layer 0 alone in a paired slot (read-merge-write), the rest whole rows
        return dict(main=[0] + [-1] * (n - 1), main_group=list(range(n)), comp=comp, comp_group=comp_group)
    raise ValueError(name)


def make_case(seed, rows, layout, *, n_layers=L, page_rows=64, dcp=4, comp_dcp=False, idle=(), contexts=None,
              prev=None, stale_idle_rings=False, rows_in=None, norms=None, trig=None):
    """One uniform 4-row-window step of `n_layers` QSA layers (KVW-ALL's `make_case`, credit, without ragged
    windows: the kernel's precondition). `rows_in` / `norms` / `trig`: the kernel's projections, norm weights and
    RoPE table (default random). Pages for every request's next two windows (`prev`: same tables, new rows)."""
    rng = np.random.default_rng(seed)
    spec = layout_spec(layout, n_layers)
    nreq = rows // T
    lens = prev["lens"] if prev is not None else np.array([0 if r in idle else T for r in range(nreq)], np.int32)
    if contexts is None:
        phases = [0, 1, 2, 3, 5, 6, 7, page_rows - 3, page_rows - 2, page_rows - 1, 37 % page_rows, 4]
        contexts = []
        for r in range(nreq):
            window = (r % max(dcp, 1)) + max(dcp, 1) * int(rng.integers(0, 3))
            contexts.append(window * page_rows + phases[(r + seed) % len(phases)])
        if nreq > 2:
            contexts[2] = 0  # a request's first step (empty ring)
    ctx = np.asarray(contexts, np.int32)
    starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    positions = np.zeros(rows, np.int32)
    for r in range(nreq):
        positions[starts[r]:starts[r + 1]] = ctx[r] + np.arange(lens[r])

    def normal(*shape):
        return jnp.asarray(rng.normal(0, 1, shape), jnp.bfloat16)

    if rows_in is None:
        rows_in = [(normal(rows, HEAD_DIM), normal(rows, HEAD_DIM), normal(rows, INDEX_DIM)) for _ in range(n_layers)]
    layers = [QSALayerCache(compressed_slot=c, main_slot=m, k_scale=K_SCALE, v_scale=V_SCALE)
              for m, c in zip(spec["main"], spec["comp"])]
    case = dict(rows=rows_in, layers=layers, positions=jnp.asarray(positions), starts=jnp.asarray(starts),
                context=jnp.asarray(ctx), lens=lens, ctx=ctx, page_rows=page_rows, dcp=dcp, layout=layout,
                n_layers=n_layers)
    if prev is not None:
        return {**case, **{k: prev[k] for k in ("tables", "norms", "trig", "pages")}}
    span = max(dcp, 1) * page_rows
    cspan = (max(dcp, 1) if comp_dcp else 1) * page_rows
    last = ctx + 2 * T - 1
    cols = int(last.max() // page_rows) + 2
    n_main, n_comp = max(spec["main_group"]) + 1, max(spec["comp_group"]) + 1
    main = np.zeros((n_main, nreq, cols), np.int32)
    comp = np.zeros((n_comp, nreq, cols), np.int32)
    ring = np.zeros((n_layers, nreq), np.int32)
    slots = []
    for r in range(nreq):
        active = lens[r] > 0
        if active:
            for g in range(n_main):
                slots += [(main, (g, r, c)) for c in range(ctx[r] // span, last[r] // span + 1)]
            for g in range(n_comp):
                slots += [(comp, (g, r, c)) for c in range(ctx[r] // cspan, last[r] // cspan + 1)]
        if active or stale_idle_rings:
            slots += [(ring, (o, r)) for o in range(n_layers)]
    for (arr, idx), page in zip(slots, 1 + rng.permutation(len(slots))):
        arr[idx] = page
    pages = len(slots) + 1
    tables = [(jnp.asarray(ring[o]), jnp.asarray(main[spec["main_group"][o]]),
               jnp.asarray(comp[spec["comp_group"][o]])) for o in range(n_layers)]
    if norms is None:
        norms = [(normal(HEAD_DIM), normal(INDEX_DIM)) for _ in range(n_layers)]
    if trig is None:
        trig = jnp.pad(normal(8192, ROTARY), ((0, 0), (0, 128 - ROTARY)))
    return {**case, "tables": tables, "norms": norms, "trig": trig, "pages": pages,
            "pool": finite_fp8(rng, (pages, page_rows, 1, 4, WIDTH))}


def seed_rings(case, ring_in):
    """Every live request's ring pages (all layers) hold `ring_in` [requests, 8, 128] and coherent tags, so the
    writers' ring read is the kernel's ring input (the kernel zeroes context-0 rings itself, as the stock writer)."""
    live = np.nonzero(case["lens"] > 0)[0]
    ids = np.concatenate([np.asarray(t[0])[live] for t in case["tables"]]).astype(np.int32)
    ring = np.tile(np.asarray(ring_in, np.float32)[live].reshape(len(live), -1), (case["n_layers"], 1))
    tags = np.tile(coherent_tags(case["ctx"][live]).reshape(len(live), -1), (case["n_layers"], 1))
    pool = jax.jit(write_ring_and_tags)(case["pool"], jnp.asarray(ring, jnp.bfloat16), jnp.asarray(tags),
                                        jnp.asarray(ids))
    return {**case, "pool": pool}


def run(fn, case, pool, rank, *, ring_tags=None, kernel_values=None, comp_dcp=False):
    """`fn` (a writer) jitted as the whole step calls it: DCP size static, the rank an int32 [1]."""
    d = case["dcp"]

    def f(pool, rows, norms, tables, positions, trig, starts, context, rank_arr, ring_tags, kernel_values):
        kw = {}
        if d > 1:
            pair = (rank_arr, d)
            kw = dict(dcp=pair, compressed_dcp=pair if comp_dcp else None)
        if ring_tags is not None:
            kw["ring_tags"] = ring_tags
        if kernel_values is not None:
            kw["kernel_values"] = kernel_values
        return fn(pool, rows, norms, tables, case["layers"], positions, trig, starts, context,
                  compress_ratio=RATIO, norm_eps=EPS, rotary_dim=ROTARY, **kw)

    return jax.jit(f)(pool, case["rows"], case["norms"], case["tables"], case["positions"], case["trig"],
                      case["starts"], case["context"], jnp.asarray([rank], jnp.int32), ring_tags, kernel_values)


def run_dispatch(case, pool, rank, kernel_values, *, extra_requests=0):
    """`wholestep_qsa_writes(kernel_values=...)` as `per_rank_writes` calls it: flat runner tables, the kernel's
    request count (`extra_requests` idle kernel requests past the runner's), max_query_len 4."""
    nreq = len(case["ctx"])

    def f(pool, rows, norms, tables, positions, trig, starts, context, rank_arr, kv):
        flat = [(r.reshape(-1), m.reshape(-1), c.reshape(-1)) for r, m, c in tables]
        d = (rank_arr, case["dcp"]) if case["dcp"] > 1 else None
        return wc.wholestep_qsa_writes(pool, rows, norms, flat, case["layers"], positions, trig, starts, context,
                                       compress_ratio=RATIO, norm_eps=EPS, rotary_dim=ROTARY, max_query_len=T,
                                       num_requests=nreq + extra_requests, dcp=d, kernel_values=kv)

    return jax.jit(f)(pool, case["rows"], case["norms"], case["tables"], case["positions"], case["trig"],
                      case["starts"], case["context"], jnp.asarray([rank], jnp.int32), kernel_values)


@jax.jit
def _stock_parts(pool, rows, norms, tables, positions, trig, starts, context):
    n, count = len(context), len(positions)
    keys = jnp.stack([produce_kv(k, v, nk, positions, trig, EPS, ROTARY)[0].reshape(count, -1)
                      for (k, v, _), (nk, _) in zip(rows, norms)])
    active = starts[1:] > starts[:-1]
    slots = jnp.where(active, jnp.arange(n, dtype=jnp.int32), -1)
    ids = jnp.concatenate([t[0] for t in tables])
    ring, tags = read_ring_and_tags(pool, ids, RING_ROWS * INDEX_DIM, RING_ROWS * 2)
    ring = ring.reshape(len(rows), n, RING_ROWS, -1)
    tags = tags.reshape(len(rows), n, RING_ROWS, 2)
    completed, complete, req, _, ring_new, _ = jax.vmap(
        lambda raw, ring, tags, nik: _indexer_updates(raw, positions, ring, tags, nik, trig, starts, context, slots,
                                                      RATIO, EPS, ROTARY)
    )(jnp.stack([r[2] for r in rows]), ring, tags, jnp.stack([x[1] for x in norms]))
    return keys, completed, complete[0], req[0], ring_new


def stock_kernel_values(case, pool):
    """The values the kernel emits, from the stock formulas: key rows (`produce_kv`), each live request's
    group-end completed key and new ring (`_indexer_updates`). Invalid rows' keys and idle requests' keys / rings
    are NaN (the writer must never use them)."""
    keys, completed, complete, req, ring_new = (np.asarray(x) for x in _stock_parts(
        pool, case["rows"], case["norms"], case["tables"], case["positions"], case["trig"], case["starts"],
        case["context"]))
    n, starts = len(case["ctx"]), np.asarray(case["starts"])
    keys = keys.astype(np.float32)
    keys[:, starts[-1]:] = np.nan
    ck = np.full((keys.shape[0], n, INDEX_DIM), np.nan, np.float32)
    ends = np.nonzero(complete)[0]
    assert sorted(req[ends].tolist()) == sorted(np.nonzero(case["lens"] > 0)[0].tolist()), "one group end per request"
    for t in ends:
        ck[:, req[t]] = completed[:, t].astype(np.float32)
    ring = ring_new.astype(np.float32)
    ring[:, case["lens"] == 0] = np.nan
    return wa.KernelValues(jnp.asarray(keys, jnp.bfloat16), jnp.asarray(ck, jnp.bfloat16),
                           jnp.asarray(ring, jnp.bfloat16))


@pytest.fixture
def counters(monkeypatch):
    """Trace-time calls of the value computations the KVAL path must drop, and `write_all_call`'s modes."""
    c = {"produce_kv": 0, "_indexer_updates": 0, "kernel_indexer_updates": 0, "stats": []}
    for name in ("produce_kv", "_indexer_updates", "kernel_indexer_updates"):
        orig = getattr(wa, name)

        def wrapped(*a, _orig=orig, _name=name, **k):
            c[_name] += 1
            return _orig(*a, **k)

        monkeypatch.setattr(wa, name, wrapped)
    orig_call = wa.write_all_call

    def recording(pool, mp, cp):
        def cb(m_mode, r_live, c_mode):
            m_mode, r_live, c_mode = map(np.asarray, (m_mode, r_live, c_mode))
            c["stats"].append(dict(m1=int((m_mode == 1).sum()), m2=int((m_mode == 2).sum()), runs=int(r_live.sum()),
                                   run=mp["run"], c1=int((c_mode == 1).sum()), c2=int((c_mode == 2).sum())))

        jax.debug.callback(cb, mp["mode"], mp["run_live"], cp["mode"])
        return orig_call(pool, mp, cp)

    monkeypatch.setattr(wa, "write_all_call", recording)
    return c


def compare_three(case, rank, counters, kernel_values, *, ring_tags=None, comp_dcp=False, tag=""):
    """Stock, KVW-ALL and KVAL from the same pool: identical bytes, something written, KVAL engaged (no value
    computation traced, the kernel-value indexer traced, the one write-all call ran)."""
    pool = case["pool"]
    want = run(wc._batched_writes, case, pool, rank, comp_dcp=comp_dcp)
    kvw = run(wa.batched_writes_all, case, pool, rank, ring_tags=ring_tags, comp_dcp=comp_dcp)
    before = dict(counters)
    got = run(wa.batched_writes_all, case, pool, rank, ring_tags=ring_tags, kernel_values=kernel_values,
              comp_dcp=comp_dcp)
    traced = {k: counters[k] - before[k] for k in ("produce_kv", "_indexer_updates", "kernel_indexer_updates")}
    jax.block_until_ready(got)
    jax.effects_barrier()  # the write_all_call recorder's debug callback has run
    st = counters["stats"][-1]
    a = u8(want)
    assert not np.array_equal(a, u8(pool)), "the reference wrote nothing"
    np.testing.assert_array_equal(u8(kvw), a, err_msg="KVW-ALL != stock")
    np.testing.assert_array_equal(u8(got), a, err_msg="KVAL != stock")
    print("Q38_WFOLD_ENGAGE " + str(dict(tag=tag, layout=case["layout"], rows=len(case["positions"]), dcp=case["dcp"],
                                         rank=rank, layers=case["n_layers"], traced=traced, **st)), flush=True)
    assert traced == {"produce_kv": 0, "_indexer_updates": 0, "kernel_indexer_updates": 1}, traced
    assert st["c2"] + st["c1"] > 0, ("no compressed pair written", st)
    return got


# ------------------------------------------------------------------------- (a) the whole-step kernel's own values
KR, KL, KBUDGET, KPAGES, KMAXG = 4, 8, 64, 8, 1024
# Contexts (< 4 * KMAXG): DCP-4 owner changes inside the step and the recent tile (L1-D14's CTX_D4 first), every
# group / ring phase.
K_CTX = [3070, 37, 1021, 254, 509, 2047, 3001, 102, 766, 1534, 255, 4089, 130, 2561, 768, 1986]
_KTILES = None


def _ktiles():
    from vllm_torchtpu.megakernel.exec_spec import Tiles

    return Tiles(bank=1 << 20, nbd=10, nbe=8, ebank=512 << 10, vmem_mib=64)


def _dcp_split(d, D):
    """Main-KV DCP-D layout of the replicated test pool (tests/megakernel/test_wholestep.py `_dcp_split`, credit)."""
    pool = np.array(d["pool"])
    mtab = np.asarray(d["mtab"])
    cols = [mtab[..., k::D] for k in range(D)]
    n = cols[-1].shape[-1]
    nan = np.full(pool.shape[2:], np.nan, np.float32).astype(pool.dtype)
    lead = set(cols[0].ravel().tolist())
    rest = set(np.concatenate([c.ravel() for c in cols[1:]]).tolist())
    for r in range(pool.shape[0]):
        src = pool[r].copy()
        k = r % D
        if k:
            for e, o in zip(cols[0][..., :n].ravel(), cols[k][..., :n].ravel()):
                pool[r, e] = src[o]
        for o in rest - lead:
            pool[r, o] = nan
    d["pool"] = jnp.asarray(pool)
    pad = np.zeros_like(mtab[..., : mtab.shape[-1] - cols[0].shape[-1]])
    d["mtab"] = jnp.asarray(np.concatenate([cols[0], pad], axis=-1))


@functools.lru_cache(maxsize=None)
def kernel_run(rows, variant):
    """The D14 whole-step kernel (interpret, R = 4 one DCP-4 group, `variant` "d14" or "s1" = every D08 S1 stage)
    at `rows` rows, the last request of the step idle (lens 0 rows, as `qsa_step_inputs` passes idle requests) and
    the kernel's padded requests past `rows // T` idle too (M > B at 8 rows). Returns numpy per-rank outputs over the
    kernel's M rows / M // T requests and the inputs the writer needs (`nb` = the step's requests, `nreq` = the
    kernel's)."""
    from jax.sharding import Mesh

    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.exec_spec import ExecSpec
    from vllm_torchtpu.megakernel.models import qwen38
    from vllm_torchtpu.megakernel.models.qwen38_pack import shard_step_inputs
    from vllm_torchtpu.megakernel.ops import BF16, F32
    from vllm_torchtpu.megakernel.wholestep import inputs as wi
    from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, IR_NEW, KR_NEW, RING_NEW
    from vllm_torchtpu.megakernel.wholestep.spec import wholestep_spec

    dcp_env = "VLLM_TPU_QSA_MAIN_KV_DCP"
    saved = os.environ.get(dcp_env)
    os.environ[dcp_env] = "4"
    try:
        R, nb = KR, rows // T
        Gx = dataclasses.replace(qwen38.TINY, NH=3 * R)
        ex = ExecSpec(R=R, B=rows, T=T, ctx=128, interpret=True, tiles=_ktiles())
        M = ex.M
        nreq = M // T
        assert M >= rows and M % T == 0, (M, rows)
        key = 17 + rows
        ctx = K_CTX[:nreq]
        d = wi.random_wholestep_inputs(key, Gx, ex, KL, budget=KBUDGET, n_pages=KPAGES, identical_experts=True)
        d = wi.add_indexer_inputs(key, Gx, ex, d, KL, max_groups=KMAXG, ctx=ctx)
        d = wi.add_step_inputs(key, Gx, ex, d, KL, ctx=ctx)
        # Served-like: BF16 norm weights (the model's), one RoPE table for every RoPE row (`qsa_step_inputs`:
        # qsa_rope = trig[positions], qsa_rope_ck = trig[first position of the completed group]).
        d["kn"] = d["kn"].astype(BF16).astype(F32)
        d["ikn"] = d["ikn"].astype(BF16).astype(F32)
        ctxq = np.asarray(d["ctxq"])
        positions = (ctxq[:, None] + np.arange(T)[None]).reshape(-1).astype(np.int32)
        trig = trig_table(int(ctxq.max()) + 2 * T)
        j0 = ctxq + (3 - ctxq % 4) % 4 - 3
        d["rope"] = jnp.asarray(trig[positions], F32)
        d["rope_ck"] = jnp.asarray(trig[np.maximum(j0, 0)], F32)[:, None, :]
        idle = nb - 1
        nvis, tail = np.array(d["nvis"]), np.array(d["tail"])
        nvis[idle * T:] = 0  # the step's last request and the kernel's padded requests (add_indexer_inputs: lens 0)
        tail[idle * T:] = 0
        d["nvis"], d["tail"] = jnp.asarray(nvis), jnp.asarray(tail)
        d = wi.add_carrier(key, d)
        d["hbm_tables"] = True
        kvw = 2 * Gx.HD  # the ranks share one KV head (a DCP group): same K / V columns, same pool
        d["qkvg"] = d["qkvg"].at[1:, ..., -kvw:].set(
            jnp.broadcast_to(d["qkvg"][:1, ..., -kvw:], d["qkvg"][1:, ..., -kvw:].shape))
        d["pool"] = jnp.broadcast_to(d["pool"][:1], d["pool"].shape)
        pps = d["mtab"].shape[1]
        _dcp_split(d, 4)
        lean = dict(union_bounded=True, union_lean=True, slot_batch=True) if variant == "s1" else {}
        long_kw = dict(carrier="bf16", chunk_pages=8, hbm_tables=True, per_request_scores=True, bounded_chunks=True,
                       shard_index=True, union_attn=True, union_stats=False, union_phys=False, loop_requests=False,
                       dcp=4, compressed_dcp=False, **lean)
        spec = wholestep_spec(Gx, R, L=KL, qsa=dict(budget=KBUDGET), step=True,
                              indexed=dict(max_groups=KMAXG, pages_per_seq=pps, debug=True, **long_kw))
        w_mk, states = wi.pack_wholestep(spec, Gx, ex, d, KL)
        mesh = Mesh(np.array(jax.devices()[:R]), ("tp",))
        pos_mk, hid_mk, w_mk, st_mk = shard_step_inputs(mesh, ex, d["inp"].pos, d["inp"].x, w_mk, states)
        step, _ = kernel.make_step(mesh, spec, ex)
        _, outs = step(pos_mk, hid_mk, w_mk, st_mk, *wi.extras_c(d, mesh, R))
        n_att = KL // Gx.att_every
        out = {k: np.asarray(outs[name]).reshape(R, n_att, *shape) for k, name, shape in (
            ("kv", KV_NEW, (M, -1)), ("kr", KR_NEW, (M, -1)), ("ir", IR_NEW, (M, -1)),
            ("ck", CK_NEW, (nreq, 1, -1)), ("ring", RING_NEW, (nreq, RING_ROWS, -1)))}
        return dict(out, kn=np.asarray(d["kn"]), ikn=np.asarray(d["ikn"]), ctx=ctxq.astype(np.int32),
                    positions=positions, trig=trig, ring_in=np.asarray(d["ring"]), idle=idle, n_att=n_att,
                    rows=rows, nreq=nreq, nb=nb, M=M, hd=Gx.HD)
    finally:
        if saved is None:
            os.environ.pop(dcp_env, None)
        else:
            os.environ[dcp_env] = saved


def _indexer_ref(ko, r, o):
    """`_indexer_updates` (as the writer runs it) on rank r / layer o's raw index rows: completed, complete, ring."""
    B, nreq = ko["rows"], ko["nb"]  # the writer sees the step's requests (wholestep_qsa_writes crops the kernel's)
    lens = np.full(nreq, T, np.int32)
    lens[ko["idle"]] = 0
    starts = jnp.asarray(np.concatenate([[0], np.cumsum(lens)]).astype(np.int32))
    context = jnp.asarray(ko["ctx"][:nreq])
    slots = jnp.where(starts[1:] > starts[:-1], jnp.arange(nreq, dtype=jnp.int32), -1)
    tags = jnp.asarray(coherent_tags(ko["ctx"][:nreq]))
    f = jax.jit(lambda raw, ring, tags, nik, pos, trig: _indexer_updates(
        raw, pos, ring, tags, nik, trig, starts, context, slots, RATIO, EPS, ROTARY))
    completed, complete, req, _, ring, _ = f(jnp.asarray(ko["ir"][r, o, :B]), jnp.asarray(ko["ring_in"][:nreq]),
                                             tags[:nreq], jnp.asarray(ko["ikn"][o], jnp.bfloat16),
                                             jnp.asarray(ko["positions"][:B]), ko["trig"])
    return np.asarray(completed), np.asarray(complete), np.asarray(req), np.asarray(ring)


@pytest.mark.parametrize("variant", ["d14", "s1"])
@pytest.mark.parametrize("rows", ROWS)
def test_kernel_values_equal_recomputed(monkeypatch, rows, variant):
    """(a) Bitwise: the kernel's key rows / completed keys / rings == `_batched_writes`' recomputation from the
    kernel's own `qsa_kr_new` / `qsa_ir_new`, every rank, QSA layer, live row and live request."""
    ko = kernel_run(rows, variant)
    B, hd, nreq, idle = rows, ko["hd"], ko["nb"], ko["idle"]
    live_rows = np.arange(B) < idle * T
    k_ref_fn = jax.jit(lambda kr, v, nk, pos, trig: produce_kv(kr, v, nk, pos, trig, EPS, ROTARY)[0])
    n_rows = n_req = bad_k = bad_c = bad_r = 0
    for shift in (False, True):
        monkeypatch.setattr(envs, SHIFT, shift)
        jax.clear_caches()
        for r in range(KR):
            for o in range(ko["n_att"]):
                kv = ko["kv"][r, o, :B]
                k_ref = np.asarray(k_ref_fn(jnp.asarray(ko["kr"][r, o, :B]), jnp.asarray(kv[:, hd:]),
                                            jnp.asarray(ko["kn"][o], jnp.bfloat16), jnp.asarray(ko["positions"][:B]),
                                            ko["trig"])).reshape(B, -1)
                diff = bits(kv[:, :hd])[live_rows] != bits(k_ref)[live_rows]
                bad_k += int(diff.sum())
                n_rows += int(live_rows.sum())
                completed, complete, req, ring = _indexer_ref(ko, r, o)
                ends = np.nonzero(complete)[0]
                assert sorted(req[ends].tolist()) == [q for q in range(nreq) if q != idle], (req[ends], idle)
                for t in ends:
                    q = int(req[t])
                    bad_c += int((bits(ko["ck"][r, o, q, 0]) != bits(completed[t])).sum())
                    bad_r += int((bits(ko["ring"][r, o, q]) != bits(ring[q])).sum())
                    n_req += 1
    print("Q38_WFOLD_KVAL_EQ " + str(dict(variant=variant, rows=rows, ranks=KR, layers=ko["n_att"], live_rows=n_rows,
                                          live_requests=n_req, key_bits_diff=bad_k, ck_bits_diff=bad_c,
                                          ring_bits_diff=bad_r)), flush=True)
    assert n_rows > 0 and n_req > 0
    assert bad_k == 0 and bad_c == 0 and bad_r == 0, (bad_k, bad_c, bad_r)


# ----------------------------------------------------------------- (b) pool bytes from the real kernel outputs
@pytest.mark.parametrize("layout", ["served", "paired", "lone"])
@pytest.mark.parametrize("rows", ROWS)
def test_pool_bytes_from_kernel_outputs(counters, rows, layout):
    """(b) The kernel's own outputs (variant d14) on a 2-layer writer: KVAL == KVW-ALL == stock bytes on every DCP-4
    rank, directly (the kernel's requests cropped to the step's) and through `wholestep_qsa_writes(kernel_values=...)`
    with the kernel's M // T requests (as `per_rank_writes` hands them in)."""
    ko = kernel_run(rows, "d14")
    B, hd, n_att, nb = rows, ko["hd"], ko["n_att"], ko["nb"]
    owned = 0
    with interpret():
        for r in range(KR):
            rows_in = [(jnp.asarray(ko["kr"][r, o, :B]), jnp.asarray(ko["kv"][r, o, :B, hd:]),
                        jnp.asarray(ko["ir"][r, o, :B])) for o in range(n_att)]
            norms = [(jnp.asarray(ko["kn"][o], jnp.bfloat16), jnp.asarray(ko["ikn"][o], jnp.bfloat16))
                     for o in range(n_att)]
            case = make_case(rows * 13 + r, rows, layout, n_layers=n_att, dcp=4, idle=(ko["idle"],),
                             contexts=list(ko["ctx"][:nb]), rows_in=rows_in, norms=norms, trig=ko["trig"])
            case = seed_rings(case, ko["ring_in"][:nb])
            kv = wa.kernel_values_from_outputs(jnp.asarray(ko["kv"][r]), jnp.asarray(ko["ck"][r]),
                                               jnp.asarray(ko["ring"][r]), B)
            got = compare_three(case, r, counters, kv._replace(ck=kv.ck[:, :nb], ring=kv.ring[:, :nb]), tag="kernel")
            disp = run_dispatch(case, case["pool"], r, kv, extra_requests=ko["nreq"] - nb)
            np.testing.assert_array_equal(u8(disp), u8(got), err_msg="wholestep_qsa_writes dispatch != direct")
            jax.effects_barrier()
            st = counters["stats"][-1]
            owned += st["m1"] + st["m2"] + st["runs"] * st["run"]
    valid = int((np.arange(B) < ko["idle"] * T).sum())
    want = n_att * valid if layout == "served" else (valid * 2 if layout == "lone" else valid)
    assert owned == want, ("main rows written over the 4 ranks", owned, want)


# -------------------------------------------------------------- (b) 11 layers, values from the stock formulas
MATRIX = [
    (8, "served", 0, True), (16, "served", 1, False), (32, "served", 2, True), (48, "served", 3, False),
    (64, "served", 1, True), (16, "paired", 1, True), (48, "paired", 2, False), (64, "paired", 0, True),
    (32, "bench", 3, False), (64, "bench", 2, True),
]


@pytest.mark.parametrize("rows,layout,rank,served", MATRIX,
                         ids=[f"{r}rows-{lay}-rank{k}-{'senv' if s else 'denv'}" for r, lay, k, s in MATRIX])
def test_eleven_layers_match(counters, monkeypatch, rows, layout, rank, served):
    if served:
        for k, v in SERVED_QSA_ENV.items():
            monkeypatch.setattr(envs, k, v)
    with interpret():
        case = make_case(rows * 7 + rank, rows, layout, dcp=4, idle=(rows // T - 1,))
        compare_three(case, rank, counters, stock_kernel_values(case, case["pool"]), tag="eleven")


def test_compressed_dcp_and_512_row_pages(counters):
    with interpret():
        case = make_case(23, 64, "served", page_rows=512, dcp=4, idle=(4, 15))
        compare_three(case, 1, counters, stock_kernel_values(case, case["pool"]), tag="p512")
        case = make_case(24, 32, "served", dcp=4, comp_dcp=True, idle=(7,))
        compare_three(case, 2, counters, stock_kernel_values(case, case["pool"]), comp_dcp=True, tag="cdcp")


@pytest.mark.parametrize("masked", [False, True])
def test_ring_tags_handed_in(counters, masked):
    with interpret():
        case = make_case(19, 32, "served", dcp=4, idle=(2, 5), stale_idle_rings=True)
        ids = jnp.concatenate([t[0] for t in case["tables"]])
        if masked:
            active = case["starts"][1:] > case["starts"][:-1]
            ids = jnp.where(jnp.tile(active, L), ids, 0)
        ring_tags = jax.jit(lambda p, i: read_ring_and_tags(p, i, RING_ROWS * INDEX_DIM, 16))(case["pool"], ids)
        compare_three(case, 1, counters, stock_kernel_values(case, case["pool"]), ring_tags=ring_tags, tag="ringtags")


def test_two_consecutive_steps(counters):
    """The second step reads the first step's rings and overwrites its rejected drafts."""
    with interpret():
        case = make_case(17, 32, "paired", dcp=4, idle=(7,))
        accept = np.array([1, 4, 2, 3, 1, 4, 2, 0], np.int32)
        p_ref = p_new = case["pool"]
        pool0 = u8(case["pool"])
        for step in range(2):
            kv = stock_kernel_values(case, p_new)
            p_ref = run(wc._batched_writes, case, p_ref, 0)
            p_new = run(wa.batched_writes_all, case, p_new, 0, kernel_values=kv)
            np.testing.assert_array_equal(u8(p_new), u8(p_ref))
            case = make_case(18 + step, 32, "paired", dcp=4, prev=case,
                             contexts=list(case["ctx"] + np.where(case["lens"] > 0, accept, 0)))
        assert not np.array_equal(u8(p_ref), pool0)


def test_dispatch_crops_kernel_requests(counters):
    """`wholestep_qsa_writes` crops the kernel's `M // T` requests to the step's (`num_requests`)."""
    with interpret():
        case = make_case(31, 32, "served", dcp=4, idle=(7,))
        kv = stock_kernel_values(case, case["pool"])
        want = run(wc._batched_writes, case, case["pool"], 3)
        pad = lambda x: jnp.concatenate([x, jnp.full((x.shape[0], 2) + x.shape[2:], jnp.nan, x.dtype)], 1)  # noqa: E731
        got = run_dispatch(case, case["pool"], 3, kv._replace(ck=pad(kv.ck), ring=pad(kv.ring)), extra_requests=2)
        np.testing.assert_array_equal(u8(got), u8(want))


# ----------------------------------------------------------------------- (c) the values are used, not recomputed
@pytest.mark.parametrize("field", ["keys", "ck", "ring"])
def test_writer_uses_kernel_values(counters, field):
    """A changed value in one layer's handed-in key rows / completed keys / rings changes the pool (a writer that
    kept recomputing would not notice)."""
    with interpret():
        case = make_case(41, 32, "served", dcp=4, idle=(7,))
        kv = stock_kernel_values(case, case["pool"])
        base = run(wa.batched_writes_all, case, case["pool"], 0, kernel_values=kv)
        np.testing.assert_array_equal(u8(base), u8(run(wc._batched_writes, case, case["pool"], 0)))
        x = np.array(getattr(kv, field), np.float32)
        x = np.where(np.isnan(x), x, x * 2 + 1)  # past FP8 rounding for the key rows
        bumped = kv._replace(**{field: getattr(kv, field).at[3].set(jnp.asarray(x[3], jnp.bfloat16))})
        got = run(wa.batched_writes_all, case, case["pool"], 0, kernel_values=bumped)
        assert not np.array_equal(u8(got), u8(base)), f"a changed {field} did not reach the pool"


def test_kval_trace_has_no_value_computation():
    """The lowered KVAL writer has no rsqrt (the key's and the completed group's gemma norms) and fewer ops."""
    with interpret():
        case = make_case(43, 32, "served", dcp=4, idle=(7,))
        kv = stock_kernel_values(case, case["pool"])

        def lowered(kernel_values):
            def f(pool, rows, norms, tables, positions, trig, starts, context, rank, kv_):
                return wa.batched_writes_all(pool, rows, norms, tables, case["layers"], positions, trig, starts,
                                             context, compress_ratio=RATIO, norm_eps=EPS, rotary_dim=ROTARY,
                                             dcp=(rank, 4), kernel_values=kv_)

            return jax.jit(f).lower(case["pool"], case["rows"], case["norms"], case["tables"], case["positions"],
                                    case["trig"], case["starts"], case["context"], jnp.asarray([0], jnp.int32),
                                    kernel_values).as_text()

        off, on = lowered(None), lowered(kv)
    count = lambda t, op: t.count("stablehlo." + op)  # noqa: E731
    n_ops = lambda t: sum(1 for ln in t.splitlines() if "= stablehlo." in ln or "= \"stablehlo." in ln)  # noqa: E731
    print("Q38_WFOLD_OPS " + str(dict(rsqrt_off=count(off, "rsqrt"), rsqrt_on=count(on, "rsqrt"),
                                      ops_off=n_ops(off), ops_on=n_ops(on))), flush=True)
    assert count(off, "rsqrt") > 0 and count(on, "rsqrt") == 0
    assert n_ops(on) < n_ops(off)


def test_kernel_values_checked():
    """Wrong shapes and steps with more than one completed group per request are refused (no silent fallback)."""
    rows = [(jnp.zeros((8, HEAD_DIM), jnp.bfloat16), jnp.zeros((8, HEAD_DIM), jnp.bfloat16),
             jnp.zeros((8, INDEX_DIM), jnp.bfloat16))] * 2
    good = wa.KernelValues(jnp.zeros((2, 8, HEAD_DIM), jnp.bfloat16), jnp.zeros((2, 2, INDEX_DIM), jnp.bfloat16),
                           jnp.zeros((2, 2, RING_ROWS, INDEX_DIM), jnp.bfloat16))
    wa.check_kernel_values(good, rows, 8, 2, RING_ROWS, RATIO)
    with pytest.raises(ValueError, match="keys"):
        wa.check_kernel_values(good._replace(keys=good.keys[:1]), rows, 8, 2, RING_ROWS, RATIO)
    with pytest.raises(ValueError, match="ring"):
        wa.check_kernel_values(good._replace(ring=good.ring[:, :1]), rows, 8, 2, RING_ROWS, RATIO)
    with pytest.raises(ValueError, match="one completed group"):
        wa.check_kernel_values(good._replace(ck=good.ck[:, :1], ring=good.ring[:, :1]), rows, 8, 1, RING_ROWS, RATIO)
    with pytest.raises(ValueError, match="key rows"):
        wa.check_kernel_values(good._replace(keys=good.keys.astype(jnp.float32)), rows, 8, 2, RING_ROWS, RATIO)


# ------------------------------------------------------------------- the whole-step call's plumbing (stubbed)
CR, CB, CL, CN_ATT, CH, CHD = 2, 16, 8, 2, 8, 4
CPOOL = 1024


def _build_stubbed_call(monkeypatch, *, stats=False):
    """`build_wholestep_call` with the kernel, the QSA inputs, the scatter and the writer stubbed (the pattern of
    test_l1d07b1_gdn_meta_shared, credit). The stub kernel returns, per requested extra output, a distinct array;
    the stub writer stores what reaches it (rows, the kernel values) in the pool."""
    from types import SimpleNamespace

    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.megakernel import kernel
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, RING_NEW, UNION_STATS
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    names = ("qsa_pool", "qsa_ck", "qsa_cktab", "qsa_nvis", "qsa_tail", "qsa_mtab", "qsa_off", "qsa_scales",
             "qsa_rope", "qsa_ctx", "qsa_ring", "qsa_rope_ck")
    extra = names + ("gdn_pool", "gdn_rb", "gdn_hi")
    seen = {}

    def mfss(mesh, spec, ex, extra_outs=()):
        seen["extra_outs"] = tuple(extra_outs)
        M, nreq = CB, CB // T

        def fn(pos, x, *rest):
            xs = dict(zip(extra, rest[: len(extra)]))
            rb, hi = xs["gdn_rb"], xs["gdn_hi"]
            outs = [x, jnp.broadcast_to(rb[None], (CR,) + rb.shape), jnp.broadcast_to(hi[None], (CR,) + hi.shape)]
            for i, n in enumerate(extra_outs):
                shape = {CK_NEW: (nreq, 1, CHD), RING_NEW: (nreq, RING_ROWS, CHD), UNION_STATS: (16, 1, 128)}.get(
                    n, (M, 2 * CHD))
                outs.append((jnp.arange(CR * CN_ATT * int(np.prod(shape)), dtype=jnp.float32).reshape(
                    (CR * CN_ATT,) + shape) * 0.25 + 1000 * (i + 1)).astype(jnp.bfloat16 if n != UNION_STATS
                                                                          else jnp.int32))
            return tuple(outs)

        fn.extra_names = extra
        return fn, (P(), P("tp")) + (P(),) * len(extra), None, 1

    def inputs(pool, tables, slots, trig, positions, starts, context, **kw):
        return {n: jnp.zeros((1,), jnp.int32) for n in names}

    def scatter(pool, conv_ck, rec_ck, wb, layout=None, interpret=False):
        return pool

    def writes(pool, rows, norms, tables, qsa_layers, positions, trig, starts, context, kernel_values=None, **kw):
        flat = [r[0].reshape(-1).astype(jnp.float32) for r in rows]
        if kernel_values is not None:
            flat += [x.reshape(-1).astype(jnp.float32) for x in kernel_values]
        flat = jnp.concatenate(flat)
        pool = pool.at[: flat.shape[0]].set(flat)
        return pool.at[-1].set(1.0 if kernel_values is not None else 0.0)

    monkeypatch.setattr(wsc, "wholestep_spec", lambda *a, **k: None)
    monkeypatch.setattr(wsc, "make_flat_segment_step", mfss)
    monkeypatch.setattr(kernel, "make_step", lambda mesh, spec, ex: (
        None, {"plan": SimpleNamespace(w_names=["w0"]), "wshape": {"w0": ((4, CH), jnp.float32)}}))
    monkeypatch.setattr(wsc, "qsa_step_inputs", inputs)
    monkeypatch.setattr(wsc, "scatter_gdn_ckpts_fast", scatter)
    monkeypatch.setattr(wsc, "wholestep_qsa_writes", writes)
    monkeypatch.setattr(wsc, "UNION_ATTN", stats)
    monkeypatch.setattr(wsc, "UNION_STATS_EVERY", 1 if stats else 0)
    layers = [QSALayerCache(compressed_slot=o % 2, main_slot=-1) for o in range(CN_ATT)]
    from jax.sharding import Mesh

    mesh = Mesh(np.array(jax.devices()[:CR]), ("tp",))
    call, in_specs, out_specs, donate = wsc.build_wholestep_call(
        mesh, SimpleNamespace(att_every=4), CB, None, layers, L=CL, budget=64, max_groups=1024, norm_eps=1e-6,
        rotary_dim=64, interpret=True)
    return call, in_specs, out_specs, seen


def _call_args():
    from vllm_torchtpu.megakernel_runtime.qwen38_segment_call import GDN_META

    rng = np.random.default_rng(3)
    n_gdn = CL - CN_ATT
    meta = []
    for _ in range(n_gdn):  # per GDN layer: state ids, read offsets, ckpt ids, query starts, seq lens
        meta += [np.arange(CB, dtype=np.int32), np.zeros((97,), np.int32), np.zeros((CB, T), np.int32),
                 (np.arange(CB + 1) * 1).astype(np.int32), np.ones((CB,), np.int32)]
    assert len(meta) == GDN_META * n_gdn
    tabs = [rng.integers(0, 50, (CB * n,)).astype(np.int32) for _ in range(CN_ATT) for n in (8, 32, 32)]
    tail = [*tabs, np.arange(CB, dtype=np.int32), (np.arange(CB // T + 1) * T).astype(np.int32),
            np.full((CB // T,), 40, np.int32), rng.standard_normal((16, 128)).astype(np.float32),
            *[rng.standard_normal((n,)).astype(np.float32) for _ in range(CN_ATT) for n in (2 * CHD, 128)]]
    hidden = rng.standard_normal((CR * CB, CH)).astype(np.float32)
    w0 = rng.standard_normal((CR * 4, CH)).astype(np.float32)
    pool = np.zeros((CR * CPOOL,), np.float32)
    return [hidden, w0, pool, *meta, *tail]


@pytest.mark.parametrize("stats", [False, True])
@pytest.mark.parametrize("flags", ["off", "kval_only", "write_all_only", "both"])
def test_call_plumbs_kernel_values(monkeypatch, flags, stats):
    """With both flags the call requests `qsa_ck_new` / `qsa_ring_new` after every other output and the writer
    receives `KernelValues` = (rank's `qsa_kv_new[:, :B, :D]`, `qsa_ck_new[..., 0, :]`, `qsa_ring_new`); otherwise
    neither the outputs nor the writer arguments change."""
    from vllm_torchtpu.megakernel.wholestep.qsa_full import KV_NEW
    from vllm_torchtpu.megakernel.wholestep.qsa_full_c import CK_NEW, IR_NEW, KR_NEW, RING_NEW, UNION_STATS

    monkeypatch.delenv("VLLM_TPU_MK_GDN_META_DEDUP", raising=False)
    for name, on in (("VLLM_TPU_MK_QSA_WRITE_ALL", flags in ("write_all_only", "both")),
                     ("VLLM_TPU_MK_QSA_WRITE_KVAL", flags in ("kval_only", "both"))):
        if on:
            monkeypatch.setenv(name, "1")
        else:
            monkeypatch.delenv(name, raising=False)
    call, in_specs, out_specs, seen = _build_stubbed_call(monkeypatch, stats=stats)
    base = (KV_NEW, KR_NEW, IR_NEW) + ((UNION_STATS,) if stats else ())
    kval = flags == "both"
    assert seen["extra_outs"] == base + ((CK_NEW, RING_NEW) if kval else ()), seen["extra_outs"]
    outs = jax.jit(call)(*_call_args())
    assert len(outs) == (3 if stats else 2)
    pool = np.asarray(outs[1]).reshape(CR, CPOOL)
    # What the stub kernel returned (per rank): output i of the extra outs.
    def kout(name):
        i = seen["extra_outs"].index(name)
        shape = {CK_NEW: (CB // T, 1, CHD), RING_NEW: (CB // T, RING_ROWS, CHD)}.get(name, (CB, 2 * CHD))
        x = (np.arange(CR * CN_ATT * int(np.prod(shape)), dtype=np.float32).reshape((CR * CN_ATT,) + shape) * 0.25
             + 1000 * (i + 1))
        return np.asarray(jnp.asarray(x, jnp.bfloat16).astype(jnp.float32)).reshape((CR, CN_ATT) + shape)

    kr = kout(KR_NEW)
    for r in range(CR):
        want = [kr[r, o, :CB].reshape(-1) for o in range(CN_ATT)]
        if kval:
            want += [kout(KV_NEW)[r, :, :CB, :CHD].reshape(-1), kout(CK_NEW)[r][:, :, 0].reshape(-1),
                     kout(RING_NEW)[r].reshape(-1)]
        want = np.concatenate(want)
        np.testing.assert_array_equal(pool[r, : len(want)], want, err_msg=f"rank {r}")
        assert pool[r, -1] == (1.0 if kval else 0.0)


def test_flag_registered_default_off_and_needs_write_all(monkeypatch):
    for name in ("VLLM_TPU_MK_QSA_WRITE_KVAL", "VLLM_TPU_MK_QSA_WRITE_ALL"):
        monkeypatch.delenv(name, raising=False)
        assert name in envs.environment_variables
        assert envs.environment_variables[name]() is False
    assert not wa.qsa_write_kval_enabled()
    monkeypatch.setenv("VLLM_TPU_MK_QSA_WRITE_KVAL", "1")
    assert envs.environment_variables["VLLM_TPU_MK_QSA_WRITE_KVAL"]() is True
    assert not wa.qsa_write_kval_enabled()  # effective only with VLLM_TPU_MK_QSA_WRITE_ALL
    monkeypatch.setenv("VLLM_TPU_MK_QSA_WRITE_ALL", "1")
    assert wa.qsa_write_kval_enabled()


def test_import_origin():
    """Provenance guard: with Q38_EXPECT_ROOT set, the modules come from under it."""
    from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_call as wsc

    root = os.environ.get("Q38_EXPECT_ROOT", "")
    files = tuple(os.path.realpath(m.__file__) for m in (wa, wc, wsc, envs))
    print("Q38_WFOLD_IMPORT_ORIGIN " + " ".join(files))
    if root:
        root = os.path.realpath(root)
        assert all(f.startswith(root + os.sep) for f in files), (root, files)
