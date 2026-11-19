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
"""q38 L2 KVW-ALL: the all-layer QSA cache writer vs today's `_batched_writes` (CPU, Pallas interpret).

Every case writes one step of 11 QSA layers (the whole step's count) twice from the same operands:
`qwen38_wholestep_cache._batched_writes` (today: per layer the main writer + `per_pair_kernel`, one
batched ring read / write) and `qwen38_qsa_write_all.batched_writes_all` (one Pallas call for every
layer's main rows and compressed pairs). The pool bytes must be identical. The pool starts as random
finite FP8 bytes, so a writer that touches a byte it should keep is caught.

Layouts (11 layers):
  served  whole-row main KV (main_slot -1, own pages per layer), packed compressed keys (slots 0-7,
          0-2: two groups), the PD served decode geometry;
  paired  QWEN4_EXP_PACK_MAIN_KV pairs (slots 0 / 1, consecutive layers share main pages); the 11th
          layer's partner (layer 47) is outside the whole step, so its rows need read-merge-write;
  bench   tests/megakernel/tpu_bench_wholestep_call.py's layout (main_slot (o + 1) % 2, compressed
          slot o % 2).
Rows 8 / 16 / 32 / 48 / 64 (4-row verify windows), main-KV DCP 4 with owner masking on every rank
(compressed keys replicated or DCP-split), partial compressed pairs (every 4-row window leaves one),
full pairs (8-row windows), ragged windows, idle requests, page crossings, two consecutive steps,
the ring / tag read handed in (`ring_tags`), 512-row pages.

Engagement per case (a recorder around `write_all_call`): main rows written = layers x owned tokens
(served) or 5 pairs + the lone layer's read-merge-writes (paired / bench); partial pairs > 0.
Provenance: with Q38_EXPECT_ROOT set, both modules must import from under it.
Run: JAX_PLATFORMS=cpu PYTHONPATH=src pytest tests/megakernel_runtime/test_qwen38_qsa_write_all.py
"""

from __future__ import annotations

import logging
import os
import sys
import types

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
from vllm_torchtpu.kernels.qwen4_exp.qsa_state import read_ring_and_tags  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_qsa_write_all as wa  # noqa: E402
from vllm_torchtpu.megakernel_runtime import qwen38_wholestep_cache as wc  # noqa: E402
from vllm_torchtpu.megakernel_runtime.qwen38_wholestep_cache import QSALayerCache  # noqa: E402

L, WIDTH, HEAD_DIM, INDEX_DIM, RATIO, ROTARY = 11, 256, 256, 128, 4, 64
K_SCALE, V_SCALE = 0.375, 2.5
FP8 = jnp.float8_e4m3fn
SERVED_QSA_ENV = {  # the served decode QSA flags that reach this code (both writers share them)
    "VLLM_TPU_QSA_INDEXER_UPDATE_SHIFT": True,
    "VLLM_TPU_QSA_RING_DMA_BATCH": 32,
}

COMP_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2]
COMP_GROUP = [0] * 8 + [1] * 3
LAYOUTS = {
    "served": dict(main=[-1] * L, main_group=list(range(L)), comp=COMP_SLOTS, comp_group=COMP_GROUP),
    "paired": dict(main=[o % 2 for o in range(L)], main_group=[o // 2 for o in range(L)],
                   comp=COMP_SLOTS, comp_group=COMP_GROUP),
    "bench": dict(main=[(o + 1) % 2 for o in range(L)], main_group=[o // 2 for o in range(L)],
                  comp=[o % 2 for o in range(L)], comp_group=[o // 2 for o in range(L)]),
}


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(pl, "pallas_call", lambda *a, **k: original(*a, **{**k, "interpret": True}))
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_FUSED_KV_WRITE", False)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", False)


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(jnp.asarray(x), jnp.uint8))


def finite_fp8(rng, shape):
    """Random FP8 E4M3 bytes without the two NaN codes (0x7F / 0xFF)."""
    b = rng.integers(0, 256, shape, dtype=np.uint8)
    b = np.where((b & 0x7F) == 0x7F, b ^ 0x01, b).astype(np.uint8)
    return jax.lax.bitcast_convert_type(jnp.asarray(b), FP8)


def layers_of(layout):
    spec = LAYOUTS[layout]
    return [QSALayerCache(compressed_slot=c, main_slot=m, k_scale=K_SCALE, v_scale=V_SCALE)
            for m, c in zip(spec["main"], spec["comp"])]


def make_case(seed, rows, layout, *, page_rows=64, dcp=1, comp_dcp=False, per=4, ragged=False,
              idle=(), stale_idle_rings=False, contexts=None, prev=None):
    """One step: rows // per request slots of `per` rows (0 for `idle`, 1..per when `ragged`).

    Pages are allocated for every request's next two windows, so a following step (`prev`: same
    requests, tables, norms and rope table; new projections) stays on allocated pages."""
    rng = np.random.default_rng(seed)
    spec = LAYOUTS[layout]
    nreq = rows // per
    if prev is not None:
        lens = prev["lens"]
    else:
        lens = np.array([0 if r in idle else (int(rng.integers(1, per + 1)) if ragged else per)
                         for r in range(nreq)], np.int32)
    if contexts is None:
        # Request r sits in DCP owner window r % dcp (main pages of page_rows rows); phases cover every
        # ring / group (mod 4) and pair (mod 8) phase and page crossings inside the window.
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

    rows_in = [(normal(rows, HEAD_DIM), normal(rows, HEAD_DIM), normal(rows, INDEX_DIM)) for _ in range(L)]
    case = dict(rows=rows_in, layers=layers_of(layout), positions=jnp.asarray(positions),
                starts=jnp.asarray(starts), context=jnp.asarray(ctx), lens=lens, ctx=ctx, page_rows=page_rows,
                dcp=dcp, comp_dcp=comp_dcp, layout=layout, per=per)
    if prev is not None:
        return {**case, **{k: prev[k] for k in ("tables", "norms", "trig", "pages")}}
    span = max(dcp, 1) * page_rows
    cspan = (max(dcp, 1) if comp_dcp else 1) * page_rows
    last = ctx + 2 * per - 1
    cols = int(last.max() // page_rows) + 2
    n_main, n_comp = max(spec["main_group"]) + 1, max(spec["comp_group"]) + 1
    main = np.zeros((n_main, nreq, cols), np.int32)
    comp = np.zeros((n_comp, nreq, cols), np.int32)
    ring = np.zeros((L, nreq), np.int32)
    slots = []  # (array, index) of every allocated page, numbered after a shuffle
    for r in range(nreq):
        active = lens[r] > 0
        if active:
            for g in range(n_main):
                slots += [(main, (g, r, c)) for c in range(ctx[r] // span, last[r] // span + 1)]
            for g in range(n_comp):
                slots += [(comp, (g, r, c)) for c in range(ctx[r] // cspan, last[r] // cspan + 1)]
        if active or stale_idle_rings:
            slots += [(ring, (o, r)) for o in range(L)]
    for (arr, idx), page in zip(slots, 1 + rng.permutation(len(slots))):
        arr[idx] = page
    pages = len(slots) + 1
    tables = [(jnp.asarray(ring[o]), jnp.asarray(main[spec["main_group"][o]]),
               jnp.asarray(comp[spec["comp_group"][o]])) for o in range(L)]
    return {**case, "tables": tables, "norms": [(normal(HEAD_DIM), normal(INDEX_DIM)) for _ in range(L)],
            "trig": jnp.pad(normal(8192, ROTARY), ((0, 0), (0, 128 - ROTARY))), "pages": pages,
            "pool": finite_fp8(rng, (pages, page_rows, 1, 4, WIDTH))}


def owned_tokens(case, rank):
    """Host count of (valid tokens, main-KV tokens this rank owns): `qsa_cache_writes`' owner rule."""
    lens, ctx, B, D = case["lens"], case["ctx"], case["page_rows"], max(case["dcp"], 1)
    valid = owned = 0
    for r in range(len(lens)):
        for t in range(lens[r]):
            valid += 1
            owned += int(((ctx[r] + t) // B) % D == rank) if case["dcp"] > 1 else 1
    return valid, owned


def run(fn, case, pool, rank, ring_tags=None):
    """`fn` (either writer) jitted as the whole step calls it: DCP size static, the rank an int32 [1]."""
    d, cd = case["dcp"], case["comp_dcp"]

    def f(pool, rows, norms, tables, positions, trig, starts, context, rank_arr, ring_tags):
        kw = {}
        if d > 1:
            pair = (rank_arr, d)
            kw = dict(dcp=pair, compressed_dcp=pair if cd else None)
        if ring_tags is not None:
            kw["ring_tags"] = ring_tags
        return fn(pool, rows, norms, tables, case["layers"], positions, trig, starts, context,
                  compress_ratio=RATIO, norm_eps=1e-6, rotary_dim=ROTARY, **kw)

    return jax.jit(f)(pool, case["rows"], case["norms"], case["tables"], case["positions"], case["trig"],
                      case["starts"], case["context"], jnp.asarray([rank], jnp.int32), ring_tags)


@pytest.fixture
def recorder(monkeypatch):
    """Per call of `write_all_call`: counts of each main / pair mode and of live runs."""
    stats = []
    original = wa.write_all_call

    def recording(pool, mp, cp):
        def cb(m_mode, r_live, c_mode):
            m_mode, r_live, c_mode = map(np.asarray, (m_mode, r_live, c_mode))
            stats.append(dict(m1=int((m_mode == 1).sum()), m2=int((m_mode == 2).sum()),
                              runs=int(r_live.sum()), run=mp["run"], c1=int((c_mode == 1).sum()),
                              c2=int((c_mode == 2).sum()), merge=mp["take"] is not None))

        jax.debug.callback(cb, mp["mode"], mp["run_live"], cp["mode"])
        return original(pool, mp, cp)

    monkeypatch.setattr(wa, "write_all_call", recording)
    return stats


def check_engagement(case, rank, st):
    valid, owned = owned_tokens(case, rank)
    rows_written = st["m1"] + st["m2"] + st["runs"] * st["run"]
    print("Q38_KVW_ENGAGE " + str(dict(layout=case["layout"], rows=len(case["positions"]), dcp=case["dcp"],
                                       rank=rank, valid=valid, owned=owned, **st)))
    if case["layout"] == "served":
        assert not st["merge"] and st["m2"] == 0, st
        assert rows_written == L * owned, (st, owned)
    else:
        assert st["merge"], st
        assert st["m2"] == owned, (st, owned)  # the lone 11th layer: read-merge-write
        assert st["m1"] + st["runs"] * st["run"] == 5 * owned, (st, owned)  # pair writes, folded
    assert valid > 0 and owned > 0, (valid, owned)
    if case["dcp"] > 1:
        assert owned < valid, ("owner masking not exercised", valid, owned)
    assert st["c2"] > 0, ("no partial compressed pair", st)
    if case["per"] > 4:
        assert st["c1"] > 0, ("no full compressed pair", st)


def compare(case, rank, recorder, ring_tags=None):
    """Both writers on the same operands: identical pool bytes, something written, engagement."""
    want = run(wc._batched_writes, case, case["pool"], rank)
    got = run(wa.batched_writes_all, case, case["pool"], rank, ring_tags=ring_tags)
    a, b = raw(want), raw(got)
    assert not np.array_equal(a, raw(case["pool"])), "the reference wrote nothing"
    np.testing.assert_array_equal(b, a)
    check_engagement(case, rank, recorder[-1])


# (rows, layout, dcp, rank, comp_dcp, served env)
MATRIX = [
    (8, "served", 4, 0, False, True), (16, "served", 4, 1, False, False), (32, "served", 4, 2, True, True),
    (48, "served", 4, 3, False, False), (64, "served", 4, 1, True, True),
    (32, "served", 4, 0, False, False), (32, "served", 4, 1, False, True), (32, "served", 4, 3, True, False),
    (64, "served", 1, 0, False, True),
    (8, "paired", 1, 0, False, False), (32, "paired", 1, 0, False, True), (64, "paired", 1, 0, False, False),
    (16, "paired", 4, 1, False, True), (48, "paired", 4, 2, True, False),
    (16, "bench", 1, 0, False, True), (64, "bench", 4, 3, False, False),
]


@pytest.mark.parametrize("rows,layout,dcp,rank,comp_dcp,served", MATRIX,
                         ids=[f"{r}rows-{lay}-dcp{d}-rank{k}-{'cdcp' if c else 'crep'}-{'senv' if s else 'denv'}"
                              for r, lay, d, k, c, s in MATRIX])
def test_write_all_matches_batched_writes(interpret, recorder, monkeypatch, rows, layout, dcp, rank, comp_dcp,
                                          served):
    if served:
        for k, v in SERVED_QSA_ENV.items():
            monkeypatch.setattr(envs, k, v)
    idle = (rows // 4 - 1,) if rows // 4 > 3 else ()  # the last request idle: rank r owns request r
    case = make_case(rows * 7 + rank, rows, layout, dcp=dcp, comp_dcp=comp_dcp, idle=idle)
    compare(case, rank, recorder)


@pytest.mark.parametrize("layout", ["served", "paired"])
def test_full_pairs_with_8_row_windows(interpret, recorder, layout):
    case = make_case(11, 32, layout, per=8)
    compare(case, 0, recorder)


def test_ragged_windows_and_idle_requests(interpret, recorder):
    case = make_case(13, 48, "paired", ragged=True, idle=(3, 7))
    compare(case, 0, recorder)


def test_two_consecutive_steps(interpret, recorder):
    """The second step overwrites the first step's rejected drafts and reads its rings."""
    case = make_case(17, 32, "paired", dcp=4, idle=(5,))
    accept = np.array([1, 4, 2, 3, 1, 0, 4, 2], np.int32)
    p_ref = p_new = case["pool"]
    pool0 = raw(case["pool"])
    for step in range(2):
        p_ref = run(wc._batched_writes, case, p_ref, 0)
        p_new = run(wa.batched_writes_all, case, p_new, 0)
        np.testing.assert_array_equal(raw(p_new), raw(p_ref))
        check_engagement(case, 0, recorder[-1])
        case = make_case(18 + step, 32, "paired", dcp=4, prev=case,
                         contexts=list(case["ctx"] + np.where(case["lens"] > 0, accept, 0)))
    assert not np.array_equal(raw(p_ref), pool0)


@pytest.mark.parametrize("masked", [False, True])
def test_ring_tags_handed_in(interpret, recorder, masked):
    """The kernel-input ring / tag read reused (`ring_tags`), with ids as read (masked: idle requests
    read the null page, as `qsa_step_inputs` masks them) instead of a second read."""
    case = make_case(19, 32, "served", dcp=4, idle=(2, 5), stale_idle_rings=True)
    ids = jnp.concatenate([t[0] for t in case["tables"]])
    if masked:
        active = case["starts"][1:] > case["starts"][:-1]
        ids = jnp.where(jnp.tile(active, L), ids, 0)
    ring_tags = jax.jit(lambda p, i: read_ring_and_tags(p, i, 8 * INDEX_DIM, 16))(case["pool"], ids)
    compare(case, 1, recorder, ring_tags=ring_tags)


def test_512_row_pages(interpret, recorder):
    case = make_case(23, 64, "served", page_rows=512, dcp=4, idle=(4,))
    compare(case, 1, recorder)


def test_check_catches_swapped_main_tables(interpret, recorder):
    """Positive control: two served layers' main tables swapped must change the pool."""
    case = make_case(29, 32, "served", dcp=1)
    want = run(wc._batched_writes, case, case["pool"], 0)
    t = list(case["tables"])
    t[3], t[4] = (t[3][0], t[4][1], t[3][2]), (t[4][0], t[3][1], t[4][2])
    got = run(wa.batched_writes_all, {**case, "tables": t}, case["pool"], 0)
    assert not np.array_equal(raw(got), raw(want))


def test_unsupported_geometry_raises():
    pool = jnp.zeros((4, 64, 1, 2, WIDTH), jnp.bfloat16)
    with pytest.raises(ValueError, match="paired main slot"):
        wa.check_supported(pool, [QSALayerCache(main_slot=0)])
    with pytest.raises(ValueError, match="carriers"):
        wa.check_supported(jnp.zeros((4, 64, 1, 4, WIDTH), jnp.bfloat16), [QSALayerCache()])


def test_flag_registered_default_off(monkeypatch):
    monkeypatch.delenv("VLLM_TPU_MK_QSA_WRITE_ALL", raising=False)
    assert "VLLM_TPU_MK_QSA_WRITE_ALL" in envs.environment_variables
    assert envs.environment_variables["VLLM_TPU_MK_QSA_WRITE_ALL"]() is False
    monkeypatch.setenv("VLLM_TPU_MK_QSA_WRITE_ALL", "1")
    assert envs.environment_variables["VLLM_TPU_MK_QSA_WRITE_ALL"]() is True


def test_import_origin():
    """Provenance guard: with Q38_EXPECT_ROOT set, both modules come from under it."""
    root = os.environ.get("Q38_EXPECT_ROOT", "")
    files = (os.path.realpath(wa.__file__), os.path.realpath(wc.__file__), os.path.realpath(envs.__file__))
    print("Q38_KVW_IMPORT_ORIGIN " + " ".join(files))
    if root:
        root = os.path.realpath(root)
        assert all(f.startswith(root + os.sep) for f in files), (root, files)
