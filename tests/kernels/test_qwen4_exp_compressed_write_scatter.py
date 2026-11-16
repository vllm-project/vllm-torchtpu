# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_COMPRESSED_WRITE: decode-sized steps under compressed DCP write
their compressed keys with write_compressed's per-pair kernel at every row count
(``scatter``) or with compressed_write.write_pairs (``inflight``), and leave the
same pool bytes as the served choice (run writer from
VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS rows, tiled writer on multiples of
128, per-pair kernel below).

CPU (Pallas interpret mode): the writer alone over multi-step decode sequences
with DCP4 ownership, MTP verify rows, rejected drafts, padded rows and idle
request slots; two groups completing in one request window; padded rows write
nothing; which kernel each step takes.

TPU (real kernels): the production pooled_qsa_compressed_dcp_select_step for 8
simulated ranks (TP8 x DCP4, DCP rank = TP rank % 4) over 33 steps per bucket on
the served pool layout; pool bytes and each step's local selection (vals, pos)
compared bit for bit after every step; then each writer's time per owned row at
96 to 512 rows.
"""

import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import dcp_select, pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.qsa import _token_metadata



def on_tpu():
    # Checked inside tests: a backend touched at collection claims the TPU.
    return jax.default_backend() == "tpu"


def tpu_only():
    if not on_tpu():
        pytest.skip("TPU only")
RATIO, DCP, SLOTS, INDEX_DIM, WIDTH = 4, 4, 128, 128, 256
COMPRESSED_SLOT = 3
SERVED_MIN_TOKENS = 128

# rows -> (bucket rows, active requests): 4 verify rows a request, the rest padding.
BUCKETS = {
    96: (96, 24),
    128: (128, 30),
    192: (192, 46),
    256: (256, 64),
    384: (384, 93),
}


@pytest.fixture
def interpret(monkeypatch):
    if on_tpu():
        return
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


PATHS = ("served", "scatter", "inflight")


def set_path(monkeypatch, path):
    """``served``: the served flags (run writer from 128 rows); else the flag's value."""
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", True)
    monkeypatch.setattr(
        envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE_MIN_TOKENS", SERVED_MIN_TOKENS
    )
    monkeypatch.setattr(
        envs, "VLLM_TPU_QSA_COMPRESSED_WRITE", "" if path == "served" else path
    )


def raw(x):
    return np.asarray(jax.lax.bitcast_convert_type(x, jnp.uint8))


class Decode:
    """A decode batch on ``SLOTS`` request rows: ``active`` requests on random
    slots (the rest idle), DCP4 tables of ``entry = DCP * page`` tokens per
    column, contexts placed just below owner / entry boundaries so the steps
    cross them, and MTP verify steps of ``qlen`` rows whose drafts are partly
    rejected (the next context advances by 1..qlen)."""

    def __init__(self, rng, active, page, columns, max_context=None, growth=256):
        self.rng, self.page, self.entry = rng, page, DCP * page
        self.slot_ids = np.sort(rng.choice(SLOTS, active, replace=False))
        top = max_context or (columns - 1) * self.entry
        base = rng.integers(self.entry, top, active)
        near = rng.random(active) < 0.5
        # Half the requests start 1..40 tokens below an owner (page) boundary.
        base = np.where(near, (base // page + 1) * page - rng.integers(1, 40, active), base)
        self.context = np.zeros(SLOTS, np.int64)
        self.context[self.slot_ids] = base
        # Distinct pages for the columns a request reaches within ``growth``
        # tokens; the other columns point at the null page 0.
        self.table = np.zeros((SLOTS, columns), np.int32)
        pages = iter(range(1, 1 << 30))
        for s in self.slot_ids:
            for c in range(min(columns, (self.context[s] + growth) // self.entry + 1)):
                self.table[s, c] = next(pages)
        self.pages_used = next(pages)

    def step(self, bucket, qlen=4, long_rows=None):
        """Operands of one step over ``bucket`` rows; ``long_rows`` maps slot ->
        query length for requests that send more than ``qlen`` rows."""
        lens = np.zeros(SLOTS, np.int64)
        lens[self.slot_ids] = qlen
        for s, n in (long_rows or {}).items():
            lens[s] = n
        starts = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        assert starts[-1] <= bucket, (starts[-1], bucket)
        positions = np.zeros(bucket, np.int32)
        for s in self.slot_ids:
            positions[starts[s] : starts[s + 1]] = self.context[s] + np.arange(lens[s])
        out = dict(
            starts=jnp.asarray(starts),
            context=jnp.asarray(self.context, jnp.int32),
            positions=jnp.asarray(positions),
        )
        return out, lens

    def advance(self, lens):
        accepted = np.minimum(self.rng.integers(1, 5, SLOTS), np.maximum(lens, 1))
        self.context = np.where(lens > 0, self.context + accepted, self.context)


def writer_operands(rng, decode, bucket, **step_kw):
    meta, lens = decode.step(bucket, **step_kw)
    active = meta["starts"][1:] > meta["starts"][:-1]
    slots = jnp.where(active, jnp.arange(SLOTS, dtype=jnp.int32), -1)
    req, logical, valid = _token_metadata(bucket, meta["starts"], meta["context"], slots)
    complete = valid & (logical % RATIO == RATIO - 1)
    keys = jnp.asarray(rng.normal(0, 1, (bucket, INDEX_DIM)), jnp.bfloat16)
    args = (keys, complete, req, logical, meta["starts"], meta["context"],
            jnp.asarray(decode.table))
    return args, lens, valid


@functools.lru_cache(maxsize=None)
def _jitted_writer(path):
    # One jit per path: the env flags are read at trace time.
    def write(pool, keys, complete, req, logical, starts, context, table, rank):
        return pooled_qsa.write_compressed(
            pool, keys, complete, req, logical, starts, context, table, RATIO,
            COMPRESSED_SLOT, dcp=(rank, DCP), decode=True,
        )

    return jax.jit(write)


def write(monkeypatch, path, pool, args, rank):
    set_path(monkeypatch, path)
    return _jitted_writer(path)(pool, *args, jnp.asarray([rank], jnp.int32))


def small_pool(rng, pages, page):
    return jnp.asarray(rng.normal(0, 1, (pages, page, 1, 4, WIDTH)), jnp.float8_e4m3fn)


# CPU pages are 64 rows (16-token owner blocks of pairs, 256-token table entries).
CPU_PAGE, CPU_COLUMNS = 64, 8


@pytest.mark.parametrize("rows", sorted(BUCKETS))
def test_scatter_matches_served_writer_over_decode_steps(interpret, monkeypatch, rows):
    bucket, active = BUCKETS[rows]
    rng = np.random.default_rng(rows)
    decode = Decode(rng, active, CPU_PAGE, CPU_COLUMNS)
    pools = {}
    initial = small_pool(rng, decode.pages_used, CPU_PAGE)
    for rank in range(DCP):
        pools[rank] = dict.fromkeys(PATHS, initial)
    written = 0
    for _ in range(6):
        args, lens, _ = writer_operands(rng, decode, bucket)
        for rank in range(DCP):
            before = pools[rank]["served"]
            for path in PATHS:
                pools[rank][path] = write(monkeypatch, path, pools[rank][path], args, rank)
            for path in PATHS[1:]:
                np.testing.assert_array_equal(
                    raw(pools[rank][path]), raw(pools[rank]["served"]), err_msg=path
                )
            written += int(np.any(raw(pools[rank]["served"]) != raw(before), axis=(-3, -2, -1)).sum())
        decode.advance(lens)
    assert written > 0


def test_two_groups_in_one_window(interpret, monkeypatch):
    # Eight-row requests: rows completing groups (2k, 2k+1) fill a whole pair
    # in one write; rows completing (2k+1, 2k+2) write two partial pairs.
    rng = np.random.default_rng(7)
    decode = Decode(rng, 40, CPU_PAGE, CPU_COLUMNS)
    s0, s1 = decode.slot_ids[:2]
    decode.context[s0] = 8 * 37  # groups 74, 75: one full pair
    decode.context[s1] = 8 * 41 + 4  # groups 83, 84: two partial pairs
    args, _, _ = writer_operands(rng, decode, 192, long_rows={s0: 8, s1: 8})
    pool = small_pool(rng, decode.pages_used, CPU_PAGE)
    for rank in range(DCP):
        served = write(monkeypatch, "served", pool, args, rank)
        for path in PATHS[1:]:
            got = write(monkeypatch, path, pool, args, rank)
            np.testing.assert_array_equal(raw(got), raw(served), err_msg=path)
    # The full pair is written whole on its owner.
    keys, complete, req, logical, starts, context, table = args
    pair, wr, mask, _ = pooled_qsa.compressed_pairs(
        pool, keys, complete, req, logical, starts, context, table, RATIO,
        COMPRESSED_SLOT, dcp=(jnp.asarray([((8 * 37) // CPU_PAGE) % DCP], jnp.int32), DCP),
    )
    rows0 = np.arange(int(starts[s0]), int(starts[s0 + 1]))
    full = np.asarray(wr)[rows0] & np.asarray(mask).all(axis=1)[rows0]
    assert full.sum() == 1


def padded_batch():
    rng = np.random.default_rng(11)
    decode = Decode(rng, 46, CPU_PAGE, CPU_COLUMNS)
    args, lens, valid = writer_operands(rng, decode, 192)
    real = int(lens.sum())
    assert real == 184 and not np.asarray(valid)[real:].any()
    return rng, decode, args, real


def test_padded_rows_schedule_no_writes():
    # The per-pair kernel's grid covers every row; a row writes only where
    # ``write`` is set, and padded rows never set it on any rank.
    rng, decode, args, real = padded_batch()
    pool = small_pool(rng, decode.pages_used, CPU_PAGE)
    keys, complete, req, logical, starts, context, table = args
    assert not np.asarray(complete)[real:].any()
    owned = 0
    for rank in range(DCP):
        _, wr, mask, _ = pooled_qsa.compressed_pairs(
            pool, keys, complete, req, logical, starts, context, table, RATIO,
            COMPRESSED_SLOT, dcp=(jnp.asarray([rank], jnp.int32), DCP),
        )
        assert not np.asarray(wr)[real:].any()
        owned += int(np.asarray(wr).sum())
    # Every real request completes one group a step and exactly one rank owns it.
    assert owned == 46


def test_padded_rows_leave_the_pool_alone(interpret, monkeypatch):
    rng, decode, args, real = padded_batch()
    pool = small_pool(rng, decode.pages_used, CPU_PAGE)
    keys, complete, req, logical, starts, context, table = args
    for rank in range(DCP):
        _, wr, _, _ = pooled_qsa.compressed_pairs(
            pool, keys, complete, req, logical, starts, context, table, RATIO,
            COMPRESSED_SLOT, dcp=(jnp.asarray([rank], jnp.int32), DCP),
        )
        # Garbage keys in the padded rows leave the same pool as the batch
        # without them, and only the rank's writers change a pair row.
        poisoned = (keys.at[real:].set(jnp.bfloat16(1e4)), *args[1:])
        cut = (keys[:real], complete[:real], req[:real], logical[:real], starts, context,
               table)
        for path in PATHS[1:]:
            padded = write(monkeypatch, path, pool, poisoned, rank)
            unpadded = write(monkeypatch, path, pool, cut, rank)
            np.testing.assert_array_equal(raw(padded), raw(unpadded), err_msg=path)
            changed = np.any(raw(padded) != raw(pool), axis=(-3, -2, -1)).sum()
            assert changed == int(np.asarray(wr).sum()), path


def kernels_traced(monkeypatch, path, *, rows, dcp=True, decode=True, run_write=True):
    """Pallas kernels one write_compressed call traces."""
    set_path(monkeypatch, path)
    monkeypatch.setattr(envs, "VLLM_TPU_QSA_COMPRESSED_RUN_WRITE", run_write)
    rng = np.random.default_rng(rows)
    state = Decode(rng, rows // 4, CPU_PAGE, CPU_COLUMNS)
    args, _, _ = writer_operands(rng, state, rows)
    pool = small_pool(rng, state.pages_used, CPU_PAGE)
    kw = {"dcp": (jnp.asarray([0], jnp.int32), DCP)} if dcp else {}
    jaxpr = jax.make_jaxpr(
        lambda p, *a: pooled_qsa.write_compressed(
            p, *a, RATIO, COMPRESSED_SLOT, decode=decode, **kw)
    )(pool, *args)
    kinds = {
        "qwen4_exp_compressed_kv_run_write": "run",
        "qwen4_exp_compressed_kv_page_tiled_scatter": "tiled",
        "qwen4_exp_compressed_kv_pair_write": "inflight",
        # The per-pair kernel is unnamed, one program per row.
        None: "per_pair",
    }
    found = set()
    for eqn in jaxpr.jaxpr.eqns:
        if eqn.primitive.name == "pallas_call":
            kind = kinds[eqn.params["name"]]
            if kind == "per_pair":
                assert eqn.params["grid_mapping"].grid == (rows,)
            found.add(kind)
    return found


def test_kernel_choice(interpret, monkeypatch):
    # Served flags: per-pair below 128 rows, run writer from 128.
    assert kernels_traced(monkeypatch, "served", rows=96) == {"per_pair"}
    assert kernels_traced(monkeypatch, "served", rows=256) == {"run"}
    assert kernels_traced(monkeypatch, "served", rows=256, run_write=False) == {"tiled"}
    # scatter: per-pair at every decode-sized row count under compressed DCP,
    # with or without the run writer flag.
    for rows in (96, 128, 192, 256, 384, 512):
        assert kernels_traced(monkeypatch, "scatter", rows=rows) == {"per_pair"}
        assert kernels_traced(
            monkeypatch, "scatter", rows=rows, run_write=False
        ) == {"per_pair"}
    for rows in (96, 192, 256, 512):
        assert kernels_traced(monkeypatch, "inflight", rows=rows) == {"inflight"}
    # Prefill-sized steps and replicated keys keep the served choice.
    assert kernels_traced(monkeypatch, "inflight", rows=256, decode=False) == {"run"}
    assert kernels_traced(monkeypatch, "inflight", rows=256, dcp=False) == {"run"}
    assert kernels_traced(monkeypatch, "scatter", rows=256, decode=False) == {"run"}
    assert kernels_traced(monkeypatch, "scatter", rows=256, dcp=False) == {"run"}
    assert kernels_traced(
        monkeypatch, "scatter", rows=256, dcp=False, run_write=False
    ) == {"tiled"}


# ---------------------------------------------------------------------------
# TPU: the production select step, 8 ranks, served pool layout.

PAGE = 512
H, DH, INDEX_HEADS, ROTARY, TOKEN_TOPK = 3, 256, 4, 64, 2048
SERVED_COLUMNS = 128  # max_model_len 262144 / (DCP * 512)
TPU_STEPS = 33


def _normal(seed, shape, scale=1.0):
    return jax.random.normal(jax.random.key(seed), shape, jnp.float32).astype(jnp.bfloat16) * scale


@pytest.mark.parametrize("case", ["96", "128", "192", "256", "384", "192_two_groups"])
def test_tpu_select_step_bit_exact(monkeypatch, case):
    tpu_only()
    rows = int(case.split("_")[0])
    bucket, active = BUCKETS[rows]
    two_groups = case.endswith("two_groups")
    if two_groups:
        active = 44  # 176 rows, plus 4 more for each of two 8-row requests
    rng = np.random.default_rng(1000 + rows + two_groups)
    decode = Decode(rng, active, PAGE, SERVED_COLUMNS, max_context=16 * 2048,
                    growth=8 * TPU_STEPS)
    long_rows = None
    max_query_len = 4
    if two_groups:
        s0, s1 = decode.slot_ids[:2]
        decode.context[s0] = (decode.context[s0] // 8) * 8  # groups 2k, 2k+1: a full pair
        decode.context[s1] = (decode.context[s1] // 8) * 8 + 4  # 2k+1, 2k+2: two partial
        long_rows = {s0: 8, s1: 8}
        max_query_len = 8
    # The plan (contexts, rows) of every step, shared by all ranks and both paths.
    plan = []
    for step in range(TPU_STEPS):
        meta, lens = decode.step(bucket, long_rows=long_rows if step % 4 == 0 else None)
        plan.append(meta)
        decode.advance(lens)
    comp_table = decode.table
    used = comp_table > 0
    main_table = np.where(used, comp_table + decode.pages_used, 0).astype(np.int32)
    ring_ids = np.zeros(SLOTS, np.int32)
    ring_ids[decode.slot_ids] = 2 * decode.pages_used + np.arange(active)
    pages = 2 * decode.pages_used + active + 1
    nq, nk = _normal(1, (DH,), 0.1), _normal(2, (DH,), 0.1)
    niq, nik = _normal(3, (INDEX_DIM,), 0.1), _normal(4, (INDEX_DIM,), 0.1)
    trig = jnp.pad(_normal(5, (65536, ROTARY)), ((0, 0), (0, 128 - ROTARY)))
    tables = (jnp.asarray(ring_ids), jnp.asarray(main_table), jnp.asarray(comp_table))
    comp_pages = jnp.asarray(np.unique(comp_table[used]))
    opts = dict(compress_ratio=RATIO, token_topk=TOKEN_TOPK, norm_eps=1e-6,
                rotary_dim=ROTARY, dcp_size=DCP, compressed_slot=COMPRESSED_SLOT,
                main_slot=1, max_query_len=max_query_len)
    steps = {}

    def run(path, *args):
        set_path(monkeypatch, path)
        if path not in steps:
            steps[path] = jax.jit(functools.partial(
                pooled_qsa.pooled_qsa_compressed_dcp_select_step, **opts))
        return steps[path](*args)

    def u8(x):
        return jax.lax.bitcast_convert_type(x, jnp.uint8)

    written = np.zeros(8, np.int64)
    selected = np.zeros(8, np.int64)
    for r in range(8):
        # Ranks r and r + 4 share a DCP rank and differ in their KV / q heads.
        group = r // DCP
        init = jax.random.normal(
            jax.random.key(rows * 16 + r), (pages, PAGE, 1, 4, WIDTH), jnp.float32
        ).astype(jnp.float8_e4m3fn)
        pools = dict.fromkeys(PATHS, init)
        del init
        for step, meta in enumerate(plan):
            seed = 100000 * step
            args = (
                _normal(seed + 10 + group, (bucket, DCP * 2 * H * DH)),
                _normal(seed + 20 + group, (bucket, DH)),
                _normal(seed + 30 + group, (bucket, DH)),
                _normal(seed + 40, (bucket, (INDEX_HEADS + 1) * INDEX_DIM)),
                nq, nk, niq, nik, meta["positions"], trig,
            )
            tail = (meta["starts"], meta["context"], *tables,
                    jnp.asarray([r % DCP], jnp.int32))
            before = pools["served"]
            outs = {}
            for path in PATHS:
                outs[path] = run(path, *args, pools[path], *tail)
                pools[path] = outs[path][0]
            a = outs["served"]
            for path in PATHS[1:]:
                b = outs[path]
                assert bool(jnp.array_equal(u8(a[0]), u8(b[0]))), (path, case, step, r)
                np.testing.assert_array_equal(
                    np.asarray(jax.lax.bitcast_convert_type(b[6], jnp.uint32)),
                    np.asarray(jax.lax.bitcast_convert_type(a[6], jnp.uint32)),
                    err_msg=f"vals {path} {case} step {step} rank {r}")
                np.testing.assert_array_equal(np.asarray(b[7]), np.asarray(a[7]),
                                              err_msg=f"pos {path} {case} step {step} rank {r}")
            written[r] += int(jnp.sum(jnp.any(
                u8(a[0][comp_pages]) != u8(before[comp_pages]), axis=(-3, -2, -1))))
            selected[r] += int(jnp.sum((a[7] >= 0) & (a[7] != dcp_select.EMPTY)))
            del before, outs, a, b
        del pools
    print(f"F11_EXACT case={case} bucket={bucket} active={active} steps={TPU_STEPS} "
          f"ranks=8 paths={'/'.join(PATHS)} pages={pages} compressed_rows_changed={written.tolist()} "
          f"selected_entries={selected.tolist()} OK")
    assert (written > 0).all() and (selected > 0).all()


def test_tpu_per_pair_time(monkeypatch):
    """Time per write_compressed call (prep + kernel) inside one program, per
    rank and path; prints us per call, per owned row (rows / 4) and per pair
    this rank writes."""
    tpu_only()
    iters = 64
    for rows in (96, 128, 192, 256, 384, 512):
        active = rows // 4
        rng = np.random.default_rng(rows)
        decode = Decode(rng, active, PAGE, 48)
        pool0 = jax.random.normal(
            jax.random.key(0), (decode.pages_used, PAGE, 1, 4, WIDTH), jnp.float32
        ).astype(jnp.float8_e4m3fn)
        args, _, _ = writer_operands(rng, decode, rows)
        line = {}
        for path in PATHS:
            set_path(monkeypatch, path)

            def loop(pool, *a, path=path):
                def body(i, carry):
                    pool, i_in = carry
                    # Keeps the writer's preparation inside the loop.
                    inputs = jax.lax.optimization_barrier((a, i_in))
                    out = pooled_qsa.write_compressed(
                        pool, *inputs[0][:7], RATIO, COMPRESSED_SLOT,
                        dcp=(inputs[0][7], DCP), decode=True)
                    return out, i_in + 1

                return jax.lax.fori_loop(0, iters, body, (pool, jnp.int32(0)))[0]

            fn = jax.jit(loop, donate_argnums=0)
            per_rank = []
            for rank in range(DCP):
                rank_arr = jnp.asarray([rank], jnp.int32)
                pool = jnp.copy(pool0)
                pool = fn(pool, *args, rank_arr)
                pool.block_until_ready()
                samples = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    pool = fn(pool, *args, rank_arr)
                    pool.block_until_ready()
                    samples.append((time.perf_counter() - t0) / iters * 1e6)
                _, wr, _, _ = pooled_qsa.compressed_pairs(
                    pool0, *args, RATIO, COMPRESSED_SLOT, dcp=(rank_arr, DCP))
                per_rank.append((float(np.median(samples)), int(np.asarray(wr).sum())))
            us = np.mean([p[0] for p in per_rank])
            pairs = np.mean([p[1] for p in per_rank])
            line[path] = (us, pairs)
            print(f"F11_TIME rows={rows} path={path} us_per_call={us:.2f} "
                  f"us_per_owned_row={us / (rows / 4):.3f} "
                  f"pairs_per_rank={pairs:.1f} us_per_pair={us / max(pairs, 1):.3f} "
                  f"per_rank={[(round(u, 2), w) for u, w in per_rank]}")
        assert all(line[path][0] > 0 for path in PATHS)
