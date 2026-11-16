# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_QSA_PREFILL_COMPARE_CORE: QSA prefill attention over 16-row union groups (qsa_compare_core).

Steps are packed multi-request prefills as in ``test_qwen4_exp_union_walk_prefill.py``: requests at unaligned
boundaries, each with its own cached context and its own pages of a permuted FP8 pool, so a wrong page row, a
wrong union group's codes or a misaligned request changes the output.

Checks: the device tables against a brute-force construction (padded rows schedule nothing: no row bit, no slot,
no tile), the SparseCore code math (``compare_code_lanes``) against the XLA reference, the kernel against a
float64 reference, and the ``_prefill_attention`` dispatch (flag on: the core within the served kernel's error;
its cap and short-context fallbacks bit for bit the flag-off call; flag off: unchanged), and the stats counters
(an extra output of an exported step, no host callback). Off TPU the Pallas kernels run in interpret mode.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as bsp
from vllm_torchtpu.kernels.qwen4_exp import qsa_compare_core as cc
from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa
from vllm_torchtpu.kernels.qwen4_exp import qsa_union_sc_gather as scg
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles

RATIO, TOPK, PAGE = 4, 64, 512
# Three requests at unaligned boundaries (aligned to 16 rows they take 144 of the bucket's 160 rows); 32
# trailing bucket rows of length 0.
ROWS, CACHED, TOKENS = (40, 72, 16), (900, 1700, 600), 160


@pytest.fixture
def kernels(monkeypatch):
    if jax.default_backend() != "tpu":
        try:
            from jax._src import tpu_info
        except ImportError:
            from jax._src.pallas.mosaic import tpu_info

        monkeypatch.setitem(
            tpu_info.registry,
            jax.devices()[0].device_kind,
            lambda: tpu_info._get_tpu_info_impl(tpu_info.ChipVersion.TPU_7X, 1),
        )
        original = pl.pallas_call

        def interpreted(*args, **kwargs):
            kwargs["interpret"] = True
            return original(*args, **kwargs)

        monkeypatch.setattr(bsp.pl, "pallas_call", interpreted)
    for name in ("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP",
                 "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH", "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES",
                 "VLLM_TPU_QSA_PREFILL_UNION_WALK", "VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT",
                 "VLLM_TPU_QSA_PREFILL_UNION_SC_BITSET", "VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS"):
        monkeypatch.delenv(name, raising=False)
    yield
    jax.clear_caches()


def _packed_step(seed, rows=ROWS, cached=CACHED, tokens=TOKENS):
    rng = np.random.default_rng(seed)
    pages_per_seq = max(-(-(c + n) // PAGE) for n, c in zip(rows, cached))
    pool_pages = pages_per_seq * len(rows)
    starts = np.concatenate([[0], np.cumsum(rows)]).astype(np.int32)
    owner = rng.permutation(pool_pages)
    lengths = np.zeros(tokens, np.int32)
    blocks = np.full((tokens, TOPK), -1, np.int32)
    table = np.zeros((tokens, pages_per_seq), np.int32)
    for r, (n, ctx) in enumerate(zip(rows, cached)):
        pages = owner[r * pages_per_seq:(r + 1) * pages_per_seq]
        for i in range(n):
            t = starts[r] + i
            length = ctx + i + 1
            lengths[t] = length
            table[t] = pages
            complete = length // RATIO
            local = list(range(max(0, complete - 8), complete))
            rest = rng.permutation(complete)[: TOPK - len(local)]
            pick = np.unique(np.concatenate([local, rest]))[:TOPK]
            blocks[t, : len(pick)] = rng.permutation(pick)
    q = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, (tokens, 24, 256)), jnp.bfloat16)
    raw = rng.integers(0, 0x40, size=(pool_pages, PAGE, 1, 4, 256), dtype=np.uint8)
    raw |= rng.integers(0, 2, size=raw.shape, dtype=np.uint8) << 7
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    return q, gate, jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table), pool, jnp.asarray(starts)


def _aligned(step):
    """The step in the 16-row aligned layout (each request from a 16-row boundary, padding of length 0, at least
    the bucket's rows) and the aligned row of each original row."""
    q, gate, blocks, lengths, table, pool, starts = step
    starts = np.asarray(starts)
    real = np.diff(starts)
    offsets = np.concatenate([[0], np.cumsum(-(-real // 16) * 16)])
    size = max(int(offsets[-1]), -(-len(np.asarray(lengths)) // 16) * 16)  # the bucket's trailing rows stay
    where = np.concatenate([offsets[r] + np.arange(n) for r, n in enumerate(real)])

    def put(x, fill):
        x = np.asarray(x)
        out = np.full((size, *x.shape[1:]), fill, x.dtype)
        out[where] = x[: starts[-1]]
        return out

    return (put(q.astype(jnp.float32), 0), put(blocks, -1), put(lengths, 0), put(table, 0)), where


def _visible(blocks, length):
    """Tokens a row of ``length`` attends: its selected groups and its partial tail group, below its length."""
    groups = {int(g) for g in blocks if g >= 0}
    if length % RATIO:
        groups.add(length // RATIO)
    return sorted(t for g in groups for t in range(g * RATIO, g * RATIO + RATIO) if t < length)


def _f64(q, blocks, lengths, table, pool):
    """Ungated float64 attention [T, 24, 256] of every live row."""
    q64 = np.asarray(q, np.float64)
    flat = np.asarray(pool.astype(jnp.float32)).astype(np.float64).reshape(-1, 4, 256)
    out = np.zeros(q64.shape)
    for t in range(len(lengths)):
        if lengths[t] <= 0:
            continue
        toks = np.asarray(_visible(blocks[t], int(lengths[t])))
        kv = flat[table[t][toks // PAGE] * PAGE + toks % PAGE]
        for h in range(24):
            k, v = kv[:, 2 * (h // 12)], kv[:, 2 * (h // 12) + 1]
            s = k @ q64[t, h] / 16.0
            p = np.exp(s - s.max())
            out[t, h] = (p / p.sum()) @ v
    return out


def _max_rel(got, want):
    """Compare's numerics measure: the largest absolute error over the reference's mean magnitude."""
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return np.abs(got - want).max() / np.abs(want).mean()


def _bits(x):
    return np.asarray(x).view(np.uint16)


def test_tables_match_brute_force_and_padding_schedules_nothing(kernels):
    step = _packed_step(0)
    (_, blocks, lengths, table), _ = _aligned(step)
    groups_cap = table.shape[1] * PAGE // RATIO
    cap = 512
    rows, codes, usize, ok = cc.compare_tables(jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
                                               groups_cap=groups_cap, union_cap=cap, page_size=PAGE)
    rows, codes, usize = np.asarray(rows), np.asarray(codes), np.asarray(usize)
    assert bool(ok)
    for q in range(len(lengths) // 16):
        rr = slice(16 * q, 16 * q + 16)
        sel = [set(int(g) for g in blocks[t] if g >= 0) | ({int(lengths[t]) // RATIO} if lengths[t] % RATIO else set())
               if lengths[t] > 0 else set() for t in range(rr.start, rr.stop)]
        union = sorted(set().union(*sel))
        assert usize[q] == len(union)
        p0 = int(lengths[16 * q]) - 1
        for s, g in enumerate(union):
            code = sum(1 << r for r in range(16) if g in sel[r])
            edge = min(max(RATIO * g - p0, -cc.EDGE_BIAS), cc.EDGE_MAX) + cc.EDGE_BIAS
            assert codes[q, s] == code | (edge << 16), (q, s)
            tok = g * RATIO
            assert rows[q, s] == table[16 * q][tok // PAGE] * PAGE + tok % PAGE
        assert not codes[q, len(union):].any()
        # Padding rows never own a bit.
        for r in range(16):
            if lengths[16 * q + r] == 0:
                assert not (codes[q] & (1 << r)).any()
    # All-padding union groups: no slot and no tile.
    dead = np.all(lengths.reshape(-1, 16) == 0, axis=1)
    assert dead.any() and not usize[dead].any()
    ntiles = np.where(usize > 0, -(-usize // cc.GPT), 0)
    assert not ntiles[dead].any()
    # The SparseCore code math on the same lists.
    words = -(-groups_cap // 32)
    words = -(-words // 8) * 8
    bits = np.asarray(cc.group_bits(jnp.asarray(blocks), jnp.asarray(lengths), groups_cap=groups_cap, words=words))
    bits = bits.reshape(-1, words * 16)
    ids, _ = cc.union_lists(jnp.asarray(bits.reshape(-1, words, 16)), union_cap=cap)
    ids = np.asarray(ids)
    for q in range(len(ids)):
        p0 = int(lengths[16 * q]) - 1
        lanes = scg.compare_code_lanes(jnp.asarray(ids[q]), jnp.int32(p0),
                                       lambda idx, b=bits[q]: jnp.asarray(b)[idx], RATIO)
        np.testing.assert_array_equal(np.asarray(lanes), codes[q])


def test_kernel_against_float64(kernels):
    step = _packed_step(1)
    (q, blocks, lengths, table), _ = _aligned(step)
    pool = step[5]
    groups_cap = table.shape[1] * PAGE // RATIO
    rows, codes, usize, ok = cc.compare_tables(jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
                                               groups_cap=groups_cap, union_cap=512, page_size=PAGE)
    assert bool(ok) and int(np.asarray(usize).max()) > 2 * cc.GPT  # unions of several tiles
    got = cc.compare_core_kernel(jnp.asarray(q, jnp.bfloat16), rows, codes, usize, pool, union_cap=512, depth=4)
    live = lengths > 0
    got = np.asarray(got.astype(jnp.float32))[live]
    assert np.isfinite(got).all()
    want = _f64(q, blocks, lengths, table, pool)[live]
    # The bf16 output alone is off the float64 reference by its rounding; the core may add as much again.
    floor = _max_rel(want.astype(np.float32).astype(jnp.bfloat16).astype(np.float32), want)
    assert _max_rel(got, want) <= 2 * floor, (_max_rel(got, want), floor)


def _dispatch(step):
    q, gate, blocks, lengths, table, pool, starts = step
    out, cache = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO)
    assert cache is pool
    return np.asarray(out)


def _served(step):
    q, gate, blocks, lengths, table, pool, starts = step
    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(q, pool, kv_heads=2, ratio=RATIO,
                                                                with_query_tile=True)
    out, _ = bsp.block_sparse_gqa_attention(
        q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO, query_block=query_block,
        kv_tile=kv_tile, **({"query_tile": query_tile} if query_tile else {}))
    return np.asarray(out)


def _spy(monkeypatch):
    calls = []
    original = cc.compare_core_kernel

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(cc, "compare_core_kernel", spy)
    return calls


def test_dispatch_flag_on_off_and_fallbacks(kernels, monkeypatch):
    rows, cached, tokens = (40, 200, 120, 72), (900, 1700, 600, 2300), 1024
    step = _packed_step(2, rows, cached, tokens)
    served = _served(step)
    calls = _spy(monkeypatch)
    # Flag off: the served call, no compare core.
    np.testing.assert_array_equal(_bits(_dispatch(step)), _bits(served))
    assert calls == []
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "1")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES", "reference")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH", "4")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "512")
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "1000")
    jax.clear_caches()
    got = _dispatch(step)
    assert calls and calls[0]["union_cap"] == 512 and calls[0]["depth"] == 4
    q, gate, blocks, lengths, table, pool, starts = step
    live = np.asarray(lengths) > 0
    assert not got[~live].astype(np.float32).any()  # rows of length 0 stay zero
    want = _f64(np.asarray(q.astype(jnp.float32)), np.asarray(blocks), np.asarray(lengths), np.asarray(table), pool)
    want = want / (1.0 + np.exp(-np.asarray(gate, np.float64)))
    got_rel = _max_rel(got[live].astype(np.float32), want[live])
    served_rel = _max_rel(served[live].astype(np.float32), want[live])
    assert got_rel <= 2 * served_rel, (got_rel, served_rel)
    assert not np.array_equal(_bits(got), _bits(served))  # the core ran
    # A union past the cap and a step below the minimum context: the served call, bit for bit.
    for env, value in (("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "128"),
                       ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", str(int(np.asarray(lengths).max()) + 1))):
        monkeypatch.setenv(env, value)
        jax.clear_caches()
        np.testing.assert_array_equal(_bits(_dispatch(step)), _bits(served), err_msg=env)


def test_unwritten_nan_bytes_do_not_reach_the_output(kernels):
    """Bytes past each request's last row in its pages are NaN (0x7F); rows never see them (causal edge, masked
    scores by select, values guarded), so the output stays finite and within the bf16 floor."""
    # Every request's last length leaves its tail group partial (941, 1770, 619), so the NaN bytes sit in a group
    # the last rows' unions load.
    step = _packed_step(4, rows=(41, 70, 19))
    (q, blocks, lengths, table), _ = _aligned(step)
    raw = np.asarray(jax.lax.bitcast_convert_type(step[5], jnp.uint8)).copy()
    starts = np.asarray(step[6])
    tab = np.asarray(step[4])
    for r in range(len(starts) - 1):
        last = int(np.asarray(step[3])[starts[r + 1] - 1])            # tokens written for request r
        page = tab[starts[r]][last // PAGE]
        raw[page, last % PAGE:] = 0x7F
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    groups_cap = table.shape[1] * PAGE // RATIO
    rows, codes, usize, ok = cc.compare_tables(jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
                                               groups_cap=groups_cap, union_cap=512, page_size=PAGE)
    got = cc.compare_core_kernel(jnp.asarray(q, jnp.bfloat16), rows, codes, usize, pool, union_cap=512, depth=4)
    live = lengths > 0
    got = np.asarray(got.astype(jnp.float32))[live]
    assert np.isfinite(got).all()
    want = _f64(q, blocks, lengths, table, pool)[live]
    floor = _max_rel(want.astype(np.float32).astype(jnp.bfloat16).astype(np.float32), want)
    assert _max_rel(got, want) <= 2 * floor, (_max_rel(got, want), floor)


def test_stats_counters_leave_the_exported_step_as_an_output(kernels, monkeypatch):
    """VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS: ``with_compare_stats`` returns the step's counters (union groups,
    tiles, largest union of the aligned layout's tables, the core's error against the flag-off call) from a
    function ``jax.export`` takes, and the attention output is the stats-off output bit for bit."""
    rows, cached, tokens = (40, 200, 120, 72), (900, 1700, 600, 2300), 1024
    step = _packed_step(2, rows, cached, tokens)
    for env, value in (("VLLM_TPU_QSA_PREFILL_COMPARE_CORE", "1"), ("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_TABLES", "reference"),
                       ("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_DEPTH", "4"), ("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "512"),
                       ("VLLM_TPU_QSA_PREFILL_UNION_MIN_CONTEXT", "1000")):
        monkeypatch.setenv(env, value)
    jax.clear_caches()
    plain = _dispatch(step)
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_STATS", "30")
    jax.clear_caches()

    def attend(q, gate, blocks, lengths, table, pool, starts):
        out, _ = qsa_rpa._prefill_attention(q, gate, blocks, lengths, table, pool, starts, kv_heads=2, ratio=RATIO)
        return (out,)

    exported = jax.export.export(jax.jit(cc.with_compare_stats(attend)), platforms=[jax.default_backend()])(*step)
    out, stats = exported.call(*step)
    assert cc.STATS_SINK == []
    np.testing.assert_array_equal(_bits(np.asarray(out)), _bits(plain))
    (_, blocks, lengths, table), _ = _aligned(step)
    _, _, usize, ok = cc.compare_tables(jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
                                        groups_cap=table.shape[1] * PAGE // RATIO, union_cap=512, page_size=PAGE)
    usize = np.asarray(usize)
    assert bool(ok)
    stats = np.asarray(stats)
    which = int(np.sum(int(np.asarray(step[3]).max()) >= np.asarray(cc.STATS_EDGES)))
    assert not np.delete(stats, which, axis=0).any()
    calls, core, groups, tiles, rows_, union_max, max_rel, nonfinite = stats[which]
    assert (calls, core, rows_, nonfinite) == (1, 1, tokens, 0)
    assert groups == np.sum(usize > 0)
    assert tiles == np.sum(-(-usize[usize > 0] // cc.GPT))
    assert union_max == usize.max()
    assert 0 < max_rel < 0.1  # one bf16 output ulp against the served kernel
    # A union past the cap: counted as a call, not as the core.
    monkeypatch.setenv("VLLM_TPU_QSA_PREFILL_COMPARE_CORE_CAP", "128")
    jax.clear_caches()
    _, stats = jax.jit(cc.with_compare_stats(attend))(*step)
    assert np.asarray(stats)[which, :2].tolist() == [1, 0] and not np.asarray(stats)[which, 2:4].any()


def test_dominant_scores_of_another_row_stay_out_of_a_rows_max(kernels):
    """One row of every union group (its first live row) also selects 16 old groups of its request that no other
    row selects, whose keys every row's query scores ~190 above its own keys. Each row's softmax max is over the
    keys it sees: a max over the union group's raw scores (masked slots included) would underflow every other
    row's visible probabilities (exp(-190) is below f32's smallest normal) and collapse those rows."""
    step = _packed_step(5)
    (q, blocks, lengths, table), _ = _aligned(step)
    rng = np.random.default_rng(5)
    q = np.asarray(jnp.asarray(q + 0.25, jnp.bfloat16).astype(jnp.float32))
    raw = np.asarray(jax.lax.bitcast_convert_type(step[5], jnp.uint8)).copy()
    owners, dominant, picks = [], {}, {}
    for g in range(len(lengths) // cc.T):
        rows = [r for r in range(g * cc.T, (g + 1) * cc.T) if lengths[r] > 0]
        if len(rows) < 2:
            continue
        # One set a request (its pages), from groups every row of the request sees; no other row selects it.
        req = int(table[rows[0], 0])
        if req not in picks:
            picks[req] = rng.choice(int(lengths[rows].min()) // RATIO - 8, 16, replace=False)
        pick = picks[req]
        owner = rows[0]
        own = [x for x in blocks[owner] if x >= 0 and x not in pick][: TOPK - 16]
        blocks[owner] = -1
        blocks[owner, : 16 + len(own)] = np.concatenate([pick, own])
        for r in rows[1:]:
            keep = [x for x in blocks[r] if x >= 0 and x not in pick]
            blocks[r] = -1
            blocks[r, : len(keep)] = keep
        for x in pick:
            for tok in range(x * RATIO, x * RATIO + RATIO):
                raw[table[owner][tok // PAGE], tok % PAGE, 0, (0, 2)] = 0x64      # K of both heads: 48.0
        owners.append(owner)
        dominant[g] = pick
    pool = jax.lax.bitcast_convert_type(jnp.asarray(raw), jnp.float8_e4m3fn)
    # The case is sharp: the other rows' raw scores on the owner's dominant keys sit far above their own.
    flat = np.asarray(pool.astype(jnp.float32)).reshape(-1, 4, 256)
    gaps = []
    for g, pick in dominant.items():
        for r in range(g * cc.T + 1, (g + 1) * cc.T):
            if lengths[r] <= 0:
                continue
            seen = np.asarray(_visible(blocks[r], int(lengths[r])))
            hidden = np.asarray([tok for x in pick for tok in range(x * RATIO, x * RATIO + RATIO)])
            k_seen = flat[table[r][seen // PAGE] * PAGE + seen % PAGE][:, 0]
            k_hidden = flat[table[r][hidden // PAGE] * PAGE + hidden % PAGE][:, 0]
            gaps.append((k_hidden @ q[r, 0]).max() / 16 - (k_seen @ q[r, 0]).max() / 16)
    assert len(owners) >= 8 and min(gaps) > 120, (len(owners), min(gaps))
    groups_cap = table.shape[1] * PAGE // RATIO
    rows, codes, usize, ok = cc.compare_tables(jnp.asarray(blocks), jnp.asarray(lengths), jnp.asarray(table),
                                               groups_cap=groups_cap, union_cap=512, page_size=PAGE)
    assert bool(ok)
    got = cc.compare_core_kernel(jnp.asarray(q, jnp.bfloat16), rows, codes, usize, pool, union_cap=512, depth=4)
    live = lengths > 0
    got = np.asarray(got.astype(jnp.float32))[live]
    assert np.isfinite(got).all()
    want = _f64(q, blocks, lengths, table, pool)[live]
    floor = _max_rel(want.astype(np.float32).astype(jnp.bfloat16).astype(np.float32), want)
    assert _max_rel(got, want) <= 2 * floor, (_max_rel(got, want), floor)
