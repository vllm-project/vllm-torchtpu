# SPDX-License-Identifier: Apache-2.0
"""Port of MM's D12 fetch-mode test (q38mm DSTOCK tests/q38mm_dstock/test_rpad_fetch.py at 92c1f3fbb) for the PD port of
`fast` alone (DEC-O4-4, q38-deco44-rpadfast): VLLM_TPU_QSA_DECODE_RPA_FETCH=fast must copy exactly the same cache tokens to
exactly the same VMEM slots as today's per-page compact fetch (mode ""), so the attention is bitwise identical. MM's other
modes (coalesce / fastrun / pages / runs / loop) are not ported and their cases are dropped. PD-shape cases added at the end:
the DCP-4 virtual page table (513 columns per row), SLOT_READ (kv_slot), 4 lanes x 128 pages (bkv 512), the qsa_rpa pass-through
at RPA_MAX_QUERIES=128 (a 256-row step is two 128-row calls), and full-tile / in-range row-path coverage.

Batched RPA does not run in any CPU interpreter, so this runs the kernel's own fetch code -- the real
`KVBufferedRefHeadAlongSublane.copy_in` -- with pallas' DMA / when / ds / multiple_of (and lax.fori_loop for loop mode)
replaced by recorders, at the served MM shape (page 4 tokens, 4 lanes, 128 pages per lane-tile = bkv 512, compact
schedule). The table modes get their tables from the real `kv_tables.build`. Page lists: real-like (runs of
consecutive ids), runs crossing a physical-page boundary, single-page runs, empty lane-tiles, a partial last page, the
pipeline's no-op step, and clamping past the last page-table entry.
Run: PYTHONPATH=src python -m pytest -q -p no:cacheprovider --noconftest tests/q38_deco44_rpadfast/test_rpad_fetch_fast.py
"""
import types

import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.batched_rpa import bref_override as bo
from vllm_torchtpu.kernels.experimental.batched_rpa import configs as rpa_configs
from vllm_torchtpu.kernels.experimental.batched_rpa import kernel as rpa_kernel

PAGE, LANES, PAGES = 4, 4, 128
MODES = ["fast"]
ROW_SLICES = []        # page-table row slices taken by the last _run (fast mode's full-tile path)


class _Ref:
    def __init__(self, name, shape=(1, 1), key=None):
        self.name, self.shape, self.key = name, shape, key

    def reshape(self, *shape):
        return _Ref(self.name + "_flat", tuple(shape))

    @property
    def at(self):
        ref = self

        class _At:
            def __getitem__(self, key):
                return _Ref(ref.name, ref.shape, key)
        return _At()


class _Table:
    """The page table as the kernel sees it: scalar reads and row slices (`.at[pl.ds(start, size)]`)."""

    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def __getitem__(self, i):
        return int(self.arr[int(i)])

    @property
    def at(self):
        arr = self.arr

        class _At:
            def __getitem__(self, key):
                _, start, size = key
                assert start + size <= len(arr), "row slice past the page table"
                ROW_SLICES.append((start, size))
                return _Table(arr[start:start + size])
        return _At()


def _loop(lo, hi, body, init):
    carry = init
    for r in range(int(lo), int(hi)):
        carry = body(r, carry)
    return carry


def _run(monkeypatch, page_indices, tokens_per_lane, p0_per_lane, mode, no_op=False, kv_slot=None):
    dmas = []
    ROW_SLICES.clear()

    class _Copy:
        def __init__(self, src, dst, sem):
            self.src, self.dst = src, dst

        def start(self):
            dmas.append((self.src, self.dst))

    monkeypatch.setattr(bo.pltpu, "make_async_copy", _Copy)
    monkeypatch.setattr(bo.pl, "ds", lambda start, size: ("ds", int(start), int(size)))
    monkeypatch.setattr(bo.pl, "when", lambda c: (lambda f: f() if bool(c) else None))
    monkeypatch.setattr(bo.pl, "multiple_of", lambda x, m: x)
    monkeypatch.setattr(bo.jax.lax, "fori_loop", _loop)
    sched = types.SimpleNamespace(dma_kv_cache=np.zeros((1, LANES, 1, 3), np.int32))
    for b in range(LANES):
        sched.dma_kv_cache[0, b, 0, 0] = p0_per_lane[b]
        sched.dma_kv_cache[0, b, 0, 2] = tokens_per_lane[b]
    entry = types.SimpleNamespace(fetch_hbm=np.array(0), fetch_vmem=np.array(0), fetch_val=0)
    sched.dma_kv_new = type("N", (), {"__getitem__": lambda self, k: entry})()
    serve = types.SimpleNamespace(lean_kv_fetch=True, compact_kv_schedule=True, kv_fetch_mode=mode,
                                  fast_kv_fetch=True, page_size=PAGE, num_page_indices=len(page_indices),
                                  kv_slot=kv_slot)
    fake = types.SimpleNamespace(
        src_ref=(_Ref("kv", (len(page_indices) + 8, PAGE, 1, 4, 256)), _Ref("new"), sched, _Table(page_indices)),
        sem_recvs=_Ref("sem"), window_ref=_Ref("vmem"),
        cfgs=types.SimpleNamespace(serve=serve, batch_size=LANES, bkv_p_cache=PAGES, bkv_p_new=0, kv_hbm_stride=8))
    bo.KVBufferedRefHeadAlongSublane.copy_in(fake, 1 if no_op else 0, 0, 1)
    return [(("ref", s.name, s.key), ("ref", d.name, d.key)) for s, d in dmas if s.name == "kv_flat"]


def _token_map(dmas):
    """{(lane, vmem slot): cache token}: what the fetch delivers, token by token (0-size DMAs deliver nothing)."""
    out = {}
    for src, dst in dmas:
        _, s0, n = src[2]
        b, (_, d0, n2) = dst[2][0], dst[2][1]
        assert n == n2
        for t in range(n):
            assert (b, d0 + t) not in out
            out[(b, d0 + t)] = s0 + t
    return out


def _lane_pages(rng, n_pages, kind):
    if kind == "runs":       # real-like: mostly singletons, some runs (QSA selections: 72% singletons at 102.4K)
        ids, v = [], int(rng.integers(0, 1000))
        while len(ids) < n_pages:
            for _ in range(int(rng.geometric(0.6))):
                ids.append(v)
                v += 1
            v += int(rng.integers(2, 40))
        return ids[:n_pages]
    if kind == "boundary":   # one long run crossing the physical page boundary 128 -> 129 x 128
        return list(range(128 * 129 - 7, 128 * 129 - 7 + n_pages))
    if kind == "single":     # no two consecutive ids
        return list(range(5, 5 + 3 * n_pages, 3))
    raise ValueError(kind)


CASES = [
    ("runs_full", "runs", [512, 512, 512, 512]),
    ("runs_partial_last", "runs", [512, 509, 4, 1]),
    ("runs_empty_lanes", "runs", [0, 512, 0, 6]),
    ("boundary", "boundary", [512, 300, 512, 128]),
    ("singletons", "single", [512, 512, 13, 2]),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("name,kind,tokens", CASES)
@pytest.mark.parametrize("seed", range(3))
def test_fetch_mode_delivers_identical_tokens(monkeypatch, mode, name, kind, tokens, seed):
    rng = np.random.default_rng(seed)
    page_indices, p0 = [], []
    for b in range(LANES):
        p0.append(len(page_indices))
        page_indices += _lane_pages(rng, PAGES, kind)
    off = _run(monkeypatch, page_indices, tokens, p0, "")
    on = _run(monkeypatch, page_indices, tokens, p0, mode)
    assert _token_map(on) == _token_map(off)
    assert sum(d[0][2][2] for d in on) == sum(tokens)
    assert len(off) == LANES * PAGES             # today: one descriptor per page entry, empty ones included
    # today's nonzero descriptors, in order; empty pages issue nothing; every nonempty tile takes the row path
    assert on == [d for d in off if d[0][2][2] > 0]
    assert ROW_SLICES == [(p0[b], PAGES) for b in range(LANES) if tokens[b] > 0]


@pytest.mark.parametrize("mode", MODES)
def test_noop_step_issues_nothing(monkeypatch, mode):
    rng = np.random.default_rng(0)
    pi = _lane_pages(rng, LANES * PAGES, "runs")
    on = _run(monkeypatch, pi, [512] * LANES, [0, PAGES, 2 * PAGES, 3 * PAGES], mode, no_op=True)
    assert on == []


@pytest.mark.parametrize("mode", MODES)
def test_clamped_tail_past_the_page_table(monkeypatch, mode):
    """The last lane-tile's p0 + i runs past the table: entries clamp to the last index, as the per-page path does."""
    pi = list(range(100, 100 + PAGES + 10))
    off = _run(monkeypatch, pi, [512, 512, 512, 40], [0, 0, 0, len(pi) - 5], "")
    on = _run(monkeypatch, pi, [512, 512, 512, 40], [0, 0, 0, len(pi) - 5], mode)
    assert _token_map(on) == _token_map(off)


@pytest.mark.parametrize("mode,suffix", [("", ""), ("fast", "-f")])
def test_kernel_name_marks_the_fetch_mode(mode, suffix):
    serve = types.SimpleNamespace(kv_layout=types.SimpleNamespace(symbol="snh"), page_size=4, kv_fetch_mode=mode)
    cfgs = types.SimpleNamespace(serve=serve, mode=types.SimpleNamespace(symbol="d"), batch_size=4, bq_sz=1,
                                 bkv_sz=512, model=types.SimpleNamespace(sliding_window=0))
    assert rpa_kernel.get_kernel_name(cfgs) == "RPAd-snh-p4-b4-q1-k512" + suffix


def test_flag_registered_and_compile_keyed():
    from vllm_torchtpu import envs
    assert "VLLM_TPU_QSA_DECODE_RPA_FETCH" in envs.environment_variables     # registered => in the compile factors
    assert "VLLM_TPU_QSA_DECODE_RPA_COALESCE" not in envs.environment_variables


def test_only_fast_is_ported():
    assert rpa_configs.KV_FETCH_MODES == {"": "", "fast": "-f"}
    assert not hasattr(rpa_configs, "KV_FETCH_TABLE_MODES")


# ---------------- PD-shape cases (DEC-O4-4) ----------------
COLS_DCP4 = 513   # the DCP-4 decode RPA virtual page table: 512 owned groups + the current one per row (C, qsa_rpa)


def _dcp4_layout(rng, rows, tokens_of_tile):
    """rows x 513-column table, real-like ids per row; lane-tiles of 128 pages start at row*513 + k*128 (k = 0..4,
    the last one a single column). Returns (page_indices, [(p0, tokens) per tile])."""
    pi, tiles = [], []
    for r in range(rows):
        pi += _lane_pages(rng, COLS_DCP4, "runs")
        for k in range(5):
            tiles.append((r * COLS_DCP4 + k * PAGES, tokens_of_tile(r, k)))
    return pi, tiles


@pytest.mark.parametrize("seed", range(3))
def test_pd_dcp4_513_column_table(monkeypatch, seed):
    """Every lane-tile of a 3-row DCP-4 table, 4 lanes at a time: the fast fetch delivers off's tokens; in-range nonempty
    tiles take the row path; the last row's 1-column tile (its row slice would leave the table) takes the per-page path."""
    rng = np.random.default_rng(seed)

    def tok(r, k):
        if k == 4:
            return 3 if r % 2 == 0 else 4          # the single-column tile: partial or one full page
        return [512, 509, 0, 128][(r + k + seed) % 4]
    pi, tiles = _dcp4_layout(rng, 3, tok)
    last = len(pi) - 1
    for g in range(0, len(tiles), LANES):
        grp = (tiles[g:g + LANES] + [(0, 0)] * LANES)[:LANES]
        p0, tokens = [t[0] for t in grp], [t[1] for t in grp]
        off = _run(monkeypatch, pi, tokens, p0, "")
        on = _run(monkeypatch, pi, tokens, p0, "fast")
        assert _token_map(on) == _token_map(off)
        # the same nonzero descriptors in order (an out-of-range tile keeps the per-page path, 0-size DMAs included)
        assert [d for d in on if d[0][2][2] > 0] == [d for d in off if d[0][2][2] > 0]
        oor = [b for b in range(LANES) if tokens[b] > 0 and p0[b] + PAGES - 1 > last]
        assert len([d for d in on if d[0][2][2] == 0]) == sum(PAGES - (-(-tokens[b] // PAGE)) for b in oor)
        want = [(p0[b], PAGES) for b in range(LANES) if tokens[b] > 0 and p0[b] + PAGES - 1 <= last]
        assert ROW_SLICES == want


def test_pd_full_tile_is_always_in_range(monkeypatch):
    """MM's claim, checked at PD's table width: a full tile (512 tokens) never starts past row*513 + 384, so its row slice
    stays inside the table (the recording table asserts any slice past its end)."""
    rng = np.random.default_rng(7)
    pi, tiles = _dcp4_layout(rng, 2, lambda r, k: 512 if k < 4 else 4)
    full = [t for t in tiles if t[1] == PAGES * PAGE]
    for g in range(0, len(full), LANES):
        grp = (full[g:g + LANES] + [(0, 0)] * LANES)[:LANES]
        on = _run(monkeypatch, pi, [t[1] for t in grp], [t[0] for t in grp], "fast")
        assert len(on) == sum(PAGES for t in grp if t[1])


@pytest.mark.parametrize("slot", [0, 1])
def test_pd_slot_read_does_not_change_the_fetch(monkeypatch, slot):
    """SLOT_READ (kv_slot) changes which K/V pair the kernel unpacks, not what it fetches (whole rows)."""
    rng = np.random.default_rng(3)
    pi, p0 = [], []
    for b in range(LANES):
        p0.append(len(pi))
        pi += _lane_pages(rng, PAGES, "runs")
    tokens = [512, 300, 0, 4]
    a = _run(monkeypatch, pi, tokens, p0, "fast")
    b = _run(monkeypatch, pi, tokens, p0, "fast", kv_slot=slot)
    assert a == b


@pytest.mark.parametrize("rows,compact,mode,want_rows,want_mode", [
    (256, True, "fast", 128, "fast"),   # RPA_MAX_QUERIES=128: two 128-row calls (one scan body), each with the mode
    (128, True, "fast", 128, "fast"),   # one call
    (128, False, "fast", None, ""),     # no compact schedule: the mode is not passed on
    (256, True, "", 128, ""),           # flag unset: per-page path
])
def test_pd_qsa_rpa_pass_through(monkeypatch, rows, compact, mode, want_rows, want_mode):
    import jax
    import jax.numpy as jnp

    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.experimental.batched_rpa import wrapper
    from vllm_torchtpu.kernels.qwen4_exp import qsa_rpa

    for k, v in {"VLLM_TPU_QSA_DECODE_RPA_MAX_QUERIES": 128, "VLLM_TPU_QSA_DECODE_RPA_BKV": 512,
                 "VLLM_TPU_QSA_DECODE_RPA_LANES": 4, "VLLM_TPU_QSA_DECODE_RPA_FETCH": mode}.items():
        monkeypatch.setattr(envs, k, v, raising=False)
    calls = []

    def fake_rpa(query, k, v, cache, *args, **kw):
        calls.append((query.shape[0], kw.get("kv_fetch_mode"), kw.get("compact_kv_schedule")))
        out = jnp.zeros(query.shape, jnp.float32)
        return (out, cache, jnp.zeros(query.shape[:2], jnp.float32)) if kw.get("return_lse") else (out, cache)

    monkeypatch.setattr(wrapper, "ragged_paged_attention", fake_rpa)
    H, D, cols, pool = 12, 256, 8, 16
    q = jax.ShapeDtypeStruct((rows, H, D), jnp.bfloat16)
    blocks = jax.ShapeDtypeStruct((rows, 512), jnp.int32)
    lengths = jax.ShapeDtypeStruct((rows,), jnp.int32)
    table = jax.ShapeDtypeStruct((rows, cols), jnp.int32)
    cache = jax.ShapeDtypeStruct((pool, 512, 1, 4, 256), jnp.float8_e4m3fn)

    def f(q, gate, blocks, lengths, table, cache):
        return qsa_rpa.rpa_attention(q, gate, blocks, lengths, table, cache, kv_heads=1, ratio=4, lean_fetch=True,
                                     fast_fetch=True, compact=compact, return_lse=True)
    jax.eval_shape(f, q, q, blocks, lengths, table, cache)
    assert calls, "the RPA wrapper was never reached"
    assert all(m == want_mode for _, m, _ in calls), calls
    if want_rows is not None:
        assert all(n == want_rows for n, _, _ in calls), calls


def test_pd_tile_ending_at_the_last_entry_takes_the_row_path(monkeypatch):
    """A partial tile whose row slice ends exactly at the table's last entry is in range: the row path, not the per-page
    fallback (pins the in-range boundary)."""
    rng = np.random.default_rng(11)
    pi = _lane_pages(rng, 3 * PAGES, "runs")
    last = len(pi) - 1
    p0 = [0, PAGES, last - PAGES + 1, 0]
    tokens = [512, 0, 37, 0]
    off = _run(monkeypatch, pi, tokens, p0, "")
    on = _run(monkeypatch, pi, tokens, p0, "fast")
    assert _token_map(on) == _token_map(off)
    assert on == [d for d in off if d[0][2][2] > 0]
    assert ROW_SLICES == [(0, PAGES), (last - PAGES + 1, PAGES)]
