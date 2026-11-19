# SPDX-License-Identifier: Apache-2.0
# q38 PORT-BL UNION: from Benliang Wang <benliangw@google.com>, commit 2ce02bb1a on
# benliangw/q38fh-integ-pin-5b44702aea, unchanged except: the dumped-selection case (his scratch data, absent here)
# is dropped.
"""Union page lists and row-mask words (``qsa_union_meta``) against the reference.

CPU: the kernel's own per-request code (``request_body``) runs on numpy
stand-ins for the SparseCore primitives, with the kernel's staging (garbage in
every scratch buffer, buffers reused across a worker's requests, rows past a
request's count left stale) and must equal ``union_reference`` request by
request: the same pages as a set with the final partial page last, the same
word per page, kv_len and per-row token counts, zeros past U. The stand-ins
check that every access stays inside its buffer and that no two iterations of a
``parallel_loop`` touch one address when either writes (the loop's contract).
Padded requests and dead rows must schedule nothing. The glue
(``union_glue``) is checked on the emulated outputs. A lowering test traces the
kernel to Mosaic for an abstract TPU7x device. The SparseCore compile and the
primitives' hardware semantics are checked on TPU by
``scripts/bench_union_meta.py``.
"""

import functools
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import qsa_union_meta as um

LANES = um.LANES
GARBAGE = -77777
DUMP = Path(
    os.environ.get("F9U_DATA", Path(__file__).resolve().parents[2] / "scratch/f9_union")
)


class _NumpyOps:
    """``um.ScOps`` on numpy buffers.

    Masked-off lanes of a gather read ``GARBAGE``; a compressed store writes
    ``GARBAGE`` on its unselected slots. With ``races`` set, every access inside
    a loop is recorded per iteration and checked at the loop's end.
    """

    xp = np

    def __init__(self, races=True):
        self.races = races
        self.trace = None

    @staticmethod
    def lane():
        return np.arange(LANES, dtype=np.int32)

    @staticmethod
    def full(x):
        return np.full((LANES,), x, np.int32)

    @staticmethod
    def scalar(ref, at):
        return int(ref[at])

    def _note(self, kind, ref, addrs):
        if self.trace is not None:
            self.trace[-1][kind].update((id(ref), int(a)) for a in addrs)

    @staticmethod
    def _range(ref, at, n=LANES):
        at = int(at)
        assert 0 <= at and at + n <= ref.size, (at, n, ref.size)
        return at

    def load(self, ref, at):
        at = self._range(ref, at)
        self._note("r", ref, range(at, at + LANES))
        return ref[at : at + LANES].copy()

    def store(self, ref, at, x):
        at = self._range(ref, at)
        self._note("w", ref, range(at, at + LANES))
        ref[at : at + LANES] = x

    def store_compressed(self, ref, at, x, mask):
        at = self._range(ref, at)
        keep = np.asarray(mask, bool)
        n = int(keep.sum())
        self._note("w", ref, range(at, at + n))
        ref[at : at + n] = np.asarray(x)[keep]
        ref[at + n : at + LANES] = GARBAGE

    def _indices(self, ref, index, mask):
        index = np.broadcast_to(np.asarray(index), (LANES,))
        mask = np.ones(LANES, bool) if mask is None else np.asarray(mask, bool)
        live = index[mask]
        assert (live >= 0).all() and (live < ref.size).all(), live
        return index, mask, live

    def gather(self, ref, index, mask):
        index, mask, live = self._indices(ref, index, mask)
        self._note("r", ref, live)
        out = np.full(LANES, GARBAGE, np.int32)
        out[mask] = ref[live]
        return out

    def store_scatter(self, ref, index, x, mask):
        index, mask, live = self._indices(ref, index, mask)
        assert len(set(live.tolist())) == len(live), "a scatter's lanes collide"
        self._note("w", ref, live)
        ref[live] = np.broadcast_to(np.asarray(x), (LANES,))[mask]

    def addupdate_scatter(self, ref, index, x, mask):
        index, mask, live = self._indices(ref, index, mask)
        assert len(set(live.tolist())) == len(live), "a scatter's lanes collide"
        self._note("w", ref, live)
        ref[live] += np.broadcast_to(np.asarray(x), (LANES,))[mask]

    @staticmethod
    def cumsum(x):
        return np.cumsum(x, dtype=np.int32)

    @staticmethod
    def popcount(mask):
        return int(np.asarray(mask, bool).sum())

    def loop(self, lower, upper, *, carry=None, unroll=1):
        def run(body):
            assert self.trace is None, "nested loops"
            if self.races:
                self.trace = []
            state = carry
            for i in range(int(lower), int(upper)):
                if self.races:
                    self.trace.append({"r": set(), "w": set()})
                state = body(i) if carry is None else body(i, state)
            if self.races:
                writer = {}
                for it, acc in enumerate(self.trace):
                    for a in acc["w"]:
                        assert writer.setdefault(a, it) == it, (
                            f"iterations {writer[a]} and {it} write {a}"
                        )
                for it, acc in enumerate(self.trace):
                    for a in acc["r"]:
                        assert writer.get(a, it) == it, (
                            f"iteration {it} reads {a} written by {writer[a]}"
                        )
            self.trace = None
            return state

        return run


def emulate(
    blocks,
    lengths,
    starts,
    table,
    rank,
    *,
    pages=2080,
    own_list=False,
    clear="scatter",
    insert="store",
    workers=8,
    races=True,
    seed=0,
):
    """The kernel's staging and ``request_body`` on numpy: (pages, words, meta)."""
    rng = np.random.default_rng(seed)
    header, tab = (
        np.asarray(x)
        for x in um.stage_inputs(
            jnp.asarray(lengths),
            jnp.asarray(starts),
            jnp.asarray(table),
            jnp.asarray([rank], jnp.int32),
        )
    )
    reqs, cols16 = tab.shape
    cols = np.asarray(table).shape[1]
    pos_words = -(-cols * um.PAGE_GROUPS // LANES) * LANES
    blocks = np.asarray(blocks, np.int32)
    ops = _NumpyOps(races)
    out_pages = np.full((reqs, pages), GARBAGE, np.int32)
    out_words = np.full((reqs, pages), GARBAGE, np.int32)
    out_meta = np.full((reqs, um.META), GARBAGE, np.int32)
    for w in range(workers):

        def junk(n):
            return rng.integers(-5, 3000, n).astype(np.int32)

        rows_v, head_v, tab_v = (
            junk(um.MAX_ROWS * um.TOPK),
            junk(2 * LANES),
            junk(cols16),
        )
        pos_v, pages_v, words_v, meta_v = (
            junk(pos_words),
            junk(pages + 32),
            junk(pages + 32),
            junk(um.META),
        )
        for r in range(w, reqs, workers):
            head_v[: um.META] = header[r]
            tab_v[:] = tab[r]
            start, n = int(header[r, um.H_START]), int(header[r, um.H_ROWS])
            for j in range(n):
                rows_v[j * um.TOPK : (j + 1) * um.TOPK] = blocks[start + j]
            um.request_body(
                ops,
                rows_v,
                head_v,
                tab_v,
                pos_v,
                pages_v,
                words_v,
                meta_v,
                pages=pages,
                pos_words=pos_words,
                world=4,
                own_list=own_list,
                clear=clear,
                insert=insert,
            )
            out_pages[r], out_words[r], out_meta[r] = (
                pages_v[:pages],
                words_v[:pages],
                meta_v,
            )
    return out_pages, out_words, out_meta


def random_batch(rng, *, reqs, own_rank=None, max_ctx=250_000, padded=3):
    """Requests of 1 to 4 consecutive-position rows (some dead), then padded
    requests and rows. ``own_rank``: rows in F10's own-list form for that rank."""
    starts, lengths, blocks = [0], [], []
    for _ in range(reqs):
        n = int(rng.choice([1, 2, 3, 4, 4, 4]))
        base = int(
            rng.choice(
                [1, 2, 3, 5, 30, 700, 2047, 2049, 5000, 66_000]
                + [int(rng.integers(1, max_ctx))] * 4
            )
        )
        for j in range(n):
            length = 0 if rng.random() < 0.06 else base + j
            complete = length // 4
            if complete <= um.TOPK:
                row = rng.permutation(complete)
                # Slots past the count are junk the served rule ignores.
                row = np.concatenate(
                    (row, rng.integers(-1, max(complete, 1), um.TOPK - complete))
                )
            else:
                recent = np.arange(max(0, complete - 64), complete)
                rest = rng.choice(
                    complete - len(recent), um.TOPK - len(recent), replace=False
                )
                row = rng.permutation(np.concatenate((recent, rest)))
            if own_rank is not None:
                owned = (row >= 0) & (row < complete) & ((row // 128) % 4 == own_rank)
                if complete <= um.TOPK:
                    owned &= np.arange(um.TOPK) < complete
                row = rng.permutation(np.where(owned, row, -1))
            if rng.random() < 0.03:
                row = np.full(um.TOPK, -1)
            blocks.append(row)
            lengths.append(length)
        starts.append(len(lengths))
    for _ in range(padded):
        starts.append(len(lengths))
    rows = len(lengths) + 5
    blocks += [np.zeros(um.TOPK, np.int64)] * 5
    lengths += [0] * 5
    cols = max(lengths) // 2048 + 2
    table = np.zeros((rows, cols), np.int32)
    for r in range(len(starts) - 1):
        table[starts[r] : starts[r + 1]] = rng.choice(100_000, cols, replace=False)
    return (
        np.stack(blocks).astype(np.int32),
        np.asarray(lengths, np.int32),
        np.asarray(starts, np.int32),
        table,
    )


def check_all(blocks, lengths, starts, table, rank, out, own_list=False):
    ref = um.union_reference(blocks, lengths, starts, table, rank, own_list=own_list)
    pages, words, meta = out
    bad = []
    for r, entry in enumerate(ref):
        for msg in um.union_mismatches(pages[r], words[r], meta[r], entry):
            bad.append((r, msg))
    return bad, ref


@pytest.mark.parametrize(
    "clear,insert", [("scatter", "store"), ("full", "store"), ("scatter", "add")]
)
def test_emulation_is_the_reference(clear, insert):
    rng = np.random.default_rng(11)
    for trial in range(3):
        batch = random_batch(rng, reqs=int(rng.integers(5, 14)))
        for rank in range(4):
            out = emulate(
                *batch, rank, clear=clear, insert=insert, workers=4, seed=trial
            )
            bad, ref = check_all(*batch, rank, out)
            assert not bad, bad[:5]
            starts = batch[2]
            for r in range(len(starts) - 1):
                if starts[r + 1] == starts[r]:  # padded request: nothing scheduled
                    assert (
                        not out[2][r].any()
                        and not out[0][r].any()
                        and not out[1][r].any()
                    )


def test_emulation_own_list_form():
    rng = np.random.default_rng(12)
    for rank in range(4):
        batch = random_batch(rng, reqs=10, own_rank=rank)
        out = emulate(*batch, rank, own_list=True, workers=3, seed=rank)
        bad, _ = check_all(*batch, rank, out, own_list=True)
        assert not bad, bad[:5]


def test_glue_padded_rows_schedule_nothing():
    rng = np.random.default_rng(13)
    blocks, lengths, starts, table = random_batch(rng, reqs=9, padded=4)
    rank = 1
    out = emulate(blocks, lengths, starts, table, rank, races=False)
    ref = um.union_reference(blocks, lengths, starts, table, rank)
    pages, words, meta = out
    kv_lens, cu_q_lens, row_live = (
        np.asarray(x)
        for x in um.union_glue(jnp.asarray(meta), jnp.asarray(starts), len(lengths))
    )
    sel = np.asarray(um.widen_words(jnp.asarray(words)))
    # The kernel's own expansion (emit_sel) on the emulated words.
    ops = _NumpyOps()
    for r in range(len(starts) - 1):
        words_v = np.concatenate((words[r], np.full(32, GARBAGE, np.int32)))
        sel_v = np.full(sel.shape[1], GARBAGE, np.int32)
        um.expand_sel(ops, words_v, sel_v, sel.shape[1] // LANES)
        np.testing.assert_array_equal(sel_v, sel[r])
    nrows = np.minimum(np.diff(starts), 4)
    want_q = [n if len(e[0]) else 0 for n, e in zip(nrows, ref)]
    np.testing.assert_array_equal(cu_q_lens, np.concatenate(([0], np.cumsum(want_q))))
    np.testing.assert_array_equal(kv_lens, [e[2] for e in ref])
    live = np.zeros(len(lengths), bool)
    for r, e in enumerate(ref):
        for j in range(nrows[r]):
            live[starts[r] + j] = e[4][j] > 0
    np.testing.assert_array_equal(row_live, live)
    assert not row_live[starts[-1] :].any()
    for r, e in enumerate(ref):
        u = len(e[0])
        # Per token, in the kernel's order: the reference word of that page's token.
        word_of = {p: e[1][4 * k : 4 * k + 4] for k, p in enumerate(e[0])}
        got = sel[r].reshape(-1, 4)
        for k in range(u):
            assert list(got[k]) == word_of[int(pages[r, k])]
        assert not sel[r, 4 * u :].any()
        if u:
            assert not sel[r, kv_lens[r] :].any()  # unwritten tokens of the final page
    pad = [r for r in range(len(starts) - 1) if starts[r + 1] == starts[r]]
    assert pad and all(kv_lens[r] == 0 for r in pad)


@pytest.fixture
def tpu7x_abstract():
    from jax.sharding import AbstractDevice, AbstractMesh, AxisType, use_abstract_mesh

    device = AbstractDevice(device_kind="TPU7x", num_cores=1, platform="tpu")
    mesh = AbstractMesh((1,), ("x",), (AxisType.Explicit,), abstract_device=device)
    with use_abstract_mesh(mesh):
        yield
    jax.clear_caches()


@pytest.mark.parametrize(
    "reqs,rows,kw",
    [
        (24, 128, {}),
        (32, 128, {"clear": "full"}),
        (100, 512, {"insert": "add", "own_list": True}),
    ],
)
def test_kernel_lowers_to_mosaic_for_tpu7x(tpu7x_abstract, reqs, rows, kw):
    spec = jax.ShapeDtypeStruct
    fn = jax.jit(functools.partial(um.union_page_metadata, **kw))
    lowered = fn.trace(
        spec((rows, 512), jnp.int32),
        spec((rows,), jnp.int32),
        spec((reqs + 1,), jnp.int32),
        spec((rows, 122), jnp.int32),
        spec((1,), jnp.int32),
    ).lower(lowering_platforms=("tpu",))
    assert "tpu_custom_call" in lowered.as_text()
