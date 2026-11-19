# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever admitinc (layers/core/attention_metadata.py): ADMIT-INC and batched row writes.

Host-only; runs in a CDK CPU pod. On real vLLM BlockTables / MultiGroupBlockTables:

1. (i) ADMIT-INC: over random BlockTable mutation streams with request-count changes, padded-bucket
   changes, row-width changes, clear(), dummy runs and writes to rows outside the window, the staged
   upload equals a freshly built full pack byte for byte at every step, and count changes inside a
   bucket are actually packed incrementally;
2. (ii) batched row writes: over random add / append / clear / move / swap streams (empty lists,
   src == tgt, a hybrid-block group, error inputs) the lever-on MultiGroupBlockTable ends every call
   with the lever-off one's block tables, row counts, dirty marks and version-bump decision;
3. end to end through test_l6_steady.py's decode world (real runner walk, STEADY, BT-INC, FMB): the
   lever-on runner's metadata and staged tables equal a lever-off replica's at every step;
4. CHECK mode compares (both parts), catches a stale pack / a wrong batched write, keeps the stock
   result and turns the lever off; VLLM_TPU_L6_CHECK_STEPS bounds it;
5. lever off: no admit pack, no batched write; flag unset: vLLM's MultiGroupBlockTable untouched;
6. registration (l6_ab.LEVERS, envs, compile-key ignore list), the BlockTable source pin, and the
   import guard.
"""

import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.layers.core import attention_metadata as am

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_l6_steady as S  # noqa: E402  (the decode world of the STEADY proof)

CPU = torch.device("cpu")
MAX_REQS = 16
WIDTHS = [12, 4, 12, 7]


# ---------------------------------------------------------------- fixtures


def _counts():
    return dict(l6_ab._state["counts"])


@pytest.fixture
def lever(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)  # one window: counts accumulate
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    monkeypatch.setattr(am.logger, "info", lambda *a, **k: None)
    errors = []
    monkeypatch.setattr(am.logger, "error", lambda fmt, *a: errors.append(fmt % a))
    # On CPU an upload can alias the host scratch; copy, as a device transfer does.
    monkeypatch.setattr(am, "_inc_upload", lambda src, device: src.clone())
    monkeypatch.setattr(am, "_STAGE_INC", True)
    monkeypatch.setattr(am, "_STAGE_INC_CHECK", [0])
    monkeypatch.setattr(am, "_STAGE_INC_STATS", {"inc": 0, "full": 0, "rows": 0, "logged": False})
    monkeypatch.setattr(am, "_ADMIT_OFF", [False])
    monkeypatch.setattr(am, "_ADMIT_TURNS", {"pack": 0, "batch": 0})
    monkeypatch.setattr(am, "_ADMIT_STATS", {"inc": 0, "rows": 0, "zeroed": 0, "batch": 0, "logged": False})
    assert am._admit_batch_install(), "the batched row writes must install on this vLLM"

    def setup(value="1", check=0, check_steps=None, steady=False):
        if value:
            monkeypatch.setenv("VLLM_TPU_L6_ADMIT_INC", value)
        else:
            monkeypatch.delenv("VLLM_TPU_L6_ADMIT_INC", raising=False)
        if steady:
            monkeypatch.setenv("VLLM_TPU_L6_STEADY_STEP", "1")
        if check_steps is not None:
            monkeypatch.setenv("VLLM_TPU_L6_CHECK_STEPS", str(check_steps))
        l6_ab._resolve()
        monkeypatch.setattr(am, "_ADMIT_CHECK", [check])
        l6_ab.begin_step()
        return errors

    yield setup
    for k in ("VLLM_TPU_L6_ADMIT_INC", "VLLM_TPU_L6_STEADY_STEP", "VLLM_TPU_L6_CHECK_STEPS"):
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _arm(on):
    """Fix the lever's state for the calls that follow (two arms in one process)."""
    l6_ab._state["on"]["admitinc"] = on


# ---------------------------------------------------------------- 1. ADMIT-INC packs


def _tables(widths=WIDTHS):
    return [BlockTable(16, MAX_REQS, w, 64, False, CPU, 16, 1) for w in widths]


def _entries(bts, widths):
    return [(i, bt, w) for i, (bt, w) in enumerate(zip(bts, widths))]


def _reference(bts, widths, ctx, nrp):
    if ctx.position_ids_override is not None:
        return torch.zeros(nrp * sum(widths), dtype=torch.int32)  # dummy runs stage zeros
    parts = []
    for bt, w in zip(bts, widths):
        rows = torch.zeros((nrp, w), dtype=torch.int32)
        rows[: ctx.num_reqs] = bt.get_cpu_tensor()[ctx.start_index : ctx.start_index + ctx.num_reqs, :w]
        parts.append(rows.flatten())
    return torch.cat(parts)


def _pack(runner, bts, widths, ctx, nrp):
    entries = _entries(bts, widths)
    geom, off = [], 0
    for _, _, w in entries:
        geom.append((off, nrp, w))
        off += nrp * w
    key = (am._BLOCK_TABLE_VERSION[0], tuple(id(bt) for _, bt, _ in entries), ctx.start_index, ctx.num_reqs,
           tuple(geom))
    return am._pack_and_upload(runner, ctx, entries, nrp, geom, key)


def _ctx(start, n, dummy=False):
    return SimpleNamespace(position_ids_override=torch.zeros(1) if dummy else None, start_index=start, num_reqs=n)


def _random_op(rng, bts):
    bt = rng.choice(bts)
    w = bt.max_num_blocks_per_req
    op = rng.random()
    row = rng.randrange(MAX_REQS)  # any row: rows outside the window are written too
    if op < 0.45:
        room = w - int(bt.num_blocks_per_row[row])
        if room > 0:
            bt.append_row([rng.randrange(1, 9999) for _ in range(rng.randint(1, room))], row)
    elif op < 0.6:
        bt.add_row([rng.randrange(1, 9999) for _ in range(rng.randint(0, w))], row)
    elif op < 0.7:
        bt.clear_row(row)
    elif op < 0.8:
        bt.move_row(row, rng.randrange(MAX_REQS))
    elif op < 0.9:
        bt.swap_row(row, rng.randrange(MAX_REQS))
    elif op < 0.905:
        bt.clear()
    # else: no mutation


NARROW = [8, 4, 6, 7]  # a row-width change (another use_max_model_len / model-length bucket)


@pytest.mark.parametrize("reuse,start", [("1", 0), ("0", 0), ("1", 3)])
def test_admit_pack_equals_full_pack_over_random_streams(lever, monkeypatch, reuse, start):
    """Every step's upload equals a fresh full pack; count changes inside a bucket go incremental."""
    lever("1")
    monkeypatch.setenv("VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE", reuse)
    rng = random.Random(4321 + start + 7 * int(reuse))
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    cap = MAX_REQS - start
    n, nrp, widths = 5, 8, WIDTHS
    count_steps = admit_taken = 0
    for step in range(1500):
        for _ in range(rng.randint(0, 3)):
            _random_op(rng, bts)
        r = rng.random()
        dummy = False
        if r < 0.30:
            n = rng.randint(0, min(nrp, cap))  # count change inside the bucket
        elif r < 0.36:
            nrp = rng.choice([b for b in (8, 16) if b <= cap])  # bucket change (16 -> 8 keeps the scratch)
            n = rng.randint(0, min(nrp, cap))
        elif r < 0.38:
            widths = NARROW if widths is WIDTHS else WIDTHS  # row-width change, same bucket
        elif r < 0.40:
            dummy = True
        ctx = _ctx(start, n, dummy)
        before = am._ADMIT_STATS["inc"]
        state = runner.__dict__.get("_bt_inc_state")
        got = _pack(runner, bts, widths, ctx, nrp)
        assert torch.equal(got, _reference(bts, widths, ctx, nrp)), f"step {step}"
        if state is not None and state[0] is not None and not dummy and state[0][2] != n:
            count_steps += 1
            admit_taken += am._ADMIT_STATS["inc"] > before
    st, bst = am._ADMIT_STATS, am._STAGE_INC_STATS
    assert st["inc"] == admit_taken > 250, (st, count_steps)
    assert st["rows"] > 0 and st["zeroed"] > 0, st
    assert bst["inc"] > 300 and bst["full"] > 50, bst
    assert _counts().get("admitinc.inc") == st["inc"]


def test_count_changes_go_incremental_and_geometry_changes_full(lever):
    """The deterministic path table: which change takes which pack (and the bytes at each)."""
    lever("1")
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    for row in range(10):
        for bt in bts:
            bt.add_row([100 * row + j + 1 for j in range(3)], row)
    seen = []

    def pack(n, nrp=16, start=0, widths=WIDTHS, dummy=False):
        a0, f0 = am._ADMIT_STATS["inc"], am._STAGE_INC_STATS["full"]
        ctx = _ctx(start, n, dummy)
        got = _pack(runner, bts, widths, ctx, nrp)
        assert torch.equal(got, _reference(bts, widths, ctx, nrp))
        kind = "admit" if am._ADMIT_STATS["inc"] > a0 else "full" if am._STAGE_INC_STATS["full"] > f0 else "inc"
        seen.append(kind)
        return kind

    assert pack(6) == "full"  # first pack
    for bt in bts:
        bt.add_row([7, 8], 6)  # admission at row 6
    assert pack(7) == "admit"
    for bt in bts:
        bt.clear_row(2)
        bt.move_row(6, 2)  # finish row 2, condense
    assert pack(6) == "admit"
    assert pack(9) == "admit"  # rows 6..8 enter with no mark (their marks were cleared at n = 6)
    assert pack(9) == "inc"  # no change: BT-INC reuses
    assert pack(4) == "admit"  # shrink: zeroes rows 4..8
    assert pack(4, nrp=8) == "full"  # bucket change
    assert pack(5, nrp=8) == "admit"
    assert pack(6, nrp=8, widths=NARROW) == "full"  # width change
    assert pack(6, nrp=8, start=1, widths=NARROW) == "full"  # window start change
    assert pack(5, nrp=8, start=1, widths=NARROW, dummy=True) == "full"  # dummy run
    assert pack(4, nrp=8, start=1, widths=NARROW) == "full"  # after a dummy: no state
    bts[1].clear()
    assert pack(3, nrp=8, start=1, widths=NARROW) == "full"  # clear(): every row dirty
    assert pack(5, nrp=8, start=1, widths=NARROW) == "admit"


def test_lever_off_count_changes_are_full_packs(lever):
    """The gate: with the lever off (unset, 0, or an ab: class not current) no count change is admitted."""
    for value in ("", "0", "ab:1"):
        if value == "ab:1":
            os.environ["VLLM_TPU_L6_AB_STEPS"] = "1000"
        try:
            lever(value)
        finally:
            os.environ.pop("VLLM_TPU_L6_AB_STEPS", None)
        assert not am._admit_on()
        bts = _tables()
        runner = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
        f0 = am._STAGE_INC_STATS["full"]
        for n in (3, 4, 2, 5):
            for bt in bts:
                bt.add_row([n, n + 1], n)
            ctx = _ctx(0, n)
            assert torch.equal(_pack(runner, bts, WIDTHS, ctx, 8), _reference(bts, WIDTHS, ctx, 8))
        assert am._STAGE_INC_STATS["full"] == f0 + 4
        assert am._ADMIT_STATS["inc"] == 0 and am._ADMIT_STATS["batch"] == 0
        assert "admitinc.inc" not in _counts()


# ---------------------------------------------------------------- 2. batched row writes


def _mg(widths, hybrid=()):
    """A MultiGroupBlockTable over BlockTables of the given widths (kernel block 8 for `hybrid` groups)."""
    m = object.__new__(MultiGroupBlockTable)
    m.block_tables = [
        BlockTable(16, MAX_REQS, w, 64, False, CPU, 8 if i in hybrid else 16, 1) for i, w in enumerate(widths)
    ]
    return m


def _state_of(m):
    return [
        (bt.block_table.np.copy(), bt.num_blocks_per_row.copy(), set(bt.__dict__.get("_q38_dirty", ())),
         bt.__dict__.get("_q38_dirty_all", False))
        for bt in m.block_tables
    ]


def _same_state(a, b):
    for g, (x, y) in enumerate(zip(_state_of(a), _state_of(b))):
        assert np.array_equal(x[0], y[0]), ("rows", g)
        assert np.array_equal(x[1], y[1]), ("counts", g)
        assert x[2] == y[2], ("dirty", g, x[2], y[2])
        assert x[3] == y[3], ("dirty_all", g)


def _random_mg_op(rng, m):
    tables = m.block_tables
    row = rng.randrange(MAX_REQS)
    op = rng.random()
    if op < 0.25:
        ids = tuple(
            [] if rng.random() < 0.1 else [rng.randrange(1, 9999) for _ in range(rng.randint(1, bt.max_num_blocks_per_req // bt.blocks_per_kv_block))]
            for bt in tables
        )
        return "add_row", (ids, row)
    if op < 0.6:
        ids = []
        for bt in tables:
            room = (bt.max_num_blocks_per_req - int(bt.num_blocks_per_row[row])) // bt.blocks_per_kv_block
            ids.append([rng.randrange(1, 9999) for _ in range(rng.randint(1, room))] if room > 0 and rng.random() < 0.3 else [])
        return "append_row", (tuple(ids), row)
    if op < 0.75:
        return "clear_row", (row,)
    if op < 0.9:
        return "move_row", (row, rng.choice([row, rng.randrange(MAX_REQS)]))
    return "swap_row", (row, rng.choice([row, rng.randrange(MAX_REQS)]))


@pytest.mark.parametrize("stage_inc", [True, False])
@pytest.mark.parametrize("shape", ["mixed", "hybrid", "served65"])
def test_batched_row_writes_equal_vllms_loop(lever, monkeypatch, stage_inc, shape):
    """Lever-on writes leave every group exactly as vLLM's per-group loop does; the version moves iff it would."""
    lever("1")
    monkeypatch.setattr(am, "_STAGE_INC", stage_inc)
    widths, hybrid = {
        "mixed": (WIDTHS, ()),
        "hybrid": (WIDTHS, (1, 3)),
        "served65": ([32] * 39 + [8] * 13 + [4] * 13, ()),
    }[shape]
    a, b = _mg(widths, hybrid), _mg(widths, hybrid)
    rng = random.Random(99 + len(widths) + len(hybrid) + stage_inc)
    calls = 0
    for step in range(400 if shape != "served65" else 150):
        name, args = _random_mg_op(rng, b)
        v0 = am._BLOCK_TABLE_VERSION[0]
        _arm(False)
        getattr(b, name)(*args)
        vb = am._BLOCK_TABLE_VERSION[0]
        _arm(True)
        getattr(a, name)(*args)
        va = am._BLOCK_TABLE_VERSION[0]
        calls += 1
        assert (vb != v0) == (va != vb), (step, name)
        _same_state(a, b)
    assert am._ADMIT_STATS["batch"] == calls
    assert _counts().get("admitinc.batch") == calls


def test_batched_error_inputs_match_vllms_loop(lever):
    """Too few per-group lists, a list longer than the row, kwargs: same exception, same rows after."""
    lever("1")
    a, b = _mg(WIDTHS), _mg(WIDTHS)
    for m, on in ((b, False), (a, True)):
        _arm(on)
        m.add_row(([1, 2], [3], [4, 5, 6], [7]), 2)
    cases = [
        ("add_row", (([1, 2, 3], [4]), 5), IndexError),
        ("append_row", (([9], [8], [7]), 2), IndexError),
        ("add_row", ((list(range(1, 14)), [1], [2], [3]), 4), ValueError),
        ("append_row", (([1] * 11, [], [], []), 2), ValueError),
    ]
    for name, args, exc in cases:
        for m, on in ((b, False), (a, True)):
            _arm(on)
            with pytest.raises(exc):
                getattr(m, name)(*args)
        _same_state(a, b)
    batch0 = am._ADMIT_STATS["batch"]
    for m, on in ((b, False), (a, True)):
        _arm(on)
        m.add_row(block_ids=([5], [6], [7], [8]), row_idx=3)  # keyword call: vLLM's loop
    _same_state(a, b)
    assert am._ADMIT_STATS["batch"] == batch0


# ---------------------------------------------------------------- 3. end to end through the STEADY decode world


class MGWorld(S.World):
    """test_l6_steady's world with InputBatch's own call pattern: one MultiGroupBlockTable call per
    request (add_row on admission; clear_row then move_row on a finish; swap_row; append_row with
    one list per group), plus an occasional all-empty append_row."""

    def __init__(self):
        super().__init__()
        self.mg = object.__new__(MultiGroupBlockTable)
        self.mg.block_tables = self.tables

    def admit(self, length):
        row = len(self.reqs)
        nb = self._need(length)
        self.mg.add_row(tuple(self._blocks(nb) for _ in self.tables), row)
        self.reqs.append(dict(id=f"r{self.next_id}", s=length, h=length, nb=nb))
        self.next_id += 1

    def finish(self, row):
        last = len(self.reqs) - 1
        self.mg.clear_row(row)
        if row != last:
            self.mg.move_row(last, row)
        self.reqs[row] = self.reqs[last]
        self.reqs.pop()

    def swap(self, i, j):
        self.mg.swap_row(i, j)
        self.reqs[i], self.reqs[j] = self.reqs[j], self.reqs[i]

    def advance(self, accepted):
        for row, (req, a) in enumerate(zip(self.reqs, accepted)):
            s0 = req["s"]
            req["s"], req["h"] = s0 + 1 + a, s0 + 1 + S.K
            need = self._need(req["h"])
            if need > req["nb"]:
                self.mg.append_row(tuple(self._blocks(need - req["nb"]) for _ in self.tables), row)
                req["nb"] = need
            elif row % 3 == 0:
                self.mg.append_row(tuple([] for _ in self.tables), row)


@pytest.mark.parametrize("steady", [True, False])
def test_end_to_end_identity_through_the_steady_world(lever, monkeypatch, steady):
    """Real runner walk (served flags: BT-INC, upload reuse, entries cache, FMB; STEADY optional): the
    lever-on runner equals a lever-off replica at every step, with admissions and finishes mixed in."""
    lever("1", steady=steady)
    S._flags(monkeypatch, True)
    monkeypatch.setattr(am, "_STAGE_INC_STATS", {"inc": 0, "full": 0, "rows": 0, "logged": False})
    rng = random.Random(77 + steady)
    wa, wb = MGWorld(), MGWorld()
    ra, rb = S._runner(wa), S._runner(wb)
    ra.input_batch.block_table, rb.input_batch.block_table = wa.mg, wb.mg

    def both(fn, *args):
        _arm(True)
        getattr(wa, fn)(*args)
        _arm(False)
        getattr(wb, fn)(*args)

    for _ in range(4):
        both("admit", rng.randint(20, 300))
    count_changes = 0
    for t in range(1, 401):
        n0 = len(wa.reqs)
        r = rng.random()
        n = len(wa.reqs)
        if r < 0.08 and n < S.MAXR:
            both("admit", rng.randint(20, 300))
        elif r < 0.16 and n > 1:
            both("finish", rng.randrange(n))
        elif r < 0.18 and n > 1:
            i, j = rng.sample(range(n), 2)
            both("swap", i, j)
        accepted = [rng.randint(0, S.K) for _ in wa.reqs]
        for row in reversed(range(len(wa.reqs))):
            if wa.reqs[row]["s"] + 2 * (1 + S.K) + S.K > S.WIDTH * S.BS - S.BS:
                both("finish", row)
                accepted.pop(row)
        both("advance", accepted)
        count_changes += len(wa.reqs) != n0
        l6_ab.begin_step()
        sa, sb = S.Step(wa, t), S.Step(wb, t)
        sb.positions, sb.seq_lens, sb.qsl, sb.rd, sb.mrd, sb.offsets = (
            sa.positions, sa.seq_lens, sa.qsl, sa.rd, sa.mrd, sa.offsets
        )
        _arm(True)
        out_a, ctx_a = S._call(ra, sa, steady)
        _arm(False)
        out_b, ctx_b = S._call(rb, sb, False)
        S._same(out_a, ctx_a, out_b, ctx_b)
        S._step_fields(out_a, ra, ctx_a)
    c = _counts()
    assert count_changes > 30, count_changes
    assert c.get("admitinc.inc", 0) > 20 and c.get("admitinc.batch", 0) > 200, c


# ---------------------------------------------------------------- 4. CHECK


def test_check_mode_compares_and_agrees(lever):
    errors = lever("1", check=1)
    bts = _tables()
    m = object.__new__(MultiGroupBlockTable)
    m.block_tables = bts
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    rng = random.Random(5)
    n = 4
    for step in range(200):
        name, args = _random_mg_op(rng, m)
        getattr(m, name)(*args)
        n = rng.randint(1, 8)
        ctx = _ctx(0, n)
        assert torch.equal(_pack(runner, bts, WIDTHS, ctx, 8), _reference(bts, WIDTHS, ctx, 8))
    assert not errors and am._admit_on()
    assert _counts().get("admitinc.check", 0) == am._ADMIT_STATS["inc"] + am._ADMIT_STATS["batch"] > 150


def test_check_catches_a_stale_pack_and_keeps_the_full_one(lever):
    errors = lever("1", check=1)
    bts = _tables()
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    for row in range(6):
        for bt in bts:
            bt.add_row([row + 1, row + 2], row)
    _pack(runner, bts, WIDTHS, _ctx(0, 5), 8)  # full
    bts[2].get_cpu_tensor()[1, 0] = 4242  # not through BlockTable: no dirty mark
    ctx = _ctx(0, 6)
    got = _pack(runner, bts, WIDTHS, ctx, 8)
    assert torch.equal(got, _reference(bts, WIDTHS, ctx, 8))  # the full pack's bytes, not the stale ones
    assert len(errors) == 1 and "CHECK mismatch" in errors[0], errors
    assert not am._admit_on() and _counts().get("admitinc.mismatch") == 1
    f0 = am._STAGE_INC_STATS["full"]
    ctx = _ctx(0, 4)
    assert torch.equal(_pack(runner, bts, WIDTHS, ctx, 8), _reference(bts, WIDTHS, ctx, 8))
    assert am._STAGE_INC_STATS["full"] == f0 + 1  # lever off for the process


def test_check_catches_a_wrong_batched_write_and_keeps_vllms_rows(lever, monkeypatch):
    errors = lever("1", check=1)
    a, b = _mg(WIDTHS), _mg(WIDTHS)

    def broken_clear(tables, row_idx):
        for bt in tables:
            bt.num_blocks_per_row[row_idx] = 0  # forgets to zero the row

    monkeypatch.setitem(am._ADMIT_MG, "clear_row", (broken_clear,) + am._ADMIT_MG["clear_row"][1:])
    for m, on in ((b, False), (a, True)):
        _arm(on)
        m.add_row(([1, 2], [3], [4, 5, 6], [7]), 2)
        m.clear_row(2)
    _same_state(a, b)  # vLLM's rows kept
    assert len(errors) == 1 and "MultiGroupBlockTable.clear_row" in errors[0], errors
    assert not am._admit_on() and _counts().get("admitinc.mismatch") == 1


def test_check_steps_bound_the_check_turns(lever):
    """VLLM_TPU_L6_CHECK_STEPS=M: CHECK turns only in the first M worker steps."""
    lever("1", check=1, check_steps=20)
    bts = _tables()
    m = object.__new__(MultiGroupBlockTable)
    m.block_tables = bts
    runner = SimpleNamespace(_block_table_stage_cpu=None, device=CPU)
    rng = random.Random(11)
    per_step = []
    for step in range(60):
        if step:
            l6_ab.begin_step()
        before = _counts().get("admitinc.check", 0)
        name, args = _random_mg_op(rng, m)
        getattr(m, name)(*args)
        ctx = _ctx(0, 1 + step % 7)
        assert torch.equal(_pack(runner, bts, WIDTHS, ctx, 8), _reference(bts, WIDTHS, ctx, 8))
        per_step.append(_counts().get("admitinc.check", 0) - before)
    assert all(per_step[:20]) and not any(per_step[20:]), per_step
    assert am._ADMIT_STATS["inc"] > 40 and am._ADMIT_STATS["batch"] == 60


# ---------------------------------------------------------------- 5. flags-off identity, 6. registration


_PROBE = r"""
import vllm_torchtpu.layers.core.attention_metadata as am
from vllm.v1.worker.block_table import MultiGroupBlockTable as M
from vllm_torchtpu.core import l6_ab
patched = sorted(n for n in ("add_row", "append_row", "clear_row", "move_row", "swap_row")
                 if hasattr(M.__dict__[n], "_q38_admit_orig"))
print("PROBE", bool(M.__dict__.get("_q38_admit_patch")), ",".join(patched) or "-", l6_ab.configured("admitinc"))
"""


@pytest.mark.parametrize("value,want", [("", "PROBE False - False"), ("0", "PROBE False - False"),
                                        ("1", "PROBE True add_row,append_row,clear_row,move_row,swap_row True"),
                                        ("ab:1", "PROBE True add_row,append_row,clear_row,move_row,swap_row True")])
def test_flag_decides_whether_vllm_is_patched_at_import(value, want):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VLLM_TPU_L6_")}
    env.update(JAX_PLATFORMS="cpu", VLLM_TPU_L6_ADMIT_INC=value)
    if value.startswith("ab:"):
        env["VLLM_TPU_L6_AB_STEPS"] = "200"
    p = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=600)
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("PROBE")]
    assert p.returncode == 0 and line == [want], (p.returncode, line, p.stderr[-2000:])


def test_source_pin_matches_this_vllm():
    got = am._admit_bt_source_sha1(BlockTable, MultiGroupBlockTable)
    print(f"ADMIT_BT_SRC_SHA1 observed {got} pinned {am._ADMIT_BT_SRC_SHA1}")
    assert got == am._ADMIT_BT_SRC_SHA1, got


def test_a_different_blocktable_refuses_the_batched_writes(monkeypatch):
    warned = []
    monkeypatch.setattr(am.logger, "warning", lambda fmt, *a: warned.append(fmt % a))

    class BT2(BlockTable):
        def clear_row(self, row_idx: int) -> None:  # different row semantics
            self.num_blocks_per_row[row_idx] = 0

    class MG2(MultiGroupBlockTable):
        pass

    assert am._admit_batch_install(BT2, MG2) is False
    assert "_q38_admit_patch" not in MG2.__dict__ and "add_row" not in MG2.__dict__
    assert len(warned) == 1 and "not the pinned" in warned[0]

    class MG3(MultiGroupBlockTable):
        pass

    assert am._admit_batch_install(BlockTable, MG3) is True
    assert MG3.__dict__["_q38_admit_patch"] and hasattr(MG3.__dict__["add_row"], "_q38_admit_orig")


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert l6_ab.LEVERS["admitinc"] == "VLLM_TPU_L6_ADMIT_INC"
    for name in ("VLLM_TPU_L6_ADMIT_INC", "VLLM_TPU_L6_ADMIT_INC_CHECK"):
        assert name in envs.environment_variables, name
        assert name in _TPU_COMPILE_ENV_IGNORED, name


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
