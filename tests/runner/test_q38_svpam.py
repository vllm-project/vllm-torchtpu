# SPDX-License-Identifier: Apache-2.0
"""q38 SV-PAM on CPU: VLLM_TPU_PAM_INCR (runner/pam_incr.py), the row-incremental target-walk block-table pack.

On K3's stock pack (q38 SV-PAM2 re-cut on H-b.3 e2f2fe8490: ``stage_block_table_uploads`` has no ``mark`` argument).

The real ``stage_block_table_uploads`` runs over real vLLM ``BlockTable`` objects (host tensors only), mutated through the
real ``MultiGroupBlockTable`` / ``BlockTable`` methods; only the walk-program dispatch is stubbed and the H2D is a host
copy. The reference at every walk is the real stock ``_pack_and_upload`` (the function the patch replaces), run on a
second runner over the same tables. A late-reading fake H2D reads each upload's source at the next upload.
"""

import functools
import inspect
import logging
import random
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

KEY = "VLLM_TPU_PAM_INCR"
AB = "VLLM_TPU_PAM_INCR_AB_MS"
MUTATORS = ("append_row", "add_row", "clear_row", "move_row", "swap_row", "clear")
# (most_model_len width, max_model_len width) per group: the first two swap widths when use_max_model_len flips, so
# target_num_blocks changes while the staged total stays the same.
WIDTHS = ((4, 6), (6, 4), (5, 9), (3, 3), (7, 11))


def _bt(rows, width):
    from vllm.v1.worker.block_table import BlockTable

    bt = object.__new__(BlockTable)
    cpu = torch.zeros((rows, width), dtype=torch.int32)
    bt.block_table = SimpleNamespace(cpu=cpu, np=cpu.numpy(), gpu=torch.zeros_like(cpu))
    bt.num_blocks_per_row = np.zeros(rows, dtype=np.int32)
    bt.use_hybrid_blocks = False
    bt.max_num_blocks_per_req = width
    return bt


class Builder:
    def __init__(self, gid, bt, widths):
        self.kv_cache_group_id, self.bt, self.widths = gid, bt, widths
        self.is_mamba_group = False

    def _block_table_geometry(self, ctx):
        return self.bt, self.widths[1 if ctx.use_max_model_len else 0]


def _runner(builders):
    groups = [SimpleNamespace(metadata_builders=[b]) for b in builders]
    return SimpleNamespace(
        attn_groups=[groups],
        input_batch=object(),
        device=torch.device("cpu"),
        # A preset scratch keeps the pinned-allocation branch out of a CPU-only run.
        _block_table_stage_cpu=torch.zeros((1 << 14,), dtype=torch.int32),
        _unified_kv_layout=False,
        _profiler_rank=0,
        _input_staging_fence=None,
    )


class Sim:
    """A persistent batch over real BlockTables: rows [0, N) are live; a walk packs rows [start, N)."""

    def __init__(self, seed, R=12, widths=WIDTHS):
        from vllm.v1.worker.block_table import MultiGroupBlockTable

        self.MG = MultiGroupBlockTable
        self.rng = random.Random(seed)
        self.R = R
        self.bts = [_bt(R, max(w)) for w in widths]
        self.mg = SimpleNamespace(block_tables=self.bts)
        self.builders = [Builder(g, bt, w) for g, (bt, w) in enumerate(zip(self.bts, widths))]
        self.N, self.npad, self.start, self.use_max = 0, R, 0, False
        self.next_block = 1
        self.runner = _runner(self.builders)
        self.ref = _runner(self.builders)

    # --- BlockTable traffic (the real methods) ---
    def ids(self, row, k=None, reset=False):
        out = []
        rng = self.rng
        for bt in self.bts:
            room = bt.max_num_blocks_per_req - (0 if reset else int(bt.num_blocks_per_row[row]))
            n = min(room, rng.choice((0, 1, 1, 2)) if k is None else k)
            out.append(list(range(self.next_block, self.next_block + n)))
            self.next_block += n
        return tuple(out)

    def fill(self, n):
        for _ in range(n):
            self.admit()

    def admit(self):
        if self.N < self.R:
            self.MG.add_row(self.mg, self.ids(self.N, 2, reset=True), self.N)
            self.N += 1

    def renew(self, row):
        """The row gets two fresh blocks per group (add_row): always a content change and a key miss."""
        self.MG.add_row(self.mg, self.ids(row, 2, reset=True), row)

    def churn(self):
        if self.N:
            self.renew(self.rng.randrange(self.N))

    def append(self, row=None):
        if self.N:
            row = self.rng.randrange(self.N) if row is None else row
            self.MG.append_row(self.mg, self.ids(row), row)

    def finish_condense(self):
        if self.N:
            h = self.rng.randrange(self.N)
            if self.rng.random() < 0.5:
                self.MG.clear_row(self.mg, h)
            if h != self.N - 1:
                self.MG.move_row(self.mg, self.N - 1, h)
            self.N -= 1

    def move_within(self):
        if self.N >= 2:
            i, j = self.rng.sample(range(self.N), 2)
            self.MG.move_row(self.mg, i, j)

    def swap(self):
        if self.N >= 2:
            i, j = self.rng.sample(range(self.N), 2)
            self.MG.swap_row(self.mg, i, j)

    def clear_row(self):
        if self.N:
            self.MG.clear_row(self.mg, self.rng.randrange(self.N))

    def shrink_stale(self):
        # The last live row leaves without any BlockTable call: its source row keeps its ids.
        if self.N:
            self.N -= 1

    def grow_stale(self):
        # A row joins without any BlockTable call: whatever its source row holds is live.
        if self.N < self.R:
            self.N += 1

    def clear_all(self):
        self.MG.clear(self.mg)

    # --- walks ---
    def ctx(self, dummy=False):
        start = min(self.start, self.N)
        return SimpleNamespace(
            use_max_model_len=self.use_max,
            mamba_ckpt_window=1,
            start_index=start,
            num_reqs=self.N - start,
            position_ids_override=torch.zeros(1, dtype=torch.int32) if dummy else None,
            staged_block_tables=None,
        )

    def entries(self, ctx):
        return [(b.kv_cache_group_id, b.bt, b._block_table_geometry(ctx)[1]) for b in self.builders]

    def walk(self, pam, dummy=False, unfenced=False):
        """One target walk through the real stage_block_table_uploads; returns (staged bytes, stock bytes)."""
        seen = pam.seen
        seen.clear()
        if unfenced:
            self.runner._input_staging_fence = object()
        try:
            pam.am.stage_block_table_uploads(self.runner, self.ctx(dummy), self.npad)
        finally:
            self.runner._input_staging_fence = None
        ctx = self.ctx(dummy)
        entries = self.entries(ctx)
        geom, off = [], 0
        for _, _, tnb in entries:
            geom.append((off, self.npad, tnb))
            off += self.npad * tnb
        pam.stock(self.ref, ctx, entries, self.npad, geom, None)
        want = self.ref._block_table_stage_cpu[:off].clone()
        # Independent of either pack: zeros, then each group's live rows.
        direct = torch.zeros((off,), dtype=torch.int32)
        o = 0
        for _, bt, tnb in entries:
            if not dummy:
                direct[o : o + self.npad * tnb].view(self.npad, tnb)[: ctx.num_reqs] = bt.get_cpu_tensor()[
                    ctx.start_index : ctx.start_index + ctx.num_reqs, :tnb
                ]
            o += self.npad * tnb
        assert torch.equal(want, direct)
        return seen["staged"], want

    def random_step(self, pam):
        rng = self.rng
        dummy = unfenced = False
        for _ in range(rng.choice((1, 1, 2, 3))):
            r = rng.random() * 100
            if r < 30:
                self.append()
            elif r < 40:
                self.admit()
            elif r < 50:
                self.finish_condense()
            elif r < 58:
                self.swap()
            elif r < 62:
                self.move_within()
            elif r < 66:
                self.clear_row()
            elif r < 70:
                self.shrink_stale()
            elif r < 74:
                self.grow_stale()
            elif r < 74.5:
                self.clear_all()
            elif r < 75.2:
                self.use_max = not self.use_max
            elif r < 75.9:
                self.npad = self.R + 3 if self.npad == self.R else self.R
            elif r < 76.6:
                self.start = 1 - self.start
            elif r < 78.1:
                dummy = True
            elif r < 79.6:
                unfenced = True
            # else: no change (the stage key is reused)
        return self.walk(pam, dummy=dummy, unfenced=unfenced)


@pytest.fixture
def pam(monkeypatch):
    from vllm.v1.worker.block_table import BlockTable

    from vllm_torchtpu.layers.core import attention_metadata as am
    from vllm_torchtpu.runner import pam_incr

    # Everything the install touches is restored at teardown.
    for name in MUTATORS:
        monkeypatch.setattr(BlockTable, name, BlockTable.__dict__[name])
    monkeypatch.setattr(BlockTable, pam_incr._PATCH_FLAG, False, raising=False)
    monkeypatch.setattr(am, "_pack_and_upload", am._pack_and_upload)
    monkeypatch.setattr(pam_incr, "ENABLED", pam_incr.ENABLED)
    monkeypatch.setattr(pam_incr, "AB_MS", pam_incr.AB_MS)
    monkeypatch.setitem(pam_incr._state, "ab_window", None)
    # No pinned allocation on a CPU-only run; the H2D is a host copy taken at enqueue (a completed transfer).
    monkeypatch.setattr(pam_incr, "_alloc", lambda n: torch.zeros((n,), dtype=torch.int32))
    monkeypatch.setattr(pam_incr, "_h2d", lambda flat, device: flat.clone())
    monkeypatch.setenv("VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE", "1")
    monkeypatch.setenv("VLLM_TPU_STAGE_ENTRIES_CACHE", "1")
    monkeypatch.delenv("VLLM_TPU_BLOCK_TABLE_STAGE_CHECK", raising=False)
    monkeypatch.delenv("VLLM_TPU_PAM_STATS", raising=False)
    monkeypatch.delenv(AB, raising=False)
    monkeypatch.setenv(KEY, "1")
    pam_incr._load()
    seen = {}

    def fake_issue(runner, ctx, entries, entry_builders, nrp, geom, staged_dev, reused_key=None):
        seen["staged"] = staged_dev.clone()

    monkeypatch.setattr(am, "_issue_walk_products", fake_issue)
    return SimpleNamespace(
        mp=monkeypatch, am=am, pi=pam_incr, BT=BlockTable, stock=am._pack_and_upload, seen=seen,
        before={name: BlockTable.__dict__[name] for name in MUTATORS},
    )


def _bump(pam):
    """A BlockTable version bump with no content change: the next walk misses the key and packs."""
    pam.am._BLOCK_TABLE_VERSION[0] += 1


def _on(pam):
    assert pam.pi.patch_pam_incr() is True
    assert pam.am._pack_and_upload is not pam.stock


def _stats(sim):
    return sim.runner._pam_incr["stats"]


def _primed(pam, seed=0, n=6, widths=WIDTHS):
    """A sim with n live rows whose two buffers were both packed (one walk each)."""
    _on(pam)
    sim = Sim(seed, widths=widths)
    sim.fill(n)
    got, want = sim.walk(pam)
    assert torch.equal(got, want)
    sim.renew(0)
    got, want = sim.walk(pam)
    assert torch.equal(got, want)
    assert _stats(sim)["full"] == 2
    return sim


# --- keys, install, pins --------------------------------------------------------------------------------------------


def test_keys_are_registered_and_compile_ignored(monkeypatch):
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    for k in (KEY, AB):
        assert k in envs.environment_variables
        assert k in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv(AB, raising=False)
    assert envs.VLLM_TPU_PAM_INCR is False
    assert envs.VLLM_TPU_PAM_INCR_AB_MS == 0


@pytest.mark.parametrize("period,ok", [(0, True), (1000, True), (60000, True), (43_200_000, True), (7, False),
                                       (86_400_000, False), (3_600_001, False)])
def test_ab_period_must_divide_the_day_evenly(pam, period, ok):
    pam.mp.setenv(AB, str(period))
    if ok:
        pam.pi._load()
        assert pam.pi.AB_MS == period
    else:
        with pytest.raises(ValueError, match="VLLM_TPU_PAM_INCR_AB_MS"):
            pam.pi._load()


def test_runner_init_installs_it():
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    src = textwrap.dedent(inspect.getsource(TPUModelRunner.__init__))
    assert src.count("pam_incr.patch_pam_incr()") == 1


@pytest.mark.parametrize("ab", [None, "60000"])
def test_key_off_patches_nothing(pam, caplog, ab):
    caplog.set_level(logging.INFO)
    pam.mp.delenv(KEY, raising=False)
    if ab:
        pam.mp.setenv(AB, ab)
    pam.pi._load()
    assert pam.pi.patch_pam_incr() is False
    assert pam.am._pack_and_upload is pam.stock
    assert all(pam.BT.__dict__[name] is pam.before[name] for name in MUTATORS)
    assert not getattr(pam.BT, pam.pi._PATCH_FLAG, False)
    sim = Sim(3)
    for _ in range(300):
        got, want = sim.random_step(pam)
        assert torch.equal(got, want)
    assert "_pam_incr" not in sim.runner.__dict__
    assert all(pam.pi._DIRTY not in bt.__dict__ for bt in sim.bts)
    assert "PAM_INCR" not in caplog.text


def test_key_on_installs_once_and_logs_the_install_line(pam, caplog):
    caplog.set_level(logging.INFO)
    _on(pam)
    patched = pam.am._pack_and_upload
    assert pam.pi.patch_pam_incr() is True
    assert pam.am._pack_and_upload is patched
    assert getattr(pam.BT, pam.pi._PATCH_FLAG) is True
    assert all(pam.BT.__dict__[name] is not pam.before[name] for name in MUTATORS)
    assert caplog.text.count("VLLM_TPU_PAM_INCR=1: row-incremental") == 1


def test_source_pins_match_and_a_changed_stock_pack_is_refused(pam):
    from vllm_torchtpu.runner import pam_incr

    assert pam_incr.source_sha256(pam.stock) == pam_incr.STOCK_PACK_SHA256

    def other(runner, ctx, entries, num_reqs_padded, table_geom, stage_key):
        return None

    pam.mp.setattr(pam.am, "_pack_and_upload", other)
    with pytest.raises(RuntimeError, match="VLLM_TPU_PAM_INCR refused"):
        pam_incr.patch_pam_incr()
    assert pam.am._pack_and_upload is other
    assert all(pam.BT.__dict__[name] is pam.before[name] for name in MUTATORS)


def test_a_changed_block_table_mutator_is_refused(pam):
    from vllm_torchtpu.runner import pam_incr

    BT = pam.BT
    assert pam_incr.source_sha256(*(getattr(BT, n) for n in MUTATORS), BT.get_cpu_tensor) == pam_incr.BLOCK_TABLE_SHA256
    orig = BT.__dict__["swap_row"]

    def swap_row(self, src, tgt):  # a mutator whose source differs from the pinned one (e.g. another vLLM)
        return orig(self, src, tgt)

    pam.mp.setattr(BT, "swap_row", swap_row)
    with pytest.raises(RuntimeError, match="VLLM_TPU_PAM_INCR refused"):
        pam_incr.patch_pam_incr()
    assert pam.am._pack_and_upload is pam.stock
    assert BT.__dict__["swap_row"] is swap_row
    assert not getattr(BT, pam_incr._PATCH_FLAG, False)


# --- the core assertion -----------------------------------------------------------------------------------------------


def test_staged_bytes_equal_the_stock_pack_over_random_steps(pam):
    _on(pam)
    sim = Sim(11)
    for i in range(10_000):
        got, want = sim.random_step(pam)
        assert torch.equal(got, want), "step %d" % i
    s = _stats(sim)
    packs = s["incr"] + s["full"]
    # The incremental path carries the bulk; full packs, fallbacks and key reuse all occur.
    assert s["incr"] >= 0.6 * packs, s
    assert s["full"] >= 10 and s["fallback"] >= 50, s
    assert s["calls"] < 10_000, s  # some walks reused the key
    assert s["rows"] / s["incr"] < sim.R / 2, s


def test_clear_marks_every_row(pam):
    sim = _primed(pam)
    sim.clear_all()
    for _ in range(2):
        got, want = sim.walk(pam)
        assert torch.equal(got, want)
        _bump(pam)
    assert int(want.abs().sum()) == 0
    assert _stats(sim)["full"] == 4


def test_growth_copies_the_rows_that_join(pam):
    sim = _primed(pam)
    sim.shrink_stale()
    sim.shrink_stale()
    for _ in range(2):
        got, want = sim.walk(pam)
        assert torch.equal(got, want)
        _bump(pam)
    sim.grow_stale()
    sim.grow_stale()
    for _ in range(2):
        got, want = sim.walk(pam)
        assert torch.equal(got, want)
        _bump(pam)
    assert _stats(sim)["incr"] == 4


# --- the mutation targets ---------------------------------------------------------------------------------------------


def test_swap_row_updates_both_rows(pam):
    sim = _primed(pam)
    sim.MG.swap_row(sim.mg, 1, 4)
    for _ in range(2):  # both buffers
        got, want = sim.walk(pam)
        assert torch.equal(got, want)
        _bump(pam)
    assert _stats(sim)["incr"] == 2


def test_move_row_updates_source_and_target(pam):
    sim = _primed(pam)
    sim.MG.move_row(sim.mg, 1, 3)  # both rows stay live: the source row was zeroed
    got, want = sim.walk(pam)
    assert torch.equal(got, want)
    assert _stats(sim)["incr"] == 1


def test_shrink_zeroes_the_dropped_rows(pam):
    sim = _primed(pam)
    sim.shrink_stale()
    sim.shrink_stale()  # rows 4 and 5 leave with their ids still in the source
    got, want = sim.walk(pam)
    assert torch.equal(got, want)
    assert _stats(sim)["incr"] == 1


def test_target_num_blocks_change_repacks(pam):
    sim = _primed(pam, widths=((4, 6), (6, 4), (5, 5)))
    sim.use_max = True  # groups 0 and 1 swap widths: same total, other layout
    got, want = sim.walk(pam)
    assert torch.equal(got, want)
    assert _stats(sim)["full"] == 3


def test_pending_rows_reach_the_other_buffer(pam):
    sim = _primed(pam)
    sim.renew(2)
    got, want = sim.walk(pam)  # buffer A takes row 2
    assert torch.equal(got, want)
    sim.shrink_stale()  # a key miss that changes no row
    got, want = sim.walk(pam)  # buffer B must take row 2 too
    assert torch.equal(got, want)
    assert _stats(sim)["incr"] == 2


def test_dummy_runs_take_the_stock_zero_pack(pam):
    sim = _primed(pam)
    got, want = sim.walk(pam, dummy=True)
    assert torch.equal(got, want)
    assert int(got.abs().sum()) == 0
    s = _stats(sim)
    assert s["fallback"] == 1 and s["full"] == 2 and s["incr"] == 0


def test_unfenced_walks_take_the_stock_pack(pam):
    sim = _primed(pam)
    bufs = [b["flat"].clone() for b in sim.runner._pam_incr["bufs"]]
    sim.renew(1)
    got, want = sim.walk(pam, unfenced=True)
    assert torch.equal(got, want)
    assert all(torch.equal(b["flat"], c) for b, c in zip(sim.runner._pam_incr["bufs"], bufs))
    assert _stats(sim)["fallback"] == 1
    got, want = sim.walk(pam)  # the unfenced walk keyed its upload; this walk reuses it
    assert torch.equal(got, want)
    assert _stats(sim)["calls"] == 3


def test_late_reader_never_sees_its_buffer_rewritten(pam):
    """Each upload reads its source at the next upload, after the next pack has written: a buffer still being read
    is never the one being written."""
    inflight = []
    bad = []

    def late_h2d(flat, device):
        while inflight:
            src, snap = inflight.pop()
            if not torch.equal(src, snap):
                bad.append(int((src != snap).sum()))
        inflight.append((flat, flat.clone()))
        return flat.clone()

    pam.mp.setattr(pam.pi, "_h2d", late_h2d)
    _on(pam)
    sim = Sim(21)
    for i in range(3000):
        got, want = sim.random_step(pam)
        assert torch.equal(got, want), "step %d" % i
    assert _stats(sim)["incr"] > 1000
    assert bad == [], "%d uploads saw their source rewritten in flight" % len(bad)


def test_ab_windows_switch_between_incremental_and_stock(pam, caplog):
    caplog.set_level(logging.INFO)
    clock = {"t": 0}
    pam.mp.setattr(pam.pi, "_now_ns", lambda: clock["t"])
    pam.mp.setenv(AB, "1000")
    pam.pi._load()
    calls = []
    stock = pam.stock

    @functools.wraps(stock)
    def spy(runner, *a):
        calls.append(runner)
        return stock(runner, *a)

    pam.mp.setattr(pam.am, "_pack_and_upload", spy)
    _on(pam)
    sim = Sim(31)
    sim.fill(6)

    def run(k):
        for _ in range(k):
            sim.churn()
            got, want = sim.walk(pam)
            assert torch.equal(got, want)

    run(20)  # window 0: on
    s = dict(_stats(sim))
    assert s["ab_off"] == 0 and s["incr"] + s["full"] == s["calls"] and s["incr"] > 0
    assert sum(r is sim.runner for r in calls) == 0
    bufs = [b["flat"].clone() for b in sim.runner._pam_incr["bufs"]]
    clock["t"] = 1_000_000_000  # window 1: off
    run(20)
    s2 = dict(_stats(sim))
    assert s2["ab_off"] == s2["calls"] - s["calls"] > 0
    assert s2["incr"] == s["incr"] and s2["full"] == s["full"]
    assert sum(r is sim.runner for r in calls) == s2["ab_off"]
    assert all(torch.equal(b["flat"], c) for b, c in zip(sim.runner._pam_incr["bufs"], bufs))
    clock["t"] = 2_000_000_000  # window 2: on again, after a full pack of each buffer
    run(20)
    s3 = dict(_stats(sim))
    assert s3["full"] == s2["full"] + 2 and s3["incr"] > s2["incr"]
    lines = [l for l in caplog.text.splitlines() if "PAM_INCR AB window=" in l]
    assert [l.split("window=")[1].split()[:2] for l in lines] == [["0", "on=1"], ["1", "on=0"], ["2", "on=1"]]


# --- the runtime engagement line --------------------------------------------------------------------------------------


def _misses(pam, sim, k):
    for _ in range(k):
        sim.churn()
        got, want = sim.walk(pam)
        assert torch.equal(got, want)


def test_runtime_stats_line_every_log_every_packs(pam, caplog):
    caplog.set_level(logging.INFO)
    _on(pam)
    sim = Sim(41)
    sim.fill(8)
    _misses(pam, sim, pam.pi.LOG_EVERY - 1)
    assert "PAM_INCR incr=" not in caplog.text
    _misses(pam, sim, 1)
    s = _stats(sim)
    assert s["calls"] == pam.pi.LOG_EVERY and s["incr"] + s["full"] == pam.pi.LOG_EVERY and s["incr"] > 0
    lines = [l for l in caplog.text.splitlines() if "PAM_INCR incr=" in l]
    assert len(lines) == 1
    assert "PAM_INCR incr=%d full=%d fallback=0 ab_off=0 dirty_rows_mean=" % (s["incr"], s["full"]) in lines[0]
    assert "pack_ms_mean=" in lines[0] and "rank=0" in lines[0]


def test_no_runtime_line_with_the_key_off(pam, caplog):
    caplog.set_level(logging.INFO)
    pam.mp.delenv(KEY, raising=False)
    pam.pi._load()
    assert pam.pi.patch_pam_incr() is False
    sim = Sim(41)
    sim.fill(8)
    _misses(pam, sim, pam.pi.LOG_EVERY)
    assert "PAM_INCR" not in caplog.text
