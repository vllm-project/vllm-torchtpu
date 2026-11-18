# SPDX-License-Identifier: Apache-2.0
"""q38 HCDEC-HOSTADM SEED-SPEC (lever seedspec, VLLM_TPU_L6_SEED_SPEC): CPU proof against the stock blocking seed.

The real ``TPUModelRunner._collect_mamba_state_seed_copies`` (with ``runner/l6_seedspec.py``), the real
``_spec_seed_rows_impl`` / ``_rollback_offsets_seed`` / ``_flush_mamba_state_seed_copies`` (copy kernel replaced by
its CPU semantics ``pool[dst] = pool[src]``, gathered before the scatter), both upload paths (per-tensor and D1N
L-GDNCOPY packed) and the real post-verify read-offset scatter ``_update_read_offsets_impl`` run over a simulated
async MTP decode (k = 3): a scheduler that learns each step's rejections one step late, random acceptance, joins
with prompts ending at or next to a column edge, leaves at a pending edge, a finished id resubmitted, recycled blocks
carrying stale bytes and stale read offsets, partial draft counts, and an output thread that lands each copy one step
later (or not). The device side is modelled by the contract the GDN op follows (``models/vllm/qwen4_exp/model.py``
``ple_speculative_state``; ``megakernel_runtime/qwen38_gdn_commit.py``): a request reads checkpoint
``read_offsets[state block]`` of the group at its device column ``(seq_len - 1) // stride`` and writes one
checkpoint per window position (``full``) or only the accepted one (``commit``, GDN-COMMIT1's contract).

Port onto Benliang Wang's tree (q38 BLSTACK-SEED; from q38-hcdec-seedspec 3552fb4b2, credit HCDEC-HOSTADM): no D1N
packed upload path and no in-run A/B on that tree (``core/l6_ab.py`` is a plain shim: "1" on, "ab:" refused), so the
packed arm and the ab-window tests are not carried; his GDN window-log groups (VLLM_TPU_GDN_WINDOW_LOG, on in his
served decode) are added: such a group keeps the state after window position 0 in the state block plus a log of the
draft positions (modelled as the block's second word = the log's base position), a read applies the first
``read_offsets[state block]`` logged positions, a forward writes only the state block, and a crossing copies
checkpoint 0 with its log and migrates (not resets) the read offset (``tpu_runner.gdn_window_log_group_ids``).

Every state the model consumes is checked against an oracle (the recurrence folded over the request's tokens), so a
missing, stale or wrong seed fails on its own, in either arm. Across arms (lever off = stock wait, lever on, ab:
windows switching mid-request) the consumed states, the read offset and contents of every block a reader can reach,
and the host's ``_mamba_state_pos`` (pending entries resolved) are compared step by step. The poison runs overwrite
the extra block of every not-crossed speculative seed (unless it was a self-copy) with a sentinel and its read
offset with garbage: the oracle must still hold, i.e. nothing reads them before the real crossing rewrites them.
"""

from __future__ import annotations

import ast
import itertools
import os
import re
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import vllm_torchtpu.runner.tpu_runner as runner_mod
from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import l6_seedspec
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner, _update_read_offsets_impl
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID, AsyncPreResults

K = 3
W = K + 1
P61 = (1 << 61) - 1
POISON = 777_777_777_777
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURRENT: list = [None]  # the World whose step is running (resolution waits are told apart from stock waits)


def tok(rnum: int, p: int) -> int:
    return (rnum * 7919 + p * 104729 + 13) % 50021


def fold(s: int, t: int, g: int) -> int:
    return (s * 1000003 + t * (g + 3) + 17) % P61


class FakeCopy:
    def __init__(self, world, sampled):
        self.world = world
        self._s = sampled
        self.completed = False

    def wait(self):
        if not self.completed:
            self.world.waits.append((self.world.n, self.world.phase))
            self.completed = True

    @property
    def sampled_token_ids_cpu(self):
        return self._s


class Req:
    def __init__(self, rid, rnum, prompt):
        self.rid = rid
        self.rnum = rnum
        self.N = prompt  # exact tokens computed before the next step
        self.rej = 0  # the previous step's rejected drafts (the scheduler learns it one step late)
        self.k = 0  # decode steps taken
        self.blocks = None  # per group: list of block ids per column
        self.orc = None  # per group: oracle states, index p + 1 = state after position p


class World:
    """One decode engine's mamba-seed loop over a fake runner (lever state from the real l6_ab)."""

    def __init__(self, seed, *, n0=6, stride=8, groups=2, mode="full", split=1, max_reqs=16, cols=None, nsteps=160,
                 variable=False, p_join=0.08, p_leave=0.06, p_reuse=0.3, p_land=1.0, poison=False,
                 near_edge=True, land_prev=False, log_groups=()):
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.stride, self.G, self.mode, self.split = stride, groups, mode, split
        self.log = frozenset(log_groups)  # window-log GDN groups (state block + log; read offset = logged positions)
        if cols is None:
            cols = (6 * stride + 4 * (nsteps + 4)) // stride + W + 4
        self.max_reqs, self.cols = max_reqs, cols
        self.variable, self.p_join, self.p_leave, self.p_reuse, self.p_land = variable, p_join, p_leave, p_reuse, p_land
        self.poison = poison
        self.land_prev = land_prev  # this rank's output thread already waited on the previous step's copy
        self.near_edge = near_edge
        self.nblocks = 2 + groups * (max_reqs + 2) * cols
        self.free = list(range(self.nblocks - 1, 0, -1))  # 0 = null block
        self.pools = [torch.from_numpy(self.rng.integers(1, 1 << 40, (self.nblocks * split, 2))) for _ in range(groups)]
        self.offsets = torch.from_numpy(self.rng.integers(0, W, self.nblocks).astype(np.int32))  # stale garbage
        self.tables = [torch.zeros((max_reqs + 4, cols), dtype=torch.int32) for _ in range(groups)]
        self.reqs: list[Req] = []
        self.next_num = 0
        self.n = 0
        self.phase = "-"
        self.waits = []
        self.consumed = []
        self.reach = []
        self.posview = []
        self.bad = []
        self.stale_pending = []  # (step, rid): an entry older than this step's results object survived the collector
        self.stats = dict(spec=0, cross=0, stay=0, cross_after_stay=0, back_to_back=0, first_edge=0, leave_pending=0,
                          reuse_pending=0, poisoned=0, noncanon_edge=0, on_steps=0, off_steps=0, seed_steps=0)
        self.prev_results = None
        self.older_results = []
        self.last_outcome = {}
        self.last_col = {}
        self.last_change = {}
        self.r = self._make_runner()
        for _ in range(n0):
            self._join()

    # ---- fake runner -------------------------------------------------
    def _make_runner(self):
        G = self.G
        plan = [(g + 1, [self.pools[g]]) for g in range(G)]
        ib = NS(
            req_ids=[None] * (self.max_reqs + 4),
            num_computed_tokens_cpu=np.zeros(self.max_reqs + 4, dtype=np.int64),
            block_table=[NS(get_cpu_tensor=lambda: torch.zeros((1, 1), dtype=torch.int32))]
            + [NS(get_cpu_tensor=(lambda t=self.tables[g]: t)) for g in range(G)],
        )
        r = NS(
            requests={},
            _mamba_copy_plan=plan,
            scheduler_config=NS(async_scheduling=True),
            speculative_config=NS(num_speculative_tokens=K),
            _pre_async_results=None,
            _mamba_state_pos={},
            _mamba_offset_seeded=set(),
            _mamba_state_block_size=self.stride,
            _pool_block_split=self.split,
            bm_pool_layout=None,
            _mamba_ckpt_window=W,
            _mamba_window_log_gids=frozenset(g + 1 for g in self.log),
            device=torch.device("cpu"),
            mamba_slot_read_offsets=self.offsets,
            _pending_mamba_state_copies=[],
            input_batch=ib,
        )
        r._bucket_len = TPUModelRunner._bucket_len
        r._pad_to_bucket = TPUModelRunner._pad_to_bucket
        for name in ("_spec_seed_copies", "_copy_mamba_state_blocks", "_flush_mamba_state_seed_copies"):
            setattr(r, name, getattr(TPUModelRunner, name).__get__(r))
        return r

    # ---- batch edits ---------------------------------------------------
    def _alloc(self, req, upto_col):
        for g in range(self.G):
            while len(req.blocks[g]) <= min(upto_col, self.cols - 1):
                req.blocks[g].append(self.free.pop())

    def _sync_table_row(self, row, req):
        for g in range(self.G):
            t = self.tables[g]
            t[row].zero_()
            ids = req.blocks[g]
            t[row, : len(ids)] = torch.tensor(ids, dtype=torch.int32)

    def _oracle(self, req, g, p):
        o = req.orc[g]
        while len(o) <= p + 1:
            q = len(o) - 1
            o.append(fold(o[-1], tok(req.rnum, q), g))
        return o[p + 1]

    def _put(self, g, block, value, base=0):
        for j in range(self.split):
            self.pools[g][block * self.split + j, 0] = value + j
            self.pools[g][block * self.split + j, 1] = base

    def _read(self, req, g, blk, col, off):
        """What the GDN op consumes: checkpoint ``off`` of the group at column ``col`` (regular), or the state block's
        checkpoint 0 with its first ``off`` logged positions applied (window log)."""
        if g not in self.log:
            ck = [blk[col + t] if col + t < len(blk) else 0 for t in range(W)]
            return self._get(g, ck[off])
        st = blk[col]
        v = self._get(g, st)
        if isinstance(v, tuple):
            return v
        base = int(self.pools[g][st * self.split, 1])
        for q in range(base + 1, base + off + 1):
            v = fold(v, tok(req.rnum, q), g)
        return v

    def _get(self, g, block):
        v = int(self.pools[g][block * self.split, 0])
        for j in range(1, self.split):
            if int(self.pools[g][block * self.split + j, 0]) != v + j:
                return ("torn", v)
        return v

    def _join(self, rid=None):
        if len(self.reqs) >= self.max_reqs:
            return None
        rnum = self.next_num
        self.next_num += 1
        rid = rid or f"r{rnum}"
        s = self.stride
        if self.near_edge:
            base = int(self.rng.integers(2, 6)) * s
            prompt = base + int(self.rng.choice([-1, 0, 1, 2, -2]))  # the first decode token at / next to an edge
        else:
            prompt = int(self.rng.integers(s, 6 * s))
        req = Req(rid, rnum, prompt)
        req.orc = [[(rnum * 31 + g * 7 + 5) % P61] for g in range(self.G)]
        req.blocks = [[] for _ in range(self.G)]
        self._alloc(req, (prompt + K) // s + W)
        row = len(self.reqs)
        self.reqs.append(req)
        self._sync_table_row(row, req)
        # the landed prefill state: the state after the prompt's last token, in the block of its column
        for g in range(self.G):
            self._put(g, req.blocks[g][(prompt - 1) // s], self._oracle(req, g, prompt - 1), base=prompt - 1)
        self.r.requests[rid] = object()
        return req

    def _leave(self, j):
        req = self.reqs[j]
        last = len(self.reqs) - 1
        if j != last:
            self.reqs[j] = self.reqs[last]
            self._sync_table_row(j, self.reqs[j])
        self.reqs.pop()
        for g in range(self.G):
            self.tables[g][last].zero_()
            self.free.extend(req.blocks[g])  # recycled with their bytes and read offsets
        # _update_states: a finished request's column and seeded flag go, and so does its previous-results row
        self.r._mamba_state_pos.pop(req.rid, None)
        if self.prev_results is not None:
            self.prev_results.req_id_to_index_copy.pop(req.rid, None)
        self.r._mamba_offset_seeded.discard(req.rid)
        del self.r.requests[req.rid]
        return req

    # ---- one step ------------------------------------------------------
    def _accept(self, req, d):
        h = (self.seed * 1_000_003 + req.rnum * 7919 + req.k * 104729) % 2_147_483_647
        return int(np.random.default_rng(h).integers(0, d + 1))

    def _drafts(self, req):
        if req.k == 0:
            return 0  # the first decode step after landing: no drafts yet
        if not self.variable:
            return K
        h = (self.seed * 7 + req.rnum * 131 + req.k * 977) % 2_147_483_647
        return int(np.random.default_rng(h).choice([3, 3, 3, 2, 1, 0]))

    def step(self):
        rng = self.rng
        self.n += 1
        CURRENT[0] = self
        # output thread: results older than the previous step's have landed (or not, p_land; a hash of the step and
        # the results' own step, so both arms land the same copies whatever the collector waited on)
        for res in self.older_results:
            h = (self.seed * 2_654_435_761 + self.n * 40_503 + res._step * 9_973) % 1_000_003
            if h / 1_000_003 < self.p_land and not res.copy_state.completed:
                res.copy_state.wait()
        self.older_results = [x for x in self.older_results if not x.copy_state.completed]
        self.phase = "-"
        # scheduler events, applied before the step (as _update_states does)
        pend = self.r.__dict__.get("_seedspec_pending") or {}
        if self.reqs and rng.random() < self.p_leave:
            j = int(rng.integers(0, len(self.reqs)))
            if self.reqs[j].rid in pend:
                self.stats["leave_pending"] += 1
            gone = self._leave(j)
            if rng.random() < self.p_reuse:  # the finished id resubmitted in this step
                if gone.rid in pend:
                    self.stats["reuse_pending"] += 1
                self._join(rid=gone.rid)
        if rng.random() < self.p_join:
            self._join()
        if not self.reqs:
            self._join()
        n = len(self.reqs)
        ib = self.r.input_batch
        sched = {}
        ds = []
        for row, req in enumerate(self.reqs):
            d = self._drafts(req)
            ds.append(d)
            host = req.N + req.rej
            ib.req_ids[row] = req.rid
            ib.num_computed_tokens_cpu[row] = host
            sched[req.rid] = 1 + d
            self._alloc(req, (host + d) // self.stride + W)
            self._sync_table_row(row, req)
        for row in range(n, len(ib.req_ids)):
            ib.req_ids[row] = None
        self.r._pre_async_results = self.prev_results
        # lever state for this step
        l6_ab.begin_step()
        lever = l6_seedspec.on()
        self.stats["on_steps" if lever else "off_steps"] += 1
        prev = self.prev_results
        if self.land_prev and prev is not None and not prev.copy_state.completed:
            self.phase = "output-thread"
            prev.copy_state.wait()
        nwait = len(self.waits)
        self.phase = "collector"
        TPUModelRunner._collect_mamba_state_seed_copies(self.r, NS(num_scheduled_tokens=sched), 0, n)
        self.phase = "-"
        if lever and any(ph == "collector" for _, ph in self.waits[nwait:]):
            self.stats["noncanon_edge"] += 1  # a lever-on step that still took the stock wait
        after = dict((self.r.__dict__.get("_seedspec_pending") or {}))
        new_spec = {rid: p for rid, p in after.items() if p.results is prev}
        self.stale_pending += [(self.n, rid) for rid, p in after.items() if p.results is not prev]
        self.stats["spec"] += len(new_spec)
        if self.r._pending_mamba_state_copies:
            self.stats["seed_steps"] += 1  # a step that dispatches the seed programs
        self.r._flush_mamba_state_seed_copies()
        rows_of = {req.rid: row for row, req in enumerate(self.reqs)}
        # poison the extra block of every not-crossed speculative seed (unless it was a self-copy)
        if self.poison:
            for rid, p in new_spec.items():
                req = self.reqs[rows_of[rid]]
                true_col = (p.end - req.rej) // self.stride
                if true_col != p.low:
                    continue
                for g in range(self.G):
                    lo_blk, hi_blk = req.blocks[g][p.low], req.blocks[g][p.high]
                    if g not in self.log and int(self.offsets[lo_blk]) == 1:
                        continue
                    self._put(g, hi_blk, POISON, base=int(rng.integers(0, 1 << 20)))
                    self.offsets[hi_blk] = int(rng.integers(0, W))
                    self.stats["poisoned"] += 1
        # _reset_read_offsets_for_new_requests (after the flush): new owners resume from checkpoint 0
        for row, req in enumerate(self.reqs):
            if req.rid not in self.r._mamba_offset_seeded:
                self.r._mamba_offset_seeded.add(req.rid)
                col = (req.N + ds[row]) // self.stride
                for g in range(self.G):
                    self.offsets[req.blocks[g][col]] = 0
        # forward: read checkpoint read_offsets[state block] of the device column, write the window
        cons = []
        st_rows = [[0] * n for _ in range(self.G)]
        accepted = []
        for row, req in enumerate(self.reqs):
            d = ds[row]
            e_dev = req.N + d
            col = e_dev // self.stride
            a = self._accept(req, d)
            accepted.append(a)
            for g in range(self.G):
                blk = req.blocks[g]
                st = blk[col]
                st_rows[g][row] = st
                off = max(0, min(int(self.offsets[st]), W - 1))
                ck = [blk[col + t] if col + t < len(blk) else 0 for t in range(W)]
                got = self._read(req, g, blk, col, off)
                want = self._oracle(req, g, req.N - 1)
                cons.append((req.rid, g, got))
                if got != want:
                    self.bad.append((self.n, req.rid, g, "read", got, want, col, off))
                if g in self.log:
                    # window log: the state after window position 0 plus the log of positions 1..d, in the state block
                    self._put(g, st, self._oracle(req, g, req.N), base=req.N)
                    continue
                for t in range(d + 1):
                    if self.mode == "full" or t == a:
                        self._put(g, ck[t], self._oracle(req, g, req.N + t))
        self.consumed.append(cons)
        # verify: the real post-verify read-offset scatter (offset = accepted drafts)
        nt = torch.full((n, K + 1), INVALID_TOKEN_ID, dtype=torch.int64)
        for row, a in enumerate(accepted):
            nt[row, : 1 + a] = 7
        _update_read_offsets_impl(
            self.offsets, [torch.tensor(st_rows[g], dtype=torch.int64) for g in range(self.G)], nt, torch.tensor(n)
        )
        # host view: the stock path's column for every request (pending ones resolved with the true rejection)
        view = {}
        outcome = {}
        for rid, c in self.r._mamba_state_pos.items():
            p = after.get(rid)
            if p is not None and p.results is prev and rid in rows_of:
                req = self.reqs[rows_of[rid]]
                c = (p.end - req.rej) // self.stride
                outcome[rid] = "cross" if c == p.high else "stay"
                self.stats[outcome[rid]] += 1
            view[rid] = c
        self.posview.append(view)
        for row, req in enumerate(self.reqs):
            rid = req.rid
            c_now = view.get(rid)
            if req.k == 0 and (req.N - 1) // self.stride != (req.N + ds[row]) // self.stride:
                self.stats["first_edge"] += 1
            if c_now is None:
                continue
            if rid in self.last_col and c_now != self.last_col[rid]:
                if self.last_change.get(rid) == self.n - 1:
                    self.stats["back_to_back"] += 1
                if self.last_outcome.get(rid) == ("stay", self.n - 1):
                    self.stats["cross_after_stay"] += 1
                self.last_change[rid] = self.n
            self.last_col[rid] = c_now
            if rid in outcome:
                self.last_outcome[rid] = (outcome[rid], self.n)
        # reachable state after the step: every live request's read offset and read checkpoint
        reach = []
        for row, req in enumerate(self.reqs):
            col = (req.N + ds[row]) // self.stride
            for g in range(self.G):
                blk = req.blocks[g]
                st = blk[col]
                off = int(self.offsets[st])
                reach.append((req.rid, g, off, self._read(req, g, blk, col, max(0, min(off, W - 1)))))
        self.reach.append(reach)
        # results for the next step
        sampled = torch.full((n, K + 1), INVALID_TOKEN_ID, dtype=torch.int64)
        for row, a in enumerate(accepted):
            sampled[row, : 1 + a] = 11
        res = AsyncPreResults(
            req_ids=[q.rid for q in self.reqs],
            next_tokens_tpu=torch.zeros(1),
            request_seq_lens=[],
            discard_sampled_tokens_req_indices=[],
            req_id_to_index_copy={q.rid: row for row, q in enumerate(self.reqs)},
            copy_state=None,
            num_draft_per_req={row: ds[row] for row in range(n)},
        )
        res.copy_state = FakeCopy(self, sampled)
        res._step = self.n
        if self.prev_results is not None:
            self.older_results.append(self.prev_results)
        self.prev_results = res
        for row, req in enumerate(self.reqs):
            a = accepted[row]
            req.N += 1 + a
            req.rej = ds[row] - a
            req.k += 1


# ---- fixtures ------------------------------------------------------------

LEVER_ENVS = ("VLLM_TPU_L6_SEED_SPEC",)


def _set_lever(monkeypatch, value):
    monkeypatch.delenv("VLLM_TPU_L6_SEED_SPEC", raising=False)
    if value:
        monkeypatch.setenv("VLLM_TPU_L6_SEED_SPEC", value)
    l6_ab._resolve()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    def cpu_copy(raws, src, dst):
        for pool in raws:
            vals = pool[src.long()].clone()
            pool[dst.long()] = vals

    monkeypatch.setattr(runner_mod, "copy_mamba_state_blocks", cpu_copy)
    monkeypatch.setattr(runner_mod, "get_pcp_group", lambda: NS(world_size=1), raising=False)
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    counts = {}
    orig = l6_ab.note

    def note(lever, key, n=1):
        counts[f"{lever}.{key}"] = counts.get(f"{lever}.{key}", 0) + n
        orig(lever, key, n)

    monkeypatch.setattr(l6_ab, "note", note)
    for name in ("resolve", "settle"):
        orig_fn = getattr(l6_seedspec, name)

        def wrapped(*a, _orig=orig_fn, _name=name):
            w = CURRENT[0]
            old = w.phase if w is not None else None
            if w is not None:
                w.phase = _name
            try:
                return _orig(*a)
            finally:
                if w is not None:
                    w.phase = old

        monkeypatch.setattr(l6_seedspec, name, wrapped)
    yield counts
    for k in LEVER_ENVS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def _run(monkeypatch, lever, *, nsteps=160, **kw):
    _set_lever(monkeypatch, lever)
    w = World(nsteps=nsteps, **kw)
    for _ in range(nsteps):
        w.step()
    return w


def _assert_same(a: World, b: World, what: str):
    assert not a.bad, f"{what}: arm A read a wrong state: {a.bad[:3]}"
    assert not b.bad, f"{what}: arm B read a wrong state: {b.bad[:3]}"
    for i, (x, y) in enumerate(zip(a.consumed, b.consumed)):
        assert x == y, f"{what}: consumed states differ at step {i + 1}"
    for i, (x, y) in enumerate(zip(a.reach, b.reach)):
        assert x == y, f"{what}: reachable read offset / checkpoint differ at step {i + 1}: " \
                       f"{[p for p in zip(x, y) if p[0] != p[1]][:3]}"
    for i, (x, y) in enumerate(zip(a.posview, b.posview)):
        assert x == y, f"{what}: host columns differ at step {i + 1}: " \
                       f"{ {k: (x.get(k), y.get(k)) for k in set(x) | set(y) if x.get(k) != y.get(k)} }"


SCEN = [
    dict(seed=1, stride=8),
    dict(seed=2, stride=4),  # an edge every step or two: back-to-back crossings
    dict(seed=3, stride=6, variable=True),  # partial drafts: non-canonical edges take the stock wait
    dict(seed=4, stride=16, groups=3, split=2),
    dict(seed=5, stride=8, mode="commit"),
    dict(seed=6, stride=4, mode="commit", variable=True, p_leave=0.15, p_reuse=0.6),
    dict(seed=7, stride=8, p_land=0.5),  # the step-before copy not always landed at resolution
    # window-log GDN groups beside regular (PLE-like) checkpoint groups, as his served decode
    dict(seed=8, stride=8, groups=3, log_groups=(0, 1)),
    dict(seed=9, stride=4, groups=2, log_groups=(0,), variable=True, p_leave=0.15, p_reuse=0.6),
    dict(seed=10, stride=16, groups=4, split=2, log_groups=(0, 1, 2), p_land=0.5),
    dict(seed=17, stride=4, groups=2, log_groups=(0, 1)),  # every group window-log, back-to-back edges
]


@pytest.mark.parametrize("kw", SCEN, ids=[f"s{k['seed']}" for k in SCEN])
def test_lever_on_matches_stock_and_oracle(kw, monkeypatch, _env):
    packed = False  # no D1N packed upload path on this tree
    off = _run(monkeypatch, "", **kw)
    stock_waits = sum(1 for _, ph in off.waits if ph == "collector")
    on = _run(monkeypatch, "1", **kw)
    _assert_same(off, on, f"{kw}")
    assert stock_waits > 0, "the stock arm never waited at an edge (vacuous scenario)"
    assert on.stats["spec"] > 0 and on.stats["cross"] > 0 and on.stats["stay"] > 0, on.stats
    on_waits = sum(1 for _, ph in on.waits if ph == "collector")
    assert on_waits < stock_waits, (on_waits, stock_waits)
    if not kw.get("variable"):
        assert on_waits == 0, f"the lever left {on_waits} stock waits in canonical-only traffic"
    if kw.get("p_land", 1.0) == 1.0:
        assert _env.get("seedspec.rwait", 0) == 0, _env
    assert _env.get("seedspec.late", 0) == 0 and _env.get("seedspec.rsync", 0) == 0, _env
    assert not on.stale_pending, on.stale_pending[:5]
    print(f"\nQ38_SEEDSPEC eq {kw} packed={packed} steps={off.n} stock_waits={stock_waits} on_waits={on_waits} "
          f"stats={on.stats} counts={ {k: v for k, v in _env.items() if k.startswith('seedspec')} }")


@pytest.mark.parametrize("kw", SCEN, ids=[f"s{k['seed']}" for k in SCEN])
def test_poisoned_extra_writes_are_never_read(kw, monkeypatch):
    w = _run(monkeypatch, "1", poison=True, **kw)
    assert w.stats["poisoned"] > 0, "nothing poisoned (vacuous)"
    assert not w.bad, f"a poisoned block or read offset was read: {w.bad[:3]}"
    print(f"\nQ38_SEEDSPEC poison {kw} poisoned={w.stats['poisoned']} spec={w.stats['spec']}")


@pytest.mark.parametrize("kw", SCEN, ids=[f"s{k['seed']}" for k in SCEN])
def test_pending_entries_never_outlive_their_step(kw, monkeypatch, _env):
    """resolve() at the top of every collector call takes every entry staged against an older results object: the
    resolution of requests met again (settle() would also do that, counting seedspec.late) AND of requests that are
    not in the batch any more (finished, aborted, preempted: dropped). After each collector call only this step's
    entries remain, and no entry is ever resolved late."""
    w = _run(monkeypatch, "1", **kw)
    assert w.stats["spec"] > 0
    assert not w.stale_pending, f"entries outlived their step: {w.stale_pending[:5]}"
    assert _env.get("seedspec.late", 0) == 0, _env
    assert not w.bad


def test_settle_resolves_a_same_results_entry_by_waiting(monkeypatch, _env):
    """A request met again while the results object it was staged against is still the current one (no newer
    sampled-token copy yet): settle() must resolve its column from that object by waiting, to the stock path's
    column for that step; an entry from an older object is resolved the same way and counted late."""
    kw = dict(seed=16, stride=4, p_leave=0.0, p_join=0.0)
    off = _run(monkeypatch, "", nsteps=40, **kw)
    _set_lever(monkeypatch, "1")
    w = World(nsteps=40, **kw)
    checked = 0
    for k in range(40):
        w.step()
        pend = dict(w.r.__dict__.get("_seedspec_pending") or {})
        cur = w.r._pre_async_results  # the results object this step's collector read
        for rid, p in pend.items():
            assert p.results is cur
            n0 = _env.get("seedspec.rsync", 0)
            l6_seedspec.settle(w.r, rid, cur)
            assert _env.get("seedspec.rsync", 0) == n0 + 1
            assert rid not in (w.r.__dict__.get("_seedspec_pending") or {})
            assert w.r._mamba_state_pos[rid] == off.posview[k][rid], (k, rid)
            assert p.results.copy_state.completed  # it waited
            checked += 1
        if checked >= 20:
            break
    assert checked >= 5, checked
    assert _env.get("seedspec.late", 0) == 0
    # an entry from an older results object is resolved the same way and counted late
    w.step()
    pend = dict(w.r.__dict__.get("_seedspec_pending") or {})
    assert pend, "no pending entry to test the late path"
    rid, p = next(iter(pend.items()))
    l6_seedspec.settle(w.r, rid, object())
    assert _env.get("seedspec.late", 0) == 1 and rid not in (w.r.__dict__.get("_seedspec_pending") or {})


def test_coverage_of_the_requested_cases(monkeypatch):
    tot = {}
    for kw in SCEN:
        w = _run(monkeypatch, "1", **kw)
        for k, v in w.stats.items():
            tot[k] = tot.get(k, 0) + v
    print(f"\nQ38_SEEDSPEC coverage {tot}")
    for k in ("cross", "stay", "cross_after_stay", "back_to_back", "first_edge", "leave_pending", "reuse_pending",
              "noncanon_edge"):
        assert tot[k] > 0, (k, tot)
    assert tot["spec"] == tot["cross"] + tot["stay"], tot
    wl = {}
    for kw in [k for k in SCEN if k.get("log_groups")]:
        w = _run(monkeypatch, "1", **kw)
        for k, v in w.stats.items():
            wl[k] = wl.get(k, 0) + v
    assert wl["cross"] > 0 and wl["stay"] > 0 and wl["back_to_back"] > 0, ("window-log coverage", wl)


@pytest.mark.parametrize("value", ["ab:1", "ab:1+2+5+6", "2", "on"])
def test_shim_refuses_ab_and_unknown_values(value, monkeypatch):
    """No in-run A/B on this tree: "ab:" (and any value but 1 / 0 / unset) is refused loudly, never read as on/off."""
    monkeypatch.setenv("VLLM_TPU_L6_SEED_SPEC", value)
    with pytest.raises(ValueError, match="VLLM_TPU_L6_SEED_SPEC"):
        l6_ab._resolve()
    monkeypatch.delenv("VLLM_TPU_L6_SEED_SPEC")
    l6_ab._resolve()


@pytest.mark.parametrize("value,want", [("", False), ("0", False), ("1", True)])
def test_shim_plain_values(value, want, monkeypatch):
    _set_lever(monkeypatch, value)
    assert l6_ab.configured("seedspec") is want and l6_seedspec.on() is want


def test_shim_engaged_line_once_and_count_summary(monkeypatch):
    lines = []
    monkeypatch.setattr(l6_ab.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    monkeypatch.setattr(l6_ab, "_SUMMARY_EVERY", 3)
    _set_lever(monkeypatch, "1")
    for _ in range(4):
        assert l6_seedspec.on()
    for k in ("spec", "rwait", "spec", "cross", "spec"):
        l6_ab.note("seedspec", k, 250 if k == "rwait" else 1)
    assert lines == [
        "q38 BLSTACK seedspec ENGAGED (VLLM_TPU_L6_SEED_SPEC=1, plain; ab: refused)",
        "q38 BLSTACK seedspec counts cross=1 rwait=250 spec=3",
    ], lines
    _set_lever(monkeypatch, "")
    lines.clear()
    assert not l6_seedspec.on() and lines == []


def test_edge_declines_a_multi_chunk_previous_step(monkeypatch):
    """His collector asserts previous.num_chunks == 1 before it reads the sampled copy by previous_row; the lever
    declines such a step so that assert (the stock path) still applies."""
    _set_lever(monkeypatch, "1")
    r = NS(mamba_slot_read_offsets=torch.zeros(4, dtype=torch.int32), _mamba_ckpt_window=W, _mamba_state_pos={"x": 2})
    prev = NS(copy_state=NS(completed=False), num_chunks=2)
    assert l6_seedspec.edge(r, "x", prev, 0, K, 3 * 8 + 1, 8) is False
    assert not r.__dict__.get("_seedspec_pending")
    prev.num_chunks = 1
    assert l6_seedspec.edge(r, "x", prev, 0, K, 3 * 8 + 1, 8) is True


def test_decision_does_not_depend_on_this_ranks_copy_having_landed(monkeypatch):
    """Two TP ranks share the scheduler output but not their output threads' timing: one whose previous copy has
    always landed before the collector runs must stage the same seeds and reach the same states as one whose copy
    never has, so every rank dispatches the same seed programs."""
    kw = dict(seed=15, stride=4, p_leave=0.1)
    a = _run(monkeypatch, "1", **kw)
    b = _run(monkeypatch, "1", land_prev=True, **kw)
    _assert_same(a, b, "landed vs not landed")
    assert a.stats["spec"] == b.stats["spec"] > 0 and a.stats["seed_steps"] == b.stats["seed_steps"], (a.stats, b.stats)
    assert sum(1 for _, ph in b.waits if ph == "collector") == 0


def test_unset_lever_is_the_stock_path(monkeypatch, _env):
    w = _run(monkeypatch, "", seed=13, stride=4)
    assert w.stats["spec"] == 0 and not w.r.__dict__.get("_seedspec_pending")
    assert not any(k.startswith("seedspec.") for k in _env), _env
    assert not w.bad


def test_engagement_counts(monkeypatch, _env):
    w = _run(monkeypatch, "1", seed=14, stride=4, p_leave=0.12)
    assert _env.get("seedspec.spec", 0) == w.stats["spec"] > 0, (_env, w.stats)
    assert _env.get("seedspec.cross", 0) > 0 and _env.get("seedspec.stay", 0) > 0, _env
    assert _env.get("seedspec.drop", 0) > 0, _env


def test_served_shape_64_requests(monkeypatch, _env):
    """64 live requests (256 verify rows), 4 mamba groups (3 window-log GDN + 1 regular), stride 32: the edge census
    of a c112 decode."""
    kw = dict(seed=21, n0=64, max_reqs=64, stride=32, groups=4, p_join=0.1, p_leave=0.1, log_groups=(0, 1, 2))
    off = _run(monkeypatch, "", nsteps=60, **kw)
    on = _run(monkeypatch, "1", nsteps=60, **kw)
    _assert_same(off, on, "served shape")
    sw = sum(1 for _, ph in off.waits if ph == "collector")
    ow = sum(1 for _, ph in on.waits if ph == "collector")
    print(f"\nQ38_SEEDSPEC served-shape steps=60 stock_waits={sw} on_waits={ow} stats={on.stats}")
    assert sw > 0 and ow == 0 and on.stats["spec"] > 0


# ---- source / registration ------------------------------------------------

def test_key_is_registered_lever_and_compile_ignored():
    src = open(os.path.join(ROOT, "src/vllm_torchtpu/envs.py")).read()
    assert len(re.findall(r'"VLLM_TPU_L6_SEED_SPEC":', src)) == 1
    comp = open(os.path.join(ROOT, "src/vllm_torchtpu/compilation/tpu_compiler.py")).read()
    ign = re.search(r"_TPU_COMPILE_ENV_IGNORED = \{(.*?)\n\}", comp, re.S).group(1)
    assert len(re.findall(r'"VLLM_TPU_L6_SEED_SPEC",', ign)) == 1
    assert l6_ab.LEVERS["seedspec"] == "VLLM_TPU_L6_SEED_SPEC"


def test_stock_wait_statement_kept_and_guarded_by_the_lever():
    src = open(os.path.join(ROOT, "src/vllm_torchtpu/runner/tpu_runner.py")).read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_collect_mamba_state_seed_copies")
    seg = ast.get_source_segment(src, fn)
    assert seg.count("previous.wait_for_copy()") == 1
    assert "_l6_seedspec.settle(self, req_id, previous)" in seg
    i_res = seg.index("_l6_seedspec.resolve(self, previous)")
    i_loop = seg.index("for i in range(num_reqs):")
    i_edge = seg.index("_l6_seedspec.edge(")
    i_wait = seg.index("previous.wait_for_copy()")
    assert i_res < i_loop < i_edge < i_wait


def test_lever_module_reads_the_step_state_not_the_configuration():
    src = open(os.path.join(ROOT, "src/vllm_torchtpu/runner/l6_seedspec.py")).read()
    assert "return _l6_ab.on(LEVER)" in src


def test_microbench_collector_host_time(monkeypatch, capsys):
    """Descriptive: host time of the collector per step on a 64-request batch, lever off vs on (no real wait: the
    stock arm's copy wait is instant here; on the device it is the 8.6 ms median wait the lever removes)."""
    out = {}
    for lever in ("", "1"):
        kw = dict(seed=31, n0=64, max_reqs=64, stride=32, groups=4, p_join=0.0, p_leave=0.0, nsteps=80)
        _set_lever(monkeypatch, lever)
        w = World(**kw)
        ts = []
        orig = TPUModelRunner._collect_mamba_state_seed_copies

        def timed(self, *a, _orig=orig):
            t0 = time.perf_counter_ns()
            _orig(self, *a)
            ts.append(time.perf_counter_ns() - t0)

        monkeypatch.setattr(TPUModelRunner, "_collect_mamba_state_seed_copies", timed)
        for _ in range(80):
            w.step()
        monkeypatch.setattr(TPUModelRunner, "_collect_mamba_state_seed_copies", orig)
        out[lever or "off"] = (round(float(np.median(ts)) / 1e3, 1), round(float(np.mean(ts)) / 1e3, 1),
                               w.stats["seed_steps"], w.n)
    with capsys.disabled():
        print(f"\nQ38_SEEDSPEC_BENCH collector us per step (median, mean; CPU, includes the eager CPU run of the staged "
              f"seed programs), steps dispatching seed programs, steps; 64 requests x 4 groups: {out}")
