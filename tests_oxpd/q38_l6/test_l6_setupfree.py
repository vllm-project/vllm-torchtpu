# SPDX-License-Identifier: Apache-2.0
"""q38 L6 lever setupfree (runner/l6_setupfree.py, T-5(b)). Host-only: runs in a CDK CPU pod.

Every stream test drives the real ``TPUModelRunner._build_attention_metadata``
(STEADY's fast path, runner/l6_steady.py) on real vLLM BlockTables and real
AttentionMetadataBuilders (the decode world of tests/q38_l6/test_l6_steady.py,
L6-STEADY, generalised to more groups and multi-layer groups) twice from the
same seed, once with STEADY alone and once with ``VLLM_TPU_L6_SETUP_FREE=1``,
and compares every step, byte for byte: the per-layer metadata handed to the
forward (layer names and order, object sharing, per object the keys in order and
every value: tensor dtype / shape / bytes, other values by repr, and which
values are the same object within the step) and the context side effects
(staged block tables, mamba products, unified state list). With the lever on,
every fast step's output must also equal a fresh ``l6_steady._clone`` of the
same cache with the same value objects (identity), and the per-step fields must
be this step's tensors.

Streams mix admissions, finishes, preemptions with resumption, batch-order
swaps, random draft rejections with block growth (STEADY re-records, the lever
re-templates) and padded-row changes. Outputs are held as the served
DraftChunkInputs holds them (released after the next step), unless a test says
otherwise.
"""

from __future__ import annotations

import os
import pathlib
import random
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.block_table import BlockTable

from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.layers.core import attention_metadata as am
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadataBuilder,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.runner import l6_setupfree, l6_steady
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

BS = 128
K = 2
WIDTH = 16
MAXR = 8
CPU = torch.device("cpu")
SHAPES = {
    # the STEADY test's world: 2 attention + 3 mamba groups, two layer names each
    "g5": dict(n_attn=2, n_mamba=3, multi=5),
    # served-shaped: 59 groups, 37 mamba, two groups with two layer names
    "g59": dict(n_attn=22, n_mamba=37, multi=2),
}
ENV_KEYS = (
    "VLLM_TPU_L6_SETUP_FREE",
    "VLLM_TPU_L6_SETUP_FREE_CHECK",
    "VLLM_TPU_L6_STEADY_STEP",
    "VLLM_TPU_L6_STEADY_STEP_CHECK",
    "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_AB_PERIOD_MS",
    "VLLM_TPU_L6_AB_STEPS",
    "VLLM_TPU_L6_AB_CLASSES",
    "VLLM_TPU_L6_AB_CONFIRM",
    "VLLM_TPU_HOST_AB_PERIOD_MS",
    "VLLM_TPU_D1N_AB_PERIOD_MS",
)


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)  # one wall-clock window
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    errors: list[str] = []
    monkeypatch.setattr(l6_setupfree.logger, "error", lambda fmt, *a: errors.append(fmt % a))

    def lever(value=None, check=None, **extra):
        for k in ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("VLLM_TPU_L6_STEADY_STEP", "1")
        if value is not None:
            monkeypatch.setenv("VLLM_TPU_L6_SETUP_FREE", value)
        if check is not None:
            monkeypatch.setenv("VLLM_TPU_L6_SETUP_FREE_CHECK", str(check))
        for k, v in extra.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        l6_setupfree._CHECK[0] = None
        l6_steady._CHECK[0] = None

    def flags(served):
        """FMB on (STEADY caches FMB-built metadata); `served` adds the served decode
        env's staging levers (as tests/q38_l6/test_l6_steady.py _flags)."""
        monkeypatch.setenv("VLLM_TPU_FAST_METADATA_BUILD", "1")
        for name in (
            "VLLM_TPU_BLOCK_TABLE_UPLOAD_REUSE",
            "VLLM_TPU_STAGE_ENTRIES_CACHE",
            "VLLM_TPU_STAGED_WALK_CLASS_GATHER",
        ):
            if served:
                monkeypatch.setenv(name, "1")
            else:
                monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(am, "_STAGE_INC", bool(served))
        monkeypatch.setattr(am, "_STAGE_INC_CHECK", [0])
        monkeypatch.setattr(am, "_STAGE_INC_STATS", {"inc": 0, "full": 0, "rows": 0, "logged": False})
        # On CPU an upload would alias the host scratch; copy, as a device transfer does.
        monkeypatch.setattr(am, "_inc_upload", lambda src, device: src.clone())

    yield NS(lever=lever, flags=flags, errors=errors, monkeypatch=monkeypatch)
    for k in ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()
    l6_setupfree._CHECK[0] = None
    l6_steady._CHECK[0] = None


def counts() -> dict:
    return dict(l6_ab._state["counts"])


# ---------------------------------------------------------------- the decode world


class World:
    """Requests (device length s, host length h) and one real BlockTable per KV
    group; mutations go through BlockTable's methods, as vLLM's InputBatch does
    (tests/q38_l6/test_l6_steady.py World, plus preemption / resumption)."""

    def __init__(self, n_groups):
        self.tables = [BlockTable(BS, MAXR, WIDTH, 64, False, CPU, BS, 1) for _ in range(n_groups)]
        self.reqs = []
        self.parked = []
        self.next_id = 0
        self.next_block = 1

    def _blocks(self, n):
        out = list(range(self.next_block, self.next_block + n))
        self.next_block += n
        return out

    @staticmethod
    def _need(h):
        return -(-(h + K) // BS)

    def admit(self, length, rid=None):
        row = len(self.reqs)
        nb = self._need(length)
        for bt in self.tables:
            bt.add_row(self._blocks(nb), row)
        if rid is None:
            rid = f"r{self.next_id}"
            self.next_id += 1
        self.reqs.append(dict(id=rid, s=length, h=length, nb=nb))

    def finish(self, row):
        last = len(self.reqs) - 1
        for bt in self.tables:
            if row == last:
                bt.clear_row(row)
            else:
                bt.move_row(last, row)
        self.reqs[row] = self.reqs[last]
        self.reqs.pop()

    def preempt(self, row):
        req = self.reqs[row]
        self.parked.append((req["id"], req["s"]))
        self.finish(row)

    def resume(self):
        rid, s = self.parked.pop(0)
        self.admit(max(1, s - 5), rid=rid)  # recompute a 5-token tail

    def swap(self, i, j):
        for bt in self.tables:
            bt.swap_row(i, j)
        self.reqs[i], self.reqs[j] = self.reqs[j], self.reqs[i]

    def advance(self, accepted):
        """One verify step: a request accepting a of K drafts moves s by 1 + a; the
        host assumed all K (h = s_prev + 1 + K). Blocks follow h + K."""
        for row, (req, a) in enumerate(zip(self.reqs, accepted)):
            s0 = req["s"]
            req["s"], req["h"] = s0 + 1 + a, s0 + 1 + K
            need = self._need(req["h"])
            if need > req["nb"]:
                for bt in self.tables:
                    bt.append_row(self._blocks(need - req["nb"]), row)
                req["nb"] = need


def make_runner(world, n_attn, n_mamba, multi):
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.device = CPU
    r.max_num_reqs = MAXR
    r.most_model_len = None
    r._block_table_stage_cpu = None
    r._unified_kv_layout = True
    r.speculative_config = NS(num_speculative_tokens=K)
    r.input_batch = NS(block_table=world.tables, req_ids=[], num_reqs=0)
    attn = FullAttentionSpec(
        block_size=BS, num_kv_heads=2, head_size=128, dtype=torch.bfloat16, page_size_padded=16384
    )
    mamba = MambaSpec(
        block_size=BS, shapes=[(4, 128), (8, 64, 32)], dtypes=[torch.bfloat16, torch.float32],
        page_size_padded=16384,
    )
    groups = []
    for gid in range(n_attn + n_mamba):
        spec = attn if gid < n_attn else mamba
        b = AttentionMetadataBuilder(spec, [f"g{gid}"], MagicMock(), CPU, r, gid)
        names = [f"l{gid}.a"] + ([f"l{gid}.b"] if gid < multi else [])
        groups.append([NS(metadata_builders=[b], layer_names=names)])
    r.attn_groups = groups
    return r


def call(r, world, t, nrp):
    n = len(world.reqs)
    r.input_batch.req_ids = [q["id"] for q in world.reqs]
    r.input_batch.num_reqs = n
    r.position_ids = torch.arange(3 * nrp, dtype=torch.int32) + t
    offsets = r.__dict__.setdefault("_t_offsets", torch.zeros(4 * MAXR, dtype=torch.int32))
    ctx = AttentionMetadataBuilderContext(
        num_reqs=n,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor([q["s"] for q in world.reqs] + [0] * (nrp - n), dtype=torch.int32),
        query_start_loc=torch.arange(nrp + 1, dtype=torch.int32) * 3,
        request_distribution=torch.tensor([n, n, n], dtype=torch.int32) + t % 2,
        mamba_slot_read_offsets=offsets,
        mamba_ckpt_window=K + 1,
        mamba_request_distribution=torch.tensor([n, n, n], dtype=torch.int32),
        unified_mamba_state_indices=[],
        steady_seq_lens_hi=np.array([q["h"] for q in world.reqs], dtype=np.int64),
        steady_max_rejected=K,
    )
    r._attn_metadata_builder_ctx = ctx
    out, unused = r._build_attention_metadata(num_tokens=3 * n, num_reqs=nrp, num_reqs_padded=nrp)
    assert unused is None
    return out, ctx


def _vbytes(v) -> bytes:
    if isinstance(v, torch.Tensor):
        a = v.detach().cpu().contiguous()
        return f"T{a.dtype}{tuple(a.shape)}|".encode() + a.reshape(-1).view(torch.uint8).numpy().tobytes()
    return b"R" + repr(v).encode()


def digest(out, ctx) -> dict[str, bytes]:
    """Everything the forward reads from the metadata, plus the context side effects."""
    ids: dict[int, int] = {}

    def cls(v):
        return ids.setdefault(id(v), len(ids))

    d = {"names": repr(list(out)).encode()}
    objs: dict[int, str] = {}
    for name, md in out.items():
        first = objs.setdefault(id(md), name)
        if first != name:
            d[f"{name}"] = f"shares {first}".encode()
            continue
        items = list(md.__dict__.items())
        d[f"{name}"] = repr((type(md).__name__, [k for k, _ in items], [cls(v) for _, v in items])).encode()
        for k, v in items:
            d[f"{name}.{k}"] = _vbytes(v)
    st = ctx.staged_block_tables or {}
    d["staged"] = repr(sorted(st)).encode() + b"".join(_vbytes(st[g]) for g in sorted(st))
    pr = ctx.staged_mamba_products
    if pr is None:
        d["products"] = b"None"
    else:
        d["products"] = repr(sorted(pr)).encode() + b"".join(
            _vbytes(pr[g][0]) + _vbytes(pr[g][1]) for g in sorted(pr)
        )
    us = ctx.unified_mamba_state_indices
    d["states"] = b"None" if us is None else b"".join(_vbytes(x) for x in us)
    return d


def random_events(rng, world):
    n = len(world.reqs)
    ev = []
    u = rng.random()
    if u < 0.04 and n < MAXR:
        ev.append("admit")
    elif u < 0.07 and world.parked and n < MAXR:
        ev.append("resume")
    elif u < 0.10 and n > 1:
        ev.append("finish")
    elif u < 0.13 and n > 1:
        ev.append("preempt")
    if rng.random() < 0.03 and n > 1:
        ev.append("swap")
    if rng.random() < 0.03:
        ev.append("pad")
    return ev


def run_stream(shape, steps, *, seed, hold="served", served=False, on_step=None, rand=True,
               check_clone=True, check_fields=True):
    """One runner for `steps` steps; returns (per-step digests, runner, world, info).

    hold: "served" keeps step t's output until step t + 1 has run (the
    DraftChunkInputs lifetime); "none" drops it at once."""
    rng = random.Random(seed)
    w = World(shape["n_attn"] + shape["n_mamba"])
    r = make_runner(w, **shape)
    for _ in range(4):
        w.admit(rng.randint(20, 300))
    rows = []
    held = None
    info = NS(fast=0, clone_checked=0)
    pad = False
    for t in range(steps):
        l6_ab.begin_step()
        if t > 0:
            ev = random_events(rng, w) if rand else []
            n = len(w.reqs)
            if "admit" in ev:
                w.admit(rng.randint(20, 300))
            if "resume" in ev:
                w.resume()
            if "finish" in ev:
                w.finish(rng.randrange(n))
            if "preempt" in ev:
                w.preempt(rng.randrange(n))
            if "swap" in ev and len(w.reqs) > 1:
                i, j = rng.sample(range(len(w.reqs)), 2)
                w.swap(i, j)
            if "pad" in ev:
                pad = not pad
            accepted = [rng.randint(0, K) if rand else K for _ in w.reqs]
            for row in reversed(range(len(w.reqs))):
                if w.reqs[row]["s"] + 2 * (1 + K) + K > WIDTH * BS - BS:
                    w.finish(row)
                    accepted.pop(row)
            w.advance(accepted)
        nrp = 4 if pad and len(w.reqs) <= 4 else MAXR
        before = counts().get("steady.fast", 0)
        out, ctx = call(r, w, t, nrp)
        is_fast = counts().get("steady.fast", 0) > before
        d = digest(out, ctx)
        d["fast"] = repr(is_fast).encode()
        rows.append(d)
        if is_fast:
            info.fast += 1
            if check_fields:
                _step_fields(out, r, ctx)
            if check_clone:
                _same_as_clone(out, r, ctx, t)
                info.clone_checked += 1
        if on_step is not None:
            on_step(t, out, r)
        # sample_setup: the previous step's DraftChunkInputs is released.
        if hold == "served":
            held = out
        del out
    held = None
    return rows, r, w, info


def _step_fields(out, r, ctx):
    """Per-step fields are this step's objects (a function: no loop variable
    outlives the step and holds an object)."""
    for name, md in out.items():
        assert md.input_positions is r.position_ids, name
        assert md.seq_lens is ctx.seq_lens, name
        assert md.query_start_loc is ctx.query_start_loc, name
        assert md.request_distribution is ctx.request_distribution, name


def _same_as_clone(out, r, ctx, t):
    """Identity-exact against a fresh STEADY clone of the same cache."""
    ref = l6_steady._clone(r._l6_steady_am, r, ctx)
    why = l6_setupfree._differ(ref, out)
    assert why is None, (t, why)


def first_difference(a, b):
    if len(a) != len(b):
        return ("len", len(a), len(b))
    for i, (x, y) in enumerate(zip(a, b)):
        if x.keys() != y.keys():
            return (i, "keys", sorted(set(x) ^ set(y))[:5])
        for k in x:
            if x[k] != y[k]:
                return (i, k)
    return None


def both(env, shape, steps, *, seed, value="1", check=None, served=False, check_clone=True, **kw):
    """(lever-off digests, lever-on digests, lever-on counts, lever-off counts,
    lever-on runner, lever-on info) over the same seeded stream."""
    env.flags(served)
    env.lever(None)
    base, br, bw, binfo = run_stream(shape, steps, seed=seed, served=served, check_clone=False, **kw)
    bc = counts()
    env.lever(value, check=check)
    lev, lr, lw, linfo = run_stream(shape, steps, seed=seed, served=served, check_clone=check_clone, **kw)
    return base, lev, counts(), bc, lr, linfo


# ---------------------------------------------------------------- lever == STEADY's clone


@pytest.mark.parametrize("served", [False, True], ids=["plain", "served_staging"])
@pytest.mark.parametrize("name", sorted(SHAPES))
def test_pingpong_equals_clone_over_random_streams(env, name, served):
    steps = 300 if name == "g5" else 120
    base, lev, c, bc, r, info = both(env, SHAPES[name], steps, seed=sorted(SHAPES).index(name) + 31,
                                     served=served)
    assert first_difference(base, lev) is None
    assert not any(k.startswith("setupfree.") for k in bc)
    assert info.fast > steps // 3 and info.clone_checked == info.fast
    # Served hold: the set whose turn it is is always free; two fills, the rest reused.
    assert c.get("setupfree.busy", 0) == 0, c
    assert c.get("setupfree.fill", 0) == 2, c
    assert c.get("setupfree.reuse", 0) == info.fast - 2, c
    assert c.get("setupfree.retemplate", 0) > 0, c  # STEADY re-recorded (block growth, batch changes)
    assert c.get("setupfree.reset", 0) == 0 and c.get("setupfree.mismatch", 0) == 0, c


def test_streams_are_not_vacuous(env):
    """Every event kind occurs; fast steps carry most of the stream; full walks
    (STEADY re-records) are interleaved."""
    seen = {"admit": 0, "resume": 0, "finish": 0, "preempt": 0, "swap": 0, "pad": 0}
    real = random_events

    def spy(rng, world):
        ev = real(rng, world)
        for e in ev:
            seen[e] += 1
        return ev

    env.monkeypatch.setitem(globals(), "random_events", spy)
    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 300, seed=33)
    assert first_difference(base, lev) is None
    assert all(v > 0 for v in seen.values()), seen
    assert c.get("steady.full", 0) > 20 and info.fast > 100, (c, info.fast)


def test_no_hold_reuses_the_same_set_in_turns(env):
    """With nothing holding an output (no DraftChunkInputs), the two sets
    alternate and each is rewritten every other fast step; equal to STEADY's clone."""
    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 80, seed=35, hold="none", rand=False)
    assert first_difference(base, lev) is None
    assert c.get("setupfree.busy", 0) == 0 and c.get("setupfree.fill", 0) == 2


def test_served_hold_alternates_the_two_sets(env):
    """Served hold on a steady stream: the kept set handed out alternates
    between the two on every engaged step (never the set still held), with no
    busy step."""
    used: list[int] = []

    def which(t, out, r):
        st = r.__dict__.get("_l6_sf")
        if st is not None:
            idx = [i for i, s in enumerate(st.sets) if s is not None and s.layer is out]
            if idx:
                used.append(idx[0])

    env.flags(False)
    env.lever("1")
    run_stream(SHAPES["g5"], 40, seed=36, rand=False, on_step=which)
    c = counts()
    assert c.get("setupfree.busy", 0) == 0 and c.get("setupfree.reuse", 0) > 15, c
    assert len(used) > 15
    assert all(used[i] != used[i - 1] for i in range(1, len(used))), used


@pytest.mark.parametrize("what", ["object", "dict"])
def test_held_set_is_never_rewritten(env, what):
    """An output held beyond the served lifetime (one object of it, or its
    per-layer dict) for 20 steps: the lever never rewrites it (its values stay
    as handed out), takes STEADY's clone instead (busy) and the stream still
    equals the lever-off stream."""
    held: dict = {}

    def grab(t, out, r):
        if t == 0:
            held.clear()  # each of the two runs grabs its own output
        st = r.__dict__.get("_l6_sf")
        kept = st is not None and any(s is not None and s.layer is out for s in st.sets)
        if "t0" not in held and t >= 10 and (kept or "VLLM_TPU_L6_SETUP_FREE" not in os.environ):
            held["t0"] = t
            mds = [out[next(iter(out))]] if what == "object" else list(out.values())
            held["x"] = mds[0] if what == "object" else out
            held["snap"] = [(md, list(md.__dict__.items())) for md in mds]
        elif "snap" in held:
            for md, items in held["snap"]:
                now = list(md.__dict__.items())
                assert [k for k, _ in now] == [k for k, _ in items], t
                assert all(a is b for (_, a), (_, b) in zip(now, items)), (t, "held metadata rewritten")
            if t == held["t0"] + 20:
                held.pop("snap")
                held.pop("x")

    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 45, seed=37, rand=False, on_step=grab)
    assert first_difference(base, lev) is None
    assert "t0" in held
    assert c.get("setupfree.busy", 0) >= 3, c  # the held set's turns while held
    assert c.get("setupfree.reuse", 0) > 5, c


def test_probe_counts_match_a_free_set(env):
    """The guard's expectation: a kept set that nothing else references reads
    exactly its expected counts; one extra reference to any object or to the
    dict changes them."""
    env.flags(False)
    env.lever("1")
    w = World(5)
    r = make_runner(w, **SHAPES["g5"])
    for length in (10, 150, 400):
        w.admit(length)
    for t in range(2):
        l6_ab.begin_step()
        out, ctx = call(r, w, t, MAXR)
    l6_ab.begin_step()
    out, ctx = call(r, w, 2, MAXR)  # second fast step: set 1 filled
    del out
    st = r._l6_sf
    s0, s1 = st.sets
    assert s0 is not None and s1 is not None
    assert l6_setupfree._counts(s1) == s1.want
    assert l6_setupfree._counts(s0) == s0.want
    x = s0.objs[3]
    assert l6_setupfree._counts(s0) != s0.want
    del x
    assert l6_setupfree._counts(s0) == s0.want
    y = s0.layer
    assert l6_setupfree._counts(s0) != s0.want
    del y
    assert l6_setupfree._counts(s0) == s0.want


def test_group_structure_change_resets(env):
    """A new group plan (a layer name added to a group): the kept sets are dropped
    (reset) and rebuilt; every step equals STEADY's clone."""

    def regroup(t, out, r):
        if t == 15:
            g = r.attn_groups[3][0]
            r.attn_groups = [list(x) for x in r.attn_groups]
            r.attn_groups[3] = [NS(metadata_builders=g.metadata_builders, layer_names=g.layer_names + ["l3.c"])]

    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 40, seed=38, rand=False, on_step=regroup)
    assert first_difference(base, lev) is None
    assert c.get("setupfree.reset", 0) == 1 and c.get("setupfree.fill", 0) == 4, c


def test_steady_drop_releases_the_kept_sets(env):
    """STEADY's drop (its lever off, or a STEADY CHECK mismatch) also forgets the
    kept sets, so no old step's tensors stay referenced."""
    env.flags(False)
    env.lever("1")
    _, r, _, info = run_stream(SHAPES["g5"], 8, seed=45, rand=False)
    assert info.fast > 3 and "_l6_sf" in r.__dict__
    l6_steady.drop(r)
    assert "_l6_sf" not in r.__dict__


def test_lever_off_takes_the_clone(env):
    """Unset or '0': the lever's clone never runs and nothing is kept; turning
    the lever off mid-run drops the kept sets on the next fast step."""
    forbid = {"on": True}
    real = l6_setupfree.clone

    def guarded(*a, **k):
        if forbid["on"]:
            raise AssertionError("setupfree ran with the lever off")
        return real(*a, **k)

    env.monkeypatch.setattr(l6_setupfree, "clone", guarded)
    env.flags(False)
    for value in (None, "0"):
        env.lever(value)
        _, r, _, info = run_stream(SHAPES["g5"], 20, seed=39, check_clone=False)
        assert info.fast > 5
        assert "_l6_sf" not in r.__dict__
        assert not any(k.startswith("setupfree.") for k in counts())
    forbid["on"] = False
    env.lever("1")
    state = {}

    def flip(t, out, r):
        if t == 10:
            state["had"] = "_l6_sf" in r.__dict__
            env.monkeypatch.delenv("VLLM_TPU_L6_SETUP_FREE")
            l6_ab._resolve()
            forbid["on"] = True
        if t == 10:
            state["fast10"] = counts().get("steady.fast", 0)
        if t == 24:
            state["fast_after"] = counts().get("steady.fast", 0) > state["fast10"]
            state["after"] = "_l6_sf" in r.__dict__

    run_stream(SHAPES["g5"], 25, seed=40, rand=False, on_step=flip, check_clone=False)
    assert state["had"] is True and state["fast_after"] is True and state["after"] is False, state


def test_ab_windows_switch_the_lever(env):
    """ab:1 with step windows of 6: the in-place rewrite runs only in class-1
    windows and every step equals the lever-off stream."""
    env.flags(False)
    env.lever(None)
    base, _, _, _ = run_stream(SHAPES["g5"], 60, seed=41, check_clone=False)
    env.lever("ab:1", VLLM_TPU_L6_AB_STEPS="6", VLLM_TPU_L6_AB_CLASSES="2", VLLM_TPU_L6_AB_CONFIRM="2")
    calls: list[int] = []
    real = l6_setupfree.clone

    def spy(*a, **k):
        calls.append(l6_ab._state["j"])
        return real(*a, **k)

    env.monkeypatch.setattr(l6_setupfree, "clone", spy)
    lev, _, _, _ = run_stream(SHAPES["g5"], 60, seed=41)
    assert first_difference(base, lev) is None
    assert calls and all((j // 6) % 2 == 1 for j in calls), calls
    assert len(calls) >= 10


def test_check_mode_compares_and_finds_no_mismatch(env):
    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 80, seed=42, check=1)
    assert first_difference(base, lev) is None
    assert c.get("setupfree.check", 0) == c.get("setupfree.reuse", 0) > 20, c
    assert c.get("setupfree.mismatch", 0) == 0 and not env.errors
    assert not r.__dict__.get("_l6_sf_off", False)


def _break_rewrite(mp):
    """The rewrite writes another step tensor into seq_lens."""
    real = l6_setupfree._steps

    def bad(runner, ctx):
        a, m = real(runner, ctx)
        a = dict(a, seq_lens=ctx.query_start_loc)
        m = dict(m, seq_lens=ctx.query_start_loc)
        return a, m

    mp.setattr(l6_setupfree, "_steps", bad)


def test_check_catches_a_broken_rewrite(env):
    """A broken rewrite under CHECK=1: the first rewritten step is caught, the
    clone is handed out, one ERROR is logged and the lever turns off, so the
    whole stream still equals the lever-off stream."""
    _break_rewrite(env.monkeypatch)
    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 40, seed=43, check=1, check_clone=False)
    assert first_difference(base, lev) is None
    assert c.get("setupfree.mismatch") == 1, c
    assert len(env.errors) == 1 and "CHECK mismatch" in env.errors[0]
    assert r._l6_sf_off is True and "_l6_sf" not in r.__dict__


def test_broken_rewrite_diverges_without_check(env):
    """The same broken rewrite without CHECK makes the stream differ (the
    digest comparison has teeth on its own: the identity checks are off here)."""
    _break_rewrite(env.monkeypatch)
    base, lev, c, bc, r, info = both(env, SHAPES["g5"], 40, seed=43, check_clone=False, check_fields=False)
    assert first_difference(base, lev) is not None


def test_check_steps_bound_the_check(env):
    """VLLM_TPU_L6_CHECK_STEPS=8 with CHECK=1: compares only in the first 8 worker steps."""
    env.flags(False)
    env.lever("1", check=1, VLLM_TPU_L6_CHECK_STEPS="8")
    seen: list[int] = []
    real = l6_setupfree._differ

    def spy(*a, **k):
        seen.append(l6_ab._state["j"])
        return real(*a, **k)

    env.monkeypatch.setattr(l6_setupfree, "_differ", spy)
    run_stream(SHAPES["g5"], 30, seed=44, rand=False, check_clone=False)
    assert seen and max(seen) <= 8 and len(seen) >= 3, seen


# ---------------------------------------------------------------- registration / staging


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["setupfree"] == "VLLM_TPU_L6_SETUP_FREE"
    for name in ("VLLM_TPU_L6_SETUP_FREE", "VLLM_TPU_L6_SETUP_FREE_CHECK"):
        assert name in envs.environment_variables
        assert name in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    reg = pathlib.Path(__file__).resolve().parents[1] / "compilation" / "test_tpu_compile_key_registration.py"
    text = reg.read_text()
    assert '("VLLM_TPU_L6_SETUP_FREE", "1", "")' in text
    assert '("VLLM_TPU_L6_SETUP_FREE_CHECK", "200", 0)' in text


def test_import_resolves_to_the_tree_under_test():
    """Guard against a staging that silently imports the pod's installed package."""
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
