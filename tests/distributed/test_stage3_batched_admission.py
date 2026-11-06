# SPDX-License-Identifier: Apache-2.0
"""QWEN4_EXP_PD_BATCHED_ADMISSION exactness: four CPU gloo ranks, randomized and
hand-built passes of parked and new loads (mixed sizes, prefix-aware skips,
deadlines expired on no rank, one rank or all ranks, ranks whose local free maps
differ). On every rank the batched pass must make the same admission decisions,
in the same order, with the same pages, the same expiry outcomes and the same
leftover state as today's per-load pass, with one collective instead of
1 + (loads that do not fit) per pass; and every rank must agree.

The per-load reference is the real code: ``StagedConsumer.reserve`` (pd_staging)
with the TP group, and ``_stage3_destination_wait_expired`` extracted verbatim
from tpu_connector.py (that module needs the TPU runtime to import). The loop
around the two calls mirrors ``_submit_stage3_loads``.
"""

import ast
import copy
import os
import random
import sys
import tempfile
import textwrap
import time
import types

import pytest
import torch
import torch.multiprocessing as mp

try:  # the Mac CPU environment has no vLLM: stand-ins for the names imported here
    import vllm.v1.kv_cache_interface  # noqa: F401
    import vllm.model_executor.models.config  # noqa: F401
except ImportError:
    def _stub(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules.setdefault(name, m)
        return sys.modules[name]

    _stub("vllm"); _stub("vllm.v1")
    _stub("vllm.v1.kv_cache_interface", CircularBufferSpec=object, FullAttentionSpec=object,
          KVCacheGroupSpec=object, MambaSpec=object, UniformTypeKVCacheSpecs=object)
    import vllm_torchtpu  # noqa: F401
    _pkg = types.ModuleType("vllm_torchtpu.models.vllm")
    _pkg.__path__ = [os.path.join(os.path.dirname(vllm_torchtpu.__file__), "models", "vllm")]
    sys.modules.setdefault("vllm_torchtpu.models.vllm", _pkg)

from vllm_torchtpu.distributed.kv_transfer import stage3_admission as adm  # noqa: E402
from vllm_torchtpu.models.vllm.qwen4_exp import pd_staging  # noqa: E402
from vllm_torchtpu.models.vllm.qwen4_exp.pd import TransferExtent, TransferGroup  # noqa: E402

WORLD = 4
BUDGET = 150.0  # 0.5 x the stubbed p2p wait-pull timeout
NOW = 1_000_000.0
SRC = os.path.join(os.path.dirname(pd_staging.__file__), "..", "..", "..", "distributed",
                   "kv_transfer", "tpu_connector.py")


def _wait_expired_fn():
    """tpu_connector.TPURaidenConnectorWorker._stage3_destination_wait_expired, verbatim."""
    src = open(SRC).read()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "_stage3_destination_wait_expired":
            code = textwrap.dedent(ast.get_source_segment(src, node))
            ns = {"time": types.SimpleNamespace(time=lambda: NOW), "torch": torch,
                  "dist_utils": types.SimpleNamespace(get_p2p_wait_pull_timeout=lambda: 2 * BUDGET),
                  "logger": types.SimpleNamespace(warning=lambda *a, **k: None), "Any": object}
            exec(code, ns)
            return ns["_stage3_destination_wait_expired"]
    raise AssertionError("_stage3_destination_wait_expired not found")


class Meta(types.SimpleNamespace):
    pass


def _staging(free):
    s = object.__new__(pd_staging.StagedConsumer)
    s.src_groups = [TransferGroup(0, ("fa",), 16, TransferExtent.TOKEN_PAGES),
                    TransferGroup(1, ("gdn",), 16, TransferExtent.REQUEST_STATE)]
    s.manifest = types.SimpleNamespace(pools=[types.SimpleNamespace(layer_name=n) for n in ("fa0", "gdn0", "fa1")])
    s._src_owner = {"fa0": 0, "gdn0": 1, "fa1": 0}
    s.free = free.clone()
    s.reservations, s.skips = {}, {}
    return s


def _scenario(seed, rank):
    """(free map, waits, loads) for one pass; loads and sizes identical on every rank,
    local free maps and deadlines rank-specific."""
    g = random.Random(seed)
    pages = g.choice([24, 48, 96])
    free = torch.ones(pages, dtype=torch.int32)
    free[0] = 0
    for p in g.sample(range(1, pages), g.randint(0, pages // 3)):  # held on every rank
        free[p] = 0
    r = random.Random(seed * 131 + rank)
    for p in r.sample(range(1, pages), r.randint(0, pages // 6)):  # not yet freed on this rank
        free[p] = 0
    loads, waits = [], {}
    for i in range(g.randint(1, 10)):
        tokens = g.choice([16, 40, 64, 100, 160, 200, 320])
        skip = g.choice([0, 0, 0, 16, 32]) if tokens > 48 else 0
        dest = f"d{seed}_{i}"
        meta = Meta(uuid=seed * 100 + i, num_tokens=tokens, skip_tokens=skip)
        kind = g.choice(["new", "parked", "parked_one_rank_expired", "parked_all_expired"])
        if kind == "parked":
            waits[dest] = (meta, NOW + g.uniform(1, 100))
        elif kind == "parked_one_rank_expired":
            victim = g.randrange(WORLD)
            waits[dest] = (meta, NOW - 1 if rank == victim else NOW + 50)
        elif kind == "parked_all_expired":
            waits[dest] = (meta, NOW - g.uniform(0, 10))
        loads.append((dest, meta))
    # _submit_stage3_loads order: parked first (insertion order), then new ones
    order = [(d, m) for d, m in loads if d in waits] + [(d, m) for d, m in loads if d not in waits]
    return free, waits, order


def _exact_fit(rank):
    """Hand-built: after the first load, exactly as many pages are free (on every
    rank's MIN map) as the second load needs; the third cannot fit."""
    free = torch.ones(16, dtype=torch.int32)
    free[0] = 0
    free[14 + rank % 2] = 0  # ranks hold pages 14 or 15: the MIN map has 13 free
    m = [Meta(uuid=900 + i, num_tokens=t, skip_tokens=0) for i, t in enumerate((48, 64, 16))]
    # needs: 48 tokens -> 3+1+3 = 7 pages (6 left); 64 -> 4+1+4 = 9 (no fit); 16 -> 1+1+1 = 3 (3 left)
    m2 = Meta(uuid=905, num_tokens=16, skip_tokens=0)  # 3 pages: exactly the 3 left (S2 must not skip it)
    loads = [("e0", m[0]), ("e1", m[1]), ("e2", m[2]), ("e3", m2), ("e4", Meta(uuid=906, num_tokens=16, skip_tokens=0))]
    return free, {}, loads


def _per_load(free, waits, loads, group, wait_expired):
    st = _staging(free)
    self = types.SimpleNamespace(_stage3_destination_waits=dict(waits))
    out = []
    for dest, meta in loads:  # the loop's two calls
        if st.reserve(meta.uuid, meta.num_tokens, group, skip_tokens=meta.skip_tokens):
            self._stage3_destination_waits.pop(dest, None)
            out.append((dest, "reserved", st.reservations[meta.uuid]))
        elif not wait_expired(self, dest, meta, group):
            out.append((dest, "parked", None))
        else:
            out.append((dest, "expired", None))
    return out, st, self._stage3_destination_waits


def _batched(free, waits, loads, group):
    st = _staging(free)
    w = dict(waits)
    a = adm.prepare(loads, st.free, w, BUDGET, NOW, group)
    out = []
    for dest, meta in loads:
        ok = a.reserve(dest, meta,
                       lambda m, rf: st.reserve(m.uuid, m.num_tokens, skip_tokens=m.skip_tokens, reduced_free=rf),
                       lambda m: st.pages_needed(m.uuid, m.num_tokens, skip_tokens=m.skip_tokens))
        if ok:
            w.pop(dest, None)
            out.append((dest, "reserved", st.reservations[meta.uuid]))
        elif not a.expired(dest, meta, w):
            out.append((dest, "parked", None))
        else:
            out.append((dest, "expired", None))
    return out, st, w, a.collectives


def _rank(rank, init_file, seeds, q):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=WORLD)
    group = types.SimpleNamespace(world_size=WORLD, cpu_group=dist.group.WORLD)
    wait_expired = _wait_expired_fn()
    calls = {"n": 0}
    real = dist.all_reduce

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    dist.all_reduce = counting
    fails, stats = [], {"passes": 0, "per_load_collectives": 0, "batched_collectives": 0,
                        "expired": 0, "parked": 0, "reserved": 0}
    for seed in seeds:
        free, waits, loads = _exact_fit(rank) if seed == -1 else _scenario(seed, rank)
        c0 = calls["n"]
        ref, st_r, w_r = _per_load(free, waits, loads, group, wait_expired)
        c1 = calls["n"]
        got, st_b, w_b, coll = _batched(free, waits, loads, group)
        c2 = calls["n"]
        stats["passes"] += 1
        stats["per_load_collectives"] += c1 - c0
        stats["batched_collectives"] += c2 - c1
        for _, what, _ in ref:
            stats[what] += 1
        if ref != got:
            fails.append((seed, "decisions", ref, got))
        if not torch.equal(st_r.free, st_b.free) or st_r.reservations != st_b.reservations or st_r.skips != st_b.skips:
            fails.append((seed, "staging state"))
        if {k: v[1] for k, v in w_r.items()} != {k: v[1] for k, v in w_b.items()}:
            fails.append((seed, "waits"))
        if c2 - c1 != (1 if loads else 0) or coll != (1 if loads else 0):
            fails.append((seed, "collectives", c2 - c1))
        every = [None] * WORLD
        dist.all_gather_object(every, [(d, w, p) for d, w, p in got])
        if any(e != every[0] for e in every):
            fails.append((seed, "ranks disagree"))
        if seed == -1:
            want = ["reserved", "no_fit", "reserved", "reserved", "no_fit"]
            if [w if w == "reserved" else "no_fit" for _, w, _ in got] != want:
                fails.append((seed, "exact-fit expectation", got))
    dist.barrier()
    q.put((rank, fails, stats))
    dist.destroy_process_group()


def _run(seeds):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    with tempfile.TemporaryDirectory() as d:
        init = os.path.join(d, "init")
        procs = [ctx.Process(target=_rank, args=(r, init, seeds, q)) for r in range(WORLD)]
        for p in procs:
            p.start()
        res = [q.get(timeout=600) for _ in procs]
        for p in procs:
            p.join(timeout=60)
    return sorted(res)


SEEDS = [-1] + list(range(1, 81))


def test_batched_admission_matches_per_load_on_every_rank():
    res = _run(SEEDS)
    for rank, fails, stats in res:
        assert not fails, (rank, fails[:3])
    stats = res[0][2]
    # The scenarios exercise every branch, including one-rank expiry (the MAX).
    assert stats["reserved"] > 50 and stats["parked"] > 20 and stats["expired"] > 10, stats
    assert stats["batched_collectives"] == len(SEEDS)
    assert stats["per_load_collectives"] > 2 * stats["batched_collectives"], stats


def test_ab_mode_flips_off_first():
    seq = [adm.ab_mode(p, 3, False) for p in range(1, 13)]
    assert seq == [False] * 3 + [True] * 3 + [False] * 3 + [True] * 3
    assert adm.ab_mode(5, 0, True) and not adm.ab_mode(5, 0, False)


def test_no_group_no_collective():
    free = torch.ones(8, dtype=torch.int32)
    a = adm.prepare([("x", Meta(uuid=1, num_tokens=16, skip_tokens=0))], free, {}, BUDGET, NOW, None)
    assert a.collectives == 0 and a.not_expired == {"x": True}


def _connector_methods(names):
    """Methods of TPURaidenConnectorWorker extracted verbatim from tpu_connector.py."""
    src = open(SRC).read()
    found = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names and node.name not in found:
            found[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
    assert set(found) == set(names), set(names) - set(found)
    return found


class _LoadMeta(types.SimpleNamespace):
    pass


def test_connector_pass_wiring(monkeypatch):
    """The connector side, real code: candidates are exactly the loads the loop sends
    to admission; flag off and no A/B builds nothing; the A/B flips off -> on every N
    passes; the inline check compares the batched decisions with the per-load ones."""
    names = ["_stage3_batched_admission", "_stage3_admission_period_note",
             "_stage3_admission_after_pass", "_stage3_admission_compare", "_stage3_admission_reference"]
    envs_ns = types.SimpleNamespace(QWEN4_EXP_PD_BATCHED_ADMISSION=False,
                                    QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES=0,
                                    QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK=0)
    logs = []
    ns = {"time": time, "torch": torch, "Any": object, "tpu_envs": envs_ns, "_Stage3LoadMeta": _LoadMeta,
          "dist_utils": types.SimpleNamespace(get_p2p_wait_pull_timeout=lambda: 2 * BUDGET),
          "logger": types.SimpleNamespace(info=lambda *a: logs.append(a[0] % a[1:]), warning=lambda *a, **k: None)}
    W = type("W", (), {})
    for n, code in _connector_methods(names).items():
        exec(code, ns)
        setattr(W, n, ns[n])
    st = _staging(torch.tensor([0] + [1] * 15, dtype=torch.int32))
    w = W()
    w.tp_rank = 0
    w._stage3_destination_waits = {}
    w._stage3_submitted_loads = {"done": 7}
    w._stage3_admission_free_map = lambda: st.free
    w._stage3_admission_snapshot = lambda: (st.free.clone(), dict(st.reservations), dict(st.skips))

    def restore(saved):
        st.free.copy_(saved[0]); st.reservations.clear(); st.reservations.update(saved[1])
        st.skips.clear(); st.skips.update(saved[2])
    w._stage3_admission_restore = restore
    w._stage3_reserve_destination = lambda d, m, g: st.reserve(m.uuid, m.num_tokens, g, skip_tokens=0)
    w._stage3_destination_wait_expired = lambda d, m, g: False
    loads = [("a", _LoadMeta(uuid=1, num_tokens=48, release_only=False, fail_only=False)),
             ("rel", _LoadMeta(uuid=2, num_tokens=16, release_only=True, fail_only=False)),
             ("fail", _LoadMeta(uuid=3, num_tokens=16, release_only=False, fail_only=True)),
             ("done", _LoadMeta(uuid=7, num_tokens=16, release_only=False, fail_only=False)),
             ("b", _LoadMeta(uuid=4, num_tokens=64, release_only=False, fail_only=False)),
             ("odd", object())]
    assert W._stage3_batched_admission(w, loads, None) is None  # default: today's path
    assert getattr(w, "_stage3_pass_meta", None) is None
    envs_ns.QWEN4_EXP_PD_BATCHED_ADMISSION = True
    envs_ns.QWEN4_EXP_PD_BATCHED_ADMISSION_CHECK = 1
    a = W._stage3_batched_admission(w, loads, None)
    assert list(a.not_expired) == ["a", "b"]  # only what reaches admission
    assert a.reference == [("a", "reserved"), ("b", "parked")]
    assert torch.equal(st.free, torch.tensor([0] + [1] * 15, dtype=torch.int32)) and not st.reservations
    ok_a = a.reserve("a", loads[0][1], lambda m, rf: st.reserve(m.uuid, m.num_tokens, reduced_free=rf),
                     lambda m: st.pages_needed(m.uuid, m.num_tokens))
    ok_b = a.reserve("b", loads[4][1], lambda m, rf: st.reserve(m.uuid, m.num_tokens, reduced_free=rf),
                     lambda m: st.pages_needed(m.uuid, m.num_tokens))
    assert (ok_a, ok_b) == (True, False)
    a.expired("b", loads[4][1], w._stage3_destination_waits)
    W._stage3_admission_after_pass(w, a)
    assert any("compared 2 mismatches 0" in m for m in logs), logs
    # Positive control: the check reports a decision that differs.
    a.reference = [("a", "parked"), ("b", "parked")]
    w._stage3_admission_checks_left = 1
    W._stage3_admission_compare(w, a)
    assert any("compared 2 mismatches 1" in m for m in logs), logs
    # A/B: off for 2 passes, on for 2, off ... counted per pass
    envs_ns.QWEN4_EXP_PD_BATCHED_ADMISSION, envs_ns.QWEN4_EXP_PD_BATCHED_ADMISSION_AB_PASSES = False, 2
    w._stage3_admission_passes = 0
    modes = []
    for _ in range(6):
        modes.append(W._stage3_batched_admission(w, loads[4:5], None) is not None)
        W._stage3_admission_after_pass(w, None)
    assert modes == [False, False, True, True, False, False]
    assert sum("Stage-3 batched admission ON from pass 3" in m for m in logs) == 1
