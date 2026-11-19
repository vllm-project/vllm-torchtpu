# SPDX-License-Identifier: Apache-2.0
"""q38 L6 seed family: the steady seed loop (H0-2 item 6) and setupfree (T-5(b)): CPU microbench.

CDK CPU pod only (never the Mac or the VM). Usage, from the tree root:
  python3 tests/q38_l6/bench_l6_seed.py [--n 4,9,11,25,52] [--steps 400] [--warm 40] [--reps 3]
      [--parts seed,setup]

Part ``seed`` -- the steady seed loop: the per-request loop of
``TPUModelRunner._collect_mamba_state_seed_copies`` (the state-column /
crossing scan, PR-0 sub-marks ``~pi_seed_sync`` / ``~pi_collect``).
  * In situ: the real ``_prepare_inputs`` driven through the CARRY harness
    (l6_carry_harness.py; async scheduling, MTP k = 3, POSTFWD_FUSED, compact
    mamba slots, decode request bucket 64) with STEADY + CARRY on and a mamba
    align-mode seed plan (2 groups, 512-token state column, checkpoint window
    4), timing each ``_collect_mamba_state_seed_copies`` call (host time;
    the upload stand-ins are excluded). Arm ``sc`` (STEADY + CARRY); the
    ``sv`` / ``ab`` arms run only on a tree that registers a ``seedvec``
    lever. Streams ``steady`` and ``churn`` (a join / leave every 25 steps).
  * Fragments: the stock loop (a verbatim copy) against candidate rewrites on
    the runner state after a steady run, at a steady step (no crossing) and at
    an edge step (one request straddles a column edge: the copy wait and the
    rejected-count correction run).
  * Outcome (job j-b4e57118, b1): the stock loop is 15.9 us at 25 requests
    and the best rewrite saves 1.5 us there (numpy loses below ~40 requests),
    under the ~10 us bar, so no seedvec lever was built; this part stays as
    the record.

Part ``setup`` -- T-5 sample_setup: what STEADY's fast path allocates per step
(``l6_steady._clone``: one AttentionMetadata per KV group and the per-layer
dict) and what dies when the previous step's DraftChunkInputs is released at
sample_setup. A served-shaped world (59 KV groups, 37 of them mamba, 64 request
rows, real vLLM BlockTables and real metadata builders, as
tests/q38_l6/test_l6_steady.py) runs the real ``_build_attention_metadata``;
each output is held for one step (as the DraftChunkInputs holds it) and
released after the next call (timed: the sample_setup release). Arms:
``clone`` (STEADY's clone) and, when the tree registers it, ``pp``
(VLLM_TPU_L6_SETUP_FREE=1). Fragments: clone vs in-place rewrites, the release
of a 59-object set, and a DraftChunkInputs release with and without hidden
tensors (CPU tensors: a TPU buffer free is not modelled here).

GC is off inside timed loops. Last line: ``BENCH_DONE ok=<bool>``.
"""

from __future__ import annotations

import argparse
import gc
import itertools
import os
import statistics
import sys
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import l6_carry_harness as H  # noqa: E402

from vllm_torchtpu.core import l6_ab  # noqa: E402
from vllm_torchtpu.runner import tpu_runner  # noqa: E402
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID  # noqa: E402

PERF = time.perf_counter_ns
PADS = (16, 32, 64, 128, 256, 512)
MAX_NUM_REQS = 64
MAXLEN = 4096
COL = 512
WINDOW = 4
SEED_ARMS = {
    "sc": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1"},
    "sv": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1", "VLLM_TPU_L6_SEED_VEC": "1"},
    "ab": {"VLLM_TPU_L6_STEADY_STEP": "1", "VLLM_TPU_L6_DECODE_CARRY": "1", "VLLM_TPU_L6_SEED_VEC": "ab:1",
           "VLLM_TPU_L6_AB_STEPS": "50", "VLLM_TPU_L6_AB_CLASSES": "2"},
}
ENV_KEYS = sorted({k for v in SEED_ARMS.values() for k in v} | {
    "VLLM_TPU_L6_SETUP_FREE", "VLLM_TPU_L6_SETUP_FREE_CHECK",
    "VLLM_TPU_L6_STEADY_STEP_CHECK", "VLLM_TPU_L6_DECODE_CARRY_CHECK", "VLLM_TPU_L6_CHECK_STEPS",
    "VLLM_TPU_L6_AB_PERIOD_MS", "VLLM_TPU_L6_AB_CONFIRM"})


def set_env(env):
    for k in ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update(env)
    l6_ab._resolve()
    from vllm_torchtpu.runner import l6_steady

    l6_steady._CHECK[0] = None
    try:
        from vllm_torchtpu.runner import l6_setupfree

        l6_setupfree._CHECK[0] = None
    except ImportError:
        pass


def tmin(fn, reps=7, loops=300):
    """Best-of-reps mean us per call."""
    best = None
    for _ in range(reps):
        t0 = PERF()
        for _ in range(loops):
            fn()
        dt = (PERF() - t0) / loops / 1e3
        best = dt if best is None or dt < best else best
    return best


# ============================================================ part seed


class BT:
    def __init__(self, t):
        self.t = t

    def get_cpu_tensor(self):
        return self.t


def add_seed_plan(r, *, col=COL, window=WINDOW, groups=2, seed=0):
    """A mamba align-mode seed plan on a harness runner (make_runner leaves it empty)."""
    rng = np.random.default_rng(77 + seed)
    rows = r.input_batch.token_ids_cpu.shape[0]
    width = -(-MAXLEN // col) + window + 1
    nblk = 1 + rows * width * groups
    r._mamba_copy_plan = [(g, [torch.zeros(1)]) for g in range(groups)]
    r._mamba_state_block_size = col
    r._mamba_state_pos = {}
    r._pool_block_split = 1
    r._pending_mamba_state_copies = []
    r._mamba_ckpt_window = window
    r.mamba_slot_read_offsets = torch.zeros(nblk + 1, dtype=torch.int32)
    ids = rng.permutation(np.arange(1, nblk, dtype=np.int64))[: rows * width * groups]
    r.input_batch.block_table = [
        BT(torch.from_numpy(ids[g * rows * width : (g + 1) * rows * width].reshape(rows, width).astype(np.int32)))
        for g in range(groups)
    ]
    return r


def stock_loop(self, scheduler_output, start_index, num_reqs, col_stride, previous, pos):
    """Verbatim copy of the per-request loop of _collect_mamba_state_seed_copies
    (b488e1cb0 tpu_runner.py:2470-2505) on `pos` (the state-column map)."""
    num_computed = self.input_batch.num_computed_tokens_cpu
    req_ids = self.input_batch.req_ids
    crossings = []
    sampled_cpu = None
    for i in range(num_reqs):
        row = start_index + i
        req_id = req_ids[row]
        assert req_id is not None
        computed = int(num_computed[row])
        scheduled = scheduler_output.num_scheduled_tokens[req_id]
        end_position = computed + scheduled - 1
        if previous is not None and previous.num_draft_per_req:
            previous_row = previous.req_id_to_index_copy.get(req_id)
            drafts = (
                previous.num_draft_per_req.get(previous_row, 0)
                if previous_row is not None
                else 0
            )
            if end_position // col_stride != (end_position - drafts) // col_stride:
                if sampled_cpu is None:
                    sampled_cpu = previous.wait_for_copy()
                    assert sampled_cpu is not None
                assert previous_row is not None
                accepted = int(
                    (sampled_cpu[previous_row] != INVALID_TOKEN_ID).sum().item()
                )
                rejected = max(0, drafts + 1 - accepted)
                computed -= rejected
                end_position -= rejected
        curr = end_position // col_stride
        prev = pos.get(req_id, (computed - 1) // col_stride)
        pos[req_id] = curr
        if 0 <= prev != curr:
            crossings.append((row, prev, curr))
    return crossings


def cand_vec(self, so, start, n, col, previous, pos):
    """Candidate V1: numpy over the chunk, map() for the dict reads."""
    ids = self.input_batch.req_ids[start : start + n]
    nst = so.num_scheduled_tokens
    computed = self.input_batch.num_computed_tokens_cpu[start : start + n].astype(np.int64)
    end = computed + np.fromiter(map(nst.__getitem__, ids), np.int64, n) - 1
    if previous is not None and previous.num_draft_per_req:
        ndr = previous.num_draft_per_req
        prows = list(map(previous.req_id_to_index_copy.get, ids))
        drafts = np.fromiter(map(ndr.get, prows, itertools.repeat(0)), np.int64, n)
        need = np.flatnonzero(end // col != (end - drafts) // col)
        if need.size:
            sampled = previous.wait_for_copy()
            for i in need.tolist():
                acc = int((sampled[prows[i]] != INVALID_TOKEN_ID).sum().item())
                rej = max(0, int(drafts[i]) + 1 - acc)
                computed[i] -= rej
                end[i] -= rej
    curr = (end // col).tolist()
    prev = list(map(pos.get, ids, ((computed - 1) // col).tolist()))
    pos.update(zip(ids, curr))
    if prev == curr:
        return []
    return [(start + i, p, c) for i, (p, c) in enumerate(zip(prev, curr)) if 0 <= p != c]


def cand_lean(self, so, start, n, col, previous, pos):
    """Candidate V2: plain Python lists (no numpy)."""
    ids = self.input_batch.req_ids[start : start + n]
    nst = so.num_scheduled_tokens
    comp = self.input_batch.num_computed_tokens_cpu[start : start + n].tolist()
    end = [c + s - 1 for c, s in zip(comp, map(nst.__getitem__, ids))]
    if previous is not None and previous.num_draft_per_req:
        ndr = previous.num_draft_per_req
        prows = list(map(previous.req_id_to_index_copy.get, ids))
        drafts = list(map(ndr.get, prows, itertools.repeat(0)))
        need = [i for i, (e, d) in enumerate(zip(end, drafts)) if e // col != (e - d) // col]
        if need:
            sampled = previous.wait_for_copy()
            for i in need:
                acc = int((sampled[prows[i]] != INVALID_TOKEN_ID).sum().item())
                rej = max(0, drafts[i] + 1 - acc)
                comp[i] -= rej
                end[i] -= rej
    curr = [e // col for e in end]
    prev = list(map(pos.get, ids, [(c - 1) // col for c in comp]))
    pos.update(zip(ids, curr))
    if prev == curr:
        return []
    return [(start + i, p, c) for i, (p, c) in enumerate(zip(prev, curr)) if 0 <= p != c]


class SeedRec:
    def __init__(self):
        self.excl = 0


SREC = SeedRec()


class TimedUploads(H.Uploads):
    def __enter__(self):
        self._had = "to" in torch.Tensor.__dict__
        self._prev = torch.Tensor.__dict__.get("to")
        orig = torch.Tensor.to
        me = self

        def to(t, *args, **kwargs):
            t0 = PERF()
            out = orig(t, *args, **kwargs)
            dev = args[0] if args else kwargs.get("device")
            if isinstance(dev, torch.device):
                me.n += 1
                if out is t:
                    out = t.clone()
            SREC.excl += PERF() - t0
            return out

        torch.Tensor.to = to
        return self


def seed_events(stream, step):
    if stream == "churn" and step > 0 and step % 25 == 0:
        return ("join",) if (step // 25) % 2 else ("leave",)
    return ()


def run_seed(n, arm, stream, steps, warm, seed):
    set_env(SEED_ARMS[arm])
    r = H.make_runner(nreq=n, max_num_reqs=MAX_NUM_REQS, paddings=PADS, mamba=True, comb=True,
                      max_len=MAXLEN, seed=seed)
    add_seed_plan(r, seed=seed)
    sim = H.Sim(r, nreq=n, seed=seed)
    sim._digest = lambda out, nn: {}
    orig = r._collect_mamba_state_seed_copies
    rows = []

    def timed(*a, **k):
        on = l6_ab.on("seedvec")
        SREC.excl = 0
        t0 = PERF()
        out = orig(*a, **k)
        rows.append((PERF() - t0 - SREC.excl, on, len(r._pending_mamba_state_copies)))
        r._pending_mamba_state_copies.clear()
        return out

    r._collect_mamba_state_seed_copies = timed
    gc.collect()
    gc.disable()
    try:
        for s in range(warm + steps):
            sim.step(seed_events(stream, s))
    finally:
        gc.enable()
    counts = dict(l6_ab._state.get("counts", {}))
    return rows[warm:], counts, r, sim


def seed_fragments(r, n):
    """us per call of the stock loop and the candidates on a steady runner, at a
    steady step and at an edge step (one request straddles a column edge)."""
    prev = r._pre_async_results
    ib = r.input_batch
    so = NS(num_scheduled_tokens={rid: 1 + H.K for rid in ib.req_ids[:n]})
    out = {}
    fns = (("stock", stock_loop), ("V1 numpy", cand_vec), ("V2 lists", cand_lean))
    # Steady: every request's column fixed; all three agree on the result.
    ref = None
    for name, fn in fns:
        pos = dict(r._mamba_state_pos)
        res = fn(r, so, 0, n, COL, prev, pos)
        got = (res, sorted(pos.items()))
        assert ref is None or got == ref, (name, "steady result differs")
        ref = got
    for name, fn in fns:
        pos = dict(r._mamba_state_pos)
        out[("steady", name)] = tmin(lambda: fn(r, so, 0, n, COL, prev, pos))
    # Edge: request 0 at num_computed = k*COL - 1, so its last 3 positions straddle the edge.
    saved = int(ib.num_computed_tokens_cpu[0])
    ib.num_computed_tokens_cpu[0] = (saved // COL + 1) * COL - 1
    try:
        ref = None
        for name, fn in fns:
            pos = dict(r._mamba_state_pos)
            res = fn(r, so, 0, n, COL, prev, pos)
            got = (res, sorted(pos.items()))
            assert ref is None or got == ref, (name, "edge result differs")
            ref = got
        for name, fn in fns:
            pos = dict(r._mamba_state_pos)
            out[("edge", name)] = tmin(lambda: fn(r, so, 0, n, COL, prev, pos))
    finally:
        ib.num_computed_tokens_cpu[0] = saved
    return out


def part_seed(a, ns):
    arms = ["sc"] + (["sv", "ab"] if "seedvec" in l6_ab.LEVERS else [])
    if len(arms) == 1:
        print("BENCH seed: this tree registers no seedvec lever (arms sc only)")
    streams = ["steady", "churn"]
    res: dict = {}
    counts_seen: dict = {}
    last = {}
    for _rep in range(a.reps):
        for stream in streams:
            for n in ns:
                for arm in arms:
                    rows, counts, r, sim = run_seed(n, arm, stream, a.steps, a.warm, seed=200 + n)
                    if arm == "ab":
                        for flag, key in ((True, "ab.on"), (False, "ab.off")):
                            sel = [x[0] for x in rows if x[1] == flag]
                            res.setdefault((stream, n, key), []).append(sel)
                    else:
                        res.setdefault((stream, n, arm), []).append([x[0] for x in rows])
                        counts_seen[(stream, n, arm)] = (counts, sum(1 for x in rows if x[2]))
                    if stream == "steady" and arm == "sc":
                        last[n] = r
    tarms = [x for x in arms if x != "ab"] + (["ab.off", "ab.on"] if "ab" in arms else [])
    for stream in streams:
        print(f"BENCH seed table stream={stream}: host us per _collect_mamba_state_seed_copies call "
              f"(median over {a.reps} reps of the per-run mean / p50; upload stand-ins excluded)")
        for n in ns:
            for arm in tarms:
                runs = res[(stream, n, arm)]
                mean = statistics.median(float(np.mean(x)) / 1e3 for x in runs)
                p50 = statistics.median(float(np.median(x)) / 1e3 for x in runs)
                print(f"BENCH seed {stream:6s} n={n:3d} {arm:6s} mean {mean:7.2f} p50 {p50:7.2f}")
        for n in ns:
            for arm in arms:
                if arm == "ab":
                    continue
                c, ncross = counts_seen[(stream, n, arm)]
                keys = ",".join(f"{k}={v}" for k, v in sorted(c.items()) if k.startswith(("seedvec", "carry.carried")))
                print(f"BENCH seed counts stream={stream} n={n} arm={arm} crossing_calls={ncross} {keys or '-'}")
        if "sv" in arms:
            for n in ns:
                d = (statistics.median(float(np.mean(x)) for x in res[(stream, n, "sv")])
                     - statistics.median(float(np.mean(x)) for x in res[(stream, n, "sc")])) / 1e3
                on_, off_ = res[(stream, n, "ab.on")], res[(stream, n, "ab.off")]
                pd = (statistics.median(float(np.median(x)) for x in on_)
                      - statistics.median(float(np.median(x)) for x in off_)) / 1e3
                print(f"BENCH seed delta stream={stream} n={n} sv-sc mean {d:+.2f} us; paired ab on-off p50 {pd:+.2f} us")
    print("BENCH seed fragments: us per call (best of 7 x 300, isolated, cache-warm)")
    for n in ns:
        if n not in last:
            continue
        set_env(SEED_ARMS["sc"])
        frag = seed_fragments(last[n], n)
        for kind in ("steady", "edge"):
            s = frag[(kind, "stock")]
            line = " ".join(f"{name} {frag[(kind, name)]:6.2f} ({frag[(kind, name)] - s:+6.2f})"
                            for name in ("stock", "V1 numpy", "V2 lists"))
            print(f"BENCH seed frag n={n:3d} {kind:6s} {line}")
    return True


# ============================================================ part setup


def setup_world(n_attn=22, n_mamba=37, rows=64, nreq=25, bs=512, k=3, width=16):
    """A served-shaped STEADY world (cf. tests/q38_l6/test_l6_steady.py): real
    BlockTables and AttentionMetadataBuilders, one layer name per group (the
    first two attention groups carry two, as the draft layer's group does)."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
    from vllm.v1.worker.block_table import BlockTable

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilder
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

    cpu = torch.device("cpu")
    g = n_attn + n_mamba
    tables = [BlockTable(bs, rows, width, 64 * rows, False, cpu, bs, 1) for _ in range(g)]
    nxt = [1]
    lens = []
    for row in range(nreq):
        length = 100 + 37 * row
        nb = -(-(length + 2 * k) // bs)
        for bt in tables:
            bt.add_row(list(range(nxt[0], nxt[0] + nb)), row)
            nxt[0] += nb
        lens.append(length)
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.device = cpu
    r.max_num_reqs = rows
    r.most_model_len = None
    r._block_table_stage_cpu = None
    r._unified_kv_layout = True
    r.speculative_config = NS(num_speculative_tokens=k)
    r.input_batch = NS(block_table=tables, req_ids=[f"r{i}" for i in range(nreq)], num_reqs=nreq)
    attn = FullAttentionSpec(block_size=bs, num_kv_heads=2, head_size=128, dtype=torch.bfloat16,
                             page_size_padded=16384)
    mamba = MambaSpec(block_size=bs, shapes=[(4, 128), (8, 64, 32)], dtypes=[torch.bfloat16, torch.float32],
                      page_size_padded=16384)
    groups = []
    for gid in range(g):
        spec = attn if gid < n_attn else mamba
        b = AttentionMetadataBuilder(spec, [f"g{gid}"], MagicMock(), cpu, r, gid)
        names = [f"l{gid}.a"] + ([f"l{gid}.b"] if gid < 2 else [])
        groups.append([NS(metadata_builders=[b], layer_names=names)])
    r.attn_groups = groups
    return r, np.array(lens, dtype=np.int64), rows, k


def setup_call(r, lens, rows, k, t):
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilderContext

    n = len(lens)
    hi = lens + 1 + k
    r.position_ids = torch.arange(4 * rows, dtype=torch.int32) + t
    ctx = AttentionMetadataBuilderContext(
        num_reqs=n,
        start_index=0,
        use_max_model_len=True,
        seq_lens=torch.tensor(list(lens + 1) + [0] * (rows - n), dtype=torch.int32),
        query_start_loc=torch.arange(rows + 1, dtype=torch.int32) * 4,
        request_distribution=torch.tensor([n, n, n], dtype=torch.int32),
        mamba_slot_read_offsets=r.__dict__.setdefault("_bench_offsets", torch.zeros(4096, dtype=torch.int32)),
        mamba_ckpt_window=k + 1,
        mamba_request_distribution=torch.tensor([n, n, n], dtype=torch.int32),
        unified_mamba_state_indices=[],
        steady_seq_lens_hi=hi,
        steady_max_rejected=k,
    )
    r._attn_metadata_builder_ctx = ctx
    t0 = PERF()
    out, _ = r._build_attention_metadata(num_tokens=4 * n, num_reqs=rows, num_reqs_padded=rows)
    return out, ctx, PERF() - t0


def run_setup(arm, steps, warm):
    env = {"VLLM_TPU_L6_STEADY_STEP": "1"}
    if arm == "pp":
        env["VLLM_TPU_L6_SETUP_FREE"] = "1"
    set_env(env)
    os.environ["VLLM_TPU_FAST_METADATA_BUILD"] = "1"
    r, lens, rows, k = setup_world()
    held = None
    call_us, rel_us, fast = [], [], 0
    gc.collect()
    gc.disable()
    try:
        for t in range(warm + steps):
            l6_ab.begin_step()
            before = l6_ab._state["counts"].get("steady.fast", 0)
            out, ctx, dt = setup_call(r, lens, rows, k, t)
            # sample_setup: the previous step's DraftChunkInputs (and with it its metadata) is released.
            prev, held = held, None
            t2 = PERF()
            del prev
            t3 = PERF()
            held = (out, ctx)
            is_fast = l6_ab._state["counts"].get("steady.fast", 0) > before
            if t >= warm and is_fast:
                fast += 1
                call_us.append(dt / 1e3)
                rel_us.append((t3 - t2) / 1e3)
    finally:
        gc.enable()
    counts = dict(l6_ab._state.get("counts", {}))
    return call_us, rel_us, fast, counts, r


def setup_fragments():
    """Object-level costs on a recorded 59-group STEADY cache."""
    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadataBuilderContext
    from vllm_torchtpu.runner import l6_steady
    from vllm_torchtpu.spec_decode.utils import DraftChunkInputs

    set_env({"VLLM_TPU_L6_STEADY_STEP": "1"})
    os.environ["VLLM_TPU_FAST_METADATA_BUILD"] = "1"
    r, lens, rows, k = setup_world()
    l6_ab.begin_step()
    setup_call(r, lens, rows, k, 0)
    l6_ab.begin_step()
    _, ctx, _dt = setup_call(r, lens, rows, k, 1)
    c = r._l6_steady_am
    out = {}
    out["clone (l6_steady._clone, 59 objects + per-layer dict)"] = tmin(lambda: l6_steady._clone(c, r, ctx))

    def rel():
        x = l6_steady._clone(c, r, ctx)
        t0 = PERF()
        del x
        return PERF() - t0

    samples = []
    for _ in range(2000):
        samples.append(rel())
    out["release of one cloned set (del)"] = float(np.median(samples)) / 1e3
    per_layer = l6_steady._clone(c, r, ctx)
    objs = []
    seen = set()
    for name, _i in c.name_idx:
        o = per_layer[name]
        if id(o) not in seen:
            seen.add(id(o))
            objs.append(o)
    step_all = {"input_positions": r.position_ids, "seq_lens": ctx.seq_lens,
                "query_start_loc": ctx.query_start_loc, "request_distribution": ctx.request_distribution}
    step_m = dict(step_all, mamba_slot_read_offsets=ctx.mamba_slot_read_offsets,
                  mamba_request_distribution=ctx.mamba_request_distribution)
    md = [o.__dict__ for o, (_t, m) in zip(objs, c.groups) if m]
    ad = [o.__dict__ for o, (_t, m) in zip(objs, c.groups) if not m]
    dicts = [o.__dict__ for o in objs]
    tm = [t for t, _m in c.groups]

    def pp_step():
        list(map(dict.update, md, itertools.repeat(step_m)))
        list(map(dict.update, ad, itertools.repeat(step_all)))

    def pp_full():
        list(map(dict.update, dicts, tm))
        pp_step()

    getrc = sys.getrefcount

    def guard():
        return getrc(per_layer), list(map(getrc, objs))

    out["pp step-only rewrite (59 dict.update)"] = tmin(pp_step)
    out["pp full rewrite (template + step)"] = tmin(pp_full)
    out["pp refcount guard (59 objects + dict)"] = tmin(guard)
    # A DraftChunkInputs as the target forward leaves it (STEADY fast step).
    T = 512

    def chunk(hidden):
        cctx = AttentionMetadataBuilderContext(
            num_reqs=25, start_index=0, use_max_model_len=True, seq_lens=ctx.seq_lens,
            query_start_loc=ctx.query_start_loc, request_distribution=ctx.request_distribution,
            mamba_slot_read_offsets=ctx.mamba_slot_read_offsets, mamba_ckpt_window=k + 1,
            mamba_request_distribution=ctx.mamba_request_distribution, unified_mamba_state_indices=[],
            steady_seq_lens_hi=lens + 1 + k, steady_max_rejected=k)
        cctx.staged_block_tables = dict(c.tables)
        cctx.staged_mamba_products = None if c.products is None else dict(c.products)
        return [DraftChunkInputs(
            input_ids=torch.zeros(T, dtype=torch.int32), position_ids=r.position_ids,
            query_start_loc_np=np.zeros(26, dtype=np.int32), attn_ctx=cctx, start_index=0, num_reqs=25,
            aux_hidden_states=[], hidden_states=torch.zeros(T, 2560, dtype=torch.bfloat16) if hidden else None,
            draft_lengths=torch.zeros(64, dtype=torch.int32),
            attn_metadata=l6_steady._clone(c, r, ctx))]

    for hidden in (False, True):
        samples = []
        for _ in range(2000):
            x = chunk(hidden)
            t0 = PERF()
            del x
            samples.append(PERF() - t0)
        out[f"DraftChunkInputs release ({'with' if hidden else 'no'} CPU hidden tensor [512, 2560] bf16)"] = (
            float(np.median(samples)) / 1e3)
    out["(shape) groups / mamba groups / layer names"] = (len(c.groups), sum(1 for _t, m in c.groups if m),
                                                         len(c.name_idx))
    return out


def part_setup(a):
    arms = ["clone"] + (["pp"] if "setupfree" in l6_ab.LEVERS else [])
    if len(arms) == 1:
        print("BENCH setup: this tree registers no setupfree lever (arm clone only)")
    res: dict = {}
    counts_seen = {}
    for _rep in range(a.reps):
        for arm in arms:
            call_us, rel_us, fast, counts, r = run_setup(arm, a.steps, a.warm)
            res.setdefault(arm, []).append((call_us, rel_us, fast))
            counts_seen[arm] = counts
    print(f"BENCH setup table: STEADY fast steps only, 59 groups (37 mamba), 25 requests, 64 rows; "
          f"median over {a.reps} reps of per-run mean / p50 (us)")
    for arm in arms:
        runs = res[arm]
        cm = statistics.median(float(np.mean(x[0])) for x in runs)
        cp = statistics.median(float(np.median(x[0])) for x in runs)
        rm = statistics.median(float(np.mean(x[1])) for x in runs)
        rp = statistics.median(float(np.median(x[1])) for x in runs)
        fast = statistics.median(x[2] for x in runs)
        print(f"BENCH setup arm={arm:5s} fast_steps={fast} am_call mean {cm:7.2f} p50 {cp:7.2f} | "
              f"sample_setup release mean {rm:6.2f} p50 {rp:6.2f} | sum p50 {cp + rp:7.2f}")
        keys = ",".join(f"{k}={v}" for k, v in sorted(counts_seen[arm].items())
                        if k.startswith(("steady.fast", "steady.full", "setupfree")))
        print(f"BENCH setup counts arm={arm} {keys}")
    print("BENCH setup fragments: us (best of 7 x 300; releases: median of 2000)")
    for name, v in setup_fragments().items():
        print(f"BENCH setup frag {name:70s} {v if isinstance(v, tuple) else f'{v:8.2f}'}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", default="4,9,11,25,52")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warm", type=int, default=40)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--parts", default="seed,setup")
    a = ap.parse_args()
    ns = [int(x) for x in a.n.split(",")]
    parts = [x for x in a.parts.split(",") if x]
    tpu_runner_upl = H.Uploads
    H.Uploads = TimedUploads
    os.environ["VLLM_TPU_POSTFWD_FUSED"] = "1"
    undo = H.install_eager_programs()
    l6_ab.logger.info = lambda *x, **k: None
    print(f"BENCH start {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} nproc {os.cpu_count()} "
          f"python {sys.version.split()[0]} torch {torch.__version__} numpy {np.__version__} "
          f"parts={parts} n={ns} steps={a.steps} warm={a.warm} reps={a.reps} levers={sorted(l6_ab.LEVERS)}")
    ok = True
    try:
        if "seed" in parts:
            ok &= part_seed(a, ns)
        if "setup" in parts:
            ok &= part_setup(a)
    finally:
        undo()
        H.Uploads = tpu_runner_upl
        set_env({})
    print(f"BENCH_DONE ok={ok}")


if __name__ == "__main__":
    main()
