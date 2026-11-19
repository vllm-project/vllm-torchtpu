# SPDX-License-Identifier: Apache-2.0
"""q38 L6-STEADY: the steady-step fast path (runner/l6_steady.py, lever ``steady``).

Host-only; runs in a CDK CPU pod. Through the real runner methods on real vLLM
BlockTables and real AttentionMetadataBuilders:

1. attention metadata: over random decode sequences with admissions, finishes,
   reorders and block growth mixed in, the lever-on runner's metadata, staged
   tables, mamba products and unified state list equal a lever-off replica's
   byte for byte at every step; the fast path carries most steps and never a
   step that mutated a table or moved a state column; per-step tensors are this
   step's objects and no metadata object is reused across steps;
2. each change kind falls back to the full path (and the next steady step is
   fast again where it can be);
3. CHECK mode compares and turns the lever off on a mismatch;
4. reorder + chunk count: cached results equal the real walks over random
   schedules;
5. upstream reorder: the no-op predicate never skips a reorder vLLM's real
   function would perform;
6. adapter block-table views: reused only for the same source tensors;
7. lever off: identical behaviour (no l6 call, nothing cached).
"""

import ast
import os
import pathlib
import random
from types import SimpleNamespace
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
from vllm_torchtpu.layers.core.sequence_layout import (
    PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL,
    SequenceLayoutDescriptor,
    SequenceLayoutKind,
)
from vllm_torchtpu.runner import l6_steady, tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

BS = 128  # tokens per block, every group
K = 2  # speculative tokens: a request's device length lies in [hi - K, hi]
WIDTH = 16  # blocks per row
MAXR = 8  # rows (= the padded request count)
N_ATTN, N_MAMBA = 2, 3
CPU = torch.device("cpu")


# ---------------------------------------------------------------- fixtures


def _counts():
    return dict(l6_ab._state["counts"])


@pytest.fixture
def lever(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in (
            "VLLM_TPU_HOST_AB_PERIOD_MS",
            "VLLM_TPU_D1N_AB_PERIOD_MS",
        ):
            monkeypatch.delenv(k, raising=False)
    # One fixed window: the counts accumulate and no window line is logged.
    monkeypatch.setattr(l6_ab, "_now_ns", lambda: 10**18)
    monkeypatch.setattr(l6_ab.logger, "info", lambda *a, **k: None)
    errors = []
    monkeypatch.setattr(l6_steady.logger, "error", lambda fmt, *a: errors.append(fmt % a))

    def setup(value="1", check=0):
        if value:
            monkeypatch.setenv("VLLM_TPU_L6_STEADY_STEP", value)
        else:
            monkeypatch.delenv("VLLM_TPU_L6_STEADY_STEP", raising=False)
        l6_ab._resolve()
        l6_steady._CHECK[0] = check
        l6_ab.begin_step()
        return errors

    yield setup
    monkeypatch.delenv("VLLM_TPU_L6_STEADY_STEP", raising=False)
    l6_ab._resolve()
    l6_steady._CHECK[0] = None


def _flags(monkeypatch, served):
    """FMB on (the fast path caches FMB-built metadata); `served` adds the
    served decode env's staging levers (upload reuse, entries cache, class
    gather, BT-INC)."""
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


# ---------------------------------------------------------------- the decode world


class World:
    """Requests (device length s, host length h) and one real BlockTable per KV
    group; mutations go through BlockTable's methods, as vLLM's InputBatch does."""

    def __init__(self):
        self.tables = [
            BlockTable(BS, MAXR, WIDTH, 64, False, CPU, BS, 1)
            for _ in range(N_ATTN + N_MAMBA)
        ]
        self.reqs = []
        self.next_id = 0
        self.next_block = 1

    def _blocks(self, n):
        out = list(range(self.next_block, self.next_block + n))
        self.next_block += n
        return out

    @staticmethod
    def _need(h):
        return -(-(h + K) // BS)  # blocks for h plus the K lookahead tokens

    def admit(self, length):
        row = len(self.reqs)
        nb = self._need(length)
        for bt in self.tables:
            bt.add_row(self._blocks(nb), row)
        self.reqs.append(dict(id=f"r{self.next_id}", s=length, h=length, nb=nb))
        self.next_id += 1

    def finish(self, row):
        last = len(self.reqs) - 1
        for bt in self.tables:
            if row == last:
                bt.clear_row(row)
            else:
                bt.move_row(last, row)
        self.reqs[row] = self.reqs[last]
        self.reqs.pop()

    def swap(self, i, j):
        for bt in self.tables:
            bt.swap_row(i, j)
        self.reqs[i], self.reqs[j] = self.reqs[j], self.reqs[i]

    def advance(self, accepted):
        """One verify step: a request that accepts a of K drafts moves s by
        1 + a; the host assumed all K (h = s_prev + 1 + K). Blocks follow h + K."""
        for row, (req, a) in enumerate(zip(self.reqs, accepted)):
            s0 = req["s"]
            req["s"], req["h"] = s0 + 1 + a, s0 + 1 + K
            need = self._need(req["h"])
            if need > req["nb"]:
                for bt in self.tables:
                    bt.append_row(self._blocks(need - req["nb"]), row)
                req["nb"] = need


def _runner(world):
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.device = CPU
    r.max_num_reqs = MAXR
    r.most_model_len = None
    r._block_table_stage_cpu = None
    r._unified_kv_layout = True
    r.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    r.input_batch = SimpleNamespace(block_table=world.tables, req_ids=[], num_reqs=0)
    attn = FullAttentionSpec(
        block_size=BS, num_kv_heads=2, head_size=128, dtype=torch.bfloat16, page_size_padded=16384
    )
    mamba = MambaSpec(
        block_size=BS,
        shapes=[(4, 128), (8, 64, 32)],
        dtypes=[torch.bfloat16, torch.float32],
        page_size_padded=16384,
    )
    groups = []
    for gid in range(N_ATTN + N_MAMBA):
        spec = attn if gid < N_ATTN else mamba
        b = AttentionMetadataBuilder(spec, [f"g{gid}"], MagicMock(), CPU, r, gid)
        groups.append([SimpleNamespace(metadata_builders=[b], layer_names=[f"l{gid}.a", f"l{gid}.b"])])
    r.attn_groups = groups
    return r


class Step:
    """One step's per-step tensors, shared by the lever-on runner and its replica."""

    def __init__(self, world, t, nrp=MAXR, descriptor=None):
        n = len(world.reqs)
        self.n, self.nrp = n, nrp
        self.seq_lens = torch.tensor([q["s"] for q in world.reqs] + [0] * (nrp - n), dtype=torch.int32)
        self.hi = np.array([q["h"] for q in world.reqs], dtype=np.int64)
        self.positions = torch.arange(3 * nrp, dtype=torch.int32) + t
        self.qsl = torch.arange(nrp + 1, dtype=torch.int32) * 3
        self.rd = torch.tensor([n, n, n], dtype=torch.int32) + t % 2
        self.mrd = torch.tensor([n, n, n], dtype=torch.int32)
        self.offsets = torch.zeros(4 * MAXR, dtype=torch.int32)
        self.descriptor = descriptor
        self.req_ids = [q["id"] for q in world.reqs]


def _call(runner, step, steady, override=None, compact=None):
    runner.input_batch.req_ids = list(step.req_ids)
    runner.input_batch.num_reqs = step.n
    runner.position_ids = step.positions
    extra = {}
    if step.descriptor is not None:
        extra["sequence_layout_descriptor"] = step.descriptor
    ctx = AttentionMetadataBuilderContext(
        num_reqs=step.n,
        start_index=0,
        use_max_model_len=True,
        seq_lens=step.seq_lens,
        query_start_loc=step.qsl,
        request_distribution=step.rd,
        position_ids_override=override,
        mamba_state_indices=compact,
        mamba_slot_read_offsets=step.offsets,
        mamba_ckpt_window=K + 1,
        mamba_request_distribution=step.mrd,
        unified_mamba_state_indices=[],
        steady_seq_lens_hi=step.hi if steady else None,
        steady_max_rejected=K,
        **extra,
    )
    runner._attn_metadata_builder_ctx = ctx
    out, unused = runner._build_attention_metadata(
        num_tokens=3 * step.n, num_reqs=step.nrp, num_reqs_padded=step.nrp
    )
    assert unused is None
    return out, ctx


def _same(a, ctx_a, b, ctx_b):
    assert a.keys() == b.keys()
    for name in b:
        for f in l6_steady._FIELDS:
            x, y = getattr(a[name], f, None), getattr(b[name], f, None)
            assert (x is None) == (y is None), (name, f)
            if isinstance(x, torch.Tensor):
                assert x.dtype == y.dtype and torch.equal(x, y), (name, f)
            else:
                assert x == y, (name, f)
    assert ctx_a.staged_block_tables.keys() == ctx_b.staged_block_tables.keys()
    for g, t in ctx_b.staged_block_tables.items():
        assert torch.equal(ctx_a.staged_block_tables[g], t), ("table", g)
    pa, pb = ctx_a.staged_mamba_products, ctx_b.staged_mamba_products
    assert (pa is None) == (pb is None)
    if pb is not None:
        assert pa.keys() == pb.keys()
        for g, (st, ck) in pb.items():
            assert torch.equal(pa[g][0], st), ("state", g)
            assert (ck is None) == (pa[g][1] is None)
            if ck is not None:
                assert torch.equal(pa[g][1], ck), ("ckpt", g)
    sa, sb = ctx_a.unified_mamba_state_indices, ctx_b.unified_mamba_state_indices
    assert len(sa) == len(sb) and all(torch.equal(x, y) for x, y in zip(sa, sb))


def _step_fields(layers, runner, ctx):
    """Per-step fields are this step's objects (never a cached step's)."""
    for name, md in layers.items():
        assert md.input_positions is runner.position_ids, name
        assert md.seq_lens is ctx.seq_lens, name
        assert md.query_start_loc is ctx.query_start_loc, name
        assert md.request_distribution is ctx.request_distribution, name
        if md.mamba_state_indices is not None:
            assert md.mamba_slot_read_offsets is ctx.mamba_slot_read_offsets, name
            assert md.mamba_request_distribution is ctx.mamba_request_distribution, name


class Pair:
    """The lever-on runner and a lever-off replica on identical worlds."""

    def __init__(self):
        self.wa, self.wb = World(), World()
        self.a, self.b = _runner(self.wa), _runner(self.wb)
        self.t = 0
        self.prev_a = None

    def both(self, fn, *args):
        getattr(self.wa, fn)(*args)
        getattr(self.wb, fn)(*args)

    def step(self, nrp=MAXR, descriptor=None, steady=True, compare=True):
        """Returns the attention-metadata outcome key counted this step."""
        self.t += 1
        l6_ab.begin_step()
        before = _counts()
        sa = Step(self.wa, self.t, nrp, descriptor)
        sb = Step(self.wb, self.t, nrp, descriptor)
        sb.positions, sb.seq_lens, sb.qsl, sb.rd, sb.mrd, sb.offsets = (
            sa.positions, sa.seq_lens, sa.qsl, sa.rd, sa.mrd, sa.offsets
        )
        out_a, ctx_a = _call(self.a, sa, steady)
        out_b, ctx_b = _call(self.b, sb, False)
        if compare:
            _same(out_a, ctx_a, out_b, ctx_b)
            _step_fields(out_a, self.a, ctx_a)
            if self.prev_a is not None:
                for name in out_a:
                    assert out_a[name] is not self.prev_a[name], name
        self.prev_a = out_a
        after = _counts()
        got = [k for k in ("fast", "full", "check") if after.get(f"steady.{k}", 0) > before.get(f"steady.{k}", 0)]
        return got[0] if got else None


# ---------------------------------------------------------------- 1. fast == full over random sequences


@pytest.mark.parametrize("served", [False, True])
def test_fast_equals_full_over_random_steady_sequences(lever, monkeypatch, served):
    lever("1")
    _flags(monkeypatch, served)
    rng = random.Random(1234 + served)
    p = Pair()
    for _ in range(4):
        p.both("admit", rng.randint(20, 300))
    stage_calls = []
    real_stage = tpu_runner.stage_block_table_uploads
    monkeypatch.setattr(
        tpu_runner,
        "stage_block_table_uploads",
        lambda runner, *a, **k: (stage_calls.append(runner), real_stage(runner, *a, **k))[1],
    )
    outcome = {"fast": 0, "full": 0}
    mutated_fast = qmove_fast = 0
    prev_q = None
    for _ in range(500):
        v0 = am._BLOCK_TABLE_VERSION[0]
        r = rng.random()
        n = len(p.wa.reqs)
        if r < 0.04 and n < MAXR:
            p.both("admit", rng.randint(20, 300))
        elif r < 0.08 and n > 1:
            p.both("finish", rng.randrange(n))
        elif r < 0.10 and n > 1:
            i, j = rng.sample(range(n), 2)
            p.both("swap", i, j)
        accepted = [rng.randint(0, K) for _ in p.wa.reqs]
        # Requests about to outgrow a row finish instead.
        for row in reversed(range(len(p.wa.reqs))):
            if p.wa.reqs[row]["s"] + 2 * (1 + K) + K > WIDTH * BS - BS:
                p.both("finish", row)
                accepted.pop(row)
        p.both("advance", accepted)
        mutated = am._BLOCK_TABLE_VERSION[0] != v0
        q = [(req["s"] - 1) // BS for req in p.wa.reqs]
        qmoved = prev_q is not None and q != prev_q[: len(q)]
        prev_q = q
        stage_calls.clear()
        got = p.step()
        assert got in ("fast", "full")
        outcome[got] += 1
        if got == "fast":
            assert p.a not in stage_calls, "a fast step staged block tables"
            mutated_fast += mutated
            qmove_fast += qmoved
    assert mutated_fast == 0, "a step that mutated a block table took the fast path"
    assert qmove_fast == 0, "a step that moved a state column took the fast path"
    assert outcome["fast"] > 250 and outcome["full"] > 40, outcome


def test_seq_lens_bound_reads_num_computed_plus_scheduled():
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.speculative_config = SimpleNamespace(num_speculative_tokens=3)
    r.input_batch = SimpleNamespace(num_computed_tokens_cpu=np.array([5, 100, 200, 300, 7], dtype=np.int32))
    hi, k = l6_steady.seq_lens_bound(r, 1, 3, np.array([4, 1, 4], dtype=np.int32))
    assert hi.tolist() == [104, 201, 304] and hi.dtype == np.int64 and k == 3
    r2 = TPUModelRunner.__new__(TPUModelRunner)
    r2.speculative_config = None
    r2.input_batch = r.input_batch
    assert l6_steady.seq_lens_bound(r2, 0, 1, np.array([1]))[1] == 0


# ---------------------------------------------------------------- 2. each change kind falls back


def _steady_pair(monkeypatch, lever, lengths=(10, 150, 400)):
    lever("1")
    _flags(monkeypatch, False)
    p = Pair()
    for length in lengths:
        p.both("admit", length)
    assert p.step() == "full"
    p.both("advance", [0] * len(lengths))
    assert p.step() == "fast"
    return p


@pytest.mark.parametrize("change", ["append", "admit", "finish", "swap", "clear"])
def test_each_table_change_falls_back_then_resumes(lever, monkeypatch, change):
    p = _steady_pair(monkeypatch, lever)
    if change == "append":
        p.both("advance", [0, 0, 0])
        for w in (p.wa, p.wb):
            w.tables[N_ATTN].append_row([w._blocks(1)[0]], 1)
    elif change == "admit":
        p.both("admit", 33)
    elif change == "finish":
        p.both("finish", 0)
    elif change == "swap":
        p.both("swap", 0, 2)
    elif change == "clear":
        for w in (p.wa, p.wb):
            w.tables[0].clear_row(7)  # an unused row: still a table write
    assert p.step() == "full"
    p.both("advance", [0] * len(p.wa.reqs))
    assert p.step() == "fast"


def test_a_state_column_move_without_a_table_change_falls_back(lever, monkeypatch):
    """Lookahead allocates the next block K tokens early, so the state column
    moves on a later step with no table write: that step must not reuse the
    cached mamba products."""
    p = _steady_pair(monkeypatch, lever, lengths=(94,))  # then s 95, h 97
    while p.wa.reqs[0]["s"] + 1 + K < BS:
        p.both("advance", [K])  # every draft accepted: s = h, +3 a step, column 0
        assert p.step() == "fast"
    assert p.wa.reqs[0]["s"] + 1 + K == BS
    v0 = am._BLOCK_TABLE_VERSION[0]
    p.both("advance", [K])  # s = h = BS: the second block is appended (h + K > BS)
    assert am._BLOCK_TABLE_VERSION[0] != v0
    assert p.step() == "full"  # column 0, determined over [BS - K, BS]: cached
    v1 = am._BLOCK_TABLE_VERSION[0]
    p.both("advance", [K])  # s = h = BS + 3: column 1, no table write
    assert am._BLOCK_TABLE_VERSION[0] == v1
    assert p.step() == "full"
    p.both("advance", [K])  # s = h = BS + 6: column 1 again
    assert p.step() == "fast"


def test_the_column_bound_counts_pending_rejections(lever, monkeypatch):
    """Host BS + 1, device BS - 1: column 0 on device, column 1 by the host
    length. The bound [hi - K, hi] is undetermined, so no products are cached;
    the next step (device = host = BS + 2) is again undetermined; nothing stale
    is reused."""
    p = _steady_pair(monkeypatch, lever, lengths=(50,))
    for w in (p.wa, p.wb):
        req = w.reqs[0]
        req["s"], req["h"] = BS - 1, BS + 1
        need = w._need(BS + 1)
        for bt in w.tables:
            bt.append_row(w._blocks(need - req["nb"]), 0)
        req["nb"] = need
    assert p.step() == "full"
    p.both("advance", [K])  # s = h = BS + 2: lo = BS - 1, still undetermined
    assert p.step() == "full"
    p.both("advance", [K])  # s = h = BS + 5: column 1, determined (the cache had none)
    assert p.step() == "full"
    p.both("advance", [K])
    assert p.step() == "fast"


def test_geometry_and_layout_changes_fall_back(lever, monkeypatch):
    p = _steady_pair(monkeypatch, lever)
    p.both("advance", [0, 0, 0])
    assert p.step(nrp=4) == "full"  # padded rows changed
    p.both("advance", [0, 0, 0])
    assert p.step(nrp=4) == "fast"
    d = SequenceLayoutDescriptor(
        kind=SequenceLayoutKind.PARTIAL, protocol=PCP_STREAMING_SEQUENCE_LAYOUT_PROTOCOL, version=7
    )
    p.both("advance", [0, 0, 0])
    assert p.step(nrp=4, descriptor=d) == "full"
    p.both("advance", [0, 0, 0])
    assert p.step(nrp=4, descriptor=d) == "fast"


def test_non_target_walks_never_use_or_record(lever, monkeypatch):
    p = _steady_pair(monkeypatch, lever)
    cache = p.a._l6_steady_am
    calls = []
    for fn in ("am_try", "am_record"):
        real = getattr(l6_steady, fn)
        monkeypatch.setattr(l6_steady, fn, lambda *a, real=real, fn=fn: (calls.append(fn), real(*a))[1])
    p.both("advance", [0, 0, 0])
    step = Step(p.wa, 99)
    # A draft / dummy walk: no steady bound on its context.
    _call(p.a, step, steady=False)
    _call(p.a, step, steady=False, override=torch.zeros(3 * MAXR, dtype=torch.int32))
    assert calls == [] and p.a._l6_steady_am is cache
    # A compact-pool context (per-step slot ids) is never fast and drops the cache.
    _call(p.a, step, steady=True, compact=torch.arange(MAXR, dtype=torch.int32))
    assert calls == ["am_try", "am_record"] and p.a.__dict__.get("_l6_steady_am") is None


def test_fmb_off_never_caches(lever, monkeypatch):
    """Only FMB-built metadata is cached: with FMB off every step runs the
    builder walk and nothing is cached. A cache recorded under FMB stays exact
    if FMB is switched off later (smoke j-a408919f found that step fast): its
    clone equals the builder walk, which every step() compares."""
    lever("1")
    _flags(monkeypatch, False)
    monkeypatch.delenv("VLLM_TPU_FAST_METADATA_BUILD")
    p = Pair()
    for length in (10, 150, 400):
        p.both("admit", length)
    for _ in range(4):
        assert p.step() == "full"
        assert p.a.__dict__.get("_l6_steady_am") is None
        p.both("advance", [0, 0, 0])
    monkeypatch.setenv("VLLM_TPU_FAST_METADATA_BUILD", "1")
    assert p.step() == "full"
    assert p.a.__dict__.get("_l6_steady_am") is not None
    p.both("advance", [0, 0, 0])
    assert p.step() == "fast"
    monkeypatch.delenv("VLLM_TPU_FAST_METADATA_BUILD")
    p.both("advance", [0, 0, 0])
    assert p.step() == "fast"  # equal to the lever-off replica's builder walk


def test_lever_off_step_drops_the_caches(lever, monkeypatch):
    p = _steady_pair(monkeypatch, lever)
    p.a._l6_steady_ro = object()
    lever("0")
    assert not l6_steady.on(p.a)
    l6_steady.drop(p.a)  # what execute_model does on a lever-off step
    assert "_l6_steady_am" not in p.a.__dict__ and "_l6_steady_ro" not in p.a.__dict__
    lever("1")
    p.both("advance", [0, 0, 0])
    assert p.step() == "full"
    p.both("advance", [0, 0, 0])
    assert p.step() == "fast"


def test_execute_model_drops_the_caches_on_a_lever_off_step():
    src = pathlib.Path(tpu_runner.__file__).read_text()
    fn = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "execute_model"
    )
    body = ast.unparse(fn)
    i = body.index("_l6_envcache.on_step()")
    seg = body[i : i + 160]
    assert "if _l6_ab.ACTIVE and" in seg and "_l6_steady.on(self)" in seg, seg
    assert "_l6_steady.drop(self)" in seg, seg
    assert body.index("_l6_steady.drop(self)") < body.index("self._update_states(scheduler_output)")


# ---------------------------------------------------------------- 3. CHECK


def test_check_steps_run_the_full_path_and_agree(lever, monkeypatch):
    errors = lever("1", check=2)
    _flags(monkeypatch, True)
    p = Pair()
    for length in (10, 150, 400):
        p.both("admit", length)
    seen = [p.step()]
    for _ in range(8):
        p.both("advance", [0, 0, 0])
        seen.append(p.step())
    assert seen == ["full"] + ["fast", "check"] * 4, seen
    assert errors == [] and not p.a.__dict__.get("_l6_steady_off")


@pytest.mark.parametrize("corrupt", ["table", "state", "template", "states_list"])
def test_check_catches_a_stale_cache_and_turns_the_lever_off(lever, monkeypatch, corrupt):
    errors = lever("1", check=1)
    _flags(monkeypatch, False)
    p = Pair()
    for length in (10, 150, 400):
        p.both("admit", length)
    assert p.step() == "full"
    c = p.a._l6_steady_am
    if corrupt == "table":
        g = next(iter(c.tables))
        c.tables[g] = c.tables[g] + 1
    elif corrupt == "state":
        g = next(iter(c.products))
        st, ck = c.products[g]
        c.products[g] = (st + 1, ck)
    elif corrupt == "template":
        tmpl = next(t for t, is_mamba in c.groups if not is_mamba)
        tmpl["block_tables"] = tmpl["block_tables"] + 1
    else:
        c.states[0] = c.states[0] + 1
    p.both("advance", [0, 0, 0])
    assert p.step() == "check"
    assert _counts().get("steady.mismatch") == 1 and len(errors) == 1, errors
    assert p.a._l6_steady_off and not l6_steady.on(p.a)
    for _ in range(3):
        p.both("advance", [0, 0, 0])
        assert p.step(steady=l6_steady.on(p.a)) is None  # full path, no steady accounting


# ---------------------------------------------------------------- 4. reorder + chunk count


class FakeBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)

    @property
    def num_reqs(self):
        return len(self.req_ids)

    def swap_states(self, i, j):
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]


def _ro_runner(req_ids):
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.input_batch = FakeBatch(req_ids)
    r.most_model_len = None
    r.num_reqs_max_model_len = 4
    r.num_reqs_most_model_len = None
    return r


def _so(sched, spec, new=(), finished=(), resumed=(), preempted=None):
    return SimpleNamespace(
        num_scheduled_tokens=dict(sched),
        scheduled_spec_decode_tokens={r: [7] * (sched[r] - 1) for r in spec},
        scheduled_new_reqs=list(new),
        finished_req_ids=set(finished),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set(resumed)),
        preempted_req_ids=preempted,
        total_num_scheduled_tokens=sum(sched.values()),
    )


def test_reorder_and_count_equals_the_walks_over_random_schedules(lever):
    lever("1")
    rng = random.Random(99)
    ids = [f"q{i}" for i in range(6)]
    a, b = _ro_runner(ids), _ro_runner(ids)
    kind = {r: "verify" for r in ids}
    fast = 0
    for step in range(400):
        l6_ab.begin_step()
        new, finished = [], []
        x = rng.random()
        live = a.input_batch.req_ids
        for r in [r for r in live if kind[r] == "prefill"]:
            kind[r] = "verify"  # a prefill chunk lasts one step here
        if x < 0.05:
            r = rng.choice(live)
            kind[r] = rng.choice(["verify", "decode", "decode", "prefill"])
        elif x < 0.08 and len(live) < 9:
            r = f"q{len(kind)}"
            kind[r] = "verify"
            new = [SimpleNamespace(req_id=r)]
            for rr in (a, b):
                rr.input_batch.req_ids.append(r)
        elif x < 0.11 and len(live) > 2:
            r = rng.choice(live)
            finished = [r]
            for rr in (a, b):
                rr.input_batch.req_ids.remove(r)
        elif x < 0.13 and len(live) > 2:
            i, j = rng.sample(range(len(live)), 2)
            for rr in (a, b):
                rr.input_batch.swap_states(i, j)
        live = a.input_batch.req_ids
        sched = {r: {"verify": 1 + K, "decode": 1, "prefill": 40}[kind[r]] for r in live}
        spec = [r for r in live if kind[r] == "verify"]
        so = _so(sched, spec, new=new, finished=finished)
        before = _counts().get("steady.ro_fast", 0)
        got = l6_steady.reorder_and_count(a, so)
        want = TPUModelRunner._reorder_batch_for_rpa(b, so) + TPUModelRunner._count_input_chunks(b, so)
        assert got == want, step
        assert a.input_batch.req_ids == b.input_batch.req_ids, step
        fast += _counts().get("steady.ro_fast", 0) > before
    assert fast > 200, fast


@pytest.mark.parametrize("change", ["new", "finished", "resumed", "preempted", "sched", "spec", "order"])
def test_reorder_change_kinds_run_the_walks(lever, change):
    lever("1")
    ids = ["a", "b", "c"]
    r = _ro_runner(ids)
    sched = {"a": 3, "b": 3, "c": 1}
    spec = ["a", "b"]
    l6_steady.reorder_and_count(r, _so(sched, spec))
    l6_ab.begin_step()
    assert l6_steady.reorder_try(r, _so(sched, spec)) is not None
    kw = {}
    if change == "new":
        kw["new"] = [SimpleNamespace(req_id="z")]
    elif change == "finished":
        kw["finished"] = ["zz"]
    elif change == "resumed":
        kw["resumed"] = ["a"]
    elif change == "preempted":
        kw["preempted"] = {"zz"}
    elif change == "sched":
        sched = dict(sched, c=3)
    elif change == "spec":
        spec = ["a"]
    elif change == "order":
        r.input_batch.swap_states(0, 1)
    assert l6_steady.reorder_try(r, _so(sched, spec, **kw)) is None


def test_reorder_with_a_prefill_segment_is_never_cached(lever):
    lever("1")
    r = _ro_runner(["a", "b"])
    so = _so({"a": 1, "b": 40}, [])
    l6_steady.reorder_and_count(r, so)
    assert r.__dict__.get("_l6_steady_ro") is None
    assert l6_steady.reorder_try(r, so) is None


def test_reorder_check_mode_compares_and_disables(lever):
    errors = lever("1", check=1)
    r = _ro_runner(["a", "b", "c"])
    so = _so({"a": 3, "b": 3, "c": 3}, ["a", "b", "c"])
    first = l6_steady.reorder_and_count(r, so)
    assert l6_steady.reorder_and_count(r, so) == first
    assert _counts().get("steady.check") == 1 and errors == []
    r._l6_steady_ro.result = (9, 9, 9, 9)  # a stale cache
    assert l6_steady.reorder_and_count(r, so) == first  # the walks' result is used
    assert _counts().get("steady.mismatch") == 1 and r._l6_steady_off and len(errors) == 1


# ---------------------------------------------------------------- 5. upstream reorder


class UpBatch:
    def __init__(self, req_ids, nc, npt):
        self.req_ids = list(req_ids)
        self.num_computed_tokens_cpu = np.array(nc, dtype=np.int32)
        self.num_prompt_tokens = np.array(npt, dtype=np.int32)
        self.swaps = 0

    def swap_states(self, i, j):
        self.swaps += 1
        self.req_ids[i], self.req_ids[j] = self.req_ids[j], self.req_ids[i]
        for arr in (self.num_computed_tokens_cpu, self.num_prompt_tokens):
            arr[[i, j]] = arr[[j, i]]


def _random_up(rng, all_decode):
    n = rng.randint(1, 8)
    ids, nc, npt, sched = [], [], [], {}
    for i in range(n):
        r = f"u{i}"
        ids.append(r)
        prompt = rng.randint(10, 500)
        if all_decode or rng.random() < 0.7:
            c, t = prompt + rng.randint(0, 50), rng.randint(1, 1 + K)
        else:
            c, t = rng.choice(
                [
                    (0, rng.randint(1, 600)),  # first chunk: prefill
                    (prompt - 5, rng.randint(1, 1 + K)),  # short extend
                    (prompt // 2, 64),  # long extend
                    (prompt + 5, 2 + K + rng.randint(0, 8)),  # done prefilling, above the threshold
                ]
            )
        nc.append(c)
        npt.append(prompt)
        sched[r] = t
    return UpBatch(ids, nc, npt), SimpleNamespace(num_scheduled_tokens=sched)


def test_upstream_noop_predicate_never_skips_a_real_reorder():
    from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills

    rng = random.Random(5)
    skipped = swapped = 0
    for trial in range(2000):
        batch, so = _random_up(rng, all_decode=trial % 3 == 0)
        r = SimpleNamespace(input_batch=batch, reorder_batch_threshold=1 + K)
        noop = l6_steady.upstream_reorder_noop(r, so)
        before = list(batch.req_ids)
        modified = reorder_batch_to_split_decodes_and_prefills(batch, so, decode_threshold=1 + K)
        if noop:
            skipped += 1
            assert not modified and batch.req_ids == before and batch.swaps == 0, trial
        swapped += bool(modified)
    assert skipped > 600 and swapped > 200, (skipped, swapped)


def _up_runner(batch):
    r = TPUModelRunner.__new__(TPUModelRunner)
    r.input_batch = batch
    r.reorder_batch_threshold = 1 + K
    r.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    return r


def _spy_upstream(monkeypatch):
    import vllm.v1.worker.gpu_model_runner as gmr

    calls = []
    real = gmr.reorder_batch_to_split_decodes_and_prefills
    monkeypatch.setattr(
        gmr,
        "reorder_batch_to_split_decodes_and_prefills",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1],
    )
    return calls


def test_may_reorder_batch_skips_only_with_the_lever_on(lever, monkeypatch):
    calls = _spy_upstream(monkeypatch)
    batch = UpBatch(["a", "b"], [100, 200], [50, 60])
    so = SimpleNamespace(num_scheduled_tokens={"a": 3, "b": 1})
    lever("")
    TPUModelRunner._may_reorder_batch(_up_runner(batch), so)
    assert len(calls) == 1
    lever("1")
    TPUModelRunner._may_reorder_batch(_up_runner(batch), so)
    assert len(calls) == 1 and _counts().get("steady.up_skip") == 1
    # A request still prefilling: upstream runs.
    batch2 = UpBatch(["a", "b"], [100, 20], [50, 60])
    TPUModelRunner._may_reorder_batch(_up_runner(batch2), so)
    assert len(calls) == 2 and _counts().get("steady.up_run") == 1


def test_may_reorder_batch_check_runs_upstream_and_verifies(lever, monkeypatch):
    calls = _spy_upstream(monkeypatch)
    errors = lever("1", check=1)
    batch = UpBatch(["a", "b"], [100, 200], [50, 60])
    so = SimpleNamespace(num_scheduled_tokens={"a": 3, "b": 1})
    TPUModelRunner._may_reorder_batch(_up_runner(batch), so)
    assert len(calls) == 1 and _counts().get("steady.check") == 1 and errors == []


# ---------------------------------------------------------------- 6. adapter block-table views


def _adapter():
    from vllm_torchtpu.models.vllm.qwen4_exp.inputs import Qwen4ExpAdapter

    ad = Qwen4ExpAdapter.__new__(Qwen4ExpAdapter)
    ad.attention_prefix = "model.layers.3"
    ad.pooled = False
    ad.cache = None
    return ad


def _snap(main, comp):
    return SimpleNamespace(
        attention_metadata={
            "model.layers.3.self_attn": SimpleNamespace(block_tables=main),
            "model.layers.3.self_attn.compressed_k": SimpleNamespace(block_tables=comp),
        }
    )


def test_adapter_tables_reused_only_for_the_same_sources(lever):
    lever("1")
    ad = _adapter()
    main, comp = torch.arange(12, dtype=torch.int32), torch.arange(12, dtype=torch.int32) + 50
    first = ad._block_tables(_snap(main, comp), 4)
    assert [t.tolist() for t in first] == [main.reshape(4, -1).tolist(), comp.reshape(4, -1).tolist()]
    again = ad._block_tables(_snap(main, comp), 4)
    assert again[0] is first[0] and again[1] is first[1] and _counts().get("steady.tbl_reuse") == 1
    other = ad._block_tables(_snap(main.clone(), comp), 4)
    assert other[0] is not first[0] and torch.equal(other[0], first[0])
    rows = ad._block_tables(_snap(main.clone(), comp), 3)
    assert tuple(rows[0].shape) == (3, 4)


def test_adapter_tables_lever_off_never_caches(lever):
    lever("")
    ad = _adapter()
    main, comp = torch.arange(12, dtype=torch.int32), torch.arange(12, dtype=torch.int32)
    a = ad._block_tables(_snap(main, comp), 4)
    b = ad._block_tables(_snap(main, comp), 4)
    assert a[0] is not b[0] and "_l6_steady_tables" not in ad.__dict__


# ---------------------------------------------------------------- 7. lever off = identical behaviour


def test_lever_off_makes_no_steady_call(lever, monkeypatch):
    lever("")
    assert not l6_ab.ACTIVE
    _flags(monkeypatch, True)
    for fn in ("am_try", "am_record", "seq_lens_bound", "reorder_and_count", "reorder_try",
               "upstream_verdict", "upstream_checked", "drop"):
        monkeypatch.setattr(l6_steady, fn, lambda *a, fn=fn, **k: pytest.fail(f"{fn} called with the lever off"))
    w = World()
    for length in (10, 150):
        w.admit(length)
    r = _runner(w)
    out, ctx = _call(r, Step(w, 1), steady=False)
    assert len(out) == 2 * (N_ATTN + N_MAMBA) and "_l6_steady_am" not in r.__dict__
    batch = UpBatch(["a"], [100], [50])
    TPUModelRunner._may_reorder_batch(_up_runner(batch), SimpleNamespace(num_scheduled_tokens={"a": 1}))


def test_every_runner_call_site_is_guarded():
    """Each l6_steady call in tpu_runner.py runs only under `_l6_ab.ACTIVE` (or
    under `steady`, which only a lever-on `_prepare_inputs` can set)."""
    tree = ast.parse(pathlib.Path(tpu_runner.__file__).read_text())
    sites = []

    def visit(node, guards):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            visit(node.test, guards)
            for child in node.body:
                visit(child, guards + [test])
            for child in node.orelse:
                visit(child, guards + ["not (" + test + ")"])
            return
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            for i, value in enumerate(node.values):
                visit(value, guards + [ast.unparse(v) for v in node.values[:i]])
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "_l6_steady"
        ):
            sites.append((node.func.attr, node.lineno, guards))
        for child in ast.iter_child_nodes(node):
            visit(child, guards)

    visit(tree, [])
    assert len(sites) >= 9, sites
    for name, line, guards in sites:
        ok = any(g.startswith("_l6_ab.ACTIVE") or g == "steady" for g in guards if not g.startswith("not "))
        assert ok, (name, line, guards)


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation.tpu_compiler import _TPU_COMPILE_ENV_IGNORED

    assert l6_ab.LEVERS["steady"] == "VLLM_TPU_L6_STEADY_STEP"
    for name in ("VLLM_TPU_L6_STEADY_STEP", "VLLM_TPU_L6_STEADY_STEP_CHECK"):
        assert name in envs.environment_variables and name in _TPU_COMPILE_ENV_IGNORED
