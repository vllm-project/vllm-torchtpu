# SPDX-License-Identifier: Apache-2.0
"""q38 L6-H2 lever fskip: FENCE-SKIP (runner/l6_fskip.py). Host-only; runs in a CDK CPU pod.

The stream harness drives the real ``TPUModelRunner._prepare_inputs``, the real pre-forward glue
of execute_model (``_flush_mamba_state_seed_copies``, ``_reset_read_offsets_for_new_requests``,
the fused or separate token substitution, the deferred draft-id upload), the real
``_record_input_staging_fence`` and the real ``_modify_prev_results`` / ``_update_placeholder``
through async speculative decode steps (l6_carry_harness.py), with a fake device:

* every ``Tensor.to(<torch.device>)`` and every ``copy_(..., non_blocking=True)`` is counted as
  an upload (``.to`` returns a copy, so a reused upload keeps its values); an upload whose source
  storage is a runner-held host tensor (an attribute of the runner or its input batch, a reuse
  cache entry, the read-offset scratch) is also counted as a staging upload;
* ``torch.tpu.Event().record()`` is counted, so a skipped fence is observed directly.

At every fence point the harness reads the uploads since the previous fence point. The proof:
at every skipped fence that count is 0 (all uploads, so staging uploads too), over 25 streams
(5 batch shapes x 5 schedules: steady, a fixed churn sequence, two random churn mixes and a
burst), with CARRY + STEADY + fskip on; the step outputs equal the lever-off run's; steady
streams skip; carry off never skips; fskip off records every chunk.
"""

from __future__ import annotations

import ast
import os
import pathlib
import random
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import l6_carry_harness as H
from vllm_torchtpu.core import l6_ab
from vllm_torchtpu.runner import l6_fskip
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

ENV = "VLLM_TPU_L6_FENCE_SKIP"
SHAPES = {
    "b1_mamba_comb": dict(nreq=1, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b3_padded": dict(nreq=3, max_num_reqs=8, paddings=(16, 32, 64), mamba=True, comb=True),
    "b5_pooled": dict(nreq=5, max_num_reqs=8, paddings=(16, 32, 64), mamba=False, comb=True, pooled=True),
    "b8_full_nocomb": dict(nreq=8, max_num_reqs=8, paddings=(32, 64), mamba=True, comb=False),
    "b2_exact_bucket": dict(nreq=2, max_num_reqs=16, paddings=(8, 16, 64), mamba=True, comb=True),
}
EVENT_KINDS = ("join", "leave", "swap", "partial", "nonspec", "pad", "skew", "flush", "discard", "remap")
STEPS = 40


def _schedule(kind: str, seed: int) -> dict[int, tuple]:
    if kind == "steady":
        return {}
    if kind == "churn_fixed":
        return {6 + 3 * i: (e,) for i, e in enumerate(EVENT_KINDS)}
    if kind == "burst":
        return {s: (EVENT_KINDS[s % len(EVENT_KINDS)],) for s in range(10, 16)}
    rng = random.Random(seed)
    p = 0.35 if kind == "churn_a" else 0.6
    return {s: (rng.choice(EVENT_KINDS),) for s in range(3, STEPS) if rng.random() < p}


SCHEDULES = ("steady", "churn_fixed", "churn_a", "churn_b", "burst")
STREAMS = [(shape, sched) for shape in sorted(SHAPES) for sched in SCHEDULES]
assert len(STREAMS) == 25


class _Event:
    records = 0

    def record(self):
        _Event.records += 1

    def synchronize(self):
        pass


def _held_ranges(r) -> list[tuple[int, int]]:
    """Byte ranges of every host buffer the runner holds across steps (its attributes, its input
    batch's, one level into dicts / lists / tuples: reuse-cache entries, read-offset scratch)."""
    out: list[tuple[int, int]] = []

    def add(v, depth=0):
        if isinstance(v, torch.Tensor):
            if v.device.type == "cpu" and v.numel():
                st = v.untyped_storage()
                out.append((st.data_ptr(), st.data_ptr() + st.nbytes()))
        elif isinstance(v, np.ndarray):
            b = v
            while isinstance(b.base, np.ndarray):
                b = b.base
            if b.size:
                p = b.__array_interface__["data"][0]
                out.append((p, p + b.nbytes))
        elif depth < 2 and isinstance(v, (list, tuple)):
            for x in v:
                add(x, depth + 1)
        elif depth < 2 and isinstance(v, dict):
            for x in v.values():
                add(x, depth + 1)

    for v in list(vars(r).values()):
        add(v)
    ib = getattr(r, "input_batch", None)
    if ib is not None:
        for v in list(vars(ib).values()):
            add(v)
    return out


class Device:
    """Counts uploads (and staging uploads) and fence records while active."""

    def __init__(self, r):
        self.r = r
        self.total = 0
        self.staging = 0
        self.sites: list[str] = []
        self.paused = False

    def _count(self, src):
        if self.paused:
            return
        self.total += 1
        if isinstance(src, torch.Tensor) and src.device.type == "cpu" and src.numel():
            p = src.data_ptr()
            if any(a <= p < b for a, b in _held_ranges(self.r)):
                self.staging += 1
        import traceback

        fr = [f for f in traceback.extract_stack(limit=8)[:-2] if "vllm_torchtpu" in f.filename]
        self.sites.append(f"{pathlib.Path(fr[-1].filename).name}:{fr[-1].lineno}" if fr else "?")

    def since(self):
        t, s, sites = self.total, self.staging, list(self.sites)
        self.total = self.staging = 0
        self.sites = []
        return t, s, sites

    def __enter__(self):
        self._to = torch.Tensor.__dict__.get("to")
        self._copy = torch.Tensor.__dict__.get("copy_")
        orig_to, orig_copy = torch.Tensor.to, torch.Tensor.copy_
        me = self

        def to(t, *args, **kwargs):
            out = orig_to(t, *args, **kwargs)
            dev = args[0] if args else kwargs.get("device")
            if isinstance(dev, torch.device):
                me._count(t)
                if out is t:
                    out = t.clone()
            return out

        def copy_(t, src, *args, **kwargs):
            if kwargs.get("non_blocking") or (len(args) >= 1 and args[0] is True):
                me._count(src)
            return orig_copy(t, src, *args, **kwargs)

        torch.Tensor.to = to
        torch.Tensor.copy_ = copy_
        return self

    def __exit__(self, *exc):
        for name, prev in (("to", self._to), ("copy_", self._copy)):
            if prev is None:
                delattr(torch.Tensor, name)
            else:
                setattr(torch.Tensor, name, prev)
        return False


class FenceSim(H.Sim):
    """H.Sim with execute_model's pre-forward glue and the fence record in each step. The fake
    device stays active for the whole step, so a fence point sees every upload since the
    previous one (prepare, glue, the previous step's sampling side)."""

    def __init__(self, runner, dev: Device, **kw):
        self.dev = dev
        self.fences: list[dict] = []
        super().__init__(runner, **kw)

    def step(self, events=()):
        r, ib = self.r, self.r.input_batch
        ev = set(events)
        # keep every event applicable to the current batch
        if "join" in ev and ib.num_reqs >= r.max_num_reqs:
            ev.discard("join")
        if "leave" in ev and ib.num_reqs <= 1:
            ev.discard("leave")
        if "swap" in ev and ib.num_reqs < 2:
            ev.discard("swap")
        rng = self.rng
        join_rid = None
        if "join" in ev:
            join_rid = self._add(self.prompt - 5)
        if "leave" in ev:
            self.leave(0)
        if "swap" in ev:
            self.swap(0, ib.num_reqs - 1)
        if "skew" in ev:
            self.computed[ib.req_ids[0]] += 1
        if "flush" in ev:
            r._pre_async_results = None
        n = ib.num_reqs
        req_ids = list(ib.req_ids[:n])
        nsched, nd = {}, {}
        for i, rid in enumerate(req_ids):
            if rid == join_rid:
                nsched[rid], nd[rid] = 5, 0
            elif "nonspec" in ev:
                nsched[rid], nd[rid] = 1, 0
            elif "partial" in ev and i == 0:
                nsched[rid], nd[rid] = 2, 1
            else:
                nsched[rid], nd[rid] = H.STRIDE, H.K
            ib.num_computed_tokens_cpu[i] = self.computed[rid]
        r._dp_target_bucket = max(r.num_tokens_paddings) if "pad" in ev else None
        so = NS(
            total_num_scheduled_tokens=sum(nsched.values()),
            num_scheduled_tokens=dict(nsched),
            scheduled_spec_decode_tokens={rid: [0] * nd[rid] for rid in req_ids if nd[rid]},
        )
        if l6_ab.ACTIVE:
            l6_ab.begin_step()
        out = r._prepare_inputs(so, 0, 0, n)
        self.dev.paused = True
        dig = self._digest(out, n)
        self.dev.paused = False
        # execute_model's pre-forward glue (the real methods, in its order)
        r._flush_mamba_state_seed_copies()
        r._reset_read_offsets_for_new_requests(0, n)
        spec = out[7]
        postfwd_out = r.__dict__.pop("_postfwd_fused_out", None)
        if postfwd_out is not None and (postfwd_out[0] != 0 or spec is None):
            postfwd_out = None
        if postfwd_out is not None:
            spec.draft_token_ids = postfwd_out[2]
        else:
            r._apply_async_token_substitution(r.input_ids, out[5], out[6])
            if spec is not None and spec.draft_token_ids is None and not len(out[5]):
                spec.draft_token_ids = spec.upload_deferred_draft_token_ids(r.device)
        rec0 = _Event.records
        before = (l6_fskip.S.n, l6_fskip.S.carried)
        t, s, sites = self.dev.since()
        r._record_input_staging_fence()
        self.fences.append(dict(step=self.n, skipped=_Event.records == rec0, total=t, staging=s,
                                sites=sites, prod=before, events=sorted(ev)))
        # sample_tokens side
        rej = [int(rng.integers(0, nd[rid] + 1)) for rid in req_ids]
        r._modify_prev_results()
        discard = [n - 1] if "discard" in ev else []
        src_pos = list(range(n))
        if "remap" in ev:
            src_pos = [(i + 1) % n for i in range(n)]
        self._results(nd=[nd[rid] for rid in req_ids], rej=rej, discard=discard, src_pos=src_pos)
        for i, rid in enumerate(req_ids):
            self.computed[rid] += nsched[rid] - self.last_rej.get(rid, 0)
        self.last_rej = {rid: rej[i] for i, rid in enumerate(req_ids)}
        r._dp_target_bucket = None
        self.n += 1
        return dig


@pytest.fixture
def env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_") or k in ("VLLM_TPU_HOST_AB_PERIOD_MS", "VLLM_TPU_D1N_AB_PERIOD_MS"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_TPU_POSTFWD_FUSED", "1")
    monkeypatch.setattr(torch, "tpu", NS(Event=_Event), raising=False)
    sub = tpu_runner._substitute_placeholder_token
    eager = getattr(sub, "_torchdynamo_orig_callable", None) or getattr(sub, "__wrapped__", None)
    if eager is not None:
        monkeypatch.setattr(tpu_runner, "_substitute_placeholder_token", eager)
    undo = H.install_eager_programs()
    l6_fskip.S.n, l6_fskip.S.carried = 0, False

    def lever(**kv):
        for k in list(os.environ):
            if k.startswith("VLLM_TPU_L6_"):
                monkeypatch.delenv(k, raising=False)
        for k, v in kv.items():
            monkeypatch.setenv(k, v)
        l6_ab._resolve()
        l6_fskip.S.n, l6_fskip.S.carried = 0, False

    yield NS(lever=lever, mp=monkeypatch)
    undo()
    for k in list(os.environ):
        if k.startswith("VLLM_TPU_L6_"):
            monkeypatch.delenv(k, raising=False)
    l6_ab._resolve()


def run_stream(shape_name: str, sched: str, steps: int = STEPS):
    shape = SHAPES[shape_name]
    seed = sorted(SHAPES).index(shape_name) * 10 + SCHEDULES.index(sched)
    r = H.make_runner(seed=seed, **{k: v for k, v in shape.items() if k != "pooled"})
    r._pending_mamba_state_copies = []
    r._read_offset_scratch = {}
    r._dp_lockstep_enabled = lambda: False
    if r.mamba_slot_read_offsets is not None:
        # indexed by slot id (up to 1 + 4 * STRIDE * max_num_reqs in the harness's slot pool)
        r.mamba_slot_read_offsets = torch.zeros(4 * H.STRIDE * r.max_num_reqs + 8, dtype=torch.int32)
    dev = Device(r)
    events = _schedule(sched, seed)
    with dev:
        sim = FenceSim(r, dev, nreq=shape.get("nreq", 3), seed=seed, pooled=shape.get("pooled", False))
        digests = [sim.step(events.get(sim.n, ())) for _ in range(steps)]
    return digests, sim


ON = {"VLLM_TPU_L6_DECODE_CARRY": "1", "VLLM_TPU_L6_STEADY_STEP": "1", ENV: "1"}


@pytest.mark.parametrize("stream", STREAMS, ids=[f"{a}-{b}" for a, b in STREAMS])
def test_no_upload_since_the_previous_fence_point_at_any_skipped_fence(env, stream):
    shape_name, sched = stream
    env.lever(**{k: v for k, v in ON.items() if k != ENV})
    base, _ = run_stream(shape_name, sched)
    env.lever(**ON)
    got, sim = run_stream(shape_name, sched)
    assert H.first_difference(base, got) is None
    bad = [f for f in sim.fences if f["skipped"] and (f["total"] or f["staging"])]
    assert not bad, bad[:3]
    skipped = sum(f["skipped"] for f in sim.fences)
    rec_up = [f for f in sim.fences if not f["skipped"] and f["total"]]
    staging = sum(f["staging"] for f in sim.fences)
    print(f"FSKIP_STREAM {shape_name}-{sched} fences={len(sim.fences)} skipped={skipped} "
          f"recorded_with_uploads={len(rec_up)} uploads={sum(f['total'] for f in sim.fences)} "
          f"staging_uploads={staging}")
    if sched == "steady" and SHAPES[shape_name]["comb"]:
        # Every step after the carry's first record (its reuse slots fill on the first carried step).
        assert skipped >= STEPS - 4, [(f["step"], f["skipped"], f["total"], f["sites"]) for f in sim.fences[:6]]
    elif sched == "steady":
        # No hybrid read offsets (harness comb=False): _prepare_inputs uploads the RPA request
        # distribution on every chunk (no reuse helper), so every fence records with it counted.
        assert skipped == 0 and len(rec_up) == len(sim.fences), [(f["step"], f["total"]) for f in sim.fences[:6]]
    else:
        assert rec_up, "a churn stream must record fences with uploads"
    if sched != "steady":
        assert staging > 0  # the staging classifier sees the runner's host tensors


def test_churn_streams_have_skips_and_uploads(env):
    """The churn proof is not vacuous: on a churn stream the lever both skips (steady stretches)
    and records with staging uploads counted since the previous fence point."""
    env.lever(**ON)
    _, sim = run_stream("b3_padded", "churn_fixed", steps=48)
    assert any(f["skipped"] for f in sim.fences)
    assert any((not f["skipped"]) and f["staging"] > 0 for f in sim.fences)


def test_carry_off_never_skips(env):
    env.lever(VLLM_TPU_L6_STEADY_STEP="1", **{ENV: "1"})
    _, sim = run_stream("b3_padded", "steady")
    assert not any(f["skipped"] for f in sim.fences)
    assert all(f["prod"][1] is False for f in sim.fences)


def test_lever_off_records_every_chunk(env):
    env.lever(**{k: v for k, v in ON.items() if k != ENV})
    rec0 = _Event.records
    _, sim = run_stream("b3_padded", "steady")
    assert _Event.records - rec0 == STEPS
    assert not any(f["skipped"] for f in sim.fences)


def test_ab_windows_skip_only_in_on_windows(env):
    clock = {"t": 0}
    env.mp.setattr(l6_ab, "_now_ns", lambda: clock["t"])
    env.lever(VLLM_TPU_L6_DECODE_CARRY="1", VLLM_TPU_L6_STEADY_STEP="1", **{ENV: "ab:1"},
              VLLM_TPU_L6_AB_PERIOD_MS="20000")
    P = 20000
    seen = []
    real_begin = l6_ab.begin_step

    def begin(*a, **k):
        i = len(seen)
        clock["t"] = int(((4_000_000 + (i // 5)) + 0.5) * P * 1_000_000)  # 5-step windows
        real_begin(*a, **k)
        seen.append(l6_ab.on("fskip"))

    env.mp.setattr(l6_ab, "begin_step", begin)
    _, sim = run_stream("b3_padded", "steady", steps=30)
    assert any(seen) and not all(seen)
    for f, on in zip(sim.fences, seen):
        assert not f["skipped"] or on, (f, on)
    assert any(f["skipped"] for f in sim.fences)


# ------------------------------------------------------------------ the rule itself


def test_skip_rule_and_reset(env):
    env.lever(**{ENV: "1"})
    l6_ab.begin_step()
    S = l6_fskip.S
    for carried, n, want in ((True, 0, True), (False, 0, False), (True, 1, False), (False, 3, False)):
        S.carried, S.n = carried, n
        assert l6_fskip.skip() is want, (carried, n)
        assert S.n == 0 and S.carried is False
    # A counted upload is forgotten at the next record-site visit, recorded or not.
    S.carried, S.n = True, 2
    assert l6_fskip.skip() is False
    S.carried = True
    assert l6_fskip.skip() is True
    env.lever()
    l6_ab.begin_step()
    S.carried, S.n = True, 0
    assert l6_fskip.skip() is False and S.carried is False


def test_record_site_returns_before_the_event(env):
    env.lever(**{ENV: "1"})
    l6_ab.begin_step()
    r = NS(_input_staging_fence=None)
    l6_fskip.S.carried, l6_fskip.S.n = True, 0
    rec0 = _Event.records
    TPUModelRunner._record_input_staging_fence(r)
    assert _Event.records == rec0 and r._input_staging_fence is None
    TPUModelRunner._record_input_staging_fence(r)  # not carried: records
    assert _Event.records == rec0 + 1 and r._input_staging_fence is not None


# ------------------------------------------------------------------ per-site marks


def test_reuse_helpers_count_uploads_not_hits(env):
    r = NS(_h2d_reuse={}, device=torch.device("cpu"))
    host = torch.arange(6, dtype=torch.int32)
    l6_fskip.S.n = 0
    TPUModelRunner._upload_unless_unchanged(r, "s", host)
    assert l6_fskip.S.n == 1
    TPUModelRunner._upload_unless_unchanged(r, "s", host)
    assert l6_fskip.S.n == 1
    host[0] = 9
    TPUModelRunner._upload_unless_unchanged(r, "s", host)
    assert l6_fskip.S.n == 2
    TPUModelRunner._upload_keyed(r, "k", host, b"a")
    TPUModelRunner._upload_keyed(r, "k", host, b"a")
    assert l6_fskip.S.n == 3


def test_carry_upload_counts(env):
    from vllm_torchtpu.runner import decode_carry

    r = NS(input_ids_cpu=torch.zeros(8, dtype=torch.int32), positions_cpu=torch.zeros(8, dtype=torch.int32),
           seq_lens_cpu=torch.zeros(8, dtype=torch.int32), device=torch.device("cpu"))
    l6_fskip.S.n = 0
    # q38 L6-V6 compose: decode_carry.StepCtx carries PREP-SHORT's materialize slot (None unless the short path set it).
    decode_carry.upload(r, NS(token_slice=slice(0, 4), target=4, materialize=None))
    assert l6_fskip.S.n >= 1


def test_spec_metadata_uploads_count(env):
    from vllm_torchtpu.runner.speculative_decoding_manager import SpecDecodeMetadata

    md = SpecDecodeMetadata.__new__(SpecDecodeMetadata)
    md.deferred_draft_token_ids_cpu = np.zeros(8, dtype=np.int32)
    l6_fskip.S.n = 0
    SpecDecodeMetadata.upload_deferred_draft_token_ids(md, torch.device("cpu"))
    assert l6_fskip.S.n == 1


def test_block_table_staging_upload_counts(env):
    from vllm_torchtpu.layers.core import attention_metadata as am

    l6_fskip.S.n = 0
    am._inc_upload(torch.zeros(16, dtype=torch.int32), torch.device("cpu"))
    assert l6_fskip.S.n == 1


def test_read_offset_reset_counts_on_a_new_request(env):
    env.lever(VLLM_TPU_L6_DECODE_CARRY="1", **{ENV: "1"})
    _, sim = run_stream("b3_padded", "churn_fixed", steps=12)  # a join at step 6
    joins = [f for f in sim.fences if "join" in f["events"]]
    assert joins and all(not f["skipped"] for f in joins)
    assert any("tpu_runner.py" in s for f in joins for s in f["sites"])


# ------------------------------------------------------------------ source scan


SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "vllm_torchtpu"
# (relative file, function) -> (number of non-blocking host-to-device calls, reason): calls that need
# no mark. The count is exact, so a new upload in one of these functions still fails the scan.
EXEMPT = {
    ("models/vllm/qwen4_exp/inputs.py", "_speculative_ngram_context"):
        (1, "adapter context row ids from a fresh tensor per call; never rewritten"),
    ("models/vllm/qwen4_exp/inputs.py", "_prepare_inputs_eager"):
        (1, "adapter n-gram window from a fresh array per call; never rewritten"),
    ("models/vllm/qwen4_exp/inputs.py", "_host_upload"):
        (1, "adapter temporaries kept alive for 8 calls by _host_keep; never rewritten"),
    ("runner/tpu_pooling.py", "forward"): (1, "pooling models only; never on an async-spec carried chunk"),
    ("runner/tpu_pooling.py", "_to_device"): (1, "pooling models only; never on an async-spec carried chunk"),
}
_EXITS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def _is_mark(node) -> bool:
    return (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Attribute)
            and node.target.attr == "n" and isinstance(node.target.value, ast.Attribute)
            and node.target.value.attr == "S" and isinstance(node.target.value.value, ast.Name)
            and node.target.value.value.id == "_l6_fskip")


def _non_blocking(call) -> bool:
    """A call that may be non-blocking: a ``non_blocking=`` keyword that is not the literal False
    (``non_blocking=raw.is_pinned()`` counts), or ``copy_(src, <not False>)``."""
    for k in call.keywords:
        if k.arg == "non_blocking":
            return not (isinstance(k.value, ast.Constant) and k.value.value is False)
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr == "copy_" and len(call.args) >= 2:
        a = call.args[1]
        return not (isinstance(a, ast.Constant) and a.value is False)
    return False


def _is_d2h(call) -> bool:
    f = call.func
    if isinstance(f, ast.Attribute) and f.attr == "to" and call.args:
        a = call.args[0]
        if isinstance(a, ast.Constant) and a.value == "cpu":
            return True
    if isinstance(f, ast.Attribute) and f.attr == "copy_":
        recv = f.value
        if isinstance(recv, ast.Call) and getattr(recv.func, "id", "") == "_pinned_buffer":
            return True
    return False


def _blocks(st):
    out = []
    for name in ("body", "orelse", "finalbody"):
        b = getattr(st, name, None)
        if isinstance(b, list) and b and isinstance(b[0], ast.stmt):
            out.append(b)
    for h in getattr(st, "handlers", None) or []:
        out.append(h.body)
    for c in getattr(st, "cases", None) or []:
        out.append(c.body)
    return out


def _own_nodes(st):
    """The statement's own expressions (a compound statement's header, not its blocks)."""
    if not _blocks(st):
        return list(ast.walk(st))
    out = []
    for name, val in ast.iter_fields(st):
        if name in ("body", "orelse", "finalbody", "handlers", "cases"):
            continue
        vals = val if isinstance(val, list) else [val]
        for v in vals:
            if isinstance(v, ast.AST):
                out.extend(ast.walk(v))
    return out


def _has_exit(st) -> bool:
    """Whether a statement can leave its block early (return / raise / continue / break anywhere
    inside it, nested functions and classes excluded)."""
    stack = [st]
    while stack:
        n = stack.pop()
        if isinstance(n, _EXITS):
            return True
        if n is not st and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return False


def _h2d_calls(tree):
    """(call, chain, function name) for every possibly non-blocking call that is not a
    device-to-host copy; chain = [(block, index of the statement holding the call or its
    ancestor)] from the function body down to the innermost block."""
    found = []

    def visit(stmts, path, func):
        for i, st in enumerate(stmts):
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(st.body, [], st.name)
                continue
            if isinstance(st, ast.ClassDef):
                visit(st.body, [], func)
                continue
            here = path + [(stmts, i)]
            for node in _own_nodes(st):
                if isinstance(node, ast.Call) and _non_blocking(node) and not _is_d2h(node):
                    found.append((node, here, func))
            for b in _blocks(st):
                visit(b, here, func)

    visit(tree.body, [], "<module>")
    return found


def _marked(chain) -> str | None:
    """How the upload is counted, or None. Sound rules only (the mark runs whenever the upload
    does): "before" = a mark statement precedes the upload's statement (or its ancestor) in an
    enclosing block; "after" = a mark follows the upload's statement in its own block with no
    statement in between that can leave the block (return / raise / continue / break). A mark in
    a sibling branch, or after an early return, does not count."""
    for stmts, i in chain:
        if any(_is_mark(s) for s in stmts[:i]):
            return "before"
    stmts, i = chain[-1]
    for s in stmts[i + 1:]:
        if _is_mark(s):
            return "after"
        if _has_exit(s):
            break
    return None


def _scan():
    missing, marked, exempt = [], [], {}
    for path in sorted(SRC.rglob("*.py")):
        rel = path.relative_to(SRC).as_posix()
        text = path.read_text()
        if "non_blocking" not in text:
            continue
        for call, chain, func in _h2d_calls(ast.parse(text)):
            if (rel, func) in EXEMPT:
                exempt.setdefault((rel, func), []).append(call.lineno)
                continue
            (marked if _marked(chain) else missing).append((rel, func, call.lineno))
    return missing, marked, exempt


def test_every_host_to_device_upload_is_marked_or_exempt():
    missing, marked, exempt = _scan()
    print(f"FSKIP_SCAN marked={len(marked)} exempt={sum(len(v) for v in exempt.values())} missing={len(missing)}")
    assert not missing, missing
    assert {k: len(v) for k, v in exempt.items()} == {k: v[0] for k, v in EXEMPT.items()}, exempt
    assert len(marked) >= 55, len(marked)


def _verdicts(src):
    return [_marked(chain) for _, chain, _ in _h2d_calls(ast.parse(src))]


def test_scan_sees_a_missing_mark():
    """The scan's own mutants: an unmarked upload, a mark in a sibling branch, a mark after an
    early return and a mark behind a return in the same block are all missing; a mark before
    the upload in an enclosing block, or right after it, counts; non-literal non_blocking and a
    positional copy_ flag are scanned; device-to-host copies and blocking calls are not."""
    assert _verdicts("def f(self, t):\n    x = t.to(self.device, non_blocking=True)\n    return x\n") == [None]
    assert _verdicts("def f(self, t, c):\n    _l6_fskip.S.n += 1\n    if c:\n"
                     "        x = t.to(self.device, non_blocking=True)\n    return x\n") == ["before"]
    assert _verdicts("def f(self, t):\n    x = t.to(self.device, non_blocking=True)\n"
                     "    y = 1\n    _l6_fskip.S.n += 1\n    return x\n") == ["after"]
    # a mark in a sibling branch
    assert _verdicts("def f(self, t, c):\n    if c:\n        _l6_fskip.S.n += 1\n    else:\n"
                     "        x = t.to(self.device, non_blocking=True)\n    return x\n") == [None]
    # an early return before the function-level mark (the old scan accepted this)
    assert _verdicts("def f(self, t, c):\n    if c:\n        return t.to(self.device, non_blocking=True)\n"
                     "    _l6_fskip.S.n += 1\n    return t.to(self.device, non_blocking=True)\n") == [None, "before"]
    # a mark that cannot run after the upload's block returns
    assert _verdicts("def f(self, t, c):\n    x = t.to(self.device, non_blocking=True)\n"
                     "    if c:\n        return x\n    _l6_fskip.S.n += 1\n    return x\n") == [None]
    # a conditional mark next to an unconditional upload
    assert _verdicts("def f(self, t, c):\n    if c:\n        _l6_fskip.S.n += 1\n"
                     "    return t.to(self.device, non_blocking=True)\n") == [None]
    # non-literal non_blocking and a positional copy_ flag are scanned
    assert _verdicts("def f(self, t):\n    return t.to(self.device, non_blocking=t.is_pinned())\n") == [None]
    assert _verdicts("def f(self, d, t):\n    d.copy_(t, True)\n") == [None]
    # not scanned: device-to-host, an explicit non_blocking=False, a blocking call
    assert _h2d_calls(ast.parse("def f(t):\n    return t.to('cpu', non_blocking=True)\n")) == []
    assert _h2d_calls(ast.parse("def f(self, t):\n    return t.to(self.device, non_blocking=False)\n")) == []
    assert _h2d_calls(ast.parse("def f(self, t):\n    return t.to(self.device)\n")) == []


# ------------------------------------------------------------------ registration and import origin


def test_lever_is_registered_and_compile_ignored():
    from vllm_torchtpu import envs
    from vllm_torchtpu.compilation import tpu_compiler

    assert l6_ab.LEVERS["fskip"] == ENV and l6_fskip.LEVER == "fskip"
    assert ENV in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert hasattr(envs, ENV)


def test_import_resolves_to_the_tree_under_test():
    import vllm_torchtpu

    want = os.environ.get("Q38_L6_EXPECT_TREE")
    if not want:
        pytest.skip("Q38_L6_EXPECT_TREE not set (the CPU harness sets it)")
    got = os.path.realpath(vllm_torchtpu.__file__)
    assert got.startswith(os.path.realpath(want) + os.sep), (got, want)
