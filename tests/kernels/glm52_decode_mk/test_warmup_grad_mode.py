# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0
"""glm52_decode_mk adapter: warmup compiles exactly what serving runs.

The step programs are torch.compile(fullgraph, dynamic=False) closures of one code
object. Dynamo guards each cache entry on the grad mode (GLOBAL_STATE) and on the
program's custom op (___check_obj_id). capture_model calls `warmup` with grad
enabled; serving calls the programs under no_grad. When warmup called the programs
directly, every program re-traced on its first served call (stageB/best-ab-lv:
0.8-1.7 s decode stalls inside the measured window).

CPU only. The programs are stand-ins with `_build_op`'s structure: one custom op per
program, called from a closure of one shared code object, compiled with a counting
backend. `warmup`, `decision`, `try_call` and `__call__` are the adapter's own."""

import dataclasses
import itertools

import pytest
import torch

from vllm_torchtpu.kernels.glm52_decode_mk import config as C
from vllm_torchtpu.kernels.glm52_decode_mk import vllm_adapter as A

ROWS = [16, 32]
H = 8
_OPS = {}
_UID = itertools.count()


def _op(name, ran, tag):
    """One custom op per program, as `_build_op` registers step_b<T>[f]. Opaque to
    dynamo, so its body records which program ran without putting a guarded side
    effect in the traced step."""
    if name not in _OPS:

        @torch.library.custom_op(f"glm52mk_warmtest::{name}", mutates_args=())
        def op(
            ids: torch.Tensor,
            qsl: torch.Tensor,
            smalls: torch.Tensor,
            caches: list[torch.Tensor],
        ) -> torch.Tensor:
            ran.append(tag)
            return ids.float()[:, None] * smalls.sum() + torch.zeros(ids.shape[0], H)

        @op.register_fake
        def _(ids, qsl, smalls, caches):
            return torch.empty((ids.shape[0], H), dtype=torch.float32)

        _OPS[name] = op
    return _OPS[name]


def _make_step(op, T, backend):
    # ONE code object for every program (as `step` inside `_build_op`): dynamo keys
    # the programs on op identity
    def step(ids, positions, qsl, seq_lens, bt_flat, smalls, weights, caches):
        h = op(ids.reshape(-1).to(torch.int32), qsl, smalls, list(caches))
        return h[:T] + positions.reshape(-1)[:T, None].float() * 0

    return torch.compile(step, backend=backend, fullgraph=True, dynamic=False)


class _MD:
    def __init__(self, S, W):
        self.query_start_loc = torch.zeros(S + 1, dtype=torch.int32)
        self.seq_lens = torch.zeros(S, dtype=torch.int32)
        self.block_tables = torch.zeros(S * W, dtype=torch.int32)


def _mk(graphs, lv_full=True):
    """A bound Glm52DecodeMK with LV's four programs (b16, b16 full, b32, b32 full)
    as compiled stand-ins."""

    def backend(gm, example_inputs):
        graphs.append(1)
        return gm.forward

    mk = object.__new__(A.Glm52DecodeMK)
    mk.cfg = dataclasses.replace(C.FULL, TMAX=max(ROWS), BM=8)
    mk.rows = list(ROWS)
    mk.rank = 0
    mk.enabled = True
    mk.counts = {r: 0 for r in A.REASONS}
    mk.decisions = mk.consumed = mk.calls = 0
    mk.log_every = 0
    mk.unbound = None
    mk.call_shapes, mk.warm_shapes = {}, {}
    mk.smalls = torch.ones(3)
    mk.dense, mk.experts = [torch.ones(2)], []
    mk.caches = [torch.zeros(4)]
    mk.max_pages = 256
    mk.lv_full = lv_full
    mk.ran = []
    uid = next(_UID)  # fresh ops per test: no entry from another test can match
    mk._steps = {
        t: _make_step(_op(f"s{uid}_b{t}", mk.ran, f"b{t}"), t, backend) for t in ROWS
    }
    mk._steps_full = (
        {
            t: _make_step(_op(f"s{uid}_b{t}f", mk.ran, f"b{t}f"), t, backend)
            for t in ROWS
        }
        if lv_full
        else {}
    )
    md = _MD(mk.cfg.BMAX, mk.max_pages)
    mk._metadata = lambda: md
    return mk


def _warmup_a66f6c89f(mk):
    """a66f6c89f's warmup loop: the programs called directly, in capture_model's
    grad mode."""
    cfg = mk.cfg
    for t in mk.rows:
        ids = torch.zeros(t, dtype=torch.int32)
        pos = torch.arange(t, dtype=torch.int32) % cfg.QMAX
        S = cfg.BMAX
        qsl = torch.tensor([0] + [cfg.QMAX] * S, dtype=torch.int32)
        seq = torch.tensor([cfg.QMAX] + [0] * (S - 1), dtype=torch.int32)
        bt = torch.zeros(S * mk.max_pages, dtype=torch.int32)
        for steps in (mk._steps, mk._steps_full):
            if t in steps:
                steps[t](
                    ids, pos, qsl, seq, bt, mk.smalls, mk.dense + mk.experts, mk.caches
                )


V = (4, 3, False)  # an MTP-3 verify request: 4 rows, 3 drafts, past its prompt
# (requests, runner token bucket, the program that must run): partial and full
# buckets of both row buckets
SERVED = [
    ([V], 4, "b16"),
    ([V] * 2, 8, "b16"),
    ([V] * 4, 16, "b16f"),
    ([V] * 5, 32, "b32"),
    ([V] * 8, 32, "b32f"),
    ([V] * 3, 16, "b16"),
]


def _serve(mk):
    """execute_model's grad mode (no_grad) around step_ok's decision and
    forward_model's try_call."""
    with torch.no_grad():
        for reqs, n_pad, want in SERVED:
            b, why, _ = mk.decision(reqs)
            assert why == "mk", (reqs, why)
            ids = torch.zeros(n_pad, dtype=torch.int32)
            n0 = len(mk.ran)
            out = mk.try_call(None, b, ids, ids.clone())
            assert out is not None and tuple(out.shape) == (n_pad, H), (reqs, out)
            assert mk.ran[n0:] == [want], (reqs, mk.ran[n0:], want)


@pytest.fixture(autouse=True)
def _fresh_dynamo(monkeypatch):
    torch._dynamo.reset()
    # the runner sets 1024
    monkeypatch.setattr(torch._dynamo.config, "cache_size_limit", 64)
    import vllm_torchtpu.utils as U

    # torch_tpu's sync
    monkeypatch.setattr(U, "synchronize_tensors", lambda tensors=None, wait=True: None)
    yield
    torch._dynamo.reset()


def test_served_call_after_warmup_compiles_nothing():
    graphs = []
    mk = _mk(graphs)
    assert torch.is_grad_enabled()  # capture_model's mode
    mk.warmup(None)
    assert len(graphs) == 4  # every built program compiled once
    n_warm = len(mk.ran)
    _serve(mk)
    assert len(graphs) == 4, "a served-mode call after warmup compiled a new graph"
    # warmed: each program ran once, then once more in the replay, in serving's
    # mode, which compiled nothing
    assert sorted(mk.ran[:n_warm]) == sorted(["b16", "b16f", "b32", "b32f"] * 2)
    assert mk.warm_replay_compiles == 0
    assert set(mk.warm_shapes) == set(ROWS)
    assert mk.calls == len(SERVED) and mk.counts["shape"] == 0
    # served from grad-enabled code too: __call__'s own no_grad sets the mode
    ids = torch.zeros(16, dtype=torch.int32)
    assert mk.try_call(None, A.StepBucket(16, True), ids, ids.clone()) is not None
    assert len(graphs) == 4


def test_without_lv_two_programs_warm_and_serve_without_compiles():
    graphs = []
    mk = _mk(graphs, lv_full=False)
    mk.warmup(None)
    assert len(graphs) == 2 and mk.warm_replay_compiles == 0
    _serve_plain = [(r, n, w.rstrip("f")) for r, n, w in SERVED]
    with torch.no_grad():
        for reqs, n_pad, want in _serve_plain:
            b, _, _ = mk.decision(reqs)
            assert not b.full
            n0 = len(mk.ran)
            assert (
                mk.try_call(
                    None,
                    b,
                    torch.zeros(n_pad, dtype=torch.int32),
                    torch.zeros(n_pad, dtype=torch.int32),
                )
                is not None
            )
            assert mk.ran[n0:] == [want]
    assert len(graphs) == 2


def test_positive_control_a66f6c89f_warmup_recompiles_every_program_when_served():
    """The test can see the defect: a66f6c89f's warmup leaves each of the 4 programs
    to re-trace on its first served call (the GLOBAL_STATE grad_mode guard, as in
    best-ab-lv's TORCH_LOGS=recompiles)."""
    graphs = []
    mk = _mk(graphs)
    _warmup_a66f6c89f(mk)
    assert len(graphs) == 4
    _serve(mk)
    assert len(graphs) == 8


def test_positive_control_replay_counts_a_mode_mismatch(monkeypatch):
    """The warmup's replay count is live: a first pass in another grad mode than the
    replay's is counted (4) and logged as a warning."""
    graphs = []
    mk = _mk(graphs)
    n = {"calls": 0}

    def run_flipped(step, ids, pos, qsl, seq, bt):
        n["calls"] += 1
        # first pass grad-enabled, replay no_grad
        with torch.set_grad_enabled(n["calls"] <= 4):
            return step(
                ids, pos, qsl, seq, bt, mk.smalls, mk.dense + mk.experts, mk.caches
            )

    mk._run = run_flipped
    warned = []
    monkeypatch.setattr(A.logger, "warning", lambda f, *a: warned.append(f % a))
    mk.warmup(None)
    assert mk.warm_replay_compiles == 4 and len(graphs) == 8
    assert len(warned) == 1 and "replay compiled 4 new graphs" in warned[0], warned
