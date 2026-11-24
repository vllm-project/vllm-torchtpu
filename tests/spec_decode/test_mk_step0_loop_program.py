# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM: MK draft step 0 and the stock fused loop
(steps 1..K-1) as one compiled program instead of two.

On CPU the programs run eagerly on a stub draft whose every output depends on the
values each step reads (positions, the step's query starts and seq_lens, the
carried hidden state), so a wrong metadata source, a missed in-place seq_lens bump
or a misrouted carry changes the tokens. The checks: flag ON equals flag OFF
bitwise (draft tokens, the hidden carry and positions each loop step reads, the
seq_lens it attends with) across request buckets 4..32 and bucket switches, K=3,
with and without VLLM_TPU_SPEC_TAIL_STACK, sync and async; the ON path launches
one program where OFF launches two (MK step 0, then the fused loop); chunks the
program cannot take and step shapes warmup did not compile run the two programs.
"""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.spec_decode.test_eagle3 import _make_chunk, _make_proposer
from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode import eagle3
from vllm_torchtpu.spec_decode import mtp_fused_loop as fl
from vllm_torchtpu.spec_decode.utils import DraftStepTokens

K = 3
HIDDEN = 8
VOCAB = 97
PADDINGS = [1, 2, 4, 8, 16, 32, 64, 128]
# The forward context the stub draft reads (set_forward_context stand-in).
_CTX: list[dict] = []


def _bucket(n: int) -> int:
    return next(b for b in PADDINGS if b >= n)


class _StubMtp(torch.nn.Module):
    """A draft whose step-0 kernel and loop forward depend on everything they read."""

    def __init__(self):
        super().__init__()
        self.model = SimpleNamespace(mk_draft=SimpleNamespace(attn_prefix="attn"))
        g = torch.Generator().manual_seed(5)
        self.head = torch.randn(HIDDEN, VOCAB, generator=g)
        self.trace: list[tuple] = []

    def mk_draft_step(self, input_ids, positions, hidden, rows_per_request,
                      query_start_loc=None, seq_lens=None):
        md = _CTX[-1]["attn"]
        qsl = md.query_start_loc if query_start_loc is None else query_start_loc
        seq = md.seq_lens if seq_lens is None else seq_lens
        req = torch.clamp(torch.arange(input_ids.shape[0]) // rows_per_request,
                          max=seq.shape[0] - 1)
        s = seq[req] * 3 + qsl[req + 1]
        shift = (positions + s).to(hidden.dtype)[:, None]
        carry = torch.tanh(hidden * 1.3 + shift * 0.01)
        tokens = ((input_ids.long() * 7 + positions.long() + s.long()) % VOCAB).to(
            torch.int32
        )
        return carry, tokens

    def forward(self, input_ids, positions, hidden, intermediate_tensors=None,
                inputs_embeds=None, spec_step_idx=0):
        seq = _CTX[-1]["attn"].seq_lens
        n = input_ids.shape[0]
        sl = seq[:n]
        self.trace.append((spec_step_idx, input_ids.clone(), positions.clone(),
                           hidden.clone(), sl.clone()))
        bias = (positions + sl * 5 + spec_step_idx).to(hidden.dtype)[:, None]
        return torch.tanh(hidden * 0.9 + bias * 0.003 + input_ids[:, None] * 1e-3)

    def compute_logits(self, hidden):
        return hidden @ self.head


def _module(cls, mtp, **attrs):
    m = cls.__new__(cls)
    torch.nn.Module.__init__(m)
    m.prefix, m.mtp = "", mtp
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


class _Counted:
    def __init__(self, module):
        self.module, self.calls = module, 0

    def __call__(self, *args):
        self.calls += 1
        return self.module.forward(*args)


@pytest.fixture
def env(monkeypatch):
    for name in ("set_model_tag", "set_vllm_model_wrapper_context"):
        monkeypatch.setattr(eagle3, name, lambda *a, **k: contextlib.nullcontext())

    @contextlib.contextmanager
    def fwd_ctx(md, *a, **k):
        _CTX.append(md)
        try:
            yield
        finally:
            _CTX.pop()

    monkeypatch.setattr(eagle3, "set_forward_context", fwd_ctx)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_COMPILED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_LOOP", False, raising=False)
    return monkeypatch


def _proposer(cap: int):
    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.speculative_config.num_speculative_tokens = K
    proposer._draft_rows = True
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=0),
        max_num_reqs=cap,
        num_reqs_max_model_len=cap,
        num_reqs_most_model_len=cap,
        num_tokens_paddings=PADDINGS,
        device=torch.device("cpu"),
        uses_mrope=False,
        mesh=None,
        warmup_done=False,
        _dp_lockstep_enabled=lambda: False,
        _dp_num_tokens_across_dp=lambda n: None,
    )
    mtp = _StubMtp()
    proposer.draft_model = mtp
    proposer.draft_vllm_config = None
    proposer._dp_lockstep_sharded = lambda: False
    proposer._mk_draft_step0_ready = lambda chunk, num_chunks, num_tokens=0: True
    proposer._draft_moe_ctx = contextlib.nullcontext
    proposer._mk_draft_step0 = _Counted(
        _module(fl._MkDraftStep0, mtp, rows_per_request=4)
    )
    proposer._mk_draft_step0_warm = True
    proposer._mtp_fused_loop = _Counted(
        _module(fl._MtpFusedLoop, mtp, num_steps=K - 1, use_mk=False)
    )
    proposer._mtp_fused_loop_warm = True
    proposer._mk_step0_loop_program = _Counted(
        _module(fl._MkDraftStep0LoopProgram, mtp, num_steps=K - 1, rows_per_request=4)
    )
    proposer._mk_step0_loop_program_calls = 0
    return proposer


def _step(proposer, num_reqs: int, seed: int, is_async: bool):
    """One verify step of `num_reqs` requests (4 rows each) for `propose`."""
    cap = proposer.runner.num_reqs_max_model_len
    g = torch.Generator().manual_seed(seed)
    p = _bucket(num_reqs)
    tokens = _bucket(4 * num_reqs)
    input_ids = torch.randint(0, VOCAB, (tokens,), generator=g, dtype=torch.int32)
    positions = torch.randint(100, 10000, (tokens,), generator=g, dtype=torch.int32)
    hidden = torch.randn(tokens, HIDDEN, generator=g)
    last = torch.zeros(p, dtype=torch.int32)
    last[:num_reqs] = torch.arange(num_reqs, dtype=torch.int32) * 4 + 3
    rejected = torch.randint(0, K + 1, (num_reqs,), generator=g, dtype=torch.int32)
    base_seq = torch.zeros(cap, dtype=torch.int32)
    base_seq[:num_reqs] = torch.randint(50, 5000, (num_reqs,), generator=g,
                                        dtype=torch.int32)
    qsl0 = torch.from_numpy(
        np.minimum(np.arange(cap + 1, dtype=np.int32) * 4, 4 * num_reqs)
    )
    chunk = _make_chunk(
        input_ids=input_ids,
        position_ids=positions,
        query_start_loc_np=np.arange(0, 4 * num_reqs + 1, 4, dtype=np.int32),
        start_index=0,
        num_reqs=num_reqs,
        hidden=HIDDEN,
        device=torch.device("cpu"),
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        hidden_states=hidden,
    )
    proposer.draft_chunks = [chunk]
    proposer.runner.input_batch.num_reqs = num_reqs

    def prepare(chunk, *_a, **_k):
        if is_async:
            nr = torch.zeros(p, dtype=torch.int32)
            nr[:num_reqs] = rejected
        else:
            nr = rejected.numpy().copy()
        return input_ids, positions, last, nr

    def build_md(*, chunk, step_idx, seq_lens_delta, num_rejected_np,
                 num_tokens_padded=None, num_rejected_dev=None,
                 loop_query_start_loc=None, loop_request_distribution=None):
        if step_idx == 0:
            md = SimpleNamespace(query_start_loc=qsl0, seq_lens=base_seq)
        else:
            if num_rejected_dev is not None:
                rej = num_rejected_dev[:cap]
            elif num_rejected_np is not None:
                rej = torch.zeros(cap, dtype=torch.int32)
                rej[:num_reqs] = torch.from_numpy(num_rejected_np)
            else:
                rej = 0
            md = SimpleNamespace(
                query_start_loc=loop_query_start_loc,
                seq_lens=base_seq + seq_lens_delta - rej,
            )
        return {"attn": md, "attn.raw_ring": md}

    proposer._prepare_draft_inputs = prepare
    proposer._build_draft_attn_metadata = build_md
    return SimpleNamespace(
        next_tokens=[torch.zeros(p, K + 1, dtype=torch.int32)] if is_async else None
    )


def _propose(proposer, step, is_async: bool):
    proposer.draft_model.trace.clear()
    out = proposer.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
        return_device=is_async,
        next_tokens_per_chunk=step.next_tokens,
    )
    if isinstance(out, DraftStepTokens):
        out = out.materialize()
    elif isinstance(out, list):
        out = torch.tensor(out, dtype=torch.int32)
    # The last call's loop steps (a warmup call runs the program twice: compile,
    # then the compiled rerun).
    return out, list(proposer.draft_model.trace)[-(K - 1):]


def _same(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape)
    if a.is_floating_point():
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
    else:
        assert torch.equal(a, b)


def _same_loop_trace(off, on, n):
    """Each loop step reads the same ids, positions, carried hidden state and
    seq_lens for the chunk's requests (rows past them are padding)."""
    assert [t[0] for t in off] == [t[0] for t in on] == list(range(1, K))
    for a, b in zip(off, on):
        for x, y in zip(a[1:], b[1:]):
            _same(x[:n], y[:n])


def _run_pair(env, cap, order, *, is_async, stack, rows_fresh=False):
    env.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", stack, raising=False)
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", rows_fresh, raising=False)
    off_p, on_p = _proposer(cap), _proposer(cap)
    for i, n in enumerate(order):
        env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", False, raising=False)
        step = _step(off_p, n, 100 * i + n, is_async)
        off, off_trace = _propose(off_p, step, is_async)
        env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", True, raising=False)
        on, on_trace = _propose(on_p, _step(on_p, n, 100 * i + n, is_async), is_async)
        assert off.shape == (n, K)
        _same(off, on)
        _same_loop_trace(off_trace, on_trace, n)
    return off_p, on_p


ORDERS = {
    4: [4, 1, 3, 2, 4],
    8: [8, 3, 5, 1, 8, 6],
    16: [16, 3, 9, 1, 8, 5, 16, 2, 12],
    32: [32, 17, 9, 1, 24, 5, 32, 16, 3],
}


@pytest.mark.parametrize("rows_fresh", [False, True])
@pytest.mark.parametrize(
    "is_async,stack", [(True, True), (True, False), (False, False)]
)
@pytest.mark.parametrize("cap", [4, 8, 16, 32])
def test_on_matches_off_across_buckets(env, cap, is_async, stack, rows_fresh):
    off_p, on_p = _run_pair(env, cap, ORDERS[cap], is_async=is_async, stack=stack,
                            rows_fresh=rows_fresh)
    steps = len(ORDERS[cap])
    # OFF: two programs per step; ON: one, and neither of the two.
    assert off_p._mk_draft_step0.calls == steps
    assert off_p._mtp_fused_loop.calls == steps
    assert off_p._mk_step0_loop_program.calls == 0
    # Every (step-0 tokens, loop width) key was compiled once (warmup: the trace
    # call plus the compiled rerun) and reused.
    keys = {(_bucket(4 * n), cap) for n in ORDERS[cap]}
    assert on_p._mk_step0_loop_keys == keys
    assert on_p._mk_step0_loop_program.calls == steps + len(keys)
    assert on_p._mk_draft_step0.calls == 0
    assert on_p._mtp_fused_loop.calls == 0


def test_on_returns_capacity_steps_for_spec_tail(env):
    """Async + SPEC_TAIL_STACK: ON hands the spec tail K unstacked [capacity]
    tokens (the loop's width), which materialize to OFF's drafts."""
    env.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", True, raising=False)
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", True, raising=False)
    prop = _proposer(16)
    step = _step(prop, 5, 1, True)
    out = prop.propose(None, [], None, SimpleNamespace(num_scheduled_tokens={}),
                       return_device=True, next_tokens_per_chunk=step.next_tokens)
    assert isinstance(out, DraftStepTokens)
    assert [tuple(t.shape) for t in out.steps] == [(16,)] * K
    assert all(t._base is None for t in out.steps)


def test_capacity_not_a_loop_bucket_runs_two_programs(env, monkeypatch):
    """Capacity 12 is no loop bucket: the program declines, step 0 and the loop
    run as two programs, same drafts."""
    off_p, on_p = _run_pair(env, 12, [8, 5, 3], is_async=True, stack=True)
    assert on_p._mk_step0_loop_program.calls == 0
    assert on_p._mk_draft_step0.calls == 3 and on_p._mtp_fused_loop.calls == 3


def test_unwarmed_shape_after_warmup_falls_back(env, caplog):
    """After warmup a step shape the program never compiled runs the two programs
    (one warning per shape), never a serve-time compile; a warmed shape keeps the
    one program."""
    env.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", True, raising=False)
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", False, raising=False)
    ref_p = _proposer(16)
    refs = {n: _propose(ref_p, _step(ref_p, n, n, True), True)[0] for n in (16, 3)}
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", True, raising=False)
    prop = _proposer(16)
    # Warmup compiles the 16-request shape only.
    _propose(prop, _step(prop, 16, 16, True), True)
    assert prop._mk_step0_loop_keys == {(64, 16)}
    prop.runner.warmup_done = True
    eagle3.logger.propagate = True
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            out, _ = _propose(prop, _step(prop, 3, 3, True), True)
            _same(out, refs[3])
    assert prop._mk_step0_loop_program.calls == 2  # warmup: trace + rerun
    assert prop._mk_draft_step0.calls == 2 and prop._mtp_fused_loop.calls == 2
    assert prop._mk_step0_loop_keys == {(64, 16)}
    warned = [
        r for r in caplog.records
        if "not compiled during warmup" in r.getMessage()
    ]
    assert len(warned) == 1 and "(16, 16)" in warned[0].getMessage()
    out, _ = _propose(prop, _step(prop, 16, 16, True), True)
    _same(out, refs[16])
    assert prop._mk_step0_loop_program.calls == 3


@pytest.mark.parametrize("mk_loop,lockstep,k", [(True, False, 3), (False, True, 3),
                                                (False, False, 1)])
def test_not_wanted_keeps_two_programs(env, mk_loop, lockstep, k):
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", True, raising=False)
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_LOOP", mk_loop, raising=False)
    prop = _proposer(16)
    prop.runner._dp_lockstep_enabled = lambda: lockstep
    assert not prop._mk_step0_loop_program_wanted(k)
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_LOOP", False, raising=False)
    prop.runner._dp_lockstep_enabled = lambda: False
    assert prop._mk_step0_loop_program_wanted(3)
    env.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", False, raising=False)
    assert not prop._mk_step0_loop_program_wanted(3)


def test_refuses_without_sorted_gather(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", True,
                        raising=False)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", False, raising=False)
    with pytest.raises(ValueError, match="MOE_FUSED_EP_SORTED_GATHER"):
        eagle3._check_mk_step0_loop_program_prereq()
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", True, raising=False)
    eagle3._check_mk_step0_loop_program_prereq()
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", False,
                        raising=False)
    monkeypatch.setattr(envs, "MOE_FUSED_EP_SORTED_GATHER", False, raising=False)
    eagle3._check_mk_step0_loop_program_prereq()


def test_program_module_is_step0_then_loop(env):
    """The program's forward = `_MkDraftStep0` (step 0's own starts / seq_lens) then
    `_MtpFusedLoop` (the context's seq_lens, bumped in place) on its outputs."""
    mtp = _StubMtp()
    cap, n, T = 8, 5, 32
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(0, VOCAB, (T,), generator=g, dtype=torch.int32)
    pos = torch.randint(10, 999, (T,), generator=g, dtype=torch.int32)
    hid = torch.randn(T, HIDDEN, generator=g)
    rows = torch.full((cap,), -1, dtype=torch.int32)
    rows[:n] = torch.arange(n, dtype=torch.int32) * 4 + 3
    qsl0 = torch.from_numpy(np.minimum(np.arange(cap + 1, dtype=np.int32) * 4, 4 * n))
    seq0 = torch.arange(cap, dtype=torch.int32) * 11 + 40
    loop_seq = seq0 + 1
    step0_md = {"attn": SimpleNamespace(query_start_loc=qsl0, seq_lens=seq0)}
    loop_qsl = torch.from_numpy(np.minimum(np.arange(cap + 1, dtype=np.int32), n))

    def loop_md(seq):
        return {"attn": SimpleNamespace(query_start_loc=loop_qsl, seq_lens=seq)}

    program = _module(fl._MkDraftStep0LoopProgram, mtp, num_steps=K - 1,
                      rows_per_request=4)
    seq_a = loop_seq.clone()
    _CTX.append(loop_md(seq_a))
    try:
        got = program.forward(ids, pos, hid, rows, qsl0, seq0, seq_a)
    finally:
        _CTX.pop()
    got_trace = list(mtp.trace)
    mtp.trace.clear()

    env.setattr(envs, "VLLM_TPU_MK_DRAFT_ROWS_FRESH", False, raising=False)
    step0 = _module(fl._MkDraftStep0, mtp, rows_per_request=4)
    loop = _module(fl._MtpFusedLoop, mtp, num_steps=K - 1, use_mk=False)
    _CTX.append(step0_md)
    try:
        carry, tok0, pos0 = step0.forward(ids, pos, hid, rows)
    finally:
        _CTX.pop()
    seq_b = loop_seq.clone()
    _CTX.append(loop_md(seq_b))
    try:
        want = (tok0, *loop.forward(tok0, pos0 + 1, carry, seq_b))
    finally:
        _CTX.pop()
    assert len(got) == K
    for a, b in zip(got, want):
        _same(a, b)
    _same(seq_a, seq_b)
    _same_loop_trace(mtp.trace, got_trace, cap)


def _tail_runner(cap: int):
    runner = object.__new__(TPUModelRunner)
    runner.device = "cpu"
    runner.max_num_reqs = cap
    runner.num_reqs_max_model_len = cap
    runner.num_reqs_most_model_len = cap
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=K)
    runner.drafter = SimpleNamespace(bucketed_drafts=False, _loop_bucket=_bucket)
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner._spec_tail_stack_keys = set()
    runner._precompile_timed = lambda *_: contextlib.nullcontext()
    return runner


@pytest.mark.parametrize("flag", [False, True])
def test_spec_tail_warms_capacity_steps(monkeypatch, flag):
    """With the flag, warmup also compiles the spec tail for K [capacity] steps at
    every request count, so ON's drafts never stack eagerly; OFF warms nothing new."""
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_COMPILED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_DENSE_STEP0", False, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", flag,
                        raising=False)
    for name in ("_spec_tail_compiled", "_spec_tail_stack_compiled",
                 "_spec_tail_stack_dense0_compiled"):
        monkeypatch.setattr(runner_mod, name, lambda *a: (a[0], a[1]))
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *a, **k: None)
    cap = 16
    runner = _tail_runner(cap)
    runner._precompile_spec_tail()
    widths = {}
    for key in runner._spec_tail_stack_keys:
        widths.setdefault(key[4], set()).add(key[5][0][0])
    for n in range(1, cap + 1):
        want = {(_bucket(n),)} | ({(cap,)} if flag else set())
        assert widths[n] == want, (n, widths[n])
