# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_SPEC_TAIL_STACK: eagle3 hands its per-step draft tokens to the
spec-tail program unstacked, and the program stacks and slices them in-graph.

On CPU the compiled regions are replaced by their eager bodies, and a
dispatch mode counts the aten ops that run outside them: those are the ops
that become eager programs on TPU.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from tests.spec_decode.test_eagle3 import _make_chunk, _make_proposer
from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner as runner_mod
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.spec_decode.utils import DraftStepTokens


class _EagerOps(TorchDispatchMode):
    """Counts aten ops dispatched outside the stand-ins for compiled regions."""

    def __init__(self):
        super().__init__()
        self.ops: list[str] = []
        self.depth = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.depth == 0:
            self.ops.append(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


_MODE: list[_EagerOps] = []


def _compiled(fn):
    """`fn` as a compiled region: its ops are not eager programs."""

    def call(*args, **kwargs):
        mode = _MODE[-1] if _MODE else None
        if mode is not None:
            mode.depth += 1
        try:
            return fn(*args, **kwargs)
        finally:
            if mode is not None:
                mode.depth -= 1

    return call


def _count(fn):
    mode = _EagerOps()
    _MODE.append(mode)
    try:
        with mode:
            out = fn()
    finally:
        _MODE.pop()
    return out, mode.ops


def _bucket(n: int) -> int:
    for b in (1, 2, 4, 8, 16):
        if n <= b:
            return b
    raise AssertionError(n)


def _runner(k: int, max_reqs: int = 16):
    runner = object.__new__(TPUModelRunner)
    runner.device = "cpu"
    runner.max_num_reqs = max_reqs
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=k)
    runner.drafter = SimpleNamespace(bucketed_drafts=False, _loop_bucket=_bucket)
    runner.scheduler_config = SimpleNamespace(async_scheduling=True)
    runner._spec_tail_stack_keys = set()
    return runner


def _step(k: int, num_reqs: int, seed: int):
    """One steady step's sampler output, metadata and drafter tokens."""
    g = torch.Generator().manual_seed(seed)
    rows = _bucket(num_reqs)
    p = _bucket(num_reqs)
    next_tokens = torch.randint(0, 1000, (rows, k + 1), generator=g, dtype=torch.int32)
    # Random accepted prefix per row; INVALID (-1) after it.
    keep = torch.randint(1, k + 2, (rows,), generator=g)
    next_tokens[torch.arange(k + 1)[None, :] >= keep[:, None]] = -1
    lengths = torch.full((rows,), k, dtype=torch.int32)
    lengths[num_reqs:] = 0
    state = SimpleNamespace(
        num_reqs_list=[num_reqs],
        spec_decode_metadata_list=[
            SimpleNamespace(draft_lengths=lengths, draft_lengths_cpu=lengths.numpy())
        ],
    )
    steps = [
        torch.randint(0, 1000, (p,), generator=g, dtype=torch.int32) for _ in range(k)
    ]
    return next_tokens, state, steps


@pytest.fixture
def tail_env(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_COMPILED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", True, raising=False)
    monkeypatch.setattr(
        runner_mod, "_spec_tail_compiled", _compiled(runner_mod._spec_tail)
    )
    monkeypatch.setattr(
        runner_mod, "_spec_tail_stack_compiled", _compiled(runner_mod._spec_tail_stack)
    )
    monkeypatch.setattr(runner_mod, "synchronize_tensors", lambda *a, **k: None)


def _precompile(runner):
    runner._precompile_timed = lambda *_: __import__("contextlib").nullcontext()
    runner._precompile_spec_tail()


def _equal(a, b):
    assert torch.equal(a[0][0], b[0][0])
    assert a[0][0].dtype == b[0][0].dtype
    assert torch.equal(a[2], b[2])
    assert a[1] == b[1] and a[3] == b[3]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_on_matches_off_across_buckets(tail_env, monkeypatch, k):
    """ON == OFF bitwise (source, rejected counts, indices, draft counts) for
    every request count, in one runner whose steps switch buckets."""
    runner = _runner(k)
    _precompile(runner)
    assert len(runner._spec_tail_stack_keys) == 16
    order = list(range(1, 17)) + [16, 3, 9, 1, 8, 5, 16, 2]
    for i, n in enumerate(order):
        nt, state, steps = _step(k, n, seed=1000 * k + i)
        dense = DraftStepTokens(steps, n).materialize()
        # Stock: the propose return's dense tensor through the stock tail.
        ref = runner._assemble_async_spec_substitution(dense, [nt], state)
        got = runner._assemble_async_spec_substitution(
            DraftStepTokens(steps, n), [nt], state
        )
        _equal(got, ref)
        # The dense drafts equal the stock propose return's ops.
        stacked = torch.stack(steps, dim=1)
        assert torch.equal(dense, torch.cat([stacked[:n]], dim=0))


def test_on_path_emits_no_eager_stack_cat(tail_env):
    """The ON path runs no stack/cat/slice outside compiled regions; the
    stock path runs the eager stack + cat (+ the per-chunk slice)."""
    k, n = 3, 16
    runner = _runner(k)
    _precompile(runner)
    nt, state, steps = _step(k, n, seed=7)

    def off():
        dense = DraftStepTokens(steps, n).materialize()
        return runner._assemble_async_spec_substitution(dense, [nt], state)

    def on():
        return runner._assemble_async_spec_substitution(
            DraftStepTokens(steps, n), [nt], state
        )

    ref, off_ops = _count(off)
    got, on_ops = _count(on)
    _equal(got, ref)
    assert "stack" in off_ops and "cat" in off_ops, off_ops
    assert "slice" in off_ops, off_ops
    assert on_ops == [], on_ops


def test_unwarmed_key_falls_back_to_dense(tail_env):
    """A step shape warmup did not compile is stacked eagerly (never a
    serve-time compile of the stack variant), with the same result."""
    k, n = 3, 5
    runner = _runner(k)
    nt, state, steps = _step(k, n, seed=3)
    called = []
    orig = runner_mod._spec_tail_stack_compiled
    runner_mod._spec_tail_stack_compiled = lambda *a: called.append(a) or orig(*a)
    try:
        dense = DraftStepTokens(steps, n).materialize()
        ref = runner._assemble_async_spec_substitution(dense, [nt], state)
        got = runner._assemble_async_spec_substitution(
            DraftStepTokens(steps, n), [nt], state
        )
    finally:
        runner_mod._spec_tail_stack_compiled = orig
    assert not called
    _equal(got, ref)


def test_warmup_off_compiles_no_stack_variant(tail_env, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", False, raising=False)
    runner = _runner(3)
    _precompile(runner)
    assert runner._spec_tail_stack_keys == set()


@pytest.mark.parametrize(
    "fence, structured, draft_tp, tp, spec, keep",
    [
        (False, False, None, 4, True, True),
        (True, False, None, 4, True, False),
        (False, True, None, 4, True, False),
        (False, False, 1, 4, True, False),
        (False, False, 1, 1, True, True),
        (False, False, None, 4, False, False),
    ],
)
def test_keep_draft_steps_only_when_tail_is_sole_reader(
    monkeypatch, fence, structured, draft_tp, tp, spec, keep
):
    import vllm.distributed

    runner = _runner(3)
    runner.speculative_config.draft_tensor_parallel_size = draft_tp
    monkeypatch.setattr(
        TPUModelRunner, "_fence_spec_kv_under_token_copy", property(lambda s: fence)
    )
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: tp
    )
    out = runner._keep_draft_steps(
        spec, object(), SimpleNamespace(has_structured_output_requests=structured)
    )
    assert out is keep


def _propose(monkeypatch, k: int, num_reqs: int, flag: bool):
    """Eagle3Proposer.propose on the async device path (draft rows, separate
    loop steps), with the compiled helpers run eagerly."""
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_COMPILED", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_STACK", flag, raising=False)
    device = torch.device("cpu")
    hidden_size, vocab = 8, 1024
    p = _bucket(num_reqs)
    proposer = _make_proposer(draft_tp=1, method="mtp")
    proposer.speculative_config.num_speculative_tokens = k
    proposer._draft_rows = True
    proposer.runner = SimpleNamespace(
        input_batch=SimpleNamespace(num_reqs=num_reqs),
        max_num_reqs=16,
        num_reqs_max_model_len=16,
        num_reqs_most_model_len=16,
        num_tokens_paddings=[1, 2, 4, 8, 16, 32, 64],
        device=device,
        uses_mrope=False,
        _dp_lockstep_enabled=lambda: False,
    )
    tokens = 4 * p
    chunk = _make_chunk(
        input_ids=torch.zeros(tokens, dtype=torch.int32, device=device),
        position_ids=torch.zeros(tokens, dtype=torch.int32, device=device),
        query_start_loc_np=np.arange(0, 4 * num_reqs + 1, 4, dtype=np.int32),
        start_index=0,
        num_reqs=num_reqs,
        hidden=hidden_size,
        device=device,
        attn_ctx=SimpleNamespace(use_max_model_len=True),
        hidden_states=torch.zeros(tokens, hidden_size, device=device),
    )
    proposer.draft_chunks = [chunk]

    class _Draft:
        def compute_logits(self, hidden):
            return torch.nn.functional.one_hot(
                hidden[:, 0].long() % vocab, vocab
            ).float()

    proposer.draft_model = _Draft()
    proposer._draft_gather_positions = lambda pos, idx: pos.index_select(0, idx)
    proposer._draft_propose_token = lambda hidden: (
        proposer.draft_model.compute_logits(hidden).argmax(-1).to(torch.int32)
    )
    last = torch.full((p,), 4 * num_reqs - 1)
    last[:num_reqs] = torch.arange(num_reqs) * 4 + 3
    positions = torch.arange(100, 100 + tokens, dtype=torch.int32, device=device)

    def _prepare(chunk, *_args, **_kwargs):
        return (
            torch.zeros(tokens, dtype=torch.int32, device=device),
            positions,
            last,
            # Async: the rejected counts stay on device.
            torch.zeros(p, dtype=torch.int32),
        )

    proposer._prepare_draft_inputs = _prepare

    def _forward_draft(*, input_ids, positions, step_idx, draft_rows=None, **_):
        n = input_ids.shape[0] if draft_rows is None else len(draft_rows)
        value = (positions[:n].float() * (step_idx + 3)) % vocab
        if draft_rows is not None:
            value = torch.arange(n, dtype=torch.float32) * 7 + 11
        hidden = value[:, None].expand(n, hidden_size).contiguous()
        return hidden, hidden

    proposer._forward_draft = _forward_draft
    return proposer.propose(
        sampled_token_ids=None,
        discard_sampled_tokens_req_indices=[],
        num_rejected_tokens_np=None,
        scheduler_output=SimpleNamespace(num_scheduled_tokens={}),
        return_device=True,
        next_tokens_per_chunk=[torch.zeros(p, k + 1, dtype=torch.int32)],
    )


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("num_reqs", [1, 3, 4, 16])
def test_propose_on_returns_unstacked_steps(monkeypatch, k, num_reqs):
    """propose's ON return materializes to the OFF return bitwise, and its
    steps are the K [p] per-step tokens, not yet stacked."""
    off, off_ops = _count(lambda: _propose(monkeypatch, k, num_reqs, flag=False))
    on, on_ops = _count(lambda: _propose(monkeypatch, k, num_reqs, flag=True))
    # The return's stack + cat are gone from propose (the tail does them).
    assert "stack" in off_ops and "stack" not in on_ops
    assert off_ops.count("cat") == on_ops.count("cat") + 1
    assert isinstance(off, torch.Tensor) and off.shape == (num_reqs, k)
    assert isinstance(on, DraftStepTokens)
    assert len(on.steps) == k and on.num_reqs == num_reqs
    assert all(t.shape == (_bucket(num_reqs),) for t in on.steps)
    assert torch.equal(on.materialize(), off)
    assert on.materialize().dtype == off.dtype


def test_stack_tail_traces_fullgraph_static():
    """The stack variant traces as one static graph (K tensors as varargs)
    and matches its eager body bitwise."""
    import torch._dynamo

    torch._dynamo.reset()
    fn = torch.compile(
        runner_mod._spec_tail_stack, backend="eager", fullgraph=True, dynamic=False
    )
    k, n = 3, 5
    nt, state, steps = _step(k, n, seed=11)
    lengths = state.spec_decode_metadata_list[0].draft_lengths
    args = (nt, lengths, n, n, 16 * (1 + k), n, *steps)
    got = fn(*args)
    ref = runner_mod._spec_tail_stack(*args)
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
