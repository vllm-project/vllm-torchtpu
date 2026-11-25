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
    # Served padding of the sampler rows (min MIN_NUM_SEQS), as warmup uses;
    # the drafter's per-step tokens use its loop bucket.
    rows = runner_mod._get_padded_num_reqs_with_upper_limit(num_reqs, 16)
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
    monkeypatch.setattr(
        runner_mod,
        "_spec_tail_stack_dense0_compiled",
        _compiled(runner_mod._spec_tail_stack_dense0),
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
        # The ON step must hit a warmed key (compiled stack), not the eager
        # fallback.
        key = runner_mod._spec_tail_stack_key(
            nt, state.spec_decode_metadata_list[0].draft_lengths, n, steps
        )
        assert key in runner._spec_tail_stack_keys, (n, key)
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


# VLLM_TPU_SPEC_TAIL_DENSE_STEP0 ------------------------------------------


def _spy(monkeypatch, name):
    """Records the args of every call to runner_mod.<name>."""
    calls = []
    orig = getattr(runner_mod, name)

    def call(*args):
        calls.append(args)
        return orig(*args)

    monkeypatch.setattr(runner_mod, name, call)
    return calls


def _views(args):
    return [a for a in args if isinstance(a, torch.Tensor) and a._base is not None]


def _served_step(k: int, n: int, seed: int):
    """`_step` with the sampler's rows padded as serving pads them
    (`_get_padded_num_reqs_with_upper_limit`), the shapes warmup compiles."""
    nt, state, steps = _step(k, n, seed)
    rows = runner_mod._get_padded_num_reqs_with_upper_limit(n, 16)
    pad = rows - nt.shape[0]
    if pad > 0:
        nt = torch.cat([nt, torch.full((pad, k + 1), -1, dtype=torch.int32)])
        lengths = state.spec_decode_metadata_list[0].draft_lengths
        lengths = torch.cat([lengths, torch.zeros(pad, dtype=torch.int32)])
        state.spec_decode_metadata_list[0] = SimpleNamespace(
            draft_lengths=lengths, draft_lengths_cpu=lengths.numpy()
        )
    return nt, state, steps


def _mk_steps(k: int, n: int, cap: int, seed: int):
    """MK-draft-shaped steps: step 0 is `_mk_step0_trim`'s [:p] view of a [cap]
    token (garbage past p), steps 1.. are dense [p] loop outputs."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    nt, state, steps = _served_step(k, n, seed)
    p = steps[0].shape[0]
    g = torch.Generator().manual_seed(seed + 1)
    token = torch.randint(0, 1000, (cap,), generator=g, dtype=torch.int32)
    token[:p] = steps[0]
    carry = torch.randn(cap, 4, generator=g)
    pos = torch.arange(cap, dtype=torch.int32)
    _, step0, _, _ = Eagle3Proposer._mk_step0_trim((carry, token, pos), p)
    return nt, state, [step0, *steps[1:]], token


@pytest.fixture
def dense_env(tail_env, monkeypatch):
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_DENSE_STEP0", True, raising=False)


def _mk_runner(k, caps=(16, 16)):
    runner = _runner(k)
    runner.num_reqs_max_model_len, runner.num_reqs_most_model_len = caps
    return runner


@pytest.mark.parametrize("fresh", [False, True])
def test_mk_step0_token_is_a_prefix_view(fresh):
    """The premise: at p < capacity MK draft step 0's token reaches the tail as
    a [:p] view (offset 0) of its [capacity] output, with or without
    MK_DRAFT_ROWS_FRESH's 4th output."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    cap = 16
    token = torch.arange(cap, dtype=torch.int32)
    out = (torch.zeros(cap, 4), token, torch.zeros(cap, dtype=torch.int32))
    if fresh:
        out = out + (torch.zeros(cap, dtype=torch.int32),)
    for p in (4, 8):
        step0 = Eagle3Proposer._mk_step0_trim(out, p)[1]
        assert step0._base is token
        assert step0.storage_offset() == 0 and step0.shape == (p,)
        assert runner_mod._dense_step0_base(step0) is token


def test_stack_path_hands_the_step0_view_to_the_program(tail_env, monkeypatch):
    """Flag off (SPEC_TAIL_STACK alone): the MK step-0 view crosses the
    compiled-call boundary -- the param -> copy program on TPU."""
    k, n = 3, 6
    runner = _mk_runner(k)
    _precompile(runner)
    calls = _spy(monkeypatch, "_spec_tail_stack_compiled")
    nt, state, steps, token = _mk_steps(k, n, 16, seed=5)
    runner._assemble_async_spec_substitution(DraftStepTokens(steps, n), [nt], state)
    assert len(calls) == 1
    assert [v._base for v in _views(calls[0])] == [token]


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("caps", [(16, 16), (16, 8)])
def test_dense_step0_on_matches_off(dense_env, monkeypatch, k, caps):
    """ON == OFF == stock bitwise over every request count and each warmed
    capacity >= p (MK draft: cap 16, p in {4, 8}), and ON passes no view."""
    on_runner = _mk_runner(k, caps)
    _precompile(on_runner)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_DENSE_STEP0", False, raising=False)
    off_runner = _mk_runner(k, caps)
    _precompile(off_runner)
    dense_calls = _spy(monkeypatch, "_spec_tail_stack_dense0_compiled")
    stack_calls = _spy(monkeypatch, "_spec_tail_stack_compiled")
    order = list(range(1, 17)) + [8, 4, 3, 16, 5, 1, 7]
    checked = 0
    for i, n in enumerate(order):
        for cap in sorted(set(caps)):
            if cap < _bucket(n):
                continue
            nt, state, steps, _ = _mk_steps(k, n, cap, seed=100 * k + 10 * i + cap)
            dense = DraftStepTokens(steps, n).materialize()
            ref = off_runner._assemble_async_spec_substitution(dense, [nt], state)
            monkeypatch.setattr(
                envs, "VLLM_TPU_SPEC_TAIL_DENSE_STEP0", False, raising=False
            )
            off = off_runner._assemble_async_spec_substitution(
                DraftStepTokens(steps, n), [nt], state
            )
            monkeypatch.setattr(
                envs, "VLLM_TPU_SPEC_TAIL_DENSE_STEP0", True, raising=False
            )
            del dense_calls[:], stack_calls[:]
            on = on_runner._assemble_async_spec_substitution(
                DraftStepTokens(steps, n), [nt], state
            )
            assert len(dense_calls) == 1 and not stack_calls, (n, cap)
            assert _views(dense_calls[0]) == []
            _equal(on, off)
            _equal(on, ref)
            checked += 1
    assert checked >= len(order)


def test_dense_step0_fallbacks_match_off(dense_env, monkeypatch):
    """Step 0 dense, an offset view, a strided view or a view of a 2-D tensor:
    the stack variant runs as before, with the stock result."""
    k, n = 3, 6
    runner = _mk_runner(k)
    _precompile(runner)
    dense_calls = _spy(monkeypatch, "_spec_tail_stack_dense0_compiled")
    stack_calls = _spy(monkeypatch, "_spec_tail_stack_compiled")
    nt, state, steps = _served_step(k, n, seed=9)
    p = steps[0].shape[0]
    shifted = torch.zeros(16, dtype=torch.int32)
    shifted[1 : p + 1] = steps[0]
    strided = torch.zeros(2 * p, dtype=torch.int32)
    strided[::2] = steps[0]
    wide = torch.zeros(16, 2, dtype=torch.int32)
    wide[:p, 0] = steps[0]
    for step0 in (steps[0], shifted[1 : p + 1], strided[::2], wide[:p, 0]):
        assert torch.equal(step0, steps[0])
        assert runner_mod._dense_step0_base(step0) is None
        cur = [step0, *steps[1:]]
        ref = runner._assemble_async_spec_substitution(
            DraftStepTokens(cur, n).materialize(), [nt], state
        )
        del stack_calls[:]
        got = runner._assemble_async_spec_substitution(
            DraftStepTokens(cur, n), [nt], state
        )
        assert len(stack_calls) == 1
        _equal(got, ref)
    assert not dense_calls


def test_dense_step0_unwarmed_cap_falls_back(dense_env, monkeypatch):
    """A base length warmup did not compile keeps the (warmed) stack variant:
    no serve-time compile of either program."""
    k, n = 3, 6
    runner = _mk_runner(k, caps=(16, 16))
    _precompile(runner)
    dense_calls = _spy(monkeypatch, "_spec_tail_stack_dense0_compiled")
    stack_calls = _spy(monkeypatch, "_spec_tail_stack_compiled")
    nt, state, steps, _ = _mk_steps(k, n, 32, seed=13)
    ref = runner._assemble_async_spec_substitution(
        DraftStepTokens(steps, n).materialize(), [nt], state
    )
    got = runner._assemble_async_spec_substitution(
        DraftStepTokens(steps, n), [nt], state
    )
    assert not dense_calls and len(stack_calls) == 1
    _equal(got, ref)


@pytest.mark.parametrize("caps", [(16, 16), (16, 8), (8, 16)])
def test_dense_step0_warmup_covers_serving_keys(dense_env, caps):
    """Every key serving can build -- the stock dense steps at each request
    count, and the MK step-0 view at each capacity >= p -- is warmed."""
    k = 3
    runner = _mk_runner(k, caps)
    _precompile(runner)
    keys = runner._spec_tail_stack_keys
    want = set()
    for n in range(1, 17):
        nt, state, steps = _served_step(k, n, seed=n)
        lengths = state.spec_decode_metadata_list[0].draft_lengths
        want.add(runner_mod._spec_tail_stack_key(nt, lengths, n, steps))
        for cap in set(caps):
            if cap < _bucket(n):
                continue
            _, _, mk_steps, token = _mk_steps(k, n, cap, seed=n)
            base = runner_mod._dense_step0_base(mk_steps[0])
            assert base is token
            want.add(
                runner_mod._spec_tail_stack_dense0_key(nt, lengths, n, mk_steps, base)
            )
    assert want <= keys
    assert keys == want  # nothing warmed that serving cannot reach


def test_dense_step0_off_warms_no_dense_keys(tail_env):
    runner = _mk_runner(3)
    _precompile(runner)
    assert len(runner._spec_tail_stack_keys) == 16
    assert all(len(key) == 6 for key in runner._spec_tail_stack_keys)


def test_dense_step0_wrong_slice_is_caught(dense_env, monkeypatch):
    """Negative control: a variant slicing the base one row late differs from
    OFF, so the equality checks above can fail."""
    k, n = 3, 6
    runner = _mk_runner(k)
    _precompile(runner)

    def wrong(nt, dl, rows, num_reqs, bound, n_, p, base, *rest):
        return runner_mod._spec_tail_stack(
            nt, dl, rows, num_reqs, bound, n_, base[1 : p + 1], *rest
        )

    nt, state, steps, _ = _mk_steps(k, n, 16, seed=21)
    ref = runner._assemble_async_spec_substitution(
        DraftStepTokens(steps, n).materialize(), [nt], state
    )
    monkeypatch.setattr(runner_mod, "_spec_tail_stack_dense0_compiled", wrong)
    got = runner._assemble_async_spec_substitution(
        DraftStepTokens(steps, n), [nt], state
    )
    with pytest.raises(AssertionError):
        _equal(got, ref)


@pytest.mark.parametrize("k", [1, 3])
def test_dense0_tail_traces_fullgraph_static(k):
    """The dense-step-0 variant traces as one static graph and matches the
    stack variant on the view bitwise."""
    import torch._dynamo

    torch._dynamo.reset()
    fn = torch.compile(
        runner_mod._spec_tail_stack_dense0,
        backend="eager",
        fullgraph=True,
        dynamic=False,
    )
    n = 5
    nt, state, steps, token = _mk_steps(k, n, 16, seed=31)
    lengths = state.spec_decode_metadata_list[0].draft_lengths
    p = steps[0].shape[0]
    got = fn(nt, lengths, n, n, 16 * (1 + k), n, p, token, *steps[1:])
    ref = runner_mod._spec_tail_stack(nt, lengths, n, n, 16 * (1 + k), n, *steps)
    assert all(torch.equal(a, b) for a, b in zip(got, ref))


# --- the drafter-recorded step-0 base and the "not used" self-check ----------


def _not_used_lines(log):
    return [
        c.args[0] % c.args[1:]
        for c in log.info.call_args_list
        if c.args and "not used" in c.args[0]
    ]


@pytest.fixture
def dense_log(dense_env, monkeypatch):
    from unittest.mock import Mock

    log = Mock()
    monkeypatch.setattr(runner_mod, "logger", log)
    monkeypatch.setattr(runner_mod, "_prep_fold_rank0", lambda: True)
    return log


@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("n", [1, 3, 6, 8, 16])
def test_dense_step0_uses_the_drafter_base_without_view_tracking(
    dense_log, monkeypatch, k, n
):
    """Step 0 with no `_base` (a backend that does not track the [:p] view, or
    a copy): the base the drafter recorded (`DraftStepTokens.step0_base`, MK
    step 0's [capacity] token) still takes the dense variant, bitwise OFF."""
    runner = _mk_runner(k)
    _precompile(runner)
    dense_calls = _spy(monkeypatch, "_spec_tail_stack_dense0_compiled")
    nt, state, steps, token = _mk_steps(k, n, 16, seed=31 * k + n)
    untracked = [steps[0].clone(), *steps[1:]]
    assert untracked[0]._base is None
    assert runner_mod._dense_step0_base(untracked[0]) is None
    ref = runner._assemble_async_spec_substitution(
        DraftStepTokens(steps, n).materialize(), [nt], state
    )
    got = runner._assemble_async_spec_substitution(
        DraftStepTokens(untracked, n, step0_base=token, step0_source="mk"), [nt], state
    )
    assert len(dense_calls) == 1 and dense_calls[0][7] is token
    _equal(got, ref)
    assert _not_used_lines(dense_log) == [
        "SPEC_TAIL_DENSE_STEP0: not used (the drafter returned stacked drafts "
        "(not a one-chunk async SPEC_TAIL_STACK step))"
    ]  # only the reference call's (stacked drafts)
    passed = [
        c.args[0] % c.args[1:]
        for c in dense_log.info.call_args_list
        if "step 0 passed as its" in c.args[0]
    ]
    assert passed == [
        f"SPEC_TAIL_DENSE_STEP0: step 0 passed as its [16] base, sliced to "
        f"{steps[0].shape[0]} rows in-graph (base: recorded by the drafter; the "
        "view's own _base: step 0 is not a view (no _base); base is not a view)"
    ]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("stock", "step 0 is the stock draft step's token (no MK step 0); "
                  "step 0 is not a view (no _base)"),
        ("mk_fresh", "step 0 is MK draft step 0's own [capacity] output, no [:p] "
                     "slice (MK_DRAFT_ROWS_FRESH at p == capacity); step 0 is not a "
                     "view (no _base)"),
        ("mk_step0_loop", "step 0 comes from VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM's "
                          "one program, [capacity] long, no [:p] slice; step 0 is not "
                          "a view (no _base)"),
        ("offset_view", "step 0 does not start at its base's first element"),
        ("bad_base", "the drafter's step-0 base (16,) torch.int64 does not hold "
                     "step 0 (8,) torch.int32"),
    ],
)
def test_dense_step0_not_used_reasons(dense_log, monkeypatch, case, reason):
    """Each way step 0 has no usable base logs its reason once (rank 0), and the
    step takes the stack variant with the stock result."""
    k, n = 3, 6
    monkeypatch.setattr(
        envs, "VLLM_TPU_MK_DRAFT_STEP0_LOOP_PROGRAM", case == "mk_step0_loop",
        raising=False,
    )
    runner = _mk_runner(k)
    _precompile(runner)
    dense_calls = _spy(monkeypatch, "_spec_tail_stack_dense0_compiled")
    stack_calls = _spy(monkeypatch, "_spec_tail_stack_compiled")
    nt, state, steps = _served_step(k, n, seed=41)
    p = steps[0].shape[0]
    if case == "mk_step0_loop":
        # That program's K tokens are all [capacity] (warmed as their own key).
        g = torch.Generator().manual_seed(5)
        steps = [torch.randint(0, 1000, (16,), generator=g, dtype=torch.int32)
                 for _ in range(k)]
        drafts = DraftStepTokens(steps, n, step0_source="mk_step0_loop")
    elif case == "offset_view":
        shifted = torch.zeros(16, dtype=torch.int32)
        shifted[1 : p + 1] = steps[0]
        steps = [shifted[1 : p + 1], *steps[1:]]
        drafts = DraftStepTokens(steps, n, step0_source="mk")
    elif case == "bad_base":
        drafts = DraftStepTokens(
            steps, n, step0_base=torch.zeros(16, dtype=torch.int64), step0_source="mk"
        )
    else:
        drafts = DraftStepTokens(steps, n, step0_source=case.split("_")[0])
    ref = runner._assemble_async_spec_substitution(
        DraftStepTokens(steps, n).materialize(), [nt], state
    )
    del stack_calls[:]
    dense_log.reset_mock()
    for _ in range(2):
        got = runner._assemble_async_spec_substitution(drafts, [nt], state)
        _equal(got, ref)
    assert not dense_calls and len(stack_calls) == 2
    assert _not_used_lines(dense_log) == [f"SPEC_TAIL_DENSE_STEP0: not used ({reason})"]


@pytest.mark.parametrize(
    "case,reason",
    [
        ("compiled_off", "VLLM_TPU_SPEC_TAIL_COMPILED is off"),
        ("two_chunks", "a step of more than one chunk"),
        ("no_metadata", "a step without spec decode metadata"),
        ("count", "a step whose chunk request count differs from the drafts'"),
        ("bucketed", "a bucketed-draft drafter"),
    ],
)
def test_spec_tail_early_return_reasons(dense_log, monkeypatch, case, reason):
    """`_spec_tail_from_steps` returning early names its condition once."""
    k, n = 3, 6
    runner = _mk_runner(k)
    _precompile(runner)
    nt, state, steps, token = _mk_steps(k, n, 16, seed=43)
    chunks = [nt]
    if case == "compiled_off":
        monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_COMPILED", False, raising=False)
    elif case == "two_chunks":
        chunks = [nt, nt]
    elif case == "no_metadata":
        state.spec_decode_metadata_list[0] = None
    elif case == "count":
        state.num_reqs_list[0] = n - 1
    else:
        runner.drafter.bucketed_drafts = True
    drafts = DraftStepTokens(steps, n, step0_base=token, step0_source="mk")
    for _ in range(2):
        assert runner._spec_tail_from_steps(drafts, chunks, state) is None
    assert _not_used_lines(dense_log) == [
        f"SPEC_TAIL_DENSE_STEP0: not used ({reason}: the stack path is not taken)"
    ]


@pytest.mark.parametrize(
    "fence,structured,draft_tp,tp,spec,reason",
    [
        (False, False, 8, 8, False, "not a spec decode step"),
        (True, False, 8, 8, True, "the KV producer fence reads them"),
        (False, True, 8, 8, True, "structured-output requests read them on the host"),
        (False, False, 1, 8, True, "the replicated drafter's TP pin reads them"),
        (False, False, 8, 8, True, None),
        (False, False, 1, 1, True, None),
    ],
)
def test_draft_steps_stack_reason(monkeypatch, fence, structured, draft_tp, tp, spec,
                                  reason):
    import vllm.distributed

    runner = _runner(3)
    runner.speculative_config.draft_tensor_parallel_size = draft_tp
    monkeypatch.setattr(
        TPUModelRunner, "_fence_spec_kv_under_token_copy", property(lambda s: fence)
    )
    monkeypatch.setattr(
        vllm.distributed, "get_tensor_model_parallel_world_size", lambda: tp
    )
    sched = SimpleNamespace(has_structured_output_requests=structured)
    assert runner._draft_steps_stack_reason(spec, object(), sched) == reason
    assert runner._keep_draft_steps(spec, object(), sched) is (reason is None)
    runner.scheduler_config.async_scheduling = False
    assert runner._draft_steps_stack_reason(spec, object(), sched) == (
        "not a spec decode step" if not spec else "no async scheduling"
    )


def test_dense_step0_not_used_rank0_once_and_flag_gated(dense_log, monkeypatch):
    runner = _mk_runner(3)
    for _ in range(3):
        runner._spec_tail_dense0_not_used("r1")
    runner._spec_tail_dense0_not_used("r2")
    assert _not_used_lines(dense_log) == [
        "SPEC_TAIL_DENSE_STEP0: not used (r1)",
        "SPEC_TAIL_DENSE_STEP0: not used (r2)",
    ]
    dense_log.reset_mock()
    monkeypatch.setattr(runner_mod, "_prep_fold_rank0", lambda: False)
    runner._spec_tail_dense0_not_used("r3")  # other TP ranks: silent
    monkeypatch.setattr(runner_mod, "_prep_fold_rank0", lambda: True)
    monkeypatch.setattr(envs, "VLLM_TPU_SPEC_TAIL_DENSE_STEP0", False, raising=False)
    runner._spec_tail_dense0_not_used("r4")  # flag off: silent
    assert _not_used_lines(dense_log) == []


def test_propose_records_the_mk_step0_base(monkeypatch):
    """With MK draft step 0, propose's DraftStepTokens carries the [capacity]
    token step 0 was sliced from (`step0_base`, source "mk"); step 0 is exactly
    its leading [:p] rows; the drafts materialize as before."""
    from vllm_torchtpu.spec_decode.eagle3 import Eagle3Proposer

    k, num_reqs, cap = 3, 3, 16
    p = _bucket(num_reqs)
    seen = {}
    orig_propose = Eagle3Proposer.propose

    def run(mk: bool):
        def propose(self, *a, **kw):
            if mk:
                self._mk_draft_step0_ready = lambda *_a, **_k: True
                self._mk_step0_loop_program_wanted = lambda *_a: False

                def step0(chunk, input_ids, positions, ths, last, rej):
                    g = torch.Generator().manual_seed(3)
                    out = (
                        torch.randn(cap, 8, generator=g),
                        torch.randint(0, 1000, (cap,), generator=g, dtype=torch.int32),
                        torch.arange(100, 100 + cap, dtype=torch.int32),
                    )
                    seen["token"] = out[1]
                    return self._mk_step0_trim_recorded(out, last.shape[0])

                self._run_mk_draft_step0 = step0
            return orig_propose(self, *a, **kw)

        monkeypatch.setattr(Eagle3Proposer, "propose", propose)
        return _propose(monkeypatch, k, num_reqs, flag=True)

    stock = run(False)
    assert stock.step0_source == "stock" and stock.step0_base is None
    on = run(True)
    assert isinstance(on, DraftStepTokens) and on.step0_source == "mk"
    assert on.step0_base is seen["token"] and on.step0_base.shape == (cap,)
    assert on.steps[0].shape == (p,)
    assert torch.equal(on.steps[0], on.step0_base[:p])
    assert torch.equal(on.materialize()[:, 0], seen["token"][:num_reqs])
