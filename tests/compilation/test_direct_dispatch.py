# SPDX-License-Identifier: Apache-2.0
"""VLLM_TPU_DIRECT_DISPATCH: a region called again with inputs it has seen
calls its compiled graph directly and returns what Dynamo's path returns."""

import pytest
import torch

from vllm_torchtpu.compilation import direct_dispatch
from vllm_torchtpu.compilation.direct_dispatch import DirectDispatch


@pytest.fixture(autouse=True)
def _aot_eager_backend(monkeypatch):
    """The "tpu" backend traces TPU tensors only: compile with aot_eager,
    which keeps AOTAutograd's runtime wrapper (input mutations, aliasing)."""
    import torch._dynamo

    real = torch._dynamo.lookup_backend
    monkeypatch.setattr(
        torch._dynamo,
        "lookup_backend",
        lambda name: real("aot_eager" if name == "tpu" else name),
    )
    torch._dynamo.reset()


def _direct_only(region: DirectDispatch):
    """Makes any call that would go through Dynamo fail."""

    def fail(*a, **k):
        raise AssertionError("went through Dynamo")

    region._compiled = fail


def _spec_tail_like(next_tokens, drafts, rows: int, bound: int):
    src = torch.cat([next_tokens[:rows], drafts[:rows]], dim=1).reshape(-1)
    if src.shape[0] < bound:
        src = torch.nn.functional.pad(src, (0, bound - src.shape[0]))
    return src, (drafts[:rows] < 0).sum(dim=1)


def test_repeated_inputs_call_the_graph_directly():
    region = DirectDispatch(_spec_tail_like)
    a, b = torch.arange(8).view(4, 2), torch.arange(12).view(4, 3) - 5
    want = _spec_tail_like(a, b, 2, 16)
    first = region(a, b, 2, 16)
    _direct_only(region)
    for x, y in ((a, b), (a + 1, b * 2)):
        got = region(x, y, 2, 16)
        ref = _spec_tail_like(x, y, 2, 16)
        assert isinstance(got, tuple) and len(got) == 2
        for g, r in zip(got, ref):
            assert torch.equal(g, r)
    for g, w in zip(first, want):
        assert torch.equal(g, w)


def test_new_shapes_values_and_aliasing_go_through_dynamo():
    region = DirectDispatch(_spec_tail_like)
    a, b = torch.arange(8).view(4, 2), torch.arange(8).view(4, 2)
    region(a, b, 2, 16)
    calls = []
    compiled = region._compiled
    region._compiled = lambda *x, **k: calls.append(1) or compiled(*x, **k)
    region(a, b, 3, 16)  # an int Dynamo specializes on
    region(a[:3], b[:3], 2, 16)  # another shape
    region(a, a, 2, 16)  # the same tensor twice
    region(a.t().contiguous().t(), b, 2, 16)  # another stride
    assert len(calls) == 4
    region(a, b, 3, 16)
    assert len(calls) == 4


def _scatter(buf, idx, vals) -> None:
    buf.index_put_((idx,), vals)


def test_input_mutation_applies_on_direct_calls():
    region = DirectDispatch(_scatter)
    buf = torch.zeros(8, dtype=torch.int32)
    region(buf, torch.tensor([1, 2]), torch.tensor([5, 6], dtype=torch.int32))
    _direct_only(region)
    assert (
        region(buf, torch.tensor([3, 4]), torch.tensor([7, 8], dtype=torch.int32))
        is None
    )
    assert buf.tolist() == [0, 5, 6, 7, 8, 0, 0, 0]


def test_graph_returning_none_is_replayed(monkeypatch):
    """The tpu backend's graph of an input-only region can return None."""
    import torch._dynamo

    def none_backend(gm, example_inputs):
        def run(*args):
            gm(*args)

        return run

    monkeypatch.setattr(torch._dynamo, "lookup_backend", lambda name: none_backend)
    region = DirectDispatch(_scatter)
    buf = torch.zeros(4, dtype=torch.int32)
    region(buf, torch.tensor([0]), torch.tensor([1], dtype=torch.int32))
    _direct_only(region)
    assert region(buf, torch.tensor([3]), torch.tensor([2], dtype=torch.int32)) is None
    assert buf.tolist() == [1, 0, 0, 2]


def _fused_like(starts, groups, *, pooled: bool, substitution=None):
    out = starts * 2 if pooled else starts
    if substitution is not None:
        src, idx = substitution
        out = out + src.index_select(0, idx)
    return out + torch.stack(groups).sum(0), groups[0]


def test_lists_keywords_and_returned_inputs():
    region = DirectDispatch(_fused_like)
    groups = [torch.full((4,), i) for i in range(3)]
    sub = (torch.arange(10), torch.tensor([0, 2, 4, 6]))
    region(torch.ones(4, dtype=torch.int64), groups, pooled=True, substitution=sub)
    _direct_only(region)
    groups2 = [g + 1 for g in groups]
    starts = torch.arange(4)
    got, first = region(starts, groups2, pooled=True, substitution=sub)
    ref, _ = _fused_like(starts, groups2, pooled=True, substitution=sub)
    assert torch.equal(got, ref)
    assert first is groups2[0]


class _Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 3, bias=False)


def test_method_reads_module_weights_each_call():
    class Owner:
        def __init__(self):
            self.head = _Head()

        @DirectDispatch
        def logits(self, hidden, indices):
            return self.head.proj(hidden.index_select(0, indices))

    owner = Owner()
    hidden, idx = torch.randn(6, 4), torch.tensor([1, 3])
    with torch.no_grad():
        owner.logits(hidden, idx)
        _direct_only(Owner.logits)
        owner.head.proj.weight = torch.nn.Parameter(torch.randn(3, 4))
        got = owner.logits(hidden, idx)
        assert torch.allclose(got, owner.head.proj(hidden[idx]))


def test_reset_forgets_direct_calls():
    region = DirectDispatch(_spec_tail_like)
    a = torch.arange(8).view(4, 2)
    region(a, a + 1, 2, 16)
    assert any(e is not None for e in region._entries.values())
    direct_dispatch.reset()
    assert not region._entries


def test_flag_off_is_plain_torch_compile(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "VLLM_TPU_DIRECT_DISPATCH", True, raising=False)
    assert isinstance(direct_dispatch.tpu_compile(_scatter), DirectDispatch)
    monkeypatch.setattr(envs, "VLLM_TPU_DIRECT_DISPATCH", False, raising=False)
    monkeypatch.setattr(torch, "compile", lambda fn, **kw: (fn, kw))
    assert direct_dispatch.tpu_compile(_scatter) == (
        _scatter,
        dict(backend="tpu", fullgraph=True, dynamic=False),
    )
