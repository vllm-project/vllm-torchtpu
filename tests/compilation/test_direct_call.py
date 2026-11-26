# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm_torchtpu.compilation.direct_call import DirectCompiled


def _scaled_sum(x, y, scale: int):
    return x * scale + y, (x - y).sum()


def _fail(*args):
    raise AssertionError("torch.compile entered for a warmed specialization")


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def test_warmed_specialization_skips_torch_compile():
    fn = DirectCompiled(_scaled_sum, backend="eager")
    x, y = torch.arange(6.0), torch.ones(6)
    first = fn(x, y, 3)
    compiled = fn._compiled
    fn._compiled = _fail
    for a, b in ((x, y), (torch.randn(6), torch.randn(6))):
        got = fn(a, b, 3)
        want = _scaled_sum(a, b, 3)
        assert all(torch.equal(g, w) for g, w in zip(got, want))
    assert all(torch.equal(g, w) for g, w in zip(first, _scaled_sum(x, y, 3)))
    fn._compiled = compiled
    # Another shape, scale value or dtype is another specialization.
    for args in ((torch.ones(4), torch.ones(4), 3), (x, y, 2), (x.double(), y, 3)):
        got = fn(*args)
        assert all(torch.equal(g, w) for g, w in zip(got, _scaled_sum(*args)))
    assert len(fn._direct) == 4


def test_key_separates_aliased_and_strided_inputs():
    fn = DirectCompiled(_scaled_sum, backend="eager")
    x, y = torch.arange(6.0), torch.ones(6)
    fn(x, y, 1)
    fn(x, x, 1)
    fn(torch.arange(12.0)[::2], y, 1)
    assert len(fn._direct) == 3


def test_unkeyable_arguments_go_through_torch_compile():
    fn = DirectCompiled(lambda x, opts: x * opts["scale"], backend="eager")
    out = fn(torch.ones(2), {"scale": 2.0})
    assert torch.equal(out, torch.full((2,), 2.0))
    assert not fn._direct
