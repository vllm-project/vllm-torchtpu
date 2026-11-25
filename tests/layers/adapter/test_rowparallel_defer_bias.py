# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Equivalence test for the TPU RowParallelLinear bias deferral.

Pins both halves of the claim: vLLM instantiates our out-of-tree class, and its
forward returns what upstream's returns on every rank.
"""

from types import SimpleNamespace

import pytest
import torch
import vllm.model_executor.layers.linear as linear_mod
from vllm.model_executor.custom_op import maybe_get_oot_by_class, op_registry_oot
from vllm.model_executor.layers.linear import RowParallelLinear

from tests.layers.adapter.rowparallel_test_utils import (
    as_pair,
    assert_same,
    run_tp_group,
)
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear


def _make_layer(
    tp_rank,
    tp_size,
    bias,
    weight_block,
    skip_bias_add,
    reduce_results=True,
    return_bias=False,
    input_is_parallel=True,
):
    """A RowParallelLinear stand-in: ``__new__`` without ``__init__``, which
    would need an initialised distributed group."""

    def apply(layer, inp, bias_):
        out = inp @ weight_block.T
        if bias_ is not None:
            out = out + bias_
        return out

    layer = TpuRowParallelLinear.__new__(TpuRowParallelLinear)
    layer.input_is_parallel = input_is_parallel
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.reduce_results = reduce_results
    layer.skip_bias_add = skip_bias_add
    layer.bias = bias
    layer.return_bias = return_bias
    layer.quant_method = SimpleNamespace(apply=apply)
    return layer


def _run_tp_group(monkeypatch, forward, w_blocks, x_blocks, **layer_kwargs):
    """Runs ``forward`` on every rank of a simulated TP group."""
    tp_size = len(w_blocks)
    return run_tp_group(
        monkeypatch,
        linear_mod,
        tp_size,
        lambda rank: forward(
            _make_layer(rank, tp_size, weight_block=w_blocks[rank], **layer_kwargs),
            x_blocks[rank],
        ),
    )


def test_registered_under_the_upstream_class_name():
    """Importing the module is what makes vLLM build our class."""
    assert op_registry_oot["RowParallelLinear"] is TpuRowParallelLinear
    # The lookup LoRA uses.
    assert maybe_get_oot_by_class(RowParallelLinear) is TpuRowParallelLinear
    # The lookup every RowParallelLinear() call site goes through.
    assert isinstance(
        RowParallelLinear.__new__(RowParallelLinear), TpuRowParallelLinear
    )
    # Guard against a vacuous equivalence check below.
    assert TpuRowParallelLinear.forward is not RowParallelLinear.forward


@pytest.mark.parametrize("bias_present", [True, False])
@pytest.mark.parametrize("skip_bias_add", [False, True])
@pytest.mark.parametrize("return_bias", [False, True])
@pytest.mark.parametrize("reduce_results", [True, False])
@pytest.mark.parametrize("input_is_parallel", [True, False])
def test_matches_upstream_tp2(
    monkeypatch,
    bias_present,
    skip_bias_add,
    return_bias,
    reduce_results,
    input_is_parallel,
):
    """Our forward == upstream's, rank by rank, on a simulated TP=2 group."""
    rng = torch.Generator().manual_seed(0)
    out_f, in_f, tokens = 6, 8, 3
    W = torch.randn(out_f, in_f, generator=rng)
    x = torch.randn(tokens, in_f, generator=rng)
    b = torch.randn(out_f, generator=rng) if bias_present else None

    # RowParallelLinear shards the INPUT dim: W = [W0 | W1] column blocks and
    # x = [x0 | x1], one block of each per rank.
    half = in_f // 2
    w_blocks = [W[:, :half], W[:, half:]]
    x_blocks = [x[:, :half], x[:, half:]]
    if not input_is_parallel:
        # Every rank gets the whole input and takes its own block.
        x_blocks = [x, x]
    kwargs = dict(
        bias=b,
        skip_bias_add=skip_bias_add,
        return_bias=return_bias,
        reduce_results=reduce_results,
        input_is_parallel=input_is_parallel,
    )

    ours = _run_tp_group(
        monkeypatch, TpuRowParallelLinear.forward, w_blocks, x_blocks, **kwargs
    )
    upstream = _run_tp_group(
        monkeypatch, RowParallelLinear.forward, w_blocks, x_blocks, **kwargs
    )
    for rank, (mine, theirs) in enumerate(zip(ours, upstream)):
        assert_same(mine, theirs, f"rank {rank}")

    if not reduce_results:
        return  # no collective: each rank keeps its own partial product

    # Every rank also agrees with what one unsharded layer would produce.
    expected = x @ W.T
    if b is not None and not skip_bias_add:
        expected = expected + b
    for rank, mine in enumerate(ours):
        torch.testing.assert_close(as_pair(mine)[0], expected, msg=f"rank {rank}")


def test_delegates_to_upstream_at_tp1(monkeypatch):
    """At TP=1 there is no all-reduce, so nothing is deferred."""
    rng = torch.Generator().manual_seed(0)
    out_f, in_f, tokens = 5, 4, 2
    W = torch.randn(out_f, in_f, generator=rng)
    x = torch.randn(tokens, in_f, generator=rng)
    b = torch.randn(out_f, generator=rng)

    kwargs = dict(bias=b, skip_bias_add=False)
    ours = _run_tp_group(monkeypatch, TpuRowParallelLinear.forward, [W], [x], **kwargs)
    upstream = _run_tp_group(monkeypatch, RowParallelLinear.forward, [W], [x], **kwargs)

    assert_same(ours[0], upstream[0], "tp1")
    torch.testing.assert_close(as_pair(ours[0])[0], x @ W.T + b)
