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

import vllm_torchtpu.layers.adapter.linear as tpu_linear_mod
from vllm_torchtpu.layers.adapter.linear import TpuRowParallelLinear


def _make_layer(
    tp_rank,
    tp_size,
    bias,
    weight_block,
    skip_bias_add,
    reduce_results=True,
    return_bias=False,
):
    """A RowParallelLinear stand-in: ``__new__`` without ``__init__``, which
    would need an initialised distributed group."""

    def apply(layer, inp, bias_):
        out = inp @ weight_block.T
        if bias_ is not None:
            out = out + bias_
        return out

    layer = TpuRowParallelLinear.__new__(TpuRowParallelLinear)
    layer.input_is_parallel = True
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.reduce_results = reduce_results
    layer.skip_bias_add = skip_bias_add
    layer.bias = bias
    layer.return_bias = return_bias
    layer.quant_method = SimpleNamespace(apply=apply)
    return layer


def _run_tp_group(monkeypatch, forward, w_blocks, x_blocks, **layer_kwargs):
    """Run ``forward`` on every rank of a simulated TP group.

    A real all-reduce hands every rank the same sum, which takes two passes to
    imitate: capture each rank's pre-reduce output, then replay every rank with
    the all-reduce returning their sum.
    """
    tp_size = len(w_blocks)

    def run_all_ranks(all_reduce):
        # Upstream reads all_reduce from vLLM's module globals, ours from ours.
        monkeypatch.setattr(linear_mod, "tensor_model_parallel_all_reduce", all_reduce)
        monkeypatch.setattr(
            tpu_linear_mod, "tensor_model_parallel_all_reduce", all_reduce
        )
        return [
            forward(
                _make_layer(r, tp_size, weight_block=w, **layer_kwargs), x_blocks[r]
            )
            for r, w in enumerate(w_blocks)
        ]

    pre_reduce = []

    def capture(tensor):
        pre_reduce.append(tensor)
        return tensor

    run_all_ranks(capture)
    reduced = sum(pre_reduce) if pre_reduce else None
    return run_all_ranks(lambda tensor: reduced)


def _as_pair(ret):
    return ret if isinstance(ret, tuple) else (ret, None)


def _assert_same(ours, upstream, msg):
    out, bias = _as_pair(ours)
    want_out, want_bias = _as_pair(upstream)
    torch.testing.assert_close(out, want_out, msg=msg)
    assert (bias is None) == (want_bias is None), msg
    if bias is not None:
        torch.testing.assert_close(bias, want_bias, msg=msg)


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
def test_matches_upstream_tp2(
    monkeypatch, bias_present, skip_bias_add, return_bias, reduce_results
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
    kwargs = dict(
        bias=b,
        skip_bias_add=skip_bias_add,
        return_bias=return_bias,
        reduce_results=reduce_results,
    )

    ours = _run_tp_group(
        monkeypatch, TpuRowParallelLinear.forward, w_blocks, x_blocks, **kwargs
    )
    upstream = _run_tp_group(
        monkeypatch, RowParallelLinear.forward, w_blocks, x_blocks, **kwargs
    )
    for rank, (mine, theirs) in enumerate(zip(ours, upstream)):
        _assert_same(mine, theirs, f"rank {rank}")

    if not reduce_results:
        return  # no collective: each rank keeps its own partial product

    # Every rank also agrees with what one unsharded layer would produce.
    expected = x @ W.T
    if b is not None and not skip_bias_add:
        expected = expected + b
    for rank, mine in enumerate(ours):
        torch.testing.assert_close(_as_pair(mine)[0], expected, msg=f"rank {rank}")


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

    _assert_same(ours[0], upstream[0], "tp1")
    torch.testing.assert_close(_as_pair(ours[0])[0], x @ W.T + b)
