# Copyright 2026 Google LLC
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
"""The TPU row-parallel LoRA layer returns what vLLM's returns on every rank,
and passes every rank the same bias, so the ranks trace the same graph."""

from types import SimpleNamespace

import pytest
import torch
import vllm.lora.layers.row_parallel_linear as vllm_row_mod
from torch import nn
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)

from tests.layers.adapter.rowparallel_test_utils import (
    as_pair,
    assert_same,
    run_tp_group,
)
from vllm_torchtpu.lora.layers import (
    TpuRowParallelLinearWithLoRA,
    use_tpu_lora_layers,
)


def _make_layer(
    cls,
    tp_rank,
    tp_size,
    bias,
    weight_block,
    lora_block,
    skip_bias_add,
    reduce_results=True,
    return_bias=False,
    seen_biases=None,
):
    """A row-parallel LoRA layer stand-in: no base model or distributed group,
    and an apply that computes the base GEMM plus a LoRA delta as dense math."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.tp_rank = tp_rank
    layer.tp_size = tp_size
    layer.base_layer = SimpleNamespace(
        input_is_parallel=True,
        reduce_results=reduce_results,
        skip_bias_add=skip_bias_add,
        bias=bias,
        return_bias=return_bias,
        tp_size=tp_size,
        tp_rank=tp_rank,
    )

    def apply(x, bias_=None):
        if seen_biases is not None:
            seen_biases.append(bias_)
        out = x @ weight_block.T + x @ lora_block.T
        if bias_ is not None:
            out = out + bias_
        return out

    layer.apply = apply
    return layer


def _run_tp_group(monkeypatch, cls, w_blocks, lora_blocks, x_blocks, **layer_kwargs):
    """Runs the forward of cls on every rank of a simulated TP group."""
    tp_size = len(w_blocks)
    return run_tp_group(
        monkeypatch,
        vllm_row_mod,
        tp_size,
        lambda rank: _make_layer(
            cls,
            rank,
            tp_size,
            weight_block=w_blocks[rank],
            lora_block=lora_blocks[rank],
            **layer_kwargs,
        )(x_blocks[rank]),
    )


def _blocks(seed=0, out_features=6, in_features=8, tokens=3):
    """Weights, LoRA delta and input of a row-parallel layer, whole and split
    along the input dim into one block per rank."""
    rng = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=rng)
    lora = torch.randn(out_features, in_features, generator=rng)
    x = torch.randn(tokens, in_features, generator=rng)
    bias = torch.randn(out_features, generator=rng)
    half = in_features // 2

    def split(t):
        return [t[:, :half], t[:, half:]]

    return weight, lora, x, bias, split(weight), split(lora), split(x)


@pytest.mark.parametrize("bias_present", [True, False])
@pytest.mark.parametrize("skip_bias_add", [False, True])
@pytest.mark.parametrize("return_bias", [False, True])
@pytest.mark.parametrize("reduce_results", [True, False])
def test_matches_vllm_tp2(
    monkeypatch, bias_present, skip_bias_add, return_bias, reduce_results
):
    weight, lora, x, bias, w_blocks, lora_blocks, x_blocks = _blocks()
    kwargs = dict(
        bias=bias if bias_present else None,
        skip_bias_add=skip_bias_add,
        return_bias=return_bias,
        reduce_results=reduce_results,
    )

    ours = _run_tp_group(
        monkeypatch,
        TpuRowParallelLinearWithLoRA,
        w_blocks,
        lora_blocks,
        x_blocks,
        **kwargs,
    )
    upstream = _run_tp_group(
        monkeypatch,
        RowParallelLinearWithLoRA,
        w_blocks,
        lora_blocks,
        x_blocks,
        **kwargs,
    )

    for rank, (mine, theirs) in enumerate(zip(ours, upstream)):
        assert_same(mine, theirs, f"rank {rank}")
    if not reduce_results:
        return  # No collective: each rank keeps its own partial product.
    # Every rank also agrees with one unsharded layer with the adapter.
    expected = x @ weight.T + x @ lora.T
    if bias_present and not skip_bias_add:
        expected = expected + bias
    for rank, mine in enumerate(ours):
        torch.testing.assert_close(as_pair(mine)[0], expected, msg=f"rank {rank}")


def test_every_rank_passes_the_same_bias_to_the_gemm(monkeypatch):
    """vLLM passes the bias on rank 0 and None on the others, which is what
    makes the ranks' graphs differ."""
    _, _, _, bias, w_blocks, lora_blocks, x_blocks = _blocks()

    def biases_seen(cls):
        seen = []
        _run_tp_group(
            monkeypatch,
            cls,
            w_blocks,
            lora_blocks,
            x_blocks,
            bias=bias,
            skip_bias_add=False,
            seen_biases=seen,
        )
        return seen

    ours = biases_seen(TpuRowParallelLinearWithLoRA)
    assert ours and all(seen is None for seen in ours)
    # Guards against a vacuous check: vLLM's forward does differ by rank.
    upstream = biases_seen(RowParallelLinearWithLoRA)
    assert any(seen is None for seen in upstream)
    assert any(seen is not None for seen in upstream)


def test_delegates_to_vllm_at_tp1(monkeypatch):
    """At TP=1 there is no all-reduce, so nothing is deferred."""
    weight, lora, x, bias, _, _, _ = _blocks()
    seen = []

    out = _run_tp_group(
        monkeypatch,
        TpuRowParallelLinearWithLoRA,
        [weight],
        [lora],
        [x],
        bias=bias,
        skip_bias_add=False,
        seen_biases=seen,
    )

    torch.testing.assert_close(out[0], x @ weight.T + x @ lora.T + bias)
    assert all(seen_bias is bias for seen_bias in seen)


def test_only_the_row_parallel_layer_is_retyped():
    def bare(cls):
        layer = cls.__new__(cls)
        nn.Module.__init__(layer)
        return layer

    row = bare(RowParallelLinearWithLoRA)
    sharded = bare(RowParallelLinearWithShardedLoRA)
    column = bare(ColumnParallelLinearWithLoRA)

    use_tpu_lora_layers([row, sharded, column])

    assert type(row) is TpuRowParallelLinearWithLoRA
    # Sharded LoRA has its own forward, and the TPU platform rejects it.
    assert type(sharded) is RowParallelLinearWithShardedLoRA
    assert type(column) is ColumnParallelLinearWithLoRA
