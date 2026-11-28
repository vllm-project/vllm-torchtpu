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
"""Sequence-parallel forwards against upstream's TP forwards on the same shards,
on CPU: each TP rank is a thread, each collective a barrier over their inputs."""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import vllm.distributed
from torch import nn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.models import qwen2, qwen2_5_vl, qwen3, qwen3_vl
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.distributed import sequence_parallel
from vllm_torchtpu.models.vllm.sp_prefill import SequenceParallelPrefill
from vllm_torchtpu.models.vllm.sp_vision import SequenceParallelVision, _BlockShards

HIDDEN, HEAD_DIM, INTER, ROWS = 32, 8, 24, 16
# float64: SP and TP sum in different orders; fp32 rounding then varies by CPU.
F64 = torch.float64


class FakeTP:
    """A TP group of `world_size` thread ranks."""

    def __init__(self, world_size: int, monkeypatch) -> None:
        self.world_size = world_size
        self._barrier = threading.Barrier(world_size, timeout=60)
        self._inputs: list = [None] * world_size
        self._local = threading.local()
        pp = SimpleNamespace(world_size=1, is_first_rank=True, is_last_rank=True)
        for owner, name, value in [
            (vllm.distributed, "get_tp_group", lambda: self),
            (vllm.distributed, "get_pp_group", lambda: pp),
            (qwen3_vl, "get_pp_group", lambda: pp),
            (vllm.distributed, "tensor_model_parallel_all_gather", self.all_gather),
            (sequence_parallel, "all_to_all", self.all_to_all),
        ]:
            monkeypatch.setattr(owner, name, value)

    @property
    def rank_in_group(self) -> int:
        return self._local.rank

    def _every(self, x: torch.Tensor) -> list[torch.Tensor]:
        self._inputs[self.rank_in_group] = x
        self._barrier.wait()
        every = list(self._inputs)
        self._barrier.wait()
        return every

    def all_gather(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        return torch.cat(self._every(x), dim)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(self._every(x)).sum(0)

    def all_to_all(self, x: torch.Tensor, group: "FakeTP") -> torch.Tensor:
        return torch.stack([blocks[self.rank_in_group] for blocks in self._every(x)])

    def run(self, fn):
        def on_rank(rank: int):
            self._local.rank = rank
            return fn(rank)

        with ThreadPoolExecutor(self.world_size) as pool:
            return list(pool.map(on_rank, range(self.world_size)))


def module(forward, base: type[nn.Module] = nn.Module, **attrs) -> nn.Module:
    """A `base` that runs `forward` (upstream's) over `attrs`."""
    m = type("Fake", (base,), {"forward": forward, "__init__": nn.Module.__init__})()
    for name, value in attrs.items():
        setattr(m, name, value)
    return m


def linear(
    out_size: int, in_size: int, reduce=None, base=nn.Module, bias=None, tuple_out=True
) -> nn.Module:
    """One rank's shard of a TP linear; `reduce` all-reduces a row-parallel
    one's output, before its `bias`."""

    def forward(self, x: torch.Tensor):
        y = x @ self.weight.t()
        y = y if reduce is None else reduce(y)
        y = y if self.bias is None else y + self.bias
        return (y, None) if tuple_out else y

    w = torch.randn(out_size, in_size, dtype=F64) / in_size**0.5
    w = nn.Parameter(w, requires_grad=False)
    b = None if bias is None else nn.Parameter(bias, requires_grad=False)
    return module(
        forward, base, weight=w, bias=b, quant_method=UnquantizedLinearMethod()
    )


def norm(x: torch.Tensor, residual: torch.Tensor | None = None):
    """RMSNorm, with and without the fused residual add."""
    y = x if residual is None else x + residual
    out = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + 1e-6)
    return out if residual is None else (out, y)


def causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention over [..., rows, heads, head_dim]; causal, so row order matters."""
    q, k, v = (t.transpose(-3, -2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return out.transpose(-3, -2)


def decoder_layer(tp: FakeTP) -> nn.Module:
    """One rank's shard of a Qwen3 decoder layer: 2 query heads, 1 KV head."""
    attn = module(
        qwen3.Qwen3Attention.forward,
        head_dim=HEAD_DIM,
        q_size=2 * HEAD_DIM,
        kv_size=HEAD_DIM,
        qkv_proj=linear(4 * HEAD_DIM, HIDDEN),
        o_proj=linear(HIDDEN, 2 * HEAD_DIM, tp.all_reduce),
        q_norm=norm,
        k_norm=norm,
        rotary_emb=lambda pos, q, k: (q * pos.cos()[:, None], k * pos.sin()[:, None]),
        attn=lambda *qkv: causal(
            *(t.unflatten(-1, (-1, HEAD_DIM)) for t in qkv)
        ).flatten(-2),
    )
    mlp = module(
        qwen2.Qwen2MLP.forward,
        gate_up_proj=linear(2 * INTER, HIDDEN),
        down_proj=linear(HIDDEN, INTER, tp.all_reduce),
        act_fn=lambda x: F.silu(x[..., :INTER]) * x[..., INTER:],
    )
    return module(
        qwen3.Qwen3DecoderLayer.forward,
        self_attn=attn,
        mlp=mlp,
        input_layernorm=norm,
        post_attention_layernorm=norm,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_sp_prefill_matches_tp(monkeypatch, world_size):
    tp = FakeTP(world_size, monkeypatch)
    torch.manual_seed(0)
    backbones = [
        module(
            qwen3_vl.Qwen3LLMModel.forward,
            layers=nn.ModuleList(decoder_layer(tp) for _ in range(2)),
            start_layer=0,
            end_layer=2,
            norm=norm,
            _maybe_add_hidden_state=lambda *args: [],
        )
        for _ in range(world_size)
    ]
    x, positions = torch.randn(ROWS, HIDDEN, dtype=F64), torch.arange(ROWS, dtype=F64)
    deepstack = IntermediateTensors({"deepstack_input_embeds_0": torch.randn_like(x)})

    def on_rank(rank: int):
        backbone = backbones[rank]
        tp_out = backbone(None, positions, None, x, deepstack)
        sp = SequenceParallelPrefill(backbone, ROWS, [ROWS])
        return tp_out, sp(backbone, None, positions, x, deepstack)

    for tp_out, sp_out in tp.run(on_rank):
        torch.testing.assert_close(sp_out, tp_out)


def test_sp_prefill_rejects_tp_buckets_above_the_threshold(monkeypatch):
    # A TP-graph bucket after sequence-parallel ones used to crash warmup.
    tp = FakeTP(2, monkeypatch)
    with pytest.raises(ValueError, match=r"multiple of TP=2, got \[1025\]"):
        tp.run(lambda _: SequenceParallelPrefill(nn.Linear(1, 1), 1024, [512, 1025]))


def vision_block(tp: FakeTP, row_bias: torch.Tensor) -> nn.Module:
    """One rank's shard of a Qwen3-VL vision block: 2 heads. A row-parallel
    bias is the same on every rank."""

    def column(out_size: int, in_size: int, **kwargs) -> nn.Module:
        bias = torch.randn(out_size, dtype=F64)
        return linear(out_size, in_size, base=ColumnParallelLinear, bias=bias, **kwargs)

    row = partial(linear, reduce=tp.all_reduce, base=RowParallelLinear, bias=row_bias)
    attn = module(
        qwen2_5_vl.Qwen2_5_VisionAttention.forward,
        num_attention_heads_per_partition=2,
        hidden_size_per_attention_head=HEAD_DIM,
        qkv=column(6 * HEAD_DIM, HIDDEN),
        proj=row(HIDDEN, 2 * HEAD_DIM),
        apply_rotary_emb=lambda x, cos, sin: x * torch.cat([cos, sin], -1)[:, None],
        attn=lambda query, key, value, **_: causal(query, key, value),
    )
    mlp = module(
        qwen3_vl.Qwen3_VisionMLP.forward,
        linear_fc1=column(INTER, HIDDEN, tuple_out=False),
        linear_fc2=row(HIDDEN, INTER, tuple_out=False),
        act_fn=F.gelu,
    )
    return module(
        qwen3_vl.Qwen3_VisionBlock.forward, attn=attn, mlp=mlp, norm1=norm, norm2=norm
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_sp_vision_block_matches_tp(monkeypatch, world_size):
    tp = FakeTP(world_size, monkeypatch)
    torch.manual_seed(0)
    row_bias = torch.randn(HIDDEN, dtype=F64)
    blocks = [vision_block(tp, row_bias) for _ in range(world_size)]
    x = torch.randn(ROWS, 1, HIDDEN, dtype=F64)
    angles = torch.randn(ROWS, HEAD_DIM // 2, dtype=F64)
    inputs = dict(
        cu_seqlens=None,
        rotary_pos_emb_cos=angles.cos(),
        rotary_pos_emb_sin=angles.sin(),
        max_seqlen=None,
        sequence_lengths=None,
    )

    def on_rank(rank: int):
        block = blocks[rank]
        exchange = sequence_parallel.RowExchange(torch.device("cpu"))
        sp, shards = SimpleNamespace(exchange=exchange), _BlockShards(block)
        own = exchange.own_rows(x)
        return block(x, **inputs), SequenceParallelVision._block(
            sp, block, shards, own, inputs
        )

    results = tp.run(on_rank)
    torch.testing.assert_close(
        torch.cat([sp_out for _, sp_out in results]), results[0][0]
    )
