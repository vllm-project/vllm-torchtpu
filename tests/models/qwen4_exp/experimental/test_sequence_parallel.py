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
"""Sequence-parallel prefill matches the replicated TP path.

TP ranks are simulated by threads that exchange tensors through a barrier, so
the decoder layer runs its real control flow on each rank's shard.
"""

import threading
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from tests.models.qwen4_exp.experimental.qwen38_test_utils import pin_plan
from vllm_torchtpu.models.qwen4_exp.experimental import moe_method
from vllm_torchtpu.models.qwen4_exp.experimental import sequence_parallel as sp
from vllm_torchtpu.models.qwen4_exp.experimental.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

HIDDEN, HC, LOWRANK, TOKENS = 128, 2, 16, 32


@pytest.fixture(autouse=True)
def cpu_default():
    with torch.device("cpu"):
        yield


class SimulatedTP:
    """Collectives over threads; each thread is one TP rank."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.local = threading.local()

    @property
    def rank(self):
        return self.local.rank

    def exchange(self, x):
        self.barrier.wait()
        self.slots[self.rank] = x
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values

    def total(self, x):
        values = self.exchange(x)
        out = values[0].float()
        for v in values[1:]:
            out = out + v.float()
        return out.to(x.dtype)

    def shard(self, x):
        return x.chunk(self.size, dim=0)[self.rank].clone()

    def gather(self, x, num_tokens):
        return torch.cat(self.exchange(x), dim=0).narrow(0, 0, num_tokens)

    def reduce(self, x, sequence_parallel):
        summed = self.total(x)
        return self.shard(summed) if sequence_parallel else summed

    def run(self, fn):
        results, errors = [None] * self.size, []

        def body(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


@pytest.fixture
def tp2(monkeypatch):
    tp = SimulatedTP(2)
    monkeypatch.setattr(sp, "shard", tp.shard)
    monkeypatch.setattr(sp, "gather", tp.gather)
    monkeypatch.setattr(sp, "reduce", tp.reduce)
    monkeypatch.setattr(sp, "tp_world_size", lambda: tp.size)
    return tp


def hc_module():
    module = GatedResidual(
        HyperConnectionConfig(
            hidden_size=HIDDEN,
            hc_count=HC,
            hc_lowrank=LOWRANK,
            hc_per_branch_norm=True,
        )
    )
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn_like(p) * 0.1)
    return module


class PartialAttention(nn.Module):
    """Rank r returns a fixed share of a linear map, like a row-parallel o_proj."""

    def __init__(self, tp, weight, layer_type, selection=None):
        super().__init__()
        self.tp, self.weight, self.layer_type = tp, weight, layer_type
        self.selection = selection
        self.decode_token_shard = False

    def _partial(self, block):
        share = (self.tp.rank + 1) / (self.tp.size * (self.tp.size + 1) / 2)
        return (block.float() @ self.weight.T * share).to(block.dtype)

    def forward(self, block, step_metadata_cache=None):  # GDN: un-reduced o_proj
        return self._partial(block)

    def forward_with_indices(
        self,
        block,
        positions,
        metadata,
        reduce=True,
        replicated_indexer=False,
        token_shard=False,
        decode_token_shard=False,
    ):
        assert reduce is False
        selected = None if self.selection is None else self.selection[: block.shape[0]]
        return self._partial(block), selected


class RowwiseMoE(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, block):
        return torch.tanh(block.float() @ self.weight.T).to(block.dtype)


class SplitMoE(nn.Module):
    """Routed experts on any rows + a TP-sliced shared expert, like MoERunner."""

    def __init__(self, tp, weight):
        super().__init__()
        self.tp, self.weight = tp, weight
        self.replicate_shared_expert = False
        self.gate = lambda x: (x.float() @ weight[:4].T, None)
        method = NS(**{moe_method.SEQUENCE_PARALLEL_OP_ATTR: object()})
        self.experts = NS(
            routed_experts=NS(
                quant_method=method,
                w13_weight=None,
                w2_weight=None,
                w13_weight_scale_inv=None,
                w2_weight_scale_inv=None,
            )
        )
        self.shared_expert = self._shared_partial

    def routed(self, block, logits):
        scale = torch.sigmoid(logits[:, :1])
        return (torch.tanh(block.float() @ self.weight.T) * scale).to(block.dtype)

    def _shared_partial(self, block):
        share = (self.tp.rank + 1) / (self.tp.size * (self.tp.size + 1) / 2)
        return (torch.sin(block.float() @ self.weight) * share).to(block.dtype)

    def forward(self, block):
        logits, _ = self.gate(block)
        return self.tp.total(self._shared_partial(block)) + self.routed(block, logits)


def make_layer(tp, layer_type, attn_hc, mlp_hc, attn_weight, moe_weight):
    from vllm_torchtpu.models.qwen4_exp.experimental.model import (
        Qwen4ExpDecoderLayer,
    )

    layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
    nn.Module.__init__(layer)
    layer.ple = None
    layer.layer_type = layer_type
    attention = PartialAttention(tp, attn_weight, layer_type)
    if layer_type == "linear_attention":
        layer.linear_attn = attention
    else:
        layer.self_attn = attention
    layer.attn_hyper_connection = attn_hc
    layer.mlp_hyper_connection = mlp_hc
    layer.mlp = RowwiseMoE(moe_weight)
    return layer


@pytest.mark.parametrize(
    "split_moe,shared_local", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_decoder_layer_sequence_parallel_matches_tp(
    tp2, monkeypatch, layer_type, split_moe, shared_local
):
    """shared_local: QWEN4_EXP_PREFILL_SP_SHARED_LOCAL, the shared expert's replicated
    copy on
    this rank's rows (no gather / reduce); one bf16 sum instead of the TP partials'
    reduce."""
    from vllm_torchtpu.models.qwen4_exp.experimental.kernels.adaptive_fused_moe import (  # noqa: E501
        vllm_adapter as adaptive_moe,
    )

    pin_plan(
        monkeypatch,
        QWEN4_EXP_PREFILL_SP=True,
        QWEN4_EXP_PREFILL_SP_MOE=split_moe,
        QWEN4_EXP_PREFILL_SP_SHARED_LOCAL=shared_local,
    )
    calls = []

    def fake_adaptive(method, block, w1, w2, s1, s2, logits, attr):
        calls.append((attr, block.shape[0]))
        return layers[tp2.rank].mlp.routed(block, logits)

    monkeypatch.setattr(adaptive_moe, "run_adaptive_fused_moe", fake_adaptive)
    layers = {}
    torch.random.default_generator.manual_seed(0)
    attn_hc, mlp_hc = hc_module(), hc_module()
    attn_w = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    moe_w = torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5
    hidden = torch.randn(TOKENS, HC * HIDDEN).bfloat16()
    input_ids = torch.zeros(TOKENS, dtype=torch.long)
    valid = torch.arange(TOKENS) < TOKENS - 5  # trailing padding rows
    # NaN padding rows must not reach the MoE on either path.
    hidden[~valid] = float("nan")

    def run(rank, sequence_parallel):
        layer = make_layer(tp2, layer_type, attn_hc, mlp_hc, attn_w, moe_w)
        layer.mlp = SplitMoE(tp2, moe_w)
        if shared_local and sequence_parallel:

            def gathered_shared(block):
                raise AssertionError(
                    "shared_expert_local must replace the gathered TP call"
                )

            layer.mlp.shared_expert = gathered_shared
            layer.mlp.shared_expert_local = lambda b: torch.sin(b.float() @ moe_w).to(
                b.dtype
            )
        layers[rank] = layer
        x = tp2.shard(hidden) if sequence_parallel else hidden
        return layer(
            x,
            None,
            input_ids,
            None,
            None,
            None,
            None,
            valid,
            None,
            None,
            sequence_parallel=sequence_parallel,
        )

    replicated = tp2.run(lambda r: run(r, False))
    sharded = tp2.run(lambda r: run(r, True))
    torch.testing.assert_close(
        replicated[0], replicated[1], rtol=0, atol=0, equal_nan=True
    )
    joined = torch.cat(sharded, dim=0)
    assert joined.shape == replicated[0].shape
    rows = valid
    # The replicated shared expert sums in one bf16 rounding, the TP path in two.
    atol = 3e-2 if shared_local else 0
    torch.testing.assert_close(joined[rows], replicated[0][rows], rtol=0, atol=atol)
    expected_calls = [(moe_method.SEQUENCE_PARALLEL_OP_ATTR, TOKENS // 2)] * 2
    assert calls == (expected_calls if split_moe else [])


def test_ple_gate_sequence_parallel_matches_tp(tp2):
    from vllm_torchtpu.models.qwen4_exp.experimental.ple import Qwen4ExpPLELayer

    torch.random.default_generator.manual_seed(1)
    embed = 24
    key_w = torch.randn(HC * HIDDEN, embed).bfloat16()
    value_w = torch.randn(HIDDEN, embed).bfloat16()

    def rms(x):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + 1e-6)).to(
            x.dtype
        )

    owner = NS(
        key_proj=lambda e: (e @ key_w.T, None),
        value_proj=lambda e: (e @ value_w.T, None),
        norm_key=rms,
        norm_query=rms,
        hc_count=HC,
        hidden_size=HIDDEN,
        gate_dot=None,  # the unfused gate
    )
    hidden = torch.randn(TOKENS, HC * HIDDEN).bfloat16()
    embeddings = torch.randn(TOKENS, embed).bfloat16()
    expected = Qwen4ExpPLELayer.gated_values(owner, hidden, embeddings)
    got = tp2.run(
        lambda r: Qwen4ExpPLELayer.gated_values(
            owner, tp2.shard(hidden), embeddings, TOKENS
        )
    )
    for g in got:  # values stay complete on every rank
        torch.testing.assert_close(g, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "enabled,tokens,tp,expected",
    [
        (False, 8192, 8, False),
        (True, 8192, 8, True),
        (True, 1024, 8, True),
        (True, 512, 8, False),
        (True, 1056, 8, False),  # shard would not be 8-row aligned
        (True, 8192, 1, False),
        (True, 64, 8, False),  # decode buckets stay under the 1024 floor
        (True, 1088, 8, True),
    ],
)
def test_sequence_parallel_eligibility(monkeypatch, enabled, tokens, tp, expected):
    pin_plan(monkeypatch, QWEN4_EXP_PREFILL_SP=enabled)
    assert sp.sequence_parallel_eligible(tokens, tp) is expected


def test_shard_rows_keep_every_hc_kernel_on_its_pallas_path():
    # Full mix tile 128 runs on 256-row multiples, up and norm on 128.
    assert sp.shard_rows([256, 128, 128]) == 256
    assert sp.shard_rows([0, 0, 0]) == sp.ROW_ALIGNMENT


@pytest.mark.parametrize(
    "tokens,expected", [(1024, False), (2048, True), (3072, False), (8192, True)]
)
def test_eligibility_with_hc_shard_rows(monkeypatch, tokens, expected):
    pin_plan(monkeypatch, QWEN4_EXP_PREFILL_SP=True)
    assert sp.sequence_parallel_eligible(tokens, 8, rows=256) is expected
