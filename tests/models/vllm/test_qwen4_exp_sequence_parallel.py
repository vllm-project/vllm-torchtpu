# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel prefill matches the replicated TP path.

TP ranks are simulated by threads that exchange tensors through a barrier, so
the decoder layer runs its real control flow on each rank's shard.
"""

import threading
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.qwen4_exp import sequence_parallel as sp
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
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

    def __init__(self, tp, weight, layer_type):
        super().__init__()
        self.tp, self.weight, self.layer_type = tp, weight, layer_type

    def _partial(self, block):
        share = (self.tp.rank + 1) / (self.tp.size * (self.tp.size + 1) / 2)
        return (block.float() @ self.weight.T * share).to(block.dtype)

    def forward(self, block):  # GDN: un-reduced output projection
        return self._partial(block)

    def forward_with_indices(self, block, positions, metadata, reduce=True):
        assert reduce is False
        return self._partial(block), None


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
        method = NS(_adaptive_fused_moe_op_sp=object())
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
    from vllm_torchtpu.models.vllm.qwen4_exp.model import Qwen4ExpDecoderLayer

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


@pytest.mark.parametrize("sp_pad", [0, 24])
@pytest.mark.parametrize("split_moe", [False, True])
@pytest.mark.parametrize("layer_type", ["linear_attention", "full_attention"])
def test_decoder_layer_sequence_parallel_matches_tp(
    tp2, monkeypatch, layer_type, split_moe, sp_pad
):
    from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import (
        vllm_adapter as adaptive_moe,
    )

    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MOE", split_moe, raising=False)
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
        layers[rank] = layer
        x = tp2.shard(hidden) if sequence_parallel else hidden
        pad = sp_pad if sequence_parallel else 0
        out = layer(
            sp.pad_shard(x, pad),
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
            sp_pad=pad,
        )
        if pad:
            # 16 real rows per rank, padded to 24: the tail stays zero.
            assert out.shape[0] == pad
            torch.testing.assert_close(
                out[x.shape[0] :], torch.zeros_like(out[x.shape[0] :])
            )
        return sp.unpad(out, x.shape[0])

    replicated = tp2.run(lambda r: run(r, False))
    sharded = tp2.run(lambda r: run(r, True))
    torch.testing.assert_close(
        replicated[0], replicated[1], rtol=0, atol=0, equal_nan=True
    )
    joined = torch.cat(sharded, dim=0)
    assert joined.shape == replicated[0].shape
    rows = valid
    torch.testing.assert_close(joined[rows], replicated[0][rows], rtol=0, atol=0)
    expected_calls = [("_adaptive_fused_moe_op_sp", TOKENS // 2)] * 2
    assert calls == (expected_calls if split_moe else [])


def test_ple_gate_sequence_parallel_matches_tp(tp2):
    from vllm_torchtpu.models.vllm.qwen4_exp.ple import Qwen4ExpPLELayer

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
    "enabled,minimum,tokens,tp,expected",
    [
        (False, 1024, 8192, 8, False),
        (True, 1024, 8192, 8, True),
        (True, 1024, 1024, 8, True),
        (True, 1024, 512, 8, False),
        (True, 1024, 1056, 8, False),  # shard would not be 8-row aligned
        (True, 1024, 8192, 1, False),
        (True, 0, 64, 8, False),  # decode buckets stay under the 1024 floor
        (True, 0, 1088, 8, True),
        (True, 2048, 1088, 8, False),
    ],
)
def test_sequence_parallel_eligibility(
    monkeypatch, enabled, minimum, tokens, tp, expected
):
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP", enabled, raising=False)
    monkeypatch.setattr(
        sp.envs, "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", minimum, raising=False
    )
    assert sp.sequence_parallel_eligible(tokens, tp) is expected


def test_shard_rows_keep_every_hc_kernel_on_its_pallas_path():
    # Full mix tile 128 runs on 256-row multiples, up and norm on 128.
    assert sp.shard_rows([256, 128, 128]) == 256
    assert sp.shard_rows([0, 0, 0]) == sp.ROW_ALIGNMENT


@pytest.mark.parametrize(
    "tokens,expected", [(1024, False), (2048, True), (3072, False), (8192, True)]
)
def test_eligibility_with_hc_shard_rows(monkeypatch, tokens, expected):
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP", True, raising=False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024, raising=False)
    assert sp.sequence_parallel_eligible(tokens, 8, rows=256) is expected


def test_pad_shard_rounds_up_with_zero_rows_and_unpad_restores():
    x = torch.randn(384, 4)
    padded = sp.pad_shard(x, 256)
    assert padded.shape == (512, 4)
    assert torch.equal(padded[384:], torch.zeros(128, 4))
    assert torch.equal(sp.unpad(padded, 384), x)
    flags = sp.pad_shard(torch.ones(128, dtype=torch.bool), 256)
    assert flags.dtype == torch.bool and not flags[128:].any()
    assert sp.pad_shard(x, 0) is x
    assert sp.padded_rows(256, 256) == 256


@pytest.mark.parametrize(
    "tokens,expected", [(960, False), (1024, True), (3072, True), (1088, True)]
)
def test_padded_shards_admit_every_row_aligned_bucket(monkeypatch, tokens, expected):
    # QWEN4_EXP_PREFILL_SP_PAD_HC: the rule falls back to ROW_ALIGNMENT rows.
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP", True, raising=False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MIN_TOKENS", 1024, raising=False)
    assert sp.sequence_parallel_eligible(tokens, 8) is expected


@pytest.mark.parametrize(
    "pad,moe,expected", [(True, True, True), (True, False, False), (False, True, False)]
)
def test_padded_shards_only_with_the_sequence_parallel_moe(
    monkeypatch, pad, moe, expected
):
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_PAD_HC", pad, raising=False)
    monkeypatch.setattr(sp.envs, "QWEN4_EXP_PREFILL_SP_MOE", moe, raising=False)
    assert sp.pad_hc_enabled() is expected
