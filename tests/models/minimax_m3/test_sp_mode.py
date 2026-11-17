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
"""MiniMax M3 sequence parallelism (additional_config minimax_sp_*), on CPU.

The TP ranks are simulated in one process: every activation carries a leading
rank axis ``[R, ...]`` and the TP collectives act across it. Attention and
FFN are stand-ins that return each rank's partial sum (a rank's weight
shard), all-reduced unless the layer took the reduction over. The SP layers
must give the plain path's result (f64: up to summation order), and the fp8
gathers that result within e4m3 rounding.
"""

import types

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.minimax_m3 import model as mm

R, T, H = 4, 16, 64


def _rs(x, dim=0):
    total = x.sum(0)
    return torch.stack(total.chunk(x.shape[0], dim=0))


def _ag(x, dim=0):
    full = torch.cat(list(x), dim=0)
    return full.unsqueeze(0).expand(x.shape[0], *full.shape).clone()


def _ar(x):
    return x.sum(0, keepdim=True).expand_as(x).clone()


@pytest.fixture
def collectives(monkeypatch):
    monkeypatch.setattr(mm, "tensor_model_parallel_reduce_scatter", _rs)
    monkeypatch.setattr(mm, "tensor_model_parallel_all_gather", _ag)
    monkeypatch.setattr(mm, "tensor_model_parallel_all_reduce", _ar)


class _Norm(nn.Module):
    """GemmaRMSNorm's interface: norm(x) or fused add + norm(x, residual)."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(H, generator=g, dtype=torch.float64) * 0.1

    def _norm(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + 1e-6) * (1.0 + self.weight)

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        x = x + residual
        return self._norm(x), x


class _Partial(nn.Module):
    """Rank r's shard: tanh(x @ A_r) @ B_r; summed over ranks = the full op."""

    def __init__(self, seed, reduce_flag):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.a = torch.randn(R, H, H, generator=g, dtype=torch.float64) / H**0.5
        self.b = torch.randn(R, H, H, generator=g, dtype=torch.float64) / H**0.5
        self.reduce_flag = reduce_flag

    def partial(self, x):
        return torch.einsum("rth,rhk->rtk", torch.tanh(torch.einsum(
            "rth,rhk->rtk", x, self.a)), self.b)

    def __call__(self, x):
        out = self.partial(x)
        return _ar(out) if self.reduce_flag() else out


def _layer(seed, moe, sp):
    layer = object.__new__(mm.MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.is_moe_layer = moe
    layer.tp_size = R
    layer.sp = sp
    layer.split_min_tokens = 0
    o_proj = types.SimpleNamespace(reduce_results=True)
    attn = _Partial(seed, lambda: o_proj.reduce_results)
    layer.self_attn = _Attn(attn, o_proj)
    if moe:
        cfg = types.SimpleNamespace(skip_final_all_reduce=False)
        experts = types.SimpleNamespace(moe_config=cfg, _quant_method=None)
        ffn = _Partial(seed + 1, lambda: not cfg.skip_final_all_reduce)
        layer.block_sparse_moe = _Ffn(ffn, experts=experts)
    else:
        down = types.SimpleNamespace(reduce_results=True)
        ffn = _Partial(seed + 1, lambda: down.reduce_results)
        layer.mlp = _Ffn(ffn, down_proj=down)
    layer.input_layernorm = _Norm(seed + 2)
    layer.post_attention_layernorm = _Norm(seed + 3)
    return layer


class _Attn:
    def __init__(self, partial, o_proj):
        self.partial, self.o_proj = partial, o_proj

    def __call__(self, positions, x, metadata=None, *index_reuse):
        # Only given with MSA index reuse on (`_self_attn`).
        self.index_reuse = index_reuse[0] if index_reuse else None
        return self.partial(x)


class _Ffn:
    def __init__(self, partial, **attrs):
        self.partial = partial
        for k, v in attrs.items():
            setattr(self, k, v)

    def __call__(self, x):
        return self.partial(x)


def _plain(layers, x):
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer(None, hidden, residual)
    return hidden + residual


def _input():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(T, H, generator=g, dtype=torch.float64)
    return x.unsqueeze(0).expand(R, T, H).clone()


def _layers(sp):
    return [_layer(10 * i, moe=bool(i % 2), sp=sp) for i in range(4)]


def test_attn_mode_matches_plain(collectives):
    sp = mm.SpConfig("attn", 8)
    layers, x = _layers(sp), _input()
    want = _plain(layers, x)
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer(None, hidden, residual, attn_rs=True)
    got = hidden + residual
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    # The flags are restored after each SP layer.
    for layer in layers:
        assert layer.self_attn.o_proj.reduce_results is True
        if layer.is_moe_layer:
            assert not layer.block_sparse_moe.experts.moe_config.skip_final_all_reduce
        else:
            assert layer.mlp.down_proj.reduce_results is True


def test_full_mode_matches_plain(collectives):
    sp = mm.SpConfig("full", 8)
    layers, x = _layers(sp), _input()
    want = _plain(layers, x)
    shard = None
    for layer in layers:
        shard = layer.forward_sp(None, x, shard)
    got = _ag(shard)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["attn", "full"])
def test_fp8_gathers_stay_within_e4m3_rounding(collectives, mode):
    """e4m3 keeps 3 mantissa bits: a gathered value is off by at most 2^-4
    of itself (plus the per-row scale's subnormal floor). Through 4 layers of
    tanh MLPs the stream stays within a few percent of its scale."""
    layers, x = _layers(mm.SpConfig(mode, 8)), _input()
    want = _plain(layers, x)
    for layer in layers:
        layer.sp = mm.SpConfig(mode, 8, fp8_gather=True, fp8_attn_gather=True)
    if mode == "attn":
        hidden, residual = x, None
        for layer in layers:
            hidden, residual = layer(None, hidden, residual, attn_rs=True)
        got = hidden + residual
    else:
        shard = None
        for layer in layers:
            shard = layer.forward_sp(None, x, shard)
        got = _ag(shard)
    err = (got - want).abs().max() / want.abs().max()
    assert 0 < err < 0.05


def test_all_gather_rows_fp8(collectives):
    g = torch.Generator().manual_seed(3)
    x = (torch.randn(R, T, H, generator=g) * 10).to(torch.bfloat16)
    exact = mm._all_gather_rows(x, False)
    assert torch.equal(exact, _ag(x))
    got = mm._all_gather_rows(x, True).float()
    ref = exact.float()
    amax = ref.abs().amax(-1, keepdim=True)
    # Relative 2^-4 of the value (e4m3) plus bf16 output rounding; values
    # below e4m3's normal range (2^-6 of the row amax / 448 * 448) round to
    # its subnormal step.
    tol = ref.abs() * (2**-4 + 2**-8) + amax * 2**-9
    assert torch.all((got - ref).abs() <= tol)


def test_moe_layer_armed_with_its_own_reduction_is_refused(collectives):
    layer = _layer(0, moe=True, sp=mm.SpConfig("attn", 8))
    layer.block_sparse_moe.experts._fused_moe_decode_armed = True
    with pytest.raises(ValueError, match="minimax_sp_mode"):
        layer(None, _input(), None, attn_rs=True)


def test_sp_step_needs_a_warmup_trace_and_the_thresholds(monkeypatch):
    sp = mm.SpConfig("full", 1024)
    assert not mm._sp_step(sp, 4096, 8)  # eager: never
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(mm.shape_variants, "aot_cache_tag", lambda: "trace1")
    assert mm._sp_step(sp, 1024, 8)
    assert not mm._sp_step(sp, 1016, 8)  # below the min
    assert not mm._sp_step(sp, 1028, 8)  # not a multiple of TP
    assert not mm._sp_step(sp, 4096, 1)  # no TP
    assert not mm._sp_step(mm.SpConfig(), 4096, 8)  # off
    monkeypatch.setattr(mm.shape_variants, "aot_cache_tag", lambda: None)
    assert not mm._sp_step(sp, 4096, 8)  # not a warmup trace


def test_keys():
    assert mm.sp_config({}) == mm.SpConfig()
    assert mm.sp_config({"minimax_sp_mode": "off"}) == mm.SpConfig()
    assert mm.sp_config({"minimax_sp_mode": "full"}) == mm.SpConfig("full", 4096)
    assert mm.sp_config(
        {
            "minimax_sp_mode": "full",
            "minimax_sp_min_tokens": 1024,
            "minimax_sp_fp8_gather": True,
            "minimax_sp_fp8_attn_gather": True,
        }
    ) == mm.SpConfig("full", 1024, True, True)
    with pytest.raises(ValueError, match="minimax_sp_mode"):
        mm.sp_config({"minimax_sp_mode": "both"})
    with pytest.raises(ValueError, match="minimax_sp_min_tokens"):
        mm.sp_config({"minimax_sp_mode": "attn", "minimax_sp_min_tokens": -1})


@pytest.mark.parametrize("mode", ["", "attn", "full"])
def test_model_forward_sp_branches(collectives, monkeypatch, mode):
    """`MiniMaxM3Model.forward` on an SP step: the final norm's output and
    the drafter's aux hidden states match the plain path's."""
    sp = mm.SpConfig(mode, 8)
    layers, x = _layers(sp), _input()
    monkeypatch.setattr(mm, "_sp_step", lambda config, n, tp: bool(config.mode))
    norm = _Norm(99)
    fake = types.SimpleNamespace(
        split_min_tokens=0,
        aux_hidden_state_layers=(0, 2, 4),
        layers=layers,
        norm=norm,
        sp=sp,
        tp_size=R,
    )
    positions = torch.zeros(T, dtype=torch.long)
    got, aux = mm.MiniMaxM3Model.forward(fake, None, positions, x)
    plain = types.SimpleNamespace(**{**vars(fake), "sp": mm.SpConfig()})
    want, want_aux = mm.MiniMaxM3Model.forward(plain, None, positions, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert len(aux) == len(want_aux) == 3
    for a, b in zip(aux, want_aux):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("mode", ["", "attn", "full"])
def test_model_forward_shares_the_index_reuse_dict(collectives, monkeypatch, mode):
    """MSA index reuse on: every layer's attention gets the step's one
    selection dict on SP steps too (so the decode pattern's Shared layers
    find their Full layer's selection), and None with it off."""
    sp = mm.SpConfig(mode, 8)
    layers, x = _layers(sp), _input()
    monkeypatch.setattr(mm, "_sp_step", lambda config, n, tp: bool(config.mode))
    positions = torch.zeros(T, dtype=torch.long)
    for on in (True, False):
        fake = types.SimpleNamespace(
            split_min_tokens=0,
            aux_hidden_state_layers=(),
            layers=layers,
            norm=_Norm(99),
            sp=sp,
            tp_size=R,
            index_reuse=on,
        )
        mm.MiniMaxM3Model.forward(fake, None, positions, x)
        dicts = [layer.self_attn.index_reuse for layer in layers]
        if on:
            assert isinstance(dicts[0], dict)
            assert all(d is dicts[0] for d in dicts)
        else:
            assert dicts == [None] * len(layers)
