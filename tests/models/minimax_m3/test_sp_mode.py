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
"""MiniMax M3 sequence parallelism (TPU_MINIMAX_M3_SP_*), on CPU.

The TP ranks are simulated in one process: every activation carries a leading
rank axis ``[R, ...]`` and the SP collectives act across it. Attention and
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
from vllm_torchtpu.models.minimax_m3 import sp

R, T, H = 4, 16, 64

_SP_ENV = (
    "TPU_MINIMAX_M3_SP_MODE",
    "TPU_MINIMAX_M3_SP_MIN_TOKENS",
    "TPU_MINIMAX_M3_SP_FP8_GATHER",
    "TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER",
    "TPU_MOE_COMBINE_HIER_RS",
    "TPU_MINIMAX_M3_SP_ATTN_HIER_RS",
)


def _rs(x, hier=False):
    total = x.sum(0)
    return torch.stack(total.chunk(x.shape[0], dim=0))


def _fp8_rows_torch(x):
    scale = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / sp.FP8_MAX
    q = (x.float() / scale).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(x.dtype)


def _ag(x, fp8=False, num_tokens=None):
    if fp8:
        x = _fp8_rows_torch(x)
    full = torch.cat(list(x), dim=0)
    if num_tokens is not None:
        assert full.shape[0] == num_tokens
    return full.unsqueeze(0).expand(x.shape[0], *full.shape).clone()


def _ar(x):
    return x.sum(0, keepdim=True).expand_as(x).clone()


@pytest.fixture
def collectives(monkeypatch):
    calls = {"rs": [], "ag": []}

    def rs(x, hier=False):
        calls["rs"].append(hier)
        return _rs(x)

    def ag(x, fp8, num_tokens):
        calls["ag"].append(fp8)
        return _ag(x, fp8, num_tokens)

    monkeypatch.setattr(sp, "reduce_scatter", rs)
    monkeypatch.setattr(sp, "all_gather", ag)
    monkeypatch.setattr(mm, "tensor_model_parallel_all_reduce", _ar)
    return calls


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
        return torch.einsum(
            "rth,rhk->rtk", torch.tanh(torch.einsum("rth,rhk->rtk", x, self.a)), self.b
        )

    def __call__(self, x):
        out = self.partial(x)
        return _ar(out) if self.reduce_flag() else out


class _Attn:
    def __init__(self, partial, o_proj):
        self.partial, self.o_proj = partial, o_proj
        self.reuse_seen = []

    def __call__(self, positions, x, index_reuse=None):
        self.reuse_seen.append(index_reuse)
        return self.partial(x)


class _Ffn:
    def __init__(self, partial, **attrs):
        self.partial = partial
        for k, v in attrs.items():
            setattr(self, k, v)

    def __call__(self, x):
        return self.partial(x)


def _layer(seed, moe, cfg):
    layer = object.__new__(mm.MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.is_moe_layer = moe
    layer.tp_size = R
    layer.sp = cfg
    o_proj = types.SimpleNamespace(reduce_results=True)
    layer.self_attn = _Attn(_Partial(seed, lambda: o_proj.reduce_results), o_proj)
    if moe:
        moe_cfg = types.SimpleNamespace(skip_final_all_reduce=False)
        experts = types.SimpleNamespace(moe_config=moe_cfg, _quant_method=None)
        ffn = _Partial(seed + 1, lambda: not moe_cfg.skip_final_all_reduce)
        layer.block_sparse_moe = _Ffn(ffn, experts=experts)
    else:
        down = types.SimpleNamespace(reduce_results=True)
        ffn = _Partial(seed + 1, lambda: down.reduce_results)
        layer.mlp = _Ffn(ffn, down_proj=down)
    layer.input_layernorm = _Norm(seed + 2)
    layer.post_attention_layernorm = _Norm(seed + 3)
    return layer


def _plain(layers, x):
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer(None, hidden, residual)
    return hidden + residual


def _input():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(T, H, generator=g, dtype=torch.float64)
    return x.unsqueeze(0).expand(R, T, H).clone()


def _layers(cfg):
    return [_layer(10 * i, moe=bool(i % 2), cfg=cfg) for i in range(4)]


def _assert_flags_restored(layers):
    for layer in layers:
        assert layer.self_attn.o_proj.reduce_results is True
        if layer.is_moe_layer:
            assert not layer.block_sparse_moe.experts.moe_config.skip_final_all_reduce
        else:
            assert layer.mlp.down_proj.reduce_results is True


def test_attn_mode_matches_plain(collectives):
    layers, x = _layers(sp.SpConfig("attn", 8)), _input()
    want = _plain(layers, x)
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer(None, hidden, residual, attn_rs_tokens=T)
    got = hidden + residual
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    _assert_flags_restored(layers)


def test_full_mode_matches_plain(collectives):
    layers, x = _layers(sp.SpConfig("full", 8)), _input()
    want = _plain(layers, x)
    shard = None
    for layer in layers:
        shard = layer.forward_sp(None, x, shard, T)
    got = _ag(shard)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    _assert_flags_restored(layers)
    # Per layer: 2 reduce-scatters; 1 gather on layer 0, 2 after.
    assert len(collectives["rs"]) == 8
    assert len(collectives["ag"]) == 7


@pytest.mark.parametrize("mode", ["attn", "full"])
def test_fp8_gathers_stay_within_e4m3_rounding(collectives, mode):
    """e4m3 keeps 3 mantissa bits: a gathered value is off by at most 2^-4 of
    itself. Through 4 layers of tanh MLPs the stream stays within a few
    percent of its scale."""
    layers, x = _layers(sp.SpConfig(mode, 8)), _input()
    want = _plain(layers, x)
    for layer in layers:
        layer.sp = sp.SpConfig(mode, 8, fp8_gather=True, fp8_attn_gather=True)
    if mode == "attn":
        hidden, residual = x, None
        for layer in layers:
            hidden, residual = layer(None, hidden, residual, attn_rs_tokens=T)
        got = hidden + residual
    else:
        shard = None
        for layer in layers:
            shard = layer.forward_sp(None, x, shard, T)
        got = _ag(shard)
    err = (got - want).abs().max() / want.abs().max()
    assert 0 < err < 0.05
    assert all(collectives["ag"])


@pytest.mark.parametrize("attn_hier", [False, True])
def test_full_mode_routes_hier_reduce_scatters(collectives, attn_hier):
    """ffn_hier_rs / attn_hier_rs reach sp.reduce_scatter's `hier` flag for
    the FFN / attention outputs of a "full" step, nothing else."""
    cfg = sp.SpConfig("full", 8, ffn_hier_rs=True, attn_hier_rs=attn_hier)
    layers, x = _layers(cfg), _input()
    shard = None
    for layer in layers:
        shard = layer.forward_sp(None, x, shard, T)
    # Per layer: attention RS then FFN RS.
    assert collectives["rs"] == [attn_hier, True] * 4


def test_index_reuse_reaches_attention_on_sp_paths(collectives):
    layers, x = _layers(sp.SpConfig("full", 8)), _input()
    reuse = {}
    shard = None
    for layer in layers:
        shard = layer.forward_sp(None, x, shard, T, index_reuse=reuse)
    assert all(layer.self_attn.reuse_seen == [reuse] for layer in layers)


def test_moe_layer_armed_with_its_own_reduction_is_refused(collectives):
    layer = _layer(0, moe=True, cfg=sp.SpConfig("attn", 8))
    layer.block_sparse_moe.experts._fused_moe_decode_armed = True
    with pytest.raises(ValueError, match="TPU_MINIMAX_M3_SP_MODE"):
        layer(None, _input(), None, attn_rs_tokens=T)


def test_step_needs_a_warmup_trace_arming_and_the_thresholds(monkeypatch):
    cfg = sp.SpConfig("full", 1024)
    monkeypatch.setattr(sp, "_OPS", object())
    assert not sp.step(cfg, 4096, 8)  # eager: never
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(sp.shape_variants, "aot_cache_tag", lambda: "trace1")
    assert sp.step(cfg, 1024, 8)
    assert not sp.step(cfg, 1016, 8)  # below the min
    assert not sp.step(cfg, 1028, 8)  # not a multiple of TP
    assert not sp.step(cfg, 4096, 1)  # no TP
    assert not sp.step(sp.OFF, 4096, 8)  # off
    assert not sp.step(sp.SpConfig("full", 0), 4096, 8)  # min 0 = never
    monkeypatch.setattr(sp.shape_variants, "aot_cache_tag", lambda: None)
    assert not sp.step(cfg, 4096, 8)  # not a warmup trace
    monkeypatch.setattr(sp.shape_variants, "aot_cache_tag", lambda: "trace1")
    monkeypatch.setattr(sp, "_OPS", None)
    assert not sp.step(cfg, 4096, 8)  # not armed


def test_step_off_never_reads_the_token_count():
    """Key off: no comparison on the token count, so no shape guard and no
    retrace (the plain trace stays the only one)."""

    class Untouchable:
        def __ge__(self, other):
            raise AssertionError("token count compared")

        __mod__ = __lt__ = __le__ = __gt__ = __ge__

    assert not sp.step(sp.OFF, Untouchable(), 8)


def _set_env(monkeypatch, env):
    for name in _SP_ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_config_from_env(monkeypatch):
    _set_env(monkeypatch, {})
    assert sp.config_from_env() == sp.OFF
    _set_env(monkeypatch, {"TPU_MINIMAX_M3_SP_MODE": "off"})
    assert sp.config_from_env() == sp.OFF
    _set_env(monkeypatch, {"TPU_MINIMAX_M3_SP_MODE": "full"})
    assert sp.config_from_env() == sp.SpConfig("full", 4096)
    _set_env(
        monkeypatch,
        {"TPU_MINIMAX_M3_SP_MODE": "full", "TPU_MINIMAX_M3_SP_MIN_TOKENS": "0"},
    )
    assert sp.config_from_env() == sp.OFF
    _set_env(
        monkeypatch,
        {
            "TPU_MINIMAX_M3_SP_MODE": "full",
            "TPU_MINIMAX_M3_SP_MIN_TOKENS": "1024",
            "TPU_MINIMAX_M3_SP_FP8_GATHER": "1",
            "TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER": "1",
        },
    )
    assert sp.config_from_env() == sp.SpConfig("full", 1024, True, True)
    _set_env(
        monkeypatch,
        {
            "TPU_MINIMAX_M3_SP_MODE": "full",
            "TPU_MINIMAX_M3_SP_MIN_TOKENS": "1024",
            "TPU_MOE_COMBINE_HIER_RS": "1",
            "TPU_MINIMAX_M3_SP_ATTN_HIER_RS": "1",
        },
    )
    assert sp.config_from_env() == sp.SpConfig("full", 1024, False, False, True, True)
    # hierrs only takes "full" steps' reduce-scatters.
    _set_env(
        monkeypatch,
        {"TPU_MINIMAX_M3_SP_MODE": "attn", "TPU_MOE_COMBINE_HIER_RS": "1"},
    )
    assert sp.config_from_env() == sp.SpConfig("attn", 4096)
    _set_env(monkeypatch, {"TPU_MINIMAX_M3_SP_MODE": "both"})
    with pytest.raises(ValueError, match="TPU_MINIMAX_M3_SP_MODE"):
        sp.config_from_env()
    _set_env(
        monkeypatch,
        {"TPU_MINIMAX_M3_SP_MODE": "full", "TPU_MINIMAX_M3_SP_ATTN_HIER_RS": "1"},
    )
    with pytest.raises(ValueError, match="TPU_MOE_COMBINE_HIER_RS"):
        sp.config_from_env()


def _fake_model(layers, cfg, aux=(0, 2, 4)):
    return types.SimpleNamespace(
        aux_hidden_state_layers=aux,
        layers=layers,
        norm=_Norm(99),
        sp=cfg,
        tp_size=R,
        index_reuse=False,
    )


@pytest.mark.parametrize("mode", ["off", "attn", "full"])
def test_model_forward_sp_branches(collectives, monkeypatch, mode):
    """`MiniMaxM3Model.forward` on an SP step: the final norm's output and
    the drafter's aux hidden states match the plain path's."""
    cfg = sp.SpConfig(mode, 8)
    layers, x = _layers(cfg), _input()
    monkeypatch.setattr(sp, "step", lambda config, n, tp: config.on)
    fake = _fake_model(layers, cfg)
    positions = torch.zeros(T, dtype=torch.long)
    # x carries the simulated rank axis first; the step's token count is T.
    ids = torch.zeros(T, dtype=torch.long)
    got, aux = mm.MiniMaxM3Model.forward(fake, ids, positions, x)
    plain = _fake_model(layers, sp.OFF)
    want, want_aux = mm.MiniMaxM3Model.forward(plain, ids, positions, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert len(aux) == len(want_aux) == 3
    for a, b in zip(aux, want_aux):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    _assert_flags_restored(layers)


def test_model_forward_off_never_calls_sp(monkeypatch):
    """Key off (the default): the forward never asks `sp` anything and calls
    every layer exactly as the base did."""

    def boom(*_a, **_k):
        raise AssertionError("sp reached with SP off")

    monkeypatch.setattr(sp, "step", boom)
    monkeypatch.setattr(sp, "reduce_scatter", boom)
    monkeypatch.setattr(sp, "all_gather", boom)
    seen = []

    class _Layer:
        def __call__(self, positions, hidden, residual, **kw):
            seen.append(kw)
            return hidden, hidden if residual is None else residual

    fake = types.SimpleNamespace(
        aux_hidden_state_layers=(),
        layers=[_Layer(), _Layer()],
        norm=lambda h, r: (h + r, None),
        index_reuse=False,
        tp_size=8,
    )  # no `sp` attribute at all: stand-ins built without __init__
    x = torch.ones(4, 8)
    mm.MiniMaxM3Model.forward(fake, None, torch.zeros(4), x)
    fake.sp = sp.OFF
    mm.MiniMaxM3Model.forward(fake, None, torch.zeros(4), x)
    assert seen == [{}] * 4


def test_reduce_scatter_hier_routing(monkeypatch):
    """`hier` takes hierrs_tc only when it is armed for TP, lines up with the
    SP ops and the call is eligible; else the SP reduce-scatter (never the
    group's torch collective)."""
    from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

    calls = []

    def own(x, *rank):
        calls.append(("own", len(rank)))
        return x[: x.shape[0] // 8]

    def hier(x):
        calls.append(("hier", 0))
        return x[: x.shape[0] // 8]

    rank = torch.zeros(1, 1, dtype=torch.int32)
    ops = sp._Ops(8, own, rank, None, None, hier_ok=True)
    monkeypatch.setattr(sp, "_OPS", ops)
    monkeypatch.setattr(hrs, "_ARMED", hrs._Armed(hier, tuple(range(8)), "tp"))
    x = torch.zeros(128, 1024, dtype=torch.bfloat16)
    sp.reduce_scatter(x)
    sp.reduce_scatter(x, hier=True)
    sp.reduce_scatter(x.float(), hier=True)  # not bf16: ineligible
    monkeypatch.setattr(hrs, "_ARMED", hrs._Armed(hier, tuple(range(8)), "ep"))
    sp.reduce_scatter(x, hier=True)  # armed for the EP=DP combine only
    monkeypatch.setattr(hrs, "_ARMED", hrs._Armed(hier, tuple(range(8)), "tp"))
    monkeypatch.setattr(sp, "_OPS", sp._Ops(8, own, None, None, None, hier_ok=False))
    sp.reduce_scatter(x, hier=True)  # mesh not rank-ordered
    assert calls == [
        ("own", 1),
        ("hier", 0),
        ("own", 1),
        ("own", 1),
        ("own", 0),
    ]
