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
"""MiniMax M3 two-batch overlap (TPU_MINIMAX_M3_PREFILL_OVERLAP*), on CPU.

The TP ranks are simulated in one process: every activation is
``[T, R, ...]`` (tokens first, so the layer's token halves are row slices)
and the all-reduces sum over the rank axis. Attention is a stand-in that
couples the tokens (a causal running mean of its qkv), o_proj and the FFN
return each rank's partial sum, all-reduced unless the layer took the
reduction over. The overlap layers must give the plain path's result (f64:
up to summation order).
"""

import types

import pytest
import torch
from torch import nn

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.models.minimax_m3 import model as mm
from vllm_torchtpu.models.minimax_m3 import overlap

R, T, H = 4, 32, 64

_ENV = (
    "TPU_MINIMAX_M3_PREFILL_OVERLAP",
    "TPU_MINIMAX_M3_PREFILL_OVERLAP_MIN_TOKENS",
    "TPU_MINIMAX_M3_PREFILL_OVERLAP_AR",
    "TPU_MINIMAX_M3_PREFILL_OVERLAP_QKV",
)


def _ar(x):
    return x.sum(1, keepdim=True).expand_as(x).clone()


@pytest.fixture
def collectives(monkeypatch):
    calls = {"ar": [], "wait": 0}

    def ar(x, route):
        calls["ar"].append((route, x.shape[0]))
        return _ar(x)

    def wait(x):
        calls["wait"] += 1
        return x

    monkeypatch.setattr(overlap, "all_reduce", ar)
    monkeypatch.setattr(overlap, "wait", wait)
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


def _weights(seed, n=2):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(R, H, H, generator=g, dtype=torch.float64) / H**0.5 for _ in range(n)]


class _Partial:
    """Rank r's shard: tanh(x @ A_r) @ B_r; summed over ranks = the full op.
    Records the token-padding row offset it was traced under."""

    def __init__(self, seed, reduce_flag):
        self.a, self.b = _weights(seed)
        self.reduce_flag = reduce_flag
        self.offsets = []

    def __call__(self, x):
        self.offsets.append(token_padding._row_offset)
        h = torch.tanh(torch.einsum("trh,rhk->trk", x, self.a))
        out = torch.einsum("trh,rhk->trk", h, self.b)
        return _ar(out) if self.reduce_flag() else out


class _OProj:
    def __init__(self, seed):
        (self.w,) = _weights(seed, 1)
        self.reduce_results = True
        self.rows = []

    def __call__(self, x):
        self.rows.append(x.shape[0])
        out = torch.einsum("trh,rhk->trk", x, self.w)
        return (_ar(out) if self.reduce_results else out), None


class _Attn:
    """qkv_proj: rank r's column shard; attend: causal running mean over the
    tokens (couples the halves); o_proj: row-parallel partial."""

    def __init__(self, seed):
        (self.wq,) = _weights(seed + 5, 1)
        self.o_proj = _OProj(seed + 6)
        self.reuse_seen = []
        self.qkv_rows = []

    def qkv_proj(self, x):
        self.qkv_rows.append(x.shape[0])
        return torch.einsum("trh,rhk->trk", x, self.wq), None

    def attend(self, positions, x, index_reuse=None, qkv=None):
        self.reuse_seen.append(index_reuse)
        if qkv is None:
            qkv, _ = self.qkv_proj(x)
        steps = torch.arange(1, qkv.shape[0] + 1, dtype=qkv.dtype)
        return qkv.cumsum(0) / steps[:, None, None]

    def __call__(self, positions, x, index_reuse=None):
        out, _ = self.o_proj(self.attend(positions, x, index_reuse))
        return out


class _Flags:
    """Attribute bag (Dynamo traces setattr on it, unlike SimpleNamespace)."""

    def __init__(self, **attrs):
        self.__dict__.update(attrs)


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
    layer.overlap = cfg
    layer.self_attn = _Attn(seed)
    if moe:
        moe_cfg = _Flags(skip_final_all_reduce=False)
        experts = _Flags(moe_config=moe_cfg, _quant_method=None)
        ffn = _Partial(seed + 1, lambda: not moe_cfg.skip_final_all_reduce)
        layer.block_sparse_moe = _Ffn(ffn, experts=experts)
    else:
        down = _Flags(reduce_results=True)
        ffn = _Partial(seed + 1, lambda: down.reduce_results)
        layer.mlp = _Ffn(ffn, down_proj=down)
    layer.input_layernorm = _Norm(seed + 2)
    layer.post_attention_layernorm = _Norm(seed + 3)
    return layer


def _ffn(layer):
    return (layer.block_sparse_moe if layer.is_moe_layer else layer.mlp).partial


def _input():
    g = torch.Generator().manual_seed(7)
    x = torch.randn(T, H, generator=g, dtype=torch.float64)
    return x.unsqueeze(1).expand(T, R, H).clone()


def _layers(cfg, n=4):
    return [_layer(10 * i, moe=bool(i % 2), cfg=cfg) for i in range(n)]


def _plain(layers, x):
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer(None, hidden, residual)
    return hidden + residual


def _overlapped(layers, x, **kw):
    hidden, residual = x, None
    for layer in layers:
        hidden, residual = layer.forward_overlap(None, hidden, residual, T // 2, **kw)
    return torch.cat([h + r for h, r in zip(hidden, residual)])


def _assert_flags_restored(layers):
    for layer in layers:
        assert layer.self_attn.o_proj.reduce_results is True
        if layer.is_moe_layer:
            assert not layer.block_sparse_moe.experts.moe_config.skip_final_all_reduce
        else:
            assert layer.mlp.down_proj.reduce_results is True


@pytest.mark.parametrize("qkv", [True, False], ids=["qkv_split", "qkv_full"])
@pytest.mark.parametrize("route", ["xla", "group"])
def test_overlap_layers_match_plain(collectives, qkv, route):
    cfg = overlap.OverlapConfig(8, route, qkv)
    layers, x = _layers(cfg), _input()
    want = _plain(layers, x)
    got = _overlapped(layers, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    _assert_flags_restored(layers)
    # Per layer: o_proj A, o_proj B, FFN A, FFN B, all half-size, in order.
    assert collectives["ar"] == [(route, T // 2)] * 16


def test_overlap_layer_work_runs_on_halves(collectives):
    cfg = overlap.OverlapConfig(8, "xla", True)
    layers, x = _layers(cfg), _input()
    _overlapped(layers, x)
    for i, layer in enumerate(layers):
        # o_proj per half; qkv_proj on all tokens for layer 0, per half after.
        assert layer.self_attn.o_proj.rows == [T // 2, T // 2]
        assert layer.self_attn.qkv_rows == ([T] if i == 0 else [T // 2, T // 2])
        # Half B's FFN reads the token-padding mask from row T // 2.
        assert _ffn(layer).offsets == [None, T // 2]
    assert token_padding._row_offset is None


def test_index_reuse_reaches_attention_on_the_overlap_path(collectives):
    cfg = overlap.OverlapConfig(8, "xla", True)
    layers, x = _layers(cfg), _input()
    reuse = {}
    _overlapped(layers, x, index_reuse=reuse)
    assert all(layer.self_attn.reuse_seen == [reuse] for layer in layers)


def test_moe_layer_armed_with_its_own_reduction_is_refused(collectives):
    layer = _layer(1, moe=True, cfg=overlap.OverlapConfig(8))
    layer.block_sparse_moe.experts._fused_moe_decode_armed = True
    with pytest.raises(ValueError, match="TPU_MINIMAX_M3_PREFILL_OVERLAP"):
        layer.forward_overlap(None, _input(), None, T // 2)


def test_step_needs_a_warmup_trace_and_the_thresholds(monkeypatch):
    cfg = overlap.OverlapConfig(4096)
    assert not overlap.step(cfg, 8192, 8)  # eager: never
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(overlap.shape_variants, "aot_cache_tag", lambda: "trace1")
    assert overlap.step(cfg, 4096, 8)
    assert overlap.step(cfg, 8192, 8)
    assert not overlap.step(cfg, 3072, 8)  # below the min
    assert not overlap.step(cfg, 4112, 8)  # not a multiple of 32
    assert not overlap.step(cfg, 8192, 1)  # no TP
    assert not overlap.step(overlap.OFF, 8192, 8)  # off
    monkeypatch.setattr(overlap.shape_variants, "aot_cache_tag", lambda: None)
    assert not overlap.step(cfg, 8192, 8)  # not a warmup trace


def test_step_off_never_reads_the_token_count():
    class Untouchable:
        def __ge__(self, other):
            raise AssertionError("token count compared")

        __mod__ = __lt__ = __le__ = __gt__ = __ge__

    assert not overlap.step(overlap.OFF, Untouchable(), 8)


def _set_env(monkeypatch, env):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)


def test_config_from_env(monkeypatch):
    _set_env(monkeypatch, {})
    assert overlap.config_from_env() == overlap.OFF
    _set_env(monkeypatch, {"TPU_MINIMAX_M3_PREFILL_OVERLAP_MIN_TOKENS": "1024"})
    assert overlap.config_from_env() == overlap.OFF  # switch off
    _set_env(monkeypatch, {"TPU_MINIMAX_M3_PREFILL_OVERLAP": "1"})
    assert overlap.config_from_env() == overlap.OverlapConfig(4096, "xla", True)
    _set_env(
        monkeypatch,
        {
            "TPU_MINIMAX_M3_PREFILL_OVERLAP": "1",
            "TPU_MINIMAX_M3_PREFILL_OVERLAP_MIN_TOKENS": "6144",
            "TPU_MINIMAX_M3_PREFILL_OVERLAP_AR": "group",
            "TPU_MINIMAX_M3_PREFILL_OVERLAP_QKV": "0",
        },
    )
    assert overlap.config_from_env() == overlap.OverlapConfig(6144, "group", False)
    _set_env(
        monkeypatch,
        {
            "TPU_MINIMAX_M3_PREFILL_OVERLAP": "1",
            "TPU_MINIMAX_M3_PREFILL_OVERLAP_MIN_TOKENS": "0",
        },
    )
    assert overlap.config_from_env() == overlap.OFF
    _set_env(
        monkeypatch,
        {"TPU_MINIMAX_M3_PREFILL_OVERLAP": "1", "TPU_MINIMAX_M3_PREFILL_OVERLAP_AR": "tc"},
    )
    with pytest.raises(ValueError, match="TPU_MINIMAX_M3_PREFILL_OVERLAP_AR"):
        overlap.config_from_env()


def _fake_model(layers, cfg, aux=(0, 2, 4)):
    fake = types.SimpleNamespace(
        aux_hidden_state_layers=aux,
        layers=layers,
        norm=_Norm(99),
        overlap=cfg,
        tp_size=R,
        index_reuse=False,
    )
    fake._forward_overlap = types.MethodType(mm.MiniMaxM3Model._forward_overlap, fake)
    return fake


@pytest.mark.parametrize("on", [False, True], ids=["off", "on"])
def test_model_forward_overlap_branch(collectives, monkeypatch, on):
    """`MiniMaxM3Model.forward` on an overlap step: the final norm's output
    and the drafter's aux hidden states match the plain path's."""
    cfg = overlap.OverlapConfig(8 if on else 0)
    layers, x = _layers(cfg), _input()
    monkeypatch.setattr(overlap, "step", lambda config, n, tp: config.on)
    fake = _fake_model(layers, cfg)
    positions = torch.zeros(T, dtype=torch.long)
    ids = torch.zeros(T, dtype=torch.long)
    got, aux = mm.MiniMaxM3Model.forward(fake, ids, positions, x)
    plain = _fake_model(layers, overlap.OFF)
    want, want_aux = mm.MiniMaxM3Model.forward(plain, ids, positions, x)
    torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)
    assert len(aux) == len(want_aux) == 3
    for a, b in zip(aux, want_aux):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    assert len(collectives["ar"]) == (16 if on else 0)
    _assert_flags_restored(layers)


def test_model_forward_off_never_calls_overlap(monkeypatch):
    """Key off (the default): the forward never asks `overlap` anything, never
    reads the token count and calls every layer exactly as the base did."""

    def boom(*_a, **_k):
        raise AssertionError("overlap reached with the switch off")

    for name in ("step", "all_reduce", "wait", "split"):
        monkeypatch.setattr(overlap, name, boom)
    seen = []

    class _Layer:
        def __call__(self, positions, hidden, residual, **kw):
            seen.append(kw)
            return hidden, hidden if residual is None else residual

    class _NoShape:
        @property
        def shape(self):
            raise AssertionError("token count read")

    fake = types.SimpleNamespace(
        aux_hidden_state_layers=(),
        layers=[_Layer(), _Layer()],
        norm=lambda h, r: (h + r, None),
        index_reuse=False,
        tp_size=8,
    )  # no `overlap` attribute at all: stand-ins built without __init__
    x = torch.ones(4, 8)
    mm.MiniMaxM3Model.forward(fake, _NoShape(), torch.zeros(4), x)
    fake.overlap = overlap.OFF
    mm.MiniMaxM3Model.forward(fake, _NoShape(), torch.zeros(4), x)
    assert seen == [{}] * 4


def test_all_reduce_routes(monkeypatch):
    from torch.distributed import _functional_collectives as funcol

    calls = []
    group = object()
    monkeypatch.setattr(
        overlap, "get_tp_group", lambda: types.SimpleNamespace(device_group=group)
    )
    monkeypatch.setattr(
        overlap,
        "tensor_model_parallel_all_reduce",
        lambda x: calls.append("group") or x,
    )
    monkeypatch.setattr(
        funcol,
        "all_reduce",
        lambda x, op, g: calls.append(("funcol", op, g is group)) or x,
    )
    x = torch.ones(4, 8)
    overlap.all_reduce(x, "group")
    overlap.all_reduce(x, "xla")
    assert calls == ["group", ("funcol", "sum", True)]
    assert overlap.wait(x) is x


def test_padding_mask_row_offset():
    state = token_padding.TokenPaddingState.create(16, "cpu")
    state.update(10, 16)  # rows 10..15 are padding
    token_padding.set_padding_state(state)
    try:
        assert state.get_local_padding_mask(8).tolist() == [False] * 8
        with token_padding.row_offset(8):
            assert state.get_local_padding_mask(8).tolist() == [False] * 2 + [True] * 6
            ids = torch.zeros(8, 2, dtype=torch.long)
            w = torch.ones(8, 2)
            _, w = token_padding.zero_routing_weights_for_padding(ids, w, True)
            assert w[:, 0].tolist() == [1.0] * 2 + [0.0] * 6
        assert token_padding._row_offset is None
        assert state.get_local_padding_mask(8).tolist() == [False] * 8
    finally:
        token_padding.set_padding_state(None)


def test_attention_attend_with_and_without_qkv(monkeypatch):
    """`forward` = o_proj(attend(...)); `attend(qkv=...)` skips qkv_proj and
    gives the same output (memory-profiling branch, no caches bound)."""
    from vllm_torchtpu.models.minimax_m3 import attention as attn_mod

    heads, dim = 2, 8
    a = object.__new__(attn_mod.MiniMaxM3SparseAttention)
    nn.Module.__init__(a)
    g = torch.Generator().manual_seed(3)
    wq = torch.randn(16, heads * dim, generator=g)
    wo = torch.randn(heads * dim, 16, generator=g)
    qkv_calls = []

    def qkv_proj(x):
        qkv_calls.append(x.shape[0])
        return x @ wq, None

    a.qkv_proj = qkv_proj
    a.o_proj = lambda x: (x @ wo, None)
    a.num_heads, a.num_kv_heads, a.head_dim = heads, 1, dim
    a.q_size = heads * dim
    a.q_head_major = False
    a.q_norm = a.k_norm = a.index_q_norm = a.index_k_norm = None
    a.rotary_emb = a.norm_rope_op = None
    a.kv_cache = types.SimpleNamespace(kv_cache=torch.empty(0))
    monkeypatch.setattr(
        attn_mod, "norm_rope_heads", lambda qkv, *rest: (qkv * 2, None, None, None, None)
    )
    monkeypatch.setattr(
        attn_mod,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=None),
    )
    x = torch.randn(6, 16, generator=g)
    out = a(None, x)
    torch.testing.assert_close(out, (x @ wq * 2) @ wo)
    torch.testing.assert_close(a.attend(None, x), x @ wq * 2)
    torch.testing.assert_close(a.attend(None, None, qkv=x @ wq), x @ wq * 2)
    assert qkv_calls == [6, 6]
