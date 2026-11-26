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
"""MiniMax-M3 sequence parallelism (TPU_MINIMAX_M3_SP_*), on CPU.

The TP ranks run as threads of one process; vLLM's TP collectives (and the
all-to-all of the o_proj path) are swapped for thread-barrier exchanges.
Each rank runs the real `MiniMaxM3Model.forward` / decoder layer / MLP / MoE
code over stand-in attention, linears and experts that return the rank's
partial sum (its weight shard), all-reduced unless the layer took the
reduction over. The torch_tpu jax_ops run their JAX bodies on CPU (the
fused SP norm kernel in interpret mode). Every SP variant must give the
plain path's result: up to summation order without e4m3, within e4m3
rounding with it. With every key off the forward makes the old calls.
"""

import dataclasses
import threading
import types
import zlib

import jax.numpy as jnp
import numpy as np
import pytest
import torch
from torch import nn

dataclasses_replace = dataclasses.replace

pytest.importorskip("torch_tpu")

from vllm_torchtpu import envs  # noqa: E402
from vllm_torchtpu.distributed import pallas_all_reduce  # noqa: E402
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import attention as attn_mod  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import model as mm  # noqa: E402
from vllm_torchtpu.models.minimax_m3 import quant, sp_o_proj  # noqa: E402
from vllm_torchtpu.models.minimax_m3.kernels import sp_norm  # noqa: E402
from vllm_torchtpu.models.minimax_m3.moe import MiniMaxM3MLP, MiniMaxM3MoE  # noqa: E402

TP, T, H, K, I = 4, 64, 128, 32, 32  # K, I: per-rank attention / FFN width
EPS = 1e-6
E4M3 = torch.float8_e4m3fn

_SP_ENV = {
    "TPU_MINIMAX_M3_SP_MODE": "off",
    "TPU_MINIMAX_M3_SP_MIN_TOKENS": 4096,
    "TPU_MINIMAX_M3_SP_FP8_GATHER": False,
    "TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER": False,
    "TPU_MINIMAX_M3_SP_FUSED_NORM": False,
    "TPU_MINIMAX_M3_SP_E4M3_QKV": False,
    "TPU_MINIMAX_M3_SP_A2A_OPROJ": "off",
    "TPU_MINIMAX_M3_SP_A2A_E4M3": False,
    "TPU_MINIMAX_M3_SP_QRS": False,
    "TPU_MINIMAX_M3_SP_QRS_MIN_TOKENS": 2048,
    "TPU_MINIMAX_M3_SP_HIER_RS": False,
    "TPU_MINIMAX_M3_SP_HIER_RS_MIN_TOKENS": 8192,
}
# The production config this port targets.
_TARGET = dict(
    TPU_MINIMAX_M3_SP_MODE="full",
    TPU_MINIMAX_M3_SP_MIN_TOKENS=T,
    TPU_MINIMAX_M3_SP_FP8_GATHER=True,
    TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER=True,
    TPU_MINIMAX_M3_SP_FUSED_NORM=True,
    TPU_MINIMAX_M3_SP_E4M3_QKV=True,
    TPU_MINIMAX_M3_SP_A2A_OPROJ="gather",
    TPU_MINIMAX_M3_SP_A2A_E4M3=True,
)


def _set_env(monkeypatch, **kw):
    for key, default in _SP_ENV.items():
        monkeypatch.setattr(envs, key, kw.get(key, default), raising=False)


# ------------------------------------------------------------ JAX on CPU
_JNP = {torch.bfloat16: jnp.bfloat16, E4M3: jnp.float8_e4m3fn, torch.float32: jnp.float32}
_TORCH = {np.dtype(v): k for k, v in _JNP.items()}


def _jnp(t: torch.Tensor):
    return jnp.asarray(t.float().numpy()).astype(_JNP[t.dtype])


def _torch(a) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(_TORCH[np.dtype(a.dtype)])


def _jax_call(fn, *ts):
    out = fn(*[_jnp(t) for t in ts])
    if isinstance(out, (tuple, list)):
        return tuple(_torch(o) for o in out)
    return _torch(out)


# ------------------------------------------------------- simulated TP group
class _TP:
    """TP collectives among the threads of `run`, rank = thread's index."""

    def __init__(self, n: int) -> None:
        self.n = n
        self.barrier = threading.Barrier(n, timeout=120)
        self.slots: list = [None] * n
        self.local = threading.local()
        self.calls: list[str] = []

    @property
    def rank(self) -> int:
        return self.local.rank

    def _exchange(self, x):
        self.slots[self.rank] = x
        self.barrier.wait()
        seen = list(self.slots)
        self.barrier.wait()
        return seen

    def _log(self, name):
        if self.rank == 0:
            self.calls.append(name)

    def all_gather(self, x, dim=0):
        self._log(f"ag:{x.dtype}")
        return torch.cat(self._exchange(x), dim)

    def reduce_scatter(self, x, dim=0):
        self._log("rs")
        assert dim == 0
        total = sum(self._exchange(x))
        return total.chunk(self.n, 0)[self.rank].clone()

    def all_reduce(self, x):
        self._log("ar")
        return sum(self._exchange(x))

    def all_to_all_rows(self, x):
        self._log(f"a2a:{x.dtype}")
        parts = self._exchange(x)
        return torch.cat([p.chunk(self.n, 0)[self.rank] for p in parts], 0)

    def group(self):
        return types.SimpleNamespace(
            world_size=self.n, rank_in_group=self.rank, ranks=list(range(self.n))
        )

    def run(self, fn):
        out: list = [None] * self.n
        errors: list = []

        def body(r):
            self.local.rank = r
            try:
                out[r] = fn(r)
            except BaseException as e:  # noqa: BLE001 - reraised below
                errors.append(e)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return out


@pytest.fixture
def tp(monkeypatch):
    return _install_tp(monkeypatch)


def _install_tp(monkeypatch):
    g = _TP(TP)
    monkeypatch.setattr(mm, "tensor_model_parallel_reduce_scatter", g.reduce_scatter)
    monkeypatch.setattr(mm, "tensor_model_parallel_all_gather", g.all_gather)
    monkeypatch.setattr(mm, "tensor_model_parallel_all_reduce", g.all_reduce)
    monkeypatch.setattr(sp_o_proj, "tensor_model_parallel_all_gather", g.all_gather)
    monkeypatch.setattr(sp_o_proj, "all_to_all_rows", g.all_to_all_rows)
    monkeypatch.setattr(sp_o_proj, "get_tp_group", g.group)
    monkeypatch.setattr(
        pallas_all_reduce,
        "armed_rank",
        lambda ranks: torch.tensor([[g.rank]], dtype=torch.int32),
    )
    # The jax_ops' bodies on CPU.
    monkeypatch.setattr(
        sp_o_proj, "_quantize_op", lambda x: _jax_call(sp_o_proj._quantize_rows_pow2, x)
    )
    monkeypatch.setattr(
        sp_o_proj, "_o_proj_e4m3_op", lambda *a: _jax_call(sp_o_proj._o_proj_e4m3, *a)
    )
    monkeypatch.setattr(quant, "_e4m3_matmul", lambda *a: _jax_call(quant._e4m3_rows_matmul, *a))
    return g


def _fused_norm_op(shard, update, weight):
    fn = lambda s, u, w: sp_norm.add_norm_quant(s, u, w, eps=EPS, interpret=True)  # noqa: E731
    return _jax_call(fn, shard, update, weight)


# -------------------------------------------------------------- stand-ins
def _gen(*seed):
    return torch.Generator().manual_seed(zlib.crc32(repr(seed).encode()))


def _randn(shape, seed, scale=1.0):
    return torch.randn(shape, generator=_gen(*seed)) * scale


class _Norm(nn.Module):
    """GemmaRMSNorm's interface: norm(x) or fused add + norm(x, residual)."""

    def __init__(self, seed):
        super().__init__()
        self.weight = _randn(H, seed, 0.1).to(torch.bfloat16)

    def _norm(self, x):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        return (xf * torch.rsqrt(var + EPS) * (1.0 + self.weight.float())).to(x.dtype)

    def forward(self, x, residual=None):
        if residual is None:
            return self._norm(x)
        x = x + residual
        return self._norm(x), x


class _Linear:
    """An e4m3 bias-free linear on the MXFP8 method (rank's column shard)."""

    def __init__(self, k, n, seed, reduce_results=None, tp=None):
        self.weight = _randn((k, n), seed, 1.0).to(E4M3)
        self.weight_scale = torch.full((n,), 2.0**-5, dtype=torch.bfloat16)
        self.quant_method = quant.MiniMaxM3Mxfp8LinearMethod(True)
        self.reduce_results = reduce_results
        self.tp = tp
        self.inputs: list = []

    def __call__(self, x):
        self.inputs.append(x)
        out = self.quant_method.apply(self, x)
        if self.reduce_results:
            out = self.tp.all_reduce(out)
        return out, None


class _Attn:
    """Rank's heads: a causal running mean of tanh(x A_r) over the tokens
    (attention sees every row), then o_proj. Takes e4m3 rows dequantized."""

    def __init__(self, seed, rank, tp):
        self.a = _randn((H, K), (*seed, rank), H**-0.5).to(torch.bfloat16)
        # o_proj blocks share one per-channel scale (sp_o_proj.prepare).
        self.o_proj = _Linear(K, H, (*seed, rank, "o"), reduce_results=True, tp=tp)
        self.calls: list = []

    def __call__(self, positions, x, index_reuse=None, project=True):
        self.calls.append((type(x).__name__, index_reuse, project))
        if isinstance(x, tuple):
            x = (x[0].float() * x[1]).to(torch.bfloat16)
        h = torch.tanh((x @ self.a).float()).cumsum(0)
        h = (h / torch.arange(1, h.shape[0] + 1).unsqueeze(1)).to(x.dtype)
        if not project:
            return h
        return self.o_proj(h)[0]


def _mlp(seed, rank, tp):
    mlp = object.__new__(MiniMaxM3MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = _Linear(H, 2 * I, (*seed, rank, "gu"))
    mlp.down_proj = _Linear(I, H, (*seed, rank, "d"), reduce_results=True, tp=tp)
    mlp.act_fn = lambda gu: (torch.nn.functional.silu(gu[:, :I].float()) * gu[:, I:].float()).to(gu.dtype)
    return mlp


class _Experts:
    """Routed + shared experts' rank partial, all-reduced unless skipped."""

    def __init__(self, seed, rank, tp):
        self.a = _randn((H, I), (*seed, rank, "ea"), H**-0.5).to(torch.bfloat16)
        self.b = _randn((I, H), (*seed, rank, "eb"), I**-0.5).to(torch.bfloat16)
        self.moe_config = types.SimpleNamespace(skip_final_all_reduce=False)
        self.tp = tp
        self.inputs: list = []

    def __call__(self, hidden_states, router_logits):
        self.inputs.append(hidden_states)
        gate = torch.sigmoid(router_logits).mean(-1, keepdim=True)
        out = ((torch.tanh(hidden_states @ self.a).float() * gate).to(torch.bfloat16)) @ self.b
        if not self.moe_config.skip_final_all_reduce:
            out = self.tp.all_reduce(out)
        return out


def _moe(seed, rank, tp):
    moe = object.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    g = _randn((H, 8), (*seed, "router"), H**-0.5)
    moe.gate = lambda x: (x.float() @ g, None)
    moe.experts = _Experts(seed, rank, tp)
    return moe


def _layer(idx, rank, tp, fused):
    layer = object.__new__(mm.MiniMaxM3DecoderLayer)
    nn.Module.__init__(layer)
    layer.is_moe_layer = bool(idx % 2)
    layer.tp_size = TP
    layer.self_attn = _Attn((idx, "attn"), rank, tp)
    if layer.is_moe_layer:
        layer.block_sparse_moe = _moe((idx, "moe"), rank, tp)
    else:
        layer.mlp = _mlp((idx, "mlp"), rank, tp)
    layer.input_layernorm = _Norm((idx, "in"))
    layer.post_attention_layernorm = _Norm((idx, "post"))
    if fused:
        layer._add_norm_quant = _fused_norm_op
    if envs.TPU_MINIMAX_M3_SP_A2A_OPROJ != "off":
        setattr(layer.self_attn.o_proj, sp_o_proj.PREPARED_ATTR, True)
    return layer


def _model(rank, tp, sp_mode, aux=()):
    model = object.__new__(mm.MiniMaxM3Model)
    nn.Module.__init__(model)
    fused = envs.TPU_MINIMAX_M3_SP_FUSED_NORM
    model.layers = nn.ModuleList(_layer(i, rank, tp, fused) for i in range(4))
    model.norm = _Norm(("final",))
    model.aux_hidden_state_layers = tuple(aux)
    model.tp_size = TP
    model.sp_mode = sp_mode
    model.index_reuse = False
    if envs.TPU_MINIMAX_M3_SP_A2A_OPROJ == "replicate":
        for layer in model.layers:
            o = layer.self_attn.o_proj
            # Every rank's block, rotated so this rank's comes first (prepare).
            blocks = [
                _randn((K, H), (layer_idx_of(model, layer), "attn", r, "o"), 1.0).to(E4M3)
                for r in range(TP)
            ]
            o.weight = torch.roll(torch.stack(blocks), -rank, dims=0)
    return model


def layer_idx_of(model, layer):
    return list(model.layers).index(layer)


def _input():
    return _randn((T, H), ("x",), 1.0).to(torch.bfloat16)


def _forward(tp, monkeypatch, sp_mode, aux=(), sp=True):
    monkeypatch.setattr(mm, "_sp_step", lambda n, size: sp and n >= envs.TPU_MINIMAX_M3_SP_MIN_TOKENS and n % size == 0)
    x = _input()
    positions = torch.arange(T)
    models = [None] * TP

    def run(r):
        models[r] = _model(r, tp, sp_mode, aux)
        return mm.MiniMaxM3Model.forward(models[r], None, positions, inputs_embeds=x)

    return tp.run(run), models


def _rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def _plain(tp, monkeypatch):
    _set_env(monkeypatch)
    outs, _ = _forward(tp, monkeypatch, "off", aux=(0, 2, 4))
    return outs[0]


def _flags_restored(models):
    for model in models:
        for layer in model.layers:
            assert layer.self_attn.o_proj.reduce_results is True
            if layer.is_moe_layer:
                assert not layer.block_sparse_moe.experts.moe_config.skip_final_all_reduce
            else:
                assert layer.mlp.down_proj.reduce_results is True


# Max rel L2 to the plain path (bf16 sums in another order: ~0.7%; e4m3
# rows: ~2%; measured).
_VARIANTS = {
    "attn": (dict(TPU_MINIMAX_M3_SP_MODE="attn"), 1.5e-2),
    "attn-fp8": (dict(TPU_MINIMAX_M3_SP_MODE="attn", TPU_MINIMAX_M3_SP_FP8_GATHER=True), 4e-2),
    "full": (dict(TPU_MINIMAX_M3_SP_MODE="full"), 1.5e-2),
    "full-fp8": (
        dict(
            TPU_MINIMAX_M3_SP_MODE="full",
            TPU_MINIMAX_M3_SP_FP8_GATHER=True,
            TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER=True,
        ),
        4e-2,
    ),
    "full-fused": (
        dict(
            TPU_MINIMAX_M3_SP_MODE="full",
            TPU_MINIMAX_M3_SP_FP8_GATHER=True,
            TPU_MINIMAX_M3_SP_FP8_ATTN_GATHER=True,
            TPU_MINIMAX_M3_SP_FUSED_NORM=True,
            TPU_MINIMAX_M3_SP_E4M3_QKV=True,
        ),
        4e-2,
    ),
    "a2a-gather": (dict(TPU_MINIMAX_M3_SP_MODE="full", TPU_MINIMAX_M3_SP_A2A_OPROJ="gather"), 1.5e-2),
    "a2a-replicate-e4m3": (
        dict(
            TPU_MINIMAX_M3_SP_MODE="full",
            TPU_MINIMAX_M3_SP_A2A_OPROJ="replicate",
            TPU_MINIMAX_M3_SP_A2A_E4M3=True,
        ),
        4e-2,
    ),
    "target": (_TARGET, 4e-2),
}


@pytest.mark.parametrize("name", list(_VARIANTS))
def test_sp_matches_plain(tp, monkeypatch, name):
    want = _plain(tp, monkeypatch)
    kw, tol = _VARIANTS[name]
    kw = {"TPU_MINIMAX_M3_SP_MIN_TOKENS": T, **kw}
    _set_env(monkeypatch, **kw)
    tp.calls.clear()
    outs, models = _forward(tp, monkeypatch, kw["TPU_MINIMAX_M3_SP_MODE"], aux=(0, 2, 4))
    # Every rank ends with the same full hidden states and aux entries.
    for got in outs[1:]:
        torch.testing.assert_close(got[0], outs[0][0], rtol=0, atol=0)
    hidden, aux = outs[0]
    assert hidden.shape == (T, H) and len(aux) == 3
    assert _rel(hidden, want[0]) < tol, _rel(hidden, want[0])
    for got_aux, want_aux in zip(aux, want[1]):
        assert _rel(got_aux, want_aux) < tol
    _flags_restored(models)
    # "full" runs no all-reduce; "attn" one per layer (FFN + attention).
    if kw["TPU_MINIMAX_M3_SP_MODE"] == "full":
        assert "ar" not in tp.calls and "rs" in tp.calls
    else:
        assert tp.calls.count("ar") == 4 and tp.calls.count("rs") == 4
    if kw.get("TPU_MINIMAX_M3_SP_A2A_OPROJ", "off") != "off":
        dtype = "float8_e4m3fn" if kw.get("TPU_MINIMAX_M3_SP_A2A_E4M3") else "bfloat16"
        assert sum(c == f"a2a:torch.{dtype}" for c in tp.calls) >= 4
    if kw.get("TPU_MINIMAX_M3_SP_FP8_GATHER"):
        assert any(c == "ag:torch.float8_e4m3fn" for c in tp.calls)


def test_target_takes_every_switch(tp, monkeypatch):
    """The production config: fused norm, e4m3 qkv / gate_up rows and the
    e4m3 all-to-all o_proj each actually run."""
    _set_env(monkeypatch, **_TARGET)
    calls = {"fused": 0, "e4m3_mm": 0, "o_proj_e4m3": 0}

    def count(key, fn):
        def wrapped(*a):
            if tp.rank == 0:
                calls[key] += 1
            return fn(*a)

        return wrapped

    monkeypatch.setitem(globals(), "_fused_norm_op", count("fused", _fused_norm_op))
    monkeypatch.setattr(quant, "_e4m3_matmul", count("e4m3_mm", quant._e4m3_matmul))
    monkeypatch.setattr(sp_o_proj, "_o_proj_e4m3_op", count("o_proj_e4m3", sp_o_proj._o_proj_e4m3_op))
    outs, models = _forward(tp, monkeypatch, "full")
    # 4 layers; layer 0's attention input is the full embedding (no gather).
    assert calls["o_proj_e4m3"] == 4
    # Fused passes: before every gather with a pending update; layer 0's two
    # norms have none (its attention output is added into the shard).
    assert calls["fused"] == 6
    # e4m3 x e4m3 qkv... only via apply_e4m3_rows: the dense MLPs' gate_up
    # (layers 0 and 2); the stand-in attention dequantizes.
    assert calls["e4m3_mm"] == 2
    attn_inputs = [c[0] for c in models[0].layers[1].self_attn.calls]
    assert attn_inputs == ["tuple"]
    assert all(c[2] is False for m in models for layer in m.layers for c in layer.self_attn.calls)


def _qrs_emulation(tp):
    """Stand-in for the armed int8 reduce-scatter op: the exact chunk sum
    rounded to int8 with one absmax scale per lane (logged as "qrs")."""

    def reduce_scatter(ranks, x):
        assert list(ranks) == list(range(tp.n)) and x.dim() == 2
        if tp.rank == 0:
            tp.calls.append("qrs")
        total = sum(tp._exchange(x.float())).chunk(tp.n, 0)[tp.rank]
        scale = total.abs().amax(0, keepdim=True).clamp(min=1e-30) / 127
        return ((total / scale).round().clamp(-127, 127) * scale).to(x.dtype)

    return reduce_scatter


@pytest.mark.parametrize("mode", ["full", "attn"])
def test_qrs_takes_every_sp_reduce_scatter(tp, monkeypatch, mode):
    """TPU_MINIMAX_M3_SP_QRS: every SP reduce-scatter goes through the armed
    op (none through XLA's), and the result stays within int8 rounding of
    the plain path."""
    want = _plain(tp, monkeypatch)
    kw = dict(_TARGET, TPU_MINIMAX_M3_SP_MODE=mode, TPU_MINIMAX_M3_SP_QRS=True)
    if mode == "attn":
        kw["TPU_MINIMAX_M3_SP_A2A_OPROJ"] = "off"
    _set_env(monkeypatch, **kw)
    monkeypatch.setattr(mm, "get_tp_group", tp.group)
    monkeypatch.setattr(pallas_all_reduce, "reduce_scatter", _qrs_emulation(tp))
    tp.calls.clear()
    outs, _ = _forward(tp, monkeypatch, mode, aux=(0, 2, 4))
    for got in outs[1:]:
        torch.testing.assert_close(got[0], outs[0][0], rtol=0, atol=0)
    assert "rs" not in tp.calls
    # full + a2a o_proj: one (FFN) per layer; attn: one (attention) per layer.
    assert tp.calls.count("qrs") == 4
    assert _rel(outs[0][0], want[0]) < 6e-2, _rel(outs[0][0], want[0])


def test_qrs_unarmed_falls_back(tp, monkeypatch):
    """TPU_MINIMAX_M3_SP_QRS on but the op not armed (None): XLA's reduce-scatter."""
    _set_env(monkeypatch, **dict(_TARGET, TPU_MINIMAX_M3_SP_QRS=True))
    monkeypatch.setattr(mm, "get_tp_group", tp.group)
    monkeypatch.setattr(pallas_all_reduce, "reduce_scatter", lambda ranks, x: None)
    tp.calls.clear()
    _forward(tp, monkeypatch, "full")
    assert tp.calls.count("rs") == 4


def test_qrs_key_off_never_touches_op(tp, monkeypatch):
    _set_env(monkeypatch, **_TARGET)
    monkeypatch.setattr(pallas_all_reduce, "reduce_scatter", _boom)
    monkeypatch.setattr(mm, "get_tp_group", _boom)
    tp.calls.clear()
    _forward(tp, monkeypatch, "full")
    assert tp.calls.count("rs") == 4


def _exact_rs(tp):
    """Stand-in for the armed bf16 hierrs_tc op: the f32 chunk sum (logged as "hier")."""

    def reduce_scatter(ranks, x):
        assert list(ranks) == list(range(tp.n)) and x.dim() == 2
        if tp.rank == 0:
            tp.calls.append("hier")
        return sum(tp._exchange(x.float())).chunk(tp.n, 0)[tp.rank].to(x.dtype)

    return reduce_scatter


@pytest.mark.parametrize("mode", ["full", "attn"])
def test_hier_rs_takes_every_sp_reduce_scatter(tp, monkeypatch, mode):
    """TPU_MINIMAX_M3_SP_HIER_RS: every SP reduce-scatter goes through the
    armed op (none through XLA's); with an exact-sum op the result matches
    the plain path to bf16 rounding."""
    want = _plain(tp, monkeypatch)
    kw = dict(_TARGET, TPU_MINIMAX_M3_SP_MODE=mode, TPU_MINIMAX_M3_SP_HIER_RS=True)
    if mode == "attn":
        kw["TPU_MINIMAX_M3_SP_A2A_OPROJ"] = "off"
    _set_env(monkeypatch, **kw)
    monkeypatch.setattr(mm, "get_tp_group", tp.group)
    monkeypatch.setattr(pallas_all_reduce, "reduce_scatter", _exact_rs(tp))
    tp.calls.clear()
    outs, _ = _forward(tp, monkeypatch, mode, aux=(0, 2, 4))
    for got in outs[1:]:
        torch.testing.assert_close(got[0], outs[0][0], rtol=0, atol=0)
    assert "rs" not in tp.calls
    assert tp.calls.count("hier") == 4
    assert _rel(outs[0][0], want[0]) < 6e-2, _rel(outs[0][0], want[0])


def test_hier_rs_key_off_never_touches_op(tp, monkeypatch):
    _set_env(monkeypatch, **dict(_TARGET, TPU_MINIMAX_M3_SP_HIER_RS=False))
    monkeypatch.setattr(pallas_all_reduce, "reduce_scatter", _boom)
    tp.calls.clear()
    _forward(tp, monkeypatch, "full")
    assert tp.calls.count("rs") == 4


def test_hier_applies():
    ha = pallas_all_reduce.hier_applies
    assert ha(8192, 6144, 1024) and ha(1024, 6144, 1024) and ha(3072, 6144, 1024)
    assert not ha(512, 6144, 1024)  # below the minimum
    assert not ha(8192, 6144, 0)  # off
    assert not ha(1028, 6144, 8)  # rows not a multiple of 8 ranks


def test_chip_perm_rank_mesh():
    """Served rank mesh (device ids 0,1,4,5,6,7,2,3 on a 2x2 grid, chip =
    coords, id = 2 * chip + core in x-major order): chips (0,0),(0,1),(1,1),
    (1,0) at mesh chips 0..3 -> physical chip k (coords sorted) is on mesh chip
    perm[k]."""
    cc = pallas_all_reduce._ChipCore
    coords = {0: (0, 0, 0), 1: (0, 0, 0), 2: (1, 0, 0), 3: (1, 0, 0),
              4: (0, 1, 0), 5: (0, 1, 0), 6: (1, 1, 0), 7: (1, 1, 0)}
    mk = lambda ids: [cc(i, coords[i], i % 2) for i in ids]  # noqa: E731
    # ids in order: mesh chips (0,0) (1,0) (0,1) (1,1)
    assert pallas_all_reduce._chip_perm(mk(range(8))) == (0, 2, 1, 3)
    assert pallas_all_reduce._chip_perm(mk((0, 1, 4, 5, 2, 3, 6, 7))) is None
    # sorted chips: (0,0,0) (0,1,0) (1,0,0) (1,1,0); mesh chips: (0,0) (0,1) (1,1) (1,0)
    assert pallas_all_reduce._chip_perm(mk((0, 1, 4, 5, 6, 7, 2, 3))) == (0, 1, 3, 2)
    with pytest.raises(ValueError):
        pallas_all_reduce._chip_perm(mk((0, 2, 1, 3, 4, 5, 6, 7)))


def test_pallas_reduce_scatter_gate(monkeypatch):
    rank = torch.tensor([[1]], dtype=torch.int32)
    seen = []
    rs_op = lambda x, r: seen.append((tuple(x.shape), r)) or x[: x.shape[0] // 4]  # noqa: E731
    x = torch.zeros(64, 8, dtype=torch.bfloat16)
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", None)
    assert pallas_all_reduce.reduce_scatter([0, 1, 2, 3], x) is None
    armed = pallas_all_reduce._Armed(ranks=(0, 1, 2, 3), op=None, rank=rank, rank_ordered=True)
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", armed)
    assert pallas_all_reduce.reduce_scatter([0, 1, 2, 3], x) is None  # no rs_op
    armed = dataclasses_replace(armed, rs_op=rs_op)
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", armed)
    assert pallas_all_reduce.reduce_scatter([0, 1], x) is None
    assert pallas_all_reduce.reduce_scatter([0, 1, 2, 3], x.float()) is None
    assert pallas_all_reduce.reduce_scatter([0, 1, 2, 3], x[None]) is None
    assert pallas_all_reduce.reduce_scatter([0, 1, 2, 3], x).shape == (16, 8)
    assert seen == [((64, 8), rank)]


def test_sp_step_off_below_min_tokens_or_ragged(tp, monkeypatch):
    """A step below MIN_TOKENS, or not a multiple of TP, runs the plain path."""
    _set_env(monkeypatch, TPU_MINIMAX_M3_SP_MODE="full", TPU_MINIMAX_M3_SP_MIN_TOKENS=T + 1)
    tp.calls.clear()
    _forward(tp, monkeypatch, "full")
    assert tp.calls.count("ar") == 8 and "rs" not in tp.calls


def test_sp_step_gate(monkeypatch):
    """Only inside a warmup trace (compiling + shape_variants' AOT tag), at
    least MIN_TOKENS tokens and a multiple of TP."""
    _set_env(monkeypatch, TPU_MINIMAX_M3_SP_MODE="full", TPU_MINIMAX_M3_SP_MIN_TOKENS=1024)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    monkeypatch.setattr(mm.shape_variants, "aot_cache_tag", lambda: "tag")
    assert mm._sp_step(1024, 8) and mm._sp_step(4096, 8)
    assert not mm._sp_step(1016, 8) and not mm._sp_step(1028, 8)
    monkeypatch.setattr(mm.shape_variants, "aot_cache_tag", lambda: None)
    assert not mm._sp_step(4096, 8)
    monkeypatch.setattr(mm.shape_variants, "aot_cache_tag", lambda: "tag")
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    assert not mm._sp_step(4096, 8)


def test_sp_mode_resolution(monkeypatch):
    _set_env(monkeypatch)
    assert mm._sp_mode() == "off"
    _set_env(monkeypatch, TPU_MINIMAX_M3_SP_MODE="full", TPU_MINIMAX_M3_SP_MIN_TOKENS=0)
    assert mm._sp_mode() == "off"
    _set_env(monkeypatch, TPU_MINIMAX_M3_SP_MODE="full", TPU_MINIMAX_M3_SP_MIN_TOKENS=1024)
    assert mm._sp_mode() == "full"


def test_fits_sp_norm():
    assert mm._fits_sp_norm(torch.empty(64, 8))
    assert mm._fits_sp_norm(torch.empty(1024, 8))
    assert not mm._fits_sp_norm(torch.empty(192, 8))


# ------------------------------------------------------------- key off
def test_env_defaults_off(monkeypatch):
    for key, default in _SP_ENV.items():
        monkeypatch.delenv(key, raising=False)
        assert envs.environment_variables[key]() == default, key


def _boom(*a, **k):
    raise AssertionError("SP code touched with every key off")


def test_key_off_forward_makes_plain_calls(monkeypatch):
    """Every key off: the model loop calls each layer as before (no kwargs,
    or the index-reuse dict), the layers keep their own reductions and no SP
    helper runs."""
    _set_env(monkeypatch)
    for name in ("_sp_step", "_all_gather_rows", "_quantize_rows", "_gather_rows_e4m3", "_fits_sp_norm"):
        monkeypatch.setattr(mm, name, _boom)
    for name in ("forward_sp", "_forward_sp", "_attn_rs", "_norm_gather", "_layer_owned_reductions"):
        monkeypatch.setattr(mm.MiniMaxM3DecoderLayer, name, _boom)
    for name in ("sp_o_proj", "applies", "local_rows", "all_to_all_rows", "prepare"):
        monkeypatch.setattr(sp_o_proj, name, _boom)
    monkeypatch.setattr(mm, "tensor_model_parallel_reduce_scatter", _boom)
    monkeypatch.setattr(mm, "tensor_model_parallel_all_gather", _boom)

    class _Rec(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, *args, **kwargs):
            self.calls.append((len(args), dict(kwargs)))
            return args[1] + 1, args[1]

    for index_reuse in (False, True):
        model = object.__new__(mm.MiniMaxM3Model)
        nn.Module.__init__(model)
        model.layers = nn.ModuleList(_Rec() for _ in range(3))
        model.norm = _Norm(("final",))
        model.aux_hidden_state_layers = ()
        model.tp_size = TP
        model.sp_mode = mm._sp_mode()
        model.index_reuse = index_reuse
        out = mm.MiniMaxM3Model.forward(model, None, torch.arange(T), inputs_embeds=_input())
        assert out.shape == (T, H)
        want = (3, {"index_reuse": {}} if index_reuse else {})
        assert all(layer.calls == [want] for layer in model.layers)


def test_key_off_layer_calls(monkeypatch):
    """The decoder layer, MLP and MoE with tensor inputs make the old calls."""
    _set_env(monkeypatch)
    monkeypatch.setattr(mm.MiniMaxM3DecoderLayer, "_layer_owned_reductions", _boom)
    monkeypatch.setattr(mm.MiniMaxM3DecoderLayer, "_attn_rs", _boom)
    tp = _TP(1)
    tp.local.rank = 0
    for idx in (0, 1):
        layer = _layer(idx, 0, tp, fused=False)
        layer.self_attn.o_proj.reduce_results = False
        if layer.is_moe_layer:
            layer.block_sparse_moe.experts.moe_config.skip_final_all_reduce = True
        else:
            layer.mlp.down_proj.reduce_results = False
        x = _input()
        layer(torch.arange(T), x, None)
        assert layer.self_attn.calls == [("Tensor", None, True)]
        layer(torch.arange(T), x, x, {"shared": 1})
        assert layer.self_attn.calls[-1] == ("Tensor", {"shared": 1}, True)
        if layer.is_moe_layer:
            assert all(isinstance(i, torch.Tensor) for i in layer.block_sparse_moe.experts.inputs)
        else:
            assert all(isinstance(i, torch.Tensor) for i in layer.mlp.gate_up_proj.inputs)


def test_key_off_mark_and_local_weight(monkeypatch):
    _set_env(monkeypatch)
    o = nn.Linear(4, 4)
    sp_o_proj.mark(o)
    assert not hasattr(o, sp_o_proj.PREPARE_ATTR)
    assert not sp_o_proj.applies(o)
    assert sp_o_proj.local_weight(o) is o.weight
    _set_env(monkeypatch, TPU_MINIMAX_M3_SP_A2A_OPROJ="gather")
    sp_o_proj.mark(o)
    assert getattr(o, sp_o_proj.PREPARE_ATTR)


def test_key_off_linear_apply_unchanged(monkeypatch):
    """With a 2-D weight, MXFP8 `apply` reads the same weight tensor as before."""
    _set_env(monkeypatch)
    lin = _Linear(H, 64, ("lin",))
    x = _input()
    want = torch.matmul(x, lin.weight.to(x.dtype)) * lin.weight_scale
    torch.testing.assert_close(lin.quant_method.apply(lin, x), want, rtol=0, atol=0)


# ---------------------------------------------------- e4m3 row inputs
class _Stop(Exception):
    pass


def test_attention_e4m3_rows_input(monkeypatch):
    """A tuple input reaches qkv_proj's apply_e4m3_rows; a tensor input (and
    a linear without it) the linear's forward."""
    seen = []

    def stop(qkv, *a):
        seen.append(qkv)
        raise _Stop

    monkeypatch.setattr(attn_mod, "norm_rope_heads", stop)
    monkeypatch.setattr(quant, "_e4m3_matmul", lambda *a: _jax_call(quant._e4m3_rows_matmul, *a))
    attn = object.__new__(attn_mod.MiniMaxM3SparseAttention)
    nn.Module.__init__(attn)
    attn.qkv_proj = _Linear(H, 96, ("qkv",))
    for name in ("num_heads", "num_kv_heads", "head_dim", "rotary_emb", "norm_rope_op"):
        object.__setattr__(attn, name, None)
    for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        object.__setattr__(attn, name, None)
    x = _input()
    q8, scale = mm._quantize_rows(x)
    for hidden in (x, (q8, scale)):
        with pytest.raises(_Stop):
            attn.forward(torch.arange(T), hidden)
    assert len(attn.qkv_proj.inputs) == 1 and attn.qkv_proj.inputs[0] is x
    deq = (q8.float() * scale).to(torch.bfloat16)
    want = attn.qkv_proj.quant_method.apply(attn.qkv_proj, deq)
    assert _rel(seen[1], want) < 1e-2
    # Without apply_e4m3_rows: dequantized rows through the linear's forward.
    attn.qkv_proj.quant_method = types.SimpleNamespace(
        apply=quant.MiniMaxM3Mxfp8LinearMethod(True).apply
    )
    with pytest.raises(_Stop):
        attn.forward(torch.arange(T), (q8, scale))
    torch.testing.assert_close(attn.qkv_proj.inputs[-1], deq)


def test_moe_dequantizes_e4m3_rows():
    tp = _TP(1)
    tp.local.rank = 0
    moe = _moe(("m",), 0, tp)
    x = _input()
    q8, scale = mm._quantize_rows(x)
    out = moe((q8, scale))
    torch.testing.assert_close(moe.experts.inputs[-1], (q8.float() * scale).to(torch.bfloat16))
    assert _rel(out, moe(x)) < 8e-2


# -------------------------------------------------------------- prepare
def test_prepare_rescales_onto_common_scale(monkeypatch):
    """Rank shards with different per-channel power-of-two scales: the
    weight times the common (max) scale equals the original product (no
    underflow here); "replicate" keeps every rank's block, own block first."""
    rng = np.random.default_rng(3)
    n = 16
    scales = [torch.tensor(2.0 ** rng.integers(-8, -4, n), dtype=torch.bfloat16) for _ in range(TP)]
    weights = [torch.tensor(rng.integers(-16, 16, (8, n)), dtype=torch.float32).to(E4M3) for _ in range(TP)]
    common = torch.stack([s.float() for s in scales]).amax(0)
    for mode in ("gather", "replicate"):
        _set_env(monkeypatch, TPU_MINIMAX_M3_SP_A2A_OPROJ=mode)
        for r in range(TP):
            tp = types.SimpleNamespace(world_size=TP, rank_in_group=r)
            monkeypatch.setattr(sp_o_proj, "get_tp_group", lambda tp=tp: tp)

            def gather(t, r=r):
                if t.dtype == torch.int32:
                    return torch.stack([x.view(torch.int32) for x in rescaled_now()])
                return torch.stack([s.float() for s in scales])

            def rescaled_now():
                return [
                    (weights[j].float() * (scales[j].float() / common)).to(E4M3)
                    for j in range(TP)
                ]

            monkeypatch.setattr(sp_o_proj, "_all_gather_cpu", gather)
            o = nn.Module()
            o.weight = nn.Parameter(weights[r].clone(), requires_grad=False)
            o.weight_scale = nn.Parameter(scales[r].clone(), requires_grad=False)
            setattr(o, WEIGHT_FLIPPED_ATTR, True)
            sp_o_proj.prepare(o)
            assert getattr(o, sp_o_proj.PREPARED_ATTR)
            torch.testing.assert_close(o.weight_scale.float(), common)
            own = sp_o_proj.local_weight(o)
            torch.testing.assert_close(
                own.float() * o.weight_scale.float(), weights[r].float() * scales[r].float()
            )
            if mode == "replicate":
                assert o.weight.shape == (TP, 8, n)
                for j in range(TP):
                    torch.testing.assert_close(
                        o.weight[j].float(), rescaled_now()[(r + j) % TP].float()
                    )


def test_armed_rank(monkeypatch):
    rank = torch.tensor([[2]], dtype=torch.int32)
    for ordered, want in ((True, rank), (False, None)):
        armed = pallas_all_reduce._Armed(ranks=(0, 1, 2, 3), op=None, rank=rank, rank_ordered=ordered)
        monkeypatch.setattr(pallas_all_reduce, "_ARMED", armed)
        assert pallas_all_reduce.armed_rank([0, 1, 2, 3]) is want
        assert pallas_all_reduce.armed_rank([0, 1]) is None
    monkeypatch.setattr(pallas_all_reduce, "_ARMED", None)
    assert pallas_all_reduce.armed_rank([0, 1, 2, 3]) is None


# ------------------------------------------------------- e4m3 shared expert
def _shared_moe():
    moe = _moe(("m",), 0, _TP(1))
    mlp = object.__new__(MiniMaxM3MLP)
    nn.Module.__init__(mlp)
    mlp.gate_up_proj = _Linear(H, 64, ("sh", "gu"))
    mlp.down_proj = _Linear(32, H, ("sh", "dn"))
    mlp.act_fn = lambda gu: torch.tanh(gu[..., :32].float()).to(gu.dtype) * gu[..., 32:]
    moe.shared_experts = mlp
    # The MoE runner calls the shared expert on the bf16 hidden states.
    moe.experts = lambda hidden_states, router_logits: mlp(hidden_states)
    return moe, mlp


def test_e4m3_shared_expert(monkeypatch):
    monkeypatch.setattr(quant, "_e4m3_matmul", lambda *a: _jax_call(quant._e4m3_rows_matmul, *a))
    moe, mlp = _shared_moe()
    x = _input()
    q8, scale = mm._quantize_rows(x)
    deq = (q8.float() * scale).to(torch.bfloat16)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", False, raising=False)
    off = moe((q8, scale))
    assert len(mlp.gate_up_proj.inputs) == 1
    torch.testing.assert_close(mlp.gate_up_proj.inputs[0], deq)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", True, raising=False)
    on = moe((q8, scale))
    # gate_up took the e4m3 rows (apply_e4m3_rows), not the linear's forward.
    assert len(mlp.gate_up_proj.inputs) == 1
    assert "_e4m3_rows" not in mlp.__dict__
    assert _rel(on, off) < 2e-2
    # A bf16 input (non-SP step) keeps the bf16 path even with the switch on.
    moe(x)
    assert len(mlp.gate_up_proj.inputs) == 2 and mlp.gate_up_proj.inputs[1] is x


def test_e4m3_shared_expert_dynamo(monkeypatch):
    """The e4m3 rows handed to the shared expert trace under Dynamo
    (fullgraph) and leave no attribute behind."""
    monkeypatch.setattr(
        quant,
        "_e4m3_matmul",
        lambda q, rs, w, ws: ((q.float() @ w.float()) * rs * ws.float()).to(torch.bfloat16),
    )
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SP_E4M3_SHARED", True, raising=False)
    moe, mlp = _shared_moe()
    q8, scale = mm._quantize_rows(_input())
    want = moe((q8, scale))
    torch._dynamo.reset()
    got = torch.compile(lambda a, b: moe((a, b)), backend="eager", fullgraph=True)(q8, scale)
    torch.testing.assert_close(got, want)
    assert "_e4m3_rows" not in mlp.__dict__
    assert len(mlp.gate_up_proj.inputs) == 0


# ------------------------------------------------ shared expert in combine
from vllm_torchtpu.kernels.megablox import tc_combine  # noqa: E402

class _RoutedQM:
    """Routed experts' quant method stand-in: routed = tanh(x A) B, and the
    shared-in-combine entry adds the shared down_proj like the kernel's
    epilogue (tc_combine._shared_epilogue)."""

    def __init__(self, experts):
        self.experts = experts
        self.calls = 0

    def shared_in_combine_supported(self):
        return True

    def apply_monolithic_shared(self, layer, x, router_logits, act, w, scale):
        from vllm_torchtpu.kernels.megablox import tc_combine

        self.calls += 1
        routed = self.experts.routed_only(x, router_logits)
        shared = (_jnp(act), _jnp(w).astype(jnp.bfloat16), _jnp(scale).reshape(1, -1), 1.0)
        return _torch(tc_combine._shared_epilogue(_jnp(routed), shared))


def _moe_with_shared(monkeypatch, on):
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SHARED_IN_COMBINE", on, raising=False)
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE", True, raising=False)
    monkeypatch.setattr(tc_combine, "TOKEN_BLOCK", 32)
    monkeypatch.setattr(envs, "TPU_MINIMAX_M3_SHARED_IN_COMBINE_MIN_TOKENS", T, raising=False)
    tp = _TP(1)
    tp.local.rank = 0
    moe = _moe(("sc",), 0, tp)
    shared = _mlp(("sc",), 0, tp)
    shared.down_proj.reduce_results = False
    moe.shared_experts = shared
    ex = moe.experts

    def routed_only(x, router_logits):
        gate = torch.sigmoid(router_logits).mean(-1, keepdim=True)
        return ((torch.tanh(x @ ex.a).float() * gate).to(torch.bfloat16)) @ ex.b

    def runner(hidden_states, router_logits):
        # The runner's SP-step result: shared + routed, unreduced.
        ex.inputs.append(hidden_states)
        return shared(hidden_states) + routed_only(hidden_states, router_logits)

    ex.routed_only = routed_only
    ex.__call__ = runner
    moe.experts = types.SimpleNamespace(
        moe_config=types.SimpleNamespace(skip_final_all_reduce=True),
        routed_scaling_factor=1.0,
        routed_only=routed_only,
        inputs=ex.inputs,
    )
    moe.experts.routed_experts = types.SimpleNamespace(quant_method=_RoutedQM(moe.experts))
    return moe, runner


def test_shared_in_combine_matches_runner(monkeypatch):
    """The SP-step MoE with the shared down_proj handed to the combine equals
    the runner's shared + routed sum bit for bit (same roundings)."""
    moe, runner = _moe_with_shared(monkeypatch, True)
    x = _input()
    router_logits, _ = moe.gate(x)
    want = runner(x, router_logits)
    got = moe(x)
    assert moe.experts.routed_experts.quant_method.calls == 1
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_shared_in_combine_gates(monkeypatch):
    moe, _ = _moe_with_shared(monkeypatch, True)
    assert moe._shared_in_combine(T)
    assert not moe._shared_in_combine(T + 8)  # not whole combine blocks
    assert not moe._shared_in_combine(T // 2)  # below MIN_TOKENS
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE", False)
    assert not moe._shared_in_combine(T)
    monkeypatch.setattr(envs, "TPU_MOE_TC_COMBINE", True)
    moe.experts.moe_config.skip_final_all_reduce = False  # not an SP step
    assert not moe._shared_in_combine(T)
    moe.experts.moe_config.skip_final_all_reduce = True
    moe.experts.routed_scaling_factor = 2.0
    assert not moe._shared_in_combine(T)
    moe.experts.routed_scaling_factor = 1.0
    moe.shared_experts.down_proj.weight = moe.shared_experts.down_proj.weight.to(torch.bfloat16)
    assert not moe._shared_in_combine(T)
    moe, _ = _moe_with_shared(monkeypatch, False)
    assert not moe._shared_in_combine(T)


def test_shared_in_combine_keeps_trace_input_order(monkeypatch):
    """The warmup traces each bucket group separately and hands buckets from
    the small-bucket trace to the live (largest) one, which needs the same
    subgraph inputs in the same order (shape_variants.BucketExecutables).
    vLLM's split_graph orders them by first use, so the shared-in-combine
    step must consume the shared down_proj where the runner path does:
    after the shared gate_up, before the routed experts' weights."""
    from vllm.compilation.backends import split_graph

    from vllm_torchtpu.compilation import shape_variants

    bf16 = torch.bfloat16

    def param(*shape, dtype=bf16):
        return nn.Parameter(torch.randn(*shape).to(dtype), requires_grad=False)

    class Down(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = param(16, H)
            self.weight_scale = param(H, dtype=torch.float32)

    class Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up = param(H, 32)
            self.down_proj = Down()

        def activation(self, x):
            gu = x @ self.gate_up
            return torch.nn.functional.silu(gu[:, :16]) * gu[:, 16:]

        def forward(self, x):
            return (self.activation(x) @ self.down_proj.weight) * self.down_proj.weight_scale

    class QM:
        def apply_monolithic_shared(self, layer, x, router_logits, act, w, scale):
            return layer.routed(x, router_logits) + (act @ w) * scale

    class Routed(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = param(8, dtype=torch.float32)
            self.a = param(H, I)
            self.b = param(I, H)
            self.quant_method = QM()

        def routed(self, x, router_logits):
            gate = torch.sigmoid(router_logits + self.bias).mean(-1, keepdim=True)
            return (torch.tanh(x @ self.a) * gate).to(bf16) @ self.b

    class Runner(nn.Module):
        def __init__(self):
            super().__init__()
            self.routed_experts = Routed()

    moe = object.__new__(MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.shared_experts = Shared()
    moe.experts = Runner()
    routed = moe.experts.routed_experts

    def signature(fn):
        out = []

        def backend(gm, example_inputs):
            _, items = split_graph(gm, [])
            out.extend(shape_variants.graph_signature(it.graph) for it in items)
            return gm.forward

        torch._dynamo.reset()
        torch.compile(fn, backend=backend, fullgraph=True)(
            _input(), torch.randn(T, 8)
        )
        return out

    # The runner path: the shared expert first, then the routed experts.
    plain = signature(lambda x, logits: moe.shared_experts(x) + routed.routed(x, logits))
    combined = signature(lambda x, logits: moe._forward_shared_in_combine(x, logits))
    assert len(plain) == len(combined) == 1
    assert combined == plain


def test_small_all_gather_takes_every_scale_gather(tp, monkeypatch):
    """TPU_TP_PALLAS_SMALL_ALL_GATHER: every e4m3 row-scale gather goes through
    the armed small all-gather (none through XLA's f32 all-gather), with the
    same bits; key off never asks for it."""
    _set_env(monkeypatch, **_TARGET)
    monkeypatch.setattr(mm, "get_tp_group", tp.group)
    monkeypatch.setattr(envs, "TPU_TP_PALLAS_SMALL_ALL_GATHER", False, raising=False)
    monkeypatch.setattr(pallas_all_reduce, "all_gather_small", _boom)
    tp.calls.clear()
    want, _ = _forward(tp, monkeypatch, "full")
    xla_f32 = tp.calls.count("ag:torch.float32")
    assert xla_f32 > 0

    def small(ranks, x):
        assert x.dtype == torch.float32
        if tp.rank == 0:
            tp.calls.append("small")
        return torch.cat(tp._exchange(x), 0)

    monkeypatch.setattr(envs, "TPU_TP_PALLAS_SMALL_ALL_GATHER", True, raising=False)
    monkeypatch.setattr(pallas_all_reduce, "all_gather_small", small)
    tp.calls.clear()
    got, _ = _forward(tp, monkeypatch, "full")
    assert tp.calls.count("small") == xla_f32
    assert tp.calls.count("ag:torch.float32") == 0
    for g, w in zip(got, want):
        torch.testing.assert_close(g[0], w[0], rtol=0, atol=0)
    # Unarmed (None): XLA's all-gather as before.
    monkeypatch.setattr(pallas_all_reduce, "all_gather_small", lambda ranks, x: None)
    tp.calls.clear()
    _forward(tp, monkeypatch, "full")
    assert tp.calls.count("ag:torch.float32") == xla_f32
