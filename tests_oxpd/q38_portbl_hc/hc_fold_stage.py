# SPDX-License-Identifier: Apache-2.0
"""Shared harness for the q38 PORT-BL HC CPU proof (TPU_QWEN4_HC_DECODE_FOLD, TPU_QWEN4_HC_DECODE_TILE_ROWS; 0 chips).

Adapted (credit) from L3-HCDEF's DEFER-D harness tests/q38_l3_hcdef/hcdef_stage.py (q38-l3-hcdef-deferd 60b51e114). The
fold and the decode row tile are ported from Benliang Wang <benliangw@google.com>, commits d523c2ce0 / 28419c4a5 on
benliangw/q38fh-integ-pin-5b44702aea.

What is real and what is a stand-in:

* REAL: ``Qwen4ExpBackbone._forward_impl`` (the stage loop with the fold decision and the final-mixer hand-off),
  ``Qwen4ExpDecoderLayer.forward`` (pending hand-off, attn mix, attn -> mlp combine_mix, valid-token mask, defer
  return, mlp combine) and ``Qwen4ExpDecoderLayer.apply_ple`` for layers without PLE, ``GatedResidual`` (mix /
  combine / combine_mix / decode_fold_rows / _decode_fused / HCGATE / HCX routing) from the tree under test.
* REAL op builders: every fused op is built by vllm_adapter's own ``build_*`` functions (the GatedResidual
  constructor calls them from TPU_QWEN4_HC_DECODE_FUSED / _UP_T / _FOLD / _TILE_ROWS). Only torch_tpu's
  ``pallas.jax_op`` is replaced, by a recorder (``CoreOp``) that runs the builder's own JAX ``core`` on CPU with
  ``hc_decode.hc_decode_mix`` in Pallas TPU interpret mode. So the static arguments the builders choose (up_t,
  inject, max_tile, prologue) reach the real kernel, and every kernel call is logged (``CALLS``).
* STAND-IN: the attention block (one bf16 linear in the linear_attention slot), the MoE block (silu + one bf16
  linear), the PLE addend of the PLE layer (an elementwise affine map; only ``ple is not None`` matters to the
  hand-off rule). Neither the fold nor the tile writes KV / GDN / PLE state.

Served geometry (C): hc 4, hidden 2560, low rank 320 (Qwen3.8), PLE at layer index 1, HC prefill tiles
up / norm / full = 128 / 128 / 256 (v8 record env Q38CD_ADDITIONAL_CONFIG).
"""

import contextlib
import os
import types

import torch
import torch.nn.functional as F
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import hc_decode
from vllm_torchtpu.models.vllm.qwen4_exp import hyperconnection as hcmod
from vllm_torchtpu.models.vllm.qwen4_exp import model as M

HC = 4
DIMS = {"real": dict(hidden=2560, lowrank=320), "small": dict(hidden=128, lowrank=32)}
TILES = dict(prefill_up_tile=128, prefill_norm_tile=128, prefill_full_tile=256)
EPS = 1e-6
PLE_AT = 1
FOLD = "TPU_QWEN4_HC_DECODE_FOLD"
TILE = "TPU_QWEN4_HC_DECODE_TILE_ROWS"
GATE = "VLLM_TPU_HC_DECODE_FUSED_MAX_ROWS"
HCX = "VLLM_TPU_HC_XLA_MAX_ROWS"
FUSED = "TPU_QWEN4_HC_DECODE_FUSED"
UPT = "TPU_QWEN4_HC_DECODE_UP_T"
DEFER = "TPU_QWEN4_HC_PREFILL_DEFER_COMBINE"
DEFER_WHOLE = "TPU_QWEN4_HC_PREFILL_DEFER_WHOLE"
KEYS = (FOLD, TILE, GATE, HCX, FUSED, UPT, DEFER, DEFER_WHOLE)
SERVED = {FUSED: "1", UPT: "1", GATE: "128", HCX: "256"}   # the v8 record env's HC keys (C)


def tree_root() -> str:
    import vllm_torchtpu

    return os.path.dirname(os.path.dirname(os.path.realpath(vllm_torchtpu.__file__)))


@contextlib.contextmanager
def env_state(**values):
    """Exactly ``values`` among KEYS in os.environ (every other KEY unset); envs reads them live."""
    saved = {k: os.environ.get(k) for k in KEYS}
    stale = {k: vars(envs)[k] for k in KEYS if k in vars(envs)}

    def clear():
        for k in KEYS:
            if k in vars(envs):
                delattr(envs, k)
        cache_clear = getattr(envs.__getattr__, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()

    try:
        for k in KEYS:
            os.environ.pop(k, None)
        for k, v in values.items():
            assert k in KEYS, k
            if v is not None:
                os.environ[k] = str(v)
        clear()
        # Registered keys also as module attributes, parsed by envs' own getters (as after enable_envs_cache):
        # Dynamo traces a module attribute, not the module-level __getattr__.
        for k in KEYS:
            if k in envs.environment_variables:
                setattr(envs, k, envs.environment_variables[k]())
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        clear()
        for k, v in stale.items():
            setattr(envs, k, v)


# ---------------------------------------------------------------- torch <-> jax
def to_jax(t):
    import jax.numpy as jnp
    import ml_dtypes
    import numpy as np

    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        return jnp.asarray(t.view(torch.int16).numpy().view(ml_dtypes.bfloat16))
    if t.dtype == torch.float32:
        return jnp.asarray(np.asarray(t.numpy()))
    raise TypeError(f"unexpected dtype {t.dtype}")


def to_torch(a):
    import ml_dtypes
    import numpy as np

    arr = np.asarray(a)
    if arr.dtype == ml_dtypes.bfloat16:
        return torch.from_numpy(arr.view(np.int16).copy()).view(torch.bfloat16)
    if arr.dtype == np.float32:
        return torch.from_numpy(arr.copy())
    raise TypeError(f"unexpected dtype {arr.dtype}")


# ---------------------------------------------------------------- the real builders, run on CPU
CALLS: list = []   # one dict per hc_decode_mix call made through a CoreOp
_INTERP = None


def _interp_params():
    global _INTERP
    if _INTERP is None:
        from jax.experimental.pallas import tpu as pltpu

        _INTERP = pltpu.InterpretParams()
    return _INTERP


def _recording_mix(real):
    """hc_decode.hc_decode_mix in interpret mode, logging the static choices the builder made."""

    def mix(x, weight, down, up, inject_weight=None, **kw):
        tokens = int(x.shape[0])
        max_tile = int(kw.get("max_tile", 0))
        CALLS.append(dict(
            tokens=tokens, up_t=bool(kw.get("up_t", False)), max_tile=max_tile,
            prologue=kw.get("block") is not None, inject=inject_weight is not None,
            tile=hc_decode.decode_kernel_tile(tokens, max_tile),
        ))
        return real(x, weight, down, up, inject_weight, interpret=_interp_params(), **kw)

    return mix


class CoreOp:
    """Stands in for torch_tpu's jax_op result: calls the builder's own JAX ``core`` with CPU tensors."""

    def __init__(self, name, core):
        self.name, self.core = name, core

    def register_fake(self, fn):
        return fn

    def __call__(self, *tensors):
        from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va

        real = va.hc_decode_mix
        va.hc_decode_mix = _recording_mix(real)
        try:
            out = self.core(*[to_jax(t) for t in tensors])
        finally:
            va.hc_decode_mix = real
        if isinstance(out, (tuple, list)):
            return tuple(to_torch(o) for o in out)
        return to_torch(out)


BUILT: list = []   # (op name) per jax_op call made while building through ``core_builders``


def _builders():
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va

    return [getattr(va, n) for n in dir(va) if n.startswith("build_") and hasattr(getattr(va, n), "cache_clear")]


@contextlib.contextmanager
def core_builders():
    """While active, vllm_adapter's builders register ``CoreOp``s instead of torch_tpu jax ops (caches cleared on
    entry and exit, so no CoreOp leaks into a later real build)."""
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as va

    for b in _builders():
        b.cache_clear()
    real = va.pallas.jax_op

    def spy(name, fn=None, /, **kw):
        BUILT.append(name)
        return CoreOp(name, fn)

    va.pallas.jax_op = spy
    try:
        yield
    finally:
        va.pallas.jax_op = real
        for b in _builders():
            b.cache_clear()


# ---------------------------------------------------------------- modules
_SCALES = {
    "hc_norm.weight": 0.3,
    "input_mix_weight_down.weight": 0.02,
    "input_mix_weight_up.weight": 0.05,
    "block_inject_weight.weight": 0.02,
}


def _fill(mod, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            p.copy_((torch.randn(p.shape, generator=gen) * _SCALES.get(name, 0.02)).to(p.dtype))


def hc_config(dims):
    return hcmod.HyperConnectionConfig(
        hc_count=HC, hidden_size=dims["hidden"], params_dtype=torch.bfloat16, hc_lowrank=dims["lowrank"],
        rms_norm_eps=EPS, hc_per_branch_norm=True,
    )


def build_hc(dims, seed, inject=True, tiles=True):
    mod = hcmod.GatedResidual(hc_config(dims), use_combine=inject, **(TILES if tiles else {}))
    _fill(mod, seed)
    return mod


class FoldLayer(nn.Module):
    """One decoder layer: the REAL ``Qwen4ExpDecoderLayer.forward`` on real GatedResidual HCs; attention
    (linear_attention slot), MoE and the PLE addend are stand-ins. ``seen`` (a list or None) records the
    attention and MoE inputs, i.e. both HC mixes' outputs."""

    layer_type = "linear_attention"
    pooled = False
    forward = M.Qwen4ExpDecoderLayer.forward

    def __init__(self, dims, seed, ple, tiles=True):
        super().__init__()
        h = dims["hidden"]
        self.attn_hyper_connection = build_hc(dims, seed * 7919 + 1, tiles=tiles)
        self.mlp_hyper_connection = build_hc(dims, seed * 7919 + 2, tiles=tiles)
        gen = torch.Generator().manual_seed(seed * 7919 + 3)
        self.attn_w = nn.Parameter((torch.randn(h, h, generator=gen) / h**0.5).bfloat16(), requires_grad=False)
        self.moe_w = nn.Parameter((torch.randn(h, h, generator=gen) / h**0.5).bfloat16(), requires_grad=False)
        self.ple = types.SimpleNamespace(stand_in=True) if ple else None
        self.seen = None

    def linear_attn(self, block):
        if self.seen is not None:
            self.seen.append(("attn_in", block))
        return F.linear(block, self.attn_w)

    def mlp(self, block):
        if self.seen is not None:
            self.seen.append(("moe_in", block))
        return F.linear(F.silu(block.float()).to(block.dtype), self.moe_w)

    def apply_ple(self, hidden, *args, **kwargs):
        if self.ple is None:
            return M.Qwen4ExpDecoderLayer.apply_ple(self, hidden, *args, **kwargs)  # the real method
        return (hidden.float() * 0.5 + 0.25).to(hidden.dtype)


class FoldStage(nn.Module):
    """A last pipeline stage of ``layers`` decoder layers (PLE at PLE_AT) plus the final HC mixer, run through the
    REAL ``Qwen4ExpBackbone._forward_impl``; returns (mixed, pre-mixer streams) like the MTP-emitting backbone."""

    def __init__(self, dims, layers=4, seed=0, tiles=True, ple_at=PLE_AT, hooks=True):
        super().__init__()
        self.layers = nn.ModuleList(FoldLayer(dims, seed * 101 + i, i == ple_at, tiles) for i in range(layers))
        self.hyper_connection_mixer = build_hc(dims, seed * 7919 + 99991, inject=False, tiles=tiles)
        self.stage = range(0, layers)
        self.first_stage, self.last_stage = False, True
        self.host_ple_index = None
        self.hc_count = HC
        self.sp_pad_rows = 0
        self.embed_tokens = None
        self.emit_mtp_hidden = True
        self.defers: list = []
        if hooks:
            for layer in self.layers:
                layer.register_forward_pre_hook(self._record, with_kwargs=True)

    def _record(self, module, args, kwargs):
        self.defers.append(dict(defer=bool(kwargs.get("defer")), pending=kwargs.get("pending") is not None,
                                decode_fold=bool(kwargs.get("decode_fold"))))

    def forward(self, hidden, ids, valid):
        return M.Qwen4ExpBackbone._forward_impl(
            self, ids, ids, None, sequence_parallel=False, intermediate_tensors={"hidden_states": hidden},
            query_start_loc=None, ngram_context=None, state_slots=None, context_lens=None, valid_tokens=valid,
            main_block_table=None, compressed_block_table=None,
        )

    def record(self, on: bool):
        for layer in self.layers:
            layer.seen = [] if on else None

    def seen(self):
        return [(i, k, t) for i, layer in enumerate(self.layers) for k, t in (layer.seen or [])]


def build_stage(env: dict, dims, layers=4, seed=0, tiles=True, hooks=True):
    """A FoldStage built under ``env`` through the real builders (CoreOps)."""
    BUILT.clear()
    with env_state(**env), core_builders():
        stage = FoldStage(dims, layers=layers, seed=seed, tiles=tiles, hooks=hooks)
    return stage, list(BUILT)


def stage_inputs(rows, hidden, seed=0):
    g = torch.Generator().manual_seed(77 + rows * 13 + seed)
    x = torch.randn(rows, HC * hidden, generator=g).bfloat16()
    ids = torch.zeros(rows, dtype=torch.int64)
    valid = torch.ones(rows, dtype=torch.bool)
    valid[-1] = False   # one padded row: the MoE input is masked there, the combine still runs on it
    return x, ids, valid


def run(stage, env, rows, dims, seed=0):
    """One forward under ``env``; returns dict(mixed, hidden, seen, calls, defers)."""
    x, ids, valid = stage_inputs(rows, dims["hidden"], seed)
    del CALLS[:]
    stage.defers.clear()
    stage.record(True)
    with env_state(**env), torch.no_grad():
        mixed, hidden = stage(x, ids, valid)
    out = dict(mixed=mixed, hidden=hidden, seen=stage.seen(), calls=list(CALLS), defers=list(stage.defers))
    stage.record(False)
    return out


def counts(calls):
    """Kernel calls by kind: plain mixes, combine prologues with / without the injection gate."""
    c = {"mix": 0, "prologue_inj": 0, "prologue_noinj": 0}
    for k in calls:
        if not k["prologue"]:
            c["mix"] += 1
        elif k["inject"]:
            c["prologue_inj"] += 1
        else:
            c["prologue_noinj"] += 1
    return c


def expected(layers, ple_at, fold):
    """Kernel calls of one fused stage forward (C, the stage loop): the attn -> mlp prologue per layer; with the fold
    one more inj prologue per hand-off into a layer without PLE and one noinj prologue into the final mixer, each
    replacing a plain mix."""
    handoffs = sum(1 for i in range(layers - 1) if (i + 1) != ple_at) if fold else 0
    final = 1 if fold else 0
    return dict(mix=layers + 1 - handoffs - final, prologue_inj=layers + handoffs, prologue_noinj=final), \
        handoffs + final


def compare(a, b):
    """Bitwise comparison of two run() results: (all_equal, detail dict)."""
    import numpy as np

    def ulp(x, y):
        xi = x.contiguous().view(torch.int16).numpy().astype(np.int64)
        yi = y.contiguous().view(torch.int16).numpy().astype(np.int64)

        def ordered(v):
            v = v & 0xFFFF
            return np.where(v & 0x8000, -(v & 0x7FFF), v & 0x7FFF)

        d = np.abs(ordered(xi) - ordered(yi))
        return int(d.max()) if d.size else 0, int((d > 0).sum())

    det = {}
    ok = True
    for name in ("mixed", "hidden"):
        eq = torch.equal(a[name], b[name])
        det[name] = (eq,) + ((ulp(a[name], b[name]),) if not eq else ())
        ok = ok and eq
    sa, sb = a["seen"], b["seen"]
    same_len = len(sa) == len(sb) and [(i, k) for i, k, _ in sa] == [(i, k) for i, k, _ in sb]
    neq = [(i, k) for (i, k, x), (_, _, y) in zip(sa, sb) if not torch.equal(x, y)] if same_len else ["shape"]
    det["blocks"] = (len(sa), len(sb), neq)
    ok = ok and same_len and not neq
    return ok, det


