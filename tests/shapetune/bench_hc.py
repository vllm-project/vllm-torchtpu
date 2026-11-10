# SPDX-License-Identifier: Apache-2.0
"""Per-shape HyperConnection sweep (Qwen3.8-Flash, one v7x chip).

HC is TP-replicated: 4 streams x hidden 2560, low rank 320, two HCs a layer
(attn, mlp), 48 layers. One timed call = one layer's HC work:
  attn mix -> attn combine + mlp mix (combine_mix) -> mlp combine,
with the mixed outputs standing in for the attention / MoE block outputs.

decode (rows = all tokens, every rank): the served stock path is the unfused
  XLA chain (TPU_QWEN4_HC_DECODE_FUSED is off in the recipe), mirrored here in
  JAX from GatedResidual.mix / combine. Knobs: fused (the hc_decode Pallas
  kernels; ulp-level, NOT bitwise vs XLA), tile (forced decode_token_tile),
  row_chunk (hc_decode._ROW_CHUNK, the norm loop step; bitwise expected).
prefill (sequence parallel: each rank holds T/8 rows): the full-mix kernels
  (hc_full) when rows >= 2 x hc_prefill_full_tile, else the norm / up-epilogue
  kernels + XLA. Served: hc_prefill_full_tile = norm = up = 128,
  VLLM_TPU_HC_PREFILL_WIDE_TILE 256, TPU_QWEN4_HC_PREFILL_NORM bf16ep64; UP_T
  and INJECT are set but IGNORED in serving (the recipe turns on the
  SparseCore all-gather offload, see hyperconnection.prefill_full_variant).
  Knobs: full_tile, wide, norm (vpu/rows64/bf16ep/bf16ep64 bitwise; gram64
  not), up_t / inject (bitwise), norm_tile / up_tile (fallback rows only).

ST_PHASES=decode,prefill restricts phases; ST_TINY=1 builds inputs only.
"""

import contextlib
import functools
import os

import jax
import jax.numpy as jnp
import numpy as np

from common import bench, digest, try_config

TINY = os.environ.get("ST_TINY") == "1"
PHASES = os.environ.get("ST_PHASES", "decode,prefill").split(",")

HC, H, LR, EPS = (4, 256, 128, 1e-6) if TINY else (4, 2560, 320, 1e-6)
W = HC * H
TP = 8
DECODE_ROWS = (4, 8, 16, 32, 64, 96, 128)
PREFILL_TOKENS = (1024, 2048, 4096, 8192)


@contextlib.contextmanager
def patched(obj, **values):
    old = {k: getattr(obj, k) for k in values}
    try:
        for k, v in values.items():
            setattr(obj, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def _flag(outs):
    leaf = jax.tree_util.tree_leaves(outs)[0]
    return (leaf.reshape(-1)[0].astype(jnp.float32) > 1e30).astype(jnp.float32)


def measure(fwd, ops):
    ops = jax.tree_util.tree_map(jnp.copy, ops)
    outs0 = jax.jit(lambda o: fwd(o, jnp.float32(0)))(ops)
    dg = digest(outs0)

    def body(carry):
        o, prev = carry
        return o, fwd(o, _flag(prev))

    us, _ = bench(body, (ops, outs0))
    return us, dg


def _hc_weights(seed):
    rng = np.random.default_rng(seed)

    def bf16(shape, scale):
        return jnp.asarray(rng.normal(size=shape) * scale, jnp.bfloat16)

    # hc_norm.weight [W], input_mix_weight_down [LR, W], _up [W, LR],
    # block_inject_weight [HC, W].
    return (bf16((W,), 0.3), bf16((LR, W), 0.02), bf16((W, LR), 0.05),
            bf16((HC, W), 0.02))


def _hyper(rows, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(rows, W)), jnp.bfloat16)


# --- the torch path (GatedResidual), in JAX --------------------------------


def _linear(x, w):
    """F.linear on BF16: x @ w.T, f32 accumulation, BF16 out."""
    return jax.lax.dot_general(
        x, w, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)


def xla_normalize(x, weight):
    flat = x.astype(jnp.float32)
    parts = []
    for b in range(HC):
        v = flat[:, b * H : (b + 1) * H]
        parts.append(v * jax.lax.rsqrt(jnp.mean(v * v, -1, keepdims=True) + EPS))
    out = jnp.concatenate(parts, -1)
    return (out * (1.0 + weight.astype(jnp.float32))).astype(x.dtype)


def xla_gate_mid(norm, down):
    gate = _linear(norm, down) / jnp.bfloat16(HC)
    return jax.nn.silu(gate.astype(jnp.float32)).astype(jnp.bfloat16)


def xla_up_mix(mid, norm, up):
    g = jax.nn.sigmoid(_linear(mid, up).astype(jnp.float32))
    weighted = g * norm.astype(jnp.float32)
    mixed = weighted[:, :H]
    for b in range(1, HC):
        mixed = mixed + weighted[:, b * H : (b + 1) * H]
    return (mixed / HC).astype(jnp.bfloat16)


def xla_injection(norm, inject):
    return 2.0 * jax.nn.sigmoid(_linear(norm, inject).astype(jnp.float32) / HC)


def xla_mix(x, weights):
    w, down, up, inject = weights
    norm = xla_normalize(x, w)
    return xla_up_mix(xla_gate_mid(norm, down), norm, up), xla_injection(norm, inject)


def xla_combine(h, block, inj):
    hf, bf = h.astype(jnp.float32), block.astype(jnp.float32)
    return jnp.concatenate(
        [hf[:, b * H : (b + 1) * H] + bf * inj[:, b : b + 1] for b in range(HC)], -1
    ).astype(h.dtype)


# --- decode -----------------------------------------------------------------


def _decode_fwd(cfg):
    from vllm_torchtpu.kernels.qwen4_exp.hc_decode import hc_decode_mix

    fused = cfg.get("fused", 0)
    mix = functools.partial(hc_decode_mix, branches=HC, hidden=H, eps=EPS)

    def fwd(ops, flag):
        h, *ws = ops
        wa, wm = tuple(ws[:4]), tuple(ws[4:])
        h = h + flag.astype(h.dtype)
        if not fused:
            m1, i1 = xla_mix(h, wa)
            h2 = xla_combine(h, m1, i1)
            m2, i2 = xla_mix(h2, wm)
        else:
            m1, i1 = mix(h, *wa)
            m2, i2, h2 = mix(h, *wm, block=m1, prev_injection=i1)
        return xla_combine(h2, m2, i2), m1, m2, i2

    return fwd


def _decode_configs(rows):
    cfgs = [{"fused": 0}, {"fused": 1}]
    if rows >= 64:
        cfgs += [{"fused": 1, "tile": 32}, {"fused": 1, "row_chunk": 16},
                 {"fused": 1, "row_chunk": 64}]
    if rows >= 32:
        cfgs += [{"fused": 1, "tile": 16}]
    return cfgs


def _run_decode(emit, wa, wm):
    from vllm_torchtpu.kernels.qwen4_exp import hc_decode

    for rows in DECODE_ROWS:
        bucket = {"phase": "decode", "rows": rows}
        ops = (_hyper(rows, rows), *wa, *wm)
        for i, cfg in enumerate(_decode_configs(rows)):
            tile = cfg.get("tile")
            consts = dict(_ROW_CHUNK=cfg.get("row_chunk", hc_decode._ROW_CHUNK))
            if tile:
                consts["decode_token_tile"] = lambda t, tile=tile: (
                    tile if t % tile == 0 else None
                )

            def fn(cfg=cfg, consts=consts):
                hc_decode._hc_decode_mix.clear_cache()
                try:
                    with patched(hc_decode, **consts):
                        return measure(_decode_fwd(cfg), ops)
                finally:
                    hc_decode._hc_decode_mix.clear_cache()

            try_config(emit, bucket, cfg, fn, default=i == 0)


# --- prefill ----------------------------------------------------------------

PREFILL_DEFAULT = dict(full_tile=128, wide=256, norm="bf16ep64", up_t=0, inject=0,
                       norm_tile=128, up_tile=128)


def _prefill_plan(rows, cfg):
    """The served dispatch (GatedResidual._prefill_full / _prefill_tile)."""
    full, wide = cfg["full_tile"], cfg["wide"]
    if rows < 2 * full or rows % (2 * full):
        return None
    tiles = (wide, full) if wide else (full,)
    return next(t for t in tiles if rows % (2 * t) == 0)


def _prefill_fwd(rows, cfg):
    from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
    from vllm_torchtpu.kernels.qwen4_exp.hc_full import _combine_full_mix, _full_mix
    from vllm_torchtpu.kernels.qwen4_exp.hc_norm import _normalize
    from vllm_torchtpu.kernels.qwen4_exp.hc_up import _up_epilogue

    tile = _prefill_plan(rows, cfg)
    up_t, inject = bool(cfg["up_t"]), bool(cfg["inject"])
    static = dict(branches=HC, hidden=H, eps=EPS, up_t=up_t, norm_impl=cfg["norm"])

    def gate_of(second, weights):
        if inject:
            return 2.0 * jax.nn.sigmoid(second.astype(jnp.float32) / HC)
        return xla_injection(second, weights[3])

    def kernel_args(weights):
        w, down, up, inj = weights
        return (w, down, up.T if up_t else up) + ((inj,) if inject else ())

    def fallback_mix(x, weights):
        w, down, up, inj = weights
        nt, ut = cfg["norm_tile"], cfg["up_tile"]
        if nt and rows >= nt and rows % nt == 0:
            norm = _normalize(x, w, branches=HC, hidden=H, eps=EPS, token_tile=nt)
        else:
            norm = xla_normalize(x, w)
        mid = xla_gate_mid(norm, down)
        if ut and rows >= ut and rows % ut == 0:
            mixed = _up_epilogue(mid, norm, up, branches=HC, hidden=H, token_tile=ut)
        else:
            mixed = xla_up_mix(mid, norm, up)
        return mixed, xla_injection(norm, inj)

    def fwd(ops, flag):
        h, *ws = ops
        wa, wm = tuple(ws[:4]), tuple(ws[4:])
        h = h + flag.astype(h.dtype)
        if tile is None:
            m1, i1 = fallback_mix(h, wa)
            h2 = xla_combine(h, m1, i1)
            m2, i2 = fallback_mix(h2, wm)
            return xla_combine(h2, m2, i2), m1, m2
        m1, s1 = _full_mix(h, *kernel_args(wa), token_tile=tile, **static)
        i1 = gate_of(s1, wa)
        m2, s2, h2 = _combine_full_mix(h, m1, i1, *kernel_args(wm), token_tile=tile,
                                       **static)
        i2 = gate_of(s2, wm)
        h3 = gated_residual_fn(HC, H, cfg["full_tile"])(h2, m2, i2)
        return h3, m1, m2

    return fwd


def _prefill_configs(rows):
    if _prefill_plan(rows, PREFILL_DEFAULT) is None:
        return [{}, {"norm_tile": 64}, {"norm_tile": 0, "up_tile": 0}]
    cands = [
        {}, {"wide": 0}, {"wide": 512}, {"full_tile": 256, "wide": 0},
        {"full_tile": 256, "wide": 512}, {"norm": "vpu"}, {"norm": "rows64"},
        {"norm": "gram64"}, {"up_t": 1}, {"inject": 1}, {"up_t": 1, "inject": 1},
    ]
    out, seen = [], set()
    for c in cands:
        full = {**PREFILL_DEFAULT, **c}
        tile = _prefill_plan(rows, full)
        if tile is None:
            continue
        key = (tile, full["full_tile"], full["norm"], full["up_t"], full["inject"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _run_prefill(emit, wa, wm):
    from vllm_torchtpu.kernels.qwen4_exp.hc_full import _combine_full_mix, _full_mix

    for tokens in PREFILL_TOKENS:
        rows = tokens // TP
        bucket = {"phase": "prefill", "tokens": tokens, "rows": rows}
        ops = (_hyper(rows, tokens), *wa, *wm)
        for i, c in enumerate(_prefill_configs(rows)):
            cfg = {**PREFILL_DEFAULT, **c}

            def fn(cfg=cfg):
                _full_mix.clear_cache()
                _combine_full_mix.clear_cache()
                return measure(_prefill_fwd(rows, cfg), ops)

            note_cfg = dict(c, tile=_prefill_plan(rows, cfg))
            try_config(emit, bucket, note_cfg, fn, default=i == 0)


def main(emit):
    wa, wm = _hc_weights(11), _hc_weights(12)
    if TINY:
        h = _hyper(8, 0)
        print("bench_hc tiny", h.shape, [w.shape for w in wa],
              _decode_configs(128), _prefill_configs(128), _prefill_configs(1024))
        # The XLA reference runs on CPU.
        print("xla mix ok", jax.jit(_decode_fwd({"fused": 0}))((h, *wa, *wm),
                                                         jnp.float32(0))[0].shape)
        return
    if "decode" in PHASES:
        _run_decode(emit, wa, wm)
    if "prefill" in PHASES:
        _run_prefill(emit, wa, wm)
