"""
tpu_gemma4.py
-------------
The ``"tpu"`` calibrator of ``sparsity.prune_gemma4``: sequential SparseGPT
over a Gemma-4 text stack, entirely on TPU (~2.6 s per 31B layer vs ~370 s
for the CPU reference). Same contract as ``gemma4.calibrate_layers_cpu``.

The decoder layer is a JAX port of ``Gemma4TextDecoderLayer`` (float32
RMSNorm, bf16 matmul outputs, float32 softmax, tanh GELU) with the calibration
samples sharded over every local chip; the four Hessians of a layer (q/k/v
share one, gate/up share one) are accumulated on the MXU during that forward
and the OBS solves (``tpu_sparsegpt.sparsegpt_solve``) are spread over the
chips. Dense text variants only (12B, 31B); others raise NotImplementedError.

libtpu admits one process per TPU and JAX holds the chips until exit, so
callers that start vLLM afterwards run this as a child process::

    python -m sparsity.tpu_gemma4 --json '<prune_gemma4 kwargs as JSON>'
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .tpu_sparsegpt import (BF16, F32, _check_ok, _gram, _to_np, _to_torch,
                            check_nm, prepare_hinv, sparsegpt_solve)

try:  # jax >= 0.6
    from jax import shard_map as _shard_map
except ImportError:  # pragma: no cover
    from jax.experimental.shard_map import shard_map as _shard_map

# Linear names in the order find_layers() returns them.
_ATTN = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj")
_MLP = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
# Which Hessian each linear uses: linears that read the same input share one.
_H_OF = {
    "self_attn.q_proj": "attn_in",
    "self_attn.k_proj": "attn_in",
    "self_attn.v_proj": "attn_in",
    "self_attn.o_proj": "o_in",
    "mlp.gate_proj": "mlp_in",
    "mlp.up_proj": "mlp_in",
    "mlp.down_proj": "down_in",
}
_H_ORDER = ("attn_in", "o_in", "mlp_in", "down_in")

# --------------------------------------------------------------------------
# Gemma-4 decoder layer in JAX
# --------------------------------------------------------------------------


def _rms(x, w, eps):
    xf = x.astype(F32)
    ms = jnp.mean(xf * xf, axis=-1, keepdims=True) + eps
    y = xf * jnp.power(ms, -0.5)
    if w is not None:
        y = y * w.astype(F32)
    return y.astype(x.dtype)


def _linear(x, w):
    """x @ w.T with bfloat16 inputs, float32 accumulation, bfloat16 output."""
    y = lax.dot_general(x,
                        w, (((x.ndim - 1, ), (1, )), ((), ())),
                        preferred_element_type=F32)
    return y.astype(x.dtype)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return jnp.concatenate((-x[..., h:], x[..., :h]), axis=-1)


def _rope(x, cos, sin):
    # x: (B, S, H, hd); cos/sin: (S, hd)
    return x * cos[None, :, None, :] + _rotate_half(x) * sin[None, :, None, :]


def _layer_forward(x, p, cos, sin, spec):
    """One Gemma4TextDecoderLayer. x: (B, S, D) bf16.

    Returns (out, linear inputs keyed like _H_ORDER).
    """
    B, S, D = x.shape
    nh, nkv, hd = spec["num_heads"], spec["num_kv"], spec["head_dim"]
    eps = spec["eps"]

    residual = x
    h = _rms(x, p["input_layernorm"], eps)
    q = _linear(h, p["self_attn.q_proj"]).reshape(B, S, nh, hd)
    q = _rope(_rms(q, p["q_norm"], eps), cos, sin)
    k_raw = _linear(h, p["self_attn.k_proj"]).reshape(B, S, nkv, hd)
    if "self_attn.v_proj" in p:
        v = _linear(h, p["self_attn.v_proj"]).reshape(B, S, nkv, hd)
    else:  # attention_k_eq_v: values are the raw keys
        v = k_raw
    k = _rope(_rms(k_raw, p["k_norm"], eps), cos, sin)
    v = _rms(v, None, eps)

    g = nh // nkv
    qg = q.reshape(B, S, nkv, g, hd)
    s = jnp.einsum("bqkgd,bskd->bkgqs", qg, k, preferred_element_type=F32)
    qi = lax.broadcasted_iota(jnp.int32, (S, S), 0)
    ki = lax.broadcasted_iota(jnp.int32, (S, S), 1)
    allowed = ki <= qi
    if spec["sliding_window"]:
        allowed = allowed & (qi - ki < spec["sliding_window"])
    s = jnp.where(allowed, s, -jnp.inf)
    m = jnp.max(s, axis=-1, keepdims=True)
    pr = jnp.exp(s - m)
    denom = jnp.sum(pr, axis=-1, keepdims=True)
    o = jnp.einsum(
        "bkgqs,bskd->bkgqd", pr.astype(
            x.dtype), v, preferred_element_type=F32) / denom
    o = o.astype(x.dtype)
    o_in = jnp.transpose(o, (0, 3, 1, 2, 4)).reshape(B, S, nh * hd)
    a = _linear(o_in, p["self_attn.o_proj"])
    a = _rms(a, p["post_attention_layernorm"], eps)
    x = residual + a

    residual = x
    h2 = _rms(x, p["pre_feedforward_layernorm"], eps)
    gt = _linear(h2, p["mlp.gate_proj"])
    up = _linear(h2, p["mlp.up_proj"])
    act = jax.nn.gelu(gt.astype(F32), approximate=True).astype(x.dtype) * up
    mo = _linear(act, p["mlp.down_proj"])
    mo = _rms(mo, p["post_feedforward_layernorm"], eps)
    x = residual + mo
    x = x * p["layer_scalar"]
    return x, {"attn_in": h, "o_in": o_in, "mlp_in": h2, "down_in": act}


def _layer_spec(cfg, layer_idx: int) -> dict:
    layer_type = cfg.layer_types[layer_idx]
    sliding = layer_type == "sliding_attention"
    alt = bool(getattr(cfg, "attention_k_eq_v", False)) and not sliding
    head_dim = (cfg.global_head_dim if
                (not sliding
                 and getattr(cfg, "global_head_dim", None)) else cfg.head_dim)
    return {
        "layer_type": layer_type,
        "num_heads": cfg.num_attention_heads,
        "num_kv":
        cfg.num_global_key_value_heads if alt else cfg.num_key_value_heads,
        "head_dim": head_dim,
        "sliding_window": cfg.sliding_window if sliding else 0,
        "eps": cfg.rms_norm_eps,
    }


def _check_supported(cfg) -> None:
    bad = []
    if getattr(cfg, "enable_moe_block", False):
        bad.append("enable_moe_block")
    if getattr(cfg, "hidden_size_per_layer_input", 0):
        bad.append("hidden_size_per_layer_input")
    if getattr(cfg, "num_kv_shared_layers", 0):
        bad.append("num_kv_shared_layers")
    if getattr(cfg, "attention_bias", False):
        bad.append("attention_bias")
    if getattr(cfg, "hidden_activation", "") != "gelu_pytorch_tanh":
        bad.append(f"hidden_activation={cfg.hidden_activation}")
    if getattr(cfg, "use_double_wide_mlp", False) and getattr(
            cfg, "num_kv_shared_layers", 0):
        bad.append("use_double_wide_mlp")
    if bad:
        raise NotImplementedError(
            "calibrate_layers_tpu has no JAX port for this Gemma-4 variant "
            f"({', '.join(bad)}); use the CPU calibrator or TPUSparseGPT with "
            "the torch forward")


def _layer_params(layer) -> Dict[str, np.ndarray]:
    attn = layer.self_attn
    p = {
        "input_layernorm": layer.input_layernorm.weight,
        "post_attention_layernorm": layer.post_attention_layernorm.weight,
        "pre_feedforward_layernorm": layer.pre_feedforward_layernorm.weight,
        "post_feedforward_layernorm": layer.post_feedforward_layernorm.weight,
        "q_norm": attn.q_norm.weight,
        "k_norm": attn.k_norm.weight,
        "layer_scalar": layer.layer_scalar,
        "self_attn.q_proj": attn.q_proj.weight,
        "self_attn.k_proj": attn.k_proj.weight,
        "self_attn.o_proj": attn.o_proj.weight,
        "mlp.gate_proj": layer.mlp.gate_proj.weight,
        "mlp.up_proj": layer.mlp.up_proj.weight,
        "mlp.down_proj": layer.mlp.down_proj.weight,
    }
    if attn.v_proj is not None:
        p["self_attn.v_proj"] = attn.v_proj.weight
    return {k: _to_np(v) for k, v in p.items()}


class _Gemma4Runner:
    """Holds the device mesh and the compiled per-layer passes."""

    def __init__(self, nsamples: int, microbatch: int = 4):
        devs = jax.devices()
        ndev = len(devs)
        while nsamples % ndev:
            ndev -= 1
        self.devices = devs[:ndev]
        self.mesh = Mesh(np.array(self.devices), ("d", ))
        n_local = nsamples // ndev
        mb = min(microbatch, n_local)
        while n_local % mb:
            mb -= 1
        self.microbatch = mb
        self.rep = NamedSharding(self.mesh, P())
        self.shard = NamedSharding(self.mesh, P("d"))
        self._cache = {}

    def _pass(self, spec: dict, collect: bool, has_v: bool):
        key = (tuple(sorted(spec.items())), collect, has_v)
        fn = self._cache.get(key)
        if fn is not None:
            return fn
        mb = self.microbatch

        def local(x, p, cos, sin):
            n, S, D = x.shape
            xs = x.reshape(n // mb, mb, S, D)

            def body(hs, xb):
                out, acts = _layer_forward(xb, p, cos, sin, spec)
                if collect:
                    hs = tuple(h +
                               _gram(acts[k].reshape(-1, acts[k].shape[-1]))
                               for h, k in zip(hs, _H_ORDER))
                return hs, out

            if collect:
                dims = {
                    "attn_in": D,
                    "o_in": spec["num_heads"] * spec["head_dim"],
                    "mlp_in": D,
                    "down_in": p["mlp.down_proj"].shape[1],
                }
                h0 = tuple(
                    jnp.zeros((dims[k], dims[k]), F32) for k in _H_ORDER)
            else:
                h0 = ()
            hs, outs = lax.scan(body, h0, xs)
            hs = tuple(lax.psum(h, "d") for h in hs)
            return outs.reshape(n, S, D), hs

        sm = _shard_map(local,
                        mesh=self.mesh,
                        in_specs=(P("d"), P(), P(), P()),
                        out_specs=(P("d"), P()),
                        check_vma=False)
        fn = jax.jit(sm, donate_argnums=() if collect else (0, ))
        self._cache[key] = fn
        return fn

    def local_copy(self, arr, device):
        for s in arr.addressable_shards:
            if s.device == device:
                return s.data
        raise ValueError(f"no shard of array on {device}")


# --------------------------------------------------------------------------
# The calibrator
# --------------------------------------------------------------------------


def calibrate_layers_tpu(lm,
                         inps: torch.Tensor,
                         position_embeddings: dict,
                         spec,
                         dev=None,
                         microbatch: int = 4) -> None:
    """Sequential SparseGPT over a Gemma-4 text stack on TPU.

    Same contract as ``gemma4.calibrate_layers_cpu``:

    lm: ``model.model.language_model`` (weights on CPU, bfloat16).
    inps: (nsamples, seqlen, hidden) bfloat16 inputs of layer 0.
    position_embeddings: {layer_type: (cos, sin)} from capture_layer0_inputs.
    spec: a ``sparsity.PruneSpec``.
    dev: ignored (the model stays on CPU; compute runs on the TPU chips).

    Pruned weights are written back into ``lm`` in place (bfloat16, CPU).
    """
    del dev
    cfg = lm.config
    _check_supported(cfg)
    if getattr(spec, "use_lrc", False):
        raise NotImplementedError("LRC is only implemented for the CPU calibrator (sparsity/lrc.py)")
    prunen, prunem = int(spec.prunen), int(spec.prunem)
    blocksize, percdamp = int(spec.blocksize), float(spec.percdamp)
    sparsity = float(getattr(spec, "sparsity", 0.0) or 0.0)
    check_nm(prunen, prunem, blocksize)
    if prunen == 0 and not 0 < sparsity < 1:
        raise ValueError("unstructured mode needs 0 < sparsity < 1")
    prune_only = getattr(spec, "prune_only", "") or ""

    layers = lm.layers
    total = len(layers)
    sparse = spec.layer_set(total)
    if not sparse:
        print(">>> [TPU SPARSEGPT] no layers to prune")
        return
    last = max(sparse)
    ratio = f"{prunen}:{prunem}" if prunem else f"{sparsity * 100:.1f}%"

    nsamples = inps.shape[0]
    run = _Gemma4Runner(nsamples, microbatch)
    print("=" * 85)
    print(f">>> [TPU SPARSEGPT] {ratio} on {len(sparse)} of {total} layers "
          f"(last pruned: {last}); {nsamples} samples over "
          f"{len(run.devices)} TPU chips, microbatch {run.microbatch}")
    print("=" * 85)

    t0 = time.time()
    x = jax.device_put(_to_np(inps), run.shard)
    pe = {
        k: tuple(jax.device_put(_to_np(t[0]), run.rep) for t in v)
        for k, v in position_embeddings.items()
    }
    print(f"    inputs on TPU in {time.time() - t0:.1f}s")
    overall = time.time()

    for i in range(last + 1):
        lt0 = time.time()
        layer = layers[i]
        lspec = _layer_spec(cfg, i)
        cos, sin = pe[lspec["layer_type"]]
        host = _layer_params(layer)
        params = {k: jax.device_put(v, run.rep) for k, v in host.items()}
        has_v = "self_attn.v_proj" in params
        pruned = i in sparse
        tag = f"PRUNING ({ratio} SparseGPT)" if pruned else "DENSE (Preserved)"
        print(f"\n>>> [Layer {i:02d}/{total:02d}] Type: "
              f"{lspec['layer_type']:<18} | Status: {tag}")
        t_fwd1 = t_prune = 0.0
        if pruned:
            t = time.time()
            _, hs = run._pass(lspec, True, has_v)(x, params, cos, sin)
            hs = dict(zip(_H_ORDER, hs))
            jax.block_until_ready(hs)
            t_fwd1 = time.time() - t
            t = time.time()
            names = [n for n in _ATTN + _MLP if n in params]
            if prune_only:
                names = [n for n in names if prune_only in n]
            # Spread the four Hessian groups over the chips; the solves for
            # different chips run concurrently (JAX dispatch is async).
            groups = [g for g in _H_ORDER if any(_H_OF[n] == g for n in names)]
            order = sorted(groups, key=lambda g: -hs[g].shape[0])
            dev_of = {
                g: run.devices[j % len(run.devices)]
                for j, g in enumerate(order)
            }
            scale = 2.0 / nsamples
            hinv = {}
            for g in order:
                Hg = run.local_copy(hs[g], dev_of[g]) * scale
                hinv[g] = prepare_hinv(Hg, percdamp)
            results = {}
            for n in names:
                g = _H_OF[n]
                Wd = jax.device_put(host[n], dev_of[g]).astype(F32)
                results[n] = sparsegpt_solve(Wd,
                                             None,
                                             sparsity,
                                             prunen,
                                             prunem,
                                             blocksize,
                                             percdamp,
                                             hinv=hinv[g],
                                             check=False)
            for g in order:
                _check_ok(hinv[g][2])
            del hs, hinv
            for n in names:
                Wp, loss = results[n]
                wb = np.asarray(Wp.astype(BF16))
                mod = layer.get_submodule(n)
                mod.weight.data = _to_torch(wb).reshape(mod.weight.shape)
                params[n] = jax.device_put(wb, run.rep)
                print(f"    ├─ [{n:<24}] shape {str(list(wb.shape)):<16} "
                      f"error {float(loss):.4f}")
            t_prune = time.time() - t
        t = time.time()
        x, _ = run._pass(lspec, False, has_v)(x, params, cos, sin)
        x.block_until_ready()
        t_fwd2 = time.time() - t
        dt = time.time() - lt0
        el = time.time() - overall
        eta = el / (i + 1) * (last - i)
        print(f"    └─ Layer {i:02d} completed in {dt:.2f}s (H pass "
              f"{t_fwd1:.1f}s, prune {t_prune:.1f}s, forward {t_fwd2:.1f}s) "
              f"(Elapsed: {el / 60:.1f}m | ETA: {eta / 60:.1f}m)")

    if last + 1 < total:
        print(
            f"\n>>> [TPU SPARSEGPT] layers {last + 1}..{total - 1} are dense "
            "and after the last pruned layer; nothing left to calibrate")
    print("\n" + "=" * 85)
    print(
        f">>> [TPU SPARSEGPT] done in {(time.time() - overall) / 60:.2f} min")
    print("=" * 85)


# --------------------------------------------------------------------------
# Child-process entry point (see resolvers/sparsegpt_resolver.py)
# --------------------------------------------------------------------------


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(
        description="Gemma-4 SparseGPT calibration on TPU.")
    ap.add_argument("--json",
                    required=True,
                    help="prune_gemma4 keyword arguments as a JSON object")
    kw = json.loads(ap.parse_args(argv).json)
    if kw.get("sparse_layer_indices") is not None:
        kw["sparse_layer_indices"] = set(kw["sparse_layer_indices"])
    kw.pop("calibrator", None)
    from sparsity import prune_gemma4
    prune_gemma4(**kw, calibrator="tpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
