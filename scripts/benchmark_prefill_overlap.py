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
"""Does XLA overlap a large TP all-reduce with independent TensorCore work?
(MiniMax M3 TPU_MINIMAX_M3_PREFILL_OVERLAP, tpu7x 2x2x1, 8 TensorCores.)

A stand-in for the post-attention part of an 8192-token MiniMax M3 layer, per
rank: o_proj partial [T, H] -> TP all-reduce -> add + RMSNorm -> FFN partial
(gelu MLP, --ffn columns) -> TP all-reduce, L layers in one jit, bf16, H 6144.

Modes:
  serial        the plain layer on all T rows (today's step).
  tbo           two token halves, issued in the order of
                `MiniMaxM3DecoderLayer.forward_overlap` (AR(oA), AR(oB) |
                FFN(A), AR(fA) | FFN(B), AR(fB) | next layer norm(A)...): the
                torch functional all-reduce path (TPU_MINIMAX_M3_PREFILL_OVERLAP_AR
                =xla), all in the step's computation.
  tbo_call      tbo with every psum inside its own non-inlined jit (a called
                computation, like the psum of the Pallas all-reduce jax_op:
                ..._AR=group). Only an approximation of the torch_tpu custom
                kernel boundary.
  ar_only       the 2 L all-reduces alone (chained).
  compute_only  serial without the all-reduces.

For each mode it prints ms per layer (median of --reps calls of the L-layer
program) and, from the scheduled HLO, every async collective pair (`*-start`
.. `*-done`) with the number of compute instructions (fusion / convolution /
dot / custom-call) the scheduler put between them: 0 everywhere = nothing
overlaps. hidden_ms = serial - tbo; overlap_frac = hidden / min(ar, compute).

Run once per LIBTPU_INIT_ARGS setting; the BENCH/SUMMARY lines carry a
--tag. The flags change only XLA's schedule, so the HLO pairs show directly
which setting lets the all-reduce overlap.
"""

import argparse
import json
import os
import re
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

AXIS = "x"
MODES = ("serial", "tbo", "tbo_call", "ar_only", "compute_only")


def _rms(x, w):
    xf = x.astype(jnp.float32)
    var = jnp.mean(xf * xf, axis=-1, keepdims=True)
    return (xf * jax.lax.rsqrt(var + 1e-6) * (1.0 + w)).astype(x.dtype)


def _psum(x):
    return jax.lax.psum(x, AXIS)


_psum_call = jax.jit(_psum, inline=False)


def _layer_fns(ar):
    def o_proj(a, wo):
        return jnp.dot(a, wo, preferred_element_type=jnp.float32).astype(a.dtype)

    def ffn(x, w1, w2):
        h = jax.nn.gelu(jnp.dot(x, w1, preferred_element_type=jnp.float32))
        return jnp.dot(h.astype(x.dtype), w2, preferred_element_type=jnp.float32).astype(
            x.dtype
        )

    def serial(attn, res, p):
        o = ar(o_proj(attn, p["wo"]))
        res = o + res
        x = _rms(res, p["n2"])
        f = ar(ffn(x, p["w1"], p["w2"]))
        res = f + res
        return _rms(res, p["n1"]), res

    def tbo(attn, res, p):
        half = attn.shape[0] // 2
        ra, rb = res[:half], res[half:]
        oa = ar(o_proj(attn[:half], p["wo"]))
        ob = ar(o_proj(attn[half:], p["wo"]))
        ra = oa + ra
        fa = ar(ffn(_rms(ra, p["n2"]), p["w1"], p["w2"]))
        rb = ob + rb
        fb = ar(ffn(_rms(rb, p["n2"]), p["w1"], p["w2"]))
        ra = fa + ra
        xa = _rms(ra, p["n1"])  # next layer's norm(A): overlaps AR(fB)
        rb = fb + rb
        xb = _rms(rb, p["n1"])
        return jnp.concatenate([xa, xb]), jnp.concatenate([ra, rb])

    return serial, tbo


def _program(mode, layers):
    ar = {"tbo_call": _psum_call, "compute_only": lambda x: x}.get(mode, _psum)
    serial, tbo = _layer_fns(ar)

    def body(attn, res, params):
        if mode == "ar_only":
            x = attn
            for _ in range(2 * layers):
                x = (_psum(x) * 0.125).astype(x.dtype)
            return x
        layer = tbo if mode.startswith("tbo") else serial
        x = attn
        for i in range(layers):
            p = jax.tree.map(lambda w: w[i], params)
            # Attention stand-in: a cheap token-coupled mix of the normed input.
            a = jnp.tanh(x[:, : p["wo"].shape[0]])
            x, res = layer(a, res, p)
        return x + res

    return body


def _params(key, layers, h, q, f, n):
    ks = jax.random.split(key, 3)
    scale = lambda k, shape, fan: (
        jax.random.normal(k, shape, jnp.float32) / np.sqrt(fan) / np.sqrt(n)
    ).astype(jnp.bfloat16)
    return {
        "wo": scale(ks[0], (n, layers, q, h), q),
        "w1": scale(ks[1], (n, layers, h, f), h),
        "w2": scale(ks[2], (n, layers, f, h), f),
        "n1": jnp.zeros((n, layers, h), jnp.float32),
        "n2": jnp.zeros((n, layers, h), jnp.float32),
    }


_HEAVY = re.compile(r"=\s*\S+\s+(fusion|convolution|dot|custom-call)\(")
_START = re.compile(r"^\s*(?:ROOT\s+)?(%?[\w.\-]+)\s*=\s*.*?\b([\w\-]+-start)\(")
_DONE = re.compile(r"^\s*(?:ROOT\s+)?(%?[\w.\-]+)\s*=\s*.*?\b([\w\-]+-done)\((.*)\)")


def schedule_pairs(hlo_text: str) -> list[dict]:
    """Async (`*-start` .. `*-done`) pairs of each computation of a scheduled
    HLO module, in order, with the compute instructions between them."""
    pairs = []
    for comp in re.split(r"\n(?=\S.*\{\s*$)", hlo_text):
        lines = comp.splitlines()
        open_ = {}
        for i, line in enumerate(lines):
            m = _START.match(line)
            if m:
                open_[m.group(1).lstrip("%")] = (i, m.group(2))
                continue
            m = _DONE.match(line)
            if m:
                for name, (j, op) in list(open_.items()):
                    if re.search(rf"%?{re.escape(name)}\b", m.group(3)):
                        heavy = sum(1 for ln in lines[j + 1 : i] if _HEAVY.search(ln))
                        pairs.append({"start": name, "op": op, "between": heavy})
                        del open_[name]
                        break
    return pairs


def _time(fn, args, reps):
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192])
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--attn-cols", type=int, default=768, help="o_proj input cols per rank")
    ap.add_argument("--ffn", type=int, default=4096, help="FFN columns per rank")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--tag", default=os.environ.get("BENCH_TAG", ""))
    ap.add_argument("--dump-hlo", default="", help="directory for the scheduled HLO text")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    devices = jax.devices()
    n = len(devices)
    print(
        "DEVICES", n, [(d.id, getattr(d, "coords", None), getattr(d, "core_on_chip", None)) for d in devices],
        flush=True,
    )
    print("LIBTPU_INIT_ARGS", os.environ.get("LIBTPU_INIT_ARGS", ""), flush=True)
    mesh = Mesh(np.asarray(devices), (AXIS,))
    rows = []
    for t in args.tokens:
        key = jax.random.PRNGKey(0)
        params = _params(key, args.layers, args.hidden, args.attn_cols, args.ffn, n)
        params = jax.tree.map(
            lambda w: jax.device_put(w, NamedSharding(mesh, P(AXIS))), params
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (n * t, args.hidden), jnp.float32)
        x = jax.device_put(x.astype(jnp.bfloat16), NamedSharding(mesh, P(AXIS)))
        res = jax.device_put(
            jnp.zeros((n * t, args.hidden), jnp.bfloat16), NamedSharding(mesh, P(AXIS))
        )
        ms = {}
        ref = None
        for mode in args.modes:
            body = _program(mode, args.layers)

            def per_rank(x, res, params, body=body):
                params = jax.tree.map(lambda w: w[0], params)
                return body(x, res, params)

            fn = jax.jit(
                jax.shard_map(
                    per_rank,
                    mesh=mesh,
                    in_specs=(P(AXIS), P(AXIS), P(AXIS)),
                    out_specs=P(AXIS),
                    check_vma=False,
                )
            )
            compiled = fn.lower(x, res, params).compile()
            text = compiled.as_text()
            if args.dump_hlo:
                os.makedirs(args.dump_hlo, exist_ok=True)
                with open(f"{args.dump_hlo}/{args.tag or 'run'}_{mode}_T{t}.hlo.txt", "w") as f:
                    f.write(text)
            pairs = schedule_pairs(text)
            out = np.asarray(compiled(x, res, params).astype(jnp.float32))
            if mode == "serial":
                ref = out
            rel = (
                float(np.linalg.norm(out - ref) / max(np.linalg.norm(ref), 1e-30))
                if ref is not None and mode in ("tbo", "tbo_call")
                else None
            )
            sec = _time(compiled, (x, res, params), args.reps)
            ms[mode] = sec * 1e3 / args.layers
            row = {
                "tag": args.tag,
                "tokens": t,
                "mode": mode,
                "ms_per_layer": ms[mode],
                "rel_err_vs_serial": rel,
                "async_pairs": len(pairs),
                "pairs_with_overlap": sum(1 for p in pairs if p["between"] > 0),
                "between": [p["between"] for p in pairs],
                "ops": sorted({p["op"] for p in pairs}),
            }
            rows.append(row)
            print("BENCH", json.dumps(row), flush=True)
        if {"serial", "tbo", "ar_only", "compute_only"} <= ms.keys():
            hidden = ms["serial"] - ms["tbo"]
            bound = min(ms["ar_only"], ms["compute_only"])
            summary = {
                "tag": args.tag,
                "tokens": t,
                "serial": ms["serial"],
                "tbo": ms["tbo"],
                "tbo_call": ms.get("tbo_call"),
                "ar_only": ms["ar_only"],
                "compute_only": ms["compute_only"],
                "hidden_ms_per_layer": hidden,
                "overlap_frac": hidden / bound if bound > 0 else None,
                "step_ms_saved_60_layers": hidden * 60,
            }
            rows.append({"summary": summary})
            print("SUMMARY", json.dumps(summary), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
