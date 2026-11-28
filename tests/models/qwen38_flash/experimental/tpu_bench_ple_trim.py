# SPDX-License-Identifier: Apache-2.0
"""On-TPU parity and latency for the VLLM_TPU_MK_PLE_TRIM collectives and projection.

    python tests/models/vllm/tpu_bench_ple_trim.py --tokens 4 16 64

Over all local devices (TP ranks), at the Qwen3.8-Flash decode geometry
(hidden 2560, 4 HC streams, 2560 raw PLE bytes per token):

* merge: the token-embedding all-reduce (BF16 [T, 2560], one owner per row) plus the
  PLE raw all-reduce (u32 [T, 640], one owner per word) vs ONE BF16 all-reduce of
  [2T, 2560] carrying the raw bytes as integer-valued lanes. Parity is bitwise.
* key: the replicated key projection [T, 2560] x [2560, 10240] vs this rank's 1280
  columns plus an all-gather. Parity is bitwise (each column's reduction is the
  same).

Latency: mean of `--iters` chained calls inside one jit (per-call slope).
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

H, HC, RAW = 2560, 4, 2560


def shard_map(f, mesh, in_specs, out_specs):
    return jax.shard_map(
        f, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
    )


def timed(fn, *args, iters):
    out = fn(*args)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.perf_counter() - t0) / iters * 1e6, out


def loop(body, n):
    """`n` dependent calls of `body(carry) -> carry` in one program."""

    def run(*x):
        return jax.lax.fori_loop(0, n, lambda i, c: body(*c), x)

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--chain", type=int, nargs=2, default=[8, 40])
    a = ap.parse_args()
    devs = jax.devices()
    R = len(devs)
    mesh = Mesh(np.array(devs), ("tp",))
    print("devices", R, devs[0].device_kind, flush=True)
    rng = np.random.default_rng(0)
    out = {}
    for T in a.tokens:
        res = {}
        # ---- merge -----------------------------------------------------------
        full = rng.normal(0, 1, (T, H)).astype(np.float32)
        full[0, :7] = -0.0
        full_bf = jnp.asarray(full, jnp.bfloat16)
        owner = rng.integers(0, R, T)
        emb = jnp.stack(
            [jnp.where(jnp.asarray(owner == r)[:, None], full_bf, 0) for r in range(R)]
        )
        raw_full = rng.integers(0, 2**32, (T, RAW // 4), dtype=np.uint64).astype(
            np.uint32
        )
        raw_owner = rng.integers(0, R, (T, 1))
        raw = jnp.stack(
            [jnp.asarray(np.where(raw_owner == r, raw_full, 0)) for r in range(R)]
        )
        spec = P("tp")
        emb = jax.device_put(emb, NamedSharding(mesh, spec))
        raw = jax.device_put(raw, NamedSharding(mesh, spec))

        def separate(e, w):
            return jax.lax.psum(e[0], "tp")[None], jax.lax.psum(w[0], "tp")[None]

        def merged(e, w):
            lanes = jax.lax.bitcast_convert_type(w[0], jnp.uint8).reshape(-1, H)
            lanes = lanes.astype(jnp.bfloat16)
            red = jax.lax.psum(jnp.concatenate([e[0], lanes]), "tp")
            back = red[T:].astype(jnp.uint8).reshape(T, RAW // 4, 4)  # noqa: B023
            return red[:T][None], jax.lax.bitcast_convert_type(back, jnp.uint32)[None]  # noqa: B023

        sm = lambda f: shard_map(f, mesh, (spec, spec), (spec, spec))  # noqa: B023, E731
        e_s, w_s = jax.jit(sm(separate))(emb, raw)
        e_m, w_m = jax.jit(sm(merged))(emb, raw)
        res["merge_bitwise"] = bool(
            np.array_equal(
                np.asarray(e_s).view(np.uint16), np.asarray(e_m).view(np.uint16)
            )
            and np.array_equal(np.asarray(w_s), np.asarray(w_m))
            and np.array_equal(np.asarray(w_m)[0], raw_full)
        )
        for name, f in (("separate", separate), ("merged", merged)):
            ts = []
            for n in a.chain:
                fn = jax.jit(sm(loop(f, n)))
                ts.append(timed(fn, emb, raw, iters=a.iters)[0])
            res[f"merge_{name}_us"] = round(
                (ts[1] - ts[0]) / (a.chain[1] - a.chain[0]), 2
            )
        # ---- key projection -------------------------------------------------
        x = jnp.asarray(rng.normal(0, 1, (T, H)), jnp.bfloat16)
        w = jnp.asarray(rng.normal(0, 0.02, (H, HC * H)), jnp.bfloat16)
        xs = jax.device_put(jnp.broadcast_to(x, (R, T, H)), NamedSharding(mesh, spec))
        w_rep = jax.device_put(
            jnp.broadcast_to(w, (R, H, HC * H)), NamedSharding(mesh, spec)
        )
        cols = HC * H // R
        w_sh = jax.device_put(
            w.reshape(H, R, cols).transpose(1, 0, 2), NamedSharding(mesh, spec)
        )

        def rep(xx, ww):
            return (
                jnp.dot(xx[0], ww[0], preferred_element_type=jnp.float32).astype(
                    jnp.bfloat16
                )[None],
                ww,
            )

        def col(xx, ww):
            loc = jnp.dot(xx[0], ww[0], preferred_element_type=jnp.float32).astype(
                jnp.bfloat16
            )
            return jax.lax.all_gather(loc, "tp", axis=1, tiled=True)[None], ww

        k_r = jax.jit(shard_map(rep, mesh, (spec, spec), (spec, spec)))(xs, w_rep)[0]
        k_c = jax.jit(shard_map(col, mesh, (spec, spec), (spec, spec)))(xs, w_sh)[0]
        res["key_bitwise"] = bool(
            np.array_equal(
                np.asarray(k_r).view(np.uint16), np.asarray(k_c).view(np.uint16)
            )
        )

        def chain_x(f):
            def body(xx, ww):
                k, ww = f(xx, ww)
                # feed back so calls stay dependent; cheap slice keeps shape [1, T, H]
                return (k[:, :, :H] * 0 + xx).astype(xx.dtype), ww

            return body

        for name, f, ww in (("replicated", rep, w_rep), ("colshard", col, w_sh)):
            ts = []
            for n in a.chain:
                fn = jax.jit(
                    shard_map(loop(chain_x(f), n), mesh, (spec, spec), (spec, spec))
                )
                ts.append(timed(fn, xs, ww, iters=a.iters)[0])
            res[f"key_{name}_us"] = round(
                (ts[1] - ts[0]) / (a.chain[1] - a.chain[0]), 2
            )
        out[T] = res
        print("RESULT", T, json.dumps(res), flush=True)
    print("SUMMARY", json.dumps(out), flush=True)
    # key_bitwise is informational: the column-sharded projection is NOT bitwise on
    # v7x (the MXU reduction differs with N), so VLLM_TPU_MK_PLE_TRIM does not use it.
    assert all(r["merge_bitwise"] for r in out.values()), out


if __name__ == "__main__":
    main()
