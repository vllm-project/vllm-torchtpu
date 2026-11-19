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
"""ms per TP all-reduce of bf16 [T, 6144] over 8 TensorCores (tpu7x 2x2x1):
the current path (lax.psum inside shard_map, as the Pallas all-reduce op runs
a prefill chunk) against the fp8-wire variants of
kernels/collectives/fp8_all_reduce.py (TPU_TP_FP8_ALL_REDUCE), plus the bf16
reduce-scatter + all-gather and the int8 wire alone (no quantize) as
references.

Each mode runs --chain all-reduces back to back in one jit (x = ar(x) / 8,
so values stay in range); ms_per_ar = median wall time / chain. SNR (dB) is
of one all-reduce of realistic partials (per rank: Student-t-like values,
log-normal row norms, 8 outlier channels x50, 1/64 of the rows x30) against
the f32 sum. The scheduled HLO's collectives are listed per mode, so the
SparseCore offload shows.

The fp8 module is loaded by file path (--src, default ../src next to this
script), so the image's installed vllm_torchtpu does not matter.
"""

import argparse
import importlib.util
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


def _load_fp8(src):
    path = os.path.join(src, "vllm_torchtpu/kernels/collectives/fp8_all_reduce.py")
    spec = importlib.util.spec_from_file_location("fp8_all_reduce", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _modes(f8, n):
    def wire_only(x):
        # The int8 payload bytes of "a2a" without the quantize / dequantize.
        t, h = x.shape
        p = jax.lax.bitcast_convert_type(x[:, : h // 2], jnp.int8).reshape(t, h)
        p = jnp.concatenate([p, jnp.zeros((t, 256), jnp.int8)], axis=1)
        r = jax.lax.all_to_all(p, AXIS, 0, 0, tiled=True)
        blk = r.reshape(n, t // n, -1).sum(0, dtype=jnp.int8)
        g = jax.lax.all_gather(blk, AXIS, axis=0, tiled=True)[:, :h]
        y = jax.lax.bitcast_convert_type(g.reshape(t, h // 2, 2), x.dtype)
        return jnp.concatenate([y, y], axis=1)

    modes = {
        "psum": lambda x: jax.lax.psum(x, AXIS),
        "rs_ag_bf16": lambda x: jax.lax.all_gather(
            jax.lax.psum_scatter(x, AXIS, scatter_dimension=0, tiled=True),
            AXIS,
            axis=0,
            tiled=True,
        ),
        "wire_only_a2a_ag_int8": wire_only,
    }
    for m in f8.MODES:
        for g in f8.GRANULARITIES:
            modes[f"{m}_{g}"] = lambda x, m=m, g=g: f8.all_reduce(x, AXIS, n, m, g)
    return modes


def _partials(t, h, n):
    def gen(seed):
        k = jax.random.fold_in(jax.random.PRNGKey(0), jax.lax.axis_index(AXIS))
        k1, k2, k3, k4 = jax.random.split(k, 4)
        x = jax.random.t(k1, 4.0, (t, h), jnp.float32)
        x = x * jnp.exp(jax.random.normal(k2, (t, 1)))
        ch = jax.random.choice(k3, h, (8,), replace=False)
        x = x.at[:, ch].multiply(50.0)
        rows = jax.random.choice(k4, t, (max(t // 64, 1),), replace=False)
        x = x.at[rows, :].multiply(30.0)
        return x.astype(jnp.bfloat16) + seed[0, 0].astype(jnp.bfloat16) * 0

    return gen


def _ref_block(t, n):
    """f32 sum of this rank's row block of the partials, from a bf16
    all-to-all (exact data movement: an f32 all-reduce of this size does not
    run under the SparseCore hierarchical all-reduce flags)."""

    def ref(v):
        r = jax.lax.all_to_all(v, AXIS, 0, 0, tiled=True)
        return r.astype(jnp.float32).reshape(n, t // n, -1).sum(0)

    return ref


_COLL = re.compile(
    r"\b(all-reduce|all-gather|reduce-scatter|all-to-all)(-start|-done)?\(", re.I
)


def collectives(hlo_text):
    seen = {}
    for line in hlo_text.splitlines():
        m = _COLL.search(line)
        if not m:
            continue
        key = m.group(1) + (m.group(2) or "")
        if "sparse" in line.lower() or "_sc_" in line.lower():
            key += "[sc]"
        seen[key] = seen.get(key, 0) + 1
    return seen


def _time(fn, args, reps):
    jax.block_until_ready(fn(*args))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src", default=os.path.join(here, "..", "src"))
    ap.add_argument("--tokens", type=int, nargs="+", default=[8192, 4096])
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument("--chain", type=int, default=16)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--modes", nargs="*", default=None)
    ap.add_argument("--tag", default=os.environ.get("BENCH_TAG", ""))
    ap.add_argument("--dump-hlo", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    f8 = _load_fp8(args.src)
    devices = jax.devices()
    n = len(devices)
    print(
        "DEVICES", n, [(d.id, getattr(d, "coords", None)) for d in devices], flush=True
    )
    print("LIBTPU_INIT_ARGS", os.environ.get("LIBTPU_INIT_ARGS", ""), flush=True)
    mesh = Mesh(np.asarray(devices), (AXIS,))
    sh = NamedSharding(mesh, P(AXIS))
    modes = _modes(f8, n)
    names = args.modes or list(modes)
    rows = []

    def smap(body):
        return jax.jit(
            jax.shard_map(
                body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS), check_vma=False
            )
        )

    for t in args.tokens:
        seed = jax.device_put(jnp.zeros((n, 1), jnp.int32), sh)
        x = smap(_partials(t, args.hidden, n))(seed)
        ref = smap(_ref_block(t, n))(x)
        base_ms = None
        for name in names:
            ar = modes[name]

            def chain(v, ar=ar):
                for _ in range(args.chain):
                    v = (ar(v).astype(jnp.float32) * 0.125).astype(v.dtype)
                return v

            def err(v, r, ar=ar):
                # This rank's row block of the result against its f32 sum.
                blk = t // n
                o = jax.lax.dynamic_slice_in_dim(
                    ar(v), jax.lax.axis_index(AXIS) * blk, blk, axis=0
                ).astype(jnp.float32)
                e = jnp.sum((o - r) ** 2)
                s = jnp.sum(r**2)
                return jnp.stack([e, s]).reshape(1, 2)

            row = {"tag": args.tag, "tokens": t, "mode": name}
            try:
                fn = smap(chain).lower(x).compile()
                text = fn.as_text()
                if args.dump_hlo:
                    os.makedirs(args.dump_hlo, exist_ok=True)
                    with open(
                        f"{args.dump_hlo}/{args.tag}_{name}_T{t}.hlo.txt", "w"
                    ) as f:
                        f.write(text)
                sec = _time(fn, (x,), args.reps)
                ms = sec * 1e3 / args.chain
                es = np.asarray(
                    jax.jit(
                        jax.shard_map(
                            err,
                            mesh=mesh,
                            in_specs=(P(AXIS), P(AXIS)),
                            out_specs=P(AXIS),
                            check_vma=False,
                        )
                    )(x, ref)
                )
                snr = None
                if name != "wire_only_a2a_ag_int8":
                    snr = float(
                        10 * np.log10(es[:, 1].sum() / max(es[:, 0].sum(), 1e-30))
                    )
                if name == "psum":
                    base_ms = ms
                row.update(
                    ms_per_ar=ms,
                    snr_db=snr,
                    vs_psum=(ms / base_ms) if base_ms else None,
                    saved_ms_per_step_121=(base_ms - ms) * 121 if base_ms else None,
                    collectives=collectives(text),
                )
            except Exception as e:  # one mode's failure must not end the run
                row["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            rows.append(row)
            print("BENCH", json.dumps(row), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
