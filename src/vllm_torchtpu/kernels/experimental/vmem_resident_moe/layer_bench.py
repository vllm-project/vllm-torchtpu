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
"""Stage L1 A/B: EP-sharded MoE block over all devices, resident vs streamed,
with and without the in-kernel all-reduce (the removal arm prices it).

Prints ``CHECK`` (all-reduce arms vs the single-device oracle, per local
device), then one ``RESULT`` line per batch. Multi-process safe: every process
runs the same program; each prints its own lines prefixed with its index.

Example (TPU 8i 4x1x1, 8 TensorCores, Gemma 4 26B-A4B MoE: 128 experts, top-8)::

    python -m vllm_torchtpu.kernels.experimental.vmem_resident_moe.layer_bench \\
        --experts 128 --top-k 8 --batches 1,16,64 --steps 4,16 --vmem-limit-mib 180
"""

import argparse
import json
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.experimental.vmem_resident_moe import kernel as K
from vllm_torchtpu.kernels.experimental.vmem_resident_moe import layer as L

ARMS = (
    ("resident", True),
    ("streamed", True),
    ("resident", False),
    ("streamed", False),
)


def _emit(tag, obj):
    sys.stdout.write(f"P{jax.process_index()} {tag} {json.dumps(obj)}\n")
    sys.stdout.flush()


def _setup(mesh, dims, tokens, steps, top_k, seed):
    sched, x0, gate, ws = K.make_inputs(
        jax.random.PRNGKey(0),
        tokens=tokens,
        steps=steps,
        seed=seed,
        top_k=top_k,
        **dims,
    )
    starts, flat = L.device_schedules(sched, steps, dims["experts"], mesh.size)
    return sched, x0, gate, ws, L.place(mesh, starts, flat, x0, gate, ws), starts


def _publish(metrics):
    """Hand one job result to cdk's result-updater (exos recipes stage it on PATH)."""
    payload = {"exit_code": 0, "bench": "vmempin-l1", "metrics": metrics}
    try:
        subprocess.run(["result-updater", json.dumps(payload)], check=True)
    except (OSError, subprocess.CalledProcessError) as ex:
        sys.stderr.write(f"WARN: result-updater failed: {ex}\n")


def _time(f, args, reps):
    f(*args).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f(*args).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--inter", type=int, default=704)
    p.add_argument("--rows", type=int, default=0, help="0: B rounded up to 16")
    p.add_argument("--ar-dtype", default="bfloat16", choices=["float32", "bfloat16"])
    p.add_argument("--batches", default="1,16,64")
    p.add_argument("--steps", default="4,16")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--vmem-limit-mib", type=int, default=180)
    p.add_argument("--chips", type=int, default=0, help="for per-chip throughput")
    p.add_argument("--check-tokens", type=int, default=16)
    p.add_argument(
        "--point",
        default="",
        help="ARM:B, e.g. resident_ar:16; time one point, publish it",
    )
    a = p.parse_args()

    mesh = jax.make_mesh((jax.device_count(),), (L.AXIS,))
    a.chips = a.chips or jax.device_count() // 2
    base = dict(experts=a.experts, hidden=a.hidden, inter=a.inter)
    ar_dtype = jnp.dtype(a.ar_dtype)

    def dims_for(tokens):
        rows = a.rows or max(16, -(-tokens // 16) * 16)
        return dict(base, rows=rows)

    def common_for(tokens):
        return dict(
            vmem_limit_bytes=a.vmem_limit_mib * 2**20,
            ar_dtype=ar_dtype,
            **dims_for(tokens),
        )

    n_lo, n_hi = (int(v) for v in a.steps.split(","))
    _emit(
        "META",
        {
            "jax": jax.__version__,
            "devices": jax.device_count(),
            "local": jax.local_device_count(),
            "processes": jax.process_count(),
            "experts_per_device": a.experts // jax.device_count(),
        },
    )

    b = a.check_tokens
    check_err = 0.0
    sched, x0, gate, ws, args, _ = _setup(mesh, dims_for(b), b, n_lo, a.top_k, 0)
    ref = np.asarray(K.reference(sched, x0, gate, ws))[:b]
    for w in ("resident", "streamed"):
        f = L.build_layer(mesh, weights=w, allreduce=True, steps=n_lo, **common_for(b))
        out = f(*args)
        errs = [
            float(np.max(np.abs(np.asarray(s.data)[:b] - ref)))
            for s in out.addressable_shards
        ]
        check_err = max(check_err, *errs)
        _emit("CHECK", {"weights": w, "max_abs_err_per_local_device": errs})

    if a.point:
        arm, b = a.point.split(":")
        b = int(b)
        w, ar = arm.split("_")[0], arm.endswith("_ar")
        t, items = {}, {}
        for n in (n_lo, n_hi):
            *_, args, starts = _setup(mesh, dims_for(b), b, n, a.top_k, b)
            f = L.build_layer(mesh, weights=w, allreduce=ar, steps=n, **common_for(b))
            t[n] = _time(f, args, a.reps)
            items[n] = int(starts[:, -1].max())
        us = (t[n_hi] - t[n_lo]) / (n_hi - n_lo) * 1e6
        info = jax.devices()[0]
        metrics = {
            "arm": arm,
            "weights": w,
            "allreduce": ar,
            "B": b,
            "us_per_step": us,
            "tokens_per_s": b / (us * 1e-6),
            "tokens_per_s_per_chip": b / (us * 1e-6) / a.chips,
            "ar_dtype": a.ar_dtype,
            "experts": a.experts,
            "devices": jax.device_count(),
            "experts_per_device": a.experts // jax.device_count(),
            "chips": a.chips,
            "device_kind": getattr(info, "device_kind", ""),
            "top_k": a.top_k,
            "steps": [n_lo, n_hi],
            "raw_s": [t[n_lo], t[n_hi]],
            "max_items_per_device": [items[n_lo], items[n_hi]],
            "check_max_abs_err": check_err,
        }
        _emit("RESULT", metrics)
        if jax.process_index() == 0:
            _publish(metrics)
        return

    for b in (int(v) for v in a.batches.split(",")):
        row = {"B": b, "rows": dims_for(b)["rows"], "ar_dtype": a.ar_dtype}
        for w, ar in ARMS:
            t = {}
            items = {}
            for n in (n_lo, n_hi):
                *_, args, starts = _setup(mesh, dims_for(b), b, n, a.top_k, b)
                f = L.build_layer(
                    mesh, weights=w, allreduce=ar, steps=n, **common_for(b)
                )
                t[n] = _time(f, args, a.reps)
                items[n] = int(starts[:, -1].max())
            name = f"{w}_{'ar' if ar else 'noar'}"
            row[name] = {
                "us_per_step": (t[n_hi] - t[n_lo]) / (n_hi - n_lo) * 1e6,
                "max_items_per_device": [items[n_lo], items[n_hi]],
            }
        r = {k: v["us_per_step"] for k, v in row.items() if isinstance(v, dict)}
        row["allreduce_us_resident"] = r["resident_ar"] - r["resident_noar"]
        row["allreduce_us_streamed"] = r["streamed_ar"] - r["streamed_noar"]
        row["speedup_with_ar"] = r["streamed_ar"] / r["resident_ar"]
        row["speedup_no_ar"] = r["streamed_noar"] / r["resident_noar"]
        _emit("RESULT", row)


if __name__ == "__main__":
    main()
