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
"""Resident-vs-streamed A/B on one TensorCore.

Prints ``META``, ``CHECK`` (both arms vs the oracle), ``FAULT`` (a corrupted
expert the oracle must reject) and one ``RESULT`` JSON line per batch. Per-step
time is the slope between two step counts, which removes launch cost and the
resident arm's one-time weight load.

Example (Gemma 4 26B-A4B expert shapes, 8 local experts on v7x)::

    LIBTPU_INIT_ARGS=--xla_tpu_scoped_vmem_limit_kib=63488 \\
    python -m vllm_torchtpu.kernels.experimental.vmem_resident_moe.benchmark \\
        --experts 8 --hidden 2816 --inter 704 --rows 64 --steps 4,16 \\
        --batches 1,4,16,64 --vmem-limit-mib 61
"""

import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.vmem_resident_moe import kernel as K


def _emit(*parts):
    sys.stdout.write(" ".join(parts) + "\n")
    sys.stdout.flush()


def _time(fn, args, reps):
    fn(*args).block_until_ready()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args).block_until_ready()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), float(np.min(ts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--inter", type=int, default=704)
    p.add_argument("--rows", type=int, default=64, help="padded token rows")
    p.add_argument("--batches", default="1,4,16,64")
    p.add_argument("--steps", default="4,16", help="two step counts for the slope")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--vmem-limit-mib", type=int, default=61)
    p.add_argument("--check-tokens", type=int, default=16)
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--check-only", action="store_true")
    a = p.parse_args()

    info = {}
    if not a.interpret and hasattr(pltpu, "get_tpu_info"):
        info = {k: str(v) for k, v in vars(pltpu.get_tpu_info()).items()}
    _emit(
        "META",
        json.dumps(
            {"jax": jax.__version__, "device": str(jax.devices()[0]), "tpu_info": info}
        ),
    )

    dims = dict(experts=a.experts, hidden=a.hidden, inter=a.inter, rows=a.rows)
    common = dict(
        vmem_limit_bytes=a.vmem_limit_mib * 2**20, interpret=a.interpret, **dims
    )
    n_lo, n_hi = (int(v) for v in a.steps.split(","))
    key = jax.random.PRNGKey(0)

    b = a.check_tokens
    sched, x0, gate, ws = K.make_inputs(key, tokens=b, steps=n_lo, **dims)
    ref = K.reference(sched, x0, gate, ws)
    for arm in K.ARMS:
        out = K.build(arm, items=sched.items, **common)(sched, x0, gate, ws)
        err = float(jnp.max(jnp.abs(out[:b] - ref[:b])))
        _emit(
            "CHECK",
            json.dumps(
                {
                    "arm": arm,
                    "max_abs_err": err,
                    "ref_absmax": float(jnp.max(jnp.abs(ref[:b]))),
                }
            ),
        )
    e0 = int(sched.expert[0])
    bad = list(ws)
    bad[2] = ws[2].at[e0].set(ws[2][(e0 + 1) % a.experts])
    out = K.build("resident", items=sched.items, **common)(sched, x0, gate, tuple(bad))
    _emit(
        "FAULT",
        json.dumps(
            {"max_abs_err_with_fault": float(jnp.max(jnp.abs(out[:b] - ref[:b])))}
        ),
    )
    if a.check_only:
        return

    for b in (int(v) for v in a.batches.split(",")):
        row = {"B": b, "experts": a.experts}
        for arm in K.ARMS:
            meas = {}
            for n in (n_lo, n_hi):
                sched, x0, gate, ws = K.make_inputs(
                    key, tokens=b, steps=n, seed=b, **dims
                )
                f = K.build(arm, items=sched.items, **common)
                meas[n] = (*_time(f, (sched, x0, gate, ws), a.reps), sched.items)
            (m0, mn0, t0), (m1, mn1, t1) = meas[n_lo], meas[n_hi]
            slope = (m1 - m0) / (n_hi - n_lo)
            row[arm] = {
                "us_per_step": slope * 1e6,
                "us_per_step_min": (mn1 - mn0) / (n_hi - n_lo) * 1e6,
                "us_per_expert_item": (m1 - m0) / (t1 - t0) * 1e6,
                "fixed_us": (m0 - slope * n_lo) * 1e6,
                "items": [t0, t1],
            }
        row["speedup"] = row["streamed"]["us_per_step"] / row["resident"]["us_per_step"]
        _emit("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
