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
from vllm_torchtpu.kernels.experimental.vmem_resident_moe import percall as P


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


def _runner(arm, sched, steps, common):
    """(callable, args-builder) for any arm, in-kernel or per-call."""
    if arm in K.ARMS:
        f = K.build(arm, items=sched.items, **common)
        return lambda x0, gate, ws: f(sched, x0, gate, ws)
    count, order = P.step_schedule(sched, steps, common["experts"])
    f = P.build_percall(arm, **common)
    return lambda x0, gate, ws: f(count, order, x0, gate, ws)


def _hlo_scan(arm, sched, steps, common, x0, gate, ws, hlo_dir):
    count, order = P.step_schedule(sched, steps, common["experts"])
    f = P.build_percall(arm, **common)
    text = f.lower(count, order, x0, gate, ws).compile().as_text()
    if hlo_dir:
        path = f"{hlo_dir}/{arm}_e{common['experts']}.hlo.txt"
        with open(path, "w") as fh:
            fh.write(text)
    dims = (common["experts"], common["hidden"], common["inter"])
    return P.weight_copies_in_loop(text, *dims)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--hidden", type=int, default=2816)
    p.add_argument("--inter", type=int, default=704)
    p.add_argument("--rows", type=int, default=0, help="0: B rounded up to 32")
    p.add_argument(
        "--per-item-quant",
        action="store_true",
        help="quantize the input per expert (the pre-hoist kernel)",
    )
    p.add_argument("--batches", default="1,4,16,64")
    p.add_argument("--steps", default="4,16", help="two step counts for the slope")
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--vmem-limit-mib", type=int, default=61)
    p.add_argument("--check-tokens", type=int, default=16)
    p.add_argument("--interpret", action="store_true")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--hlo-dir", default="", help="write compiled HLO of per-call arms")
    p.add_argument(
        "--arms",
        default="resident,streamed",
        help=f"comma list from {K.ARMS + P.PERCALL_ARMS}",
    )
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

    def dims_for(tokens):
        rows = a.rows or max(32, -(-tokens // 32) * 32)
        return dict(experts=a.experts, hidden=a.hidden, inter=a.inter, rows=rows)

    def common_for(tokens):
        return dict(
            vmem_limit_bytes=a.vmem_limit_mib * 2**20,
            interpret=a.interpret,
            hoist=not a.per_item_quant,
            **dims_for(tokens),
        )

    dims, common = dims_for(a.check_tokens), common_for(a.check_tokens)
    n_lo, n_hi = (int(v) for v in a.steps.split(","))
    arms = a.arms.split(",")
    key = jax.random.PRNGKey(0)

    b = a.check_tokens
    sched, x0, gate, ws = K.make_inputs(key, tokens=b, steps=n_lo, **dims)
    ref = K.reference(sched, x0, gate, ws)
    failed = set()
    for arm in arms:
        try:
            out = _runner(arm, sched, n_lo, common)(x0, gate, ws)
        except Exception as ex:  # noqa: BLE001  report and keep the other arms
            failed.add(arm)
            _emit("ARMERROR", json.dumps({"arm": arm, "error": repr(ex)[:600]}))
            continue
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
    if not a.interpret:
        for arm in (x for x in arms if x in P.PERCALL_ARMS and x not in failed):
            scan = _hlo_scan(arm, sched, n_lo, common, x0, gate, ws, a.hlo_dir)
            _emit("HLO", json.dumps({"arm": arm, **scan}))
    if a.check_only:
        return

    arms = [x for x in arms if x not in failed]
    for b in (int(v) for v in a.batches.split(",")):
        row = {
            "B": b,
            "experts": a.experts,
            "rows": dims_for(b)["rows"],
            "hoist": not a.per_item_quant,
        }
        for arm in arms:
            meas = {}
            for n in (n_lo, n_hi):
                sched, x0, gate, ws = K.make_inputs(
                    key, tokens=b, steps=n, seed=b, **dims_for(b)
                )
                f = _runner(arm, sched, n, common_for(b))
                meas[n] = (*_time(f, (x0, gate, ws), a.reps), sched.items)
            (m0, mn0, t0), (m1, mn1, t1) = meas[n_lo], meas[n_hi]
            slope = (m1 - m0) / (n_hi - n_lo)
            row[arm] = {
                "us_per_step": slope * 1e6,
                "us_per_step_min": (mn1 - mn0) / (n_hi - n_lo) * 1e6,
                "us_per_expert_item": (m1 - m0) / (t1 - t0) * 1e6,
                "fixed_us": (m0 - slope * n_lo) * 1e6,
                "items": [t0, t1],
            }
        if "streamed" in row:
            for arm in arms:
                row[arm]["speedup_vs_streamed"] = (
                    row["streamed"]["us_per_step"] / row[arm]["us_per_step"]
                )
        _emit("RESULT", json.dumps(row))


if __name__ == "__main__":
    main()
