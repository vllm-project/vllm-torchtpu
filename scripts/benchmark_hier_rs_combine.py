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
"""MoE combine reduce-scatter on tpu7x 2x2x1 (8 TensorCores): XLA vs hierrs_tc.

Per device the input is the step's [T, hidden] bf16 partial FFN output; each
device gets its [T / 8, hidden] block of the sum (MiniMax-M3 minimax_sp_mode
"full": `tensor_model_parallel_reduce_scatter(ffn(normed), 0)`).

Modes (per T):
  xla_sorted / xla_rank   jax.lax.psum_scatter over the device-id-ordered /
                          vLLM rank-ordered mesh (what the group collective
                          lowers to; LIBTPU_INIT_ARGS decides whether XLA offloads
                          it to the SparseCores).
  hier_sorted             hierrs_tc bf16 wire, device-id-ordered mesh (upstream).
  hier_rank               hierrs_tc bf16, rank-ordered mesh + chip_perm (served
                          with DCP on: the EP mesh is rank-ordered).
  hier_rank_noperm        same mesh without chip_perm (one hypercube step goes
                          to the diagonal chip): shows what the permutation buys.
  hier_fp8_s<scale>       hierrs_tc fp8 wire, rank mesh + chip_perm, static scale
                          (engages at >= 16 MiB per device, i.e. T >= 1366 at
                          hidden 6144; smaller T keeps the bf16 wire).
  hier_fp8f_s<scale>      fp8 forced at every T (FP8_MIN_BYTES = 0).

Timing: a fori_loop of n calls, each writing its output back into the input
rows (scaled by 1/8, so values stay bounded); per-call time = slope between
n_lo and n_hi calls, median of --reps. Numerics: one call on a fresh normal
input (std --data-std) against the f32 reference sum and against XLA's bf16
reduce-scatter: relative L2 error, SNR dB, max abs error.

Run the script once per LIBTPU_INIT_ARGS setting (SparseCore reduce-scatter
offload off / on); its BENCH lines carry `sc_rs` from the environment.
"""

import argparse
import json
import os
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.hierrs_tc import dispatch

AXIS = "x"
N = 8


def _chips_by_coords(devices):
    """Device ids per physical chip, chips sorted by ICI coordinates (as
    distributed/chip_topology.py), or None without coords."""
    grouped = {}
    for d in devices:
        coords = getattr(d, "coords", None)
        if coords is None:
            return None
        grouped.setdefault(tuple(coords), []).append(int(d.id))
    return [sorted(grouped[k]) for k in sorted(grouped)]


def _chip_perm(mesh_ids, chips):
    from types import SimpleNamespace

    from vllm_torchtpu.distributed.hier_reduce_scatter import _chip_perm

    mesh = SimpleNamespace(
        devices=np.asarray([SimpleNamespace(id=i) for i in mesh_ids], dtype=object)
    )
    perm, reason = _chip_perm(mesh, chips)
    if reason:
        raise SystemExit(f"chip_perm refused: {reason}")
    return perm


def _local_fn(mode, mesh_ids, chips, fp8_scale):
    if mode.startswith("xla"):
        return lambda part: jax.lax.psum_scatter(
            part, AXIS, scatter_dimension=0, tiled=True
        )
    perm = None
    if mode in ("hier_rank",) or mode.startswith("hier_fp8"):
        perm = _chip_perm(mesh_ids, chips)
    fp8 = mode.startswith("hier_fp8")
    fn = dispatch.make_reduce_scatter_fn(
        None,
        tuple(range(N)),
        axis_name=AXIS,
        fp8_comm=fp8,
        fp8_static_scale=fp8_scale if fp8 else None,
        chip_perm=perm,
    )
    return fn.local


def _mesh_ids(mode, rank_ids):
    if mode.endswith("_sorted"):
        return tuple(range(N))
    return rank_ids


def _loop(local, n):
    def body(part):
        def step(_, x):
            out = local(x)
            return jax.lax.dynamic_update_slice_in_dim(
                x, (out * 0.125).astype(x.dtype), 0, axis=0
            )

        return jax.lax.fori_loop(0, n, step, part)

    return body


def _time(fn, x, reps):
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _snr_db(want, err):
    p_sig = float(np.mean(np.square(want, dtype=np.float64)))
    p_err = float(np.mean(np.square(err, dtype=np.float64)))
    return float("inf") if p_err == 0 else 10 * np.log10(p_sig / p_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="384,512,1024,2048,4096")
    ap.add_argument("--hidden", type=int, default=6144)
    ap.add_argument(
        "--modes",
        default="xla_sorted,xla_rank,hier_sorted,hier_rank,hier_rank_noperm,hier_fp8,hier_fp8f",
    )
    ap.add_argument("--fp8-scales", default="4,16")
    ap.add_argument("--rank-device-ids", default="0,1,4,5,6,7,2,3")
    ap.add_argument("--n-lo", type=int, default=4)
    ap.add_argument("--n-hi", type=int, default=24)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--data-std", type=float, default=1.0)
    ap.add_argument("--layers", type=int, default=60)
    ap.add_argument("--json", default="")
    ap.add_argument(
        "--interpret",
        action="store_true",
        help="CPU smoke test: Pallas interpret mode over 8 host devices "
        "(set XLA_FLAGS=--xla_force_host_platform_device_count=8); timings "
        "are meaningless.",
    )
    args = ap.parse_args()
    if args.interpret:
        from jax._src import tpu_info
        from jax.experimental import pallas as pl
        from jax.experimental.pallas import tpu as pltpu

        real = pl.pallas_call

        def call(*a, **kw):
            kw.pop("compiler_params", None)
            kw["interpret"] = pltpu.InterpretParams()
            return real(*a, **kw)

        pl.pallas_call = call
        tpu_info.get_device_kind = lambda: "TPU7x"

    libtpu = os.environ.get("LIBTPU_INIT_ARGS", "")
    sc_rs = "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true" in libtpu
    devices = {int(d.id): d for d in jax.devices()}
    assert len(devices) >= N, f"need {N} devices, got {len(devices)}"
    chips = _chips_by_coords(devices.values())
    if chips is None and args.interpret:
        chips = [[2 * k, 2 * k + 1] for k in range(N // 2)]
    rank_ids = tuple(int(i) for i in args.rank_device_ids.split(","))
    print(
        "DEVICES "
        + json.dumps(
            {
                "ids_coords_core": [
                    (i, list(getattr(d, "coords", [])), getattr(d, "core_on_chip", None))
                    for i, d in sorted(devices.items())
                ],
                "chips_by_coords": chips,
                "rank_mesh_ids": rank_ids,
                "rank_mesh_chip_perm": _chip_perm(rank_ids, chips),
                "sc_rs": sc_rs,
                "libtpu": libtpu,
                "jax": jax.__version__,
            }
        ),
        flush=True,
    )

    modes = []
    for m in args.modes.split(","):
        if m in ("hier_fp8", "hier_fp8f"):
            modes += [(f"{m}_s{s}", float(s)) for s in args.fp8_scales.split(",")]
        else:
            modes.append((m, None))

    records = []
    real_fp8_min = dispatch.FP8_MIN_BYTES
    for t in (int(v) for v in args.tokens.split(",")):
        rng = np.random.default_rng(t)
        # per_dev[i]: mesh index i's partial (all modes reduce rank order =
        # mesh order, identity `order`).
        per_dev = (
            rng.standard_normal((N, t, args.hidden), dtype=np.float32) * args.data_std
        )
        per_dev = np.asarray(jnp.asarray(per_dev, jnp.bfloat16), np.float32)
        want = per_dev.sum(0).reshape(N, t // N, args.hidden)  # block i -> mesh i
        ref_xla = None
        for mode, scale in modes:
            dispatch.FP8_MIN_BYTES = 0 if mode.startswith("hier_fp8f") else real_fp8_min
            mesh_ids = _mesh_ids(mode, rank_ids)
            mesh = Mesh(np.asarray([devices[i] for i in mesh_ids]), (AXIS,))
            sharding = NamedSharding(mesh, P(AXIS, None))
            rec = {"T": t, "hidden": args.hidden, "mode": mode, "sc_rs": sc_rs}
            try:
                local = _local_fn(mode, mesh_ids, chips, scale)
                x = jax.device_put(
                    jnp.asarray(per_dev.reshape(N * t, args.hidden), jnp.bfloat16),
                    sharding,
                )
                one = jax.jit(
                    jax.shard_map(
                        local, mesh=mesh, in_specs=P(AXIS, None),
                        out_specs=P(AXIS, None), check_vma=False,
                    )
                )
                got = np.asarray(one(x), np.float32).reshape(N, t // N, args.hidden)
                err = got - want
                rec["rel_l2_vs_f32"] = float(np.linalg.norm(err) / np.linalg.norm(want))
                rec["snr_db_vs_f32"] = _snr_db(want, err)
                rec["max_abs_err"] = float(np.abs(err).max())
                if mode == "xla_sorted":
                    ref_xla = got
                if ref_xla is not None:
                    d = got - ref_xla
                    rec["rel_l2_vs_xla"] = float(
                        np.linalg.norm(d) / np.linalg.norm(ref_xla)
                    )
                    rec["bitwise_vs_xla"] = bool(np.array_equal(got, ref_xla))
                fns = {
                    n: jax.jit(
                        jax.shard_map(
                            _loop(local, n), mesh=mesh, in_specs=P(AXIS, None),
                            out_specs=P(AXIS, None), check_vma=False,
                        )
                    )
                    for n in (args.n_lo, args.n_hi)
                }
                lo = _time(fns[args.n_lo], x, args.reps)
                hi = _time(fns[args.n_hi], x, args.reps)
                rec["us_per_call"] = (hi - lo) / (args.n_hi - args.n_lo) * 1e6
                rec["bytes_per_dev_MiB"] = t * args.hidden * 2 / 2**20
            except Exception as exc:  # noqa: BLE001 - report and continue
                rec["error"] = repr(exc)[:600]
            records.append(rec)
            print("BENCH " + json.dumps(rec), flush=True)
    dispatch.FP8_MIN_BYTES = real_fp8_min

    print(f"\nSUMMARY (sc_rs={sc_rs}; us per call; ms/step = x {args.layers} layers)")
    by_t = {}
    for r in records:
        by_t.setdefault(r["T"], {})[r["mode"]] = r
    for t, rs in by_t.items():
        base = rs.get("xla_rank", {}).get("us_per_call")
        line = [f"T={t:5d}"]
        for mode, r in rs.items():
            us = r.get("us_per_call")
            if us is None:
                line.append(f"{mode}=ERR")
                continue
            extra = f" ({(us - base) * args.layers / 1e3:+.2f} ms/step)" if base else ""
            line.append(f"{mode}={us:.1f}{extra} snr={r.get('snr_db_vs_f32', 0):.0f}dB")
        print("  " + " | ".join(line), flush=True)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(records, f, indent=1)


if __name__ == "__main__":
    main()
