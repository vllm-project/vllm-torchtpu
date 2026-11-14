#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Low-latency all-reduce microbenchmark for TPU.

Measures per-op all-reduce latency at decode-sized payloads, comparing XLA's
default collective (what ``torch_tpu`` / vLLM emit for a TP all-reduce) against
the custom Pallas kernels in
``vllm_torchtpu.kernels.collectives.low_latency_all_reduce``.

Adapted from AI-Hypercomputer/accelerator-microbenchmarks ``collectives.py``,
with two changes aimed at the small-payload regime:

* **Two-point slope timing.** The reference harness times one dispatch per
  measurement, so host dispatch and launch overhead (~10us here) swamp a 3us
  collective.  Instead we time a *chain* of ``n_lo`` and then ``n_hi``
  data-dependent all-reduces in a single jitted call and take the slope
  ``(T(n_hi) - T(n_lo)) / (n_hi - n_lo)``.  Every fixed cost -- dispatch,
  transfer-in, transfer-out, host sync -- cancels exactly, leaving the
  steady-state back-to-back latency of one collective.  Chaining also matches
  how TP all-reduce actually appears in decode: on the critical path, one per
  layer, nothing to overlap with.
* **Small-payload sweep.** Sizes start at 2KB rather than MB.

Usage:
    python scripts/allreduce_latency_bench.py                     # 8 devices
    python scripts/allreduce_latency_bench.py --devices 2,4,8
    python scripts/allreduce_latency_bench.py --csv out.csv
"""

import argparse
import os
import sys
import time


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--devices",
                   default="8",
                   help="Comma-separated mesh sizes to sweep, e.g. '2,4,8'.")
    p.add_argument("--dtype",
                   default="bfloat16",
                   choices=["bfloat16", "float32"])
    p.add_argument(
        "--cols",
        type=int,
        default=1024,
        help="Trailing (lane) dim of the payload. Must be a multiple "
        "of 128.")
    # Empty by default so the sweep can follow the device count: the workspace
    # kernels split the leading dim across the mesh and refuse
    # rows % num_devices != 0, so a fixed list starting at 1 leaves their
    # columns blank for the first several rows of the table and reads as if the
    # kernel had failed rather than been skipped.
    p.add_argument(
        "--rows",
        default="",
        help="Comma-separated leading dims to sweep, each a multiple of the "
        "mesh size. Payload bytes = rows * cols * itemsize. Default: 1x "
        "through 256x the device count.")
    p.add_argument(
        "--shapes",
        default=None,
        help="Explicit 'ROWSxCOLS' payloads, e.g. '2x3584,2x7168'. Overrides "
        "--rows/--cols. Use this to measure a model's actual shapes rather "
        "than byte-equivalents at a fixed cols: the kernel grids over rows, "
        "so (2,3584) and (16,1024) are the same 14/16 KiB but not the same "
        "amount of parallelism.")
    p.add_argument("--impls",
                   default="xla,twoshot,twoshot_nb",
                   help="Comma-separated subset of xla, twoshot, "
                   "twoshot_nb, twoshot_rs.")
    # No --profile-dir here: jax.profiler.trace segfaults in this venv
    # ("Unexpected PLUGIN_Profiler_Api size: expected 80, got 104" -- libtpu
    # 0.0.44.1 against xprof 2.22.3). For device self time per op, which is
    # what a production trace reports, use
    # scripts/debug/profile_ll_allreduce_shapes.py: it goes through
    # torch.profiler + TpuProfilerConfig, the path production uses.
    p.add_argument("--iters-lo", type=int, default=20)
    p.add_argument("--iters-hi", type=int, default=60)
    p.add_argument("--repeats",
                   type=int,
                   default=25,
                   help="Timed repeats per measurement; the median is used.")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--multihost",
                   action="store_true",
                   help="Call jax.distributed.initialize() so this process "
                   "joins a multi-process slice. Auto-enabled under a "
                   "multi-task SLURM step.")
    p.add_argument("--csv",
                   default=None,
                   help="Optional path to write results.")
    p.add_argument("--sc-offload-bytes",
                   type=int,
                   default=None,
                   help="Set --xla_tpu_sparse_core_{all_reduce,all_gather}_"
                   "offload_min_size_in_bytes. Below this size collectives "
                   "stay on TensorCore; above it they are offloaded to "
                   "SparseCore. Pass 0 to force SparseCore offload, or a "
                   "large value to force TensorCore. Must be set before "
                   "JAX initializes the TPU.")
    return p.parse_args(argv)


ARGS = _parse_args()

# LIBTPU_INIT_ARGS is consumed at TPU-client construction, so it has to be set
# before anything imports/initializes JAX's TPU backend.
if ARGS.sc_offload_bytes is not None:
    _extra = " ".join(
        f"--xla_tpu_sparse_core_{name}_offload_min_size_in_bytes="
        f"{ARGS.sc_offload_bytes}" for name in ("all_reduce", "all_gather"))
    os.environ["LIBTPU_INIT_ARGS"] = (
        _extra + " " + os.environ.get("LIBTPU_INIT_ARGS", "")).strip()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import mesh_utils, multihost_utils  # noqa: E402
from jax.experimental.shard_map import shard_map  # noqa: E402
from jax.sharding import Mesh, NamedSharding  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

# Must happen before anything touches a device. JAX reads SLURM_STEP_NODELIST /
# SLURM_NTASKS / SLURM_PROCID / SLURM_LOCALID on its own, so no args needed.
if ARGS.multihost or int(os.environ.get("SLURM_NTASKS", "1")) > 1:
    jax.distributed.initialize()

IS_LEAD = jax.process_index() == 0


def rprint(*a, **kw):
    """Print from the lead process only."""
    if IS_LEAD:
        print(*a, **kw)


def _load_kernels():
    """Load the kernel module straight from this checkout, by path.

    Deliberately bypasses ``import vllm_torchtpu``: the package ``__init__``
    pulls in vLLM and, more importantly, ``env_override`` injects
    ``LIBTPU_INIT_ARGS`` (including the SparseCore all-reduce offload
    thresholds) at import time, which would silently change the very baseline
    we are trying to measure.  The kernel module itself only needs JAX.
    """
    import importlib.util
    path = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))) +
            "/src/vllm_torchtpu/kernels/collectives/low_latency_all_reduce.py")
    spec = importlib.util.spec_from_file_location("low_latency_all_reduce",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


llar = _load_kernels()  # noqa: E402

AXIS = "x"


def build_mesh(num_devices: int) -> Mesh:
    """1-D mesh in a topology-aware device order."""
    devices = jax.devices()
    if num_devices > len(devices):
        raise ValueError(
            f"asked for {num_devices} devices, only {len(devices)} present")
    mesh_devices = mesh_utils.create_device_mesh((num_devices, ),
                                                 devices=devices[:num_devices])
    return Mesh(mesh_devices, (AXIS, ))


def make_op(impl: str, num_devices: int):
    """Returns a fn mapping a local shard to its all-reduced value.

    Only the XLA reference is built here. Every kernel arm is a workspace
    kernel and is built in make_chain instead, because the workspace has to be
    threaded across the whole chain rather than created per op.
    """
    if impl == "xla":
        return lambda v: jax.lax.psum(v, axis_name=AXIS)
    raise ValueError(f"unknown impl {impl!r}")


def make_chain(impl: str, mesh: Mesh, num_devices: int, n_ops: int):
    """Jitted chain of ``n_ops`` data-dependent all-reduces.

    Each all-reduce is scaled by 1/N (exact in binary, so it neither perturbs
    the timing nor lets values overflow across the chain) and feeds the next,
    which stops XLA from CSE-ing or reordering them.
    """
    scale = 1.0 / num_devices

    # *_nb drop the steady-state barrier, keeping only round 0's. The recv wait
    # is all-to-all, so being in round k already proves every peer finished the
    # round that last read this slot; see the kernel docstring for the full
    # argument and its precondition (every rank calling the collective the same
    # number of times in the same order). The per-round check below is a
    # tolerance-scaled comparison against psum, not a proof; treat these arms as
    # measurements of a configuration whose safety the caller must establish.
    if impl == "twoshot_rs":
        # (1, numel) in, reshaped to (n, numel/n) inside the loop and back out
        # again -- exactly _build_two_shot's body. The reshape is not a bitcast:
        # a 2-D bf16 array is tiled (16, 128), so (1, 10752) is lane-aligned but
        # sublane-padded 1->16, while (32, 336) has clean sublanes but pads lanes
        # 336->384. Different tiled layouts means a real relayout, and this arm
        # is what prices it.
        def body(local_x):
            v = local_x[0]
            rows, cols = v.shape
            numel = rows * cols
            inner = numel // num_devices
            rs, ag, step = llar.make_two_shot_workspace((num_devices, inner),
                                                        v.dtype, num_devices)
            for _ in range(n_ops):
                flat = v.reshape(num_devices, inner)
                out, rs, ag, step = llar.two_shot_all_reduce(
                    flat,
                    rs,
                    ag,
                    step,
                    axis_name=AXIS,
                    num_devices=num_devices)
                v = out.reshape(v.shape) * scale
            return v[None]
    elif impl in ("twoshot", "twoshot_nb"):
        steady = impl == "twoshot"

        def body(local_x):
            v = local_x[0]
            rs, ag, step = llar.make_two_shot_workspace(
                v.shape, v.dtype, num_devices)
            for _ in range(n_ops):
                v, rs, ag, step = llar.two_shot_all_reduce(
                    v,
                    rs,
                    ag,
                    step,
                    axis_name=AXIS,
                    num_devices=num_devices,
                    steady_barrier=steady)
                v = v * scale
            return v[None]
    else:
        op = make_op(impl, num_devices)

        def body(local_x):
            v = local_x[0]  # (1, rows, cols) -> (rows, cols)
            for _ in range(n_ops):
                v = op(v) * scale
            return v[None]

    spec = P(AXIS, None, None)
    return jax.jit(
        shard_map(body,
                  mesh=mesh,
                  in_specs=spec,
                  out_specs=spec,
                  check_rep=False))


def make_input(num_devices: int, rows: int, cols: int, dtype, sharding):
    """Build the globally-sharded input on device.

    Deliberately not via ``device_put`` of a host array: under multi-process
    JAX only this process's shards are addressable, so a host-side global array
    is the wrong shape of object. Generating under ``out_shardings`` sidesteps
    that and costs nothing.
    """
    shape = (num_devices, rows, cols)
    return jax.jit(lambda: jax.random.normal(jax.random.key(0), shape, dtype),
                   out_shardings=sharding)()


def time_fn(fn, x, repeats: int, warmup: int) -> float:
    """Median wall time in seconds of a blocked call."""
    for _ in range(warmup):
        jax.block_until_ready(fn(x))
    samples = []
    for _ in range(repeats):
        # Line the processes up so one straggler's dispatch does not land
        # inside another's timed region.
        multihost_utils.sync_global_devices("bench")
        t0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def verify(impl: str, mesh: Mesh, num_devices: int, x_dev) -> float:
    """Max relative deviation from ``jax.lax.psum``, computed on device.

    Compared against psum rather than a host reference so this works unchanged
    across processes; the result is reduced to a replicated scalar, which is
    addressable everywhere.
    """
    op = make_op(impl, num_devices)

    def body(local_x):
        v = local_x[0]
        got = op(v).astype(jnp.float32)
        want = jax.lax.psum(v, axis_name=AXIS).astype(jnp.float32)
        rel = jnp.max(jnp.abs(got - want)) / jnp.maximum(
            jnp.max(jnp.abs(want)), 1e-6)
        return jax.lax.pmax(rel, axis_name=AXIS)

    fn = jax.jit(
        shard_map(body,
                  mesh=mesh,
                  in_specs=P(AXIS, None, None),
                  out_specs=P(),
                  check_rep=False))
    return float(fn(x_dev))


def verify_chain(impl: str, mesh: Mesh, num_devices: int, x_dev,
                 n_ops: int) -> float:
    """Max relative deviation of an n_ops chain vs the same chain of psums.

    Needed for the two-shot kernel: its failure mode is a peer writing a slot
    while a *previous* invocation still reads it, which a single-op check
    cannot see.

    The comparison is reduced to a replicated scalar on device. Pulling the
    chain outputs to host with device_get works single-process but raises on a
    multi-process slice, where each process only addresses its own shards.
    """
    got = make_chain(impl, mesh, num_devices, n_ops)(x_dev)
    want = make_chain("xla", mesh, num_devices, n_ops)(x_dev)

    def rel(a, b):
        a = a.astype(jnp.float32)
        b = b.astype(jnp.float32)
        return (jnp.max(jnp.abs(a - b)) /
                jnp.maximum(jnp.max(jnp.abs(b)), 1e-6))

    return float(
        jax.jit(rel, out_shardings=NamedSharding(mesh, P()))(got, want))


def main():
    dtype = {"bfloat16": jnp.bfloat16, "float32": jnp.float32}[ARGS.dtype]
    itemsize = np.dtype(ARGS.dtype).itemsize
    device_counts = [int(d) for d in ARGS.devices.split(",")]
    impls = [s.strip() for s in ARGS.impls.split(",") if s.strip()]
    cols = ARGS.cols
    if ARGS.shapes:
        shape_sweep = []
        for tok in ARGS.shapes.split(","):
            r, _, c = tok.strip().lower().partition("x")
            shape_sweep.append((int(r), int(c)))
    elif ARGS.rows.strip():
        shape_sweep = [(int(r), cols) for r in ARGS.rows.split(",")]
    else:
        n_dev = len(jax.devices())
        shape_sweep = [(n_dev * m, cols)
                       for m in (1, 2, 4, 8, 16, 32, 64, 128, 256)]

    rprint(f"jax {jax.__version__} | {len(jax.devices())} devices "
           f"({jax.process_count()} process(es)) | "
           f"{jax.devices()[0].device_kind}")
    rprint(
        f"LIBTPU_INIT_ARGS={os.environ.get('LIBTPU_INIT_ARGS', '') or '(unset)'}"
    )
    rprint(f"dtype={ARGS.dtype} cols={cols} "
           f"slope over {ARGS.iters_lo}->{ARGS.iters_hi} chained ops, "
           f"median of {ARGS.repeats}\n")

    rows_out = []
    for num_devices in device_counts:
        mesh = build_mesh(num_devices)
        sharding = NamedSharding(mesh, P(AXIS, None, None))
        rprint(f"=== {num_devices} devices "
               f"({[d.id for d in mesh.devices.flatten()]}) ===")
        rprint(f"{'shape':>12} {'KiB':>9} " + "".join(f"{i + ' us':>13}"
                                                      for i in impls) +
               f"  {'best vs xla':>12}")

        for rows, cols in shape_sweep:
            payload = rows * cols * itemsize
            sharding = NamedSharding(mesh, P(AXIS, None, None))
            x_dev = make_input(num_devices, rows, cols, dtype, sharding)

            lat = {}
            absolute = {}
            for impl in impls:
                if (impl.startswith("twoshot") and rows % num_devices != 0):
                    lat[impl] = None
                    continue
                try:
                    if not ARGS.no_verify:
                        err = (verify_chain(impl, mesh, num_devices, x_dev,
                                            ARGS.iters_lo) if impl
                               in ("twoshot", "twoshot_nb") else verify(
                                   impl, mesh, num_devices, x_dev))
                        # bfloat16 sums N-1 peers in a different order than
                        # psum, and that drift grows with rounds and devices
                        # (0.02 at 8 devices, 0.05 at 32). A fixed threshold
                        # reports it as corruption; scale it instead. The
                        # For an exact check, run the sweep in float32,
                        # where psum and the kernel agree to ~1e-7.
                        scale_tol = (1e-3 if ARGS.dtype == "float32" else
                                     0.01 * (ARGS.iters_lo * num_devices)**0.5)
                        if err > scale_tol:
                            rprint(
                                f"  !! {impl} rows={rows} rel err {err:.3g}",
                                file=sys.stderr)
                    lo = make_chain(impl, mesh, num_devices, ARGS.iters_lo)
                    hi = make_chain(impl, mesh, num_devices, ARGS.iters_hi)
                    t_lo = time_fn(lo, x_dev, ARGS.repeats, ARGS.warmup)
                    t_hi = time_fn(hi, x_dev, ARGS.repeats, ARGS.warmup)
                    lat[impl] = (t_hi - t_lo) / (ARGS.iters_hi - ARGS.iters_lo)
                    # Wall time per call *without* cancelling fixed cost. The
                    # slope is the marginal cost of one more collective; this
                    # is what a caller actually waits for, amortized. A large
                    # gap between them means the fixed cost the slope discards
                    # is not small relative to the op.
                    absolute[impl] = t_hi / ARGS.iters_hi
                except Exception as e:  # noqa: BLE001 - report and keep sweeping
                    rprint(
                        f"  !! {impl} rows={rows} failed: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr)
                    lat[impl] = None

            # A non-positive slope means the two chain lengths were not
            # separable above noise -- report it rather than ranking on it.
            for k, v in lat.items():
                if v is not None and v <= 0:
                    lat[k] = None
            cells = "".join(
                f"{lat[i] * 1e6:>13.2f}" if lat.get(i) else f"{'-':>13}"
                for i in impls)
            best = min((v for k, v in lat.items() if k != "xla" and v),
                       default=None)
            ratio = (f"{lat['xla'] / best:>11.2f}x"
                     if best and lat.get("xla") else f"{'-':>12}")
            rprint(f"{f'{rows}x{cols}':>12} {payload / 1024:>9.2f} {cells}"
                   f"  {ratio}")
            rprint(f"{'(absolute)':>12} {'':>9} " +
                   "".join(f"{absolute[i] * 1e6:>13.2f}" if absolute.
                           get(i) else f"{'-':>13}" for i in impls))

            for impl in impls:
                rows_out.append({
                    "num_devices":
                    num_devices,
                    "dtype":
                    ARGS.dtype,
                    "rows":
                    rows,
                    "cols":
                    cols,
                    "payload_bytes":
                    payload,
                    "impl":
                    impl,
                    "latency_us": (lat[impl] * 1e6) if lat.get(impl) else "",
                    "absolute_us":
                    (absolute[impl] * 1e6) if absolute.get(impl) else "",
                })
        rprint()

    if ARGS.csv and IS_LEAD:
        import csv
        with open(ARGS.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
        rprint(f"wrote {ARGS.csv}")


if __name__ == "__main__":
    main()
