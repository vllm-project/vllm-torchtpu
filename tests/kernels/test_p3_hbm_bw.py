# SPDX-License-Identifier: Apache-2.0
"""HBM read bandwidth per TensorCore, chip, host and slice, timed on the device (phase 3 port
of the G1 tool in agentx-p1-v2-moewide tools/g1_hbm_bw/).

The reader is the megakernel's expert ring (p3_proto_moe_wide.py, mode "stream": DMA only,
1 MiB tiles, 2 experts in flight), reading NE fp8 experts of 12 MiB each per TensorCore,
reps passes per call, so one call moves 805 MB (ne64), 3.2 GB (ne256) or 8 GB (ne64x10) per
core and runs for milliseconds: long enough that host wall around block_until_ready is
within a few percent of the device time, and the profiler trace of the same call gives the
per-core device duration (scripts/vllm/benchmark/p3_xplane_device_time.py). An XLA f32
reduction over 2 GiB is the non-Pallas control.

Core sets (the "group" axis of the measurement; every core of a set reads at once, each
with its own independent program):

  c1        one TensorCore
  c2same    the two cores of one chip (do they share HBM bandwidth)
  c2diff    one core on each of two chips
  c4chips   one core on each of four chips of a host
  c8        all eight cores of one host
  c16, c32  all cores of a 2x2x2 / 2x2x4 slice (one program per core, every host at once)

Output: <out_dir>/results.jsonl with one row per (size, core set), host_median_us,
host_per_core_GBps, bytes_per_core, trace_dir; env.json with the environment id and the
device list. Collected only with P3_BENCH=1; needs the serving image (jax.profiler
segfaults on the unittest default image) and at least 8 devices. On a multi-host slice
every host runs the module; the per-core programs are local, so each host reports its own
cores and the c16 / c32 rows are the union (process 0 writes results.jsonl).
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import sys
import time

import pytest

if not os.environ.get("P3_BENCH"):
    pytest.skip("phase 3 harness bench; set P3_BENCH=1 to run", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(_ROOT, "scripts", "vllm", "benchmark"))
sys.path.insert(0, _HERE)
import p3_bench_common as common  # noqa: E402
import p3_proto_moe_wide as PW  # noqa: E402
from vllm_torchtpu.kernels.qwen35_megakernel.config import FP8  # noqa: E402

OUT = os.environ.get("P3_HBM_OUT") or common.run_dir("hbm_bw")
NE_MAX = 256  # experts allocated per core: 3.2 GB of fp8 weights
HOST_ITERS = 10
TRACE_ITERS = 5
TRACE = os.environ.get("P3_HBM_TRACE", "1") == "1"
SIZES = {
    "ne64": (64, 1),
    "ne256": (256, 1),
    "ne64x10": (64, 10),
}
SIZE_SEL = [x for x in os.environ.get("P3_HBM_SIZES", "ne64,ne64x10").split(",") if x]
SET_SEL = [x for x in os.environ.get("P3_HBM_SETS", "").split(",") if x]
if os.path.isdir("/workspace/tmp"):
    jax.config.update("jax_compilation_cache_dir", "/workspace/tmp/p3-harness/jax-cache")
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def _chips(devs):
    groups: dict = {}
    for d in devs:
        groups.setdefault(tuple(getattr(d, "coords", (d.id,))), []).append(d)
    for g in groups.values():
        g.sort(key=lambda d: getattr(d, "core_on_chip", d.id))
    return [groups[k] for k in sorted(groups)]


def _devices():
    """Returns global JAX devices after optional multi-host initialization."""
    if not jax.distributed.is_initialized():
        addr = os.environ.get("JAX_COORDINATOR_ADDRESS", "")
        nproc = int(os.environ.get("JAX_NUM_PROCESSES", "1"))
        if addr and nproc > 1:
            jax.distributed.initialize(
                coordinator_address=addr,
                num_processes=nproc,
                process_id=int(os.environ.get("JAX_PROCESS_ID", "0")),
            )
    return jax.devices()


def _core_sets():
    """name -> local devices that read at once (global sets list every host's local cores)."""
    local = jax.local_devices()
    chips = _chips(local)
    sets = {"c1": [chips[0][0]]}
    if len(chips[0]) >= 2:
        sets["c2same"] = chips[0][:2]
    if len(chips) >= 2:
        sets["c2diff"] = [chips[0][0], chips[1][0]]
    if len(chips) >= 4:
        sets["c4chips"] = [g[0] for g in chips[:4]]
    if len(local) >= 8:
        sets["c8"] = list(local)
    n_all = len(_devices())
    if n_all >= 16 and jax.process_count() >= 2:
        sets[f"c{n_all}"] = list(local)  # every host runs its 8 at the same time
    if SET_SEL:
        sets = {k: v for k, v in sets.items() if k in SET_SEL}
    return sets


def _random(dev, shape, dtype, seed):
    """Random bits of dtype committed to dev (any bit pattern is fine for a DMA read)."""
    width = jnp.dtype(dtype).itemsize
    per_word = 4 // width
    assert shape[-1] % per_word == 0
    with jax.default_device(dev):
        words = jax.random.bits(jax.random.key(seed), (*shape[:-1], shape[-1] // per_word), jnp.uint32)
        if per_word > 1:
            narrow = {1: jnp.uint8, 2: jnp.uint16}[width]
            words = lax.bitcast_convert_type(words, narrow).reshape(shape)
        return jax.block_until_ready(jax.device_put(lax.bitcast_convert_type(words, dtype), dev))


def _host_time(run, n=HOST_ITERS):
    for _ in range(2):
        jax.block_until_ready(run())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        jax.block_until_ready(run())
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts), min(ts)


def _trace(path, run, n=TRACE_ITERS):
    if not TRACE:
        return None
    jax.block_until_ready(run())
    jax.profiler.start_trace(path)
    for _ in range(n):
        jax.block_until_ready(run())
        time.sleep(0.005)
    jax.profiler.stop_trace()
    return path


_ENV: dict = {}


def _record(row):
    row = {"process": jax.process_index(), **row}
    if jax.process_index() != 0:
        print("HBM_BW " + json.dumps(row), flush=True)
        return
    if not _ENV:
        os.makedirs(OUT, exist_ok=True)
        _ENV.update(common.write_env(OUT, {"bench": "hbm_bw"}, devices=True))
    row = {"environment_id": _ENV["id"], **row}
    common.append_jsonl(os.path.join(OUT, "results.jsonl"), row)
    print("HBM_BW " + json.dumps(row), flush=True)


class _Args:
    """Per-core stream inputs: NE_MAX experts of w13 / w2 and the routing dummies the stream
    mode never reads; built once per process."""

    def __init__(self):
        c = PW.ProtoCfg(NE=NE_MAX, mode="stream")
        self.per = {}
        for i, d in enumerate(jax.local_devices()):
            w13 = _random(d, (NE_MAX, c.T13, c.TK13, 2 * c.IR), FP8, 2 * i)
            w2 = _random(d, (NE_MAX, c.T2, c.TK2, c.H), FP8, 2 * i + 1)
            small = tuple(jax.device_put(jnp.zeros((8, 128), dt), d) for dt in (
                jnp.bfloat16, jnp.float32, jnp.int32, jnp.int32, jnp.float32, jnp.float32))
            self.per[d] = (*small, w13, w2)


_ARGS: list = []


def _args():
    if not _ARGS:
        _ARGS.append(_Args())
    return _ARGS[0]


def test_devices():
    if len(_devices()) < 8:
        pytest.skip("needs at least 8 TPU devices")
    _record({"label": "devices", "sets": {k: [d.id for d in v] for k, v in _core_sets().items()},
             "jax": jax.__version__})


def test_profiler_smoke():
    """One profiler session around a tiny op: does jax.profiler work in this process."""
    if len(_devices()) < 8:
        pytest.skip("needs at least 8 TPU devices")
    x = jnp.ones((1024, 1024), jnp.float32)
    run = lambda: x @ x  # noqa: E731
    med, _ = _host_time(run)
    path = _trace(os.path.join(OUT, "trace", f"smoke_p{jax.process_index()}"), run)
    _record({"test": "smoke", "host_median_us": round(med, 1), "trace_dir": path})


@pytest.mark.parametrize("size", SIZE_SEL)
def test_stream(size):
    """The expert ring (DMA only) on every core set, one independent program per core."""
    if len(_devices()) < 8:
        pytest.skip("needs at least 8 TPU devices")
    ne, reps = SIZES[size]
    args = _args()
    for set_name, devices in _core_sets().items():
        tag = f"p3s_{size}_{set_name}"
        try:
            c = PW.ProtoCfg(NE=ne, MB=80, CAP=16, mode="stream", depth=2, reps=reps, tag=tag)
            fn = jax.jit(PW.build(c))
            per = [args.per[d] for d in devices]

            def run(fn=fn, per=per):
                return [fn(*a) for a in per]

            med, best = _host_time(run)
            trace_dir = _trace(os.path.join(OUT, "trace", f"stream_{tag}_p{jax.process_index()}"), run)
            _record({"test": "stream", "size": size, "NE": ne, "reps": reps, "set": set_name,
                     "cores": len(devices), "devices": [d.id for d in devices],
                     "bytes_per_core": c.read_bytes,
                     "host_median_us": round(med, 1), "host_min_us": round(best, 1),
                     "host_per_core_GBps": round(c.read_bytes / (med * 1e-6) / 1e9, 1),
                     "host_total_TBps": round(len(devices) * c.read_bytes / (med * 1e-6) / 1e12, 2),
                     "trace_dir": trace_dir})
        except Exception as e:  # noqa: BLE001
            _record({"test": "stream", "size": size, "set": set_name,
                     "failed": f"{type(e).__name__}: {str(e)[:300]}"})
            raise
        gc.collect()


def test_xla_reduce():
    """XLA f32 sum over 2 GiB per core (read only), the non-Pallas control."""
    if len(_devices()) < 8:
        pytest.skip("needs at least 8 TPU devices")
    nelem = 512 * 1024 * 1024
    per = {d: _random(d, (nelem // 1024, 1024), jnp.float32, 200 + i)
           for i, d in enumerate(jax.local_devices())}
    fn = jax.jit(lambda x: jnp.sum(x, axis=0, keepdims=True))
    for set_name, devices in _core_sets().items():
        if set_name not in ("c1", "c2same", "c8") and not set_name.startswith("c16") \
                and not set_name.startswith("c32"):
            continue
        arrs = [per[d] for d in devices]

        def run(fn=fn, arrs=arrs):
            return [fn(a) for a in arrs]

        med, best = _host_time(run)
        trace_dir = _trace(os.path.join(OUT, "trace", f"reduce_{set_name}_p{jax.process_index()}"), run)
        _record({"test": "xla_reduce_f32", "set": set_name, "cores": len(devices),
                 "devices": [d.id for d in devices], "bytes_per_core": nelem * 4,
                 "host_median_us": round(med, 1), "host_min_us": round(best, 1),
                 "host_per_core_GBps": round(nelem * 4 / (med * 1e-6) / 1e9, 1),
                 "trace_dir": trace_dir})
    del per
    gc.collect()
