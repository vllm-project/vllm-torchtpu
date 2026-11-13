# Copyright 2025 Google LLC
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
"""HBM->VMEM latency and bounded-window DMA throughput on all 8 local devices."""

import json
import sys
import time
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from common import prepare_run
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

ROOT = prepare_run()
TOTAL = 32 * 1024**2
WIDTH = 128
NROWS = TOTAL // 4 // WIDTH
mesh = Mesh(np.array(jax.devices()), ("d",))
assert mesh.size == 8
host = (
    (np.arange(TOTAL // 4, dtype=np.int32) % 4093)
    .astype(np.float32)
    .reshape(1, NROWS, WIDTH)
)
sh = NamedSharding(mesh, P("d", None, None))
x = jax.make_array_from_single_device_arrays(
    (8, NROWS, WIDTH), sh, [jax.device_put(host, d) for d in mesh.devices.flat]
)
jax.block_until_ready(x)
(ROOT / "devices.json").write_text(
    json.dumps(
        [
            dict(
                id=d.id, coords=list(d.coords), core=d.core_on_chip, kind=d.device_kind
            )
            for d in mesh.devices.flat
        ],
        indent=2,
    )
)


def build(mode: str, b: int, q: int) -> tuple[str, Callable, Any]:
    rows = b // 4 // WIDTH
    n = TOTAL // b
    tag = f"{mode}_b{b}_q{q}"
    outq = q if mode == "window" else 1

    def kernel(src, dst, buf, sem):
        def desc(i, slot):
            return pltpu.make_async_copy(
                src.at[pl.ds(i * rows, rows), :], buf.at[slot], sem.at[slot]
            )

        if mode == "latency":

            def step(i, _):
                with jax.named_scope("DMA_" + tag):
                    cp = desc(i % n, 0)
                    cp.start()
                    cp.wait()

            jax.lax.fori_loop(0, 64, step, None)
        else:
            with jax.named_scope("DMA_" + tag):
                jax.lax.fori_loop(0, q, lambda i, _: desc(i, i).start(), None)

                def exchange(i):
                    slot = i % q
                    desc(i, slot).wait()
                    desc(i + q, slot).start()

                def step(group, _):
                    for u in range(8):
                        exchange(group * 8 + u)

                jax.lax.fori_loop(0, (n - q) // 8, step, None)
                jax.lax.fori_loop(
                    (n - q) // 8 * 8, n - q, lambda i, _: exchange(i), None
                )
                jax.lax.fori_loop(n - q, n, lambda i, _: desc(i, i % q).wait(), None)

        # Validation sample reads/stores are outside the timed DMA scope.
        def sample(i, _):
            dst[i, :] = buf[i, 0, :]

        jax.lax.fori_loop(0, outq, sample, None)

    call = pl.pallas_call(
        kernel,
        name="hbm_" + tag,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=60 * 1024**2, disable_bounds_checks=True
        ),
        out_shape=jax.ShapeDtypeStruct((outq, WIDTH), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=(
                pltpu.VMEM((q, rows, WIDTH), jnp.float32),
                pltpu.SemaphoreType.DMA((q,)),
            ),
        ),
    )
    fn = jax.jit(
        jax.shard_map(
            lambda a: call(a[0])[None, :, :],
            mesh=mesh,
            in_specs=P("d", None, None),
            out_specs=P("d", None, None),
            check_vma=False,
        )
    )
    if mode == "latency":
        idx = [63 % n]
    else:
        idx = [i + ((n - 1 - i) // q) * q for i in range(q)]
    expected = np.stack([host[0, i * rows, :] for i in idx])
    return tag, fn, expected


sizes = [4096 * 2**i for i in range(14)]
qs = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
configs = [("latency", b, 1) for b in sizes]
configs += [("window", b, q) for b in sizes[:12] for q in qs if q * b <= TOTAL]
if "--smoke" in sys.argv:
    configs = [
        ("latency", 4096, 1),
        ("latency", 32 * 1024**2, 1),
        ("window", 4096, 256),
        ("window", 1024**2, 8),
    ]
report = {
    "suite": "contiguous",
    "total_bytes": TOTAL,
    "peak_bytes_per_second": 3.69e12,
    "configs": [],
}


def save() -> None:
    (ROOT / "run.json").write_text(json.dumps(report, indent=2))


opts = jax.profiler.ProfileOptions()
opts.host_tracer_level = 2
opts.python_tracer_level = 0
opts.enable_hlo_proto = True
opts.advanced_configuration = {
    "tpu_trace_mode": "TRACE_COMPUTE",
    "tpu_perf_counters": False,
    "tpu_enable_periodic_counter_sampling": False,
}
for off in range(0, len(configs), 16):
    batch = []
    for mode, b, q in configs[off : off + 16]:
        tag, fn, expected = build(mode, b, q)
        t = time.time()
        out = jax.block_until_ready(fn(x))
        a = np.asarray(out)
        np.testing.assert_array_equal(a, np.broadcast_to(expected, a.shape))
        batch.append((tag, fn, mode, b, q))
        print("COMPILED_VALIDATED", tag, round(time.time() - t, 2), flush=True)
    with jax.profiler.trace(str(ROOT / f"profile_{off:03d}"), profiler_options=opts):
        for tag, fn, mode, b, q in batch:
            times = []
            for i in range(10):
                with jax.profiler.TraceAnnotation(tag, step_num=i):
                    t = time.perf_counter()
                    jax.block_until_ready(fn(x))
                    times.append((time.perf_counter() - t) * 1e6)
                time.sleep(0.2)
            report["configs"].append(
                dict(
                    tag=tag,
                    mode=mode,
                    block_bytes=b,
                    q=q,
                    host_call_us=times,
                    validated_ranks=8,
                )
            )
            save()
    print("PROFILE_BATCH_DONE", off, flush=True)
report["status"] = "passed"
save()
print("DONE", len(configs), flush=True)
