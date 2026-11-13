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
"""Contiguous vs paged HBM reads. Device logical KV pages, not OS page faults."""

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
SOURCE = 2 * TOTAL
WIDTH = 128
NROWS = SOURCE // 4 // WIDTH
mesh = Mesh(np.array(jax.devices()), ("d",))
assert mesh.size == 8
host = (
    (np.arange(SOURCE // 4, dtype=np.int32) % 4093)
    .astype(np.float32)
    .reshape(1, NROWS, WIDTH)
)
x = jax.make_array_from_single_device_arrays(
    (8, NROWS, WIDTH),
    NamedSharding(mesh, P("d", None, None)),
    [jax.device_put(host, d) for d in mesh.devices.flat],
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
patterns = ["direct", "offset", "table", "stride2", "random", "spread"]
configs = []
for b in [16384, 8192, 4096, 2048, 1024]:
    for pat in patterns:
        configs.append(("latency", b, 1, pat, 1))
        for q in [1, 8, 16, 32, 64, 128, 256, 512]:
            configs.append(("window", b, q, pat, 1))
if "--smoke" in sys.argv:
    configs = [("window", 1024, 512, "spread", 1)]
report = {
    "suite": "small_pages",
    "total_bytes": TOTAL,
    "source_bytes": SOURCE,
    "peak_bytes_per_second": 3.69e12,
    "configs": [],
}


def save() -> None:
    (ROOT / "run.json").write_text(json.dumps(report, indent=2))


def build(
    mode: str, b: int, q: int, pat: str, fragments: int
) -> tuple[str, Callable, Any, Any]:
    rows = b // 4 // WIDTH
    n = TOTAL // b
    slots = q * fragments
    outq = slots if mode != "latency" else 1
    tag = f"{mode}_b{b}_q{q}_{pat}_f{fragments}"
    ids = np.arange(n, dtype=np.int32)
    if pat == "stride2":
        ids *= 2
    elif pat == "random":
        ids = np.random.default_rng(8128 + b).permutation(n).astype(np.int32)
    elif pat == "spread":
        ids = np.random.default_rng(8128 + b).permutation(2 * n)[:n].astype(np.int32)
    table = jax.device_put(ids, NamedSharding(mesh, P()))

    def kernel(mapping, src, dst, buf, sem):
        def desc(i, slot):
            if pat in ("direct", "offset"):
                start = i * rows + (rows // 2 if pat == "offset" else 0)
            else:
                start = mapping[i] * rows
            return pltpu.make_async_copy(
                src.at[pl.ds(start, rows), :], buf.at[slot], sem.at[slot]
            )

        if mode == "latency":

            def step(i, _):
                with jax.named_scope("DMA_" + tag):
                    cp = desc(i % n, 0)
                    cp.start()
                    cp.wait()

            jax.lax.fori_loop(0, 64, step, None)
        elif mode == "window":
            with jax.named_scope("DMA_" + tag):
                jax.lax.fori_loop(0, q, lambda i, _: desc(i, i).start(), None)

                def exchange(i):
                    slot = i % q
                    desc(i, slot).wait()
                    desc(i + q, slot).start()

                def step(g, _):
                    for u in range(8):
                        exchange(g * 8 + u)

                jax.lax.fori_loop(0, (n - q) // 8, step, None)
                jax.lax.fori_loop(
                    (n - q) // 8 * 8, n - q, lambda i, _: exchange(i), None
                )
                jax.lax.fori_loop(n - q, n, lambda i, _: desc(i, i % q).wait(), None)
        else:
            count = n // fragments

            def start_group(g, slot):
                for f in range(fragments):
                    desc(g * fragments + f, slot * fragments + f).start()

            def wait_group(g, slot):
                for f in range(fragments):
                    desc(g * fragments + f, slot * fragments + f).wait()

            with jax.named_scope("DMA_" + tag):
                jax.lax.fori_loop(0, q, lambda i, _: start_group(i, i), None)

                def step(i, _):
                    wait_group(i, i % q)
                    start_group(i + q, i % q)

                jax.lax.fori_loop(0, count - q, step, None)
                jax.lax.fori_loop(
                    count - q, count, lambda i, _: wait_group(i, i % q), None
                )

        # Both ends of every retained fragment, outside the measurement scope.
        def sample(i, _):
            dst[i, 0, :] = buf[i, 0, :]
            dst[i, 1, :] = buf[i, rows - 1, :]

        jax.lax.fori_loop(0, outq, sample, None)

    call = pl.pallas_call(
        kernel,
        name="hbm_" + tag,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=60 * 1024**2, disable_bounds_checks=True
        ),
        out_shape=jax.ShapeDtypeStruct((outq, 2, WIDTH), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=(
                pltpu.VMEM((slots, rows, WIDTH), jnp.float32),
                pltpu.SemaphoreType.DMA((slots,)),
            ),
        ),
    )
    fn = jax.jit(
        jax.shard_map(
            lambda a, t: call(t, a[0])[None, :, :, :],
            mesh=mesh,
            in_specs=(P("d", None, None), P()),
            out_specs=P("d", None, None, None),
            check_vma=False,
        )
    )
    if mode == "latency":
        idx = [63 % n]
    elif mode == "window":
        idx = [i + ((n - 1 - i) // q) * q for i in range(q)]
    else:
        idx = [
            (i + ((n // fragments - 1 - i) // q) * q) * fragments + f
            for i in range(q)
            for f in range(fragments)
        ]
    starts = [
        i * rows + (rows // 2 if pat == "offset" else 0)
        if pat in ("direct", "offset")
        else int(ids[i]) * rows
        for i in idx
    ]
    expected = np.stack(
        [
            np.stack([host[0, start, :], host[0, start + rows - 1, :]])
            for start in starts
        ]
    )
    return tag, fn, table, expected


opts = jax.profiler.ProfileOptions()
opts.host_tracer_level = 2
opts.python_tracer_level = 0
opts.enable_hlo_proto = True
opts.advanced_configuration = {
    "tpu_trace_mode": "TRACE_COMPUTE",
    "tpu_perf_counters": False,
    "tpu_enable_periodic_counter_sampling": False,
}
batches = []
off = 0
while off < len(configs):
    count = min(16, configs[off][1] // 1024)
    end = off
    while end < min(off + count, len(configs)) and configs[end][1] == configs[off][1]:
        end += 1
    batches.append((off, configs[off:end]))
    off = end
for off, batch_cfg in batches:
    batch = []
    for mode, b, q, pat, frags in batch_cfg:
        tag, fn, table, expected = build(mode, b, q, pat, frags)
        t = time.time()
        out = jax.block_until_ready(fn(x, table))
        a = np.asarray(out)
        np.testing.assert_array_equal(a, np.broadcast_to(expected, a.shape))
        print("COMPILED_VALIDATED", tag, round(time.time() - t, 2), flush=True)
        batch.append((tag, fn, table, mode, b, q, pat, frags))
    with jax.profiler.trace(
        str(ROOT / f"profile_{off:03d}_n{len(batch_cfg)}"), profiler_options=opts
    ):
        for tag, fn, table, mode, b, q, pat, frags in batch:
            times = []
            for i in range(10):
                with jax.profiler.TraceAnnotation(tag, step_num=i):
                    t = time.perf_counter()
                    jax.block_until_ready(fn(x, table))
                    times.append((time.perf_counter() - t) * 1e6)
                time.sleep(0.2)
            report["configs"].append(
                dict(
                    tag=tag,
                    mode=mode,
                    block_bytes=b,
                    q=q,
                    pattern=pat,
                    fragments=frags,
                    physical_window=q * frags,
                    vmem_payload_bytes=q * frags * b,
                    host_call_us=times,
                    validated_ranks=8,
                )
            )
            save()
    print("PROFILE_BATCH_DONE", off, flush=True)
report["status"] = "passed"
save()
print("DONE", len(configs), flush=True)
