# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DP8 TorchTPU Host gather benchmark; one process per TPU core.

Every step runs one graph:
    Prepare -> Mat_A(a) -> start -> Mat_B(b) -> finish -> Mat_C(c)
Mat_B is independent of the rows and covers the Host gather and H2D. Mat_C
consumes the rows. Backends differ only in start and finish:
    E  compute only; finish returns rows already in HBM.
    A  SC writes indices and a sequence to pinned Host; a C++ worker gathers to
       pinned staging; SC polls the response, then copies the rows to HBM.
    B  as A, but the worker also uploads the rows into a retained HBM buffer
       through PJRT raw H2D; SC polls and a TC kernel publishes a fresh copy.
    E2 B's graph with no worker: the same kernels and Host buffers, one poll and
       zero rows, so B minus E2 is the cost of the Host gather itself.
    C  HostExecute FFI: the device writes the indices to pinned Host, XLA runs a
       C++ gather on the Host inside the graph, then one DMA copies the rows to
       HBM. Uses the sources in reference/ and libtpu's PJRT FFI.
    G  The device DMAs the 4 KiB pinned Host page holding each row to HBM and
       extracts the row there; no Host thread.

Phases: validate (byte checks and queued steps; for A and B also error
injection and negative controls), calibrate (per-matmul time and Host path),
sweep b, tail (2048 steps at b* to count rare Host stalls), profile, summary.
"""

import ctypes
import itertools
import json
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import host_embedding
import host_ffi
import host_ffi_registration
import host_gather_ffi
import jax
import jax.numpy as jnp
import kernels
import numpy as np
import torch
from jax import lax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from torch import distributed as dist
from torch._functorch._activation_offloading import offload_ops  # noqa: F401
from torch_tpu._internal import sync
from torch_tpu._internal.compile import TpuBackend, _backend
from torch_tpu._internal.pallas import get_pallas_mesh
from torch_tpu._internal.pallas.pallas import JaxCallable
from torch_tpu._internal.profiler import TpuProfilerConfig

ROWS = 48_000_000  # 256 B rows per owner, about 12.3 GB
WIDTH = 64  # int32 words per row
COUNT = 6912  # rows per request
PAGE = kernels.PAGE
IDS_WORDS = -(-COUNT // PAGE) * PAGE
STAGE_WORDS = COUNT * WIDTH
MATRIX = 2048

# Raw runs before nowait so the worker is idle when each control starts.
VARIANTS = (
    ("E", "wait"),
    ("E2", "wait"),
    ("A", "wait"),
    ("A", "nowait"),
    ("B", "wait"),
    ("B", "raw"),
    ("B", "nowait"),
    ("C", "wait"),
    ("G", "wait"),
)
HOST_ARGUMENTS = {}


def row_ids(rank, seq, xp):
    """Pseudo-random rows that change every step; the CPU recomputes them."""
    lane = xp.arange(COUNT, dtype=xp.uint32)
    key = xp.asarray(seq).astype(xp.uint32) * xp.uint32(40503)
    key = key + xp.asarray(rank).astype(xp.uint32) * xp.uint32(97)
    return ((lane * xp.uint32(2654435761) + key) % xp.uint32(ROWS)).astype(xp.int32)


def salt(rank):
    return (rank + 1) * 100000007


def expected_rows(rank, seq):
    ids = row_ids(rank, seq, np).astype(np.int64)
    words = (ids[:, None] * WIDTH + np.arange(WIDTH)).astype(np.uint32)
    return (words ^ np.uint32(salt(rank))).view(np.int32).reshape(-1)


def chain(hidden, weight, depth):
    return lax.fori_loop(0, depth, lambda _, x: x @ weight, hidden)


def make_step(backend, mode):
    def local(control, hidden, weight, *buffers):
        seq, a, b, c, bad, rank, handle = (control[i] for i in range(7))

        # Prepare and Mat_A. The weight is an identity, so hidden stays exact.
        ids = row_ids(rank, seq, jnp)
        ids = jnp.where((bad > 0) & (jnp.arange(COUNT) == 3), ROWS, ids)
        ids = jnp.pad(ids, (0, IDS_WORDS - COUNT))
        hidden = chain(hidden, weight, a)
        ids, hidden = lax.optimization_barrier((ids, hidden))

        extra = ()
        if backend == "E":
            hidden = chain(hidden, weight, b)
            rows, status = buffers[0], jnp.zeros(PAGE, jnp.int32)
        elif backend == "C":
            # The HLO pass in host_ffi orders the Host gather before Mat_B.
            request, staging, semaphore, destination = host_gather_ffi.start_gather(
                ids[:COUNT], handle, *buffers
            )
            hidden = chain(hidden, weight, b)
            semaphore, destination, hidden = lax.optimization_barrier(
                (semaphore, destination, hidden)
            )
            rows = host_gather_ffi.finish_gather(staging, semaphore, destination)
            status = jnp.zeros(PAGE, jnp.int32).at[0].set(seq)
            extra = (request, staging)
        elif backend == "G":
            (table,) = buffers
            semaphore, pages = host_embedding.start_host_rows(
                table, ids[:COUNT], row_words=WIDTH
            )
            # The pages are unfinished until the wait; Mat_B runs in between.
            semaphore, pages, hidden = lax.optimization_barrier(
                (semaphore, pages, hidden)
            )
            hidden = chain(hidden, weight, b)
            semaphore, pages, hidden = lax.optimization_barrier(
                (semaphore, pages, hidden)
            )
            rows = host_embedding.finish_host_rows(
                table, ids[:COUNT], semaphore, pages, row_words=WIDTH
            ).reshape(-1)
            status = jnp.zeros(PAGE, jnp.int32).at[0].set(seq)
        else:
            h_ids, h_req, h_resp, rows_buffer = buffers
            sequence = jnp.full((PAGE,), seq, jnp.int32)
            receipt, h_ids, h_req = kernels.request(ids, sequence, h_ids, h_req)

            # Mat_B runs after the request and before the receive.
            receipt, hidden = lax.optimization_barrier((receipt, hidden))
            hidden = chain(hidden, weight, b)
            receipt, hidden = lax.optimization_barrier((receipt, hidden))

            wait = mode != "nowait" and backend != "E2"
            if backend == "A":
                status, rows = kernels.receive(receipt, h_resp, rows_buffer, wait=wait)
            else:
                status = kernels.receive(receipt, h_resp, wait=wait)
                if backend == "E2":
                    # No worker answers; a positive status makes publish copy as in B.
                    status = jnp.maximum(status, 1)
                # raw returns the externally written buffer without publishing.
                rows = (
                    rows_buffer
                    if mode == "raw"
                    else kernels.publish(status, rows_buffer)
                )
            # Host placement must stay inside the manual DP8 partition.
            extra = tuple(
                jax.device_put(x, jax.memory.Space.Host) for x in (h_ids, h_req)
            )

        # Mat_C consumes the rows through their bit parity.
        parity = (jnp.sum(rows & 1) & 1).astype(hidden.dtype)
        hidden = chain(hidden + parity, weight, c)
        return (hidden, rows, status, *extra)

    return local


class HostCallable(JaxCallable):
    """Gives pinned Host inputs a pinned_host sharding in the traced signature."""

    def _get_jax_placeholders(self, norm_args, norm_kwargs=None):
        values, keywords = super()._get_jax_placeholders(norm_args, norm_kwargs)
        return [
            jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s)
            for v, s in zip(values, self.placements, strict=True)
        ], keywords


def build(mesh, backend, mode):
    """Compiles one backend variant as a TorchTPU graph around a JAX step."""
    # Pinned Host inputs, the donated subset, and the input count.
    host, donated, inputs = {
        "E": ((), (), 4),
        "E2": ((3, 4, 5), (3, 4), 7),
        "A": ((3, 4, 5, 6), (3, 4), 7),
        "B": ((3, 4, 5), (3, 4), 7),
        "C": ((3, 4), (3, 4), 5),
        "G": ((3,), (), 4),
    }[backend]
    in_specs = (P("dp"), P("dp", None), P("dp", None)) + (P("dp"),) * (inputs - 3)
    out_specs = (P("dp", None), P("dp"), P("dp")) + (P("dp"),) * len(donated)
    placements = tuple(
        NamedSharding(
            mesh.abstract_mesh,
            spec,
            memory_kind="pinned_host" if i in host else "device",
        )
        for i, spec in enumerate(in_specs)
    )
    mapped = jax.shard_map(
        make_step(backend, mode),
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_vma=False,
    )
    name = f"host_gather_bench_{backend}_{mode}"
    call = HostCallable(
        name=name,
        trace_key=name,
        jit_fn=jax.jit(mapped, in_shardings=placements, donate_argnums=donated),
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=out_specs,
        donate_argnums=list(donated),
    )
    call.placements = placements

    arguments = ", ".join(f"Tensor x{i}" for i in range(inputs))
    results = ", ".join(["Tensor"] * len(out_specs))
    op = torch.library.custom_op(
        f"host_gather_bench_ops::{backend}_{mode}",
        lambda *tensors: tuple(call(*tensors)),
        mutates_args=(),
        schema=f"({arguments}) -> ({results})",
    )

    def fake(*tensors):
        control, hidden = tensors[:2]
        outputs = [
            torch.empty_like(hidden),
            control.new_empty(STAGE_WORDS),
            control.new_empty(PAGE),
        ]
        return (*outputs, *(torch.empty_like(tensors[i]) for i in donated))

    op.register_fake(fake)
    HOST_ARGUMENTS[
        getattr(torch.ops.host_gather_bench_ops, f"{backend}_{mode}").default
    ] = host

    def forward(*args):
        outputs = op(*args)
        # The donated request pages come back as new arrays at the same address.
        for index, value in zip(donated, outputs[3:], strict=True):
            args[index].copy_(torch.ops.ao.offload(value))
        return outputs[:3]

    return torch.compile(forward, backend=TpuBackend(), fullgraph=True)


def install_host_marking():
    previous = _backend._mark_pinned_host_inputs

    def mark(graph_module, example_inputs):
        previous(graph_module, example_inputs)
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        for node in graph_module.graph.nodes:
            for index in HOST_ARGUMENTS.get(node.target, ()):
                argument = node.args[index]
                assert argument in placeholders, "Host buffers must be graph inputs"
                example_inputs[placeholders.index(argument)]._is_pinned_host = True

    _backend._mark_pinned_host_inputs = mark


def declare(native):
    native.StartChannel.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
    ]
    native.StartChannel.restype = ctypes.c_void_p
    native.StopChannel.argtypes = [ctypes.c_void_p]
    native.SetChannelDelay.argtypes = [ctypes.c_void_p, ctypes.c_int]
    native.ChannelTimings.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int64,
    ]
    native.ChannelTimings.restype = ctypes.c_int64
    native.AllocateTable.argtypes = [ctypes.c_int64, ctypes.c_uint32]
    native.AllocateTable.restype = ctypes.c_void_p
    native.FreeTable.argtypes = [ctypes.c_void_p, ctypes.c_int64]


def pinned(words):
    value = torch.ops.ao.offload(torch.zeros(words, dtype=torch.int32, device="tpu"))
    sync.synchronize([value], wait=True)
    return value


class Channel:
    """One request in flight; owns the pinned pages and the worker thread."""

    def __init__(self, native, table, runtime_h2d, start=True):
        self.native = native
        ids, req, self.resp, self.stage = (
            pinned(n) for n in (IDS_WORDS, PAGE, PAGE, STAGE_WORDS)
        )
        # The graph donates views of the request pages; leases keep originals.
        self.views = [
            torch.classes.host_gather_bench.PinnedHostView(x) for x in (ids, req)
        ]
        self.ids, self.req = (view.tensor() for view in self.views)
        self.leases = [
            torch.classes.host_gather_bench.PinnedHostBuffer(x)
            for x in (self.resp, self.stage)
        ]
        self.response = ctypes.c_int32.from_address(self.leases[0].data_ptr())

        self.rows, destination = self.stage, None
        if runtime_h2d:
            self.rows = torch.zeros(STAGE_WORDS, dtype=torch.int32, device="tpu")
            sync.synchronize([self.rows], wait=True)
            self.destination = torch.classes.host_gather_bench.RawHbmBuffer(self.rows)
            destination = self.destination.context()

        self.original = self.addresses()
        self.worker = None
        if start:
            self.worker = native.StartChannel(
                (ctypes.c_size_t * 5)(table, *self.original), ROWS, COUNT, destination
            )
            assert self.worker, "Channel rejected its buffers"

    def addresses(self):
        return tuple(
            torch.ops.host_gather_bench.pinned_host_pointer(x)
            for x in (self.ids, self.req, self.resp, self.stage)
        )

    def buffers(self):
        return self.ids, self.req, self.resp, self.rows

    def drain(self, sequence, timeout=60):
        deadline = time.monotonic() + timeout
        while abs(self.response.value) != sequence:
            assert time.monotonic() < deadline, "Worker did not reach the last request"
            time.sleep(0.001)

    def timings(self, n):
        buffer = (ctypes.c_int64 * (6 * n))()
        self.native.ChannelTimings(self.worker, buffer, n)
        records = np.frombuffer(buffer, np.int64).reshape(n, 6)
        return {
            int(seq): {
                "gather_us": (gathered - seen) / 1e3,
                "submit_us": (submitted - gathered) / 1e3,
                "wait_us": (uploaded - submitted) / 1e3,
                "upload_us": (uploaded - gathered) / 1e3,
                "host_us": (published - seen) / 1e3,
            }
            for seq, seen, gathered, submitted, uploaded, published in records
        }

    def close(self):
        if self.worker:
            self.native.StopChannel(self.worker)


class Bench:
    def __init__(self, root, rank, world, native_path, ffi_path):
        self.root, self.rank, self.world = root, rank, world
        self.out = root / f"rank-{rank}"
        (self.out / "benchmark").mkdir(parents=True, exist_ok=True)
        self.metrics = (self.out / "benchmark/metrics.jsonl").open("a")
        torch._dynamo.config.recompile_limit = 64

        torch.ops.load_library(str(native_path))
        self.native = ctypes.CDLL(str(native_path))
        declare(self.native)
        install_host_marking()
        # C's Host handler and its HLO layout fix must exist before any compile.
        self.ffi = ctypes.CDLL(str(ffi_path))
        self.ffi.BindHostGatherTable.argtypes = [ctypes.c_void_p, ctypes.c_int64]
        self.ffi.BindHostGatherTable.restype = ctypes.c_int32
        self.ffi.ReleaseHostGatherTable.argtypes = [ctypes.c_int32]
        host_ffi_registration.register_host_target(
            host_ffi_registration._plugin(),
            host_gather_ffi.FFI_TARGET.encode(),
            ctypes.cast(self.ffi.HostGather256, ctypes.c_void_p),
        )
        host_ffi.install_host_ffi()
        mesh = get_pallas_mesh(axis_names=("dp",))
        assert mesh.size == world == 8
        self.compiled = {(b, m): build(mesh, b, m) for b, m in VARIANTS}

        self.table = self.native.AllocateTable(ROWS, salt(rank))
        assert self.table, "Table allocation failed"
        self.channels = {
            "A": Channel(self.native, self.table, runtime_h2d=False),
            "B": Channel(self.native, self.table, runtime_h2d=True),
            # Same buffers as B but no worker, so its HBM rows stay zero.
            "E2": Channel(self.native, self.table, runtime_h2d=True, start=False),
        }
        # C gathers from the same CPU table as A and B.
        self.handle = self.ffi.BindHostGatherTable(self.table, ROWS)
        assert self.handle >= 0, "FFI table registration failed"
        ffi_buffers = tuple(pinned(n) for n in (IDS_WORDS, STAGE_WORDS))
        # G needs a pinned copy: the device cannot DMA from pageable memory.
        g_table = pinned(ROWS * WIDTH)
        self.g_lease = torch.classes.host_gather_bench.PinnedHostBuffer(g_table)
        ctypes.memmove(self.g_lease.data_ptr(), self.table, g_table.nbytes)
        self.value = torch.full(
            (MATRIX, MATRIX), 0.125, dtype=torch.bfloat16, device="tpu"
        )
        self.weight = torch.eye(MATRIX, dtype=torch.bfloat16).to("tpu")
        self.baseline = torch.zeros(STAGE_WORDS, dtype=torch.int32, device="tpu")
        sync.synchronize([self.value, self.weight, self.baseline], wait=True)
        # Graph inputs after control, hidden and weight; mutated only in place.
        self.inputs = {
            "E": (self.baseline,),
            "C": ffi_buffers,
            "G": (g_table,),
            **{name: channel.buffers() for name, channel in self.channels.items()},
        }
        self.sequence = Counter()
        self.step_index = itertools.count()

    def control(self, backend, a, b, c, bad=False):
        self.sequence[backend] += 1
        seq = self.sequence[backend]
        values = [seq, a, b, c, int(bad), self.rank, self.handle, 0]
        return seq, torch.tensor(values, dtype=torch.int32).to("tpu")

    def call(self, backend, mode, control):
        return self.compiled[backend, mode](
            control, self.value, self.weight, *self.inputs[backend]
        )

    def check(self, backend, seq, bad, outputs):
        hidden, rows, status = (t.cpu() for t in outputs)
        rows = rows.numpy()
        parity = int(np.bitwise_and(rows, 1).sum()) & 1
        result = {"hidden": bool((hidden == 0.125 + parity).all())}
        if backend in ("E", "E2"):
            return {**result, "status": True, "rows": not rows.any()}, 0
        rows_ok = bad or np.array_equal(rows, expected_rows(self.rank, seq))
        status_ok = int(status[0]) == (-seq if bad else seq)
        return {**result, "status": status_ok, "rows": rows_ok}, int(status[1])

    def validate(self, backend, mode, steps=48, delay_us=200):
        """Queued steps without a Host sync; the worker sleeps mid-gather."""
        channel = self.channels.get(backend)
        worker = channel.worker if channel else None
        if worker:
            self.native.SetChannelDelay(worker, delay_us)
        plan = []
        for i in range(steps):
            bad = worker is not None and mode == "wait" and i % 16 == 7
            plan.append((*self.control(backend, 1, (0, 16, 64)[i % 3], 1, bad), bad))
        sync.synchronize([control for _, control, _ in plan], wait=True)
        outputs = [self.call(backend, mode, control) for _, control, _ in plan]
        sync.synchronize([t for output in outputs for t in output], wait=True)

        failures, max_polls = Counter(), 0
        for (seq, _, bad), output in zip(plan, outputs, strict=True):
            result, polls = self.check(backend, seq, bad, output)
            failures.update(key for key, ok in result.items() if not ok)
            max_polls = max(max_polls, polls)
        if worker:
            channel.drain(plan[-1][0])
            self.native.SetChannelDelay(worker, 0)
        if channel:
            assert channel.addresses() == channel.original, "Host buffer moved"
        return {
            "backend": backend,
            "mode": mode,
            "steps": steps,
            "bad_status": failures["status"],
            "wrong_rows": failures["rows"],
            "bad_hidden": failures["hidden"],
            "max_polls": max_polls,
        }

    def measure(self, phase, backend, a, b, c, steps=16, warmup=4):
        records = []
        for i in range(warmup + steps):
            seq, control = self.control(backend, a, b, c)
            sync.synchronize([control], wait=True)
            dist.barrier()
            begin = time.perf_counter_ns()
            outputs = self.call(backend, "wait", control)
            sync.synchronize(list(outputs), wait=True)
            wall_us = (time.perf_counter_ns() - begin) / 1e3
            index = next(self.step_index)
            if i < warmup:
                continue
            status = outputs[2].cpu()
            records.append(
                {
                    "phase": phase,
                    "backend": backend,
                    "a": a,
                    "b": b,
                    "c": c,
                    "index": index,
                    "seq": seq,
                    "wall_us": wall_us,
                    "polls": int(status[1]),
                    "status_ok": backend in ("E", "E2") or int(status[0]) == seq,
                }
            )
        if backend in ("A", "B"):
            timings = self.channels[backend].timings(warmup + steps)
            for record in records:
                record.update(timings[record["seq"]])
        for record in records:
            self.metrics.write(json.dumps(record) + "\n")
        self.metrics.flush()
        return records

    def exchange(self, name, value):
        path = self.root / "exchange"
        path.mkdir(exist_ok=True)
        (path / f"{name}-{self.rank}.json").write_text(json.dumps(value))
        dist.barrier()
        return [
            json.loads((path / f"{name}-{rank}.json").read_text())
            for rank in range(self.world)
        ]

    def calibrate(self):
        """Sizes Mat_B to twice the slowest Host path, both from measurements."""

        def slowest(name, backend, b, key):
            per_rank = self.exchange(name, self.measure("calibrate", backend, 1, b, 1))
            return np.max([[r[key] for r in records] for records in per_rank], axis=0)

        idle = np.median(slowest("e0", "E", 0, "wall_us"))
        busy = np.median(slowest("e64", "E", 64, "wall_us"))
        per_matmul_us = float((busy - idle) / 64)
        assert per_matmul_us > 0, "Matmul calibration failed"
        host_p99_us = {
            backend: float(
                np.percentile(slowest(f"host{backend}", backend, 0, "host_us"), 99)
            )
            for backend in ("A", "B")
        }
        b_star = math.ceil(2 * max(host_p99_us.values()) / per_matmul_us)
        return {
            "per_matmul_us": per_matmul_us,
            "host_p99_us": host_p99_us,
            "b_star": b_star,
        }

    def sweep(self, b_star):
        # b* covers twice the Host path, so the exposed wait ends near b*/4..b*/2.
        fine = range(b_star // 4, b_star // 2 + 1, max(1, b_star // 32))
        depths = sorted({0, *fine, 3 * b_star // 4, b_star, 2 * b_star})
        for b in depths:
            for backend in ("E", "E2", "A", "B", "C", "G"):
                self.measure("sweep", backend, 4, b, 4, steps=64)

    def tail(self, b_star, steps=2048):
        """Long runs at b* to count rare Host-path stalls; E2 is the control."""
        for backend in ("E2", "A", "B"):
            self.measure("tail", backend, 4, b_star, 4, steps=steps)

    def profile(self, b_star):
        for b, backend in itertools.product(
            (0, b_star // 2, b_star, 2 * b_star), ("E", "E2", "A", "B", "C", "G")
        ):
            name = f"{backend}-b{b}"
            capture = self.out / "capture" / name
            profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.PrivateUse1,
                ],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(capture), use_gzip=True
                ),
                experimental_config=TpuProfilerConfig(
                    run_dir=str(capture),
                    worker_rank=str(self.rank),
                    host_tracer_level=2,
                    device_tracer_level=1,
                    python_tracer_level=0,
                    experimental_options={
                        "tpu_trace_mode": "TRACE_COMPUTE",
                        "tpu_num_sparse_cores_to_trace": 2,
                        "tpu_num_sparse_core_tiles_to_trace": 1,
                    },
                ),
            )
            dist.barrier()
            with profiler:
                self.measure("profile", backend, 4, b, 4, steps=8, warmup=0)
            dist.barrier()
            destination = self.root / "profiling/xprof" / name
            destination.mkdir(parents=True, exist_ok=True)
            for path in capture.rglob("*.xplane.pb"):
                shutil.copyfile(path, destination / f"rank{self.rank}_{path.name}")

    def close(self):
        for channel in self.channels.values():
            channel.close()
        self.ffi.ReleaseHostGatherTable(self.handle)
        self.native.FreeTable(self.table, ROWS)
        self.metrics.close()


def stats(values):
    return {
        "p50": float(np.median(values)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def summarize(root, world, calibration, validation):
    """Aggregates per step over ranks (the slowest rank), then over steps."""
    steps = defaultdict(list)
    for rank in range(world):
        for line in (
            (root / f"rank-{rank}/benchmark/metrics.jsonl").read_text().splitlines()
        ):
            record = json.loads(line)
            record["rank"] = rank
            steps[record["index"]].append(record)
    groups = defaultdict(list)
    for records in steps.values():
        if records[0]["phase"] == "sweep":
            groups[records[0]["backend"], records[0]["b"]].append(records)

    sweep = []
    for (backend, b), entries in sorted(groups.items()):
        row = {
            "backend": backend,
            "b": b,
            "steps": len(entries),
            "status_ok": all(r["status_ok"] for records in entries for r in records),
            "wall_us": stats(
                [max(r["wall_us"] for r in records) for records in entries]
            ),
        }
        if backend in ("A", "B"):
            for key in (
                "polls",
                "gather_us",
                "submit_us",
                "wait_us",
                "upload_us",
                "host_us",
            ):
                row[key] = stats([max(r[key] for r in records) for records in entries])
        sweep.append(row)
    # E2 matches the A/B graph shape, so overhead over E2 isolates the channel.
    for control in ("E", "E2"):
        baseline = {
            row["b"]: row["wall_us"]["p50"]
            for row in sweep
            if row["backend"] == control
        }
        for row in sweep:
            row[f"overhead_vs_{control}_p50_us"] = (
                row["wall_us"]["p50"] - baseline[row["b"]]
            )

    totals, polls = defaultdict(Counter), Counter()
    for records in validation:
        for record in records:
            key = record["backend"], record["mode"]
            totals[key].update(
                {
                    k: v
                    for k, v in record.items()
                    if k not in ("backend", "mode", "max_polls")
                }
            )
            polls[key] = max(polls[key], record["max_polls"])
    checks = [
        {"backend": b, "mode": m, **counts, "max_polls": polls[b, m]}
        for (b, m), counts in totals.items()
    ]
    failures = ("bad_status", "wrong_rows", "bad_hidden")
    passed = all(
        sum(c[k] for k in failures) == 0 if c["mode"] == "wait" else True
        for c in checks
    ) and all(c["wrong_rows"] > 0 for c in checks if c["mode"] == "nowait")

    # Rare stalls at b*: per step the slowest rank, plus the worst steps.
    tail = defaultdict(list)
    for records in steps.values():
        if records[0]["phase"] == "tail":
            tail[records[0]["backend"]].append(records)
    tails = {}
    for backend, entries in sorted(tail.items()):
        wall = [max(r["wall_us"] for r in records) for records in entries]
        row = {
            "steps": len(entries),
            "wall_us": stats(wall),
            "wall_over_3ms": sum(w > 3000 for w in wall),
        }
        if backend in ("A", "B"):
            slowest = [max(records, key=lambda r: r["host_us"]) for records in entries]
            fields = ("index", "rank", "gather_us", "submit_us", "wait_us")
            row.update(
                host_us=stats([r["host_us"] for r in slowest]),
                host_over_1ms=sum(r["host_us"] > 1000 for r in slowest),
                exposed_steps=sum(
                    max(r["polls"] for r in records) > 1 for records in entries
                ),
                worst=[
                    {k: r[k] for k in (*fields, "host_us", "polls", "wall_us")}
                    for r in sorted(slowest, key=lambda r: -r["host_us"])[:8]
                ],
            )
        tails[backend] = row

    window_us = calibration["b_star"] * calibration["per_matmul_us"]
    ratio = window_us / max(calibration["host_p99_us"].values())
    summary = {
        "passed": passed,
        "calibration": {**calibration, "window_us": window_us, "window_ratio": ratio},
        "window_ok": ratio >= 1.5,
        "validation": checks,
        "sweep": sweep,
        "tail": tails,
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("HOST_GATHER_BENCH", json.dumps(summary), flush=True)
    return passed


def run(root: Path, native_path: Path, ffi_path: Path) -> None:
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    bench = Bench(root, rank, world, native_path, ffi_path)
    passed = True
    try:
        validation = bench.exchange(
            "validation", [bench.validate(b, m) for b, m in VARIANTS]
        )
        calibration = bench.calibrate()
        bench.sweep(calibration["b_star"])
        bench.tail(calibration["b_star"])
        bench.profile(calibration["b_star"])
        dist.barrier()
        if rank == 0:
            passed = summarize(root, world, calibration, validation)
            try:
                import xprof_summary

                xprof_summary.main(root)
            except Exception as error:  # The trace summary is diagnostic only.
                print("XPROF_SUMMARY_FAILED", repr(error), flush=True)
    finally:
        bench.close()
    if not passed:
        raise SystemExit("Validation failed; see summary.json")
