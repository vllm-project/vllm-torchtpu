# SPDX-License-Identifier: Apache-2.0
"""Native Torch versus Pallas compressor-ring address preparation, kernel only."""

import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import time
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/vllm_torchtpu/models/vllm/deepseek_v41/v2/attention.py"
CACHE = SOURCE.with_name("cache.py")
LABELS = ("has_previous", "valid_group", "state_write_row", "state_previous_row")
Metadata = namedtuple("Metadata", "block_tables seq_lens query_start_loc")


def native_function(capacity, stride, offset, window=128):
    """Extract unchanged helpers and the actual four compressor operands."""
    import torch

    tree = ast.parse(SOURCE.read_text())
    names = {"_block_table", "_token_requests", "_replayed", "_cache_slots"}
    helpers = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(helpers, []), str(SOURCE), "exec"), namespace)
    cache_tree = ast.parse(CACHE.read_text())
    cache_class = next(
        n
        for n in cache_tree.body
        if isinstance(n, ast.ClassDef) and n.name == "PagedCacheRows"
    )
    row_indices = next(
        n
        for n in cache_class.body
        if isinstance(n, ast.FunctionDef) and n.name == "row_indices"
    )
    exec(compile(ast.Module([row_indices], []), str(CACHE), "exec"), namespace)

    cache = SimpleNamespace(shape=(1, capacity), page_stride=stride, row_offset=offset)

    def native(table, starts_loc, kv_starts, positions, minimum):
        state_meta = Metadata(table, minimum, starts_loc)
        kv_meta = Metadata(table, minimum, kv_starts)
        rows = torch.arange(positions.shape[0], device=positions.device)
        valid = (rows < kv_starts[-1]) & ((positions + 1) % 2 == 0)
        valid = valid & ~namespace["_replayed"](kv_meta, positions, minimum, window)
        requests = namespace["_token_requests"](state_meta, positions).clamp(
            max=starts_loc.numel() - 2
        )
        has_prev = rows != starts_loc[requests]
        slots = namespace["_cache_slots"](cache, state_meta, positions, ring=True)
        previous = namespace["_cache_slots"](
            cache, state_meta, positions - 1, ring=True
        )
        stored = (slots >= 0) & (rows >= starts_loc[requests + 1] - capacity)
        return (
            has_prev[:, None].int(),
            valid[:, None].int(),
            namespace["row_indices"](cache, slots).masked_fill(~stored, -1).int(),
            namespace["row_indices"](cache, previous)
            .masked_fill(has_prev | (slots < 0), -1)
            .int(),
        )

    return native


def body(table, starts, kv_starts, positions, minimum, capacity, stride, offset):
    tokens, requests = positions.size, table.shape[0]
    rows = jnp.arange(tokens, dtype=jnp.int32)
    r = jnp.arange(requests, dtype=jnp.int32)[:, None]

    def map_requests(boundaries):
        req = jnp.minimum(
            jnp.sum(rows[None, :] >= boundaries[1:, None], axis=0), requests - 1
        )
        return req[None, :] == r

    selected = map_requests(starts)

    def select(values):
        return jnp.sum(jnp.where(selected, values[:, None], 0), axis=0)

    first, end, blocks = select(starts[:-1]), select(starts[1:]), select(table[:, 0])
    has_prev = rows != first
    active = rows < starts[-1]
    slots = jnp.where(active, blocks * capacity + positions % capacity, -1)
    previous = jnp.where(active, blocks * capacity + (positions - 1) % capacity, -1)
    stored = (slots >= 0) & (rows >= end - capacity)
    replay_start = jnp.sum(
        jnp.where(map_requests(kv_starts), minimum[:, None], 0), axis=0
    )
    replay = (
        (replay_start > 0)
        & (positions >= replay_start)
        & (positions < replay_start + 128)
    )
    valid = (rows < kv_starts[-1]) & ((positions + 1) % 2 == 0) & ~replay

    def physical(slot):
        return offset + slot // capacity * stride + slot % capacity

    return (
        has_prev.astype(jnp.int32),
        valid.astype(jnp.int32),
        jnp.where(stored, physical(slots), -1),
        jnp.where(has_prev | (slots < 0), -1, physical(previous)),
    )


def pallas_function(capacity, stride, offset):
    def call(
        table: jax.Array,
        starts: jax.Array,
        kv_starts: jax.Array,
        positions: jax.Array,
        minimum: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        def kernel(*refs):
            outputs = body(*(r[...] for r in refs[:5]), capacity, stride, offset)
            for ref, value in zip(refs[5:], outputs, strict=True):
                ref[...] = value

        n = positions.shape[0]
        outputs = pl.pallas_call(
            kernel,
            out_shape=tuple(jax.ShapeDtypeStruct((n,), jnp.int32) for _ in range(4)),
            name="dsv41_compressor_state_addresses",
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=16 * 1024 * 1024),
        )(table, starts, kv_starts, positions, minimum)
        return outputs[0][:, None], outputs[1][:, None], outputs[2], outputs[3]

    return call


def reference(arrays, capacity, stride, offset):
    table, starts, kv_starts, positions, minimum = arrays
    result = [np.zeros((positions.size, 1), np.int32) for _ in range(2)]
    result += [np.full(positions.size, -1, np.int32) for _ in range(2)]
    for t, pos in enumerate(positions.tolist()):
        req = min(int(np.searchsorted(starts[1:], t, side="right")), table.shape[0] - 1)
        kr = min(
            int(np.searchsorted(kv_starts[1:], t, side="right")), table.shape[0] - 1
        )
        first, end = int(starts[req]), int(starts[req + 1])
        prev = t != first
        slot = int(table[req, 0]) * capacity + pos % capacity if t < starts[-1] else -1
        old = int(table[req, 0]) * capacity + (pos - 1) % capacity
        replay = minimum[kr] > 0 and minimum[kr] <= pos < minimum[kr] + 128
        result[0][t, 0] = prev
        result[1][t, 0] = t < kv_starts[-1] and (pos + 1) % 2 == 0 and not replay
        if slot >= 0 and t >= end - capacity:
            result[2][t] = offset + slot // capacity * stride + slot % capacity
        if not prev and slot >= 0:
            result[3][t] = offset + old // capacity * stride + old % capacity
    return result


def make_inputs(kind):
    rng = np.random.default_rng(20261104)
    lengths = np.full(48, 6, np.int32)
    if kind == "padding":
        lengths[31:] = 0
    elif kind == "ragged":
        lengths = rng.multinomial(267, np.ones(48) / 48).astype(np.int32)
        lengths[3:6] = 0
    elif kind == "empty":
        lengths[:] = 0
    starts = np.r_[np.int32(0), lengths.cumsum(dtype=np.int32)]
    kv_starts = starts.copy()
    if kind == "distinct":
        kv_starts = np.r_[
            np.int32(0), rng.multinomial(275, np.ones(48) / 48).cumsum()
        ].astype(np.int32)
    positions = rng.integers(0, 100000, 288, dtype=np.int32)
    positions[:8] = [0, 1, 127, 128, 129, 255, 98303, 98304]
    minimum = rng.integers(0, 100000, 48, dtype=np.int32)
    minimum[:8] = [0, 1, 128, 256, 98304, 98200, 1, 0]
    table = rng.integers(0, 8192, (48, 8), dtype=np.int32)
    if kind == "negative":
        table[::3, 0] = -1
    if kind == "decode":
        positions = 98304 + np.tile(np.arange(6, dtype=np.int32), 48)
        minimum[:] = 0
    return table, starts, kv_starts, positions, minimum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--offset", type=int, default=16)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--with-compact", action="store_true")
    parser.add_argument("--shared-starts", action="store_true")
    parser.add_argument(
        "--timing-case",
        default="decode",
        choices=("decode", "uniform", "padding", "ragged", "empty"),
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas, sync

    torch.set_num_threads(1)
    torch.empty(1, device="tpu").cpu()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))
    candidate = pallas.jax_op(
        "pallas::dsv41_attention_state_bench",
        pallas_function(args.capacity, args.stride, args.offset),
        mesh=mesh,
        input_partition_specs=(jax.sharding.PartitionSpec(),) * 5,
    )

    @candidate.register_fake
    def fake(table, starts, kv_starts, positions, minimum):
        n = positions.shape[0]
        return tuple(
            positions.new_empty(s, dtype=torch.int32)
            for s in ((n, 1), (n, 1), (n,), (n,))
        )

    original_call = native_function(args.capacity, args.stride, args.offset)
    fused_call = lambda *xs: candidate(*xs)  # noqa: E731
    compact = None
    if args.with_compact:
        prep_path = (
            ROOT / "src/vllm_torchtpu/kernels/experimental/decode_preparation.py"
        )
        spec = importlib.util.spec_from_file_location(
            "decode_preparation_probe", prep_path
        )
        prep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prep)

        def compact_consumer(
            has_prev: jax.Array,
            valid: jax.Array,
            write: jax.Array,
            read: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return (
                has_prev,
                valid,
                *prep.compact_records(write, 7759456, 288),
                *prep.compact_records(read, 7759456, 288),
            )

        compact = pallas.jax_op(
            "pallas::dsv41_attention_state_compact_bench",
            compact_consumer,
            mesh=mesh,
            input_partition_specs=(jax.sharding.PartitionSpec(),) * 4,
        )

        @compact.register_fake
        def fake_compact(has_prev, valid, write, read):
            return (
                torch.empty_like(has_prev),
                torch.empty_like(valid),
                write.new_empty((576,)),
                write.new_empty((2,)),
                read.new_empty((576,)),
                read.new_empty((2,)),
            )

        def original_call_with_compact(*xs):
            return compact(*original_call(*xs))

        def fused_call_with_compact(*xs):
            return compact(*fused_call(*xs))

        original = torch.compile(
            original_call_with_compact, backend="tpu", fullgraph=True, dynamic=False
        )
        fused = torch.compile(
            fused_call_with_compact, backend="tpu", fullgraph=True, dynamic=False
        )
    else:
        original = torch.compile(
            original_call, backend="tpu", fullgraph=True, dynamic=False
        )
        fused = torch.compile(fused_call, backend="tpu", fullgraph=True, dynamic=False)
    result = {
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (SOURCE, CACHE, Path(__file__))
        },
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "numpy")
        },
        "geometry": {
            "capacity": args.capacity,
            "stride": args.stride,
            "offset": args.offset,
        },
        "correctness": [],
        "with_compact": args.with_compact,
        "shared_starts": args.shared_starts,
        "timing_case": args.timing_case,
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2))

    def device_inputs(arrays):
        tensors = list(torch.from_numpy(a).to("tpu") for a in arrays)
        if args.shared_starts and np.array_equal(arrays[1], arrays[2]):
            tensors[2] = tensors[1]
        return tuple(tensors)

    save()
    for kind in (
        "decode",
        "uniform",
        "padding",
        "ragged",
        "empty",
        "distinct",
        "negative",
    ):
        arrays = make_inputs(kind)
        values = device_inputs(arrays)
        expected = reference(arrays, args.capacity, args.stride, args.offset)
        labels = LABELS
        if args.with_compact:
            final = expected[:2]
            for records in expected[2:]:
                indices = np.flatnonzero((records >= 0) & (records < 7759456))
                pairs = np.zeros((288, 2), np.int32)
                pairs[: indices.size] = np.stack((indices, records[indices]), axis=1)
                final.extend((pairs.flatten(), np.array([0, indices.size], np.int32)))
            expected = final
            labels = LABELS[:2] + (
                "write_pairs",
                "write_offsets",
                "read_pairs",
                "read_offsets",
            )
        for name, fn in (("native", original), ("pallas", fused)):
            output = fn(*values)
            for label, actual, want in zip(labels, output, expected, strict=True):
                np.testing.assert_array_equal(
                    actual.cpu().numpy(), want, err_msg=f"{kind}/{name}/{label}"
                )
        result["correctness"].append(
            {"kind": kind, "outputs": len(labels), "exact": True}
        )
        save()
    values = device_inputs(make_inputs(args.timing_case))

    def run(fn):
        output = fn(*values)
        sync.synchronize(tensors=list(output), wait=True)

    for _ in range(20):
        run(original)
        run(fused)
    samples = {"native": [], "pallas": []}
    for i in range(args.samples):
        order = (("native", original), ("pallas", fused))
        for name, fn in order if i % 2 == 0 else order[::-1]:
            start = time.perf_counter_ns()
            run(fn)
            samples[name].append((time.perf_counter_ns() - start) / 1000)
    result["synchronous_us"] = {
        name: {
            "median": float(np.median(xs)),
            "p10": float(np.percentile(xs, 10)),
            "p90": float(np.percentile(xs, 90)),
            "samples": xs,
        }
        for name, xs in samples.items()
    }
    save()
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile")):
            for i in range(16):
                with jax.profiler.TraceAnnotation("native", step_num=i):
                    run(original)
                with jax.profiler.TraceAnnotation("pallas", step_num=i):
                    run(fused)
    result["passed"] = True
    save()
    print(
        json.dumps({k: v for k, v in result.items() if k != "synchronous_us"}),
        flush=True,
    )
    print({k: v["median"] for k, v in result["synchronous_us"].items()}, flush=True)


if __name__ == "__main__":
    main()
