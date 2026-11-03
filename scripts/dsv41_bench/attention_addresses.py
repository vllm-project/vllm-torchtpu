# SPDX-License-Identifier: Apache-2.0
"""Measured-shape main/index address extension; kernel-only candidate."""

import argparse
import ast
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from attention_state import (
    CACHE,
    ROOT,
    SOURCE,
    Metadata,
    load_preparation,
    make_inputs,
    native_function,
    reference,
)
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

prep = load_preparation()
SHAPES = ((288, 1), (288, 1), (576,), (2,), (576,), (2,), (288,), (288,))


def expanded(
    table: jax.Array,
    starts: jax.Array,
    kv_starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    main: jax.Array,
    index: jax.Array,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    def compact(records):
        r = records.reshape(1, 288)
        valid = (r >= 0) & (r < 7759456)
        rank = prep.prefix_sum(valid.astype(jnp.int32), axis=1) - 1
        match = valid & (rank == jnp.arange(288, dtype=jnp.int32)[:, None])
        token = jnp.sum(jnp.where(match, jnp.arange(288)[None, :], 0), axis=1)
        record = jnp.sum(jnp.where(match, r, 0), axis=1)
        return jnp.stack((token, record), axis=1), jnp.stack(
            (jnp.int32(0), jnp.sum(valid, dtype=jnp.int32))
        )

    def kernel(*refs):
        state, ss, ks, pos, mins, mt, it = (r[...] for r in refs[:7])
        prev, valid, write, read = prep._compressor_state_addresses(
            state, ss, ks, pos, mins
        )
        rows = jnp.arange(288, dtype=jnp.int32)
        request = jnp.minimum(jnp.sum(rows[None, :] >= ks[1:, None], axis=0), 47)
        selected = request[None, :] == jnp.arange(48)[:, None]

        def slots(page_table, capacity):
            logical = pos // 2
            col = jnp.minimum(logical // capacity, page_table.shape[1] - 1)
            col = jnp.where(col < 0, col + page_table.shape[1], col)
            indices = jnp.broadcast_to((col % 128)[None, :], (48, 288))
            chunks = (page_table.shape[1] + 127) // 128
            padded = jnp.pad(
                page_table, ((0, 0), (0, chunks * 128 - page_table.shape[1]))
            )
            per_request = jnp.zeros((48, 288), jnp.int32)
            for chunk in range(chunks):
                tile = padded[:, chunk * 128 : (chunk + 1) * 128]
                gathered = prep.take_along(tile, indices, axis=1)
                per_request += jnp.where(col[None, :] // 128 == chunk, gathered, 0)
            blocks = jnp.sum(jnp.where(selected, per_request, 0), axis=0)
            return jnp.where(rows < ks[-1], blocks * capacity + logical % capacity, -1)

        main_slots, index_slots = slots(mt, 128), slots(it, 256)
        records = jnp.where((valid != 0) & (main_slots >= 0), main_slots, -1)
        outputs = (prev, valid, *compact(write), *compact(read), records, index_slots)
        for ref, value in zip(refs[7:], outputs, strict=True):
            ref[...] = value

    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(s, jnp.int32)
            for s in ((288,), (288,), (288, 2), (2,), (288, 2), (2,), (288,), (288,))
        ),
        name="dsv41_compressor_addresses_packed_c128_i256",
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=16 * 1024 * 1024),
    )(table, starts, kv_starts, positions, minimum, main, index)
    return tuple(x.reshape(s) for x, s in zip(outputs, SHAPES, strict=True))


def native_addresses():
    import torch

    tree = ast.parse(SOURCE.read_text())
    helpers = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("_block_table", "_token_requests", "_cache_slots")
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(helpers, []), str(SOURCE), "exec"), namespace)
    caches = (SimpleNamespace(shape=(1, 128)), SimpleNamespace(shape=(1, 256)))

    def call(starts, positions, minimum, main, index, valid):
        values = []
        for table, cache in zip((main, index), caches, strict=True):
            meta = Metadata(table, minimum, starts)
            values.append(
                namespace["_cache_slots"](cache, meta, positions, compress_ratio=2)
            )
        records = (
            values[0]
            .clamp(min=0)
            .masked_fill(~(valid.reshape(-1) != 0) | (values[0] < 0), -1)
        )
        # These are the actual bridge boundary dtypes: publish int32, index int64
        # cast to int32 here equally for both paths; the index writer accepts int.
        return records.int(), values[1].int()

    return call


def inputs(kind):
    arrays = list(
        make_inputs(kind if kind != "negative_positions" else "uniform", 16, 0)
    )
    rng = np.random.default_rng(94819)
    main = rng.integers(0, 400000, (48, 386), dtype=np.int32)
    index = rng.integers(0, 400000, (48, 193), dtype=np.int32)
    if kind == "negative":
        main[::3, :] = -1
        index[::3, :] = -1
    if kind == "negative_positions":
        arrays[3][:8] = [-1, -2, -255, -256, -257, -511, -512, -513]
    return (*arrays, main, index)


def expected(arrays):
    prev, valid, write, read = reference(arrays[:5], 16, 16, 0)
    result = [prev, valid]
    for records in (write, read):
        chosen = np.flatnonzero((records >= 0) & (records < 7759456))
        pairs = np.zeros((288, 2), np.int32)
        pairs[: len(chosen)] = np.stack((chosen, records[chosen]), axis=1)
        result.extend((pairs.reshape(-1), np.array([0, len(chosen)], np.int32)))
    starts, positions = arrays[2], arrays[3]
    slots = []
    for table, capacity in ((arrays[5], 128), (arrays[6], 256)):
        out = np.full(288, -1, np.int32)
        for t, pos in enumerate(positions.tolist()):
            if t < starts[-1]:
                req = min(int(np.searchsorted(starts[1:], t, side="right")), 47)
                logical = pos // 2
                block = int(table[req, min(logical // capacity, table.shape[1] - 1)])
                out[t] = block * capacity + logical % capacity
        slots.append(out)
    slots[0][(valid.reshape(-1) == 0) | (slots[0] < 0)] = -1
    return (*result, *slots)


def fullconsumer_functions(ops, mesh):
    """Compare against the retained production bridge, with live address outputs."""
    import torch
    from torch_tpu._internal import pallas

    old_op = ops.compress_rope_packed(1e-6)
    addresses = native_addresses()

    def fused(
        table: jax.Array,
        starts: jax.Array,
        positions: jax.Array,
        minimum: jax.Array,
        x: jax.Array,
        weight: jax.Array,
        gamma: jax.Array,
        rope: jax.Array,
        low: jax.Array,
        high: jax.Array,
        mt: jax.Array,
        it: jax.Array,
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        prev, valid, wp, wo, rp, ro, records, slots = expanded(
            table, starts, starts, positions, minimum, mt, it
        )
        latent, words, scales, banks = ops.compress_kernel.fused_compress_rope(
            x,
            weight,
            gamma,
            rope,
            prev,
            valid,
            (low, high),
            ratio=2,
            eps=1e-6,
            ring_tables=(wp, wo, rp, ro),
        )
        return latent, words, scales, *banks, valid[:, 0] != 0, records, slots

    new_op = pallas.jax_op(
        "pallas::dsv41_consumer_address_tiled_v26",
        fused,
        mesh=mesh,
        input_partition_specs=(jax.sharding.PartitionSpec(),) * 12,
        donate_argnums=(8, 9),
    )

    @new_op.register_fake
    def fake(table, starts, pos, minimum, x, weight, gamma, rope, a, b, mt, it):
        return (
            x.new_empty((288, 512)),
            x.new_empty((288, 128), dtype=torch.int32),
            x.new_empty((288, 1), dtype=torch.float32),
            torch.empty_like(a),
            torch.empty_like(b),
            x.new_empty((288,), dtype=torch.bool),
            pos.new_empty((288,)),
            pos.new_empty((288,)),
        )

    def baseline(table, starts, pos, minimum, x, weight, gamma, rope, a, b, mt, it):
        latent, words, scales, low, high, valid = old_op(
            table, starts, starts, pos, minimum, x, weight, gamma, rope, a, b
        )
        a.copy_(low)
        b.copy_(high)
        records, slots = addresses(starts, pos, minimum, mt, it, valid)
        return latent, words, scales, valid, records, slots

    def candidate(table, starts, pos, minimum, x, weight, gamma, rope, a, b, mt, it):
        latent, words, scales, low, high, valid, records, slots = new_op(
            table, starts, pos, minimum, x, weight, gamma, rope, a, b, mt, it
        )
        a.copy_(low)
        b.copy_(high)
        return latent, words, scales, valid, records, slots

    return baseline, candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_tpu  # noqa: F401
    from torch_tpu._internal import pallas, sync

    torch.set_num_threads(1)
    torch.empty(1, device="tpu").cpu()
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",))

    def register(name, fn, count, shapes):
        op = pallas.jax_op(
            name,
            fn,
            mesh=mesh,
            input_partition_specs=(jax.sharding.PartitionSpec(),) * count,
        )

        @op.register_fake
        def fake(*xs):
            return tuple(xs[0].new_empty(s, dtype=torch.int32) for s in shapes)

        return op

    def retained(
        t: jax.Array, s: jax.Array, k: jax.Array, p: jax.Array, m: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return prep.compressor_state_packed(t, s, k, p, m, total_records=7759456)

    old = register("pallas::dsv41_addr_retained_v23", retained, 5, SHAPES[:6])
    new = register("pallas::dsv41_addr_expanded_tiled_v25", expanded, 7, SHAPES)
    native = native_function(16, 16, 0)
    addresses = native_addresses()

    def old_call(t, s, k, p, m, mt, it):
        a = old(t, s, k, p, m)
        return (*a, *addresses(k, p, m, mt, it, a[1]))

    baseline = torch.compile(old_call, backend="tpu", fullgraph=True, dynamic=False)
    candidate = torch.compile(
        lambda *xs: new(*xs), backend="tpu", fullgraph=True, dynamic=False
    )
    original = torch.compile(native, backend="tpu", fullgraph=True, dynamic=False)
    address_original = torch.compile(
        addresses, backend="tpu", fullgraph=True, dynamic=False
    )
    result = {
        "runtime": {
            n: importlib.metadata.version(n)
            for n in ("torch", "torch-tpu", "jax", "jaxlib", "libtpu", "numpy")
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                SOURCE,
                CACHE,
                Path(__file__),
                ROOT / "src/vllm_torchtpu/kernels/experimental/decode_preparation.py",
            )
        },
        "correctness": [],
        "scope": "metadata + main records/index int32 slots; no consumer or SC",
    }

    def save():
        (args.out / "results.json").write_text(json.dumps(result, indent=2))

    def device(a):
        x = [torch.from_numpy(v).to("tpu") for v in a]
        if np.array_equal(a[1], a[2]):
            x[2] = x[1]
        return x

    save()
    for case in (
        "decode",
        "uniform",
        "padding",
        "ragged",
        "empty",
        "distinct",
        "negative",
        "long_request",
        "record_bounds",
        "negative_positions",
    ):
        a = inputs(case)
        xs = device(a)
        want = expected(a)
        raw = original(*xs[:5])
        for x, y in zip(raw, reference(a[:5], 16, 16, 0), strict=True):
            np.testing.assert_array_equal(x.cpu().numpy(), y)
        for x, y in zip(
            address_original(xs[2], xs[3], xs[4], xs[5], xs[6], raw[1]),
            want[6:],
            strict=True,
        ):
            np.testing.assert_array_equal(x.cpu().numpy(), y)
        for name, fn in (("retained", baseline), ("expanded", candidate)):
            for i, (x, y) in enumerate(zip(fn(*xs), want, strict=True)):
                np.testing.assert_array_equal(
                    x.cpu().numpy(), y, err_msg=f"{case}/{name}/{i}"
                )
        result["correctness"].append({"case": case, "outputs": 8, "exact": True})
        save()
    xs = device(inputs("decode"))

    def run(fn):
        sync.synchronize(tensors=list(fn(*xs)), wait=True)

    for _ in range(20):
        run(baseline)
        run(candidate)
    samples = {"retained": [], "expanded": []}
    for i in range(200):
        order = (("retained", baseline), ("expanded", candidate))
        for name, fn in order if i % 2 == 0 else order[::-1]:
            start = time.perf_counter_ns()
            run(fn)
            samples[name].append((time.perf_counter_ns() - start) / 1000)
    result["synchronous_us"] = {
        k: {"median": float(np.median(v)), "samples": v} for k, v in samples.items()
    }
    save()
    with jax.profiler.trace(str(args.out / "profile")):
        for i in range(16):
            for name, fn in (("retained", baseline), ("expanded", candidate)):
                with jax.profiler.TraceAnnotation(name, step_num=i):
                    run(fn)
    result["passed"] = True
    save()
    print({k: v["median"] for k, v in result["synchronous_us"].items()}, flush=True)


if __name__ == "__main__":
    main()
