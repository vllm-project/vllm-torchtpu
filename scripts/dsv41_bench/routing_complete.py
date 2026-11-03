# SPDX-License-Identifier: Apache-2.0
"""Routing replay retaining only outputs used by the production MoE consumer."""

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import routing_metadata as meta
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

host, push, prep = meta.host, meta.push, meta.prep
EP, EXPERTS, TOKENS, TOPK = 16, 384, 288, 6
BLOCK = host.routing_block(TOKENS, TOPK)
STRIDE = meta.STRIDE
META_USED = (3, 4, 5, 8, 10, 13, 14, 15)
LABELS = ("pos", "mirror_pos", "local_slab_row") + tuple(
    meta.LABELS[i] for i in META_USED
)


def native(pairs, local, hist, true, aligned, starts, rows, rank):
    """Call the unchanged production plan and retain its actual live outputs."""
    gathered = host.GatheredPairs(
        expert_blocks=pairs.reshape(-1, BLOCK),
        local_blocks=local,
        block_hist=hist,
        rows_by_dest=true.T,
        n_tokens=EP * TOKENS,
        topk=TOPK,
    )
    routing = push.build_routing_tables_sharded(
        rank[0],
        gathered,
        (true, aligned, starts, rows),
        e_total=EXPERTS,
        ep=EP,
        t_local=TOKENS,
        block=BLOCK,
        tile_m=128,
        shard_stride=STRIDE,
    )
    blocks = host.shard_transport_tables_in_blocks(
        routing, rank[0], e_total=EXPERTS, ep=EP
    )
    rows_table = host.shard_push_tables_in_rows(
        routing, rank[0], e_total=EXPERTS, ep=EP
    )
    expert_rows, base = host.shard_expert_slabs(
        routing, rank[0], e_total=EXPERTS, ep=EP
    )
    return (
        routing.pos,
        routing.mirror_pos,
        host.local_slab_rows(routing, rank[0], shard_stride=STRIDE),
        *blocks[:3],
        blocks[5],
        rows_table[0],
        expert_rows,
        base,
        host.shard_count_vector(routing, expert_rows, rank[0], e_total=EXPERTS, ep=EP),
    )


def position_metadata_body(true, aligned, starts, rows, rank, local, hist):
    tables = meta.flat_candidate(true, aligned, starts, rows, rank)
    offset = prep.prefix_sum(hist, axis=0) - hist
    bins = jnp.arange(EXPERTS, dtype=jnp.int32)
    within = host._rank_within_block(local, BLOCK)
    pos_table = offset + tables[0]
    mirror_table = (
        offset + ((tables[0] >> 7) + 2 * bins[None, :]) * 128 + (tables[1] & 127)
    )
    pos = host._lookup_per_pair(pos_table, local, bins) + within
    mirror = host._lookup_per_pair(mirror_table, local, bins) + within
    return pos, mirror, tables[2], *(tables[i] for i in META_USED)


def slab_call(interpret=False):
    """One source rank per program; preserve stable order within that source."""
    nblock = TOKENS * TOPK // BLOCK
    groups = EXPERTS // EP

    def kernel(pair_ref, base_ref, rank_ref, out_ref):
        pairs, me = pair_ref[...], rank_ref[0]
        # A one-row BlockSpec cannot slice a [16,24] TPU tile. Keep the small
        # base table resident across source programs, then select in VMEM.
        base = jnp.sum(
            jnp.where(jnp.arange(EP)[:, None] == pl.program_id(0), base_ref[...], 0),
            axis=0,
            keepdims=True,
        )
        bins = me * groups + jnp.arange(groups, dtype=jnp.int32)
        hist = host._block_hist(pairs, bins)
        table = prep.prefix_sum(hist, axis=0) - hist + base
        within = host._rank_within_block(pairs, BLOCK)
        row = host._lookup_per_pair(table, pairs, bins) + within - me * STRIDE
        off = host.off_shard_slab_row(STRIDE, EP) - me * STRIDE
        valid = (pairs >= me * groups) & (pairs < (me + 1) * groups)
        out_ref[...] = jnp.where(valid, row, off)

    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((EP, nblock, BLOCK), jnp.int32),
        grid=(EP,),
        in_specs=(
            pl.BlockSpec((None, nblock, BLOCK), lambda source: (source, 0, 0)),
            pl.BlockSpec((EP, groups), lambda source: (0, 0)),
            pl.BlockSpec((1,), lambda source: (0,)),
        ),
        out_specs=pl.BlockSpec((None, nblock, BLOCK), lambda source: (source, 0, 0)),
        interpret=interpret,
        name="dsv41_routing_slab",
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 * 1024),
    )


def make_candidate(xs, interpret=False, slab_mode="grid"):
    def adapt(pairs, local, hist, true, aligned, starts, rows, rank):
        del pairs
        return true.T, aligned.T, starts.T, rows[None, :], rank, local, hist

    fuse = meta.original.fuse_function if interpret else prep.fuse_function
    position = fuse(
        position_metadata_body,
        adapt(*xs),
        name="dsv41_routing_position_metadata",
        interpret=interpret,
    )
    slab = slab_call(interpret)

    def call(*args):
        tables = list(position(*adapt(*args)))
        slabs = (
            native(*args)[2]
            if slab_mode == "native"
            else slab(args[0].reshape(EP, -1, BLOCK), tables[2], args[-1])
        )
        # Input/output layout adapters are deliberately inside the timed call.
        for i in range(3, 8):
            tables[i] = tables[i].T
        for i in (8, 9):
            tables[i] = tables[i].reshape(-1)
        return (
            tables[0].reshape(TOKENS, TOPK),
            tables[1].reshape(TOKENS, TOPK),
            slabs.reshape(-1),
            *tables[3:],
        )

    return call


def fixture(case, rank, seed=7):
    rng = np.random.default_rng(seed)
    pairs = rng.integers(
        0, 24 if case == "skew" else EXPERTS, (EP, TOKENS * TOPK), dtype=np.int32
    )
    if case == "empty":
        pairs[:] = EXPERTS
    elif case == "edge":
        pairs[:, ::17] = EXPERTS
        pairs[:, 56:72] = EXPERTS - 1
        pairs[7, 0:128] = 7 * (EXPERTS // EP)
    true = np.stack(
        [np.bincount(p[p < EXPERTS], minlength=EXPERTS) for p in pairs]
    ).T.astype(np.int32)
    aligned = (true + (host.ROWBLK - 1)) & -host.ROWBLK
    starts = np.cumsum(aligned, axis=1, dtype=np.int32) - aligned
    rows = aligned.sum(axis=1, dtype=np.int32)
    local = pairs[rank].reshape(-1, BLOCK).copy()
    hist = np.stack(
        [np.bincount(p[p < EXPERTS], minlength=EXPERTS) for p in local]
    ).astype(np.int32)
    return (
        pairs.reshape(-1),
        local,
        hist,
        true,
        aligned,
        starts,
        rows,
        np.array([rank], np.int32),
    )


def reference(pairs, local, hist, true, aligned, starts, rows, rank):
    """Stable per-source expert counters, independent of block histogram math."""
    del hist
    me, groups = int(rank[0]), EXPERTS // EP
    tables = meta.reference(true, aligned, starts, rows, rank)
    pos, mirror = np.zeros(local.size, np.int32), np.zeros(local.size, np.int32)
    counts = np.zeros(EXPERTS, np.int32)
    block_invalid = {}
    for i, e in enumerate(local.reshape(-1)):
        if i % BLOCK == 0:
            block_invalid = {}
        if 0 <= e < EXPERTS:
            pos[i] = tables[0][e] + counts[e]
            mirror[i] = (
                ((tables[0][e] // 128) + 2 * e) * 128
                + (starts[e, me] % 128)
                + counts[e]
            )
            counts[e] += 1
        else:
            pos[i] = mirror[i] = block_invalid.get(int(e), 0)
            block_invalid[int(e)] = int(pos[i]) + 1
    off = host.off_shard_slab_row(STRIDE, EP) - me * STRIDE
    slabs = np.full((EP, TOKENS * TOPK), off, np.int32)
    for source, source_pairs in enumerate(pairs.reshape(EP, -1)):
        counters = np.zeros(groups, np.int32)
        for i, e in enumerate(source_pairs):
            if me * groups <= e < (me + 1) * groups:
                le = e - me * groups
                slabs[source, i] = tables[14][le] + starts[e, source] + counters[le]
                counters[le] += 1
    return (
        pos.reshape(TOKENS, TOPK),
        mirror.reshape(TOKENS, TOPK),
        slabs.reshape(-1),
        *(tables[i] for i in META_USED),
    )


def with_scale_scatter(fn):
    """Keep the production barrier/scatter as an independent consumer."""

    def call(*inputs):
        outputs = fn(*inputs[:-1])
        slab_row, scale_bits = jax.lax.optimization_barrier((outputs[2], inputs[-1]))
        scale_rows = host.act_scale_slab_rows(STRIDE)
        bits = (
            jnp.zeros((scale_rows * 128,), jnp.int32)
            .at[slab_row]
            .add(jnp.repeat(scale_bits, TOPK), mode="drop")
        )
        scale = jax.lax.bitcast_convert_type(bits, jnp.float32).reshape(scale_rows, 128)
        return *outputs[:2], *outputs[3:], scale

    return call


def scale_fixture():
    # The production scatter transports float32 bits without FP arithmetic.
    rng = np.random.default_rng(19)
    return rng.integers(0, 2**32, EP * TOKENS, dtype=np.uint32).view(np.int32)


def scale_reference(outputs, bits):
    scale = np.zeros(host.act_scale_slab_rows(STRIDE) * 128, np.int32)
    rows = outputs[2]
    valid = (rows >= 0) & (rows < scale.size)
    np.add.at(scale, rows[valid], np.repeat(bits, TOPK)[valid])
    return *outputs[:2], *outputs[3:], scale.view(np.float32).reshape(-1, 128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--with-scale", action="store_true")
    parser.add_argument("--slab-mode", choices=("grid", "native"), default="grid")
    parser.add_argument(
        "--timing-case", choices=("uniform", "skew", "empty", "edge"), default="uniform"
    )
    parser.add_argument("--timing-rank", type=int, choices=range(EP), default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    xs = tuple(jax.device_put(x) for x in fixture(args.timing_case, args.timing_rank))
    funcs = {
        "baseline": native,
        "candidate": make_candidate(xs, args.interpret, args.slab_mode),
    }
    labels = LABELS
    if args.with_scale:
        funcs = {name: with_scale_scatter(fn) for name, fn in funcs.items()}
        xs = (*xs, jax.device_put(scale_fixture()))
        labels = (*LABELS[:2], *LABELS[3:], "scale_slab")
    results = {
        "correctness": [],
        "timings": {},
        "devices": str(jax.devices()),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib")},
        "with_scale": args.with_scale,
        "slab_mode": args.slab_mode,
        "timing_case": args.timing_case,
        "timing_rank": args.timing_rank,
    }
    compiled = {}
    for name, fn in funcs.items():
        lowered = jax.jit(fn).lower(*xs)
        (args.out / f"{name}.stablehlo.txt").write_text(lowered.as_text())
        compiled[name] = lowered.compile()
        (args.out / f"{name}.hlo.txt").write_text(compiled[name].as_text())
        results[name + "_memory"] = str(compiled[name].memory_analysis())
    for case in ("uniform", "skew", "empty", "edge"):
        for rank in (0, 7, 15):
            inputs = fixture(case, rank)
            expected = reference(*inputs)
            if args.with_scale:
                bits = scale_fixture()
                expected = scale_reference(expected, bits)
                inputs = (*inputs, bits)
            device_inputs = tuple(jax.device_put(x) for x in inputs)
            for name, fn in compiled.items():
                actual = jax.device_get(fn(*device_inputs))
                for label, got, want in zip(labels, actual, expected, strict=True):
                    if got.dtype == np.float32:
                        got, want = got.view(np.uint32), want.view(np.uint32)
                    np.testing.assert_array_equal(
                        got, want, err_msg=f"{case}/{rank}/{name}/{label}"
                    )
            results["correctness"].append({"case": case, "rank": rank, "exact": True})
    for fn in compiled.values():
        for _ in range(10):
            jax.block_until_ready(fn(*xs))
    raw = {name: [] for name in compiled}
    for i in range(args.samples):
        for name in (
            ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        ):
            start = time.perf_counter_ns()
            jax.block_until_ready(compiled[name](*xs))
            raw[name].append((time.perf_counter_ns() - start) / 1000)
    for name, values in raw.items():
        results["timings"][name] = {
            "median_us": float(np.median(values)),
            "p10_us": float(np.percentile(values, 10)),
            "p90_us": float(np.percentile(values, 90)),
            "raw_us": values,
        }
    (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile-routing")):
            for _ in range(16):
                for fn in compiled.values():
                    jax.block_until_ready(fn(*xs))
    print(json.dumps({k: v for k, v in results.items() if k != "timings"}))
    print(
        {
            n: {k: v for k, v in t.items() if k != "raw_us"}
            for n, t in results["timings"].items()
        }
    )


if __name__ == "__main__":
    main()
