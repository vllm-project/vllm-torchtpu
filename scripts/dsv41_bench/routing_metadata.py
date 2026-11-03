# SPDX-License-Identifier: Apache-2.0
"""Standalone, three-way validation of post-communication MoE address tables.

No serving registration changes. The native prefix is extracted from the pinned
production function; native consumer helpers are called without modification.
Pair-dependent pos/slab indices, SC prepush and scale scatter remain outside.
"""

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import region_fusion as original

from vllm_torchtpu.kernels.experimental import decode_preparation as prep

host, push = original.host, original.push


def native_prefix():
    """Extract the unmodified run-table prefix, before pair-dependent lookup."""
    source = inspect.getsource(push.build_routing_tables_sharded)
    parsed = ast.parse(source).body[0]
    selected = []
    active = False
    for node in parsed.body:
        if isinstance(node, ast.Assign):
            names = [n.id for n in ast.walk(node.targets[0]) if isinstance(n, ast.Name)]
            if "run_rows" in names:
                active = True
            if active:
                selected.append(ast.get_source_segment(source, node))
            if active and "recv_rows" in names:
                break
    assert len(selected) == 8, selected
    namespace = dict(push.build_routing_tables_sharded.__globals__)
    exec(
        "def extracted(runs, ep, g_local, shard_stride, e_total):\n"
        + "\n".join("    " + line for s in selected for line in s.splitlines())
        + "\n    return expert_base, region_rows, recv_base, recv_rows\n",
        namespace,
    )
    return namespace["extracted"], hashlib.sha256(source.encode()).hexdigest()


PREFIX, PREFIX_HASH = native_prefix()
STRIDE = host.ragged_stride_bound(4608, 6, 384, 128)
LABELS = (
    "my_recv_base",
    "my_run_start",
    "slab_run_base",
    "run_start_blocks",
    "aligned_blocks",
    "contrib_off_blocks",
    "push_src_blocks",
    "push_len_blocks",
    "push_dst_blocks",
    "block_totals",
    "true_rows",
    "recv_row_off",
    "row_totals",
    "expert_rows",
    "slab_base",
    "counts",
)


def baseline(run_rows, aligned, starts, expert_rows, rank):
    experts, ep = run_rows.shape
    g, me = experts // ep, rank[0]
    base, regions, recv, recv_rows = PREFIX(
        (run_rows, aligned, starts, expert_rows), ep, g, STRIDE, experts
    )
    routing = types.SimpleNamespace(
        run_rows=run_rows,
        run_rows_aligned=aligned,
        run_start_aligned=starts,
        expert_rows_aligned=expert_rows,
        expert_base=base,
        region_rows=regions,
        recv_base=recv,
        recv_rows=recv_rows,
        run_tiles=host.mirror_run_tiles(starts, aligned, 128),
    )
    rows, slab_base = host.shard_expert_slabs(routing, me, e_total=experts, ep=ep)
    local_base = jax.lax.dynamic_slice(base, (me * g,), (g,))
    local_starts = jax.lax.dynamic_slice(starts, (me * g, 0), (g, ep))
    return (
        jax.lax.dynamic_slice(recv, (me, 0, 0), (1, ep, g)).reshape(experts),
        jax.lax.dynamic_slice(starts, (0, me), (experts, 1)).reshape(experts),
        local_base[None, :] + local_starts.T,
        *host.shard_transport_tables_in_blocks(routing, me, e_total=experts, ep=ep),
        *host.shard_push_tables_in_rows(routing, me, e_total=experts, ep=ep),
        rows,
        slab_base,
        host.shard_count_vector(routing, rows, me, e_total=experts, ep=ep),
    )


def flat_candidate(true, aligned, starts, expert_rows, rank):
    """All tables stay [destination, expert]; no [16,24,16] reshape."""
    ep, experts = true.shape
    g = experts // ep
    d = jnp.arange(ep, dtype=jnp.int32)[:, None]
    mine = d == rank[0]
    recv = prep.prefix_sum(aligned, axis=1) - aligned

    def local(table):
        # Mosaic dynamic gather cannot cross source vregs (E384 > lane128).
        # Static group slices expose the small fixed movement to the compiler.
        result = jnp.zeros((table.shape[0], g), jnp.int32)
        for owner in range(ep):
            result += jnp.where(
                rank[0] == owner, table[:, owner * g : (owner + 1) * g], 0
            )
        return result

    local_true, local_aligned, local_start = local(true), local(aligned), local(starts)
    rows = local(expert_rows)
    slab_base = prep.prefix_sum(rows, axis=1) - rows
    local_base = slab_base + rank[0] * STRIDE
    off = prep.prefix_sum(local_aligned, axis=1) - local_aligned
    out_total = local_aligned.sum(axis=1, keepdims=True)
    contrib = off + prep.prefix_sum(out_total, axis=0) - out_total
    recv_off = local(recv)
    tiles = jnp.where(
        aligned > 0,
        ((starts + jnp.maximum(aligned, 1) - 1) >> 7) - (starts >> 7) + 1,
        0,
    )
    local_tiles = local(tiles)
    true_local_total = local_true.sum(axis=1, keepdims=True)
    tiles_local_total = local_tiles.sum(axis=1, keepdims=True)
    true_recv = mine * (true.sum(axis=1, keepdims=True) - true_local_total)
    aligned_recv = mine * (aligned.sum(axis=1, keepdims=True) - out_total)
    block_totals = (
        jnp.concatenate(
            (
                jnp.sum(out_total * ~mine, keepdims=True),
                jnp.sum(aligned_recv, keepdims=True),
            ),
            axis=1,
        )
        >> host.ROWBLK_SHIFT
    )
    row_totals = jnp.concatenate(
        (
            jnp.sum(true_local_total * ~mine, keepdims=True),
            jnp.sum(true_recv, keepdims=True),
        ),
        axis=1,
    )
    count_cols = (
        true_local_total * ~mine,
        true_recv,
        out_total * ~mine,
        aligned_recv,
        mine * (rows > 0).sum(axis=1, keepdims=True),
        tiles_local_total * ~mine,
        mine * (tiles.sum(axis=1, keepdims=True) - tiles_local_total),
        tiles_local_total * mine,
    )
    return (
        jnp.sum(jnp.where(mine, recv, 0), axis=0, keepdims=True),
        jnp.sum(jnp.where(mine, starts, 0), axis=0, keepdims=True),
        local_base + local_start,
        local_start >> host.ROWBLK_SHIFT,
        local_aligned >> host.ROWBLK_SHIFT,
        contrib >> host.ROWBLK_SHIFT,
        contrib >> host.ROWBLK_SHIFT,
        local_aligned >> host.ROWBLK_SHIFT,
        recv_off >> host.ROWBLK_SHIFT,
        block_totals,
        local_true,
        recv_off,
        row_totals,
        rows,
        slab_base,
        jnp.concatenate(count_cols, axis=1).T,
    )


def candidate(xs, interpret=False):
    def adapt(a, b, c, rows, rank):
        return a.T, b.T, c.T, rows[None, :], rank

    fused = (original.fuse_function if interpret else prep.fuse_function)(
        flat_candidate, adapt(*xs), name="dsv41_route_metadata", interpret=interpret
    )

    def call(*args):
        out = list(fused(*adapt(*args)))
        for i in (0, 1, 9, 12, 13, 14):
            out[i] = out[i].reshape(-1)
        for i in (3, 4, 5, 6, 7, 8, 10, 11):
            out[i] = out[i].T
        return tuple(out)

    return call


def reference(true, aligned, starts, expert_rows, rank):
    """Independent scalar prefix accumulation, in native consumer units."""
    experts, ep = true.shape
    me, g = int(rank[0]), experts // ep
    recv = np.zeros((ep, experts), np.int32)
    base = np.zeros(experts, np.int32)
    tiles = np.zeros_like(true)
    for d in range(ep):
        running = 0
        for e in range(experts):
            recv[d, e] = running
            running += int(aligned[e, d])
            if aligned[e, d]:
                tiles[e, d] = (
                    (starts[e, d] + aligned[e, d] - 1) // 128 - starts[e, d] // 128 + 1
                )
    for owner in range(ep):
        running = owner * STRIDE
        for e in range(owner * g, (owner + 1) * g):
            base[e] = running
            running += int(expert_rows[e])
    sl = slice(me * g, (me + 1) * g)
    tr, al, st = true[sl], aligned[sl], starts[sl]
    off = np.zeros_like(al)
    running = 0
    for d in range(ep):
        for j in range(g):
            off[j, d] = running
            running += int(al[j, d])
    remote = [d for d in range(ep) if d != me]
    block_totals = (
        np.array(
            [al[:, remote].sum(), aligned[:, me].sum() - al[:, me].sum()], np.int32
        )
        >> host.ROWBLK_SHIFT
    )
    row_totals = np.array(
        [tr[:, remote].sum(), true[:, me].sum() - tr[:, me].sum()], np.int32
    )
    counts = np.zeros((8, ep), np.int32)
    for d in range(ep):
        if d == me:
            counts[1, d] = true[:, d].sum() - tr[:, d].sum()
            counts[3, d] = aligned[:, d].sum() - al[:, d].sum()
            counts[4, d] = np.count_nonzero(expert_rows[sl])
            counts[6, d] = tiles[:, d].sum() - tiles[sl, d].sum()
            counts[7, d] = tiles[sl, d].sum()
        else:
            counts[0, d] = tr[:, d].sum()
            counts[2, d] = al[:, d].sum()
            counts[5, d] = tiles[sl, d].sum()
    return (
        recv[me],
        starts[:, me],
        base[sl][None, :] + st.T,
        st >> host.ROWBLK_SHIFT,
        al >> host.ROWBLK_SHIFT,
        off >> host.ROWBLK_SHIFT,
        off >> host.ROWBLK_SHIFT,
        al >> host.ROWBLK_SHIFT,
        recv[:, sl].T >> host.ROWBLK_SHIFT,
        block_totals,
        tr,
        recv[:, sl].T,
        row_totals,
        expert_rows[sl],
        base[sl] - base[me * g],
        counts,
    )


def fixture(case, rank, seed=3):
    rng = np.random.default_rng(seed)
    counts = np.zeros((16, 384), np.int32)
    for d in range(16):
        if case == "empty":
            continue
        choices = 384 if case == "uniform" else 24
        ids = rng.integers(0, choices, size=288 * 6)
        if case == "edge":
            ids[:128] = 383
            ids[128:256] = 384  # Invalid route sentinels do not count.
        counts[d] = np.bincount(ids[ids < 384], minlength=384)
    true = counts.T.copy()
    aligned = (true + 1) & -2  # push.run_tables: two-row alignment.
    starts = (np.cumsum(aligned, axis=1, dtype=np.int32) - aligned).astype(np.int32)
    rows = aligned.sum(axis=1, dtype=np.int32)
    return true, aligned, starts, rows, np.array([rank], np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    xs = tuple(jax.device_put(x) for x in fixture("uniform", 0))
    result = {
        "prefix_sha256": PREFIX_HASH,
        "stride": STRIDE,
        "devices": str(jax.devices()),
        "correctness": [],
        "timings": {},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "runtime": {p: importlib.metadata.version(p) for p in ("jax", "jaxlib")},
    }
    funcs = {"baseline": baseline, "candidate": candidate(xs, args.interpret)}
    compiled = {}
    for name, fn in funcs.items():
        lowered = jax.jit(fn).lower(*xs)
        (args.out / f"{name}.stablehlo.txt").write_text(lowered.as_text())
        compiled[name] = lowered.compile()
        (args.out / f"{name}.hlo.txt").write_text(compiled[name].as_text())
        result[name + "_memory"] = str(compiled[name].memory_analysis())
    for case in ("uniform", "skew", "empty", "edge"):
        for rank in (0, 7, 15):
            arrays = fixture(case, rank)
            expected = reference(*arrays)
            inputs = tuple(jax.device_put(x) for x in arrays)
            for name, fn in compiled.items():
                actual = jax.device_get(fn(*inputs))
                for label, got, want in zip(LABELS, actual, expected, strict=True):
                    np.testing.assert_array_equal(
                        got, want, err_msg=f"{case}/{rank}/{name}/{label}"
                    )
            result["correctness"].append({"case": case, "rank": rank, "exact": True})
    for fn in compiled.values():
        for _ in range(10):
            jax.block_until_ready(fn(*xs))
    raw = {name: [] for name in compiled}
    for i in range(args.samples):
        for name in (
            ("baseline", "candidate") if i % 2 == 0 else ("candidate", "baseline")
        ):
            t = time.perf_counter_ns()
            jax.block_until_ready(compiled[name](*xs))
            raw[name].append((time.perf_counter_ns() - t) / 1000)
    for name, values in raw.items():
        result["timings"][name] = {
            "median_us": float(np.median(values)),
            "p10_us": float(np.percentile(values, 10)),
            "p90_us": float(np.percentile(values, 90)),
            "raw_us": values,
        }
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.profile:
        with jax.profiler.trace(str(args.out / "profile-metadata")):
            for _ in range(16):
                for fn in compiled.values():
                    jax.block_until_ready(fn(*xs))
    print(json.dumps({k: v for k, v in result.items() if k != "timings"}))
    print(
        {
            n: {k: v for k, v in s.items() if k != "raw_us"}
            for n, s in result["timings"].items()
        }
    )


if __name__ == "__main__":
    main()
