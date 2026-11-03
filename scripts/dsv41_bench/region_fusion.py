# SPDX-License-Identifier: Apache-2.0
"""Kernel-only region replays against pinned production callables.

Revision 3 contains full dispatch tables, count-based local reorder, DMA
combine including clamp/mask, and SWA metadata/sort/pack plus original SC
writers. Earlier lowering probes remain selectable for failure reproduction.
No serving registration or production dispatch is changed.
"""

import argparse
import ast
import functools
import importlib.metadata
import json
import logging
import sys
import time
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROOT = Path(__file__).resolve().parents[2]
for name in ("vllm_torchtpu", "vllm_torchtpu.kernels"):
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / "src" / name.replace(".", "/"))]
    sys.modules[name] = package
logger_module = types.ModuleType("vllm_torchtpu.logger")
logger = logging.getLogger("region_fusion")
logger.warning_once = functools.lru_cache(None)(logger.warning)
logger_module.init_logger = lambda name: logger
sys.modules[logger_module.__name__] = logger_module

from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import (  # noqa: E402
    prepare_swa_metadata,
)
from vllm_torchtpu.kernels.experimental.push_fused_moe import host as push  # noqa: E402
from vllm_torchtpu.kernels.experimental.push_fused_moe.layer import (  # noqa: E402
    _combine_arrivals_fs,
)
from vllm_torchtpu.kernels.fused_moe.v2 import host  # noqa: E402
from vllm_torchtpu.kernels.mla.paged_swa import write_swa_pages  # noqa: E402


def take_along(values, indices, axis):
    """Explicit batched gather; retain singleton dimensions for Mosaic."""
    batch = tuple(i for i in range(values.ndim) if i != axis)
    return lax.gather(
        values,
        indices[..., None],
        lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(axis,),
            start_index_map=(axis,),
            operand_batching_dims=batch,
            start_indices_batching_dims=batch,
        ),
        slice_sizes=(1,) * values.ndim,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def prefix_sum(a, axis=None, dtype=None, out=None):
    """Integer-only inclusive scan with explicit shift/add dependencies."""
    assert out is None
    if axis is None:
        a, axis = a.reshape(-1), 0
    if dtype is not None:
        a = a.astype(dtype)
    assert jnp.issubdtype(a.dtype, jnp.integer)
    distance = 1
    while distance < a.shape[axis]:
        tail = lax.slice_in_dim(a, 0, a.shape[axis] - distance, axis=axis)
        pad_shape = list(a.shape)
        pad_shape[axis] = distance
        a = a + jnp.concatenate((jnp.zeros(pad_shape, a.dtype), tail), axis=axis)
        distance *= 2
    return a


def pallas_host_copy(fn, memo=None):
    """Prototype-only clone: retain source math; substitute unsupported scan.

    This avoids changing imported baseline globals. Every recursively cloned
    function remains the actual pinned host helper, except its integer scan.
    """
    memo = {} if memo is None else memo
    if fn in memo:
        return memo[fn]
    namespace = dict(fn.__globals__)
    copy = types.FunctionType(
        fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__
    )
    memo[fn] = copy
    copy.__kwdefaults__ = fn.__kwdefaults__
    namespace["jnp"] = types.SimpleNamespace(**{**vars(jnp), "cumsum": prefix_sum})
    for key in fn.__code__.co_names:
        value = namespace.get(key)
        if isinstance(value, types.FunctionType) and value.__module__.startswith(
            "vllm_torchtpu.kernels."
        ):
            namespace[key] = pallas_host_copy(value, memo)
    return copy


def dispatch_region(counts, seg_start, rank):
    ep, experts = counts.shape
    local = experts // ep
    runs = push.run_tables(counts)
    seg, meta, visit_all = push.push_dispatch_segments(
        runs[3], runs[2], counts, rank[0], seg_start, e_total=experts, ep=ep
    )
    recv = meta[local + 1 : 2 * local + 1]
    seg_n = (seg[:experts] & 65535).reshape(local, ep)
    remote = (jnp.arange(ep)[None, :] != rank[0]).astype(jnp.int32)
    visit = lax.dynamic_slice(visit_all, (rank[0], 0), (1, local)).reshape(local)
    hit = (visit[:2, None] == jnp.arange(local)[None, :]) & (visit[:2, None] >= 0)
    rows = jnp.sum(jnp.where(hit, recv[None, :], 0), axis=1)
    n_rem = jnp.sum(seg_n[2:] * remote).reshape(1)
    meta = jnp.concatenate((meta[: 2 * local + 1], n_rem, meta[2 * local + 2 :]))
    sc_counts = jnp.stack(
        (rows[0], rows[1], jnp.sum(seg_n[:1] * remote), jnp.sum(seg_n[1:2] * remote))
    )
    return *runs, seg, meta, visit, sc_counts


def dispatch_flat(counts, seg_start, rank):
    """Same tables, without dynamic slices or non-tiled expert reshapes."""
    ep, experts = counts.shape
    local = experts // ep
    e = jnp.arange(experts, dtype=jnp.int32)
    d = jnp.arange(ep, dtype=jnp.int32)
    aligned = (counts + 1) & -2
    starts = prefix_sum(aligned, axis=0) - aligned
    rows = aligned.sum(axis=0)
    same = e[:, None] // local == e[None, :] // local
    key = e % local - rows * local
    order = jnp.sum(same & (key[None, :] < key[:, None]), axis=1)
    # Each segment is visit-major, then destination-major.
    visit_number, destination = e // ep, e % ep
    selected = (e[None, :] // local == destination[:, None]) & (
        order[None, :] == visit_number[:, None]
    )
    chosen = jnp.sum(jnp.where(selected, e[None, :], 0), axis=1)
    selected = selected & (rows[None, :] > 0)
    mine = jnp.sum(jnp.where(d[:, None] == rank[0], counts, 0), axis=0)
    run_mine = jnp.sum(jnp.where(d[:, None] == rank[0], starts, 0), axis=0)
    base = jnp.sum(
        jnp.where(same & (e[None, :] < e[:, None]), rows[None, :], 0), axis=1
    )

    def select(values):
        return jnp.sum(jnp.where(selected, values[None, :], 0), axis=1)

    count, source, target = select(mine), select(seg_start), select(base + run_mine)
    seg = jnp.concatenate(
        (
            (destination << 24) | ((chosen % local) << 16) | count,
            source,
            jnp.where(count > 0, target, 0),
        )
    )
    local_e = jnp.arange(local, dtype=jnp.int32)
    recv = jnp.sum(
        jnp.where(
            e[None, :] == rank[0] * local + local_e[:, None],
            counts.sum(axis=0)[None, :],
            0,
        ),
        axis=1,
    )
    remote = destination != rank[0]
    remaining = jnp.sum(jnp.where(remote & (visit_number >= 2), count, 0))
    meta = jnp.concatenate(
        (jnp.zeros(local + 1, jnp.int32), recv, jnp.stack((remaining, count.sum())))
    )
    visit = jnp.sum(
        jnp.where(
            (destination[None, :] == rank[0])
            & (visit_number[None, :] == local_e[:, None]),
            (chosen % local)[None, :],
            0,
        ),
        axis=1,
    )
    recv_two = jnp.sum(
        jnp.where(visit[:2, None] == local_e[None, :], recv[None, :], 0), axis=1
    )
    sc = jnp.concatenate(
        (
            recv_two,
            jnp.stack(
                [
                    jnp.sum(jnp.where(remote & (visit_number == v), count, 0))
                    for v in range(2)
                ]
            ),
        )
    )
    return counts.T, aligned.T, starts.T, rows, seg, meta, visit, sc


def combine_tiled(xs, interpret=False):
    """Feature-tiled gather, scale lookup, K accumulation and valid mask."""
    arrivals, scales, pos, mirror, weights, valid = xs
    tokens, topk = pos.shape
    rows, groups, lanes = arrivals.shape

    def kernel(a, s, p, m, w, v, out):
        data = a[...].astype(jnp.float32)
        scale = s[...]
        acc = None
        for k in range(topk):
            index = jnp.minimum(p[k : k + 1, :], rows - 1)
            scale_index = jnp.minimum(m[k : k + 1, :], scales.size - 1)
            coeff = take_along(scale, scale_index, axis=1) * w[k : k + 1, :]
            values = take_along(
                data, jnp.broadcast_to(index.T, (tokens, lanes)), axis=0
            ).astype(jnp.float32)
            term = values * coeff.T
            acc = term if acc is None else acc + term
        out[...] = jnp.where(v[...].T, acc.astype(jnp.bfloat16), jnp.bfloat16(0))

    all_spec = pl.BlockSpec(memory_space=pltpu.VMEM)
    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((tokens, groups * lanes), jnp.bfloat16),
        grid=(groups,),
        in_specs=(
            pl.BlockSpec((rows, lanes), lambda g: (0, g)),
            all_spec,
            all_spec,
            all_spec,
            all_spec,
            all_spec,
        ),
        out_specs=pl.BlockSpec((tokens, lanes), lambda g: (0, g)),
        name="dsv41_moe_combine_tiled",
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 * 1024),
    )

    def wrapped(a, s, p, m, w, v):
        return call(
            a.reshape(rows, groups * lanes), s.reshape(1, -1), p.T, m.T, w.T, v[None, :]
        )

    return wrapped


def dispatch_layout_call(xs, interpret=False):
    """Keep run-table storage in the baseline's expert-minor layout."""

    def body(*args):
        a, b, c, *rest = dispatch_flat(*args)
        return a.T, b.T, c.T, *rest

    fused = fuse_function(
        body, xs, name="dsv41_moe_dispatch_layout", interpret=interpret
    )

    def call(*args):
        a, b, c, *rest = fused(*args)
        return a.T, b.T, c.T, *rest

    return call


def combine_dma(xs):
    """Reuse the existing pipeline with 16-token steps and two buffer slots."""
    from vllm_torchtpu.kernels.experimental.push_fused_moe import kernel as pk

    original = pk._build_combine_fs_kernel
    factory = types.FunctionType(
        original.__code__,
        {
            **original.__globals__,
            "COMBINE_PARITIES": 2,
            "COMBINE_LEAD": 1,
            "combine_step_tokens": lambda tokens: 16 if tokens == 288 else 0,
        },
    )
    a, s, p, m, w, v = xs
    fused = factory(
        t_local=288,
        topk=6,
        hidden=5120,
        lane_blocks=40,
        wire_dtype=a.dtype,
        out_dtype=jnp.bfloat16,
    )

    def call(a, s, p, m, w, v):
        out = fused(a, s, jnp.minimum(m, s.size - 1), w, jnp.minimum(p, a.shape[0] - 1))
        return jnp.where(v[:, None], out, jnp.bfloat16(0))

    return call


def combine_complete(xs):
    """Extend the pinned DMA kernel with scalar clamp and final valid mask.

    AST edits are confined to this replay. The production function and its
    cache stay untouched; assertions make a changed source fail explicitly.
    """
    from vllm_torchtpu.kernels.experimental.push_fused_moe import kernel as pk

    path = ROOT / "src/vllm_torchtpu/kernels/experimental/push_fused_moe/kernel.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_build_combine_fs_kernel"
    )
    edits = {"pos": 0, "scale": 0, "mask": 0, "prefetch": 0, "call": 0}

    class Complete(ast.NodeTransformer):
        def visit_FunctionDef(self, n):
            if n.name == "kernel":
                n.args.args.insert(4, ast.arg(arg="valid_sm"))
            if n.name == "fn":
                n.args.args.append(ast.arg(arg="valid"))
            return self.generic_visit(n)

        def visit_Subscript(self, n):
            n = self.generic_visit(n)
            if isinstance(n.value, ast.Name) and n.value.id in ("pos_sm", "scl_sm"):
                key = "pos" if n.value.id == "pos_sm" else "scale"
                limit = xs[0].shape[0] - 1 if key == "pos" else xs[1].size - 1
                edits[key] += 1
                return ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id="jnp", ctx=ast.Load()),
                        attr="minimum",
                        ctx=ast.Load(),
                    ),
                    args=[n, ast.Constant(value=limit)],
                    keywords=[],
                )
            return n

        def visit_Assign(self, n):
            n = self.generic_visit(n)
            target = n.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "tile_vm"
            ):
                wrapped = ast.parse(
                    "jnp.where(valid_sm[base + i] != 0, value, jnp.zeros_like(value))",
                    mode="eval",
                ).body

                class Value(ast.NodeTransformer):
                    def visit_Name(self, v):
                        return n.value if v.id == "value" else v

                n.value = Value().visit(wrapped)
                edits["mask"] += 1
            return n

        def visit_Call(self, n):
            n = self.generic_visit(n)
            if (
                isinstance(n.func, ast.Attribute)
                and n.func.attr == "PrefetchScalarGridSpec"
            ):
                kw = next(k for k in n.keywords if k.arg == "num_scalar_prefetch")
                assert kw.value.value == 4
                kw.value = ast.Constant(value=5)
                edits["prefetch"] += 1
            if isinstance(n.func, ast.Name) and n.func.id == "call":
                n.args.insert(
                    4,
                    ast.parse("valid.astype(jnp.int32).reshape(-1)", mode="eval").body,
                )
                edits["call"] += 1
            return n

    node = Complete().visit(node)
    assert edits == dict(pos=1, scale=1, mask=1, prefetch=1, call=1), edits
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        **vars(pk),
        "COMBINE_PARITIES": 2,
        "COMBINE_LEAD": 1,
        "combine_step_tokens": lambda tokens: 16 if tokens == 288 else 0,
    }
    exec(compile(module, str(path), "exec"), namespace)
    fused = namespace[node.name](
        t_local=288,
        topk=6,
        hidden=5120,
        lane_blocks=40,
        wire_dtype=xs[0].dtype,
        out_dtype=jnp.bfloat16,
    )
    return lambda a, s, p, m, w, v: fused(a, s, m, w, p, v)


def route_region(pairs, counts, rank, scales):
    ep, experts = counts.shape
    tokens, topk = pairs.shape[0] // ep, pairs.shape[1]
    block = host.routing_block(tokens, topk)
    stride = host.ragged_stride_bound(ep * tokens, topk, experts, 128)
    local = lax.dynamic_slice(pairs, (rank[0] * tokens, 0), (tokens, topk)).reshape(
        -1, block
    )
    gathered = host.GatheredPairs(
        expert_blocks=pairs.reshape(-1, block),
        local_blocks=local,
        block_hist=host.pair_block_hist(local, experts),
        rows_by_dest=counts,
        n_tokens=ep * tokens,
        topk=topk,
    )
    routing = push.build_routing_tables_sharded(
        rank[0],
        gathered,
        push.run_tables(counts),
        e_total=experts,
        ep=ep,
        t_local=tokens,
        block=block,
        tile_m=128,
        shard_stride=stride,
    )
    # Full plan outputs through the real consumer-facing helpers. Scale scatter
    # remains a distinct SC boundary; its table is returned here, not discarded.
    slab_rows = host.local_slab_rows(routing, rank[0], shard_stride=stride)
    blocks = host.shard_transport_tables_in_blocks(
        routing, rank[0], e_total=experts, ep=ep
    )
    rows = host.shard_push_tables_in_rows(routing, rank[0], e_total=experts, ep=ep)
    expert_rows, slab_base = host.shard_expert_slabs(
        routing, rank[0], e_total=experts, ep=ep
    )
    counts_out = host.shard_count_vector(
        routing, expert_rows, rank[0], e_total=experts, ep=ep
    )
    return (
        routing.pos,
        routing.mirror_pos,
        slab_rows,
        *blocks,
        *rows,
        expert_rows,
        slab_base,
        counts_out,
    )


def combine_region(arrivals, scales, pos, mirror, weights, valid):
    out = _combine_arrivals_fs(arrivals, scales, pos, mirror, weights, jnp.bfloat16)
    return jnp.where(valid[:, None], out, jnp.zeros_like(out))


def swa_region(low, high, table, starts, positions, minimum, values):
    metadata, bounds, records = prepare_swa_metadata(
        table, starts, positions, minimum, page_size=64, page_stride=64, row_offset=0
    )
    low, high = write_swa_pages((low, high), records, values)
    return low, high, metadata, bounds, records


def metadata_candidate(table, starts, positions, minimum):
    """Full metadata/record function without repeat's scatter lowering."""
    tokens, requests = positions.size, table.shape[0]
    first = take_along(positions, jnp.minimum(starts[:-1], tokens - 1), axis=0)
    count = starts[1:] - starts[:-1]
    left = jnp.where(count > 0, jnp.maximum(first - 127, 0), 0)
    right = jnp.where(count > 0, first + count, 0)
    columns = (tokens + 128 + 63) // 64 + 1
    logical = left[:, None] // 64 + jnp.arange(columns)[None, :]
    blocks = take_along(table, jnp.minimum(logical, table.shape[1] - 1), axis=1)
    visible = (
        (count[:, None] > 0)
        & (logical < table.shape[1])
        & (logical * 64 < right[:, None])
    )
    metadata = jnp.concatenate(
        (left[:, None] // 64 * 64, minimum[:, None], jnp.where(visible, blocks, 0)),
        axis=1,
    )
    request = jnp.minimum(
        jnp.sum(jnp.arange(tokens)[:, None] >= starts[None, 1:], axis=1), requests - 1
    )
    indices = jnp.broadcast_to(
        jnp.minimum(positions // 64, table.shape[1] - 1)[None, :], (requests, tokens)
    )
    per_request = take_along(table, indices, axis=1)
    current = jnp.sum(
        jnp.where(jnp.arange(requests)[:, None] == request[None, :], per_request, 0),
        axis=0,
    )
    records = jnp.where(
        jnp.arange(tokens) < starts[-1], current * 64 + positions % 64, -1
    )
    return metadata, jnp.stack((left, right), axis=1), records


def pack_bank(records, values, *, base, capacity, padded):
    """Stable permutation and full BF16 word preparation in one TC body."""
    n = records.size
    local = jnp.where(
        (records >= max(base, 64)) & (records < base + capacity),
        records - base,
        capacity,
    )
    idx = jnp.arange(n, dtype=jnp.int32)
    precedes = (local[None, :] < local[:, None]) | (
        (local[None, :] == local[:, None]) & (idx[None, :] < idx[:, None])
    )
    ranks = jnp.sum(precedes.astype(jnp.int32), axis=1)
    order = jnp.sum(jnp.where(ranks[None, :] == idx[:, None], idx[None, :], 0), axis=1)
    ids = take_along(local, order, axis=0)
    next_ids = jnp.concatenate((ids[1:], jnp.full((1,), capacity, jnp.int32)))
    pairs = ids // 2
    boundary = (ids < capacity) & jnp.concatenate(
        (jnp.ones(1, bool), pairs[1:] != pairs[:-1])
    )
    even = ids % 2 == 0
    has_odd = (~even) | (next_ids // 2 == pairs)
    v = take_along(values, jnp.broadcast_to(order[:, None], values.shape), axis=0)
    bits = lax.bitcast_convert_type(v, jnp.uint16).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
    low = jnp.where(even[:, None], bits, jnp.uint32(0))
    high = jnp.where(
        has_odd[:, None], jnp.where(even[:, None], next_bits, bits), jnp.uint32(0)
    )
    words = lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        has_odd, jnp.int32(-65536), jnp.int32(0)
    )
    destinations = jnp.pad(
        jnp.where(boundary, pairs, -1), (0, padded - n), constant_values=-1
    )
    masks = jnp.broadcast_to(jnp.pad(masks, (0, padded - n))[:, None], (padded, 512))
    return destinations, masks, jnp.pad(words, ((0, padded - n), (0, 0)))


def prepacked_writer():
    """Reuse the exact pinned SC writer AST, replacing only its TC preparation.

    Kept in this standalone harness to avoid changing the production interface.
    The writer's alias, partial-word handling and DMA schedule are unchanged.
    """
    from vllm_torchtpu.kernels.mla import paged_swa

    path = ROOT / "src/vllm_torchtpu/kernels/mla/paged_swa.py"
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "write_swa_pages"
    )
    node.name = "write_prepacked"
    node.decorator_list = []
    node.args.args.append(ast.arg(arg="prepared"))
    loop = next(n for n in node.body if isinstance(n, ast.For))
    owner = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "owner" for t in n.targets)
    )
    prefix = ast.parse(
        "capacity = bank.shape[0] * page_size\n"
        "destinations, masks, words = prepared[len(result)]"
    ).body
    loop.body = prefix + loop.body[owner:]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = dict(vars(paged_swa))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["write_prepacked"]


def metadata_matrix(table, starts, positions, minimum):
    """Keep explicit row/column dimensions through gather and reductions."""
    tokens, requests = positions.shape[1], table.shape[0]
    first = jnp.sum(
        jnp.where(
            jnp.arange(tokens)[None, :] == jnp.minimum(starts[:, :-1].T, tokens - 1),
            positions,
            0,
        ),
        axis=1,
        keepdims=True,
    )
    count = (starts[:, 1:] - starts[:, :-1]).T
    left = jnp.where(count > 0, jnp.maximum(first - 127, 0), 0)
    right = jnp.where(count > 0, first + count, 0)
    columns = (tokens + 128 + 63) // 64 + 1
    logical = left // 64 + jnp.arange(columns)[None, :]
    hit = (
        jnp.arange(table.shape[1])[None, None, :]
        == jnp.minimum(logical, table.shape[1] - 1)[:, :, None]
    )
    blocks = jnp.sum(jnp.where(hit, table[:, None, :], 0), axis=2)
    visible = (count > 0) & (logical < table.shape[1]) & (logical * 64 < right)
    metadata = jnp.concatenate(
        (left // 64 * 64, minimum, jnp.where(visible, blocks, 0)), axis=1
    )
    index = jnp.arange(tokens)[None, :]
    request = jnp.minimum(
        jnp.sum(index >= starts[:, 1:].T, axis=0, keepdims=True), requests - 1
    )
    page_hit = (
        logical[:, :, None]
        == jnp.minimum(positions // 64, table.shape[1] - 1)[:, None, :]
    )
    per_request = jnp.sum(jnp.where(page_hit, blocks[:, :, None], 0), axis=1)
    current = jnp.sum(
        jnp.where(jnp.arange(requests)[:, None] == request, per_request, 0),
        axis=0,
        keepdims=True,
    )
    records = jnp.where(index < starts[:, -1:], current * 64 + positions % 64, -1)
    return metadata, jnp.concatenate((left, right), axis=1), records


def pack_matrix(records, values, *, base, capacity, padded):
    n = values.shape[0]
    local = jnp.where(
        (max(base, 64) <= records.T) & (base + capacity > records.T),
        records.T - base,
        capacity,
    )
    idx = jnp.arange(n, dtype=jnp.int32)[:, None]
    ranks = jnp.sum(
        (local > local.T) | ((local == local.T) & (idx > idx.T)), axis=1, keepdims=True
    )
    permutation = idx == ranks.T
    ids = jnp.sum(jnp.where(permutation, local.T, 0), axis=1, keepdims=True)
    next_ids = jnp.concatenate((ids[1:], jnp.full((1, 1), capacity, jnp.int32)), axis=0)
    pairs = ids // 2
    boundary = (ids < capacity) & (
        jnp.concatenate(
            (jnp.ones((1, 1), jnp.int32), (pairs[1:] != pairs[:-1]).astype(jnp.int32)),
            axis=0,
        )
        != 0
    )
    even = ids % 2 == 0
    odd = (~even) | (next_ids // 2 == pairs)
    value_bits = lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.uint32)
    bits = jnp.dot(
        permutation.astype(jnp.float32),
        value_bits.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
    ).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)), axis=0)
    low = jnp.where(even, bits, jnp.uint32(0))
    high = jnp.where(odd, jnp.where(even, next_bits, bits), jnp.uint32(0))
    words = lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        odd, jnp.int32(-65536), jnp.int32(0)
    )
    return (
        jnp.pad(
            jnp.where(boundary, pairs, -1),
            ((0, padded - n), (0, 0)),
            constant_values=-1,
        ),
        jnp.broadcast_to(jnp.pad(masks, ((0, padded - n), (0, 0))), (padded, 512)),
        jnp.pad(words, ((0, padded - n), (0, 0))),
    )


def swa_candidate(xs, interpret=False, revision=1):
    low, high, *tail = xs
    info = pltpu.get_tpu_info().sparse_core
    lanes_cores = info.num_lanes * info.num_cores * info.num_subcores
    padded = (tail[-1].shape[0] + lanes_cores - 1) // lanes_cores * lanes_cores
    cap0, cap1 = low.shape[0] * 64, high.shape[0] * 64

    def prepare(table, starts, positions, minimum, values):
        meta_fn = metadata_matrix if revision >= 2 else metadata_candidate
        pack_fn = pack_matrix if revision >= 2 else pack_bank
        metadata, bounds, records = meta_fn(table, starts, positions, minimum)
        p0 = pack_fn(records, values, base=0, capacity=cap0, padded=padded)
        p1 = pack_fn(records, values, base=cap0, capacity=cap1, padded=padded)
        return metadata, bounds, records, p0, p1

    def matrix_args(table, starts, positions, minimum, values):
        return table, starts[None, :], positions[None, :], minimum[:, None], values

    if revision >= 2:
        tail = matrix_args(*tail)
    fused = fuse_function(
        prepare, tail, name="dsv41_swa_metadata_sort_pack", interpret=interpret
    )
    publish = prepacked_writer()

    def call(low, high, table, starts, positions, minimum, values):
        inputs = (table, starts, positions, minimum, values)
        if revision >= 2:
            inputs = matrix_args(*inputs)
        metadata, bounds, records, p0, p1 = fused(*inputs)
        if revision >= 2:
            records = records.reshape(-1)
            p0 = (p0[0].reshape(-1), *p0[1:])
            p1 = (p1[0].reshape(-1), *p1[1:])
        low, high = publish((low, high), records, values, (p0, p1))
        return low, high, metadata, bounds, records

    return call


def combine_candidate(arrivals, scales, pos, mirror, weights, valid):
    rows = arrivals.reshape(arrivals.shape[0], -1)
    pos = jnp.minimum(pos, rows.shape[0] - 1)
    mirror = jnp.minimum(mirror, scales.size - 1)
    row_scales = take_along(scales.reshape(-1), mirror.reshape(-1), axis=0).reshape(
        mirror.shape
    )
    coef = row_scales * weights.astype(jnp.float32)
    acc = None
    for k in range(pos.shape[1]):
        indices = jnp.broadcast_to(pos[:, k, None], (pos.shape[0], rows.shape[1]))
        term = take_along(rows, indices, axis=0).astype(jnp.float32) * coef[:, k, None]
        acc = term if acc is None else acc + term
    return jnp.where(valid[:, None], acc.astype(jnp.bfloat16), jnp.bfloat16(0))


def local_candidate(pairs, counts, q):
    tokens, blocks, lanes = q.shape
    topk = pairs.size // tokens
    end = prefix_sum(counts)
    start = end - counts
    pid = jnp.where(pairs < counts.size, pairs, counts.size)
    find = pallas_host_copy(push._sorted_tokens_by_count)
    order = find(pid, start, end, e_total=counts.size, t_local=tokens, topk=topk)
    indices = jnp.broadcast_to(order[:, None], (pairs.size, blocks * lanes))
    values = take_along(q.reshape(tokens, blocks * lanes), indices, axis=0)
    return values.reshape(pairs.size, blocks, lanes), start


def local_matrix(pairs, counts, q):
    """Same count-based stable permutation with pair axes tiled on entry."""
    topk, tokens = pairs.shape
    experts = counts.size
    bins = jnp.arange(experts, dtype=jnp.int32)
    selection = jnp.zeros((tokens, experts), bool)
    for k in range(topk):
        selection = selection | (pairs[k, :, None] == bins[None, :])
    cumulative = prefix_sum(selection.astype(jnp.int32), axis=0)
    end = prefix_sum(counts)
    start = end - counts
    jj = jnp.arange(tokens * topk, dtype=jnp.int32)
    expert = jnp.minimum(jnp.sum(end[None, :] <= jj[:, None], axis=1), experts - 1)
    onehot = expert[:, None] == bins[None, :]
    offset = jj - jnp.sum(jnp.where(onehot, start[None, :], 0), axis=1)
    col = jnp.dot(
        onehot.astype(jnp.float32),
        cumulative.T.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
    ).astype(jnp.int32)
    order = jnp.minimum(jnp.sum(col <= offset[:, None], axis=1), tokens - 1)
    permutation = (order[:, None] == jnp.arange(tokens)[None, :]).astype(jnp.bfloat16)
    # Select raw FP8 bytes: 0 * NaN/Inf must not contaminate unselected rows.
    qbits = lax.bitcast_convert_type(q, jnp.uint8).astype(jnp.bfloat16)
    values = jnp.dot(permutation, qbits, preferred_element_type=jnp.float32).astype(
        jnp.uint8
    )
    values = lax.bitcast_convert_type(values, q.dtype)
    return values, start


def local_matrix_call(xs, interpret=False):
    p, c, q = xs
    tokens, groups, lanes = q.shape

    def adapt(p, c, q):
        return p.reshape(tokens, -1).T, c, q.reshape(tokens, groups * lanes)

    fused = fuse_function(
        local_matrix, adapt(*xs), name="dsv41_moe_local_matrix", interpret=interpret
    )

    def call(p, c, q):
        values, start = fused(*adapt(p, c, q))
        return values.reshape(-1, groups, lanes), start

    return call


def host_namespace(module):
    return types.SimpleNamespace(
        **{
            key: pallas_host_copy(value)
            if isinstance(value, types.FunctionType)
            else value
            for key, value in vars(module).items()
        }
    )


def fuse_function(fn, args, *, name, interpret=False):
    """Real Pallas boundary, with the same externally visible array outputs."""
    out_shape = jax.eval_shape(fn, *args)
    leaves, tree = jax.tree.flatten(out_shape)

    def kernel(*refs):
        outputs = fn(*(r[...] for r in refs[: len(args)]))
        for ref, value in zip(refs[len(args) :], jax.tree.leaves(outputs)):
            ref[...] = value

    call = pl.pallas_call(
        kernel,
        out_shape=tuple(leaves),
        name=name,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=60 * 1024 * 1024),
    )
    return lambda *xs: jax.tree.unflatten(tree, call(*xs))


def make_moe(seed=0, skew=False):
    rng = np.random.default_rng(seed)
    ep, tokens, topk, experts = 16, 288, 6, 384
    hot = np.arange(24 if skew else experts)
    pairs = np.stack(
        [rng.choice(hot, topk, replace=False) for _ in range(ep * tokens)]
    ).astype(np.int32)
    pairs[::31] = experts
    counts = np.stack(
        [
            np.bincount(p.reshape(-1), minlength=experts + 1)[:experts]
            for p in pairs.reshape(ep, tokens, topk)
        ]
    ).astype(np.int32)
    rank = np.array([0], np.int32)
    seg_start = np.cumsum(counts[0], dtype=np.int32) - counts[0]
    q = (rng.standard_normal((tokens, 40, 128)) * 2).astype(ml_dtypes.float8_e4m3fn)
    # Real arrival extent varies with destination traffic. Include padding and
    # out-of-range clamped positions independently of the plan fixture.
    arrivals = (rng.standard_normal((2112, 40, 128)) * 2).astype(
        ml_dtypes.float8_e4m3fn
    )
    scales = rng.uniform(0.01, 0.1, (786, 128)).astype(np.float32)
    pos = rng.integers(0, 2112, (tokens, topk), dtype=np.int32)
    mirror = rng.integers(0, scales.size, (tokens, topk), dtype=np.int32)
    pos[-1] = 3000
    mirror[-1] = scales.size + 20
    weights = rng.uniform(0, 1, (tokens, topk)).astype(np.float32)
    weights /= weights.sum(axis=1, keepdims=True)
    valid = np.ones(tokens, bool)
    valid[::31] = False
    return {
        "dispatch": (counts, seg_start, rank),
        "route": (pairs, counts, rank, np.ones((ep * tokens,), np.float32)),
        "local": (pairs[:tokens].reshape(-1), counts[0], q),
        "combine": (arrivals, scales, pos, mirror, weights, valid),
    }


def make_swa(seed=0):
    rng = np.random.default_rng(seed)
    batch, query, page_size, pages = 48, 6, 64, 1024
    first = 98304 + np.arange(batch, dtype=np.int32) * 3
    positions = (first[:, None] + np.arange(query, dtype=np.int32)).reshape(-1)
    width = int(positions.max()) // page_size + 1
    table = np.zeros((batch, width), np.int32)
    free = iter(rng.permutation(np.arange(1, 2 * pages, dtype=np.int32)))
    for b in range(batch):
        for c in range(
            (first[b] - 127) // page_size, (first[b] + query - 1) // page_size + 1
        ):
            table[b, c] = next(free)
    pools = [
        np.full((pages, page_size, 512), 0.25, dtype=ml_dtypes.bfloat16)
        for _ in range(2)
    ]
    values = rng.standard_normal((batch * query, 512)).astype(ml_dtypes.bfloat16)
    return (
        *pools,
        table,
        np.arange(batch + 1, dtype=np.int32) * query,
        positions,
        np.maximum(first - 80, 0),
        values,
    )


def compare(want, got):
    maximum = 0.0
    for a, b in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
        a, b = np.asarray(a), np.asarray(b)
        if np.issubdtype(a.dtype, np.integer) or a.dtype == bool:
            np.testing.assert_array_equal(a, b)
        else:
            maximum = max(
                maximum,
                float(np.nanmax(np.abs(a.astype(np.float32) - b.astype(np.float32)))),
            )
            if a.dtype.itemsize in (1, 2):
                # NumPy does not recognize ml_dtypes FP8 NaNs as floating NaNs.
                # Raw storage also checks signed zero and NaN payloads.
                storage = np.uint8 if a.dtype.itemsize == 1 else np.uint16
                np.testing.assert_array_equal(a.view(storage), b.view(storage))
            else:
                np.testing.assert_array_equal(a, b)
    return {"exact": True, "max_abs": maximum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--regions", nargs="+", default=["dispatch", "route", "local", "combine", "swa"]
    )
    parser.add_argument("--candidate", action="store_true")
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--revision", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument(
        "--case",
        choices=["uniform", "skew", "empty", "edge", "bits"],
        default="uniform",
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fixtures = make_moe(skew=args.case == "skew")
    counts = fixtures["dispatch"][0]
    if args.case == "empty":
        counts = np.zeros_like(counts)
        fixtures["route"] = (
            np.full_like(fixtures["route"][0], 384),
            counts,
            *fixtures["route"][2:],
        )
        fixtures["combine"] = (
            *fixtures["combine"][:-1],
            np.zeros_like(fixtures["combine"][-1]),
        )
    rank = np.array([args.rank], np.int32)
    fixtures["dispatch"] = (
        counts,
        np.cumsum(counts[args.rank], dtype=np.int32) - counts[args.rank],
        rank,
    )
    fixtures["route"] = (fixtures["route"][0], counts, rank, fixtures["route"][3])
    fixtures["local"] = (
        fixtures["route"][0][args.rank * 288 : (args.rank + 1) * 288].reshape(-1),
        counts[args.rank],
        fixtures["local"][2],
    )
    if args.case == "bits":
        p, c, q = fixtures["local"]
        q = (
            np.resize(np.arange(256, dtype=np.uint8), q.size)
            .reshape(q.shape)
            .view(ml_dtypes.float8_e4m3fn)
        )
        fixtures["local"] = (p, c, q)
    if "swa" in args.regions:
        fixtures["swa"] = make_swa()
    if args.case == "edge" and "swa" in fixtures:
        import region_references

        fixtures["swa"] = region_references.swa_edge_fixture(fixtures["swa"])
    fns = {
        "dispatch": dispatch_region,
        "route": route_region,
        "local": lambda p, c, q: push.local_sort_activations(
            p, c, q, e_total=384, topk=6
        ),
        "combine": combine_region,
        "swa": swa_region,
    }
    results = {
        "runtime": {
            p: importlib.metadata.version(p) for p in ["jax", "jaxlib", "numpy"]
        },
        "devices": [str(d) for d in jax.devices()],
        "case": args.case,
        "rank": args.rank,
        "regions": {},
    }
    for region in args.regions:
        print("START", region, flush=True)
        xs = jax.device_put(fixtures[region], jax.devices()[0])
        f = fns[region]
        variants = {"baseline": f}
        if args.candidate and region == "dispatch":
            # Clone function globals explicitly so no baseline helper changes.
            cloned = types.FunctionType(
                dispatch_region.__code__,
                {
                    **globals(),
                    "push": types.SimpleNamespace(
                        run_tables=pallas_host_copy(push.run_tables),
                        push_dispatch_segments=pallas_host_copy(
                            push.push_dispatch_segments
                        ),
                    ),
                },
            )
            variants["fused"] = fuse_function(
                cloned, xs, name="dsv41_moe_dispatch_region", interpret=args.interpret
            )
            if args.revision >= 2:
                variants["fused"] = fuse_function(
                    dispatch_flat,
                    xs,
                    name="dsv41_moe_dispatch_flat",
                    interpret=args.interpret,
                )
            if args.revision >= 3:
                variants["fused"] = dispatch_layout_call(xs, interpret=args.interpret)
        if args.candidate and region == "combine":
            variants["fused"] = fuse_function(
                combine_candidate,
                xs,
                name="dsv41_moe_combine_region",
                interpret=args.interpret,
            )
            if args.revision >= 2:
                variants["fused"] = (
                    combine_dma(xs)
                    if not args.interpret
                    else combine_tiled(xs, interpret=True)
                )
            if args.revision >= 3 and not args.interpret:
                variants["fused"] = combine_complete(xs)
        if args.candidate and region == "local":
            variants["fused"] = fuse_function(
                local_candidate,
                xs,
                name="dsv41_moe_local_region",
                interpret=args.interpret,
            )
            if args.revision >= 2:
                variants["fused"] = local_matrix_call(xs, interpret=args.interpret)
        if args.candidate and region == "route":
            cloned = types.FunctionType(
                route_region.__code__,
                {
                    **globals(),
                    "host": host_namespace(host),
                    "push": host_namespace(push),
                },
            )
            variants["fused"] = fuse_function(
                cloned, xs, name="dsv41_moe_route_region", interpret=args.interpret
            )
        if args.candidate and region == "swa":
            variants["fused"] = swa_candidate(
                xs, interpret=args.interpret, revision=args.revision
            )
        r = {}
        executables = {}
        states = {}
        reference = None
        if args.verify_reference and region != "route":
            import region_references

            reference = getattr(region_references, region)(*fixtures[region])
        for name, fn in variants.items():
            try:
                current = jax.device_put(fixtures[region], jax.devices()[0])
                compiled = (
                    jax.jit(fn, donate_argnums=(0, 1) if region == "swa" else ())
                    .lower(*current)
                    .compile()
                )
                executables[name] = compiled
                (args.out / f"{region}-{name}.hlo").write_text(compiled.as_text())
                result = jax.block_until_ready(compiled(*current))
                if region == "swa":
                    current = (*result[:2], *current[2:])
                if name == "baseline":
                    expected = jax.tree.map(np.asarray, result)
                else:
                    r[name] = {"correctness": compare(expected, result)}
                if reference is not None:
                    r.setdefault(name, {})["numpy_reference"] = compare(
                        reference, result
                    )
                for _ in range(5):
                    result = jax.block_until_ready(compiled(*current))
                    if region == "swa":
                        current = (*result[:2], *current[2:])
                states[name] = current
                r.setdefault(name, {})["host_us"] = []
            except Exception as exc:
                import traceback

                r[name] = {"error": str(exc), "traceback": traceback.format_exc()}
                print(r[name]["traceback"], flush=True)
        for i in range(args.repeat):
            for name in list(executables) if i % 2 == 0 else list(executables)[::-1]:
                if "error" in r[name]:
                    continue
                start = time.perf_counter_ns()
                result = jax.block_until_ready(executables[name](*states[name]))
                r[name]["host_us"].append((time.perf_counter_ns() - start) / 1000)
                if region == "swa":
                    states[name] = (*result[:2], *states[name][2:])
        if args.profile:
            with jax.profiler.trace(str(args.out / f"profile-{region}")):
                for i in range(12):
                    for name, fn in executables.items():
                        if "error" not in r[name]:
                            result = jax.block_until_ready(fn(*states[name]))
                            if region == "swa":
                                states[name] = (*result[:2], *states[name][2:])
        for name, v in r.items():
            if "host_us" in v:
                v["host_median_us"] = float(np.median(v["host_us"]))
        results["regions"][region] = r
        (args.out / "results.json").write_text(json.dumps(results, indent=2))
        print(
            "DONE",
            region,
            {
                n: {k: v for k, v in x.items() if k != "host_us" and k != "traceback"}
                for n, x in r.items()
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
