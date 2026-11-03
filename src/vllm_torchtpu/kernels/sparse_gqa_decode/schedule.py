# Copyright 2026 Google LLC
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

"""Compact decode task assignment and fused vector DMA-schedule expansion."""

import dataclasses
import enum
import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.batched_rpa import configs, schedule

from .abi import QUERY_DMA_FIELDS, RPA_WAIT_ROW_BYTES


class CompactRow(enum.IntEnum):
    SEQUENCE = 0
    KV_TILE = 1
    LENGTH = 2
    LIVE = 3
    QUERY_ROW = 4
    LAST_TILE = 5


POPULATED_FIELDS = frozenset(
    {
        "s_idx",
        "k_idx",
        "is_last_k",
        "dma_q",
        "dma_kv_cache",
        "total_wait_kv_in",
        "total_wait_q_in",
        "total_wait_o_out",
        "total_wait_lse_out",
    }
)
UNUSED_FIELDS = frozenset({"q_idx", "do_writeback", "dma_kv_new", "total_wait_kv_out"})


def field_names(shape: schedule.RpaSchedule) -> tuple[str, ...]:
    """Fail closed if the shared protocol gains an unhandled field."""
    names = tuple(f.name for f in dataclasses.fields(shape))
    if set(names) != POPULATED_FIELDS | UNUSED_FIELDS | {"actual_steps", "cfgs"}:
        raise ValueError("Unhandled RpaSchedule fields in decode specialization")
    return tuple(name for name in names if name in POPULATED_FIELDS | UNUSED_FIELDS)


def _kernel(cu_ref, lengths_ref, *out_refs, cfgs, names, steps):
    lanes = cfgs.batch_size
    # Keep steps on vector lanes while assigning tasks, then expand directly
    # into the existing RPA protocol without an intermediate HBM owner.
    slots = jnp.arange(steps)[None, :] * lanes + jnp.arange(lanes)[:, None]
    seq = jnp.zeros((lanes, steps), jnp.int32)
    tile = jnp.zeros_like(seq)
    length = jnp.zeros_like(seq)
    query = jnp.zeros_like(seq)
    last = jnp.zeros_like(seq)
    begin = jnp.int32(0)
    for row in range(lengths_ref.shape[0]):
        count = (lengths_ref[row] + cfgs.bkv_sz - 1) // cfgs.bkv_sz
        end = begin + count
        mask = (slots >= begin) & (slots < end)
        seq = jnp.where(mask, row, seq)
        tile = jnp.where(mask, slots - begin, tile)
        length = jnp.where(mask, lengths_ref[row], length)
        query = jnp.where(mask, cu_ref[row], query)
        last = jnp.where(mask, (slots == end - 1).astype(jnp.int32), last)
        begin = end
    live = (slots < begin).astype(jnp.int32)
    meta = tuple(
        jnp.pad(row.T, ((0, 0), (0, pltpu.get_tpu_info().num_lanes - lanes)))
        for row in (seq, tile, length, live, query, last)
    )
    out_refs[-1][...] = jnp.full((1,), (begin + lanes - 1) // lanes, jnp.int32)
    out_refs = out_refs[:-1]

    def lane(row):
        return meta[row][:, :lanes]

    def gather(row, indices):
        return jnp.take_along_axis(
            meta[row],
            jnp.broadcast_to(indices[None, :], (steps, indices.size)),
            axis=1,
        )

    qcol = jnp.arange(lanes * len(QUERY_DMA_FIELDS))
    dq = jnp.where(
        qcol[None, :] % len(QUERY_DMA_FIELDS) == 0,
        gather(CompactRow.QUERY_ROW, qcol // len(QUERY_DMA_FIELDS)),
        gather(CompactRow.LIVE, qcol // len(QUERY_DMA_FIELDS)),
    )
    live = lane(CompactRow.LIVE)
    count = jnp.sum(live, axis=1, keepdims=True)
    last = lane(CompactRow.LAST_TILE)
    outs = jnp.sum(last, axis=1, keepdims=True)
    kv_count = jnp.sum(
        jnp.where(
            live != 0,
            jnp.clip(
                lane(CompactRow.LENGTH) - lane(CompactRow.KV_TILE) * cfgs.bkv_sz,
                0,
                cfgs.bkv_sz,
            ),
            0,
        ),
        axis=1,
        keepdims=True,
    )
    values = dict(
        s_idx=jnp.where(live != 0, lane(CompactRow.SEQUENCE), -1),
        k_idx=jnp.where(live != 0, lane(CompactRow.KV_TILE), 0),
        is_last_k=last,
        dma_q=dq,
        total_wait_kv_in=kv_count * cfgs.kv_bytes_per_token // RPA_WAIT_ROW_BYTES,
        total_wait_q_in=count * cfgs.q_bytes_per_token // RPA_WAIT_ROW_BYTES,
        total_wait_o_out=outs * cfgs.o_bytes_per_token // RPA_WAIT_ROW_BYTES,
        total_wait_lse_out=outs
        * cfgs.model.num_kv_heads
        * cfgs.aligned_num_q_heads_per_kv_head,
    )
    for name, ref in zip(names, out_refs):
        if name in values:
            ref[...] = values[name]
        elif name in UNUSED_FIELDS:
            ref[...] = jnp.zeros(ref.shape, jnp.int32)
        else:
            raise ValueError(f"Unhandled decode schedule field: {name}")


def generate(
    cu_q_lens: jax.Array,
    lengths: jax.Array,
    pages: jax.Array,
    cfgs: configs.RpaConfigs,
) -> schedule.RpaSchedule:
    shape = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs)
    capacity, lanes = cfgs.max_steps_ub, cfgs.batch_size
    max_tasks = len(lengths) * pl.cdiv(
        cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz
    )
    assert capacity * lanes >= max_tasks
    # The shared ABI reserves SMEM capacity, which can greatly exceed this
    # query bucket's worst-case work. Only expand the reachable schedule rows.
    vector_rows = pltpu.get_tpu_info().num_sublanes
    steps = min(capacity, pl.cdiv(pl.cdiv(max_tasks, lanes), vector_rows) * vector_rows)
    fields = (
        ("s_idx", shape.s_idx),
        ("q_idx", shape.q_idx),
        ("k_idx", shape.k_idx),
        ("is_last_k", shape.is_last_k),
        ("do_writeback", shape.do_writeback),
        ("dma_q", shape.dma_q),
        ("dma_kv_cache", shape.dma_kv_cache),
        ("dma_kv_new", shape.dma_kv_new),
        ("total_wait_kv_in", shape.total_wait_kv_in),
        ("total_wait_kv_out", shape.total_wait_kv_out),
        ("total_wait_q_in", shape.total_wait_q_in),
        ("total_wait_o_out", shape.total_wait_o_out),
        ("total_wait_lse_out", shape.total_wait_lse_out),
    )
    names = tuple(name for name, _ in fields)
    if names != field_names(shape):
        raise ValueError("RpaSchedule field order changed")
    shapes = {
        name: jax.ShapeDtypeStruct((steps, ref.data.size // capacity), jnp.int32)
        for name, ref in fields
    }
    # Physical page addresses are expanded below; the common fields can be
    # shared across layers, with task assignment fused into the same kernel.
    shared_names = tuple(name for name in names if name != "dma_kv_cache")
    shared_outputs = pl.pallas_call(
        functools.partial(_kernel, cfgs=cfgs, names=shared_names, steps=steps),
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=2),
        out_shape=[
            *(shapes[name] for name in shared_names),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ],
        name="qsa_decode_schedule",
    )(cu_q_lens, lengths)
    outputs = dict(zip(shared_names, shared_outputs[:-1]))
    actual_steps = shared_outputs[-1]
    owners = outputs["s_idx"].reshape(steps, lanes)
    seq = jnp.maximum(owners, 0)
    k = outputs["k_idx"].reshape(steps, lanes)
    live = owners >= 0
    # Select whole contiguous page windows before expanding their DMA fields.
    # Scalar gathers from the expanded schedule create one HBM lookup per page
    # and a strided scatter, even though every task reads adjacent page IDs.
    page_count = cfgs.serve.pages_per_seq
    page_tiles = pl.cdiv(page_count, cfgs.bkv_p_cache)
    window_words = page_tiles * cfgs.bkv_p_cache
    flat_pages = jnp.pad(pages.reshape(-1), (0, window_words - page_count), mode="edge")
    windows = jnp.stack(
        [
            flat_pages[row * page_count : row * page_count + window_words]
            for row in range(len(lengths))
        ]
    ).reshape(-1, cfgs.bkv_p_cache)
    physical = windows[seq * page_tiles + jnp.clip(k, 0, page_tiles - 1)]
    dma_shape = shapes["dma_kv_cache"]
    # Let XLA fuse physical DMA fields with the contiguous page-window gather.
    # A separate Pallas call forces these intermediates into HBM for each layer.
    physical = physical.reshape(steps, lanes, cfgs.bkv_p_cache)
    dst = jnp.broadcast_to(
        jnp.arange(cfgs.bkv_p_cache, dtype=jnp.int32) * cfgs.serve.page_size,
        physical.shape,
    )
    size = jnp.clip(
        lengths[seq, None] - k[..., None] * cfgs.bkv_sz - dst,
        0,
        cfgs.serve.page_size,
    )
    outputs["dma_kv_cache"] = jnp.where(
        live[..., None, None], jnp.stack((physical, dst, size), axis=-1), 0
    ).reshape(dma_shape.shape)
    return jax.tree.unflatten(
        jax.tree.structure(shape),
        [
            *(
                jnp.pad(
                    outputs[name],
                    ((0, capacity - steps), (0, 0)),
                    constant_values=-1 if name == "s_idx" else 0,
                ).reshape(-1)
                for name in names
            ),
            actual_steps,
        ],
    )
