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

from .abi import KV_DMA_FIELDS, QUERY_DMA_FIELDS, RPA_WAIT_ROW_BYTES


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


def _kernel(meta_ref, *out_refs, cfgs, names):
    steps, lanes = cfgs.max_steps_ub, cfgs.batch_size

    def lane(row):
        return meta_ref[row, :, :lanes]

    def gather(row, indices):
        return jnp.take_along_axis(
            meta_ref[row, :, :],
            jnp.broadcast_to(indices[None, :], (steps, indices.size)),
            axis=1,
        )

    col = jnp.arange(lanes * cfgs.bkv_p_cache * len(KV_DMA_FIELDS))
    idx, page, field = (
        col // (cfgs.bkv_p_cache * len(KV_DMA_FIELDS)),
        (col // len(KV_DMA_FIELDS)) % cfgs.bkv_p_cache,
        col % len(KV_DMA_FIELDS),
    )
    seq, k, length, live = (
        gather(row, idx)
        for row in (
            CompactRow.SEQUENCE,
            CompactRow.KV_TILE,
            CompactRow.LENGTH,
            CompactRow.LIVE,
        )
    )
    src = jnp.minimum(
        seq * cfgs.serve.pages_per_seq + k * cfgs.bkv_p + page[None, :],
        cfgs.serve.num_page_indices - 1,
    )
    dst = jnp.broadcast_to(page[None, :] * cfgs.serve.page_size, src.shape)
    size = jnp.clip(length - k * cfgs.bkv_sz - dst, 0, cfgs.serve.page_size)
    dma = jnp.where(
        live != 0,
        jnp.where(field[None, :] == 0, src, jnp.where(field[None, :] == 1, dst, size)),
        0,
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
        dma_kv_cache=dma,
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
    cu_q_lens: jax.Array, lengths: jax.Array, cfgs: configs.RpaConfigs
) -> schedule.RpaSchedule:
    shape = schedule.BaseMetadataComputer.get_rpa_schedule(cfgs)
    steps, lanes = cfgs.max_steps_ub, cfgs.batch_size
    assert steps * lanes >= len(lengths) * pl.cdiv(
        cfgs.serve.pages_per_seq * cfgs.serve.page_size, cfgs.bkv_sz
    )
    tiles = (lengths + cfgs.bkv_sz - 1) // cfgs.bkv_sz
    ends = jnp.cumsum(tiles)
    slots = jnp.arange(steps * lanes).reshape(steps, lanes)
    seq = jnp.minimum(jnp.sum(slots[..., None] >= ends, axis=-1), len(lengths) - 1)
    begins = jnp.concatenate((jnp.zeros(1, jnp.int32), ends[:-1]))
    k = slots - begins[seq]
    live = slots < ends[-1]
    last = live & (k == tiles[seq] - 1)
    meta = jnp.stack(
        (
            seq,
            k,
            lengths[seq],
            live.astype(jnp.int32),
            jnp.where(live, cu_q_lens[seq], 0),
            last.astype(jnp.int32),
        )
    )
    meta = jnp.pad(meta, ((0, 0), (0, 0), (0, pltpu.get_tpu_info().num_lanes - lanes)))
    names = field_names(shape)
    out_shape = [
        jax.ShapeDtypeStruct(
            (steps, getattr(shape, name).data.size // steps), jnp.int32
        )
        for name in names
    ]
    outputs = pl.pallas_call(
        functools.partial(_kernel, cfgs=cfgs, names=names),
        out_shape=out_shape,
        name="qsa_decode_schedule",
    )(meta)
    result = {
        name: dataclasses.replace(getattr(shape, name), data=value.reshape(-1))
        for name, value in zip(names, outputs)
    }
    return dataclasses.replace(
        shape,
        **result,
        actual_steps=((ends[-1:] + lanes - 1) // lanes).astype(jnp.int32),
    )
