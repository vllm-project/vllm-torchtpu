# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Request unions and reusable index keys for sparse-layer decode.

The caller owns cache lifetime and falls back when any union overflows. IDs are
ascending logical blocks; membership is applied before selection for every Q.

Keys are int32 [groups, capacity_records // 2, 128]: low/high halves hold
even/odd records, each in even64-then-odd64 BF16 channel order.
"""

import functools
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41 import indexer
from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes
from vllm_torchtpu.kernels.deepseek_v41.index_selection.bf16 import common, decode
from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    streaming_topk,
)


class UnionMetadata(NamedTuple):
    union_block_ids: jax.Array
    membership: jax.Array
    union_block_count: jax.Array
    overflow: jax.Array


def request_query_count(rows: int, requests: int) -> int | None:
    """Uniform request groups whose query memberships fit in one int32 word."""
    if requests < 1:
        return None
    queries, tail = divmod(rows, requests)
    return queries if not tail and 0 < queries <= jnp.iinfo(jnp.int32).bits else None


def _sorted_union_metadata(
    candidate_block_ids: jax.Array, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    requests, queries, _ = candidate_block_ids.shape
    valid = (candidate_block_ids >= 0) & (candidate_block_ids < logical_blocks)
    # Include query ownership in the sort key so duplicate IDs within one
    # query contribute its membership bit only once.
    encoded = jnp.where(
        valid,
        candidate_block_ids * queries + jnp.arange(queries)[None, :, None],
        logical_blocks * queries,
    )
    encoded = jnp.sort(encoded.reshape(requests, -1), axis=1)
    blocks = encoded // queries
    valid = blocks < logical_blocks
    previous = jnp.pad(encoded[:, :-1], ((0, 0), (1, 0)), constant_values=-1)
    first = valid & (blocks != previous // queries)
    count = first.sum(1).astype(jnp.int32)
    slots = jnp.where(valid, jnp.cumsum(first, axis=1) - 1, capacity_blocks)
    rows = jnp.arange(requests)[:, None]
    ids = jnp.full((requests, capacity_blocks), -1, jnp.int32)
    ids = ids.at[rows, slots].max(jnp.where(valid, blocks, -1), mode="drop")
    bits = jnp.where(valid & (encoded != previous), 1 << (encoded % queries), 0)
    membership = jnp.zeros_like(ids).at[rows, slots].add(bits, mode="drop")
    return UnionMetadata(ids, membership, count, count > capacity_blocks)


def union_metadata(
    candidate_block_ids: jax.Array, *, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    """Compact request-local logical block IDs and retain full union counts."""
    requests, queries, _ = candidate_block_ids.shape
    if (
        not 1 <= queries <= jnp.iinfo(jnp.int32).bits
        or min(logical_blocks, capacity_blocks) < 1
    ):
        raise ValueError("Query memberships must fit int32 and capacities be positive")
    if logical_blocks > jnp.iinfo(jnp.int32).max // queries:
        raise ValueError("Encoded block/query IDs must fit int32")
    # For longer contexts, bound metadata work by candidate count, not ID space.
    if logical_blocks > tuned_block_sizes.CANDIDATE_DENSE_UNION_MAX_BLOCKS:
        return _sorted_union_metadata(
            candidate_block_ids, logical_blocks, capacity_blocks
        )
    if (
        jax.default_backend() == "tpu"
        and candidate_block_ids.dtype == jnp.int32
        and requests > 0
        and candidate_block_ids.shape[-1] > 0
    ):
        from .union import metadata_layout, sparse_union_metadata

        tpu_info = pltpu.get_tpu_info()
        info = tpu_info.sparse_core
        if (
            info is None
            or tpu_info.chip_version != pltpu.ChipVersion.TPU_7X
            or candidate_block_ids.shape[-1] % tpu_info.num_lanes
        ):
            return _dense_union_metadata(
                candidate_block_ids, logical_blocks, capacity_blocks
            )
        layout = metadata_layout(
            candidate_block_ids.shape, logical_blocks, capacity_blocks
        )
        if layout.scratch_bytes <= info.vmem_capacity_bytes:
            return UnionMetadata(
                *sparse_union_metadata(
                    candidate_block_ids, logical_blocks, capacity_blocks
                )
            )
    return _dense_union_metadata(candidate_block_ids, logical_blocks, capacity_blocks)


def _dense_union_metadata(
    candidate_block_ids: jax.Array, logical_blocks: int, capacity_blocks: int
) -> UnionMetadata:
    requests, queries, _ = candidate_block_ids.shape
    valid = (candidate_block_ids >= 0) & (candidate_block_ids < logical_blocks)
    safe = jnp.where(valid, candidate_block_ids, logical_blocks)
    presence = jnp.zeros((requests, queries, logical_blocks + 1), jnp.int32)
    presence = presence.at[
        jnp.arange(requests)[:, None, None], jnp.arange(queries)[None, :, None], safe
    ].max(valid.astype(jnp.int32))
    bits = (presence[:, :, :-1] << jnp.arange(queries)[None, :, None]).sum(1)
    count = (bits != 0).sum(1).astype(jnp.int32)
    ids = jax.vmap(
        lambda row: jnp.nonzero(row, size=capacity_blocks, fill_value=-1)[0]
    )(bits)
    membership = jnp.take_along_axis(bits, jnp.maximum(ids, 0), axis=1)
    return UnionMetadata(
        ids, jnp.where(ids >= 0, membership, 0), count, count > capacity_blocks
    )


def union_record_starts(
    metadata: UnionMetadata,
    records: jax.Array,
    *,
    block_size: int,
    pool_records: int,
) -> tuple[jax.Array, jax.Array]:
    """Resolve logical blocks to aligned physical runs; invalidate bad pages."""
    logical = jnp.maximum(metadata.union_block_ids, 0) * block_size
    starts = jnp.take_along_axis(records, jnp.minimum(logical, records.shape[1] - 1), 1)
    valid = (metadata.union_block_ids >= 0) & (logical < records.shape[1])
    valid &= (starts >= 0) & (starts + block_size <= pool_records)
    # D128 BF16 packs two records into one pool word row. Logical blocks
    # need not start at a multiple of block_size in the physical allocation.
    valid &= starts % SLOTS == 0
    return jnp.where(valid, starts, -1), jnp.where(valid, metadata.membership, 0)


def _expand_metadata(values, repeats, group_width):
    if repeats == 1:
        return values
    # Lane-local gather avoids repeat's unsupported narrow-minor reshape.
    width = math.gcd(values.shape[0], group_width)
    source = values.reshape(-1, width)
    indices = jnp.broadcast_to(
        (jnp.arange(width * repeats) // repeats)[None, :, None],
        (source.shape[0], width * repeats, 1),
    )
    return jax.lax.gather(
        source,
        indices,
        jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(1,),
            start_index_map=(1,),
            operand_batching_dims=(0,),
            start_indices_batching_dims=(0,),
        ),
        slice_sizes=(1, 1),
        mode="promise_in_bounds",
    ).reshape(-1)


def _score_kernel(
    start,
    q,
    weights,
    keys,
    ids,
    membership,
    lengths,
    offsets,
    out,
    key_buffers,
    input_sems,
    *,
    queries,
    request_tile,
    compute_tile,
    head_stripe,
):
    dim = q.shape[-1]
    heads = math.prod(q.shape[1:-1]) // queries
    width = out.shape[-1]
    repeat = width // ids.shape[-1]
    group = out.shape[0] // queries
    first = pl.program_id(1) * group
    tile_start = pl.program_id(0) * width
    dma_tile = key_buffers.shape[1] * SLOTS
    fragments = width // dma_tile
    tasks = jnp.minimum(group, request_tile - first) * fragments
    if request_tile % group:

        @pl.when(first + group > request_tile)
        def clear_padding():
            start_row = request_tile % group * queries
            out[pl.ds(start_row, out.shape[0] - start_row), :] = jnp.full(
                (out.shape[0] - start_row, width), -jnp.inf, jnp.float32
            )

    def copy(task):
        row = start[0] + first + task // fragments
        key_start = tile_start + task % fragments * dma_tile
        return pltpu.make_async_copy(
            keys.at[row, pl.ds(key_start // SLOTS, dma_tile // SLOTS), :],
            key_buffers.at[task % 2],
            input_sems.at[task % 2],
        )

    copy(0).start()

    def body(local, task, unused):
        copy(task).wait()

        @pl.when(task + 1 < tasks)
        def prefetch():
            copy(task + 1).start()

        row = start[0] + first + local
        fragment_start = task % fragments * dma_tile
        for offset in range(0, dma_tile, compute_tile):
            key_start = fragment_start + offset
            section = pl.ds(key_start, compute_tile)
            kk = pltpu.bitcast(
                key_buffers[task % 2, pl.ds(offset // SLOTS, compute_tile // SLOTS), :],
                jnp.bfloat16,
            )
            logits = jax.lax.dot_general(
                q[row].reshape(queries * heads, dim),
                kk,
                (((1,), (1,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            if head_stripe is None:
                logits = logits.reshape(queries, heads, compute_tile)
                values = (jnp.maximum(logits, 0) * weights[row][:, :, None]).sum(1)
            else:
                values = common._reduce_heads(
                    logits, weights[row], queries, head_stripe
                )
            # Divide the static strides first so Mosaic can prove lane alignment.
            meta_start = task % fragments * (dma_tile // repeat) + offset // repeat
            meta_section = pl.ds(meta_start, compute_tile // repeat)
            base = _expand_metadata(ids[row, meta_section], repeat, dim)
            logical = base + offsets[section]
            members = _expand_metadata(membership[row, meta_section], repeat, dim)
            allow = ((members[None, :] >> jnp.arange(queries)[:, None]) & 1) != 0
            allow &= (logical[None, :] >= 0) & (
                logical[None, :] < lengths[row][:, None]
            )
            out[pl.ds(local * queries, queries), section] = jnp.where(
                allow, values, -jnp.inf
            )
        return unused

    # Q=6 output rows cross sublane boundaries; keep their offsets static.
    for local in range(group):
        jax.lax.fori_loop(
            local * fragments,
            jnp.minimum((local + 1) * fragments, tasks),
            functools.partial(body, local),
            None,
        )


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "key_tile",
        "interpret",
        "request_tile",
        "compute_tile",
        "head_stripe",
        "dma_tile",
    ),
)
def _candidate_scores(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    start_request: jax.Array,
    request_tile: int,
    block_size: int,
    key_tile: int = 4096,
    interpret: bool = False,
    compute_tile: int | None = None,
    head_stripe: int | None = None,
    dma_tile: int | None = None,
) -> jax.Array:
    """Stream packed keys into request-major 2-D score windows."""
    groups, queries, heads, dim = q.shape
    width = ids.shape[1] * block_size
    key_rows = width // SLOTS
    request_group = request_tile
    if not interpret:
        sublanes = pltpu.get_tpu_info().num_sublanes
        request_group = min(request_tile, sublanes // math.gcd(queries, sublanes))
    compute_tile = key_tile if compute_tile is None else compute_tile
    dma_tile = math.lcm(compute_tile, SLOTS) if dma_tile is None else dma_tile
    if (
        dim != DIM
        or q.dtype != jnp.bfloat16
        or keys.dtype != jnp.int32
        or weights.dtype != jnp.float32
        or weights.shape != (groups, queries, heads)
        or keys.shape != (groups, key_rows, ROW_WORDS)
        or membership.shape != ids.shape
        or lengths.shape != (groups, queries)
        or width < SLOTS
        or width % SLOTS
        or key_tile < SLOTS
        or width % key_tile
        or key_tile % SLOTS
        or compute_tile < 1
        or key_tile % compute_tile
        or dma_tile < SLOTS
        or dma_tile % SLOTS
        or key_tile % dma_tile
        or dma_tile % compute_tile
    ):
        raise ValueError("Expected aligned halfplane int32 candidate keys")
    if head_stripe is not None:
        if head_stripe < 1 or head_stripe & (head_stripe - 1):
            raise ValueError("head_stripe must be a positive power of two")
        query_group = math.gcd(queries, head_stripe)
        head_group = head_stripe // query_group
        if heads % head_group:
            raise ValueError("head_stripe must not require padding the head axis")
        shape = (
            groups,
            queries // query_group,
            query_group,
            heads // head_group,
            head_group,
        )
        q = (
            q.reshape(*shape, dim)
            .transpose(0, 3, 1, 4, 2, 5)
            .reshape(groups, queries * heads, dim)
        )
        weights = (
            weights.reshape(shape)
            .transpose(0, 3, 1, 4, 2)
            .reshape(groups, heads // head_group, queries * head_group)
        )
    # Keep compact metadata windows lane-aligned, including sub-block key tiles.
    metadata_stride = math.gcd(block_size, key_tile // min(key_tile, dim))
    metadata_tile = key_tile // metadata_stride
    if compute_tile % metadata_stride:
        raise ValueError("compute_tile must align with compact metadata")
    # A negative block must stay negative after adding any within-block offset.
    bases = jnp.where(ids >= 0, ids * block_size, -block_size)
    bases = jnp.repeat(bases, block_size // metadata_stride, axis=1)
    members = jnp.repeat(membership, block_size // metadata_stride, axis=1)
    offsets = jnp.arange(width, dtype=jnp.int32) % block_size
    metadata_spec = pl.BlockSpec((groups, metadata_tile), lambda t, g, start: (0, t))
    return pl.pallas_call(
        functools.partial(
            _score_kernel,
            queries=queries,
            request_tile=request_tile,
            compute_tile=compute_tile,
            head_stripe=head_stripe,
        ),
        out_shape=jax.ShapeDtypeStruct((request_tile * queries, width), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(width // key_tile, pl.cdiv(request_tile, request_group)),
            in_specs=[
                pl.BlockSpec(q.shape, lambda t, g, start: (0,) * q.ndim),
                pl.BlockSpec(weights.shape, lambda t, g, start: (0,) * weights.ndim),
                pl.BlockSpec(memory_space=pltpu.HBM),
                metadata_spec,
                metadata_spec,
                pl.BlockSpec((groups, queries), lambda t, g, start: (0, 0)),
                pl.BlockSpec((key_tile,), lambda t, g, start: (t,)),
            ],
            out_specs=pl.BlockSpec(
                (request_group * queries, key_tile), lambda t, g, start: (g, t)
            ),
            scratch_shapes=(
                pltpu.VMEM((2, dma_tile // SLOTS, ROW_WORDS), jnp.int32),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",) * 2),
        interpret=interpret,
        name="candidate_union_packed_scores",
    )(start_request.reshape(1), q, weights, keys, bases, members, lengths, offsets)


@functools.partial(
    jax.jit,
    static_argnames=(
        "block_size",
        "key_tile",
        "request_tile",
        "k",
        "overlap",
    ),
)
def candidate_indexer(
    q: jax.Array,
    weights: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    lengths: jax.Array,
    *,
    block_size: int,
    key_tile: int | None = None,
    request_tile: int | None = None,
    k: int = 512,
    overlap: bool = True,
) -> jax.Array:
    """Pipeline canonical int32 key scoring and SC selection over a reusable union."""
    requests, queries = q.shape[:2]
    width = ids.shape[1] * block_size
    config = tuned_block_sizes.get_candidate_block_sizes(
        tuned_block_sizes.CandidateTuningKey(
            jax.devices()[0].device_kind,
            queries,
            q.shape[2],
            width,
            key_tile,
            k,
        ),
        requests,
    )
    if request_tile is None:
        request_tile = config.request_tile
    elif type(request_tile) is not int or request_tile < 1:
        raise ValueError("request_tile must be a positive integer")
    request_tile = min(request_tile, requests)
    fuse_topk_id_mapping = config.fuse_topk_id_mapping

    def prepare(i):
        return i * request_tile

    def score(start, size=request_tile):
        return _candidate_scores(
            q,
            weights,
            keys,
            ids,
            membership,
            lengths,
            start_request=start,
            request_tile=size,
            block_size=block_size,
            key_tile=config.key_tile,
            compute_tile=config.compute_tile,
            head_stripe=config.head_stripe,
            dma_tile=config.dma_tile,
        )

    def select(i, values):
        group_size = values.shape[0] // queries
        group_ids = jax.lax.dynamic_slice_in_dim(ids, prepare(i), group_size)
        union_lengths = jnp.maximum(jnp.sum(group_ids >= 0, axis=1) * block_size, 1)
        scan_lengths = jnp.broadcast_to(
            union_lengths[:, None], (group_size, queries)
        ).astype(jnp.int32)
        if fuse_topk_id_mapping:
            return streaming_topk(
                values,
                valid_lengths=scan_lengths,
                k=k,
                block_size=config.topk_block_size,
                reconstruct_threshold=config.reconstruct_threshold,
                exclude_neg_inf=True,
                logical_blocks=ids,
                start_request=prepare(i),
            )
        selected = decode._select(
            values,
            scan_lengths.reshape(-1),
            k,
            config.topk_block_size,
            config.reconstruct_threshold,
        ).reshape(group_size, queries, k)
        blocks = jnp.take_along_axis(
            group_ids[:, None, :], jnp.maximum(selected, 0) // block_size, axis=2
        )
        logical = blocks * block_size + selected % block_size
        return jnp.where(selected >= 0, logical, -1).reshape(-1, k)

    def write(i, selected, out):
        return jax.lax.dynamic_update_slice_in_dim(
            out, selected, i * request_tile * queries, axis=0
        )

    count, tail = divmod(requests, request_tile)
    if overlap and (fuse_topk_id_mapping or tail):
        count += bool(tail)
        current = score(jnp.int32(0))
        previous = None
        selected_groups = []
        for i in range(count - 1):
            next_start = jnp.int32(prepare(i + 1))
            if previous is None:
                next_start, current = jax.lax.optimization_barrier(
                    (next_start, current)
                )
            else:
                next_start, current, previous = jax.lax.optimization_barrier(
                    (next_start, current, previous)
                )
                selected_groups.append(previous)
            following = score(
                next_start, min(request_tile, requests - (i + 1) * request_tile)
            )
            previous = select(i, current)
            current = following
        if previous is not None:
            previous, current = jax.lax.optimization_barrier((previous, current))
            selected_groups.append(previous)
        selected_groups.append(select(count - 1, current))
        return selected_groups[0] if count == 1 else jnp.concatenate(selected_groups)

    out = jnp.full((requests * queries, k), -1, jnp.int32)
    if overlap:
        return indexer.pipeline_score_topk(count, prepare, score, select, write, out)

    def serial(i, out):
        args, out = jax.lax.optimization_barrier((prepare(i), out))
        return write(i, select(i, score(args)), out)

    out = jax.lax.fori_loop(0, count, serial, out)
    if tail:
        start, out = jax.lax.optimization_barrier((prepare(count), out))
        out = write(count, select(count, score(start, tail)), out)
    return out


def cache_capacity(
    query_shape: tuple[int, ...],
    record_shape: tuple[int, ...],
    candidate_count: int,
    block_size: int,
    page_records: int,
) -> int | None:
    """Static envelope without specializing Torch's dynamic token dimension.

    Request/query alignment is checked after JAX lowering, where bucket shapes
    are concrete. Python guards on the token count break Dynamo's explicitly
    dynamic input contract.
    """
    _, width = record_shape
    _, heads, dim = query_shape
    if block_size < SLOTS or block_size % SLOTS:
        return None
    capacity = tuned_block_sizes.CANDIDATE_KEY_CAPACITY // block_size * block_size
    if (
        dim != DIM
        or heads < 1
        or candidate_count < 1
        or candidate_count * block_size > capacity
        or not 0 < capacity < width
        or page_records < block_size
        or page_records % block_size
        or width % page_records
    ):
        return None
    return capacity
