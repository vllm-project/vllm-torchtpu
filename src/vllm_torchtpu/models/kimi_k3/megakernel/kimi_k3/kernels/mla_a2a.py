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

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3 import config as k3_config
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels import (
    prefill_megakernel as pm,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kimi_k3.kernels.prefill_megakernel import (
    _HEADS,
    _Copies,
    _dot,
    _half_ring_streams,
    _mla_up_heads_into_bufs,
    _rank_half_gather,
)

from .mla_flash_deep_prefix import (
    can_fuse_mla_flash_a2a_deep_prefix,
    mla_flash_prefill_packed_a2a_deep_prefix,
    mla_flash_prefill_packed_deep_prefix,
)

RANKS = 32
HEAD_COLS = _HEADS * 128
HIDDEN = 7168
HALF_HIDDEN = HIDDEN // 2
TOTAL_HEAD_COLS = RANKS * HEAD_COLS
HALF_HEAD_COLS = TOTAL_HEAD_COLS // 2
OWNER_GATE_SOURCES = 2
OWNER_GATE_FP8_SOURCES = 4
OWNER_OUT_SOURCES = 4
OWNER_OUT_FP8_SOURCES = 8
OWNER_OUT_COLS = 3584
_FP8_E4M3_MAX = 448.0
_FP8_SCALE_EPS = 1e-12


def _quantize_rows_fp8(x):
    x_f32 = x.astype(jnp.float32)
    max_abs = jnp.maximum(
        jnp.max(jnp.abs(x_f32), axis=1, keepdims=True), _FP8_SCALE_EPS
    )
    return (x_f32 * (_FP8_E4M3_MAX / max_abs)).astype(jnp.float8_e4m3fn), max_abs * (
        1.0 / _FP8_E4M3_MAX
    )


def unstripe_sp_window(arr, num_ranks: int, stripe: int):
    tokens, width = arr.shape
    block_rows = num_ranks * stripe
    if block_rows <= 0 or tokens % block_rows:
        raise ValueError(
            f"unstripe_sp_window requires tokens ({tokens}) divisible by "
            f"num_ranks * stripe ({num_ranks} * {stripe})"
        )
    blocks = tokens // block_rows
    return (
        arr.reshape(num_ranks, blocks, stripe, width)
        .transpose(1, 0, 2, 3)
        .reshape(tokens, width)
    )


def _sp_token_index_ranks(token_owner, local, stripe: int, num_ranks: int):
    return (
        (local // stripe) * num_ranks * stripe + token_owner * stripe + local % stripe
    )


def mla_qkv_up_pallas(
    both_640,
    query_up_projection,
    key_value_up_projection,
    *,
    q_rank: int,
    tile: int,
    query_scale: float,
    length,
    interpret: bool = False,
):
    tokens = both_640.shape[0]
    n_tiles = tokens // tile
    len_arr = (
        jnp.asarray([tokens], jnp.int32)
        if length is None
        else jnp.asarray(length, jnp.int32).reshape((1,))
    )

    def body(
        len_ref,
        both_ref,
        qu_vmem,
        kvu_vmem,
        q_out_ref,
        k_out_ref,
        v_out_ref,
        both_bufs,
        q_bufs,
        k_bufs,
        v_bufs,
        in_sems,
        out_sems,
    ):
        live = len_ref[0]

        def load_tile(t, buf):
            return pltpu.make_async_copy(
                both_ref.at[pl.ds(t * tile, tile), :],
                both_bufs.at[buf],
                in_sems.at[buf],
            )

        def store_outs(t, buf):
            rs = pl.ds(t * tile, tile)
            copies = []
            for h in range(_HEADS):
                copies.append(
                    pltpu.make_async_copy(
                        q_bufs.at[buf, h], q_out_ref.at[h, rs, :], out_sems.at[buf, h]
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        k_bufs.at[buf, h],
                        k_out_ref.at[h, rs, :],
                        out_sems.at[buf, 3 + h],
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        v_bufs.at[buf, h],
                        v_out_ref.at[h, rs, :],
                        out_sems.at[buf, 6 + h],
                    )
                )
            return _Copies(copies)

        load_tile(0, 0).start()

        @pl.loop(0, n_tiles)
        def tile_loop(t):
            buf = t % 2
            load_tile(t, buf).wait()

            @pl.when(t >= 2)
            def wait_prev_store():
                store_outs(t - 2, buf).wait()

            @pl.when(t + 1 < n_tiles)
            def prefetch_next():
                load_tile(t + 1, 1 - buf).start()

            both_tile = both_bufs.at[buf][...]
            q_lat = both_tile[:, :q_rank]
            kv_lat = both_tile[:, q_rank : q_rank + 512]
            rope_pad_raw = both_tile[:, q_rank + 512 : q_rank + 640]
            real_rows = jnp.arange(tile, dtype=jnp.int32) + t * tile < live
            _mla_up_heads_into_bufs(
                q_lat,
                kv_lat,
                rope_pad_raw,
                qu_vmem,
                kvu_vmem,
                real_rows,
                q_bufs.at[buf],
                k_bufs.at[buf],
                v_bufs.at[buf],
                query_scale=query_scale,
            )
            store_outs(t, buf).start()

            @pl.when(t == n_tiles - 1)
            def drain_stores():
                store_outs(t, buf).wait()
                if n_tiles > 1:
                    store_outs(t - 1, 1 - buf).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out_shapes = (
        jax.ShapeDtypeStruct((_HEADS, tokens, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HEADS, tokens, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HEADS, tokens, 128), jnp.bfloat16),
    )
    return pl.pallas_call(
        body,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(hbm, vmem, vmem),
            out_specs=(hbm,) * 3,
            scratch_shapes=(
                pltpu.VMEM((2, tile, q_rank + 640), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 128), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2, 9)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=32 * 1024**2),
        interpret=interpret,
        name="kimi_mla_qkv_up_sp",
    )(len_arr, both_640, query_up_projection, key_value_up_projection)


def mla_qkv_up_unstriped_pallas(
    both_raw,
    query_up_projection,
    key_value_up_projection,
    *,
    num_ranks: int,
    stripe: int,
    q_rank: int,
    tile: int,
    query_scale: float,
    length,
    return_padded_latent_640: bool = False,
    interpret: bool = False,
):
    tokens, full_width = both_raw.shape
    rows = tokens // num_ranks
    blocks = rows // stripe
    block_rows = num_ranks * stripe
    if (
        tokens % block_rows
        or tokens % tile
        or tile % stripe
        or block_rows % tile
        or full_width != q_rank + 640
    ):
        raise ValueError(
            "mla_qkv_up_unstriped_pallas incompatible shapes: "
            f"both_raw={both_raw.shape}, "
            f"num_ranks={num_ranks}, stripe={stripe}, q_rank={q_rank}, tile={tile}"
        )
    len_arr = (
        jnp.asarray([tokens], jnp.int32)
        if length is None
        else jnp.asarray(length, jnp.int32).reshape((1,))
    )
    latent_cols = 640 if return_padded_latent_640 else 576
    if blocks == 1:
        q, k, v = mla_qkv_up_pallas(
            both_raw,
            query_up_projection,
            key_value_up_projection,
            q_rank=q_rank,
            tile=tile,
            query_scale=query_scale,
            length=len_arr,
            interpret=interpret,
        )
        return q, k, v, both_raw[:, q_rank : q_rank + latent_cols]

    both_4d = both_raw.reshape(num_ranks, blocks, stripe, full_width)
    n_tiles = tokens // tile
    owners_per_tile = tile // stripe
    tiles_per_block = num_ranks // owners_per_tile

    def body(
        len_ref,
        both_4d_ref,
        qu_vmem,
        kvu_vmem,
        q_out_ref,
        k_out_ref,
        v_out_ref,
        lat_out_ref,
        both_bufs,
        q_bufs,
        k_bufs,
        v_bufs,
        in_sems,
        out_sems,
        lat_sems,
    ):
        live = len_ref[0]

        def load_tile(t, slot):
            block_idx = t // tiles_per_block
            owner_base = (t % tiles_per_block) * owners_per_tile
            return _Copies(
                [
                    pltpu.make_async_copy(
                        both_4d_ref.at[owner_base + k, block_idx],
                        both_bufs.at[slot, pl.ds(k * stripe, stripe)],
                        in_sems.at[slot],
                    )
                    for k in range(owners_per_tile)
                ]
            )

        def store_qkv(t, slot):
            rs = pl.ds(t * tile, tile)
            copies = []
            for h in range(_HEADS):
                copies.append(
                    pltpu.make_async_copy(
                        q_bufs.at[slot, h], q_out_ref.at[h, rs, :], out_sems.at[slot, h]
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        k_bufs.at[slot, h],
                        k_out_ref.at[h, rs, :],
                        out_sems.at[slot, 3 + h],
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        v_bufs.at[slot, h],
                        v_out_ref.at[h, rs, :],
                        out_sems.at[slot, 6 + h],
                    )
                )
            return _Copies(copies)

        def store_latent(t, in_slot):
            return pltpu.make_async_copy(
                both_bufs.at[in_slot, :, pl.ds(q_rank, 640)],
                lat_out_ref.at[pl.ds(t * tile, tile), :],
                lat_sems.at[in_slot],
            )

        load_tile(0, 0).start()

        @pl.loop(0, n_tiles)
        def tile_loop(t):
            in_slot = t % 3
            out_slot = t % 2
            load_tile(t, in_slot).wait()

            @pl.when(t >= 2)
            def wait_prev_store():
                store_qkv(t - 2, out_slot).wait()
                store_latent(t - 2, (t - 2) % 3).wait()

            @pl.when(t + 1 < n_tiles)
            def prefetch_next():
                load_tile(t + 1, (t + 1) % 3).start()

            both_tile = both_bufs.at[in_slot][...]
            q_lat = both_tile[:, :q_rank]
            kv_lat = both_tile[:, q_rank : q_rank + 512]
            rope_pad_raw = both_tile[:, q_rank + 512 : q_rank + 640]
            real_rows = jnp.arange(tile, dtype=jnp.int32) + t * tile < live
            _mla_up_heads_into_bufs(
                q_lat,
                kv_lat,
                rope_pad_raw,
                qu_vmem,
                kvu_vmem,
                real_rows,
                q_bufs.at[out_slot],
                k_bufs.at[out_slot],
                v_bufs.at[out_slot],
                query_scale=query_scale,
            )
            store_qkv(t, out_slot).start()
            store_latent(t, in_slot).start()

            @pl.when(t == n_tiles - 1)
            def drain_stores():
                store_qkv(t, out_slot).wait()
                store_latent(t, in_slot).wait()
                if n_tiles > 1:
                    store_qkv(t - 1, 1 - out_slot).wait()
                    store_latent(t - 1, (t - 1) % 3).wait()

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out_shapes = (
        jax.ShapeDtypeStruct((_HEADS, tokens, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HEADS, tokens, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HEADS, tokens, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 640), jnp.bfloat16),
    )
    q, k, v, lat_640 = pl.pallas_call(
        body,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(hbm, vmem, vmem),
            out_specs=(hbm,) * 4,
            scratch_shapes=(
                pltpu.VMEM((3, tile, full_width), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 128), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.SemaphoreType.DMA((2, 9)),
                pltpu.SemaphoreType.DMA((3,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=32 * 1024**2),
        interpret=interpret,
        name="kimi_mla_qkv_up_unstriped_sp",
    )(len_arr, both_4d, query_up_projection, key_value_up_projection)
    latent = lat_640 if return_padded_latent_640 else lat_640[:, :576]
    return q, k, v, latent


def mla_qkv_up_unstriped_and_prefix_gather_pallas(
    both_raw,
    query_up_projection,
    key_value_up_projection,
    pool,
    tables,
    counts,
    ids,
    order,
    *,
    num_ranks: int,
    stripe: int,
    q_rank: int,
    tile: int,
    query_scale: float,
    length,
    sp_axis: str = "k3_sp",
    collective_id: int = 44,
    return_padded_latent_640: bool = False,
    interpret: bool | pltpu.InterpretParams = False,
):
    tokens, full_width = both_raw.shape
    rows = tokens // num_ranks
    blocks = rows // stripe
    block_rows = num_ranks * stripe
    if (
        (num_ranks != 1 and num_ranks % 2 != 0)
        or tokens % block_rows
        or tokens % tile
        or tile % stripe
        or block_rows % tile
        or full_width != q_rank + 640
        or pool.ndim != 4
        or tables.shape != (3, 3, num_ranks)
        or counts.shape != (num_ranks,)
        or ids.ndim != 1
        or order.ndim != 1
    ):
        raise ValueError(
            "mla_qkv_up_unstriped_and_prefix_gather_pallas incompatible shapes: "
            f"both_raw={both_raw.shape}, pool={pool.shape}, tables={tables.shape}, "
            f"counts={counts.shape}, ids={ids.shape}, order={order.shape}, "
            f"num_ranks={num_ranks}, stripe={stripe}, q_rank={q_rank}, tile={tile}"
        )
    len_arr = (
        jnp.asarray([tokens], jnp.int32)
        if length is None
        else jnp.asarray(length, jnp.int32).reshape((1,))
    )
    slots = ids.shape[0]
    both_4d = both_raw.reshape(num_ranks, blocks, stripe, full_width)
    n_tiles = tokens // tile
    owners_per_tile = tile // stripe
    tiles_per_block = num_ranks // owners_per_tile

    def body(
        len_ref,
        tables_ref,
        counts_ref,
        ids_ref,
        order_ref,
        both_4d_ref,
        qu_vmem,
        kvu_vmem,
        pool_hbm,
        q_out_ref,
        k_out_ref,
        v_out_ref,
        lat_out_ref,
        pages_out_hbm,
        both_bufs,
        q_bufs,
        k_bufs,
        v_bufs,
        page_buf,
        in_sems,
        out_sems,
        lat_sems,
        local_sems,
        send_sems,
        recv_sems,
        done,
    ):
        live = len_ref[0]
        total_pages = order_ref[0]

        def load_tile(t, slot):
            block_idx = t // tiles_per_block
            owner_base = (t % tiles_per_block) * owners_per_tile
            return _Copies(
                [
                    pltpu.make_async_copy(
                        both_4d_ref.at[owner_base + k, block_idx],
                        both_bufs.at[slot, pl.ds(k * stripe, stripe)],
                        in_sems.at[slot],
                    )
                    for k in range(owners_per_tile)
                ]
            )

        def store_qkv(t, slot):
            rs = pl.ds(t * tile, tile)
            copies = []
            for h in range(_HEADS):
                copies.append(
                    pltpu.make_async_copy(
                        q_bufs.at[slot, h],
                        q_out_ref.at[h, rs, :],
                        out_sems.at[slot, h],
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        k_bufs.at[slot, h],
                        k_out_ref.at[h, rs, :],
                        out_sems.at[slot, 3 + h],
                    )
                )
                copies.append(
                    pltpu.make_async_copy(
                        v_bufs.at[slot, h],
                        v_out_ref.at[h, rs, :],
                        out_sems.at[slot, 6 + h],
                    )
                )
            return _Copies(copies)

        def store_latent(t, in_slot):
            return pltpu.make_async_copy(
                both_bufs.at[in_slot, :, pl.ds(q_rank, 640)],
                lat_out_ref.at[pl.ds(t * tile, tile), :],
                lat_sems.at[in_slot],
            )

        max_hops = num_ranks // 2
        if num_ranks > 1:
            rank = jax.lax.axis_index(sp_axis)
            positions = [tables_ref[f, 0, rank] for f in range(3)]

            def ring(family, position):
                return tables_ref[family, 1, position % num_ranks]

            neighbours = [ring(f, positions[f] + d) for f in range(3) for d in (-1, 1)]
            streams = []
            for stream in range(pm.GATHER_STREAMS):
                family = stream // 2
                direction = 1 if stream % 2 == 0 else -1
                hops = max_hops if direction == 1 else max_hops - 1
                streams.append(
                    (
                        stream,
                        family,
                        direction,
                        hops,
                        ring(family, positions[family] + direction),
                    )
                )

            def chunks(owner, family):
                return jnp.maximum(counts_ref[owner] - family + 2, 0) // 3

            def chunk(owner, family, i):
                return pages_out_hbm.at[owner * slots + 1 + family + 3 * i]

            def remote(src, stream, step, peer):
                return pltpu.make_async_remote_copy(
                    src,
                    src,
                    send_sems.at[stream],
                    recv_sems.at[stream, step],
                    device_id=(peer,),
                    device_id_type=pl.DeviceIdType.MESH,
                )

            def do_hop(k):
                for stream, family, direction, hops, downstream in streams:

                    @pl.when(k <= hops)
                    def active_hop(
                        stream=stream,
                        family=family,
                        direction=direction,
                        hops=hops,
                        downstream=downstream,
                    ):
                        owner = ring(family, positions[family] - direction * k)
                        size = chunks(owner, family)

                        @pl.loop(0, size)
                        def receive(i, stream=stream, family=family, owner=owner):
                            remote(chunk(owner, family, i), stream, k, rank).wait_recv()

                        @pl.when(k < hops)
                        def forward(
                            stream=stream,
                            family=family,
                            owner=owner,
                            size=size,
                            downstream=downstream,
                        ):
                            @pl.loop(0, size)
                            def send(i):
                                remote(
                                    chunk(owner, family, i), stream, k + 1, downstream
                                ).start()

            hop_stride = max(1, n_tiles // max_hops)
            in_loop_hops = min(max_hops, max(0, (n_tiles - 1) // hop_stride))
        else:
            hop_stride = 1
            in_loop_hops = 0

        load_tile(0, 0).start()

        @pl.when(total_pages > 0)
        def setup_prefix_ring():
            if num_ranks == 1:
                pm.stage_local_prefix_pages(
                    pool_hbm,
                    ids_ref,
                    pages_out_hbm,
                    page_buf,
                    local_sems,
                    0,
                    slots,
                    counts_ref[0],
                )
                return
            count = counts_ref[rank]
            barrier = pltpu.get_barrier_semaphore()
            for peer in neighbours:
                pl.semaphore_signal(
                    barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
                )
            pm.stage_local_prefix_pages(
                pool_hbm,
                ids_ref,
                pages_out_hbm,
                page_buf,
                local_sems,
                rank,
                slots,
                count,
            )
            pl.semaphore_wait(barrier, len(neighbours))
            for stream, family, _, _, downstream in streams:

                @pl.loop(0, chunks(rank, family))
                def seed(i, stream=stream, family=family, downstream=downstream):
                    remote(chunk(rank, family, i), stream, 1, downstream).start()

        @pl.loop(0, n_tiles)
        def tile_loop(t):
            in_slot = t % 3
            out_slot = t % 2
            if in_loop_hops > 0:
                hop = t // hop_stride

                @pl.when(
                    (total_pages > 0)
                    & (t > 0)
                    & (t % hop_stride == 0)
                    & (hop <= in_loop_hops)
                )
                def advance_ring():
                    do_hop(hop)

            load_tile(t, in_slot).wait()

            @pl.when(t >= 2)
            def wait_prev_store():
                store_qkv(t - 2, out_slot).wait()
                store_latent(t - 2, (t - 2) % 3).wait()

            @pl.when(t + 1 < n_tiles)
            def prefetch_next():
                load_tile(t + 1, (t + 1) % 3).start()

            both_tile = both_bufs.at[in_slot][...]
            q_lat = both_tile[:, :q_rank]
            kv_lat = both_tile[:, q_rank : q_rank + 512]
            rope_pad_raw = both_tile[:, q_rank + 512 : q_rank + 640]
            real_rows = jnp.arange(tile, dtype=jnp.int32) + t * tile < live
            _mla_up_heads_into_bufs(
                q_lat,
                kv_lat,
                rope_pad_raw,
                qu_vmem,
                kvu_vmem,
                real_rows,
                q_bufs.at[out_slot],
                k_bufs.at[out_slot],
                v_bufs.at[out_slot],
                query_scale=query_scale,
            )
            store_qkv(t, out_slot).start()
            store_latent(t, in_slot).start()

            @pl.when(t == n_tiles - 1)
            def drain_stores():
                store_qkv(t, out_slot).wait()
                store_latent(t, in_slot).wait()
                if n_tiles > 1:
                    store_qkv(t - 1, 1 - out_slot).wait()
                    store_latent(t - 1, (t - 1) % 3).wait()

        if num_ranks > 1:

            @pl.when(total_pages > 0)
            def finish_prefix_ring():
                if in_loop_hops < max_hops:

                    @pl.loop(in_loop_hops + 1, max_hops + 1)
                    def tail_hops(k):
                        do_hop(k)

                @pl.loop(0, max_hops)
                def drain(k):
                    for stream, family, direction, hops, downstream in streams:

                        @pl.when(k < hops)
                        def active_drain(
                            stream=stream,
                            family=family,
                            direction=direction,
                            downstream=downstream,
                        ):
                            owner = ring(family, positions[family] - direction * k)

                            @pl.loop(0, chunks(owner, family))
                            def sent(
                                i,
                                stream=stream,
                                family=family,
                                owner=owner,
                                downstream=downstream,
                            ):
                                remote(
                                    chunk(owner, family, i), stream, k + 1, downstream
                                ).wait_send()

                for peer in neighbours:
                    pl.semaphore_signal(
                        done, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
                    )
                pl.semaphore_wait(done, len(neighbours))

    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    out_shapes = (
        jax.ShapeDtypeStruct((_HEADS, tokens, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HEADS, tokens, 256), jnp.bfloat16),
        jax.ShapeDtypeStruct((_HEADS, tokens, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((tokens, 640), jnp.bfloat16),
        jax.ShapeDtypeStruct((num_ranks * slots, *pool.shape[1:]), pool.dtype),
    )
    compiler_params = (
        pltpu.CompilerParams(collective_id=collective_id, vmem_limit_bytes=32 * 1024**2)
        if num_ranks > 1
        else pltpu.CompilerParams(vmem_limit_bytes=32 * 1024**2)
    )
    q, k, v, lat_640, pages = pl.pallas_call(
        body,
        out_shape=out_shapes,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=5,
            in_specs=(hbm, vmem, vmem, hbm),
            out_specs=(hbm,) * 5,
            scratch_shapes=(
                pltpu.VMEM((3, tile, full_width), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 256), jnp.bfloat16),
                pltpu.VMEM((2, _HEADS, tile, 128), jnp.bfloat16),
                pltpu.VMEM((2, *pool.shape[1:]), pool.dtype),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.SemaphoreType.DMA((2, 9)),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.SemaphoreType.DMA((2, 2)),
                pltpu.SemaphoreType.DMA((pm.GATHER_STREAMS,)),
                pltpu.SemaphoreType.DMA((pm.GATHER_STREAMS, num_ranks)),
                pltpu.SemaphoreType.REGULAR,
            ),
        ),
        compiler_params=compiler_params,
        interpret=interpret,
        name="kimi_mla_qkv_up_unstriped_prefix_gather_sp",
    )(
        len_arr,
        jnp.asarray(tables, jnp.int32),
        jnp.asarray(counts, jnp.int32),
        jnp.asarray(ids, jnp.int32),
        jnp.asarray(order, jnp.int32),
        both_4d,
        query_up_projection,
        key_value_up_projection,
        pool,
    )
    latent = lat_640 if return_padded_latent_640 else lat_640[:, :576]
    return q, k, v, latent, pages


def _attended_a2a_kernel(
    tables_ref, attended_ref, out_ref, send_sem, recv_sem, *, num_ranks, stripe, blocks
):
    rank = jax.lax.axis_index("k3_sp")

    def remote_block(block, peer):
        owner = tables_ref[0, 2, peer]
        src = attended_ref.at[pl.ds((block * num_ranks + owner) * stripe, stripe)]
        dst = out_ref.at[rank, pl.ds(block * stripe, stripe)]
        return pltpu.make_async_remote_copy(
            src,
            dst,
            send_sem,
            recv_sem,
            device_id=(peer,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    def local_block(block):
        owner = tables_ref[0, 2, rank]
        src = attended_ref.at[pl.ds((block * num_ranks + owner) * stripe, stripe)]
        dst = out_ref.at[rank, pl.ds(block * stripe, stripe)]
        return pltpu.make_async_copy(src, dst, recv_sem)

    @pl.loop(0, blocks)
    def start_local(block):
        local_block(block).start()

    if num_ranks > 1:
        barrier = pltpu.get_barrier_semaphore()
        for peer in range(num_ranks):
            pl.semaphore_signal(
                barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(barrier, num_ranks)

        @pl.loop(0, blocks)
        def start_remote(block):
            @pl.loop(1, num_ranks)
            def _(offset):
                remote_block(block, (rank + offset) % num_ranks).start()

        @pl.loop(0, blocks)
        def wait_remote(block):
            @pl.loop(1, num_ranks)
            def _(offset):
                remote_block(block, (rank + offset) % num_ranks).wait_send()

    @pl.loop(0, blocks * num_ranks)
    def wait_recv(_):
        local_block(0).wait()


def mla_attended_all_to_all_pallas(
    tables, attended, *, num_ranks: int, stripe: int, collective_id: int = 31
):
    tokens, cols = attended.shape
    rows = tokens // num_ranks
    if tokens % (num_ranks * stripe) or cols != HEAD_COLS:
        raise ValueError(
            f"attended {attended.shape} must have cols={HEAD_COLS} and "
            f"tokens divisible by num_ranks * stripe ({num_ranks * stripe})"
        )
    blocks = rows // stripe
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _attended_a2a_kernel, num_ranks=num_ranks, stripe=stripe, blocks=blocks
        ),
        out_shape=jax.ShapeDtypeStruct((num_ranks, rows, cols), attended.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=(hbm,),
            out_specs=hbm,
            scratch_shapes=(pltpu.SemaphoreType.DMA, pltpu.SemaphoreType.DMA),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=32 * 1024**2
        ),
        name="kimi_prefill_mla_attended_a2a",
    )(jnp.asarray(tables, jnp.int32), attended)


def mla_attended_all_to_all_ref(
    attended, *, token_ranks, num_ranks: int, stripe: int, sp_axis: str
):
    tokens, cols = attended.shape
    rows = tokens // num_ranks
    blocks = rows // stripe
    owner_order = jnp.asarray(token_ranks, jnp.int32)
    by_owner = attended.reshape(blocks, num_ranks, stripe, cols).transpose(1, 0, 2, 3)
    by_peer = by_owner[owner_order].reshape(num_ranks, rows, cols)
    return jax.lax.all_to_all(by_peer, sp_axis, split_axis=0, concat_axis=0)


def owner_mla_gate_and_output_ref(
    local_normed,
    owner_rows,
    m_gate,
    m_o,
    prefix,
    *,
    stripe: int,
    length,
    token_owner,
    m_gate_scale=None,
    m_o_scale=None,
):
    num_ranks, rows, cols = owner_rows.shape
    if m_gate_scale is not None:
        q_normed, norm_scale = _quantize_rows_fp8(local_normed)
        raw_gate = jnp.dot(q_normed, m_gate, preferred_element_type=jnp.float32)
        output_gate = (raw_gate * norm_scale * m_gate_scale[0:1, :]).astype(
            jnp.bfloat16
        )
    else:
        output_gate = jnp.dot(
            local_normed, m_gate, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
    gate = jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
    attended = owner_rows.transpose(1, 0, 2).reshape(rows, num_ranks * cols)
    gated = (attended * gate).astype(jnp.bfloat16)
    if m_o_scale is not None:
        q_gated, g_scale = _quantize_rows_fp8(gated)
        projection = (
            jnp.dot(q_gated, m_o, preferred_element_type=jnp.float32)
            * g_scale
            * m_o_scale[0:1, :]
        )
    else:
        projection = jnp.dot(gated, m_o, preferred_element_type=jnp.float32)
    local = jnp.arange(rows, dtype=jnp.int32)
    valid = _sp_token_index_ranks(token_owner, local, stripe, num_ranks) < length
    return (
        prefix.astype(jnp.float32) + jnp.where(valid[:, None], projection, 0.0)
    ).astype(jnp.bfloat16)


def _owner_gate_kernel(scalars_ref, normed_ref, rows_ref, gate_w_ref, gated_out_ref):
    del scalars_ref
    group = rows_ref.shape[0]
    output_gate = jnp.dot(
        normed_ref[...], gate_w_ref[...], preferred_element_type=jnp.float32
    ).astype(jnp.bfloat16)
    gate = jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
    attended = jnp.concatenate([rows_ref[j] for j in range(group)], axis=1)
    gated_out_ref[...] = (attended * gate).astype(jnp.bfloat16)


def _owner_gate_fp8_kernel(
    scalars_ref,
    q_normed_ref,
    norm_scale_ref,
    rows_ref,
    gate_w_ref,
    gate_scale_ref,
    gated_out_ref,
):
    del scalars_ref
    group = rows_ref.shape[0]
    raw_gate = jnp.dot(
        q_normed_ref[...], gate_w_ref[...], preferred_element_type=jnp.float32
    )
    output_gate = (raw_gate * norm_scale_ref[:, 0:1] * gate_scale_ref[0:1, :]).astype(
        jnp.bfloat16
    )
    gate = jax.nn.sigmoid(output_gate.astype(jnp.float32)).astype(jnp.bfloat16)
    attended = jnp.concatenate([rows_ref[j] for j in range(group)], axis=1)
    gated_out_ref[...] = (attended * gate).astype(jnp.bfloat16)


def _owner_output_kernel(
    scalars_ref, gated_ref, out_w_ref, prefix_ref, out_ref, acc_ref, *, stripe: int
):
    source_group = pl.program_id(1)
    rows = gated_ref.shape[0]

    @pl.when(source_group == 0)
    def init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, acc_ref.dtype)

    acc_ref[...] += jnp.dot(
        gated_ref[...], out_w_ref[...], preferred_element_type=jnp.float32
    )

    @pl.when(source_group == pl.num_programs(1) - 1)
    def finish():
        length, token_owner = scalars_ref[1], scalars_ref[2]
        local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
        valid = pm.sp_token_index(token_owner, local, stripe) < length
        residual = prefix_ref[...].astype(jnp.float32)
        out_ref[...] = (residual + jnp.where(valid, acc_ref[...], 0.0)).astype(
            out_ref.dtype
        )


def _owner_output_fp8_kernel(
    scalars_ref,
    gated_ref,
    out_w_ref,
    out_scale_ref,
    prefix_ref,
    out_ref,
    acc_ref,
    *,
    stripe: int,
):
    source_group = pl.program_id(1)
    rows = gated_ref.shape[0]

    @pl.when(source_group == 0)
    def init():
        acc_ref[...] = jnp.zeros(acc_ref.shape, acc_ref.dtype)

    q_gated, g_row_scale = _quantize_rows_fp8(gated_ref[...])
    acc_ref[...] += (
        jnp.dot(q_gated, out_w_ref[...], preferred_element_type=jnp.float32)
        * g_row_scale
    )

    @pl.when(source_group == pl.num_programs(1) - 1)
    def finish():
        length, token_owner = scalars_ref[1], scalars_ref[2]
        local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
        valid = pm.sp_token_index(token_owner, local, stripe) < length
        residual = prefix_ref[...].astype(jnp.float32)
        proj = acc_ref[...] * out_scale_ref[0:1, :]
        out_ref[...] = (residual + jnp.where(valid, proj, 0.0)).astype(out_ref.dtype)


def owner_mla_gate_and_output_pallas(
    local_normed,
    owner_rows,
    m_gate_family,
    m_o_family,
    prefix,
    *,
    layer,
    length,
    token_owner,
    stripe: int,
    m_gate_scales=None,
    m_o_scales=None,
    interpret: bool = False,
):
    rows, width = local_normed.shape
    ranks, owner_len, cols = owner_rows.shape
    total_cols = RANKS * HEAD_COLS
    if (
        ranks != RANKS
        or cols != HEAD_COLS
        or owner_len != rows
        or width % OWNER_OUT_COLS
        or m_gate_family.shape[-2:] != (width, total_cols)
        or m_o_family.shape[-2:] != (total_cols, width)
    ):
        raise ValueError(
            "MLA owner projection expects owner_rows=[32, rows, 384], "
            "m_gate=[slots, H, 12288], m_o=[slots, 12288, H]"
        )
    is_gate_fp8 = m_gate_family.dtype == jnp.float8_e4m3fn
    is_o_fp8 = m_o_family.dtype == jnp.float8_e4m3fn
    if is_gate_fp8 != is_o_fp8:
        raise ValueError("m_gate_family and m_o_family must share BF16 or FP8 dtype")
    scalars = jnp.stack(
        [jnp.asarray(v, jnp.int32).reshape(()) for v in (layer, length, token_owner)]
    )
    if is_gate_fp8:
        slots = m_gate_family.shape[0]
        if m_gate_scales is None or m_o_scales is None:
            raise ValueError(
                "FP8 MLA owner projections require m_gate_scales and m_o_scales"
            )
        if m_gate_scales.shape != (slots, 8, total_cols) or m_o_scales.shape != (
            slots,
            8,
            width,
        ):
            raise ValueError(
                f"FP8 MLA owner scales must be {(slots, 8, total_cols)} and "
                f"{(slots, 8, width)}"
            )
        if rows % 8:
            raise ValueError(
                f"FP8 MLA owner projections require rows ({rows}) divisible by 8"
            )
        q_normed, norm_row_scale = _quantize_rows_fp8(local_normed)
        norm_scale_128 = jnp.broadcast_to(norm_row_scale, (rows, 128))
        gated = pl.pallas_call(
            _owner_gate_fp8_kernel,
            out_shape=jax.ShapeDtypeStruct((rows, total_cols), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(RANKS // OWNER_GATE_FP8_SOURCES,),
                in_specs=[
                    pl.BlockSpec((rows, width), lambda s, sc: (0, 0)),
                    pl.BlockSpec((rows, 128), lambda s, sc: (0, 0)),
                    pl.BlockSpec(
                        (OWNER_GATE_FP8_SOURCES, rows, HEAD_COLS),
                        lambda s, sc: (s, 0, 0),
                    ),
                    pl.BlockSpec(
                        (None, width, OWNER_GATE_FP8_SOURCES * HEAD_COLS),
                        lambda s, sc: (sc[0], 0, s),
                    ),
                    pl.BlockSpec(
                        (None, 8, OWNER_GATE_FP8_SOURCES * HEAD_COLS),
                        lambda s, sc: (sc[0], 0, s),
                    ),
                ],
                out_specs=pl.BlockSpec(
                    (rows, OWNER_GATE_FP8_SOURCES * HEAD_COLS), lambda s, sc: (0, s)
                ),
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel",), vmem_limit_bytes=64 * 1024**2
            ),
            interpret=interpret,
            name="kimi_prefill_mla_owner_gate_fp8",
        )(scalars, q_normed, norm_scale_128, owner_rows, m_gate_family, m_gate_scales)
        return pl.pallas_call(
            functools.partial(_owner_output_fp8_kernel, stripe=stripe),
            out_shape=jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(width // OWNER_OUT_COLS, RANKS // OWNER_OUT_FP8_SOURCES),
                in_specs=[
                    pl.BlockSpec(
                        (rows, OWNER_OUT_FP8_SOURCES * HEAD_COLS),
                        lambda n, s, sc: (0, s),
                    ),
                    pl.BlockSpec(
                        (None, OWNER_OUT_FP8_SOURCES * HEAD_COLS, OWNER_OUT_COLS),
                        lambda n, s, sc: (sc[0], s, n),
                    ),
                    pl.BlockSpec(
                        (None, 8, OWNER_OUT_COLS),
                        lambda n, s, sc: (sc[0], 0, n),
                    ),
                    pl.BlockSpec((rows, OWNER_OUT_COLS), lambda n, s, sc: (0, n)),
                ],
                out_specs=pl.BlockSpec((rows, OWNER_OUT_COLS), lambda n, s, sc: (0, n)),
                scratch_shapes=[pltpu.VMEM((rows, OWNER_OUT_COLS), jnp.float32)],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
                vmem_limit_bytes=64 * 1024**2,
            ),
            interpret=interpret,
            name="kimi_prefill_mla_owner_projection_fp8",
        )(scalars, gated, m_o_family, m_o_scales, prefix)

    if m_gate_scales is not None or m_o_scales is not None:
        raise ValueError("m_gate_scales and m_o_scales require float8_e4m3fn weights")
    gated = pl.pallas_call(
        _owner_gate_kernel,
        out_shape=jax.ShapeDtypeStruct((rows, total_cols), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(RANKS // OWNER_GATE_SOURCES,),
            in_specs=[
                pl.BlockSpec((rows, width), lambda s, sc: (0, 0)),
                pl.BlockSpec(
                    (OWNER_GATE_SOURCES, rows, HEAD_COLS), lambda s, sc: (s, 0, 0)
                ),
                pl.BlockSpec(
                    (None, width, OWNER_GATE_SOURCES * HEAD_COLS),
                    lambda s, sc: (sc[0], 0, s),
                ),
            ],
            out_specs=pl.BlockSpec(
                (rows, OWNER_GATE_SOURCES * HEAD_COLS), lambda s, sc: (0, s)
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), vmem_limit_bytes=64 * 1024**2
        ),
        interpret=interpret,
        name="kimi_prefill_mla_owner_gate",
    )(scalars, local_normed, owner_rows, m_gate_family)
    return pl.pallas_call(
        functools.partial(_owner_output_kernel, stripe=stripe),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(width // OWNER_OUT_COLS, RANKS // OWNER_OUT_SOURCES),
            in_specs=[
                pl.BlockSpec(
                    (rows, OWNER_OUT_SOURCES * HEAD_COLS), lambda n, s, sc: (0, s)
                ),
                pl.BlockSpec(
                    (None, OWNER_OUT_SOURCES * HEAD_COLS, OWNER_OUT_COLS),
                    lambda n, s, sc: (sc[0], s, n),
                ),
                pl.BlockSpec((rows, OWNER_OUT_COLS), lambda n, s, sc: (0, n)),
            ],
            out_specs=pl.BlockSpec((rows, OWNER_OUT_COLS), lambda n, s, sc: (0, n)),
            scratch_shapes=[pltpu.VMEM((rows, OWNER_OUT_COLS), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            vmem_limit_bytes=64 * 1024**2,
        ),
        interpret=interpret,
        name="kimi_prefill_mla_owner_projection",
    )(scalars, gated, m_o_family, prefix)


def mla_local_normed_a2a_attention(
    local_hidden,
    prefix,
    *,
    query_down_projection,
    query_up_projection,
    key_value_down_projection,
    key_value_up_projection,
    query_normalization,
    key_value_normalization,
    m_gate_family,
    m_o_family,
    slot,
    length,
    segment_desc,
    num_ranks: int,
    sp_axis: str,
    segments: int,
    flash_tile: int,
    token_ranks,
    m_gate_scales=None,
    m_o_scales=None,
    tables=None,
    a2a_collective_id: int = 31,
    gather_collective_id: int = 45,
    prefix_gather_collective_id: int = 44,
    prefix_gather_fn: Callable[[], tuple[jax.Array, jax.Array]] | None = None,
    prefix_gather_inputs: tuple | None = None,
    arena_override=None,
    flash_fn=None,
    epsilon: float = 1e-5,
    query_scale: float = pm.SCALE_LOG2,
    use_pallas_a2a: bool = True,
    use_pallas_owner: bool = True,
    return_padded_latent_640: bool = False,
    interpret: bool | pltpu.InterpretParams = False,
):
    if prefix_gather_fn is not None and prefix_gather_inputs is not None:
        raise ValueError("Pass at most one of prefix_gather_fn or prefix_gather_inputs")
    if prefix_gather_inputs is not None and tables is None:
        raise ValueError("prefix_gather_inputs requires ring tables")
    rows = local_hidden.shape[0]
    tokens = rows * num_ranks
    stripe = pm.window_stripe(rows)

    query_latent = _dot(local_hidden, query_down_projection)
    query_latent = (
        query_latent
        * jax.lax.rsqrt(
            jnp.mean(query_latent * query_latent, axis=1, keepdims=True) + epsilon
        )
        * query_normalization
    ).astype(jnp.bfloat16)
    compressed = _dot(local_hidden, key_value_down_projection).astype(jnp.bfloat16)
    kv_latent = compressed[:, :512].astype(jnp.float32)
    kv_latent = (
        kv_latent
        * jax.lax.rsqrt(
            jnp.mean(kv_latent * kv_latent, axis=1, keepdims=True) + epsilon
        )
        * key_value_normalization
    ).astype(jnp.bfloat16)
    rope = compressed[:, 512:576]

    q_rank = query_latent.shape[1]
    padded_rope = jnp.pad(rope, ((0, 0), (0, 64)))
    packed_local = jnp.concatenate((query_latent, kv_latent, padded_rope), axis=1)
    slot_scalar = jnp.asarray(slot, jnp.int32).reshape(())
    width = local_hidden.shape[1]
    fuse_ag_gate = can_fuse_mla_ag_and_gate_chip_pair(
        tables=tables,
        sp_axis=sp_axis,
        num_ranks=num_ranks,
        rows=rows,
        width=width,
        packed_width=packed_local.shape[1],
        m_gate_family=m_gate_family,
        m_o_family=m_o_family,
        m_gate_scales=m_gate_scales,
        m_o_scales=m_o_scales,
        use_pallas_owner=use_pallas_owner,
        interpret=interpret,
    ) and (num_ranks == RANKS or isinstance(interpret, pltpu.InterpretParams))
    gate_both_precomputed = None
    if fuse_ag_gate:
        raw_rank_major, gate_both_precomputed = (
            rank_half_gather_and_mla_gate_chip_pair_pallas(
                packed_local,
                local_hidden,
                m_gate_family,
                tables,
                layer=slot_scalar,
                num_ranks=num_ranks,
                collective_id=gather_collective_id,
                interpret=interpret,
            )
        )
    elif (
        tables is not None
        and sp_axis == "k3_sp"
        and num_ranks == RANKS
        and (q_rank + 576) % 128 == 64
    ):
        raw_rank_major = _rank_half_gather(
            packed_local,
            tables,
            collective_id=gather_collective_id,
        )
    else:
        both_raw = jax.lax.all_gather(packed_local, sp_axis, tiled=True)
        owner_inv = jnp.argsort(jnp.asarray(token_ranks, jnp.int32))
        raw_rank_major = both_raw.reshape(num_ranks, rows, q_rank + 640)[
            owner_inv
        ].reshape(tokens, q_rank + 640)

    use_pallas_up = (
        tokens >= 256
        and tokens % 256 == 0
        and q_rank % 128 == 0
        and query_up_projection.shape == (q_rank, _HEADS * 256)
        and key_value_up_projection.shape == (512, _HEADS * 256)
    )
    up_tile = 512 if tokens % 512 == 0 else 256
    block_rows = num_ranks * stripe
    can_fuse_unstripe = use_pallas_up and (
        rows == stripe or (block_rows % up_tile == 0 and up_tile % stripe == 0)
    )
    fuse_prefix_gather = (
        prefix_gather_inputs is not None
        and can_fuse_unstripe
        and rows > stripe
        and tables is not None
    )
    if fuse_prefix_gather:
        (
            pool,
            prefix_ids,
            prefix_local,
            prefix_capacity,
            prefix_counts,
            prefix_order,
        ) = prefix_gather_inputs
        prefix_counts, prefix_order = pm.resolve_prefix_gather_meta(
            pool,
            prefix_ids,
            prefix_local,
            sp_axis,
            count=segments,
            capacity=prefix_capacity,
            counts=prefix_counts,
            order=prefix_order,
        )
        query, new_key, new_value, latent, pages = (
            mla_qkv_up_unstriped_and_prefix_gather_pallas(
                raw_rank_major,
                query_up_projection,
                key_value_up_projection,
                pool,
                tables,
                prefix_counts,
                prefix_ids,
                prefix_order,
                num_ranks=num_ranks,
                stripe=stripe,
                q_rank=q_rank,
                tile=up_tile,
                query_scale=query_scale,
                length=length,
                sp_axis=sp_axis,
                collective_id=prefix_gather_collective_id,
                return_padded_latent_640=return_padded_latent_640,
                interpret=interpret,
            )
        )
        arena = pm.expand_prefix_pages(pages, prefix_order, key_value_up_projection)
    else:
        latent_cols = 640 if return_padded_latent_640 else 576
        if can_fuse_unstripe:
            query, new_key, new_value, latent = mla_qkv_up_unstriped_pallas(
                raw_rank_major,
                query_up_projection,
                key_value_up_projection,
                num_ranks=num_ranks,
                stripe=stripe,
                q_rank=q_rank,
                tile=up_tile,
                query_scale=query_scale,
                length=length,
                return_padded_latent_640=return_padded_latent_640,
                interpret=interpret,
            )
        elif use_pallas_up:
            both_640 = unstripe_sp_window(raw_rank_major, num_ranks, stripe)
            query, new_key, new_value = mla_qkv_up_pallas(
                both_640,
                query_up_projection,
                key_value_up_projection,
                q_rank=q_rank,
                tile=up_tile,
                query_scale=query_scale,
                length=length,
                interpret=interpret,
            )
            latent = both_640[:, q_rank : q_rank + latent_cols]
        else:
            both_640 = unstripe_sp_window(raw_rank_major, num_ranks, stripe)
            q_lat = both_640[:, :q_rank]
            kv_lat = both_640[:, q_rank : q_rank + 512]
            rope_win = both_640[:, q_rank + 512 : q_rank + 576]
            real_1d = (
                (jnp.arange(tokens) < jnp.asarray(length).reshape(()))[:, None]
                if length is not None
                else None
            )
            rope_m = (
                jnp.where(real_1d, rope_win, 0) if real_1d is not None else rope_win
            )
            zeros_64 = jnp.zeros((tokens, 64), jnp.bfloat16)
            q_heads, k_heads, v_heads = [], [], []
            for h in range(_HEADS):
                sl = slice(h * 256, (h + 1) * 256)
                q_h = (_dot(q_lat, query_up_projection[:, sl]) * query_scale).astype(
                    jnp.bfloat16
                )
                kv_h = _dot(kv_lat, key_value_up_projection[:, sl]).astype(jnp.bfloat16)
                k_nope = (
                    jnp.where(real_1d, kv_h[:, :128], 0)
                    if real_1d is not None
                    else kv_h[:, :128]
                )
                v_h = (
                    jnp.where(real_1d, kv_h[:, 128:], 0)
                    if real_1d is not None
                    else kv_h[:, 128:]
                )
                q_heads.append(q_h)
                k_heads.append(jnp.concatenate((k_nope, rope_m, zeros_64), axis=1))
                v_heads.append(v_h)
            query = jnp.stack(q_heads, axis=0)
            new_key = jnp.stack(k_heads, axis=0)
            new_value = jnp.stack(v_heads, axis=0)
            latent = both_640[:, q_rank : q_rank + latent_cols]

        if prefix_gather_inputs is not None:
            (
                pool,
                prefix_ids,
                prefix_local,
                prefix_capacity,
                prefix_counts,
                prefix_order,
            ) = prefix_gather_inputs
            pages, prefix_order = pm.gathered_prefix_pages(
                pool,
                prefix_ids,
                prefix_local,
                tables,
                sp_axis,
                count=segments,
                capacity=prefix_capacity,
                counts=prefix_counts,
                order=prefix_order,
                collective_id=prefix_gather_collective_id,
                interpret=interpret,
            )
            arena = pm.expand_prefix_pages(pages, prefix_order, key_value_up_projection)
        elif prefix_gather_fn is not None:
            pages, order = prefix_gather_fn()
            arena = pm.expand_prefix_pages(pages, order, key_value_up_projection)
        else:
            arena = arena_override

    seg_fields = 3 if arena is None else 5
    cfg = k3_config.active()
    can_fuse_flash_a2a = (
        flash_fn is None
        and use_pallas_a2a
        and tables is not None
        and can_fuse_mla_flash_a2a_deep_prefix(
            tokens,
            num_ranks,
            stripe,
            tile_rows=flash_tile,
            arena=arena,
            mla_flash_qtile=cfg.mla_flash_qtile,
            mla_flash_ktile=cfg.mla_flash_ktile,
            mla_flash_prefix_tiles=cfg.mla_flash_prefix_tiles,
        )
    )
    if can_fuse_flash_a2a:
        owner_rows = mla_flash_prefill_packed_a2a_deep_prefix(
            tables,
            query,
            new_key,
            new_value,
            length,
            segment_desc[: seg_fields * segments],
            count=segments,
            num_ranks=num_ranks,
            stripe=stripe,
            tile=flash_tile,
            arena=arena,
            collective_id=a2a_collective_id,
            sp_axis=sp_axis,
            mla_flash_qtile=cfg.mla_flash_qtile,
            mla_flash_ktile=cfg.mla_flash_ktile,
            mla_flash_prefix_tiles=cfg.mla_flash_prefix_tiles,
            interpret=interpret,
        )
    else:
        if flash_fn is not None:
            run_flash = flash_fn
        else:
            run_flash = functools.partial(
                mla_flash_prefill_packed_deep_prefix,
                mla_flash_qtile=cfg.mla_flash_qtile,
                mla_flash_ktile=cfg.mla_flash_ktile,
                mla_flash_prefix_tiles=cfg.mla_flash_prefix_tiles,
                interpret=interpret,
            )
        attended_heads = run_flash(
            query,
            new_key,
            new_value,
            length,
            segment_desc[: seg_fields * segments],
            count=segments,
            tile=flash_tile,
            arena=arena,
        )
        attended = attended_heads.transpose(1, 0, 2).reshape(tokens, HEAD_COLS)

        if use_pallas_a2a and tables is not None:
            owner_rows = mla_attended_all_to_all_pallas(
                tables,
                attended,
                num_ranks=num_ranks,
                stripe=stripe,
                collective_id=a2a_collective_id,
            )
        else:
            owner_rows = mla_attended_all_to_all_ref(
                attended,
                token_ranks=token_ranks,
                num_ranks=num_ranks,
                stripe=stripe,
                sp_axis=sp_axis,
            )

    rank = jax.lax.axis_index(sp_axis)
    if tables is not None:
        token_owner = tables[0, 2, rank]
        own_even = tables[0, 2, rank & ~1]
        own_odd = tables[0, 2, rank | 1]
    else:
        owners_arr = jnp.asarray(token_ranks, jnp.int32)
        token_owner = owners_arr[rank]
        own_even = owners_arr[rank & ~1]
        own_odd = owners_arr[rank | 1]
    length_scalar = jnp.asarray(length, jnp.int32).reshape(())
    total_head_cols = num_ranks * HEAD_COLS
    is_chip_pair = m_gate_family.shape[-2:] == (
        width,
        total_head_cols // 2,
    ) and m_o_family.shape[-2:] == (total_head_cols, width // 2)

    if is_chip_pair:
        if m_gate_scales is not None or m_o_scales is not None:
            raise ValueError(
                "chip-pair MLA owner projections do not support FP8 scales"
            )
        if gate_both_precomputed is not None:
            after_attention = owner_mla_output_gated_chip_pair_pallas(
                owner_rows,
                gate_both_precomputed,
                m_o_family,
                prefix,
                layer=slot_scalar,
                length=length_scalar,
                own_even=own_even,
                own_odd=own_odd,
                stripe=stripe,
                num_ranks=num_ranks,
                interpret=interpret,
            )
        elif use_pallas_owner and num_ranks == RANKS:
            after_attention = owner_mla_gate_and_output_chip_pair_pallas(
                local_hidden,
                owner_rows,
                m_gate_family,
                m_o_family,
                prefix,
                layer=slot_scalar,
                length=length_scalar,
                own_even=own_even,
                own_odd=own_odd,
                stripe=stripe,
                interpret=interpret,
                sp_axis=sp_axis,
            )
        else:
            after_attention = owner_mla_gate_and_output_chip_pair_ref(
                local_hidden,
                owner_rows,
                m_gate_family[slot_scalar],
                m_o_family[slot_scalar],
                prefix,
                stripe=stripe,
                length=length_scalar,
                own_even=own_even,
                own_odd=own_odd,
                sp_axis=sp_axis,
            )
    elif use_pallas_owner and num_ranks == RANKS:
        after_attention = owner_mla_gate_and_output_pallas(
            local_hidden,
            owner_rows,
            m_gate_family,
            m_o_family,
            prefix,
            layer=slot_scalar,
            length=length_scalar,
            token_owner=token_owner,
            stripe=stripe,
            m_gate_scales=m_gate_scales,
            m_o_scales=m_o_scales,
            interpret=interpret,
        )
    else:
        after_attention = owner_mla_gate_and_output_ref(
            local_hidden,
            owner_rows,
            m_gate_family[slot_scalar],
            m_o_family[slot_scalar],
            prefix,
            stripe=stripe,
            length=length_scalar,
            token_owner=token_owner,
            m_gate_scale=None if m_gate_scales is None else m_gate_scales[slot_scalar],
            m_o_scale=None if m_o_scales is None else m_o_scales[slot_scalar],
        )
    return after_attention, latent


def make_indexed_mla_layer_sp_a2a(
    mesh,
    *,
    seq_len: int = 8192,
    segments: int,
    prefix_capacity: int,
    flash_tile: int = 256,
    token_ranks=None,
    collective_id: int = 31,
    gather_collective_id: int = 45,
    prefix_gather_collective_id: int = 44,
):
    if not segments:
        raise ValueError("the MLA layer runs packed windows")
    if mesh.size != RANKS or tuple(mesh.axis_names) != ("k3_sp",):
        raise ValueError("MLA (all-to-all) requires a flat k3_sp TP32 mesh")
    pm.check_window(seq_len, "MLA (all-to-all)")
    if token_ranks is None:
        token_ranks = tuple(range(RANKS))
    tables = jnp.asarray(
        pm.ring_tables(list(mesh.devices.flat), token_ranks), jnp.int32
    )

    def _run_local(
        local_hidden,
        prefix,
        m_qa,
        m_qb,
        m_ka,
        m_kb,
        m_qnorm,
        m_knorm,
        m_gate_fam,
        m_o_fam,
        segment_desc,
        length,
        slot,
        pool,
        ids,
        local,
        m_gate_scales=None,
        m_o_scales=None,
        counts=None,
        order=None,
        owned_pages=None,
    ):
        is_replicated = m_gate_fam.shape[-2:] == (
            HIDDEN,
            RANKS * HEAD_COLS,
        ) and m_o_fam.shape[-2:] == (RANKS * HEAD_COLS, HIDDEN)
        is_chip_pair = m_gate_fam.shape[-2:] == (
            HIDDEN,
            HALF_HEAD_COLS,
        ) and m_o_fam.shape[-2:] == (RANKS * HEAD_COLS, HALF_HIDDEN)
        if not (is_replicated or is_chip_pair):
            raise ValueError(
                "MLA (all-to-all) needs replicated ([slots, 7168, 12288], [slots, "
                "12288, 7168]) "
                "or chip-pair ([slots, 7168, 6144], [slots, 12288, 3584]) m_gate/m_o"
            )
        if is_chip_pair and (
            m_gate_fam.dtype != jnp.bfloat16
            or m_o_fam.dtype != jnp.bfloat16
            or m_gate_scales is not None
            or m_o_scales is not None
        ):
            raise ValueError(
                "chip-pair MLA m_gate/m_o requires BF16 weights without scales"
            )
        after_attention, latent = mla_local_normed_a2a_attention(
            local_hidden[0],
            prefix[0],
            query_down_projection=m_qa[0],
            query_up_projection=m_qb[0],
            key_value_down_projection=m_ka[0],
            key_value_up_projection=m_kb[0],
            query_normalization=m_qnorm[0],
            key_value_normalization=m_knorm[0],
            m_gate_family=m_gate_fam[0],
            m_o_family=m_o_fam[0],
            m_gate_scales=None if m_gate_scales is None else m_gate_scales[0],
            m_o_scales=None if m_o_scales is None else m_o_scales[0],
            slot=slot,
            length=length,
            segment_desc=segment_desc,
            num_ranks=RANKS,
            sp_axis="k3_sp",
            segments=segments,
            flash_tile=flash_tile,
            token_ranks=token_ranks,
            tables=tables,
            a2a_collective_id=collective_id,
            gather_collective_id=gather_collective_id,
            prefix_gather_collective_id=prefix_gather_collective_id,
            prefix_gather_inputs=(
                pool,
                ids[0],
                local[0],
                prefix_capacity,
                counts,
                order,
            ),
            return_padded_latent_640=(owned_pages is not None),
        )
        latent_out = (
            pm.compact_owned_mla_pages(
                latent,
                owned_pages[0],
                block_shape=pool.shape[-3:],
                pool_dtype=pool.dtype,
            )
            if owned_pages is not None
            else latent
        )
        return after_attention[None], latent_out[None]

    def local_packed_cp(
        local_hidden,
        prefix,
        m_qa,
        m_qb,
        m_ka,
        m_kb,
        m_qnorm,
        m_knorm,
        m_gate_fam,
        m_o_fam,
        segment_desc,
        length,
        slot,
        pool,
        ids,
        local,
        counts=None,
        order=None,
        owned_pages=None,
    ):
        return _run_local(
            local_hidden,
            prefix,
            m_qa,
            m_qb,
            m_ka,
            m_kb,
            m_qnorm,
            m_knorm,
            m_gate_fam,
            m_o_fam,
            segment_desc,
            length,
            slot,
            pool,
            ids,
            local,
            counts=counts,
            order=order,
            owned_pages=owned_pages,
        )

    def local_packed_cp_fp8(
        local_hidden,
        prefix,
        m_qa,
        m_qb,
        m_ka,
        m_kb,
        m_qnorm,
        m_knorm,
        m_gate_fam,
        m_o_fam,
        m_gate_scale_fam,
        m_o_scale_fam,
        segment_desc,
        length,
        slot,
        pool,
        ids,
        local,
        counts=None,
        order=None,
        owned_pages=None,
    ):
        return _run_local(
            local_hidden,
            prefix,
            m_qa,
            m_qb,
            m_ka,
            m_kb,
            m_qnorm,
            m_knorm,
            m_gate_fam,
            m_o_fam,
            segment_desc,
            length,
            slot,
            pool,
            ids,
            local,
            m_gate_scales=m_gate_scale_fam,
            m_o_scales=m_o_scale_fam,
            counts=counts,
            order=order,
            owned_pages=owned_pages,
        )

    def run(local_hidden, prefix, families, segment_desc, length, slot, arena):
        has_gate_scale = "m_gate_scale" in families
        has_o_scale = "m_o_scale" in families
        if has_gate_scale != has_o_scale:
            raise ValueError(
                "MLA (all-to-all) requires both m_gate_scale and m_o_scale or neither"
            )
        idx = slot[0]
        m_qa = jax.lax.dynamic_index_in_dim(
            families["m_qa"], idx, axis=1, keepdims=False
        )
        m_qb = jax.lax.dynamic_index_in_dim(
            families["m_qb"], idx, axis=1, keepdims=False
        )
        m_ka = jax.lax.dynamic_index_in_dim(
            families["m_ka"], idx, axis=1, keepdims=False
        )
        m_kb = jax.lax.dynamic_index_in_dim(
            families["m_kb"], idx, axis=1, keepdims=False
        )
        m_qnorm = jax.lax.dynamic_index_in_dim(
            families["m_qnorm"], idx, axis=1, keepdims=False
        )
        m_knorm = jax.lax.dynamic_index_in_dim(
            families["m_knorm"], idx, axis=1, keepdims=False
        )
        (pool,) = arena.pools
        weight_args = (
            (
                families["m_gate"],
                families["m_o"],
                families["m_gate_scale"],
                families["m_o_scale"],
            )
            if has_gate_scale
            else (families["m_gate"], families["m_o"])
        )
        fn_local = local_packed_cp_fp8 if has_gate_scale else local_packed_cp
        base_specs = (P("k3_sp"),) * (8 + len(weight_args)) + (
            P(),
            P(),
            P(),
            P("k3_sp"),
            P("k3_sp"),
            P("k3_sp"),
        )
        counts, order = arena.counts, arena.order
        if arena.owned_pages is not None and (counts is None or order is None):
            counts, order = pm.cp_prefix_meta(
                mesh,
                pool,
                arena.ids,
                arena.local,
                segments=segments,
                capacity=prefix_capacity,
            )
        has_meta = counts is not None and order is not None
        has_owned = arena.owned_pages is not None
        extra_args = ((counts, order) if has_meta else ()) + (
            (arena.owned_pages,) if has_owned else ()
        )
        extra_specs = ((P(), P()) if has_meta else ()) + (
            (P("k3_sp"),) if has_owned else ()
        )
        return jax.shard_map(
            fn_local,
            mesh=mesh,
            in_specs=base_specs + extra_specs,
            out_specs=(P("k3_sp"), P("k3_sp")),
            check_vma=False,
        )(
            local_hidden,
            prefix,
            m_qa,
            m_qb,
            m_ka,
            m_kb,
            m_qnorm,
            m_knorm,
            *weight_args,
            segment_desc,
            length,
            slot,
            pool,
            arena.ids,
            arena.local,
            *extra_args,
        )

    return jax.jit(run)


def gather_gate_shares(share, *, axis: str):
    return (jax.lax.all_gather(share, axis, axis=3, tiled=True),)


def gather_output_projection_shares(share, *, axis: str):
    return (jax.lax.all_gather(share, axis, axis=2, tiled=True),)


def gather_gate_shares_fp8(share, *, axis: str):
    share_f32 = share.astype(jnp.float32)
    col_max = jnp.maximum(
        jnp.max(jnp.abs(share_f32), axis=2, keepdims=True), _FP8_SCALE_EPS
    )
    q_share = (share_f32 * (_FP8_E4M3_MAX / col_max)).astype(jnp.float8_e4m3fn)
    full_fp8 = jax.lax.all_gather(q_share, axis, axis=3, tiled=True)
    local_scale_8 = jnp.broadcast_to(
        col_max * (1.0 / _FP8_E4M3_MAX), (*share.shape[:2], 8, share.shape[3])
    )
    full_scale_8 = jax.lax.all_gather(local_scale_8, axis, axis=3, tiled=True)
    return full_fp8, full_scale_8


def gather_output_projection_shares_fp8(share, *, axis: str):
    share_f32 = share.astype(jnp.float32)
    global_max = jnp.maximum(
        jax.lax.pmax(jnp.max(jnp.abs(share_f32), axis=2, keepdims=True), axis),
        _FP8_SCALE_EPS,
    )
    q_share = (share_f32 * (_FP8_E4M3_MAX / global_max)).astype(jnp.float8_e4m3fn)
    full_fp8 = jax.lax.all_gather(q_share, axis, axis=2, tiled=True)
    scale_8 = jnp.broadcast_to(
        global_max * (1.0 / _FP8_E4M3_MAX), (*share.shape[:2], 8, share.shape[3])
    )
    return full_fp8, scale_8


def _sync_partner(sem_ref, partner):
    pl.semaphore_signal(
        sem_ref, 1, device_id=(partner,), device_id_type=pl.DeviceIdType.MESH
    )
    pl.semaphore_wait(sem_ref, 1)


def _partner_enter(partner, ready_sem):
    barrier = pltpu.get_barrier_semaphore()
    pl.semaphore_signal(
        barrier, 1, device_id=(partner,), device_id_type=pl.DeviceIdType.MESH
    )
    pl.semaphore_wait(barrier, 1)
    _sync_partner(ready_sem, partner)


def owner_mla_gate_and_output_chip_pair_ref(
    local_normed,
    owner_rows,
    m_gate_half,
    m_o_half,
    prefix,
    *,
    stripe: int,
    length,
    own_even,
    own_odd,
    sp_axis: str = "k3_sp",
):
    rows, width = local_normed.shape
    num_ranks, owner_len, cols = owner_rows.shape
    total_cols = num_ranks * cols
    half_cols = total_cols // 2
    half_w = width // 2
    if (
        num_ranks % 2
        or owner_len != rows
        or width % 2
        or m_gate_half.shape != (width, half_cols)
        or m_o_half.shape != (total_cols, half_w)
        or prefix.shape != (rows, width)
        or m_gate_half.dtype != jnp.bfloat16
        or m_o_half.dtype != jnp.bfloat16
    ):
        raise ValueError(
            "owner_mla_gate_and_output_chip_pair_ref expects even R, "
            "m_gate_half=[W, R*cols//2] bf16, and m_o_half=[R*cols, W//2] bf16"
        )
    swap_pairs = [(i, i ^ 1) for i in range(num_ranks)]
    parity = jax.lax.axis_index(sp_axis) & 1

    partner_normed = jax.lax.ppermute(local_normed, sp_axis, swap_pairs)
    even_normed = jnp.where(parity == 0, local_normed, partner_normed)
    odd_normed = jnp.where(parity == 0, partner_normed, local_normed)

    half_ranks = num_ranks // 2
    own_sub = jax.lax.dynamic_slice_in_dim(
        owner_rows, parity * half_ranks, half_ranks, axis=0
    )
    send_sub = jax.lax.dynamic_slice_in_dim(
        owner_rows, (1 - parity) * half_ranks, half_ranks, axis=0
    )
    recv_sub = jax.lax.ppermute(send_sub, sp_axis, swap_pairs)
    even_heads = (
        jnp.where(parity == 0, own_sub, recv_sub)
        .transpose(1, 0, 2)
        .reshape(rows, half_cols)
    )
    odd_heads = (
        jnp.where(parity == 0, recv_sub, own_sub)
        .transpose(1, 0, 2)
        .reshape(rows, half_cols)
    )

    even_logits = _dot(even_normed, m_gate_half).astype(jnp.bfloat16)
    even_gate = jax.nn.sigmoid(even_logits.astype(jnp.float32)).astype(jnp.bfloat16)
    even_gated_half = (even_heads * even_gate).astype(jnp.bfloat16)

    odd_logits = _dot(odd_normed, m_gate_half).astype(jnp.bfloat16)
    odd_gate = jax.nn.sigmoid(odd_logits.astype(jnp.float32)).astype(jnp.bfloat16)
    odd_gated_half = (odd_heads * odd_gate).astype(jnp.bfloat16)

    my_gated_pair = jnp.stack((even_gated_half, odd_gated_half), axis=0)
    partner_gated_pair = jax.lax.ppermute(my_gated_pair, sp_axis, swap_pairs)
    left_gated = jnp.where(parity == 0, my_gated_pair, partner_gated_pair)
    right_gated = jnp.where(parity == 0, partner_gated_pair, my_gated_pair)
    gated_both = jnp.concatenate((left_gated, right_gated), axis=2)

    proj_even = _dot(gated_both[0], m_o_half)
    proj_odd = _dot(gated_both[1], m_o_half)

    own_px = jax.lax.dynamic_slice_in_dim(prefix, parity * half_w, half_w, axis=1)
    send_px = jax.lax.dynamic_slice_in_dim(
        prefix, (1 - parity) * half_w, half_w, axis=1
    )
    recv_px = jax.lax.ppermute(send_px, sp_axis, swap_pairs)
    even_px = jnp.where(parity == 0, own_px, recv_px)
    odd_px = jnp.where(parity == 0, recv_px, own_px)

    local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    len_scalar = jnp.asarray(length, jnp.int32).reshape(())
    valid_even = _sp_token_index_ranks(own_even, local, stripe, num_ranks) < len_scalar
    valid_odd = _sp_token_index_ranks(own_odd, local, stripe, num_ranks) < len_scalar
    out_even = (
        even_px.astype(jnp.float32) + jnp.where(valid_even, proj_even, 0.0)
    ).astype(prefix.dtype)
    out_odd = (odd_px.astype(jnp.float32) + jnp.where(valid_odd, proj_odd, 0.0)).astype(
        prefix.dtype
    )

    own_out = jnp.where(parity == 0, out_even, out_odd)
    for_partner = jnp.where(parity == 0, out_odd, out_even)
    from_partner = jax.lax.ppermute(for_partner, sp_axis, swap_pairs)
    left_out = jnp.where(parity == 0, own_out, from_partner)
    right_out = jnp.where(parity == 0, from_partner, own_out)
    return jnp.concatenate((left_out, right_out), axis=1)


def _mla_owner_gate_chip_pair_kernel(
    scalars_ref,
    normed_hbm,
    owner_rows_hbm,
    m_gate_half_hbm,
    gated_both_hbm,
    both_normed_vmem,
    both_rows_bufs,
    w_bufs,
    gated_half_bufs,
    local_sems,
    send_sems,
    recv_sems,
    sync_sems,
):
    rank = jax.lax.axis_index("k3_sp")
    parity = rank & 1
    partner = rank ^ 1
    group = OWNER_GATE_SOURCES
    steps = (RANKS // 2) // group
    layer = scalars_ref[0]
    src_rank_base = parity * (RANKS // 2)

    _partner_enter(partner, sync_sems.at[0])

    n_local = pltpu.make_async_copy(
        normed_hbm, both_normed_vmem.at[parity], local_sems.at[4]
    )
    n_remote = pltpu.make_async_remote_copy(
        normed_hbm,
        both_normed_vmem.at[parity],
        send_sems.at[2],
        recv_sems.at[2],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    n_local.start()
    n_remote.start()

    def start_step(s):
        slot = s % 2
        own_src = owner_rows_hbm.at[pl.ds(src_rank_base + s * group, group), :, :]
        send_src = owner_rows_hbm.at[
            pl.ds((1 - parity) * (RANKS // 2) + s * group, group), :, :
        ]
        pltpu.make_async_copy(
            own_src, both_rows_bufs.at[slot, parity], local_sems.at[slot]
        ).start()
        pltpu.make_async_remote_copy(
            send_src,
            both_rows_bufs.at[slot, parity],
            send_sems.at[slot],
            recv_sems.at[slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
        pltpu.make_async_copy(
            m_gate_half_hbm.at[
                layer, :, pl.ds(s * group * HEAD_COLS, group * HEAD_COLS)
            ],
            w_bufs.at[slot],
            local_sems.at[2 + slot],
        ).start()

    def wait_step(s):
        slot = s % 2
        pltpu.make_async_copy(
            both_rows_bufs.at[slot, parity],
            both_rows_bufs.at[slot, parity],
            local_sems.at[slot],
        ).wait()
        send_src = owner_rows_hbm.at[
            pl.ds((1 - parity) * (RANKS // 2) + s * group, group), :, :
        ]
        pltpu.make_async_remote_copy(
            send_src,
            both_rows_bufs.at[slot, parity],
            send_sems.at[slot],
            recv_sems.at[slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).wait_send()
        pltpu.make_async_copy(
            both_rows_bufs.at[slot, 1 - parity],
            both_rows_bufs.at[slot, 1 - parity],
            recv_sems.at[slot],
        ).wait()
        pltpu.make_async_copy(
            w_bufs.at[slot], w_bufs.at[slot], local_sems.at[2 + slot]
        ).wait()

    start_step(0)
    start_step(1)
    n_local.wait()
    n_remote.wait_send()
    pltpu.make_async_copy(
        both_normed_vmem.at[1 - parity],
        both_normed_vmem.at[1 - parity],
        recv_sems.at[2],
    ).wait()

    @pl.loop(0, steps)
    def run_step(s):
        slot = s % 2
        wait_step(s)
        even_heads = jnp.concatenate(
            [both_rows_bufs[slot, 0, j] for j in range(group)], axis=1
        )
        odd_heads = jnp.concatenate(
            [both_rows_bufs[slot, 1, j] for j in range(group)], axis=1
        )
        w = w_bufs[slot][...]

        even_logits = jnp.dot(
            both_normed_vmem[0], w, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        even_gate = jax.nn.sigmoid(even_logits.astype(jnp.float32)).astype(jnp.bfloat16)
        even_gated = (even_heads * even_gate).astype(jnp.bfloat16)

        odd_logits = jnp.dot(
            both_normed_vmem[1], w, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        odd_gate = jax.nn.sigmoid(odd_logits.astype(jnp.float32)).astype(jnp.bfloat16)
        odd_gated = (odd_heads * odd_gate).astype(jnp.bfloat16)

        @pl.when(s >= 2)
        def wait_prev_out_send():
            prev_col = (parity * HALF_HEAD_COLS) + (s - 2) * group * HEAD_COLS
            pltpu.make_async_copy(
                gated_half_bufs.at[slot],
                gated_both_hbm.at[:, :, pl.ds(prev_col, group * HEAD_COLS)],
                local_sems.at[5 + slot],
            ).wait()
            pltpu.make_async_remote_copy(
                gated_half_bufs.at[slot],
                gated_both_hbm.at[:, :, pl.ds(prev_col, group * HEAD_COLS)],
                send_sems.at[3 + slot],
                recv_sems.at[3 + slot],
                device_id=(partner,),
                device_id_type=pl.DeviceIdType.MESH,
            ).wait_send()
            partner_prev_col = ((1 - parity) * HALF_HEAD_COLS) + (
                s - 2
            ) * group * HEAD_COLS
            pltpu.make_async_copy(
                gated_both_hbm.at[:, :, pl.ds(partner_prev_col, group * HEAD_COLS)],
                gated_both_hbm.at[:, :, pl.ds(partner_prev_col, group * HEAD_COLS)],
                recv_sems.at[3 + slot],
            ).wait()

        gated_half_bufs[slot, 0, ...] = even_gated
        gated_half_bufs[slot, 1, ...] = odd_gated
        dst_col = (parity * HALF_HEAD_COLS) + s * group * HEAD_COLS
        pltpu.make_async_copy(
            gated_half_bufs.at[slot],
            gated_both_hbm.at[:, :, pl.ds(dst_col, group * HEAD_COLS)],
            local_sems.at[5 + slot],
        ).start()
        pltpu.make_async_remote_copy(
            gated_half_bufs.at[slot],
            gated_both_hbm.at[:, :, pl.ds(dst_col, group * HEAD_COLS)],
            send_sems.at[3 + slot],
            recv_sems.at[3 + slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()

        @pl.when(s + 2 < steps)
        def prefetch_next():
            _sync_partner(sync_sems.at[1 + slot], partner)
            start_step(s + 2)

    for tail_s in (steps - 2, steps - 1):
        slot = tail_s % 2
        dst_col = (parity * HALF_HEAD_COLS) + tail_s * group * HEAD_COLS
        pltpu.make_async_copy(
            gated_half_bufs.at[slot],
            gated_both_hbm.at[:, :, pl.ds(dst_col, group * HEAD_COLS)],
            local_sems.at[5 + slot],
        ).wait()
        pltpu.make_async_remote_copy(
            gated_half_bufs.at[slot],
            gated_both_hbm.at[:, :, pl.ds(dst_col, group * HEAD_COLS)],
            send_sems.at[3 + slot],
            recv_sems.at[3 + slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).wait_send()
        partner_col = ((1 - parity) * HALF_HEAD_COLS) + tail_s * group * HEAD_COLS
        pltpu.make_async_copy(
            gated_both_hbm.at[:, :, pl.ds(partner_col, group * HEAD_COLS)],
            gated_both_hbm.at[:, :, pl.ds(partner_col, group * HEAD_COLS)],
            recv_sems.at[3 + slot],
        ).wait()

    _sync_partner(sync_sems.at[3], partner)


def _mla_owner_out_chip_pair_kernel(
    scalars_ref,
    gated_both_hbm,
    m_o_half_hbm,
    prefix_hbm,
    out_hbm,
    g_bufs,
    w_bufs,
    acc_even_vmem,
    acc_odd_vmem,
    prefix_half_vmem,
    out_half_vmem,
    local_sems,
    send_sems,
    recv_sems,
    sync_sems,
    *,
    stripe: int,
):
    rank = jax.lax.axis_index("k3_sp")
    parity = rank & 1
    partner = rank ^ 1
    rows = gated_both_hbm.shape[1]
    group = OWNER_OUT_SOURCES
    steps = RANKS // group
    layer = scalars_ref[0]

    _partner_enter(partner, sync_sems.at[0])

    own_px = prefix_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)]
    send_px = prefix_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)]
    px_local = pltpu.make_async_copy(
        own_px, prefix_half_vmem.at[parity], local_sems.at[4]
    )
    px_remote = pltpu.make_async_remote_copy(
        send_px,
        prefix_half_vmem.at[parity],
        send_sems.at[0],
        recv_sems.at[0],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    px_local.start()
    px_remote.start()

    def start_step(s):
        slot = s % 2
        pltpu.make_async_copy(
            gated_both_hbm.at[:, :, pl.ds(s * group * HEAD_COLS, group * HEAD_COLS)],
            g_bufs.at[slot],
            local_sems.at[slot],
        ).start()
        pltpu.make_async_copy(
            m_o_half_hbm.at[layer, pl.ds(s * group * HEAD_COLS, group * HEAD_COLS), :],
            w_bufs.at[slot],
            local_sems.at[2 + slot],
        ).start()

    def wait_step(s):
        slot = s % 2
        pltpu.make_async_copy(
            g_bufs.at[slot], g_bufs.at[slot], local_sems.at[slot]
        ).wait()
        pltpu.make_async_copy(
            w_bufs.at[slot], w_bufs.at[slot], local_sems.at[2 + slot]
        ).wait()

    start_step(0)
    start_step(1)
    acc_even_vmem[...] = jnp.zeros((rows, HALF_HIDDEN), jnp.float32)
    acc_odd_vmem[...] = jnp.zeros((rows, HALF_HIDDEN), jnp.float32)

    @pl.loop(0, steps)
    def run_step(s):
        slot = s % 2
        wait_step(s)
        w = w_bufs[slot][...]
        acc_even_vmem[...] += jnp.dot(
            g_bufs[slot, 0], w, preferred_element_type=jnp.float32
        )
        acc_odd_vmem[...] += jnp.dot(
            g_bufs[slot, 1], w, preferred_element_type=jnp.float32
        )

        @pl.when(s + 2 < steps)
        def prefetch_next():
            start_step(s + 2)

    px_local.wait()
    px_remote.wait_send()
    pltpu.make_async_copy(
        prefix_half_vmem.at[1 - parity],
        prefix_half_vmem.at[1 - parity],
        recv_sems.at[0],
    ).wait()

    length, own_even, own_odd = scalars_ref[1], scalars_ref[2], scalars_ref[3]
    local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    valid_even = pm.sp_token_index(own_even, local, stripe) < length
    valid_odd = pm.sp_token_index(own_odd, local, stripe) < length
    out_half_vmem[0, ...] = (
        prefix_half_vmem[0].astype(jnp.float32)
        + jnp.where(valid_even, acc_even_vmem[...], 0.0)
    ).astype(out_hbm.dtype)
    out_half_vmem[1, ...] = (
        prefix_half_vmem[1].astype(jnp.float32)
        + jnp.where(valid_odd, acc_odd_vmem[...], 0.0)
    ).astype(out_hbm.dtype)

    own_out = out_half_vmem.at[parity]
    partner_out = out_half_vmem.at[1 - parity]
    out_local = pltpu.make_async_copy(
        own_out,
        out_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)],
        local_sems.at[5],
    )
    out_remote = pltpu.make_async_remote_copy(
        partner_out,
        out_hbm.at[:, pl.ds(parity * HALF_HIDDEN, HALF_HIDDEN)],
        send_sems.at[1],
        recv_sems.at[1],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    out_local.start()
    out_remote.start()
    out_local.wait()
    out_remote.wait_send()
    pltpu.make_async_copy(
        out_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)],
        out_hbm.at[:, pl.ds((1 - parity) * HALF_HIDDEN, HALF_HIDDEN)],
        recv_sems.at[1],
    ).wait()
    _sync_partner(sync_sems.at[1], partner)


def owner_mla_gate_and_output_chip_pair_pallas(
    local_normed,
    owner_rows,
    m_gate_half,
    m_o_half,
    prefix,
    *,
    layer,
    length,
    own_even,
    own_odd,
    stripe: int,
    gate_collective_id: int = 56,
    out_collective_id: int = 57,
    interpret: bool = False,
    sp_axis: str = "k3_sp",
):
    rows, width = local_normed.shape
    ranks, owner_len, cols = owner_rows.shape
    if (
        ranks != RANKS
        or cols != HEAD_COLS
        or owner_len != rows
        or width != HIDDEN
        or prefix.shape != (rows, HIDDEN)
        or m_gate_half.ndim != 3
        or m_o_half.ndim != 3
        or m_gate_half.shape[-2:] != (HIDDEN, HALF_HEAD_COLS)
        or m_o_half.shape[-2:] != (TOTAL_HEAD_COLS, HALF_HIDDEN)
        or m_gate_half.dtype != jnp.bfloat16
        or m_o_half.dtype != jnp.bfloat16
    ):
        raise ValueError(
            "chip-pair MLA projection expects owner_rows=[32, rows, 384], "
            "m_gate_half=[slots, 7168, 6144] bf16, and m_o_half=[slots, 12288, "
            "3584] bf16"
        )
    if interpret:
        slot_idx = jnp.asarray(layer, jnp.int32).reshape(())
        return owner_mla_gate_and_output_chip_pair_ref(
            local_normed,
            owner_rows,
            m_gate_half[slot_idx],
            m_o_half[slot_idx],
            prefix,
            stripe=stripe,
            length=length,
            own_even=own_even,
            own_odd=own_odd,
            sp_axis=sp_axis,
        )
    scalars = jnp.stack(
        [
            jnp.asarray(v, jnp.int32).reshape(())
            for v in (layer, length, own_even, own_odd)
        ]
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    gated_both = pl.pallas_call(
        _mla_owner_gate_chip_pair_kernel,
        out_shape=jax.ShapeDtypeStruct((2, rows, TOTAL_HEAD_COLS), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[hbm, hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((2, rows, HIDDEN), jnp.bfloat16),
                pltpu.VMEM((2, 2, OWNER_GATE_SOURCES, rows, HEAD_COLS), jnp.bfloat16),
                pltpu.VMEM((2, HIDDEN, OWNER_GATE_SOURCES * HEAD_COLS), jnp.bfloat16),
                pltpu.VMEM((2, 2, rows, OWNER_GATE_SOURCES * HEAD_COLS), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((7,)),
                pltpu.SemaphoreType.DMA((5,)),
                pltpu.SemaphoreType.DMA((5,)),
                pltpu.SemaphoreType.REGULAR((4,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=gate_collective_id, vmem_limit_bytes=64 * 1024**2
        ),
        name="kimi_prefill_mla_owner_gate_chip_pair",
    )(scalars, local_normed, owner_rows, m_gate_half)

    return pl.pallas_call(
        functools.partial(_mla_owner_out_chip_pair_kernel, stripe=stripe),
        out_shape=jax.ShapeDtypeStruct((rows, HIDDEN), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[hbm, hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((2, 2, rows, OWNER_OUT_SOURCES * HEAD_COLS), jnp.bfloat16),
                pltpu.VMEM(
                    (2, OWNER_OUT_SOURCES * HEAD_COLS, HALF_HIDDEN), jnp.bfloat16
                ),
                pltpu.VMEM((rows, HALF_HIDDEN), jnp.float32),
                pltpu.VMEM((rows, HALF_HIDDEN), jnp.float32),
                pltpu.VMEM((2, rows, HALF_HIDDEN), jnp.bfloat16),
                pltpu.VMEM((2, rows, HALF_HIDDEN), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((6,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.REGULAR((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=out_collective_id, vmem_limit_bytes=64 * 1024**2
        ),
        name="kimi_prefill_mla_owner_projection_chip_pair",
    )(scalars, gated_both, m_o_half, prefix)


def gather_gate_shares_chip_pair(share, *, axis: str):
    if share.dtype != jnp.bfloat16:
        raise ValueError("chip-pair MLA m_gate gather requires BF16 shares")
    full = jax.lax.all_gather(share, axis, axis=3, tiled=True)
    if full.shape[3] % 2:
        raise ValueError("chip-pair MLA m_gate gather requires even total head columns")
    half_cols = full.shape[3] // 2
    parity = jax.lax.axis_index(axis) & 1
    return (jax.lax.dynamic_slice_in_dim(full, parity * half_cols, half_cols, axis=3),)


def gather_output_projection_shares_chip_pair(share, *, axis: str):
    if share.dtype != jnp.bfloat16 or share.shape[3] % 2:
        raise ValueError(
            "chip-pair MLA m_o gather requires BF16 shares with even hidden width"
        )
    half_w = share.shape[3] // 2
    parity = jax.lax.axis_index(axis) & 1
    even_cols = jax.lax.all_gather(share[..., :half_w], axis, axis=2, tiled=True)
    odd_cols = jax.lax.all_gather(share[..., half_w:], axis, axis=2, tiled=True)
    return (jnp.where(parity == 0, even_cols, odd_cols),)


def can_fuse_mla_ag_and_gate_chip_pair(
    *,
    tables,
    sp_axis: str,
    num_ranks: int,
    rows: int,
    width: int,
    packed_width: int,
    m_gate_family,
    m_o_family,
    m_gate_scales,
    m_o_scales,
    use_pallas_owner: bool,
    interpret: bool | pltpu.InterpretParams,
) -> bool:
    if (
        tables is None
        or sp_axis != "k3_sp"
        or not use_pallas_owner
        or interpret is True
        or num_ranks < 4
        or num_ranks % 4 != 0
        or tables.shape != (3, 3, num_ranks)
        or rows < 8
        or rows % 8 != 0
        or width < 256
        or width % 256 != 0
        or packed_width < 384
        or packed_width % 128 != 0
    ):
        return False
    total_head_cols = num_ranks * HEAD_COLS
    return (
        m_gate_scales is None
        and m_o_scales is None
        and m_gate_family.ndim == 3
        and m_o_family.ndim == 3
        and m_gate_family.dtype == jnp.bfloat16
        and m_o_family.dtype == jnp.bfloat16
        and m_gate_family.shape[-2:] == (width, total_head_cols // 2)
        and m_o_family.shape[-2:] == (total_head_cols, width // 2)
    )


def _ring_enter_ranks(tables_ref, rank, *, num_ranks: int):
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    peers = [
        tables_ref[f, 1, (positions[f] + d) % num_ranks]
        for f in range(3)
        for d in (-1, 1)
    ]
    barrier = pltpu.get_barrier_semaphore()
    for peer in peers:
        pl.semaphore_signal(
            barrier, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
        )
    pl.semaphore_wait(barrier, len(peers))

    @functools.partial(pl.run_scoped, ready=pltpu.SemaphoreType.REGULAR)
    def _(ready):
        for peer in peers:
            pl.semaphore_signal(
                ready, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(ready, len(peers))


def _rank_half_gather_and_mla_gate_chip_pair_kernel(
    tables_ref,
    scalars_ref,
    local_ref,
    normed_hbm,
    m_gate_half_hbm,
    window_ref,
    gate_both_hbm,
    send_sems,
    recv_sems,
    own_sem,
    done,
    both_normed_vmem,
    w_gate_bufs,
    gate_half_bufs,
    g_local_sems,
    g_send_sems,
    g_recv_sems,
    partner_sems,
    *,
    num_ranks: int,
    gate_cols: int,
    grid_steps: int,
):
    step = pl.program_id(1)
    rank = jax.lax.axis_index("k3_sp")
    parity = rank & 1
    partner = rank ^ 1
    rows = local_ref.shape[0]
    half_head_cols = (num_ranks * HEAD_COLS) // 2
    gate_steps = half_head_cols // gate_cols
    half_hops = {1: num_ranks // 2, -1: num_ranks // 2 - 1}
    layer = scalars_ref[0]
    positions = [tables_ref[f, 0, rank] for f in range(3)]
    neighbours = [
        tables_ref[f, 1, (positions[f] + d) % num_ranks]
        for f in range(3)
        for d in (-1, 1)
    ]

    def owner_rows_slice(family, position):
        owner = tables_ref[family, 2, tables_ref[family, 1, position % num_ranks]]
        return pl.ds(owner * rows, rows)

    def start_w_gate(g_s):
        slot = g_s % 2
        pltpu.make_async_copy(
            m_gate_half_hbm.at[layer, :, pl.ds(g_s * gate_cols, gate_cols)],
            w_gate_bufs.at[slot],
            g_local_sems.at[slot],
        ).start()

    def wait_gate_out(g_s):
        slot = g_s % 2
        dst_col = parity * half_head_cols + g_s * gate_cols
        part_col = (1 - parity) * half_head_cols + g_s * gate_cols
        pltpu.make_async_copy(
            gate_half_bufs.at[slot],
            gate_both_hbm.at[:, :, pl.ds(dst_col, gate_cols)],
            g_local_sems.at[3 + slot],
        ).wait()
        pltpu.make_async_remote_copy(
            gate_half_bufs.at[slot],
            gate_both_hbm.at[:, :, pl.ds(dst_col, gate_cols)],
            g_send_sems.at[1 + slot],
            g_recv_sems.at[1 + slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).wait_send()
        pltpu.make_async_copy(
            gate_both_hbm.at[:, :, pl.ds(part_col, gate_cols)],
            gate_both_hbm.at[:, :, pl.ds(part_col, gate_cols)],
            g_recv_sems.at[1 + slot],
        ).wait()

    @pl.when(step == 0)
    def enter_and_own_rows():
        _ring_enter_ranks(tables_ref, rank, num_ranks=num_ranks)
        _sync_partner(partner_sems.at[0], partner)
        pltpu.make_async_copy(
            local_ref, window_ref.at[owner_rows_slice(0, positions[0])], own_sem
        ).start()

    for stream, (family, direction, col, col_w) in enumerate(
        _half_ring_streams(local_ref.shape[1])
    ):
        cols = pl.ds(col, col_w)
        hops = half_hops[direction]
        position = positions[family]
        target = (tables_ref[family, 1, (position + direction) % num_ranks],)

        def send(
            index,
            source,
            family=family,
            direction=direction,
            cols=cols,
            position=position,
            target=target,
            stream=stream,
        ):
            r_sl = owner_rows_slice(family, position - direction * index)
            return pltpu.make_async_remote_copy(
                source if source is not None else window_ref.at[r_sl, cols],
                window_ref.at[r_sl, cols],
                send_sems.at[stream],
                recv_sems.at[stream, index],
                device_id=target,
                device_id_type=pl.DeviceIdType.MESH,
            )

        own = local_ref.at[:, cols]
        pl.when(step == 0)(lambda send=send, own=own: send(0, own).start())

        @pl.when((step >= 1) & (step <= hops))
        def arrived(send=send):
            copy = send(step - 1, None)
            copy.wait_recv()
            copy.wait_send()

        @pl.when((step >= 1) & (step < hops))
        def forward(send=send):
            send(step, None).start()

    @pl.when(step == 0)
    def start_gate_inputs():
        pltpu.make_async_copy(
            normed_hbm, both_normed_vmem.at[parity], g_local_sems.at[2]
        ).start()
        pltpu.make_async_remote_copy(
            normed_hbm,
            both_normed_vmem.at[parity],
            g_send_sems.at[0],
            g_recv_sems.at[0],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()
        start_w_gate(0)
        start_w_gate(1)

    @pl.when(step == 1)
    def wait_normed():
        pltpu.make_async_copy(
            both_normed_vmem.at[parity],
            both_normed_vmem.at[parity],
            g_local_sems.at[2],
        ).wait()
        pltpu.make_async_remote_copy(
            normed_hbm,
            both_normed_vmem.at[parity],
            g_send_sems.at[0],
            g_recv_sems.at[0],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).wait_send()
        pltpu.make_async_copy(
            both_normed_vmem.at[1 - parity],
            both_normed_vmem.at[1 - parity],
            g_recv_sems.at[0],
        ).wait()

    @pl.when((step >= 1) & (step <= gate_steps))
    def compute_gate_step():
        g_s = step - 1
        slot = g_s % 2
        pltpu.make_async_copy(
            w_gate_bufs.at[slot], w_gate_bufs.at[slot], g_local_sems.at[slot]
        ).wait()

        @pl.when(g_s >= 2)
        def wait_prev_gate_store():
            wait_gate_out(g_s - 2)

        w_g = w_gate_bufs[slot][...]
        even_logits = jnp.dot(
            both_normed_vmem[0], w_g, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        even_gate = jax.nn.sigmoid(even_logits.astype(jnp.float32)).astype(jnp.bfloat16)
        odd_logits = jnp.dot(
            both_normed_vmem[1], w_g, preferred_element_type=jnp.float32
        ).astype(jnp.bfloat16)
        odd_gate = jax.nn.sigmoid(odd_logits.astype(jnp.float32)).astype(jnp.bfloat16)

        gate_half_bufs[slot, 0, ...] = even_gate
        gate_half_bufs[slot, 1, ...] = odd_gate
        dst_col = parity * half_head_cols + g_s * gate_cols
        pltpu.make_async_copy(
            gate_half_bufs.at[slot],
            gate_both_hbm.at[:, :, pl.ds(dst_col, gate_cols)],
            g_local_sems.at[3 + slot],
        ).start()
        pltpu.make_async_remote_copy(
            gate_half_bufs.at[slot],
            gate_both_hbm.at[:, :, pl.ds(dst_col, gate_cols)],
            g_send_sems.at[1 + slot],
            g_recv_sems.at[1 + slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        ).start()

        @pl.when(g_s + 2 < gate_steps)
        def prefetch_next_gate_w():
            start_w_gate(g_s + 2)

    @pl.when(step == gate_steps + 1)
    def drain_gate_penultimate():
        wait_gate_out(gate_steps - 2)

    @pl.when(step == gate_steps + 2)
    def drain_gate_final():
        wait_gate_out(gate_steps - 1)

    @pl.when(step == grid_steps - 1)
    def finish_local_and_sync():
        pltpu.make_async_copy(
            local_ref, window_ref.at[owner_rows_slice(0, positions[0])], own_sem
        ).wait()
        for peer in neighbours:
            pl.semaphore_signal(
                done, 1, device_id=(peer,), device_id_type=pl.DeviceIdType.MESH
            )
        pl.semaphore_wait(done, len(neighbours))
        _sync_partner(partner_sems.at[1], partner)


def rank_half_gather_and_mla_gate_chip_pair_pallas(
    packed_local,
    local_normed,
    m_gate_half,
    tables,
    *,
    layer,
    num_ranks: int = RANKS,
    collective_id: int = 45,
    interpret: bool | pltpu.InterpretParams = False,
):
    rows, packed_width = packed_local.shape
    norm_rows, width = local_normed.shape
    total_head_cols = num_ranks * HEAD_COLS
    half_head_cols = total_head_cols // 2
    if (
        num_ranks < 4
        or num_ranks % 4 != 0
        or tables.shape != (3, 3, num_ranks)
        or norm_rows != rows
        or rows < 8
        or rows % 8 != 0
        or width < 256
        or width % 256 != 0
        or packed_width < 384
        or packed_width % 128 != 0
        or m_gate_half.ndim != 3
        or m_gate_half.shape[-2:] != (width, half_head_cols)
        or packed_local.dtype != jnp.bfloat16
        or local_normed.dtype != jnp.bfloat16
        or m_gate_half.dtype != jnp.bfloat16
    ):
        raise ValueError(
            "rank_half_gather_and_mla_gate_chip_pair_pallas requires R%4==0, "
            "packed_local=[rows, 128k] bf16, local_normed=[rows, W] bf16, and "
            "m_gate_half=[slots, W, R*192] bf16"
        )
    gate_cols = OWNER_GATE_SOURCES * HEAD_COLS if num_ranks >= 8 else 256
    gate_steps = half_head_cols // gate_cols
    ring_steps = num_ranks // 2 + 1
    grid_steps = max(ring_steps, gate_steps + 3)
    streams = len(_half_ring_streams(packed_width))
    scalars = jnp.asarray([jnp.asarray(layer, jnp.int32).reshape(())], jnp.int32)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _rank_half_gather_and_mla_gate_chip_pair_kernel,
            num_ranks=num_ranks,
            gate_cols=gate_cols,
            grid_steps=grid_steps,
        ),
        out_shape=(
            jax.ShapeDtypeStruct((num_ranks * rows, packed_width), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, rows, total_head_cols), jnp.bfloat16),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(1, grid_steps),
            in_specs=(hbm, hbm, hbm),
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.SemaphoreType.DMA((streams,)),
                pltpu.SemaphoreType.DMA((streams, num_ranks // 2)),
                pltpu.SemaphoreType.DMA,
                pltpu.SemaphoreType.REGULAR,
                pltpu.VMEM((2, rows, width), jnp.bfloat16),
                pltpu.VMEM((2, width, gate_cols), jnp.bfloat16),
                pltpu.VMEM((2, 2, rows, gate_cols), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((5,)),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.SemaphoreType.DMA((3,)),
                pltpu.SemaphoreType.REGULAR((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=collective_id, vmem_limit_bytes=64 * 1024**2
        ),
        interpret=interpret,
        name="kimi_prefill_mla_ag_and_owner_gate_chip_pair",
    )(jnp.asarray(tables, jnp.int32), scalars, packed_local, local_normed, m_gate_half)


def _mla_owner_out_gated_chip_pair_kernel(
    scalars_ref,
    owner_rows_hbm,
    gate_both_hbm,
    m_o_half_hbm,
    prefix_hbm,
    out_hbm,
    own_bufs,
    partner_q_bufs,
    gate_bufs,
    w_bufs,
    acc_even_vmem,
    acc_odd_vmem,
    px_out_half_vmem,
    local_sems,
    send_sems,
    recv_sems,
    sync_sems,
    *,
    stripe: int,
    num_ranks: int,
    group: int,
    q_ranks: int,
):
    rank = jax.lax.axis_index("k3_sp")
    parity = rank & 1
    partner = rank ^ 1
    rows = prefix_hbm.shape[0]
    half_w = prefix_hbm.shape[1] // 2
    steps = num_ranks // group
    steps_per_q = q_ranks // group
    out_cols = group * HEAD_COLS
    layer = scalars_ref[0]

    _partner_enter(partner, sync_sems.at[0])

    def partner_quarter(q):
        q_slot = q % 2
        return pltpu.make_async_remote_copy(
            owner_rows_hbm.at[pl.ds(q * q_ranks, q_ranks)],
            partner_q_bufs.at[q_slot],
            send_sems.at[q_slot],
            recv_sems.at[q_slot],
            device_id=(partner,),
            device_id_type=pl.DeviceIdType.MESH,
        )

    partner_quarter(0).start()
    partner_quarter(1).start()

    own_px = prefix_hbm.at[:, pl.ds(parity * half_w, half_w)]
    send_px = prefix_hbm.at[:, pl.ds((1 - parity) * half_w, half_w)]
    px_local = pltpu.make_async_copy(
        own_px, px_out_half_vmem.at[parity], local_sems.at[6]
    )
    px_remote = pltpu.make_async_remote_copy(
        send_px,
        px_out_half_vmem.at[parity],
        send_sems.at[2],
        recv_sems.at[2],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    px_local.start()
    px_remote.start()

    def start_step(s):
        slot = s % 2
        pltpu.make_async_copy(
            owner_rows_hbm.at[pl.ds(s * group, group)],
            own_bufs.at[slot],
            local_sems.at[slot],
        ).start()
        pltpu.make_async_copy(
            gate_both_hbm.at[:, :, pl.ds(s * out_cols, out_cols)],
            gate_bufs.at[slot],
            local_sems.at[2 + slot],
        ).start()
        pltpu.make_async_copy(
            m_o_half_hbm.at[layer, pl.ds(s * out_cols, out_cols), :],
            w_bufs.at[slot],
            local_sems.at[4 + slot],
        ).start()

    start_step(0)
    start_step(1)
    acc_even_vmem[...] = jnp.zeros((rows, half_w), jnp.float32)
    acc_odd_vmem[...] = jnp.zeros((rows, half_w), jnp.float32)

    @pl.loop(0, steps)
    def run_step(s):
        slot = s % 2
        q = s // steps_per_q
        q_slot = q % 2
        q_sub = s % steps_per_q

        pltpu.make_async_copy(
            own_bufs.at[slot], own_bufs.at[slot], local_sems.at[slot]
        ).wait()
        pltpu.make_async_copy(
            gate_bufs.at[slot], gate_bufs.at[slot], local_sems.at[2 + slot]
        ).wait()
        pltpu.make_async_copy(
            w_bufs.at[slot], w_bufs.at[slot], local_sems.at[4 + slot]
        ).wait()

        @pl.when(q_sub == 0)
        def wait_quarter_partner():
            pltpu.make_async_copy(
                partner_q_bufs.at[q_slot],
                partner_q_bufs.at[q_slot],
                recv_sems.at[q_slot],
            ).wait()

        own_flat = jnp.concatenate([own_bufs[slot, j] for j in range(group)], axis=1)
        part_flat = jnp.concatenate(
            [partner_q_bufs[q_slot, q_sub * group + j] for j in range(group)], axis=1
        )
        even_heads = jnp.where(parity == 0, own_flat, part_flat)
        odd_heads = jnp.where(parity == 0, part_flat, own_flat)

        even_gated = (even_heads * gate_bufs[slot, 0]).astype(jnp.bfloat16)
        odd_gated = (odd_heads * gate_bufs[slot, 1]).astype(jnp.bfloat16)
        w = w_bufs[slot][...]
        acc_even_vmem[...] += jnp.dot(even_gated, w, preferred_element_type=jnp.float32)
        acc_odd_vmem[...] += jnp.dot(odd_gated, w, preferred_element_type=jnp.float32)

        @pl.when((q_sub == steps_per_q - 1) & (q + 2 < 4))
        def advance_quarter_partner():
            partner_quarter(q).wait_send()
            _sync_partner(sync_sems.at[1 + q_slot], partner)
            partner_quarter(q + 2).start()

        @pl.when(s + 2 < steps)
        def prefetch_next():
            start_step(s + 2)

    partner_quarter(2).wait_send()
    partner_quarter(3).wait_send()
    px_local.wait()
    px_remote.wait_send()
    pltpu.make_async_copy(
        px_out_half_vmem.at[1 - parity],
        px_out_half_vmem.at[1 - parity],
        recv_sems.at[2],
    ).wait()

    length, own_even, own_odd = scalars_ref[1], scalars_ref[2], scalars_ref[3]
    local = jax.lax.broadcasted_iota(jnp.int32, (rows, 1), 0)
    valid_even = _sp_token_index_ranks(own_even, local, stripe, num_ranks) < length
    valid_odd = _sp_token_index_ranks(own_odd, local, stripe, num_ranks) < length
    out_even = (
        px_out_half_vmem[0].astype(jnp.float32)
        + jnp.where(valid_even, acc_even_vmem[...], 0.0)
    ).astype(out_hbm.dtype)
    out_odd = (
        px_out_half_vmem[1].astype(jnp.float32)
        + jnp.where(valid_odd, acc_odd_vmem[...], 0.0)
    ).astype(out_hbm.dtype)
    px_out_half_vmem[0, ...] = out_even
    px_out_half_vmem[1, ...] = out_odd

    out_local = pltpu.make_async_copy(
        px_out_half_vmem.at[parity],
        out_hbm.at[:, pl.ds(parity * half_w, half_w)],
        local_sems.at[7],
    )
    out_remote = pltpu.make_async_remote_copy(
        px_out_half_vmem.at[1 - parity],
        out_hbm.at[:, pl.ds(parity * half_w, half_w)],
        send_sems.at[3],
        recv_sems.at[3],
        device_id=(partner,),
        device_id_type=pl.DeviceIdType.MESH,
    )
    out_local.start()
    out_remote.start()
    out_local.wait()
    out_remote.wait_send()
    pltpu.make_async_copy(
        out_hbm.at[:, pl.ds((1 - parity) * half_w, half_w)],
        out_hbm.at[:, pl.ds((1 - parity) * half_w, half_w)],
        recv_sems.at[3],
    ).wait()
    _sync_partner(sync_sems.at[3], partner)


def owner_mla_output_gated_chip_pair_pallas(
    owner_rows,
    gate_both,
    m_o_half,
    prefix,
    *,
    layer,
    length,
    own_even,
    own_odd,
    stripe: int,
    num_ranks: int = RANKS,
    out_collective_id: int = 57,
    interpret: bool | pltpu.InterpretParams = False,
):
    ranks, rows, cols = owner_rows.shape
    px_rows, width = prefix.shape
    total_head_cols = num_ranks * HEAD_COLS
    half_w = width // 2
    if (
        num_ranks < 4
        or num_ranks % 4 != 0
        or ranks != num_ranks
        or cols != HEAD_COLS
        or px_rows != rows
        or rows < 8
        or rows % 8 != 0
        or width < 256
        or width % 256 != 0
        or gate_both.shape != (2, rows, total_head_cols)
        or m_o_half.ndim != 3
        or m_o_half.shape[-2:] != (total_head_cols, half_w)
        or owner_rows.dtype != jnp.bfloat16
        or gate_both.dtype != jnp.bfloat16
        or m_o_half.dtype != jnp.bfloat16
        or prefix.dtype != jnp.bfloat16
    ):
        raise ValueError(
            "owner_mla_output_gated_chip_pair_pallas expects R%4==0, "
            "owner_rows=[R, rows, 384], "
            "gate_both=[2, rows, R*384], m_o_half=[slots, R*384, W//2], and "
            "prefix=[rows, W] bf16"
        )
    q_ranks = num_ranks // 4
    group = OWNER_OUT_SOURCES if num_ranks == RANKS else max(1, num_ranks // 8)
    out_cols = group * HEAD_COLS
    scalars = jnp.stack(
        [
            jnp.asarray(v, jnp.int32).reshape(())
            for v in (layer, length, own_even, own_odd)
        ]
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    return pl.pallas_call(
        functools.partial(
            _mla_owner_out_gated_chip_pair_kernel,
            stripe=stripe,
            num_ranks=num_ranks,
            group=group,
            q_ranks=q_ranks,
        ),
        out_shape=jax.ShapeDtypeStruct((rows, width), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=(hbm, hbm, hbm, hbm),
            out_specs=hbm,
            scratch_shapes=(
                pltpu.VMEM((2, group, rows, HEAD_COLS), jnp.bfloat16),
                pltpu.VMEM((2, q_ranks, rows, HEAD_COLS), jnp.bfloat16),
                pltpu.VMEM((2, 2, rows, out_cols), jnp.bfloat16),
                pltpu.VMEM((2, out_cols, half_w), jnp.bfloat16),
                pltpu.VMEM((rows, half_w), jnp.float32),
                pltpu.VMEM((rows, half_w), jnp.float32),
                pltpu.VMEM((2, rows, half_w), jnp.bfloat16),
                pltpu.SemaphoreType.DMA((8,)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.DMA((4,)),
                pltpu.SemaphoreType.REGULAR((4,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            collective_id=out_collective_id, vmem_limit_bytes=64 * 1024**2
        ),
        interpret=interpret,
        name="kimi_prefill_mla_owner_projection_chip_pair",
    )(scalars, owner_rows, gate_both, m_o_half, prefix)
