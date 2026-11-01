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

import dataclasses
import enum
from typing import Any, cast

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import config


def _floor_div(x, divisor: int):
    """``x // divisor``; a shift when divisor is a power of two."""
    if divisor & (divisor - 1) == 0:
        return jax.lax.shift_right_arithmetic(x, divisor.bit_length() - 1)
    return x // divisor


def _flat_rows(ref):
    """(num_pages, rows, ...) -> (num_pages * rows, ...): a slot's row index is
    then its page-major row number."""
    return ref.reshape(ref.shape[0] * ref.shape[1], *ref.shape[2:])


class CopyCount(enum.IntEnum):
    """The per-tile copy counts `compress_norm_rope_store` prefetches."""

    VALID = 0  # tokens with a slot
    FIRST = 1  # tokens that write their NoPE row (is_first_mask)
    FIRST_ROPE = 2  # tokens that write their RoPE row (is_first_mask_rope)
    WHOLE_ROW = 3  # tokens that fill whole rows in slot order (see by_rows)


def by_rows(cfgs: config.Configs, whole_row_count, whole_rows, by_tokens):
    """Runs whole_rows (if any) on a whole-row step, by_tokens on any other.

    In a whole-row step, every row tile fills one packed row in slot order
    (all tile_n tokens count as WHOLE_ROW), so its merged rows depend on no
    row read back. Only modes that merge rows have such steps.
    """
    if not cfgs.merges_rows:
        by_tokens()
        return
    whole = whole_row_count == cfgs.tile_sizes.tile_n
    if whole_rows is not None:
        pl.when(whole)(whole_rows)
    pl.when(jnp.logical_not(whole))(by_tokens)


def _wait_copies(vmem_ref, count, sem):
    """Waits for ``count`` copies of one leading-dim block of vmem_ref each.

    A DMA wait takes its descriptor's bytes off the semaphore, so the waits
    need not name the blocks that were copied, only how many: one wait per
    set bit of ``count``, on that many blocks.
    """
    for bit in range(vmem_ref.shape[0].bit_length()):
        blocks = vmem_ref.at[pl.ds(0, 1 << bit)]

        @pl.when((count & (1 << bit)) != 0)
        def _wait():
            pltpu.make_async_copy(blocks, blocks, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class _BufferedRef(pltpu.BufferedRef):
    cfgs: config.Configs = dataclasses.field(metadata=dict(static=True))

    @classmethod
    def create(
        cls,
        spec: pl.BlockSpec,
        dtype_or_type: jax.typing.DTypeLike,
        buffer_type: pltpu.BufferType,
        buffer_count: int,
        use_lookahead: bool,
        cfgs: config.Configs,
    ):
        standard_ref = pltpu.BufferedRef.create(
            spec=spec,
            dtype_or_type=dtype_or_type,
            buffer_type=buffer_type,
            buffer_count=buffer_count,
            grid_rank=1,
            use_lookahead=use_lookahead,
        )
        return cls(
            cfgs=cfgs,
            **{
                f.name: getattr(standard_ref, f.name)
                for f in dataclasses.fields(pltpu.BufferedRef)
            },
        )


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class CosSinRef(_BufferedRef):
    def copy_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        cos_sin_cache_ref, positions_ref = src_ref
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        pid = grid_indices[0]

        dest_ref = cast(jax.Ref, self.window_ref.at[slot])  # (tile_n, rope_head_dim)

        tile_n = self.cfgs.tile_sizes.tile_n
        compress_ratio = self.cfgs.dims.compress_ratio

        @pl.loop(0, tile_n, unroll=True)
        def loop_body(i):
            global_idx = pid * tile_n + i
            position = positions_ref[global_idx]
            compressed_pos = _floor_div(position, compress_ratio) * compress_ratio

            src = cos_sin_cache_ref.at[
                compressed_pos, pl.ds(0, self.window_ref.shape[-1])
            ]
            dest = dest_ref.at[i, :]
            pltpu.make_async_copy(src, dest, sem).start()

    def wait_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class WindowGroupRef(_BufferedRef):
    # VMEM block: cfgs.window_buffer_shape() + the state array's row shape,
    #   (tile_n, window_groups, group_positions, state_rows_per_token, ...),
    # each token's window states in window order.

    def copy_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        pid = grid_indices[0]

        # state_groups_ref: (num_pages * groups_per_page, group_positions,
        # state_rows_per_token, ...); group_index_ref[n * groups + g] is the
        # group holding token n's g-th window group (see `window_group_index`).
        state_groups_ref, group_index_ref = src_ref
        groups = self.cfgs.window_groups
        tile_n = self.cfgs.tile_sizes.tile_n
        buffer_ref = cast(jax.Ref, self.window_ref.at[slot])
        base = pid * tile_n * groups

        for i in range(tile_n * groups):
            group = group_index_ref[base + i]
            dest = buffer_ref.at[i // groups, i % groups]
            pltpu.make_async_copy(state_groups_ref.at[group], dest, sem).start()

    def wait_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = self.window_ref.at[slot]
        # The copies fill the whole block, so one wait covers their bytes.
        pltpu.make_async_copy(vmem_ref, vmem_ref, sem).wait()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class OutputRef(_BufferedRef):
    # VMEM block: cfgs.output_shape() -> (tile_n, record_rows, hbm_pack, last_dim_size)

    def copy_in(self, src_ref, grid_indices):
        # Only called in CSA_INDEXER mode (INPUT_OUTPUT)
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        cache_ref, kv_slot_mapping_ref, is_first_mask_ref, whole_counts_ref, *_ = (
            src_ref
        )
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        dest_ref = cast(jax.Ref, self.window_ref.at[slot])
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        # cache_ref: [num_pages * kv_block_size / 4, 4, 256], 4 tokens per row
        tokens_per_row = self.cfgs.tokens_in_second_minor

        def read_rows():
            @pl.loop(0, tile_n, unroll=True)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]
                is_first = is_first_mask_ref[idx_n]
                valid = kv_slot >= 0

                # Only is_first_mask tokens write their merged row back, so
                # only their rows need the cache contents.
                @pl.when(valid & is_first)
                def _read_nope():
                    # we fetch the entire tile (4, 128), with 4 pages per tile
                    src = cache_ref.at[_floor_div(kv_slot, tokens_per_row)]
                    dest = dest_ref.at[n, 0, :, :]
                    pltpu.make_async_copy(src, dest, sem).start()

        by_rows(self.cfgs, whole_counts_ref[pid], None, read_rows)

    def wait_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = cast(jax.Ref, self.window_ref.at[slot])
        _, _, _, whole_counts_ref, read_counts_ref, _ = src_ref
        pid = grid_indices[0]
        by_rows(
            self.cfgs,
            whole_counts_ref[pid],
            None,
            lambda: _wait_copies(vmem_ref, read_counts_ref[pid], sem),
        )

    def copy_out(self, dest_ref, grid_indices):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]
        src_ref = cast(jax.Ref, self.window_ref.at[slot])
        pid = grid_indices[0]
        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        if self.cfgs.dims.mode == config.Mode.CSA_INDEXER:
            cache_ref, kv_slot_mapping_ref, is_first_mask_ref, *_ = dest_ref

            @pl.loop(0, tile_n, unroll=True)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]
                is_first = is_first_mask_ref[idx_n]
                valid = (kv_slot >= 0) & is_first

                @pl.when(valid)
                def _write_nope():
                    row = _floor_div(kv_slot, self.cfgs.tokens_in_second_minor)
                    src = src_ref.at[n, 0, :, :]
                    dest = cache_ref.at[row]
                    pltpu.make_async_copy(src, dest, sem).start()
        else:
            cache_ref, kv_slot_mapping_ref, _ = dest_ref

            # HCA: [num_pages * kv_block_size * 2, 4, 128] uint8
            # CSA: [num_pages * kv_block_size, 4, 128] uint8
            @pl.loop(0, tile_n, unroll=True)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]

                @pl.when(kv_slot >= 0)
                def _write_nope():
                    @pl.loop(0, self.cfgs.record_rows, unroll=True)
                    def write_loop(o):
                        src_nope = src_ref.at[n, o, ...]
                        dest_nope = cache_ref.at[kv_slot + o, ...]
                        pltpu.make_async_copy(src_nope, dest_nope, sem).start()

    def wait_out(self, dest_ref, grid_indices):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = cast(jax.Ref, self.window_ref.at[slot])
        write_counts_ref = dest_ref[-1]
        _wait_copies(vmem_ref, write_counts_ref[grid_indices[0]], sem)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RoPEOutputRef(_BufferedRef):
    # VMEM block: cfgs.rope_output_shape() -> (tile_n, 4, LANE) uint8, each
    # token's whole 4-token RoPE row, read-modify-written.
    # Only used for CSA (has_rope_cache=True).

    def copy_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        rope_cache_ref, kv_slot_mapping_ref, is_first_mask_ref, whole_counts_ref, _ = (
            src_ref
        )
        slot = self.current_copy_in_slot
        sem = self.sem_recvs.at[slot]
        dest_ref = cast(jax.Ref, self.window_ref.at[slot])
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        def read_rows():
            @pl.loop(0, tile_n, unroll=True)
            def loop_body(n):
                idx_n = global_idx + n
                kv_slot = kv_slot_mapping_ref[idx_n]
                is_first = is_first_mask_ref[idx_n]
                valid = kv_slot >= 0

                @pl.when(valid & is_first)
                def _read_rope():
                    # layout: [num_pages * kv_block_size / 4, 4, 128], 4 tokens a row
                    row = _floor_div(kv_slot, csa_cache_layout.ROPE_TOKENS_PER_ROW)
                    src = rope_cache_ref.at[row]
                    dest = dest_ref.at[n, :, :]
                    pltpu.make_async_copy(src, dest, sem).start()

        by_rows(self.cfgs, whole_counts_ref[pid], None, read_rows)

    def wait_in(self, src_ref, grid_indices):
        assert self.sem_recvs is not None
        assert self.window_ref is not None
        slot = self.current_wait_in_slot
        sem = self.sem_recvs.at[slot]
        vmem_ref = cast(jax.Ref, self.window_ref.at[slot])
        *_, whole_counts_ref, row_counts_ref = src_ref
        pid = grid_indices[0]
        by_rows(
            self.cfgs,
            whole_counts_ref[pid],
            None,
            lambda: _wait_copies(vmem_ref, row_counts_ref[pid], sem),
        )

    def copy_out(self, dest_ref, grid_indices):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        rope_cache_ref, kv_slot_mapping_ref, is_first_mask_ref, *_ = dest_ref
        slot = self.current_copy_out_slot
        sem = self.sem_sends.at[slot]

        src_ref = cast(jax.Ref, self.window_ref.at[slot])
        pid = grid_indices[0]

        tile_n = self.cfgs.tile_sizes.tile_n
        global_idx = pid * tile_n

        @pl.loop(0, tile_n, unroll=True)
        def loop_body(n):
            idx_n = global_idx + n
            kv_slot = kv_slot_mapping_ref[idx_n]
            is_first = is_first_mask_ref[idx_n]
            valid = kv_slot >= 0

            @pl.when(valid & is_first)
            def _write_rope():
                src_rope = src_ref.at[n, :, :]
                row = _floor_div(kv_slot, csa_cache_layout.ROPE_TOKENS_PER_ROW)
                dest_rope = rope_cache_ref.at[row]
                pltpu.make_async_copy(src_rope, dest_rope, sem).start()

    def wait_out(self, dest_ref, grid_indices):
        assert self.sem_sends is not None
        assert self.window_ref is not None
        slot = self.current_wait_out_slot
        sem = self.sem_sends.at[slot]
        vmem_ref = cast(jax.Ref, self.window_ref.at[slot])
        row_counts_ref = dest_ref[-1]
        _wait_copies(vmem_ref, row_counts_ref[grid_indices[0]], sem)


def create_allocs_and_specs(
    cfgs: config.Configs,
    *,
    cache_ref,
    rope_cache_ref,
    state_cache_ref,
    cos_sin_cache_ref,
    positions_ref,
    token_pages_ref,
    kv_slot_mapping_ref,
    is_first_mask_ref,
    is_first_mask_rope_ref,
    copy_counts_refs,
) -> tuple[
    tuple[
        CosSinRef | None,
        OutputRef,
        WindowGroupRef,
        RoPEOutputRef | None,
    ],
    tuple[pl.BlockSpec | None, pl.BlockSpec, pl.BlockSpec, pl.BlockSpec | None],
    tuple[
        tuple[Any, ...] | None,
        tuple[Any, ...],
        tuple[Any, ...],
        tuple[Any, ...] | None,
    ],
]:
    # The window DMAs read the state rows as the array declares them: an
    # int32 word array's word l of a row is the bits of f32 lane l.
    state_ref = cache_ref if state_cache_ref is None else state_cache_ref
    num_pages, _, *state_row_shape = state_ref.shape
    state_groups_ref = state_ref.reshape(
        num_pages * cfgs.groups_per_page,
        *cfgs.window_buffer_shape()[2:],
        *state_row_shape,
    )
    # CSA's int32 NoPE / RoPE arrays are read and written through their uint8
    # slab view (see `csa_cache_layout`).
    cache_ref = csa_cache_layout.as_u8_slabs(cache_ref)
    if rope_cache_ref is not None:
        rope_cache_ref = csa_cache_layout.as_u8_slabs(rope_cache_ref)
    # Records are addressed by slot, so read and write them through flat rows.
    cache_rows_ref = _flat_rows(cache_ref)

    # cos_sin
    if cfgs.dims.has_rope:
        tile_n = cfgs.tile_sizes.tile_n
        cos_sin_spec = pl.BlockSpec(
            block_shape=cfgs.cos_sin_shape(),
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i * tile_n, 0),
        )
        cos_sin_alloc = CosSinRef.create(
            spec=cos_sin_spec,
            dtype_or_type=cfgs.dims.cos_sin_dtype,
            buffer_type=pltpu.BufferType.INPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfgs,
        )
        cos_sin_args = (cos_sin_cache_ref, positions_ref)
    else:
        cos_sin_alloc = cos_sin_spec = cos_sin_args = None

    # output
    output_spec = pl.BlockSpec(
        block_shape=cfgs.output_shape(),
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i, 0, 0, 0),
    )
    is_indexer = cfgs.dims.mode == config.Mode.CSA_INDEXER
    buffer_type = (
        pltpu.BufferType.INPUT_OUTPUT if is_indexer else pltpu.BufferType.OUTPUT
    )
    output_alloc = OutputRef.create(
        spec=output_spec,
        dtype_or_type=jnp.uint8,
        buffer_type=buffer_type,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    if is_indexer:
        output_args = (
            cache_rows_ref,
            kv_slot_mapping_ref,
            is_first_mask_ref,
            copy_counts_refs[CopyCount.WHOLE_ROW],
            copy_counts_refs[CopyCount.FIRST],  # reads
            copy_counts_refs[CopyCount.FIRST],  # writes
        )
    else:
        output_args = (
            cache_rows_ref,
            kv_slot_mapping_ref,
            copy_counts_refs[CopyCount.VALID],
        )

    # state: each token's window groups
    block_shape = (*cfgs.window_buffer_shape(), *state_row_shape)
    state_spec = pl.BlockSpec(
        block_shape=block_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,) + (0,) * (len(block_shape) - 1),
    )
    state_alloc = WindowGroupRef.create(
        spec=state_spec,
        dtype_or_type=state_ref.dtype,
        buffer_type=pltpu.BufferType.INPUT,
        buffer_count=2,
        use_lookahead=False,
        cfgs=cfgs,
    )
    state_args = (state_groups_ref, token_pages_ref)

    # rope
    if cfgs.dims.has_rope_cache:
        rope_spec = pl.BlockSpec(
            block_shape=cfgs.rope_output_shape(),
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i, 0, 0),
        )
        rope_alloc = RoPEOutputRef.create(
            spec=rope_spec,
            dtype_or_type=jnp.uint8,
            buffer_type=pltpu.BufferType.INPUT_OUTPUT,
            buffer_count=2,
            use_lookahead=False,
            cfgs=cfgs,
        )
        rope_args = (
            _flat_rows(rope_cache_ref),
            kv_slot_mapping_ref,
            is_first_mask_rope_ref,
            copy_counts_refs[CopyCount.WHOLE_ROW],
            copy_counts_refs[CopyCount.FIRST_ROPE],
        )
    else:
        rope_alloc = rope_spec = rope_args = None

    return (
        (cos_sin_alloc, output_alloc, state_alloc, rope_alloc),
        (cos_sin_spec, output_spec, state_spec, rope_spec),
        (cos_sin_args, output_args, state_args, rope_args),
    )
