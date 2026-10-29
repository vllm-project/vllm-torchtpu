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
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout

# Tokens whose f32 accumulators stay resident in VMEM while one weight block is
# reused across them; each weight block is read from HBM once per this many.
# Costs 2 * tokens * state_width * 4 bytes of VMEM (8 MiB at state_width 1024),
# next to two (tile_k, state_width) bf16 weight buffers (8 MiB at tile_k 2048);
# 1024 is the largest serving token padding, so every call is a single block.
MAX_RESIDENT_TOKENS = 1024
# Slot mappings read from SMEM back to back before their tokens' write DMAs are
# issued, so the load latency is paid once per group instead of once per token.
SLOT_LOAD_GROUP = 8


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Tile sizes for Pallas kernel."""

    tile_m: int  # output dimension
    tile_k: int  # hidden dimension
    tile_n: int  # sequence length
    tiles_per_block: int  # token tiles sharing one fetch of a weight block


@dataclasses.dataclass(frozen=True)
class Dimensions:
    """Dimensions of the input and output tensors."""

    size_m: int  # output dimension
    size_k: int  # hidden dimension
    size_n: int  # sequence length

    # other values
    compress_ratio: int


@dataclasses.dataclass(frozen=True)
class Configs:
    """Configuration parameters for the projection and save state kernel."""

    tile_sizes: TileSizes
    dims: Dimensions

    @property
    def num_m(self) -> int:
        return pl.cdiv(self.dims.size_m, self.tile_sizes.tile_m)

    @property
    def num_k(self) -> int:
        return pl.cdiv(self.dims.size_k, self.tile_sizes.tile_k)

    @property
    def num_n(self) -> int:
        return pl.cdiv(self.dims.size_n, self.tile_sizes.tile_n)

    @property
    def num_token_blocks(self) -> int:
        return self.num_n // self.tile_sizes.tiles_per_block


def load_from_acc_ref(ref, dtype=None):
    """Loads data from VMEM accumulator with strided access, handling
    128-lane alignment."""
    tile_n, slots_per_m_tile, _, last_dim = ref.shape
    num_lanes = 128

    # How many 128-lane hardware vectors make up one slot (1 for 128, 2 for 256)
    lanes_per_slot = last_dim // num_lanes
    num_chunks = lanes_per_slot * slots_per_m_tile
    ref_flat = ref.reshape(-1, num_lanes)

    # Gather the chunks into the original head dimension.
    chunks = [ref_flat[pl.ds(m, tile_n, num_chunks)] for m in range(num_chunks)]
    vec = jnp.concat(chunks, axis=1)
    if dtype is not None:
        return pltpu.bitcast(vec, dtype)
    return vec


def store_to_acc_ref(ref, val, dtype=None):
    """Stores data to VMEM accumulator with strided access, handling
    128-lane alignment."""
    tile_n, slots_per_m_tile, _, last_dim = ref.shape
    num_lanes = 128

    # How many 128-lane hardware vectors make up one slot (1 for 128, 2 for 256)
    lanes_per_slot = last_dim // num_lanes
    num_chunks = lanes_per_slot * slots_per_m_tile

    ref_flat = ref.reshape(-1, num_lanes)
    for i in range(num_chunks):
        val_slice = val[:, i * num_lanes : (i + 1) * num_lanes]
        ref_flat[pl.ds(i, tile_n, num_chunks)] = val_slice


def kernel(
    # scalar prefetch
    slot_mapping,
    positions,
    # input
    hidden_states,  # [num_tokens, hidden_size]
    wkv_wgate,  # [2 * state_width, hidden_size]
    ape,  # [compress_ratio, state_width]
    cache,  # [num_pages, page_size, 4, 128] uint8 or [..., 128] int32 (aliased)
    # output
    _,  # unused output ref (aliased to cache)
    # scratch_memory
    acc_ref,  # (num_m, tiles_per_block, tile_n, slots_per_m_tile, 1, 128)
    sem_ref,  # [tiles_per_block]
    write_counts_ref,  # [tiles_per_block]
    *,
    cfgs: Configs,
):
    """Inner kernel executing matmul, APE addition, and paged writes."""
    tile_m = cfgs.tile_sizes.tile_m
    tile_n = cfgs.tile_sizes.tile_n
    tiles_per_block = cfgs.tile_sizes.tiles_per_block
    last_dim = acc_ref.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
    cache = csa_cache_layout.as_u8_slabs(cache)
    num_m = cfgs.num_m
    for slot in range(tiles_per_block):
        write_counts_ref[slot] = 0

    def _dma_copy_out(n_idx_val, buf_idx_val):
        """Copys one token tile, every m tile, from accumulator to the KV cache.

        We run the dma one grid step after the last m tile is done computing;
        each token tile of a token block has its own accumulator slot per m
        tile, and one strided DMA per token writes all of them.

        The reason we do this in the following tile is that we hope to pipeline the
        DMA with the matmuls.

        (however, this is awaiting RAW no-hazard flag to land)
        """
        num_valid_writes = 0
        # A slot is page * page_size + row, i.e. the row of the flattened cache.
        cache_rows = cache.reshape(-1, *cache.shape[2:])
        for i in range(tile_n):
            if i % SLOT_LOAD_GROUP == 0:
                group_slots = [
                    slot_mapping[n_idx_val * tile_n + t]
                    for t in range(i, min(i + SLOT_LOAD_GROUP, tile_n))
                ]
            global_slot_idx = group_slots[i % SLOT_LOAD_GROUP]
            valid = global_slot_idx >= 0
            num_valid_writes = num_valid_writes + lax.select(valid, 1, 0)

            @pl.when(valid)
            def _start(global_slot_idx=global_slot_idx, i=i):
                # The m tiles of a token are consecutive slots in the cache.
                dest = cache_rows.at[
                    pl.ds(global_slot_idx, num_m * slots_per_m_tile)
                ].reshape(num_m, slots_per_m_tile, *cache.shape[2:])
                src = acc_ref_u8.at[:, buf_idx_val, i]
                pltpu.make_async_copy(src, dest, sem_ref.at[buf_idx_val]).start()

        write_counts_ref[buf_idx_val] = num_valid_writes

    def inner_kernel(
        tiled_wgate_ref,  # (tile_k, tile_m)
        tiled_hidden_states_ref,  # (tile_n, tile_k)
        tiled_positions_ref,  # (tile_n,)
    ):
        b_idx = pl.program_id(0)
        m_idx = pl.program_id(1)
        k_idx = pl.program_id(2)
        j_idx = pl.program_id(3)
        n_idx = b_idx * tiles_per_block + j_idx
        buf_idx = j_idx

        state_width = cfgs.dims.size_m // 2

        def _matmul():
            lhs = tiled_hidden_states_ref[...]
            rhs = tiled_wgate_ref[...]
            prod = lax.dot_general(
                lhs,
                rhs,
                (((1,), (0,)), ((), (()))),
                preferred_element_type=jnp.float32,
            )
            return prod

        def _add_ape():
            tile_positions = tiled_positions_ref[...]
            ape_rows = tile_positions % cfgs.dims.compress_ratio
            ape_val = ape[...].astype(jnp.float32)
            # Gather Path: doesn't work for compiler reasons?
            # gather_indices = jnp.broadcast_to(
            #     ape_rows[:, None], (tile_n, ape_val.shape[1])
            # )
            # ape_selected = jnp.take_along_axis(ape_val, gather_indices, axis=0)
            # matmul path
            one_hot = (
                ape_rows[:, None] == jnp.arange(cfgs.dims.compress_ratio)[None, :]
            ).astype(jnp.float32)
            ape_selected = jnp.matmul(
                one_hot, ape_val, precision=lax.Precision.HIGHEST
            )  #  (tile_n, state_width)
            existing_acc = load_from_acc_ref(
                acc_ref.at[m_idx, buf_idx], dtype=jnp.float32
            )
            new_acc = existing_acc + ape_selected  # (tile_n, state_width)
            store_to_acc_ref(acc_ref.at[m_idx, buf_idx], new_acc)

        def process(copy_dma: bool, add_ape: bool):
            # 1. Semaphore wait (only on the slot's first step)
            @pl.when((m_idx == 0) & (k_idx == 0))
            def _wait_for_buffer():
                count = write_counts_ref[buf_idx][...]

                @pl.when(count > 0)
                def _do_wait():
                    pl.semaphore_wait(sem_ref.at[buf_idx], count)

            # 2. Launch DMA for the PREVIOUS token tile
            if copy_dma:
                prev_buf_idx = jnp.where(j_idx == 0, tiles_per_block - 1, j_idx - 1)
                _dma_copy_out(n_idx - 1, prev_buf_idx)

            # 3. Matmul
            prod = _matmul()

            # 4. Unified Accumulate/Assign
            acc_val = load_from_acc_ref(acc_ref.at[m_idx, buf_idx], dtype=jnp.float32)
            old_acc = lax.select(
                k_idx > 0,
                acc_val,
                jnp.zeros_like(acc_val),
            )
            new_acc = old_acc + prod
            store_to_acc_ref(acc_ref.at[m_idx, buf_idx], new_acc)

            # 5. APE addition if last
            if add_ape:
                _add_ape()

        # Select and execute
        num_k = cfgs.num_k
        is_last_k = k_idx == (num_k - 1)
        is_score = m_idx * tile_m >= state_width

        # True when the previous grid step finished the last m tile of a token tile.
        copy_dma = ((m_idx == num_m - 1) & is_last_k & (j_idx > 0)) | (
            (b_idx > 0) & (m_idx == 0) & (k_idx == 0) & (j_idx == 0)
        )
        add_ape = is_last_k & is_score

        @pl.when(jnp.logical_not(copy_dma) & jnp.logical_not(add_ape))
        @jax.named_scope("matmul")
        def matmul():
            process(False, False)

        @pl.when(copy_dma & add_ape)
        @jax.named_scope("matmul_ape_write_out")
        def matmul_ape_write_out():
            process(True, True)

        @pl.when(copy_dma & jnp.logical_not(add_ape))
        @jax.named_scope("matmul_write_out")
        def matmul_write_out():
            process(True, False)

        @pl.when(jnp.logical_not(copy_dma) & add_ape)
        @jax.named_scope("matmul_ape_only")
        def matmul_ape_only():
            process(False, True)

    wkv_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_k, cfgs.tile_sizes.tile_m),
        lambda b, m, k, j: (k, m),
        memory_space=pltpu.VMEM,
    )
    hidden_states_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n, cfgs.tile_sizes.tile_k),
        lambda b, m, k, j: (b * tiles_per_block + j, k),
        memory_space=pltpu.VMEM,
    )
    positions_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n,),
        lambda b, m, k, j: (b * tiles_per_block + j,),
        memory_space=pltpu.VMEM,
    )

    pipeline_fn = pltpu.emit_pipeline(
        inner_kernel,
        grid=(cfgs.num_token_blocks, cfgs.num_m, cfgs.num_k, tiles_per_block),
        in_specs=(
            wkv_spec,
            hidden_states_spec,
            positions_spec,
        ),
    )

    pipeline_fn(
        wkv_wgate,
        hidden_states,
        positions,
    )

    # Epilogue: launch DMA for the last tile and wait for all outstanding DMAs
    if cfgs.num_n > 0:
        _dma_copy_out(cfgs.num_n - 1, tiles_per_block - 1)

    for b in range(tiles_per_block):
        count = write_counts_ref[b][...]

        @pl.when(count > 0)
        def _wait_epilogue(b=b, count=count):
            pl.semaphore_wait(sem_ref.at[b], count)


def _select_tile_k(size_k: int, cap: int = 3584, align: int = 128) -> int:
    """Largest 128-aligned k tile that *divides* ``size_k``."""
    cap = max(align, (min(cap, size_k) // align) * align)
    for cand in range(cap, 0, -align):
        if size_k % cand == 0:
            return cand
    raise AssertionError(f"Cannot find a tile_k that divides size_k={size_k}")


@functools.partial(
    jax.jit,
    static_argnames=[
        "compress_ratio",
        "interpret",
    ],
)
def proj_and_save_state(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    wkv_wgate: jax.Array,  # [hidden_size, 2 * state_width]
    ape: jax.Array,  # [compress_ratio, state_width]
    positions: jax.Array,  # [num_tokens]
    slot_mapping: jax.Array,  # [num_tokens]
    cache: jax.Array,  # [num_pages, page_size, 4, 128] uint8 / [..., 128] int32
    compress_ratio: int,
    interpret: bool = False,
) -> jax.Array:
    """Projects hidden states and saves them to the KV cache in HBM.

    Args:
      hidden_states: Input hidden states.
      wkv_wgate: Projection weights.
      ape: Absolute Position Embeddings.
      positions: Token positions.
      slot_mapping: Mapping from token index to cache slot.
      cache: KV cache tensor in HBM (updated in-place).
      compress_ratio: APE compression ratio.

    Returns:
      The updated KV cache tensor.
    """
    num_tokens, hidden_size = hidden_states.shape
    _, state_dim = wkv_wgate.shape

    state_width = state_dim // 2
    tile_m = state_width
    tile_k = _select_tile_k(hidden_size)

    # The token dimension must be an exact multiple of `tile_n`.
    sublane_tile = 8
    tile_n = min(128, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
    num_n = pl.cdiv(num_tokens, tile_n)
    # At least 2 tiles per block: the last tile of a block is written out one
    # step into the next block, after that step's wait on its slot.
    num_token_blocks = pl.cdiv(num_n, max(2, MAX_RESIDENT_TOKENS // tile_n))
    tiles_per_block = pl.cdiv(num_n, num_token_blocks)
    padded_num_tokens = num_token_blocks * tiles_per_block * tile_n
    if padded_num_tokens != num_tokens:
        pad = padded_num_tokens - num_tokens
        hidden_states = jnp.pad(hidden_states, ((0, pad), (0, 0)))
        positions = jnp.pad(positions, (0, pad))
        slot_mapping = jnp.pad(slot_mapping, (0, pad), constant_values=-1)
        num_tokens = padded_num_tokens

    dims = Dimensions(
        size_m=state_dim,
        size_k=hidden_size,
        size_n=num_tokens,
        compress_ratio=compress_ratio,
    )

    tile_sizes = TileSizes(
        tile_m=tile_m, tile_k=tile_k, tile_n=tile_n, tiles_per_block=tiles_per_block
    )

    cfgs = Configs(
        tile_sizes=tile_sizes,
        dims=dims,
    )

    last_dim = cache.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    scratch_shapes = [
        pltpu.VMEM(
            (cfgs.num_m, tiles_per_block, tile_n, slots_per_m_tile, 1, last_dim),
            jnp.float32,
        ),
        pltpu.SemaphoreType.REGULAR((tiles_per_block,)),
        pltpu.SMEM((tiles_per_block,), jnp.int32),
    ]

    out_shape = jax.ShapeDtypeStruct(cache.shape, cache.dtype)

    return pl.pallas_call(
        functools.partial(kernel, cfgs=cfgs),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # positions
                pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states
                pl.BlockSpec(memory_space=pltpu.HBM),  # wkv_wgate
                pl.BlockSpec(memory_space=pltpu.VMEM),  # ape (prefetch to VMEM)
                pl.BlockSpec(memory_space=pltpu.HBM),  # cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        input_output_aliases={5: 0},  # Alias output to cache (index 5)
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name="proj_and_save_state",
    )(slot_mapping, positions, hidden_states, wkv_wgate, ape, cache)
