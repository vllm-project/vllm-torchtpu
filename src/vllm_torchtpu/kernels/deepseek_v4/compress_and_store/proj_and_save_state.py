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

# Token rows per matmul step.
TILE_N = 128
# Tokens whose f32 state rows stay in VMEM while each weight tile is fetched once.
TOKEN_CHUNK = 1024


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Tile sizes for Pallas kernel."""

    tile_m: int  # output dimension
    tile_k: int  # hidden dimension
    tile_n: int  # sequence length


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
    chunk_n: int  # tokens per weight pass; a multiple of tile_n dividing size_n

    @property
    def num_m(self) -> int:
        return pl.cdiv(self.dims.size_m, self.tile_sizes.tile_m)

    @property
    def num_k(self) -> int:
        return pl.cdiv(self.dims.size_k, self.tile_sizes.tile_k)

    @property
    def num_n(self) -> int:
        """Token tiles per chunk."""
        return self.chunk_n // self.tile_sizes.tile_n

    @property
    def num_chunks(self) -> int:
        return self.dims.size_n // self.chunk_n


def _acc_rows(ref, m_idx, slots_per_m_tile):
    """128-lane rows of the m-th state half of every token in `ref`.

    `ref` is (tile_n, slots_per_token, 1, last_dim); returns the flat view, the
    row stride between tokens, and the first row of each 128-lane chunk.
    """
    tile_n, slots_per_token, _, last_dim = ref.shape
    num_lanes = 128
    lanes_per_slot = last_dim // num_lanes
    chunks_per_m = lanes_per_slot * slots_per_m_tile
    stride = lanes_per_slot * slots_per_token
    starts = [m_idx * chunks_per_m + c for c in range(chunks_per_m)]
    return ref.reshape(-1, num_lanes), tile_n, stride, starts


def load_from_acc_ref(ref, m_idx, slots_per_m_tile, dtype=None):
    """Loads one state half of a token tile from the VMEM accumulator."""
    ref_flat, tile_n, stride, starts = _acc_rows(ref, m_idx, slots_per_m_tile)
    chunks = [ref_flat[pl.ds(s, tile_n, stride)] for s in starts]
    vec = jnp.concat(chunks, axis=1)
    if dtype is not None:
        return pltpu.bitcast(vec, dtype)
    return vec


def store_to_acc_ref(ref, m_idx, slots_per_m_tile, val):
    """Stores one state half of a token tile to the VMEM accumulator."""
    ref_flat, tile_n, stride, starts = _acc_rows(ref, m_idx, slots_per_m_tile)
    num_lanes = 128
    for i, s in enumerate(starts):
        ref_flat[pl.ds(s, tile_n, stride)] = val[:, i * num_lanes : (i + 1) * num_lanes]


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
    acc_ref,  # (num_n, tile_n, num_m * slots_per_m_tile, 1, last_dim)
    sem_ref,
    write_count_ref,  # [1]
    *,
    cfgs: Configs,
):
    """Inner kernel executing matmul, APE addition, and paged writes.

    The grid runs the token tiles innermost, so each (k, m) weight tile is
    fetched once per chunk of `chunk_n` tokens instead of once per token tile;
    the chunk's f32 state rows stay in `acc_ref` until they are written out.
    """
    tile_m = cfgs.tile_sizes.tile_m
    tile_n = cfgs.tile_sizes.tile_n
    num_m, num_k, num_n = cfgs.num_m, cfgs.num_k, cfgs.num_n
    last_dim = acc_ref.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    slots_per_token = acc_ref.shape[2]
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
    cache = csa_cache_layout.as_u8_slabs(cache)
    # A slot is page * page_size + row, so the flat row index is the slot itself.
    num_pages, page_size, sub_rows, lanes = cache.shape
    cache_rows = cache.reshape(num_pages * page_size, sub_rows, lanes)
    write_count_ref[0] = 0

    def _wait_for_writes():
        count = write_count_ref[0]

        @pl.when(count > 0)
        def _do_wait():
            pl.semaphore_wait(sem_ref.at[0], count)

        write_count_ref[0] = 0

    def _dma_copy_out(token_base, n_idx_val):
        """Copies a finished token tile to the KV cache, one DMA per token.

        A token's kv and score states are adjacent slots in both the
        accumulator and the cache. The copy is issued in the grid step after
        the tile's last matmul, so the accumulator stores have landed.
        """
        num_valid_writes = 0
        for i in range(tile_n):
            global_token_idx = token_base + n_idx_val * tile_n + i
            global_slot_idx = slot_mapping[global_token_idx]
            valid = global_slot_idx >= 0
            num_valid_writes = num_valid_writes + lax.select(valid, 1, 0)

            # Guard index calculations to avoid out-of-bounds even for size 0
            safe_slot_idx = lax.select(valid, global_slot_idx, 0)

            size = lax.select(valid, slots_per_token, 0)
            dest = cache_rows.at[pl.ds(safe_slot_idx, size), :, :]
            src = acc_ref_u8.at[n_idx_val, i, pl.ds(0, size), :, :]
            pltpu.make_async_copy(src, dest, sem_ref.at[0]).start()

        write_count_ref[0] = write_count_ref[0] + num_valid_writes

    def run_chunk(token_base):
        def inner_kernel(
            tiled_wgate_ref,  # (tile_k, tile_m)
            tiled_hidden_states_ref,  # (tile_n, tile_k)
            tiled_positions_ref,  # (tile_n,)
        ):
            m_idx = pl.program_id(0)
            k_idx = pl.program_id(1)
            n_idx = pl.program_id(2)
            acc_n = acc_ref.at[n_idx]

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
                one_hot = (
                    ape_rows[:, None] == jnp.arange(cfgs.dims.compress_ratio)[None, :]
                ).astype(jnp.float32)
                ape_selected = jnp.matmul(
                    one_hot, ape_val, precision=lax.Precision.HIGHEST
                )  #  (tile_n, state_width)
                existing_acc = load_from_acc_ref(
                    acc_n, m_idx, slots_per_m_tile, dtype=jnp.float32
                )
                new_acc = existing_acc + ape_selected  # (tile_n, state_width)
                store_to_acc_ref(acc_n, m_idx, slots_per_m_tile, new_acc)

            def process(copy_dma: bool, add_ape: bool):
                if copy_dma:
                    # Token tile n - 1 finished in the previous grid step.
                    _dma_copy_out(token_base, n_idx - 1)

                prod = _matmul()

                acc_val = load_from_acc_ref(
                    acc_n, m_idx, slots_per_m_tile, dtype=jnp.float32
                )
                old_acc = lax.select(
                    k_idx > 0,
                    acc_val,
                    jnp.zeros_like(acc_val),
                )
                new_acc = old_acc + prod
                store_to_acc_ref(acc_n, m_idx, slots_per_m_tile, new_acc)

                if add_ape:
                    _add_ape()

            is_last_k = k_idx == (num_k - 1)
            is_score = m_idx * tile_m >= state_width

            copy_dma = (m_idx == num_m - 1) & is_last_k & (n_idx > 0)
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
            lambda m, k, n: (k, m),
            memory_space=pltpu.VMEM,
        )
        hidden_states_spec = pl.BlockSpec(
            (cfgs.tile_sizes.tile_n, cfgs.tile_sizes.tile_k),
            lambda m, k, n: (n, k),
            memory_space=pltpu.VMEM,
        )
        positions_spec = pl.BlockSpec(
            (cfgs.tile_sizes.tile_n,),
            lambda m, k, n: (n,),
            memory_space=pltpu.VMEM,
        )

        pipeline_fn = pltpu.emit_pipeline(
            inner_kernel,
            grid=(num_m, num_k, num_n),
            in_specs=(
                wkv_spec,
                hidden_states_spec,
                positions_spec,
            ),
        )

        chunk = pl.ds(token_base, cfgs.chunk_n)
        pipeline_fn(
            wkv_wgate,
            hidden_states.at[chunk],
            positions.at[chunk],
        )

        # Epilogue: write the chunk's last tile.
        _dma_copy_out(token_base, num_n - 1)

    if cfgs.num_chunks == 1:
        run_chunk(0)
    else:

        def chunk_body(c, carry):
            # The next chunk reuses the accumulator the writes read from.
            _wait_for_writes()
            run_chunk(c * cfgs.chunk_n)
            return carry

        lax.fori_loop(0, cfgs.num_chunks, chunk_body, 0)

    _wait_for_writes()


def _select_tile_k(size_k: int, cap: int = 3584, align: int = 128) -> int:
    """Largest 128-aligned k tile that *divides* ``size_k``."""
    cap = max(align, (min(cap, size_k) // align) * align)
    for cand in range(cap, 0, -align):
        if size_k % cand == 0:
            return cand
    raise AssertionError(f"Cannot find a tile_k that divides size_k={size_k}")


def _select_chunk_n(num_tokens: int, tile_n: int) -> int:
    """Largest multiple of ``tile_n`` dividing ``num_tokens``, up to TOKEN_CHUNK."""
    num_tiles = num_tokens // tile_n
    max_tiles = max(1, TOKEN_CHUNK // tile_n)
    for tiles in range(min(num_tiles, max_tiles), 0, -1):
        if num_tiles % tiles == 0:
            return tiles * tile_n
    raise AssertionError(f"num_tokens={num_tokens} is not a multiple of {tile_n}")


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
    tile_n = min(TILE_N, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
    padded_num_tokens = pl.cdiv(num_tokens, tile_n) * tile_n
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

    tile_sizes = TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)

    cfgs = Configs(
        tile_sizes=tile_sizes,
        dims=dims,
        chunk_n=_select_chunk_n(num_tokens, tile_n),
    )

    last_dim = cache.shape[-1]
    slots_per_token = state_dim // last_dim
    scratch_shapes = [
        pltpu.VMEM((cfgs.num_n, tile_n, slots_per_token, 1, last_dim), jnp.float32),
        pltpu.SemaphoreType.REGULAR((1,)),
        pltpu.SMEM((1,), jnp.int32),
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
