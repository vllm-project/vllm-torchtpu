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

# Tokens whose f32 state is accumulated in VMEM at once; the weight is
# read from HBM once per group of this many tokens.
GROUP_TOKENS = 1024
# A step holds the whole hidden dimension of the weight, double-buffered:
# 28 MiB for DeepSeek-V4-Pro's CSA compressor, past the default scoped limit.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Tile sizes for Pallas kernel."""

    tile_m: int  # output dimension
    tile_k: int  # hidden dimension of one partial product
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
    token_page_size: int
    tiles_per_group: int  # token tiles sharing one pass over the weight

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
    def num_groups(self) -> int:
        return pl.cdiv(self.num_n, self.tiles_per_group)


def _acc_half_rows(ref, half, num_halves, tile_n):
    """Strided row windows of one output half of a token-major accumulator.

    ``ref`` is ``(tile_n * rows_per_token, 1, last_dim)``: each token's rows
    hold all ``num_halves`` output halves back to back, as the cache stores a
    token's state, so the returned windows are the 128-lane chunks of half
    ``half`` of every token, in head-dimension order.
    """
    num_lanes = 128
    lanes_per_row = ref.shape[-1] // num_lanes
    chunks_per_token = ref.shape[0] // tile_n * lanes_per_row
    chunks_per_half = chunks_per_token // num_halves
    ref_flat = ref.reshape(-1, num_lanes)
    first = half * chunks_per_half
    return ref_flat, [
        pl.ds(first + c, tile_n, chunks_per_token) for c in range(chunks_per_half)
    ]


def load_from_acc_ref(ref, half, num_halves, tile_n, dtype=None):
    """Loads one output half of every token from the VMEM accumulator."""
    ref_flat, rows = _acc_half_rows(ref, half, num_halves, tile_n)
    vec = jnp.concat([ref_flat[r] for r in rows], axis=1)
    if dtype is not None:
        return pltpu.bitcast(vec, dtype)
    return vec


def store_to_acc_ref(ref, val, half, num_halves, tile_n):
    """Stores one output half of every token to the VMEM accumulator."""
    num_lanes = 128
    ref_flat, rows = _acc_half_rows(ref, half, num_halves, tile_n)
    for i, r in enumerate(rows):
        ref_flat[r] = val[:, i * num_lanes : (i + 1) * num_lanes]


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
    acc_ref,  # (tiles_per_group, tile_n * num_m * slots_per_m_tile, 1, 128)
    partial_ref,  # (tile_n, tile_m) f32: the k-partial sum of one step
    sem_ref,  # DMA semaphores [2]
    copies_ref,  # [2, 2]: (runs, accumulator) of the copy on each semaphore
    runs_ref,  # [2 * 2 * (tile_n + 1)]: see `run_stride`
    *,
    cfgs: Configs,
):
    """Inner kernel executing matmul, APE addition, and paged writes."""
    tile_m = cfgs.tile_sizes.tile_m
    tile_n = cfgs.tile_sizes.tile_n
    num_m = cfgs.num_m
    group_n = cfgs.tiles_per_group
    last_dim = acc_ref.shape[-1]
    # Cache rows holding one token's state: all its output halves.
    rows_per_token = num_m * (tile_m // last_dim)
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
    cache = csa_cache_layout.as_u8_slabs(cache)
    copies_ref[0, 0] = 0
    copies_ref[1, 0] = 0

    page_size = cfgs.token_page_size

    def _row_in_page(slot):
        """``slot % page_size`` for a non-negative slot."""
        if page_size & (page_size - 1) == 0:
            return slot & (page_size - 1)
        return slot % page_size

    # Per semaphore, the runs of the copy in flight: (first token, slot) of
    # each, then a sentinel first token. A flat SMEM array keeps the
    # per-token stores free of tiled address arithmetic.
    run_stride = tile_n + 1

    def _runs_base(sem_idx):
        return sem_idx * 2 * run_stride

    def _run_copy(sem_idx, buf_idx, r):
        """The copy of run ``r``: consecutive tokens whose states are
        consecutive rows of one cache page, written by one DMA."""
        base = _runs_base(sem_idx)
        first_token = runs_ref[base + r]
        slot = runs_ref[base + run_stride + r]
        rows = (runs_ref[base + r + 1] - first_token) * rows_per_token
        src = acc_ref_u8.at[buf_idx, pl.ds(first_token * rows_per_token, rows), :, :]
        dest = cache.at[slot // page_size, pl.ds(_row_in_page(slot), rows), :, :]
        return slot >= 0, pltpu.make_async_copy(src, dest, sem_ref.at[sem_idx])

    def _for_each_run(sem_idx, buf_idx, num_runs, fn):
        """Applies ``fn`` to the copy of every run of valid slots."""

        def body(r, carry):
            valid, copy = _run_copy(sem_idx, buf_idx, r)
            pl.when(valid)(lambda: fn(copy))
            return carry

        lax.fori_loop(0, num_runs, body, 0)

    def _wait_copy(sem_idx):
        """Waits for the copy in flight on ``sem_idx``, if any."""
        _for_each_run(
            sem_idx,
            copies_ref[sem_idx, 1],
            copies_ref[sem_idx, 0],
            lambda copy: copy.wait(),
        )
        copies_ref[sem_idx, 0] = 0

    def _copy_index(g_idx_val, n_idx_val):
        return g_idx_val * group_n + n_idx_val

    def _dma_copy_out(g_idx_val, n_idx_val):
        """Copies one finished token tile, all output halves, to the cache.

        Called on the step after the tile's last step, so the copy
        overlaps the next tile's matmul. A token whose slot continues the
        previous token's rows in the same page joins its run, so a prefill
        tile needs one DMA per page it touches rather than one per token and
        half; an invalid slot or a page change starts a new run. The two
        semaphores alternate between tiles, and a tile's copy waits for the
        copy two tiles back.
        """
        sem_idx = _copy_index(g_idx_val, n_idx_val) % 2
        _wait_copy(sem_idx)

        token_tile = _copy_index(g_idx_val, n_idx_val)
        slots = slot_mapping.at[pl.ds(token_tile * tile_n, tile_n)]
        base = _runs_base(sem_idx)
        # The first free entry. Each token is written there, and only a token
        # that starts a run keeps it.
        entry = base
        prev_slot = jnp.int32(-1)
        for i in range(tile_n):
            slot = slots[i]
            runs_ref[entry] = jnp.int32(i)
            runs_ref[entry + run_stride] = slot
            # A token joins the previous run if both slots are invalid, or if
            # its rows follow the previous token's in the same page (the row
            # test also fails for a valid slot after an invalid one).
            joins = ((slot & prev_slot) < 0) | (
                (slot == prev_slot + rows_per_token)
                & (_row_in_page(slot) >= rows_per_token)
            )
            if i > 0:
                entry = entry + jnp.logical_not(joins).astype(jnp.int32)
            else:
                entry = entry + 1
            prev_slot = slot
        num_runs = entry - base
        runs_ref[base + num_runs] = jnp.int32(tile_n)

        _for_each_run(sem_idx, n_idx_val, num_runs, lambda copy: copy.start())
        copies_ref[sem_idx, 0] = num_runs
        copies_ref[sem_idx, 1] = n_idx_val

    def inner_kernel(
        tiled_wgate_ref,  # (size_k, tile_m)
        tiled_hidden_states_ref,  # (tile_n, size_k)
        tiled_positions_ref,  # (tile_n,)
    ):
        g_idx = pl.program_id(0)
        m_idx = pl.program_id(1)
        n_idx = pl.program_id(2)
        # Every token tile of a group has its own accumulator, holding all
        # output halves of each token; m selects the half.
        acc = acc_ref.at[n_idx]

        state_width = cfgs.dims.size_m // 2

        def _matmul():
            """Sums the tile_k-wide partial products in k order.

            Each partial sum goes through ``partial_ref`` and is read back
            after the next product, then added to the product's popped f32
            result. The accumulation order within a product is the
            compiler's, so the sum is not bit-identical to one k per step.
            """
            tile_k = cfgs.tile_sizes.tile_k
            for k in range(cfgs.num_k):
                ks = pl.ds(k * tile_k, tile_k)
                prod = lax.dot_general(
                    tiled_hidden_states_ref[:, ks],
                    tiled_wgate_ref[ks, :],
                    (((1,), (0,)), ((), (()))),
                    preferred_element_type=jnp.float32,
                )
                acc_val = prod if k == 0 else partial_ref[...] + prod
                if k < cfgs.num_k - 1:
                    partial_ref[...] = acc_val
            return acc_val

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
                acc, m_idx, num_m, tile_n, dtype=jnp.float32
            )
            new_acc = existing_acc + ape_selected  # (tile_n, state_width)
            store_to_acc_ref(acc, new_acc, m_idx, num_m, tile_n)

        def process(copy_dma: bool, add_ape: bool):
            # 1. The previous group's copies still in flight read the last
            # two accumulators; wait for one before overwriting it.
            @pl.when((g_idx > 0) & (m_idx == 0) & (n_idx >= group_n - 2))
            def _free_acc():
                _wait_copy(_copy_index(g_idx - 1, n_idx) % 2)

            # 2. Launch DMA for the PREVIOUS token tile
            if copy_dma:
                first_n = n_idx == 0
                prev_n_idx = lax.select(first_n, group_n - 1, n_idx - 1)
                prev_g_idx = lax.select(first_n, g_idx - 1, g_idx)
                _dma_copy_out(prev_g_idx, prev_n_idx)

            # 3. Matmul over the whole hidden dimension
            store_to_acc_ref(acc, _matmul(), m_idx, num_m, tile_n)

            # 4. APE addition
            if add_ape:
                _add_ape()

        # Select and execute
        is_score = m_idx * tile_m >= state_width

        # Token tile (g, n) finishes at step (g, num_m - 1, n); the next grid
        # step is (g, num_m - 1, n + 1), or (g + 1, 0, 0) after the last n.
        copy_dma = ((m_idx == num_m - 1) & (n_idx > 0)) | (
            (g_idx > 0) & (m_idx == 0) & (n_idx == 0)
        )
        add_ape = is_score

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

    # n innermost: the weight block stays resident across a group's token
    # tiles, so each weight tile is read from HBM once per group. A step
    # covers the whole hidden dimension, so a token tile's half is one step.
    wkv_spec = pl.BlockSpec(
        (cfgs.dims.size_k, cfgs.tile_sizes.tile_m),
        lambda g, m, n: (0, m),
        memory_space=pltpu.VMEM,
    )
    hidden_states_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n, cfgs.dims.size_k),
        lambda g, m, n: (g * group_n + n, 0),
        memory_space=pltpu.VMEM,
    )
    positions_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n,),
        lambda g, m, n: (g * group_n + n,),
        memory_space=pltpu.VMEM,
    )

    pipeline_fn = pltpu.emit_pipeline(
        inner_kernel,
        grid=(cfgs.num_groups, cfgs.num_m, group_n),
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
    _dma_copy_out(cfgs.num_groups - 1, group_n - 1)
    _wait_copy(0)
    _wait_copy(1)


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

    token_page_size = cache.shape[1]

    state_width = state_dim // 2
    tile_m = state_width
    tile_k = _select_tile_k(hidden_size)

    # The token dimension must be an exact multiple of `tile_n`.
    sublane_tile = 8
    tile_n = min(128, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
    num_n = pl.cdiv(num_tokens, tile_n)
    # Equal groups of at most GROUP_TOKENS, padded to whole groups. A group
    # spans at least two tiles, which the accumulator reuse relies on.
    num_groups = pl.cdiv(num_n * tile_n, max(GROUP_TOKENS, 2 * tile_n))
    tiles_per_group = pl.cdiv(num_n, num_groups)
    padded_num_tokens = num_groups * tiles_per_group * tile_n
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
        token_page_size=token_page_size,
        tiles_per_group=tiles_per_group,
    )

    last_dim = cache.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    acc_rows = tile_n * cfgs.num_m * slots_per_m_tile
    scratch_shapes = [
        pltpu.VMEM((tiles_per_group, acc_rows, 1, last_dim), jnp.float32),
        pltpu.VMEM((tile_n, tile_m), jnp.float32),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SMEM((2, 2), jnp.int32),
        pltpu.SMEM((2 * 2 * (tile_n + 1),), jnp.int32),
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
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="proj_and_save_state",
    )(slot_mapping, positions, hidden_states, wkv_wgate, ape, cache)
