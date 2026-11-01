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

# Tokens whose f32 state is accumulated, and whose hidden states are held,
# in VMEM at once.
GROUP_TOKENS = 1024
# The whole weight stays in VMEM: 28 MiB for DeepSeek-V4-Pro's CSA
# compressor, past the default scoped limit.
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024
# Most tokens per tile: the rows one grid step streams through the MXU.
TILE_N = 256
# Token tiles of the first group in flight ahead of the step that reads them.
HIDDEN_FETCH_AHEAD = 2
# Compress ratios up to this pick each token's APE row from a table that
# the one-hot APE product computes once per call; larger ratios run that
# product on every token tile.
APE_TABLE_MAX_RATIO = 8


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


def store_to_acc_ref(ref, val, half, num_halves, tile_n):
    """Stores one output half of every token to the VMEM accumulator."""
    num_lanes = 128
    ref_flat, rows = _acc_half_rows(ref, half, num_halves, tile_n)
    for i, r in enumerate(rows):
        ref_flat[r] = val[:, i * num_lanes : (i + 1) * num_lanes]


def kernel(
    # scalar prefetch
    slot_mapping,
    run_ends,  # [num_tokens]: see `_plan_runs`
    # input
    positions,  # [num_tokens]
    hidden_states,  # [num_tokens, hidden_size]
    wkv_wgate,  # [2 * state_width, hidden_size]
    ape,  # [compress_ratio, state_width]
    cache,  # [num_pages, page_size, 4, 128] uint8 or [..., 128] int32 (aliased)
    # output
    _,  # unused output ref (aliased to cache)
    # scratch_memory
    weight_ref,  # (num_m, hidden_size, tile_m): the whole weight, read once
    hidden_ref,  # (tiles_per_group, tile_n, hidden_size): a group's tokens
    acc_ref,  # (tiles_per_group, tile_n * num_m * slots_per_m_tile, 1, 128)
    partial_ref,  # (tile_n, tile_m) f32: the k-partial sum of one step
    ape_table_ref,  # (ape_table_rows, state_width) f32: row r is APE row r % ratio
    in_sems,  # DMA semaphores: weight halves, then token slots
    sem_ref,  # DMA semaphores [2]
    copies_ref,  # [2, 2]: (token tile or -1, accumulator) of each semaphore's copy
    *,
    cfgs: Configs,
):
    """Inner kernel executing matmul, APE addition, and paged writes."""
    tile_m = cfgs.tile_sizes.tile_m
    tile_n = cfgs.tile_sizes.tile_n
    num_m = cfgs.num_m
    group_n = cfgs.tiles_per_group
    num_groups = cfgs.num_groups
    last_dim = acc_ref.shape[-1]
    # Cache rows holding one token's state: all its output halves.
    rows_per_token = num_m * (tile_m // last_dim)
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
    cache = csa_cache_layout.as_u8_slabs(cache)
    copies_ref[0, 0] = -1
    copies_ref[1, 0] = -1

    def _weight_copy(m):
        """The copy of output half ``m`` of the weight."""
        return pltpu.make_async_copy(
            wkv_wgate.at[:, pl.ds(m * tile_m, tile_m)], weight_ref.at[m], in_sems.at[m]
        )

    def _hidden_copy(token_tile, slot):
        """The copy of token tile ``token_tile`` into group slot ``slot``."""
        rows = pl.ds(pl.multiple_of(token_tile * tile_n, tile_n), tile_n)
        return pltpu.make_async_copy(
            hidden_states.at[rows], hidden_ref.at[slot], in_sems.at[num_m + slot]
        )

    # The first step's operands; the rest stream behind the matmul
    # (`_fetch_ahead`), so only the first tile and weight half are exposed.
    _hidden_copy(0, 0).start()
    _weight_copy(0).start()

    ratio = cfgs.dims.compress_ratio

    def _one_hot_ape(ape_rows):
        """APE row ``ape_rows[i]`` for every i, as a one-hot product."""
        ape_val = ape[...].astype(jnp.float32)
        one_hot = (ape_rows[:, None] == jnp.arange(ratio)[None, :]).astype(jnp.float32)
        return jnp.matmul(
            one_hot, ape_val, precision=lax.Precision.HIGHEST
        )  #  (len(ape_rows), state_width)

    # An output row of the one-hot product depends on its own one-hot row
    # only, so the table holds bit for bit the row a tile's product would.
    use_ape_table = ratio <= APE_TABLE_MAX_RATIO
    if use_ape_table:
        table_rows = ape_table_ref.shape[0]
        ape_table_ref[...] = _one_hot_ape(
            lax.broadcasted_iota(jnp.int32, (table_rows,), 0) % ratio
        )

    page_size = cfgs.token_page_size

    def _row_in_page(slot):
        """``slot % page_size`` for a non-negative slot."""
        if page_size & (page_size - 1) == 0:
            return slot & (page_size - 1)
        return slot % page_size

    def _run_copy(sem_idx, token_tile, buf_idx, first_token):
        """The copy of the run of ``token_tile`` starting at ``first_token``
        (consecutive tokens whose states are consecutive rows of one cache
        page, written by one DMA), and the run's end."""
        token = token_tile * tile_n + first_token
        end = run_ends[token]
        slot = slot_mapping[token]
        rows = (end - first_token) * rows_per_token
        src = acc_ref_u8.at[buf_idx, pl.ds(first_token * rows_per_token, rows), :, :]
        dest = cache.at[slot // page_size, pl.ds(_row_in_page(slot), rows), :, :]
        return end, slot >= 0, pltpu.make_async_copy(src, dest, sem_ref.at[sem_idx])

    def _for_each_run(sem_idx, token_tile, buf_idx, fn):
        """Applies ``fn`` to the copy of every run of valid slots."""

        def body(first_token):
            end, valid, copy = _run_copy(sem_idx, token_tile, buf_idx, first_token)
            pl.when(valid)(lambda: fn(copy))
            return end

        lax.while_loop(lambda first_token: first_token < tile_n, body, jnp.int32(0))

    def _wait_copy(sem_idx):
        """Waits for the copy in flight on ``sem_idx``, if any."""
        token_tile = copies_ref[sem_idx, 0]

        @pl.when(token_tile >= 0)
        def _():
            _for_each_run(
                sem_idx, token_tile, copies_ref[sem_idx, 1], lambda copy: copy.wait()
            )

        copies_ref[sem_idx, 0] = -1

    def _copy_index(g_idx_val, n_idx_val):
        return g_idx_val * group_n + n_idx_val

    def _dma_copy_out(g_idx_val, n_idx_val):
        """Copies one finished token tile, all output halves, to the cache.

        Called on the step after the tile's last step, so the copy
        overlaps the next tile's matmul. A prefill tile needs one DMA per
        page it touches rather than one per token and half (`_plan_runs`).
        The two semaphores alternate between tiles, and a tile's copy waits
        for the copy two tiles back.
        """
        sem_idx = _copy_index(g_idx_val, n_idx_val) % 2
        _wait_copy(sem_idx)

        token_tile = _copy_index(g_idx_val, n_idx_val)
        _for_each_run(sem_idx, token_tile, n_idx_val, lambda copy: copy.start())
        copies_ref[sem_idx, 0] = token_tile
        copies_ref[sem_idx, 1] = n_idx_val

    def _fetch_ahead(g_idx, m_idx, n_idx):
        """Waits for step (g, m, n)'s inputs and starts later steps' copies.

        The weight is read once per call: half m + 1 is fetched behind half
        m's first step. A group's token tiles are read once, on its first
        half, then stay resident for the other half; the first group's
        tiles are fetched HIDDEN_FETCH_AHEAD tiles ahead of their step, and
        a later group's tile replaces the previous group's in the step after
        the previous one's last use (`_dma_copy_out`'s step).
        """
        token_tile = _copy_index(g_idx, n_idx)

        @pl.when(m_idx == 0)
        def _():
            _hidden_copy(token_tile, n_idx).wait()

        for m in range(num_m):

            @pl.when((g_idx == 0) & (m_idx == m) & (n_idx == 0))
            def _():
                _weight_copy(m).wait()
                if m + 1 < num_m:
                    _weight_copy(m + 1).start()

        first_tiles = (g_idx == 0) & (m_idx == 0)

        @pl.when(first_tiles & (n_idx == 0))
        def _():
            for t in range(1, min(HIDDEN_FETCH_AHEAD, group_n - 1) + 1):
                _hidden_copy(t, t).start()

        ahead = n_idx + HIDDEN_FETCH_AHEAD

        @pl.when(first_tiles & (n_idx > 0) & (ahead < group_n))
        def _():
            _hidden_copy(ahead, ahead).start()

    def inner_kernel(g_idx, m_idx, n_idx):
        # Every token tile of a group has its own accumulator, holding all
        # output halves of each token; m selects the half.
        acc = acc_ref.at[n_idx]
        tiled_hidden_states_ref = hidden_ref.at[n_idx]  # (tile_n, size_k)
        tiled_wgate_ref = weight_ref.at[m_idx]  # (size_k, tile_m)
        token_tile = _copy_index(g_idx, n_idx)

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

        def _ape_rows():
            # A tile narrower than TILE_N is the call's only tile.
            first = 0
            if cfgs.num_n > 1:
                first = pl.multiple_of(token_tile * tile_n, tile_n)
            ape_rows = positions[pl.ds(first, tile_n)] % ratio
            if not use_ape_table:
                return _one_hot_ape(ape_rows)
            ape_selected = jnp.broadcast_to(
                ape_table_ref[ratio - 1 : ratio, :], (tile_n, tile_m)
            )
            for r in range(ratio - 2, -1, -1):
                ape_selected = jnp.where(
                    ape_rows[:, None] == r, ape_table_ref[r : r + 1, :], ape_selected
                )
            return ape_selected

        def process(copy_dma: bool, add_ape: bool):
            # 1. The previous group's copies still in flight read the last
            # two accumulators; wait for one before overwriting it.
            @pl.when((g_idx > 0) & (m_idx == 0) & (n_idx >= group_n - 2))
            def _free_acc():
                _wait_copy(_copy_index(g_idx - 1, n_idx) % 2)

            # 2. Launch DMA for the PREVIOUS token tile, and refill its
            # token slot with the next group's tile.
            if copy_dma:
                first_n = n_idx == 0
                prev_n_idx = lax.select(first_n, group_n - 1, n_idx - 1)
                prev_g_idx = lax.select(first_n, g_idx - 1, g_idx)
                _dma_copy_out(prev_g_idx, prev_n_idx)

                @pl.when(prev_g_idx + 1 < num_groups)
                def _():
                    next_tile = _copy_index(prev_g_idx + 1, prev_n_idx)
                    _hidden_copy(next_tile, prev_n_idx).start()

            # 3. Matmul over the whole hidden dimension, plus the APE rows
            # on the score half, stored once. The APE product is traced
            # first: a sum defined before it would be folded into its MXU
            # accumulation (a Mosaic scheduling heuristic, not a guarantee).
            ape_selected = _ape_rows() if add_ape else None
            new_acc = _matmul()
            if add_ape:
                new_acc = new_acc + ape_selected
            store_to_acc_ref(acc, new_acc, m_idx, num_m, tile_n)

        _fetch_ahead(g_idx, m_idx, n_idx)

        # Select and execute
        is_score = m_idx * tile_m >= state_width

        # Token tile (g, n) finishes at step (g, num_m - 1, n); the next
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

    # n innermost: a weight half serves every token tile of a group in a
    # row, and a token tile serves both halves from VMEM.
    def step(i, carry):
        g_idx = i // (num_m * group_n)
        m_idx = i // group_n % num_m
        n_idx = i % group_n
        inner_kernel(g_idx, m_idx, n_idx)
        return carry

    lax.fori_loop(0, num_groups * num_m * group_n, step, 0)

    # Epilogue: launch DMA for the last tile and wait for all outstanding DMAs
    _dma_copy_out(num_groups - 1, group_n - 1)
    _wait_copy(0)
    _wait_copy(1)


def _plan_runs(slot_mapping, tile_n, rows_per_token, page_size):
    """The end of the DMA run starting at each token, tile-relative.

    A token joins the previous token's run if both slots are invalid, or if
    its rows follow the previous token's in the same page (the row test also
    fails for a valid slot after an invalid one); a tile's first token
    starts a run. A page holds at most ``page_size // rows_per_token``
    tokens, so a valid run never spans more; an invalid run is cut there
    too, which keeps the walk to one elementwise pass of bounded width.
    Only the entries at run starts are read.
    """
    num_tokens = slot_mapping.shape[0]
    max_run = min(tile_n, page_size // rows_per_token)
    token = lax.broadcasted_iota(jnp.int32, (num_tokens,), 0)
    padded = jnp.pad(slot_mapping, (1, max_run), constant_values=-1)

    def joins(d):
        """Whether token ``i + d`` joins token ``i + d - 1``'s run."""
        slot = padded[1 + d : 1 + d + num_tokens]
        prev = padded[d : d + num_tokens]
        return (
            ((slot & prev) < 0)
            | ((slot == prev + rows_per_token) & (slot % page_size >= rows_per_token))
        ) & ((token + d) % tile_n != 0)

    # Recomputing `joins` per offset keeps the whole walk one fusion.
    run_end = token % tile_n + 1
    alive = jnp.ones((num_tokens,), jnp.bool_)
    for d in range(1, max_run):
        alive = alive & joins(d)
        run_end = run_end + alive.astype(jnp.int32)
    return run_end


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
    tile_n = min(TILE_N, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
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

    ape_table_rows = sublane_tile * pl.cdiv(
        min(compress_ratio, APE_TABLE_MAX_RATIO), sublane_tile
    )
    last_dim = cache.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    acc_rows = tile_n * cfgs.num_m * slots_per_m_tile
    scratch_shapes = [
        pltpu.VMEM((cfgs.num_m, hidden_size, tile_m), wkv_wgate.dtype),
        pltpu.VMEM((tiles_per_group, tile_n, hidden_size), hidden_states.dtype),
        pltpu.VMEM((tiles_per_group, acc_rows, 1, last_dim), jnp.float32),
        pltpu.VMEM((tile_n, tile_m), jnp.float32),
        pltpu.VMEM((ape_table_rows, state_width), jnp.float32),
        pltpu.SemaphoreType.DMA((cfgs.num_m + tiles_per_group,)),
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SMEM((2, 2), jnp.int32),
    ]
    run_ends = _plan_runs(slot_mapping, tile_n, state_dim // last_dim, token_page_size)

    out_shape = jax.ShapeDtypeStruct(cache.shape, cache.dtype)

    return pl.pallas_call(
        functools.partial(kernel, cfgs=cfgs),
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),  # positions
                pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states
                pl.BlockSpec(memory_space=pltpu.HBM),  # wkv_wgate
                pl.BlockSpec(memory_space=pltpu.VMEM),  # ape (prefetch to VMEM)
                pl.BlockSpec(memory_space=pltpu.HBM),  # cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        input_output_aliases={6: 0},  # Alias output to cache (index 6)
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="proj_and_save_state",
    )(slot_mapping, run_ends, positions, hidden_states, wkv_wgate, ape, cache)
