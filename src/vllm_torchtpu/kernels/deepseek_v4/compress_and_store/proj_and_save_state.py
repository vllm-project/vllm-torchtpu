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
from typing import NamedTuple

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


class _Projection:
    """One projection of a `kernel` call: its operands and per-step work.

    Output half m of this projection runs on the call's steps
    (g, first_half + m, n), one per token tile n of group g.
    """

    def __init__(self, cfgs, hbm, scratch, hidden_ref, positions):
        self.cfgs = cfgs
        (
            self.slot_mapping_hbm,  # [num_tokens]
            self.run_ends_hbm,  # [num_tokens]: see `_plan_runs`
            self.wkv_wgate,  # [hidden_size, 2 * state_width]
            self.ape_hbm,  # [compress_ratio, state_width]
            cache,  # [num_pages, page_size, 4, 128] uint8 / [..., 128] int32
        ) = hbm
        (
            self.slot_mapping,  # SMEM copy of slot_mapping_hbm
            self.run_ends,  # SMEM copy of run_ends_hbm
            self.ape,  # VMEM copy of ape_hbm
            self.weight_ref,  # (num_m, hidden_size, tile_m): read once
            self.acc_ref,  # (tiles_per_group, tile_n * rows_per_token, 1, lanes)
            self.partial_ref,  # (tile_n, tile_m) f32: k-partial sum of a step
            self.ape_table_ref,  # (rows, state_width) f32: row r is APE row r % ratio
            self.sem_ref,  # DMA semaphores [2] of the state copies
            self.copies_ref,  # [2, 2]: (token tile or -1, accumulator) per semaphore
        ) = scratch
        self.hidden_ref = hidden_ref
        self.positions = positions
        # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
        self.cache = csa_cache_layout.as_u8_slabs(cache)
        self.acc_ref_u8 = self.acc_ref.bitcast(jnp.uint8)
        tile_m = cfgs.tile_sizes.tile_m
        # Cache rows holding one token's state: all its output halves.
        self.rows_per_token = cfgs.num_m * (tile_m // self.acc_ref.shape[-1])
        self.ratio = cfgs.dims.compress_ratio
        self.use_ape_table = self.ratio <= APE_TABLE_MAX_RATIO
        self.copies_ref[0, 0] = -1
        self.copies_ref[1, 0] = -1

    def small_copies(self, sem):
        return [
            pltpu.make_async_copy(src, dst, sem)
            for src, dst in (
                (self.slot_mapping_hbm, self.slot_mapping),
                (self.run_ends_hbm, self.run_ends),
                (self.ape_hbm, self.ape),
            )
        ]

    def weight_copy(self, m, sem):
        """The copy of output half ``m`` of the weight."""
        tile_m = self.cfgs.tile_sizes.tile_m
        return pltpu.make_async_copy(
            self.wkv_wgate.at[:, pl.ds(m * tile_m, tile_m)], self.weight_ref.at[m], sem
        )

    def _one_hot_ape(self, ape_rows):
        """APE row ``ape_rows[i]`` for every i, as a one-hot product."""
        ape_val = self.ape[...].astype(jnp.float32)
        one_hot = (ape_rows[:, None] == jnp.arange(self.ratio)[None, :]).astype(
            jnp.float32
        )
        return jnp.matmul(
            one_hot, ape_val, precision=lax.Precision.HIGHEST
        )  #  (len(ape_rows), state_width)

    def fill_ape_table(self):
        # An output row of the one-hot product depends on its own one-hot row
        # only, so the table holds bit for bit the row a tile's product would.
        if self.use_ape_table:
            table_rows = self.ape_table_ref.shape[0]
            self.ape_table_ref[...] = self._one_hot_ape(
                lax.broadcasted_iota(jnp.int32, (table_rows,), 0) % self.ratio
            )

    def _row_in_page(self, slot):
        """``slot % page_size`` for a non-negative slot."""
        page_size = self.cfgs.token_page_size
        if page_size & (page_size - 1) == 0:
            return slot & (page_size - 1)
        return slot % page_size

    def _run_copy(self, sem_idx, token_tile, buf_idx, first_token):
        """The copy of the run of ``token_tile`` starting at ``first_token``
        (consecutive tokens whose states are consecutive rows of one cache
        page, written by one DMA), and the run's end."""
        tile_n = self.cfgs.tile_sizes.tile_n
        rows_per_token = self.rows_per_token
        token = token_tile * tile_n + first_token
        end = self.run_ends[token]
        slot = self.slot_mapping[token]
        rows = (end - first_token) * rows_per_token
        src = self.acc_ref_u8.at[
            buf_idx, pl.ds(first_token * rows_per_token, rows), :, :
        ]
        dest = self.cache.at[
            slot // self.cfgs.token_page_size,
            pl.ds(self._row_in_page(slot), rows),
            :,
            :,
        ]
        return (
            end,
            slot >= 0,
            pltpu.make_async_copy(src, dest, self.sem_ref.at[sem_idx]),
        )

    def _for_each_run(self, sem_idx, token_tile, buf_idx, fn):
        """Applies ``fn`` to the copy of every run of valid slots."""

        def body(first_token):
            end, valid, copy = self._run_copy(sem_idx, token_tile, buf_idx, first_token)
            pl.when(valid)(lambda: fn(copy))
            return end

        lax.while_loop(
            lambda first_token: first_token < self.cfgs.tile_sizes.tile_n,
            body,
            jnp.int32(0),
        )

    def wait_copy(self, sem_idx):
        """Waits for the copy in flight on ``sem_idx``, if any."""
        token_tile = self.copies_ref[sem_idx, 0]

        @pl.when(token_tile >= 0)
        def _():
            self._for_each_run(
                sem_idx,
                token_tile,
                self.copies_ref[sem_idx, 1],
                lambda copy: copy.wait(),
            )

        self.copies_ref[sem_idx, 0] = -1

    def dma_copy_out(self, g_idx_val, n_idx_val):
        """Copies one finished token tile, all output halves, to the cache.

        Called on the step after the tile's last step, so the copy
        overlaps the next step's matmul. A prefill tile needs one DMA per
        page it touches rather than one per token and half (`_plan_runs`).
        The two semaphores alternate between tiles, and a tile's copy waits
        for the copy two tiles back.
        """
        token_tile = g_idx_val * self.cfgs.tiles_per_group + n_idx_val
        sem_idx = token_tile % 2
        self.wait_copy(sem_idx)
        self._for_each_run(sem_idx, token_tile, n_idx_val, lambda copy: copy.start())
        self.copies_ref[sem_idx, 0] = token_tile
        self.copies_ref[sem_idx, 1] = n_idx_val

    def free_acc(self, g_idx, m_idx, n_idx):
        """Before a group's first write to an accumulator, waits for the
        previous group's copy still reading it (one of the last two)."""
        group_n = self.cfgs.tiles_per_group

        @pl.when((g_idx > 0) & (m_idx == 0) & (n_idx >= group_n - 2))
        def _():
            self.wait_copy(((g_idx - 1) * group_n + n_idx) % 2)

    def _matmul(self, n_idx, m_idx):
        """Sums the tile_k-wide partial products in k order.

        Each partial sum goes through ``partial_ref`` and is read back
        after the next product, then added to the product's popped f32
        result. The accumulation order within a product is the
        compiler's, so the sum is not bit-identical to one k per step.
        """
        hidden = self.hidden_ref.at[n_idx]  # (tile_n, size_k)
        weight = self.weight_ref.at[m_idx]  # (size_k, tile_m)
        tile_k = self.cfgs.tile_sizes.tile_k
        for k in range(self.cfgs.num_k):
            ks = pl.ds(k * tile_k, tile_k)
            prod = lax.dot_general(
                hidden[:, ks],
                weight[ks, :],
                (((1,), (0,)), ((), (()))),
                preferred_element_type=jnp.float32,
            )
            acc_val = prod if k == 0 else self.partial_ref[...] + prod
            if k < self.cfgs.num_k - 1:
                self.partial_ref[...] = acc_val
        return acc_val

    def _ape_rows(self, token_tile):
        tile_n = self.cfgs.tile_sizes.tile_n
        tile_m = self.cfgs.tile_sizes.tile_m
        ratio = self.ratio
        # A tile narrower than TILE_N is the call's only tile.
        first = 0
        if self.cfgs.num_n > 1:
            first = pl.multiple_of(token_tile * tile_n, tile_n)
        ape_rows = self.positions[pl.ds(first, tile_n)] % ratio
        if not self.use_ape_table:
            return self._one_hot_ape(ape_rows)
        table = self.ape_table_ref
        ape_selected = jnp.broadcast_to(table[ratio - 1 : ratio, :], (tile_n, tile_m))
        for r in range(ratio - 2, -1, -1):
            ape_selected = jnp.where(
                ape_rows[:, None] == r, table[r : r + 1, :], ape_selected
            )
        return ape_selected

    def compute(self, g_idx, m_idx, n_idx, add_ape):
        """Matmul over the whole hidden dimension, plus the APE rows on the
        score half, stored once to token tile n's accumulator."""
        tile_n = self.cfgs.tile_sizes.tile_n
        # The APE product is traced first: a sum defined before it would be
        # folded into its MXU accumulation (a Mosaic scheduling heuristic,
        # not a guarantee).
        token_tile = g_idx * self.cfgs.tiles_per_group + n_idx
        ape_selected = self._ape_rows(token_tile) if add_ape else None
        new_acc = self._matmul(n_idx, m_idx)
        if add_ape:
            new_acc = new_acc + ape_selected
        store_to_acc_ref(
            self.acc_ref.at[n_idx], new_acc, m_idx, self.cfgs.num_m, tile_n
        )


def kernel(
    # input
    positions_hbm,  # [num_tokens]
    hidden_states,  # [num_tokens, hidden_size]
    projection_hbm,  # per projection: the HBM operands `_Projection` takes
    # One unused output ref per projection (aliased to its cache), then the
    # scratch memory: the VMEM copy of positions_hbm, a group's token tiles
    # (tiles_per_group, tile_n, hidden_size), the DMA semaphores of every
    # weight half, token slot and the small inputs, and per projection the
    # scratch `_Projection` takes.
    *outputs_and_scratch,
    cfgs: tuple[Configs, ...],
):
    """Inner kernel executing matmul, APE addition, and paged writes.

    Every projection reads the same token tiles; its output halves follow
    the previous projection's, so a group's tiles are fetched once for all
    of them.
    """
    positions, hidden_ref, in_sems, projection_scratch = outputs_and_scratch[
        len(cfgs) :
    ]
    shared = cfgs[0]
    tile_n = shared.tile_sizes.tile_n
    group_n = shared.tiles_per_group
    num_groups = shared.num_groups
    first_halves = [sum(c.num_m for c in cfgs[:p]) for p in range(len(cfgs))]
    num_halves = sum(c.num_m for c in cfgs)
    projections = [
        _Projection(c, hbm, scratch, hidden_ref, positions)
        for c, hbm, scratch in zip(cfgs, projection_hbm, projection_scratch)
    ]
    # (projection, its output half) of every half of the call, in step order.
    halves = [(proj, m) for proj in projections for m in range(proj.cfgs.num_m)]

    def _weight_copy(half):
        proj, m = halves[half]
        return proj.weight_copy(m, in_sems.at[half])

    def _hidden_copy(token_tile, slot):
        """The copy of token tile ``token_tile`` into group slot ``slot``."""
        rows = pl.ds(pl.multiple_of(token_tile * tile_n, tile_n), tile_n)
        return pltpu.make_async_copy(
            hidden_states.at[rows], hidden_ref.at[slot], in_sems.at[num_halves + slot]
        )

    small_sem = in_sems.at[num_halves + group_n]
    small_copies = [pltpu.make_async_copy(positions_hbm, positions, small_sem)]
    for proj in projections:
        small_copies += proj.small_copies(small_sem)
    # The first step's operands go first; the rest stream behind the matmul
    # (`_fetch_ahead`), so only the first tile and weight half are exposed.
    _hidden_copy(0, 0).start()
    _weight_copy(0).start()
    for copy in small_copies:
        copy.start()
    for copy in small_copies:
        copy.wait()

    for proj in projections:
        proj.fill_ape_table()

    def _copy_index(g_idx_val, n_idx_val):
        return g_idx_val * group_n + n_idx_val

    def _finish_tile(proj, g_idx_val, n_idx_val):
        """Copies out ``proj``'s finished tile (g, n); after the last
        projection's, refills the tile's token slot with the next group's."""
        proj.dma_copy_out(g_idx_val, n_idx_val)
        if proj is projections[-1]:

            @pl.when(g_idx_val + 1 < num_groups)
            def _():
                next_tile = _copy_index(g_idx_val + 1, n_idx_val)
                _hidden_copy(next_tile, n_idx_val).start()

    def _fetch_ahead(g_idx, half_idx, n_idx):
        """Waits for step (g, half, n)'s inputs and starts later steps' copies.

        The weight is read once per call: half h + 1 is fetched behind half
        h's first step. A group's token tiles are read once, on its first
        half, then stay resident for the other halves; the first group's
        tiles are fetched HIDDEN_FETCH_AHEAD tiles ahead of their step, and
        a later group's tile replaces the previous group's in the step after
        the previous one's last use (`_finish_tile`).
        """
        token_tile = _copy_index(g_idx, n_idx)

        @pl.when(half_idx == 0)
        def _():
            _hidden_copy(token_tile, n_idx).wait()

        for half in range(num_halves):

            @pl.when((g_idx == 0) & (half_idx == half) & (n_idx == 0))
            def _(*, half=half):
                _weight_copy(half).wait()
                if half + 1 < num_halves:
                    _weight_copy(half + 1).start()

        first_tiles = (g_idx == 0) & (half_idx == 0)

        @pl.when(first_tiles & (n_idx == 0))
        def _():
            for t in range(1, min(HIDDEN_FETCH_AHEAD, group_n - 1) + 1):
                _hidden_copy(t, t).start()

        ahead = n_idx + HIDDEN_FETCH_AHEAD

        @pl.when(first_tiles & (n_idx > 0) & (ahead < group_n))
        def _():
            _hidden_copy(ahead, ahead).start()

    def projection_step(p, g_idx, m_idx, n_idx):
        proj = projections[p]
        num_m = proj.cfgs.num_m
        # The tile finished on the previous step: this projection's previous
        # tile, or after a group's last tile, the previous projection's last
        # tile (the previous group's, for the first projection).
        prev = projections[p - 1]
        prev_g_on_first_n = g_idx - 1 if p == 0 else g_idx

        def process(copy_dma: bool, add_ape: bool):
            # 1. The previous group's copies still in flight read the last
            # two accumulators; wait for one before overwriting it.
            proj.free_acc(g_idx, m_idx, n_idx)

            # 2. Launch DMA for the PREVIOUS token tile.
            if copy_dma:
                first_n = n_idx == 0
                prev_n_idx = lax.select(first_n, group_n - 1, n_idx - 1)
                if prev is proj:
                    prev_g_idx = lax.select(first_n, prev_g_on_first_n, g_idx)
                    _finish_tile(proj, prev_g_idx, prev_n_idx)
                else:
                    pl.when(first_n)(
                        lambda: _finish_tile(prev, prev_g_on_first_n, prev_n_idx)
                    )
                    pl.when(jnp.logical_not(first_n))(
                        lambda: _finish_tile(proj, g_idx, prev_n_idx)
                    )

            # 3. Matmul and store.
            proj.compute(g_idx, m_idx, n_idx, add_ape)

        # Select and execute
        state_width = proj.cfgs.dims.size_m // 2
        is_score = m_idx * proj.cfgs.tile_sizes.tile_m >= state_width

        # Token tile (g, n) finishes at step (g, last half, n); the next
        # step is (g, last half, n + 1), or the next projection's (or
        # group's) first step after the last n.
        has_prev_on_first_n = (g_idx > 0) if p == 0 else True
        copy_dma = ((m_idx == num_m - 1) & (n_idx > 0)) | (
            has_prev_on_first_n & (m_idx == 0) & (n_idx == 0)
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

    def inner_kernel(g_idx, half_idx, n_idx):
        _fetch_ahead(g_idx, half_idx, n_idx)
        if len(projections) == 1:
            projection_step(0, g_idx, half_idx, n_idx)
            return
        for p, first in enumerate(first_halves):
            last = first + projections[p].cfgs.num_m

            @pl.when((half_idx >= first) & (half_idx < last))
            def _(*, p=p, first=first):
                projection_step(p, g_idx, half_idx - first, n_idx)

    # n innermost: a weight half serves every token tile of a group in a
    # row, and a token tile serves all halves from VMEM.
    def step(i, carry):
        g_idx = i // (num_halves * group_n)
        half_idx = i // group_n % num_halves
        n_idx = i % group_n
        inner_kernel(g_idx, half_idx, n_idx)
        return carry

    lax.fori_loop(0, num_groups * num_halves * group_n, step, 0)

    # Epilogue: launch DMA for the last tile and wait for all outstanding DMAs
    _finish_tile(projections[-1], num_groups - 1, group_n - 1)
    for proj in projections:
        proj.wait_copy(0)
        proj.wait_copy(1)


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


class StateProjection(NamedTuple):
    """One state projection of the hidden states, and where it is saved."""

    wkv_wgate: jax.Array  # [hidden_size, 2 * state_width]
    ape: jax.Array  # [compress_ratio, state_width]
    slot_mapping: jax.Array  # [num_tokens]
    cache: jax.Array  # [num_pages, page_size, 4, 128] uint8 / [..., 128] int32


@functools.partial(
    jax.jit,
    static_argnames=[
        "compress_ratios",
        "interpret",
    ],
)
def proj_and_save_states(
    hidden_states: jax.Array,  # [num_tokens, hidden_size]
    positions: jax.Array,  # [num_tokens]
    projections: tuple[StateProjection, ...],
    compress_ratios: tuple[int, ...],
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """`proj_and_save_state` for several projections of the same tokens.

    One kernel reads each token tile once for all projections, and pays
    one call's fixed cost. The first projection's first weight half is the
    call's only exposed weight copy, so the smallest weight goes first.

    Returns:
      Each projection's updated cache.
    """
    num_tokens, hidden_size = hidden_states.shape
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
    pad = padded_num_tokens - num_tokens
    if pad:
        hidden_states = jnp.pad(hidden_states, ((0, pad), (0, 0)))
        positions = jnp.pad(positions, (0, pad))

    cfgs = []
    projection_operands = []
    projection_scratch = []
    for proj, compress_ratio in zip(projections, compress_ratios, strict=True):
        _, state_dim = proj.wkv_wgate.shape
        state_width = state_dim // 2
        tile_m = state_width
        token_page_size = proj.cache.shape[1]
        cfg = Configs(
            tile_sizes=TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n),
            dims=Dimensions(
                size_m=state_dim,
                size_k=hidden_size,
                size_n=padded_num_tokens,
                compress_ratio=compress_ratio,
            ),
            token_page_size=token_page_size,
            tiles_per_group=tiles_per_group,
        )
        cfgs.append(cfg)

        slot_mapping = proj.slot_mapping
        if pad:
            slot_mapping = jnp.pad(slot_mapping, (0, pad), constant_values=-1)
        last_dim = proj.cache.shape[-1]
        run_ends = _plan_runs(
            slot_mapping, tile_n, state_dim // last_dim, token_page_size
        )
        projection_operands.append(
            (slot_mapping, run_ends, proj.wkv_wgate, proj.ape, proj.cache)
        )

        ape_table_rows = sublane_tile * pl.cdiv(
            min(compress_ratio, APE_TABLE_MAX_RATIO), sublane_tile
        )
        slots_per_m_tile = tile_m // last_dim
        acc_rows = tile_n * cfg.num_m * slots_per_m_tile
        projection_scratch.append(
            (
                pltpu.SMEM((padded_num_tokens,), jnp.int32),  # slot_mapping
                pltpu.SMEM((padded_num_tokens,), jnp.int32),  # run_ends
                pltpu.VMEM(proj.ape.shape, proj.ape.dtype),
                pltpu.VMEM((cfg.num_m, hidden_size, tile_m), proj.wkv_wgate.dtype),
                pltpu.VMEM((tiles_per_group, acc_rows, 1, last_dim), jnp.float32),
                pltpu.VMEM((tile_n, tile_m), jnp.float32),
                pltpu.VMEM((ape_table_rows, state_width), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((2, 2), jnp.int32),
            )
        )

    num_halves = sum(cfg.num_m for cfg in cfgs)
    scratch_shapes = (
        pltpu.VMEM((padded_num_tokens,), positions.dtype),
        pltpu.VMEM((tiles_per_group, tile_n, hidden_size), hidden_states.dtype),
        pltpu.SemaphoreType.DMA((num_halves + tiles_per_group + 1,)),
        tuple(projection_scratch),
    )
    operands = (positions, hidden_states, tuple(projection_operands))
    caches = [proj.cache for proj in projections]
    # Flat operand index of each projection's cache: its operands follow
    # positions and hidden_states, five each, the cache last.
    cache_operand = [2 + 5 * p + 4 for p in range(len(projections))]

    # Every operand stays in HBM: the kernel copies the small ones itself,
    # behind the first step's, rather than waiting for them in a prologue.
    return pl.pallas_call(
        functools.partial(kernel, cfgs=tuple(cfgs)),
        out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in caches),
        in_specs=jax.tree.map(lambda _: pl.BlockSpec(memory_space=pltpu.HBM), operands),
        out_specs=tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in caches),
        scratch_shapes=scratch_shapes,
        input_output_aliases={idx: p for p, idx in enumerate(cache_operand)},
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=DEFAULT_VMEM_LIMIT_BYTES,
        ),
        interpret=interpret,
        name="proj_and_save_state",
    )(*operands)


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
    (cache,) = proj_and_save_states(
        hidden_states,
        positions,
        (StateProjection(wkv_wgate, ape, slot_mapping, cache),),
        compress_ratios=(compress_ratio,),
        interpret=interpret,
    )
    return cache
