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
    token_page_size: int

    @property
    def num_m(self) -> int:
        return pl.cdiv(self.dims.size_m, self.tile_sizes.tile_m)

    @property
    def num_k(self) -> int:
        return pl.cdiv(self.dims.size_k, self.tile_sizes.tile_k)

    @property
    def num_n(self) -> int:
        return pl.cdiv(self.dims.size_n, self.tile_sizes.tile_n)


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
    acc_ref,  # (2, tile_n, slots_per_m_tile, 1, 128)
    sem_ref,
    write_counts_ref,  # [2]
    *,
    cfgs: Configs,
):
    """Inner kernel executing matmul, APE addition, and paged writes."""
    tile_m = cfgs.tile_sizes.tile_m
    tile_n = cfgs.tile_sizes.tile_n
    last_dim = acc_ref.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    # A CSA int32 NoPE array hosts the state too; write it as uint8 slabs.
    cache = csa_cache_layout.as_u8_slabs(cache)
    write_counts_ref[0] = 0
    write_counts_ref[1] = 0

    def _dma_copy_out(n_idx_val, m_idx_val, buf_idx_val):
        """Copys one tile from accumulator to the KV cache.

        We run the dma one tile after the previous tile is done computing, using a
        double buffer on the k dimension.

        The reason we do this in the following tile is that we hope to pipeline the
        DMA with the matmuls.

        (however, this is awaiting RAW no-hazard flag to land)
        """
        slot_offset = m_idx_val * slots_per_m_tile
        num_valid_writes = 0
        for i in range(tile_n):
            global_token_idx = n_idx_val * tile_n + i
            global_slot_idx = slot_mapping[global_token_idx]
            valid = global_slot_idx >= 0
            num_valid_writes = num_valid_writes + lax.select(valid, 1, 0)

            # Guard index calculations to avoid out-of-bounds even for size 0
            safe_slot_idx = lax.select(valid, global_slot_idx, 0)
            page_size_tokens = cfgs.token_page_size
            p = safe_slot_idx // page_size_tokens
            start_slot_token = safe_slot_idx % page_size_tokens
            start_slot = start_slot_token + slot_offset

            size = lax.select(valid, slots_per_m_tile, 0)
            dest = cache.at[p, pl.ds(start_slot, size), :, :]
            src = acc_ref_u8.at[buf_idx_val, i, pl.ds(0, size), :, :]
            pltpu.make_async_copy(src, dest, sem_ref.at[buf_idx_val]).start()

        write_counts_ref[buf_idx_val] = num_valid_writes

    def inner_kernel(
        tiled_wgate_ref,  # (tile_k, tile_m)
        tiled_hidden_states_ref,  # (tile_n, tile_k)
        tiled_positions_ref,  # (tile_n,)
    ):
        n_idx = pl.program_id(0)
        m_idx = pl.program_id(1)
        k_idx = pl.program_id(2)
        tile_idx = (
            n_idx * cfgs.num_m + m_idx
        )  # omit k_idx because we write to the same buffer, double buffer on k
        buf_idx = tile_idx % 2

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
                acc_ref.at[buf_idx, ...], dtype=jnp.float32
            )
            new_acc = existing_acc + ape_selected  # (tile_n, state_width)
            store_to_acc_ref(acc_ref.at[buf_idx], new_acc)

        def process(copy_dma: bool, add_ape: bool):
            # 1. Semaphore wait (only on first step)
            @pl.when(k_idx == 0)
            def _wait_for_buffer():
                count = write_counts_ref[buf_idx][...]

                @pl.when(count > 0)
                def _do_wait():
                    pl.semaphore_wait(sem_ref.at[buf_idx], count)

            # 2. Launch DMA for the PREVIOUS tile
            if copy_dma:
                prev_tile_idx = tile_idx - 1
                prev_n_idx = prev_tile_idx >> 1
                prev_m_idx = prev_tile_idx & 1
                prev_buf_idx = 1 - buf_idx
                _dma_copy_out(prev_n_idx, prev_m_idx, prev_buf_idx)

            # 3. Matmul
            prod = _matmul()

            # 4. Unified Accumulate/Assign
            acc_val = load_from_acc_ref(acc_ref.at[buf_idx, ...], dtype=jnp.float32)
            old_acc = lax.select(
                k_idx > 0,
                acc_val,
                jnp.zeros_like(acc_val),
            )
            new_acc = old_acc + prod
            store_to_acc_ref(acc_ref.at[buf_idx], new_acc)

            # 5. APE addition if last
            if add_ape:
                _add_ape()

        # Select and execute
        num_k = cfgs.num_k
        is_last_k = k_idx == (num_k - 1)
        is_score = m_idx * tile_m >= state_width

        copy_dma = (k_idx == 0) & (tile_idx > 0)
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
        lambda n, m, k: (k, m),
        memory_space=pltpu.VMEM,
    )
    hidden_states_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n, cfgs.tile_sizes.tile_k),
        lambda n, m, k: (n, k),
        memory_space=pltpu.VMEM,
    )
    positions_spec = pl.BlockSpec(
        (cfgs.tile_sizes.tile_n,),
        lambda n, m, k: (n,),
        memory_space=pltpu.VMEM,
    )

    pipeline_fn = pltpu.emit_pipeline(
        inner_kernel,
        grid=(cfgs.num_n, cfgs.num_m, cfgs.num_k),
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
    last_tile_idx = cfgs.num_n * cfgs.num_m - 1
    last_n_idx = last_tile_idx // cfgs.num_m
    last_m_idx = last_tile_idx % cfgs.num_m
    last_buf_idx = last_tile_idx % 2

    if last_tile_idx >= 0:
        _dma_copy_out(last_n_idx, last_m_idx, last_buf_idx)

    for b in range(2):
        count = write_counts_ref[b][...]

        @pl.when(count > 0)
        def _wait_epilogue(b=b, count=count):
            pl.semaphore_wait(sem_ref.at[b], count)


def kernel_runs(
    # scalar prefetch
    slot_mapping,  # [num_tokens] state row of each token, -1 = no write
    run_token,  # [num_tokens] first token of each write run (compacted)
    run_len,  # [num_tokens] tokens in each write run (compacted)
    tile_run_offset,  # [num_n + 1] first run of each token tile
    # input
    hidden_states,  # [num_tokens, hidden_size]
    wkv_wgate,  # [hidden_size, 2 * state_width]
    ws_ref,  # [1, 2 * state_width] f32 per-column scale (read when `scaled`)
    ape,  # [compress_ratio, state_width]
    positions,  # [num_tokens]
    cache,  # [num_pages, page_size, 4, lanes] uint8 or [..., lanes] int32 (aliased)
    # output
    _,  # unused output ref (aliased to cache)
    # scratch_memory
    acc_ref,  # (2, tile_n * rows_per_token, 1, lanes) f32, write staging
    sem_ref,
    write_counts_ref,  # [2]
    dense_ref,  # (tile_n, tile_m) f32, running sum over k
    *,
    cfgs: Configs,
    scaled: bool = False,
):
    """`kernel` with one write DMA per run of page-contiguous tokens.

    A token tile's whole state (kv then score, `rows_per_token` rows) is staged
    token-major before it is written, so consecutive tokens of one state page
    are one contiguous HBM range: a prefill chunk writes one DMA per page
    instead of one per token and m tile. The dots, the k accumulation order
    and the APE add are `kernel`'s, so the stored bytes are identical.
    """
    tile_n = cfgs.tile_sizes.tile_n
    lanes = acc_ref.shape[-1]
    rows_per_token = cfgs.dims.size_m // lanes
    state_width = cfgs.dims.size_m // 2
    acc_ref_u8 = acc_ref.bitcast(jnp.uint8)
    cache = csa_cache_layout.as_u8_slabs(cache)
    page_rows = cfgs.token_page_size
    write_counts_ref[0] = 0
    write_counts_ref[1] = 0

    def _run_copy(n_idx_val, buf_idx_val, r):
        tok = run_token[r]
        slot = slot_mapping[tok]
        nrows = run_len[r] * rows_per_token
        src_row = (tok - n_idx_val * tile_n) * rows_per_token
        src = acc_ref_u8.at[buf_idx_val, pl.ds(src_row, nrows), :, :]
        dest = cache.at[slot // page_rows, pl.ds(slot % page_rows, nrows), :, :]
        return pltpu.make_async_copy(src, dest, sem_ref.at[buf_idx_val])

    def _dma_copy_out(n_idx_val, buf_idx_val):
        @pl.loop(tile_run_offset[n_idx_val], tile_run_offset[n_idx_val + 1])
        def _(r):
            _run_copy(n_idx_val, buf_idx_val, r).start()

        # Remember which tile's runs this buffer is writing, for `_wait`.
        write_counts_ref[buf_idx_val] = n_idx_val + 1

    def _wait(buf_idx_val):
        n_plus_1 = write_counts_ref[buf_idx_val]

        @pl.when(n_plus_1 > 0)
        def _():
            n_idx_val = n_plus_1 - 1

            @pl.loop(tile_run_offset[n_idx_val], tile_run_offset[n_idx_val + 1])
            def _(r):
                _run_copy(n_idx_val, buf_idx_val, r).wait()

        write_counts_ref[buf_idx_val] = 0

    def inner_kernel(
        tiled_wgate_ref,  # (tile_k, state_width): this m tile's half
        tiled_hidden_states_ref,  # (tile_n, tile_k)
        tiled_positions_ref,  # (tile_n,)
    ):
        n_idx = pl.program_id(0)
        m_idx = pl.program_id(1)
        k_idx = pl.program_id(2)
        buf_idx = n_idx % 2
        is_last_k = k_idx == (cfgs.num_k - 1)

        @pl.when((m_idx == 0) & (k_idx == 0))
        def _start_tile():
            # Free this buffer (its writes were started two tiles ago), then
            # start the previous tile's writes so they overlap this tile's dots.
            _wait(buf_idx)

            @pl.when(n_idx > 0)
            def _():
                _dma_copy_out(n_idx - 1, 1 - buf_idx)

        num_m = cfgs.num_m  # 1: both halves per step; 2: one half, as `kernel`
        halves = 2 // num_m

        def _stage(val, h):
            # Token-major rows for the write DMA: token t's 128-lane chunk c of
            # half h lands on row t * R + h * R / 2 + c.
            ref_flat = acc_ref.at[buf_idx].reshape(-1, LANES)
            num_chunks = cfgs.dims.size_m // LANES
            half_chunks = state_width // LANES
            for c in range(half_chunks):
                ref_flat[pl.ds(h * half_chunks + c, tile_n, num_chunks)] = val[
                    :, c * LANES : (c + 1) * LANES
                ]

        def _ape():
            ape_rows = tiled_positions_ref[...] % cfgs.dims.compress_ratio
            one_hot = (
                ape_rows[:, None] == jnp.arange(cfgs.dims.compress_ratio)[None, :]
            ).astype(jnp.float32)
            return jnp.matmul(
                one_hot, ape[...].astype(jnp.float32), precision=lax.Precision.HIGHEST
            )

        x = tiled_hidden_states_ref[...]
        for hh in range(halves):
            # Each half is `kernel`'s m-tile dot, summed the same way
            # (0 + dot at k = 0) into a dense running sum.
            cols = pl.ds(hh * state_width, state_width)
            prod = lax.dot_general(
                x,
                tiled_wgate_ref[:, cols] if halves > 1 else tiled_wgate_ref[...],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            old = lax.select(k_idx > 0, dense_ref[:, cols], jnp.zeros_like(prod))
            new = old + prod

            @pl.when(jnp.logical_not(is_last_k))
            def _(new=new, cols=cols):
                dense_ref[:, cols] = new

            h = hh if halves > 1 else m_idx  # which half this dot produced
            if scaled:
                new = new * ws_ref[
                    :, pl.ds(hh * state_width if halves > 1 else m_idx * state_width,
                             state_width)
                ]

            @pl.when(is_last_k & (h == 0))
            def _(new=new):
                _stage(new, 0)

            @pl.when(is_last_k & (h == 1))
            def _(new=new):
                # `kernel` stores the sum, then adds APE to the score half.
                _stage(new + _ape(), 1)

    tile_m = cfgs.tile_sizes.tile_m
    pipeline_fn = pltpu.emit_pipeline(
        inner_kernel,
        grid=(cfgs.num_n, cfgs.num_m, cfgs.num_k),
        in_specs=(
            pl.BlockSpec(
                (cfgs.tile_sizes.tile_k, tile_m),
                lambda n, m, k: (k, m),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(
                (cfgs.tile_sizes.tile_n, cfgs.tile_sizes.tile_k),
                lambda n, m, k: (n, k),
                memory_space=pltpu.VMEM,
            ),
            pl.BlockSpec(
                (cfgs.tile_sizes.tile_n,),
                lambda n, m, k: (n,),
                memory_space=pltpu.VMEM,
            ),
        ),
    )
    pipeline_fn(wkv_wgate, hidden_states, positions)

    last_n = cfgs.num_n - 1
    _dma_copy_out(last_n, last_n % 2)
    _wait(0)
    _wait(1)


LANES = 128


def kernel_stream(*refs, groups, t, nbuf, ndma):
    """Decode projection as a weight stream, for one or more projections of the
    same tokens (`groups`: one dict of static facts per projection).

    Every weight stays in HBM and walks through one VMEM ring of `nbuf` k-tiles
    (group after group) with their DMAs in flight; each tile's two half dots
    (one per state half, as `kernel`'s m tiles) accumulate into the group's f32
    sum (0 + dot first, then +=). Each group's epilogue applies its per-column
    fp8 scale (fp8 weights only), adds APE to the score half, stages the states
    token-major and writes one DMA per token into the group's state cache.
    The k tiling does not change the stored bytes.

    refs: G slot mappings (SMEM) | x, (w, ws, ape) x G, positions | G caches
    (aliased) | G outputs | G weight rings, ring semaphores, G accumulators,
    G staging buffers, output semaphore.
    """
    G = len(groups)
    slots = refs[:G]
    x_ref = refs[G]
    wsa = [refs[G + 1 + 3 * g : G + 4 + 3 * g] for g in range(G)]
    pos_ref = refs[4 * G + 1]
    caches = refs[4 * G + 2 : 5 * G + 2]
    scratch = refs[6 * G + 2 :]
    wbufs = scratch[:G]
    wsem = scratch[G]
    accs = scratch[G + 1 : 2 * G + 1]
    stages = scratch[2 * G + 1 : 3 * G + 1]
    osem = scratch[3 * G + 1]

    num_tokens, size_k = x_ref.shape
    n_tiles = size_k // t
    rows = t // ndma
    tiles = [(g, j) for g in range(G) for j in range(n_tiles)]

    def copies(q):
        g, j = tiles[q]
        w_hbm = wsa[g][0]
        sw = groups[g]["state_width"]
        return [
            pltpu.make_async_copy(
                w_hbm.at[pl.ds(j * t + r * rows, rows), pl.ds(h * sw, sw)],
                wbufs[g].at[q % nbuf, h, pl.ds(r * rows, rows), :],
                wsem.at[q % nbuf],
            )
            for r in range(ndma)
            for h in range(2)
        ]

    for q in range(min(nbuf, len(tiles))):
        for c in copies(q):
            c.start()

    dn = (((1,), (0,)), ((), ()))
    for q, (g, j) in enumerate(tiles):
        for c in copies(q):
            c.wait()
        x = x_ref[:, j * t : (j + 1) * t]
        sw = groups[g]["state_width"]
        for h in range(2):
            cols = slice(h * sw, (h + 1) * sw)
            prod = lax.dot_general(
                x, wbufs[g][q % nbuf, h], dn, preferred_element_type=jnp.float32
            )
            if j == 0:
                accs[g][:, cols] = jnp.zeros_like(prod) + prod
            else:
                accs[g][:, cols] = accs[g][:, cols] + prod
        if q + nbuf < len(tiles):
            for c in copies(q + nbuf):
                c.start()

    all_copies = []
    for g in range(G):
        info = groups[g]
        sw = info["state_width"]
        size_m = 2 * sw
        _, ws_ref, ape = wsa[g]
        stage_ref = stages[g]
        lanes = stage_ref.shape[-1]
        rpt = size_m // lanes
        flat = stage_ref.reshape(-1, LANES)
        num_chunks = size_m // LANES
        half_chunks = sw // LANES
        for h in range(2):
            cols = slice(h * sw, (h + 1) * sw)
            val = accs[g][:, cols]
            if info["scaled"]:
                val = val * ws_ref[:, cols]
            if h == 1:
                ratio = info["compress_ratio"]
                ape_rows = pos_ref[...] % ratio
                one_hot = (ape_rows[:, None] == jnp.arange(ratio)[None, :]).astype(
                    jnp.float32
                )
                val = val + jnp.matmul(
                    one_hot,
                    ape[...].astype(jnp.float32),
                    precision=lax.Precision.HIGHEST,
                )
            for c in range(half_chunks):
                flat[pl.ds(h * half_chunks + c, num_tokens, num_chunks)] = val[
                    :, c * LANES : (c + 1) * LANES
                ]
        cache = csa_cache_layout.as_u8_slabs(caches[g])
        stage_u8 = stage_ref.bitcast(jnp.uint8)
        page = info["token_page_size"]
        # One DMA per token (decode tokens rarely share a page run), as `kernel`.
        for i in range(num_tokens):
            slot = slots[g][i]
            safe = jnp.maximum(slot, 0)
            cp = pltpu.make_async_copy(
                stage_u8.at[pl.ds(i * rpt, rpt), :, :],
                cache.at[safe // page, pl.ds(safe % page, rpt), :, :],
                osem.at[0],
            )
            pl.when(slot >= 0)(cp.start)
            all_copies.append((slot >= 0, cp))
    for valid, cp in all_copies:
        pl.when(valid)(cp.wait)


def _proj_and_save_state_stream_multi(
    hidden_states: jax.Array,
    positions: jax.Array,
    groups: list,  # dicts: wkv_wgate, wkv_scale, ape, slot_mapping, cache, compress_ratio
    *,
    t: int,
    nbuf: int,
    ndma: int,
    interpret: bool,
    name: str = "proj_and_save_state_stream",
) -> tuple[jax.Array, ...]:
    num_tokens, hidden_size = hidden_states.shape
    assert hidden_size % t == 0 and (t // ndma) % 32 == 0, (hidden_size, t, ndma)
    padded = pl.cdiv(num_tokens, 8) * 8
    pad = padded - num_tokens
    if pad:
        hidden_states = jnp.pad(hidden_states, ((0, pad), (0, 0)))
        positions = jnp.pad(positions, (0, pad))
    num_tokens = padded
    G = len(groups)
    infos, slots, ins, in_specs, caches = [], [], [], [], []
    scratch_w, scratch_acc, scratch_stage = [], [], []
    n_tiles = hidden_size // t
    nbuf = max(1, min(nbuf, G * n_tiles))
    vmem_need = 2 * num_tokens * hidden_size * 2
    for gr in groups:
        w, cache = gr["wkv_wgate"], gr["cache"]
        state_dim = w.shape[1]
        lanes = cache.shape[-1]
        assert state_dim % lanes == 0 and lanes % LANES == 0, (state_dim, lanes)
        sm = gr["slot_mapping"]
        if pad:
            sm = jnp.pad(sm, (0, pad), constant_values=-1)
        slots.append(sm)
        scaled = gr["wkv_scale"] is not None
        ws = (
            gr["wkv_scale"].reshape(1, state_dim).astype(jnp.float32)
            if scaled
            else jnp.ones((1, state_dim), jnp.float32)
        )
        # Keep the weight in HBM: otherwise XLA's memory-space assignment may
        # stage a weight that fits in VMEM with a separate copy before the call.
        ins += [pltpu.with_memory_space_constraint(w, pltpu.HBM), ws, gr["ape"]]
        in_specs += [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ]
        caches.append(cache)
        infos.append(
            dict(
                state_width=state_dim // 2,
                compress_ratio=gr["compress_ratio"],
                token_page_size=cache.shape[1],
                scaled=scaled,
            )
        )
        scratch_w.append(pltpu.VMEM((nbuf, 2, t, state_dim // 2), w.dtype))
        scratch_acc.append(pltpu.VMEM((num_tokens, state_dim), jnp.float32))
        scratch_stage.append(
            pltpu.VMEM((num_tokens * (state_dim // lanes), 1, lanes), jnp.float32)
        )
        vmem_need += (
            nbuf * t * state_dim * jnp.dtype(w.dtype).itemsize
            + 2 * num_tokens * state_dim * 4 * 2
            + 3 * num_tokens * state_dim * 4
        )
    # Tight scoped VMEM: a large limit stops XLA keeping its own VMEM prefetches
    # (e.g. the next MoE operands) live across this call (measured: +4 us per
    # decode MoE call at a 32 MiB floor, c8 twin j-2ae4b46e vs j-52b5ae5a).
    vmem_limit = min(max(vmem_need + (4 << 20), 12 << 20), 60 << 20)
    n_in = 1 + 3 * G + 1  # x, groups, positions (after the G scalar prefetches)
    outs = pl.pallas_call(
        functools.partial(
            kernel_stream, groups=tuple(infos), t=t, nbuf=nbuf, ndma=ndma
        ),
        out_shape=tuple(jax.ShapeDtypeStruct(c.shape, c.dtype) for c in caches),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=G,
            grid=(),
            in_specs=[pl.BlockSpec(memory_space=pltpu.VMEM)]  # hidden_states
            + in_specs
            + [pl.BlockSpec(memory_space=pltpu.VMEM)]  # positions
            + [pl.BlockSpec(memory_space=pltpu.HBM)] * G,  # caches
            out_specs=tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in range(G)),
            scratch_shapes=scratch_w
            + [pltpu.SemaphoreType.DMA((nbuf,))]
            + scratch_acc
            + scratch_stage
            + [pltpu.SemaphoreType.DMA((1,))],
        ),
        input_output_aliases={G + n_in + g: g for g in range(G)},
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
        name=name,
    )(*slots, hidden_states, *ins, positions, *caches)
    return tuple(outs)


def _proj_and_save_state_stream(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    wkv_scale: jax.Array | None,
    ape: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    cache: jax.Array,
    *,
    compress_ratio: int,
    t: int,
    nbuf: int,
    ndma: int,
    interpret: bool,
) -> jax.Array:
    (out,) = _proj_and_save_state_stream_multi(
        hidden_states,
        positions,
        [
            dict(
                wkv_wgate=wkv_wgate,
                wkv_scale=wkv_scale,
                ape=ape,
                slot_mapping=slot_mapping,
                cache=cache,
                compress_ratio=compress_ratio,
            )
        ],
        t=t,
        nbuf=nbuf,
        ndma=ndma,
        interpret=interpret,
    )
    return out


def _write_runs(slot_mapping, tile_n, rows_per_token, page_rows):
    """Splits the tokens' state writes into page-contiguous runs.

    A run is a maximal set of consecutive tokens of one token tile whose state
    rows are consecutive in one page. Returns the compacted run starts and
    lengths (padded with 0) and each tile's first run.
    """
    num_tokens = slot_mapping.shape[0]
    idx = jnp.arange(num_tokens)
    valid = slot_mapping >= 0
    prev = jnp.roll(slot_mapping, 1)
    prev_valid = jnp.roll(valid, 1)
    same_page = (slot_mapping // page_rows) == (prev // page_rows)
    continues = (
        valid
        & prev_valid
        & (idx % tile_n != 0)
        & (slot_mapping == prev + rows_per_token)
        & same_page
    )
    starts = valid & ~continues
    run_token = jnp.nonzero(starts, size=num_tokens, fill_value=0)[0].astype(jnp.int32)
    num_runs = jnp.sum(starts.astype(jnp.int32))
    # Each run ends where the next one starts, or at the end of its own valid
    # stretch; count a run's tokens as its start's id over the stretch.
    run_id = jnp.cumsum(starts.astype(jnp.int32)) - 1
    run_len = (
        jnp.zeros((num_tokens,), jnp.int32)
        .at[jnp.where(valid, run_id, num_tokens)]
        .add(1, mode="drop")
    )
    run_len = jnp.where(idx < num_runs, run_len, 0)
    num_n = num_tokens // tile_n
    tile_starts = jnp.arange(num_n + 1) * tile_n
    # Runs never cross a tile, so a tile's runs are those starting inside it.
    starts_before = jnp.concatenate(
        [jnp.zeros((1,), jnp.int32), jnp.cumsum(starts.astype(jnp.int32))]
    )
    tile_run_offset = starts_before[tile_starts].astype(jnp.int32)
    return run_token, run_len, tile_run_offset


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
        "runs_min_tokens",
        "runs_tile_n",
        "tile_k_cap",
        "stream_max_tokens",
        "stream_cfg",
        "stream_min_width",
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
    runs_min_tokens: int = 0,
    runs_tile_n: int = 512,
    tile_k_cap: int = 3584,
    wkv_scale: jax.Array | None = None,  # [2 * state_width] f32, fp8 weights only
    stream_max_tokens: int = 0,
    stream_cfg: tuple[int, int, int] = (3584, 2, 1),  # (k tile, ring slots, DMAs/tile)
    stream_min_width: int = 0,  # stream only projections at least this wide
) -> jax.Array:
    """Projects hidden states and saves them to the KV cache in HBM.

    With ``runs_min_tokens > 0``, a call of at least that many tokens computes
    each token's whole state in one grid step and writes page-contiguous runs
    of tokens with one DMA each (`kernel_runs`); the stored bytes are the same.

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

    if wkv_scale is not None and num_tokens > stream_max_tokens:
        # fp8 weights: the runs kernel serves every call the stream does not.
        return _proj_and_save_state_runs(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            compress_ratio=compress_ratio,
            tile_n_cap=runs_tile_n,
            tile_k_cap=tile_k_cap,
            wkv_scale=wkv_scale,
            hbm_pin=True,
            interpret=interpret,
        )
    if 0 < num_tokens <= stream_max_tokens and (
        state_dim >= stream_min_width or wkv_scale is not None  # fp8: stream only
    ):
        return _proj_and_save_state_stream(
            hidden_states, wkv_wgate, wkv_scale, ape, positions, slot_mapping,
            cache, compress_ratio=compress_ratio, t=stream_cfg[0],
            nbuf=stream_cfg[1], ndma=stream_cfg[2], interpret=interpret,
        )

    if runs_min_tokens > 0 and num_tokens >= runs_min_tokens:
        return _proj_and_save_state_runs(
            hidden_states,
            wkv_wgate,
            ape,
            positions,
            slot_mapping,
            cache,
            compress_ratio=compress_ratio,
            tile_n_cap=runs_tile_n,
            tile_k_cap=tile_k_cap,
            # The stream knob opts the whole projection family into HBM-pinned
            # weights; off, the runs path is byte- and placement-identical.
            hbm_pin=stream_max_tokens > 0,
            interpret=interpret,
        )

    state_width = state_dim // 2
    tile_m = state_width
    tile_k = _select_tile_k(hidden_size, tile_k_cap)

    # The token dimension must be an exact multiple of `tile_n`. Every token
    # tile streams the whole weight again, so tiles stay large.
    sublane_tile = 8
    tile_n = min(512, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
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
        token_page_size=token_page_size,
    )

    last_dim = cache.shape[-1]
    slots_per_m_tile = tile_m // last_dim
    scratch_shapes = [
        pltpu.VMEM((2, tile_n, slots_per_m_tile, 1, last_dim), jnp.float32),
        pltpu.SemaphoreType.REGULAR((2,)),
        pltpu.SMEM((2,), jnp.int32),
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


def _proj_and_save_state_runs(
    hidden_states: jax.Array,
    wkv_wgate: jax.Array,
    ape: jax.Array,
    positions: jax.Array,
    slot_mapping: jax.Array,
    cache: jax.Array,
    *,
    compress_ratio: int,
    tile_n_cap: int,
    interpret: bool,
    tile_k_cap: int = 3584,
    wkv_scale: jax.Array | None = None,
    hbm_pin: bool = False,
) -> jax.Array:
    num_tokens, hidden_size = hidden_states.shape
    _, state_dim = wkv_wgate.shape
    token_page_size = cache.shape[1]
    lanes = cache.shape[-1]
    assert state_dim % lanes == 0 and lanes % LANES == 0, (state_dim, lanes)
    rows_per_token = state_dim // lanes
    # Same k tiling as `kernel`, so each dot accumulates in the same order.
    tile_k = _select_tile_k(hidden_size, tile_k_cap)

    sublane_tile = 8
    tile_n = min(tile_n_cap, pl.cdiv(num_tokens, sublane_tile) * sublane_tile)
    padded_num_tokens = pl.cdiv(num_tokens, tile_n) * tile_n
    if padded_num_tokens != num_tokens:
        pad = padded_num_tokens - num_tokens
        hidden_states = jnp.pad(hidden_states, ((0, pad), (0, 0)))
        positions = jnp.pad(positions, (0, pad))
        slot_mapping = jnp.pad(slot_mapping, (0, pad), constant_values=-1)
        num_tokens = padded_num_tokens

    cfgs = Configs(
        # Halves narrower than 512 lanes keep `kernel`'s one-half grid steps:
        # one dot per step there is what reproduces its bytes (measured).
        tile_sizes=TileSizes(
            tile_m=state_dim if state_dim // 2 >= 512 else state_dim // 2,
            tile_k=tile_k,
            tile_n=tile_n,
        ),
        dims=Dimensions(
            size_m=state_dim,
            size_k=hidden_size,
            size_n=num_tokens,
            compress_ratio=compress_ratio,
        ),
        token_page_size=token_page_size,
    )
    run_token, run_len, tile_run_offset = _write_runs(
        slot_mapping, tile_n, rows_per_token, token_page_size
    )

    w_bytes = jnp.dtype(wkv_wgate.dtype).itemsize
    x_bytes = jnp.dtype(hidden_states.dtype).itemsize
    vmem_need = (
        2 * tile_k * cfgs.tile_sizes.tile_m * w_bytes  # weight tiles, double-buffered
        + 2 * tile_n * tile_k * x_bytes  # hidden tiles, double-buffered
        + 2 * tile_n * rows_per_token * lanes * 4  # write staging
        + tile_n * cfgs.tile_sizes.tile_m * 4  # running sum
        + tile_n * state_dim * 4  # dot / sum temporaries
    )
    vmem_limit = min(max(vmem_need + (8 << 20), 32 << 20), 60 << 20)

    scaled = wkv_scale is not None
    ws = (
        wkv_scale.reshape(1, state_dim).astype(jnp.float32)
        if scaled
        else jnp.ones((1, state_dim), jnp.float32)
    )
    if hbm_pin:
        # No XLA VMEM staging copy of the weight before the call (see the stream).
        wkv_wgate = pltpu.with_memory_space_constraint(wkv_wgate, pltpu.HBM)
    return pl.pallas_call(
        functools.partial(kernel_runs, cfgs=cfgs, scaled=scaled),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=4,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # hidden_states
                pl.BlockSpec(memory_space=pltpu.HBM),  # wkv_wgate
                pl.BlockSpec(memory_space=pltpu.VMEM),  # ws
                pl.BlockSpec(memory_space=pltpu.VMEM),  # ape
                pl.BlockSpec(memory_space=pltpu.HBM),  # positions
                pl.BlockSpec(memory_space=pltpu.HBM),  # cache
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, tile_n * rows_per_token, 1, lanes), jnp.float32),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((2,), jnp.int32),
                pltpu.VMEM((tile_n, cfgs.tile_sizes.tile_m), jnp.float32),
            ],
        ),
        input_output_aliases={9: 0},  # cache
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit,
        ),
        interpret=interpret,
        name="proj_and_save_state_runs",
    )(
        slot_mapping,
        run_token,
        run_len,
        tile_run_offset,
        hidden_states,
        wkv_wgate,
        ws,
        ape,
        positions,
        cache,
    )
