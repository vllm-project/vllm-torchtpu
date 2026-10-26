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
"""Compress-norm-rope-store for the prefill boundaries of a step, in blocks.

`compress_norm_rope_store` walks the boundary tokens four at a time: each grid
step fetches two whole state pages per token, gathers the window row by row,
and writes one record per DMA. A prefill chunk's boundaries are consecutive
compressed tokens of one request, so this kernel takes up to ``block`` of them
at once instead:

* the window states of the whole block are one contiguous range of token
  states, fetched with one DMA per state page;
* the window of every boundary is filled with strided copies (boundary ``b``'s
  window token ``w`` is staged token ``4 b + w``);
* the math is `kernel.inner_kernel`'s, op for op, on ``block`` tokens;
* the NoPE records (CSA) are one DMA per block, the RoPE rows are built whole
  (blocks cover whole 4-token rows), and the SEQ_ALONG_LANE records (indexer)
  are one read-modify-write of one 128-lane tile per block.

Only CSA (head 512, ratio 4, overlap) and the SEQ_ALONG_LANE CSA indexer (head
128) are handled. Every decode token's boundary stays with
`compress_norm_rope_store`, and for CSA so do a request's boundaries outside its
4-aligned span ``[C_lo, C_hi)`` -- at most three at each end of the chunk -- so
a RoPE row is never shared between the two kernels. The SAL indexer takes every
prefill boundary (see `plan_blocks`).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import compute, config

LANE = config.LANE


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Plan:
    """Blocks of prefill boundaries, from `plan_blocks`."""

    num_blocks: jax.Array  # i32[1]
    blk_c0: jax.Array  # i32[MB] first compressed index of the block
    blk_n: jax.Array  # i32[MB] boundaries in the block (multiple of 4)
    blk_spages: jax.Array  # i32[MB * PPB] state pages from the block's first window token
    blk_kvpage: jax.Array  # i32[MB] compressed-KV page holding the block's records
    cos_sin: jax.Array  # f32[MB * block, rope_head_dim] per boundary
    in_block: jax.Array  # bool[T] token's boundary is written by this kernel


def pages_per_block(block: int, compress_ratio: int, state_block_size: int) -> int:
    window_tokens = (block + 1) * compress_ratio
    return pl.cdiv(window_tokens, state_block_size) + 1


def max_blocks(num_tokens: int, num_reqs: int, block: int, compress_ratio: int) -> int:
    return num_tokens // (compress_ratio * block) + num_reqs


def plan_blocks(
    positions: jax.Array,  # [T]
    query_start_loc: jax.Array,  # [R + 1]
    distribution: jax.Array,  # i32[3]
    token_to_req: jax.Array,  # [T]
    block_table: jax.Array,  # [R * stride] state pages
    kv_block_table: jax.Array,  # [R * kv_stride] compressed-KV pages
    cos_sin_cache: jax.Array,  # [max_pos, rope_head_dim]
    *,
    block: int,
    compress_ratio: int,
    state_block_size: int,
    kv_block_size: int,  # compressed tokens per KV page
    seq_along_lane: bool,
) -> Plan:
    num_tokens = positions.shape[0]
    num_reqs = query_start_loc.shape[0] - 1
    stride = block_table.shape[0] // num_reqs
    kv_stride = kv_block_table.shape[0] // num_reqs
    ratio = compress_ratio
    assert kv_block_size % block == 0 and block % config.SLOT_PACK == 0
    assert not seq_along_lane or block <= LANE

    r = jnp.arange(num_reqs)
    start = query_start_loc[:-1]
    length = query_start_loc[1:] - start
    first_pos = positions[jnp.clip(start, 0, num_tokens - 1)]
    is_prefill = (r >= distribution[0]) & (r < distribution[2]) & (length > 0)
    # Boundary at position p = ratio * c + ratio - 1 owns compressed index c.
    c_lo = (first_pos + ratio - 1) // ratio
    c_hi = (first_pos + length) // ratio
    # CSA: whole 4-token RoPE rows only (see the module docstring). The SAL
    # indexer takes every prefill boundary: leaving some to
    # `compress_norm_rope_store` would put two of its runs in one 128-lane
    # tile, and its read-modify-writes of one tile race within a wave.
    rows = 1 if seq_along_lane else config.SLOT_PACK
    big_lo = (c_lo + rows - 1) // rows * rows
    big_hi = c_hi // rows * rows
    valid = is_prefill & (big_hi > big_lo)
    nb = jnp.where(valid, (big_hi - 1) // block - big_lo // block + 1, 0)
    ends = jnp.cumsum(nb)
    offs = ends - nb

    mb = max_blocks(num_tokens, num_reqs, block, ratio)
    j = jnp.arange(mb)
    req = jnp.clip(jnp.searchsorted(ends, j, side="right"), 0, num_reqs - 1)
    i = j - offs[req]
    lo = big_lo[req]
    c0 = jnp.maximum(lo, (lo // block + i) * block)
    c1 = jnp.minimum(big_hi[req], (lo // block + i + 1) * block)
    num_blocks = ends[-1]
    live = j < num_blocks
    c0 = jnp.where(live, c0, 0)
    n = jnp.where(live, c1 - c0, 0)

    # State pages from the block's first window token (clamped at position 0).
    ppb = pages_per_block(block, ratio, state_block_size)
    p0 = jnp.maximum(ratio * c0 - ratio, 0)
    pg = p0[:, None] // state_block_size + jnp.arange(ppb)[None, :]
    pg = jnp.clip(pg, 0, stride - 1)
    spages = block_table[req[:, None] * stride + pg].reshape(-1)
    kvpage = kv_block_table[req * kv_stride + jnp.clip(c0 // kv_block_size, 0, kv_stride - 1)]

    b = jnp.arange(block)
    cpos = ratio * (c0[:, None] + b[None, :])
    cpos = jnp.clip(cpos, 0, cos_sin_cache.shape[0] - 1)
    cos_sin = cos_sin_cache[cpos.reshape(-1)]

    tok_c = positions // ratio
    tok_req = token_to_req
    is_bnd = ((positions + 1) % ratio) == 0
    in_block = (
        is_bnd
        & valid[tok_req]
        & (tok_c >= big_lo[tok_req])
        & (tok_c < big_hi[tok_req])
        & (jnp.arange(num_tokens) < query_start_loc[distribution[2]])
        & (jnp.arange(num_tokens) >= query_start_loc[distribution[0]])
    )
    return Plan(
        num_blocks=num_blocks.reshape(1).astype(jnp.int32),
        blk_c0=c0.astype(jnp.int32),
        blk_n=n.astype(jnp.int32),
        blk_spages=spages.astype(jnp.int32),
        blk_kvpage=kvpage.astype(jnp.int32),
        cos_sin=cos_sin,
        in_block=in_block,
    )


def _sal_words(records, groups):
    """(n, width) u8 records -> (groups, n) u32 SEQ_ALONG_LANE words, exactly.

    `compute.stage_sal_columns`' byte pick as one matmul over the whole block:
    row ``k * groups_pad + g`` of ``pick @ records^T`` is byte ``4 g + k`` of
    every record. Every operand is 0/1 or a byte value, so nothing rounds.
    """
    n, width = records.shape
    groups_pad = pl.cdiv(groups, 8) * 8
    rec = records.astype(jnp.int32).astype(jnp.bfloat16)
    shape = (4 * groups_pad, width)
    out_row = lax.broadcasted_iota(jnp.int32, shape, 0)
    byte_idx = lax.broadcasted_iota(jnp.int32, shape, 1)
    k, g = out_row // groups_pad, out_row % groups_pad
    pick = ((byte_idx == 4 * g + k) & (g < groups)).astype(jnp.bfloat16)
    planes = lax.dot_general(
        pick, rec, (((1,), (1,)), ((), ())), preferred_element_type=jnp.float32
    ).astype(jnp.int32)  # (4 * groups_pad, n)
    words = planes[:groups_pad]
    for b in range(1, 4):
        words = words | (planes[b * groups_pad : (b + 1) * groups_pad] << (8 * b))
    return pltpu.bitcast(words[:groups], jnp.uint32)


def _kernel(
    # scalar prefetch
    num_blocks_ref,
    blk_c0_ref,
    blk_n_ref,
    blk_spages_ref,
    blk_kvpage_ref,
    # inputs
    rms_weight_ref,  # VMEM (head_tiles, 128)
    cos_sin_hbm,  # HBM (MB * block, rope_head_dim)
    cache_hbm,  # HBM cache (aliased)
    *rest,
    cfgs: config.Configs,
    block: int,
    ppb: int,
):
    is_sal = cfgs.dims.seq_along_lane
    if is_sal:
        (out_cache, stage_ref, cs_ref, window_ref, rec_ref, in_sem, out_sem) = rest
        rope_u8 = rope_rows_ref = words_ref = None
    else:
        (
            rope_hbm,
            out_cache,
            out_rope,
            stage_ref,
            cs_ref,
            window_ref,
            rec_ref,
            rope_rows_ref,
            words_ref,
            in_sem,
            out_sem,
        ) = rest
        del rope_hbm
        rope_u8 = csa_cache_layout.as_u8_slabs(out_rope)
    del cache_hbm
    ratio = cfgs.dims.compress_ratio
    window = cfgs.window
    ht = cfgs.head_tiles
    sbs = cfgs.state_block_size
    rpt = cfgs.state_rows_per_token
    kvbs = cfgs.kv_block_size
    cache_u8 = csa_cache_layout.as_u8_slabs(out_cache)
    num_blocks = num_blocks_ref[0]

    def _state_copies(j, slot, fn):
        c0 = blk_c0_ref[j]
        n = blk_n_ref[j]
        p0 = ratio * c0 - ratio  # staged token 0
        p1 = ratio * (c0 + n) - 1
        p0c = jnp.maximum(p0, 0)
        first_pg = p0c // sbs
        npg = p1 // sbs - first_pg + 1
        # SAL rows are 256 lanes; stage them as two 128-lane halves so the
        # window copies below are strided loads of 128-lane rows.
        nh = 2 if is_sal else 1
        dsts = [stage_ref.at[slot, h].bitcast(jnp.uint8) for h in range(nh)]

        @pl.loop(0, npg)
        def _(i):
            pg = first_pg + i
            lo = jnp.maximum(p0c, pg * sbs)
            hi = jnp.minimum(p1, pg * sbs + sbs - 1)
            cnt = (hi - lo + 1) * rpt
            page = blk_spages_ref[j * ppb + i]
            for h in range(nh):
                src = cache_u8.at[
                    page, pl.ds((lo - pg * sbs) * rpt, cnt), :, pl.ds(h * LANE, LANE)
                ]
                d = dsts[h].at[pl.ds((lo - p0) * rpt, cnt), :, :]
                fn(pltpu.make_async_copy(src, d, in_sem.at[slot]))

    def _lane0(j):
        return pl.multiple_of((blk_c0_ref[j] % kvbs) // LANE * LANE, LANE)

    def _fetch(j, slot, fn):
        _state_copies(j, slot, fn)
        fn(
            pltpu.make_async_copy(
                cos_sin_hbm.at[pl.ds(j * block, block), :],
                cs_ref.at[slot],
                in_sem.at[slot],
            )
        )
        if is_sal:
            src = cache_u8.at[blk_kvpage_ref[j], :, :, pl.ds(_lane0(j), LANE)]
            fn(pltpu.make_async_copy(src, rec_ref.at[slot], in_sem.at[slot]))

    def _writes(j, slot, fn):
        c0 = blk_c0_ref[j]
        n = blk_n_ref[j]
        page = blk_kvpage_ref[j]
        if is_sal:
            dst = cache_u8.at[page, :, :, pl.ds(_lane0(j), LANE)]
            fn(pltpu.make_async_copy(rec_ref.at[slot], dst, out_sem.at[slot]))
        else:
            row = c0 % kvbs
            fn(
                pltpu.make_async_copy(
                    rec_ref.at[slot, pl.ds(0, n)],
                    cache_u8.at[page, pl.ds(row, n)],
                    out_sem.at[slot],
                )
            )
            rows_u8 = csa_cache_layout.as_u8_slabs(rope_rows_ref).at[slot]
            fn(
                pltpu.make_async_copy(
                    rows_u8.at[pl.ds(0, n // 4)],
                    rope_u8.at[page, pl.ds(row // 4, n // 4)],
                    out_sem.at[slot],
                )
            )

    def _compute(j, slot):
        c0 = blk_c0_ref[j]
        n = blk_n_ref[j]

        # Boundary b's window token w is staged token ratio * b + w.
        half = window // 2
        for w in range(window):
            is_prev = w < half
            for f in range(2):  # 0: kv, 1: score
                if is_sal:
                    # The 256-lane indexer state row of each field holds the
                    # prev half in lanes 0:128, the curr half in 128:256.
                    src = stage_ref[
                        slot, 0 if is_prev else 1, pl.ds(w * rpt + f, block, ratio * rpt)
                    ]
                    window_ref[slot, f, :, w, :, :] = src
                else:
                    fr = cfgs.field_rows
                    if f == 0:
                        base = 0 if is_prev else fr
                    else:
                        base = 2 * fr if is_prev else 3 * fr
                    for d in range(ht):
                        src = stage_ref[
                            slot, 0, pl.ds(w * rpt + base + d, block, ratio * rpt)
                        ]
                        window_ref[slot, f, :, w, pl.ds(d, 1), :] = src

        # --- `kernel.inner_kernel`'s math, on `block` tokens ---
        bidx = lax.broadcasted_iota(jnp.int32, (block,), 0)
        start = ratio * (c0 + bidx) + ratio - 1 - window + 1
        kv_val = window_ref[slot, 0]  # (block, window, ht, 128)
        scores_val = window_ref[slot, 1]
        rms_weight_tiled = rms_weight_ref[...].astype(jnp.float32)

        curr_pos = start[:, None] + jnp.arange(window)[None, :]
        mask = curr_pos >= 0
        mask_float = mask.astype(scores_val.dtype)
        mask_float_reshaped = mask_float[:, :, None, None]
        neg_inf = jnp.array(-jnp.inf, dtype=scores_val.dtype)
        masked_scores = jnp.where(mask_float_reshaped > 0.5, scores_val, neg_inf)
        weights = jax.nn.softmax(masked_scores, axis=1)
        kv_val = jnp.where(mask_float_reshaped > 0.5, kv_val, 0.0)
        compressed = jnp.sum(weights * kv_val, axis=1)  # (block, ht, 128)

        variance = jnp.mean(jnp.square(compressed), axis=(1, 2), keepdims=True)
        normed = (
            compressed
            * jax.lax.rsqrt(variance + cfgs.dims.rms_eps)
            * rms_weight_tiled[None, :, :]
        )

        rope_slot = cfgs.rope_slot
        rope_val = normed[:, rope_slot : rope_slot + 1]
        cos_sin = cs_ref[slot][:, None, :]
        cos_val = cos_sin[:, :, : cfgs.half_rope]
        sin_val = cos_sin[:, :, cfgs.half_rope :]
        rope_ropped = compute.interleaved_rope_vector(rope_val, cos_val, sin_val)
        if ht > 1:
            normed = jnp.concatenate([normed[:, :rope_slot], rope_ropped], axis=1)
        else:
            normed = rope_ropped

        if is_sal:
            q, scale = compute.quantize_fp8_tiled(normed, cfgs.dims.quant_block)
            nope = compute.pack_nope_tiled(
                q,
                scale,
                cfgs.nope_store_dim,
                cfgs.dims.quant_block,
                nope_width_bytes=cfgs.record_bytes,
                last_dim_size=cfgs.last_dim_size,
            )  # (block, 1, 256)
            groups = rec_ref.shape[1]
            words = _sal_words(nope.reshape(block, -1), groups)  # (groups, block)
            o = c0 % LANE
            if block < LANE:
                words = jnp.concatenate(
                    [words, jnp.zeros((groups, LANE - block), jnp.uint32)], axis=1
                )
            words = pltpu.roll(words, o, 1)
            lane = lax.broadcasted_iota(jnp.int32, (groups, LANE), 1)
            cur = pltpu.bitcast(rec_ref[slot].reshape(groups * 4, LANE), jnp.uint32)
            new = jnp.where((lane >= o) & (lane < o + n), words, cur)
            rec_ref[slot] = pltpu.bitcast(new, jnp.uint8).reshape(groups, 4, LANE)
        else:
            q, scale = compute.quantize_fp8_lane_periodic(
                normed, cfgs.dims.quant_block, cfgs.nope_store_dim
            )
            nope = compute.pack_nope_lane_periodic(
                q, scale, cfgs.nope_store_dim, cfgs.record_bytes, cfgs.last_dim_size
            )  # (block, 4, 128)
            rec_ref[slot] = nope
            words = compute.rope_words_tiled(rope_ropped, cfgs.dims.rope_head_dim)
            words_ref[...] = words.reshape(block, LANE)
            per_row = csa_cache_layout.ROPE_TOKENS_PER_ROW
            width = csa_cache_layout.ROPE_WORDS
            rope_lane = LANE - cfgs.dims.rope_head_dim
            nrows = block // per_row
            win_id = lax.broadcasted_iota(jnp.int32, (nrows, LANE), 1) // width
            row = jnp.zeros((nrows, LANE), jnp.int32)
            for jj in range(per_row):
                wj = words_ref[pl.ds(jj, nrows, per_row), :]
                placed = pltpu.roll(wj, (jj * width - rope_lane) % LANE, 1)
                row = jnp.where(win_id == jj, placed, row)
            rope_rows_ref[slot] = row

    @pl.when(num_blocks > 0)
    def _():
        _fetch(0, 0, lambda c: c.start())

    @pl.loop(0, num_blocks)
    def _(j):
        slot = j % 2

        @pl.when(j + 1 < num_blocks)
        def _():
            # The other slot's writes (block j - 1) must land before block
            # j + 1's fetch: the SAL tile read could hit the same tile.
            @pl.when(j >= 1)
            def _():
                _writes(j - 1, 1 - slot, lambda c: c.wait())

            _fetch(j + 1, 1 - slot, lambda c: c.start())

        _fetch(j, slot, lambda c: c.wait())
        _compute(j, slot)
        _writes(j, slot, lambda c: c.start())

    @pl.when(num_blocks >= 2)
    def _():
        jl = num_blocks - 2
        _writes(jl, jl % 2, lambda c: c.wait())

    @pl.when(num_blocks >= 1)
    def _():
        jl = num_blocks - 1
        _writes(jl, jl % 2, lambda c: c.wait())


@functools.partial(
    jax.jit,
    static_argnames=(
        "state_block_size",
        "compress_ratio",
        "quant_block",
        "rms_eps",
        "seq_along_lane",
        "block",
    ),
    donate_argnames=("cache", "rope_cache"),
)
def prefill_block_compress(
    cache: jax.Array,
    rope_cache: jax.Array | None,
    plan: Plan,
    rms_weight: jax.Array,
    *,
    state_block_size: int,
    compress_ratio: int,
    quant_block: int,
    rms_eps: float,
    seq_along_lane: bool,
    block: int,
) -> tuple[jax.Array, jax.Array | None]:
    head_dim = rms_weight.shape[0]
    rope_head_dim = plan.cos_sin.shape[1]
    cfgs = config.Configs.make(
        config.select_mode(head_dim, True),
        size_n=block,
        physical_page_size=cache.shape[1],
        state_block_size=state_block_size,
        rms_eps=rms_eps,
        tile_n=block,
        head_dim=head_dim,
        rope_head_dim=rope_head_dim,
        compress_ratio=compress_ratio,
        quant_block=quant_block,
        seq_along_lane=seq_along_lane,
        sal_page_size=cache.shape[-1] if seq_along_lane else None,
    )
    mode = cfgs.dims.mode
    assert mode in (config.Mode.CSA, config.Mode.CSA_INDEXER), mode
    assert seq_along_lane == (mode == config.Mode.CSA_INDEXER)
    ppb = pages_per_block(block, compress_ratio, state_block_size)
    ht = cfgs.head_tiles
    lanes = cfgs.last_dim_size
    rows = (block + 1) * compress_ratio * cfgs.state_rows_per_token
    scratch = [
        pltpu.VMEM((2, lanes // LANE, rows, 1, LANE), jnp.float32),
        pltpu.VMEM((2, block, rope_head_dim), jnp.float32),
        pltpu.VMEM((2, 2, block, cfgs.window, ht, LANE), jnp.float32),
    ]
    if seq_along_lane:
        scratch += [pltpu.VMEM((2, config.SAL_SUBLANE_GROUPS, 4, LANE), jnp.uint8)]
    else:
        scratch += [
            pltpu.VMEM((2, block, 4, LANE), jnp.uint8),
            pltpu.VMEM((2, block // 4, LANE), jnp.int32),
            pltpu.VMEM((block, LANE), jnp.int32),
        ]
    scratch += [pltpu.SemaphoreType.DMA((2,)), pltpu.SemaphoreType.DMA((2,))]

    scalar = (plan.num_blocks, plan.blk_c0, plan.blk_n, plan.blk_spages, plan.blk_kvpage)
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # rms weight
        pl.BlockSpec(memory_space=pltpu.HBM),  # cos_sin
        pl.BlockSpec(memory_space=pltpu.HBM),  # cache
    ]
    operands = [rms_weight.reshape(ht, LANE), plan.cos_sin, cache]
    out_shape = [jax.ShapeDtypeStruct(cache.shape, cache.dtype)]
    out_specs = [pl.BlockSpec(memory_space=pltpu.HBM)]
    aliases = {len(scalar) + 2: 0}
    if not seq_along_lane:
        in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
        operands.append(rope_cache)
        out_shape.append(jax.ShapeDtypeStruct(rope_cache.shape, rope_cache.dtype))
        out_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
        aliases[len(scalar) + 3] = 1

    outs = pl.pallas_call(
        functools.partial(_kernel, cfgs=cfgs, block=block, ppb=ppb),
        out_shape=tuple(out_shape),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(scalar),
            grid=(),
            in_specs=in_specs,
            out_specs=tuple(out_specs),
            scratch_shapes=scratch,
        ),
        input_output_aliases=aliases,
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, vmem_limit_bytes=48 << 20
        ),
        name="compress_prefill_block",
    )(*scalar, *operands)
    if seq_along_lane:
        return outs[0], rope_cache
    return outs[0], outs[1]
