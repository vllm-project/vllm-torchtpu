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
"""Small-batch NVFP4 (W4A16) MoE without token permutation.

The decode-sized counterpart of moe_dense_expert for fp4 experts with fp32
[128, 1] block scales. Each nonempty local expert is streamed once, in units of
`chunk` intermediate columns (its gate, up and down slices), through a ring of
VMEM buffers; every token is evaluated in its original order and the weighted
expert outputs accumulate in fp32 VMEM. This replaces the routing sort, the
one-hot permute, the grouped GEMM and the combine with one call. The fp4
weights feed the MXU as fp8 (exact) against bf16 activations, and each 128-row
block scale is applied to the block's partial product, as gmm_v2's decode
tiles do. Rounding follows the GMM path: the activation is rounded to bf16
before the down projection, each expert's output to bf16 before its routing
weight is applied.

With ``sparse_rows`` (TPU_MOE_DENSE_FP4_SPARSE_MAX_TOKENS) a second kernel,
`_rows_kernel`, evaluates each nonempty expert on its own routes' rows only:
the routed tokens are grouped by expert in the kernel and gathered into
16-row tiles, so a unit's cost follows its expert's route count instead of
the step's token count. Its per-row arithmetic is the all-rows kernel's.
Calls without the option run the all-rows kernel unchanged.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn, interleave_lane

_QUANT_BLOCK = 128
_NUM_BUFFERS = 3
_ACTIVATIONS = ("silu", "swigluoai")

# `_rows_kernel` (sparse rows) only, from r15-merge. Weight-ring slots when a
# unit is a whole (TP-sliced) expert: one more slot keeps more of the small
# units' DMAs in flight where VMEM allows.
_WHOLE_UNIT_BUFFERS = 4
# Rows of the in-kernel token tiles: bf16 MXU operands pack 16 sublanes.
_ROW_ALIGN = 16
# Columns per block of the final output pass (bounds its f32 temporary).
_OUT_COLS = 512
# Rows per block of the sparse mode's f32 token copy.
_TOKEN_COPY_ROWS = 64
# Sparse mode groups tokens by expert on one vreg's lanes (`_group_by_expert`).
_SPARSE_MAX_EXPERTS = 128
# VMEM for the kernel's small operands.
_VMEM_HEADROOM = 1 << 20


def _chunk(i: int) -> int:
    for c in (1024, 512, 384, 256, 128):
        if i % c == 0:
            return c
    return 0


def can_use_dense_expert_fp4(
    hidden, w1, w2, s1, s2, *, activation, sparse_rows: bool = False
) -> bool:
    """Layout and arithmetic support; the token threshold is the caller's."""
    if activation not in _ACTIVATIONS or hidden.ndim != 2:
        return False
    m, h = hidden.shape
    if w1.ndim != 3 or w2.ndim != 3 or s1 is None or s2 is None:
        return False
    e, i, h2 = w2.shape
    fp4 = jnp.float4_e2m1fn
    if (
        hidden.dtype != jnp.bfloat16
        or w1.dtype != fp4
        or w2.dtype != fp4
        or h2 != h
        or w1.shape != (e, h, 2 * i)
        or min(m, h, i, e) <= 0
        or h % _QUANT_BLOCK
    ):
        return False
    if _rows_mode(sparse_rows):
        if e > _SPARSE_MAX_EXPERTS:
            return False
    elif _chunk(i) == 0:
        return False
    if s1.shape != (e, h // _QUANT_BLOCK, 1, 2 * i) or s2.shape != (
        e,
        i // _QUANT_BLOCK,
        1,
        h,
    ):
        return False
    if s1.dtype != jnp.float32 or s2.dtype != jnp.float32:
        return False
    try:
        info = pltpu.get_tpu_info()
    except ValueError:  # not a TPU (e.g. a CPU host)
        return False
    if not info.is_matmul_supported(jnp.bfloat16, jnp.float8_e4m3fn):
        return False
    if _rows_mode(sparse_rows):
        return bool(_rows_chunk(m, h, i, info.vmem_capacity_bytes))
    return _vmem_bytes(m, h, i) <= info.vmem_capacity_bytes * 4 // 5


def _rows_mode(sparse_rows: bool) -> bool:
    """Whether a call runs `_rows_kernel` (else the all-rows `_kernel`)."""
    return sparse_rows


def _vmem_bytes(m: int, h: int, i: int) -> int:
    """Scratch of one call: the weight ring, its scales (the size-1 axis padded
    to an 8-row f32 tile) and the token-sized buffers."""
    c = _chunk(i)
    weights = 3 * h * c // 2
    scales = (2 * (h // _QUANT_BLOCK) * c + (c // _QUANT_BLOCK) * h) * 8 * 4
    mp = m + (-m % 16)
    return _NUM_BUFFERS * (weights + scales) + mp * h * (2 + 4 + 4 + 2)


def _kernel(
    active_ref,  # SMEM [E] (scalar prefetch)
    x_ref,  # VMEM [m, h] bf16
    coeff_ref,  # VMEM [m, E_pad] f32
    w1_hbm,  # [E, h, 2i] fp4
    w2_hbm,  # [E, i, h] fp4
    s1_hbm,  # [E, h / 128, 1, 2i] f32
    s2_hbm,  # [E, i / 128, 1, h] f32
    out_ref,  # VMEM [m, h] bf16
    wg_ref,
    wu_ref,
    wd_ref,
    sg_ref,
    su_ref,
    sd_ref,
    expert_acc_ref,  # VMEM [m, h] f32: the current expert's down projection
    acc_ref,  # VMEM [m, h] f32: the routing-weighted sum
    active_ids_ref,  # SMEM [E]
    sem_ref,
    *,
    h,
    i,
    chunk,
    experts,
    activation,
):
    fp8 = jnp.float8_e4m3fn
    num_chunks = i // chunk
    acc_ref[...] = jnp.zeros_like(acc_ref)
    expert_columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))
    units = count * num_chunks

    def transfers(unit):
        e = active_ids_ref[unit // num_chunks]
        col = pl.multiple_of((unit % num_chunks) * chunk, chunk)
        blk = (unit % num_chunks) * (chunk // _QUANT_BLOCK)
        slot = unit % _NUM_BUFFERS
        sem = sem_ref.at[slot]
        return [
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(col, chunk)], wg_ref.at[slot], sem.at[0]
            ),
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(i + col, chunk)], wu_ref.at[slot], sem.at[1]
            ),
            pltpu.make_async_copy(
                w2_hbm.at[e, pl.ds(col, chunk), :], wd_ref.at[slot], sem.at[2]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(col, chunk)], sg_ref.at[slot], sem.at[3]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(i + col, chunk)], su_ref.at[slot], sem.at[4]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, pl.ds(blk, chunk // _QUANT_BLOCK), :],
                sd_ref.at[slot],
                sem.at[5],
            ),
        ]

    def compute(unit):
        slot = unit % _NUM_BUFFERS
        c = unit % num_chunks
        x = x_ref[...]
        gate = jnp.zeros((x.shape[0], chunk), jnp.float32)
        up = jnp.zeros((x.shape[0], chunk), jnp.float32)
        for k in range(0, h, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            xk = x[:, k : k + _QUANT_BLOCK]
            gate += (
                jnp.matmul(
                    xk,
                    wg_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=jnp.float32,
                )
                * sg_ref[slot, b]
            )
            up += (
                jnp.matmul(
                    xk,
                    wu_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=jnp.float32,
                )
                * su_ref[slot, b]
            )
        # apply_act_fn takes [gate | up] lane-interleaved (its dispatch is reused).
        act = apply_act_fn(interleave_lane(gate, up), activation).astype(x.dtype)
        down = jnp.zeros((x.shape[0], h), jnp.float32)
        for k in range(0, chunk, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            down += (
                jnp.matmul(
                    act[:, k : k + _QUANT_BLOCK],
                    wd_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=jnp.float32,
                )
                * sd_ref[slot, b]
            )

        @pl.when(c == 0)
        def first():
            expert_acc_ref[...] = down

        @pl.when(c > 0)
        def rest():
            expert_acc_ref[...] += down

        @pl.when(c == num_chunks - 1)
        def finish():
            e = active_ids_ref[unit // num_chunks]
            coeff = jnp.sum(
                jnp.where(expert_columns == e, coeff_ref[...], 0),
                axis=1,
                keepdims=True,
            )
            out_e = expert_acc_ref[...].astype(out_ref.dtype).astype(jnp.float32)
            # Zero routing must suppress even a non-finite unused output.
            acc_ref[...] += jnp.where(coeff != 0, out_e * coeff, 0)

    @pl.when(units > 0)
    def run():
        for initial in range(_NUM_BUFFERS - 1):

            @pl.when(units > initial)
            def prime(*, initial=initial):
                for copy in transfers(jnp.int32(initial)):
                    copy.start()

        def body(unit, carry):
            for copy in transfers(unit):
                copy.wait()

            @pl.when(unit + _NUM_BUFFERS - 1 < units)
            def prefetch():
                for copy in transfers(unit + _NUM_BUFFERS - 1):
                    copy.start()

            compute(unit)
            return carry

        lax.fori_loop(0, units, body, 0)

    out_ref[...] = acc_ref[...].astype(out_ref.dtype)


def _rows_chunk(m: int, h: int, i: int, vmem_capacity: int) -> int:
    """`_rows_kernel`'s widest unit of intermediate columns whose buffers fit
    in VMEM, or 0."""
    budget = vmem_capacity - _VMEM_HEADROOM
    for c in (1024, 768, 512, 384, 256, 128):
        if i % c == 0 and _rows_vmem_bytes(m, h, c) <= budget:
            return c
    return 0


def _rows_vmem_bytes(m: int, h: int, chunk: int, slots: int = _NUM_BUFFERS) -> int:
    """VMEM of one `_rows_kernel` call: the weight ring (VMEM holds fp4 one
    byte per element) and its scales, the bf16 input and output, the f32
    tokens, expert output and sum, a few 16-row tiles, and the temporaries of
    `_group_by_expert` (the triangular matrix, the one-hot, pos and hit).
    Whole-expert units hold gate|up in one ring: the same bytes."""
    weights = 3 * h * chunk
    scales = (2 * (h // _QUANT_BLOCK) * chunk + (chunk // _QUANT_BLOCK) * h) * 4
    mp = m + (-m % _ROW_ALIGN)
    tokens = mp * h * (2 + 4 + 4 + 4 + 2) + 16 * _ROW_ALIGN * h
    width = mp + (-mp % _SPARSE_MAX_EXPERTS)
    tokens += mp * mp * 2 + mp * width * 6 + mp * _SPARSE_MAX_EXPERTS * 8
    return slots * (weights + scales) + tokens


def _group_by_expert(id_cols, m, mp, experts, stage, smem, sem):
    """Each local expert's tokens, ascending, without a scalar loop over the
    routes: ``smem[e, k]`` the k-th token routing to expert e and
    ``smem[experts, e]`` their count (a token naming e twice counts once).
    ``stage`` is the [rows, width] int32 VMEM it is built in."""
    f32, bf16 = jnp.float32, jnp.bfloat16
    lanes = _SPARSE_MAX_EXPERTS
    width = stage.shape[1]
    columns = lax.broadcasted_iota(jnp.int32, (m, lanes), 1)
    hit = jnp.zeros((m, lanes), jnp.bool_)
    for col in id_cols:
        hit |= (col == columns) & (col >= 0) & (col < experts)
    hit = hit.astype(f32)
    if mp != m:
        hit = jnp.concatenate([hit, jnp.zeros((mp - m, lanes), f32)], 0)
    # Position of each token among its expert's tokens: a strictly
    # lower-triangular ones matrix sums the earlier rows (exact in f32).
    row = lax.broadcasted_iota(jnp.int32, (mp, mp), 0)
    earlier = (lax.broadcasted_iota(jnp.int32, (mp, mp), 1) < row).astype(bf16)
    pos = jnp.dot(earlier, hit.astype(bf16), preferred_element_type=f32)
    counts = jnp.sum(hit, axis=0, keepdims=True)  # [1, lanes]
    slots = lax.broadcasted_iota(jnp.int32, (mp, width), 1).astype(f32)
    tokens = lax.broadcasted_iota(jnp.int32, (8, mp), 1)
    # Token ids as two bf16-exact parts (ids above 256 are not bf16-exact).
    sub = lax.broadcasted_iota(jnp.int32, (8, mp), 0)
    parts = jnp.where(sub == 0, tokens // 16, jnp.where(sub == 1, tokens % 16, 0))
    parts = parts.astype(bf16)
    expert_lanes = lax.broadcasted_iota(jnp.int32, (1, lanes), 1)

    def per_expert(e, carry):
        sel = expert_lanes == e
        pos_e = jnp.sum(jnp.where(sel, pos, 0), axis=1, keepdims=True)  # [mp, 1]
        hit_e = jnp.sum(jnp.where(sel, hit, 0), axis=1, keepdims=True)
        onehot = ((pos_e == slots) & (hit_e > 0)).astype(bf16)  # [mp, width]
        r = jnp.dot(parts, onehot, preferred_element_type=f32)
        stage[pl.ds(e, 1), :] = (16 * r[0:1] + r[1:2]).astype(jnp.int32)
        return carry

    lax.fori_loop(0, experts, per_expert, 0)
    count_row = counts.astype(jnp.int32)
    if width > lanes:
        count_row = jnp.concatenate(
            [count_row, jnp.zeros((1, width - lanes), jnp.int32)], 1
        )
    stage[experts : experts + 1, :] = count_row[:, :width]
    copy = pltpu.make_async_copy(stage, smem, sem.at[0])
    copy.start()
    copy.wait()


def _rows_kernel(
    ids_smem,  # SMEM [m * topk] local expert ids, -1 off this rank (unused)
    x_ref,  # VMEM [m, h] bf16
    ids_ref,  # VMEM [m, topk] int32
    tw_ref,  # VMEM [m, topk] f32 routing weights
    w1_hbm,  # [E, h, 2i] fp4
    w2_hbm,  # [E, i, h] fp4
    s1_hbm,  # [E, h / 128, 1, 2i] f32
    s2_hbm,  # [E, i / 128, 1, h] f32
    out_ref,  # VMEM [m, h] bf16
    wg_ref,
    wu_ref,
    wd_ref,
    sg_ref,
    su_ref,
    sd_ref,
    x_pad_ref,  # VMEM [mp, h] f32: the tokens, non-finite values zeroed,
    # padded with zero rows (single rows load only from 32-bit refs)
    coeff_ref,  # VMEM [mp, E_pad] f32: each token's weight per local expert
    expert_acc_ref,  # VMEM [mp, h] f32: the current expert's down projection
    acc_ref,  # VMEM [mp, h] f32: the routing-weighted sum
    active_ref,  # SMEM [E]: the expert's route count
    active_ids_ref,  # SMEM [E]
    sem_ref,
    group_vmem,  # VMEM `_group_by_expert` staging
    sorted_ref,  # SMEM: its copy
    group_sem,
    rows_ref,  # VMEM [16, h] f32: one tile's rows
    *,
    h,
    i,
    chunk,
    experts,
    activation,
):
    del ids_smem
    fp8 = jnp.float8_e4m3fn
    f32 = jnp.float32
    m = x_ref.shape[0]
    mp = acc_ref.shape[0]
    slots = sem_ref.shape[0]
    num_chunks = i // chunk
    # A whole expert per unit (TP-sliced experts): one contiguous w13 copy.
    whole = chunk == i

    topk = ids_ref.shape[1]
    ids = ids_ref[...]
    tw = tw_ref[...]
    id_cols = [ids[:, j : j + 1] for j in range(topk)]
    tw_cols = [tw[:, j : j + 1] for j in range(topk)]

    _group_by_expert(id_cols, m, mp, experts, group_vmem, sorted_ref, group_sem)

    def mark_counts(e, carry):
        active_ref[e] = sorted_ref[experts, e]
        return carry

    lax.fori_loop(0, experts, mark_counts, 0)

    def compact(e, count):
        @pl.when(active_ref[e] != 0)
        def record():
            active_ids_ref[count] = e

        return count + (active_ref[e] != 0).astype(jnp.int32)

    count = lax.fori_loop(0, experts, compact, jnp.int32(0))
    units = count * num_chunks

    def transfers(unit):
        e = active_ids_ref[unit // num_chunks]
        col = pl.multiple_of((unit % num_chunks) * chunk, chunk)
        blk = (unit % num_chunks) * (chunk // _QUANT_BLOCK)
        slot = unit % slots
        sem = sem_ref.at[slot]
        if whole:
            return [
                pltpu.make_async_copy(w1_hbm.at[e], wg_ref.at[slot], sem.at[0]),
                pltpu.make_async_copy(w2_hbm.at[e], wd_ref.at[slot], sem.at[2]),
                pltpu.make_async_copy(s1_hbm.at[e], sg_ref.at[slot], sem.at[3]),
                pltpu.make_async_copy(s2_hbm.at[e], sd_ref.at[slot], sem.at[5]),
            ]
        return [
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(col, chunk)], wg_ref.at[slot], sem.at[0]
            ),
            pltpu.make_async_copy(
                w1_hbm.at[e, :, pl.ds(i + col, chunk)], wu_ref.at[slot], sem.at[1]
            ),
            pltpu.make_async_copy(
                w2_hbm.at[e, pl.ds(col, chunk), :], wd_ref.at[slot], sem.at[2]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(col, chunk)], sg_ref.at[slot], sem.at[3]
            ),
            pltpu.make_async_copy(
                s1_hbm.at[e, :, :, pl.ds(i + col, chunk)], su_ref.at[slot], sem.at[4]
            ),
            pltpu.make_async_copy(
                s2_hbm.at[e, pl.ds(blk, chunk // _QUANT_BLOCK), :],
                sd_ref.at[slot],
                sem.at[5],
            ),
        ]

    for initial in range(slots - 1):

        @pl.when(units > initial)
        def prime(*, initial=initial):
            for copy in transfers(jnp.int32(initial)):
                copy.start()

    # Token-side setup overlaps the first weight copies.
    def pad_rows(v):
        if mp == m:
            return v
        return jnp.concatenate([v, jnp.zeros((mp - m, v.shape[1]), v.dtype)], 0)

    expert_columns = lax.broadcasted_iota(jnp.int32, (m, coeff_ref.shape[1]), 1)
    coeff = jnp.zeros((m, coeff_ref.shape[1]), f32)
    for col, w in zip(id_cols, tw_cols):
        # Zero routing must suppress even a non-finite routing weight.
        w = jnp.where(jnp.isfinite(w), w, 0)
        coeff += jnp.where(col == expert_columns, w, 0)
    coeff_ref[...] = pad_rows(coeff)
    acc_ref[...] = jnp.zeros_like(acc_ref)

    # f32 tokens, written in row blocks so no [mp, h] temporary exists.
    for r0 in range(0, mp, _TOKEN_COPY_ROWS):
        r1 = min(r0 + _TOKEN_COPY_ROWS, mp)
        if r0 >= m:
            x_pad_ref[r0:r1, :] = jnp.zeros((r1 - r0, h), f32)
            continue
        v = x_ref[r0 : min(r1, m), :].astype(f32)
        x_pad_ref[r0 : min(r1, m), :] = jnp.where(jnp.isfinite(v), v, 0)
        if r1 > m:
            x_pad_ref[m:r1, :] = jnp.zeros((r1 - m, h), f32)

    def expert_mlp(x, slot):
        """The unit's gate, up and down columns on rows ``x``: f32 [rows, h]."""
        gate = jnp.zeros((x.shape[0], chunk), f32)
        up = jnp.zeros((x.shape[0], chunk), f32)
        for k in range(0, h, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            xk = x[:, k : k + _QUANT_BLOCK]
            if whole:
                gu = (
                    jnp.matmul(
                        xk,
                        wg_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                        preferred_element_type=f32,
                    )
                    * sg_ref[slot, b]
                )
                gate += gu[:, :chunk]
                up += gu[:, chunk:]
                continue
            gate += (
                jnp.matmul(
                    xk,
                    wg_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=f32,
                )
                * sg_ref[slot, b]
            )
            up += (
                jnp.matmul(
                    xk,
                    wu_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=f32,
                )
                * su_ref[slot, b]
            )
        # apply_act_fn takes [gate | up] lane-interleaved (its dispatch is reused).
        act = apply_act_fn(interleave_lane(gate, up), activation).astype(x.dtype)
        down = jnp.zeros((x.shape[0], h), f32)
        for k in range(0, chunk, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            down += (
                jnp.matmul(
                    act[:, k : k + _QUANT_BLOCK],
                    wd_ref[slot, k : k + _QUANT_BLOCK, :].astype(fp8),
                    preferred_element_type=f32,
                )
                * sd_ref[slot, b]
            )
        return down

    tile = _ROW_ALIGN
    expert_lanes = lax.broadcasted_iota(jnp.int32, (1, coeff_ref.shape[1]), 1)

    def compute(unit):
        """The unit on its expert's routes only, 16 rows at a time: each tile's
        rows load and its weighted outputs add back one token row at a time,
        so the cost does not grow with the step's token count."""
        slot = unit % slots
        c = unit % num_chunks
        e = active_ids_ref[unit // num_chunks]
        end = sorted_ref[experts, e]
        tiles = (end + tile - 1) // tile

        def run(t, carry):
            base = t * tile
            rows = pl.ds(pl.multiple_of(t * tile, tile), tile)
            for j in range(tile):
                # Rows past the expert's routes repeat a real token; their
                # outputs are never added.
                tok = sorted_ref[e, jnp.minimum(base + j, end - 1)]
                rows_ref[j : j + 1, :] = x_pad_ref[pl.ds(tok, 1), :]
            down = expert_mlp(rows_ref[...].astype(jnp.bfloat16), slot)

            @pl.when(c == 0)
            def first():
                expert_acc_ref[rows, :] = down

            @pl.when(c > 0)
            def rest():
                expert_acc_ref[rows, :] += down

            @pl.when(c == num_chunks - 1)
            def finish():
                # The expert's output rounds to bf16 before its routing weight
                # (the GMM path's rounding); a non-finite one is dropped.
                # (isfinite takes f32 in Mosaic: test the bf16 round trip.)
                out_e = expert_acc_ref[rows, :].astype(jnp.bfloat16).astype(f32)
                out_e = jnp.where(jnp.isfinite(out_e), out_e, 0)
                for j in range(tile):

                    @pl.when(base + j < end)
                    def add(j=j):
                        tok = sorted_ref[e, base + j]
                        coeff = jnp.sum(
                            jnp.where(
                                expert_lanes == e, coeff_ref[pl.ds(tok, 1), :], 0
                            ),
                            axis=1,
                            keepdims=True,
                        )
                        acc_ref[pl.ds(tok, 1), :] += coeff * out_e[j : j + 1, :]

            return carry

        lax.fori_loop(0, tiles, run, 0)

    def body(unit, carry):
        for copy in transfers(unit):
            copy.wait()

        @pl.when(unit + slots - 1 < units)
        def prefetch():
            for copy in transfers(unit + slots - 1):
                copy.start()

        compute(unit)
        return carry

    lax.fori_loop(0, units, body, 0)

    # Column blocks: no [m, h] temporary.
    for c0 in range(0, h, _OUT_COLS):
        cols = slice(c0, min(c0 + _OUT_COLS, h))
        out_ref[:, cols] = acc_ref[:m, cols].astype(out_ref.dtype)


def _dense_expert_moe_fp4_rows(
    hidden, w1, w2, s1, s2, topk_weights, local_ids, *, activation
):
    """`dense_expert_moe_fp4` on `_rows_kernel` (sparse rows)."""
    m, h = hidden.shape
    experts, i, _ = w2.shape
    # `_group_by_expert` holds the local experts on one vreg's lanes.
    assert experts <= _SPARSE_MAX_EXPERTS, experts
    info = pltpu.get_tpu_info()
    chunk = _rows_chunk(m, h, i, info.vmem_capacity_bytes)
    whole = chunk == i
    slots = _NUM_BUFFERS
    budget = info.vmem_capacity_bytes - _VMEM_HEADROOM
    if whole and _rows_vmem_bytes(m, h, chunk, _WHOLE_UNIT_BUFFERS) <= budget:
        slots = _WHOLE_UNIT_BUFFERS
    # Whole f32 sublane tiles of tokens; the kernel pads to _ROW_ALIGN itself.
    pad = -m % 8
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        local_ids = jnp.pad(local_ids, ((0, pad), (0, 0)), constant_values=-1)
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
    mk = m + pad
    mp = mk + (-mk % _ROW_ALIGN)
    prefetch = local_ids.astype(jnp.int32).reshape(-1)
    local_ids = local_ids.astype(jnp.int32)
    topk_weights = topk_weights.astype(jnp.float32)
    lanes = info.num_lanes
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    # `_group_by_expert`: rows 0..experts-1 the tokens, row experts the counts.
    group_shape = (
        experts + 1 + (-(experts + 1)) % 8,
        max(-(-mp // lanes), 1) * lanes,
    )
    gate_cols = 2 * chunk if whole else chunk
    out = pl.pallas_call(
        functools.partial(
            _rows_kernel,
            h=h,
            i=i,
            chunk=chunk,
            experts=experts,
            activation=activation,
        ),
        out_shape=jax.ShapeDtypeStruct((mk, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[vmem, vmem, vmem, *[hbm] * 4],
            out_specs=vmem,
            scratch_shapes=[
                # Whole-expert units keep gate|up in the first ring: no up ring.
                pltpu.VMEM((slots, h, gate_cols), w1.dtype),
                pltpu.VMEM((1, 128, 128) if whole else (slots, h, chunk), w1.dtype),
                pltpu.VMEM((slots, chunk, h), w2.dtype),
                pltpu.VMEM((slots, h // _QUANT_BLOCK, 1, gate_cols), jnp.float32),
                pltpu.VMEM(
                    (1, 1, 1, 128) if whole else (slots, h // _QUANT_BLOCK, 1, chunk),
                    jnp.float32,
                ),
                pltpu.VMEM((slots, chunk // _QUANT_BLOCK, 1, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, experts + (-experts) % lanes), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((slots, 6)),
                pltpu.VMEM(group_shape, jnp.int32),
                pltpu.SMEM(group_shape, jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.VMEM((_ROW_ALIGN, h), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=info.vmem_capacity_bytes),
        name=f"dense_expert_moe_fp4-e_{experts}-m_{mp}-h_{h}-i_{i}-c_{chunk}-sparse",
    )(
        prefetch,
        hidden,
        local_ids,
        topk_weights,
        w1,
        w2,
        # 4-D as the GMM path holds them: a reshape that drops the size-1
        # axis is a relayout copy of every scale on each call.
        s1,
        s2,
    )
    return out[:m] if pad else out


@functools.partial(jax.jit, static_argnames=("activation", "sparse_rows"))
def dense_expert_moe_fp4(
    hidden,
    w1,
    w2,
    s1,
    s2,
    topk_weights,
    local_ids,
    *,
    activation="swigluoai",
    sparse_rows=False,
):
    """Local routed contribution of fp4 experts, [tokens, hidden] bf16, as
    fused_moe_func's GMM path returns it. IDs outside this rank must be -1.
    Callers check can_use_dense_expert_fp4 (with the same ``sparse_rows``)
    first. Non-finite hidden states and weights are zeroed, as the GMM path's
    one-hot permute (which serves these token counts) does.

    ``sparse_rows`` evaluates each expert on its own routes' rows only
    (`_rows_kernel`); a non-finite expert output is then dropped even where
    its routing weight is nonzero."""
    if _rows_mode(sparse_rows):
        return _dense_expert_moe_fp4_rows(
            hidden, w1, w2, s1, s2, topk_weights, local_ids, activation=activation
        )
    m, h = hidden.shape
    experts, i, _ = w2.shape
    chunk = _chunk(i)
    sublane = 16
    pad = -m % sublane
    x = jnp.where(jnp.isfinite(hidden), hidden, 0)
    expert_ids = jnp.arange(experts, dtype=jnp.int32)[None, :]
    weights = topk_weights.astype(jnp.float32)
    weights = jnp.where(jnp.isfinite(weights), weights, 0)
    coeff = jnp.zeros((m, experts), jnp.float32)
    for route in range(local_ids.shape[1]):
        selected = local_ids[:, route : route + 1] == expert_ids
        coeff = coeff + selected.astype(jnp.float32) * weights[:, route : route + 1]
    active = jnp.any(coeff != 0, axis=0).astype(jnp.int32)
    lanes = pltpu.get_tpu_info().num_lanes
    coeff = jnp.pad(coeff, ((0, pad), (0, (-experts) % lanes)))
    x = jnp.pad(x, ((0, pad), (0, 0)))
    mp = m + pad
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            h=h,
            i=i,
            chunk=chunk,
            experts=experts,
            activation=activation,
        ),
        out_shape=jax.ShapeDtypeStruct((mp, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.VMEM),
                pl.BlockSpec(memory_space=pltpu.VMEM),
                *[pl.BlockSpec(memory_space=pltpu.HBM)] * 4,
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
            scratch_shapes=[
                pltpu.VMEM((_NUM_BUFFERS, h, chunk), w1.dtype),
                pltpu.VMEM((_NUM_BUFFERS, h, chunk), w1.dtype),
                pltpu.VMEM((_NUM_BUFFERS, chunk, h), w2.dtype),
                pltpu.VMEM((_NUM_BUFFERS, h // _QUANT_BLOCK, 1, chunk), jnp.float32),
                pltpu.VMEM((_NUM_BUFFERS, h // _QUANT_BLOCK, 1, chunk), jnp.float32),
                pltpu.VMEM((_NUM_BUFFERS, chunk // _QUANT_BLOCK, 1, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((_NUM_BUFFERS, 6)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes
        ),
        name=f"dense_expert_moe_fp4-e_{experts}-m_{mp}-h_{h}-i_{i}-c_{chunk}",
    )(
        active,
        x,
        coeff,
        w1,
        w2,
        # 4-D as the GMM path holds them: a reshape that drops the size-1
        # axis is a relayout copy of every scale on each call.
        s1,
        s2,
    )
    return out[:m]
