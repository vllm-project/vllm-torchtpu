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
one-hot permute, the grouped GEMM and the combine with one call; the routing
coefficients and the set of nonempty experts are derived in the kernel from the
raw top-k ids and weights (an expert counts as nonempty when any id names it,
even if its routing weights are zero). The fp4 weights feed the MXU as fp8
(exact) against bf16 activations, and each 128-row block scale is applied to
the block's partial product, as gmm_v2's decode tiles do. Rounding follows the
GMM path: the activation is rounded to bf16 before the down projection, each
expert's output to bf16 before its routing weight is applied.

Optionally the call also evaluates an always-on shared expert (`SharedExpert`)
and returns routed + shared: its e4m3 weights are copied into VMEM alongside
the routed ring, so their DMA overlaps the routed experts' instead of following
them as separate XLA matmuls.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.megablox.gmm_v2 import apply_act_fn, interleave_lane

_QUANT_BLOCK = 128
_NUM_BUFFERS = 3
_ACTIVATIONS = ("silu", "swigluoai")
# Rows of the in-kernel token buffer: bf16 MXU operands pack 16 sublanes.
_ROW_ALIGN = 16
_SHARED_K_BLOCK = 128
# MiniMax-M3's shared-expert SwiGLU-OAI clamp and sigmoid scale (1.702 in
# bf16); dense_fp4_shared admits only shared experts with these values.
_SWIGLU_LIMIT = 7.0
_SWIGLU_ALPHA = 1.703125
# VMEM for the kernel's small operands.
_VMEM_HEADROOM = 1 << 20
# `_vmem_limit_bytes` at decode sizes: token scratch plus compiler temporaries
# per token element, and a margin. tpu7x AOT compiles (libtpu 0.0.50) of the
# MiniMax-M3 EP8 call (h 6144, i 3072, shared 384) need 50.3 MiB at 16 rows
# and 52.7 MiB at 32; XLA keeps the call's neighbours in VMEM only with
# limits up to ~53.7 MiB. The sparse-row mode's need stays within the same
# per-element bound (EP4xTP2 call, i 1536: 54.6 / 56.8 / 46.1 / 50.7 MiB at
# 48 / 64 / 96 / 128 rows, chunk 768 / 768 / 512 / 512), above that neighbour
# limit at 48 and 64 rows: 512-column units would fit under it but leave the
# call issue-bound (twin 48-row step: +8.7 us per MoE layer for -4.3 us of
# neighbour stalls).
_CALIBRATED_ROWS = 32
_CALIBRATED_TOKEN_BYTES = 24
_CALIBRATED_MARGIN = 256 << 10


class SharedExpert(NamedTuple):
    """A rank's slice of an always-on SwiGLU-OAI MLP with e4m3 K-major weights
    and one f32 scale per output column, rounded as torch_tpu evaluates
    MiniMax-M3's shared expert on bf16 tensors: each projection
    ``bf16(bf16(x @ bf16(w)) * s)``, the activation per op
    (`shared_activation`), and the bf16 sum with the routed output."""

    w1: jax.Array  # [h, 2 * i_s] e4m3: [gate | up]
    s1: jax.Array  # [1, 2 * i_s] f32
    w2: jax.Array  # [i_s, h] e4m3
    s2: jax.Array  # [1, h] f32


def shared_activation(gate: jax.Array, up: jax.Array) -> jax.Array:
    """SiluAndMulWithClamp(7, 1.702, 1) on bf16 values held in f32, rounded to
    bf16 after every op as torch_tpu does (alpha too): bit-identical to the
    torch path on tpu7x."""

    def bf16(v):
        return v.astype(jnp.bfloat16).astype(jnp.float32)

    gate = jnp.minimum(gate, _SWIGLU_LIMIT)
    up = jnp.clip(up, -_SWIGLU_LIMIT, _SWIGLU_LIMIT)
    sig = bf16(jax.nn.sigmoid(bf16(_SWIGLU_ALPHA * gate)))
    return bf16(bf16(gate * sig) * bf16(up + 1.0))


def _chunk(
    m: int, h: int, i: int, i_s: int, vmem_capacity: int, sparse: bool = False
) -> int:
    """The widest unit of intermediate columns whose buffers fit in VMEM, or 0."""
    budget = vmem_capacity - _VMEM_HEADROOM
    for c in (1024, 768, 512, 384, 256, 128):
        if i % c == 0 and _vmem_bytes(m, h, c, i_s, sparse) <= budget:
            return c
    return 0


def _shared_ok(shared: SharedExpert | None, h: int, activation: str) -> bool:
    if shared is None:
        return True
    fp8 = jnp.float8_e4m3fn
    i_s = shared.w2.shape[0]
    return (
        activation == "swigluoai"
        and i_s > 0
        and i_s % _QUANT_BLOCK == 0
        and shared.w1.shape == (h, 2 * i_s)
        and shared.w2.shape == (i_s, h)
        and shared.s1.shape == (1, 2 * i_s)
        and shared.s2.shape == (1, h)
        and shared.w1.dtype == fp8
        and shared.w2.dtype == fp8
        and shared.s1.dtype == jnp.float32
        and shared.s2.dtype == jnp.float32
    )


def can_use_dense_expert_fp4(
    hidden,
    w1,
    w2,
    s1,
    s2,
    *,
    activation,
    shared: SharedExpert | None = None,
    sparse_rows: bool = False,
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
        or not _shared_ok(shared, h, activation)
    ):
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
    i_s = 0 if shared is None else shared.w2.shape[0]
    return info.is_matmul_supported(jnp.bfloat16, jnp.float8_e4m3fn) and bool(
        _chunk(m, h, i, i_s, info.vmem_capacity_bytes, sparse_rows)
    )


def _vmem_limit_bytes(
    m: int, h: int, chunk: int, i_s: int, vmem_capacity: int, sparse: bool = False
) -> int:
    """The call's VMEM limit. VMEM it reserves cannot hold the activations live
    across it (the residual stream, its operands and output): XLA spills them
    to HBM, and each op that reads or writes one pays an HBM round trip. At up
    to _CALIBRATED_ROWS rows, and in the sparse-row mode, the limit is the
    measured need (the token temporaries at _CALIBRATED_TOKEN_BYTES, not
    `_vmem_bytes`'s budget); larger dense calls keep the whole of VMEM."""
    mp = m + (-m % _ROW_ALIGN)
    if not sparse and mp > _CALIBRATED_ROWS:
        return vmem_capacity
    need = _weight_vmem_bytes(h, chunk, i_s) + mp * h * _CALIBRATED_TOKEN_BYTES
    return min(vmem_capacity, need + _CALIBRATED_MARGIN)


def _weight_vmem_bytes(h: int, chunk: int, i_s: int) -> int:
    """The weight ring (VMEM holds fp4 one byte per element), its scales and
    the shared expert's weights."""
    weights = 3 * h * chunk
    scales = (2 * (h // _QUANT_BLOCK) * chunk + (chunk // _QUANT_BLOCK) * h) * 4
    return _NUM_BUFFERS * (weights + scales) + 3 * h * i_s


def _vmem_bytes(m: int, h: int, chunk: int, i_s: int, sparse: bool = False) -> int:
    """VMEM of one call: `_weight_vmem_bytes`, the token-sized scratch and the
    compiler's temporaries (tpu7x AOT compiles: ~16 bytes per token element
    when a unit evaluates every token, measured at 16 and 64 rows). A
    sparse-row call holds _CALIBRATED_TOKEN_BYTES per token element (its
    measured need, 54.4 / 56.6 MiB at 48 / 64 rows and 64.6 MiB at 128 with
    768-column units) and 16 per element of one 16-row tile."""
    mp = m + (-m % _ROW_ALIGN)
    if sparse:
        tokens = mp * _CALIBRATED_TOKEN_BYTES + 16 * _ROW_ALIGN
    else:
        tokens = mp * (2 + 4 + 4 + 2 + 16)
    return _weight_vmem_bytes(h, chunk, i_s) + tokens * h


def _choose_replicas(
    ids_ref,
    table_ref,
    block_index,
    choice_vmem,
    choice_smem,
    sem,
    *,
    block,
    groups,
):
    """Which copy of each replicated expert serves this step, the same choice
    on every block (each sees all routes): a block's load is the slots it must
    stream, first its unreplicated touched slots, then each touched replicated
    slot in table order goes to the copy's block if that block's load is lower
    than its home block's (home on a tie).

    Fills choice_smem's rows: 0 each slot's local expert on this block (-1:
    another block's), 3 at [r] the slot of local expert block + r (a copy);
    rows 1 and 2 hold the touched slots and the blocks' loads."""
    m, topk = ids_ref.shape
    num_slots = choice_vmem.shape[1]
    replicas = table_ref.shape[0] // 4
    lanes = lax.broadcasted_iota(jnp.int32, (1, num_slots), 1)
    # Touched and replicated slots, and the blocks' unreplicated loads, as
    # vectors; one DMA hands them to the scalar decisions.
    ids = ids_ref[...]
    routed = lax.broadcasted_iota(jnp.int32, (m, num_slots), 1)
    hits = jnp.zeros((m, num_slots), jnp.int32)
    for route in range(topk):
        hits = jnp.maximum(
            hits, (ids[:, route : route + 1] == routed).astype(jnp.int32)
        )
    touched = jnp.max(hits, axis=0, keepdims=True)
    replicated = jnp.zeros_like(touched)
    for d in range(replicas):
        replicated = jnp.where(lanes == table_ref[4 * d], 1, replicated)
    unreplicated = touched * (1 - replicated)
    home = lanes // block
    loads = jnp.zeros_like(touched)
    for g in range(groups):
        loads = jnp.where(
            lanes == g, jnp.sum(jnp.where(home == g, unreplicated, 0)), loads
        )
    own = (unreplicated != 0) & (home == block_index)
    rows = [jnp.where(own, lanes - home * block, -1), touched, loads]
    rows += [jnp.zeros_like(touched)] * (choice_vmem.shape[0] - len(rows))
    choice_vmem[...] = jnp.concatenate(rows, axis=0)
    copy = pltpu.make_async_copy(choice_vmem, choice_smem, sem)
    copy.start()
    copy.wait()

    def decide(d, carry):
        s = table_ref[4 * d]
        home, block_copy, copy_local = (table_ref[4 * d + k] for k in (1, 2, 3))

        @pl.when((s >= 0) & (choice_smem[1, jnp.maximum(s, 0)] != 0))
        def touched():
            to_copy = choice_smem[2, block_copy] < choice_smem[2, home]
            pick = jnp.where(to_copy, block_copy, home)
            choice_smem[2, pick] += 1

            @pl.when(pick == block_index)
            def here():
                @pl.when(to_copy)
                def on_copy():
                    choice_smem[0, s] = copy_local
                    choice_smem[3, copy_local - block] = s

                @pl.when(~to_copy)
                def at_home():
                    choice_smem[0, s] = s - home * block

        return carry

    lax.fori_loop(0, replicas, decide, 0)


def _kernel(
    ids_smem,  # SMEM [m * topk] local expert ids, -1 off this rank (prefetch)
    *refs,
    h,
    i,
    chunk,
    experts,
    activation,
    i_s,
    sparse,
    groups,
):
    # With expert replicas (``groups`` blocks of experts, each with replica
    # slots after its own), ids_smem holds slots instead: [D * 4] the replica
    # table (expert_placement.ExpertPlacement.replica_table) and [1] this
    # block's index come next.
    if groups:
        table_smem, block_smem, *refs = refs
    (
        x_ref,  # VMEM [m, h] bf16
        ids_ref,  # VMEM [m, topk] int32: ids_smem as rows
        tw_ref,  # VMEM [m, topk] f32 routing weights
        w1_hbm,  # [E, h, 2i] fp4
        w2_hbm,  # [E, i, h] fp4
        s1_hbm,  # [E, h / 128, 1, 2i] f32
        s2_hbm,  # [E, i / 128, 1, h] f32
        *refs,
    ) = refs
    if i_s:
        sw1_hbm, ss1_ref, sw2_hbm, ss2_ref, *refs = refs
    (
        out_ref,  # VMEM [m, h] bf16
        wg_ref,
        wu_ref,
        wd_ref,
        sg_ref,
        su_ref,
        sd_ref,
        x_pad_ref,  # VMEM [mp, h] bf16: the tokens, padded with zero rows
        coeff_ref,  # VMEM [mp, E_pad] f32: each token's weight per expert column
        expert_acc_ref,  # VMEM [mp, h] f32: the current expert's down projection
        acc_ref,  # VMEM [mp, h] f32: the routing-weighted sum
        active_ref,  # SMEM [E]: nonzero for an expert some token routes to
        active_ids_ref,  # SMEM [E]
        sem_ref,
        *extra_scratch,
    ) = refs
    shared_scratch = extra_scratch[:4] if i_s else ()
    extra_scratch = extra_scratch[len(shared_scratch) :]
    if sparse:
        # SMEM [E] routes per local expert, [E + 1] their offsets, [m * k] the
        # routes' tokens grouped by expert; VMEM [mp, h] bf16 the gathered
        # rows, [mp, lanes] f32 the current expert's coefficient per token.
        counts_ref, offs_ref, sorted_ref, xs_ref, coeff_rows_ref = extra_scratch[:5]
        extra_scratch = extra_scratch[5:]
    if groups:
        # VMEM and SMEM [8, S] the copy choice (_choose_replicas) and its
        # DMA's semaphore; SMEM [m * k] each route's local expert (`mark`).
        choice_vmem, choice_smem, choice_sem, lids_ref = extra_scratch
    fp8 = jnp.float8_e4m3fn
    f32 = jnp.float32
    m = x_ref.shape[0]
    mp = acc_ref.shape[0]
    num_chunks = i // chunk

    if i_s:
        swg_ref, swu_ref, swd_ref, shared_sem = shared_scratch
        shared_copies = [
            pltpu.make_async_copy(
                sw1_hbm.at[:, pl.ds(0, i_s)], swg_ref, shared_sem.at[0]
            ),
            pltpu.make_async_copy(
                sw1_hbm.at[:, pl.ds(i_s, i_s)], swu_ref, shared_sem.at[1]
            ),
            pltpu.make_async_copy(sw2_hbm, swd_ref, shared_sem.at[2]),
        ]
        for copy in shared_copies:
            copy.start()

    def clear(e, carry):
        active_ref[e] = 0
        if sparse:
            counts_ref[e] = 0
        return carry

    lax.fori_loop(0, experts, clear, 0)
    topk = ids_ref.shape[1]
    routes = ids_smem.shape[0]
    route_ids = ids_smem
    if groups:
        block = experts - table_smem.shape[0] // 4 // groups
        _choose_replicas(
            ids_ref,
            table_smem,
            block_smem[0],
            choice_vmem,
            choice_smem,
            choice_sem,
            block=block,
            groups=groups,
        )
        route_ids = lids_ref

    def column(e):
        """Local expert e's column of coeff_ref: its slot with replicas."""
        if not groups:
            return e
        copy_slot = choice_smem[3, jnp.maximum(e - block, 0)]
        return jnp.where(e < block, block_smem[0] * block + e, copy_slot)

    def first_route(r, e):
        """No earlier route of r's token names e (each token adds an expert's
        output once, weighted by its summed coefficient)."""
        first = True
        for back in range(1, topk):
            earlier = jnp.maximum(r - back, 0)
            first &= (back > r % topk) | (route_ids[earlier] != e)
        return first

    def mark(r, carry):
        if groups:
            s = ids_smem[r]
            e = jnp.where(s >= 0, choice_smem[0, jnp.maximum(s, 0)], -1)
            lids_ref[r] = e
        else:
            e = ids_smem[r]

        @pl.when((e >= 0) & (e < experts))
        def local():
            active_ref[e] = 1
            if sparse:
                counts_ref[e] += first_route(r, e).astype(jnp.int32)

        return carry

    lax.fori_loop(0, ids_smem.shape[0], mark, 0)

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

    for initial in range(_NUM_BUFFERS - 1):

        @pl.when(units > initial)
        def prime(*, initial=initial):
            for copy in transfers(jnp.int32(initial)):
                copy.start()

    if sparse:
        # Group the routes' tokens by local expert (a counting sort) while the
        # first weight copies fly; counts_ref becomes each expert's cursor.
        offs_ref[0] = 0

        def offsets(e, carry):
            offs_ref[e + 1] = offs_ref[e] + counts_ref[e]
            counts_ref[e] = offs_ref[e]
            return carry

        lax.fori_loop(0, experts, offsets, 0)

        def place(r, carry):
            e = route_ids[r]

            @pl.when((e >= 0) & (e < experts) & first_route(r, e))
            def local():
                sorted_ref[counts_ref[e]] = r // topk
                counts_ref[e] += 1

            return carry

        lax.fori_loop(0, routes, place, 0)

    # Token-side setup overlaps the first weight copies.
    def pad_rows(v):
        if mp == m:
            return v
        return jnp.concatenate([v, jnp.zeros((mp - m, v.shape[1]), v.dtype)], 0)

    x = pad_rows(x_ref[...].astype(f32))
    tw = tw_ref[...]
    # Zero routing must suppress even a non-finite routing weight.
    tw = jnp.where(jnp.isfinite(tw), tw, 0)
    ids = ids_ref[...]
    expert_columns = lax.broadcasted_iota(jnp.int32, (m, coeff_ref.shape[1]), 1)
    coeff = jnp.zeros((m, coeff_ref.shape[1]), f32)
    for route in range(ids.shape[1]):
        coeff += jnp.where(
            ids[:, route : route + 1] == expert_columns, tw[:, route : route + 1], 0
        )
    coeff_ref[...] = pad_rows(coeff)
    acc_ref[...] = jnp.zeros_like(acc_ref)
    if i_s:
        # The shared expert sees the tokens as given, the routed experts with
        # non-finite values zeroed (as the GMM path's one-hot permute does).
        # Its output waits in out_ref, which the routed sum is added to last.
        for copy in shared_copies:
            copy.wait()
        xs = x_ref[...]

        def matmul(lhs, w_ref):
            # K blocks keep the MXU operands' temporaries small.
            y = jnp.zeros((lhs.shape[0], w_ref.shape[1]), f32)
            for k in range(0, w_ref.shape[0], _SHARED_K_BLOCK):
                y += jnp.matmul(
                    lhs[:, k : k + _SHARED_K_BLOCK],
                    w_ref[k : k + _SHARED_K_BLOCK, :],
                    preferred_element_type=f32,
                )
            return y

        def projection(lhs, w_ref, scale):
            y = matmul(lhs, w_ref).astype(jnp.bfloat16).astype(f32) * scale
            return y.astype(jnp.bfloat16)

        gate = projection(xs, swg_ref, ss1_ref[:, :i_s]).astype(f32)
        up = projection(xs, swu_ref, ss1_ref[:, i_s:]).astype(f32)
        act = shared_activation(gate, up).astype(jnp.bfloat16)
        out_ref[...] = projection(act, swd_ref, ss2_ref[...])
    x_pad_ref[...] = jnp.where(jnp.isfinite(x), x, 0).astype(jnp.bfloat16)

    def expert_mlp(x, slot):
        """The unit's gate, up and down columns on rows ``x``: f32 [rows, h]."""
        gate = jnp.zeros((x.shape[0], chunk), f32)
        up = jnp.zeros((x.shape[0], chunk), f32)
        for k in range(0, h, _QUANT_BLOCK):
            b = k // _QUANT_BLOCK
            xk = x[:, k : k + _QUANT_BLOCK]
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
    tile_rows = lax.broadcasted_iota(jnp.int32, (tile, 1), 0)

    def tile_tokens(begin, end):
        """The tokens of routes [begin, begin + tile) as a [tile, 1] column,
        -1 at or past ``end``."""
        col = jnp.full((tile, 1), -1, jnp.int32)
        for j in range(tile):
            r = begin + j
            tok = jnp.where(r < end, sorted_ref[jnp.minimum(r, routes - 1)], -1)
            col = jnp.where(tile_rows == j, tok, col)
        return col

    def compute_sparse(unit):
        slot = unit % _NUM_BUFFERS
        c = unit % num_chunks
        e = active_ids_ref[unit // num_chunks]
        begin, end = offs_ref[e], offs_ref[e + 1]
        tiles = (end - begin + tile - 1) // tile

        def rows_at(t):
            return pl.ds(pl.multiple_of(t * tile, tile), tile)

        @pl.when(c == 0)
        def gather():
            # One-hot products: exact copies of the expert's token rows.
            def body(t, carry):
                col = tile_tokens(begin + t * tile, end)
                lanes = lax.broadcasted_iota(jnp.int32, (tile, mp), 1)
                onehot = (col == lanes).astype(jnp.bfloat16)
                xs_ref[rows_at(t), :] = jnp.dot(
                    onehot, x_pad_ref[...], preferred_element_type=f32
                ).astype(jnp.bfloat16)
                return carry

            lax.fori_loop(0, tiles, body, 0)

        def run(t, carry):
            down = expert_mlp(xs_ref[rows_at(t), :], slot)

            @pl.when(c == 0)
            def first():
                expert_acc_ref[rows_at(t), :] = down

            @pl.when(c > 0)
            def rest():
                expert_acc_ref[rows_at(t), :] += down

            return carry

        lax.fori_loop(0, tiles, run, 0)

        @pl.when(c == num_chunks - 1)
        def finish():
            columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)
            coeff = jnp.sum(
                jnp.where(columns == column(e), coeff_ref[...], 0),
                axis=1,
                keepdims=True,
            )
            coeff_rows_ref[...] = jnp.broadcast_to(coeff, coeff_rows_ref.shape)

            # Each routed row adds its bf16-rounded output, times its token's
            # coefficient, to that token's row alone: the dense kernel's
            # arithmetic, and no product spans other tokens.
            def body(t, carry):
                rows = jnp.minimum(end - begin - t * tile, tile)
                for j in range(tile):

                    @pl.when(j < rows)
                    def add(*, j=j):
                        tok = pl.ds(sorted_ref[begin + t * tile + j], 1)
                        w = jnp.broadcast_to(coeff_rows_ref[tok, 0:1], (1, h))
                        out_e = expert_acc_ref[pl.ds(t * tile + j, 1), :]
                        out_e = out_e.astype(out_ref.dtype).astype(f32)
                        acc_ref[tok, :] += jnp.where(w != 0, out_e * w, 0)

                return carry

            lax.fori_loop(0, tiles, body, 0)

    def compute(unit):
        if sparse:
            compute_sparse(unit)
            return
        slot = unit % _NUM_BUFFERS
        c = unit % num_chunks
        down = expert_mlp(x_pad_ref[...], slot)

        @pl.when(c == 0)
        def first():
            expert_acc_ref[...] = down

        @pl.when(c > 0)
        def rest():
            expert_acc_ref[...] += down

        @pl.when(c == num_chunks - 1)
        def finish():
            e = active_ids_ref[unit // num_chunks]
            columns = lax.broadcasted_iota(jnp.int32, coeff_ref.shape, 1)
            coeff = jnp.sum(
                jnp.where(columns == column(e), coeff_ref[...], 0),
                axis=1,
                keepdims=True,
            )
            out_e = expert_acc_ref[...].astype(out_ref.dtype).astype(f32)
            # Zero routing must suppress even a non-finite unused output.
            acc_ref[...] += jnp.where(coeff != 0, out_e * coeff, 0)

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

    out = acc_ref[:m, :]
    if i_s:
        routed = out.astype(out_ref.dtype).astype(f32)
        out = routed + out_ref[...].astype(f32)
    out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jax.jit, static_argnames=("activation", "sparse_rows", "replica_groups")
)
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
    shared: SharedExpert | None = None,
    sparse_rows: bool = False,
    replicas=None,
    block_index=None,
    replica_groups: int = 0,
):
    """Local routed contribution of fp4 experts, [tokens, hidden] bf16, as
    fused_moe_func's GMM path returns it, plus the shared expert's output
    when `shared` is given. IDs outside this rank must be -1. With
    `replica_groups` blocks of experts whose local experts end in replica
    slots, `local_ids` are slots instead (-1 for none), `replicas` is the
    layer's replica table (expert_placement.ExpertPlacement.replica_table,
    int32) and `block_index` this rank's block: the kernel picks which copy of
    each replicated expert runs. Callers check can_use_dense_expert_fp4 first.
    Non-finite hidden states and weights are zeroed for the routed experts, as
    the GMM path's one-hot permute (which serves these token counts) does."""
    m, h = hidden.shape
    experts, i, _ = w2.shape
    i_s = 0 if shared is None else shared.w2.shape[0]
    info = pltpu.get_tpu_info()
    chunk = _chunk(m, h, i, i_s, info.vmem_capacity_bytes, sparse_rows)
    # Whole f32 sublane tiles of tokens; the kernel pads to _ROW_ALIGN itself.
    pad = -m % 8
    if pad:
        hidden = jnp.pad(hidden, ((0, pad), (0, 0)))
        local_ids = jnp.pad(local_ids, ((0, pad), (0, 0)), constant_values=-1)
        topk_weights = jnp.pad(topk_weights, ((0, pad), (0, 0)))
    mk = m + pad
    mp = mk + (-mk % _ROW_ALIGN)
    local_ids = local_ids.astype(jnp.int32)
    lanes = info.num_lanes
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    shared_operands, shared_specs, shared_scratch = (), [], []
    if shared is not None:
        shared_operands = tuple(shared)
        shared_specs = [hbm, vmem, hbm, vmem]
        shared_scratch = [
            pltpu.VMEM((h, i_s), shared.w1.dtype),
            pltpu.VMEM((h, i_s), shared.w1.dtype),
            pltpu.VMEM((i_s, h), shared.w2.dtype),
            pltpu.SemaphoreType.DMA((3,)),
        ]
    columns = experts
    prefetch, replica_scratch = (local_ids.reshape(-1),), []
    if replica_groups:
        replica_count = replicas.shape[0] // 4
        # The kernel derives the block size from a table of exactly
        # replica_groups * R rows (unused rows name slot -1).
        assert replica_count % replica_groups == 0, replicas.shape
        columns = replica_groups * (experts - replica_count // replica_groups)
        prefetch += (
            replicas.astype(jnp.int32),
            block_index.reshape(1).astype(jnp.int32),
        )
        replica_scratch = [
            pltpu.VMEM((8, columns), jnp.int32),
            pltpu.SMEM((8, columns), jnp.int32),
            pltpu.SemaphoreType.DMA(()),
            pltpu.SMEM((local_ids.size,), jnp.int32),
        ]
    sparse_scratch = []
    if sparse_rows:
        sparse_scratch = [
            pltpu.SMEM((experts,), jnp.int32),
            pltpu.SMEM((experts + 1,), jnp.int32),
            pltpu.SMEM((local_ids.size,), jnp.int32),
            pltpu.VMEM((mp, h), jnp.bfloat16),
            pltpu.VMEM((mp, lanes), jnp.float32),
        ]
    out = pl.pallas_call(
        functools.partial(
            _kernel,
            h=h,
            i=i,
            chunk=chunk,
            experts=experts,
            activation=activation,
            i_s=i_s,
            sparse=sparse_rows,
            groups=replica_groups,
        ),
        out_shape=jax.ShapeDtypeStruct((mk, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            in_specs=[vmem, vmem, vmem, *[hbm] * 4, *shared_specs],
            out_specs=vmem,
            scratch_shapes=[
                pltpu.VMEM((_NUM_BUFFERS, h, chunk), w1.dtype),
                pltpu.VMEM((_NUM_BUFFERS, h, chunk), w1.dtype),
                pltpu.VMEM((_NUM_BUFFERS, chunk, h), w2.dtype),
                pltpu.VMEM((_NUM_BUFFERS, h // _QUANT_BLOCK, 1, chunk), jnp.float32),
                pltpu.VMEM((_NUM_BUFFERS, h // _QUANT_BLOCK, 1, chunk), jnp.float32),
                pltpu.VMEM((_NUM_BUFFERS, chunk // _QUANT_BLOCK, 1, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.bfloat16),
                pltpu.VMEM((mp, columns + (-columns) % lanes), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((_NUM_BUFFERS, 6)),
                *shared_scratch,
                *sparse_scratch,
                *replica_scratch,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=_vmem_limit_bytes(
                mk, h, chunk, i_s, info.vmem_capacity_bytes, sparse_rows
            )
        ),
        name=f"dense_expert_moe_fp4-e_{experts}-m_{mp}-h_{h}-i_{i}-c_{chunk}"
        + (f"-s_{i_s}" if i_s else "")
        + ("-sparse" if sparse_rows else ""),
    )(
        *prefetch,
        hidden,
        local_ids,
        topk_weights.astype(jnp.float32),
        w1,
        w2,
        # 4-D as the GMM path holds them: a reshape that drops the size-1
        # axis is a relayout copy of every scale on each call.
        s1,
        s2,
        *shared_operands,
    )
    return out[:m] if pad else out
