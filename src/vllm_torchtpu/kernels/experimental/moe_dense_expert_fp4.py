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
# limits up to ~53.7 MiB. The sparse-row mode stays within the same bound at
# every size it serves (EP4xTP2 call, i 1536: 54.6 / 56.8 / 46.1 / 50.7 MiB at
# 48 / 64 / 96 / 128 rows, chunk 768 / 768 / 512 / 512).
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
    when a unit evaluates every token, measured at 16 and 64 rows; with
    ``sparse``, ~8 for the gather/scatter products plus 16 per element of one
    16-row tile, fitted at 64-256 rows)."""
    mp = m + (-m % _ROW_ALIGN)
    scratch = 2 + 4 + 4 + 2 + (2 if i_s else 0) + (2 if sparse else 0)
    temporaries = (8 * mp + 16 * _ROW_ALIGN) if sparse else 16 * mp
    return _weight_vmem_bytes(h, chunk, i_s) + (mp * scratch + temporaries) * h


def _kernel(
    ids_smem,  # SMEM [m * topk] local expert ids, -1 off this rank (prefetch)
    x_ref,  # VMEM [m, h] bf16
    ids_ref,  # VMEM [m, topk] int32: ids_smem as rows
    tw_ref,  # VMEM [m, topk] f32 routing weights
    w1_hbm,  # [E, h, 2i] fp4
    w2_hbm,  # [E, i, h] fp4
    s1_hbm,  # [E, h / 128, 1, 2i] f32
    s2_hbm,  # [E, i / 128, 1, h] f32
    *refs,
    h,
    i,
    chunk,
    experts,
    activation,
    i_s,
    sparse,
):
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
        coeff_ref,  # VMEM [mp, E_pad] f32: each token's weight per local expert
        expert_acc_ref,  # VMEM [mp, h] f32: the current expert's down projection
        acc_ref,  # VMEM [mp, h] f32: the routing-weighted sum
        active_ref,  # SMEM [E]: nonzero for an expert some token routes to
        active_ids_ref,  # SMEM [E]
        sem_ref,
        *extra_scratch,
    ) = refs
    shared_scratch = extra_scratch[:5] if i_s else ()
    if sparse:
        # SMEM [E] routes per local expert, [E + 1] their offsets, [m * k] the
        # routes' tokens grouped by expert; VMEM [mp, h] bf16 the gathered rows.
        counts_ref, offs_ref, sorted_ref, xs_ref = extra_scratch[len(shared_scratch) :]
    fp8 = jnp.float8_e4m3fn
    f32 = jnp.float32
    m = x_ref.shape[0]
    mp = acc_ref.shape[0]
    num_chunks = i // chunk

    if i_s:
        swg_ref, swu_ref, swd_ref, shared_out_ref, shared_sem = shared_scratch
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

    def first_route(r, e):
        """No earlier route of r's token names e (each token adds an expert's
        output once, weighted by its summed coefficient)."""
        first = True
        for back in range(1, topk):
            earlier = jnp.maximum(r - back, 0)
            first &= (back > r % topk) | (ids_smem[earlier] != e)
        return first

    def mark(r, carry):
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
            e = ids_smem[r]

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
        x_pad_ref[...] = x.astype(jnp.bfloat16)
        for copy in shared_copies:
            copy.wait()
        xs = x_pad_ref[...]

        def matmul(lhs, w_ref):
            # K blocks keep the MXU operands' temporaries small.
            y = jnp.zeros((mp, w_ref.shape[1]), f32)
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
        shared_out_ref[...] = projection(act, swd_ref, ss2_ref[...])
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
    tile_lanes = lax.broadcasted_iota(jnp.int32, (1, tile), 1)

    def tile_tokens(begin, end):
        """The tokens of routes [begin, begin + tile) (-1 at or past ``end``),
        as a [tile, 1] column and a [1, tile] row."""
        col = jnp.full((tile, 1), -1, jnp.int32)
        row = jnp.full((1, tile), -1, jnp.int32)
        for j in range(tile):
            r = begin + j
            tok = jnp.where(r < end, sorted_ref[jnp.minimum(r, routes - 1)], -1)
            col = jnp.where(tile_rows == j, tok, col)
            row = jnp.where(tile_lanes == j, tok, row)
        return col, row

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
                col, _ = tile_tokens(begin + t * tile, end)
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
                jnp.where(columns == e, coeff_ref[...], 0), axis=1, keepdims=True
            )
            token_rows = lax.broadcasted_iota(jnp.int32, (mp, tile), 0)

            def body(t, carry):
                _, row = tile_tokens(begin + t * tile, end)
                weights = jnp.where(token_rows == row, coeff, 0)  # [mp, tile]
                # The product spans every token: keep a non-finite bf16 expert
                # output from turning other tokens' rows into NaN (0 * inf).
                # (isfinite takes f32 in Mosaic: test the bf16 round trip.)
                out_e = expert_acc_ref[rows_at(t), :]
                out_e = out_e.astype(jnp.bfloat16).astype(f32)
                out_e = jnp.where(jnp.isfinite(out_e), out_e, 0).astype(jnp.bfloat16)
                # One route per (token, row), so each output element is one
                # product: two bf16 passes carry a bf16 routing weight exactly
                # (an f32 one to 16 bits).
                hi = weights.astype(jnp.bfloat16)
                lo = (weights - hi.astype(f32)).astype(jnp.bfloat16)
                acc_ref[...] += jnp.dot(
                    hi, out_e, preferred_element_type=f32
                ) + jnp.dot(lo, out_e, preferred_element_type=f32)
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
                jnp.where(columns == e, coeff_ref[...], 0), axis=1, keepdims=True
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
        out = routed + shared_out_ref[:m, :].astype(f32)
    out_ref[...] = out.astype(out_ref.dtype)


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
    shared: SharedExpert | None = None,
    sparse_rows: bool = False,
):
    """Local routed contribution of fp4 experts, [tokens, hidden] bf16, as
    fused_moe_func's GMM path returns it, plus the shared expert's output
    when `shared` is given. IDs outside this rank must
    be -1. Callers check can_use_dense_expert_fp4 first. Non-finite hidden
    states and weights are zeroed for the routed experts, as the GMM path's
    one-hot permute (which serves these token counts) does."""
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
            pltpu.VMEM((mp, h), jnp.bfloat16),
            pltpu.SemaphoreType.DMA((3,)),
        ]
    sparse_scratch = []
    if sparse_rows:
        sparse_scratch = [
            pltpu.SMEM((experts,), jnp.int32),
            pltpu.SMEM((experts + 1,), jnp.int32),
            pltpu.SMEM((local_ids.size,), jnp.int32),
            pltpu.VMEM((mp, h), jnp.bfloat16),
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
        ),
        out_shape=jax.ShapeDtypeStruct((mk, h), jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
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
                pltpu.VMEM((mp, experts + (-experts) % lanes), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.VMEM((mp, h), jnp.float32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SMEM((experts,), jnp.int32),
                pltpu.SemaphoreType.DMA((_NUM_BUFFERS, 6)),
                *shared_scratch,
                *sparse_scratch,
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
        local_ids.reshape(-1),
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
