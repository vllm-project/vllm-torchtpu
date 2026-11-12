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
"""A decode-step MoE for a few tokens: one Pallas call per layer.

At a handful of tokens the expert-parallel MoE is bound by reading the hit
experts' weights and by the chain of small operations around the grouped
matmuls: sorts, a permute, a pad to the matmul tile, an unpermute. This
kernel drops the chain. Every token's routing is a dense table of weights over
the shard's experts. The kernel walks the hit experts only, streams their
weights through a ring of VMEM slots in equal-sized units (the gate|up halves
and the down projection of one expert are three units), and applies each
expert to every token with that token's weight as the mask (zero for tokens
the expert does not serve). Routing needs no sort or permutation. The output
is the shard's partial sum of expert outputs per token; the combine across
shards stays with the caller.

Weights are fp8 with one scale per output channel. Activations are quantized
per token to fp8 inside the kernel before each matmul, the whole row at once,
as the grouped matmul does (W8A8). Rows are padded to the fp8 tile (32) by the
wrapper; the kernel computes the padded rows and the wrapper drops them. The
hit list is built inside the kernel from a 0/1 mask, so the wrapper adds no
sort.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

ROW_TILE = 32  # fp8 sublane packing: four rows per 32-bit word, eight words
LANES = 512  # the routing table's expert axis
LANE = 128  # the lane tile; column offsets inside a unit must sit on it
# The most local experts a shard may hold on this path, and the gate|up width
# it needs: two lane-aligned halves per expert.
MAX_LOCAL_EXPERTS = 128
GATE_UP_ALIGNMENT = 4 * LANE
VMEM_FRACTION = 0.75  # of the core's vector memory, read at trace time
VMEM_MARGIN = 1.25  # over the declared scratch, for the compiler's own tiles
COMPILER_VMEM_FLOOR = 512 * 1024  # minimum compiler tile slack on small shapes
FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0
NEG = -3.0e38  # scores at or below it never win a selection pass
HOT_SLOTS = 128  # the copied-expert tables' fixed length (-1 padded)


def vmem_limit(fraction=VMEM_FRACTION):
    """The kernel's scratch ceiling: `fraction` of the core's vector memory."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * fraction)


def _try_tpu_info():
    """Return active `TpuInfo` when a TPU target is available, else `None`."""
    try:
        return pltpu.get_tpu_info()
    except (ValueError, RuntimeError):
        return None


def _try_vmem_limit(fraction=VMEM_FRACTION) -> int | None:
    """Return `vmem_limit(fraction)` when a TPU target is active, else `None`."""
    info = _try_tpu_info()
    return None if info is None else int(info.vmem_capacity_bytes * fraction)


def vmem_required_bytes(
    rows,
    hidden,
    inter,
    w13_parts,
    two_inter,
    *,
    whole_expert=False,
    scores=False,
    nc=HOT_SLOTS,
):
    """Required VMEM bytes for scratch, operands, and compiler tile slack."""
    if whole_expert:
        weights = 2 * (hidden * two_inter + inter * hidden)
        # [2, 1, cols] f32 refs pad sublane 1 to 8 in TPU VMEM.
        scales = 2 * 8 * 4 * (two_inter + hidden)
        accumulators = rows * 4 * hidden
    else:
        unit = w13_parts + 1
        weights = unit * (hidden * 2 * (inter // w13_parts) + inter * hidden)
        scales = unit * 8 * 4 * (2 * (inter // w13_parts) + hidden)
        accumulators = rows * 4 * (two_inter + hidden)
    # x in and out, quantized x, the routing table, and the hit and slot tables.
    operands = rows * 2 * hidden * 2 + rows * hidden + rows * 4 * LANES + 4 * LANES * 8
    if scores:
        operands += rows * 4 * LANES + LANES * 2 * nc + 8 * 4 * nc + 2 * 8 * 4 * LANES
    return (
        int((weights + scales + accumulators + operands) * VMEM_MARGIN)
        + COMPILER_VMEM_FLOOR
    )


def vmem_budget(
    rows,
    hidden,
    inter,
    w13_parts,
    two_inter,
    *,
    whole_expert=False,
    scores=False,
    nc=HOT_SLOTS,
):
    """Reserve the scratch and VMEM operands plus a compiler margin.

    A shape-dependent reservation leaves room for surrounding operations to
    prefetch their weights. The device-derived ceiling bounds the reservation.
    """
    need = vmem_required_bytes(
        rows,
        hidden,
        inter,
        w13_parts,
        two_inter,
        whole_expert=whole_expert,
        scores=scores,
        nc=nc,
    )
    return min(need, vmem_limit())


def _quantize_rows(v):
    """Per-row fp8 quantization of a [rows, cols] f32 block: the value in fp8
    and the per-row scale that restores it."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / FP8_MAX
    return (v / scale).astype(FP8), scale


def _stream(
    gid_of,
    hits_s,
    wtab_ref,
    x_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    h_acc,
    acc,
    *,
    n_experts,
    inter,
    w13_parts,
    copy_hbm=None,
    choice=None,
):
    """Walk the hit experts and accumulate their weighted outputs.

    gid_of(e): the routing-table lane of local slot e (a scalar read);
    hits_s int32[1, LANES] (SMEM): 1 in the lanes served by this shard;
    wtab_ref [rows, LANES] f32: the routing weight of lane g for every row;
    x_ref [rows, hidden] bf16; s13_hbm [E, 1, 2*inter] and s2_hbm [E, 1,
    hidden] f32 per-channel scales in HBM (each expert's scales ride with its
    weight DMAs); w13_hbm [E, hidden, 2*inter] and w2_hbm [E, inter, hidden]
    fp8 in HBM; out_ref [rows, hidden].

    Scratch: ids int32[E] (SMEM) the hit list; w13_buf [slots, hidden,
    2*half] one gate|up unit per slot (its gate columns, then its up columns);
    w2_buf [slots, inter, hidden]; s13_buf [slots, 1, 2*half] and s2_buf
    [slots, 1, hidden] the unit's scale rows; the DMA semaphores; h_acc
    [rows, 2*inter] f32 the current expert's gate|up rows; acc the output
    accumulator.

    copy_hbm: optional (s13, s2, w13, w2) of the copy slots, which follow the
    primaries in slot order: slot e >= len(w13_hbm) reads copy slot
    e - len(w13_hbm). choice: an optional hook called with the count of hits
    collected so far (whose first weight units are already in flight); it may
    append local slots to `ids` from that index on and returns the new count.
    """
    rows = x_ref.shape[0]
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)

    # The hit list of this shard's experts, in slot order: a scalar loop.
    def collect(e, n):
        h = hits_s[0, gid_of(e)]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    n_primary = w13_hbm.shape[0]

    def source_copies(refs, expert, slot, col):
        """The DMA descriptors of one unit of `expert` from refs (s13, s2,
        w13, w2) into ring slot `slot`: the gate|up half at column `col`, and
        the down projection."""
        s13_src, s2_src, w13_src, w2_src = refs
        w13_copies = (
            pltpu.make_async_copy(
                w13_src.at[expert, :, pl.ds(col, half)],
                w13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                w13_src.at[expert, :, pl.ds(inter + col, half)],
                w13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_src.at[expert, :, pl.ds(col, half)],
                s13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_src.at[expert, :, pl.ds(inter + col, half)],
                s13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
        )
        w2_copies = (
            pltpu.make_async_copy(w2_src.at[expert], w2_buf.at[slot], w2_sem.at[slot]),
            pltpu.make_async_copy(s2_src.at[expert], s2_buf.at[slot], w2_sem.at[slot]),
        )
        return w13_copies, w2_copies

    def unit_copies(e_idx, part):
        """The DMA descriptors of a unit into its slot (slot = part, since the
        ring has as many slots as an expert has units), per weight source, as
        (condition, gate|up descriptors, down descriptors): parts below
        w13_parts are gate|up halves (gate columns [p*half, (p+1)*half) and up
        columns [inter + p*half, ...) of w13); part w13_parts is the down
        projection.
        """
        expert = ids[e_idx]
        slot = part
        # The down unit builds gate|up descriptors it never starts; clamp
        # their column into range.
        col = pl.multiple_of(jnp.minimum(part, w13_parts - 1) * half, half)
        primary = (s13_hbm, s2_hbm, w13_hbm, w2_hbm)
        if copy_hbm is None:
            return part, ((True, *source_copies(primary, expert, slot, col)),)
        in_primary = expert < n_primary
        return part, (
            (
                in_primary,
                *source_copies(primary, jnp.minimum(expert, n_primary - 1), slot, col),
            ),
            (
                jnp.logical_not(in_primary),
                *source_copies(copy_hbm, jnp.maximum(expert - n_primary, 0), slot, col),
            ),
        )

    def advance(e_idx, part, by):
        """The (expert index, part) of the unit `by` units ahead, without a
        division: `by` is at most `units`."""
        p = part + by
        wrap = p >= units
        return jnp.where(wrap, e_idx + 1, e_idx), jnp.where(wrap, p - units, p)

    def start(e_idx, part):
        part, sources = unit_copies(e_idx, part)
        for in_source, w13_copies, w2_copies in sources:

            @pl.when(in_source & (part < w13_parts))
            def _(*, w13_copies=w13_copies):
                for c in w13_copies:
                    c.start()

            @pl.when(in_source & (part == w13_parts))
            def _(*, w2_copies=w2_copies):
                for c in w2_copies:
                    c.start()

    def wait(e_idx, part):
        part, sources = unit_copies(e_idx, part)
        for in_source, w13_copies, w2_copies in sources:

            @pl.when(in_source & (part < w13_parts))
            def _(*, w13_copies=w13_copies):
                for c in w13_copies:
                    c.wait()

            @pl.when(in_source & (part == w13_parts))
            def _(*, w2_copies=w2_copies):
                for c in w2_copies:
                    c.wait()

    acc[...] = jnp.zeros_like(acc)
    x_q, x_s = _quantize_rows(x_ref[...].astype(jnp.float32))

    for ahead in range(slots - 1):

        @pl.when(ahead < n_units)
        def _prime(*, ahead=ahead):
            start(jnp.int32(0), jnp.int32(ahead))

    if choice is not None:
        n_before = n_hit
        n_hit = choice(n_hit)
        n_units = n_hit * units
        # No fixed hit on this shard: nothing was primed above; prime now
        # from the appended hits.
        for ahead in range(slots - 1):

            @pl.when((n_before == 0) & (ahead < n_units))
            def _prime_late(*, ahead=ahead):
                start(jnp.int32(0), jnp.int32(ahead))

    def body(u, carry):
        e_idx, part = carry
        slot = part
        expert = ids[e_idx]

        @pl.when(u + slots - 1 < n_units)
        def _prefetch():
            start(*advance(e_idx, part, slots - 1))

        wait(e_idx, part)

        # This half's gate|up columns for every row, scaled per token and per
        # channel, into h_acc's columns; one static branch per half, since
        # the lane offsets must be static.
        for p in range(w13_parts):

            @pl.when(part == p)
            def _gate_up_half(p=p):
                h = jax.lax.dot_general(
                    x_q,
                    w13_buf[slot],
                    (((1,), (0,)), ((), ())),
                    preferred_element_type=jnp.float32,
                )
                # [1, 2*half]: this half's gate columns, then its up columns.
                scales = s13_buf[slot]
                col = p * half
                h_acc[:, col : col + half] = h[:, :half] * x_s * scales[:, :half]
                h_acc[:, inter + col : inter + col + half] = (
                    h[:, half:] * x_s * scales[:, half:]
                )

        @pl.when(part == w13_parts)
        def _down():
            gate = h_acc[:, pl.ds(0, inter)]
            up = h_acc[:, pl.ds(inter, inter)]
            a = gate * jax.nn.sigmoid(gate) * up
            a_q, a_s = _quantize_rows(a)
            o = jax.lax.dot_general(
                a_q,
                w2_buf[slot],
                (((1,), (0,)), ((), ())),
                preferred_element_type=jnp.float32,
            )
            o = o * a_s * s2_buf[slot]
            # The routing weight is the mask: zero for rows the expert does
            # not serve.
            w = jnp.sum(
                jnp.where(lane == gid_of(expert), wtab_ref[...], 0.0),
                axis=1,
                keepdims=True,
            )
            acc[...] += jnp.where(w != 0, w * o, 0.0)

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def _stream_whole_expert(
    gid_of,
    hits_s,
    wtab_ref,
    x_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    ids,
    w13_buf,
    w2_buf,
    s13_buf,
    s2_buf,
    w13_sem,
    w2_sem,
    acc,
    *,
    n_experts,
    inter,
    copy_hbm=None,
    choice=None,
):
    """Walk the hit experts with two-slot whole-expert ping-pong buffering.

    Unlike `_stream`, each ring slot holds a full `w13` `[hidden, 2*inter]`
    and `w2` `[inter, hidden]` expert block (`slots = 2`, `slot = e_idx % 2`).
    Each hit expert issues contiguous 1-D DMAs for `w13`, `s13`, `w2`, and
    `s2` and executes both projections in a single loop iteration without an
    intermediate `h_acc` VMEM buffer.
    """
    rows = x_ref.shape[0]
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)

    def collect(e, n):
        h = hits_s[0, gid_of(e)]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_primary = w13_hbm.shape[0]

    def w13_copies(refs, expert, slot):
        s13_src, _, w13_src, _ = refs
        return (
            pltpu.make_async_copy(
                w13_src.at[expert], w13_buf.at[slot], w13_sem.at[slot]
            ),
            pltpu.make_async_copy(
                s13_src.at[expert], s13_buf.at[slot], w13_sem.at[slot]
            ),
        )

    def w2_copies(refs, expert, slot):
        _, s2_src, _, w2_src = refs
        return (
            pltpu.make_async_copy(w2_src.at[expert], w2_buf.at[slot], w2_sem.at[slot]),
            pltpu.make_async_copy(s2_src.at[expert], s2_buf.at[slot], w2_sem.at[slot]),
        )

    def for_expert_source(e_idx, fn):
        expert = ids[e_idx]
        primary = (s13_hbm, s2_hbm, w13_hbm, w2_hbm)
        if copy_hbm is None:
            fn(primary, expert)
            return
        in_primary = expert < n_primary

        @pl.when(in_primary)
        def _():
            fn(primary, jnp.minimum(expert, n_primary - 1))

        @pl.when(jnp.logical_not(in_primary))
        def _():
            fn(copy_hbm, jnp.maximum(expert - n_primary, 0))

    def start_expert(e_idx, slot):
        def _start(refs, expert):
            for c in w13_copies(refs, expert, slot):
                c.start()
            for c in w2_copies(refs, expert, slot):
                c.start()

        for_expert_source(e_idx, _start)

    def wait_w13(e_idx, slot):
        def _wait(refs, expert):
            for c in w13_copies(refs, expert, slot):
                c.wait()

        for_expert_source(e_idx, _wait)

    def wait_w2(e_idx, slot):
        def _wait(refs, expert):
            for c in w2_copies(refs, expert, slot):
                c.wait()

        for_expert_source(e_idx, _wait)

    acc[...] = jnp.zeros_like(acc)
    x_q, x_s = _quantize_rows(x_ref[...].astype(jnp.float32))

    @pl.when(n_hit > 0)
    def _prime():
        start_expert(jnp.int32(0), jnp.int32(0))

    if choice is not None:
        n_before = n_hit
        n_hit = choice(n_hit)

        @pl.when((n_before == 0) & (n_hit > 0))
        def _prime_late():
            start_expert(jnp.int32(0), jnp.int32(0))

    def body(e_idx, _):
        slot = jax.lax.rem(e_idx, jnp.int32(2))
        next_slot = jnp.int32(1) - slot
        expert = ids[e_idx]

        @pl.when(e_idx + 1 < n_hit)
        def _prefetch():
            start_expert(e_idx + 1, next_slot)

        wait_w13(e_idx, slot)
        h = jax.lax.dot_general(
            x_q,
            w13_buf[slot],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        h = h * x_s * s13_buf[slot]
        gate = h[:, :inter]
        up = h[:, inter:]
        a = gate * jax.nn.sigmoid(gate) * up
        a_q, a_s = _quantize_rows(a)

        wait_w2(e_idx, slot)
        o = jax.lax.dot_general(
            a_q,
            w2_buf[slot],
            (((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        o = o * a_s * s2_buf[slot]
        w = jnp.sum(
            jnp.where(lane == gid_of(expert), wtab_ref[...], 0.0),
            axis=1,
            keepdims=True,
        )
        acc[...] += jnp.where(w != 0, w * o, 0.0)
        return jnp.int32(0)

    jax.lax.fori_loop(0, n_hit, body, jnp.int32(0))
    out_ref[...] = acc[...].astype(out_ref.dtype)


def _kernel_ids(
    gid_ref,
    hit_ref,
    x_ref,
    wtab_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    out_ref,
    *scratch,
    n_experts,
    inter,
    w13_parts=None,
    whole_expert=False,
):
    """The ids entry: the routing table over local ids and the hit vector come
    from the caller; gid_ref is the identity over local ids."""
    if whole_expert:
        _stream_whole_expert(
            lambda e: gid_ref[0, e],
            hit_ref,
            wtab_ref,
            x_ref,
            s13_hbm,
            s2_hbm,
            w13_hbm,
            w2_hbm,
            out_ref,
            *scratch,
            n_experts=n_experts,
            inter=inter,
        )
        return
    assert w13_parts is not None
    _stream(
        lambda e: gid_ref[0, e],
        hit_ref,
        wtab_ref,
        x_ref,
        s13_hbm,
        s2_hbm,
        w13_hbm,
        w2_hbm,
        out_ref,
        *scratch,
        n_experts=n_experts,
        inter=inter,
        w13_parts=w13_parts,
    )


def can_use_moe_lowtok(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    w1_bias,
    w2_bias,
    *,
    activation,
    whole_expert: bool = False,
    scores: bool = False,
    vmem_limit_bytes: int | None = None,
):
    """Whether the kernel serves this layer: silu, fp8 weights with one scale
    per output channel, bf16 activations, no expert bias, lane-aligned hidden
    and gate|up widths, at most MAX_LOCAL_EXPERTS local experts, and VMEM fit."""
    if activation != "silu" or w1_bias is not None or w2_bias is not None:
        return False
    if w1_scale is None or w2_scale is None:
        return False
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    if hidden_states.dtype != jnp.bfloat16 or w1.dtype != FP8 or w2.dtype != FP8:
        return False
    experts, hidden, two_inter = w1.shape
    inter = two_inter // 2
    if hidden <= 0 or hidden % LANE:
        return False
    if hidden != hidden_states.shape[1] or w2.shape != (
        experts,
        inter,
        hidden,
    ):
        return False
    alignment = 2 * LANE if whole_expert else GATE_UP_ALIGNMENT
    if not 0 < experts <= MAX_LOCAL_EXPERTS or two_inter <= 0 or two_inter % alignment:
        return False
    # One scale per output channel, with any singleton axes in between.
    if w1_scale.size != experts * two_inter or w2_scale.size != experts * hidden:
        return False
    info = _try_tpu_info() if vmem_limit_bytes is None else None
    if info is not None and not info.is_matmul_supported(FP8, FP8):
        return False
    limit = (
        vmem_limit_bytes
        if vmem_limit_bytes is not None
        else (
            int(info.vmem_capacity_bytes * VMEM_FRACTION) if info is not None else None
        )
    )
    if limit is not None:
        rows = -(-max(int(hidden_states.shape[0]), 1) // ROW_TILE) * ROW_TILE
        w13_parts = 2 if (inter // 2) % LANE == 0 else 1
        need = vmem_required_bytes(
            rows,
            hidden,
            inter,
            w13_parts,
            two_inter,
            whole_expert=whole_expert,
            scores=scores,
        )
        if need > limit:
            return False
    return True


def normalize_whole_expert_min_tokens(
    whole_expert: bool,
    whole_expert_min_tokens: int,
    *,
    active: bool = True,
) -> int:
    """Canonicalize ``whole_expert_min_tokens`` for custom-op and compile caches.

    When ``whole_expert`` is False (or ``active`` is False because the caller
    cannot run ``moe_lowtok``), ``whole_expert_min_tokens`` cannot alter any
    traced program and normalizes to ``0`` so cache keys are not split. Logs a
    one-time warning if a positive threshold was requested while
    ``whole_expert`` is False.
    """
    if whole_expert_min_tokens < 0:
        raise ValueError(
            "whole_expert_min_tokens must be non-negative, got "
            f"{whole_expert_min_tokens}"
        )
    if active and whole_expert_min_tokens > 0 and not whole_expert:
        logger.warning_once(
            "TPU_MOE_LOWTOK_WHOLE_EXPERT_MIN_TOKENS=%d has no effect while "
            "TPU_MOE_LOWTOK_WHOLE_EXPERT is False",
            whole_expert_min_tokens,
        )
    if not active or not whole_expert:
        return 0
    return int(whole_expert_min_tokens)


def resolve_whole_expert(
    hidden_states,
    w1,
    w2,
    w1_scale,
    w2_scale,
    w1_bias=None,
    w2_bias=None,
    *,
    activation: str,
    whole_expert: bool,
    whole_expert_min_tokens: int = 0,
    scores: bool = False,
    vmem_limit_bytes: int | None = None,
) -> bool:
    """Whether a traced step should run two-slot whole-expert streaming.

    Returns False when ``whole_expert`` is False or when whole-expert streaming
    itself does not support the layer or shape. Otherwise, when
    ``whole_expert_min_tokens`` is positive and the post-all-gather row count is
    below ``whole_expert_min_tokens``, prefers three-slot half-``w13`` streaming
    (``False``) whenever ``can_use_moe_lowtok(..., whole_expert=False)`` admits
    the layer and shape, falling back to ``True`` on shapes only whole-expert
    streaming supports (such as ``two_inter == 2 * LANE`` or tight VMEM caps).
    """
    if whole_expert_min_tokens < 0:
        raise ValueError(
            "whole_expert_min_tokens must be non-negative, got "
            f"{whole_expert_min_tokens}"
        )
    if not whole_expert:
        return False
    if not can_use_moe_lowtok(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        activation=activation,
        whole_expert=True,
        scores=scores,
        vmem_limit_bytes=vmem_limit_bytes,
    ):
        return False
    rows = int(hidden_states.shape[0])
    if whole_expert_min_tokens == 0 or rows >= whole_expert_min_tokens:
        return True
    return not can_use_moe_lowtok(
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        w1_bias,
        w2_bias,
        activation=activation,
        whole_expert=False,
        scores=scores,
        vmem_limit_bytes=vmem_limit_bytes,
    )


@functools.partial(
    jax.jit, static_argnames=("vmem_limit_bytes", "whole_expert", "interpret")
)
def moe_lowtok(
    x,
    topk_ids_local,
    topk_weights,
    w13,
    s13,
    w2,
    s2,
    *,
    vmem_limit_bytes=None,
    whole_expert=False,
    interpret=False,
):
    """The shard's MoE output for a few tokens.

    x [tokens, hidden] bf16; topk_ids_local [tokens, k] int32 with -1 for
    experts the shard does not hold; topk_weights [tokens, k]; w13 [E, hidden,
    2*inter] fp8 (gate | up) with s13 [E, 2*inter] scales (any singleton axes
    in between); w2 [E, inter, hidden] fp8 with s2 [E, hidden] scales.
    Returns [tokens, hidden] in x's dtype: the sum over the shard's experts of
    weight * expert(x), zero for tokens routed elsewhere. The reduction over
    shards is the caller's. `interpret` runs the kernel on the host, for tests
    without a TPU.
    """
    tokens, hidden = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    if n_experts > LANES:
        raise ValueError(
            f"the routing table holds up to {LANES} experts per shard, not {n_experts}"
        )
    if hidden % LANE:
        raise ValueError(
            f"the hidden width {hidden} must be a multiple of the lane tile"
        )
    if two_inter != 2 * inter or inter <= 0 or inter % LANE:
        raise ValueError(
            f"the gate-up width {two_inter} must equal 2 * inter with inter "
            f"a multiple of {LANE}"
        )
    if w2.shape != (n_experts, inter, hidden):
        raise ValueError(
            f"expected w2 of shape {(n_experts, inter, hidden)}, got {w2.shape}"
        )
    w13_parts = 2 if (inter // 2) % LANE == 0 else 1
    rows = -(-tokens // ROW_TILE) * ROW_TILE
    # The dense routing-weight table over the shard's experts, [rows, E], then
    # padded to the lane width.
    valid = topk_ids_local >= 0
    onehot = jax.nn.one_hot(
        jnp.where(valid, topk_ids_local, 0), n_experts, dtype=jnp.float32
    )
    wtab = jnp.einsum(
        "tke,tk->te", onehot, jnp.where(valid, topk_weights, 0).astype(jnp.float32)
    )
    hit = jnp.pad(
        jnp.any(wtab != 0, axis=0).astype(jnp.int32), (0, LANES - n_experts)
    ).reshape(1, LANES)
    gid = jnp.arange(n_experts, dtype=jnp.int32).reshape(1, n_experts)
    wtab = jnp.pad(wtab, ((0, rows - tokens), (0, LANES - n_experts)))
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    s13 = s13.reshape(n_experts, 1, two_inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, 1, hidden).astype(jnp.float32)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    if vmem_limit_bytes is None and not interpret:
        vmem_limit_bytes = vmem_budget(
            rows,
            hidden,
            inter,
            w13_parts,
            two_inter,
            whole_expert=whole_expert,
            scores=False,
        )
    if whole_expert:
        scratch_shapes = (
            pltpu.SMEM((n_experts,), jnp.int32),
            pltpu.VMEM((2, hidden, two_inter), FP8),
            pltpu.VMEM((2, inter, hidden), FP8),
            pltpu.VMEM((2, 1, two_inter), jnp.float32),
            pltpu.VMEM((2, 1, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((rows, hidden), jnp.float32),
        )
        kernel_fn = functools.partial(
            _kernel_ids,
            n_experts=n_experts,
            inter=inter,
            whole_expert=True,
        )
    else:
        scratch_shapes = (
            pltpu.SMEM((n_experts,), jnp.int32),
            pltpu.VMEM((w13_parts + 1, hidden, 2 * (inter // w13_parts)), FP8),
            pltpu.VMEM((w13_parts + 1, inter, hidden), FP8),
            pltpu.VMEM((w13_parts + 1, 1, 2 * (inter // w13_parts)), jnp.float32),
            pltpu.VMEM((w13_parts + 1, 1, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((w13_parts + 1,)),
            pltpu.SemaphoreType.DMA((w13_parts + 1,)),
            pltpu.VMEM((rows, two_inter), jnp.float32),
            pltpu.VMEM((rows, hidden), jnp.float32),
        )
        kernel_fn = functools.partial(
            _kernel_ids,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            whole_expert=False,
        )
    out = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=scratch_shapes,
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        interpret=pltpu.InterpretParams(detect_races=True) if interpret else False,
        name="moe_lowtok",
    )(gid, hit, x_rows, wtab, s13, s2, w13, w2)
    return out[:tokens]


def _kernel_scores(
    tab_s,
    prim_ref,
    copy_ref,
    hot1h_ref,
    x_hbm,
    scores_ref,
    s13_hbm,
    s2_hbm,
    w13_hbm,
    w2_hbm,
    *refs,
    n_experts,
    inter,
    w13_parts=None,
    topk,
    ranks,
    has_copies,
    offsets,
    whole_expert=False,
):
    """The scores entry: the router's top-k runs here, then the assignment of
    every hit expert to one holder.

    tab_s int32[length] (SMEM): the scalar tables packed by `pack_tables`
    (`offsets` = their (gid, hot, hprim, hcopy, slot) offsets); prim_ref /
    copy_ref [1, LANES] int32: the rank holding each expert's primary / copy
    (-1: none); hot1h_ref [LANES, NC] bf16: one-hot, 1 at (hot[i], i); x_hbm
    [rows, hidden]: the rows (copied from HBM under selection); scores_ref
    [rows, LANES] f32: the router's softmax. With copies, the copy slots'
    (s13, s2, w13, w2) follow in refs.

    Single-holder experts are fixed, and their first weight units are fetched
    first. The hit copied experts are compacted into a short list on the
    vector units (a one-hot gather on the MXU, a lane prefix sum, a one-hot
    scatter) and assigned one after another in hot order, each to the holder
    with fewer experts so far (fixed plus copies already placed), ties to the
    primary. Every shard computes the same assignment.
    """
    if has_copies:
        copy_hbm, refs = refs[:4], refs[4:]
    else:
        copy_hbm = None
    (
        out_ref,
        ids,
        wtab_ref,
        hits_v,
        hits_s,
        cmp_v,
        cmp_s,
        hsem,
        x_vmem,
        xsem,
        *stream_scratch,
    ) = refs
    # Copy the rows into VMEM while selection and assignment run.
    x_copy = pltpu.make_async_copy(x_hbm, x_vmem, xsem.at[0])
    x_copy.start()
    rows = x_vmem.shape[0]
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)
    o_gid, o_hot, o_hprim, o_hcopy, o_slot = offsets
    my_rank = tab_s[0]

    # The router's selection: topk passes of max-and-mask over the softmax
    # scores (ties to the lowest id), the weights renormalized and rounded
    # through bf16, written as the dense routing table [rows, LANES].
    x = scores_ref[...]
    x = jnp.where(x > NEG, x, NEG)
    masked = jnp.finfo(jnp.float32).min
    sel = jnp.zeros_like(x)
    wsum = jnp.zeros((rows, 1), jnp.float32)
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, lane, LANES - 1), axis=1, keepdims=True)
        pick = lane == idx
        sel = jnp.where(pick, m, sel)
        wsum = wsum + m
        x = jnp.where(pick, masked, x)
    wt = jnp.where(wsum > 0, sel / wsum, 0.0).astype(jnp.bfloat16).astype(jnp.float32)
    wtab_ref[...] = wt
    # The experts that serve a row this call, as a lane vector [1, LANES].
    hit = jnp.any(wt != 0, axis=0, keepdims=True)
    prim = prim_ref[...]
    fixed = hit & jnp.logical_not(copy_ref[...] >= 0)
    lane1 = lane[:1]
    # This shard's fixed experts to SMEM first: the stream primes its first
    # fetch from them while the assignment below runs.
    hits_v[...] = jnp.where(fixed & (prim == my_rank), 1, 0).astype(jnp.int32)
    cp1 = pltpu.make_async_copy(hits_v, hits_s, hsem.at[1])
    cp1.start()

    def stream(choice):
        x_copy.wait()
        if whole_expert:
            _stream_whole_expert(
                lambda e: tab_s[o_gid + e],
                hits_s,
                wtab_ref,
                x_vmem,
                s13_hbm,
                s2_hbm,
                w13_hbm,
                w2_hbm,
                out_ref,
                ids,
                *stream_scratch,
                n_experts=n_experts,
                inter=inter,
                copy_hbm=copy_hbm,
                choice=choice,
            )
            return
        assert w13_parts is not None
        _stream(
            lambda e: tab_s[o_gid + e],
            hits_s,
            wtab_ref,
            x_vmem,
            s13_hbm,
            s2_hbm,
            w13_hbm,
            w2_hbm,
            out_ref,
            ids,
            *stream_scratch,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            copy_hbm=copy_hbm,
            choice=choice,
        )

    if not has_copies:
        # Every hit expert has one holder: no assignment.
        cp1.wait()
        stream(None)
        return
    # The fixed load of every rank, in lane r.
    loads = jnp.zeros_like(lane1)
    for r in range(ranks):
        load_r = jnp.sum(jnp.where(fixed & (prim == r), 1, 0), axis=1, keepdims=True)
        loads = jnp.where(lane1 == r, load_r, loads)
    # The hit copied experts, compacted: hh[i] = hit[hot[i]] through the
    # one-hot on the MXU; an inclusive lane prefix sum gives each hit entry
    # its position; a one-hot scatter puts entry i at column pos[i].
    nc = hot1h_ref.shape[1]
    hit8 = jnp.broadcast_to(jnp.where(hit, 1.0, 0.0).astype(jnp.bfloat16), (8, LANES))
    hh = jnp.dot(hit8, hot1h_ref[...], preferred_element_type=jnp.float32)[:1]
    lane_c = jax.lax.broadcasted_iota(jnp.int32, (1, nc), 1)
    ps = hh
    k = 1
    while k < nc:
        ps = ps + jnp.where(lane_c >= k, pltpu.roll(ps, k, axis=1), 0.0)
        k *= 2
    n_active = ps[:, nc - 1 : nc].astype(jnp.int32)
    pos = jnp.where(hh > 0.5, ps - 1.0, -1.0)
    posm = jnp.transpose(jnp.broadcast_to(pos, (nc, nc)))  # row i: pos[i]
    row_i = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 0)
    lane_p = jax.lax.broadcasted_iota(jnp.int32, (nc, nc), 1).astype(jnp.float32)
    comp = jnp.sum(jnp.where(posm == lane_p, row_i, 0), axis=0, keepdims=True)
    # One buffer to SMEM: row 0 the compact list, row 1 its length, row 2
    # the fixed load per rank.
    cmp_v[...] = jnp.zeros_like(cmp_v)
    cmp_v[0:1, :] = comp
    cmp_v[1:2, :] = jnp.broadcast_to(n_active, (1, nc))
    cmp_v[2:3, :] = loads[:, :nc].astype(jnp.int32)
    cp2 = pltpu.make_async_copy(cmp_v, cmp_s, hsem.at[2])
    cp2.start()
    cp1.wait()

    def choice(n_fixed):
        """The hit copied experts, hottest first, each to the holder with
        fewer experts so far, ties to the primary; the ones this shard serves
        are appended to the hit list (their local slot) from n_fixed on."""
        cp2.wait()

        def assign(j, n):
            i = cmp_s[0, j]
            e = tab_s[o_hot + i]
            o = tab_s[o_hprim + i]
            c = tab_s[o_hcopy + i]
            pick = jnp.where(cmp_s[2, c] < cmp_s[2, o], c, o)
            cmp_s[2, pick] = cmp_s[2, pick] + 1
            mine = pick == my_rank

            @pl.when(mine)
            def _():
                hits_s[0, e] = 1
                ids[n] = tab_s[o_slot + e]

            return n + jnp.where(mine, 1, 0)

        return jax.lax.fori_loop(0, cmp_s[1, 0], assign, n_fixed)

    stream(choice)


@functools.partial(
    jax.jit,
    static_argnames=(
        "topk",
        "ranks",
        "has_copies",
        "vmem_limit_bytes",
        "whole_expert",
        "interpret",
    ),
)
def moe_lowtok_scores(
    x,
    scores,
    tables,
    prim,
    copy,
    hot1h,
    w13,
    s13,
    w2,
    s2,
    copy_weights=None,
    *,
    topk,
    ranks,
    has_copies,
    vmem_limit_bytes=None,
    whole_expert=False,
    interpret=False,
):
    """The shard's MoE output for a few tokens, with the router's top-k and
    the holder assignment inside.

    x [tokens, hidden] bf16; scores [tokens, LANES] f32: the softmax over all
    experts (a row of zeros selects nothing); tables: this shard's scalar
    tables from `pack_tables`; prim / copy [LANES] int32: the rank holding
    each expert's primary / copy (-1: none); hot1h [LANES, NC] from
    `hot_onehot`; w13 / s13 / w2 / s2 the primary slots as in `moe_lowtok`;
    copy_weights: (w13, s13, w2, s2) of the copy slots, read with
    has_copies. A hit expert is served by exactly one rank: its only holder,
    or, with a copy, the holder with fewer experts assigned so far when its
    turn comes (hottest first, ties to the primary). Returns [tokens, hidden]
    in x's dtype; the reduction over ranks is the caller's.
    """
    tokens, hidden = x.shape
    n_primary, _, two_inter = w13.shape
    inter = two_inter // 2
    if scores.shape[1] != LANES or prim.size != LANES or copy.size != LANES:
        raise ValueError(f"the routing table is {LANES} experts wide")
    if hidden % LANE:
        raise ValueError(
            f"the hidden width {hidden} must be a multiple of the lane tile"
        )
    if two_inter != 2 * inter or inter <= 0 or inter % LANE:
        raise ValueError(
            f"the gate-up width {two_inter} must equal 2 * inter with inter "
            f"a multiple of {LANE}"
        )
    if w2.shape != (n_primary, inter, hidden):
        raise ValueError(
            f"expected w2 of shape {(n_primary, inter, hidden)}, got {w2.shape}"
        )
    if has_copies != (copy_weights is not None):
        raise ValueError("copy weights come exactly with has_copies")
    n_experts = n_primary + (
        copy_weights[0].shape[0] if copy_weights is not None else 0
    )
    nc = hot1h.shape[-1]
    if nc % LANE:
        raise ValueError(f"the hot table's length {nc} must be a multiple of {LANE}")
    *offsets, length = table_offsets(n_experts, nc)
    if tables.size != length:
        raise ValueError(
            f"packed tables for {n_experts} slots and {nc} hot entries are "
            f"[{length}], got {tables.shape}"
        )
    w13_parts = 2 if (inter // 2) % LANE == 0 else 1
    rows = -(-tokens // ROW_TILE) * ROW_TILE
    x_rows = jnp.pad(x, ((0, rows - tokens), (0, 0)))
    sc = jnp.pad(scores.astype(jnp.float32), ((0, rows - tokens), (0, 0)))

    def scales(s13, s2, slots):
        return (
            s13.reshape(slots, 1, two_inter).astype(jnp.float32),
            s2.reshape(slots, 1, hidden).astype(jnp.float32),
        )

    weights = (*scales(s13, s2, n_primary), w13, w2)
    if copy_weights is not None:
        cw13, cs13, cw2, cs2 = copy_weights
        weights += (*scales(cs13, cs2, cw13.shape[0]), cw13, cw2)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    if vmem_limit_bytes is None and not interpret:
        vmem_limit_bytes = vmem_budget(
            rows,
            hidden,
            inter,
            w13_parts,
            two_inter,
            whole_expert=whole_expert,
            scores=True,
            nc=nc,
        )
    half_width = 2 * (inter // w13_parts)
    in_specs = [vmem, vmem, vmem, hbm, vmem] + [hbm] * len(weights)
    common_scratch = (
        pltpu.SMEM((n_experts,), jnp.int32),
        pltpu.VMEM((rows, LANES), jnp.float32),
        pltpu.VMEM((1, LANES), jnp.int32),
        pltpu.SMEM((1, LANES), jnp.int32),
        pltpu.VMEM((8, nc), jnp.int32),
        pltpu.SMEM((8, nc), jnp.int32),
        pltpu.SemaphoreType.DMA((3,)),
        pltpu.VMEM((rows, hidden), x_rows.dtype),
        pltpu.SemaphoreType.DMA((1,)),
    )
    if whole_expert:
        scratch_shapes = common_scratch + (
            pltpu.VMEM((2, hidden, two_inter), FP8),
            pltpu.VMEM((2, inter, hidden), FP8),
            pltpu.VMEM((2, 1, two_inter), jnp.float32),
            pltpu.VMEM((2, 1, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.SemaphoreType.DMA((2,)),
            pltpu.VMEM((rows, hidden), jnp.float32),
        )
        kernel_fn = functools.partial(
            _kernel_scores,
            n_experts=n_experts,
            inter=inter,
            topk=topk,
            ranks=ranks,
            has_copies=has_copies,
            offsets=tuple(offsets),
            whole_expert=True,
        )
    else:
        scratch_shapes = common_scratch + (
            pltpu.VMEM((w13_parts + 1, hidden, half_width), FP8),
            pltpu.VMEM((w13_parts + 1, inter, hidden), FP8),
            pltpu.VMEM((w13_parts + 1, 1, half_width), jnp.float32),
            pltpu.VMEM((w13_parts + 1, 1, hidden), jnp.float32),
            pltpu.SemaphoreType.DMA((w13_parts + 1,)),
            pltpu.SemaphoreType.DMA((w13_parts + 1,)),
            pltpu.VMEM((rows, two_inter), jnp.float32),
            pltpu.VMEM((rows, hidden), jnp.float32),
        )
        kernel_fn = functools.partial(
            _kernel_scores,
            n_experts=n_experts,
            inter=inter,
            w13_parts=w13_parts,
            topk=topk,
            ranks=ranks,
            has_copies=has_copies,
            offsets=tuple(offsets),
            whole_expert=False,
        )
    out = pl.pallas_call(
        kernel_fn,
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            in_specs=in_specs,
            out_specs=vmem,
            scratch_shapes=scratch_shapes,
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        interpret=pltpu.InterpretParams(detect_races=True) if interpret else False,
        name="moe_lowtok_scores",
    )(
        jnp.asarray(tables, jnp.int32).reshape(-1),
        jnp.asarray(prim, jnp.int32).reshape(1, LANES),
        jnp.asarray(copy, jnp.int32).reshape(1, LANES),
        jnp.asarray(hot1h, jnp.bfloat16).reshape(LANES, nc),
        x_rows,
        sc,
        *weights,
    )
    return out[:tokens]


def table_offsets(n_experts, nc):
    """The packed table's layout for `n_experts` local slots and `nc` hot
    entries: (gid, hot, hprim, hcopy, slot) offsets and the padded length (a
    multiple of the lane tile)."""
    o_gid = 8
    o_hot = o_gid + n_experts
    o_hprim = o_hot + nc
    o_hcopy = o_hprim + nc
    o_slot = o_hcopy + nc
    length = -(-(o_slot + LANES) // LANE) * LANE
    return o_gid, o_hot, o_hprim, o_hcopy, o_slot, length


def pack_tables(rank, gid, hot, hprim, hcopy):
    """The scores entry's scalar tables as one flat int32 array (see
    table_offsets): [0] this shard's EP rank, the expert in each local slot,
    the copied experts hottest first with the rank of each one's primary and
    copy (-1 padded), and this shard's local slot of every expert (-1: none).
    """
    gid = np.asarray(gid, np.int32).reshape(-1)
    hot, hprim, hcopy = (
        np.asarray(t, np.int32).reshape(-1) for t in (hot, hprim, hcopy)
    )
    if not hprim.size == hot.size == hcopy.size:
        raise ValueError("hot, hprim and hcopy must be the same length")
    o_gid, o_hot, o_hprim, o_hcopy, o_slot, length = table_offsets(gid.size, hot.size)
    tab = np.full((length,), -1, np.int32)
    tab[0] = int(rank)
    tab[o_gid : o_gid + gid.size] = gid
    tab[o_hot : o_hot + hot.size] = hot
    tab[o_hprim : o_hprim + hot.size] = hprim
    tab[o_hcopy : o_hcopy + hot.size] = hcopy
    tab[o_slot + gid] = np.arange(gid.size, dtype=np.int32)
    return tab


def hot_tables(prim, copy, order, slots=HOT_SLOTS):
    """The copied experts in `order` (hottest first) with the rank of each
    one's primary and copy, -1 padded to `slots` entries."""
    prim = np.asarray(prim, np.int32).reshape(-1)
    copy = np.asarray(copy, np.int32).reshape(-1)
    order = [int(e) for e in order]
    if len(order) > slots:
        raise ValueError(
            f"{len(order)} copied experts exceed the {slots}-entry hot table"
        )
    if sorted(order) != sorted(np.flatnonzero(copy >= 0).tolist()):
        raise ValueError("the hot order must list every copied expert once")
    hot = np.full(slots, -1, np.int32)
    hot[: len(order)] = order
    hprim = np.where(hot >= 0, prim[np.maximum(hot, 0)], -1).astype(np.int32)
    hcopy = np.where(hot >= 0, copy[np.maximum(hot, 0)], -1).astype(np.int32)
    return hot, hprim, hcopy


def hot_onehot(hot):
    """[LANES, len(hot)] with a 1 at (hot[i], i) for the real entries (-1
    padding gives an all-zero column)."""
    hot = np.asarray(hot, np.int32).reshape(1, -1)
    return (np.arange(LANES, dtype=np.int32).reshape(-1, 1) == hot).astype(np.float32)


def router_select_reference(scores, topk):
    """The scores entry's router in plain JAX: max-and-mask passes, ties to
    the lowest id, renormalized, rounded through bf16."""
    x = jnp.where(scores > NEG, scores, NEG)
    n = x.shape[1]
    iota = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    ws, ids = [], []
    for _ in range(topk):
        m = jnp.max(x, axis=1, keepdims=True)
        idx = jnp.min(jnp.where(x == m, iota, n - 1), axis=1)
        ws.append(m[:, 0])
        ids.append(idx)
        x = jnp.where(iota == idx[:, None], jnp.finfo(x.dtype).min, x)
    w = jnp.stack(ws, axis=1)
    i = jnp.stack(ids, axis=1)
    w = w / jnp.sum(w, axis=1, keepdims=True)
    return w.astype(jnp.bfloat16), i.astype(jnp.int32)


def moe_lowtok_reference(x, topk_ids_local, topk_weights, w13, s13, w2, s2):
    """Plain JAX with the same arithmetic (fp8 weights, per-channel scales,
    per-token fp8 activations), for parity tests."""
    tokens, hidden = x.shape
    n_experts, _, two_inter = w13.shape
    inter = two_inter // 2
    x_q, x_s = _quantize_rows(x.astype(jnp.float32))
    s13 = s13.reshape(n_experts, two_inter).astype(jnp.float32)
    s2 = s2.reshape(n_experts, hidden).astype(jnp.float32)
    out = jnp.zeros((tokens, hidden), jnp.float32)
    for e in range(n_experts):
        w = jnp.sum(
            jnp.where(topk_ids_local == e, topk_weights, 0).astype(jnp.float32),
            axis=1,
        )
        h = (
            jnp.dot(x_q, w13[e], preferred_element_type=jnp.float32)
            * x_s
            * s13[e][None, :]
        )
        a = h[:, :inter] * jax.nn.sigmoid(h[:, :inter]) * h[:, inter:]
        a_q, a_s = _quantize_rows(a)
        o = (
            jnp.dot(a_q, w2[e], preferred_element_type=jnp.float32)
            * a_s
            * s2[e][None, :]
        )
        out = out + jnp.where(w[:, None] != 0, w[:, None] * o, 0.0)
    return out.astype(x.dtype)
