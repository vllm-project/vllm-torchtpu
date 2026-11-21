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
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

ROW_TILE = 32  # fp8 sublane packing: four rows per 32-bit word, eight words
LANES = 512  # the routing table's expert axis
LANE = 128  # the lane tile; column offsets inside a unit must sit on it
# The most local experts a shard may hold on this path, and the gate|up width
# it needs: two lane-aligned halves per expert.
MAX_LOCAL_EXPERTS = 128
GATE_UP_ALIGNMENT = 4 * LANE
VMEM_FRACTION = 0.75  # of the core's vector memory, read at trace time
VMEM_MARGIN = 1.25  # over the declared scratch, for the compiler's own tiles
FP8 = jnp.float8_e4m3fn
FP8_MAX = 448.0


def vmem_limit(fraction=VMEM_FRACTION):
    """The kernel's scratch ceiling: `fraction` of the core's vector memory."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * fraction)


def vmem_budget(rows, hidden, inter, w13_parts, two_inter):
    """Reserve the scratch and VMEM operands plus a compiler margin.

    A shape-dependent reservation leaves room for surrounding operations to
    prefetch their weights. The device-derived ceiling bounds the reservation.
    """
    unit = w13_parts + 1
    weights = unit * (hidden * 2 * (inter // w13_parts) + inter * hidden)
    scales = unit * 4 * (2 * (inter // w13_parts) + hidden)
    accumulators = rows * 4 * (two_inter + hidden)
    # x in and out, the routing table, the hit and slot tables.
    operands = rows * 2 * hidden * 2 + rows * 4 * LANES + 4 * LANES * 8
    need = int((weights + scales + accumulators + operands) * VMEM_MARGIN)
    return min(need, vmem_limit())


def _quantize_rows(v):
    """Per-row fp8 quantization of a [rows, cols] f32 block: the value in fp8
    and the per-row scale that restores it."""
    amax = jnp.max(jnp.abs(v), axis=1, keepdims=True)
    scale = jnp.maximum(amax, 1e-12) / FP8_MAX
    return (v / scale).astype(FP8), scale


def _stream(
    gid_s,
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
):
    """Walk the hit experts and accumulate their weighted outputs.

    gid_s int32[1, n_experts] (SMEM): the routing-table lane of each local
    slot; hits_s int32[1, LANES] (SMEM): 1 in the lanes served by this shard;
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
    """
    rows = x_ref.shape[0]
    half = inter // w13_parts
    units = w13_parts + 1
    slots = units
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, LANES), 1)

    # The hit list of this shard's experts, in slot order: a scalar loop.
    def collect(e, n):
        h = hits_s[0, gid_s[0, e]]

        @pl.when(h == 1)
        def _():
            ids[n] = e

        return n + h

    n_hit = jax.lax.fori_loop(0, n_experts, collect, jnp.int32(0))
    n_units = n_hit * units

    def unit_copies(e_idx, part):
        """The DMA descriptors of a unit into its slot (slot = part, since the
        ring has as many slots as an expert has units): parts below w13_parts
        are gate|up halves (gate columns [p*half, (p+1)*half) and up columns
        [inter + p*half, ...) of w13); part w13_parts is the down projection.
        """
        expert = ids[e_idx]
        slot = part
        # The down unit builds gate|up descriptors it never starts; clamp
        # their column into range.
        col = pl.multiple_of(jnp.minimum(part, w13_parts - 1) * half, half)
        w13_copies = (
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(col, half)],
                w13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                w13_hbm.at[expert, :, pl.ds(inter + col, half)],
                w13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(col, half)],
                s13_buf.at[slot, :, pl.ds(0, half)],
                w13_sem.at[slot],
            ),
            pltpu.make_async_copy(
                s13_hbm.at[expert, :, pl.ds(inter + col, half)],
                s13_buf.at[slot, :, pl.ds(half, half)],
                w13_sem.at[slot],
            ),
        )
        w2_copies = (
            pltpu.make_async_copy(w2_hbm.at[expert], w2_buf.at[slot], w2_sem.at[slot]),
            pltpu.make_async_copy(s2_hbm.at[expert], s2_buf.at[slot], w2_sem.at[slot]),
        )
        return part, w13_copies, w2_copies

    def advance(e_idx, part, by):
        """The (expert index, part) of the unit `by` units ahead, without a
        division: `by` is at most `units`."""
        p = part + by
        wrap = p >= units
        return jnp.where(wrap, e_idx + 1, e_idx), jnp.where(wrap, p - units, p)

    def start(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.start()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.start()

    def wait(e_idx, part):
        part, w13_copies, w2_copies = unit_copies(e_idx, part)

        @pl.when(part < w13_parts)
        def _():
            for c in w13_copies:
                c.wait()

        @pl.when(part == w13_parts)
        def _():
            for c in w2_copies:
                c.wait()

    acc[...] = jnp.zeros_like(acc)
    x_q, x_s = _quantize_rows(x_ref[...].astype(jnp.float32))

    for ahead in range(slots - 1):

        @pl.when(ahead < n_units)
        def _prime(*, ahead=ahead):
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
                jnp.where(lane == gid_s[0, expert], wtab_ref[...], 0.0),
                axis=1,
                keepdims=True,
            )
            acc[...] += w * o

        return advance(e_idx, part, 1)

    jax.lax.fori_loop(0, n_units, body, (jnp.int32(0), jnp.int32(0)))
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
    w13_parts,
):
    """The ids entry: the routing table over local ids and the hit vector come
    from the caller; gid_ref is the identity over local ids."""
    _stream(
        gid_ref,
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
    hidden_states, w1, w2, w1_scale, w2_scale, w1_bias, w2_bias, *, activation
):
    """Whether the kernel serves this layer: silu, fp8 weights with one scale
    per output channel, bf16 activations, no expert bias, at most
    MAX_LOCAL_EXPERTS local experts and a gate|up width that splits into two
    lane-aligned halves."""
    if activation != "silu" or w1_bias is not None or w2_bias is not None:
        return False
    if w1_scale is None or w2_scale is None:
        return False
    if hidden_states.ndim != 2 or w1.ndim != 3 or w2.ndim != 3:
        return False
    if hidden_states.dtype != jnp.bfloat16 or w1.dtype != FP8 or w2.dtype != FP8:
        return False
    experts, hidden, two_inter = w1.shape
    if hidden != hidden_states.shape[1] or w2.shape != (
        experts,
        two_inter // 2,
        hidden,
    ):
        return False
    if experts > MAX_LOCAL_EXPERTS or two_inter % GATE_UP_ALIGNMENT:
        return False
    # One scale per output channel, with any singleton axes in between.
    return w1_scale.size == experts * two_inter and w2_scale.size == experts * hidden


@functools.partial(jax.jit, static_argnames=("vmem_limit_bytes", "interpret"))
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
    if inter % LANE:
        raise ValueError(
            f"the intermediate width {inter} must be a multiple of the lane tile"
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
        vmem_limit_bytes = vmem_budget(rows, hidden, inter, w13_parts, two_inter)
    out = pl.pallas_call(
        functools.partial(
            _kernel_ids, n_experts=n_experts, inter=inter, w13_parts=w13_parts
        ),
        out_shape=jax.ShapeDtypeStruct((rows, hidden), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[vmem, vmem, hbm, hbm, hbm, hbm],
            out_specs=vmem,
            scratch_shapes=(
                pltpu.SMEM((n_experts,), jnp.int32),
                pltpu.VMEM((w13_parts + 1, hidden, 2 * (inter // w13_parts)), FP8),
                pltpu.VMEM((w13_parts + 1, inter, hidden), FP8),
                pltpu.VMEM((w13_parts + 1, 1, 2 * (inter // w13_parts)), jnp.float32),
                pltpu.VMEM((w13_parts + 1, 1, hidden), jnp.float32),
                pltpu.SemaphoreType.DMA((w13_parts + 1,)),
                pltpu.SemaphoreType.DMA((w13_parts + 1,)),
                pltpu.VMEM((rows, two_inter), jnp.float32),
                pltpu.VMEM((rows, hidden), jnp.float32),
            ),
            grid=(),
        ),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes),
        interpret=pltpu.InterpretParams() if interpret else False,
        name="moe_lowtok",
    )(gid, hit, x_rows, wtab, s13, s2, w13, w2)
    return out[:tokens]


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
        out = out + w[:, None] * o
    return out.astype(x.dtype)
