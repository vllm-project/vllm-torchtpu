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
"""Batched native FP8/NVFP4 Qwen decode in one distributed Pallas call."""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu
from jax.sharding import PartitionSpec as P

from .attention import attend_paged, validate_kv_scales
from .attention_append import attend_paged_append, exchange_scratch
from .attention_batched_append import (
    attend_paged_batched_append as attend_paged_batched,
)
from .attention_causal_four import attend_paged_causal_four
from .attention_causal_pair import attend_paged_causal_pair
from .collectives import (
    broadcast_sum,
    drain_broadcast_sum,
    drain_hierarchical_sum,
    drain_paired_sum,
    drain_scattered_sum,
    hierarchical_sum,
    paired_sum,
    scattered_sum,
)
from .collectives import canonical_sum as canonical_sum
from .dense import dense_matmul
from .expert_w4a8 import EarlyStart, run_experts
from .expert_w4a8 import scratch_shapes as expert_scratch_shapes
from .gdn_ba import fused_ba_projection
from .input_prefetch import resident_fp8_projection
from .layout import Layout
from .paged_kv import PoolPages
from .shared_prefetch import resident_shared_projection
from .tracing import STAGES

STATE_NAMES = ("conv", "rec", "kcache", "vcache")
LAYER_ROWS = ("norm_in", "norm_post", "shared_gate")
# Smallest live-row count whose residual reduction uses scattered_sum.
SCATTER_MIN_ROWS = 2
# Smallest live-row count whose residual reduction pairs each chip's cores first, when
# Layout.reduction_pairs allows it.
PAIR_MIN_ROWS = 64
HEAD_ROWS = ("qn", "kn")
# Pipelined attention keeps each KV block bank one iteration longer: five banks
# read three blocks ahead.
KV_BUFFERS = 5
F32 = jnp.float32
BF16 = jnp.bfloat16


def scatter_parts(chunk):
    """Column pieces per scattered_sum chunk: staggering the phases pays for wide chunks."""
    return 4 if chunk >= 1024 else 1


def gdn_sum(value):
    """FP32 low/high-half tree; fix recurrence association across TPU backends."""
    width = value.shape[-1]
    if width == 0 or width & (width - 1):
        raise ValueError("GDN reduction width must be a positive power of two")
    value = value.astype(F32)
    while width > 1:
        width //= 2
        value = value[..., :width] + value[..., width:]
    return value[..., 0]


def gdn_recurrence(state, query, key, value, beta, decay):
    """The FP32 delta update and readout of a [heads, value, key] state, before storage rounding."""
    state = state.astype(F32) * decay[:, None, None]
    prediction = gdn_sum(state * key[:, None, :])
    state += ((value - prediction) * beta[:, None])[:, :, None] * key[:, None, :]
    return state, gdn_sum(state * query[:, None, :])


def gdn_sum_rows(value):
    """gdn_sum's halving tree over the second-to-last axis: same pairs, same order."""
    width = value.shape[-2]
    if width == 0 or width & (width - 1):
        raise ValueError("GDN reduction width must be a positive power of two")
    value = value.astype(F32)
    while width > 1:
        width //= 2
        value = value[..., :width, :] + value[..., width:, :]
    return value[..., 0, :]


def gdn_recurrence_columns(state, query, key, value, beta, decay):
    """gdn_recurrence on the transposed [heads, key, value] state the decode stack keeps.

    ``query``/``key`` are [heads, key, value] columns, constant along value.
    Every product and sum is gdn_recurrence's, so the state is its exact
    transpose and the readout is identical, but key reductions add whole
    sublane rows instead of rotating lanes.
    """
    state = state.astype(F32) * decay[:, None, None]
    prediction = gdn_sum_rows(state * key)
    state += key * ((value - prediction) * beta[:, None])[:, None, :]
    return state, gdn_sum_rows(state * query)


def top_k_by_logit(logits, probabilities, top_k, on_selection=None):
    """Top-k of ``probabilities`` by highest probability, then lowest expert.

    BF16 ``logits`` order the probabilities, so each step is one integer maximum
    over (logit, -expert) keys; ``on_selection`` sees each step's experts. Returns
    per-step [rows, 1] experts and probabilities and a [rows, 1] flag of rows whose
    selection may differ from probability order (equal probabilities from
    different logits); those rows need the probability-order selection.
    """
    experts = logits.shape[-1]
    indices = jnp.arange(experts)[None, :]
    bits = jax.lax.bitcast_convert_type(jnp.where(logits == 0, 0.0, logits), jnp.int32)
    ordered = jnp.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    keys = jnp.left_shift(jnp.right_shift(ordered, 16), 16) | (experts - 1 - indices)
    selected = []
    for _ in range(top_k):
        best = jnp.max(keys, axis=-1, keepdims=True)
        keys = jnp.where(keys == best, jnp.iinfo(jnp.int32).min, keys)
        selected.append(experts - 1 - (best & 0xFFFF))
        if on_selection is not None:
            on_selection(selected[-1][:, 0])
    chosen = [
        jnp.sum(jnp.where(indices == expert, probabilities, 0), axis=-1, keepdims=True)
        for expert in selected
    ]
    disagree = jnp.zeros(chosen[0].shape, jnp.bool_)
    for k in range(1, top_k):
        disagree |= (chosen[k] > chosen[k - 1]) | (
            (chosen[k] == chosen[k - 1]) & (selected[k] < selected[k - 1])
        )
    outranked = (keys != jnp.iinfo(jnp.int32).min) & (
        (probabilities > chosen[-1]) | ((probabilities == chosen[-1]) & (indices < selected[-1]))
    )
    disagree |= jnp.max(jnp.where(outranked, 1, 0), axis=-1, keepdims=True) > 0
    return selected, chosen, disagree


def transpose_heads(value):
    """Per-head [a, b] -> [b, a] of a [heads, a, b] value."""
    return jnp.stack([jnp.transpose(value[head]) for head in range(value.shape[0])])


def fp8_block_dot(source, packed, scales, destination):
    """Native 128x128 scales multiply FP32 partials; weights are never requantized."""
    _, contraction = source.shape
    columns = packed.shape[1]
    if contraction % 128 or columns % 128:
        raise ValueError("FP8 dot requires complete 128x128 blocks")
    destination[...] = jnp.zeros(destination.shape, F32)

    @pl.loop(0, columns // 128)
    def output_block(n):
        @pl.loop(0, contraction // 128)
        def input_block(k):
            lhs = source[:, pl.ds(k * 128, 128)].astype(BF16)
            rhs = packed[pl.ds(k * 128, 128), pl.ds(n * 128, 128)].astype(BF16)
            value = jnp.dot(lhs, rhs, preferred_element_type=F32)
            # Scales are a small tiled VMEM matrix. Arbitrary scalar ref
            # slices are unaligned; select from a complete static load instead.
            selected = (jnp.arange(scales.shape[0])[:, None] == k) & (
                jnp.arange(scales.shape[1])[None, :] == n
            )
            scale = jnp.sum(
                jnp.where(selected, scales[...].astype(F32), 0),
                axis=(0, 1),
                keepdims=True,
            )
            destination[:, pl.ds(n * 128, 128)] += value * scale


def fp4_values(nibbles):
    """Decode E2M1 directly in registers; low nibble precedes high along K."""
    magnitude = (nibbles.astype(jnp.int32) & 7).astype(F32)
    value = jnp.where(
        magnitude < 2,
        magnitude * 0.5,
        jnp.where(
            magnitude < 4,
            1 + (magnitude - 2) * 0.5,
            jnp.where(magnitude < 6, magnitude - 2, 4 + (magnitude - 6) * 2),
        ),
    )
    return jnp.where((nibbles.astype(jnp.int32) & 8) != 0, -value, value)


def fp4_activation(source, global_scale):
    """Calibrated NVFP4 activation quantization with FP32 RNE division."""
    value = source.astype(F32)
    blocks = value.reshape(value.shape[0], value.shape[1] // 16, 16)
    peak = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
    inverse = 1.0 / global_scale
    scale = jnp.clip(inverse * (peak * jnp.float32(1 / 6)), 0, 448)
    scale = scale.astype(jnp.float8_e4m3fn).astype(F32)
    inverse_block = jnp.where(scale != 0, inverse / scale, 0)
    normalized = jnp.clip(blocks * inverse_block, -6, 6)
    absolute = jnp.abs(normalized)
    code = jnp.zeros(absolute.shape, jnp.int32)
    for index, threshold in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        # At a midpoint choose the even E2M1 significand encoding.
        code += ((absolute > threshold) | ((absolute == threshold) & (index % 2 == 1))).astype(
            jnp.int32
        )
    sign = jax.lax.bitcast_convert_type(normalized, jnp.int32) < 0
    local = fp4_values(code | jnp.where(sign, 8, 0)) * scale
    return local.reshape(value.shape).astype(BF16)


def nvfp4_dot(source, packed, scales, input_global, weight_global, destination):
    """Unpack only one output tile; exact local scale products feed BF16 MXU."""
    contraction, columns = packed.shape[0] * 2, packed.shape[1]
    activation = fp4_activation(source[:, :contraction], input_global)

    @pl.loop(0, columns // 128)
    def output_block(n):
        payload = packed[:, pl.ds(n * 128, 128)].astype(jnp.int32)
        low, high = fp4_values(payload & 15), fp4_values(payload >> 4)
        decoded = jnp.stack((low, high), axis=1).reshape(contraction, 128)
        block_scales = scales[:, pl.ds(n * 128, 128)].astype(F32)
        expanded = jnp.broadcast_to(block_scales[:, None, :], (contraction // 16, 16, 128)).reshape(
            contraction, 128
        )
        # E2M1 × E4M3 has at most six significand bits: this BF16 cast is exact.
        weight = (decoded * expanded).astype(BF16)
        value = jnp.dot(activation, weight, preferred_element_type=F32)
        destination[:, pl.ds(n * 128, 128)] = value * (input_global * weight_global)


def _stack(
    c,
    layout,
    context,
    batch_size,
    weights,
    states,
    tokens,
    positions,
    interpret,
    trace=False,
    kv_scales=(),
    return_hidden=False,
    causal_pair=False,
    _mtp_state_owner_axis=False,
    _mtp_causal_row_update=None,
    _mtp_requests=1,
    _mtp_active=None,
    _mtp_separate_cuts=False,
    paged=None,
):
    """Decode one token per request, or verify consecutive causal rows (causal_pair).

    Verification rows are [request, position] major: one request with two or
    four positions, or two, four, eight or sixteen requests with four positions each.
    Every request's GDN state stays resident while its positions advance in order, and the
    state after each non-final position is stored as a bounded recurrent cut.
    An optional per-row ``_mtp_active`` mask skips GDN updates and KV appends of
    rows beyond each request's verified prefix.

    ``paged`` (``{"pools", "tables", "pages_per_block"}``) reads every request's
    KV pages from vLLM's unified pools (``paged_kv``) through its ``[batch,
    width]`` rank-local block table instead of the ``kcache``/``vcache`` states;
    the pools are read-only operands, and the step's new key/value columns
    ``[attention layers, batch, 2, D]`` (FP8) are returned, as the states'
    ``"columns"``, for the caller to commit.
    """
    if _mtp_causal_row_update is not None and (not causal_pair or not callable(_mtp_causal_row_update)):
        raise ValueError("Internal row update requires a callable and causal target verification")
    if not causal_pair and _mtp_requests != 1:
        raise ValueError("Independent verification requests require causal target verification")
    requests = _mtp_requests if causal_pair else batch_size
    causal_rows = batch_size // requests
    if causal_pair and (
        (requests, causal_rows) not in ((1, 2), (1, 4), (2, 4), (4, 4), (8, 4), (16, 4))
        or not return_hidden
        or layout.context_parallel != 8
    ):
        raise ValueError(
            "Causal target verification requires 1x2, 1x4, 2x4, 4x4, 8x4 or 16x4 rows, "
            "hidden output and CP8"
        )
    if (_mtp_active is not None or _mtp_separate_cuts) and causal_rows != 4:
        raise ValueError("Row masks and separate cuts are four-position verification options")
    if _mtp_causal_row_update is not None and causal_rows != 2:
        raise ValueError("Internal row update is a two-position verification option")
    if _mtp_state_owner_axis and causal_rows == 4:
        raise ValueError("Four-position verification owns its state without an owner axis")
    # One output per cut, or every cut stacked in one array per state family.
    cut_count = causal_rows - 1
    separate_cuts = _mtp_separate_cuts or cut_count == 1
    cut_outputs = 2 * (cut_count if separate_cuts else 1) if causal_pair else 0
    if paged is not None and (
        causal_pair or _mtp_state_owner_axis or trace or return_hidden or layout.context_parallel != 8
    ):
        raise ValueError("Pool pages serve plain CP8 decode")
    state_names = ("conv", "rec") if paged is not None else STATE_NAMES
    pool_count = len(paged["pools"]) if paged is not None else 0
    scalar_count = 2 + int(_mtp_active is not None) + int(paged is not None)
    # Ordinary C1-C16 decode and every causal verification share the selected
    # fast-path ownership; other batches keep the general path.
    small = batch_size in (1, 2, 4, 8, 16) or causal_pair
    # Rows past eight take eight-row tiles. Lane reductions run per tile: Mosaic
    # associates a 16-row reduction differently.
    tp, h, rows = layout.tp, c.dim, max(8, batch_size)
    eight_rows = tuple(slice(start, start + 8) for start in range(0, rows, 8))
    nvfp4 = c.quantization == "nvfp4"
    weight_names = layout.weight_names
    kh, vh = layout.local_k_heads, layout.local_v_heads
    qh, kvh = layout.local_q_heads, layout.local_kv_heads
    kl, vl = kh * c.k_dim, vh * c.v_dim
    conv_width = layout.local_conv
    ew, sw = layout.expert_width, layout.shared_width
    rw, rt = c.experts // layout.router_tp, layout.router_tp
    trace_width = max(h, c.experts, 2 * sw)
    lm_width = layout.vocab_shard_padded
    # Whole row tiles of logits leave through HBM, sparing their VMEM output window.
    logits_in_hbm = batch_size % 8 == 0
    width = max(
        h, weights["qkvz"].shape[-1], weights["qwkv"].shape[-1], 2 * ew, 2 * sw, lm_width, c.experts
    )
    local_size, hosts = min(8, tp), max(1, tp // 8)
    # Residual reductions use one send and one receive semaphore per phase and parity.
    transfer_count = 4
    # Isolated candidate: only the fast decode and verification ownership participates.
    input_prefetch_enabled = (
        layout.fast
        and tp == 32
        and (layout.expert_tp == 1 or layout.expert_reduction == "residual")
        and layout.context_parallel == 8
        and small
        and h % 128 == 0
        and max(conv_width, 2 * qh * c.head_dim) <= 1024
        and conv_width % 128 == 0
    )
    next_input_width = max(conv_width, 2 * qh * c.head_dim)
    gdn_writeback_enabled = layout.fast and small
    # Verification and multi-request decode keep every request's GDN state
    # resident, moving each layer's request block in one transfer.
    resident_requests = causal_pair or (gdn_writeback_enabled and batch_size > 1)
    # Beyond sixteen verification (eight decode) rows they advance in aligned eight-row
    # groups of two four-position or eight one-row requests, and only one group's
    # working state and cuts are resident.
    grouped = resident_requests and rows > (16 if causal_pair else 8)
    group_requests = max(1, 8 // causal_rows) if grouped else requests
    groups = requests // group_requests
    # Fast-path decode and verification send their live rows in one aligned row-tile packet.
    compact = small and h % 1024 == 0
    # A BF16 wire packs the same live rows into whole [16, 128] BF16 tiles.
    wire_dtype = BF16 if compact and layout.reduction_wire == "bfloat16" else F32
    packet_rows = 16 if wire_dtype == BF16 else rows
    packet_cols = h * batch_size // packet_rows if compact else h
    # Multi-row packets reduce-scatter over the host-local group instead of
    # sending every peer the whole packet; both keep the same FP32 sum order.
    scattered = (
        compact
        and batch_size >= SCATTER_MIN_ROWS
        and local_size == 8
        and packet_cols % (local_size * 128) == 0
    )
    sum_parts = scatter_parts(packet_cols // local_size) if scattered else 1
    # Two BF16 rows broadcast each owner's host sum to every rank instead of
    # exchanging it with the other hosts and gathering the total: one phase fewer.
    broadcast = scattered and wire_dtype == BF16 and batch_size == 2 and hosts > 1
    # Wide BF16 packets halve their host-local off-chip bytes by reducing over each
    # chip's two cores first.
    paired = (
        scattered
        and layout.reduction_pairs
        and wire_dtype == BF16
        and batch_size >= PAIR_MIN_ROWS
        and batch_size % packet_rows == 0
        and h % (packet_cols // local_size) == 0
    )
    if scattered:
        transfer_count = 4 if broadcast else (10 if paired else 6) * sum_parts
    # Live rows filling several packet rows take their sum in blocks straight into the
    # residual, never holding the whole batch's sum as one value.
    delivered = (
        scattered and not broadcast and batch_size > packet_rows and batch_size % packet_rows == 0
    )

    # The fast path accumulates the MoE in proj: routing has consumed proj by then, and
    # nothing else writes it before the layer's second residual sum reduces it.
    moe_in_proj = layout.fast

    def v(shape, dtype=BF16):
        return tpu.VMEM(shape, dtype)

    def dm(n):
        return tpu.SemaphoreType.DMA((max(1, n),))

    scratch = dict(
        residual=v((rows, h)),
        normed=v((rows, h)),
        proj=v((rows, width), F32),
        norm=v((h,)),
        copy_sem=dm(1),
        attention_barriers=tpu.SemaphoreType.REGULAR((2,)),
        embrow=v((h,)),
        row_tile=v((8, h)),
        row_tile_f32=v((8, 128), F32),
        # BF16 values held in FP32, so a group's rows update whole aligned row tiles.
        attended=v((rows, max(vl, qh * c.head_dim)), F32),
        middle=v((rows, max(ew, layout.shared_width_padded))),
        gateup=v((rows, 2 * max(ew, layout.shared_width_padded)), F32),
        sends=dm(transfer_count),
        recvs=dm(transfer_count),
        # Separate channels let GDN metadata and request-0 state reads overlap
        # projection weight DMAs. All six reads complete before the first use.
        gdn_prefetch_sem=dm(6),
        gdn_alog_tile=v((8, weights["alog"].shape[-1]), F32),
        gdn_dtb_tile=v((8, weights["dtb"].shape[-1])),
        gdn_onorm_tile=v((8, weights["onorm"].shape[-1])),
        cb=v(
            (requests, c.conv_size - 1, conv_width)
            if resident_requests
            else (c.conv_size - 1, conv_width)
        ),
        cw=v((c.conv_size, conv_width)),
        rec=v(
            (requests, vh, c.v_dim, c.k_dim) if resident_requests else (vh, c.v_dim, c.k_dim),
            states["rec"].dtype,
        ),
        # The recurrence's transposed working copy of rec; DMAs use rec.
        rec_t=v(
            (group_requests, vh, c.k_dim, c.v_dim)
            if resident_requests
            else (vh, c.k_dim, c.v_dim),
            states["rec"].dtype,
        ),
        alog=v((vh,), F32),
        dtb=v((vh,)),
        onorm=v((c.v_dim,)),
        qnorm=v((c.head_dim,)),
        knorm=v((c.head_dim,)),
        router_parts=v((rt, rows, rw)),
        router=v((rows, c.experts), F32),
        route_ids=v((rows, c.top_k), jnp.int32),
        route_probs=v((rows, c.top_k), F32),
        small_sends=dm(tp - 1),
        small_recvs=dm(tp - 1),
        candidates=v((tp, rows, 128), F32),
    )
    if _mtp_active is not None:
        # Each row's verification flag in lane 0; rows outside their request's verified
        # prefix route to no expert.
        scratch["row_live"] = v((rows, 128), jnp.int32)
    # The routed-expert pipeline and full-attention cache reads have disjoint
    # lifetimes. Scoped allocations let Mosaic reuse their VMEM without
    # changing either MXU contractions or the whole-context softmax.
    expert_scratch = dict(
        egu=v((2, *weights["egu"].shape[-2:]), weights["egu"].dtype),
        egus=v((2, *weights["egu_scale"].shape[-2:]), weights["egu_scale"].dtype),
        edw=v((2, *weights["edw"].shape[-2:]), weights["edw"].dtype),
        edws=v((2, *weights["edw_scale"].shape[-2:]), weights["edw_scale"].dtype),
        expert_sem=dm(10 if nvfp4 else 8),
    )
    if not moe_in_proj:
        scratch["moe"] = v((rows, h), F32)
    if broadcast:
        scratch.update(
            stage=v((packet_rows, packet_cols), wire_dtype),
            scatter=v((2, local_size, packet_rows, packet_cols // local_size), wire_dtype),
            sums=v((2, hosts, packet_rows, packet_cols), wire_dtype),
        )
    elif paired:
        chunk = packet_cols // local_size
        scratch.update(
            stage=v((packet_rows, packet_cols // 2), wire_dtype),
            scatter=v((2, local_size // 2, packet_rows, chunk), wire_dtype),
            host=v((2, hosts, packet_rows, chunk), wire_dtype),
            gathered=v((2, packet_rows, packet_cols), BF16),
            pin=v((local_size // 2, packet_rows, chunk), F32),
        )
    elif scattered:
        chunk = packet_cols // local_size
        scratch.update(
            stage=v((packet_rows, packet_cols), wire_dtype),
            scatter=v((2, local_size, packet_rows, chunk), wire_dtype),
            host=v((2, hosts, packet_rows, chunk), wire_dtype),
            gathered=v((2, packet_rows, packet_cols), BF16),
        )
    else:
        scratch.update(
            local=v((2, local_size, packet_rows, packet_cols), wire_dtype),
            remote=v((2, hosts, packet_rows, packet_cols), wire_dtype),
        )
    if nvfp4:
        expert_scratch["eparams"] = v((2, 8, 128), F32)
        scratch.update(
            attention_params=v((8, 128), F32),
        )
    if gdn_writeback_enabled:
        scratch["gdn_writeback_sem"] = dm(2)
    # Verification stages a group's recurrent cut after each position and stores it
    # while later rows run; the next group reuses the cut's slot once that store
    # completes, and the next GDN layer drains the last group's with the final state.
    staged_cuts = causal_pair and gdn_writeback_enabled
    if causal_pair:
        scratch.update(
            cut_conv=v((cut_count, group_requests, c.conv_size - 1, conv_width)),
            cut_rec=v((cut_count, group_requests, vh, c.v_dim, c.k_dim), states["rec"].dtype),
            cut_sem=dm(2 * cut_count),
        )
    if trace:
        scratch["trace_tile"] = v((rows, trace_width), F32)
    # A production TP4 router shard is8192x128 BF16: exactly2MiB.
    # Keep larger nonproduction layouts on the selected bounded dense path.
    router_prefetch = h * rw * weights["router"].dtype.itemsize <= 2 * 1024**2
    if router_prefetch:
        scratch.update(router_weight=v((h, rw), weights["router"].dtype), router_sem=dm(1))
    # The small router exchange owns only router_parts and its DMA semaphores.
    # Shared gating uses separate row/norm scratch and the finished RMSNorm row.
    shared_gate_overlap = layout.fast and layout.tp == 32 and rt == 4 and small
    # Dedicated ordinary storage is scoped after O and before routed experts.
    # The strict gate also preserves the selected full-N8192 DOWN contraction.
    shared_weights_prefetch_enabled = (
        layout.fast
        and tp == 32
        and (layout.expert_tp == 1 or layout.expert_reduction == "residual")
        and layout.context_parallel == 8
        and rt == 4
        and small
        and h == 8192
        and sw == 64
        and layout.dense_output_tile == 8192
        and weights["gu"].shape[-2:] == (8192, 128)
        and weights["dw"].shape[-2:] == (64, 8192)
        and weights["gu"].dtype == BF16
        and weights["dw"].dtype == BF16
    )
    # Whole-layer input/output weight residency and early expert transfers fit
    # beside quarter-width routed-expert buffers, not whole-expert ones.
    layer_prefetch_enabled = (
        input_prefetch_enabled
        and shared_weights_prefetch_enabled
        and ew <= 512
        and weights["outw"].shape[-2:] == weights["ow"].shape[-2:] == (vl, h)
        and weights["outw"].dtype == weights["ow"].dtype
    )
    # Starting the first expert during top-k changes the expert summation order.
    early_experts = layer_prefetch_enabled and layout.expert_reduction == "residual"
    if layer_prefetch_enabled:
        next_input_width = max(weights["qkvz"].shape[-1], weights["qwkv"].shape[-1])
        if rows > 32:
            # Sixty-four rows hold the successor's leading 1024 columns, leaving VMEM for
            # routed-expert gate/up transfers two experts ahead; the rest streams in place.
            next_input_width = min(next_input_width, 1024)
        scratch.update(
            row_tiles=v((len(LAYER_ROWS), 8, h), weights["norm_in"].dtype),
            head_row_tiles=v((len(HEAD_ROWS), 8, c.head_dim), weights["qn"].dtype),
            row_sem=dm(len(LAYER_ROWS) + len(HEAD_ROWS)),
        )
    elif causal_pair and requests == 2:
        # Two resident GDN requests beside whole-expert buffers leave room for a
        # 512-column successor prefetch; the first projection streams the rest.
        next_input_width = min(next_input_width, 512)
    if input_prefetch_enabled:
        scratch.update(
            next_input_weight=v((h, next_input_width), weights["qkvz"].dtype),
            next_input_params=v((8, 128), F32),
            next_input_sem=dm(2),
        )
    attention_optimized = small and layout.context_parallel == 8 and kvh == 1
    # Fast-path decode and four-position verification keep their CP inboxes for
    # the whole call, so their attention needs no entry/exit barriers.
    attention_inbox = {}
    if attention_optimized and (not causal_pair or causal_rows == 4):
        attention_inbox = exchange_scratch(layout, batch_size)
        scratch.update(attention_inbox)
    if paged is not None:
        if not attention_optimized:
            raise ValueError("Pool pages serve the optimized CP8 attention (one KV head per rank)")
        scratch["new_columns"] = v((batch_size, 2, c.head_dim), jnp.float8_e4m3fn)
    scratch_names = tuple(scratch)
    legacy_interpreter = interpret is True
    id_type = pl.DeviceIdType.LOGICAL if legacy_interpreter else pl.DeviceIdType.MESH

    def peer_id(rank):
        # The one-axis CPU DMA interpreter requires scalar logical IDs.
        return rank if legacy_interpreter else (rank,)

    # One contraction serves every row's B and A projections.
    fused_ba = layout.fast and (batch_size <= 16 or causal_pair)
    single_attention = attend_paged_append if attention_optimized else attend_paged

    def kernel(position_ref, token_ref, *args):
        active_ref = tables_ref = None
        if _mtp_active is not None:
            active_ref, args = args[0], args[1:]
        if paged is not None:
            tables_ref, args = args[0], args[1:]
        nr, ns = len(weight_names), len(state_names)
        w = dict(zip(weight_names, args[:nr], strict=True))
        pools, args = args[nr + ns : nr + ns + pool_count], args[nr + ns + pool_count :]
        logits_out, next_out = args[:2]
        state_out = dict(zip(state_names, args[2 : 2 + ns], strict=True))
        args = args[2 + ns :]
        if paged is not None:
            columns_out, args = args[0], args[1:]
        trace_out = args[0] if trace else None
        if return_hidden:
            hidden_out = args[int(trace)]
        cut_refs = args[int(trace) + int(return_hidden) :][:cut_outputs]
        args = args[int(trace) + int(return_hidden) + cut_outputs :]

        def pool_pages(ordinal, row):
            return PoolPages(pools, ordinal, tables_ref, row, paged["pages_per_block"])

        def publish_columns(ordinal, scales, keys, values):
            # This layer's new key/value columns as the pools store them, for the commit.
            for b, (key, value) in enumerate(zip(keys, values, strict=True)):
                for kind, (row, scale) in enumerate(((key, scales[0]), (value, scales[1]))):
                    encoded = jnp.clip(row.astype(F32) / scale, -448, 448).astype(jnp.float8_e4m3fn)
                    r["new_columns"][b, kind] = encoded
            copy(r["new_columns"], columns_out.at[ordinal])

        def cut(name, slot, ordinal):
            # Every request's state after position slot, as one cut or stacked cuts.
            offset = ("conv", "rec").index(name)
            if separate_cuts:
                return cut_refs[2 * slot + offset].at[ordinal]
            return cut_refs[offset].at[slot, ordinal]

        r = dict(zip(scratch_names, args, strict=True))
        inbox = {name: r[name] for name in attention_inbox}
        moe = r["proj"].at[:, :h] if moe_in_proj else r["moe"]
        if active_ref is not None:
            live = jnp.zeros((rows, 128), jnp.int32)
            for b in range(batch_size):
                live = jnp.where(jnp.arange(rows)[:, None] == b, active_ref[b], live)
            r["row_live"][...] = live

        def live_rows():
            return None if active_ref is None else r["row_live"][:, :1] != 0

        rank = jax.lax.axis_index("tp")
        # Aliased state is updated in place. All peers must allocate communication scratch first.
        if tp > 1:
            barrier = tpu.get_barrier_semaphore()
            for peer in range(1, tp):
                pl.semaphore_signal(
                    barrier, 1, device_id=peer_id(rank ^ peer), device_id_type=id_type
                )
            pl.semaphore_wait(barrier, tp - 1)

        def copy(src, dst):
            tpu.make_async_copy(src, dst, r["copy_sem"].at[0]).start()
            tpu.make_async_copy(src, dst, r["copy_sem"].at[0]).wait()

        def gdn_state(name, ordinal, request):
            # Resident requests move as one block; otherwise one request at a time.
            if resident_requests:
                return state_out[name].at[ordinal]
            return state_out[name].at[ordinal, request]

        def gdn_writebacks(ordinal, request):
            # Persistent source buffers and dedicated channels survive all
            # projection/expert scopes. Destinations retain actual ownership.
            return (
                tpu.make_async_copy(
                    r["cb"], gdn_state("conv", ordinal, request), r["gdn_writeback_sem"].at[0]
                ),
                tpu.make_async_copy(
                    r["rec"], gdn_state("rec", ordinal, request), r["gdn_writeback_sem"].at[1]
                ),
            )

        def when_active(b):
            # Inactive verification rows leave their request's state untouched.
            if active_ref is None:
                return lambda body: body()
            return pl.when(active_ref[b] != 0)

        def cut_writebacks(ordinal, slot, first=0):
            # The cut after position ``slot`` of the group whose first request is ``first``.
            return tuple(
                tpu.make_async_copy(
                    r["cut_" + name].at[slot],
                    cut(name, slot, ordinal).at[pl.ds(first, group_requests)],
                    r["cut_sem"].at[2 * slot + offset],
                )
                for offset, name in enumerate(("conv", "rec"))
            )

        def drain_gdn_writebacks(ordinal, request):
            for transfer in gdn_writebacks(ordinal, request):
                transfer.wait()
            if staged_cuts:
                # Earlier groups' cuts were waited before their slots' reuse.
                for slot in range(cut_count):
                    for transfer in cut_writebacks(ordinal, slot, requests - group_requests):
                        transfer.wait()

        def capture(layer, stage, value):
            if trace:
                r["trace_tile"][...] = jnp.pad(
                    value.astype(F32), ((0, 0), (0, trace_width - value.shape[-1]))
                )
                copy(r["trace_tile"], trace_out.at[layer, STAGES.index(stage)])

        def read_row(matrix, row, destination):
            # HBM is tiled by rows. Move the containing complete tile, then
            # select in registers; arbitrary token/layer indices are not aligned.
            base = pl.multiple_of(row // 8 * 8, 8)
            width = matrix.shape[1]
            tile = (
                r["row_tile_f32"].at[:, :width]
                if matrix.dtype == F32
                else r["row_tile"].at[:, :width]
            )
            copy(matrix.at[pl.ds(base, 8), :], tile)
            select_row(tile, row, destination)

        def select_row(tile, row, destination):
            selected = jnp.sum(
                jnp.where(jnp.arange(8)[:, None] == row % 8, tile[...].astype(F32), 0),
                axis=0,
            )
            destination[...] = selected[: destination.shape[0]].astype(destination.dtype)

        def row_copy(name, row):
            # One dedicated tile per table; a successor prefetch is waited by its layer.
            if name in LAYER_ROWS:
                index = LAYER_ROWS.index(name)
                tile = r["row_tiles"].at[index]
            else:
                index = len(LAYER_ROWS) + HEAD_ROWS.index(name)
                tile = r["head_row_tiles"].at[HEAD_ROWS.index(name)]
            base = pl.multiple_of(row // 8 * 8, 8)
            transfer = tpu.make_async_copy(
                w[name].at[pl.ds(base, 8), :], tile, r["row_sem"].at[index]
            )
            return transfer, tile

        def start_rows(layer, full_ordinal=None):
            for name in LAYER_ROWS:
                row_copy(name, layer)[0].start()
            if full_ordinal is not None:
                for name in HEAD_ROWS:
                    row_copy(name, full_ordinal)[0].start()

        def layer_row(name, row, destination):
            if layer_prefetch_enabled:
                transfer, tile = row_copy(name, row)
                transfer.wait()
                select_row(tile, row, destination)
            else:
                read_row(w[name], row, destination)

        def matmul(src, matrix, dst, *, rounded=False):
            dense_matmul(src, matrix, dst, output_tile=layout.dense_output_tile, rounded=rounded)

        def parameter(params, column):
            return jnp.sum(jnp.where(jnp.arange(128)[None, :] == column, params[...], 0))

        def quantized_matmul(src, matrix, dst, input_scale, weight_scale, mask=None, rounded=False):
            dense_matmul(
                src,
                matrix,
                dst,
                output_tile=layout.dense_output_tile,
                input_scale=input_scale,
                weight_scale=weight_scale,
                mask=mask,
                rounded=rounded,
            )

        def next_input_copies(full, ordinal):
            family, parameters = ("qwkv", "fparams") if full else ("qkvz", "lparams")
            columns = 2 * qh * c.head_dim if full else conv_width
            if layer_prefetch_enabled:
                columns = w[family].shape[-1]
            columns = min(columns, next_input_width)
            return (
                tpu.make_async_copy(
                    w[family].at[ordinal, :, pl.ds(0, columns)],
                    r["next_input_weight"].at[:, pl.ds(0, columns)],
                    r["next_input_sem"].at[0],
                ),
                tpu.make_async_copy(
                    w[parameters].at[ordinal],
                    r["next_input_params"],
                    r["next_input_sem"].at[1],
                ),
            )

        def input_parameters(ordinal, full):
            family = "fparams" if full else "lparams"
            if input_prefetch_enabled:
                # Layer 0's copies start at kernel entry, every later layer's
                # in its predecessor's MoE.
                with jax.named_scope("next_input_prefetch_wait"):
                    for transfer in next_input_copies(full, ordinal):
                        transfer.wait()
                r["attention_params"][...] = r["next_input_params"][...]
            else:
                copy(w[family].at[ordinal], r["attention_params"])

        def input_projection(src, matrix, params, segments, after=None):
            # Columns past a narrowed successor bank stream into scoped VMEM from
            # entry, behind the bank's contractions. ``after`` runs inside that scope
            # with the resident view of whole columns, or None where they stream.
            tail_width = matrix.shape[-1] - next_input_width if layer_prefetch_enabled else 0

            def project(tail=None):
                def tail_columns(base, width):
                    if tail is None or base < next_input_width:
                        return None
                    if not waited:
                        tail[1].wait()
                        waited.append(True)
                    return tail[0].at[:, pl.ds(base - next_input_width, width)]

                waited = []
                for component, start, width in segments:
                    # QKV/Z/Q/K/V begin on whole output tiles. GDN B/A share the
                    # final tile but retain their separate calibrated input scales.
                    base = start // 128 * 128
                    end = (start + width + 127) // 128 * 128
                    mask = None
                    if start != base or end != start + width:
                        mask = (jnp.arange(128) >= start - base) & (
                            jnp.arange(128) < start - base + width
                        )

                    def synchronous_projection():
                        quantized_matmul(
                            src,
                            matrix.at[:, pl.ds(base, end - base)],
                            r["proj"].at[:, pl.ds(base, end - base)],
                            parameter(params, component * 2),
                            parameter(params, component * 2 + 1),
                            mask=mask,
                            rounded=True,
                        )

                    resident = component == 0 or (
                        layer_prefetch_enabled and mask is None and base < next_input_width
                    )
                    streamed = tail_columns(base, end - base) if mask is None else None
                    if streamed is not None:
                        with jax.named_scope("prefetched_projection"):
                            resident_fp8_projection(
                                src,
                                streamed,
                                r["proj"].at[:, pl.ds(base, end - base)],
                                parameter(params, component * 2),
                                parameter(params, component * 2 + 1),
                            )
                    elif input_prefetch_enabled and resident:
                        with jax.named_scope("prefetched_projection"):
                            held = min(end, next_input_width) - base
                            resident_fp8_projection(
                                src,
                                r["next_input_weight"].at[:, pl.ds(base, held)],
                                r["proj"].at[:, pl.ds(base, held)],
                                parameter(params, component * 2),
                                parameter(params, component * 2 + 1),
                            )
                            rest = tail_columns(base + held, end - base - held)
                            if rest is not None:
                                resident_fp8_projection(
                                    src,
                                    rest,
                                    r["proj"].at[:, pl.ds(base + held, end - base - held)],
                                    parameter(params, component * 2),
                                    parameter(params, component * 2 + 1),
                                )
                            elif base + held < end:
                                quantized_matmul(
                                    src,
                                    matrix.at[:, pl.ds(base + held, end - base - held)],
                                    r["proj"].at[:, pl.ds(base + held, end - base - held)],
                                    parameter(params, component * 2),
                                    parameter(params, component * 2 + 1),
                                    rounded=True,
                                )
                    else:
                        synchronous_projection()
                if after is not None:
                    after(
                        lambda base, width: (
                            r["next_input_weight"].at[:, pl.ds(base, width)]
                            if layer_prefetch_enabled and base + width <= next_input_width
                            else tail_columns(base, width)
                        )
                    )
                if tail is not None and not waited:
                    tail[1].wait()

            if tail_width <= 0:
                project()
                return

            def scoped(tail, tail_sem):
                transfer = tpu.make_async_copy(
                    matrix.at[:, pl.ds(next_input_width, tail_width)], tail, tail_sem.at[0]
                )
                transfer.start()
                project((tail, transfer))

            pl.run_scoped(scoped, tail=v((h, tail_width), matrix.dtype), tail_sem=dm(1))

        def output_copy(full, ordinal, output):
            # ``output`` is the attention phase's scoped weight buffer and semaphore.
            family = "ow" if full else "outw"
            return tpu.make_async_copy(w[family].at[ordinal], output[0], output[1].at[0])

        def output_projection(full, ordinal, first_parameter, output):
            input_scale = parameter(r["attention_params"], first_parameter)
            weight_scale = parameter(r["attention_params"], first_parameter + 1)
            if layer_prefetch_enabled:
                with jax.named_scope("output_weight_wait"):
                    output_copy(full, ordinal, output).wait()
                resident_fp8_projection(
                    r["attended"],
                    output[0],
                    r["proj"],
                    input_scale,
                    weight_scale,
                    rounded=False,
                )
            else:
                family = "ow" if full else "outw"
                quantized_matmul(
                    r["attended"], w[family].at[ordinal], r["proj"], input_scale, weight_scale
                )

        sum_names = (
            ("stage", "scatter", "sums", "sends", "recvs")
            if broadcast
            else ("stage", "scatter", "host", "gathered", "sends", "recvs")
        ) + (("pin",) if paired else ())

        def reduce_sum_impl(src, iteration, deliver=None):
            # The last phase's sends complete off the critical path; the next
            # call (or kernel exit) drains them before their source is reused.
            if broadcast:
                return broadcast_sum(
                    src,
                    iteration,
                    rank,
                    {name: r[name] for name in sum_names},
                    id_type=id_type,
                    defer_send=True,
                )
            if paired:
                return paired_sum(
                    src,
                    iteration,
                    rank,
                    {name: r[name] for name in sum_names},
                    id_type=id_type,
                    deliver=deliver,
                    defer_send=True,
                    parts=sum_parts,
                )
            if scattered:
                return scattered_sum(
                    src,
                    iteration,
                    rank,
                    {name: r[name] for name in sum_names},
                    id_type=id_type,
                    defer_send=True,
                    parts=sum_parts,
                    deliver=deliver,
                )
            return hierarchical_sum(
                src,
                iteration,
                rank,
                r["local"],
                r["remote"],
                r["sends"],
                r["recvs"],
                compact=compact,
                id_type=id_type,
                defer_send=True,
            )

        def drain_sum(iteration):
            if broadcast:
                drain_broadcast_sum(iteration, {name: r[name] for name in sum_names})
            elif paired:
                drain_paired_sum(iteration, {name: r[name] for name in sum_names}, sum_parts)
            elif scattered:
                drain_scattered_sum(iteration, {name: r[name] for name in sum_names}, sum_parts)
            else:
                drain_hierarchical_sum(iteration, r["remote"], r["sends"])

        def reduce_sum(src, iteration, deliver=None):
            with jax.named_scope("layer_residual_allreduce"):
                if not isinstance(iteration, int) or iteration > 0:
                    drain_sum(iteration - 1)
                return reduce_sum_impl(src, iteration, deliver)

        def normalize(name=None, ordinal=None):
            if name is None:
                copy(w["fnorm"], r["norm"])
            else:
                layer_row(name, ordinal, r["norm"])
            for tile in eight_rows:
                x = r["residual"][tile].astype(F32)
                x *= jax.lax.rsqrt(jnp.mean(x * x, axis=-1, keepdims=True) + c.eps)
                r["normed"][tile] = (x * (1 + r["norm"][...].astype(F32))).astype(BF16)

        def residual_add(iteration):
            if delivered:

                def add(rows, columns, total):
                    residual = r["residual"].at[rows, columns]
                    residual[...] = (residual[...] + total.astype(BF16)).astype(BF16)

                reduce_sum(r["proj"].at[:, :h], iteration, add)
                return
            total = reduce_sum(r["proj"].at[:, :h], iteration).astype(BF16)
            r["residual"][...] = (r["residual"][...] + total).astype(BF16)

        if layer_prefetch_enabled:
            start_rows(0, 0 if 0 in c.full_attention else None)
        if input_prefetch_enabled:
            # Layer 0 consumes its input projection like every later layer;
            # the transfer overlaps the embedding and its reduction.
            for transfer in next_input_copies(0 in c.full_attention, 0):
                transfer.start()
        r["proj"][:, :h] = jnp.zeros((rows, h), F32)
        for b in range(batch_size):
            token = token_ref[b]
            start = rank * layout.vocab_shard
            valid = (token >= start) & (token < jnp.minimum(start + layout.vocab_shard, c.vocab))

            @pl.when(valid)
            def embedding():
                read_row(w["emb"], token - start, r["embrow"])
                r["proj"][b, :h] = r["embrow"][...].astype(F32)

        if delivered:

            def embed(rows, columns, total):
                r["residual"][rows, columns] = total.astype(BF16)

            reduce_sum(r["proj"].at[:, :h], 0, embed)
        else:
            r["residual"][...] = reduce_sum(r["proj"].at[:, :h], 0).astype(BF16)

        def linear_attention(layer, ordinal, output=None):
            if gdn_writeback_enabled:
                # There are no outstanding writes on the first GDN layer.
                # Every later layer drains before any inbound buffer reuse.
                @pl.when(ordinal > 0)
                def drain_previous_gdn():
                    drain_gdn_writebacks(jnp.maximum(ordinal - 1, 0), batch_size - 1)

            # HBM layer metadata uses complete eight-row tiles, exactly as
            # read_row above. Dedicated tile destinations avoid sharing any
            # projection scratch while these asynchronous reads are in flight.
            base = pl.multiple_of(ordinal // 8 * 8, 8)
            prefetch = (
                tpu.make_async_copy(w["convw"].at[ordinal], r["cw"], r["gdn_prefetch_sem"].at[0]),
                tpu.make_async_copy(
                    w["alog"].at[pl.ds(base, 8), :],
                    r["gdn_alog_tile"],
                    r["gdn_prefetch_sem"].at[1],
                ),
                tpu.make_async_copy(
                    w["dtb"].at[pl.ds(base, 8), :],
                    r["gdn_dtb_tile"],
                    r["gdn_prefetch_sem"].at[2],
                ),
                tpu.make_async_copy(
                    w["onorm"].at[pl.ds(base, 8), :],
                    r["gdn_onorm_tile"],
                    r["gdn_prefetch_sem"].at[3],
                ),
                tpu.make_async_copy(
                    gdn_state("conv", ordinal, 0), r["cb"], r["gdn_prefetch_sem"].at[4]
                ),
                tpu.make_async_copy(
                    gdn_state("rec", ordinal, 0), r["rec"], r["gdn_prefetch_sem"].at[5]
                ),
            )
            for transfer in prefetch:
                transfer.start()
            if layer_prefetch_enabled:
                output_copy(False, ordinal, output).start()
            if nvfp4:
                input_parameters(ordinal, False)
                r["proj"][...] = jnp.zeros(r["proj"].shape, F32)
                ba_base = conv_width + vl
                if fused_ba and (ba_base % 128 or 2 * vh > 128):
                    raise ValueError("Fused B/A requires a shared aligned N128 tile")

                def ba_projection(resident_columns):
                    matrix = resident_columns(ba_base, 128)
                    fused_ba_projection(
                        r["normed"],
                        w["qkvz"].at[ordinal, :, pl.ds(ba_base, 128)] if matrix is None else matrix,
                        r["proj"].at[:, pl.ds(ba_base, 128)],
                        batch_size=batch_size,
                        channels=vh,
                        b_input_scale=parameter(r["attention_params"], 4),
                        b_weight_scale=parameter(r["attention_params"], 5),
                        a_input_scale=parameter(r["attention_params"], 6),
                        a_weight_scale=parameter(r["attention_params"], 7),
                        resident=matrix is not None,
                    )

                input_projection(
                    r["normed"],
                    w["qkvz"].at[ordinal],
                    r["attention_params"],
                    (
                        (0, 0, conv_width),
                        (1, conv_width, vl),
                        *(
                            ()
                            if fused_ba
                            else (
                                (2, conv_width + vl, vh),
                                (3, conv_width + vl + vh, vh),
                            )
                        ),
                    ),
                    after=ba_projection if fused_ba else None,
                )
            else:
                matmul(r["normed"], w["qkvz"].at[ordinal], r["proj"], rounded=True)
            # Complete all inbound GDN transfers before reading their VMEM.
            # Dedicated writeback transfers below have separately paired waits.
            for transfer in prefetch:
                transfer.wait()
            for name in ("alog", "dtb", "onorm"):
                destination = r[name]
                tile = r["gdn_" + name + "_tile"]
                selected = jnp.sum(
                    jnp.where(jnp.arange(8)[:, None] == ordinal % 8, tile[...].astype(F32), 0),
                    axis=0,
                )
                destination[...] = selected[: destination.shape[0]].astype(destination.dtype)
            r["attended"][...] = jnp.zeros(r["attended"].shape, F32)

            def load_state(name, request):
                # Decode holds one request's state; verification indexes its request.
                return r[name][...] if request is None else r[name][request]

            def store_state(name, request, value):
                if request is None:
                    r[name][...] = value
                else:
                    r[name][request] = value

            def transpose_into(destination, source):
                # Exact change between stored [v, k] and working [k, v] state, by head.
                for head in range(vh):
                    destination[head] = jnp.transpose(source[head].astype(F32)).astype(
                        destination.dtype
                    )

            def tile_row(base, index, columns):
                # Row ``index`` of the aligned eight-row projection tile at ``base``, which
                # a traced group index reaches as a whole tile.
                if isinstance(base, int):
                    return r["proj"][base + index, columns]
                return r["proj"][pl.ds(base, 8), columns][index]

            def columns(vectors):
                # [kh, k] lane vectors -> [kh, k, v], k on sublanes and constant along v.
                return transpose_heads(jnp.broadcast_to(vectors[:, None, :], (kh, c.v_dim, c.k_dim)))

            def norm_gate(out, z):
                # ``out`` holds BF16 readouts with one value head per row; so does ``z``.
                out *= jax.lax.rsqrt(jnp.mean(out * out, axis=-1, keepdims=True) + c.eps)
                # Native fallback rounds normalization and its BF16 weight
                # product before multiplying by the FP32 output gate.
                out = (out.astype(BF16) * r["onorm"][...]).astype(BF16).astype(F32)
                out *= jax.nn.silu(z)
                return out.astype(BF16).astype(F32)

            def gated_tile(base):
                # norm_gate over eight rows at once, one value head at a time.
                rows = pl.ds(base, 8)
                for head in range(vh):
                    lanes = pl.ds(head * c.v_dim, c.v_dim)
                    gate = r["proj"][rows, pl.ds(conv_width + head * c.v_dim, c.v_dim)]
                    r["attended"][rows, lanes] = norm_gate(r["attended"][rows, lanes], gate)

            ba = conv_width + vl

            def decays(b, a):
                # Write strengths and state decays from the B and A projections, by value head.
                beta = jax.nn.sigmoid(b).astype(BF16).astype(F32)
                decay = jnp.exp(
                    -jnp.exp(r["alog"][...]) * jax.nn.softplus(a + r["dtb"][...].astype(F32))
                )
                return beta, decay

            def gdn_row(
                base,
                index,
                request=None,
                local=None,
                conv_done=None,
                rec_done=None,
                gated=True,
                gates=None,
            ):
                # Without ``gated`` the row keeps its BF16 readout for gated_tile; ``gates``
                # are its precomputed decays.
                qkv = tile_row(base, index, slice(0, conv_width)).astype(BF16)
                history = jnp.concatenate((load_state("cb", request), qkv[None, :]), axis=0)
                convolved = jnp.sum(history.astype(F32) * r["cw"][...].astype(F32), axis=0).astype(
                    BF16
                )
                filtered = jax.nn.silu(convolved.astype(F32)).astype(BF16).astype(F32)
                store_state("cb", request, history[1:])
                if conv_done is not None:
                    conv_done()
                q = filtered[:kl].reshape(kh, c.k_dim)
                k = filtered[kl : 2 * kl].reshape(kh, c.k_dim)
                val = filtered[2 * kl :].reshape(vh, c.v_dim)
                q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) * c.k_dim**-0.5
                k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                if gates is None:
                    gates = decays(
                        tile_row(base, index, slice(ba, ba + vh)),
                        tile_row(base, index, slice(ba + vh, ba + 2 * vh)),
                    )
                beta, decay = gates
                query_columns, key_columns = columns(q), columns(k)
                working = r["rec_t"] if local is None else r["rec_t"].at[local]
                readout = []
                # One value head at a time: its state and its key head's columns stay in
                # registers.
                for head in range(vh):
                    group = head // (vh // kh)
                    rec, head_readout = gdn_recurrence_columns(
                        working[head][None],
                        query_columns[group][None],
                        key_columns[group][None],
                        val[head][None],
                        beta[head][None],
                        decay[head][None],
                    )
                    working[head] = rec[0].astype(working.dtype)
                    readout.append(head_readout)
                readout = jnp.concatenate(readout, axis=0)
                if rec_done is not None:
                    rec_done()
                out = readout.astype(BF16).astype(F32)
                if gated:
                    z = tile_row(base, index, slice(conv_width, conv_width + vl))
                    out = norm_gate(out, z.reshape(vh, c.v_dim))
                out = out.reshape(vl)
                if isinstance(base, int):
                    r["attended"][base + index, :vl] = out
                else:
                    tile = r["attended"].at[pl.ds(base, 8), :vl]
                    tile[...] = jnp.where(jnp.arange(8)[:, None] == index, out[None, :], tile[...])

            def gdn_row_pair(base, specs):
                # Two resident requests' rows at one position: each row's chain of norms,
                # key reductions and state updates overlaps the other's. ``specs`` are
                # (tile index, request, local slot, gates, verification row); an
                # inactive row is computed but changes no state.
                alive = [
                    None if active_ref is None else active_ref[row] != 0 for *_, row in specs
                ]

                def kept(live, new, old):
                    return new if live is None else jnp.where(live, new, old)

                histories = []
                for index, request, *_ in specs:
                    qkv = tile_row(base, index, slice(0, conv_width)).astype(BF16)
                    old = load_state("cb", request)
                    histories.append((old, jnp.concatenate((old, qkv[None, :]), axis=0)))
                inputs = []
                for (_, request, _, gates, _), live, (old, history) in zip(
                    specs, alive, histories, strict=True
                ):
                    convolved = jnp.sum(
                        history.astype(F32) * r["cw"][...].astype(F32), axis=0
                    ).astype(BF16)
                    filtered = jax.nn.silu(convolved.astype(F32)).astype(BF16).astype(F32)
                    store_state("cb", request, kept(live, history[1:], old))
                    q = filtered[:kl].reshape(kh, c.k_dim)
                    k = filtered[kl : 2 * kl].reshape(kh, c.k_dim)
                    val = filtered[2 * kl :].reshape(vh, c.v_dim)
                    q *= jax.lax.rsqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6) * c.k_dim**-0.5
                    k *= jax.lax.rsqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6)
                    inputs.append((columns(q), columns(k), val, gates))
                readouts = [[] for _ in specs]
                for head in range(vh):
                    group = head // (vh // kh)
                    for (_, _, local, _, _), live, (qc, kc, val, (beta, decay)), readout in zip(
                        specs, alive, inputs, readouts, strict=True
                    ):
                        working = r["rec_t"].at[local]
                        old = working[head]
                        rec, head_readout = gdn_recurrence_columns(
                            old[None],
                            qc[group][None],
                            kc[group][None],
                            val[head][None],
                            beta[head][None],
                            decay[head][None],
                        )
                        working[head] = kept(live, rec[0].astype(working.dtype), old)
                        readout.append(head_readout)
                outs = [
                    jnp.concatenate(readout, axis=0).astype(BF16).astype(F32).reshape(vl)
                    for readout in readouts
                ]
                if isinstance(base, int):
                    for (index, *_), live, out in zip(specs, alive, outs, strict=True):
                        row = r["attended"].at[base + index, :vl]
                        row[...] = kept(live, out, row[...])
                    return
                tile = r["attended"].at[pl.ds(base, 8), :vl]
                written = tile[...]
                for (index, *_), live, out in zip(specs, alive, outs, strict=True):
                    target = jnp.arange(8)[:, None] == index
                    if live is not None:
                        target &= live
                    written = jnp.where(target, out[None, :], written)
                tile[...] = written

            if resident_requests:

                def group_rows(group):
                    # Positions advance in order; requests' recurrences are independent.
                    first = group * group_requests
                    base = pl.multiple_of(first * causal_rows, 8) if groups > 1 else 0

                    def stage_cut(slot):
                        # A bounded GDN checkpoint, never a historical KV clone.
                        if staged_cuts and groups > 1:

                            @pl.when(group > 0)
                            def reuse():
                                for transfer in cut_writebacks(
                                    ordinal, slot, first - group_requests
                                ):
                                    transfer.wait()

                        for local in range(group_requests):
                            transpose_into(r["cut_rec"].at[slot, local], r["rec_t"].at[local])
                        r["cut_conv"][slot] = r["cb"][pl.ds(first, group_requests)]
                        transfers = cut_writebacks(ordinal, slot, first)
                        for transfer in transfers:
                            transfer.start()
                        if not staged_cuts:
                            for transfer in transfers:
                                transfer.wait()

                    for local in range(group_requests):
                        transpose_into(r["rec_t"].at[local], r["rec"].at[first + local])
                    tiles = range(0, max(8, group_requests * causal_rows), 8)
                    # Decays depend only on the projection, so whole tiles take them at once.
                    tile_decays = {
                        start: decays(
                            r["proj"][pl.ds(base + start, 8), pl.ds(ba, vh)],
                            r["proj"][pl.ds(base + start, 8), pl.ds(ba + vh, vh)],
                        )
                        for start in tiles
                    }
                    for slot in range(causal_rows):
                        specs = []
                        for local in range(group_requests):
                            index = local * causal_rows + slot
                            gates = tuple(
                                gate[index % 8] for gate in tile_decays[index - index % 8]
                            )
                            specs.append(
                                (index, first + local, local, gates, first * causal_rows + index)
                            )
                        for pair in range(0, len(specs) - 1, 2):
                            gdn_row_pair(base, specs[pair : pair + 2])
                        if len(specs) % 2:
                            index, request, local, gates, row = specs[-1]
                            when_active(row)(
                                lambda: gdn_row(
                                    base, index, request, local, gated=False, gates=gates
                                )
                            )
                        if slot < cut_count:
                            stage_cut(slot)
                    for start in tiles:
                        gated_tile(base + start)
                    for local in range(group_requests):
                        transpose_into(r["rec"].at[first + local], r["rec_t"].at[local])

                if groups > 1:
                    pl.loop(0, groups)(group_rows)
                else:
                    group_rows(0)
                if gdn_writeback_enabled:
                    for transfer in gdn_writebacks(ordinal, 0):
                        transfer.start()
                else:
                    copy(r["cb"], gdn_state("conv", ordinal, 0))
                    copy(r["rec"], gdn_state("rec", ordinal, 0))
            else:
                for b in range(batch_size):
                    # Request 0 is already resident. Before any later request
                    # reuses either source, complete both preceding state stores.
                    if b:
                        if gdn_writeback_enabled:
                            drain_gdn_writebacks(ordinal, b - 1)
                        copy(state_out["conv"].at[ordinal, b], r["cb"])
                        copy(state_out["rec"].at[ordinal, b], r["rec"])
                    transpose_into(r["rec_t"], r["rec"])

                    def conv_done(b=b):
                        if gdn_writeback_enabled:
                            gdn_writebacks(ordinal, b)[0].start()
                        else:
                            copy(r["cb"], state_out["conv"].at[ordinal, b])

                    def rec_done(b=b):
                        transpose_into(r["rec"], r["rec_t"])
                        if gdn_writeback_enabled:
                            gdn_writebacks(ordinal, b)[1].start()
                        else:
                            copy(r["rec"], state_out["rec"].at[ordinal, b])

                    gdn_row(b // 8 * 8, b % 8, conv_done=conv_done, rec_done=rec_done)
            if nvfp4:
                output_projection(False, ordinal, 8, output)
            else:
                matmul(r["attended"], w["outw"].at[ordinal], r["proj"])
            capture(layer, "attention_partial", r["proj"][:, :h])
            if not shared_weights_prefetch_enabled:
                if router_prefetch:
                    # The weight is independent of the residual being reduced.
                    # Start only after the output projection's scoped DMAs finish.
                    tpu.make_async_copy(
                        w["router"].at[layer], r["router_weight"], r["router_sem"].at[0]
                    ).start()
                residual_add(2 * layer + 1)
                capture(layer, "residual_attention", r["residual"][...])

        def rms_head(x, norm):
            value = x.astype(F32)
            value *= jax.lax.rsqrt(jnp.mean(value * value, axis=-1, keepdims=True) + c.eps)
            return (value * (1 + norm.astype(F32))).astype(BF16)

        def rope(x, pos):
            half = c.rotary_dim // 2
            frequency = c.rope_theta ** (
                -jnp.arange(half, dtype=jnp.int32).astype(F32) * 2 / c.rotary_dim
            )
            angles = pos.astype(F32) * frequency
            co, si = jnp.cos(angles).astype(BF16), jnp.sin(angles).astype(BF16)
            a, b = x[..., :half], x[..., half : c.rotary_dim]
            return jnp.concatenate(
                (a * co - b * si, b * co + a * si, x[..., c.rotary_dim :]), axis=-1
            ).astype(BF16)

        def full_attention(layer, ordinal, output=None):
            if layer_prefetch_enabled:
                output_copy(True, ordinal, output).start()
            if nvfp4:
                input_parameters(ordinal, True)
                input_projection(
                    r["normed"],
                    w["qwkv"].at[ordinal],
                    r["attention_params"],
                    (
                        (0, 0, 2 * qh * c.head_dim),
                        (1, 2 * qh * c.head_dim, kvh * c.head_dim),
                        (2, (2 * qh + kvh) * c.head_dim, kvh * c.head_dim),
                    ),
                )
            else:
                matmul(r["normed"], w["qwkv"].at[ordinal], r["proj"], rounded=True)
            layer_row("qn", ordinal, r["qnorm"])
            layer_row("kn", ordinal, r["knorm"])
            r["attended"][...] = jnp.zeros(r["attended"].shape, F32)

            groups = qh // kvh
            # Calibration is a compile-time contract supplied by the prefill
            # bridge. Select the layer's pair without adding mutable state.
            # Pallas disallows captured array constants; scalar literals are
            # embedded directly in the program and selected by the loop ordinal.
            key_scale, value_scale = jnp.float32(1), jnp.float32(1)
            for scale_index, (ks, vs) in enumerate(kv_scales):
                key_scale = jnp.where(ordinal == scale_index, jnp.float32(ks), key_scale)
                value_scale = jnp.where(ordinal == scale_index, jnp.float32(vs), value_scale)
            scale_pair = (key_scale, value_scale)
            if attention_optimized and (causal_pair or batch_size > 1):
                batch_queries, batch_keys, batch_values, batch_gates = [], [], [], []
                for b in range(batch_size):
                    pos = position_ref[b]
                    packed = (
                        r["proj"][b, : 2 * qh * c.head_dim].reshape(qh, 2 * c.head_dim).astype(BF16)
                    )
                    query = rope(rms_head(packed[:, : c.head_dim], r["qnorm"][...]), pos)
                    gate = jax.nn.sigmoid(packed[:, c.head_dim :].astype(F32)).astype(BF16)
                    offset = 2 * qh * c.head_dim
                    key = (
                        r["proj"][b, offset : offset + kvh * c.head_dim]
                        .reshape(kvh, c.head_dim)
                        .astype(BF16)
                    )
                    key = rope(rms_head(key, r["knorm"][...]), pos)
                    value = (
                        r["proj"][b, offset + kvh * c.head_dim : offset + 2 * kvh * c.head_dim]
                        .reshape(kvh, c.head_dim)
                        .astype(BF16)
                    )
                    batch_queries.append(query)
                    batch_keys.append(key[0])
                    batch_values.append(value[0])
                    batch_gates.append(gate)
                if causal_pair and causal_rows == 2:
                    attended = attend_paged_causal_pair(
                        jnp.stack(batch_queries), jnp.stack(batch_keys), jnp.stack(batch_values),
                        state_out["kcache"].at[ordinal, 0, 0], state_out["vcache"].at[ordinal, 0, 0],
                        position_ref, rank, layout, scale_pair, id_type,
                        barriers=r["attention_barriers"],
                        _row_update=_mtp_causal_row_update,
                    )
                elif causal_pair:
                    attended = attend_paged_causal_four(
                        jnp.stack(batch_queries),
                        jnp.stack(batch_keys),
                        jnp.stack(batch_values),
                        state_out["kcache"].at[ordinal, :, 0],
                        state_out["vcache"].at[ordinal, :, 0],
                        tuple(position_ref[b] for b in range(batch_size)),
                        rank,
                        layout,
                        scale_pair,
                        id_type,
                        inbox,
                        active=None
                        if active_ref is None
                        else tuple(active_ref[b] != 0 for b in range(batch_size)),
                        buffers=KV_BUFFERS,
                    )
                else:
                    if paged is not None:
                        publish_columns(ordinal, scale_pair, batch_keys, batch_values)
                    attended = attend_paged_batched(
                        jnp.stack(batch_queries),
                        jnp.stack(batch_keys),
                        jnp.stack(batch_values),
                        pool_pages(ordinal, 0) if paged is not None else state_out["kcache"].at[ordinal, :, 0],
                        None if paged is not None else state_out["vcache"].at[ordinal, :, 0],
                        position_ref,
                        rank,
                        layout,
                        scale_pair,
                        id_type,
                        inbox,
                        buffers=KV_BUFFERS,
                    )
                for b in range(batch_size):
                    result = attended[b] * batch_gates[b]
                    if active_ref is not None:
                        result = jnp.where(active_ref[b] != 0, result, 0)
                    r["attended"][b, : groups * c.head_dim] = (
                        result.astype(BF16).astype(F32).reshape(-1)
                    )
            else:
                for b in range(batch_size):
                    pos = position_ref[b]
                    packed = (
                        r["proj"][b, : 2 * qh * c.head_dim].reshape(qh, 2 * c.head_dim).astype(BF16)
                    )
                    query = rope(rms_head(packed[:, : c.head_dim], r["qnorm"][...]), pos)
                    gate = jax.nn.sigmoid(packed[:, c.head_dim :].astype(F32)).astype(BF16)
                    offset = 2 * qh * c.head_dim
                    key = (
                        r["proj"][b, offset : offset + kvh * c.head_dim]
                        .reshape(kvh, c.head_dim)
                        .astype(BF16)
                    )
                    key = rope(rms_head(key, r["knorm"][...]), pos)
                    value = (
                        r["proj"][b, offset + kvh * c.head_dim : offset + 2 * kvh * c.head_dim]
                        .reshape(kvh, c.head_dim)
                        .astype(BF16)
                    )
                    if paged is not None:
                        publish_columns(ordinal, scale_pair, [key[0]], [value[0]])
                    for g in range(kvh):
                        result = single_attention(
                            query[g * groups : (g + 1) * groups],
                            key[g],
                            value[g],
                            pool_pages(ordinal, b) if paged is not None else state_out["kcache"].at[ordinal, b, g],
                            None if paged is not None else state_out["vcache"].at[ordinal, b, g],
                            pos,
                            rank,
                            layout,
                            scale_pair,
                            id_type,
                            **(
                                {"inbox": inbox, "buffers": KV_BUFFERS}
                                if inbox
                                else {"barriers": r["attention_barriers"]}
                            ),
                        )
                        result = (result * gate[g * groups : (g + 1) * groups]).astype(BF16)
                        r["attended"][b, pl.ds(g * groups * c.head_dim, groups * c.head_dim)] = (
                            result.astype(F32).reshape(-1)
                        )
            if nvfp4:
                output_projection(True, ordinal, 6, output)
            else:
                matmul(r["attended"], w["ow"].at[ordinal], r["proj"])
            capture(layer, "attention_partial", r["proj"][:, :h])
            if not shared_weights_prefetch_enabled:
                if router_prefetch:
                    # The weight is independent of the residual being reduced.
                    # Start only after the output projection's scoped DMAs finish.
                    tpu.make_async_copy(
                        w["router"].at[layer], r["router_weight"], r["router_sem"].at[0]
                    ).start()
                residual_add(2 * layer + 1)
                capture(layer, "residual_attention", r["residual"][...])

        def shared_gate(layer):
            layer_row("shared_gate", layer, r["norm"])
            shared_logit = (
                jnp.concatenate([
                    jnp.sum(r["normed"][tile].astype(F32) * r["norm"][...].astype(F32), axis=1)
                    for tile in eight_rows
                ])
                .astype(BF16)
                .astype(F32)
            )
            capture(layer, "shared_gate_logit", shared_logit[:, None])
            shared_probability = jax.nn.sigmoid(shared_logit).astype(BF16).astype(F32)
            capture(layer, "shared_multiplier", shared_probability[:, None])
            return shared_probability

        def route(layer, on_selection=None):
            if router_prefetch:
                # Pair with the start before this layer's attention reduction.
                # This private buffer/semaphore is never reused while DMA is
                # outstanding; its contents survive post-attention RMSNorm.
                tpu.make_async_copy(
                    w["router"].at[layer], r["router_weight"], r["router_sem"].at[0]
                ).wait()
                activation = r["normed"][:, :h].astype(BF16)
                with jax.named_scope("router_prefetched_projection"):
                    for column in range(0, rw, 128):
                        value = jnp.dot(
                            activation,
                            r["router_weight"][:, pl.ds(column, 128)].astype(BF16),
                            preferred_element_type=F32,
                        )
                        # Identical full-K/N128 contraction and BF16 midpoint
                        # rounding to dense_matmul(..., rounded=True).
                        r["proj"][:, pl.ds(column, 128)] = value.astype(BF16).astype(F32)
            else:
                matmul(r["normed"], w["router"].at[layer], r["proj"], rounded=True)
            r["router_parts"][0, ...] = r["proj"][:, :rw].astype(BF16)
            for peer in range(1, rt):
                tpu.make_async_remote_copy(
                    r["router_parts"].at[0],
                    r["router_parts"].at[peer],
                    r["small_sends"].at[peer - 1],
                    r["small_recvs"].at[peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                ).start()
            if shared_gate_overlap:
                # Complete shared-gate work while the router messages are in
                # flight. No router source/inbox or semaphore is touched here.
                shared_probability = shared_gate(layer)
            for peer in range(1, rt):
                tpu.make_async_remote_copy(
                    r["router_parts"].at[0],
                    r["router_parts"].at[peer],
                    r["small_sends"].at[peer - 1],
                    r["small_recvs"].at[peer - 1],
                    device_id=peer_id(rank ^ peer),
                    device_id_type=id_type,
                ).wait()
            for peer in range(rt):
                column = ((rank % rt) ^ peer) * rw
                r["router"][:, pl.ds(column, rw)] = r["router_parts"][peer, ...].astype(F32)
            capture(layer, "router_logits", r["router"][...])
            logits = r["router"][...]
            for tile in eight_rows:
                r["router"][tile] = jax.nn.softmax(r["router"][tile], axis=-1)
            indices = jnp.arange(c.experts)[None, :]
            selected, chosen, disagree = top_k_by_logit(
                logits, r["router"][...], c.top_k, on_selection
            )
            for k in range(c.top_k):
                r["route_ids"][:, k] = selected[k][:, 0]
                r["route_probs"][:, k] = chosen[k][:, 0]

            # Early expert transfers keep the key order; the expert loop weights each
            # started expert by its reselected probability (zero if no longer routed).
            @pl.when(jnp.max(jnp.where(disagree, 1, 0)) > 0)
            def select_by_probability():
                for k in range(c.top_k):
                    best = jnp.max(r["router"][...], axis=-1, keepdims=True)
                    expert = jnp.min(
                        jnp.where(r["router"][...] == best, indices, c.experts), axis=-1
                    )
                    r["route_ids"][:, k] = expert
                    r["route_probs"][:, k] = best[:, 0]
                    r["router"][...] = jnp.where(
                        indices == expert[:, None], -jnp.inf, r["router"][...]
                    )

            for tile in eight_rows:
                probs = r["route_probs"][tile]
                r["route_probs"][tile] = (
                    (probs / jnp.maximum(jnp.sum(probs, axis=1, keepdims=True), 1e-20))
                    .astype(BF16)
                    .astype(F32)
                )

            return shared_probability if shared_gate_overlap else None

        def shared_expert(layer, gu=None, down=None, sem=None, on_selection=None):
            shared_probability = route(layer, on_selection)
            capture(layer, "route_ids", r["route_ids"][...])
            capture(layer, "route_probabilities", r["route_probs"][...])
            # Shared FFN is TP32; its partial is included once in the global reduction.
            if not shared_gate_overlap:
                shared_probability = shared_gate(layer)
            if gu is None:
                matmul(r["normed"], w["gu"].at[layer], r["gateup"], rounded=True)
            else:
                with jax.named_scope("shared_gu_prefetch_wait"):
                    tpu.make_async_copy(w["gu"].at[layer], gu, sem.at[0]).wait()
                resident_shared_projection(r["normed"], gu, r["gateup"], rounded=True)
            capture(layer, "shared_gateup", r["gateup"][:, : 2 * sw])
            if sw % 128 == 0:
                gate, up = r["gateup"][:, :sw], r["gateup"][:, sw : 2 * sw]
            else:
                # Gate and up shards share one output tile; rotate up onto gate's lanes.
                tile = r["gateup"][:, :128]
                gate, up = tile[:, :sw], tpu.roll(tile, 128 - sw, 1)[:, :sw]
            r["middle"][:, :sw] = (jax.nn.silu(gate).astype(BF16) * up.astype(BF16)).astype(BF16)
            capture(layer, "shared_activation", r["middle"][:, :sw])
            if down is None:
                matmul(r["middle"], w["dw"].at[layer], moe)
            else:
                with jax.named_scope("shared_down_prefetch_wait"):
                    tpu.make_async_copy(w["dw"].at[layer], down, sem.at[1]).wait()
                resident_shared_projection(r["middle"], down, moe, rounded=False)
            moe[...] *= shared_probability[:, None]
            capture(layer, "shared_partial", moe[...])

        def shared_weight_phase(layer):
            # The original O scope has returned. Preserve the original router
            # start first; no shared buffer overlaps attention streaming or O.
            if router_prefetch:
                tpu.make_async_copy(
                    w["router"].at[layer], r["router_weight"], r["router_sem"].at[0]
                ).start()

            def compute(gu, down, sem, experts=None):
                with jax.named_scope("shared_weights_prefetch_start"):
                    tpu.make_async_copy(w["gu"].at[layer], gu, sem.at[0]).start()
                    tpu.make_async_copy(w["dw"].at[layer], down, sem.at[1]).start()
                residual_add(2 * layer + 1)
                capture(layer, "residual_attention", r["residual"][...])
                normalize("norm_post", layer)
                capture(layer, "norm_post", r["normed"][...])
                if experts is None:
                    shared_expert(layer, gu, down, sem)
                    return
                # Routed-expert scratch opens before routing so the first local
                # expert's transfers overlap top-k and the shared expert.
                live = live_rows()
                early = EarlyStart(
                    routed_weights(layer), experts, rank, layout, batch_size, live=live
                )
                shared_expert(layer, gu, down, sem, on_selection=early)
                with jax.named_scope("routed_experts"):
                    run_experts(
                        routed_weights(layer),
                        r["normed"],
                        r["route_ids"],
                        r["route_probs"],
                        moe,
                        rank,
                        layout,
                        batch_size,
                        id_type=id_type,
                        scratch=experts,
                        started=early.first,
                        live=live,
                    )

            # Both transfers are waited by shared_expert before scope exit.
            # Pass scoped Refs explicitly; no private Ref closes over a loop.
            pl.run_scoped(
                compute,
                gu=v((8192, 128)),
                down=v((64, 8192)),
                sem=dm(2),
                **(
                    {"experts": expert_scratch_shapes(layout, batch_size, rows)}
                    if early_experts
                    else {}
                ),
            )

        def routed_weights(layer):
            return {name: w[name].at[layer] for name in ("egu", "egu_scale", "edw", "edw_scale")}

        def mixture_of_experts(layer, successor=None, *, shared_ready=False, routed_ready=False):
            if not shared_ready:
                shared_expert(layer)

            def routed_experts(egu, egus, edw, edws, expert_sem, eparams=None):
                start = rank // layout.expert_tp * layout.local_experts
                live = live_rows()
                active = jnp.arange(rows)[:, None] < batch_size if live is None else live
                valid = (
                    active
                    & (r["route_ids"][...] >= start)
                    & (r["route_ids"][...] < start + layout.local_experts)
                )
                r["route_ids"][...] = jnp.where(valid, r["route_ids"][...], c.experts)

                def next_expert():
                    return jnp.min(r["route_ids"][...])

                def expert_copies(expert, slot):
                    local = expert - start
                    count = 5 if nvfp4 else 4
                    copies = (
                        tpu.make_async_copy(
                            w["egu"].at[layer, local],
                            egu.at[slot],
                            expert_sem.at[slot * count],
                        ),
                        tpu.make_async_copy(
                            w["egu_scale"].at[layer, local],
                            egus.at[slot],
                            expert_sem.at[slot * count + 1],
                        ),
                        tpu.make_async_copy(
                            w["edw"].at[layer, local],
                            edw.at[slot],
                            expert_sem.at[slot * count + 2],
                        ),
                        tpu.make_async_copy(
                            w["edw_scale"].at[layer, local],
                            edws.at[slot],
                            expert_sem.at[slot * count + 3],
                        ),
                    )
                    if nvfp4:
                        copies += (
                            tpu.make_async_copy(
                                w["eparams"].at[layer, local],
                                eparams.at[slot],
                                expert_sem.at[slot * count + 4],
                            ),
                        )
                    return copies

                first = next_expert()

                @pl.when(first < c.experts)
                def prefetch_first():
                    for dma in expert_copies(first, 0):
                        dma.start()

                def work(carry):
                    expert, slot = carry
                    for dma in expert_copies(expert, slot):
                        dma.wait()
                    selected = r["route_ids"][...] == expert
                    probability = jnp.sum(jnp.where(selected, r["route_probs"][...], 0), axis=-1)
                    r["route_ids"][...] = jnp.where(selected, c.experts, r["route_ids"][...])
                    following = next_expert()

                    @pl.when(following < c.experts)
                    def prefetch_next():
                        for dma in expert_copies(following, 1 - slot):
                            dma.start()

                    if nvfp4:
                        for component in range(2):
                            offset = component * ew
                            nvfp4_dot(
                                r["normed"],
                                egu.at[slot, :, pl.ds(offset, ew)],
                                egus.at[slot, :, pl.ds(offset, ew)],
                                parameter(eparams.at[slot], component * 2),
                                parameter(eparams.at[slot], component * 2 + 1),
                                r["gateup"].at[:, pl.ds(offset, ew)],
                            )
                    else:
                        fp8_block_dot(
                            r["normed"],
                            egu.at[slot],
                            egus.at[slot],
                            r["gateup"].at[:, : 2 * ew],
                        )
                    gate = r["gateup"][:, :ew].astype(BF16).astype(F32)
                    up = r["gateup"][:, ew : 2 * ew].astype(BF16)
                    r["middle"][:, :ew] = (jax.nn.silu(gate).astype(BF16) * up).astype(BF16)
                    if nvfp4:
                        nvfp4_dot(
                            r["middle"].at[:, :ew],
                            edw.at[slot],
                            edws.at[slot],
                            parameter(eparams.at[slot], 4),
                            parameter(eparams.at[slot], 5),
                            r["proj"].at[:, :h],
                        )
                    else:
                        fp8_block_dot(
                            r["middle"].at[:, :ew],
                            edw.at[slot],
                            edws.at[slot],
                            r["proj"].at[:, :h],
                        )
                    moe[...] += r["proj"][:, :h] * probability[:, None]
                    return following, 1 - slot

                jax.lax.while_loop(lambda carry: carry[0] < c.experts, work, (first, jnp.int32(0)))

            # Each prefetch is waited at the next iteration's entry; the last
            # iteration starts no successor DMA, so scope exit owns no transfer.
            if not routed_ready:
                with jax.named_scope("routed_experts"):
                    if layout.fast:
                        run_experts(
                            routed_weights(layer),
                            r["normed"],
                            r["route_ids"],
                            r["route_probs"],
                            moe,
                            rank,
                            layout,
                            batch_size,
                            id_type=id_type,
                            live=live_rows(),
                        )
                    else:
                        pl.run_scoped(routed_experts, **expert_scratch)
            if input_prefetch_enabled and successor is not None:
                successor_full, successor_ordinal, successor_valid = successor

                def start_successor(full):
                    with jax.named_scope("next_input_prefetch_start"):
                        for transfer in next_input_copies(full, successor_ordinal):
                            transfer.start()
                        if layer_prefetch_enabled:
                            start_rows(layer + 1, successor_ordinal if full else None)

                if isinstance(successor_full, bool):
                    pl.when(successor_valid)(lambda: start_successor(successor_full))
                else:
                    pl.when(successor_valid & successor_full)(lambda: start_successor(True))
                    pl.when(successor_valid & ~successor_full)(lambda: start_successor(False))

            capture(layer, "moe_partial", moe[...])
            if not moe_in_proj:
                r["proj"][:, :h] = moe[...]
            residual_add(2 * layer + 2)
            capture(layer, "residual_moe", r["residual"][...])

        def layer(index, ordinal, full, successor=None):
            normalize("norm_in", index)
            capture(index, "norm_in", r["normed"][...])

            def attention(full):
                def scoped(output_weight=None, output_sem=None):
                    output = (output_weight, output_sem)
                    if full:
                        with jax.named_scope("full_attention_layer"):
                            full_attention(index, ordinal, output)
                    else:
                        with jax.named_scope("linear_attention_layer"):
                            linear_attention(index, ordinal, output)

                if layer_prefetch_enabled:
                    # The output weight's transfer and use both fall within the attention
                    # phase, so its buffer is free during the MoE.
                    pl.run_scoped(
                        scoped,
                        output_weight=v((vl, h), weights["outw"].dtype),
                        output_sem=dm(1),
                    )
                else:
                    scoped()

            if isinstance(full, bool):
                attention(full)
            else:
                pl.when(full)(lambda: attention(True))
                pl.when(~full)(lambda: attention(False))
            if shared_weights_prefetch_enabled:
                shared_weight_phase(index)
                # Dedicated6MiB storage is dead before run_experts allocates
                # its own scratch (or, with early experts, holds both scopes)
                # and before successor input prefetch starts.
                with jax.named_scope("moe_layer"):
                    mixture_of_experts(
                        index, successor, shared_ready=True, routed_ready=early_experts
                    )
            else:
                normalize("norm_post", index)
                capture(index, "norm_post", r["normed"][...])
                with jax.named_scope("moe_layer"):
                    mixture_of_experts(index, successor)

        repeating = c.layers % 4 == 0 and c.full_attention == tuple(range(3, c.layers, 4))
        if repeating and rows == 16 and causal_pair:
            # Sixteen verification rows keep four unrolled layers per iteration; plain
            # sixteen-request decode fits instruction memory only with one.

            @pl.loop(0, c.layers // 4)
            def block(i):
                for j in range(3):
                    successor = (False, i * 3 + j + 1, True) if j < 2 else (True, i, True)
                    layer(i * 4 + j, i * 3 + j, False, successor)
                layer(i * 4 + 3, i, True, (False, (i + 1) * 3, i + 1 < c.layers // 4))
        elif repeating:

            def ordinal_of(index):
                # Every fourth layer is full attention; ordinals count each kind.
                full = index % 4 == 3
                return full, jnp.where(full, index // 4, index - index // 4)

            # One layer per iteration keeps a single copy of each layer kind and of
            # the MoE in the loop body.
            @pl.loop(0, c.layers)
            def one_layer(index):
                full, ordinal = ordinal_of(index)
                following = ordinal_of(index + 1)
                layer(index, ordinal, full, (*following, index + 1 < c.layers))
        else:
            linear_ord = full_ord = 0
            for i in range(c.layers):
                is_full = i in c.full_attention
                successor = None
                if i + 1 < c.layers:
                    next_full = i + 1 in c.full_attention
                    ordinals = c.full_layers if next_full else c.linear_layers
                    successor = (next_full, ordinals.index(i + 1), True)
                layer(i, full_ord if is_full else linear_ord, is_full, successor)
                full_ord += int(is_full)
                linear_ord += int(not is_full)

        drain_sum(2 * c.layers)
        if gdn_writeback_enabled and c.linear_layers:
            # Return/donation may not expose state until both final stores end.
            drain_gdn_writebacks(len(c.linear_layers) - 1, batch_size - 1)
        normalize()
        if return_hidden:
            # MTP consumes the exact post-final-norm BF16 operand used by LM head.
            # copy() waits for completion; no recurrent/cache DMA remains pending.
            copy(r["normed"], hidden_out)
        matmul(r["normed"], w["lmw"], r["proj"])
        ids = rank * layout.vocab_shard + jnp.arange(lm_width)
        valid = (jnp.arange(lm_width) < layout.vocab_shard) & (ids < c.vocab)
        r["candidates"][0, ...] = jnp.zeros((rows, 128), F32)

        # One row tile per iteration keeps a single tile's logits live.
        @pl.loop(0, rows // 8)
        def logit_tile(index):
            tile = pl.ds(pl.multiple_of(index * 8, 8), 8)
            logits = jnp.where(valid, r["proj"][tile, :lm_width], -jnp.inf)
            r["proj"][tile, :lm_width] = logits
            maximum = jnp.max(logits, axis=-1)
            token = jnp.min(jnp.where(logits == maximum[:, None], ids[None, :], c.vocab), axis=-1)
            r["candidates"][0, tile, 0] = maximum
            r["candidates"][0, tile, 1] = token.astype(F32)
        if logits_in_hbm:
            copy(r["proj"].at[:, pl.ds(0, lm_width)], logits_out)
        else:
            logits_out[...] = r["proj"][:batch_size, :lm_width]
        for peer in range(1, tp):
            tpu.make_async_remote_copy(
                r["candidates"].at[0],
                r["candidates"].at[peer],
                r["small_sends"].at[peer - 1],
                r["small_recvs"].at[peer - 1],
                device_id=peer_id(rank ^ peer),
                device_id_type=id_type,
            ).start()
        for peer in range(1, tp):
            tpu.make_async_remote_copy(
                r["candidates"].at[0],
                r["candidates"].at[peer],
                r["small_sends"].at[peer - 1],
                r["small_recvs"].at[peer - 1],
                device_id=peer_id(rank ^ peer),
                device_id_type=id_type,
            ).wait()
        values = r["candidates"][:, :, 0]
        best = jnp.max(values, axis=0)
        chosen = jnp.min(
            jnp.where(values == best[None, :], r["candidates"][:, :, 1], c.vocab), axis=0
        )
        next_out[...] = chosen[:batch_size].astype(jnp.int32)

    def shape(x):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)

    args = (
        positions,
        tokens,
        *(() if _mtp_active is None else (_mtp_active.astype(jnp.int32),)),
        *(() if paged is None else (paged["tables"].astype(jnp.int32),)),
        *(weights[n] for n in weight_names),
        *(states[n] for n in state_names),
        *(() if paged is None else paged["pools"]),
    )
    state_offset = scalar_count + len(weight_names)
    outputs = (
        jax.ShapeDtypeStruct((batch_size, lm_width), F32),
        jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        *(shape(states[n]) for n in state_names),
    )
    if paged is not None:
        outputs += (
            jax.ShapeDtypeStruct(
                (len(c.full_layers), batch_size, 2, c.head_dim), jnp.float8_e4m3fn
            ),
        )
    if trace:
        outputs += (jax.ShapeDtypeStruct((c.layers, len(STAGES), rows, trace_width), F32),)
    if return_hidden:
        outputs += (jax.ShapeDtypeStruct((rows, h), BF16),)
    if causal_pair and separate_cuts:
        outputs += tuple(shape(states[n]) for _ in range(cut_count) for n in ("conv", "rec"))
    elif causal_pair:
        outputs += tuple(
            jax.ShapeDtypeStruct((cut_count, *states[n].shape), states[n].dtype)
            for n in ("conv", "rec")
        )
    if _mtp_state_owner_axis:
        if not return_hidden or trace:
            raise ValueError("Owner-axis state is an internal MTP hidden-output mode")
        owner_ranks = {"conv": 5, "rec": 6, "kcache": 7, "vcache": 7}
        if any(value.ndim != owner_ranks[name] or value.shape[0] != 1 for name, value in states.items()):
            raise ValueError("MTP state must retain one local owner axis")

        def owner_kernel(*refs):
            # Keep full-shape HBM alias operands at the call boundary. Only Ref
            # views lose the singleton owner axis; kernel arithmetic is unchanged.
            refs = list(refs)
            for index in range(state_offset, state_offset + 4):
                refs[index] = refs[index].at[0]
            for index in range(len(args) + 2, len(args) + 6):
                refs[index] = refs[index].at[0]
            if causal_pair:
                begin = len(args) + 6 + int(trace) + int(return_hidden)
                for index in range(begin, begin + cut_outputs):
                    refs[index] = refs[index].at[0]
            kernel(*refs)

    result = pl.pallas_call(
        owner_kernel if _mtp_state_owner_axis else kernel,
        out_shape=outputs,
        input_output_aliases={state_offset + i: 2 + i for i in range(len(state_names))},
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=scalar_count,
            in_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * (len(args) - scalar_count),
            out_specs=[
                pl.BlockSpec(memory_space=tpu.HBM) if logits_in_hbm else pl.BlockSpec(),
                pl.BlockSpec(),
            ]
            + [pl.BlockSpec(memory_space=tpu.HBM)]
            * (len(state_names) + int(paged is not None) + int(trace) + int(return_hidden) + cut_outputs),
            scratch_shapes=tuple(scratch.values()),
        ),
        compiler_params=tpu.CompilerParams(
            collective_id=37, vmem_limit_bytes=64 * 1024**2, shape_invariant_numerics=True
        ),
        interpret=interpret,
        name=f"qwen_{c.quantization}_{layout.precision}_cp{layout.context_parallel}_decode",
    )(*args)
    if paged is not None:
        ns = len(state_names)
        updated = dict(zip(state_names, result[2 : 2 + ns], strict=True))
        return result[1], result[0], {**updated, "columns": result[2 + ns]}
    base = (result[1], result[0], dict(zip(STATE_NAMES, result[2:6], strict=True)))
    if return_hidden:
        extra = ((result[6],) if trace else ()) + (result[6 + int(trace)][:batch_size],)
        if causal_pair:
            begin = 6 + int(trace) + int(return_hidden)
            if separate_cuts and cut_count > 1:
                extra += ({
                    name: tuple(result[begin + 2 * slot + offset] for slot in range(cut_count))
                    for offset, name in enumerate(("conv", "rec"))
                },)
            else:
                extra += (dict(zip(("conv", "rec"), result[begin:begin + 2], strict=True)),)
        return (*base, *extra)
    return (*base, result[6]) if trace else base


def paged_decode_local(
    config, layout, weights, states, pools, tables, tokens, positions, *, pages_per_block, kv_scales=()
):
    """One decode step over vLLM's pools, inside ``shard_map`` (rank-local
    operands): the fused decoder reads every request's pages through
    ``tables`` (``[batch, width]`` rank-local blocks of ``pages_per_block``
    kernel pages), then ``paged_kv.commit_columns`` writes the new key/value
    columns. ``states`` are the requests' dense ``conv``/``rec``. Returns the
    next tokens, rank-sharded logits, updated states and pools."""
    from .paged_kv import commit_columns

    pools = tuple(pools)
    nxt, logits, states = _stack(
        config,
        layout,
        0,
        tokens.shape[0],
        weights,
        states,
        tokens,
        positions,
        False,
        kv_scales=kv_scales,
        paged={"pools": pools, "tables": tables, "pages_per_block": pages_per_block},
    )[:3]
    columns = states.pop("columns")
    pools = commit_columns(pools, columns, positions, tables, pages_per_block=pages_per_block)
    return nxt, logits, states, pools


def make_decode(
    mesh,
    config,
    context,
    batch_size,
    expert_tp=4,
    donate=True,
    interpret=False,
    trace=False,
    *,
    layout=None,
    context_parallel=1,
    precision="native",
    kv_page_size=128,
    kv_scales=None,
    return_hidden=False,
    _mtp_state_owner_axis=False,
):
    """Build decode; optional return_hidden appends post-final-norm BF16 [B,H].

    The default return tuple and execution remain unchanged. With trace=True,
    hidden is appended after the trace. MTP callers use the exact normalized
    hidden operand, never reconstruct it from logits or pre-normalized residuals.
    """
    if _mtp_state_owner_axis and (not return_hidden or trace):
        raise ValueError("Owner-axis state is an internal MTP hidden-output mode")
    tp = mesh.size
    layout = layout or Layout(
        config,
        tp=tp,
        expert_tp=expert_tp,
        router_tp=min(4, tp),
        context_parallel=context_parallel,
        precision=precision,
        kv_page_size=kv_page_size,
    )
    if layout.config != config or layout.tp != tp:
        raise ValueError("Explicit layout must match the model and mesh")
    kv_scales = validate_kv_scales(layout, kv_scales)
    if batch_size not in (1, 2, 4, 8, 16) or context <= 0 or context % 128:
        raise ValueError("Batch must be 1/2/4/8/16 and context a positive multiple of 128")
    if config.head_dim % 128 or config.k_dim % 128 or config.v_dim % 128:
        raise ValueError("Head dimensions must be multiples of 128")
    if (config.experts // layout.router_tp) % 128:
        raise ValueError("Router output shards must be multiples of 128")

    def local(w, s, token, pos):
        local_weights = jax.tree.map(lambda x: x[0], w)
        local_states = s if _mtp_state_owner_axis else jax.tree.map(lambda x: x[0], s)
        result = _stack(
            config,
            layout,
            context,
            batch_size,
            local_weights,
            local_states,
            token,
            pos,
            interpret,
            trace,
            kv_scales,
            return_hidden,
            _mtp_state_owner_axis=_mtp_state_owner_axis,
        )
        nxt, logits, state = result[:3]
        base = (nxt, logits[None], state if _mtp_state_owner_axis else jax.tree.map(lambda x: x[None], state))
        if return_hidden:
            extra = ((result[3][None],) if trace else ()) + (result[3 + int(trace)],)
            return (*base, *extra)
        return (*base, result[3][None]) if trace else base

    def decode(weights, states, tokens, positions):
        if tokens.shape != (batch_size,) or positions.shape != (batch_size,):
            raise ValueError("tokens and positions must have shape [batch_size]")
        ws, ss = jax.tree.map(lambda _: P("tp"), weights), jax.tree.map(lambda _: P("tp"), states)
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(ws, ss, P(), P()),
            out_specs=(P(), P("tp"), ss) + ((P("tp"),) if trace else ())
            + ((P(),) if return_hidden else ()),
            check_vma=False,
        )(weights, states, tokens, positions)

    return jax.jit(decode, donate_argnums=(1,) if donate else ())
