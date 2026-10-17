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

"""Core Pallas kernel implementations for Batched Ragged Paged Attention.

Orchestrates asynchronous VMEM DMA pipelines for paged KV cache gather, online
softmax accumulation, and output writeback across decode and mixed batches.
"""

import dataclasses
import functools
import os
from collections.abc import Callable
from typing import cast

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp
from jax import lax

from vllm_torchtpu.kernels.experimental.batched_rpa import (
    bref_override,
    configs,
    flash_attention,
    pipeline,
    schedule,
    stitch_utils,
    utils,
)


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    valid_mask: jax.Array | None = None,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
    assert start % cfgs.serve.packing_kv == 0
    start //= cfgs.serve.packing_kv
    kv_u32_ref = kv_in_vref.at[b_idx].bitcast(jnp.uint32)
    kv_ref = kv_u32_ref.reshape(-1, cfgs.aligned_kv_head_dim)

    # kv_ref must be annotated with no_hazard_no_deps so XLA knows not to add a
    # dependency between the HBM->VMEM DMA copy-in from stage n - 1 and the vld
    # instructions for matmuls at stage n.
    kv_ref = pltpu.annotate(kv_ref, no_hazard_no_deps=True)

    if cfgs.serve.packing_kv == 1:
        k = utils.strided_load(
            kv_ref,
            start,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        v = utils.strided_load(
            kv_ref,
            start + 1,
            cfgs.bkv_sz * cfgs.bkv_stride,
            cfgs.bkv_stride,
            dtype=cfgs.serve.dtype_kv,
        )
        if valid_mask is not None:
            v = jnp.where(valid_mask, v, jnp.zeros_like(v))
        return [(k, v)]

    kv = utils.strided_load(
        kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride, cfgs.bkv_stride
    )
    if valid_mask is not None:
        kv = jnp.where(valid_mask, kv, jnp.zeros_like(kv))
    bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)

    return utils.convert_to_target_bitwidth(
        kv, target_bitwidth=bitwidth, kv_dtype=cfgs.serve.dtype_kv
    )


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_list: jax.Array,
    l_list: jax.Array,
    m_list: jax.Array,
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    *,
    cfgs: configs.RpaConfigs,
):
    def _accum(b_idx: int | jax.Array, batch_acc: jax.Array, batch_l: jax.Array):
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (
            cfgs.serve.dtype_out == jnp.float32
            or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16
        ):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc * pl.reciprocal(batch_l, approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        pad_width = [[0, 0] for _ in range(out.ndim)]
        pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.compute_kv_head_dim
        out = jnp.pad(out, pad_width, constant_values=0)
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

    def _stage_lse(b_idx: int, batch_m: jax.Array, batch_l: jax.Array):
        assert lse_o_vref is not None
        lse_val = batch_m + jnp.log(jnp.maximum(batch_l, 1e-9))
        lse_u32 = lse_o_vref.at[b_idx].bitcast(jnp.uint32)
        lse_ref = lse_u32.reshape(-1, lse_o_vref.shape[-1])
        lse_val = pltpu.bitcast(
            lse_val.astype(cfgs.serve.dtype_out), jnp.uint32
        ).reshape(lse_ref.shape)
        utils.strided_store(lse_ref, 0, lse_ref.shape[0], 1, lse_val)

    if cfgs.fuse_accum:
        for b in range(cfgs.batch_size):
            _accum(b, acc_list[b], l_list[b])
    else:
        # Adding a conditional causes a scheduling barrier. In prefill, we often
        # use small block sizes, so it's not worth executing the accumulation
        # on every block. In decode, because of the large block sizes / and or
        # batch sizes, we almost always use accumulation on every block. Please
        # tune `fuse_accum` for your use case.
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            acc_val = acc_list[b]
            l_val = l_list[b]
            accum_named_call = jax.named_call(_accum, name=f"accum_{b}")
            jax.lax.cond(
                is_last_k, accum_named_call, lambda *_: None, b, acc_val, l_val
            )

    if cfgs.serve.return_lse:
        for b in range(cfgs.batch_size):
            is_last_k = schedule_ref.is_last_k[step_idx, b] == 1
            m_val = m_list[b]
            l_val = l_list[b]
            jax.lax.cond(is_last_k, _stage_lse, lambda *_: None, b, m_val, l_val)


def get_scale_factors(
    k: jax.Array,
    v: jax.Array,
    *,
    cfgs: configs.RpaConfigs,
):
    if not cfgs.serve.per_token_scale:
        return k, v, cfgs.serve.scale_k, cfgs.serve.scale_v

    b = k.shape[0]
    num_heads = k.shape[1]
    # Number of scale channels in VMEM is determined by the scale factor bitwidth
    # and the VMEM kv bitwidth (k.dtype). However, we cannot use the
    # scale_channels configs.py member variable because of the case of unpacking
    # uint8_t to fp4_e2m1fn.
    if cfgs.serve.per_token_scale_dtype is None:
        scale_channels = 0
    else:
        scale_bits = jax.dtypes.itemsize_bits(cfgs.serve.per_token_scale_dtype)
        kv_bits = jax.dtypes.itemsize_bits(k.dtype)
        scale_channels = max(1, scale_bits // kv_bits)

    # Extract multi-channel scale factor slices from the trailing sublanes
    k_scale_slice = k[
        :, :, cfgs.model.head_dim : cfgs.model.head_dim + scale_channels, :
    ]
    v_scale_slice = v[
        :, :, cfgs.model.head_dim : cfgs.model.head_dim + scale_channels, :
    ]

    # If there are multiple scale channels, pltpu.bitcast operates on the
    # second to last dimension, which is the head dimension, which is the same
    # as the scale_channels dimension, so this works out of the box.
    if scale_channels > 1:
        assert cfgs.serve.per_token_scale_dtype is not None
        k_scale = pltpu.bitcast(k_scale_slice, cfgs.serve.per_token_scale_dtype)
        v_scale = pltpu.bitcast(v_scale_slice, cfgs.serve.per_token_scale_dtype)

    # If there is only one scale channel, just change dtype to the scale dtype.
    else:
        k_scale = k_scale_slice.astype(cfgs.serve.per_token_scale_dtype)
        v_scale = v_scale_slice.astype(cfgs.serve.per_token_scale_dtype)

    # In QK multiplication, we always scale after the qk matmul, so we can
    # unify per-token and per-tensor scaling by broadcasting the scale factor
    # to the head dimension outside of the flash attention kernel.
    k_scale = k_scale.reshape(b, num_heads, cfgs.bkv_sz)
    # To match the accumulation dtype of qk.
    k_scale = k_scale.astype(jnp.float32)
    # Add a new dimension for the head dimension.
    k_scale = k_scale[:, :, jnp.newaxis, :]

    # In PV multiplication, we scale the p rather than the pv for per-token
    # scaling (as it is mathematically necessary to scale before the matmul for
    # per-token scaling) so we can NOT unify per-token and per-tensor scaling,
    # so we broadcast the scale factor along the head dimension inside the
    # flash attention kernel itself and do not do so here.
    v_scale = v_scale.reshape(b, num_heads, cfgs.bkv_sz)

    k = k[:, :, : cfgs.compute_kv_head_dim, :]
    v = v[:, :, : cfgs.compute_kv_head_dim, :]

    return k, v, k_scale, v_scale


# The `pl.when` region in `rpa_body` is a basic block boundary, so splitting
# these keeps the mask arithmetic isolated to the bodies that need it.
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StitchMetadata:
    """Scalars needed to merge newly arrived KV into the cached KV window."""

    bkv_sz_frm_cache: list[jax.Array]
    bkv_sz_frm_new: list[jax.Array]
    new_kv_len_start: list[jax.Array]
    # CACHE_ONLY: rank-local end of the attended KV, relative to the block. The
    # stitch and the in-place V zeroing both stop there.
    local_k_end: list[jax.Array] | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ComputeMetadata:
    """Scalars needed to build the QK mask, but not needed by the stitch."""

    causal_offset: list[jax.Array]
    local_k_start: list[jax.Array] | None = None
    local_k_end: list[jax.Array] | None = None


def _compute_valid_kv_mask(
    token: jax.Array,
    b_idx: int,
    stitch_meta: StitchMetadata,
    step_meta: ComputeMetadata,
) -> jax.Array:
    """Builds a boolean mask for valid token slices before PV."""
    # PV reduces over the whole tile: zero probabilities do not isolate NaNs
    # from stale VMEM or another request in a fetched new-KV page.
    valid = token < (
        stitch_meta.bkv_sz_frm_cache[b_idx] + stitch_meta.bkv_sz_frm_new[b_idx]
    )
    if step_meta.local_k_start is not None:
        valid &= token >= step_meta.local_k_start[b_idx]
    if step_meta.local_k_end is not None:
        valid &= token < step_meta.local_k_end[b_idx]
    return valid


def fetch_stitch_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    cu_q_lens_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> StitchMetadata:
    """Fetches the scalars consumed by the KV stitch.

    Args:
        step: Current pipeline step.
        schedule_ref: Per-step sequence, query, and KV block indices.
        cu_q_lens_ref: Cumulative query lengths, indexed by sequence.
        kv_cache_lens_ref: Cached KV length per sequence.
        kv_new_lens_ref: Newly arrived KV length per sequence.
        cfgs: Kernel configs.

    Returns:
        The stitch boundaries for every batch element in this step.
    """
    bkv_sz_frm_cache_list = []
    bkv_sz_frm_new_list = []
    new_kv_len_start_list = []
    local_k_end_list = (
        [] if cfgs.serve.attention_scope == configs.AttentionScope.CACHE_ONLY else None
    )

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        s_idx = jnp.maximum(s_idx, 0)  # padded lanes (-1): no SMEM read at -1
        k_idx = schedule_ref.k_idx[step, b_idx]
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[s_idx], 0)
        kv_new_len_val = jnp.where(is_valid, kv_new_lens_ref[s_idx], 0)
        q_end = jnp.where(is_valid, cu_q_lens_ref[s_idx + 1], 0)

        # KV stitch boundaries: where the cached tokens end and the newly arrived
        # tokens begin, which is what `stitch_utils.stitch_new_kv_lane` merges on.
        total_kv_len = kv_cache_len_val + kv_new_len_val
        kv_left = jnp.maximum(total_kv_len - k_id, 0)
        if cfgs.serve.skip_kv_update:
            kv_left_frm_cache = kv_left
        else:
            kv_left_frm_cache = jnp.maximum(kv_cache_len_val - k_id, 0)
        kv_left_frm_new = jnp.maximum(kv_left - kv_left_frm_cache, 0)
        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz)
        bkv_sz_frm_cache_list.append(bkv_sz_frm_cache)
        bkv_sz_frm_new_list.append(jnp.minimum(kv_left, cfgs.bkv_sz) - bkv_sz_frm_cache)
        new_kv_len_start_list.append(q_end - kv_left_frm_new)
        if local_k_end_list is not None:
            local_k_end_list.append(kv_cache_len_val - k_id)

    return StitchMetadata(
        bkv_sz_frm_cache=bkv_sz_frm_cache_list,
        bkv_sz_frm_new=bkv_sz_frm_new_list,
        new_kv_len_start=new_kv_len_start_list,
        local_k_end=local_k_end_list,
    )


def fetch_compute_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> ComputeMetadata:
    """Fetches the scalars consumed by the QK mask.

    Args:
        step: Current pipeline step.
        schedule_ref: Per-step sequence, query, and KV block indices.
        q_offsets_ref: Query offset per sequence.
        kv_cache_lens_ref: Cached KV length per sequence. Only read when the
            attention scope needs `local_k_end`.
        cfgs: Kernel configs.

    Returns:
        The mask intervals for every batch element in this step.
    """
    causal_offset_list = []
    local_k_start_list = (
        []
        if cfgs.serve.attention_scope == configs.AttentionScope.NEW_TOKENS_ONLY
        else None
    )
    local_k_end_list = (
        [] if cfgs.serve.attention_scope == configs.AttentionScope.CACHE_ONLY else None
    )

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
        s_idx = jnp.maximum(s_idx, 0)  # padded lanes (-1): no SMEM read at -1
        q_idx = schedule_ref.q_idx[step, b_idx]
        k_idx = schedule_ref.k_idx[step, b_idx]
        k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
        q_offset = jnp.where(is_valid, q_offsets_ref[s_idx], 0)

        # Causal base offset: K_base - Q_base
        q_base = q_idx * cfgs.bq_sz + q_offset
        causal_offset = k_id - q_base
        causal_offset_list.append(causal_offset)

        if local_k_start_list is not None:
            local_k_start_list.append(q_offset - k_id)
        if local_k_end_list is not None:
            kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[s_idx], 0)
            local_k_end_list.append(kv_cache_len_val - k_id)

    return ComputeMetadata(
        causal_offset=causal_offset_list,
        local_k_start=local_k_start_list,
        local_k_end=local_k_end_list,
    )


def generate_mask(
    shape: tuple[int, int, int, int],
    *,
    bq_start: int,
    step_meta: ComputeMetadata,
    cfgs: configs.RpaConfigs,
) -> jax.Array:
    """Generates causal, sliding window, and attention scope mask for QK computation."""
    b, k_heads, tq, s = shape

    kv_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 2)
    q_iota = lax.broadcasted_iota(jnp.int32, (k_heads, tq, s), 1)
    q_iota //= cfgs.aligned_num_q_heads_per_kv_head
    q_kv_diff = q_iota - kv_iota

    masks = []
    for b_idx in range(b):
        # NOTE: Goal is to compute q_len >= kv_len. But we want to utilize scalar
        # compute as much as possible before involving vector compute. Therefore, we
        # break down a computational steps into following equations to separate out
        # scalar and vector compute.
        # q_len = q_iota + (bq_start + processed_q_len)
        # kv_len = kv_iota + processed_kv_len
        # Step 1: We already preprocessed causal_offset
        #   causal_offset = kv_len - q_len
        # Step 2
        #   offset = causal_offset - bq_start
        offset = step_meta.causal_offset[b_idx] - bq_start
        mask_b = q_kv_diff >= offset
        if (sliding_window := cfgs.model.sliding_window) is not None:
            mask_b = jnp.logical_and(mask_b, q_kv_diff < sliding_window + offset)

        if step_meta.local_k_start is not None:
            mask_b = jnp.logical_and(mask_b, kv_iota >= step_meta.local_k_start[b_idx])
        if step_meta.local_k_end is not None:
            mask_b = jnp.logical_and(mask_b, kv_iota < step_meta.local_k_end[b_idx])

        masks.append(mask_b)

    return masks


def _mask_unattended_v(
    v: jax.Array,
    v_scale: jax.Array | float | None,
    stitch_meta: StitchMetadata,
    step_meta: ComputeMetadata,
    *,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.Array, jax.Array | float | None]:
    """Zeroes unattended rows of V and v_scale to prevent FP8 NaNs in PV matmul."""
    seq_along_lane = cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
    kv_iota = lax.broadcasted_iota(jnp.int32, (cfgs.bkv_sz,), 0)
    has_per_token_v_scale = (
        isinstance(v_scale, jax.Array) and cfgs.serve.per_token_scale
    )

    masked_v = []
    masked_v_scale: list[jax.Array] = []

    for b in range(cfgs.batch_size):
        n_valid = stitch_meta.bkv_sz_frm_cache[b] + stitch_meta.bkv_sz_frm_new[b]
        keep = kv_iota < n_valid
        if step_meta.local_k_start is not None:
            keep = jnp.logical_and(keep, kv_iota >= step_meta.local_k_start[b])
        if step_meta.local_k_end is not None:
            keep = jnp.logical_and(keep, kv_iota < step_meta.local_k_end[b])

        v_b = v[b]
        keep_v = keep[None, None, :] if seq_along_lane else keep[None, :, None]
        masked_v.append(jnp.where(keep_v, v_b, jnp.zeros_like(v_b)))

        if has_per_token_v_scale:
            assert isinstance(v_scale, jax.Array)
            keep_sc = keep[None, :]
            masked_v_scale.append(
                jnp.where(keep_sc, v_scale[b], jnp.zeros_like(v_scale[b]))
            )

    new_v = jnp.stack(masked_v, axis=0)
    new_v_scale = (
        jnp.stack(masked_v_scale, axis=0) if has_per_token_v_scale else v_scale
    )
    return new_v, new_v_scale


def _zero_unattended_v_lanes(
    kv_in_vref: jax.Ref,
    stitch_meta: StitchMetadata,
    *,
    cfgs: configs.RpaConfigs,
):
    """Zeroes, in VMEM, the V lanes at or past the last attended KV token.

    SEQ_ALONG_LANE counterpart of `_mask_unattended_v`, for scopes without a
    `local_k_start` (FULL and CACHE_ONLY), where every unattended token of a
    lane lies in the suffix ``[hi, bkv_sz)``. The 128-lane column holding
    ``hi`` is rewritten by `stitch_utils._stitch_decode_lane`, which zeroes
    its lanes past the last valid token, so only the whole columns after it
    are stored here, with zeros and without reading them. A full KV block
    costs no vector work. The per-token v_scale rows sit in the same V head
    rows and are zeroed with them.

    Zeroing in place (rather than selecting on the loaded V) is safe for the KV
    writeback: lanes past ``hi`` are past the sequence's end (or its
    rank-local end), which the decode stitch already zero-fills on writeback.
    It is also what makes the per-call zeroing of every KV VMEM buffer
    unnecessary: lanes the step did not fetch can hold anything, and they are
    exactly the lanes the stitch and this function zero.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    for b in range(cfgs.batch_size):
        hi = stitch_meta.bkv_sz_frm_cache[b] + stitch_meta.bkv_sz_frm_new[b]
        if stitch_meta.local_k_end is not None:
            hi = jnp.minimum(hi, stitch_meta.local_k_end[b])
        # Row r of `flat` is lane column (r % lanes_per_col) of u32 word row
        # (r // lanes_per_col), as in `stitch_utils._stitch_decode_lane`.
        flat = kv_in_vref.at[b].bitcast(jnp.uint32).reshape(-1, num_lanes)
        zero_v_lanes_from(flat, hi, cfgs=cfgs)


def zero_v_lanes_from(flat: jax.Ref, hi: jax.Array, *, cfgs: configs.RpaConfigs):
    """Zeroes every V head's whole lane columns past the one holding `hi`.

    `flat` is the lane's SEQ_ALONG_LANE buffer viewed as u32 [-1, num_lanes].
    Lanes at or past `hi` in the column holding it are the stitch's to zero
    (see `_zero_unattended_v_lanes`).
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    lanes_per_col = cfgs.kv_vmem_lanes // num_lanes
    n_cols = cfgs.bkv_sz // num_lanes
    pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv
    zeros = jnp.zeros((pack_dim, num_lanes), jnp.uint32)

    hi = jnp.clip(hi, 0, cfgs.bkv_sz).astype(jnp.int32)
    first_full_col = pl.cdiv(hi, num_lanes)

    for kv_head in range(cfgs.model.num_kv_heads):
        base = (kv_head * 2 + 1) * pack_dim * lanes_per_col

        @pl.loop(first_full_col, n_cols)
        def _full_cols(c, base=base):
            flat[pl.ds(base + c, pack_dim, lanes_per_col)] = zeros


def block_fully_visible(
    step_meta: ComputeMetadata, cfgs: configs.RpaConfigs
) -> jax.Array | None:
    """Scalar: True iff `generate_mask` would be all-True for every batch entry.

    Returns None when the fast path is statically unavailable. The causal test
    `q_kv_diff >= causal_offset - bq_start` holds for the whole tile iff it
    holds at the smallest `q_kv_diff` (1 - bkv_sz) of the first q chunk
    (bq_start=0), i.e. the block's last key is at or before the first query row.
    This covers ~98% of blocks for a chunk over a long cached context.
    """
    if not flash_attention.vpu_diet_enabled():
        return None
    if cfgs.mode != configs.RpaCase.MIXED or cfgs.model.sliding_window is not None:
        return None
    full = None
    for b_idx in range(cfgs.batch_size):
        full_b = step_meta.causal_offset[b_idx] <= 1 - cfgs.bkv_sz
        if step_meta.local_k_start is not None:
            full_b = jnp.logical_and(full_b, step_meta.local_k_start[b_idx] <= 0)
        if step_meta.local_k_end is not None:
            full_b = jnp.logical_and(
                full_b, step_meta.local_k_end[b_idx] >= cfgs.bkv_sz
            )
        full = full_b if full is None else jnp.logical_and(full, full_b)
    return full


def rpa_body(
    step: jax.Array,
    slot: int | jax.Array,
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    # Passed refs
    cu_q_lens_ref: jax.Ref,
    q_offsets_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    # Configs.
    cfgs: configs.RpaConfigs,
    # DMA-issue hooks. Both only start copies; the matching waits live in
    # `pipeline.py`
    start_next_fetch: Callable[[], None],
    start_kv_writeback: Callable[[], None],
):
    del slot

    # pyrefly: ignore[bad-assignment]
    kv_in_vref = pltpu.annotate(kv_in_vref, no_hazard_no_deps=True)

    # Step 1: Fetch the metadata the stitch needs. Everything the mask needs is
    # deferred until after the fence below, so it does not gate the writeback.
    stitch_meta = fetch_stitch_metadata(
        step,
        schedule_ref,
        cu_q_lens_ref,
        kv_cache_lens_ref,
        kv_new_lens_ref,
        cfgs=cfgs,
    )

    always_true = kv_cache_lens_ref[0] >= 0

    # This conditional always executes. We create a basic block boundary here so:
    # 1. We force our KV writeback to be scheduled at the beginning of the
    #    compute region (as soon as stitching completes).
    # 2. We guarantee our stitching is correct: with `no_hazard_no_deps=True`, the
    #    boundary ensures VMEM writes from stitching finish before subsequent
    #    loads read the updated KV cache.

    @pl.when(always_true)
    @jax.named_scope("stitch_and_issue_writeback")
    def _kv_store_fence():
        if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
            stitch_results = []
            for b_idx in range(cfgs.batch_size):
                res = stitch_utils.stitch_new_kv_lane(
                    kv_in_vref,
                    b_idx,
                    stitch_meta.bkv_sz_frm_cache[b_idx],
                    stitch_meta.new_kv_len_start[b_idx],
                    stitch_meta.bkv_sz_frm_new[b_idx],
                    cfgs=cfgs,
                    local_k_end=(
                        stitch_meta.local_k_end[b_idx]
                        if stitch_meta.local_k_end is not None
                        else None
                    ),
                )
                stitch_results.append(res)
            for b_idx in range(cfgs.batch_size):
                stitch_utils.store_new_kv_lane(
                    kv_in_vref,
                    b_idx,
                    stitch_results[b_idx],
                    cfgs=cfgs,
                )

        start_kv_writeback()

        # Inside the fence, so these stores land before the K/V loads below.
        # With `no_hazard_no_deps` nothing orders them against the stitch
        # stores above, so they must not touch a lane the stitch writes with
        # anything but zero: they only store zeros into whole columns past the
        # one holding the last valid token, which the stitch rewrites itself,
        # and never read. The writeback issued above ends with that column.
        if (
            _zeroes_v_in_place(cfgs)
            and cfgs.block.bq_sz <= pltpu.get_tpu_info().num_lanes
        ):
            _zero_unattended_v_lanes(kv_in_vref, stitch_meta, cfgs=cfgs)

    start_next_fetch()

    # Step 2a: Fetch mask metadata after the fence to overlap with compute and
    # in-flight writeback.
    step_meta = fetch_compute_metadata(
        step,
        schedule_ref,
        q_offsets_ref,
        kv_cache_lens_ref,
        cfgs=cfgs,
    )

    run_step = functools.partial(
        _rpa_step,
        q_vref,
        kv_in_vref,
        o_vref,
        lse_o_vref,
        schedule_ref,
        m_scratch_ref,
        l_scratch_ref,
        acc_scratch_ref,
        step=step,
        stitch_meta=stitch_meta,
        step_meta=step_meta,
        cfgs=cfgs,
    )
    full = block_fully_visible(step_meta, cfgs)
    if full is None:
        run_step(apply_mask=True)
        return

    @pl.when(full)
    def _unmasked():
        run_step(apply_mask=False)

    @pl.when(jnp.logical_not(full))
    def _masked():
        run_step(apply_mask=True)


def _zeroes_v_in_place(cfgs: configs.RpaConfigs) -> bool:
    """Reports whether unattended V lanes are zeroed in VMEM, not on the loaded V.

    SEQ_ALONG_LANE scopes without a `local_k_start` (FULL and CACHE_ONLY) keep
    every unattended token of a lane in a suffix, which the decode path zeroes
    with `_zero_unattended_v_lanes` and `stitch_utils._stitch_prefill_lane`
    zeroes while stitching. Every other case masks the loaded V with
    `_mask_unattended_v`.
    """
    return (
        cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
        and cfgs.serve.attention_scope != configs.AttentionScope.NEW_TOKENS_ONLY
        and cfgs.serve.dtype_kv != jnp.uint8
    )


def _rpa_step(
    # Inputs.
    q_vref: jax.Ref,
    kv_in_vref: jax.Ref,
    # Outputs
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    # Scratches.
    schedule_ref: schedule.RpaSchedule,
    m_scratch_ref: jax.Ref,
    l_scratch_ref: jax.Ref,
    acc_scratch_ref: jax.Ref,
    *,
    step: jax.Array,
    stitch_meta: StitchMetadata,
    step_meta: ComputeMetadata,
    apply_mask: bool,
    cfgs: configs.RpaConfigs,
):
    """Runs one step's attention over the stitched KV block.

    `apply_mask=False` skips the QK mask, for a block `block_fully_visible`
    proved fully visible.
    """
    # Step 2b: Fetch inputs.
    q_p = cfgs.aligned_num_q_heads_per_kv_head // cfgs.serve.packing_q
    q_ref = q_vref.bitcast(jnp.uint32).reshape(-1, cfgs.aligned_q_head_dim)
    q_loaded = utils.strided_load(
        q_ref,
        0,
        cfgs.batch_size * cfgs.model.num_kv_heads * cfgs.bq_sz * q_p,
        1,
        dtype=cfgs.serve.dtype_q,
    )
    q = q_loaded.reshape(
        cfgs.batch_size,
        cfgs.model.num_kv_heads,
        cfgs.bq_sz * cfgs.aligned_num_q_heads_per_kv_head,
        cfgs.aligned_q_head_dim,
    )
    q = q[..., : cfgs.compute_kv_head_dim]

    # We want to load k, v from (batch, bkv_sz, bkv_stride, kv_packing, d)
    # where bkv_stride ~= num_kv_heads * 2 // kv_packing
    # to 2x (batch, num_kv_heads, bkv_sz, d)
    # We use strided_load to avoid the expensive transpose.
    k_b = []
    v_b = []

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv
        token = (
            lax.broadcasted_iota(jnp.int32, (pack_dim, cfgs.bkv_sz), 1)
            if cfgs.serve.mask_invalid_v
            else None
        )
        for b_idx in range(cfgs.batch_size):
            valid = (
                _compute_valid_kv_mask(token, b_idx, stitch_meta, step_meta)
                if token is not None
                else None
            )
            ks = []
            vs = []
            for kv_head in range(cfgs.model.num_kv_heads):
                k_slice = kv_in_vref.at[b_idx, kv_head * 2].bitcast(jnp.uint32)
                v_slice = kv_in_vref.at[b_idx, kv_head * 2 + 1].bitcast(jnp.uint32)

                target_shape = (-1, cfgs.kv_vmem_lanes)
                k_head_ref = k_slice.reshape(target_shape)
                v_head_ref = v_slice.reshape(target_shape)

                # Load as uint32 to avoid dtype conversion during strided load
                k_head_u32 = utils.strided_load(k_head_ref, 0, pack_dim, 1)
                v_head_u32 = utils.strided_load(v_head_ref, 0, pack_dim, 1)
                if valid is not None:
                    # HND packs channels of one token together. Mask the words
                    # before unpacking, including any co-located V scale channels.
                    v_head_u32 = jnp.where(
                        valid,
                        v_head_u32[:, : cfgs.bkv_sz],
                        jnp.zeros_like(v_head_u32[:, : cfgs.bkv_sz]),
                    )

                if cfgs.serve.dtype_kv == jnp.uint8:
                    fp4 = jnp.float4_e2m1fn
                    k_head = pltpu.bitcast(k_head_u32, fp4)[:, : cfgs.bkv_sz]
                    v_head = pltpu.bitcast(v_head_u32, fp4)[:, : cfgs.bkv_sz]
                    ks.append(k_head)
                    vs.append(v_head)
                else:
                    k_head_loaded = pltpu.bitcast(k_head_u32, cfgs.serve.dtype_kv)
                    v_head_loaded = pltpu.bitcast(v_head_u32, cfgs.serve.dtype_kv)
                    k_head = k_head_loaded[:, : cfgs.bkv_sz]
                    v_head = v_head_loaded[:, : cfgs.bkv_sz]
                    ks.append(k_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))
                    vs.append(v_head.reshape(cfgs.aligned_kv_head_dim, cfgs.bkv_sz))

            k_b.append(jnp.stack(ks, axis=0))
            v_b.append(jnp.stack(vs, axis=0))
    else:
        token = None
        if cfgs.serve.mask_invalid_v:
            token = lax.broadcasted_iota(
                jnp.int32, (cfgs.bkv_sz, cfgs.aligned_kv_head_dim), 0
            )
        for b_idx in range(cfgs.batch_size):
            valid = (
                _compute_valid_kv_mask(token, b_idx, stitch_meta, step_meta)
                if token is not None
                else None
            )
            heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
            ks = []
            vs = []
            for kv_head_start in range(0, cfgs.model.num_kv_heads, heads_per_load):
                bkv_lst = strided_load_bkv(
                    kv_in_vref,
                    b_idx,
                    kv_head_start * 2,
                    valid_mask=valid,
                    cfgs=cfgs,
                )
                ks.append(jnp.stack([k for k, _ in bkv_lst], axis=0))
                vs.append(jnp.stack([v for _, v in bkv_lst], axis=0))
            k, v = jnp.concat(ks, axis=0), jnp.concat(vs, axis=0)
            k = k.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)
            v = v.reshape(-1, cfgs.bkv_sz, cfgs.aligned_kv_head_dim)

            k = k[: cfgs.model.num_kv_heads]
            v = v[: cfgs.model.num_kv_heads]
            k_b.append(k)
            v_b.append(v)

    # Stack to (batch, num_heads, bkv_sz, num_lanes)
    k = jnp.stack(k_b, axis=0)
    v = jnp.stack(v_b, axis=0)

    k, v, k_scale, v_scale = get_scale_factors(k, v, cfgs=cfgs)

    # Zero V and v_scale rows that no query of the lane attends to. Score
    # masking only zeroes their softmax weights, but 0 * NaN = NaN in the PV
    # matmul, and those rows can hold anything: page tails past the (rank-local)
    # KV length, lanes with no sequence, or unfetched VMEM. In the unified
    # hybrid pool a recycled page's tail still holds Mamba state bytes, many of
    # which decode to fp8 NaN. SEQ_ALONG_LANE without a `local_k_start` zeroes
    # them in VMEM instead, before this load (`_zeroes_v_in_place`).
    if not _zeroes_v_in_place(cfgs):
        v, v_scale = _mask_unattended_v(v, v_scale, stitch_meta, step_meta, cfgs=cfgs)

    # Step 3: Perform compute.
    m_val = m_scratch_ref[...]
    l_val = l_scratch_ref[...]
    acc_val = acc_scratch_ref[...]

    l_new_list = []
    m_new_list = []
    acc_new_list = []

    prev_p = prev_alpha_list = prev_q_slice = None
    for bq_start in range(0, cfgs.bq_sz, cfgs.bq_c_sz):
        bq_end = min(bq_start + cfgs.bq_c_sz, cfgs.bq_sz)
        q_start = bq_start * cfgs.aligned_num_q_heads_per_kv_head
        q_end = bq_end * cfgs.aligned_num_q_heads_per_kv_head
        q_slice = slice(q_start, q_end)

        custom_mask = None
        if apply_mask:
            custom_mask = generate_mask(
                shape=(
                    cfgs.batch_size,
                    cfgs.model.num_kv_heads,
                    q_end - q_start,
                    cfgs.bkv_sz,
                ),
                bq_start=bq_start,
                step_meta=step_meta,
                cfgs=cfgs,
            )

        p, alpha_list, m_next, l_next, m_carry, l_carry = (
            flash_attention.flash_attention_qk_softmax(
                step,
                q[:, :, q_slice],
                k,
                m_val[:, q_slice],
                l_val[:, q_slice],
                schedule_ref.is_last_k,
                custom_mask=custom_mask,
                cfgs=cfgs,
                bq_start=bq_start,
                k_scale=k_scale,
            )
        )
        m_scratch_ref[:, q_slice] = m_carry
        l_scratch_ref[:, q_slice] = l_carry
        if cfgs.serve.return_lse:
            m_new_list.append(m_next)
        l_new_list.append(l_next)

        if prev_p is not None:
            o_next, o_carry = flash_attention.flash_attention_pv(
                prev_p,
                v,
                cast(list[jax.Array], prev_alpha_list),
                acc_val[:, prev_q_slice],
                step,
                schedule_ref.is_last_k,
                cfgs=cfgs,
                v_scale=v_scale,
            )
            acc_scratch_ref[:, prev_q_slice] = o_carry
            acc_new_list.append(o_next)

        prev_p = p
        prev_alpha_list = alpha_list
        prev_q_slice = q_slice

    assert prev_p is not None
    o_next, o_carry = flash_attention.flash_attention_pv(
        prev_p,
        v,
        cast(list[jax.Array], prev_alpha_list),
        acc_val[:, prev_q_slice],
        step,
        schedule_ref.is_last_k,
        cfgs=cfgs,
        v_scale=v_scale,
    )
    acc_scratch_ref[:, prev_q_slice] = o_carry
    acc_new_list.append(o_next)
    if cfgs.serve.return_lse:
        m_next = jnp.concatenate(m_new_list, axis=2)
    l_next = jnp.concatenate(l_new_list, axis=2)
    acc_next = jnp.concatenate(acc_new_list, axis=2)

    # Step 4: Write back outputs.
    calculate_and_store_out(
        step,
        schedule_ref,
        acc_next,
        l_next,
        m_next,
        o_vref,
        lse_o_vref,
        cfgs=cfgs,
    )


# Define main kernel.


def create_allocs(
    kv_cache_hbm_ref: jax.Ref,
    o_hbm_ref: jax.Ref,
    lse_hbm_ref: jax.Ref | None,
    cfgs: configs.RpaConfigs,
) -> tuple[
    bref_override.BatchingQRef,
    bref_override.KVBufferedRefSeqAlongLane
    | bref_override.KVBufferedRefHeadAlongSublane,
    bref_override.BatchingORef,
    bref_override.BatchingLSERef | None,
]:
    # `use_lookahead` is disabled everywhere: slot rotation and prefetch depth are
    # driven explicitly by `pipeline.run_manual_triple_buffer`.
    in_slots = configs.NUM_INPUT_SLOTS
    out_slots = configs.NUM_OUTPUT_SLOTS
    kv_cache_spec = pl.BlockSpec(
        block_shape=cfgs.kv_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=in_slots, use_lookahead=False),
    )
    q_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=in_slots, use_lookahead=False),
    )
    o_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
    )

    if cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE:
        kv_cache_alloc_cls = bref_override.KVBufferedRefSeqAlongLane
    else:
        kv_cache_alloc_cls = bref_override.KVBufferedRefHeadAlongSublane

    kv_cache_alloc = kv_cache_alloc_cls.input_output(
        spec=kv_cache_spec,
        dtype_or_type=kv_cache_hbm_ref,
        buffer_count=in_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )
    q_alloc = bref_override.BatchingQRef.input(
        spec=q_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=in_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )
    o_alloc = bref_override.BatchingORef.output(
        spec=o_spec,
        dtype_or_type=o_hbm_ref,
        buffer_count=out_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )

    lse_alloc = None
    if cfgs.serve.return_lse:
        lse_spec = pl.BlockSpec(
            block_shape=cfgs.lse_vmem_shape,
            memory_space=pltpu.VMEM,
            index_map=lambda i: (i,),
            pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
        )
        lse_alloc = bref_override.BatchingLSERef.output(
            spec=lse_spec,
            dtype_or_type=lse_hbm_ref,
            buffer_count=out_slots,
            use_lookahead=False,
            cfgs=cfgs,
        )

    return q_alloc, kv_cache_alloc, o_alloc, lse_alloc


def get_kernel_name(cfgs: configs.RpaConfigs) -> str:
    serve = cfgs.serve
    name = f"RPA{cfgs.mode.symbol}-{serve.kv_layout.symbol}-p{serve.page_size}"
    name += f"-b{cfgs.batch_size}-q{cfgs.bq_sz}-k{cfgs.bkv_sz}"
    if cfgs.model.sliding_window:
        name += f"-sw{cfgs.model.sliding_window}"
    return name


def get_kernel_metadata(
    cfgs: configs.RpaConfigs,
) -> dict[str, str | int | float]:
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if not isinstance(val, str | int | float):
            val = str(val)
        ret[key] = val
    return ret


def rpa_kernel(
    cu_q_lens: jax.Array,
    q_offsets: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
    computer_cls: type[schedule.BaseMetadataComputer] = schedule.BaseMetadataComputer,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Perform batched ragged paged attention with scheduler data.

    Args:
        cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
            length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
            b=cu_q_lens[i+1] represents q/k/v of sequence i.
        q_offsets: [max_num_seqs]. Token offset for queries in each sequence.
        kv_cache_lens: [max_num_seqs]. Existing kv cache length of each sequence.
        kv_new_lens: [max_num_seqs]. New kv length of each sequence.
        schedule_hbm: Output of scheduler kernel. It informs which: 1. seqs 2. q
            block 3. kv block that should be processed at a given step.
        q_hbm: [max_num_tokens, num_q_heads_per_kv_heads, cdiv(num_kv_heads,
            q_packing), q_packing, head_dim]. Output of q projection that has been
            pre-processed to align with existing kv cache data layout.
        new_kv_hbm: [max_num_tokens, cdiv(num_kv_heads * 2, kv_packing), kv_packing,
            head_dim]. Output of k & v projection that has been pre-processed to align
            with existing kv cache data layout.
        kv_cache_hbm: [num_pages, page_size, cdiv(num_kv_heads * 2, kv_packing),
            kv_packing, head_dim]. Stores existing kv cache data where k & vs are
            concatenated along num kv heads dim.
        lse_hbm: pre-allocated buffer for LSE output. None when return_lse=False.
        cfgs: Configuration of the kernel.

    Returns:
        out: [max_num_tokens, num_q_heads, head_dim]. Output of self attention.
        new_kv_cache: [num_pages, page_size, num_kv_heads // kv_packing, kv_packing,
            head_dim]. Result of new kv cache.
        lse_out: [max_num_tokens, num_q_heads] LSE values, or None.
    """
    return_lse = cfgs.serve.return_lse

    def ragged_paged_attention_pipeline(
        # Scalar prefetch.
        cu_q_lens_ref: jax.Ref,
        q_offsets_ref: jax.Ref,
        kv_cache_lens_ref: jax.Ref,
        kv_new_lens_ref: jax.Ref,
        # Inputs.
        schedule_hbm_ref: schedule.RpaSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        lse_hbm_ref: jax.Ref | None,
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
        o_lse_hbm_ref: jax.Ref | None = None,
    ):
        del o_kv_cache_hbm_ref
        if o_lse_hbm_ref is not None:
            del o_lse_hbm_ref

        q_alloc, kv_cache_alloc, o_alloc, lse_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs
        )

        actual_steps = schedule_hbm_ref.actual_steps[0]
        num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
            schedule_ref=computer_cls.get_rpa_schedule(cfgs).scratch_shapes(),
            dma_sem=pltpu.SemaphoreType.DMA((1,)),
            scratches=(
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # m
                pltpu.VMEM(
                    cfgs.lm_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # l
                pltpu.VMEM(
                    cfgs.acc_scratch_shape,
                    dtype=cfgs.serve.dtype_out,
                ),  # acc
            ),
        )
        def _run(final_allocs, schedule_ref, dma_sem, scratches):
            # Initialize Q to zeros to prevent NaN pollution.
            #
            # When a query block is partially filled, tail slots in uninitialized VMEM
            # may contain residual NaNs. In our implementation, invalid Q rows bypass
            # invalid row masking and produce NaNs in p_rowsum and later pollute
            # l_scratch and acc_scratch. Once l_scratch / acc_scratch becomes NaN,
            # subsequent iterations cannot recover (0 * NaN = NaN).

            # Initialize KV cache to zeros.
            # When perfomring p * v, we perform causal masking on lhs (p) by zeroing
            # out columns that should not be processed for a given row. Even if we
            # don't perform masking on rows of rhs (v), the output is still correct
            # since reuslt of multiplication will be zero thanks zero on lhs. However,
            # this assumption does not hold if a row of rhs has NaNs. To avoid this,
            # we initiallize scratch memory with non-zero values. Even if the scratch
            # memory is storing kv cache from previous step, as long as the data is
            # not NaNs, there will be no numeric concerns.
            #
            # Every step zeroes the V rows its lanes do not attend to
            # (`_zero_unattended_v_lanes` / `_mask_unattended_v`), and NaNs in
            # unattended K only reach masked scores, which `jnp.where` replaces.
            # So the KV buffers (~27 MB of VMEM for decode, a fixed few us per
            # call) stay unzeroed unless TPU_RPA_ZERO_KV_VMEM=1.
            zero_kv_vmem = os.environ.get("TPU_RPA_ZERO_KV_VMEM", "0") == "1"

            q_alloc, kv_cache_alloc, o_alloc, lse_alloc = final_allocs

            @pl.when(actual_steps > 0)
            def _active():
                scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)

                num_lanes = pltpu.get_tpu_info().num_lanes
                q_ref_flat = q_alloc.window_ref.bitcast(jnp.uint32).reshape(
                    -1, num_lanes
                )
                zero_refs = [scratches[1], scratches[2], q_ref_flat]
                if zero_kv_vmem:
                    zero_refs.append(
                        kv_cache_alloc.window_ref.bitcast(jnp.uint32).reshape(
                            -1, num_lanes
                        )
                    )

                # Explicitly zero out scratches and the Q (and optionally KV)
                # buffers.
                for ref in zero_refs:
                    ref[...] = jnp.zeros_like(ref)

                def execute_schedule_chunk(start_step, num_steps):
                    # All reads are aligned to 128 and some extra steps are copied
                    # in the process.
                    aligned_start_step = (start_step // 128) * 128
                    prefix_steps = start_step - aligned_start_step

                    flat_hbm = jax.tree_util.tree_leaves(schedule_hbm_ref)
                    flat_smem = jax.tree_util.tree_leaves(schedule_ref)
                    dma_list = []
                    for h, s in zip(flat_hbm, flat_smem):
                        if jax.typeof(h).memory_space == pltpu.HBM:
                            element_size = s.shape[0] // cfgs.max_steps_ub
                            read_size = element_size * (num_steps + prefix_steps)
                            read_size = utils.align_to(read_size, 1024)
                            read_size = jnp.minimum(read_size, s.shape[0])

                            src_off = element_size * aligned_start_step
                            src_off = pl.multiple_of(src_off, 128)

                            copy = pltpu.make_async_copy(
                                h.at[pl.ds(src_off, read_size)],
                                s.at[pl.ds(0, read_size)],
                                dma_sem.at[0],
                            )
                            copy.start()
                            dma_list.append(copy)
                    jax.tree.map(lambda x: x.wait(), dma_list)

                    total_steps = num_steps + prefix_steps
                    kv_bound = kv_cache_alloc.bind(
                        src_ref=(
                            kv_cache_hbm_ref,
                            new_kv_hbm_ref,
                            schedule_ref,
                            page_indices_ref,
                        ),
                        dst_ref=(
                            kv_cache_hbm_ref,
                            new_kv_hbm_ref,
                            schedule_ref,
                            page_indices_ref,
                        ),
                    )
                    q_bound = q_alloc.bind(src_ref=(q_hbm_ref, schedule_ref))
                    o_bound = o_alloc.bind(dst_ref=(o_hbm_ref, schedule_ref))
                    lse_bound = (
                        lse_alloc.bind(dst_ref=(lse_hbm_ref, schedule_ref))
                        if return_lse
                        else None
                    )

                    def compute_fn(
                        step,
                        slot,
                        total_steps,
                        out_slot,
                        start_next_fetch,
                        start_kv_writeback,
                    ):
                        del total_steps
                        rpa_body(
                            step,
                            slot,
                            q_alloc.window_ref.at[slot],
                            kv_cache_alloc.window_ref.at[slot],
                            o_alloc.window_ref.at[out_slot],
                            lse_alloc.window_ref.at[out_slot] if return_lse else None,
                            schedule_ref,
                            scratches[0],
                            scratches[1],
                            scratches[2],
                            cu_q_lens_ref=cu_q_lens_ref,
                            q_offsets_ref=q_offsets_ref,
                            kv_cache_lens_ref=kv_cache_lens_ref,
                            kv_new_lens_ref=kv_new_lens_ref,
                            cfgs=cfgs,
                            start_next_fetch=start_next_fetch,
                            start_kv_writeback=start_kv_writeback,
                        )

                    pipeline.run_manual_triple_buffer(
                        total_steps,
                        compute_fn,
                        q_bound,
                        o_bound,
                        kv_bound,
                        lse_bref=lse_bound,
                    )

                @pl.loop(0, num_safe_step_iterations)
                def loop_body(step_idx):
                    start = step_idx * cfgs.max_steps_ub
                    rem = actual_steps % cfgs.max_steps_ub
                    last_step_size = jnp.where(rem == 0, cfgs.max_steps_ub, rem)
                    is_last_step = step_idx == num_safe_step_iterations - 1
                    size = jnp.where(is_last_step, last_step_size, cfgs.max_steps_ub)

                    execute_schedule_chunk(start, size)

        _run()

    scalar_prefetches = (
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
    )
    num_scalar_prefetch = len(scalar_prefetches)
    num_active_scalers = len(scalar_prefetches)

    out_shape = [q_hbm, kv_cache_hbm, lse_hbm if return_lse else None]

    schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    input_output_aliases = {
        num_active_scalers + schedule_leaves: 0,
        num_active_scalers + schedule_leaves + 2: 1,
    }
    if return_lse:
        input_output_aliases[num_active_scalers + schedule_leaves + 3] = (
            2  # lse_hbm -> out[2]
        )

    return pl.pallas_call(
        ragged_paged_attention_pipeline,
        out_shape=out_shape,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=num_scalar_prefetch,
            in_specs=[
                schedule_hbm.in_specs(),
                pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=cfgs.vmem_limit_bytes,
            disable_bounds_checks=True,
            flags={
                "XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True,
            },
        ),
        input_output_aliases=input_output_aliases,
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )(
        cu_q_lens,
        q_offsets,
        kv_cache_lens,
        kv_new_lens,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        lse_hbm if return_lse else None,
    )
