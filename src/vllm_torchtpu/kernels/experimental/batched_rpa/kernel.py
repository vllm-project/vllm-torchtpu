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
from collections.abc import Callable, Sequence

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

# Headroom left in the VMEM pool when RPA's scoped window is shrunk to make
# room for the input windows handed in by `rpa_prefetch_start`.
_VMEM_MARGIN_BYTES = 1024 * 1024
# Scoped VMEM for `rpa_prefetch_start`, which uses no VMEM scratch. Its output
# windows are live while it runs, so the default scoped limit (32 MiB) plus
# large windows would overflow the VMEM pool.
_PREFETCH_VMEM_LIMIT_BYTES = _VMEM_MARGIN_BYTES


def strided_load_bkv(
    kv_in_vref: jax.Ref,
    b_idx: int,
    start: int,
    *,
    cfgs: configs.RpaConfigs,
) -> list[tuple[jax.Array, jax.Array]]:
    assert start % cfgs.serve.packing_kv == 0
    start //= cfgs.serve.packing_kv
    kv_u32_ref = kv_in_vref[b_idx].bitcast(jnp.uint32)
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
        return [(k, v)]

    kv = utils.strided_load(
        kv_ref, start, cfgs.bkv_sz * cfgs.bkv_stride, cfgs.bkv_stride
    )
    bitwidth = jax.dtypes.itemsize_bits(cfgs.serve.dtype_kv)

    return utils.convert_to_target_bitwidth(
        kv, target_bitwidth=bitwidth, kv_dtype=cfgs.serve.dtype_kv
    )


def calculate_and_store_out(
    step_idx: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    acc_list: list[jax.Array],
    l_list: list[jax.Array],
    m_list: list[jax.Array],
    o_vref: jax.Ref,
    lse_o_vref: jax.Ref | None,
    *,
    cfgs: configs.RpaConfigs,
):
    def _accum(b_idx: jax.Array, batch_acc: jax.Array, batch_l: jax.Array):
        batch_l = utils.broadcast_minor(batch_l, batch_acc.shape)

        if (
            cfgs.serve.dtype_out == jnp.float32
            or cfgs.serve.dtype_out == batch_l.dtype == jnp.bfloat16
        ):
            result = lax.div(batch_acc, batch_l)
        else:
            result = batch_acc * pl.reciprocal(batch_l, approx=True)
        out = result.astype(cfgs.serve.dtype_out)

        o_u32_vref = o_vref[b_idx].bitcast(jnp.uint32)
        out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
        pad_width = [[0, 0] for _ in range(out.ndim)]
        pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.compute_kv_head_dim
        out = jnp.pad(out, pad_width, constant_values=0)
        out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
        utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

    def _stage_lse(b_idx: int, batch_m: jax.Array, batch_l: jax.Array):
        lse_val = batch_m + jnp.log(jnp.maximum(batch_l, 1e-9))
        lse_u32 = lse_o_vref[b_idx].bitcast(jnp.uint32)
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


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StitchMetadata:
    """Scalars needed to merge newly arrived KV into the cached KV window."""

    bkv_sz_frm_cache: list[jax.Array]
    new_kv_len_start: list[jax.Array]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class ComputeMetadata:
    """Scalars needed to build the QK mask, but not needed by the stitch."""

    causal_offset: list[jax.Array]
    local_k_start: list[jax.Array] | None = None
    local_k_end: list[jax.Array] | None = None


# The `pl.when` region in `rpa_body` is a basic block boundary, so splitting
# these keeps the mask arithmetic isolated to the bodies that need it.
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
    new_kv_len_start_list = []

    for b_idx in range(cfgs.batch_size):
        s_idx = schedule_ref.s_idx[step, b_idx]
        is_valid = s_idx != -1
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
        bkv_sz_frm_cache_list.append(jnp.minimum(kv_left_frm_cache, cfgs.bkv_sz))
        new_kv_len_start_list.append(q_end - kv_left_frm_new)

    return StitchMetadata(
        bkv_sz_frm_cache=bkv_sz_frm_cache_list,
        new_kv_len_start=new_kv_len_start_list,
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


def rpa_body(
    step: int | jax.Array,
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
                    cfgs=cfgs,
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
        for b_idx in range(cfgs.batch_size):
            ks = []
            vs = []
            for kv_head in range(cfgs.model.num_kv_heads):
                k_slice = kv_in_vref[b_idx, kv_head * 2].bitcast(jnp.uint32)
                v_slice = kv_in_vref[b_idx, kv_head * 2 + 1].bitcast(jnp.uint32)

                target_shape = (-1, cfgs.kv_vmem_lanes)
                k_head_ref = k_slice.reshape(target_shape)
                v_head_ref = v_slice.reshape(target_shape)
                pack_dim = cfgs.aligned_kv_head_dim // cfgs.serve.packing_kv

                # Load as uint32 to avoid dtype conversion during strided load
                k_head_u32 = utils.strided_load(k_head_ref, 0, pack_dim, 1)
                v_head_u32 = utils.strided_load(v_head_ref, 0, pack_dim, 1)

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
        for b_idx in range(cfgs.batch_size):
            heads_per_load = pl.cdiv(cfgs.serve.packing_kv, 2)
            ks = []
            vs = []
            for kv_head_start in range(0, cfgs.model.num_kv_heads, heads_per_load):
                bkv_lst = strided_load_bkv(
                    kv_in_vref,
                    b_idx,
                    kv_head_start * 2,
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

        p, alpha_list, m_next, l_next, m_carry = (
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
        l_scratch_ref[:, q_slice] = l_next[-1]
        if cfgs.serve.return_lse:
            m_new_list.append(m_next)
        l_new_list.append(l_next)

        if prev_p is not None:
            o_next = flash_attention.flash_attention_pv(
                prev_p,
                v,
                prev_alpha_list,
                acc_val[:, prev_q_slice],
                cfgs=cfgs,
                v_scale=v_scale,
            )
            acc_scratch_ref[:, prev_q_slice] = o_next[-1]
            acc_new_list.append(o_next)

        prev_p = p
        prev_alpha_list = alpha_list
        prev_q_slice = q_slice

    assert prev_p is not None
    o_next = flash_attention.flash_attention_pv(
        prev_p,
        v,
        prev_alpha_list,
        acc_val[:, prev_q_slice],
        cfgs=cfgs,
        v_scale=v_scale,
    )
    acc_scratch_ref[:, prev_q_slice] = o_next[-1]
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


@dataclasses.dataclass(frozen=True)
class RpaPrefetch:
    """Outputs of `rpa_prefetch_start`, consumed by `rpa_kernel`.

    Attributes:
      metadata_smem: SMEM metadata (cu_q_lens, q_offsets, kv_cache_lens,
        kv_new_lens, page_indices), already landed.
      schedule_chunk_smem: SMEM schedule chunk buffers holding chunk 0, already
        landed. RPA DMAs later chunks into them in place.
      q_window: VMEM Q input window, `(NUM_INPUT_SLOTS, *cfgs.q_vmem_shape)`.
      kv_window: VMEM KV input window, `(NUM_INPUT_SLOTS, *cfgs.kv_vmem_shape)`.
      q_sems: Per-slot Q receive DMA semaphores, with the first-use DMAs of every
        slot in flight.
      kv_sems: Per-slot KV receive DMA semaphores, likewise.
      actual_steps_smem: `(1,)` int32 SMEM copy of the schedule's `actual_steps`,
        so RPA has no HBM -> SMEM operand to load at entry.
    """

    metadata_smem: tuple[jax.Array, ...]
    schedule_chunk_smem: tuple[jax.Array, ...]
    q_window: jax.Array
    kv_window: jax.Array
    q_sems: jax.Array
    kv_sems: jax.Array
    actual_steps_smem: jax.Array


def create_input_allocs(
    kv_cache_hbm_ref: jax.Ref,
    q_hbm_ref: jax.Ref,
    cfgs: configs.RpaConfigs,
) -> tuple[
    bref_override.BatchingQRef,
    bref_override.KVBufferedRefSeqAlongLane
    | bref_override.KVBufferedRefHeadAlongSublane,
]:
    """Q / KV input allocs. Their windows are `(NUM_INPUT_SLOTS, *vmem_shape)`."""
    # `use_lookahead` is disabled everywhere: slot rotation and prefetch depth are
    # driven explicitly by `pipeline.run_manual_triple_buffer`.
    in_slots = configs.NUM_INPUT_SLOTS
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
        dtype_or_type=q_hbm_ref,
        buffer_count=in_slots,
        use_lookahead=False,
        cfgs=cfgs,
    )
    return q_alloc, kv_cache_alloc


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
    out_slots = configs.NUM_OUTPUT_SLOTS
    q_alloc, kv_cache_alloc = create_input_allocs(kv_cache_hbm_ref, o_hbm_ref, cfgs)
    o_spec = pl.BlockSpec(
        block_shape=cfgs.q_vmem_shape,
        memory_space=pltpu.VMEM,
        index_map=lambda i: (i,),
        pipeline_mode=pl.Buffered(buffer_count=out_slots, use_lookahead=False),
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


def input_window_shapes(
    q_dtype: jax.typing.DTypeLike,
    kv_dtype: jax.typing.DTypeLike,
    cfgs: configs.RpaConfigs,
) -> tuple[jax.ShapeDtypeStruct, jax.ShapeDtypeStruct]:
    """Shapes of the Q / KV input windows (all slots)."""
    in_slots = configs.NUM_INPUT_SLOTS
    return (
        jax.ShapeDtypeStruct((in_slots, *cfgs.q_vmem_shape), q_dtype),
        jax.ShapeDtypeStruct((in_slots, *cfgs.kv_vmem_shape), kv_dtype),
    )


def rpa_prefetch_start(
    metadata: Sequence[jax.Array],
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    zero_fill_bufs: tuple[jax.Array, jax.Array],
    *,
    cfgs: configs.RpaConfigs,
    computer_cls: type[schedule.BaseMetadataComputer] = schedule.BaseMetadataComputer,
) -> RpaPrefetch:
    """Starts everything RPA's first schedule chunk needs, without waiting.

    1. DMAs the metadata and the first schedule chunk into XLA-owned SMEM output
       buffers, and waits for them (the VMEM DMAs below are addressed from them).
    2. Issues the first use of every input slot into XLA-owned VMEM output
       windows: Q / KV data for steps 0 and 1 plus the zero fill of each slot's
       remainder, and the zero fill of slot 2 (its data comes from RPA's step 0).
       These DMAs signal the returned per-slot semaphores and are waited on by
       RPA's `wait_in(..., init=True)`.

    The VMEM windows stay in place until RPA consumes them: XLA reserves a
    VMEM-colored buffer for its whole live range (see vmem_handoff_test.py).

    Args:
      metadata: 1-D int32 arrays consumed as RPA scalar prefetch (cu_q_lens,
        q_offsets, kv_cache_lens, kv_new_lens, page_indices).
      schedule_hbm: Output of `generate_rpa_metadata`.
      q_hbm: Q in HBM; also RPA's O buffer.
      new_kv_hbm: New K/V in HBM.
      kv_cache_hbm: KV cache in HBM.
      zero_fill_bufs: (q_zeros, kv_zeros), HBM zeros shaped like one Q / KV slot.
      cfgs: Configuration of the kernel.
      computer_cls: Metadata computer class used for schedule generation.

    Returns:
      The SMEM buffers, VMEM windows and receive semaphores for RPA.
    """
    metadata = tuple(metadata)
    chunk_shapes = tuple(computer_cls.get_smem_chunk_shapes(cfgs))
    metadata_smem_shapes = tuple(pltpu.SMEM(m.shape, m.dtype) for m in metadata)
    q_window, kv_window = input_window_shapes(q_hbm.dtype, kv_cache_hbm.dtype, cfgs)
    in_slots = configs.NUM_INPUT_SLOTS

    def _kernel(
        metadata_hbm_refs,
        schedule_hbm_ref,
        q_hbm_ref,
        new_kv_hbm_ref,
        kv_cache_hbm_ref,
        zero_hbm_refs,
        metadata_smem_refs,
        chunk_smem_refs,
        q_window_ref,
        kv_window_ref,
        q_sems,
        kv_sems,
        actual_steps_smem_ref,
        smem_sem,
    ):
        actual_steps = schedule_hbm_ref.actual_steps[0]
        actual_steps_smem_ref[0] = actual_steps
        flat_hbm, schedule_treedef = jax.tree_util.tree_flatten(schedule_hbm_ref)
        hbm_leaves = [h for h in flat_hbm if jax.typeof(h).memory_space == pltpu.HBM]
        assert len(hbm_leaves) == len(chunk_smem_refs)

        # Step 1: SMEM metadata and the first schedule chunk.
        copies = []
        for src, dst in zip(metadata_hbm_refs, metadata_smem_refs):
            copies.append(pltpu.make_async_copy(src, dst, smem_sem[0]))
        for src, dst in zip(hbm_leaves, chunk_smem_refs):
            size = schedule.schedule_chunk_read_size(
                dst.shape[0], cfgs.max_steps_ub, actual_steps
            )
            copies.append(
                pltpu.make_async_copy(
                    src[pl.ds(0, size)], dst[pl.ds(0, size)], smem_sem[0]
                )
            )
        for c in copies:
            c.start()
        for c in copies:
            c.wait()

        chunk_iter = iter(chunk_smem_refs)
        schedule_smem = schedule_treedef.unflatten(
            [
                next(chunk_iter) if jax.typeof(h).memory_space == pltpu.HBM else h
                for h in flat_hbm
            ]
        )
        page_indices_ref = metadata_smem_refs[4]

        # Step 2: first use of every input slot.
        q_alloc, kv_alloc = create_input_allocs(kv_cache_hbm_ref, q_hbm_ref, cfgs)
        kv_bref = dataclasses.replace(
            kv_alloc, window_ref=kv_window_ref, sem_recvs=kv_sems, sem_sends=None
        ).bind(
            src_ref=(
                kv_cache_hbm_ref,
                new_kv_hbm_ref,
                schedule_smem,
                page_indices_ref,
            ),
            zero_src_ref=zero_hbm_refs[1],
        )
        q_bref = dataclasses.replace(
            q_alloc, window_ref=q_window_ref, sem_recvs=q_sems
        ).bind(src_ref=(q_hbm_ref, schedule_smem), zero_src_ref=zero_hbm_refs[0])

        total_steps = jnp.minimum(cfgs.max_steps_ub, actual_steps)
        for s in range(2):
            q_bref.init_copy_in(s, s, total_steps, True)
            kv_bref.init_copy_in(s, s, total_steps, True)
        # Slots >= 2 get their data DMAs from RPA's steady-state loop. Their zero
        # regions are disjoint from those DMAs, so they can be issued now.
        for s in range(2, in_slots):
            q_bref.zero_fill_in(s, s, total_steps)
            kv_bref.zero_fill_in(s, s, total_steps)

    (
        metadata_smem,
        chunk_smem,
        q_win,
        kv_win,
        q_sems,
        kv_sems,
        actual_steps_smem,
    ) = pl.pallas_call(
        _kernel,
        in_specs=[
            tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in metadata),
            schedule_hbm.in_specs(),
            pl.BlockSpec(memory_space=pltpu.HBM),  # q_hbm
            pl.BlockSpec(memory_space=pltpu.HBM),  # new_kv_hbm
            pl.BlockSpec(memory_space=pltpu.HBM),  # kv_cache_hbm
            tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in zero_fill_bufs),
        ],
        out_specs=(
            tuple(pl.BlockSpec(memory_space=pltpu.SMEM) for _ in metadata),
            tuple(pl.BlockSpec(memory_space=pltpu.SMEM) for _ in chunk_shapes),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SEMAPHORE),
            pl.BlockSpec(memory_space=pltpu.SMEM),  # actual_steps
        ),
        out_shape=(
            metadata_smem_shapes,
            chunk_shapes,
            pltpu.VMEM(q_window.shape, q_window.dtype),
            pltpu.VMEM(kv_window.shape, kv_window.dtype),
            pltpu.SemaphoreType.DMA((in_slots,)),
            pltpu.SemaphoreType.DMA((in_slots,)),
            pltpu.SMEM((1,), jnp.int32),
        ),
        scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        compiler_params=pltpu.CompilerParams(
            has_side_effects=pltpu.SideEffectType.DATAFLOW_SIDE_EFFECTING,
            disable_bounds_checks=True,
            vmem_limit_bytes=_PREFETCH_VMEM_LIMIT_BYTES,
        ),
        name="rpa_prefetch_start",
    )(
        metadata,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        tuple(zero_fill_bufs),
    )
    return RpaPrefetch(
        metadata_smem=tuple(metadata_smem),
        schedule_chunk_smem=tuple(chunk_smem),
        q_window=q_win,
        kv_window=kv_win,
        q_sems=q_sems,
        kv_sems=kv_sems,
        actual_steps_smem=actual_steps_smem,
    )


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
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
    prefetch: RpaPrefetch,
    zero_fill_bufs: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
    """Perform batched ragged paged attention with scheduler data.

    Args:
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
      prefetch: Outputs of `rpa_prefetch_start`: the landed SMEM metadata and
        first schedule chunk, and the Q / KV input windows with the first use of
        every slot in flight. The windows are this kernel's input windows for all
        schedule chunks.
      zero_fill_bufs: The zero-fill DMA sources used by `rpa_prefetch_start`.
        Passed only to keep them alive until those DMAs are waited on here.

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
        page_indices_ref: jax.Ref,
        schedule_chunk_refs: tuple[jax.Ref, ...],
        actual_steps_ref: jax.Ref,
        # Inputs.
        schedule_hbm_ref: schedule.RpaSchedule,
        q_hbm_ref: jax.Ref,
        new_kv_hbm_ref: jax.Ref,
        kv_cache_hbm_ref: jax.Ref,
        lse_hbm_ref: jax.Ref | None,
        q_window_ref: jax.Ref,
        kv_window_ref: jax.Ref,
        q_sems: jax.Ref,
        kv_sems: jax.Ref,
        zero_hbm_refs: tuple[jax.Ref, jax.Ref],
        # Outputs.
        o_hbm_ref: jax.Ref,
        o_kv_cache_hbm_ref: jax.Ref,
        o_lse_hbm_ref: jax.Ref | None,
        o_schedule_chunk_refs: tuple[jax.Ref, ...],
        o_q_window_ref: jax.Ref,
        o_kv_window_ref: jax.Ref,
    ):
        # The aliased outputs are the same buffers as their inputs.
        del o_kv_cache_hbm_ref, o_lse_hbm_ref, o_schedule_chunk_refs
        del o_q_window_ref, o_kv_window_ref
        del zero_hbm_refs  # Only kept alive for the prefetch kernel's DMAs.

        q_alloc, kv_cache_alloc, o_alloc, lse_alloc = create_allocs(
            kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs
        )
        # The Q / KV windows and receive semaphores come from the prefetch kernel;
        # only the KV writeback semaphores are allocated here.
        q_alloc = dataclasses.replace(q_alloc, window_ref=None, sem_recvs=None)
        kv_cache_alloc = dataclasses.replace(
            kv_cache_alloc, window_ref=None, sem_recvs=None
        )

        actual_steps = actual_steps_ref[0]

        @pl.with_scoped(
            final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
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
        def _run(final_allocs, dma_sem, scratches):
            q_alloc, kv_cache_alloc, o_alloc, lse_alloc = final_allocs
            q_alloc = dataclasses.replace(
                q_alloc, window_ref=q_window_ref, sem_recvs=q_sems
            )
            kv_cache_alloc = dataclasses.replace(
                kv_cache_alloc, window_ref=kv_window_ref, sem_recvs=kv_sems
            )

            def post_init(is_first):
                @pl.when(is_first)
                def _():
                    scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
                    scratches[1][...] = jnp.zeros_like(scratches[1])
                    scratches[2][...] = jnp.zeros_like(scratches[2])

            def run_schedule_pipeline(schedule_view, total_steps, is_first):
                kv_bound = kv_cache_alloc.bind(
                    src_ref=(
                        kv_cache_hbm_ref,
                        new_kv_hbm_ref,
                        schedule_view,
                        page_indices_ref,
                    ),
                    dst_ref=(
                        kv_cache_hbm_ref,
                        new_kv_hbm_ref,
                        schedule_view,
                        page_indices_ref,
                    ),
                )
                q_bound = q_alloc.bind(src_ref=(q_hbm_ref, schedule_view))
                o_bound = o_alloc.bind(dst_ref=(o_hbm_ref, schedule_view))
                lse_bound = (
                    lse_alloc.bind(dst_ref=(lse_hbm_ref, schedule_view))
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
                        q_alloc.window_ref[slot],
                        kv_cache_alloc.window_ref[slot],
                        o_alloc.window_ref[out_slot],
                        lse_alloc.window_ref[out_slot] if return_lse else None,
                        schedule_view,
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
                    is_first=is_first,
                    post_init_fn=lambda: post_init(is_first),
                )

            # The SMEM schedule buffers are the XLA-owned outputs of
            # `rpa_prefetch_start`, which filled them with the first chunk before
            # this kernel started; later chunks are DMA'd into the same buffers.
            flat_hbm, schedule_treedef = jax.tree_util.tree_flatten(schedule_hbm_ref)
            hbm_leaves = [
                h for h in flat_hbm if jax.typeof(h).memory_space == pltpu.HBM
            ]
            chunk_iter = iter(schedule_chunk_refs)
            schedule_smem = schedule_treedef.unflatten(
                [
                    next(chunk_iter) if jax.typeof(h).memory_space == pltpu.HBM else h
                    for h in flat_hbm
                ]
            )

            def fetch_schedule_chunk(start_step, num_steps):
                # `start_step` is a multiple of `max_steps_ub`, which is a multiple of
                # 128, so reads are 128-aligned.
                dma_list = []
                for h, s in zip(hbm_leaves, schedule_chunk_refs):
                    element_size = s.shape[0] // cfgs.max_steps_ub
                    read_size = schedule.schedule_chunk_read_size(
                        s.shape[0], cfgs.max_steps_ub, num_steps
                    )

                    src_off = element_size * start_step
                    src_off = pl.multiple_of(src_off, 128)

                    copy = pltpu.make_async_copy(
                        h[pl.ds(src_off, read_size)],
                        s[pl.ds(0, read_size)],
                        dma_sem[0],
                    )
                    copy.start()
                    dma_list.append(copy)
                jax.tree.map(lambda x: x.wait(), dma_list)

            @pl.loop(0, pl.cdiv(actual_steps, cfgs.max_steps_ub))
            def _(chunk_idx):
                start = chunk_idx * cfgs.max_steps_ub
                size = jnp.minimum(cfgs.max_steps_ub, actual_steps - start)
                is_first = chunk_idx == 0

                # Chunk 0 was prefetched into the SMEM buffers by `rpa_prefetch_start`.
                @pl.when(jnp.logical_not(is_first))
                def _():
                    fetch_schedule_chunk(start, size)

                run_schedule_pipeline(schedule_smem, size, is_first)

        _run()

    # `actual_steps` comes from the prefetch kernel's SMEM output rather than as
    # a schedule leaf with an SMEM BlockSpec: such an operand is DMA'd into SMEM
    # at kernel entry, and that DMA queues behind the prefetch's VMEM DMAs.
    schedule_hbm = dataclasses.replace(schedule_hbm, actual_steps=None)
    scalar_prefetches = tuple(prefetch.metadata_smem) + (
        tuple(prefetch.schedule_chunk_smem),
        prefetch.actual_steps_smem,
    )
    extra_in_specs = [
        pl.BlockSpec(memory_space=pltpu.VMEM),  # q_window
        pl.BlockSpec(memory_space=pltpu.VMEM),  # kv_window
        pl.BlockSpec(memory_space=pltpu.SEMAPHORE),  # q_sems
        pl.BlockSpec(memory_space=pltpu.SEMAPHORE),  # kv_sems
        tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in zero_fill_bufs),
    ]
    extra_inputs = (
        prefetch.q_window,
        prefetch.kv_window,
        prefetch.q_sems,
        prefetch.kv_sems,
        tuple(zero_fill_bufs),
    )
    num_scalar_prefetch = len(scalar_prefetches)
    num_active_scalers = len(jax.tree_util.tree_leaves(scalar_prefetches))

    # The kernel DMAs later schedule chunks into the SMEM chunk buffers in place,
    # and keeps filling the input windows, so both are aliased input -> output.
    chunk_out_shape = tuple(
        pltpu.SMEM(c.shape, c.dtype) for c in prefetch.schedule_chunk_smem
    )
    window_out_shape = (
        pltpu.VMEM(prefetch.q_window.shape, prefetch.q_window.dtype),
        pltpu.VMEM(prefetch.kv_window.shape, prefetch.kv_window.dtype),
    )
    out_shape = [
        q_hbm,
        kv_cache_hbm,
        lse_hbm if return_lse else None,
        chunk_out_shape,
        *window_out_shape,
    ]

    schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
    base = num_active_scalers + schedule_leaves
    input_output_aliases = {base: 0, base + 2: 1}
    if return_lse:
        input_output_aliases[base + 3] = 2  # lse_hbm -> out[2]
    num_hbm_outs = 3 if return_lse else 2
    num_chunks = len(prefetch.schedule_chunk_smem)
    for j in range(num_chunks):
        input_output_aliases[len(prefetch.metadata_smem) + j] = num_hbm_outs + j
    window_in = base + (4 if return_lse else 3)
    input_output_aliases[window_in] = num_hbm_outs + num_chunks
    input_output_aliases[window_in + 1] = num_hbm_outs + num_chunks + 1

    # The input windows live outside this kernel's scoped VMEM but share the
    # same VMEM pool, so the scoped window shrinks by their size. XLA's pool is
    # a bit smaller than `vmem_capacity_bytes` (65472 vs 65536 KiB on v7) and
    # other small VMEM buffers can be live too, hence the margin.
    window_bytes = sum(
        w.size * w.dtype.itemsize for w in (prefetch.q_window, prefetch.kv_window)
    )
    vmem_limit_bytes = cfgs.vmem_limit_bytes - window_bytes - _VMEM_MARGIN_BYTES

    o_out, kv_out, lse_out, *_ = pl.pallas_call(
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
                *extra_in_specs,
            ],
            out_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_o_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM),  # aliased_kv_cache_hbm_ref
                pl.BlockSpec(memory_space=pltpu.HBM) if return_lse else None,
                tuple(
                    pl.BlockSpec(memory_space=pltpu.SMEM) for _ in chunk_out_shape
                ),  # aliased schedule chunk buffers
                pl.BlockSpec(memory_space=pltpu.VMEM),  # aliased q_window
                pl.BlockSpec(memory_space=pltpu.VMEM),  # aliased kv_window
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
            flags={
                "XLA_TPU_EVENLY_DISTRIBUTE_CRITICAL_NODES": True,
            },
        ),
        input_output_aliases=input_output_aliases,
        name=get_kernel_name(cfgs),
        metadata=get_kernel_metadata(cfgs),
    )(
        *scalar_prefetches,
        schedule_hbm,
        q_hbm,
        new_kv_hbm,
        kv_cache_hbm,
        lse_hbm if return_lse else None,
        *extra_inputs,
    )
    return o_out, kv_out, lse_out
