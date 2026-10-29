import dataclasses
from typing import Any, Callable

import jax
from jax import lax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from vllm_torchtpu.kernels.experimental.batched_rpa_new import (
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

    o_u32_vref = o_vref.at[b_idx].bitcast(jnp.uint32)
    out_ref = o_u32_vref.reshape(-1, cfgs.aligned_q_head_dim)
    pad_width = [[0, 0] for _ in range(out.ndim)]
    pad_width[-1][-1] = cfgs.aligned_q_head_dim - cfgs.compute_kv_head_dim
    out = jnp.pad(out, pad_width, constant_values=0)
    out = pltpu.bitcast(out, out_ref.dtype).reshape(out_ref.shape)
    utils.strided_store(out_ref, 0, out_ref.shape[0], 1, out)

  def _stage_lse(b_idx: int, batch_m: jax.Array, batch_l: jax.Array):
    lse_val = batch_m + jnp.log(jnp.maximum(batch_l, 1e-9))
    lse_val = lse_val.astype(cfgs.serve.dtype_out)
    if cfgs.serve.is_packed_lse:
      kv_heads, _, lanes = lse_val.shape
      num_heads = cfgs.aligned_num_q_heads_per_kv_head
      # One row per token, heads along lanes (zero-padded).
      lse_heads = lse_val[..., 0].reshape(kv_heads, cfgs.bq_sz, num_heads)
      lse_o_vref[b_idx] = jnp.pad(
          lse_heads, ((0, 0), (0, 0), (0, lanes - num_heads))
      )
    else:
      lse_u32 = lse_o_vref.at[b_idx].bitcast(jnp.uint32)
      lse_ref = lse_u32.reshape(-1, lse_o_vref.shape[-1])
      lse_val = pltpu.bitcast(lse_val, jnp.uint32).reshape(lse_ref.shape)
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
  bkv_sz_frm_new: list[jax.Array]
  new_kv_len_start: list[jax.Array]


ComputeMetadata = schedule.ComputeMetadata


# The `pl.when` region in `rpa_body` is a basic block boundary, so splitting
# these keeps the mask arithmetic isolated to the bodies that need it.
def fetch_stitch_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    kv_new_starts_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
) -> StitchMetadata:
  """Fetches the scalars consumed by the KV stitch.

  Args:
    step: Current pipeline step.
    schedule_ref: Per-step sequence, query, and KV block indices.
    kv_cache_lens_ref: Cached KV length per sequence.
    kv_new_lens_ref: Newly arrived KV length per sequence.
    kv_new_starts_ref: Offset of each sequence's new KV in `new_kv_hbm`.
    cfgs: Kernel configs.

  Returns:
    The stitch boundaries for every batch element in this step.
  """
  bkv_sz_frm_cache_list = []
  bkv_sz_frm_new_list = []
  new_kv_len_start_list = []

  for b_idx in range(cfgs.batch_size):
    s_idx = schedule_ref.s_idx[step, b_idx]
    is_valid = s_idx != -1
    k_idx = schedule_ref.k_idx[step, b_idx]
    k_id = jnp.where(is_valid, k_idx * cfgs.bkv_sz, 0)
    kv_cache_len_val = jnp.where(is_valid, kv_cache_lens_ref[s_idx], 0)
    kv_new_len_val = jnp.where(is_valid, kv_new_lens_ref[s_idx], 0)
    kv_new_start_val = jnp.where(is_valid, kv_new_starts_ref[s_idx], 0)

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
    bkv_sz_frm_new_list.append(
        jnp.minimum(kv_left, cfgs.bkv_sz) - bkv_sz_frm_cache
    )
    new_kv_len_start_list.append(
        kv_new_start_val + kv_new_len_val - kv_left_frm_new
    )

  return StitchMetadata(
      bkv_sz_frm_cache=bkv_sz_frm_cache_list,
      bkv_sz_frm_new=bkv_sz_frm_new_list,
      new_kv_len_start=new_kv_len_start_list,
  )


def fetch_compute_metadata(
    step: jax.Array,
    schedule_ref: schedule.RpaSchedule,
    q_positions_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    cp_rank_ref: jax.Ref,
    *,
    cfgs: configs.RpaConfigs,
    kv_new_lens_ref: jax.Ref | None = None,
    cu_q_lens_ref: jax.Ref | None = None,
    mask_computer_cls: type[
        schedule.BaseMaskComputer
    ] = schedule.ContiguousCausalMask,
) -> ComputeMetadata:
  """Fetches the scalars consumed by the QK mask."""
  return mask_computer_cls.fetch_compute_metadata(
      step,
      schedule_ref,
      q_positions_ref,
      kv_cache_lens_ref,
      cp_rank_ref,
      cfgs=cfgs,
      kv_new_lens_ref=kv_new_lens_ref,
      cu_q_lens_ref=cu_q_lens_ref,
  )


def generate_mask(
    shape: tuple[int, int, int, int],
    *,
    bq_start: int,
    step_meta: ComputeMetadata,
    cfgs: configs.RpaConfigs,
    mask_computer_cls: type[
        schedule.BaseMaskComputer
    ] = schedule.ContiguousCausalMask,
) -> list[jax.Array]:
  """Generates causal, sliding window, and valid-KV mask for QK computation."""
  return mask_computer_cls.generate_mask(
      shape,
      bq_start=bq_start,
      step_meta=step_meta,
      cfgs=cfgs,
  )


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
    q_positions_ref: jax.Ref,
    kv_cache_lens_ref: jax.Ref,
    kv_new_lens_ref: jax.Ref,
    kv_new_starts_ref: jax.Ref,
    cp_rank_ref: jax.Ref,
    attention_sink_ref: jax.Ref | None = None,
    # Configs.
    cfgs: configs.RpaConfigs,
    # DMA-issue hooks. Both only start copies; the matching waits live in
    # `pipeline.py`
    start_next_fetch: Callable[[], None],
    start_kv_writeback: Callable[[], None],
    ring: Any = None,
    total_steps: int | jax.Array | None = None,
    computer_cls: type[
        schedule.BaseMetadataComputer
    ] = schedule.BaseMetadataComputer,
):
  del slot

  kv_in_vref = pltpu.annotate(kv_in_vref, no_hazard_no_deps=True)

  # Step 1: Fetch the metadata the stitch needs. Everything the mask needs is
  # deferred until after the fence below, so it does not gate the writeback.
  stitch_meta = (
      fetch_stitch_metadata(
          step,
          schedule_ref,
          kv_cache_lens_ref,
          kv_new_lens_ref,
          kv_new_starts_ref,
          cfgs=cfgs,
      )
      if cfgs.serve.has_new_kv
      else None
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
    if (
        stitch_meta is not None
        and cfgs.serve.kv_layout == configs.KVLayout.SEQ_ALONG_LANE
    ):
      stitch_results = []
      for b_idx in range(cfgs.batch_size):
        res = stitch_utils.stitch_new_kv_lane(
            kv_in_vref,
            b_idx,
            stitch_meta.bkv_sz_frm_cache[b_idx],
            stitch_meta.new_kv_len_start[b_idx],
            stitch_meta.bkv_sz_frm_new[b_idx],
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

  # The prefetch below reuses the slot the ring forwarded last step.
  if ring is not None:
    ring.wait_prev_send(step, schedule_ref)
  start_next_fetch()

  # Step 2a: Fetch mask metadata after the fence to overlap with compute and
  # in-flight writeback.
  mask_computer_cls = computer_cls.mask_computer_cls
  step_meta = fetch_compute_metadata(
      step,
      schedule_ref,
      q_positions_ref,
      kv_cache_lens_ref,
      cp_rank_ref,
      cfgs=cfgs,
      kv_new_lens_ref=kv_new_lens_ref,
      cu_q_lens_ref=cu_q_lens_ref,
      mask_computer_cls=mask_computer_cls,
  )

  # Step 2b: Take this step's rotated KV block and forward it to the next rank,
  # so the hop overlaps the attention math below.
  if ring is not None:
    ring.receive_and_forward(step, schedule_ref, total_steps)

  # Step 2c: Fetch inputs.
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
        k_slice = kv_in_vref.at[b_idx, kv_head * 2].bitcast(jnp.uint32)
        v_slice = kv_in_vref.at[b_idx, kv_head * 2 + 1].bitcast(jnp.uint32)

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
  sink_val = None
  if attention_sink_ref is not None:
    sink_val = jnp.concat([attention_sink_ref[...]] * cfgs.bq_c_sz, axis=1)

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
        mask_computer_cls=mask_computer_cls,
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
            attention_sink=sink_val,
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
          prev_alpha_list,
          acc_val[:, prev_q_slice],
          step,
          schedule_ref.is_last_k,
          cfgs=cfgs,
          v_scale=v_scale,
          attention_sink=sink_val,
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
      prev_alpha_list,
      acc_val[:, prev_q_slice],
      step,
      schedule_ref.is_last_k,
      cfgs=cfgs,
      v_scale=v_scale,
      attention_sink=sink_val,
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

  if ring is not None:
    ring.finish(step, schedule_ref, total_steps)


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
  if not serve.has_cache_kv:
    name += "-new_only"
  elif not serve.has_new_kv:
    name += "-cache_only"
  else:
    name += "-full"
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
    q_positions: jax.Array,
    kv_cache_lens: jax.Array,
    kv_new_lens: jax.Array,
    kv_new_starts: jax.Array,
    cp_rank: jax.Array,
    page_indices: jax.Array,
    schedule_hbm: schedule.RpaSchedule,
    q_hbm: jax.Array,
    new_kv_hbm: jax.Array,
    kv_cache_hbm: jax.Array,
    lse_hbm: jax.Array | None,
    attention_sink_hbm: jax.Array | None,
    *,
    cfgs: configs.RpaConfigs,
    computer_cls: type[
        schedule.BaseMetadataComputer
    ] = schedule.BaseMetadataComputer,
) -> tuple[jax.Array, jax.Array, jax.Array | None]:
  """Perform batched ragged paged attention with scheduler data.

  Args:
    cu_q_lens: [max_num_seqs + 1]. Cumulative sum of each sequence's query
      length. queries[a:b], keys[a:b], and values[a:b] where a=cu_q_lens[i] and
      b=cu_q_lens[i+1] represents q/k/v of sequence i.
    q_positions: [max_num_seqs]. Position of each sequence's first query token
      in the sequence's KV index space.
    kv_cache_lens: [max_num_seqs]. Existing kv cache length of each sequence.
    kv_new_lens: [max_num_seqs]. New kv length of each sequence.
    kv_new_starts: [max_num_seqs]. Offset of each sequence's new KV in
      `new_kv_hbm`.
    cp_rank: [1]. This device's rank in the CP group; 0 without CP.
    page_indices: [max_num_seqs * pages_per_seqs]. kv cache page table of each
      sequence.
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
    attention_sink_hbm: [num_kv_heads, aligned_num_q_heads_per_kv_head,
      num_lanes]. Per q head sink logit, or None.
    cfgs: Configuration of the kernel.
    computer_cls: Metadata computer class that generated `schedule_hbm`.

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
      q_positions_ref: jax.Ref,
      kv_cache_lens_ref: jax.Ref,
      kv_new_lens_ref: jax.Ref,
      kv_new_starts_ref: jax.Ref,
      cp_rank_ref: jax.Ref,
      page_indices_ref: jax.Ref,
      # Inputs.
      schedule_hbm_ref: schedule.RpaSchedule,
      q_hbm_ref: jax.Ref,
      new_kv_hbm_ref: jax.Ref,
      kv_cache_hbm_ref: jax.Ref,
      lse_hbm_ref: jax.Ref | None,
      attention_sink_ref: jax.Ref | None,
      # Outputs: o_hbm_ref, then the aliased KV cache (when written) and LSE.
      o_hbm_ref: jax.Ref,
      *aliased_out_refs: jax.Ref | None,
  ):
    del aliased_out_refs

    q_alloc, kv_cache_alloc, o_alloc, lse_alloc = create_allocs(
        kv_cache_hbm_ref, q_hbm_ref, lse_hbm_ref, cfgs
    )

    actual_steps = schedule_hbm_ref.actual_steps[0]
    num_safe_step_iterations = pl.cdiv(actual_steps, cfgs.max_steps_ub)

    @pl.with_scoped(
        final_allocs=(q_alloc, kv_cache_alloc, o_alloc, lse_alloc),
        schedule_ref=computer_cls.get_rpa_schedule(cfgs).scratch_shapes(),
        dma_sem=pltpu.SemaphoreType.DMA((1,)),
        ring_sems=computer_cls.get_ring_sems_spec(),
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
    def _run(final_allocs, schedule_ref, dma_sem, ring_sems, scratches):
      ring = computer_cls.create_ring(
          cfgs, ring_sems, kv_window_ref=final_allocs[1].window_ref
      )
      if ring is not None:
        ring.initial_handshake()
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

      if attention_sink_ref is None:
        scratches[0][...] = jnp.full_like(scratches[0], -jnp.inf)
        scratches[1][...] = jnp.zeros_like(scratches[1])
      else:
        scratches[0][...] = jnp.concat(
            [attention_sink_ref[...]] * cfgs.bq_sz, axis=1
        )
        scratches[1][...] = jnp.ones_like(scratches[1])

      q_alloc, kv_cache_alloc, o_alloc, lse_alloc = final_allocs

      num_lanes = pltpu.get_tpu_info().num_lanes
      q_ref_flat = q_alloc.window_ref.bitcast(jnp.uint32).reshape(-1, num_lanes)
      kv_ref_flat = kv_cache_alloc.window_ref.bitcast(jnp.uint32).reshape(
          -1, num_lanes
      )

      # Explicitly zero out scratches and Q/KV buffers.
      for ref in (scratches[2], q_ref_flat, kv_ref_flat):
        ref[...] = jnp.zeros_like(ref)

      def execute_schedule_chunk(start_step, num_steps):
        # All reads are aligned to 128 and some extra steps are copied in the
        # process.
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
              q_positions_ref=q_positions_ref,
              kv_cache_lens_ref=kv_cache_lens_ref,
              kv_new_lens_ref=kv_new_lens_ref,
              kv_new_starts_ref=kv_new_starts_ref,
              cp_rank_ref=cp_rank_ref,
              attention_sink_ref=attention_sink_ref,
              cfgs=cfgs,
              start_next_fetch=start_next_fetch,
              start_kv_writeback=start_kv_writeback,
              ring=ring,
              total_steps=total_steps,
              computer_cls=computer_cls,
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

      if ring is not None:
        ring.drain_credits()

    _run()

  scalar_prefetches = (
      cu_q_lens,
      q_positions,
      kv_cache_lens,
      kv_new_lens,
      kv_new_starts,
      cp_rank,
      page_indices,
  )
  num_scalar_prefetch = len(scalar_prefetches)
  num_active_scalers = len(jax.tree_util.tree_leaves(scalar_prefetches))

  # A pass without new KV (PCP's cache pass) only reads the cache, so it does
  # not alias it: aliasing makes XLA copy a cache it may not overwrite.
  writes_kv_cache = cfgs.serve.has_new_kv
  out_shape = [q_hbm]
  out_specs = [pl.BlockSpec(memory_space=pltpu.HBM)]  # aliased_o_hbm_ref
  schedule_leaves = len(jax.tree_util.tree_leaves(schedule_hbm))
  q_operand = num_active_scalers + schedule_leaves
  input_output_aliases = {q_operand: 0}
  if writes_kv_cache:
    input_output_aliases[q_operand + 2] = len(out_shape)  # kv_cache_hbm
    out_shape.append(kv_cache_hbm)
    out_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
  if return_lse:
    input_output_aliases[q_operand + 3] = len(out_shape)  # lse_hbm
    out_shape.append(lse_hbm)
    out_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))

  outs = pl.pallas_call(
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
              pl.BlockSpec(memory_space=pltpu.VMEM)
              if attention_sink_hbm is not None
              else None,
          ],
          out_specs=out_specs,
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
      q_positions,
      kv_cache_lens,
      kv_new_lens,
      kv_new_starts,
      cp_rank,
      page_indices,
      schedule_hbm,
      q_hbm,
      new_kv_hbm,
      kv_cache_hbm,
      lse_hbm if return_lse else None,
      None
      if attention_sink_hbm is None
      else attention_sink_hbm.astype(cfgs.serve.dtype_out),
  )
  outs = list(outs)
  o_out = outs.pop(0)
  kv_out = outs.pop(0) if writes_kv_cache else kv_cache_hbm
  lse_out = outs.pop(0) if return_lse else None
  return o_out, kv_out, lse_out
