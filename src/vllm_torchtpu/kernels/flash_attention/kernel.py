# Copyright 2025 Google LLC
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
"""Flash Attention TPU kernel."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.flash_attention.tuned_params import (
    get_tuned_params,
    make_tuning_key,
)
from vllm_torchtpu.utils import align_to

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
NUM_LANES = 128
NUM_SUBLANES = 8


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are used to generate segment mask, which prevents attention between
    different segments in the input sequence. Each array is a list of ids
    (integers).
    Only the token with the same id can attend to each other.

    Attributes:
      q: segment ids along the Q sequence.
      kv: segment ids along the KV sequence.
    """

    q: jax.Array  # [batch_size, q_seq_len]
    kv: jax.Array  # [batch_size, kv_seq_len]


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tile sizes parameterizing FlashAttention kernels.

    Those parameters have negligible effect on numerics, but affect performance
    greatly.
    """

    block_q: int
    block_k_major: int
    block_k: int
    block_b: int
    # With segment ids: walk only each row group's kv window, in row groups of
    # block_q_group rows. Pays off on packed inputs with many segments (vision
    # buckets of several images); on one long segment it is slower, so
    # only the shapes measured with it turn it on.
    kv_windows: bool = False
    # The kv-window walk's row group and the chunk that covers what block_k
    # chunks leave of a window (at block_k 2048, rounding a window up to
    # block_k wastes 1024 columns on average). Defaults: the Cosmos3 2x72
    # measurements.
    block_q_group: int = 1024
    block_k_tail: int = 512

    def __post_init__(self):
        def verify_major_minor(prefix, suffix, major, minor):
            if minor > major:
                raise ValueError(
                    f"{prefix}{suffix}={minor} should be smaller than"
                    f" {prefix}_major{suffix}={major}"
                )
            if major % minor != 0:
                raise ValueError(
                    f"{prefix}{suffix}={minor} should divide"
                    f" {prefix}_major{suffix}={major}"
                )

        verify_major_minor("block_k", "", self.block_k_major, self.block_k)

    @classmethod
    def get_default(cls, batch_size, num_heads, q_seq_len, kv_len, d_model):
        # TODO(apaszke,sharadmv): Select better parameters based on a heuristic.
        del batch_size, num_heads, q_seq_len, kv_len, d_model  # Unused.
        return BlockSizes(
            block_q=128,
            block_k_major=128,
            block_k=128,
            block_b=1,
        )


def calculate_vmem_usage_bytes(
    block_sizes: BlockSizes,
    q_dtype,
    kv_dtype,
    head_dim: int,
    kv_seq_len: int,
    ab=None,
    segment_ids=None,
) -> int:
    """Estimate VMEM needed by the whole-KV single-step kernel."""
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    num_sublanes = tpu_info.num_sublanes
    aligned_head_dim = align_to(head_dim, num_lanes)

    q_elements = block_sizes.block_b * block_sizes.block_q * aligned_head_dim
    kv_elements = block_sizes.block_b * kv_seq_len * aligned_head_dim
    logits_elements = block_sizes.block_q * kv_seq_len

    q_and_output = q_elements * jnp.dtype(q_dtype).itemsize * 2
    k_and_v = kv_elements * jnp.dtype(kv_dtype).itemsize * 2
    logits_and_probs = logits_elements * jnp.dtype(jnp.float32).itemsize * 2

    attention_bias = 0
    if ab is not None:
        attention_bias = (
            block_sizes.block_b
            * block_sizes.block_q
            * kv_seq_len
            * jnp.dtype(ab.dtype).itemsize
        )

    segments = 0
    if segment_ids is not None:
        segment_bytes = jnp.dtype(jnp.int32).itemsize
        segments = (
            block_sizes.block_b * block_sizes.block_q * num_lanes * segment_bytes
            + block_sizes.block_b * num_sublanes * kv_seq_len * segment_bytes
        )

    return q_and_output + k_and_v + logits_and_probs + attention_bias + segments


@functools.partial(
    jax.jit,
    static_argnames=[
        "causal",
        "sm_scale",
        "block_sizes",
        "vmem_limit_bytes",
        "debug",
    ],
)
def flash_attention(
    q,  # [batch_size, num_heads, q_seq_len, d_model]
    k,  # [batch_size, num_heads, kv_seq_len, d_model]
    v,  # [batch_size, num_heads, kv_seq_len, d_model]
    ab=None,  # [batch_size, num_heads, q_seq_len, kv_seq_len]
    segment_ids=None,  # q of [batch_size, q_seq_len] and kv of [batch_size, kv_seq_len]
    *,
    causal: bool = False,
    sm_scale: float = 1.0,
    block_sizes: BlockSizes | None = None,
    vmem_limit_bytes: int | None,
    debug: bool = False,
):
    batch_size, num_heads, q_seq_len, d_model = q.shape
    batch_size_k, num_heads_k, kv_seq_len, d_model_k = k.shape
    batch_size_v, num_heads_v, kv_seq_len_v, d_model_v = v.shape
    if batch_size != batch_size_k or batch_size != batch_size_v:
        raise ValueError(
            f"Batch size mismatch: got {batch_size}, {batch_size_k} and"
            f" {batch_size_v} (for q, k, v respectively)"
        )
    if num_heads != num_heads_k or num_heads != num_heads_v:
        raise ValueError(
            f"Head count mismatch: got {num_heads}, {num_heads_k},"
            f" {num_heads_v} (for q, k, v respectively)"
        )
    if d_model != d_model_k:
        raise ValueError(
            f"Model dimension mismatch: got {d_model} and {d_model_k} (for q and k"
            " respectively)"
        )
    if d_model != d_model_v:
        raise NotImplementedError(
            "V model dimension unequal to KV model dimension unsupported"
        )
    if kv_seq_len != kv_seq_len_v:
        raise ValueError(
            f"KV sequence length mismatch: got {kv_seq_len} and {kv_seq_len_v}"
        )
    if ab is not None:
        if ab.shape != (batch_size, num_heads, q_seq_len, kv_seq_len):
            raise ValueError(
                f"Attention bias shape mismatch: expected ({batch_size=},"
                f" {num_heads=}, {q_seq_len=}, {kv_seq_len=}), got {ab.shape}"
            )
    if segment_ids is not None:
        if segment_ids.q.shape != (batch_size, q_seq_len):
            raise ValueError(
                f"Q segment ids shape mismatch: expected ({batch_size=},"
                f" {q_seq_len=},), got {segment_ids.q.shape}"
            )
        if segment_ids.kv.shape != (batch_size, kv_seq_len):
            raise ValueError(
                f"KV segment ids shape mismatch: expected ({batch_size=},"
                f" {kv_seq_len=},), got {segment_ids.kv.shape}"
            )
    if block_sizes is None:
        tuning_key = make_tuning_key(
            q,
            k,
            v,
            causal=causal,
            has_segment_ids=segment_ids is not None,
            has_attention_bias=ab is not None,
            vmem_limit_bytes=vmem_limit_bytes,
        )
        tuned_params = get_tuned_params(tuning_key)
        if tuned_params is None:
            block_sizes = BlockSizes.get_default(
                batch_size, num_heads, q_seq_len, kv_seq_len, d_model
            )
            estimated_vmem = calculate_vmem_usage_bytes(
                block_sizes, q.dtype, k.dtype, d_model, kv_seq_len, ab, segment_ids
            )
            vmem_limit = (
                pltpu.get_tpu_info().vmem_capacity_bytes
                if vmem_limit_bytes is None
                else vmem_limit_bytes
            )
            if estimated_vmem <= vmem_limit * 0.9:
                block_sizes = BlockSizes(
                    block_q=block_sizes.block_q,
                    block_b=block_sizes.block_b,
                    block_k_major=kv_seq_len,
                    block_k=kv_seq_len,
                )
        else:
            block_sizes = BlockSizes(**dataclasses.asdict(tuned_params))
    return _flash_attention(
        q,
        k,
        v,
        ab,
        segment_ids,
        False,
        causal,
        sm_scale,
        block_sizes,
        vmem_limit_bytes,
        debug,
    )


def _flash_attention(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_sizes,
    vmem_limit_bytes,
    debug,
):
    return _flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        save_residuals,
        causal,
        sm_scale,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
        # Only the multi-step body has a kv loop to window.
        (block_sizes.block_q_group, block_sizes.block_k_tail)
        if segment_ids is not None
        and block_sizes.kv_windows
        and block_sizes.block_k != k.shape[2]
        else None,
        vmem_limit_bytes,
        debug,
    )


MIN_BLOCK_SIZE = 128
TRANS_B_DIM_NUMBERS = (((1,), (1,)), ((), ()))


def below_or_on_diag(r, r_blk_size, c, c_blk_size):
    # A block is considered below or on diagonal as long as the bottom left
    # corner of the block is below or on diagonal.
    return ((r + 1) * r_blk_size - 1) > (c * c_blk_size)


def _flash_attention_kernel(*refs, kv_windows: tuple[int, int] | None, **kwargs):
    # With kv windows (row group, tail block_k), the scalar-prefetched kv
    # windows of every q block's row groups come first.
    kv_walk = None
    if kv_windows is not None:
        kv_windows_ref, *refs = refs
        kv_walk = (kv_windows_ref, *kv_windows)
    q_tile_ref, *args = refs
    block_b = q_tile_ref.shape[0]
    # If we're not going to tile the softmax, then we can avoid a bunch of VPU ops.
    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = functools.partial(
            _flash_attention_kernel_single_batch, kv_walk=kv_walk
        )
    for batch_idx in range(block_b):
        kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
    kv_walk: tuple[Any, int, int] | None = None,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = jnp.full(
            m_scratch_ref.shape[2:], -jnp.inf, jnp.float32
        )
        l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
        acc_scratch_ref[batch_idx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)

    q_seq_idx = pl.program_id(2)
    if causal:
        should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
    else:
        should_run = True

    # The kv-window body (BlockSizes.kv_windows) normalizes acc once, at the
    # store; the other calls keep the per-step normalization they were tuned
    # and validated with.
    deferred_norm = kv_walk is not None
    head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)
    l_broadcast = lambda l: jnp.tile(l, (1, head_dim_repeats))
    if rem:
        if head_dim_repeats == 0:
            l_broadcast = lambda l: l[:, :head_dim]
        else:
            raise NotImplementedError(
                f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger"
            )

    @pl.when(should_run)
    def run():
        def _attend_rows(start_k, block_k, row, rows):
            rs = pl.dslice(row, rows)
            m_prev = m_scratch_ref[(*batch_idx, rs)]
            l_prev = l_scratch_ref[(*batch_idx, rs)]
            q = q_tile_ref[(*batch_idx, rs)]  # [rows, head_dim]
            k = k_tile_ref[
                (*batch_idx, pl.dslice(start_k, block_k), slice(None))
            ]  # [block_k, head_dim]

            s = jax.lax.dot_general(
                q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
            )  # [rows, block_k]

            # Add attention bias if needed.
            # TODO(tanburn) Should the attention bias be added before or after
            # multiplication by sm_scale?
            if ab_tile_ref is not None:
                ab = ab_tile_ref[(*batch_idx, rs, pl.dslice(start_k, block_k))].astype(
                    jnp.float32
                )
                s += ab

            if sm_scale != 1.0:
                s *= sm_scale

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(
                        f"kv block size must be a multiple of {NUM_LANES}"
                    )
                q_segment_ids = jnp.tile(
                    q_segment_ids_tile_ref[batch_idx[0], rs], (1, repeats)
                )  # [rows, block_k].
                kv_segment_ids = kv_segment_ids_tile_ref[
                    batch_idx[0], :1, pl.dslice(start_k, block_k)
                ]  # [1, block_k].
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (rows, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_idx * block_q + row
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_idx * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = (
                    causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
                )

            s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]  # Row max, shape [rows, 1].
            m_next = jnp.maximum(m_prev, m_curr)  # Shape [rows, 128].

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(
                    f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}"
                )
            p = jnp.exp(s - jnp.tile(m_next, (1, block_k_repeats)))

            alpha = jnp.exp(m_prev - m_next)  # Shape [rows, 128].

            l_corr = alpha * l_prev
            l_next = jnp.sum(p, axis=1)[:, None] + l_corr

            l_scratch_ref[(*batch_idx, rs)] = l_next
            m_scratch_ref[(*batch_idx, rs)] = m_next

            v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
            o_curr = jax.lax.dot(
                p.astype(v.dtype), v, preferred_element_type=jnp.float32
            )
            if deferred_norm:
                # acc stays unnormalized until the store: normalizing every
                # step costs a reciprocal and two more passes over acc per kv
                # block.
                acc_scratch_ref[(*batch_idx, rs)] = (
                    acc_scratch_ref[(*batch_idx, rs)] * l_broadcast(alpha) + o_curr
                )
            else:
                l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
                acc_scratch_ref[(*batch_idx, rs)] *= l_broadcast(
                    l_corr * l_next_inv_safe
                )
                acc_scratch_ref[(*batch_idx, rs)] += o_curr * l_broadcast(
                    l_next_inv_safe
                )

        # With kv windows, row groups of at most block_q_group rows: the
        # scheduler overlaps one group's softmax with the next group's matmuls,
        # which a single [block_q, block_k] tile serializes.
        group = block_q if kv_walk is None else kv_walk[1]
        groups = [(row, min(group, block_q - row)) for row in range(0, block_q, group)]

        def _attend(start_k, block_k=block_k, groups=groups):
            for row, rows in groups:
                _attend_rows(start_k, block_k, row, rows)

        if kv_walk is None:
            pl.loop(0, block_k_major, step=block_k, unroll=True)(_attend)
            return
        kv_windows_ref, _, tail_block_k = kv_walk

        # A partial last batch tile's rows past the batch are dropped; they
        # read the last row's windows to stay inside the table.
        batch = jnp.minimum(
            pl.program_id(0) * q_tile_ref.shape[0] + batch_idx[0],
            kv_windows_ref.shape[0] - 1,
        )
        kv_start = kv_seq_idx * block_k_major
        tail_k = math.gcd(block_k, block_k_major, tail_block_k)

        def _window(g, i):
            return kv_windows_ref[batch, (q_seq_idx * len(groups) + g) * 2 + i]

        def _walk(g, groups):
            # Only row group g's kv window [lo, hi) can hold a matching
            # segment. Cover its part in this kv block with block_k chunks and
            # the rest with tail_k ones, placed inside the kv block: columns
            # they add past the window are masked like any other mismatch.
            lo = jnp.maximum(_window(g, 0), kv_start)
            hi = jnp.minimum(_window(g, 1), kv_start + block_k_major)
            width = jnp.maximum(hi - lo, 0)
            if tail_k == block_k:
                num_main, num_tail = pl.cdiv(width, block_k), 0
            else:
                num_main = width // block_k
                num_tail = pl.cdiv(width - num_main * block_k, tail_k)
            # tail_k divides block_k and block_k_major, so the chunks fit.
            first = jnp.minimum(
                lo - kv_start, block_k_major - num_main * block_k - num_tail * tail_k
            )

            @pl.loop(0, num_main)
            def _main(i):
                _attend(pl.multiple_of(first + i * block_k, NUM_LANES), groups=groups)

            if tail_k != block_k:

                @pl.loop(0, num_tail)
                def _tail(i):
                    start = first + num_main * block_k + i * tail_k
                    _attend(pl.multiple_of(start, NUM_LANES), tail_k, groups)

        # Row groups with one window walk it together; a q block that
        # straddles a segment boundary walks each group's own window instead
        # of their union.
        same = functools.reduce(
            jnp.logical_and,
            [
                _window(g, i) == _window(0, i)
                for g in range(1, len(groups))
                for i in range(2)
            ],
            jnp.bool_(True),
        )
        pl.when(same)(lambda: _walk(0, groups))
        if len(groups) > 1:

            @pl.when(jnp.logical_not(same))
            def _apart():
                for g, group in enumerate(groups):
                    _walk(g, [group])

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        acc = acc_scratch_ref[batch_idx]
        if deferred_norm:
            l = l_scratch_ref[batch_idx]
            acc *= l_broadcast(jnp.where(l == 0.0, 1.0, 1.0 / l))
        o_tile_ref[batch_idx] = acc.astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


# ruff: noqa #731
# ruff: noqa #741
def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,  # Input arrays
    o_tile_ref,  # Output arrays
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    sm_scale,
    block_k,
    kv_seq_len,
    mask_value,
):
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]

    assert kv_seq_len == block_k_major == block_k

    q = q_tile_ref[batch_idx]  # [block_q, head_dim]
    k = k_tile_ref[batch_idx]  # [block_k, head_dim]
    s = jax.lax.dot_general(
        q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32
    )  # [block_q, block_k]

    if ab_tile_ref is not None:
        s += ab_tile_ref[batch_idx].astype(jnp.float32)
    if sm_scale != 1.0:
        s *= sm_scale

    mask = None
    if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
            raise NotImplementedError(
                f"kv block size must be a multiple of {NUM_LANES}"
            )
        q_segment_ids = q_segment_ids_tile_ref[batch_idx[0]]  # [block_q, NUM_LANES].
        q_segment_ids = jnp.tile(q_segment_ids, (1, repeats))  # [block_q, block_k].
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]  # [1, block_k].
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
        q_seq_idx = pl.program_id(2)
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

    m = jnp.max(s, axis=1)[:, None]
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1)[:, None]
    p /= l

    if m_ref is not None:
        m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
    if l_ref is not None:
        l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

    v = v_tile_ref[batch_idx]
    o_tile_ref[batch_idx] = jax.lax.dot(
        p.astype(v.dtype), v, preferred_element_type=jnp.float32
    ).astype(o_tile_ref.dtype)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct) -> int:
    return math.prod(x.shape) * x.dtype.itemsize


def _fwd_cost_estimate(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    ab: jax.Array | None,
    segment_ids: SegmentIds | None,
    *,
    causal: bool,
    sm_scale: jax.Array | None,
    kernel_inputs_specs,
    kernel_outputs_specs,
) -> pl.CostEstimate | None:
    body_cost = pl.estimate_cost(
        mha_reference, q, k, v, ab, segment_ids, causal=causal, sm_scale=sm_scale
    )
    input_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_inputs_specs))
    output_bytes = sum(_bytes(x) for x in jax.tree.leaves(kernel_outputs_specs))
    return pl.CostEstimate(
        flops=body_cost.flops,
        transcendentals=body_cost.transcendentals,
        bytes_accessed=input_bytes + output_bytes,
    )


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    sm_scale,
    block_b,
    block_q,
    block_k_major,
    block_k,
    kv_windows,
    vmem_limit_bytes,
    debug,
):
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    # TODO(apaszke): Tile over heads as well.
    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            # If the kv block is skipped, prefetch the next valid kv block, i.e. the
            # 0th one to be used for the next block_q rows.
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            should_run = below_or_on_diag(
                q_seq_index, block_q, kv_seq_index, block_k_major
            )
            # If the ab block is skipped, prefetch the next valid ab block, i.e. the
            # 0th kv to be used for the next block_q rows.
            next_q_index = lax.select(
                should_run,
                q_seq_index,
                lax.select(
                    q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1
                ),
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q_index = q_seq_index
            next_kv_index = kv_seq_index

        return (batch_index, head_index, next_q_index, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        kv_windows=kv_windows,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        sm_scale=sm_scale,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        ]
        l = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        m = jax.ShapeDtypeStruct(
            (batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32
        )
        out_shape = (*out_shape, l, m)
    else:
        out_specs = [*out_specs, None, None]
        out_shape = (*out_shape, None, None)

    ab_block_spec = (
        pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map)
        if ab is not None
        else None
    )

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(
            batch_index, head_index, q_seq_index, kv_seq_index
        ):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec(
            (block_b, block_q, NUM_LANES), q_segment_ids_index_map
        )
        kv_segment_ids_spec = pl.BlockSpec(
            (block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map
        )

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    windows = ()
    if kv_windows is not None:
        # Each row group's kv window, for the kernel to skip the pairs that
        # cannot match (packed vision inputs: most).
        windows = (_kv_windows(segment_ids, block_q, kv_windows[0]),)
        num_prefetch = len(windows)

        def with_prefetch(spec):
            if spec is None:
                return None
            index_map = spec.index_map
            return pl.BlockSpec(
                spec.block_shape, lambda *idx: index_map(*idx[:-num_prefetch])
            )

        in_specs = [with_prefetch(spec) for spec in in_specs]
        out_specs = [with_prefetch(spec) for spec in out_specs]

    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(windows),
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        debug=debug,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            ),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            ab,
            segment_ids,
            causal=causal,
            sm_scale=sm_scale,
            kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(*windows, q, k, v, ab, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o


def _block_ranges(segment_ids: jax.Array, block: int) -> jax.Array:
    """[batch, blocks, 2] int32: each block's lowest and highest segment id. A
    partial last block repeats its last id."""
    batch_size, seq_len = segment_ids.shape
    pad = -seq_len % block
    ids = jnp.pad(segment_ids, ((0, 0), (0, pad)), mode="edge")
    ids = ids.reshape(batch_size, -1, block).astype(jnp.int32)
    return jnp.stack([ids.min(axis=-1), ids.max(axis=-1)], axis=-1)


def _kv_windows(segment_ids: SegmentIds, block_q: int, row_group: int) -> jax.Array:
    """[batch, q blocks * row groups * 2] int32, flat for SMEM: for each row
    group of each q block, the kv columns [lo, hi) from the first to the last
    128-column block whose segment range meets the group's (empty: lo >= hi).
    Columns outside cannot match any of the group's ids; with sorted ids
    (packed inputs) the window holds only the group's segments."""
    batch_size, seq_len = segment_ids.q.shape
    ids = jnp.pad(segment_ids.q, ((0, 0), (0, -seq_len % block_q)), mode="edge")
    ids = ids.reshape(batch_size, -1, block_q).astype(jnp.int32)
    groups = [ids[..., r : r + row_group] for r in range(0, block_q, row_group)]
    q_lo = jnp.stack([g.min(axis=-1) for g in groups], axis=-1)[..., None]
    q_hi = jnp.stack([g.max(axis=-1) for g in groups], axis=-1)[..., None]
    kv = _block_ranges(segment_ids.kv, NUM_LANES)[:, None, None]
    meets = (q_lo <= kv[..., 1]) & (kv[..., 0] <= q_hi)  # [b, q, groups, kv]
    col = jnp.arange(kv.shape[3], dtype=jnp.int32) * NUM_LANES
    lo = jnp.min(jnp.where(meets, col, segment_ids.kv.shape[1]), axis=-1)
    hi = jnp.max(jnp.where(meets, col + NUM_LANES, 0), axis=-1)
    return jnp.stack([lo, hi], axis=-1).reshape(batch_size, -1)


# For autograd testing.
def mha_reference_no_custom_vjp(
    q,
    k,
    v,
    ab: jax.Array | None = None,
    segment_ids: SegmentIds | None = None,
    *,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale: float = 1.0,
    save_residuals: bool = False,
):
    logits = jnp.einsum("bhqc,bhkc->bhqk", q, k)
    if ab is not None:
        logits += ab
    if sm_scale != 1.0:
        logits *= sm_scale

    mask = None
    if segment_ids is not None:
        mask = segment_ids.q[:, :, None] == segment_ids.kv[:, None, :]
        mask = mask[:, None, :, :]

    if causal:
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        mask_shape = (q_seq_len, kv_seq_len)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = (col_ids <= row_ids)[None, None, :, :]
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    logits = logits if mask is None else logits + jnp.where(mask, 0.0, mask_value)

    m = logits.max(axis=-1)
    unnormalized = jnp.exp(logits - m[..., None])
    l = unnormalized.sum(axis=-1)
    weights = unnormalized / l[..., None]
    out = jnp.einsum("bhqk,bhkc->bhqc", weights, v)
    if save_residuals:
        return out, l, m
    return out


@functools.partial(jax.jit, static_argnames=["causal", "mask_value", "sm_scale"])
@jax.default_matmul_precision("bfloat16")
def mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None = None,
    causal: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    sm_scale=1.0,
):
    return _mha_reference(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=False,
    )


def _mha_reference(
    q,
    k,
    v,
    ab,
    segment_ids: SegmentIds | None,
    causal: bool,
    mask_value: float,
    sm_scale: float,
    save_residuals: bool,
):
    return mha_reference_no_custom_vjp(
        q,
        k,
        v,
        ab,
        segment_ids,
        causal=causal,
        mask_value=mask_value,
        sm_scale=sm_scale,
        save_residuals=save_residuals,
    )


def _verify_block(block_name, dim_name, block, dim, should_divide=True):
    if block > dim:
        raise ValueError(
            f"{block_name}={block} should be smaller or equal to {dim_name}={dim}"
        )
    if should_divide and dim % block != 0:
        raise ValueError(
            f"{dim_name}={dim} should be divisible by {block_name}={block}"
        )
