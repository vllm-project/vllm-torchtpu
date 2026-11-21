# Copyright 2025 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================

"""Implementation of Sparse Flash Attention, a.k.a. "Splash" attention.

Source provenance:
  Ported from https://github.com/openxla/tokamax
  (`tokamax/_src/ops/splash_attention/splash_attention_kernel.py`),
  commit `5dff06a967d941abc5035edff68766e22a315430` (2026-10-20).
  Local modifications vs upstream Tokamax:
  - Pruned training-only backward (`dq`/`dkv`, `custom_vjp`, `k_in_lanes=False`,
    `phase`), dropout, experimental-scheduler, remat checkpointing, and unused
    `make_splash_mqa` / `SplashConfig.get_default` paths.
  - Added segment tile pruning and demotion in `_segment_block_overlap` /
    `_refine_mask_info_with_segments` (`_unvmap_all_p`, `seg_all`, `q_sequence`
    causal bounds, `BlockMaskCode.SEGMENT_ONLY` code, `downcast_smem_data`
    wiring) and gated in-kernel segment masking on `segment_only_full`.
  - Added `CausalMask(offset=0)` and square-block/layout preconditions for
    `qk_diag_skip` / `sv_diag_skip` (with unified `diag_grid`) and rejected
    diagonal skip for dynamic masks.
  - Added `q_prescaled_log2e`, natural-log residual `/ LOG2E` conversion under
    `use_base2_exp`, `s_curr_flat.astype(q.dtype)`, inlined lane/mask constants,
    `to_i32`, `SplashAttentionKernel.__call__` overloads, `jax.tree.map` in
    `manual_sharding_spec`, and `MaskInfo._fields`-derived dynamic
    `mask_info_specs`.
  - Omitted unused `CompilerParams(flags=...)` dict on `_splash_attention_forward`.
"""

import dataclasses
import enum
import functools
import json
import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, overload

import jax
import jax.extend.core
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_mask as mask_lib,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa import (
    splash_attention_mask_info as mask_info_lib,
)

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)
SplashCustomReturnType = jax.Array | tuple[jax.Array, dict[str, jax.Array]]
DotDimensionNumbers = tuple[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[int, ...], tuple[int, ...]],
]


BlockMaskCode = mask_info_lib.BlockMaskCode
MaskInfo = mask_info_lib.MaskInfo
NUM_LANES = 128
NUM_SUBLANES = 8
_MIN_DIAG_GRID = 2
# We predefine some useful dimension numbers for dot_general
NN_DIM_NUMBERS = (((1,), (0,)), ((), ()))  # standard matmul
NT_DIM_NUMBERS = (((1,), (1,)), ((), ()))  # RHS transposed

LOG2E = math.log2(math.e)


def _is_fp8(x) -> bool:
    return x.dtype == jnp.float8_e4m3fn


def _unquantized_dtype(x):
    """bf16 for fp8 inputs; otherwise the input dtype (legacy behavior)."""
    return jnp.bfloat16 if _is_fp8(x) else x.dtype


def _make_unvmap_primitive(
    name: str, reduce_fn: Callable[..., jax.Array]
) -> jax.extend.core.Primitive:
    """Primitive that reduces `vmap` batch dims so `MaskInfo` stays unbatched."""
    prim = jax.extend.core.Primitive(name)
    prim.def_impl(lambda x: x)
    prim.def_abstract_eval(lambda x: jax.core.ShapedArray(x.shape, x.dtype))
    jax.interpreters.batching.primitive_batchers[prim] = lambda xs, bdims: (
        prim.bind(xs[0] if bdims[0] is None else reduce_fn(xs[0], axis=bdims[0])),
        None,
    )
    jax.interpreters.mlir.register_lowering(
        prim, jax.interpreters.mlir.lower_fun(lambda x: x, multiple_results=False)
    )
    return prim


_unvmap_any_p = _make_unvmap_primitive("splash_unvmap_any", jnp.any)
_unvmap_all_p = _make_unvmap_primitive("splash_unvmap_all", jnp.all)


class SegmentIds(NamedTuple):
    """SegmentIds for Q and KV sequences.

    SegmentIds are a mechanism to ensure that there is no cross-attention between
    segments (fraction of a sequence) that have been concatenated together into a
    sequence. Each array is a list of ids (integers). Only tokens with the same
    id are allowed to attend to each other.

    The static mask (e.g. causal) is "and-ed" with the segment id mask to form
    the actual attention mask. It is important that the latter does not have any
    all-zero rows (along dimension kv). Otherwise it would result in a invalid
    softmax (the denominator would be 0).
    This condition holds for causal self-attention because in this case segment
    ids form a block diagonal matrix so at least one element in each row is set.
    It is easy to break this condition with non-self-attention configurations.
    Attributes:
      q: segment ids along the Q sequence
      kv: segment ids along the KV sequence
    """

    q: jax.Array  # [q_seq_len]
    kv: jax.Array  # [kv_seq_len]


MaskFunctionType = Callable[..., jax.Array]


def get_kernel_name(
    *,
    is_mqa: bool,
    save_residuals: bool,
    is_segmented: bool,
) -> str:
    """Return a unique name for a forward SplashAttention kernel variant.

    Args:
      is_mqa: Whether the kernel computes multi-query attention.
      save_residuals: Whether logsumexp and max-logit residuals are returned.
      is_segmented: Whether segment IDs are supplied.

    Returns:
      Kernel name string.
    """
    residuals = "_residuals" if save_residuals else "_no_residuals"
    attention_type = "mqa" if is_mqa else "mha"
    segments = "_segmented" if is_segmented else ""
    return f"splash_{attention_type}_fwd{segments}{residuals}"


# Splash attention implementation


# We use an IntEnum to make it JSON serializable as regen metadata.
class QKVLayout(enum.IntEnum):
    HEAD_DIM_MINOR = enum.auto()  # [..., seq_len, head_dim]
    SEQ_MINOR = enum.auto()  # [..., head_dim, seq_len]


def from_head_minor(vals: tuple[Any, ...], layout: QKVLayout):
    if layout == QKVLayout.HEAD_DIM_MINOR:
        return vals
    return (*vals[:-2], vals[-1], vals[-2])


@dataclasses.dataclass(frozen=True, slots=True)
class SplashConfig:
    """Configuration parameterizing SplashAttention kernels.

    Tile sizes and memory layouts have negligible effect on numerics, while
    greatly affecting performance. Numerics-affecting fields
    (`attn_logits_soft_cap`, `max_logit_const`, `use_base2_exp`,
    `q_prescaled_log2e`, `fuse_reciprocal`) directly change the attention math.

    Note that changing the layouts only influences the physical layout that the
    kernel will enforce. The logical interface to splash attention always takes
    the head dimension as the minormost one.
    """

    block_q: int
    block_kv: int
    block_kv_compute: int | None = None
    num_stacked_q_heads: int = 1
    q_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
    k_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR
    v_layout: QKVLayout = QKVLayout.HEAD_DIM_MINOR

    fwd_cost_estimate: pl.CostEstimate | None = None

    attn_logits_soft_cap: float | None = None
    fuse_reciprocal: bool = True  # whether to compute o / lse inside the kernel
    use_base2_exp: bool = True
    max_logit_const: float | None = None
    interpret: bool = False
    # Skip the wasted causal-diagonal QK matmul. On a partial-mask (diagonal) block,
    # split the QK matmul into a diag_grid x diag_grid sub-grid over (kv rows,
    # q cols) and skip every sub-tile that lies entirely above the causal line
    # (kv > q), filling mask_value instead of computing it. Those entries are
    # overwritten to mask_value by _apply_mask_and_soft_cap regardless, so the result
    # is mathematically equivalent; elementwise/softmax ops still run on the full
    # assembled tile.
    # PRECONDITION (enforced below): pure CausalMask + aligned SQUARE blocks with a
    # single compute tile per block (block_q == block_kv == block_kv_compute),
    # num_stacked_q_heads == 1, diag_grid dividing block_q and block_kv_compute,
    # and sequence length a multiple of the block. Any other config raises.
    qk_diag_skip: bool = False
    # Granularity of the diagonal skip: default grid=2 uses quadrants (skips 1/4
    # of a diagonal block); larger power-of-2 grids skip more of the upper
    # triangle at the cost of smaller sub-tile matmuls. Must be a power of 2 >= 2
    # dividing both block_q and block_kv_compute.
    diag_grid: int = 2
    # Skip the wasted causal-diagonal SV matmul. On a partial-mask (diagonal)
    # block, split the SV matmul into a diag_grid x diag_grid sub-grid and
    # skip sub-tiles strictly above the diagonal (kj > qi) during dot_general.
    sv_diag_skip: bool = False
    # [experimental] q arrives already multiplied by LOG2E (the caller folds it
    # into the softmax scale before the fp8 cast, so q is rounded to fp8 once).
    # The kernel skips its own q scaling.
    q_prescaled_log2e: bool = False

    def _validate_diag_skip(self, block_kv_compute: int) -> None:
        """Validate tile, grid, stacking, and layout preconditions for diag skip.

        Args:
          block_kv_compute: Resolved non-None KV compute block size.
        """
        if not (self.block_q == self.block_kv == block_kv_compute):
            msg = (
                "diag_skip requires square forward blocks "
                "(block_q == block_kv == block_kv_compute); got "
                f"{self.block_q}/{self.block_kv}/{block_kv_compute}."
            )
            raise ValueError(msg)
        if self.diag_grid < _MIN_DIAG_GRID or (self.diag_grid & (self.diag_grid - 1)):
            msg = f"diag_grid must be a power of 2 >= 2; got {self.diag_grid}."
            raise ValueError(msg)
        if self.block_q % self.diag_grid != 0 or block_kv_compute % self.diag_grid != 0:
            msg = (
                f"diag_grid ({self.diag_grid}) must divide block_q ({self.block_q}) "
                f"and block_kv_compute ({block_kv_compute})."
            )
            raise ValueError(msg)
        if self.num_stacked_q_heads != 1:
            msg = (
                "diag_skip requires num_stacked_q_heads == 1; "
                f"got {self.num_stacked_q_heads}."
            )
            raise ValueError(msg)
        if self.sv_diag_skip and self.v_layout != QKVLayout.HEAD_DIM_MINOR:
            msg = (
                "sv_diag_skip requires v_layout == QKVLayout.HEAD_DIM_MINOR; "
                f"got {self.v_layout}."
            )
            raise ValueError(msg)

    def __post_init__(self) -> None:
        """Resolve `block_kv_compute` default and validate configuration invariants."""
        block_kv_compute: int = (
            self.block_kv if self.block_kv_compute is None else self.block_kv_compute
        )
        if self.block_kv_compute is None:
            object.__setattr__(self, "block_kv_compute", block_kv_compute)
        if self.q_prescaled_log2e and not self.use_base2_exp:
            msg = "q_prescaled_log2e requires use_base2_exp=True."
            raise ValueError(msg)
        if self.qk_diag_skip or self.sv_diag_skip:
            self._validate_diag_skip(block_kv_compute)


def _qk_diag_skip_matmul(
    q_flat: jax.Array,
    k: jax.Array,
    *,
    mask_value: float,
    config: SplashConfig,
    qk_dims: DotDimensionNumbers,
) -> jax.Array:
    """Compute diagonal-block QK matmul while skipping strictly upper sub-tiles.

    Args:
      q_flat: Flattened query tile of shape `[block_q, head_dim]`.
      k: Key tile in `config.k_layout`.
      mask_value: Fill value for skipped strictly upper sub-tiles.
      config: SplashConfig with `diag_grid` and `k_layout`.
      qk_dims: `lax.dot_general` dimension numbers for QK.

    Returns:
      Assembled float32 QK logit tile of shape `[block_q, block_kv_compute]`.
    """
    diag_grid = config.diag_grid
    sub_q = q_flat.shape[0] // diag_grid
    k_seq_len = (
        k.shape[0] if config.k_layout == QKVLayout.HEAD_DIM_MINOR else k.shape[1]
    )
    sub_kv = k_seq_len // diag_grid
    q_parts = [q_flat[i * sub_q : (i + 1) * sub_q, :] for i in range(diag_grid)]
    if config.k_layout == QKVLayout.HEAD_DIM_MINOR:
        k_parts = [k[j * sub_kv : (j + 1) * sub_kv, :] for j in range(diag_grid)]
    else:
        k_parts = [k[:, j * sub_kv : (j + 1) * sub_kv] for j in range(diag_grid)]
    rows = []
    for qi in range(diag_grid):
        cols = []
        for kj in range(diag_grid):
            if kj > qi:
                cols.append(jnp.full((sub_q, sub_kv), mask_value, dtype=jnp.float32))
            else:
                cols.append(
                    lax.dot_general(
                        q_parts[qi],
                        k_parts[kj],
                        qk_dims,
                        preferred_element_type=jnp.float32,
                    )
                )
        rows.append(jnp.concatenate(cols, axis=1))
    return jnp.concatenate(rows, axis=0)


def _sv_diag_skip_matmul(
    s_curr_flat: jax.Array,
    v: jax.Array,
    *,
    diag_grid: int,
    sv_dims: DotDimensionNumbers,
) -> jax.Array:
    """Compute diagonal-block SV matmul while skipping strictly upper sub-tiles.

    Args:
      s_curr_flat: Softmax weight tile of shape `[block_q, block_kv_compute]`.
      v: Value tile in `HEAD_DIM_MINOR` layout `[block_kv_compute, head_dim_v]`.
      diag_grid: Number of sub-grid divisions along Q and KV.
      sv_dims: `lax.dot_general` dimension numbers for SV.

    Returns:
      Assembled float32 output tile of shape `[block_q, head_dim_v]`.
    """
    sub_q = s_curr_flat.shape[0] // diag_grid
    sub_kv = s_curr_flat.shape[1] // diag_grid
    s_parts = [
        [
            s_curr_flat[i * sub_q : (i + 1) * sub_q, j * sub_kv : (j + 1) * sub_kv]
            for j in range(diag_grid)
        ]
        for i in range(diag_grid)
    ]
    v_parts = [v[j * sub_kv : (j + 1) * sub_kv, :] for j in range(diag_grid)]
    o_rows = []
    for qi in range(diag_grid):
        o_row = lax.dot_general(
            s_parts[qi][0],
            v_parts[0],
            sv_dims,
            preferred_element_type=jnp.float32,
        )
        for kj in range(1, qi + 1):
            o_row = o_row + lax.dot_general(
                s_parts[qi][kj],
                v_parts[kj],
                sv_dims,
                preferred_element_type=jnp.float32,
            )
        o_rows.append(o_row)
    return jnp.concatenate(o_rows, axis=0)


def to_i32(x):
    return x.astype(jnp.int32)


def _apply_mask_and_soft_cap(
    qk: jax.Array,
    mask_value: float,
    mask_ref,
    q_sequence_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    *,
    attn_logits_soft_cap: float | None,
    k_slice: pl.Slice,
    k_offset: int | jax.Array,
    bq: int,
    mask_function=None,
    has_partial_mask: bool = False,
    has_segment_mask: bool = False,
) -> jax.Array:
    assert mask_ref is None or q_sequence_ref is None
    assert (q_sequence_ref is None) == (mask_function is None)

    masks = []
    if has_partial_mask:
        if mask_ref is not None:
            mask = mask_ref[:, k_slice]
            masks.append(mask.astype(jnp.bool_))
        elif mask_function is not None:
            # Compute the mask using the given q_sequence indices.
            # KV indices are computed on the fly. This works because we only support Q
            # sequence sharding. If we wanted to compute Q indices too, then we would
            # need to keep into account the current shard along Q sequence.
            assert q_sequence_ref is not None

            if q_sequence_ref.shape != (bq, NUM_LANES):
                msg = (
                    f"Expected q_sequence_ref shape {(bq, NUM_LANES)}, "
                    f"got {q_sequence_ref.shape}."
                )
                raise ValueError(msg)

            k_sequence = k_offset + jax.lax.broadcasted_iota(
                jnp.int32, (bq, k_slice.size), 1
            )

            repeats, rem = divmod(k_slice.size, NUM_LANES)
            if rem != 0:
                msg = f"k_slice.size ({k_slice.size}) must be a multiple of {NUM_LANES}"
                raise ValueError(msg)
            q_sequence = jnp.tile(
                q_sequence_ref[...],
                (1, repeats),
            )  # [bq, k_slice.size]

            assert q_sequence.shape == k_sequence.shape
            computed_mask = mask_function(q_sequence, k_sequence)
            if computed_mask.dtype != jnp.dtype(jnp.bool_):
                raise ValueError(
                    "Mask function must return a boolean-valued array, but got:"
                    f" {computed_mask.dtype}"
                )
            masks.append(computed_mask)

    if has_segment_mask and q_segment_ids_ref is not None:
        kv_ids = kv_segment_ids_ref[:1, k_slice]  # [1, k_slice]
        repeats, rem = divmod(kv_ids.shape[1], NUM_LANES)
        if rem:
            msg = f"block_kv must be a multiple of {NUM_LANES}"
            raise NotImplementedError(msg)
        q_ids = jnp.tile(q_segment_ids_ref[:], (1, repeats))  # [bq, bkv]
        masks.append(q_ids == kv_ids)

    def cap_logits(logits):
        if attn_logits_soft_cap is not None:
            logits = jnp.tanh(qk / attn_logits_soft_cap)
            return logits * attn_logits_soft_cap
        else:
            return logits

    if masks:
        masks = [m.astype(jnp.bool_) for m in masks]
        mask = functools.reduce(jnp.logical_and, masks)
        qk = cap_logits(qk)
        if mask.ndim == 2 and qk.ndim == 3:
            mask = jnp.expand_dims(mask, axis=0)

        qk = jnp.where(mask, qk, mask_value)
    else:
        qk = cap_logits(qk)
    return qk


def flash_attention_kernel(
    # Prefetched inputs
    active_rows_ref,
    active_cols_ref,
    mask_next_ref,
    bounds_start_ref,
    bounds_end_ref,
    block_mask_ref,
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    q_segment_ids_ref,
    kv_segment_ids_ref,
    sinks_ref,
    mask_ref,
    q_sequence_ref,
    max_logit_value_ref,
    # Outputs
    o_ref,
    logsumexp_ref,
    l_linear_ref,
    max_logits_ref,
    # Scratch
    m_scratch_ref,
    l_scratch_ref,
    o_scratch_ref,
    *,
    mask_value: float,
    kv_steps: int,
    bq: int,
    bkv: int,
    bkv_compute: int,
    head_dim_v: int,
    num_stacked_q_heads: int,
    mask_function: MaskFunctionType | None,
    fuse_reciprocal: bool,  # config.fuse_reciprocal or not save_residuals
    config: SplashConfig,
    segment_only_full: bool,
):
    del active_rows_ref, mask_next_ref
    float32 = jnp.float32
    HEAD_DIM_MINOR = QKVLayout.HEAD_DIM_MINOR
    attn_logits_soft_cap = config.attn_logits_soft_cap
    if attn_logits_soft_cap is not None and config.use_base2_exp:
        attn_logits_soft_cap *= LOG2E

    # If the head_dim_v is not a multiple of the number of lanes, it will be
    # padded to that multiple with zeros.
    head_dim_v_repeats = pl.cdiv(head_dim_v, NUM_LANES)

    grid_idx = pl.program_id(1)
    h = pl.program_id(0)

    if block_mask_ref is not None:
        block_code = block_mask_ref[grid_idx].astype(jnp.int32)
        should_initialize = bounds_start_ref[grid_idx].astype(jnp.bool_)
        should_write = bounds_end_ref[grid_idx].astype(jnp.bool_)
        j = active_cols_ref[grid_idx].astype(jnp.int32)
    else:
        block_code = None
        j = grid_idx % kv_steps
        should_initialize = j == 0
        should_write = j == kv_steps - 1

    max_logit_estimate = config.max_logit_const  # potentially None
    if max_logit_value_ref is not None:  # already ensures max_logit_const is None
        assert num_stacked_q_heads == 1
        max_logit_estimate = max_logit_value_ref[0, h]

    if config.use_base2_exp and max_logit_estimate is not None:
        max_logit_estimate *= LOG2E

    @pl.when(should_initialize)
    def init():
        o_scratch_ref[...] = jnp.zeros_like(o_scratch_ref)

        sink = None
        if sinks_ref is not None:
            sink = sinks_ref[0, h].astype(m_scratch_ref.dtype)
            if config.use_base2_exp:
                sink *= LOG2E

        if sink is None and max_logit_estimate is None:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, mask_value)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        elif sink is None and max_logit_estimate is not None:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
            l_scratch_ref[...] = jnp.zeros_like(l_scratch_ref)
        elif sink is not None and max_logit_estimate is None:
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, sink)
            l_scratch_ref[...] = jnp.ones_like(l_scratch_ref)
        elif sink is not None and max_logit_estimate is not None:
            exp = jnp.exp2 if config.use_base2_exp else jnp.exp
            m_scratch_ref[...] = jnp.full_like(m_scratch_ref, max_logit_estimate)
            l_scratch_ref[...] = exp(
                sink - jnp.full_like(l_scratch_ref, max_logit_estimate)
            )

    def body(
        kv_compute_index,
        _,
        has_partial_mask: bool = False,
        has_segment_mask: bool = False,
    ):
        slice_k = pl.ds(kv_compute_index * bkv_compute, bkv_compute)
        m_prev, l_prev = m_scratch_ref[...], l_scratch_ref[...]
        assert m_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)
        assert l_prev.shape == (num_stacked_q_heads, bq, NUM_LANES)

        q = q_ref[...] if config.q_layout == HEAD_DIM_MINOR else q_ref[...].mT
        if config.use_base2_exp and not config.q_prescaled_log2e:
            if q.dtype == jnp.float8_e4m3fn:
                q = (q.astype(jnp.bfloat16) * LOG2E).astype(jnp.float8_e4m3fn)
            else:
                q *= LOG2E

        head_dim_qk = q.shape[-1]
        # Collapse the head and sequence dimensions for a larger matmul.
        q_flat = q.reshape((num_stacked_q_heads * bq, head_dim_qk))

        if config.k_layout == HEAD_DIM_MINOR:
            k = k_ref[slice_k, :]
            qk_dims = NT_DIM_NUMBERS
        else:
            k = k_ref[:, slice_k]
            qk_dims = NN_DIM_NUMBERS

        if config.qk_diag_skip and has_partial_mask:
            qk_flat = _qk_diag_skip_matmul(
                q_flat,
                k,
                mask_value=mask_value,
                config=config,
                qk_dims=qk_dims,
            )
        else:
            qk_flat = lax.dot_general(
                q_flat, k, qk_dims, preferred_element_type=float32
            )
        qk = qk_flat.reshape((num_stacked_q_heads, bq, bkv_compute))

        qk = _apply_mask_and_soft_cap(
            qk,
            mask_value,
            mask_ref,
            q_sequence_ref,
            q_segment_ids_ref,
            kv_segment_ids_ref,
            attn_logits_soft_cap=attn_logits_soft_cap,
            k_slice=slice_k,
            k_offset=j * bkv + kv_compute_index * bkv_compute,
            bq=bq,
            mask_function=mask_function,
            has_partial_mask=has_partial_mask,
            has_segment_mask=has_segment_mask,
        )

        if max_logit_estimate is None:
            m_curr = qk.max(axis=-1)[..., None]
            assert m_curr.shape == (num_stacked_q_heads, bq, 1)
            m_next = jnp.maximum(m_prev, m_curr)
            assert m_next.shape == (num_stacked_q_heads, bq, NUM_LANES)
        else:
            m_next = None

        bkv_repeats, rem = divmod(bkv_compute, NUM_LANES)
        if rem != 0:
            raise NotImplementedError(
                f"{bkv_compute=} should be a multiple of {NUM_LANES}"
            )

        exp = jnp.exp2 if config.use_base2_exp else jnp.exp
        if m_next is not None:
            s_curr = exp(qk - jnp.tile(m_next, (1, 1, bkv_repeats)))
        else:
            assert max_logit_estimate is not None
            s_curr = exp(qk - max_logit_estimate)
        assert s_curr.shape == (num_stacked_q_heads, bq, bkv_compute)

        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0, 1))
        assert l_curr.shape == (num_stacked_q_heads, bq, NUM_LANES)

        if m_next is not None:
            alpha = exp(m_prev - m_next)
            l_next = l_curr + alpha * l_prev
            m_scratch_ref[...], l_scratch_ref[...] = m_next, l_next
        else:
            alpha = None
            l_scratch_ref[...] = l_curr + l_prev

        s_curr_flat = s_curr.reshape((num_stacked_q_heads * bq, bkv_compute)).astype(
            q.dtype
        )

        if config.v_layout == HEAD_DIM_MINOR:
            v = v_ref[slice_k, :]
            sv_dims = NN_DIM_NUMBERS
        else:
            v = v_ref[:, slice_k]
            sv_dims = NT_DIM_NUMBERS

        if config.sv_diag_skip and has_partial_mask:
            o_curr_flat = _sv_diag_skip_matmul(
                s_curr_flat,
                v,
                diag_grid=config.diag_grid,
                sv_dims=sv_dims,
            )
        else:
            o_curr_flat = lax.dot_general(
                s_curr_flat, v, sv_dims, preferred_element_type=jnp.float32
            )
        o_curr = o_curr_flat.reshape((num_stacked_q_heads, bq, head_dim_v))

        if alpha is not None:
            alpha_o = jnp.tile(alpha, (1, 1, head_dim_v_repeats))
            alpha_o = alpha_o[..., : o_scratch_ref.shape[-1]]
            o_scratch_ref[...] = alpha_o * o_scratch_ref[...] + o_curr
        else:
            o_scratch_ref[...] = o_scratch_ref[...] + o_curr

    assert bkv % bkv_compute == 0
    num_iters = (
        k_ref.shape[0 if config.k_layout == HEAD_DIM_MINOR else 1] // bkv_compute
    )

    if block_code is not None:

        @pl.when(block_code == BlockMaskCode.FULL)
        def _():
            lax.fori_loop(
                0,
                num_iters,
                functools.partial(body, has_partial_mask=False, has_segment_mask=False),
                None,
                unroll=True,
            )

        @pl.when(block_code == BlockMaskCode.PARTIAL)
        def _():
            lax.fori_loop(
                0,
                num_iters,
                functools.partial(body, has_partial_mask=True, has_segment_mask=True),
                None,
                unroll=True,
            )

        if segment_only_full:

            @pl.when(block_code == BlockMaskCode.SEGMENT_ONLY)
            def _():
                lax.fori_loop(
                    0,
                    num_iters,
                    functools.partial(
                        body, has_partial_mask=False, has_segment_mask=True
                    ),
                    None,
                    unroll=True,
                )

    else:
        lax.fori_loop(
            0,
            num_iters,
            functools.partial(body, has_partial_mask=True, has_segment_mask=True),
            None,
            unroll=True,
        )

    @pl.when(should_write)
    def end():
        l_acc = l_scratch_ref[...]
        m = m_scratch_ref[...]
        if fuse_reciprocal:  # allows fusing reciprocal out of the kernel
            l_inv = jnp.tile(
                jnp.where(l_acc == 0.0, 0.0, 1.0 / l_acc), (1, 1, head_dim_v_repeats)
            )
            l_inv = l_inv[..., : o_scratch_ref.shape[-1]]
            o_ref[...] = (o_scratch_ref[...] * l_inv).astype(o_ref.dtype)
        else:
            o_ref[...] = o_scratch_ref[...].astype(o_ref.dtype)
        if logsumexp_ref is not None:
            assert logsumexp_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
            log = jnp.log2 if config.use_base2_exp else jnp.log
            logsumexp = m + log(l_acc)
            logsumexp_ref[...] = logsumexp.astype(logsumexp_ref.dtype)
        if l_linear_ref is not None:
            assert l_linear_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
            l_linear_ref[...] = l_acc.astype(l_linear_ref.dtype)
        if max_logits_ref is not None:
            assert max_logits_ref.shape == (num_stacked_q_heads, bq, NUM_LANES)
            max_logits_ref[...] = m.astype(max_logits_ref.dtype)


def _div(dividend: int, divisor: int):
    if divisor == 1:
        return dividend

    return lax.div(dividend, divisor)


def _bytes(x: jax.Array | jax.ShapeDtypeStruct | np.ndarray | None) -> int:
    if x is None:
        return 0

    if jnp.issubdtype(x.dtype, jnp.floating):
        info = jnp.finfo
    elif jnp.issubdtype(x.dtype, jnp.integer):
        info = jnp.iinfo
    else:
        raise ValueError(f"Unsupported dtype: {x.dtype}")
    return math.ceil(math.prod(x.shape) * info(x.dtype).bits / 8)


def _segment_block_overlap(
    segment_ids: SegmentIds,
    q_idx: jax.Array,
    kv_idx: jax.Array,
    *,
    q_blocks: int,
    kv_blocks: int,
    bq: int,
    bkv: int,
) -> tuple[jax.Array, jax.Array]:
    """Return per-tile segment overlap and uniform-segment flags.

    Conservative: `seg_any` tests whether the `[min, max]` id intervals of the
    two blocks intersect. If the blocks share an id, that id lies in both
    intervals, so pruning on `seg_any` never drops a contributing tile. False
    positives (e.g. q ids `{1, 5}` against kv id `{3}`) merely leave a fully
    masked tile scheduled. This argument does not assume ids are sorted.

    Non-negative segment ids (`>= 0`) denote active segments; negative values
    are padding sentinels (e.g. `-1` for padded queries or `-2` for inactive KV
    slots) and never set `seg_all`, keeping padded tiles demoted so in-kernel
    segment masking runs instead of treating them as `BlockMaskCode.FULL`.

    Both outputs are reduced over any `vmap` batch dimension (`any` for
    `seg_any`, `all` for `seg_all`) so the whole batch shares one schedule,
    keeping `pallas_call` batch fusion intact.

    Args:
      segment_ids: Unbatched q / kv segment ids.
      q_idx: Integer array of q block indices.
      kv_idx: Integer array of kv block indices, broadcastable against `q_idx`.
      q_blocks: Number of q blocks, `q_seq_len // bq`.
      kv_blocks: Number of kv blocks, `kv_seq_len // bkv`.
      bq: q block size.
      bkv: kv block size.

    Returns:
      Tuple `(seg_any, seg_all)` of boolean arrays with the broadcast shape of
      `q_idx` and `kv_idx`: `seg_any` is True when the two blocks' `[min, max]`
      intervals intersect, and `seg_all` is True when both blocks share a single
      uniform non-negative segment id (`>= 0`).
    """
    q_segs = segment_ids.q.reshape(q_blocks, bq)
    kv_segs = segment_ids.kv.reshape(kv_blocks, bkv)
    q_lo = _onehot_lookup(jnp.min(q_segs, axis=-1), q_idx)
    q_hi = _onehot_lookup(jnp.max(q_segs, axis=-1), q_idx)
    kv_lo = _onehot_lookup(jnp.min(kv_segs, axis=-1), kv_idx)
    kv_hi = _onehot_lookup(jnp.max(kv_segs, axis=-1), kv_idx)
    seg_any = _unvmap_any_p.bind((q_hi >= kv_lo) & (kv_hi >= q_lo))
    seg_all = _unvmap_all_p.bind(
        (q_lo == q_hi) & (kv_lo == kv_hi) & (q_lo == kv_lo) & (q_lo >= 0)
    )
    return seg_any, seg_all


def _onehot_lookup(table: jax.Array, idx: jax.Array) -> jax.Array:
    """Returns `table[idx]` for in-range `idx` as a one-hot select and sum.

    Uses `O(len(idx) * len(table))` compares over the last axis, matching the
    `row_eq` shape below, instead of an indexed gather.

    Args:
      table: `[n]` array (may carry a leading `vmap` batch dim).
      idx: Integer array with values in `[0, n)`.

    Returns:
      An array of `idx.shape` with `table`'s dtype.
    """
    onehot = idx[..., None] == jnp.arange(table.shape[-1], dtype=idx.dtype)
    return jnp.sum(jnp.where(onehot, table, jnp.zeros_like(table)), axis=-1)


class _SegmentRefinementGrid(NamedTuple):
    """Block grid geometry for segment-aware mask refinement."""

    q_blocks: int
    kv_blocks: int
    block_q: int
    block_kv: int


def _extract_initial_schedule(
    mask_info: MaskInfo,
    num_rows: int,
    num_cols: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build dense or existing compacted schedule arrays before segment filtering.

    Args:
      mask_info: Base `MaskInfo` before segment refinement.
      num_rows: Total number of Q blocks.
      num_cols: Total number of KV blocks.

    Returns:
      Tuple `(rows, cols, block_mask, num_active)` of 1D schedule arrays.
    """
    if mask_info.active_rows is None:
        num_tiles = num_rows * num_cols
        rows, cols = jnp.unravel_index(
            jnp.arange(num_tiles, dtype=jnp.int32), (num_rows, num_cols)
        )
        num_active = jnp.array([num_tiles], dtype=jnp.int32)
    else:
        rows = jnp.asarray(mask_info.active_rows, dtype=jnp.int32)
        cols = jnp.asarray(mask_info.active_cols, dtype=jnp.int32)
        num_tiles = rows.size
        num_active = jnp.asarray(mask_info.num_active_blocks, dtype=jnp.int32)

    block_mask = (
        jnp.full((num_tiles,), BlockMaskCode.FULL, dtype=jnp.int8)
        if mask_info.block_mask is None
        else jnp.asarray(mask_info.block_mask).reshape(num_tiles)
    )
    return rows, cols, block_mask, num_active


def _refine_mask_info_with_segments(
    mask_info: MaskInfo,
    segment_ids: SegmentIds,
    grid: _SegmentRefinementGrid,
    *,
    segment_only_full: bool,
    downcast_smem_data: bool = True,
) -> tuple[
    MaskInfo,
    jax.Array | np.ndarray | None,
    jax.Array | np.ndarray | None,
]:
    """Prune disjoint tiles and demote mixed/segment-masked tiles.

    Args:
      mask_info: Base static or dynamic `MaskInfo` before segment refinement.
      segment_ids: Unbatched query and key/value segment IDs.
      grid: Block grid geometry `(q_blocks, kv_blocks, block_q, block_kv)`.
      segment_only_full: Whether demoted base-full tiles use
        `BlockMaskCode.SEGMENT_ONLY` instead of `BlockMaskCode.PARTIAL`.
      downcast_smem_data: Whether to downcast `active_rows`/`active_cols` below
        `int32` when index bounds allow.

    Returns:
      Tuple `(refined_mask_info, bounds_start, bounds_end)` with compacted tile
      schedules and per-row start/end markers.
    """
    num_rows, num_cols = grid.q_blocks, grid.kv_blocks
    rows, cols, block_mask, num_active = _extract_initial_schedule(
        mask_info, num_rows, num_cols
    )
    num_tiles = rows.size
    idx = jnp.arange(num_tiles, dtype=jnp.int32)
    valid_static = (block_mask > BlockMaskCode.EMPTY) & (idx < num_active[0])

    rows_clipped = jnp.clip(rows, 0, num_rows - 1)
    cols_clipped = jnp.clip(cols, 0, num_cols - 1)
    seg_any, seg_all = _segment_block_overlap(
        segment_ids,
        rows_clipped,
        cols_clipped,
        q_blocks=grid.q_blocks,
        kv_blocks=grid.kv_blocks,
        bq=grid.block_q,
        bkv=grid.block_kv,
    )
    if mask_info.q_sequence is not None and mask_info.block_mask is None:
        q_seqs = jnp.asarray(mask_info.q_sequence, dtype=jnp.int32).reshape(
            grid.q_blocks, grid.block_q
        )
        q_seq_lo = _onehot_lookup(jnp.min(q_seqs, axis=-1), rows_clipped)
        q_seq_hi = _onehot_lookup(jnp.max(q_seqs, axis=-1), rows_clipped)
        kv_seq_lo = cols_clipped * grid.block_kv
        kv_seq_hi = kv_seq_lo + (grid.block_kv - 1)
        seg_any = seg_any & _unvmap_any_p.bind(q_seq_hi >= kv_seq_lo)
        seg_all = seg_all & _unvmap_all_p.bind(q_seq_lo >= kv_seq_hi)

    active = valid_static & seg_any
    # Every output row needs >= 1 scheduled step (with BlockMaskCode.EMPTY if
    # inactive) so `bounds_start`/`bounds_end` still zero-initialize and write
    # that row. Per-tile row lookups use `row_eq` one-hot instead of
    # `table[rows_clipped]`, and compaction below uses `lax.sort` instead of
    # `x[argsort(...)]`: this function uses no gather (see `_onehot_lookup`).
    row_eq = rows_clipped[:, None] == jnp.arange(num_rows, dtype=jnp.int32)[None, :]
    row_has_active = jnp.any(row_eq & active[:, None], axis=0)
    first_in_row = jnp.argmax(row_eq, axis=0).astype(jnp.int32)
    keep = active | (
        jnp.any(active)
        & ~jnp.any(row_eq & row_has_active[None, :], axis=1)
        & (idx == jnp.sum(jnp.where(row_eq, first_in_row[None, :], 0), axis=1))
    )

    demoted_code = (
        BlockMaskCode.SEGMENT_ONLY if segment_only_full else BlockMaskCode.PARTIAL
    )
    demoted = jnp.where(
        block_mask == BlockMaskCode.FULL,
        demoted_code,
        block_mask,
    )
    block_mask = jnp.where(
        active,
        jnp.where(seg_all, block_mask, demoted),
        BlockMaskCode.EMPTY,
    ).astype(block_mask.dtype)
    mask_next = (
        None
        if mask_info.mask_next is None
        else jnp.asarray(mask_info.mask_next).reshape(num_tiles)
    )

    # A stable sort on `~keep` moves kept tiles to the front in schedule order.
    new_num_active = jnp.sum(keep.astype(jnp.int32), keepdims=True)
    payload = (rows_clipped, cols_clipped, block_mask) + (
        () if mask_next is None else (mask_next,)
    )
    _, *payload = lax.sort(
        ((~keep).astype(jnp.int32), *payload), num_keys=1, is_stable=True
    )
    is_active_slot = idx < new_num_active[0]
    rows_sorted, cols_sorted, block_mask, *mask_next_sorted = [
        jnp.where(is_active_slot, x, 0).astype(x.dtype) for x in payload
    ]
    bounds_start, bounds_end = mask_info_lib.find_bounds(
        jnp.where(is_active_slot, payload[0], -1)
    )
    row_dtype = mask_info_lib.downcast_index_dtype(
        num_rows - 1, downcast_smem_data=downcast_smem_data
    )
    col_dtype = mask_info_lib.downcast_index_dtype(
        num_cols - 1, downcast_smem_data=downcast_smem_data
    )
    return (
        mask_info._replace(
            active_rows=rows_sorted.astype(row_dtype),
            active_cols=cols_sorted.astype(col_dtype),
            block_mask=block_mask.astype(jnp.int8),
            mask_next=mask_next_sorted[0] if mask_next_sorted else None,
            num_active_blocks=new_num_active,
        ),
        bounds_start,
        bounds_end,
    )


@overload
def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: Literal[False],
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    *,
    downcast_smem_data: bool = True,
) -> jax.Array: ...


@overload
def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: Literal[True],
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    *,
    downcast_smem_data: bool = True,
) -> tuple[jax.Array, dict[str, jax.Array]]: ...


@overload
def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    *,
    downcast_smem_data: bool = True,
) -> SplashCustomReturnType: ...


def _splash_attention_forward(
    mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    max_logit_value: jax.Array | None = None,
    *,
    downcast_smem_data: bool = True,
) -> SplashCustomReturnType:
    num_q_heads, q_seq_len, head_dim_qk = q.shape
    head_dim_v = v.shape[-1]
    bq, bkv = config.block_q, config.block_kv
    bkv_compute = config.block_kv_compute
    assert bkv_compute is not None
    fuse_reciprocal = config.fuse_reciprocal or not save_residuals
    num_stacked_q_heads = config.num_stacked_q_heads

    if num_stacked_q_heads > 1 and (sinks is not None or max_logit_value is not None):
        raise ValueError(
            "Stacked heads are not supported with sinks or max_logit_value."
        )

    if is_mqa:
        expected_kv_rank = 2
        num_kv_heads = 1
    else:
        expected_kv_rank = 3
        num_kv_heads = k.shape[0]

    if len(k.shape) != expected_kv_rank:
        raise ValueError(
            f"Expected {expected_kv_rank}-dim 'key' tensor for MQA. Instead got a"
            f" {len(k.shape)}-dim one."
        )

    if k.shape[-1] != head_dim_qk:
        raise ValueError(
            f"Expected 'key' head dimension to be: {head_dim_qk}. Instead got:"
            f" {k.shape[-1]}."
        )

    if not is_mqa and num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"In MHA, expected number of 'key' heads ({num_kv_heads}) to be a"
            f" multiple of the number of 'query' heads ({num_q_heads})"
        )

    if num_q_heads % num_stacked_q_heads != 0:
        raise ValueError(f"{num_q_heads=} must be a multiple of {num_stacked_q_heads=}")

    q_heads_per_kv_head = num_q_heads // num_kv_heads
    if q_heads_per_kv_head % num_stacked_q_heads != 0:
        raise ValueError(
            f"{q_heads_per_kv_head=} must be a multiple of {num_stacked_q_heads=}"
        )

    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError(
            f"Expected 'key' {k.shape} and 'value' {v.shape} to have the same "
            "leading dimensions."
        )

    if bkv % bkv_compute:
        raise ValueError(f"{bkv=} must be a multiple of {bkv_compute=}.")
    if bkv_compute % NUM_LANES:
        raise ValueError(f"{bkv_compute=} must be a multiple of {NUM_LANES}.")

    kv_seq_len = k.shape[-2]
    kv_steps = kv_seq_len // bkv

    segment_only_full = segment_ids is not None and (
        mask_info.partial_mask_blocks is not None
        or config.qk_diag_skip
        or config.sv_diag_skip
    )
    if segment_ids is not None:
        if segment_ids.q.shape != (q_seq_len,):
            raise ValueError(
                "Invalid shape for q segment_ids: "
                f"{segment_ids.q.shape}. Expected: {(q_seq_len,)}"
            )
        if segment_ids.kv.shape != (kv_seq_len,):
            raise ValueError(
                "Invalid shape for kv segment_ids: "
                f"{segment_ids.kv.shape}. Expected: {(kv_seq_len,)}"
            )
        mask_info, bounds_start, bounds_end = _refine_mask_info_with_segments(
            mask_info,
            segment_ids,
            _SegmentRefinementGrid(
                q_blocks=q_seq_len // bq,
                kv_blocks=kv_steps,
                block_q=bq,
                block_kv=bkv,
            ),
            segment_only_full=segment_only_full,
            downcast_smem_data=downcast_smem_data,
        )
    else:
        bounds_start, bounds_end = mask_info_lib.find_bounds(mask_info.active_rows)
    dynamic_grid = mask_info.active_rows is not None

    if config.max_logit_const is not None and max_logit_value is not None:
        raise ValueError(
            f"Only one of {config.max_logit_const=} and {max_logit_value=} can be set."
        )
    if max_logit_value is not None:
        if max_logit_value.shape not in ((), (1,), (num_q_heads,)):
            raise ValueError(
                "max_logit_value should be a 0,1-dim jax.Array of shape (), (1,) or"
                f" ({num_q_heads=},) but got {jax.typeof(max_logit_value)}"
            )
        max_logit_value = jnp.broadcast_to(
            jnp.atleast_1d(max_logit_value), (num_q_heads,)
        )

    q_layout = config.q_layout
    k_layout = config.k_layout
    v_layout = config.v_layout

    def unravel(f):
        def index_map(h_block, grid_idx, rows_ref, cols_ref, *_):
            if dynamic_grid:
                i = to_i32(rows_ref[grid_idx])
                j = to_i32(cols_ref[grid_idx])
            else:
                i = grid_idx // kv_steps
                j = grid_idx % kv_steps
            return f(h_block, i, j)

        return index_map

    def create_kv_index_map(layout):
        def index_map(h_block, i, j):
            del i  # Unused.
            first_h_in_block = h_block * num_stacked_q_heads
            prefix = () if is_mqa else (_div(first_h_in_block, q_heads_per_kv_head),)
            return from_head_minor((*prefix, j, 0), layout)

        return index_map

    q_index_map = unravel(
        lambda h_block, i, j: from_head_minor((h_block, i, 0), q_layout)
    )
    out_index_map = unravel(lambda h_block, i, j: (h_block, i, 0))
    k_index_map = unravel(create_kv_index_map(k_layout))
    v_index_map = unravel(create_kv_index_map(v_layout))

    def mask_index_map(h_block, grid_idx, rows_ref, cols_ref, mask_next_ref, *_):
        del h_block, rows_ref, cols_ref  # Unused.
        next_m = to_i32(mask_next_ref[grid_idx])
        return next_m, 0, 0

    q_segment_ids_index_map = unravel(lambda h_block, i, j: (i, 0))
    kv_segment_ids_index_map = unravel(lambda h_block, i, j: (0, j))

    # Convert the logical shape from head-minor to sequence-minor.
    in_specs: list[pl.BlockSpec | None] = [
        pl.BlockSpec(
            from_head_minor((num_stacked_q_heads, bq, head_dim_qk), q_layout),
            q_index_map,
        ),
        pl.BlockSpec(
            from_head_minor(
                (bkv, head_dim_qk) if is_mqa else (None, bkv, head_dim_qk),
                k_layout,
            ),
            k_index_map,
        ),
        pl.BlockSpec(
            from_head_minor(
                (bkv, head_dim_v) if is_mqa else (None, bkv, head_dim_v), v_layout
            ),
            v_index_map,
        ),
    ]
    if segment_ids is not None:
        in_specs += [
            pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map),
            pl.BlockSpec((NUM_SUBLANES, bkv), kv_segment_ids_index_map),
        ]
        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (q_seq_len, NUM_LANES),
            (0,),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (NUM_SUBLANES, kv_seq_len),
            (1,),
        )
    else:
        in_specs += [None, None]
        q_segment_ids = kv_segment_ids = None

    if sinks is not None:
        if sinks.shape != (num_q_heads,):
            msg = f"Invalid shape for sinks: {sinks.shape}. Expected: {(num_q_heads,)}"
            raise ValueError(msg)
        # align sinks to sublanes to allow vmap and shard_map over the kernel
        in_specs += [
            pl.BlockSpec(
                (NUM_SUBLANES, num_q_heads),
                lambda h, i, j, *_: (0, 0),
                memory_space=pltpu.SMEM,
            )
        ]
        sinks = jnp.broadcast_to(
            sinks.astype(jnp.float32)[None, :], (NUM_SUBLANES, num_q_heads)
        )
    else:
        in_specs += [None]

    if mask_info.partial_mask_blocks is not None:
        in_specs.append(pl.BlockSpec((None, bq, bkv), mask_index_map))
    else:
        in_specs.append(None)

    assert mask_info.partial_mask_blocks is None or mask_info.q_sequence is None

    if mask_info.q_sequence is not None:
        q_sequence = jax.lax.broadcast_in_dim(
            mask_info.q_sequence, (q_seq_len, NUM_LANES), (0,)
        )
        in_specs.append(pl.BlockSpec((bq, NUM_LANES), q_segment_ids_index_map))
    else:
        q_sequence = None
        in_specs.append(None)

    if max_logit_value is not None:
        # reshape to allow sublane selection for vmap-ping and shard_map-ping
        max_logit_value = jnp.broadcast_to(
            max_logit_value.astype(jnp.float32)[None, :],
            (NUM_SUBLANES, num_q_heads),
        )
        in_specs += [
            pl.BlockSpec(
                (NUM_SUBLANES, num_q_heads),
                lambda *_: (0, 0),
                memory_space=pltpu.SMEM,
            )
        ]
    else:
        in_specs.append(None)

    out_shapes = [
        jax.ShapeDtypeStruct(
            (num_q_heads, q_seq_len, head_dim_v), _unquantized_dtype(q)
        ),
    ]
    out_specs = [
        pl.BlockSpec((num_stacked_q_heads, bq, head_dim_v), out_index_map),
    ]
    if save_residuals:
        logsumexp_index_map = unravel(lambda h_block, i, j, *_: (h_block, i, 0))

        out_shapes += [
            # logsumexp
            jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
            if fuse_reciprocal
            else None,
            # l_linear
            jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32)
            if not fuse_reciprocal
            else None,
            # max_logits
            jax.ShapeDtypeStruct((num_q_heads, q_seq_len, NUM_LANES), jnp.float32),
        ]
        out_specs += [
            pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map)
            if fuse_reciprocal
            else None,
            pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map)
            if not fuse_reciprocal
            else None,
            pl.BlockSpec((num_stacked_q_heads, bq, NUM_LANES), logsumexp_index_map),
        ]
    else:
        out_shapes += [None, None, None]
        out_specs += [None, None, None]

    kernel_name = get_kernel_name(
        is_mqa=is_mqa,
        save_residuals=save_residuals,
        is_segmented=segment_ids is not None,
    )
    metadata = {"xprof_metadata": json.dumps(dataclasses.asdict(config))}

    def _fwd_cost_estimate(
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        q_segment_ids: jax.Array | None,
        kv_segment_ids: jax.Array | None,
        partial_mask_blocks: jax.Array | np.ndarray | None,
        out_shapes: list[jax.ShapeDtypeStruct | None],
        mask_sparsity: float,
    ) -> pl.CostEstimate:
        num_q_heads, q_seq_len, head_dim_qk = q.shape
        kv_seq_len, head_dim_v = v.shape[-2:]

        matmul_flops = (
            2 * q_seq_len * kv_seq_len * head_dim_qk
            + 2 * q_seq_len * kv_seq_len * head_dim_v
        )

        # This is an upper bound because `mask_sparsity` is actually the mean
        # sparsity of the non-fully masked **blocks**.
        total_flops = num_q_heads * matmul_flops * mask_sparsity

        # Count expensive exp() calls
        transcendentals = num_q_heads * q_seq_len * kv_seq_len * mask_sparsity

        inputs_ = [q, k, v, q_segment_ids, kv_segment_ids, partial_mask_blocks]
        input_bytes = sum(map(_bytes, inputs_))
        output_bytes = sum(map(_bytes, out_shapes))
        return pl.CostEstimate(
            flops=int(total_flops),
            transcendentals=int(transcendentals),
            bytes_accessed=int(input_bytes + output_bytes),
        )

    cost_estimate = config.fwd_cost_estimate or _fwd_cost_estimate(
        q,
        k,
        v,
        q_segment_ids,
        kv_segment_ids,
        mask_info.partial_mask_blocks,
        out_shapes,
        fwd_mask_sparsity,
    )

    grid_size_h = num_q_heads // num_stacked_q_heads
    if dynamic_grid and mask_info.num_active_blocks is not None:
        num_active_blocks = mask_info.num_active_blocks[0]
        grid = (grid_size_h, num_active_blocks)
        is_empty_attention_block = num_active_blocks == 0
    else:
        grid = (grid_size_h, kv_steps * (q_seq_len // bq))
        is_empty_attention_block = False

    scalar_prefetch = (
        mask_info.active_rows,
        mask_info.active_cols,
        mask_info.mask_next,
        bounds_start,
        bounds_end,
        mask_info.block_mask,
    )
    with jax.named_scope(kernel_name):
        all_out = pl.pallas_call(
            functools.partial(
                flash_attention_kernel,
                mask_value=mask_value,
                kv_steps=kv_steps,
                bq=bq,
                bkv=bkv,
                bkv_compute=bkv_compute,
                head_dim_v=head_dim_v,
                num_stacked_q_heads=num_stacked_q_heads,
                # note: fuse_reciprocal can only be False if save_residuals is True
                # fuse_reciprocal = (config.fuse_reciprocal or not save_residuals)
                fuse_reciprocal=fuse_reciprocal,
                config=config,
                segment_only_full=segment_only_full,
                mask_function=mask_function,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetch),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=[
                    pltpu.VMEM(
                        (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                    ),  # m_scratch
                    pltpu.VMEM(
                        (num_stacked_q_heads, bq, NUM_LANES), jnp.float32
                    ),  # l_scratch
                    pltpu.VMEM(
                        (num_stacked_q_heads, bq, head_dim_v), jnp.float32
                    ),  # o_scratch
                ],
            ),
            compiler_params=pltpu.CompilerParams(
                dimension_semantics=("parallel", "arbitrary"),
            ),
            out_shape=out_shapes,
            name=kernel_name,
            cost_estimate=cost_estimate,
            interpret=config.interpret,
            metadata=metadata,
        )(
            *scalar_prefetch,
            q if q_layout == QKVLayout.HEAD_DIM_MINOR else q.mT,
            k if k_layout == QKVLayout.HEAD_DIM_MINOR else k.mT,
            v if v_layout == QKVLayout.HEAD_DIM_MINOR else v.mT,
            q_segment_ids,
            kv_segment_ids,
            sinks,
            mask_info.partial_mask_blocks,
            q_sequence,
            max_logit_value,
        )
    out, logsumexp, l_linear, max_logits = all_out

    # If there is no compute to do within an attention block, then we want to
    # initialize the output and residuals to default values. Otherwise, we will
    # read uninitialized memory. This is a common case in ring attention.
    def init_if_empty(x: jax.Array, value: float) -> jax.Array:
        if not dynamic_grid:
            return x

        return jnp.where(is_empty_attention_block, value, x)

    out = init_if_empty(out, 0.0)

    if save_residuals:
        assert max_logits is not None
        max_logits = init_if_empty(max_logits[..., 0], mask_value)

        if fuse_reciprocal:
            assert logsumexp is not None
            logsumexp = init_if_empty(logsumexp[..., 0], mask_value)
        else:
            assert l_linear is not None
            log = jnp.log2 if config.use_base2_exp else jnp.log

            l_sum = l_linear[..., 0]
            logsumexp = max_logits + log(l_sum)
            out = (out / l_sum[..., None]).astype(out.dtype)
    else:
        # If we're not saving residuals, then we can't fuse the reciprocal
        # out of the kernel.
        assert fuse_reciprocal

    if save_residuals:
        stats = {"logsumexp": logsumexp, "max_logits": max_logits}
        stats = jax.tree.map(jax.lax.stop_gradient, stats)
        return out, stats
    return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "config",
        "save_residuals",
        "mask_value",
        "mask_function",
        "fwd_mask_sparsity",
        "downcast_smem_data",
    ],
)
def _splash_attention(
    fwd_mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    config: SplashConfig,
    save_residuals: bool,
    mask_value: float,
    max_logit_value: jax.Array | None = None,
    mask_function: MaskFunctionType | None,
    fwd_mask_sparsity: float,
    downcast_smem_data: bool = True,
) -> SplashCustomReturnType:
    ret = _splash_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        save_residuals=save_residuals,
        config=config,
        max_logit_value=max_logit_value,
        mask_function=mask_function,
        fwd_mask_sparsity=fwd_mask_sparsity,
        downcast_smem_data=downcast_smem_data,
    )
    if isinstance(ret, tuple):
        out, stats = ret
        if config.use_base2_exp:  # for user, output values in natural base
            stats["logsumexp"] = stats["logsumexp"] / LOG2E
            stats["max_logits"] = stats["max_logits"] / LOG2E
        return out, stats
    return ret


@jax.tree_util.register_pytree_node_class
class SplashAttentionKernel:
    def __init__(
        self,
        fwd_mask_info: MaskInfo,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.fwd_mask_info = fwd_mask_info

    @overload
    def __call__(
        self, *args, save_residuals: Literal[False], **kwargs
    ) -> jax.Array: ...

    @overload
    def __call__(
        self, *args, save_residuals: Literal[True], **kwargs
    ) -> tuple[jax.Array, dict[str, jax.Array]]: ...

    @overload
    def __call__(self, *args, **kwargs) -> SplashCustomReturnType: ...

    def __call__(self, *args, **kwargs) -> SplashCustomReturnType:
        return _splash_attention(
            self.fwd_mask_info,
            *args,
            **dict(self.kwargs, **kwargs),
        )

    def manual_sharding_spec(self, sharding: jax.sharding.NamedSharding):
        """Returns a shard_map partition spec for the kernel."""
        if self.fwd_mask_info.block_mask is not None:
            block_mask_shape = self.fwd_mask_info.block_mask.shape
            try:
                sharding.shard_shape(block_mask_shape)
            except ValueError as exc:
                raise ValueError(
                    "The sharding must divide the mask blocks evenly between devices"
                ) from exc

        if len(sharding.spec) != 1:
            raise ValueError("Only q sequence sharding is supported.")

        mask_info_specs = jax.tree.map(lambda _: sharding.spec, self.fwd_mask_info)
        if self.fwd_mask_info.partial_mask_blocks is not None:
            mask_info_specs = mask_info_specs._replace(
                partial_mask_blocks=jax.sharding.PartitionSpec()
            )
        return SplashAttentionKernel(
            mask_info_specs,
            **self.kwargs,
        )

    def tree_flatten(self):
        return ((self.fwd_mask_info,), self.kwargs)

    @classmethod
    def tree_unflatten(cls, kwargs, values):
        (fwd_mask_info,) = values
        # NamedTuples are not preserved during pytree serialization.
        return SplashAttentionKernel(MaskInfo(*fwd_mask_info), **kwargs)


def _make_splash_attention(
    mask: np.ndarray | mask_lib.Mask,
    *,
    config: SplashConfig,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
    q_seq_shards: int,
):
    if len(mask.shape) != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    if isinstance(mask, np.ndarray):
        mask = mask_lib.NumpyMask(mask)

    if (config.qk_diag_skip or config.sv_diag_skip) and not (
        isinstance(mask, mask_lib.CausalMask) and mask.offset == 0
    ):
        # The skip assumes kv > q is ALWAYS masked — a pure-causal (offset=0)
        # property. Any mask that admits a valid kv > q entry (non-zero offset,
        # bidirectional, local/sliding window, custom) would be silently
        # corrupted, so fail loud. (Square-block preconditions are enforced in
        # SplashConfig.__post_init__.)
        param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
        got_desc = (
            f"CausalMask(offset={mask.offset})"
            if isinstance(mask, mask_lib.CausalMask)
            else type(mask).__name__
        )
        raise ValueError(
            f"{param_name}=True requires a pure CausalMask(offset=0) (the skip"
            f" assumes the kv > q region is masked); got {got_desc}. Disable"
            f" {param_name} for non-causal or non-zero-offset masks."
        )

    process_fn = functools.partial(
        mask_info_lib.process_mask,
        downcast_smem_data=downcast_smem_data,
        partial_mask_blocks_dtype=partial_mask_blocks_dtype,
        q_seq_shards=q_seq_shards,
    )

    fwd_mask_info, mask_function_fwd = process_fn(
        mask,
        (config.block_q, config.block_kv),
    )
    fwd_mask_sparsity = float(np.mean(fwd_mask_info.block_mask != BlockMaskCode.EMPTY))
    fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

    return SplashAttentionKernel(
        fwd_mask_info,
        config=config,
        is_mqa=is_mqa,
        save_residuals=save_residuals,
        mask_value=mask_value,
        mask_function=mask_function_fwd,
        fwd_mask_sparsity=fwd_mask_sparsity,
        downcast_smem_data=downcast_smem_data,
    )


def _make_dynamic_splash_attention(
    mask: jax.Array,
    *,
    mesh: jax.sharding.Mesh | None = None,
    mask_spec: jax.sharding.PartitionSpec | None = None,
    config: SplashConfig,
    is_mqa: bool,
    save_residuals: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    downcast_smem_data: bool = True,
    partial_mask_blocks_dtype: jax.typing.DTypeLike = np.int8,
):
    if (mesh is not None) != (mask_spec is not None):
        raise ValueError(
            "Either both or neither of mesh and mask_spec must be specified."
        )

    if mask_spec is not None and len(mask_spec) != 1:
        raise ValueError("Only shard over the query sequence dimension.")

    if len(mask.shape) != 2:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    if config.qk_diag_skip or config.sv_diag_skip:
        param_name = "sv_diag_skip" if config.sv_diag_skip else "qk_diag_skip"
        raise ValueError(
            f"{param_name}=True requires a static pure CausalMask(offset=0)"
            f" and is unsupported for dynamic masks. Disable {param_name}."
        )

    def process_mask_shard(mask):
        return mask_info_lib.process_dynamic_mask(
            mask,
            (config.block_q, config.block_kv),
            downcast_smem_data=downcast_smem_data,
            partial_mask_blocks_dtype=partial_mask_blocks_dtype,
        )

    kwargs = {
        "config": config,
        "is_mqa": is_mqa,
        "save_residuals": save_residuals,
        "mask_value": mask_value,
        "mask_function": None,
        "fwd_mask_sparsity": 1.0,
        "downcast_smem_data": downcast_smem_data,
    }

    # If the input mask is replicated we don't need to call shard_map.
    if mask_spec is None:
        fwd_mask_info = process_mask_shard(mask)
        kernel = SplashAttentionKernel(fwd_mask_info, **kwargs)
        return kernel

    mask_info_specs = MaskInfo._make(
        None if field == "q_sequence" else mask_spec for field in MaskInfo._fields
    )

    @functools.partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=mask_spec,
        out_specs=mask_info_specs,
        check_vma=False,
    )
    def process_all_shards(mask):
        return process_mask_shard(mask)

    fwd_mask_info = process_all_shards(mask)
    kernel = SplashAttentionKernel(fwd_mask_info, **kwargs)
    kernel_spec = SplashAttentionKernel(mask_info_specs, **kwargs)

    return (kernel, kernel_spec)


make_splash_mha = functools.partial(_make_splash_attention, is_mqa=False)

make_splash_mha_single_device = functools.partial(make_splash_mha, q_seq_shards=1)

make_splash_mqa_single_device = functools.partial(
    _make_splash_attention, is_mqa=True, q_seq_shards=1
)

make_dynamic_splash_mqa = functools.partial(_make_dynamic_splash_attention, is_mqa=True)
make_dynamic_splash_mha = functools.partial(
    _make_dynamic_splash_attention, is_mqa=False
)
