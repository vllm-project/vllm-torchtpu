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
# ==============================================================================
"""Monolithic shard-mapped De-Dup MoE layer.

`dedup_ep_moe` wraps the fused kernel in a `shard_map` over the
expert-parallel mesh axis in seven stages:
  1. Token-parallel split and input row quantization.
  2. Router scoring, top-k selection (`pallas_select`), and renormalization.
  3. Single-collective packed routing exchange (`all_gather` over EP).
  4. Wave dispatch start (`wave_dispatch.start`) and optimization barrier.
  5. Overlapped sharded routing-table construction and `tg_assemble`.
  6. Fused EP MoE Pallas kernel (`build_dedup_moe_kernel`) and send drain.
  7. Arrival combine (`combine_arrivals`), shared expert FFN, and TP restore.

Stages 1-3 and the routing tables of stage 5 are shared with
`fused_moe.layer.fused_ep_moe_v2`; the wave dispatch, De-Dup, kernel and
combine are this layer's own.
"""

# Roadmap of upcoming optimizations (see also README.md):
# TODO(b/581885921): SparseCore gateway forwarding: forward the staged rows
#   on the SparseCore instead of the TensorCore. Blocked on compiler support
#   for zero-output SparseCore kernels and on the JAX writes-to-array-input
#   check; re-enable once available and re-measure against TensorCore
#   forwarding.
# TODO(b/581886491): SparseCore sender path (SparseCore gather and
#   contiguous senders): upstream in a follow-up once it no longer needs
#   private JAX patches.
# TODO(b/581885359): Optimize the combine phase with a hierarchical
#   reduce-scatter: first try a SparseCore collective-offload hierarchical
#   reduce-scatter; if that does not pan out, implement a TensorCore
#   hierarchical reduce-scatter.
# TODO(b/581886223): De-Dup on the return path (De-Dup collect): reduce
#   partial expert outputs at the gateway before sending them back across
#   boards.
# TODO(b/581886102): Offload the routing exchange (all-gather) to the
#   SparseCore collectives engine with multicast, once the compiler support
#   lands.
# TODO(b/581886436): Re-validate and retune the auto policy table (for
#   example, whether SparseCore forwarding helps), and add decode-specific
#   (very small t_local) and larger-slice configurations.
# TODO(b/581886912): Potential: EPLB / expert placement integration, and
#   more weight formats (W4A8, sub-channel, fp4).

from collections.abc import Callable, Mapping
import functools
import inspect
import math
import threading
from typing import Any, cast

import jax
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import jax.numpy as jnp
from tokamax._src.ops.experimental.dedup_moe.common import arg
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import dedup_plan as dedup_plan_mod
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology as topology_lib
from tokamax._src.ops.experimental.dedup_moe.dispatch import tg_assemble
from tokamax._src.ops.experimental.dedup_moe.dispatch import wave_dispatch
from tokamax._src.ops.experimental.dedup_moe.fused_kernel import kernel as kernel_lib
from tokamax._src.ops.experimental.dedup_moe.io_relayout import input_pack
from tokamax._src.ops.experimental.dedup_moe.io_relayout import output_reduce
from tokamax._src.ops.experimental.fused_moe import host as fused_host
from tokamax._src.ops.experimental.fused_moe import layer as fused_layer
from tokamax._src.ops.experimental.fused_moe import router_ops

WeightFormat = arg.WeightFormat
weight_form = arg.weight_form
weight_format_of_dtype = arg.weight_format_of_dtype
BoardTopology = topology_lib.BoardTopology
REP_BUCKETS = topology_lib.REP_BUCKETS
build_rep_lut = topology_lib.build_rep_lut

HIDDEN_LANE_BLOCK = fused_host.HIDDEN_LANE_BLOCK
MAX_ROUTING_BLOCK = fused_host.MAX_ROUTING_BLOCK
ROWBLK = fused_host.ROWBLK
U32_SUBLANE_TILE = fused_host.U32_SUBLANE_TILE
act_scale_slab_rows = fused_host.act_scale_slab_rows
align_up = fused_host.align_up
ragged_stride_bound = fused_host.ragged_stride_bound
routing_block = fused_host.routing_block
token_gather_window_rows = fused_host.token_gather_window_rows

GatheredPairs = fused_host.GatheredPairs
build_routing_tables_sharded = fused_host.build_routing_tables_sharded
pair_block_hist = fused_host.pair_block_hist
shard_count_vector = fused_host.shard_count_vector
shard_expert_slabs = fused_host.shard_expert_slabs
shard_push_tables_in_rows = fused_host.shard_push_tables_in_rows
shard_transport_tables_in_blocks = fused_host.shard_transport_tables_in_blocks

build_dedup_moe_kernel = kernel_lib.build_dedup_moe_kernel

TokenReplicaLayout = input_pack.TokenReplicaLayout
rowquant_fp8 = input_pack.rowquant_fp8
shared_expert_ffn = input_pack.shared_expert_ffn
_SHARED_EXPERT_PARAMS = frozenset(
    inspect.signature(shared_expert_ffn).parameters
)
combine_arrivals = output_reduce.combine_arrivals
pallas_select = router_ops.pallas_select

# Router scoring and the packed routing exchange, shared with
# `fused_moe.layer`.
# pylint: disable=protected-access
SCORING_FNS = fused_layer.SCORING_FNS
_apply_scoring_fn = fused_layer._apply_scoring_fn
_exchange_layout = fused_layer._exchange_layout
_pack_routing_exchange = fused_layer._pack_routing_exchange
_unpack_routing_exchange = fused_layer._unpack_routing_exchange
_relabel_expert_ids_to_mesh_order = (
    fused_layer._relabel_expert_ids_to_mesh_order
)
# pylint: enable=protected-access

# On a real TPU the token-gather DMA windows are 128-row aligned, so the
# kernel tile height (capacity) must be a multiple of this.
TPU_CAPACITY_ALIGNMENT = 128


def _loop_variant_one(x_l):
  """f32 1.0 that depends on this shard's own activations.

  XLA does not fold a float `0 * v` (v may be inf or NaN), so the result
  stays an op on x_l: per call, and per iteration of any loop whose x
  varies. The select keeps a non-finite first element from turning the
  scales into NaN, which `1 + 0 * v` alone would. x_l is the shard_map's
  local block, so this is a local slice with no collective.

  Args:
    x_l: This shard's activations.

  Returns:
    A f32 scalar equal to 1.0.
  """
  v = x_l[0, 0].astype(jnp.float32)
  return 1.0 + 0.0 * jnp.where(jnp.isfinite(v), v, 0.0)


def _on_tpu(mesh) -> bool:
  return all(d.platform == "tpu" for d in mesh.devices.flat)


_PALLAS_LUT_MAX_LEN = 16384
_PALLAS_LUT_VMEM_MAX_ROWS = 512
_PALLAS_LUT_GRID_BLOCK_ROWS = 64


def _pallas_lut_1d(lut_1d: jax.Array, idx: jax.Array) -> jax.Array:
  """Vectorized TPU VMEM LUT gather `lut_1d[idx]` via `tpu.dynamic_gather`.

  XLA lowers 1D `jnp.take` on TensorCore to a scalar SMEM loop (~200 us/layer
  for 40,960 indices). Reshaping a small LUT (`<= 16384` int32, `<= 64 KiB`,
  covering both 1D permutations `[512]` and 2D/3D per-board replica tables
  `[512, 32]`) into `(n_lut_rows, 128)` tiles in VMEM and running lane-axis
  `jnp.take_along_axis` (`tpu.dynamic_gather`) over `(8, 128)` index tiles
  executes entirely on the VPU in a few microseconds, bit-exact with `jnp.take`.

  Negative indices (e.g. `-1` sentinels from masked/NaN router rows) preserve
  `-1` in the output instead of wrapping around `lut_1d`.

  Args:
    lut_1d: 1D `[L]` int32 lookup table.
    idx: Arbitrarily shaped int32 index tensor with values in `[0, L)` or `< 0`.

  Returns:
    Tensor of shape `idx.shape` and dtype int32 equal to `lut_1d[idx]` for
    non-negative entries and `-1` where `idx < 0`.
  """
  lut_1d = lut_1d.astype(jnp.int32)
  idx_i32 = idx.astype(jnp.int32)
  lut_len = lut_1d.shape[0]
  orig_shape = idx_i32.shape
  n = math.prod(orig_shape)
  if n == 0:
    return jnp.empty(orig_shape, jnp.int32)
  if lut_len <= 0 or lut_len > _PALLAS_LUT_MAX_LEN:
    safe_idx = jnp.maximum(idx_i32, jnp.int32(0))
    taken = jnp.take(lut_1d, safe_idx, axis=0).astype(jnp.int32)
    return jnp.where(idx_i32 < jnp.int32(0), jnp.int32(-1), taken)

  n_lut_rows = (lut_len + 127) // 128
  lut_pad_rows = ((n_lut_rows + 7) // 8) * 8
  lut_pad_len = lut_pad_rows * 128
  if lut_pad_len > lut_len:
    lut_padded = jnp.pad(lut_1d, (0, lut_pad_len - lut_len))
  else:
    lut_padded = lut_1d
  lut_2d = lut_padded.reshape(lut_pad_rows, 128)

  idx_flat = idx_i32.reshape(n)
  n_idx_rows = ((n + 1023) // 1024) * 8
  use_grid = n_idx_rows > _PALLAS_LUT_VMEM_MAX_ROWS
  if use_grid:
    n_idx_rows = (
        (n_idx_rows + _PALLAS_LUT_GRID_BLOCK_ROWS - 1)
        // _PALLAS_LUT_GRID_BLOCK_ROWS
    ) * _PALLAS_LUT_GRID_BLOCK_ROWS
  idx_pad_len = n_idx_rows * 128
  if idx_pad_len > n:
    idx_padded = jnp.pad(idx_flat, (0, idx_pad_len - n))
  else:
    idx_padded = idx_flat
  idx_2d = idx_padded.reshape(n_idx_rows, 128)

  def _bcast_lut_row(lut_blk: jax.Array, local_r: int) -> jax.Array:
    return jnp.take_along_axis(
        lut_blk,
        jnp.full((8, 128), local_r, jnp.int32),
        axis=0,
        mode="promise_in_bounds",
        wrap_negative_indices=False,
    )

  def _run_block(lut_ref, idx_ref, out_ref, *, block_rows: int) -> None:
    n_tiles = block_rows // 8
    if n_lut_rows <= 16:
      lut_rows = []
      for blk in range(0, n_lut_rows, 8):
        lut_blk = lut_ref[blk : blk + 8, :]
        for r in range(blk, min(blk + 8, n_lut_rows)):
          lut_rows.append(_bcast_lut_row(lut_blk, r - blk))

      def _lookup_tile(x: jax.Array) -> jax.Array:
        lane = x & jnp.int32(127)
        row = x >> jnp.int32(7)
        acc = jnp.take_along_axis(
            lut_rows[0],
            lane,
            axis=1,
            mode="promise_in_bounds",
            wrap_negative_indices=False,
        )
        for r in range(1, n_lut_rows):
          val = jnp.take_along_axis(
              lut_rows[r],
              lane,
              axis=1,
              mode="promise_in_bounds",
              wrap_negative_indices=False,
          )
          acc = jnp.where(row == jnp.int32(r), val, acc)
        return jnp.where(x < jnp.int32(0), jnp.int32(-1), acc)

      if n_tiles == 1:
        out_ref[0:8, :] = _lookup_tile(idx_ref[0:8, :])
      else:

        def body(i, _):
          row0 = i * 8
          out_ref[pl.ds(row0, 8), :] = _lookup_tile(idx_ref[pl.ds(row0, 8), :])

        lax.fori_loop(0, n_tiles, body, None)
    else:
      # Invert the loop for larger LUTs (up to 16,384 entries = 16 blocks of 8
      # rows): hoist the 8 sublane broadcasts once per LUT block and stream the
      # index tiles in the inner loop (40x fewer broadcasts at t_local=4096).
      n_lut_blocks = lut_pad_rows // 8

      def _apply_lut_block(
          lut_blk: jax.Array, blk_row: jax.Array, *, init_acc: bool
      ) -> None:
        lut_rows = [_bcast_lut_row(lut_blk, r) for r in range(8)]

        def _apply_tile(tile_i):
          row0 = tile_i * 8
          x = idx_ref[pl.ds(row0, 8), :]
          lane = x & jnp.int32(127)
          row = x >> jnp.int32(7)
          if init_acc:
            acc = jnp.full((8, 128), -1, jnp.int32)
          else:
            acc = out_ref[pl.ds(row0, 8), :]
          for r in range(8):
            val = jnp.take_along_axis(
                lut_rows[r],
                lane,
                axis=1,
                mode="promise_in_bounds",
                wrap_negative_indices=False,
            )
            acc = jnp.where(row == blk_row + jnp.int32(r), val, acc)
          out_ref[pl.ds(row0, 8), :] = acc

        if n_tiles == 1:
          _apply_tile(jnp.int32(0))
        else:
          lax.fori_loop(0, n_tiles, lambda i, _: _apply_tile(i), None)

      # Peel block 0 so out_ref is written before any read (required by
      # Pallas interpret mode and clean Mosaic liveness) and negative
      # sentinel indices (< 0) stay -1 without an extra check.
      _apply_lut_block(lut_ref[0:8, :], jnp.int32(0), init_acc=True)

      def _lut_block_body(blk_idx, _):
        blk_row = blk_idx * 8
        _apply_lut_block(
            lut_ref[pl.ds(blk_row, 8), :], blk_row, init_acc=False
        )

      lax.fori_loop(1, n_lut_blocks, _lut_block_body, None)

  if use_grid:
    n_blocks = n_idx_rows // _PALLAS_LUT_GRID_BLOCK_ROWS
    out_2d = pl.pallas_call(
        functools.partial(_run_block, block_rows=_PALLAS_LUT_GRID_BLOCK_ROWS),
        out_shape=jax.ShapeDtypeStruct((n_idx_rows, 128), jnp.int32),
        grid=(n_blocks,),
        in_specs=[
            pl.BlockSpec((lut_pad_rows, 128), lambda i: (0, 0)),
            pl.BlockSpec((_PALLAS_LUT_GRID_BLOCK_ROWS, 128), lambda i: (i, 0)),
        ],
        out_specs=pl.BlockSpec(
            (_PALLAS_LUT_GRID_BLOCK_ROWS, 128), lambda i: (i, 0)
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="eplb_l2p_lut",
    )(lut_2d, idx_2d)
  else:
    out_2d = pl.pallas_call(
        functools.partial(_run_block, block_rows=n_idx_rows),
        out_shape=jax.ShapeDtypeStruct((n_idx_rows, 128), jnp.int32),
        in_specs=[
            pl.BlockSpec(memory_space=pltpu.VMEM),
            pl.BlockSpec(memory_space=pltpu.VMEM),
        ],
        out_specs=pl.BlockSpec(memory_space=pltpu.VMEM),
        name="eplb_l2p_lut",
    )(lut_2d, idx_2d)

  out_flat = out_2d.reshape(idx_pad_len)
  if idx_pad_len > n:
    out_flat = out_flat[:n]
  return out_flat.reshape(orig_shape)


def _relabel_logical_to_physical(
    topk_idx: jax.Array,
    l2p_map: jax.Array,
    me: jax.Array,
    *,
    board_size: int,
) -> jax.Array:
  """Maps selected logical expert ids `[t_local, topk]` to physical slots.

  Supports:
    * 1D `[e_logical]` permutation/map (`orig_to_slot[e]`).
    * 2D `[e_logical, n_buckets]` replica table (`lut[e, bucket]`).
    * 3D `[e_logical, num_boards, n_buckets]` board-aware quota-balanced
      replica LUT (`rep_lut[e, src_board, bucket]`).

  Args:
    topk_idx: `[t_local, topk]` int32 selected logical expert ids.
    l2p_map: 1D, 2D, or 3D int32 logical-to-physical slot mapping.
    me: This shard's index.
    board_size: Devices per board for 3D board-aware replica LUT indexing.

  Returns:
    `[t_local, topk]` int32 physical slot ids.
  """
  idx = topk_idx.astype(jnp.int32)
  if l2p_map.ndim == 1:
    return _pallas_lut_1d(l2p_map, idx)
  t_local, topk = idx.shape
  n_buckets = l2p_map.shape[-1]
  t_iota = lax.broadcasted_iota(jnp.int32, (t_local, topk), 0)
  k_iota = lax.broadcasted_iota(jnp.int32, (t_local, topk), 1)
  hash_val = t_iota * 7 + k_iota
  if n_buckets & (n_buckets - 1) == 0:
    bucket = hash_val & (n_buckets - 1)
    shift = n_buckets.bit_length() - 1
    flat_idx = (idx << shift) | bucket
  else:
    bucket = lax.rem(hash_val, jnp.int32(n_buckets))
    flat_idx = idx * n_buckets + bucket
  if l2p_map.ndim == 3:
    num_boards = l2p_map.shape[1]
    src_board = lax.rem(me // board_size, jnp.int32(num_boards))
    l2p_map = lax.dynamic_index_in_dim(
        l2p_map, src_board, axis=1, keepdims=False
    )
  elif l2p_map.ndim != 2:
    raise ValueError(
        f"logical_to_physical_map must be 1D, 2D, or 3D; got {l2p_map.shape}"
    )
  return _pallas_lut_1d(l2p_map.reshape(-1), flat_idx)


# Maximum SMEM bytes allowed for the fused token-gather image
# (matches fused_kernel.tg_fused.MAX_IMAGE_SMEM_BYTES).
_FUSE_TG_MAX_SMEM_BYTES = 256 * 1024

_SHARED_ACT_FNS = ("silu", "silu_and_mul_with_clamp", "gelu", "swigluoai")
_SHARED_OVERLAP_MODES = ("auto", "dispatch", "routing_ag", "split")

# De-Dup SparseCore forwards (dedup_sc_forward=True): set by
# sc_sender.use_sc_dispatch(sc_forward=True) to sc_sender.sc_forward_start.
_sc_forward_start = None

# Layer options that may also arrive in `kernel_options`, so a serving stack
# can set them through one JSON knob. They are popped before the rest goes to
# the kernel builder; an explicit keyword argument wins.
_LAYER_KERNEL_OPTION_KEYS = (
    "combine_dtype",
    "dedup_combine",
    "direct_order",
    "wave_order",
    "ramp_ratio",
    "topology_board_size",
    "stage_hoist",
    "dedup_fwd_lookahead",
    "dedup_trace_scopes",
    "return_order",
    "push_from_vmem",
    "dedup_sc_forward",
    "dedup_sc_return",
    "dedup_sc_combine",
)


def _fuse_tg_fits(ragged_stride: int, capacity: int) -> bool:
  """Whether the fused token-gather image fits in the 256 KiB SMEM share."""
  return (
      tg_assemble.gather_rows(ragged_stride, capacity)
      * tg_assemble.LANES
      * 4
      <= _FUSE_TG_MAX_SMEM_BYTES
  )


def _shared_linear(
    x: jax.Array,
    w: jax.Array,
    w_scale: jax.Array | None,
    *,
    on_tpu: bool,
) -> jax.Array:
  """Dense linear for the shared expert: bf16, per-channel fp8, or block-fp8."""
  if w_scale is None:
    return jnp.dot(
        x.astype(jnp.bfloat16),
        w.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
  if w_scale.ndim == 4:
    if on_tpu:
      return input_pack.gmm_v2.gmm_v2(
          lhs=x,
          rhs=w[None],
          group_sizes=jnp.array([x.shape[0]], dtype=jnp.int32),
          rhs_scale=w_scale,
          group_offset=jnp.array([0], dtype=jnp.int32),
          preferred_element_type=x.dtype,
          maybe_quantize_lhs=True,
          zero_initialize=True,
      )
    k_dim, n_out = w.shape
    n_blocks = w_scale.shape[1]
    block_k = k_dim // n_blocks
    w_f32 = (
        w.reshape(n_blocks, block_k, n_out).astype(jnp.float32)
        * w_scale.reshape(n_blocks, 1, n_out).astype(jnp.float32)
    ).reshape(k_dim, n_out)
    return jnp.dot(
        x.astype(jnp.float32),
        w_f32,
        preferred_element_type=jnp.float32,
    ).astype(x.dtype)
  x_q, x_s = rowquant_fp8(x.astype(jnp.bfloat16))
  out = (
      jnp.dot(x_q, w, preferred_element_type=jnp.float32)
      * x_s.astype(jnp.float32)
      * w_scale.astype(jnp.float32).reshape(1, -1)
  )
  return out.astype(x.dtype)


def _round_to(v: jax.Array, dtype: jnp.dtype) -> jax.Array:
  """f32 `v` rounded to `dtype`'s precision, kept f32.

  `lax.reduce_precision` on an f32 value is never a no-op, so XLA's excess
  precision rewrites cannot drop the rounding however the program fuses:
  every intermediate has the value a separate `dtype` op boundary (vLLM's
  torch ops) would give it.
  """
  v = v.astype(jnp.float32)
  info = jnp.finfo(dtype)
  if info.bits >= 32:
    return v
  return lax.reduce_precision(
      v, exponent_bits=info.nexp, mantissa_bits=info.nmant
  )


def _shared_act(
    gate: jax.Array, up: jax.Array, act_fn: str, out_dtype: jnp.dtype
) -> jax.Array:
  """The shared-expert gated activation; f32 values rounded to out_dtype."""
  gate = gate.astype(jnp.float32)
  up = up.astype(jnp.float32)
  if act_fn == "silu":
    return _round_to(
        _round_to(jax.nn.silu(gate), out_dtype) * up, out_dtype
    )
  if act_fn == "silu_and_mul_with_clamp":
    return _round_to(input_pack.silu_and_mul_with_clamp(gate, up), out_dtype)
  if act_fn == "gelu":
    return _round_to(
        _round_to(jax.nn.gelu(gate), out_dtype) * up, out_dtype
    )
  if act_fn == "swigluoai":
    return _round_to(input_pack.swigluoai(gate, up), out_dtype)
  raise ValueError(
      f"shared_expert_act_fn must be one of {_SHARED_ACT_FNS}; got {act_fn!r}"
  )


def _shared_ffn_up(
    x: jax.Array,
    ws1: jax.Array,
    ws1_scale: jax.Array | None,
    ws_gate: jax.Array | None,
    *,
    fs: int,
    act_fn: str,
    on_tpu: bool,
    matmul_fn: Callable[..., jax.Array] | None,
) -> tuple[jax.Array, jax.Array | None]:
  """First half of `_shared_ffn_local`: gate/up projection, activation, and gate."""
  if ws1.ndim == 3:
    ws1 = ws1[0]
  hidden = x.shape[1]
  dt = x.dtype
  mm = (
      matmul_fn
      if matmul_fn is not None
      else functools.partial(_shared_linear, on_tpu=on_tpu)
  )
  gate_up = _round_to(mm(x, ws1, ws1_scale), dt)
  mid = _shared_act(gate_up[:, :fs], gate_up[:, fs:], act_fn, dt).astype(dt)
  gate_scale = None
  if ws_gate is not None:
    gate_logits = _round_to(
        jnp.dot(
            x,
            ws_gate.reshape(hidden, 1).astype(dt),
            preferred_element_type=jnp.float32,
        ),
        dt,
    )
    gate_scale = _round_to(jax.nn.sigmoid(gate_logits), dt)
  return mid, gate_scale


def _shared_ffn_down(
    mid: jax.Array,
    gate_scale: jax.Array | None,
    ws2: jax.Array,
    ws2_scale: jax.Array | None,
    *,
    out_dtype: jnp.dtype,
    on_tpu: bool,
    matmul_fn: Callable[..., jax.Array] | None,
) -> jax.Array:
  """Second half of `_shared_ffn_local`: down projection and optional gating."""
  if ws2.ndim == 3:
    ws2 = ws2[0]
  mm = (
      matmul_fn
      if matmul_fn is not None
      else functools.partial(_shared_linear, on_tpu=on_tpu)
  )
  out = _round_to(mm(mid, ws2, ws2_scale), out_dtype)
  if gate_scale is not None:
    out = _round_to(gate_scale * out, out_dtype)
  return out


def _shared_ffn_local(
    x: jax.Array,
    ws1: jax.Array,
    ws2: jax.Array,
    ws1_scale: jax.Array | None,
    ws2_scale: jax.Array | None,
    ws_gate: jax.Array | None,
    *,
    act_fn: str,
    on_tpu: bool,
    matmul_fn: Callable[..., jax.Array] | None,
) -> jax.Array:
  """Local shared-expert FFN with optional per-token sigmoid gate.

  Returns f32 values rounded to x.dtype (see `_round_to`), so a TP partial
  can be summed in f32 and rounded once, as a bf16 all-reduce does.
  """
  if ws1.ndim == 3:
    ws1 = ws1[0]
    ws2 = ws2[0]
  fs = ws2.shape[0]
  hidden = x.shape[1]
  dt = x.dtype
  mm = (
      matmul_fn
      if matmul_fn is not None
      else functools.partial(_shared_linear, on_tpu=on_tpu)
  )
  gate_up = _round_to(mm(x, ws1, ws1_scale), dt)
  mid = _shared_act(gate_up[:, :fs], gate_up[:, fs:], act_fn, dt)
  out = _round_to(mm(mid.astype(dt), ws2, ws2_scale), dt)
  if ws_gate is not None:
    gate_logits = _round_to(
        jnp.dot(
            x,
            ws_gate.reshape(hidden, 1).astype(dt),
            preferred_element_type=jnp.float32,
        ),
        dt,
    )
    gate_scale = _round_to(jax.nn.sigmoid(gate_logits), dt)
    out = _round_to(gate_scale * out, dt)
  return out


def _validate_shared_scale(
    name: str,
    scale: jax.Array,
    *,
    k_dim: int,
    n_dim: int,
    ep: int,
    sharded: bool,
) -> None:
  """Validates a shared-expert weight scale tensor."""
  if scale.ndim == 4:
    want_g = ep if sharded else 1
    if (
        scale.shape[0] != want_g
        or scale.shape[2] != 1
        or scale.shape[3] != n_dim
        or scale.shape[1] <= 0
        or k_dim % scale.shape[1] != 0
    ):
      raise ValueError(
          f"{name} 4-D block scale must have shape [{want_g}, n_blocks, 1,"
          f" {n_dim}] with n_blocks dividing {k_dim}; got {tuple(scale.shape)}"
      )
    return
  if sharded:
    if scale.ndim != 1 or scale.shape[0] not in (ep * n_dim, ep):
      raise ValueError(
          f"sharded {name} must have shape [{ep * n_dim}], [{ep}], or"
          f" [{ep}, n_blocks, 1, {n_dim}]; got {tuple(scale.shape)}"
      )
    return
  if scale.ndim == 1 and scale.shape[0] in (n_dim, 1):
    return
  if scale.ndim == 2 and scale.shape in ((1, n_dim), (1, 1)):
    return
  raise ValueError(
      f"{name} must have shape [{n_dim}], [1, {n_dim}], [1], or"
      f" [1, n_blocks, 1, {n_dim}]; got {tuple(scale.shape)}"
  )


# Per-config cache of the shard_map'd MoE callable; a fresh shard_map
# body would re-trace the whole kernel for every hidden layer.
_LAYER_SM_CACHE = {}
_LAYER_SM_CACHE_LOCK = threading.Lock()


def dedup_ep_moe(
    x,
    w1,
    w2,
    w1_scale,
    w2_scale,
    gating,
    w1_bias=None,
    w2_bias=None,
    *,
    topk,
    renormalize,
    mesh,
    capacity,
    block=None,
    ragged_stride=None,
    weight_format=WeightFormat.FP8,
    act_fn="silu",
    rank=None,
    mesh_ep_ranks=None,
    logical_to_physical_map=None,
    expert_placement=None,
    scoring_fn="softmax",
    score_bias=None,
    hash_topk_ids=None,
    routed_scaling_factor=1.0,
    token_replica_groups=None,
    waves=4,
    dedup_topology: BoardTopology | None = None,
    dedup_min_fanout=2,
    dedup_fwd_max_runs=dedup_plan_mod.MAX_FWD_RUNS,
    ws1=None,
    ws2=None,
    ws1_scale=None,
    ws2_scale=None,
    ws_gate=None,
    shared_expert_sharded: bool = False,
    shared_expert_act_fn: str | None = None,
    shared_expert_overlap: str = "auto",
    shared_matmul_fn: Callable[..., jax.Array] | None = None,
    dedup_sc_forward=False,
    direct_order=None,
    wave_order=None,
    ramp_ratio=None,
    topology_board_size=0,
    stage_hoist=None,
    dedup_fwd_lookahead=0,
    dedup_trace_scopes=False,
    return_order="ring",
    combine_dtype=None,
    push_from_vmem=None,
    dedup_combine=False,
    dedup_sc_return=False,
    dedup_sc_combine=False,
    kernel_options: Mapping[str, Any] | None = None,
    combine_options: Mapping[str, Any] | None = None,
):
  """Run one MoE layer through the wave-dispatch (and De-Dup) fused kernel.

  x [tokens, hidden], w1 [experts, hidden, 2 * inter], w2 [experts,
  inter, hidden] and gating [tokens, experts], all sharded over the mesh
  axis. Returns [tokens, hidden], sharded like x.

  weight_format names one of the kernel's accepted weight forms (fp8 W8A8
  or bf16 W16A16). w1_scale and w2_scale are per output channel for fp8
  (e4m3) weights, [experts, 2 * inter] and [experts, hidden]; on an
  unquantized weight (bf16) there are no scales and both must be None. The
  token rows are quantized to fp8 only where the format's matmuls take fp8
  rows.

  w1_bias [experts, 1, 2 * inter] and w2_bias [experts, 1, hidden] are
  optional and independent: the gate and up halves are added to the first
  matmul's post-scale accumulator before the activation, and the down bias
  to the second matmul's post-scale row before it is quantized for the wire.

  block and ragged_stride default to the values this function derives;
  an explicitly passed one still has to divide tokens // ep * topk and to
  reach the no-drop bound respectively. act_fn selects the fused FFN
  activation from the kernel's ACT_FNS and is a build-time constant.

  token_replica_groups lists ordered groups of device-mesh indices whose
  x and gating rows are identical (for example, a TP group). Each member
  routes and computes a disjoint slice, then gathers the combined outputs
  within its group to restore the input row order. None means singleton
  groups with no token replication.

  mesh_ep_ranks names the EP rank at every device-mesh index. A non-identity
  order relabels only the selected expert ids after top-k.

  hash_topk_ids [tokens, topk] i32 replaces the router's selection with
  expert ids the caller already chose. The gate logits are still scored: the
  WEIGHTS are the scores at the selected ids.

  Wave dispatch (wave_dispatch.py): each shard sends every (token,
  destination shard) pair once, over a split-phase ragged all-to-all, in
  `waves` groups ("waves") of each destination's experts. Experts are
  visited fewest rows first and cut into waves of growing modeled compute
  (1 : 2 : 4 : ..., wave_dispatch.wave_plan), and the fused kernel starts
  each expert as soon as its wave has landed, so the later waves arrive
  under the first experts' matmuls. `waves` must divide the experts per
  shard; the mesh needs at least 3 devices. Every buffer is sized for the
  worst case, so every routing is dropless.

  dedup_topology adds De-Dup to the dispatch: a token needed by at least
  `dedup_min_fanout` devices of a remote board is sent once to that board's
  gateway device, which forwards it to those devices inside the fused
  kernel. A forwarded row counts in its target's own first wave
  (per-target-wave forwarding), so a receiver waits for exactly the rows the
  direct send would have given it in each wave. dedup_fwd_max_runs is the
  gateway forward-table depth per (served source, wave): size it statically
  (layer.plan.static_fwd_max_runs) or from the routing
  (wave_dispatch.forward_runs_needed); a gateway that would need more runs
  sends the excess staged tokens direct instead. Without a topology the
  layer runs the direct wave dispatch. Both give the same bits.

  The weight scale tables (and the optional bias tables) reach the kernel
  as `w * one` with `one = 1 + 0 * x_l[0, 0]` (`_loop_variant_one`). Inside
  an XLA while loop a Mosaic operand that is a loop-invariant while
  parameter can lose its HBM memory-space annotation, and memory-space
  assignment may then prefetch the small f32 scale tables into VMEM while
  the kernel DMAs them as HBM. The product keeps them per-call values. Same
  values bit for bit: `w * 1.0 == w`.

  ws1 [hidden, 2 * Fs] and ws2 [Fs, hidden] add a replicated shared expert
  (`shared_expert_ffn`) to every token's routed output.

  Args:
    x: [tokens, hidden] activations, sharded over the mesh axis.
    w1: [experts, hidden, 2 * inter] gate/up weights.
    w2: [experts, inter, hidden] down weights.
    w1_scale: [experts, 2 * inter] f32 scales (fp8), or None (bf16).
    w2_scale: [experts, hidden] f32 scales, or None.
    gating: [tokens, experts] router logits.
    w1_bias: Optional [experts, 1, 2 * inter] gate/up bias.
    w2_bias: Optional [experts, 1, hidden] down bias.
    topk: Experts per token.
    renormalize: Renormalize the top-k weights.
    mesh: One-axis EP mesh.
    capacity: Kernel tile height; a multiple of 128 on TPU.
    block: Routing block width, or None to derive it.
    ragged_stride: Slab rows per shard, or None to derive it.
    weight_format: fp8 or bf16.
    act_fn: Fused FFN activation.
    rank: Optional [ep, 1] i32 shard index of every device, sharded over the
      mesh; None uses arange(ep).
    mesh_ep_ranks: EP rank at every mesh index, or None for identity.
    scoring_fn: Router scoring function.
    score_bias: Optional selection bias added to the scores.
    hash_topk_ids: Optional [tokens, topk] precomputed expert ids.
    routed_scaling_factor: Scale applied to the routed weights.
    token_replica_groups: Groups of mesh indices with identical rows, or None.
    waves: Expert waves of the dispatch.
    dedup_topology: The board topology to De-Dup over, or None for the direct
      wave dispatch.
    dedup_min_fanout: Staging threshold.
    dedup_fwd_max_runs: Gateway forward runs per (served source, wave).
    ws1: Optional [hidden, 2 * Fs] shared expert gate/up weights.
    ws2: Optional [Fs, hidden] shared expert down weights.
    dedup_sc_forward: Whether gateway K4 forwards run on the SparseCore.
    direct_order: Direct-send destination order (wave_dispatch.start), or None
      for its default.
    wave_order: Wave order ("rows" or "xboard"), or None for the default.
    ramp_ratio: Growth of each wave's modeled compute, or None for the default.
    topology_board_size: Devices per board for the board-aware options when
      there is no dedup_topology.
    stage_hoist: De-Dup stage-send placement, or None for the default.
    dedup_fwd_lookahead: Gateway forwarding lookahead in waves.
    dedup_trace_scopes: Named trace scopes inside the fused kernel.
    return_order: Order of the per-expert return pushes (bit-identical).
    combine_dtype: Return-wire / combine dtype override ("fp8" or "bf16"); None
      keeps the weight format's wire. "bf16" under fp8 is not bit-exact.
    push_from_vmem: Optional override for the VMEM-to-remote return push.
    dedup_combine: Return pushes go to the local board's gateway, which reduces
      the board partials before the cross-board return.
    dedup_sc_return: Per-expert return pushes run on the SparseCore.
    dedup_sc_combine: Gateway-combine cross-board sends run on the SparseCore.

  Returns:
    [tokens, hidden] output, sharded like x.

  Raises:
    ValueError: If the operands, formats or options are inconsistent.
  """
  form = weight_form(weight_format)
  weight_format = form.name
  for name, w in (("w1", w1), ("w2", w2)):
    actual = weight_format_of_dtype(w.dtype)
    if actual != weight_format:
      raise ValueError(
          f"weight format {weight_format!r} takes "
          f"{jnp.dtype(cast(Any, form.weight_dtype)).name} expert weights and "
          f"{name} is {jnp.dtype(w.dtype).name}"
          f"{f', which is the {actual!r} form' if actual else ''}"
      )
  if form.has_scales != (w1_scale is not None):
    raise ValueError(
        f"weight format {weight_format!r} carries "
        f"{form.scale_layout} weight scales and w1_scale is "
        f"{'present' if w1_scale is not None else 'absent'}"
    )
  if (w1_scale is None) != (w2_scale is None):
    raise ValueError("both weight scales are supplied together or neither is")
  n_tokens, hidden = x.shape
  e_total = w1.shape[0]
  inter = w1.shape[2] // 2
  # The scales are constants of the weights, so whatever prepares the
  # weights gives them the kernel's [E, N] layout once; a table carrying a
  # loader's singleton axes is refused rather than reshaped on every call.
  if form.has_scales:
    for name, s, n in (
        ("w1_scale", w1_scale, 2 * inter),
        ("w2_scale", w2_scale, hidden),
    ):
      if s.ndim != 2 or s.shape[0] != e_total or s.shape[-1] != n:
        want = (e_total, n)
        raise ValueError(
            f"{name} layout {tuple(s.shape)} is not the "
            f"{form.scale_layout} form {want} the kernel "
            "takes"
        )
  if _on_tpu(mesh) and capacity % TPU_CAPACITY_ALIGNMENT:
    raise ValueError(
        f"capacity={capacity} must be a multiple of {TPU_CAPACITY_ALIGNMENT}"
        " on TPU: the kernel's token-gather DMA windows are"
        f" {TPU_CAPACITY_ALIGNMENT}-row aligned"
    )
  (ax,) = mesh.axis_names
  ep = mesh.shape[ax]
  if mesh_ep_ranks is not None:
    mesh_ep_ranks = tuple(int(rank) for rank in mesh_ep_ranks)
    if tuple(sorted(mesh_ep_ranks)) != tuple(range(ep)):
      raise ValueError(
          "mesh_ep_ranks must be a permutation of "
          f"range({ep}); got {mesh_ep_ranks}"
      )
  g_local = e_total // ep
  dedup = dedup_topology is not None
  dedup_board_size = (
      dedup_topology.board_size
      if dedup_topology is not None
      else min(dedup_plan_mod.MAX_BOARD_SIZE, ep)
  )
  if logical_to_physical_map is not None and expert_placement is not None:
    raise ValueError(
        "pass at most one of logical_to_physical_map or expert_placement"
    )
  if expert_placement is not None:
    # Static placement config: decide on the host so a caller's jax.jit trace
    # never needs a concrete bool.
    p_arr = jax.device_get(jnp.asarray(expert_placement, dtype=jnp.int32))
    if p_arr.ndim == 1:
      if p_arr.shape != (e_total,):
        raise ValueError(
            f"expert_placement must have shape ({e_total},); got {p_arr.shape}"
        )
      e_logical_cand = gating.shape[1]
      p_list = [int(i) for i in p_arr.tolist()]
      if e_total == e_logical_cand and sorted(p_list) == list(range(e_total)):
        if p_list != list(range(e_total)):
          inv = [0] * e_total
          for slot, orig in enumerate(p_list):
            inv[orig] = slot
          logical_to_physical_map = inv
      else:
        logical_to_physical_map = jnp.asarray(
            build_rep_lut(
                p_arr,
                ep=ep,
                e_total=e_logical_cand,
                board_size=dedup_board_size,
            )
        )
    else:
      logical_to_physical_map = p_arr
  has_l2p_map = logical_to_physical_map is not None
  if has_l2p_map:
    logical_to_physical_map = jnp.asarray(
        logical_to_physical_map, dtype=jnp.int32
    )
    if logical_to_physical_map.ndim not in (1, 2, 3):
      raise ValueError(
          "logical_to_physical_map must be 1D, 2D, or 3D; got"
          f" {logical_to_physical_map.shape}"
      )
    e_logical = int(logical_to_physical_map.shape[0])
  else:
    e_logical = e_total
  if n_tokens % ep or gating.shape != (n_tokens, e_logical):
    raise ValueError(
        "input and gating rows must agree and partition evenly over EP"
    )
  token_layout = TokenReplicaLayout.create(
      ep=ep,
      input_rows=n_tokens // ep,
      groups=token_replica_groups,
      row_alignment=U32_SUBLANE_TILE,
  )
  t_local = token_layout.local_rows
  n_tokens = ep * t_local
  pspec = jax.sharding.PartitionSpec
  stride_bound = ragged_stride_bound(n_tokens, topk, e_total, capacity)
  if ragged_stride is None:
    ragged_stride = stride_bound
  if block is None:
    block = routing_block(t_local, topk)
  if ragged_stride % capacity != 0:
    raise ValueError(
        f"ragged_stride must be a multiple of capacity {capacity}; got "
        f"{ragged_stride}. Pass {stride_bound}."
    )
  if ragged_stride < stride_bound:
    raise ValueError(
        f"ragged_stride {ragged_stride} is below the no-drop bound "
        f"{stride_bound}, so one shard's slab could bleed into the "
        f"next. Pass {stride_bound} or more."
    )
  has_w1_bias = w1_bias is not None
  has_w2_bias = w2_bias is not None
  has_score_bias = score_bias is not None
  has_hash_ids = hash_topk_ids is not None
  if scoring_fn not in SCORING_FNS:
    raise ValueError(
        f"scoring_fn must be one of {SCORING_FNS}; got {scoring_fn!r}"
    )
  if has_score_bias and score_bias.shape != (e_logical,):
    raise ValueError(
        f"score_bias must be one value per global expert, [{e_logical}]; "
        f"got {tuple(score_bias.shape)}"
    )
  if has_hash_ids:
    if hash_topk_ids.shape != (x.shape[0], topk):
      raise ValueError(
          "hash_topk_ids must be one expert id per token slot, "
          f"[{x.shape[0]}, {topk}]; got {tuple(hash_topk_ids.shape)}"
      )
    if not jnp.issubdtype(hash_topk_ids.dtype, jnp.integer):
      raise ValueError(
          "hash_topk_ids are expert ids and must be an integer dtype; "
          f"got {jnp.dtype(hash_topk_ids.dtype).name}"
      )
  waves = int(waves)
  if waves < 1 or g_local % waves:
    raise ValueError(
        "waves must be a positive divisor of the experts per shard "
        f"{g_local}; got {waves}"
    )
  if ep < 3:
    raise ValueError(
        "wave dispatch carries its wave cuts in the dispatch table's "
        f"padding rows, which needs ep >= 3; got ep={ep}"
    )
  if (ws1 is None) != (ws2 is None):
    raise ValueError(
        "both shared expert weights (ws1, ws2) are supplied together or"
        " neither is"
    )
  if (ws1_scale is None) != (ws2_scale is None):
    raise ValueError(
        "both shared expert scales (ws1_scale, ws2_scale) are supplied"
        " together or neither is"
    )
  has_shared_expert = ws1 is not None
  has_shared_scales = ws1_scale is not None
  has_shared_gate = ws_gate is not None
  if has_shared_scales and not has_shared_expert:
    raise ValueError("ws1_scale and ws2_scale require ws1 and ws2")
  if has_shared_gate and not has_shared_expert:
    raise ValueError("ws_gate requires ws1 and ws2")
  if shared_expert_sharded and not has_shared_expert:
    raise ValueError("shared_expert_sharded requires ws1 and ws2")
  if shared_expert_overlap not in _SHARED_OVERLAP_MODES:
    raise ValueError(
        f"shared_expert_overlap must be one of {_SHARED_OVERLAP_MODES}; got"
        f" {shared_expert_overlap!r}"
    )
  if (
      shared_expert_act_fn is not None
      and shared_expert_act_fn not in _SHARED_ACT_FNS
  ):
    raise ValueError(
        f"shared_expert_act_fn must be one of {_SHARED_ACT_FNS}; got"
        f" {shared_expert_act_fn!r}"
    )
  if ws1 is not None and ws2 is not None:
    if shared_expert_sharded:
      if ws1.ndim == 2 and ws2.ndim == 2:
        fs_local = ws2.shape[0] // ep
        valid_ws = (
            ws1.shape[0] == ep * hidden
            and ws2.shape[0] == ep * fs_local
            and fs_local > 0
            and ws1.shape[1] == 2 * fs_local
            and ws2.shape[1] == hidden
        )
      elif ws1.ndim == 3 and ws2.ndim == 3:
        fs_local = ws2.shape[1]
        valid_ws = (
            ws1.shape[0] == ep
            and ws2.shape[0] == ep
            and ws1.shape[1] == hidden
            and ws2.shape[2] == hidden
            and ws1.shape[2] == 2 * fs_local
        )
      else:
        valid_ws = False
        fs_local = 0
      if not valid_ws:
        raise ValueError(
            "sharded shared expert weights must have shapes"
            f" ws1=[{ep * hidden}, 2*Fs] and ws2=[{ep}*Fs, {hidden}] (or"
            f" [{ep}, {hidden}, 2*Fs] and [{ep}, Fs, {hidden}]); got"
            f" ws1={tuple(ws1.shape)}, ws2={tuple(ws2.shape)}"
        )
    else:
      fs_local = ws2.shape[0] if ws2.ndim == 2 else 0
      if (
          ws1.ndim != 2
          or ws2.ndim != 2
          or ws1.shape[0] != hidden
          or ws2.shape[1] != hidden
          or ws1.shape[1] != 2 * fs_local
      ):
        raise ValueError(
            f"shared expert weights must have shapes ws1=[{hidden}, 2*Fs] and"
            f" ws2=[Fs, {hidden}]; got ws1={tuple(ws1.shape)},"
            f" ws2={tuple(ws2.shape)}"
        )
    if ws1_scale is not None and ws2_scale is not None:
      _validate_shared_scale(
          "ws1_scale",
          ws1_scale,
          k_dim=hidden,
          n_dim=2 * fs_local,
          ep=ep,
          sharded=shared_expert_sharded,
      )
      _validate_shared_scale(
          "ws2_scale",
          ws2_scale,
          k_dim=fs_local,
          n_dim=hidden,
          ep=ep,
          sharded=shared_expert_sharded,
      )
    if ws_gate is not None and tuple(ws_gate.shape) not in (
        (1, hidden),
        (hidden, 1),
        (hidden,),
    ):
      raise ValueError(
          f"ws_gate must have shape [1, {hidden}], [{hidden}, 1], or"
          f" [{hidden}]; got {tuple(ws_gate.shape)}"
      )
  has_4d_shared_scales = (
      ws1_scale is not None and ws1_scale.ndim == 4
  ) or (ws2_scale is not None and ws2_scale.ndim == 4)
  use_legacy_shared = (
      has_shared_expert
      and not shared_expert_sharded
      and not has_shared_gate
      and not has_4d_shared_scales
      and shared_expert_act_fn is None
      and shared_matmul_fn is None
  )
  shared_act = (
      act_fn
      if use_legacy_shared
      else (shared_expert_act_fn if shared_expert_act_fn is not None else "silu")
  )
  if shared_expert_overlap == "auto":
    shared_overlap = (
        "dispatch"
        if (not has_shared_expert or use_legacy_shared)
        else "routing_ag"
    )
  else:
    shared_overlap = shared_expert_overlap
  shared_needs_reduce = (
      has_shared_expert
      and shared_expert_sharded
      and len(token_layout.groups[0]) > 1
  )
  on_tpu = _on_tpu(mesh)
  dedup = dedup_topology is not None
  dedup_fwd_max_runs = int(dedup_fwd_max_runs)
  if dedup:
    assert dedup_topology is not None
    if dedup_topology.num_devices != ep:
      raise ValueError(
          f"dedup_topology covers {dedup_topology.num_devices} devices; the"
          f" mesh axis has ep={ep}"
      )
    if waves > dedup_plan_mod.MAX_PTW_WAVES:
      raise ValueError(
          "per-target-wave forwarding packs the target wave in 4 bits: "
          f"waves <= {dedup_plan_mod.MAX_PTW_WAVES}; got {waves}"
      )
    if dedup_fwd_max_runs < 1:
      raise ValueError(
          f"dedup_fwd_max_runs must be >= 1; got {dedup_fwd_max_runs}"
      )
    dedup_tables = dedup_topology.tables()
    dedup_slots = dedup_tables.max_served
    dedup_board_size = dedup_tables.board_size
  else:
    dedup_tables = None
    dedup_slots = 0
    dedup_board_size = dedup_plan_mod.MAX_BOARD_SIZE
  kernel_kw = dict(kernel_options) if kernel_options else {}
  combine_kw = dict(combine_options) if combine_options else {}
  kernel_opt_key = tuple(sorted(kernel_kw.items()))
  combine_opt_key = tuple(sorted(combine_kw.items()))
  # With `fuse_tg_assemble` the fused kernel assembles token_gather itself
  # from the landed sender lists, so it is built inside local_fn where the
  # list buffer's height is known.
  fuse_tg = (
      bool(kernel_kw.pop("fuse_tg_assemble", False))
      and not dedup
      and _fuse_tg_fits(ragged_stride, capacity)
  )
  # Layer options carried in kernel_options; explicit kwargs win.
  layer_opts = {
      k: kernel_kw.pop(k) for k in _LAYER_KERNEL_OPTION_KEYS if k in kernel_kw
  }
  if layer_opts:
    defaults = dict(
        combine_dtype=None,
        dedup_combine=False,
        direct_order=None,
        wave_order=None,
        ramp_ratio=None,
        topology_board_size=0,
        stage_hoist=None,
        dedup_fwd_lookahead=0,
        dedup_trace_scopes=False,
        return_order="ring",
        push_from_vmem=None,
        dedup_sc_forward=False,
        dedup_sc_return=False,
        dedup_sc_combine=False,
    )
    current = dict(
        combine_dtype=combine_dtype,
        dedup_combine=dedup_combine,
        direct_order=direct_order,
        wave_order=wave_order,
        ramp_ratio=ramp_ratio,
        topology_board_size=topology_board_size,
        stage_hoist=stage_hoist,
        dedup_fwd_lookahead=dedup_fwd_lookahead,
        dedup_trace_scopes=dedup_trace_scopes,
        return_order=return_order,
        push_from_vmem=push_from_vmem,
        dedup_sc_forward=dedup_sc_forward,
        dedup_sc_return=dedup_sc_return,
        dedup_sc_combine=dedup_sc_combine,
    )
    for k, v in layer_opts.items():
      if current[k] == defaults[k]:
        current[k] = v
    combine_dtype = current["combine_dtype"]
    dedup_combine = current["dedup_combine"]
    direct_order = current["direct_order"]
    wave_order = current["wave_order"]
    ramp_ratio = current["ramp_ratio"]
    topology_board_size = current["topology_board_size"]
    stage_hoist = current["stage_hoist"]
    dedup_fwd_lookahead = current["dedup_fwd_lookahead"]
    dedup_trace_scopes = current["dedup_trace_scopes"]
    return_order = current["return_order"]
    push_from_vmem = current["push_from_vmem"]
    dedup_sc_forward = current["dedup_sc_forward"]
    dedup_sc_return = current["dedup_sc_return"]
    dedup_sc_combine = current["dedup_sc_combine"]
  dedup_sc_forward = bool(dedup_sc_forward)
  dedup_sc_return = bool(dedup_sc_return)
  dedup_sc_combine = bool(dedup_sc_combine)
  dedup_combine = bool(dedup_combine)
  if (dedup_sc_forward or dedup_sc_return or dedup_sc_combine) and (
      not dedup or _sc_forward_start is None
  ):
    raise ValueError(
        "dedup_sc_forward/dedup_sc_return/dedup_sc_combine need dedup_topology"
        " and the SparseCore dispatch"
        " (sc_sender.wrap_layer(..., sc_forward=True))"
    )
  if dedup_sc_combine and not dedup_combine:
    raise ValueError("dedup_sc_combine=True requires dedup_combine=True")
  # The board size of board-aware return orders and the gateway combine: the
  # De-Dup topology's, else topology_board_size, else a return_board_size
  # given directly in kernel_options.
  return_board_size = (
      dedup_board_size
      if dedup_slots
      else int(topology_board_size)
      or int(kernel_kw.get("return_board_size", 0))
  )
  if dedup_combine:
    if (
        return_board_size <= 0
        or ep % return_board_size
        or ep == return_board_size
    ):
      raise ValueError(
          "dedup_combine=True requires >= 2 boards of return_board_size"
          f" devices (ep={ep}, return_board_size={return_board_size})"
      )
    if fuse_tg:
      raise ValueError("dedup_combine=True does not support fuse_tg_assemble")
  if dedup_sc_return and fuse_tg:
    raise ValueError("dedup_sc_return=True does not support fuse_tg_assemble")
  # Only non-default options reach the kernel builder, so the default build
  # (and its program name and cache entry) is unchanged.
  extra_kernel_kw: dict[str, Any] = {}
  if dedup_slots and int(dedup_fwd_lookahead):
    extra_kernel_kw["fwd_lookahead"] = int(dedup_fwd_lookahead)
  if dedup_trace_scopes:
    extra_kernel_kw["trace_scopes"] = True
  if return_order != "ring":
    extra_kernel_kw["return_order"] = return_order
  if return_order != "ring" or dedup_combine:
    extra_kernel_kw["return_board_size"] = int(return_board_size)
  if combine_dtype is not None:
    extra_kernel_kw["combine_dtype"] = combine_dtype
  if push_from_vmem is not None:
    extra_kernel_kw["push_from_vmem_override"] = bool(push_from_vmem)
  if dedup_combine:
    extra_kernel_kw["dedup_combine"] = True
  if dedup_sc_return:
    extra_kernel_kw["dedup_sc_return"] = True
  if dedup_sc_forward:
    extra_kernel_kw["dedup_sc_forward"] = True
  kernel_kw.update(extra_kernel_kw)
  lane_blocks = hidden // HIDDEN_LANE_BLOCK
  if combine_dtype == "bf16":
    wire_dtype_resolved = jnp.bfloat16
  elif combine_dtype == "fp8":
    wire_dtype_resolved = jnp.float8_e4m3fn
  else:
    wire_dtype_resolved = form.wire_dtype
  fp8_wire = jnp.dtype(cast(Any, wire_dtype_resolved)) == jnp.dtype(
      jnp.float8_e4m3fn
  )
  start_kw = {
      k: v
      for k, v in (
          ("direct_order", direct_order),
          ("wave_order", wave_order),
          ("ramp_ratio", ramp_ratio),
          ("stage_hoist", stage_hoist),
      )
      if v is not None
  }
  if start_kw and int(topology_board_size):
    start_kw["board_size"] = int(topology_board_size)
  if dedup_combine:
    static_recv_rows = align_up(
        (ep // return_board_size) * t_local * topk
        + (ROWBLK - 1) * e_total
        + ROWBLK,
        ROWBLK,
    )
  else:
    static_recv_rows = align_up(t_local * topk + (ROWBLK - 1) * e_total, ROWBLK)
  build_kw = dict(
      g_local=g_local,
      capacity=capacity,
      hidden=hidden,
      inter=inter,
      ep=ep,
      ragged_rows_alloc=ragged_stride,
      weight_format=weight_format,
      act_fn=act_fn,
      has_w1_bias=has_w1_bias,
      has_w2_bias=has_w2_bias,
      waves=waves,
      dedup_slots=dedup_slots,
      dedup_fwd_max_runs=dedup_fwd_max_runs,
      dedup_board_size=dedup_board_size,
      **kernel_kw,
  )
  kfn = None if fuse_tg else build_dedup_moe_kernel(**build_kw)

  def local_fn(x_l, rank_l, w1_l, w2_l, *scales_gating_biases):
    """One shard's half of the layer: route, dispatch, kernel, combine."""
    # The scales come before the gating logits and the biases after
    # them, which is the operand order the scaled formats have always
    # had: a format that supplies no scales drops them out of the middle
    # rather than moving anything that stayed.
    operands = iter(scales_gating_biases)
    w1s_l = next(operands) if form.has_scales else None
    w2s_l = next(operands) if form.has_scales else None
    gate_l = next(operands)
    hash_l = next(operands) if has_hash_ids else None
    score_bias_l = next(operands) if has_score_bias else None
    l2p_l = next(operands) if has_l2p_map else None
    w1b_raw = next(operands) if has_w1_bias else None
    w2b_raw = next(operands) if has_w2_bias else None
    ws1_l = next(operands) if has_shared_expert else None
    ws2_l = next(operands) if has_shared_expert else None
    ws1s_l = next(operands) if has_shared_scales else None
    ws2s_l = next(operands) if has_shared_scales else None
    wsg_l = next(operands) if has_shared_gate else None

    # ------------------------------------------------------------------
    # Step 1: Token-parallel split and input row quantization.
    # ------------------------------------------------------------------
    # This shard's index arrives as data, not from `lax.axis_index`: that
    # lowers to `partition-id`, which the SPMD partitioner rejects when the
    # module is serialized and recompiled (for example through jax.export).
    me = rank_l[0, 0]
    x_raw = x_l
    x_l = token_layout.split(x_l, me)
    gate_l = token_layout.split(gate_l, me, fill_value=-jnp.inf)
    if hash_l is not None:
      hash_l = token_layout.split(hash_l, me)
    if form.quantized_activations:
      # Whatever produced x_l -- the replica slice above, the caller's t
      # bucket padding (modular.dedup_moe concatenates pad rows in the same
      # XLA program), any model op -- must not fuse into the row
      # quantization: on TPU the fused form changes the BF16 rounding of the
      # row quantization, so fp8 rows (and every output of their tokens)
      # would change with the producer. Materialize x_l first.
      x_l = lax.optimization_barrier(x_l)
    rows_bf16 = x_l.astype(jnp.bfloat16)
    if form.quantized_activations:
      q_l, row_scale_l = rowquant_fp8(rows_bf16)
    else:
      # The rows go on the wire as they arrived. There is no scale to
      # carry, so nothing downstream builds or ships one.
      q_l, row_scale_l = rows_bf16, None
    q_v = q_l.reshape(t_local, hidden // HIDDEN_LANE_BLOCK, HIDDEN_LANE_BLOCK)

    # ------------------------------------------------------------------
    # Step 2: Router scoring, top-k selection, and renormalization.
    # ------------------------------------------------------------------
    scores = _apply_scoring_fn(gate_l, scoring_fn)
    select_rows = MAX_ROUTING_BLOCK
    while t_local % select_rows:
      select_rows //= 2
    if hash_l is not None:
      # The selection was made by the caller's table, so the selector
      # does not run at all; the weights are still the scores at the
      # selected ids, which is the only thing the selector would have
      # contributed here.
      topk_idx = hash_l.astype(jnp.int32)
      topk_weights = jnp.take_along_axis(scores, topk_idx, axis=-1)
    elif score_bias_l is None:
      topk_weights, topk_idx = pallas_select(
          scores, topk=topk, block_rows=select_rows
      )
    else:
      # Selection is based on (scores + score_bias_l); the weights come
      # from the scores.
      topk_weights, topk_idx = pallas_select(
          scores + score_bias_l.astype(jnp.float32)[None, :],
          topk=topk,
          block_rows=select_rows,
          weight_scores=scores,
      )
    if l2p_l is not None:
      topk_idx = _relabel_logical_to_physical(
          topk_idx,
          l2p_l,
          me,
          board_size=dedup_board_size,
      )
    topk_idx = _relabel_expert_ids_to_mesh_order(
        topk_idx, g_local=g_local, mesh_ep_ranks=mesh_ep_ranks
    )
    # A row of scores carrying no real value routes nowhere. It was
    # dropped by arithmetic accident: the selector gives every slot of
    # such a row a large negative sentinel, and the renormalization below
    # divided by their sum, which OVERFLOWED float32 to -inf and so
    # returned exactly zero. That outcome is a property of the
    # accumulation dtype and the association order, not of the design --
    # a wider accumulator, a reassociating compiler pass, or a different
    # renormalization would silently turn such a row from dropped into
    # routed at full weight to expert zero. Mask it explicitly instead.
    row_routes = (
        jnp.any(jnp.isfinite(scores), axis=-1, keepdims=True)
        & jnp.any(jnp.isfinite(gate_l), axis=-1, keepdims=True)
    )
    if renormalize:
      denom = jnp.maximum(topk_weights.sum(axis=-1, keepdims=True), 1e-20)
      topk_weights = topk_weights / jnp.where(row_routes, denom, 1.0)
    if routed_scaling_factor != 1.0:
      topk_weights = topk_weights * jnp.float32(routed_scaling_factor)
    topk_weights = jnp.where(row_routes, topk_weights, 0.0)
    # Send masked/padded rows to out-of-range sentinel `e_total` so they
    # contribute zero pairs to histograms, dispatch plans, and expert compute.
    topk_idx = jnp.where(row_routes, topk_idx, jnp.int32(e_total))

    # ------------------------------------------------------------------
    # Step 3: Single-collective packed routing exchange across EP.
    # ------------------------------------------------------------------
    # A count of the masked rows would be the other half of this: an
    # incident today is degraded output against a completely clean log.
    # It is not added here because the only in-trace channel for it,
    # jax.debug.print, makes the program refuse to lower for TPU from a
    # host, which is how this repository verifies that a change served
    # the same program. A counter belongs in a metrics channel outside
    # the trace.
    # One all-gather carries the indices, the row scales where the rows
    # were quantized, and the per-expert pair counts the routing plan
    # would otherwise exchange in a collective of its own.
    fast_glue = getattr(wave_dispatch.source_plan, "_FAST_GLUE", False)
    fast_glue_v2 = getattr(wave_dispatch.source_plan, "_FAST_GLUE_V2", False)
    local_blocks = topk_idx.astype(jnp.int32).reshape(
        t_local * topk // block, block
    )
    skip_block_hist = (
        fast_glue_v2
        and waves <= 2
        and not dedup_combine
        and t_local * topk <= 1024
    )
    if skip_block_hist:
      hist_me = jnp.sum(
          (
              local_blocks.reshape(-1)[:, None]
              == jnp.arange(e_total, dtype=jnp.int32)[None, :]
          ).astype(jnp.int32),
          axis=0,
      )
      block_hist_carry = hist_me
    else:
      block_hist = pair_block_hist(local_blocks, e_total)
      hist_me = block_hist.sum(axis=0)
      block_hist_carry = block_hist
    scale_bits_l = None
    if row_scale_l is not None:
      scale_bits_l = lax.bitcast_convert_type(
          row_scale_l[:, 0].astype(jnp.float32), jnp.int32
      )
    def _run_shared_local(x_s, w1_s, w2_s, s1_s, s2_s, wg_s):
      assert w1_s is not None and w2_s is not None
      if use_legacy_shared:
        shared_kw: dict[str, Any] = dict(act_fn=act_fn)
        if has_shared_scales and "ws1_scale" in _SHARED_EXPERT_PARAMS:
          shared_kw.update(ws1_scale=s1_s, ws2_scale=s2_s)
        if "q_l" in _SHARED_EXPERT_PARAMS:
          shared_kw.update(
              q_l=q_l if form.quantized_activations else None,
              row_scale_l=row_scale_l,
          )
        return shared_expert_ffn(x_s, w1_s, w2_s, **shared_kw).reshape(
            t_local, hidden
        )
      return _shared_ffn_local(
          x_s,
          w1_s,
          w2_s,
          s1_s,
          s2_s,
          wg_s,
          act_fn=shared_act,
          on_tpu=on_tpu,
          matmul_fn=shared_matmul_fn,
      )

    early_enabled = getattr(wave_dispatch, "early_dispatch_enabled", None)
    use_early_dispatch = (
        early_enabled is not None
        and not dedup_combine
        and not start_kw
        and early_enabled(waves=waves, dedup=dedup)
    )
    compact_w1_ag = (
        fast_glue_v2
        and waves == 1
        and not dedup
        and not dedup_combine
        and not use_early_dispatch
    )
    if compact_w1_ag:
      cnt_dir_me = wave_dispatch.w1_dest_counts(
          topk_idx, ep=ep, g_local=g_local
      )
      cnt_at = align_up(ep, HIDDEN_LANE_BLOCK)
      packed = jnp.concatenate([
          cnt_dir_me,
          jnp.zeros((cnt_at - ep,), jnp.int32),
          hist_me,
      ])
      layout = (0, cnt_at, cnt_at + e_total)
      weight_at = align_up(layout[-1], HIDDEN_LANE_BLOCK)
    else:
      exchange_quantized = form.quantized_activations and not fast_glue_v2
      layout = _exchange_layout(
          t_local, topk, e_total, exchange_quantized
      )
      packed = _pack_routing_exchange(
          local_blocks.reshape(-1),
          scale_bits_l if exchange_quantized else None,
          hist_me,
          layout,
      )
      weight_at = align_up(layout[-1], HIDDEN_LANE_BLOCK)
      if dedup_combine:
        # The gateway combine weighs board partials, so the routing weights
        # ride the same all-gather.
        weight_bits_l = lax.bitcast_convert_type(
            topk_weights.reshape(-1).astype(jnp.float32), jnp.int32
        )
        weight_pad = align_up(t_local * topk, HIDDEN_LANE_BLOCK) - (
            t_local * topk
        )
        packed = jnp.concatenate([
            packed,
            jnp.zeros((weight_at - layout[-1],), jnp.int32),
            weight_bits_l,
            jnp.zeros((weight_pad,), jnp.int32),
        ])
    exchange_width = packed.shape[0]
    wd_early = None
    if use_early_dispatch:
      # Direct W=1: the exchange needs only this shard's own routing, so it
      # is issued before the routing all-gather, which then runs under the
      # dispatch DMAs in flight.
      wd_early = wave_dispatch.start_early(  # pyrefly: ignore[missing-attribute]
          q_v,
          topk_idx.astype(jnp.int32),
          hist_me,
          me,
          ep=ep,
          g_local=g_local,
          n_tokens=n_tokens,
          scale_bits=scale_bits_l,
      )
      packed, fut = lax.optimization_barrier((packed, wd_early.fut))
      wd_early = wd_early._replace(fut=fut)
    shared_y = None
    shared_mid = None
    shared_gscale = None
    x_shared_ag = None
    prepared_w1 = None
    async_ag_fn = getattr(wave_dispatch, "async_routing_ag_enabled", None)
    use_async_ag = (
        async_ag_fn is not None
        and not dedup_combine
        and wd_early is None
        and async_ag_fn(waves=waves, dedup=dedup, t_local=t_local)
    )
    if use_async_ag:
      topk_w1 = topk_idx.astype(jnp.int32)
      hist_w1 = hist_me
      sb_w1 = scale_bits_l
      pallas_ag = on_tpu or (
          getattr(wave_dispatch._uninit_hbm, "__name__", "") == "_uninit_hbm"  # pylint: disable=protected-access
          and not getattr(wave_dispatch.source_plan, "_PALLAS_INTERPRET", False)
      )
      if pallas_ag:
        ag_fut = wave_dispatch.routing_ag_start(
            packed,
            me,
            ep=ep,
            direct_order=direct_order,
            board_size=int(topology_board_size),
        )
        if has_shared_expert and shared_overlap == "routing_ag":
          x_shared_ag = x_raw if shared_expert_sharded else x_l
          (
              topk_w1,
              hist_w1,
              sb_w1,
              x_shared_ag,
              ws1_l,
              ws2_l,
              ws1s_l,
              ws2s_l,
              wsg_l,
              ag_fut,
          ) = lax.optimization_barrier((
              topk_w1,
              hist_w1,
              sb_w1,
              x_shared_ag,
              ws1_l,
              ws2_l,
              ws1s_l,
              ws2s_l,
              wsg_l,
              ag_fut,
          ))
        elif (
            has_shared_expert
            and shared_overlap == "split"
            and not use_legacy_shared
        ):
          x_shared_ag = x_raw if shared_expert_sharded else x_l
          (
              topk_w1,
              hist_w1,
              sb_w1,
              x_shared_ag,
              ws1_l,
              ws1s_l,
              wsg_l,
              ag_fut,
          ) = lax.optimization_barrier((
              topk_w1,
              hist_w1,
              sb_w1,
              x_shared_ag,
              ws1_l,
              ws1s_l,
              wsg_l,
              ag_fut,
          ))
        else:
          topk_w1, hist_w1, sb_w1, ag_fut = lax.optimization_barrier(
              (topk_w1, hist_w1, sb_w1, ag_fut)
          )
      else:
        if has_shared_expert and shared_overlap in ("routing_ag", "split"):
          x_shared_ag = x_raw if shared_expert_sharded else x_l
        ag_fut = None
      prepared_w1 = wave_dispatch.prepare_local_w1(
          topk_w1,
          hist_w1,
          me,
          ep=ep,
          g_local=g_local,
          scale_bits=sb_w1,
      )
      if has_shared_expert and shared_overlap == "routing_ag":
        shared_y = _run_shared_local(
            x_shared_ag, ws1_l, ws2_l, ws1s_l, ws2s_l, wsg_l
        )
      elif (
          has_shared_expert
          and shared_overlap == "split"
          and not use_legacy_shared
      ):
        assert x_shared_ag is not None and ws1_l is not None and ws2_l is not None
        shared_mid, shared_gscale = _shared_ffn_up(
            x_shared_ag,
            ws1_l,
            ws1s_l,
            wsg_l,
            fs=ws2_l.shape[-2],
            act_fn=shared_act,
            on_tpu=on_tpu,
            matmul_fn=shared_matmul_fn,
        )
      if pallas_ag:
        assert ag_fut is not None
        if shared_y is not None:
          prepared_w1, shared_y, ag_fut = lax.optimization_barrier(
              (prepared_w1, shared_y, ag_fut)
          )
        elif shared_mid is not None:
          prepared_w1, shared_mid, shared_gscale, ag_fut = (
              lax.optimization_barrier(
                  (prepared_w1, shared_mid, shared_gscale, ag_fut)
              )
          )
        else:
          prepared_w1, ag_fut = lax.optimization_barrier((prepared_w1, ag_fut))
        exchange = wave_dispatch.routing_ag_done(
            ag_fut, ep=ep, width=exchange_width
        )
      else:
        exchange = lax.all_gather(
            packed,
            ax,
            axis=0,
            tiled=True,
        ).reshape(ep, exchange_width)
    else:
      if has_shared_expert and shared_overlap == "routing_ag":
        x_shared_ag = x_raw if shared_expert_sharded else x_l
        packed, x_shared_ag, ws1_l, ws2_l, ws1s_l, ws2s_l, wsg_l = (
            lax.optimization_barrier(
                (packed, x_shared_ag, ws1_l, ws2_l, ws1s_l, ws2s_l, wsg_l)
            )
        )
      elif (
          has_shared_expert
          and shared_overlap == "split"
          and not use_legacy_shared
      ):
        x_shared_ag = x_raw if shared_expert_sharded else x_l
        packed, x_shared_ag, ws1_l, ws1s_l, wsg_l = lax.optimization_barrier(
            (packed, x_shared_ag, ws1_l, ws1s_l, wsg_l)
        )
      exchange = lax.all_gather(
          packed,
          ax,
          axis=0,
          tiled=True,
      ).reshape(ep, exchange_width)
      if has_shared_expert and shared_overlap == "routing_ag":
        shared_y = _run_shared_local(
            x_shared_ag, ws1_l, ws2_l, ws1s_l, ws2s_l, wsg_l
        )
      elif (
          has_shared_expert
          and shared_overlap == "split"
          and not use_legacy_shared
      ):
        assert x_shared_ag is not None and ws1_l is not None and ws2_l is not None
        shared_mid, shared_gscale = _shared_ffn_up(
            x_shared_ag,
            ws1_l,
            ws1s_l,
            wsg_l,
            fs=ws2_l.shape[-2],
            act_fn=shared_act,
            on_tpu=on_tpu,
            matmul_fn=shared_matmul_fn,
        )
    if compact_w1_ag:
      cnt_at = layout[1]
      recv_cnt_w1 = (
          lax.dynamic_index_in_dim(
              exchange[:, :ep], me, axis=1, keepdims=False
          )
          .sum(keepdims=True)
          .astype(jnp.int32)
      )
      rows_by_dest = lax.slice(exchange, (0, cnt_at), (ep, cnt_at + e_total))
      pairs_g = jnp.zeros((0,), jnp.int32)
    else:
      recv_cnt_w1 = None
      pairs_g, _, rows_by_dest = _unpack_routing_exchange(
          exchange[:, : layout[-1]] if dedup_combine else exchange,
          layout,
          t_local,
          topk,
          e_total,
      )
    weights_g = None
    if dedup_combine:
      weights_g = lax.bitcast_convert_type(
          lax.slice(exchange, (0, weight_at), (ep, weight_at + t_local * topk)),
          jnp.float32,
      )

    # ------------------------------------------------------------------
    # Step 4: Wave dispatch start and optimization barrier.
    # ------------------------------------------------------------------
    if prepared_w1 is not None:
      wd = wave_dispatch.launch_prepared_w1(
          prepared_w1,
          q_v,
          rows_by_dest,
          me,
          ep=ep,
          g_local=g_local,
          n_tokens=n_tokens,
          topk=topk,
          recv_cnt=recv_cnt_w1,
          pairs=None if compact_w1_ag else pairs_g,
          **start_kw,
      )
    elif wd_early is not None:
      wd = wave_dispatch.finish_early(  # pyrefly: ignore[missing-attribute]
          wd_early,
          pairs_g,
          rows_by_dest,
          me,
          ep=ep,
          g_local=g_local,
          t_local=t_local,
      )
    else:
      wd = wave_dispatch.start(
          q_v,
          None if compact_w1_ag else pairs_g,
          rows_by_dest,
          me,
          ep=ep,
          g_local=g_local,
          waves=waves,
          scale_bits=scale_bits_l,
          dedup_tables=dedup_tables,
          min_fanout=dedup_min_fanout,
          fwd_max_runs=dedup_fwd_max_runs,
          pairs_me=topk_idx.astype(jnp.int32),
          hist_me=hist_me,
          recv_cnt_override=recv_cnt_w1,
          **start_kw,
      )
    # SparseCore return / gateway-combine staging buffers (SparseCore path).
    sc_bufs = ()
    if dedup_sc_return:
      sc_bufs += (
          wave_dispatch._uninit_hbm(  # pylint: disable=protected-access
              (static_recv_rows, lane_blocks, HIDDEN_LANE_BLOCK),
              wire_dtype_resolved,
          ),
          wave_dispatch._uninit_hbm(  # pylint: disable=protected-access
              (ragged_stride, lane_blocks, HIDDEN_LANE_BLOCK),
              wire_dtype_resolved,
          ),
      )
    if dedup_sc_combine:
      rem_boards = (ep // return_board_size) - 1
      sc_bufs += (
          wave_dispatch._uninit_hbm(  # pylint: disable=protected-access
              (rem_boards * t_local, lane_blocks, HIDDEN_LANE_BLOCK),
              x_l.dtype,
          ),
          wave_dispatch._uninit_hbm(  # pylint: disable=protected-access
              (rem_boards * t_local, lane_blocks, HIDDEN_LANE_BLOCK),
              x_l.dtype,
          ),
      )
    extra = ((weights_g,) if dedup_combine else ()) + sc_bufs
    # The barrier makes every table op below consume a value the start
    # kernel produced, so none can be hoisted in front of it. The
    # weight-scale relayouts ride it too, so they also land under the
    # exchange.
    # pairs_g rides it rather than its expert_blocks view: pinned at
    # the barrier, the view was relaid out for its consumers after it
    # (three copies per layer); formed after it, XLA lays it out for
    # them directly.
    w1b_k = w1b_raw.reshape(g_local, 2 * inter) if w1b_raw is not None else None
    w2b_k = w2b_raw.reshape(g_local, hidden) if w2b_raw is not None else None
    if has_shared_expert and shared_overlap == "dispatch":
      g_arrays = (
          pairs_g,
          local_blocks,
          block_hist_carry,
          rows_by_dest,
          ws1_l,
          ws2_l,
          ws1s_l,
          ws2s_l,
          wsg_l,
      )
    elif (
        has_shared_expert
        and shared_overlap == "split"
        and not use_legacy_shared
    ):
      g_arrays = (
          pairs_g,
          local_blocks,
          block_hist_carry,
          rows_by_dest,
          shared_mid,
          shared_gscale,
          ws2_l,
          ws2s_l,
      )
    else:
      g_arrays = (pairs_g, local_blocks, block_hist_carry, rows_by_dest)
    if dedup:
      wave_of_b = wd.wave_of
      if extra:
        (g_arrays, wave_of_b, w1s_l, w2s_l, extra), fut = (
            lax.optimization_barrier(
                ((g_arrays, wave_of_b, w1s_l, w2s_l, extra), wd.fut)
            )
        )
      else:
        (g_arrays, wave_of_b, w1s_l, w2s_l), fut = lax.optimization_barrier(
            ((g_arrays, wave_of_b, w1s_l, w2s_l), wd.fut)
        )
      wd = wd._replace(fut=fut, wave_of=wave_of_b)
      # The gateway's forward tables are built after the start kernel,
      # so they overlap the dispatch DMAs in flight.
      wd = wave_dispatch.materialize_smem_ops(
          wd,
          g_arrays[0],
          me,
          t_local=t_local,
          waves=waves,
          dedup_tables=dedup_tables,
          min_fanout=dedup_min_fanout,
      )
    else:
      if extra:
        (g_arrays, w1s_l, w2s_l, extra), fut = lax.optimization_barrier(
            ((g_arrays, w1s_l, w2s_l, extra), wd.fut)
        )
      else:
        (g_arrays, w1s_l, w2s_l), fut = lax.optimization_barrier(
            ((g_arrays, w1s_l, w2s_l), wd.fut)
        )
      wd = wd._replace(fut=fut)
    if dedup_combine:
      weights_g, sc_bufs = extra[0], extra[1:]
    else:
      sc_bufs = extra
    b_idx = 0
    if dedup_sc_return:
      recv_ret_buf, contrib_buf = sc_bufs[b_idx], sc_bufs[b_idx + 1]
      b_idx += 2
    else:
      recv_ret_buf = contrib_buf = None
    if dedup_sc_combine:
      send_rem_buf, recv_rem_buf = sc_bufs[b_idx], sc_bufs[b_idx + 1]
    else:
      send_rem_buf = recv_rem_buf = None
    if dedup_sc_forward and not (dedup_sc_return or dedup_sc_combine):
      # The gateway K4 forwards, issued from the SparseCore as each stage
      # run lands (the fused kernel is built without them).
      assert _sc_forward_start is not None
      _sc_forward_start(wd, ep=ep)
    if has_shared_expert and shared_overlap == "dispatch":
      ws1_l, ws2_l, ws1s_l, ws2s_l, wsg_l = g_arrays[4:]
      x_shared_disp = x_raw if shared_expert_sharded else x_l
      shared_y = _run_shared_local(
          x_shared_disp, ws1_l, ws2_l, ws1s_l, ws2s_l, wsg_l
      )
    elif (
        has_shared_expert
        and shared_overlap == "split"
        and not use_legacy_shared
    ):
      shared_mid, shared_gscale, ws2_l, ws2s_l = g_arrays[4:]
      assert shared_mid is not None and ws2_l is not None
      shared_y = _shared_ffn_down(
          shared_mid,
          shared_gscale,
          ws2_l,
          ws2s_l,
          out_dtype=x_l.dtype,
          on_tpu=on_tpu,
          matmul_fn=shared_matmul_fn,
      )
    if skip_block_hist:
      block_hist_b = jnp.zeros((0, e_total), jnp.int32)
      hist_me_b = g_arrays[2]
    else:
      block_hist_b = g_arrays[2]
      hist_me_b = block_hist_b.sum(axis=0) if fast_glue_v2 else None
    gathered = GatheredPairs(
        expert_blocks=(
            jnp.zeros((0, block), jnp.int32)
            if compact_w1_ag
            else g_arrays[0].reshape(n_tokens * topk // block, block)
        ),
        local_blocks=g_arrays[1],
        block_hist=block_hist_b,
        rows_by_dest=g_arrays[3],
        n_tokens=n_tokens,
        topk=topk,
    )
    q_g = wd.fut[3]  # the receive buffer, filled wave by wave

    # ------------------------------------------------------------------
    # Step 5: Overlapped sharded routing tables and token-gather assembly.
    # ------------------------------------------------------------------
    # tg_assemble builds token_gather and scale_slab from the sender
    # lists, so the routing plan's slab rows (and, on an unquantized wire,
    # its mirror positions) are unused and dead-code-eliminated.
    nw = 2 if form.quantized_activations else 1
    if fast_glue and not dedup_combine:
      skt = fused_host.build_shard_kernel_tables(
          gathered,
          me,
          e_total=e_total,
          ep=ep,
          t_local=t_local,
          block=block,
          tile_m=capacity,
          shard_stride=ragged_stride,
          hist_me=hist_me_b,
          fast_v2=fast_glue_v2 and waves <= 2,
      )
      pos_arr, mirror_pos_arr = skt.pos, skt.mirror_pos
      kernel_tables_fast = skt.kernel_tables
      rbd_me_fast = skt.rbd_me
      block_tables = row_tables = routing = None
      expert_rows, slab_base = skt.expert_rows, skt.slab_base
      counts_fast = skt.counts
      run_start_me = skt.run_start_me
    else:
      kernel_tables_fast = None
      rbd_me_fast = None
      counts_fast = None
      routing_kw = {}
      if dedup_combine:
        routing_kw["skip_pos"] = True
      if getattr(wave_dispatch, "COMPACT_PLAN", False) or dedup_combine:
        routing_kw["skip_slab_row"] = True
      routing = build_routing_tables_sharded(
          None,
          me,
          gathered=gathered,
          e_total=e_total,
          ep=ep,
          t_local=t_local,
          block=block,
          tile_m=capacity,
          shard_stride=ragged_stride,
          **routing_kw,
      )
      pos_arr, mirror_pos_arr = routing.pos, routing.mirror_pos
      block_tables = shard_transport_tables_in_blocks(
          routing, me, e_total=e_total, ep=ep
      )
      row_tables = shard_push_tables_in_rows(
          routing, me, e_total=e_total, ep=ep
      )
      expert_rows, slab_base = shard_expert_slabs(
          routing, me, e_total=e_total, ep=ep
      )
      run_start_me = lax.dynamic_slice(
          routing.run_start_aligned, (me * g_local, 0), (g_local, ep)
      )
    # token_gather (and the fp8 activation scale slab when the rows are
    # quantized) come from the sender lists landed in wd.tg_land, with
    # no XLA scatter.
    scale_rows = act_scale_slab_rows(ragged_stride) if nw == 2 else None
    tg_runs, tg_meta = tg_assemble.assemble_runs(
        rows_by_dest,
        me,
        ep=ep,
        g_local=g_local,
        t_local=t_local,
        topk=topk,
        nw=nw,
        slab_base=slab_base,
        run_start=run_start_me,
        rbd_me=rbd_me_fast,
    )
    tg_rows = tg_assemble.gather_rows(ragged_stride, capacity)
    # The list semaphore is a DMA semaphore from the start kernel; an
    # emulated exchange (CPU tests) hands over complete lists and an integer
    # placeholder instead, and then there is nothing to wait for.
    tg_sem = (
        None
        if (wd.tg_sem is None or jnp.issubdtype(wd.tg_sem.dtype, jnp.integer))
        else wd.tg_sem
    )
    if fuse_tg:
      # The fused kernel builds token_gather (and the scale slab) from
      # wd.tg_land and the runs in its prologue.
      token_gather = scale_slab = None
      tg_kw = dict(
          tg_land=wd.tg_land, tg_runs=tg_runs, tg_meta=tg_meta, tg_sem=tg_sem
      )
      kfn_l = build_dedup_moe_kernel(
          **build_kw,
          fuse_tg_assemble=True,
          tg_land_rows=int(wd.tg_land.shape[0]),
          tg_n_tokens=n_tokens,
      )
    else:
      tg_flat, scale_bits_slab = tg_assemble.tg_assemble(
          wd.tg_land,
          tg_runs,
          tg_meta,
          tg_sem,
          ep=ep,
          nw=nw,
          n_tokens=n_tokens,
          tg_rows=tg_rows,
          scale_rows=scale_rows,
      )
      token_gather = tg_flat[
          : ragged_stride + token_gather_window_rows(capacity)
      ]
      scale_slab = (
          None
          if scale_bits_slab is None
          else lax.bitcast_convert_type(scale_bits_slab, jnp.float32)
      )
      tg_kw = {}
      assert kfn is not None
      kfn_l = kfn
    recv_rows = static_recv_rows
    if shared_y is not None:
      if fuse_tg:
        shared_y, tg_runs_anchor = lax.optimization_barrier((shared_y, tg_runs))
        tg_kw["tg_runs"] = tg_runs_anchor
      else:
        shared_y, token_gather = lax.optimization_barrier(
            (shared_y, token_gather)
        )
    # Fewest rows first, the order the waves were laid out in, which the
    # kernel's per-wave wait keys on; empty experts come last.
    visit = lax.dynamic_index_in_dim(wd.visit_order, me, axis=0, keepdims=False)
    gw_pos = gw_coef = gw_meta = gw_mirror = None
    if dedup_combine:
      assert routing is not None
      # Return pushes go to the local board's gateway gw(s, my_b), which
      # reduces the board partials before the cross-board return.
      gw_push_dst, counts, gw_recv_base_flat = (
          output_reduce.shard_gateway_return_tables(  # pyrefly: ignore[missing-attribute]
              routing,
              expert_rows,
              me,
              e_total=e_total,
              ep=ep,
              board_size=return_board_size,
          )
      )
      assert weights_g is not None
      gw_tables = output_reduce.build_gateway_combine_tables(  # pyrefly: ignore[missing-attribute]
          gw_recv_base_flat,
          g_arrays[0],
          weights_g,
          me,
          e_total=e_total,
          ep=ep,
          board_size=return_board_size,
          t_local=t_local,
          topk=topk,
          block=block,
          sentinel_row=recv_rows - 1,
          **(
              dict(
                  mirror_tile_m=capacity,
                  run_start_aligned=routing.run_start_aligned,
              )
              if fp8_wire
              else {}
          ),
      )
      gw_pos, gw_coef, gw_meta = gw_tables[:3]
      gw_mirror = gw_tables[3] if fp8_wire else None
      push_dst_blocks = gw_push_dst
    else:
      # One pass builds every count the kernel needs, still spread over
      # the expert-parallel axis; the kernel's scalar core closes them.
      if counts_fast is not None:
        counts = counts_fast
      else:
        counts = shard_count_vector(
            routing, expert_rows, me, e_total=e_total, ep=ep
        )
      if kernel_tables_fast is not None:
        push_dst_blocks = kernel_tables_fast[3]
      else:
        assert block_tables is not None
        push_dst_blocks = block_tables[5]

    # ------------------------------------------------------------------
    # Step 6: Fused EP MoE Pallas kernel and outgoing send drain.
    # ------------------------------------------------------------------
    # The kernel takes four of the seven transport tables and one of the
    # three push tables. The two totals are rows of `counts`, and three
    # more are the same array as one already here -- push_src is the
    # commit offset, push_len is the commit length, and the receive row
    # offset is the push destination in rows -- so each goes over once and
    # the kernel derives the rest. Every table shipped is another blocking
    # HBM->SMEM DMA at the head of the kernel, and the prefetch also has to
    # fit in 1 MB of SMEM.
    if kernel_tables_fast is not None:
      kernel_tables = kernel_tables_fast
    else:
      assert block_tables is not None and row_tables is not None
      kernel_tables = (*block_tables[:3], push_dst_blocks, row_tables[0])
    # q_g is the exchange's receive buffer, and the kernel waits for each
    # wave on the semaphores the start kernel gave.
    wave_kw = dict(disp_tab=wd.disp_tab, wave_sems=wd.fut[1])
    if dedup:
      smem_ops = wd.smem_ops
      assert smem_ops is not None
      wave_kw.update(
          stage=wd.stage,
          stage_tab=smem_ops.stage_tab,
          fwd_tab=smem_ops.fwd_tab,
          fwd_cnt=smem_ops.fwd_cnt,
          stage_sems=wd.stage_sems,
      )
    f_idx = 11
    ret_tab = None
    if dedup_sc_return:
      sc_ret_sems, ret_recv_sem = wd.fut[f_idx], wd.fut[f_idx + 1]
      f_idx += 2
      n_visit_1d = counts[fused_host.COUNT_VISITS].sum(keepdims=True)
      dest_order = lax.rem(
          me + 1 + jnp.arange(ep, dtype=jnp.int32), jnp.int32(ep)
      )
      if dedup_combine:
        my_b_base = lax.div(
            me, jnp.int32(return_board_size)
        ) * jnp.int32(return_board_size)
        ret_target_dev = my_b_base + lax.rem(
            dest_order, jnp.int32(return_board_size)
        )
      else:
        ret_target_dev = dest_order
      v_idx = visit[:, None]
      d_idx = dest_order[None, :]
      ret_src_rows = (kernel_tables[2][v_idx, d_idx] * ROWBLK).reshape(-1)
      ret_dst_rows = (push_dst_blocks[v_idx, d_idx] * ROWBLK).reshape(-1)
      ret_true_rows = jnp.where(
          kernel_tables[1][v_idx, d_idx] > 0, kernel_tables[4][v_idx, d_idx], 0
      ).reshape(-1)
      ret_tab = jnp.concatenate([
          n_visit_1d,
          ret_target_dev,
          ret_src_rows,
          ret_dst_rows,
          ret_true_rows,
      ]).astype(jnp.int32)
      wave_kw.update(
          recv_buf=recv_ret_buf,
          contrib_buf=contrib_buf,
          ret_recv_sem=ret_recv_sem,
          sc_ret_sems=sc_ret_sems,
      )
    if dedup_sc_combine:
      sc_comb_sems, comb_recv_sem = wd.fut[f_idx], wd.fut[f_idx + 1]
    else:
      sc_comb_sems = comb_recv_sem = None
    if dedup_sc_forward and (dedup_sc_return or dedup_sc_combine):
      assert _sc_forward_start is not None
      comb_tb = output_reduce.combine_step_tokens(t_local) or t_local
      _sc_forward_start(
          wd,
          ep=ep,
          ret_tab=ret_tab,
          contrib_buf=contrib_buf,
          recv_ret_buf=recv_ret_buf,
          g_local=g_local,
          send_rem_buf=send_rem_buf,
          recv_rem_buf=recv_rem_buf,
          board_size=return_board_size,
          t_local=t_local,
          comb_tb=comb_tb,
      )
    # Loop-safe scales (see the docstring): the small per-expert tables
    # leave this function as products with a per-call value, never as the
    # bare (possibly loop-invariant) parameters an enclosing XLA while
    # loop would let memory-space assignment move into VMEM. Exact: * 1.0.
    one = _loop_variant_one(x_l)
    w1s_l, w2s_l, w1b_k, w2b_k = (
        None if a is None else a * one.astype(a.dtype)
        for a in (w1s_l, w2s_l, w1b_k, w2b_k)
    )
    arrivals, arrival_scales = kfn_l(
        kernel_tables,
        token_gather,
        expert_rows,
        slab_base,
        q_g,
        scale_slab,
        w1_l,
        w2_l,
        w1s_l,
        w2s_l,
        w1b_k,
        w2b_k,
        recv_rows=recv_rows,
        visit=visit,
        counts=counts,
        rank=rank_l,
        **wave_kw,
        **tg_kw,
    )
    if dedup_sc_return:
      order_after = wave_dispatch.finish(wd, arrival_scales, ep=ep)
    else:
      arrivals = wave_dispatch.finish(wd, arrivals, ep=ep)
      order_after = None

    # ------------------------------------------------------------------
    # Step 7: Arrival combine, shared expert FFN, and TP restore.
    # ------------------------------------------------------------------
    if shared_needs_reduce:
      assert shared_y is not None
      arrivals, shared_y = lax.optimization_barrier((arrivals, shared_y))
      width = len(token_layout.groups[0])
      padding = t_local * width - token_layout.input_rows
      shared_padded = (
          jnp.pad(shared_y, ((0, padding), (0, 0))) if padding else shared_y
      )
      shared_y = lax.psum_scatter(
          shared_padded,
          ax,
          scatter_dimension=0,
          tiled=True,
          axis_index_groups=token_layout.groups,
      )
      shared_y = _round_to(shared_y, x_l.dtype)
    # The destination's own table: one arrival row per selection slot.
    combine_extra = {} if order_after is None else dict(order_after=order_after)
    if dedup_combine:
      assert gw_pos is not None and gw_coef is not None and gw_meta is not None
      gw_extra = dict(combine_extra)
      if fp8_wire:
        gw_extra.update(arrival_scales=arrival_scales, mirror_pos_sorted=gw_mirror)
      combined = output_reduce.combine_gateway_arrivals(  # pyrefly: ignore[missing-attribute]
          arrivals,
          gw_pos,
          gw_coef,
          gw_meta,
          t_local=t_local,
          topk=topk,
          ep=ep,
          board_size=return_board_size,
          out_dtype=x_l.dtype,
          sc_combine=dedup_sc_combine,
          send_rem_buf=send_rem_buf,
          recv_rem_buf=recv_rem_buf,
          comb_recv_sem=comb_recv_sem,
          sc_comb_sems=sc_comb_sems,
          **gw_extra,
      )
    elif combine_kw:
      orig_cbuild = output_reduce.build_combine_kernel
      output_reduce.build_combine_kernel = functools.partial(
          orig_cbuild, **combine_kw
      )
      try:
        combined = combine_arrivals(
            arrivals,
            arrival_scales,
            pos_arr,
            mirror_pos_arr,
            topk_weights,
            x_l.dtype,
            **combine_extra,
        )
      finally:
        output_reduce.build_combine_kernel = orig_cbuild
    else:
      combined = combine_arrivals(
          arrivals,
          arrival_scales,
          pos_arr,
          mirror_pos_arr,
          topk_weights,
          x_l.dtype,
          **combine_extra,
      )
    # Masked rows (row_routes=False) never route or receive valid arrivals, so
    # their combine inputs may read unwritten HBM (NaN * 0.0 = NaN); zero them
    # explicitly before adding any shared expert output.
    combined = jnp.where(row_routes, combined, jnp.zeros((), combined.dtype))
    if shared_y is not None:
      combined = (
          combined.astype(jnp.float32) + shared_y.astype(jnp.float32)
      ).astype(x_l.dtype)
    return token_layout.restore(combined, ax)

  # The biases ride the same expert-axis sharding as the weights they
  # belong to, so no shard ever holds a bias for an expert it does not own.
  bias_args = tuple(b for b in (w1_bias, w2_bias) if b is not None)
  shared_args = (ws1, ws2) if has_shared_expert else ()
  shared_scale_args = (ws1_scale, ws2_scale) if has_shared_scales else ()
  shared_gate_args = (ws_gate,) if has_shared_gate else ()
  scale_args = (w1_scale, w2_scale) if form.has_scales else ()
  # `rank` rides the mesh axis so each shard reads its own index out of it.
  # A caller that serializes and recompiles the module MUST pass it as a
  # real operand:
  # an arange built here is a constant of the traced function, and sharding a
  # jit-internal constant compiles to `dynamic-slice(constant, partition-id)`
  # -- which XLA then refuses when the module is recompiled for one device
  # ("partitionId instruction is not supported for SPMD partitioning").
  # Under plain `jax.jit` with a mesh the constant form is fine, so the
  # default keeps the device tests and any JAX-native caller working.
  if rank is None:
    rank = jnp.arange(ep, dtype=jnp.int32).reshape(ep, 1)
  # The score bias is the one operand that is NOT sharded on the mesh axis:
  # it is indexed by GLOBAL expert id and every shard scores the full
  # expert set, so `pspec()` hands each shard the whole [e_total] vector.
  # Sharding it would give each shard a 1/ep slice of the experts and
  # silently bias the wrong ones.
  score_bias_args = (score_bias,) if has_score_bias else ()
  l2p_args = (logical_to_physical_map,) if has_l2p_map else ()
  hash_args = (hash_topk_ids,) if has_hash_ids else ()
  shared_spec = pspec(ax) if shared_expert_sharded else pspec()
  in_specs = (
      (pspec(ax),) * (5 + len(scale_args) + len(hash_args))
      + (pspec(),) * (len(score_bias_args) + len(l2p_args))
      + (pspec(ax),) * len(bias_args)
      + (shared_spec,) * (len(shared_args) + len(shared_scale_args))
      + (pspec(),) * len(shared_gate_args)
  )
  args = (
      (x, rank, w1, w2)
      + scale_args
      + (gating,)
      + hash_args
      + score_bias_args
      + l2p_args
      + bias_args
      + shared_args
      + shared_scale_args
      + shared_gate_args
  )
  named_sharding = jax.sharding.NamedSharding
  args = tuple(
      jax.device_put(a, named_sharding(mesh, sp))
      for a, sp in zip(args, in_specs)
  )
  # local_fn closes over config-static values only -- the per-call data
  # are the shard_map arguments -- so this key is exact.
  key = (
      mesh,
      n_tokens,
      hidden,
      e_total,
      e_logical,
      inter,
      topk,
      bool(renormalize),
      capacity,
      block,
      ragged_stride,
      weight_format,
      act_fn,
      mesh_ep_ranks,
      (
          logical_to_physical_map.shape
          if logical_to_physical_map is not None
          else False
      ),
      (
          logical_to_physical_map.dtype
          if logical_to_physical_map is not None
          else False
      ),
      token_layout,
      x.dtype,
      w1.dtype,
      w2.dtype,
      form.has_scales and w1_scale.dtype,
      gating.dtype,
      has_w1_bias and w1_bias.dtype,
      has_w2_bias and w2_bias.dtype,
      scoring_fn,
      score_bias.dtype if score_bias is not None else False,
      has_hash_ids and hash_topk_ids.dtype,
      float(routed_scaling_factor),
      waves,
      dedup_topology,
      int(dedup_min_fanout),
      dedup_fwd_max_runs,
      ws1.shape if ws1 is not None else False,
      ws1.dtype if ws1 is not None else False,
      ws1_scale.shape if ws1_scale is not None else False,
      ws1_scale.dtype if ws1_scale is not None else False,
      ws2_scale.shape if ws2_scale is not None else False,
      ws2_scale.dtype if ws2_scale is not None else False,
      ws2.shape if ws2 is not None else False,
      ws2.dtype if ws2 is not None else False,
      ws_gate.shape if ws_gate is not None else False,
      ws_gate.dtype if ws_gate is not None else False,
      bool(shared_expert_sharded),
      shared_act,
      shared_overlap,
      use_legacy_shared,
      shared_matmul_fn,
      dedup_sc_forward,
      dedup_sc_return,
      dedup_sc_combine,
      _sc_forward_start,
      tuple(sorted(start_kw.items())),
      tuple(sorted(extra_kernel_kw.items())),
      kernel_opt_key,
      combine_opt_key,
      getattr(wave_dispatch.source_plan, "_FAST_GLUE", False),
      getattr(wave_dispatch.source_plan, "_FAST_GLUE_V2", False),
      getattr(wave_dispatch, "_ASYNC_ROUTING_AG", False),
  )
  sm = _LAYER_SM_CACHE.get(key)
  if sm is None:
    with _LAYER_SM_CACHE_LOCK:
      sm = _LAYER_SM_CACHE.get(key)
      if sm is None:
        sm = jax.shard_map(
            local_fn,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=pspec(ax),
            check_vma=False,
        )
        # The exchange's start kernel hands its DMA semaphores to the
        # fused kernel and to the drain, which is only valid inside
        # one XLA program. Under a caller's trace this jit is inlined;
        # an eager call would otherwise run the shard_map one
        # primitive at a time, a program each.
        sm = jax.jit(sm)
        _LAYER_SM_CACHE[key] = sm
  return sm(*args)
