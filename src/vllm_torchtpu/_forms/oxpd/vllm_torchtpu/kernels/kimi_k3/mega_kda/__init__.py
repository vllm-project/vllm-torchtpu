# SPDX-License-Identifier: Apache-2.0
# Adapted from the private upstream repository primatrix/pallas-kernel.
# Upstream contact: @pathfinder-pf.
"""Inference-only Kimi Delta Attention forward operator.

This module deliberately does not import ``chunk.py`` or register a custom
VJP.  It provides a small public API around the native segment-ID Pallas
mega-kernel so inference optimization cannot change the training path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from vllm_torchtpu.kernels.kimi_k3.mega_kda.kernel import _chunk_kda_fwd_native_segids


def _layout_supported(
    query_start_loc: jax.Array,
    num_tokens: int,
    chunk_size: int = 64,
) -> jax.Array:
    """Whether every mega-KDA tile intersects at most two live requests.

    The native boundary path handles the first and last segment in a tile. A
    third segment in the same tile would therefore be dropped. Derive the
    runtime guard from request intervals instead of scanning all token rows:
    serving buckets have few request slots but can have thousands of tokens.

    Empty request slots and starts beyond the token bucket do not overlap any
    tile. The result is a scalar device boolean so one compiled executable can
    select the mega or chunked branch for each scheduled batch.
    """
    if num_tokens % chunk_size:
        raise ValueError("mega KDA layout guard requires full token tiles")
    if query_start_loc.shape[0] <= 3:
        return jnp.asarray(True)

    starts = jnp.minimum(query_start_loc[:-1], num_tokens)
    ends = jnp.minimum(query_start_loc[1:], num_tokens)
    tile_starts = jnp.arange(0, num_tokens, chunk_size, dtype=jnp.int32)
    tile_ends = tile_starts + chunk_size
    overlaps = (
        (starts < ends)[:, None]
        & (starts[:, None] < tile_ends[None, :])
        & (ends[:, None] > tile_starts[None, :])
    )
    return jnp.all(jnp.sum(overlaps, axis=0) <= 2)


def kda_forward_inference(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    *,
    segment_ids: jax.Array,
    A_log: jax.Array | None = None,
    dt_bias: jax.Array | None = None,
    scale: float | None = None,
    initial_state: jax.Array | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    safe_gate: bool = True,
    lower_bound: float | None = None,
    chunk_size: int = 64,
    N_max: int | None = None,
    mini_batch: int | None = None,
) -> tuple[jax.Array, jax.Array | None]:
    """Run the standalone KDA inference forward kernel.

    Args:
      q: Query tensor ``[B, T, H, K]`` in BF16.
      k: Key tensor ``[B, T, H, K]`` with the same dtype as ``q``.
      v: Value tensor ``[B, T, H, V]`` with the same dtype as ``q``.
      g: Raw gate tensor ``[B, T, H, K]``.
      beta: Update coefficient ``[B, T, H]``.
      segment_ids: One-indexed packed-sequence IDs ``[B, T]`` or ``[T]``;
        zero denotes padding. Each segment is an independent request/state
        boundary, not a parallel chunk of one request.
      A_log: Per-head log decay ``[H]``.
      dt_bias: Flattened gate bias ``[H*K]``.
      scale: Attention scale. Defaults to ``K**-0.5``.
      initial_state: Optional FP32 state ``[B, N, H, K, V]`` or
        ``[N, H, K, V]``.
      output_final_state: Return FP32 final state ``[B, N, H, K, V]``.
      use_qk_l2norm_in_kernel: Must be True.
      use_gate_in_kernel: Must be True.
      safe_gate: Must be True.
      lower_bound: Sigmoid-gate lower bound in ``[-5, 0)``.
      chunk_size: Recurrently ordered token tile size; currently required to
        be 64. Tiles within one segment are not sequence-parallel.
      N_max: Static maximum number of packed segments.
      mini_batch: Optional heads per Pallas program.

    Returns:
      A pair ``(output, final_state)``. Output is ``[B, T, H, V]`` and has
      the input dtype. ``final_state`` is FP32 when requested, otherwise None.

    Raises:
      ValueError: If shapes, dtypes, or static inference options are invalid.
    """
    if q.dtype != jnp.bfloat16:
        raise ValueError(
            f"kda_forward_inference currently requires BF16, got {q.dtype}"
        )
    if not (q.dtype == k.dtype == v.dtype == g.dtype):
        raise ValueError("q, k, v, and g must have the same dtype")
    if beta.dtype != q.dtype:
        raise ValueError("beta must have the same dtype as q")
    if chunk_size != 64:
        raise ValueError(
            "the optimized inference kernel currently requires chunk_size=64"
        )
    if q.shape[:3] != k.shape[:3] or q.shape != g.shape:
        raise ValueError("q, k, and g must agree on [B, T, H, K]")
    if q.shape[:3] != v.shape[:3] or q.shape[:3] != beta.shape:
        raise ValueError("v and beta must agree with q on [B, T, H]")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k must have the same key dimension")
    if q.shape[1] % chunk_size:
        raise ValueError("T must be padded to a multiple of chunk_size")
    if segment_ids.ndim == 1:
        segment_ids = segment_ids[None, :]
    if segment_ids.shape != q.shape[:2]:
        raise ValueError(
            f"segment_ids must have shape {q.shape[:2]}, got {segment_ids.shape}"
        )
    if segment_ids.dtype != jnp.int32:
        raise ValueError(f"segment_ids must use int32, got {segment_ids.dtype}")
    if A_log is None or dt_bias is None:
        raise ValueError("fused gate activation requires A_log and dt_bias")
    if not use_qk_l2norm_in_kernel:
        raise ValueError(
            "standalone inference requires in-kernel Q/K L2 normalization "
            "to bound the triangular solve"
        )
    if not use_gate_in_kernel or not safe_gate:
        raise ValueError("standalone inference requires safe fused gate activation")
    if lower_bound is None or not (-5 <= lower_bound < 0):
        raise ValueError("safe fused gate activation requires lower_bound in [-5, 0)")
    if N_max is None:
        if initial_state is not None:
            N_max = initial_state.shape[-4]
        else:
            raise ValueError("N_max is required when initial_state is None")
    if initial_state is not None and initial_state.dtype != jnp.float32:
        raise ValueError("initial_state must use float32")

    # exp(80) is finite in FP32 and already far beyond the sigmoid saturation
    # point. Preserve valid model values while preventing exp(A_log) from
    # becoming inf before gate activation.
    A_log = jnp.minimum(A_log, jnp.asarray(80.0, dtype=A_log.dtype))

    actual_scale = q.shape[-1] ** -0.5 if scale is None else scale
    output, final_state, *_ = _chunk_kda_fwd_native_segids(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        segment_ids=segment_ids,
        initial_state=initial_state,
        output_final_state=output_final_state,
        scale=actual_scale,
        chunk_size=chunk_size,
        store_h=False,
        store_v_new=False,
        disable_recompute=False,
        only_fwd=True,
        safe_gate=safe_gate,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        mini_batch=mini_batch,
        N_max=N_max,
        residual_chunk_layout=False,
        batch_first=True,
    )
    return output, final_state
