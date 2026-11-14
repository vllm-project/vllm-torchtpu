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
"""Loop-free ragged depthwise causal 1D convolution in JAX for Mamba-2 and KDA."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from vllm_torchtpu.layers.core.mamba_utils import token_sequence_ids

# Optional module-top-level Tokamax experimental seam (never imported inside a
# traced or compiled hot path). Disabled by default while vllm-torchtpu builds
# without a tokamax pin.
USE_TOKAMAX: bool = False

try:
    # pyrefly: ignore[missing-import]
    from tokamax._src.ops.experimental.ragged_causal_conv1d import (
        ragged_causal_conv1d as _tokamax_ragged_causal_conv1d,
    )
except ImportError:
    _tokamax_ragged_causal_conv1d = None


def _short_conv_tokens(
    x: jax.Array,
    conv_state: jax.Array,
    w_kd: jax.Array,
    conv_bias: jax.Array | None,
    qsl: jax.Array,
    state_indices: jax.Array,
    has_init: jax.Array,
    num_valid_seqs: jax.Array,
    first_seq: jax.Array,
    first_token: jax.Array,
    total_tokens: jax.Array,
    *,
    activation: str | None,
) -> tuple[jax.Array, jax.Array]:
    """Loop-free ragged convolution body over static ``kernel_size`` taps."""
    num_tokens, dim = x.shape
    num_seqs = state_indices.shape[0]
    num_slots = conv_state.shape[0]
    kernel_size = w_kd.shape[0]
    history = kernel_size - 1

    sequence_ids = token_sequence_ids(qsl, num_tokens, num_seqs)
    query_lens = jnp.maximum(qsl[1:] - qsl[:-1], 0)
    sequence_all = jnp.arange(num_seqs, dtype=jnp.int32)
    valid_slot = (state_indices >= 0) & (state_indices < num_slots)
    use_init = has_init & valid_slot & (sequence_all < num_valid_seqs)

    # Gather per-sequence carried windows; zero out sequences without history
    # or with invalid sentinel slot indices (e.g. PAD_SLOT_ID = -1).
    safe_read_slots = jnp.where(use_init, state_indices, 0)
    state_broadcast = use_init.reshape((num_seqs,) + (1,) * (conv_state.ndim - 1))
    windows = jnp.where(state_broadcast, conv_state[safe_read_slots], 0)
    windows_2d = windows.reshape(num_seqs, history, dim)
    windows_flat = windows_2d.reshape(num_seqs * history, dim)

    token_ids = jnp.arange(num_tokens, dtype=jnp.int32)
    position = token_ids - qsl[sequence_ids]

    taps = []
    for j in range(kernel_size):
        shift = history - j
        from_x = jnp.pad(x, ((shift, 0), (0, 0)))[:num_tokens]
        flat = sequence_ids * history + position + j
        from_window = windows_flat[jnp.clip(flat, 0, num_seqs * history - 1)]
        taps.append(jnp.where((position >= shift)[:, None], from_x, from_window))

    acc = jnp.sum(
        jnp.stack(taps, axis=0).astype(jnp.float32)
        * w_kd[:, None, :].astype(jnp.float32),
        axis=0,
    )
    if conv_bias is not None:
        acc = acc + conv_bias.astype(jnp.float32)[None, :]
    if activation in ("silu", "swish"):
        acc = jax.nn.silu(acc)

    output = acc.astype(x.dtype)
    walked = (token_ids >= first_token) & (token_ids < total_tokens)
    output = jnp.where(walked[:, None], output, 0)

    # Update outgoing history windows from the last `history` inputs of each
    # sequence, preserving older incoming state rows when query_len < history.
    rows = jnp.arange(history, dtype=jnp.int32)
    local = query_lens[:, None] - history + rows
    from_x = x[jnp.clip(qsl[:-1][:, None] + local, 0, num_tokens - 1)]
    from_window = jnp.take_along_axis(
        windows_2d,
        jnp.clip(query_lens[:, None] + rows, 0, history - 1)[..., None],
        axis=1,
    )
    new_window = jnp.where((local >= 0)[..., None], from_x, from_window)
    if conv_state.ndim > 3:
        new_window = new_window.reshape(num_seqs, history, *conv_state.shape[2:])

    # Route inactive/padded sequences and -1 sentinels to `num_slots` with
    # `mode="drop"` so live requests in slot 0 are never overwritten.
    keep = (
        (query_lens > 0)
        & (sequence_all >= first_seq)
        & (sequence_all < num_valid_seqs)
        & (qsl[:-1] < total_tokens)
        & valid_slot
    )
    write_indices = jnp.where(keep, state_indices, num_slots)
    new_state = conv_state.at[write_indices].set(
        new_window.astype(conv_state.dtype), mode="drop"
    )
    return output, new_state


def ragged_causal_conv1d(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None = None,
    query_start_loc: jax.Array | None = None,
    state_indices: jax.Array | None = None,
    distribution: jax.Array | None = None,
    has_initial_state: jax.Array | None = None,
    *,
    seq_lens: jax.Array | None = None,
    start_seq: jax.Array | int | None = None,
    activation: str | None = None,
    kernel_size: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal 1D convolution over a ragged batch without a token loop.

    Extracted and hardened from ``kda_attention_op.kimi_short_conv_scan`` so
    Mamba-2 and KDA share one pure-JAX implementation without Pallas buffer
    donation hazards or slot-0 clobbering.

    Args:
      x: ``(num_tokens, dim)`` in ``bfloat16`` or ``float32``.
      conv_state: ``(num_slots, kernel_size - 1, dim)`` (3D) or
        ``(num_slots, kernel_size - 1, 1, dim)`` (4D), in ``bfloat16`` or
        ``float32``.
      conv_weight: ``(dim, 1, kernel_size)`` (3D PyTorch depthwise layout) or
        ``(kernel_size, dim)`` (2D tap-major layout).
      conv_bias: Optional ``(dim,)`` bias added in ``float32`` before activation.
      query_start_loc: ``(num_seqs + 1,)`` cumulative token offsets.
      state_indices: ``(num_seqs,)`` cache slot per sequence (``-1`` ignored).
      distribution: Optional ``(3,)`` ``[num_decode, ..., num_valid_seqs]``.
      has_initial_state: Optional ``(num_seqs,)`` bool mask; derived from
        ``seq_lens > query_lens`` when omitted and ``seq_lens`` is provided.
      seq_lens: Optional ``(num_seqs,)`` total sequence lengths.
      start_seq: Optional sequence index below which tokens/states are skipped.
      activation: ``None``, ``"silu"``, or ``"swish"``.
      kernel_size: Optional static kernel width check against ``conv_state``.

    Returns:
      ``(output, new_conv_state)`` matching ``x`` and ``conv_state`` shapes/dtypes.
    """
    if query_start_loc is None or state_indices is None:
        raise ValueError("query_start_loc and state_indices are required.")
    if x.ndim != 2:
        raise ValueError(f"Expected 2D x (num_tokens, dim), got {x.shape}.")
    num_tokens, dim = x.shape

    if conv_state.ndim not in (3, 4):
        raise ValueError(
            "Expected 3D (num_slots, K-1, dim) or 4D (num_slots, K-1, 1, dim) "
            f"conv_state, got {conv_state.shape}."
        )
    history = conv_state.shape[1]
    resolved_kernel_size = history + 1
    if resolved_kernel_size < 2:
        raise ValueError(
            "Expected kernel_size >= 2 (conv_state.shape[1] >= 1), "
            f"got {conv_state.shape}."
        )
    if kernel_size is not None and kernel_size != resolved_kernel_size:
        raise ValueError(
            f"conv_state history {history} does not match kernel_size={kernel_size}."
        )
    if math.prod(conv_state.shape[2:]) != dim:
        raise ValueError(
            f"conv_state trailing shape {conv_state.shape[2:]} "
            f"does not match x dim {dim}."
        )

    if conv_weight.ndim == 3:
        if conv_weight.shape != (dim, 1, resolved_kernel_size):
            raise ValueError(
                f"Expected 3D conv_weight shape {(dim, 1, resolved_kernel_size)}, "
                f"got {conv_weight.shape}."
            )
        w_kd = jnp.swapaxes(conv_weight[:, 0, :], 0, 1)
    elif conv_weight.ndim == 2:
        if conv_weight.shape != (resolved_kernel_size, dim):
            raise ValueError(
                f"Expected 2D conv_weight shape {(resolved_kernel_size, dim)}, "
                f"got {conv_weight.shape}."
            )
        w_kd = conv_weight
    else:
        raise ValueError(f"Expected 2D or 3D conv_weight, got {conv_weight.shape}.")

    if conv_bias is not None and conv_bias.shape != (dim,):
        raise ValueError(f"Expected conv_bias shape {(dim,)}, got {conv_bias.shape}.")
    if activation not in (None, "silu", "swish"):
        raise ValueError(
            f"Unsupported activation {activation!r}; expected None, 'silu', or 'swish'."
        )
    if state_indices.ndim != 1:
        raise ValueError(
            f"Expected 1D state_indices (num_seqs,), got {state_indices.shape}."
        )
    num_seqs = state_indices.shape[0]
    if query_start_loc.shape != (num_seqs + 1,):
        raise ValueError(
            f"Expected query_start_loc shape {(num_seqs + 1,)}, "
            f"got {query_start_loc.shape}."
        )

    if USE_TOKAMAX and _tokamax_ragged_causal_conv1d is not None:
        return _tokamax_ragged_causal_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias=conv_bias,
            query_start_loc=query_start_loc,
            state_indices=state_indices,
            distribution=distribution,
            has_initial_state=has_initial_state,
            seq_lens=seq_lens,
            start_seq=start_seq,
            activation=activation,
            kernel_size=resolved_kernel_size,
        )

    if num_tokens == 0 or num_seqs == 0:
        return jnp.zeros_like(x), conv_state

    if distribution is not None:
        num_valid_seqs = jnp.clip(distribution[2], 0, num_seqs)
        valid_loc_mask = jnp.arange(num_seqs + 1, dtype=jnp.int32) <= num_valid_seqs
        qsl = jnp.where(
            valid_loc_mask, query_start_loc, query_start_loc[num_valid_seqs]
        )
    else:
        num_valid_seqs = jnp.asarray(num_seqs, dtype=jnp.int32)
        qsl = query_start_loc

    query_lens = jnp.maximum(qsl[1:] - qsl[:-1], 0)
    if has_initial_state is not None:
        if has_initial_state.shape != (num_seqs,):
            raise ValueError(
                f"Expected has_initial_state shape {(num_seqs,)}, "
                f"got {has_initial_state.shape}."
            )
        has_init = has_initial_state.astype(bool)
    elif seq_lens is not None:
        if seq_lens.shape != (num_seqs,):
            raise ValueError(
                f"Expected seq_lens shape {(num_seqs,)}, got {seq_lens.shape}."
            )
        has_init = seq_lens > query_lens
    else:
        has_init = jnp.zeros((num_seqs,), dtype=bool)

    total_tokens = jnp.minimum(qsl[-1], num_tokens)
    if start_seq is None:
        first_seq = jnp.zeros((), jnp.int32)
    else:
        first_seq = jnp.minimum(jnp.asarray(start_seq, jnp.int32), num_valid_seqs)
    first_token = jnp.minimum(qsl[first_seq], total_tokens)

    return jax.lax.cond(
        first_token < total_tokens,
        lambda: _short_conv_tokens(
            x,
            conv_state,
            w_kd,
            conv_bias,
            qsl,
            state_indices,
            has_init,
            num_valid_seqs,
            first_seq,
            first_token,
            total_tokens,
            activation=activation,
        ),
        lambda: (jnp.zeros_like(x), conv_state),
    )
