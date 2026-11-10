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
# Copied from src/vllm_torchtpu/kernels/gdn/v3/compute_conv1d.py at vllm-torchtpu main
# c350823af74ba9cdf728daad64df7dd5bcf84e8e. This copy carries the Qwen3.5 GDN v3
# changes and serves the Qwen GDN architectures only (gdn_kernel_wrapper in
# layers/core/gdn_attention.py picks it); every other model keeps kernels/gdn/v3.
# Upstream v3 changes are not picked up automatically.

import jax
import jax.numpy as jnp

from vllm_torchtpu.models.qwen3_5.kernels.gdn import config


def causal_conv1d_out(
    lhs: jax.Array,  # [seq, prev_kernel_size + chunk, q, dim_size]
    conv_weight: jax.Array,  # [kernel_size, 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    cfg: config.GDNConfig,
) -> jax.Array:
    """The conv output rows of `causal_conv1d` without the conv-state select."""
    assert lhs.ndim == 4
    out_list = []
    w_taps = [conv_weight[k : k + 1] for k in range(cfg.kernel_size)]
    bias = conv_bias.reshape(1, 1, -1) if conv_bias is not None else None
    for c_idx in range(cfg.chunk_size):
        out = lhs[:, c_idx] * w_taps[0]
        for k in range(1, cfg.kernel_size):
            out = out + lhs[:, c_idx + k] * w_taps[k]
        if bias is not None:
            out = out + bias
        out_list.append(out)
    return jnp.stack(out_list, axis=1)


def conv_state_rows(
    lhs: jax.Array, real_size: jax.Array, cfg: config.GDNConfig
) -> jax.Array:
    """Conv state of one sequence after `real_size` tokens of its chunk: rows
    [real_size, real_size + prev_kernel_size) of `lhs` ([prev_kernel_size +
    chunk, q, dim_size]), selected row by row."""
    state = lhs[cfg.chunk_size : cfg.chunk_size + cfg.prev_kernel_size]
    for c_idx in range(1, cfg.chunk_size):
        state = jnp.where(
            c_idx == real_size, lhs[c_idx : c_idx + cfg.prev_kernel_size], state
        )
    return state


def causal_conv1d(
    real_sizes: jax.Array,  # [seq]
    lhs: jax.Array,  # [seq, chunk, q, dim_size]
    conv_weight: jax.Array,  # [prev_kernel_size, 1, dim_size]
    conv_bias: jax.Array | None,  # [dim_size]
    cfg: config.GDNConfig,
) -> tuple[jax.Array, jax.Array]:
    """Perform causal Conv1D. Returns Conv1D output and convolution states.

    States are [seq, window_size, prev_kernel_size, q, dim_size]: checkpoint
    w holds the last prev_kernel_size inputs ending at token
    `chunk_size - window_size + w`, clamped to the last real token, i.e. the
    state a sequence resuming right after that token must start from.
    """

    assert lhs.ndim == 4

    out_list = []

    for c_idx in range(cfg.chunk_size):
        out = jnp.zeros((cfg.seq_tile_size, 1, cfg.dim_size), jnp.float32)

        end_idx = c_idx + cfg.prev_kernel_size
        start_idx = 1 + end_idx - cfg.kernel_size
        for k in range(cfg.kernel_size):
            lhs_curr = lhs[:, start_idx + k]
            out += lhs_curr * conv_weight[k : k + 1]

        if conv_bias is not None:
            out += conv_bias.reshape(1, 1, -1)

        out_list.append(out)

    # Last prev_kernel_size elements needs to be returned as conv_state. However,
    # real_sizes may be smaller than chunk_size. Therefore, slicing last
    # prev_kernel_size elements does not gurantee numeric correctness. Instead,
    # kernel iterate each rows and perform masking to fetch correct values.
    # NOTE: lhs[:, : prev_kernel_size] can be skipped since they were loaded from
    # previous conv states.
    real_sizes = real_sizes.reshape(-1, 1, 1, 1)
    state_list = []
    # NOTE: Even though for loop is invoked multiple times, since they are
    # static loops, compiler will perform loop fusion.
    for w_idx in range(cfg.window_size):
        last_row = 1 + cfg.chunk_size - cfg.window_size + w_idx
        # Checkpoint of sequences whose last real token is at or past this
        # window position; shorter ones are picked by the masking loop below.
        new_conv_state = lhs[:, last_row : last_row + cfg.prev_kernel_size]
        for c_idx in range(1, last_row):
            new_conv_state = jnp.where(
                c_idx == real_sizes,
                lhs[:, c_idx : c_idx + cfg.prev_kernel_size],
                new_conv_state,
            )
        state_list.append(new_conv_state)

    return jnp.stack(out_list, axis=1), jnp.stack(state_list, axis=1)
