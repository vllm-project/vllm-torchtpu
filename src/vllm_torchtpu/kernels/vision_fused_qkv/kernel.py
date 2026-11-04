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

"""Vision Encoder Fused QKV Projection and 2D mRoPE Kernel.

Adapted from Alyssa Nie's fused_qkv kernel (go/fused_qkv_proj, cl/989795783).

Key differences from GQA / Batched RPA:
1. Strips out batched RPA layout reshuffling (no KV cache, no paged memory,
   no slot mapping, no block tables).
2. Applies 2D mRoPE with in-register lane-swap FMA across spatial patch grids (h, w).
3. Emits contiguous Q, K, V in [batch_size, num_heads, seq_len, head_dim]
   directly matching the FlashAttention (TpuMMEncoderAttention) input contract.
"""

from __future__ import annotations

from typing import Sequence
from vllm_torchtpu.kernels.vision_fused_qkv.configs import VisionAttentionConfig
from vllm_torchtpu.kernels.vision_fused_qkv.mrope import (
    compute_inv_freq,
    compute_2d_mrope_angles,
    LaneRope2D,
)


def vision_fused_qkv_project(
    hidden_states: list[list[list[float]]],  # [batch_size, seq_len, hidden_size]
    w_qkv: list[list[float]],  # [hidden_size, 3 * num_heads * head_dim]
    config: VisionAttentionConfig,
    patch_coords: list[tuple[int, int]],  # (pos_h, pos_w) per token in seq_len
    bias_qkv: list[float] | None = None,
) -> tuple[list[list[list[list[float]]]], list[list[list[list[float]]]], list[list[list[list[float]]]]]:
    """Execute fused QKV projection with 2D mRoPE without RPA layout shuffling.

    Args:
        hidden_states: Batch of token features [batch_size, seq_len, hidden_size].
        w_qkv: Fused projection weight matrix.
        config: Vision attention configuration.
        patch_coords: Spatial (h, w) coordinates for each sequence position.
        bias_qkv: Optional projection bias.

    Returns:
        tuple of (q, k, v), each formatted as [batch_size, num_heads, seq_len, head_dim]
        directly compatible with flash_attention(q, k, v).
    """
    batch_size = len(hidden_states)
    seq_len = len(hidden_states[0]) if batch_size > 0 else 0
    num_heads = config.num_heads
    head_dim = config.head_dim
    hidden_size = config.hidden_size

    assert len(patch_coords) == seq_len, f"patch_coords len ({len(patch_coords)}) != seq_len ({seq_len})"

    # 1. Precompute 2D mRoPE operators per patch token if mRoPE is enabled
    mrope_ops: list[LaneRope2D | None] = []
    if config.mrope is not None:
        inv_freq = compute_inv_freq(config.mrope.rotary_dim, config.mrope.theta)
        for h, w in patch_coords:
            cos_vals, sin_vals = compute_2d_mrope_angles(
                h, w, inv_freq, config.mrope.section_h, config.mrope.section_w
            )
            mrope_ops.append(LaneRope2D(config.mrope, head_dim, cos_vals, sin_vals))
    else:
        mrope_ops = [None] * seq_len

    # 2. Allocate output tensors in Flash Attention layout [batch_size, num_heads, seq_len, head_dim]
    q_out: list[list[list[list[float]]]] = [
        [[[0.0] * head_dim for _ in range(seq_len)] for _ in range(num_heads)]
        for _ in range(batch_size)
    ]
    k_out: list[list[list[list[float]]]] = [
        [[[0.0] * head_dim for _ in range(seq_len)] for _ in range(num_heads)]
        for _ in range(batch_size)
    ]
    v_out: list[list[list[list[float]]]] = [
        [[[0.0] * head_dim for _ in range(seq_len)] for _ in range(num_heads)]
        for _ in range(batch_size)
    ]

    total_q_dim = num_heads * head_dim
    total_k_dim = num_heads * head_dim
    total_v_dim = num_heads * head_dim

    # 3. Fused Matmul + 2D Lane-Swap mRoPE execution
    for b in range(batch_size):
        for s in range(seq_len):
            x = hidden_states[b][s]
            # Matmul: projected = x @ w_qkv (+ bias)
            proj = [0.0] * (3 * total_q_dim)
            for j in range(3 * total_q_dim):
                dot = sum(x[k] * w_qkv[k][j] for k in range(hidden_size))
                if bias_qkv is not None:
                    dot += bias_qkv[j]
                proj[j] = dot

            # Slices: Q, K, V
            q_raw = proj[:total_q_dim]
            k_raw = proj[total_q_dim : total_q_dim + total_k_dim]
            v_raw = proj[total_q_dim + total_k_dim :]

            rope_op = mrope_ops[s]

            # Distribute into heads and apply 2D mRoPE via in-register lane swap
            for h in range(num_heads):
                start = h * head_dim
                end = start + head_dim

                q_head = q_raw[start:end]
                k_head = k_raw[start:end]
                v_head = v_raw[start:end]

                if rope_op is not None:
                    q_head = rope_op.rotate_head_list(q_head)
                    k_head = rope_op.rotate_head_list(k_head)

                q_out[b][h][s] = q_head
                k_out[b][h][s] = k_head
                v_out[b][h][s] = v_head

    return q_out, k_out, v_out
