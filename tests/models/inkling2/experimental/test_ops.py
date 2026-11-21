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
"""CPU parity tests for TPU-portable Inkling 2 Loop building blocks."""

from __future__ import annotations

import pytest
import torch

from vllm_torchtpu.models.inkling2.experimental.ops import (
    attn_res,
    compute_log_scaling_tau,
    inkling2_moe_forward,
    message_controller_blend,
    relative_bias_attention_dense,
    relative_bias_attention_split_tpu,
    short_conv_dynamic,
    short_conv_static,
)


@pytest.mark.parametrize(
    ("seq_len", "rel_extent", "window"),
    [
        (32, 16, None),
        (64, 16, None),
        (64, 16, 16),
        (12, 16, None),
    ],
)
def test_split_tpu_attention_matches_dense_reference(
    seq_len: int, rel_extent: int, window: int | None
) -> None:
    torch.manual_seed(42)
    num_heads, num_kv_heads, head_dim, d_rel = 6, 2, 16, 8
    q = torch.randn(seq_len, num_heads, head_dim)
    k = torch.randn(seq_len, num_kv_heads, head_dim)
    v = torch.randn(seq_len, num_kv_heads, head_dim)
    r = torch.randn(seq_len, num_heads, d_rel)
    rel_proj = torch.randn(d_rel, rel_extent) * 0.2
    gate = torch.randn(seq_len, num_heads, head_dim)
    sinks = torch.randn(seq_len, num_heads)
    pos = torch.arange(seq_len, dtype=torch.long)

    ref = relative_bias_attention_dense(
        q, k, v, r, rel_proj, gate, sinks, pos, pos, sliding_window=window
    )
    split = relative_bias_attention_split_tpu(
        q, k, v, r, rel_proj, gate, sinks, pos, pos, sliding_window=window
    )
    torch.testing.assert_close(split, ref, atol=1e-5, rtol=1e-5)


def test_short_conv_prefill_matches_incremental_decode() -> None:
    torch.manual_seed(7)
    num_tokens, channels, kernel_size, num_kv_heads = 13, 32, 4, 2
    x = torch.randn(num_tokens, channels)

    static_w = torch.randn(channels, kernel_size)
    y_full, _ = short_conv_static(x, static_w)
    hist = None
    steps = []
    for idx in range(num_tokens):
        step_out, hist = short_conv_static(x[idx : idx + 1], static_w, hist)
        steps.append(step_out)
    torch.testing.assert_close(torch.cat(steps, dim=0), y_full, atol=1e-6, rtol=1e-6)

    dyn_static_w = torch.randn(kernel_size, channels)
    dyn_taps = torch.randn(num_tokens, num_kv_heads * kernel_size)
    yd_full, _ = short_conv_dynamic(x, dyn_static_w, dyn_taps, num_kv_heads)
    hist = None
    dyn_steps = []
    for idx in range(num_tokens):
        step_out, hist = short_conv_dynamic(
            x[idx : idx + 1],
            dyn_static_w,
            dyn_taps[idx : idx + 1],
            num_kv_heads,
            hist,
        )
        dyn_steps.append(step_out)
    torch.testing.assert_close(
        torch.cat(dyn_steps, dim=0), yd_full, atol=1e-6, rtol=1e-6
    )


def test_attn_res_controller_and_situ_moe_shapes_and_bounds() -> None:
    torch.manual_seed(11)
    num_tokens, hidden = 5, 32
    stack = torch.randn(2, num_tokens, hidden, dtype=torch.bfloat16)
    partial = torch.randn(num_tokens, hidden, dtype=torch.bfloat16)
    queries = torch.randn(3, hidden) * (hidden**-0.5)
    norm_w = torch.ones(3, hidden)

    mixed = attn_res(stack, partial, None, queries, None)
    assert mixed.shape == (3, num_tokens, hidden)
    assert mixed.dtype == torch.float32

    blended = message_controller_blend(
        mixed,
        mixed * 0.5,
        torch.randn(3, 8, 2 * hidden, dtype=torch.bfloat16) * 0.05,
        torch.randn(3, hidden, 8, dtype=torch.bfloat16) * 0.05,
        torch.randn(3, hidden, 8, dtype=torch.bfloat16) * 0.05,
        norm_w,
    )
    assert blended.shape == (3, num_tokens, hidden)
    assert blended.dtype == torch.bfloat16

    tau = compute_log_scaling_tau(torch.tensor([0, 8191, 16383]))
    assert tau[0].item() == 1.0
    assert tau[1].item() == 1.0
    assert tau[2].item() > 1.0

    out = inkling2_moe_forward(
        blended[0],
        torch.randn(8, hidden, dtype=torch.bfloat16) * 0.1,
        torch.zeros(8, dtype=torch.float32),
        torch.randn(8, 32, hidden, dtype=torch.bfloat16) * 0.1,
        torch.randn(8, hidden, 16, dtype=torch.bfloat16) * 0.1,
        topk=2,
    )
    assert out.shape == (num_tokens, hidden)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
