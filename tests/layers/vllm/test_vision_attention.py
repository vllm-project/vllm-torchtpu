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

import torch

from vllm_torchtpu.layers.vllm.vision_attention import TpuMMEncoderAttention


def make_attention(num_heads=2, head_size=4, scale=1.0, num_kv_heads=None):
    attn = TpuMMEncoderAttention.__new__(TpuMMEncoderAttention)
    attn.num_heads = num_heads
    attn.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    attn.head_size = head_size
    attn.scale = scale
    return attn


def test_vision_flash_attention_pads_and_builds_segments(monkeypatch):
    calls = {}

    def fake_vision_op(q, k, v, q_seg, kv_seg):
        calls["q"] = q
        calls["k"] = k
        calls["v"] = v
        calls["q_seg"] = q_seg
        calls["kv_seg"] = kv_seg
        return q

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", fake_vision_op)

    attn = make_attention(scale=0.5)
    q = torch.arange(1 * 5 * 2 * 4, dtype=torch.float32).reshape(1, 5, 2, 4)
    k = q + 100
    v = q + 200
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    out = attn._flash_attention(q, k, v, cu_seqlens)

    assert out.shape == q.shape
    assert calls["q"].shape == (1, 2, 128, 4)
    assert calls["k"].shape == (1, 2, 128, 4)
    assert calls["v"].shape == (1, 2, 128, 4)
    assert calls["q_seg"].shape == (1, 128)
    assert calls["kv_seg"].shape == (1, 128)
    expected = [1, 1, 2, 2, 2] + [3] * (128 - 5)
    assert calls["q_seg"][0].tolist() == expected
    assert calls["kv_seg"][0].tolist() == expected


def test_vision_attention_falls_back_for_cross_attention(monkeypatch):

    def fail_vision_op(*args, **kwargs):
        raise AssertionError("flash path should not handle q_len != kv_len")

    monkeypatch.setattr(TpuMMEncoderAttention, "_vision_op", fail_vision_op)

    attn = make_attention()
    query = torch.zeros(1, 1, 8)
    key = torch.zeros(1, 4, 8)
    value = torch.ones(1, 4, 8)

    called = {}

    def fake_native(query, key, value, cu_seqlens, max_seqlen,
                    sequence_lengths):
        called["args"] = (query, key, value, cu_seqlens, max_seqlen,
                          sequence_lengths)
        return torch.full_like(query, 7.0)

    monkeypatch.setattr(attn, "forward_native", fake_native)

    out = attn.forward_oot(query, key, value)

    assert torch.equal(out, torch.full_like(query, 7.0))
    assert called["args"][0] is query
    assert called["args"][1] is key
    assert called["args"][2] is value
