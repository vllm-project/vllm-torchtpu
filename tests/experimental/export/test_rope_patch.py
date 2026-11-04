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
"""Unit tests for pure in-graph 1-D/2-D/3-D RoPE and M-RoPE export patching."""

from __future__ import annotations

import copy
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_torchtpu.experimental.export import rope_patch

_HEAD_SIZE = 64
_NUM_HEADS = 4
_NUM_KV_HEADS = 2


class _SyntheticRotaryEmbedding(nn.Module):
    """Lightweight standalone RotaryEmbedding for CPU unit tests."""

    def __init__(
        self,
        head_size: int = _HEAD_SIZE,
        rotary_dim: int = _HEAD_SIZE,
        max_position: int = 512,
        base: float = 10000.0,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.float32,
        mrope_section: list[int] | None = None,
        mrope_interleaved: bool = False,
        scaling_factor: float | None = None,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position
        self.base = base
        self.is_neox_style = is_neox_style
        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        self.scaling_factor = scaling_factor
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        return 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float32)
                / self.rotary_dim
            )
        )

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if offsets is not None:
            positions = positions + offsets
        if positions.ndim >= 2 and self.mrope_section:
            pos_2d = positions.reshape(3, -1)
            num_tokens = pos_2d.shape[-1]
            cos_sin = self.cos_sin_cache[pos_2d]  # [3, N, D]
            cos, sin = cos_sin.chunk(2, dim=-1)  # each [3, N, D/2]
            if self.mrope_interleaved:
                half = self.rotary_dim // 2
                cos_out = cos[0].clone()
                sin_out = sin[0].clone()
                for ch in range(half):
                    if ch % 3 == 1 and ch < self.mrope_section[1] * 3:
                        cos_out[:, ch] = cos[1, :, ch]
                        sin_out[:, ch] = sin[1, :, ch]
                    elif (
                        ch % 3 == 2
                        and len(self.mrope_section) > 2
                        and ch < self.mrope_section[2] * 3
                    ):
                        cos_out[:, ch] = cos[2, :, ch]
                        sin_out[:, ch] = sin[2, :, ch]
                cos_f, sin_f = cos_out, sin_out
            else:
                cos_f = torch.cat(
                    [
                        cos[i, :, sl]
                        for i, sl in enumerate(self._slice_ranges(self.mrope_section))
                    ],
                    dim=-1,
                )
                sin_f = torch.cat(
                    [
                        sin[i, :, sl]
                        for i, sl in enumerate(self._slice_ranges(self.mrope_section))
                    ],
                    dim=-1,
                )
        else:
            pos_flat = positions.flatten()
            num_tokens = pos_flat.shape[0]
            cos_sin = self.cos_sin_cache[pos_flat]
            cos_f, sin_f = cos_sin.chunk(2, dim=-1)

        cos_f = cos_f.to(query.dtype)
        sin_f = sin_f.to(query.dtype)
        q_shape = query.shape
        q = query.view(num_tokens, -1, self.head_size)
        q_rot = rope_patch._apply_rotary_emb(
            q[..., : self.rotary_dim], cos_f, sin_f, self.is_neox_style
        )
        q_out = torch.cat((q_rot, q[..., self.rotary_dim :]), dim=-1).reshape(q_shape)
        k_out = None
        if key is not None:
            k_shape = key.shape
            k = key.view(num_tokens, -1, self.head_size)
            k_rot = rope_patch._apply_rotary_emb(
                k[..., : self.rotary_dim], cos_f, sin_f, self.is_neox_style
            )
            k_out = torch.cat((k_rot, k[..., self.rotary_dim :]), dim=-1).reshape(
                k_shape
            )
        return q_out, k_out

    @staticmethod
    def _slice_ranges(sections: list[int]) -> list[slice]:
        out: list[slice] = []
        offset = 0
        for count in sections:
            out.append(slice(offset, offset + count))
            offset += count
        return out

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets=offsets)


def _qk(num_tokens: int, dtype: torch.dtype = torch.float32):
    gen = torch.Generator().manual_seed(42)
    q = torch.randn(num_tokens, _NUM_HEADS * _HEAD_SIZE, generator=gen).to(dtype)
    k = torch.randn(num_tokens, _NUM_KV_HEADS * _HEAD_SIZE, generator=gen).to(dtype)
    return q, k


def test_1d_in_graph_rope_matches_cached_and_removes_buffer() -> None:
    rope = _SyntheticRotaryEmbedding(dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 7, 31, 128, 511], dtype=torch.int32)
    q, k = _qk(positions.numel(), dtype=torch.bfloat16)
    ref_q, ref_k = copy.deepcopy(rope)(positions, q.clone(), k.clone())

    wrapper = nn.Module()
    wrapper.rope = rope
    patched = rope_patch.patch_in_graph_rotary_embedding(wrapper)
    assert patched == 1
    assert "cos_sin_cache" not in rope._buffers
    assert not hasattr(rope, "cos_sin_cache")

    out_q, out_k = rope(positions, q.clone(), k.clone())
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)


@pytest.mark.parametrize("interleaved", [False, True])
def test_2d_and_3d_mrope_in_graph_matches_cached(interleaved: bool) -> None:
    sections = [8, 12, 12]  # sum == 32 == _HEAD_SIZE // 2
    rope = _SyntheticRotaryEmbedding(
        mrope_section=sections,
        mrope_interleaved=interleaved,
        dtype=torch.float32,
    )
    positions_2d = torch.tensor(
        [[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]], dtype=torch.int32
    )
    q, k = _qk(4, dtype=torch.float32)
    ref_q, ref_k = copy.deepcopy(rope)(positions_2d, q.clone(), k.clone())

    wrapper = nn.Module()
    wrapper.rope = rope
    patched = rope_patch.patch_in_graph_rotary_embedding(wrapper)
    assert patched == 1
    assert "cos_sin_cache" not in rope._buffers
    assert rope._in_graph_rope_dim_modalities is not None
    assert len(rope._in_graph_rope_dim_modalities) == _HEAD_SIZE // 2

    # Verify 2-D [3, N] positions
    out_q, out_k = rope(positions_2d, q.clone(), k.clone())
    torch.testing.assert_close(out_q, ref_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-6, atol=1e-6)

    # Verify 3-D [3, B, S] positions
    positions_3d = positions_2d.view(3, 2, 2)
    out_q_3d, out_k_3d = rope(positions_3d, q.clone(), k.clone())
    torch.testing.assert_close(out_q_3d, ref_q, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(out_k_3d, ref_k, rtol=1e-6, atol=1e-6)

    # Verify 1-D fallback on M-RoPE module
    pos_1d = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    out_q_1d, out_k_1d = rope(pos_1d, q.clone(), k.clone())
    assert out_q_1d.shape == q.shape
    assert out_k_1d is not None and out_k_1d.shape == k.shape


def test_scaled_yarn_rope_keeps_its_forward_and_computes_cache_rows() -> None:
    rope = _SyntheticRotaryEmbedding(scaling_factor=4.0)
    positions = torch.tensor([0, 3, 511], dtype=torch.int32)
    q, k = _qk(positions.numel())
    ref_q, ref_k = copy.deepcopy(rope)(positions, q.clone(), k.clone())
    wrapper = nn.Module()
    wrapper.rope = rope
    assert rope_patch.patch_in_graph_rotary_embedding(wrapper) == 1
    assert not hasattr(rope, "_in_graph_rope_inv_freq")
    assert "cos_sin_cache" not in dict(wrapper.named_buffers(recurse=True))
    out_q, out_k = rope(positions, q.clone(), k.clone())
    torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-5)


def test_real_vllm_rotary_and_mrotary_modules_when_available() -> None:
    pytest.importorskip("vllm")
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding as rope_lib

    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["none"]
    with set_current_vllm_config(vllm_config):
        rope_1d = rope_lib.get_rope(
            _HEAD_SIZE,
            max_position=512,
            is_neox_style=True,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6},
            dtype=torch.float32,
        )
        mrope_2d = rope_lib.get_rope(
            _HEAD_SIZE,
            max_position=512,
            is_neox_style=True,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e6,
                "mrope_section": [8, 12, 12],
            },
            dtype=torch.float32,
        )

    pos_1d = torch.tensor([0, 2, 9, 31], dtype=torch.int64)
    pos_2d = torch.tensor([[0, 1, 2, 3], [0, 4, 4, 5], [0, 7, 8, 9]], dtype=torch.int64)
    q, k = _qk(4, dtype=torch.float32)

    ref_1d_q, ref_1d_k = copy.deepcopy(rope_1d)(pos_1d, q.clone(), k.clone())
    ref_2d_q, ref_2d_k = copy.deepcopy(mrope_2d)(pos_2d, q.clone(), k.clone())

    container = nn.ModuleDict({"r1": rope_1d, "r2": mrope_2d})
    assert rope_patch.patch_in_graph_rotary_embedding(container) == 2
    assert "cos_sin_cache" not in rope_1d._buffers
    assert "cos_sin_cache" not in mrope_2d._buffers

    out_1d_q, out_1d_k = rope_1d(pos_1d, q.clone(), k.clone())
    out_2d_q, out_2d_k = mrope_2d(pos_2d, q.clone(), k.clone())
    torch.testing.assert_close(out_1d_q, ref_1d_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_1d_k, ref_1d_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_2d_q, ref_2d_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_2d_k, ref_2d_k, rtol=1e-5, atol=1e-5)


def _vllm_config(max_model_len: int) -> MagicMock:
    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["none"]
    vllm_config.model_config.max_model_len = max_model_len
    return vllm_config


def _assert_cache_buffers_gone(module: nn.Module) -> None:
    names = [name for name, _ in module.named_buffers()]
    assert not [n for n in names if "cos_sin_cache" in n], names


def test_deepseek_yarn_rope_cache_is_computed_in_graph() -> None:
    pytest.importorskip("vllm")
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
    )

    with set_current_vllm_config(_vllm_config(2048)):
        rope = DeepseekScalingRotaryEmbedding(
            _HEAD_SIZE,
            _HEAD_SIZE,
            256,
            10000.0,
            False,
            40.0,
            torch.bfloat16,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0.707,
        )
    rows = rope.cos_sin_cache.shape[0]
    positions = torch.tensor([0, 1, 7, 255, 4096, rows - 1])
    q, k = _qk(positions.numel(), dtype=torch.bfloat16)
    q, k = q.view(-1, _NUM_HEADS, _HEAD_SIZE), k.view(-1, _NUM_KV_HEADS, _HEAD_SIZE)
    ref_q, ref_k = copy.deepcopy(rope).forward_native(positions, q, k)

    assert rope_patch.patch_in_graph_rotary_embedding(rope) == 1
    _assert_cache_buffers_gone(rope)
    out_q, out_k = rope.forward_native(positions, q, k)
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=3e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=3e-2)


@pytest.mark.parametrize("max_model_len", [512, 4096])
def test_phi3_long_rope_cache_is_computed_in_graph(max_model_len: int) -> None:
    pytest.importorskip("vllm")
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.phi3_long_rope_scaled_rope import (
        Phi3LongRoPEScaledRotaryEmbedding,
    )

    rotary_dim = 48
    with set_current_vllm_config(_vllm_config(max_model_len)):
        rope = Phi3LongRoPEScaledRotaryEmbedding(
            _HEAD_SIZE,
            rotary_dim,
            8192,
            1024,
            10000.0,
            True,
            torch.bfloat16,
            short_factor=[1.0 + 0.1 * i for i in range(rotary_dim // 2)],
            long_factor=[2.0 + i for i in range(rotary_dim // 2)],
        )
    # The long-factor table follows the short one, so positions select either.
    assert rope.use_long_rope == (max_model_len > 1024)
    positions = torch.tensor([0, 1, 9, 1023, 1024, 7000])
    q, k = _qk(positions.numel(), dtype=torch.bfloat16)
    ref_q, ref_k = copy.deepcopy(rope)(positions, q, k)

    assert rope_patch.patch_in_graph_rotary_embedding(rope) == 1
    _assert_cache_buffers_gone(rope)
    out_q, out_k = rope(positions, q, k)
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=3e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=3e-2)


def test_buffer_that_is_not_a_cos_sin_table_is_kept() -> None:
    module = nn.Module()
    noise = torch.rand(64, 16, generator=torch.Generator().manual_seed(0))
    module.register_buffer("cos_sin_cache", noise, persistent=False)
    assert rope_patch.patch_in_graph_rotary_embedding(module) == 0
    assert module.cos_sin_cache is noise
