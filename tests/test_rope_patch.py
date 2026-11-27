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
"""Unit tests for pure in-graph 1-D/2-D/3-D RoPE and M-RoPE patching."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

import vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op  # noqa: F401
import vllm_torchtpu.layers.adapter.rotary_embedding as tpu_rope_mod
from tests.test_fresh_buffer import _RecordOps
from vllm_torchtpu import rope_in_graph, rope_in_graph_flag, rope_patch

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

        def rotate(x: torch.Tensor) -> torch.Tensor:
            return rope_in_graph.rotate_fp32(
                x,
                cos_f,
                sin_f,
                num_tokens=num_tokens,
                head_size=self.head_size,
                rotary_dim=self.rotary_dim,
                is_neox_style=self.is_neox_style,
            )

        return rotate(query), None if key is None else rotate(key)

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


class _SyntheticProportionalRotaryEmbedding(_SyntheticRotaryEmbedding):
    """Synthetic Gemma-4 proportional RoPE (`rotary_dim = head_size`, zero-padded)."""

    def __init__(
        self,
        head_size: int = _HEAD_SIZE,
        rope_angles: int = _HEAD_SIZE // 4,
        max_position: int = 512,
        base: float = 1000000.0,
        is_neox_style: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.rope_angles = rope_angles
        self.nope_angles = (head_size // 2) - rope_angles
        super().__init__(
            head_size=head_size,
            rotary_dim=head_size,
            max_position=max_position,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        exponents = (
            torch.arange(0, 2 * self.rope_angles, 2, dtype=torch.float32)
            / self.head_size
        )
        inv_freq = 1.0 / (base**exponents)
        if self.nope_angles > 0:
            inv_freq = torch.cat(
                (inv_freq, torch.zeros(self.nope_angles, dtype=torch.float32))
            )
        return inv_freq


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
    patched = rope_patch.patch_in_graph_rotary_embedding(
        wrapper, allow_table_recovery=True
    )
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
    patched = rope_patch.patch_in_graph_rotary_embedding(
        wrapper, allow_table_recovery=True
    )
    assert patched == 1
    assert "cos_sin_cache" not in rope._buffers
    assert rope._in_graph_spec.dim_modalities is not None
    assert len(rope._in_graph_spec.dim_modalities) == _HEAD_SIZE // 2

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

    # Verify 2-D positions with leading dim != 3 raise ValueError (not 1-D flatten)
    pos_bad = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.int32)
    with pytest.raises(ValueError, match="leading dimension 3"):
        rope(pos_bad, q.clone(), k.clone())


def test_scaled_yarn_rope_keeps_its_forward_and_computes_cache_rows() -> None:
    rope = _SyntheticRotaryEmbedding(scaling_factor=4.0)
    positions = torch.tensor([0, 3, 511], dtype=torch.int32)
    q, k = _qk(positions.numel())
    ref_q, ref_k = copy.deepcopy(rope)(positions, q.clone(), k.clone())
    wrapper = nn.Module()
    wrapper.rope = rope
    assert (
        rope_patch.patch_in_graph_rotary_embedding(wrapper, allow_table_recovery=False)
        == 0
    )
    assert (
        rope_patch.patch_in_graph_rotary_embedding(wrapper, allow_table_recovery=True)
        == 1
    )
    assert not hasattr(rope, "_in_graph_spec")
    assert "cos_sin_cache" not in dict(wrapper.named_buffers(recurse=True))
    out_q, out_k = rope(positions, q.clone(), k.clone())
    torch.testing.assert_close(out_q, ref_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_k, ref_k, rtol=1e-5, atol=1e-5)


def test_real_vllm_rotary_and_mrotary_modules_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding as rope_lib

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["none"]
    vllm_config.additional_config = {rope_in_graph_flag.KEY: False}
    with set_current_vllm_config(vllm_config):
        # Build CPU Tensor-backed modules (`tpu_rope_in_graph` off) to verify both
        # Tier-2 post-load patching and Tier-1 init-time construction against
        # the real upstream cos_sin_cache Tensor.
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
        assert "cos_sin_cache" in rope_1d._buffers
        assert "cos_sin_cache" in mrope_2d._buffers

        vllm_config.additional_config[rope_in_graph_flag.KEY] = True
        rope_lib._ROPE_DICT.clear()
        tier1_1d = rope_lib.get_rope(
            _HEAD_SIZE,
            max_position=512,
            is_neox_style=True,
            rope_parameters={"rope_type": "default", "rope_theta": 1e6},
            dtype=torch.float32,
        )
        tier1_2d = rope_lib.get_rope(
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

    # Tier 1 already has 0 buffers before Tier 2 runs, and Tier 2 is a no-op on it.
    assert "cos_sin_cache" not in tier1_1d._buffers
    assert "cos_sin_cache" not in tier1_2d._buffers
    tier1_container = nn.ModuleDict({"r1": tier1_1d, "r2": tier1_2d})
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier1_container, allow_table_recovery=False
        )
        == 0
    )
    t1_q1, t1_k1 = tier1_1d(pos_1d, q.clone(), k.clone())
    t1_q2, t1_k2 = tier1_2d(pos_2d, q.clone(), k.clone())
    torch.testing.assert_close(t1_q1, ref_1d_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(t1_k1, ref_1d_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(t1_q2, ref_2d_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(t1_k2, ref_2d_k, rtol=1e-5, atol=1e-5)

    # Tier 2 converts Tensor-backed modules (e.g., CPU-traced export models).
    container = nn.ModuleDict({"r1": rope_1d, "r2": mrope_2d})
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            container, allow_table_recovery=False
        )
        == 2
    )
    assert "cos_sin_cache" not in rope_1d._buffers
    assert "cos_sin_cache" not in mrope_2d._buffers

    out_1d_q, out_1d_k = rope_1d(pos_1d, q.clone(), k.clone())
    out_2d_q, out_2d_k = mrope_2d(pos_2d, q.clone(), k.clone())
    torch.testing.assert_close(out_1d_q, ref_1d_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_1d_k, ref_1d_k, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_2d_q, ref_2d_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out_2d_k, ref_2d_k, rtol=1e-5, atol=1e-5)


def _vllm_config(max_model_len: int, *, in_graph: bool = False) -> MagicMock:
    vllm_config = MagicMock()
    vllm_config.compilation_config.custom_ops = ["none"]
    vllm_config.model_config.max_model_len = max_model_len
    vllm_config.additional_config = {rope_in_graph_flag.KEY: in_graph}
    return vllm_config


def _assert_cache_buffers_gone(module: nn.Module) -> None:
    names = [name for name, _ in module.named_buffers()]
    assert not [n for n in names if "cos_sin_cache" in n], names


def test_deepseek_yarn_rope_cache_is_computed_in_graph() -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekScalingRotaryEmbedding,
    )

    cfg = _vllm_config(2048, in_graph=False)
    with set_current_vllm_config(cfg):
        ref_rope = DeepseekScalingRotaryEmbedding(
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
        assert isinstance(ref_rope.cos_sin_cache, torch.Tensor)

        cfg.additional_config[rope_in_graph_flag.KEY] = True
        tier1_rope = DeepseekScalingRotaryEmbedding(
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
    rows = ref_rope.cos_sin_cache.shape[0]
    positions = torch.tensor([0, 1, 7, 255, 4096, rows - 1])
    q, k = _qk(positions.numel(), dtype=torch.bfloat16)
    q, k = q.view(-1, _NUM_HEADS, _HEAD_SIZE), k.view(-1, _NUM_KV_HEADS, _HEAD_SIZE)
    ref_q, ref_k = copy.deepcopy(ref_rope).forward_native(positions, q, k)

    # Tier 1 (VllmDeepseekScalingRotaryEmbedding) allocates 0 cache buffers at init.
    _assert_cache_buffers_gone(tier1_rope)
    assert isinstance(tier1_rope.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier1_rope, allow_table_recovery=False
        )
        == 0
    )
    t1_q, t1_k = tier1_rope.forward_native(positions, q, k)
    torch.testing.assert_close(t1_q, ref_q, rtol=0, atol=3e-2)
    torch.testing.assert_close(t1_k, ref_k, rtol=0, atol=3e-2)

    # Tier 2 also patches a Tensor-backed DeepseekScalingRotaryEmbedding instance.
    assert (
        rope_patch.patch_in_graph_rotary_embedding(ref_rope, allow_table_recovery=False)
        == 1
    )
    _assert_cache_buffers_gone(ref_rope)
    out_q, out_k = ref_rope.forward_native(positions, q, k)
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=3e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=3e-2)


@pytest.mark.parametrize("max_model_len", [512, 4096])
def test_phi3_long_rope_cache_is_computed_in_graph(max_model_len: int) -> None:
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

    assert (
        rope_patch.patch_in_graph_rotary_embedding(rope, allow_table_recovery=False)
        == 1
    )
    _assert_cache_buffers_gone(rope)
    out_q, out_k = rope(positions, q, k)
    torch.testing.assert_close(out_q, ref_q, rtol=0, atol=3e-2)
    torch.testing.assert_close(out_k, ref_k, rtol=0, atol=3e-2)


def test_buffer_that_is_not_a_cos_sin_table_is_kept() -> None:
    module = nn.Module()
    noise = torch.rand(64, 16, generator=torch.Generator().manual_seed(0))
    module.register_buffer("cos_sin_cache", noise, persistent=False)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(module, allow_table_recovery=False)
        == 0
    )
    assert module.cos_sin_cache is noise


class _PairedTableRope(nn.Module):
    """Muse Spark's rotary: `cos`/`sin` tables of `max_len x d / 2`, by position."""

    def __init__(self, head_dim: int, max_len: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(max_len).float(), inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[positions].unsqueeze(1)
        sin = self.sin[positions].unsqueeze(1)
        real, imag = x.float().chunk(2, dim=-1)
        return torch.cat((real * cos - imag * sin, real * sin + imag * cos), dim=-1)


@pytest.mark.parametrize("theta", [10000.0, 500000.0])
def test_paired_cos_sin_tables_are_computed_in_graph(theta: float) -> None:
    rope = _PairedTableRope(64, 4096, theta)
    x = torch.randn(5, 2, 64)
    positions = torch.tensor([0, 1, 777, 2048, 4095])
    expected = rope(x, positions)
    assert rope_patch.patch_cos_sin_table_pairs(rope) == 1
    assert "cos" not in rope._buffers and "sin" not in rope._buffers
    torch.testing.assert_close(rope(x, positions), expected, rtol=0, atol=1e-5)


def test_a_table_that_is_not_rope_stays_a_buffer() -> None:
    module = nn.Module()
    module.register_buffer("cos", torch.rand(16, 4))
    module.register_buffer("sin", torch.rand(16, 4))
    assert rope_patch.patch_cos_sin_table_pairs(module) == 0
    assert "cos" in module._buffers


@pytest.mark.parametrize(
    ("rope_params", "rotary_dim"),
    [
        (
            {
                "rope_type": "llama3",
                "rope_theta": 500000.0,
                "factor": 8.0,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": 8192,
            },
            _HEAD_SIZE,
        ),
        (
            {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.5,
            },
            _HEAD_SIZE,
        ),
        (
            {
                "rope_type": "ntk",
                "rope_theta": 10000.0,
                "factor": 4.0,
            },
            _HEAD_SIZE // 2,
        ),
    ],
)
def test_tpu_oot_custom_inv_freq_ropes_match_and_support_get_cos_sin(
    monkeypatch: pytest.MonkeyPatch,
    rope_params: dict,
    rotary_dim: int,
) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding as rope_lib

    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    rope_params_local = dict(rope_params)
    rope_params_local.setdefault(
        "partial_rotary_factor", float(rotary_dim) / float(_HEAD_SIZE)
    )
    cfg = _vllm_config(1024, in_graph=False)
    with set_current_vllm_config(cfg):
        ref_rope = rope_lib.get_rope(
            _HEAD_SIZE,
            max_position=1024,
            is_neox_style=True,
            rope_parameters=rope_params_local,
            dtype=torch.bfloat16,
        )
        assert isinstance(ref_rope.cos_sin_cache, torch.Tensor)

        cfg.additional_config[rope_in_graph_flag.KEY] = True
        rope_lib._ROPE_DICT.clear()
        tier1_rope = rope_lib.get_rope(
            _HEAD_SIZE,
            max_position=1024,
            is_neox_style=True,
            rope_parameters=rope_params_local,
            dtype=torch.bfloat16,
        )
    tier2_rope = copy.deepcopy(ref_rope)
    assert type(tier1_rope).__module__ == tpu_rope_mod.__name__

    positions = torch.tensor([0, 1, 13, 127, 512, 1023], dtype=torch.int64)
    q, k = _qk(positions.numel(), dtype=torch.bfloat16)
    ref_q, ref_k = ref_rope(positions, q.clone(), k.clone())
    # `_maybe_compact_neox_rope_cache` may narrow the table-path buffer, so take
    # the full-width upstream table as the `get_cos_sin` reference.
    ref_table = ref_rope._compute_cos_sin_cache().to(ref_rope.cos_sin_cache.dtype)
    ref_cos, ref_sin = ref_table[:32].chunk(2, dim=-1)

    # Tier 1 allocates 0 buffers at init and Tier 2 skips it idempotently.
    _assert_cache_buffers_gone(tier1_rope)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier1_rope, allow_table_recovery=False
        )
        == 0
    )

    # Tier 2 converts a Tensor-backed instance idempotently.
    assert rope_patch.is_in_graph_rope_eligible(tier2_rope)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier2_rope, allow_table_recovery=False
        )
        == 1
    )
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier2_rope, allow_table_recovery=False
        )
        == 0
    )
    _assert_cache_buffers_gone(tier2_rope)

    for candidate in (tier1_rope, tier2_rope):
        for method in (
            candidate,
            candidate.forward_native,
            candidate.forward_oot,
            candidate.forward_tpu,
        ):
            out_q, out_k = method(positions, q.clone(), k.clone())
            torch.testing.assert_close(out_q, ref_q, rtol=0, atol=1.6e-2)
            torch.testing.assert_close(out_k, ref_k, rtol=0, atol=1.6e-2)

        cos, sin = candidate.get_cos_sin(32)
        torch.testing.assert_close(cos, ref_cos, rtol=0, atol=1.6e-2)
        torch.testing.assert_close(sin, ref_sin, rtol=0, atol=1.6e-2)


@pytest.mark.parametrize(
    "variant",
    ["yarn", "linear_multi", "dynamic_ntk", "dynamic_ntk_alpha", "mrope_yarn"],
)
def test_analytical_cache_backed_rope_variants_and_dtype_matching(
    variant: str,
) -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers import rotary_embedding as rope_lib

    def _build() -> tuple[nn.Module, torch.Tensor]:
        if variant == "yarn":
            mod = rope_lib.YaRNScalingRotaryEmbedding(
                _HEAD_SIZE,
                _HEAD_SIZE,
                512,
                10000.0,
                True,
                4.0,
                torch.float32,
                attention_factor=1.2,
            )
            pos = torch.tensor([0, 3, 129, 1024, 2047], dtype=torch.int64)
        elif variant == "linear_multi":
            mod = rope_lib.LinearScalingRotaryEmbedding(
                _HEAD_SIZE,
                _HEAD_SIZE,
                256,
                10000.0,
                True,
                [1.0, 2.0, 4.0],
                torch.float32,
            )
            pos = torch.tensor([0, 7, 255, 256, 767, 768, 1791], dtype=torch.int64)
        elif variant == "dynamic_ntk":
            mod = rope_lib.DynamicNTKScalingRotaryEmbedding(
                _HEAD_SIZE,
                _HEAD_SIZE,
                2048,
                512,
                10000.0,
                True,
                4.0,
                torch.float32,
            )
            pos = torch.tensor([0, 5, 512, 2047], dtype=torch.int64)
        elif variant == "dynamic_ntk_alpha":
            mod = rope_lib.DynamicNTKAlphaRotaryEmbedding(
                _HEAD_SIZE,
                _HEAD_SIZE,
                2048,
                10000.0,
                True,
                4.0,
                torch.float32,
            )
            pos = torch.tensor([0, 5, 512, 2047], dtype=torch.int64)
        else:
            mod = rope_lib.MRotaryEmbedding(
                _HEAD_SIZE,
                _HEAD_SIZE,
                512,
                10000.0,
                True,
                torch.float32,
                mrope_section=[8, 12, 12],
                scaling_factor=4.0,
            )
            pos = torch.tensor(
                [[0, 1, 15, 200], [0, 2, 16, 201], [0, 3, 17, 202]],
                dtype=torch.int64,
            )
        return mod, pos

    cfg = _vllm_config(2048, in_graph=False)
    with set_current_vllm_config(cfg):
        ref_rope, positions = _build()
        assert isinstance(ref_rope.cos_sin_cache, torch.Tensor)

        cfg.additional_config[rope_in_graph_flag.KEY] = True
        tier1_rope, _ = _build()

    num_tokens = positions.shape[-1]
    q, k = _qk(num_tokens, dtype=torch.bfloat16)
    tier2_rope = copy.deepcopy(ref_rope)
    ref_q, ref_k = ref_rope(positions, q.clone(), k.clone())
    ref_slice = ref_rope.cos_sin_cache[:16].to(torch.bfloat16)

    # Tier 1 builds InGraphCosSinCache directly at __init__ with 0 buffers.
    _assert_cache_buffers_gone(tier1_rope)
    assert isinstance(tier1_rope.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier1_rope, allow_table_recovery=False
        )
        == 0
    )

    # Tier 2 converts a Tensor-backed instance via patch_cos_sin_cache_buffers.
    assert not rope_patch.is_in_graph_rope_eligible(tier2_rope)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier2_rope, allow_table_recovery=False
        )
        == 1
    )
    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            tier2_rope, allow_table_recovery=False
        )
        == 0
    )
    _assert_cache_buffers_gone(tier2_rope)
    assert isinstance(tier2_rope.cos_sin_cache, rope_in_graph.InGraphCosSinCache)

    for candidate in (tier1_rope, tier2_rope):
        moved = candidate.cos_sin_cache.to(q.device, dtype=q.dtype)
        torch.testing.assert_close(moved[:16], ref_slice, rtol=1e-3, atol=1.6e-2)

        out_q, out_k = candidate(positions, q.clone(), k.clone())
        torch.testing.assert_close(out_q, ref_q, rtol=0, atol=1.6e-2)
        torch.testing.assert_close(out_k, ref_k, rtol=0, atol=1.6e-2)


def test_content_fallback_recovers_mscale_table() -> None:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2).float() / 32.0))
    angles = torch.outer(torch.arange(512, dtype=torch.float32), inv_freq)
    mscale = 1.34
    table = torch.cat((angles.cos() * mscale, angles.sin() * mscale), dim=-1)

    module = nn.Module()
    module.register_buffer("cos_sin_cache", table, persistent=False)
    assert rope_patch.patch_cos_sin_cache_buffers(module) == 0
    assert isinstance(module.cos_sin_cache, torch.Tensor)
    assert (
        rope_patch.patch_cos_sin_cache_buffers(module, allow_table_recovery=True) == 1
    )
    assert isinstance(module.cos_sin_cache, rope_in_graph.InGraphCosSinCache)
    idx = torch.tensor([0, 1, 37, 511])
    torch.testing.assert_close(
        module.cos_sin_cache[idx], table[idx], rtol=1e-5, atol=1e-5
    )


def test_deepseek_v4_fope_complex_and_dflash_alias_handling() -> None:
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
        DeepseekV4ScalingRotaryEmbedding,
    )
    from vllm.model_executor.layers.rotary_embedding.fope import FourierRotaryEmbedding
    from vllm.model_executor.layers.rotary_embedding.llama4_vision_rope import (
        Llama4VisionRotaryEmbedding,
    )

    with set_current_vllm_config(_vllm_config(1024)):
        dsv4_rope = DeepseekV4ScalingRotaryEmbedding(
            _HEAD_SIZE,
            _HEAD_SIZE,
            256,
            10000.0,
            False,
            4.0,
            torch.float32,
            beta_fast=32,
            beta_slow=1,
            mscale=1.0,
            mscale_all_dim=1.0,
        )
        fope = FourierRotaryEmbedding(
            _HEAD_SIZE,
            _HEAD_SIZE,
            256,
            10000.0,
            True,
            torch.float32,
            init_cache=False,
            num_key_value_heads=2,
            num_inv_freq=8,
            fope_sep_head=True,
            fope_init_factor=0.1,
        )
        vision_rope = Llama4VisionRotaryEmbedding(
            _HEAD_SIZE,
            _HEAD_SIZE,
            16,
            10000.0,
            False,
            torch.complex64,
        )

    assert vision_rope.cos_sin_cache.dtype == torch.complex64

    draft_wrapper = nn.Module()
    draft_wrapper.dsv4 = dsv4_rope
    draft_wrapper.fope = fope
    draft_wrapper.vision = vision_rope
    draft_wrapper.normal = _SyntheticRotaryEmbedding(dtype=torch.float32)
    draft_wrapper._rope_cos_sin_cache = draft_wrapper.normal.cos_sin_cache

    assert (
        rope_patch.patch_in_graph_rotary_embedding(
            draft_wrapper, allow_table_recovery=True
        )
        == 1
    )
    assert "_rope_cos_sin_cache" not in draft_wrapper.__dict__
    assert isinstance(dsv4_rope.cos_sin_cache, torch.Tensor)
    assert isinstance(fope.cos_sin_cache, torch.Tensor)
    assert isinstance(vision_rope.cos_sin_cache, torch.Tensor)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "variant", ["standard", "partial", "mrope_2d", "gptj", "proportional"]
)
def test_in_graph_spec_forward_matches_table_across_variants(
    dtype: torch.dtype, variant: str
) -> None:
    if variant == "standard":
        rope: nn.Module = _SyntheticRotaryEmbedding(dtype=dtype)
        positions = torch.tensor([0, 1, 13, 127, 510], dtype=torch.int32)
    elif variant == "partial":
        rope = _SyntheticRotaryEmbedding(rotary_dim=_HEAD_SIZE // 2, dtype=dtype)
        positions = torch.tensor([0, 1, 13, 127, 510], dtype=torch.int32)
    elif variant == "mrope_2d":
        rope = _SyntheticRotaryEmbedding(
            mrope_section=[8, 12, 12], mrope_interleaved=True, dtype=dtype
        )
        positions = torch.tensor(
            [[0, 1, 2, 3, 4], [0, 4, 5, 6, 7], [0, 8, 9, 10, 11]], dtype=torch.int32
        )
    elif variant == "gptj":
        rope = _SyntheticRotaryEmbedding(is_neox_style=False, dtype=dtype)
        positions = torch.tensor([0, 1, 13, 127, 510], dtype=torch.int32)
    else:
        rope = _SyntheticProportionalRotaryEmbedding(dtype=dtype)
        positions = torch.tensor([0, 1, 13, 127, 510], dtype=torch.int32)

    ref_rope = copy.deepcopy(rope)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(rope, allow_table_recovery=True) == 1
    )
    num_tokens = positions.shape[-1]
    q, k = _qk(num_tokens, dtype=dtype)
    offsets = torch.ones_like(positions)

    out_q, out_k = rope(positions, q.clone(), k.clone(), offsets=offsets)
    ref_q, ref_k = ref_rope(positions, q.clone(), k.clone(), offsets=offsets)
    assert out_k is not None and ref_k is not None
    tol = (
        {"rtol": 1e-6, "atol": 1e-6}
        if dtype == torch.float32
        else {"rtol": 0, "atol": 1.6e-2}
    )
    torch.testing.assert_close(out_q, ref_q, **tol)
    torch.testing.assert_close(out_k, ref_k, **tol)


def test_patched_forward_embeds_inv_freq_without_pow() -> None:
    """The patched forward must not recompute `inv_freq` as `base ** (2i / d)`."""
    rope = _SyntheticRotaryEmbedding()
    assert (
        rope_patch.patch_in_graph_rotary_embedding(rope, allow_table_recovery=True) == 1
    )
    positions = torch.tensor([0, 1, 13, 127, 511], dtype=torch.int32)
    q, k = _qk(positions.numel())

    with _RecordOps() as rec:
        rope(positions, q, k)
    assert "aten.cos" in rec.ops, rec.ops
    assert "aten.pow" not in rec.ops, rec.ops


class _TwoSegmentLongRope(nn.Module):
    """Two-segment LongRoPE module for testing per-segment sampled verification."""

    def __init__(self, short_len: int = 1024, long_len: int = 8192) -> None:
        super().__init__()
        self.head_size = _HEAD_SIZE
        self.rotary_dim = _HEAD_SIZE
        self.base = 10000.0
        self.original_max_position_embeddings = short_len
        self.max_position_embeddings = long_len
        half = _HEAD_SIZE // 2
        self.short_factor = [1.0 + 0.05 * i for i in range(half)]
        self.long_factor = [2.0 + 0.25 * i for i in range(half)]
        self.short_mscale = 1.1
        self.long_mscale = 1.3

        short_inv = self._compute_inv_freq(self.short_factor)
        long_inv = self._compute_inv_freq(self.long_factor)
        f_short = torch.outer(torch.arange(short_len, dtype=torch.float32), short_inv)
        f_long = torch.outer(torch.arange(long_len, dtype=torch.float32), long_inv)
        c_short = torch.cat(
            (f_short.cos() * self.short_mscale, f_short.sin() * self.short_mscale),
            dim=-1,
        )
        c_long = torch.cat(
            (f_long.cos() * self.long_mscale, f_long.sin() * self.long_mscale),
            dim=-1,
        )
        self.register_buffer(
            "cos_sin_cache", torch.cat((c_short, c_long), dim=0), persistent=False
        )

    def _compute_inv_freq(self, rescale_factors: list[float]) -> torch.Tensor:
        rf = torch.tensor(rescale_factors, dtype=torch.float32)
        exp = torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim
        return 1.0 / (rf * (self.base**exp))


def test_segment_verification_samples_rows() -> None:
    # > 4096 rows exercises the sampled prefix + strided + tail index path.
    clean = _SyntheticRotaryEmbedding(max_position=8192, scaling_factor=4.0)
    assert rope_patch.patch_cos_sin_cache_buffers(clean, allow_table_recovery=True) == 1
    assert isinstance(clean.cos_sin_cache, rope_in_graph.InGraphCosSinCache)

    # Corrupt a prefix row (7), a strided row (4096), and a tail-only row (8190):
    # analytical and content fallback must reject each.
    for bad_row in (7, 4096, 8190):
        corrupt = _SyntheticRotaryEmbedding(max_position=8192, scaling_factor=4.0)
        buf = corrupt.cos_sin_cache
        buf[bad_row] = -buf[bad_row] + 2.0
        assert (
            rope_patch.patch_cos_sin_cache_buffers(corrupt, allow_table_recovery=True)
            == 0
        )
        assert corrupt.cos_sin_cache is buf

    # ~1.5e-4 OpenMP CPU slice jitter on FP32 tables must NOT split workers,
    # whereas a semantic 1e-2 error on a synthetic rotary table is rejected.
    jittered = _SyntheticRotaryEmbedding(
        max_position=8192, scaling_factor=4.0, dtype=torch.float32
    )
    jittered.cos_sin_cache[7, 0] += 1.8e-4
    assert (
        rope_patch.patch_cos_sin_cache_buffers(jittered, allow_table_recovery=True) == 1
    )
    assert isinstance(jittered.cos_sin_cache, rope_in_graph.InGraphCosSinCache)

    wrong_mscale = _SyntheticRotaryEmbedding(
        max_position=8192, scaling_factor=4.0, dtype=torch.float32
    )
    wrong_mscale.cos_sin_cache[7, 0] += 1e-2
    assert (
        rope_patch._module_cos_sin_segments(
            wrong_mscale, wrong_mscale.cos_sin_cache, allow_table_recovery=True
        )
        is None
    )

    # Corrupt row 7 of the second segment (`short_len + 7`) on a multi-segment table.
    clean_long = _TwoSegmentLongRope()
    assert (
        rope_patch.patch_cos_sin_cache_buffers(clean_long, allow_table_recovery=True)
        == 1
    )
    assert isinstance(clean_long.cos_sin_cache, rope_in_graph.InGraphCosSinCache)

    corrupt_seg2 = _TwoSegmentLongRope()
    buf_seg2 = corrupt_seg2.cos_sin_cache
    seg2_row7 = corrupt_seg2.original_max_position_embeddings + 7
    buf_seg2[seg2_row7] = -buf_seg2[seg2_row7] + 2.0
    assert (
        rope_patch.patch_cos_sin_cache_buffers(corrupt_seg2, allow_table_recovery=True)
        == 0
    )
    assert corrupt_seg2.cos_sin_cache is buf_seg2


def test_fope_2d_buffer_skips_content_fallback() -> None:
    """A 2D FoPE module (`cos_coef` / `update_cache`) must never hit the fallback."""

    class _FakeFoPE2D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cos_coef = nn.Parameter(torch.eye(4), requires_grad=False)
            self.update_cache = True
            inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2).float() / 8.0))
            angles = torch.outer(torch.arange(32, dtype=torch.float32), inv_freq)
            self.register_buffer(
                "cos_sin_cache",
                torch.cat((angles.cos(), angles.sin()), dim=-1),
                persistent=False,
            )

    fope = _FakeFoPE2D()
    orig = fope.cos_sin_cache
    assert rope_patch.patch_cos_sin_cache_buffers(fope, allow_table_recovery=True) == 0
    assert fope.cos_sin_cache is orig


def test_fallback_memo_distinguishes_storage_views() -> None:
    """Two views of one storage at different offsets must recover distinct segments."""

    def _make_table(base: float) -> torch.Tensor:
        inv = 1.0 / (base ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8.0))
        ang = torch.outer(torch.arange(16, dtype=torch.float32), inv)
        return torch.cat((ang.cos(), ang.sin()), dim=-1)

    t1 = _make_table(10000.0)
    t2 = _make_table(500000.0)
    slab = torch.cat((t1, t2), dim=0)
    assert slab[:16].untyped_storage().data_ptr() == (
        slab[16:].untyped_storage().data_ptr()
    )

    m1 = nn.Module()
    m1.register_buffer("cos_sin_cache", slab[:16], persistent=False)
    m2 = nn.Module()
    m2.register_buffer("cos_sin_cache", slab[16:], persistent=False)
    root = nn.ModuleDict({"a": m1, "b": m2})

    assert rope_patch.patch_cos_sin_cache_buffers(root, allow_table_recovery=True) == 2
    idx = torch.tensor([1, 5, 11], dtype=torch.int64)
    torch.testing.assert_close(m1.cos_sin_cache[idx], t1[idx], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m2.cos_sin_cache[idx], t2[idx], atol=1e-6, rtol=1e-6)


def test_in_graph_cos_sin_cache_normalizes_indexless_device() -> None:
    seg = rope_in_graph.CosSinSegment(start=0, length=16, inv_freq=(1.0, 0.1))
    cache = rope_in_graph.InGraphCosSinCache(
        [seg], torch.bfloat16, 16, 4, device=torch.device("cuda")
    )
    assert cache.device == torch.device("cuda", 0)
    assert cache.to("cuda") is cache
    assert cache.to(torch.device("cuda")) is cache
    assert cache.to(torch.device("cuda", 0), dtype=torch.bfloat16) is cache

    rank3 = rope_in_graph.InGraphCosSinCache(
        [seg], torch.bfloat16, 16, 4, device=torch.device("cuda", 3)
    )
    assert rank3.to("cuda") is rank3
    assert rank3.device == torch.device("cuda", 3)


def test_rotary_base_never_uses_content_recovery_and_warns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unrecognized `RotaryEmbeddingBase` subclasses must not hit content recovery."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

    class _UnknownFutureRotaryEmbedding(RotaryEmbeddingBase):
        pass

    warnings_seen: list[str] = []
    monkeypatch.setattr(
        rope_patch.logger,
        "warning_once",
        lambda msg, *args: warnings_seen.append(msg % args if args else msg),
    )

    with set_current_vllm_config(_vllm_config(1024)):
        unknown = _UnknownFutureRotaryEmbedding(
            _HEAD_SIZE, _HEAD_SIZE, 256, 10000.0, True, torch.float32
        )

    orig_buf = unknown.cos_sin_cache
    assert isinstance(orig_buf, torch.Tensor)
    # Its buffer IS a valid harmonic [cos | sin] table that content recovery
    # would match on a plain nn.Module; the RotaryEmbeddingBase gate skips it.
    assert (
        rope_patch._recover_rope_segment_from_table(*orig_buf.chunk(2, dim=-1))
        is not None
    )
    assert rope_in_graph.spec_from_module(unknown) is None
    assert (
        rope_patch.patch_in_graph_rotary_embedding(unknown, allow_table_recovery=True)
        == 0
    )
    assert unknown.cos_sin_cache is orig_buf
    assert any("_UnknownFutureRotaryEmbedding" in w for w in warnings_seen)


def test_require_cos_sin_table_exempts_module_from_sweep() -> None:
    """A module marked with `require_cos_sin_table` keeps its table in a sweep."""
    synth = _SyntheticRotaryEmbedding(dtype=torch.float32)
    table = synth.cos_sin_cache
    rope_in_graph.require_cos_sin_table(synth)
    assert (
        rope_patch.patch_in_graph_rotary_embedding(synth, allow_table_recovery=True)
        == 0
    )
    assert synth.cos_sin_cache is table


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        ({}, False),
        ({"tpu_rope_in_graph": False}, False),
        ({"tpu_rope_in_graph": True}, True),
    ],
)
def test_in_graph_enabled_reads_additional_config(
    additional_config: dict, expected: bool
) -> None:
    config = SimpleNamespace(additional_config=additional_config)
    assert rope_in_graph_flag.in_graph_enabled(config) is expected


def test_in_graph_enabled_is_off_without_a_config() -> None:
    assert rope_in_graph_flag.in_graph_enabled(None) is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_in_graph_enabled_rejects_non_bool(value: object) -> None:
    config = SimpleNamespace(additional_config={"tpu_rope_in_graph": value})
    with pytest.raises(ValueError, match="tpu_rope_in_graph"):
        rope_in_graph_flag.in_graph_enabled(config)


@pytest.mark.parametrize("enabled", [False, True])
def test_tier1_reads_the_flag_from_the_current_vllm_config(enabled: bool) -> None:
    """No monkeypatching: the flag reaches Tier 1 through `set_current_vllm_config`."""
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        additional_config={"tpu_rope_in_graph": enabled},
    )
    with set_current_vllm_config(config):
        rope = RotaryEmbedding(
            _HEAD_SIZE, _HEAD_SIZE, 128, 10000.0, True, torch.float32
        )
    expected = rope_in_graph.InGraphCosSinCache if enabled else torch.Tensor
    assert isinstance(rope.cos_sin_cache, expected)


def test_in_graph_cos_sin_cache_nd_index_and_half_chunks() -> None:
    """`InGraphCosSinCache` handles N-D indices and `half` across segments."""
    segs = (
        rope_in_graph.CosSinSegment(
            start=0, length=8, inv_freq=(1.0, 0.25), mscale=1.0
        ),
        rope_in_graph.CosSinSegment(
            start=8, length=8, inv_freq=(0.5, 0.125), mscale=1.25
        ),
    )
    full = rope_in_graph.InGraphCosSinCache(segs, torch.float32, 16, 4)
    cos_only = rope_in_graph.InGraphCosSinCache(segs, torch.float32, 16, 2, half="cos")
    sin_only = rope_in_graph.InGraphCosSinCache(segs, torch.float32, 16, 2, half="sin")

    idx_2d = torch.tensor([[0, 3, 8], [7, 11, 15]], dtype=torch.int64)
    expected_2d = torch.stack([full[idx_2d[0]], full[idx_2d[1]]], dim=0)
    assert torch.equal(full[idx_2d], expected_2d)

    exp_cos, exp_sin = expected_2d.chunk(2, dim=-1)
    assert torch.equal(cos_only[idx_2d], exp_cos)
    assert torch.equal(sin_only[idx_2d], exp_sin)

    idx_1d = idx_2d.reshape(-1)
    exp_cos_1d, exp_sin_1d = full[idx_1d].chunk(2, dim=-1)
    assert torch.equal(cos_only[idx_1d], exp_cos_1d)
    assert torch.equal(sin_only[idx_1d], exp_sin_1d)


def test_in_graph_cos_sin_cache_torch_index_select_reads_arguments_by_name() -> None:
    segment = rope_in_graph.CosSinSegment(start=0, length=16, inv_freq=(1.0, 0.25))
    cache = rope_in_graph.InGraphCosSinCache([segment], torch.float32, 16, 4)
    idx = torch.tensor([0, 3, 15], dtype=torch.int64)
    expected = cache[idx]

    assert torch.equal(torch.index_select(cache, 0, idx), expected)
    assert torch.equal(torch.index_select(cache, index=idx, dim=0), expected)
    assert torch.equal(torch.index_select(input=cache, dim=0, index=idx), expected)


def test_linear_scaling_spec_folds_inv_freq_and_matches_table() -> None:
    """`LinearScaling` folds power-of-two factors into `inv_freq` per segment."""
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.layers.rotary_embedding import LinearScalingRotaryEmbedding

    factors = [1.0, 2.0]
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with set_current_vllm_config(_vllm_config(64, in_graph=False)):
            rope = LinearScalingRotaryEmbedding(
                _HEAD_SIZE, _HEAD_SIZE, 64, 10000.0, True, factors, torch.bfloat16
            )
        ref_table = rope.cos_sin_cache
        assert isinstance(ref_table, torch.Tensor)

        spec = rope_in_graph.spec_from_module(rope)
        assert spec is not None
        assert len(spec.segments) == 2
        base_inv = rope._compute_inv_freq(rope.base)
        for seg, sf in zip(spec.segments, factors, strict=True):
            assert seg.inv_freq == rope_in_graph.to_f32_tuple(base_inv / sf)

        all_pos = torch.arange(ref_table.shape[0], dtype=torch.int64)
        assert torch.equal(
            rope_in_graph.eval_spec_cos_sin(spec, all_pos, torch.bfloat16),
            ref_table,
        )
    finally:
        torch.set_num_threads(prev_threads)


def test_serving_default_leaves_unregistered_tables_and_pairs_untouched() -> None:
    """Serving mode (`allow_table_recovery=False`) uses per-class specs only."""
    pair = _PairedTableRope(64, 512, 10000.0)
    synth = _SyntheticRotaryEmbedding(dtype=torch.float32)
    raw = nn.Module()
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2).float() / 32.0))
    angles = torch.outer(torch.arange(128, dtype=torch.float32), inv_freq)
    raw.register_buffer(
        "cos_sin_cache",
        torch.cat((angles.cos(), angles.sin()), dim=-1),
        persistent=False,
    )
    root = nn.ModuleDict({"pair": pair, "synth": synth, "raw": raw})

    assert (
        rope_patch.patch_in_graph_rotary_embedding(root, allow_table_recovery=False)
        == 0
    )
    assert isinstance(pair.cos, torch.Tensor) and isinstance(pair.sin, torch.Tensor)
    assert isinstance(synth.cos_sin_cache, torch.Tensor)
    assert isinstance(raw.cos_sin_cache, torch.Tensor)

    # Export mode (`allow_table_recovery=True`) still recovers all three.
    assert (
        rope_patch.patch_in_graph_rotary_embedding(root, allow_table_recovery=True) == 3
    )
    assert isinstance(pair.cos, rope_in_graph.InGraphCosSinCache)
    assert isinstance(pair.sin, rope_in_graph.InGraphCosSinCache)
    assert not hasattr(synth, "cos_sin_cache")
    assert isinstance(raw.cos_sin_cache, rope_in_graph.InGraphCosSinCache)


def test_cos_sin_at_indexes_tensor_and_in_graph_cache() -> None:
    """`cos_sin_at` delegates 1-D `positions` to `cos_sin_cache.index_select`."""
    inv_freq = tuple(float(x) for x in 1.0 / (10000.0 ** (torch.arange(0, 8, 2) / 8.0)))
    seg = rope_in_graph.CosSinSegment(0, 32, inv_freq, 1.0)
    in_graph_cache = rope_in_graph.InGraphCosSinCache(
        (seg,), torch.bfloat16, 32, 8, cache_dtype=torch.float32
    )
    table = in_graph_cache[:]
    pos = torch.tensor([0, 5, 17, 31], dtype=torch.int32)
    expected = table.index_select(0, pos)

    assert torch.equal(
        rope_in_graph.cos_sin_at(SimpleNamespace(cos_sin_cache=table), pos),
        expected,
    )
    assert torch.equal(
        rope_in_graph.cos_sin_at(SimpleNamespace(cos_sin_cache=in_graph_cache), pos),
        expected,
    )


def test_vllm_deepseek_scaling_forward_uses_spec_without_indexing_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`VllmDeepseekScalingRotaryEmbedding.forward_native` routes through its spec."""
    from vllm.config import set_current_vllm_config

    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmDeepseekScalingRotaryEmbedding,
    )

    kwargs = {
        "head_size": _HEAD_SIZE,
        "rotary_dim": _HEAD_SIZE,
        "max_position_embeddings": 256,
        "base": 10000.0,
        "is_neox_style": False,
        "scaling_factor": 4.0,
        "dtype": torch.bfloat16,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        cfg = _vllm_config(1024, in_graph=False)
        with set_current_vllm_config(cfg):
            ref = VllmDeepseekScalingRotaryEmbedding(**kwargs)
            cfg.additional_config[rope_in_graph_flag.KEY] = True
            uut = VllmDeepseekScalingRotaryEmbedding(**kwargs)

        positions = torch.tensor([0, 3, 17, 127], dtype=torch.int32)
        q, k = _qk(positions.numel(), dtype=torch.bfloat16)
        ref_q, ref_k = ref.forward_native(positions, q.clone(), k.clone())

        def _forbid_index_select(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("forward_native/get_cos_sin must not index cache")

        monkeypatch.setattr(uut.cos_sin_cache, "index_select", _forbid_index_select)
        monkeypatch.setattr(
            type(uut.cos_sin_cache),
            "__getitem__",
            _forbid_index_select,
        )
        out_q, out_k = uut.forward_native(positions, q.clone(), k.clone())
        torch.testing.assert_close(out_q, ref_q, rtol=0, atol=0)
        torch.testing.assert_close(out_k, ref_k, rtol=0, atol=0)

        ref_cos, ref_sin = ref.get_cos_sin(16)
        uut_cos, uut_sin = uut.get_cos_sin(16)
        torch.testing.assert_close(uut_cos, ref_cos, rtol=0, atol=0)
        torch.testing.assert_close(uut_sin, ref_sin, rtol=0, atol=0)
    finally:
        torch.set_num_threads(prev_threads)
