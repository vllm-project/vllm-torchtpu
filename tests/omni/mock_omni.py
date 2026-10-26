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
"""Mocks for vllm-omni modules for unit testing."""

import sys
from unittest.mock import MagicMock

import torch

# Unconditionally mock vllm-omni for unit tests so test behavior is
# deterministic and hermetic regardless of whether vllm-omni is installed.
for mod_name in list(sys.modules.keys()):
    if mod_name == "vllm_omni" or mod_name.startswith("vllm_omni."):
        del sys.modules[mod_name]


class _MockOmniPlatform:
    device_control_env_var = "TPU_VISIBLE_CHIPS"

    def is_cuda(self):
        return False

    def is_npu(self):
        return False

    def is_xpu(self):
        return False

    def is_rocm(self):
        return False

    def is_musa(self):
        return False

    def is_out_of_tree(self):
        return True

    def has_flash_attn_package(self):
        return False

    def supports_torch_inductor(self):
        return False

    def get_torch_device(self):
        return None


class _MockSDPABackend:
    @classmethod
    def get_name(cls):
        return "SDPA"

    @classmethod
    def get_impl_cls(cls):
        return _MockSDPAImpl

    @classmethod
    def supports_attention_mask(cls):
        return True


class _MockSDPAImpl:
    def __init__(self, num_heads, head_size, scale=None, num_kv_heads=None, **kwargs):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.softmax_scale = kwargs.get("softmax_scale", scale)
        self.causal = kwargs.get("causal", False)

    def forward(self, query, key, value, attn_metadata=None):
        # AttentionImpl.forward has no branch for out-of-tree platforms.
        raise NotImplementedError("No forward implementation for platform: oot")

    def _forward_impl(
        self, query, key, value, attn_metadata=None, mask_mode="broadcast_k"
    ):
        # Mirrors _maybe_reshape_attn_mask: a 2D [B, Sk] keep mask is cast to
        # bool and broadcast over heads and queries.
        assert mask_mode == "broadcast_k"
        attn_mask = getattr(attn_metadata, "attn_mask", None)
        if (
            attn_mask is not None
            and attn_mask.ndim == 2
            and attn_mask.shape == (query.shape[0], key.shape[1])
        ):
            attn_mask = attn_mask.to(torch.bool)[:, None, None, :]
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key = torch.repeat_interleave(key, n_rep, dim=2)
            value = torch.repeat_interleave(value, n_rep, dim=2)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, scale=self.softmax_scale
        )
        return out.transpose(1, 2)


class _MockOmniPlatformEnum:
    OOT = "oot"
    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    MUSA = "musa"


def _init_mock_qwen_rope(self):
    self.pos_freqs = torch.polar(
        torch.ones(8, 4), torch.arange(32, dtype=torch.float32).view(8, 4)
    )


def _forward_mock_qwen_rope(self, video_fhw, txt_seq_lens, device):
    return (
        torch.cat([self.pos_freqs[:2]], dim=0),
        self.pos_freqs[2 : 2 + max(txt_seq_lens), ...],
    )


class _MockQwenEmbedRope:
    __init__ = _init_mock_qwen_rope
    forward = _forward_mock_qwen_rope


class _MockQwenEmbedLayer3DRope:
    __init__ = _init_mock_qwen_rope
    forward = _forward_mock_qwen_rope


class _MockColumnParallelGELU(torch.nn.Module):
    """vLLM-Omni's Wan FFN up-projection + tanh GELU."""

    def __init__(self, dim_in, dim_out, approximate="tanh"):
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)
        self.approximate = approximate

    def forward(self, x):
        return torch.nn.functional.gelu(self.proj(x), approximate=self.approximate)


def _mock_ltx2_rope_cls():
    """A fresh class of vLLM-Omni's LTX-2 RoPE (patches install on the class)."""

    class LTX2AudioVideoRotaryPosEmbed(torch.nn.Module):
        """LTX-2 RoPE: (cos, sin) of coords with per-head frequencies."""

        def __init__(self, rope_type="split", heads=3, head_dim=8):
            super().__init__()
            self.rope_type = rope_type
            self.freqs = 4 * torch.rand(heads, head_dim // 2)
            self.calls = 0

        def forward(self, coords, device=None, out_dtype=None):
            self.calls += 1
            # (B, P, T, 2) patch bounds -> (B, H, T, D/2) angles.
            pos = coords.float().mean(dim=(1, 3))
            angles = pos[:, None, :, None] * self.freqs[:, None, :]
            if self.rope_type == "interleaved":
                # (B, T, H * D), each frequency repeated for its pair.
                angles = angles.transpose(1, 2).flatten(2).repeat_interleave(2, -1)
            cos, sin = angles.cos(), angles.sin()
            if out_dtype is not None:
                cos, sin = cos.to(out_dtype), sin.to(out_dtype)
            return cos, sin

    return LTX2AudioVideoRotaryPosEmbed


def _mock_apply_split_rotary_emb(x, freqs, *, head_dim):
    """vLLM-Omni's LTX-2 split RoPE of (B, T, H * D) with (B, H, T, D/2) tables.

    Same operations as upstream: products in the promoted dtype of x and the
    tables, addcmul_, and no cast back to x.dtype.
    """
    cos, sin = freqs
    b, t = x.shape[:2]
    h, r = cos.shape[1], cos.shape[-1]
    split_x = x.reshape(b, t, h, head_dim).swapaxes(1, 2).reshape(b, h, t, 2, r)
    cos_u, sin_u = cos.unsqueeze(-2), sin.unsqueeze(-2)
    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, split_x[..., 1:, :])
    out[..., 1:, :].addcmul_(sin_u, split_x[..., :1, :])
    return out.reshape(b, h, t, head_dim).swapaxes(1, 2).reshape(b, t, h * head_dim)


class _MockRotaryEmbedding(torch.nn.Module):
    """vLLM-Omni's CustomOp-style base RoPE (native formula)."""

    def __init__(self, is_neox_style=False, half_head_dim=True):
        super().__init__()
        self.is_neox_style = is_neox_style
        self.interleaved = not is_neox_style
        self.half_head_dim = half_head_dim
        self._forward_method = self.forward_native

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def _prepare_half_head_dim_cos_sin(self, cos, sin):
        if cos.dim() == 3:
            cos = cos[0]
            sin = sin[0]
        if not self.half_head_dim:
            cos = cos[..., ::2] if self.interleaved else cos[..., : cos.shape[-1] // 2]
            sin = sin[..., ::2] if self.interleaved else sin[..., : sin.shape[-1] // 2]
        return cos, sin

    def forward_native(self, x, cos, sin):
        cos, sin = self._prepare_half_head_dim_cos_sin(cos, sin)
        ro_dim = cos.shape[-1] * 2
        x_rot = x[..., :ro_dim]
        if self.interleaved:
            cos = cos.unsqueeze(-2).repeat_interleave(2, dim=-1)
            sin = sin.unsqueeze(-2).repeat_interleave(2, dim=-1)
            x1, x2 = x_rot[..., ::2], x_rot[..., 1::2]
            rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
        else:
            cos = torch.cat([cos, cos], dim=-1).unsqueeze(-2)
            sin = torch.cat([sin, sin], dim=-1).unsqueeze(-2)
            x1, x2 = x_rot.chunk(2, dim=-1)
            rot = torch.cat((-x2, x1), dim=-1)
        return torch.cat([x_rot * cos + rot * sin, x[..., ro_dim:]], dim=-1)


class _MockRotaryEmbeddingWan(_MockRotaryEmbedding):
    """vLLM-Omni's CustomOp-style Wan RoPE (native interleaved formula)."""

    def __init__(self, is_neox_style=False, half_head_dim=False):
        super().__init__(is_neox_style=is_neox_style, half_head_dim=half_head_dim)

    def forward_native(self, x, cos, sin):
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2, -1).to(x.dtype)


class _MockRMSNorm(torch.nn.Module):
    """vLLM-Omni's RMSNorm parameters (weight, variance_epsilon)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps


class _MockWanSelfAttention(torch.nn.Module):
    """vLLM-Omni's Wan self-attention; reports that its forward ran."""

    def forward(self, hidden_states, rotary_emb=None, attn_metadata=None):
        return "native"


class _MockWorkerProc:
    """vLLM-Omni's diffusion WorkerProc with its async output loop."""

    def _async_output_loop(self):
        pass


vllm_omni_mock = MagicMock()
vllm_omni_mock.platforms.interface.OmniPlatform = _MockOmniPlatform
vllm_omni_mock.platforms.interface.OmniPlatformEnum = _MockOmniPlatformEnum
vllm_omni_mock.diffusion.attention.backends.sdpa.SDPABackend = _MockSDPABackend
vllm_omni_mock.diffusion.attention.backends.sdpa.SDPAImpl = _MockSDPAImpl
vllm_omni_mock.diffusion.attention.backends.registry.register_diffusion_backend = (
    lambda *args, **kwargs: None
)
qwen_transformer_mock = (
    vllm_omni_mock.diffusion.models.qwen_image.qwen_image_transformer
)
qwen_transformer_mock.QwenEmbedRope = _MockQwenEmbedRope
qwen_transformer_mock.QwenEmbedLayer3DRope = _MockQwenEmbedLayer3DRope

_ltx2_transformer = vllm_omni_mock.diffusion.models.ltx2.ltx2_transformer
_ltx2_transformer.LTX2AudioVideoRotaryPosEmbed = _mock_ltx2_rope_cls()
_ltx2_transformer.apply_split_rotary_emb = _mock_apply_split_rotary_emb
_ltx2_transformer.get_tensor_model_parallel_world_size = lambda: 1
_ltx2_transformer.get_tensor_model_parallel_rank = lambda: 0
vllm_omni_mock.diffusion.layers.rope.RotaryEmbedding = _MockRotaryEmbedding
vllm_omni_mock.diffusion.layers.rope.RotaryEmbeddingWan = _MockRotaryEmbeddingWan
vllm_omni_mock.diffusion.layers.norm.RMSNorm = _MockRMSNorm
_wan2_2 = vllm_omni_mock.diffusion.models.wan2_2
_wan2_2.wan2_2_transformer.ColumnParallelGELU = _MockColumnParallelGELU
_wan2_2.wan2_2_transformer.WanSelfAttention = _MockWanSelfAttention
vllm_omni_mock.diffusion.worker.diffusion_worker.WorkerProc = _MockWorkerProc

sys.modules["vllm_omni"] = vllm_omni_mock
sys.modules["vllm_omni.platforms"] = vllm_omni_mock.platforms
sys.modules["vllm_omni.platforms.interface"] = vllm_omni_mock.platforms.interface
sys.modules["vllm_omni.diffusion"] = vllm_omni_mock.diffusion
sys.modules["vllm_omni.diffusion.ipc"] = vllm_omni_mock.diffusion.ipc
sys.modules["vllm_omni.diffusion.diffusion_engine"] = (
    vllm_omni_mock.diffusion.diffusion_engine
)
sys.modules["vllm_omni.diffusion.attention"] = vllm_omni_mock.diffusion.attention
sys.modules["vllm_omni.diffusion.attention.backends"] = (
    vllm_omni_mock.diffusion.attention.backends
)
sys.modules["vllm_omni.diffusion.attention.backends.registry"] = (
    vllm_omni_mock.diffusion.attention.backends.registry
)
sys.modules["vllm_omni.diffusion.attention.backends.sdpa"] = (
    vllm_omni_mock.diffusion.attention.backends.sdpa
)
sys.modules["vllm_omni.diffusion.models"] = vllm_omni_mock.diffusion.models
sys.modules["vllm_omni.diffusion.models.ltx2"] = vllm_omni_mock.diffusion.models.ltx2
sys.modules["vllm_omni.diffusion.models.qwen_image"] = (
    vllm_omni_mock.diffusion.models.qwen_image
)
sys.modules["vllm_omni.diffusion.models.qwen_image.qwen_image_transformer"] = (
    vllm_omni_mock.diffusion.models.qwen_image.qwen_image_transformer
)
sys.modules["vllm_omni.diffusion.models.wan2_2"] = _wan2_2
sys.modules["vllm_omni.diffusion.models.wan2_2.wan2_2_transformer"] = (
    _wan2_2.wan2_2_transformer
)
sys.modules["vllm_omni.diffusion.models.wan2_2.wan2_2_vace_transformer"] = (
    _wan2_2.wan2_2_vace_transformer
)
sys.modules["vllm_omni.diffusion.models.wan2_2.wan2_2_s2v_transformer"] = (
    _wan2_2.wan2_2_s2v_transformer
)
sys.modules["vllm_omni.diffusion.layers"] = vllm_omni_mock.diffusion.layers
sys.modules["vllm_omni.diffusion.layers.rope"] = vllm_omni_mock.diffusion.layers.rope
sys.modules["vllm_omni.diffusion.layers.norm"] = vllm_omni_mock.diffusion.layers.norm
sys.modules["vllm_omni.diffusion.worker"] = vllm_omni_mock.diffusion.worker
sys.modules["vllm_omni.diffusion.worker.diffusion_worker"] = (
    vllm_omni_mock.diffusion.worker.diffusion_worker
)
