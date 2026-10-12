# SPDX-License-Identifier: Apache-2.0
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

    def _forward_impl(
        self, query, key, value, attn_metadata=None, mask_mode="broadcast_k"
    ):
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            key = torch.repeat_interleave(key, n_rep, dim=2)
            value = torch.repeat_interleave(value, n_rep, dim=2)
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=self.scale
        )
        return out.transpose(1, 2)


class _MockOmniPlatformEnum:
    OOT = "oot"
    CUDA = "cuda"
    ROCM = "rocm"
    NPU = "npu"
    XPU = "xpu"
    MUSA = "musa"


class _MockLTXRuntime:
    output = (None, None)

    def _decode_output(self, *args, **kwargs):
        return MagicMock(output=self.output)


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


class _MockRotaryEmbeddingWan(torch.nn.Module):
    """vLLM-Omni's CustomOp-style Wan RoPE (native interleaved formula)."""

    def __init__(self, is_neox_style=False, half_head_dim=False):
        super().__init__()
        self.interleaved = not is_neox_style
        self.half_head_dim = half_head_dim
        self._forward_method = self.forward_native

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_native(self, x, cos, sin):
        x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
        rotated = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return rotated.flatten(-2, -1).to(x.dtype)


vllm_omni_mock = MagicMock()
vllm_omni_mock.platforms.interface.OmniPlatform = _MockOmniPlatform
vllm_omni_mock.platforms.interface.OmniPlatformEnum = _MockOmniPlatformEnum
vllm_omni_mock.diffusion.attention.backends.sdpa.SDPABackend = _MockSDPABackend
vllm_omni_mock.diffusion.attention.backends.sdpa.SDPAImpl = _MockSDPAImpl
vllm_omni_mock.diffusion.attention.backends.registry.register_diffusion_backend = (
    lambda *args, **kwargs: None
)
vllm_omni_mock.diffusion.models.ltx2.ltx2_runtime.LTXRuntime = _MockLTXRuntime
qwen_transformer_mock = (
    vllm_omni_mock.diffusion.models.qwen_image.qwen_image_transformer
)
qwen_transformer_mock.QwenEmbedRope = _MockQwenEmbedRope
qwen_transformer_mock.QwenEmbedLayer3DRope = _MockQwenEmbedLayer3DRope
vllm_omni_mock.diffusion.layers.rope.RotaryEmbeddingWan = _MockRotaryEmbeddingWan

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
sys.modules["vllm_omni.diffusion.models.ltx2.ltx2_runtime"] = (
    vllm_omni_mock.diffusion.models.ltx2.ltx2_runtime
)
sys.modules["vllm_omni.diffusion.models.qwen_image"] = (
    vllm_omni_mock.diffusion.models.qwen_image
)
sys.modules["vllm_omni.diffusion.models.qwen_image.qwen_image_transformer"] = (
    vllm_omni_mock.diffusion.models.qwen_image.qwen_image_transformer
)
sys.modules["vllm_omni.diffusion.layers"] = vllm_omni_mock.diffusion.layers
sys.modules["vllm_omni.diffusion.layers.rope"] = vllm_omni_mock.diffusion.layers.rope
