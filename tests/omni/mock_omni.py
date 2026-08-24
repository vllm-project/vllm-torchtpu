# SPDX-License-Identifier: Apache-2.0
import sys
from unittest.mock import MagicMock

import torch

try:
    import vllm_omni  # noqa: F401
except ImportError:

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

        def __init__(self,
                     num_heads,
                     head_size,
                     scale=None,
                     num_kv_heads=None,
                     **kwargs):
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = scale
            self.num_kv_heads = num_kv_heads or num_heads

        def _forward_impl(self,
                          query,
                          key,
                          value,
                          attn_metadata=None,
                          mask_mode="broadcast_k"):
            if self.num_kv_heads != self.num_heads:
                n_rep = self.num_heads // self.num_kv_heads
                key = torch.repeat_interleave(key, n_rep, dim=2)
                value = torch.repeat_interleave(value, n_rep, dim=2)
            q = query.transpose(1, 2)
            k = key.transpose(1, 2)
            v = value.transpose(1, 2)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, scale=self.scale)
            return out.transpose(1, 2)

    class _MockOmniPlatformEnum:
        OOT = "oot"
        CUDA = "cuda"
        ROCM = "rocm"
        NPU = "npu"
        XPU = "xpu"
        MUSA = "musa"

    vllm_omni_mock = MagicMock()
    vllm_omni_mock.platforms.interface.OmniPlatform = _MockOmniPlatform
    vllm_omni_mock.platforms.interface.OmniPlatformEnum = _MockOmniPlatformEnum
    vllm_omni_mock.diffusion.attention.backends.sdpa.SDPABackend = _MockSDPABackend
    vllm_omni_mock.diffusion.attention.backends.sdpa.SDPAImpl = _MockSDPAImpl
    vllm_omni_mock.diffusion.attention.backends.registry.register_diffusion_backend = (
        lambda *args, **kwargs: None)

    sys.modules["vllm_omni"] = vllm_omni_mock
    sys.modules["vllm_omni.platforms"] = vllm_omni_mock.platforms
    sys.modules[
        "vllm_omni.platforms.interface"] = vllm_omni_mock.platforms.interface
    sys.modules["vllm_omni.diffusion"] = vllm_omni_mock.diffusion
    sys.modules[
        "vllm_omni.diffusion.attention"] = vllm_omni_mock.diffusion.attention
    sys.modules["vllm_omni.diffusion.attention.backends"] = (
        vllm_omni_mock.diffusion.attention.backends)
    sys.modules["vllm_omni.diffusion.attention.backends.registry"] = (
        vllm_omni_mock.diffusion.attention.backends.registry)
    sys.modules["vllm_omni.diffusion.attention.backends.sdpa"] = (
        vllm_omni_mock.diffusion.attention.backends.sdpa)
