# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch

try:
    from . import mock_omni  # noqa: F401
except (ImportError, ValueError):
    from tests.omni import mock_omni  # noqa: F401

from vllm_torchtpu.omni import register_omni_tpu_platform
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = pytest.mark.cpu_test


def test_register_omni_tpu_platform(monkeypatch):
    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 0)
    assert register_omni_tpu_platform() is None

    monkeypatch.setattr("vllm_torchtpu.omni.get_num_chips", lambda: 8)
    assert register_omni_tpu_platform(
    ) == "vllm_torchtpu.omni.platform.OmniTpuPlatform"


def test_platform_attributes():
    platform = OmniTpuPlatform()
    assert platform.device_name == "tpu"
    assert platform.device_type == "tpu"
    assert platform.dist_backend == "gloo"
    assert platform.supports_torch_inductor() is False
    assert (platform.get_diffusion_attn_backend_cls(
        None, 64) == "vllm_torchtpu.omni.attention.TpuSDPABackend")
    with patch("torch.device") as mock_dev:
        platform.get_torch_device()
        mock_dev.assert_called_once_with("tpu")


def test_device_control_env_var():
    assert OmniTpuPlatform.device_control_env_var == "TPU_VISIBLE_CHIPS"


def test_memory_methods(monkeypatch):

    class MockAccelerator:

        @staticmethod
        def get_memory_info(device=None):
            return (16 * 1024**3, 32 * 1024**3)

        @staticmethod
        def max_memory_reserved(device=None):
            return 16 * 1024**3

        @staticmethod
        def max_memory_allocated(device=None):
            return 16 * 1024**3

    monkeypatch.setattr(torch, "accelerator", MockAccelerator(), raising=False)
    with patch(
            "torch_tpu._internal.utils.hardware.get_hbm_bytes_per_device",
            return_value=32 * 1024**3,
    ):
        assert OmniTpuPlatform.get_device_total_memory() == 32 * 1024**3
        assert OmniTpuPlatform.get_free_memory() == 16 * 1024**3
        assert OmniTpuPlatform.get_device_memory() == (16 * 1024**3,
                                                       32 * 1024**3)
        assert OmniTpuPlatform.max_memory_reserved() == 16 * 1024**3
        assert OmniTpuPlatform.max_memory_allocated() == 16 * 1024**3


def test_tpu_sdpa_attention_backend():
    from vllm_torchtpu.omni.attention import TpuSDPABackend, TpuSDPAImpl

    assert TpuSDPABackend.get_name() == "TPU_SDPA"
    assert TpuSDPABackend.get_impl_cls() is TpuSDPAImpl
    assert TpuSDPABackend.supports_attention_mask() is True

    impl = TpuSDPAImpl(num_heads=4, head_size=16, softmax_scale=0.25)
    query = torch.randn(2, 8, 4, 16)
    key = torch.randn(2, 8, 4, 16)
    value = torch.randn(2, 8, 4, 16)

    res = impl.forward(query, key, value)
    assert res.shape == (2, 8, 4, 16)

    # GQA
    impl_gqa = TpuSDPAImpl(num_heads=8,
                           head_size=16,
                           softmax_scale=0.25,
                           num_kv_heads=2)
    query_gqa = torch.randn(1, 16, 8, 16)
    key_gqa = torch.randn(1, 16, 2, 16)
    value_gqa = torch.randn(1, 16, 2, 16)
    res_gqa = impl_gqa.forward(query_gqa, key_gqa, value_gqa)
    assert res_gqa.shape == (1, 16, 8, 16)


def test_vllm_omni_is_mocked():
    import sys

    import vllm_omni

    try:
        from .mock_omni import vllm_omni_mock
    except (ImportError, ValueError):
        from tests.omni.mock_omni import vllm_omni_mock

    assert sys.modules["vllm_omni"] is vllm_omni_mock
    assert getattr(vllm_omni, "__file__", None) is None


def test_ltx2_safe_decode_output():
    from vllm_omni.diffusion.models.ltx2 import ltx2_runtime

    from vllm_torchtpu.omni.patches import apply_omni_model_specific_patches

    apply_omni_model_specific_patches()

    class TrackedTensor(torch.Tensor):
        cpu_called: bool = False

        def cpu(self, *args, **kwargs):
            self.cpu_called = True
            return super().cpu(*args, **kwargs)

    runtime = ltx2_runtime.LTXRuntime()

    decode_kwargs = dict(
        latents=torch.randn(1, 128, 4, 4),
        audio_latents=torch.randn(1, 8, 16),
        output_type="pt",
        connector_prompt_embeds=torch.randn(1, 10, 64),
        generator=None,
        device=torch.device("cpu"),
        decode_timestep=0.0,
        decode_noise_scale=None,
        prompt_batch_size=1,
    )

    # Case 1: Video and audio are both Tensors (verify .cpu() is invoked)
    v = torch.randn(1, 3, 16, 16).as_subclass(TrackedTensor)
    a = torch.randn(1, 100).as_subclass(TrackedTensor)
    runtime.output = (v, a)
    res = runtime._decode_output(**decode_kwargs)
    assert v.cpu_called is True
    assert a.cpu_called is True
    assert res.output[0].device.type == "cpu"
    assert res.output[1].device.type == "cpu"

    # Case 2: Video is Tensor, audio is None
    v = torch.randn(1, 3, 16, 16).as_subclass(TrackedTensor)
    runtime.output = (v, None)
    res = runtime._decode_output(**decode_kwargs)
    assert v.cpu_called is True
    assert res.output[0].device.type == "cpu"
    assert res.output[1] is None

    # Case 3: Both are None (verify non-tensor outputs pass through safely)
    runtime.output = (None, None)
    res = runtime._decode_output(**decode_kwargs)
    assert res.output == (None, None)
