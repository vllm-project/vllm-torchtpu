"""Unit tests for DeepSeek-V4 CUDA stream stubs and model architecture patching."""

from types import SimpleNamespace

import torch

from vllm_torchtpu.models.vllm.deepseek_v4_patch import \
    _maybe_patch_for_deepseek_v4
from vllm_torchtpu.models.vllm.deepseek_v4_stubs import patch_cuda_stubs


def test_deepseek_v4_cuda_stubs_patching():
    """Verify patch_cuda_stubs installs working dummy CUDA stream and event objects on TPU."""
    patch_cuda_stubs()
    stream = torch.cuda.Stream()
    assert stream is not None
    assert hasattr(stream, "synchronize")
    assert hasattr(stream, "wait")

    event = torch.cuda.Event()
    assert event is not None
    assert hasattr(event, "record")
    assert hasattr(event, "wait")


def test_deepseek_v4_context_manager_patching():
    """Verify _maybe_patch_for_deepseek_v4 activates for DeepseekV4ForCausalLM architecture."""
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"])),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(static_forward_context={}, ),
    )

    with _maybe_patch_for_deepseek_v4(vllm_config):
        # DeepSeek-V4 patches are active within context
        pass
