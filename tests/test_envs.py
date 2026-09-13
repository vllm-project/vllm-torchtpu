"""USE_MOE_EP_KERNEL was deleted: setting it must now fail startup.

It is not aliased to USE_MOE_FUSED_EP_KERNEL, which would enable the kernel
under the recipes that still set the old name and change results they have
already recorded.
"""
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import VllmConfig

from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

REMOVED = "USE_MOE_EP_KERNEL"


@pytest.fixture
def vllm_config():
    """A plain single-host config: nothing else in the check may reject it."""
    config = MagicMock(spec=VllmConfig)
    config.additional_config = {}
    config.model_config.is_hybrid = False
    config.cache_config.enable_prefix_caching = False
    config.compilation_config.compile_sizes = [16, 32]
    config.speculative_config = None
    config.parallel_config.nnodes = 1
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.kv_transfer_config = None
    return config


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_startup_rejects_the_removed_knob(monkeypatch, vllm_config, value):
    """Including `0` and the empty string: what is wrong is naming the knob."""
    monkeypatch.setenv(REMOVED, value)

    with pytest.raises(ValueError, match="USE_MOE_FUSED_EP_KERNEL=1"):
        TpuPlatform.check_and_update_config(vllm_config)


@patch("vllm_torchtpu.platforms.tpu_platform.apply_tpu_patches")
@patch("vllm_torchtpu.platforms.tpu_platform."
       "TpuPlatform._prepare_singlehost_tpu_env")
def test_startup_accepts_the_knob_unset(mock_prepare_env, mock_apply_patches,
                                        monkeypatch, vllm_config):
    monkeypatch.delenv(REMOVED, raising=False)

    TpuPlatform.check_and_update_config(vllm_config)
