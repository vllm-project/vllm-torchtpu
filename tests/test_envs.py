"""USE_MOE_EP_KERNEL was deleted: setting it must now fail startup.

It is not aliased to USE_MOE_FUSED_EP_KERNEL, which would enable the kernel
under the recipes that still set the old name and change results they have
already recorded.
"""

from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig, VllmConfig

from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

REMOVED = "USE_MOE_EP_KERNEL"


@pytest.fixture
def vllm_config():
    """A plain single-host config: nothing else in the check may reject it."""
    config = MagicMock(spec=VllmConfig)
    config.additional_config = {}
    config.model_config.is_hybrid = False
    config.cache_config = CacheConfig(enable_prefix_caching=False)
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


@patch("vllm_torchtpu.patch_registry.apply")
@patch("vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env")
def test_startup_accepts_the_knob_unset(
    mock_prepare_env, mock_apply_patches, monkeypatch, vllm_config
):
    monkeypatch.delenv(REMOVED, raising=False)

    TpuPlatform.check_and_update_config(vllm_config)


def test_qwen35_dcp_cli_defaults_set_moe_v2_and_disable_reassociate(monkeypatch):
    import sys

    from vllm_torchtpu import env_override

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vllm",
            "serve",
            "Qwen/Qwen3.5-397B-A17B-FP8",
            "--decode-context-parallel-size",
            "4",
        ],
    )
    monkeypatch.delenv(
        "TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS", raising=False
    )
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_IMPL", raising=False)
    env_override._apply_qwen35_serve_cli_defaults()
    assert (
        env_override.os.environ["TORCH_TPU_INTERNAL_ENABLE_REASSOCIATE_NORM_WEIGHTS"]
        == "0"
    )
    assert env_override.os.environ["MOE_FUSED_EP_KERNEL_IMPL"] == "v2"


def test_qwen35_serve_cli_defaults_ignores_vllm_bench_serve(monkeypatch):
    import sys

    from vllm_torchtpu import env_override

    orig_argv = [
        "vllm",
        "bench",
        "serve",
        "--model",
        "Qwen/Qwen3.5-397B-A17B-FP8",
    ]
    monkeypatch.setattr(sys, "argv", list(orig_argv))
    env_override._apply_qwen35_serve_cli_defaults()
    assert sys.argv == orig_argv


def test_restored_qwen35_sampling_and_draft_defaults(monkeypatch):
    import vllm_torchtpu.envs as envs

    for key in (
        "TPU_GREEDY_TOP_TOKENS",
        "TPU_SAMPLER_TOPK_CANDIDATES",
        "VLLM_TPU_FUSED_DRAFT_LOOP",
        "VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP",
    ):
        monkeypatch.delenv(key, raising=False)

    assert envs.environment_variables["TPU_GREEDY_TOP_TOKENS"]() is True
    assert envs.environment_variables["TPU_SAMPLER_TOPK_CANDIDATES"]() == 64
    assert envs.environment_variables["VLLM_TPU_FUSED_DRAFT_LOOP"]() is True
    assert envs.environment_variables["VLLM_TPU_DISABLE_EAGLE_BLOCK_DROP"]() is True
