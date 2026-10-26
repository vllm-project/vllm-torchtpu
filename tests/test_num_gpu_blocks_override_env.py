# SPDX-License-Identifier: Apache-2.0
"""TPU_NUM_GPU_BLOCKS_OVERRIDE pins cache_config.num_gpu_blocks_override."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms import tpu_platform


@pytest.mark.parametrize(
    "env, cli, want",
    [
        (None, None, None),  # off: pool sized from gpu_memory_utilization
        ("0", None, None),
        ("4288", None, 4288),
        ("4288", 4288, 4288),
        ("4288", 1000, 1000),  # the CLI flag wins
        (None, 1000, 1000),
    ],
)
def test_num_gpu_blocks_override_env(monkeypatch, env, cli, want):
    if env is None:
        monkeypatch.delenv("TPU_NUM_GPU_BLOCKS_OVERRIDE", raising=False)
    else:
        monkeypatch.setenv("TPU_NUM_GPU_BLOCKS_OVERRIDE", env)
    cache_config = SimpleNamespace(num_gpu_blocks_override=cli)
    tpu_platform._apply_num_gpu_blocks_override_env(cache_config)
    tpu_platform._apply_num_gpu_blocks_override_env(cache_config)  # idempotent
    assert cache_config.num_gpu_blocks_override == want
