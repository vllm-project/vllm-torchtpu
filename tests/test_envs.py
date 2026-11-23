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
"""Removed env vars (``REMOVED_ENV_VARS``): setting one must fail startup.

None is aliased to its replacement. USE_MOE_EP_KERNEL, for example, is not
an alias of USE_MOE_FUSED_EP_KERNEL, which would enable the kernel under the
recipes that still set the old name and change results they have already
recorded.
"""

import re
from unittest.mock import MagicMock, patch

import pytest
from vllm.config import CacheConfig, VllmConfig
from vllm.config.aux_output import AuxOutputConfig

from vllm_torchtpu.env_validation import REMOVED_ENV_VARS
from vllm_torchtpu.platforms.tpu_platform import TpuPlatform


@pytest.fixture(autouse=True)
def _no_removed_env_vars(monkeypatch):
    """Every test starts with no removed name set, whatever the shell sets."""
    for name in REMOVED_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def vllm_config():
    """A plain single-host config: nothing else in the check may reject it."""
    config = MagicMock(spec=VllmConfig)
    config.additional_config = {}
    config.model_config.is_hybrid = False
    config.cache_config = CacheConfig(enable_prefix_caching=False)
    config.compilation_config.compile_sizes = [16, 32]
    # Budget within the largest compile size, which the platform requires.
    config.scheduler_config.max_num_scheduled_tokens = None
    config.scheduler_config.max_num_batched_tokens = 32
    config.speculative_config = None
    config.lora_config = None
    config.aux_output_config = AuxOutputConfig()
    config.parallel_config.nnodes = 1
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.kv_transfer_config = None
    return config


@patch("vllm_torchtpu.patch_registry.apply")
@patch("vllm_torchtpu.platforms.tpu_platform.TpuPlatform._prepare_singlehost_tpu_env")
def test_startup_accepts_no_removed_name_set(
    mock_prepare_env, mock_apply_patches, monkeypatch, vllm_config
):
    TpuPlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize(
    "key", ["tpu_moe_silu_bf16", "tpu_moe_topk_weights_float32", "tpu_rope_in_graph"]
)
def test_startup_rejects_non_bool_additional_config(monkeypatch, vllm_config, key):
    vllm_config.additional_config = {key: "true"}

    with pytest.raises(ValueError, match=key):
        TpuPlatform.check_and_update_config(vllm_config)


@pytest.mark.parametrize("value", ["1", "0", ""])
@pytest.mark.parametrize("name", list(REMOVED_ENV_VARS))
def test_startup_rejects_every_removed_name(monkeypatch, vllm_config, name, value):
    """Including `0` and the empty string, and the TPU_PP_* names without a
    pipeline: what is wrong is naming the variable."""
    monkeypatch.setenv(name, value)
    expected = f"{name} has been removed; {REMOVED_ENV_VARS[name]}"

    with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
        TpuPlatform.check_and_update_config(vllm_config)
