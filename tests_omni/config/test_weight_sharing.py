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
"""TPU tests for cross-stage rank-paired weight sharing."""

from __future__ import annotations

import pytest
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


@pytest.mark.xfail(
    strict=True,
    raises=ModuleNotFoundError,
    reason="not in vllm-omni 0.32.0rc1",
)
def test_cross_stage_rank_paired_weight_sharing() -> None:
    """Cross-Stage Rank-Paired Weight Sharing.

    - TPU path: import ``WeightSharingConfig``, ``parse_weight_sharing``, and
      ``TorchIpcBackend`` / ``SharingPlan``, parse a provider-consumer sharing
      policy, and attempt to build the rank-paired IPC sharing backend on TPU.
    - Setup check: ``current_omni_platform.device_type == "tpu"`` and
      ``parse_weight_sharing`` validates distinct stages and memory ratio.
    - Outcome check: ``TorchIpcBackend`` and ``SharingPlan`` validate TPU
      memory budgets and device IPC support.
    """
    from vllm_omni.config.weight_sharing import (
        WeightSharingConfig,
        parse_weight_sharing,
    )
    from vllm_omni.distributed.weight_sharing.backend import TorchIpcBackend
    from vllm_omni.distributed.weight_sharing.runtime import (
        DeviceBudget,
        SharingPlan,
    )

    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    assert parse_weight_sharing(None) is None, (
        "Expected parse_weight_sharing(None) to disable weight sharing"
    )
    cfg = parse_weight_sharing(
        {
            "provider_stage": 0,
            "consumer_stage": 1,
            "gpu_memory_utilization": 0.9,
            "timeout": 120.0,
        }
    )
    assert isinstance(cfg, WeightSharingConfig), (
        f"Expected WeightSharingConfig, got {type(cfg).__name__}"
    )
    assert cfg.provider_stage == 0, (
        f"Expected provider_stage=0, got {cfg.provider_stage}"
    )
    assert cfg.consumer_stage == 1, (
        f"Expected consumer_stage=1, got {cfg.consumer_stage}"
    )
    with pytest.raises(ValueError, match="distinct stages"):
        parse_weight_sharing({"provider_stage": 0, "consumer_stage": 0})

    total_hbm = current_omni_platform.get_device_total_memory(0)
    capacity = int(total_hbm * cfg.gpu_memory_utilization)
    budget = DeviceBudget(
        capacity_bytes=capacity,
        reserve_bytes=total_hbm - capacity,
        kv_cache_bytes=capacity // 4,
        consumer_kv_cache_bytes=capacity // 4,
    )
    budget.validate()

    plan = SharingPlan(
        provider_stage=cfg.provider_stage,
        consumer_stage=cfg.consumer_stage,
        device_uuids=("TPU-0",),
        budgets=(budget,),
        model_id="tiny-hunyuan-image3",
        checkpoint_revision="rev-0",
        layout="tp1",
        timeout=cfg.timeout,
    )
    assert plan.provider_stage == 0, (
        f"Unexpected plan provider_stage: {plan.provider_stage}"
    )
    backend = TorchIpcBackend()
    assert backend is not None, "Expected TorchIpcBackend instance"
