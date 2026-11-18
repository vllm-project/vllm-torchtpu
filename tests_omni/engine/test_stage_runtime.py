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
"""TPU L0 tests for NVIDIA MPS stage colocation."""

from __future__ import annotations

from pathlib import Path

import pytest
import vllm_omni
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.engine.stage_runtime import StageRuntime
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason="NVIDIA MPS requires CUDA; current_omni_platform.is_cuda() is False on TPU",
)
def test_nvidia_mps_stage_colocation() -> None:
    """NVIDIA MPS Stage Colocation.

    - TPU path: with ``OmniTpuPlatform`` active, load a colocated
      ``qwen3_tts`` deploy YAML setting ``cuda_mps: true`` and ``devices: "0"``
      and invoke ``StageRuntime._mps_environment``.
    - Setup check: ``current_omni_platform.device_type == "tpu"``,
      ``current_omni_platform.is_cuda()`` is False, baseline ``cuda_mps: false``
      resolves ``runtime_config.cuda_mps is False`` with ``{}``, and
      colocated ``cuda_mps: true`` resolves ``runtime_config.cuda_mps is True``.
    - Outcome check: ``StageRuntime._mps_environment("0", stage.runtime_config)``
      raises ``ValueError`` on TPU and would contain ``CUDA_MPS_PIPE_DIRECTORY``
      if it succeeded.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )
    assert current_omni_platform.is_cuda() is False, (
        "Expected current_omni_platform.is_cuda() to be False on TPU"
    )

    deploy_dir = Path(vllm_omni.__file__).resolve().parent / "deploy"
    pipeline = resolve_pipeline_config("qwen3_tts")

    cfg_off = VllmOmniConfig.from_pipeline_config(
        pipeline,
        deploy_config_path=str(deploy_dir / "qwen3_tts.yaml"),
        cli_overrides={"model": "fake-qwen3-tts"},
    )
    cfg_on = VllmOmniConfig.from_pipeline_config(
        pipeline,
        deploy_config_path=str(
            deploy_dir / "qwen3_tts_high_concurrency_mrv2_single_gpu.yaml"
        ),
        cli_overrides={"model": "fake-qwen3-tts"},
    )

    stage_off = cfg_off.stage_configs[0]
    stage_on = cfg_on.stage_configs[0]
    assert stage_off.runtime_config.cuda_mps is False, (
        f"Expected baseline cuda_mps False, got {stage_off.runtime_config.cuda_mps}"
    )
    assert stage_on.runtime_config.cuda_mps is True, (
        f"Expected colocated cuda_mps True, got {stage_on.runtime_config.cuda_mps}"
    )

    runtime = object.__new__(StageRuntime)
    runtime._mps_servers = {}

    assert runtime._mps_environment("0", stage_off.runtime_config) == {}, (
        "Expected empty MPS env dict when cuda_mps is False"
    )

    mps_env = runtime._mps_environment("0", stage_on.runtime_config)
    assert "CUDA_MPS_PIPE_DIRECTORY" in mps_env, (
        f"Expected CUDA_MPS_PIPE_DIRECTORY in MPS env, got {mps_env}"
    )
