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
"""TPU L0 tests for vLLM-Omni composable parallel strategy resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from vllm_omni.config.composable_parallel import (
    AxisTranslationError,
    StrategyApplyError,
)
from vllm_omni.config.composable_parallel.strategy_loader import load_strategy_specs
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


@dataclass(frozen=True)
class _StrategyCase:
    case_id: str
    pipeline_key: str
    model_manifest_name: str
    model_manifest_body: dict[str, Any]
    deploy_yaml: str
    strategy_yaml: str
    stage_overrides: dict[str, dict[str, Any]] | None
    expected_roles: tuple[str, ...]
    expected_stage_parallel: dict[str, tuple[int, int, int, bool, int]]
    expected_lb_policy: str | None


CASES = (
    _StrategyCase(
        case_id="qwen2_5_omni_wired_axes",
        pipeline_key="qwen2_5_omni",
        model_manifest_name="config.json",
        model_manifest_body={"model_type": "qwen2_5_omni"},
        deploy_yaml=(
            "async_chunk: false\n"
            "stages:\n"
            "  - stage_id: 0\n"
            '    devices: "0,1,2,3"\n'
            "  - stage_id: 1\n"
            '    devices: "4,5"\n'
            "  - stage_id: 2\n"
            '    devices: "6,7"\n'
        ),
        strategy_yaml=(
            "strategies:\n"
            "  thinker:\n"
            "    - axis: tp\n"
            "      size: 2\n"
            "    - axis: dp\n"
            "      size: 2\n"
            "      routing: round_robin\n"
            "    - axis: ep\n"
            "      size: 4\n"
            "  talker:\n"
            "    - axis: pp\n"
            "      size: 2\n"
            "  code2wav:\n"
            "    - axis: stage_replica\n"
            "      size: 2\n"
            "      routing: round_robin\n"
        ),
        stage_overrides=None,
        expected_roles=("thinker", "talker", "code2wav"),
        expected_stage_parallel={
            "thinker": (2, 2, 1, True, 1),
            "talker": (1, 1, 2, False, 1),
            "code2wav": (1, 1, 1, False, 2),
        },
        expected_lb_policy="round-robin",
    ),
    _StrategyCase(
        case_id="wan2_2_eg_diffusion_stages",
        pipeline_key="wan2_2_eg",
        model_manifest_name="model_index.json",
        model_manifest_body={"_class_name": "WanPipeline"},
        deploy_yaml=(
            "pipeline: wan2_2_eg\n"
            "async_chunk: false\n"
            "connectors:\n"
            "  wan_encode_connector:\n"
            "    name: SharedMemoryConnector\n"
            "stages:\n"
            "  - stage_id: 0\n"
            '    devices: "0,1"\n'
            "    output_connectors:\n"
            "      to_stage_1: wan_encode_connector\n"
            "  - stage_id: 1\n"
            '    devices: "2,3,4,5"\n'
            "    input_connectors:\n"
            "      from_stage_0: wan_encode_connector\n"
        ),
        strategy_yaml=(
            "strategies:\n"
            "  text_encode:\n"
            "    - axis: stage_replica\n"
            "      size: 2\n"
            "      routing: least_queue\n"
            "  dit:\n"
            "    - axis: tp\n"
            "      size: 2\n"
            "    - axis: dp\n"
            "      size: 2\n"
            "      routing: round_robin\n"
        ),
        stage_overrides=None,
        expected_roles=("text_encode", "dit"),
        expected_stage_parallel={
            "text_encode": (1, 1, 1, False, 2),
            "dit": (2, 2, 1, False, 1),
        },
        expected_lb_policy="least-queue-length",
    ),
    _StrategyCase(
        case_id="qwen2_5_omni_cli_precedence",
        pipeline_key="qwen2_5_omni",
        model_manifest_name="config.json",
        model_manifest_body={"model_type": "qwen2_5_omni"},
        deploy_yaml=(
            "async_chunk: false\n"
            "stages:\n"
            "  - stage_id: 0\n"
            '    devices: "0,1,2,3"\n'
            "  - stage_id: 1\n"
            '    devices: "4"\n'
            "  - stage_id: 2\n"
            '    devices: "5"\n'
        ),
        strategy_yaml=(
            "strategies:\n"
            "  thinker:\n"
            "    - axis: tp\n"
            "      size: 2\n"
            "    - axis: dp\n"
            "      size: 2\n"
            "  talker:\n"
            "    - axis: stage_replica\n"
            "      size: 2\n"
            "      routing: random\n"
        ),
        stage_overrides={
            "0": {"tensor_parallel_size": 4, "data_parallel_size": 1},
            "1": {"num_replicas": 3},
        },
        expected_roles=("thinker", "talker"),
        expected_stage_parallel={
            "thinker": (4, 1, 1, False, 1),
            "talker": (1, 1, 1, False, 3),
            "code2wav": (1, 1, 1, False, 1),
        },
        expected_lb_policy="random",
    ),
)


def _snapshot_stage_parallel(
    stages: tuple[Any, ...],
) -> dict[str, tuple[int, int, int, bool, int]]:
    return {
        stage.model_stage: (
            stage.parallel_config.tensor_parallel_size,
            stage.parallel_config.data_parallel_size,
            stage.parallel_config.pipeline_parallel_size,
            stage.parallel_config.enable_expert_parallel,
            stage.runtime_config.num_replicas,
        )
        for stage in stages
    }


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
def test_composable_parallel_strategy(case: _StrategyCase, tmp_path: Path) -> None:
    """Composable Parallel Strategy: overlay --strategy-config onto pipeline stages.

    - TPU path: resolve strategy YAML files via ``resolve_omni_config`` with
      ``OmniTpuPlatform`` active across LLM and diffusion multi-stage pipelines.
    - Setup check: assert ``current_omni_platform`` is ``OmniTpuPlatform`` with
      ``TPU_VISIBLE_CHIPS``, and ``load_strategy_specs`` matches expected roles.
    - Outcome check: assert resolved per-stage ``(tp, dp, pp, ep, num_replicas)``
      and ``omni_lb_policy``, plus ``StrategyApplyError`` on device-count conflict
      and ``AxisTranslationError`` on reserved diffusion axes.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"{case.case_id}: expected OmniTpuPlatform, got {type(current_omni_platform)}"
    )
    assert current_omni_platform.device_control_env_var == "TPU_VISIBLE_CHIPS", (
        f"{case.case_id}: unexpected device env var "
        f"{current_omni_platform.device_control_env_var!r}"
    )

    model_dir = tmp_path / case.pipeline_key
    model_dir.mkdir()
    (model_dir / case.model_manifest_name).write_text(
        json.dumps(case.model_manifest_body), encoding="utf-8"
    )
    deploy_path = tmp_path / "deploy.yaml"
    deploy_path.write_text(case.deploy_yaml, encoding="utf-8")
    strategy_path = tmp_path / "strategy.yaml"
    strategy_path.write_text(case.strategy_yaml, encoding="utf-8")

    parsed_specs = load_strategy_specs(str(strategy_path))
    assert tuple(parsed_specs) == case.expected_roles, (
        f"{case.case_id}: parsed roles {tuple(parsed_specs)} != {case.expected_roles}"
    )

    resolved = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=str(deploy_path),
        cli_overrides=None,
        stage_overrides=case.stage_overrides,
        strategy_config_path=str(strategy_path),
    )
    assert resolved.pipeline_config is not None, (
        f"{case.case_id}: expected registered PipelineConfig, got None"
    )
    assert resolved.pipeline_config.model_type == case.pipeline_key, (
        f"{case.case_id}: pipeline model_type {resolved.pipeline_config.model_type!r} "
        f"!= {case.pipeline_key!r}"
    )

    actual_parallel = _snapshot_stage_parallel(resolved.stage_configs)
    assert actual_parallel == case.expected_stage_parallel, (
        f"{case.case_id}: stage parallel snapshot {actual_parallel} != "
        f"{case.expected_stage_parallel}"
    )
    assert resolved.omni_lb_policy == case.expected_lb_policy, (
        f"{case.case_id}: omni_lb_policy {resolved.omni_lb_policy!r} != "
        f"{case.expected_lb_policy!r}"
    )

    first_role = case.expected_roles[0]
    conflict_path = tmp_path / "conflict_strategy.yaml"
    conflict_path.write_text(
        f"strategies:\n  {first_role}:\n    - axis: tp\n      size: 8\n",
        encoding="utf-8",
    )
    with pytest.raises(StrategyApplyError):
        resolve_omni_config(
            str(model_dir),
            trust_remote_code=False,
            deploy_config_path=str(deploy_path),
            cli_overrides=None,
            stage_overrides=None,
            strategy_config_path=str(conflict_path),
        )

    reserved_path = tmp_path / "reserved_strategy.yaml"
    reserved_path.write_text(
        f"strategies:\n  {first_role}:\n    - axis: sp_ulysses\n      size: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(AxisTranslationError):
        resolve_omni_config(
            str(model_dir),
            trust_remote_code=False,
            deploy_config_path=str(deploy_path),
            cli_overrides=None,
            stage_overrides=None,
            strategy_config_path=str(reserved_path),
        )
