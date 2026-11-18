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
"""Tests for vllm-omni pipeline selection overrides on TPU."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from transformers import Qwen3OmniMoeConfig
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import StageExecutionType
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


def _write_wan_dir(root: Path) -> str:
    model_dir = root / "wan_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "WanPipeline"}),
        encoding="utf-8",
    )
    return str(model_dir)


def _write_qwen3_omni_dir(root: Path) -> str:
    model_dir = root / "qwen3_omni_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = Qwen3OmniMoeConfig(enable_audio_output=True)
    cfg.save_pretrained(model_dir)
    return str(model_dir)


def _write_flux_dir(root: Path) -> str:
    model_dir = root / "flux_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}),
        encoding="utf-8",
    )
    return str(model_dir)


def _write_deploy_yaml(root: Path, pipeline_key: str) -> str:
    deploy_path = root / f"deploy_{pipeline_key}.yaml"
    deploy_path.write_text(f"pipeline: {pipeline_key}\n", encoding="utf-8")
    return str(deploy_path)


@pytest.mark.parametrize(
    ("write_model_dir", "default_key", "override_key"),
    [
        pytest.param(_write_wan_dir, "wan2_2_ti2v", "wan2_2_eg", id="wan_eg"),
        pytest.param(
            _write_qwen3_omni_dir,
            "qwen3_omni_moe",
            "qwen3_omni_moe_thinker_only",
            id="qwen3_omni_thinker_only",
        ),
    ],
)
def test_pipeline_selection_overrides(
    tmp_path: Path,
    write_model_dir: Callable[[Path], str],
    default_key: str,
    override_key: str,
) -> None:
    """Pipeline Selection Overrides: deploy YAML pipeline key.

    - TPU path: with OmniTpuPlatform active, call resolve_omni_config on a minimal
      local model dir with and without a deploy YAML pipeline override.
    - Setup check: assert current_omni_platform.device_name == "tpu" and the
      baseline resolve_omni_config auto-detects the default topology.
    - Outcome check: assert the override selects OMNI_PIPELINES[key] stage
      count/types/input sources.
    """
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    assert current_omni_platform.device_name == "tpu", (
        f"Expected active platform 'tpu', got {current_omni_platform.device_name!r}"
    )

    model_dir = write_model_dir(tmp_path)
    baseline = resolve_omni_config(
        model_dir,
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    expected_default = resolve_pipeline_config(
        default_key, StageConfigFactory.get_hf_config(model_dir, False)
    )
    assert expected_default is not None, f"Default pipeline {default_key!r} missing"
    assert len(baseline.stage_configs) == len(expected_default.stages), (
        f"Baseline stage count {len(baseline.stage_configs)} != "
        f"{len(expected_default.stages)} for {default_key}"
    )

    deploy_path = _write_deploy_yaml(tmp_path, override_key)
    overridden = resolve_omni_config(
        model_dir,
        trust_remote_code=False,
        deploy_config_path=deploy_path,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    expected_override = OMNI_PIPELINES[override_key]
    assert not callable(expected_override), (
        f"Expected concrete PipelineConfig for {override_key!r}"
    )
    assert len(overridden.stage_configs) == len(expected_override.stages), (
        f"Override {override_key} stage count {len(overridden.stage_configs)} != "
        f"{len(expected_override.stages)}"
    )
    expected_stage_types = [
        "diffusion" if s.execution_type == StageExecutionType.DIFFUSION else "llm"
        for s in expected_override.stages
    ]
    assert [s.stage_type.value for s in overridden.stage_configs] == (
        expected_stage_types
    ), f"Override {override_key} stage types mismatch"
    assert [tuple(s.input_sources) for s in overridden.stage_configs] == [
        tuple(s.input_sources) for s in expected_override.stages
    ], f"Override {override_key} input_sources mismatch"


def test_pipeline_selection_overrides_unknown_key(tmp_path: Path) -> None:
    """Pipeline Selection Overrides: unknown deploy YAML pipeline key.

    - TPU path: with OmniTpuPlatform active, call resolve_omni_config on a minimal
      local Wan model dir with a deploy YAML naming an unregistered pipeline.
    - Setup check: assert current_omni_platform.device_name == "tpu" and the
      baseline resolve_omni_config auto-detects the wan2_2_ti2v topology.
    - Outcome check: the unknown pipeline key raises KeyError.
    """
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    assert current_omni_platform.device_name == "tpu", (
        f"Expected active platform 'tpu', got {current_omni_platform.device_name!r}"
    )

    model_dir = _write_wan_dir(tmp_path)
    baseline = resolve_omni_config(
        model_dir,
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    expected_default = resolve_pipeline_config(
        "wan2_2_ti2v", StageConfigFactory.get_hf_config(model_dir, False)
    )
    assert expected_default is not None, (
        f"Default pipeline 'wan2_2_ti2v' resolved to {expected_default!r}"
    )
    assert len(baseline.stage_configs) == len(expected_default.stages), (
        f"Baseline stage count {len(baseline.stage_configs)} != "
        f"{len(expected_default.stages)} for wan2_2_ti2v"
    )

    deploy_path = _write_deploy_yaml(tmp_path, "nonexistent_pipeline_key")
    with pytest.raises(KeyError):
        resolve_omni_config(
            model_dir,
            trust_remote_code=False,
            deploy_config_path=deploy_path,
            cli_overrides=None,
            stage_overrides=None,
            strategy_config_path=None,
        )


def test_pipeline_selection_overrides_cli_model_class(tmp_path: Path) -> None:
    """Pipeline Selection Overrides: CLI --model-class-name.

    - TPU path: with OmniTpuPlatform active, call resolve_omni_config on a minimal
      local Flux model dir with and without a model_class_name CLI override.
    - Setup check: assert current_omni_platform.device_name == "tpu" and the
      baseline resolve_omni_config auto-detects FluxPipeline.
    - Outcome check: the override sets
      stage_configs[0].diffusion_config.model_class_name to QwenImagePipeline.
    """
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    assert current_omni_platform.device_name == "tpu", (
        f"Expected active platform 'tpu', got {current_omni_platform.device_name!r}"
    )

    model_dir = _write_flux_dir(tmp_path)
    baseline = resolve_omni_config(
        model_dir,
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    baseline_cls = baseline.stage_configs[0].diffusion_config.model_class_name
    assert baseline_cls == "FluxPipeline", (
        f"Baseline model_class_name {baseline_cls!r} != expected 'FluxPipeline'"
    )
    overridden = resolve_omni_config(
        model_dir,
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides={"model_class_name": "QwenImagePipeline"},
        stage_overrides=None,
        strategy_config_path=None,
    )
    overridden_cls = overridden.stage_configs[0].diffusion_config.model_class_name
    assert overridden_cls == "QwenImagePipeline", (
        f"Overridden model_class_name {overridden_cls!r} != target 'QwenImagePipeline'"
    )
