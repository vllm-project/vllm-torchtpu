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
"""Tests for out-of-tree model and pipeline registration on TPU."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch.nn as nn
from transformers import PretrainedConfig
from vllm_omni.config.config_factory import StageConfigFactory
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, register_pipeline
from vllm_omni.config.resolver import resolve_omni_config
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.diffusion import registry as diffusion_registry
from vllm_omni.diffusion.registry import (
    DiffusionModelRegistry,
    register_diffusion_model,
)
from vllm_omni.model_executor.models import OmniModelRegistry
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


class _OutOfTreeOmniModel(nn.Module):
    """Minimal out-of-tree LLM/omni stage model for registry resolution."""


class _OutOfTreeDiffusionModel(nn.Module):
    """Minimal out-of-tree diffusion pipeline for registry resolution."""


def _register_pipeline_config(pipeline: PipelineConfig) -> None:
    register_pipeline(pipeline)


def _register_resolver_func(pipeline: PipelineConfig) -> None:
    def _custom_resolver(_hf_config: PretrainedConfig | None) -> PipelineConfig:
        return pipeline

    register_pipeline(_custom_resolver, pipeline.model_type)


@pytest.mark.parametrize(
    "registration_kind,register",
    [
        ("pipeline_config", _register_pipeline_config),
        ("resolver_func", _register_resolver_func),
    ],
    ids=["pipeline_config", "resolver_func"],
)
def test_out_of_tree_model_and_pipeline_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    registration_kind: str,
    register: Callable[[PipelineConfig], None],
) -> None:
    """Out-of-Tree Model & Pipeline Registration

    - TPU path: with OmniTpuPlatform active and monkeypatch isolating global
      registry dicts, register custom pipelines via register_pipeline and custom
      models via OmniModelRegistry.register_model and register_diffusion_model.
    - Setup check: assert current_omni_platform.device_name == "tpu" and custom
      keys are absent prior to registration and present afterward.
    - Outcome check: assert resolve_omni_config resolves the registered
      PipelineConfig stages and OmniModelRegistry / DiffusionModelRegistry load
      the registered custom nn.Module classes.
    """
    assert current_omni_platform.device_name == "tpu", (
        f"Expected active platform 'tpu', got {current_omni_platform.device_name!r}"
    )

    monkeypatch.setattr(
        "vllm_omni.config.pipeline_registry.OMNI_PIPELINES", dict(OMNI_PIPELINES)
    )
    monkeypatch.setattr(
        "vllm_omni.config.config_factory.OMNI_PIPELINES",
        __import__(
            "vllm_omni.config.pipeline_registry", fromlist=["OMNI_PIPELINES"]
        ).OMNI_PIPELINES,
    )
    monkeypatch.setattr(OmniModelRegistry, "models", dict(OmniModelRegistry.models))
    monkeypatch.setattr(
        DiffusionModelRegistry, "models", dict(DiffusionModelRegistry.models)
    )
    monkeypatch.setattr(
        diffusion_registry,
        "_DIFFUSION_MODELS",
        dict(diffusion_registry._DIFFUSION_MODELS),
    )

    pipeline_key = f"oot_pipeline_{registration_kind}"
    omni_arch = f"OutOfTreeOmniArch_{registration_kind}"
    diff_arch = f"OutOfTreeDiffArch_{registration_kind}"

    from vllm_omni.config import pipeline_registry as pr_mod

    assert pipeline_key not in pr_mod.OMNI_PIPELINES, (
        f"{pipeline_key!r} unexpectedly present before registration"
    )
    assert omni_arch not in OmniModelRegistry.models, (
        f"{omni_arch!r} unexpectedly present in OmniModelRegistry"
    )
    assert diff_arch not in DiffusionModelRegistry.models, (
        f"{diff_arch!r} unexpectedly present in DiffusionModelRegistry"
    )
    assert OmniModelRegistry._try_load_model_cls(omni_arch) is None, (
        f"Expected pre-registration miss for {omni_arch!r}"
    )
    assert DiffusionModelRegistry._try_load_model_cls(diff_arch) is None, (
        f"Expected pre-registration miss for {diff_arch!r}"
    )

    expected_pipeline = PipelineConfig(
        model_type=pipeline_key,
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="thinker",
                execution_type=StageExecutionType.LLM_AR,
                final_output=False,
            ),
            StagePipelineConfig(
                stage_id=1,
                model_stage="dit",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(0,),
                final_output=True,
                final_output_type="image",
            ),
        ),
    )

    register(expected_pipeline)

    OmniModelRegistry.register_model(omni_arch, _OutOfTreeOmniModel)
    register_diffusion_model(
        model_arch=diff_arch,
        module_name=__name__,
        class_name="_OutOfTreeDiffusionModel",
    )

    assert pipeline_key in pr_mod.OMNI_PIPELINES, (
        f"{pipeline_key!r} missing from OMNI_PIPELINES after registration"
    )
    assert omni_arch in OmniModelRegistry.models, (
        f"{omni_arch!r} missing from OmniModelRegistry after registration"
    )
    assert diff_arch in DiffusionModelRegistry.models, (
        f"{diff_arch!r} missing from DiffusionModelRegistry after registration"
    )

    model_dir = tmp_path / pipeline_key
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": pipeline_key, "architectures": [omni_arch]}),
        encoding="utf-8",
    )
    StageConfigFactory.get_hf_config.cache_clear()
    StageConfigFactory.try_infer_model_type.cache_clear()

    resolution = resolve_omni_config(
        str(model_dir),
        trust_remote_code=False,
        deploy_config_path=None,
        cli_overrides=None,
        stage_overrides=None,
        strategy_config_path=None,
    )
    assert resolution.pipeline_config == expected_pipeline, (
        f"Resolved pipeline_config {resolution.pipeline_config} != {expected_pipeline}"
    )
    assert len(resolution.stage_configs) == len(expected_pipeline.stages), (
        f"Resolved stage count {len(resolution.stage_configs)} != "
        f"{len(expected_pipeline.stages)}"
    )
    expected_stage_types = [
        "diffusion" if s.execution_type == StageExecutionType.DIFFUSION else "llm"
        for s in expected_pipeline.stages
    ]
    assert [s.stage_type.value for s in resolution.stage_configs] == (
        expected_stage_types
    ), "Resolved stage types do not match registered PipelineConfig"

    loaded_omni = OmniModelRegistry._try_load_model_cls(omni_arch)
    loaded_diff = DiffusionModelRegistry._try_load_model_cls(diff_arch)
    assert loaded_omni is _OutOfTreeOmniModel, (
        f"OmniModelRegistry resolved {loaded_omni} != {_OutOfTreeOmniModel}"
    )
    assert loaded_diff is _OutOfTreeDiffusionModel, (
        f"DiffusionModelRegistry resolved {loaded_diff} != {_OutOfTreeDiffusionModel}"
    )
