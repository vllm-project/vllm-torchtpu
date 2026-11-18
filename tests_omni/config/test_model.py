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
"""TPU L0 config tests for startup task_type selection."""

from __future__ import annotations

from pathlib import Path

import pytest
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm_omni.config.model import OmniModelConfig
from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.pipeline_registry import resolve_pipeline_config
from vllm_omni.entrypoints.cli.serve import OmniServeCommand
from vllm_omni.model_executor.models.minimax_h3.checkpoint import (
    resolve_minimax_h3_encoder_model_root,
    resolve_minimax_h3_partition,
)
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]


@pytest.mark.parametrize("valid_task", [None, "CustomVoice", "VoiceDesign", "Base"])
def test_startup_task_type_selection(tmp_path: Path, valid_task: str | None) -> None:
    """Startup Task Type Selection.

    - TPU path: with ``OmniTpuPlatform`` active, parse ``--task-type`` via
      ``OmniServeCommand``, resolve ``VllmOmniConfig`` for ``qwen3_tts`` and
      ``minimax_h3_disaggregated``, and test ``_validate_startup_task_type``
      and MiniMax-H3 partition/path resolvers.
    - Setup check: ``current_omni_platform.device_type == "tpu"``, parsed
      ``args.task_type`` matches CLI flag, and resolved stage configs populate
      ``model_config.task_type`` and ``diffusion_config.task_type``.
    - Outcome check: Qwen3-TTS accepts ``CustomVoice``, ``VoiceDesign``, and
      ``Base`` and rejects invalid values with ``ValueError``; MiniMax-H3
      resolves ``t2va``/``fl2va`` to ``FL2VA`` and ``ref2va`` to ``Ref2VA``
      and rejects invalid values with ``ValueError``.
    """
    assert current_omni_platform.device_type == "tpu", (
        f"Expected TPU platform, got {current_omni_platform.device_type}"
    )

    parser = FlexibleArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")
    OmniServeCommand().subparser_init(subparsers)
    args = parser.parse_args(
        ["serve", "fake-qwen3-tts", "--omni", "--task-type", "CustomVoice"]
    )
    assert args.task_type == "CustomVoice", (
        f"Expected args.task_type == 'CustomVoice', got {args.task_type!r}"
    )

    qwen_pipeline = resolve_pipeline_config("qwen3_tts")
    qwen_cfg = VllmOmniConfig.from_pipeline_config(
        qwen_pipeline,
        cli_overrides={"model": "fake-qwen3-tts", "task_type": "CustomVoice"},
    )
    assert qwen_cfg.stage_configs[0].model_config.task_type == "CustomVoice", (
        f"Expected qwen3_tts stage 0 task_type 'CustomVoice', "
        f"got {qwen_cfg.stage_configs[0].model_config.task_type!r}"
    )

    qwen_default_cfg = VllmOmniConfig.from_pipeline_config(
        qwen_pipeline,
        cli_overrides={"model": "fake-qwen3-tts"},
    )
    assert qwen_default_cfg.stage_configs[0].model_config.task_type is None, (
        f"Expected default qwen3_tts task_type None, "
        f"got {qwen_default_cfg.stage_configs[0].model_config.task_type!r}"
    )

    minimax_pipeline = resolve_pipeline_config("minimax_h3_disaggregated")
    minimax_cfg = VllmOmniConfig.from_pipeline_config(
        minimax_pipeline,
        cli_overrides={"model": "fake-minimax-h3", "task_type": "t2va"},
    )
    assert minimax_cfg.stage_configs[0].model_config.task_type == "t2va", (
        f"Expected minimax_h3 stage 0 task_type 't2va', "
        f"got {minimax_cfg.stage_configs[0].model_config.task_type!r}"
    )
    assert minimax_cfg.stage_configs[1].diffusion_config.task_type == "t2va", (
        f"Expected minimax_h3 stage 1 diffusion task_type 't2va', "
        f"got {minimax_cfg.stage_configs[1].diffusion_config.task_type!r}"
    )

    model_cfg = object.__new__(OmniModelConfig)
    model_cfg.model_arch = "Qwen3TTSTalkerForConditionalGenerationARVLLM"
    model_cfg.task_type = valid_task
    model_cfg._validate_startup_task_type()

    model_cfg.task_type = "InvalidTask"
    with pytest.raises(ValueError, match="Qwen3-TTS --task-type must be one of"):
        model_cfg._validate_startup_task_type()

    model_root = tmp_path / "minimax_h3"
    (model_root / "FL2VA" / "text_encoder").mkdir(parents=True)
    (model_root / "Ref2VA" / "text_encoder").mkdir(parents=True)

    assert (
        resolve_minimax_h3_partition(str(model_root), None, auto_partition="fl2va")
        == "fl2va"
    ), "Expected default None task_type to resolve to auto_partition 'fl2va'"
    assert (
        resolve_minimax_h3_partition(str(model_root), "t2va", auto_partition="combined")
        == "fl2va"
    ), "Expected t2va to resolve to 'fl2va'"
    assert (
        resolve_minimax_h3_partition(
            str(model_root), "fl2va", auto_partition="combined"
        )
        == "fl2va"
    ), "Expected fl2va to resolve to 'fl2va'"
    assert (
        resolve_minimax_h3_partition(
            str(model_root), "ref2va", auto_partition="combined"
        )
        == "ref2va"
    ), "Expected ref2va to resolve to 'ref2va'"

    assert resolve_minimax_h3_encoder_model_root(str(model_root), None, "t2va") == str(
        model_root / "FL2VA" / "text_encoder"
    ), "Expected t2va encoder root under FL2VA/text_encoder"
    assert resolve_minimax_h3_encoder_model_root(
        str(model_root), None, "ref2va"
    ) == str(model_root / "Ref2VA" / "text_encoder"), (
        "Expected ref2va encoder root under Ref2VA/text_encoder"
    )

    with pytest.raises(ValueError, match="MiniMax-H3 task_type must be one of"):
        resolve_minimax_h3_partition(
            str(model_root), "InvalidTask", auto_partition="fl2va"
        )
