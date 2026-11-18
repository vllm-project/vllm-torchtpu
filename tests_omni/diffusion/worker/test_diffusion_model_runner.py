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
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
    merge_pipeline_deploy,
)
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker import diffusion_model_runner
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity
from tests_omni.tiny_models import tiny_diffusion_dir

pytestmark = [
    pytest.mark.omni_feature,
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]

MAX_REL_L2 = 2e-2
MAX_ABS = 5e-2


class _TinyRepeatedBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x)) + x


class _TinyRepeatedTransformer(nn.Module):
    _repeated_blocks = ["_TinyRepeatedBlock"]

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyRepeatedBlock(hidden_size) for _ in range(2)])
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(x)


@pytest.mark.parametrize(
    "force_inductor", [False, True], ids=["platform_default", "inductor_true"]
)
def test_enforce_eager_execution(
    monkeypatch: pytest.MonkeyPatch,
    force_inductor: bool,
) -> None:
    """Enforce Eager Execution

    - TPU path: resolve enforce_eager=True via merge_pipeline_deploy and load_model.
    - Setup check: assert enforce_eager, _compiled_call_impl is None, uncompiled blocks.
    - Outcome check: run eager TPU bf16 forward and verify parity via assert_parity.
    """
    platform_supports_inductor = current_omni_platform.supports_torch_inductor
    # inductor_true reports inductor support; platform_default keeps the real answer.
    monkeypatch.setattr(
        current_omni_platform,
        "supports_torch_inductor",
        lambda: force_inductor or platform_supports_inductor(),
    )

    pipeline_cfg = PipelineConfig(
        model_type="tiny-eager",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="diffusion",
                execution_type=StageExecutionType.DIFFUSION,
                final_output=True,
            ),
        ),
    )
    deploy_cfg = DeployConfig(
        stages=[StageDeployConfig(stage_id=0, enforce_eager=True)]
    )
    stage_configs = merge_pipeline_deploy(pipeline_cfg, deploy_cfg)
    assert len(stage_configs) == 1, f"expected 1 stage config, got {len(stage_configs)}"
    enforce_eager = bool(stage_configs[0].yaml_engine_args["enforce_eager"])

    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(), enforce_eager=enforce_eager
    )
    assert od_config.enforce_eager is True, (
        f"expected od_config.enforce_eager is True, got {od_config.enforce_eager}"
    )

    torch.manual_seed(0)
    cpu_model = _TinyRepeatedTransformer(hidden_size=32).float().eval()
    tpu_model = copy.deepcopy(cpu_model).to(device="tpu", dtype=torch.bfloat16).eval()
    pipeline = SimpleNamespace(transformer=tpu_model)

    runner = object.__new__(DiffusionModelRunner)
    runner.vllm_config = VllmConfig()
    runner.od_config = od_config
    runner.device = torch.device("tpu")
    runner.pipeline = None
    runner.cache_backend = None
    runner.offload_backend = None
    runner.model_memory_usage = 0

    monkeypatch.setattr(
        diffusion_model_runner.DiffusersPipelineLoader,
        "load_model",
        lambda _self, **_kwargs: pipeline,
    )
    monkeypatch.setattr(
        diffusion_model_runner,
        "enable_offload_backend",
        lambda _cfg, pipe, device=None: (pipe, None),
    )

    runner.load_model()
    loaded = runner.pipeline.transformer

    assert loaded._compiled_call_impl is None, (
        f"expected _compiled_call_impl is None, got {loaded._compiled_call_impl}"
    )
    for idx, block in enumerate(loaded.blocks):
        assert block.forward == _TinyRepeatedBlock.forward.__get__(
            block, _TinyRepeatedBlock
        ), f"expected block[{idx}].forward to remain an uncompiled bound method"

    x_cpu = torch.randn((2, 8, 32), dtype=torch.bfloat16)
    ref = cpu_model(x_cpu.float())
    with torch.no_grad():
        actual = loaded(x_cpu.to("tpu"))
    assert_parity(
        actual,
        ref,
        name=f"enforce_eager_inductor_{force_inductor}",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
