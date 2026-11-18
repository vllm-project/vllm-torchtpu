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
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.worker import diffusion_model_runner
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

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
        self.stem = nn.Linear(hidden_size, hidden_size)
        self.blocks = nn.ModuleList([_TinyRepeatedBlock(hidden_size) for _ in range(2)])
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _is_compiled_callable(fn: object) -> bool:
    fn_dict = fn.__dict__
    return "_torchdynamo_orig_callable" in fn_dict or "__wrapped__" in fn_dict


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "OmniTpuPlatform.supports_torch_inductor() returns False, "
        "so DiffusionModelRunner skips torch.compile"
    ),
)
@pytest.mark.parametrize(
    ("granularity", "dynamic", "seq_lens", "expect_full", "expect_blocks"),
    [
        pytest.param("regional", True, (8, 12), False, True, id="regional-True"),
        pytest.param("regional", False, (8,), False, True, id="regional-False"),
        pytest.param("full", True, (8, 12), True, False, id="full-True"),
    ],
)
def test_regional_compilation(
    monkeypatch: pytest.MonkeyPatch,
    granularity: str,
    dynamic: bool,
    seq_lens: tuple[int, ...],
    expect_full: bool,
    expect_blocks: bool,
) -> None:
    """Regional Compilation (torch.compile)

    - TPU path: run DiffusionModelRunner.load_model on a tiny TPU bf16 transformer.
    - Setup check: assert which callables are compiled: only the repeated blocks
      (regional) or the whole transformer via _compiled_call_impl (full).
    - Outcome check: run compiled forward at each sequence length (8 and 12 when
      dynamic) and verify parity via assert_parity.
    """
    torch.manual_seed(0)
    cpu_model = _TinyRepeatedTransformer(hidden_size=32).float().eval()
    tpu_model = copy.deepcopy(cpu_model).to(device="tpu", dtype=torch.bfloat16).eval()
    pipeline = SimpleNamespace(transformer=tpu_model)

    od_config = OmniDiffusionConfig(
        model=tiny_diffusion_dir(),
        enforce_eager=False,
        diffusion_compile_granularity=granularity,
        diffusion_compile_dynamic=dynamic,
    )
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
    compiled_model = runner.pipeline.transformer

    full_compiled = compiled_model._compiled_call_impl is not None
    assert full_compiled is expect_full, (
        f"{granularity}: transformer._compiled_call_impl set={full_compiled}, "
        f"expected {expect_full}"
    )
    blocks_compiled = [
        _is_compiled_callable(block.forward) for block in compiled_model.blocks
    ]
    assert blocks_compiled == [expect_blocks] * len(blocks_compiled), (
        f"{granularity}: compiled block forwards {blocks_compiled}, "
        f"expected all {expect_blocks}"
    )
    assert not _is_compiled_callable(compiled_model.stem.forward), (
        f"{granularity}: compilation unexpectedly wrapped non-repeated stem.forward"
    )
    assert not _is_compiled_callable(compiled_model.head.forward), (
        f"{granularity}: compilation unexpectedly wrapped non-repeated head.forward"
    )

    # One compiled model serves every length in turn (two for dynamic shapes).
    for seq_len in seq_lens:
        x_cpu = torch.randn((2, seq_len, 32), dtype=torch.bfloat16)
        ref = cpu_model(x_cpu.float())
        with torch.no_grad():
            actual = compiled_model(x_cpu.to("tpu"))
        assert_parity(
            actual,
            ref,
            name=f"regional_compilation_{granularity}_dyn{dynamic}_s{seq_len}",
            max_rel_l2=MAX_REL_L2,
            max_abs=MAX_ABS,
        )
