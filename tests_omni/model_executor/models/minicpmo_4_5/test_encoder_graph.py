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
from vllm_omni.model_executor.models.minicpmo_4_5.encoder_graph import (
    make_encoder_graph,
)

from tests_omni.omni_test_utils import PR_TIMEOUT_S, assert_parity

pytestmark = [pytest.mark.omni, pytest.mark.timeout(PR_TIMEOUT_S)]

MAX_REL_L2 = 2e-2
MAX_ABS = 5e-2


class _TinyEncoderMLP(nn.Module):
    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        "EncoderCudaGraph.__call__ marks non-CUDA tensors ineligible on TPU "
        "(num_graphs == 0)"
    ),
)
def test_minicpmo_4_5_input_encoder_cuda_graphs() -> None:
    """MiniCPM-o 4.5 Input Encoder CUDA Graphs

    - TPU path: wrap a tiny 2-layer MLP encoder on TPU bf16 and call it 3 times.
    - Setup check: assert stats reports num_graphs == 1 and graph_hits == 1.
    - Outcome check: compare TPU bf16 output against CPU fp32 via assert_parity.
    """
    torch.manual_seed(0)
    cpu_encoder = _TinyEncoderMLP(hidden_size=32).float().eval()
    tpu_encoder = (
        copy.deepcopy(cpu_encoder).to(device="tpu", dtype=torch.bfloat16).eval()
    )

    vllm_config = object.__new__(VllmConfig)
    vllm_config.model_config = SimpleNamespace(  # type: ignore[assignment]
        enforce_eager=False,
        hf_config=SimpleNamespace(
            encoder_cuda_graph=True,
            encoder_cuda_graph_min_capture_calls=2,
            encoder_cuda_graph_min_free_bytes=0,
        ),
    )
    vllm_config.additional_config = {}

    graph_runner = make_encoder_graph(tpu_encoder.forward, vllm_config)
    assert graph_runner is not None, (
        "expected make_encoder_graph to return a graph runner"
    )

    x_cpu = torch.randn((2, 8, 32), dtype=torch.bfloat16)
    with torch.no_grad():
        ref = cpu_encoder(x_cpu.float())
        x_tpu = x_cpu.to("tpu")
        _ = graph_runner(x_tpu)
        _ = graph_runner(x_tpu)
        out = graph_runner(x_tpu)

    stats = graph_runner.get_cumulative_stats()
    assert stats["num_graphs"] == 1, (
        f"expected num_graphs == 1, got {stats['num_graphs']}"
    )
    assert stats["graph_hits"] == 1, (
        f"expected graph_hits == 1, got {stats['graph_hits']}"
    )

    assert_parity(
        out,
        ref,
        name="minicpmo_4_5_input_encoder_cuda_graphs",
        max_rel_l2=MAX_REL_L2,
        max_abs=MAX_ABS,
    )
