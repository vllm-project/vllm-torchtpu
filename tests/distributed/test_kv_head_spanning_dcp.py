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
"""GQA DCP up to the TP size for registered architectures."""

from types import SimpleNamespace

import pytest
from vllm.config.model import ModelConfig

from vllm_torchtpu.distributed import dcp


def _parallel(tp: int, dcp_size: int) -> SimpleNamespace:
    # The ParallelConfig fields vLLM's verify_with_parallel_config reads.
    return SimpleNamespace(
        enable_hc_sp=False,
        tensor_parallel_size=tp,
        enable_expert_parallel=False,
        pipeline_parallel_size=1,
        decode_context_parallel_size=dcp_size,
        world_size_across_dp=tp,
    )


def _model(arch: str) -> SimpleNamespace:
    # The ModelConfig fields it reads: 32 query heads over 4 KV heads.
    return SimpleNamespace(
        architectures=[arch],
        use_mla=False,
        model_arch_config=SimpleNamespace(
            total_num_attention_heads=32, total_num_kv_heads=4
        ),
        get_total_num_kv_heads=lambda: 4,
        multimodal_config=None,
    )


@pytest.fixture(autouse=True)
def patched(monkeypatch):
    # Undo the installed wrapper and the registry after each test.
    monkeypatch.setattr(
        ModelConfig,
        "verify_with_parallel_config",
        ModelConfig.verify_with_parallel_config,
    )
    monkeypatch.setattr(dcp, "_GQA_DCP_LIMIT_PATCHED", False)
    monkeypatch.setattr(dcp, "_KV_HEAD_SPANNING_DCP_ARCHITECTURES", set())
    dcp.register_kv_head_spanning_dcp_architectures("ArchA")
    dcp.register_kv_head_spanning_dcp_architectures("ArchA")  # patches once


def test_registered_architecture_spans_kv_heads() -> None:
    # TP 8 over 4 KV heads caps vLLM's GQA DCP at 2; ArchA may use 8.
    ModelConfig.verify_with_parallel_config(_model("ArchA"), _parallel(8, 8))
    ModelConfig.verify_with_parallel_config(_model("ArchA"), _parallel(8, 4))


def test_unregistered_architecture_keeps_vllm_check() -> None:
    with pytest.raises(ValueError, match="exceeds the maximum"):
        ModelConfig.verify_with_parallel_config(_model("ArchB"), _parallel(8, 8))


def test_registered_dcp_must_divide_tp() -> None:
    with pytest.raises(ValueError, match="must divide"):
        ModelConfig.verify_with_parallel_config(_model("ArchA"), _parallel(12, 8))
