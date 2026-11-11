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
from vllm.config import ParallelConfig
from vllm.config.model import ModelConfig

from vllm_torchtpu import patch_registry
from vllm_torchtpu.distributed import dcp


def _parallel(tp: int, dcp_size: int) -> ParallelConfig:
    return ParallelConfig(
        tensor_parallel_size=tp, decode_context_parallel_size=dcp_size
    )


def _model(arch: str, heads: int = 32, kv_heads: int = 4) -> SimpleNamespace:
    # The ModelConfig fields vLLM's verify_with_parallel_config reads.
    return SimpleNamespace(
        architectures=[arch],
        use_mla=False,
        model_arch_config=SimpleNamespace(
            total_num_attention_heads=heads, total_num_kv_heads=kv_heads
        ),
        get_total_num_kv_heads=lambda: kv_heads,
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
    monkeypatch.setattr(dcp, "_KV_HEAD_SPANNING_DCP_ARCHITECTURES", set())
    dcp._patch_vllm_gqa_dcp_limit()
    wrapper = ModelConfig.verify_with_parallel_config
    dcp._patch_vllm_gqa_dcp_limit()  # the marker makes it install once
    assert ModelConfig.verify_with_parallel_config is wrapper
    dcp.register_kv_head_spanning_dcp_architectures("ArchA")


def test_patch_carries_its_marker() -> None:
    assert ModelConfig.verify_with_parallel_config._tpu_gqa_dcp_limit_patch


def test_patch_is_registered_at_plugin_load() -> None:
    plugin_load = [
        p.target for p in patch_registry.PATCHES if "plugin_load" in p.stages
    ]
    assert plugin_load == ["vllm_torchtpu.distributed.dcp:_patch_vllm_gqa_dcp_limit"]


def test_registered_architecture_spans_kv_heads() -> None:
    # TP 8 over 4 KV heads caps vLLM's GQA DCP at 2; ArchA may use 8.
    ModelConfig.verify_with_parallel_config(_model("ArchA"), _parallel(8, 8))
    ModelConfig.verify_with_parallel_config(_model("ArchA"), _parallel(8, 4))


def test_unregistered_architecture_keeps_vllm_check() -> None:
    with pytest.raises(ValueError, match="exceeds the maximum"):
        ModelConfig.verify_with_parallel_config(_model("ArchB"), _parallel(8, 8))


def test_tp_equal_to_kv_heads() -> None:
    # TP 4 over 4 KV heads caps vLLM's GQA DCP at 1.
    ModelConfig.verify_with_parallel_config(_model("ArchA"), _parallel(4, 4))
    with pytest.raises(ValueError, match="greater than the model's total number"):
        ModelConfig.verify_with_parallel_config(_model("ArchB"), _parallel(4, 4))


def test_registered_dcp_groups_hold_whole_kv_heads() -> None:
    # TP 12 over 4 KV heads: 3 replicas per KV head, so DCP 4 would split the
    # replicas of one KV head across groups.
    model = _model("ArchA", heads=48, kv_heads=4)
    with pytest.raises(ValueError, match="multiple of"):
        ModelConfig.verify_with_parallel_config(model, _parallel(12, 4))
    ModelConfig.verify_with_parallel_config(model, _parallel(12, 6))
