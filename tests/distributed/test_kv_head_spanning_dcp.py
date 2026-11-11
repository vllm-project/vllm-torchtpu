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

from vllm_torchtpu.distributed import dcp


def _parallel(tp: int, dcp_size: int):
    return SimpleNamespace(
        tensor_parallel_size=tp, decode_context_parallel_size=dcp_size
    )


def test_kv_head_spanning_dcp_clamps_vllm_gqa_check(monkeypatch) -> None:
    from vllm.config.model import ModelConfig

    seen = []
    monkeypatch.setattr(
        ModelConfig,
        "verify_with_parallel_config",
        lambda self, pc: seen.append(pc.decode_context_parallel_size),
    )
    monkeypatch.setattr(dcp, "_KV_HEAD_SPANNING_DCP_ARCHITECTURES", set())
    dcp.register_kv_head_spanning_dcp_architectures("ArchA")
    dcp.register_kv_head_spanning_dcp_architectures("ArchA")  # patches once

    def model(arch):
        return SimpleNamespace(
            architectures=[arch], use_mla=False, get_total_num_kv_heads=lambda: 4
        )

    ModelConfig.verify_with_parallel_config(model("ArchA"), _parallel(8, 8))
    ModelConfig.verify_with_parallel_config(model("ArchB"), _parallel(8, 8))
    # TP 8 over 4 KV heads caps vLLM's GQA DCP at 2.
    assert seen == [2, 8]
