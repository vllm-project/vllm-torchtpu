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
"""`validate_pcp_manager_config` rejects the cache layouts the PCP manager
does not handle."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.runner.pcp import validate_pcp_manager_config


def _config():
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=4,
            decode_context_parallel_size=8,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            is_moe_model=True,
            enable_expert_parallel=True,
        ),
        model_config=SimpleNamespace(
            use_mla=True,
            hf_text_config=SimpleNamespace(index_topk=2048),
            is_multimodal_model=False,
        ),
        speculative_config=None,
        lora_config=None,
    )


def test_sparse_mla_headtail_passes(monkeypatch):
    monkeypatch.delenv("TPU_SPARSE_MLA_PACKED_ROPE", raising=False)
    validate_pcp_manager_config(_config())


def test_packed_rope_is_rejected(monkeypatch):
    monkeypatch.setenv("TPU_SPARSE_MLA_PACKED_ROPE", "1")
    with pytest.raises(NotImplementedError, match="TPU_SPARSE_MLA_PACKED_ROPE"):
        validate_pcp_manager_config(_config())
