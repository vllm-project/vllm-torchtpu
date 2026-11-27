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
"""End-to-end TPU serving test with `tpu_rope_in_graph`."""

from __future__ import annotations

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers import rotary_embedding as rope_lib
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbeddingBase

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


def _rope_cache_kinds(worker) -> list[str]:
    model = worker.model_runner.model
    return [
        type(getattr(mod, "cos_sin_cache", None)).__name__
        for mod in model.modules()
        if isinstance(mod, RotaryEmbeddingBase)
    ]


@pytest.mark.parametrize(
    ("additional_config", "expected_cache_kind"),
    [
        ({}, "InGraphCosSinCache"),
        ({"tpu_rope_in_graph": False}, "Tensor"),
    ],
)
def test_generate_with_in_graph_rope(
    monkeypatch: pytest.MonkeyPatch,
    additional_config: dict[str, object],
    expected_cache_kind: str,
) -> None:
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setattr(rope_lib, "_ROPE_DICT", {})
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=256,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        additional_config=additional_config,
    )
    try:
        kinds = llm.collective_rpc(_rope_cache_kinds)[0]
        outputs = llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0.0, max_tokens=8),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    assert kinds and set(kinds) == {expected_cache_kind}
    assert "Paris" in outputs[0].outputs[0].text
