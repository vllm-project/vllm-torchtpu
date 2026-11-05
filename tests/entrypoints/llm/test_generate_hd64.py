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
"""End-to-end generation for a head_dim=64 model on one TPU chip.

head_dim=64 layers use their own RPA kernel and KV page layout. The kernel is
picked from the query width, so engine init and the first forward pass are
the parts a CPU test cannot cover. The Qwen3 models in the other entrypoint
tests have head_dim=128 and never reach this path.

Run with: pytest tests/entrypoints/llm/test_generate_hd64.py -v
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.mark.parametrize("attention_backend", ["FLASH_ATTN", "CUSTOM"])
def test_head_dim_64_model_generates(attention_backend):
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        max_model_len=256,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        attention_backend=attention_backend,
    )
    try:
        assert llm.model_config.get_head_size() == 64
        outputs = llm.generate(
            ["The capital of France is"],
            SamplingParams(temperature=0.0, max_tokens=8),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    assert "Paris" in outputs[0].outputs[0].text
