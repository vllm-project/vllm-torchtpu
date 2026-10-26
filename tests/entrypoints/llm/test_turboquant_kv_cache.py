# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""End-to-end TurboQuant KV cache on TPU with Qwen3-0.6B (head_dim 128)."""

import gc
from itertools import takewhile

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.tpu_test

PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:",
    "List the first five prime numbers:",
    "Explain what a hash table is in one sentence:",
]
SAMPLING = SamplingParams(temperature=0.0, max_tokens=32)


def _generate(kv_cache_dtype: str) -> tuple[list[list[int]], list[str]]:
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        kv_cache_dtype=kv_cache_dtype,
    )
    try:
        outs = llm.generate(PROMPTS, SAMPLING)
        return [list(o.outputs[0].token_ids) for o in outs], [
            o.outputs[0].text for o in outs
        ]
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def bf16_tokens() -> list[list[int]]:
    return _generate("auto")[0]


@pytest.mark.parametrize(
    ("preset", "min_total", "min_max"),
    [("turboquant_k8v4", 12, 8), ("turboquant_4bit_nc", 6, 4)],
)
def test_turboquant_tracks_bf16(
    bf16_tokens: list[list[int]], preset: str, min_total: int, min_max: int
):
    tq_tokens, tq_texts = _generate(preset)
    prefixes = [
        sum(1 for _ in takewhile(lambda ab: ab[0] == ab[1], zip(a, b)))
        for a, b in zip(bf16_tokens, tq_tokens)
    ]
    assert sum(prefixes) >= min_total and max(prefixes) >= min_max, prefixes
    assert "Paris" in tq_texts[0], tq_texts[0]
    assert "ocean" in tq_texts[1].lower(), tq_texts[1]
    assert prefixes[2] >= 4 or ("2" in tq_texts[2] and "3" in tq_texts[2]), tq_texts[2]
    assert "data structure" in tq_texts[3].lower(), tq_texts[3]
