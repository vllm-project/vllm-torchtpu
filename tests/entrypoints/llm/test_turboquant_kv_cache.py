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
"""End-to-end TurboQuant KV cache on TPU with Qwen3-0.6B (head_dim 128).

Greedy outputs with a TurboQuant cache are compared with a bf16 cache. FP8
keys (`k8v4`) should track bf16 closely; 4-bit MSE keys a little less so.

Run with: pytest tests/entrypoints/llm/test_turboquant_kv_cache.py -sv
"""

import gc

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "The capital of France is",
    "Write a haiku about the ocean:",
    "List the first five prime numbers:",
    "Explain what a hash table is in one sentence:",
]
SAMPLING = SamplingParams(temperature=0.0, max_tokens=32)


def _generate(kv_cache_dtype: str) -> tuple[list[list[int]], list[str]]:
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=8,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        kv_cache_dtype=kv_cache_dtype,
    )
    try:
        outputs = llm.generate(PROMPTS, SAMPLING)
        for p, o in zip(PROMPTS, outputs):
            print(f"[{kv_cache_dtype}] {p!r} -> {o.outputs[0].text!r}")
        tokens = [list(o.outputs[0].token_ids) for o in outputs]
        texts = [o.outputs[0].text for o in outputs]
        return tokens, texts
    finally:
        del llm
        gc.collect()
        cleanup_dist_env_and_memory()


def _prefix_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.fixture(scope="module")
def bf16_outputs():
    return _generate("auto")


@pytest.mark.parametrize(
    "preset,min_total_prefix,min_max_prefix",
    [("turboquant_k8v4", 12, 8), ("turboquant_4bit_nc", 6, 4)],
)
def test_turboquant_tracks_bf16(bf16_outputs, preset, min_total_prefix, min_max_prefix):
    bf16_tokens, _ = bf16_outputs
    tq_tokens, tq_texts = _generate(preset)
    prefixes = [_prefix_match(a, b) for a, b in zip(bf16_tokens, tq_tokens)]
    print(f"{preset}: greedy prefix match vs bf16 per prompt = {prefixes}")

    # Qwen3-0.6B with all 28 layers quantized (including boundary layers 0, 1,
    # 26, 27) can flip near-tied logits on open-ended prompts, so check both
    # token-prefix agreement with bf16 and factual content of each completion.
    assert sum(prefixes) >= min_total_prefix, prefixes
    assert max(prefixes) >= min_max_prefix, prefixes
    assert "Paris" in tq_texts[0], tq_texts[0]
    assert "ocean" in tq_texts[1].lower(), tq_texts[1]
    assert prefixes[2] >= 4 or ("2" in tq_texts[2] and "3" in tq_texts[2]), tq_texts[2]
    assert "data structure" in tq_texts[3].lower(), tq_texts[3]
