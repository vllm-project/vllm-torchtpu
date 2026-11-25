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
"""OLMo 3.5 end to end with dummy weights and default flags.

The unit tests stub vLLM out. This one starts the real engine on a small
OLMo 3.5 config, so it covers what only a served model exercises: the
config hook turning prefix caching off under default flags, the KDA state
allocation on the per-layer layout, prefill and decode through the fused
KDA kernel with value heads wider than key heads, the NoPE attention layer
and the latent MoE.

The config keeps the production layer pattern at four layers: three KDA
layers then a full-attention one, a dense FFN first and latent MoE after.
No checkpoint or tokenizer is needed: the config is saved locally, the
weights are random and the prompts are token ids, so only that the model
runs is checked, not what it says.

Run with: pytest tests/entrypoints/llm/test_generate_olmo35.py -v

The engine runs in a spawned process and needs the TPU free, so the module is
marked to run before the in-process TPU tests (tests/conftest.py).
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from vllm_torchtpu.models.olmo35.config import Olmo35Config

# LLM(...) starts the engine core in a spawned child that needs the chip, so
# this module is marked spawns_tpu_workers to run before in-process TPU tests
# (see tests/conftest.py::pytest_collection_modifyitems).
pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]

NUM_PROMPTS = 4
MAX_TOKENS = 8


@pytest.fixture(scope="module")
def tiny_olmo35(tmp_path_factory: pytest.TempPathFactory) -> str:
    model_dir = tmp_path_factory.mktemp("tiny-olmo35")
    config = Olmo35Config(
        vocab_size=1024,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        full_attention_interval=4,
        linear_num_heads=2,
        # Value heads twice as wide as key heads, as in OLMo 3.5.
        linear_key_head_dim=128,
        linear_value_head_dim=256,
        intermediate_size=1024,
        first_k_dense_replace=1,
        shared_expert_intermediate_size=256,
        moe_intermediate_size=256,
        moe_latent_size=256,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=2048,
        architectures=["Olmo35ForCausalLM"],
    )
    assert config.layer_types == ["linear_attention"] * 3 + ["full_attention"]
    config.save_pretrained(model_dir)
    return str(model_dir)


@pytest.mark.timeout(1800)
def test_olmo35_generates_with_default_flags(monkeypatch, tiny_olmo35: str):
    # Default flags: the model picks its own KV layout and prefix caching
    # stays at vLLM's default (on), which the config hook turns off.
    monkeypatch.delenv("TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL", raising=False)
    llm = LLM(
        model=tiny_olmo35,
        load_format="dummy",
        skip_tokenizer_init=True,
        max_model_len=1024,
        max_num_seqs=NUM_PROMPTS,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.5,
        seed=0,
    )
    try:
        assert not llm.llm_engine.vllm_config.cache_config.enable_prefix_caching
        # Prompts of different lengths, so the batch mixes prefill lengths and
        # every request then decodes several steps.
        prompts = [
            {"prompt_token_ids": [(7 * i + j) % 1000 + 10 for j in range(40 + 9 * i)]}
            for i in range(NUM_PROMPTS)
        ]
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True
            ),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    assert len(outputs) == NUM_PROMPTS
    for output in outputs:
        token_ids = output.outputs[0].token_ids
        assert len(token_ids) == MAX_TOKENS
        assert all(0 <= t < 1024 for t in token_ids)
