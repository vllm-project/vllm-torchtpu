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
"""GLM-5.2 under decode context parallelism, end to end, with dummy weights.

A smoke test for vLLM upgrades: GLM-5.2's MLA DCP path leans on upstream
pieces (the model config hook that sets the DCP defaults, `MLAAttention`'s
weight processing, the q-replicate projection) that a vLLM bump can change
under it. Two such bumps broke it, one at startup and one in the decode
graph compile, neither visible in unit tests that stub upstream out.

The real GLM-5.2 config, cut to four layers that keep the production
pattern (three full indexers then a shared one; one dense MLP then MoE),
serves a few greedy tokens at TP 8 x DCP 8 with q-replicate off and on.
The weights are random, so only that it runs is checked, not what it says.

Run with: pytest tests/entrypoints/llm/test_generate_glm52_dcp.py -v
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.multichip

GLM52_MODEL_NAME = "zai-org/GLM-5.2"
NUM_LAYERS = 4
HF_OVERRIDES = {
    "num_hidden_layers": NUM_LAYERS,
    "num_nextn_predict_layers": 0,
    "first_k_dense_replace": 1,
    "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
    "indexer_types": ["full", "full", "full", "shared"],
}
DCP_SIZE = 8
# The production interleave; one cycle is DCP_SIZE * 4 = 32 positions.
INTERLEAVE = 4
# Long enough that every prompt spans several cycles, so each DCP rank owns
# keys and the decode merge combines all of them.
PROMPT_LEN = 80
NUM_PROMPTS = 4
MAX_TOKENS = 4


def _get_local_tpu_chip_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("q_replicate", [False, True], ids=["ag_q", "q_replicate"])
def test_glm52_dcp_generates(monkeypatch, q_replicate: bool):
    tp_size = _get_local_tpu_chip_count()
    if tp_size < DCP_SIZE:
        pytest.skip(f"needs {DCP_SIZE} chips for TP {DCP_SIZE} x DCP {DCP_SIZE}")
    # Upstream reads the variable only when it is set, so set it both ways
    # rather than leave one case to the model config's default.
    monkeypatch.setenv("VLLM_DCP_Q_REPLICATE", "1" if q_replicate else "0")

    llm = LLM(
        model=GLM52_MODEL_NAME,
        hf_overrides=HF_OVERRIDES,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=DCP_SIZE,
        decode_context_parallel_size=DCP_SIZE,
        cp_kv_cache_interleave_size=INTERLEAVE,
        enable_expert_parallel=True,
        block_size=256,
        # The sparse MLA kernel reads an fp8 latent cache, as GLM-5.2 serves.
        kv_cache_dtype="fp8",
        max_model_len=1024,
        max_num_batched_tokens=512,
        max_num_seqs=NUM_PROMPTS,
        gpu_memory_utilization=0.6,
        seed=0,
    )
    prompts = [
        {"prompt_token_ids": [(7 * i + j) % 1000 + 100 for j in range(PROMPT_LEN)]}
        for i in range(NUM_PROMPTS)
    ]
    try:
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
        assert len(output.outputs) == 1
        # One prefill and MAX_TOKENS - 1 decode steps each produced a token.
        assert len(output.outputs[0].token_ids) == MAX_TOKENS
