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
serves a few greedy tokens at TP 8 x DCP 8. Per PR: the FP8 checkpoint
this repo serves, with the model config's default (q-replicate on).
Nightly: the NVFP4 checkpoint, and q-replicate off on the bf16 one. The
weights are random, so only that it runs is checked, not what it says.

Run with: pytest tests/entrypoints/llm/test_generate_glm52_dcp.py -v
"""

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

pytestmark = pytest.mark.multichip

NUM_LAYERS = 4
HF_OVERRIDES = {
    "num_hidden_layers": NUM_LAYERS,
    "num_nextn_predict_layers": 0,
    # vLLM builds MoE from first_k_dense_replace and the indexer layout from
    # index_topk_pattern ("S" shares the previous layer's top-k).
    "first_k_dense_replace": 1,
    "index_topk_pattern": "FFFS",
}
EXPECTED_SKIP_TOPK = [False, False, False, True]
DCP_SIZE = 8
# The production interleave; one cycle is DCP_SIZE * INTERLEAVE positions.
INTERLEAVE = 4
# Long enough that every prompt spans several cycles, so each DCP rank owns
# keys and the decode merge combines all of them.
PROMPT_LEN = 80
NUM_PROMPTS = 4
MAX_TOKENS = 4


def _mla_qrep_state(model) -> list[tuple[bool, bool, bool, bool]]:
    """(dcp_q_replicate, is_sparse, replicated W_UK_T present, skip_topk) per
    MLA layer."""
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        VllmMultiHeadLatentAttentionWrapper,
    )

    return [
        (
            module.dcp_q_replicate,
            module.is_sparse,
            module.mla_attn.W_UK_T_dcp_qrep is not None,
            module.skip_topk,
        )
        for module in model.modules()
        if isinstance(module, VllmMultiHeadLatentAttentionWrapper)
    ]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "model, dcp_q_replicate, env",
    [
        # None leaves q-replicate to the GLM-5.2 config hook, which turns it on.
        pytest.param("zai-org/GLM-5.2-FP8", None, {}, id="fp8"),
        # Block-512 MoE scales, as the NVFP4 nightly integration job serves.
        pytest.param(
            "nvidia/GLM-5.2-NVFP4",
            None,
            {"MOE_REQUANTIZE_BLOCK_SIZE": "512"},
            marks=pytest.mark.nightly,
            id="nvfp4",
        ),
        pytest.param(
            "zai-org/GLM-5.2", False, {}, marks=pytest.mark.nightly, id="bf16-ag_q"
        ),
    ],
)
def test_glm52_dcp_generates(
    monkeypatch,
    local_tpu_chip_count: int,
    model: str,
    dcp_q_replicate: bool | None,
    env: dict[str, str],
):
    if local_tpu_chip_count < DCP_SIZE:
        pytest.skip(f"needs {DCP_SIZE} chips for TP {DCP_SIZE} x DCP {DCP_SIZE}")
    # The variable overrides dcp_q_replicate when set; leave the choice to it.
    monkeypatch.delenv("VLLM_DCP_Q_REPLICATE", raising=False)
    # apply_model sends a function to the workers.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    q_replicate = dcp_q_replicate is not False

    llm = LLM(
        model=model,
        hf_overrides=HF_OVERRIDES,
        load_format="dummy",
        skip_tokenizer_init=True,
        tensor_parallel_size=DCP_SIZE,
        decode_context_parallel_size=DCP_SIZE,
        cp_kv_cache_interleave_size=INTERLEAVE,
        dcp_q_replicate=dcp_q_replicate,
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
        layers_per_rank = llm.apply_model(_mla_qrep_state)
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=MAX_TOKENS, ignore_eos=True
            ),
        )
    finally:
        del llm
        cleanup_dist_env_and_memory()

    # Both cases must prove which path they ran: a vLLM bump that drops
    # q-replicate silently would still generate on the all-gather path.
    assert len(layers_per_rank) == DCP_SIZE
    for layers in layers_per_rank:
        assert len(layers) == NUM_LAYERS
        assert [skip for *_, skip in layers] == EXPECTED_SKIP_TOPK
        for flag, is_sparse, has_qrep_weights, _ in layers:
            assert flag == q_replicate
            assert has_qrep_weights == (q_replicate and is_sparse)
        if q_replicate:
            assert any(has_qrep_weights for _, _, has_qrep_weights, _ in layers)

    assert len(outputs) == NUM_PROMPTS
    for output in outputs:
        assert len(output.outputs) == 1
        # One prefill and MAX_TOKENS - 1 decode steps each produced a token.
        assert len(output.outputs[0].token_ids) == MAX_TOKENS
