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
"""End-to-end check of a tiny Qwen4Exp against the HF reference on CPU.

The tiny model keeps the release's structure (3 GDN + 1 QSA layer, four
hyper-connection streams, n-gram PLE on layer 2, MoE with a shared expert)
and its 128-wide heads (the TPU GDN and attention kernels lay heads out on
128 lanes), with random weights and a QSA budget small enough that prompts
past 19 tokens take the sparse path.

TPU serves the model in bfloat16 only, so the gate is the same-dtype HF
reference: HF in bfloat16 on CPU with identical weights must produce TPU's
greedy tokens. Teacher-forced log-probabilities against float32 are a
diagnostic, and a prefix-cache hit must reproduce the miss exactly.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

pytestmark = [pytest.mark.tpu_test, pytest.mark.spawns_tpu_workers]

DECODE_STEPS = 16
PROMPT_LENGTHS = (7, 33, 61, 1000)
# Large enough that the HF float32 greedy margins exceed MIN_MARGIN.
LM_HEAD_SCALE = 4.0
MIN_MARGIN = 0.5
# The first query position whose query sees more complete QSA blocks than
# the budget keeps: (indexer_budget / compress_ratio + 1) * compress_ratio - 1.
FIRST_SPARSE_POSITION = 19


def _tiny_config() -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        architectures=["Qwen4ExpForCausalLM"],
        vocab_size=512,
        hidden_size=256,
        num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        hc_count=4,
        hc_lowrank=16,
        ple_layer_ids=[2],
        ple_embed_dim=256,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=997,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=4,
        ple_conv_kernel_size=4,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=16,
        indexer_compress_ratio=4,
        num_experts=16,
        num_experts_per_tok=4,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1e7,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
        max_position_embeddings=4096,
        mtp_num_hidden_layers=0,
        tie_word_embeddings=False,
        # Release values the defaults do not cover.
        output_gate_type="sigmoid",
        norm_topk_prob=True,
        mamba_ssm_dtype="float32",
        full_attention_interval=4,
        seed=1234,
        bos_token_id=1,
        eos_token_id=2,
        dtype="float32",
    )


def _token_logprobs(model: Qwen4ExpForCausalLM, ids: list[int]) -> torch.Tensor:
    """log p(ids[i] | ids[:i]) for i >= 1, teacher-forced."""
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, :-1].float()
    return logits.log_softmax(-1).gather(-1, torch.tensor(ids[1:])[:, None])[:, 0]


def _greedy_with_margins(model, prompt: list[int]) -> tuple[list[int], list[float]]:
    """Greedy tokens and each step's top-1 minus top-2 log-probability."""
    with torch.no_grad():
        out = model.generate(
            torch.tensor([prompt]),
            max_new_tokens=DECODE_STEPS,
            do_sample=False,
            eos_token_id=None,
            output_logits=True,
            return_dict_in_generate=True,
        )
    margins = []
    for step in out.logits:
        top2 = step[0].float().log_softmax(-1).topk(2).values
        margins.append(float(top2[0] - top2[1]))
    return out.sequences[0, len(prompt) :].tolist(), margins


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Saves the tiny model and runs HF on CPU in float32 and bfloat16.

    Each sequence is a random prompt plus its float32 greedy continuation;
    fp32 and bf16 hold the teacher-forced logprobs of those sequences.
    """
    # Seed only the CPU generator: torch.manual_seed would also seed (and so
    # claim) the TPU, which the engine process needs.
    torch.default_generator.manual_seed(0)
    model = Qwen4ExpForCausalLM(_tiny_config()).eval()
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        # Perturb every parameter so no norm or gate stays at its init value.
        for param in model.parameters():
            param.add_(torch.randn(param.shape, generator=generator) * 0.02)
        # A random model scores the vocabulary almost uniformly: its top-2
        # logits sit within ~0.01 of each other, so greedy decoding flips
        # between any two runs of different precision. Reading each token's
        # successor off a fixed permutation of the embeddings makes the
        # prediction decisive while every layer still adds to the residual.
        permutation = torch.randperm(512, generator=generator)
        model.lm_head.weight.copy_(
            LM_HEAD_SCALE * model.model.embed_tokens.weight[permutation]
        )
    path = tmp_path_factory.mktemp("qwen4_exp_tiny")
    model.save_pretrained(path, safe_serialization=True)
    generator = torch.Generator().manual_seed(2)
    prompts = [
        torch.randint(3, 512, (n,), generator=generator).tolist()
        for n in PROMPT_LENGTHS
    ]
    fp32_greedy = [_greedy_with_margins(model, p) for p in prompts]
    margins = [m for _, step_margins in fp32_greedy for m in step_margins]
    print(f"HF fp32 greedy margins: min={min(margins):.3f}")
    assert min(margins) > MIN_MARGIN, "the tiny model is not decisive enough"
    sequences = [p + tokens for p, (tokens, _) in zip(prompts, fp32_greedy)]
    fp32 = [_token_logprobs(model, ids) for ids in sequences]
    model.to(torch.bfloat16)
    bf16_greedy = [_greedy_with_margins(model, p) for p in prompts]
    bf16 = [_token_logprobs(model, ids) for ids in sequences]
    return SimpleNamespace(
        path=str(path),
        sequences=sequences,
        fp32=fp32,
        bf16=bf16,
        fp32_greedy=fp32_greedy,
        bf16_greedy=bf16_greedy,
    )


@pytest.fixture(scope="module")
def llm(reference):
    path = reference.path
    llm = LLM(
        model=path,
        skip_tokenizer_init=True,
        max_model_len=2048,
        max_num_seqs=8,
        max_num_batched_tokens=512,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        disable_log_stats=False,
    )
    yield weakref.proxy(llm)
    del llm


def _prompt_logprobs(llm, sequences) -> list[torch.Tensor]:
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in sequences], params
    )
    return [
        torch.tensor([lp[t].logprob for t, lp in zip(ids[1:], o.prompt_logprobs[1:])])
        for ids, o in zip(sequences, outputs)
    ]


def test_greedy_tokens_match_hf_bf16(llm, reference) -> None:
    sequences = reference.sequences
    prompts = [ids[:-DECODE_STEPS] for ids in sequences]
    params = SamplingParams(
        temperature=0.0, max_tokens=DECODE_STEPS, ignore_eos=True, logprobs=2
    )
    outputs = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    failures = []
    for prompt, out, (fp32_tokens, fp32_margins), (bf16_tokens, bf16_margins) in zip(
        prompts, outputs, reference.fp32_greedy, reference.bf16_greedy
    ):
        tokens = list(out.outputs[0].token_ids)
        tpu_margins = []
        for step in out.outputs[0].logprobs:
            top = sorted((lp.logprob for lp in step.values()), reverse=True)
            tpu_margins.append(top[0] - top[1])
        print(f"prompt len={len(prompt)}")
        for i in range(DECODE_STEPS):
            print(
                f"  step {i}: tpu={tokens[i]} hf_bf16={bf16_tokens[i]} "
                f"hf_fp32={fp32_tokens[i]} margin tpu={tpu_margins[i]:.4f} "
                f"bf16={bf16_margins[i]:.4f} fp32={fp32_margins[i]:.4f}"
            )
        if tokens != bf16_tokens:
            failures.append(len(prompt))
    assert not failures, f"TPU greedy differs from HF bf16 for prompts {failures}"


def test_logprobs_track_the_reference(llm, reference) -> None:
    sequences, fp32, bf16 = reference.sequences, reference.fp32, reference.bf16
    tpu = _prompt_logprobs(llm, sequences)
    # Position i of a logprob vector scores token i + 1, queried at i.
    for name, positions in (
        ("dense", slice(0, FIRST_SPARSE_POSITION)),
        ("sparse", slice(FIRST_SPARSE_POSITION, None)),
    ):
        tpu_err = torch.cat([(t - r).abs()[positions] for t, r in zip(tpu, fp32)])
        hf_err = torch.cat([(b - r).abs()[positions] for b, r in zip(bf16, fp32)])
        print(
            f"{name}: positions={tpu_err.numel()} "
            f"tpu-fp32 mean={tpu_err.mean():.3e} max={tpu_err.max():.3e} | "
            f"hf_bf16-fp32 mean={hf_err.mean():.3e} max={hf_err.max():.3e}"
        )
        bf16_err = torch.cat([(t - b).abs()[positions] for t, b in zip(tpu, bf16)])
        print(
            f"{name}: tpu-hf_bf16 mean={bf16_err.mean():.3e} max={bf16_err.max():.3e}"
        )
        assert torch.isfinite(tpu_err).all()
        bound = max(float(hf_err.mean()) * 4.0, 0.1)
        assert float(tpu_err.mean()) <= bound, f"{name}: tpu-fp32 {tpu_err.mean()}"
        assert float(bf16_err.mean()) <= bound, f"{name}: tpu-bf16 {bf16_err.mean()}"


def test_greedy_decode_matches_teacher_forcing(llm, reference) -> None:
    """Decode steps (paged caches, PLE and GDN state carried between steps)
    give the same next tokens as one prefill over the same sequence."""
    sequences = reference.sequences
    prompts = [ids[:-DECODE_STEPS] for ids in sequences]
    params = SamplingParams(temperature=0.0, max_tokens=DECODE_STEPS, ignore_eos=True)
    decoded = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    full = [p + list(o.outputs[0].token_ids) for p, o in zip(prompts, decoded)]
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    forced = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in full], params)
    for prompt, ids, out in zip(prompts, full, forced):
        for i in range(len(prompt), len(ids)):
            scores = {t: lp.logprob for t, lp in out.prompt_logprobs[i].items()}
            top = max(scores.values())
            print(f"len={len(prompt)} pos={i} gap={top - scores[ids[i]]:.3e}")
            assert scores[ids[i]] == top


def _prefix_cache_hits(llm) -> float:
    return sum(
        m.value
        for m in llm.get_metrics()
        if m.name == "vllm:prefix_cache_hits" and hasattr(m, "value")
    )


def test_prefix_cache_hit_reproduces_the_miss(llm, reference) -> None:
    """The second pass of a fresh long prompt hits the prefix cache (GDN and PLE
    state restored from cached blocks, QSA keys from the index cache) and
    matches the uncached first pass."""
    # Fresh prompt longer than FIRST_SPARSE_POSITION and spanning multiple KV
    # blocks that no earlier test sent, so the first pass is a true miss.
    fresh_len = max(PROMPT_LENGTHS) - DECODE_STEPS
    long_prompt = [
        torch.randint(
            1, 512, (fresh_len,), generator=torch.Generator().manual_seed(999)
        ).tolist()
    ]
    params = SamplingParams(
        temperature=0.0, max_tokens=DECODE_STEPS, ignore_eos=True, logprobs=0
    )
    hits_before = _prefix_cache_hits(llm)
    first = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in long_prompt], params
    )
    hits_after_miss = _prefix_cache_hits(llm)
    assert hits_after_miss == hits_before
    second = llm.generate(
        [TokensPrompt(prompt_token_ids=p) for p in long_prompt], params
    )
    assert _prefix_cache_hits(llm) > hits_after_miss
    for a, b in zip(first, second):
        assert a.outputs[0].token_ids == b.outputs[0].token_ids
        assert [
            lp[t].logprob
            for t, lp in zip(a.outputs[0].token_ids, a.outputs[0].logprobs)
        ] == [
            lp[t].logprob
            for t, lp in zip(b.outputs[0].token_ids, b.outputs[0].logprobs)
        ]
