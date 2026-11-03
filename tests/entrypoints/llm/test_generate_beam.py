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
"""End-to-end tests for Reflection Beam through vllm.LLM on one TPU chip: the real
config with 2 layers, smaller hidden sizes, TP=8's per-device heads (6 Q, 1 KV), top-8
of 32 experts in 8 groups, the real 512-token window and block 256."""

import math
import weakref

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from vllm_torchtpu.models.vllm.beam.router import split_weight_bf16

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "reflection/Beam-1.0-501B-A23B"
HF_OVERRIDES = {
    "num_hidden_layers": 2,
    "layer_types": ["sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse"],
    "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 256,
    "num_attention_heads": 6, "num_key_value_heads": 1, "num_experts": 32,
}  # fmt: skip
WINDOW, BLOCK = 512, 256
GREEDY = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, logprobs=5)


def _reinit_weights(model) -> None:
    """Seeded N(0, 1/fan_in) weights and unit norms, so logits aren't near-uniform."""
    gen = torch.Generator().manual_seed(0)
    for prefix, module in model.named_modules():
        # TPU linears are stored [in, out], experts [E, in, out], the rest [out, in].
        method = getattr(module, "quant_method", None)
        in_dim = 0 if isinstance(method, UnquantizedLinearMethod) else -1
        for name, param in module.named_parameters(prefix, recurse=False):
            router = name.endswith("mlp.gate.weight")  # One fp32 weight as 3 parts.
            shape = (param.shape[0] // 3, param.shape[1]) if router else param.shape
            fan_in = shape[1] if len(shape) == 3 else shape[in_dim]
            value = torch.randn(shape, generator=gen) / math.sqrt(fan_in)
            value = torch.ones(shape) if "norm" in name else value
            value = split_weight_bf16(value, 3) if router else value
            param.data.copy_(value.to(param.dtype))


@pytest.fixture(scope="module")
def llm():
    with pytest.MonkeyPatch.context() as env:
        # The engine's own process holds the TPU; helpers sent to it need pickle.
        env.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        llm = LLM(
            model=MODEL_NAME,
            load_format="dummy",
            # Required: the router takes bf16 activations only.
            dtype="bfloat16",
            hf_overrides=HF_OVERRIDES,
            max_model_len=1024,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            attention_backend="CUSTOM",
            block_size=BLOCK,
            enable_prefix_caching=True,
        )
        llm.apply_model(_reinit_weights)
        yield weakref.proxy(llm)
        del llm
        cleanup_dist_env_and_memory()


def _describe(model):
    """Runs in the worker: per-layer windows, and the max |logit| of a huge input."""
    windows = [layer.self_attn.attn.sliding_window for layer in model.model.layers]
    norm = model.model.norm.weight
    huge = torch.full((1, norm.shape[0]), 1e3, device=norm.device, dtype=norm.dtype)
    peak = model.compute_logits(huge).abs().max().item()
    return windows, peak, model.final_logit_softcapping


def _cache_misses(worker) -> int:
    stats = torch.tpu._get_cache_stats()  # In the engine process.
    return stats.num_cache_reqs - stats.num_cache_hits


def test_beam_engine_serves_on_tpu(llm: LLM):
    """Windows the sliding layer only, soft-caps the logits and decodes past the window;
    once warm, new lengths in the same buckets compile nothing (first use compiles
    sampler/logprobs programs)."""
    [(windows, peak, softcap)] = llm.apply_model(_describe)
    assert list(windows) == [WINDOW, None] and softcap == 30.0  # The real config's.
    assert 0 < peak <= softcap

    def generate(lengths):
        assert llm.reset_prefix_cache()
        prompts = [{"prompt_token_ids": list(range(3, 3 + n))} for n in lengths]
        return llm.generate(prompts, GREEDY)

    generate((8, 600))  # Warm-up.
    misses = llm.llm_engine.collective_rpc(_cache_misses)
    # Same buckets; 600/601 both cross the window in prefill, but every decode
    # position moves by one, so this run reaches new last positions.
    outputs = generate((9, 601))
    assert [len(o.outputs[0].token_ids) for o in outputs] == [GREEDY.max_tokens] * 2
    assert llm.llm_engine.collective_rpc(_cache_misses) == misses


def _assert_greedy_close(cached, reference, vocab_size: int, atol: float = 0.1) -> None:
    """Each reference pick sits 3+ nats above uniform; tokens match (close log-probs)
    until a near-tie: each pick in the other's top 5, their log-probs within atol."""
    floor = 3.0 - math.log(vocab_size)
    for step, (token, ref) in enumerate(zip(cached.token_ids, reference.token_ids)):
        logprobs, ref_logprobs = cached.logprobs[step], reference.logprobs[step]
        assert ref_logprobs[ref].logprob >= floor, (step, ref_logprobs[ref].logprob)
        if token != ref:
            assert token in ref_logprobs and ref in logprobs, step
            gap = abs(ref_logprobs[ref].logprob - ref_logprobs[token].logprob)
            assert gap < atol, (step, gap)
            return
        assert abs(logprobs[token].logprob - ref_logprobs[token].logprob) < atol, step


def test_beam_prefix_caching_past_window_matches_uncached(llm: LLM):
    """An 800-token prefix hits 3 cached blocks and gives the uncached output."""
    prompt_a = [3 + 11 * i % 1000 for i in range(800)]
    prompt_b = prompt_a + [3 + 13 * i % 1000 for i in range(40)]

    def generate(prompt):
        return llm.generate([{"prompt_token_ids": prompt}], GREEDY)[0]

    assert llm.reset_prefix_cache()
    generate(prompt_a)
    cached = generate(prompt_b)
    assert llm.reset_prefix_cache()
    reference = generate(prompt_b)
    assert (cached.num_cached_tokens, reference.num_cached_tokens) == (3 * BLOCK, 0)
    vocab_size = llm.llm_engine.vllm_config.model_config.hf_config.vocab_size
    _assert_greedy_close(cached.outputs[0], reference.outputs[0], vocab_size)
