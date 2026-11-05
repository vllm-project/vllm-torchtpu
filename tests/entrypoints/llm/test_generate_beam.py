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
"""End-to-end tests for Reflection Beam through vllm.LLM on one TPU chip.

The real config with 2 layers, smaller hidden sizes, TP=8's per-device heads
(6 Q, 1 KV), top-8 of 32 experts in 8 groups, the real 512-token window and
block 256.
"""

import math
import weakref

import pytest
import torch
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod

from vllm_torchtpu.models.vllm.beam.model import RMSNormWeight
from vllm_torchtpu.models.vllm.beam.router import BeamRouter, split_weight_bf16

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "reflection/Beam-1.0-501B-A23B"
HF_OVERRIDES = {
    "num_hidden_layers": 2,
    "layer_types": ["sliding_attention", "full_attention"],
    "mlp_layer_types": ["dense", "sparse"],
    "hidden_size": 256, "intermediate_size": 512, "moe_intermediate_size": 256,
    "num_attention_heads": 6, "num_key_value_heads": 1, "num_experts": 32,
}  # fmt: skip
MAX_MODEL_LEN, BLOCK = 1024, 256
WINDOW, SOFTCAP = 512, 30.0  # The real config's, which the tests pin.
GREEDY = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True, logprobs=5)

# Prompt token ids start past the special tokens and stay below 1003.
FIRST_TOKEN_ID, TOKEN_ID_RANGE = 3, 1000
# Decode-test prompt lengths: one short, one past WINDOW.
SHORT_LEN, PAST_WINDOW_LEN = 8, 600
# The prefix-caching prompt: a prefix past WINDOW, then a fresh suffix.
PREFIX_LEN, SUFFIX_LEN = 800, 40

# Each reference pick beats uniform by this many nats, so logits aren't flat.
MIN_PICK_MARGIN_NATS = 3.0
# Log-prob tolerance between the cached and the uncached run.
LOGPROB_ATOL = 0.1


def _token_ids(length: int, stride: int = 1) -> list[int]:
    return [FIRST_TOKEN_ID + (stride * i) % TOKEN_ID_RANGE for i in range(length)]


def _reinit_weights(model) -> None:
    """Seeded N(0, 1/fan_in) weights and unit norms, so logits aren't near-uniform."""
    generator = torch.Generator().manual_seed(0)
    for module in model.modules():
        # TPU linears are stored [in, out], experts [E, in, out], the rest [out, in].
        is_tpu_linear = isinstance(module, LinearBase) and isinstance(
            module.quant_method, UnquantizedLinearMethod
        )
        in_dim = 0 if is_tpu_linear else -1
        for name, param in module.named_parameters(recurse=False):
            # The router holds its one fp32 weight as stacked bf16 parts.
            is_router_weight = isinstance(module, BeamRouter) and name == "weight"
            shape = param.shape[1:] if is_router_weight else param.shape
            fan_in = shape[1] if len(shape) == 3 else shape[in_dim]
            value = torch.randn(shape, generator=generator) / math.sqrt(fan_in)
            if isinstance(module, RMSNormWeight):
                value = torch.ones(shape)
            elif is_router_weight:
                value = split_weight_bf16(value)
            param.data.copy_(value.to(param.dtype))


@pytest.fixture(scope="module")
def llm():
    with pytest.MonkeyPatch.context() as monkeypatch:
        # The engine's own process holds the TPU; helpers sent to it need pickle.
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        llm = LLM(
            model=MODEL_NAME,
            load_format="dummy",
            # What auto resolves to; pinned, as TPUBeamConfig rejects fp16/fp32.
            dtype="bfloat16",
            hf_overrides=HF_OVERRIDES,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=4,
            max_num_batched_tokens=256,
            attention_backend="CUSTOM",
            block_size=BLOCK,
            enable_prefix_caching=True,
        )
        llm.apply_model(_reinit_weights)
        # pytest caches the fixture, so hand out a weakref.proxy to let the
        # engine be garbage collected.
        yield weakref.proxy(llm)
        del llm
        cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def hf_config(llm):
    return llm.llm_engine.vllm_config.model_config.hf_config


# The helpers below run in the engine process.


def _layer_windows(model) -> list[int | None]:
    return [layer.self_attn.attn.sliding_window for layer in model.model.layers]


def _rope_caches(model) -> list:
    """Per layer, the RoPE cache's [rows, dtype], or None on a NoPE layer."""
    ropes = [layer.self_attn.rotary_emb for layer in model.model.layers]
    return [
        None
        if rope is None
        else [rope.cos_sin_cache.shape[0], rope.cos_sin_cache.dtype]
        for rope in ropes
    ]


def _softcap_and_peak_logit(model) -> list[float]:
    """The model's logit softcap and the max |logit| of a huge final hidden state."""
    norm = model.model.norm.weight
    huge = torch.full((1, norm.shape[0]), 1e3, device=norm.device, dtype=norm.dtype)
    peak = model.compute_logits(huge).abs().max().item()
    return [model.final_logit_softcapping, peak]


def _cache_misses(worker) -> int:
    stats = torch.tpu._get_cache_stats()
    return stats.num_cache_reqs - stats.num_cache_hits


def test_beam_windows_only_the_sliding_layers(llm: LLM):
    [windows] = llm.apply_model(_layer_windows)
    assert list(windows) == [WINDOW, None]


def test_beam_rope_cache_covers_max_model_len(llm: LLM):
    """Sized to max_model_len, not the 1M max_position_embeddings; NoPE has none."""
    [ropes] = llm.apply_model(_rope_caches)
    assert list(ropes) == [[MAX_MODEL_LEN, torch.float32], None]


def test_beam_softcaps_the_logits(llm: LLM):
    [(softcap, peak)] = llm.apply_model(_softcap_and_peak_logit)
    assert softcap == SOFTCAP
    assert 0 < peak <= SOFTCAP


def test_beam_decodes_past_the_window_without_recompiling(llm: LLM):
    """Once warm, new lengths in the same buckets compile nothing.

    The warm-up's first use compiles the sampler and logprobs programs.
    """

    def generate(lengths):
        assert llm.reset_prefix_cache()
        prompts = [{"prompt_token_ids": _token_ids(n)} for n in lengths]
        return llm.generate(prompts, GREEDY)

    generate((SHORT_LEN, PAST_WINDOW_LEN))  # Warm-up.
    misses = llm.llm_engine.collective_rpc(_cache_misses)
    # Same buckets; both long prompts cross the window in prefill, but every
    # decode position moves by one, so this run reaches new last positions.
    outputs = generate((SHORT_LEN + 1, PAST_WINDOW_LEN + 1))
    assert [len(o.outputs[0].token_ids) for o in outputs] == [GREEDY.max_tokens] * 2
    assert llm.llm_engine.collective_rpc(_cache_misses) == misses


def _first_divergence(tokens, ref_tokens) -> int:
    """The first step where the two greedy runs differ, or the run length."""
    steps = zip(tokens, ref_tokens, strict=True)
    return next((i for i, (a, b) in enumerate(steps) if a != b), len(ref_tokens))


def _assert_near_tie(cached, reference, step: int) -> None:
    """Each run's pick is in the other's top 5, with log-probs within atol."""
    token, ref_token = cached.token_ids[step], reference.token_ids[step]
    cached_logprobs, ref_logprobs = cached.logprobs[step], reference.logprobs[step]
    assert token in ref_logprobs and ref_token in cached_logprobs, step
    for candidate in (token, ref_token):
        gap = abs(cached_logprobs[candidate].logprob - ref_logprobs[candidate].logprob)
        assert gap < LOGPROB_ATOL, (step, candidate, gap)
    tie_gap = abs(ref_logprobs[ref_token].logprob - ref_logprobs[token].logprob)
    assert tie_gap < LOGPROB_ATOL, (step, tie_gap)


def _assert_greedy_close(cached, reference, vocab_size: int) -> None:
    """Asserts the cached run matches the reference run up to a near-tie.

    Both pick the same tokens, with log-probs within LOGPROB_ATOL, until they
    first diverge, and that step must be a near-tie. Up to there, every
    reference pick beats uniform by MIN_PICK_MARGIN_NATS.
    """
    num_steps = len(reference.token_ids)
    diverged_at = _first_divergence(cached.token_ids, reference.token_ids)
    floor = MIN_PICK_MARGIN_NATS - math.log(vocab_size)
    for step in range(min(diverged_at + 1, num_steps)):
        pick = reference.logprobs[step][reference.token_ids[step]].logprob
        assert pick >= floor, (step, pick)
    for step in range(diverged_at):
        token = reference.token_ids[step]
        gap = abs(
            cached.logprobs[step][token].logprob
            - reference.logprobs[step][token].logprob
        )
        assert gap < LOGPROB_ATOL, (step, gap)
    if diverged_at < num_steps:
        _assert_near_tie(cached, reference, diverged_at)


def test_beam_prefix_caching_past_window_matches_uncached(llm: LLM, hf_config):
    """A prefix past the window hits its full cached blocks, as if uncached."""
    prompt = _token_ids(PREFIX_LEN, stride=11) + _token_ids(SUFFIX_LEN, stride=13)

    def generate(prompt_token_ids):
        return llm.generate([{"prompt_token_ids": prompt_token_ids}], GREEDY)[0]

    assert llm.reset_prefix_cache()
    generate(prompt[:PREFIX_LEN])
    cached = generate(prompt)
    assert llm.reset_prefix_cache()
    reference = generate(prompt)
    assert cached.num_cached_tokens == PREFIX_LEN // BLOCK * BLOCK
    assert reference.num_cached_tokens == 0
    _assert_greedy_close(cached.outputs[0], reference.outputs[0], hf_config.vocab_size)
