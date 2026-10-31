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
"""Multi-LoRA end to end on one chip, with Qwen3-0.6B and random adapters.

Outputs are compared token for token only between runs with the same config
and batch shape, since programs for different buckets, and eager versus
compiled mode, round differently. Other runs are compared by distribution, or
may pick a different token where the top two are nearly tied. Correctness
against independent math uses the adapter merged into the base weights,
compared by distribution.
"""

import contextlib
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest

pytestmark = [pytest.mark.tpu_test, pytest.mark.nightly]

MODEL_NAME = "Qwen/Qwen3-0.6B"
RANK = 8
ENGINE_ARGS = {
    "model": MODEL_NAME,
    "max_model_len": 256,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 64,
    "gpu_memory_utilization": 0.6,
}
LORA_ARGS = {"enable_lora": True, "max_loras": 2, "max_lora_rank": RANK}

PROMPTS = [
    "The capital of France is",
    "Hello, my name is",
    "The future of AI is",
    "1, 2, 3, 4,",
    "The best way to learn is",
    "Water boils at",
    "My favorite food is",
    "In 1492, Columbus",
]
GREEDY = SamplingParams(temperature=0.0, max_tokens=8)
ORACLE_PROMPTS = PROMPTS + [
    "The largest planet is",
    "To make tea, first",
    "The opposite of hot is",
    "Once upon a time",
    "The chemical symbol for gold is",
    "A good programmer",
    "The sun rises in the",
    "Monday comes after",
]
FIRST_TOKEN = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
# Largest gap between the top two logprobs at which two programs that round
# differently may pick different greedy tokens. The lm_head adapter's logits
# lie in 32-64, where a bf16 step is 0.25, so this allows 8 steps against the
# 1-2 that rounding moves them. Dropping the delta changes the first token of
# every prompt, mostly with a larger gap.
NEAR_TIE_LOGPROB_GAP = 2.0


@pytest.fixture(scope="module", autouse=True)
def _allow_callable_rpcs():
    """collective_rpc can only send a function to the worker with this set."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        yield


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    """A request per random adapter, each with its own id, as OpenRL sends a
    new id for every saved adapter."""
    root = tmp_path_factory.mktemp("adapters")
    names = ["a", "b", "c", "swap0", "swap1", "swap2", "pair0", "pair1"]
    requests = {
        name: LoRARequest(name, lora_id, _write_adapter(root / name, seed=lora_id))
        for lora_id, name in enumerate(names, start=1)
    }
    lm_head_path = _write_adapter(root / "lm_head", seed=99, lm_head=True)
    requests["lm_head"] = LoRARequest("lm_head", 99, lm_head_path)
    return requests


def _write_adapter(path, seed, lm_head=False):
    """Writes a random rank-8 PEFT adapter, as OpenRL's validation script does:
    on every attention and MLP projection, or on lm_head alone. B is non-zero,
    unlike a fresh PEFT init, so that the adapter changes the output."""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    if lm_head:
        shapes = {"lm_head": (config.hidden_size, config.vocab_size)}
        # Large enough that the lm_head delta alone changes greedy tokens.
        b_scale = 0.5
    else:
        shapes = {
            f"model.layers.{layer}.{name}": dims
            for layer in range(config.num_hidden_layers)
            for name, dims in _projection_shapes(config).items()
        }
        b_scale = 0.02
    gen = torch.Generator().manual_seed(seed)
    weights = {}
    for module, (in_features, out_features) in shapes.items():
        prefix = f"base_model.model.{module}"
        weights[f"{prefix}.lora_A.weight"] = (
            torch.randn(RANK, in_features, generator=gen) / in_features**0.5
        )
        weights[f"{prefix}.lora_B.weight"] = (
            torch.randn(out_features, RANK, generator=gen) * b_scale
        )
    path.mkdir()
    save_file(weights, path / "adapter_model.safetensors")
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": RANK,
        "lora_alpha": 4 * RANK,
        "target_modules": sorted({module.rsplit(".", 1)[-1] for module in shapes}),
    }
    (path / "adapter_config.json").write_text(json.dumps(adapter_config))
    return str(path)


def _projection_shapes(config):
    """(in_features, out_features) of each projection that OpenRL trains."""
    hidden, mlp = config.hidden_size, config.intermediate_size
    q = config.num_attention_heads * config.head_dim
    kv = config.num_key_value_heads * config.head_dim
    return {
        "self_attn.q_proj": (hidden, q),
        "self_attn.k_proj": (hidden, kv),
        "self_attn.v_proj": (hidden, kv),
        "self_attn.o_proj": (q, hidden),
        "mlp.gate_proj": (hidden, mlp),
        "mlp.up_proj": (hidden, mlp),
        "mlp.down_proj": (mlp, hidden),
    }


@pytest.fixture(scope="module")
def merged_model(adapters, tmp_path_factory):
    """The base checkpoint with adapter "a" folded into its weights,
    W += (alpha / r) B @ A, computed without vLLM's LoRA path."""
    base = Path(
        snapshot_download(
            MODEL_NAME, allow_patterns=["*.json", "*.safetensors", "*.txt"]
        )
    )
    out = tmp_path_factory.mktemp("merged")
    for path in base.iterdir():
        if path.name != "model.safetensors":
            # copyfile rather than copy: the cached files are read-only.
            shutil.copyfile(path, out / path.name)
    adapter = Path(adapters["a"].lora_path)
    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    scaling = adapter_config["lora_alpha"] / adapter_config["r"]
    lora = load_file(adapter / "adapter_model.safetensors")
    weights = load_file(base / "model.safetensors")
    for key, lora_a in lora.items():
        if not key.endswith(".lora_A.weight"):
            continue
        lora_b = lora[key.replace("lora_A", "lora_B")]
        name = key.removeprefix("base_model.model.").replace(".lora_A", "")
        merged = weights[name].float() + scaling * (lora_b @ lora_a)
        weights[name] = merged.to(weights[name].dtype)
    save_file(weights, out / "model.safetensors", metadata={"format": "pt"})
    return str(out)


@contextlib.contextmanager
def _engine(**overrides):
    """Runs one engine at a time, since the chip holds only one."""
    llm = LLM(**{**ENGINE_ARGS, **overrides})
    try:
        yield llm
    finally:
        llm.llm_engine.engine_core.shutdown()
        cleanup_dist_env_and_memory()


def _greedy(llm, lora_request):
    outputs = llm.generate(PROMPTS, GREEDY, lora_request=lora_request, use_tqdm=False)
    return [list(output.outputs[0].token_ids) for output in outputs]


def _greedy_steps(llm, lora_request, logprobs):
    """Greedy tokens of each prompt, with the top logprobs of each step."""
    params = SamplingParams(temperature=0.0, max_tokens=8, logprobs=logprobs)
    outputs = llm.generate(PROMPTS, params, lora_request=lora_request, use_tqdm=False)
    return [
        (
            list(o.outputs[0].token_ids),
            [
                {t: lp.logprob for t, lp in step.items()}
                for step in o.outputs[0].logprobs
            ],
        )
        for o in outputs
    ]


def _first_token_distributions(llm, lora_request=None):
    outputs = llm.generate(
        ORACLE_PROMPTS, FIRST_TOKEN, lora_request=lora_request, use_tqdm=False
    )
    return [
        {token: math.exp(lp.logprob) for token, lp in o.outputs[0].logprobs[0].items()}
        for o in outputs
    ]


def _total_variation(x, y):
    """A token outside one side's top 20 counts as 0 there."""
    return 0.5 * sum(abs(x.get(t, 0.0) - y.get(t, 0.0)) for t in x.keys() | y.keys())


def _mean_total_variation(p, q):
    """Mean over prompts."""
    distances = [_total_variation(x, y) for x, y in zip(p, q)]
    return sum(distances) / len(distances)


def _aligned_mean_total_variation(p, q):
    """Mean over the steps of each prompt up to the first one whose greedy
    tokens differ; after it, the two runs continue from different text."""
    distances = []
    for (p_tokens, p_steps), (q_tokens, q_steps) in zip(p, q):
        for p_token, q_token, x, y in zip(p_tokens, q_tokens, p_steps, q_steps):
            distances.append(
                _total_variation(
                    {t: math.exp(lp) for t, lp in x.items()},
                    {t: math.exp(lp) for t, lp in y.items()},
                )
            )
            if p_token != q_token:
                break
    return sum(distances) / len(distances)


def _assert_same_tokens_but_for_a_near_tie(tokens, expected, steps, prompt):
    """tokens follow expected until a step where the other top token is
    nearly tied with it; after that step, the text differs."""
    for token, expected_token, logprobs in zip(tokens, expected, steps):
        if token != expected_token:
            assert token in logprobs, prompt
            gap = logprobs[expected_token] - logprobs[token]
            assert gap <= NEAR_TIE_LOGPROB_GAP, (prompt, gap)
            return


def _xla_cache_entries(worker):
    return len(torch.tpu._get_cache_stats().per_entry_stats)


def _dynamo_graphs(worker):
    return torch._dynamo.utils.counters["stats"]["unique_graphs"]


def _assert_mixed_batch_matches_uniform_batches(llm, adapters):
    """Rows of one batch that use different adapters, or none, come out as
    they do in batches of the same shape that use a single one."""
    cycle = [adapters["a"], adapters["b"], None]
    uniform = [_greedy(llm, request) for request in cycle]
    mixed = _greedy(llm, [cycle[i % len(cycle)] for i in range(len(PROMPTS))])
    for i, tokens in enumerate(mixed):
        assert tokens == uniform[i % len(cycle)][i], PROMPTS[i]


@pytest.fixture(scope="class")
def lora_llm():
    with _engine(**LORA_ARGS) as llm:
        yield llm


class TestLoRAEngine:
    def test_new_adapters_swap_in_without_compiling(self, lora_llm, adapters):
        """OpenRL sends a new adapter id at every step. Filling a free slot,
        replacing an adapter in a full set of slots, and swapping in more new
        ids must compile nothing, also through the add_lora RPC, which runs
        with grad enabled."""
        # Without this, a test that runs earlier on the shared engine would
        # fill the slots and leave filling a free slot untested.
        assert not lora_llm.llm_engine.list_loras()
        base = _greedy(lora_llm, None)
        entries = lora_llm.collective_rpc(_xla_cache_entries)
        assert _greedy(lora_llm, adapters["a"]) != base
        # The first swap still traces the slot write once more: Dynamo guards
        # on the torch thread count, which the worker lowers after warm-up.
        graphs = lora_llm.collective_rpc(_dynamo_graphs)

        # "b" fills the second slot and "c" replaces "a".
        for name in ("b", "c", "swap0", "swap1", "swap2"):
            assert _greedy(lora_llm, adapters[name]) != base, name
        rpc = LoRARequest("rpc_swap", 60, adapters["a"].lora_path)
        assert lora_llm.llm_engine.add_lora(rpc)
        assert lora_llm.collective_rpc(_xla_cache_entries) == entries
        assert lora_llm.collective_rpc(_dynamo_graphs) == graphs

    def test_one_batch_brings_in_two_new_adapters(self, lora_llm, adapters):
        """Both adapters are activated in the same step, one right after the
        other, and each row still matches a batch that uses its adapter
        alone."""
        pair = [adapters["pair0"], adapters["pair1"]]
        loaded = lora_llm.llm_engine.list_loras()
        assert not loaded & {request.lora_int_id for request in pair}

        mixed = _greedy(lora_llm, [pair[i % 2] for i in range(len(PROMPTS))])
        uniform = [_greedy(lora_llm, request) for request in pair]
        for i, tokens in enumerate(mixed):
            assert tokens == uniform[i % 2][i], PROMPTS[i]

    def test_adapters_change_the_output(self, lora_llm, adapters):
        base = _greedy(lora_llm, None)
        a = _greedy(lora_llm, adapters["a"])
        b = _greedy(lora_llm, adapters["b"])

        assert a != base and b != base and a != b

    def test_mixed_batch_matches_uniform_batches(self, lora_llm, adapters):
        _assert_mixed_batch_matches_uniform_batches(lora_llm, adapters)

    def test_evicted_adapter_reloads_with_the_same_output(self, lora_llm, adapters):
        before = _greedy(lora_llm, adapters["a"])
        _greedy(lora_llm, adapters["b"])
        _greedy(lora_llm, adapters["c"])
        # The CPU cache holds max_loras adapters, so "a" is now gone from it.
        assert adapters["a"].lora_int_id not in lora_llm.llm_engine.list_loras()

        assert _greedy(lora_llm, adapters["a"]) == before

    def test_adapter_rpcs(self, lora_llm, adapters):
        engine = lora_llm.llm_engine
        request = LoRARequest("rpc", 50, adapters["a"].lora_path)

        assert engine.add_lora(request)
        assert request.lora_int_id in engine.list_loras()
        assert engine.pin_lora(request.lora_int_id)
        assert engine.remove_lora(request.lora_int_id)
        assert request.lora_int_id not in engine.list_loras()


@pytest.fixture(scope="module")
def lora_outputs(adapters):
    with _engine(**LORA_ARGS) as llm:
        return {
            "base": _greedy_steps(llm, None, logprobs=20),
            # The top two, to tell a near tie in the fast path test.
            "lm_head": _greedy_steps(llm, adapters["lm_head"], logprobs=2),
            "oracle": _first_token_distributions(llm, adapters["a"]),
        }


@pytest.fixture(scope="module")
def base_outputs():
    with _engine() as llm:
        return {
            "base": _greedy_steps(llm, None, logprobs=20),
            "oracle": _first_token_distributions(llm),
        }


def test_requests_without_an_adapter_match_the_base_model(lora_outputs, base_outputs):
    """Without an adapter, the LoRA engine must stay much closer to the base
    model than an adapter moves it. Its graph also holds the LoRA ops, so it
    may round differently; a request that picks up an adapter's delta puts the
    ratio near 1."""
    noise = _aligned_mean_total_variation(lora_outputs["base"], base_outputs["base"])
    adapter_effect = _mean_total_variation(
        lora_outputs["oracle"], base_outputs["oracle"]
    )
    assert noise < 0.5 * adapter_effect, (noise, adapter_effect)


def test_lora_matches_the_adapter_merged_into_the_weights(
    lora_outputs, base_outputs, merged_model
):
    """The LoRA output sits much closer to the merged model than the adapter
    moves it from the base model. A LoRA path that drops or misplaces the
    delta puts the ratio near 1 or above; bf16 noise alone keeps it near 0.2."""
    with _engine(model=merged_model) as llm:
        merged = _first_token_distributions(llm)

    lora_gap = _mean_total_variation(lora_outputs["oracle"], merged)
    adapter_effect = _mean_total_variation(merged, base_outputs["oracle"])
    assert lora_gap < 0.5 * adapter_effect, (lora_gap, adapter_effect)


def _count_fast_path_steps(worker):
    runner = worker.model_runner
    decide = runner._greedy_top_tokens_this_step
    runner.fast_path_steps = 0

    def counted():
        took = decide()
        runner.fast_path_steps += took
        return took

    runner._greedy_top_tokens_this_step = counted


def test_greedy_top_tokens_keep_the_lm_head_delta(monkeypatch, adapters, lora_outputs):
    # The lm_head delta decides tokens here, so dropping it would show.
    lm_head = lora_outputs["lm_head"]
    assert [tokens for tokens, _ in lm_head] != [
        tokens for tokens, _ in lora_outputs["base"]
    ]
    monkeypatch.setenv("TPU_GREEDY_TOP_TOKENS", "1")

    with _engine(**LORA_ARGS) as llm:
        llm.collective_rpc(_count_fast_path_steps)
        tokens = _greedy(llm, adapters["lm_head"])
        fast_path_steps = llm.collective_rpc(
            lambda worker: worker.model_runner.fast_path_steps
        )

    # The fast path compiles the argmax with the logits, so it may round them
    # differently from the full-logits path.
    for prompt, fast, (expected, steps) in zip(PROMPTS, tokens, lm_head):
        _assert_same_tokens_but_for_a_near_tie(fast, expected, steps, prompt)
    assert fast_path_steps[0] > 0


def test_one_slot_serves_adapters_in_turn(adapters):
    with _engine(**{**LORA_ARGS, "max_loras": 1}) as llm:
        base = _greedy(llm, None)
        first = _greedy(llm, adapters["a"])
        other = _greedy(llm, adapters["b"])
        again = _greedy(llm, adapters["a"])

    assert first == again
    assert first != base and other != base and other != first


def test_eager_mode_applies_adapters(adapters):
    with _engine(**LORA_ARGS, enforce_eager=True) as llm:
        assert _greedy(llm, adapters["a"]) != _greedy(llm, None)
        _assert_mixed_batch_matches_uniform_batches(llm, adapters)
