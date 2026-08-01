# Copyright 2025 Google LLC
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

from __future__ import annotations

import random
import string
import time

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.metrics.reader import Counter

from vllm_torchtpu import tpu_info

# Heavy Llama-3.1-8B spec-decode integration suite; presubmit deselects it via
# `-m "not nightly"`, full coverage runs in the nightly workflow.
pytestmark = pytest.mark.nightly


def _is_v7x():
    return (tpu_info.get_tpu_type() or "").startswith("tpu7x")


def _get_tensor_parallel_size():
    if _is_v7x():
        return 2
    return 1


def get_ngram_test_prompts():
    num_prompts = 10
    prompts = []
    for _ in range(num_prompts):
        w = random.choice(list(string.ascii_lowercase))
        prompts.append(
            f"Keep repeating: {w} {w} {w} {w} {w} {w} {w} {w} {w} {w}")
    return prompts


def get_eagle3_test_prompts():
    num_prompts = 10
    return [
        "Predict the continuation of this sequence: 1 2 3 4 5 6 7 8"
        for _ in range(num_prompts)
    ]


def get_dflash_test_prompts():
    num_prompts = 10
    return [
        "Predict the continuation of this sequence: 1 2 3 4 5 6 7 8"
        for _ in range(num_prompts)
    ]


def get_test_prompts(speculative_config: dict):
    method = speculative_config["method"]
    if method == "ngram":
        return get_ngram_test_prompts()
    if method == "eagle3":
        return get_eagle3_test_prompts()
    if method == "dflash":
        return get_dflash_test_prompts()
    raise NotImplementedError(f"{method} is not supported yet.")


def _make_sampling_config(temperature: float = 0) -> SamplingParams:
    return SamplingParams(temperature=temperature,
                          max_tokens=16,
                          ignore_eos=True,
                          repetition_penalty=1,
                          frequency_penalty=0,
                          presence_penalty=0,
                          min_p=0,
                          logprobs=None)


@pytest.fixture
def sampling_config():
    return _make_sampling_config()


@pytest.fixture
def model_name():
    return "Qwen/Qwen3-0.6B"


def _test_correctness_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
    speculative_config: dict,
    max_num_seqs: int = 4,
    async_scheduling: bool | None = None,
    extra_kwargs: dict | None = None,
):
    with monkeypatch.context():
        test_prompts = get_test_prompts(speculative_config)

        kwargs = {
            "max_model_len": 256,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "async_scheduling": async_scheduling,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        # 1. Reference LLM Run
        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        ref_llm.llm_engine.engine_core.shutdown()
        del ref_llm
        cleanup_dist_env_and_memory()

        # Waiting for TPUs to be fully released.
        time.sleep(15)

        # 2. Speculative LLM Run
        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)
        try:
            spec_outputs = spec_llm.generate(test_prompts, sampling_config)

            matches = 0
            misses = 0
            for ref_output, spec_output in zip(ref_outputs, spec_outputs):
                if ref_output.outputs[0].text == spec_output.outputs[0].text:
                    matches += 1
                else:
                    misses += 1
                    print(f"ref_output: {ref_output.outputs[0].text}")
                    print(f"spec_output: {spec_output.outputs[0].text}")

            assert misses == 0
        finally:
            spec_llm.llm_engine.engine_core.shutdown()
            del spec_llm
            cleanup_dist_env_and_memory()


@pytest.mark.timeout(1800)
def test_ngram_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    _test_correctness_helper(
        monkeypatch, sampling_config, model_name, {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 3,
            "num_speculative_tokens": 3,
        })


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
@pytest.mark.parametrize(
    "max_num_seqs", [pytest.param(1, id="bs1"),
                     pytest.param(10, id="bs10")])
def test_eagle3_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
):
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"

    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        model_name,
        {
            "method": "eagle3",
            "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 3,
            "draft_tensor_parallel_size": 1,
        },
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
@pytest.mark.parametrize(
    "max_num_seqs", [pytest.param(1, id="bs1"),
                     pytest.param(10, id="bs10")])
def test_dflash_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
):
    model_name = "Qwen/Qwen3-4B"
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")

    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        model_name,
        {
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 15,
            "draft_tensor_parallel_size": 1,
        },
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
        extra_kwargs={"gpu_memory_utilization": 0.6},
    )


def _test_performance_helper(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    speculative_config: dict,
    min_acceptance_rate: float,
    max_num_seqs: int = 4,
    model_name: str = "Qwen/Qwen3-0.6B",
    async_scheduling: bool | None = None,
    extra_kwargs: dict | None = None,
):
    with monkeypatch.context():
        test_prompts = get_test_prompts(speculative_config)

        kwargs = {
            "max_model_len": 256,
            "max_num_seqs": max_num_seqs,
            "tensor_parallel_size": _get_tensor_parallel_size(),
            "enable_prefix_caching": False,
            "async_scheduling": async_scheduling,
            "disable_log_stats": False,
        }
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)

        spec_llm.generate(test_prompts, sampling_config)

        metrics = spec_llm.get_metrics()
        num_draft_tokens = num_accepted_tokens = 0
        acceptance_rate = 0.0
        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        if num_draft_tokens > 0:
            acceptance_rate = num_accepted_tokens / num_draft_tokens
            print(f"Acceptance rate: {acceptance_rate:.2%}")
            print("num_accepted_tokens:" + str(num_accepted_tokens))
            print("num_draft_tokens:" + str(num_draft_tokens))

        spec_llm.llm_engine.engine_core.shutdown()
        del spec_llm
        cleanup_dist_env_and_memory()

        assert num_draft_tokens > 0, "Draft tokens should be greater than 0."
        assert acceptance_rate >= min_acceptance_rate, f"Expected at least {min_acceptance_rate:.2%} acceptance rate for {speculative_config['method']}, got {acceptance_rate:.2%}"


@pytest.mark.timeout(1800)
def test_ngram_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    _test_performance_helper(monkeypatch,
                             sampling_config, {
                                 "method": "ngram",
                                 "prompt_lookup_max": 2,
                                 "prompt_lookup_min": 2,
                                 "num_speculative_tokens": 4,
                             },
                             min_acceptance_rate=0.85)


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
@pytest.mark.parametrize(
    "max_num_seqs", [pytest.param(1, id="bs1"),
                     pytest.param(10, id="bs10")])
@pytest.mark.parametrize(
    "temperature, min_acceptance_rate",
    # Non-greedy accepts fewer drafts than greedy (the target samples instead
    # of taking the argmax), so it gets a lower floor. The floor is a
    # regression guard against a broken non-greedy/async pipeline (acceptance
    # collapsing to ~0, hangs, or empty output), not a quality target. The run
    # is deterministic: sampling draws come from the runner's seeded generator
    # (model_config.seed, default 0), so the rate is reproducible run-to-run.
    [
        pytest.param(0.0, 0.75, id="greedy"),
        pytest.param(0.7, 0.3, id="non_greedy")
    ])
def test_eagle3_performance(
    monkeypatch: pytest.MonkeyPatch,
    max_num_seqs: int,
    async_scheduling: bool,
    temperature: float,
    min_acceptance_rate: float,
):
    _test_performance_helper(
        monkeypatch,
        _make_sampling_config(temperature),
        {
            "method": "eagle3",
            "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
            "num_speculative_tokens": 2,
            "draft_tensor_parallel_size": 1,
        },
        min_acceptance_rate=min_acceptance_rate,
        max_num_seqs=max_num_seqs,
        model_name="NousResearch/Meta-Llama-3.1-8B-Instruct",
        async_scheduling=async_scheduling,
    )


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
@pytest.mark.parametrize(
    "max_num_seqs, min_acceptance_rate",
    # The floor is a collapse guard (acceptance ~0, hangs), not a quality
    # target. Nightly v7x/TP2 runs are deterministic per batch size: bs1
    # accepts exactly one draft token per step (80/1200 = 6.67%), bs4 lands
    # at ~10.5-10.8%, so bs1 gets a lower floor.
    [pytest.param(1, 0.05, id="bs1"),
     pytest.param(4, 0.08, id="bs4")])
def test_dflash_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    min_acceptance_rate: float,
    async_scheduling: bool,
):
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")

    _test_performance_helper(
        monkeypatch,
        sampling_config,
        {
            "method": "dflash",
            "model": "z-lab/Qwen3-4B-DFlash-b16",
            "num_speculative_tokens": 15,
            "draft_tensor_parallel_size": 1,
        },
        min_acceptance_rate=min_acceptance_rate,
        max_num_seqs=max_num_seqs,
        model_name="Qwen/Qwen3-4B",
        async_scheduling=async_scheduling,
        extra_kwargs={"gpu_memory_utilization": 0.6},
    )


def test_sd_supports_non_greedy(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str = "Qwen/Qwen3-0.6B",
):
    # Non-greedy (temperature > 0) sampling is now supported under
    # speculative decoding on TPU; generation should succeed. Rejection
    # sampling is stochastic, so we assert non-empty output (a smoke check)
    # rather than exact-match against a reference. ngram is sync-only here
    # (vLLM rejects ngram + async_scheduling at config time); the eagle3 +
    # async + non-greedy path is covered by test_eagle3_performance.
    with monkeypatch.context():
        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=False,
        )
        try:
            prompts = get_ngram_test_prompts()
            non_greedy = SamplingParams(temperature=0.7)
            outputs = spec_llm.generate(prompts, non_greedy)
            assert len(outputs) == len(prompts)
            for output in outputs:
                assert output.outputs[0].text, \
                    "non-greedy spec decode produced empty output"
        finally:
            spec_llm.llm_engine.engine_core.shutdown()
            del spec_llm
            cleanup_dist_env_and_memory()


def _force_multi_chunk_cap(self, cap: int = 2):
    runner = self.model_runner
    runner.num_reqs_max_model_len = cap
    if runner.num_reqs_most_model_len is not None:
        runner.num_reqs_most_model_len = cap


@pytest.mark.timeout(1800)
def test_sd_correctness_greedy_multi_chunk(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    model_name: str,
):
    with monkeypatch.context() as mp:
        # collective_rpc below sends a Python function to TPU workers,
        # which requires pickle fallback in vLLM's IPC encoder.
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        test_prompts = get_ngram_test_prompts()
        kwargs = dict(
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=False,
        )

        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        ref_llm.llm_engine.engine_core.shutdown()
        del ref_llm
        cleanup_dist_env_and_memory()
        time.sleep(15)

        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "ngram",
                "prompt_lookup_max": 5,
                "prompt_lookup_min": 3,
                "num_speculative_tokens": 3,
            },
            **kwargs,
        )
        # Force the runner's while-loop in execute_model to iterate at
        # least twice for any batch larger than 2.
        spec_llm.llm_engine.collective_rpc(_force_multi_chunk_cap, args=(2, ))

        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        misses = 0
        for ref, spec in zip(ref_outputs, spec_outputs):
            if ref.outputs[0].text != spec.outputs[0].text:
                misses += 1
                print(f"ref:  {ref.outputs[0].text}")
                print(f"spec: {spec.outputs[0].text}")
        assert misses == 0

        spec_llm.llm_engine.engine_core.shutdown()
        del spec_llm
        cleanup_dist_env_and_memory()


@pytest.mark.timeout(1200)
def test_eagle3_correctness_greedy_multi_chunk(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """Force the runner to split each batch into >1 chunk and verify the
    chunked eagle3 draft path:
      (a) never changes greedy output vs a no-spec reference (chunking must
          not corrupt results), and
      (b) still gets draft tokens accepted (the chunked drafter actually
          works, rather than silently falling back to target tokens).

    Unlike the ngram multi-chunk test, eagle3 exercises the per-chunk draft
    propose()/forward path. The cap is shrunk via collective_rpc after
    capture_model precompiled at the original cap; as with the ngram
    multi-chunk test, the attention metadata reaches the kernel via the
    forward context rather than as a traced input, so no new torch.compile
    graph is expected at runtime."""
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    speculative_config = {
        "method": "eagle3",
        "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 3,
        "draft_tensor_parallel_size": 1,
    }
    with monkeypatch.context() as mp:
        # collective_rpc ships a Python function to the TPU workers, which
        # needs pickle fallback in vLLM's IPC encoder.
        mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        test_prompts = get_eagle3_test_prompts()
        kwargs = dict(
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            enable_prefix_caching=False,
            async_scheduling=False,
            disable_log_stats=False,
        )

        # 1. Reference run (no speculation).
        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        ref_llm.llm_engine.engine_core.shutdown()
        del ref_llm
        cleanup_dist_env_and_memory()
        time.sleep(15)

        # 2. Speculative run, forced multi-chunk.
        spec_llm = LLM(model=model_name,
                       speculative_config=speculative_config,
                       **kwargs)
        # cap=2 with max_num_seqs=4 -> any batch with >2 reqs chunks, so the
        # draft propose() runs its per-chunk loop for >=2 chunks.
        spec_llm.llm_engine.collective_rpc(_force_multi_chunk_cap, args=(2, ))
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        # (a) Chunked drafting must not change greedy output.
        misses = 0
        for ref, spec in zip(ref_outputs, spec_outputs):
            if ref.outputs[0].text != spec.outputs[0].text:
                misses += 1
                print(f"ref:  {ref.outputs[0].text}")
                print(f"spec: {spec.outputs[0].text}")
        assert misses == 0

        # (b) The chunked draft actually proposed and got tokens accepted.
        num_draft_tokens = num_accepted_tokens = 0
        for metric in spec_llm.get_metrics():
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        print(f"multi-chunk eagle3: accepted={num_accepted_tokens} "
              f"drafted={num_draft_tokens}")
        assert num_draft_tokens > 0, "no draft tokens produced under chunking"
        assert num_accepted_tokens > 0, \
            "no draft tokens accepted under chunking"

        spec_llm.llm_engine.engine_core.shutdown()
        del spec_llm
        cleanup_dist_env_and_memory()


@pytest.mark.skipif(_get_tensor_parallel_size() < 2,
                    reason="sharded draft (draft_tp == target_tp) needs TP>1")
@pytest.mark.timeout(1200)
def test_eagle3_sharded_draft(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """Sharded eagle3 draft: draft_tensor_parallel_size == target tp.
    Verifies the sharded draft (a) doesn't change greedy output vs a no-spec reference
    (b) actually proposes + accepts drafts.
    """
    tp = _get_tensor_parallel_size()
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    with monkeypatch.context():

        test_prompts = get_eagle3_test_prompts()
        kwargs = dict(
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=tp,
            enable_prefix_caching=False,
            # Spec decode + async scheduling isn't supported on TPU yet; the
            # other eagle3 tests set this too. This test exercises the sharded
            # DRAFT, orthogonal to the scheduling mode.
            async_scheduling=False,
            disable_log_stats=False,
        )

        # 1. Reference run (no speculation).
        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, sampling_config)
        ref_llm.llm_engine.engine_core.shutdown()
        del ref_llm
        cleanup_dist_env_and_memory()
        time.sleep(15)

        # 2. Sharded eagle3 draft (draft_tp == target tp).
        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "eagle3",
                "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                "num_speculative_tokens": 3,
                "draft_tensor_parallel_size": tp,  # == target tp -> SHARDED
            },
            **kwargs,
        )
        spec_outputs = spec_llm.generate(test_prompts, sampling_config)

        # (a) Sharded drafting must not change greedy output.
        misses = 0
        for ref, spec in zip(ref_outputs, spec_outputs):
            if ref.outputs[0].text != spec.outputs[0].text:
                misses += 1
                print(f"ref:  {ref.outputs[0].text}")
                print(f"spec: {spec.outputs[0].text}")
        assert misses == 0

        # (b) The sharded draft actually proposed + got tokens accepted.
        num_draft_tokens = num_accepted_tokens = 0
        for metric in spec_llm.get_metrics():
            if metric.name == "vllm:spec_decode_num_draft_tokens":
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        print(f"sharded eagle3 (tp={tp}, draft_tp={tp}): "
              f"accepted={num_accepted_tokens} drafted={num_draft_tokens}")
        assert num_draft_tokens > 0, "no draft tokens produced (sharded draft)"
        assert num_accepted_tokens > 0, "no draft tokens accepted (sharded draft)"

        spec_llm.llm_engine.engine_core.shutdown()
        del spec_llm
        cleanup_dist_env_and_memory()
