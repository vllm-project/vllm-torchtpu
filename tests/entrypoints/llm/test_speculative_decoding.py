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

import collections
import contextlib
import json
import os
import random
import socket
import string
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.sampling_params import StructuredOutputsParams
from vllm.v1.metrics.reader import Counter

from vllm_torchtpu import tpu_info

# Spec-decode e2e integration suite, split across the PR and nightly steps in
# `.buildkite/pipeline_tests.yml`:
#
#   correctness tests  -- unmarked, so they run at PR time AND nightly. A
#       spec-decode correctness regression is expensive to find after the
#       fact, and these paths (rejection sampling, chunked verify, draft
#       proposal, the DP lockstep) have no cheaper coverage. Anything added
#       here lands on the critical path of every PR; weigh its runtime.
#
#   `*_performance*`   -- `@pytest.mark.nightly`, so the PR steps
#       (`-m "not nightly"`) deselect them and only the nightly steps run
#       them. They assert acceptance-rate floors that are calibration
#       sensitive: a loaded shared agent can push an otherwise healthy run
#       under the floor, and the floors are re-measured against the nightly
#       baselines anyway. They gate throughput, not correctness.


def _is_v7x():
    return (tpu_info.get_tpu_type() or "").startswith("tpu7x")


def _get_tensor_parallel_size():
    if _is_v7x():
        return 2
    return 1


def get_ngram_test_prompts():
    # Seeded: unseeded prompts made every nightly compare a different letter
    # set, so test_sd_correctness_greedy_multi_chunk flapped between 0 and 2
    # divergences on the identical commit (a rolled letter can land on a
    # near-tie between the reference and chunked-spec batch shapes). A fixed
    # seed makes the comparison deterministic run-to-run, like the fixed
    # prompt strings the other methods use.
    rng = random.Random(0)
    num_prompts = 10
    prompts = []
    for _ in range(num_prompts):
        w = rng.choice(list(string.ascii_lowercase))
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


def get_mtp_test_prompts():
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
    if method == "mtp":
        return get_mtp_test_prompts()
    raise NotImplementedError(f"{method} is not supported yet.")


# Qwen3.5 ships its MTP head inside the target checkpoint, so the speculative
# config carries no separate draft `model`: vLLM resolves the draft to the
# target model itself (SpeculativeConfig, method="mtp"). Qwen3.5-4B declares
# `mtp_num_hidden_layers: 1`, i.e. one MTP decoder layer; K > 1 re-runs that
# same layer per draft step.
QWEN35_MTP_MODEL = "Qwen/Qwen3.5-4B"

# The data-parallel tests run the MoE sibling rather than the dense 4B, and the
# choice is load-bearing. TorchTPU runs ONE chip slice across every DP engine,
# so a forward's collectives pair mesh-wide: an engine that stops issuing
# forwards strands the engines still running, which block in the device program
# on a peer that never arrives. The idle-rank pairing that prevents that
# (TPUModelRunner._run_dp_idle_pairing) is gated on expert parallelism, and
# vLLM refuses `--enable-expert-parallel` for a dense model ("Number of experts
# in the model must be greater than 0 when expert parallelism is enabled"), so
# dense DP deadlocks as soon as the engines drain unevenly. A3B keeps the
# active parameter count small; the MTP head and method="mtp" are identical to
# the 4B, so these tests still exercise the same speculative path.
QWEN35_MTP_DP_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

# Separate draft checkpoint; block size 16, hence num_speculative_tokens=15.
QWEN35_DFLASH_DP_DRAFT = "z-lab/Qwen3.5-35B-A3B-DFlash"

# Measured 50.03% per engine on v7x at dp=2 x tp=4.
DFLASH_DP_MIN_ACCEPTANCE_RATE = 0.45

# dp=2 x tp=4: the deployed topology, and the only TP where the draft and
# target KV geometry match. Same 8 chips as the MTP DP tests.
DFLASH_DP_SIZE = 2
DFLASH_DP_TP_SIZE = 4


def _qwen35_dflash_dp_kwargs() -> dict:
    return {
        **QWEN35_KWARGS,
        "data_parallel_size": DFLASH_DP_SIZE,
        "tensor_parallel_size": DFLASH_DP_TP_SIZE,
        "kv_cache_dtype": "fp8",
        "enable_expert_parallel": True,
    }


def _qwen35_dflash_speculative_config() -> dict:
    return {
        "method": "dflash",
        "model": QWEN35_DFLASH_DP_DRAFT,
        "num_speculative_tokens": 15,
        # A DFlash draft must shard exactly like its target; draft_tp=1 against
        # a sharded target is rejected at KV-pool materialisation.
        "draft_tensor_parallel_size": DFLASH_DP_TP_SIZE,
    }


# Engine kwargs every Qwen3.5 run in this file needs, on top of the shared
# ones the helpers set.
QWEN35_KWARGS = {
    # Qwen3.5 declares a vision modality; run it text-only so vLLM takes the
    # language-model-only path (see TpuPlatform.check_and_update_config).
    "language_model_only": True,
    "limit_mm_per_prompt": {
        "image": 0,
        "video": 0
    },
    # Qwen3.5 is hybrid (GDN linear attention + full attention). TpuPlatform
    # rejects prefix caching together with speculative decoding on hybrid
    # Mamba models, so keep it off on both the reference and the spec engine
    # (the reference must match the spec run's cache behaviour anyway).
    "enable_prefix_caching": False,
}


def _qwen35_mtp_speculative_config(num_speculative_tokens: int = 1) -> dict:
    return {
        "method": "mtp",
        "num_speculative_tokens": num_speculative_tokens,
    }


# Generation horizon for eagle3 tests, pinned separately from the
# shared one below. The horizon and the floor are one calibration: changing either without
# re-measuring the other invalidates the test.
EAGLE3_PERF_MAX_TOKENS = 16


def _make_sampling_config(temperature: float = 0,
                          max_tokens: int = 200) -> SamplingParams:
    return SamplingParams(temperature=temperature,
                          max_tokens=max_tokens,
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


# Spec-decode counter names. `llm.get_metrics()` reports the bare names;
# the HTTP /metrics endpoint reports the Prometheus form with a `_total`
# suffix. Same counters, so both transports derive their names from here.
SPEC_DRAFT_METRIC = "vllm:spec_decode_num_draft_tokens"
SPEC_ACCEPTED_METRIC = "vllm:spec_decode_num_accepted_tokens"


def _assert_no_spec_divergence(prompts: list[str],
                               ref_texts: list[str],
                               spec_texts: list[str],
                               group_keys: list | None = None):
    """Speculation must not change greedy output. Shared by the in-process
    and served correctness tests so both apply the same standard.

    Greedy decoding of one prompt has a single answer, but on TPU at
    batch > 1 NEITHER run reproduces it on every row: a row's logits depend
    on its position in the batch (tiling and reduction order), so a near-tie
    argmax can flip for one row while its siblings agree. Speculation does
    not cause this -- the same flip, to the same alternative text, has been
    observed in the no-spec reference -- and because the two runs batch
    differently it lands on different rows. Comparing spec[i] to ref[i]
    positionally charges it to speculation.

    Compare each group's majority output instead. Every row of a group is
    the same prompt with the same params, so each run must agree with itself
    on a majority of the group, and the two majorities must then match. A
    group of one degenerates to exact comparison, so unique-prompt callers
    lose no strictness. A systematic masking or rejection bug moves a
    group's majority and is still caught; a single flipped row inside an
    otherwise-agreeing group is indistinguishable from the tiling noise
    above and is tolerated -- for structured requests the caller's per-row
    schema check still covers every row.

    Rows are grouped by prompt. Pass `group_keys` when rows sharing a prompt
    do NOT share sampling params (a mixed structured/unconstrained batch):
    only rows that are supposed to produce the same text may vote in the
    same majority, so the key has to carry whatever else varies per row.
    """
    assert len(ref_texts) == len(spec_texts) == len(prompts), (
        f"{len(prompts)} prompts, reference produced {len(ref_texts)} "
        f"outputs, speculative produced {len(spec_texts)}")
    keys = prompts if group_keys is None else group_keys
    assert len(keys) == len(prompts), (
        f"{len(prompts)} prompts but {len(keys)} group keys")

    rows_by_key: dict[object, list[int]] = collections.defaultdict(list)
    for i, key in enumerate(keys):
        rows_by_key[key].append(i)
    # A group votes on one expected text, so a key must be at least as
    # fine-grained as the prompt. A key that pools rows with different
    # prompts would compare unrelated outputs and still report a pass.
    for key, rows in rows_by_key.items():
        distinct = {prompts[i] for i in rows}
        assert len(distinct) == 1, (
            f"group key {key!r} pools {len(distinct)} different prompts "
            f"across rows {rows}; the key must include the prompt")

    def majority(texts: list[str], rows: list[int], label: str,
                 key: object) -> str:
        counts = collections.Counter(texts[i] for i in rows)
        modal_text, modal_rows = counts.most_common(1)[0]
        if modal_rows != len(rows):
            print(f"{label} is not row-uniform for {key!r}: {len(counts)} "
                  f"distinct outputs across {len(rows)} rows "
                  f"(majority {modal_rows}/{len(rows)})")
        assert modal_rows * 2 > len(rows), (
            f"no majority {label} output for {key!r}; it is too unstable to "
            f"compare against: {dict(counts)}")
        return modal_text

    misses = 0
    for key, rows in rows_by_key.items():
        ref_modal = majority(ref_texts, rows, "reference", key)
        spec_modal = majority(spec_texts, rows, "speculative", key)
        if spec_modal != ref_modal:
            misses += 1
            print(f"ref_output: {ref_modal}")
            print(f"spec_output: {spec_modal}")

    assert misses == 0


def _assert_acceptance_rate(
    num_draft_tokens: float,
    num_accepted_tokens: float,
    min_acceptance_rate: float,
    method: str,
) -> float:
    """Report the acceptance rate and hold it above the collapse floor.

    Shared by the in-process and served performance tests; each reads the
    counters from its own transport and hands the totals here.
    """
    acceptance_rate = 0.0
    if num_draft_tokens > 0:
        acceptance_rate = num_accepted_tokens / num_draft_tokens
        print(f"Acceptance rate: {acceptance_rate:.2%}")
        print("num_accepted_tokens:" + str(num_accepted_tokens))
        print("num_draft_tokens:" + str(num_draft_tokens))

    assert num_draft_tokens > 0, "Draft tokens should be greater than 0."
    assert acceptance_rate >= min_acceptance_rate, \
        f"Expected at least {min_acceptance_rate:.2%} acceptance rate for " \
        f"{method}, got {acceptance_rate:.2%}"
    return acceptance_rate


class _InProcessEngine:
    """Engine backed by LLM(...) in this process -- the default."""

    def __init__(self, llm: LLM):
        self._llm = llm

    def generate(self, prompts: list[str],
                 sampling_config: SamplingParams) -> list[str]:
        outputs = self._llm.generate(prompts, sampling_config)
        return [output.outputs[0].text for output in outputs]

    def spec_counters(self) -> tuple[float, float, dict[str, float]]:
        num_draft_tokens = num_accepted_tokens = 0.0
        for metric in self._llm.get_metrics():
            if metric.name == SPEC_DRAFT_METRIC:
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == SPEC_ACCEPTED_METRIC:
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        # One engine, reported under the same shape the served backend uses so
        # callers can check per-engine counts without knowing the transport.
        return num_draft_tokens, num_accepted_tokens, {"0": num_draft_tokens}

    def shutdown(self) -> None:
        self._llm.llm_engine.engine_core.shutdown()


@contextlib.contextmanager
def _engine(model_name: str, speculative_config: dict | None, kwargs: dict):
    """Yield a spec-decode engine, in-process or served.

    Data parallelism forces the served path: vLLM rejects
    LLM(data_parallel_size>1) for single-process use unless
    `current_platform.is_tpu()`, and this plugin registers TpuPlatform with
    `_enum = PlatformEnum.OOT`, so the exemption never applies and the
    constructor raises. Everything above the backend -- prompts, the
    reference/speculative sequencing, the assertions -- stays shared.
    """
    if kwargs.get("data_parallel_size", 1) > 1:
        with _serve(model_name, speculative_config, kwargs) as served:
            yield served
        return

    llm_kwargs = {k: v for k, v in kwargs.items() if k != "data_parallel_size"}
    llm = LLM(model=model_name,
              speculative_config=speculative_config,
              **llm_kwargs)
    engine = _InProcessEngine(llm)
    try:
        yield engine
    finally:
        engine.shutdown()
        del llm, engine
        cleanup_dist_env_and_memory()
        # Waiting for TPUs to be fully released.
        time.sleep(15)


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

        # 1. Reference Run (no speculation)
        with _engine(model_name, None, kwargs) as ref_engine:
            ref_texts = ref_engine.generate(test_prompts, sampling_config)

        # 2. Speculative Run
        with _engine(model_name, speculative_config, kwargs) as spec_engine:
            spec_texts = spec_engine.generate(test_prompts, sampling_config)

        _assert_no_spec_divergence(test_prompts, ref_texts, spec_texts)


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


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
@pytest.mark.parametrize(
    "max_num_seqs", [pytest.param(1, id="bs1"),
                     pytest.param(4, id="bs4")])
def test_qwen35_mtp_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
):
    """Qwen3.5 MTP must not change greedy output vs a no-spec reference.

    This is the hybrid-model spec-decode path: between the target's verify
    forward and the draft's propose the runner has to roll the GDN recurrent
    state back over the rejected tokens. A broken rollback shows up here as a
    text mismatch (the target decodes from contaminated state), which the
    acceptance-rate check in test_qwen35_mtp_performance_greedy cannot see.
    """
    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        QWEN35_MTP_MODEL,
        _qwen35_mtp_speculative_config(),
        max_num_seqs=max_num_seqs,
        async_scheduling=async_scheduling,
        extra_kwargs=QWEN35_KWARGS,
    )


@pytest.mark.timeout(1800)
def test_qwen35_mtp_correctness_greedy_multi_step(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """Same correctness guarantee with K=2 draft tokens per step.

    Qwen3.5-4B has a single MTP layer, so K=2 runs that layer twice per step
    and exercises the drafter's multi-step loop (per-step draft attention
    metadata, GDN state advance/rollback across >1 speculated token) that the
    K=1 configuration never reaches.
    """
    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        QWEN35_MTP_MODEL,
        _qwen35_mtp_speculative_config(num_speculative_tokens=2),
        max_num_seqs=4,
        async_scheduling=False,
        extra_kwargs=QWEN35_KWARGS,
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

        with _engine(model_name, speculative_config, kwargs) as spec_engine:
            spec_engine.generate(test_prompts, sampling_config)
            num_draft_tokens, num_accepted_tokens, per_engine = \
                spec_engine.spec_counters()

        # Every engine must have drafted. Under data parallelism a drafter
        # that failed to load on one engine leaves the summed rate looking
        # healthy; with one engine this is just "the drafter ran".
        expected_engines = kwargs.get("data_parallel_size", 1)
        print(f"{speculative_config['method']} per_engine={per_engine}")
        assert len(per_engine) == expected_engines, \
            f"expected {expected_engines} engine(s) to report draft " \
            f"counters, got {per_engine}"
        idle = sorted(e for e, n in per_engine.items() if n <= 0)
        assert not idle, f"engine(s) {idle} produced no draft tokens"

        _assert_acceptance_rate(num_draft_tokens, num_accepted_tokens,
                                min_acceptance_rate,
                                speculative_config["method"])


@pytest.mark.nightly
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


@pytest.mark.nightly
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
        _make_sampling_config(temperature, max_tokens=EAGLE3_PERF_MAX_TOKENS),
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


@pytest.mark.nightly
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
    # at ~7.7-10.8%, so bs1 gets a 5% floor and bs4 gets a 7% floor.
    [pytest.param(1, 0.05, id="bs1"),
     pytest.param(4, 0.07, id="bs4")])
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


@pytest.mark.nightly
@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
@pytest.mark.parametrize(
    "max_num_seqs", [pytest.param(1, id="bs1"),
                     pytest.param(4, id="bs4")])
def test_qwen35_mtp_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
    max_num_seqs: int,
    async_scheduling: bool,
):
    """Qwen3.5 MTP proposes and gets drafts accepted at a useful rate.

    The floor is a collapse guard: it catches a drafter that silently stops
    proposing, or one whose accepted tokens fall because the GDN state handed
    to the draft is wrong. The prompt is a trivially predictable sequence and
    K=1, so a healthy v7x/TP2 run accepts every draft — all four
    configurations measure 80/80 = 100%.
    """
    _test_performance_helper(
        monkeypatch,
        sampling_config,
        _qwen35_mtp_speculative_config(),
        min_acceptance_rate=0.9,
        max_num_seqs=max_num_seqs,
        model_name=QWEN35_MTP_MODEL,
        async_scheduling=async_scheduling,
        extra_kwargs=QWEN35_KWARGS,
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


# ---------------------------------------------------------------------------
# Structured outputs + speculative decoding
#
# The scheduler emits 1 + s_r bitmask rows per structured request on verify
# steps (one per draft position + the bonus row); the runner scatters them
# onto the gathered target/bonus logits before rejection sampling. These
# tests drive that path end to end. Structured decoding WITHOUT speculation
# is covered separately in test_structured_output.py.
# ---------------------------------------------------------------------------

STRUCTURED_SPEC_JSON = '{"name": "Alice", "age": 30}'


def get_structured_spec_prompts() -> list[str]:
    # The expected JSON is embedded verbatim so ngram prompt-lookup proposes
    # drafts inside the constrained generation, driving real (masked) verify
    # traffic instead of degenerating to draft-free steps.
    return [
        f"Repeat exactly this JSON and nothing else: {STRUCTURED_SPEC_JSON}\n"
        f"JSON: " for _ in range(8)
    ]


def _structured_spec_ngram_config() -> dict:
    return {
        "method": "ngram",
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 3,
        "num_speculative_tokens": 3,
    }


def _assert_valid_structured_json(text: str) -> None:
    parsed = json.loads(text)
    assert isinstance(parsed, dict), text
    assert isinstance(parsed.get("name"), str), text
    assert isinstance(parsed.get("age"), int), text


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("multi_chunk", [
    pytest.param(False, id="single_chunk"),
    pytest.param(True, id="multi_chunk")
])
def test_structured_output_spec_decode_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    multi_chunk: bool,
):
    """Structured outputs under greedy speculative decoding.

    Asserts three things on a mixed batch (structured + unconstrained
    requests interleaved):
      (a) every structured output satisfies its JSON schema (the grammar
          bitmask actually reached the spec-verify logits);
      (b) greedy output is identical to a no-spec run with the same
          structured params (masking + rejection must not diverge);
      (c) drafts were proposed and accepted (the masked verify path really
          ran, rather than every step degenerating to draft-free sampling).
    The multi_chunk variant shrinks the per-chunk request cap so the
    chunk-local row mapping of both bitmask is exercised.
    """
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
        },
        "required": ["name", "age"],
    }
    structured = SamplingParams(
        temperature=0,
        max_tokens=64,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    unconstrained = SamplingParams(temperature=0, max_tokens=64)

    with monkeypatch.context() as mp:
        if multi_chunk:
            # collective_rpc below ships a Python function to the TPU
            # workers, which needs pickle fallback in vLLM's IPC encoder.
            mp.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        test_prompts = get_structured_spec_prompts()
        params = [
            structured if i % 2 == 0 else unconstrained
            for i in range(len(test_prompts))
        ]
        kwargs = dict(
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=False,
            disable_log_stats=False,
        )

        # 1. Reference run: structured outputs, no speculation.
        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, params)
        ref_llm.llm_engine.engine_core.shutdown()
        del ref_llm
        cleanup_dist_env_and_memory()
        time.sleep(15)

        # 2. Speculative run with the same structured params.
        spec_llm = LLM(model=model_name,
                       speculative_config=_structured_spec_ngram_config(),
                       **kwargs)
        if multi_chunk:
            # Cap=2 with max_num_seqs=4: any batch with >2 reqs splits, so
            # the bitmask scatter runs its chunk-local path.
            spec_llm.llm_engine.collective_rpc(_force_multi_chunk_cap,
                                               args=(2, ))
        spec_outputs = spec_llm.generate(test_prompts, params)

        # (a) Constraint holds on every structured request.
        for i, spec in enumerate(spec_outputs):
            if i % 2 == 0:
                _assert_valid_structured_json(spec.outputs[0].text)

        # (b) Speculation must not change greedy output. Compared per-group
        # majority, not row-for-row: the unconstrained rows run the full 64
        # free tokens, and at this batch size either run can flip one row on
        # a near-tie (see _assert_no_spec_divergence) -- both directions have
        # been observed here. Structured and unconstrained rows share a
        # prompt but not sampling params, so the group key carries the params
        # too; they must not vote in one majority. Every structured row is
        # still checked individually by (a) above.
        ref_texts = [o.outputs[0].text for o in ref_outputs]
        spec_texts = [o.outputs[0].text for o in spec_outputs]
        group_keys = [(prompt, "structured" if i % 2 == 0 else "free")
                      for i, prompt in enumerate(test_prompts)]
        _assert_no_spec_divergence(test_prompts,
                                   ref_texts,
                                   spec_texts,
                                   group_keys=group_keys)

        # (c) The masked verify path actually saw draft traffic.
        num_draft_tokens = num_accepted_tokens = 0
        for metric in spec_llm.get_metrics():
            if metric.name == SPEC_DRAFT_METRIC:
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == SPEC_ACCEPTED_METRIC:
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        print(f"structured+spec: accepted={num_accepted_tokens} "
              f"drafted={num_draft_tokens}")
        assert num_draft_tokens > 0, \
            "no draft tokens proposed under structured outputs"
        assert num_accepted_tokens > 0, \
            "no draft tokens accepted under structured outputs"

        spec_llm.llm_engine.engine_core.shutdown()
        del spec_llm
        cleanup_dist_env_and_memory()


@pytest.mark.timeout(1800)
def test_structured_output_spec_decode_non_greedy(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
):
    """Non-greedy spec decode with a choice constraint.

    Rejection sampling is stochastic, so no reference comparison; the mask
    guarantee is absolute though: every output must be one of the choices
    (bonus sampling, draft verification, and recovery sampling all happen
    on masked distributions).
    """
    choices = ["Positive", "Negative"]
    params = SamplingParams(
        temperature=0.7,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(choice=choices),
    )
    # The choice words appear in the prompt so ngram lookup has material
    # to draft from.
    prompts = [
        "Positive Positive Positive. The sentiment of 'great movie' is: "
        for _ in range(8)
    ]

    with monkeypatch.context():
        spec_llm = LLM(
            model=model_name,
            speculative_config=_structured_spec_ngram_config(),
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=False,
        )
        try:
            outputs = spec_llm.generate(prompts, params)
            assert len(outputs) == len(prompts)
            for output in outputs:
                assert output.outputs[0].text in choices, (
                    f"constraint violated: {output.outputs[0].text!r}")
        finally:
            spec_llm.llm_engine.engine_core.shutdown()
            del spec_llm
            cleanup_dist_env_and_memory()


@pytest.mark.timeout(2400)
@pytest.mark.parametrize(
    "async_scheduling",
    [pytest.param(False, id="sync"),
     pytest.param(True, id="async")])
def test_structured_output_eagle3_greedy(monkeypatch: pytest.MonkeyPatch,
                                         async_scheduling: bool):
    """Structured outputs under greedy eagle3 speculative decoding.

    Same three assertions as the ngram combo test -- (a) every structured
    output satisfies its schema, (b) greedy output matches a no-spec
    reference, (c) drafts were proposed and accepted -- but with a
    model-based drafter. The async archive additionally exercises the
    deferred grammar-bitmask path: the worker stages a host copy of each
    step's device drafts for take_draft_token_ids(), which the engine
    substitutes for the async scheduler's -1 placeholders.
    """
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string"
            },
            "age": {
                "type": "integer"
            },
        },
        "required": ["name", "age"],
    }
    structured = SamplingParams(
        temperature=0,
        max_tokens=64,
        structured_outputs=StructuredOutputsParams(json=schema),
    )
    unconstrained = SamplingParams(temperature=0, max_tokens=64)

    with monkeypatch.context():
        test_prompts = get_structured_spec_prompts()
        params = [
            structured if i % 2 == 0 else unconstrained
            for i in range(len(test_prompts))
        ]
        kwargs = dict(
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=async_scheduling,
            disable_log_stats=False,
        )

        # 1. Reference run: structured outputs, no speculation.
        ref_llm = LLM(model=model_name, **kwargs)
        ref_outputs = ref_llm.generate(test_prompts, params)
        ref_llm.llm_engine.engine_core.shutdown()
        del ref_llm
        cleanup_dist_env_and_memory()
        time.sleep(15)

        # 2. eagle3 speculative run with the same structured params.
        spec_llm = LLM(model=model_name,
                       speculative_config={
                           "method": "eagle3",
                           "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                           "num_speculative_tokens": 3,
                           "draft_tensor_parallel_size": 1,
                       },
                       **kwargs)
        spec_outputs = spec_llm.generate(test_prompts, params)

        # (a) Constraint holds on every structured request.
        for i, spec in enumerate(spec_outputs):
            if i % 2 == 0:
                _assert_valid_structured_json(spec.outputs[0].text)

        # (b) Speculation must not change greedy output. Compared per-group
        # majority, not row-for-row: the unconstrained rows run the full 64
        # free tokens, and at this batch size either run can flip one row on
        # a near-tie (see _assert_no_spec_divergence) -- both directions have
        # been observed here. Structured and unconstrained rows share a
        # prompt but not sampling params, so the group key carries the params
        # too; they must not vote in one majority. Every structured row is
        # still checked individually by (a) above.
        ref_texts = [o.outputs[0].text for o in ref_outputs]
        spec_texts = [o.outputs[0].text for o in spec_outputs]
        group_keys = [(prompt, "structured" if i % 2 == 0 else "free")
                      for i, prompt in enumerate(test_prompts)]
        _assert_no_spec_divergence(test_prompts,
                                   ref_texts,
                                   spec_texts,
                                   group_keys=group_keys)

        # (c) The masked verify path actually saw draft traffic.
        num_draft_tokens = num_accepted_tokens = 0
        for metric in spec_llm.get_metrics():
            if metric.name == SPEC_DRAFT_METRIC:
                assert isinstance(metric, Counter)
                num_draft_tokens += metric.value
            elif metric.name == SPEC_ACCEPTED_METRIC:
                assert isinstance(metric, Counter)
                num_accepted_tokens += metric.value
        print(f"structured+eagle3({'async' if async_scheduling else 'sync'}): "
              f"accepted={num_accepted_tokens} drafted={num_draft_tokens}")
        assert num_draft_tokens > 0, \
            "no draft tokens proposed under structured outputs"
        assert num_accepted_tokens > 0, \
            "no draft tokens accepted under structured outputs"

        spec_llm.llm_engine.engine_core.shutdown()
        del spec_llm
        cleanup_dist_env_and_memory()


@pytest.mark.timeout(2400)
def test_structured_output_eagle3_async_adversarial_choice(
        monkeypatch: pytest.MonkeyPatch):
    """Async + eagle3 + a choice constraint that fights greedy decoding.

    The repeat-JSON workload above can pass even with a degraded bitmask:
    greedy argmax already lands inside the grammar, so an all-allow mask row
    is a fuse that never blows. Here the prompt pulls greedy decoding toward
    free text while the constraint only allows two fixed multi-token strings.
    If the bitmask were built from the async scheduler's -1 placeholders
    (all-allow from the second draft position on), the model's own
    continuation would slip through a spec-verify step and either violate
    the choice or kill the request with a grammar error -- so every valid,
    completed output is direct evidence the deferred-bitmask path saw the
    real drafts.

    Odd rows run unconstrained on the file's standard eagle3 prompt, whose
    continuation the drafter predicts well. Those rows carry the
    "speculation actually ran" floor, which the constrained rows cannot --
    see the metric comment below.
    """
    model_name = "NousResearch/Meta-Llama-3.1-8B-Instruct"
    choices = ["strawberry banana kiwi", "pineapple mango papaya"]
    constrained = SamplingParams(
        temperature=0,
        max_tokens=16,
        structured_outputs=StructuredOutputsParams(choice=choices),
    )
    unconstrained = SamplingParams(temperature=0, max_tokens=16)
    # Nothing in the constrained prompt suggests the allowed strings: greedy
    # wants a color word, the grammar only permits fruit. The control rows
    # use the drafter-friendly sequence prompt the other eagle3 tests rest
    # their acceptance floors on -- the adversarial prompt is a poor control
    # (measured: 8 accepted tokens across 4 rows, a thin margin over the
    # floor), and nothing about the constrained rows depends on the control
    # rows sharing their prompt.
    prompts = [
        "My favorite color is " if i % 2 == 0 else get_eagle3_test_prompts()[0]
        for i in range(8)
    ]
    params = [
        constrained if i % 2 == 0 else unconstrained
        for i in range(len(prompts))
    ]

    with monkeypatch.context():
        spec_llm = LLM(
            model=model_name,
            speculative_config={
                "method": "eagle3",
                "model": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
                "num_speculative_tokens": 3,
                "draft_tensor_parallel_size": 1,
            },
            max_model_len=256,
            max_num_seqs=4,
            tensor_parallel_size=_get_tensor_parallel_size(),
            async_scheduling=True,
            disable_log_stats=False,
        )
        try:
            outputs = spec_llm.generate(prompts, params)
            assert len(outputs) == len(prompts)
            for i, output in enumerate(outputs):
                if i % 2:
                    continue  # unconstrained control row
                assert output.outputs[0].text in choices, (
                    f"constraint violated: {output.outputs[0].text!r}")

            # Speculation must have actually run, or the checks above are
            # just plain decoding and say nothing about the masked verify
            # path. The floor is on ACCEPTED tokens and the unconstrained
            # rows are what supply it; a drafted-token floor cannot work
            # here.
            #
            # `vllm:spec_decode_num_draft_tokens` is reported net of the
            # drafts the grammar threw out: under async scheduling the
            # scheduler counts the K placeholder spec tokens it scheduled,
            # then make_spec_decoding_stats() subtracts
            # scheduler_output.num_invalid_spec_tokens -- the drafts
            # update_draft_token_ids_in_output() handed to
            # grammar.validate_tokens() and the grammar refused -- and
            # records nothing once the remainder reaches 0. validate_tokens
            # keeps only the accepted prefix, and this grammar is built to
            # refuse the model's own drafts from the first position on, so a
            # healthy deferred-bitmask path reports exactly 0 drafted on the
            # constrained rows. A path that regressed to the -1 placeholders
            # would never call validate_tokens and would report K per step
            # instead: a >0 drafted floor fails on the working path and
            # passes on the broken one.
            num_draft_tokens = num_accepted_tokens = 0
            for metric in spec_llm.get_metrics():
                if metric.name == SPEC_DRAFT_METRIC:
                    assert isinstance(metric, Counter)
                    num_draft_tokens += metric.value
                elif metric.name == SPEC_ACCEPTED_METRIC:
                    assert isinstance(metric, Counter)
                    num_accepted_tokens += metric.value
            print("adversarial choice + eagle3(async): "
                  f"accepted={num_accepted_tokens} "
                  f"drafted(net of grammar-rejected)={num_draft_tokens}")
            assert num_accepted_tokens > 0, \
                "no draft tokens accepted on the unconstrained control rows"
        finally:
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


# ---------------------------------------------------------------------------
# Served backend (data parallelism)
#
# `_engine` routes here when data_parallel_size > 1, because vLLM rejects
# LLM(data_parallel_size>1) for single-process use unless
# `current_platform.is_tpu()`, and this plugin registers TpuPlatform with
# `_enum = PlatformEnum.OOT`, so the exemption never applies. Serving is also
# how DP actually ships here (see scripts/vllm/benchmarking/configs/*.sh).
# ---------------------------------------------------------------------------

DP_SIZE = 4

# Cold TPU compilation of a DP engine pair takes minutes, and an idle DP
# engine whose peer is still compiling can trip an RPC timeout, so give
# startup room and always drive every engine with concurrent load.
DP_SERVER_STARTUP_TIMEOUT_S = 1800
DP_REQUEST_TIMEOUT_S = 1500
DP_EXECUTE_MODEL_TIMEOUT_S = 1800
DP_CONCURRENCY = 8

# Prometheus renders the same counters the in-process reader exposes, with a
# `_total` suffix.
_DRAFT_METRIC = SPEC_DRAFT_METRIC + "_total"
_ACCEPTED_METRIC = SPEC_ACCEPTED_METRIC + "_total"

# Engine kwargs `_serve` knows how to express as `vllm serve` flags. Anything
# else must be added here, or a served run would quietly differ from the
# in-process run built from the same kwargs.
_SERVE_TRANSLATED_KEYS = {
    "max_model_len",
    "max_num_seqs",
    "tensor_parallel_size",
    "data_parallel_size",
    "async_scheduling",
    "enable_prefix_caching",
    "language_model_only",
    "limit_mm_per_prompt",
    "disable_log_stats",
    "enable_expert_parallel",
    "kv_cache_dtype",
}


def _serve_args(kwargs: dict) -> list[str]:
    """Translate engine kwargs into `vllm serve` flags.

    Deriving the flags from the same kwargs dict the in-process backend
    receives is what keeps a served run honest: the two transports cannot
    drift apart without this raising.
    """
    unhandled = {k for k, v in kwargs.items() if v is not None} - \
        _SERVE_TRANSLATED_KEYS
    assert not unhandled, (
        f"engine kwargs gained {sorted(unhandled)}; add the matching `vllm "
        "serve` flag here so served runs keep matching in-process runs")

    args: list[str] = []
    for key in ("max_model_len", "max_num_seqs", "tensor_parallel_size",
                "data_parallel_size", "kv_cache_dtype"):
        if kwargs.get(key) is not None:
            args += [f"--{key.replace('_', '-')}", str(kwargs[key])]

    # Tri-state flags: absent means "leave the server default alone".
    for key in ("async_scheduling", "enable_prefix_caching"):
        value = kwargs.get(key)
        if value is not None:
            flag = key.replace("_", "-")
            args.append(f"--{flag}" if value else f"--no-{flag}")

    if kwargs.get("language_model_only"):
        args.append("--language-model-only")
    if kwargs.get("enable_expert_parallel"):
        args.append("--enable-expert-parallel")
    if kwargs.get("limit_mm_per_prompt") is not None:
        args += [
            "--limit-mm-per-prompt",
            json.dumps(kwargs["limit_mm_per_prompt"])
        ]
    # Stats are on by default, which is what /metrics needs; only the
    # disabling direction has a flag.
    if kwargs.get("disable_log_stats"):
        args.append("--disable-log-stats")
    return args


def _get_local_tpu_chip_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:  # noqa: BLE001 - detection failure is a skip
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


def _require_chips_for_dp(
    dp_size: int = DP_SIZE,
    tp_size: int | None = None,
) -> None:
    tp = tp_size if tp_size is not None else _get_tensor_parallel_size()
    needed = dp_size * tp
    available = _get_local_tpu_chip_count()
    if available < needed:
        pytest.skip(f"DP spec decode needs {needed} chips (dp={dp_size} x "
                    f"tp={tp}), found {available}")


def _pick_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class _ServedEngine:
    """Engine behind an out-of-process `vllm serve`."""

    def __init__(self, base: str, model_name: str):
        self._base = base
        self._model = model_name

    def generate(self, prompts: list[str],
                 sampling_config: SamplingParams) -> list[str]:
        """Send the prompts concurrently and return their texts.

        Concurrency is required, not incidental: vLLM's DP load balancer only
        spreads work over every engine when several requests are in flight,
        and an engine left idle while its peer compiles can time out its own
        RPC.
        """

        def one(prompt: str) -> str:
            payload = json.dumps({
                "model": self._model,
                "prompt": prompt,
                "max_tokens": sampling_config.max_tokens,
                "temperature": sampling_config.temperature,
                "ignore_eos": sampling_config.ignore_eos,
            }).encode()
            request = urllib.request.Request(
                f"{self._base}/v1/completions",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(
                    request, timeout=DP_REQUEST_TIMEOUT_S) as response:
                body = json.loads(response.read().decode())
            return body["choices"][0]["text"]

        with ThreadPoolExecutor(max_workers=DP_CONCURRENCY) as pool:
            return list(pool.map(one, prompts))

    def spec_counters(self) -> tuple[float, float, dict[str, float]]:
        """Sum the counters, keeping the per-engine draft breakdown.

        Each DP engine exports its own series (`engine="0"`, `engine="1"`,
        ...). Matching on `name{` stops the accepted-token counter from also
        swallowing `..._per_pos_total`, which shares its prefix.
        """
        with urllib.request.urlopen(f"{self._base}/metrics",
                                    timeout=60) as response:
            body = response.read().decode()

        totals = {_DRAFT_METRIC: 0.0, _ACCEPTED_METRIC: 0.0}
        per_engine: dict[str, float] = {}
        for line in body.splitlines():
            if line.startswith("#"):
                continue
            for name in totals:
                if not (line.startswith(name + "{")
                        or line.startswith(name + " ")):
                    continue
                try:
                    value = float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    continue
                totals[name] += value
                if name == _DRAFT_METRIC and 'engine="' in line:
                    engine = line.split('engine="', 1)[1].split('"', 1)[0]
                    per_engine[engine] = per_engine.get(engine, 0.0) + value
        return totals[_DRAFT_METRIC], totals[_ACCEPTED_METRIC], per_engine


@contextlib.contextmanager
def _serve(model_name: str, speculative_config: dict | None, kwargs: dict):
    """Run `vllm serve` with these engine kwargs; yield a _ServedEngine."""
    port = _pick_free_port()
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", model_name,
        "--port",
        str(port), *_serve_args(kwargs)
    ]
    if speculative_config is not None:
        cmd += ["--speculative-config", json.dumps(speculative_config)]

    base = f"http://localhost:{port}"

    # Log to a file rather than a pipe. vLLM startup writes far more than a
    # pipe buffer holds, and nothing here drains it, so PIPE would block the
    # server mid-startup and the health poll would never succeed.
    log = tempfile.NamedTemporaryFile(mode="w+",
                                      suffix=".log",
                                      prefix="vllm_serve_dp_",
                                      delete=False)

    def _fail(message: str) -> RuntimeError:
        log.flush()
        with open(log.name) as fh:
            tail = fh.read()[-4000:]
        return RuntimeError(f"{message}\n--- {log.name} (tail) ---\n{tail}")

    # The first request against a fresh engine pair compiles the runtime
    # shapes inside sample_tokens, and on TPU that outruns vLLM's 300s default
    # worker-RPC deadline -- the engine is torn down mid-compile and every
    # request 500s. Raise the deadline past the compile.
    env = dict(
        os.environ,
        VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=str(DP_EXECUTE_MODEL_TIMEOUT_S))

    server = subprocess.Popen(cmd,
                              stdout=log,
                              stderr=subprocess.STDOUT,
                              env=env,
                              text=True)
    try:
        deadline = time.time() + DP_SERVER_STARTUP_TIMEOUT_S
        while True:
            if server.poll() is not None:
                raise _fail(
                    f"vllm serve exited with {server.returncode} during "
                    "startup")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=10).close()
                break
            except (urllib.error.URLError, OSError):
                pass
            if time.time() > deadline:
                raise _fail("vllm serve did not become healthy within "
                            f"{DP_SERVER_STARTUP_TIMEOUT_S}s")
            time.sleep(10)
        print(f"vllm serve ready at {base}; log: {log.name}")
        yield _ServedEngine(base, model_name)
    finally:
        server.terminate()
        try:
            server.wait(timeout=180)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=60)
        log.close()
        # Waiting for TPUs to be fully released before the next engine starts.
        time.sleep(15)


def _qwen35_dp_kwargs() -> dict:
    return {
        **QWEN35_KWARGS,
        "data_parallel_size": DP_SIZE,
        # Required, not incidental -- see QWEN35_MTP_DP_MODEL: without it the
        # DP engines deadlock once they drain unevenly.
        "enable_expert_parallel": True,
    }


@pytest.mark.multichip
@pytest.mark.timeout(3600)
def test_qwen35_mtp_dp_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """MTP under data parallelism must not change greedy output.

    Same helper, same prompts, same comparison as the single-engine
    correctness tests -- only data_parallel_size differs, which routes both
    the reference and the speculative run through the served backend.
    """
    _require_chips_for_dp()

    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        QWEN35_MTP_DP_MODEL,
        _qwen35_mtp_speculative_config(),
        async_scheduling=False,
        extra_kwargs=_qwen35_dp_kwargs(),
    )


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.timeout(2400)
def test_qwen35_mtp_dp_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """Every DP engine drafts, and the drafts are accepted at a useful rate.

    The helper's per-engine check does the DP-specific work: a drafter that
    failed to load on one engine leaves the summed rate looking healthy.
    """
    _require_chips_for_dp()

    _test_performance_helper(
        monkeypatch,
        sampling_config,
        _qwen35_mtp_speculative_config(),
        min_acceptance_rate=0.9,
        model_name=QWEN35_MTP_DP_MODEL,
        async_scheduling=False,
        extra_kwargs=_qwen35_dp_kwargs(),
    )


@pytest.mark.multichip
@pytest.mark.timeout(3600)
def test_qwen35_dflash_dp_correctness_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """DFlash under data parallelism must not change greedy output."""
    _require_chips_for_dp(DFLASH_DP_SIZE, DFLASH_DP_TP_SIZE)
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")

    _test_correctness_helper(
        monkeypatch,
        sampling_config,
        QWEN35_MTP_DP_MODEL,
        _qwen35_dflash_speculative_config(),
        async_scheduling=False,
        extra_kwargs=_qwen35_dflash_dp_kwargs(),
    )


@pytest.mark.nightly
@pytest.mark.multichip
@pytest.mark.timeout(2400)
def test_qwen35_dflash_dp_performance_greedy(
    monkeypatch: pytest.MonkeyPatch,
    sampling_config: SamplingParams,
):
    """Every DP engine drafts with DFlash, and drafts are accepted."""
    _require_chips_for_dp(DFLASH_DP_SIZE, DFLASH_DP_TP_SIZE)
    monkeypatch.setenv("MODEL_IMPL_TYPE", "vllm")

    _test_performance_helper(
        monkeypatch,
        sampling_config,
        _qwen35_dflash_speculative_config(),
        min_acceptance_rate=DFLASH_DP_MIN_ACCEPTANCE_RATE,
        model_name=QWEN35_MTP_DP_MODEL,
        async_scheduling=False,
        extra_kwargs=_qwen35_dflash_dp_kwargs(),
    )
