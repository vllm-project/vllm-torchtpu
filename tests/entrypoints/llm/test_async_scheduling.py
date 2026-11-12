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
"""
End-to-end tests for vLLM LLM generation API with async scheduling on TPU.

This module tests the async scheduling feature by comparing its output against
synchronous scheduling.
"""

import multiprocessing
import traceback

import pytest
from vllm import LLM, SamplingParams

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

pytestmark = pytest.mark.tpu_test

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Capture the original refresh to allow safe monkeypatching and clean restoration.
# The runner recomputes `num_reqs_max_model_len` / `num_reqs_most_model_len` in
# `_refresh_smem_request_caps` whenever the KV block size is finalized
# (`get_kv_cache_spec`, `profile_run`, `may_reinitialize_input_batch`), so a
# value pinned in `__init__` would be overwritten before the first forward.
_ORIGINAL_TPU_RUNNER_REFRESH = TPUModelRunner._refresh_smem_request_caps


def force_multiple_forward_passes_refresh(self, *args, **kwargs):
    """
    A custom `_refresh_smem_request_caps` for TPUModelRunner that artificially
    limits the number of requests processed in a single forward pass.
    This forces the runner to split a batch into multiple chunks during
    the same engine step, and survives every later cap refresh.
    """
    _ORIGINAL_TPU_RUNNER_REFRESH(self, *args, **kwargs)
    self.num_reqs_max_model_len = 2
    if self.num_reqs_most_model_len is not None:
        self.num_reqs_most_model_len = 2


def _run_generation_worker(
    async_scheduling: bool,
    prompts: list[str],
    max_tokens: int,
    max_num_seqs: int,
    monkeypatch_tpu_runner_refresh,
    queue: multiprocessing.Queue,
):
    """
    Helper function to instantiate LLM and run generation within a child process.
    """
    try:
        # We apply the monkeypatch dynamically
        if monkeypatch_tpu_runner_refresh is not None:
            TPUModelRunner._refresh_smem_request_caps = monkeypatch_tpu_runner_refresh

        llm = LLM(
            model=MODEL_NAME,
            max_model_len=256,
            max_num_batched_tokens=16,  # Ensures the long prompt gets chunked.
            max_num_seqs=max_num_seqs,
            enforce_eager=False,
            async_scheduling=async_scheduling,
            disable_log_stats=True,
        )

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        tokens = [o.outputs[0].token_ids for o in outputs]
        queue.put(("SUCCESS", tokens))
    except Exception:
        queue.put(("ERROR", traceback.format_exc()))
    finally:
        # Restore the original refresh to guarantee we do not pollute subsequent runs
        if monkeypatch_tpu_runner_refresh is not None:
            TPUModelRunner._refresh_smem_request_caps = _ORIGINAL_TPU_RUNNER_REFRESH


def run_llm_generation_in_isolated_process(
    async_scheduling: bool,
    prompts: list[str],
    max_tokens: int,
    max_num_seqs: int,
    monkeypatch_tpu_runner_refresh=None,
) -> list[list[int]]:
    """
    Spawns a clean OS process to run the LLM generation and retrieves the token IDs.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(
        target=_run_generation_worker,
        args=(
            async_scheduling,
            prompts,
            max_tokens,
            max_num_seqs,
            monkeypatch_tpu_runner_refresh,
            queue,
        ),
    )
    p.start()
    p.join()
    status, result = queue.get()
    if status == "ERROR":
        raise RuntimeError(f"Isolated worker failed:\n{result}")
    return result


@pytest.mark.timeout(1200)
def test_async_scheduling_preserves_output_when_chunked_prefill_requests_are_discarded():  # noqa: E501
    """
    When a long prompt is processed over multiple forward passes, it is 'discarded' from
    the current step's output because no token is sampled.
    """
    prompts = [
        # A short prompt that finishes prefill early and generates tokens while the
        # long prompt is still chunking, providing the 'garbage' token to be erroneously
        # substituted.
        "Short prompt",
        # A very long prompt that will exceed `max_num_batched_tokens` and force
        # multiple chunked prefill passes.
        "The quick brown fox jumps over the lazy dog. " * 4,
    ]

    # Parameters to trigger the edge case:
    max_tokens = (
        10  # Ensures we generate enough tokens to observe the async substitution.
    )
    max_num_seqs = 2  # Ensures both prompts are batched together.
    sync_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=False,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
    )

    async_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=True,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
    )

    assert sync_tokens == async_tokens, (
        "Async scheduling corrupted output for discarded chunked prefill requests."
    )


@pytest.mark.timeout(1200)
def test_async_scheduling_preserves_output_when_batch_is_split_into_multiple_forward_passes():  # noqa: E501
    """
    Ensure we are correctly offseting into the request batch when the batch is chunked
    into multiple forward passes.
    """
    prompts = [
        "Prompt 1",
        "Prompt 2",
        "Prompt 3",
        "Prompt 4",
        # Including a long prompt to maintain mixed states within the scheduler queue
        "The quick brown fox jumps over the lazy dog. " * 4,
    ]

    max_tokens = (
        2  # Limits the runtime since we only need to verify the first few tokens.
    )
    max_num_seqs = 4  # Ensures 4 prompts are batched together by the scheduler.

    # - We use monkeypatch_tpu_runner_refresh to force TPUModelRunner to split
    #   this batch of 4 into 2 execution chunks (2 requests each).
    sync_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=False,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
        monkeypatch_tpu_runner_refresh=force_multiple_forward_passes_refresh,
    )

    async_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=True,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
        monkeypatch_tpu_runner_refresh=force_multiple_forward_passes_refresh,
    )

    assert sync_tokens == async_tokens, (
        "Async scheduling corrupted output during chunked multi-pass generation!"
    )
