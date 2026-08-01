"""
End-to-end tests for vLLM LLM generation API with async scheduling on TPU.

This module tests the async scheduling feature by comparing its output against
synchronous scheduling.
"""

import multiprocessing
import os
import traceback

import pytest
from vllm import LLM, SamplingParams

from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Capture the original __init__ to allow safe monkeypatching and clean restoration.
_ORIGINAL_TPU_RUNNER_INIT = TPUModelRunner.__init__


def force_multiple_forward_passes_init(self, *args, **kwargs):
    """
    A custom __init__ for TPUModelRunner that artificially limits the number of
    requests processed in a single forward pass.
    This forces the runner to split a batch into multiple chunks during the same engine step.
    """
    _ORIGINAL_TPU_RUNNER_INIT(self, *args, **kwargs)
    self.num_reqs_max_model_len = 2
    if hasattr(self, "num_reqs_most_model_len"):
        self.num_reqs_most_model_len = 2


def _run_generation_worker(
    async_scheduling: bool,
    prompts: list[str],
    max_tokens: int,
    max_num_seqs: int,
    monkeypatch_tpu_runner_init,
    queue: multiprocessing.Queue,
):
    """
    Helper function to instantiate LLM and run generation within a child process.
    """
    try:
        # Avoid JAX hanging during sequential tests
        os.environ["SKIP_JAX_PRECOMPILE"] = "1"

        # We apply the monkeypatch dynamically
        if monkeypatch_tpu_runner_init is not None:
            TPUModelRunner.__init__ = monkeypatch_tpu_runner_init

        llm = LLM(
            model=MODEL_NAME,
            max_model_len=256,
            max_num_batched_tokens=16,  # Ensures the long prompt gets chunked.
            max_num_seqs=max_num_seqs,
            enforce_eager=False,
            async_scheduling=async_scheduling,
            disable_log_stats=True,
        )

        sampling_params = SamplingParams(temperature=0.0,
                                         max_tokens=max_tokens)
        outputs = llm.generate(prompts, sampling_params=sampling_params)

        tokens = [o.outputs[0].token_ids for o in outputs]
        queue.put(("SUCCESS", tokens))
    except Exception:
        queue.put(("ERROR", traceback.format_exc()))
    finally:
        # Restore the original __init__ to guarantee we do not pollute subsequent runs
        if monkeypatch_tpu_runner_init is not None:
            TPUModelRunner.__init__ = _ORIGINAL_TPU_RUNNER_INIT


def run_llm_generation_in_isolated_process(
    async_scheduling: bool,
    prompts: list[str],
    max_tokens: int,
    max_num_seqs: int,
    monkeypatch_tpu_runner_init=None,
) -> list[list[int]]:
    """
    Spawns a clean OS process to run the LLM generation and retrieves the token IDs.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    p = ctx.Process(target=_run_generation_worker,
                    args=(async_scheduling, prompts, max_tokens, max_num_seqs,
                          monkeypatch_tpu_runner_init, queue))
    p.start()
    p.join()
    status, result = queue.get()
    if status == "ERROR":
        raise RuntimeError(f"Isolated worker failed:\n{result}")
    return result


@pytest.mark.timeout(1200)
def test_async_scheduling_preserves_output_when_chunked_prefill_requests_are_discarded(
):
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
        "The quick brown fox jumps over the lazy dog. " * 4
    ]

    # Parameters to trigger the edge case:
    max_tokens = 10  # Ensures we generate enough tokens to observe the async substitution.
    max_num_seqs = 2  # Ensures both prompts are batched together.
    sync_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=False,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs)

    async_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=True,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs)

    assert sync_tokens == async_tokens, (
        "Async scheduling corrupted output for discarded chunked prefill requests."
    )


@pytest.mark.timeout(1200)
def test_async_scheduling_preserves_output_when_batch_is_split_into_multiple_forward_passes(
):
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
        "The quick brown fox jumps over the lazy dog. " * 4
    ]

    max_tokens = 2  # Limits the runtime since we only need to verify the first few tokens.
    max_num_seqs = 4  # Ensures 4 prompts are batched together by the scheduler.

    # - We use monkeypatch_tpu_runner_init to force TPUModelRunner to split
    #   this batch of 4 into 2 execution chunks (2 requests each).
    sync_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=False,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
        monkeypatch_tpu_runner_init=force_multiple_forward_passes_init)

    async_tokens = run_llm_generation_in_isolated_process(
        async_scheduling=True,
        prompts=prompts,
        max_tokens=max_tokens,
        max_num_seqs=max_num_seqs,
        monkeypatch_tpu_runner_init=force_multiple_forward_passes_init)

    assert sync_tokens == async_tokens, (
        "Async scheduling corrupted output during chunked multi-pass generation!"
    )
