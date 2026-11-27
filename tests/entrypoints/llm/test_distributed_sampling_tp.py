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
"""End-to-end TPU multichip test for vocabulary-parallel distributed sampling.

Run with: pytest tests/entrypoints/llm/test_distributed_sampling_tp.py -v
"""

import os

import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

from vllm_torchtpu.layers.adapter.sample.distributed_sampling import (
    DistributedSamplingMode,
)
from vllm_torchtpu.runner.utils import VOCAB_PARALLEL_SAMPLING_KEY

pytestmark = pytest.mark.multichip

MODEL_NAME = "Qwen/Qwen3-0.6B"

_TPU_LAUNCH_ENV_KEYS = (
    "TORCH_TPU_TOPOLOGY",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_XPROF_SESSION_ID",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
)


def _get_local_tpu_chip_count() -> int:
    try:
        from tpu_info import device as tpu_device

        _, num_chips = tpu_device.get_local_chips()
        return num_chips
    except Exception as e:
        pytest.skip(f"Unable to detect local TPU chip count: {e}")


def _install_distributed_sampling_counters(worker):
    runner = worker.model_runner
    orig_dist = runner._sample_distributed
    orig_greedy = runner._greedy_top_tokens_this_step
    runner.dist_mode_counts = {}
    runner.greedy_fast_steps = 0

    def counted_dist(
        sample_hidden_states, mode, cur_start_idx, cur_end_idx, sampling_generator
    ):
        runner.dist_mode_counts[mode.value] = (
            runner.dist_mode_counts.get(mode.value, 0) + 1
        )
        return orig_dist(
            sample_hidden_states, mode, cur_start_idx, cur_end_idx, sampling_generator
        )

    def counted_greedy():
        took = orig_greedy()
        runner.greedy_fast_steps += int(took)
        return took

    runner._sample_distributed = counted_dist
    runner._greedy_top_tokens_this_step = counted_greedy


def test_distributed_sampling_tp_runs_topk_pure_temp_mixed_and_greedy(monkeypatch):
    num_chips = _get_local_tpu_chip_count()
    if num_chips < 2:
        pytest.skip(f"Distributed sampling TP test needs >= 2 chips, found {num_chips}")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Save/restore TPU distributed bootstrap env vars mutated by TpuPlatform so
    # this TP=2 test does not leak a 2-chip TORCH_TPU_TOPOLOGY into subsequent
    # TP=8 tests running in the same pytest process.
    saved_env = {k: os.environ.get(k) for k in _TPU_LAUNCH_ENV_KEYS}
    llm = None
    try:
        for k in ("TORCH_TPU_TOPOLOGY", "TORCH_TPU_SLICEBUILDER_ADDRESSES"):
            os.environ.pop(k, None)
        llm = LLM(
            model=MODEL_NAME,
            max_num_batched_tokens=32,
            max_num_seqs=8,
            max_model_len=64,
            tensor_parallel_size=2,
            async_scheduling=True,
            gpu_memory_utilization=0.6,
            enforce_eager=False,
            additional_config={VOCAB_PARALLEL_SAMPLING_KEY: "all"},
        )
        llm.collective_rpc(_install_distributed_sampling_counters)
        out_topk = llm.generate(
            ["Explain tensor parallelism in one sentence."],
            sampling_params=SamplingParams(
                temperature=0.8, top_k=64, top_p=0.95, max_tokens=6
            ),
        )
        out_pure = llm.generate(
            ["Name three prime numbers."],
            sampling_params=SamplingParams(temperature=1.0, max_tokens=6),
        )
        out_mixed = llm.generate(
            ["Count from one to five.", "What color is the sky?"],
            sampling_params=[
                SamplingParams(temperature=0.8, top_k=32, top_p=0.95, max_tokens=6),
                SamplingParams(temperature=1.0, max_tokens=6),
            ],
        )
        out_greedy = llm.generate(
            ["2 + 2 ="],
            sampling_params=SamplingParams(temperature=0.0, max_tokens=4),
        )
        mode_counts = llm.collective_rpc(
            lambda worker: worker.model_runner.dist_mode_counts
        )
        greedy_steps = llm.collective_rpc(
            lambda worker: worker.model_runner.greedy_fast_steps
        )
    finally:
        if llm is not None:
            del llm
        cleanup_dist_env_and_memory()
        for k, val in saved_env.items():
            if val is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = val

    for outs in (out_topk, out_pure, out_mixed, out_greedy):
        assert all(len(o.outputs[0].token_ids) > 0 for o in outs)

    rank0_counts = mode_counts[0]
    assert rank0_counts.get(DistributedSamplingMode.TOPK_TOPP.value, 0) > 0
    assert rank0_counts.get(DistributedSamplingMode.PURE_TEMP.value, 0) > 0
    assert rank0_counts.get(DistributedSamplingMode.MIXED.value, 0) > 0
    assert greedy_steps[0] > 0
