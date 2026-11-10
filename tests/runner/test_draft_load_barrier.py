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
"""The draft load barrier: ranks meet on the gloo world group before the draft
model load, with vLLM's CPU distributed timeout when it is set, and the CPU
collective timeout reaches the world group from the same vLLM option."""

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import ParallelConfig

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _run_barrier(world_size, timeout):
    runner = SimpleNamespace(
        parallel_config=SimpleNamespace(cpu_distributed_timeout=timeout)
    )
    world = SimpleNamespace(world_size=world_size, rank=0, cpu_group="gloo-world")
    with (
        patch.object(tpu_runner, "get_world_group", return_value=world),
        patch("torch.distributed.barrier") as barrier,
        patch("torch.distributed.monitored_barrier") as monitored,
    ):
        TPUModelRunner._draft_load_barrier(runner)
    return barrier, monitored


def test_single_rank_world_skips_the_barrier():
    barrier, monitored = _run_barrier(1, timedelta(hours=3))
    barrier.assert_not_called()
    monitored.assert_not_called()


def test_without_a_timeout_ranks_meet_on_a_plain_barrier():
    barrier, monitored = _run_barrier(8, None)
    barrier.assert_called_once_with(group="gloo-world")
    monitored.assert_not_called()


def test_with_a_timeout_ranks_meet_on_a_monitored_barrier():
    barrier, monitored = _run_barrier(8, timedelta(seconds=10800))
    barrier.assert_not_called()
    monitored.assert_called_once_with(
        group="gloo-world", timeout=timedelta(seconds=10800), wait_all_ranks=True
    )


@pytest.mark.parametrize("seconds", [None, 10800])
def test_vllm_parallel_config_exposes_the_cpu_timeout(seconds):
    """The barrier and the worker read ParallelConfig.cpu_distributed_timeout,
    which --cpu-distributed-timeout-seconds sets in vLLM."""
    config = ParallelConfig(cpu_distributed_timeout_seconds=seconds)
    expected = None if seconds is None else timedelta(seconds=seconds)
    assert config.cpu_distributed_timeout == expected
