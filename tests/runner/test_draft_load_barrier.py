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
collective timeout reaches the world group from the same vLLM option. Every
pipeline stage joins it, not only the last one, which hosts the draft."""

import ast
import inspect
import textwrap
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import ParallelConfig

from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner
from vllm_torchtpu.worker import tpu_worker


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


@pytest.mark.parametrize("seconds", [None, 10800])
def test_the_world_group_gets_the_cpu_timeout_only_when_set(seconds):
    config = ParallelConfig(cpu_distributed_timeout_seconds=seconds)
    expected = {} if seconds is None else {"timeout": timedelta(seconds=seconds)}
    assert tpu_worker._world_group_init_kwargs(config) == expected


def test_init_device_hands_the_timeout_to_the_world_group():
    """init_device passes the helper's arguments to init_distributed_environment;
    without them the world group falls back to gloo's 30 min."""
    source = textwrap.dedent(inspect.getsource(tpu_worker.TPUWorker.init_device))
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "init_distributed_environment"
    ]
    assert len(calls) == 1
    unpacked = [ast.unparse(kw.value) for kw in calls[0].keywords if kw.arg is None]
    assert unpacked == ["_world_group_init_kwargs(self.parallel_config)"]


class _Spec:
    def __init__(self, method):
        self.method = method

    def use_eagle(self):
        return self.method in ("eagle", "eagle3", "mtp")


def _barrier_calls_per_stage(method, num_stages=8):
    """Run the pre-draft-load step on a fake world of pipeline stages, with
    the runner's own drafter predicates: only the last stage hosts the draft.
    Returns the barrier calls per stage and which stages host the draft."""
    calls = [0] * num_stages
    hosts_draft = []
    for pp_rank in range(num_stages):
        stage = SimpleNamespace(
            speculative_config=None if method is None else _Spec(method),
            _pp_is_last=pp_rank == num_stages - 1,
        )
        stage._uses_hidden_state_drafter = (
            TPUModelRunner._uses_hidden_state_drafter.fget(stage)
        )
        hosts_draft.append(TPUModelRunner._is_async_drafter.fget(stage))

        def barrier(pp_rank=pp_rank):
            calls[pp_rank] += 1

        stage._draft_load_barrier = barrier
        TPUModelRunner._maybe_draft_load_barrier(stage)
    return calls, hosts_draft


@pytest.mark.parametrize("method", ["mtp", "eagle3", "dflash"])
def test_every_pipeline_stage_joins_the_draft_load_barrier(method):
    # A world-group barrier that only the last stage entered would leave it
    # waiting for the other stages, which never arrive.
    calls, hosts_draft = _barrier_calls_per_stage(method)
    assert calls == [1] * 8
    assert hosts_draft == [False] * 7 + [True]


@pytest.mark.parametrize("method", [None, "ngram"])
def test_no_draft_load_barrier_without_a_hidden_state_drafter(method):
    calls, hosts_draft = _barrier_calls_per_stage(method)
    assert calls == [0] * 8
    assert hosts_draft == [False] * 8


def test_load_model_reaches_the_barrier_outside_the_drafter_branch():
    """load_model must reach the barrier on every rank: not inside the
    `if self._is_async_drafter:` branch, which only the last stage takes."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(TPUModelRunner.load_model)))
    parents = {
        child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
    }
    names = {"_draft_load_barrier", "_maybe_draft_load_barrier"}
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in names
    ]
    assert calls, "load_model no longer reaches the draft load barrier"
    for call in calls:
        node = call
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.If):
                assert "_is_async_drafter" not in ast.unparse(node.test), (
                    "the draft load barrier is gated on this stage hosting a draft"
                )
