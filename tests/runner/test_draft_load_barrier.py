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
import multiprocessing
import socket
import textwrap
import time
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch.distributed as dist
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
def test_the_world_group_gets_the_cpu_timeout_only_when_set(seconds):
    config = ParallelConfig(cpu_distributed_timeout_seconds=seconds)
    expected = {} if seconds is None else {"timeout": timedelta(seconds=seconds)}
    assert tpu_worker._world_group_init_kwargs(config) == expected


def test_init_device_hands_the_timeout_to_the_world_group():
    """init_device passes the helper's arguments to init_distributed_environment;
    without them the world TCPStore keeps torch's 30 min default. The world
    group's gloo cpu_group takes its timeout from the vLLM config either way
    (the two-rank test below)."""
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


# The failure the barrier fixes, on two gloo ranks. torch_tpu's new_group,
# which the draft load calls, waits a fixed 300 s for every rank; a
# monitored_barrier with a 1 s timeout stands in for it, and rank 1 finishes
# "loading the target weights" 3 s after rank 0.
_GROUP_WAIT = timedelta(seconds=1)
_LATE_S = 3.0


def _draft_load_on_rank(rank, port, align, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        if rank == 1:
            time.sleep(_LATE_S)
        if align:
            runner = SimpleNamespace(
                parallel_config=SimpleNamespace(cpu_distributed_timeout=None)
            )
            world = SimpleNamespace(world_size=2, rank=rank, cpu_group=dist.group.WORLD)
            with patch.object(tpu_runner, "get_world_group", return_value=world):
                TPUModelRunner._draft_load_barrier(runner)
        dist.monitored_barrier(timeout=_GROUP_WAIT, wait_all_ranks=True)
        results.put((rank, "ok"))
    except Exception as e:  # the stand-in's timeout, reported to the parent
        results.put((rank, type(e).__name__))
    finally:
        dist.destroy_process_group()


def _draft_load_outcomes(align):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    procs = [
        ctx.Process(target=_draft_load_on_rank, args=(rank, port, align, results))
        for rank in range(2)
    ]
    try:
        for proc in procs:
            proc.start()
        return dict(results.get(timeout=240) for _ in procs)
    finally:
        for proc in procs:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)


def test_a_late_rank_fails_the_fixed_group_wait_without_the_barrier():
    assert _draft_load_outcomes(align=False)[0] != "ok"


def test_the_barrier_lets_a_late_rank_pass_the_fixed_group_wait():
    assert _draft_load_outcomes(align=True) == {0: "ok", 1: "ok"}


# The world group's gloo cpu_group, which the draft load barrier waits on, gets
# vLLM's CPU timeout: vLLM's GroupCoordinator builds it with new_group(...,
# timeout=...) from the current vLLM config, whatever the default group's own
# timeout. Here the default group waits 120 s, the option says 2 s, and rank 1
# stays away from the barrier until rank 0 has given up on it; gloo's error
# names the wait that fired.
_CPU_TIMEOUT_S = 2
_DEFAULT_GROUP_TIMEOUT = timedelta(seconds=120)


def _world_cpu_group_wait_on_rank(rank, port, results, rank0_done):
    import vllm.platforms
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.platforms.cpu import CpuPlatform

    # The group coordinator also picks a device from the platform; without a
    # TPU that has to be the CPU. The CPU group it builds does not depend on it.
    vllm.platforms._current_platform = CpuPlatform()
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=_DEFAULT_GROUP_TIMEOUT,
    )
    try:
        config = SimpleNamespace(
            parallel_config=ParallelConfig(
                cpu_distributed_timeout_seconds=_CPU_TIMEOUT_S
            )
        )
        try:
            with set_current_vllm_config(config):
                world = parallel_state.init_world_group([0, 1], rank, "gloo")
        except Exception as e:  # reported at once instead of a queue timeout
            results.put((rank, f"setup failed: {type(e).__name__}: {e}"))
            return
        if rank == 1:
            rank0_done.wait(timeout=120)
            results.put((rank, "absent"))
            return
        try:
            dist.barrier(group=world.cpu_group)
            outcome = "passed"
        except Exception as e:  # the gloo timeout, reported to the parent
            outcome = f"{type(e).__name__}: {e}"
        results.put((rank, outcome))
    finally:
        if rank == 0:
            rank0_done.set()
        dist.destroy_process_group()


def test_the_world_cpu_group_waits_the_cpu_timeout_not_the_default_groups():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ctx = multiprocessing.get_context("spawn")
    results = ctx.Queue()
    rank0_done = ctx.Event()
    procs = [
        ctx.Process(
            target=_world_cpu_group_wait_on_rank, args=(rank, port, results, rank0_done)
        )
        for rank in range(2)
    ]
    try:
        for proc in procs:
            proc.start()
        outcomes = dict(results.get(timeout=240) for _ in procs)
    finally:
        rank0_done.set()
        for proc in procs:
            proc.join(timeout=60)
            if proc.is_alive():
                proc.terminate()
                proc.join(5)
    assert not any(o.startswith("setup failed") for o in outcomes.values()), outcomes
    assert outcomes[1] == "absent"
    assert f"Timed out waiting {_CPU_TIMEOUT_S * 1000}ms" in outcomes[0], outcomes
