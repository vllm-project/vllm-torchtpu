# SPDX-License-Identifier: Apache-2.0
"""TPU_PP_WARMUP_BARRIER: pipeline stages meet at a host barrier after their
standalone compiles and before the hand-off warmup runs the buckets in step."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def _runner(events):
    def dummy_run(num_tokens, *_args, through_handoff=False, **_kwargs):
        events.append(("dummy", num_tokens, through_handoff))

    runner = SimpleNamespace(
        _pp_wave=SimpleNamespace(settle=lambda: events.append(("settle",))),
        num_tokens_paddings=[16, 6400],
        num_reqs_max_model_len=1,
        max_num_blocks_per_req=1,
        most_model_len=None,
        _dummy_run=dummy_run,
    )
    runner._pp_warmup_barrier = TPUModelRunner._pp_warmup_barrier.__get__(runner)
    return runner


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(tpu_runner, "synchronize_device", lambda: seen.append(("sync",)))
    import torch.distributed as dist
    import vllm.distributed.parallel_state as ps

    group = SimpleNamespace(cpu_group=object())
    monkeypatch.setattr(ps, "get_world_group", lambda: group)
    monkeypatch.setattr(
        dist,
        "barrier",
        lambda group=None, **_kwargs: seen.append(("barrier", group)),
    )
    seen.append(("group", group.cpu_group))
    return seen


def _calls(events, kind):
    return [e for e in events if e[0] == kind]


def test_off_by_default_no_barrier(monkeypatch, events):
    monkeypatch.delenv("TPU_PP_WARMUP_BARRIER", raising=False)
    assert envs.TPU_PP_WARMUP_BARRIER is False
    TPUModelRunner._warmup_handoff_forwards(_runner(events))
    assert _calls(events, "barrier") == []
    assert [e[1:] for e in _calls(events, "dummy")] == [(16, True), (6400, True)]


def test_on_one_barrier_on_cpu_group_before_first_handoff_run(monkeypatch, events):
    monkeypatch.setenv("TPU_PP_WARMUP_BARRIER", "1")
    TPUModelRunner._warmup_handoff_forwards(_runner(events))
    cpu_group = events[0][1]
    barriers = _calls(events, "barrier")
    assert barriers == [("barrier", cpu_group)]
    kinds = [e[0] for e in events[1:]]
    # The device drains, then the stages meet, then the buckets run in step.
    assert kinds.index("sync") < kinds.index("barrier") < kinds.index("dummy")
    assert [e[1:] for e in _calls(events, "dummy")] == [(16, True), (6400, True)]


def test_registered_and_compile_neutral():
    from vllm_torchtpu.compilation import tpu_compiler

    assert "TPU_PP_WARMUP_BARRIER" in envs.environment_variables
    assert "TPU_PP_WARMUP_BARRIER" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "TPU_PP_WARMUP_BARRIER" not in tpu_compiler._tpu_compile_env_factors()
