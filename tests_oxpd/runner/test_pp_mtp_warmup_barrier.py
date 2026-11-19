# SPDX-License-Identifier: Apache-2.0
"""TPU_PP_MTP_WARMUP_BARRIER: pipeline stages meet at the host barrier of
``_pp_warmup_barrier`` before each prompt length of the MTP prefill warmup,
so a last stage that cold-compiles its draft inside one prompt cannot hold
the stages ahead of it on the ring's 300 s cross-host descriptor wait
(q38 x1c r j-ab73d6ff)."""

import contextlib
from types import SimpleNamespace

import pytest

from vllm_torchtpu import envs
from vllm_torchtpu.runner import tpu_runner
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

PADDINGS = [16, 32, 6400]


def _runner(events, pp=8, fail_at=None):
    def one(P, quiet=False):
        events.append(("step", P, quiet))
        return P != fail_at

    runner = SimpleNamespace(
        enforce_eager=False,
        speculative_config=object(),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, pipeline_parallel_size=pp
        ),
        max_model_len=8192,
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8192),
        num_tokens_paddings=list(PADDINGS),
        num_xla_graphs=0,
        _pp_wave=SimpleNamespace(settle=lambda: events.append(("settle",))),
        _precompile_timed=lambda name: contextlib.nullcontext(),
        _warmup_one_mtp_prefill=one,
    )
    runner._pp_warmup_barrier = TPUModelRunner._pp_warmup_barrier.__get__(runner)
    return runner


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.delenv("SPEC_WARMUP", raising=False)
    monkeypatch.setattr(
        tpu_runner, "synchronize_device", lambda: seen.append(("sync",))
    )
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


def _without_group(events):
    return [e for e in events if e[0] != "group"]


def test_off_by_default_no_barrier(monkeypatch, events):
    monkeypatch.delenv("TPU_PP_MTP_WARMUP_BARRIER", raising=False)
    assert envs.TPU_PP_MTP_WARMUP_BARRIER is False
    TPUModelRunner._warmup_mtp_prefill(_runner(events))
    # The flag-off sequence is the one the tree had before the flag.
    assert _without_group(events) == [
        e for P in PADDINGS for e in (("step", P, False), ("settle",))
    ]


def test_on_barrier_on_cpu_group_before_each_prompt(monkeypatch, events):
    monkeypatch.setenv("TPU_PP_MTP_WARMUP_BARRIER", "1")
    TPUModelRunner._warmup_mtp_prefill(_runner(events))
    cpu_group = events[0][1]
    # Each prompt: the device drains, the stages meet on the world group's
    # CPU group, then the prompt runs and settles.
    assert _without_group(events) == [
        e
        for P in PADDINGS
        for e in (("sync",), ("barrier", cpu_group), ("step", P, False), ("settle",))
    ]


def test_on_hand_off_barrier_flag_alone_does_not_add_it(monkeypatch, events):
    monkeypatch.delenv("TPU_PP_MTP_WARMUP_BARRIER", raising=False)
    monkeypatch.setenv("TPU_PP_WARMUP_BARRIER", "1")
    TPUModelRunner._warmup_mtp_prefill(_runner(events))
    assert [e for e in events if e[0] == "barrier"] == []


def test_on_without_pipeline_no_barrier(monkeypatch, events):
    monkeypatch.setenv("TPU_PP_MTP_WARMUP_BARRIER", "1")
    TPUModelRunner._warmup_mtp_prefill(_runner(events, pp=1))
    assert [e for e in events if e[0] == "barrier"] == []
    assert [e[1:] for e in events if e[0] == "step"] == [(P, True) for P in PADDINGS]


def test_on_failed_prompt_still_refuses(monkeypatch, events):
    monkeypatch.setenv("TPU_PP_MTP_WARMUP_BARRIER", "1")
    with pytest.raises(RuntimeError, match="MTP prefill warmup failed at P=32"):
        TPUModelRunner._warmup_mtp_prefill(_runner(events, fail_at=32))
    kinds = [e[0] for e in _without_group(events)]
    assert kinds == ["sync", "barrier", "step", "settle"] * 2


def test_barrier_log_lines(monkeypatch, events):
    lines = []
    monkeypatch.setattr(
        tpu_runner,
        "logger",
        SimpleNamespace(info=lambda fmt, *args: lines.append(fmt % args)),
    )
    runner = _runner(events)
    runner._pp_warmup_barrier()
    runner._pp_warmup_barrier("MTP prefill warmup P=16")
    # The hand-off barrier's line is unchanged, so its readers still match.
    assert lines[0].startswith("Pipeline hand-off warmup barrier: waited ")
    assert lines[1].startswith("Pipeline MTP prefill warmup P=16 barrier: waited ")
    assert all(line.endswith(" s for the other stages") for line in lines)


def test_registered_and_compile_neutral(monkeypatch):
    from vllm_torchtpu.compilation import tpu_compiler

    assert "TPU_PP_MTP_WARMUP_BARRIER" in envs.environment_variables
    assert "TPU_PP_MTP_WARMUP_BARRIER" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    monkeypatch.delenv("TPU_PP_MTP_WARMUP_BARRIER", raising=False)
    off = tpu_compiler._tpu_compile_env_factors()
    monkeypatch.setenv("TPU_PP_MTP_WARMUP_BARRIER", "1")
    on = tpu_compiler._tpu_compile_env_factors()
    assert "TPU_PP_MTP_WARMUP_BARRIER" not in off
    assert on == off
