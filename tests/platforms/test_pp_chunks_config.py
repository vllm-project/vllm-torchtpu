# SPDX-License-Identifier: Apache-2.0
"""Platform gates for the pipeline chunk scheduler."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core.pp_chunk_scheduler import SCHEDULER_CLS
from vllm_torchtpu.platforms import tpu_platform


def _config(dp=1, cls=None, chunked=True):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=8, data_parallel_size=dp
        ),
        scheduler_config=SimpleNamespace(
            scheduler_cls=cls, enable_chunked_prefill=chunked
        ),
    )


def _env(monkeypatch, dynamic=True, dp_sched=False, slack=0.1):
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_DYNAMIC_CHUNKS", dynamic)
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", dp_sched)
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_CHUNK_SLACK", slack)


def _configure(config):
    with patch(
        "vllm_torchtpu.core.pp_chunk_scheduler.patch_engine_core_for_pp_chunks"
    ) as hook:
        tpu_platform._configure_pipeline_chunks(config)
    return hook


def test_pipeline_gets_the_chunk_scheduler_and_the_engine_hook(monkeypatch):
    _env(monkeypatch)
    config = _config()
    hook = _configure(config)
    assert config.scheduler_config.scheduler_cls == SCHEDULER_CLS
    hook.assert_called_once_with(config)


@pytest.mark.parametrize(
    "dynamic,dp_sched,dp,cls,chunked",
    [
        (False, False, 1, None, True),
        (True, True, 2, None, True),
        (True, False, 1, "x.Y", True),
        (True, False, 1, None, False),
    ],
)
def test_chunk_scheduler_is_off_when_gated(
    monkeypatch, dynamic, dp_sched, dp, cls, chunked
):
    _env(monkeypatch, dynamic=dynamic, dp_sched=dp_sched)
    config = _config(dp=dp, cls=cls, chunked=chunked)
    hook = _configure(config)
    assert config.scheduler_config.scheduler_cls == cls
    hook.assert_not_called()


def test_the_dp_scheduler_with_one_dp_rank_does_not_gate(monkeypatch):
    _env(monkeypatch, dp_sched=True)
    config = _config(dp=1)
    _configure(config)
    assert config.scheduler_config.scheduler_cls == SCHEDULER_CLS


def test_negative_slack_fails_at_config_time(monkeypatch):
    _env(monkeypatch, slack=-0.5)
    with pytest.raises(ValueError, match="TPU_PP_CHUNK_SLACK"):
        _configure(_config())


def test_chunk_knobs_do_not_change_the_compile_cache_key():
    assert "TPU_PP_DYNAMIC_CHUNKS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "TPU_PP_CHUNK_SLACK" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
