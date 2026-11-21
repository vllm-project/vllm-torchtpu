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
"""Platform gates for the pipeline chunk scheduler."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_torchtpu.compilation import tpu_compiler
from vllm_torchtpu.core.pp_chunk_scheduler import (
    PREFIX_AWARE_PP_SCHEDULER_CLS,
    SCHEDULER_CLS,
    uses_dynamic_chunks,
)
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


def _env(monkeypatch, lone_prompt=True, equal_cost=True, dp_sched=False, slack=0.1):
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_LONE_PROMPT_CHUNKS", lone_prompt)
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_EQUAL_COST_CHUNKS", equal_cost)
    monkeypatch.setattr(tpu_platform.envs, "DP_SCHED_ENABLED", dp_sched)
    monkeypatch.setattr(tpu_platform.envs, "TPU_PP_CHUNK_SLACK", slack)


def _configure(config):
    with patch(
        "vllm_torchtpu.core.pp_chunk_scheduler.patch_engine_core_for_pp_chunks"
    ) as hook:
        tpu_platform._configure_pipeline_chunks(config)
    return hook


@pytest.mark.parametrize(
    "lone_prompt,equal_cost", [(True, True), (True, False), (False, True)]
)
def test_pipeline_gets_the_chunk_scheduler_and_the_engine_hook(
    monkeypatch, lone_prompt, equal_cost
):
    _env(monkeypatch, lone_prompt=lone_prompt, equal_cost=equal_cost)
    config = _config()
    hook = _configure(config)
    assert config.scheduler_config.scheduler_cls == SCHEDULER_CLS
    hook.assert_called_once_with(config)


@pytest.mark.parametrize(
    "both,dp_sched,dp,cls,chunked",
    [
        (False, False, 1, None, True),
        (True, True, 2, None, True),
        (True, False, 1, "x.Y", True),
        (True, False, 1, None, False),
    ],
)
def test_chunk_scheduler_is_off_when_gated(
    monkeypatch, both, dp_sched, dp, cls, chunked
):
    _env(monkeypatch, lone_prompt=both, equal_cost=both, dp_sched=dp_sched)
    config = _config(dp=dp, cls=cls, chunked=chunked)
    hook = _configure(config)
    assert config.scheduler_config.scheduler_cls == cls
    hook.assert_not_called()


def test_the_dp_scheduler_with_one_dp_rank_does_not_gate(monkeypatch):
    _env(monkeypatch, dp_sched=True)
    config = _config(dp=1)
    _configure(config)
    assert config.scheduler_config.scheduler_cls == SCHEDULER_CLS


def test_custom_scheduler_logs_configured_scheduler_when_dp_sched_env_set(
    monkeypatch, caplog
):
    _env(monkeypatch, dp_sched=True)
    config = _config(dp=2, cls="x.Y")
    hook = _configure(config)
    assert config.scheduler_config.scheduler_cls == "x.Y"
    hook.assert_not_called()
    assert "scheduler x.Y is configured" in caplog.text
    assert "the DP scheduler is enabled" not in caplog.text


def test_negative_slack_fails_at_config_time(monkeypatch):
    _env(monkeypatch, slack=-0.5)
    with pytest.raises(ValueError, match="TPU_PP_CHUNK_SLACK"):
        _configure(_config())


def test_slack_is_not_checked_without_equal_cost_chunks(monkeypatch):
    _env(monkeypatch, equal_cost=False, slack=-0.5)
    config = _config()
    _configure(config)
    assert config.scheduler_config.scheduler_cls == SCHEDULER_CLS


def test_pipeline_knobs_do_not_change_the_compile_cache_key():
    assert "TPU_PP_LONE_PROMPT_CHUNKS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "TPU_PP_EQUAL_COST_CHUNKS" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "TPU_PP_CHUNK_SLACK" in tpu_compiler._TPU_COMPILE_ENV_IGNORED
    assert "TPU_PP_HANDOFF_DEPTH" in tpu_compiler._TPU_COMPILE_ENV_IGNORED


def test_pipeline_preserves_prefix_aware_pipeline_chunk_scheduler(monkeypatch):
    _env(monkeypatch)
    config = _config(cls=PREFIX_AWARE_PP_SCHEDULER_CLS)
    hook = _configure(config)
    assert config.scheduler_config.scheduler_cls == PREFIX_AWARE_PP_SCHEDULER_CLS
    assert uses_dynamic_chunks(config) is True
    hook.assert_called_once_with(config)
