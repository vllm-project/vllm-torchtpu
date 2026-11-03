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
"""Static checks for pipelined prefill across DP engines."""

from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms import dp_pipeline_validation as validation


def _config(**overrides):
    parallel = dict(
        enable_expert_parallel=False,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        data_parallel_size=4,
    )
    parallel.update(overrides.pop("parallel", {}))
    config = dict(
        parallel_config=SimpleNamespace(**parallel),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=16384, max_num_scheduled_tokens=None
        ),
        speculative_config=None,
        kv_transfer_config=None,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.delenv("TORCH_TPU_DP_SIZE", raising=False)
    monkeypatch.setattr(validation.envs, "TPU_DP_PIPELINE_PREFILL", True)
    monkeypatch.setattr(validation.envs, "TPU_DP_PIPELINE_STAGES", 4)


def test_flag_off_checks_nothing(monkeypatch):
    monkeypatch.setattr(validation.envs, "TPU_DP_PIPELINE_PREFILL", False)
    validation.validate_dp_pipeline_config(_config(speculative_config=object()))


def test_accepts_dense_dp(enabled):
    validation.validate_dp_pipeline_config(_config())


@pytest.mark.parametrize(
    "overrides, match",
    [
        (dict(parallel=dict(enable_expert_parallel=True)), "expert parallelism"),
        (dict(parallel=dict(pipeline_parallel_size=2)), "PP"),
        (
            dict(cache_config=SimpleNamespace(enable_prefix_caching=True)),
            "prefix caching",
        ),
        (dict(speculative_config=object()), "speculative decoding"),
        (dict(kv_transfer_config=object()), "KV transfer"),
        (dict(parallel=dict(data_parallel_size=2)), "between 2 and"),
        (
            dict(
                scheduler_config=SimpleNamespace(
                    max_num_batched_tokens=16384, max_num_scheduled_tokens=8192
                )
            ),
            "max-num-scheduled-tokens",
        ),
    ],
)
def test_refuses(enabled, overrides, match):
    with pytest.raises(ValueError, match=match):
        validation.validate_dp_pipeline_config(_config(**overrides))


@pytest.mark.parametrize(
    "name, value",
    [
        ("TPU_DP_PIPELINE_STAGE_TOKENS", "3:80000;4:40000"),
        ("TPU_DP_PIPELINE_STAGE_TOKENS", "2:1000"),
        ("TPU_DP_PIPELINE_SPLITS", "2:30,40"),
    ],
)
def test_refuses_malformed_stage_settings_at_startup(enabled, monkeypatch, name, value):
    monkeypatch.setattr(validation.envs, name, value)
    with pytest.raises(ValueError, match=name):
        validation.validate_dp_pipeline_config(_config())
