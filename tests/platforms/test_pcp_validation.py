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

from types import SimpleNamespace

import pytest

from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator


def _vllm_config(
    *,
    pcp_size=4,
    interleave_size=16,
    dcp_size=1,
    pipeline_parallel_size=1,
    async_scheduling=False,
    speculative_enabled=False,
    is_kv_producer=None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            decode_context_parallel_size=dcp_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=(object() if speculative_enabled else None),
        kv_transfer_config=(None if is_kv_producer is None else
                            SimpleNamespace(is_kv_producer=is_kv_producer)),
    )


def test_from_vllm_config_records_kv_role():
    config = PcpStaticSupportValidator.from_vllm_config(
        _vllm_config(is_kv_producer=False))

    assert config.enabled
    assert config.pcp_size == 4
    assert config.interleave_size == 16
    assert config.is_kv_producer is False


@pytest.mark.parametrize(
    ("config", "multihost_backend", "error_type", "message"),
    [
        (_vllm_config(is_kv_producer=False), "", NotImplementedError,
         "KV consumer"),
        (_vllm_config(dcp_size=2), "", NotImplementedError, "DCP"),
        (_vllm_config(pipeline_parallel_size=2), "", NotImplementedError,
         "pipeline parallelism"),
        (_vllm_config(speculative_enabled=True), "", NotImplementedError,
         "speculative decoding"),
        (_vllm_config(), "ray", NotImplementedError, "multihost"),
    ],
)
def test_static_validator_rejects_unsupported_platform_config(
        config, multihost_backend, error_type, message):
    with pytest.raises(error_type, match=message):
        PcpStaticSupportValidator.validate_platform_config(
            config,
            multihost_backend=multihost_backend,
        )


def test_static_validator_accepts_supported_pcp_platform_config():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(),
        multihost_backend="",
    )

    assert config.enabled
    assert config.pcp_size == 4


def test_static_validator_accepts_pcp_async_non_speculative_config():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(async_scheduling=True),
        multihost_backend="",
    )

    assert config.enabled
    assert config.async_scheduling is True
