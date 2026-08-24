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
# Initialize vLLM's plugin registry before importing the plugin package.
import vllm  # noqa: F401

from vllm_torchtpu.platforms.pcp_validation import PcpStaticSupportValidator


def _vllm_config(
    *,
    pcp_size=4,
    interleave_size=16,
    dcp_size=1,
    pipeline_parallel_size=1,
    async_scheduling=False,
    speculative_method=None,
    num_speculative_tokens=0,
    kv_role=None,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            decode_context_parallel_size=dcp_size,
            pipeline_parallel_size=pipeline_parallel_size,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=(None
                            if speculative_method is None else SimpleNamespace(
                                method=speculative_method,
                                num_speculative_tokens=num_speculative_tokens,
                            )),
        kv_transfer_config=(None if kv_role is None else SimpleNamespace(
            kv_role=kv_role,
            is_kv_producer=kv_role in ("kv_producer", "kv_both"),
        )),
    )


def test_from_vllm_config_records_kv_role():
    config = PcpStaticSupportValidator.from_vllm_config(
        _vllm_config(kv_role="kv_consumer"))

    assert config.enabled
    assert config.pcp_size == 4
    assert config.interleave_size == 16
    assert config.is_kv_producer is False
    assert config.kv_role == "kv_consumer"
    assert config.speculative_method is None
    assert config.num_speculative_tokens == 0
    assert config.pcp_mtp_k1_enabled is False


@pytest.mark.parametrize(
    ("config", "multihost_backend", "error_type", "message"),
    [
        (_vllm_config(kv_role="kv_consumer"), "", NotImplementedError,
         "KV consumer"),
        (_vllm_config(dcp_size=2), "", NotImplementedError, "DCP"),
        (_vllm_config(pipeline_parallel_size=2), "", NotImplementedError,
         "pipeline parallelism"),
        (_vllm_config(
            speculative_method="eagle3",
            num_speculative_tokens=1,
            kv_role="kv_producer"), "", NotImplementedError, "method=mtp"),
        (_vllm_config(speculative_method="mtp",
                      num_speculative_tokens=2,
                      kv_role="kv_producer"), "", NotImplementedError,
         "num_speculative_tokens=1"),
        (_vllm_config(speculative_method="mtp", num_speculative_tokens=1), "",
         NotImplementedError, "kv_role=kv_producer"),
        (_vllm_config(), "ray", NotImplementedError, "multihost"),
        (_vllm_config(interleave_size=0), "", ValueError,
         "cp_kv_cache_interleave_size > 0"),
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


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_static_validator_accepts_pcp_mtp_k1_producer(async_scheduling):
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(
            speculative_method="mtp",
            num_speculative_tokens=1,
            kv_role="kv_producer",
            async_scheduling=async_scheduling,
        ),
        multihost_backend="",
    )

    assert config.pcp_mtp_k1_enabled is True
    assert config.speculative_enabled is True
    assert config.speculative_method == "mtp"
    assert config.num_speculative_tokens == 1
    assert config.kv_role == "kv_producer"
    assert config.async_scheduling is async_scheduling


def test_static_validator_rejects_pcp_mtp_k1_kv_both():
    vllm_config = _vllm_config(
        speculative_method="mtp",
        num_speculative_tokens=1,
        kv_role="kv_both",
    )

    config = PcpStaticSupportValidator.from_vllm_config(vllm_config)
    assert config.pcp_mtp_k1_enabled is False

    with pytest.raises(NotImplementedError, match="kv_role=kv_producer"):
        PcpStaticSupportValidator.validate_platform_config(
            vllm_config,
            multihost_backend="",
        )


def test_static_validator_does_not_restrict_non_pcp_mtp_k3_consumer():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(
            pcp_size=1,
            speculative_method="mtp",
            num_speculative_tokens=3,
            kv_role="kv_consumer",
        ),
        multihost_backend="",
    )

    assert config.enabled is False
    assert config.pcp_mtp_k1_enabled is False
