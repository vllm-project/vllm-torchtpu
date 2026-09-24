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

pytestmark = pytest.mark.cpu_test


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
    is_moe_model=None,
    enable_expert_parallel=False,
):
    return SimpleNamespace(
        model_config=None,
        additional_config={},
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=pcp_size,
            cp_kv_cache_interleave_size=interleave_size,
            decode_context_parallel_size=dcp_size,
            pipeline_parallel_size=pipeline_parallel_size,
            enable_expert_parallel=enable_expert_parallel,
            is_moe_model=is_moe_model,
        ),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        speculative_config=(
            None
            if speculative_method is None
            else SimpleNamespace(
                method=speculative_method,
                num_speculative_tokens=num_speculative_tokens,
            )
        ),
        kv_transfer_config=(
            None
            if kv_role is None
            else SimpleNamespace(
                kv_role=kv_role,
                is_kv_producer=kv_role in ("kv_producer", "kv_both"),
            )
        ),
    )


def test_from_vllm_config_without_speculation():
    config = PcpStaticSupportValidator.from_vllm_config(
        _vllm_config(kv_role="kv_consumer")
    )

    assert config.enabled
    assert config.pcp_size == 4
    assert config.interleave_size == 16
    assert config.speculative_method is None
    assert config.num_speculative_tokens == 0


@pytest.mark.parametrize(
    ("config", "error_type", "message"),
    [
        (_vllm_config(dcp_size=2), NotImplementedError, "DCP"),
        (
            _vllm_config(pipeline_parallel_size=2),
            NotImplementedError,
            "pipeline parallelism",
        ),
        (
            _vllm_config(
                speculative_method="eagle3",
                num_speculative_tokens=1,
                kv_role="kv_producer",
            ),
            NotImplementedError,
            "method=mtp",
        ),
        (
            _vllm_config(
                speculative_method="mtp",
                num_speculative_tokens=2,
                kv_role="kv_producer",
            ),
            NotImplementedError,
            "num_speculative_tokens=1",
        ),
        (
            _vllm_config(
                speculative_method="mtp", num_speculative_tokens=3, kv_role="kv_both"
            ),
            NotImplementedError,
            "num_speculative_tokens=1",
        ),
        (
            _vllm_config(interleave_size=0),
            ValueError,
            "cp_kv_cache_interleave_size > 0",
        ),
        (
            _vllm_config(is_moe_model=True),
            NotImplementedError,
            "requires --enable-expert-parallel",
        ),
    ],
)
def test_static_validator_rejects_unsupported_platform_config(
    config, error_type, message
):
    with pytest.raises(error_type, match=message):
        PcpStaticSupportValidator.validate_platform_config(
            config,
            kv_cache_layout="NHD",
        )


def test_static_validator_accepts_supported_pcp_platform_config():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(),
        kv_cache_layout="NHD",
    )

    assert config.enabled
    assert config.pcp_size == 4


def test_static_validator_accepts_moe_pcp_with_expert_parallel():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(is_moe_model=True, enable_expert_parallel=True),
        kv_cache_layout="NHD",
    )

    assert config.is_moe is True
    assert config.expert_parallel is True


def test_static_validator_ignores_expert_parallel_without_pcp():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(pcp_size=1, is_moe_model=True),
        kv_cache_layout="NHD",
    )

    assert config.enabled is False


@pytest.mark.parametrize("is_moe_model", [None, False])
def test_static_validator_accepts_pcp_when_model_is_not_moe(is_moe_model):
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(is_moe_model=is_moe_model),
        kv_cache_layout="NHD",
    )

    assert config.is_moe is False


@pytest.mark.parametrize("nnodes", [2, 4])
def test_static_validator_accepts_multihost_pcp_platform_config(nnodes):
    vllm_config = _vllm_config()
    vllm_config.parallel_config.nnodes = nnodes
    config = PcpStaticSupportValidator.validate_platform_config(
        vllm_config,
        kv_cache_layout="NHD",
    )

    assert config.enabled
    assert config.pcp_size == 4


def test_static_validator_accepts_pcp_async_non_speculative_config():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(async_scheduling=True),
        kv_cache_layout="NHD",
    )

    assert config.enabled
    assert config.async_scheduling is True


def test_pcp_hnd_accepts_aligned_pcp_config():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(pcp_size=8, interleave_size=128),
        kv_cache_layout="HND",
    )

    assert config.enabled
    assert config.pcp_size == 8


def test_pcp_hnd_rejects_unaligned_interleave():
    with pytest.raises(ValueError, match="multiple of 128"):
        PcpStaticSupportValidator.validate_platform_config(
            _vllm_config(pcp_size=8, interleave_size=64),
            kv_cache_layout="HND",
        )


def test_hnd_without_pcp_remains_valid():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(pcp_size=1, interleave_size=64),
        kv_cache_layout="HND",
    )

    assert not config.enabled


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("kv_role", [None, "kv_producer", "kv_consumer", "kv_both"])
def test_static_validator_accepts_pcp_mtp_k1_for_any_kv_role(async_scheduling, kv_role):
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(
            speculative_method="mtp",
            num_speculative_tokens=1,
            kv_role=kv_role,
            async_scheduling=async_scheduling,
        ),
        kv_cache_layout="NHD",
    )

    assert config.enabled is True
    assert config.speculative_enabled is True
    assert config.speculative_method == "mtp"
    assert config.num_speculative_tokens == 1
    assert config.async_scheduling is async_scheduling


def test_static_validator_does_not_restrict_non_pcp_mtp_k3_consumer():
    config = PcpStaticSupportValidator.validate_platform_config(
        _vllm_config(
            pcp_size=1,
            speculative_method="mtp",
            num_speculative_tokens=3,
            kv_role="kv_consumer",
        ),
        kv_cache_layout="NHD",
    )

    assert config.enabled is False
