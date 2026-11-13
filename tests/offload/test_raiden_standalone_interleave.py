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
"""CPU unit tests for _patch_raiden_identity_keeps_cp_interleave."""

from types import SimpleNamespace

import pytest
from vllm.config import VllmConfig

import vllm_torchtpu
import vllm_torchtpu.envs as envs

pytestmark = pytest.mark.cpu_test


def _install_fresh_patch(
    monkeypatch, *, glm_admission: bool = True, identity: bool = True
):
    calls = []

    def fake_original(self, kv_cache_config) -> None:
        calls.append(kv_cache_config)
        self.parallel_config.cp_kv_cache_interleave_size = 256

    monkeypatch.setattr(
        VllmConfig, "adjust_dcp_kv_cache_interleave_size", fake_original
    )
    monkeypatch.setattr(
        VllmConfig, "_tpu_raiden_keep_interleave_patch", False, raising=False
    )
    monkeypatch.setattr(envs, "use_raiden_glm_admission", lambda: glm_admission)
    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_DCP_IDENTITY", identity)
    vllm_torchtpu._patch_raiden_identity_keeps_cp_interleave()
    return calls


def _dummy_self(
    transfer, *, dcp: int = 16, cp_interleave: int = 4, dcp_interleave: int = 1
):
    return SimpleNamespace(
        kv_transfer_config=transfer,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=cp_interleave,
            dcp_kv_cache_interleave_size=dcp_interleave,
        ),
    )


def test_standalone_raiden_offloading_keeps_cp_interleave(monkeypatch):
    """Single-leg prefill/decode runs use TPURaidenOffloadingConnector without P/D."""
    calls = _install_fresh_patch(monkeypatch)
    cfg = _dummy_self(
        SimpleNamespace(
            kv_connector="TPURaidenOffloadingConnector",
            kv_connector_extra_config={"cpu_bytes_to_use": 1 << 30},
        ),
        cp_interleave=4,
    )
    VllmConfig.adjust_dcp_kv_cache_interleave_size(cfg, object())
    assert cfg.parallel_config.cp_kv_cache_interleave_size == 4
    assert calls == []


def test_multi_connector_raiden_pd_keeps_cp_interleave(monkeypatch):
    calls = _install_fresh_patch(monkeypatch)
    cfg = _dummy_self(
        SimpleNamespace(
            kv_connector="TPUMultiConnector",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "TPURaidenConnector", "kv_role": "kv_producer"},
                    {
                        "kv_connector": "TPURaidenOffloadingConnector",
                        "kv_role": "kv_both",
                    },
                ]
            },
        ),
        cp_interleave=4,
    )
    VllmConfig.adjust_dcp_kv_cache_interleave_size(cfg, object())
    assert cfg.parallel_config.cp_kv_cache_interleave_size == 4
    assert calls == []


def test_non_raiden_or_disabled_identity_delegates_to_upstream(monkeypatch):
    calls = _install_fresh_patch(monkeypatch)
    nixl_cfg = _dummy_self(
        SimpleNamespace(kv_connector="NixlConnector", kv_connector_extra_config={}),
        cp_interleave=4,
    )
    VllmConfig.adjust_dcp_kv_cache_interleave_size(nixl_cfg, "kv_cfg")
    assert nixl_cfg.parallel_config.cp_kv_cache_interleave_size == 256
    assert calls == ["kv_cfg"]

    monkeypatch.setattr(envs, "TPU_RAIDEN_GLM_DCP_IDENTITY", False)
    offload_cfg = _dummy_self(
        SimpleNamespace(
            kv_connector="TPURaidenOffloadingConnector",
            kv_connector_extra_config={},
        ),
        cp_interleave=4,
    )
    VllmConfig.adjust_dcp_kv_cache_interleave_size(offload_cfg, "kv_cfg_2")
    assert offload_cfg.parallel_config.cp_kv_cache_interleave_size == 256
    assert calls == ["kv_cfg", "kv_cfg_2"]
