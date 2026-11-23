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
"""CPU unit tests for TPUDecodeBenchConnector and its interleave patch."""

from types import SimpleNamespace

import pytest
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorMetadata,
)

import vllm_torchtpu
from vllm_torchtpu.distributed.kv_transfer.decode_bench_connector import (
    CONNECTOR_NAME,
    TPUDecodeBenchConnector,
)

pytestmark = pytest.mark.cpu_test


class _TransferConfig(SimpleNamespace):
    def get_from_extra_config(self, key, default):
        return (self.kv_connector_extra_config or {}).get(key, default)


def _vllm_config(*, is_hybrid: bool = False, extra=None):
    return SimpleNamespace(
        kv_transfer_config=_TransferConfig(
            kv_connector=CONNECTOR_NAME,
            kv_role="kv_both",
            kv_connector_extra_config=extra or {},
        ),
        cache_config=SimpleNamespace(block_size=256),
        model_config=SimpleNamespace(is_hybrid=is_hybrid),
    )


def _connector(role):
    return TPUDecodeBenchConnector(_vllm_config(), role, kv_cache_config=None)


def test_scheduler_claims_all_uncached_prompt_tokens_but_the_last():
    conn = _connector(KVConnectorRole.SCHEDULER)
    req = SimpleNamespace(request_id="r0", num_tokens=126_000)
    # 122_880 tokens hit the local prefix cache (30 DCP16 blocks of 4096).
    assert conn.get_num_new_matched_tokens(req, 122_880) == (
        126_000 - 122_880 - 1,
        False,
    )

    blocks = SimpleNamespace(get_block_ids=lambda: ([7, 8],))
    conn.update_state_after_alloc(req, blocks, 126_000 - 122_880 - 1)
    # A request is claimed once; a re-schedule (e.g. after preemption) computes.
    assert conn.get_num_new_matched_tokens(req, 122_880) == (0, False)
    conn.request_finished(req, [7, 8])
    assert conn.get_num_new_matched_tokens(req, 122_880) == (3119, False)


def test_scheduler_claims_nothing_when_only_the_last_token_is_uncached():
    conn = _connector(KVConnectorRole.SCHEDULER)
    req = SimpleNamespace(request_id="r1", num_tokens=4097)
    assert conn.get_num_new_matched_tokens(req, 4096) == (0, False)


def test_worker_never_touches_kv():
    conn = _connector(KVConnectorRole.WORKER)

    class _Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"KV cache accessed: {name}")

        def __setitem__(self, key, value):
            raise AssertionError("KV cache written")

    conn.register_kv_caches({"model.layers.0.self_attn": _Untouchable()})
    assert conn.connector_worker.kv_caches is None
    conn.bind_connector_metadata(
        DecodeBenchConnectorMetadata(reqs_to_fill={"r0": (([1, 2, 3],), 700)})
    )
    conn.start_load_kv(None)
    conn.start_load_kv(None, wait_for_completion=True, report_completion=False)


def test_marked_mode_claims_only_marked_requests():
    conn = TPUDecodeBenchConnector(
        _vllm_config(extra={"claim_mode": "marked"}), KVConnectorRole.SCHEDULER, None
    )
    plain = SimpleNamespace(request_id="p", num_tokens=10_000, kv_transfer_params=None)
    assert conn.get_num_new_matched_tokens(plain, 4096) == (0, False)
    marked = SimpleNamespace(
        request_id="m",
        num_tokens=10_000,
        kv_transfer_params={"decode_bench_claim": True},
    )
    assert conn.get_num_new_matched_tokens(marked, 4096) == (10_000 - 4096 - 1, False)


def test_unknown_claim_mode_is_rejected():
    with pytest.raises(ValueError, match="claim_mode"):
        TPUDecodeBenchConnector(
            _vllm_config(extra={"claim_mode": "some"}), KVConnectorRole.SCHEDULER, None
        )


def test_hybrid_models_are_rejected():
    with pytest.raises(ValueError, match="hybrid"):
        TPUDecodeBenchConnector(
            _vllm_config(is_hybrid=True), KVConnectorRole.SCHEDULER, None
        )


def _install_fresh_patch(monkeypatch):
    calls = []

    def fake_original(self, kv_cache_config) -> None:
        calls.append(kv_cache_config)
        self.parallel_config.cp_kv_cache_interleave_size = 256

    monkeypatch.setattr(
        VllmConfig, "adjust_dcp_kv_cache_interleave_size", fake_original
    )
    monkeypatch.setattr(
        VllmConfig, "_tpu_decode_bench_keep_interleave_patch", False, raising=False
    )
    vllm_torchtpu._patch_decode_bench_keeps_cp_interleave()
    vllm_torchtpu._patch_decode_bench_keeps_cp_interleave()  # idempotent
    return calls


def _dummy_self(connector: str, *, dcp: int = 16, dcp_interleave: int = 1):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector=connector),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=4,
            dcp_kv_cache_interleave_size=dcp_interleave,
        ),
    )


def test_decode_bench_keeps_cp_interleave(monkeypatch):
    calls = _install_fresh_patch(monkeypatch)
    cfg = _dummy_self(CONNECTOR_NAME)
    VllmConfig.adjust_dcp_kv_cache_interleave_size(cfg, object())
    assert cfg.parallel_config.cp_kv_cache_interleave_size == 4
    assert calls == []


def test_decode_bench_honors_deprecated_dcp_interleave(monkeypatch):
    _install_fresh_patch(monkeypatch)
    cfg = _dummy_self(CONNECTOR_NAME, dcp_interleave=64)
    VllmConfig.adjust_dcp_kv_cache_interleave_size(cfg, object())
    assert cfg.parallel_config.cp_kv_cache_interleave_size == 64


def test_other_connectors_delegate_to_upstream(monkeypatch):
    calls = _install_fresh_patch(monkeypatch)
    cfg = _dummy_self("DecodeBenchConnector")
    VllmConfig.adjust_dcp_kv_cache_interleave_size(cfg, "kv_cfg")
    assert cfg.parallel_config.cp_kv_cache_interleave_size == 256
    assert calls == ["kv_cfg"]
