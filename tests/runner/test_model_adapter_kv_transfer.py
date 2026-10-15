# SPDX-License-Identifier: Apache-2.0
"""Model-owned caches must register the same live buffers after binding."""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.runner import kv_cache_manager


@pytest.mark.parametrize("transfer_enabled", [False, True])
@pytest.mark.parametrize("runner_hooks", [False, True])
def test_adapter_cache_registration_order_and_identity(
    monkeypatch, transfer_enabled, runner_hooks
):
    events = []
    raw = torch.zeros(2, 4)
    caches = {"attention": raw, "ring": raw, "state": [raw]}
    config = object()
    runner = SimpleNamespace()

    class Adapter:
        def initialize_kv_cache(self, actual_config):
            assert actual_config is config
            runner.kv_caches = [raw]
            events.append("bound")
            return caches

    class Connector:
        def register_kv_caches(self, actual_caches):
            assert actual_caches is caches
            assert runner.kv_caches[0] is actual_caches["ring"]
            events.append("registered")

    class ConnectorWithHooks(Connector):
        def set_host_xfer_buffer_ops(self, copy_op):
            assert copy_op is kv_cache_manager.copy_kv_blocks
            events.append("copy")

        def register_runner(self, actual_runner):
            assert actual_runner is runner
            events.append("runner")

    connector = ConnectorWithHooks() if runner_hooks else Connector()

    def get_connector():
        assert transfer_enabled
        return connector

    monkeypatch.setattr(
        kv_cache_manager, "has_kv_transfer_group", lambda: transfer_enabled
    )
    monkeypatch.setattr(kv_cache_manager, "get_kv_transfer_group", get_connector)
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.runner = runner
    manager.model_adapter = Adapter()
    manager.initialize_kv_cache(config)
    expected = ["bound"]
    if transfer_enabled:
        expected += ["registered"]
        if runner_hooks:
            expected += ["copy", "runner"]
    assert events == expected


@pytest.mark.parametrize("transfer_enabled", [False, True])
def test_missing_live_cache_mapping_is_rejected_only_for_transfer(
    monkeypatch, transfer_enabled
):
    monkeypatch.setattr(
        kv_cache_manager, "has_kv_transfer_group", lambda: transfer_enabled
    )
    manager = object.__new__(kv_cache_manager.KVCacheManager)
    manager.runner = SimpleNamespace()
    manager.model_adapter = SimpleNamespace(initialize_kv_cache=lambda config: None)
    if transfer_enabled:
        with pytest.raises(NotImplementedError, match="live named buffers"):
            manager.initialize_kv_cache(object())
    else:
        manager.initialize_kv_cache(object())
