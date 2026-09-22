# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

import vllm_torchtpu


class _FakeHybridCoordinator(HybridKVCacheCoordinator):
    def __init__(self) -> None:
        pass


class _FakeKVCacheManager:
    def __init__(self) -> None:
        self.coordinator = _FakeHybridCoordinator()

    def get_computed_blocks(self, request):
        del request
        return ("common-blocks",), 3072, 6144

    def get_computed_blocks_for_connector(self, request):
        del request
        return ("divergent-blocks",), 9216, 0, True


def _scheduler(*, is_kv_producer: bool):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            kv_transfer_config=SimpleNamespace(is_kv_producer=is_kv_producer)
        ),
        has_mamba_layers=True,
        kv_cache_manager=_FakeKVCacheManager(),
    )


def test_producer_connector_uses_common_hit_and_preserves_shared_boundary():
    scheduler = _scheduler(is_kv_producer=True)

    vllm_torchtpu._reconcile_hybrid_producer_prefix_hits(scheduler)

    result = scheduler.kv_cache_manager.get_computed_blocks_for_connector(object())
    assert result == (("common-blocks",), 3072, 6144, False)


def test_consumer_connector_keeps_divergent_hit_lookup():
    scheduler = _scheduler(is_kv_producer=False)

    vllm_torchtpu._reconcile_hybrid_producer_prefix_hits(scheduler)

    result = scheduler.kv_cache_manager.get_computed_blocks_for_connector(object())
    assert result == (("divergent-blocks",), 9216, 0, True)
