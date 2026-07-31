# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu import (_patch_vllm_hybrid_producer_prefix_hits,
                           _reconcile_hybrid_producer_prefix_hits)


class _FakeHybridCoordinator(HybridKVCacheCoordinator):

    def __init__(self) -> None:
        self.kv_cache_config = SimpleNamespace(kv_cache_groups=(object(),
                                                                object(),
                                                                object(),
                                                                object()))

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        del block_hashes, max_cache_hit_length
        return ("common-blocks", ), 0, 17

    def find_longest_cache_hit_per_group(self, block_hashes,
                                         max_cache_hit_length):
        del block_hashes, max_cache_hit_length
        return ("divergent-blocks", ), (0, 0, 0, 2304)


def _scheduler(*, is_kv_producer: bool):
    coordinator = _FakeHybridCoordinator()
    return SimpleNamespace(
        vllm_config=SimpleNamespace(kv_transfer_config=SimpleNamespace(
            is_kv_producer=is_kv_producer)),
        has_mamba_layers=True,
        kv_cache_manager=SimpleNamespace(coordinator=coordinator),
    )


def test_only_producer_reconciles_divergent_hybrid_prefix_hits():
    producer = _scheduler(is_kv_producer=True)
    consumer = _scheduler(is_kv_producer=False)

    _reconcile_hybrid_producer_prefix_hits(producer)
    _reconcile_hybrid_producer_prefix_hits(consumer)

    producer_blocks, producer_hits = (
        producer.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
            [], 3089))
    consumer_blocks, consumer_hits = (
        consumer.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
            [], 3089))

    assert producer_blocks == ("common-blocks", )
    assert producer_hits == (0, 0, 0, 0)
    assert consumer_blocks == ("divergent-blocks", )
    assert consumer_hits == (0, 0, 0, 2304)


def test_patch_reconciles_first_scheduler(monkeypatch):

    def original_init(self):
        state = _scheduler(is_kv_producer=True)
        self.vllm_config = state.vllm_config
        self.has_mamba_layers = state.has_mamba_layers
        self.kv_cache_manager = state.kv_cache_manager

    monkeypatch.setattr(Scheduler, "__init__", original_init)
    monkeypatch.delattr(
        Scheduler,
        "_tpu_hybrid_producer_prefix_hit_patch",
        raising=False,
    )

    _patch_vllm_hybrid_producer_prefix_hits()

    producer = Scheduler()
    blocks, hits = (
        producer.kv_cache_manager.coordinator.find_longest_cache_hit_per_group(
            [], 3089))
    assert blocks == ("common-blocks", )
    assert hits == (0, 0, 0, 0)
