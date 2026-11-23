# SPDX-License-Identifier: Apache-2.0
"""TIER-X2 (VLLM_TPU_OFFLOAD_TIER_X2): a second host tier behind the tier classes, on CPU.

Each class store joins Raiden's global registry; Raiden's evict sweep demotes cold host
blocks to the store nodes of the class's pool group, and a host-tier miss reads them
back with read_remote. Raiden's transfer spec wants one block geometry per store, so
the classes are cut into shards of equal per-worker region count.

- The plan: the served classes cut into the reviewed shard table, every shard uniform,
  node ids dense, capacities and bytes those of the class.
- The spec: the refusals (no classes, no registry / address / pool group, no evict
  sweep), the shard stores on the registry with their own pool group and key
  namespace, the worker registrations under dense node ids, and the flag off
  unchanged.
- The manager over byte-moving fakes (a store node a shard, one shared registry): a
  block demoted to the node and read back lands the bytes it was saved with, alone or
  next to a local load; a key is held only if every shard holds it; a key the host pool
  holds costs no registry call; a registry miss is remembered for the TTL; a slice that
  went stale fails the load closed; a peer read takes and gives back no pin; exclusive
  restore evicts only the local part; a key a shard's node holds is never saved
  again by that shard, from its own store probe or because a sibling shard needs it,
  so a peer read lands the bytes last saved (smoke j-a44349b7 P1); the flag off keeps
  the local-only probe and saves what it is handed.
"""

import os
from types import SimpleNamespace as NS
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key

from tests.offload.test_raiden_store_offload import (
    FAKE_STORE_TYPES,
    FakeBlockStatus,
    FakeKVCacheStore,
    FakeRaidenBlockId,
)
from tests.offload.test_tier_classes import (
    GDN_A,
    GDN_B,
    KEYS,
    MAIN,
    PLE,
    SERVED_BUDGET,
    _register,
    _served,
    _spec,
    _spec_config,
    _vllm_config,
)

REGISTRY = {
    "global_registry_address": "registry:31600",
    "store_server_ip": "10.0.0.1",
    "kv_pool_group": "q38x2",
}
SWEEP_ENV = {
    "RAIDEN_ENABLE_STORE_MONITOR": "true",
    "RAIDEN_ENABLE_EVICT_SWEEP": "true",
    "PYTHONHASHSEED": "0",
}


def _ctx(req_id="req"):
    return ReqContext(req_id=req_id)


def _key(n, group=MAIN):
    return make_offload_key(b"x2hash_%04d" % n, group)


def _x2_env(monkeypatch, sweep=True):
    monkeypatch.setenv("VLLM_TPU_OFFLOAD_TIER_X2", "1")
    for name, value in SWEEP_ENV.items():
        if sweep or name == "PYTHONHASHSEED":
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)


# -- byte-moving fakes ----------------------------------------------------------------


class FakeNode:
    """A store node on another host: its own host bytes, published in the registry."""

    def __init__(self, name):
        self.name = name
        self.host: dict[bytes, bytes] = {}

    def __repr__(self):
        return f"FakeNode({self.name})"


class ByteStore(FakeKVCacheStore):
    """FakeKVCacheStore that moves bytes: device rows (``hbm``), host entries
    (``host``), and a peer read from the node a REMOTE slice names."""

    def __init__(self, name, registry, node):
        super().__init__()
        self.name = name
        self.hbm: dict[int, bytes] = {}
        self.host: dict[bytes, bytes] = {}
        self.node = node
        # The registry is shared by every store, as Raiden's global registry is.
        self.global_entries = registry
        self._remote_src: dict[bytes, object] = {}
        self.evicted: list[bytes] = []
        # The bytes each sub-hash's last landed save copied to the host.
        self.saved: dict[bytes, bytes] = {}

    def evict(self, block_hashes):
        n = 0
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None or entry.pin_count or entry.status not in (
                FakeBlockStatus.HOST,
                FakeBlockStatus.HOST_AND_HBM,
            ):
                continue
            del self.entries[h]
            self.host.pop(h, None)
            self.evicted.append(h)
            n += 1
        # Raiden's Evict fires the eviction callback before it returns.
        if n:
            self.simulate_evict(self.evicted[-n:])
        return n

    def read_remote(self, block_hashes, slices, device_block_ids):
        ok = super().read_remote(block_hashes, slices, device_block_ids)
        if ok:
            for h, sl in zip(block_hashes, slices):
                self._remote_src[h] = sl.raiden_id
        return ok

    # -- what Raiden's transfers would do ------------------------------------------

    def finish(self):
        """Lands every transfer in flight: saves copy device rows to the host, loads
        copy host bytes to device rows, peer reads copy the named node's bytes (and
        fail when the node no longer holds the block)."""
        for h in list(self._saving):
            self.host[h] = self.hbm[self.entries[h].device_block_id]
            self.saved[h] = self.host[h]
            self.complete_save([h])
        for h in list(self._loading):
            dst = self._load_dst[h]
            self.hbm[dst] = self.host[h]
            self.complete_load([h])
        for h in list(self._remote_reading):
            node = self._remote_src.pop(h)
            dst = self._remote_device_ids[h]
            if h in node.host:
                self.hbm[dst] = node.host[h]
                self.complete_remote_read([h])
            else:
                self.complete_remote_read([h], success=False)

    def load(self, block_hashes, device_block_ids, slices=None):
        ok = super().load(block_hashes, device_block_ids, slices)
        if ok:
            self.__dict__.setdefault("_load_dst", {}).update(
                zip(block_hashes, device_block_ids)
            )
        return ok

    def demote(self, block_hashes):
        """Raiden's evict sweep: an unpinned host entry is copied to this store's node,
        published there, then evicted here (the eviction callback fires). A node that
        already holds the hash keeps its own copy and the local one is freed all the
        same: WriteRemote answers WRITE_ALL_EXIST without pulling a byte
        (kv_cache_store_service.cc) and SweepOnce evicts the `existing` hashes
        (kv_cache_store.cc WaitForBatchWriteResult)."""
        moved = []
        for h in block_hashes:
            entry = self.entries.get(h)
            if entry is None or entry.pin_count or entry.status not in (
                FakeBlockStatus.HOST,
                FakeBlockStatus.HOST_AND_HBM,
            ):
                continue
            mine = self.host.pop(h)
            self.node.host.setdefault(h, mine)
            self.global_entries[h] = FakeRaidenBlockId(
                raiden_id=self.node, status=FakeBlockStatus.REMOTE
            )
            del self.entries[h]
            moved.append(h)
        if moved:
            self.simulate_evict(moved)
        return moved

    def node_evicts(self, block_hashes):
        """The node's own LRU drops blocks: gone from it and from the registry."""
        for h in block_hashes:
            self.node.host.pop(h, None)
            self.global_entries.pop(h, None)


def _shard_tier(ttl_s=0.0, excl=False):
    """A sharded tier over byte stores: main in two shards (1 and 2 regions a
    worker), ple whole, each with its own node; one registry."""
    from vllm_torchtpu.offload.raiden_store import (
        RaidenOffloadingManager,
        x2_shard_namespace,
    )
    from vllm_torchtpu.offload.tier_classes import (
        TierClass,
        TieredOffloadingManager,
        TierPlan,
    )

    classes = (
        TierClass("main.r1", 0, "main", (MAIN,), ((0,), ()), 100, shard_of="main"),
        TierClass("main.r2", 1, "main", (MAIN,), ((), (1, 2)), 100, shard_of="main"),
        TierClass("ple", 2, "ple", (PLE,), ((0,), ()), 100, shard_of="ple"),
    )
    plan = TierPlan(classes, budget=0, ratio=2.0, token_blocks=16, checkpoints=8)
    registry: dict = {}
    stores = {
        c.name: ByteStore(c.name, registry, FakeNode(f"node-{c.name}")) for c in classes
    }
    base_ns = b"N" * 16
    with patch.dict(os.environ, {"PYTHONHASHSEED": "0"}):
        managers = [
            RaidenOffloadingManager(
                kernel_physical_blocks_capacity=plan.capacity(c),
                offload_logical_blocks_capacity=plan.capacity(c),
                device_block_size_factor=1,
                world_size=c.num_workers,
                raiden_controller_port=0,
                raiden_job_name=f"test-{c.name}",
                global_registry_address=REGISTRY["global_registry_address"],
                store_server_ip=REGISTRY["store_server_ip"],
                kv_pool_group=f"q38x2-{c.name}",
                key_namespace=x2_shard_namespace(base_ns, c.name),
                track_held=True,
                tier_x2=True,
                x2_miss_ttl_s=ttl_s,
                store=stores[c.name],
                store_types=FAKE_STORE_TYPES,
            )
            for c in classes
        ]
    tier = TieredOffloadingManager(plan, managers)
    if excl:
        tier.set_exclusive_restore(True)
    return tier, stores, registry


def _payload(store, key):
    return b"%s|%s" % (store.encode(), bytes(key))


def _finish(tier, stores):
    for store in stores.values():
        store.finish()
    return tier.poll_finished_jobs()


def _save(tier, stores, keys, blocks, job_id=1):
    """Write each store's own rows of each key into its device blocks, then store
    (prepare first, as the connector does: a store job follows prepare_store or
    prepare_victim_store)."""
    for name, store in stores.items():
        for key, b in zip(keys, blocks):
            store.hbm[b] = _payload(name, key)
    out = tier.prepare_store(list(keys), _ctx())
    assert list(out.keys_to_store) == list(keys)
    tier.submit_store_job(job_id, list(keys), list(blocks))
    (done,) = _finish(tier, stores)
    assert done.job_id == job_id and done.success
    # The device rows are reused: any later read of them must come from a transfer.
    for store in stores.values():
        store.hbm = {b: b"garbage" for b in store.hbm}


def _sub_of(tier, store_name, key):
    (manager,) = [
        m for c, m in zip(tier.classes, tier.managers) if c.name == store_name
    ]
    return manager._sub_hashes(key)[0]


def _demote(tier, stores, keys, only=None):
    for name, store in stores.items():
        if only is not None and name not in only:
            continue
        subs = [
            _sub_of(tier, name, k)
            for k in keys
            if _sub_of(tier, name, k) in store.entries
        ]
        store.demote(subs)


def _load(tier, stores, keys, blocks, job_id=2, req="req"):
    for key in keys:
        assert tier.lookup(key, _ctx(req)) == LookupResult.HIT, key
    spec = tier.prepare_load(keys, _ctx(req))
    assert spec.pinned
    tier.submit_load_job(job_id, list(keys), list(blocks), req)
    return _finish(tier, stores)


# -- the plan ---------------------------------------------------------------------------


def test_served_classes_cut_into_the_reviewed_shards(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import tier_group_kind
    from vllm_torchtpu.offload.tier_classes import (
        SERVED_SHARDS,
        build_tier_plan,
        check_served_plan,
        check_served_shards,
        shard_plan,
    )

    _, _, stages, kvc = _served(monkeypatch)
    plan = build_tier_plan(
        stages, kvc.prefix_cacheable_group_ids, tier_group_kind, SERVED_BUDGET, 4.7
    )
    check_served_plan(plan)
    shards = shard_plan(plan)
    check_served_shards(shards)
    assert [c.name for c in shards.classes] == list(SERVED_SHARDS)
    assert [c.index for c in shards.classes] == list(range(len(SERVED_SHARDS)))
    by_class = {c.name: c for c in plan.classes}
    for shard in shards.classes:
        parent = by_class[shard.shard_of]
        counts = {len(shard.worker_regions[w]) for w in shard.workers}
        assert len(counts) == 1, (shard.name, counts)  # one geometry a shard
        assert shard.groups == parent.groups and shard.kind == parent.kind
        assert shard.row_bytes == parent.row_bytes
        assert shards.capacity(shard) == plan.capacity(parent)
        for w, regions in enumerate(shard.worker_regions):
            assert regions in ((), parent.worker_regions[w])
        assert [shard.node_id(w) for w in shard.workers] == list(
            range(shard.num_workers)
        )
    for parent in plan.classes:
        mine = [s for s in shards.classes if s.shard_of == parent.name]
        # The shards partition the class's workers and add up to its entry.
        assert sorted(w for s in mine for w in s.workers) == list(parent.workers)
        assert sum(s.bytes_per_entry for s in mine) == parent.bytes_per_entry
    assert (shards.token_blocks, shards.checkpoints) == (
        plan.token_blocks,
        plan.checkpoints,
    )
    # The served stages: main / keys split by odd and even stage, gdn-c by its table.
    names = {c.name: c.workers for c in shards.classes}
    assert names["main.r1"] == (0, 2, 4, 6) and names["main.r2"] == (1, 3, 5, 7)
    assert names["gdn-ab.r1"] == (1, 3, 5, 7) and names["gdn-ab.r2"] == (0, 2, 4, 6)
    assert names["gdn-c.r1"] == (0, 2, 4, 7) and names["gdn-c.r2"] == (1, 3, 5, 6)
    assert names["ple"] == (0,)


# -- the spec ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "classes, extra, sweep, error",
    [
        (False, REGISTRY, True, "needs VLLM_TPU_OFFLOAD_TIER_CLASSES=1"),
        (True, {}, True, "global_registry_address, store_server_ip, kv_pool_group"),
        (
            True,
            {k: v for k, v in REGISTRY.items() if k != "kv_pool_group"},
            True,
            "needs kv_pool_group",
        ),
        (True, REGISTRY, False, "RAIDEN_ENABLE_STORE_MONITOR and RAIDEN_ENABLE_EVICT"),
    ],
)
def test_x2_refusals(monkeypatch, classes, extra, sweep, error):
    cfg, _, stages, kvc = _served(monkeypatch, classes=classes)
    _x2_env(monkeypatch, sweep=sweep)
    with pytest.raises(ValueError, match=error):
        _spec(monkeypatch, cfg, stages, kvc, **extra)


def test_x2_expands_the_pods_address_from_the_environment(monkeypatch):
    cfg, _, stages, kvc = _served(monkeypatch)
    _x2_env(monkeypatch)
    monkeypatch.setenv("TPU_RAIDEN_ADVERTISE_HOST", "10.1.2.3")
    extra = {
        "global_registry_address": "127.0.0.1:31600",
        "store_server_ip": "${TPU_RAIDEN_ADVERTISE_HOST}",
        "kv_pool_group": "q38x2",
    }
    spec = _spec(monkeypatch, cfg, stages, kvc, **extra)
    assert spec.store_server_ip == "10.1.2.3"
    assert spec.global_registry_address == "127.0.0.1:31600"
    assert spec.tier_x2_pool_group(spec.tier_plan.classes[0]) == "q38x2-main.r1"
    monkeypatch.delenv("TPU_RAIDEN_ADVERTISE_HOST")
    with pytest.raises(ValueError, match="names an unset environment variable"):
        _spec(monkeypatch, cfg, stages, kvc, **extra)


def test_flag_off_keeps_the_extra_config_literal(monkeypatch):
    cfg, _, stages, kvc = _served(monkeypatch, classes=False)
    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_X2", raising=False)
    monkeypatch.setenv("TPU_RAIDEN_ADVERTISE_HOST", "10.1.2.3")
    spec = _spec(monkeypatch, cfg, stages, kvc, store_server_ip="${TPU_RAIDEN_ADVERTISE_HOST}")
    assert spec.store_server_ip == "${TPU_RAIDEN_ADVERTISE_HOST}"


def test_flag_off_still_refuses_the_registry_with_the_classes(monkeypatch):
    cfg, _, stages, kvc = _served(monkeypatch)
    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_X2", raising=False)
    for name, value in SWEEP_ENV.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="global registry"):
        _spec(monkeypatch, cfg, stages, kvc, **REGISTRY)


def _manager_kwargs(monkeypatch, spec):
    from vllm_torchtpu.offload import raiden_store

    built = MagicMock()
    monkeypatch.setattr(raiden_store, "RaidenOffloadingManager", built)
    monkeypatch.setattr(raiden_store, "x2_wait_registry", lambda address: 0.0)
    monkeypatch.setattr(
        "vllm_torchtpu.offload.tier_classes.TieredOffloadingManager",
        lambda plan, managers, check=False, fp4_slots=None: NS(plan=plan, managers=managers),
    )
    spec.build_tier_manager()
    return [call.kwargs for call in built.call_args_list]


def test_x2_shard_stores_join_the_registry(monkeypatch):
    from vllm_torchtpu.offload.raiden_store import x2_shard_namespace
    from vllm_torchtpu.offload.tier_classes import SERVED_SHARDS

    cfg, _, stages, kvc = _served(monkeypatch)
    _x2_env(monkeypatch)
    spec = _spec(monkeypatch, cfg, stages, kvc, **REGISTRY)
    plan = spec.tier_plan
    assert [c.name for c in plan.classes] == list(SERVED_SHARDS)
    assert [spec.tier_controller_port(c) for c in plan.classes] == list(
        range(31515, 31515 + 9)
    )
    calls = _manager_kwargs(monkeypatch, spec)
    assert len(calls) == 9
    for c, kw in zip(plan.classes, calls):
        assert kw["global_registry_address"] == REGISTRY["global_registry_address"]
        assert kw["store_server_ip"] == REGISTRY["store_server_ip"]
        assert kw["kv_pool_group"] == f"q38x2-{c.name}"
        assert kw["tier_x2"] is True and kw["x2_miss_ttl_s"] == 1.0
        assert kw["world_size"] == c.num_workers
        assert kw["key_namespace"] == x2_shard_namespace(spec.key_namespace, c.name)
        assert kw["raiden_controller_port"] == 31515 + c.index
    assert len({kw["key_namespace"] for kw in calls}) == 9
    assert spec.key_namespace not in {kw["key_namespace"] for kw in calls}


def test_flag_off_builds_the_class_stores_as_before(monkeypatch):
    cfg, _, stages, kvc = _served(monkeypatch)
    monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_X2", raising=False)
    spec = _spec(monkeypatch, cfg, stages, kvc)
    assert [c.name for c in spec.tier_plan.classes] == [
        "main",
        "keys",
        "gdn-ab",
        "gdn-c",
        "ple",
    ]
    assert all(c.shard_of is None for c in spec.tier_plan.classes)
    calls = _manager_kwargs(monkeypatch, spec)
    for kw in calls:
        assert set(kw) == {
            "kernel_physical_blocks_capacity",
            "offload_logical_blocks_capacity",
            "device_block_size_factor",
            "world_size",
            "raiden_controller_port",
            "raiden_job_name",
            "store_server_ip",
            "key_namespace",
            "device_row_stride",
            "track_held",
            "store",
            "store_types",
        }
        assert kw["key_namespace"] == spec.key_namespace


@pytest.mark.parametrize("x2", [False, True])
def test_each_rank_registers_with_its_shards_under_dense_node_ids(monkeypatch, x2):
    from vllm.distributed import parallel_state

    from tests.runner.test_qwen4_exp_host_tier import _stage_caches
    from tests.runner.test_qwen4_exp_offload import _fake_tpu_sync
    from vllm_torchtpu.models.vllm.qwen4_exp.offload import pool_regions
    from vllm_torchtpu.offload import raiden_store
    from vllm_torchtpu.offload.raiden_store import RaidenStoreOffloadingWorker

    cfg, _, stages, kvc = _served(monkeypatch)
    extra = {}
    if x2:
        _x2_env(monkeypatch)
        extra = REGISTRY
    else:
        monkeypatch.delenv("VLLM_TPU_OFFLOAD_TIER_X2", raising=False)
    _spec(monkeypatch, cfg, stages, kvc, **extra)  # records the per-worker configs
    shape = (stages[0].num_blocks, 3200, 1, 4, 256)
    node_ids: dict[str, dict[int, int]] = {}
    for rank, stage in enumerate(stages):
        config = _spec_config(kvc, stages)
        config.extra_config.update(extra)
        spec = raiden_store.TPURaidenStoreOffloadingSpec(
            config, _vllm_config(cfg), stage
        )
        plan = spec.tier_plan
        regions = [
            torch.zeros(shape, dtype=torch.float8_e4m3fn)
            for _ in range(pool_regions(stage))
        ]
        caches, _ = _stage_caches(stage, regions)
        (canonical,) = _register(stage, caches, spec.vllm_config)
        registered = []
        modules, sys = _fake_tpu_sync(registered)
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                RaidenStoreOffloadingWorker, "_controller_reachable", return_value=True
            ),
            patch.object(
                parallel_state, "get_world_group", return_value=NS(rank_in_group=rank)
            ),
        ):
            worker = spec.get_worker(canonical)
            for class_worker in worker.workers.values():
                class_worker._register_thread.join(timeout=10)
        assert set(worker.workers) == {
            c.name for c in plan.classes if c.worker_regions[rank]
        }
        by_port = {
            kw["raiden_controller_address"]: (tensors, kw) for tensors, kw in registered
        }
        host = REGISTRY["store_server_ip"] if x2 else "127.0.0.1"
        for c in plan.classes:
            address = f"{host}:{31515 + c.index}"
            if not c.worker_regions[rank]:
                assert address not in by_port
                continue
            tensors, kw = by_port[address]
            assert len(tensors) == len(c.worker_regions[rank])
            assert kw["worker_id"] == f"shard_{rank}"
            node_ids.setdefault(c.name, {})[rank] = kw["node_id"]
        worker.shutdown()
    for name, ids in node_ids.items():
        if x2:
            assert sorted(ids.values()) == list(range(len(ids))), (name, ids)
            assert list(ids.values()) == sorted(ids.values())
        else:
            assert all(node == rank for rank, node in ids.items()), (name, ids)
    if x2:
        assert node_ids["main.r2"] == {1: 0, 3: 1, 5: 2, 7: 3}
        assert node_ids["gdn-c.r1"] == {0: 0, 2: 1, 4: 2, 7: 3}


# -- the manager over byte-moving fakes -------------------------------------------------


def test_a_demoted_block_reads_back_the_bytes_it_was_saved_with():
    tier, stores, registry = _shard_tier()
    keys = [_key(1), _key(2), _key(3), _key(4, PLE)]
    _save(tier, stores, keys, [10, 11, 12, 13])
    _demote(tier, stores, keys)
    for store in stores.values():
        assert not store.entries  # every host entry left for the node
    (done,) = _load(tier, stores, keys, [20, 21, 22, 23])
    assert done.success and done.failed_device_block_ids == []
    for name, store in stores.items():
        for key, dst in zip(keys, [20, 21, 22, 23]):
            group = MAIN if name.startswith("main") else PLE
            if int.from_bytes(bytes(key)[-4:], "big") != group:
                continue
            assert store.hbm[dst] == _payload(name, key), (name, key, dst)
        # Every byte came from the store's own node.
        assert store.calls_named("read_remote")
        for call in store.calls_named("read_remote"):
            assert {sl.raiden_id for sl in call.args[1]} == {store.node}
    stats = tier.x2_stats()
    assert stats["remote_hit_keys"] == 3 * 2 + 1
    assert stats["remote_read_blocks_done"] == 3 * 2 + 1
    assert stats["remote_read_blocks_failed"] == 0


def test_a_mixed_load_reads_one_shard_locally_and_the_other_from_its_node():
    tier, stores, _ = _shard_tier()
    keys = [_key(1), _key(2)]
    _save(tier, stores, keys, [10, 11])
    _demote(tier, stores, keys, only={"main.r2"})
    (done,) = _load(tier, stores, keys, [30, 31])
    assert done.success
    for name in ("main.r1", "main.r2"):
        for key, dst in zip(keys, [30, 31]):
            assert stores[name].hbm[dst] == _payload(name, key)
    assert stores["main.r1"].calls_named("load") and not stores[
        "main.r1"
    ].calls_named("read_remote")
    assert stores["main.r2"].calls_named("read_remote") and not stores[
        "main.r2"
    ].calls_named("load")


def test_a_key_is_held_only_where_every_shard_holds_it():
    tier, stores, _ = _shard_tier()
    key = _key(7)
    _save(tier, stores, [key], [10])
    # One shard's copy is gone everywhere: the class does not hold the key.
    stores["main.r2"].demote([_sub_of(tier, "main.r2", key)])
    stores["main.r2"].node_evicts([_sub_of(tier, "main.r2", key)])
    assert tier.lookup(key, _ctx()) == LookupResult.MISS
    # One shard's save in flight: the class answers pending.
    other = _key(8)
    _save(tier, stores, [other], [12])
    tier.managers[1]._inflight_store_keys.add(other)
    assert tier.lookup(other, _ctx()) == LookupResult.HIT_PENDING
    tier.managers[1]._inflight_store_keys.discard(other)
    assert tier.lookup(other, _ctx()) == LookupResult.HIT


def test_a_key_the_host_pool_holds_costs_no_registry_call():
    tier, stores, _ = _shard_tier()
    keys = [_key(1), _key(2)]
    _save(tier, stores, keys, [10, 11])
    for key in keys:
        assert tier.lookup(key, _ctx()) == LookupResult.HIT
    stats = tier.x2_stats()
    assert stats["registry_lookups"] == 0 and stats["remote_hit_keys"] == 0


def test_a_registry_miss_is_remembered_for_the_ttl(monkeypatch):
    """A key this shard evicted but nobody holds: the registry is asked once, then the miss is remembered for the
    TTL; a new eviction (it may be on the node now) voids the remembered miss at once."""
    from vllm_torchtpu.offload import raiden_store

    now = [100.0]
    monkeypatch.setattr(raiden_store.time, "monotonic", lambda: now[0])
    tier, stores, _ = _shard_tier(ttl_s=1.0)
    key = _key(5)
    _save(tier, stores, [key], [10])
    _demote(tier, stores, [key])
    for name in ("main.r1", "main.r2"):
        stores[name].node_evicts([_sub_of(tier, name, key)])
    assert tier.lookup(key, _ctx()) == LookupResult.MISS
    assert tier.lookup(key, _ctx()) == LookupResult.MISS
    stats = tier.x2_stats()
    # The first shard missed both tiers once; the second ask was answered from memory.
    assert stats["registry_lookups"] == 1 and stats["miss_cached"] == 1
    now[0] += 1.1
    assert tier.lookup(key, _ctx()) == LookupResult.MISS
    assert tier.x2_stats()["registry_lookups"] == 2
    # The key reaches the second tier again: its eviction voids the remembered miss.
    assert tier.lookup(key, _ctx()) == LookupResult.MISS  # remembered again (miss_cached)
    _save(tier, stores, [key], [11], job_id=7)
    _demote(tier, stores, [key])
    assert tier.lookup(key, _ctx()) == LookupResult.HIT
    # A TTL of 0 asks every time.
    tier0, stores0, _ = _shard_tier(ttl_s=0.0)
    _save(tier0, stores0, [key], [10])
    _demote(tier0, stores0, [key])
    stores0["main.r1"].node_evicts([_sub_of(tier0, "main.r1", key)])
    assert tier0.lookup(key, _ctx()) == LookupResult.MISS
    assert tier0.lookup(key, _ctx()) == LookupResult.MISS
    assert tier0.x2_stats()["registry_lookups"] == 2
    assert tier0.x2_stats()["miss_cached"] == 0


def test_a_lookup_of_a_key_this_shard_never_evicted_costs_no_registry_call():
    """A key this shard never evicted cannot be on its node: a host-pool miss is a miss, no registry call."""
    tier, stores, _ = _shard_tier()
    for i in range(5):
        assert tier.lookup(_key(40 + i), _ctx()) == LookupResult.MISS
    stats = tier.x2_stats()
    assert stats["registry_lookups"] == 0 and stats["lookup_unrecorded"] == 5
    assert not [c for st in stores.values() for c in st.calls_named("lookup") if c.args[1]]


def _victim_scenario(tier, stores, n_old=10, n_demoted=6, n_lost=2, n_new=6):
    """n_old keys saved; n_demoted of them demoted to the nodes, n_lost of those then dropped by the nodes; all
    recomputed, plus n_new never-seen keys, and offered as victims. Returns (old, demoted, lost, new, chosen, the
    x2_stats taken just before the victim call)."""
    old = [_key(100 + i) for i in range(n_old)]
    new = [_key(300 + i) for i in range(n_new)]
    _save(tier, stores, old, list(range(1000, 1000 + n_old)))
    demoted, lost = old[:n_demoted], old[:n_lost]
    _demote(tier, stores, demoted)
    for name in ("main.r1", "main.r2"):
        stores[name].node_evicts([_sub_of(tier, name, k) for k in lost])
    for st in stores.values():
        st.calls.clear()
    tier.set_victim_mode(True)
    before = tier.x2_stats()
    chosen = tier.prepare_victim_store(old + new)
    return old, demoted, lost, new, chosen, before


def test_the_store_probe_asks_the_registry_only_for_keys_this_shard_evicted_in_one_batch():
    """Of 16 victims per shard: 6 never seen (no registry call), 4 still in the host pool (local), 6 this shard
    demoted (one batched registry lookup: 4 on the node, 2 dropped by it, so 1 + 2 calls). The chosen keys are the new
    and the dropped ones."""
    tier, stores, _ = _shard_tier()
    old, demoted, lost, new, chosen, before = _victim_scenario(tier, stores)
    assert set(chosen) == set(new) | set(lost)
    after = tier.x2_stats()
    stats = {k: after[k] - before[k] for k in after}
    assert stats["store_probe_unrecorded"] == 2 * len(new)
    assert stats["store_probe_lookups"] == 2 * len(demoted)
    assert stats["store_remote_skips"] == 2 * (len(demoted) - len(lost))
    assert stats["store_probe_rpcs"] == 2 * (1 + len(lost))
    global_calls = [c for st in stores.values() for c in st.calls_named("lookup") if c.args[1]]
    assert len(global_calls) == stats["store_probe_rpcs"]
    # A key the registry found absent leaves the record (only a new eviction brings it back); a resident one stays.
    for manager in tier.managers[:2]:
        assert not any(manager._x2_maybe_remote(k) for k in lost)
        assert all(manager._x2_maybe_remote(k) for k in demoted[len(lost):])


def test_a_remembered_miss_spares_the_store_probe(monkeypatch):
    """A recorded key whose lookup just missed both tiers is a store without another registry call."""
    from vllm_torchtpu.offload import raiden_store

    now = [100.0]
    monkeypatch.setattr(raiden_store.time, "monotonic", lambda: now[0])
    tier, stores, _ = _shard_tier(ttl_s=5.0)
    key = _key(9)
    _save(tier, stores, [key], [10])
    _demote(tier, stores, [key])
    for name in ("main.r1", "main.r2"):
        stores[name].node_evicts([_sub_of(tier, name, key)])
    assert tier.lookup(key, _ctx()) == LookupResult.MISS  # main.r1 asks and remembers
    assert tier.managers[1].lookup(key, _ctx()) == LookupResult.MISS  # main.r2 too
    before = tier.x2_stats()
    tier.set_victim_mode(True)
    assert tier.prepare_victim_store([key]) == [key]
    after = tier.x2_stats()
    assert after["store_probe_neg_cached"] - before["store_probe_neg_cached"] == 2
    assert after["store_probe_rpcs"] == before["store_probe_rpcs"]


def test_an_exclusive_restore_discard_is_not_recorded():
    """An exclusive-restore discard drops a host copy whose bytes are in HBM: it never reaches the node, so it costs
    no registry call later."""
    tier, stores, _ = _shard_tier(excl=True)
    keys = [_key(1), _key(2)]
    _save(tier, stores, keys, [10, 11])
    (done,) = _load(tier, stores, keys, [30, 31])
    assert done.success
    r1 = tier.managers[0]
    assert set(stores["main.r1"].evicted) == {_sub_of(tier, "main.r1", k) for k in keys}
    assert not any(r1._x2_maybe_remote(k) for k in keys)


def test_a_slice_that_went_stale_fails_the_load_closed():
    tier, stores, _ = _shard_tier()
    keys = [_key(1), _key(2)]
    _save(tier, stores, keys, [10, 11])
    _demote(tier, stores, keys)
    for key in keys:
        assert tier.lookup(key, _ctx()) == LookupResult.HIT
    spec = tier.prepare_load(keys, _ctx())
    # The node drops one block between the lookup and the read.
    stores["main.r2"].node_evicts([_sub_of(tier, "main.r2", keys[1])])
    tier.submit_load_job(5, keys, [40, 41], "req")
    (done,) = _finish(tier, stores)
    assert not done.success
    assert sorted(done.failed_device_block_ids) == [40, 41]
    # No byte of another block landed where the stale one should have.
    assert 41 not in stores["main.r2"].hbm
    assert tier.x2_stats()["remote_read_blocks_failed"] == 1
    assert spec.pinned


def test_a_peer_read_takes_and_gives_back_no_pin():
    tier, stores, _ = _shard_tier()
    keys = [_key(1), _key(2), _key(3, PLE)]
    _save(tier, stores, keys, [10, 11, 12])
    _demote(tier, stores, keys)
    for store in stores.values():
        store.calls.clear()
    (done,) = _load(tier, stores, keys, [20, 21, 22])
    assert done.success
    for name, store in stores.items():
        remote = {h for call in store.calls_named("read_remote") for h in call.args[0]}
        released = {h for call in store.calls_named("release") for h in call.args[0]}
        assert remote and not (remote & released), name
        assert store.pinned_hashes() == {}


def test_exclusive_restore_evicts_only_the_local_part():
    tier, stores, _ = _shard_tier(excl=True)
    keys = [_key(1), _key(2)]
    _save(tier, stores, keys, [10, 11])
    _demote(tier, stores, keys, only={"main.r2"})
    (done,) = _load(tier, stores, keys, [30, 31])
    assert done.success
    r1 = {_sub_of(tier, "main.r1", k) for k in keys}
    assert set(stores["main.r1"].evicted) == r1
    assert stores["main.r2"].evicted == []
    # The node keeps its copy: the second tier is not exclusive.
    assert {_sub_of(tier, "main.r2", k) for k in keys} <= set(
        stores["main.r2"].node.host
    )


def test_the_sweep_keeps_a_nodes_copy_of_a_hash_it_already_holds():
    """The fake's demotion follows Raiden: a node that holds the hash keeps its bytes,
    and the host copy is freed all the same."""
    tier, stores, registry = _shard_tier()
    key = _key(6)
    _save(tier, stores, [key], [10])
    store = stores["main.r1"]
    h = _sub_of(tier, "main.r1", key)
    store.node.host[h] = b"older copy"
    assert store.demote([h]) == [h]
    assert store.node.host[h] == b"older copy"
    assert h not in store.entries and h not in store.host
    assert registry[h].raiden_id is store.node


def _recompute(stores, names, key, block):
    """The key's blocks recomputed in HBM: same key, other bytes (another chunking)."""
    for name in names:
        stores[name].hbm[block] = _payload(name, key) + b"|recomputed"


def _landed_is_last_saved(tier, stores, names, key, block):
    for name in names:
        store = stores[name]
        assert store.hbm[block] == store.saved[_sub_of(tier, name, key)], name


@pytest.mark.parametrize("victim", [False, True], ids=["store", "victim"])
def test_a_key_its_node_holds_is_not_saved_again(victim):
    """Smoke j-a44349b7 failed P1: 29 keys restored on main.r1, keys.r1 and keys.r2
    differed from what was saved. A key demoted to the node and later recomputed was
    saved again from HBM (a local-only probe never sees the node's block); the sweep
    then freed the new host copy because the node already held the hash, and the peer
    read landed the node's older bytes, not the bytes last saved."""
    tier, stores, _ = _shard_tier()
    key = _key(1)
    names = ("main.r1", "main.r2")
    _save(tier, stores, [key], [5])
    _demote(tier, stores, [key])
    _recompute(stores, names, key, 7)
    before = tier.x2_stats()
    if victim:
        tier.set_victim_mode(True)
        chosen = tier.prepare_victim_store([key])
    else:
        chosen = list(tier.prepare_store([key], _ctx()).keys_to_store)
    after = tier.x2_stats()
    if chosen:
        tier.submit_store_job(3, chosen, [7] * len(chosen))
        _finish(tier, stores)
        # The sweep demotes the new host copies; each node already holds the hash.
        _demote(tier, stores, [key])
    (done,) = _load(tier, stores, [key], [9], job_id=4)
    assert done.success
    # P1: the bytes a load lands are the bytes last saved.
    _landed_is_last_saved(tier, stores, names, key, 9)
    # Every shard's node holds the key, so no shard saves it; one registry probe each.
    assert chosen == []
    assert after["store_probe_lookups"] - before["store_probe_lookups"] == 2
    assert after["store_remote_skips"] - before["store_remote_skips"] == 2


def test_a_shard_whose_node_holds_a_key_skips_it_when_a_sibling_saves_it():
    """A sharded class sends a key to every shard once any shard needs it. Here
    main.r2's node lost the key and main.r1's still holds it: main.r2 saves the
    recomputed bytes, main.r1 must not (its node would keep the older copy)."""
    tier, stores, _ = _shard_tier()
    key = _key(2)
    names = ("main.r1", "main.r2")
    _save(tier, stores, [key], [5])
    _demote(tier, stores, [key])
    stores["main.r2"].node_evicts([_sub_of(tier, "main.r2", key)])
    _recompute(stores, names, key, 7)
    out = tier.prepare_store([key], _ctx())
    assert list(out.keys_to_store) == [key]
    tier.submit_store_job(3, [key], [7])
    (done,) = _finish(tier, stores)
    assert done.success
    held_skips = tier.x2_stats().get("store_held_skips")
    _demote(tier, stores, [key])
    (done,) = _load(tier, stores, [key], [9], job_id=4)
    assert done.success
    # P1: the bytes a load lands are the bytes last saved, in both shards.
    _landed_is_last_saved(tier, stores, names, key, 9)
    assert stores["main.r2"].hbm[9] == _payload("main.r2", key) + b"|recomputed"
    r1 = _sub_of(tier, "main.r1", key)
    assert stores["main.r1"].saved[r1] == _payload("main.r1", key)
    assert held_skips == 1


def test_flag_off_keeps_the_local_probe_and_saves_what_it_is_handed():
    """Without TIER-X2 a store asks no registry and saves every key it is handed."""
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager

    store = ByteStore("main", {}, FakeNode("node-main"))
    manager = RaidenOffloadingManager(
        kernel_physical_blocks_capacity=8,
        offload_logical_blocks_capacity=8,
        device_block_size_factor=1,
        world_size=1,
        raiden_controller_port=0,
        raiden_job_name="t-off",
        store=store,
        store_types=FAKE_STORE_TYPES,
    )
    keys = [_key(1), _key(2)]
    out = manager.prepare_store(keys[:1], _ctx())
    assert list(out.keys_to_store) == keys[:1]
    assert all(not call.args[1] for call in store.calls_named("lookup"))
    for key, b in zip(keys, [3, 4]):
        store.hbm[b] = b"row%d" % b
    # The second key was never prepared: the flag off still saves it.
    manager.submit_store_job(1, keys, [3, 4])
    saved = {h for call in store.calls_named("save") for h in call.args[0]}
    assert saved == {manager._sub_hashes(k)[0] for k in keys}


def test_the_shards_never_read_each_others_blocks():
    """Each shard keys its entries under its own namespace, so one registry holds both
    shards' copies of a key apart."""
    tier, stores, registry = _shard_tier()
    key = _key(9)
    _save(tier, stores, [key], [10])
    _demote(tier, stores, [key])
    r1, r2 = _sub_of(tier, "main.r1", key), _sub_of(tier, "main.r2", key)
    assert r1 != r2
    assert registry[r1].raiden_id is stores["main.r1"].node
    assert registry[r2].raiden_id is stores["main.r2"].node


def test_the_lookup_log_probe_needs_every_shard():
    tier, stores, _ = _shard_tier()
    keys = [_key(1), _key(2)]
    _save(tier, stores, keys, [10, 11])
    subs = [s for k in keys for s in tier._sub_hashes(k)]
    assert [h for h, _ in tier._probe(subs)] == subs
    stores["main.r2"].demote([_sub_of(tier, "main.r2", keys[1])])
    assert [h for h, _ in tier._probe(subs)] == subs[:1]


def test_telemetry_shows_the_x2_counters_only_with_the_flag(monkeypatch):
    from tests.offload.test_tier_classes import _tier
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    monkeypatch.setenv("VLLM_TPU_OFFLOAD_TELEMETRY_S", "60")
    tier, stores, _ = _shard_tier()
    keys = [_key(1)]
    _save(tier, stores, keys, [10])
    _demote(tier, stores, keys)
    _load(tier, stores, keys, [20])
    sched = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    sched._raiden_manager = tier
    sched._fence_pending = {}
    line = sched.telemetry_line()
    assert " tier_x2=registry_lookups:" in line
    assert "remote_read_blocks_done:2" in line
    off, _ = _tier()
    sched_off = TPURaidenOffloadingScheduler.__new__(TPURaidenOffloadingScheduler)
    sched_off._raiden_manager = off
    sched_off._fence_pending = {}
    assert "tier_x2" not in sched_off.telemetry_line()
    assert off.x2_stats() is None


def test_x2_needs_the_registry_in_the_manager():
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager

    with pytest.raises(ValueError, match="needs the global registry"):
        RaidenOffloadingManager(
            kernel_physical_blocks_capacity=4,
            offload_logical_blocks_capacity=4,
            device_block_size_factor=1,
            world_size=1,
            raiden_controller_port=0,
            raiden_job_name="t",
            tier_x2=True,
            store=FakeKVCacheStore(),
            store_types=FAKE_STORE_TYPES,
        )


def test_gdn_groups_of_one_class_share_its_shards():
    """A class of two groups (gdn-ab) keeps both in each shard: every key of either
    group goes to every shard of the class."""
    from vllm_torchtpu.offload.raiden_store import RaidenOffloadingManager
    from vllm_torchtpu.offload.tier_classes import (
        TierClass,
        TieredOffloadingManager,
        TierPlan,
    )

    classes = (
        TierClass("gdn-ab.r1", 0, "gdn", (GDN_A, GDN_B), ((), (1,)), 100, shard_of="gdn-ab"),
        TierClass("gdn-ab.r2", 1, "gdn", (GDN_A, GDN_B), ((0, 1), ()), 100, shard_of="gdn-ab"),
        TierClass("keys", 2, "keys", (KEYS,), ((0,), (1,)), 100, shard_of="keys"),
    )
    plan = TierPlan(classes, budget=0, ratio=2.0, token_blocks=4, checkpoints=4)
    stores = [FakeKVCacheStore() for _ in classes]
    managers = [
        RaidenOffloadingManager(
            kernel_physical_blocks_capacity=plan.capacity(c),
            offload_logical_blocks_capacity=plan.capacity(c),
            device_block_size_factor=1,
            world_size=c.num_workers,
            raiden_controller_port=0,
            raiden_job_name=f"t-{c.name}",
            store=s,
            store_types=FAKE_STORE_TYPES,
        )
        for c, s in zip(classes, stores)
    ]
    tier = TieredOffloadingManager(plan, managers)
    keys = [_key(1, GDN_A), _key(1, GDN_B), _key(1, KEYS)]
    assert tier.classes_of(keys[0]) == (0, 1) == tier.classes_of(keys[1])
    assert tier.classes_of(keys[2]) == (2,)
    parts = tier._split(keys, [5, 6, 7])
    assert parts == {0: (keys[:2], [5, 6]), 1: (keys[:2], [5, 6]), 2: ([keys[2]], [7])}
    assert tier.bytes_of_keys(keys) == 2 * (100 + 200) + 200


# -- the served scheduler end to end ----------------------------------------------------


class AutoByteStore(ByteStore):
    """ByteStore whose saves land at once with a row of their device block (the
    scheduler owns the device rows here)."""

    def save(self, block_hashes, dst_raiden_id=None):
        ok = FakeKVCacheStore.save(self, block_hashes, dst_raiden_id)
        if ok:
            for h in block_hashes:
                self.host[h] = b"%s|row%d" % (self.name.encode(), self.entries[h].device_block_id)
            self.complete_save(list(block_hashes))
        return ok


def test_served_scheduler_hits_and_reads_the_second_tier_after_demotion(monkeypatch):
    """The real connector scheduler on the served PP8 groups, victim policy, nine shard
    stores: turn one's evicted blocks land in every shard, the sweep demotes them all,
    and a returning prompt hits them through the registry and reads them back."""
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec

    from tests.offload.test_tier_classes import _turn_one, _victims_of_turn_one
    from tests.runner.test_qwen4_exp_host_tier import BLOCK
    from tests.runner.test_qwen4_exp_host_tier import _request as request
    from tests.runner.test_qwen4_exp_offload import _Blocks, _tokens
    from vllm_torchtpu.offload import victim
    from vllm_torchtpu.offload.raiden_connector import TPURaidenOffloadingScheduler

    monkeypatch.setenv(victim.POLICY_ENV, "victim")
    victim.deactivate()
    cfg, _, stages, kvc = _served(monkeypatch)
    _x2_env(monkeypatch)
    spec = _spec(monkeypatch, cfg, stages, kvc, cpu_bytes=64 * 1024**3, **REGISTRY)
    classes = spec.tier_plan.classes
    registry: dict = {}
    stores = [AutoByteStore(c.name, registry, FakeNode(c.name)) for c in classes]
    spec._manager = spec.build_tier_manager(stores, FAKE_STORE_TYPES)
    sched = TPURaidenOffloadingScheduler(spec, spec.vllm_config, kvc)
    try:
        first, table, _ = _turn_one(sched, kvc)
        _victims_of_turn_one(sched, first, table, kvc)
        sched.build_connector_meta(SchedulerOutput.make_empty())
        for c, store in zip(classes, stores):
            assert store.calls_named("save"), c.name
        # A returning prompt the host pool holds: every hit is local (the registry is
        # asked only past the stored blocks).
        second = request("r2", first.prompt_token_ids + _tokens(2000, 1))
        sched.on_new_request(second)
        assert sched.get_num_new_matched_tokens(second, 0) == (3 * BLOCK, True)
        assert sched._raiden_manager.x2_stats()["remote_hit_keys"] == 0
        # The sweep demotes every host entry to the shard's node.
        for store in stores:
            store.demote(list(store.entries))
            assert not store.entries
        third = request("r3", first.prompt_token_ids + _tokens(2000, 2))
        sched.on_new_request(third)
        assert sched.get_num_new_matched_tokens(third, 0) == (3 * BLOCK, True)
        stats = sched._raiden_manager.x2_stats()
        assert stats["registry_lookups"] > 0 and stats["remote_hit_keys"] > 0
        blocks = _Blocks()
        blocks.next_id = 100
        alloc = []
        for group in kvc.kv_cache_groups:
            if isinstance(group.kv_cache_spec, CircularBufferSpec):
                ids = blocks.take(1)
            elif isinstance(group.kv_cache_spec, MambaSpec):
                ids = [0, 0, *blocks.take(1)]
            else:
                ids = blocks.take(3)
            alloc.append([KVCacheBlock(i, is_null=i == 0) for i in ids])
        sched.update_state_after_alloc(third, KVCacheBlocks(tuple(alloc)), 3 * BLOCK)
        sched.build_connector_meta(SchedulerOutput.make_empty())
        for c, store in zip(classes, stores):
            calls = store.calls_named("read_remote")
            assert calls and not store.calls_named("load"), c.name
            for call in calls:
                assert {sl.raiden_id for sl in call.args[1]} == {store.node}
            # Each block's bytes are its own shard's: the row the save took.
            for h, dst in zip(call.args[0], call.args[2]):
                assert store.node.host[h].startswith(c.name.encode() + b"|row")
            store.finish()
        sched.build_connector_meta(SchedulerOutput.make_empty())
        stats = sched._raiden_manager.x2_stats()
        assert stats["remote_read_blocks_failed"] == 0
        assert stats["remote_read_blocks_done"] == stats["remote_read_blocks"] > 0
        assert not sched._raiden_manager.has_pending_work()
        line = sched.telemetry_line()
        assert " tier_x2=" in line and "tier[main.r1]" in line and "tier[ple]" in line
    finally:
        victim.deactivate()
