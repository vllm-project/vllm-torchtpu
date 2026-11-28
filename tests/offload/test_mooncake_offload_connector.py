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
"""Unit tests for TPUMooncakeOffloadingConnector, manager, and worker.

Runs on CPU tensors with in-memory FakeMooncakeStore and fake TPU fence events
(no TPU hardware or live mooncake_master required).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
    LookupResult,
    ReqContext,
    make_offload_key,
)

from vllm_torchtpu.distributed.kv_transfer import mooncake_tpu_utils as mc_utils
from vllm_torchtpu.offload.mooncake_connector import TPUMooncakeOffloadingConnector
from vllm_torchtpu.offload.mooncake_store import (
    MooncakeLoadStoreSpec,
    MooncakeOffloadConfig,
    MooncakeOffloadingManager,
    MooncakeStoreOffloadingWorker,
    build_mooncake_key_prefix,
    format_mooncake_block_key,
    parse_mooncake_offload_config,
)


@dataclass(frozen=True)
class _FakeShard:
    fake_addr: int
    size: int
    device: int = 0


class _FakeMooncakeTpu:
    def __init__(self):
        self.unregistered: list[_FakeShard] = []
        self._next_addr = 0x1000_0000

    def prepare(self, tensors, *, provider):
        del provider
        out = []
        for t in tensors:
            shard = _FakeShard(
                fake_addr=self._next_addr,
                size=t.numel() * t.element_size(),
                device=0,
            )
            self._next_addr += 0x1000_0000
            out.append([shard])
        return out

    def unregister(self, shards):
        self.unregistered.extend(shards)


class _FakeFenceEvent:
    def __init__(self, retired: bool = False):
        self.retired = retired
        self.synced = False

    def query(self) -> bool:
        return self.retired

    def synchronize(self) -> None:
        self.retired = True
        self.synced = True


class _FakeMooncakeStore:
    """In-memory stand-in for `mooncake.store.MooncakeDistributedStore`."""

    def __init__(self):
        self.setup_calls: list[tuple] = []
        self.registered: dict[int, int] = {}
        self.unregistered: list[int] = []
        self.objects: dict[str, list[tuple[int, int]]] = {}
        self.put_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
        self.get_calls: list[tuple[list[str], list[list[int]], list[list[int]]]] = []
        self.fail_put = False
        self.fail_get = False
        self.fail_exist = False
        self.closed = False

    def setup(self, *args) -> int:
        self.setup_calls.append(args)
        return 0

    def register_buffer(self, addr: int, size: int) -> int:
        self.registered[addr] = size
        return 0

    def unregister_buffer(self, addr: int) -> int:
        self.registered.pop(addr, None)
        self.unregistered.append(addr)
        return 0

    def batch_is_exist(self, keys: list[str]) -> list[int]:
        if self.fail_exist:
            raise RuntimeError("master unreachable")
        return [1 if k in self.objects else 0 for k in keys]

    def batch_put_from_multi_buffers(
        self,
        keys: list[str],
        all_ptrs: list[list[int]],
        all_sizes: list[list[int]],
    ) -> list[int]:
        self.put_calls.append((list(keys), list(all_ptrs), list(all_sizes)))
        if self.fail_put:
            return [-1 for _ in keys]
        for k, ptrs, sizes in zip(keys, all_ptrs, all_sizes):
            self.objects[k] = list(zip(ptrs, sizes))
        return [0 for _ in keys]

    def batch_get_into_multi_buffers(
        self,
        keys: list[str],
        all_ptrs: list[list[int]],
        all_sizes: list[list[int]],
    ) -> list[int]:
        self.get_calls.append((list(keys), list(all_ptrs), list(all_sizes)))
        if self.fail_get:
            return [-1 for _ in keys]
        out = []
        for k, sizes in zip(keys, all_sizes):
            out.append(sum(sizes) if k in self.objects else -1)
        return out

    def remove_all(self) -> int:
        self.objects.clear()
        return 0

    def close(self) -> None:
        self.closed = True


def _sample_config() -> MooncakeOffloadConfig:
    return MooncakeOffloadConfig(
        master_server_address="127.0.0.1:50051",
        metadata_server="http://127.0.0.1:8080/metadata",
        protocol="hp_tcp",
        device_name="",
        global_segment_size=64 * 1024 * 1024,
        local_buffer_size=16 * 1024 * 1024,
        tent_connections=4,
        transfer_threads=2,
        namespace="unit",
        host_ip="127.0.0.1",
    )


def _wait_for_results(worker: MooncakeStoreOffloadingWorker, expected: int):
    deadline = time.monotonic() + 5.0
    results = []
    while len(results) < expected and time.monotonic() < deadline:
        results.extend(worker.get_finished())
        if len(results) < expected:
            time.sleep(0.01)
    assert len(results) == expected
    return results


def test_parse_offload_config_and_key_prefix(monkeypatch):
    monkeypatch.setenv("TPU_MOONCAKE_MASTER_SERVER_ADDRESS", "10.0.0.1:50051")
    monkeypatch.setenv("TPU_MOONCAKE_METADATA_SERVER", "http://10.0.0.1:8080/meta")
    cfg = parse_mooncake_offload_config(
        {"global_segment_size": "2GB", "local_buffer_size": "256MB"}
    )
    assert cfg.master_server_address == "10.0.0.1:50051"
    assert cfg.metadata_server == "http://10.0.0.1:8080/meta"
    assert cfg.global_segment_size == 2 * 1024**3
    assert cfg.local_buffer_size == 256 * 1024**2

    vllm_cfg = SimpleNamespace(
        model_config=SimpleNamespace(model="llama-3", dtype="bfloat16"),
        cache_config=SimpleNamespace(cache_dtype="auto", block_size=64),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
        ),
    )
    p1 = build_mooncake_key_prefix(vllm_cfg, namespace="a")
    p2 = build_mooncake_key_prefix(vllm_cfg, namespace="b")
    assert p1 != p2
    assert p1 == build_mooncake_key_prefix(vllm_cfg, namespace="a")


def test_manager_lookup_requires_all_ranks_and_dedups_stores():
    store = _FakeMooncakeStore()
    mgr = MooncakeOffloadingManager(
        config=_sample_config(),
        key_prefix="mc_test",
        num_workers=2,
        enable_kv_cache_events=True,
        store=store,
    )
    ctx = ReqContext(req_id="req-1")
    k0 = make_offload_key(b"blk0", 0)
    k1 = make_offload_key(b"blk1", 0)

    # Empty store -> MISS.
    assert mgr.lookup(k0, ctx) == LookupResult.MISS

    # Prepare store marks keys in-flight -> HIT_PENDING.
    prep = mgr.prepare_store([k0, k1], ctx)
    assert prep is not None
    assert prep.keys_to_store == [k0, k1]
    assert mgr.lookup(k0, ctx) == LookupResult.HIT_PENDING
    assert mgr.prepare_store([k0, k1], ctx) is None

    # Only rank 0 persisted -> still MISS until all ranks exist.
    store.objects[format_mooncake_block_key("mc_test", 0, k0)] = [(0, 64)]
    mgr.complete_store([k0, k1], ctx, success=True)
    assert mgr.lookup(k0, ctx) == LookupResult.MISS

    # Both ranks present -> HIT; k0 skipped on next prepare_store, k1 offered.
    store.objects[format_mooncake_block_key("mc_test", 1, k0)] = [(0, 64)]
    assert mgr.lookup(k0, ctx) == LookupResult.HIT
    prep2 = mgr.prepare_store([k0, k1], ctx)
    assert prep2 is not None
    assert prep2.keys_to_store == [k1]

    # KV cache events emitted on complete_store.
    events = list(mgr.take_events())
    assert len(events) == 1
    assert events[0].keys == [k0, k1]
    assert events[0].removed is False

    # Degrades safely to MISS if master is temporarily unreachable.
    store.fail_exist = True
    assert mgr.lookup(k0, ctx) == LookupResult.MISS
    store.fail_exist = False

    mgr.reset_cache()
    assert store.objects == {}
    mgr.shutdown()
    assert store.closed


def test_worker_save_fence_async_put_get_wait_and_shutdown(monkeypatch):
    store = _FakeMooncakeStore()
    fake_tpu = _FakeMooncakeTpu()
    events_created: list[_FakeFenceEvent] = []
    enqueued: list[int] = []

    monkeypatch.setattr(
        mc_utils, "enqueue_pool_writes", lambda tensors: enqueued.append(len(tensors))
    )
    monkeypatch.setattr(
        mc_utils,
        "record_fence_event",
        lambda: (
            events_created.append(_FakeFenceEvent(retired=False)) or events_created[-1]
        ),
    )
    monkeypatch.setattr(mc_utils, "event_is_retired", lambda ev: ev.query())
    monkeypatch.setattr(
        mc_utils, "synchronize_fence_event", lambda ev: ev.synchronize()
    )

    pool0 = torch.zeros(4, 64, dtype=torch.int8)
    pool1 = torch.zeros(4, 128, dtype=torch.int8)
    canonical = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=pool0, page_size_bytes=64),
            CanonicalKVCacheTensor(tensor=pool1, page_size_bytes=128),
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=64),
                CanonicalKVCacheRef(tensor_idx=1, page_size_bytes=128),
            ]
        ],
    )

    worker = MooncakeStoreOffloadingWorker(
        canonical,
        config=_sample_config(),
        key_prefix="mc_test",
        rank=0,
        store=store,
        provider=object(),
        mc_tpu_module=fake_tpu,
    )
    assert len(store.registered) == 2

    k0 = make_offload_key(b"blk0", 0)
    k1 = make_offload_key(b"blk1", 0)
    gpu_spec = GPULoadStoreSpec([1, 3], group_sizes=[2], block_indices=[0])
    mc_spec = MooncakeLoadStoreSpec([k0, k1])

    # Submit store while fence is NOT yet retired: PUT must NOT launch yet.
    assert worker.submit_store(10, gpu_spec, mc_spec) is True
    assert enqueued == [2]
    assert worker.get_finished() == []
    assert store.put_calls == []

    # Retire the fence event -> next poll dispatches PUT and completes.
    events_created[0].retired = True
    results = _wait_for_results(worker, expected=1)
    assert results[0].job_id == 10
    assert results[0].success is True
    assert results[0].transfer_size == 2 * (64 + 128)
    assert len(store.put_calls) == 1

    # Load back into blocks [0, 2] -> succeeds immediately without fencing.
    load_dst = GPULoadStoreSpec([0, 2], group_sizes=[2], block_indices=[0])
    assert worker.submit_load(11, mc_spec, load_dst) is True
    load_results = _wait_for_results(worker, expected=1)
    assert load_results[0].job_id == 11
    assert load_results[0].success is True
    assert len(store.get_calls) == 1

    # jobs_to_flush wait() synchronizes an unretired fence and drains the PUT.
    k2 = make_offload_key(b"blk2", 0)
    assert (
        worker.submit_store(
            12,
            GPULoadStoreSpec([2], group_sizes=[1], block_indices=[0]),
            MooncakeLoadStoreSpec([k2]),
        )
        is True
    )
    assert not events_created[-1].retired
    worker.wait({12})
    assert events_created[-1].synced is True
    drained = worker.get_finished()
    assert len(drained) == 1 and drained[0].job_id == 12 and drained[0].success

    # Shutdown unregisters buffers from store and shards from mooncake.tpu.
    worker.shutdown()
    assert len(store.unregistered) == 2
    assert store.closed is True
    assert len(fake_tpu.unregistered) == 2


def test_connector_register_kv_caches_and_failed_load_reporting(monkeypatch):
    store = _FakeMooncakeStore()
    fake_tpu = _FakeMooncakeTpu()
    monkeypatch.setattr(mc_utils, "enqueue_pool_writes", lambda tensors: None)
    monkeypatch.setattr(
        mc_utils, "record_fence_event", lambda: _FakeFenceEvent(retired=True)
    )
    monkeypatch.setattr(mc_utils, "event_is_retired", lambda ev: True)

    pool0 = torch.zeros(4, 64, dtype=torch.int8)
    canonical_captured: list[CanonicalKVCaches] = []

    def _init_worker(canonical: CanonicalKVCaches) -> None:
        canonical_captured.append(canonical)
        fake_conn_worker.worker = MooncakeStoreOffloadingWorker(
            canonical,
            config=_sample_config(),
            key_prefix="mc_test",
            rank=0,
            store=store,
            provider=object(),
            mc_tpu_module=fake_tpu,
        )

    fake_conn_worker = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            num_blocks=4,
            kv_cache_groups=[object()],
            hisparse_host_num_blocks=None,
            prefix_cacheable_group_ids=(0,),
        ),
        worker=None,
        _load_jobs={},
        _unsubmitted_store_jobs=[],
        _connector_worker_meta=OffloadingWorkerMetadata(),
        _init_worker=_init_worker,
        prepare_store_kv=lambda meta: None,
    )
    conn = object.__new__(TPUMooncakeOffloadingConnector)
    conn.connector_worker = fake_conn_worker  # type: ignore[assignment]
    conn.connector_scheduler = None

    conn.register_kv_caches({"layer.0": pool0, "layer.1": pool0.view(4, 2, 32)})
    assert len(canonical_captured) == 1
    assert len(canonical_captured[0].tensors) == 1
    assert canonical_captured[0].tensors[0].page_size_bytes == 64

    # Simulate a load that fails because the key was evicted from Mooncake Store.
    store.fail_get = True
    k_missing = make_offload_key(b"gone", 0)
    dst_spec = GPULoadStoreSpec([0], group_sizes=[1], block_indices=[0])
    src_spec = MooncakeLoadStoreSpec([k_missing])
    fake_conn_worker._load_jobs[42] = "req-fail"
    assert fake_conn_worker.worker.submit_load(42, src_spec, dst_spec) is True

    meta = OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
    conn.bind_connector_metadata(meta)

    deadline = time.monotonic() + 5.0
    transfer_res = None
    while time.monotonic() < deadline:
        res = conn.get_transfer_results(set())
        if res.finished_recving:
            transfer_res = res
            break
        time.sleep(0.01)

    assert transfer_res is not None
    assert transfer_res.finished_recving == {"req-fail"}
    assert transfer_res.failed_recving == {"req-fail"}
    assert fake_conn_worker._connector_worker_meta.completed_jobs == {42: 1}

    fake_conn_worker.worker.shutdown()
