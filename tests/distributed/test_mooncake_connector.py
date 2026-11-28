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
"""Unit tests for TPUMooncakeConnector direct P2P disaggregated P/D transfers.

Tests scheduler handshake, partial local prefix-hit slicing, contiguous
block-run coalescing, TPU write fence gating, ZMQ control protocol (ready/done),
expiration safety, and error reporting using a FakeTransferEngine over loopback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.v1.request import RequestStatus

from vllm_torchtpu.distributed.kv_transfer import mooncake_tpu_utils as mc_utils
from vllm_torchtpu.distributed.kv_transfer.mooncake_connector import (
    MooncakeP2PConfig,
    MooncakeP2PLoadMeta,
    MooncakeP2PMetadata,
    MooncakeP2PSendMeta,
    TPUMooncakeConnectorScheduler,
    TPUMooncakeConnectorWorker,
    parse_mooncake_p2p_config,
)


@dataclass(frozen=True)
class _FakeShard:
    fake_addr: int
    size: int
    device: int = 0


class _FakeMooncakeTpu:
    def __init__(self, base_addr: int = 0x1000_0000):
        self._next_addr = base_addr
        self.registered_with_engine: list[tuple[object, list[list[_FakeShard]]]] = []
        self.unregistered: list[_FakeShard] = []

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

    def register_with_engine(self, engine, shards):
        self.registered_with_engine.append((engine, shards))
        flat = [s for group in shards for s in group]
        engine.batch_register_memory(
            [s.fake_addr for s in flat], [s.size for s in flat]
        )

    def unregister(self, shards):
        self.unregistered.extend(shards)


class _FakeTransferEngine:
    def __init__(self, rpc_port: int, fail_read: bool = False):
        self.rpc_port = rpc_port
        self.fail_read = fail_read
        self.init_args: tuple | None = None
        self.registered_addrs: list[int] = []
        self.unregistered_addrs: list[int] = []
        self.read_calls: list[tuple[str, list[int], list[int], list[int]]] = []

    def initialize(self, host, metadata_server, protocol, device_name) -> int:
        self.init_args = (host, metadata_server, protocol, device_name)
        return 0

    def get_rpc_port(self) -> int:
        return self.rpc_port

    def batch_register_memory(self, addrs: list[int], sizes: list[int]) -> int:
        del sizes
        self.registered_addrs.extend(addrs)
        return 0

    def batch_unregister_memory(self, addrs: list[int]) -> int:
        self.unregistered_addrs.extend(addrs)
        return 0

    def batch_transfer_sync_read(
        self,
        remote_session_id: str,
        local_addrs: list[int],
        remote_addrs: list[int],
        lengths: list[int],
    ) -> int:
        self.read_calls.append(
            (remote_session_id, list(local_addrs), list(remote_addrs), list(lengths))
        )
        return -1 if self.fail_read else 0


class _FakeFenceEvent:
    def __init__(self, retired: bool = False):
        self.retired = retired

    def query(self) -> bool:
        return self.retired

    def synchronize(self) -> None:
        self.retired = True


def _make_vllm_config(*, is_producer: bool, tp_size: int = 1, block_size: int = 64):
    return SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            is_kv_producer=is_producer,
            kv_connector_extra_config={},
        ),
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            data_parallel_rank=0,
        ),
    )


def _make_p2p_config(*, ctrl_port_base: int = 0, wait_pull_timeout_s: float = 30.0):
    return MooncakeP2PConfig(
        protocol="hp_tcp",
        metadata_server="P2PHANDSHAKE",
        device_name="",
        host_ip="127.0.0.1",
        ctrl_port_base=ctrl_port_base,
        ctrl_bind_host="127.0.0.1",
        wait_pull_timeout_s=wait_pull_timeout_s,
        ready_timeout_s=5.0,
        tent_connections=4,
        transfer_threads=2,
        remote_hosts=("127.0.0.1",),
    )


def _fake_blocks(block_ids: list[int]):
    return SimpleNamespace(get_block_ids=lambda: (list(block_ids),))


def test_scheduler_p2p_handshake_and_partial_prefix_hit():
    p_vllm_cfg = _make_vllm_config(is_producer=True, tp_size=2, block_size=64)
    p_cfg = parse_mooncake_p2p_config(
        p_vllm_cfg,
        {"local_hostname": "10.0.0.1", "ctrl_port": 27600},
    )
    p_sched = TPUMooncakeConnectorScheduler(p_vllm_cfg, p_cfg)

    # 260 tokens -> 4 full blocks (256 tokens), trailing partial block dropped.
    req_p = SimpleNamespace(
        request_id="req-1",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        num_computed_tokens=260,
        prompt_token_ids=list(range(260)),
        kv_transfer_params=None,
    )
    delay_free, params = p_sched.request_finished(req_p, [10, 11, 12, 13, 14])
    assert delay_free is True
    assert params is not None
    assert params["remote_block_ids"] == [10, 11, 12, 13]
    assert params["remote_tp_size"] == 2
    assert params["remote_ctrl_endpoints"] == [
        "tcp://10.0.0.1:27600",
        "tcp://10.0.0.1:27601",
    ]
    p_meta = p_sched.build_connector_meta()
    assert "req-1" in p_meta.reqs_to_send
    assert p_meta.reqs_to_send["req-1"].local_block_ids == [10, 11, 12, 13]

    # Decode side with 128 locally cached tokens (2 blocks hit -> load 2 suffix blocks).
    d_vllm_cfg = _make_vllm_config(is_producer=False, tp_size=2, block_size=64)
    d_cfg = parse_mooncake_p2p_config(d_vllm_cfg, {"local_hostname": "10.0.0.2"})
    d_sched = TPUMooncakeConnectorScheduler(d_vllm_cfg, d_cfg)
    req_d = SimpleNamespace(
        request_id="req-1",
        prompt_token_ids=list(range(260)),
        kv_transfer_params=dict(params),
    )
    d_sched.on_new_request(req_d)
    matched, is_async = d_sched.get_num_new_matched_tokens(req_d, 128)
    assert (matched, is_async) == (128, True)

    d_sched.update_state_after_alloc(
        req_d, _fake_blocks([1, 2, 20, 21]), num_external_tokens=128
    )
    d_meta = d_sched.build_connector_meta()
    load = d_meta.reqs_to_load["req-1"]
    assert load.release_only is False
    assert load.local_block_ids == [20, 21]
    assert load.remote_block_ids == [12, 13]


def test_scheduler_rejects_mismatched_tp_size_and_handles_full_local_hit():
    d_vllm_cfg = _make_vllm_config(is_producer=False, tp_size=2, block_size=64)
    d_cfg = parse_mooncake_p2p_config(d_vllm_cfg, {"local_hostname": "10.0.0.2"})
    d_sched = TPUMooncakeConnectorScheduler(d_vllm_cfg, d_cfg)

    bad_req = SimpleNamespace(
        request_id="req-bad",
        prompt_token_ids=list(range(128)),
        kv_transfer_params={
            "uuid": 123,
            "remote_block_ids": [0, 1],
            "remote_ctrl_endpoints": ["tcp://10.0.0.1:27600"],
            "remote_tp_size": 1,
        },
    )
    assert d_sched.get_num_new_matched_tokens(bad_req, 0) == (0, False)

    # Full local prefix hit -> release_only notify with report_completion=False.
    hit_req = SimpleNamespace(
        request_id="req-hit",
        prompt_token_ids=list(range(128)),
        kv_transfer_params={
            "uuid": 456,
            "remote_block_ids": [0, 1],
            "remote_ctrl_endpoints": [
                "tcp://10.0.0.1:27600",
                "tcp://10.0.0.1:27601",
            ],
            "remote_tp_size": 2,
        },
    )
    assert d_sched.get_num_new_matched_tokens(hit_req, 128) == (0, False)
    d_sched.update_state_after_alloc(hit_req, _fake_blocks([4, 5]), 0)
    meta = d_sched.build_connector_meta()
    assert meta.reqs_to_load["req-hit"].release_only is True
    assert meta.reqs_to_load["req-hit"].report_completion is False


def test_worker_p2p_fenced_pull_coalescing_and_release(monkeypatch):
    fence_events: list[_FakeFenceEvent] = []
    monkeypatch.setattr(mc_utils, "enqueue_pool_writes", lambda tensors: None)
    monkeypatch.setattr(
        mc_utils,
        "record_fence_event",
        lambda: (
            fence_events.append(_FakeFenceEvent(retired=False)) or fence_events[-1]
        ),
    )
    monkeypatch.setattr(mc_utils, "event_is_retired", lambda ev: ev.query())

    p_engine = _FakeTransferEngine(rpc_port=15001)
    d_engine = _FakeTransferEngine(rpc_port=15002)
    p_tpu = _FakeMooncakeTpu(base_addr=0x1000_0000)
    d_tpu = _FakeMooncakeTpu(base_addr=0x5000_0000)

    kv_cfg = SimpleNamespace(num_blocks=8)
    p_worker = TPUMooncakeConnectorWorker(
        _make_vllm_config(is_producer=True, tp_size=1),
        kv_cfg,
        _make_p2p_config(ctrl_port_base=0),
        tp_rank=0,
        tp_size=1,
        engine_factory=lambda: p_engine,
        provider=object(),
        mc_tpu_module=p_tpu,
    )
    d_worker = TPUMooncakeConnectorWorker(
        _make_vllm_config(is_producer=False, tp_size=1),
        kv_cfg,
        _make_p2p_config(ctrl_port_base=0),
        tp_rank=0,
        tp_size=1,
        engine_factory=lambda: d_engine,
        provider=object(),
        mc_tpu_module=d_tpu,
    )

    try:
        p_pool = torch.zeros(8, 128, dtype=torch.int8)
        d_pool = torch.zeros(8, 128, dtype=torch.int8)
        p_worker.register_kv_caches({"layer.0": p_pool})
        d_worker.register_kv_caches({"layer.0": d_pool})

        hs = p_worker.get_handshake_metadata()
        assert hs is not None
        assert hs.session_id == "127.0.0.1:15001"
        assert hs.ctrl_endpoint.startswith("tcp://127.0.0.1:")

        # Start Decode pull FIRST (testing out-of-band registration race + fence wait).
        d_worker.process_send_load(
            MooncakeP2PMetadata(
                reqs_to_load={
                    "req-race": MooncakeP2PLoadMeta(
                        uuid=999,
                        local_block_ids=[2, 3, 4, 7],
                        remote_block_ids=[0, 1, 2, 5],
                        remote_ctrl_endpoints=[hs.ctrl_endpoint],
                    )
                }
            )
        )

        # Register send on Producer with fence unretired: pull must wait.
        p_worker.process_send_load(
            MooncakeP2PMetadata(
                reqs_to_send={
                    "req-race": MooncakeP2PSendMeta(
                        uuid=999,
                        local_block_ids=[0, 1, 2, 5],
                        expiration_time=time.time() + 30.0,
                    )
                }
            )
        )
        time.sleep(0.05)
        assert d_engine.read_calls == []

        # Retire TPU fence -> Producer releases ready -> Decode pulls and sends done.
        fence_events[0].retired = True
        deadline = time.monotonic() + 5.0
        d_done: set[str] = set()
        p_done: set[str] = set()
        while (not d_done or not p_done) and time.monotonic() < deadline:
            p_res = p_worker.get_transfer_results(set())
            d_res = d_worker.get_transfer_results(set())
            p_done.update(p_res.finished_sending)
            d_done.update(d_res.finished_recving)
            assert not d_res.failed_recving
            time.sleep(0.01)

        assert d_done == {"req-race"}
        assert p_done == {"req-race"}

        # Verify coalescing: 4 blocks [(0->2, 1->3, 2->4), (5->7)] -> 2 TENT spans.
        assert len(d_engine.read_calls) == 1
        session, local_ptrs, remote_ptrs, lengths = d_engine.read_calls[0]
        assert session == "127.0.0.1:15001"
        assert local_ptrs == [0x5000_0000 + 2 * 128, 0x5000_0000 + 7 * 128]
        assert remote_ptrs == [0x1000_0000 + 0 * 128, 0x1000_0000 + 5 * 128]
        assert lengths == [3 * 128, 128]
    finally:
        d_worker.shutdown()
        p_worker.shutdown()

    assert len(p_engine.unregistered_addrs) == 1
    assert len(d_engine.unregistered_addrs) == 1
    assert len(p_tpu.unregistered) == 1
    assert len(d_tpu.unregistered) == 1


def test_worker_p2p_transfer_failure_and_producer_ttl_expiry(monkeypatch):
    monkeypatch.setattr(mc_utils, "enqueue_pool_writes", lambda tensors: None)
    monkeypatch.setattr(
        mc_utils, "record_fence_event", lambda: _FakeFenceEvent(retired=True)
    )
    monkeypatch.setattr(mc_utils, "event_is_retired", lambda ev: True)

    p_engine = _FakeTransferEngine(rpc_port=15011)
    d_engine = _FakeTransferEngine(rpc_port=15012, fail_read=True)
    kv_cfg = SimpleNamespace(num_blocks=4)

    p_worker = TPUMooncakeConnectorWorker(
        _make_vllm_config(is_producer=True),
        kv_cfg,
        _make_p2p_config(ctrl_port_base=0, wait_pull_timeout_s=0.05),
        tp_rank=0,
        tp_size=1,
        engine_factory=lambda: p_engine,
        provider=object(),
        mc_tpu_module=_FakeMooncakeTpu(0x1000_0000),
    )
    d_worker = TPUMooncakeConnectorWorker(
        _make_vllm_config(is_producer=False),
        kv_cfg,
        _make_p2p_config(ctrl_port_base=0),
        tp_rank=0,
        tp_size=1,
        engine_factory=lambda: d_engine,
        provider=object(),
        mc_tpu_module=_FakeMooncakeTpu(0x5000_0000),
    )

    try:
        pool = torch.zeros(4, 64, dtype=torch.int8)
        p_worker.register_kv_caches({"layer.0": pool})
        d_worker.register_kv_caches({"layer.0": torch.zeros(4, 64, dtype=torch.int8)})
        hs = p_worker.get_handshake_metadata()
        assert hs is not None

        # Failed read still notifies producer to free blocks and reports failed_recving.
        p_worker.process_send_load(
            MooncakeP2PMetadata(
                reqs_to_send={
                    "req-err": MooncakeP2PSendMeta(
                        uuid=101,
                        local_block_ids=[0, 1],
                        expiration_time=time.time() + 30.0,
                    )
                }
            )
        )
        d_worker.process_send_load(
            MooncakeP2PMetadata(
                reqs_to_load={
                    "req-err": MooncakeP2PLoadMeta(
                        uuid=101,
                        local_block_ids=[0, 1],
                        remote_block_ids=[0, 1],
                        remote_ctrl_endpoints=[hs.ctrl_endpoint],
                    )
                }
            ),
            wait_for_completion=True,
        )
        d_res = d_worker.get_transfer_results(set())
        assert d_res.finished_recving == {"req-err"}
        assert d_res.failed_recving == {"req-err"}

        p_res = p_worker.get_transfer_results(set())
        assert p_res.finished_sending == {"req-err"}

        # Abandoned send expires past TTL so Producer never leaks HBM blocks.
        p_worker.process_send_load(
            MooncakeP2PMetadata(
                reqs_to_send={
                    "req-expire": MooncakeP2PSendMeta(
                        uuid=202,
                        local_block_ids=[2],
                        expiration_time=time.time() - 1.0,
                    )
                }
            )
        )
        exp_res = p_worker.get_transfer_results(set())
        assert exp_res.finished_sending == {"req-expire"}
    finally:
        d_worker.shutdown()
        p_worker.shutdown()
