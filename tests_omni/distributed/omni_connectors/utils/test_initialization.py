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
"""TPU tests for OmniConnector topology and ZMQ control-plane initialization."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest
import vllm_omni
from vllm_omni.distributed.omni_connectors.connectors.nixl_connector import (
    NixlConnector,
    _PendingPayload,
)
from vllm_omni.distributed.omni_connectors.utils.config import ConnectorSpec
from vllm_omni.distributed.omni_connectors.utils.initialization import (
    KV_RANK_PORT_STRIDE,
    KV_REPLICA_PORT_STRIDE,
    KV_TRANSFER_PORT_OFFSET,
    ORCHESTRATOR_PORT_OFFSET,
    REQUEST_FORWARDING_PORT_OFFSET,
    compute_connector_zmq_port,
    get_connectors_config_for_stage,
    load_omni_transfer_config,
    resolve_connector_spec,
)
from vllm_omni.platforms import current_omni_platform

from tests_omni.omni_test_utils import PR_TIMEOUT_S
from vllm_torchtpu.omni.platform import OmniTpuPlatform

pytestmark = [
    pytest.mark.omni,
    pytest.mark.timeout(PR_TIMEOUT_S),
]


def _make_handshake_nixl_connector(config: dict[str, Any]) -> NixlConnector:
    """Construct a NixlConnector instance for ZMQ handshake-only checks."""
    conn = object.__new__(NixlConnector)
    conn.config = config
    conn._state_lock = threading.RLock()
    conn._close_lock = threading.Lock()
    conn._closed = False
    conn._closing = False
    conn._close_thread = None
    conn._pending = {}
    conn._published = {}
    conn._registered_descs = []
    conn._remote_agents = set()
    conn._deferred_transfers = []
    conn._abandoned_queries = []
    conn._lease_seconds = 60.0
    conn._poll_interval_s = 0.01
    conn._lease_wakeup = threading.Event()
    conn._transfer_wakeup = threading.Event()
    conn._lease_thread = None
    conn._transfer_thread = None
    conn._listener_thread = None
    conn._init_handshake()
    return conn


def test_connector_edge_and_topology_configuration() -> None:
    """Connector Edge and Topology Configuration.

    - TPU path: Resolve explicit stage mappings, top-level ``edges`` fallback,
      and OmniTpuPlatform's default ``wan2_2_eg.yaml`` with OmniTpuPlatform
      active.
    - Setup check: Assert ``current_omni_platform`` is ``OmniTpuPlatform`` and
      ``get_default_stage_config_path() == "vllm_omni/deploy"``.
    - Outcome check: Verify directed edges map to declared ``ConnectorSpec`` or
      auto-configured ``SharedMemoryConnector`` with shared ``wakeup_scope``,
      per-stage configs inject directional roles, and mismatched edge names
      raise ``ValueError``.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )
    default_cfg_path = current_omni_platform.get_default_stage_config_path()
    assert default_cfg_path == "vllm_omni/deploy", (
        f"Unexpected default stage config path: {default_cfg_path!r}"
    )

    # The default path is relative to site-packages, where vllm_omni is installed.
    deploy_yaml = (
        Path(vllm_omni.__file__).resolve().parent.parent
        / default_cfg_path
        / "wan2_2_eg.yaml"
    )
    wan_cfg = load_omni_transfer_config(deploy_yaml)
    assert wan_cfg is not None, "wan2_2_eg.yaml transfer config resolved to None"
    assert ("0", "1") in wan_cfg.connectors, "Missing edge ('0', '1') in wan2_2_eg"
    assert wan_cfg.connectors[("0", "1")].name == "SharedMemoryConnector", (
        f"Expected SharedMemoryConnector, got {wan_cfg.connectors[('0', '1')].name}"
    )

    topology_dict = {
        "connectors": {
            "custom_shm": {
                "name": "SharedMemoryConnector",
                "extra": {"shm_threshold_bytes": 4096},
            },
        },
        "edges": [
            {"from": 0, "to": 1},
            {"from": 1, "to": 2},
        ],
        "stages": [
            {
                "stage_id": 0,
                "output_connectors": {"to_stage_1": "custom_shm"},
            },
            {
                "stage_id": 1,
                "input_connectors": {"from_stage_0": "custom_shm"},
            },
            {
                "stage_id": 2,
                "engine_input_source": [1],
            },
        ],
    }
    transfer_cfg = load_omni_transfer_config(config_dict=topology_dict)
    assert transfer_cfg is not None, "Topology dict resolved to None"

    edge_01 = transfer_cfg.connectors[("0", "1")]
    edge_12 = transfer_cfg.connectors[("1", "2")]
    assert edge_01.name == "SharedMemoryConnector", (
        f"Edge 0->1 connector mismatch: {edge_01.name}"
    )
    assert edge_01.extra.get("shm_threshold_bytes") == 4096, (
        f"Edge 0->1 extra lost shm_threshold_bytes: {edge_01.extra}"
    )
    assert edge_12.name == "SharedMemoryConnector", (
        f"Edge 1->2 fallback connector mismatch: {edge_12.name}"
    )
    scope_01 = edge_01.extra.get("wakeup_scope")
    scope_12 = edge_12.extra.get("wakeup_scope")
    assert isinstance(scope_01, str), (
        f"Edge 0->1 wakeup_scope expected str, got {type(scope_01).__name__}"
    )
    assert len(scope_01) > 0, (
        f"Edge 0->1 wakeup_scope expected non-empty str, got {scope_01!r}"
    )
    assert scope_01 == scope_12, (
        f"Shared wakeup_scope mismatch: {scope_01!r} != {scope_12!r}"
    )

    stage0_cfg = get_connectors_config_for_stage(transfer_cfg, 0)
    assert stage0_cfg["to_stage_1"]["spec"]["extra"]["role"] == "sender", (
        f"Stage 0 role mismatch: {stage0_cfg}"
    )
    stage1_cfg = get_connectors_config_for_stage(transfer_cfg, 1)
    assert stage1_cfg["from_stage_0"]["spec"]["extra"]["role"] == "receiver", (
        f"Stage 1 role mismatch: {stage1_cfg}"
    )
    assert "to_stage_2" not in stage1_cfg, (
        f"Non-stage-0 outgoing edge should not be injected: {stage1_cfg}"
    )

    mismatch_dict = {
        "stages": [
            {
                "stage_id": 0,
                "output_connectors": {
                    "to_stage_1": {"name": "SharedMemoryConnector"},
                },
            },
            {
                "stage_id": 1,
                "input_connectors": {
                    "from_stage_0": {"name": "NixlConnector"},
                },
            },
        ],
    }
    with pytest.raises(ValueError, match="Connector type mismatch"):
        load_omni_transfer_config(config_dict=mismatch_dict)


def test_connector_zmq_control_plane_side_channel() -> None:
    """Connector ZMQ Control-Plane Side-Channel.

    - TPU path: Compute deterministic ZMQ ports and resolve ``NixlConnector``
      specs across purposes, TP ranks, replicas, explicit ``sender_zmq_port``,
      and middle-stage outgoing configs, then run a localhost ZMQ ROUTER/REQ
      handshake.
    - Setup check: Assert ``OmniTpuPlatform`` is active, producer
      ``_serving_handshake`` is ``True`` on a bound port, and receiver
      ``_zmq_port`` is ``None`` (``_serving_handshake`` is ``False``).
    - Outcome check: Verify port offsets and verify ``_resolve_metadata``
      acquires a generation-scoped claim and ``_notify_transfer_done`` clears
      producer pending state.
    """
    assert isinstance(current_omni_platform, OmniTpuPlatform), (
        f"Expected OmniTpuPlatform, got {type(current_omni_platform).__name__}"
    )

    base = 50051
    offsets = {
        "request_forwarding": REQUEST_FORWARDING_PORT_OFFSET,
        "kv_transfer": KV_TRANSFER_PORT_OFFSET,
        "orchestrator": ORCHESTRATOR_PORT_OFFSET,
    }
    rank_replica_offset = 1 * KV_REPLICA_PORT_STRIDE + 3 * KV_RANK_PORT_STRIDE + 2
    ports = {
        purpose: compute_connector_zmq_port(
            base,
            purpose=purpose,  # type: ignore[arg-type]
            from_stage=2,
            local_rank=3,
            replica_id=1,
        )
        for purpose in offsets
    }
    expected_ports = {
        purpose: base + offset + rank_replica_offset
        for purpose, offset in offsets.items()
    }
    assert ports == expected_ports, (
        f"ZMQ port per purpose {ports} != expected {expected_ports}"
    )

    explicit_spec = ConnectorSpec(
        name="NixlConnector",
        extra={"sender_zmq_port": 55000, "from_stage": 0},
    )
    resolved_explicit = resolve_connector_spec(
        explicit_spec, stage_id=1, role="receiver", replica_id=3, local_rank=2
    )
    assert resolved_explicit.extra["sender_zmq_port"] == 55000, (
        f"Explicit sender_zmq_port overridden: {resolved_explicit.extra}"
    )

    middle_spec = ConnectorSpec(
        name="NixlConnector",
        extra={
            "host": "127.0.0.1",
            "zmq_port": 51000,
            "from_stage": 0,
            "outgoing": {"host": "127.0.0.1", "zmq_port": 52000, "from_stage": 1},
        },
    )
    resolved_middle = resolve_connector_spec(
        middle_spec, stage_id=1, role="receiver", purpose="request_forwarding"
    )
    assert resolved_middle.extra["sender_zmq_port"] == 51000, (
        f"Middle stage incoming port mismatch: {resolved_middle.extra}"
    )
    assert resolved_middle.extra["zmq_port"] == 52001, (
        f"Middle stage outgoing port mismatch: {resolved_middle.extra}"
    )

    producer = _make_handshake_nixl_connector(
        {"role": "sender", "host": "127.0.0.1", "zmq_port": 0}
    )
    consumer: NixlConnector | None = None
    try:
        assert producer._serving_handshake is True, (
            "Producer _serving_handshake should be True"
        )
        assert isinstance(producer._zmq_port, int), (
            f"Producer _zmq_port expected int, got {type(producer._zmq_port).__name__}"
        )
        assert producer._zmq_port > 0, (
            f"Producer bound invalid port {producer._zmq_port}, expected > 0"
        )

        consumer = _make_handshake_nixl_connector(
            {
                "role": "receiver",
                "sender_host": "127.0.0.1",
                "sender_zmq_port": producer._zmq_port,
            }
        )
        assert consumer._zmq_port is None, (
            f"Receiver _zmq_port should be None, got {consumer._zmq_port}"
        )
        assert consumer._serving_handshake is False, (
            "Receiver _serving_handshake should be False"
        )

        key = "req-zmq-0"
        pending = _PendingPayload(
            tensors=[],
            registrations=[],
            deadline=time.monotonic() + 30.0,
        )
        published = {
            "schema_version": 1,
            "generation": pending.generation,
            "sender_host": "127.0.0.1",
            "sender_zmq_port": producer._zmq_port,
            "size": 16,
        }
        with producer._state_lock:
            producer._pending[key] = pending
            producer._published[key] = published

        resolved_meta = consumer._resolve_metadata(key, None)
        assert resolved_meta is not None, "Consumer failed to resolve metadata"
        assert resolved_meta["generation"] == pending.generation, (
            "Generation mismatch: "
            f"{resolved_meta['generation']} != {pending.generation}"
        )
        claim_id = resolved_meta["claim_id"]
        assert claim_id in pending.claims, (
            f"Claim {claim_id!r} not recorded on producer pending payload"
        )

        acked = consumer._notify_transfer_done(key, resolved_meta)
        assert acked is True, "Consumer _notify_transfer_done did not receive ACK"
        with producer._state_lock:
            remaining = key in producer._pending
        assert remaining is False, "Producer pending payload not released after ACK"
    finally:
        if consumer is not None:
            consumer._close_once()
        producer._close_once()
