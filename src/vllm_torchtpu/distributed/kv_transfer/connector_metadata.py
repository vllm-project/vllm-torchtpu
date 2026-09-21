# SPDX-License-Identifier: Apache-2.0
"""Scheduler->worker metadata contract shared by the TPU KV connectors.

These dataclasses are the connector-neutral request bookkeeping (what to
send, what to load) attached to scheduler output. They are used by both the
zmq/shm transport stack (zmq_shm_base) and the Raiden connector, so they
live here rather than inside either transport module.
"""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import \
    KVConnectorMetadata

ReqId = str


@dataclass
class SendMeta:
    uuid: int
    local_block_ids: list[int]
    expiration_time: float
    # Exact transfer extent for controller-planned PCP striping. Legacy send
    # paths leave this unset.
    num_tokens: int | None = None
    # Uniform-mamba-layout source state slots, one per mamba kv-cache group
    # ordinal (the block holding the final recurrent state); None for FA-only
    # Stage-3 models and all legacy paths.
    mamba_state_block_ids: list[int] | None = None


@dataclass
class LoadMeta:
    uuid: int
    local_block_ids: list[int]
    remote_block_ids: list[int]
    remote_host: str | list[str]
    remote_port: int | list[int]
    remote_side_channel_port: int | None = None
    # Whether the worker reports this load's completion to the scheduler as
    # finished_recving. False when this connector doesn't own the request's
    # load state (the request is not WAITING_FOR_REMOTE_KVS here, e.g. a full
    # local cache hit or delegation to another MultiConnector child), where a
    # report would trip the scheduler's assert or prematurely resume the
    # request.
    report_completion: bool = True


@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, LoadMeta] = field(default_factory=dict)
