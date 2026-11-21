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
"""Scheduler->worker metadata contract of the Raiden Stage-3 P/D connector.

These dataclasses are the Stage-3 request bookkeeping (what to send, what to
load) that TPURaidenConnectorScheduler attaches to scheduler output and
TPURaidenConnectorWorker consumes.
"""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

ReqId = str


@dataclass
class SendMeta:
    uuid: int
    local_block_ids: list[int]
    # Unix timestamp in seconds, compared by workers on different hosts.
    expiration_time: float
    # Exact transfer extent for controller-planned PCP striping.
    num_tokens: int | None = None
    # Uniform-mamba-layout source state slots, one per mamba kv-cache group
    # ordinal (the block holding the final recurrent state); None for FA-only
    # Stage-3 models.
    mamba_state_block_ids: list[int] | None = None
    # One block table per transferred kv-cache group, in Stage-3 transfer-plan
    # order. Only DSv4 sets it, because its groups page at different sizes;
    # single-group models leave it None and use local_block_ids.
    group_block_ids: list[list[int]] | None = None
    # The prompt token the producer dropped for the consumer to compute; a
    # prefill-only drafter seeds its boundary row with it.
    next_prompt_token_id: int | None = None
    # Published in the step that completes the prompt, ahead of the finish:
    # the worker registration starts finish_pending (no TTL, no status
    # probe) until finished_req_ids names the request.
    registered_ahead: bool = False


@dataclass
class LoadMeta:
    """Decode-local declaration for one controller-driven FA transfer.

    Source physical block IDs deliberately do not appear here. They stay in
    the source controller's D5 registry; the only source-side routing datum
    crossing the P/D boundary is its controller address.
    """

    uuid: int
    source_req_id: str
    local_block_ids: list[int]
    num_tokens: int
    src_controller_address: str
    src_job_name: str
    src_engine_id: str
    src_data_replica_idx: int
    src_parallelism: int
    # Uniform-mamba-layout destination state slots, one per mamba kv-cache
    # group ordinal: the block holding the recurrent state for the resumed
    # request (state follows each group's block table; None for FA-only
    # models).
    mamba_state_block_ids: list[int] | None = None
    # One destination block table per transferred kv-cache group, in the
    # Stage-3 transfer-group plan's order. See SendMeta.group_block_ids.
    group_block_ids: list[list[int]] | None = None
    # Prefix-aware load: tokens satisfied by the decode-local prefix cache
    # and excluded from the transfer. local_block_ids then holds only the
    # suffix pages; the source store clips its plan at
    # skip_tokens * token_bytes of the FA global destination byte space.
    skip_tokens: int = 0
    # Full local hit: nothing to transfer; the worker only releases the
    # producer's request-block registration instead of pulling.
    release_only: bool = False
    # When True on a release_only entry (e.g. a consumer request aborted while
    # queued in reqs_to_load before reaching the worker), also record the
    # request in _done_recving so get_finished() surfaces it and the vLLM
    # scheduler frees the request's delayed local KV blocks.
    report_completion: bool = False
    # The scheduler rejected this load after allocation.
    fail_only: bool = False


@dataclass
class TPUConnectorMetadata(KVConnectorMetadata):
    reqs_to_send: dict[ReqId, SendMeta] = field(default_factory=dict)
    reqs_to_load: dict[ReqId, LoadMeta] = field(default_factory=dict)
    # Producer request id -> transfer uuid whose ahead-of-finish request-block
    # registration every pipeline stage retires.
    reqs_to_release: dict[ReqId, int] = field(default_factory=dict)
