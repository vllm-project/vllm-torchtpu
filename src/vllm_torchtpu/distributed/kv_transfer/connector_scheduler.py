# SPDX-License-Identifier: Apache-2.0
"""Scheduler halves of the TPU KV connectors.

``TPUConnectorScheduler`` is the scheduler-side half behind the zmq/shm
transport: it decides per request what to send or load and builds the
``TPUConnectorMetadata`` the workers act on. ``TPURaidenConnectorScheduler``
extends it for the opt-in Raiden backend and its Stage-3 resharding.
``TPUConnector`` instantiates one of the two when vLLM builds the connector
with ``KVConnectorRole.SCHEDULER``.
"""

import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Optional
from uuid import uuid4

from vllm.config import VllmConfig
from vllm.utils.math_utils import round_down
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import (
    LoadMeta, ReqId, SendMeta, TPUConnectorMetadata)
from vllm_torchtpu.distributed.kv_transfer.raiden_config import (
    _STAGE3_FINISH_DEDUP_LIMIT, _engine_is_pipeline, _pipeline_parallel_size,
    _prefix_aware_load_enabled, _prefix_aware_load_requested,
    _resolved_reshard_controller_address, _use_raiden_glm_admission,
    _use_raiden_stage3_transport)
from vllm_torchtpu.distributed.kv_transfer.stage3_types import _Stage3LoadMeta
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _select_committed_mamba_blocks(
    block_tables: list[list[int]],
    num_speculative_blocks: int,
) -> list[int]:
    """Select each Mamba group's last committed state checkpoint.

    The final ``K`` entries are speculative, so the committed checkpoint is
    the ``-(K + 1)`` entry for a group with speculative depth ``K``.
    """
    required_blocks = num_speculative_blocks + 1
    committed_blocks: list[int] = []
    for ordinal, block_ids in enumerate(block_tables):
        assert len(block_ids) >= required_blocks, (
            "Mamba block table cannot contain the committed checkpoint and "
            f"speculative checkpoints: ordinal={ordinal}, "
            f"blocks={len(block_ids)}, required={required_blocks}")
        committed_blocks.append(int(block_ids[-required_blocks]))
    return committed_blocks


class TPUConnectorScheduler:

    def __init__(self, vllm_config: "VllmConfig"):
        self.vllm_config = vllm_config
        self.config = vllm_config.kv_transfer_config
        self.is_producer = self.config.is_kv_producer

        self.block_size = vllm_config.cache_config.block_size

        # Populated by request_finished() on P.
        self.reqs_to_send: dict[ReqId, SendMeta] = {}
        # Populated by update_state_after_alloc() on D.
        self.reqs_to_load: dict[ReqId, LoadMeta] = {}

        self.kv_ip = dist_utils.get_kv_ips()
        # Get DP rank and TP size from config and stagger kv_port and side_channel_port
        dp_rank = vllm_config.parallel_config.data_parallel_rank if vllm_config.parallel_config else 0
        tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config.parallel_config else 1
        port_base = dist_utils.get_kv_ports()
        if isinstance(port_base, list):
            self.kv_port = [int(p) + dp_rank * tp_size for p in port_base]
        else:
            self.kv_port = int(port_base) + dp_rank * tp_size
        self.side_channel_port = int(
            dist_utils.get_side_channel_port()) + dp_rank
        logger.info(
            "TPUConnectorScheduler --> kv_ip=%s | kv_port=%s | side_channel_port=%s",
            self.kv_ip, self.kv_port, self.side_channel_port)

    def _maybe_truncate_for_mamba(self, request: "Request") -> None:
        """P-side: drop the last prompt token so prefill ships the mamba
        state h(N-1); D's recompute of token N then reproduces h(N) instead
        of advancing the recurrence a second time."""
        if request.num_prompt_tokens <= 1:
            return
        params = request.kv_transfer_params
        if params is not None and params.get("_p_side_truncated"):
            return
        if request.prompt_token_ids is None:
            return
        request.prompt_token_ids.pop()
        request._all_token_ids.pop()
        request.num_prompt_tokens -= 1
        if request.kv_transfer_params is None:
            request.kv_transfer_params = {}
        request.kv_transfer_params["_p_side_truncated"] = True

    def on_new_request(self, request: "Request") -> None:
        """Leaves the request as submitted: the ZMQ backend needs no
        admission-time normalization. The Raiden stage-3 subclass overrides
        this to truncate Mamba prompts."""
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        D workers use this to get the number of new tokens
        that can be loaded from remote P workers.
        No-op for P workers.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that will be loaded from the
                  external KV cache.
                - If async loading. Must be 'False' for TPU connector
                  because TPU pulls KV cache in a blocking way.

        """
        if (self.is_producer or not request.kv_transfer_params
                or request.kv_transfer_params.get("_remote_kv_processed")):
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        # Rounding must match request_finished()'s remote_block_ids computation.
        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        # The pull is blocking at the ZMQ layer, but we wrap it in a thread
        # pool so from the scheduler's perspective it's async.
        if count > 0:
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            local_block_ids = blocks.get_block_ids()[0]
            # D must pull the whole prefill blocks regardless of partial
            # prefix-cache hits, because the transport has no RDMA-style
            # partial-pull and P publishes the full payload under a uuid.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=params["remote_block_ids"],
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                remote_side_channel_port=params["remote_side_channel_port"]
                if "remote_side_channel_port" in params else None,
            )
        else:
            # Full prefix-cache hit or async pull done -- we still need to
            # notify P so it can free the pending buffer.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                remote_side_channel_port=params["remote_side_channel_port"]
                if "remote_side_channel_port" in params else None,
            )
        params["_remote_kv_processed"] = True
        logger.info(
            "TPUConnectorScheduler update_state_after_alloc --> reqs_to_load=%s",
            self.reqs_to_load)

    def build_connector_meta(self) -> TPUConnectorMetadata:
        meta = TPUConnectorMetadata()
        if self.is_producer:
            meta.reqs_to_send = self.reqs_to_send
            self.reqs_to_send = {}
        else:
            meta.reqs_to_load = self.reqs_to_load
            self.reqs_to_load = {}
        return meta

    def get_finished_count(self) -> int:
        return len(self.kv_ip) if isinstance(self.kv_ip, list) else 1

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not self.is_producer:
            return False, None

        # max_tokens is forced to 1 by the proxy on P, so the only way the
        # prefill finishes cleanly is length cap.
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Only transfer full blocks; let D re-prefill the trailing partial
        # block locally.
        all_full = request.num_computed_tokens % self.block_size == 0
        computed_block_ids = block_ids if all_full else block_ids[:-1]

        delay_free_blocks = len(computed_block_ids) > 0
        if delay_free_blocks:
            uuid = get_uuid()
            expiration_time = (time.perf_counter() +
                               dist_utils.get_p2p_wait_pull_timeout())
            self.reqs_to_send[request.request_id] = SendMeta(
                uuid=uuid,
                local_block_ids=computed_block_ids,
                expiration_time=expiration_time)
            kv_transfer_params = dict(
                uuid=uuid,
                remote_block_ids=computed_block_ids,
                remote_host=self.kv_ip,
                remote_port=self.kv_port,
                remote_side_channel_port=self.side_channel_port)
            logger.info(
                "TPUConnectorScheduler --> reqs_to_send=%s | kv_transfer_params=%s",
                self.reqs_to_send, kv_transfer_params)
        else:
            kv_transfer_params = {}

        return delay_free_blocks, kv_transfer_params


class TPURaidenConnectorScheduler(TPUConnectorScheduler):
    """Scheduler half for the opt-in Raiden transfer backend."""

    def __init__(self, vllm_config: "VllmConfig"):
        super().__init__(vllm_config)
        stage3_consumer = (not self.is_producer
                           and _use_raiden_stage3_transport())
        self._stage3_prefix_aware_load_enabled = (stage3_consumer and
                                                  _prefix_aware_load_enabled())
        if (stage3_consumer
                and bool(vllm_config.cache_config.enable_prefix_caching)
                and not self._stage3_prefix_aware_load_enabled):
            log = (logger.warning
                   if _prefix_aware_load_requested() else logger.info)
            log("Prefix-aware Stage-3 loads are unavailable; decode-local "
                "prefix hits are ignored and the full producer payload is "
                "pulled into every destination page")
        # req_id -> (uuid, global block ids, exact token count, wire params).
        # This survives build_connector_meta() so a duplicate request_finished
        # callback cannot mint a conflicting UUID or enqueue a second D5
        # registration.
        self._stage3_finished_sends: OrderedDict[str, tuple[int, tuple[
            int, ...], int, dict[str, Any]]] = OrderedDict()
        self._stage3_fa_group_index = 0
        self._stage3_mamba_group_indices: list[int] = []
        self._stage3_mamba_num_speculative_blocks: int = 0
        # Get DP rank and TP size from config and stagger kv_port and side_channel_port
        dp_rank = vllm_config.parallel_config.data_parallel_rank if vllm_config.parallel_config else 0
        tp_size = vllm_config.parallel_config.tensor_parallel_size if vllm_config.parallel_config else 1
        port_base = dist_utils.get_kv_ports()
        if isinstance(port_base, list):
            self.kv_port = [int(p) + 2 * dp_rank * tp_size for p in port_base]
        else:
            self.kv_port = int(port_base) + 2 * dp_rank * tp_size
        logger.info("TPURaidenConnectorScheduler --> kv_ip=%s | kv_port=%s",
                    self.kv_ip, self.kv_port)

    def on_new_request(self, request: "Request") -> None:
        """Truncates a P-side Mamba prompt at admission, ahead of the
        scheduler's prefix-cache lookup. The lookup caps its hit at
        num_tokens - 1 of whatever length it sees, so shortening the prompt
        any later can leave zero new tokens to schedule."""
        if (self.is_producer and _use_raiden_stage3_transport()
                and self._stage3_mamba_group_indices):
            self._maybe_truncate_for_mamba(request)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self.is_producer:
            return 0, False
        if (not request.kv_transfer_params
                or request.kv_transfer_params.get("_remote_kv_processed")):
            return 0, False

        assert num_computed_tokens % self.block_size == 0
        if _use_raiden_stage3_transport():
            # Unlike the legacy equal-page pull, the controller planner has a
            # byte-precise partial-tail entry. The producer's scheduler-owned
            # num_computed_tokens is the transfer extent: in the deployed
            # proxy's one-token-prefill flow this can be prompt_tokens - 1.
            # Do not replace it with prompt length here.
            transfer_tokens = int(
                request.kv_transfer_params.get("num_tokens", 0))
            if transfer_tokens <= 0:
                raise ValueError(
                    "Stage-3 metadata requires a positive num_tokens")
            count = max(transfer_tokens - num_computed_tokens, 0)
            if count > 0 and dist_utils.get_raiden_inline_load():
                return count, False
            return count, count > 0

        rounded_num_prompt_tokens = round_down(len(request.prompt_token_ids),
                                               self.block_size)
        count = max(rounded_num_prompt_tokens - num_computed_tokens, 0)
        if count > 0:
            if dist_utils.get_raiden_inline_load():
                total_external_tokens = num_computed_tokens + count
                if total_external_tokens >= len(request.prompt_token_ids):
                    count = max(count - 1, 0)
                logger.info(
                    "TPURaidenConnectorScheduler inline load req_id=%s "
                    "external_tokens=%d", request.request_id, count)
                return count, False
            return count, True
        return 0, False

    def update_state_after_alloc(self, request: "Request",
                                 blocks: "KVCacheBlocks",
                                 num_external_tokens: int):
        if self.is_producer or not request.kv_transfer_params:
            return

        if _use_raiden_stage3_transport():
            self._update_stage3_state_after_alloc(request, blocks,
                                                  num_external_tokens)
            return

        params = request.kv_transfer_params
        if num_external_tokens > 0:
            # D must pull the WHOLE prefill payload: P publishes it under one
            # uuid with no partial-pull, so load into every allocated block.
            local_block_ids = blocks.get_block_ids()[0]
            if not local_block_ids:
                self.reqs_to_load[request.request_id] = LoadMeta(
                    uuid=params["uuid"],
                    local_block_ids=None,
                    remote_block_ids=None,
                    remote_host=params["remote_host"],
                    remote_port=params["remote_port"],
                )
                params["_remote_kv_processed"] = True
                logger.info(
                    "TPURaidenConnectorScheduler prefix hit req_id=%s "
                    "releases remote send uuid=%s", request.request_id,
                    params["uuid"])
                return

            remote_block_ids = params["remote_block_ids"]
            if len(local_block_ids) > len(remote_block_ids):
                raise ValueError(
                    "TPURaidenConnector cannot pull more local blocks than "
                    "the producer published: "
                    f"local={len(local_block_ids)} remote="
                    f"{len(remote_block_ids)}")
            remote_block_ids = remote_block_ids[-len(local_block_ids):]
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
            )
        else:
            # If num_external_tokens is 0, we don't need to pull any data through
            # this connector (either due to a full local cache hit, or because
            # another MultiConnector child owns the load). But the producer
            # (prefill node) doesn't know that: it has the prefill KV blocks
            # registered under this uuid and keeps them pinned until they are
            # pulled or p2p_wait_pull_timeout expires. So we still enqueue an
            # empty read (LoadMeta with no block ids), which the worker turns
            # into a "release now" signal to the producer instead of a pull.
            # TODO: Remove this branch once the producer releases
            # unpulled sends by another path (e.g. a direct scheduler-side
            # release, or the unified Raiden connector); otherwise its blocks
            # stay pinned until p2p_wait_pull_timeout.
            self.reqs_to_load[request.request_id] = LoadMeta(
                uuid=params["uuid"],
                local_block_ids=None,
                remote_block_ids=None,
                remote_host=params["remote_host"],
                remote_port=params["remote_port"],
                report_completion=False,
            )
            logger.info(
                "TPURaidenConnectorScheduler no-load release req_id=%s "
                "uuid=%s", request.request_id, params["uuid"])
        params["_remote_kv_processed"] = True
        logger.info(
            "TPURaidenConnectorScheduler update_state_after_alloc --> "
            "reqs_to_load=%s", self.reqs_to_load)

    def _update_stage3_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Declares a destination transfer without source physical IDs."""
        if num_external_tokens <= 0:
            # First admission with a full local prefix hit: nothing to pull,
            # but the producer holds a request-block registration pinned until
            # unless released. request.num_computed_tokens is still 0 here
            # only on first admission — the post-async-load resume re-enters
            # with the loaded count set, and must not re-release.
            if (self._stage3_prefix_aware_load_enabled
                    and int(request.num_computed_tokens) == 0):
                self._enqueue_stage3_release(request)
                if request.kv_transfer_params is not None:
                    request.kv_transfer_params["_remote_kv_processed"] = True
            return
        params = request.kv_transfer_params
        assert params is not None
        forbidden = {"remote_block_ids", "remote_host", "remote_port"}
        leaked = sorted(forbidden.intersection(params))
        if leaked:
            raise ValueError(
                "Stage-3 cross-server metadata must not carry legacy "
                f"source routing/block fields: {leaked}")

        source_req_id = params.get("req_id")
        if not isinstance(source_req_id, str) or not source_req_id:
            raise ValueError(
                "Stage-3 metadata requires a non-empty source request ID")
        uuid = int(params.get("uuid", 0))
        num_tokens = int(params.get("num_tokens", 0))
        src_controller_address = str(params.get("src_controller_address",
                                                "")).strip()
        src_job_name = str(params.get("src_job_name", "")).strip()
        src_engine_id = str(params.get("src_engine_id", "")).strip()
        src_data_replica_idx = int(params.get("src_data_replica_idx", -1))
        src_parallelism = int(params.get("src_parallelism", 0))
        if uuid <= 0:
            raise ValueError("Stage-3 request UUID must be positive")
        if num_tokens <= 0:
            raise ValueError("Stage-3 transfer token count must be positive")
        if not src_controller_address:
            raise ValueError(
                "Stage-3 metadata requires an explicit source controller "
                "address")
        if not src_job_name or not src_engine_id:
            raise ValueError(
                "Stage-3 metadata requires explicit source job and engine "
                "identity")
        if src_data_replica_idx < 0:
            raise ValueError(
                "Stage-3 source data replica identity must be non-negative")
        if src_parallelism <= 0:
            raise ValueError(
                "Stage-3 source transfer parallelism must be positive")

        grouped_block_ids = blocks.get_block_ids()
        if self._stage3_fa_group_index >= len(grouped_block_ids):
            raise ValueError(
                "Stage-3 FA KV cache group is absent from decode allocation: "
                f"index={self._stage3_fa_group_index}, "
                f"groups={len(grouped_block_ids)}")
        local_block_ids = list(grouped_block_ids[self._stage3_fa_group_index])
        expected_pages = ((num_tokens + self.block_size - 1) //
                          self.block_size)
        if len(local_block_ids) != expected_pages:
            raise ValueError(
                "Stage-3 FA resharding requires the complete destination "
                "page set (prefix-suffix pulls are unsupported): "
                f"blocks={len(local_block_ids)}, expected={expected_pages}, "
                f"num_tokens={num_tokens}, page_tokens={self.block_size}")
        external_tokens = int(num_external_tokens)
        if external_tokens > num_tokens:
            raise ValueError(
                "External token count exceeds the published reshard payload: "
                f"external={external_tokens}, num_tokens={num_tokens}")
        skip_tokens = (num_tokens - external_tokens
                       if self._stage3_prefix_aware_load_enabled else 0)
        if skip_tokens:
            # Partial local prefix hit: pull only the suffix, and only into
            # the trailing (newly allocated) pages — the leading adopted
            # cache pages are shared and must never be transfer targets.
            if skip_tokens % self.block_size != 0:
                raise ValueError(
                    "Prefix hits must be destination-page aligned: "
                    f"skip_tokens={skip_tokens}, "
                    f"page_tokens={self.block_size}")
            prefix_pages = skip_tokens // self.block_size
            suffix_pages = ((external_tokens + self.block_size - 1) //
                            self.block_size)
            if prefix_pages + suffix_pages != expected_pages:
                raise ValueError("Suffix page arithmetic is inconsistent: "
                                 f"prefix_pages={prefix_pages}, "
                                 f"suffix_pages={suffix_pages}, "
                                 f"expected_pages={expected_pages}")
            local_block_ids = local_block_ids[prefix_pages:]
        mamba_state_block_ids: Optional[list[int]] = None
        if self._stage3_mamba_group_indices:
            mamba_block_ids: list[list[int]] = []
            for mamba_gid in self._stage3_mamba_group_indices:
                if mamba_gid >= len(grouped_block_ids):
                    raise ValueError(
                        "GDN state reshard: mamba KV cache group is absent "
                        f"from decode allocation: index={mamba_gid}, "
                        f"groups={len(grouped_block_ids)}")
                mamba_block_ids.append(list(grouped_block_ids[mamba_gid]))
            mamba_state_block_ids = _select_committed_mamba_blocks(
                mamba_block_ids, self._stage3_mamba_num_speculative_blocks)
        self.reqs_to_load[request.request_id] = _Stage3LoadMeta(
            uuid=uuid,
            source_req_id=source_req_id,
            local_block_ids=local_block_ids,
            num_tokens=num_tokens,
            src_controller_address=src_controller_address,
            src_job_name=src_job_name,
            src_engine_id=src_engine_id,
            src_data_replica_idx=src_data_replica_idx,
            src_parallelism=src_parallelism,
            mamba_state_block_ids=mamba_state_block_ids,
            skip_tokens=skip_tokens,
        )
        params["_remote_kv_processed"] = True
        logger.info(
            "TPURaidenConnectorScheduler Stage-3 load req_id=%s "
            "source_req_id=%s uuid=%d num_tokens=%d skip_tokens=%d "
            "destination_pages=%d source_controller=%s",
            request.request_id,
            source_req_id,
            uuid,
            num_tokens,
            skip_tokens,
            len(local_block_ids),
            src_controller_address,
        )

    def _raiden_engine_id(self) -> str:
        env_id = str(tpu_envs.TPU_RAIDEN_ENGINE_ID).strip()
        if env_id:
            return env_id
        config_id = getattr(
            getattr(self.vllm_config, "kv_transfer_config", None),
            "engine_id",
            None,
        )
        if config_id:
            return str(config_id).strip()
        default_job = "prefill" if self.is_producer else "decode"
        return f"{default_job}-engine"

    def _enqueue_stage3_release(self, request: "Request") -> None:
        """Full local hit: nothing to pull; ask the worker to promptly
        release the producer's request-block registration instead of leaving
        until p2p_wait_pull_timeout. Best-effort: the TTL remains the
        backstop, so malformed params degrade to a warning, not a failure."""
        params = request.kv_transfer_params or {}
        source_req_id = params.get("req_id")
        uuid = int(params.get("uuid", 0) or 0)
        src_controller_address = str(params.get("src_controller_address",
                                                "")).strip()
        if (not isinstance(source_req_id, str) or not source_req_id
                or uuid <= 0 or not src_controller_address):
            logger.warning(
                "Full prefix hit for req_id=%s carries incomplete "
                "source metadata; leaving producer release to the TTL",
                request.request_id)
            return
        self.reqs_to_load[request.request_id] = _Stage3LoadMeta(
            uuid=uuid,
            source_req_id=source_req_id,
            local_block_ids=[],
            num_tokens=int(params.get("num_tokens", 0) or 0),
            src_controller_address=src_controller_address,
            src_job_name=str(params.get("src_job_name", "")),
            src_engine_id=str(params.get("src_engine_id", "")),
            src_data_replica_idx=int(params.get("src_data_replica_idx", -1)),
            src_parallelism=int(params.get("src_parallelism", 0) or 0),
            release_only=True,
        )
        logger.info(
            "TPURaidenConnectorScheduler full prefix hit req_id=%s "
            "source_req_id=%s uuid=%d releases the producer registration",
            request.request_id, source_req_id, uuid)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
        mamba_block_ids: Optional[list[list[int]]] = None,
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not _use_raiden_stage3_transport():
            return super().request_finished(request, block_ids)
        if not self.is_producer:
            return False, None
        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            return False, None

        # Decode deliberately computes the final prompt token locally to
        # produce its first logits. Publish no more than the preceding prompt
        # prefix even when vLLM reports the producer's full prompt as computed.
        # The controller emits a short final chunk, so retain that partial
        # source page instead of inheriting the legacy full-block-only trim.
        computed_tokens = int(request.num_computed_tokens)
        prompt_tokens = int(request.num_prompt_tokens)
        finish_params = request.kv_transfer_params
        if isinstance(finish_params, dict) and bool(
                finish_params.get("_p_side_truncated")):
            # The producer already dropped one token; N-1 is already applied.
            num_tokens = min(computed_tokens, prompt_tokens)
        else:
            num_tokens = min(computed_tokens, prompt_tokens - 1)
        if num_tokens <= 0 or not block_ids:
            return False, {}
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        if not _use_raiden_glm_admission() and pcp_size > 1:
            # The interleave minimum applies to PCP sources only.
            interleave_size = int(parallel_config.cp_kv_cache_interleave_size)
            min_transfer_tokens = pcp_size * interleave_size
            if (num_tokens < min_transfer_tokens):
                return False, {}

        max_transfer_tokens = tpu_envs.TPU_RAIDEN_MAX_TRANSFER_TOKENS
        if max_transfer_tokens is not None and max_transfer_tokens > 0:
            if num_tokens > max_transfer_tokens:
                logger.warning(
                    "Stage-3 producer token count (%d) exceeds "
                    "TPU_RAIDEN_MAX_TRANSFER_TOKENS (%d) for req_id=%s; "
                    "skipping transfer and freeing blocks immediately.",
                    num_tokens, max_transfer_tokens, request.request_id)
                return False, {}

        scheduler_block_tokens = self.block_size * pcp_size
        expected_scheduler_blocks = (
            (num_tokens + scheduler_block_tokens - 1) //
            scheduler_block_tokens)
        normalized_block_ids = tuple(int(block_id) for block_id in block_ids)
        # The producer may allocate trailing blocks for its excluded final
        # prompt token or locally generated tokens. They are outside the
        # transfer prefix; too few blocks remains an error.
        if len(normalized_block_ids) > expected_scheduler_blocks:
            normalized_block_ids = (
                normalized_block_ids[:expected_scheduler_blocks])
        if len(normalized_block_ids) != expected_scheduler_blocks:
            raise ValueError(
                "Stage-3 producer block IDs must cover every PCP scheduler "
                "block, including the partial tail: "
                f"blocks={len(block_ids)}, transfer_blocks="
                f"{len(normalized_block_ids)}, expected="
                f"{expected_scheduler_blocks}, num_tokens={num_tokens}, "
                f"page_tokens={self.block_size}, pcp_size={pcp_size}")

        producer_dp_rank = (self.vllm_config.parallel_config.data_parallel_rank
                            if self.vllm_config.parallel_config else 0)
        controller_address = _resolved_reshard_controller_address(
            producer_dp_rank)
        if not controller_address:
            raise ValueError("Stage-3 producer metadata requires "
                             "TPU_RAIDEN_CONTROLLER_ADDRESS")
        normalized_ids = normalized_block_ids
        src_job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or "prefill"
        src_engine_id = self._raiden_engine_id()
        src_parallelism = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        parallel_config = self.vllm_config.parallel_config
        src_data_replica_idx = int(parallel_config.data_parallel_rank or 0)
        if not src_engine_id:
            raise ValueError("TPU_RAIDEN_ENGINE_ID must not be empty")
        if src_parallelism <= 0:
            raise ValueError(
                "TPU_RAIDEN_TRANSFER_PARALLELISM must be positive")
        if _use_raiden_glm_admission():
            tp_size = int(parallel_config.tensor_parallel_size or 1)
            if src_parallelism != tp_size:
                raise ValueError(
                    "Stage-3 GLM producer transfer parallelism must equal "
                    f"TP size: parallelism={src_parallelism}, "
                    f"tp_size={tp_size}")
        elif _engine_is_pipeline(self.vllm_config):
            pp_size = _pipeline_parallel_size(self.vllm_config)
            if src_parallelism != pp_size:
                raise ValueError(
                    "Stage-3 pipeline producer transfer parallelism must "
                    f"equal PP size: parallelism={src_parallelism}, "
                    f"pp_size={pp_size}")
        elif src_parallelism != pcp_size:
            raise ValueError(
                "Stage-3 producer transfer parallelism must equal PCP size: "
                f"parallelism={src_parallelism}, pcp_size={pcp_size}")
        if src_data_replica_idx < 0:
            raise ValueError(
                "Producer data-parallel rank must be non-negative")
        existing = self._stage3_finished_sends.get(request.request_id)
        if existing is not None:
            uuid, old_ids, old_num_tokens, params = existing
            if old_ids != normalized_ids or old_num_tokens != num_tokens:
                raise ValueError(
                    "Conflicting duplicate Stage-3 request finish for "
                    f"req_id={request.request_id}")
            self._stage3_finished_sends.move_to_end(request.request_id)
            return True, dict(params)

        # Never evict an in-flight UUID merely to satisfy a memory bound: a
        # delayed duplicate callback could otherwise mint a conflicting UUID
        # while native/D5 state still exists. Scheduler completion removes
        # records through update_connector_output(). Refuse excess concurrency
        # rather than weakening idempotency.
        if len(self._stage3_finished_sends) >= _STAGE3_FINISH_DEDUP_LIMIT:
            raise RuntimeError(
                "Stage-3 in-flight finish dedup capacity exhausted: "
                f"limit={_STAGE3_FINISH_DEDUP_LIMIT}")

        mamba_state_block_ids: Optional[list[int]] = None
        if self._stage3_mamba_group_indices:
            if not mamba_block_ids:
                raise ValueError("GDN state is present but the producer's "
                                 "mamba block tables were not provided")
            mamba_state_block_ids = _select_committed_mamba_blocks(
                mamba_block_ids, self._stage3_mamba_num_speculative_blocks)
        uuid = get_uuid()
        now = time.perf_counter()
        expiration_time = (now + dist_utils.get_p2p_wait_pull_timeout())
        self.reqs_to_send[request.request_id] = SendMeta(
            uuid=uuid,
            local_block_ids=list(normalized_ids),
            expiration_time=expiration_time,
            num_tokens=num_tokens,
            mamba_state_block_ids=mamba_state_block_ids,
        )
        params: dict[str, Any] = {
            "req_id": request.request_id,
            "uuid": uuid,
            "num_tokens": num_tokens,
            "src_controller_address": controller_address,
            "src_job_name": src_job_name,
            "src_engine_id": src_engine_id,
            "src_data_replica_idx": src_data_replica_idx,
            "src_parallelism": src_parallelism,
        }
        self._stage3_finished_sends[request.request_id] = (
            uuid,
            normalized_ids,
            num_tokens,
            dict(params),
        )
        logger.info(
            "TPURaidenConnectorScheduler Stage-3 send req_id=%s uuid=%d "
            "num_tokens=%d source_pages=%d source_controller=%s",
            request.request_id,
            uuid,
            num_tokens,
            len(normalized_ids),
            controller_address,
        )
        return True, params

    def update_connector_output(self, connector_output: Any) -> None:
        for req_id in connector_output.finished_sending or ():
            self._stage3_finished_sends.pop(req_id, None)

    def get_finished_count(self) -> int:
        if not _use_raiden_stage3_transport():
            # Preserve the legacy connector's model-runner-world aggregation.
            return 0
        if not self.is_producer:
            # DP8 routes one request to one decode engine; a pipeline decode
            # engine loads each stage's layers on that stage.
            return _pipeline_parallel_size(self.vllm_config)
        parallel_config = self.vllm_config.parallel_config
        if _use_raiden_glm_admission():
            return int(parallel_config.tensor_parallel_size or 1)
        if _engine_is_pipeline(self.vllm_config):
            return _pipeline_parallel_size(self.vllm_config)
        return int(parallel_config.prefill_context_parallel_size or 1)


def get_uuid() -> int:
    int128 = uuid4().int
    # Stay under 64-bit so JSON-encoded responses through the proxy are safe.
    return int128 >> 78
