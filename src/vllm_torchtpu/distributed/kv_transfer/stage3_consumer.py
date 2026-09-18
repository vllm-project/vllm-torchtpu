# SPDX-License-Identifier: Apache-2.0
"""Consumer (decode) side of Stage-3 Raiden KV loads.

``_Stage3ConsumerMixin`` holds the ``TPURaidenConnectorWorker`` methods that
turn each ``_Stage3LoadMeta`` into exactly one source-controller transfer:
staging the coordination call, issuing it from a small pool of submit
threads, applying the outcome to the scheduler-visible tables, re-attempting
parked loads once per scheduler step, binding the controller's request
identity to the local scheduler ID, and releasing the producer's
registration on a full local cache hit.
"""

import importlib
import json
import threading
import time
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import \
    TPUConnectorMetadata
from vllm_torchtpu.distributed.kv_transfer.raiden_config import (
    _STAGE3_D5_REGISTRATION_WAIT_S, _STAGE3_GLM_TRANSFER_POOL_TAGS,
    _STAGE3_REGISTRATION_REATTEMPT_MIN_INTERVAL_S, _STAGE3_SUBMIT_WORKERS,
    _engine_is_pipeline, _pipeline_parallel_size, _use_raiden_glm_admission,
    stage3_fa_raiden_id_fields)
from vllm_torchtpu.distributed.kv_transfer.stage3_types import (
    _Stage3LoadMeta, _Stage3PendingSubmit, _Stage3SubmitOutcome,
    _Stage3SubmitTask)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class _Stage3ConsumerMixin:
    """Decode-side Stage-3 load submission for ``TPURaidenConnectorWorker``.

    Mixin: no state of its own. Every attribute it reads is set in
    ``TPURaidenConnectorWorker.__init__``, and the geometry helpers it calls
    (``_stage3_fa_token_bytes`` and friends) live on the worker's other
    mixins.
    """

    @staticmethod
    def _raiden_hbm_memory_type() -> Any:
        rpc = importlib.import_module("tpu_sync.rpc.raiden_controller")
        return rpc.RaidenMemoryType.HBM

    def _stage3_source_work_units(self,
                                  req_meta: _Stage3LoadMeta) -> list[Any]:
        """Enumerates the producer units this destination pulls from: the
        complete rank set. A pipeline decode stage names them all too (the
        planner keeps the prefill stage registering its layers, which needs
        equal stage counts on both sides); the coordinator requires the
        source ranks to be contiguous from zero."""
        parallelism = int(req_meta.src_parallelism)
        expected = int(tpu_envs.TPU_RAIDEN_TRANSFER_PARALLELISM)
        if parallelism != expected:
            raise ValueError(
                "Stage-3 source parallelism does not match destination "
                f"configuration: source={parallelism}, destination={expected}")
        ranks = range(parallelism)
        if _engine_is_pipeline(self.vllm_config):
            pp_size = _pipeline_parallel_size(self.vllm_config)
            if parallelism != pp_size:
                raise ValueError(
                    "A pipeline decode stage pulls from the prefill stage "
                    "with the same layers, so both engines need the same "
                    f"stage count: source={parallelism}, decode={pp_size}")
        return [
            self._new_raiden_id(
                stage3_fa_raiden_id_fields(
                    job_name=req_meta.src_job_name,
                    engine_id=req_meta.src_engine_id,
                    dp_rank=req_meta.src_data_replica_idx,
                    transfer_rank=rank,
                    is_producer=True,
                )) for rank in ranks
        ]

    def _record_stage3_load_failure(self, req_id: str,
                                    block_ids: list[int]) -> None:
        self._failed_recving.add(req_id)
        self._failed_block_ids.update(block_ids)
        self._stage3_terminal_loads.add(req_id)

    @staticmethod
    def _start_stage3_transfer_with_d5_retry(facade: Any,
                                             **kwargs: Any) -> bool:
        """Bridges the one-step P scheduler-to-worker registration window.

        vLLM returns producer transfer parameters from update_from_output(),
        while the D5 block registrations reach producer workers in the next
        scheduler step. The source controller fails before receiver arming
        when those registrations are incomplete, so retry only that precise,
        side-effect-free error for a short bounded interval.
        """
        configured_timeout = float(dist_utils.get_p2p_wait_pull_timeout())
        deadline = time.perf_counter() + min(max(configured_timeout, 0.0),
                                             _STAGE3_D5_REGISTRATION_WAIT_S)
        attempts = 0
        while True:
            attempts += 1
            try:
                return facade.start_transfer(**kwargs)
            except RuntimeError as exc:
                if ("Missing producer block registration" not in str(exc)
                        or time.perf_counter() >= deadline):
                    raise
                if attempts == 1:
                    logger.info(
                        "Stage-3 waiting for producer D5 registrations "
                        "req_id=%s uuid=%s", kwargs.get("req_id"),
                        kwargs.get("uuid"))
                time.sleep(0.01)

    def _bind_stage3_request_ids(self, destination_req_id: str,
                                 source_req_id: str) -> None:
        """Binds the native/controller identity to the local scheduler ID."""
        existing_source = self._stage3_source_req_ids.get(destination_req_id)
        if existing_source is not None and existing_source != source_req_id:
            raise ValueError(
                "Conflicting Stage-3 source request ID for destination "
                f"request {destination_req_id!r}: existing="
                f"{existing_source!r}, new={source_req_id!r}")
        existing_destination = self._stage3_destination_req_ids.get(
            source_req_id)
        if (existing_destination is not None
                and existing_destination != destination_req_id):
            raise ValueError(
                "Stage-3 source request ID is already bound to a different "
                f"destination request: source={source_req_id!r}, existing="
                f"{existing_destination!r}, new={destination_req_id!r}")
        self._stage3_source_req_ids[destination_req_id] = source_req_id
        self._stage3_destination_req_ids[source_req_id] = destination_req_id

    def _stage3_source_request_id(self, destination_req_id: str) -> str:
        source_req_id = self._stage3_source_req_ids.get(destination_req_id)
        if source_req_id is None:
            raise RuntimeError(
                "Stage-3 destination request has no source identity binding: "
                f"destination={destination_req_id!r}")
        return source_req_id

    def _stage3_destination_request_id(self,
                                       source_req_id: str) -> Optional[str]:
        # Native completion IDs are controller plan IDs. Never treat an
        # unknown ID as a decode-local ID: it may be a stale terminal record
        # that happens to collide with an active destination request.
        return self._stage3_destination_req_ids.get(source_req_id)

    def _stage3_release_producer_registration(
            self,
            destination_req_id: str,
            req_meta: _Stage3LoadMeta,
            synchronous: bool = False) -> None:
        """Full local hit: cancel the producer's unclaimed request-block
        registration.

        cancel_request_blocks_if_unclaimed is claim-safe — a concurrently
        claimed (in-flight) registration refuses cancellation — and the TTL
        remains the backstop, so this is best-effort fire-and-forget and
        rides the submit workers instead of the model-runner thread."""
        address = str(req_meta.src_controller_address).strip()
        facade = self._stage3_source_facades.get(address)
        if facade is None:
            facade = self._new_raiden_controller_facade(address)
            self._stage3_source_facades[address] = facade
        source_req_id = req_meta.source_req_id
        uuid = int(req_meta.uuid)

        def _release() -> None:
            cancelled = False
            try:
                cancelled = bool(
                    facade.cancel_request_blocks_if_unclaimed(
                        req_id=source_req_id, uuid=uuid))
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "Release-only cancellation failed req_id=%s "
                    "uuid=%d: %s (p2p_wait_pull_timeout remains the backstop)",
                    source_req_id, uuid, exc)
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": "raiden_stage3_release_only",
                        "req_id": source_req_id,
                        "destination_req_id": destination_req_id,
                        "uuid": uuid,
                        "cancelled": cancelled,
                    },
                    sort_keys=True,
                ),
            )

        if synchronous:
            _release()
            return
        self._stage3_submit_queue.put(_release)
        self._ensure_stage3_submit_workers()

    def _submit_stage3_loads(self,
                             metadata: TPUConnectorMetadata,
                             engine: "KVCacheManager",
                             synchronous: bool = False) -> None:
        """Starts exactly one source-controller transfer per request.

        Each request's scheduler-visible tables are staged here, on the
        model-runner thread; the coordination RPC itself then runs on a
        background submit worker (or inline when ``synchronous``) and its
        outcome is applied when the poll drains the mailbox.
        """
        del engine  # Completion is observed through the manager in poll_stats.
        _, dst_controller_address = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError(
                "Stage-3 destination work unit is not registered")

        for destination_req_id, req_meta in metadata.reqs_to_load.items():
            if not isinstance(req_meta, _Stage3LoadMeta):
                raise TypeError(
                    "Stage-3 consumer requires controller load metadata, got "
                    f"{type(req_meta).__name__}")
            if req_meta.release_only:
                self._stage3_release_producer_registration(
                    destination_req_id, req_meta, synchronous=synchronous)
                continue
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens)
            source_req_id = req_meta.source_req_id
            self._bind_stage3_request_ids(destination_req_id, source_req_id)
            existing_uuid = self._stage3_submitted_loads.get(
                destination_req_id)
            if existing_uuid is not None:
                if (existing_uuid != uuid
                        or self._stage3_submitted_load_tokens.get(
                            destination_req_id) != num_tokens):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 load submission for "
                        f"req_id={destination_req_id}")
                continue
            self._stage3_submitted_loads[destination_req_id] = uuid
            self._stage3_submitted_load_tokens[destination_req_id] = num_tokens
            self._stage3_load_start_times[
                destination_req_id] = time.perf_counter()
            local_blocks = list(req_meta.local_block_ids)
            self._load_block_ids[destination_req_id] = local_blocks
            now = time.perf_counter()
            defer_wait_s = (dist_utils.get_stage3_registration_wait_s()
                            if dist_utils.get_stage3_deferred_submit_enabled()
                            else None)
            pending = _Stage3PendingSubmit(
                req_meta=req_meta,
                source_req_id=source_req_id,
                destination_req_id=destination_req_id,
                uuid=uuid,
                num_tokens=num_tokens,
                local_blocks=local_blocks,
                dst_controller_address=dst_controller_address,
                dst_units=[self._raiden_work_unit],
                defer_deadline=(None if defer_wait_s is None else now +
                                defer_wait_s),
                first_attempt_s=now)
            self._dispatch_stage3_load_submit(pending, synchronous)

    def _stage_stage3_submit_task(
            self, pending: _Stage3PendingSubmit,
            synchronous: bool) -> Optional[_Stage3SubmitTask]:
        """Prepares the coordination call for one staged load.

        Model-runner thread only. A failure here precedes any controller
        contact, so the load resolves to the pre-arm failure terminal and
        None is returned.
        """
        req_meta = pending.req_meta
        source_req_id = pending.source_req_id
        destination_req_id = pending.destination_req_id
        uuid = pending.uuid
        local_blocks = pending.local_blocks
        try:
            src_units = self._stage3_source_work_units(req_meta)
            src_controller_address = str(
                req_meta.src_controller_address).strip()
            if not src_controller_address:
                raise ValueError("empty Stage-3 source controller address")
            source_facade = self._stage3_source_facades.get(
                src_controller_address)
            if source_facade is None:
                source_facade = self._new_raiden_controller_facade(
                    src_controller_address)
                self._stage3_source_facades[src_controller_address] = (
                    source_facade)
            # ONE transfer carries every replicated cache class; tag
            # order fixes the H2D order ranks executor-side (the first
            # tag = group 0 uploads first). Each tag replays over the
            # same destination pages.
            if _use_raiden_glm_admission():
                transfer_tags = list(_STAGE3_GLM_TRANSFER_POOL_TAGS)
            else:
                transfer_tags = self._stage3_fa_registration_tags()
            fa_tag_count = len(transfer_tags)
            dst_blocks = list(local_blocks) * len(transfer_tags)
            dst_counts = [len(local_blocks)] * len(transfer_tags)
            mamba_state_block_ids = req_meta.mamba_state_block_ids
            if self._stage3_state_group_count:
                if not mamba_state_block_ids or len(
                        mamba_state_block_ids
                ) != self._stage3_state_group_count:
                    raise RuntimeError(
                        "GDN state load requires one destination state "
                        "block per mamba group")
                for tag, ordinal in self._stage3_state_registration_tags():
                    transfer_tags.append(tag)
                    dst_blocks.append(int(mamba_state_block_ids[ordinal]))
                    dst_counts.append(1)
            # Prefix-aware suffix pull: clip the FA global destination
            # byte space at the locally-cached prefix; GDN state classes
            # are not prefix-decomposable and always transfer whole
            # (skip 0). Omit the kwarg entirely when there is no clip so
            # skip-free requests stay byte-identical on the wire.
            skip_tokens = req_meta.skip_tokens
            fa_skip_bytes = 0
            clip_kwargs: dict[str, Any] = {}
            if skip_tokens > 0:
                token_bytes = self._stage3_fa_token_bytes(
                    int(self.vllm_config.cache_config.block_size))
                fa_skip_bytes = skip_tokens * token_bytes
                clip_kwargs["dst_skip_bytes"] = (
                    [fa_skip_bytes] * fa_tag_count + [0] *
                    (len(transfer_tags) - fa_tag_count))
        except Exception as exc:  # pylint: disable=broad-except
            # Local staging failed before any controller contact, so no
            # receiver can have been armed and no late H2D is possible.
            self._record_stage3_load_failure(destination_req_id, local_blocks)
            logger.error(
                "Stage-3 controller transfer failed before receiver "
                "arming req_id=%s destination_req_id=%s uuid=%d: %s",
                source_req_id, destination_req_id, uuid, exc)
            return None
        return _Stage3SubmitTask(
            pending=pending,
            skip_tokens=skip_tokens,
            fa_skip_bytes=fa_skip_bytes,
            source_units=len(src_units),
            facade=source_facade,
            call_kwargs=dict(
                src_units=src_units,
                dst_units=list(pending.dst_units),
                # The registration is keyed by the producer's internal
                # request ID. The destination's independently randomized
                # ID is strictly a local scheduler lifecycle key.
                req_id=source_req_id,
                dst_device_block_ids=dst_blocks,
                dst_mem_type=self._raiden_hbm_memory_type(),
                use_block_chunks=True,
                src_controller_address=src_controller_address,
                dst_controller_address=pending.dst_controller_address,
                uuid=uuid,
                is_sender=True,
                num_tokens=pending.num_tokens,
                transfer_pool_tags=transfer_tags,
                dst_block_counts=dst_counts,
                **clip_kwargs,
            ),
            retry_inline=pending.defer_deadline is None or synchronous,
        )

    def _dispatch_stage3_load_submit(self,
                                     pending: _Stage3PendingSubmit,
                                     synchronous: bool = False) -> None:
        """Issues one coordination attempt for a staged load.

        Synchronous (inline-load) mode runs the RPC on the calling thread and
        applies its outcome at once; otherwise the RPC rides a submit worker
        and the outcome is applied by the poll that drains the mailbox.
        """
        task = pending.task
        if task is None:
            task = self._stage_stage3_submit_task(pending, synchronous)
            if task is None:
                return
            pending.task = task
        if synchronous:
            self._apply_stage3_submit_outcome(
                self._execute_stage3_submit(task))
            return
        # The uncertainty window opens at enqueue: past this deadline a
        # still-unresolved RPC is treated exactly like a post-arm failure
        # (blocks held, request abandoned) so one black-holed peer cannot
        # pin a request forever.
        self._stage3_inflight_submits[pending.destination_req_id] = (
            time.perf_counter() +
            float(dist_utils.get_p2p_wait_pull_timeout()))
        self._stage3_submit_queue.put(task)
        self._ensure_stage3_submit_workers()

    def _ensure_stage3_submit_workers(self) -> None:
        if self._stage3_submit_threads:
            return
        for index in range(_STAGE3_SUBMIT_WORKERS):
            thread = threading.Thread(
                target=self._stage3_submit_worker_loop,
                name=f"raiden-stage3-submit-r{self.tp_rank}-{index}",
                daemon=True)
            thread.start()
            self._stage3_submit_threads.append(thread)

    def _stage3_submit_worker_loop(self) -> None:
        while True:
            task = self._stage3_submit_queue.get()
            try:
                # Release-only cancellations ride the queue as bare
                # callables; they have no scheduler-visible outcome to hand
                # back.
                if callable(task):
                    task()
                    continue
                outcome = self._execute_stage3_submit(task)
                with self._stage3_submit_lock:
                    self._stage3_submit_outcomes.append(outcome)
            finally:
                self._stage3_submit_queue.task_done()

    def _execute_stage3_submit(
            self, task: _Stage3SubmitTask) -> _Stage3SubmitOutcome:
        """Runs one coordination RPC; safe on any thread.

        The facade RPC returns only after the source controller has awaited
        its transfer future (receiver arm + sender dispatch). Actual byte
        completion is deliberately *not* inferred from that acknowledgement;
        get_finished polls the local manager.
        """
        start_submit = time.perf_counter()
        if (task.pending.destination_req_id
                in self._stage3_finished_loads_pending_cleanup):
            return _Stage3SubmitOutcome(
                task=task,
                error=RuntimeError(
                    "Missing producer block registration (aborted before submit)"
                ),
                submit_ms=0.0,
                attempted_at=start_submit,
            )
        error: Optional[BaseException] = None
        try:
            if task.retry_inline:
                accepted = self._start_stage3_transfer_with_d5_retry(
                    task.facade, **task.call_kwargs)
            else:
                # Deferred mode: one attempt. A missing registration is
                # handed back as an outcome and the request is re-attempted
                # from a later step instead of spinning here.
                accepted = task.facade.start_transfer(**task.call_kwargs)
            if accepted is not True:
                raise RuntimeError(
                    "source controller rejected Stage-3 transfer")
        except Exception as exc:  # pylint: disable=broad-except
            error = exc
        return _Stage3SubmitOutcome(
            task=task,
            error=error,
            submit_ms=(time.perf_counter() - start_submit) * 1000,
            attempted_at=start_submit)

    def _apply_stage3_submit_outcome(self,
                                     outcome: _Stage3SubmitOutcome) -> None:
        """Applies one RPC outcome to the scheduler-visible tables.

        Model-runner thread only: every table it touches pairs with the
        get_finished/get_block_ids_with_load_errors drain of the same pass.
        """
        task = outcome.task
        pending = task.pending
        destination_req_id = pending.destination_req_id
        source_req_id = pending.source_req_id
        uuid = pending.uuid
        self._stage3_inflight_submits.pop(destination_req_id, None)
        if destination_req_id in self._stage3_abandoned_submits:
            self._stage3_abandoned_submits.discard(destination_req_id)
            logger.warning(
                "Discarding Stage-3 submission outcome for an abandoned "
                "request req_id=%s destination_req_id=%s uuid=%d error=%s",
                source_req_id, destination_req_id, uuid, outcome.error)
            return
        if destination_req_id not in self._stage3_submitted_loads:
            logger.warning(
                "Discarding Stage-3 submission outcome for an already "
                "released request req_id=%s destination_req_id=%s uuid=%d "
                "error=%s", source_req_id, destination_req_id, uuid,
                outcome.error)
            return
        if outcome.error is not None:
            exc = outcome.error
            missing_registration = ("Missing producer block registration"
                                    in str(exc))
            aborted = (destination_req_id
                       in self._stage3_finished_loads_pending_cleanup)
            if (missing_registration and pending.defer_deadline is not None
                    and not aborted
                    and time.perf_counter() < pending.defer_deadline):
                # The precise pre-arm missing-registration result is
                # side-effect free (registrations ride a later producer
                # scheduler step): park for a per-step re-attempt until the
                # deadline.
                pending.last_attempt_s = outcome.attempted_at
                if not pending.wait_logged:
                    pending.wait_logged = True
                    logger.info(
                        "Stage-3 deferring load until producer request-"
                        "block registrations arrive req_id=%s "
                        "destination_req_id=%s uuid=%d wait_budget_s=%.1f",
                        source_req_id, destination_req_id, uuid,
                        dist_utils.get_stage3_registration_wait_s())
                self._stage3_pending_submits[destination_req_id] = pending
                return
            if missing_registration:
                # The FA registration lookup fails before receiver arming,
                # so no late H2D is possible. A request the scheduler already
                # finished resolves here as well, so its delayed block-free
                # path completes.
                self._record_stage3_load_failure(destination_req_id,
                                                 pending.local_blocks)
                logger.error(
                    "Stage-3 controller transfer failed before receiver "
                    "arming req_id=%s destination_req_id=%s uuid=%d: %s",
                    source_req_id, destination_req_id, uuid, exc)
                return
            # Generic RPC rejection can occur after receiver arming.
            # Keep the request blocked and accept native terminal
            # records; only surface recompute after manager failure or
            # a full post-RPC manager timeout.
            self._stage3_controller_accepted.add(destination_req_id)
            deadline = (time.perf_counter() +
                        float(dist_utils.get_p2p_wait_pull_timeout()))
            self._stage3_pending_controller_failures[
                destination_req_id] = deadline
            logger.error(
                "Stage-3 controller transfer outcome uncertain; "
                "waiting for native terminal state req_id=%s "
                "destination_req_id=%s uuid=%d: %s", source_req_id,
                destination_req_id, uuid, exc)
            return
        self._stage3_controller_accepted.add(destination_req_id)
        logger.info(
            "%s",
            json.dumps(
                {
                    "event": "raiden_stage3_transfer_submitted",
                    "req_id": source_req_id,
                    "destination_req_id": destination_req_id,
                    "uuid": uuid,
                    "num_tokens": pending.num_tokens,
                    "skip_tokens": task.skip_tokens,
                    "fa_skip_bytes": task.fa_skip_bytes,
                    "recv_armed_before_push": True,
                    "state_group_count": self._stage3_state_group_count,
                    "destination_pages": len(pending.local_blocks),
                    "controller_submit_ms": outcome.submit_ms,
                },
                sort_keys=True,
            ),
        )
        if pending.wait_logged:
            logger.info(
                "Stage-3 deferred load submitted after %.2fs wait "
                "req_id=%s destination_req_id=%s uuid=%d",
                time.perf_counter() - pending.first_attempt_s, source_req_id,
                destination_req_id, uuid)
        logger.info(
            "Raiden Stage 3 reshard: req_id=%s "
            "destination_req_id=%s "
            "recv_armed_before_push=1 state_groups=%d",
            source_req_id,
            destination_req_id,
            self._stage3_state_group_count,
        )
        logger.debug(
            "TPURaidenConnectorWorker rank%d --> submitted Stage-3 load "
            "req_id=%s destination_req_id=%s uuid=%d src_units=%d "
            "destination_pages=%d num_tokens=%d",
            self.tp_rank,
            source_req_id,
            destination_req_id,
            uuid,
            task.source_units,
            len(pending.local_blocks),
            pending.num_tokens,
        )

    def _drain_stage3_submit_outcomes(self) -> None:
        """Applies queued RPC outcomes and abandons overdue submissions."""
        with self._stage3_submit_lock:
            outcomes = self._stage3_submit_outcomes
            self._stage3_submit_outcomes = []
        for outcome in outcomes:
            self._apply_stage3_submit_outcome(outcome)
        now = time.perf_counter()
        for req_id, deadline in list(self._stage3_inflight_submits.items()):
            if deadline > now:
                continue
            # The receiver may already be armed by an RPC this slow, so the
            # request follows the post-arm-failure path: blocks stay held on
            # the uncertainty deadline instead of being freed under a
            # possible late H2D.
            self._stage3_inflight_submits.pop(req_id)
            self._stage3_abandoned_submits.add(req_id)
            self._stage3_controller_accepted.add(req_id)
            self._stage3_pending_controller_failures[req_id] = (
                now + float(dist_utils.get_p2p_wait_pull_timeout()))
            logger.error(
                "Stage-3 controller submission RPC exceeded its deadline and "
                "was abandoned; waiting for native terminal state "
                "destination_req_id=%s", req_id)

    def _drain_stage3_pending_submits(
            self, finished_req_ids: set[str] | None) -> None:
        """Re-attempts parked Stage-3 loads once per scheduler step, paced
        by the re-attempt interval."""
        aborted_req_ids = (set(finished_req_ids or ())
                           | self._stage3_finished_loads_pending_cleanup)
        if aborted_req_ids:
            for req_id in aborted_req_ids:
                pending = self._stage3_pending_submits.pop(req_id, None)
                if pending is None:
                    continue
                # Scheduler-finished (aborted) before the producer
                # registered: resolve to the pre-arm failure terminal so the
                # delayed block-free path completes.
                self._record_stage3_load_failure(req_id, pending.local_blocks)
                logger.warning(
                    "Stage-3 deferred load aborted before producer "
                    "registration req_id=%s destination_req_id=%s uuid=%d",
                    pending.source_req_id, req_id, pending.uuid)
        now = time.perf_counter()
        for destination_req_id in list(self._stage3_pending_submits):
            pending = self._stage3_pending_submits[destination_req_id]
            if (now - pending.last_attempt_s
                    < _STAGE3_REGISTRATION_REATTEMPT_MIN_INTERVAL_S):
                continue
            del self._stage3_pending_submits[destination_req_id]
            self._dispatch_stage3_load_submit(pending)
