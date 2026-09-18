# SPDX-License-Identifier: Apache-2.0
"""Producer (prefill) side of Stage-3 Raiden KV transfers.

``_Stage3ProducerMixin`` holds the ``TPURaidenConnectorWorker`` methods that
run on the prefill engine after a request's KV is ready: registering this
rank's physical source blocks with the cluster controller (FA plus GDN state
pools for Qwen3.5, the row-granular stripe for GLM), the byte, tag and span
geometry those registrations are computed from, tracking each send to its
terminal state so the scheduler can free the blocks, and probing the
controller registry for consumer-side cancellations.
"""

import dataclasses
import json
import time
from typing import Any

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu.distributed.kv_transfer.connector_metadata import \
    TPUConnectorMetadata
from vllm_torchtpu.distributed.kv_transfer.raiden_config import (
    _STAGE3_GLM_TRANSFER_POOL_TAGS, _STAGE3_REGISTRATION_CANCELLED_ERROR,
    _STAGE3_REGISTRY_STATUS_CANCELLED, _STAGE3_REGISTRY_STATUS_UNKNOWN,
    _STAGE3_STATE_CLASS_TAGS, _engine_is_pipeline, _use_per_layer_pool_tags)
from vllm_torchtpu.distributed.kv_transfer.stage3_types import \
    _Stage3RegisteredSend
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class _Stage3ProducerMixin:
    """Prefill-side Stage-3 registration for ``TPURaidenConnectorWorker``.

    Mixin: no state of its own. The attributes it reads are declared in
    ``TPURaidenConnectorWorker.__init__``, except ``_stage3_glm_geometry``,
    which GLM admission creates; the rank and parallelism helpers it calls
    live on the worker itself.
    """

    @staticmethod
    def _maybe_delay_stage3_registration_for_test(
            metadata: TPUConnectorMetadata) -> None:
        """TEST-ONLY fault injection (TPU_RAIDEN_TEST_REGISTRATION_DELAY_S):
        widen the out-of-band registration race by delaying the producer's
        registration step, reproducing a saturated producer whose
        registration-carrying step runs long after the finish response."""
        delay_s = dist_utils.get_stage3_test_registration_delay_s()
        if delay_s > 0 and metadata.reqs_to_send:
            logger.warning(
                "TEST INJECTION: delaying Stage-3 request-block registration by "
                "%.1fs for req_ids=%s", delay_s, sorted(metadata.reqs_to_send))
            time.sleep(delay_s)

    def _register_stage3_request_blocks(
            self, metadata: TPUConnectorMetadata) -> None:
        """Registers only this PCP rank's interleaved physical source blocks."""
        facade, _ = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError("Stage-3 producer work unit is not registered")
        parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        if transfer_rank < 0 or transfer_rank >= parallelism:
            raise ValueError(
                "Stage-3 producer transfer rank is outside parallelism: "
                f"rank={transfer_rank}, parallelism={parallelism}")
        self._maybe_delay_stage3_registration_for_test(metadata)
        # A pipeline stage holds every token of its own layers: its byte
        # spans are lowered as a single contiguous owner, and the layer
        # subset is expressed through per-layer pool tags instead.
        pipeline = _engine_is_pipeline(self.vllm_config)
        lowering_parallelism = 1 if pipeline else parallelism
        lowering_rank = 0 if pipeline else transfer_rank

        now = time.perf_counter()
        expired_terminal_req_ids = [
            req_id
            for req_id, registration in self._stage3_terminal_sends.items()
            if registration.expiration_time <= now
        ]
        for req_id in expired_terminal_req_ids:
            self._stage3_terminal_sends.pop(req_id, None)
            self._stage3_reported_sends.discard(req_id)
            self._stage3_prune_state_send_tracking(req_id)

        for req_id, req_meta in metadata.reqs_to_send.items():
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens or 0)
            expiration_time = float(req_meta.expiration_time)
            if num_tokens <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires num_tokens")
            if expiration_time <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires a positive "
                    "expiration_time")
            page_tokens = int(self.vllm_config.cache_config.block_size)
            interleave_tokens = self._raiden_interleave_tokens(
                page_tokens, lowering_parallelism)
            scheduler_block_tokens = page_tokens * lowering_parallelism
            expected_scheduler_blocks = (
                (num_tokens + scheduler_block_tokens - 1) //
                scheduler_block_tokens)
            scheduler_ids = tuple(
                int(block_id) for block_id in req_meta.local_block_ids)
            if len(scheduler_ids) != expected_scheduler_blocks:
                raise ValueError(
                    "Stage-3 producer scheduler block count is inconsistent "
                    f"with PCP geometry: blocks={len(scheduler_ids)}, "
                    f"expected={expected_scheduler_blocks}, "
                    f"num_tokens={num_tokens}, page_tokens={page_tokens}, "
                    f"interleave_tokens={interleave_tokens}, "
                    f"parallelism={parallelism}")
            # This rank's declared source map, lowered from the kernel's own
            # layout package to the byte-span IR: owned interleave slices
            # packed dense rank-major over this rank's blocks (a prefix of
            # the shared scheduler BlockTable ids), split at destination
            # page boundaries, scaled to bytes. The producer-side semantic
            # checks live in this construction and Raiden's byte-level
            # registration/planning validation.
            from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (  # noqa: E501
                lower_fa_spans, owned_token_ranges)
            owned_tokens = sum(end - start
                               for start, end in owned_token_ranges(
                                   num_tokens=num_tokens,
                                   transfer_rank=lowering_rank,
                                   parallelism=lowering_parallelism,
                                   interleave_tokens=interleave_tokens,
                               ))
            owned_blocks = (owned_tokens + page_tokens - 1) // page_tokens
            local_ids = scheduler_ids[:owned_blocks]
            fa_pool_spans = []
            token_bytes = self._stage3_fa_token_bytes(page_tokens)
            fa_registration = lower_fa_spans(
                num_tokens=num_tokens,
                transfer_rank=lowering_rank,
                parallelism=lowering_parallelism,
                interleave_tokens=interleave_tokens,
                page_tokens=page_tokens,
                token_bytes=token_bytes,
                block_ids=list(local_ids),
            )
            if fa_registration.spans:
                # One identical declaration per FA tag this rank holds
                # (one per FA layer under per-layer tags).
                fa_pool_spans = [
                    dataclasses.replace(fa_registration, tag=tag)
                    for tag in self._stage3_fa_registration_tags()
                ]
            terminal = self._stage3_terminal_sends.get(req_id)
            if terminal is not None:
                if (terminal.uuid != uuid
                        or terminal.local_block_ids != local_ids
                        or terminal.num_tokens != num_tokens):
                    raise ValueError(
                        "Conflicting replay after terminal Stage-3 producer "
                        f"registration for req_id={req_id}")
                continue
            existing = self._stage3_registered_sends.get(req_id)
            if existing is not None:
                if (existing.uuid != uuid
                        or existing.local_block_ids != local_ids
                        or existing.num_tokens != num_tokens):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 producer registration "
                        f"for req_id={req_id}")
                continue
            try:
                # T3.1 sibling collapse: one registration carries the FA
                # spans plus every GDN state class's spans under the SAME
                # req_id/uuid — one D5 row per rank, one claim lifecycle.
                # The controller filters registrations per requested tag at
                # plan build.
                pool_spans = list(fa_pool_spans)
                pool_spans.extend(
                    self._stage3_state_pool_spans(
                        getattr(req_meta, "mamba_state_block_ids", None),
                        lowering_rank, lowering_parallelism))
                facade.register_request_blocks(
                    req_id=req_id,
                    uuid=uuid,
                    unit=self._raiden_work_unit,
                    block_ids=list(local_ids),
                    pool_spans=pool_spans,
                )
            except (RuntimeError, ValueError) as exc:
                if _STAGE3_REGISTRATION_CANCELLED_ERROR not in str(exc):
                    raise
                # Another rank won the expiry race and atomically sealed this
                # request generation before this slow rank reached D5. There
                # is no local registration to complete or release. Retain a
                # fresh local tombstone so metadata replay is idempotent, and
                # contribute this rank's terminal vote to PCP aggregation.
                tombstone_deadline = (
                    time.perf_counter() +
                    float(dist_utils.get_p2p_wait_pull_timeout()))
                self._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(
                    uuid=uuid,
                    local_block_ids=local_ids,
                    num_tokens=num_tokens,
                    expiration_time=tombstone_deadline,
                )
                self._done_sending.add(req_id)
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> Stage-3 request "
                    "registration was already cancelled req_id=%s uuid=%d",
                    self.tp_rank,
                    req_id,
                    uuid,
                )
                continue
            self._stage3_registered_sends[req_id] = _Stage3RegisteredSend(
                uuid=uuid,
                local_block_ids=local_ids,
                num_tokens=num_tokens,
                expiration_time=expiration_time,
            )
            if not local_ids:
                # This PCP rank owns no physical page for the request. It has
                # nevertheless published an empty D5 entry so controller
                # lookup sees the complete source unit set, and is terminal
                # immediately: there is no native transfer to wait for.
                self._stage3_send_outcomes.setdefault(req_id, "done")
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": ("raiden_stage3_request_blocks_registered"),
                        "req_id":
                        req_id,
                        "uuid":
                        uuid,
                        "num_tokens":
                        num_tokens,
                        "transfer_rank":
                        transfer_rank,
                        "parallelism":
                        parallelism,
                        "page_tokens":
                        page_tokens,
                        "interleave_tokens":
                        interleave_tokens,
                        "scheduler_blocks":
                        expected_scheduler_blocks,
                        "owned_blocks":
                        len(local_ids),
                        "owned_tokens":
                        owned_tokens,
                        "declared_spans":
                        (len(fa_pool_spans[0].spans) if fa_pool_spans else 0),
                    },
                    sort_keys=True,
                ),
            )
            logger.debug(
                "TPURaidenConnectorWorker rank%d --> registered Stage-3 "
                "send req_id=%s uuid=%d transfer_rank=%d/%d blocks=%s",
                self.tp_rank,
                req_id,
                uuid,
                transfer_rank,
                parallelism,
                list(local_ids),
            )

    def _register_stage3_request_blocks_glm(
            self, metadata: TPUConnectorMetadata) -> None:
        """Registers this producer rank's row-granular span stripe.

        The caches are TP-replicated: every producer rank shares the full
        scheduler block table and owns a round-robin stripe of its pages.
        """
        from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import \
            lower_glm_row_spans

        facade, _ = self._require_stage3_controller()
        if self._raiden_work_unit is None:
            raise RuntimeError("Stage-3 producer work unit is not registered")
        parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        self._maybe_delay_stage3_registration_for_test(metadata)

        now = time.perf_counter()
        expired_terminal_req_ids = [
            req_id
            for req_id, registration in self._stage3_terminal_sends.items()
            if registration.expiration_time <= now
        ]
        for req_id in expired_terminal_req_ids:
            self._stage3_terminal_sends.pop(req_id, None)
            self._stage3_reported_sends.discard(req_id)
            self._stage3_prune_state_send_tracking(req_id)

        tag_geometry = self._stage3_glm_geometry
        page_tokens = int(self.vllm_config.cache_config.block_size)
        for req_id, req_meta in metadata.reqs_to_send.items():
            uuid = int(req_meta.uuid)
            num_tokens = int(req_meta.num_tokens or 0)
            expiration_time = float(req_meta.expiration_time)
            if num_tokens <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires num_tokens")
            if expiration_time <= 0:
                raise ValueError(
                    "Stage-3 producer send metadata requires a positive "
                    "expiration_time")
            scheduler_ids = tuple(
                int(block_id) for block_id in req_meta.local_block_ids)
            pool_spans = [
                lower_glm_row_spans(
                    tag=tag,
                    num_tokens=num_tokens,
                    transfer_rank=transfer_rank,
                    parallelism=parallelism,
                    page_tokens=page_tokens,
                    live_bytes_per_block=tag_geometry[tag][0],
                    row_bytes=tag_geometry[tag][1],
                    block_ids=list(scheduler_ids),
                ) for tag in _STAGE3_GLM_TRANSFER_POOL_TAGS
            ]
            # Every tag stripes identically, so one tag's span count is this
            # rank's page count.
            owned_pages = len(pool_spans[0].spans)
            if not owned_pages:
                # Ranks owning no page publish a fully empty registration
                # (only empty block_ids may complete before the claim) and
                # self-complete below.
                pool_spans = []
                registered_ids: tuple[int, ...] = ()
            else:
                registered_ids = scheduler_ids
            terminal = self._stage3_terminal_sends.get(req_id)
            if terminal is not None:
                if (terminal.uuid != uuid
                        or terminal.local_block_ids != registered_ids
                        or terminal.num_tokens != num_tokens):
                    raise ValueError(
                        "Conflicting replay after terminal Stage-3 producer "
                        f"registration for req_id={req_id}")
                continue
            existing = self._stage3_registered_sends.get(req_id)
            if existing is not None:
                if (existing.uuid != uuid
                        or existing.local_block_ids != registered_ids
                        or existing.num_tokens != num_tokens):
                    raise ValueError(
                        "Conflicting duplicate Stage-3 producer registration "
                        f"for req_id={req_id}")
                continue
            try:
                facade.register_request_blocks(
                    req_id=req_id,
                    uuid=uuid,
                    unit=self._raiden_work_unit,
                    block_ids=list(registered_ids),
                    pool_spans=pool_spans,
                )
            except (RuntimeError, ValueError) as exc:
                if _STAGE3_REGISTRATION_CANCELLED_ERROR not in str(exc):
                    raise
                tombstone_deadline = (
                    time.perf_counter() +
                    float(dist_utils.get_p2p_wait_pull_timeout()))
                self._stage3_terminal_sends[req_id] = _Stage3RegisteredSend(
                    uuid=uuid,
                    local_block_ids=registered_ids,
                    num_tokens=num_tokens,
                    expiration_time=tombstone_deadline,
                )
                self._done_sending.add(req_id)
                logger.warning(
                    "TPURaidenConnectorWorker rank%d --> Stage-3 GLM "
                    "registration was already cancelled req_id=%s uuid=%d",
                    self.tp_rank, req_id, uuid)
                continue
            self._stage3_registered_sends[req_id] = _Stage3RegisteredSend(
                uuid=uuid,
                local_block_ids=registered_ids,
                num_tokens=num_tokens,
                expiration_time=expiration_time,
            )
            if not owned_pages:
                # No pages owned: terminal immediately, no transfer.
                self._stage3_send_outcomes.setdefault(req_id, "done")
            event = {
                "event": "raiden_stage3_glm_request_blocks_registered",
                "req_id": req_id,
                "uuid": uuid,
                "num_tokens": num_tokens,
                "transfer_rank": transfer_rank,
                "parallelism": parallelism,
                "page_tokens": page_tokens,
                "request_pages": len(scheduler_ids),
                "owned_pages": owned_pages,
                "declared_bytes": {
                    reg.tag: reg.declared_bytes
                    for reg in pool_spans
                },
            }
            logger.info("%s", json.dumps(event, sort_keys=True))

    def _stage3_fa_token_bytes(self, page_tokens: int) -> int:
        """Bytes per token of the admitted FA pools, measured from the
        manifest (live bytes per block over the page geometry)."""
        manifest = self._raiden_manifest
        if manifest is None or not hasattr(manifest, "geometry_by_tag"):
            raise RuntimeError(
                "Stage-3 byte-span lowering requires the admitted pool "
                "manifest")
        geometry = manifest.geometry_by_tag().get("fa")
        if not geometry:
            raise RuntimeError("Stage-3 admitted manifest contains no FA "
                               "pools")
        live_bytes = int(geometry["live_bytes_per_block"])
        if page_tokens <= 0 or live_bytes <= 0 or live_bytes % page_tokens:
            raise RuntimeError(
                "Stage-3 FA pool live bytes are not token-aligned: "
                f"live={live_bytes}, page_tokens={page_tokens}")
        return live_bytes // page_tokens

    def _stage3_state_live_bytes(self, tag: str) -> int:
        """Return the admitted whole-slot byte size for an exact state tag."""
        manifest = self._raiden_manifest
        if manifest is None or not hasattr(manifest, "geometry_by_tag"):
            raise RuntimeError(
                "GDN state reshard requires the admitted pool manifest")
        from vllm_torchtpu.distributed.kv_transfer.raiden.tags import class_tag
        geometry = manifest.geometry_by_tag().get(class_tag(tag))
        if not geometry:
            raise RuntimeError(
                f"GDN state tag {tag!r} is absent from the pool manifest")
        live_bytes = int(geometry["live_bytes_per_block"])
        if live_bytes <= 0:
            raise RuntimeError(
                f"GDN state tag {tag!r} has invalid live bytes {live_bytes}")
        return live_bytes

    def _stage3_state_regions(self, tag: str) -> tuple[Any, ...]:
        """Return one uniform admitted live-region map for a state tag."""
        manifest = self._raiden_manifest
        pools = tuple(getattr(manifest, "pools", ()) or ())
        matching = tuple(pool for pool in pools
                         if str(getattr(pool, "tag", "")) == tag)
        if not matching:
            raise RuntimeError(
                f"GDN state tag {tag!r} is absent from the pool manifest")

        def signature(pool: Any) -> tuple[tuple[Any, ...], ...]:
            return tuple((str(region.name), int(region.offset_bytes),
                          int(region.stride_bytes), int(region.unit_bytes),
                          int(region.num_units), int(region.units_per_stride))
                         for region in pool.regions)

        expected = signature(matching[0])
        if any(signature(pool) != expected for pool in matching[1:]):
            raise RuntimeError(
                f"GDN state pools tagged {tag!r} disagree on live regions")
        return tuple(matching[0].regions)

    def _stage3_fa_registration_tags(self) -> list[str]:
        """FA pool tags this rank declares spans for: the class tag, or one
        tag per FA layer under per-layer tags."""
        if not _use_per_layer_pool_tags():
            return ["fa"]
        manifest = self._raiden_manifest
        if manifest is None:
            raise RuntimeError("Stage-3 registration requires the admitted "
                               "pool manifest")
        tags = [pool.tag for pool in manifest.pools_of_class("fa")]
        if not tags:
            raise RuntimeError("Stage-3 admitted manifest contains no FA "
                               "pools")
        return tags

    def _stage3_state_registration_tags(self) -> list[tuple[str, int]]:
        """(exact state tag, mamba group ordinal) pairs this rank declares:
        one per state class and group, or one per state pool under
        per-layer tags. Class order first, then manifest order."""
        from vllm_torchtpu.distributed.kv_transfer.raiden.tags import \
            split_layer_tag

        if not _use_per_layer_pool_tags():
            return [(f"{tag}.g{ordinal}", ordinal)
                    for tag in _STAGE3_STATE_CLASS_TAGS
                    for ordinal in range(self._stage3_state_group_count)]
        manifest = self._raiden_manifest
        if manifest is None:
            raise RuntimeError("Stage-3 registration requires the admitted "
                               "pool manifest")
        pairs = []
        for tag in _STAGE3_STATE_CLASS_TAGS:
            for pool in manifest.pools:
                class_value, layer = split_layer_tag(pool.tag)
                if layer is None or not class_value.startswith(f"{tag}.g"):
                    continue
                ordinal_text = class_value[len(tag) + 2:]
                if not ordinal_text.isdigit():
                    raise RuntimeError(
                        f"GDN state tag {pool.tag!r} carries no mamba group "
                        "ordinal")
                pairs.append((pool.tag, int(ordinal_text)))
        return pairs

    def _stage3_state_pool_spans(self, mamba_state_block_ids,
                                 transfer_rank: int, parallelism: int) -> list:
        """Producer: the GDN state classes' span registrations.

        PCP GDN execution exchanges token shards for head shards.  Every PCP
        rank therefore owns a distinct q/k/v and SSM head slice.  Each rank
        declares its slice in compact-live byte space, and the controller
        combines the complete rank set into the full non-PCP destination
        state. T3.1: the registrations ride the base request's single
        register_request_blocks call (same req_id/uuid as FA) instead of
        derived sibling registrations.
        """
        if not self._stage3_state_group_count:
            return []
        if not mamba_state_block_ids:
            raise RuntimeError(
                "GDN state registration requires the producer's per-group "
                "mamba state block ids")
        if len(mamba_state_block_ids) != self._stage3_state_group_count:
            raise RuntimeError(
                "GDN state registration group count disagrees with the "
                f"manifest: blocks={len(mamba_state_block_ids)}, "
                f"groups={self._stage3_state_group_count}")
        from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import \
            lower_gdn_state_shard_spans
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
            EXPECTED_FA_MINOR_TO_MAJOR, EXPECTED_FA_TILES)
        from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import \
            BINDING_ALIASED_RAW

        aliased_raw = self._raiden_manifest.binding == BINDING_ALIASED_RAW
        if aliased_raw:
            # Raw transfers move physical bytes; admit only the fingerprinted
            # tiled layout.  The whole-token QK pair-blocked conv layout is
            # enforced downstream: the span lowering rejects any other conv
            # region vocabulary, and the layout version rides the compared
            # fingerprint payload.
            payload = self._raiden_layout_fingerprint_payload
            if not payload:
                raise RuntimeError(
                    "aliased raw GDN lowering requires an admitted physical "
                    "layout fingerprint")
            measured_minor_to_major = tuple(
                int(value) for value in payload.get("minor_to_major", ()))
            measured_tiles = tuple(
                tuple(int(value) for value in tile)
                for tile in payload.get("tiles", ()))
            measured_element_bits = int(payload.get("element_size_in_bits", 0))
            if (measured_minor_to_major != EXPECTED_FA_MINOR_TO_MAJOR
                    or measured_tiles != EXPECTED_FA_TILES
                    or measured_element_bits != 8):
                raise RuntimeError(
                    "aliased raw GDN lowering is unsupported for the "
                    "admitted physical layout")

        registrations = []
        for exact_tag, ordinal in self._stage3_state_registration_tags():
            block_id = mamba_state_block_ids[ordinal]
            if aliased_raw:
                matching_pools = tuple(
                    pool for pool in self._raiden_manifest.pools
                    if str(getattr(pool, "tag", "")) == exact_tag)
                if any(
                        int(pool.base_offset_bytes) %
                        1024 or int(pool.block_stride_bytes) % 1024
                        for pool in matching_pools):
                    raise RuntimeError(
                        "aliased raw GDN pool base and block stride must "
                        f"be physical-token aligned ({exact_tag})")
            registration = lower_gdn_state_shard_spans(
                tag=exact_tag,
                block_id=int(block_id),
                transfer_rank=transfer_rank,
                parallelism=parallelism,
                regions=self._stage3_state_regions(exact_tag),
            )
            admitted_live_bytes = self._stage3_state_live_bytes(exact_tag)
            if registration.declared_bytes != admitted_live_bytes:
                raise RuntimeError(
                    "GDN state byte lowering disagrees with the admitted "
                    f"manifest for {exact_tag}: lowered="
                    f"{registration.declared_bytes}, admitted="
                    f"{admitted_live_bytes}")
            registrations.append(registration)
        return registrations

    def _stage3_prune_state_send_tracking(self, base_req_id: str) -> None:
        """Drops producer terminal tracking with the base tombstone."""
        self._stage3_send_outcomes.pop(base_req_id, None)

    def _stage3_record_producer_send_terminals(self, done_sending,
                                               failed_sending) -> None:
        """Records native terminals (T3.1: one uuid, base req ids only)."""
        for req_id, failed in [
            (str(req_id), False) for req_id in done_sending
        ] + [(str(req_id), True) for req_id in failed_sending]:
            if req_id in self._stage3_registered_sends:
                if failed:
                    self._stage3_send_outcomes[req_id] = "failed"
                else:
                    self._stage3_send_outcomes.setdefault(req_id, "done")
                continue
            # Replayed native terminals after the base moved to its tombstone
            # are harmless and must not become duplicate scheduler reports.
            if (req_id not in self._stage3_terminal_sends
                    and req_id not in self._stage3_reported_sends):
                logger.warning(
                    "Ignoring unmapped Stage-3 producer terminal req_id=%s",
                    req_id)

    def _stage3_ready_producer_send_terminals(
            self) -> tuple[set[str], set[str], set[str]]:
        """Returns terminal base requests (T3.1: no sibling aggregation)."""
        outcomes, self._stage3_send_outcomes = self._stage3_send_outcomes, {}
        done = {r for r, outcome in outcomes.items() if outcome == "done"}
        failed = {r for r, outcome in outcomes.items() if outcome == "failed"}
        cancelled = {
            r
            for r, outcome in outcomes.items() if outcome == "cancelled"
        }
        return done, failed, cancelled

    def _probe_stage3_registrations(self, facade: Any, now: float) -> None:
        """Observes consumer-side cancellations of this rank's unclaimed
        request-block registrations.

        cancel_request_blocks_if_unclaimed retires only the source store's
        registry row and no native transfer terminal follows, so without this
        probe the registration (and vLLM's delayed block free that waits on
        every rank's send terminal) would sit until p2p_wait_pull_timeout.
        Registrations older than the probe interval and still without a
        terminal are batch-queried at most once per interval; a cancelled
        (or vanished) row is recorded as a cancelled terminal, exactly what
        the TTL sweep records for its own cancellation."""
        probe_s = dist_utils.get_stage3_status_probe_s()
        probe = getattr(facade, "get_request_block_status", None)
        if probe_s <= 0 or probe is None:
            return
        for req_id in list(self._stage3_status_probe_next):
            if req_id not in self._stage3_registered_sends:
                del self._stage3_status_probe_next[req_id]
        due: list[str] = []
        for req_id in self._stage3_registered_sends:
            if (req_id in self._stage3_terminal_cleanup
                    or req_id in self._stage3_send_outcomes):
                continue
            next_probe = self._stage3_status_probe_next.get(req_id)
            if next_probe is None:
                # Grace period: a healthy registration is claimed and
                # completes natively well within one interval.
                self._stage3_status_probe_next[req_id] = now + probe_s
            elif next_probe <= now:
                due.append(req_id)
        if not due:
            return
        for req_id in due:
            self._stage3_status_probe_next[req_id] = now + probe_s
        keys = [(req_id, int(self._stage3_registered_sends[req_id].uuid))
                for req_id in due]
        try:
            statuses = [int(status) for status in probe(keys)]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(
                "Stage-3 registry status probe failed for %d registrations; "
                "p2p_wait_pull_timeout remains the backstop: %s", len(keys),
                exc)
            return
        if len(statuses) != len(keys):
            raise RuntimeError(
                "Stage-3 registry status probe returned "
                f"{len(statuses)} statuses for {len(keys)} keys")
        for req_id, status in zip(due, statuses):
            if status not in (_STAGE3_REGISTRY_STATUS_CANCELLED,
                              _STAGE3_REGISTRY_STATUS_UNKNOWN):
                continue
            self._stage3_send_outcomes[req_id] = "cancelled"
            logger.info(
                "%s",
                json.dumps(
                    {
                        "event": "raiden_stage3_registration_cancelled",
                        "req_id": req_id,
                        "uuid": int(
                            self._stage3_registered_sends[req_id].uuid),
                        "registry_status": status,
                        "transfer_rank": self._local_raiden_transfer_rank(),
                    },
                    sort_keys=True,
                ),
            )
