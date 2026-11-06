# SPDX-License-Identifier: Apache-2.0
"""Model-owned Stage-3 page planning for Qwen's hybrid caches.

Native admission binds the loaded model identity and measured page layout.
This connector does not borrow another model's native pool admission.
"""

import json
import math
import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.v1.outputs import KVConnectorOutput
from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.distributed.kv_transfer.raiden.byte_spans import (
    PoolByteSpan,
    PoolSpanRegistration,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import (
    canonical_layout_fingerprint,
)
from vllm_torchtpu.distributed.kv_transfer.raiden.pool_manifest import (
    PoolEntry,
    PoolManifest,
    materialize_storages,
    verify_storage_binding,
)
from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
    SendMeta,
    TPURaidenConnector,
    TPURaidenConnectorWorker,
    _engine_is_pipeline,
    _resolved_reshard_controller_address,
    _Stage3LoadMeta,
    _Stage3TransferGroup,
    _use_per_layer_pool_tags,
)

from .pd import (
    TransferExtent,
    TransferGroup,
    build_transfer_groups,
    handoff_block_ids,
    skipped_source_pages,
    suffix_pages,
)
from .pd_identity import canonical_rank_identity
from .pd_manifest import build_transfer_manifest, measure_transfer_layout

logger = init_logger(__name__)

def sync_repack_enabled() -> bool:
    """QWEN4_EXP_PD_SYNC_REPACK, or its wall-clock A/B, is configured."""
    return bool(envs.QWEN4_EXP_PD_SYNC_REPACK) or int(envs.QWEN4_EXP_PD_SYNC_REPACK_AB_S) > 0


def sync_repack_on(now: float | None = None) -> bool:
    """Whether a load landing now waits for the scheduler-named step.

    QWEN4_EXP_PD_SYNC_REPACK_AB_S > 0 alternates by wall clock (off in even
    periods, on in odd ones), so one serving job holds both arms. Every
    process of an engine reads the same host clock; a load deferred in an on
    period is flushed at the first step of the next off period.
    """
    period = int(envs.QWEN4_EXP_PD_SYNC_REPACK_AB_S)
    if period > 0:
        t = time.time() if now is None else now
        return int(t // period) % 2 == 1
    return bool(envs.QWEN4_EXP_PD_SYNC_REPACK)


# A hybrid pool's block IDs are only unique within a group, so vLLM rejects
# invalid_block_ids for it; failed loads go back as request-level failures.
_REQUEST_LEVEL_LOAD_FAILURES = "failed_recving" in getattr(
    KVConnectorOutput, "__dataclass_fields__", {}
)


class Qwen4ExpRaidenConnector(TPURaidenConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        parallel = vllm_config.parallel_config
        cache = vllm_config.cache_config
        # Prefix caching is admitted on either leg: a producer publishes the
        # running state of its align-mode state tables (and ordinary, possibly
        # shared, token pages); a consumer adopts its local prefix and pulls
        # the suffix (supports_stage3_grouped_prefix_loads).
        # MTP is admitted on a consumer behind TP1 pipeline stages only: the
        # staging repack builds the producer's plan without the draft and
        # lands states in the committed entry of a speculative table. The
        # producer ships no draft KV, so the draft's prompt KV starts empty
        # (meant for synthetic rejection sampling).
        transfer = vllm_config.kv_transfer_config
        decode_only_mtp = (
            vllm_config.speculative_config is not None
            and transfer is not None
            and not transfer.is_kv_producer
            and _use_per_layer_pool_tags()
            and parallel.pipeline_parallel_size == 1
            and parallel.tensor_parallel_size > 1
        )
        # A PCP producer (TP1, PCP ranks splitting main KV by token) ships one
        # transfer unit per PCP rank; its decode stages them (pd_staging).
        pcp_producer = (
            parallel.prefill_context_parallel_size > 1
            and transfer is not None
            and transfer.is_kv_producer
            and parallel.tensor_parallel_size == 1
        )
        if (
            (parallel.prefill_context_parallel_size != 1 and not pcp_producer)
            or parallel.decode_context_parallel_size != 1
            or (vllm_config.speculative_config is not None and not decode_only_mtp)
        ):
            raise NotImplementedError(
                "Qwen PD first requires DCP1, PCP only on a TP1 producer, and MTP "
                "only on a staged TP>1 decode"
            )
        # A pipeline producer (Qwen3.5's prefill shape): each TP1 stage ships
        # its own layers' pages whole, paired with a TP1 consumer's pools by
        # per-layer tag, so both peers hold the same per-layer page encoding.
        if parallel.pipeline_parallel_size != 1 and (
            parallel.tensor_parallel_size != 1
            or vllm_config.kv_transfer_config is None
            or not vllm_config.kv_transfer_config.is_kv_producer
            or not _use_per_layer_pool_tags()
        ):
            raise NotImplementedError(
                "Qwen PD pipeline stages produce only, at TP1, with "
                "TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 on both peers"
            )
        if envs.TPU_KV_RESHARD_TRANSPORT != "raiden":
            raise ValueError("Qwen PD requires the Raiden Stage-3 transport")
        if any(
            (
                envs.TPU_RAIDEN_QWEN35_ADMISSION,
                envs.TPU_RAIDEN_KIMIK3_ADMISSION,
                envs.TPU_RAIDEN_GLM_ADMISSION,
                envs.TPU_RAIDEN_DSV4_ADMISSION,
            )
        ):
            raise ValueError("Qwen PD cannot use another model's pool admission")
        super().__init__(vllm_config, role, kv_cache_config)
        # QWEN4_EXP_PD_SYNC_REPACK: the scheduler names the loads that landed
        # on every rank; each rank's staged repack runs in the step that
        # names them (start_load_kv, before the forward), on all ranks alike.
        self._sync_repack = sync_repack_enabled() and not (
            transfer is not None and transfer.is_kv_producer
        )
        self._landed_on_all_ranks: list[str] = []
        self._failed_on_some_rank: list[str] = []

    # --- QWEN4_EXP_PD_SYNC_REPACK -----------------------------------------

    def update_connector_output(self, connector_output: Any) -> None:
        super().update_connector_output(connector_output)
        if not self._sync_repack or self.connector_scheduler is None:
            return
        # vLLM's aggregator reports a receive only once every rank has, and
        # a failure on any rank in failed_recving beside it. A failed load
        # recomputes locally: no rank repacks it, and every rank frees the
        # pages it staged for it in the named step.
        finished = set(connector_output.finished_recving or ())
        failed = set(getattr(connector_output, "failed_recving", None) or ())
        for req_id in sorted(finished - failed):
            self._landed_on_all_ranks.append(req_id)
        for req_id in sorted(finished & failed):
            self._failed_on_some_rank.append(req_id)

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        if self._sync_repack:
            meta.qwen_repack_req_ids = self._landed_on_all_ranks
            meta.qwen_drop_req_ids = self._failed_on_some_rank
            self._landed_on_all_ranks, self._failed_on_some_rank = [], []
        return meta

    def start_load_kv(self, *args, **kwargs) -> None:
        worker = self.connector_worker
        if self._sync_repack and worker is not None:
            # Before this step's forward, so a request admitted in this step
            # reads its repacked pages.
            meta = self._connector_metadata
            worker.repack_landed(
                getattr(meta, "qwen_repack_req_ids", ()),
                drop=getattr(meta, "qwen_drop_req_ids", ()),
            )
        super().start_load_kv(*args, **kwargs)

    @staticmethod
    def _build_stage3_transfer_groups(
        config: KVCacheConfig,
    ) -> tuple[_Stage3TransferGroup, ...]:
        return tuple(
            _Stage3TransferGroup(
                cache_group_index=group.group_index,
                page_tokens=group.block_size,
                request_state=group.extent is TransferExtent.REQUEST_STATE,
                positional_state=group.positional_state,
                speculative_blocks=_speculative_blocks(
                    config.kv_cache_groups[group.group_index]
                )
                if group.extent is TransferExtent.REQUEST_STATE
                else 0,
            )
            for group in build_transfer_groups(config.kv_cache_groups)
        )

    # The worker plans per-group suffix pages for decode-local prefix hits
    # (token groups clip their local prefix, request-state groups move whole).
    supports_stage3_grouped_prefix_loads = True

    @staticmethod
    def _supports_stage3_grouped_transfer() -> bool:
        return True

    @staticmethod
    def _raiden_worker_type() -> type[TPURaidenConnectorWorker]:
        return Qwen4ExpRaidenWorker

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return False

    if _REQUEST_LEVEL_LOAD_FAILURES:

        def get_transfer_results(self, finished_req_ids: set[str]):
            from vllm.distributed.kv_transfer.kv_connector.v1.base import (
                KVConnectorTransferResults,
            )

            worker = self.connector_worker
            assert worker is not None
            sending, recving = worker.get_finished(finished_req_ids)
            return KVConnectorTransferResults(
                finished_sending=set(sending or ()),
                finished_recving=set(recving or ()),
                failed_recving=set(getattr(worker, "last_failed_recving", ())),
            )

        def get_block_ids_with_load_errors(self) -> set[int]:
            # Drained so they do not accumulate; the failure is reported per
            # request by get_transfer_results instead.
            super().get_block_ids_with_load_errors()
            return set()


class Qwen4ExpRaidenWorker(TPURaidenConnectorWorker):
    """Whole-page transport over the shared Stage-3 request lifecycle.

    Native admission binds ``_qwen_page_plan`` after validating the loaded
    model identity and measured layout.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self._qwen_page_plan: QwenPagePlan | None = None
        self._qwen_semantics: dict[str, Any] | None = None
        # QWEN4_EXP_PD_ROUTED_DST (pd_route): set at admission on a pipeline
        # producer and on a staged consumer; None broadcasts as before.
        self._qwen_route = None
        self.last_failed_recving: set[str] = set()
        # QWEN4_EXP_PD_SYNC_REPACK: destination req id -> (uuid, group tables,
        # tokens) of a load whose bytes landed on this rank and whose repack
        # waits for the step the scheduler names it in (repack_landed).
        self._sync_repack = sync_repack_enabled()
        self._sync_on = sync_repack_on
        self._pending_repacks: dict[str, tuple[int, list[list[int]], int]] = {}

    def get_finished(self, finished_req_ids: set[str] | None = None):
        failed_before = set(self._failed_recving)
        sending, recving = super().get_finished(finished_req_ids)
        self.last_failed_recving = set(recving) & (
            failed_before | self._failed_recving
        )
        if self.last_failed_recving:
            logger.warning(
                "Qwen4ExpRaidenWorker rank%d --> KV loads failed for %s; "
                "they will be recomputed locally",
                self.tp_rank,
                sorted(self.last_failed_recving),
            )
        return sending, recving

    def register_runner(self, runner) -> None:
        from .inputs import Qwen4ExpAdapter
        from .pooled_cache import Qwen4ExpPooledCache

        if self._raiden_transfer_engine is not None:
            if self.runner is not runner:
                raise ValueError("Qwen PD cannot replace an admitted model runner")
            return
        adapter = runner.model_adapter
        if not isinstance(adapter, Qwen4ExpAdapter) or not isinstance(
            adapter.cache, Qwen4ExpPooledCache
        ):
            raise ValueError("Qwen PD requires the model-owned native unified pool")
        cache = adapter.cache
        if cache.materialized is None or self.named_kv_caches is None:
            raise ValueError("Qwen PD requires initialized and registered cache owners")
        if set(self.named_kv_caches) != set(cache.materialized.kv_caches) or any(
            self.named_kv_caches[name] is not value
            for name, value in cache.materialized.kv_caches.items()
        ):
            raise ValueError(
                "Qwen PD registered owner bindings differ from live caches"
            )
        if cache.native_config != runner.kv_cache_config:
            raise ValueError("Qwen PD scheduler and materialized cache plans differ")
        if not self._raiden_stage3_enabled():
            raise ValueError("Qwen PD requires the Raiden Stage-3 transport")
        self._validate_stage3_transfer_parallelism(self._raiden_transfer_parallelism())
        if self._staged_consumer():
            # A TP-T decode behind TP1 pipeline stages: Raiden lands the
            # producer's pages in a staging carrier of the producer's geometry
            # and every load is repacked into this rank's pool (pd_staging).
            from .pd_staging import RankSlice, StagedConsumer

            parallel = self.vllm_config.parallel_config
            self._staging = StagedConsumer(
                self.vllm_config,
                cache.layout,
                cache.get_transfer_groups(),
                runner.kv_cache_raw_tensors[0],
                RankSlice(
                    tp_size=int(parallel.tensor_parallel_size),
                    tp_rank=int(self.tp_rank),
                    num_kv_heads=int(self.vllm_config.model_config.hf_text_config.num_key_value_heads),
                    main_dcp=cache.layout.main_dcp,
                ),
                staging_bytes=int(envs.QWEN4_EXP_PD_STAGING_BYTES),
            )
            manifest = self._staging.manifest
            materialize_storages(manifest)
            plan_groups = self._staging.src_groups
            self._qwen_route = self._staging.route
            self._log_hbm_after_staging(runner)
        else:
            self._qwen_route = self._producer_route(cache.layout)
            manifest = build_transfer_manifest(
                cache.native_config, cache.materialized, cache.layout
            )
            verify_storage_binding(
                manifest, self.named_kv_caches, runner.kv_cache_raw_tensors
            )
            materialize_storages(manifest)
            plan_groups = cache.get_transfer_groups()
        semantics = self._collect_pd_semantics(runner)
        # Validate physical geometry before starting any network/native manager.
        measure_transfer_layout(manifest)
        self.runner = runner
        self._qwen_page_plan = QwenPagePlan(plan_groups, manifest)
        self._qwen_semantics = semantics
        self._maybe_host_reshard_store()
        address = _resolved_reshard_controller_address(self.dp_rank)
        if not address:
            raise ValueError("Qwen PD requires a Raiden controller address")
        engine = self._construct_raiden_transfer_engine(
            list(manifest.storages), num_slots=1
        )
        summary = dict(engine.register_pools(manifest.pool_dicts()))
        summary["stage3_registration"] = self._register_raiden_stage3_work_unit(
            engine=engine, manifest=manifest, controller_address=address
        )
        self._raiden_manifest = manifest
        self._raiden_transfer_engine = engine
        self._raiden_admission_summary = summary

    def _producer_route(self, layout):
        """QWEN4_EXP_PD_ROUTED_DST on a TP1 pipeline producer: its staged
        consumer's geometry (decode TP, DCP, block), else None."""
        from types import SimpleNamespace

        from . import pd_route
        from .pd_staging import route_geometry_for

        parsed = pd_route.parse_route_env(envs.QWEN4_EXP_PD_ROUTED_DST)
        if parsed is None:
            return None
        if not (self.is_producer and _engine_is_pipeline(self.vllm_config)):
            raise ValueError(
                "QWEN4_EXP_PD_ROUTED_DST routes a TP1 pipeline producer to a staged "
                "TP>1 decode; this engine is neither"
            )
        tp, dcp, block = parsed
        return route_geometry_for(
            self.vllm_config,
            layout,
            SimpleNamespace(block_size=block, token_bytes=layout.token_bytes),
            SimpleNamespace(tp_size=tp, main_dcp=dcp),
        )

    def _staged_consumer(self) -> bool:
        """A TP>1 consumer of TP1 pipeline stages (per-layer pool tags)."""
        return (
            not self.is_producer and _use_per_layer_pool_tags() and int(self.tp_size) > 1
        )

    def _stage3_reserve_destination(self, destination_req_id, req_meta, tp_group) -> bool:
        staging = getattr(self, "_staging", None)
        if staging is None:
            return True
        ok = staging.reserve(
            int(req_meta.uuid),
            int(req_meta.num_tokens),
            tp_group,
            skip_tokens=int(getattr(req_meta, "skip_tokens", 0) or 0),
        )
        if not ok:
            # The load parks and is re-attempted each step until room frees
            # (or the wait budget ends; the connector logs that fallback).
            logger.debug(
                "Qwen PD staging full (%d pages): req=%s waits", staging.pages, destination_req_id
            )
        return ok

    def _log_hbm_after_staging(self, runner) -> None:
        """One line per rank with the device's HBM after the staging carrier.

        The carrier is allocated here, after the KV pool is sized and after
        the worker's ``Memory statistics`` line, and it is not part of the
        connector HBM reserve; this is the only record of the margin a larger
        ``QWEN4_EXP_PD_STAGING_BYTES`` would eat (XFER L4)."""
        staging = self._staging
        try:
            device = getattr(runner, "device", None) or torch.device("tpu")
            free, limit = torch.accelerator.get_memory_info(device)
            logger.info(
                "Qwen PD staging carrier rank%d: %d pages, %.2f GiB requested | HBM after the "
                "carrier: used %.2f GiB of %.2f GiB, free %.2f GiB",
                self.tp_rank,
                staging.pages,
                int(envs.QWEN4_EXP_PD_STAGING_BYTES) / 2**30,
                (limit - free) / 2**30,
                limit / 2**30,
                free / 2**30,
            )
        except Exception as exc:  # observability only; never block the load path
            logger.info("Qwen PD staging carrier rank%d: HBM unavailable (%s)", self.tp_rank, exc)

    # --- QWEN4_EXP_PD_BATCHED_ADMISSION hooks (distributed/kv_transfer/stage3_admission.py) ---

    def _stage3_admission_free_map(self):
        staging = getattr(self, "_staging", None)
        return None if staging is None else staging.free

    def _stage3_reserve_destination_reduced(self, req_meta, reduced_free) -> bool:
        staging = getattr(self, "_staging", None)
        if staging is None:
            return True
        return staging.reserve(
            int(req_meta.uuid),
            int(req_meta.num_tokens),
            skip_tokens=int(getattr(req_meta, "skip_tokens", 0) or 0),
            reduced_free=reduced_free,
        )

    def _stage3_destination_pages_needed(self, req_meta) -> int:
        staging = getattr(self, "_staging", None)
        if staging is None:
            return 0
        return staging.pages_needed(
            int(req_meta.uuid),
            int(req_meta.num_tokens),
            skip_tokens=int(getattr(req_meta, "skip_tokens", 0) or 0),
        )

    def _stage3_admission_snapshot(self):
        staging = getattr(self, "_staging", None)
        if staging is None:
            return None
        return (staging.free.clone(), dict(staging.reservations), dict(staging.skips))

    def _stage3_admission_restore(self, saved) -> None:
        staging = getattr(self, "_staging", None)
        if staging is None or saved is None:
            return
        free, reservations, skips = saved
        staging.free.copy_(free)
        staging.reservations.clear()
        staging.reservations.update(reservations)
        staging.skips.clear()
        staging.skips.update(skips)

    def _stage3_finish_destination(self, destination_req_id, ok) -> None:
        staging = getattr(self, "_staging", None)
        if staging is None:
            return
        if not ok:
            # Cleaned up (aborted, failed or finished) before the scheduler
            # named it for the synchronized repack: free its staging pages.
            pending = self._pending_repacks.pop(destination_req_id, None)
            if pending is not None:
                staging.release(pending[0])
                return
        req_meta = self._stage3_submitted_load_metas.get(destination_req_id)
        if req_meta is None or int(req_meta.uuid) not in staging.reservations:
            return
        uuid = int(req_meta.uuid)
        if ok and self._sync_repack and self._sync_on():
            # Keep the staged pages; every rank repacks in the step the
            # scheduler names this load in, after it landed on all of them.
            self._pending_repacks[destination_req_id] = (
                uuid,
                [list(table) for table in req_meta.group_block_ids],
                int(req_meta.num_tokens),
            )
            return
        try:
            if ok:
                staging.repack(uuid, req_meta.group_block_ids, int(req_meta.num_tokens))
        finally:
            staging.release(uuid)

    def repack_landed(self, req_ids, drop=()) -> None:
        """Repack the named loads (QWEN4_EXP_PD_SYNC_REPACK). The scheduler
        names a load only once every rank reported it landed, and every rank
        receives the same names in the same step. ``drop``: loads that
        failed on some rank; their staged pages are freed unrepacked."""
        staging = getattr(self, "_staging", None)
        if staging is None:
            return
        for req_id in drop:
            pending = self._pending_repacks.pop(req_id, None)
            if pending is not None:
                staging.release(pending[0])
        names = list(req_ids)
        if not self._sync_on():
            # An off period of the A/B: loads deferred in the last on period
            # repack now, before any of them can be scheduled.
            names += [r for r in self._pending_repacks if r not in names]
        for req_id in names:
            pending = self._pending_repacks.pop(req_id, None)
            if pending is None:
                # Repacked on landing (an off period of the A/B), never staged
                # on this rank, or cleaned up first: not this repack's pages.
                if int(envs.QWEN4_EXP_PD_SYNC_REPACK_AB_S) <= 0:
                    logger.warning(
                        "Qwen4ExpRaidenWorker rank%d --> no staged load to repack for %s",
                        self.tp_rank,
                        req_id,
                    )
                continue
            uuid, tables, num_tokens = pending
            try:
                staging.repack(uuid, tables, num_tokens)
            finally:
                staging.release(uuid)

    def _ensure_raiden_transfer_engine(self):
        if self._raiden_transfer_engine is None:
            raise RuntimeError("Qwen PD native admission must finish before transfer")
        return self._raiden_transfer_engine

    def _collect_pd_semantics(self, runner) -> dict[str, Any]:
        model = runner.model_adapter.cache.model
        if model.pd_gdn_qk_pair_layout != envs.TPU_GDN_CONV_QK_PAIR_LAYOUT:
            raise ValueError("Qwen PD GDN layout changed since model construction")
        scales = []
        # This pipeline stage's own layers (all of them without PP).
        for index in model.model.stage:
            layer = model.model.layers[index]
            if layer.layer_type == "full_attention":
                attention = layer.self_attn
                k, v = attention._k_scale_float, attention._v_scale_float
                if not all(math.isfinite(x) and x > 0 for x in (k, v)):
                    raise ValueError(
                        "Qwen PD requires finite positive loaded FP8 scales"
                    )
                if k != float(attention._k_scale_cpu.item()) or v != float(
                    attention._v_scale_cpu.item()
                ):
                    raise ValueError(
                        "Qwen PD FP8 kernel scales differ from loaded buffers"
                    )
                scales.append((index, k, v))
        parallel = self.vllm_config.parallel_config
        group = get_ep_group() if parallel.enable_expert_parallel else get_tp_group()
        config = model.config.to_dict()
        for name in ("_name_or_path", "_commit_hash", "transformers_version"):
            config.pop(name, None)
        record = dict(
            rank=group.rank_in_group,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            checkpoint=model.pd_loaded_weight_identity,
            ple_checkpoint=model.pd_loaded_ple_identity,
            config=config,
            kv_scales=scales,
            state_flags=dict(
                gdn_qk_pair=model.pd_gdn_qk_pair_layout,
                expert_parallel=parallel.enable_expert_parallel,
                ssm_dtype=str(self.vllm_config.cache_config.mamba_ssm_cache_dtype),
                cp=1,
                pp=1,
                mtp=False,
                # Prefix caching is a per-leg choice: the pages that cross are
                # the same encoding either way (a positional state table ships
                # its running-state page), so it is not part of the identity.
            ),
        )
        records: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(records, record, group=group.cpu_group)
        return canonical_rank_identity(records, world_size=group.world_size)

    def _measure_stage3_layout(self, manifest) -> tuple[str, dict[str, Any], int]:
        if self._qwen_semantics is None:
            raise RuntimeError("Qwen PD loaded-state identity has not been admitted")
        _, layout = measure_transfer_layout(manifest)
        if self._qwen_route is not None:
            if not _use_per_layer_pool_tags():
                raise ValueError("QWEN4_EXP_PD_ROUTED_DST needs TPU_RAIDEN_POOL_TAGS_PER_LAYER=1")
            payload = dict(
                layout=_routed_layout(_owner_invariant_layout(layout), self._qwen_route),
                semantics=_owner_invariant_semantics(self._qwen_semantics),
            )
        elif _use_per_layer_pool_tags():
            payload = dict(
                layout=_owner_invariant_layout(layout),
                semantics=_owner_invariant_semantics(self._qwen_semantics),
            )
        else:
            payload = dict(layout=layout, semantics=self._qwen_semantics)
        token_sizes = {
            g.block_size
            for g in self._page_plan().groups
            if g.extent is TransferExtent.TOKEN_PAGES
        }
        if len(token_sizes) == 2 and max(token_sizes) % min(token_sizes) == 0:
            # A PCP producer's (or its decode's staged) split main KV: entries of
            # P pages over P ranks, each rank's page as long as every other's.
            token_sizes = {min(token_sizes)}
        if len(token_sizes) != 1:
            raise ValueError("Qwen PD requires one native token-page size")
        fingerprint = canonical_layout_fingerprint(payload)
        # Per-field digests, so a peer mismatch names the field that differs.
        fields = {
            f"layout.{k}": v for k, v in payload["layout"].items()
        } | {f"semantics.{k}": v for k, v in payload["semantics"].items()}
        logger.info(
            "Qwen PD rank%d layout fingerprint %s fields %s carriers %s",
            self.tp_rank,
            fingerprint[:12],
            {k: canonical_layout_fingerprint({"v": v})[:12] for k, v in fields.items()},
            json.dumps(payload["layout"].get("carriers"), sort_keys=True)[:600],
        )
        return fingerprint, payload, token_sizes.pop()

    def _validate_stage3_transfer_parallelism(self, transfer_parallelism: int) -> None:
        parallel = self.vllm_config.parallel_config
        pp = int(parallel.pipeline_parallel_size or 1)
        pcp = int(parallel.prefill_context_parallel_size or 1)
        if pp > 1:
            # A pipeline producer: one transfer unit per TP1 stage, and per PCP
            # rank of a stage (tpu_connector._local_raiden_transfer_rank).
            units = (
                self.is_producer and self.tp_size == 1 and transfer_parallelism == pp * pcp
            )
        elif _use_per_layer_pool_tags() and not self.is_producer:
            # A consumer of pipeline stages pulls every stage's layers: whole
            # pages at TP1, through the staging repack at TP>1.
            units = True
        else:
            # Matching TP: producer rank k ships its pages to decode rank k.
            units = transfer_parallelism == self.tp_size
        if (
            not units
            or (pcp != 1 and not (self.is_producer and self.tp_size == 1))
            or parallel.decode_context_parallel_size != 1
            or (
                self.vllm_config.speculative_config is not None
                and (self.is_producer or self.tp_size == 1)
            )
        ):
            raise ValueError(
                "Qwen PD requires matching TP (or a TP1 pipeline producer with "
                "a TP1 consumer), CP1 and MTP off"
            )

    def _raiden_interleave_tokens(
        self, page_tokens: int, transfer_parallelism: int
    ) -> int:
        self._validate_stage3_transfer_parallelism(transfer_parallelism)
        return page_tokens

    @staticmethod
    def _uses_stage3_grouped_pages() -> bool:
        return True

    def _local_raiden_transfer_rank(self) -> int:
        if _engine_is_pipeline(self.vllm_config):
            # The stage index (TP1 stages, no PCP).
            return super()._local_raiden_transfer_rank()
        return int(self.tp_rank)

    def _page_plan(self) -> "QwenPagePlan":
        if self._qwen_page_plan is None:
            raise RuntimeError("Qwen PD native page admission has not completed")
        self._validate_stage3_transfer_parallelism(self._raiden_transfer_parallelism())
        if not _engine_is_pipeline(self.vllm_config) and (
            self._local_raiden_transfer_rank() != self.tp_rank
        ):
            raise ValueError("Qwen PD requires matching TP rank ownership")
        return self._qwen_page_plan

    def _stage3_grouped_spans(
        self, req_meta: SendMeta, num_tokens: int
    ) -> tuple[tuple[tuple[int, ...], ...], list[PoolSpanRegistration]]:
        plan = self._page_plan()
        # A pipeline stage behind a larger scheduler block-ID space reads its
        # own groups' pages through the stage map; the request's tables, the
        # controller's block list and every replay check stay in block IDs.
        translate = getattr(
            getattr(self, "runner", None), "qwen4_stage_page_map", None
        )
        translate = None if translate is None else translate.lookup
        pages = plan.request_pages(req_meta.group_block_ids, num_tokens, translate)
        if tuple(req_meta.local_block_ids) != pages[0]:
            raise ValueError("Qwen PD primary pages differ from the full group table")
        # A producer serving a caching decode declares its token pages in the
        # request-global destination byte space (dst_space_version=1), page i
        # at i * stride, which the controller splits at destination pages and
        # can clip at a decode-local prefix (dst_skip_bytes). Request-state
        # pages stay page-indexed (version 0) and always move whole.
        clippable = int(envs.TPU_RAIDEN_HANDOFF_ALIGN_TOKENS) > 0
        # An unpacked compressed-key page holds one layer's keys for the page's
        # tokens in its first block_size / 128 kernel pages (12% of a TP1
        # stage's 3200-token page); the rest is padding. Send only the keys.
        from .pd_manifest import compressed_live_bytes

        compressed_bytes = compressed_live_bytes(self.runner.model_adapter.cache.layout)
        trim_compressed = compressed_bytes is not None
        if self._qwen_route is not None:
            return pages, self._routed_spans(plan, pages, num_tokens, translate, compressed_bytes)
        spans = []
        for pool, group_index in plan.pool_groups():
            ids = pages[group_index]
            if translate is not None:
                ids = translate(group_index, ids)
            size_bytes = (
                compressed_bytes
                if trim_compressed and pool.layer_name.endswith(".compressed_k")
                else pool.block_stride_bytes
            )
            global_space = (
                clippable
                and plan.groups[group_index].extent is TransferExtent.TOKEN_PAGES
            )
            spans.append(
                PoolSpanRegistration(
                    tag=pool.tag,
                    block_ids=ids,
                    spans=tuple(
                        PoolByteSpan(
                            src_block_ordinal=index,
                            src_offset_bytes=0,
                            dst_block_index=0 if global_space else index,
                            # The request-global destination space counts
                            # destination pages by their live bytes (the
                            # controller splits it, and checks dst_skip_bytes,
                            # in those units): a trimmed compressed-key page
                            # is 409,600 bytes there, not its 3.2 MB stride.
                            dst_offset_bytes=(
                                index * size_bytes if global_space else 0
                            ),
                            size_bytes=size_bytes,
                            # A pipeline stage's pages go to every decode unit
                            # (a TP>1 decode repacks its slice); matching TP
                            # sends rank k to rank k.
                            dst_unit_ordinal=None
                            if _engine_is_pipeline(self.vllm_config)
                            else self.tp_rank,
                        )
                        for index in range(len(ids))
                    ),
                    declared_bytes=len(ids) * size_bytes,
                    dst_space_version=1 if global_space else 0,
                )
            )
        return pages, spans

    def _routed_spans(self, plan, pages, num_tokens, translate, compressed_bytes):
        """QWEN4_EXP_PD_ROUTED_DST: each destination rank's slice, at its
        compact rank-local offsets (pd_route; ``dst_space_version=1``)."""
        from . import pd_route
        from .pd_staging import owner_kind

        route = self._qwen_route
        if compressed_bytes is not None and compressed_bytes != (
            route.pairs_per_src_page * pd_route.ROW_BYTES
        ):
            raise ValueError("routed compressed keys need one pair row per 2*ratio tokens")
        registrations = []
        for pool, group_index in plan.pool_groups():
            ids = pages[group_index]
            if translate is not None:
                ids = translate(group_index, ids)
            routed = pd_route.producer_spans(
                owner_kind(pool.layer_name), num_tokens, len(ids), route
            )
            spans = tuple(
                PoolByteSpan(
                    src_block_ordinal=src,
                    src_offset_bytes=src_off,
                    dst_block_index=0,
                    dst_offset_bytes=dst_off,
                    size_bytes=size,
                    dst_unit_ordinal=unit,
                )
                for src, src_off, dst_off, size, unit in routed
            )
            registrations.append(
                PoolSpanRegistration(
                    tag=pool.tag,
                    block_ids=ids,
                    spans=spans,
                    declared_bytes=sum(span.size_bytes for span in spans),
                    dst_space_version=1,
                )
            )
        return registrations

    def _stage3_grouped_destination_pages(
        self, req_meta: _Stage3LoadMeta, num_tokens: int
    ) -> tuple[list[str], list[int], list[int]]:
        """Destination pages per pool tag, in plan order.

        A prefix-aware load (``skip_tokens`` > 0: the decode-local prefix
        cache already holds that prefix of every token group) lists only each
        token group's suffix pages, which the transfer fills from the clipped
        source (``_stage3_dst_skip_bytes``); request-state pages move whole.
        The adopted prefix pages are shared cache pages and never targets.
        """
        plan = self._page_plan()
        if req_meta.src_parallelism != self._raiden_transfer_parallelism():
            raise ValueError("Qwen PD needs the admitted source units")
        skip = int(req_meta.skip_tokens or 0)
        if skip and not self._prefix_loads_supported():
            raise ValueError("Qwen PD prefix-aware loads are not enabled on this rank")
        staging = getattr(self, "_staging", None)
        if staging is not None:
            return staging.destination(int(req_meta.uuid))
        pages = plan.request_pages(req_meta.group_block_ids, num_tokens)
        if tuple(req_meta.local_block_ids) != pages[0]:
            raise ValueError("Qwen PD primary pages differ from the full group table")
        tags, blocks, counts = [], [], []
        for pool, group_index in plan.pool_groups():
            ids = suffix_pages(plan.groups[group_index], pages[group_index], skip)
            tags.append(pool.tag)
            blocks.extend(ids)
            counts.append(len(ids))
        return tags, blocks, counts

    @staticmethod
    def _prefix_loads_supported() -> bool:
        from vllm_torchtpu.distributed.kv_transfer.tpu_connector import (
            _prefix_aware_load_enabled,
        )

        return _prefix_aware_load_enabled()

    def _stage3_dst_skip_bytes(
        self, req_meta: Any, skip_tokens: int, num_tags: int, fa_tag_count: int
    ) -> list[int]:
        """Per-tag clips of a prefix-aware load, in the order of
        ``_stage3_grouped_destination_pages``: each token pool skips the source
        pages wholly inside the local prefix, request-state pools nothing."""
        staging = getattr(self, "_staging", None)
        if staging is not None:
            clips = staging.skip_bytes(int(req_meta.uuid))
        else:
            plan = self._page_plan()
            clips = [
                skipped_source_pages(plan.groups[group_index], skip_tokens)
                * _live_page_bytes(pool)
                for pool, group_index in plan.pool_groups()
            ]
        if len(clips) != num_tags or num_tags != fa_tag_count:
            raise ValueError(
                "Qwen PD clips must cover every grouped transfer tag: "
                f"clips={len(clips)}, tags={num_tags}, grouped={fa_tag_count}"
            )
        return clips


def _live_page_bytes(pool) -> int:
    """A pool page's live bytes: its single leading region's unit (a trimmed
    compressed-key page's keys), else the whole stride. The request-global
    destination byte space and its dst_skip_bytes clips are in these units."""
    regions = getattr(pool, "regions", ()) or ()
    if len(regions) == 1 and getattr(regions[0], "num_units", 1) == 1:
        return int(regions[0].unit_bytes)
    return int(pool.block_stride_bytes)


def _speculative_blocks(group) -> int:
    """Speculative state checkpoints of a request-state cache group."""
    spec = group.kv_cache_spec
    specs = (
        tuple(spec.kv_cache_specs.values())
        if hasattr(spec, "kv_cache_specs")
        else (spec,)
    )
    depths = {int(getattr(s, "num_speculative_blocks", 0) or 0) for s in specs}
    if len(depths) != 1:
        raise ValueError("A Qwen PD state group needs one speculative depth")
    return depths.pop()


def _owner_invariant_layout(layout: dict[str, Any]) -> dict[str, Any]:
    """The page encoding without the owner list or carrier count.

    Under per-layer tags the peers hold different owner subsets of one
    whole-model plan (a pipeline stage its layers, the decode every layer):
    carriers and tags differ, while every page must encode the same way. The
    controller pairs pools by tag; this pins the encoding they share.
    """

    def unique(items):
        encoded = {json.dumps(item, sort_keys=True) for item in items}
        return [json.loads(item) for item in sorted(encoded)]

    return dict(
        schema="qwen4-whole-fp8-page-layout-v2",
        torch_tpu=layout["torch_tpu"],
        libtpu=layout["libtpu"],
        carriers=unique(layout["carriers"]),
        # Compressed-key pools are left out: merged keys
        # (QWEN4_EXP_MERGE_COMPRESSED_K) are listed by a PCP producer's rank 0
        # only, so its other units would otherwise encode different pages;
        # unmerged keys share the page stride the other pools already pin.
        pages=unique(
            dict(block_stride_bytes=p["block_stride_bytes"], dtype_tag=p["dtype_tag"])
            for p in layout["pools"]
            # Split keys are tagged "<owner>.compressed_k.pcp<r>".
            if ".compressed_k" not in p["tag"]
        ),
    )


def _routed_layout(layout: dict[str, Any], route) -> dict[str, Any]:
    """The routed page encoding (QWEN4_EXP_PD_ROUTED_DST): the producer's
    pages and the decode's compact staging pages differ in rows per page, so
    both describe a carrier per 1,024-byte row and name the routing geometry
    instead; a peer routing for a different decode fails closed."""

    def per_row(carrier):
        carrier = dict(carrier)
        shape = list(carrier["shape_per_kernel_page"])
        carrier["shape_per_kernel_page"] = [1] + shape[1:]
        return carrier

    encoded = {json.dumps(per_row(c), sort_keys=True) for c in layout["carriers"]}
    tags = {json.dumps({"dtype_tag": p["dtype_tag"]}, sort_keys=True) for p in layout["pages"]}
    return dict(
        schema="qwen4-routed-fp8-row-layout-v1",
        torch_tpu=layout["torch_tpu"],
        libtpu=layout["libtpu"],
        carriers=[json.loads(c) for c in sorted(encoded)],
        pages=[json.loads(t) for t in sorted(tags)],
        routed=route.key(),
    )


def _owner_invariant_semantics(semantics: dict[str, Any]) -> dict[str, Any]:
    """The loaded model state, without rank ownership.

    A pipeline stage and a decode differ in rank count, TP size, expert
    parallelism and the layers whose KV scales they hold; the scales come from
    the checkpoint, whose digest is compared, and the page geometry the TP size
    implies is in the layout half of the fingerprint (a staged TP>1 decode
    presents the producer's geometry).
    """
    ranks = semantics["ranks"]
    checkpoints = sorted({r["checkpoint"] for r in ranks})
    if len(checkpoints) != 1:
        raise ValueError("Qwen PD ranks loaded different checkpoints")
    first = ranks[0]
    flags = {k: v for k, v in first["state_flags"].items() if k != "expert_parallel"}
    return dict(
        schema="qwen4-loaded-state-v2",
        checkpoint=checkpoints[0],
        ple_checkpoint=first["ple_checkpoint"],
        config=first["config"],
        state_flags=flags,
    )


@dataclass(frozen=True)
class QwenPagePlan:
    groups: tuple[TransferGroup, ...]
    manifest: PoolManifest

    def pool_groups(self) -> tuple[tuple[PoolEntry, int], ...]:
        owners = {
            name: group.group_index
            for group in self.groups
            for name in group.layer_names
        }
        from .pd_manifest import merged_key_owner

        # QWEN4_EXP_MERGE_COMPRESSED_K: a layer's keys are a pool of their own
        # in its main-KV carrier, on the main-KV group's block IDs.
        merged = {}
        for pool in self.manifest.pools:
            main = merged_key_owner(pool.layer_name, owners)
            if main is not None:
                merged[pool.layer_name] = owners[main]
        names = [pool.layer_name for pool in self.manifest.pools if pool.layer_name not in merged]
        tags = [pool.tag for pool in self.manifest.pools]
        from vllm_torchtpu.distributed.pcp import get_pcp_rank

        # A PCP rank past 0 lists only its split main-KV owners
        # (pd_manifest.build_transfer_manifest); every other unit lists all.
        covered = (
            set(names) <= set(owners) if get_pcp_rank() > 0 else set(names) == set(owners)
        )
        if len(names) != len(set(names)) or len(tags) != len(set(tags)) or not covered:
            raise ValueError("Qwen PD page plan must cover every owner exactly once")
        owners = owners | merged
        return tuple((pool, owners[pool.layer_name]) for pool in self.manifest.pools)

    def request_pages(
        self,
        tables: list[list[int]] | None,
        num_tokens: int,
        translate=None,
    ) -> tuple[tuple[int, ...], ...]:
        """The committed block IDs per group; ``translate(group, ids)``, when
        given, maps a group's IDs to the pages checked against the carrier."""
        if tables is None or num_tokens <= 0:
            raise ValueError("Qwen PD requires complete committed group tables")
        # The scheduler verifies actual computed == published before emitting
        # metadata; workers consume that committed record, not a fresh model count.
        pages = handoff_block_ids(
            self.groups, tables, committed_tokens=num_tokens, computed_tokens=num_tokens
        )
        occupied: set[tuple[int, int]] = set()
        for pool, group_index in self.pool_groups():
            ids = pages[group_index]
            if translate is not None:
                ids = translate(group_index, ids)
            for page in ids:
                if page >= pool.num_blocks:
                    raise ValueError("Qwen PD page is outside its native carrier")
                # Merged keys share their main-KV pages' IDs in another byte
                # region of the carrier (base offset past the pages).
                key = (pool.storage_index, pool.base_offset_bytes, page)
                if key in occupied:
                    raise ValueError("Qwen PD groups overlap a physical carrier page")
                occupied.add(key)
        return pages
