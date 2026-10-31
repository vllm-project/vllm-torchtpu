# SPDX-License-Identifier: Apache-2.0
"""Model-owned Stage-3 page planning for Qwen's hybrid caches.

Native admission binds the loaded model identity and measured page layout.
This connector does not borrow another model's native pool admission.
"""

import json
import math
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
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

from .pd import TransferExtent, TransferGroup, build_transfer_groups, handoff_block_ids
from .pd_identity import canonical_rank_identity
from .pd_manifest import build_transfer_manifest, measure_transfer_layout

logger = init_logger(__name__)


class Qwen4ExpRaidenConnector(TPURaidenConnector):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        parallel = vllm_config.parallel_config
        cache = vllm_config.cache_config
        if (
            parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or vllm_config.speculative_config is not None
            or cache.enable_prefix_caching
        ):
            raise NotImplementedError("Qwen PD first requires CP1, APC and MTP off")
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

    @staticmethod
    def _build_stage3_transfer_groups(
        config: KVCacheConfig,
    ) -> tuple[_Stage3TransferGroup, ...]:
        return tuple(
            _Stage3TransferGroup(
                cache_group_index=group.group_index,
                page_tokens=group.block_size,
                request_state=group.extent is TransferExtent.REQUEST_STATE,
            )
            for group in build_transfer_groups(config.kv_cache_groups)
        )

    @staticmethod
    def _supports_stage3_grouped_transfer() -> bool:
        return True

    @staticmethod
    def _raiden_worker_type() -> type[TPURaidenConnectorWorker]:
        return Qwen4ExpRaidenWorker

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        return False


class Qwen4ExpRaidenWorker(TPURaidenConnectorWorker):
    """Whole-page transport over the shared Stage-3 request lifecycle.

    Native admission binds ``_qwen_page_plan`` after validating the loaded
    model identity and measured layout.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self._qwen_page_plan: QwenPagePlan | None = None
        self._qwen_semantics: dict[str, Any] | None = None

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
        else:
            manifest = build_transfer_manifest(cache.native_config, cache.materialized)
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

    def _staged_consumer(self) -> bool:
        """A TP>1 consumer of TP1 pipeline stages (per-layer pool tags)."""
        return (
            not self.is_producer and _use_per_layer_pool_tags() and int(self.tp_size) > 1
        )

    def _stage3_reserve_destination(self, destination_req_id, req_meta, tp_group) -> bool:
        staging = getattr(self, "_staging", None)
        if staging is None:
            return True
        ok = staging.reserve(int(req_meta.uuid), int(req_meta.num_tokens), tp_group)
        if not ok:
            logger.warning(
                "Qwen PD staging full (%d pages): req=%s recomputes on the decode",
                staging.pages,
                destination_req_id,
            )
        return ok

    def _stage3_finish_destination(self, destination_req_id, ok) -> None:
        staging = getattr(self, "_staging", None)
        if staging is None:
            return
        req_meta = self._stage3_submitted_load_metas.get(destination_req_id)
        if req_meta is None or int(req_meta.uuid) not in staging.reservations:
            return
        uuid = int(req_meta.uuid)
        try:
            if ok:
                staging.repack(uuid, req_meta.group_block_ids, int(req_meta.num_tokens))
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
                apc=False,
            ),
        )
        records: list[Any] = [None] * group.world_size
        torch.distributed.all_gather_object(records, record, group=group.cpu_group)
        return canonical_rank_identity(records, world_size=group.world_size)

    def _measure_stage3_layout(self, manifest) -> tuple[str, dict[str, Any], int]:
        if self._qwen_semantics is None:
            raise RuntimeError("Qwen PD loaded-state identity has not been admitted")
        _, layout = measure_transfer_layout(manifest)
        if _use_per_layer_pool_tags():
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
        if len(token_sizes) != 1:
            raise ValueError("Qwen PD requires one native token-page size")
        return canonical_layout_fingerprint(payload), payload, token_sizes.pop()

    def _validate_stage3_transfer_parallelism(self, transfer_parallelism: int) -> None:
        parallel = self.vllm_config.parallel_config
        pp = int(parallel.pipeline_parallel_size or 1)
        if pp > 1:
            # A pipeline producer: one transfer unit per TP1 stage.
            units = self.is_producer and self.tp_size == 1 and transfer_parallelism == pp
        elif _use_per_layer_pool_tags() and not self.is_producer:
            # A consumer of pipeline stages pulls every stage's layers: whole
            # pages at TP1, through the staging repack at TP>1.
            units = True
        else:
            # Matching TP: producer rank k ships its pages to decode rank k.
            units = transfer_parallelism == self.tp_size
        if (
            not units
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or self.vllm_config.speculative_config is not None
            or self.vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError(
                "Qwen PD requires matching TP (or a TP1 pipeline producer with "
                "a TP1 consumer), CP1 and APC/MTP off"
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
        pages = plan.request_pages(req_meta.group_block_ids, num_tokens)
        if tuple(req_meta.local_block_ids) != pages[0]:
            raise ValueError("Qwen PD primary pages differ from the full group table")
        spans = []
        for pool, group_index in plan.pool_groups():
            ids = pages[group_index]
            spans.append(
                PoolSpanRegistration(
                    tag=pool.tag,
                    block_ids=ids,
                    spans=tuple(
                        PoolByteSpan(
                            src_block_ordinal=index,
                            src_offset_bytes=0,
                            dst_block_index=index,
                            dst_offset_bytes=0,
                            size_bytes=pool.block_stride_bytes,
                            # A pipeline stage's pages go to every decode unit
                            # (a TP>1 decode repacks its slice); matching TP
                            # sends rank k to rank k.
                            dst_unit_ordinal=None
                            if _engine_is_pipeline(self.vllm_config)
                            else self.tp_rank,
                        )
                        for index in range(len(ids))
                    ),
                    declared_bytes=len(ids) * pool.block_stride_bytes,
                )
            )
        return pages, spans

    def _stage3_grouped_destination_pages(
        self, req_meta: _Stage3LoadMeta, num_tokens: int
    ) -> tuple[list[str], list[int], list[int]]:
        plan = self._page_plan()
        if req_meta.skip_tokens or (
            req_meta.src_parallelism != self._raiden_transfer_parallelism()
        ):
            raise ValueError("Qwen PD needs a full prefix and the admitted source units")
        staging = getattr(self, "_staging", None)
        if staging is not None:
            return staging.destination(int(req_meta.uuid))
        pages = plan.request_pages(req_meta.group_block_ids, num_tokens)
        if tuple(req_meta.local_block_ids) != pages[0]:
            raise ValueError("Qwen PD primary pages differ from the full group table")
        tags, blocks, counts = [], [], []
        for pool, group_index in plan.pool_groups():
            ids = pages[group_index]
            tags.append(pool.tag)
            blocks.extend(ids)
            counts.append(len(ids))
        return tags, blocks, counts


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
        pages=unique(
            dict(block_stride_bytes=p["block_stride_bytes"], dtype_tag=p["dtype_tag"])
            for p in layout["pools"]
        ),
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
        names = [pool.layer_name for pool in self.manifest.pools]
        tags = [pool.tag for pool in self.manifest.pools]
        if (
            len(names) != len(set(names))
            or len(tags) != len(set(tags))
            or set(names) != set(owners)
        ):
            raise ValueError("Qwen PD page plan must cover every owner exactly once")
        return tuple((pool, owners[pool.layer_name]) for pool in self.manifest.pools)

    def request_pages(
        self, tables: list[list[int]] | None, num_tokens: int
    ) -> tuple[tuple[int, ...], ...]:
        if tables is None or num_tokens <= 0:
            raise ValueError("Qwen PD requires complete committed group tables")
        # The scheduler verifies actual computed == published before emitting
        # metadata; workers consume that committed record, not a fresh model count.
        pages = handoff_block_ids(
            self.groups, tables, committed_tokens=num_tokens, computed_tokens=num_tokens
        )
        occupied: set[tuple[int, int]] = set()
        for pool, group_index in self.pool_groups():
            for page in pages[group_index]:
                if page >= pool.num_blocks:
                    raise ValueError("Qwen PD page is outside its native carrier")
                key = (pool.storage_index, page)
                if key in occupied:
                    raise ValueError("Qwen PD groups overlap a physical carrier page")
                occupied.add(key)
        return pages
