# SPDX-License-Identifier: Apache-2.0
"""Admission of the model's KV cache into the Raiden transfer engine.

``_RaidenAdmissionMixin`` holds the ``TPURaidenConnectorWorker`` methods that
run once, when the runner hands over its KV caches: measuring the admitted
layout and page geometry, validating the one supported topology for Qwen3.5
or GLM, registering each cache tensor as a Raiden pool with bounded host
staging hints, registering this worker as a Stage-3 work unit with its
cluster controller, and optionally hosting the reshard store in-engine.
"""

import time
from typing import Any

from vllm.v1.kv_cache_interface import MambaSpec

import vllm_torchtpu.distributed.utils as dist_utils
from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.raiden_config import (
    _STAGE3_GLM_TRANSFER_POOL_TAGS, _engine_is_pipeline,
    _pipeline_parallel_size, _reshard_advertise_host,
    _reshard_dispatch_address, _reshard_service_address, _reshard_service_port,
    _reshard_store_mode, _resolved_reshard_controller_address,
    _use_per_layer_pool_tags, _use_raiden_glm_admission,
    stage3_fa_raiden_id_fields)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class _RaidenAdmissionMixin:
    """Startup-time Raiden admission for ``TPURaidenConnectorWorker``.

    Mixin: no state of its own. The attributes it fills are declared in
    ``TPURaidenConnectorWorker.__init__``, except ``_stage3_glm_geometry``,
    which GLM admission creates for the producer's GLM registration; the
    engine construction, rank and identity helpers it calls live on the
    worker itself.
    """

    def _admit_raiden_qwen35_kv_cache(self, runner: TPUModelRunner) -> None:
        """Constructs the v1 Raiden engine over the explicit Qwen3.5 pools."""
        if self._raiden_transfer_engine is not None:
            return

        stage3_enabled = self._raiden_stage3_enabled()
        controller_address = ""
        if stage3_enabled:
            self._maybe_host_reshard_store()
            controller_address = _resolved_reshard_controller_address(
                self.dp_rank)
            if not controller_address:
                raise ValueError(
                    "TPU_RAIDEN_CONTROLLER_ADDRESS is required when "
                    "TPU_KV_RESHARD_TRANSPORT=raiden")

        from vllm_torchtpu.distributed.kv_transfer.raiden import \
            pool_manifest as rpm

        topology = self._raiden_qwen35_admission_topology()
        role = "kv_producer" if self.is_producer else "kv_consumer"
        named_kv_caches = self.named_kv_caches
        if not named_kv_caches:
            raise ValueError("Raiden pool admission requires "
                             "register_kv_caches() before register_runner()")
        kv_cache_groups = tuple(runner.kv_cache_config.kv_cache_groups or ())
        raw_tensors = tuple(runner.kv_cache_raw_tensors or ())

        mamba_group_ordinal_by_layer = None
        if stage3_enabled:
            group_ordinals: dict[str, int] = {}
            ordinal = 0
            for group in kv_cache_groups:
                if not isinstance(group.kv_cache_spec, MambaSpec):
                    continue
                for layer_name in group.layer_names:
                    group_ordinals[layer_name] = ordinal
                ordinal += 1
            if group_ordinals:
                mamba_group_ordinal_by_layer = group_ordinals
                self._stage3_state_group_count = ordinal
        manifest = rpm.build_qwen35_pool_manifest(
            named_kv_caches=named_kv_caches,
            kv_cache_groups=kv_cache_groups,
            raw_tensors=raw_tensors,
            mamba_group_ordinal_by_layer=mamba_group_ordinal_by_layer,
            per_layer_tags=_use_per_layer_pool_tags(),
            gdn_geometry=rpm.GdnHeadGeometry(
                local_key_heads=self._local_gdn_head_count(
                    self._model_config_int("linear_num_key_heads", 0)),
                local_value_heads=self._local_gdn_head_count(
                    self._model_config_int("linear_num_value_heads", 0)),
                key_head_dim=self._model_config_int("linear_key_head_dim", 1),
                value_head_dim=self._model_config_int(
                    "linear_value_head_dim",
                    self._model_config_int("linear_key_head_dim", 1)),
                conv_kernel_size=self._model_config_int(
                    "linear_conv_kernel_dim", 4),
            ),
        )
        # Hard-fail before manager construction if the pools point at storage
        # that the model kernels do not actually use.
        verified_storages = rpm.verify_storage_binding(
            manifest=manifest,
            named_kv_caches=named_kv_caches,
            raw_tensors=raw_tensors,
        )
        rpm.materialize_storages(manifest)

        storages = list(manifest.storages)
        engine = self._construct_raiden_transfer_engine(storages, num_slots=1)
        logger.info(
            "TPURaidenConnectorWorker rank%d --> registering %d pools with transfer engine",
            self.tp_rank, len(manifest.pools))
        summary = self._register_raiden_pools(engine, manifest,
                                              kv_cache_groups)
        logger.info(
            "TPURaidenConnectorWorker rank%d --> transfer engine pools registered",
            self.tp_rank)

        if stage3_enabled:
            logger.info(
                "TPURaidenConnectorWorker rank%d --> registering stage3 work unit with %s",
                self.tp_rank, controller_address)
            registration = self._register_raiden_stage3_work_unit(
                engine=engine,
                manifest=manifest,
                controller_address=controller_address,
            )
            summary["stage3_registration"] = registration
            logger.info(
                "TPURaidenConnectorWorker rank%d --> stage3 work unit registered successfully",
                self.tp_rank)

        self._raiden_transfer_engine = engine
        self._raiden_manifest = manifest

        counts = manifest.tag_counts()
        geometry = manifest.geometry_by_tag()
        gdn_conv_count = sum(count for tag, count in counts.items()
                             if str(tag).startswith(rpm.TAG_GDN_CONV))
        gdn_ssm_count = sum(count for tag, count in counts.items()
                            if str(tag).startswith(rpm.TAG_GDN_SSM))
        summary_geometry = {tag: dict(geo) for tag, geo in geometry.items()}
        summary.update({
            "topology": topology,
            "model_server_role": role,
            "binding": manifest.binding,
            "tag_counts": dict(counts),
            "geometry": summary_geometry,
        })
        self._raiden_admission_summary = dict(summary)

        logger.info(
            "Raiden pool admission complete topology=%s role=%s binding=%s "
            "pools=%d storages=%d fa=%d gdn.conv=%d gdn.ssm=%d",
            topology,
            role,
            manifest.binding,
            len(manifest.pools),
            len(manifest.storages),
            len(manifest.pools_of_class(rpm.TAG_FA)),
            gdn_conv_count,
            gdn_ssm_count,
        )
        for tag, geo in geometry.items():
            logger.info(
                "Raiden pool geometry tag=%s num_blocks=%d "
                "block_stride_bytes=%d live_bytes_per_block=%d",
                tag,
                int(geo["num_blocks"]),
                int(geo["block_stride_bytes"]),
                int(geo["live_bytes_per_block"]),
            )
        logger.info(
            "Raiden pool binding verified: %d/%d pool storages matched "
            "typed KV cache storages",
            verified_storages,
            len(manifest.storages),
        )

    def _admit_raiden_glm_kv_cache(self, runner: TPUModelRunner) -> None:
        """Constructs Raiden engine over the GLM MLA cache classes
        (MLA latent + DSA indexer)."""
        if self._raiden_transfer_engine is not None:
            return
        if not self._raiden_stage3_enabled():
            raise RuntimeError("TPU_RAIDEN_GLM_ADMISSION requires "
                               "TPU_KV_RESHARD_TRANSPORT=raiden")
        self._maybe_host_reshard_store()
        controller_address = _resolved_reshard_controller_address(self.dp_rank)
        if not controller_address:
            raise ValueError("TPU_RAIDEN_CONTROLLER_ADDRESS is required when "
                             "TPU_KV_RESHARD_TRANSPORT=raiden")

        from vllm_torchtpu.distributed.kv_transfer.raiden import \
            pool_manifest as rpm

        topology = self._raiden_glm_admission_topology()
        role = "kv_producer" if self.is_producer else "kv_consumer"
        named_kv_caches = self.named_kv_caches
        if not named_kv_caches:
            raise ValueError("Raiden pool admission requires "
                             "register_kv_caches() before register_runner()")
        raw_tensors = tuple(runner.kv_cache_raw_tensors or ())

        manifest = rpm.build_glm_mla_pool_manifest(
            named_kv_caches=named_kv_caches,
            raw_tensors=raw_tensors,
            block_size_tokens=int(self.vllm_config.cache_config.block_size),
        )
        rpm.verify_storage_binding(
            manifest=manifest,
            named_kv_caches=named_kv_caches,
            raw_tensors=raw_tensors,
        )
        rpm.materialize_storages(manifest)

        storages = list(manifest.storages)
        engine = self._construct_raiden_transfer_engine(storages, num_slots=1)
        summary = self._register_raiden_pools(
            engine, manifest,
            getattr(getattr(runner, "kv_cache_config", None),
                    "kv_cache_groups", None))

        registration = self._register_raiden_stage3_work_unit(
            engine=engine,
            manifest=manifest,
            controller_address=controller_address,
        )
        summary["stage3_registration"] = registration

        self._raiden_transfer_engine = engine
        self._raiden_manifest = manifest
        # Per-tag (live_bytes_per_block, row_bytes) is fixed by the admitted
        # manifest; cache it here because the producer's span lowering reads it every step.
        self._stage3_glm_geometry = self._measure_glm_tag_geometry(manifest)

        counts = manifest.tag_counts()
        geometry = manifest.geometry_by_tag()
        summary_geometry = {tag: dict(geo) for tag, geo in geometry.items()}
        summary.update({
            "topology": topology,
            "model_server_role": role,
            "binding": manifest.binding,
            "tag_counts": dict(counts),
            "geometry": summary_geometry,
        })
        self._raiden_admission_summary = dict(summary)
        logger.info(
            "Raiden GLM pool admission complete topology=%s role=%s "
            "binding=%s pools=%d storages=%d tag_counts=%s",
            topology,
            role,
            manifest.binding,
            len(manifest.pools),
            len(manifest.storages),
            dict(counts),
        )
        for tag, geo in geometry.items():
            logger.info(
                "Raiden pool geometry tag=%s num_blocks=%d "
                "block_stride_bytes=%d live_bytes_per_block=%d",
                tag,
                int(geo["num_blocks"]),
                int(geo["block_stride_bytes"]),
                int(geo["live_bytes_per_block"]),
            )

    def _raiden_glm_admission_topology(self) -> str:
        """Validates the one supported topology. Anything else fails closed here."""
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        tp_size = int(self.tp_size)
        dp_size = int(parallel_config.data_parallel_size or 1)
        supported = ("Raiden GLM admission supports prefill tp{N}/dp1 -> "
                     "decode tp1/dp{M} only")
        if pcp_size != 1:
            raise ValueError(f"{supported}; got "
                             f"prefill_context_parallel_size={pcp_size}")
        if self.is_producer:
            if dp_size != 1:
                raise ValueError(f"{supported}; got producer "
                                 f"tensor_parallel_size={tp_size}, "
                                 f"data_parallel_size={dp_size}")
            return f"tp{tp_size}_prefill"
        if tp_size != 1:
            raise ValueError(f"{supported}; got consumer "
                             f"tensor_parallel_size={tp_size}, "
                             f"data_parallel_size={dp_size}")
        return f"dp{dp_size}_decode"

    @staticmethod
    def _measure_raiden_fa_layout(
        manifest: Any, ) -> tuple[str, dict[str, Any]]:
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import \
            measured_fa_layout_fingerprint

        return measured_fa_layout_fingerprint(manifest)

    @staticmethod
    def _measure_raiden_glm_layout(
            manifest: Any, page_tokens: int) -> tuple[str, dict[str, Any]]:
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import \
            measured_glm_layout_fingerprint

        return measured_glm_layout_fingerprint(manifest,
                                               page_tokens=page_tokens)

    def _measure_stage3_layout(
            self, manifest: Any) -> tuple[str, dict[str, Any], int]:
        """Layout identity and page geometry of the admitted model."""
        if _use_raiden_glm_admission():
            page_tokens = int(self.vllm_config.cache_config.block_size)
            fingerprint, payload = self._measure_raiden_glm_layout(
                manifest, page_tokens)
            return fingerprint, payload, page_tokens
        from vllm_torchtpu.distributed.kv_transfer.raiden.layout_fingerprint import \
            fa_page_tokens
        fingerprint, payload = self._measure_raiden_fa_layout(manifest)
        return fingerprint, payload, fa_page_tokens(manifest)

    def _validate_stage3_transfer_parallelism(
            self, transfer_parallelism: int) -> None:
        """The producer must shard the transfer over exactly the rank set its
        byte-span lowering assumes: TP ranks for GLM, PCP ranks otherwise."""
        if _use_raiden_glm_admission():
            if transfer_parallelism != self.tp_size:
                raise ValueError(
                    "GLM producer transfer parallelism must equal the "
                    "complete TP rank count: "
                    f"parallelism={transfer_parallelism}, "
                    f"tp_size={self.tp_size}")
            return
        if _engine_is_pipeline(self.vllm_config):
            pp_size = _pipeline_parallel_size(self.vllm_config)
            if transfer_parallelism != pp_size:
                raise ValueError(
                    "Pipeline producer transfer parallelism must equal the "
                    "complete PP stage count: "
                    f"parallelism={transfer_parallelism}, pp_size={pp_size}")
            if not _use_per_layer_pool_tags():
                raise ValueError(
                    "A pipeline-parallel producer pairs pools by layer; set "
                    "TPU_RAIDEN_POOL_TAGS_PER_LAYER=1 on both peers")
            return
        pcp_size = int(
            self.vllm_config.parallel_config.prefill_context_parallel_size
            or 1)
        if transfer_parallelism != pcp_size:
            raise ValueError(
                "Producer transfer parallelism must equal the complete "
                "PCP rank count: "
                f"parallelism={transfer_parallelism}, pcp_size={pcp_size}")

    def _maybe_host_reshard_store(self) -> None:
        """In-engine hosting (zero sidecars): each engine's transfer-rank-0
        worker owns the in-process reshard store — the framed reshard
        service plus the dispatch controller its local workers register
        with. Peer ranks only register into it, with bounded retries."""
        if not _reshard_store_mode() or self._reshard_store is not None:
            return
        if self._local_raiden_transfer_rank() != 0:
            return
        from tpu_sync.api.torch.reshard_store import RaidenId as _StoreRaidenId
        from tpu_sync.api.torch.reshard_store import ReshardStore

        default_job = "prefill" if self.is_producer else "decode"
        job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or default_job
        self._reshard_store = ReshardStore(
            raiden_id=_StoreRaidenId(
                job_name=job_name,
                job_replica_id=self._raiden_engine_id(),
                data_name="reshard_store",
                data_replica_idx=self.dp_rank,
            ),
            store_server_ip=_reshard_advertise_host(),
            raiden_controller_port=(
                int(tpu_envs.TPU_RAIDEN_STORE_DISPATCH_PORT_BASE) +
                int(self.dp_rank)),
            reshard_service_port=_reshard_service_port(self.dp_rank),
        )
        logger.info(
            "TPURaidenConnectorWorker rank%d --> hosting reshard store "
            "service=%s dispatch=%s", self.tp_rank,
            _reshard_service_address(self.dp_rank),
            _reshard_dispatch_address(self.dp_rank))

    def _raiden_work_unit_fields(self, transfer_rank: int) -> dict[str, Any]:
        default_job = "prefill" if self.is_producer else "decode"
        job_name = str(tpu_envs.TPU_RAIDEN_JOB_NAME).strip() or default_job
        engine_id = self._raiden_engine_id()
        return stage3_fa_raiden_id_fields(
            job_name=job_name,
            engine_id=engine_id,
            dp_rank=self.dp_rank,
            transfer_rank=transfer_rank,
            is_producer=self.is_producer,
            per_rank_unit=(self.is_producer
                           or _engine_is_pipeline(self.vllm_config)),
        )

    def _register_raiden_stage3_work_unit(
        self,
        *,
        engine: Any,
        manifest: Any,
        controller_address: str,
    ) -> dict[str, Any]:
        """Measure and register this worker with its cluster controller."""
        data_address = str(getattr(engine, "transfer_address", "")).strip()
        listener_address = str(getattr(engine, "listener_address", "")).strip()
        if not data_address:
            raise RuntimeError(
                "Stage-3 Raiden manager did not advertise a data endpoint")
        if not listener_address:
            raise RuntimeError(
                "Stage-3 Raiden manager did not advertise a listener endpoint")

        fingerprint, fingerprint_payload, page_tokens = (
            self._measure_stage3_layout(manifest))
        transfer_parallelism = self._raiden_transfer_parallelism()
        transfer_rank = self._local_raiden_transfer_rank()
        interleave_tokens = self._raiden_interleave_tokens(
            page_tokens, transfer_parallelism)
        if self.is_producer:
            self._validate_stage3_transfer_parallelism(transfer_parallelism)
        if transfer_rank < 0 or transfer_rank >= transfer_parallelism:
            raise ValueError(
                "Raiden transfer rank is outside the admitted parallelism: "
                f"rank={transfer_rank}, parallelism={transfer_parallelism}")

        unit = self._new_raiden_id(
            self._raiden_work_unit_fields(transfer_rank))
        if _reshard_store_mode():
            self._wait_for_address_ready(controller_address, timeout_s=60.0)
        facade = self._new_raiden_controller_facade(controller_address)
        # Store mode: the store is brought up by this engine's rank-0
        # worker in parallel with the peer ranks' init — bounded retry
        # instead of the sidecar era's launcher-ordered readiness.
        registration_deadline = time.perf_counter() + (
            120.0 if _reshard_store_mode() else 0.0)
        while True:
            try:
                facade.register_work_unit(
                    unit=unit,
                    shards=[data_address],
                    control_plane_rpc_address=listener_address,
                    pool_manifest=manifest.pool_dicts(),
                    layout_fingerprint=fingerprint,
                    page_tokens=page_tokens,
                    transfer_parallelism=transfer_parallelism,
                    transfer_rank=transfer_rank,
                )
                break
            except Exception as exc:  # noqa: BLE001 — retry-or-reraise
                if time.perf_counter() >= registration_deadline:
                    raise
                logger.info(
                    "Raiden work-unit registration to %s not accepted yet "
                    "(%s); retrying", controller_address, exc)
                time.sleep(0.5)

        self._raiden_controller_facade = facade
        self._raiden_controller_address = controller_address
        self._raiden_work_unit = unit
        self._raiden_layout_fingerprint = fingerprint
        self._raiden_layout_fingerprint_payload = dict(fingerprint_payload)
        logger.info(
            "Raiden work unit registered controller=%s unit=%s "
            "data=%s listener=%s page_tokens=%d transfer_rank=%d/%d "
            "interleave_tokens=%d layout_fingerprint=%s",
            controller_address,
            unit,
            data_address,
            listener_address,
            page_tokens,
            transfer_rank,
            transfer_parallelism,
            interleave_tokens,
            fingerprint,
        )
        return {
            "controller_address": controller_address,
            "unit": dict(self._raiden_work_unit_fields(transfer_rank)),
            "shards": [data_address],
            "control_plane_rpc_address": listener_address,
            "layout_fingerprint": fingerprint,
            "layout_fingerprint_payload": dict(fingerprint_payload),
            "page_tokens": page_tokens,
            "interleave_tokens": interleave_tokens,
            "transfer_parallelism": transfer_parallelism,
            "transfer_rank": transfer_rank,
        }

    def _raiden_qwen35_admission_topology(self) -> str:
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size or 1)
        tp_size = int(parallel_config.tensor_parallel_size or self.tp_size)
        dp_size = int(parallel_config.data_parallel_size or 1)
        pp_size = _pipeline_parallel_size(self.vllm_config)
        if self.is_producer:
            if (tp_size == 1 and pcp_size == 1 and dp_size == 1
                    and pp_size in (2, 4, 8)):
                return f"pp{pp_size}_prefill"
            if tp_size == 1 and pcp_size in (4, 8) and dp_size == 1:
                return f"pcp{pcp_size}_prefill"
            if tp_size == 1 and pcp_size == 1 and dp_size in (4, 8, 16):
                return f"dp{dp_size}_prefill"
            raise ValueError(
                "Raiden Qwen3.5 admission topology pcp8_prefill, "
                "pcp4_prefill, dp8_prefill, dp4_prefill, dp16_prefill, or "
                "pp{2,4,8}_prefill requires kv_producer with "
                "tensor_parallel_size=1 and either "
                "prefill_context_parallel_size in (4, 8), "
                "data_parallel_size=1; or "
                "prefill_context_parallel_size=1, "
                "data_parallel_size in (4, 8, 16); or "
                "pipeline_parallel_size in (2, 4, 8) alone; got "
                f"prefill_context_parallel_size={pcp_size}, "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}, "
                f"pipeline_parallel_size={pp_size}")
        if (tp_size == 1 and pcp_size == 1 and dp_size == 1
                and pp_size in (2, 4, 8)):
            return f"pp{pp_size}_decode"
        if pcp_size != 1 or tp_size != 1 or dp_size not in (4, 8, 16):
            raise ValueError(
                "Raiden Qwen3.5 admission topology dp8_decode, dp4_decode, "
                "dp16_decode, or pp{2,4,8}_decode requires kv_consumer with "
                "prefill_context_parallel_size=1, tensor_parallel_size=1 and "
                "either data_parallel_size in (4, 8, 16) or "
                "pipeline_parallel_size in (2, 4, 8); got "
                f"prefill_context_parallel_size={pcp_size}, "
                f"tensor_parallel_size={tp_size}, "
                f"data_parallel_size={dp_size}, "
                f"pipeline_parallel_size={pp_size}")
        return f"dp{dp_size}_decode"

    def _model_config_int(self, name: str, default: int) -> int:
        model_config = self.vllm_config.model_config
        hf_config = model_config.hf_config
        if hf_config is None:
            hf_config = model_config
        text_config = getattr(hf_config, "text_config", hf_config)
        value = getattr(text_config, name, None)
        return int(default if value is None else value)

    def _local_head_count(self, total_heads: int) -> int:
        total_heads = int(total_heads)
        if total_heads <= 0:
            return 0
        parallel_config = self.vllm_config.parallel_config
        tp_size = int(parallel_config.tensor_parallel_size or self.tp_size)
        if total_heads < tp_size:
            return 1
        if total_heads % tp_size:
            raise ValueError(f"total_heads={total_heads} must be divisible by "
                             f"tp_size={tp_size}")
        return total_heads // tp_size

    def _local_gdn_head_count(self, total_heads: int) -> int:
        """Return the GDN head count stored by this TP/PCP worker.

        GDN cache specs are localized across PCP as well as TP.  Full-
        attention head geometry is only TP-local, so keep this adjustment
        specific to GDN admission.
        """
        tp_local_heads = self._local_head_count(total_heads)
        if tp_local_heads <= 0:
            return 0
        parallel_config = self.vllm_config.parallel_config
        pcp_size = int(parallel_config.prefill_context_parallel_size)
        if tp_local_heads % pcp_size:
            raise ValueError(
                f"TP-local GDN heads={tp_local_heads} must be divisible by "
                f"pcp_size={pcp_size}")
        return tp_local_heads // pcp_size

    @staticmethod
    def _measure_glm_tag_geometry(manifest: Any) -> dict[str, tuple[int, int]]:
        """Per-tag (live_bytes_per_block, row_bytes) of an admitted GLM
        manifest."""
        geometry = manifest.geometry_by_tag()
        result: dict[str, tuple[int, int]] = {}
        for tag in _STAGE3_GLM_TRANSFER_POOL_TAGS:
            geo = geometry.get(tag)
            if not geo:
                raise RuntimeError(
                    f"Stage-3 GLM manifest is missing tag {tag!r}")
            pool = next(p for p in manifest.pools if p.tag == tag)
            (region, ) = pool.regions
            result[tag] = (int(geo["live_bytes_per_block"]),
                           int(region.unit_bytes))
        return result

    def _raiden_staging_blocks_per_pool(self, manifest: Any,
                                        kv_cache_groups: Any) -> list[int]:
        """Bounded host staging hints, one per pool in manifest order.

        Raiden sizes each storage's staging arena as leases x max(hint) over
        the pools on that storage and leases one slot per distinct device
        block id a transfer touches there. One transfer carries a request's
        full-attention pages plus one state block per GDN group under a
        single uuid, and pools of different kv-cache groups draw block ids
        from different block tables, so pools sharing a storage touch
        disjoint ids. Every pool therefore reports the union for its whole
        storage: pages per max-length request for each full-attention group
        present plus one per GDN group present (a group's conv and ssm pools
        address the same block). A storage holding a pool of unknown kind
        reports 0, which keeps it on the full host mirror (see
        RESHARD_BOUNDED_STAGING_DESIGN.md).
        """
        from vllm_torchtpu.distributed.kv_transfer.raiden import \
            pool_manifest as rpm

        group_of_layer: dict[str, int] = {}
        for index, group in enumerate(kv_cache_groups or ()):
            for layer_name in getattr(group, "layer_names", ()):
                group_of_layer[str(layer_name)] = index
        fa_pages = int(self._max_request_blocks())
        # storage -> {block table: blocks one transfer touches from it}
        tables_by_storage: dict[int, dict[Any, int]] = {}
        full_mirror: set[int] = set()
        for pool in manifest.pools:
            tag = str(pool.tag)
            storage = int(pool.storage_index)
            if tag.startswith(rpm.TAG_FA):
                blocks = fa_pages
            elif tag.startswith(rpm.TAG_GDN_CONV) or tag.startswith(
                    rpm.TAG_GDN_SSM):
                blocks = 1
            else:
                full_mirror.add(storage)
                continue
            # Layers of one kv-cache group share a block table; a layer
            # outside every group is its own table.
            layer_name = str(pool.layer_name)
            table = group_of_layer.get(layer_name, layer_name)
            tables = tables_by_storage.setdefault(storage, {})
            tables[table] = max(tables.get(table, 0), blocks)
        hints: list[int] = []
        for pool in manifest.pools:
            storage = int(pool.storage_index)
            if storage in full_mirror:
                hints.append(0)
            else:
                hints.append(sum(tables_by_storage[storage].values()))
        return hints

    def _register_raiden_pools(self, engine: Any, manifest: Any,
                               kv_cache_groups: Any) -> dict:
        """Registers the manifest's pools, with bounded host staging when the
        installed tpu_sync supports it (TPU_RAIDEN_POOL_STAGING_LEASES > 0).

        The full host shadow of the device KV pool does not fit libtpu's
        premapped (pinned) pool once the pool is auto-sized, which silently
        routes every D2H/H2D through the staged-copy slow path; bounded
        staging leases N transfers' worth of pages instead.
        """
        pool_dicts = manifest.pool_dicts()
        leases = int(dist_utils.get_raiden_pool_staging_leases())
        summary: dict
        if leases > 0:
            hints = self._raiden_staging_blocks_per_pool(
                manifest, kv_cache_groups)
            try:
                summary = dict(
                    engine.register_pools(pool_dicts,
                                          staging_leases=leases,
                                          staging_blocks_per_pool=hints))
            except TypeError as exc:
                # Older tpu_sync wheels predate bounded staging.
                logger.warning(
                    "Raiden bounded host staging unavailable in the installed "
                    "tpu_sync (%s); falling back to the full host mirror "
                    "(make sure TPU_PREMAPPED_BUFFER_SIZE covers it)", exc)
                summary = dict(engine.register_pools(pool_dicts))
        else:
            summary = dict(engine.register_pools(pool_dicts))
        staging = summary.get("host_staging")
        if isinstance(staging, dict):
            logger.info(
                "Raiden host staging mode=%s leases=%s "
                "bounded_bytes_per_shard=%s full_mirror_bytes_per_shard=%s "
                "storages=%d",
                staging.get("mode"),
                staging.get("leases"),
                staging.get("bounded_storage_bytes_per_shard"),
                staging.get("full_storage_bytes_per_shard"),
                len(staging.get("storages") or ()),
            )
        else:
            logger.info("Raiden host staging mode=full (legacy tpu_sync)")
        return summary
