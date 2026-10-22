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
"""Model-owned Stage-3 page planning for Qwen's hybrid caches.

Native admission binds the loaded model identity and measured page layout.
This connector does not borrow another model's native pool admission.
"""

import math
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.parallel_state import get_ep_group, get_tp_group
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_torchtpu import envs
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
    _resolved_reshard_controller_address,
    _Stage3LoadMeta,
    _Stage3TransferGroup,
)

from .pd import TransferExtent, TransferGroup, build_transfer_groups, handoff_block_ids
from .pd_identity import canonical_rank_identity
from .pd_manifest import build_transfer_manifest, measure_transfer_layout


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
            or parallel.pipeline_parallel_size != 1
            or vllm_config.speculative_config is not None
            or cache.enable_prefix_caching
        ):
            raise NotImplementedError("Qwen PD first requires CP1/PP1, APC and MTP off")
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
        manifest = build_transfer_manifest(cache.native_config, cache.materialized)
        verify_storage_binding(
            manifest, self.named_kv_caches, runner.kv_cache_raw_tensors
        )
        materialize_storages(manifest)
        semantics = self._collect_pd_semantics(runner)
        # Validate physical geometry before starting any network/native manager.
        measure_transfer_layout(manifest)
        self.runner = runner
        self._qwen_page_plan = QwenPagePlan(cache.get_transfer_groups(), manifest)
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

    def _ensure_raiden_transfer_engine(self):
        if self._raiden_transfer_engine is None:
            raise RuntimeError("Qwen PD native admission must finish before transfer")
        return self._raiden_transfer_engine

    def _collect_pd_semantics(self, runner) -> dict[str, Any]:
        model = runner.model_adapter.cache.model
        if model.pd_gdn_qk_pair_layout != envs.TPU_GDN_CONV_QK_PAIR_LAYOUT:
            raise ValueError("Qwen PD GDN layout changed since model construction")
        scales = []
        for index, layer in enumerate(model.model.layers):
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
        if (
            transfer_parallelism != self.tp_size
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or parallel.pipeline_parallel_size != 1
            or self.vllm_config.speculative_config is not None
            or self.vllm_config.cache_config.enable_prefix_caching
        ):
            raise ValueError("Qwen PD requires matching TP, CP1/PP1 and APC/MTP off")

    def _raiden_interleave_tokens(
        self, page_tokens: int, transfer_parallelism: int
    ) -> int:
        self._validate_stage3_transfer_parallelism(transfer_parallelism)
        return page_tokens

    @staticmethod
    def _uses_stage3_grouped_pages() -> bool:
        return True

    def _local_raiden_transfer_rank(self) -> int:
        return self.tp_rank

    def _page_plan(self) -> "QwenPagePlan":
        if self._qwen_page_plan is None:
            raise RuntimeError("Qwen PD native page admission has not completed")
        if (
            self._raiden_transfer_parallelism() != self.tp_size
            or self._local_raiden_transfer_rank() != self.tp_rank
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
                            dst_unit_ordinal=self.tp_rank,
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
        if req_meta.skip_tokens or req_meta.src_parallelism != self.tp_size:
            raise ValueError("Qwen PD needs a full prefix and identical source TP")
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
        # Compressed keys follow their main IDs; rings retain independent IDs.
        expected = set(owners) | {
            name + ".compressed_k" for name in owners if name.endswith(".self_attn")
        }
        names = [pool.layer_name for pool in self.manifest.pools]
        tags = [pool.tag for pool in self.manifest.pools]
        if (
            len(names) != len(set(names))
            or len(tags) != len(set(tags))
            or set(names) != expected
        ):
            raise ValueError("Qwen PD page plan must cover every owner exactly once")
        return tuple(
            (
                pool,
                owners[
                    pool.layer_name
                    if pool.layer_name in owners
                    else pool.layer_name.removesuffix(".compressed_k")
                ],
            )
            for pool in self.manifest.pools
        )

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
