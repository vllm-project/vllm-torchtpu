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
"""Connector wiring for MooncakeDistributedStore KV cache offloading on TPU.

Enable via:

    --kv-transfer-config '{
        "kv_connector": "TPUMooncakeOffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "master_server_address": "10.0.0.1:50051",
            "metadata_server": "http://10.0.0.1:8080/metadata",
            "protocol": "hp_tcp",
            "global_segment_size": "8GB",
            "local_buffer_size": "512MB"
        }
    }'

Architecture:
- Subclasses vLLM's `OffloadingConnector` so `OffloadingConnectorScheduler`
  owns prefix-cache admission, block reuse fencing (`jobs_to_flush`), and
  metrics/event reporting.
- Overrides `register_kv_caches` to deduplicate TPU unified block pool (UBP)
  storages into zero-copy 32-byte-aligned `int8` byte views, avoiding
  per-layer GPU assumptions and fragmenting small per-layer transfers.
- Records non-blocking TPU stream events at the end of every step carrying new
  store jobs so `MooncakeStoreOffloadingWorker` waits for device writes to
  retire before reading HBM, without adding device idle bubbles.
- Reports load failures cleanly via `KVConnectorTransferResults.failed_recving`
  so vLLM recomputes or fails affected requests per `kv_load_failure_policy`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    config as offloading_config_module,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.distributed.kv_transfer.mooncake_tpu_utils import (
    canonicalize_tpu_kv_pools,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.offload.mooncake_store import (
    MooncakeStoreOffloadingWorker,
    TPUMooncakeStoreOffloadingSpec,
)

logger = init_logger(__name__)


class TPUMooncakeOffloadingConnector(OffloadingConnector):
    """Top-level TPU KV offloading connector backed by MooncakeDistributedStore."""

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """TPU KV pages are transferred as opaque bytes; no GPU layout is required."""
        del vllm_config
        return None

    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return bool(tpu_envs.VLLM_TPU_BLOCK_MAJOR_KV)

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)

        # Call through module to preserve PCP-aware patch on build_offloading_config.
        offloading_config = offloading_config_module.build_offloading_config(
            vllm_config, kv_cache_config
        )
        spec = TPUMooncakeStoreOffloadingSpec(
            offloading_config, vllm_config, kv_cache_config
        )

        self.connector_scheduler: OffloadingConnectorScheduler | None = None
        self.connector_worker: OffloadingConnectorWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = OffloadingConnectorScheduler(
                spec, vllm_config, kv_cache_config
            )
        elif role == KVConnectorRole.WORKER:
            self.connector_worker = OffloadingConnectorWorker(
                spec, vllm_config, kv_cache_config
            )

    def register_kv_caches(
        self, kv_caches: Mapping[str, torch.Tensor | Sequence[torch.Tensor]]
    ) -> None:
        """Register deduplicated TPU UBP storages as canonical offload tensors."""
        assert self.connector_worker is not None
        num_blocks = self.connector_worker.kv_cache_config.num_blocks
        tpu_pools = canonicalize_tpu_kv_pools(kv_caches, num_blocks)

        pool_tensors = [
            CanonicalKVCacheTensor(
                tensor=pool.tensor, page_size_bytes=pool.page_size_bytes
            )
            for pool in tpu_pools
        ]
        group_data_refs = [
            [
                CanonicalKVCacheRef(
                    tensor_idx=idx, page_size_bytes=pool_tensor.page_size_bytes
                )
                for idx, pool_tensor in enumerate(pool_tensors)
            ]
            for _ in offloading_config_module.get_offloading_group_ids(
                self.connector_worker.kv_cache_config
            )
        ]

        self.connector_worker._init_worker(
            CanonicalKVCaches(tensors=pool_tensors, group_data_refs=group_data_refs)
        )

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        """Record TPU fences for this step's deferred stores and drain completed
        transfers, surfacing load errors through `failed_recving`."""
        del finished_req_ids
        assert self.connector_worker is not None
        meta = self._get_connector_metadata()
        assert isinstance(meta, OffloadingConnectorMetadata)

        worker = self.connector_worker.worker
        assert worker is not None
        if meta.store_jobs and isinstance(worker, MooncakeStoreOffloadingWorker):
            worker.record_step_store_fence(meta.store_jobs.keys())
        self.connector_worker.prepare_store_kv(meta)

        finished_recving: set[str] = set()
        failed_recving: set[str] = set()
        for transfer_result in worker.get_finished():
            job_id = transfer_result.job_id
            is_load = job_id in self.connector_worker._load_jobs
            if (
                transfer_result.success
                and transfer_result.transfer_time is not None
                and transfer_result.transfer_size is not None
            ):
                transfer_stats = (
                    self.connector_worker._connector_worker_meta.transfer_stats
                )
                stats = transfer_stats.load if is_load else transfer_stats.store
                stats.record(
                    transfer_result.transfer_size,
                    transfer_result.transfer_time,
                )

            self.connector_worker._connector_worker_meta.mark_completed(job_id)
            req_id = self.connector_worker._load_jobs.pop(job_id, None)
            if req_id is not None:
                # Failed loads still belong in finished_recving to release the
                # request from WAITING_FOR_REMOTE_KVS.
                finished_recving.add(req_id)
                if not transfer_result.success:
                    failed_recving.add(req_id)

        return KVConnectorTransferResults(
            finished_recving=finished_recving,
            failed_recving=failed_recving,
        )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        results = self.get_transfer_results(finished_req_ids)
        if results.failed_recving:
            raise RuntimeError(
                "TPUMooncakeOffloadingConnector.get_finished() cannot report "
                f"failed loads for requests {sorted(results.failed_recving)}; "
                "call get_transfer_results() instead"
            )
        return results.finished_sending, results.finished_recving
