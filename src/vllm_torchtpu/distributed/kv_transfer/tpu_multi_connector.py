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
"""TPU Multi Connector wrapper for combining disagg and offloading."""

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector,
    MultiKVConnectorWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput

from vllm_torchtpu.distributed.kv_transfer.connector_options import (
    MultiConnectorOptions,
    read_options,
)
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class TPUMultiConnector(MultiConnector):
    """MultiConnector combining TPURaidenConnector (disaggregation) with
    TPURaidenOffloadingConnector (offloading).

    Children must be listed explicitly in
    ``kv_connector_extra_config["connectors"]``. No defaults are synthesized:
    upstream classmethods (``get_required_kvcache_layout``,
    ``requires_piecewise_for_cudagraph``) parse the original config before
    the connector is instantiated, so children injected at init time would
    be invisible to them.
    """

    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        # Opt-in: on a producer, report divergent local hybrid hits when any
        # child can complete them. See supports_divergent_local_hybrid_hits.
        options = read_options(
            MultiConnectorOptions,
            vllm_config.kv_transfer_config.kv_connector_extra_config,
        )
        self._producer_divergent_hits = (
            options.producer_divergent_hits
            and vllm_config.kv_transfer_config.is_kv_producer
        )
        # Only a consumer starts from a decode-local hit; on a producer the
        # transfer child always reports False, whatever the other children say.
        if (
            role != KVConnectorRole.SCHEDULER
            or vllm_config.kv_transfer_config.is_kv_producer
        ):
            return
        # Upstream reports divergent local hybrid hits only if every child
        # does. A child that cannot turns Stage-3 suffix loads off for the
        # whole engine and decode pulls whole payloads, so say which one.
        capable = {
            type(c).__name__: bool(
                getattr(c, "supports_divergent_local_hybrid_hits", False)
            )
            for c in self._connectors
        }
        if any(capable.values()) and not all(capable.values()):
            logger.warning(
                "Prefix-aware Stage-3 loads are off for this engine: %s cannot "
                "complete divergent local hybrid hits, so decode pulls whole "
                "payloads. Capability per child: %s",
                [name for name, ok in capable.items() if not ok],
                capable,
            )

    def register_runner(self, runner) -> None:
        # torchtpu extension unknown to upstream MultiConnector: forward it to
        # the children, or the Raiden child never gets a runner and fails with
        # "register_runner must be called before transfer" on the first load.
        for c in self._connectors:
            fn = getattr(c, "register_runner", None)
            if fn is not None:
                fn(runner)

    def get_block_ids_with_load_errors_group_index(self) -> int | None:
        """Return the KV cache group shared by all children's load errors.

        Failed block ids are unioned by upstream
        ``get_block_ids_with_load_errors``; this index only tells the
        scheduler which group's block table those ids resolve against.
        One scalar cannot scope two groups, so conflicting child indices
        raise instead of mis-scoping.
        """
        index: int | None = None
        for c in self._connectors:
            fn = getattr(c, "get_block_ids_with_load_errors_group_index", None)
            if fn is None:
                continue
            child_index = fn()
            if child_index is None:
                continue
            if index is not None and child_index != index:
                raise RuntimeError(
                    "Conflicting KV-load-error group indices across child "
                    f"connectors: {index} vs {child_index}; cannot scope "
                    "invalid blocks to a single cache group"
                )
            index = child_index
        return index

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        """Claims divergent hybrid hits when any child can complete them.

        Upstream's MultiConnector requires every child. The offloading child
        restores the lagging recurrent state from the host tier by itself, so
        a disagg producer next to it (which never loads) must not veto the
        capability. The pipeline chunk scheduler's local-hit lookup
        (``TpuPipelineChunkScheduler._get_local_prefix_cache_hit``) then asks
        that child, through ``resolve_divergent_local_hit_tokens``, how much
        of the hit to keep.

        This applies only on a producer whose kv_connector_extra_config sets
        producer_divergent_hits; otherwise upstream's every-child rule holds,
        which on a producer is False because the transfer child never loads.
        """
        if self._producer_divergent_hits:
            return any(c.supports_divergent_local_hybrid_hits for c in self._connectors)
        return super().supports_divergent_local_hybrid_hits

    def poll_and_complete_transfers(self, idle: bool) -> KVConnectorOutput | None:
        """The children's finished transfers, from each child's schedule-start
        poll, in one output shaped like the workers' echo: the finished
        requests of all children, and each child's worker metadata in its
        slot, which ``update_connector_output`` hands back to that child.
        None when no child has anything (asked by the pipeline chunk
        scheduler's prefix-aware admission at the start of each step)."""
        finished_recving: set[str] = set()
        metadata: list[KVConnectorWorkerMetadata | None] = []
        polled = False
        for c in self._connectors:
            fn = getattr(c, "poll_and_complete_transfers", None)
            output = fn(idle) if fn is not None else None
            if output is None:
                metadata.append(None)
                continue
            polled = True
            metadata.append(output.kv_connector_worker_meta)
            finished_recving |= output.finished_recving or set()
        if not polled:
            return None
        return KVConnectorOutput(
            finished_recving=finished_recving,
            kv_connector_worker_meta=MultiKVConnectorWorkerMetadata(
                metadata=tuple(metadata)
            ),
        )

    def resolve_divergent_local_hit_tokens(
        self,
        request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        """Returns the first child's answer; None when no child can serve it
        (asked by the pipeline chunk scheduler's local-hit lookup)."""
        for c in self._connectors:
            fn = getattr(c, "resolve_divergent_local_hit_tokens", None)
            if fn is not None:
                res = fn(request, num_local, num_reconciled, block_size)
                if res is not None:
                    return res
        return None

    def estimate_external_hit_tokens(
        self,
        request,
        num_local_tokens: int,
    ) -> int:
        """Returns the largest external hit any child expects to load.

        A prefill leg's prefix-aware scheduler reaches the offloading child's
        host-tier estimate through here; children without one count as 0.
        """
        best = 0
        for c in self._connectors:
            fn = getattr(c, "estimate_external_hit_tokens", None)
            if fn is not None:
                res = fn(request, num_local_tokens)
                if res and res > best:
                    best = res
        return best
