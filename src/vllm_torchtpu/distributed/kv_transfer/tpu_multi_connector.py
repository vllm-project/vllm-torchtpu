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

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

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
        capability. PrefixAwareScheduler then asks that child, through
        ``resolve_divergent_local_hit_tokens``, how much of the hit to keep.
        """
        return any(c.supports_divergent_local_hybrid_hits for c in self._connectors)

    def poll_and_complete_transfers(self, scheduler) -> tuple[int, int, int, bool]:
        """Sums every child's schedule-start completion poll."""
        total_ns = 0
        total_early = 0
        total_waited = 0
        any_waited = False
        for c in self._connectors:
            fn = getattr(c, "poll_and_complete_transfers", None)
            if callable(fn):
                ns, early, waited_reqs, waited = fn(scheduler)
                total_ns += ns
                total_early += early
                total_waited += waited_reqs
                any_waited = any_waited or waited
        return total_ns, total_early, total_waited, any_waited

    def resolve_divergent_local_hit_tokens(
        self,
        request,
        num_local: int,
        num_reconciled: int,
        block_size: int,
    ) -> int | None:
        """Returns the first child's answer; None when no child can serve it."""
        for c in self._connectors:
            fn = getattr(c, "resolve_divergent_local_hit_tokens", None)
            if fn is not None:
                res = fn(request, num_local, num_reconciled, block_size)
                if res is not None:
                    return res
        return None
