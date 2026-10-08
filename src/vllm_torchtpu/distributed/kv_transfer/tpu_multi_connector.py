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
        if role != KVConnectorRole.SCHEDULER:
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
