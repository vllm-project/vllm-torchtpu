"""TPU Multi Connector wrapper for combining disagg and offloading."""

from collections.abc import Callable
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector


@dataclass
class TPUMultiHandshakeMetadata(KVConnectorHandshakeMetadata):
    metadata: tuple[KVConnectorHandshakeMetadata | None, ...]
    transfer_rank: int | None = None


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

    def start_load_kv_for_forward(
        self, forward_context, **kwargs
    ) -> Callable[[], None] | None:
        callbacks = []
        for connector in self._connectors:
            start = getattr(connector, "start_load_kv_for_forward", None)
            if start is None:
                connector.start_load_kv(forward_context, **kwargs)
            else:
                callback = start(forward_context, **kwargs)
                if callback is not None:
                    callbacks.append(callback)
        if not callbacks:
            return None

        def publish() -> None:
            for callback in callbacks:
                callback()

        return publish

    def get_handshake_metadata(self) -> TPUMultiHandshakeMetadata | None:
        children = tuple(c.get_handshake_metadata() for c in self._connectors)
        if all(item is None for item in children):
            return None
        ranks = {
            int(item.transfer_rank)
            for item in children
            if getattr(item, "transfer_rank", None) is not None
        }
        if len(ranks) > 1:
            raise ValueError("Child connectors disagree on the worker transfer rank")
        return TPUMultiHandshakeMetadata(children, next(iter(ranks), None))

    def _dispatch_handshake(self, metadata: dict, method: str) -> None:
        for item in metadata.values():
            if not isinstance(item, TPUMultiHandshakeMetadata):
                raise TypeError("Expected TPU multi-connector handshake metadata")
            if len(item.metadata) != len(self._connectors):
                raise ValueError("Multi-connector handshake child count mismatch")
        for index, connector in enumerate(self._connectors):
            child_metadata = {
                rank: item.metadata[index]
                for rank, item in metadata.items()
                if item.metadata[index] is not None
            }
            if child_metadata:
                getattr(connector, method)(child_metadata)

    def set_xfer_handshake_metadata(self, metadata: dict) -> None:
        self._dispatch_handshake(metadata, "set_xfer_handshake_metadata")

    def set_xfer_handshake_metadata_pp_aware(self, metadata: dict) -> None:
        self._dispatch_handshake(metadata, "set_xfer_handshake_metadata_pp_aware")

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
