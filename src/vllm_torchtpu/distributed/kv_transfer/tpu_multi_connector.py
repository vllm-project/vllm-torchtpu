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

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from vllm_torchtpu import envs as tpu_envs


def _pure_producer(connector) -> bool:
    """A child configured ``kv_role="kv_producer"``: it hands KV off and never
    reports external tokens to load (the PD prefill's Raiden child)."""
    config = getattr(connector, "_kv_transfer_config", None)
    return getattr(config, "kv_role", None) == "kv_producer"


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

    @property
    def supports_divergent_local_hybrid_hits(self) -> bool:
        """With ``VLLM_TPU_OFFLOAD_DIVERGENT_HITS=1``, ignore pure ``kv_producer``
        children (which never load) when checking divergent hybrid-hit support.
        Flag off keeps upstream ``MultiConnector`` behavior."""
        if not tpu_envs.VLLM_TPU_OFFLOAD_DIVERGENT_HITS:
            return super().supports_divergent_local_hybrid_hits
        loaders = [c for c in self._connectors if not _pure_producer(c)]
        return bool(loaders) and all(
            c.supports_divergent_local_hybrid_hits for c in loaders
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

    def get_finished_count(self) -> int | None:
        """The one report count every child's completions need, else an error.

        The executor's KVOutputAggregator applies a single expected count to
        finished_sending and finished_recving alike. A child returning None
        (or 0, the TPU connector's legacy marker) counts on every worker
        reporting, i.e. the world size. Upstream returns None whenever any
        child does, which silently imposes the world size on a child that
        expects fewer reports (its requests would then never finish). Here
        the children must agree after that normalization; a PD producer of
        PP stages (one report per stage) and the Raiden offload connector
        (one echo per worker) both count world-size reports.
        """
        world = int(self._vllm_config.parallel_config.world_size)
        counts = {c.get_finished_count() or world for c in self._connectors}
        if len(counts) > 1:
            raise ValueError(
                "TPUMultiConnector children expect different numbers of "
                f"completion reports per request ({sorted(counts)}, world "
                f"size {world}); the executor applies one count to all"
            )
        count = counts.pop() if counts else None
        return None if count == world else count
