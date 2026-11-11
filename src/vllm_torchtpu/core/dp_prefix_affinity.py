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
"""Prefix-affinity routing across internal data-parallel engines.

Upstream ``DPLBAsyncMPClient`` picks the least-loaded engine for every request
unless the request carries an explicit ``X-data-parallel-rank``. Multi-turn
traffic then splits each conversation's turns across engines, and every engine
caches about half of every conversation.

Enabled with ``--additional-config '{"dp_prefix_affinity": true}'``, the client
keeps one approximate prefix model per engine: an LRU set of chained hashes of
the ``block_size``-token chunks it routed there. A request goes to the engine
with the longest prefix match, as if it carried ``X-data-parallel-rank``, when
that match covers more than half of the prompt, is strictly longer than every
other engine's match, and the load bound holds. Otherwise the upstream load
balancer decides. A prefix that all engines hold (a shared system prompt)
therefore never pins traffic to one engine; only the conversation-specific
tail does. This is the cache-aware
policy of SGLang's router, run inside the single API server.

Routing state lives in one API-server process, so the mode requires
``--api-server-count=1``.
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core_client import DPLBAsyncMPClient, EngineCoreClient

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.runner.utils import DP_PREFIX_AFFINITY_KEY

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class PrefixAffinityRouter:
    """Per-engine prefix models and the routing decision; no vLLM state."""

    # Load imbalance (in requests) tolerated before the affinity engine yields
    # to the least-loaded one: both the absolute and the relative bound must be
    # exceeded, as in SGLang's cache-aware router. Moving a conversation
    # re-prefills its whole context, so the bound only guards against hot
    # spots; new conversations already go to the least-loaded engine.
    BALANCE_ABS_THRESHOLD = 8
    BALANCE_REL_THRESHOLD = 1.5
    # Minimum matched fraction of the prompt, as SGLang's cache threshold: a
    # short shared prefix (a system prompt cached on one engine only during
    # warm-up) does not attract a mostly new prompt.
    CACHE_THRESHOLD = 0.5

    def __init__(self, num_engines: int, chunk_size: int, capacity_chunks: int):
        assert num_engines > 1, "prefix-affinity routing needs data parallelism"
        assert chunk_size > 0 and capacity_chunks > 0
        self.chunk_size = chunk_size
        self.capacity_chunks = capacity_chunks
        self.engines: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(num_engines)
        ]

    def chunk_hashes(self, token_ids: Sequence[int]) -> list[int]:
        hashes = []
        prev = 0
        size = self.chunk_size
        for start in range(0, len(token_ids) - size + 1, size):
            prev = hash((prev, tuple(token_ids[start : start + size])))
            hashes.append(prev)
        return hashes

    def match_lengths(self, hashes: list[int]) -> list[int]:
        matches = []
        for model in self.engines:
            n = 0
            while n < len(hashes) and hashes[n] in model:
                n += 1
            matches.append(n)
        return matches

    def choose(self, hashes: list[int], loads: Sequence[float]) -> int | None:
        """Return the affinity engine, or None to defer to load balancing."""
        assert len(loads) == len(self.engines)
        matches = self.match_lengths(hashes)
        best = max(range(len(matches)), key=matches.__getitem__)
        runner_up = max(m for i, m in enumerate(matches) if i != best)
        min_match = self.CACHE_THRESHOLD * len(hashes)
        if matches[best] <= max(runner_up, min_match):
            return None
        least = min(loads)
        if (
            loads[best] - least > self.BALANCE_ABS_THRESHOLD
            and loads[best] > self.BALANCE_REL_THRESHOLD * least
        ):
            return None
        return best

    def record(self, engine: int, hashes: list[int]) -> None:
        model = self.engines[engine]
        # Deepest chunk first so LRU eviction drops leaves before their parents
        # and every model stays prefix-closed.
        for h in reversed(hashes):
            model[h] = None
            model.move_to_end(h)
        while len(model) > self.capacity_chunks:
            model.popitem(last=False)


def prefix_affinity_enabled(vllm_config: "VllmConfig") -> bool:
    config = vllm_config.additional_config
    if not isinstance(config, dict) or DP_PREFIX_AFFINITY_KEY not in config:
        return False
    enabled = config[DP_PREFIX_AFFINITY_KEY]
    if not isinstance(enabled, bool):
        raise ValueError(f"additional_config.{DP_PREFIX_AFFINITY_KEY} must be a bool")
    parallel_config = vllm_config.parallel_config
    if enabled and (
        parallel_config.data_parallel_size < 2
        or parallel_config.data_parallel_external_lb
    ):
        raise ValueError(
            f"additional_config.{DP_PREFIX_AFFINITY_KEY} routes across internal "
            "data-parallel engines: set --data-parallel-size > 1 without "
            "external load balancing, or drop the key."
        )
    return enabled


class PrefixAffinityDPLBAsyncMPClient(DPLBAsyncMPClient):
    """DPLBAsyncMPClient that routes each request to its prefix's engine."""

    # Stale entries only send a request to an engine that already dropped the
    # prefix, which costs the same as a load-balanced pick. Tracking a few
    # times the HBM pool keeps prefixes that KV offloading still holds.
    _CAPACITY_HBM_MULTIPLE = 4

    def __init__(self, vllm_config: "VllmConfig", *args: Any, **kwargs: Any):
        super().__init__(vllm_config, *args, **kwargs)
        if self.client_count != 1:
            raise ValueError(
                f"additional_config.{DP_PREFIX_AFFINITY_KEY} keeps routing state in "
                f"one API server, got {self.client_count}: pass --api-server-count=1."
            )
        self._cache_config = vllm_config.cache_config
        self._router: PrefixAffinityRouter | None = None
        logger.info(
            "Prefix-affinity DP routing enabled across %d engines.",
            len(self.core_engines),
        )

    def _get_router(self) -> PrefixAffinityRouter:
        num_engines = len(self.core_engines)
        if self._router is None or len(self._router.engines) != num_engines:
            num_gpu_blocks = self._cache_config.num_gpu_blocks
            assert num_gpu_blocks, "engine handshake must report num_gpu_blocks"
            self._router = PrefixAffinityRouter(
                num_engines,
                self._cache_config.block_size,
                self._CAPACITY_HBM_MULTIPLE * num_gpu_blocks // num_engines,
            )
        return self._router

    def get_core_engine_for_request(self, request: EngineCoreRequest):
        token_ids = request.prompt_token_ids
        if token_ids is None or request.request_id in self.reqs_in_flight:
            return super().get_core_engine_for_request(request)
        router = self._get_router()
        hashes = router.chunk_hashes(token_ids)
        if request.data_parallel_rank is None and hashes:
            loads = [
                max(self.engine_inflight[engine], waiting + running)
                for engine, (waiting, running, _) in zip(
                    self.core_engines, self.lb_engines
                )
            ]
            request.data_parallel_rank = router.choose(hashes, loads)
        engine = super().get_core_engine_for_request(request)
        router.record(self.core_engines.index(engine), hashes)
        return engine


def patch_make_async_mp_client() -> None:
    if getattr(EngineCoreClient, "_tpu_dp_prefix_affinity_patch", False):
        return
    original = EngineCoreClient.make_async_mp_client

    def make_async_mp_client(vllm_config: "VllmConfig", *args: Any, **kwargs: Any):
        if prefix_affinity_enabled(vllm_config):
            return PrefixAffinityDPLBAsyncMPClient(vllm_config, *args, **kwargs)
        return original(vllm_config, *args, **kwargs)

    EngineCoreClient.make_async_mp_client = staticmethod(make_async_mp_client)
    EngineCoreClient._tpu_dp_prefix_affinity_patch = True
