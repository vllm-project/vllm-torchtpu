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
"""Prefix-aware engine choice for vLLM's data-parallel front end.

Each data-parallel engine keeps its own prefix cache. vLLM's load balancer
(DPLBAsyncMPClient.get_core_engine_for_request) picks the least loaded
engine and never asks which engine already holds the prompt's prefix, so a
conversation's next turn often lands on an engine that recomputes the whole
conversation so far.

With VLLM_TPU_DP_PREFIX_AFFINITY=1 the front end remembers, for every
CHUNK_TOKENS boundary of each prompt it routes, which engines took that prefix
(a chained hash of the token ids up to the boundary, in a bounded LRU table).
A new request goes to the least loaded engine among those holding its deepest
remembered boundary, unless that engine's load score exceeds the least loaded
engine's by more than LOAD_MARGIN requests. The stock choice decides when
that engine is too busy, when no boundary is remembered, and when every
engine holds the deepest one: a prefix every conversation shares (a system
prompt) is soon held by every engine and favors none. The choice only decides
where a request runs: an engine computes the same tokens with or without a
prefix-cache hit.

With VLLM_TPU_DP_PROMPT_BACKLOG=1 as well, the stock choice is the backlog one
(core/dp_prompt_backlog.py): it is installed first and this choice wraps it,
so the table records the engine a request finally went to.

Run with one API server process (--api-server-count 1): each API server
process has its own client and table, and the stock front end spreads a
conversation's turns over all of them.
"""

from collections import OrderedDict
from collections.abc import Sequence

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def boundary_hashes(token_ids: Sequence[int], chunk_tokens: int) -> list[int]:
    """One hash per whole chunk of the prompt; hash k covers tokens
    [0, (k + 1) * chunk_tokens), chained so equal hashes mean equal prefixes."""
    hashes: list[int] = []
    previous = 0
    for end in range(chunk_tokens, len(token_ids) + 1, chunk_tokens):
        previous = hash((previous, tuple(token_ids[end - chunk_tokens : end])))
        hashes.append(previous)
    return hashes


class PrefixTable:
    """Bounded map from a prefix hash to the indices of the engines that took
    a prompt with that prefix."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.engines: OrderedDict[int, set[int]] = OrderedDict()

    def deepest(self, hashes: list[int]) -> set[int] | None:
        """The engine indices of the deepest remembered boundary, or None."""
        for h in reversed(hashes):
            engines = self.engines.get(h)
            if engines:
                return engines
        return None

    def remember(self, hashes: list[int], engine: int) -> None:
        for h in hashes:
            self.engines.setdefault(h, set()).add(engine)
            self.engines.move_to_end(h)
        while len(self.engines) > self.capacity:
            self.engines.popitem(last=False)


def load_scores(client) -> list[float]:
    """The stock balancer's load score of every engine, computed as
    DPLBAsyncMPClient.get_core_engine_for_request computes it."""
    scores = []
    for idx, (waiting, running, kv_cache_usage) in enumerate(client.lb_engines):
        inflight = client.engine_inflight[client.core_engines[idx]]
        score = float(max(client.client_count * inflight, waiting + running))
        if waiting:
            score += waiting * 6.0 * max(0.0, kv_cache_usage - 0.5)
        scores.append(score)
    return scores


def make_router(stock, chunk_tokens: int, load_margin: int, capacity: int):
    """A get_core_engine_for_request that prefers the engine holding the
    prompt's deepest remembered prefix and otherwise calls `stock`."""

    def get_core_engine_for_request(self, request):
        token_ids = request.prompt_token_ids
        if (
            request.request_id in self.reqs_in_flight
            or request.data_parallel_rank is not None
            or request.pooling_params is not None
            or not token_ids
        ):
            return stock(self, request)

        table = self.__dict__.get("_prefix_table")
        if table is None:
            table = self._prefix_table = PrefixTable(capacity)

        hashes = boundary_hashes(token_ids, chunk_tokens)
        match = table.deepest(hashes)
        chosen_index = None
        if match is not None and len(match) < len(self.core_engines):
            scores = load_scores(self)
            engine_index = min(match, key=scores.__getitem__)
            if scores[engine_index] - min(scores) <= load_margin:
                chosen_index = engine_index

        if chosen_index is None:
            chosen = stock(self, request)
            chosen_index = self.core_engines.index(chosen)
        else:
            chosen = self.core_engines[chosen_index]
            self.lb_engines[chosen_index][0] += self.client_count
            self.reqs_in_flight[request.request_id] = chosen
            self.engine_inflight[chosen] += 1

        table.remember(hashes, chosen_index)
        return chosen

    # pyrefly: ignore[missing-attribute]
    get_core_engine_for_request.stock = stock
    return get_core_engine_for_request


def install(chunk_tokens: int, load_margin: int, capacity: int) -> None:
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    stock = DPLBAsyncMPClient.get_core_engine_for_request
    if hasattr(stock, "stock"):
        return
    DPLBAsyncMPClient.get_core_engine_for_request = make_router(
        stock, chunk_tokens, load_margin, capacity
    )
    logger.info(
        "dp prefix affinity on: %d-token chunks, load margin %d requests, "
        "table of %d prefixes",
        chunk_tokens,
        load_margin,
        capacity,
    )


def install_if_enabled() -> None:
    from vllm_torchtpu import envs

    if envs.VLLM_TPU_DP_PREFIX_AFFINITY:
        install(
            envs.VLLM_TPU_DP_PREFIX_AFFINITY_CHUNK_TOKENS,
            envs.VLLM_TPU_DP_PREFIX_AFFINITY_LOAD_MARGIN,
            envs.VLLM_TPU_DP_PREFIX_AFFINITY_TABLE_SIZE,
        )
