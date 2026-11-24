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
"""Parts shared by the schedulers that serve short prompts beside a running
long prefill.

vLLM schedules running requests before waiting ones, and a long prompt in
mid-prefill counts as running, so its chunks fill the step and a short prompt
that arrives behind it waits until that prefill ends. A scheduler built on
ShortPromptsFirstScheduler keeps part of the step's budget for short waiting
prompts and caps the chunks of running prefills at the budget left.

Two schedulers use it:
- ShortPrefillFirstScheduler (TPU_SHORT_PREFILL_FIRST_TOKENS) is the
  standalone one: it promotes short prompts in arrival order and caps every
  running prefill. Use it when nothing else picks the scheduler.
- PrefixAwareScheduler with short prompts first (TPU_PREFIX_AWARE_SHORT_PROMPTS)
  ranks short prompts by their uncached tokens, as its admission does, and
  keeps the oldest long prefill's full chunk. Use it with prefix caching,
  where PrefixAwareScheduler is the scheduler.
"""

from collections.abc import Iterable

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.request import Request, RequestStatus

from vllm_torchtpu import envs

# Statuses of a waiting request the waiting loop can admit as a new prefill.
ADMITTABLE_STATUSES = (RequestStatus.WAITING, RequestStatus.PREEMPTED)


def fit_short_prompts[T](
    candidates: Iterable[tuple[T, int]],
    reserved: int,
    free_tokens: int,
    free_slots: int,
) -> tuple[int, list[T]]:
    """Take candidates in order while a sequence slot and tokens are free,
    skipping each one that does not fit.

    A candidate is drawn only while room is left, so a lazy ``candidates``
    looks up no prompt it does not need.

    Args:
        candidates: (prompt, tokens) pairs in the order to take them.
        reserved: tokens already reserved this step.
        free_tokens: tokens the step can reserve in all.
        free_slots: sequence slots free for new prompts.

    Returns:
        The tokens reserved, including ``reserved``, and the prompts taken.
    """
    taken: list[T] = []
    remaining = iter(candidates)
    while len(taken) < free_slots and reserved < free_tokens:
        try:
            prompt, tokens = next(remaining)
        except StopIteration:
            break
        if reserved + tokens <= free_tokens:
            taken.append(prompt)
            reserved += tokens
    return reserved, taken


class ShortPromptsFirstScheduler(AsyncScheduler):
    """The running-prefill cap and the lookups both schedulers share."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._running_prefill_budget: int | None = None
        self._uncached_by_request: dict[str, int] = {}
        self._tier = (
            self._connector_tier() if envs.TPU_PREFIX_AWARE_RANK_WITH_TIER else None
        )
        self._tier_prefix_by_request: dict[str, int] = {}

    def _connector_tier(self) -> OffloadingConnectorScheduler:
        """The offloading KV connector's scheduler side: the tier admission
        reads past a prompt's prefix in HBM. A multi-connector is searched
        for its one offloading child."""
        connector = self.connector
        children = getattr(connector, "_connectors", None) or (connector,)
        schedulers = (getattr(child, "connector_scheduler", None) for child in children)
        tiers: list[OffloadingConnectorScheduler] = [
            scheduler
            for scheduler in schedulers
            if isinstance(scheduler, OffloadingConnectorScheduler)
        ]
        if len(tiers) != 1:
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_RANK_WITH_TIER needs one offloading KV "
                "connector; set one in --kv-transfer-config or unset it."
            )
        (tier,) = tiers
        if any(
            group.sliding_window_size_in_chunks is not None
            for group in tier.config.kv_group_configs
        ):
            raise NotImplementedError(
                "TPU_PREFIX_AWARE_RANK_WITH_TIER reads the tier as one prefix "
                "and needs full-attention KV cache groups only; unset it."
            )
        return tier

    def _reserve_prefill_lookahead(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        budget = self._running_prefill_budget if self._is_capped(request) else None
        if budget is not None:
            num_new_tokens = min(num_new_tokens, budget)
        num_new_tokens = self._lookahead_within_cap(
            request, num_computed_tokens, num_new_tokens
        )
        if budget is not None:
            self._running_prefill_budget = budget - num_new_tokens
        return num_new_tokens

    def _is_capped(self, request: Request) -> bool:
        """Whether the running-prefill budget caps this request's chunk."""
        return (
            self._running_prefill_budget is not None
            and request.status == RequestStatus.RUNNING
            and request.is_prefill_chunk
        )

    def _lookahead_within_cap(
        self, request: Request, num_computed_tokens: int, num_new_tokens: int
    ) -> int:
        return super()._reserve_prefill_lookahead(
            request, num_computed_tokens, num_new_tokens
        )

    def _uncached_once(self, request: Request) -> int:
        """The request's uncached prompt tokens, looked up once per stay in
        the waiting queue."""
        if request.request_id not in self._uncached_by_request:
            self._uncached_by_request[request.request_id] = self._look_up_uncached(
                request
            )
        return self._uncached_by_request[request.request_id]

    def _look_up_uncached(self, request: Request) -> int:
        return request.num_tokens - self._admission_cached_tokens(request)

    def _admission_cached_tokens(self, request: Request) -> int:
        """Prompt tokens of a request that has computed none that its
        admission would find cached: its prefix in HBM and, with
        TPU_PREFIX_AWARE_RANK_WITH_TIER, the chunks of the connector's tier
        that follow it."""
        cached = self._cached_prefix_tokens(request)
        if self._tier is None or not self.kv_cache_manager.prefix_cache_lookup_enabled(
            request
        ):
            return cached
        return self._tier_prefix_once(request, cached)

    def _tier_prefix_once(self, request: Request, hbm_tokens: int) -> int:
        """Where the prefix admission would find cached ends, given the
        request's HBM prefix now. The tier is read once per stay in the
        waiting queue; the HBM prefix is read at every call."""
        request_id = request.request_id
        if request_id not in self._tier_prefix_by_request:
            self._tier_prefix_by_request[request_id] = self._tier_prefix_end(
                request, hbm_tokens
            )
        return max(hbm_tokens, self._tier_prefix_by_request[request_id])

    def _tier_prefix_end(self, request: Request, hbm_tokens: int) -> int:
        """The HBM prefix, extended by the tier's consecutive chunks after it,
        counted as the connector's lookup counts them: from the chunk that
        holds the HBM prefix's end, nothing when a fine-hash HBM prefix ends
        inside a chunk (vllm_torchtpu's chunk-aligned offload loads), at least
        one whole chunk, and without a flagged drafter group's last chunk.

        The read pins, reorders and stashes nothing, so the connector's state
        of the request stays as admission will find it.
        """
        tier = self._tier
        assert tier is not None
        state = tier._req_status.get(request.request_id)
        if state is None or request.skip_reading_prefix_cache:
            return hbm_tokens
        end = request.num_tokens - 1
        for group in tier.config.kv_group_configs:
            chunk_tokens = group.tokens_per_chunk
            hashes = group.hashes_per_chunk
            if hashes > 1 and hbm_tokens % chunk_tokens:
                return hbm_tokens
            first = hbm_tokens // chunk_tokens
            last = min(end // chunk_tokens, len(request.block_hashes) // hashes)
            keys = [
                make_offload_key(
                    request.block_hashes[(chunk + 1) * hashes - 1], group.group_idx
                )
                for chunk in range(first, last)
            ]
            chunks = self._count_tier_chunks(tier, keys, state.req_context)
            if group.is_eagle_group:
                chunks = max(chunks - 1, 0)
            end = min(end, (first + chunks) * chunk_tokens)
            if end - hbm_tokens < chunk_tokens:
                return hbm_tokens
        return max(hbm_tokens, end)

    @staticmethod
    def _count_tier_chunks(
        tier: OffloadingConnectorScheduler,
        keys: list[OffloadKey],
        req_context: ReqContext,
    ) -> int:
        """How many of ``keys``, from the first, a load could serve now or
        soon. The Raiden store answers with one side-effect-free probe;
        another manager is asked key by key, as admission asks it."""
        if not keys:
            return 0
        probe = getattr(tier.manager, "probe_warm_chunks", None)
        if probe is not None:
            return probe(keys)
        chunks = 0
        for key in keys:
            found = tier.manager.lookup(key, req_context)
            if found not in (LookupResult.HIT, LookupResult.HIT_PENDING):
                break
            chunks += 1
        return chunks

    def _cached_prefix_tokens(self, request: Request) -> int:
        """Prompt tokens of a request that has computed none in the prefix
        cache, as its admission would find them."""
        kvm = self.kv_cache_manager
        if not kvm.prefix_cache_lookup_enabled(request):
            return 0
        _, cached, _ = kvm.coordinator.find_longest_cache_hit(
            request.block_hashes, request.num_tokens - 1
        )
        return cached

    def _forget_requests_that_left_waiting(self) -> None:
        waiting = {request.request_id for request in self.waiting}
        for request_id in self._uncached_by_request.keys() - waiting:
            del self._uncached_by_request[request_id]
        for request_id in self._tier_prefix_by_request.keys() - waiting:
            del self._tier_prefix_by_request[request_id]

    def _free_sequence_slots(self) -> int:
        return (
            self.max_num_running_reqs
            - len(self.running)
            - self.num_waiting_for_streaming_input
        )

    def _decode_tokens(self) -> int:
        return sum(
            max(
                r.num_tokens_with_spec
                + r.num_output_placeholders
                - r.num_computed_tokens,
                0,
            )
            for r in self.running
            if not r.is_prefill_chunk
        )
