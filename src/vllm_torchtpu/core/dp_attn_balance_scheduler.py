# SPDX-License-Identifier: Apache-2.0
"""Async scheduler that caps each step's context-dependent attention work.

Under DP + EP every rank advances in lockstep: a step lasts as long as the
slowest rank's forward. For DeepSeek V4.1 prefill the only per-token cost that
grows with the cached context is the dense indexer scoring (index-source
layers score every compressed key), so a rank extending a request that sits on
a several-hundred-k-token context runs a much longer step than its peers, and
the peers idle in the next MoE collective. Over a minute the ranks see about
the same total work; the imbalance is a burst on whichever rank happens to be
extending a long context.

This scheduler bounds the burst locally. Per step it tracks
``W = sum(new_tokens * context)`` over the scheduled prefill chunks. A
request whose context is at least ``TPU_DP_ATTN_HEAVY_CTX`` tokens ("heavy")
gets at most ``(TPU_DP_ATTN_STEP_CAP - W) / context`` tokens; the first heavy
chunk of a step always gets ``TPU_DP_ATTN_MIN_CHUNK`` tokens so it keeps
advancing. Lighter requests are not clipped and fill the rest of the token
budget. The heavy request's prefill is spread over more, shorter steps while
the rank keeps the batch full with cheap tokens, which shortens the lockstep
step for every rank.

Enable with ``--scheduler-cls
vllm_torchtpu.core.dp_attn_balance_scheduler.TpuDpAttnBalanceScheduler``.
"""

import mmap
import os
import struct
import time
from typing import Any

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import RequestStatus

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


def _prefill_end(request: Any) -> int:
    return max(request.num_prompt_tokens, request.num_tokens - 1)


class TpuDpAttnBalanceScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._step_cap = float(os.environ.get("TPU_DP_ATTN_STEP_CAP", "4e8"))
        self._heavy_ctx = int(os.environ.get("TPU_DP_ATTN_HEAVY_CTX", "131072"))
        self._min_chunk = int(os.environ.get("TPU_DP_ATTN_MIN_CHUNK", "256"))
        self._admit_chunk = self.cache_config.block_size
        # Whale handling (TPU_P_WHALE_TOKENS > 0): a request with more than
        # that many prefill tokens left is a whale. While the rank has other
        # work, all whales together get at most TPU_P_WHALE_STEP_TOKENS per
        # step, and running requests go shortest-remaining-first, so a single
        # several-hundred-k-token prefill no longer blocks every other session
        # pinned to its rank (prefix routing) for a minute.
        self._whale_tokens = int(os.environ.get("TPU_P_WHALE_TOKENS", "0"))
        self._whale_step = int(os.environ.get("TPU_P_WHALE_STEP_TOKENS", "1024"))
        self._whale_used = 0
        self._whale_contended = False
        # Route every chunk through the split hook; keep the base mamba split
        # when the model actually needs it.
        self._mamba_split = self.need_mamba_block_aligned_split
        self.need_mamba_block_aligned_split = True
        self._step_work = 0.0
        self._step_demand = 0.0
        self._step_heavy = 0
        self._cap = self._step_cap
        self._stats = {
            "steps": 0, "clipped": 0, "deferred": 0, "heavy_tokens": 0, "raised": 0,
            "whale_steps": 0, "whale_clipped": 0,
        }
        self._last_log = time.monotonic()
        # TPU_DP_ATTN_SHARED=1: the DP engines of one server (same host) post
        # their unclipped per-step demand in a shared-memory slot array; a
        # rank only clips when no other rank wants a heavier step anyway, so
        # ranks that all have heavy work run it in the same steps.
        self._shared = None
        if os.environ.get("TPU_DP_ATTN_SHARED", "0") == "1":
            pc = self.vllm_config.parallel_config
            self._dp, self._rank = pc.data_parallel_size, pc.data_parallel_rank
            path = f"/dev/shm/vllm_dp_attn_{pc.data_parallel_master_port}.bin"
            fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
            try:
                if os.fstat(fd).st_size < 8 * self._dp:
                    os.ftruncate(fd, 8 * self._dp)
                self._shared = mmap.mmap(fd, 8 * self._dp)
            finally:
                os.close(fd)
        logger.info(
            "DP attention-balance scheduler: step cap %.3g token*ctx, heavy ctx "
            ">= %d, min heavy chunk %d, shared demand %s, whale > %d tokens "
            "(%d tokens/step when contended)",
            self._step_cap,
            self._heavy_ctx,
            self._min_chunk,
            self._shared is not None,
            self._whale_tokens,
            self._whale_step,
        )

    def _remaining(self, request: Any) -> int:
        return max(_prefill_end(request) - request.num_computed_tokens, 0)

    def _prepare_whales(self) -> None:
        """Order running requests shortest-remaining-first and decide whether
        whales must share this step with other work."""
        self._whale_used = 0
        self._whale_contended = False
        if self._whale_tokens <= 0:
            return
        self.running.sort(key=self._remaining)
        whales = sum(self._remaining(r) > self._whale_tokens for r in self.running)
        # Waiting requests have not looked up their prefix-cache hit yet, so
        # their remaining prefill is unknown here (a follow-up turn with a
        # 150k-token cached prefix looks like a whale); any waiting request
        # counts as other work. The split hook classifies with the real hit.
        self._whale_contended = len(self.running) > whales or len(self.waiting) > 0
        if self._whale_contended:
            self._stats["whale_steps"] += 1

    def _peer_demand(self) -> float:
        vals = struct.unpack_from(f"{self._dp}d", self._shared, 0)
        return max((v for r, v in enumerate(vals) if r != self._rank), default=0.0)

    def schedule(self, *args, **kwargs):
        self._step_work = 0.0
        self._step_demand = 0.0
        self._step_heavy = 0
        self._cap = self._step_cap
        if self._shared is not None:
            peer = self._peer_demand()
            if peer > self._cap:
                self._cap = peer
                self._stats["raised"] += 1
        self._prepare_whales()
        self._stats["steps"] += 1
        now = time.monotonic()
        if now - self._last_log > 60.0:
            self._last_log = now
            logger.info("DP attention-balance scheduler stats: %s", self._stats)
        out = super().schedule(*args, **kwargs)
        if self._shared is not None:
            struct.pack_into("d", self._shared, 8 * self._rank, self._step_demand)
        return out

    def _mamba_block_aligned_split(
        self,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        if num_new_tokens > 0 and start < _prefill_end(request):
            self._step_demand += float(num_new_tokens) * start
            if (
                self._whale_contended
                and _prefill_end(request) - start > self._whale_tokens
            ):
                cap = max(self._whale_step - self._whale_used, 0)
                if cap == 0 and request.status != RequestStatus.RUNNING:
                    cap = self._admit_chunk  # keep admitting the queue behind it
                if cap < num_new_tokens:
                    self._stats["whale_clipped"] += 1
                    num_new_tokens = cap
                self._whale_used += num_new_tokens
                if num_new_tokens == 0:
                    return 0
            if self._step_cap > 0 and start >= self._heavy_ctx:
                room = max(0.0, self._cap - self._step_work)
                cap = int(room // max(start, 1))
                if self._step_heavy == 0:
                    cap = max(cap, self._min_chunk)
                elif cap == 0 and request.status != RequestStatus.RUNNING:
                    # vLLM stops admitting the waiting queue at the first
                    # request that gets 0 tokens; admit this one with one
                    # block instead so lighter requests behind it still fill
                    # the step.
                    cap = self._admit_chunk
                if cap < num_new_tokens:
                    self._stats["clipped" if cap > 0 else "deferred"] += 1
                    num_new_tokens = cap
                if num_new_tokens > 0:
                    self._step_heavy += 1
                    self._stats["heavy_tokens"] += num_new_tokens
            if self._mamba_split and num_new_tokens > 0:
                num_new_tokens = super()._mamba_block_aligned_split(
                    request,
                    num_new_tokens,
                    num_new_local_computed_tokens,
                    num_external_computed_tokens,
                )
            self._step_work += float(num_new_tokens) * start
        elif self._mamba_split and num_new_tokens > 0:
            num_new_tokens = super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
        return num_new_tokens
