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
"""Interleaved scheduler subclass for decode-owning megakernel programs.

When a `DecodeOwner` owns decode steps (`move_owned_to_front`), two invariants
must hold on the engine-core scheduler:

1. **Prompt-only prefix caching**: Owned decode steps bypass vLLM's KV and
   Mamba pools, so generated assistant-output blocks in vLLM's pools remain
   unwritten. Both `kv_cache_manager.cache_blocks` and
   `kv_cache_manager.coordinator.cache_blocks` are capped at
   `request.num_prompt_tokens` so multi-turn chats never hit unwritten
   generated blocks from an earlier turn.
2. **Interleaved prefill & non-preemptive admission**: Running owned requests
   decode every step, prefill tokens in a step are capped at
   `max_num_batched_tokens` (aligned to `block_size`), and waiting requests are
   admitted at most one per step when free KV blocks cover both the candidate's
   full length and the remaining decode growth of all running requests.
"""

from __future__ import annotations

import functools
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

SCHEDULER_CLS = (
    "vllm_torchtpu.megakernel.adapter.decode_owner_scheduler.DecodeOwnerScheduler"
)


def _cap_instance_cache_blocks(kv_cache_manager: Any) -> None:
    """Cap `cache_blocks` on `kv_cache_manager` and its `coordinator` at prompt end."""
    for target in (
        kv_cache_manager,
        getattr(kv_cache_manager, "coordinator", None),
    ):
        if target is None:
            continue
        orig = getattr(target, "cache_blocks", None)
        if orig is None or getattr(orig, "_decode_owner_prompt_cap", False):
            continue

        @functools.wraps(orig)
        def _capped(
            request: Any,
            num_computed_tokens: int,
            *args: Any,
            _orig: Any = orig,
            **kwargs: Any,
        ) -> Any:
            prompt_toks = int(
                getattr(request, "num_prompt_tokens", num_computed_tokens)
                or num_computed_tokens
            )
            if num_computed_tokens > prompt_toks:
                num_computed_tokens = prompt_toks
            return _orig(request, num_computed_tokens, *args, **kwargs)

        _capped._decode_owner_prompt_cap = True  # type: ignore[attr-defined]
        target.cache_blocks = _capped


class DecodeOwnerScheduler(Scheduler):
    """Scheduler for engines running with a `DecodeOwner` megakernel plugin."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._orig_need_mamba_split = bool(self.need_mamba_block_aligned_split)
        self.need_mamba_block_aligned_split = True
        self._prefill_used = 0
        _cap_instance_cache_blocks(getattr(self, "kv_cache_manager", None))
        logger.info(
            "DecodeOwnerScheduler active (block_size=%d, max_num_batched_tokens=%d).",
            self._block_size(),
            int(self.scheduler_config.max_num_batched_tokens),
        )

    def _block_size(self) -> int:
        bs = int(getattr(self.cache_config, "block_size", 0) or 0)
        if bs <= 0:
            raise ValueError(
                f"DecodeOwnerScheduler requires cache_config.block_size > 0, got {bs}."
            )
        return bs

    def _admit_fits(self, cand: Any) -> bool:
        kvm = getattr(self, "kv_cache_manager", None)
        block_pool = getattr(kvm, "block_pool", None)
        if block_pool is None:
            return True
        bs = self._block_size()
        vcfg = getattr(self, "vllm_config", None)
        mcfg = getattr(vcfg, "model_config", None)
        max_len = int(
            getattr(mcfg, "max_model_len", 0) or getattr(self, "max_model_len", 0) or 0
        )

        def full(req: Any) -> int:
            tot = int(req.num_prompt_tokens) + int(req.max_tokens)
            if max_len > 0:
                tot = min(tot, max_len)
            return (tot + bs - 1) // bs + 4

        growth = sum(
            max(0, full(r) - (int(r.num_computed_tokens) + bs - 1) // bs)
            for r in self.running
        )
        free = int(block_pool.get_num_free_blocks())
        return free - full(cand) >= growth

    def _mamba_block_aligned_split(
        self,
        request: Any,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        out = (
            super()._mamba_block_aligned_split(
                request,
                num_new_tokens,
                num_new_local_computed_tokens,
                num_external_computed_tokens,
            )
            if self._orig_need_mamba_split
            else num_new_tokens
        )
        start = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        prefill_end = max(request.num_prompt_tokens, request.num_tokens - 1)
        if start >= prefill_end or out <= 0:
            return out
        bs = self._block_size()
        chunk_cap = int(self.scheduler_config.max_num_batched_tokens)
        left = max(0, chunk_cap - self._prefill_used)
        if out > left:
            cut = (left // bs) * bs
            out = cut if (cut > 0 or self._prefill_used > 0) else out
        elif start + out < prefill_end and out >= bs:
            out = (out // bs) * bs
        self._prefill_used += out
        return out

    def schedule(self, throttle_prefills: bool = False) -> Any:
        self._prefill_used = 0
        _cap_instance_cache_blocks(getattr(self, "kv_cache_manager", None))
        q = self.kv_holding_waiting if self.kv_holding_waiting else self.waiting
        num_active = len(self.running) + int(
            getattr(self, "num_waiting_for_streaming_input", 0)
        )
        blocked = (
            bool(q) and bool(self.running) and not self._admit_fits(q.peek_request())
        )
        cap = self.max_num_active_reqs
        self.max_num_active_reqs = num_active if blocked else min(cap, num_active + 1)
        try:
            return super().schedule(throttle_prefills)
        finally:
            self.max_num_active_reqs = cap
