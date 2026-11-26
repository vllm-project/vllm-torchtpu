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
"""Scheduler-side gather-page budget for token-parallel prefill.

Token-parallel DCP prefill copies every virtual page of each prefill request
in a step into one gather buffer of `max_pages` pages, and the runner refuses a
step that needs more (`dcp_token_parallel_op.check_step_capacity`). Without
prefix caching only the running chunked prefill carries history, so a step needs
at most one max-model-len request plus one chunk, which is what the buffer holds.
A prefix-cache hit hands a newly admitted request its whole cached history at
once, so a few such requests in one step overflow the buffer.

`install_token_parallel_page_budget` makes the scheduler stop adding prefill
work to a step once its pages would exceed the buffer: a waiting request is left
for a later step and a running prefill skips the step, exactly as vLLM handles a
chunk it cannot fit (`Scheduler._reserve_prefill_lookahead` returning 0).
"""

import functools
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import glm_dsa_v7x_active

logger = init_logger(__name__)

# NOTE(cross-boundary-contract): `dcp_token_parallel_op.layout_for` sizes the
# gather buffer as ceil(max_model_len / (block_size * dcp)) + EXTRA_PAGES. This
# module mirrors the formula instead of importing the op module, whose kernel
# imports do not belong in the engine-core (scheduler) process.
# tests/test_token_parallel_page_budget.py locks the equality.
EXTRA_GATHER_PAGES = 4
# NOTE(cross-boundary-contract): the runner orders the batch [decode][prefill]
# with `reorder_batch_threshold = 1` (`tpu_runner._reorder_batch_for_rpa`), and
# decode rows sit inside `split_rows` on the DCP path, so a request scheduling
# one token never enters the gather buffer.
DECODE_ROW_TOKENS = 1
_LOG_EVERY = 1000
_PATCH_MARK = "_tpu_token_parallel_page_budget"
# vLLM's per-chunk fit check, called in both scheduling loops right before slot
# allocation; returning 0 skips the chunk without preempting anything.
CHUNK_FIT_HOOK = "_reserve_prefill_lookahead"
_USED_PAGES = "_tpu_token_parallel_used_pages"
_BUDGET = "_tpu_token_parallel_budget"
_DEFERRALS = "_tpu_token_parallel_deferrals"


@dataclass(frozen=True)
class PageBudget:
    virtual_page_size: int
    max_pages: int

    def request_pages(self, num_computed_tokens: int, num_new_tokens: int) -> int:
        seq_len = num_computed_tokens + num_new_tokens
        return -(-seq_len // self.virtual_page_size)


def budget_for(vllm_config: Any) -> PageBudget | None:
    """The step's page budget, or None where the gather buffer cannot overflow."""
    if vllm_config is None:
        return None
    model_cfg = getattr(vllm_config, "model_config", None)
    parallel_cfg = getattr(vllm_config, "parallel_config", None)
    cache_cfg = getattr(vllm_config, "cache_config", None)
    sched_cfg = getattr(vllm_config, "scheduler_config", None)
    if (
        model_cfg is None
        or parallel_cfg is None
        or cache_cfg is None
        or sched_cfg is None
        or not glm_dsa_v7x_active(model_cfg)
    ):
        return None
    dcp = int(getattr(parallel_cfg, "decode_context_parallel_size", 1))
    il = int(getattr(parallel_cfg, "cp_kv_cache_interleave_size", 1))
    block_size = int(getattr(cache_cfg, "block_size", 0))
    if (
        dcp <= 1
        or il < 1
        or not bool(getattr(cache_cfg, "enable_prefix_caching", False))
        or block_size <= 0
        or block_size % (dcp * il) != 0
        or getattr(vllm_config, "lora_config", None) is not None
        or getattr(vllm_config, "speculative_config", None) is not None
        or getattr(vllm_config, "kv_transfer_config", None) is not None
    ):
        return None
    virtual_page_size = block_size * dcp
    max_num_seqs = int(sched_cfg.max_num_seqs)
    max_batched = int(getattr(sched_cfg, "max_num_batched_tokens", 0) or 0)
    max_model_len = int(model_cfg.max_model_len)
    token_budget = max(max_model_len, max_batched)
    return PageBudget(
        virtual_page_size=virtual_page_size,
        max_pages=-(-token_budget // virtual_page_size)
        + max_num_seqs
        + EXTRA_GATHER_PAGES,
    )


def admit_new_tokens(
    budget: PageBudget, used_pages: int, num_computed_tokens: int, num_new_tokens: int
) -> tuple[int, int]:
    """(tokens to schedule, step pages after) for one request's chunk.

    Returns 0 tokens when the chunk's pages would push the step past the buffer.
    """
    if num_new_tokens <= DECODE_ROW_TOKENS:
        return num_new_tokens, used_pages
    pages = budget.request_pages(num_computed_tokens, num_new_tokens)
    if used_pages + pages > budget.max_pages:
        return 0, used_pages
    return num_new_tokens, used_pages + pages


def _scheduler_budget(scheduler: Any) -> PageBudget | None:
    state = scheduler.__dict__
    if _BUDGET not in state:
        state[_BUDGET] = budget_for(scheduler.vllm_config)
    return state[_BUDGET]


def _log_deferral(scheduler: Any, budget: PageBudget, pages_needed: int) -> None:
    count = scheduler.__dict__.get(_DEFERRALS, 0) + 1
    scheduler.__dict__[_DEFERRALS] = count
    if count == 1 or count % _LOG_EVERY == 0:
        logger.info(
            "Token-parallel prefill: deferred a prefill chunk needing %d gather "
            "pages to the next step (this step holds %d of %d); %d deferrals so "
            "far. Expected when several prefix-cached long prompts arrive "
            "together; frequent deferrals mean prefix-cache admission is "
            "page-bound, not token-bound.",
            pages_needed,
            scheduler.__dict__.get(_USED_PAGES, 0),
            budget.max_pages,
            count,
        )


def install_token_parallel_page_budget() -> None:
    """Patch vLLM's Scheduler to keep each step inside the gather buffer."""
    original_reserve = getattr(Scheduler, CHUNK_FIT_HOOK)
    if getattr(original_reserve, _PATCH_MARK, False):
        return
    original_schedule = Scheduler.schedule

    @functools.wraps(original_schedule)
    def schedule(self, *args, **kwargs):
        self.__dict__[_USED_PAGES] = 0
        return original_schedule(self, *args, **kwargs)

    @functools.wraps(original_reserve)
    def reserve_within_gather_buffer(
        self, request, num_computed_tokens, num_new_tokens
    ):
        num_new_tokens = original_reserve(
            self, request, num_computed_tokens, num_new_tokens
        )
        budget = _scheduler_budget(self)
        if budget is None:
            return num_new_tokens
        admitted, used = admit_new_tokens(
            budget,
            self.__dict__.get(_USED_PAGES, 0),
            num_computed_tokens,
            num_new_tokens,
        )
        if admitted == 0 and num_new_tokens > 0:
            _log_deferral(
                self, budget, budget.request_pages(num_computed_tokens, num_new_tokens)
            )
        self.__dict__[_USED_PAGES] = used
        return admitted

    for wrapper in (schedule, reserve_within_gather_buffer):
        setattr(wrapper, _PATCH_MARK, True)
    Scheduler.schedule = schedule
    setattr(Scheduler, CHUNK_FIT_HOOK, reserve_within_gather_buffer)
