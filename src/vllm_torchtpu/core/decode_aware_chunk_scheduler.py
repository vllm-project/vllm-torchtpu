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
"""Share each step's token budget among the prompts still being prefilled.

With chunked prefill every step carries the running decodes plus prefill
chunks. vLLM hands a whole step's budget (8192 tokens here) to the first
waiting prompt, so concurrent 200K prompts are prefilled one after another,
and a request that has started decoding emits one token per *chunk* of the
others' prefills -- ~0.7 s per token at 200K.

Splitting the budget evenly over the `n` prompts in prefill (a per-request
chunk cap of `budget / n`, block aligned, never below
`TPU_PREFILL_CHUNK_FLOOR`) keeps every step full -- one prompt alone still
gets the whole 8192 -- while the prompts advance together and finish
together, so their decodes then run without prefill chunks in between.
Measured at 200K on one TPU7x host: a fixed 2048 cap took c4 TPOT from 111 ms
to 7 ms but cost 32% TTFT at c1, which this avoids.

Select it with ``--scheduler-cls`` set to
``vllm_torchtpu.core.decode_aware_chunk_scheduler.SharedPrefillBudgetScheduler``.
"""

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


class SharedPrefillBudgetScheduler(AsyncScheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._configured_threshold = self.scheduler_config.long_prefill_token_threshold
        self._floor = envs.TPU_PREFILL_CHUNK_FLOOR
        self._align = self.cache_config.block_size
        logger.info(
            "SharedPrefillBudgetScheduler: step budget %d split over prefilling "
            "prompts (floor %d, aligned to %d).",
            self.max_num_scheduled_tokens,
            self._floor,
            self._align,
        )

    def schedule(self, *args, **kwargs) -> SchedulerOutput:
        decoding = sum(
            req.num_computed_tokens >= req.num_prompt_tokens for req in self.running
        )
        # Only prompts that can hold a running slot share the budget; the rest
        # of `waiting` cannot be scheduled this step anyway.
        prefilling = min(
            len(self.running) - decoding + len(self.waiting),
            self.max_num_running_reqs - decoding,
        )
        threshold = self._configured_threshold
        if prefilling > 1:
            share = self.max_num_scheduled_tokens // prefilling
            share = max(self._floor, share // self._align * self._align)
            threshold = min(threshold, share) if threshold > 0 else share
        self.scheduler_config.long_prefill_token_threshold = threshold
        return super().schedule(*args, **kwargs)
