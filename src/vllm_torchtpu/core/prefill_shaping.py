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
"""Prefill shaping policies of ``PrefixAwareScheduler``.

Each is a ``tpu_scheduler`` option (core/scheduler_options.py), off by
default and refused unless ``prefix_aware_levers`` is on:

* ``prefill_chunk_cap`` (:class:`PrefillChunkCap`): while many requests
  decode, a cold prefill gets a bounded chunk of a step and a step that
  already carries a prefill takes further prefill tokens only up to a step
  total, so decoders do not share a step with a full-budget prefill.
* ``srpf_running`` (:func:`remaining_prompt_tokens`): running prefills are
  scheduled fewest remaining prompt tokens first.
* ``admission_tiers`` (:func:`admission_tier`, :func:`free_blocks_for_fit`):
  the warmest-first ranking puts a warm request whose uncached blocks fit the
  free blocks ahead of everything when the queue's head does not fit them,
  and warm requests ahead of cold ones.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Admission tiers of the warmest-first ranking; lower admits earlier. The
# default ranking's two tiers are TIER_PREEMPTED and TIER_WARM.
TIER_FREE_BLOCK_FIT = 0
TIER_PREEMPTED = 1
TIER_WARM = 2
TIER_COLD = 3

_LOG_INTERVAL_S = 60.0


def _prefill_end(request: Any) -> int:
    """The position a prefill runs to; past the prompt for a request resumed
    from preemption, which replays its output tokens."""
    return max(request.num_prompt_tokens, request.num_tokens - 1)


def _num_blocks(tokens: int, block_size: int) -> int:
    return -(-max(tokens, 0) // block_size)


def remaining_prompt_tokens(request: Any) -> int:
    """The SRPF key of a running prefill."""
    return max(request.num_prompt_tokens - request.num_computed_tokens, 0)


def admission_tier(
    request: Any, warmth: int, free_blocks_for_fit: int | None, block_size: int
) -> int:
    """The tier of a waiting request with ``warmth`` cached prompt tokens.

    ``free_blocks_for_fit`` is the free block count when the queue's head
    does not fit it (:func:`free_blocks_for_fit`), else None.
    """
    if (
        free_blocks_for_fit is not None
        and warmth > 0
        and _num_blocks(request.num_tokens - warmth, block_size) <= free_blocks_for_fit
    ):
        return TIER_FREE_BLOCK_FIT
    if request.num_preemptions > 0:
        return TIER_PREEMPTED
    return TIER_WARM if warmth > 0 else TIER_COLD


def free_blocks_for_fit(
    free_blocks: int | None, head_cold_tokens: int, block_size: int
) -> int | None:
    """``free_blocks`` when the head's ``head_cold_tokens`` uncached tokens
    need more blocks than that, else None."""
    if free_blocks is None:
        return None
    if free_blocks < _num_blocks(head_cold_tokens, block_size):
        return free_blocks
    return None


@dataclass(frozen=True)
class ChunkCapLimits:
    """The limits of :class:`PrefillChunkCap`, in tokens except
    ``cold_min_decoders``."""

    cold_chunk_cap: int
    """A cold prefill's chunk plus the step's decode tokens."""
    cold_threshold: int
    """A prefill with more prompt tokens than this left is cold."""
    cold_min_decoders: int
    """Decoding requests from which the caps apply."""
    step_total_cap: int
    """The step's tokens once it carries more than one prefill chunk."""

    @classmethod
    def derive(cls, token_budget: int, max_num_seqs: int) -> "ChunkCapLimits":
        """Half the token budget (at least 1024) for the step, half of that
        (at least 512) for a cold chunk, the budget as the cold threshold,
        and three quarters of ``max_num_seqs``, counted up to 16, as the
        decoder count. A budget of 8192 gives 4096 / 2048 / 8192, and
        ``max_num_seqs`` 16 or more gives 12 decoders."""
        step_total_cap = max(1024, token_budget // 2)
        return cls(
            cold_chunk_cap=max(512, step_total_cap // 2),
            cold_threshold=token_budget,
            cold_min_decoders=max(1, 3 * min(max_num_seqs, 16) // 4),
            step_total_cap=step_total_cap,
        )


class PrefillChunkCap:
    """Caps prefill chunks against the decoders of the step (option
    ``prefill_chunk_cap``).

    ``begin_step`` records the running set before the scheduler's loops run;
    ``cap`` sizes each prefill chunk before the Mamba block alignment (or,
    without it, the prefill-lookahead reserve), and ``record`` keeps the
    chunk those hooks finally leave. A chunk counts from then on: one whose
    block allocation then fails without a preemption still counts for the
    rest of the step.
    """

    def __init__(self, limits: ChunkCapLimits, block_size: int) -> None:
        self.limits = limits
        self.block_size = block_size
        self._num_decoding = 0
        self._decode_tokens = 0
        self._running_prefills: set[str] = set()
        # Prefill chunks taken this step, by request id.
        self._chunks: dict[str, int] = {}
        self._steps = 0
        self._capped = 0
        self._held = 0
        self._log_t = time.monotonic()

    def begin_step(self, running: Sequence[Any]) -> None:
        """Starts a step over the running set. A request decodes once it has
        computed its whole prefill; a request resumed from preemption
        prefills until it has replayed its output tokens."""
        decoding = [r for r in running if r.num_computed_tokens >= _prefill_end(r)]
        self._num_decoding = len(decoding)
        # Each decoder schedules its unverified tokens, at least one.
        self._decode_tokens = sum(
            max(
                r.num_tokens_with_spec
                + r.num_output_placeholders
                - r.num_computed_tokens,
                1,
            )
            for r in decoding
        )
        self._running_prefills = {
            r.request_id for r in running if r.num_computed_tokens < _prefill_end(r)
        }
        self._chunks = {}
        self._steps += 1
        self._maybe_log()

    def cap(
        self, request: Any, start: int, num_new_tokens: int, has_waiting: bool
    ) -> int:
        """The chunk of ``request``'s prefill from ``start`` computed tokens,
        at most ``num_new_tokens``; 0 holds the prefill for this step.

        With at least ``cold_min_decoders`` requests decoding:

        * a prefill behind another prefill of the step gets what is left of
          ``step_total_cap`` after the decode tokens and the earlier chunks,
          and is held when that is under a block and does not finish it;
        * a cold prefill gets at most ``cold_chunk_cap`` minus the decode
          tokens, but at least a block; behind another prefill, what is left
          of ``cold_chunk_cap`` after the step's tokens when that is at least
          a block.

        A cold prefill already running gets the cold cap with fewer decoders
        too while ``has_waiting``.
        """
        prefill_end = _prefill_end(request)
        remaining = prefill_end - start
        if remaining <= 0 or num_new_tokens <= 0:
            return num_new_tokens
        limits = self.limits
        block_size = self.block_size
        req_id = request.request_id
        earlier = [n for rid, n in self._chunks.items() if rid != req_id]
        behind_prefill = any(n > 0 for n in earlier)
        step_tokens = self._decode_tokens + sum(earlier)
        many_decoders = self._num_decoding >= limits.cold_min_decoders
        chunk = num_new_tokens
        if behind_prefill and many_decoders:
            room = limits.step_total_cap - step_tokens
            if room <= 0 or (room < block_size and start + room < prefill_end):
                self._chunks[req_id] = 0
                self._held += 1
                return 0
            chunk = min(chunk, room)
        if remaining > limits.cold_threshold and (
            many_decoders or (req_id in self._running_prefills and has_waiting)
        ):
            cold_room = limits.cold_chunk_cap - step_tokens
            if not (behind_prefill and cold_room >= block_size):
                cold_room = max(block_size, limits.cold_chunk_cap - self._decode_tokens)
            chunk = min(chunk, cold_room)
        if chunk < num_new_tokens:
            self._capped += 1
        self._chunks[req_id] = chunk
        return chunk

    def record(self, request: Any, num_tokens: int) -> None:
        """Keeps the chunk the scheduler took for a prefill ``cap`` sized."""
        if request.request_id in self._chunks:
            self._chunks[request.request_id] = num_tokens

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._log_t < _LOG_INTERVAL_S:
            return
        self._log_t = now
        logger.info(
            "prefill-chunk-cap telemetry: steps=%d capped=%d held=%d",
            self._steps,
            self._capped,
            self._held,
        )
