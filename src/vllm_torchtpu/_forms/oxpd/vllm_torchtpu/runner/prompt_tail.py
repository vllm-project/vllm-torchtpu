# SPDX-License-Identifier: Apache-2.0
"""q38 ADM-P: ship only a prompt's tail to the PD decode workers.

A PD decode (kv consumer) admits requests whose prompt KV was landed from the
prefill, so its workers compute only positions >= num_computed_tokens and read
host tokens at most a short window below that (the qwen4_exp n-gram context:
(ngram_size - 1) + (spec_window - 1) tokens). Shipping the whole prompt as a
Python list costs O(prompt) on the admission step: pickling in the engine core,
unpickling in every worker, and InputBatch.add_request's list -> numpy copy
(~0.24 ms per 1K tokens on a 2.2 GHz Xeon).

With VLLM_TPU_PD_PROMPT_TAIL=W the decode engine core (DEC-HOST-A's half)
replaces NewRequestData.prompt_token_ids by ``PromptTail(len(p), offset,
p[offset:])`` with offset = num_computed_tokens - W, when that is positive and
the request reads no early prompt position (no penalties, prompt logprobs,
logits processors, embeds or multimodal inputs; no ngram drafter).

``len()`` is the FULL prompt length, so every length reader stays right
(CachedRequestState.num_prompt_tokens / num_tokens, InputBatch bookkeeping,
NewRequestData.prompt_len). Reads at or above ``start`` return the tail; any
read below ``start``, iteration, and numpy conversion raise PromptTailError,
so an unexpected early-position reader fails loudly instead of reading zeros.
"""

from __future__ import annotations

import numpy as np

# Host tokens the decode may read below num_computed_tokens (qwen4_exp n-gram
# window: (ngram_size - 1) + (spec_window - 1) < 10 for Qwen3.8). The worker
# refuses a tail that starts closer than this to num_computed_tokens.
MIN_TOKENS_BELOW_COMPUTED = 16


class PromptTailError(RuntimeError):
    pass


class WorkerTailStats:
    """Worker-side engagement counts (logged by the runner at the first
    accepted tail and every LOG_EVERY after)."""

    LOG_EVERY = 100
    accepted = 0
    tokens_not_received = 0

    @classmethod
    def record(cls, tail: "PromptTail") -> bool:
        cls.accepted += 1
        cls.tokens_not_received += tail.start
        return cls.accepted == 1 or cls.accepted % cls.LOG_EVERY == 0


class PromptTail:
    """The prompt's tokens at positions [start, length) of a length-``length``
    prompt (see the module docstring)."""

    __slots__ = ("length", "start", "tail")

    def __init__(self, length: int, start: int, tail) -> None:
        tail = list(tail)
        if not 0 <= start <= length or len(tail) != length - start:
            raise ValueError(
                f"PromptTail(length={length}, start={start}) with {len(tail)} tail tokens"
            )
        self.length = int(length)
        self.start = int(start)
        self.tail = tail

    def __len__(self) -> int:
        return self.length

    def _early(self, what: str) -> PromptTailError:
        return PromptTailError(
            f"{what} reads prompt positions below {self.start}, which were not "
            f"shipped to this worker (VLLM_TPU_PD_PROMPT_TAIL; prompt length "
            f"{self.length}). Turn the flag off for this workload."
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            lo, hi, step = idx.indices(self.length)
            if step != 1:
                raise self._early(f"slice {idx}")
            if lo >= hi:
                return []
            if lo < self.start:
                raise self._early(f"slice [{lo}:{hi}]")
            return self.tail[lo - self.start : hi - self.start]
        i = int(idx)
        if i < 0:
            i += self.length
        if not 0 <= i < self.length:
            raise IndexError(idx)
        if i < self.start:
            raise self._early(f"index {i}")
        return self.tail[i - self.start]

    def __iter__(self):
        raise self._early("iteration")

    def __array__(self, *args, **kwargs):
        raise self._early("numpy conversion")

    def __reduce__(self):
        return (PromptTail, (self.length, self.start, self.tail))

    def __repr__(self) -> str:
        return f"PromptTail(length={self.length}, start={self.start}, tail_len={len(self.tail)})"

    def zeros(self) -> np.ndarray:
        """A full-length, zero-strided int32 view (O(1) memory): the stand-in
        add_request copies in O(length) without a Python-level loop."""
        return np.broadcast_to(np.int32(0), (self.length,))


def check_admissible(new_req, num_computed_tokens: int, spec_method: str | None) -> None:
    """Raise unless a request shipped with a PromptTail reads no prompt
    position below the tail (the worker-side copy of the engine core's trim
    conditions)."""
    tail = new_req.prompt_token_ids
    problems = []
    if num_computed_tokens - tail.start < MIN_TOKENS_BELOW_COMPUTED:
        problems.append(
            f"tail starts at {tail.start}, within {MIN_TOKENS_BELOW_COMPUTED} of "
            f"num_computed_tokens {num_computed_tokens}"
        )
    if getattr(new_req, "prompt_embeds", None) is not None:
        problems.append("prompt_embeds")
    if getattr(new_req, "prompt_is_token_ids", None) is not None:
        problems.append("prompt_is_token_ids")
    if getattr(new_req, "mm_features", None):
        problems.append("multimodal features")
    if spec_method == "ngram":
        problems.append("the ngram drafter reads the full token history")
    sp = getattr(new_req, "sampling_params", None)
    if sp is not None:
        if getattr(sp, "frequency_penalty", 0.0) != 0.0:
            problems.append("frequency_penalty")
        if getattr(sp, "presence_penalty", 0.0) != 0.0:
            problems.append("presence_penalty")
        if getattr(sp, "repetition_penalty", 1.0) != 1.0:
            problems.append("repetition_penalty")
        if getattr(sp, "prompt_logprobs", None) is not None:
            problems.append("prompt_logprobs")
        if getattr(sp, "logits_processors", None):
            problems.append("logits_processors")
    if problems:
        raise PromptTailError(
            f"request {getattr(new_req, 'req_id', '?')} arrived with a {tail!r} but "
            f"reads early prompt positions: {', '.join(problems)}"
        )
