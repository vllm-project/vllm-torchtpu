# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The interface between vLLM's engine and a megakernel program.

A program is a whole model as a handful of JAX/Pallas SPMD ops. It owns
everything on the device -- weights, KV writes, recurrent state, sampling,
speculation -- and the runner (``megakernel.runner``) owns the vLLM contract:
request lifecycle, the KV connector, async output. They meet at
:meth:`MegakernelProgram.step`, which takes a :class:`StepInput` (the step's
scheduled requests as plain host data) and returns the tokens they produced.

No vLLM dependencies: the standalone tools drive programs through this
interface exactly as serving does.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class SamplingSpec:
    """The sampling parameters a program implements in its ops."""

    temperature: float = 0.0  # <= 0: greedy
    top_p: float = 1.0
    top_k: int = 0  # <= 0: all tokens
    seed: int = 0

    @classmethod
    def from_sampling_params(cls, params: Any, req_id: str) -> SamplingSpec:
        """Request ``req_id``'s vLLM ``SamplingParams``: greedy below vLLM's
        sampling epsilon (1e-5), and an unseeded request seeded from its id
        (the ops sample with a seeded generator, which every rank must agree
        on)."""
        return cls(
            temperature=float(params.temperature) if params.temperature > 1e-5 else 0.0,
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            seed=int(params.seed)
            if params.seed is not None
            else zlib.crc32(req_id.encode()),
        )

    def as_tuple(self) -> tuple[float, float, int, int]:
        return (self.temperature, self.top_p, self.top_k, self.seed)


GREEDY = SamplingSpec()


@dataclasses.dataclass
class RequestInput:
    """One request of a step, as the runner's persistent batch holds it
    (host data only)."""

    req_id: str
    computed: int  # tokens in the KV cache before this step
    scheduled: int  # tokens scheduled this step (0: not scheduled)
    # The request's token row (prompt, then outputs; the first num_tokens are known).
    token_ids: np.ndarray
    # tokens known (async scheduling: placeholders for outputs still on the device)
    num_tokens: int
    num_prompt_tokens: int
    # The request's block-table row in every KV cache group.
    block_tables: tuple[np.ndarray, ...]
    sampling: SamplingSpec = GREEDY

    @property
    def end(self) -> int:
        """Tokens in the KV cache after this step."""
        return self.computed + self.scheduled

    @property
    def completes_prompt(self) -> bool:
        return self.end >= self.num_prompt_tokens

    @property
    def partial(self) -> bool:
        """A prefill chunk that leaves known tokens unscheduled: no token this step."""
        return self.end < self.num_tokens


@dataclasses.dataclass(frozen=True)
class KVGroup:
    """One of vLLM's KV cache groups: its layers and its block size."""

    layer_names: tuple[str, ...]
    # Tokens per block-table entry, as the attention metadata builders index it.
    block_size: int
    table_width: int  # entries in a request's block-table row


@dataclasses.dataclass(frozen=True)
class Capabilities:
    """What a program supports, known from the configuration alone (the
    platform checks requests against it before any program is loaded)."""

    roles: frozenset[str]
    sampling: frozenset[
        str
    ]  # SamplingParams fields the ops implement beyond the defaults
    speculation: str | None  # the speculation method drafted in the ops, if any
    num_speculative_tokens: int
    max_step_tokens: int  # the most tokens one step may schedule
    # prefill chunks must end on KV pages (the scheduler's align-mode split does it)
    page_aligned_chunks: bool
    # applies the scheduler's copy-on-write block copies (prefix hits inside a block)
    block_copies: bool
    # applies the grammar bitmask of vLLM structured outputs to the tokens the
    # ops choose (the prefill declares it to accept the requests; it ignores
    # the masks -- only the decode role's tokens are served)
    structured_outputs: bool = False


@dataclasses.dataclass
class StepInput:
    """Everything one step hands the program."""

    # The batch's requests in the runner's batch order (scheduled or not).
    requests: Sequence[RequestInput]
    # Requests finished since the last step: the program drops their state.
    finished: Iterable[str] = ()
    # Copy-on-write block copies (src, dst) to apply before the step.
    copies: Sequence[tuple[int, int]] = ()
    # Structured-output grammar masks: row ``i`` is request ``i``'s xgrammar
    # bitmask, ``ceil(vocab_size / 32)`` int32 words with bit ``j`` of word
    # ``w`` the allowed-ness of token ``32 * w + j`` (1: allowed); an all-ones
    # row is unconstrained. None: no structured request this step.
    mask: np.ndarray | None = None


@dataclasses.dataclass
class StepOutput:
    """The tokens a step produced: row ``i`` of ``tokens`` (device int32
    ``[>= requests, width]``) holds request ``i``'s tokens, ``-1`` padded
    (all ``-1`` for a request that produced none: a partial prefill chunk)."""

    tokens: torch.Tensor


class MegakernelProgram(Protocol):
    """A model run as megakernel ops (see the module docstring)."""

    def load_weights(self) -> None: ...

    def bind_kv_caches(
        self, caches: Mapping[str, torch.Tensor], groups: Sequence[KVGroup]
    ) -> None:
        """vLLM's allocated KV cache tensors by layer name and its KV cache
        groups."""
        ...

    def warmup(self) -> None:
        """Compile every op the program will run."""
        ...

    def step(self, step_input: StepInput) -> StepOutput:
        """Run one step (see :class:`StepInput` and :class:`StepOutput`)."""
        ...
