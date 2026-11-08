# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/program.py
"""The interface between vLLM's engine and a megakernel program.

A program is a whole model as a handful of JAX/Pallas SPMD ops. It owns
everything on the device -- weights, KV writes, recurrent state, sampling,
speculation -- and the runner (``megakernel.runner``) owns the vLLM contract:
request lifecycle, the KV connector, async output. They meet at the program's
``step`` method, which takes a :class:`StepInput` (the step's scheduled
requests as plain host data) and returns a :class:`StepOutput` carrying the
tokens they produced.

No vLLM dependencies: the standalone tools drive programs through this
interface.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy as np
    import torch


@dataclasses.dataclass(frozen=True)
class SamplingSpec:
    """The sampling parameters a program implements in its ops."""

    temperature: float = 0.0  # <= 0: greedy
    top_p: float = 1.0
    top_k: int = 0  # <= 0: all tokens
    seed: int = 0

    def as_tuple(self) -> tuple[float, float, int, int]:
        """Return ``(temperature, top_p, top_k, seed)`` as a plain 4-tuple.

        Returns:
            Tuple of ``(temperature, top_p, top_k, seed)``.
        """
        return (self.temperature, self.top_p, self.top_k, self.seed)


GREEDY = SamplingSpec()


@dataclasses.dataclass
class RequestInput:
    """Host-side state for one request in the runner's persistent batch."""

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
    """Static program capabilities derived from the engine configuration.

    The platform checks incoming requests against these capabilities before any
    program is loaded.
    """

    roles: frozenset[str]
    sampling: frozenset[str]
    speculation: str | None
    num_speculative_tokens: int
    max_step_tokens: int
    page_aligned_chunks: bool
    block_copies: bool
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
    """Device int32 token matrix ``[>= requests, width]`` produced by one step.

    Row ``i`` of ``tokens`` holds request ``i``'s tokens padded with ``-1``
    (all ``-1`` for a partial prefill chunk that produced no output token).
    """

    tokens: torch.Tensor
