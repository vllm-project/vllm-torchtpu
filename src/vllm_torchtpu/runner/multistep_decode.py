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
"""Greedy decode chunks that produce several tokens per row per step.

Multi-step decode (``TPU_MULTISTEP_DECODE=K``, scheduler side in
``core/multistep_decode.py``): a burst step schedules every running request
``k <= K`` tokens, ``k - 1`` of them placeholder drafts, so vLLM allocates
their KV slots, counts ``k`` output placeholders and cuts the tokens past a
stop. The runner presents the burst to its input preparation as the 1-token
decode step its first iteration is (``decode_view``), and decodes the ``k``
iterations of each chunk with the same compiled programs: after an
iteration's greedy tokens, ``advance_decode_inputs`` makes them the next
iteration's input ids, one position later on every real row; the attention
derives each new token's KV slot from ``seq_lens`` and the block tables the
scheduler allocated for all ``k`` positions.

A K-step decode backend (``TPU_K_STEP_DECODE_BACKEND``) replaces the backbone,
the LM head and the argmax of a decode-only chunk with one call that returns
``steps`` greedy tokens per row (1 outside a burst, ``k`` in one). The runner
calls it for every decode-only chunk of a step whose requests the top-token
path can serve (greedy, no logprobs, no grammar, none below min_tokens, no
drafts); the tokens then take the ``TPU_GREEDY_TOP_TOKENS`` path of
``sample_tokens``. Every other decode chunk keeps the XLA programs, counted
by reason (``KStepPathCounter``).
"""

from __future__ import annotations

import collections
import copy
import dataclasses
import importlib
from typing import TYPE_CHECKING, Any, Protocol

import torch

from vllm_torchtpu.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata

logger = init_logger(__name__)

# Decode steps between two INFO summaries of the per-reason counts.
_SUMMARY_EVERY_STEPS = 20000


class KStepDecodeBackend(Protocol):
    """Decodes ``steps`` greedy tokens for every row of a decode-only chunk."""

    def supports(self, num_reqs: int, padded_num_reqs: int, steps: int) -> bool:
        """Whether ``decode`` serves this chunk; False keeps the XLA programs."""
        ...

    def decode(
        self,
        *,
        steps: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict[str, AttentionMetadata],
        num_reqs: int,
        padded_num_reqs: int,
        kv_caches: list[torch.Tensor],
    ) -> torch.Tensor:
        """The chunk's tokens, int64 ``[padded_num_reqs, steps]``.

        ``input_ids`` / ``positions``: int32 ``[num_tokens_padded]``, row ``r``
        is request ``r``'s token at its position (async-substituted, on
        device). ``attn_metadata``: the per-layer metadata of a 1-token decode
        step (``seq_lens`` = position + 1 on real rows and 0 on padding,
        ``query_start_loc``, ``request_distribution``, flat ``block_tables``
        per KV cache group, already covering positions .. positions + steps -
        1). Entry ``[r, j]`` is the greedy token at ``positions[r] + j + 1``,
        ties to the lowest id as ``torch.argmax``; rows past ``num_reqs`` are
        ignored. ``kv_caches`` (the runner's, bound per layer) are written in
        place at every position decoded; no stop rule applies inside the call.
        """
        ...

    def warmup(self, *, kv_caches: list[torch.Tensor]) -> None:
        """Build the backend's programs before serving (``capture_model``)."""
        ...


def load_backend(spec: str, runner: Any) -> KStepDecodeBackend | None:
    """The backend ``TPU_K_STEP_DECODE_BACKEND`` names (``module:factory``,
    the factory taking the runner), or None when it is unset."""
    if not spec:
        return None
    module_name, sep, factory_name = spec.partition(":")
    if not sep or not module_name or not factory_name:
        raise ValueError(
            f"TPU_K_STEP_DECODE_BACKEND must be 'module:factory', got {spec!r}"
        )
    factory = getattr(importlib.import_module(module_name), factory_name)
    return factory(runner)


class KStepPathCounter:
    """Counts decode-only chunks that kept the XLA programs, by reason.

    The first chunk per reason is logged at INFO with its request ids, then a
    summary every ``_SUMMARY_EVERY_STEPS`` decode chunks; served traffic only
    (the runner's warmup requests have ids ``__..__``).
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.counts: collections.Counter[str] = collections.Counter()
        self.chunks = 0

    def record(self, reason: str | None, req_ids: list[str | None]) -> None:
        if all(req_id is None or req_id.startswith("__") for req_id in req_ids):
            return
        self.chunks += 1
        if reason is not None:
            if reason not in self.counts:
                logger.info(
                    "%s: a decode chunk keeps the XLA programs (%s), requests %s",
                    self.name,
                    reason,
                    [r for r in req_ids if r is not None][:4],
                )
            self.counts[reason] += 1
        if self.chunks % _SUMMARY_EVERY_STEPS == 0:
            logger.info(
                "%s: %d decode chunks, kept the XLA programs: %s",
                self.name,
                self.chunks,
                dict(self.counts) or "none",
            )


def burst_len(scheduler_output: SchedulerOutput) -> int:
    """``k`` when the step is a multi-step burst, else 1.

    A burst schedules every request ``k > 1`` tokens of which ``k - 1`` are
    placeholder drafts (-1) and admits no new request; anything else carrying
    drafts without a speculative config breaks the contract and raises.
    """
    drafts = scheduler_output.scheduled_spec_decode_tokens
    if not drafts:
        return 1
    scheduled = scheduler_output.num_scheduled_tokens
    counts = set(scheduled.values())
    k = max(counts)
    if not (
        len(counts) == 1
        and k > 1
        and not scheduler_output.scheduled_new_reqs
        and drafts.keys() == scheduled.keys()
        and all(len(ids) == k - 1 and set(ids) == {-1} for ids in drafts.values())
    ):
        raise RuntimeError(
            "TPU_MULTISTEP_DECODE: a step with drafts is not a uniform burst: "
            f"scheduled {dict(list(scheduled.items())[:8])}, drafts "
            f"{ {r: len(d) for r, d in list(drafts.items())[:8]} }, "
            f"{len(scheduler_output.scheduled_new_reqs)} new requests"
        )
    return k


def decode_view(scheduler_output: SchedulerOutput) -> SchedulerOutput:
    """A burst as the 1-token decode step its first iteration is."""
    scheduled = scheduler_output.num_scheduled_tokens
    return dataclasses.replace(
        scheduler_output,
        num_scheduled_tokens=dict.fromkeys(scheduled, 1),
        total_num_scheduled_tokens=len(scheduled),
        scheduled_spec_decode_tokens={},
    )


def _advance_decode_inputs_impl(
    tokens: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Decode layout: token row r is request r; a request is real iff its
    # seq_len is positive (padding rows stay 0 and keep their inputs).
    rows = min(tokens.shape[0], input_ids.shape[0])
    real_reqs = seq_lens > 0
    span = min(real_reqs.shape[0], input_ids.shape[0])
    real_rows = torch.zeros_like(positions, dtype=torch.bool)
    real_rows[:span] = real_reqs[:span]
    next_ids = input_ids.clone()
    next_ids[:rows] = tokens.reshape(-1)[:rows].to(input_ids.dtype)
    next_ids = torch.where(real_rows, next_ids, input_ids)
    return (
        next_ids,
        positions + real_rows.to(positions.dtype),
        seq_lens + real_reqs.to(seq_lens.dtype),
    )


def _stack_tokens_impl(*tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    block = torch.cat(tokens, dim=1)
    return block, block[:, -1].contiguous()


_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_advance_decode_inputs_compiled = _tpu_compile(_advance_decode_inputs_impl)
_stack_tokens_compiled = _tpu_compile(_stack_tokens_impl)


def advance_decode_inputs(
    tokens: torch.Tensor,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    seq_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The next iteration's (input_ids, positions, seq_lens) of a decode
    chunk: each real row's greedy token (``tokens`` ``[padded_num_reqs, 1]``)
    at one position later; padding rows unchanged."""
    fn = (
        _advance_decode_inputs_compiled
        if input_ids.device.type == "tpu"
        else _advance_decode_inputs_impl
    )
    return fn(tokens, input_ids, positions, seq_lens)


def stack_tokens(tokens: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """The iterations' ``[rows, 1]`` tokens as one ``[rows, k]`` block, and
    its last column (the next step's input token of each row)."""
    fn = (
        _stack_tokens_compiled if tokens[0].device.type == "tpu" else _stack_tokens_impl
    )
    return fn(*tokens)


def advance_metadata(
    attn_metadata: dict[str, Any], positions: torch.Tensor, seq_lens: torch.Tensor
) -> dict[str, Any]:
    """The per-layer metadata with the next iteration's positions and
    seq_lens; layers that shared one metadata object still share one."""
    advanced: dict[int, Any] = {}
    out = {}
    for name, md in attn_metadata.items():
        new = advanced.get(id(md))
        if new is None:
            new = copy.copy(md)
            new.input_positions = positions
            new.seq_lens = seq_lens
            advanced[id(md)] = new
        out[name] = new
    return out
