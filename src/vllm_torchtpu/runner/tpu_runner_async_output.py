# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import functools
from typing import Any

import numpy as np
import torch
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
)

INVALID_TOKEN_ID = -1


@dataclass
class SpecLogprobsChunk:
    """One chunk's logprobs in a spec-decode step, one row per logits row.

    A verify chunk's logits hold, per request, one row per draft token and
    then its bonus row, so request `i`'s verify slots are rows
    `row_starts[i]`, `row_starts[i] + 1`, ... in slot order, and its accepted
    tokens are a prefix of them. A chunk without drafts has one row per
    request (`row_starts[i] == i`). Rows are computed for every slot, accepted
    or not, and filtered on the host once the accepted counts are known.
    """

    # Device tensors until the async copy is issued, CPU tensors after.
    tensors: LogprobsTensors
    # [num_reqs] first logprobs row of each real request in the chunk.
    row_starts: np.ndarray


def spec_logprobs_lists(
    chunks: list[SpecLogprobsChunk], gen_lens: list[int]
) -> LogprobsLists:
    """Keep each request's accepted rows, in request order.

    `gen_lens[i]` is request `i`'s number of generated tokens this step,
    counted across all chunks before any discard, as vLLM's rejection
    sampler does.
    """
    ids, lps, ranks = [], [], []
    offset = 0
    for chunk in chunks:
        n = len(chunk.row_starts)
        take = [
            np.arange(start, start + g, dtype=np.int64)
            for start, g in zip(chunk.row_starts, gen_lens[offset : offset + n])
        ]
        rows = np.concatenate(take) if take else np.zeros(0, dtype=np.int64)
        ids.append(chunk.tensors.logprob_token_ids.numpy()[rows])
        lps.append(chunk.tensors.logprobs.numpy()[rows])
        ranks.append(chunk.tensors.selected_token_ranks.numpy()[rows])
        offset += n
    assert offset == len(gen_lens), (offset, len(gen_lens))
    return LogprobsLists(
        logprob_token_ids=np.concatenate(ids),
        logprobs=np.concatenate(lps),
        sampled_token_ranks=np.concatenate(ranks),
        cu_num_generated_tokens=np.concatenate(([0], np.cumsum(gen_lens))).tolist(),
    )


@dataclass
class AsyncTPUCopyState:
    """Async D2H copy state for sampled tokens.

    Per-chunk device tensors are kept at their bucketed (padded) shapes so
    each D2H copy hits a precompiled program. Host-side trim and concat
    happen lazily on `wait()` and produce a single contiguous CPU tensor
    matching the real (unpadded) request count.
    """

    chunks_cpu: list[torch.Tensor]
    chunk_real_lens: list[int]
    copy_ready_event: Any
    chunks_tpu: list[torch.Tensor] | None = None
    completed: bool = False
    _sampled_token_ids_cpu: torch.Tensor | None = None
    _sampled_token_ids_list: list[list[int]] | None = None
    # Spec-decode step with logprobs requested: per-chunk logprobs, copied
    # with the tokens and filtered to the accepted rows in `get_output`.
    spec_logprobs: list[SpecLogprobsChunk] | None = None
    # Device sources of `spec_logprobs`, kept alive until the copy lands.
    _spec_logprobs_tpu: list[LogprobsTensors] | None = None

    @classmethod
    def from_device_chunks(
        cls,
        chunks_tpu: list[torch.Tensor],
        chunk_real_lens: list[int],
        spec_logprobs: list[SpecLogprobsChunk] | None = None,
    ) -> "AsyncTPUCopyState":
        assert len(chunks_tpu) == len(chunk_real_lens) and len(chunks_tpu) > 0
        chunks_cpu = [t.to("cpu", non_blocking=True) for t in chunks_tpu]
        spec_logprobs_tpu = None
        if spec_logprobs:
            spec_logprobs_tpu = [c.tensors for c in spec_logprobs]
            spec_logprobs = [
                SpecLogprobsChunk(c.tensors.to_cpu_nonblocking(), c.row_starts)
                for c in spec_logprobs
            ]
        copy_ready_event = torch.tpu.Event()
        copy_ready_event.record()
        return cls(
            chunks_cpu=chunks_cpu,
            chunk_real_lens=chunk_real_lens,
            copy_ready_event=copy_ready_event,
            chunks_tpu=chunks_tpu,
            spec_logprobs=spec_logprobs or None,
            _spec_logprobs_tpu=spec_logprobs_tpu,
        )

    def fence_pending_device_work(self) -> None:
        """Re-arms the ready event behind everything enqueued on the device
        since the copy was issued. A torch.tpu.Event covers all work
        enqueued before it is recorded, so wait() then also proves that
        later work (a drafter's KV writes) has executed."""
        event = torch.tpu.Event()
        event.record()
        self.copy_ready_event = event

    def wait(self) -> None:
        if self.completed:
            return
        self.copy_ready_event.synchronize()
        self.chunks_tpu = None
        self._spec_logprobs_tpu = None
        self.completed = True

    @property
    def sampled_token_ids_cpu(self) -> torch.Tensor:
        """Return the unpadded concatenated CPU tensor.

        Performed lazily after `wait()`; trims each bucketed chunk to its real
        length on the host (no device recompile).
        """
        if self._sampled_token_ids_cpu is None:
            trimmed = [t[:n] for t, n in zip(self.chunks_cpu, self.chunk_real_lens)]
            if len(trimmed) == 1:
                self._sampled_token_ids_cpu = trimmed[0]
            else:
                self._sampled_token_ids_cpu = torch.cat(trimmed, dim=0)
        return self._sampled_token_ids_cpu

    @property
    def sampled_token_ids_list(self) -> list[list[int]]:
        """Return the unpadded CPU token rows as nested Python lists (cached)."""
        if self._sampled_token_ids_list is None:
            t = self.sampled_token_ids_cpu
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            self._sampled_token_ids_list = t.tolist()
        return self._sampled_token_ids_list


@dataclass
class AsyncPreResults:
    req_ids: list[str]
    next_tokens_tpu: torch.Tensor
    request_seq_lens: list[tuple[int, Any, int, str]]
    discard_sampled_tokens_req_indices: list[int]
    req_id_to_index_copy: dict[str, int]
    copy_state: AsyncTPUCopyState | None = None
    # Spec decode only: per-request rejected-draft count from this step, kept
    # on-device so the next step can correct its optimistically-advanced
    # positions/seq_lens. None for non-spec.
    spec_decode_num_rejected_tokens: torch.Tensor | None = None
    # Per-request draft count VERIFIED this step, keyed by request_seq_lens
    # index. Sizes the (1 + num_draft) output placeholders and their rollback;
    # it does not count proposals for the next step. Bootstrap/non-spec: 0 or
    # absent entry (None for an entirely non-spec result).
    num_draft_per_req: dict[int, int] | None = None

    def wait_for_copy(self) -> torch.Tensor | None:
        if self.copy_state is None:
            return None
        self.copy_state.wait()
        return self.copy_state.sampled_token_ids_cpu


class AsyncTPUModelRunnerOutput(AsyncModelRunnerOutput):
    """Holds asynchronous model output specifically from a TPU runner."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        copy_state: AsyncTPUCopyState | None,
        discard_sampled_tokens_req_indices: list[int],
    ):
        self._model_runner_output: ModelRunnerOutput = model_runner_output
        self.copy_state = copy_state
        self.discard_sampled_tokens_req_indices = discard_sampled_tokens_req_indices

    def get_output(self) -> ModelRunnerOutput:
        if self.copy_state is None:
            self._model_runner_output.sampled_token_ids = []
            return self._model_runner_output

        self.copy_state.wait()
        selected_token_ids = self.copy_state.sampled_token_ids_cpu
        if selected_token_ids.numel() == 0:
            self._model_runner_output.sampled_token_ids = []
            return self._model_runner_output
        if selected_token_ids.dim() == 1:
            selected_token_ids = selected_token_ids.unsqueeze(-1)

        max_gen_len = selected_token_ids.shape[-1]
        raw_rows = self.copy_state.sampled_token_ids_list
        if max_gen_len == 1:
            valid_sampled_token_ids = [row[:] for row in raw_rows]
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            invalid = INVALID_TOKEN_ID
            valid_sampled_token_ids = [
                [tok for tok in row if tok != invalid] for row in raw_rows
            ]
            if self.copy_state.spec_logprobs:
                gen_lens = [len(row) for row in valid_sampled_token_ids]
                self._model_runner_output.logprobs = spec_logprobs_lists(
                    self.copy_state.spec_logprobs, gen_lens
                )
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids
        return self._model_runner_output


def assemble_spec_next_tokens(
    next_tokens: torch.Tensor, drafts: torch.Tensor, num_reqs: int | None = None
) -> torch.Tensor:
    """Async substitution source for speculative decoding.

    Per request the next verify step consumes ``[bonus, draft_1, ..., draft_K]``:
    the bonus is this step's last verified token (the continuation), the drafts
    are this step's proposal. Returns them packed row-major and flattened to
    ``[num_reqs * (1 + K)]`` — the layout the substitution index builder
    addresses as ``req_idx * (1 + K) + j`` (bonus at ``j == 0``).

    Args:
        next_tokens: rejection-sampler output ``[>=num_reqs, K+1]`` (accepted
            prefix + bonus, padded with INVALID_TOKEN_ID); the bonus is the last
            non-padding column of each row.
        drafts: device proposer output ``[>=num_reqs, K]``.
        num_reqs: slice both inputs to this many rows, or None to keep the
            bucketed rows (the runner's choice: the padded tail is never
            addressed, and an exact slice would compile a program per count).
    """
    if num_reqs is not None:
        next_tokens = next_tokens[:num_reqs]
        drafts = drafts[:num_reqs]
    nt = next_tokens
    # Last non-padding column per row is the bonus.
    last_valid_col = (nt != INVALID_TOKEN_ID).sum(dim=1).clamp(min=1) - 1
    bonus = nt.gather(1, last_valid_col.unsqueeze(1))  # [num_reqs, 1]
    src = torch.cat([bonus.to(drafts.dtype), drafts], dim=1)  # [num_reqs, 1+K]
    return src.reshape(-1)  # [num_reqs * (1 + K)]


def subtract_num_rejected_tokens(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove the previous async step's per-request rejected-token over-count.

    In async spec decode the host advances ``num_computed_tokens`` optimistically,
    assuming *every* speculative draft was accepted. The real acceptance is only known
    on-device, so this subtracts the actual per-request rejection count here —
    on-device, before the verify forward.

    Args:
        seq_lens: ``[num_reqs]`` optimistic sequence lengths.
        positions: ``[total_tokens]`` (or ``[3, total_tokens]`` mrope)
            optimistic positions.
        num_rejected: ``[num_placeholder_reqs]`` rejection count per source index.
        seq_lens_subtract_indices: ``[num_reqs]`` index into ``num_rejected`` for
            each seq_lens slot, ``-1`` to leave it unchanged.
        positions_subtract_indices: ``[total_tokens]`` index into ``num_rejected``
            for each position slot, ``-1`` to leave it unchanged.
    """
    zero = num_rejected.new_zeros(())
    seq_subtract = torch.where(
        seq_lens_subtract_indices >= 0,
        num_rejected[seq_lens_subtract_indices.clamp(min=0)],
        zero,
    )
    seq_lens = seq_lens - seq_subtract

    pos_subtract = torch.where(
        positions_subtract_indices >= 0,
        num_rejected[positions_subtract_indices.clamp(min=0)],
        zero,
    )
    if positions.ndim == 2:  # mrope (out of text-MVP scope; kept for parity)
        pos_subtract = pos_subtract.unsqueeze(0)
    positions = positions - pos_subtract
    return seq_lens, positions


def extract_draft_token_ids(
    input_ids: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather the draft tokens the target just verified, from the (post-async-
    substitution) device ``input_ids``.
    """
    return input_ids[logits_indices[target_logits_indices + 1]]


def subtract_and_substitute_placeholder_token(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rejected-draft subtraction and placeholder token substitution."""
    seq_lens, positions = subtract_num_rejected_tokens(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
    )
    token_in_tpu_cur_input_indices = substitution_indices[0]
    token_in_tpu_pre_next_tokens_indices = substitution_indices[1]
    mask = token_in_tpu_pre_next_tokens_indices > -1
    safe_idx = torch.clamp_min(token_in_tpu_pre_next_tokens_indices, 0)
    new_token_values = next_tokens_tpu[safe_idx].to(input_ids.dtype)
    original_values = input_ids[token_in_tpu_cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids = input_ids.scatter(0, token_in_tpu_cur_input_indices, update_values)
    return seq_lens, positions, input_ids


def subtract_and_substitute_draft_token_ids(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rejected-draft subtraction, placeholder token substitution, and
    draft token extraction in a single compiled TPU graph.
    """
    seq_lens, positions, input_ids = subtract_and_substitute_placeholder_token(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
        input_ids,
        substitution_indices,
        next_tokens_tpu,
    )
    draft_token_ids = extract_draft_token_ids(
        input_ids, logits_indices, target_logits_indices
    )
    return seq_lens, positions, input_ids, draft_token_ids


def carry_subtract_and_substitute_draft_token_ids(
    seq_lens_prev: torch.Tensor,
    positions_prev: torch.Tensor,
    step_delta: int,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids_prev: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Steady-state form of `subtract_and_substitute_draft_token_ids` that needs
    no host upload of seq_lens / positions / input_ids.

    Inputs are the previous step's corrected device tensors. In steady async
    spec decode (same requests, same 1+K tokens each, the host's optimistic
    counts advanced by exactly `step_delta` minus the previous step's rejected
    count, which the runner checks on the host before taking this path), the
    host's next optimistic values equal `prev + step_delta` on every slot the
    subtraction touches (index >= 0); the other slots (padding) keep their
    values, which the host also left unchanged. Every real input slot is
    substituted from `next_tokens_tpu`, so `input_ids_prev` only supplies the
    (zero) padding. The result is bit-identical to uploading the host values
    and calling `subtract_and_substitute_draft_token_ids`.
    """
    zero_s = seq_lens_prev.new_zeros(())
    seq_lens = seq_lens_prev + torch.where(
        seq_lens_subtract_indices >= 0, zero_s + step_delta, zero_s
    )
    zero_p = positions_prev.new_zeros(())
    pos_add = torch.where(positions_subtract_indices >= 0, zero_p + step_delta, zero_p)
    if positions_prev.ndim == 2:
        pos_add = pos_add.unsqueeze(0)
    positions = positions_prev + pos_add
    return subtract_and_substitute_draft_token_ids(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
        input_ids_prev,
        substitution_indices,
        next_tokens_tpu,
        logits_indices,
        target_logits_indices,
    )


def compute_num_rejected(
    next_tokens: torch.Tensor, num_draft: torch.Tensor
) -> torch.Tensor:
    """Per-request rejected-draft count from the rejection output, on-device.

    Args:
        next_tokens: rejection output ``[>=num_reqs, K+1]`` padded with
            INVALID_TOKEN_ID.
        num_draft: ``[num_reqs]`` scheduled draft count per request.
    """
    num_reqs = num_draft.shape[0]
    num_valid = (next_tokens[:num_reqs] != INVALID_TOKEN_ID).sum(dim=1)
    return (num_draft - num_valid + 1).clamp(min=0).to(torch.int32)


def _static_shape_args(fn):
    """Strip dynamo dynamic-dim marks from a compiled region's tensor args."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        def unmark(a):
            if isinstance(a, torch.Tensor) and getattr(
                a, "_dynamo_dynamic_indices", None
            ):
                return a[:]
            return a

        return fn(
            *[unmark(a) for a in args], **{k: unmark(v) for k, v in kwargs.items()}
        )

    return wrapper


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_subtract_num_rejected_tokens(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return subtract_num_rejected_tokens(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
    )


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_extract_draft_token_ids(
    input_ids: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> torch.Tensor:
    return extract_draft_token_ids(
        input_ids, logits_indices, target_logits_indices
    )


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_subtract_and_substitute_placeholder_token(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return subtract_and_substitute_placeholder_token(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
        input_ids,
        substitution_indices,
        next_tokens_tpu,
    )


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_carry_subtract_and_substitute_draft_token_ids(
    seq_lens_prev: torch.Tensor,
    positions_prev: torch.Tensor,
    step_delta: int,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids_prev: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return carry_subtract_and_substitute_draft_token_ids(
        seq_lens_prev,
        positions_prev,
        step_delta,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
        input_ids_prev,
        substitution_indices,
        next_tokens_tpu,
        logits_indices,
        target_logits_indices,
    )


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_subtract_and_substitute_draft_token_ids(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids: torch.Tensor,
    substitution_indices: torch.Tensor,
    next_tokens_tpu: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return subtract_and_substitute_draft_token_ids(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
        input_ids,
        substitution_indices,
        next_tokens_tpu,
        logits_indices,
        target_logits_indices,
    )



@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_assemble_spec_and_rejected(
    next_tokens: torch.Tensor,
    drafts: torch.Tensor,
    draft_lengths: torch.Tensor,
    bound: int,
    max_num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused bucketed on-device substitution source assembly + num_rejected.

    Operates on bucketed ``next_tokens`` ``[B, K+1]``, ``drafts`` ``[D, K]``
    (with ``D >= B``), and ``draft_lengths`` ``[B]`` (padded tail rows are 0),
    returning ``(src_padded [bound], num_rejected_padded [max_num_reqs])`` in a
    single compiled graph.
    """
    B = next_tokens.shape[0]
    drafts_b = (
        torch.nn.functional.pad(drafts, (0, 0, 0, B - drafts.shape[0]))
        if drafts.shape[0] < B
        else drafts[:B]
    )
    num_valid = (next_tokens != INVALID_TOKEN_ID).sum(dim=1)
    last_valid_col = num_valid.clamp(min=1) - 1
    bonus = next_tokens.gather(1, last_valid_col.unsqueeze(1))
    src = torch.cat([bonus.to(drafts_b.dtype), drafts_b], dim=1).reshape(-1)
    if src.shape[0] < bound:
        src = torch.nn.functional.pad(src, (0, bound - src.shape[0]))
    num_rejected = (draft_lengths - num_valid + 1).clamp(min=0).to(torch.int32)
    if B < max_num_reqs:
        num_rejected = torch.nn.functional.pad(num_rejected, (0, max_num_reqs - B))
    return src, num_rejected


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_pad_non_spec_next_tokens(
    selected: torch.Tensor,
    k: int,
) -> torch.Tensor:
    col0 = selected.reshape(-1, 1).to(torch.int32)
    return torch.nn.functional.pad(col0, (0, k), value=INVALID_TOKEN_ID)


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_assemble_two_chunks_spec_and_rejected(
    nt0: torch.Tensor,
    drafts0: torch.Tensor,
    dl0: torch.Tensor,
    nt1: torch.Tensor,
    drafts1: torch.Tensor,
    dl1: torch.Tensor,
    bound: int,
    max_num_reqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused 2-chunk bucketed substitution source assembly + num_rejected.

    Packs chunk 0 at request indices ``[0 .. B0 - 1]`` and chunk 1 at bucketed
    offset ``[B0 .. B0 + B1 - 1]`` so neither chunk slices by unbucketed
    ``num_reqs``. Requires ``B0 + B1 <= max_num_reqs``.
    """
    B0 = nt0.shape[0]
    d0 = (
        torch.nn.functional.pad(drafts0, (0, 0, 0, B0 - drafts0.shape[0]))
        if drafts0.shape[0] < B0
        else drafts0[:B0]
    )
    nv0 = (nt0 != INVALID_TOKEN_ID).sum(dim=1)
    lvc0 = nv0.clamp(min=1) - 1
    bonus0 = nt0.gather(1, lvc0.unsqueeze(1))
    src0 = torch.cat([bonus0.to(d0.dtype), d0], dim=1).reshape(-1)
    rej0 = (dl0 - nv0 + 1).clamp(min=0).to(torch.int32)

    B1 = nt1.shape[0]
    d1 = (
        torch.nn.functional.pad(drafts1, (0, 0, 0, B1 - drafts1.shape[0]))
        if drafts1.shape[0] < B1
        else drafts1[:B1]
    )
    nv1 = (nt1 != INVALID_TOKEN_ID).sum(dim=1)
    lvc1 = nv1.clamp(min=1) - 1
    bonus1 = nt1.gather(1, lvc1.unsqueeze(1))
    src1 = torch.cat([bonus1.to(d1.dtype), d1], dim=1).reshape(-1)
    rej1 = (dl1 - nv1 + 1).clamp(min=0).to(torch.int32)

    src = torch.cat([src0, src1], dim=0)
    if src.shape[0] < bound:
        src = torch.nn.functional.pad(src, (0, bound - src.shape[0]))
    rej = torch.cat([rej0, rej1], dim=0)
    if rej.shape[0] < max_num_reqs:
        rej = torch.nn.functional.pad(rej, (0, max_num_reqs - rej.shape[0]))
    return src, rej


@_static_shape_args
@torch.compile(backend="tpu", fullgraph=True, dynamic=False)
def _compiled_assemble_prefill_bootstrap(
    selected_tokens: torch.Tensor,
    padded_drafts: torch.Tensor,
    bound: int,
) -> torch.Tensor:
    """Bucketed on-device substitution source assembly for async prefill bootstrap.

    Packs ``selected_tokens`` ``[B, 1]`` and ``padded_drafts`` ``[D, K]`` (with
    ``D >= B``) into ``src_padded [bound]`` in a single compiled graph.
    """
    B = selected_tokens.shape[0]
    drafts_b = (
        torch.nn.functional.pad(padded_drafts, (0, 0, 0, B - padded_drafts.shape[0]))
        if padded_drafts.shape[0] < B
        else padded_drafts[:B]
    )
    bonus = selected_tokens.reshape(B, 1).to(drafts_b.dtype)
    src = torch.cat([bonus, drafts_b], dim=1).reshape(-1)
    if src.shape[0] < bound:
        src = torch.nn.functional.pad(src, (0, bound - src.shape[0]))
    return src

