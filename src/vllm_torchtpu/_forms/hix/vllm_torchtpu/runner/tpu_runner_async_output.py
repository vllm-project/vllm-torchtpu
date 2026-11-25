# SPDX-License-Identifier: Apache-2.0
import threading
from dataclasses import dataclass, field
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
from vllm_torchtpu import envs
from vllm_torchtpu.core import l6_ab as _l6_ab
from vllm_torchtpu.logger import init_logger

_logger = init_logger(__name__)

# q38 L6-H2 TOKEN-LISTS-ONCE: lists-path get_output calls on the worker's output thread. Only that
# thread writes it; the main thread reads it into the L6AB counts (l6_ab is not thread-safe).
_L6_OUT_LISTS = [0]


def l6_valid_rows(raw_rows: list[list[int]], max_gen_len: int) -> list[list[int]]:
    """The accepted token rows from the cached Python rows (port of IX a46d91d20, Jordan
    Dotzel): every row is a new list, so a consumer may edit it (discard clears) while the
    cached rows stay shared. Equal to the tensor path: with one column the rows as they are,
    else each row without its INVALID_TOKEN_ID entries, in order."""
    if max_gen_len == 1:
        return [row[:] for row in raw_rows]
    invalid = INVALID_TOKEN_ID
    return [[tok for tok in row if tok != invalid] for row in raw_rows]


# Main-thread state of the lever: CHECK cadence (None = read VLLM_TPU_L6_TOKEN_LISTS_CHECK at first
# use), calls since the last check, output-thread count already noted.
_L6_TL: dict = {"check": None, "n": 0, "out_seen": 0}


def l6_main_rows(
    copy_state: "AsyncTPUCopyState", n: int, max_gen_len: int, selected: torch.Tensor
) -> list[list[int]]:
    """The main thread's accepted rows for the first `n` requests of `copy_state` from its
    shared Python rows (`selected` is the same tokens as a [n, cols] tensor, read only by
    the CHECK). Main thread only: the l6_ab counts are kept here."""
    rows = l6_valid_rows(copy_state.sampled_token_ids_list[:n], max_gen_len)
    _l6_ab.note("toklists", "main")
    out = _L6_OUT_LISTS[0]
    if out != _L6_TL["out_seen"]:
        _l6_ab.note("toklists", "out", out - _L6_TL["out_seen"])
        _L6_TL["out_seen"] = out
    check = _L6_TL["check"]
    if check is None:
        check = _L6_TL["check"] = int(envs.VLLM_TPU_L6_TOKEN_LISTS_CHECK)
    if check > 0 and _l6_ab.check_allowed():
        _L6_TL["n"] += 1
        if _L6_TL["n"] >= check:
            _L6_TL["n"] = 0
            stock, _ = tensor_valid_rows(selected)
            _l6_ab.note("toklists", "checked")
            if stock != rows:
                _l6_ab.note("toklists", "mismatch")
                _logger.error(
                    "q38 L6 toklists CHECK mismatch: lists %s != tensor path %s (tensor path used)",
                    rows[:4], stock[:4],
                )
                return stock
    return rows


def tensor_valid_rows(selected: torch.Tensor) -> tuple[list[list[int]], list[int] | None]:
    """The stock tensor path over a 2-D [rows, cols] CPU tensor: (rows, gen_lens or None)."""
    if selected.shape[-1] == 1:
        return selected.tolist(), None
    valid_mask = selected != INVALID_TOKEN_ID
    gen_lens = valid_mask.sum(dim=1).tolist()
    return [seq.tolist() for seq in selected[valid_mask].split(gen_lens)], gen_lens


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


# VLLM_TPU_PINNED_TOKEN_COPY: per (shape, dtype) a ring of preallocated pinned
# host buffers the sampled tokens are copied into, instead of a fresh pageable
# tensor whose pages the driver would have to pin and map on every step. A
# buffer is reused after _PINNED_RING steps; `sampled_token_ids_cpu` copies out
# of it at wait time, well within that window.
_PINNED_RING = 8
_pinned: dict[tuple, list] = {}


def _pinned_buffer(like: torch.Tensor) -> torch.Tensor:
    key = (tuple(like.shape), like.dtype)
    entry = _pinned.get(key)
    if entry is None:
        entry = _pinned[key] = [[], 0]
    buffers, turn = entry
    if len(buffers) < _PINNED_RING:
        buffers.append(torch.empty(like.shape, dtype=like.dtype, pin_memory=True))
        out = buffers[-1]
    else:
        out = buffers[turn % _PINNED_RING]
    entry[1] = turn + 1
    return out


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
    # Spec-decode step with logprobs requested: per-chunk logprobs, copied
    # with the tokens and filtered to the accepted rows in `get_output`.
    spec_logprobs: list[SpecLogprobsChunk] | None = None
    # Device sources of `spec_logprobs`, kept alive until the copy lands.
    _spec_logprobs_tpu: list[LogprobsTensors] | None = None
    # The chunks live in reused pinned buffers (VLLM_TPU_PINNED_TOKEN_COPY).
    pinned: bool = False
    # VLLM_TPU_SERIALIZE_COPY_WAIT: the worker's async-output thread
    # (get_output) and its main thread (the deferred previous-results wait)
    # can wait on the same state at once; the lock lets only one synchronize.
    _wait_lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )
    # q38 L6-H2 TOKEN-LISTS-ONCE (lever toklists): the issuing step's lever state, fixed on the
    # main thread when the copy is issued, so both consumers of this copy take the same path.
    l6_lists: bool = False
    _sampled_token_ids_list: list[list[int]] | None = None

    @classmethod
    def from_device_chunks(
        cls,
        chunks_tpu: list[torch.Tensor],
        chunk_real_lens: list[int],
        spec_logprobs: list[SpecLogprobsChunk] | None = None,
    ) -> "AsyncTPUCopyState":
        assert len(chunks_tpu) == len(chunk_real_lens) and len(chunks_tpu) > 0
        pinned = envs.VLLM_TPU_PINNED_TOKEN_COPY
        if pinned:
            chunks_cpu = [_pinned_buffer(t).copy_(t, non_blocking=True) for t in chunks_tpu]
        else:
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
            pinned=pinned,
            spec_logprobs=spec_logprobs or None,
            _spec_logprobs_tpu=spec_logprobs_tpu,
            l6_lists=_l6_ab.ACTIVE and _l6_ab.on("toklists"),
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
        if not envs.VLLM_TPU_SERIALIZE_COPY_WAIT:
            self._wait()
            return
        with self._wait_lock:
            if not self.completed:
                self._wait()

    def _wait(self) -> None:
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
                # Pinned buffers are reused: copy the tokens out of the ring.
                self._sampled_token_ids_cpu = (
                    trimmed[0].clone() if self.pinned else trimmed[0]
                )
            else:
                self._sampled_token_ids_cpu = torch.cat(trimmed, dim=0)
        return self._sampled_token_ids_cpu

    @property
    def sampled_token_ids_list(self) -> list[list[int]]:
        """The unpadded CPU token rows as nested Python lists, converted once and shared by
        get_output (output thread) and the main thread's rollback (IX a46d91d20, Jordan
        Dotzel). Read-only: consumers take copies (l6_valid_rows) before editing rows."""
        rows = self._sampled_token_ids_list
        if rows is None:
            t = self.sampled_token_ids_cpu
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            rows = self._sampled_token_ids_list = t.tolist()
        return rows


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
        if self.copy_state.l6_lists:
            # q38 L6-H2 TOKEN-LISTS-ONCE: filter the shared Python rows, no tensor masks.
            valid_sampled_token_ids = l6_valid_rows(
                self.copy_state.sampled_token_ids_list, max_gen_len
            )
            if max_gen_len != 1 and self.copy_state.spec_logprobs:
                self._model_runner_output.logprobs = spec_logprobs_lists(
                    self.copy_state.spec_logprobs,
                    [len(row) for row in valid_sampled_token_ids],
                )
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
            _L6_OUT_LISTS[0] += 1
        elif max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist() for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            if self.copy_state.spec_logprobs:
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


def subtract_substitute_extract(
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    num_rejected: torch.Tensor,
    seq_lens_subtract_indices: torch.Tensor,
    positions_subtract_indices: torch.Tensor,
    input_ids: torch.Tensor,
    cur_input_indices: torch.Tensor,
    pre_next_tokens_indices: torch.Tensor,
    next_tokens: torch.Tensor,
    logits_indices: torch.Tensor,
    target_logits_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """VLLM_TPU_POSTFWD_FUSED (q38 DEC-FB-2 G2, after Qwen3.5 E4c 1784e91de):
    ``subtract_num_rejected_tokens``, then the async placeholder substitution
    (``tpu_runner._substitute_placeholder_token``), then
    ``extract_draft_token_ids``, as one function for one compiled program.

    The same integer values as the separate calls. The substitution reads
    ``next_tokens`` flat (the separate path reads it through
    ``_warmed_next_tokens``, an int64 copy zero-padded to a warmed length: the
    padding is never read, since every source index is below ``numel``) and
    writes out of place (``input_ids`` is a fresh per-step upload that only
    the substitution reads). ``cur_input_indices`` / ``pre_next_tokens_indices``
    are the padded device indices ``_apply_async_token_substitution`` builds
    (-1 source = keep the slot).

    Returns ``(seq_lens, positions, input_ids, draft_token_ids)``.
    """
    seq_lens, positions = subtract_num_rejected_tokens(
        seq_lens,
        positions,
        num_rejected,
        seq_lens_subtract_indices,
        positions_subtract_indices,
    )
    mask = pre_next_tokens_indices > -1
    safe_idx = torch.clamp_min(pre_next_tokens_indices, 0)
    new_token_values = next_tokens.reshape(-1)[safe_idx].to(input_ids.dtype)
    original_values = input_ids[cur_input_indices]
    update_values = torch.where(mask, new_token_values, original_values)
    input_ids = input_ids.scatter(0, cur_input_indices, update_values)
    draft_token_ids = extract_draft_token_ids(
        input_ids, logits_indices, target_logits_indices
    )
    return seq_lens, positions, input_ids, draft_token_ids


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
