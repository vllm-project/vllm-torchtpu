# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Any

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

INVALID_TOKEN_ID = -1


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

    @classmethod
    def from_device_chunks(cls, chunks_tpu: list[torch.Tensor],
                           chunk_real_lens: list[int]) -> "AsyncTPUCopyState":
        assert len(chunks_tpu) == len(chunk_real_lens) and len(chunks_tpu) > 0
        chunks_cpu = [t.to("cpu", non_blocking=True) for t in chunks_tpu]
        copy_ready_event = torch.tpu.Event()
        copy_ready_event.record()
        return cls(
            chunks_cpu=chunks_cpu,
            chunk_real_lens=chunk_real_lens,
            copy_ready_event=copy_ready_event,
            chunks_tpu=chunks_tpu,
        )

    def wait(self) -> None:
        if self.completed:
            return
        self.copy_ready_event.synchronize()
        self.chunks_tpu = None
        self.completed = True

    @property
    def sampled_token_ids_cpu(self) -> torch.Tensor:
        """Return the unpadded concatenated CPU tensor.

        Performed lazily after `wait()`; trims each bucketed chunk to its real
        length on the host (no device recompile).
        """
        if self._sampled_token_ids_cpu is None:
            trimmed = [
                t[:n] for t, n in zip(self.chunks_cpu, self.chunk_real_lens)
            ]
            if len(trimmed) == 1:
                self._sampled_token_ids_cpu = trimmed[0]
            else:
                self._sampled_token_ids_cpu = torch.cat(trimmed, dim=0)
        return self._sampled_token_ids_cpu


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
    # Spec decode only: per-request draft count proposed this step, keyed by the
    # request_seq_lens index. _modify_prev_results uses it to size the
    # (1 + num_draft) optimistic-placeholder rollback. None for non-spec.
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
        if max_gen_len == 1:
            valid_sampled_token_ids = selected_token_ids.tolist()
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()
        else:
            valid_mask = selected_token_ids != INVALID_TOKEN_ID
            gen_lens = valid_mask.sum(dim=1).tolist()
            valid_sampled_token_ids = [
                seq.tolist()
                for seq in selected_token_ids[valid_mask].split(gen_lens)
            ]
            for i in self.discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[i].clear()

        self._model_runner_output.sampled_token_ids = valid_sampled_token_ids
        return self._model_runner_output


def assemble_spec_next_tokens(next_tokens: torch.Tensor, drafts: torch.Tensor,
                              num_reqs: int) -> torch.Tensor:
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
        num_reqs: real request count (both inputs may be bucket-padded in dim 0).
    """
    nt = next_tokens[:num_reqs]
    drafts = drafts[:num_reqs]
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

    In async spec decode the host advances ``num_computed_tokens`` optimistically, assuming *every* speculative
    draft was accepted. The real acceptance is only known on-device, so this
    subtracts the actual per-request rejection count here — on-device, before the
    verify forward.

    Args:
        seq_lens: ``[num_reqs]`` optimistic sequence lengths.
        positions: ``[total_tokens]`` (or ``[3, total_tokens]`` mrope) optimistic positions.
        num_rejected: ``[num_placeholder_reqs]`` rejection count per source index.
        seq_lens_subtract_indices: ``[num_reqs]`` index into ``num_rejected`` for
            each seq_lens slot, ``-1`` to leave it unchanged.
        positions_subtract_indices: ``[total_tokens]`` index into ``num_rejected``
            for each position slot, ``-1`` to leave it unchanged.
    """
    zero = num_rejected.new_zeros(())
    seq_subtract = torch.where(
        seq_lens_subtract_indices >= 0,
        num_rejected[seq_lens_subtract_indices.clamp(min=0)], zero)
    seq_lens = seq_lens - seq_subtract

    pos_subtract = torch.where(
        positions_subtract_indices >= 0,
        num_rejected[positions_subtract_indices.clamp(min=0)], zero)
    if positions.ndim == 2:  # mrope (out of text-MVP scope; kept for parity)
        pos_subtract = pos_subtract.unsqueeze(0)
    positions = positions - pos_subtract
    return seq_lens, positions


def extract_draft_token_ids(
        input_ids: torch.Tensor, logits_indices: torch.Tensor,
        target_logits_indices: torch.Tensor) -> torch.Tensor:
    """Gather the draft tokens the target just verified, from the (post-async-
    substitution) device ``input_ids``.
    """
    return input_ids[logits_indices][target_logits_indices + 1]


def compute_num_rejected(next_tokens: torch.Tensor,
                         num_draft: torch.Tensor) -> torch.Tensor:
    """Per-request rejected-draft count from the rejection output, on-device.

    Args:
        next_tokens: rejection output ``[>=num_reqs, K+1]`` padded with
            INVALID_TOKEN_ID.
        num_draft: ``[num_reqs]`` scheduled draft count per request.
    """
    num_reqs = num_draft.shape[0]
    num_valid = (next_tokens[:num_reqs] != INVALID_TOKEN_ID).sum(dim=1)
    return (num_draft - num_valid + 1).clamp(min=0).to(torch.int32)
