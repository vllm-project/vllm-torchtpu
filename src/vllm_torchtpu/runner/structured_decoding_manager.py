# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY

from vllm_torchtpu.tracing.annotation import TraceAnnotation

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class StructuredDecodingManager:

    def __init__(self, runner: TPUModelRunner):
        self.runner = runner
        self.vocab_size = runner.vocab_size
        num_words = cdiv(self.vocab_size, 32)
        self.grammar_bitmask_cpu = torch.zeros(
            (runner.max_num_reqs, num_words),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY)
        self.require_structured_out_cpu = torch.zeros((runner.max_num_reqs, 1),
                                                      dtype=torch.bool,
                                                      device="cpu",
                                                      pin_memory=PIN_MEMORY)
        self.device = runner.device
        # Pre-allocate directly on device to avoid repeated host-to-device transfers every step
        self.structured_decode_arange = torch.arange(0,
                                                     32,
                                                     dtype=torch.int32,
                                                     device=self.device)
        self.device_all_true_require = torch.ones((runner.max_num_reqs, 1),
                                                  dtype=torch.bool,
                                                  device=self.device)
        self.device_all_false_require = torch.zeros((runner.max_num_reqs, 1),
                                                    dtype=torch.bool,
                                                    device=self.device)
        self.device_dummy_bitmask = torch.zeros(
            (runner.max_num_reqs, num_words),
            dtype=torch.int32,
            device=self.device)
        # Only needed with speculative decoding:
        # one row per draft position of the chunk's target_logits, which is
        # padded to a num-tokens bucket (up to max_num_reqs * (1 + K) rows).
        # Prefill-only PCP MTP K1 skips _precompie_rejection_sampler(), so we
        # skip it here as well.
        if (runner.speculative_config is not None
                and not runner._pcp_mtp_k1_enabled):
            from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len
            num_spec_tokens = runner.speculative_config.num_speculative_tokens
            needed = runner.max_num_reqs * (1 + num_spec_tokens)
            paddings = runner.num_tokens_paddings
            # Use min here is because needed can exceed max_num_batched_tokens
            # (which is the largest bucket), and target_logits rows never exceed
            # the largest bucket.
            max_target_rows = _get_padded_token_len(paddings,
                                                    min(needed, paddings[-1]))
            self.target_grammar_bitmask_cpu = torch.zeros(
                (max_target_rows, num_words),
                dtype=torch.int32,
                device="cpu",
                pin_memory=PIN_MEMORY)
            self.require_structured_out_target_cpu = torch.zeros(
                (max_target_rows, 1),
                dtype=torch.bool,
                device="cpu",
                pin_memory=PIN_MEMORY)

    def prepare_structured_decoding_input(
        self,
        logits: torch.Tensor,
        grammar_output: GrammarOutput,
        cur_start_idx: int,
        cur_end_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the device inputs for `structured_decode` for one chunk.

        Bitmask rows follow the order of `structured_output_request_ids`
        (the scheduler order). Each chunk covers requests
        [cur_start_idx, cur_end_idx) of the input batch, so global batch
        indices are shifted to chunk-local rows and requests outside the
        chunk are skipped.
        """
        padded_num_reqs = logits.shape[0]
        req_id_to_index = self.runner.input_batch.req_id_to_index

        bitmask = grammar_output.grammar_bitmask
        local_indices = []
        mask_rows = []
        for mask_row, req_id in enumerate(
                grammar_output.structured_output_request_ids):
            batch_index = req_id_to_index.get(req_id)
            if (batch_index is not None
                    and cur_start_idx <= batch_index < cur_end_idx):
                local_indices.append(batch_index - cur_start_idx)
                mask_rows.append(mask_row)

        # Fast path 1: No structured requests in this chunk.
        # Bypass both requirement and bitmask host transfers completely by
        # returning pre-allocated device-resident zero tensors.
        if not local_indices:
            return (
                self.device_all_false_require[:padded_num_reqs],
                self.device_dummy_bitmask[:padded_num_reqs],
                self.structured_decode_arange,
            )

        num_local = len(local_indices)

        # Standard path: handles all structured batches (all-structured, mixed, or permuted).
        self.grammar_bitmask_cpu[:padded_num_reqs].zero_()
        self.require_structured_out_cpu[:padded_num_reqs].zero_()

        if num_local == 1:
            self.grammar_bitmask_cpu[local_indices[0]].copy_(
                torch.from_numpy(bitmask[mask_rows[0]]))
        else:
            self.grammar_bitmask_cpu[local_indices] = torch.from_numpy(
                bitmask[mask_rows])
        self.require_structured_out_cpu[local_indices] = True

        return (
            self.require_structured_out_cpu[:padded_num_reqs].to(
                logits.device),
            self.grammar_bitmask_cpu[:padded_num_reqs].to(logits.device),
            self.structured_decode_arange,
        )

    def prepare_spec_structured_decoding_input(
        self,
        target_logits: torch.Tensor | None,
        bonus_logits: torch.Tensor,
        grammar_output: GrammarOutput,
        scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
        draft_lengths_cpu: np.ndarray | None,
        cur_start_idx: int,
        cur_end_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor,
               torch.Tensor, torch.Tensor]:
        """Builds the `structured_decode` inputs for one spec decoding chunk.

        On spec decoding path, the scheduler emits `1 + s_r` consecutive
        bitmask rows for one structured request, where `s_r` is the number of drafts
        scheduled for this step.

        Returns `(require_target, target_bitmask, require_bonus,
        bonus_bitmask, arange)`; the first two are None when `target_logits`
        is None.
        """
        padded_bonus_rows = bonus_logits.shape[0]
        self.grammar_bitmask_cpu[:padded_bonus_rows].zero_()
        self.require_structured_out_cpu[:padded_bonus_rows].zero_()

        padded_target_rows = 0
        target_row_starts = None
        if target_logits is not None:
            assert draft_lengths_cpu is not None
            padded_target_rows = target_logits.shape[0]
            self.target_grammar_bitmask_cpu[:padded_target_rows].zero_()
            self.require_structured_out_target_cpu[:padded_target_rows].zero_()
            # Chunk-local start row of each request's drafts in target_logits.
            target_row_starts = np.zeros(len(draft_lengths_cpu) + 1,
                                         dtype=np.int64)
            np.cumsum(draft_lengths_cpu, out=target_row_starts[1:])

        req_id_to_index = self.runner.input_batch.req_id_to_index
        bitmask = grammar_output.grammar_bitmask
        mask_row = 0
        for req_id in grammar_output.structured_output_request_ids:
            num_drafts = len(scheduled_spec_decode_tokens.get(req_id, ()))
            batch_index = req_id_to_index.get(req_id)
            if (batch_index is None
                    or not cur_start_idx <= batch_index < cur_end_idx):
                mask_row += 1 + num_drafts
                continue
            local_index = batch_index - cur_start_idx
            if num_drafts:
                assert target_row_starts is not None, (
                    "chunk without target_logits scheduled drafts for "
                    f"request {req_id}")
                assert draft_lengths_cpu[local_index] == num_drafts
                row = int(target_row_starts[local_index])
                self.target_grammar_bitmask_cpu[row:row + num_drafts] = (
                    torch.from_numpy(bitmask[mask_row:mask_row + num_drafts]))
                self.require_structured_out_target_cpu[row:row +
                                                       num_drafts] = True
            # Bonus row.
            self.grammar_bitmask_cpu[local_index] = torch.from_numpy(
                bitmask[mask_row + num_drafts])
            self.require_structured_out_cpu[local_index] = True
            mask_row += 1 + num_drafts

        if target_logits is not None:
            require_target = self.require_structured_out_target_cpu[:padded_target_rows].to(
                bonus_logits.device)
            target_bitmask = self.target_grammar_bitmask_cpu[:
                                                             padded_target_rows].to(
                                                                 bonus_logits.
                                                                 device)
        else:
            require_target = None
            target_bitmask = None
        return (
            require_target,
            target_bitmask,
            self.require_structured_out_cpu[:padded_bonus_rows].to(
                bonus_logits.device),
            self.grammar_bitmask_cpu[:padded_bonus_rows].to(
                bonus_logits.device),
            self.structured_decode_arange,
        )

    def mask_logits(
        self,
        logits: torch.Tensor,
        grammar_output: GrammarOutput,
        cur_start_idx: int,
        cur_end_idx: int,
    ) -> torch.Tensor:
        """Applies the grammar bitmask to one non-spec chunk's logits."""
        num_reqs = cur_end_idx - cur_start_idx
        padded_num_reqs = logits.shape[0]
        with TraceAnnotation("SD:PrepareInput",
                             num_reqs=num_reqs,
                             padded_num_reqs=padded_num_reqs,
                             cur_start_idx=cur_start_idx,
                             cur_end_idx=cur_end_idx):
            require_sd, bitmask, arange = self.prepare_structured_decoding_input(
                logits, grammar_output, cur_start_idx, cur_end_idx)
        with TraceAnnotation("SD:MaskLogits",
                             num_reqs=num_reqs,
                             padded_num_reqs=padded_num_reqs,
                             cur_start_idx=cur_start_idx,
                             cur_end_idx=cur_end_idx):
            return self.structured_decode(require_sd, bitmask, logits, arange)

    def mask_spec_logits(
        self,
        target_logits: torch.Tensor | None,
        bonus_logits: torch.Tensor,
        grammar_output: GrammarOutput,
        scheduled_spec_decode_tokens: Mapping[str, Sequence[int]],
        draft_lengths_cpu: np.ndarray | None,
        cur_start_idx: int,
        cur_end_idx: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Applies the grammar bitmask to both bonus and target(if not None).

        `target_logits` passes as None for non-draft chunks (when `md` is None),
        where only the bonus applies.
        """
        num_reqs = cur_end_idx - cur_start_idx
        num_target_rows = target_logits.shape[
            0] if target_logits is not None else 0
        num_bonus_rows = bonus_logits.shape[0]
        with TraceAnnotation("SD:PrepareSpecInput",
                             num_reqs=num_reqs,
                             num_target_rows=num_target_rows,
                             num_bonus_rows=num_bonus_rows,
                             cur_start_idx=cur_start_idx,
                             cur_end_idx=cur_end_idx):
            (require_target, target_bitmask, require_bonus, bonus_bitmask,
             arange) = self.prepare_spec_structured_decoding_input(
                 target_logits, bonus_logits, grammar_output,
                 scheduled_spec_decode_tokens, draft_lengths_cpu,
                 cur_start_idx, cur_end_idx)
        with TraceAnnotation("SD:MaskSpecLogits",
                             num_reqs=num_reqs,
                             num_target_rows=num_target_rows,
                             num_bonus_rows=num_bonus_rows,
                             cur_start_idx=cur_start_idx,
                             cur_end_idx=cur_end_idx):
            if target_logits is not None:
                target_logits = self.structured_decode(require_target,
                                                       target_bitmask,
                                                       target_logits, arange)
            bonus_logits = self.structured_decode(require_bonus, bonus_bitmask,
                                                  bonus_logits, arange)
            return target_logits, bonus_logits

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return self._structured_decode(require_struct_decoding,
                                       grammar_bitmask, logits, arange)

    def _structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return torch.where(
            require_struct_decoding,
            self.apply_grammar_bitmask(logits, grammar_bitmask, arange),
            logits)

    def apply_grammar_bitmask(self, logits: torch.Tensor,
                              grammar_bitmask: torch.Tensor,
                              arange: torch.Tensor) -> torch.Tensor:
        assert logits.shape[0] == grammar_bitmask.shape[0]
        target_dim = logits.shape[-1]
        if target_dim < self.vocab_size:
            raise ValueError(
                f"TPU logits vocab dimension must be at least vocab_size ({self.vocab_size}) "
                f"(padded to hardware alignment boundary), but got logits.shape[-1]={target_dim}"
            )
        # Unpack the bitmask for the entire batch at once.
        # grammar_bitmask: (B, N) where B=num_reqs, N=cdiv(vocab_size, 32)
        # arange: (32,)
        # (B, N, 1) and (1, 1, 32) broadcast to (B, N, 32)
        unpacked_bitmask = (torch.bitwise_right_shift(
            grammar_bitmask[:, :, None], arange[None, None, :])
                            & 1) == 0
        # Reshape to (B, vocab_size) and apply to logits.
        # (B, N * 32) -> (B, vocab_size)
        unpacked_bitmask = unpacked_bitmask.reshape(logits.shape[0],
                                                    -1)[:, :self.vocab_size]
        if target_dim > self.vocab_size:
            unpacked_bitmask = torch.nn.functional.pad(
                unpacked_bitmask,
                (0, target_dim - self.vocab_size),
                value=True,
            )
        return torch.where(unpacked_bitmask, float("-inf"), logits)
