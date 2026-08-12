# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput

    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


class StructuredDecodingManager:

    def __init__(self, runner: TPUModelRunner):
        self.runner = runner
        self.vocab_size = runner.vocab_size
        # One bitmask row per request; grows to 1 + num_spec_tokens rows
        # per request once speculative decoding is supported.
        max_rows = runner.max_num_reqs
        self.grammar_bitmask_cpu = torch.zeros(
            (max_rows, cdiv(self.vocab_size, 32)),
            dtype=torch.int32,
            device="cpu",
            pin_memory=PIN_MEMORY)
        self.require_structured_out_cpu = torch.zeros((max_rows, 1),
                                                      dtype=torch.bool,
                                                      device="cpu",
                                                      pin_memory=PIN_MEMORY)
        self.structured_decode_arange = torch.arange(0,
                                                     32,
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

        # Reset only the rows this chunk will read.
        self.grammar_bitmask_cpu[:padded_num_reqs].zero_()
        self.require_structured_out_cpu[:padded_num_reqs].zero_()

        bitmask = grammar_output.grammar_bitmask
        for mask_row, req_id in enumerate(
                grammar_output.structured_output_request_ids):
            batch_index = req_id_to_index.get(req_id)
            if (batch_index is None
                    or not cur_start_idx <= batch_index < cur_end_idx):
                continue
            local_index = batch_index - cur_start_idx
            self.grammar_bitmask_cpu[local_index] = torch.from_numpy(
                bitmask[mask_row])
            # Not all requests in the batch require structured output, so
            # mark the rows that need masking.
            self.require_structured_out_cpu[local_index] = True

        return (
            self.require_structured_out_cpu[:padded_num_reqs].to(
                logits.device),
            self.grammar_bitmask_cpu[:padded_num_reqs].to(logits.device),
            self.structured_decode_arange.to(logits.device),
        )

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
        return torch.where(unpacked_bitmask, float("-inf"), logits)
