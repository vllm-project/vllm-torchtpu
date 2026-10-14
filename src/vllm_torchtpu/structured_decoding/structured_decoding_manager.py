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
            pin_memory=PIN_MEMORY,
        )
        self.require_structured_out_cpu = torch.zeros(
            (runner.max_num_reqs, 1),
            dtype=torch.bool,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        self.grammar_bitmask_np = self.grammar_bitmask_cpu.numpy()
        self.require_structured_out_np = self.require_structured_out_cpu.numpy()
        self.device = runner.device
        # Pre-allocate directly on device to avoid repeated host-to-device transfers
        # every step
        self.structured_decode_arange = torch.arange(
            0, 32, dtype=torch.int32, device=self.device
        )
        self.device_all_true_require = torch.ones(
            (runner.max_num_reqs, 1), dtype=torch.bool, device=self.device
        )
        self.device_all_false_require = torch.zeros(
            (runner.max_num_reqs, 1), dtype=torch.bool, device=self.device
        )
        self.device_dummy_bitmask = torch.zeros(
            (runner.max_num_reqs, num_words), dtype=torch.int32, device=self.device
        )
        self.max_num_spec_tokens = (
            runner.speculative_config.num_speculative_tokens
            if runner.speculative_config is not None
            else 0
        )
        # Only needed with speculative decoding:
        # one row per draft position of the chunk's target_logits, which is
        # padded to a num-tokens bucket (up to max_num_reqs * (1 + K) rows).
        # Prefill-only MTP skips _precompile_rejection_sampler(), so we
        # skip it here as well.
        if runner.speculative_config is not None and not runner._mtp_prefill_only:
            from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

            num_spec_tokens = runner.speculative_config.num_speculative_tokens
            needed = runner.max_num_reqs * (1 + num_spec_tokens)
            paddings = runner.num_tokens_paddings
            # Use min here is because needed can exceed max_num_batched_tokens
            # (which is the largest bucket), and target_logits rows never exceed
            # the largest bucket.
            max_target_rows = _get_padded_token_len(paddings, min(needed, paddings[-1]))
            max_device_rows = max(max_device_rows, max_target_rows)
            self.target_grammar_bitmask_cpu = torch.zeros(
                (max_target_rows, num_words),
                dtype=torch.int32,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            self.require_structured_out_target_cpu = torch.zeros(
                (max_target_rows, 1),
                dtype=torch.bool,
                device="cpu",
                pin_memory=PIN_MEMORY,
            )
            self.target_grammar_bitmask_np = self.target_grammar_bitmask_cpu.numpy()
            self.require_structured_out_target_np = (
                self.require_structured_out_target_cpu.numpy()
            )
        self.device_all_true_require = torch.ones(
            (max_device_rows, 1), dtype=torch.bool, device=self.device
        )
        self.device_all_false_require = torch.zeros(
            (max_device_rows, 1), dtype=torch.bool, device=self.device
        )
        self.device_dummy_bitmask = torch.zeros(
            (max_device_rows, num_words), dtype=torch.int32, device=self.device
        )

    def _mask_stride(self, bitmask: np.ndarray, req_ids: Sequence[str]) -> int:
        """Rows per structured request in the scheduler's grammar bitmask."""
        mask_stride = 1 + self.max_num_spec_tokens
        assert bitmask.shape[0] == len(req_ids) * mask_stride, (
            f"grammar bitmask has {bitmask.shape[0]} rows for {len(req_ids)} "
            f"requests, expected {mask_stride} per request"
        )
        return mask_stride

    def prepare_structured_decoding_input(
        self,
        logits: torch.Tensor,
        grammar_output: GrammarOutput,
        cur_start_idx: int,
        cur_end_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds the device inputs for `structured_decode` for one chunk.

        Bitmask rows follow the order of `structured_output_request_ids`
        (the scheduler order), with `mask_stride = 1 + max_num_spec_tokens`
        rows allocated per request when speculative decoding is enabled.
        Each chunk covers requests [cur_start_idx, cur_end_idx) of the input
        batch, so global batch indices are shifted to chunk-local rows and
        requests outside the chunk are skipped.
        """
        padded_num_reqs = logits.shape[0]
        req_id_to_index = self.runner.input_batch.req_id_to_index

        bitmask = grammar_output.grammar_bitmask
        req_ids = grammar_output.structured_output_request_ids
        mask_stride = self._mask_stride(bitmask, req_ids)
        local_indices = []
        mask_rows = []
        for grammar_idx, req_id in enumerate(req_ids):
            batch_index = req_id_to_index.get(req_id)
            if batch_index is not None and cur_start_idx <= batch_index < cur_end_idx:
                local_indices.append(batch_index - cur_start_idx)
                mask_rows.append(grammar_idx * mask_stride)

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
        all_rows_structured = num_local == padded_num_reqs

        # Standard path: handles all structured batches (all-structured,
        # mixed, or permuted).
        if not all_rows_structured:
            self.grammar_bitmask_np[:padded_num_reqs].fill(0)
            self.require_structured_out_np[:padded_num_reqs].fill(False)
            if is_contiguous:
                l0 = local_indices[0]
                self.require_structured_out_np[l0 : l0 + num_local] = True
            else:
                self.require_structured_out_np[local_indices] = True

        if is_contiguous:
            l0 = local_indices[0]
            m0 = mask_rows[0]
            self.grammar_bitmask_np[l0 : l0 + num_local] = bitmask[m0 : m0 + num_local]
        else:
            self.grammar_bitmask_np[local_indices] = bitmask[mask_rows]

        require_sd = (
            self.device_all_true_require[:padded_num_reqs]
            if all_rows_structured
            else self.require_structured_out_cpu[:padded_num_reqs].to(logits.device)
        )
        return (
            require_sd,
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
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Builds the `structured_decode` inputs for one spec decoding chunk.

        On spec decoding path, the scheduler allocates a fixed block of
        `mask_stride = 1 + max_num_spec_tokens` rows per structured request,
        where the first `s_r` rows cover the `s_r` drafts scheduled for this
        step and row `s_r` covers the bonus token.

        Returns `(require_target, target_bitmask, require_bonus,
        bonus_bitmask, arange)`; the first two are None when `target_logits`
        is None.
        """
        padded_bonus_rows = bonus_logits.shape[0]
        padded_target_rows = target_logits.shape[0] if target_logits is not None else 0

        req_id_to_index = self.runner.input_batch.req_id_to_index
        bitmask = grammar_output.grammar_bitmask
<<<<<<< HEAD
        req_ids = grammar_output.structured_output_request_ids
        mask_stride = self._mask_stride(bitmask, req_ids)
        for grammar_idx, req_id in enumerate(req_ids):
            mask_row = grammar_idx * mask_stride
            num_drafts = len(scheduled_spec_decode_tokens.get(req_id, ()))
            batch_index = req_id_to_index.get(req_id)
            if batch_index is None or not cur_start_idx <= batch_index < cur_end_idx:
                continue
            local_index = batch_index - cur_start_idx
            if num_drafts:
                assert target_row_starts is not None, (
                    f"chunk without target_logits scheduled drafts for request {req_id}"
                )
                assert draft_lengths_cpu is not None
                if draft_lengths_cpu[local_index] != num_drafts:
                    raise ValueError(
                        f"Draft lengths mismatch for request {req_id}: "
                        f"scheduler scheduled {num_drafts} drafts, but runner recorded "
                        f"{draft_lengths_cpu[local_index]} drafts."
                    )
                row = int(target_row_starts[local_index])
                self.target_grammar_bitmask_np[row : row + num_drafts] = bitmask[
                    mask_row : mask_row + num_drafts
                ]
                self.require_structured_out_target_np[row : row + num_drafts] = True
            # Bonus row.
            self.grammar_bitmask_np[local_index] = bitmask[mask_row + num_drafts]
            self.require_structured_out_np[local_index] = True
=======
        local_indices: list[int] = []
        mask_rows: list[int] = []
        draft_counts: list[int] = []
        is_contiguous = True
        uniform_drafts = True
        total_structured_drafts = 0
        mask_row = 0

        for req_id in grammar_output.structured_output_request_ids:
            num_drafts = len(scheduled_spec_decode_tokens.get(req_id, ()))
            batch_index = req_id_to_index.get(req_id)
            if batch_index is not None and cur_start_idx <= batch_index < cur_end_idx:
                local_idx = batch_index - cur_start_idx
                if num_drafts:
                    assert target_logits is not None, (
                        "chunk without target_logits scheduled drafts "
                        f"for request {req_id}"
                    )
                    assert draft_lengths_cpu is not None
                    if draft_lengths_cpu[local_idx] != num_drafts:
                        raise ValueError(
                            f"Draft lengths mismatch for request {req_id}: "
                            f"scheduler scheduled {num_drafts} drafts, "
                            f"but runner recorded {draft_lengths_cpu[local_idx]} "
                            "drafts."
                        )
                if local_indices:
                    if (
                        local_idx != local_indices[-1] + 1
                        or mask_row != mask_rows[-1] + 1 + draft_counts[-1]
                    ):
                        is_contiguous = False
                    if num_drafts != draft_counts[0]:
                        uniform_drafts = False
                local_indices.append(local_idx)
                mask_rows.append(mask_row)
                draft_counts.append(num_drafts)
                total_structured_drafts += num_drafts
            mask_row += 1 + num_drafts
>>>>>>> 20e2b6b ([Structured Decoding] Add fast paths and vectorized 3D copy for speculative decoding)

        # Fast path 1: No structured requests in this chunk.
        if not local_indices:
            if target_logits is not None:
                require_target: torch.Tensor | None = self.device_all_false_require[
                    :padded_target_rows
                ]
                target_bitmask: torch.Tensor | None = self.device_dummy_bitmask[
                    :padded_target_rows
                ]
            else:
                require_target = None
                target_bitmask = None
            return (
                require_target,
                target_bitmask,
                self.device_all_false_require[:padded_bonus_rows],
                self.device_dummy_bitmask[:padded_bonus_rows],
                self.structured_decode_arange,
            )

        num_local = len(local_indices)
        all_bonus_structured = num_local == padded_bonus_rows
        all_target_structured = (
            target_logits is not None and total_structured_drafts == padded_target_rows
        )

        if not all_bonus_structured:
            self.grammar_bitmask_np[:padded_bonus_rows].fill(0)
            self.require_structured_out_np[:padded_bonus_rows].fill(False)

        if (
            target_logits is not None
            and total_structured_drafts > 0
            and not all_target_structured
        ):
            self.target_grammar_bitmask_np[:padded_target_rows].fill(0)
            self.require_structured_out_target_np[:padded_target_rows].fill(False)

        if is_contiguous and uniform_drafts:
            l0 = local_indices[0]
            m0 = mask_rows[0]
            k = draft_counts[0]
            if k == 0:
                self.grammar_bitmask_np[l0 : l0 + num_local] = bitmask[
                    m0 : m0 + num_local
                ]
            else:
                assert draft_lengths_cpu is not None
                r0 = 0 if l0 == 0 else int(draft_lengths_cpu[:l0].sum())
                stride = 1 + k
                chunk_view = bitmask[m0 : m0 + num_local * stride].reshape(
                    num_local, stride, -1
                )
                self.target_grammar_bitmask_np[r0 : r0 + total_structured_drafts] = (
                    chunk_view[:, :k, :].reshape(total_structured_drafts, -1)
                )
                self.grammar_bitmask_np[l0 : l0 + num_local] = chunk_view[:, k, :]
                if not all_target_structured:
                    self.require_structured_out_target_np[
                        r0 : r0 + total_structured_drafts
                    ] = True
            if not all_bonus_structured:
                self.require_structured_out_np[l0 : l0 + num_local] = True
        else:
            target_row_starts = None
            if target_logits is not None and total_structured_drafts > 0:
                assert draft_lengths_cpu is not None
                target_row_starts = np.zeros(len(draft_lengths_cpu) + 1, dtype=np.int64)
                np.cumsum(draft_lengths_cpu, out=target_row_starts[1:])

            for local_index, mask_row_i, num_drafts in zip(
                local_indices, mask_rows, draft_counts
            ):
                if num_drafts:
                    assert target_row_starts is not None
                    row = int(target_row_starts[local_index])
                    self.target_grammar_bitmask_np[row : row + num_drafts] = bitmask[
                        mask_row_i : mask_row_i + num_drafts
                    ]
                    if not all_target_structured:
                        self.require_structured_out_target_np[
                            row : row + num_drafts
                        ] = True
                self.grammar_bitmask_np[local_index] = bitmask[mask_row_i + num_drafts]
                if not all_bonus_structured:
                    self.require_structured_out_np[local_index] = True

        if target_logits is not None:
            if total_structured_drafts == 0:
                require_target = self.device_all_false_require[:padded_target_rows]
                target_bitmask = self.device_dummy_bitmask[:padded_target_rows]
            else:
                require_target = (
                    self.device_all_true_require[:padded_target_rows]
                    if all_target_structured
                    else self.require_structured_out_target_cpu[:padded_target_rows].to(
                        bonus_logits.device
                    )
                )
                target_bitmask = self.target_grammar_bitmask_cpu[
                    :padded_target_rows
                ].to(bonus_logits.device)
        else:
            require_target = None
            target_bitmask = None

        require_bonus = (
            self.device_all_true_require[:padded_bonus_rows]
            if all_bonus_structured
            else self.require_structured_out_cpu[:padded_bonus_rows].to(
                bonus_logits.device
            )
        )
        return (
            require_target,
            target_bitmask,
            require_bonus,
            self.grammar_bitmask_cpu[:padded_bonus_rows].to(bonus_logits.device),
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
        with TraceAnnotation(
            "SD:PrepareInput",
            num_reqs=num_reqs,
            padded_num_reqs=padded_num_reqs,
            cur_start_idx=cur_start_idx,
            cur_end_idx=cur_end_idx,
        ):
            require_sd, bitmask, arange = self.prepare_structured_decoding_input(
                logits, grammar_output, cur_start_idx, cur_end_idx
            )
        with TraceAnnotation(
            "SD:MaskLogits",
            num_reqs=num_reqs,
            padded_num_reqs=padded_num_reqs,
            cur_start_idx=cur_start_idx,
            cur_end_idx=cur_end_idx,
        ):
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
        num_target_rows = target_logits.shape[0] if target_logits is not None else 0
        num_bonus_rows = bonus_logits.shape[0]
        with TraceAnnotation(
            "SD:PrepareSpecInput",
            num_reqs=num_reqs,
            num_target_rows=num_target_rows,
            num_bonus_rows=num_bonus_rows,
            cur_start_idx=cur_start_idx,
            cur_end_idx=cur_end_idx,
        ):
            (require_target, target_bitmask, require_bonus, bonus_bitmask, arange) = (
                self.prepare_spec_structured_decoding_input(
                    target_logits,
                    bonus_logits,
                    grammar_output,
                    scheduled_spec_decode_tokens,
                    draft_lengths_cpu,
                    cur_start_idx,
                    cur_end_idx,
                )
            )
        with TraceAnnotation(
            "SD:MaskSpecLogits",
            num_reqs=num_reqs,
            num_target_rows=num_target_rows,
            num_bonus_rows=num_bonus_rows,
            cur_start_idx=cur_start_idx,
            cur_end_idx=cur_end_idx,
        ):
            if target_logits is not None:
                assert require_target is not None
                assert target_bitmask is not None
                target_logits = self.structured_decode(
                    require_target, target_bitmask, target_logits, arange
                )
            bonus_logits = self.structured_decode(
                require_bonus, bonus_bitmask, bonus_logits, arange
            )
            return target_logits, bonus_logits

    def mask_invalid_draft_tokens(
        self,
        draft_token_ids: torch.Tensor,
        grammar_output: GrammarOutput,
        cur_start_idx: int,
        cur_end_idx: int,
        draft_lengths_cpu: np.ndarray,
    ) -> torch.Tensor:
        """Pins draft tokens beyond `grammar_output.num_acceptable_drafts` to -1.

        `StructuredOutputManager.grammar_bitmask` fills every row after the first
        `-1` placeholder with the all-permissive `_full_mask`, so a request's
        drafts at positions `>= num_acceptable_drafts` must be invalidated (-1)
        before rejection sampling. `draft_lengths_cpu` holds each chunk-local
        request's draft count, in the order of `draft_token_ids`.
        """
        req_ids = grammar_output.structured_output_request_ids
        if not req_ids or draft_token_ids.numel() == 0:
            return draft_token_ids

        chunk_num_reqs = cur_end_idx - cur_start_idx
        if chunk_num_reqs <= 0:
            return draft_token_ids

        req_id_to_index = self.runner.input_batch.req_id_to_index
        num_acceptable = grammar_output.num_acceptable_drafts
        # Requests without a grammar stay unconstrained (INT32_MAX).
        limits = np.full(chunk_num_reqs, np.iinfo(np.int32).max, dtype=np.int32)
        has_structured = False
        for grammar_idx, req_id in enumerate(req_ids):
            batch_index = req_id_to_index.get(req_id)
            if batch_index is not None and cur_start_idx <= batch_index < cur_end_idx:
                local_idx = batch_index - cur_start_idx
                limits[local_idx] = (
                    num_acceptable[grammar_idx] if num_acceptable is not None else 0
                )
                has_structured = True

        if not has_structured:
            return draft_token_ids

        if np.all(limits[: len(draft_lengths_cpu)] >= draft_lengths_cpu):
            return draft_token_ids
        invalid_mask_np = np.zeros(draft_token_ids.shape[0], dtype=np.bool_)
        row = 0
        for local_idx, num_drafts in enumerate(draft_lengths_cpu):
            n = int(num_drafts)
            limit = max(0, int(limits[local_idx]))
            if limit < n:
                invalid_mask_np[row + limit : row + n] = True
            row += n
        invalid_mask = torch.from_numpy(invalid_mask_np).to(
            device=draft_token_ids.device
        )
        return self.pin_invalid_draft_tokens(invalid_mask, draft_token_ids)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def pin_invalid_draft_tokens(
        self, invalid_mask: torch.Tensor, draft_token_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.where(invalid_mask, -1, draft_token_ids)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def structured_decode(
        self,
        require_struct_decoding: torch.Tensor,
        grammar_bitmask: torch.Tensor,
        logits: torch.Tensor,
        arange: torch.Tensor,
    ) -> torch.Tensor:
        return self._structured_decode(
            require_struct_decoding, grammar_bitmask, logits, arange
        )

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
            logits,
        )

    def apply_grammar_bitmask(
        self, logits: torch.Tensor, grammar_bitmask: torch.Tensor, arange: torch.Tensor
    ) -> torch.Tensor:
        # Both input checks use torch._assert, not assert or raise: it survives
        # python -O, and under torch.compile(fullgraph=True) the caller still
        # gets a plain AssertionError with the message.
        torch._assert(
            logits.shape[0] == grammar_bitmask.shape[0],
            "grammar_bitmask must have one row per logits row: "
            f"logits.shape[0]={logits.shape[0]}, "
            f"grammar_bitmask.shape[0]={grammar_bitmask.shape[0]}",
        )
        target_dim = logits.shape[-1]
        torch._assert(
            target_dim >= self.vocab_size,
            "TPU logits vocab dimension must be at least "
            f"vocab_size ({self.vocab_size}) (padded to hardware alignment "
            f"boundary), but got logits.shape[-1]={target_dim}",
        )
        # Unpack the bitmask for the entire batch at once.
        # grammar_bitmask: (B, N) where B=num_reqs, N=cdiv(vocab_size, 32)
        # arange: (32,)
        # (B, N, 1) and (1, 1, 32) broadcast to (B, N, 32)
        unpacked_bitmask = (
            torch.bitwise_right_shift(
                grammar_bitmask[:, :, None], arange[None, None, :]
            )
            & 1
        ) == 0
        # Reshape to (B, vocab_size) and apply to logits.
        # (B, N * 32) -> (B, vocab_size)
        unpacked_bitmask = unpacked_bitmask.reshape(logits.shape[0], -1)[
            :, : self.vocab_size
        ]
        if target_dim > self.vocab_size:
            unpacked_bitmask = torch.nn.functional.pad(
                unpacked_bitmask,
                (0, target_dim - self.vocab_size),
                value=True,
            )
        return torch.where(unpacked_bitmask, float("-inf"), logits)
