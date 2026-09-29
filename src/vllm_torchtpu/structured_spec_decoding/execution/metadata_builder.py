# ruff: noqa: E501
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
"""Metadata builder and tensor cache for structured speculative decoding."""

from typing import Any

import numpy as np
import torch

from vllm_torchtpu.structured_spec_decoding.types import SpecDecodeMetadata


class SpecDecodeMetadataBuilder:
    """Constructs device-side SpecDecodeMetadata and draft flag tensors."""

    @staticmethod
    def build_draft_flags_tensor(
        runner: Any,
        num_draft_tokens: np.ndarray,
        padded_logits_length: int,
        scheduled_spec_decode_flags: dict[str, list[int]] | None,
        start_index: int = 0,
    ) -> tuple[torch.Tensor | None, bool]:
        if not scheduled_spec_decode_flags:
            return None, True

        num_reqs = num_draft_tokens.shape[0]
        req_ids = runner.input_batch.req_ids[start_index : start_index + num_reqs]
        flag_chunks = []
        has_model_drafts = False
        has_any_drafts = False

        for req_id, n in zip(req_ids, num_draft_tokens):
            n_int = int(n)
            if n_int <= 0:
                continue
            has_any_drafts = True
            flags = scheduled_spec_decode_flags.get(req_id)
            if flags is not None:
                f = list(flags[:n_int])
                if len(f) < n_int:
                    f.extend([1] * (n_int - len(f)))
                if any(x == 1 for x in f):
                    has_model_drafts = True
                flag_chunks.append(np.array(f, dtype=np.int32))
            else:
                flag_chunks.append(np.ones(n_int, dtype=np.int32))
                has_model_drafts = True

        padded_draft_flags = np.zeros(padded_logits_length, dtype=np.int32)
        if flag_chunks:
            flat_flags = np.concatenate(flag_chunks)
            padded_draft_flags[: flat_flags.shape[0]] = flat_flags

        draft_flags_tensor = torch.from_numpy(padded_draft_flags).to(
            runner.device, non_blocking=True
        )
        return draft_flags_tensor, (has_model_drafts if has_any_drafts else False)

    @classmethod
    def build_metadata(
        cls,
        manager: Any,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
        scheduled_spec_decode_flags: dict[str, list[int]] | None = None,
        start_index: int = 0,
    ) -> SpecDecodeMetadata:
        """Calculates indices for speculative decoding forward pass and rejection sampling.  # noqa: E501  # noqa: E501

        Args:
          num_draft_tokens: The number of draft tokens for each request.
          cu_num_scheduled_tokens: The cumulative number of scheduled tokens.
          padded_num_reqs: The padded number of requests.
          scheduled_spec_decode_flags: Optional per-request draft flags list.
          start_index: Offset of this chunk into ``input_batch.req_ids``. Only the
            flag lookup needs it; every other input is already chunk-local.

        Returns:
          A SpecDecodeMetadata object containing the necessary indices for
          speculative decoding.
        """
        cache_key = (
            num_draft_tokens.tobytes(),
            cu_num_scheduled_tokens.tobytes(),
            padded_num_reqs,
        )
        if cache_key == manager._meta_cache_key:
            assert manager._meta_cache is not None
            assert manager._meta_cache_np is not None
            logits_indices, target_logits_indices, padded_logits_length = (
                manager._meta_cache_np
            )
            # draft_token_ids depends on this step's input_ids_cpu content —
            # always re-extracted and re-transferred.
            all_token_ids = manager.runner.input_ids_cpu.numpy()
            draft_token_ids = all_token_ids[logits_indices[target_logits_indices + 1]]
            padded_draft_token_ids = np.zeros(padded_logits_length, dtype=np.int32)
            padded_draft_token_ids[: draft_token_ids.shape[0]] = draft_token_ids
            draft_flags_tensor, has_model_drafts = cls.build_draft_flags_tensor(
                manager.runner,
                num_draft_tokens,
                padded_logits_length,
                scheduled_spec_decode_flags,
                start_index,
            )
            return SpecDecodeMetadata(
                draft_token_ids=torch.from_numpy(padded_draft_token_ids).to(
                    manager.runner.device, non_blocking=True
                ),
                draft_lengths=manager._meta_cache.draft_lengths,
                draft_lengths_cpu=manager._meta_cache.draft_lengths_cpu,
                target_logits_indices=manager._meta_cache.target_logits_indices,
                bonus_logits_indices=manager._meta_cache.bonus_logits_indices,
                final_logits_indices=manager._meta_cache.final_logits_indices,
                segment_ids=manager._meta_cache.segment_ids,
                group_indices=manager._meta_cache.group_indices,
                draft_flags=draft_flags_tensor,
                has_model_drafts=has_model_drafts,
            )

        # [num_reqs]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)

        arange_cpu_np = manager.runner.arange_np

        arange = np.concatenate([arange_cpu_np[:n] for n in num_sampled_tokens])

        # Step 2. logits_indices for all tokens (draft + target)
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )

        # Step 3. actual indices in the input batch
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the target logits indices
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )

        arange_draft = np.concatenate([arange_cpu_np[:n] for n in num_draft_tokens])
        target_logits_indices += arange_draft

        # The host indexing just below and the async device path both read
        # logits_indices[target_logits_indices + 1], the bonus slot must stay
        # within the verified tokens.
        assert (
            target_logits_indices.size == 0
            or int(target_logits_indices.max()) + 1 < logits_indices.shape[0]
        ), (
            f"draft bonus index {int(target_logits_indices.max()) + 1} >= "
            f"logits_indices length {logits_indices.shape[0]}"
        )

        # Compute the draft token ids for validation.
        all_token_ids = manager.runner.input_ids_cpu.numpy()
        draft_token_ids = all_token_ids[logits_indices[target_logits_indices + 1]]

        # Padding
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        padded_logits_length = _get_padded_token_len(
            manager.runner.num_tokens_paddings, logits_indices.shape[0]
        )

        padded_logits_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_logits_indices[: logits_indices.shape[0]] = logits_indices

        padded_bonus_logits_indices = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_bonus_logits_indices[: bonus_logits_indices.shape[0]] = (
            bonus_logits_indices
        )

        padded_num_draft_tokens = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_num_draft_tokens[: num_draft_tokens.shape[0]] = num_draft_tokens

        padded_draft_token_ids = np.zeros(padded_logits_length, dtype=np.int32)
        padded_draft_token_ids[: draft_token_ids.shape[0]] = draft_token_ids

        padded_target_logits_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_target_logits_indices[: target_logits_indices.shape[0]] = (
            target_logits_indices
        )

        segment_ids = np.repeat(
            np.arange(num_draft_tokens.shape[0], dtype=np.int64), num_draft_tokens
        )
        group_indices = (
            np.concatenate([np.arange(n, dtype=np.int32) for n in num_draft_tokens])
            if num_draft_tokens.any()
            else np.array([], dtype=np.int32)
        )

        padded_segment_ids = np.full(
            padded_logits_length, num_draft_tokens.shape[0], dtype=np.int64
        )
        padded_segment_ids[: segment_ids.shape[0]] = segment_ids

        padded_group_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_group_indices[: group_indices.shape[0]] = group_indices

        # CPU -> TPU copy.
        device = manager.runner.device
        draft_flags_tensor, has_model_drafts = cls.build_draft_flags_tensor(
            manager.runner,
            num_draft_tokens,
            padded_logits_length,
            scheduled_spec_decode_flags,
            start_index,
        )

        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.from_numpy(padded_draft_token_ids).to(
                device, non_blocking=True
            ),
            draft_lengths=torch.from_numpy(padded_num_draft_tokens).to(
                device, non_blocking=True
            ),
            draft_lengths_cpu=num_draft_tokens,
            target_logits_indices=torch.from_numpy(padded_target_logits_indices).to(
                device, non_blocking=True
            ),
            bonus_logits_indices=torch.from_numpy(padded_bonus_logits_indices).to(
                device, non_blocking=True
            ),
            final_logits_indices=torch.from_numpy(padded_logits_indices).to(
                device, non_blocking=True
            ),
            segment_ids=torch.from_numpy(padded_segment_ids).to(
                device, non_blocking=True
            ),
            group_indices=torch.from_numpy(padded_group_indices).to(
                device, non_blocking=True
            ),
            draft_flags=draft_flags_tensor,
            has_model_drafts=has_model_drafts,
        )
        # draft_lengths_cpu aliases the caller's array; copy so a later caller
        # mutation can't diverge the cache from its key.
        metadata.draft_lengths_cpu = num_draft_tokens.copy()
        manager._meta_cache_key = cache_key
        manager._meta_cache = metadata
        manager._meta_cache_np = (
            logits_indices,
            target_logits_indices,
            padded_logits_length,
        )
        return metadata
