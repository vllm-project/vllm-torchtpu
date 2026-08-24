# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ngram_proposer import NgramProposer

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


@dataclass
class SpecDecodeMetadata:
    """Metadata for speculative decoding on Torch/TPU, containing all necessary indices."""

    # [padded_logits_length] draft token id validated at each target_logits_indices slot.
    draft_token_ids: torch.Tensor
    # [padded_num_reqs] per-request draft-token count (device copy).
    draft_lengths: torch.Tensor
    # [num_reqs] per-request draft-token count (host copy, unpadded).
    draft_lengths_cpu: np.ndarray
    # [padded_logits_length] row into this step's logits/hidden-states for each draft token being validated.
    target_logits_indices: torch.Tensor
    # [padded_num_reqs] row into this step's logits/hidden-states for each request's bonus (last sampled) token.
    bonus_logits_indices: torch.Tensor
    # [padded_logits_length] row into the input batch for every draft + bonus
    # position the target model forward needs logits for.
    final_logits_indices: torch.Tensor
    # [padded_logits_length] request index each position belongs to; padding
    # uses the out-of-range sentinel num_reqs (not 0).
    segment_ids: torch.Tensor
    # [padded_logits_length] within-request draft-slot index of each position.
    group_indices: torch.Tensor


class SpeculativeDecodingManager:
    """Manages speculative decoding operations, including draft proposal and metadata generation."""

    def __init__(self, runner: TPUModelRunner):
        self.runner = runner
        # Cached draft tokens.
        self._draft_token_ids: Optional[list[list[int]]] = None
        # For async-scheduling: in-flight D2H copy of drafts on device,
        # as (cpu_tensor, ready_event, req_ids snapshot). See
        # stage_draft_token_ids_for_host.
        self._staged_draft_copy: Optional[tuple[torch.Tensor,
                                                Optional[torch.Event],
                                                list[str]]] = None
        self.spec_token_ids: dict[str, list[int]] = {}
        # Content-keyed cache for get_spec_decode_metadata: 7 of its 8 device
        # tensors are pure functions of (num_draft_tokens,
        # cu_num_scheduled_tokens, padded_num_reqs), which repeat every decode
        # step at steady state. Only draft_token_ids carries per-step values.
        self._meta_cache_key: Optional[tuple] = None
        self._meta_cache: Optional[SpecDecodeMetadata] = None
        # (logits_indices, target_logits_indices, padded_logits_length) from
        # the cached build — needed to re-extract draft_token_ids on a hit.
        self._meta_cache_np: Optional[tuple[np.ndarray, np.ndarray,
                                            int]] = None

    def take_draft_token_ids(self) -> Optional[DraftTokenIds]:
        if self._staged_draft_copy is not None:
            # For async scheduling: the engine only asks when a step carries
            # structured-output requests, to swap real drafts into the
            # deferred grammar-bitmask computation.
            drafts_cpu, ready_event, req_ids = self._staged_draft_copy
            self._staged_draft_copy = None
            if ready_event is not None:
                ready_event.synchronize()
            return DraftTokenIds(req_ids, drafts_cpu.tolist())
        if self._draft_token_ids is None:
            return None
        num_reqs = self.runner.input_batch.num_reqs
        req_ids = self.runner.input_batch.req_ids[:num_reqs]
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def stage_draft_token_ids_for_host(self, drafts: torch.Tensor) -> None:
        """Starts a non-blocking D2H copy of drafts on device.

        Async scheduling keeps drafts on device (return_device=True), but the
        deferred grammar-bitmask path still needs a host copy via
        take_draft_token_ids(). Launching the copy here enqueues it before
        the next step's forward, so take only waits for the copy itself.
        Note that req_ids must be snapshotted now because take_draft_token_ids()
        is called after the next step's execute_model mutates input_batch.
        """
        num_reqs = self.runner.input_batch.num_reqs
        assert drafts.shape[0] == num_reqs, (
            f"drafts rows {drafts.shape[0]} != num_reqs {num_reqs}")
        req_ids = list(self.runner.input_batch.req_ids[:num_reqs])
        drafts_cpu = drafts.to("cpu", non_blocking=True)
        ready_event = None
        if drafts.device.type != "cpu":
            ready_event = torch.tpu.Event()
            ready_event.record()
        self._staged_draft_copy = (drafts_cpu, ready_event, req_ids)

    def propose_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]] | None,
        discard_sampled_tokens_req_indices: list[int] | None = None,
        num_rejected_tokens_np: np.ndarray | None = None,
        scheduler_output=None,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Proposes draft token IDs based on the speculative decoding method.

    The single dispatch point for all draft proposing. eagle3 is called
    once per step from ``sample_tokens`` (sync AND async), seeded on-device
    from either ``next_tokens_per_chunk`` (spec verify step) or
    ``device_seed`` (prefill / non-spec step); ``return_device`` — the
    scheduling mode — only selects the return form (async: raw device
    tensor for the substitution assemblers; sync: host list cached for
    ``take_draft_token_ids``). ngram has no device path and passes
    host-materialized ``sampled_token_ids`` from the sync block instead.
    See ``Eagle3Proposer.propose`` for what each seed kwarg means.

    Args:
      sampled_token_ids: Host token IDs sampled in the previous step
        (ngram, and the eagle3 host-seed path kept for standalone/unit-test
        use); None/[] when seeding from device via
        next_tokens_per_chunk/device_seed.
      discard_sampled_tokens_req_indices: Request indices whose sampled
        tokens should be discarded (partial-prefill case).
      num_rejected_tokens_np: Per-request count of draft tokens rejected
        this step (host-seed path only; the device paths derive this
        on-device).
      scheduler_output: vLLM SchedulerOutput; carries the per-step spec
        token budget for ngram and the partial-prefill next-token lookup
        for eagle3.
      return_device: return the raw ``[num_reqs, K]`` device tensor
        instead of caching a host list for take_draft_token_ids.
      next_tokens_per_chunk: on-device rejection-sampler output (spec
        verify step).
      device_seed: on-device just-sampled tokens (prefill / non-spec
        step).

    Returns:
      The raw device tensor when return_device=True; otherwise None (the
      host list is cached in self._draft_token_ids).
    """
        num_reqs = self.runner.input_batch.num_reqs
        # Device-seeded / device-returning draft proposing is eagle3-only
        # (mirrors the tpu-inference reference's
        # `if async_scheduling: assert use_eagle()` guard; vLLM's config
        # validation also rejects ngram+async at engine construction).
        if (return_device or next_tokens_per_chunk is not None
                or device_seed is not None):
            assert (self.runner._is_async_drafter), (
                "device-seeded draft proposing is only supported with an async drafter"
            )
        if self.runner.speculative_config.method == "ngram":
            assert isinstance(self.runner.drafter, NgramProposer)
            assert sampled_token_ids is not None
            self._draft_token_ids = self.runner.drafter.propose(
                scheduler_output.num_spec_tokens_to_schedule,
                sampled_token_ids[:num_reqs],
                self.runner.input_batch.num_tokens_no_spec,
                self.runner.input_batch.token_ids_cpu,
            )
            return None
        elif (self.runner._is_async_drafter):
            result = self.runner.drafter.propose(
                sampled_token_ids[:num_reqs]
                if sampled_token_ids else sampled_token_ids,
                discard_sampled_tokens_req_indices or [],
                num_rejected_tokens_np,
                scheduler_output,
                return_device=return_device,
                next_tokens_per_chunk=next_tokens_per_chunk,
                device_seed=device_seed,
            )
            if return_device:
                return result
            self._draft_token_ids = result
            return None
        else:
            raise NotImplementedError(
                "Speculative decoding method "
                f"'{self.runner.speculative_config.method}' is not supported.")

    def get_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int,
    ) -> SpecDecodeMetadata:
        """Calculates indices for speculative decoding forward pass and rejection sampling.

    Args:
      num_draft_tokens: The number of draft tokens for each request.
      cu_num_scheduled_tokens: The cumulative number of scheduled tokens.
      padded_num_reqs: The padded number of requests.

    Returns:
      A SpecDecodeMetadata object containing the necessary indices for
      speculative decoding.
    """
        cache_key = (num_draft_tokens.tobytes(),
                     cu_num_scheduled_tokens.tobytes(), padded_num_reqs)
        if cache_key == self._meta_cache_key:
            assert self._meta_cache is not None
            assert self._meta_cache_np is not None
            logits_indices, target_logits_indices, padded_logits_length = (
                self._meta_cache_np)
            # draft_token_ids depends on this step's input_ids_cpu content —
            # always re-extracted and re-transferred.
            all_token_ids = self.runner.input_ids_cpu.numpy()
            draft_token_ids = all_token_ids[logits_indices[
                target_logits_indices + 1]]
            padded_draft_token_ids = np.zeros(padded_logits_length,
                                              dtype=np.int32)
            padded_draft_token_ids[:draft_token_ids.shape[0]] = draft_token_ids
            return SpecDecodeMetadata(
                draft_token_ids=torch.from_numpy(padded_draft_token_ids).to(
                    self.runner.device, non_blocking=True),
                draft_lengths=self._meta_cache.draft_lengths,
                draft_lengths_cpu=self._meta_cache.draft_lengths_cpu,
                target_logits_indices=self._meta_cache.target_logits_indices,
                bonus_logits_indices=self._meta_cache.bonus_logits_indices,
                final_logits_indices=self._meta_cache.final_logits_indices,
                segment_ids=self._meta_cache.segment_ids,
                group_indices=self._meta_cache.group_indices,
            )

        # [num_reqs]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens)

        arange_cpu_np = self.runner.arange_np

        arange = np.concatenate(
            [arange_cpu_np[:n] for n in num_sampled_tokens])

        # Step 2. logits_indices for all tokens (draft + target)
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)

        # Step 3. actual indices in the input batch
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the target logits indices
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)

        arange_draft = np.concatenate(
            [arange_cpu_np[:n] for n in num_draft_tokens])
        target_logits_indices += arange_draft

        # The host indexing just below and the async device path both read
        # logits_indices[target_logits_indices + 1], the bonus slot must stay
        # within the verified tokens.
        assert target_logits_indices.size == 0 or \
            int(target_logits_indices.max()) + 1 < logits_indices.shape[0], (
                f"draft bonus index {int(target_logits_indices.max()) + 1} >= "
                f"logits_indices length {logits_indices.shape[0]}")

        # Compute the draft token ids for validation.
        all_token_ids = self.runner.input_ids_cpu.numpy()
        draft_token_ids = all_token_ids[logits_indices[target_logits_indices +
                                                       1]]

        # Padding
        from vllm_torchtpu.runner.tpu_runner import _get_padded_token_len

        padded_logits_length = _get_padded_token_len(
            self.runner.num_tokens_paddings, logits_indices.shape[0])

        padded_logits_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_logits_indices[:logits_indices.shape[0]] = logits_indices

        padded_bonus_logits_indices = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_bonus_logits_indices[:bonus_logits_indices.shape[0]] = (
            bonus_logits_indices)

        padded_num_draft_tokens = np.zeros(padded_num_reqs, dtype=np.int32)
        padded_num_draft_tokens[:num_draft_tokens.shape[0]] = num_draft_tokens

        padded_draft_token_ids = np.zeros(padded_logits_length, dtype=np.int32)
        padded_draft_token_ids[:draft_token_ids.shape[0]] = draft_token_ids

        padded_target_logits_indices = np.zeros(padded_logits_length,
                                                dtype=np.int32)
        padded_target_logits_indices[:target_logits_indices.shape[0]] = (
            target_logits_indices)

        segment_ids = np.repeat(
            np.arange(num_draft_tokens.shape[0], dtype=np.int64),
            num_draft_tokens)
        group_indices = (np.concatenate([
            np.arange(n, dtype=np.int32) for n in num_draft_tokens
        ]) if num_draft_tokens.any() else np.array([], dtype=np.int32))

        padded_segment_ids = np.full(padded_logits_length,
                                     num_draft_tokens.shape[0],
                                     dtype=np.int64)
        padded_segment_ids[:segment_ids.shape[0]] = segment_ids

        padded_group_indices = np.zeros(padded_logits_length, dtype=np.int32)
        padded_group_indices[:group_indices.shape[0]] = group_indices

        # CPU -> TPU copy.
        device = self.runner.device

        metadata = SpecDecodeMetadata(
            draft_token_ids=torch.from_numpy(padded_draft_token_ids).to(
                device, non_blocking=True),
            draft_lengths=torch.from_numpy(padded_num_draft_tokens).to(
                device, non_blocking=True),
            draft_lengths_cpu=num_draft_tokens,
            target_logits_indices=torch.from_numpy(
                padded_target_logits_indices).to(device, non_blocking=True),
            bonus_logits_indices=torch.from_numpy(
                padded_bonus_logits_indices).to(device, non_blocking=True),
            final_logits_indices=torch.from_numpy(padded_logits_indices).to(
                device, non_blocking=True),
            segment_ids=torch.from_numpy(padded_segment_ids).to(
                device, non_blocking=True),
            group_indices=torch.from_numpy(padded_group_indices).to(
                device, non_blocking=True),
        )
        # draft_lengths_cpu aliases the caller's array; copy so a later caller
        # mutation can't diverge the cache from its key.
        metadata.draft_lengths_cpu = num_draft_tokens.copy()
        self._meta_cache_key = cache_key
        self._meta_cache = metadata
        self._meta_cache_np = (logits_indices, target_logits_indices,
                               padded_logits_length)
        return metadata
