# SPDX-License-Identifier: Apache-2.0
from collections.abc import Collection
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilderContext,
    stage_block_table_uploads)
from vllm_torchtpu.layers.common.sequence_layout import SequenceLayoutKind
from vllm_torchtpu.runner.padding import _get_padded_num_reqs_with_upper_limit

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


def _get_sequence_layout_planner_for_runner(runner: Any):
    return runner.sequence_layout_planner


class InputPreparationMixin:
    """Batch staging into device model inputs."""

    def _reorder_batch_for_rpa(
            self, scheduler_output: "SchedulerOutput") -> tuple[int, int]:
        """Reorder active requests into an RPA-friendly decode-first layout.

        decode-only requests come first and all remaining requests stay in the
        mixed bucket. We do not create a dedicated prefill-only bucket here.

        With speculative decoding the order is three segments:
        [1-token decodes][spec verify windows][prefill/mixed]. Ragged paged
        attention keeps its 1-token decode front segment, while the GDN
        kernel's windowed mode covers the first two segments contiguously
        (see `AttentionMetadata.mamba_request_distribution`).

        Returns:
            (num_decode, num_windowed): the number of 1-token decode requests
            and the number of windowed requests (decodes + speculative verify
            windows) after reordering. Without spec decoding both are equal.
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs <= 0:
            return 0, 0

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        def segment(req_id: str) -> int:
            # 0: 1-token decode, 1: speculative verify window,
            # 2: prefill/mixed.
            if scheduler_output.num_scheduled_tokens[req_id] == 1:
                return 0
            if req_id in spec_decode_tokens:
                return 1
            return 2

        def partition(start: int, end: int, bound: int) -> int:
            """Two-pointer partition of [start, end]: requests with
            segment <= bound before the rest. Returns the first index of the
            second part."""
            i, j = start, end
            while i < j:
                i_req_id = self.input_batch.req_ids[i]
                j_req_id = self.input_batch.req_ids[j]
                assert i_req_id is not None
                assert j_req_id is not None
                if segment(i_req_id) <= bound:
                    i += 1
                elif segment(j_req_id) > bound:
                    j -= 1
                else:
                    self.input_batch.swap_states(i, j)
                    i += 1
                    j -= 1
            if i == j and segment(self.input_batch.req_ids[i]) <= bound:
                i += 1
            return i

        # Pass 1: 1-token decode requests to the front.
        num_decode = partition(0, num_reqs - 1, 0)
        # Pass 2: speculative verify windows before prefill/mixed requests.
        num_windowed = num_decode
        if spec_decode_tokens and num_decode < num_reqs:
            num_windowed = partition(num_decode, num_reqs - 1, 1)
        return num_decode, num_windowed

    def _build_attention_metadata(
        self,
        num_tokens: int = 0,
        num_reqs: int = 0,
        max_query_len: int = 1,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        slot_mappings: torch.Tensor | None = None,
    ) -> tuple[dict[str, AttentionMetadata], None]:
        """Lean TPU group walk: stage one batched block-table upload, then
        build every group's metadata straight from the builder context.
        """
        ctx = self._attn_metadata_builder_ctx
        effective_num_reqs_padded = (num_reqs_padded if num_reqs_padded
                                     is not None else num_reqs)

        if ctx.staged_block_tables is None:
            # One batched H2D for every group's block table (zero-filled for
            # dummy runs); see `stage_block_table_uploads`.
            stage_block_table_uploads(self, ctx, effective_num_reqs_padded)

        common = SimpleNamespace(num_reqs=effective_num_reqs_padded)
        per_layer_attn_metadata: dict[str, AttentionMetadata] = {}
        for group_list in self.attn_groups:
            for group in group_list:
                built = group.metadata_builders[0].build(
                    common_prefix_len=0,
                    common_attn_metadata=common,
                )
                for layer_name in group.layer_names:
                    per_layer_attn_metadata[layer_name] = built
        return per_layer_attn_metadata, None

    def build_attention_metadata_for_layers(
        self,
        layer_names: Collection[str],
        num_reqs_padded: int,
    ) -> dict[str, AttentionMetadata]:
        """Build attention metadata for only the groups owning `layer_names`.

        `_build_attention_metadata` walks every KV cache group. A drafter
        needs only its own layers, and the group count tracks the layer count
        whenever vLLM's spec bucketing collapses — which a drafter mixing
        window and full attention causes on its own. Building every group and
        discarding all but the draft's costs a block-table H2D plus a handful
        of gathers per discarded group, per chunk, per step.

        Reads the same `_attn_metadata_builder_ctx` the full build does, so
        callers stage it identically.
        """
        # `AttentionMetadataBuilder.build` takes only `num_reqs` off the
        # common metadata; everything else it needs comes from
        # `_attn_metadata_builder_ctx`.
        selected_groups = []
        for group_list in self.attn_groups:
            for group in group_list:
                selected = [n for n in group.layer_names if n in layer_names]
                if selected:
                    selected_groups.append((group, selected))
        ctx = self._attn_metadata_builder_ctx
        if ctx.staged_block_tables is None and selected_groups:
            # One H2D for exactly the groups this targeted walk will build.
            stage_block_table_uploads(
                self,
                ctx,
                num_reqs_padded,
                gids={
                    group.metadata_builders[0].kv_cache_group_id
                    for group, _ in selected_groups
                })
        common = SimpleNamespace(num_reqs=num_reqs_padded)
        metadata: dict[str, AttentionMetadata] = {}
        for group, selected in selected_groups:
            built = group.metadata_builders[0].build(
                common_prefix_len=0,
                common_attn_metadata=common,
            )
            for name in selected:
                metadata[name] = built
        return metadata

    def _prepare_inputs(self,
                        scheduler_output: "SchedulerOutput",
                        start_index: int,
                        num_decode_reqs: int,
                        num_windowed_reqs: int | None = None):
        if num_windowed_reqs is None:
            num_windowed_reqs = num_decode_reqs
        assert scheduler_output.total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        assert start_index < num_reqs
        self._wait_input_staging_fence()

        # Get the number of scheduled tokens for each request.
        use_max_model_len = self.most_model_len is None
        num_scheduled_tokens_per_req = []
        max_num_scheduled_tokens_all_reqs = 0
        end_index = start_index

        # Use either most_model_len or max_model_len depending on request size.
        for i in range(start_index, num_reqs):
            req_id = self.input_batch.req_ids[i]
            assert req_id is not None
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if (not use_max_model_len and self.most_model_len is not None
                    and num_tokens > self.most_model_len):
                use_max_model_len = True
            num_scheduled_tokens_per_req.append(num_tokens)
        if use_max_model_len:
            if len(num_scheduled_tokens_per_req) > self.num_reqs_max_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[:
                                                                            self
                                                                            .
                                                                            num_reqs_max_model_len]
                end_index = start_index + self.num_reqs_max_model_len
            else:
                end_index = num_reqs
        else:
            assert self.num_reqs_most_model_len is not None
            if len(num_scheduled_tokens_per_req
                   ) > self.num_reqs_most_model_len:
                num_scheduled_tokens_per_req = num_scheduled_tokens_per_req[:
                                                                            self
                                                                            .
                                                                            num_reqs_most_model_len]
                end_index = start_index + self.num_reqs_most_model_len
            else:
                end_index = num_reqs
        max_num_scheduled_tokens_all_reqs = max(num_scheduled_tokens_per_req)
        num_scheduled_tokens_per_req = np.array(num_scheduled_tokens_per_req,
                                                dtype=np.int32)
        total_num_scheduled_tokens = sum(num_scheduled_tokens_per_req)
        assert max_num_scheduled_tokens_all_reqs > 0

        num_reqs = len(num_scheduled_tokens_per_req)

        sequence_layout_planner = _get_sequence_layout_planner_for_runner(self)
        sequence_layout_planner.reserve_host_token_capacity(
            self, int(total_num_scheduled_tokens))

        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Fast path for decode-only: all requests have exactly 1 token.
        if max_num_scheduled_tokens_all_reqs == 1:
            # Pure decode: each request schedules exactly 1 token.
            # req_indices = [0, 1, ..., num_reqs-1]
            req_indices = self.arange_np[start_index:start_index + num_reqs]
            # positions = num_computed_tokens for each request
            positions_np = self.positions_np[:num_reqs]
            np.copyto(
                positions_np, self.input_batch.
                num_computed_tokens_cpu[start_index:start_index + num_reqs])
            # token_indices = positions + req_index * max_model_len
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])
            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:num_reqs],
            )
        else:
            # General path: mixed prefill + decode.
            # Get request indices.
            req_indices = np.repeat(
                self.arange_np[start_index:start_index + num_reqs],
                num_scheduled_tokens_per_req)

            # Get batched arange.
            # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
            arange = np.concatenate(
                [self.arange_np[:n] for n in num_scheduled_tokens_per_req])

            # Get positions.
            positions_np = self.positions_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

            # Get token indices.
            token_indices = (
                positions_np +
                req_indices * self.input_batch.token_ids_cpu.shape[1])

            torch.index_select(
                self.input_batch.token_ids_cpu_tensor.flatten(),
                0,
                torch.from_numpy(token_indices),
                out=self.input_ids_cpu[:total_num_scheduled_tokens],
            )

        # Prepare the attention metadata.
        self.query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens_per_req,
                  out=self.query_start_loc_np[1:num_reqs + 1])
        # Keep padded entries equal to the last valid location so padded
        # requests have zero length instead of a negative q_len.
        self.query_start_loc_np[num_reqs +
                                1:] = self.query_start_loc_np[num_reqs]

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[start_index:start_index +
                                                     num_reqs] +
            num_scheduled_tokens_per_req)

        request_major_input_ids_cpu: torch.Tensor
        if self._pcp_mtp_k1_enabled:
            # prepare_real() may repack input_ids_cpu in place. Preserve only
            # the real request-major prefix before crossing that boundary.
            request_major_input_ids_cpu = self.input_ids_cpu.narrow(
                0, 0, int(total_num_scheduled_tokens)).clone()

        if use_max_model_len:
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            target_num_reqs = self.num_reqs_most_model_len
        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.max_num_reqs)

        layout_plan = sequence_layout_planner.prepare_real(
            runner=self,
            scheduler_output=scheduler_output,
            start_index=start_index,
            num_reqs=num_reqs,
            num_scheduled_tokens_per_req=num_scheduled_tokens_per_req,
            total_num_scheduled_tokens=int(total_num_scheduled_tokens),
            use_max_model_len=use_max_model_len,
            target_num_reqs=target_num_reqs,
            padded_num_reqs=padded_num_reqs,
        )
        self._last_sequence_layout_plan = layout_plan
        padded_total_num_scheduled_tokens = (
            layout_plan.global_padded_num_tokens)
        request_major_input_ids = None
        if self._pcp_mtp_k1_enabled:
            request_major_input_ids_cpu_padded = torch.zeros(
                layout_plan.global_padded_num_tokens,
                dtype=request_major_input_ids_cpu.dtype,
            )
            request_major_input_ids_cpu_padded.narrow(
                0, 0, layout_plan.global_num_tokens).copy_(
                    request_major_input_ids_cpu)
            request_major_input_ids = request_major_input_ids_cpu_padded.to(
                self.device, non_blocking=True)
        local_total_num_scheduled_tokens = layout_plan.local_num_tokens
        local_padded_total_num_scheduled_tokens = (
            layout_plan.local_padded_num_tokens)
        local_token_slice = layout_plan.token_slice
        if layout_plan.kind is SequenceLayoutKind.ALL:
            # Zero out to avoid spurious values from prev iteration.
            self.input_ids_cpu[total_num_scheduled_tokens:
                               padded_total_num_scheduled_tokens] = 0
        self.input_ids = self.input_ids_cpu[local_token_slice].to(
            self.device, non_blocking=True)
        if self.uses_mrope:
            if layout_plan.kind is SequenceLayoutKind.ALL:
                self.mrope_positions.cpu[:, total_num_scheduled_tokens:
                                         padded_total_num_scheduled_tokens] = 0
            self.position_ids = self.mrope_positions.cpu[:,
                                                         local_token_slice].to(
                                                             self.device,
                                                             non_blocking=True)
        else:
            self.position_ids = self.positions_cpu[local_token_slice].to(
                self.device, non_blocking=True)
        if use_max_model_len:
            seq_lens = self.seq_lens_cpu[:self.num_reqs_max_model_len].to(
                self.device, non_blocking=True)
            target_num_reqs = self.num_reqs_max_model_len
        else:
            assert self.num_reqs_most_model_len is not None
            seq_lens = self.seq_lens_cpu[:self.num_reqs_most_model_len].to(
                self.device, non_blocking=True)
            target_num_reqs = self.num_reqs_most_model_len

        # Async spec: seq_lens/positions were advanced optimistically (every
        # draft from the previous step assumed accepted). Subtract the real
        # per-request rejected count on-device, keyed by req position.
        if (self.scheduler_config.async_scheduling and self.speculative_config
                and self._pre_async_results is not None
                and self._pre_async_results.spec_decode_num_rejected_tokens
                is not None):
            from vllm_torchtpu.runner.tpu_runner_async_output import \
                subtract_num_rejected_tokens
            seq_idx_np = np.full(target_num_reqs, -1, dtype=np.int32)
            pos_idx_np = np.full(padded_total_num_scheduled_tokens,
                                 -1,
                                 dtype=np.int32)
            acc = 0
            for i in range(num_reqs):
                req_id = self.input_batch.req_ids[start_index + i]
                n_sched = int(num_scheduled_tokens_per_req[i])
                pos = self._pre_async_results.req_id_to_index_copy.get(req_id)
                if pos is not None:
                    seq_idx_np[i] = pos
                    pos_idx_np[acc:acc + n_sched] = pos
                acc += n_sched
            seq_lens, self.position_ids = subtract_num_rejected_tokens(
                seq_lens, self.position_ids,
                self._pre_async_results.spec_decode_num_rejected_tokens,
                torch.from_numpy(seq_idx_np).to(self.device,
                                                non_blocking=True),
                torch.from_numpy(pos_idx_np).to(self.device,
                                                non_blocking=True))

        # For decode-only case, cache constant device tensors to skip H2D.
        # query_start_loc, logits_indices, and request_distribution don't
        # change between decode steps for a given (num_reqs, padded_num_reqs).
        is_decode_only = (max_num_scheduled_tokens_all_reqs == 1)
        decode_cache_key = (
            num_reqs,
            padded_num_reqs,
            use_max_model_len,
            layout_plan.descriptor.cache_key,
        )

        # Speculative decoding metadata.
        spec_decode_metadata = None
        if self.speculative_config:
            num_draft_tokens = np.array([
                len(
                    scheduler_output.scheduled_spec_decode_tokens.get(
                        req_id, ())) for req_id in
                self.input_batch.req_ids[start_index:start_index + num_reqs]
            ],
                                        dtype=np.int32)
            if num_draft_tokens.any():
                spec_decode_metadata = self.spec_decode_manager.get_spec_decode_metadata(
                    num_draft_tokens, self.query_start_loc_np[1:num_reqs + 1],
                    padded_num_reqs)

        # Default decode layouts keep logits_indices stable for a fixed
        # (num_reqs, padded_num_reqs) bucket. PCP streaming layouts provide
        # explicit rank-major logits indices that can change as q_start crosses
        # an interleave/rank boundary, so they must not reuse a stale cached
        # logits_indices tensor.
        can_cache_decode_metadata = (is_decode_only
                                     and spec_decode_metadata is None and
                                     layout_plan.logits_indices_cpu is None)

        if (can_cache_decode_metadata
                and decode_cache_key == self._decode_device_cache_key):
            # Reuse cached device tensors — skip 3 H2D copies.
            query_start_loc = self._cached_query_start_loc
            logits_indices = self._cached_logits_indices
            request_distribution = self._cached_request_distribution
        else:
            # Compute and copy to device (first call or shape changed).
            if use_max_model_len:
                query_start_loc = self.query_start_loc_cpu[:self.
                                                           num_reqs_max_model_len
                                                           + 1].to(
                                                               self.device,
                                                               non_blocking=True
                                                           )
            else:
                query_start_loc = self.query_start_loc_cpu[:self.
                                                           num_reqs_most_model_len
                                                           + 1].to(
                                                               self.device,
                                                               non_blocking=True
                                                           )

            if spec_decode_metadata is not None:
                logits_indices = spec_decode_metadata.final_logits_indices
            elif layout_plan.logits_indices_cpu is not None:
                logits_indices = layout_plan.logits_indices_cpu.to(
                    self.device, non_blocking=True)
            else:
                # Indices at which we sample (positions of last token in the
                # sequence). Padded to avoid recompiling when `num_reqs` varies.
                logits_indices = (
                    self.query_start_loc_cpu[1:padded_num_reqs + 1] - 1).to(
                        self.device, non_blocking=True)

            # For the V3 kernel, request_distribution is
            # [decode_end, prefill_end, mixed_end]. We put decode requests first,
            # no dedicated prefill-only bucket, and all remaining requests in mixed mode.
            chunk_num_decode = max(
                0, min(num_decode_reqs - start_index, num_reqs))
            self._request_distribution_cpu[0] = chunk_num_decode
            self._request_distribution_cpu[1] = chunk_num_decode
            self._request_distribution_cpu[2] = num_reqs
            request_distribution = self._request_distribution_cpu.to(
                self.device, non_blocking=True)

            if can_cache_decode_metadata:
                # Cache for future decode steps.
                self._decode_device_cache_key = decode_cache_key
                self._cached_query_start_loc = query_start_loc
                self._cached_logits_indices = logits_indices
                self._cached_request_distribution = request_distribution

        # Compact-mamba: per-request recurrent-slot ids for this chunk. Only
        # built for hybrid models (slot pool initialized in
        # initialize_kv_cache); None otherwise so non-mamba models are
        # unaffected.
        mamba_state_indices = (self._build_mamba_state_indices(
            start_index, num_reqs, target_num_reqs)
                               if self._has_mamba_state else None)

        # Spec decode with mamba layers: the GDN kernel's windowed segment
        # covers 1-token decodes AND speculative verify windows (the batch is
        # ordered [decode][verify][prefill/mixed] by _reorder_batch_for_rpa),
        # while RPA keeps its 1-token-decode-only front segment.
        mamba_request_distribution = None
        if self.mamba_slot_read_offsets is not None:
            chunk_num_windowed = max(
                0, min(num_windowed_reqs - start_index, num_reqs))
            chunk_num_decode = max(
                0, min(num_decode_reqs - start_index, num_reqs))
            # One H2D copy, two views (see the staging tensor's comment for
            # why these must not be independent device tensors). Overrides
            # the RPA tensor built above so both fields always come from the
            # same base tensor, matching the traced graph structure.
            combined = self._combined_request_distribution_cpu
            combined[0] = chunk_num_decode
            combined[1] = chunk_num_decode
            combined[2] = num_reqs
            combined[3] = chunk_num_windowed
            combined[4] = chunk_num_windowed
            combined[5] = num_reqs
            combined_device = combined.to(self.device, non_blocking=True)
            request_distribution = combined_device[0:3]
            mamba_request_distribution = combined_device[3:6]

        self._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            mamba_state_indices=mamba_state_indices,
            mamba_slot_read_offsets=self.mamba_slot_read_offsets,
            mamba_ckpt_window=self._mamba_ckpt_window,
            mamba_request_distribution=mamba_request_distribution,
            # Unified pool + spec decode: the mamba group builders append
            # their per-request state block ids here for the post-sampling
            # read-offset scatter.
            unified_mamba_state_indices=(
                [] if self._unified_kv_layout
                and self.mamba_slot_read_offsets is not None else None),
            sequence_layout_descriptor=layout_plan.descriptor,
        )
        slot_mappings = self.empty_slot_mappings
        per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = self._build_attention_metadata(
            num_tokens=local_total_num_scheduled_tokens,
            num_reqs=target_num_reqs,
            max_query_len=max_num_scheduled_tokens_all_reqs,
            num_tokens_padded=local_padded_total_num_scheduled_tokens,
            num_reqs_padded=target_num_reqs,
            slot_mappings=slot_mappings,
        )

        if self.lora_config is not None:
            # We need to respect padding when activating LoRA adapters
            padded_num_scheduled_tokens_per_req = np.copy(
                num_scheduled_tokens_per_req
            )  # Copying to avoid accidental state corruption bugs
            padded_num_scheduled_tokens_per_req[-1] += (
                padded_total_num_scheduled_tokens - total_num_scheduled_tokens)

            self.set_active_loras(self.input_batch,
                                  padded_num_scheduled_tokens_per_req)

        # Prepare token substitution indices
        cur_input_indices, pre_next_tokens_indices = self._prepare_async_token_substitution_indices(
            start_index, num_reqs, num_scheduled_tokens_per_req)

        self._collect_mamba_state_seed_copies(scheduler_output, start_index,
                                              num_reqs)

        return (
            per_layer_attn_metadata,
            logits_indices,
            padded_num_reqs,
            num_reqs,
            end_index,
            cur_input_indices,
            pre_next_tokens_indices,
            spec_decode_metadata,
            request_major_input_ids,
            layout_plan,
        )

    def _record_input_staging_fence(self) -> None:
        """Fence the uploads issued from the reusable host staging tensors.

        Every `.to(self.device, non_blocking=True)` of a staging tensor keeps
        reading its host memory until the transfer completes, and the same
        tensors are rewritten by the next chunk's input preparation. The
        event is recorded before the forward is dispatched, so waiting on it
        never waits for the forward itself.
        """
        fence = torch.tpu.Event()
        fence.record()
        self._input_staging_fence = fence

    def _wait_input_staging_fence(self) -> None:
        fence = self._input_staging_fence
        if fence is not None:
            fence.synchronize()
            self._input_staging_fence = None

    def _get_model_inputs(
        self,
        input_ids: torch.Tensor,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ):
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            # _gather_mm_embeddings returns is_mm_embed sized to the number
            # of scheduled tokens, but TPU pads input_ids to a fixed bucket.
            # Pad the multimodal mask (False for pad positions) and move it
            # to the input device so the masked_scatter in embed_input_ids
            # aligns with the padded inputs_embeds.
            if is_mm_embed is not None:
                n = input_ids.shape[0]
                if is_mm_embed.shape[0] < n:
                    pad = torch.zeros(
                        n - is_mm_embed.shape[0],
                        dtype=is_mm_embed.dtype,
                        device=is_mm_embed.device,
                    )
                    is_mm_embed = torch.cat([is_mm_embed, pad])
                is_mm_embed = is_mm_embed.to(input_ids.device)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds = self.model.embed_input_ids(
                input_ids,
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            return None, inputs_embeds
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            return input_ids, None

    def _token_padding_update(self, num_tokens_padded: int) -> None:
        """Record which of this step's token rows are padding."""
        padding_state = getattr(self, "_token_padding_state", None)
        if padding_state is None:
            return
        plan = self._last_sequence_layout_plan
        if plan is not None and plan.kind is SequenceLayoutKind.ALL:
            num_valid_tokens = min(plan.local_num_tokens, num_tokens_padded)
        else:
            # Padding-mask support is currently limited to the ALL layout.
            num_valid_tokens = num_tokens_padded
        padding_state.update(num_valid_tokens, num_tokens_padded)
