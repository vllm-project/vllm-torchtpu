# SPDX-License-Identifier: Apache-2.0
from collections.abc import Collection
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, stage_block_table_uploads)
from vllm_torchtpu.layers.common.sequence_layout import SequenceLayoutKind

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
