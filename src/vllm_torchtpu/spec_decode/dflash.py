# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from vllm.compilation.backends import set_model_tag
from vllm.config import VllmConfig, get_layers_from_vllm_config, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader

from vllm_torchtpu.layers.adapter.attention import PallasAttentionBackendImpl
from vllm_torchtpu.layers.core.attention_metadata import (
    AttentionMetadata,
    AttentionMetadataBuilderContext,
)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode.utils import (
    DraftChunkInputs,
    _force_draft_tp1,
    maybe_share_embeddings,
    maybe_share_lm_head,
    normalize_draft_config,
)
from vllm_torchtpu.utils import synchronize_tensors

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class DFlashProposer:
    """DFlash draft proposer for TPU.

    Implements a parallel block-level speculative decoding framework.
    """

    def __init__(
        self,
        runner: TPUModelRunner,
        vllm_config: VllmConfig,
    ):
        self.runner = runner
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        # Read and normalize draft configurations
        hf_config = self.speculative_config.draft_model_config.hf_config
        self.dflash_config = normalize_draft_config(hf_config)
        self.mask_token_id = self.dflash_config["mask_token_id"]
        self._target_layer_ids = list(self.dflash_config.get("target_layer_ids", []))

        draft_tp = self.speculative_config.draft_tensor_parallel_size
        target_tp = self.vllm_config.parallel_config.tensor_parallel_size
        # Default to (draft_tp == target tp) when unset. vLLM's
        # _verify_and_get_draft_tp already resolves an unset value to target tp
        # for dflash, but set it explicitly here so self-documented.
        if draft_tp is None:
            draft_tp = target_tp
            self.speculative_config.draft_tensor_parallel_size = draft_tp
        # Only two draft parallelisms are supported: fully REPLICATED (tp=1) or
        # SHARDED across the whole TP group (draft_tp == target_tp).
        if draft_tp not in (1, target_tp):
            raise ValueError(
                f"dflash draft_tensor_parallel_size={draft_tp} is unsupported "
                f"on TPU: it must be 1 (replicated draft) or {target_tp} "
                f"(== target tensor_parallel_size, sharded draft)."
            )
        self._draft_replicated = draft_tp == 1
        # Independent from replication: target TP1 / draft TP1 can alias.
        self._draft_tp_matches_target = draft_tp == target_tp
        logger.info(
            "DFlash draft parallelism: %s (draft_tp=%s).",
            "REPLICATED (tp=1)" if self._draft_replicated else "SHARDED",
            draft_tp,
        )

        self._max_model_len = vllm_config.model_config.max_model_len
        self.draft_model = None
        self._draft_has_moe_cache: bool | None = None
        self._draft_attn_layer_names: set[str] | None = None
        self.draft_chunks: list[DraftChunkInputs] | None = None
        self._static_attn_tensors_cache: dict[
            int, tuple[torch.Tensor, torch.Tensor]
        ] = {}
        # Keyed by (num_tokens, num_reqs, num_blocks); one entry per replay
        # shape, see `_dummy_kv_update_inputs`.
        self._dummy_kv_input_cache: dict[
            tuple, tuple[list[torch.Tensor], AttentionMetadata]
        ] = {}
        # Keyed by the chunk's padded request count; the all-zero
        # `draft_lengths` a chunk without spec metadata routes on.
        self._zero_draft_lengths_cache: dict[int, torch.Tensor] = {}

    @property
    def block_size(self) -> int:
        """Tokens per request in a draft forward (see _query_block_size)."""
        return self._query_block_size(self.speculative_config.num_speculative_tokens)

    def _get_static_attn_tensors(
        self, padded_num_reqs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch or create persistent static TPU tensors for draft attention metadata."""
        if padded_num_reqs not in self._static_attn_tensors_cache:
            qsl = (
                torch.arange(
                    padded_num_reqs + 1, dtype=torch.int32, device=self.runner.device
                )
                * self.block_size
            )
            rd = torch.tensor(
                [0, 0, padded_num_reqs], dtype=torch.int32, device=self.runner.device
            )
            self._static_attn_tensors_cache[padded_num_reqs] = (qsl, rd)
        return self._static_attn_tensors_cache[padded_num_reqs]

    # Whether load_model overwrites the draft's embed_tokens/lm_head with the
    # target's unconditionally. DSpark sets this False to honor the draft's
    # has_own_embed_tokens / has_own_lm_head flags instead.
    _force_share_target_embeddings = True

    def _query_block_size(self, K: int) -> int:
        """Query tokens per request: 1 anchor + K mask slots. DSpark's dense
        layout overrides this to K (the anchor is the first prediction)."""
        return K + 1

    def load_model(self, target_model) -> None:
        """Load the draft model, share embeddings, and tag target layers."""
        # Tag the target model to extract aux hidden states
        target_layers = [idx + 1 for idx in self._target_layer_ids]
        if hasattr(target_model, "set_aux_hidden_state_layers"):
            target_model.set_aux_hidden_state_layers(tuple(target_layers))
        elif hasattr(target_model, "model") and hasattr(
            target_model.model, "_set_aux_hidden_state_layers"
        ):
            target_model.model._set_aux_hidden_state_layers(tuple(target_layers))
        else:
            raise RuntimeError(
                "Target model does not support _set_aux_hidden_state_layers"
            )
        logger.info(
            "Tagged target model for DFlash auxiliary hidden states extraction."
        )

        # Snapshot the target's attention layer names before the draft model
        # adds its own; diff after load gives us the draft layer names.
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).keys()
        )

        self._load_draft_model()

        all_attn_layers = get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase
        )
        self._draft_attn_layer_names = (
            set(all_attn_layers.keys()) - target_attn_layer_names
        )

        # Share Embeddings and LM Head. DFlash drafts always predict target
        # vocab, so sharing is forced; DSpark overrides the flag because a
        # reduced-vocab checkpoint carries its own lm_head/embed_tokens.
        maybe_share_embeddings(
            self.draft_model,
            target_model,
            self._draft_tp_matches_target,
            force_share=self._force_share_target_embeddings,
        )
        maybe_share_lm_head(
            self.draft_model,
            target_model,
            self._draft_replicated,
            self._draft_tp_matches_target,
            force_share=self._force_share_target_embeddings,
        )

        if hasattr(self.draft_model, "get_draft_attn_causal"):
            layer_causal_list = self.draft_model.get_draft_attn_causal()
        elif hasattr(self.draft_model.model, "get_draft_attn_causal"):
            layer_causal_list = self.draft_model.model.get_draft_attn_causal()
        else:
            raise RuntimeError("Draft model does not support get_draft_attn_causal")

        with set_vllm_model_wrapper_context(
            mesh=self.runner.mesh, vllm_config=self.vllm_config
        ):
            for i, layer in enumerate(self.draft_model.model.layers):
                sub_attn = getattr(layer, "self_attn", None)
                attn_obj = (
                    getattr(sub_attn, "attn", None) if sub_attn is not None else None
                )
                if attn_obj is None:
                    continue
                attn_impl = attn_obj.impl
                if isinstance(attn_impl, PallasAttentionBackendImpl):
                    attn_impl.layer_idx = i
                    attn_impl.use_causal_mask = layer_causal_list[i]
                    attn_impl.initialize_kernel(attn_obj)

    def _load_draft_model(self) -> None:
        logger.info("Loading DFlash draft model...")
        model_loader = get_model_loader(self.vllm_config.load_config)
        draft_tp1_ctx = (
            _force_draft_tp1() if self._draft_replicated else contextlib.nullcontext()
        )
        draft_vllm_config = copy.copy(self.vllm_config)
        draft_model_config = copy.copy(self.speculative_config.draft_model_config)
        draft_model_config.runner_type = "draft"
        with (
            set_model_tag("dflash_head"),
            set_vllm_model_wrapper_context(mesh=self.runner.mesh),
            set_current_vllm_config(draft_vllm_config),
            draft_tp1_ctx,
        ):
            self.draft_model = model_loader.load_model(
                vllm_config=draft_vllm_config,
                model_config=draft_model_config,
            )

    def _get_padded_len(self, target_len: int) -> int:
        for p in self.runner.num_tokens_paddings:
            if p >= target_len:
                return p
        return target_len

    def _build_attn_metadata_for_draft(
        self,
        *,
        num_reqs: int,
        start_index: int,
        use_max_model_len: bool,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_distribution: torch.Tensor,
        position_ids_override: torch.Tensor | None,
        padded_num_reqs: int,
    ) -> dict:
        """Build the draft layers' attention metadata. The real path and
        the dummies share this so they dispatch the identical programs.
        No mamba fields: DFlash drafters are attention-only."""
        runner = self.runner
        assert self._draft_attn_layer_names is not None, (
            "load_model() must run before draft attention metadata is built"
        )
        saved_ctx = runner._attn_metadata_builder_ctx
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=num_reqs,
            start_index=start_index,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=position_ids_override,
        )
        try:
            # Only the draft layers' KV cache groups. The full build walks
            # every group in the deployment and we would discard all but
            # these — see `build_attention_metadata_for_layers`.
            return runner.build_attention_metadata_for_layers(
                self._draft_attn_layer_names, padded_num_reqs
            )
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

    def _build_draft_attn_metadata(
        self,
        chunk: DraftChunkInputs,
        num_tokens_padded: int,
        seq_lens: torch.Tensor,
    ) -> dict:
        """
        Construct attention metadata for DFlash draft forward.
        Scales query boundaries and request token distribution to block_size.
        """
        chunk_ctx = chunk.attn_ctx
        # `num_tokens_padded` is a bucket of at least one full block.
        padded_num_reqs = num_tokens_padded // self.block_size
        query_start_loc, request_distribution = self._get_static_attn_tensors(
            padded_num_reqs
        )
        # num_reqs must be the REAL request count: the staging walk copies
        # exactly ctx.num_reqs block-table rows, and rows beyond the live
        # batch hold departed requests' freed block ids. With the real
        # count, padding rows stage as zeros -> the null block.
        return self._build_attn_metadata_for_draft(
            num_reqs=chunk.num_reqs,
            start_index=chunk.start_index,
            use_max_model_len=chunk_ctx.use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=chunk_ctx.position_ids_override,
            padded_num_reqs=padded_num_reqs,
        )

    def _prepare_dflash_inputs(
        self,
        chunk: DraftChunkInputs,
        next_tokens_device: torch.Tensor | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepares input_ids and positions tensors for DFlash draft execution.

        Each request block is formatted as:
          input_ids: [base_token, MASK_TOKEN, ..., MASK_TOKEN]  (len = block_size)
          positions: [base_pos, base_pos+1, ..., base_pos+K]    (len = block_size)

        The base token is the last verified token from the target model.
        """
        # 1. Calculate static padded shapes
        num_reqs = chunk.num_reqs
        block_size = self.block_size
        total_draft_tokens = num_reqs * block_size

        # Round up to the nearest static padded length to avoid XLA recompilation
        if self._dp_lockstep_sharded():
            # EP-DP lockstep: every rank must emit identical collective
            # shapes, and num_reqs is rank-local. Pad to the step's DP-wide
            # max per-chunk request count, coordinated by the same
            # all-reduce that sizes the target bucket
            # (`_dp_coordinated_step`); run_dp_dummy_draft replays the same
            # shape on idle ranks.
            padded_len = self._dp_draft_bucket()
            # The coordinated max is an exact per-chunk bound
            # (_count_input_chunks); a shortfall here would silently
            # truncate draft inputs via negative F.pad, so fail loudly.
            assert padded_len >= total_draft_tokens, (
                f"lockstep draft bucket {padded_len} < local chunk's "
                f"{total_draft_tokens} draft tokens"
            )
        else:
            padded_len = self._get_padded_len(total_draft_tokens)

        query_start_loc_device = chunk.attn_ctx.query_start_loc
        position_ids_device = chunk.position_ids
        if position_ids_device.dim() > 1:
            position_ids_device = position_ids_device[0]

        # Route to the appropriate pre-compiled XLA Graph
        if next_tokens_device is not None:
            # No spec metadata means nothing was verified this step, so every
            # request anchors on the prompt-chunk path.
            draft_lengths = chunk.draft_lengths
            if draft_lengths is None:
                draft_lengths = self._zero_draft_lengths(num_reqs)
            input_ids, positions, seq_lens = self._tpu_build_dflash_inputs_next_tokens(
                next_tokens_device,
                query_start_loc_device,
                position_ids_device,
                draft_lengths,
                num_reqs,
                block_size,
                padded_len,
                self.mask_token_id,
                INVALID_TOKEN_ID,
            )
        elif device_seed is not None:
            input_ids, positions, seq_lens = self._tpu_build_dflash_inputs_seed(
                device_seed,
                query_start_loc_device,
                position_ids_device,
                num_reqs,
                block_size,
                padded_len,
                self.mask_token_id,
            )
        else:
            raise RuntimeError(
                "DFlash propose called without next_tokens_device or device_seed. "
                "This drafter is designed strictly for the fused device path."
            )

        return input_ids, positions, seq_lens

    def _zero_draft_lengths(self, num_reqs: int) -> torch.Tensor:
        """Cached all-zero `draft_lengths` for a chunk without spec
        metadata, padded like the real tensor so both share one program."""
        from vllm_torchtpu.runner.tpu_runner import (
            _get_padded_num_reqs_with_upper_limit,
        )

        padded_num_reqs = _get_padded_num_reqs_with_upper_limit(
            num_reqs, self.runner.max_num_reqs
        )
        zeros = self._zero_draft_lengths_cache.get(padded_num_reqs)
        if zeros is None:
            zeros = torch.zeros(
                padded_num_reqs, dtype=torch.int32, device=self.runner.device
            )
            self._zero_draft_lengths_cache[padded_num_reqs] = zeros
        return zeros

    def _build_draft_layer_metadata(
        self,
        attn_metadata: dict[str, AttentionMetadata],
    ) -> tuple[AttentionMetadata, ...]:
        """Per-layer metadata for the draft's KV write, in layer order.

        `attn_metadata` is the *target's* full per-layer dict, so a draft layer
        that fails to resolve must raise rather than fall back to an arbitrary
        entry: the first entry is whichever layer hashed first — plausibly a
        mamba group, whose block table indexes a different block size. Writing
        the draft's KV through it would corrupt the cache silently instead of
        failing.
        """
        if hasattr(self.draft_model, "get_draft_kv_cache_layer_names"):
            layer_names = self.draft_model.get_draft_kv_cache_layer_names()
        else:
            layer_names = [
                getattr(layer.self_attn.attn, "layer_name", None)
                for layer in self.draft_model.model.layers
            ]
        draft_md_list = []
        for layer_name in layer_names:
            md = attn_metadata.get(layer_name) if layer_name else None
            if md is None:
                raise RuntimeError(
                    f"No attention metadata for draft layer {layer_name!r}; "
                    "the draft layers must own a KV cache group of their own."
                )
            draft_md_list.append(md)
        return tuple(draft_md_list)

    def _update_draft_kv_cache_from_target(self, chunk) -> None:
        """
        Projects the target model's hidden states into the draft model's KV cache.

        This allows the draft model to maintain context without needing to run a full
        forward pass on the tokens that were just verified by the target model.
        It applies RoPE to the target's hidden states and scatters the tokens directly
        into the draft KV cache.
        """
        if chunk.aux_hidden_states is None:
            return

        target_hidden = chunk.aux_hidden_states
        aux_hidden = (
            target_hidden[0]
            if isinstance(target_hidden, (list, tuple))
            else target_hidden
        )
        num_tokens = aux_hidden.shape[0]

        positions = chunk.position_ids
        if positions.dim() > 1:
            positions = positions[0]
        positions = positions[:num_tokens]
        draft_md_tuple = self._build_draft_layer_metadata(chunk.attn_metadata)

        with set_vllm_model_wrapper_context(
            mesh=self.runner.mesh, vllm_config=self.vllm_config
        ):
            self._tpu_precompute_and_update_kv_cache(
                target_hidden, positions, draft_md_tuple
            )

    @torch.no_grad()
    def propose(
        self,
        sampled_token_ids: list[list[int]],
        discard_sampled_tokens_req_indices: list[int],
        num_rejected_tokens_np: np.ndarray | None,
        scheduler_output,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: torch.Tensor | None = None,
    ) -> list[list[int]] | torch.Tensor:
        """Main execution step to propose draft tokens.

        `sampled_token_ids`, `discard_sampled_tokens_req_indices`,
        `num_rejected_tokens_np` and `scheduler_output` are unused here: this
        drafter is strictly device-fused and seeds from `next_tokens_per_chunk`
        or `device_seed`. They stay in the signature because
        `SpeculativeDecodingManager` calls `propose` polymorphically across
        this and `Eagle3Proposer`, which does read them.
        """
        runner = self.runner
        num_reqs = runner.input_batch.num_reqs
        if num_reqs == 0:
            return []

        chunks = self.draft_chunks
        if not chunks:
            raise RuntimeError("DFlash propose called but no draft chunks captured.")

        block_size = self.block_size

        # EP-DP lockstep: a rank whose step produced fewer chunks than the
        # DP-group max must pad each draft phase to the group's chunk count —
        # collectives pair across ranks by emission order, and propose emits
        # ALL KV updates before ALL draft forwards, so the padding must
        # follow the same phase structure (never interleave per chunk).
        extra_chunks = 0
        if self._dp_lockstep_sharded():
            extra_chunks = max(0, runner._dp_step_num_chunks - len(chunks))

        # 1. Hoist all KV cache updates across chunks to the top of propose
        for chunk in chunks:
            self._update_draft_kv_cache_from_target(chunk)
        for _ in range(extra_chunks):
            self._dp_dummy_kv_update()

        # 2. Process each chunk
        draft_logits_per_chunk = []
        for i, chunk in enumerate(chunks):
            next_tokens_device = (
                next_tokens_per_chunk[i] if next_tokens_per_chunk else None
            )
            # Slice seed tokens for this chunk's requests.
            chunk_seed = (
                device_seed[chunk.start_index : chunk.start_index + chunk.num_reqs]
                if device_seed is not None
                else None
            )
            input_ids, positions, seq_lens = self._prepare_dflash_inputs(
                chunk, next_tokens_device=next_tokens_device, device_seed=chunk_seed
            )
            padded_len = input_ids.shape[0]
            draft_attn_metadata = self._build_draft_attn_metadata(
                chunk, padded_len, seq_lens
            )

            with (
                # Precomputed so set_forward_context does not run its
                # own cross-DP CPU all_reduce per draft forward; must
                # pair exactly with the idle replay's calls.
                set_forward_context(
                    draft_attn_metadata,
                    self.vllm_config,
                    num_tokens=padded_len,
                    num_tokens_across_dp=runner._dp_num_tokens_across_dp(padded_len),
                ),
                set_vllm_model_wrapper_context(
                    mesh=self.runner.mesh, vllm_config=self.vllm_config
                ),
            ):
                draft_tokens_chunk, _ = self._dflash_forward_and_sample(
                    input_ids, positions, block_size
                )

            draft_logits_per_chunk.append(draft_tokens_chunk)
        for _ in range(extra_chunks):
            self._dp_dummy_forward()

        # 3. Extract K tokens from logits
        if return_device:
            if len(draft_logits_per_chunk) == 1:
                return draft_logits_per_chunk[0][: chunks[0].num_reqs]
            sliced_chunks = [
                logits
                if logits.shape[0] == chunk.num_reqs
                else logits[: chunk.num_reqs]
                for logits, chunk in zip(draft_logits_per_chunk, chunks)
            ]
            return torch.cat(sliced_chunks, dim=0)

        # Single synchronization and transfer to the host (return_device=False)
        # We transfer the ENTIRE padded tensor to CPU to avoid dynamic-shape recompilations
        draft_tokens_list = []
        for logits, chunk in zip(draft_logits_per_chunk, chunks):
            logits_host = logits.cpu().tolist()
            draft_tokens_list.extend(logits_host[: chunk.num_reqs])

        return draft_tokens_list

    @torch.no_grad()
    def precompile(self) -> None:
        """Warm up the DFlash draft model across all padding buckets.
        Called by tpu_runner.py during _dummy_run.
        """
        if self.draft_model is None:
            return

        logger.info("Precompiling DFlash draft model buckets...")

        runner = self.runner
        block_size = self.block_size
        # A most-model-len chunk holds up to num_reqs_most_model_len
        # requests (> num_reqs_max_model_len), and under EP-DP lockstep the
        # coordinated bucket is sized from the DP-wide max chunk request
        # count — so warm draft forwards up to the larger bound, or the
        # first such step compiles XLA mid-serving.
        max_chunk_reqs = max(
            runner.num_reqs_max_model_len, runner.num_reqs_most_model_len or 0
        )
        max_draft_tokens = self._get_padded_len(max_chunk_reqs * block_size)

        with runner._precompile_timed("drafter first pass"):
            for num_tokens in runner.num_tokens_paddings:
                # Precompile KV Cache Update Graph (used in both prefill and decode)
                self._dummy_precompute_and_update_kv_cache(
                    num_tokens=num_tokens,
                    num_reqs=runner.num_reqs_max_model_len,
                    num_blocks=runner.max_num_blocks_per_req,
                    sync=True,
                )
                if runner.most_model_len is not None:
                    self._dummy_precompute_and_update_kv_cache(
                        num_tokens=num_tokens,
                        num_reqs=runner.num_reqs_most_model_len,
                        num_blocks=runner.num_blocks_per_most_len_req,
                        sync=True,
                    )

                # Context-KV updates allow short chunks; draft forward needs a full block.
                if not block_size <= num_tokens <= max_draft_tokens:
                    continue

                self._dummy_draft_forward(
                    num_tokens=num_tokens, use_max_model_len=True, sync=True
                )
                if runner.most_model_len is not None:
                    self._dummy_draft_forward(
                        num_tokens=num_tokens, use_max_model_len=False, sync=True
                    )

    def _dummy_kv_update_inputs(
        self,
        num_tokens: int,
        num_reqs: int,
        num_blocks: int,
    ) -> tuple[list[torch.Tensor], AttentionMetadata]:
        """Cached `(aux hidden list, metadata)` for one KV-update shape;
        `(num_reqs, num_blocks)` is the target chunk's block-table geometry.
        """
        key = (num_tokens, num_reqs, num_blocks)
        entry = self._dummy_kv_input_cache.get(key)
        if entry is not None:
            return entry

        runner = self.runner
        # One [num_tokens, target_hidden] tensor per tagged target layer,
        # matching what the runner hands the real update.
        target_hidden = self.vllm_config.model_config.get_hidden_size()
        dtype = self.draft_model.model.embed_tokens.weight.dtype
        dummy_hidden = [
            torch.zeros((num_tokens, target_hidden), dtype=dtype, device=runner.device)
            for _ in self._target_layer_ids
        ]
        dummy_positions = torch.zeros(
            num_tokens, dtype=torch.int32, device=runner.device
        )

        block_tables = torch.zeros(
            (num_reqs * num_blocks,), dtype=torch.int32, device=runner.device
        )
        seq_lens = torch.zeros((num_reqs,), dtype=torch.int32, device=runner.device)
        seq_lens[0] = num_tokens

        query_start_loc = torch.zeros(
            (num_reqs + 1,), dtype=torch.int32, device=runner.device
        )
        query_start_loc[1:] = num_tokens

        request_distribution = torch.tensor(
            [0, 0, num_reqs], dtype=torch.int32, device=runner.device
        )

        dummy_attn_metadata = AttentionMetadata(
            input_positions=dummy_positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )
        entry = (dummy_hidden, dummy_attn_metadata)
        self._dummy_kv_input_cache[key] = entry
        return entry

    def _dummy_precompute_and_update_kv_cache(
        self,
        num_tokens: int,
        num_reqs: int,
        num_blocks: int,
        sync: bool,
    ) -> None:
        """One KV-update pass on constant dummies; `sync` waits on the host
        (precompile), otherwise the program is only enqueued (lockstep)."""
        runner = self.runner
        dummy_hidden, dummy_attn_metadata = self._dummy_kv_update_inputs(
            num_tokens, num_reqs, num_blocks
        )
        dummy_positions = dummy_attn_metadata.input_positions

        with set_vllm_model_wrapper_context(
            mesh=runner.mesh, vllm_config=self.vllm_config
        ):
            out = self._tpu_precompute_and_update_kv_cache(
                dummy_hidden,
                dummy_positions,
                tuple([dummy_attn_metadata] * len(self.draft_model.model.layers)),
            )
            synchronize_tensors(out, wait=sync)

    def _dummy_draft_forward(
        self,
        num_tokens: int,
        use_max_model_len: bool,
        sync: bool,
    ) -> None:
        """One draft forward on constant dummies; `sync` as in
        `_dummy_precompute_and_update_kv_cache`."""
        runner = self.runner
        block_size = self.block_size

        if num_tokens < block_size:
            raise ValueError(
                f"DFlash draft-forward bucket {num_tokens} is smaller than "
                f"its query block size {block_size}"
            )

        input_ids = torch.zeros((num_tokens), dtype=torch.int32).to(runner.device)
        positions = torch.zeros(num_tokens, dtype=torch.int32).to(runner.device)

        # Map actual request count based on block allocations; the guard
        # above makes this >= 1.
        actual_num_reqs = num_tokens // block_size

        num_tokens_per_req = num_tokens // actual_num_reqs
        query_lens = [num_tokens_per_req] * actual_num_reqs
        query_start_loc = torch.cumsum(
            torch.tensor([0] + query_lens, dtype=torch.int32), dim=0, dtype=torch.int32
        ).to(runner.device)

        seq_lens = (
            torch.ones((actual_num_reqs,), dtype=torch.int32).to(runner.device)
            * num_tokens_per_req
        )

        request_distribution = torch.tensor(
            [0, 0, actual_num_reqs], dtype=torch.int32
        ).to(runner.device)

        per_layer_attn_metadata = self._build_attn_metadata_for_draft(
            num_reqs=actual_num_reqs,
            start_index=0,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=positions,
            padded_num_reqs=actual_num_reqs,
        )

        with (
            # Same contract as propose(): precomputed DP metadata, no
            # hidden per-forward cross-DP all_reduce.
            set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_tokens,
                num_tokens_across_dp=runner._dp_num_tokens_across_dp(num_tokens),
            ),
            set_vllm_model_wrapper_context(
                mesh=self.runner.mesh, vllm_config=self.vllm_config
            ),
        ):
            draft_tokens_chunk, _ = self._dflash_forward_and_sample(
                input_ids, positions, block_size
            )
            synchronize_tensors(draft_tokens_chunk, wait=sync)

    def _dp_draft_bucket(self) -> int:
        """Draft-forward token bucket every lockstep rank runs this step."""
        coordinated_reqs = max(1, self.runner._dp_step_max_reqs)
        return self._get_padded_len(coordinated_reqs * self.block_size)

    @torch.no_grad()
    def _dp_dummy_kv_update(self) -> None:
        """One padding KV-update pass at the DP-coordinated target bucket.
        Runs on the decode path, so it must not wait on the host."""
        runner = self.runner
        bucket = runner._dp_target_bucket
        assert bucket is not None, "_dp_target_bucket unset in lockstep step"
        self._dummy_precompute_and_update_kv_cache(
            num_tokens=bucket,
            num_reqs=runner.num_reqs_max_model_len,
            num_blocks=runner.max_num_blocks_per_req,
            sync=False,
        )

    @torch.no_grad()
    def _dp_dummy_forward(self) -> None:
        """One padding draft forward at the step's coordinated draft bucket."""
        self._dummy_draft_forward(
            num_tokens=self._dp_draft_bucket(), use_max_model_len=True, sync=False
        )

    def _dp_lockstep_sharded(self) -> bool:
        """True when the draft emits cross-DP collectives under EP-DP
        lockstep: it is TP-sharded, or it carries expert-parallel MoE
        layers. A replicated, MoE-free draft is entirely rank-local and
        takes no part in the lockstep pairing (as in Eagle3Proposer)."""
        if not self.runner._dp_lockstep_enabled():
            return False
        if not self._draft_replicated:
            return True
        # Model-static: resolve once, not on every decode step.
        if self._draft_has_moe_cache is None:
            if self.draft_model is None:
                return False
            from vllm.model_executor.models.interfaces import is_mixture_of_experts

            self._draft_has_moe_cache = bool(is_mixture_of_experts(self.draft_model))
        return self._draft_has_moe_cache

    @torch.no_grad()
    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        """Replay a busy rank's draft collectives on an idle/padding rank
        for `num_chunks` padding chunks (EP-DP lockstep)."""
        if num_chunks <= 0 or self.draft_model is None:
            return
        if not self._dp_lockstep_sharded():
            return
        # Collectives pair across ranks by emission order, and propose()
        # emits ALL KV-updates before ALL draft forwards, so the replay
        # keeps that phase structure -- never interleaved per chunk.
        for _ in range(num_chunks):
            self._dp_dummy_kv_update()
        for _ in range(num_chunks):
            self._dp_dummy_forward()

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_precompute_and_update_kv_cache(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor,
        positions: torch.Tensor,
        draft_md_tuple: tuple[AttentionMetadata, ...],
    ) -> torch.Tensor:
        """
        Projects target model hidden states directly into the draft model's KV space.

        Instead of computing KV projections and RoPE iteratively per layer (which would
        create L tiny, sequential XLA graphs), we project to a single massive flat tensor
        for all layers at once using `_fused_kv_weight`. We then reshape and compute
        RoPE across all layers simultaneously in a single vectorized operation.
        """
        self_model = self.draft_model.model
        if isinstance(hidden_states, (list, tuple)):
            target_hidden = (
                hidden_states[0]
                if len(hidden_states) == 1
                else torch.cat(hidden_states, dim=-1)
            )
        else:
            target_hidden = hidden_states

        # Only opt into TPU-specific hooks; upstream methods with a similar
        # name may call GPU custom ops. Other drafts use the TPU path below.
        tpu_precompute = getattr(
            self.draft_model, "tpu_precompute_and_store_context_kv", None
        )
        if callable(tpu_precompute):
            return tpu_precompute(target_hidden, positions, draft_md_tuple)
        if getattr(type(self.draft_model), "owns_context_kv", False):
            return self.draft_model.precompute_and_store_context_kv(
                target_hidden, positions, draft_md_tuple
            )

        # Collapse the concatenated aux hidden states to the draft's width.
        # `combine_hidden_states` is the modern spelling; bare `fc` is what
        # drafters that predate it expose. The concatenated width is one
        # target hidden size per tagged layer.
        if hasattr(self.draft_model, "combine_hidden_states"):
            target_hidden = self.draft_model.combine_hidden_states(target_hidden)
        elif (fc_layer := getattr(self_model, "fc", None)) is not None:
            target_hidden = fc_layer(target_hidden)
            if isinstance(target_hidden, tuple):
                target_hidden = target_hidden[0]

        if hasattr(self_model, "hidden_norm"):
            target_hidden = self_model.hidden_norm(target_hidden)

        # Project target_hidden to K and V dims for all draft layers
        # all_kv_flat shape: [num_ctx, L * 2 * nkv * hd]
        all_kv_flat = F.linear(
            target_hidden, self_model._fused_kv_weight, self_model._fused_kv_bias
        )

        num_ctx = all_kv_flat.shape[0]
        L = len(self_model.layers)
        hd = self_model.layers[0].self_attn.head_dim
        nkv = self_model.layers[0].self_attn.num_kv_heads

        # Single contiguous copy that separates K/V and transposes to layer-major layout
        all_kv = (
            all_kv_flat.view(num_ctx, L, 2, nkv, hd).permute(2, 1, 0, 3, 4).contiguous()
        )
        all_k = all_kv[0]  # [L, num_ctx, nkv, hd]
        all_v = all_kv[1]  # [L, num_ctx, nkv, hd]

        if hasattr(self_model.layers[0].self_attn, "k_norm"):
            kw = [layer.self_attn.k_norm.weight for layer in self_model.layers]
            kw_stack = torch.stack(kw, dim=0).view(L, 1, 1, hd)
            eps = self_model.layers[0].self_attn.k_norm.variance_epsilon
            variance = all_k.pow(2).mean(-1, keepdim=True)
            all_k_normed = all_k * torch.rsqrt(variance + eps) * kw_stack
        else:
            all_k_normed = all_k

        # Apply rotary_emb once across all layers by flattening L and nkv
        all_k_flat2 = all_k_normed.view(L * num_ctx, nkv, hd)
        positions_repeated = positions.repeat(L)
        nq = self_model.layers[0].self_attn.num_heads
        dummy_q = torch.zeros(
            (L * num_ctx, nq, hd),
            device=target_hidden.device,
            dtype=target_hidden.dtype,
        )
        dummy_q_roped_flat, roped_k_flat = self_model.layers[0].self_attn.rotary_emb(
            positions_repeated, dummy_q, all_k_flat2
        )
        roped_k_all = roped_k_flat.view(L, num_ctx, nkv, hd)
        roped_q_all = dummy_q_roped_flat.view(L, num_ctx, nq, hd)

        # 2. update kv cache in the compiled graph so XLA can fuse the mutations
        for i, layer in enumerate(self.draft_model.model.layers):
            attn_obj = layer.self_attn.attn
            kv_cache = attn_obj.kv_cache
            attn_impl = attn_obj.impl

            # Use normal forward pass to run RPA v3 and update KV cache
            attn_impl.forward(
                layer=attn_obj,
                query=roped_q_all[i],
                key=roped_k_all[i],
                value=all_v[i],
                kv_cache=kv_cache,
                attn_metadata=draft_md_tuple[i],
            )

        return hidden_states

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compiled XLA graph for DFlash backbone forward pass and greedy sampling.

        Performs:
          1. Backbone model forward pass on input_ids and positions.
          2. Slices out the K mask tokens (skipping slot 0 base token).
          3. Computes draft logits and performs greedy argmax sampling in-graph.
        """
        # 1. Forward pass through draft model backbone
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        # 2. Slice off padding tokens added by static TPU bucketing
        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]

        # 3. Reshape and slice off slot 0 (base token), keeping only the K mask token slots
        hidden_reshaped = valid_hidden.view(
            padded_num_reqs, block_size, hidden.shape[-1]
        )
        draft_hidden = hidden_reshaped[:, 1:, :].reshape(-1, hidden.shape[-1])

        # 4. Compute draft logits and run greedy argmax sampling inside the XLA graph
        logits = self.draft_model.compute_logits(draft_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size - 1, logits.shape[-1])
        draft_tokens_chunk = logits_3d.argmax(dim=-1)

        return draft_tokens_chunk, hidden

    def _request_window_bounds(
        self,
        query_start_loc: torch.Tensor,
        position_ids: torch.Tensor,
        num_reqs: int,
        padded_num_reqs: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per padded request row: the clamped request index, a
        `[padded_num_reqs, 1]` int32 live-row mask, and the positions of the
        request's first and last query token this step. Padding rows alias
        the last live request; callers zero them with the mask. Traced
        inline by both compiled input builders."""
        req_indices = torch.arange(
            padded_num_reqs, dtype=torch.int32, device=query_start_loc.device
        )
        safe_req_indices = torch.clamp(req_indices, max=max(num_reqs - 1, 0))
        valid_mask = (req_indices < num_reqs).unsqueeze(1).to(torch.int32)
        start_positions = position_ids[query_start_loc[safe_req_indices]].unsqueeze(1)
        end_indices = torch.clamp(query_start_loc[safe_req_indices + 1] - 1, min=0)
        last_positions = position_ids[end_indices].unsqueeze(1)
        return safe_req_indices, valid_mask, start_positions, last_positions

    def _layout_draft_block(
        self,
        base_tokens: torch.Tensor,
        base_pos: torch.Tensor,
        block_size: int,
        padded_len: int,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lay out `[base, MASK x K]` per request from `base_tokens`
        (`[padded_num_reqs]`) and `base_pos` (`[padded_num_reqs, 1]`).

        Returns `(input_ids, positions, seq_lens)` at the static shapes the
        draft forward compiles against. Traced inline by both compiled
        input builders, so the layout, padding and clamps exist once.
        """
        device = base_tokens.device
        padded_num_reqs = base_tokens.shape[0]

        # Format input_ids with base tokens and MASK placeholders
        input_ids = torch.full(
            (padded_len,), mask_token_id, dtype=torch.int32, device=device
        )
        scatter_indices = torch.arange(
            0,
            padded_num_reqs * block_size,
            block_size,
            dtype=torch.int32,
            device=device,
        )
        input_ids.scatter_(0, scatter_indices, base_tokens)

        offsets = torch.arange(
            0, block_size, dtype=torch.int32, device=device
        ).unsqueeze(0)
        positions_unpadded = (base_pos + offsets).flatten()
        positions = F.pad(
            positions_unpadded, (0, padded_len - positions_unpadded.shape[0]), value=0
        )
        # Clamp to max_model_len to prevent out-of-bounds RoPE and kernel indexing.
        positions = torch.clamp(positions, max=self._max_model_len - 1)
        seq_lens = torch.clamp(
            base_pos.squeeze(1) + block_size, max=self._max_model_len
        )
        return input_ids, positions, seq_lens

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_build_dflash_inputs_next_tokens(
        self,
        next_tokens_device: torch.Tensor,
        query_start_loc: torch.Tensor,
        position_ids: torch.Tensor,
        draft_lengths: torch.Tensor,
        num_reqs: int,
        block_size: int,
        padded_len: int,
        mask_token_id: int,
        invalid_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compiled XLA graph that builds the drafter's inputs directly from the target model's output.

        Args:
            next_tokens_device: Tensor of shape [num_reqs, K+1] containing the target
                model's accepted tokens for this step. Padded with `invalid_token_id`.
            query_start_loc: 1D Tensor containing starting query locations per request.
            position_ids: 1D Tensor containing position IDs.
            draft_lengths: Number of verified drafts per request (0 for prompt chunks).
            num_reqs: Number of active requests.
            block_size: K + 1 (1 base token + K draft mask slots).
            padded_len: Total length of the flattened tensors (padded_num_reqs * block_size).
            mask_token_id: The vocabulary ID for the MASK token.
            invalid_token_id: The ID denoting unused/padding tokens in next_tokens_device.

        Returns:
            input_ids: 1D Tensor of shape [padded_len] formatted as:
                [Base, MASK, MASK, ..., Base, MASK, MASK, ...]
            positions: 1D Tensor of shape [padded_len] containing the absolute positional encodings:
                [Pos, Pos+1, Pos+2, ..., Pos', Pos'+1, Pos'+2, ...]
            seq_lens: 1D Tensor of shape [padded_num_reqs] containing the new sequence lengths.
        """
        padded_num_reqs = padded_len // block_size

        # Pad target output tokens to static shape on-device
        next_tokens_device_padded = F.pad(
            next_tokens_device,
            (0, 0, 0, padded_num_reqs - next_tokens_device.shape[0]),
            value=invalid_token_id,
        ).to(torch.int32)

        (safe_req_indices, valid_mask, start_positions, last_positions) = (
            self._request_window_bounds(
                query_start_loc, position_ids, num_reqs, padded_num_reqs
            )
        )
        start_positions_padded = (start_positions * valid_mask).to(torch.int32)

        # Count accepted tokens per request and extract the base token
        num_valid = (next_tokens_device_padded != invalid_token_id).sum(
            dim=1, dtype=torch.int32
        )
        last_valid_col = torch.clamp(num_valid, min=1) - 1
        base_tokens = next_tokens_device_padded.gather(
            1, last_valid_col.unsqueeze(1)
        ).squeeze(1)
        # Replace invalid_token_id (-1) with mask_token_id for padding rows.
        base_tokens = torch.where(
            base_tokens == invalid_token_id,
            torch.full_like(base_tokens, mask_token_id),
            base_tokens,
        )
        num_accepted = num_valid.unsqueeze(1).to(torch.int32)

        # Anchor verify windows at start + num_accepted; anchor prompt chunks at last_pos + 1.
        num_draft = draft_lengths[safe_req_indices].unsqueeze(1)
        is_verify_window = (num_draft > 0).to(torch.int32)
        verify_base = start_positions_padded + num_accepted
        prompt_base = ((last_positions + 1) * valid_mask).to(torch.int32)
        base_pos = is_verify_window * verify_base + (1 - is_verify_window) * prompt_base

        return self._layout_draft_block(
            base_tokens, base_pos, block_size, padded_len, mask_token_id
        )

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_build_dflash_inputs_seed(
        self,
        device_seed: torch.Tensor,
        query_start_loc: torch.Tensor,
        position_ids: torch.Tensor,
        num_reqs: int,
        block_size: int,
        padded_len: int,
        mask_token_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compiled XLA graph that builds the drafter's inputs from a 1D tensor of seed tokens.

        Args:
            device_seed: 1D Tensor of shape [num_reqs] containing base tokens.
            query_start_loc: 1D Tensor containing starting query locations per request.
            position_ids: 1D Tensor containing position IDs.
            num_reqs: Number of active requests.
            block_size: K + 1 (1 base token + K draft mask slots).
            padded_len: Total length of the flattened tensors (padded_num_reqs * block_size).
            mask_token_id: The vocabulary ID for the MASK token.

        Returns:
            input_ids: 1D Tensor of shape [padded_len] formatted as:
                [SeedToken, MASK, MASK, ..., SeedToken2, MASK, MASK, ...]
            positions: 1D Tensor of shape [padded_len] containing the absolute positional encodings:
                [Pos, Pos+1, Pos+2, ..., Pos', Pos'+1, Pos'+2, ...]
            seq_lens: 1D Tensor of shape [padded_num_reqs] containing the new sequence lengths.
        """
        padded_num_reqs = padded_len // block_size

        # Pad seed tokens to static shape on-device
        base_tokens = F.pad(
            device_seed, (0, padded_num_reqs - device_seed.shape[0]), value=0
        ).to(torch.int32)

        # Anchor one past the last prompt token in the chunk.
        _, valid_mask, _, last_positions = self._request_window_bounds(
            query_start_loc, position_ids, num_reqs, padded_num_reqs
        )
        base_pos = ((last_positions + 1) * valid_mask).to(torch.int32)

        return self._layout_draft_block(
            base_tokens, base_pos, block_size, padded_len, mask_token_id
        )
