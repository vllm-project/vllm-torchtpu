# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch_tpu._internal import sync
from vllm.compilation.backends import set_model_tag
from vllm.config import (VllmConfig, get_layers_from_vllm_config,
                         set_current_vllm_config)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader

from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilderContext)
from vllm_torchtpu.layers.vllm.attention import PallasAttentionBackendImpl
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.runner.tpu_runner_async_output import INVALID_TOKEN_ID
from vllm_torchtpu.spec_decode.utils import (DraftChunkInputs,
                                             _force_draft_tp1,
                                             maybe_share_embeddings,
                                             maybe_share_lm_head)

if TYPE_CHECKING:
    from vllm_torchtpu.runner.tpu_runner import TPUModelRunner

logger = init_logger(__name__)


class DFlashProposer:
    """DFlash draft proposer for TPU.

    Implements a parallel block-level speculative decoding framework.
    """

    def __init__(
        self,
        runner: "TPUModelRunner",
        vllm_config: VllmConfig,
    ):
        self.runner = runner
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        # Read DFlash specific configurations
        hf_config = self.speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}
        self.dflash_config = hf_dict.get("dflash_config", {})
        self.mask_token_id = self.dflash_config.get("mask_token_id")
        assert self.mask_token_id is not None, (
            "DFlash requires `mask_token_id` to be defined in the draft model's config.json "
            "under the `dflash_config` dictionary.")

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
                f"(== target tensor_parallel_size, sharded draft).")
        self._draft_replicated = (draft_tp == 1)
        logger.info(
            "DFlash draft parallelism: %s (draft_tp=%s).",
            "REPLICATED (tp=1)" if self._draft_replicated else "SHARDED",
            draft_tp)

        self.draft_model = None
        self._draft_attn_layer_names: set[str] | None = None
        self.draft_chunks: list[DraftChunkInputs] | None = None
        self.draft_lm_head = None
        self._static_attn_tensors_cache: dict[int, tuple[torch.Tensor,
                                                         torch.Tensor]] = {}

    def _get_static_attn_tensors(
            self, padded_num_reqs: int,
            block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch or create persistent static TPU tensors for draft attention metadata."""
        if padded_num_reqs not in self._static_attn_tensors_cache:
            qsl = torch.arange(padded_num_reqs + 1,
                               dtype=torch.int32,
                               device=self.runner.device) * block_size
            rd = torch.tensor([0, 0, padded_num_reqs],
                              dtype=torch.int32,
                              device=self.runner.device)
            self._static_attn_tensors_cache[padded_num_reqs] = (qsl, rd)
        return self._static_attn_tensors_cache[padded_num_reqs]

    def load_model(self, target_model) -> None:
        """Load the draft model, share embeddings, and tag target layers."""
        # Tag the target model to extract aux hidden states
        target_layers = [
            idx + 1 for idx in self.dflash_config.get("target_layer_ids", [])
        ]
        assert target_layers is not None, "target_layer_ids must be specified"
        if hasattr(target_model, "set_aux_hidden_state_layers"):
            target_model.set_aux_hidden_state_layers(tuple(target_layers))
        elif hasattr(target_model, "model") and hasattr(
                target_model.model, "_set_aux_hidden_state_layers"):
            target_model.model._set_aux_hidden_state_layers(
                tuple(target_layers))
        else:
            raise RuntimeError(
                "Target model does not support _set_aux_hidden_state_layers")
        logger.info(
            "Tagged target model for DFlash auxiliary hidden states extraction."
        )

        # Snapshot the target's attention layer names before the draft model
        # adds its own; diff after load gives us the draft layer names.
        target_attn_layer_names = set(
            get_layers_from_vllm_config(self.vllm_config,
                                        AttentionLayerBase).keys())

        self._load_draft_model()

        all_attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                      AttentionLayerBase)
        self._draft_attn_layer_names = (set(all_attn_layers.keys()) -
                                        target_attn_layer_names)

        # Share Embeddings and LM Head
        maybe_share_embeddings(self.draft_model,
                               target_model,
                               self._draft_replicated,
                               force_share=True)
        maybe_share_lm_head(self.draft_model,
                            target_model,
                            self._draft_replicated,
                            force_share=True)

        # Configure draft attention kernels for DFlash decoding
        with set_vllm_model_wrapper_context(mesh=self.runner.mesh):
            for i, layer in enumerate(self.draft_model.model.layers):
                attn_impl = layer.self_attn.attn.impl
                if isinstance(attn_impl, PallasAttentionBackendImpl):
                    attn_impl.layer_idx = i
                    # DFlash uses non-causal attention across its parallel mask tokens
                    attn_impl.use_causal_mask = False
                    attn_impl.initialize_kernel(layer.self_attn.attn)

    def _load_draft_model(self) -> None:
        logger.info("Loading DFlash draft model...")
        model_loader = get_model_loader(self.vllm_config.load_config)
        draft_tp1_ctx = (_force_draft_tp1() if self._draft_replicated else
                         contextlib.nullcontext())
        with set_model_tag("dflash_head"), set_vllm_model_wrapper_context(
                mesh=self.runner.mesh), set_current_vllm_config(
                    self.vllm_config), draft_tp1_ctx:
            self.draft_model = model_loader.load_model(
                vllm_config=self.vllm_config,
                model_config=self.speculative_config.draft_model_config,
            )

    def _get_padded_len(self, target_len: int) -> int:
        for p in self.runner.num_tokens_paddings:
            if p >= target_len:
                return p
        return target_len

    def _build_draft_attn_metadata(
        self,
        chunk: DraftChunkInputs,
        num_tokens_padded: int,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> dict:
        """
        Construct attention metadata for DFlash draft forward.
        Scales query boundaries and request token distribution to block_size.
        """
        runner = self.runner
        chunk_ctx = chunk.attn_ctx
        K = self.speculative_config.num_speculative_tokens
        block_size = K + 1

        padded_num_reqs = num_tokens_padded // block_size
        if padded_num_reqs == 0:
            padded_num_reqs = 1

        query_start_loc, request_distribution = self._get_static_attn_tensors(
            padded_num_reqs, block_size)

        saved_ctx = runner._attn_metadata_builder_ctx
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=padded_num_reqs,
            start_index=chunk.start_index,
            use_max_model_len=chunk_ctx.use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )
        slot_mappings_dict = runner.empty_slot_mappings

        try:
            per_layer_attn_metadata, _ = runner._build_attention_metadata(
                num_tokens=num_tokens_padded,
                num_reqs=padded_num_reqs,
                max_query_len=num_tokens_padded // padded_num_reqs,
                num_tokens_padded=num_tokens_padded,
                num_reqs_padded=padded_num_reqs,
                slot_mappings=slot_mappings_dict,
            )
        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

        return {
            name: md
            for name, md in per_layer_attn_metadata.items()
            if name in self._draft_attn_layer_names
        }

    def _prepare_dflash_inputs(
        self,
        chunk: DraftChunkInputs,
        sampled_token_ids: list[list[int]],
        num_rejected_tokens_np: np.ndarray | None,
        discard_sampled_tokens_req_indices: list[int] | None = None,
        scheduler_output=None,
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
        K = self.speculative_config.num_speculative_tokens
        block_size = K + 1
        total_draft_tokens = num_reqs * block_size

        # Round up to the nearest static padded length to avoid XLA recompilation
        padded_len = self._get_padded_len(total_draft_tokens)

        padded_num_reqs = padded_len // block_size
        if padded_num_reqs == 0:
            padded_num_reqs = 1

        query_start_loc_device = chunk.attn_ctx.query_start_loc
        position_ids_device = chunk.position_ids

        # Route to the appropriate pre-compiled XLA Graph
        if next_tokens_device is not None:
            input_ids, positions, seq_lens = self._tpu_build_dflash_inputs_next_tokens(
                next_tokens_device,
                query_start_loc_device,
                position_ids_device,
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
                "This drafter is designed strictly for the fused device path.")

        return input_ids, positions, seq_lens

    def _update_draft_kv_cache_from_target(self,
                                           chunk,
                                           num_rejected_tokens_np=None
                                           ) -> None:
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
        aux_hidden = target_hidden[0] if isinstance(target_hidden,
                                                    (list,
                                                     tuple)) else target_hidden
        num_tokens = aux_hidden.shape[0]

        positions = chunk.position_ids[:num_tokens]

        draft_attn_metadata = next(iter(chunk.attn_metadata.values()))

        with set_vllm_model_wrapper_context(mesh=self.runner.mesh):
            kv_caches = [
                layer.self_attn.attn.kv_cache
                for layer in self.draft_model.model.layers
            ]
            _, new_kv_caches = self._tpu_precompute_and_update_kv_cache(
                target_hidden, positions, draft_attn_metadata, kv_caches)
            for layer, new_kv in zip(self.draft_model.model.layers,
                                     new_kv_caches):
                layer.self_attn.attn.kv_cache = new_kv

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
        """Main execution step to propose draft tokens."""
        runner = self.runner
        num_reqs = runner.input_batch.num_reqs
        if num_reqs == 0:
            return []

        chunks = self.draft_chunks
        if not chunks:
            raise RuntimeError(
                "DFlash propose called but no draft chunks captured.")

        K = self.speculative_config.num_speculative_tokens
        block_size = K + 1

        # 1. Hoist all KV cache updates across chunks to the top of propose
        for chunk in chunks:
            with set_vllm_model_wrapper_context(mesh=None):
                self._update_draft_kv_cache_from_target(
                    chunk, num_rejected_tokens_np)

        # 2. Process each chunk
        draft_logits_per_chunk = []
        for i, chunk in enumerate(chunks):
            next_tokens_device = next_tokens_per_chunk[
                i] if next_tokens_per_chunk else None
            input_ids, positions, seq_lens = self._prepare_dflash_inputs(
                chunk,
                sampled_token_ids,
                num_rejected_tokens_np,
                discard_sampled_tokens_req_indices,
                scheduler_output,
                next_tokens_device=next_tokens_device,
                device_seed=device_seed)
            padded_len = input_ids.shape[0]
            draft_attn_metadata = self._build_draft_attn_metadata(
                chunk, padded_len, positions, seq_lens)

            with (
                    set_forward_context(draft_attn_metadata, self.vllm_config,
                                        0),
                    set_vllm_model_wrapper_context(mesh=self.runner.mesh),
            ):
                draft_tokens_chunk, hidden = self._dflash_forward_and_sample(
                    input_ids, positions, block_size)

            draft_logits_per_chunk.append(draft_tokens_chunk)

        # 3. Extract K tokens from logits
        if return_device:
            if len(draft_logits_per_chunk) == 1:
                return draft_logits_per_chunk[0][:chunks[0].num_reqs]
            sliced_chunks = [
                logits if logits.shape[0] == chunk.num_reqs else
                logits[:chunk.num_reqs]
                for logits, chunk in zip(draft_logits_per_chunk, chunks)
            ]
            return torch.cat(sliced_chunks, dim=0)

        # Single synchronization and transfer to the host (return_device=False)
        # We transfer the ENTIRE padded tensor to CPU to avoid dynamic-shape recompilations
        draft_tokens_list = []
        for logits, chunk in zip(draft_logits_per_chunk, chunks):
            logits_host = logits.cpu().tolist()
            draft_tokens_list.extend(logits_host[:chunk.num_reqs])

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
        K = self.speculative_config.num_speculative_tokens
        block_size = K + 1
        max_draft_tokens = runner.num_reqs_max_model_len * block_size  # e.g., 32 * 16 = 512

        with runner._precompile_timed("drafter first pass"):
            fc_layer = getattr(self.draft_model.model, "fc", None)
            if fc_layer is not None:
                hidden_dim = fc_layer.weight.shape[1]
            else:
                hidden_dim = self.vllm_config.model_config.get_hidden_size()
            dtype = self.draft_model.model.embed_tokens.weight.dtype
            for num_tokens in runner.num_tokens_paddings:
                # Precompile KV Cache Update Graph (used in both prefill and decode)
                self._dummy_precompute_and_update_kv_cache(
                    num_tokens=num_tokens,
                    hidden_dim=hidden_dim,
                    dtype=dtype,
                    num_reqs=runner.num_reqs_max_model_len,
                )

                # The draft model only runs speculation on decode steps, so it never processes
                # more tokens than the max batch size * speculation block size.
                if num_tokens > max_draft_tokens:
                    continue

                self._dummy_draft_forward(
                    num_tokens=num_tokens,
                    num_reqs=runner.num_reqs_max_model_len,
                    use_max_model_len=True,
                )
                if runner.most_model_len is not None:
                    self._dummy_draft_forward(
                        num_tokens=num_tokens,
                        num_reqs=runner.num_reqs_most_model_len,
                        use_max_model_len=False,
                    )

    def clear_keep_alives(self) -> None:
        """Explicitly clear references to dummy outputs used to prevent DCE, preventing cross-batch memory leaks."""
        self._keep_alive_outputs_list = None

    def _dummy_precompute_and_update_kv_cache(
        self,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        num_reqs: int,
    ) -> None:
        runner = self.runner
        dummy_hidden = torch.zeros((num_tokens, hidden_dim),
                                   dtype=dtype,
                                   device=runner.device)
        dummy_positions = torch.zeros(num_tokens,
                                      dtype=torch.int32,
                                      device=runner.device)

        # Build a safe dummy AttentionMetadata matching the exact padded_num_reqs shape
        num_blocks = runner.max_num_blocks_per_req
        block_tables = torch.zeros((num_reqs * num_blocks, ),
                                   dtype=torch.int32,
                                   device=runner.device)
        seq_lens = torch.zeros((num_reqs, ),
                               dtype=torch.int32,
                               device=runner.device)
        seq_lens[0] = num_tokens

        query_start_loc = torch.zeros((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device=runner.device)
        query_start_loc[1:] = num_tokens

        request_distribution = torch.tensor([0, 0, num_reqs],
                                            dtype=torch.int32,
                                            device=runner.device)

        dummy_attn_metadata = AttentionMetadata(
            input_positions=dummy_positions,
            block_tables=block_tables,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
        )

        with set_vllm_model_wrapper_context(mesh=runner.mesh):
            kv_caches = [
                layer.self_attn.attn.kv_cache
                for layer in self.draft_model.model.layers
            ]
            _, dummy_outputs = self._tpu_precompute_and_update_kv_cache(
                dummy_hidden, dummy_positions, dummy_attn_metadata, kv_caches)
            for out in dummy_outputs:
                sync.synchronize(out, wait=True)

    def _dummy_draft_forward(
        self,
        num_tokens: int,
        num_reqs: int,
        use_max_model_len: bool,
    ) -> None:
        runner = self.runner
        K = self.speculative_config.num_speculative_tokens
        block_size = K + 1

        input_ids = torch.zeros((num_tokens),
                                dtype=torch.int32).to(runner.device)
        positions = torch.zeros(num_tokens,
                                dtype=torch.int32).to(runner.device)

        # Map actual request count based on block allocations
        actual_num_reqs = num_tokens // block_size
        if actual_num_reqs == 0:
            actual_num_reqs = 1

        num_tokens_per_req = num_tokens // actual_num_reqs
        query_lens = [num_tokens_per_req] * actual_num_reqs
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(runner.device)

        seq_lens = torch.ones((actual_num_reqs, ), dtype=torch.int32).to(
            runner.device) * num_tokens_per_req

        request_distribution = torch.tensor([0, 0, actual_num_reqs],
                                            dtype=torch.int32).to(
                                                runner.device)

        saved_ctx = getattr(runner, "_attn_metadata_builder_ctx", None)
        runner._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
            num_reqs=actual_num_reqs,
            start_index=0,
            use_max_model_len=use_max_model_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            request_distribution=request_distribution,
            position_ids_override=positions,
        )
        try:
            # Unit tests or dummy runners may not have kv_cache_config initialized.
            # In this case, we manually construct a fake AttentionMetadata so dummy runs don't crash.
            if getattr(runner, "kv_cache_config", None) is None:
                num_blocks = runner.max_num_blocks_per_req if use_max_model_len else runner.num_blocks_per_most_len_req
                block_tables = torch.zeros((actual_num_reqs * num_blocks, ),
                                           dtype=torch.int32).to(runner.device)
                attn_metadata = AttentionMetadata(
                    input_positions=positions,
                    block_tables=block_tables,
                    seq_lens=seq_lens,
                    query_start_loc=query_start_loc,
                    request_distribution=request_distribution,
                )
                per_layer_attn_metadata = {
                    layer_name: attn_metadata
                    for layer_name in runner._attn_layer_names
                }
            else:
                slot_mappings = runner.empty_slot_mappings
                per_layer_attn_metadata, _ = runner._build_attention_metadata(
                    num_tokens=num_tokens,
                    num_reqs=actual_num_reqs,
                    max_query_len=num_tokens_per_req,
                    num_tokens_padded=num_tokens,
                    num_reqs_padded=actual_num_reqs,
                    slot_mappings=slot_mappings,
                )
            with (
                    set_forward_context(per_layer_attn_metadata,
                                        self.vllm_config, 0),
                    set_vllm_model_wrapper_context(mesh=self.runner.mesh, ),
            ):
                draft_tokens_chunk, hidden = self._dflash_forward_and_sample(
                    input_ids, positions, block_size)
                sync.synchronize(draft_tokens_chunk, wait=True)

        finally:
            runner._attn_metadata_builder_ctx = saved_ctx

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_precompute_and_update_kv_cache(
        self,
        hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor]
        | torch.Tensor,
        positions: torch.Tensor,
        draft_attn_metadata: "AttentionMetadata",
        kv_caches: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Projects target model hidden states directly into the draft model's KV space.

        Instead of computing KV projections and RoPE iteratively per layer (which would
        create L tiny, sequential XLA graphs), we project to a single massive flat tensor
        for all layers at once using `_fused_kv_weight`. We then reshape and compute
        RoPE across all layers simultaneously in a single vectorized operation.
        """
        self_model = self.draft_model.model
        if isinstance(hidden_states, (list, tuple)):
            target_hidden = hidden_states[0] if len(
                hidden_states) == 1 else torch.cat(hidden_states, dim=-1)
        else:
            target_hidden = hidden_states

        if hasattr(self.draft_model, "combine_hidden_states"):
            target_hidden = self.draft_model.combine_hidden_states(
                target_hidden)
        elif fc_layer := getattr(self_model, "fc", None):
            target_hidden = fc_layer(target_hidden)
            if isinstance(target_hidden, tuple):
                target_hidden = target_hidden[0]

        if hasattr(self_model, "hidden_norm"):
            target_hidden = self_model.hidden_norm(target_hidden)

        # Project target_hidden to K and V dims for all draft layers
        # all_kv_flat shape: [num_ctx, L * 2 * nkv * hd]
        all_kv_flat = F.linear(target_hidden, self_model._fused_kv_weight,
                               self_model._fused_kv_bias)

        num_ctx = all_kv_flat.shape[0]
        L = len(self_model.layers)
        hd = self_model.layers[0].self_attn.head_dim
        nkv = self_model.layers[0].self_attn.num_kv_heads

        # Single contiguous copy that separates K/V and transposes to layer-major layout
        all_kv = all_kv_flat.view(num_ctx, L, 2, nkv,
                                  hd).permute(2, 1, 0, 3, 4).contiguous()
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
        dummy_q = torch.zeros((L * num_ctx, nq, hd),
                              device=target_hidden.device,
                              dtype=target_hidden.dtype)
        dummy_q_roped_flat, roped_k_flat = self_model.layers[
            0].self_attn.rotary_emb(positions_repeated, dummy_q, all_k_flat2)
        roped_k_all = roped_k_flat.view(L, num_ctx, nkv, hd)
        roped_q_all = dummy_q_roped_flat.view(L, num_ctx, nq, hd)

        new_kv_caches = []
        # 2. update kv cache in the compiled graph so XLA can fuse the mutations
        for i, layer in enumerate(self.draft_model.model.layers):
            kv_cache = kv_caches[i]
            attn_obj = layer.self_attn.attn
            attn_impl = attn_obj.impl

            # Use normal forward pass to run RPA v3 and update KV cache
            attn_impl.forward(
                layer=attn_obj,
                query=roped_q_all[i],
                key=roped_k_all[i],
                value=all_v[i],
                kv_cache=kv_cache,
                attn_metadata=draft_attn_metadata,
            )
            new_kv_caches.append(kv_cache)

        return hidden_states, new_kv_caches

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
        valid_hidden = hidden[:padded_num_reqs * block_size]

        # 3. Reshape and slice off slot 0 (base token), keeping only the K mask token slots
        hidden_reshaped = valid_hidden.view(padded_num_reqs, block_size,
                                            hidden.shape[-1])
        draft_hidden = hidden_reshaped[:, 1:, :].reshape(-1, hidden.shape[-1])

        # 4. Compute draft logits and run greedy argmax sampling inside the XLA graph
        logits = self.draft_model.compute_logits(draft_hidden)
        logits_3d = logits.view(padded_num_reqs, block_size - 1,
                                logits.shape[-1])
        draft_tokens_chunk = logits_3d.argmax(dim=-1)

        return draft_tokens_chunk, hidden

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _tpu_build_dflash_inputs_next_tokens(
        self,
        next_tokens_device: torch.Tensor,
        query_start_loc: torch.Tensor,
        position_ids: torch.Tensor,
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
            value=invalid_token_id).to(torch.int32)

        # Extract start positions on-device
        req_indices = torch.arange(padded_num_reqs,
                                   dtype=torch.int32,
                                   device=query_start_loc.device)
        safe_req_indices = torch.clamp(req_indices, max=max(num_reqs - 1, 0))
        start_indices = query_start_loc[safe_req_indices]
        start_positions = position_ids[start_indices].unsqueeze(1)
        valid_mask = (req_indices < num_reqs).unsqueeze(1).to(torch.int32)
        start_positions_padded = (start_positions * valid_mask).to(torch.int32)

        # Count accepted tokens per request and extract the base token
        num_valid = (next_tokens_device_padded
                     != invalid_token_id).sum(dim=1, dtype=torch.int32)
        last_valid_col = torch.clamp(num_valid, min=1) - 1
        base_tokens_dev_padded = next_tokens_device_padded.gather(
            1, last_valid_col.unsqueeze(1)).squeeze(1)
        num_accepted_dev_padded = num_valid.unsqueeze(1).to(torch.int32)

        # Format input_ids with base tokens and MASK placeholders
        input_ids = torch.full((padded_len, ),
                               mask_token_id,
                               dtype=torch.int32,
                               device=next_tokens_device_padded.device)
        scatter_indices = torch.arange(0,
                                       padded_num_reqs * block_size,
                                       block_size,
                                       dtype=torch.int32,
                                       device=next_tokens_device_padded.device)
        input_ids.scatter_(0, scatter_indices, base_tokens_dev_padded)

        # Compute positional encodings and sequence lengths
        base_pos = start_positions_padded + num_accepted_dev_padded
        offsets = torch.arange(
            0,
            block_size,
            dtype=torch.int32,
            device=next_tokens_device_padded.device).unsqueeze(0)
        positions_unpadded = (base_pos + offsets).flatten()
        positions = F.pad(positions_unpadded,
                          (0, padded_len - positions_unpadded.shape[0]),
                          value=0)
        seq_lens = base_pos.squeeze(1) + block_size

        return input_ids, positions, seq_lens

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
        device_seed_padded = F.pad(device_seed,
                                   (0, padded_num_reqs - device_seed.shape[0]),
                                   value=0).to(torch.int32)

        # Extract start positions on-device
        req_indices = torch.arange(padded_num_reqs,
                                   dtype=torch.int32,
                                   device=query_start_loc.device)
        safe_req_indices = torch.clamp(req_indices, max=max(num_reqs - 1, 0))
        start_indices = query_start_loc[safe_req_indices]
        start_positions = position_ids[start_indices].unsqueeze(1)
        valid_mask = (req_indices < num_reqs).unsqueeze(1).to(torch.int32)
        start_positions_padded = (start_positions * valid_mask).to(torch.int32)

        # Format input_ids with seed tokens and MASK placeholders
        input_ids = torch.full((padded_len, ),
                               mask_token_id,
                               dtype=torch.int32,
                               device=device_seed_padded.device)
        scatter_indices = torch.arange(0,
                                       padded_num_reqs * block_size,
                                       block_size,
                                       dtype=torch.int32,
                                       device=device_seed_padded.device)
        input_ids.scatter_(0, scatter_indices, device_seed_padded)

        # Compute positional encodings and sequence lengths
        base_pos = start_positions_padded
        offsets = torch.arange(0,
                               block_size,
                               dtype=torch.int32,
                               device=device_seed_padded.device).unsqueeze(0)
        positions_unpadded = (base_pos + offsets).flatten()
        positions = F.pad(positions_unpadded,
                          (0, padded_len - positions_unpadded.shape[0]),
                          value=0)
        seq_lens = base_pos.squeeze(1) + block_size

        return input_ids, positions, seq_lens
