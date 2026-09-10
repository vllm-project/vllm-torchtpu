# SPDX-License-Identifier: Apache-2.0
import contextlib
import os
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import torch
import vllm.envs as vllm_envs
from vllm.config import get_layers_from_vllm_config, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase

from vllm_torchtpu.compilation import shape_variants
from vllm_torchtpu.layers.common.attention_metadata import (
    AttentionMetadata, AttentionMetadataBuilderContext)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import \
    set_vllm_model_wrapper_context
from vllm_torchtpu.runner.input_preparation import \
    _get_sequence_layout_planner_for_runner
from vllm_torchtpu.runner.mm_encoder_manager import \
    maybe_create_mm_encoder_manager
from vllm_torchtpu.runner.padding import _get_padded_token_len
from vllm_torchtpu.utils import synchronize_device, synchronize_tensors

logger = init_logger(__name__)


class CompileWarmupMixin:
    """Ahead-of-time XLA graph precompilation."""

    @torch.no_grad()
    def _dummy_run(self,
                   num_tokens: int,
                   num_reqs: int,
                   num_blocks: int,
                   use_max_model_len: bool = True,
                   dp_lockstep: bool = False) -> None:
        kv_cache_initialized = self.kv_cache_config is not None

        if self.supports_mm_inputs:
            input_ids = None
            inputs_embeds = torch.zeros(
                (num_tokens, self.inputs_embeds_size),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            input_ids = torch.zeros((num_tokens),
                                    dtype=torch.int32).to(self.device)
            inputs_embeds = None
        actual_num_reqs = min(num_tokens, num_reqs)
        if self.uses_mrope:
            position_ids = torch.zeros((3, num_tokens),
                                       dtype=torch.int32).to(self.device)
        else:
            position_ids = torch.zeros(num_tokens,
                                       dtype=torch.int32).to(self.device)
        query_lens = ([1] * actual_num_reqs + [0] *
                      (num_reqs - actual_num_reqs))
        query_start_loc = torch.cumsum(torch.tensor([0] + query_lens,
                                                    dtype=torch.int32),
                                       dim=0,
                                       dtype=torch.int32).to(self.device)
        seq_lens = torch.ones((num_reqs, ), dtype=torch.int32).to(self.device)
        # V3: request_distribution = [decode_end, prefill_end, mixed_end].
        # Dummy runs use one scheduled token per active request, so model
        # them as pure decode to match the real single-chip path.
        request_distribution = torch.tensor(
            [actual_num_reqs, actual_num_reqs, actual_num_reqs],
            dtype=torch.int32).to(self.device)
        dummy_layout_plan = _get_sequence_layout_planner_for_runner(
            self).prepare_dummy(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                kv_cache_initialized=kv_cache_initialized,
            )

        if self.kv_cache_config is not None:
            # Dummy compact-mamba slot ids (all null slot 0): the dummy run
            # only traces shapes/HBM, so the recurrent state read/written is
            # never consumed. Shape must match the GDN op's max_reqs
            # (= seq_lens length = num_reqs).
            dummy_mamba_state_indices = (torch.zeros(
                (num_reqs, ), dtype=torch.int32).to(self.device)
                                         if self._has_mamba_state else None)
            # Match _prepare_inputs: the spec-decode mamba fields are set iff
            # the model has mamba layers and spec decoding is enabled, so the
            # dummy runs compile the same GDN program signature — including
            # the two distributions being views of one [6] base tensor (see
            # _combined_request_distribution_cpu for why).
            dummy_mamba_request_distribution = None
            if self.mamba_slot_read_offsets is not None:
                combined_device = torch.tensor([actual_num_reqs] * 6,
                                               dtype=torch.int32).to(
                                                   self.device)
                request_distribution = combined_device[0:3]
                dummy_mamba_request_distribution = combined_device[3:6]
            self._attn_metadata_builder_ctx = AttentionMetadataBuilderContext(
                num_reqs=num_reqs,
                start_index=0,
                use_max_model_len=use_max_model_len,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                position_ids_override=position_ids,
                mamba_state_indices=dummy_mamba_state_indices,
                mamba_slot_read_offsets=self.mamba_slot_read_offsets,
                mamba_ckpt_window=self._mamba_ckpt_window,
                mamba_request_distribution=dummy_mamba_request_distribution,
                sequence_layout_descriptor=dummy_layout_plan.descriptor,
            )
            slot_mappings = self.empty_slot_mappings
            per_layer_attn_metadata, _unused_spec_decode_common_attn_metadata = self._build_attention_metadata(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                max_query_len=1,
                num_tokens_padded=num_tokens,
                num_reqs_padded=num_reqs,
                slot_mappings=slot_mappings,
            )
        else:
            # Pre-init path (called before initialize_kv_cache): use the
            # caller-supplied num_blocks and a single shared metadata.
            if self._attn_layer_names is None:
                self._attn_layer_names = list(
                    get_layers_from_vllm_config(
                        self.vllm_config,
                        (AttentionLayerBase, MambaBase)).keys())
            block_tables = torch.zeros((num_reqs * num_blocks, ),
                                       dtype=torch.int32).to(self.device)
            attn_metadata = AttentionMetadata(
                input_positions=position_ids,
                block_tables=block_tables,
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                request_distribution=request_distribution,
                sequence_layout_kind=(dummy_layout_plan.descriptor.kind.value),
                sequence_layout_protocol=(
                    dummy_layout_plan.descriptor.protocol),
                sequence_layout_version=(dummy_layout_plan.descriptor.version),
            )

            per_layer_attn_metadata = {
                layer_name: attn_metadata
                for layer_name in self._attn_layer_names
            }

        padding_state = getattr(self, "_token_padding_state", None)
        if padding_state is not None:
            # Every row of a dummy step is padding.
            padding_state.update(0, num_tokens)
        with (
                self.maybe_select_dummy_loras(
                    self.lora_config, np.array([num_tokens], dtype=np.int32)),
                set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens if dp_lockstep else 0,
                    num_tokens_across_dp=self._dp_num_tokens_across_dp(
                        num_tokens) if dp_lockstep else None),
                set_vllm_model_wrapper_context(
                    mesh=self.mesh,
                    vllm_config=self.vllm_config,
                    kv_cache_bundle=self._kv_cache_bundle,
                ),
        ):
            out, _ = self.forward_model(input_ids=input_ids,
                                        positions=position_ids,
                                        inputs_embeds=inputs_embeds)
            if dp_lockstep:
                # Idle DP engines must issue the same TP logits collective as
                # busy engines before the next DP synchronization.
                _idx = torch.zeros(num_reqs,
                                   dtype=torch.int32,
                                   device=out.device)
                _ = self.compute_selected_logits(out, _idx)
            if not dp_lockstep:
                synchronize_tensors(out)
        self._hidden_states_dtype = out.dtype

    @contextlib.contextmanager
    def _precompile_timed(self, name: str):
        """Common boilerplate for each AOT bucket precompile pass."""
        logger.info("Compiling %s with different input shapes.", name)
        start = time.perf_counter()
        yield
        logger.info("Compilation finished in %.2f [secs].",
                    time.perf_counter() - start)
        self._update_num_xla_graphs(name)

    def _dummy_logits(self, num_reqs: int) -> torch.Tensor:
        return torch.zeros((num_reqs, self.vocab_size),
                           device=self.device,
                           dtype=self._hidden_states_dtype)

    def _precompile_compute_selected_logits(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_selected_logits"):
            for num_tokens in self.num_tokens_paddings:
                dummy_hidden = torch.zeros((num_tokens, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                for num_reqs in self.num_reqs_paddings:
                    indices = torch.zeros(num_reqs,
                                          dtype=torch.int32,
                                          device=self.device)
                    out = self.compute_selected_logits(dummy_hidden, indices)
                    synchronize_tensors(out)
                    logger.info("  -- num_tokens: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_compute_logits_from_hidden_states(self) -> None:
        hsize = self.model_config.get_hidden_size()
        with self._precompile_timed("compute_logits_from_hidden_states"):
            for num_reqs in self.num_reqs_paddings:
                dummy_hidden = torch.zeros((num_reqs, hsize),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                out = self.compute_logits_from_hidden_states(dummy_hidden)
                synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_structured_decoding(self) -> None:
        with self._precompile_timed("structured_decoding"):
            arange = self.structured_decoding_manager.structured_decode_arange
            for num_reqs in self.num_reqs_paddings:
                out = self.structured_decoding_manager.structured_decode(
                    self.structured_decoding_manager.
                    require_structured_out_cpu[:num_reqs].to(self.device),
                    self.structured_decoding_manager.
                    grammar_bitmask_cpu[:num_reqs].to(self.device),
                    self._dummy_logits(num_reqs),
                    arange,
                )
                synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)
            # Prefill-only PCP MTP K1 skips _precompile_rejection_sampler(), so
            # we skip it here as well.
            if self.speculative_config is None or self._pcp_mtp_k1_enabled:
                return
            # For spec decoding, target logits have padded_logits_length rows.
            max_target_rows = self.structured_decoding_manager.target_grammar_bitmask_cpu.shape[
                0]
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_target_rows:
                    break
                out = self.structured_decoding_manager.structured_decode(
                    self.structured_decoding_manager.
                    require_structured_out_target_cpu[:num_tokens].to(
                        self.device),
                    self.structured_decoding_manager.
                    target_grammar_bitmask_cpu[:num_tokens].to(self.device),
                    self._dummy_logits(num_tokens),
                    arange,
                )
                synchronize_tensors(out)
                logger.info("  -- target_logits_length: %d", num_tokens)

    def _precompile_sample_from_logits(self) -> None:
        with self._precompile_timed("sample_from_logits"):
            for num_reqs in self.num_reqs_paddings:
                dummy_logits = self._dummy_logits(num_reqs)
                dummy_temperatures = torch.ones(
                    (num_reqs, 1),
                    dtype=self._hidden_states_dtype,
                    device=self.device)
                dummy_u = torch.rand_like(dummy_logits)
                dummy_top_k = torch.zeros((num_reqs, 1),
                                          dtype=torch.int32,
                                          device=self.device)
                dummy_top_p = torch.ones((num_reqs, 1),
                                         dtype=torch.float32,
                                         device=self.device)
                for all_greedy in [False, True]:
                    out = self.sample_from_logits_func(dummy_logits,
                                                       dummy_temperatures,
                                                       dummy_u,
                                                       dummy_top_k,
                                                       dummy_top_p,
                                                       all_greedy=all_greedy)
                    synchronize_tensors(out)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_gather_logprobs(self) -> None:
        with self._precompile_timed("gather_logprobs"):
            for num_reqs in self.num_reqs_paddings:
                out = self.gather_logprobs(
                    self._dummy_logits(num_reqs),
                    torch.zeros((num_reqs, 1),
                                dtype=torch.int64).to(self.device),
                )
                synchronize_tensors(out.logprobs)
                logger.info("  -- num_seqs: %d", num_reqs)

    def _precompile_rejection_sampler(self) -> None:
        """Warm the spec-decode verify path so it doesn't recompile at runtime.

        The rejection sampler is ``@torch.compile(backend="tpu",
        dynamic=False)`` -> a fresh graph per input shape, and the
        ``logits[...]`` gathers around it are eager. None are covered by
        ``_precompile_sampling_subgraphs`` (non-spec sampling only) or
        ``drafter.precompile()`` (draft forward only), so they recompile every
        time the verify length (``padded_logits_length``) lands in a new
        num-tokens bucket as requests finish and the batch shrinks. Replay the
        verify inner body (``_sample_spec_verify_chunk``) at every
        ``(padded_logits_length, padded_num_reqs)`` bucket the runtime can
        produce so the real call is a cache hit. Shared sync + async.

        Both verify sub-branches are distinct ``dynamic=False`` graphs, so warm
        each: greedy (``do_sampling=False`` ->
        ``_greedy_rejection_sample_with_segment``, prelude
        ``spec_bonus_and_target_logits``) and non-greedy (``do_sampling=True``
        -> ``_random_rejection_sample_with_segment``, prelude
        ``spec_gather_bonus_and_target_logits``). Which one runs is decided per
        step by ``input_batch.all_greedy`` and ``grammar_output``, so both are
        reachable and must be pre-compiled here; the greedy + structured-output
        combination reuses graphs warmed by these two passes.
        """
        if not (self._is_async_drafter):
            return
        k = self.speculative_config.num_speculative_tokens
        # The verify chunk samples at most num_reqs*(K+1) positions; bound the
        # padded_logits_length sweep to the bucket that covers the full batch.
        # Use min here because num_reqs*(K+1) can exceed max_num_batched_tokens
        # (which is the largest bucket), and logits rows never exceed the largest
        # bucket.
        max_logits_len = _get_padded_token_len(
            self.num_tokens_paddings,
            min(self.max_num_reqs * (k + 1), self.num_tokens_paddings[-1]))
        with self._precompile_timed("rejection_sampler"):
            for num_tokens in self.num_tokens_paddings:
                if num_tokens > max_logits_len:
                    break
                # padded_logits_length-shaped inputs; dtypes mirror
                # get_spec_decode_metadata so the graph is byte-identical.
                dummy_logits = torch.zeros((num_tokens, self.vocab_size),
                                           device=self.device,
                                           dtype=self._hidden_states_dtype)
                draft_token_ids = torch.zeros(num_tokens,
                                              dtype=torch.int32,
                                              device=self.device)
                target_logits_indices = torch.zeros(num_tokens,
                                                    dtype=torch.int32,
                                                    device=self.device)
                segment_ids = torch.zeros(num_tokens,
                                          dtype=torch.int64,
                                          device=self.device)
                group_indices = torch.zeros(num_tokens,
                                            dtype=torch.int32,
                                            device=self.device)
                for num_reqs in self.num_reqs_paddings:
                    # padded_num_reqs-shaped inputs.
                    draft_lengths = torch.zeros(num_reqs,
                                                dtype=torch.int32,
                                                device=self.device)
                    bonus_logits_indices = torch.zeros(num_reqs,
                                                       dtype=torch.int32,
                                                       device=self.device)
                    warm_accept_u = torch.zeros(num_tokens,
                                                dtype=torch.float32,
                                                device=self.device)
                    # --- greedy verify path (do_sampling=False) ---
                    bonus_token_ids, target_logits_warm = (
                        self.spec_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices,
                            target_logits_indices))
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_warm,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                        accept_u=(warm_accept_u
                                  if self.rejection_sampler.synthetic_mode else
                                  None),
                    )
                    synchronize_tensors(out)
                    # --- non-greedy verify path (do_sampling=True) ---
                    _, target_logits_ng = (
                        self.spec_gather_bonus_and_target_logits(
                            dummy_logits, bonus_logits_indices,
                            target_logits_indices))
                    warm_temps = torch.zeros((num_tokens, 1),
                                             dtype=self._hidden_states_dtype,
                                             device=self.device)
                    warm_top_k = torch.zeros((num_tokens, 1),
                                             dtype=torch.int32,
                                             device=self.device)
                    warm_top_p = torch.ones((num_tokens, 1),
                                            dtype=torch.float32,
                                            device=self.device)
                    warm_recover_u = torch.zeros((num_tokens, self.vocab_size),
                                                 dtype=torch.float32,
                                                 device=self.device)
                    out = self.rejection_sampler(
                        draft_token_ids=draft_token_ids,
                        num_draft_tokens=draft_lengths,
                        target_logits=target_logits_ng,
                        bonus_token_ids=bonus_token_ids,
                        segment_ids=segment_ids,
                        group_indices=group_indices,
                        max_draft_tokens=k,
                        temperatures=warm_temps,
                        top_k=warm_top_k,
                        top_p=warm_top_p,
                        accept_u=warm_accept_u,
                        recover_u=warm_recover_u,
                        do_sampling=True,
                    )
                    synchronize_tensors(out)
                    logger.info("  -- padded_logits_length: %d, num_seqs: %d",
                                num_tokens, num_reqs)
                    if num_reqs >= min(num_tokens, self.max_num_reqs):
                        break

    def _precompile_sampling_subgraphs(self) -> None:
        """Compile sampling-path subgraphs so their bottom-HBM reservations
        are visible to vLLM's available-memory probe in profile_run."""
        self._precompile_compute_selected_logits()
        if (_get_sequence_layout_planner_for_runner(
                self).uses_selected_logits_hidden_states):
            self._precompile_compute_logits_from_hidden_states()
        self._precompile_structured_decoding()
        self._precompile_sample_from_logits()
        self._precompile_gather_logprobs()

    def capture_model(self) -> None:
        """Precompile every torch.compile subgraph across all input buckets."""
        if self.enforce_eager:
            return
        with self.maybe_setup_dummy_loras(self.lora_config):
            with self._precompile_timed("model backbone"):
                self._precompile_backbone()

            self._precompile_sampling_subgraphs()
            self._precompile_mamba_rollback_helpers()

            # Precompile multimodal vision encoder graphs
            self.encoder_cudagraph_manager = (maybe_create_mm_encoder_manager(
                self.vllm_config, self.device, self.model))
            if self.encoder_cudagraph_manager is not None:
                with self._precompile_timed("multimodal vision encoder"):
                    self.encoder_cudagraph_manager.precompile_vision_encoder()

            # Warm the drafter's forward + sampling subgraphs at every
            # bucket shape it may see at runtime.
            if (self._is_async_drafter):
                self.drafter.precompile()
                if self._pcp_mtp_k1_enabled:
                    logger.info(
                        "Skipping rejection replay and decode/verify warmup "
                        "for prefill-only PCP MTP K1")
                    self._warmup_pcp_mtp_prefill()
                else:
                    self._precompile_rejection_sampler()
                    # The isolated precompiles above emit standalone fused
                    # programs; the real per-step propose+verify+sampling
                    # dispatch fuses them differently, so the first real
                    # request would otherwise compile new graphs. Warm those
                    # through the real two-phase spec-decode path.
                    self._warmup_spec_decode()

    def _precompile_backbone(self) -> None:
        """Compile the backbone for every token bucket.

        One trace serves the whole ladder unless the model's structure depends
        on the token count, in which case a bucket the current graph cannot
        serve gets a trace of its own and hands its executable back, leaving
        one executable per bucket either way (see
        vllm_torchtpu.compilation.shape_variants).
        """
        with shape_variants.warmup() as buckets:
            for num_tokens in self.num_tokens_paddings:
                logger.info("  -- num_tokens: %d", num_tokens)
                if buckets.needs_retrace(num_tokens):
                    shape_variants.retrace(self.model, self.vllm_config)
                self._dummy_run(num_tokens,
                                self.num_reqs_max_model_len,
                                self.max_num_blocks_per_req,
                                use_max_model_len=True)
                if self.most_model_len is not None:
                    self._dummy_run(num_tokens,
                                    self.num_reqs_most_model_len,
                                    self.num_blocks_per_most_len_req,
                                    use_max_model_len=False)
            if buckets.refused:
                # A refusal that no trace picked up leaves a raising closure in
                # the live graph, which would surface as a failed request. Fail
                # the start instead, where it can be read.
                raise shape_variants.ShapeSpecializationError(
                    "warmup left token buckets uncompiled: "
                    f"{sorted(buckets.refused)}")

    @contextmanager
    def _profile_isolated_cache(self) -> Iterator[None]:
        # Isolate the kv=0-specialized profile compile into a dedicated
        # subdirectory ("profile_cache") so it can be cached across warm
        # restarts without colliding with the real serving cache (kv > 0)
        # or across different models / TP sizes / dtypes.
        import hashlib

        from vllm.compilation.caching import aot_compile_hash_factors

        from vllm_torchtpu.compilation.tpu_compiler import \
            compute_tpu_compilation_hash
        cc = self.vllm_config.compilation_config
        base_cache_dir = cc.cache_dir or vllm_envs.VLLM_CACHE_ROOT
        factors = aot_compile_hash_factors(self.vllm_config)
        factors.append(compute_tpu_compilation_hash(self.vllm_config))
        hash_key = hashlib.sha256(str(factors).encode()).hexdigest()[:10]
        saved_cache_dir = cc.cache_dir
        cc.cache_dir = os.path.join(base_cache_dir, "profile_cache", hash_key)
        os.makedirs(cc.cache_dir, exist_ok=True)
        try:
            yield
        finally:
            cc.cache_dir = saved_cache_dir

    def profile_run(
        self,
        num_tokens: int,
    ) -> None:
        from vllm.compilation.wrapper import reset_compile_wrapper

        from vllm_torchtpu.layers.vllm.linear_common import (
            _get_quantized_matmul_fp4_op, _get_quantized_matmul_op)

        # set_current_vllm_config is required for the post-reset compile
        # path: reset_compile_wrapper restores the wrapper's original
        # forward code, so the next trace re-instantiates CustomOps that
        # call get_current_vllm_config().
        cc = self.vllm_config.compilation_config
        saved_cache = (cc.cache_dir, cc.local_cache_dir)
        saved_sizes = cc.compile_sizes
        with set_current_vllm_config(self.vllm_config):
            self._initialize_attention_kernels()
            # Pre-warm FP8 quantized-matmul lock; Dynamo can't trace Lock.
            _get_quantized_matmul_op()
            # Same for the NVFP4 W4A16 matmul op.
            _get_quantized_matmul_fp4_op()

            # Compile backbone here so XLA materializes the FP8 activation
            # slab; vLLM's post-probe sees the real HBM. Cache writes are
            # disabled so the kv=0-specialized graph never lands in the
            # persistent serving cache. Compile only the max token bucket:
            # every program loaded on the TPU holds a permanent bottom-of-HBM
            # program-region reservation, and the kv=0-specialized graphs are
            # dead after the dynamo reset below — one throwaway program keeps
            # the region within budget for capture_model's full bucket ladder.
            with self._profile_isolated_cache():
                cc.compile_sizes = [num_tokens]
                self._dummy_run(num_tokens, self.num_reqs_max_model_len,
                                self.max_num_blocks_per_req)
            reset_compile_wrapper(self.model)
            # reset_compile_wrapper wipes cache_dir; restore so
            # capture_model can read/write the persistent cache.
            cc.cache_dir, cc.local_cache_dir = saved_cache
            cc.compile_sizes = saved_sizes
            torch._dynamo.reset()
            synchronize_device()

            # Sampling subgraphs (Phase A) — uses the restored cache_dir.
            if not self.enforce_eager:
                with self.maybe_setup_dummy_loras(self.lora_config):
                    self._precompile_sampling_subgraphs()
