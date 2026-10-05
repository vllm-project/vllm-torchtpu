# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream V4.1 model assembly with TPU device operations."""

from dataclasses import replace
from itertools import islice

import torch
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import override_forward_context
from vllm.models.deepseek_v41.nvidia import model as upstream

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.quantization.deepseek_v41 import DeepseekV41MoE
from vllm_torchtpu.logger import init_logger

from ..common.mhc import (
    collapse,
    initialize_delayed_mhc,
    initialize_mhc_post,
    pack_mix_weights,
    post,
    pre,
    shifted_post_pre,
)
from .attention import DeepseekV41Attention, _block_table, _cache_slots
from .attention_kernels import initialize_indexer, initialize_paged_attention
from .cache import initialize_cache_ops
from .ced import plan_decoder_replay
from .engram import Engram, NgramHashState
from .ops import bind_device_ops, compact_rope_caches, device_hooks

logger = init_logger(__name__)


class DeepseekV41DecoderLayer(upstream.DeepseekV4DecoderLayer):
    forward = bind_device_ops(
        upstream.DeepseekV4DecoderLayer.forward,
        mhc_pre_delayed_tilelang=pre,
        mhc_post_tilelang=post,
        mhc_shifted_post_pre=shifted_post_pre,
    )


class _StageBase(torch.nn.Module):
    def __init__(self, *, owner, vllm_config, prefix=""):
        super().__init__()
        # Parameters retain their upstream model.layers checkpoint paths. A
        # registered parent reference would create a module-ownership cycle.
        object.__setattr__(self, "owner", owner)


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class DeepseekV41FullStage(_StageBase):
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        return self.owner._forward_stage(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
            stage="full",
        )


@support_torch_compile(
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class DeepseekV41EncoderStage(_StageBase):
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        return self.owner._forward_stage(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
            stage="encoder",
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "encoder_pre_mix": 0,
    }
)
class DeepseekV41DecoderStage(_StageBase):
    def forward(self, input_ids, positions, inputs_embeds, encoder_pre_mix):
        return self.owner._forward_stage(
            input_ids,
            positions,
            None,
            inputs_embeds,
            encoder_pre_mix=encoder_pre_mix,
            stage="decoder",
        )


class DeepseekV41Model(upstream.DeepseekV4Model):
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        if not self._ced_enabled:
            return self.full_stage(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids,
            )
        context = upstream.get_forward_context()
        metadata = context.attn_metadata
        first = next(iter(metadata.values())) if isinstance(metadata, dict) else None
        plan = (
            plan_decoder_replay(
                first, self._ced_window, positions.shape[0], self._ced_compile_sizes
            )
            if first is not None
            else None
        )
        encoded = self.encoder_stage(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )
        # Match the runner's convention for compiled backends that wrap the
        # model result in a single-element output list.
        if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
            encoded = encoded[0]
        hidden, pre_mix = encoded
        if plan is not None and not plan.indices:
            # The scheduler ignores logits for incomplete prefills. The
            # encoder has still published every global KV row for this chunk.
            return hidden.new_zeros((positions.shape[0], self.config.hidden_size))

        if plan is None:
            decoder_metadata = metadata
            decoder_positions, decoder_ids = positions, input_ids
            decoder_hidden, decoder_mix = hidden, pre_mix
            requests = first.seq_lens.shape[0] if first is not None else 1
            replay_starts = positions.new_zeros(requests)
            indices = None
            decoder_padding = (
                torch.arange(positions.shape[0], device=positions.device)
                >= first.query_start_loc[-1]
                if first is not None
                else torch.zeros_like(positions, dtype=torch.bool)
            )
        else:
            # The extra output row absorbs padded scatter writes. Reusing row
            # zero for padding would overwrite a real request's first result.
            indices = torch.tensor(
                plan.indices
                + (positions.shape[0],) * (plan.padded_num_tokens - len(plan.indices)),
                dtype=torch.int64,
                device=positions.device,
            )
            gather = indices.clamp(max=positions.shape[0] - 1)
            decoder_positions = positions[gather]
            decoder_ids = input_ids[gather] if input_ids is not None else None
            decoder_hidden, decoder_mix = hidden[gather], pre_mix[gather]
            bounds = torch.tensor(
                plan.query_start_loc, dtype=torch.int32, device=positions.device
            )
            replay_starts = torch.tensor(
                plan.replay_starts, dtype=positions.dtype, device=positions.device
            )
            decoder_metadata = {
                key: replace(
                    value, input_positions=decoder_positions, query_start_loc=bounds
                )
                for key, value in metadata.items()
            }
            decoder_padding = torch.arange(
                plan.padded_num_tokens, device=positions.device
            ) >= len(plan.indices)
        decoder_context = replace(
            context,
            attn_metadata=decoder_metadata,
            is_padding=decoder_padding,
            additional_kwargs={
                **context.additional_kwargs,
                "deepseek_v41_replay_starts": replay_starts,
                "token_padding_mask": decoder_padding,
            },
        )
        with override_forward_context(decoder_context):
            output = self.decoder_stage(
                decoder_ids,
                decoder_positions,
                inputs_embeds=decoder_hidden,
                encoder_pre_mix=decoder_mix,
            )
        if isinstance(output, (list, tuple)) and len(output) == 1:
            output = output[0]
        if indices is None:
            return output
        result = output.new_zeros((positions.shape[0] + 1, output.shape[-1]))
        return result.index_copy_(0, indices, output)[:-1]

    # Keep the upstream forward order; adapt its Engram metadata access here,
    # alongside attention/compressor adapters, so the runner stays model-neutral.
    def _forward_stage(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: upstream.IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        lookback_token_ids: torch.Tensor | None = None,
        *,
        encoder_pre_mix: torch.Tensor | None = None,
        stage: str = "full",
    ) -> torch.Tensor | upstream.IntermediateTensors:
        start_layer = self._ced_boundary if stage == "decoder" else self.start_layer
        end_layer = self._ced_boundary if stage == "encoder" else self.end_layer
        if stage == "decoder":
            hidden_states = inputs_embeds
        elif upstream.get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_mega_moe:
            input_ids = input_ids.to(torch.int64)

        # Engram n-gram hashes for the whole (flattened) batch, computed once
        # on the full token stream — before any sequence-parallel sharding —
        # and consumed by the engram layers (1 and 14) below. Skipped on
        # profile runs (KV cache unbound).
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if (
            stage != "decoder"
            and self.engram_hash is not None
            and input_ids is not None
            and upstream.is_forward_context_available()
        ):
            attn_metadata = upstream.get_forward_context().attn_metadata
            if isinstance(attn_metadata, list):
                attn_metadata = attn_metadata[upstream.dbo_current_ubatch_id()]
            if isinstance(attn_metadata, dict) and self.engram_hash.ensure_cache():
                assert self.engram_swa_prefix is not None
                swa_metadata = attn_metadata[self.engram_swa_prefix]
                # Image-span tokens are dead: they break n-grams (hash op
                # takes True=dead) and their gate is zeroed (Engram.forward
                # takes True=keep).
                image_mask = upstream.image_sentinel_mask(input_ids)
                engram_mask = ~image_mask
                if lookback_token_ids is None:
                    if not self.engram_hash.use_slot_cache:
                        raise NotImplementedError(
                            "engram needs `lookback_token_ids` from the model "
                            "runner (the DBO/ubatch wrapper drops model kwargs)"
                        )
                    num_reqs = swa_metadata.seq_lens.shape[0]
                    lookback_token_ids = input_ids.new_full(
                        (num_reqs, self.engram_hash.lookback_depth), -1
                    )
                engram_hashes = self.engram_hash(
                    input_ids,
                    positions,
                    swa_metadata.query_start_loc,
                    image_mask,
                    lookback_token_ids,
                    upstream.image_sentinel_mask(lookback_token_ids),
                    _cache_slots(
                        self.engram_hash.swa_cache_module.kv_cache,
                        swa_metadata,
                        positions,
                    ),
                    _block_table(swa_metadata),
                )
            elif not self.engram_dp_shared_memory and upstream.get_engram_dp_size() > 1:
                # DP-sharded lookups are collective, so a replica skipping the
                # hash still has to reach them.
                engram_hashes, engram_mask = self.engram_hash.dummy_hashes(input_ids)
            if engram_hashes is not None:
                # Gather all Engram rows before entering the decoder layers.
                # One gather feeds every layer sharing the DP-split table.
                gathered_hashes = upstream.gather_engram_hashes(
                    engram_hashes, dp_shared_memory=self.engram_dp_shared_memory
                )
                for layer in islice(self.layers, self.start_layer, self.end_layer):
                    engram = getattr(layer, "engram", None)
                    if engram is not None:
                        engram.prepare_embeddings(
                            gathered_hashes[:, engram.layer_hash_index]
                        )

        full_num_tokens = positions.shape[0]
        if self.use_sequence_parallel:
            if (
                upstream.envs.VLLM_MOE_SKIP_PADDING
                and upstream.is_forward_context_available()
            ):
                forward_context = upstream.get_forward_context()
                forward_context.is_padding = upstream.sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = upstream.sp_shard(hidden_states)
            input_ids = upstream.sp_shard(input_ids)

        residual, post_mix, res_mix = None, None, None
        pre_mix: torch.Tensor | None = encoder_pre_mix
        if not upstream.get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        # Every layer's post runs inside the next layer's fused pre, so aux
        # hidden states are read back from there instead of recomputed.
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        for idx, layer in enumerate(
            islice(self.layers, start_layer, end_layer),
            start=start_layer,
        ):
            hidden_states, residual, post_mix, res_mix, pre_mix, previous_aux = layer(
                hidden_states,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
                capture_previous_aux=idx in self.aux_hidden_state_layers,
            )
            if previous_aux is not None:
                # idx is the one-based id of the layer whose post this is.
                if self.use_sequence_parallel:
                    previous_aux = upstream.sp_all_gather(previous_aux)[
                        :full_num_tokens
                    ]
                aux_hidden_by_layer[idx] = previous_aux
        if layer is not None:
            # The last layer has no successor to fold its post into.
            hidden_states = post(hidden_states, residual, post_mix, res_mix)
            if self.end_layer in self.aux_hidden_state_layers:
                final_aux = hidden_states.mean(dim=1)
                if self.use_sequence_parallel:
                    final_aux = upstream.sp_all_gather(final_aux)[:full_num_tokens]
                aux_hidden_by_layer[self.end_layer] = final_aux

        if stage == "encoder":
            # Decoder Full-mode global KV depends only on the complete encoder
            # output. Publish it before the host selects the SWA replay rows.
            decoder = self.layers[self._ced_boundary]
            assert pre_mix is not None
            normalized = decoder.attn_norm(collapse(hidden_states, pre_mix))
            decoder.attn.write_encoder_global_kv(normalized, positions)
            return hidden_states, pre_mix

        aux_hidden_states = [
            aux_hidden_by_layer[layer_id]
            for layer_id in self.aux_hidden_state_layers
            if layer_id in aux_hidden_by_layer
        ]

        if not upstream.get_pp_group().is_last_rank:
            return upstream.IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )

        # MTP needs full HC states; otherwise collapse and normalize locally
        # before gathering to reduce communication.
        if self._mtp_hidden_buffer is not None:
            if self.use_sequence_parallel:
                hidden_states = upstream.sp_all_gather(hidden_states)[:full_num_tokens]
                pre_mix = upstream.sp_all_gather(pre_mix)[:full_num_tokens]
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        # Collapse the hc copies with the pre-mix from the last layer's FFN
        # mixes — the mix the reference applies via
        # ``last_layer.hc_pre(h, pre_mix)`` (v4.1 has no learned hc_head).
        assert pre_mix is not None
        hidden_states = collapse(hidden_states, pre_mix)
        hidden_states = self.norm(hidden_states)
        if self.use_sequence_parallel and self._mtp_hidden_buffer is None:
            # Without MTP, gather only the collapsed and normalized hidden states.
            hidden_states = upstream.sp_all_gather(hidden_states)[:full_num_tokens]
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states

    def finalize_mhc_broadcast_weights(self) -> None:
        layers = list(islice(self.layers, self.start_layer, self.end_layer))
        if layers and layers[0].hc_attn_fn.dtype != torch.float32:
            return  # Already packed.
        super().finalize_mhc_broadcast_weights()
        pack_mix_weights(layers)

    def __init__(self, *, vllm_config, prefix=""):
        self.prefix = prefix
        initialize_cache_ops()
        initialize_delayed_mhc()
        initialize_mhc_post()
        initialize_paged_attention()
        initialize_indexer()
        with (
            device_hooks(
                upstream,
                _select_dsv4_attn_cls=lambda _: DeepseekV41Attention,
                Engram=Engram,
                NgramHashState=NgramHashState,
                DeepseekV4DecoderLayer=DeepseekV41DecoderLayer,
                DeepseekV4MoE=DeepseekV41MoE,
            ),
            device_hooks(torch.cuda, Stream=lambda: None),
        ):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        compact_rope_caches(
            (layer.attn for layer in self.layers),
            vllm_config.model_config.max_model_len,
        )
        self._ced_enabled = bool(envs.TPU_DEEPSEEK_V41_CED)
        if self._ced_enabled and vllm_config.cache_config.enable_prefix_caching:
            logger.warning(
                "CED bounded replay requires prefix caching disabled; "
                "using full forward to retain cached SWA state semantics"
            )
            self._ced_enabled = False
        self._ced_boundary = self.config.num_hidden_layers // 2
        self._ced_window = self.layers[self._ced_boundary].attn.window_size
        self._ced_compile_sizes = tuple(
            n
            for n in (vllm_config.compilation_config.compile_sizes or [])
            if isinstance(n, int)
        )
        if self._ced_enabled:
            if (
                self.use_sequence_parallel
                or getattr(
                    vllm_config.parallel_config, "prefill_context_parallel_size", 1
                )
                != 1
            ):
                raise NotImplementedError(
                    "CED bounded replay requires unsharded token rows"
                )
            sources = [
                i
                for i in range(self._ced_boundary, self.end_layer)
                if self.layers[i].attn.compressor is not None
            ]
            if sources != [self._ced_boundary]:
                raise NotImplementedError(
                    "CED requires one decoder KV source at the encoder boundary"
                )
            if any(
                layer.engram is not None for layer in self.layers[self._ced_boundary :]
            ):
                raise NotImplementedError("CED requires Engram layers in the encoder")
            for layer in self.layers[self._ced_boundary :]:
                layer.attn._ced_decoder = True
            self.layers[self._ced_boundary].attn._ced_global_from_encoder = True
            # Separate tags prevent the two backends from sharing a cache
            # namespace. Preserve them when bucket warmup rebuilds wrappers.
            with set_model_tag("v41_encoder"):
                self.encoder_stage = DeepseekV41EncoderStage(
                    owner=self,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.encoder_stage",
                )
            self.encoder_stage._compile_prefix = "v41_encoder"
            with set_model_tag("v41_decoder"):
                self.decoder_stage = DeepseekV41DecoderStage(
                    owner=self,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.decoder_stage",
                )
            self.decoder_stage._compile_prefix = "v41_decoder"
            logger.info(
                "CED bounded replay enabled: encoder layers=%d, "
                "decoder layers=%d, window=%d",
                self._ced_boundary,
                self.end_layer - self._ced_boundary,
                self._ced_window,
            )
        else:
            self.full_stage = DeepseekV41FullStage(
                owner=self,
                vllm_config=vllm_config,
                prefix=f"{prefix}.full_stage",
            )


class DeepseekV41ForCausalLM(upstream.DeepseekV41LLMForCausalLM):
    model_cls = DeepseekV41Model

    def forward(self, *args, **kwargs):
        # Materialize upstream's persistent history outside the compiled model.
        # Its lazy first-use zeros allocation must not become an operation that
        # vLLM's guard-free graph replays at every decode step.
        if self.model.engram_hash is not None:
            self.model.engram_hash.ensure_cache()
        return super().forward(*args, **kwargs)

    def __init__(self, *, vllm_config, prefix=""):
        # Keep capability guards until their device paths are validated.
        if (
            vllm_config.parallel_config.pipeline_parallel_size != 1
            or vllm_config.parallel_config.data_parallel_size != 1
        ):
            raise NotImplementedError("V4.1 TPU currently requires PP1 and DP1")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("V4.1 TPU currently requires no speculation")
        if vllm_config.scheduler_config.async_scheduling:
            from .async_fence import install_lookback_fence

            install_lookback_fence()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # TPU's existing FP8 linear adapter registers weight_scale_inv; the
        # NVIDIA MXFP8 adapter selected by upstream uses weight_scale instead.
        self.hf_to_vllm_mapper = upstream._make_deepseek_v4_weights_mapper(
            getattr(self.config, "expert_dtype", "fp4"), "weight_scale_inv"
        )
