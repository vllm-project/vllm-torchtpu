# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Upstream V4.1 model assembly with TPU device operations."""

from dataclasses import dataclass, replace
from functools import cached_property, partial
from itertools import islice
from typing import TYPE_CHECKING, cast

import torch
from torch.utils._pytree import tree_leaves
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import override_forward_context
from vllm.models.deepseek_v41.nvidia import model as upstream

from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_mhc_op import (
    initialize_shifted_mhc,
)
from vllm_torchtpu.layers.adapter.quantization.deepseek_v41 import DeepseekV41MoE
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import synchronize_tensors

from ..common.mhc import (
    collapse,
    initialize_delayed_mhc,
    initialize_mhc_post,
    pack_mix_weights,
    post,
    pre,
    shifted_post_pre,
)
from ..config import (
    DeepseekV41Config,
    ExecutionMode,
    engram_cpu_offload,
)
from .attention import (
    DeepseekV41Attention,
    _block_table,
    _cache_slots,
    stage_attention_cache_scope,
)
from .attention_kernels import initialize_paged_attention
from .cache import initialize_cache_ops
from .ced import (
    coordinate_decoder_replay,
    plan_decoder_replay,
    prefix_replay_enabled,
)
from .engram import Engram, NgramHashState
from .host_engram import (
    HOST_ENGRAM_KEY,
    HostEngramInputs,
    gather_host_engram_hashes,
    install_host_engram_inputs,
)
from .indexer_impl import initialize_indexer
from .ops import bind_device_ops, compact_rope_caches, device_hooks

logger = init_logger(__name__)

if TYPE_CHECKING:
    from ..common.host_transfer import EngramHashDownload, PreparedEngramUpload


@dataclass
class _EngramPrefetch:
    inputs: HostEngramInputs
    hashes: torch.Tensor
    uploads: dict[int, "PreparedEngramUpload"]


def _ced_config(
    vllm_config, *, warn_prefix_caching: bool = False, fallback_config=None
) -> tuple[bool, int]:
    cache_config = getattr(vllm_config, "cache_config", None)
    enable_prefix_caching = bool(getattr(cache_config, "enable_prefix_caching", False))
    ced_enabled = (
        DeepseekV41Config.from_vllm_config(vllm_config).execution == ExecutionMode.CED
    )
    if ced_enabled and enable_prefix_caching and not prefix_replay_enabled(vllm_config):
        if warn_prefix_caching:
            logger.warning(
                "CED with prefix caching needs SWA bounded replay "
                "(cache_config.swa_bounded_replay, no PCP); using full forward "
                "to retain cached SWA state semantics"
            )
        ced_enabled = False
    hf_config = getattr(
        getattr(vllm_config, "model_config", None), "hf_config", fallback_config
    )
    assert hf_config is not None
    ced_boundary = hf_config.num_hidden_layers // 2
    return ced_enabled, ced_boundary


def _scatter_decoder_rows(
    tensor: torch.Tensor, indices: torch.Tensor, total_rows: int
) -> torch.Tensor:
    result = tensor.new_zeros((total_rows + 1, tensor.shape[-1]))
    return result.index_copy_(0, indices, tensor)[:-1]


def _mean_hc_streams(hidden_states: torch.Tensor) -> torch.Tensor:
    # Unroll the small static HC stream axis (hc_mult=4) instead of emitting a
    # second HLO reduce over dim=1 alongside collapse(hidden_states, pre_mix).
    # XLA MultiOutputFusion otherwise fuses the bf16 mean reduce and f32
    # collapse reduce into a mixed-type tuple reduce fusion that aborts in
    # libtpu FusionEmitter::ComputeWindowConfigInternal at N=2048.
    streams = hidden_states.float().unbind(dim=1)
    acc = streams[0]
    for s in streams[1:]:
        acc = acc + s
    return (acc * (1.0 / len(streams))).to(hidden_states.dtype)


class DeepseekV41DecoderLayer(upstream.DeepseekV4DecoderLayer):
    def __init__(self, vllm_config, prefix: str = "", **kwargs) -> None:
        layer_idx = upstream.extract_layer_index(prefix)
        ced_enabled, ced_boundary = _ced_config(vllm_config)
        if ced_enabled and layer_idx >= ced_boundary:
            with device_hooks(upstream, _use_sequence_parallel=lambda _: False):
                super().__init__(vllm_config, prefix=prefix, **kwargs)
        else:
            super().__init__(vllm_config, prefix=prefix, **kwargs)
        if hasattr(self, "attn"):
            self.attn._sp_sharded_input = bool(
                getattr(self, "use_sequence_parallel", False)
            )

    forward = bind_device_ops(
        upstream.DeepseekV4DecoderLayer.forward,
        mhc_pre_delayed_tilelang=pre,
        mhc_post_tilelang=post,
        mhc_shifted_post_pre=partial(shifted_post_pre, fuse_rms=True),
        sp_all_gather=lambda x: x,
    )


_upstream_use_sequence_parallel = upstream._use_sequence_parallel


def _use_sequence_parallel(vllm_config) -> bool:
    """Use upstream's topology-derived sequence parallelism."""
    return _upstream_use_sequence_parallel(vllm_config)


class _StageBase(torch.nn.Module):
    def __init__(self, *, owner, vllm_config, prefix="", compile_tag=None):
        super().__init__()
        # Parameters retain their upstream model.layers checkpoint paths. A
        # registered parent reference would create a module-ownership cycle.
        object.__setattr__(self, "owner", owner)
        if compile_tag is not None:
            # The TPU compile-prefix patch keys the first trace's cache_dir off
            # self.prefix; a bucket-warmup retrace re-inits with
            # _compile_prefix. Both must name the same directory, or the
            # retrace cannot hand over buckets the first trace compiled.
            self.prefix = compile_tag


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
    dynamic_arg_dims={"input_ids": 0, "positions": 0, "inputs_embeds": 0}
)
class DeepseekV41EncoderHeadStage(_StageBase):
    """Encoder layers before the split Engram layer (see ``_engram_split_layer``)."""

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
            stage="encoder_head",
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "encoder_residual": 0,
        "encoder_post_mix": 0,
        "encoder_res_mix": 0,
        "encoder_pre_mix": 0,
    }
)
class DeepseekV41EncoderMiddleStage(_StageBase):
    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds,
        encoder_residual,
        encoder_post_mix,
        encoder_res_mix,
        encoder_pre_mix,
    ):
        return self.owner._forward_stage(
            input_ids,
            positions,
            None,
            inputs_embeds,
            encoder_pre_mix=encoder_pre_mix,
            encoder_carry=(encoder_residual, encoder_post_mix, encoder_res_mix),
            stage="encoder_middle",
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "encoder_residual": 0,
        "encoder_post_mix": 0,
        "encoder_res_mix": 0,
        "encoder_pre_mix": 0,
    }
)
class DeepseekV41EncoderTailStage(_StageBase):
    """Encoder layers from the split Engram layer to the CED boundary.

    Takes the head's un-finalized layer carry, so the split layer still
    injects Engram between the previous post and its own pre.
    """

    def forward(
        self,
        input_ids,
        positions,
        inputs_embeds,
        encoder_residual,
        encoder_post_mix,
        encoder_res_mix,
        encoder_pre_mix,
    ):
        return self.owner._forward_stage(
            input_ids,
            positions,
            None,
            inputs_embeds,
            encoder_pre_mix=encoder_pre_mix,
            encoder_carry=(encoder_residual, encoder_post_mix, encoder_res_mix),
            stage="encoder_tail",
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
    # Encoder layer whose host Engram staging overlaps the encoder head on
    # device; derived from host placement. None runs one encoder graph.
    _engram_split_layer: int | None = None
    _engram_first_split_layer: int | None = None
    _gathered_engram_hashes: torch.Tensor | None = None
    _raw_dma_enabled = False
    _engram_prefetch: _EngramPrefetch | None = None
    _current_host_inputs: HostEngramInputs | None = None
    _prefetched_uploads: dict[int, "PreparedEngramUpload"] | None = None

    @cached_property
    def _hash_download(self) -> "EngramHashDownload":
        from ..common.host_transfer import EngramHashDownload

        return EngramHashDownload()

    def _has_cpu_offloaded_engram(self) -> bool:
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            engram = getattr(layer, "engram", None)
            if engram is not None and getattr(
                engram.embed_tokens, "host_offload", False
            ):
                return True
        return False

    def _host_hash_inputs(self, input_ids, swa_metadata, lookback_token_ids):
        """CPU ``hash_on_host`` arguments for this step, or None.

        Applies only to Host-DRAM Engram tables outside compilation, when the
        runner published host-ready copies (``host_engram.py``). Slices follow
        the device tensors' static shapes, so the host hashes cover exactly the
        device rows.
        """
        if (
            lookback_token_ids is None
            or torch.compiler.is_compiling()
            or not self._has_cpu_offloaded_engram()
        ):
            return None
        host = upstream.get_forward_context().additional_kwargs.get(HOST_ENGRAM_KEY)
        if host is None or not host.can_hash_on_host:
            return None
        if self._raw_dma_enabled and host.all_prompt:
            self._current_host_inputs = host.snapshot(
                input_ids.shape[0],
                lookback_token_ids.shape[0],
                swa_metadata.query_start_loc.shape[0],
            )
        ids = host.input_ids[: input_ids.shape[0]]
        lookback = host.lookback_token_ids[: lookback_token_ids.shape[0]]
        return (
            ids,
            host.positions[: input_ids.shape[0]],
            host.query_start_loc[: swa_metadata.query_start_loc.shape[0]],
            upstream.image_sentinel_mask(ids),
            lookback,
            upstream.image_sentinel_mask(lookback),
        )

    def _prepare_engram(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
        *,
        stage: str = "full",
        skip_lookup: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
                if skip_lookup:
                    # `Engram.embed` only reads `hash_ids.shape[0]` (`self._ready_rows(
                    # hash_ids.shape[0])`) after `prepare_embeddings` has populated
                    # `staged_rows` outside the compiled stage.
                    engram_mask = ~upstream.image_sentinel_mask(input_ids)
                    num_layers = int(self.engram_hash.multipliers.shape[0])
                    engram_hashes = input_ids[:, None, None].expand(-1, num_layers, 1)
                    assert engram_hashes.shape[0] == input_ids.shape[0]
                    return engram_hashes, engram_mask
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
                host = self._host_hash_inputs(
                    input_ids, swa_metadata, lookback_token_ids
                )
                if host is not None:
                    prefetched = self._engram_prefetch
                    self._engram_prefetch = None
                    current = self._current_host_inputs
                    if (
                        prefetched is not None
                        and current is not None
                        and prefetched.inputs.matches(current)
                    ):
                        engram_hashes = prefetched.hashes
                        self._prefetched_uploads = prefetched.uploads
                        logger.info_once("[DSV41] Engram next-prompt prefetch matched")
                    else:
                        engram_hashes = self.engram_hash.hash_on_host(*host)
                    engram_mask = ~upstream.image_sentinel_mask(host[0])
                else:
                    image_mask = upstream.image_sentinel_mask(input_ids)
                    engram_mask = ~image_mask
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
            if engram_hashes is not None and not skip_lookup:
                # Gather all Engram rows before entering the decoder layers.
                # One gather feeds every layer sharing the DP-split table.
                if self._has_cpu_offloaded_engram():
                    # Idle/decode replicas may hash on device while a prefill
                    # replica hashes on host. All must use the CPU collective.
                    engram_hashes = self._hash_download.copy_to_host(engram_hashes)
                gathered_hashes = gather_host_engram_hashes(
                    engram_hashes,
                    dp_shared_memory=self.engram_dp_shared_memory,
                    gather_fn=upstream.gather_engram_hashes,
                )
                end_layer = (
                    getattr(self, "_ced_boundary", self.end_layer)
                    if stage == "encoder"
                    else self.end_layer
                )
                split = getattr(self, "_engram_split_layer", None)
                if stage == "encoder" and split is not None:
                    # Stage later segments after their preceding graphs
                    # have been enqueued.
                    self._gathered_engram_hashes = gathered_hashes
                    first_split = self._engram_first_split_layer
                    end_layer = first_split if first_split is not None else split
                self._stage_engram_layers(self.start_layer, end_layer, gathered_hashes)
        return engram_hashes, engram_mask

    def _stage_engram_layers(
        self,
        start_layer: int,
        end_layer: int,
        gathered_hashes: torch.Tensor | None,
    ) -> None:
        """Prepare each Engram in the segment before its graph is dispatched."""
        if gathered_hashes is not None:
            for layer in islice(self.layers, start_layer, end_layer):
                engram = getattr(layer, "engram", None)
                if engram is not None:
                    if (
                        isinstance(engram, Engram)
                        and engram.embed_tokens.host_offload
                        and engram.embed_tokens.dp_size == 1
                    ):
                        upload = (
                            self._prefetched_uploads.pop(engram.layer_hash_index, None)
                            if self._prefetched_uploads is not None
                            else None
                        )
                        if upload is None:
                            upload = engram.start_prepared_embeddings(
                                gathered_hashes[:, engram.layer_hash_index]
                            )
                        engram.activate_prepared_embeddings(
                            gathered_hashes.shape[0], upload
                        )
                    else:
                        engram.prepare_embeddings(
                            gathered_hashes[:, engram.layer_hash_index]
                        )

    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        if not self._raw_dma_enabled:
            return self._forward_impl(
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
                lookback_token_ids,
            )
        self._current_host_inputs = None
        self._prefetched_uploads = None
        output = self._forward_impl(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )
        self._record_engram_consumers(self.start_layer, self.end_layer, output)
        self._prefetch_next_prompt()
        return output

    def _record_engram_consumers(self, start: int, end: int, output) -> None:
        if not self._raw_dma_enabled:
            return
        uploads = [
            layer.engram._active_upload
            for layer in islice(self.layers, start, end)
            if isinstance(layer.engram, Engram)
            and layer.engram._active_upload is not None
            and layer.engram._active_upload.needs_consumer
        ]
        if not uploads:
            return
        if isinstance(output, upstream.IntermediateTensors):
            output = output.tensors
        tensors = [
            value for value in tree_leaves(output) if isinstance(value, torch.Tensor)
        ]
        synchronize_tensors(tensors, wait=False)
        for upload in uploads:
            upload.record_consumers(tensors)

    def _prefetch_next_prompt(self) -> None:
        current = self._current_host_inputs
        self._engram_prefetch = None
        if current is None or not current.all_prompt or current.next_prompt is None:
            return
        end = self._ced_boundary if self._ced_enabled else self.end_layer
        engrams = [
            layer.engram
            for layer in islice(self.layers, self.start_layer, end)
            if isinstance(layer.engram, Engram)
            and layer.engram.embed_tokens.host_offload
        ]
        # Predicting work must never introduce an unmatched DP collective.
        if not engrams or any(engram.embed_tokens.dp_size != 1 for engram in engrams):
            return
        following = current.next_prompt(current)
        if following is None:
            return
        assert self.engram_hash is not None
        hashes = self.engram_hash.hash_on_host(
            following.input_ids,
            following.positions,
            following.query_start_loc,
            upstream.image_sentinel_mask(following.input_ids),
            following.lookback_token_ids,
            upstream.image_sentinel_mask(following.lookback_token_ids),
        )
        uploads = {
            engram.layer_hash_index: engram.start_prepared_embeddings(
                hashes[:, engram.layer_hash_index]
            )
            for engram in engrams
        }
        self._engram_prefetch = _EngramPrefetch(following, hashes, uploads)

    def _forward_impl(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
        lookback_token_ids=None,
    ):
        if (
            inputs_embeds is None
            and input_ids is not None
            and input_ids.device.type == "tpu"
            and self.parallel_config.data_parallel_size > 1
        ):
            # TP1/DP8 full-graph compilation can misinterpret the BF16 tiles
            # of the unsharded embedding lookup. The same lookup is exact
            # outside the graph. Materialize it before either compiled stage;
            # retain input_ids for Engram hashing and token-based MoE routing.
            inputs_embeds = self.embed_input_ids(input_ids)
            # Enqueue the lookup without blocking CPU Engram preparation behind
            # the previous chunk. The compiled stage consumes its device buffer.
            synchronize_tensors((inputs_embeds,), wait=False)
        offload_engram = self._has_cpu_offloaded_engram()
        if offload_engram:
            input_ids_prep = (
                input_ids.to(torch.int64)
                if self.use_native_mega_moe
                and input_ids is not None
                and input_ids.device.type == "cpu"
                else input_ids
            )
            self._prepare_engram(
                input_ids_prep,
                positions,
                lookback_token_ids,
                stage="encoder" if self._ced_enabled else "full",
            )
            self._engram_prepared_outside_stage = True
        try:
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
            first = (
                next(iter(metadata.values())) if isinstance(metadata, dict) else None
            )
            # Keep the window bounds upload after Engram's blocking hash D2H.
            requests = first.seq_lens.shape[0] if first is not None else 1
            hits = tuple(getattr(first, "replay_starts_cpu", None) or ())[:requests]
            window_starts = torch.tensor(
                hits + (0,) * (requests - len(hits)),
                dtype=positions.dtype,
                device=positions.device,
            )
            encoder_context = replace(
                context,
                additional_kwargs={
                    **context.additional_kwargs,
                    "deepseek_v41_window_starts": window_starts,
                },
            )
            with override_forward_context(encoder_context):
                if getattr(self, "_engram_split_layer", None) is None:
                    encoded = self.encoder_stage(
                        input_ids,
                        positions,
                        intermediate_tensors,
                        inputs_embeds,
                        lookback_token_ids,
                    )
                else:
                    encoded = self._forward_split_encoder(
                        input_ids,
                        positions,
                        intermediate_tensors,
                        inputs_embeds,
                        lookback_token_ids,
                    )
            self._record_engram_consumers(self.start_layer, self._ced_boundary, encoded)
        finally:
            self._engram_prepared_outside_stage = False
            self._gathered_engram_hashes = None
        # Match the runner's convention for compiled backends that wrap the
        # model result in a single-element output list.
        if isinstance(encoded, (list, tuple)) and len(encoded) == 1:
            encoded = encoded[0]
        if len(encoded) == 3:
            hidden, pre_mix, encoder_aux = encoded
        else:
            hidden, pre_mix = encoded
            encoder_aux = []
        if context.dp_metadata is not None and hidden.device.type == "tpu":
            # Host bucket coordination depends only on scheduling metadata.
            # Enqueue the encoder before Gloo waits for other DP replicas.
            synchronize_tensors((hidden, pre_mix, *encoder_aux), wait=False)
        plan = (
            plan_decoder_replay(
                first,
                self._ced_window,
                positions.shape[0],
                self._ced_compile_sizes,
            )
            if first is not None
            else None
        )
        if context.dp_metadata is not None:
            plan = coordinate_decoder_replay(plan, positions.shape[0])
        if plan is not None and plan.padded_num_tokens == 0:
            # The scheduler ignores logits for incomplete prefills. The
            # encoder has still published every global KV row for this chunk;
            # skip the inter-stage SP all-gather when no decoder replay runs.
            zero_hidden = hidden.new_zeros(
                (positions.shape[0], self.config.hidden_size)
            )
            aux_layers = getattr(self, "aux_hidden_state_layers", ())
            if aux_layers:
                return zero_hidden, [zero_hidden for _ in aux_layers]
            return zero_hidden
        if self._stage_uses_sequence_parallel("encoder"):
            # Outside the compiled stages: the replay plan indexes full rows.
            hidden = upstream.sp_all_gather(hidden)[: positions.shape[0]]
            pre_mix = upstream.sp_all_gather(pre_mix)[: positions.shape[0]]
            if hidden.device.type == "tpu":
                # Materialize the eager inter-stage SP all-gather before
                # slicing and launching the compiled decoder stage so
                # DEFER_AND_FUSE does not fuse both stages across the boundary.
                synchronize_tensors((hidden, pre_mix))

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
            dp_metadata=(
                replace(
                    context.dp_metadata,
                    num_tokens_across_dp_cpu=torch.full_like(
                        context.dp_metadata.num_tokens_across_dp_cpu,
                        decoder_positions.shape[0],
                    ),
                    local_sizes=None,
                )
                if context.dp_metadata is not None
                else None
            ),
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
            if isinstance(output, (list, tuple)) and len(output) == 2:
                dec_hidden, dec_aux = output
                return dec_hidden, [*encoder_aux, *dec_aux]
            if encoder_aux:
                return output, list(encoder_aux)
            return output
        total_rows = positions.shape[0]
        if isinstance(output, (list, tuple)) and len(output) == 2:
            dec_hidden, dec_aux = output
            return (
                _scatter_decoder_rows(dec_hidden, indices, total_rows),
                [
                    *encoder_aux,
                    *[
                        _scatter_decoder_rows(aux, indices, total_rows)
                        for aux in dec_aux
                    ],
                ],
            )
        if encoder_aux:
            return (
                _scatter_decoder_rows(output, indices, total_rows),
                list(encoder_aux),
            )
        return _scatter_decoder_rows(output, indices, total_rows)

    def _forward_split_encoder(
        self,
        input_ids,
        positions,
        intermediate_tensors,
        inputs_embeds,
        lookback_token_ids,
    ):
        """Enqueue each encoder segment before staging the next Engram layer."""
        first_split = cast(int, self._engram_first_split_layer)
        split = cast(int, self._engram_split_layer)
        carry = self.encoder_head_stage(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids,
        )
        synchronize_tensors(carry, wait=False)
        self._record_engram_consumers(self.start_layer, first_split, carry)
        self._stage_engram_layers(first_split, split, self._gathered_engram_hashes)
        carry = self.encoder_middle_stage(input_ids, positions, *carry)
        synchronize_tensors(carry, wait=False)
        self._record_engram_consumers(first_split, split, carry)
        hidden, residual, post_mix, res_mix, pre_mix = carry
        self._stage_engram_layers(
            split, self._ced_boundary, self._gathered_engram_hashes
        )
        output = self.encoder_tail_stage(
            input_ids,
            positions,
            hidden,
            residual,
            post_mix,
            res_mix,
            pre_mix,
        )
        self._record_engram_consumers(split, self._ced_boundary, output)
        return output

    def _select_engram_first_split_layer(self) -> int | None:
        """Split before the first host Engram when an earlier layer can run."""
        if self._engram_split_layer is None:
            return None
        for i in range(self.start_layer, self._engram_split_layer):
            engram = self.layers[i].engram
            if engram is not None and engram.embed_tokens.host_offload:
                return i if i > self.start_layer else None
        return None

    def _select_engram_split_layer(self) -> int | None:
        """The last host-offloaded Engram encoder layer, when splitting applies."""
        if self.use_sequence_parallel and self._tp_size > 1:
            # SP at TP1 (DP replicas) keeps every row on the rank.
            logger.warning("Engram split encoder needs unsharded encoder rows; off")
            return None
        if any(
            self.start_layer <= i <= self._ced_boundary
            for i in getattr(self, "aux_hidden_state_layers", ())
        ):
            logger.warning("Engram split encoder: encoder aux hidden states; off")
            return None
        layers = [
            i
            for i in range(self.start_layer + 1, self._ced_boundary)
            if getattr(
                getattr(getattr(self.layers[i], "engram", None), "embed_tokens", None),
                "host_offload",
                False,
            )
        ]
        return layers[-1] if layers else None

    def _stage_uses_sequence_parallel(self, stage: str) -> bool:
        # Under CED=1, encoder_stage (L0..L19) runs SP while decoder_stage
        # (L20..L39) runs TP on the bounded replay / decode rows.
        return bool(self.use_sequence_parallel and stage != "decoder")

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
        encoder_carry: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
        stage: str = "full",
    ) -> torch.Tensor | upstream.IntermediateTensors | tuple[torch.Tensor, ...]:
        split = getattr(self, "_engram_split_layer", None)
        start_layer: int = self.start_layer
        end_layer: int = self.end_layer
        if stage == "decoder":
            start_layer = self._ced_boundary
        elif stage == "encoder":
            end_layer = self._ced_boundary
        elif stage == "encoder_head":
            assert split is not None
            first_split = self._engram_first_split_layer
            end_layer = first_split if first_split is not None else split
        elif stage == "encoder_middle":
            assert self._engram_first_split_layer is not None and split is not None
            start_layer, end_layer = self._engram_first_split_layer, split
        elif stage == "encoder_tail":
            assert split is not None
            start_layer, end_layer = split, self._ced_boundary
        stage_sp = self._stage_uses_sequence_parallel(stage)
        carry_stage = stage in ("encoder_middle", "encoder_tail")
        if stage == "decoder" or carry_stage:
            hidden_states = inputs_embeds
        elif upstream.get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        if self.use_native_mega_moe:
            input_ids = input_ids.to(torch.int64)

        skip_lookup = getattr(self, "_engram_prepared_outside_stage", False) or (
            torch.compiler.is_compiling() and self._has_cpu_offloaded_engram()
        )
        engram_hashes, engram_mask = self._prepare_engram(
            input_ids,
            positions,
            lookback_token_ids,
            stage=stage,
            skip_lookup=skip_lookup,
        )

        full_num_tokens = positions.shape[0]
        if stage_sp and carry_stage:
            # The head already sharded the carry and the padding mask.
            input_ids = upstream.sp_shard(input_ids)
        elif stage_sp:
            # sp_padding_mask slices by the SP chunk under a size-oblivious
            # guard, which lets Dynamo assume chunk >= 2 and then refuse the
            # smallest token bucket at warmup. Guard chunk == 1 explicitly so
            # every bucket specializes consistently.
            tp_size = upstream.get_tensor_model_parallel_world_size()
            sp_chunk = -(-hidden_states.shape[0] // tp_size)
            _ = bool(sp_chunk == 1)
            if (
                upstream.envs.VLLM_MOE_SKIP_PADDING
                and upstream.is_forward_context_available()
                and upstream.get_forward_context().is_padding is not None
            ):
                # A None mask means "no padding"; materializing one in-graph
                # (new_zeros of a symbolic size) fails torch_tpu lowering.
                forward_context = upstream.get_forward_context()
                forward_context.is_padding = upstream.sp_padding_mask(
                    forward_context.is_padding, hidden_states
                )
            hidden_states = upstream.sp_shard(hidden_states)
            input_ids = upstream.sp_shard(input_ids)

        residual, post_mix, res_mix = encoder_carry or (None, None, None)
        pre_mix: torch.Tensor | None = encoder_pre_mix
        if stage_sp and pre_mix is not None and not carry_stage:
            # CED decoder replay: encoder rows arrive unsharded.
            pre_mix = upstream.sp_shard(pre_mix)
        if not upstream.get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
        # Every layer's post runs inside the next layer's fused pre, so aux
        # hidden states are read back from there instead of recomputed.
        aux_hidden_by_layer: dict[int, torch.Tensor] = {}
        if stage == "decoder" and start_layer in self.aux_hidden_state_layers:
            assert hidden_states is not None
            aux_hidden = _mean_hc_streams(hidden_states)
            if stage_sp:
                aux_hidden = upstream.sp_all_gather(aux_hidden)[:full_num_tokens]
            aux_hidden_by_layer[start_layer] = aux_hidden
        with stage_attention_cache_scope():
            for idx, layer in enumerate(
                islice(self.layers, start_layer, end_layer),
                start=start_layer,
            ):
                (
                    hidden_states,
                    residual,
                    post_mix,
                    res_mix,
                    pre_mix,
                    previous_aux,
                ) = layer(
                    hidden_states,
                    positions,
                    input_ids,
                    pre_mix,
                    post_mix,
                    res_mix,
                    residual,
                    engram_hashes,
                    engram_mask,
                    capture_previous_aux=(
                        idx in self.aux_hidden_state_layers and residual is not None
                    ),
                )
                if previous_aux is not None:
                    # idx is the one-based id of the layer whose post this is.
                    if stage_sp:
                        previous_aux = upstream.sp_all_gather(previous_aux)[
                            :full_num_tokens
                        ]
                    aux_hidden_by_layer[idx] = previous_aux
            if stage in ("encoder_head", "encoder_middle"):
                # Hand the next stage the raw carry: its first layer applies
                # this post fused with the Engram injection.
                assert not aux_hidden_by_layer
                assert isinstance(hidden_states, torch.Tensor)
                return hidden_states, residual, post_mix, res_mix, pre_mix
            if layer is not None:
                # The last layer has no successor to fold its post into.
                hidden_states = post(hidden_states, residual, post_mix, res_mix)
                if (
                    stage != "encoder"
                    and self.end_layer in self.aux_hidden_state_layers
                ):
                    final_aux = _mean_hc_streams(hidden_states)
                    if stage_sp:
                        final_aux = upstream.sp_all_gather(final_aux)[:full_num_tokens]
                    aux_hidden_by_layer[self.end_layer] = final_aux

            if stage in ("encoder", "encoder_tail"):
                # Decoder Full-mode global KV depends only on the complete encoder
                # output. Publish it before the host selects the SWA replay rows.
                decoder = self.layers[self._ced_boundary]
                assert pre_mix is not None
                normalized = decoder.attn_norm(collapse(hidden_states, pre_mix))
                decoder.attn.write_encoder_global_kv(
                    normalized, positions, sp_sharded=stage_sp
                )
                encoder_aux = [
                    aux_hidden_by_layer[layer_id]
                    for layer_id in self.aux_hidden_state_layers
                    if layer_id in aux_hidden_by_layer
                ]
                if encoder_aux:
                    return hidden_states, pre_mix, encoder_aux
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
            if stage_sp:
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
        if stage_sp and self._mtp_hidden_buffer is None:
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
        self._raw_dma_enabled = engram_cpu_offload(vllm_config)
        initialize_cache_ops()
        initialize_delayed_mhc()
        initialize_shifted_mhc()
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
                _use_sequence_parallel=_use_sequence_parallel,
            ),
            device_hooks(torch.cuda, Stream=lambda: None),
        ):
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        if spec_cfg is None or getattr(spec_cfg, "method", None) != "mtp":
            # Upstream allocates _mtp_hidden_buffer whenever use_eagle() is
            # True (including dspark). Only deepseek_mtp consumes it.
            self._mtp_hidden_buffer = None
        if self.use_sequence_parallel:
            tp_size = vllm_config.parallel_config.tensor_parallel_size
            sizes = vllm_config.compilation_config.compile_sizes or []
            too_small = sorted(int(s) for s in sizes if 0 < int(s) < 2 * tp_size)
            if too_small:
                # With real weights a 1-token-per-rank bucket halts the TPU
                # core at warmup ([USER] fatal error); >= 2 tokens per rank
                # serves correctly. Root cause not yet isolated.
                raise ValueError(
                    "Sequence parallelism needs every token bucket >= 2 * TP "
                    f"({2 * tp_size}); got {too_small}. Set "
                    "--compilation-config '{\"compile_sizes\": [...]}' accordingly."
                )
            logger.info("DeepSeek V4.1 sequence parallel enabled (TP-sharded tokens)")
        compact_rope_caches(
            (layer.attn for layer in self.layers),
            vllm_config.model_config.max_model_len,
        )
        self._ced_enabled, self._ced_boundary = _ced_config(
            vllm_config, warn_prefix_caching=True, fallback_config=self.config
        )
        self._ced_window = self.layers[self._ced_boundary].attn.window_size
        self._ced_compile_sizes = tuple(
            n
            for n in (vllm_config.compilation_config.compile_sizes or [])
            if isinstance(n, int)
        )
        if self._ced_enabled:
            if (
                getattr(vllm_config.parallel_config, "prefill_context_parallel_size", 1)
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
                    compile_tag="v41_encoder",
                )
            self.encoder_stage._compile_prefix = "v41_encoder"
            with set_model_tag("v41_decoder"):
                self.decoder_stage = DeepseekV41DecoderStage(
                    owner=self,
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.decoder_stage",
                    compile_tag="v41_decoder",
                )
            self.decoder_stage._compile_prefix = "v41_decoder"
            self._tp_size = vllm_config.parallel_config.tensor_parallel_size
            self._engram_split_layer = self._select_engram_split_layer()
            self._engram_first_split_layer = self._select_engram_first_split_layer()
            if self._engram_first_split_layer is None:
                self._engram_split_layer = None
            if self._engram_split_layer is not None:
                assert self._engram_first_split_layer is not None
                assert (
                    self.start_layer
                    < self._engram_first_split_layer
                    < self._engram_split_layer
                    < self._ced_boundary
                )
                with set_model_tag("v41_encoder_middle"):
                    self.encoder_middle_stage = DeepseekV41EncoderMiddleStage(
                        owner=self,
                        vllm_config=vllm_config,
                        prefix=f"{prefix}.encoder_middle_stage",
                        compile_tag="v41_encoder_middle",
                    )
                self.encoder_middle_stage._compile_prefix = "v41_encoder_middle"
                logger.info(
                    "Engram early split: head [%d, %d), middle [%d, %d)",
                    self.start_layer,
                    self._engram_first_split_layer,
                    self._engram_first_split_layer,
                    self._engram_split_layer,
                )
                with set_model_tag("v41_encoder_head"):
                    self.encoder_head_stage = DeepseekV41EncoderHeadStage(
                        owner=self,
                        vllm_config=vllm_config,
                        prefix=f"{prefix}.encoder_head_stage",
                        compile_tag="v41_encoder_head",
                    )
                self.encoder_head_stage._compile_prefix = "v41_encoder_head"
                with set_model_tag("v41_encoder_tail"):
                    self.encoder_tail_stage = DeepseekV41EncoderTailStage(
                        owner=self,
                        vllm_config=vllm_config,
                        prefix=f"{prefix}.encoder_tail_stage",
                        compile_tag="v41_encoder_tail",
                    )
                self.encoder_tail_stage._compile_prefix = "v41_encoder_tail"
                logger.info(
                    "Engram tail split: prefix layers [%d, %d), tail layers "
                    "[%d, %d); tail Engram staging overlaps the prefix",
                    self.start_layer,
                    self._engram_split_layer,
                    self._engram_split_layer,
                    self._ced_boundary,
                )
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
        # Each node-local Engram DP group owns a complete host table. EP
        # still spans the global DP group when the deployment uses more hosts.
        parallel = vllm_config.parallel_config
        if parallel.pipeline_parallel_size != 1:
            raise NotImplementedError("V4.1 TPU currently requires PP1")
        if parallel.data_parallel_size > 1:
            if (
                parallel.tensor_parallel_size != 1
                or not parallel.enable_expert_parallel
                or not engram_cpu_offload(vllm_config)
            ):
                raise NotImplementedError(
                    "V4.1 TPU DP requires TP1, EP and Engram CPU offload"
                )
            engram_dp_size = upstream.get_engram_dp_size()
            if engram_dp_size <= 1 or parallel.data_parallel_size % engram_dp_size != 0:
                raise NotImplementedError(
                    "V4.1 TPU DP requires node-local Engram groups with size > 1 "
                    "dividing the global DP size; refusing per-rank host tables"
                )
            logger.info_once(
                "V4.1 TPU Engram: node-local DP size=%d, global DP size=%d",
                engram_dp_size,
                parallel.data_parallel_size,
            )
        spec_cfg = getattr(vllm_config, "speculative_config", None)
        if spec_cfg is not None and getattr(spec_cfg, "method", None) not in (
            "dspark",
            "ngram",
            "suffix",
        ):
            raise NotImplementedError(
                f"V4.1 TPU does not support speculative method "
                f"{getattr(spec_cfg, 'method', None)!r}"
            )
        if vllm_config.scheduler_config.async_scheduling:
            from .async_fence import install_lookback_fence

            install_lookback_fence()
        if engram_cpu_offload(vllm_config):
            # Host-DRAM Engram tables: hash on the host from the runner's host
            # inputs instead of copying device hashes back every step.
            install_host_engram_inputs()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # TPU's existing FP8 linear adapter registers weight_scale_inv; the
        # NVIDIA MXFP8 adapter selected by upstream uses weight_scale instead.
        self.hf_to_vllm_mapper = upstream._make_deepseek_v4_weights_mapper(
            getattr(self.config, "expert_dtype", "fp4"), "weight_scale_inv"
        )
