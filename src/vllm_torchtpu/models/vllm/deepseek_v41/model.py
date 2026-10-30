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
"""DeepSeek-V4.1 model assembly with 40L CED replay and delayed mHC on TPU."""

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import replace
from functools import reduce
from itertools import islice
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group
from vllm.forward_context import DPMetadata, ForwardContext, override_forward_context
from vllm.model_executor.kernels.mhc.torch import mhc_pre_delayed_torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.models.deepseek_v41.nvidia import model as upstream
from vllm.sequence import IntermediateTensors

from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v4.deepseek_v4_mhc_op import (
    get_mhc_post_op,
)
from vllm_torchtpu.layers.core.attention_metadata import AttentionMetadata
from vllm_torchtpu.utils import (
    RpaSmemPageTableModel,
    TokenLookbackModel,
    synchronize_tensors,
)

from . import ced as ced_mod
from .attention import DeepseekV41Attention
from .engram import Engram, NgramHashState
from .metadata import active_cache, block_table, cache_slots
from .moe import DeepseekV41MoE

_ENC_DYNAMIC_DIMS: dict[str, int] = dict.fromkeys(
    ("input_ids", "positions", "inputs_embeds", "engram_hashes", "engram_mask"), 0
)
_DEC_DYNAMIC_DIMS: dict[str, int] = dict.fromkeys(
    ("input_ids", "positions", "inputs_embeds", "encoder_pre_mix"), 0
)
_FAST_HC_STREAM_COUNT: int = 4
_MHC_POST_ALPHA: float = 2.0
_MHC_MIX_EXTRA_ROWS: int = 2
_MHC_SCALE_DIM: int = 3
_LOOKBACK_PAD_TOKEN_ID: int = -1
StageOutput = (
    Tensor
    | tuple[Tensor, list[Tensor]]
    | tuple[Tensor, Tensor]
    | tuple[Tensor, Tensor, list[Tensor]]
)


def _param(shape: int | tuple[int, ...]) -> nn.Parameter:
    """Allocate a non-trainable ``float32`` hyper-connection parameter tensor.

    Args:
        shape: Parameter tensor shape.

    Returns:
        Uninitialized non-trainable ``float32`` ``nn.Parameter``.
    """
    return nn.Parameter(torch.empty(shape, dtype=torch.float32), requires_grad=False)


def _ced_config(vllm_config: VllmConfig) -> tuple[bool, int]:
    """Resolve ``(ced_enabled, boundary_layer_id)`` from ``vllm_config``.

    Args:
        vllm_config: Active engine configuration (Used).

    Returns:
        Tuple ``(enabled, cut_layer)`` (Created), disabling CED when prefix
        caching is enabled without SWA bounded replay or ``cut_layer`` lies
        outside ``(0, num_hidden_layers)``.
    """
    cache_config = vllm_config.cache_config
    hf_config = vllm_config.model_config.hf_config
    assert cache_config is not None
    prefix_blocks_replay = (
        cache_config.enable_prefix_caching
        and not ced_mod.prefix_replay_enabled(vllm_config)
    )
    cut_layer = max(
        hf_config.kv_source_layer_ids, default=hf_config.num_hidden_layers // 2
    )
    enabled = not prefix_blocks_replay and 0 < cut_layer < hf_config.num_hidden_layers
    return enabled, cut_layer


def _starts_tensor(
    starts: Sequence[int] | Tensor | None,
    num_seqs: int,
    device: torch.device,
) -> Tensor:
    """Materialize per-sequence ``int32[num_seqs]`` replay bounds on ``device``.

    Args:
        starts: Optional host sequence or tensor of per-sequence start positions.
        num_seqs: Number of scheduled sequences when ``starts`` is empty or ``None``.
        device: Target device for the returned tensor.

    Returns:
        Per-sequence start bounds ``int32[num_seqs]`` on ``device``.
    """
    if starts is not None and len(starts) > 0:
        return torch.as_tensor(starts, dtype=torch.int32, device=device)
    return torch.zeros(num_seqs, dtype=torch.int32, device=device)


def _ctx_with_replay(
    forward_ctx: ForwardContext,
    window_starts: Sequence[int] | Tensor | None,
    replay_starts: Sequence[int] | Tensor | None,
    num_seqs: int,
    device: torch.device,
) -> ForwardContext:
    """Return a shallow ``ForwardContext`` copy carrying V4.1 replay start tensors.

    Data lifecycle:
        Used: ``forward_ctx`` and optional host/device start sequences.
        Created: Shallow ``ForwardContext`` copy with updated ``additional_kwargs``.

    Args:
        forward_ctx: Active vLLM forward context.
        window_starts: Per-sequence SWA window lower bounds.
        replay_starts: Per-sequence CED decoder replay lower bounds.
        num_seqs: Number of scheduled sequences in the batch.
        device: Target device for start bound tensors.

    Returns:
        Shallow copy of ``forward_ctx`` with replay start tensors populated.
    """
    window_tensor = _starts_tensor(window_starts, num_seqs, device)
    replay_tensor = (
        window_tensor
        if replay_starts is None
        else _starts_tensor(replay_starts, num_seqs, device)
    )
    replay_kwargs = {
        "deepseek_v41_window_starts": window_tensor,
        "deepseek_v41_replay_starts": replay_tensor,
    }
    updated_ctx = copy(forward_ctx)
    updated_ctx.additional_kwargs = {
        **forward_ctx.additional_kwargs,
        **replay_kwargs,
    }
    return updated_ctx


def _hc_streams(hidden_states: Tensor) -> tuple[Tensor, ...]:
    """Slice ``[num_tokens, hc_mult, dim]`` into ``float32`` 2-D stream views.

    Args:
        hidden_states: Multi-stream residual tensor ``[num_tokens, hc_mult, dim]``.

    Returns:
        Tuple of ``hc_mult`` 2-D ``float32`` stream views ``[num_tokens, dim]``.
    """
    if hidden_states.shape[1] == _FAST_HC_STREAM_COUNT:
        num_tokens, _, hidden_dim = hidden_states.shape
        flat_streams = hidden_states.reshape(
            num_tokens, _FAST_HC_STREAM_COUNT * hidden_dim
        ).float()
        return tuple(
            flat_streams[:, idx * hidden_dim : (idx + 1) * hidden_dim]
            for idx in range(_FAST_HC_STREAM_COUNT)
        )
    return tuple(stream.float() for stream in hidden_states.unbind(dim=1))


def collapse(hidden_states: Tensor, pre_mix: Tensor) -> Tensor:
    """Collapse ``[num_tokens, hc_mult, dim]`` streams into ``[num_tokens, dim]``.

    Args:
        hidden_states: Multi-stream residual ``[num_tokens, hc_mult, dim]`` (Used).
        pre_mix: Per-stream combination weights ``[num_tokens, hc_mult]`` (Used).

    Returns:
        Collapsed 2-D activation tensor ``[num_tokens, dim]`` in ``hidden_states.dtype``
        (Created).
    """
    streams = _hc_streams(hidden_states)
    weights = pre_mix.float()
    weighted_terms = [
        stream * weights[:, idx : idx + 1] for idx, stream in enumerate(streams)
    ]
    return reduce(torch.add, weighted_terms).to(hidden_states.dtype)


def _mean_hc_streams(hidden_states: Tensor) -> Tensor:
    """Average ``[num_tokens, hc_mult, dim]`` hyper-connection streams in 2-D.

    Args:
        hidden_states: Multi-stream residual ``[num_tokens, hc_mult, dim]``.

    Returns:
        Stream-averaged activation ``[num_tokens, dim]`` in ``hidden_states.dtype``.
    """
    streams = _hc_streams(hidden_states)
    mean_stream = reduce(torch.add, streams) * (1.0 / len(streams))
    return mean_stream.to(hidden_states.dtype)


def mhc_post(
    layer_output: Tensor,
    residual: Tensor,
    post_mix: Tensor,
    res_mix: Tensor,
) -> Tensor:
    """Apply the platform 4-stream mHC post-recombination op unconditionally.

    Args:
        layer_output: Sublayer output ``[num_tokens, dim]`` (Used).
        residual: Multi-stream residual ``[num_tokens, hc_mult, dim]`` (Used).
        post_mix: Post-projection mix weights ``[num_tokens, hc_mult, 1]`` (Used).
        res_mix: Stream transition matrix ``[num_tokens, hc_mult, hc_mult]`` (Used).

    Returns:
        Updated multi-stream residual ``[num_tokens, hc_mult, dim]`` (Created).
    """
    post_op = get_mhc_post_op()
    recombined = post_op(
        layer_output.bfloat16(),
        residual.bfloat16(),
        post_mix.float(),
        res_mix.float(),
    )
    return recombined.to(residual.dtype)


class DeepseekV41DecoderLayer(upstream.DeepseekV4DecoderLayer):
    """TPU-adapted DeepSeek-V4.1 decoder layer with delayed mHC and Engram."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        topk_indices_buffer: Tensor | None = None,
        candidate_block_buffer: Tensor | None = None,
        engram_layout: upstream.EngramLayout | None = None,
    ) -> None:
        """Initialize the DeepSeek-V4.1 decoder layer.

        Args:
            vllm_config: Global vLLM engine configuration.
            prefix: Layer parameter prefix.
            topk_indices_buffer: Shared per-token top-K index buffer.
            candidate_block_buffer: Shared coarse candidate block buffer.
            engram_layout: Optional model-wide Engram layout metadata.
        """
        # TODO(upstream): vllm/models/deepseek_v41/nvidia/model.py:309,320,333 hooks.
        nn.Module.__init__(self)
        hf_config = vllm_config.model_config.hf_config
        layer_id = upstream.extract_layer_index(prefix)
        quant_config = vllm_config.quant_config

        self.hidden_size: int = hf_config.hidden_size
        self.rms_norm_eps: float = hf_config.rms_norm_eps
        self.use_sequence_parallel = False
        self.fuse_mhc_all_reduce = False
        self.defers_seam_all_reduce = False
        self.mhc_stream = None

        # pyrefly: ignore[bad-override-mutable-attribute]
        self.engram: Engram | None = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            engram_hash_idx = engram_layout.layer_ids.index(layer_id)
            self.engram = Engram(
                hf_config,
                quant_config,
                engram_layout,
                engram_hash_idx,
                False,
                f"{prefix}.engram",
            )

        # pyrefly: ignore[bad-override-mutable-attribute]
        self.attn: DeepseekV41Attention = DeepseekV41Attention(
            vllm_config,
            f"{prefix}.attn",
            topk_indices_buffer,
            None,
            candidate_block_buffer,
        )
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.ffn: DeepseekV41MoE = DeepseekV41MoE(vllm_config, prefix=f"{prefix}.ffn")
        # pyrefly: ignore[bad-argument-type]
        self.ffn.use_native_mega_moe = False
        self.attn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)
        self.ffn_norm = RMSNorm(self.hidden_size, self.rms_norm_eps)

        self.hc_mult: int = hf_config.hc_mult
        self.hc_eps: float = hf_config.hc_eps
        self.hc_sinkhorn_iters: int = hf_config.hc_sinkhorn_iters
        self.hc_post_alpha: float = _MHC_POST_ALPHA
        mix_rows = (_MHC_MIX_EXTRA_ROWS + self.hc_mult) * self.hc_mult
        mix_cols = self.hc_mult * hf_config.hidden_size
        self.hc_attn_fn = _param((mix_rows, mix_cols))
        self.hc_ffn_fn = _param((mix_rows, mix_cols))
        self.hc_attn_base = _param(mix_rows)
        self.hc_ffn_base = _param(mix_rows)
        self.hc_attn_scale = _param(_MHC_SCALE_DIM)
        self.hc_ffn_scale = _param(_MHC_SCALE_DIM)
        self.register_buffer("hc_attn_fn_broadcast", None, persistent=False)

    def _mhc_pre(
        self,
        residual: Tensor,
        fn_weight: Tensor,
        is_ffn: bool,
        prev_pre_mix: Tensor | None = None,
        prev_x: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run V4.1 delayed 4-output ``mhc_pre_delayed_torch`` + sublayer RMSNorm.

        Args:
            residual: Multi-stream residual ``[num_tokens, hc_mult, dim]``.
            fn_weight: Hyper-connection linear projection weight.
            is_ffn: Whether this pre-step prepares the FFN sublayer (vs attention).
            prev_pre_mix: Delayed pre-mix weights from the preceding sublayer.
            prev_x: Optional collapsed 2-D input on the first layer.

        Returns:
            Tuple ``(post_mix, res_mix, normed_input, next_pre_mix)``.
        """
        # Base get_mhc_ops().pre wraps V4's non-delayed 3-output mhc_pre_torch;
        # V4.1 requires delayed 4-output mhc_pre_delayed_torch.
        scale = (self.hc_ffn_scale if is_ffn else self.hc_attn_scale).float()
        base = (self.hc_ffn_base if is_ffn else self.hc_attn_base).float()
        norm = self.ffn_norm if is_ffn else self.attn_norm
        post_mix, res_mix, raw_input, next_pre_mix = mhc_pre_delayed_torch(
            residual,
            fn_weight.float(),
            scale,
            base,
            self.rms_norm_eps,
            self.hc_eps,
            self.hc_eps,
            self.hc_post_alpha,
            self.hc_sinkhorn_iters,
            prev_pre_mix,
            prev_x,
        )
        return post_mix, res_mix, norm(raw_input), next_pre_mix

    # pyrefly: ignore[bad-override]
    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        input_ids: Tensor | None = None,
        pre_mix: Tensor | None = None,
        post_mix: Tensor | None = None,
        res_mix: Tensor | None = None,
        residual: Tensor | None = None,
        engram_hashes: Tensor | None = None,
        engram_mask: Tensor | None = None,
        capture_previous_aux: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor | None]:
        """Execute optional delayed mHC post + Engram, attention, and MoE sublayers.

        Args:
            hidden_states: Input hidden states or preceding sublayer output.
            positions: Token positions ``int32[num_tokens]``.
            input_ids: Optional token IDs for MoE routing debug/telemetry.
            pre_mix: Delayed pre-mix weights entering the attention sublayer.
            post_mix: Delayed post-mix weights from the preceding FFN sublayer.
            res_mix: Delayed residual transition matrix from the preceding FFN.
            residual: Multi-stream residual from the preceding layer.
            engram_hashes: Optional per-token n-gram hash table ``int64[N, L, K]``.
            engram_mask: Optional boolean mask excluding image sentinel tokens.
            capture_previous_aux: Whether to capture the stream-averaged residual
                after applying the preceding layer's delayed ``mhc_post``.

        Returns:
            Tuple ``(ffn_out, res_state, post_mix, res_mix, ffn_pre, previous_aux)``.
        """
        previous_aux: Tensor | None = None
        if residual is None:
            if hidden_states.dim() == 2:
                assert self.hc_attn_fn_broadcast is not None
                res_state = (
                    hidden_states.unsqueeze(1).expand(-1, self.hc_mult, -1).contiguous()
                )
                attn_fn = self.hc_attn_fn_broadcast
                prev_x: Tensor | None = hidden_states
                attn_pre_in: Tensor | None = None
            else:
                res_state = hidden_states
                attn_fn = self.hc_attn_fn
                prev_x = None
                attn_pre_in = pre_mix
        else:
            assert post_mix is not None and res_mix is not None
            res_state = mhc_post(hidden_states, residual, post_mix, res_mix)
            if capture_previous_aux:
                previous_aux = _mean_hc_streams(res_state)
            if self.engram is not None and engram_hashes is not None:
                engram_col = engram_hashes[:, self.engram.layer_hash_index]
                res_state = self.engram(res_state, engram_col, engram_mask)
            attn_fn = self.hc_attn_fn
            prev_x = None
            attn_pre_in = pre_mix

        post_mix, res_mix, attn_in, attn_pre = self._mhc_pre(
            res_state, attn_fn, False, attn_pre_in, prev_x
        )
        attn_out = self.attn(positions, attn_in, None)
        res_state = mhc_post(attn_out, res_state, post_mix, res_mix)
        post_mix, res_mix, ffn_in, ffn_pre = self._mhc_pre(
            res_state, self.hc_ffn_fn, True, attn_pre
        )
        ffn_out = self.ffn(ffn_in, input_ids)
        return ffn_out, res_state, post_mix, res_mix, ffn_pre, previous_aux


class _StageBase(nn.Module):
    """Compiled stage wrapper delegating forward execution to ``DeepseekV41Model``."""

    # NOTE: Keep ``owner: object`` and ``stage_prefix`` (not ``prefix``) because
    # vllm/compilation/decorators.py:379 shifts positional annotations by 1 and
    # injects ``prefix`` into ``__init__``.
    def __init__(
        self,
        owner: object,
        stage_prefix: str = "",
        stage: str = "full",
        compile_tag: str | None = None,
    ) -> None:
        """Initialize the compiled stage wrapper.

        Args:
            owner: Parent ``DeepseekV41Model`` instance.
            stage_prefix: Default stage module prefix.
            stage: Stage identifier (``"full"``, ``"encoder"``, or ``"decoder"``).
            compile_tag: Optional compilation cache tag override.
        """
        super().__init__()
        object.__setattr__(self, "owner", owner)
        self.owner: object
        self.stage = stage
        self.prefix = stage_prefix if compile_tag is None else compile_tag
        if compile_tag is not None:
            self._compile_prefix = compile_tag


# pyrefly: ignore[no-matching-overload]
@support_torch_compile(dynamic_arg_dims=_ENC_DYNAMIC_DIMS)
class DeepseekV41EncoderStage(_StageBase):
    """Compiled wrapper for full-backbone or CED encoder-half layer execution."""

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        engram_hashes: Tensor | None = None,
        engram_mask: Tensor | None = None,
    ) -> StageOutput:
        """Execute the full or encoder stage on ``self.owner``.

        Args:
            input_ids: Input token IDs ``int32[num_tokens]``, or ``None``.
            positions: Token positions ``int32[num_tokens]``.
            intermediate_tensors: Unused pipeline-parallel tensors.
            inputs_embeds: Optional pre-computed input embeddings.
            engram_hashes: Optional per-token n-gram hashes.
            engram_mask: Optional boolean mask for valid Engram tokens.

        Returns:
            Stage output from ``DeepseekV41Model._forward_stage``.
        """
        del intermediate_tensors
        return cast(DeepseekV41Model, self.owner)._forward_stage(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            engram_hashes=engram_hashes,
            engram_mask=engram_mask,
            encoder_pre_mix=None,
            stage=self.stage,
        )


# pyrefly: ignore[no-matching-overload]
@support_torch_compile(dynamic_arg_dims=_DEC_DYNAMIC_DIMS)
class DeepseekV41DecoderStage(_StageBase):
    """Compiled wrapper for CED decoder-half replay execution."""

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        inputs_embeds: Tensor,
        encoder_pre_mix: Tensor,
    ) -> StageOutput:
        """Execute the CED decoder stage on ``self.owner``.

        Args:
            input_ids: Narrowed decoder token IDs, or ``None``.
            positions: Narrowed decoder positions ``int32[padded_tokens]``.
            inputs_embeds: Narrowed multi-stream residual entering the decoder.
            encoder_pre_mix: Narrowed delayed pre-mix weights from the encoder.

        Returns:
            Stage output from ``DeepseekV41Model._forward_stage``.
        """
        return cast(DeepseekV41Model, self.owner)._forward_stage(
            input_ids,
            positions,
            inputs_embeds=inputs_embeds,
            engram_hashes=None,
            engram_mask=None,
            encoder_pre_mix=encoder_pre_mix,
            stage="decoder",
        )


class DeepseekV41Model(upstream.DeepseekV4Model):
    """DeepSeek-V4.1 backbone with 40L (20+20) CED bounded replay on TPU."""

    @property
    def mhc_post_op(self) -> Callable[..., Tensor]:
        """Return the platform 4-stream mHC post-recombination op.

        Returns:
            Platform mHC post-recombination callable.
        """
        return get_mhc_post_op()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the DeepSeek-V4.1 backbone model.

        Args:
            vllm_config: Global vLLM engine configuration.
            prefix: Model parameter prefix.
        """
        # TODO(upstream): vllm/models/deepseek_v41/nvidia/model.py:694,739,767 hooks.
        nn.Module.__init__(self)
        hf_config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.vllm_config = vllm_config
        self.config = hf_config
        self.quant_config = quant_config
        self.parallel_config = parallel_config
        self.prefix = prefix
        self.vocab_size: int = hf_config.vocab_size
        self.hc_eps: float = hf_config.hc_eps
        self.hc_mult: int = hf_config.hc_mult
        self.hc_dim: int = hf_config.hc_mult * hf_config.hidden_size
        self.rms_norm_eps: float = hf_config.rms_norm_eps
        self.use_native_mega_moe = False
        self.use_sequence_parallel = False

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        topk_buffer = torch.empty(max_tokens, hf_config.index_topk, dtype=torch.int32)
        self.topk_indices_buffer = topk_buffer
        cand_blocks: int = hf_config.candidate_topk_blocks
        has_candidates = hf_config.candidate_source_layer_id >= 0 and cand_blocks > 0
        cand_buffer = (
            torch.empty(max_tokens, cand_blocks, dtype=torch.int32)
            if has_candidates
            else None
        )
        self.candidate_block_buffer = cand_buffer

        self.embed_tokens = VocabParallelEmbedding(
            hf_config.vocab_size,
            hf_config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        layout = upstream.EngramLayout.from_config(hf_config)
        self.engram_layout = layout
        engram_config = vllm_config.engram_config
        if (
            layout is not None
            and engram_config is not None
            and engram_config.dp_shared_memory
        ):
            engram_config.dp_shared_memory = upstream.can_share_engram_tables(layout)
        self.engram_dp_shared_memory = bool(
            engram_config and engram_config.dp_shared_memory
        )

        self.start_layer = 0
        self.end_layer: int = hf_config.num_hidden_layers
        decoder_layers = [
            DeepseekV41DecoderLayer(
                vllm_config,
                f"{prefix}.layers.{layer_idx}",
                topk_buffer,
                cand_buffer,
                layout,
            )
            for layer_idx in range(hf_config.num_hidden_layers)
        ]
        self.layers = nn.ModuleList(decoder_layers)
        # pyrefly: ignore[bad-override-mutable-attribute]
        self.engram_hash: NgramHashState | None = None
        self.engram_swa_prefix: str | None = None
        if layout is not None and any(
            layer.engram is not None for layer in decoder_layers
        ):
            swa_module = decoder_layers[0].attn.swa_cache_layer
            self.engram_hash = NgramHashState(vllm_config, layout, swa_module)
            self.engram_swa_prefix = swa_module.prefix

        self.norm = RMSNorm(hf_config.hidden_size, self.rms_norm_eps)
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self._mtp_hidden_buffer: Tensor | None = None
        self.decoder_replay_layers = None

        ced_enabled, ced_boundary = _ced_config(vllm_config)
        self._ced_enabled = ced_enabled
        self._ced_boundary = ced_boundary
        self._ced_window: int = hf_config.sliding_window
        self._ced_prefix_replay = self._ced_enabled and ced_mod.prefix_replay_enabled(
            vllm_config
        )
        raw_compile_sizes = vllm_config.compilation_config.compile_sizes or []
        self._ced_compile_sizes = tuple(
            int(size) for size in raw_compile_sizes if isinstance(size, int)
        )
        self._ced_dp_sync = parallel_config.data_parallel_size > 1

        if self._ced_enabled:
            boundary_layer = cast(
                DeepseekV41DecoderLayer, self.layers[self._ced_boundary]
            )
            boundary_layer.attn.ced_global_from_encoder = True
            for raw_layer in self.layers[self._ced_boundary :]:
                dec_layer = cast(DeepseekV41DecoderLayer, raw_layer)
                dec_layer.attn.ced_decoder = True
            with set_model_tag("v41_encoder"):
                self.encoder_stage = DeepseekV41EncoderStage(
                    self, "", "encoder", "v41_encoder"
                )
            with set_model_tag("v41_decoder"):
                self.decoder_stage = DeepseekV41DecoderStage(
                    self, "", "decoder", "v41_decoder"
                )
        else:
            self.full_stage = DeepseekV41EncoderStage(
                self, f"{prefix}.full_stage", "full"
            )

    def _prepare_engram(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        lookback_token_ids: Tensor | None,
        stage: str,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute n-gram hashes and stage host-offloaded Engram rows before layers.

        Args:
            input_ids: Input token IDs ``int32[num_tokens]``, or ``None``.
            positions: Token positions ``int32[num_tokens]``.
            lookback_token_ids: Optional ``(M-1)`` lookback token IDs.
            stage: Execution stage name (``"full"`` or ``"encoder"``).

        Returns:
            Tuple ``(engram_hashes, engram_mask)``.
        """
        if self.engram_hash is None or input_ids is None:
            return None, None
        forward_ctx = (
            upstream.get_forward_context()
            if upstream.is_forward_context_available()
            else None
        )
        attn_meta = forward_ctx.attn_metadata if forward_ctx is not None else None
        if isinstance(attn_meta, dict) and self.engram_hash.ensure_cache():
            assert self.engram_swa_prefix is not None
            swa_meta = cast(AttentionMetadata, attn_meta[self.engram_swa_prefix])
            num_reqs = cast(Tensor, swa_meta.seq_lens).shape[0]
            lookback_depth = self.engram_hash.lookback_depth
            lookback = (
                input_ids.new_full((num_reqs, lookback_depth), _LOOKBACK_PAD_TOKEN_ID)
                if lookback_token_ids is None
                else lookback_token_ids
            )
            dead_mask = upstream.image_sentinel_mask(input_ids)
            lookback_dead = upstream.image_sentinel_mask(lookback)
            swa_cache = active_cache(self.engram_hash.swa_cache_module)
            slots = cache_slots(swa_cache, swa_meta, positions)
            hashes = self.engram_hash(
                input_ids,
                positions,
                cast(Tensor, swa_meta.query_start_loc),
                dead_mask,
                lookback,
                lookback_dead,
                slots,
                block_table(swa_meta),
            )
            mask = ~dead_mask
        else:
            hashes, mask = self.engram_hash.dummy_hashes(input_ids)

        should_gather = self.engram_dp_shared_memory or (
            forward_ctx is not None and forward_ctx.dp_metadata is not None
        )
        gathered_hashes = (
            upstream.gather_engram_hashes(
                hashes, dp_shared_memory=self.engram_dp_shared_memory
            )
            if should_gather
            else hashes
        )
        stage_end = self._ced_boundary if stage == "encoder" else self.end_layer
        stage_layers = cast(
            Sequence[DeepseekV41DecoderLayer],
            list(islice(self.layers, self.start_layer, stage_end)),
        )
        engrams = [layer.engram for layer in stage_layers if layer.engram is not None]
        any_offloaded = any(engram_layer.is_staged() for engram_layer in engrams)
        host_or_dev_hashes = (
            gathered_hashes.cpu()
            if gathered_hashes.device.type == "tpu" and any_offloaded
            else gathered_hashes
        )
        staged_tensors: list[Tensor] = []
        for engram_layer in engrams:
            engram_layer.prepare_embeddings(
                host_or_dev_hashes[:, engram_layer.layer_hash_index]
            )
            staged_tensors.extend(engram_layer.staged_device_tensors())
        if staged_tensors and input_ids.device.type == "tpu":
            synchronize_tensors(staged_tensors, wait=True)
        return hashes, mask

    # pyrefly: ignore[bad-override]
    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        lookback_token_ids: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Execute full backbone or 2-stage CED encoder + narrowed decoder replay.

        Args:
            input_ids: Input token IDs ``int32[num_tokens]``, or ``None``.
            positions: Token positions ``int32[num_tokens]``.
            intermediate_tensors: Unused pipeline-parallel tensors.
            inputs_embeds: Optional pre-computed input embeddings.
            lookback_token_ids: Optional ``(M-1)`` lookback token IDs.

        Returns:
            Final hidden states ``[num_tokens, hidden_size]``, or a tuple
            ``(hidden_states, aux_hidden_states)`` when auxiliary layers are configured.
        """
        device = positions.device
        embeds = inputs_embeds
        if (
            embeds is None
            and input_ids is not None
            and device.type == "tpu"
            and self.parallel_config.data_parallel_size > 1
        ):
            embeds = self.embed_input_ids(input_ids)
            synchronize_tensors((embeds,), wait=False)

        forward_ctx = (
            upstream.get_forward_context()
            if upstream.is_forward_context_available()
            else None
        )
        attn_meta = forward_ctx.attn_metadata if forward_ctx is not None else None
        first_meta = (
            cast(AttentionMetadata, next(iter(attn_meta.values())))
            if isinstance(attn_meta, dict) and attn_meta
            else None
        )
        num_reqs = (
            max(cast(Tensor, first_meta.seq_lens).shape[0], 1)
            if first_meta is not None
            else 1
        )
        enc_starts = None if first_meta is None else first_meta.replay_starts_cpu

        if not self._ced_enabled:
            engram_hashes, engram_mask = self._prepare_engram(
                input_ids, positions, lookback_token_ids, "full"
            )
            if forward_ctx is None:
                full_out = self.full_stage(
                    input_ids,
                    positions,
                    intermediate_tensors,
                    embeds,
                    engram_hashes,
                    engram_mask,
                )
            else:
                enc_ctx = _ctx_with_replay(
                    forward_ctx, enc_starts, None, num_reqs, device
                )
                with override_forward_context(enc_ctx):
                    full_out = self.full_stage(
                        input_ids,
                        positions,
                        intermediate_tensors,
                        embeds,
                        engram_hashes,
                        engram_mask,
                    )
            return cast(Tensor | tuple[Tensor, list[Tensor]], full_out)

        assert forward_ctx is not None
        engram_hashes, engram_mask = self._prepare_engram(
            input_ids, positions, lookback_token_ids, "encoder"
        )
        num_tokens = positions.shape[0]
        plan = (
            ced_mod.plan_decoder_replay(
                first_meta, self._ced_window, num_tokens, self._ced_compile_sizes
            )
            if first_meta is not None
            else None
        )
        enc_ctx = _ctx_with_replay(forward_ctx, enc_starts, None, num_reqs, device)
        with override_forward_context(enc_ctx):
            encoded = self.encoder_stage(
                input_ids,
                positions,
                intermediate_tensors,
                embeds,
                engram_hashes,
                engram_mask,
            )
        if self._ced_dp_sync:
            plan = ced_mod.sync_decoder_replay_plan(
                plan,
                first_meta,
                num_tokens,
                get_dp_group().cpu_group,
                window=self._ced_window,
                prefix_replay=self._ced_prefix_replay,
                compile_sizes=self._ced_compile_sizes,
            )

        assert isinstance(encoded, tuple)
        hidden = encoded[0]
        pre_mix = encoded[1]
        enc_aux: list[Tensor] = (
            cast(list[Tensor], encoded[2]) if len(encoded) == 3 else []
        )
        if plan is not None and plan.padded_num_tokens == 0:
            zero_hidden = hidden.new_zeros((num_tokens, self.config.hidden_size))
            zero_aux = [zero_hidden for _ in self.aux_hidden_state_layers]
            return (
                (zero_hidden, zero_aux) if self.aux_hidden_state_layers else zero_hidden
            )

        dec_starts = plan.replay_starts if plan is not None else None
        dec_ctx = _ctx_with_replay(
            forward_ctx, enc_starts, dec_starts, num_reqs, device
        )
        # pyrefly: ignore[bad-assignment]
        dec_ctx.attn_metadata = attn_meta
        dec_pos = positions
        dec_ids = input_ids
        dec_hidden = hidden
        dec_mix = pre_mix
        inverse_rows: Tensor | None = None
        padded_tokens = num_tokens

        if plan is not None and not plan.is_identity:
            assert isinstance(attn_meta, dict)
            last_row = max(num_tokens - 1, 0)
            padded_tokens = plan.padded_num_tokens
            padded_indices = plan.indices + (last_row,) * (
                padded_tokens - len(plan.indices)
            )
            gather_rows = torch.tensor(
                padded_indices, dtype=torch.int32, device=device
            ).clamp_max_(last_row)
            query_bounds = torch.tensor(
                plan.query_start_loc, dtype=torch.int32, device=device
            )
            inverse_rows = ced_mod.build_decoder_inverse_rows(
                plan.indices, padded_tokens, num_tokens, device
            )
            dec_pos = positions.index_select(0, gather_rows)
            dec_hidden = hidden.index_select(0, gather_rows)
            dec_mix = pre_mix.index_select(0, gather_rows)
            dec_ids = (
                None if input_ids is None else input_ids.index_select(0, gather_rows)
            )
            cand_from_encoder = (
                0 <= self.config.candidate_source_layer_id < self._ced_boundary
            )
            cand_buf = self.candidate_block_buffer
            if cand_buf is not None and cand_from_encoder:
                cand_buf[:padded_tokens].copy_(cand_buf.index_select(0, gather_rows))
            dec_ctx.attn_metadata = {
                layer_key: copy(layer_val) for layer_key, layer_val in attn_meta.items()
            }
            for replay_meta in dec_ctx.attn_metadata.values():
                meta_obj = cast(AttentionMetadata, replay_meta)
                meta_obj.input_positions = dec_pos
                meta_obj.query_start_loc = query_bounds
            if forward_ctx.dp_metadata is not None:
                dp_tokens = forward_ctx.dp_metadata.num_tokens_across_dp_cpu
                dec_ctx.dp_metadata = DPMetadata(
                    torch.full_like(dp_tokens, padded_tokens)
                )
            batch_desc = forward_ctx.batch_descriptor
            if batch_desc is not None:
                dec_ctx.batch_descriptor = replace(batch_desc, num_tokens=padded_tokens)

        # NOTE: token_padding exposes set_padding_state/mark_padding without a
        # public getter (see moe_tp_decode.py:605); snapshot and restore the
        # rank-local mask across narrowed decoder replay.
        padding_state = (
            token_padding._padding_state if inverse_rows is not None else None
        )
        saved_padding = (
            padding_state.local_padding_mask.clone()
            if padding_state is not None
            else None
        )
        if padding_state is not None and plan is not None:
            token_padding.mark_padding(len(plan.indices), padded_tokens)
        try:
            with override_forward_context(dec_ctx):
                stage_out = self.decoder_stage(dec_ids, dec_pos, dec_hidden, dec_mix)
        finally:
            if padding_state is not None and saved_padding is not None:
                padding_state.local_padding_mask.copy_(saved_padding)

        if isinstance(stage_out, tuple):
            dec_out = stage_out[0]
            dec_aux = cast(list[Tensor], stage_out[1])
        else:
            dec_out = stage_out
            dec_aux = []
        if inverse_rows is not None:
            dec_out = ced_mod.scatter_decoder_outputs(dec_out, inverse_rows)
            dec_aux = [
                ced_mod.scatter_decoder_outputs(aux_tensor, inverse_rows)
                for aux_tensor in dec_aux
            ]
        all_aux = [*enc_aux, *dec_aux]
        return (dec_out, all_aux) if all_aux else dec_out

    def _forward_stage(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        inputs_embeds: Tensor | None = None,
        engram_hashes: Tensor | None = None,
        engram_mask: Tensor | None = None,
        encoder_pre_mix: Tensor | None = None,
        stage: str = "full",
    ) -> StageOutput:
        """Run ``full``, ``encoder``, or ``decoder`` slice of ``self.layers``.

        Args:
            input_ids: Input token IDs ``int32[num_tokens]``, or ``None``.
            positions: Token positions ``int32[num_tokens]``.
            inputs_embeds: Optional pre-computed embeddings or encoder residual.
            engram_hashes: Optional per-token n-gram hashes.
            engram_mask: Optional boolean mask for valid Engram tokens.
            encoder_pre_mix: Optional delayed pre-mix entering the decoder stage.
            stage: Stage identifier (``"full"``, ``"encoder"``, or ``"decoder"``).

        Returns:
            Stage output tensor or tuple of ``(hidden, [pre_mix,] [aux_list])``.
        """
        assert stage in ("full", "encoder", "decoder"), f"Invalid CED stage: {stage!r}"
        start_layer = self._ced_boundary if stage == "decoder" else self.start_layer
        end_layer = self._ced_boundary if stage == "encoder" else self.end_layer
        if inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            assert input_ids is not None
            hidden = self.embed_input_ids(input_ids)

        residual: Tensor | None = None
        post_mix: Tensor | None = None
        res_mix: Tensor | None = None
        pre_mix = encoder_pre_mix
        aux_by_layer: dict[int, Tensor] = {}
        if stage == "decoder" and start_layer in self.aux_hidden_state_layers:
            aux_by_layer[start_layer] = _mean_hc_streams(hidden)

        for layer_idx, raw_layer in enumerate(
            islice(self.layers, start_layer, end_layer), start=start_layer
        ):
            layer = cast(DeepseekV41DecoderLayer, raw_layer)
            capture_aux = (
                layer_idx in self.aux_hidden_state_layers and residual is not None
            )
            hidden, residual, post_mix, res_mix, pre_mix, prev_aux = layer(
                hidden,
                positions,
                input_ids,
                pre_mix,
                post_mix,
                res_mix,
                residual,
                engram_hashes,
                engram_mask,
                capture_aux,
            )
            if prev_aux is not None:
                aux_by_layer[layer_idx] = prev_aux

        assert residual is not None and post_mix is not None and res_mix is not None
        assert pre_mix is not None
        hidden = mhc_post(hidden, residual, post_mix, res_mix)

        if stage == "encoder":
            dec_boundary = cast(
                DeepseekV41DecoderLayer, self.layers[self._ced_boundary]
            )
            normed_boundary = dec_boundary.attn_norm(collapse(hidden, pre_mix))
            dec_boundary.attn.write_encoder_global_kv(normed_boundary, positions)
            enc_aux_list = [
                aux_by_layer[idx]
                for idx in self.aux_hidden_state_layers
                if idx in aux_by_layer
            ]
            return (
                (hidden, pre_mix, enc_aux_list) if enc_aux_list else (hidden, pre_mix)
            )

        if self.end_layer in self.aux_hidden_state_layers:
            aux_by_layer[self.end_layer] = _mean_hc_streams(hidden)
        hidden = self.norm(collapse(hidden, pre_mix))
        aux_list = [
            aux_by_layer[idx]
            for idx in self.aux_hidden_state_layers
            if idx in aux_by_layer
        ]
        return (hidden, aux_list) if aux_list else hidden


class DeepseekV41ForCausalLM(
    upstream.DeepseekV41LLMForCausalLM, TokenLookbackModel, RpaSmemPageTableModel
):
    """DeepSeek-V4.1 causal LM on TPU implementing ``TokenLookbackModel``."""

    model_cls = DeepseekV41Model
    uses_rpa_smem_page_table = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        """Initialize the DeepSeek-V4.1 causal language model on TPU.

        Args:
            vllm_config: Global vLLM engine configuration.
            prefix: Model parameter prefix.

        Raises:
            NotImplementedError: If pipeline parallelism is enabled, data
                parallelism is enabled without expert parallelism, or an
                unsupported speculative decoding method is configured.
        """
        parallel_config = vllm_config.parallel_config
        dp_without_ep = (
            parallel_config.data_parallel_size != 1
            and not parallel_config.enable_expert_parallel
        )
        if parallel_config.pipeline_parallel_size != 1 or dp_without_ep:
            raise NotImplementedError("V4.1 TPU requires PP=1 and DP=1 (or DP with EP)")
        spec_config = vllm_config.speculative_config
        if spec_config and spec_config.method not in ("dspark", "ngram", "suffix"):
            raise NotImplementedError(f"Unsupported spec method: {spec_config.method}")
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # TPU VllmDeepseekV4Fp8Config registers weight_scale_inv even on [32, 32]
        # FP8 linear weights paired with MXFP4 experts (where upstream defaults
        # to ModelOpt weight_scale), and text-only backbone skips gate.bias_vl.
        weights_mapper = upstream._make_deepseek_v4_weights_mapper(
            self.config.expert_dtype, "weight_scale_inv"
        )
        cast(dict[str, str | None], weights_mapper.orig_to_new_substr)[
            ".ffn.gate.bias_vl"
        ] = None
        self.hf_to_vllm_mapper = weights_mapper

        draft_config = None if spec_config is None else spec_config.draft_model_config
        if (
            spec_config is not None
            and spec_config.method == "dspark"
            and draft_config is not None
        ):
            self._dspark_target_layer_ids: tuple[int, ...] = tuple(
                draft_config.hf_config.dspark_target_layer_ids
            )
        else:
            self._dspark_target_layer_ids = ()

    def set_moe_parameters(self) -> None:
        """Configure routed/shared MoE metadata for EP/TP sharding on TPU."""
        num_routed: int = self.config.n_routed_experts
        parallel_config = self.model.parallel_config
        ep_world = (
            parallel_config.data_parallel_size * parallel_config.tensor_parallel_size
        )
        ep_divisor = ep_world if parallel_config.enable_expert_parallel else 1

        self.num_expert_groups = 1
        self.num_redundant_experts = 0
        self.num_routed_experts = num_routed
        self.num_logical_experts = num_routed
        self.num_physical_experts = num_routed
        self.num_local_physical_experts = num_routed // max(1, ep_divisor)
        self.num_shared_experts = self.config.n_shared_experts or 0
        # pyrefly: ignore[bad-assignment]
        self.moe_mlp_layers = [
            layer.ffn
            for layer in self.model.layers
            if isinstance(layer, DeepseekV41DecoderLayer)
        ]
        self.moe_layers = [moe.experts for moe in self.moe_mlp_layers]
        self.num_moe_layers = len(self.moe_layers)

    def process_weights_after_loading(self) -> None:
        """Precompute first-layer broadcast hyper-connection weights."""
        self.model.finalize_mhc_broadcast_weights()

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        """Shift 1-based DSpark target layer IDs back to 0-based capture indices.

        V4.1 DSpark (``dspark_target_layer_ids=(37, 38, 39)``) conditions on
        the layer inputs to layers 37, 38, 39, which delayed ``mhc_post`` in
        ``DeepseekV41Model._forward_stage`` captures at ``layer_idx`` in
        ``(37, 38, 39)``. Because ``DFlashProposer.load_model`` adds ``+1``
        (passing ``(38, 39, 40)``), shift back to ``(37, 38, 39)``.

        Args:
            layers: Requested 1-based or 0-based auxiliary layer indices.
        """
        if (
            self._dspark_target_layer_ids
            and tuple(idx - 1 for idx in layers) == self._dspark_target_layer_ids
        ):
            layers = self._dspark_target_layer_ids
        super().set_aux_hidden_state_layers(layers)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        lookback_token_ids: Tensor | None = None,
    ) -> Tensor | IntermediateTensors:
        """Ensure Engram cache binding and run ``DeepseekV41Model`` forward.

        Args:
            input_ids: Input token IDs ``int32[num_tokens]``.
            positions: Token positions ``int32[num_tokens]``.
            intermediate_tensors: Unused pipeline-parallel tensors.
            inputs_embeds: Optional pre-computed input embeddings.
            lookback_token_ids: Optional ``(M-1)`` lookback token IDs.

        Returns:
            Model forward output tensor or intermediate tensors.
        """
        if self.model.engram_hash is not None:
            self.model.engram_hash.ensure_cache()
        return super().forward(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
            lookback_token_ids=lookback_token_ids,
        )
