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
"""TPU implementation of the DeepSeek-V4.1 model."""

import copy
import dataclasses
import os
import re
import typing
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.model_executor.offloader import NoopOffloader, set_offloader
from vllm.sequence import IntermediateTensors

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_shard_slice,
    tp_token_shard_ok,
)
from vllm_torchtpu.kernels.deepseek_v4.mhc.utils import split_fn3_torch
from vllm_torchtpu.layers.adapter import token_padding
from vllm_torchtpu.layers.adapter.custom_ops.deepseek_v41.deepseek_v41_flash_mhc_op import (  # noqa: E501
    MHCOpsV41,
    get_mhc_final_collapse_norm_v41_op,
    get_mhc_ops_v41,
    get_mhc_post_collapse_norm_v41_op,
    get_mhc_post_v41_op,
    mhc_collapse_v41,
)
from vllm_torchtpu.models.vllm.deepseek_v4.model import (
    DeepseekV4ForCausalLM,
    DeepseekV4Model,
    _HostStagedParams,
)

_TOKEN_SHARD_MIN_TOKENS = envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS
from vllm_torchtpu.models.vllm.deepseek_v4.moe import DeepseekV4MoE
from vllm_torchtpu.models.vllm.deepseek_v41.attention import VllmDeepseekV41MLAAttention
from vllm_torchtpu.models.vllm.deepseek_v41.ced import plan_decoder_replay
from vllm_torchtpu.models.vllm.deepseek_v41.engram import (
    EngramLayout,
    TpuEngram,
    TpuNgramHashState,
    _engram_table_bytes,
    _EngramTableSlices,
    engram_prefetch_prefill,
    take_prefetched_engram_hashes,
)
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)


def _ced_gather_inputs_impl(
    hidden_states: torch.Tensor,
    pre_mix: torch.Tensor,
    positions: torch.Tensor,
    gather_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        hidden_states.index_select(0, gather_idx),
        pre_mix.index_select(0, gather_idx),
        positions.index_select(0, gather_idx),
    )


def _ced_scatter_output_impl(
    decoder_out: torch.Tensor,
    scatter_idx: torch.Tensor,
) -> torch.Tensor:
    padded_out = torch.cat((decoder_out, torch.zeros((1, *decoder_out.shape[1:]), dtype=decoder_out.dtype, device=decoder_out.device)), dim=0)
    safe_idx = torch.where(
        scatter_idx >= 0,
        scatter_idx,
        torch.full_like(scatter_idx, decoder_out.shape[0]),
    )
    return padded_out.index_select(0, safe_idx)


def _ced_zero_output_impl(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(hidden_states[:, 0, :])


_ced_tpu_compile = torch.compile(backend="tpu", fullgraph=True, dynamic=False)
_ced_gather_inputs_compiled = _ced_tpu_compile(_ced_gather_inputs_impl)
_ced_scatter_output_compiled = _ced_tpu_compile(_ced_scatter_output_impl)
_ced_zero_output_compiled = _ced_tpu_compile(_ced_zero_output_impl)


def _ced_gather_inputs(
    hidden_states: torch.Tensor,
    pre_mix: torch.Tensor,
    positions: torch.Tensor,
    gather_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fn = (
        _ced_gather_inputs_compiled
        if hidden_states.device.type == "tpu" and not torch.compiler.is_compiling()
        else _ced_gather_inputs_impl
    )
    return fn(hidden_states, pre_mix, positions, gather_idx)


def _ced_scatter_output(
    decoder_out: torch.Tensor,
    scatter_idx: torch.Tensor,
) -> torch.Tensor:
    fn = (
        _ced_scatter_output_compiled
        if decoder_out.device.type == "tpu" and not torch.compiler.is_compiling()
        else _ced_scatter_output_impl
    )
    return fn(decoder_out, scatter_idx)


def _ced_zero_output(hidden_states: torch.Tensor) -> torch.Tensor:
    fn = (
        _ced_zero_output_compiled
        if hidden_states.device.type == "tpu" and not torch.compiler.is_compiling()
        else _ced_zero_output_impl
    )
    return fn(hidden_states)


def _moe_vllm_config(
    vllm_config: VllmConfig, moe_overrides: dict | None = None
) -> VllmConfig:
    """Shallow config view for the shared V4 MoE, which reads hf_config."""
    hf_config = copy.copy(vllm_config.model_config.hf_config)
    hf_config.num_hash_layers = 0
    if moe_overrides:
        for k, v in moe_overrides.items():
            setattr(hf_config, k, v)
    model_config = copy.copy(vllm_config.model_config)
    model_config.hf_config = hf_config
    moe_vllm_config = copy.copy(vllm_config)
    moe_vllm_config.model_config = model_config
    return moe_vllm_config


class _PurePrefillBoundaryLayer(nn.Module):
    """Minimal L20 module for VLLM_DSV41_PURE_20L_PREFILL (`attn` + `attn_norm` only)."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor,
        candidate_block_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.attn = VllmDeepseekV41MLAAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            candidate_block_buffer=candidate_block_buffer,
        )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = None
        self.engram = None


class DeepseekV41DecoderLayer(nn.Module):
    """DeepSeek-V4.1 decoder layer with Single-Pass Delayed mHC, CSA2, and Engram."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        topk_indices_buffer: torch.Tensor,
        engram_layout: EngramLayout | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        moe_overrides: dict | None = None,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        layer_idx = extract_layer_index(prefix)
        self.layer_id = layer_idx
        self.is_first_stream_layer = layer_idx in (0, config.num_hidden_layers)
        tp_size = get_tensor_model_parallel_world_size()
        self.attn = VllmDeepseekV41MLAAttention(
            vllm_config=vllm_config,
            prefix=f"{prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            candidate_block_buffer=candidate_block_buffer,
        )
        self.ffn = DeepseekV4MoE(
            _moe_vllm_config(vllm_config, moe_overrides=moe_overrides),
            prefix=f"{prefix}.ffn",
            reduce_results=(tp_size <= 1),
        )
        self.engram: TpuEngram | None = None
        if engram_layout is not None and layer_idx in engram_layout.layer_ids:
            self.engram = TpuEngram(
                config=config,
                quant_config=vllm_config.quant_config,
                layout=engram_layout,
                layer_hash_index=engram_layout.layer_ids.index(layer_idx),
                prefix=f"{prefix}.engram",
            )
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rms_norm_eps = config.rms_norm_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = getattr(config, "hc_sinkhorn_iters", 20)
        self.hc_eps = getattr(config, "hc_eps", 1e-6)
        self.hc_post_mult = getattr(config, "hc_post_mult", 2.0)

        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * config.hidden_size
        self.hc_attn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(mix_hc, hc_dim, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(mix_hc, dtype=torch.float32), requires_grad=False
        )
        self.hc_attn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        self.hc_ffn_scale = nn.Parameter(
            torch.empty(3, dtype=torch.float32), requires_grad=False
        )
        _ = self.mhc_ops

    def _ensure_mhc_buffers(self, force: bool = False) -> None:
        if not force and getattr(self, "_hc_attn_fn_split", None) is not None:
            return
        m = self.hc_mult
        counts = torch.tensor([m, m, m * m], dtype=torch.int64)
        attn_fn_split = split_fn3_torch(self.hc_attn_fn)
        ffn_fn_split = split_fn3_torch(self.hc_ffn_fn)
        attn_scale_vec = (
            torch.repeat_interleave(
                self.hc_attn_scale.detach().to(device="cpu", dtype=torch.float32),
                counts,
            )
            .reshape(1, -1)
            .clone()
            .contiguous()
            .to(device=self.hc_attn_scale.device)
        )
        ffn_scale_vec = (
            torch.repeat_interleave(
                self.hc_ffn_scale.detach().to(device="cpu", dtype=torch.float32),
                counts,
            )
            .reshape(1, -1)
            .clone()
            .contiguous()
            .to(device=self.hc_ffn_scale.device)
        )
        attn_base_2d = (
            self.hc_attn_base.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(1, -1)
            .clone()
            .contiguous()
            .to(device=self.hc_attn_base.device)
        )
        ffn_base_2d = (
            self.hc_ffn_base.detach()
            .to(device="cpu", dtype=torch.float32)
            .reshape(1, -1)
            .clone()
            .contiguous()
            .to(device=self.hc_ffn_base.device)
        )
        self.register_buffer("_hc_attn_fn_split", attn_fn_split, persistent=False)
        self.register_buffer("_hc_ffn_fn_split", ffn_fn_split, persistent=False)
        self.register_buffer("_hc_attn_scale_vec", attn_scale_vec, persistent=False)
        self.register_buffer("_hc_ffn_scale_vec", ffn_scale_vec, persistent=False)
        self.register_buffer("_hc_attn_base_2d", attn_base_2d, persistent=False)
        self.register_buffer("_hc_ffn_base_2d", ffn_base_2d, persistent=False)
        if self.hc_attn_fn.device.type == "tpu":
            from torch_tpu._internal import sync

            sync.synchronize(
                [
                    self._hc_attn_fn_split,
                    self._hc_ffn_fn_split,
                    self._hc_attn_scale_vec,
                    self._hc_ffn_scale_vec,
                    self._hc_attn_base_2d,
                    self._hc_ffn_base_2d,
                ],
                wait=True,
            )

    @property
    def mhc_ops(self) -> MHCOpsV41:
        ops = self.__dict__.get("_mhc_ops_instance")
        if ops is None:
            _ = get_mhc_post_v41_op()
            _ = get_mhc_post_collapse_norm_v41_op(self.rms_norm_eps)
            _ = get_mhc_final_collapse_norm_v41_op(self.rms_norm_eps)
            ops = get_mhc_ops_v41(
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=self.hc_post_mult,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                use_stream0=self.is_first_stream_layer,
            )
            object.__setattr__(self, "_mhc_ops_instance", ops)
        return ops

    def forward(
        self,
        positions: torch.Tensor,
        residual: torch.Tensor,
        pre_mix: torch.Tensor,
        previous_x: torch.Tensor | None = None,
        previous_post_mix: torch.Tensor | None = None,
        previous_res_mix: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        token_sharded: bool = False,
        ced_decoder: bool | None = None,
        ced_global_from_encoder: bool | None = None,
        tp_rank_idx: torch.Tensor | None = None,
        capture_entry_aux: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Returns (ffn_out, residual, ffn_pre, ffn_post, ffn_comb), plus the
        layer's entry stream averaged over the hc copies when
        `capture_entry_aux` (the DSpark target hidden state, as in the
        reference `main_hiddens.append(h.mean(dim=2))` before the layer)."""
        ops = self.mhc_ops
        if (
            getattr(self, "_hc_attn_fn_split", None) is None
            and not torch.compiler.is_compiling()
        ):
            self._ensure_mhc_buffers()
        hc_attn_fn = getattr(self, "_hc_attn_fn_split", None)
        if hc_attn_fn is None:
            hc_attn_fn = self.hc_attn_fn
            hc_attn_scale = self.hc_attn_scale
            hc_attn_base = self.hc_attn_base
            hc_ffn_fn = self.hc_ffn_fn
            hc_ffn_scale = self.hc_ffn_scale
            hc_ffn_base = self.hc_ffn_base
        else:
            hc_attn_scale = self._hc_attn_scale_vec
            hc_attn_base = self._hc_attn_base_2d
            hc_ffn_fn = self._hc_ffn_fn_split
            hc_ffn_scale = self._hc_ffn_scale_vec
            hc_ffn_base = self._hc_ffn_base_2d

        use_prequant_attn = self.attn.can_accept_prequant_hidden(
            positions.numel(),
            token_sharded=token_sharded,
            ced_global_from_encoder=ced_global_from_encoder,
        )
        hidden_states_fp8: torch.Tensor | None = None
        hidden_states_scale: torch.Tensor | None = None
        engram_fused_done = False
        if self.engram is not None:
            if (
                engram_hashes is not None
                and previous_x is not None
                and (
                    ops.fused_engram_quant is not None
                    if use_prequant_attn
                    else ops.fused_engram is not None
                )
            ):
                assert previous_post_mix is not None and previous_res_mix is not None
                engram_kv, engram_qk_w, engram_sharded = (
                    self.engram.compute_kv_and_weight(
                        residual.shape[0],
                        engram_hashes[:, self.engram.layer_hash_index],
                        token_sharded=token_sharded,
                    )
                )
                if engram_sharded:
                    engram_kv = tp_all_gather(engram_kv, dim=0)
                if use_prequant_attn:
                    (
                        residual,
                        attn_post,
                        attn_comb,
                        hidden_states,
                        attn_pre,
                        hidden_states_fp8,
                        hidden_states_scale,
                    ) = ops.fused_engram_quant(
                        previous_x,
                        residual,
                        previous_post_mix,
                        previous_res_mix,
                        pre_mix,
                        engram_kv,
                        engram_qk_w,
                        self.attn_norm.weight,
                        hc_attn_fn,
                        hc_attn_scale,
                        hc_attn_base,
                    )
                else:
                    residual, attn_post, attn_comb, hidden_states, attn_pre = (
                        ops.fused_engram(
                            previous_x,
                            residual,
                            previous_post_mix,
                            previous_res_mix,
                            pre_mix,
                            engram_kv,
                            engram_qk_w,
                            self.attn_norm.weight,
                            hc_attn_fn,
                            hc_attn_scale,
                            hc_attn_base,
                        )
                    )
                engram_fused_done = True
            if not engram_fused_done:
                if previous_x is not None:
                    assert previous_post_mix is not None and previous_res_mix is not None
                    residual = ops.post(
                        previous_x, residual, previous_post_mix, previous_res_mix
                    )
                    previous_x = None
                if engram_hashes is not None:
                    residual = self.engram(
                        residual,
                        engram_hashes[:, self.engram.layer_hash_index],
                        token_sharded=token_sharded,
                    )

        if engram_fused_done:
            pass
        elif previous_x is None:
            if use_prequant_attn:
                (
                    attn_post,
                    attn_comb,
                    hidden_states,
                    attn_pre,
                    hidden_states_fp8,
                    hidden_states_scale,
                ) = ops.pre_quant(
                    residual,
                    pre_mix,
                    self.attn_norm.weight,
                    hc_attn_fn,
                    hc_attn_scale,
                    hc_attn_base,
                )
            else:
                attn_post, attn_comb, hidden_states, attn_pre = ops.pre(
                    residual,
                    pre_mix,
                    self.attn_norm.weight,
                    hc_attn_fn,
                    hc_attn_scale,
                    hc_attn_base,
                )
        else:
            assert previous_post_mix is not None and previous_res_mix is not None
            if use_prequant_attn:
                seam_a_op = (
                    ops.fused_quant_fp8_only
                    if getattr(ops, "fused_quant_fp8_only", None) is not None
                    else ops.fused_quant
                )
                (
                    residual,
                    attn_post,
                    attn_comb,
                    hidden_states,
                    attn_pre,
                    hidden_states_fp8,
                    hidden_states_scale,
                ) = seam_a_op(
                    previous_x,
                    residual,
                    previous_post_mix,
                    previous_res_mix,
                    pre_mix,
                    self.attn_norm.weight,
                    hc_attn_fn,
                    hc_attn_scale,
                    hc_attn_base,
                )
            else:
                residual, attn_post, attn_comb, hidden_states, attn_pre = ops.fused(
                    previous_x,
                    residual,
                    previous_post_mix,
                    previous_res_mix,
                    pre_mix,
                    self.attn_norm.weight,
                    hc_attn_fn,
                    hc_attn_scale,
                    hc_attn_base,
                )

        # `residual` is now this layer's entry stream (previous layer's post
        # mix applied, plus engram if any).
        entry_aux = residual.mean(dim=-2) if capture_entry_aux else None

        if token_sharded:
            x = self.attn(
                positions,
                hidden_states,
                input_is_token_sharded=True,
                return_token_sharded=True,
                ced_decoder=ced_decoder,
                ced_global_from_encoder=ced_global_from_encoder,
                tp_rank_idx=tp_rank_idx,
                hidden_states_fp8=hidden_states_fp8,
                hidden_states_scale=hidden_states_scale,
            )
        else:
            x = self.attn(
                positions,
                hidden_states,
                ced_decoder=ced_decoder,
                ced_global_from_encoder=ced_global_from_encoder,
                tp_rank_idx=tp_rank_idx,
                hidden_states_fp8=hidden_states_fp8,
                hidden_states_scale=hidden_states_scale,
            )

        use_prequant_ffn = (
            ops.fused_quant is not None
            and getattr(self.ffn, "can_accept_prequant_hidden", lambda **_: False)(
                token_sharded=token_sharded
            )
        )
        if use_prequant_ffn:
            (
                residual,
                ffn_post,
                ffn_comb,
                hidden_states,
                ffn_pre,
                ffn_fp8,
                ffn_scale,
            ) = ops.fused_quant(
                x,
                residual,
                attn_post,
                attn_comb,
                attn_pre,
                self.ffn_norm.weight,
                hc_ffn_fn,
                hc_ffn_scale,
                hc_ffn_base,
            )
            if token_sharded:
                ffn_out = self.ffn(
                    hidden_states,
                    return_token_sharded=True,
                    hidden_states_fp8=ffn_fp8,
                    hidden_states_scale=ffn_scale,
                    input_is_token_sharded=True,
                )
            else:
                ffn_out = self.ffn(
                    hidden_states,
                    hidden_states_fp8=ffn_fp8,
                    hidden_states_scale=ffn_scale,
                )
            outs = (ffn_out, residual, ffn_pre, ffn_post, ffn_comb)
            return (*outs, entry_aux) if capture_entry_aux else outs
        residual, ffn_post, ffn_comb, hidden_states, ffn_pre = ops.fused(
            x,
            residual,
            attn_post,
            attn_comb,
            attn_pre,
            self.ffn_norm.weight,
            hc_ffn_fn,
            hc_ffn_scale,
            hc_ffn_base,
        )
        if token_sharded:
            ffn_out = self.ffn(
                hidden_states,
                return_token_sharded=True,
                input_is_token_sharded=True,
            )
        else:
            ffn_out = self.ffn(hidden_states)
        outs = (ffn_out, residual, ffn_pre, ffn_post, ffn_comb)
        return (*outs, entry_aux) if capture_entry_aux else outs


def _gather_aux_hidden_states(
    aux_hidden_states: list[torch.Tensor], token_sharded_mhc: bool
) -> list[torch.Tensor]:
    if not aux_hidden_states or not token_sharded_mhc:
        return aux_hidden_states
    if len(aux_hidden_states) == 1:
        return [tp_all_gather(aux_hidden_states[0], dim=0)]
    gathered = tp_all_gather(torch.cat(aux_hidden_states, dim=-1), dim=0)
    return list(gathered.chunk(len(aux_hidden_states), dim=-1))


def _run_layer(
    layer: nn.Module,
    aux_hidden_state_layers: tuple[int, ...],
    aux_hidden_states: list[torch.Tensor],
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, ...]:
    """Run one decoder layer, appending its entry stream when it is a
    DSpark/eagle target layer."""
    if aux_hidden_state_layers and getattr(layer, "layer_id", None) in aux_hidden_state_layers:
        *outs, entry_aux = layer(*args, capture_entry_aux=True, **kwargs)
        aux_hidden_states.append(entry_aux)
        return tuple(outs)
    return layer(*args, **kwargs)


def _clear_stage_caches(
    layers: Iterable[nn.Module], *, keep_indexer_metadata: bool = False
) -> None:
    for layer in layers:
        attn = getattr(layer, "attn", None)
        if attn is not None:
            if not keep_indexer_metadata:
                attn._step_indexer_metadata = None
            attn._step_csa_union = None
            attn._step_swa_metadata = None
            attn._step_swa_metadata_by_pages = None
            attn._step_swa_win_metadata_by_pages = None
            attn._step_tp_swa_metadata = None
            attn._step_cos_sin = None
            attn._step_cos_sin_local = None


def _run_engram_prelude_and_layers(
    *,
    embed_tokens: nn.Module,
    layers: nn.ModuleList,
    engram_hash: TpuNgramHashState | None,
    engram_table_ids: list[int],
    engram_packed_dim: int,
    tp_rank_idx: torch.Tensor,
    init_pre_mix: torch.Tensor,
    hc_mult: int,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None,
    precomputed_engram_hashes: torch.Tensor | None = None,
    aux_hidden_state_layers: tuple[int, ...] = (),
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    bool,
    list[torch.Tensor],
]:
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1 and _TOKEN_SHARD_MIN_TOKENS > 0:
        # The threshold splits the bucket ladder into replicated and
        # token-sharded shape_variants traces; lift tp_rank_idx first in both
        # so their FX placeholder signatures agree (the alias is DCE'd).
        _ = torch.ops.aten.alias.default(tp_rank_idx)
    engram_hashes_raw: torch.Tensor | None = None
    engram_hashes: torch.Tensor | None = None
    if engram_hash is not None:
        assert input_ids is not None
        assert input_ids.shape[0] == positions.shape[0]
        tp_shard_engram = tp_token_shard_ok(input_ids.shape[0], tp_size)
        all_attn_metadata = get_forward_context().attn_metadata
        if (
            isinstance(all_attn_metadata, dict)
            and engram_hash.ensure_cache()
            and (
                precomputed_engram_hashes is None
                or envs.VLLM_DSV41_ENGRAM_H2D_PREFIX
                or envs.VLLM_DSV41_PURE_20L_PREFILL
            )
        ):
            swa_meta = all_attn_metadata[engram_hash.swa_cache_module.prefix]
            engram_hashes_raw = engram_hash(
                input_ids,
                positions,
                swa_meta.query_start_loc,
                swa_meta.block_tables,
                swa_meta.request_distribution,
            )
        elif precomputed_engram_hashes is None:
            engram_hashes_raw = engram_hash.dummy_hashes(input_ids)
        if precomputed_engram_hashes is not None:
            t_local = (
                input_ids.shape[0] // tp_size
                if tp_shard_engram
                else input_ids.shape[0]
            )
            n_hash_cols = int(engram_hash.offsets.shape[1])
            table_bytes = _engram_table_bytes(n_hash_cols, engram_packed_dim)
            if (
                precomputed_engram_hashes.ndim == 2
                and precomputed_engram_hashes.dtype == torch.uint8
            ):
                engram_hashes = _EngramTableSlices(
                    precomputed_engram_hashes, t_local, table_bytes
                )
            elif (
                precomputed_engram_hashes.ndim == 1
                and precomputed_engram_hashes.dtype == torch.uint8
            ):
                n_bytes = t_local * len(engram_table_ids) * table_bytes
                if engram_packed_dim >= n_hash_cols * 64:
                    engram_hashes = _EngramTableSlices(
                        precomputed_engram_hashes.narrow(0, 0, n_bytes).view(
                            t_local, len(engram_table_ids) * table_bytes
                        ),
                        t_local,
                        table_bytes,
                    )
                else:
                    engram_hashes = precomputed_engram_hashes.narrow(
                        0, 0, n_bytes
                    ).view(
                        t_local,
                        len(engram_table_ids),
                        n_hash_cols,
                        engram_packed_dim,
                    )
            else:
                engram_hashes = precomputed_engram_hashes.view(
                    -1,
                    len(engram_table_ids),
                    n_hash_cols,
                    engram_packed_dim,
                )
        else:
            if tp_shard_engram:
                tp_rank = get_tensor_model_parallel_rank()
                engram_hashes_raw = tp_shard_slice(
                    engram_hashes_raw, tp_size, tp_rank, tp_rank_idx
                )
            torch.ops.tpu.engram_hash_ready(engram_hashes_raw)

    positions = torch.ops.aten.alias.default(positions)
    num_tokens = positions.shape[0]
    token_sharded_mhc = tp_token_shard_ok(num_tokens, tp_size)

    if get_pp_group().is_first_rank:
        hidden_states = (
            inputs_embeds if inputs_embeds is not None else embed_tokens(input_ids)  # type: ignore[arg-type]
        )
        if token_sharded_mhc:
            tp_rank = get_tensor_model_parallel_rank()
            hidden_states = tp_shard_slice(hidden_states, tp_size, tp_rank, tp_rank_idx)
        pre_mix = hidden_states[:, :1].to(torch.float32) * 0.0 + init_pre_mix
        hidden_states = hidden_states.repeat(1, hc_mult).view(
            -1, hc_mult, hidden_states.shape[-1]
        )
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        pre_mix = intermediate_tensors["pre_mix"]

    previous_x, previous_post_mix, previous_res_mix = None, None, None
    aux_hidden_states: list[torch.Tensor] = []
    for layer in layers:
        if (
            layer.engram is not None
            and engram_hashes is None
            and engram_hashes_raw is not None
            and engram_hash is not None
        ):
            n_tables = len(engram_table_ids)
            n_hash_cols = engram_hash.offsets.shape[1]
            if previous_x is not None:
                assert previous_post_mix is not None and previous_res_mix is not None
                hidden_states = layer.mhc_ops.post(
                    previous_x,
                    hidden_states,
                    previous_post_mix,
                    previous_res_mix,
                )
                previous_x = None
                previous_post_mix = None
                previous_res_mix = None
            _clear_stage_caches(layers, keep_indexer_metadata=True)
            engram_hashes_flat = torch.ops.tpu.engram_cpu_lookup_all(
                engram_hashes_raw,
                engram_table_ids,
                engram_packed_dim,
            )
            positions = torch.ops.aten.alias.default(positions)
            table_bytes = _engram_table_bytes(int(n_hash_cols), engram_packed_dim)
            if engram_packed_dim >= int(n_hash_cols) * 64:
                t_lookup = engram_hashes_raw.shape[0]
                engram_hashes = _EngramTableSlices(
                    engram_hashes_flat.view(t_lookup, n_tables * table_bytes),
                    t_lookup,
                    table_bytes,
                )
            else:
                engram_hashes = engram_hashes_flat.view(
                    -1, n_tables, n_hash_cols, engram_packed_dim
                )
        (
            previous_x,
            hidden_states,
            pre_mix,
            previous_post_mix,
            previous_res_mix,
        ) = _run_layer(
            layer,
            aux_hidden_state_layers,
            aux_hidden_states,
            positions,
            hidden_states,
            pre_mix,
            previous_x=previous_x,
            previous_post_mix=previous_post_mix,
            previous_res_mix=previous_res_mix,
            engram_hashes=engram_hashes,
            token_sharded=token_sharded_mhc,
            tp_rank_idx=tp_rank_idx,
        )
    return (
        positions,
        hidden_states,
        pre_mix,
        previous_x,
        previous_post_mix,
        previous_res_mix,
        token_sharded_mhc,
        aux_hidden_states,
    )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class DeepseekV41FullStage(nn.Module):
    """Compiled full 40-layer stage when CED is disabled."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        embed_tokens: nn.Module,
        layers: Iterable[nn.Module],
        norm: nn.Module,
        engram_hash: TpuNgramHashState | None,
        engram_table_ids: list[int],
        engram_packed_dim: int,
        tp_rank_idx: torch.Tensor,
        init_pre_mix: torch.Tensor,
        hc_mult: int,
        single_backbone_cond: bool = False,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hc_mult = hc_mult
        self.rms_norm_eps = float(vllm_config.model_config.hf_config.rms_norm_eps)
        self._single_backbone_cond = single_backbone_cond
        self._engram_table_ids = engram_table_ids
        self._engram_packed_dim = engram_packed_dim
        # Layer ids whose entry stream feeds the DSpark drafter (set by
        # DeepseekV41Model._set_aux_hidden_state_layers before warmup).
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(layers)
        self.engram_hash = engram_hash
        self.norm = norm
        self.register_buffer("tp_rank_idx", tp_rank_idx, persistent=False)
        self.register_buffer("init_pre_mix", init_pre_mix, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        run_upper_layers: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if self._single_backbone_cond:
            from vllm_torchtpu.models.vllm.deepseek_v41.cond_backbone import (
                _dsv41_cond_upper_boundary_op,
            )

            ced_boundary = len(self.layers) // 2
            enc_layers = self.layers[:ced_boundary]
            dec_layers = self.layers[ced_boundary:]
            (
                positions,
                hidden_states,
                pre_mix,
                previous_x,
                previous_post_mix,
                previous_res_mix,
                token_sharded_mhc,
                aux_hidden_states,
            ) = _run_engram_prelude_and_layers(
                embed_tokens=self.embed_tokens,
                layers=enc_layers,
                engram_hash=self.engram_hash,
                engram_table_ids=self._engram_table_ids,
                engram_packed_dim=self._engram_packed_dim,
                tp_rank_idx=self.tp_rank_idx,
                init_pre_mix=self.init_pre_mix,
                hc_mult=self.hc_mult,
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                precomputed_engram_hashes=engram_hashes,
                aux_hidden_state_layers=self.aux_hidden_state_layers,
            )
            _clear_stage_caches(enc_layers)
            hidden_states, normalized = get_mhc_post_collapse_norm_v41_op(
                self.rms_norm_eps
            )(
                previous_x,
                hidden_states,
                previous_post_mix,
                previous_res_mix,
                pre_mix,
                self.layers[ced_boundary].attn_norm.weight,
            )
            self.layers[ced_boundary].attn.write_encoder_global_kv(
                normalized, positions
            )
            assert run_upper_layers is not None
            hidden_states, pre_mix = _dsv41_cond_upper_boundary_op(
                hidden_states, pre_mix, run_upper_layers
            )
            previous_x, previous_post_mix, previous_res_mix = None, None, None
            for idx, layer in enumerate(dec_layers):
                (
                    previous_x,
                    hidden_states,
                    pre_mix,
                    previous_post_mix,
                    previous_res_mix,
                ) = _run_layer(
                    layer,
                    self.aux_hidden_state_layers,
                    aux_hidden_states,
                    positions,
                    hidden_states,
                    pre_mix,
                    previous_x=previous_x,
                    previous_post_mix=previous_post_mix,
                    previous_res_mix=previous_res_mix,
                    engram_hashes=None,
                    token_sharded=token_sharded_mhc,
                    ced_decoder=True,
                    ced_global_from_encoder=(idx == 0),
                    tp_rank_idx=self.tp_rank_idx,
                )
            _clear_stage_caches(dec_layers)
            out = get_mhc_final_collapse_norm_v41_op(self.rms_norm_eps)(
                previous_x,
                hidden_states,
                previous_post_mix,
                previous_res_mix,
                pre_mix,
                self.norm.weight,
            )
            if token_sharded_mhc:
                out = tp_all_gather(out, dim=0)
            if self.aux_hidden_state_layers:
                return out, _gather_aux_hidden_states(
                    aux_hidden_states, token_sharded_mhc
                )
            return out

        (
            positions,
            hidden_states,
            pre_mix,
            previous_x,
            previous_post_mix,
            previous_res_mix,
            token_sharded_mhc,
            aux_hidden_states,
        ) = _run_engram_prelude_and_layers(
            embed_tokens=self.embed_tokens,
            layers=self.layers,
            engram_hash=self.engram_hash,
            engram_table_ids=self._engram_table_ids,
            engram_packed_dim=self._engram_packed_dim,
            tp_rank_idx=self.tp_rank_idx,
            init_pre_mix=self.init_pre_mix,
            hc_mult=self.hc_mult,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            precomputed_engram_hashes=engram_hashes,
            aux_hidden_state_layers=self.aux_hidden_state_layers,
        )
        _clear_stage_caches(self.layers)
        if not get_pp_group().is_last_rank:
            hidden_states = get_mhc_post_v41_op()(
                previous_x, hidden_states, previous_post_mix, previous_res_mix
            )
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )
        out = get_mhc_final_collapse_norm_v41_op(self.rms_norm_eps)(
            previous_x,
            hidden_states,
            previous_post_mix,
            previous_res_mix,
            pre_mix,
            self.norm.weight,
        )
        if token_sharded_mhc:
            out = tp_all_gather(out, dim=0)
        if self.aux_hidden_state_layers:
            return out, _gather_aux_hidden_states(aux_hidden_states, token_sharded_mhc)
        return out


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
    }
)
class DeepseekV41EncoderStage(nn.Module):
    """Compiled 20-layer EncoderStage (Layers 0..19 + Layer 20 global KV write)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        embed_tokens: nn.Module,
        encoder_layers: Iterable[nn.Module],
        decoder0_attn_norm: nn.Module,
        decoder0_attn: VllmDeepseekV41MLAAttention,
        engram_hash: TpuNgramHashState | None,
        engram_table_ids: list[int],
        engram_packed_dim: int,
        tp_rank_idx: torch.Tensor,
        init_pre_mix: torch.Tensor,
        hc_mult: int,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.hc_mult = hc_mult
        self.rms_norm_eps = float(vllm_config.model_config.hf_config.rms_norm_eps)
        self._engram_table_ids = engram_table_ids
        self._engram_packed_dim = engram_packed_dim
        self.embed_tokens = embed_tokens
        self.layers = nn.ModuleList(encoder_layers)
        self.decoder0_attn_norm = decoder0_attn_norm
        self.decoder0_attn = decoder0_attn
        self.engram_hash = engram_hash
        self.register_buffer("tp_rank_idx", tp_rank_idx, persistent=False)
        self.register_buffer("init_pre_mix", init_pre_mix, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            layer.attn._in_encoder_stage = True
        (
            positions,
            hidden_states,
            pre_mix,
            previous_x,
            previous_post_mix,
            previous_res_mix,
            token_sharded_mhc,
            aux_hidden_states,
        ) = _run_engram_prelude_and_layers(
            embed_tokens=self.embed_tokens,
            layers=self.layers,
            engram_hash=self.engram_hash,
            engram_table_ids=self._engram_table_ids,
            engram_packed_dim=self._engram_packed_dim,
            tp_rank_idx=self.tp_rank_idx,
            init_pre_mix=self.init_pre_mix,
            hc_mult=self.hc_mult,
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            precomputed_engram_hashes=engram_hashes,
        )
        _clear_stage_caches(self.layers)
        hidden_states, normalized = get_mhc_post_collapse_norm_v41_op(
            self.rms_norm_eps
        )(
            previous_x,
            hidden_states,
            previous_post_mix,
            previous_res_mix,
            pre_mix,
            self.decoder0_attn_norm.weight,
        )
        self.decoder0_attn.write_encoder_global_kv(normalized, positions)
        for layer in self.layers:
            layer.attn._in_encoder_stage = False
        return (
            hidden_states,
            pre_mix,
            torch.zeros_like(normalized),
        )


@support_torch_compile(
    dynamic_arg_dims={
        "positions": 0,
        "hidden_states": 0,
        "pre_mix": 0,
    }
)
class DeepseekV41DecoderStage(nn.Module):
    """Compiled 20-layer DecoderStage (Layers 20..39 + final RMSNorm)."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str,
        decoder_layers: Iterable[nn.Module],
        norm: nn.Module,
        tp_rank_idx: torch.Tensor,
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.rms_norm_eps = float(vllm_config.model_config.hf_config.rms_norm_eps)
        self.layers = nn.ModuleList(decoder_layers)
        self.norm = norm
        self.register_buffer("tp_rank_idx", tp_rank_idx, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        pre_mix: torch.Tensor,
    ) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()
        positions = torch.ops.aten.alias.default(positions)
        num_tokens = positions.shape[0]
        if tp_size > 1 and _TOKEN_SHARD_MIN_TOKENS > 0:
            # Canonical tp_rank_idx lift (see _run_engram_prelude_and_layers).
            _ = torch.ops.aten.alias.default(self.tp_rank_idx)
        token_sharded_mhc = tp_token_shard_ok(num_tokens, tp_size)
        if token_sharded_mhc and hidden_states.shape[0] == num_tokens:
            tp_rank = get_tensor_model_parallel_rank()
            hidden_states = tp_shard_slice(
                hidden_states, tp_size, tp_rank, self.tp_rank_idx
            )
            pre_mix = tp_shard_slice(pre_mix, tp_size, tp_rank, self.tp_rank_idx)

        previous_x, previous_post_mix, previous_res_mix = None, None, None
        for idx, layer in enumerate(self.layers):
            (
                previous_x,
                hidden_states,
                pre_mix,
                previous_post_mix,
                previous_res_mix,
            ) = layer(
                positions,
                hidden_states,
                pre_mix,
                previous_x=previous_x,
                previous_post_mix=previous_post_mix,
                previous_res_mix=previous_res_mix,
                engram_hashes=None,
                token_sharded=token_sharded_mhc,
                ced_decoder=True,
                ced_global_from_encoder=(idx == 0),
                tp_rank_idx=self.tp_rank_idx,
            )
        _clear_stage_caches(self.layers)
        out = get_mhc_final_collapse_norm_v41_op(self.rms_norm_eps)(
            previous_x,
            hidden_states,
            previous_post_mix,
            previous_res_mix,
            pre_mix,
            self.norm.weight,
        )
        if token_sharded_mhc:
            out = tp_all_gather(out, dim=0)
        return out


class DeepseekV41Model(nn.Module):
    """DeepSeek-V4.1 transformer model on TPU."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        set_offloader(NoopOffloader())
        config = vllm_config.model_config.hf_config
        self.config = config
        self.prefix = prefix
        self.hc_mult = config.hc_mult
        self._vllm_config = vllm_config
        self._dp_ced_decoder_bucket: int | None = None

        self.embed_tokens = (
            VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
            if get_pp_group().is_first_rank
            else PPMissingLayer()
        )
        if hasattr(config, "_tpu_candidate_block_buffer"):
            delattr(config, "_tpu_candidate_block_buffer")

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank() if tp_size > 1 else 0
        self.register_buffer(
            "tp_rank_idx",
            torch.tensor([tp_rank], dtype=torch.int64),
            persistent=False,
        )
        init_pre_mix = torch.zeros((config.hc_mult,), dtype=torch.float32)
        init_pre_mix[0] = 1.0
        self.register_buffer("init_pre_mix", init_pre_mix, persistent=False)
        self.topk_indices_buffer = torch.empty(
            max_tokens, config.index_topk, dtype=torch.int32
        )
        self.engram_layout = EngramLayout.from_config(config)
        self.candidate_block_buffer: torch.Tensor | None = None
        cand_blocks = getattr(config, "candidate_topk_blocks", 0)
        if getattr(config, "candidate_source_layer_id", -1) >= 0 and cand_blocks:
            self.candidate_block_buffer = torch.full(
                (max_tokens, cand_blocks), -1, dtype=torch.int32
            )

        total_layers = int(config.num_hidden_layers)
        ced_boundary = total_layers // 2
        pure_20l = bool(envs.VLLM_DSV41_PURE_20L_PREFILL)
        num_layers_to_build = (ced_boundary + 1) if pure_20l else total_layers

        def _build_layer(prefix: str) -> nn.Module:
            if pure_20l and extract_layer_index(prefix) == ced_boundary:
                return _PurePrefillBoundaryLayer(
                    vllm_config,
                    prefix,
                    self.topk_indices_buffer,
                    self.candidate_block_buffer,
                )
            return DeepseekV41DecoderLayer(
                vllm_config,
                prefix,
                self.topk_indices_buffer,
                self.engram_layout,
                self.candidate_block_buffer,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            num_layers_to_build,
            _build_layer,
            prefix=f"{prefix}.layers",
        )
        if hasattr(config, "_tpu_candidate_block_buffer"):
            delattr(config, "_tpu_candidate_block_buffer")

        engram_lids = (
            self.engram_layout.layer_ids if self.engram_layout is not None else ()
        )
        kv_src_lids = tuple(getattr(config, "kv_source_layer_ids", ()) or ())
        self._ced_enabled = bool(
            envs.VLLM_DSV41_CED
            and get_pp_group().world_size == 1
            and total_layers >= 2
            and total_layers % 2 == 0
            and self.start_layer == 0
            and self.end_layer in (total_layers, ced_boundary + 1)
            and all(lid < ced_boundary for lid in engram_lids)
            and [lid for lid in kv_src_lids if lid >= ced_boundary] == [ced_boundary]
        )
        self._ced_boundary = ced_boundary
        self._ced_window = int(getattr(config, "sliding_window", 128) or 128)

        active_layers = self.layers[self.start_layer : self.end_layer]
        engrams = [e for x in active_layers if (e := getattr(x, "engram", None))]
        self._engram_table_ids = [e.embed.table_id for e in engrams]
        self._engram_packed_dim = engrams[-1].embed.table_bytes if engrams else 6400
        self.engram_hash: TpuNgramHashState | None = None
        if engrams and self.engram_layout is not None:
            self.engram_hash = TpuNgramHashState(
                vllm_config, self.engram_layout, active_layers[0].attn.swa_cache_layer
            )
            t_max_local = (
                max_tokens // tp_size
                if tp_token_shard_ok(max_tokens, tp_size)
                else max_tokens
            )
            # Steps below VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS carry all T rows.
            t_max_rep = min(
                max_tokens, max(envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS - 1, 0)
            )
            n_hash_cols = int(self.engram_hash.offsets.shape[1])
            table_bytes = _engram_table_bytes(n_hash_cols, self._engram_packed_dim)
            self.register_buffer(
                "_engram_hashes_1d_buf",
                torch.zeros(
                    (
                        max(t_max_local, t_max_rep, 32),
                        len(self._engram_table_ids) * table_bytes,
                    ),
                    dtype=torch.uint8,
                ),
                persistent=False,
            )

        self.norm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if (get_pp_group().is_last_rank and not pure_20l)
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "pre_mix"], config.hidden_size
        )

        self._single_backbone_cond = bool(envs.VLLM_DSV41_SINGLE_BACKBONE_COND)
        self.full_stage: nn.Module | None
        if self._single_backbone_cond:
            from vllm_torchtpu.models.vllm.deepseek_v41.cond_backbone import (
                install_dsv41_cond_mlir_hook,
            )

            install_dsv41_cond_mlir_hook()
            self._ced_enabled = True
            for lid in range(ced_boundary, self.end_layer):
                self.layers[lid].attn._supports_ced_decoder = True
            self.full_stage = DeepseekV41FullStage(
                vllm_config=vllm_config,
                prefix=prefix,
                embed_tokens=self.embed_tokens,
                layers=self.layers[self.start_layer : self.end_layer],
                norm=self.norm,
                engram_hash=self.engram_hash,
                engram_table_ids=self._engram_table_ids,
                engram_packed_dim=self._engram_packed_dim,
                tp_rank_idx=self.tp_rank_idx,
                init_pre_mix=self.init_pre_mix,
                hc_mult=self.hc_mult,
                single_backbone_cond=True,
            )
            self.encoder_stage = None
            self.decoder_stage = None
            self.register_buffer(
                "_cond_flag_zero",
                torch.zeros((1,), dtype=torch.int32),
                persistent=False,
            )
            self.register_buffer(
                "_cond_flag_one",
                torch.ones((1,), dtype=torch.int32),
                persistent=False,
            )
            logger.info(
                "DeepSeek-V4.1 SingleBackboneCond enabled: unified 40-layer FullStage "
                "with runtime stablehlo.if at layer %d.",
                ced_boundary,
            )
        else:
            if (
                not self._ced_enabled or envs.VLLM_DSV41_HYBRID_CED_DECODE
            ) and not envs.VLLM_DSV41_PURE_20L_PREFILL:
                if os.environ.get("TPU_RAIDEN_DSV4_MAX_TRANSFER_LAYER", "").strip() == str(ced_boundary):
                    for lid in range(ced_boundary, self.end_layer):
                        self.layers[lid].attn._ced_decoder = True
                        self.layers[lid].attn._supports_ced_decoder = True
                self.full_stage = DeepseekV41FullStage(
                    vllm_config=vllm_config,
                    prefix=prefix,
                    embed_tokens=self.embed_tokens,
                    layers=self.layers[self.start_layer : self.end_layer],
                    norm=self.norm,
                    engram_hash=self.engram_hash,
                    engram_table_ids=self._engram_table_ids,
                    engram_packed_dim=self._engram_packed_dim,
                    tp_rank_idx=self.tp_rank_idx,
                    init_pre_mix=self.init_pre_mix,
                    hc_mult=self.hc_mult,
                )
            else:
                self.full_stage = None
            if self._ced_enabled:
                for lid in range(self.start_layer, self.end_layer):
                    self.layers[lid].attn._tp_attn_encoder_only = True
                for lid in range(ced_boundary, self.end_layer):
                    self.layers[lid].attn._supports_ced_decoder = True
                self.encoder_stage: nn.Module | None = DeepseekV41EncoderStage(
                    vllm_config=vllm_config,
                    prefix=f"{prefix}.encoder_stage",
                    embed_tokens=self.embed_tokens,
                    encoder_layers=self.layers[self.start_layer : ced_boundary],
                    decoder0_attn_norm=self.layers[ced_boundary].attn_norm,
                    decoder0_attn=self.layers[ced_boundary].attn,
                    engram_hash=self.engram_hash,
                    engram_table_ids=self._engram_table_ids,
                    engram_packed_dim=self._engram_packed_dim,
                    tp_rank_idx=self.tp_rank_idx,
                    init_pre_mix=self.init_pre_mix,
                    hc_mult=self.hc_mult,
                )
                self.decoder_stage: nn.Module | None = (
                    None
                    if envs.VLLM_DSV41_PURE_20L_PREFILL
                    else DeepseekV41DecoderStage(
                        vllm_config=vllm_config,
                        prefix=f"{prefix}.decoder_stage",
                        decoder_layers=self.layers[
                            ced_boundary : self.end_layer
                        ],
                        norm=self.norm,
                        tp_rank_idx=self.tp_rank_idx,
                    )
                )
                logger.info(
                    "DeepSeek-V4.1 CED enabled: single-subgraph EncoderStage [0..%d] + "
                    "DecoderStage [%d..%d] (sliding_window=%d, pure_20l_prefill=%s).",
                    ced_boundary - 1,
                    ced_boundary,
                    self.end_layer - 1,
                    self._ced_window,
                    envs.VLLM_DSV41_PURE_20L_PREFILL,
                )
            else:
                self.encoder_stage = None
                self.decoder_stage = None
        if envs.VLLM_DSV41_TP_ATTN and tp_size > 1:
            if self.encoder_stage is not None:
                logger.info(
                    "DeepSeek-V4.1 TP_ATTN enabled: stage-gated to EncoderStage "
                    "[0..%d] (FullStage/DecoderStage keep head-parallel attention).",
                    ced_boundary - 1,
                )
            else:
                logger.info(
                    "DeepSeek-V4.1 TP_ATTN enabled without EncoderStage "
                    "(ced_enabled=%s): running on FullStage (min_tokens=%d).",
                    self._ced_enabled,
                    envs.VLLM_DSV41_TP_ATTN_MIN_TOKENS,
                )

    def _set_aux_hidden_state_layers(self, layer_ids: tuple[int, ...]) -> None:
        """Capture the entry stream of `layer_ids` for the DSpark drafter.

        Only the full stage captures them; the CED encoder/decoder split does
        not (decode runs with CED off).
        """
        self.aux_hidden_state_layers = tuple(layer_ids)
        if self.full_stage is not None:
            self.full_stage.aux_hidden_state_layers = self.aux_hidden_state_layers
        if self.decoder_stage is not None and self.aux_hidden_state_layers:
            logger.warning(
                "DeepSeek-V4.1: aux hidden states %s are not captured on the "
                "CED decoder stage; DSpark gets them only from the full stage.",
                self.aux_hidden_state_layers,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def prefetch_engram_prefill(
        self,
        token_ids,
        req_idx,
        pos,
        known,
        padded_num_tokens: int,
        pre_async: Any = None,
        async_subst: Any = None,
        spec_async_subst: Any = None,
        debug_tok0_out: dict[int, int] | None = None,
    ) -> None:
        if self.engram_hash is not None:
            engram_prefetch_prefill(
                self.engram_hash,
                self._engram_table_ids,
                self._engram_packed_dim,
                token_ids,
                req_idx,
                pos,
                known,
                pre_async=pre_async,
                async_subst=async_subst,
                spec_async_subst=spec_async_subst,
                debug_tok0_out=debug_tok0_out,
                padded_num_tokens=padded_num_tokens,
            )

    @torch.no_grad()
    def precompile_ced_helpers(
        self, num_tokens_paddings: list[int], device: torch.device
    ) -> None:
        if (
            not self._ced_enabled
            or self.decoder_stage is None
            or device.type != "tpu"
        ):
            return
        dtype = self._vllm_config.model_config.dtype
        hsize = int(self.config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for n_enc in num_tokens_paddings:
            dummy_hs = torch.zeros(
                (n_enc, self.hc_mult, hsize), dtype=dtype, device=device
            )
            dummy_pm = torch.zeros(
                (n_enc, self.hc_mult), dtype=torch.float32, device=device
            )
            dummy_pos = torch.zeros((n_enc,), dtype=torch.int32, device=device)
            scatter_idx = torch.full((n_enc,), -1, dtype=torch.int64, device=device)
            sharded_enc = tp_size > 1 and tp_token_shard_ok(n_enc, tp_size)
            if sharded_enc:
                dummy_hs_local = torch.zeros(
                    (n_enc // tp_size, self.hc_mult, hsize),
                    dtype=dtype,
                    device=device,
                )
                dummy_pm_local = torch.zeros(
                    (n_enc // tp_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                )
            for n_dec in num_tokens_paddings:
                if n_dec > n_enc:
                    break
                gather_idx = torch.zeros((n_dec,), dtype=torch.int64, device=device)
                if sharded_enc:
                    t_loc = n_enc // tp_size
                    s_row = tp_rank * t_loc
                    l_idx = (gather_idx - s_row).clamp(0, t_loc - 1)
                    in_sh = (gather_idx >= s_row) & (gather_idx < s_row + t_loc)
                    dec_hs = tensor_model_parallel_all_reduce(
                        torch.where(
                            in_sh.view(-1, 1, 1),
                            dummy_hs_local[l_idx],
                            torch.zeros_like(dummy_hs_local[l_idx]),
                        )
                    )
                    dec_pm = tensor_model_parallel_all_reduce(
                        torch.where(
                            in_sh.view(-1, 1),
                            dummy_pm_local[l_idx],
                            torch.zeros_like(dummy_pm_local[l_idx]),
                        )
                    )
                    dec_pos = dummy_pos.index_select(0, gather_idx)
                    synchronize_tensors([dec_hs, dec_pm, dec_pos])
                dec_hs, dec_pm, dec_pos = _ced_gather_inputs(
                    dummy_hs, dummy_pm, dummy_pos, gather_idx
                )
                synchronize_tensors([dec_hs, dec_pm, dec_pos])
                dummy_dec_out = torch.zeros((n_dec, hsize), dtype=dtype, device=device)
                out = _ced_scatter_output(dummy_dec_out, scatter_idx)
                synchronize_tensors(out)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if not getattr(self, "_csa_ops_warmed", False):
            all_real = True
            for layer in self.layers:
                attn = layer.attn
                supports_ced = getattr(attn, "_supports_ced_decoder", False)
                if attn.compress_ratio > 0:
                    kv_holder = attn._kv_source_module()
                    main_entry = getattr(kv_holder, "kv_cache", None)
                    main_cache = (
                        main_entry[0] if isinstance(main_entry, tuple) else main_entry
                    )
                    if main_cache is not None and main_cache.numel() > 0:
                        m_u = (
                            (3 * int(attn.topk_indices_buffer.shape[1]) + 127) // 128
                        ) * 128
                        _ = attn._get_csa_union_op(
                            int(main_cache.shape[0] * main_cache.shape[1]), m_u
                        )
                        _ = attn._get_attn_op(True)
                        if supports_ced:
                            _ = attn._get_attn_op(False, ced_decoder=True)
                            # Bug #77 (decode): pre-warm the has_union CED decoder
                            # variant too. PURE_20L prefill never runs a CED
                            # decoder step, so it keeps the original warm set.
                            if not envs.VLLM_DSV41_PURE_20L_PREFILL:
                                _ = attn._get_attn_op(True, ced_decoder=True)
                    else:
                        all_real = False
                elif supports_ced:
                    swa_entry = getattr(attn.swa_cache_layer, "kv_cache", None)
                    swa_cache = (
                        swa_entry[0] if isinstance(swa_entry, tuple) else swa_entry
                    )
                    if swa_cache is not None and swa_cache.numel() > 0:
                        _ = attn._get_attn_op(False, ced_decoder=True)
                    else:
                        all_real = False
            if all_real:
                self._csa_ops_warmed = True

        num_tokens = int(positions.shape[0])
        engram_hashes: torch.Tensor | None = None
        if self.engram_hash is not None:
            n_tables = len(self._engram_table_ids)
            n_hash_cols = int(self.engram_hash.offsets.shape[1])
            buf_1d = getattr(self, "_engram_hashes_1d_buf", None)
            engram_hashes = take_prefetched_engram_hashes(
                num_tokens=num_tokens,
                n_tables=n_tables,
                n_hash_cols=n_hash_cols,
                packed_dim=self._engram_packed_dim,
                device=positions.device,
                out_1d_i32_buf=buf_1d,
            )
            if engram_hashes is not None:
                self._engram_buf_dirty = True
            elif envs.VLLM_DSV41_PURE_20L_PREFILL:
                # PURE_20L prefill: no upper layers / logits, so a step without a
                # fully-known host prefetch only needs zeroed rows (prefill path).
                tp_size = get_tensor_model_parallel_world_size()
                t_local = (
                    num_tokens // tp_size
                    if tp_token_shard_ok(num_tokens, tp_size)
                    else num_tokens
                )
                if buf_1d is not None:
                    if getattr(self, "_engram_buf_dirty", False):
                        if buf_1d.ndim == 2:
                            buf_1d[: min(max(t_local, 32), buf_1d.shape[0])].zero_()
                        else:
                            n_bytes = (
                                t_local
                                * n_tables
                                * _engram_table_bytes(
                                    n_hash_cols, self._engram_packed_dim
                                )
                            )
                            buf_1d[:n_bytes].zero_()
                        self._engram_buf_dirty = False
                    engram_hashes = buf_1d
                else:
                    cols_flat = (t_local * n_tables * n_hash_cols) // num_tokens
                    engram_hashes = torch.zeros(
                        (num_tokens, cols_flat, self._engram_packed_dim),
                        dtype=torch.uint8,
                        device=positions.device,
                    )
            elif buf_1d is not None:
                # Decode path: host prefetch missing / not all-known (e.g. DSpark
                # draft tokens under async scheduling) -> hash on device and do
                # the host table lookup into the static buffer.
                all_attn_metadata = get_forward_context().attn_metadata
                if (
                    isinstance(all_attn_metadata, dict)
                    and self.engram_hash.ensure_cache()
                    and input_ids is not None
                ):
                    swa_meta = all_attn_metadata[
                        self.engram_hash.swa_cache_module.prefix
                    ]
                    hashes_raw = self.engram_hash(
                        input_ids,
                        positions,
                        swa_meta.query_start_loc,
                        swa_meta.block_tables,
                        swa_meta.request_distribution,
                    )
                    tp_size = get_tensor_model_parallel_world_size()
                    if tp_token_shard_ok(num_tokens, tp_size):
                        stage_ref = self.full_stage or self.encoder_stage
                        assert stage_ref is not None
                        hashes_raw = tp_shard_slice(
                            hashes_raw,
                            tp_size,
                            get_tensor_model_parallel_rank(),
                            stage_ref.tp_rank_idx,
                        )
                    flat = torch.ops.tpu.engram_cpu_lookup_all(
                        hashes_raw,
                        self._engram_table_ids,
                        self._engram_packed_dim,
                    )
                    n_rows = int(hashes_raw.shape[0])
                    buf_1d[:n_rows].copy_(flat.view(n_rows, -1))
                    self._engram_buf_dirty = True
                elif getattr(self, "_engram_buf_dirty", False):
                    buf_1d.zero_()
                    self._engram_buf_dirty = False
                engram_hashes = buf_1d

        if self._single_backbone_cond:
            assert self.full_stage is not None
            fwd_ctx = get_forward_context()
            if envs.VLLM_DSV41_PURE_20L_PREFILL:
                need_upper = False
            elif self._dp_ced_decoder_bucket is not None:
                need_upper = abs(int(self._dp_ced_decoder_bucket)) > 0
            else:
                attn_metadata = fwd_ctx.attn_metadata
                swa_prefix = (
                    self.layers[self._ced_boundary].attn.swa_cache_layer.prefix
                )
                swa_meta = (
                    attn_metadata.get(swa_prefix)
                    if isinstance(attn_metadata, dict)
                    else attn_metadata
                )
                compile_sizes = (
                    self._vllm_config.compilation_config.compile_sizes
                    or [num_tokens]
                )
                plan = plan_decoder_replay(
                    swa_meta,
                    self._ced_window,
                    num_tokens,
                    compile_sizes,
                )
                need_upper = plan is None or plan.padded_num_tokens > 0
            run_upper = (
                self._cond_flag_one if need_upper else self._cond_flag_zero
            )
            return self.full_stage(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                engram_hashes=engram_hashes,
                run_upper_layers=run_upper,
            )

        if not self._ced_enabled:
            assert self.full_stage is not None
            return self.full_stage(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                engram_hashes=engram_hashes,
            )

        if self.decoder_stage is None:
            assert self.encoder_stage is not None
            _, _, zero_out = self.encoder_stage(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                engram_hashes=engram_hashes,
            )
            if zero_out.shape[0] != num_tokens:
                zero_out = tp_all_gather(zero_out, dim=0)
            if zero_out.shape[0] != num_tokens:
                zero_out = tp_all_gather(zero_out, dim=0)
            return zero_out

        fwd_ctx = get_forward_context()
        attn_metadata = fwd_ctx.attn_metadata
        swa_prefix = self.layers[self._ced_boundary].attn.swa_cache_layer.prefix
        swa_meta = (
            attn_metadata.get(swa_prefix)
            if isinstance(attn_metadata, dict)
            else attn_metadata
        )
        compile_sizes = self._vllm_config.compilation_config.compile_sizes or [
            num_tokens
        ]
        plan = plan_decoder_replay(
            swa_meta,
            self._ced_window,
            num_tokens,
            compile_sizes,
            dp_ced_decoder_bucket=self._dp_ced_decoder_bucket,
        )
        if self.full_stage is not None and (plan is None or plan.passthrough):
            return self.full_stage(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                engram_hashes=engram_hashes,
            )

        assert self.encoder_stage is not None
        hidden_states, pre_mix, _ = self.encoder_stage(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            engram_hashes=engram_hashes,
        )
        if plan is not None and not plan.passthrough and plan.padded_num_tokens <= 0:
            return torch.zeros(
                (num_tokens, int(self.config.hidden_size)),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        if plan is None or plan.passthrough:
            if hidden_states.shape[0] != num_tokens:
                hidden_states = tp_all_gather(hidden_states, dim=0)
                pre_mix = tp_all_gather(pre_mix, dim=0)
            return self.decoder_stage(
                positions=positions,
                hidden_states=hidden_states,
                pre_mix=pre_mix,
            )

        dev = positions.device
        dec_q_loc = plan.query_start_loc.to(dev, non_blocking=True)
        dec_seq_lens = plan.seq_lens.to(dev, non_blocking=True)
        dec_dist = plan.request_distribution.to(dev, non_blocking=True)
        gather_idx = plan.gather_idx.to(dev, non_blocking=True) if hasattr(plan, "gather_idx") else plan.gather_indices.to(dev, non_blocking=True)
        scatter_idx = plan.scatter_indices.to(dev, non_blocking=True)
        if hidden_states.shape[0] != num_tokens:
            tp_rank = get_tensor_model_parallel_rank()
            t_local = hidden_states.shape[0]
            start_row = tp_rank * t_local
            end_row = start_row + t_local
            local_idx = (gather_idx - start_row).clamp(0, t_local - 1)
            in_shard = (gather_idx >= start_row) & (gather_idx < end_row)
            hs_sel = torch.where(
                in_shard.view(-1, 1, 1),
                hidden_states[local_idx],
                torch.zeros_like(hidden_states[local_idx]),
            )
            pm_sel = torch.where(
                in_shard.view(-1, 1),
                pre_mix[local_idx],
                torch.zeros_like(pre_mix[local_idx]),
            )
            hs_flat = hs_sel.reshape(hs_sel.shape[0], -1)
            hs_cols = hs_flat.shape[-1]
            pm_cols = int(pm_sel.shape[-1])
            pm_pad = ((pm_cols + 127) // 128) * 128
            reduced_seam = tensor_model_parallel_all_reduce(
                torch.cat(
                    [
                        hs_flat,
                        torch.nn.functional.pad(
                            pm_sel.to(hs_flat.dtype), (0, pm_pad - pm_cols)
                        ),
                    ],
                    dim=-1,
                )
            )
            dec_hidden = reduced_seam[:, :hs_cols].contiguous().reshape(hs_sel.shape)
            dec_pre_mix = reduced_seam[:, hs_cols : hs_cols + pm_cols].contiguous().to(
                pm_sel.dtype
            )
            dec_positions = positions.index_select(0, gather_idx)
        else:
            dec_hidden, dec_pre_mix, dec_positions = _ced_gather_inputs(
                hidden_states, pre_mix, positions, gather_idx
            )

        if isinstance(attn_metadata, dict):
            decoder_metadata: Any = {
                k: dataclasses.replace(
                    v,
                    input_positions=dec_positions,
                    seq_lens=dec_seq_lens,
                    query_start_loc=dec_q_loc,
                    request_distribution=dec_dist,
                )
                for k, v in attn_metadata.items()
            }
        elif attn_metadata is not None:
            decoder_metadata = dataclasses.replace(
                attn_metadata,
                input_positions=dec_positions,
                seq_lens=dec_seq_lens,
                query_start_loc=dec_q_loc,
                request_distribution=dec_dist,
            )
        else:
            decoder_metadata = None

        padding_state = token_padding._padding_state
        if padding_state is not None:
            padding_state.update(len(plan.indices), plan.padded_num_tokens)

        dp_size = self._vllm_config.parallel_config.data_parallel_size
        has_dp_meta = fwd_ctx.dp_metadata is not None and dp_size > 1
        dec_dp_tokens = (
            torch.tensor(
                [plan.padded_num_tokens] * dp_size,
                dtype=torch.int32,
                device="cpu",
            )
            if has_dp_meta
            else None
        )
        with set_forward_context(
            decoder_metadata,
            self._vllm_config,
            num_tokens=plan.padded_num_tokens if has_dp_meta else 0,
            num_tokens_across_dp=dec_dp_tokens,
        ):
            dec_out = self.decoder_stage(
                positions=dec_positions,
                hidden_states=dec_hidden,
                pre_mix=dec_pre_mix,
            )
        return _ced_scatter_output(dec_out, scatter_idx)

    get_expert_mapping = DeepseekV4Model.get_expert_mapping

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("gate_up_proj", "w1", 0),
            ("gate_up_proj", "w3", 1),
            ("attn.fused_wqa_wkv", "attn.wq_a", 0),
            ("attn.fused_wqa_wkv", "attn.wkv", 1),
            ("compressor.fused_wkv_wgate", "compressor.wkv", 0),
            ("compressor.fused_wkv_wgate", "compressor.wgate", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        staged = _HostStagedParams()
        expert_mapping = self.get_expert_mapping()
        layer_re = re.compile(r"layers\.(\d+)\.")

        def _valid_param(pname: str) -> bool:
            return pname in params_dict and not is_pp_missing_parameter(pname, self)

        max_seen_layer = -1
        try:
            for name, loaded_weight in weights:
                m_layer = layer_re.search(name)
                if m_layer:
                    cur_layer = int(m_layer.group(1))
                    if cur_layer > max_seen_layer:
                        max_seen_layer = cur_layer
                        for k in list(staged._staged):
                            m_k = layer_re.search(k)
                            if m_k and int(m_k.group(1)) <= max_seen_layer - 2:
                                staged._flush(k)

                if ".engram.embed." in name:
                    if _valid_param(name):
                        param = params_dict[name]
                        getattr(param, "weight_loader", default_weight_loader)(
                            param, loaded_weight
                        )
                        loaded_params.add(name)
                    continue

                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if ".experts." in name or weight_name not in name:
                        continue
                    mapped = name.replace(weight_name, param_name)
                    if is_pp_missing_parameter(mapped, self):
                        break
                    if mapped not in params_dict:
                        continue
                    param = staged.stage(mapped, params_dict[mapped])
                    param.weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(mapped)
                    break
                else:
                    if ".experts." in name:
                        if (
                            "weight_scale" in name
                            and loaded_weight.dtype == torch.float8_e8m0fnu
                        ):
                            loaded_weight = loaded_weight.view(torch.uint8)
                        for (
                            param_name,
                            weight_name,
                            expert_id,
                            expert_shard_id,
                        ) in expert_mapping:
                            if weight_name not in name:
                                continue
                            name_mapped = name.replace(weight_name, param_name)
                            if not _valid_param(name_mapped):
                                continue
                            param = staged.stage(name_mapped, params_dict[name_mapped])
                            weight_loader = typing.cast(
                                Callable[..., bool], param.weight_loader
                            )
                            if weight_loader(
                                param,
                                loaded_weight,
                                name_mapped,
                                shard_id=expert_shard_id,
                                expert_id=expert_id,
                                return_success=True,
                            ):
                                loaded_params.add(name_mapped)
                                break
                    elif _valid_param(name):
                        param = staged.stage(name, params_dict[name])
                        getattr(param, "weight_loader", default_weight_loader)(
                            param, loaded_weight
                        )
                        staged._flush(name)
                        loaded_params.add(name)
        finally:
            staged.flush_all()

        for layer in self.layers[self.start_layer : self.end_layer]:
            if hasattr(layer, "_ensure_mhc_buffers"):
                layer._ensure_mhc_buffers(force=True)
            engram_mod = getattr(layer, "engram", None)
            if engram_mod is not None:
                engram_mod.process_weights_after_loading()

        missing = set(params_dict.keys()) - loaded_params
        logger.info(
            "DeepseekV41Model loaded %d/%d parameters (missing=%d: %s)",
            len(loaded_params),
            len(params_dict),
            len(missing),
            sorted(missing)[:10],
        )
        return loaded_params


class DeepseekV41ForCausalLM(DeepseekV4ForCausalLM):
    """DeepSeek-V4.1 causal language model on TPU."""

    model_cls = DeepseekV41Model  # type: ignore[assignment]

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # DSpark on V4.1 reads the *entry* stream of its target layers
        # (dspark_target_layer_ids, unshifted). The V4 base only records the
        # ids on the model, which the V4.1 stages never read.
        self.model._set_aux_hidden_state_layers(tuple(layers))

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if not envs.VLLM_DSV41_PURE_20L_PREFILL:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return
        import sys
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        sys.setrecursionlimit(50000)
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model = self.model_cls(
            vllm_config=vllm_config,
            prefix=f"{prefix}.model" if prefix else "model",
        )
        self.lm_head = None
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.lm_head is None:
            return hidden_states.new_zeros(
                (hidden_states.shape[0], self.config.vocab_size),
                dtype=torch.float32,
            )
        return super().compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        pure_20l = bool(envs.VLLM_DSV41_PURE_20L_PREFILL)
        ced_boundary = int(self.config.num_hidden_layers) // 2
        layer_re = re.compile(r"(?:^|\.)layers\.(\d+)\.")

        def _iter_filtered():
            for name, w in weights:
                if name.startswith(("mtp.", "vision.", "aligner.", "image_")):
                    continue
                if name.endswith(".bias_vl"):
                    continue
                if pure_20l:
                    if name.startswith(("head.", "lm_head.", "norm.", "model.norm.")):
                        continue
                    m = layer_re.search(name)
                    if m:
                        lid = int(m.group(1))
                        if lid > ced_boundary:
                            continue
                        if lid == ced_boundary and any(
                            tag in name
                            for tag in (".ffn.", ".mlp.", ".ffn_norm.", ".hc_")
                        ):
                            continue
                yield name, w

        return AutoWeightsLoader(self).load_weights(
            _iter_filtered(), mapper=self.hf_to_vllm_mapper
        )

    def process_weights_after_loading(self) -> None:
        for layer in self.model.layers[self.model.start_layer : self.model.end_layer]:
            if hasattr(layer, "_ensure_mhc_buffers"):
                layer._ensure_mhc_buffers()
            engram_mod = getattr(layer, "engram", None)
            if engram_mod is not None:
                engram_mod.process_weights_after_loading(allow_dummy=True)
            ffn_mod = getattr(layer, "ffn", None)
            if ffn_mod is not None and hasattr(
                ffn_mod, "process_weights_after_loading"
            ):
                ffn_mod.process_weights_after_loading()
