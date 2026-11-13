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
"""DeepSeek-V4 feed-forward and Manifold-Harmonized Coupling helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.activation import SiluAndMul, SiluAndMulWithClamp
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.tp_collectives import (
    tp_all_gather,
    tp_reduce_scatter,
    tp_shard_slice,
)
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
    _get_quantized_matmul_op,
    _get_quantized_matmul_prequant_op,
    _get_quantized_mlp_op,
    _get_quantized_mlp_prequant_op,
    quantized_matmul,
    quantized_matmul_prequant,
    quantized_mlp,
    quantized_mlp_prequant,
)
from vllm_torchtpu.utils import synchronize_tensors


# Pure PyTorch head projection for MHC manifold collapse;
# upstream vLLM only provides Triton/TileLang GPU kernels for hc_head.
def mhc_collapse_head(
    hidden_states: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Collapse multi-stream residual manifold into a single vector before LM head."""
    residual_flat = hidden_states.flatten(-2).float()
    rstd = torch.rsqrt(residual_flat.square().mean(dim=-1, keepdim=True) + rms_norm_eps)
    pre_mix = torch.nn.functional.linear(residual_flat, hc_fn) * rstd
    pre_mix = torch.sigmoid(pre_mix * hc_scale + hc_base) + hc_eps
    return torch.sum(pre_mix.unsqueeze(-1) * hidden_states.float(), dim=-2).to(
        hidden_states.dtype
    )


class DeepseekV4MLP(nn.Module):
    """DeepSeek-V4 feed-forward layer for dense/shared expert projections."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        swiglu_limit: float | None = None,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._prequant: list[torch.Tensor | None] = [None, None]
        self.swiglu_limit = swiglu_limit
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        if swiglu_limit is not None:
            self.act_fn = SiluAndMulWithClamp(swiglu_limit)
        else:
            self.act_fn = SiluAndMul()
        _get_quantized_mlp_op(swiglu_limit)
        _get_quantized_mlp_prequant_op(swiglu_limit)

    def _build_tp_weights(self) -> None:
        if "_tp_gate_up_w" in self._buffers:
            return
        try:
            tp_size = get_tensor_model_parallel_world_size()
        except Exception:
            return
        if tp_size <= 1 or getattr(self.gate_up_proj, "disable_tp", False):
            return
        gu_s = getattr(self.gate_up_proj, "weight_scale", None)
        down_s = getattr(self.down_proj, "weight_scale", None)
        if (
            gu_s is None
            or down_s is None
            or gu_s.ndim not in (1, 4)
            or (gu_s.ndim == 4 and not envs.VLLM_DSV41_REPL_SHARED_EXPERTS)
            or down_s.ndim not in (1, 4)
            or getattr(self.gate_up_proj, "bias", None) is not None
            or getattr(self.down_proj, "bias", None) is not None
        ):
            return

        gu_w = self.gate_up_proj.weight.data
        if getattr(self.gate_up_proj, KEEP_VLLM_LAYOUT_ATTR, False):
            gu_w = gu_w.transpose(0, 1)
        elif gu_w.shape[0] != getattr(self.gate_up_proj, "input_size", gu_w.shape[0]):
            return

        down_w = self.down_proj.weight.data
        if getattr(self.down_proj, KEEP_VLLM_LAYOUT_ATTR, False):
            down_w = down_w.transpose(0, 1)
        elif down_w.shape[1] != getattr(self.down_proj, "output_size", down_w.shape[1]):
            return

        def gather(x: torch.Tensor, dim: int) -> torch.Tensor:
            if x.element_size() == 1:
                out = tp_all_gather(x.view(torch.uint8), dim).view(x.dtype)
            else:
                out = tp_all_gather(x, dim)
            return out.clone(memory_format=torch.contiguous_format)

        k_dim, two_i_shard = gu_w.shape
        i_shard = two_i_shard // 2
        tp_gate_up_w = (
            gather(gu_w.reshape(k_dim, 2, i_shard).contiguous(), 2)
            .reshape(k_dim, 2 * tp_size * i_shard)
            .clone(memory_format=torch.contiguous_format)
        )
        if gu_s.ndim == 4:
            k_blk = gu_s.shape[1]
            tp_gate_up_s = (
                gather(gu_s.data.reshape(1, k_blk, 2, i_shard).contiguous(), 3)
                .reshape(1, k_blk, 1, 2 * tp_size * i_shard)
                .clone(memory_format=torch.contiguous_format)
            )
        else:
            tp_gate_up_s = (
                gather(gu_s.data.reshape(2, i_shard).contiguous(), 1)
                .reshape(-1)
                .clone(memory_format=torch.contiguous_format)
            )
        tp_down_w_raw = gather(down_w.contiguous(), 0)
        if down_s.ndim == 4:
            tp_down_w = tp_down_w_raw
            tp_down_s = gather(down_s.data, 1)
        else:
            tp_down_s_all = gather(down_s.data.unsqueeze(0), 0)
            tp_down_s = tp_down_s_all.amax(dim=0).contiguous()
            if not torch.equal(
                tp_down_s_all,
                tp_down_s.unsqueeze(0).expand_as(tp_down_s_all),
            ):
                w_3d = tp_down_w_raw.view(tp_size, -1, tp_down_w_raw.shape[-1]).float()
                ratio = (
                    tp_down_s_all / tp_down_s.unsqueeze(0).clamp_min(1e-12)
                ).unsqueeze(1)
                tp_down_w = (
                    (w_3d * ratio)
                    .to(down_w.dtype)
                    .view_as(tp_down_w_raw)
                    .clone(memory_format=torch.contiguous_format)
                )
            else:
                tp_down_w = tp_down_w_raw
        buffers = {
            "_tp_gate_up_w": tp_gate_up_w,
            "_tp_gate_up_s": tp_gate_up_s,
            "_tp_down_w": tp_down_w,
            "_tp_down_s": tp_down_s,
        }
        device = gu_w.device
        if device.type == "tpu":
            synchronize_tensors(list(buffers.values()), wait=True)
            _ = _get_quantized_matmul_op()
            _ = _get_quantized_matmul_prequant_op()
            _ = _get_quantized_mlp_op(self.swiglu_limit)
            _ = _get_quantized_mlp_prequant_op(self.swiglu_limit)
        for name, tensor in buffers.items():
            self.register_buffer(name, tensor, persistent=False)

    def process_weights_after_loading(self) -> None:
        for proj in (self.gate_up_proj, self.down_proj):
            qm = getattr(proj, "quant_method", None)
            if qm is not None and not getattr(proj, WEIGHT_FLIPPED_ATTR, False):
                qm.process_weights_after_loading(proj)
        self._build_tp_weights()

    def has_tp_local_weights(self) -> bool:
        if "_tp_gate_up_w" in self._buffers:
            return True
        if not torch.compiler.is_compiling() and getattr(
            self.gate_up_proj, WEIGHT_FLIPPED_ATTR, False
        ):
            self._build_tp_weights()
            return "_tp_gate_up_w" in self._buffers
        return False

    def _lift_tp_shape_variant_tensors(self) -> None:
        if (
            envs.VLLM_DSV41_TOKEN_SHARD_MIN_TOKENS > 0
            and "_tp_gate_up_w" in self._buffers
        ):
            _ = [
                torch.ops.aten.alias.default(t)
                for t in (
                    self.gate_up_proj.weight,
                    getattr(self.gate_up_proj, "weight_scale", None),
                    self.down_proj.weight,
                    getattr(self.down_proj, "weight_scale", None),
                    self._tp_gate_up_w,
                    self._tp_gate_up_s,
                    self._tp_down_w,
                    self._tp_down_s,
                )
                if t is not None
            ]

    def can_accept_prequant_hidden(self, *, token_sharded: bool = False) -> bool:
        if not envs.VLLM_DSV41_FUSED_SEAM_QUANT:
            return False
        w_s = getattr(self.gate_up_proj, "weight_scale", None)
        ok = (
            w_s is not None
            and w_s.ndim == 1
            and getattr(self.gate_up_proj, "bias", None) is None
        )
        return ok and (not token_sharded or self.has_tp_local_weights())

    def set_prequant_hidden(
        self,
        hidden_states_fp8: torch.Tensor | None,
        hidden_states_scale: torch.Tensor | None,
    ) -> None:
        self._prequant[0] = hidden_states_fp8
        self._prequant[1] = hidden_states_scale

    def _apply_gate_up_proj(
        self,
        x: torch.Tensor,
        hidden_states_fp8: torch.Tensor | None,
        hidden_states_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        if hidden_states_fp8 is not None and hidden_states_scale is not None:
            weight = self.gate_up_proj.weight
            if getattr(self.gate_up_proj, KEEP_VLLM_LAYOUT_ATTR, False):
                weight = weight.transpose(0, 1)
            return quantized_matmul_prequant(
                hidden_states_fp8,
                hidden_states_scale,
                weight,
                self.gate_up_proj.weight_scale,
            )
        return self.gate_up_proj(x)[0]

    def _can_fuse_quantized_mlp(self) -> bool:
        gu_s = getattr(self.gate_up_proj, "weight_scale", None)
        dn_s = getattr(self.down_proj, "weight_scale", None)
        return (
            gu_s is not None
            and dn_s is not None
            and getattr(self.gate_up_proj, "bias", None) is None
            and getattr(self.down_proj, "bias", None) is None
        )

    def forward_tp_local(
        self,
        x: torch.Tensor,
        hidden_states_fp8: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._lift_tp_shape_variant_tensors()
        if hidden_states_fp8 is not None and hidden_states_scale is not None:
            return quantized_mlp_prequant(
                hidden_states_fp8,
                hidden_states_scale,
                self._tp_gate_up_w,
                self._tp_gate_up_s,
                self._tp_down_w,
                self._tp_down_s,
                swiglu_limit=self.swiglu_limit,
            )
        return quantized_mlp(
            x,
            self._tp_gate_up_w,
            self._tp_gate_up_s,
            self._tp_down_w,
            self._tp_down_s,
            swiglu_limit=self.swiglu_limit,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden_states_fp8: torch.Tensor | None = None,
        hidden_states_scale: torch.Tensor | None = None,
        return_token_sharded: bool = False,
        *,
        input_is_token_sharded: bool = False,
    ) -> torch.Tensor:
        if hidden_states_fp8 is None:
            hidden_states_fp8 = self._prequant[0]
            hidden_states_scale = self._prequant[1]
        self._prequant[0] = None
        self._prequant[1] = None
        if (
            input_is_token_sharded
            and return_token_sharded
            and self.has_tp_local_weights()
        ):
            return self.forward_tp_local(
                x,
                hidden_states_fp8=hidden_states_fp8,
                hidden_states_scale=hidden_states_scale,
            )
        self._lift_tp_shape_variant_tensors()
        if input_is_token_sharded:
            x = tp_all_gather(x, dim=0)
            hidden_states_fp8 = None
            hidden_states_scale = None
        if self._can_fuse_quantized_mlp():
            gu_w = self.gate_up_proj.weight
            if getattr(self.gate_up_proj, KEEP_VLLM_LAYOUT_ATTR, False):
                gu_w = gu_w.transpose(0, 1)
            dn_w = self.down_proj.weight
            if getattr(self.down_proj, KEEP_VLLM_LAYOUT_ATTR, False):
                dn_w = dn_w.transpose(0, 1)
            if hidden_states_fp8 is not None and hidden_states_scale is not None:
                x = quantized_mlp_prequant(
                    hidden_states_fp8,
                    hidden_states_scale,
                    gu_w,
                    self.gate_up_proj.weight_scale,
                    dn_w,
                    self.down_proj.weight_scale,
                    swiglu_limit=self.swiglu_limit,
                )
            else:
                x = quantized_mlp(
                    x,
                    gu_w,
                    self.gate_up_proj.weight_scale,
                    dn_w,
                    self.down_proj.weight_scale,
                    swiglu_limit=self.swiglu_limit,
                )
            if (
                getattr(self.down_proj, "reduce_results", False)
                and not getattr(self.down_proj, "disable_tp", False)
                and get_tensor_model_parallel_world_size() > 1
            ):
                x = tensor_model_parallel_all_reduce(x)
            if return_token_sharded:
                if getattr(self.down_proj, "reduce_results", False):
                    return tp_shard_slice(
                        x,
                        get_tensor_model_parallel_world_size(),
                        get_tensor_model_parallel_rank(),
                        None,
                    )
                return tp_reduce_scatter(x, dim=0)
            return x
        gate_up = self._apply_gate_up_proj(x, hidden_states_fp8, hidden_states_scale)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        if return_token_sharded:
            if getattr(self.down_proj, "reduce_results", False):
                return tp_shard_slice(
                    x,
                    get_tensor_model_parallel_world_size(),
                    get_tensor_model_parallel_rank(),
                    None,
                )
            return tp_reduce_scatter(x, dim=0)
        return x
