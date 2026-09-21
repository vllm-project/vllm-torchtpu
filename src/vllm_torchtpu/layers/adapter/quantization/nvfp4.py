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
"""NVFP4 (ModelOpt FP4) quantization support for TPU.

Loads NVIDIA ModelOpt NVFP4 checkpoints (packed E2M1 weights + per-16-block
E4M3 scales + per-tensor FP32 global scale) and runs them on the TPU GMM /
quantized-matmul kernels in W4A16 mode (bf16 activations, fp4 weights).
``MOE_REQUANTIZE_BLOCK_SIZE`` changes the MoE weight block size without
changing the GMM activation precision. W4A8 requires both
``USE_MOE_FUSED_EP_KERNEL=1`` and ``MOE_FUSED_EP_ENABLE_W4A8=1``. Only admitted
EP layers default to the kernel's smallest supported weight block when no
explicit block is configured. Refused layers retain their GMM weight recipe.

Design:
  - Reuse upstream `ModelOptNvFp4{LinearMethod,RoutedExperts}.create_weights` for
    parameter registration (packed uint8 weight, E4M3 `weight_scale`, FP32
    `weight_scale_2` global, `input_scale`).
  - `process_weights_after_loading` (pure PyTorch, no JAX): fuse the E4M3 block
    scale with the FP32 global scale into a single FP32 block-16 scale and lay
    it out for the kernel.
  - Unpack the packed uint8 weight to native fp4 (`torch.float4_e2m1fn_x2`) in
    the kernel's K-major layout once here at load (`load_kmajor_fp4`), so the
    forward hands native fp4 straight to the kernel with no per-forward unpack.
    Ordinary and pipelined GMM keep activations in bf16 (W4A16), including
    layers refused by fused EP. Only fused EP MoE v2 uses FP8 activations.

Requires torch_tpu's native `torch.float4_e2m1fn_x2` dtype
(google-pytorch/torch_tpu#1560).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config, ModelOptNvFp4FusedMoE, ModelOptNvFp4LinearMethod)
from vllm.model_executor.utils import replace_parameter

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter import moe_routing
from vllm_torchtpu.layers.adapter.fused_moe import (TpuMoEActivationMixin,
                                                    fused_moe_gmm,
                                                    load_kmajor_fp4,
                                                    prebuild_fused_moe_kernel,
                                                    requant_load_kmajor_fp4)
from vllm_torchtpu.layers.adapter.linear_common import quantized_matmul_fp4
from vllm_torchtpu.layers.adapter.pipelined_fused_moe import (
    enable_pipelined_collective_and_compute, pipelined_fused_moe_gmm)
from vllm_torchtpu.layers.adapter.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from vllm_torchtpu.layers.adapter.quantization.fp8 import resolve_online_fp8
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import map_online_fp8
from vllm_torchtpu.layers.core.quant_methods import NVFP4, get_tpu_quant_method
from vllm_torchtpu.layers.core.quantization import (dequantize_tensor,
                                                    pack_fp4_indices,
                                                    quantize_tensor_to_fp4,
                                                    unpack_uint8_to_fp4)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to, synchronize_tensors

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


def _fresh(t: torch.Tensor) -> torch.Tensor:
    """Allocate a fresh contiguous device buffer (breaks view/stride chains so
    the torch_tpu pallas boundary ships a plain row-major buffer)."""
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


def _to_kernel_scale(scale_f32: torch.Tensor) -> torch.Tensor:
    """[*, out, num_blocks] fp32 -> [*, num_blocks, 1, out] fp32 (gmm_v2 rhs
    scale layout), as a fresh contiguous buffer."""
    s = scale_f32.transpose(-1, -2).contiguous()  # [*, num_blocks, out]
    return _fresh(s.unsqueeze(-2))  # [*, num_blocks, 1, out]


def _requant_moe_w4a8(w13_u8: torch.Tensor, w13_scale_f: torch.Tensor,
                      w2_u8: torch.Tensor, w2_scale_f: torch.Tensor,
                      block: int):
    """Requantize NVFP4 MoE weights from native block-16 to block-`block` fp4
    for either W4A16 GMM or W4A8 fused EP execution.

    Dequantizes with the fused block-16 scale, pads the MoE intermediate up to a
    multiple of `block` (zero-pad: w13's two output halves and w2's contracting
    dim; the padded region is zero-weight so it is numerically inert), then
    requantizes to block-`block` fp4. Pure PyTorch, reusing the common fp4
    toolkit (the only fp4 ops that cannot be torch live at the kernel boundary).
    Returns packed (w13_u8, w13_scale_4d, w2_u8, w2_scale_4d).
    """
    H = w13_u8.shape[-1] * 2  # contracting dim of w13
    two_i = w13_u8.shape[1]
    inter = two_i // 2  # MoE intermediate (per expert)
    assert H % block == 0, (
        f"NVFP4 requant needs hidden ({H}) divisible by block ({block}).")
    inter_pad = align_to(inter, block) - inter

    # w13: dequant block-16 -> [E, 2I, H] fp32; pad each output half I -> I_pad.
    w13_f = dequantize_tensor(unpack_uint8_to_fp4(w13_u8),
                              w13_scale_f,
                              axis=-1)
    gate, up = w13_f[:, :inter, :], w13_f[:, inter:, :]
    if inter_pad:
        gate = torch.nn.functional.pad(gate, (0, 0, 0, inter_pad))
        up = torch.nn.functional.pad(up, (0, 0, 0, inter_pad))
    w13_idx, w13_s = quantize_tensor_to_fp4(torch.cat([gate, up], dim=1),
                                            axis=-1,
                                            block_size=block)
    w13_out = pack_fp4_indices(w13_idx)  # [E, 2*I_pad, H/2]
    w13_scale_4d = _to_kernel_scale(w13_s.to(torch.float32))

    # w2: dequant block-16 -> [E, H, I] fp32; pad contracting dim I -> I_pad.
    w2_f = dequantize_tensor(unpack_uint8_to_fp4(w2_u8), w2_scale_f, axis=-1)
    if inter_pad:
        w2_f = torch.nn.functional.pad(w2_f, (0, inter_pad))
    w2_idx, w2_s = quantize_tensor_to_fp4(w2_f, axis=-1, block_size=block)
    w2_out = pack_fp4_indices(w2_idx)  # [E, H, I_pad/2]
    w2_scale_4d = _to_kernel_scale(w2_s.to(torch.float32))

    return (_fresh(w13_out), _fresh(w13_scale_4d), _fresh(w2_out),
            _fresh(w2_scale_4d))


def _prepare_moe_weights(layer, w13_scale_f, w2_scale_f, requant_block):
    """Prepare a candidate weight set without replacing checkpoint tensors."""
    w13, w2 = layer.w13_weight.data, layer.w2_weight.data
    if requant_block:
        requant_block = int(requant_block)
        hidden, inter = w13.shape[-1] * 2, w2.shape[-1] * 2
        if hidden % requant_block == 0 and inter % requant_block == 0:
            w13, s13 = requant_load_kmajor_fp4(_fresh(w13), w13_scale_f,
                                               requant_block)
            w2, s2 = requant_load_kmajor_fp4(_fresh(w2), w2_scale_f,
                                             requant_block)
            return (w13, w2, s13,
                    s2), f"FP4 (jax requant block-{requant_block})"
        # Pad each gate/up half and w2's contracting dimension together.
        w13, s13, w2, s2 = _requant_moe_w4a8(w13, w13_scale_f, w2, w2_scale_f,
                                             requant_block)
        mode = f"FP4 (torch requant block-{requant_block})"
    else:
        w13, w2 = _fresh(w13), _fresh(w2)
        s13, s2 = _to_kernel_scale(w13_scale_f), _to_kernel_scale(w2_scale_f)
        mode = "FP4 (block-16)"
    return (load_kmajor_fp4(w13), load_kmajor_fp4(w2), s13, s2), mode


def _prebuild_w4a8(layer, activation, configured_block):
    """Admit the proposed runtime layout while checkpoint weights are intact."""
    from vllm_torchtpu.layers.adapter.fused_moe_ep import prebuild_fused_moe_ep

    if not (envs.USE_MOE_FUSED_EP_KERNEL and envs.MOE_FUSED_EP_ENABLE_W4A8
            and layer.moe_config.moe_parallel_config.use_ep):
        return None, configured_block
    from vllm_torchtpu.kernels.fused_moe.v2.host import PACK4, U32_SUBLANE_TILE

    block = (U32_SUBLANE_TILE *
             PACK4 if configured_block is None else configured_block)
    hidden = layer.w13_weight.shape[-1] * 2
    inter = layer.w2_weight.shape[-1] * 2
    if block <= 0 or hidden % block:
        logger.info_once(
            "NVFP4 fused EP not engaged: weight block-%d cannot "
            "serve hidden=%d; retaining the GMM weight recipe.", block, hidden)
        return None, configured_block
    inter = align_to(inter, block)
    experts = layer.w13_weight.shape[0]
    # Admission reads only shapes/dtypes. Meta tensors allocate no weight
    # storage and describe Torch's packed, K-major FP4 representation.
    weights = (
        torch.empty((experts, hidden, inter),
                    dtype=torch.float4_e2m1fn_x2,
                    device="meta"),
        torch.empty((experts, inter, hidden // 2),
                    dtype=torch.float4_e2m1fn_x2,
                    device="meta"),
        torch.empty((experts, hidden // block, 1, 2 * inter),
                    dtype=torch.float32,
                    device="meta"),
        torch.empty((experts, inter // block, 1, hidden),
                    dtype=torch.float32,
                    device="meta"),
    )
    op = prebuild_fused_moe_ep(layer,
                               topk=layer.moe_config.experts_per_token,
                               renormalize=layer.renormalize,
                               activation=activation,
                               weight_format="fp4",
                               rhs_qb=block,
                               weights=weights)
    return op, block if op is not None else configured_block


@register_quantization_config(get_tpu_quant_method(NVFP4))
class VllmNvfp4Config(ModelOptNvFp4Config, VllmQuantConfig):
    """NVFP4 config for TPU. Inherits ModelOpt checkpoint parsing from upstream;
    dispatches layers to TPU-native W4A16 methods."""

    @classmethod
    def get_name(cls) -> str:
        return NVFP4

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        map_online_fp8(self, hf_to_vllm_mapper)

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> QuantizeMethodBase | None:
        base_method = self._get_checkpoint_quant_method(layer, prefix)
        return resolve_online_fp8(self, layer, prefix, base_method)

    def _get_checkpoint_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, Attention):
            # NVFP4 attention/kv-cache quantization is not implemented on TPU.
            return None
        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import \
                    VllmUnquantizedLinearMethod
                return VllmUnquantizedLinearMethod()
            return VllmNvfp4LinearMethod(self, self.get_linear_config(layer))
        if isinstance(layer, RoutedExperts):
            if self.is_layer_excluded(prefix):
                from vllm_torchtpu.layers.adapter.quantization.unquantized import \
                    VllmUnquantizedFusedMoEMethod
                return VllmUnquantizedFusedMoEMethod(
                    self.get_moe_config(layer))
            return VllmNvfp4MoEMethod(self, self.get_moe_config(layer))
        return None


class VllmNvfp4MoEMethod(TpuMoEActivationMixin, FusedMoEMethodBase):
    """NVFP4 MoE for TPU: W4A16 GMM or opt-in W4A8 fused EP v2.

    Reuses upstream `ModelOptNvFp4FusedMoE.create_weights` for parameter
    registration. The E2M1 weights are unpacked to native fp4 in the kernel's
    K-major layout at load; the FP8 block scale and FP32 global scale are fused
    into one FP32 scale, with optional weight-block requantization. GMM runs
    with `maybe_quantize_lhs=False`; only admitted fused EP layers use W4A8.
    """

    def __init__(self, quant_config: VllmNvfp4Config, moe_config):
        # Skip ModelOptNvFp4FusedMoE.__init__ (it selects a GPU experts backend).
        FusedMoEMethodBase.__init__(self, moe_config)
        self.quant_config = quant_config
        self.group_size = quant_config.group_size
        # Upstream create_weights sizes the input-scale params off this; we drop
        # those scales (W4A16), so any value works -- keep False (no global SF).
        self.use_global_sf = False

    @property
    def is_monolithic(self) -> bool:
        return True

    @property
    def supports_internal_mk(self) -> bool:
        # We need to take control of collective communication (AllGather/ReduceScatter)
        # to pipeline them with MoE computation when chunking is enabled.
        from vllm_torchtpu.layers.adapter.fused_moe_ep import \
            fused_moe_ep_supported
        return (enable_pipelined_collective_and_compute()
                or fused_moe_ep_supported(self))

    def get_fused_moe_quant_config(self, layer):
        return None

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype,
                       **extra_weight_attrs):
        layer.params_dtype = params_dtype
        ModelOptNvFp4FusedMoE.create_weights(self, layer, num_experts,
                                             hidden_size,
                                             intermediate_size_per_partition,
                                             params_dtype,
                                             **extra_weight_attrs)
        # Upstream registers the scale params before updating extra_weight_attrs
        # with `quant_method`, so vLLM's MoE weight loader does not see it on the
        # scales and rejects the load. Set them explicitly.
        from vllm.model_executor.layers.fused_moe import \
            FusedMoeWeightScaleSupported
        from vllm.model_executor.utils import set_weight_attrs
        for name in ("w13_weight_scale", "w2_weight_scale"):
            set_weight_attrs(
                getattr(layer, name),
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        for name in ("w13_weight_scale_2", "w2_weight_scale_2"):
            set_weight_attrs(
                getattr(layer, name),
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})

    def _resolve_tpu_activation(self, layer) -> str:
        activation_str = super()._resolve_tpu_activation(layer)
        if activation_str == "swigluoai":
            raise NotImplementedError(
                "NVFP4 MoE on TPU supports act_and_mul (silu/gelu) layouts; "
                "swigluoai (interleaved) is not implemented.")
        return activation_str

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)
        assert not self.moe.has_bias, "TPU NVFP4 MoE does not support bias."
        assert self.moe.is_act_and_mul, (
            "TPU NVFP4 MoE expects gated (act_and_mul) experts with a "
            "[gate; up] w13 layout and a per-w1/w3 global scale [E, 2].")
        activation_str = self._set_tpu_activation(layer)

        # --- fuse E4M3 block scale * FP32 per-tensor global -> FP32 block scale.
        # w13: global is [E, 2] (separate w1/w3 per-tensor scales); apply to the
        # gate (first half) and up (second half) output rows respectively.
        w13_scale = layer.w13_weight_scale.data  # [E, 2I, H/group] e4m3
        g13 = layer.w13_weight_scale_2.data.to(torch.float32)  # [E, 2]
        half = w13_scale.shape[1] // 2
        s_gate = w13_scale[:, :half, :].to(torch.float32) * g13[:, 0].view(
            -1, 1, 1)
        s_up = w13_scale[:, half:, :].to(torch.float32) * g13[:, 1].view(
            -1, 1, 1)
        w13_scale_f = torch.cat([s_gate, s_up], dim=1)  # [E, 2I, H/group] fp32

        w2_scale = layer.w2_weight_scale.data  # [E, H, I/group] e4m3
        g2 = layer.w2_weight_scale_2.data.to(torch.float32)  # [E]
        w2_scale_f = w2_scale.to(torch.float32) * g2.view(-1, 1, 1)

        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            FUSED_MOE_EP_OP_ATTR, fused_moe_ep_unsupported_reason,
            register_score_bias_buffer)

        # Preserve main's explicit requantization recipe for the GMM path.
        # Automatic requantization is committed only for an admitted W4A8 op.
        configured_block = envs.MOE_REQUANTIZE_BLOCK_SIZE
        op, requant_block = _prebuild_w4a8(layer, activation_str,
                                           configured_block)
        weights, mode = _prepare_moe_weights(layer, w13_scale_f, w2_scale_f,
                                             requant_block)
        if op is not None:
            reason = fused_moe_ep_unsupported_reason(layer,
                                                     weights,
                                                     activation_str,
                                                     weight_format="fp4",
                                                     rhs_qb=requant_block)
            # Admission already accepted the proposed layout. A mismatch in
            # the actual tensors is a preparation bug, not a fallback case.
            assert reason is None, (
                "NVFP4 fused EP prepared weights violate the admitted layout: "
                f"{reason}")
        w13, w2, w13_scale_4d, w2_scale_4d = weights

        for attr in ("w13_weight_scale_2", "w2_weight_scale_2",
                     "w13_input_scale", "w2_input_scale"):
            if hasattr(layer, attr):
                delattr(layer, attr)

        # Install only the admitted W4A8 candidate or the original GMM recipe.
        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale_4d,
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale_4d,
                                                   requires_grad=False)

        if layer.w13_weight.device.type == "tpu":
            synchronize_tensors([
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
            ])

        logger.info_once(
            f"NVFP4 MoE weights prepared ({mode}): "
            f"w13={list(layer.w13_weight.shape)} ({layer.w13_weight.dtype}), "
            f"w2={list(layer.w2_weight.shape)} ({layer.w2_weight.dtype}), "
            f"w13_scale={list(layer.w13_weight_scale.shape)}, "
            f"w2_scale={list(layer.w2_weight_scale.shape)}, group_size={self.group_size}"
        )
        if layer.moe_config.moe_parallel_config.use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(
            layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=layer.moe_config.moe_parallel_config.use_ep,
        )

        setattr(self, FUSED_MOE_EP_OP_ATTR, op)
        if op is not None:
            register_score_bias_buffer(layer)
            # Do the scale layout conversion once, outside compiled forward.
            layer.register_buffer(
                "_tpu_fused_w13_scale",
                layer.w13_weight_scale.squeeze(2).contiguous(),
                persistent=False)
            layer.register_buffer(
                "_tpu_fused_w2_scale",
                layer.w2_weight_scale.squeeze(2).contiguous(),
                persistent=False)
            logger.info_once("NVFP4 fused EP W4A8 enabled: block-%d",
                             requant_block)
        else:
            logger.info_once("NVFP4 GMM W4A16 enabled: block-%d", requant_block
                             or self.group_size)

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation_str = self._tpu_activation_str
        assert activation_str is not None, (
            "[moe] process_weights_after_loading did not run for this layer")
        from vllm_torchtpu.layers.adapter.fused_moe_ep import (
            fused_moe_ep, fused_moe_ep_supported, score_bias_operand)
        if fused_moe_ep_supported(self):
            return fused_moe_ep(self, x, layer.w13_weight, layer.w2_weight,
                                layer._tpu_fused_w13_scale,
                                layer._tpu_fused_w2_scale, router_logits,
                                score_bias_operand(layer))

        # Quantization-independent routing decision (simulation override ->
        # custom_routing_function -> select_experts); shared across all TPU MoE
        # methods so the routing-simulation hook lives in exactly one place.
        #
        # Behavior note: this previously called select_experts() without
        # `layer=`, which silently skipped grouped-topk routing for any NVFP4
        # model with use_grouped_topk=True (that attribute is set on every MoE
        # layer at construction, not specific to a quantization scheme, and
        # the other three TPU MoE methods already passed it). moe_routing.route
        # passes `layer` uniformly, which fixes that -- flagging separately as
        # it's a behavior change, not just a refactor.
        topk_weights, topk_ids = moe_routing.route(layer, x, router_logits)
        kwargs = {
            "hidden_states": x,
            "w1": layer.w13_weight,
            "w2": layer.w2_weight,
            "w1_scale": layer.w13_weight_scale,
            "w2_scale": layer.w2_weight_scale,
            "w1_bias": None,
            "w2_bias": None,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "experts_start": layer._experts_start,
            "topk": layer.moe_config.experts_per_token,
            "activation": activation_str,
        }
        if enable_pipelined_collective_and_compute():
            return pipelined_fused_moe_gmm(**kwargs)

        return fused_moe_gmm(**kwargs)


class _NullInputQuantKernel:
    """Placeholder kernel so upstream create_weights' expose_input_quant_key
    is a no-op. TPU NVFP4 is W4A16: activations are not pre-quantized, so the
    layer advertises no input_quant_key."""

    @staticmethod
    def input_quant_key():
        return None


class VllmNvfp4LinearMethod(ModelOptNvFp4LinearMethod):
    """NVFP4 dense-linear for TPU (W4A16).

    Reuses upstream create_weights for parameter registration; fuses the block
    and global scales in PyTorch and unpacks the weight to native fp4 (K-major)
    at load. apply runs the native-fp4 weight through `quantized_matmul_fp4` ->
    `gmm_v2`.
    """

    def __init__(self, quant_config: VllmNvfp4Config,
                 linear_config: VllmQuantLinearConfig):
        # Skip ModelOptNvFp4LinearMethod.__init__ (it builds a GPU NVFP4 kernel
        # unavailable on TPU). create_weights only needs kernel.input_quant_key.
        self.quant_config = quant_config
        self.linear_config = linear_config
        self.group_size = quant_config.group_size
        self.kernel = _NullInputQuantKernel()

    def create_weights(self, layer, input_size_per_partition,
                       output_partition_sizes, input_size, output_size,
                       params_dtype, **extra_weight_attrs):
        ModelOptNvFp4LinearMethod.create_weights(
            self, layer, input_size_per_partition, output_partition_sizes,
            input_size, output_size, params_dtype, **extra_weight_attrs)

        # input_scale / weight_scale_2 are per-tensor (no output_dim); vLLM's
        # fused QKV/MergedColumn stacked loader cannot place the per-projection
        # scalars and asserts. Load them with a scalar filler (we collapse to a
        # single global scale in process_weights_after_loading anyway).
        def scalar_weight_loader(param, loaded_weight, *args, **kwargs):
            assert loaded_weight.numel() == 1
            param.data.fill_(loaded_weight.item())

        for name in ("input_scale", "weight_scale_2"):
            if hasattr(layer, name):
                getattr(layer, name).weight_loader = scalar_weight_loader

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, LinearBase)
        # Fuse E4M3 block scale * FP32 per-tensor global -> FP32 block scale.
        # weight_scale_2 is the per-tensor global, replicated across the fused
        # output partitions; parallel q/k/v share one value. Assert agreement
        # and take it.
        global_scale = layer.weight_scale_2.data.flatten()
        assert bool((global_scale == global_scale[0]).all()), \
            "Fused NVFP4 projections must share one global scale."
        global_scale = global_scale[0].to(torch.float32)
        weight_scale = layer.weight_scale.data.to(
            torch.float32) * global_scale  # [N, K/group]
        # gmm_v2 rhs scale layout for a single group: [1, num_blocks, 1, N].
        scale_4d = _to_kernel_scale(weight_scale).unsqueeze(
            0)  # [1, K/group, 1, N]

        # Unpack to native fp4 in the kernel's K-major layout once at load (see
        # load_kmajor_fp4); the forward then runs native fp4 with no per-forward
        # unpack. [N, K/2] uint8 -> [K, N] torch.float4_e2m1fn_x2.
        weight = load_kmajor_fp4(_fresh(layer.weight.data))

        for attr in ("weight_scale_2", "input_scale"):
            if hasattr(layer, attr):
                delattr(layer, attr)

        replace_parameter(layer, "weight",
                          torch.nn.Parameter(weight, requires_grad=False))
        replace_parameter(layer, "weight_scale",
                          torch.nn.Parameter(scale_4d, requires_grad=False))

        if layer.weight.device.type == "tpu":
            synchronize_tensors([layer.weight, layer.weight_scale])

        logger.info_once(
            "NVFP4 linear weights prepared (W4A16): "
            f"weight={list(layer.weight.shape)} ({layer.weight.dtype}), "
            f"scale={list(layer.weight_scale.shape)}, group_size={self.group_size}"
        )

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: torch.Tensor | None = None) -> torch.Tensor:
        out = quantized_matmul_fp4(x, layer.weight, layer.weight_scale)
        if bias is not None:
            out = out + bias
        return out
