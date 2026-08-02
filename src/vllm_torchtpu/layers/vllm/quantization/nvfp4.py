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
quantized-matmul kernels in **W4A16** mode (bf16 activations, fp4 weights
dequantized per block inside the kernel).

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
    `maybe_quantize_lhs=False` keeps activations in bf16 (W4A16); the GMM
    kernel's unquantized-lhs path dequantizes the fp4 weight per block.

Requires torch_tpu's native `torch.float4_e2m1fn_x2` dtype
(google-pytorch/torch_tpu#1560).
"""

from typing import Optional

import torch
from torch_tpu._internal import sync
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (FusedMoEMethodBase,
                                                  RoutedExperts)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import \
    QuantizeMethodBase
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptNvFp4Config, ModelOptNvFp4FusedMoE, ModelOptNvFp4LinearMethod)
from vllm.model_executor.utils import replace_parameter

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.common.quant_methods import (NVFP4,
                                                       get_tpu_quant_method)
from vllm_torchtpu.layers.common.quantization import (dequantize_tensor,
                                                      pack_fp4_indices,
                                                      quantize_tensor_to_fp4,
                                                      unpack_uint8_to_fp4)
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 get_fused_moe_activation,
                                                 load_kmajor_fp4,
                                                 prebuild_fused_moe_kernel,
                                                 requant_load_kmajor_fp4)
from vllm_torchtpu.layers.vllm.linear_common import quantized_matmul_fp4
from vllm_torchtpu.layers.vllm.quantization.configs import (
    VllmQuantConfig, VllmQuantLinearConfig)
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.utils import align_to

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
    for W4A8 (block >= MXU -> native fp4xfp8 matmul, no in-kernel dequant).

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
        f"W4A8 requant needs hidden ({H}) divisible by block ({block}).")
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


@register_quantization_config(get_tpu_quant_method(NVFP4))
class VllmNvfp4Config(ModelOptNvFp4Config, VllmQuantConfig):
    """NVFP4 config for TPU. Inherits ModelOpt checkpoint parsing from upstream;
    dispatches layers to TPU-native W4A16 methods."""

    @classmethod
    def get_name(cls) -> str:
        return NVFP4

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, Attention):
            # NVFP4 attention/kv-cache quantization is not implemented on TPU.
            return None
        if isinstance(layer, LinearBase):
            if self.is_layer_excluded(prefix):
                return UnquantizedLinearMethod()
            return VllmNvfp4LinearMethod(self, self.get_linear_config(layer))
        if isinstance(layer, RoutedExperts):
            if self.is_layer_excluded(prefix):
                from vllm_torchtpu.layers.vllm.quantization.unquantized import \
                    VllmUnquantizedFusedMoEMethod
                return VllmUnquantizedFusedMoEMethod(
                    self.get_moe_config(layer))
            return VllmNvfp4MoEMethod(self, self.get_moe_config(layer))
        return None


class VllmNvfp4MoEMethod(FusedMoEMethodBase):
    """NVFP4 MoE for TPU (W4A16).

    Reuses upstream `ModelOptNvFp4FusedMoE.create_weights` for parameter
    registration. The E2M1 weights are unpacked to native fp4 in the kernel's
    K-major layout at load; the FP8 block scale and FP32 global scale are fused
    into one FP32 block-16 scale and the matmul runs through `fused_moe_gmm` ->
    `gmm_v2` with `maybe_quantize_lhs=False`.
    """

    def __init__(self, quant_config: 'VllmNvfp4Config', moe_config):
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

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)
        assert not self.moe.has_bias, "TPU NVFP4 MoE does not support bias."
        assert self.moe.is_act_and_mul, (
            "TPU NVFP4 MoE expects gated (act_and_mul) experts with a "
            "[gate; up] w13 layout and a per-w1/w3 global scale [E, 2].")
        activation_str = get_fused_moe_activation(layer.activation,
                                                  layer.moe_config)
        if activation_str == "swigluoai":
            raise NotImplementedError(
                "NVFP4 MoE on TPU supports act_and_mul (silu/gelu) layouts; "
                "swigluoai (interleaved) is not implemented.")
        layer._tpu_activation_str = activation_str

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

        # MOE_REQUANTIZE_BLOCK_SIZE (e.g. 512) requantizes the block-16 fp4
        # weights to that block (>= MXU). This cuts the scale data ~32x and skips
        # the in-kernel dequant, ~halving decode TPOT vs the block-16 default
        # (480B-Coder: 54.7ms -> 29.6ms) at equal accuracy. Activations stay
        # bf16 (the gmm keeps fp8 activations off for fp4 weights -- fp8xfp4
        # collapses accuracy with no decode gain; see fused_moe_gmm.gmm_wrapper).
        # Unset -> packed block-16 weights, dequant in kernel. See PR #306.
        requant_block = envs.MOE_REQUANTIZE_BLOCK_SIZE
        already_kmajor = False
        if requant_block:
            requant_block = int(requant_block)
            hidden = layer.w13_weight.data.shape[-1] * 2  # w13 contracting (H)
            inter = layer.w2_weight.data.shape[-1] * 2  # w2 contracting (I)
            if hidden % requant_block == 0 and inter % requant_block == 0:
                # Dequant block-16 -> requant block-`requant_block` -> K-major
                # native fp4, all in JAX (matches tpu-inference's
                # process_quantized_moe_weights). Returns native fp4 + kernel
                # scale directly, so no separate load_kmajor_fp4 below.
                w13, w13_scale_4d = requant_load_kmajor_fp4(
                    _fresh(layer.w13_weight.data), w13_scale_f, requant_block)
                w2, w2_scale_4d = requant_load_kmajor_fp4(
                    _fresh(layer.w2_weight.data), w2_scale_f, requant_block)
                already_kmajor = True
                mode = f"W4 (jax requant block-{requant_block} + bf16 act)"
            else:
                # Contracting dim not divisible by the block (e.g. Qwen3-30B w2
                # I=768): pad + requant in torch.
                w13, w13_scale_4d, w2, w2_scale_4d = _requant_moe_w4a8(
                    layer.w13_weight.data, w13_scale_f, layer.w2_weight.data,
                    w2_scale_f, requant_block)
                mode = f"W4 (torch requant block-{requant_block} + bf16 act)"
        else:
            # Lay block-16 scales out for gmm_v2: [E, num_blocks_over_K, 1, N]
            # (already K-major-aligned). Weights are the checkpoint-layout packed
            # uint8 [E, N, K/2]; they are unpacked + transposed to K-major native
            # fp4 below (load_kmajor_fp4).
            w13_scale_4d = _to_kernel_scale(w13_scale_f)  # [E, H/group, 1, 2I]
            w2_scale_4d = _to_kernel_scale(w2_scale_f)  # [E, I/group, 1, H]
            w13 = _fresh(layer.w13_weight.data)  # [E, 2I, H/2] uint8
            w2 = _fresh(layer.w2_weight.data)  # [E, H, I/2] uint8
            mode = "W4A16"

        for attr in ("w13_weight_scale_2", "w2_weight_scale_2",
                     "w13_input_scale", "w2_input_scale"):
            if hasattr(layer, attr):
                delattr(layer, attr)

        # Store the weights as native fp4 (torch.float4_e2m1fn_x2) in gmm_v2's
        # K-major [E, K, N] layout. The W4A8 jax-requant path already returns
        # native fp4 K-major; the W4A16 (and torch-requant) paths unpack +
        # transpose the packed uint8 here, ONCE at load, so the forward hands
        # the weight straight to gmm_v2 with no per-forward unpack or transpose.
        if not already_kmajor:
            w13 = load_kmajor_fp4(w13)  # [E, N, K/2] uint8 -> fp4 [E, K, N]
            w2 = load_kmajor_fp4(w2)

        layer.w13_weight = torch.nn.Parameter(w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale_4d,
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale_4d,
                                                   requires_grad=False)

        if layer.w13_weight.device.type == "tpu":
            sync.synchronize(layer.w13_weight, wait=True)
            sync.synchronize(layer.w2_weight, wait=True)
            sync.synchronize(layer.w13_weight_scale, wait=True)
            sync.synchronize(layer.w2_weight_scale, wait=True)

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

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation_str = layer._tpu_activation_str
        custom_routing_fn = layer.custom_routing_function
        if custom_routing_fn is not None:
            # custom_routing_fn bypasses select_experts, so apply the
            # random-routing profiling override here too (a no-op by default).
            topk_weights, topk_ids = custom_routing_fn(
                hidden_states=x,
                gating_output=moe_routing.maybe_force_random_routing(
                    router_logits),
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
            )
        else:
            topk_weights, topk_ids = moe_routing.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                scoring_fn=layer.scoring_func,
            )
        return fused_moe_gmm(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=None,
            w2_bias=None,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            experts_start=layer._experts_start,
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
        )


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

    def __init__(self, quant_config: 'VllmNvfp4Config',
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
            sync.synchronize(layer.weight, wait=True)
            sync.synchronize(layer.weight_scale, wait=True)

        logger.info_once(
            "NVFP4 linear weights prepared (W4A16): "
            f"weight={list(layer.weight.shape)} ({layer.weight.dtype}), "
            f"scale={list(layer.weight_scale.shape)}, group_size={self.group_size}"
        )

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = quantized_matmul_fp4(x, layer.weight, layer.weight_scale)
        if bias is not None:
            out = out + bias
        return out
