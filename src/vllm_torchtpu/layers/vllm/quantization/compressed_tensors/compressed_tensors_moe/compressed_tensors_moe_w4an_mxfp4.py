import torch
from torch_tpu._internal import sync
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import \
    CompressedTensorsW4A4Mxfp4MoEMethod

from vllm_torchtpu.layers.common.quantization import e8m0_to_fp32
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 load_kmajor_fp4,
                                                 prebuild_fused_moe_kernel)
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.utils import (
    get_cpu_weight_loader_hook, release_memory_to_os)


class VllmCompressedTensorsW4ANMxfp4MoEMethod(
        CompressedTensorsW4A4Mxfp4MoEMethod):
    """
    TPU compressed-tensors packed-weight W4AN MXFP4 MoE implementation.
    Accommodates both W4A4 and W4A16.
    """

    @property
    def is_monolithic(self) -> bool:
        return True

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        orig_loader = extra_weight_attrs.get("weight_loader")

        # Inject our optimized TPU CPU weight loader hook
        extra_weight_attrs["weight_loader"] = get_cpu_weight_loader_hook(
            layer,
            orig_loader,
            self.moe.tp_size,
            self.moe.tp_rank,
            is_param_transposed=False)

        super().create_weights(layer, num_experts, hidden_size,
                               intermediate_size_per_partition, params_dtype,
                               **extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)

        # Retrieve weights from the CPU scratchpad and move to TPU
        w13_weight_packed = layer.w13_weight_packed._cpu_scratch.data.to("tpu")
        w13_weight_scale = layer.w13_weight_scale._cpu_scratch.data.to("tpu")
        w2_weight_packed = layer.w2_weight_packed._cpu_scratch.data.to("tpu")
        w2_weight_scale = layer.w2_weight_scale._cpu_scratch.data.to("tpu")

        # Convert e8m0 scales to fp32
        w13_scale = e8m0_to_fp32(w13_weight_scale)
        w2_scale = e8m0_to_fp32(w2_weight_scale)

        def _to_kernel_scale(scale: torch.Tensor) -> torch.Tensor:
            # scale shape: [E, N, num_blocks]
            # gmm_v2 scale layout: [E, num_blocks, 1, N]
            return scale.movedim(-1, 1).unsqueeze(-2)

        w13_scale_4d = _to_kernel_scale(w13_scale)
        w2_scale_4d = _to_kernel_scale(w2_scale)

        # Unpack to native fp4 in the kernel's K-major layout once at load
        # [E, N, K/2] uint8 -> [E, K/2, N] torch.float4_e2m1fn_x2
        w13_weight = load_kmajor_fp4(w13_weight_packed)
        w2_weight = load_kmajor_fp4(w2_weight_packed)

        # Clean up CPU scratchpads
        delattr(layer.w13_weight_packed, "_cpu_scratch")
        delattr(layer.w13_weight_scale, "_cpu_scratch")
        delattr(layer.w2_weight_packed, "_cpu_scratch")
        delattr(layer.w2_weight_scale, "_cpu_scratch")

        # Update layer parameters
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale_4d,
                                                    requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale_4d,
                                                   requires_grad=False)

        # Remove packed attributes
        delattr(layer, "w13_weight_packed")
        delattr(layer, "w2_weight_packed")

        if hasattr(layer, "weight"):
            layer.weight = layer.w13_weight
        layer.w13_bias = None
        layer.w2_bias = None

        if layer.w13_weight.device.type == "tpu":
            sync.synchronize(layer.w13_weight, wait=True)
            sync.synchronize(layer.w2_weight, wait=True)
            sync.synchronize(layer.w13_weight_scale, wait=True)
            sync.synchronize(layer.w2_weight_scale, wait=True)

        release_memory_to_os()

        activation_str = (layer.activation if isinstance(
            layer.activation, str) else layer.activation.value)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            experts_start=moe_routing.get_experts_start(layer),
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation_str = (layer.activation if isinstance(
            layer.activation, str) else layer.activation.value)

        # Handle custom routing function if present
        custom_routing_fn = getattr(layer, "custom_routing_function", None)
        if custom_routing_fn is not None:
            topk_weights, topk_ids = custom_routing_fn(
                hidden_states=x,
                gating_output=router_logits,
                topk=layer.moe_config.experts_per_token,
                # renormalize=getattr(layer.moe_config, "norm_topk_prob", getattr(layer, "renormalize", True)))
                renormalize=layer.renormalize)
        else:
            # Fallback to standard vLLM routing if no custom routing function is defined
            topk_weights, topk_ids = moe_routing.select_experts(
                hidden_states=x,
                router_logits=router_logits,
                topk=layer.moe_config.experts_per_token,
                renormalize=layer.renormalize,
                scoring_fn=getattr(layer, "scoring_func", "softmax"),
                layer=layer)

        # Ensure correct type for routing inputs
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        # Execute GMM Kernel with native fp4 weights and scales
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
            experts_start=moe_routing.get_experts_start(layer),
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            rhs_quant_dtype=None,
        )
