import torch
from torch_tpu._internal import sync
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a4_mxfp4 import \
    CompressedTensorsW4A4Mxfp4MoEMethod

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.common.quantization import e8m0_to_fp32
from vllm_torchtpu.layers.vllm import moe_routing
from vllm_torchtpu.layers.vllm.fused_moe import (fused_moe_gmm,
                                                 get_fused_moe_activation,
                                                 load_kmajor_fp4,
                                                 prebuild_fused_moe_kernel,
                                                 requant_load_kmajor_fp4)
from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.utils import (
    get_cpu_weight_loader_hook, release_memory_to_os)


def _fresh(t: torch.Tensor) -> torch.Tensor:
    """Allocate a fresh contiguous device buffer (breaks view/stride chains so
    the torch_tpu pallas boundary ships a plain row-major buffer)."""
    return torch.empty(t.shape, dtype=t.dtype, device=t.device).copy_(t)


class VllmCompressedTensorsW4ANMxfp4MoEMethod(
        CompressedTensorsW4A4Mxfp4MoEMethod):
    """
    TPU compressed-tensors packed-weight W4AN MXFP4 MoE implementation.
    Accommodates both W4A4 and W4A16.
    """

    @property
    def is_monolithic(self) -> bool:
        return True

    @staticmethod
    def _loaded_data(parameter: torch.nn.Parameter) -> torch.Tensor:
        """Return checkpoint scratch data or an already-materialized weight.

        Normal checkpoint loading uses the CPU scratchpad installed by
        ``get_cpu_weight_loader_hook``. Other loaders, notably vLLM's dummy
        loader, materialize the parameter directly and never invoke that
        hook.
        """
        scratch = getattr(parameter, "_cpu_scratch", None)
        return parameter.data if scratch is None else scratch.data

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

    def _neutralize_padded_scales(self, layer: RoutedExperts) -> None:
        """Make unloaded E8M0 padding finite without initializing full buffers.

        The checkpoint loader only writes the logical intermediate width into
        the ``torch.empty`` CPU scratchpads. An untouched padding byte can be
        ``0xff``, which E8M0 conversion maps to infinity and can turn a padded
        zero weight into NaN. Zero only those unloaded scale slices in place;
        do not replace the full parameter/scratch allocation with
        ``torch.zeros``, whose extra initialization caused peak-memory OOMs for
        large models.
        """
        unpadded_size = self.moe.intermediate_size_per_partition_unpadded
        padded_size = self.moe.intermediate_size_per_partition
        if unpadded_size >= padded_size:
            return
        if unpadded_size % self.group_size != 0:
            raise ValueError(
                "MXFP4 unpadded intermediate size per partition must be "
                f"divisible by {self.group_size}, got {unpadded_size}.")

        w13_scale = self._loaded_data(layer.w13_weight_scale)
        w2_scale = self._loaded_data(layer.w2_weight_scale)
        w13_scale[:, unpadded_size:padded_size, :].zero_()
        w13_scale[:, padded_size + unpadded_size:2 * padded_size, :].zero_()
        w2_scale[:, :, unpadded_size // self.group_size:].zero_()

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        assert isinstance(layer, RoutedExperts)

        self._neutralize_padded_scales(layer)

        # Retrieve scratchpad or directly materialized weights and move to TPU.
        w13_weight_packed = self._loaded_data(
            layer.w13_weight_packed).to("tpu")
        w13_weight_scale = self._loaded_data(layer.w13_weight_scale).to("tpu")
        w2_weight_packed = self._loaded_data(layer.w2_weight_packed).to("tpu")
        w2_weight_scale = self._loaded_data(layer.w2_weight_scale).to("tpu")

        # Convert e8m0 scales to fp32
        w13_scale = e8m0_to_fp32(w13_weight_scale)
        w2_scale = e8m0_to_fp32(w2_weight_scale)

        requant_block = envs.MOE_REQUANTIZE_BLOCK_SIZE
        if requant_block is not None:
            requant_block = int(requant_block)

            # The last dim of packed weights is K/2. So K is shape[-1] * 2.
            hidden = w13_weight_packed.shape[-1] * 2
            inter = w2_weight_packed.shape[-1] * 2
            if hidden % requant_block != 0 or inter % requant_block != 0:
                raise ValueError(
                    f"W4A8 requantization needs hidden ({hidden}) and inter ({inter}) "
                    f"to be divisible by block ({requant_block}). Padding is not yet supported."
                )

            # Apply XLA fused requantization and kmajor layout shift
            w13_weight, w13_scale_4d = requant_load_kmajor_fp4(
                _fresh(w13_weight_packed), w13_scale, requant_block)
            w2_weight, w2_scale_4d = requant_load_kmajor_fp4(
                _fresh(w2_weight_packed), w2_scale, requant_block)
        else:

            def _to_kernel_scale(scale: torch.Tensor) -> torch.Tensor:
                # scale shape: [E, N, num_blocks]
                # gmm_v2 scale layout: [E, num_blocks, 1, N]
                return _fresh(scale.movedim(-1, 1).unsqueeze(-2))

            w13_scale_4d = _to_kernel_scale(w13_scale)
            w2_scale_4d = _to_kernel_scale(w2_scale)

            # Unpack to native fp4 in the kernel's K-major layout once at load
            # [E, N, K/2] uint8 -> [E, K/2, N] torch.float4_e2m1fn_x2
            w13_weight = load_kmajor_fp4(_fresh(w13_weight_packed))
            w2_weight = load_kmajor_fp4(_fresh(w2_weight_packed))

        # Clean up CPU scratchpads
        for parameter in (layer.w13_weight_packed, layer.w13_weight_scale,
                          layer.w2_weight_packed, layer.w2_weight_scale):
            if hasattr(parameter, "_cpu_scratch"):
                delattr(parameter, "_cpu_scratch")

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

        activation_str = get_fused_moe_activation(layer.activation,
                                                  layer.moe_config)
        layer._tpu_activation_str = activation_str
        use_ep = layer.moe_config.moe_parallel_config.use_ep
        if use_ep:
            moe_routing.validate_linear_ep_placement(layer)
        moe_routing.register_experts_start_buffer(
            layer, device=layer.w13_weight.device)
        prebuild_fused_moe_kernel(
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            use_ep=use_ep,
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        activation_str = layer._tpu_activation_str

        # Handle custom routing function if present
        custom_routing_fn = getattr(layer, "custom_routing_function", None)
        if custom_routing_fn is not None:
            topk_weights, topk_ids = custom_routing_fn(
                hidden_states=x,
                gating_output=moe_routing.maybe_force_random_routing(
                    router_logits),
                topk=layer.moe_config.experts_per_token,
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
            experts_start=layer._experts_start,
            topk=layer.moe_config.experts_per_token,
            activation=activation_str,
            rhs_quant_dtype=None,
        )
