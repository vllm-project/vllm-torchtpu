"""V4.1 model bindings for the shared DeepSeek V4 TPU quantization path."""

from typing import TYPE_CHECKING

from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.models.deepseek_v4.nvidia import model as v4_upstream
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MoE

from vllm_torchtpu.layers.adapter.custom_ops.moe_runner import VllmMoERunner
from vllm_torchtpu.layers.adapter.fused_moe_ep import fused_moe_ep_supported
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8MoEMethodTPU
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import (
    VllmDeepseekV4Mxfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.upstream_bindings import device_hooks
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    from .mxfp4 import VllmDeepseekV4Mxfp4MoEMethod


@register_quantization_config("tpu-deepseek_v41_fp8")
class VllmDeepseekV41Fp8Config(VllmDeepseekV4Fp8Config):
    """Use V4 weight preparation, activation quantization and MoE dispatch."""

    def create_mxfp4_moe_method(
        self, moe_config: "FusedMoEConfig"
    ) -> "VllmDeepseekV4Mxfp4MoEMethod":
        from vllm.config import get_current_vllm_config

        from vllm_torchtpu.models.vllm.deepseek_v41.config import (
            DeepseekV41Config,
            MoEBackend,
        )

        from .mxfp4 import VllmDeepseekV4Mxfp4MoEMethod

        policy = DeepseekV41Config.from_vllm_config(get_current_vllm_config())
        return VllmDeepseekV4Mxfp4MoEMethod(
            moe_config, fused_ep=policy.moe == MoEBackend.FUSED_EP_W4A8
        )

    @classmethod
    def get_name(cls):
        return "deepseek_v41_fp8"


class DeepseekV41MoE(DeepseekV4MoE):
    # The TPU factory and device hooks construct these concrete modules.
    experts: VllmMoERunner  # pyrefly: ignore[bad-override-mutable-attribute]
    # pyrefly: ignore[bad-override-mutable-attribute]
    shared_experts: DeepseekV4MLP | None

    def __init__(self, vllm_config, *args, **kwargs):
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError("V4.1 TPU MoE does not support EPLB")
        if vllm_config.kernel_config.moe_backend in v4_upstream.MEGA_MOE_BACKENDS:
            raise NotImplementedError("V4.1 TPU requires the native V4 MoE backend")
        factory = v4_upstream.FusedMoEFactory

        def v4_factory(**options):
            # Match the native TPU V4 model's scaling and shared-expert combine.
            options["apply_routed_scale_to_output"] = True
            return factory(**options)

        with device_hooks(
            v4_upstream,
            DeepseekV4MLP=DeepseekV4MLP,
            FusedMoEFactory=v4_factory,
        ):
            super().__init__(vllm_config, *args, **kwargs)

    def supports_fp8_hidden_input(self) -> bool:
        method = self.experts._quant_method
        return (
            isinstance(method, (VllmDeepseekV4Mxfp4MoEMethod, VllmFp8MoEMethodTPU))
            and fused_moe_ep_supported(method)
            and not self.experts.do_naive_dispatch_combine
            and self.experts.routed_input_transform is None
        )

    def forward(
        self,
        hidden_states,
        input_ids=None,
        mega_gate_metadata=None,
        *,
        hidden_states_q=None,
        hidden_states_scale=None,
    ):
        if (hidden_states_q is None) != (hidden_states_scale is None):
            raise ValueError("routed prequantized inputs must be paired")
        if (
            self.use_sequence_parallel
            and input_ids is not None
            and self.gate.tid2eid is not None
            and self.experts.do_naive_dispatch_combine
        ):
            # Hash routing indexes its table by token id, so the ids must line
            # up row-for-row with the hidden states the experts see. Under SP
            # the runner gathers those over the EP group (TpuDeviceCommunicator)
            # but passes input_ids through as this rank's shard.
            input_ids = get_ep_group().all_gather(input_ids, dim=0)
        if hidden_states_q is not None:
            # This bypasses upstream forward, including its hash/vision guards.
            if self.gate.tid2eid is not None and input_ids is None:
                raise ValueError("DeepSeek V4 hash MoE routing requires input_ids.")
            if getattr(self.gate, "bias_vl", None) is not None and input_ids is None:
                raise ValueError("DeepSeek V4 vision MoE routing requires input_ids.")
            # Upstream passes hidden_states as a placeholder only because its
            # factory gives the runner the gate. An external gate must instead
            # produce real logits from BF16, never from the FP8 side operands.
            router_logits = None
            if self.experts.gate is None:
                router_logits, _ = self.gate(hidden_states)
            output = self.experts.forward_prequantized(
                hidden_states,
                router_logits,
                input_ids=input_ids,
                hidden_states_q=hidden_states_q,
                hidden_states_scale=hidden_states_scale,
            )
            return output.view(hidden_states.shape)
        return super().forward(hidden_states, input_ids, mega_gate_metadata)
