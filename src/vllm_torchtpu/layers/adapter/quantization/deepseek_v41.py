"""V4.1 model bindings for the shared DeepSeek V4 TPU quantization path."""

from typing import TYPE_CHECKING

from vllm.distributed.parallel_state import get_ep_group
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.nvidia import model as v4_upstream
from vllm.models.deepseek_v41.nvidia.model import DeepseekV4MoE

from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.upstream_bindings import device_hooks
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.deepseek_v4.layers import DeepseekV4MLP

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig

    from .mxfp4 import VllmDeepseekV4Mxfp4MoEMethod

logger = init_logger(__name__)


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
    def __init__(self, vllm_config, *args, **kwargs):
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError(
                "V4.1 TPU MoE does not support EPLB (vLLM's dynamic rebalancing); "
                "a static expert layout with replica slots is selected with "
                "additional_config.deepseek_v41.expert_layout instead"
            )
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
        # Only upstream's router gets bias_vl; mirror it onto the experts for
        # `moe_routing.select_experts`, as DeepseekV4MoE does with its hash
        # table. The DSpark loader maps draft layers' bias_vl to their gate
        # bias, so theirs is never loaded.
        if (
            self.gate.bias_vl is not None
            and extract_layer_index(self.prefix)
            < vllm_config.model_config.hf_config.num_hidden_layers
        ):
            experts = self.experts.routed_experts
            lo = self.image_sentinel_lo
            object.__setattr__(experts, "bias_vl", self.gate.bias_vl)
            object.__setattr__(experts, "image_sentinel_lo", lo)
            # Upstream's router matches the five sentinel ids from lo.
            object.__setattr__(experts, "image_sentinel_hi", lo + 5)
            from vllm_torchtpu.models.vllm.deepseek_v41.config import (
                DeepseekV41Config,
                MoEBackend,
            )

            policy = DeepseekV41Config.from_vllm_config(vllm_config)
            if policy.moe == MoEBackend.FUSED_EP_W4A8:
                logger.warning_once(
                    "DeepSeek-V4.1 fused EP MoE routes image tokens like text; "
                    "bias_vl is applied only with moe=gmm"
                )

    def forward(self, hidden_states, input_ids=None, mega_gate_metadata=None):
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
        return super().forward(hidden_states, input_ids, mega_gate_metadata)
