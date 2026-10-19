from typing import TYPE_CHECKING

from vllm.model_executor.models.config import (
    DeepseekV4ForCausalLMConfig,
    VerifyAndUpdateConfig,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

_KIMI_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiLinearForCausalLM"
_KIMI_K3_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiK3ForConditionalGeneration"
_DSV4_MODEL = "vllm_torchtpu.models.vllm.deepseek_v4:DeepseekV4ForCausalLM"
_K3_DSPARK_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:K3DSparkForCausalLM"
_DSV4_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
)
_DSV41_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v41.v2.dspark:DSparkDeepseekV41ForCausalLM"
)


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


class _TPUDeepseekV41Config(DeepseekV4ForCausalLMConfig):
    @staticmethod
    def verify_and_update_config(config: "VllmConfig") -> None:
        from vllm_torchtpu.models.vllm.deepseek_v41.config import (
            verify_and_update_config,
        )

        verify_and_update_config(config)


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    v41_model = "vllm_torchtpu.models.vllm.deepseek_v41.v2.model"
    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    ModelRegistry.register_model(
        "DeepseekV41ForCausalLM", f"{v41_model}:DeepseekV41ForCausalLM"
    )
    ModelRegistry.register_model("K3DSparkModel", _K3_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkDraftModel", _DSV4_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkV41DraftModel", _DSV41_DSPARK_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
    MODELS_CONFIG_MAP["DeepseekV41ForCausalLM"] = _TPUDeepseekV41Config
