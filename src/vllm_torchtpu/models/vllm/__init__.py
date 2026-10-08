from vllm.model_executor.models.config import VerifyAndUpdateConfig

_KIMI_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiLinearForCausalLM"
_KIMI_K3_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:KimiK3ForConditionalGeneration"
_DSV4_MODEL = "vllm_torchtpu.models.vllm.deepseek_v4:DeepseekV4ForCausalLM"
_K3_DSPARK_MODEL = "vllm_torchtpu.models.vllm.kimi_k3:K3DSparkForCausalLM"
_DSV4_DSPARK_MODEL = (
    "vllm_torchtpu.models.vllm.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM"
)
_GLM5_NEXT_MODEL = (
    "vllm_torchtpu.models.vllm.glm5_next:Glm5NextForConditionalGeneration"
)
_GLM5_NEXT_CAUSAL_MODEL = "vllm_torchtpu.models.vllm.glm5_next:Glm5NextForCausalLM"


class _TPUKimiK3Config(VerifyAndUpdateConfig):
    """Keep K3 MXFP4 checkpoints on TPU's packed-weight loader."""


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        from vllm_torchtpu.models.vllm.glm5_next.model import Glm5NextConfig

        CONFIG_MAPPING._extra_content["glm5_next"] = Glm5NextConfig
        try:
            from vllm.transformers_utils import config as vllm_tf_config

            if hasattr(vllm_tf_config, "_CONFIG_REGISTRY"):
                vllm_tf_config._CONFIG_REGISTRY["glm5_next"] = Glm5NextConfig
        except Exception:
            pass
    except Exception as exc:
        import sys

        print(
            f"[vllm_torchtpu] Failed to register Glm5NextConfig: {exc}", file=sys.stderr
        )

    # Register TpuKimiGatedDeltaNetAttention out-of-tree layer for GLM 5.3 Flash KDA
    import vllm_torchtpu.layers.adapter.custom_ops.glm53_kda_vllm_layer_tpu  # noqa: F401

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    ModelRegistry.register_model("K3DSparkModel", _K3_DSPARK_MODEL)
    ModelRegistry.register_model("DSparkDraftModel", _DSV4_DSPARK_MODEL)
    ModelRegistry.register_model("Glm5NextForConditionalGeneration", _GLM5_NEXT_MODEL)
    ModelRegistry.register_model("Glm5NextForCausalLM", _GLM5_NEXT_CAUSAL_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiK3Config
