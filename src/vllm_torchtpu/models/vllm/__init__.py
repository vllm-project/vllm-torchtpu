_KIMI_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
               "KimiLinearForCausalLM")


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    # Kimi-K3 checkpoints wrap the same language model in a multimodal
    # conditional-generation architecture.  The TPU implementation is
    # text-only for now, but can load the language_model.* weights directly.
    ModelRegistry.register_model("KimiK3ForConditionalGeneration", _KIMI_MODEL)
