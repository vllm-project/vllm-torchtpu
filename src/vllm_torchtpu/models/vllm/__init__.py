from vllm.model_executor.models.config import VerifyAndUpdateConfig

_KIMI_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
               "KimiLinearForCausalLM")
_KIMI_K3_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
                  "KimiK3ForConditionalGeneration")
_DSV4_MODEL = ("vllm_torchtpu.models.vllm.deepseek_v4:"
               "DeepseekV4ForCausalLM")
_K3_DSPARK_MODEL = ("vllm_torchtpu.models.vllm.kimi_k3:"
                    "K3DSparkForCausalLM")


class _TPUKimiConfig(VerifyAndUpdateConfig):
    """Per-architecture ModelConfig hook for the Kimi KDA hybrids.

    ``ModelConfig.__post_init__`` ends by looking the architecture up in
    ``MODELS_CONFIG_MAP`` and calling ``verify_and_update_model_config`` on the
    entry. ``register_models`` installs this class for both
    ``KimiK3ForConditionalGeneration`` and ``KimiLinearForCausalLM``. It has
    two duties.

    Keep K3 MXFP4 checkpoints on TPU's packed-weight loader. Upstream maps K3
    to ``KimiK3ForConditionalGenerationConfig``, which rewrites
    ``quantization_config["quant_method"]`` from ``"compressed-tensors"`` to
    ``"mxfp4"`` on ``hf_config``, ``hf_text_config`` and ``model_arch_config``
    whenever the format is ``"mxfp4-pack-quantized"``, so the routed experts
    resolve to ``Mxfp4Config``. TPU loads those experts through its
    compressed-tensors packed MXFP4 MoE method, so replacing the entry, and
    never touching ``quantization_config`` here, suppresses that rewrite.

    Describe the layer mix in a form upstream can count. Kimi configs name
    their KDA layers in ``linear_attn_config["kda_layers"]`` and carry none of
    the fields ``ModelConfig.get_num_layers_by_block_type`` reads for a hybrid
    (``layers_block_type``, ``attn_type_list``, ``layer_types``), so upstream
    raises ``ValueError`` when the TPU runner asks for the attention layer
    count. Writing ``layers_block_type`` on the text config lets upstream's own
    counter, and its pipeline-stage slicing, answer for both block types.
    """

    @staticmethod
    def verify_and_update_model_config(model_config) -> None:
        # KDA metadata lives on the text config; for K3 that is the nested
        # ``text_config`` of the multimodal wrapper, not ``hf_config``.
        config = model_config.hf_text_config
        # Metadata upstream can already count wins, whether it came from the
        # checkpoint, from ``hf_overrides`` or from an earlier run of this hook
        # on the same config, which makes the hook idempotent.
        if (getattr(config, "layers_block_type", None) is not None
                or getattr(config, "layer_types", None) is not None
                or getattr(model_config.hf_config, "attn_type_list",
                           None) is not None):
            return
        linear_attn_config = getattr(config, "linear_attn_config", None)
        if linear_attn_config is None:
            return
        # Kimi numbers KDA layers from one (``KimiLinearConfig.is_kda_layer``
        # tests ``layer_idx + 1``), while upstream slices this list with
        # zero-based pipeline-stage bounds. Its length is the total those
        # bounds are derived from, so every stage slice stays in range.
        kda_layers = set(linear_attn_config.get("kda_layers") or ())
        config.layers_block_type = [
            "linear_attention" if idx + 1 in kda_layers else "attention"
            for idx in range(model_config.get_total_num_hidden_layers())
        ]


def register_models() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    ModelRegistry.register_model("KimiLinearForCausalLM", _KIMI_MODEL)
    ModelRegistry.register_model("KimiK3ForConditionalGeneration",
                                 _KIMI_K3_MODEL)
    ModelRegistry.register_model("DeepseekV4ForCausalLM", _DSV4_MODEL)
    ModelRegistry.register_model("K3DSparkModel", _K3_DSPARK_MODEL)
    MODELS_CONFIG_MAP["KimiK3ForConditionalGeneration"] = _TPUKimiConfig
    # Upstream has no entry for Kimi Linear, so this one adds rather than
    # replaces. KimiK3MTPModel keeps upstream's handler, and the K3DSparkModel
    # drafter is dense MLA, not a hybrid, so upstream counts its layers as is.
    MODELS_CONFIG_MAP["KimiLinearForCausalLM"] = _TPUKimiConfig
