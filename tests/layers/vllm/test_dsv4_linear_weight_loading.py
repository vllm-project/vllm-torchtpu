from vllm.model_executor.layers.quantization.fp8 import Fp8Config

from vllm_torchtpu.layers.common.quant_methods import DEEPSEEK_V4_FP8
from vllm_torchtpu.layers.vllm.quantization.fp8 import VllmFp8LinearMethodTPU


class _DummyDeepseekV4Fp8Config(Fp8Config):
    """Stands in for VllmDeepseekV4Fp8Config's get_name() without pulling in
    its real construction (which needs a live vllm_config for is_scale_e8m0).
    The per-channel-FP8-for-dense-layers override in VllmFp8LinearMethodTPU
    is gated on get_name() == DEEPSEEK_V4_FP8, since it's DSV4-specific
    behavior, not something every FP8 model/caller should get by default."""

    @classmethod
    def get_name(cls) -> str:
        return DEEPSEEK_V4_FP8


def test_linear_weight_loading_quant_config_resolving():
    # Setup global dummy config
    dummy_quant_config = _DummyDeepseekV4Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[1, 32],
    )

    # 1. Test Dense/Non-Expert layer (attention/projections)
    dense_prefix = "model.layers.0.self_attn.fused_wqa_wkv"
    dense_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix=dense_prefix,
    )

    # Non-expert layer MUST have block_quant disabled
    assert dense_method.block_quant is False, f"Expected block_quant to be False for prefix {dense_prefix!r}"
    assert dense_method.weight_block_size is None, f"Expected weight_block_size to be None for prefix {dense_prefix!r}"
    assert dense_method._linear_quant_config[
        2] is None, f"Expected requant_block_size to be None for prefix {dense_prefix!r}"

    # 2. Test Routed Expert layer
    expert_prefix = "model.layers.0.mlp.experts.0.w13"
    expert_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix=expert_prefix,
    )

    # Expert layer MUST have block_quant enabled
    assert expert_method.block_quant is True, f"Expected block_quant to be True for prefix {expert_prefix!r}"
    assert expert_method.weight_block_size == [
        1, 32
    ], f"Expected weight_block_size to be [1, 32] for prefix {expert_prefix!r}"

    print(
        "VllmFp8LinearMethodTPU weight loading quant config resolving test passed successfully!"
    )


def test_linear_weight_loading_unquantized_fallback():
    import torch
    dummy_quant_config = _DummyDeepseekV4Fp8Config(
        is_checkpoint_fp8_serialized=False,
        activation_scheme="dynamic",
        weight_block_size=None,
    )
    dense_method = VllmFp8LinearMethodTPU(
        quant_config=dummy_quant_config,
        prefix="model.layers.0.self_attn.fused_wqa_wkv",
    )
    layer = torch.nn.Linear(64, 32, bias=False)
    dense_method.process_weights_after_loading(layer)
    assert hasattr(layer, "weight")
    assert hasattr(layer, "weight_scale")
    assert layer.weight.dtype == torch.float8_e4m3fn
    assert layer.weight_block_size == (1, 64)


if __name__ == "__main__":
    test_linear_weight_loading_quant_config_resolving()
    test_linear_weight_loading_unquantized_fallback()
