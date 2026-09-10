from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts

from vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import \
    VllmCompressedTensorsW4ANMxfp4MoEMethod


class FakeRoutedExperts(RoutedExperts):

    def __init__(self, experts_per_token=2):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = False
        self.moe_config.norm_topk_prob = True
        self.moe_config.has_bias = False
        self.activation = "silu"
        self.use_grouped_topk = False
        self.custom_routing_function = None
        self.scoring_func = "softmax"
        self.e_score_correction_bias = None
        self.routed_scaling_factor = 1.0
        self.renormalize = True

    def _map_global_expert_id_to_local_expert_id(self, expert_id):
        return expert_id


def test_mxfp4_create_weights_and_process():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.experts_per_token = 2
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = 64

    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts(experts_per_token=2)
    layer._expert_routing_tables = lambda: (None, None)

    num_experts = 4
    hidden_size = 128
    intermediate_size_per_partition = 64

    # 1. create_weights
    method.create_weights(
        layer=layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        params_dtype=torch.bfloat16,
    )

    # Validate that extra_weight_attrs["weight_loader"] correctly captured get_tpu_cpu_weight_loader_hook
    assert hasattr(layer, "w13_weight_packed")
    assert hasattr(layer, "w13_weight_scale")
    assert hasattr(layer, "w2_weight_packed")
    assert hasattr(layer, "w2_weight_scale")

    # Simulate loading weights via the hook
    weight_loader = layer.w13_weight_packed.weight_loader
    assert weight_loader is not None

    # Load fake MXFP4 weights
    # w1/w3 checkpoint shape: [intermediate_size_per_partition, hidden_size // 2] -> [64, 64]
    # w1/w3 scale checkpoint shape: [intermediate_size_per_partition, hidden_size // 32] -> [64, 4]
    # w2 checkpoint shape: [hidden_size, intermediate_size_per_partition // 2] -> [128, 32]
    # w2 scale checkpoint shape: [hidden_size, intermediate_size_per_partition // 32] -> [128, 2]
    w13_packed_val = torch.ones(64, 64, dtype=torch.uint8)
    w13_scale_val = torch.ones(64, 4, dtype=torch.uint8)
    w2_packed_val = torch.ones(128, 32, dtype=torch.uint8)
    w2_scale_val = torch.ones(128, 2, dtype=torch.uint8)

    for expert_id in range(num_experts):
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w1", expert_id)
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w3", expert_id)

        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w1", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w3", expert_id)

        weight_loader(layer.w2_weight_packed, w2_packed_val,
                      "w2_weight_packed", "", expert_id)
        weight_loader(layer.w2_weight_scale, w2_scale_val, "w2_weight_scale",
                      "", expert_id)

    # 2. process_weights_after_loading
    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.prebuild_fused_moe_kernel"
    ) as mock_prebuild:
        method.process_weights_after_loading(layer)
    mock_prebuild.assert_called_once_with(topk=2,
                                          activation="silu",
                                          use_ep=False,
                                          skip_padded_tokens=True)

    # Validate output
    assert hasattr(layer, "w13_weight")
    assert hasattr(layer, "w2_weight")
    assert hasattr(layer, "w13_weight_scale")
    assert hasattr(layer, "w2_weight_scale")
    assert not hasattr(layer, "w13_weight_packed")

    # w13 native fp4 K-major: [E, K, N] -> N=128, K=128, last dim halved for float4_e2m1fn_x2 -> [4, 128, 64]
    assert layer.w13_weight.shape == (4, 128, 64)
    # w2 native fp4 K-major: [E, K, N] -> N=128, K=64, last dim halved for float4_e2m1fn_x2 -> [4, 64, 64]
    assert layer.w2_weight.shape == (4, 64, 64)

    # w13_scale_4d: [E, num_blocks, 1, N] -> [4, 4, 1, 128]
    assert layer.w13_weight_scale.shape == (4, 4, 1, 128)
    # w2_scale_4d: [E, num_blocks, 1, N] -> [4, 2, 1, 128]
    assert layer.w2_weight_scale.shape == (4, 2, 1, 128)

    assert layer.w13_weight.dtype == getattr(torch, "float4_e2m1fn_x2",
                                             layer.w13_weight.dtype)
    assert layer.w2_weight.dtype == getattr(torch, "float4_e2m1fn_x2",
                                            layer.w2_weight.dtype)

    # 3. _forward_monolithic_tpu
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    router_logits = torch.randn(2, num_experts, dtype=torch.float32)

    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.fused_moe_gmm"
    ) as mock_gmm:
        mock_gmm.return_value = torch.zeros_like(x)
        out = method.apply_monolithic(layer, x, router_logits)

        assert out.shape == x.shape
        mock_gmm.assert_called_once()
        assert mock_gmm.call_args.kwargs["experts_start"] is None


def test_mxfp4_apply_masks_padded_token_routes():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    method._tpu_activation_str = "silu"
    layer = FakeRoutedExperts(experts_per_token=2)
    layer.w13_weight = torch.empty(0)
    layer.w2_weight = torch.empty(0)
    layer.w13_weight_scale = torch.empty(0)
    layer.w2_weight_scale = torch.empty(0)
    layer._experts_start = None

    x = torch.randn(4, 8, dtype=torch.bfloat16)
    router_logits = torch.randn(4, 4, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]],
                            dtype=torch.int64)
    topk_weights = torch.full((4, 2), 0.5)
    masked_weights = topk_weights.clone()
    masked_weights[1:] = 0

    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.moe_routing.select_experts",
            return_value=(topk_weights, topk_ids)
    ), patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.token_padding.zero_routing_weights_for_padding",
            return_value=(topk_ids.to(torch.int32), masked_weights)
    ) as mock_mask, patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.fused_moe_gmm",
            return_value=torch.zeros_like(x)) as mock_gmm:
        method.apply_monolithic(layer, x, router_logits)

    mock_mask.assert_called_once()
    assert mock_gmm.call_args.kwargs["topk_ids"].dtype == torch.int32
    assert torch.equal(mock_gmm.call_args.kwargs["topk_weights"],
                       masked_weights)
    assert mock_gmm.call_args.kwargs["skip_padded_tokens"] is True


def test_mxfp4_duplicate_active_rows_with_padding_are_identical():
    """The decode-shaped MoE call must not mix otherwise identical rows."""
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.experts_per_token = 16
    moe_config.intermediate_size_per_partition = 128
    moe_config.intermediate_size_per_partition_unpadded = 128
    moe_config.activation_situ_beta = 4.0
    moe_config.activation_situ_linear_beta = 25.0

    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts(experts_per_token=16)
    layer.activation = "situ"
    layer.moe_config.activation_situ_beta = 4.0
    layer.moe_config.activation_situ_linear_beta = 25.0
    layer._expert_routing_tables = lambda: (None, None)
    num_experts = 16
    hidden_size = 128
    method.create_weights(
        layer=layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=128,
        params_dtype=torch.bfloat16,
    )

    generator = torch.Generator().manual_seed(0)
    weight_loader = layer.w13_weight_packed.weight_loader
    w13_scale = torch.full((128, 4), 127, dtype=torch.uint8)
    w2_scale = torch.full((128, 4), 127, dtype=torch.uint8)
    for expert_id in range(num_experts):
        w13_packed = torch.randint(0,
                                   256, (128, 64),
                                   dtype=torch.uint8,
                                   generator=generator)
        w2_packed = torch.randint(0,
                                  256, (128, 64),
                                  dtype=torch.uint8,
                                  generator=generator)
        weight_loader(
            layer.w13_weight_packed,
            w13_packed,
            "w13_weight_packed",
            "w1",
            expert_id,
        )
        weight_loader(
            layer.w13_weight_packed,
            w13_packed,
            "w13_weight_packed",
            "w3",
            expert_id,
        )
        weight_loader(
            layer.w13_weight_scale,
            w13_scale,
            "w13_weight_scale",
            "w1",
            expert_id,
        )
        weight_loader(
            layer.w13_weight_scale,
            w13_scale,
            "w13_weight_scale",
            "w3",
            expert_id,
        )
        weight_loader(
            layer.w2_weight_packed,
            w2_packed,
            "w2_weight_packed",
            "",
            expert_id,
        )
        weight_loader(
            layer.w2_weight_scale,
            w2_scale,
            "w2_weight_scale",
            "",
            expert_id,
        )

    with patch("vllm_torchtpu.layers.vllm.quantization.compressed_tensors."
               "compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4."
               "prebuild_fused_moe_kernel"):
        method.process_weights_after_loading(layer)

    active = torch.randn(1, hidden_size, dtype=torch.bfloat16)
    hidden_states = torch.cat(
        (active.expand(8, -1), torch.zeros(8,
                                           hidden_size,
                                           dtype=torch.bfloat16))).to("tpu")
    active_router = torch.randn(1, num_experts, dtype=torch.float32)
    router_logits = torch.cat(
        (active_router.expand(8, -1), torch.zeros(8, num_experts))).to("tpu")

    output = method.apply_monolithic(layer, hidden_states, router_logits)
    active_output = output[:8].cpu()
    torch.testing.assert_close(
        active_output,
        active_output[0].expand_as(active_output),
        rtol=0,
        atol=0,
    )


def test_mxfp4_create_weights_does_not_zero_initialize():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts()

    with patch("torch.zeros",
               side_effect=AssertionError("must use torch.empty")):
        method.create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=128,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )

    for name in ("w13_weight_packed", "w2_weight_packed", "w13_weight_scale",
                 "w2_weight_scale"):
        assert hasattr(layer, name)


def test_mxfp4_processes_directly_materialized_dummy_weights():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.experts_per_token = 2
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = 64
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts(experts_per_token=2)
    layer._expert_routing_tables = lambda: (None, None)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )

    # vLLM's dummy loader initializes materialized parameters without calling
    # the checkpoint weight-loader hook, so no CPU scratchpads are created.
    for parameter in (layer.w13_weight_packed, layer.w13_weight_scale,
                      layer.w2_weight_packed, layer.w2_weight_scale):
        assert not hasattr(parameter, "_cpu_scratch")
        parameter.data.zero_()

    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.prebuild_fused_moe_kernel"
    ):
        method.process_weights_after_loading(layer)

    assert hasattr(layer, "w13_weight")
    assert hasattr(layer, "w2_weight")
    assert not hasattr(layer, "w13_weight_packed")
    assert not hasattr(layer, "w2_weight_packed")


def test_mxfp4_initializes_integer_dummy_weights():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.experts_per_token = 2
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = 64
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts(experts_per_token=2)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )

    parameters = (layer.w13_weight_packed, layer.w13_weight_scale,
                  layer.w2_weight_packed, layer.w2_weight_scale)
    for parameter in parameters:
        parameter.data.fill_(255)

    vllm_config = MagicMock()
    vllm_config.load_config.load_format = "dummy"
    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.get_current_vllm_config_or_none",
            return_value=vllm_config):
        method._initialize_dummy_quantized_weights(layer)

    for parameter in parameters:
        assert torch.count_nonzero(parameter) == 0


def test_mxfp4_preserves_integer_checkpoint_weights():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.experts_per_token = 2
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = 64
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts(experts_per_token=2)
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )

    parameters = (layer.w13_weight_packed, layer.w13_weight_scale,
                  layer.w2_weight_packed, layer.w2_weight_scale)
    for parameter in parameters:
        parameter.data.fill_(255)

    vllm_config = MagicMock()
    vllm_config.load_config.load_format = "safetensors"
    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.get_current_vllm_config_or_none",
            return_value=vllm_config):
        method._initialize_dummy_quantized_weights(layer)

    for parameter in parameters:
        assert torch.all(parameter == 255)


def test_mxfp4_neutralizes_only_unloaded_padded_scales():
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = 32
    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts()
    method.create_weights(
        layer=layer,
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=64,
        params_dtype=torch.bfloat16,
    )

    for parameter in (layer.w13_weight_scale, layer.w2_weight_scale):
        parameter._cpu_scratch = torch.nn.Parameter(torch.empty_like(
            parameter, device="cpu"),
                                                    requires_grad=False)
        parameter._cpu_scratch.data.fill_(0xff)

    method._neutralize_padded_scales(layer)

    w13_scale = layer.w13_weight_scale._cpu_scratch
    w2_scale = layer.w2_weight_scale._cpu_scratch
    assert torch.all(w13_scale[:, :32, :] == 0xff)
    assert not w13_scale[:, 32:64, :].any()
    assert torch.all(w13_scale[:, 64:96, :] == 0xff)
    assert not w13_scale[:, 96:, :].any()
    assert torch.all(w2_scale[:, :, :1] == 0xff)
    assert not w2_scale[:, :, 1:].any()


def test_mxfp4_requantize_block():
    from vllm_torchtpu.layers.common.quantization import (
        dequantize_mxfp4_packed, fp4_indices_to_float, quantize_tensor_to_fp4)
    moe_config = MagicMock()
    moe_config.tp_size = 1
    moe_config.tp_rank = 0
    moe_config.experts_per_token = 2
    moe_config.intermediate_size_per_partition = 64
    moe_config.intermediate_size_per_partition_unpadded = 64

    method = VllmCompressedTensorsW4ANMxfp4MoEMethod(moe_config)
    layer = FakeRoutedExperts(experts_per_token=2)
    layer._expert_routing_tables = lambda: (None, None)

    num_experts = 4
    hidden_size = 128
    intermediate_size_per_partition = 64

    method.create_weights(
        layer=layer,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_partition=intermediate_size_per_partition,
        params_dtype=torch.bfloat16,
    )

    weight_loader = layer.w13_weight_packed.weight_loader
    w13_packed_val = torch.randint(0, 256, (64, 64), dtype=torch.uint8)
    # Range 100-150 prevents inf values when converting e8m0 -> fp32
    w13_scale_val = torch.randint(100, 150, (64, 4), dtype=torch.uint8)
    w2_packed_val = torch.randint(0, 256, (128, 32), dtype=torch.uint8)
    w2_scale_val = torch.randint(100, 150, (128, 2), dtype=torch.uint8)

    for expert_id in range(num_experts):
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w1", expert_id)
        weight_loader(layer.w13_weight_packed, w13_packed_val,
                      "w13_weight_packed", "w3", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w1", expert_id)
        weight_loader(layer.w13_weight_scale, w13_scale_val,
                      "w13_weight_scale", "w3", expert_id)
        weight_loader(layer.w2_weight_packed, w2_packed_val,
                      "w2_weight_packed", "", expert_id)
        weight_loader(layer.w2_weight_scale, w2_scale_val, "w2_weight_scale",
                      "", expert_id)

    w13_packed_orig = layer.w13_weight_packed._cpu_scratch.data.clone()
    w13_scale_orig = layer.w13_weight_scale._cpu_scratch.data.clone()

    with patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.envs.MOE_REQUANTIZE_BLOCK_SIZE",
            "64"
    ), patch(
            "vllm_torchtpu.layers.vllm.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4.prebuild_fused_moe_kernel"
    ):
        method.process_weights_after_loading(layer)

    assert layer.w13_weight.shape == (4, 128, 64)
    assert layer.w2_weight.shape == (4, 64, 64)
    assert layer.w13_weight_scale.shape == (4, 2, 1, 128)
    assert layer.w2_weight_scale.shape == (4, 1, 1, 128)

    w13_f32 = dequantize_mxfp4_packed(w13_packed_orig, w13_scale_orig)
    w13_fp4_idx, w13_fp32_scale = quantize_tensor_to_fp4(w13_f32,
                                                         axis=-1,
                                                         block_size=64)
    w13_fp4_f32_expected = fp4_indices_to_float(w13_fp4_idx)

    # 3. K-major layout for GMM: [E, K, N]
    w13_fp4_f32_expected_kmajor = w13_fp4_f32_expected.transpose(-1, -2)

    # Check that the native K-major fp4 tensor matches
    from vllm_torchtpu.layers.common.quantization import unpack_uint8_to_fp4
    actual_w13_f32 = unpack_uint8_to_fp4(layer.w13_weight.data.cpu().view(
        torch.uint8))

    # XLA and PyTorch round quantization ties slightly differently (1% of values shift by 1 bin).
    # Since the spacing in FP4 e2m1 is at most 1.0 around the low values, we permit a max diff of 1.0.
    max_diff = (actual_w13_f32 - w13_fp4_f32_expected_kmajor).abs().max()
    assert max_diff <= 1.0, f"Max diff on w13 ({max_diff}) exceeded 1.0"

    # Check the scale (gmm_v2 scale layout: [E, num_blocks, 1, N])
    w13_fp32_scale_4d = w13_fp32_scale.movedim(-1, 1).unsqueeze(-2)
    actual_scale = layer.w13_weight_scale.data.cpu()
    torch.testing.assert_close(actual_scale,
                               w13_fp32_scale_4d,
                               rtol=1e-3,
                               atol=1e-3)
