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
                                          use_ep=False)

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
