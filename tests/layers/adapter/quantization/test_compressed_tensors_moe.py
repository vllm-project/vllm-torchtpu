# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager

from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    VllmCompressedTensorsConfig,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (  # noqa: E501
    VllmCompressedTensorsMoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a16 import (  # noqa: E501
    VllmCompressedTensorsW4A16MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4an_mxfp4 import (  # noqa: E501
    VllmCompressedTensorsW4ANMxfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8MoEMethodTPU,
)


class FakeQuantArgs:
    """Mock for QuantizationArgs from compressed-tensors."""

    def __init__(
        self,
        num_bits=4,
        strategy="group",
        group_size=16,
        symmetric=True,
        dynamic=False,
        quant_type="int",
        actorder="none",
        format="int4",
    ):
        self.num_bits = num_bits
        self.strategy = strategy
        self.group_size = group_size
        self.symmetric = symmetric
        self.dynamic = dynamic
        self.type = quant_type
        self.actorder = actorder
        self.format = format

    def __getitem__(self, key):
        if key == "format":
            return self.format
        raise KeyError(key)

    def get(self, key, default=None):
        if key == "format":
            return self.format
        return default


class FakeActivation:
    """Mimics MoEActivation enum."""

    def __init__(self, value):
        self.value = value


# Need to mock vllm.model_executor.layers.fused_moe.RoutedExperts isinstance check
# Because VllmCompressedTensorsMoEMethod.get_moe_method asserts
# isinstance(layer, RoutedExperts)
class FakeRoutedExperts(RoutedExperts):
    """Subclass RoutedExperts to pass isinstance check without triggering full vLLM
    config requirements.
    """

    def __init__(self, experts_per_token=2):
        torch.nn.Module.__init__(self)
        self.moe_config = MagicMock()
        self.moe_config.experts_per_token = experts_per_token
        self.moe_config.moe_parallel_config = MagicMock()
        self.moe_config.moe_parallel_config.use_ep = False
        self.moe_config.activation = FakeActivation("silu")
        self.use_grouped_topk = False


class TestCompressedTensorsConfigRouting:
    """Verify that VllmCompressedTensorsConfig correctly routes configs."""

    def test_get_moe_method_routing(self, monkeypatch):
        # We need to mock _is_int4_w4aN, _is_weight_fp8 to make them independent of full
        # configs, or use correct FakeQuantArgs
        config = MagicMock(spec=VllmCompressedTensorsConfig)

        # Monkeypatch _is_mxfp4
        config._is_mxfp4 = MagicMock(return_value=False)

        # 1. Test W4A16 routing
        weight_quant = FakeQuantArgs(
            num_bits=4, strategy="group", group_size=16, format="pack"
        )  # using int4
        input_quant = None
        scheme_dict = {"weights": weight_quant, "input_activations": input_quant}
        config.get_scheme_dict.return_value = scheme_dict

        # Monkey patch routing methods
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors._is_int4_w4aN",
            lambda w: w.num_bits == 4 and w.format != "mxfp4",
        )
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors._is_weight_fp8",
            lambda w: w.format == "float8",
        )
        monkeypatch.setattr(
            "vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors._build_fp8_config",
            MagicMock(),
        )
        monkeypatch.setattr(
            "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16.select_wna16_moe_backend",
            MagicMock(return_value=(None, None)),
        )

        layer = FakeRoutedExperts(experts_per_token=2)

        method = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block"
        )
        assert isinstance(method, VllmCompressedTensorsW4A16MoEMethod)

        # 2. Test W4A8 routing (should route to the same runner)
        input_quant_8 = FakeQuantArgs(
            num_bits=8, strategy="token", dynamic=True, format="int8"
        )
        scheme_dict_w4a8 = {"weights": weight_quant, "input_activations": input_quant_8}
        config.get_scheme_dict.return_value = scheme_dict_w4a8

        method_w4a8 = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block"
        )
        assert isinstance(method_w4a8, VllmCompressedTensorsW4A16MoEMethod)

        # 3. Test MXFP4 routing
        weight_quant_mxfp4 = FakeQuantArgs(num_bits=4, format="mxfp4")
        scheme_dict_mxfp4 = {"weights": weight_quant_mxfp4, "input_activations": None}
        config.get_scheme_dict.return_value = scheme_dict_mxfp4
        config._is_mxfp4 = MagicMock(return_value=True)

        method_mxfp4 = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block"
        )
        assert isinstance(method_mxfp4, VllmCompressedTensorsW4ANMxfp4MoEMethod)

        # 4. Test FP8 routing
        config._is_mxfp4 = MagicMock(return_value=False)
        weight_quant_fp8 = FakeQuantArgs(num_bits=8, format="float8")
        scheme_dict_fp8 = {"weights": weight_quant_fp8, "input_activations": None}
        config.get_scheme_dict.return_value = scheme_dict_fp8

        method_fp8 = VllmCompressedTensorsMoEMethod.get_moe_method(
            config, layer, "model.layers.0.block"
        )
        assert isinstance(method_fp8, VllmFp8MoEMethodTPU)

        # 5. Test unsupported strategy routing fallback
        weight_quant_unsupported = FakeQuantArgs(num_bits=16, format="unknown")
        scheme_dict_unsupported = {
            "weights": weight_quant_unsupported,
            "input_activations": None,
        }
        config.get_scheme_dict.return_value = scheme_dict_unsupported

        with pytest.raises(RuntimeError, match="Unsupported TPU FusedMoe scheme"):
            VllmCompressedTensorsMoEMethod.get_moe_method(
                config, layer, "model.layers.0.block"
            )


def _compressed_tensors_fp8_config(weights: dict) -> VllmCompressedTensorsConfig:
    """llm-compressor FP8 `quantization_config`, reduced to what MoE dispatch reads."""
    return VllmCompressedTensorsConfig.from_config(
        {
            "config_groups": {
                "group_0": {
                    "weights": {"type": "float", **weights},
                    "targets": ["Linear"],
                }
            },
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
        }
    )


@pytest.mark.parametrize(
    "weights,w13_scale_shape,w2_scale_shape",
    [
        # One scale per 128x128 block of the 2 experts' [512, 256] w13 / [256, 256] w2.
        pytest.param(
            {"strategy": "block", "block_structure": [128, 128]},
            (2, 4, 2),
            (2, 2, 2),
            id="block",
        ),
        # Per-channel already used `weight_scale`; it must keep doing so.
        pytest.param({"strategy": "channel"}, (2, 512, 1), (2, 256, 1), id="channel"),
    ],
)
def test_fp8_experts_register_compressed_tensors_scale_names(
    monkeypatch, weights, w13_scale_shape, w2_scale_shape
):
    """compressed-tensors names every scale `weight_scale`; expert loaders look
    parameters up by that name, so the fused layer must not register
    VllmFp8MoEMethodTPU's default DeepSeek-style `weight_scale_inv` instead."""
    vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_expert_parallel=False)
    )
    monkeypatch.setattr(VllmCompressedTensorsConfig, "vllm_config", vllm_config)
    monkeypatch.setattr(VllmFp8Config, "vllm_config", vllm_config)
    # Block-scale geometry asks the TP group for its size; no group exists here.
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_tensor_model_parallel_world_size", lambda: 1
    )
    parallel = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    with set_current_vllm_config(vllm_config):
        layer = RoutedExperts(
            layer_name="model.layers.0.mlp.experts",
            params_dtype=torch.bfloat16,
            moe_config=FusedMoEConfig(
                num_experts=2,
                experts_per_token=1,
                hidden_dim=256,
                intermediate_size=256,
                num_local_experts=2,
                num_logical_experts=2,
                activation=MoEActivation.SILU,
                device=torch.device("cpu"),
                routing_method=RoutingMethodType.Renormalize,
                moe_parallel_config=parallel,
                in_dtype=torch.bfloat16,
            ),
            quant_config=_compressed_tensors_fp8_config(weights),
            expert_map_manager=ExpertMapManager(
                max_num_batched_tokens=16,
                top_k=1,
                global_num_experts=2,
                num_redundant_experts=0,
                num_expert_group=None,
                moe_parallel_config=parallel,
                placement_strategy="linear",
                enable_eplb=False,
            ),
        )

    assert isinstance(layer.quant_method, VllmFp8MoEMethodTPU)
    params = dict(layer.named_parameters())
    assert not [name for name in params if name.endswith("weight_scale_inv")]
    assert params["w13_weight_scale"].shape == w13_scale_shape
    assert params["w2_weight_scale"].shape == w2_scale_shape
    # Expert loaders pass the checkpoint's own tensor name to the loader.
    w2_scale = params["w2_weight_scale"]
    scales = torch.arange(w2_scale[0].numel(), dtype=torch.float32).reshape(
        w2_scale.shape[1:]
    )
    layer.weight_loader(w2_scale, scales, "w2_weight_scale", "w2", 0)
    torch.testing.assert_close(w2_scale[0], scales)
