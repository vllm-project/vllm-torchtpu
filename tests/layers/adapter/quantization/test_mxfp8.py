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
"""Tests for ModelOpt MXFP8 linear support (`quantization/mxfp8.py`).

The reference for every numeric test is the checkpoint's own meaning: each FP8
value times 2^(e - 127) for the E8M0 byte `e` of its 32-value block.
"""

import pytest
import torch
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase

from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.layers.adapter.quantization.mxfp8 import (
    VllmModelOptMxfp8Config,
    VllmModelOptMxfp8LinearMethod,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)
from vllm_torchtpu.layers.core.quantization import e8m0_to_fp32

BLOCK = 32
EXCLUDED = "model.layers.0.mlp.gate"


def _config(exclude_modules=(EXCLUDED,)):
    # The nested ModelOpt schema of hf_quant_config.json / quantization_config.
    return VllmModelOptMxfp8Config.from_config(
        {
            "quant_method": "modelopt",
            "quantization": {
                "quant_algo": "MXFP8",
                "kv_cache_quant_algo": None,
                "exclude_modules": list(exclude_modules),
            },
        }
    )


def _bare(cls):
    layer = cls.__new__(cls)
    torch.nn.Module.__init__(layer)
    return layer


@pytest.fixture
def linear_layer():
    layer = _bare(LinearBase)
    layer.output_size = 64
    layer.input_size = 64
    return layer


def _mock_tp(monkeypatch, tp_size: int, tp_rank: int):
    for target in (
        "vllm.distributed",
        "vllm.distributed.parallel_state",
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.parameter",
    ):
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_world_size",
            lambda: tp_size,
            raising=False,
        )
        monkeypatch.setattr(
            f"{target}.get_tensor_model_parallel_rank",
            lambda: tp_rank,
            raising=False,
        )


def _mxfp8_checkpoint(out_size: int, in_size: int, seed: int = 0):
    """Random FP8 weight, random E8M0 scales, and their float32 meaning."""
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(out_size, in_size, generator=g) * 4).to(torch.float8_e4m3fn)
    # Exponents 2^-6 .. 2^5, well inside FP8 range after requantization.
    s = torch.randint(121, 133, (out_size, in_size // BLOCK), generator=g).to(
        torch.uint8
    )
    ref = w.float() * e8m0_to_fp32(s).repeat_interleave(BLOCK, dim=1)
    return w, s, ref


def _runtime_weight(layer) -> torch.Tensor:
    """The runtime (k, n) weight times its scale, in checkpoint [n_out, n_in]."""
    assert getattr(layer, WEIGHT_FLIPPED_ATTR, False)
    w, scale = layer.weight.float(), layer.weight_scale.float()
    if scale.ndim == 1:  # per channel: [n_out]
        return (w * scale[None, :]).T
    # blockwise: [1, k / block, 1, n_out]
    assert scale.ndim == 4 and scale.shape[0] == 1 and scale.shape[2] == 1, scale.shape
    block = w.shape[0] // scale.shape[1]
    return (w * scale[0, :, 0, :].repeat_interleave(block, dim=0)).T


def _set_runtime_block(monkeypatch, block: int | None):
    """None: the FP8 runtime default (per channel); else blockwise with `block`."""
    from vllm_torchtpu import envs

    monkeypatch.setattr(envs, "REQUANTIZE_BLOCK_SIZE", block)
    monkeypatch.setattr(envs, "ENABLE_QUANTIZED_MATMUL_KERNEL", block is not None)


def _assert_close_to_fp8_rounding(got: torch.Tensor, ref: torch.Tensor):
    # FP8 requantization: E4M3 keeps 3 mantissa bits, so each value is within
    # 2^-4 of its magnitude, plus the step near zero of the coarsest scale group
    # (bounded here by the row).
    row_max = ref.abs().amax(dim=1, keepdim=True)
    tol = ref.abs() * 2**-4 + row_max * 2**-9
    assert ((got - ref).abs() <= tol).all(), (got - ref).abs().max()


class TestMxfp8Config:
    def test_parses_modelopt_checkpoint(self):
        cfg = _config()
        assert cfg.get_name() == "modelopt_mxfp8"
        assert cfg.is_checkpoint_mxfp8_serialized
        assert cfg.exclude_modules == [EXCLUDED]

    def test_presents_as_fp8_block_checkpoint(self):
        # The attributes VllmFp8LinearMethodTPU reads from its quant_config.
        cfg = _config()
        assert cfg.is_checkpoint_fp8_serialized
        assert cfg.weight_block_size == [1, BLOCK]
        assert cfg.activation_scheme == "dynamic"

    def test_registered_for_tpu(self):
        from vllm.model_executor.layers.quantization import get_quantization_config

        from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

        assert get_quantization_config("tpu-modelopt_mxfp8") is VllmModelOptMxfp8Config
        assert "modelopt_mxfp8" in TpuPlatform.supported_quantization

    def test_linear_dispatch(self, linear_layer):
        method = _config().get_quant_method(linear_layer, prefix="model.layers.0.o")
        assert type(method) is VllmModelOptMxfp8LinearMethod
        assert method.block_quant
        assert method.weight_block_size == [1, BLOCK]

    def test_excluded_linear_uses_kn_unquantized(self, linear_layer):
        cfg = _config()
        assert cfg.is_layer_excluded(EXCLUDED), "fixture must hit the excluded path"
        method = cfg.get_quant_method(linear_layer, prefix=EXCLUDED)
        assert isinstance(method, VllmUnquantizedLinearMethod)

    def test_attention_not_quantized(self):
        assert _config().get_quant_method(_bare(Attention), prefix="attn") is None

    def test_quantized_moe_not_supported(self, monkeypatch):
        cfg = _config()
        monkeypatch.setattr(cfg, "get_moe_config", lambda layer: None)
        with pytest.raises(NotImplementedError, match="MXFP8 MoE"):
            cfg.get_quant_method(_bare(RoutedExperts), prefix="model.layers.1.mlp")

    def test_excluded_moe_unquantized(self, monkeypatch):
        from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

        cfg = _config(exclude_modules=["model.layers.1.mlp*"])
        monkeypatch.setattr(cfg, "get_moe_config", lambda layer: None)
        # The unquantized MoE method is a CustomOp, built under a vLLM config.
        with set_current_vllm_config(VllmConfig(device_config=DeviceConfig("cpu"))):
            method = cfg.get_quant_method(
                _bare(RoutedExperts), prefix="model.layers.1.mlp.experts"
            )
        assert isinstance(method, VllmUnquantizedFusedMoEMethod)


class TestMxfp8Linear:
    @pytest.mark.parametrize("runtime_block", [None, BLOCK])
    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_column_parallel_load_and_convert(
        self, monkeypatch, tp_size, runtime_block
    ):
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        _set_runtime_block(monkeypatch, runtime_block)
        out_size, in_size = 128, 256
        w, s, ref = _mxfp8_checkpoint(out_size, in_size)
        shard = out_size // tp_size
        for tp_rank in range(tp_size):
            _mock_tp(monkeypatch, tp_size, tp_rank)
            layer = ColumnParallelLinear(
                input_size=in_size,
                output_size=out_size,
                bias=False,
                quant_config=_config(),
                params_dtype=torch.bfloat16,
                gather_output=False,
            )
            assert layer.weight.dtype == torch.float8_e4m3fn
            assert layer.weight_scale.dtype == torch.uint8
            assert layer.weight_scale.shape == (shard, in_size // BLOCK)

            layer.weight.weight_loader(layer.weight, w)
            layer.weight_scale.weight_loader(layer.weight_scale, s)
            layer.quant_method.process_weights_after_loading(layer)

            assert not hasattr(layer, "weight_scale_inv")
            assert layer.weight_scale.ndim == (1 if runtime_block is None else 4)
            rows = slice(tp_rank * shard, (tp_rank + 1) * shard)
            _assert_close_to_fp8_rounding(_runtime_weight(layer), ref[rows])

    @pytest.mark.parametrize("runtime_block", [None, BLOCK])
    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_merged_column_parallel_load_and_convert(
        self, monkeypatch, tp_size, runtime_block
    ):
        """Two shards of different sizes (gate / up) loaded one at a time."""
        from vllm.model_executor.layers.linear import MergedColumnParallelLinear

        _set_runtime_block(monkeypatch, runtime_block)
        sizes, in_size = [64, 128], 256
        ckpts = [_mxfp8_checkpoint(n, in_size, seed=2 + i) for i, n in enumerate(sizes)]
        for tp_rank in range(tp_size):
            _mock_tp(monkeypatch, tp_size, tp_rank)
            layer = MergedColumnParallelLinear(
                input_size=in_size,
                output_sizes=sizes,
                bias=False,
                quant_config=_config(),
                params_dtype=torch.bfloat16,
                gather_output=False,
            )
            assert layer.weight_scale.shape == (sum(sizes) // tp_size, in_size // BLOCK)

            for shard_id, (w, s, _) in enumerate(ckpts):
                layer.weight.weight_loader(layer.weight, w, shard_id)
                layer.weight_scale.weight_loader(layer.weight_scale, s, shard_id)
            layer.quant_method.process_weights_after_loading(layer)

            ref = torch.cat(
                [
                    r[tp_rank * (n // tp_size) : (tp_rank + 1) * (n // tp_size)]
                    for n, (_, _, r) in zip(sizes, ckpts)
                ]
            )
            _assert_close_to_fp8_rounding(_runtime_weight(layer), ref)

    @pytest.mark.parametrize("runtime_block", [None, BLOCK])
    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_row_parallel_load_and_convert(self, monkeypatch, tp_size, runtime_block):
        """The input dim is split, so each shard keeps whole 32-value blocks."""
        from vllm.model_executor.layers.linear import RowParallelLinear

        _set_runtime_block(monkeypatch, runtime_block)
        out_size, in_size = 64, 256
        w, s, ref = _mxfp8_checkpoint(out_size, in_size, seed=1)
        shard = in_size // tp_size
        for tp_rank in range(tp_size):
            _mock_tp(monkeypatch, tp_size, tp_rank)
            layer = RowParallelLinear(
                input_size=in_size,
                output_size=out_size,
                bias=False,
                quant_config=_config(),
                params_dtype=torch.bfloat16,
                reduce_results=False,
            )
            assert layer.weight_scale.shape == (out_size, shard // BLOCK)

            layer.weight.weight_loader(layer.weight, w)
            layer.weight_scale.weight_loader(layer.weight_scale, s)
            layer.quant_method.process_weights_after_loading(layer)

            cols = slice(tp_rank * shard, (tp_rank + 1) * shard)
            _assert_close_to_fp8_rounding(_runtime_weight(layer), ref[:, cols])

    def test_reload_reentry_is_a_no_op(self, monkeypatch):
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        _mock_tp(monkeypatch, 1, 0)
        w, s, _ = _mxfp8_checkpoint(64, 128, seed=2)
        layer = ColumnParallelLinear(
            input_size=128,
            output_size=64,
            bias=False,
            quant_config=_config(),
            params_dtype=torch.bfloat16,
        )
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, s)
        layer.quant_method.process_weights_after_loading(layer)
        weight = layer.weight.data.clone()
        scale = layer.weight_scale.data.clone()

        layer.quant_method.process_weights_after_loading(layer)

        assert torch.equal(
            layer.weight.data.view(torch.uint8), weight.view(torch.uint8)
        )
        assert torch.equal(layer.weight_scale.data, scale)

    def test_rejects_non_e8m0_scale(self, monkeypatch):
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        _mock_tp(monkeypatch, 1, 0)
        layer = ColumnParallelLinear(
            input_size=128,
            output_size=64,
            bias=False,
            quant_config=_config(),
            params_dtype=torch.bfloat16,
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.ones(64, 4, dtype=torch.float32), requires_grad=False
        )
        with pytest.raises(ValueError, match="uint8 E8M0"):
            layer.quant_method.process_weights_after_loading(layer)

    @pytest.mark.tpu_test
    def test_apply_matches_reference_matmul(self, monkeypatch, device):
        if device.type != "tpu":
            pytest.skip("the FP8 quantized matmul runs on TPU")
        from vllm.model_executor.layers.linear import ColumnParallelLinear

        _mock_tp(monkeypatch, 1, 0)
        out_size, in_size, batch = 128, 256, 8
        w, s, ref = _mxfp8_checkpoint(out_size, in_size, seed=3)
        layer = ColumnParallelLinear(
            input_size=in_size,
            output_size=out_size,
            bias=False,
            quant_config=_config(),
            params_dtype=torch.bfloat16,
        )
        layer.weight.weight_loader(layer.weight, w)
        layer.weight_scale.weight_loader(layer.weight_scale, s)
        layer.to(device)
        layer.quant_method.process_weights_after_loading(layer)

        x = torch.randn(batch, in_size, generator=torch.Generator().manual_seed(4))
        got = layer.quant_method.apply(layer, x.to(device, torch.bfloat16))
        want = x @ ref.T
        # FP8 weights plus dynamically FP8-quantized activations.
        torch.testing.assert_close(
            got.float().cpu(), want, rtol=0.1, atol=0.1 * want.abs().max().item()
        )
