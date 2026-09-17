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
"""CPU tests for actual BF16 loading and TPU online FP8 dispatch."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.model_loader.utils import \
    process_weights_after_loading
from vllm.model_executor.models.utils import WeightsMapper

from vllm_torchtpu.layers.adapter.linear_common import KEEP_VLLM_LAYOUT_ATTR
from vllm_torchtpu.layers.adapter.quantization.fp8 import VllmFp8Config
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import VllmNvfp4Config
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import (
    OnlineFp8Policy, attach_online_fp8, validate_online_fp8)


@pytest.fixture(autouse=True)
def tensor_parallel_group(monkeypatch):
    from vllm.distributed import parallel_state
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "_TP", group)
    return group


def make_config(kind="fp8", targets=None, exclude=None, **options):
    ignored = ["proj", "untouched", "merged", "gate", "row"]
    if kind == "fp8":
        config = VllmFp8Config.from_config({
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ignored,
        })
    else:
        config = VllmNvfp4Config.from_config({
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "with_input_scale": True,
            "ignore": ignored,
        })
    attach_online_fp8(
        config,
        SimpleNamespace(
            additional_config={
                "tpu_online_fp8": {
                    "enabled": True,
                    "targets": targets if targets is not None else ["proj"],
                    "exclude": exclude or [],
                    **options
                }
            }))
    return config


def linear(config, name="proj", dtype=torch.bfloat16):
    return ReplicatedLinear(32,
                            16,
                            bias=False,
                            params_dtype=dtype,
                            quant_config=config,
                            prefix=name,
                            disable_tp=True)


@pytest.mark.parametrize("value", [None, {}, {"enabled": False}])
def test_disabled(value):
    assert OnlineFp8Policy.from_dict(value) is None


@pytest.mark.parametrize("value", [
    [],
    {
        "enabled": "true"
    },
    {
        "enabled": True
    },
    {
        "enabled": True,
        "targets": "proj"
    },
    {
        "enabled": True,
        "targets": [""]
    },
    {
        "enabled": True,
        "targets": ["re:.*"]
    },
    {
        "enabled": True,
        "targets": ["proj"],
        "weight_scheme": "per_block"
    },
    {
        "enabled": True,
        "targets": ["proj"],
        "typo": True
    },
])
def test_invalid_config(value):
    with pytest.raises(ValueError):
        OnlineFp8Policy.from_dict(value)


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_dispatch_and_independent_serialized_config(kind):
    config = make_config(kind)
    before = dict(config.__dict__)
    selected = linear(config)
    untouched = linear(config, "untouched")
    assert selected.quant_method.online_fp8
    assert not selected.quant_method.quant_config.is_checkpoint_fp8_serialized
    assert selected.weight.dtype == torch.bfloat16
    assert not hasattr(selected, "weight_scale")
    assert isinstance(untouched.quant_method, UnquantizedLinearMethod)
    assert config.__dict__ == before
    other = make_config(kind)
    assert other.online_fp8_policy.selected == {}


def test_prequantized_conflict():
    config = make_config(targets=["quantized"])
    # Its base method must be created before the online conflict is checked.
    config.get_linear_config = lambda layer: None
    with pytest.raises(ValueError, match="already has checkpoint"):
        linear(config, "quantized")


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_checkpoint_method_preserved(kind, monkeypatch):
    config = make_config(kind)
    original = object()
    monkeypatch.setattr(config, "_get_checkpoint_quant_method",
                        lambda layer, prefix: original)
    assert config.get_quant_method(torch.nn.Module(),
                                   "mlp.experts") is original


def test_fused_aliases_and_exclude():
    mapping = {"qkvz": ["qkv", "z"]}
    policy = OnlineFp8Policy(["model.qkv", "model.z"], [])
    assert policy.select("model.qkvz", mapping)[0]
    policy = OnlineFp8Policy(["model.qkvz"], ["model.z"])
    with pytest.raises(ValueError, match="Partial fused"):
        policy.select("model.qkvz", mapping)
    policy = OnlineFp8Policy(["model.qkv"], [])
    with pytest.raises(ValueError, match="Partial fused"):
        policy.select("model.qkvz", mapping)
    policy = OnlineFp8Policy(["model.*"], ["model.qkvz"])
    assert not policy.select("model.qkvz", mapping)[0]


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_checkpoint_name_mapper(kind):
    config = make_config(kind, targets=["model.language_model.proj"])
    config.apply_vllm_mapper(
        WeightsMapper(orig_to_new_prefix={
            "model.language_model.": "language_model.model."
        }))
    assert config.online_fp8_policy.targets == ["language_model.model.proj"]


@pytest.mark.parametrize("keep_layout", [False, True])
@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_real_bf16_checkpoint_loading(tmp_path, kind, keep_layout):
    config = make_config(kind)
    layer = linear(config)
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, keep_layout)
    model = torch.nn.Module()
    model.add_module("proj", layer)
    source = torch.linspace(-3, 3, 16 * 32).reshape(16, 32).to(torch.bfloat16)
    source[0] = 0
    source[1] *= 0.01
    save_file({"proj.weight": source}, tmp_path / "model.safetensors")
    for name, tensor in load_file(tmp_path / "model.safetensors").items():
        param = dict(model.named_parameters())[name]
        param.weight_loader(param, tensor)
    model_config = SimpleNamespace(dtype=torch.bfloat16,
                                   quantization=None,
                                   word_embeddings_untied_by_checkpoint=False)
    process_weights_after_loading(model, model_config, torch.device("cpu"))
    validate_online_fp8(model, config)
    weight = layer.weight.float() if keep_layout else layer.weight.float().t()
    scales = layer.weight_scale
    expected_scales = source.float().abs().amax(dim=1) / 448
    torch.testing.assert_close(scales, expected_scales)
    dequantized = weight * scales[:, None]
    assert torch.isfinite(dequantized).all()
    relative_error = (dequantized -
                      source.float()).norm() / source.float().norm()
    assert relative_error < 0.04
    old_weight = layer.weight
    old_scale = layer.weight_scale
    process_weights_after_loading(model, model_config, torch.device("cpu"))
    assert layer.weight is old_weight
    assert layer.weight_scale is old_scale


def test_bf16_required():
    layer = linear(make_config(), dtype=torch.float32)
    with pytest.raises(ValueError, match="requires BF16"):
        layer.quant_method.process_weights_after_loading(layer)


def test_online_ignores_checkpoint_requant_environment(monkeypatch):
    monkeypatch.setenv("REQUANTIZE_BLOCK_SIZE", "128")
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "0")
    layer = linear(make_config())
    layer.weight.data.fill_(1)
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight_scale.shape == (16, )


def test_fused_weight_loader_scale_order():
    config = make_config(targets=["merged"])
    layer = MergedColumnParallelLinear(32, [16, 32],
                                       bias=False,
                                       params_dtype=torch.bfloat16,
                                       quant_config=config,
                                       prefix="merged",
                                       disable_tp=True)
    layer.weight.weight_loader(layer.weight,
                               torch.ones(16, 32, dtype=torch.bfloat16), 0)
    layer.weight.weight_loader(layer.weight,
                               torch.full((32, 32), 4., dtype=torch.bfloat16),
                               1)
    layer.quant_method.process_weights_after_loading(layer)
    torch.testing.assert_close(layer.weight_scale[:16],
                               torch.full((16, ), 1 / 448))
    torch.testing.assert_close(layer.weight_scale[16:],
                               torch.full((32, ), 4 / 448))


def test_audit_detects_bypassed_dispatch():
    config = make_config(targets=["gate"])
    model = torch.nn.Module()
    model.add_module("gate", linear(None, "gate"))
    with pytest.raises(ValueError, match="bypassed"):
        validate_online_fp8(model, config)


def test_audit_unmatched_and_unprocessed():
    config = make_config(targets=["proj", "typo"])
    model = torch.nn.Module()
    model.add_module("proj", linear(config))
    with pytest.raises(ValueError, match="not processed"):
        validate_online_fp8(model, config)
    model.proj.weight.data.fill_(1)
    model.proj.quant_method.process_weights_after_loading(model.proj)
    with pytest.raises(ValueError, match="typo"):
        validate_online_fp8(model, config)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_row_parallel_shard_loading(tp_rank, tensor_parallel_group):
    tensor_parallel_group.rank_in_group = tp_rank
    tensor_parallel_group.world_size = 2
    config = make_config(targets=["row"])
    layer = RowParallelLinear(32,
                              16,
                              bias=False,
                              params_dtype=torch.bfloat16,
                              quant_config=config,
                              prefix="row")
    source = torch.arange(512).reshape(16, 32).to(torch.bfloat16)
    layer.weight.weight_loader(layer.weight, source)
    layer.quant_method.process_weights_after_loading(layer)
    expected = source[:, tp_rank * 16:(tp_rank + 1) *
                      16].float().abs().amax(1) / 448
    torch.testing.assert_close(layer.weight_scale, expected)


@pytest.mark.parametrize("scheme", ["per_channel", "per_tensor"])
@pytest.mark.parametrize("keep_layout", [False, True])
def test_existing_jax_matmul_accepts_online_weights(keep_layout, monkeypatch,
                                                    scheme):
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.layers.adapter import linear_common
    from vllm_torchtpu.layers.adapter.quantization import fp8

    # Only replace the TorchTPU bridge; execute the existing JAX math on CPU.
    def cpu_bridge(x, w, scale):
        result = linear_common._quantized_matmul_jax(
            jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16),
            jnp.asarray(w.float().numpy(), dtype=jnp.float8_e4m3fn),
            jnp.asarray(scale.numpy()))
        return torch.from_numpy(np.asarray(result.astype(
            jnp.float32)).copy()).to(x.dtype)

    monkeypatch.setattr(fp8, "quantized_matmul", cpu_bridge)
    layer = linear(make_config(weight_scheme=scheme))
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, keep_layout)
    generator = torch.Generator().manual_seed(19)
    weight = torch.randn(16, 32, generator=generator).to(torch.bfloat16)
    x = torch.randn(8, 32, generator=generator).to(torch.bfloat16)
    layer.weight.weight_loader(layer.weight, weight)
    layer.quant_method.process_weights_after_loading(layer)
    bias = torch.linspace(-0.5, 0.5, 16).to(torch.bfloat16)
    result = layer.quant_method.apply(layer, x, bias)
    reference = x.float() @ weight.float().t() + bias.float()
    assert (result.float() - reference).norm() / reference.norm() < 0.06


def test_pcp_parameter_contract():
    from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import \
        VllmGatedDeltaNetAttention
    layer = linear(make_config())
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, True)
    layer.weight.data.fill_(0.5)
    layer.quant_method.process_weights_after_loading(layer)
    attention = SimpleNamespace(in_proj_qkvz=layer,
                                gqa_interleaved_layout=False)
    weight, scale = VllmGatedDeltaNetAttention._require_pcp_projection_parameters(
        attention)
    assert weight.shape == (16, 32)
    assert scale.shape == (16, )


@pytest.mark.parametrize("options", [
    {
        "weight_scheme": "unknown"
    },
    {
        "weight_scheme": "per_tensor",
        "weight_block_size": [128, 128]
    },
    {
        "weight_scheme": "per_block",
        "weight_block_size": [True, 128]
    },
    {
        "weight_scheme": "per_block",
        "weight_block_size": [0, 128]
    },
    {
        "weight_scheme": "per_block",
        "weight_block_size": [128, 16]
    },
    {
        "weight_scheme": "per_block",
        "weight_block_size": [128]
    },
])
def test_granularity_config_rejected(options):
    with pytest.raises(ValueError):
        make_config(**options)


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
@pytest.mark.parametrize("keep_layout", [False, True])
@pytest.mark.parametrize("block", [None, [1, 128], [128, 128], [128, 512]])
def test_granular_checkpoint_loading(tmp_path, kind, keep_layout, block):
    options = {"weight_scheme": "per_tensor" if block is None else "per_block"}
    if block is not None:
        options["weight_block_size"] = block
    config = make_config(kind, **options)
    layer = ReplicatedLinear(512,
                             256,
                             bias=False,
                             params_dtype=torch.bfloat16,
                             quant_config=config,
                             prefix="proj",
                             disable_tp=True)
    setattr(layer, KEEP_VLLM_LAYOUT_ATTR, keep_layout)
    model = torch.nn.Module()
    model.add_module("proj", layer)
    source = torch.randn(256, 512, generator=torch.Generator().manual_seed(42))
    source[:128, :128] = 0
    source[:, 256:] *= 10
    source = source.to(torch.bfloat16)
    save_file({"weight": source}, tmp_path / "weights.safetensors")
    layer.weight.weight_loader(
        layer.weight,
        load_file(tmp_path / "weights.safetensors")["weight"])
    layer.quant_method.process_weights_after_loading(layer)
    validate_online_fp8(model, config)
    weight = layer.weight.float() if keep_layout else layer.weight.float().t()
    if block is None:
        expected = source.float().abs().max() / 448
        torch.testing.assert_close(layer.weight_scale[0], expected)
        reconstructed = weight * layer.weight_scale
    else:
        bn, bk = block
        scales = layer.weight_scale[0, :, 0, :].t()
        reconstructed = torch.empty_like(weight)
        for n in range(0, 256, bn):
            for k in range(0, 512, bk):
                expected = source[n:n + bn, k:k + bk].float().abs().max() / 448
                torch.testing.assert_close(scales[n:n + bn, k // bk],
                                           expected.expand(bn))
                reconstructed[n:n + bn,
                              k:k + bk] = weight[n:n + bn, k:k + bk] * expected
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed -
            source.float()).norm() / source.float().norm() < 0.04
    if keep_layout:
        from vllm_torchtpu.layers.adapter.custom_ops.gdn_attention_op import \
            VllmGatedDeltaNetAttention
        VllmGatedDeltaNetAttention._require_pcp_projection_parameters(
            SimpleNamespace(in_proj_qkvz=layer, gqa_interleaved_layout=False))
    old_weight = layer.weight
    layer.quant_method.process_weights_after_loading(layer)
    assert layer.weight is old_weight
    # Audit catches a valid dtype with the wrong granularity/layout.
    layer.weight_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    with pytest.raises(ValueError, match="runtime format"):
        validate_online_fp8(model, config)


def test_block_must_divide_local_shard():
    layer = linear(
        make_config(weight_scheme="per_block", weight_block_size=[128, 128]))
    with pytest.raises(ValueError, match="after tensor-parallel sharding"):
        layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.parametrize("block", [[1, 128], [128, 128], [128, 512]])
def test_online_block_matmul_tpu(block):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from vllm_torchtpu.layers.adapter.linear_common import \
        _quantized_matmul_jax

    if jax.default_backend() != "tpu":
        pytest.skip("Requires TPU for the production GMM kernel")
    config = make_config(weight_scheme="per_block", weight_block_size=block)
    layer = ReplicatedLinear(512,
                             256,
                             bias=False,
                             params_dtype=torch.bfloat16,
                             quant_config=config,
                             prefix="proj",
                             disable_tp=True)
    generator = torch.Generator().manual_seed(42)
    source = torch.randn(256, 512, generator=generator).to(torch.bfloat16)
    x = torch.randn(128, 512, generator=generator).to(torch.bfloat16)
    layer.weight.weight_loader(layer.weight, source)
    layer.quant_method.process_weights_after_loading(layer)
    result = _quantized_matmul_jax(
        jnp.asarray(x.float().numpy(), dtype=jnp.bfloat16),
        jnp.asarray(layer.weight.float().numpy(), dtype=jnp.float8_e4m3fn),
        jnp.asarray(layer.weight_scale.numpy()))
    result = np.asarray(result.astype(jnp.float32))
    reference = (x.float() @ source.float().t()).numpy()
    assert np.linalg.norm(result -
                          reference) / np.linalg.norm(reference) < 0.07
