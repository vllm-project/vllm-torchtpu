# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""CPU regression tests for Bailing's TPU KDA adapter, without TPU kernels."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from vllm.model_executor.models import bailing_moe_v3 as upstream
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_torchtpu.layers.adapter.custom_ops import kda_attention_op as kda_ops
from vllm_torchtpu.models.bailing_moe_v3 import attention as bailing

pytestmark = pytest.mark.usefixtures("cpu_model_tests")


class _CpuLinear(nn.Linear):
    def __init__(self, input_size, output_size, params_dtype=None, **kwargs):
        super().__init__(input_size, output_size, bias=False, dtype=params_dtype)

    def forward(self, inputs):
        return super().forward(inputs), None


@pytest.fixture
def kda_builders(monkeypatch):
    builder = Mock(return_value=Mock(side_effect=AssertionError("unexpected KDA call")))
    monkeypatch.setattr(bailing, "build_kimi_dispatched_kda_op", builder)
    return builder


@pytest.fixture
def make_layer(monkeypatch, kda_builders):
    """Build local TP projections without distributed collectives or TPU kernels."""

    def make(
        separate=False,
        num_spec=0,
        safe_gate=True,
        *,
        tp_size=1,
        dtype=torch.bfloat16,
        prefix="model.layers.0.attention",
    ):
        monkeypatch.setattr(
            upstream, "get_tensor_model_parallel_world_size", lambda: tp_size
        )

        class ColumnLinear(_CpuLinear):
            def __init__(self, input_size, output_size, **kwargs):
                widths = output_size if isinstance(output_size, list) else [output_size]
                assert all(width % tp_size == 0 for width in widths)
                super().__init__(input_size, sum(widths) // tp_size, **kwargs)

        class RowLinear(_CpuLinear):
            def __init__(self, input_size, output_size, **kwargs):
                super().__init__(input_size // tp_size, output_size, **kwargs)

        monkeypatch.setattr(upstream, "ColumnParallelLinear", ColumnLinear)
        monkeypatch.setattr(upstream, "MergedColumnParallelLinear", ColumnLinear)
        monkeypatch.setattr(upstream, "RowParallelLinear", RowLinear)
        monkeypatch.setattr(
            upstream, "FusedRMSNormGated", lambda dim, **kw: nn.LayerNorm(dim)
        )
        monkeypatch.setattr(upstream, "_is_fp8_module_excluded", lambda *args: separate)
        context = SimpleNamespace(static_forward_context={}, splitting_ops=[])
        vllm_config = SimpleNamespace(compilation_config=context)
        monkeypatch.setattr(
            upstream,
            "get_current_vllm_config",
            lambda: vllm_config,
        )
        config = SimpleNamespace(
            hidden_size=10,
            head_dim=3,
            num_attention_heads=4,
            short_conv_kernel_size=3,
            no_kda_lora=True,
            kda_safe_gate=safe_gate,
            kda_lower_bound=-5.0,
            rms_norm_eps=1e-6,
        )
        with set_default_torch_dtype(dtype):
            layer = upstream.BailingMoeV3KimiDeltaAttention(
                config,
                prefix=prefix,
                model_config=SimpleNamespace(dtype=dtype),
                cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
                num_speculative_tokens=num_spec,
            )
        with torch.no_grad():
            for i, param in enumerate(layer.parameters()):
                param.copy_(
                    torch.arange(param.numel()).reshape(param.shape) / 1000 + i / 100
                )
        layer.process_weights_after_loading(dtype)
        layer.kv_cache = tuple(
            torch.zeros(8, *shape, dtype=state_dtype)
            for shape, state_dtype in zip(
                layer.get_state_shape(), layer.get_state_dtype(), strict=True
            )
        )
        return layer

    return make


@pytest.fixture
def metadata(monkeypatch):
    value = bailing.AttentionMetadata(
        input_positions=None,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        mamba_state_indices=torch.tensor([1], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        request_distribution=torch.tensor([0, 0, 1], dtype=torch.int32),
    )
    context = SimpleNamespace(attn_metadata={"model.layers.0.attention": value})
    monkeypatch.setattr(bailing, "get_forward_context", lambda: context)
    return value


@pytest.mark.parametrize("safe_gate", [False, True])
@pytest.mark.parametrize("num_spec", [0, 2])
def test_constructor_configures_kda_ops_and_state(
    make_layer, kda_builders, safe_gate, num_spec
):
    layer = make_layer(safe_gate=safe_gate, num_spec=num_spec)
    builder = kda_builders
    lower_bound = -5.0 if safe_gate else None
    builder.assert_called_once_with(
        layer.prefix,
        lower_bound=lower_bound,
        eps=1e-6,
        state_dim_first=bailing.is_conv_state_dim_first(),
        num_spec_tokens=num_spec,
    )
    assert layer.get_state_shape() == ((2, 1, 36), (4, 3, 3))
    assert layer.get_state_dtype() == (torch.float32, torch.float32)


@pytest.mark.parametrize("separate", [False, True])
@pytest.mark.parametrize("tp_size", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_forward_projects_local_heads(make_layer, metadata, separate, tp_size, dtype):
    layer = make_layer(separate=separate, tp_size=tp_size, dtype=dtype)
    heads, dim = 4 // tp_size, 3
    width = heads * dim
    assert layer.get_state_shape() == ((2, 1, 3 * width), (heads, dim, dim))
    hidden = torch.arange(20, dtype=dtype).reshape(2, 10) / 10
    core = torch.arange(2 * width, dtype=dtype).reshape(2, heads, dim) / 10
    op = Mock(return_value=core)
    layer._tpu_kda_op = op
    output = torch.full_like(hidden, float("nan"))
    layer(hidden, torch.arange(2), output)
    op.assert_called_once()
    mixed_qkv, raw_gate, beta, output_gate, *state_args = op.call_args.args
    projection = layer.qkv_proj if separate else layer.qkvb_proj
    projected = hidden @ projection.weight.t()
    torch.testing.assert_close(mixed_qkv, projected[:, : 3 * width])
    expected_beta = (
        hidden @ layer.b_proj.weight.t() if separate else projected[:, 3 * width :]
    )
    torch.testing.assert_close(beta, expected_beta)
    torch.testing.assert_close(raw_gate, hidden @ layer.f_proj.weight.t())
    torch.testing.assert_close(output_gate, hidden @ layer.g_proj.weight.t())
    assert mixed_qkv.shape == (2, 3 * width) and beta.shape == (2, heads)
    assert all(
        value.dtype == dtype for value in (mixed_qkv, raw_gate, beta, output_gate)
    )
    conv_state, recurrent_state, conv_weight, a_log, dt_bias, norm_weight, *_ = (
        state_args
    )
    assert all(
        value.dtype == torch.float32
        for value in (conv_state, recurrent_state, conv_weight, a_log, dt_bias)
    )
    assert norm_weight.dtype == dtype
    torch.testing.assert_close(output, core.flatten(1) @ layer.o_proj.weight.t())


@pytest.mark.parametrize("num_spec", [0, 2])
@torch.inference_mode()
def test_forward_routes_cache_and_metadata(make_layer, metadata, num_spec):
    layer = make_layer(num_spec=num_spec)
    if num_spec:
        metadata.mamba_request_distribution = torch.tensor([1, 1, 1], dtype=torch.int32)
        metadata.mamba_slot_read_offsets = torch.tensor(
            [0, 1, 0, 0, 0, 0, 0, 0], dtype=torch.int32
        )
    op = Mock(return_value=torch.zeros(2, 4, 3, dtype=torch.bfloat16))
    layer._tpu_kda_op = op
    window_distribution = (
        metadata.mamba_request_distribution
        if num_spec
        else metadata.request_distribution
    )
    read_offsets = (
        metadata.mamba_slot_read_offsets if num_spec else metadata.mamba_state_indices
    )
    layer(
        torch.ones(2, 10, dtype=torch.bfloat16),
        torch.arange(2),
        torch.empty(2, 10, dtype=torch.bfloat16),
    )
    op.assert_called_once()
    expected = (
        *layer.kv_cache,
        layer.tpu_conv_weight_fused,
        layer.dt_bias,
        layer.o_norm.weight,
        metadata.query_start_loc,
        metadata.mamba_state_indices,
        metadata.seq_lens,
        metadata.request_distribution,
        window_distribution,
        read_offsets,
    )
    actual = list(op.call_args.args[4:])
    torch.testing.assert_close(
        actual.pop(len(layer.kv_cache) + 1), layer.A_log.flatten()
    )
    assert not op.call_args.kwargs
    for value, reference in zip(actual, expected, strict=True):
        assert value is reference


@torch.inference_mode()
def test_forward_compiles_and_writes_back_state(make_layer, metadata, monkeypatch):
    """Keep the real bridge/fake op; replace only the TPU kernel with a CPU op."""

    def dense_kernel(
        qkv: torch.Tensor,
        gate: torch.Tensor,
        beta: torch.Tensor,
        output_gate: torch.Tensor,
        conv: torch.Tensor,
        recurrent: torch.Tensor,
        weight: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        norm: torch.Tensor,
        starts: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        distribution: torch.Tensor,
        windows: torch.Tensor,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        heads, dim = recurrent.shape[1:3]
        output = qkv[:, : heads * dim].reshape(-1, heads, dim).clone()
        return output, conv + 1, recurrent + 2

    def jax_op(name, function, donate_argnums=()):
        assert donate_argnums == (4, 5)
        return torch.library.custom_op(name, mutates_args=())(dense_kernel)

    monkeypatch.setattr(kda_ops.pallas, "jax_op", jax_op)
    monkeypatch.setattr(
        bailing, "build_kimi_dispatched_kda_op", kda_ops.build_kimi_dispatched_kda_op
    )
    layer = make_layer(prefix="test_bailing_compile")
    context = SimpleNamespace(attn_metadata={layer.prefix: metadata})
    monkeypatch.setattr(bailing, "get_forward_context", lambda: context)
    hidden = torch.arange(20, dtype=torch.bfloat16).reshape(2, 10) / 10
    positions = torch.arange(2)
    expected = torch.empty_like(hidden)
    layer(hidden, positions, expected)
    increments = (1, 2)
    for state, increment in zip(layer.kv_cache, increments, strict=True):
        torch.testing.assert_close(state, torch.full_like(state, increment))
        state.zero_()
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    actual = torch.full_like(hidden, float("nan"))
    for step in (1, 2):
        compiled(hidden, positions, actual)
        torch.testing.assert_close(actual, expected)
        for state, increment in zip(layer.kv_cache, increments, strict=True):
            torch.testing.assert_close(state, torch.full_like(state, step * increment))


@pytest.mark.parametrize("tp_size", [1, 2])
def test_conv_weight_fold_preserves_channels_and_refreshes_after_reload(
    make_layer, tp_size
):
    layer = make_layer(tp_size=tp_size)
    previous = layer.tpu_conv_weight_fused.clone()
    with torch.no_grad():
        layer.q_conv1d.weight.add_(1)
    layer.process_weights_after_loading(torch.bfloat16)
    fused = layer.tpu_conv_weight_fused
    assert fused.shape == (3, 3, 4 // tp_size, 3)
    assert fused.dtype == torch.float32 and fused.is_contiguous()
    assert "tpu_conv_weight_fused" not in layer.state_dict()
    for i, conv in enumerate((layer.q_conv1d, layer.k_conv1d, layer.v_conv1d)):
        torch.testing.assert_close(fused[:, i].flatten(1).t(), conv.weight[:, 0])
    torch.testing.assert_close(fused[:, 0], previous[:, 0] + 1)
    with torch.no_grad():
        layer.q_conv1d.weight.zero_()
    torch.testing.assert_close(fused[:, 0], previous[:, 0] + 1)


@pytest.mark.parametrize("missing", ["metadata", "cache", "empty_cache"])
def test_warmup_does_not_invoke_kda(make_layer, metadata, monkeypatch, missing):
    layer = make_layer()
    if missing == "metadata":
        monkeypatch.setattr(
            bailing,
            "get_forward_context",
            lambda: SimpleNamespace(attn_metadata=None),
        )
    elif missing == "cache":
        layer.kv_cache = ()
    else:
        layer.kv_cache = (torch.empty(0), torch.empty(0))
    output = torch.full((2, 10), float("nan"), dtype=torch.bfloat16)
    layer(torch.ones(2, 10, dtype=torch.bfloat16), torch.arange(2), output)
    torch.testing.assert_close(output, torch.zeros_like(output))
    layer._tpu_kda_op.assert_not_called()


def test_forward_rejects_missing_layer_metadata(make_layer, monkeypatch):
    layer = make_layer()
    monkeypatch.setattr(
        bailing, "get_forward_context", lambda: SimpleNamespace(attn_metadata={})
    )
    with pytest.raises(KeyError) as exc_info:
        layer(
            torch.ones(2, 10, dtype=torch.bfloat16),
            torch.arange(2),
            torch.empty(2, 10, dtype=torch.bfloat16),
        )
    assert exc_info.value.args == (layer.prefix,)
    layer._tpu_kda_op.assert_not_called()


def test_registration_keeps_upstream_class_intact(make_layer):
    layer = make_layer()
    assert type(layer) is bailing.TpuBailingKimiDeltaAttention
    context = (
        upstream.get_current_vllm_config().compilation_config.static_forward_context
    )
    assert context[layer.prefix] is layer
    for name in ("__init__", "forward", "get_state_shape", "get_state_dtype"):
        method = getattr(upstream.BailingMoeV3KimiDeltaAttention, name)
        assert method.__module__ == upstream.__name__


def _check_registration_in_fresh_process(code: str, *args: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
            "JAX_PLATFORMS": "cpu",
        },
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_patch_import_is_lightweight_and_defers_unrelated_models():
    _check_registration_in_fresh_process("""
import sys
from types import SimpleNamespace
for name in ("torch", "torch_tpu", "jax", "jaxlib", "vllm", "numpy"):
    sys.modules[name] = None
from vllm_torchtpu.models.bailing_moe_v3.patch import maybe_patch_bailing_moe_v3
assert maybe_patch_bailing_moe_v3() is False
config = SimpleNamespace(architecture="Qwen3ForCausalLM")
assert maybe_patch_bailing_moe_v3(config) is False
assert "vllm_torchtpu.models.bailing_moe_v3.attention" not in sys.modules
""")


@pytest.mark.parametrize(
    "architecture",
    ["BailingMoeV3ForCausalLM", "BailingMoeV3VLForConditionalGeneration"],
)
@pytest.mark.parametrize("stage", ["platform_activation", "model_load"])
def test_patch_registers_oot_layer_in_fresh_process(architecture, stage):
    _check_registration_in_fresh_process(
        """
import sys
from types import SimpleNamespace
from vllm_torchtpu import patch_registry as registry
target = "vllm_torchtpu.models.bailing_moe_v3.patch:maybe_patch_bailing_moe_v3"
registry.PATCHES = tuple(p for p in registry.PATCHES if p.target == target)
assert len(registry.PATCHES) == 1
registry.apply("platform_activation")
assert "vllm_torchtpu.models.bailing_moe_v3.attention" not in sys.modules
config = SimpleNamespace(architecture=sys.argv[1])
registry.apply(sys.argv[2], model_config=config)
assert "vllm_torchtpu.models.bailing_moe_v3.attention" in sys.modules
from vllm.model_executor.models.bailing_moe_v3 import BailingMoeV3KimiDeltaAttention
from vllm_torchtpu.models.bailing_moe_v3.attention import TpuBailingKimiDeltaAttention
layer = BailingMoeV3KimiDeltaAttention.__new__(BailingMoeV3KimiDeltaAttention)
assert type(layer) is TpuBailingKimiDeltaAttention
registry.apply("platform_activation", model_config=config)
registry.apply("model_load", model_config=config)
""",
        architecture,
        stage,
    )
