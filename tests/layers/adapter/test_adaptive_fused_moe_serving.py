# SPDX-License-Identifier: Apache-2.0
"""Load-time selection and communication ownership of adaptive fused MoE."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

import vllm_torchtpu.envs as envs
from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import vllm_adapter as bridge
from vllm_torchtpu.layers.adapter import fused_moe_ep as legacy
from vllm_torchtpu.layers.adapter.quantization import fp8, nvfp4

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize(
    "selection,old,enabled",
    [
        (None, "0", False),
        (None, "1", True),
        ("adaptive", "0", True),
        ("adaptive", "1", True),
        ("v2", "0", True),
        ("v2", "1", True),
    ],
)
def test_selector_precedence(monkeypatch, selection, old, enabled):
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", old)
    if selection is None:
        monkeypatch.delenv("MOE_FUSED_EP_KERNEL_IMPL", raising=False)
    else:
        monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", selection)
    assert envs.environment_variables["MOE_FUSED_EP_KERNEL_IMPL"]() == selection
    assert envs.environment_variables["USE_MOE_FUSED_EP_KERNEL"]() is enabled


@pytest.mark.parametrize("value", ["typo", "", "1"])
def test_invalid_selector_raises(monkeypatch, value):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", value)
    with pytest.raises(ValueError, match="MOE_FUSED_EP_KERNEL_IMPL"):
        envs.environment_variables["USE_MOE_FUSED_EP_KERNEL"]()


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "adaptive")
    for flag in (
        "USE_MOE_FUSED_EP_KERNEL",
        "MOE_FUSED_EP_ENABLE_W4A8",
        "MOE_FUSED_EP_V2_SHARDED_PLAN",
    ):
        monkeypatch.setenv(flag, "0")
    monkeypatch.setattr(envs, "MOE_REQUANTIZE_BLOCK_SIZE", None)
    monkeypatch.setattr(
        bridge, "build_ep_mesh", lambda: SimpleNamespace(shape={bridge.EP_AXIS_NAME: 8})
    )
    monkeypatch.setattr(bridge, "ep_mesh_index", lambda: 0)
    monkeypatch.setattr(bridge, "ep_rank_order", lambda: None)
    monkeypatch.setattr(
        bridge, "ep_token_replica_groups", lambda **kw: tuple((r,) for r in range(8))
    )
    monkeypatch.setattr(bridge, "_max_node_tokens", lambda *a: 4096)
    monkeypatch.setattr(
        bridge,
        "_device_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    op = Mock(side_effect=lambda x, *args: x)
    monkeypatch.setattr(bridge, "_build_op", Mock(return_value=op))
    forbidden = Mock(side_effect=AssertionError("legacy/GMM must not be called"))
    monkeypatch.setattr(legacy, "prebuild_fused_moe_ep", forbidden)
    for module in (fp8, nvfp4):
        monkeypatch.setattr(module, "RoutedExperts", torch.nn.Module)
        monkeypatch.setattr(module, "prebuild_fused_moe_kernel", forbidden)
        monkeypatch.setattr(module, "fused_moe_gmm", forbidden)
        monkeypatch.setattr(module, "pipelined_fused_moe_gmm", forbidden)
        monkeypatch.setattr(
            module, "enable_pipelined_collective_and_compute", lambda: False
        )
        monkeypatch.setattr(
            module.moe_routing, "validate_linear_ep_placement", lambda *a: None
        )
        monkeypatch.setattr(
            module.moe_routing, "register_experts_start_buffer", lambda *a, **kw: None
        )
        monkeypatch.setattr(module.moe_routing, "route", forbidden)
    return op, forbidden


def make_layer(fp4):
    e, h, i = 1, 512, 256
    layer = torch.nn.Module()
    layer.activation = "silu"
    layer.renormalize = True
    layer.global_num_experts = 8
    layer.custom_routing_function = None
    layer.scoring_func = "softmax"
    layer.e_score_correction_bias = None
    layer.use_grouped_topk = False
    layer.routed_scaling_factor = 1.0
    layer.moe_config = SimpleNamespace(
        experts_per_token=2,
        moe_parallel_config=SimpleNamespace(
            use_ep=True, pcp_size=1, is_sequence_parallel=False
        ),
    )
    if fp4:
        tensors = dict(
            w13_weight=torch.ones(e, 2 * i, h // 2, dtype=torch.uint8),
            w2_weight=torch.ones(e, h, i // 2, dtype=torch.uint8),
            w13_weight_scale=torch.ones(e, 2 * i, h // 16),
            w2_weight_scale=torch.ones(e, h, i // 16),
            w13_weight_scale_2=torch.ones(e, 2),
            w2_weight_scale_2=torch.ones(e),
        )
    else:
        tensors = dict(
            w13_weight=torch.empty(e, h, 2 * i, dtype=torch.float8_e4m3fn),
            w2_weight=torch.empty(e, i, h, dtype=torch.float8_e4m3fn),
        )
    for name, tensor in tensors.items():
        layer.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    return layer


def make_method(fp4):
    cls = nvfp4.VllmNvfp4MoEMethod if fp4 else fp8.VllmFp8MoEMethodTPU
    method = object.__new__(cls)
    method.moe = SimpleNamespace(has_bias=False, is_act_and_mul=True)
    method.group_size = 16
    if not fp4:
        method.quant_config = SimpleNamespace(is_checkpoint_fp8_serialized=True)
        method.weight_block_size = None
        method.weight_scale_name = "weight_scale"
    return method


def mock_conversion(monkeypatch, fp4):
    if fp4:

        def requant(w, scales, block):
            e, n, k2 = w.shape
            return (
                torch.empty(e, k2 * 2, n // 2, dtype=torch.float4_e2m1fn_x2),
                torch.ones(e, k2 * 2 // block, 1, n),
            )

        converted = Mock(side_effect=requant)
        monkeypatch.setattr(nvfp4, "requant_load_kmajor_fp4", converted)
    else:

        def prepare(layer, **kw):
            return (
                layer.w13_weight,
                torch.ones(1, 1, 1, 512),
                layer.w2_weight,
                torch.ones(1, 1, 1, 512),
                "float8_e4m3fn",
                None,
            )

        converted = Mock(side_effect=prepare)
        monkeypatch.setattr(fp8, "_process_fp8_moe_weights", converted)
    return converted


@pytest.mark.parametrize("fp4", [False, True])
def test_single_selector_load_and_forward(setup, monkeypatch, fp4):
    layer, method = make_layer(fp4), make_method(fp4)
    converted = mock_conversion(monkeypatch, fp4)
    assert not method.supports_internal_mk
    method.process_weights_after_loading(layer)
    assert method.supports_internal_mk
    assert getattr(method, bridge.FUSED_MOE_EP_OP_ATTR) is setup[0]
    assert not legacy.fused_moe_ep_supported(method)
    if fp4:
        assert [call.args[2] for call in converted.call_args_list] == [64, 64]
        assert layer._tpu_fused_w13_scale.shape == (1, 8, 512)
        assert layer._tpu_fused_w2_scale.shape == (1, 4, 512)
    # Dispatch is bound at load time, independent of later environment reads.
    monkeypatch.delenv("MOE_FUSED_EP_KERNEL_IMPL")
    x = torch.ones(2, 512, dtype=torch.bfloat16)
    assert method.apply_monolithic(layer, x, torch.ones(2, 8)) is x
    setup[1].assert_not_called()
    delattr(method, bridge.FUSED_MOE_EP_OP_ATTR)
    with pytest.raises(RuntimeError, match="not prebuilt"):
        method.apply_monolithic(layer, x, torch.ones(2, 8))
    setup[1].assert_not_called()


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("failure", ["no_ep", "old_chip"])
def test_admission_failure_never_falls_back(setup, monkeypatch, fp4, failure):
    layer, method = make_layer(fp4), make_method(fp4)
    if failure == "no_ep":
        layer.moe_config.moe_parallel_config.use_ep = False
        error = "expert-parallel"
    else:
        monkeypatch.setattr(
            bridge,
            "_device_info",
            lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_V6E, 1),
        )
        error = "requires TPU generation >= 7"
    conversion = mock_conversion(monkeypatch, fp4)
    with pytest.raises(ValueError, match=error):
        method.process_weights_after_loading(layer)
    if fp4:
        conversion.assert_not_called()
        assert layer.w13_weight.dtype == torch.uint8
    setup[1].assert_not_called()


@pytest.mark.parametrize("armed", [False, True])
def test_adaptive_runner_owns_reduction_and_pcp(monkeypatch, armed):
    from vllm.model_executor.layers.fused_moe.runner import moe_runner as mr

    import vllm_torchtpu

    class Runner:
        _fused_output_is_reduced = property(lambda self: False)

        def _maybe_dispatch(self, x, gate):
            return "outer_dispatch", gate

        def _maybe_combine(self, shared, x):
            return shared, "outer_combine"

    monkeypatch.setattr(mr, "MoERunner", Runner)
    monkeypatch.setenv("MOE_FUSED_EP_KERNEL_IMPL", "adaptive")
    monkeypatch.setenv("USE_MOE_FUSED_EP_KERNEL", "0")
    vllm_torchtpu._patch_moe_runner_fused_output_is_reduced()
    vllm_torchtpu._patch_moe_explicit_pcp_collectives()
    runner = Runner()
    runner.shared_experts = object()
    runner.moe_config = SimpleNamespace(pcp_size=8, skip_final_all_reduce=False)
    runner._quant_method = SimpleNamespace()
    if armed:
        setattr(runner._quant_method, bridge.FUSED_MOE_EP_OP_ATTR, object())
    assert runner._fused_output_is_reduced is armed
    assert runner._maybe_dispatch("x", "gate") == (
        ("x", "gate") if armed else ("outer_dispatch", "gate")
    )
    assert runner._maybe_combine("shared", "x") == (
        ("shared", "x") if armed else ("shared", "outer_combine")
    )
    runner.moe_config.skip_final_all_reduce = True
    assert not runner._fused_output_is_reduced


def test_selector_is_propagated_to_workers():
    from vllm_torchtpu.platforms.tpu_platform import TpuPlatform

    assert "MOE_FUSED_EP_KERNEL_IMPL" in TpuPlatform.additional_env_vars
