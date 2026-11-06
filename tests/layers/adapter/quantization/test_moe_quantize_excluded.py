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
"""CPU tests for tpu_online_fp8 on checkpoint-excluded RoutedExperts layers."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
from jax.experimental.pallas import tpu as pltpu
from vllm.config import CompilationConfig, set_current_vllm_config
from vllm.config.vllm import get_cached_compilation_config
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.runner import moe_runner as moe_runner_mod
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.layers.linear import LinearBase, ReplicatedLinear
from vllm.platforms import current_platform

import vllm_torchtpu.envs as envs
from tests.layers.adapter.test_moe_tp_decode import _Gate, _SharedMLP
from vllm_torchtpu.layers.adapter import moe_tp_decode, token_padding
from vllm_torchtpu.layers.adapter.quantization import (
    fp8 as fp8_mod,
)
from vllm_torchtpu.layers.adapter.quantization import (
    online_fp8 as online_fp8_mod,
)
from vllm_torchtpu.layers.adapter.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    VllmCompressedTensorsConfig,
)
from vllm_torchtpu.layers.adapter.quantization.deepseek_v4_fp8 import (
    VllmDeepseekV4Fp8Config,
)
from vllm_torchtpu.layers.adapter.quantization.fp8 import (
    VllmFp8Config,
    VllmFp8MoEMethodTPU,
    build_excluded_moe_online_fp8_method,
)
from vllm_torchtpu.layers.adapter.quantization.mxfp4 import VllmMxfp4Config
from vllm_torchtpu.layers.adapter.quantization.nvfp4 import (
    VllmNvfp4Config,
    VllmNvfp4MoEMethod,
)
from vllm_torchtpu.layers.adapter.quantization.online_fp8 import (
    attach_online_fp8,
    validate_online_fp8,
)
from vllm_torchtpu.layers.adapter.quantization.unquantized import (
    VllmUnquantizedConfig,
    VllmUnquantizedFusedMoEMethod,
    VllmUnquantizedLinearMethod,
)

# Small shapes divisible by 512 so a leaked MOE_REQUANTIZE_BLOCK_SIZE=512 would
# silently produce multi-block scales (H // 512 == 4) rather than raising.
H = 2048
INTER = 512
TOPK = 4
FP8_DTYPE = torch.float8_e4m3fn


@pytest.fixture(autouse=True)
def _ensure_tpu_platform(monkeypatch):
    """Ensure vLLM's unquantized MoE oracle sees TPU platform even when pytest
    runs from a checkout without editable plugin entry-point metadata."""
    if not current_platform.is_tpu():
        monkeypatch.setattr(current_platform, "is_tpu", lambda: True)
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(parallel_state, "_TP", group)


def _make_moe_config(
    *,
    local_experts: int = 2,
    ep_size: int = 2,
    tp_size: int = 4,
    use_ep: bool = True,
) -> tuple[FusedMoEConfig, ExpertMapManager]:
    global_experts = local_experts * (ep_size if use_ep else 1)
    parallel = FusedMoEParallelConfig(
        tp_size=tp_size,
        pcp_size=1,
        dp_size=1,
        ep_size=ep_size if use_ep else 1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=use_ep,
        all2all_backend="allgather_reducescatter",
        enable_eplb=False,
    )
    moe_cfg = FusedMoEConfig(
        num_experts=global_experts,
        experts_per_token=TOPK,
        hidden_dim=H,
        intermediate_size=INTER * tp_size,
        num_local_experts=local_experts,
        num_logical_experts=global_experts,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
    )
    expert_map = ExpertMapManager(
        max_num_batched_tokens=64,
        top_k=TOPK,
        global_num_experts=global_experts,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel,
        placement_strategy="linear",
        enable_eplb=False,
    )
    return moe_cfg, expert_map


def _attach_policy(
    cfg: VllmNvfp4Config | VllmFp8Config,
    targets: list[str] | None,
    *,
    enabled: bool = True,
    exclude: list[str] | None = None,
    **options,
) -> None:
    if targets is None and enabled:
        additional_config: dict = {}
    else:
        policy_dict: dict = {"enabled": enabled, **options}
        if targets is not None:
            policy_dict["targets"] = targets
        if exclude is not None:
            policy_dict["exclude"] = exclude
        additional_config = {"tpu_online_fp8": policy_dict}
    cfg.vllm_config.additional_config = additional_config
    attach_online_fp8(cfg, cfg.vllm_config)


def _make_nvfp4_config(
    excluded_prefixes: list[str],
    *,
    targets: list[str] | None = None,
    enabled: bool = True,
    exclude: list[str] | None = None,
    **options,
) -> VllmNvfp4Config:
    cfg = VllmNvfp4Config.from_config(
        {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "ignore": excluded_prefixes,
        }
    )
    cfg.vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(custom_ops=["all"]),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        additional_config={},
    )
    _attach_policy(cfg, targets, enabled=enabled, exclude=exclude, **options)
    return cfg


def _make_fp8_config(
    excluded_prefixes: list[str],
    *,
    targets: list[str] | None = None,
    enabled: bool = True,
    exclude: list[str] | None = None,
    **options,
) -> VllmFp8Config:
    cfg = VllmFp8Config.from_config(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": excluded_prefixes,
        }
    )
    cfg.vllm_config = SimpleNamespace(
        compilation_config=CompilationConfig(custom_ops=["all"]),
        parallel_config=SimpleNamespace(enable_expert_parallel=True),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(expert_dtype="fp8")),
        additional_config={},
    )
    _attach_policy(cfg, targets, enabled=enabled, exclude=exclude, **options)
    return cfg


def _make_quant_config(
    kind: str,
    excluded_layer_ids: tuple[int, ...] = (78,),
    *,
    targets: list[str] | None = None,
    enabled: bool = True,
    exclude: list[str] | None = None,
    **options,
) -> VllmNvfp4Config | VllmFp8Config:
    if kind == "nvfp4":
        # ModelOpt NVFP4 checkpoints (e.g. GLM-5.2-NVFP4) use glob ignore patterns.
        return _make_nvfp4_config(
            [f"model.layers.{idx}*" for idx in excluded_layer_ids],
            targets=targets,
            enabled=enabled,
            exclude=exclude,
            **options,
        )
    # Upstream FP8 is_layer_skipped matches exact layer prefixes rather than globs.
    excluded: list[str] = []
    for idx in excluded_layer_ids:
        excluded.extend(
            (
                f"model.layers.{idx}.mlp.experts",
                f"model.layers.{idx}.self_attn.q_b_proj",
                f"model.layers.{idx}.shared_proj",
            )
        )
    return _make_fp8_config(
        excluded,
        targets=targets,
        enabled=enabled,
        exclude=exclude,
        **options,
    )


def _build_routed_experts(
    quant_config: object | None,
    prefix: str,
    *,
    local_experts: int = 2,
    ep_size: int = 2,
    tp_size: int = 4,
    use_ep: bool = True,
    params_dtype: torch.dtype = torch.bfloat16,
) -> RoutedExperts:
    moe_cfg, expert_map = _make_moe_config(
        local_experts=local_experts,
        ep_size=ep_size,
        tp_size=tp_size,
        use_ep=use_ep,
    )
    vllm_cfg = (
        quant_config.vllm_config
        if quant_config is not None
        else SimpleNamespace(
            compilation_config=CompilationConfig(custom_ops=["all"]),
            parallel_config=SimpleNamespace(enable_expert_parallel=use_ep),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        )
    )
    get_cached_compilation_config.cache_clear()
    try:
        with set_current_vllm_config(vllm_cfg):
            layer = RoutedExperts(
                layer_name=prefix,
                params_dtype=params_dtype,
                moe_config=moe_cfg,
                quant_config=quant_config,
                expert_map_manager=expert_map,
            )
    finally:
        get_cached_compilation_config.cache_clear()
    return layer


def _make_linear_stub() -> LinearBase:
    layer = LinearBase.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    layer.input_size = 64
    layer.output_size = 32
    return layer


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
@pytest.mark.parametrize("mode", ["no_policy", "disabled", "untargeted"])
def test_default_and_untargeted_are_byte_identical(monkeypatch, kind, mode):
    """Unset, disabled, or non-matching policy keeps VllmUnquantizedFusedMoEMethod."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    if mode == "no_policy":
        cfg = _make_quant_config(kind, (78,))
    elif mode == "disabled":
        cfg = _make_quant_config(kind, (78,), enabled=False)
    else:
        cfg = _make_quant_config(kind, (78,), targets=["model.layers.79.mlp.experts"])

    layer = _build_routed_experts(cfg, "model.layers.78.mlp.experts")
    assert type(layer.quant_method) is VllmUnquantizedFusedMoEMethod
    assert layer.w13_weight.dtype == torch.bfloat16
    assert layer.w2_weight.dtype == torch.bfloat16
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert not hasattr(layer, "w2_weight_scale_inv")
    assert not hasattr(layer, "w13_weight_scale")
    assert not hasattr(layer, "w2_weight_scale")

    gen = torch.Generator().manual_seed(42)
    w13_src = torch.randn(
        layer.w13_weight.shape, generator=gen, dtype=torch.float32
    ).to(torch.bfloat16)
    w2_src = torch.randn(layer.w2_weight.shape, generator=gen, dtype=torch.float32).to(
        torch.bfloat16
    )
    layer.w13_weight.data.copy_(w13_src)
    layer.w2_weight.data.copy_(w2_src)

    ctrl_cfg = _make_nvfp4_config(["model.layers.78*"])
    ctrl = _build_routed_experts(ctrl_cfg, "model.layers.78.mlp.experts")
    ctrl.w13_weight.data.copy_(w13_src)
    ctrl.w2_weight.data.copy_(w2_src)

    with patch(
        "vllm_torchtpu.layers.adapter.quantization.unquantized.prebuild_fused_moe_kernel"
    ):
        layer.quant_method.process_weights_after_loading(layer)
        ctrl.quant_method.process_weights_after_loading(ctrl)

    assert layer.w13_weight.dtype == torch.bfloat16
    assert layer.w2_weight.dtype == torch.bfloat16
    assert torch.equal(layer.w13_weight.data, ctrl.w13_weight.data)
    assert torch.equal(layer.w2_weight.data, ctrl.w2_weight.data)
    layer_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    ctrl_bytes = sum(p.numel() * p.element_size() for p in ctrl.parameters())
    assert layer_bytes == ctrl_bytes
    assert not hasattr(layer, "w13_weight_scale_inv")
    assert not hasattr(layer, "w2_weight_scale_inv")


def test_already_quantized_and_untargeted_linear_layers_never_touched(monkeypatch):
    """Targeting excluded experts leaves quantized MoE and untargeted Linears intact."""
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )

    nvfp4_cfg = _make_quant_config(
        "nvfp4", (78,), targets=["model.layers.78.mlp.experts"]
    )
    moe_cfg, _ = _make_moe_config()
    target_nvfp4 = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(target_nvfp4)
    target_nvfp4.moe_config = moe_cfg

    qm_nvfp4 = nvfp4_cfg.get_quant_method(target_nvfp4, "model.layers.3.mlp.experts")
    assert type(qm_nvfp4) is VllmNvfp4MoEMethod
    assert qm_nvfp4.quant_config is nvfp4_cfg

    excluded_linear = _make_linear_stub()
    qm_lin = nvfp4_cfg.get_quant_method(
        excluded_linear, "model.layers.78.self_attn.q_b_proj"
    )
    assert type(qm_lin) is VllmUnquantizedLinearMethod

    fp8_cfg = _make_quant_config("fp8", (78,), targets=["model.layers.78.mlp.experts"])
    target_fp8 = _build_routed_experts(fp8_cfg, "model.layers.3.mlp.experts")
    assert type(target_fp8.quant_method) is VllmFp8MoEMethodTPU
    assert target_fp8.quant_method.quant_config is fp8_cfg
    assert target_fp8.quant_method.online_fp8 is False
    assert target_fp8.quant_method._force_per_channel is False
    assert target_fp8.quant_method._online_fp8_excluded_prefix is None
    qm_fp8_lin = fp8_cfg.get_quant_method(
        excluded_linear, "model.layers.78.self_attn.q_b_proj"
    )
    assert type(qm_fp8_lin) is VllmUnquantizedLinearMethod


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
@pytest.mark.parametrize(
    "options",
    [
        {"weight_scheme": "per_tensor"},
        {"weight_scheme": "per_block", "weight_block_size": [128, 128]},
    ],
)
def test_non_per_channel_scheme_rejected_for_moe_target(monkeypatch, kind, options):
    """MoE online FP8 is per_channel only; per_tensor/per_block raise ValueError."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    cfg = _make_quant_config(
        kind, (78,), targets=["model.layers.78.mlp.experts"], **options
    )
    with pytest.raises(ValueError, match="weight_scheme='per_channel'"):
        _build_routed_experts(cfg, "model.layers.78.mlp.experts")


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_already_quantized_moe_target_raises(monkeypatch, kind):
    """Targeting a RoutedExperts layer with a quantized checkpoint method raises."""
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    cfg = _make_quant_config(kind, (78,), targets=["model.layers.3.mlp.experts"])
    moe_cfg, _ = _make_moe_config()
    routed = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(routed)
    routed.moe_config = moe_cfg
    with pytest.raises(ValueError, match="already has checkpoint method"):
        cfg.get_quant_method(routed, "model.layers.3.mlp.experts")


@pytest.mark.parametrize(
    "other_cls",
    [
        VllmCompressedTensorsConfig,
        VllmDeepseekV4Fp8Config,
        VllmMxfp4Config,
        VllmUnquantizedConfig,
    ],
)
def test_sibling_quant_configs_reject_tpu_online_fp8(other_cls):
    """attach_online_fp8 rejects non-(fp8, modelopt_fp4) configs instead of ignoring."""
    stub_cfg = other_cls.__new__(other_cls)
    vllm_cfg = SimpleNamespace(
        additional_config={
            "tpu_online_fp8": {
                "enabled": True,
                "targets": ["model.layers.78.mlp.experts"],
            }
        }
    )
    with pytest.raises(ValueError, match="supports FP8 and ModelOpt NVFP4"):
        attach_online_fp8(stub_cfg, vllm_cfg)


def test_moe_online_fp8_requires_bf16_weights():
    """process_weights_after_loading rejects non-BF16 checkpoint weights."""
    cfg = _make_quant_config("nvfp4", (78,), targets=["model.layers.78.mlp.experts"])
    layer = _build_routed_experts(
        cfg, "model.layers.78.mlp.experts", params_dtype=torch.float32
    )
    with pytest.raises(ValueError, match="requires BF16"):
        layer.quant_method.process_weights_after_loading(layer)


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_requant_numerics_block512_isolation_logging_and_direct_parity(
    monkeypatch, kind
):
    """Per-channel FP8 scales hold under MOE_REQUANTIZE_BLOCK_SIZE=512, match
    direct build_excluded_moe_online_fp8_method bit-for-bit, and log per layer.
    """
    # Recipe-level NVFP4/MoE overrides must NOT leak into excluded online-FP8 experts.
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("MOE_REQUANTIZE_WEIGHT_DTYPE", "int8")
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )

    local_experts = 2
    targets = ["model.layers.78.mlp.experts", "model.layers.79.mlp.experts"]
    cfg = _make_quant_config(kind, (78, 79), targets=targets)
    layer_a = _build_routed_experts(
        cfg, "model.layers.78.mlp.experts", local_experts=local_experts
    )
    layer_b = _build_routed_experts(
        cfg, "model.layers.79.mlp.experts", local_experts=local_experts
    )

    # Direct method construction (the exact function the former env path called).
    base_cfg = _make_quant_config(kind, (78,))
    direct_cfg = SimpleNamespace(
        vllm_config=base_cfg.vllm_config,
        get_quant_method=lambda lyr, pfx: build_excluded_moe_online_fp8_method(
            base_cfg.get_moe_config(lyr), prefix=pfx
        ),
    )
    direct_layer = _build_routed_experts(
        direct_cfg,
        "model.layers.78.mlp.experts",
        local_experts=local_experts,
    )

    for layer in (layer_a, layer_b):
        assert type(layer.quant_method) is VllmFp8MoEMethodTPU
        assert layer.quant_method.online_fp8 is True
        assert layer.quant_method.quant_config.is_checkpoint_fp8_serialized is False
        assert layer.w13_weight.dtype == torch.bfloat16
        assert layer.w2_weight.dtype == torch.bfloat16

    gen = torch.Generator().manual_seed(7)
    w13_orig = (
        torch.randn((local_experts, 2 * INTER, H), generator=gen, dtype=torch.float32)
        * 1.75
    ).to(torch.bfloat16)
    w13_orig[0, 0, :] = 0.0  # exact-zero output channel check
    w2_orig = (
        torch.randn((local_experts, H, INTER), generator=gen, dtype=torch.float32)
        * 0.85
    ).to(torch.bfloat16)

    for layer in (layer_a, layer_b, direct_layer):
        layer.w13_weight.data.copy_(w13_orig)
        layer.w2_weight.data.copy_(w2_orig)

    logged_info: list[str] = []
    logged_once: list[str] = []
    fake_logger = SimpleNamespace(
        info=lambda fmt, *args: logged_info.append(fmt % args if args else fmt),
        info_once=lambda fmt, *args: logged_once.append(fmt % args if args else fmt),
        warning=lambda fmt, *args: None,
        warning_once=lambda fmt, *args: None,
    )
    monkeypatch.setattr(fp8_mod, "logger", fake_logger)

    with patch(
        "vllm_torchtpu.layers.adapter.quantization.fp8.prebuild_fused_moe_kernel"
    ):
        layer_a.quant_method.process_weights_after_loading(layer_a)
        layer_b.quant_method.process_weights_after_loading(layer_b)
        direct_layer.quant_method.process_weights_after_loading(direct_layer)

    # 1) Exact bit-for-bit equality with build_excluded_moe_online_fp8_method.
    assert torch.equal(layer_a.w13_weight.data, direct_layer.w13_weight.data)
    assert torch.equal(layer_a.w2_weight.data, direct_layer.w2_weight.data)
    assert torch.equal(
        layer_a.w13_weight_scale_inv.data, direct_layer.w13_weight_scale_inv.data
    )
    assert torch.equal(
        layer_a.w2_weight_scale_inv.data, direct_layer.w2_weight_scale_inv.data
    )

    # 2) Block-512 & int8 isolation: scales stay per-channel (1, not H // 512 == 4).
    assert layer_a.w13_weight.dtype == FP8_DTYPE
    assert layer_a.w2_weight.dtype == FP8_DTYPE
    assert layer_a.w13_weight.shape == (local_experts, H, 2 * INTER)
    assert layer_a.w2_weight.shape == (local_experts, INTER, H)
    assert layer_a.w13_weight_scale_inv.shape == (local_experts, 1, 1, 2 * INTER)
    assert layer_a.w2_weight_scale_inv.shape == (local_experts, 1, 1, H)
    assert layer_a.w13_weight_scale_inv.dtype == torch.float32
    assert layer_a.w2_weight_scale_inv.dtype == torch.float32

    # 3) Per-channel FP8 round-trip numerics vs reference and analytic E4M3 bound.
    def _ref_per_channel_fp8_kmajor(w_bf16: torch.Tensor) -> torch.Tensor:
        w_f32 = w_bf16.float()
        amax = w_f32.abs().amax(dim=-1, keepdim=True)
        fp8_max = float(torch.finfo(FP8_DTYPE).max)
        dtype_max_recip = torch.tensor(1.0 / fp8_max, dtype=torch.float32)
        scale = amax * dtype_max_recip
        scale_inv = torch.where(scale == 0, torch.zeros_like(scale), 1.0 / scale)
        q = (w_f32 * scale_inv).clamp(-fp8_max, fp8_max).to(FP8_DTYPE)
        return (q.float() * scale).transpose(1, 2).contiguous()

    w13_deq = layer_a.w13_weight.float() * layer_a.w13_weight_scale_inv.squeeze(2)
    w2_deq = layer_a.w2_weight.float() * layer_a.w2_weight_scale_inv.squeeze(2)

    assert torch.equal(w13_deq, _ref_per_channel_fp8_kmajor(w13_orig))
    assert torch.equal(w2_deq, _ref_per_channel_fp8_kmajor(w2_orig))

    w13_orig_kmajor = w13_orig.float().transpose(1, 2).contiguous()
    w2_orig_kmajor = w2_orig.float().transpose(1, 2).contiguous()
    w13_chan_amax = w13_orig_kmajor.abs().amax(dim=1, keepdim=True)
    w2_chan_amax = w2_orig_kmajor.abs().amax(dim=1, keepdim=True)
    assert torch.all((w13_deq - w13_orig_kmajor).abs() <= w13_chan_amax / 16.0 + 1e-6)
    assert torch.all((w2_deq - w2_orig_kmajor).abs() <= w2_chan_amax / 16.0 + 1e-6)
    assert torch.equal(w13_deq[0, :, 0], torch.zeros(H))
    w13_rel = (w13_deq - w13_orig_kmajor).abs().mean() / w13_orig_kmajor.abs().mean()
    w2_rel = (w2_deq - w2_orig_kmajor).abs().mean() / w2_orig_kmajor.abs().mean()
    assert w13_rel < 0.025
    assert w2_rel < 0.025

    # 4) Logging: one line per converted layer; reload is idempotent.
    assert logged_once == []
    assert len(logged_info) == 3
    assert "model.layers.78.mlp.experts" in logged_info[0]
    assert "model.layers.79.mlp.experts" in logged_info[1]
    for line in logged_info[:2]:
        assert "float8_e4m3fn" in line
        assert f"({local_experts}, 1, 1, {2 * INTER})" in line
        assert f"({local_experts}, 1, 1, {H})" in line

    old_w13 = layer_a.w13_weight
    layer_a.quant_method.process_weights_after_loading(layer_a)
    assert layer_a.w13_weight is old_w13


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_validate_online_fp8_moe_draft_model_and_wildcards(monkeypatch, kind):
    """validate_online_fp8 audits MoE + Linear across target and draft_model,
    requires draft_model for drafter-only targets, and dedupes shared modules.
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )

    targets = [
        "model.layers.*.self_attn.q_b_proj",
        "model.layers.0.shared_proj",
        "model.layers.78.mlp.experts",
    ]
    cfg = _make_quant_config(kind, (0, 78), targets=targets)

    # Target model has layer 0 q_b_proj and a shared projection reused by draft_model.
    target_model = torch.nn.Module()
    target_q_b = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.0.self_attn.q_b_proj",
        disable_tp=True,
    )
    shared_proj = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.0.shared_proj",
        disable_tp=True,
    )
    target_model.add_module("q_b", target_q_b)
    target_model.add_module("shared_proj", shared_proj)

    # Draft model has layer 78 q_b_proj, shares shared_proj, and wraps RoutedExperts
    # inside a MoERunner at mtp_block.mlp.experts (matching DeepSeek/GLM MTP layout).
    draft_q_b = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.78.self_attn.q_b_proj",
        disable_tp=True,
    )
    draft_routed = _build_routed_experts(
        cfg, "model.layers.78.mlp.experts", local_experts=2
    )
    draft_runner = MoERunner(
        layer_name="model.layers.78.mlp.experts",
        moe_config=draft_routed.moe_config,
        router=SimpleNamespace(),
        routed_experts=draft_routed,
        gate=_Gate(4),
        shared_experts=_SharedMLP(),
        routed_scaling_factor=2.5,
    )
    mtp_block = torch.nn.Module()
    mlp = torch.nn.Module()
    mlp.add_module("experts", draft_runner)
    mtp_block.add_module("mlp", mlp)
    mtp_block.add_module("q_b", draft_q_b)
    draft_model = torch.nn.Module()
    draft_model.add_module("mtp_block", mtp_block)
    draft_model.add_module("shared_proj", shared_proj)

    for lin in (target_q_b, shared_proj, draft_q_b):
        lin.weight.data.normal_(0.0, 0.1)
        lin.quant_method.process_weights_after_loading(lin)

    # Before processing draft_routed:
    # 1) Calling validate_online_fp8 without draft_model fails on missing/unprocessed.
    #    Test a drafter-only policy explicitly to verify the "unmatched" failure before
    #    draft_model is constructed, and "unprocessed" once constructed.
    drafter_only_cfg = _make_quant_config(
        kind, (78,), targets=["model.layers.78.mlp.experts"]
    )
    with pytest.raises(ValueError, match="unmatched"):
        validate_online_fp8(torch.nn.Module(), drafter_only_cfg)

    # 2) With draft_model present but draft_routed not yet processed -> "not processed".
    with pytest.raises(ValueError, match="not processed"):
        validate_online_fp8(target_model, cfg, draft_model=draft_model)

    draft_routed.w13_weight.data.normal_(0.0, 0.02)
    draft_routed.w2_weight.data.normal_(0.0, 0.02)
    with patch(
        "vllm_torchtpu.layers.adapter.quantization.fp8.prebuild_fused_moe_kernel"
    ):
        draft_routed.quant_method.process_weights_after_loading(draft_routed)

    # Without draft_model passed, drafter modules in policy.selected are missing.
    with pytest.raises(ValueError, match="model.layers.78.mlp.experts"):
        validate_online_fp8(target_model, cfg)

    audit_logs: list[str] = []
    monkeypatch.setattr(
        online_fp8_mod,
        "logger",
        SimpleNamespace(
            info=lambda fmt, *args: audit_logs.append(fmt % args if args else fmt)
        ),
    )

    validate_online_fp8(target_model, cfg, draft_model=draft_model)

    # 3 Linear (target_q_b, shared_proj once, draft_q_b) + 1 MoE + 1 summary = 5.
    assert len(audit_logs) == 5
    assert (
        audit_logs[-1] == "TPU online FP8 validated 4 runtime layers (3 Linear, 1 MoE)"
    )
    moe_payload = [
        json.loads(line.removeprefix("TPU online FP8: "))
        for line in audit_logs[:-1]
        if '"model.layers.78.mlp.experts"' in line
    ]
    assert len(moe_payload) == 1
    assert moe_payload[0]["runtime_dtype"] == "float8_e4m3fn"
    assert moe_payload[0]["w13_scale_shape"] == [2, 1, 1, 2 * INTER]
    assert moe_payload[0]["w2_scale_shape"] == [2, 1, 1, H]

    # Corrupted block-scaled MoE scale shape is rejected by validate_online_fp8.
    draft_routed.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones(2, 4, 1, 2 * INTER, dtype=torch.float32), requires_grad=False
    )
    with pytest.raises(ValueError, match="runtime format"):
        validate_online_fp8(target_model, cfg, draft_model=draft_model)

    # Bypassed MoE dispatch (constructed without quant_config) is caught.
    bypass_cfg = _make_quant_config(
        kind, (78,), targets=["model.layers.78.mlp.experts"]
    )
    bypass_model = torch.nn.Module()
    with patch(
        "vllm.model_executor.layers.fused_moe.routed_experts."
        "UnquantizedFusedMoEMethod.create_weights",
        lambda *args, **kwargs: None,
    ):
        bypass_routed = _build_routed_experts(None, "model.layers.78.mlp.experts")
    bypass_model.add_module("experts", bypass_routed)
    with pytest.raises(ValueError, match="bypassed"):
        validate_online_fp8(bypass_model, bypass_cfg)


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_validate_online_fp8_linear_only_without_drafter_targets_is_noop(
    monkeypatch, kind
):
    """For Linear-only targets that do not select drafter layers, auditing with
    draft_model produces the exact same outcome, count, and log lines as without it.
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )

    targets = [
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.0.shared_proj",
    ]
    cfg = _make_quant_config(kind, (0, 1, 78), targets=targets)

    model = torch.nn.Module()
    target_q_b = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.0.self_attn.q_b_proj",
        disable_tp=True,
    )
    shared_proj = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.0.shared_proj",
        disable_tp=True,
    )
    unselected_target_lin = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.1.self_attn.q_b_proj",
        disable_tp=True,
    )
    model.add_module("q_b", target_q_b)
    model.add_module("shared_proj", shared_proj)
    model.add_module("unselected_lin", unselected_target_lin)

    for lin in (target_q_b, shared_proj):
        lin.weight.data.normal_(0.0, 0.1)
        lin.quant_method.process_weights_after_loading(lin)

    # Draft model carries unselected BF16 Linear and MoE modules plus a shared module.
    draft_q_b = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.78.self_attn.q_b_proj",
        disable_tp=True,
    )
    draft_other_lin = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.78.shared_proj",
        disable_tp=True,
    )
    draft_routed = _build_routed_experts(
        cfg, "model.layers.78.mlp.experts", local_experts=2
    )
    draft_runner = MoERunner(
        layer_name="model.layers.78.mlp.experts",
        moe_config=draft_routed.moe_config,
        router=SimpleNamespace(),
        routed_experts=draft_routed,
        gate=_Gate(4),
        shared_experts=_SharedMLP(),
        routed_scaling_factor=2.5,
    )
    draft = torch.nn.Module()
    draft.add_module("q_b", draft_q_b)
    draft.add_module("other_lin", draft_other_lin)
    draft.add_module("experts", draft_runner)
    draft.add_module("shared_proj", shared_proj)

    logs_no_draft: list[str] = []
    monkeypatch.setattr(
        online_fp8_mod,
        "logger",
        SimpleNamespace(
            info=lambda fmt, *args: logs_no_draft.append(fmt % args if args else fmt)
        ),
    )
    validate_online_fp8(model, cfg)

    logs_with_draft: list[str] = []
    monkeypatch.setattr(
        online_fp8_mod,
        "logger",
        SimpleNamespace(
            info=lambda fmt, *args: logs_with_draft.append(fmt % args if args else fmt)
        ),
    )
    validate_online_fp8(model, cfg, draft_model=draft)

    assert logs_with_draft == logs_no_draft
    assert len(logs_with_draft) == 3
    assert (
        logs_with_draft[-1]
        == "TPU online FP8 validated 2 runtime layers (2 Linear, 0 MoE)"
    )
    assert all("layers.78" not in line for line in logs_with_draft)


@pytest.mark.parametrize("kind", ["nvfp4", "fp8"])
def test_validate_online_fp8_linear_wildcard_audits_draft_model(monkeypatch, kind):
    """Wildcard Linear targets matching draft_model Linears processed during draft
    load pass validation and have their draft modules audited as well.
    """
    monkeypatch.setattr(
        "vllm.model_executor.layers.quantization.fp8."
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )

    cfg = _make_quant_config(
        kind, (0, 78), targets=["model.layers.*.self_attn.q_b_proj"]
    )

    model = torch.nn.Module()
    target_q_b = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.0.self_attn.q_b_proj",
        disable_tp=True,
    )
    target_other = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.0.shared_proj",
        disable_tp=True,
    )
    model.add_module("q_b", target_q_b)
    model.add_module("other", target_other)
    target_q_b.weight.data.normal_(0.0, 0.1)
    target_q_b.quant_method.process_weights_after_loading(target_q_b)

    # Draft load instantiates and processes a wildcard-matched Linear alongside
    # unselected BF16 Linear and MoE modules.
    draft_q_b = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.78.self_attn.q_b_proj",
        disable_tp=True,
    )
    draft_other = ReplicatedLinear(
        32,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=cfg,
        prefix="model.layers.78.shared_proj",
        disable_tp=True,
    )
    draft_routed = _build_routed_experts(
        cfg, "model.layers.78.mlp.experts", local_experts=2
    )
    draft_runner = MoERunner(
        layer_name="model.layers.78.mlp.experts",
        moe_config=draft_routed.moe_config,
        router=SimpleNamespace(),
        routed_experts=draft_routed,
        gate=_Gate(4),
        shared_experts=_SharedMLP(),
        routed_scaling_factor=2.5,
    )
    draft = torch.nn.Module()
    draft.add_module("q_b", draft_q_b)
    draft.add_module("other", draft_other)
    draft.add_module("experts", draft_runner)

    draft_q_b.weight.data.normal_(0.0, 0.1)
    draft_q_b.quant_method.process_weights_after_loading(draft_q_b)

    audit_logs: list[str] = []
    monkeypatch.setattr(
        online_fp8_mod,
        "logger",
        SimpleNamespace(
            info=lambda fmt, *args: audit_logs.append(fmt % args if args else fmt)
        ),
    )
    validate_online_fp8(model, cfg, draft_model=draft)

    assert len(audit_logs) == 3
    assert (
        audit_logs[-1] == "TPU online FP8 validated 2 runtime layers (2 Linear, 0 MoE)"
    )
    audited_modules = [
        json.loads(line.removeprefix("TPU online FP8: "))["module"]
        for line in audit_logs[:-1]
    ]
    assert audited_modules == [
        "model.layers.0.self_attn.q_b_proj",
        "model.layers.78.self_attn.q_b_proj",
    ]

    # Draft Linear is genuinely audited: corrupted runtime format is rejected.
    draft_q_b.weight_scale = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    with pytest.raises(ValueError, match="runtime format"):
        validate_online_fp8(model, cfg, draft_model=draft)


@pytest.mark.parametrize("policy_on", [False, True])
def test_moe_tp_decode_arming_on_tiny_target_and_drafter_model(monkeypatch, policy_on):
    """End-to-end arming on a tiny EP2xTP4 (8-rank) target + excluded drafter model."""
    torch.manual_seed(0)
    # Match the real NVFP4 serving recipe environment.
    monkeypatch.setenv("MOE_REQUANTIZE_BLOCK_SIZE", "512")
    monkeypatch.setenv("TPU_MOE_TP_DECODE_HBM_OPERANDS", "drafter")
    monkeypatch.setattr(envs, "TPU_MOE_SKIP_PADDED_TOKENS", True)

    local_experts = 4
    ep_size, tp_size = 2, 4
    n_ranks = ep_size * tp_size  # 8

    monkeypatch.setattr(
        pltpu,
        "get_tpu_info",
        lambda: pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1),
    )
    runners_by_name: dict[str, MoERunner] = {}
    monkeypatch.setattr(moe_runner_mod, "get_current_vllm_config", lambda: None)
    monkeypatch.setattr(
        moe_runner_mod, "register_layer_for_moe_forward_op", lambda config, layer: None
    )
    monkeypatch.setattr(
        moe_runner_mod, "get_layer_from_name", runners_by_name.__getitem__
    )
    monkeypatch.setattr(
        moe_runner_mod,
        "get_forward_context",
        lambda: SimpleNamespace(dp_metadata=None, all_moe_layers=None),
    )
    monkeypatch.setattr(moe_runner_mod, "is_forward_context_available", lambda: False)
    monkeypatch.setattr(moe_runner_mod, "tensor_model_parallel_all_reduce", lambda t: t)
    monkeypatch.setattr(
        parallel_state, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )

    padding = token_padding.TokenPaddingState.create(64, torch.device("cpu"))
    padding.update(4, 16)
    monkeypatch.setattr(token_padding, "_padding_state", padding)
    monkeypatch.setattr(
        moe_tp_decode,
        "_topology",
        lambda: moe_tp_decode._Topology(
            mesh=SimpleNamespace(size=n_ranks), mesh_index=0
        ),
    )
    monkeypatch.setattr(moe_tp_decode, "_gather", lambda obj: [obj])

    recorded_statics = []
    op_calls = []

    def _stub_op(x, *operands):
        op_calls.append((x, *operands))
        return (x.float() * 2.0).to(x.dtype)

    def _get_op(mesh, statics):
        recorded_statics.append(statics)
        return _stub_op

    monkeypatch.setattr(moe_tp_decode, "_get_op", _get_op)
    monkeypatch.setattr(moe_tp_decode, "_selftest", lambda topo, buckets, device: [])
    monkeypatch.setattr(moe_tp_decode, "_RANK_BUFFER", None)
    monkeypatch.setattr(moe_tp_decode, "_SELFTEST_DONE", False)

    nvfp4_cfg = _make_nvfp4_config(
        ["model.layers.1*"],
        targets=["model.layers.1.mlp.experts"] if policy_on else None,
    )

    # Target layer 0: non-excluded NVFP4 layer (must stay VllmNvfp4MoEMethod).
    target_moe_cfg, _ = _make_moe_config(
        local_experts=local_experts, ep_size=ep_size, tp_size=tp_size, use_ep=True
    )
    target_routed = RoutedExperts.__new__(RoutedExperts)
    torch.nn.Module.__init__(target_routed)
    target_routed.layer_name = "model.layers.0.mlp.experts"
    target_routed.moe_config = target_moe_cfg
    target_routed.global_num_experts = local_experts * ep_size
    target_routed.quant_method = nvfp4_cfg.get_quant_method(
        target_routed, "model.layers.0.mlp.experts"
    )
    assert type(target_routed.quant_method) is VllmNvfp4MoEMethod

    target_runner = MoERunner(
        layer_name="model.layers.0.mlp.experts",
        moe_config=target_moe_cfg,
        router=SimpleNamespace(),
        routed_experts=target_routed,
        gate=_Gate(local_experts * ep_size),
        shared_experts=_SharedMLP(),
        routed_scaling_factor=2.5,
    )
    runners_by_name[target_runner.layer_name] = target_runner
    target_model = torch.nn.Module()
    target_model.add_module("layer0", target_runner)

    # Drafter layer 1: checkpoint-excluded RoutedExperts loaded from BF16.
    draft_routed = _build_routed_experts(
        nvfp4_cfg,
        "model.layers.1.mlp.experts",
        local_experts=local_experts,
        ep_size=ep_size,
        tp_size=tp_size,
        use_ep=True,
    )
    draft_routed.scoring_func = "sigmoid"
    draft_routed.renormalize = True
    draft_routed.use_grouped_topk = True
    draft_routed.num_expert_group = 1
    draft_routed.topk_group = 1
    draft_routed.routed_scaling_factor = 1.0
    draft_routed.e_score_correction_bias = torch.nn.Parameter(
        torch.zeros(local_experts * ep_size, dtype=torch.float32), requires_grad=False
    )
    draft_routed.w13_weight.data.normal_(0.0, 0.02)
    draft_routed.w2_weight.data.normal_(0.0, 0.02)

    with (
        patch(
            "vllm_torchtpu.layers.adapter.quantization.fp8.prebuild_fused_moe_kernel"
        ),
        patch(
            "vllm_torchtpu.layers.adapter.quantization.unquantized.prebuild_fused_moe_kernel"
        ),
    ):
        draft_routed.quant_method.process_weights_after_loading(draft_routed)

    draft_shared = _SharedMLP()
    draft_runner = MoERunner(
        layer_name="model.layers.1.mlp.experts",
        moe_config=draft_routed.moe_config,
        router=SimpleNamespace(),
        routed_experts=draft_routed,
        gate=_Gate(local_experts * ep_size),
        shared_experts=draft_shared,
        routed_scaling_factor=2.5,
    )
    runners_by_name[draft_runner.layer_name] = draft_runner
    draft_model = torch.nn.Module()
    draft_model.add_module("layer1", draft_runner)

    validate_online_fp8(target_model, nvfp4_cfg, draft_model=draft_model)
    armed = moe_tp_decode.arm(target_model, draft_model)

    # Target NVFP4 layer is never touched or armed by FP8 moe_tp_decode.
    assert type(target_routed.quant_method) is VllmNvfp4MoEMethod
    assert not moe_tp_decode.moe_tp_decode_supported(target_runner._quant_method)

    if not policy_on:
        assert armed == 0
        assert type(draft_routed.quant_method) is VllmUnquantizedFusedMoEMethod
        assert draft_routed.w13_weight.dtype == torch.bfloat16
        assert not moe_tp_decode.moe_tp_decode_supported(draft_runner._quant_method)
    else:
        assert armed == 1
        assert type(draft_routed.quant_method) is VllmFp8MoEMethodTPU
        assert draft_routed.w13_weight.dtype == FP8_DTYPE
        assert draft_routed.w13_weight_scale_inv.shape == (
            local_experts,
            1,
            1,
            2 * INTER,
        )
        assert draft_routed.w2_weight_scale_inv.shape == (local_experts, 1, 1, H)
        assert moe_tp_decode.moe_tp_decode_supported(draft_runner._quant_method)
        assert len(recorded_statics) == 1
        assert recorded_statics[0].hbm_operands is True
        assert recorded_statics[0].use_ep is True

        x = torch.randn(16, H, dtype=torch.bfloat16)
        out = draft_runner(hidden_states=x, router_logits=x)
        assert out.shape == (16, H)
        assert len(op_calls) == 1
        assert draft_shared.calls == 0
