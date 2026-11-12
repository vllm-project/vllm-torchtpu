# SPDX-License-Identifier: Apache-2.0
"""HC profitability is independent of mandatory numerical/resource admission."""

import dataclasses
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

MEASURED = HcHardware("7x", 64 << 20, 128, 16, 8, 256)


@pytest.fixture
def boundaries(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    calls = []

    def mix(x, weight, down, up):
        calls.append("mix")
        return x[:, :2560], x

    def combine(x, block, gates):
        calls.append("combine")
        return x

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: MEASURED)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: mix)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: combine)
    return calls


def meta_module(**kwargs):
    with torch.device("meta"):
        return GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
            ),
            **kwargs,
        )


def route_view(module):
    """Pure tensor-metadata harness for a target TPU; never allocates a device.

    Production guards are executed unmodified, while dtype/policy/factory facts
    come from the real constructor. These are CPU policy tests, not TPU runs.
    """
    view = SimpleNamespace(
        use_mix=module.use_mix,
        use_combine=module.use_combine,
        hc_count=module.hc_count,
        hidden_size=module.hidden_size,
        prefill_full_tile=module.prefill_full_tile,
        prefill_combine_tile=module.prefill_combine_tile,
        _prefill_full_ops=module._prefill_full_ops,
        _prefill_combine_op=module._prefill_combine_op,
        _combine_token_tile=module._combine_token_tile,
        _decode_mix_op=None,
        _decode_combine_mix_op=None,
    )
    view._decode_fused = lambda _x: False
    view._residual = lambda h, n: (h, n)

    def weight_view(weight):
        return SimpleNamespace(
            weight=SimpleNamespace(
                dtype=weight.dtype, device=SimpleNamespace(type="tpu", index=0)
            )
        )

    if module.use_mix:
        view.hc_norm = weight_view(module.hc_norm.weight)
        view.input_mix_weight_down = weight_view(module.input_mix_weight_down.weight)
        view.input_mix_weight_up = weight_view(module.input_mix_weight_up.weight)
    if module.use_combine:
        view.block_inject_weight = weight_view(module.block_inject_weight.weight)
    return view


def route_inputs(tokens, shape=None, dtype=torch.bfloat16):
    shape = shape or (tokens, 10240)

    def metadata(shape):
        return SimpleNamespace(
            shape=shape,
            ndim=len(shape),
            dtype=dtype,
            device=SimpleNamespace(type="tpu", index=0),
        )

    return metadata(shape), metadata((*shape[:-1], 2560))


def mix_selected(module, x):
    return GatedResidual._use_prefill_mix(route_view(module), x)


def combine_selected(module, block, x, norm):
    return GatedResidual._use_prefill_combine(route_view(module), block, x, norm)


def test_default_auto_dispatch(boundaries):
    module = meta_module()
    assert module.prefill_full_tile == module.prefill_combine_tile == "auto"
    assert tuple(module._prefill_full_ops) == (256, 128)
    assert module._combine_token_tile == 128
    x, block = route_inputs(1024)
    assert mix_selected(module, x)
    assert combine_selected(module, block, x, x)


@pytest.mark.parametrize(
    "full,combine,expected",
    [
        (0, "auto", ["combine"]),
        ("auto", 0, ["mix"]),
        ("auto", None, ["mix", "combine"]),
        (0, None, []),
        (128, None, ["mix", "combine"]),
        (512, "auto", ["combine"]),
    ],
)
def test_independent_modes(full, combine, expected, boundaries):
    module = meta_module(prefill_full_tile=full, prefill_combine_tile=combine)
    x, block = route_inputs(1024)
    selected = []
    if mix_selected(module, x):
        selected.append("mix")
    if combine_selected(module, block, x, x):
        selected.append("combine")
    assert selected == expected


@pytest.mark.parametrize(
    "tokens",
    [
        1,
        32,
        128,
        256,
        512,
        640,
        768,
        1023,
        1024,
        1152,
        1536,
        2048,
        4096,
        8192,
        8320,
        127,
        129,
        257,
        1025,
    ],
)
def test_auto_minimum_threshold_and_full_tile_alignment(tokens, boundaries):
    module = meta_module()
    x, block = route_inputs(tokens)
    selected = tokens >= 1024 and tokens % 128 == 0
    assert mix_selected(module, x) is selected
    assert combine_selected(module, block, x, x) is selected


@pytest.mark.parametrize(
    "change,mix,combine",
    [
        ({"chip": "6e"}, True, True),
        ({"chip": "next-generation"}, True, True),
        ({"vmem_bytes": 16 << 20}, False, True),
        ({"vmem_bytes": 128 << 20}, True, True),
        ({"lanes": 256}, False, False),
        ({"bf16_rows": 256}, False, False),
        ({"bf16_rows": 8}, True, True),
        ({"fp32_rows": 16}, True, True),
        ({"mxu_columns": 128}, True, True),
    ],
)
def test_hardware_admission_uses_resources_and_layout(
    change, mix, combine, boundaries, monkeypatch
):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    facts = dataclasses.replace(MEASURED, **change)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: facts)
    module = meta_module()
    assert bool(module._prefill_full_ops) is mix
    assert (module._prefill_combine_op is not None) is combine


@pytest.mark.parametrize(
    "change,mix",
    [
        ({"hc_count": 3}, True),
        ({"hidden_size": 128}, True),
        ({"hc_lowrank": 128}, True),
        ({"hc_per_branch_norm": False}, False),
        ({"rms_norm_eps": 1e-5}, True),
    ],
)
def test_geometry_requires_only_resources_and_mix_algorithm(change, mix, boundaries):
    config = dict(hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True)
    config.update(change)
    with torch.device("meta"):
        module = GatedResidual(HyperConnectionConfig(**config))
    assert bool(module._prefill_full_ops) is mix
    assert module._prefill_combine_op is not None


def test_resource_admission_still_independent(boundaries, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import hc_resources

    monkeypatch.setattr(hc_resources, "supports_full_mix", lambda *a: False)
    module = meta_module()
    assert not module._prefill_full_ops
    assert module._prefill_combine_op is not None
    x, block = route_inputs(1024)
    assert not mix_selected(module, x)
    assert combine_selected(module, block, x, x)


def test_combine_only_does_not_require_mix_weights(boundaries):
    module = meta_module(use_mix=False)
    x, block = route_inputs(1024)
    assert not module._prefill_full_ops
    assert combine_selected(module, block, x, x)
    assert module._prefill_combine_op is not None


@pytest.mark.parametrize(
    "name,mix,combine",
    [
        ("hc_norm", False, True),
        ("input_mix_weight_down", False, True),
        ("input_mix_weight_up", False, True),
        ("block_inject_weight", True, False),
    ],
)
def test_relevant_weight_dtype_guards(name, mix, combine, boundaries):
    module = meta_module()
    getattr(module, name).to(dtype=torch.float32)
    x, block = route_inputs(1024)
    assert mix_selected(module, x) is mix
    assert combine_selected(module, block, x, x) is combine


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_input_dtype_fallback(dtype, boundaries):
    module = meta_module()
    x, block = route_inputs(1024, dtype=dtype)
    assert not mix_selected(module, x)
    assert not combine_selected(module, block, x, x)


@pytest.mark.parametrize("shape", [(10240,), (1024, 1, 10240)])
def test_rank_fallback(shape, boundaries):
    module = meta_module()
    x, block = route_inputs(1024, shape=shape)
    assert not mix_selected(module, x)
    assert not combine_selected(module, block, x, x)


def test_combine_individual_input_metadata(boundaries):
    module = meta_module()
    x, block = route_inputs(1024)
    rank3, _ = route_inputs(1024, shape=(1024, 1, 10240))
    norm32, _ = route_inputs(1024, dtype=torch.float32)
    one, one_block = route_inputs(1)
    assert not combine_selected(module, rank3, x, x)
    assert not combine_selected(module, block, x, rank3)
    assert not combine_selected(module, one_block, x, x)
    assert not combine_selected(module, block, x, one)
    assert not combine_selected(module, block, x, norm32)


@pytest.mark.parametrize(
    "value", [True, False, -1, -128, "128", "AUTO", "", 1.5, [], {}]
)
@pytest.mark.parametrize("component", ["full", "combine"])
def test_invalid_modes_rejected(value, component):
    kwargs = {
        "prefill_full_tile" if component == "full" else "prefill_combine_tile": value
    }
    with pytest.raises(ValueError, match="HC prefill mode"):
        GatedResidual(HyperConnectionConfig(), **kwargs)


def test_unaligned_forced_mix_rejected():
    with pytest.raises(ValueError, match="multiple of 128"):
        GatedResidual(HyperConnectionConfig(), prefill_full_tile=16)


def test_no_backend_initialization_and_frozen_unknown_facts(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    def forbidden():
        raise AssertionError("CPU construction attempted hardware discovery")

    monkeypatch.setattr(
        torch, "tpu", SimpleNamespace(is_initialized=lambda: False), raising=False
    )
    monkeypatch.setattr(vllm_adapter.pltpu, "get_tpu_info", forbidden)
    with torch.device("cpu"):
        module = GatedResidual(HyperConnectionConfig(hidden_size=128))
    assert module.prefill_full_tile == module.prefill_combine_tile == "auto"
    assert not module._prefill_full_ops
    assert module._prefill_combine_op is None
    # Moving a module does not refresh a captured policy; no implicit init/query.
    module.to(dtype=torch.float32)
    assert not module._prefill_full_ops
    assert module._prefill_combine_op is None


def test_auto_fallback_actual_math_and_no_fused_call(monkeypatch):
    from tests.models.vllm.test_qwen4_exp_math import hc_cuda_source_reference
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    def forbidden(*args):
        pytest.fail("CPU fallback reached fused HC")

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: MEASURED)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *a: forbidden)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", lambda *a: forbidden)
    with torch.device("cpu"):
        torch.random.default_generator.manual_seed(1961)
        module = GatedResidual(
            HyperConnectionConfig(
                hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
            )
        )
        x = torch.randn(8, 10240).bfloat16()
        block = torch.randn(8, 2560).bfloat16()
        expected = hc_cuda_source_reference(module, x, block)
        mixed, residual = module.mix(x)
        result = module.combine(block, residual)
    for actual, ref in zip((mixed, result), expected, strict=True):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["auto", 128])
def test_cpu_device_falls_back_even_after_hardware_capture(mode, boundaries):
    module = meta_module(prefill_full_tile=mode)
    with FakeTensorMode():
        x = torch.empty(1024, 10240, device="cpu", dtype=torch.bfloat16)
        block = torch.empty(1024, 2560, device="cpu", dtype=torch.bfloat16)
        assert not module._use_prefill_mix(x)
        assert not module._use_prefill_combine(block, x, x)


@pytest.mark.parametrize("mode", ["auto", 128])
@pytest.mark.parametrize(
    "component,operand",
    [
        ("mix", "input"),
        ("mix", "hc_norm"),
        ("mix", "input_mix_weight_down"),
        ("mix", "input_mix_weight_up"),
        ("combine", "input"),
        ("combine", "block"),
        ("combine", "norm"),
        ("combine", "block_inject_weight"),
    ],
)
@pytest.mark.parametrize(
    "attribute,value",
    [
        ("device", SimpleNamespace(type="cpu", index=0)),
        ("device", SimpleNamespace(type="tpu", index=1)),
        ("dtype", torch.float32),
        ("dtype", torch.float16),
    ],
)
def test_all_operand_device_and_dtype_guards_are_unconditional(
    mode, component, operand, attribute, value, boundaries
):
    module = meta_module(prefill_full_tile=mode)
    view = route_view(module)
    x, block = route_inputs(1024)
    norm, _ = route_inputs(1024)
    targets = {"input": x, "block": block, "norm": norm}
    target = targets[operand] if operand in targets else getattr(view, operand).weight
    if component == "mix":
        assert GatedResidual._use_prefill_mix(view, x)
        setattr(target, attribute, value)
        assert not GatedResidual._use_prefill_mix(view, x)
    else:
        assert GatedResidual._use_prefill_combine(view, block, x, norm)
        setattr(target, attribute, value)
        assert not GatedResidual._use_prefill_combine(view, block, x, norm)


def test_forced_tiles_override_only_minimum_tokens(boundaries):
    module = meta_module(prefill_full_tile=128, prefill_combine_tile=32)
    x, block = route_inputs(128)
    assert mix_selected(module, x)
    assert combine_selected(module, block, x, x)
    x, block = route_inputs(32)
    assert not mix_selected(module, x)
    assert combine_selected(module, block, x, x)


def test_combine_does_not_read_unused_norm_or_mix_fields(boundaries):
    with torch.device("meta"):
        module = GatedResidual(
            HyperConnectionConfig(
                hidden_size=128,
                hc_count=3,
                hc_lowrank=7,
                hc_per_branch_norm=False,
                rms_norm_eps=0.001,
            ),
            use_mix=False,
        )
    del module.hc_norm
    del module.config.hc_lowrank
    del module.config.hc_per_branch_norm
    assert not module._prefill_full_ops
    x, block = route_inputs(1024)
    assert combine_selected(module, block, x, x)


@pytest.mark.parametrize("eps", [1e-5, 0.001])
def test_full_mix_builder_receives_actual_epsilon(eps, boundaries, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    calls = []
    monkeypatch.setattr(
        vllm_adapter,
        "build_full_mix_op",
        lambda *args: calls.append(args) or object(),
    )
    with torch.device("meta"):
        module = GatedResidual(
            HyperConnectionConfig(
                hidden_size=128,
                hc_lowrank=64,
                hc_per_branch_norm=True,
                rms_norm_eps=eps,
            )
        )
    assert module._prefill_full_ops
    assert calls[0][2] == eps


def test_unaligned_forced_combine_rejected(boundaries):
    with pytest.raises(ValueError, match="combine tile must be a multiple of 16"):
        meta_module(prefill_combine_tile=7)


def test_fallback_warning_once_contains_reason_and_key(boundaries, monkeypatch, caplog):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    facts = dataclasses.replace(MEASURED, lanes=256)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: facts)
    for _ in range(3):
        module = meta_module()
    records = [r.getMessage() for r in caplog.records if "HC " in r.getMessage()]
    assert len(records) == 2
    for message in records:
        assert "unsupported resources/layout" in message
        assert "branches=4, hidden=2560" in message
        assert repr(facts) in message
        assert ("tiles=(256, 128)" if "full mix" in message else "tile=128") in message
    caplog.clear()
    x, block = route_inputs(1024)
    for _ in range(3):
        assert not mix_selected(module, x)
        assert not combine_selected(module, block, x, x)
    assert not caplog.records


def test_admitted_policy_and_disabled_modes_do_not_warn(boundaries, caplog):
    meta_module()
    meta_module(prefill_full_tile=0)
    assert not [r for r in caplog.records if "HC " in r.getMessage()]


@pytest.mark.parametrize("tokens,expected", [(1024, 256), (1152, 128), (1536, 256)])
def test_auto_mix_dispatches_preferred_or_token_aligned_candidate(
    tokens, expected, boundaries, monkeypatch
):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    built, called = [], []
    mixed, normalized = object(), object()

    def build(branches, hidden, eps, tile, hardware):
        built.append(tile)

        def op(*operands):
            called.append(tile)
            return mixed, normalized

        return op

    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", build)
    module = meta_module()
    assert built == [256, 128]
    assert module._combine_token_tile == 128
    view = route_view(module)
    view._use_prefill_mix = MethodType(GatedResidual._use_prefill_mix, view)
    x, _ = route_inputs(tokens)
    actual, residuals = GatedResidual.mix(view, x)
    assert actual is mixed and residuals == (x, normalized)
    assert called == [expected]
    assert built == [256, 128]


def test_auto_mix_128_resource_fallback_keeps_combine_128(boundaries, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    hardware = dataclasses.replace(MEASURED, vmem_bytes=48 << 20)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: hardware)
    module = meta_module()
    assert tuple(module._prefill_full_ops) == (128,)
    assert module._combine_token_tile == 128
    x, block = route_inputs(1024)
    assert mix_selected(module, x)
    assert combine_selected(module, block, x, x)


@pytest.mark.parametrize(
    "full,combine,tokens,mix_selected_expected,combine_selected_expected",
    [(256, "auto", 1152, False, True), ("auto", 256, 1152, True, False)],
)
def test_explicit_component_has_no_automatic_tile_fallback(
    full, combine, tokens, mix_selected_expected, combine_selected_expected, boundaries
):
    module = meta_module(prefill_full_tile=full, prefill_combine_tile=combine)
    x, block = route_inputs(tokens)
    assert mix_selected(module, x) is mix_selected_expected
    assert combine_selected(module, block, x, x) is combine_selected_expected
    if full == 256:
        assert tuple(module._prefill_full_ops) == (256,)


@pytest.mark.parametrize(
    "tokens,tiles,expected",
    [
        (1024, (256, 128), 256),
        (1152, (256, 128), 128),
        (1024, (128,), 128),
        (1152, (256,), None),
        (1025, (256, 128), None),
        (64, (256, 128), None),
        (0, (256, 128), None),
        (1024, (), None),
    ],
)
def test_mix_tile_selection_is_pure(tokens, tiles, expected):
    from vllm_torchtpu.kernels.qwen4_exp.hc_prefill_policy import select_mix_tile

    assert select_mix_tile(tokens, tiles) == expected
    assert select_mix_tile(tokens, tiles) == expected


def test_same_tpu_bf16_checks_reference_and_all_operands_without_mutation():
    from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import _same_tpu_bf16

    x, block = route_inputs(1024)
    norm, _ = route_inputs(1024)
    before = [dict(vars(t)) for t in (x, block, norm)]
    assert _same_tpu_bf16(x)
    assert _same_tpu_bf16(x, block, norm)
    assert before == [vars(t) for t in (x, block, norm)]
    norm.dtype = torch.float32
    assert not _same_tpu_bf16(x, block, norm)
    norm.dtype = torch.bfloat16
    norm.device = SimpleNamespace(type="tpu", index=1)
    assert not _same_tpu_bf16(x, block, norm)
    x.device = SimpleNamespace(type="cpu", index=0)
    assert not _same_tpu_bf16(x)


@pytest.mark.parametrize("use_mix", [False, True])
@pytest.mark.parametrize("use_combine", [False, True])
@pytest.mark.parametrize("positional", [False, True])
def test_declared_component_flags_preserve_enabled_and_disabled_execution(
    use_mix, use_combine, positional, boundaries
):
    config = HyperConnectionConfig(
        hidden_size=128, hc_count=3, hc_lowrank=8, hc_per_branch_norm=True
    )
    with torch.device("cpu"):
        if positional:
            module = GatedResidual(
                config, 2, "attn", use_mix, use_combine, prefill_full_tile=128
            )
        else:
            module = GatedResidual(
                config,
                layer_idx=2,
                role="attn",
                use_mix=use_mix,
                use_combine=use_combine,
                prefill_full_tile=128,
            )
        x = torch.ones(128, 384, dtype=torch.bfloat16)
        block = torch.ones(128, 128, dtype=torch.bfloat16)
    assert module.use_mix is use_mix
    assert module.use_combine is use_combine
    assert bool(module._prefill_full_ops) is use_mix
    assert (module._prefill_combine_op is not None) is use_combine
    if use_mix:
        mixed, residuals = module.mix(x)
        assert mixed.shape == block.shape and mixed.dtype == x.dtype
        assert torch.isfinite(mixed).all()
    else:
        assert not module._use_prefill_mix(object())
        with pytest.raises(RuntimeError, match="mix was disabled"):
            module.mix(x)
        residuals = (x, x)
    if use_combine:
        result = module.combine(block, residuals)
        assert result.shape == x.shape and result.dtype == x.dtype
        assert torch.isfinite(result).all()
    else:
        assert not module._use_prefill_combine(object(), object(), object())
        with pytest.raises(RuntimeError, match="combine was disabled"):
            module.combine(block, residuals)


def test_positional_constructor_forwards_metadata_and_flags(boundaries, monkeypatch):
    from vllm.models.qwen4_exp.common import hyperconnection as common_hc

    calls = []
    original = common_hc.GatedResidual.__init__

    def record(self, config, layer_idx=None, role=None, use_mix=True, use_combine=True):
        calls.append((config, layer_idx, role, use_mix, use_combine))
        original(self, config, layer_idx, role, use_mix, use_combine)

    monkeypatch.setattr(common_hc.GatedResidual, "__init__", record)
    config = HyperConnectionConfig(hidden_size=128, hc_lowrank=8)
    with torch.device("meta"):
        module = GatedResidual(config, 7, "mlp", False, True)
    assert calls == [(config, 7, "mlp", False, True)]
    assert not module.use_mix and module.use_combine


@pytest.mark.parametrize("mode", ["auto", 128])
def test_both_disabled_components_do_not_discover_hardware(mode, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    def forbidden(*args):
        pytest.fail("both-disabled HC attempted hardware discovery or op creation")

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", forbidden)
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", forbidden)
    monkeypatch.setattr(vllm_adapter, "build_gated_residual_op", forbidden)
    with torch.device("meta"):
        module = GatedResidual(
            HyperConnectionConfig(hidden_size=128),
            None,
            None,
            False,
            False,
            prefill_full_tile=mode,
        )
    assert not module.use_mix and not module.use_combine
    assert not module._prefill_full_ops and module._prefill_combine_op is None
    assert not module._use_prefill_mix(object())
    assert not module._use_prefill_combine(object(), object(), object())


def test_tpu_8i_admits_m256_full_mix_and_m128_residual(boundaries, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    tpu_8i = HcHardware("8i", 192 << 20, 128, 16, 8, 256)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: tpu_8i)
    module = meta_module()
    assert tuple(module._prefill_full_ops) == (256, 128)
    assert module._combine_token_tile == 128
    assert module._prefill_combine_op is not None
    assert module.sp_tile_alignment == 256
    x, block = route_inputs(4096)
    inj = SimpleNamespace(
        shape=(4096, 4),
        ndim=2,
        dtype=torch.float32,
        device=SimpleNamespace(type="tpu", index=0),
    )
    assert mix_selected(module, x)
    assert combine_selected(module, block, x, inj)


def test_transpose_up_weight_rebuilds_full_and_combine_full_ops(
    boundaries, monkeypatch
):
    from vllm_torchtpu import envs
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    tpu_8i = HcHardware("8i", 192 << 20, 128, 16, 8, 256)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: tpu_8i)
    monkeypatch.setattr(envs, "VLLM_TPU_HC_PREFILL_COMBINE_MIX", True)
    monkeypatch.setattr(envs, "VLLM_TPU_HC_PREFILL_UP_T", True)
    monkeypatch.setattr(envs, "VLLM_TPU_HC_PREFILL_INJECT", True)

    full_calls = []
    combine_calls = []
    monkeypatch.setattr(
        vllm_adapter,
        "build_full_mix_op",
        lambda *a, **kw: full_calls.append((a, kw)) or object(),
    )
    monkeypatch.setattr(
        vllm_adapter,
        "build_combine_full_mix_op",
        lambda *a, **kw: combine_calls.append((a, kw)) or object(),
    )

    module = meta_module()
    assert tuple(module._prefill_full_ops) == (256, 128)
    assert tuple(module._prefill_combine_full_ops) == (256, 128)
    assert all(kw["up_transposed"] is False for _, kw in full_calls)
    assert all(kw["up_transposed"] is False for _, kw in combine_calls)

    full_calls.clear()
    combine_calls.clear()
    module.transpose_up_weight()
    assert module._up_transposed
    assert tuple(module._prefill_full_ops) == (256, 128)
    assert tuple(module._prefill_combine_full_ops) == (256, 128)
    assert all(
        kw == {"up_t": True, "inject": True, "up_transposed": True}
        for _, kw in full_calls
    )
    assert all(
        kw == {"up_t": True, "inject": True, "up_transposed": True}
        for _, kw in combine_calls
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("tokens", [16, 32, 128, 256, 512, 1024, 4096])
def test_tpu_8i_e2e_fused_decode_and_prefill_hc_match_torchtpu(tokens, monkeypatch):
    import numpy as np
    from torch_tpu._internal import sync

    from vllm_torchtpu import envs

    # Initialize TPU runtime so current_hc_hardware() sees physical TPU 8i
    # (8i, 192 MiB).
    _ = torch.zeros(1, device="tpu")

    def make_pair(fused: bool):
        monkeypatch.setattr(envs, "TPU_QWEN4_HC_DECODE_FUSED", fused)
        tile = "auto" if fused else 0
        torch.random.default_generator.manual_seed(77)
        cfg = HyperConnectionConfig(
            hidden_size=2560,
            hc_count=4,
            hc_lowrank=320,
            params_dtype=torch.bfloat16,
            hc_per_branch_norm=True,
        )
        a = GatedResidual(cfg, prefill_full_tile=tile).to("tpu")
        m = GatedResidual(cfg, prefill_full_tile=tile).to("tpu")
        with torch.no_grad():
            for mod, s in ((a, 101), (m, 202)):
                g = torch.Generator(device="cpu").manual_seed(s)
                mod.hc_norm.weight.copy_(
                    torch.randn(
                        mod.hc_norm.weight.shape, generator=g, dtype=torch.bfloat16
                    )
                    * 0.2
                )
                mod.input_mix_weight_down.weight.copy_(
                    torch.randn(
                        mod.input_mix_weight_down.weight.shape,
                        generator=g,
                        dtype=torch.bfloat16,
                    )
                    * 0.02
                )
                mod.input_mix_weight_up.weight.copy_(
                    torch.randn(
                        mod.input_mix_weight_up.weight.shape,
                        generator=g,
                        dtype=torch.bfloat16,
                    )
                    * 0.05
                )
                mod.block_inject_weight.weight.copy_(
                    torch.randn(
                        mod.block_inject_weight.weight.shape,
                        generator=g,
                        dtype=torch.bfloat16,
                    )
                    * 0.02
                )
        return a, m

    ref_a, ref_m = make_pair(False)
    fus_a, fus_m = make_pair(True)
    fus_a.transpose_up_weight()
    fus_m.transpose_up_weight()
    assert fus_a._up_transposed and fus_m._up_transposed
    assert tuple(fus_a.input_mix_weight_up.weight.shape) == (320, 10240)
    assert tuple(fus_a._prefill_full_ops) == (256, 128)
    assert fus_a._prefill_combine_op is not None
    assert fus_a._decode_mix_op is not None
    assert fus_m._decode_combine_mix_op is not None

    g_in = torch.Generator(device="cpu").manual_seed(500 + tokens)
    hyper = torch.randn((tokens, 10240), generator=g_in, dtype=torch.bfloat16).to("tpu")
    attn_out = torch.randn((tokens, 2560), generator=g_in, dtype=torch.bfloat16).to(
        "tpu"
    )
    mlp_out = torch.randn((tokens, 2560), generator=g_in, dtype=torch.bfloat16).to(
        "tpu"
    )

    if tokens <= 512:
        assert fus_a._decode_fused(hyper)
        assert not fus_a._use_prefill_mix(hyper)
    else:
        assert not fus_a._decode_fused(hyper)
        assert fus_a._use_prefill_mix(hyper)

    with torch.inference_mode():
        rb1, rr1 = ref_a.mix(hyper)
        rb2, rr2 = ref_a.combine_mix(attn_out, rr1, ref_m)
        rout = ref_m.combine(mlp_out, rr2)

        fb1, fr1 = fus_a.mix(hyper)
        if tokens >= 1024:
            assert fus_m._use_prefill_combine(mlp_out, hyper, fr1[1])
        fb2, fr2 = fus_a.combine_mix(attn_out, fr1, fus_m)
        fout = fus_m.combine(mlp_out, fr2)

    sync.synchronize([rb1, rb2, rout, fb1, fb2, fout], wait=True)
    for name, actual, expected in (
        ("attn_mix", fb1, rb1),
        ("mlp_mix", fb2, rb2),
        ("layer_out", fout, rout),
    ):
        a = actual.cpu().float().numpy()
        e = expected.cpu().float().numpy()
        assert np.isfinite(a).all(), name
        nrmse = np.linalg.norm(a - e) / np.linalg.norm(e)
        cos = np.vdot(a.ravel(), e.ravel()) / (np.linalg.norm(a) * np.linalg.norm(e))
        assert nrmse <= 2.5e-3, (tokens, name, nrmse)
        assert cos >= 0.99999, (tokens, name, cos)


@pytest.mark.tpu_test
def test_tpu_8i_shape_variants_partitions_16_to_8192_without_128_constant_fold(
    monkeypatch, tmp_path
):
    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context

    from tests.compilation import test_shape_variants as tsv
    from vllm_torchtpu.compilation import shape_variants, tpu_compiler
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    buckets_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    monkeypatch.setattr(tsv, "BUCKETS", list(buckets_list))
    tpu_8i = HcHardware("8i", 192 << 20, 128, 16, 8, 256)
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: tpu_8i)

    @support_torch_compile(
        dynamic_arg_dims={"input_ids": 0, "positions": 0, "valid_tokens": 0}
    )
    class _Backbone(torch.nn.Module):
        def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
            super().__init__()
            cfg = HyperConnectionConfig(
                hidden_size=2560,
                hc_count=4,
                hc_lowrank=320,
                params_dtype=torch.bfloat16,
                hc_per_branch_norm=True,
            )
            self.embed = torch.nn.Embedding(256, 2560, dtype=torch.bfloat16)
            self.attn_hc = GatedResidual(cfg, prefill_full_tile="auto")
            self.qkvz = torch.nn.Linear(2560, 4096, bias=False, dtype=torch.bfloat16)
            self.mlp_hc = GatedResidual(cfg, prefill_full_tile="auto")
            self.mlp_proj = torch.nn.Linear(
                2560, 2560, bias=False, dtype=torch.bfloat16
            )
            self.attn_hc.transpose_up_weight()
            self.mlp_hc.transpose_up_weight()

        def forward(self, input_ids, positions, *, valid_tokens):
            h = self.embed(input_ids).repeat(1, 4)
            block, res = self.attn_hc.mix(h)
            qkv, z = self.qkvz(block).split([2560, 1536], dim=-1)
            z = z.reshape(z.size(0), -1, 128)
            attn_out = qkv + z.flatten(1).repeat(1, 2)[:, :2560]
            attn_out = (
                attn_out
                + positions.to(torch.int64).unsqueeze(-1).to(torch.bfloat16) * 0.0
            )
            block2, res2 = self.attn_hc.combine_mix(attn_out, res, self.mlp_hc)
            mlp_out = torch.where(valid_tokens.unsqueeze(-1), self.mlp_proj(block2), 0)
            return self.mlp_hc.combine(mlp_out, res2)

    _ = torch.zeros(1, device="tpu")
    orig_hash = tpu_compiler.compute_tpu_compilation_hash
    orig_backend = getattr(tpu_compiler, "_tpu_backend", None)
    orig_deser = tpu_compiler._deserialize_entry
    try:
        vllm_cfg, fake_backend = tsv.prepare(str(tmp_path))
        with set_current_vllm_config(vllm_cfg):
            model = _Backbone(vllm_config=vllm_cfg).to("tpu")
        retraces = []
        with shape_variants.warmup() as buckets:
            for n in buckets_list:
                if buckets.needs_retrace(n):
                    retraces.append(n)
                    shape_variants.retrace(model, vllm_cfg, n)
                with set_forward_context(None, vllm_cfg):
                    _ = model(
                        torch.zeros(n, dtype=torch.int32, device="tpu"),
                        torch.zeros(n, dtype=torch.int32, device="tpu"),
                        valid_tokens=torch.ones(n, dtype=torch.bool, device="tpu"),
                    )
            assert not buckets.refused
        assert retraces == [256, 1024]
        assert len(fake_backend.compiled) == 10
    finally:
        tpu_compiler.compute_tpu_compilation_hash = orig_hash
        tpu_compiler._tpu_backend = orig_backend
        tpu_compiler._deserialize_entry = orig_deser
