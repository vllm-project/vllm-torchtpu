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
"""Checkpoint math contracts and TPU bridge checks with hardware markers."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionBase,
    HyperConnectionConfig,
)

TPU = pytest.mark.tpu_test


@pytest.fixture(autouse=True)
def cpu_default(request):
    # Repository device fixture defaults to TPU; CPU cases intentionally never
    # request it. Explicit factories also protect against a caller default.
    if request.node.get_closest_marker("tpu_test"):
        # Independent module cases must not exhaust shared wrapper variants.
        torch.compiler.reset()
        # Worker initialization precedes HC construction in native execution.
        torch.ones(1, device="tpu").cpu()
        yield
    else:
        with torch.device("cpu"):
            yield


def hc_fixture(per_branch, tokens=3, branches=4, hidden=2560, full_tile=0):
    # Seed only CPU: torch.manual_seed also initializes registered accelerators.
    torch.random.default_generator.manual_seed(383)
    config = HyperConnectionConfig(
        hidden_size=hidden,
        hc_count=branches,
        hc_lowrank=320,
        hc_per_branch_norm=per_branch,
    )
    module = GatedResidual(config, prefill_full_tile=full_tile)
    with torch.no_grad():
        module.hc_norm.weight.copy_(torch.randn_like(module.hc_norm.weight) * 0.1)
    x = (
        torch.randn(tokens, branches, hidden, device="cpu")
        + torch.arange(branches, device="cpu")[None, :, None]
    ).bfloat16()
    block = torch.randn(tokens, hidden, device="cpu").bfloat16()
    return module, x.flatten(1), block


def hc_cuda_source_reference(module, flat, block):
    """Reference the BF16 load/store boundaries in NVIDIA HC kernels."""
    branches, hidden = module.hc_count, module.hidden_size
    x = flat.reshape(-1, branches, hidden).float()
    weight = module.hc_norm.weight.float().reshape(-1, hidden)
    norm = (
        (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight))
        .bfloat16()
        .flatten(1)
    )
    down = F.linear(norm, module.input_mix_weight_down.weight)
    silu = F.silu(down.float() / branches).bfloat16()
    gate = F.linear(silu, module.input_mix_weight_up.weight)
    gate = torch.sigmoid(gate.float()).reshape(-1, branches, hidden)
    mixed = (gate * norm.float().reshape(-1, branches, hidden)).mean(1).bfloat16()
    inject = F.linear(norm, module.block_inject_weight.weight)
    inject = 2 * torch.sigmoid(inject.float() / branches)
    output = (
        (
            flat.float().reshape(-1, branches, hidden)
            + block.float()[:, None] * inject[:, :, None]
        )
        .bfloat16()
        .flatten(1)
    )
    return mixed, output


def hc_legacy_common_bf16_reference(module, flat, block):
    """The former common-PyTorch contract, kept as a counterexample."""
    x = flat.reshape(-1, 4, 2560).float()
    weight = module.hc_norm.weight.float().reshape(-1, 2560)
    norm = (
        (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight))
        .bfloat16()
        .flatten(1)
    )
    down = F.linear(norm, module.input_mix_weight_down.weight)
    gate = F.silu(down / 4)
    gate = torch.sigmoid(F.linear(gate, module.input_mix_weight_up.weight))
    mixed = (gate.reshape(-1, 4, 2560) * norm.reshape(-1, 4, 2560)).mean(1)
    inject = 2 * torch.sigmoid(F.linear(norm, module.block_inject_weight.weight) / 4)
    output = (flat.reshape(-1, 4, 2560) + block[:, None] * inject[:, :, None]).flatten(
        1
    )
    return mixed, output


@pytest.mark.parametrize("per_branch", [False, True])
def test_hc_checkpoint_groups_and_names(per_branch):
    module, x, block = hc_fixture(per_branch)
    expected = hc_cuda_source_reference(module, x, block)
    mixed, residual = module.mix(x)
    torch.testing.assert_close(mixed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        module.combine(block, residual), expected[1], rtol=0, atol=0
    )
    assert set(module.state_dict()) == {
        "hc_norm.weight",
        "input_mix_weight_down.weight",
        "input_mix_weight_up.weight",
        "block_inject_weight.weight",
    }
    assert module.hc_norm.weight.numel() == (10240 if per_branch else 2560)
    numbered = torch.arange(10240).float().reshape(1, 4, 2560)
    assert not torch.equal(numbered.mean(1), numbered.reshape(1, 2560, 4).mean(2))
    pooled, _ = HyperConnectionBase(module.config).mix(numbered.flatten(1))
    torch.testing.assert_close(pooled, numbered.mean(1), rtol=0, atol=0)


def test_hc_gate_mix_matches_cuda_fp32_rounding_at_real_geometry():
    """Catch BF16 rounding after gate sigmoid or gate-times-norm."""
    module = GatedResidual(
        HyperConnectionConfig(
            hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
        ),
        use_combine=False,
    )
    with torch.no_grad():
        module.hc_norm.weight.fill_(0.25)
        module.hc_norm.weight[0] = 0
        module.input_mix_weight_down.weight.zero_()
        module.input_mix_weight_down.weight[0, 0] = 24
        module.input_mix_weight_up.weight.zero_()
        module.input_mix_weight_up.weight[:, 0] = -1

    hyper_input = torch.ones((1, 4 * 2560), dtype=torch.bfloat16, device="cpu")
    actual = module.mix(hyper_input)[0][0, 1]

    # For each of four streams: BF16 gate=-6, BF16 norm=1.25. The CUDA
    # kernel computes sigmoid(-6)*1.25 and the HC average in FP32, then
    # performs its only output cast. The exact final BF16 value is below.
    expected = torch.tensor(0.0030975341796875, dtype=torch.bfloat16, device="cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hc_combine_matches_cuda_fp32_rounding_at_real_geometry():
    """Catch BF16 rounding in injection sigmoid, product, or residual add."""
    module = GatedResidual(
        HyperConnectionConfig(
            hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
        ),
        use_mix=False,
    )
    with torch.no_grad():
        module.block_inject_weight.weight.zero_()
        module.block_inject_weight.weight[:, 0] = -8

    hyper_input = torch.full((1, 4 * 2560), -1, dtype=torch.bfloat16, device="cpu")
    hyper_input_normed = torch.zeros_like(hyper_input, device="cpu")
    hyper_input_normed[:, 0] = 1
    block_output = torch.full((1, 2560), -3.125, dtype=torch.bfloat16, device="cpu")
    actual = module.combine(block_output, (hyper_input, hyper_input_normed))[0, 0]

    # BF16 injection=-8. CUDA evaluates 2*sigmoid(-8/4), then
    # -1 + -3.125*injection in FP32 and casts once to BF16.
    expected = torch.tensor(-1.7421875, dtype=torch.bfloat16, device="cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hc_legacy_common_bf16_rounding_is_a_distinct_contract():
    """Document why the former exact common-BF16 fixture is not CUDA HC."""
    gate = torch.tensor(-6, dtype=torch.bfloat16, device="cpu")
    norm = torch.tensor(1.25, dtype=torch.bfloat16, device="cpu")
    legacy_gate_mix = (torch.sigmoid(gate) * norm).to(torch.bfloat16)
    assert legacy_gate_mix.item() == 0.003082275390625
    assert legacy_gate_mix.item() != 0.0030975341796875

    injection = torch.tensor(-8, dtype=torch.bfloat16, device="cpu")
    block = torch.tensor(-3.125, dtype=torch.bfloat16, device="cpu")
    residual = torch.tensor(-1, dtype=torch.bfloat16, device="cpu")
    legacy_combine = residual + block * (2 * torch.sigmoid(injection / 4))
    assert legacy_combine.item() == -1.75
    assert legacy_combine.item() != -1.7421875

    module, flat, block_output = hc_fixture(False)
    legacy = hc_legacy_common_bf16_reference(module, flat, block_output)
    cuda = hc_cuda_source_reference(module, flat, block_output)
    assert any(not torch.equal(old, new) for old, new in zip(legacy, cuda))


def _np(tensor):
    return np.asarray(tensor.detach().float().numpy(), np.float32)


def assert_metrics(actual, expected, threshold=0.01):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    nrmse = np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-8)
    cosine = np.vdot(a.ravel(), e.ravel()) / max(
        np.linalg.norm(a) * np.linalg.norm(e), 1e-30
    )
    assert nrmse <= threshold, (nrmse, threshold)
    assert cosine >= 0.999, cosine


@TPU
@pytest.mark.parametrize("per_branch", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_cuda_source_rounding(per_branch, compiled):
    module, x, block = hc_fixture(per_branch)
    expected = hc_cuda_source_reference(module, x, block)
    module = module.to("tpu")

    def chain(x, block):
        mixed, residual = module.mix(x)
        return mixed, module.combine(block, residual)

    run = (
        torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else chain
    )
    with torch.inference_mode():
        actual = run(x.to("tpu"), block.to("tpu"))
    for a, e in zip(actual, expected):
        assert_metrics(_np(a.cpu()), _np(e))


@TPU
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_negative_silu_preserves_mix_gate(compiled):
    module = GatedResidual(
        HyperConnectionConfig(
            hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
        ),
        use_combine=False,
    )
    with torch.no_grad():
        module.hc_norm.weight.zero_()
        module.input_mix_weight_down.weight.zero_()
        module.input_mix_weight_down.weight[0, 0] = -33
        module.input_mix_weight_up.weight.zero_()
        module.input_mix_weight_up.weight[:, 0] = 100
    # Normalized ones round back to BF16 ones. The low-rank gate is -33/4;
    # SiLU(-8.25) rounds to -0.0021514892578125, then the BF16 up projection
    # rounds to -0.21484375. CUDA-style FP32 sigmoid/multiply still rounds the
    # final mix to 0.447265625. Premature BF16 tanh rounding instead makes
    # SiLU zero and the final mix 0.5 (about 12% error).
    expected = np.full((43, 2560), 0.447265625, dtype=np.float32)
    module = module.to("tpu")

    def mix(x):
        return module.mix(x)[0]

    run = (
        torch.compile(mix, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else mix
    )
    with torch.inference_mode():
        actual = run(torch.ones(43, 10240, dtype=torch.bfloat16, device="tpu"))
    assert_metrics(_np(actual.cpu()), expected)


@pytest.mark.parametrize("branches", [3, 4])
def test_hc_silu_divides_in_fp32(branches):
    module, x, block = hc_fixture(True, branches=branches, hidden=128)
    expected = hc_cuda_source_reference(module, x, block)
    mixed, residual = module.mix(x)
    torch.testing.assert_close(mixed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        module.combine(block, residual), expected[1], rtol=0, atol=0
    )


@pytest.mark.parametrize("tile", [-128, 64, 129])
def test_hc_rejects_invalid_full_tile(tile):
    with pytest.raises(ValueError, match="multiple of 128"):
        hc_fixture(True, full_tile=tile)


@pytest.mark.parametrize(
    "tokens,hidden,dtype,full_tile",
    [
        (256, 128, torch.bfloat16, 128),
        (128, 128, torch.bfloat16, 128),
        (384, 128, torch.bfloat16, 128),
        (257, 128, torch.bfloat16, 128),
        (256, 129, torch.bfloat16, 128),
        (256, 128, torch.float32, 128),
        (256, 128, torch.bfloat16, 0),
    ],
)
def test_hc_forced_cpu_device_falls_back(tokens, hidden, dtype, full_tile, monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware

    # Static mocked layout: this test checks dispatch, not TPU arithmetic.
    monkeypatch.setattr(
        vllm_adapter,
        "current_hc_hardware",
        lambda: HcHardware("mock-layout", 64 << 20, 128, 16, 8, 256),
    )
    module, x, block = hc_fixture(
        True, tokens=tokens, hidden=hidden, full_tile=full_tile
    )
    module, x, block = module.to(dtype), x.to(dtype), block.to(dtype)
    calls = []

    def full(hyper_input, weight, down, up):
        calls.append("mix")
        return hyper_input[:, :hidden], hyper_input

    def combine(hyper_input, block_output, injection):
        calls.append("combine")
        return hyper_input

    if module._prefill_full_ops:
        module._prefill_full_ops = {tile: full for tile in module._prefill_full_ops}
    if module._prefill_combine_op is not None:
        module._prefill_combine_op = combine
    mixed, residuals = module.mix(x)
    result = module.combine(block, residuals)
    assert calls == []
    assert mixed.shape == (tokens, hidden)
    assert result.shape == x.shape
    assert mixed.dtype == result.dtype == dtype


def hc_rank_fixture(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware

    def forbidden(*args):
        pytest.fail("rank-ineligible input reached a fused HC callable")

    # Exercise real admission/caller code with CPU-only boundary sentinels.
    monkeypatch.setattr(
        vllm_adapter,
        "current_hc_hardware",
        lambda: HcHardware("mock-layout", 64 << 20, 128, 16, 8, 256),
    )
    monkeypatch.setattr(vllm_adapter, "build_full_mix_op", lambda *args: forbidden)
    monkeypatch.setattr(
        vllm_adapter, "build_gated_residual_op", lambda *args: forbidden
    )
    module, _, _ = hc_fixture(True, hidden=128, full_tile=128)
    assert module._prefill_full_ops == {128: forbidden}
    assert module._prefill_combine_op is forbidden
    return module


@pytest.mark.parametrize("batch_shape", [(), (128, 2), (128, 1, 2)])
def test_hc_mix_rank_fallback_matches_reference(batch_shape, monkeypatch):
    module = hc_rank_fixture(monkeypatch)
    branches, hidden = module.hc_count, module.hidden_size
    x = torch.randn(*batch_shape, branches * hidden).bfloat16()
    streams = x.float().unflatten(-1, (branches, hidden))
    weight = module.hc_norm.weight.float().reshape(branches, hidden)
    norm = (
        (
            streams
            * torch.rsqrt(streams.square().mean(-1, keepdim=True) + 1e-6)
            * (1 + weight)
        )
        .bfloat16()
        .flatten(-2)
    )
    down = F.linear(norm, module.input_mix_weight_down.weight)
    gate = F.linear(
        F.silu(down.float() / branches).bfloat16(),
        module.input_mix_weight_up.weight,
    )
    expected = (
        (
            torch.sigmoid(gate.float()).unflatten(-1, (branches, hidden))
            * norm.float().unflatten(-1, (branches, hidden))
        )
        .mean(-2)
        .bfloat16()
    )

    actual, (residual, actual_norm) = module.mix(x)
    assert residual is x
    assert torch.isfinite(actual).all() and torch.isfinite(actual_norm).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_norm, norm, rtol=0, atol=0)


@pytest.mark.parametrize(
    "residual_batch,block_batch,norm_batch",
    [
        ((128, 2), (128, 2), (128, 2)),
        ((128, 1, 2), (128, 1, 2), (128, 1, 2)),
        ((128, 2), (2,), (2,)),
        ((128,), (128, 1), (128,)),
        ((128,), (128,), (128, 1)),
        ((), (128,), (128,)),
        ((128,), (), (128,)),
        ((128,), (128,), ()),
    ],
    ids=[
        "all-rank3",
        "all-rank4",
        "residual-rank3",
        "block-rank3",
        "norm-rank3",
        "residual-rank1",
        "block-rank1",
        "norm-rank1",
    ],
)
def test_hc_combine_rank_fallback_matches_reference(
    residual_batch, block_batch, norm_batch, monkeypatch
):
    module = hc_rank_fixture(monkeypatch)
    branches, hidden = module.hc_count, module.hidden_size
    residual = torch.randn(*residual_batch, branches * hidden).bfloat16()
    block = torch.randn(*block_batch, hidden).bfloat16()
    norm = torch.randn(*norm_batch, branches * hidden).bfloat16()
    injection = F.linear(norm, module.block_inject_weight.weight)
    gates = 2 * torch.sigmoid(injection.float() / branches)
    expected = (
        (
            residual.float().unflatten(-1, (branches, hidden))
            + block.float().unsqueeze(-2) * gates.unsqueeze(-1)
        )
        .bfloat16()
        .flatten(-2)
    )

    actual = module.combine(block, (residual, norm))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@TPU
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("tokens", [128, 256, 384])
def test_hc_tpu_full_mix_and_combine(compiled, tokens):
    with torch.device("cpu"):
        module, x, block = hc_fixture(True, tokens=tokens, full_tile=128)
        expected = hc_cuda_source_reference(module, x, block)
    assert module._prefill_full_ops
    assert module._prefill_combine_op is not None
    module = module.to("tpu")

    def chain(hyper_input, block_output):
        mixed, residuals = module.mix(hyper_input)
        return mixed, module.combine(block_output, residuals)

    run = (
        torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else chain
    )
    with torch.inference_mode():
        actual = run(x.to("tpu"), block.to("tpu"))
        repeated = run(x.to("tpu"), block.to("tpu"))
    for first, second in zip(actual, repeated, strict=True):
        torch.testing.assert_close(first.cpu(), second.cpu(), rtol=0, atol=0)
    for a, e in zip(actual, expected, strict=True):
        assert np.isfinite(_np(a.cpu())).all()
        assert_metrics(_np(a.cpu()), _np(e))


@pytest.mark.parametrize(
    "capacity,full_tile,combine_tile", [(16, 128, 16), (64, 512, 32)]
)
def test_hc_resource_fallback_keeps_independent_combine(
    capacity, full_tile, combine_tile, monkeypatch
):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter
    from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware

    monkeypatch.setattr(
        vllm_adapter,
        "current_hc_hardware",
        lambda: HcHardware("mock-layout", capacity << 20, 128, 16, 8, 256),
    )
    config = HyperConnectionConfig(
        hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=True
    )
    module = GatedResidual(
        config, prefill_full_tile=full_tile, prefill_combine_tile=combine_tile
    )
    assert not module._prefill_full_ops
    assert module._prefill_combine_op is not None
    x = torch.ones(combine_tile, 10240, dtype=torch.bfloat16)
    block = torch.ones(combine_tile, 2560, dtype=torch.bfloat16)
    calls = []

    def merge(residual, block, gates):
        calls.append(gates.dtype)
        return residual

    module._prefill_combine_op = merge
    expected = hc_cuda_source_reference(module, x, block)
    mixed, residual = module.mix(x)
    assert torch.isfinite(mixed).all()
    torch.testing.assert_close(mixed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        module.combine(block, residual), expected[1], rtol=0, atol=0
    )
    assert calls == []


def test_hc_unknown_hardware_and_default_optout_do_not_claim_device(monkeypatch):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter

    def forbidden():
        raise AssertionError("default optout must not inspect hardware")

    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", forbidden)
    module, _, _ = hc_fixture(True)
    assert not module._prefill_full_ops
    assert module._prefill_combine_op is None
    monkeypatch.setattr(vllm_adapter, "current_hc_hardware", lambda: None)
    module, x, block = hc_fixture(True, tokens=128, full_tile=128)
    assert not module._prefill_full_ops
    assert module._prefill_combine_op is None
    expected = hc_cuda_source_reference(module, x, block)
    mixed, residual = module.mix(x)
    torch.testing.assert_close(mixed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(
        module.combine(block, residual), expected[1], rtol=0, atol=0
    )


@TPU
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_residual_only_small_tile(compiled):
    with torch.device("cpu"):
        config = HyperConnectionConfig(
            hidden_size=2560, hc_count=3, hc_lowrank=320, hc_per_branch_norm=True
        )
        module = GatedResidual(config, prefill_full_tile=0, prefill_combine_tile=32)
        x = torch.ones(96, 7680, dtype=torch.bfloat16)
        block = torch.full((96, 2560), 0.25, dtype=torch.bfloat16)
        expected = hc_cuda_source_reference(module, x, block)
    assert not module._prefill_full_ops and module._prefill_combine_op is not None
    module = module.to("tpu")

    def chain(x, block):
        mixed, residual = module.mix(x)
        return mixed, module.combine(block, residual)

    run = (
        torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)
        if compiled
        else chain
    )
    with torch.inference_mode():
        actual = run(x.to("tpu"), block.to("tpu"))
    for a, e in zip(actual, expected, strict=True):
        assert np.isfinite(_np(a.cpu())).all()
        assert_metrics(_np(a.cpu()), _np(e))
