# SPDX-License-Identifier: Apache-2.0
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
    if "tpu" in request.node.name:
        yield
    else:
        with torch.device("cpu"):
            yield


def hc_fixture(per_branch):
    # Seed only CPU: torch.manual_seed also initializes registered accelerators.
    torch.random.default_generator.manual_seed(383)
    config = HyperConnectionConfig(
        hidden_size=2560, hc_count=4, hc_lowrank=320, hc_per_branch_norm=per_branch
    )
    module = GatedResidual(config)
    with torch.no_grad():
        module.hc_norm.weight.copy_(torch.randn_like(module.hc_norm.weight) * 0.1)
    x = (
        torch.randn(3, 4, 2560, device="cpu")
        + torch.arange(4, device="cpu")[None, :, None]
    ).bfloat16()
    block = torch.randn(3, 2560, device="cpu").bfloat16()
    return module, x.flatten(1), block


def hc_cuda_source_reference(module, flat, block):
    """Reference the BF16 load/store boundaries in NVIDIA HC kernels."""
    x = flat.reshape(-1, 4, 2560).float()
    weight = module.hc_norm.weight.float().reshape(-1, 2560)
    norm = (
        (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * (1 + weight))
        .bfloat16()
        .flatten(1)
    )
    down = F.linear(norm, module.input_mix_weight_down.weight)
    silu = F.silu(down.float() / 4).bfloat16()
    gate = F.linear(silu, module.input_mix_weight_up.weight)
    gate = torch.sigmoid(gate.float()).reshape(-1, 4, 2560)
    mixed = (gate * norm.float().reshape(-1, 4, 2560)).mean(1).bfloat16()
    inject = F.linear(norm, module.block_inject_weight.weight)
    inject = 2 * torch.sigmoid(inject.float() / 4)
    output = (
        (
            flat.float().reshape(-1, 4, 2560)
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


def _bf(x):
    return torch.as_tensor(np.asarray(x).copy()).bfloat16().float().numpy()


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
