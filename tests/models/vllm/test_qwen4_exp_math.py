# SPDX-License-Identifier: Apache-2.0
"""Checkpoint geometry/math contracts; TPU bridge checks are explicitly opt-in."""
import os
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm_torchtpu.models.vllm.qwen4_exp.fp8 import (ExpertGeometry,
                                                     pack_expert_projection)
from vllm_torchtpu.models.vllm.qwen4_exp.gdn import (build_gdn_op,
                                                     gdn_state_shapes)
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual, HyperConnectionBase, HyperConnectionConfig)
from vllm_torchtpu.models.vllm.qwen4_exp.moe import route_tokens, shared_expert

TPU = pytest.mark.skipif(
    os.environ.get("Q38_TEST_TPU") != "1",
    reason="Set Q38_TEST_TPU=1 in the serialized TPU window")


@pytest.fixture(autouse=True)
def cpu_default(request):
    # Repository device fixture defaults to TPU; CPU cases intentionally never
    # request it. Explicit factories also protect against a caller default.
    if "tpu" in request.node.name:
        yield
    else:
        with torch.device("cpu"):
            yield


def test_gdn_rejects_model_mesh_subdivision_before_registration(monkeypatch):
    from vllm_torchtpu.models.vllm.qwen4_exp import gdn

    def unexpected_initialization(*args, **kwargs):
        pytest.fail("Unsupported mesh must be rejected before op construction")

    monkeypatch.setattr(gdn.jax, "shard_map", unexpected_initialization)
    monkeypatch.setattr(gdn.pallas, "jax_op", unexpected_initialization)
    with pytest.raises(ValueError, match="model.*1"):
        build_gdn_op("test_invalid_mesh",
                     8,
                     mesh=SimpleNamespace(shape={"model": 2}))


def test_gdn_accepts_single_device_model_mesh():
    mesh = jax.sharding.Mesh(np.asarray(jax.devices("cpu")[:1]), ("model", ))
    assert callable(build_gdn_op("test_single_device_mesh", 8, mesh=mesh))


def hc_fixture(per_branch):
    torch.manual_seed(383)
    config = HyperConnectionConfig(hidden_size=2560,
                                   hc_count=4,
                                   hc_lowrank=320,
                                   hc_per_branch_norm=per_branch)
    module = GatedResidual(config)
    with torch.no_grad():
        module.hc_norm.weight.copy_(
            torch.randn_like(module.hc_norm.weight) * .1)
    x = (torch.randn(3, 4, 2560, device="cpu") +
         torch.arange(4, device="cpu")[None, :, None]).bfloat16()
    block = torch.randn(3, 2560, device="cpu").bfloat16()
    return module, x.flatten(1), block


def hc_cuda_source_reference(module, flat, block):
    """Reference the BF16 load/store boundaries in NVIDIA HC kernels."""
    x = flat.reshape(-1, 4, 2560).float()
    weight = module.hc_norm.weight.float().reshape(-1, 2560)
    norm = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) *
            (1 + weight)).bfloat16().flatten(1)
    down = F.linear(norm, module.input_mix_weight_down.weight)
    silu = F.silu(down.float() / 4).bfloat16()
    gate = F.linear(silu, module.input_mix_weight_up.weight)
    gate = torch.sigmoid(gate.float()).reshape(-1, 4, 2560)
    mixed = (gate * norm.float().reshape(-1, 4, 2560)).mean(1).bfloat16()
    inject = F.linear(norm, module.block_inject_weight.weight)
    inject = 2 * torch.sigmoid(inject.float() / 4)
    output = (
        flat.float().reshape(-1, 4, 2560) +
        block.float()[:, None] * inject[:, :, None]).bfloat16().flatten(1)
    return mixed, output


def hc_legacy_common_bf16_reference(module, flat, block):
    """The former common-PyTorch contract, kept as a counterexample."""
    x = flat.reshape(-1, 4, 2560).float()
    weight = module.hc_norm.weight.float().reshape(-1, 2560)
    norm = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) *
            (1 + weight)).bfloat16().flatten(1)
    down = F.linear(norm, module.input_mix_weight_down.weight)
    gate = F.silu(down / 4)
    gate = torch.sigmoid(F.linear(gate, module.input_mix_weight_up.weight))
    mixed = (gate.reshape(-1, 4, 2560) * norm.reshape(-1, 4, 2560)).mean(1)
    inject = 2 * torch.sigmoid(
        F.linear(norm, module.block_inject_weight.weight) / 4)
    output = (flat.reshape(-1, 4, 2560) +
              block[:, None] * inject[:, :, None]).flatten(1)
    return mixed, output


@pytest.mark.parametrize("per_branch", [False, True])
def test_hc_checkpoint_groups_and_names(per_branch):
    module, x, block = hc_fixture(per_branch)
    expected = hc_cuda_source_reference(module, x, block)
    mixed, residual = module.mix(x)
    torch.testing.assert_close(mixed, expected[0], rtol=0, atol=0)
    torch.testing.assert_close(module.combine(block, residual),
                               expected[1],
                               rtol=0,
                               atol=0)
    assert set(module.state_dict()) == {
        "hc_norm.weight", "input_mix_weight_down.weight",
        "input_mix_weight_up.weight", "block_inject_weight.weight"
    }
    assert module.hc_norm.weight.numel() == (10240 if per_branch else 2560)
    numbered = torch.arange(10240).float().reshape(1, 4, 2560)
    assert not torch.equal(numbered.mean(1),
                           numbered.reshape(1, 2560, 4).mean(2))
    pooled, _ = HyperConnectionBase(module.config).mix(numbered.flatten(1))
    torch.testing.assert_close(pooled, numbered.mean(1), rtol=0, atol=0)


def test_hc_gate_mix_matches_cuda_fp32_rounding_at_real_geometry():
    """Catch BF16 rounding after gate sigmoid or gate-times-norm."""
    module = GatedResidual(HyperConnectionConfig(hidden_size=2560,
                                                 hc_count=4,
                                                 hc_lowrank=320,
                                                 hc_per_branch_norm=True),
                           use_combine=False)
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
    expected = torch.tensor(0.0030975341796875,
                            dtype=torch.bfloat16,
                            device="cpu")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_hc_combine_matches_cuda_fp32_rounding_at_real_geometry():
    """Catch BF16 rounding in injection sigmoid, product, or residual add."""
    module = GatedResidual(HyperConnectionConfig(hidden_size=2560,
                                                 hc_count=4,
                                                 hc_lowrank=320,
                                                 hc_per_branch_norm=True),
                           use_mix=False)
    with torch.no_grad():
        module.block_inject_weight.weight.zero_()
        module.block_inject_weight.weight[:, 0] = -8

    hyper_input = torch.full((1, 4 * 2560),
                             -1,
                             dtype=torch.bfloat16,
                             device="cpu")
    hyper_input_normed = torch.zeros_like(hyper_input, device="cpu")
    hyper_input_normed[:, 0] = 1
    block_output = torch.full((1, 2560),
                              -3.125,
                              dtype=torch.bfloat16,
                              device="cpu")
    actual = module.combine(block_output, (hyper_input, hyper_input_normed))[0,
                                                                             0]

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


def expert_fixture():
    gen = torch.Generator(device="cpu").manual_seed(385)
    weights, scales = {}, {}
    for name, n, k in (("gate", 640, 2560), ("up", 640, 2560), ("down", 2560,
                                                                640)):
        weights[name] = (torch.randint(-4, 5,
                                       (1, n, k), generator=gen).float() /
                         2).to(torch.float8_e4m3fn)
        scales[name] = (
            torch.arange(n // 128 *
                         (k // 128)).reshape(1, n // 128, k // 128) / 16384 +
            .00390625).bfloat16()
    return weights, scales


@pytest.fixture(scope="module")
def checkpoint():
    checkpoint_dir = os.environ.get("Q38_REAL_CHECKPOINT")
    if checkpoint_dir is None:
        return expert_fixture()
    import json
    from pathlib import Path

    from safetensors import safe_open
    root = Path(checkpoint_dir)
    index = json.loads(
        (root / "model.safetensors.index.json").read_text())["weight_map"]
    weights, scales = {}, {}
    for projection in ("gate", "up", "down"):
        for field, target in (("weight", weights), ("weight_scale_inv",
                                                    scales)):
            key = f"model.language_model.layers.0.mlp.experts.0.{projection}_proj.{field}"
            with safe_open(root / index[key], framework="pt",
                           device="cpu") as f:
                target[projection] = f.get_tensor(key).unsqueeze(0)
    return weights, scales


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
@pytest.mark.parametrize("candidate", ["A", "B"])
def test_fp8_global_scale_coordinates(checkpoint, tp, candidate):
    weights, scales = checkpoint
    widths = {
        "A": {
            1: 640,
            2: 384,
            4: 256,
            8: 256
        },
        "B": {
            1: 640,
            2: 384,
            4: 256,
            8: 128
        }
    }
    for rank in range(tp):
        g = ExpertGeometry(tp, rank, candidate)
        assert g.physical_size == widths[candidate][tp]
        begin, end = rank * (640 // tp), (rank + 1) * (640 // tp)
        offset = begin - g.base
        for name in weights:
            packed, scale = pack_expert_projection(weights[name], scales[name],
                                                   name, g)
            decoded = packed.float() * scale[:, :, 0].repeat_interleave(
                g.down_block if name == "down" else 128, dim=1)
            original = weights[name].float() * scales[name].float(
            ).repeat_interleave(128, 1).repeat_interleave(128, 2)
            if name == "down":
                actual = decoded[:, offset:offset + end - begin]
                expected = original[:, :, begin:end].transpose(1, 2)
                bits = packed[:, offset:offset + end - begin].view(torch.uint8)
                source = weights[name][:, :, begin:end].transpose(1, 2).view(
                    torch.uint8)
                mask = torch.ones(g.physical_size, dtype=torch.bool)
                mask[offset:offset + end - begin] = False
                assert not torch.count_nonzero(packed.float()[:, mask])
            else:
                actual = decoded[..., offset:offset + end - begin]
                expected = original[:, begin:end].transpose(1, 2)
                bits = packed[...,
                              offset:offset + end - begin].view(torch.uint8)
                source = weights[name][:, begin:end].transpose(1, 2).view(
                    torch.uint8)
                mask = torch.ones(g.physical_size, dtype=torch.bool)
                mask[offset:offset + end - begin] = False
                assert not torch.count_nonzero(packed.float()[..., mask])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert torch.equal(bits, source)
    # The historically wrong local scale origin must disagree across row128.
    if tp == 8:
        wrong = weights["gate"][:, 80:160].float(
        ) * scales["gate"][:, :1].float().repeat_interleave(128, 2)
        correct = (weights["gate"].float() *
                   scales["gate"].float().repeat_interleave(
                       128, 1).repeat_interleave(128, 2))[:, 80:160]
        assert not torch.equal(wrong, correct)


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_native_gdn_state_geometry(tp):
    conv, state = gdn_state_shapes(tp)
    assert conv == (3, (2 * (16 // tp) + 48 // tp) * 128)
    assert state == (48 // tp, 128, 128)
    assert 48 // tp == 3 * (16 // tp)


def _np(tensor):
    return np.asarray(tensor.detach().float().numpy(), np.float32)


def _bf(x):
    return torch.as_tensor(np.asarray(x).copy()).bfloat16().float().numpy()


def test_router_top10_ties_and_padding():
    x = jnp.ones((3, 2560), jnp.bfloat16)
    router = jnp.zeros((512, 2560), jnp.bfloat16)
    probs, ids, logits = route_tokens(x, router, jnp.array([True, False,
                                                            True]))
    np.testing.assert_array_equal(ids, np.tile(np.arange(10), (3, 1)))
    np.testing.assert_array_equal(logits, np.zeros((3, 512)))
    np.testing.assert_allclose(np.asarray(probs)[[0, 2]], .1)
    np.testing.assert_array_equal(probs[1], np.zeros(10))
    assert probs.dtype == jnp.float32


def shared_fixture():
    rng = np.random.default_rng(386)
    return [
        torch.from_numpy(rng.integers(-8, 9, shape).astype(np.float32) /
                         128).bfloat16()
        for shape in ((3, 2560), (640, 2560), (640, 2560), (2560, 640), (1,
                                                                         2560))
    ]


def shared_reference(x, gate, up, down, shared_gate):
    act = F.silu(F.linear(x, gate))
    return F.linear(act * F.linear(x, up), down) * torch.sigmoid(
        F.linear(x, shared_gate))


def test_shared_bf16_rounding():
    values = shared_fixture()
    result = shared_expert(
        *[jnp.asarray(_np(v), jnp.bfloat16) for v in values])
    np.testing.assert_array_equal(np.asarray(result, np.float32),
                                  _np(shared_reference(*values)))


def assert_metrics(actual, expected, threshold=.01):
    a, e = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    nrmse = np.linalg.norm(a - e) / max(np.linalg.norm(e), 1e-8)
    cosine = np.vdot(a.ravel(), e.ravel()) / max(
        np.linalg.norm(a) * np.linalg.norm(e), 1e-30)
    assert nrmse <= threshold, (nrmse, threshold)
    assert cosine >= .999, cosine


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

    run = (torch.compile(chain, backend="tpu", fullgraph=True, dynamic=False)
           if compiled else chain)
    with torch.inference_mode():
        actual = run(x.to("tpu"), block.to("tpu"))
    for a, e in zip(actual, expected):
        assert_metrics(_np(a.cpu()), _np(e))


@TPU
@pytest.mark.parametrize("compiled", [False, True])
def test_hc_tpu_negative_silu_preserves_mix_gate(compiled):
    module = GatedResidual(HyperConnectionConfig(hidden_size=2560,
                                                 hc_count=4,
                                                 hc_lowrank=320,
                                                 hc_per_branch_norm=True),
                           use_combine=False)
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

    run = (torch.compile(mix, backend="tpu", fullgraph=True, dynamic=False)
           if compiled else mix)
    with torch.inference_mode():
        actual = run(torch.ones(43, 10240, dtype=torch.bfloat16, device="tpu"))
    assert_metrics(_np(actual.cpu()), expected)


def gdn_fixture(tp, tokens=65):
    rng = np.random.default_rng(387 + tp)
    nk, nv = 16 // tp, 48 // tp
    width = (2 * nk + nv) * 128

    def normal(shape, scale=.1):
        return torch.from_numpy(
            (rng.standard_normal(shape) * scale).astype(np.float32))

    values = [
        normal((tokens, width)).bfloat16(),
        normal((tokens, nv)).bfloat16(),
        normal((tokens, nv)).bfloat16(),
        normal((3, 3, width)),
        normal((3, nv, 128, 128), .01),
        normal((width, 1, 4)).bfloat16(),
        normal((width, )).bfloat16(),
        normal((nv, )),
        normal((nv, ))
    ]
    return values


def gdn_reference(values, tp):
    qkv, b, a, cs, ss, cw, cb, al, dt = [_np(t).copy() for t in values]
    conv, state = cs[1].copy(), ss[1].copy()
    outputs = []
    nk, nv = 16 // tp, 48 // tp
    for t, x in enumerate(qkv):
        full = np.concatenate((conv, x[None]))
        mixed = (full * cw[:, 0].T).sum(0) + cb
        mixed = mixed / (1 + np.exp(-mixed))
        conv = full[-3:]
        q, k, v = np.split(mixed, (nk * 128, nk * 256))
        q = np.repeat(q.reshape(nk, 128), 3, 0)
        k = np.repeat(k.reshape(nk, 128), 3, 0)
        q /= np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) * np.sqrt(128)
        k /= np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        state *= np.exp(-np.exp(al) * np.logaddexp(0, a[t] + dt))[:, None,
                                                                  None]
        delta = (v.reshape(nv, 128) - np.einsum('hk,hkv->hv', k, state)) / (
            1 + np.exp(-b[t]))[:, None]
        state += np.einsum('hk,hv->hkv', k, delta)
        outputs.append(_bf(np.einsum('hk,hkv->hv', q, state)).reshape(-1))
    cs[1], ss[1] = conv, state
    return cs, ss, np.stack(outputs)


@TPU
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_gdn_tpu_private_state_writeback(tp):
    values = gdn_fixture(tp)
    ec, es, ey = gdn_reference(values, tp)
    op = build_gdn_op(f"test_tp{tp}", tp, has_conv_bias=True)
    before = jax.config.jax_default_matmul_precision
    args = [v.to("tpu") for v in values]
    args += [
        torch.tensor(v, dtype=torch.int32, device="tpu")
        for v in ([1], [0, 65], [0, 0, 1], [66])
    ]
    output = torch.compile(op, backend="tpu", fullgraph=True,
                           dynamic=False)(*args)
    assert_metrics(_np(output.cpu()), ey)
    actual_c, actual_s = _np(args[3].cpu()), _np(args[4].cpu())
    np.testing.assert_array_equal(actual_c, ec)
    assert_metrics(actual_s[1], es[1], .001)
    np.testing.assert_array_equal(actual_s[[0, 2]], es[[0, 2]])
    assert args[3].dtype == args[4].dtype == torch.float32
    assert jax.config.jax_default_matmul_precision == before


def test_streaming_moe_loader_preserves_checkpoint(checkpoint):
    from vllm_torchtpu.models.vllm.qwen4_exp.moe import Qwen4ExpSparseMoeBlock
    weights, scales = checkpoint
    module = Qwen4ExpSparseMoeBlock(ep_rank=1, device="cpu")
    # Scales can precede bits, and remote experts must not allocate pending data.
    for projection in ("gate", "up", "down"):
        module.load_weights([(f"experts.64.{projection}_proj.weight_scale_inv",
                              scales[projection][0]),
                             (f"experts.0.{projection}_proj.weight",
                              weights[projection][0])])
        module.load_weights([(f"experts.64.{projection}_proj.weight",
                              weights[projection][0])])
    assert not module._pending
    assert module._packed["w13"].shape == (64, 2560, 1280)
    assert module._packed["w2"].shape == (64, 640, 2560)
    assert module.experts_start.item() == 64
    torch.testing.assert_close(module._packed["w13"][0, :, :640].float(),
                               weights["gate"][0].T.float(),
                               rtol=0,
                               atol=0)
    torch.testing.assert_close(module._packed["w13"][0, :, 640:].float(),
                               weights["up"][0].T.float(),
                               rtol=0,
                               atol=0)
    torch.testing.assert_close(module._packed["s13"][0, :, 0, :640],
                               scales["gate"][0].float().repeat_interleave(
                                   128, 0).T,
                               rtol=0,
                               atol=0)
    with pytest.raises(ValueError, match="Incomplete"):
        module.finalize_weights()


@TPU
def test_moe_tpu_fp8_bridge(checkpoint):
    from vllm_torchtpu.models.vllm.qwen4_exp.moe import build_moe_op
    weights, scales = checkpoint
    values = shared_fixture()
    x = values[0].clone()
    x[:, 0] = 1
    router = torch.zeros((512, 2560), dtype=torch.bfloat16)
    selected = [0, 1, 2, 3, 4, 10, 63, 64, 128, 511]
    router[selected, 0] = torch.arange(1, 11).bfloat16() / 4
    logits = F.linear(x, router).float()
    probs, ids = torch.softmax(logits, -1).topk(10, -1)
    probs /= probs.sum(-1, keepdim=True)
    probs[1] = 0
    # Distinct expert scales detect an incorrect expert permutation even
    # when all experts reuse the same original checkpoint FP8 bit patterns.
    bank_scales = {
        name: (scale.float() *
               (1 + torch.arange(64)[:, None, None] / 64)).bfloat16()
        for name, scale in scales.items()
    }
    expected = torch.zeros_like(x, dtype=torch.float32)
    for expert in (0, 1, 2, 3, 4, 10, 63):
        decoded = {
            name: (weights[name][0].float() *
                   bank_scales[name][expert].float().repeat_interleave(
                       128, 0).repeat_interleave(128, 1)).bfloat16().float()
            for name in weights
        }
        hidden = F.silu(
            x.float() @ decoded["gate"].T) * (x.float() @ decoded["up"].T)
        expert_output = (
            hidden.bfloat16().float() @ decoded["down"].T).bfloat16().float()
        expected += expert_output * torch.where(ids == expert, probs, 0).sum(
            -1, keepdim=True)
    pairs = {
        name:
        pack_expert_projection(weights[name].repeat(64, 1, 1),
                               bank_scales[name], name, ExpertGeometry())
        for name in weights
    }
    w13 = torch.cat([pairs[n][0].view(torch.uint8) for n in ("gate", "up")],
                    -1).view(torch.float8_e4m3fn)
    s13 = torch.cat([pairs[n][1] for n in ("gate", "up")], -1)
    packed = [w13, s13, *pairs["down"]]
    args = [
        x, router, *packed, *values[1:],
        torch.tensor(0, dtype=torch.int32),
        torch.tensor([True, False, True])
    ]
    op = torch.compile(build_moe_op(),
                       backend="tpu",
                       fullgraph=True,
                       dynamic=False)
    routed, shared = op(*[v.to("tpu") for v in args])
    assert routed.dtype == torch.float32
    assert_metrics(_np(routed.cpu()), _np(expected))
    expected_shared = shared_reference(x, *values[1:])
    expected_shared[1] = 0
    np.testing.assert_array_equal(_np(shared.cpu()), _np(expected_shared))
    # All selected ids are outside this owner's shard: unwritten GMM tails
    # must not contaminate an idle rank or padded token with NaN.
    args[-2] = torch.tensor(256, dtype=torch.int32)
    idle, _ = op(*[v.to("tpu") for v in args])
    np.testing.assert_array_equal(_np(idle.cpu()), np.zeros((3, 2560)))
