# SPDX-License-Identifier: Apache-2.0
"""Fused decode HC mix (interpret mode) against the GatedResidual torch path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.hc_decode import decode_token_tile, hc_decode_mix
from vllm_torchtpu.models.vllm.qwen4_exp.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)

# Qwen3.8-Flash: 4 streams of hidden 2560, low rank 320; HC is TP-replicated,
# so the per-rank shapes are the full shapes.
BRANCHES, HIDDEN, LOWRANK, EPS = 4, 2560, 320, 1e-6


def _module(seed, hidden=HIDDEN, lowrank=LOWRANK):
    torch.manual_seed(seed)
    module = GatedResidual(
        HyperConnectionConfig(
            hc_count=BRANCHES,
            hidden_size=hidden,
            params_dtype=torch.bfloat16,
            hc_lowrank=lowrank,
            rms_norm_eps=EPS,
            hc_per_branch_norm=True,
        )
    )
    with torch.no_grad():
        module.hc_norm.weight.normal_(0.0, 0.3)
        module.input_mix_weight_down.weight.normal_(0.0, 0.02)
        module.input_mix_weight_up.weight.normal_(0.0, 0.05)
        module.block_inject_weight.weight.normal_(0.0, 0.02)
    return module


def _jax(t):
    return jnp.asarray(t.detach().float().numpy(), jnp.bfloat16)


def _weights(module):
    return (
        _jax(module.hc_norm.weight),
        _jax(module.input_mix_weight_down.weight),
        _jax(module.input_mix_weight_up.weight),
        _jax(module.block_inject_weight.weight),
    )


def _check_bf16(actual, expected, name):
    actual = np.asarray(jnp.asarray(actual, jnp.float32))
    expected = expected.float().numpy()
    assert np.isfinite(actual).all(), name
    diff = np.abs(actual - expected)
    # Only the matmul accumulation order (and FMA contraction) differ from
    # the torch path, so a BF16 rounding boundary can move by one ulp on a
    # small fraction of elements and propagate through the gate. The mixed
    # output is a sum of branch terms, so a one-ulp term change is bounded
    # by the tensor's scale rather than by the (possibly cancelled) result.
    scale = np.abs(expected).max()
    assert diff.max() <= 2.0**-7 * scale, (name, diff.max(), scale)
    assert np.linalg.norm(diff) <= 1e-3 * np.linalg.norm(expected), name
    assert (diff > 0).mean() <= 0.02, (name, (diff > 0).mean())


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 128])
def test_tile_choice(tokens):
    assert decode_token_tile(tokens) == tokens


def test_tile_choice_large():
    assert decode_token_tile(256) == 128
    assert decode_token_tile(512) == 128
    assert decode_token_tile(192) == 64
    assert decode_token_tile(136) is None
    assert decode_token_tile(1024) is None
    assert decode_token_tile(0) is None


@pytest.mark.parametrize("tokens", [1, 4, 8, 32, 128, 256])
def test_decode_mix_matches_torch_path(tokens):
    module = _module(tokens)
    torch.manual_seed(100 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        expected_mixed, residual = module.mix(hyper)
        block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
        expected_out = module.combine(block, residual)
        injection = F_injection(module, residual[1])

    weight, down, up, inject = _weights(module)
    mixed, inj = hc_decode_mix(
        _jax(hyper),
        weight,
        down,
        up,
        inject,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        interpret=pltpu.InterpretParams(),
    )
    _check_bf16(mixed, expected_mixed, "mixed")
    np.testing.assert_allclose(np.asarray(inj), injection.numpy(), rtol=2e-2, atol=1e-6)
    # The combine consuming the kernel's injection gate.
    with torch.no_grad():
        out = module.combine(block, (hyper, torch.from_numpy(np.asarray(inj))))
    _check_bf16(_jax(out), expected_out, "combine")


def F_injection(module, normed):
    logits = torch.nn.functional.linear(normed, module.block_inject_weight.weight)
    return 2.0 * torch.sigmoid(logits.float() / BRANCHES)


@pytest.mark.parametrize("tokens", [4, 32, 128])
def test_decode_combine_mix_matches_torch_path(tokens):
    attn, mlp = _module(tokens), _module(1000 + tokens)
    torch.manual_seed(200 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        _, residual = attn.mix(hyper)
        injection = F_injection(attn, residual[1])
        expected_h = attn.combine(block, residual)
        expected_mixed, _ = mlp.mix(expected_h)

    weight, down, up, inject = _weights(mlp)
    mixed, inj, h = hc_decode_mix(
        _jax(hyper),
        weight,
        down,
        up,
        inject,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        block=_jax(block),
        prev_injection=jnp.asarray(injection.numpy()),
        interpret=pltpu.InterpretParams(),
    )
    # The combine is elementwise FP32 with one BF16 rounding; only FMA
    # contraction may move an element by one ulp.
    _check_bf16(h, expected_h, "h")
    _check_bf16(mixed, expected_mixed, "mixed")
    assert inj.shape == (tokens, BRANCHES)


def test_decode_mix_without_injection():
    module = _module(7)
    hyper = torch.randn(8, BRANCHES * HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        expected_mixed, _ = module.mix(hyper)
    weight, down, up, _ = _weights(module)
    mixed, inj = hc_decode_mix(
        _jax(hyper),
        weight,
        down,
        up,
        branches=BRANCHES,
        hidden=HIDDEN,
        eps=EPS,
        interpret=pltpu.InterpretParams(),
    )
    assert inj is None
    _check_bf16(mixed, expected_mixed, "mixed")


def test_decode_mix_rejects_unsupported_shapes():
    shape = jax.ShapeDtypeStruct
    width = BRANCHES * HIDDEN
    args = (
        shape((136, width), jnp.bfloat16),
        shape((width,), jnp.bfloat16),
        shape((LOWRANK, width), jnp.bfloat16),
        shape((width, LOWRANK), jnp.bfloat16),
    )
    with pytest.raises(ValueError, match="decode-sized"):
        jax.eval_shape(
            lambda *a: hc_decode_mix(*a, branches=BRANCHES, hidden=HIDDEN, eps=EPS),
            *args,
        )


def _interpret_ops(module):
    """Stand-ins for the registered torch ops that run the kernel in
    interpret mode on CPU tensors."""

    def to_torch(a):
        if a.dtype == jnp.bfloat16:
            return torch.from_numpy(np.asarray(a.astype(jnp.float32))).to(
                torch.bfloat16
            )
        return torch.from_numpy(np.array(a))

    def to_jax(t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            return _jax(t)
        return jnp.asarray(t.numpy())

    def mix_op(x, *weights):
        mixed, inj = hc_decode_mix(
            *map(to_jax, (x, *weights)),
            branches=BRANCHES,
            hidden=HIDDEN,
            eps=EPS,
            interpret=pltpu.InterpretParams(),
        )
        return to_torch(mixed) if inj is None else (to_torch(mixed), to_torch(inj))

    def combine_mix_op(residual, block, gate, *weights):
        outs = hc_decode_mix(
            *map(to_jax, (residual, *weights)),
            branches=BRANCHES,
            hidden=HIDDEN,
            eps=EPS,
            block=to_jax(block),
            prev_injection=to_jax(gate),
            interpret=pltpu.InterpretParams(),
        )
        return tuple(map(to_torch, outs))

    module._decode_mix_op = mix_op
    module._decode_combine_mix_op = combine_mix_op
    return module


def test_layer_plumbing_matches_unfused_layer():
    """attn mix -> combine_mix -> mlp combine, as Qwen4ExpDecoderLayer runs."""
    tokens = 8
    attn, mlp = _module(11), _module(12)
    fused_attn, fused_mlp = _interpret_ops(_module(11)), _interpret_ops(_module(12))
    torch.manual_seed(13)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    attn_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    mlp_out = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        outs = []
        for a, m in ((attn, mlp), (fused_attn, fused_mlp)):
            block, residual = a.mix(hyper)
            block2, residual = a.combine_mix(attn_out, residual, m)
            outs.append((block, block2, m.combine(mlp_out, residual)))
    (ref_block, ref_block2, ref_out), (block, block2, out) = outs
    assert fused_attn._decode_fused(hyper)
    _check_bf16(_jax(block), ref_block, "attn mixed")
    _check_bf16(_jax(block2), ref_block2, "mlp mixed")
    _check_bf16(_jax(out), ref_out, "layer output")
    # Unfused modules never take the decode path.
    assert not attn._decode_fused(hyper)
