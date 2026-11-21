# SPDX-License-Identifier: Apache-2.0
"""Fused decode HC mix (interpret mode) against the GatedResidual torch path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.models.qwen38_flash.experimental.hyperconnection import (
    GatedResidual,
    HyperConnectionConfig,
)
from vllm_torchtpu.models.qwen38_flash.experimental.kernels.hc_decode import (
    decode_token_tile,
    hc_decode_mix,
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
        # mix returns the combine's injection gate as the second residual.
        injection = residual[1]

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


@pytest.mark.parametrize("tokens", [4, 32, 128])
def test_decode_combine_mix_matches_torch_path(tokens):
    attn, mlp = _module(tokens), _module(1000 + tokens)
    torch.manual_seed(200 + tokens)
    hyper = torch.randn(tokens, BRANCHES * HIDDEN).to(torch.bfloat16)
    block = torch.randn(tokens, HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        _, residual = attn.mix(hyper)
        injection = residual[1]
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


def test_small_rows_skips_prefill_tiles_at_128_rows():
    """Megakernel verify programs mix 128 verify rows (32 requests x 4) with
    small_rows=True: the prefill-tile ops must not run, and the result must be the
    module's small-row path at the same row count bit for bit (the path it takes
    when no prefill tile is configured)."""
    module = _module(7)
    calls = []

    def prefill_op(name):
        def op(*args):
            calls.append(name)
            raise AssertionError(f"prefill {name} op taken")

        return op

    module.prefill_norm_tile = module.prefill_up_tile = module.prefill_full_tile = 128
    module._prefill_norm_op = prefill_op("norm")
    module._prefill_up_op = prefill_op("up")
    module._prefill_full_op = prefill_op("full")
    torch.manual_seed(128)
    hyper = torch.randn(128, BRANCHES * HIDDEN).to(torch.bfloat16)
    with torch.no_grad():
        mixed, (_, injection) = module.mix(hyper, small_rows=True)
    assert calls == []
    plain = _module(7)  # same weights, no prefill tiles configured
    with torch.no_grad():
        ref_mixed, (_, ref_injection) = plain.mix(hyper)
    torch.testing.assert_close(mixed, ref_mixed, rtol=0, atol=0)
    torch.testing.assert_close(injection, ref_injection, rtol=0, atol=0)
    # Positive control: without small_rows the same 128 rows take the prefill tiles.
    with pytest.raises(AssertionError, match="prefill"):  # noqa: SIM117
        with torch.no_grad():
            module.mix(hyper)
    assert calls == ["norm"]
