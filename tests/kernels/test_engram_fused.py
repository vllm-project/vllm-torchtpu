# SPDX-License-Identifier: Apache-2.0
"""Packed rows must preserve SP head order, FP8 numerics and masked residuals."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax import export

from vllm_torchtpu.kernels.engram import tuned_block_sizes
from vllm_torchtpu.kernels.engram.kernel import fused_engram

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("tokens", [256, 1024])
def test_released_engram_shapes_lower_to_one_tpu_call(tokens):
    """Check Mosaic lowering at serving shapes without allocating model weights."""
    tp, dim, width, streams = 8, 5120, 6144, 4
    shape = jax.ShapeDtypeStruct
    inputs = (
        shape((tp, tokens * tp, width // tp // 4), jnp.int32),
        shape((tp, tokens * tp, width // tp // 32 // 4), jnp.int32),
        shape((tokens, streams, dim), jnp.bfloat16),
        shape((width, (streams + 1) * dim), jnp.float8_e4m3fn),
        shape(((streams + 1) * dim,), jnp.float32),
        shape((streams, dim), jnp.bfloat16),
        shape((streams, dim), jnp.bfloat16),
        shape((tokens,), jnp.bool),
    )
    fn = jax.jit(
        functools.partial(fused_engram, rank=tp - 1, eps=1e-20, clamp_value=1e-6)
    )
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("device",),
        abstract_device=jax.sharding.AbstractDevice(
            "TPU7x", num_cores=1, platform="tpu"
        ),
    )
    with jax.sharding.use_abstract_mesh(mesh):
        lowered = export.export(fn, platforms=("tpu",))(*inputs)
    assert lowered.out_avals[0].shape == (tokens, streams, dim)
    assert lowered.out_avals[0].dtype == jnp.bfloat16
    module = lowered.mlir_module()
    assert module.count("stablehlo.custom_call @tpu_custom_call") == 1
    assert "stablehlo.transpose" not in module


@pytest.mark.parametrize("tp", [1, 2, 8])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("tile_m", [32, 64])
def test_fused_engram_matches_torch_offload(tp, masked, tile_m, monkeypatch):
    tokens, local_width, streams, dim = 2 * tile_m, 128, 4, 128
    width = tp * local_width
    monkeypatch.setitem(
        tuned_block_sizes.BLOCK_SIZES, (width, streams, dim), (tile_m, 128)
    )
    rng = torch.Generator().manual_seed(53)
    raw = (
        torch.randn(tp, tokens * tp, local_width, generator=rng)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    scale = torch.randint(
        124,
        129,
        (tp, tokens * tp, local_width // 32),
        generator=rng,
        dtype=torch.uint8,
    )
    scale[:, ::7] = 255
    raw[:, ::7] = 127
    hidden = torch.randn(tokens, streams, dim, generator=rng).bfloat16()
    weight = torch.randn(width, (streams + 1) * dim, generator=rng).to(
        torch.float8_e4m3fn
    )
    weight_scale = torch.rand((streams + 1) * dim, generator=rng) * 0.03
    q = torch.randn(streams, dim, generator=rng).bfloat16()
    k = torch.randn(streams, dim, generator=rng).bfloat16()
    mask = torch.arange(tokens) % 3 != 0 if masked else torch.ones(tokens).bool()
    rank = tp - 1
    start = rank * tokens
    local_raw = raw[:, start : start + tokens]
    local_scale = scale[:, start : start + tokens]
    power = (local_scale.int() << 23).view(torch.float32).repeat_interleave(32, -1)
    invalid = (local_scale == 255).repeat_interleave(32, -1)
    embedded = (
        torch.where(invalid, 0.0, local_raw.view(torch.float8_e4m3fn).float() * power)
        .bfloat16()
        .transpose(0, 1)
        .reshape(tokens, width)
    )
    xscale = embedded.float().abs().amax(-1, keepdim=True) / 448
    xscale = torch.where(xscale == 0, 1.0, xscale)
    xq = (embedded.float() / xscale).clamp(-448, 448).to(torch.float8_e4m3fn)
    projection = ((xq.float() @ weight.float()) * xscale * weight_scale).bfloat16()
    keys = projection[:, : streams * dim].reshape(tokens, streams, dim).float()
    values = projection[:, streams * dim :].float()
    h = hidden.float()
    dot = (h * q * k * keys).sum(-1)
    dot *= torch.rsqrt(h.square().mean(-1) + 1e-20)
    dot *= torch.rsqrt(keys.square().mean(-1) + 1e-20) * dim**-0.5
    magnitude = dot.abs().clamp(min=1e-6).sqrt()
    gate = torch.where(dot < 0, -magnitude, magnitude).sigmoid() * mask[:, None]
    expected = (h + gate[..., None] * values[:, None]).bfloat16().float().numpy()

    actual = fused_engram(
        jnp.asarray(raw.view(torch.int32).numpy()),
        jnp.asarray(scale.view(torch.int32).numpy()),
        jnp.asarray(h.numpy(), jnp.bfloat16),
        jnp.asarray(weight.float().numpy(), jnp.float8_e4m3fn),
        jnp.asarray(weight_scale.numpy()),
        jnp.asarray(q.float().numpy(), jnp.bfloat16),
        jnp.asarray(k.float().numpy(), jnp.bfloat16),
        jnp.asarray(mask.numpy()),
        rank,
        1e-20,
        1e-6,
        interpret=True,
    )
    actual = np.asarray(actual.astype(jnp.float32))
    np.testing.assert_allclose(actual, expected, atol=0.02, rtol=0.01)
    np.testing.assert_array_equal(actual[~mask.numpy()], h.numpy()[~mask.numpy()])
