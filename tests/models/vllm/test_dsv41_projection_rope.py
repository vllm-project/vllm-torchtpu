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

"""Projection/RoPE rounding, dispatch, and dynamic-token export contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.quantized_matmul import projection_rope as kernel


def _abstract_inputs(tokens=288):
    return {
        "x": jax.ShapeDtypeStruct((tokens, 1280), jnp.bfloat16),
        "w": jax.ShapeDtypeStruct((1280, 32768), jnp.float8_e4m3fn),
        "scale": jax.ShapeDtypeStruct((32768,), jnp.float32),
        "pos": jax.ShapeDtypeStruct((tokens,), jnp.int32),
        "cache": jax.ShapeDtypeStruct((98816, 64), jnp.float32),
    }


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "field,shape,dtype",
    [
        (None, None, None),
        ("x", (240, 1280), jnp.bfloat16),
        ("x", (2048, 1280), jnp.bfloat16),
        ("x", (288, 1280), jnp.float32),
        ("w", (1280, 16384), jnp.float8_e4m3fn),
        ("w", (1280, 32768), jnp.bfloat16),
        ("scale", (1, 32768), jnp.float32),
        ("scale", (32768,), jnp.bfloat16),
        ("pos", (288,), jnp.int64),
        ("cache", (98304, 64), jnp.float32),
    ],
)
def test_projection_rope_rejects_unqualified_inputs(monkeypatch, field, shape, dtype):
    monkeypatch.setattr(kernel, "get_tpu_version", lambda: 7)
    inputs = _abstract_inputs()
    if field is not None:
        inputs[field] = jax.ShapeDtypeStruct(shape, dtype)
    assert kernel.is_supported(**inputs) is (field is None)


@pytest.mark.cpu_test
@pytest.mark.parametrize("generation", [-1, 5, 6, 8])
def test_projection_rope_rejects_unqualified_devices(monkeypatch, generation):
    monkeypatch.setattr(kernel, "get_tpu_version", lambda: generation)
    assert not kernel.is_supported(**_abstract_inputs())


@pytest.fixture(scope="module")
def projection_ops():
    if kernel.get_tpu_version() != 7:
        pytest.skip("The fused projection/RoPE specialization requires TPU v7")

    from jax.sharding import Mesh
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.deepseek_v4.rope import rope
    from vllm_torchtpu.layers.adapter.linear_common import (
        _get_quantized_matmul_op,
        quantized_matmul,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.projection_rope import (
        initialize_projection_rope,
    )
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("test",))
    with set_vllm_model_wrapper_context(mesh=mesh):
        candidate = initialize_projection_rope()
        _get_quantized_matmul_op()
    rotary = pallas.jax_op("pallas::test_projection_rope_reference", rope)
    rotary.register_fake(lambda x, pos, cache: x.new_empty(x.shape))

    def both(x, w, scale, pos, cache):
        reference = rotary(
            quantized_matmul(x, w, scale).reshape(x.shape[0], 64, 512), pos, cache
        )
        return reference, candidate(x, w, scale, pos, cache)

    gen = torch.Generator().manual_seed(10037)
    weights = torch.randn(1280, 32768, generator=gen).to(torch.float8_e4m3fn).to("tpu")
    scales = torch.rand(32768, generator=gen).mul(0.025).add(0.005).to("tpu")
    frequencies = torch.arange(98816, dtype=torch.float32)[:, None] * torch.exp(
        -torch.arange(32, dtype=torch.float32)[None, :] / 4
    )
    cache = torch.cat([frequencies.cos(), frequencies.sin()], dim=1).to("tpu")
    return (
        candidate,
        torch.compile(both, backend="tpu", fullgraph=True),
        weights,
        scales,
        cache,
    )


@pytest.mark.parametrize(
    "case", ["random", "zero", "mixed_scales", "draft240", "prefill2048"]
)
def test_projection_rope_bridge_is_bitwise_equal(projection_ops, case):
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.projection_rope import (
        projection_rope,
    )

    _, compiled, weights, scales, cache = projection_ops
    tokens = {"draft240": 240, "prefill2048": 2048}.get(case, 288)
    # Verify that parity cannot pass just because the bridge silently fell back.
    jaxpr = str(jax.make_jaxpr(projection_rope)(**_abstract_inputs(tokens)))
    assert ("fused_quant_projection_rope_output48_all" in jaxpr) is (tokens == 288)
    gen = torch.Generator().manual_seed(42)
    x = torch.randn(tokens, 1280, generator=gen).to(torch.bfloat16)
    if case == "zero":
        x.zero_()
    elif case == "mixed_scales":
        x[:96] *= 0.0001
        x[96:192] *= 100
    pos = torch.randint(0, 98816, (tokens,), generator=gen, dtype=torch.int32)
    pos[:8] = torch.tensor([0, 0, 1, 98303, 98815, 42, 42, 98815], dtype=torch.int32)
    reference, actual = compiled(x.to("tpu"), weights, scales, pos.to("tpu"), cache)
    assert torch.equal(
        reference.cpu().view(torch.int16), actual.cpu().view(torch.int16)
    )


def test_projection_rope_export_preserves_dynamic_tokens(projection_ops):
    candidate, _, weights, scales, cache = projection_ops

    class Projection(torch.nn.Module):
        def forward(self, x, w, scale, pos, cache):
            return candidate(x, w, scale, pos, cache)

    tokens = torch.export.Dim("tokens", min=1, max=2048)
    inputs = (
        torch.zeros(288, 1280, dtype=torch.bfloat16, device="tpu"),
        weights,
        scales,
        torch.zeros(288, dtype=torch.int32, device="tpu"),
        cache,
    )
    exported = torch.export.export(
        Projection(),
        inputs,
        dynamic_shapes=({0: tokens}, None, None, {0: tokens}, None),
    )
    assert exported.range_constraints
