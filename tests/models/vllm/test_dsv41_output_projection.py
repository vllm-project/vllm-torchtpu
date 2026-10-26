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

"""Output-projection dispatch, compiled bridge parity, and export contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4 import o_projection as oa
from vllm_torchtpu.kernels.quantized_matmul import buffered as ob


def _abstract_inputs(tokens=288):
    return (
        jax.ShapeDtypeStruct((tokens, 64, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((4096, 8192), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((8192,), jnp.float32),
        jax.ShapeDtypeStruct((8192, 5120), jnp.float8_e4m3fn),
        jax.ShapeDtypeStruct((5120,), jnp.float32),
    )


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens,expected",
    [(240, True), (288, True), (1, False), (256, False), (2048, False)],
)
@pytest.mark.parametrize("generation", [-1, 6, 7, 8])
def test_output_projection_envelope(monkeypatch, tokens, expected, generation):
    monkeypatch.setattr(oa, "get_tpu_version", lambda: generation)
    monkeypatch.setattr(ob, "get_tpu_version", lambda: generation)
    x, wa, sa, wb, sb = _abstract_inputs(tokens)
    xb = jax.ShapeDtypeStruct((tokens, 8192), jnp.bfloat16)
    expected = expected and generation == 7
    assert oa.is_buffered_projection_supported(x, wa, sa) is expected
    assert ob.is_supported(xb, wb, sb) is expected


@pytest.mark.cpu_test
@pytest.mark.parametrize("which", ["oa", "ob"])
@pytest.mark.parametrize(
    "change",
    [
        "x_dtype",
        "w_dtype",
        "scale_dtype",
        "scale_rank",
        "sharded_weight",
        "block_scale",
    ],
)
def test_output_projection_rejects_other_contracts(monkeypatch, which, change):
    module = oa if which == "oa" else ob
    monkeypatch.setattr(module, "get_tpu_version", lambda: 7)
    x, wa, sa, wb, sb = _abstract_inputs()
    inputs = (
        [x, wa, sa]
        if which == "oa"
        else [jax.ShapeDtypeStruct((288, 8192), jnp.bfloat16), wb, sb]
    )
    if change.endswith("dtype"):
        index = {"x_dtype": 0, "w_dtype": 1, "scale_dtype": 2}[change]
        dtype = jnp.float32 if index == 0 else jnp.bfloat16
        inputs[index] = jax.ShapeDtypeStruct(inputs[index].shape, dtype)
    elif change == "scale_rank":
        inputs[2] = jax.ShapeDtypeStruct((1, *inputs[2].shape), jnp.float32)
    elif change == "block_scale":
        inputs[2] = jax.ShapeDtypeStruct((1, 64, 1, inputs[2].shape[0]), jnp.float32)
    else:
        inputs[1] = jax.ShapeDtypeStruct(
            (inputs[1].shape[0], inputs[1].shape[1] // 2), jnp.float8_e4m3fn
        )
    supported = (
        oa.is_buffered_projection_supported if which == "oa" else ob.is_supported
    )
    assert not supported(*inputs)


@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [240, 288])
def test_buffered_projection_traces_memory_space_boundary(monkeypatch, tokens):
    monkeypatch.setattr(oa, "get_tpu_version", lambda: 7)
    x, w, scale, _, _ = _abstract_inputs(tokens)
    # Tracing rejects reshape on a Pallas HBM-annotated array, before TPU
    # compilation. Exercise the public dispatch, including its scale reshape.
    graph = str(jax.make_jaxpr(oa.wo_a_projection)(x, w, scale, None))
    assert "wo_a_projection_buffered" in graph


def _reference_a(x: jax.Array, w: jax.Array, scale: jax.Array) -> jax.Array:
    # Reproduce the pre-specialization defaults, bypassing automatic dispatch.
    tile_t = next(n for n in range(min(x.shape[0], 1024), 0, -1) if x.shape[0] % n == 0)
    sub_t = next(n for n in range(min(tile_t, 128), 0, -1) if tile_t % n == 0)
    return oa.wo_a_projection(x, w, scale, None, tile_t=tile_t, sub_t=sub_t)


@pytest.fixture(scope="module")
def projection_ops():
    if ob.get_tpu_version() != 7:
        pytest.skip("Buffered output projection is qualified on TPU v7")
    from jax.sharding import Mesh
    from torch_tpu._internal import pallas

    from vllm_torchtpu.kernels.quantized_matmul.util import xla_quantized_matmul
    from vllm_torchtpu.layers.adapter.linear_common import (
        _get_quantized_matmul_op,
        quantized_matmul,
    )
    from vllm_torchtpu.models.vllm.deepseek_v41.v2 import csa2_adapter
    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        set_vllm_model_wrapper_context,
    )

    mesh = Mesh(np.array(jax.devices()[:1]), ("test",))
    with set_vllm_model_wrapper_context(mesh=mesh):
        csa2_adapter.initialize_csa2_ops()
        _get_quantized_matmul_op()
    ref_a = pallas.jax_op("pallas::test_output_projection_a_reference", _reference_a)

    # xla_quantized_matmul has an unannotated optional quantization argument;
    # expose only the three tensor arguments required by this custom op.
    def reference_b(x: jax.Array, w: jax.Array, scale: jax.Array) -> jax.Array:
        return xla_quantized_matmul(x, w, scale)

    ref_b = pallas.jax_op("pallas::test_output_projection_b_reference", reference_b)
    for op in (ref_a, ref_b):
        op.register_fake(lambda x, w, scale: x.new_empty((x.shape[0], w.shape[1])))

    def candidate(x, wa, sa, wb, sb):
        return quantized_matmul(csa2_adapter.project(x, wa, sa), wb, sb)

    def both(x, wa, sa, wb, sb):
        reference_a = ref_a(x, wa, sa)
        actual_a = csa2_adapter.project(x, wa, sa)
        return (
            reference_a,
            actual_a,
            ref_b(reference_a, wb, sb),
            quantized_matmul(actual_a, wb, sb),
        )

    gen = torch.Generator().manual_seed(4919)
    wa = torch.randn(4096, 8192, generator=gen).to(torch.float8_e4m3fn).to("tpu")
    wb = torch.randn(8192, 5120, generator=gen).to(torch.float8_e4m3fn).to("tpu")
    sa = torch.rand(8192, generator=gen).mul(0.025).add(0.005).to("tpu")
    sb = torch.rand(5120, generator=gen).mul(0.025).add(0.005).to("tpu")
    return (
        candidate,
        torch.compile(both, backend="tpu", fullgraph=True),
        (wa, sa, wb, sb),
    )


@pytest.mark.parametrize(
    "tokens,pattern",
    [
        (240, "random"),
        (288, "random"),
        (240, "zero"),
        (288, "zero"),
        (240, "mixed"),
        (288, "mixed"),
        (256, "random"),
        (2048, "random"),
    ],
)
def test_output_projection_bridge_is_bitwise_equal(projection_ops, tokens, pattern):
    from vllm_torchtpu.layers.adapter.linear_common import _quantized_matmul_jax
    from vllm_torchtpu.models.vllm.deepseek_v41.v2.csa2_adapter import _project

    _, compiled, weights = projection_ops
    x, wa, sa, wb, sb = _abstract_inputs(tokens)
    a_graph = str(jax.make_jaxpr(_project)(x, wa, sa))
    b_graph = str(
        jax.make_jaxpr(_quantized_matmul_jax)(
            jax.ShapeDtypeStruct((tokens, 8192), jnp.bfloat16), wb, sb
        )
    )
    assert ("wo_a_projection_buffered" in a_graph) is (tokens in (240, 288))
    assert ("buffered_per_channel_matmul" in b_graph) is (tokens in (240, 288))
    gen = torch.Generator().manual_seed(4919)
    activations = torch.randn(tokens, 64, 512, generator=gen).to(torch.bfloat16)
    if pattern == "zero":
        activations.zero_()
    elif pattern == "mixed":
        activations[:32] = 0
        activations[32:96] *= 0.0001
        activations[96:192] *= 100
    ref_a, actual_a, ref_chain, actual_chain = compiled(activations.to("tpu"), *weights)
    for reference, actual in ((ref_a, actual_a), (ref_chain, actual_chain)):
        reference, actual = reference.cpu(), actual.cpu()
        assert torch.isfinite(actual).all()
        assert torch.equal(reference.view(torch.int16), actual.view(torch.int16))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tile_t": 288},
        {"tile_r": 512},
        {"sub_t": 96},
        {"quantize_activations": False},
        {"cos_sin": jax.ShapeDtypeStruct((288, 256), jnp.float32)},
    ],
)
def test_output_projection_retains_explicit_and_rotary_paths(projection_ops, kwargs):
    del projection_ops
    x, w, s, _, _ = _abstract_inputs()
    options = dict(kwargs)
    cs = options.pop("cos_sin", None)
    graph = str(
        jax.make_jaxpr(lambda a, b, c, d: oa.wo_a_projection(a, b, c, d, **options))(
            x, w, s, cs
        )
    )
    assert "wo_a_projection_buffered" not in graph


def test_output_projection_export_preserves_dynamic_tokens(projection_ops):
    candidate, _, weights = projection_ops

    class Projection(torch.nn.Module):
        def forward(self, x, wa, sa, wb, sb):
            return candidate(x, wa, sa, wb, sb)

    tokens = torch.export.Dim("tokens", min=1, max=2048)
    inputs = (torch.zeros(288, 64, 512, dtype=torch.bfloat16, device="tpu"), *weights)
    exported = torch.export.export(
        Projection(), inputs, dynamic_shapes=({0: tokens}, None, None, None, None)
    )
    assert exported.range_constraints
