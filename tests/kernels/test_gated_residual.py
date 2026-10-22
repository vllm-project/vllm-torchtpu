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
"""FP32 scalar gates must retain their precision between BF16 boundaries."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.kernels.qwen4_exp import gated_residual
from vllm_torchtpu.kernels.qwen4_exp.hc_resources import HcHardware
from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_gated_residual_op


@pytest.mark.parametrize(
    "branches,hidden,tile",
    [
        (0, 128, 128),
        (-1, 128, 128),
        (4, 0, 128),
        (4, 129, 128),
        (4, 128, 0),
        (4, 128, -1),
    ],
)
def test_invalid_geometry(branches, hidden, tile):
    with pytest.raises(ValueError):
        build_gated_residual_op(branches, hidden, tile)


_RESIDUAL_LAYOUT = HcHardware("7x", 64 << 20, 128, 16, 8, 256)


@pytest.mark.parametrize("tile", [16, 32, 64])
def test_small_tile_factory_and_abstract_admission(monkeypatch, tile):
    """Exercise the CPU factory contract, not TPU lowering or numerics."""
    calls = []

    def forbidden(*args, **kwargs):
        pytest.fail("CPU factory attempted hardware discovery")

    def pallas_boundary(kernel, **kwargs):
        calls.append(kwargs)
        return lambda residual, block, gates: residual

    monkeypatch.setattr(gated_residual.pltpu, "get_tpu_info", forbidden)
    monkeypatch.setattr(gated_residual.pl, "pallas_call", pallas_boundary)
    op = build_gated_residual_op(4, 128, tile, _RESIDUAL_LAYOUT)
    assert not calls
    with FakeTensorMode():
        residual = torch.empty(2 * tile, 512, dtype=torch.bfloat16)
        block = torch.empty(2 * tile, 128, dtype=torch.bfloat16)
        gates = torch.empty(2 * tile, 4, dtype=torch.float32)
        output = op(residual, block, gates)
        assert output.shape == residual.shape
        assert output.dtype == residual.dtype
        assert output.device == residual.device
    assert not calls
    shapes = (
        jax.ShapeDtypeStruct((2 * tile, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((2 * tile, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((2 * tile, 4), jnp.float32),
    )
    result = jax.eval_shape(
        gated_residual.gated_residual_fn(4, 128, tile, _RESIDUAL_LAYOUT), *shapes
    )
    assert result.shape == shapes[0].shape and result.dtype == shapes[0].dtype
    assert len(calls) == 1
    assert calls[0]["grid"] == (2,)
    assert calls[0]["out_shape"].shape == shapes[0].shape


@pytest.mark.parametrize(
    "tile,layout",
    [
        (8, _RESIDUAL_LAYOUT),
        (64, dataclasses.replace(_RESIDUAL_LAYOUT, lanes=64)),
        (64, dataclasses.replace(_RESIDUAL_LAYOUT, vmem_bytes=1)),
    ],
    ids=["sublane-packing", "unsupported-lanes", "vmem-budget"],
)
def test_small_tile_keeps_layout_and_resource_guards(monkeypatch, tile, layout):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid layout reached hardware discovery or Pallas")

    monkeypatch.setattr(gated_residual.pltpu, "get_tpu_info", forbidden)
    monkeypatch.setattr(gated_residual.pl, "pallas_call", forbidden)
    shapes = (
        jax.ShapeDtypeStruct((2 * tile, 512), jnp.bfloat16),
        jax.ShapeDtypeStruct((2 * tile, 128), jnp.bfloat16),
        jax.ShapeDtypeStruct((2 * tile, 4), jnp.float32),
    )
    with pytest.raises(ValueError, match="packing and VMEM"):
        jax.eval_shape(gated_residual.gated_residual_fn(4, 128, tile, layout), *shapes)


TPU = pytest.mark.tpu_test


@pytest.fixture(autouse=True)
def isolated_compile_cache(request):
    # Each case builds a different op/shape; do not exhaust Dynamo's shared
    # wrapper recompile limit across independent tests. Repeats inside a case
    # deliberately keep the same graph and runtime state.
    if request.node.get_closest_marker("tpu_test"):
        torch.compiler.reset()
    yield


@TPU
@pytest.mark.parametrize(
    "branches,hidden,tokens,tile",
    [(c, h, 3 * t, t) for c in (3, 4) for h in (128, 2560) for t in (16, 32, 64, 128)]
    + [
        (2, 128, 256, 128),
        (4, 256, 512, 256),
        (4, 2560, 512, 256),
        (4, 2560, 8192, 256),
    ],
)
def test_materialized_operands_match_fp32_reference(branches, hidden, tokens, tile):
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import sync

    generator = torch.Generator(device="cpu").manual_seed(913)
    x = torch.randn(
        tokens, branches * hidden, generator=generator, device="cpu"
    ).bfloat16()
    block = torch.randn(tokens, hidden, generator=generator, device="cpu").bfloat16()
    gates = torch.rand(tokens, branches, generator=generator, device="cpu") * 2
    # Independent broadcast reference, including negative and zero gates.
    gates[0] = torch.arange(branches, device="cpu") - 1
    expected = (
        (
            x.reshape(tokens, branches, hidden).float()
            + block.float()[:, None, :] * gates[:, :, None]
        )
        .bfloat16()
        .flatten(1)
    )
    op = build_gated_residual_op(branches, hidden, tile)
    run = torch.compile(
        op, backend=tpu_compile.TpuBackend(), fullgraph=True, dynamic=False
    )
    actual = run(x.to("tpu"), block.to("tpu"), gates.to("tpu"))
    sync.synchronize([actual], wait=True)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    repeated = run(x.to("tpu"), block.to("tpu"), gates.to("tpu"))
    sync.synchronize([repeated], wait=True)
    assert torch.isfinite(repeated.cpu()).all()
    torch.testing.assert_close(repeated.cpu(), actual.cpu(), rtol=0, atol=0)


@TPU
def test_gate_precision_and_zero_identity():
    from torch_tpu._internal import compile as tpu_compile
    from torch_tpu._internal import sync

    x = torch.full((256, 4 * 128), -1, dtype=torch.bfloat16, device="cpu")
    block = torch.ones((256, 128), dtype=torch.bfloat16, device="cpu")
    gates = torch.tensor([0, 1, 1 + 2**-10, 2], device="cpu").repeat(256, 1)
    expected = torch.tensor([-1, 0, 2**-10, 1], device="cpu").bfloat16()
    expected = expected.repeat_interleave(128).repeat(256, 1)
    op = build_gated_residual_op(4, 128, 128)
    run = torch.compile(
        op, backend=tpu_compile.TpuBackend(), fullgraph=True, dynamic=False
    )
    actual = run(x.to("tpu"), block.to("tpu"), gates.to("tpu"))
    sync.synchronize([actual], wait=True)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
