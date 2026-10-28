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
"""Mocked static resource boundaries; these are not cross-chip hardware tests."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import pytest
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.qwen4_exp.gated_residual import gated_residual_fn
from vllm_torchtpu.kernels.qwen4_exp.hc_full import _full_mix
from vllm_torchtpu.kernels.qwen4_exp.hc_resources import (
    RESIDUAL_PIPELINE,
    HcHardware,
    full_mix_memory,
    supports_full_mix,
    supports_residual,
)
from vllm_torchtpu.kernels.qwen4_exp.vllm_adapter import build_full_mix_op


def hardware(capacity=64 << 20):
    return HcHardware("mock-layout", capacity, 128, 16, 8, 256)


@pytest.mark.parametrize("capacity,expected", [(16, False), (64, True), (128, True)])
def test_full_residency_and_capacity(capacity, expected):
    info = hardware(capacity << 20)
    # 320 lowrank rows, but up-weight minor320 pads to384 vector columns.
    estimate = full_mix_memory(4, 2560, 320, 256, info)
    assert estimate.resident_bytes == 10240 * 2 + (320 + 384) * 10240 * 2
    assert estimate.buffer_bytes == (4 * 256 * 10240 + 2 * 256 * 2560) * 2
    assert supports_full_mix(4, 2560, 320, 256, info) is expected
    # 512-row rings plus the same working policy exceed64MiB, without an OOM.
    assert not supports_full_mix(4, 2560, 320, 512, hardware())


def test_exact_budget_boundary_and_cache_identity():
    info = hardware()
    needed = full_mix_memory(4, 2560, 320, 256, info).total_bytes
    assert not supports_full_mix(
        4, 2560, 320, 256, replace(info, vmem_bytes=needed - 1)
    )
    assert supports_full_mix(4, 2560, 320, 256, replace(info, vmem_bytes=needed))
    a = build_full_mix_op(4, 128, 1e-6, 128, info)
    assert build_full_mix_op(4, 128, 1e-6, 128, info) is a
    assert (
        build_full_mix_op(4, 128, 1e-6, 128, replace(info, vmem_bytes=128 << 20))
        is not a
    )
    assert build_full_mix_op(4, 128, 1e-6, 128, replace(info, chip="other")) is not a


@pytest.mark.parametrize(
    "branches,hidden,lowrank,tile",
    [
        (0, 128, 128, 128),
        (4, 0, 128, 128),
        (4, 129, 128, 128),
        (4, 128, 0, 128),
        (4, 128, 128, 0),
        (4, 128, 128, 64),
        (2**31, 128, 128, 128),
    ],
)
def test_static_invalid_or_overflow_geometry(branches, hidden, lowrank, tile):
    assert not supports_full_mix(branches, hidden, lowrank, tile, hardware())


def test_unknown_or_unqualified_layout():
    assert not supports_full_mix(4, 128, 128, 128, None)
    for info in [
        replace(hardware(), vmem_bytes=0),
        replace(hardware(), lanes=64),
        replace(hardware(), bf16_rows=0),
        replace(hardware(), mxu_columns=0),
    ]:
        assert not supports_full_mix(4, 128, 128, 128, info)


@pytest.mark.parametrize("tile", [16, 32, 64, 128])
def test_residual_alignment_is_dtype_derived(tile):
    assert supports_residual(3, 128, tile, hardware())
    assert supports_residual(4, 2560, tile, hardware())
    different_layout = replace(hardware(), bf16_rows=32, fp32_rows=16)
    assert supports_residual(3, 128, tile, different_layout) is (tile % 32 == 0)
    assert not supports_residual(3, 128, tile + 1, hardware())
    assert not supports_residual(3, 128, tile, None)


@pytest.mark.parametrize(
    "branches,hidden,tile,needed",
    [(3, 128, 16, 98304), (4, 256, 32, 425984), (129, 128, 16, 2179072)],
)
def test_residual_pipeline_budget_boundary(branches, hidden, tile, needed):
    # First fixture: two slots each of 12KiB residual input, 12KiB output,
    # 4KiB block and 8KiB padded FP32 gates; three 8KiB FP32 working planes.
    # The 129-branch fixture pads the gates to 256 columns independently.
    assert supports_residual(branches, hidden, tile, hardware(needed))
    assert not supports_residual(branches, hidden, tile, hardware(needed - 1))


def _pallas_params(fn, *args):
    from jax.extend import core

    def find(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "pallas_call":
                return eqn.params
            for value in eqn.params.values():
                if isinstance(value, core.ClosedJaxpr):
                    result = find(value.jaxpr)
                    if result is not None:
                        return result
        return None

    result = find(jax.make_jaxpr(fn)(*args).jaxpr)
    assert result is not None
    return result


@pytest.mark.parametrize("tiles", [1, 3])
def test_residual_compiler_cap_and_logical_cost(tiles):
    tokens = 16 * tiles
    shape = jax.ShapeDtypeStruct
    args = (
        shape((tokens, 384), jnp.bfloat16),
        shape((tokens, 128), jnp.bfloat16),
        shape((tokens, 3), jnp.float32),
    )
    params = _pallas_params(gated_residual_fn(3, 128, 16, hardware(98304)), *args)
    assert params["compiler_params"].vmem_limit_bytes == 98304
    mapping = params["grid_mapping"]
    assert mapping.grid == (tiles,)
    assert len(mapping.block_mappings) == 4
    assert all(
        bm.pipeline_mode is RESIDUAL_PIPELINE and not bm.has_trivial_window()
        for bm in mapping.block_mappings
    )
    assert isinstance(RESIDUAL_PIPELINE, pl.Buffered)
    assert RESIDUAL_PIPELINE.buffer_count == 2
    assert params["cost_estimate"] == pl.CostEstimate(
        flops=12288 * tiles, transcendentals=0, bytes_accessed=28864 * tiles
    )
    # Admission retains both slots even for a single tile, before any lowering.
    with pytest.raises(ValueError, match="VMEM budget"):
        jax.eval_shape(gated_residual_fn(3, 128, 16, hardware(98303)), *args)


@pytest.mark.parametrize(
    "buffers,needed,mode",
    [(1, 61440, "synchronous"), (2, 98304, "double_buffered"), (3, 135168, None)],
)
def test_residual_strategy_drives_budget_and_real_lowering(
    monkeypatch, buffers, needed, mode
):
    from jax import export
    from jax._src.pallas.mosaic import lowering

    from vllm_torchtpu.kernels.qwen4_exp import gated_residual, hc_resources

    strategy = pl.Buffered(buffers)
    monkeypatch.setattr(hc_resources, "RESIDUAL_PIPELINE", strategy)
    monkeypatch.setattr(gated_residual, "RESIDUAL_PIPELINE", strategy)
    assert supports_residual(3, 128, 16, hardware(needed))
    assert not supports_residual(3, 128, 16, hardware(needed - 1))
    original_lower = lowering.lower_jaxpr_to_pipelined_module
    modules = []

    def capture(*args, **kwargs):
        module = original_lower(*args, **kwargs)
        modules.append(str(module))
        return module

    monkeypatch.setattr(lowering, "lower_jaxpr_to_pipelined_module", capture)
    mesh = jax.sharding.AbstractMesh(
        (1,),
        ("test",),
        abstract_device=jax.sharding.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"
        ),
    )
    shape = jax.ShapeDtypeStruct
    args = (
        shape((48, 384), jnp.bfloat16),
        shape((48, 128), jnp.bfloat16),
        shape((48, 3), jnp.float32),
    )
    info = replace(hardware(needed), chip="mock-lowering")
    fn = jax.jit(gated_residual_fn(3, 128, 16, info))
    with jax.sharding.use_abstract_mesh(mesh):
        if mode is None:
            with pytest.raises(lowering.LoweringException, match="Only single.*double"):
                export.export(fn, platforms=("tpu",))(*args)
        else:
            export.export(fn, platforms=("tpu",))(*args)
            assert modules[0].count(f"pipeline_mode = #tpu.pipeline_mode<{mode}>") == 4


@pytest.mark.parametrize(
    "tokens,block_dtype,gate_dtype,gate_columns",
    [
        (0, jnp.bfloat16, jnp.float32, 3),
        (17, jnp.bfloat16, jnp.float32, 3),
        (16, jnp.float32, jnp.float32, 3),
        (16, jnp.bfloat16, jnp.bfloat16, 3),
        (16, jnp.bfloat16, jnp.float32, 4),
    ],
)
def test_residual_invalid_operands_fail_before_cost_or_lowering(
    monkeypatch, tokens, block_dtype, gate_dtype, gate_columns
):
    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid operands reached pallas_call")

    monkeypatch.setattr(pl, "pallas_call", forbidden)
    shape = jax.ShapeDtypeStruct
    with pytest.raises(ValueError, match="full BF16 tiles and FP32 gates"):
        jax.eval_shape(
            gated_residual_fn(3, 128, 16, hardware()),
            shape((tokens, 384), jnp.bfloat16),
            shape((tokens, 128), block_dtype),
            shape((tokens, gate_columns), gate_dtype),
        )


@pytest.mark.parametrize("tiles", [1, 3])
def test_full_mix_cost_counts_resident_weights_once(tiles):
    tokens = 128 * tiles
    shape = jax.ShapeDtypeStruct
    args = [
        shape(s, jnp.bfloat16) for s in [(tokens, 384), (384,), (64, 384), (384, 64)]
    ]
    params = _pallas_params(
        lambda *a: _full_mix(
            *a, branches=3, hidden=128, eps=1e-6, token_tile=128, hardware=hardware()
        ),
        *args,
    )
    # Two 64x384 GEMMs: 12,582,912 FLOPs per tile. RMS/affine:
    # 246,144; SiLU/division: 16,384; sigmoid-weighted branch sum: 98,304.
    # Resident BF16 weights: 99,072 bytes; x/norm/mixed: 229,376 per tile.
    assert params["cost_estimate"] == pl.CostEstimate(
        flops=12943744 * tiles,
        transcendentals=57728 * tiles,
        bytes_accessed=99072 + 229376 * tiles,
    )


def test_direct_resource_and_alignment_failure_before_lowering():
    info = hardware()
    shape = jax.ShapeDtypeStruct
    args = [
        shape(s, jnp.bfloat16)
        for s in [(512, 10240), (10240,), (320, 10240), (10240, 320)]
    ]
    with pytest.raises(ValueError, match="VMEM budget"):
        jax.eval_shape(
            lambda *a: _full_mix(
                *a, branches=4, hidden=2560, eps=1e-6, token_tile=512, hardware=info
            ),
            *args,
        )
    fn = gated_residual_fn(3, 128, 8, info)
    with pytest.raises(ValueError, match="packing"):
        jax.eval_shape(
            fn,
            shape((8, 384), jnp.bfloat16),
            shape((8, 128), jnp.bfloat16),
            shape((8, 3), jnp.float32),
        )


def test_fake_registration_never_queries_hardware(monkeypatch):
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as adapter

    def forbidden():
        raise AssertionError("Fake registration must not query/initialize a TPU")

    monkeypatch.setattr(adapter.pltpu, "get_tpu_info", forbidden)
    monkeypatch.setattr(adapter.pltpu, "is_tpu_device", forbidden)
    monkeypatch.setattr(torch.tpu, "is_initialized", lambda: False)
    assert adapter.current_hc_hardware() is None
    full = adapter.build_full_mix_op(3, 128, 1e-6, 128)
    residual = adapter.build_gated_residual_op(3, 128, 16)
    with FakeTensorMode():
        x = torch.empty(128, 384, dtype=torch.bfloat16)
        norm = torch.empty(384, dtype=torch.bfloat16)
        down = torch.empty(128, 384, dtype=torch.bfloat16)
        up = torch.empty(384, 128, dtype=torch.bfloat16)
        assert full(x, norm, down, up)[0].shape == (128, 128)
        assert (
            residual(
                x,
                torch.empty(128, 128, dtype=torch.bfloat16),
                torch.empty(128, 3, dtype=torch.float32),
            ).shape
            == x.shape
        )


@pytest.fixture
def initialized_tpu_metadata(monkeypatch):
    import torch
    from jax._src import tpu_info

    monkeypatch.setattr(torch.tpu, "is_initialized", lambda: True)
    tpu_info.get_tpu_info.cache_clear()
    yield tpu_info
    tpu_info.get_tpu_info.cache_clear()


def test_hardware_discovery_before_worker_initialization(monkeypatch):
    import torch

    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as adapter

    # Importing torch_tpu's Pallas bridge already registers this public module.
    assert "tpu" in torch.__dict__
    monkeypatch.setattr(torch.tpu, "is_initialized", lambda: False)

    def forbidden():
        raise AssertionError("Uninitialized workers must not query a device")

    monkeypatch.setattr(adapter.pltpu, "get_tpu_info", forbidden)
    monkeypatch.setattr(adapter.pltpu, "is_tpu_device", forbidden)
    monkeypatch.setattr(jax, "devices", forbidden)
    assert adapter.current_hc_hardware() is None


def test_initialized_unknown_device_keeps_torch_fallback(
    monkeypatch, initialized_tpu_metadata
):
    from jax._src import xla_bridge

    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as adapter

    was_initialized = xla_bridge.backends_are_initialized()
    monkeypatch.setattr(
        initialized_tpu_metadata, "get_device_kind", lambda: "HC unknown device fixture"
    )
    # The real public capability predicate returns False for an unknown kind.
    assert adapter.current_hc_hardware() is None
    assert xla_bridge.backends_are_initialized() == was_initialized


def test_initialized_known_device_preserves_facts_and_admission(
    monkeypatch, initialized_tpu_metadata
):
    from jax.experimental.pallas import tpu as pltpu

    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as adapter

    monkeypatch.setattr(initialized_tpu_metadata, "get_device_kind", lambda: "TPU7x")
    monkeypatch.setattr(initialized_tpu_metadata, "get_num_device_cores", lambda: 1)
    expected = HcHardware.from_tpu_info(
        pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    )
    actual = adapter.current_hc_hardware()
    assert actual == expected
    assert actual == HcHardware("7x", 64 << 20, 128, 16, 8, 256)
    assert supports_full_mix(4, 2560, 320, 256, actual)
    assert not supports_full_mix(4, 2560, 320, 512, actual)
    assert supports_residual(3, 128, 16, actual)
    assert not supports_residual(3, 128, 8, actual)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("backend unavailable"),
        ValueError("invalid layout"),
        ValueError("Unsupported TPU chip version: fixture"),
        ValueError("Unsupported TPU device kind: fixture"),
    ],
)
@pytest.mark.parametrize("query", ["is_tpu_device", "get_tpu_info"])
def test_hardware_discovery_propagates_other_errors(
    monkeypatch, initialized_tpu_metadata, error, query
):
    from vllm_torchtpu.kernels.qwen4_exp import vllm_adapter as adapter

    def fail():
        raise error

    monkeypatch.setattr(adapter.pltpu, "is_tpu_device", lambda: True)
    monkeypatch.setattr(adapter.pltpu, query, fail)
    with pytest.raises(type(error)) as caught:
        adapter.current_hc_hardware()
    assert caught.value is error
