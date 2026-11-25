# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
# Source provenance: inferact/k3-megakernel@30c9ba966b
#   tests/megakernel/kimi_k3/test_prefill_moe_dispatch.py
"""Prefill MoE dispatch, overflow HLO lowering, and metadata capacity checks."""

from __future__ import annotations

import base64
import inspect
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import topologies
from jax.experimental.mosaic.dialects import tpu as mosaic_tpu
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.mlir import ir as mlir_ir
from jaxlib.xla_client import hlo as xla_hlo

from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.kimi_k3.megakernel import config
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    DP2_GROUPS,
    HIDDEN_SIZE,
    OVERFLOW_WINDOW_TOKENS,
    PREFILL_WINDOWS,
    ROUTED_EXPERTS,
    TP_SIZE,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    load,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels import (
    prefill_megakernel as kernels,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe import (
    fused,
    indexed_ragged_gmm,
    moe_down_valu,
)
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.prefill_moe.moe_down_valu import (
    down_valu_compact,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm_torchtpu.models.kimi_k3.megakernel.common.protocols import (
        PallasRefProtocol,
    )

# Compact route-metadata prefix threshold in fused prefill MoE.
_COMPACT_PREFIX_CAPACITY = fused.COMPACT_METADATA_PREFIX_ROWS
# 8192 tokens * top-16 routed experts per token in Kimi-K3.
_TOP_K_EXPERTS = fused.TOP_K
_OVERFLOW_ROUTES = OVERFLOW_WINDOW_TOKENS * _TOP_K_EXPERTS
# EP16 local expert count per TP2 pair: 896 / 16 = 56.
_LOCAL_EXPERTS = ROUTED_EXPERTS // (TP_SIZE // DP2_GROUPS)
_MAX_SP_STRIPE = kernels.STRIPE
_SILENT_BF16_WINDOWS = PREFILL_WINDOWS[:3]
_RETIRED_BF16_WINDOWS = PREFILL_WINDOWS[3:]
_BF16_TEST_DEPTH = 2
_BF16_REFERENCE_CUSTOM_CALLS = 11
_PACKED_GU_WORDS = 448
_PACKED_GU_OUT = 3072
_PACKED_DOWN_WORDS = 192
_PACKED_DOWN_OUT = 3584
_SHARED_GU_OUT = 6144
_SHARED_DOWN_IN = 3072
_LATENT_DOWN_OUT = 4096
_LATENT_HALF_WIDTH = 1792
_GU_SCALE_BLOCKS = 28
_DOWN_SCALE_BLOCKS = 12
_SMEM_OVERFLOW_ROUTES = 262144
_SMEM_CAPACITY_TPU7X = 1048576
_UNALIGNED_ROWS = 17
_ALIGNED_ROWS = 128
_SUBLANE_ALIGN_ROWS = 16
_NON_8K_WINDOW = PREFILL_WINDOWS[3]
_EXPECTED_OVERFLOW_CUSTOM_CALL_COUNT = 8
_MIN_MOSAIC_BODY_BYTES = 1024
_RETIRED_MOE_MODULES = (
    "layer.py",
    "packed_layout.py",
    "routing_count_kernel.py",
    "routing_topk_kernel.py",
    "shared_ring_kernel.py",
    "streaming_ring.py",
)


@pytest.mark.parametrize("window", PREFILL_WINDOWS)
@pytest.mark.parametrize("reduce_dtype", ["float32", "bfloat16"])
def test_dispatch_by_window_and_reduce_dtype(
    monkeypatch: pytest.MonkeyPatch,
    window: int,
    reduce_dtype: str,
) -> None:
    """Verify FP32 dispatches to fused MoE and BF16 dispatches to reference MoE."""
    cfg = config.KernelConfig(moe_reduce_dtype=reduce_dtype)
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda _mesh, *, seq_len: ("reference", seq_len),
    )
    monkeypatch.setattr(
        fused,
        "make_indexed_moe_sp_ring",
        lambda _mesh, *, seq_len: ("fused", seq_len),
    )
    result = kernels.make_indexed_moe_sp_ring(cast("Mesh", None), seq_len=window)
    expected = "fused" if reduce_dtype == "float32" else "reference"
    assert result == (expected, window)


@pytest.mark.parametrize(
    "peak",
    [0, _COMPACT_PREFIX_CAPACITY, _COMPACT_PREFIX_CAPACITY + 1, _OVERFLOW_ROUTES],
)
def test_metadata_overflow_is_uniform_and_uncapped(
    monkeypatch: pytest.MonkeyPatch,
    peak: int,
) -> None:
    """Verify all ranks take the same compact or overflow branch from global peak."""

    def consume(
        *positional: jax.Array | None,
        stream_metadata: bool = False,
        metadata_row_capacity: int = 0,
        **_extra: object,
    ) -> jax.Array:
        """Report input capacity, metadata row slices, and streaming flag."""
        activations = positional[0]
        destinations = positional[4]
        probabilities = positional[5]
        assert activations is not None
        assert destinations is not None
        assert probabilities is not None
        return jnp.array(
            [
                activations.shape[0],
                destinations.shape[0],
                probabilities.shape[0],
                int(stream_metadata),
                metadata_row_capacity,
            ]
        )

    monkeypatch.setattr(moe_down_valu, "down_valu", consume)
    activations = jnp.zeros((_OVERFLOW_ROUTES, 1), dtype=jnp.bfloat16)
    routes = jnp.arange(_OVERFLOW_ROUTES, dtype=jnp.int32)
    dummy_weights = jnp.zeros((1,), dtype=jnp.uint4)
    dummy_scales = jnp.zeros((1,), dtype=jnp.uint8)
    dummy_tables = jnp.zeros((1,), dtype=jnp.int32)
    results = [
        down_valu_compact(
            activations,
            dummy_weights,
            dummy_scales,
            jnp.array([local_count]),
            routes,
            routes,
            dummy_tables,
            metadata_prefix_rows=_COMPACT_PREFIX_CAPACITY,
            tokens=OVERFLOW_WINDOW_TOKENS,
            slice_input=False,
            global_route_max=jnp.int32(peak),
        )
        for local_count in (0, 17, peak)
    ]
    for result in results[1:]:
        np.testing.assert_array_equal(result, results[0])
    compact = peak <= _COMPACT_PREFIX_CAPACITY
    expected_rows = _COMPACT_PREFIX_CAPACITY if compact else _OVERFLOW_ROUTES
    np.testing.assert_array_equal(
        results[0],
        [
            _OVERFLOW_ROUTES,
            expected_rows,
            expected_rows,
            int(not compact),
            _COMPACT_PREFIX_CAPACITY if compact else 0,
        ],
    )


def test_direct_fused_factory_rejects_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the direct fused prefill MoE factory rejects BF16 reduction."""
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    monkeypatch.setattr(config, "active", lambda: cfg)
    with pytest.raises(ValueError, match="FP32"):
        fused.make_indexed_moe_sp_ring(None, seq_len=4096)


@pytest.mark.parametrize("tokens", PREFILL_WINDOWS)
def test_fused_validity_matches_sequence_parallel_ownership(tokens: int) -> None:
    """Verify stripe validity mask matches sequence-parallel token ownership."""
    rows = tokens // TP_SIZE
    stripe = min(rows, _MAX_SP_STRIPE)
    for live in (0, 1, 17, tokens - 17, tokens):
        for rank in (0, 1, 15, TP_SIZE - 1):
            owned = np.arange(rank * stripe, tokens, TP_SIZE * stripe)[:, None]
            owned = (owned + np.arange(stripe)[None, :]).reshape(rows, 1)
            actual = fused.stripe_validity(rows, jnp.int32(live), jnp.int32(rank))
            np.testing.assert_array_equal(actual, owned < live)


def test_prefill_moe_8192_overflow_structural_hlo() -> None:
    """Verify 8192-token MoE overflow lowering emits valid structural HLO and MLIR."""
    moe_dir = Path(fused.__file__).resolve().parent
    for retired in _RETIRED_MOE_MODULES:
        assert not (moe_dir / retired).exists()
    assert callable(indexed_ragged_gmm.PackedRows)
    assert callable(fused.routed_experts_prepared)

    # Production 32-rank TPU v7x topology (2x2x4 = 32 cores).
    topology = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    mesh = Mesh(topology.devices, (AXIS_SP,))
    tables = kernels.ring_tables(list(mesh.devices.flat), tuple(range(TP_SIZE)))

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P(AXIS_SP), P(AXIS_SP), P(AXIS_SP), P()),
        out_specs=P(AXIS_SP),
        check_vma=False,
    )
    def overflow_fn(
        latent: jax.Array,
        routing: dict[str, jax.Array],
        expert_weights: dict[str, jax.Array],
        scalars: jax.Array,
    ) -> jax.Array:
        """Lower prepared 8192-token routed experts across the 32-rank mesh."""
        return fused.routed_experts_prepared(
            fused.PreparedRoutes(
                latent=latent[0],
                sorted_tokens=routing["token_ids"][0],
                sorted_probs=routing["probs"][0],
                counts=routing["counts"][0],
            ),
            fused.RoutedExpertWeights(
                gate_up=expert_weights["gu"][0],
                gate_up_scales=expert_weights["gus"][0],
                down=expert_weights["down"][0],
                down_scales=expert_weights["ds"][0],
                expert_layer=scalars[:1],
            ),
            reduce_tables=tables,
            global_route_max=scalars[1],
        )[None]

    sharded = NamedSharding(mesh, P(AXIS_SP))
    replicated = NamedSharding(mesh, P())
    lowered = overflow_fn.lower(
        jax.ShapeDtypeStruct(
            (TP_SIZE, OVERFLOW_WINDOW_TOKENS, 2, _LATENT_HALF_WIDTH),
            jnp.bfloat16,
            sharding=sharded,
        ),
        {
            "token_ids": jax.ShapeDtypeStruct(
                (TP_SIZE, _OVERFLOW_ROUTES), jnp.int32, sharding=sharded
            ),
            "probs": jax.ShapeDtypeStruct(
                (TP_SIZE, _OVERFLOW_ROUTES), jnp.float32, sharding=sharded
            ),
            "counts": jax.ShapeDtypeStruct(
                (TP_SIZE, _LOCAL_EXPERTS), jnp.int32, sharding=sharded
            ),
        },
        {
            "gu": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, _PACKED_DOWN_OUT, _PACKED_GU_OUT),
                jnp.uint4,
                sharding=sharded,
            ),
            "gus": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, _GU_SCALE_BLOCKS, _PACKED_GU_OUT),
                jnp.uint8,
                sharding=sharded,
            ),
            "down": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, 1536, _PACKED_DOWN_OUT),
                jnp.uint4,
                sharding=sharded,
            ),
            "ds": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, _DOWN_SCALE_BLOCKS, _PACKED_DOWN_OUT),
                jnp.uint8,
                sharding=sharded,
            ),
        },
        jax.ShapeDtypeStruct((2,), jnp.int32, sharding=replicated),
    )
    expected_out_shape = (TP_SIZE, OVERFLOW_WINDOW_TOKENS // TP_SIZE, _PACKED_DOWN_OUT)
    assert lowered.out_info.shape == expected_out_shape
    assert lowered.out_info.dtype == jnp.float32
    stablehlo_text = lowered.as_text()
    assert f'out_shardings=[<@mesh, [{{"{AXIS_SP}"}}, {{}}, {{}}]>]' in stablehlo_text
    assert "AllocateBuffer" not in stablehlo_text

    hlo_comp = lowered.compiler_ir(dialect="hlo")
    assert hlo_comp is not None
    hlo_module = hlo_comp.as_hlo_module()
    options = xla_hlo.HloPrintOptions.canonical()
    options.print_backend_config = True
    options.print_metadata = False
    canonical_text = hlo_module.to_string(options)
    assert "partition-id" not in canonical_text
    custom_calls = [
        line.strip()
        for line in canonical_text.splitlines()
        if "custom-call(" in line and "tpu_custom_call" in line
    ]
    assert len(custom_calls) == _EXPECTED_OVERFLOW_CUSTOM_CALL_COUNT, (
        f"Expected {_EXPECTED_OVERFLOW_CUSTOM_CALL_COUNT} tpu_custom_call ops, "
        f"got {len(custom_calls)}"
    )
    parsed_bodies = 0
    for match in re.finditer(r'"body"\s*:\s*"([^"]+)"', canonical_text):
        raw_bytes = base64.b64decode(match.group(1))
        if len(raw_bytes) <= _MIN_MOSAIC_BODY_BYTES:
            continue
        with mlir_ir.Context() as ctx:
            mosaic_tpu.register_dialect(ctx)
            ctx.allow_unregistered_dialects = True
            module = mlir_ir.Module.parse(raw_bytes)
            assert "func.func" in str(module.operation.get_asm(enable_debug_info=False))
        parsed_bodies += 1
    assert parsed_bodies >= 1


def _build_production_bf16_moe_families(
    mesh: Mesh,
    *,
    depth: int,
) -> dict[str, jax.ShapeDtypeStruct]:
    """Build production presharded MoE weight shapes for BF16 reference lowering."""
    sharded = NamedSharding(mesh, P(AXIS_SP))
    shapes = load.selected_local_shapes(layers=depth)
    overrides: dict[str, tuple[tuple[int, ...], jnp.dtype]] = {
        "latent_down": ((TP_SIZE, depth, HIDDEN_SIZE, _LATENT_DOWN_OUT), jnp.bfloat16),
        "latent_up": ((TP_SIZE, depth, _PACKED_DOWN_OUT, HIDDEN_SIZE), jnp.bfloat16),
        "shared_gu": ((TP_SIZE, depth, HIDDEN_SIZE, _SHARED_GU_OUT), jnp.bfloat16),
        "shared_down": ((TP_SIZE, depth, _SHARED_DOWN_IN, HIDDEN_SIZE), jnp.bfloat16),
        "expert_gu": (
            (TP_SIZE, depth, _LOCAL_EXPERTS, _PACKED_GU_WORDS, _PACKED_GU_OUT),
            jnp.uint32,
        ),
        "expert_gus": (
            (TP_SIZE, depth, _LOCAL_EXPERTS, _GU_SCALE_BLOCKS, _PACKED_GU_OUT),
            jnp.uint8,
        ),
        "expert_down": (
            (TP_SIZE, depth, _LOCAL_EXPERTS, _PACKED_DOWN_WORDS, _PACKED_DOWN_OUT),
            jnp.uint32,
        ),
        "expert_ds": (
            (TP_SIZE, depth, _LOCAL_EXPERTS, _DOWN_SCALE_BLOCKS, _PACKED_DOWN_OUT),
            jnp.uint8,
        ),
    }
    families: dict[str, jax.ShapeDtypeStruct] = {}
    for key in kernels.MOE_FAMILIES:
        if key in overrides:
            shape, dtype = overrides[key]
            families[key] = jax.ShapeDtypeStruct(shape, dtype, sharding=sharded)
        else:
            dtype = (
                jnp.float32 if key in ("router_bias", "latent_norm") else jnp.bfloat16
            )
            families[key] = jax.ShapeDtypeStruct(
                (TP_SIZE, *shapes[key]), dtype, sharding=sharded
            )
    return families


def test_prefill_bf16_warning_dedupes_via_real_logger(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify BF16 4K/8K retirement warnings deduplicate through the real logger."""
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    dummy_mesh = cast("Mesh", None)
    logger_suffix = f"{kernels.__name__}.bf16_dedupe_test"
    real_logger = init_logger(logger_suffix)
    resolved_logger_name = str(real_logger.name)
    target_logger = logging.getLogger(resolved_logger_name)
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(kernels, "logger", real_logger)
    monkeypatch.setattr(target_logger, "propagate", False)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda _mesh, *, seq_len: ("reference", seq_len),
    )

    target_logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=resolved_logger_name):
            caplog.clear()
            for silent_window in _SILENT_BF16_WINDOWS:
                assert kernels.make_indexed_moe_sp_ring(
                    dummy_mesh, seq_len=silent_window
                ) == ("reference", silent_window)
            assert not caplog.records

            for retired_window in (*_RETIRED_BF16_WINDOWS, *_RETIRED_BF16_WINDOWS):
                assert kernels.make_indexed_moe_sp_ring(
                    dummy_mesh, seq_len=retired_window
                ) == ("reference", retired_window)
    finally:
        target_logger.removeHandler(caplog.handler)

    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == len(_RETIRED_BF16_WINDOWS)
    assert any("seq_len=4096" in msg for msg in messages)
    assert any("seq_len=8192" in msg for msg in messages)


def test_prefill_moe_bf16_4k_8k_aot_lowering_and_routed_experts_rejects_non_8k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify BF16 4K/8K AOT lowering and routed_experts_prepared 8K-only contract."""
    sig = inspect.signature(fused.routed_experts_prepared)
    assert "reduce_dtype" not in sig.parameters

    stub = jnp.zeros((1,), dtype=jnp.int32)
    routes_4k = fused.PreparedRoutes(
        latent=jnp.zeros((_NON_8K_WINDOW, 2, 1), dtype=jnp.bfloat16),
        sorted_tokens=stub,
        sorted_probs=stub,
        counts=stub,
    )
    stub_weights = fused.RoutedExpertWeights(
        gate_up=stub,
        gate_up_scales=stub,
        down=stub,
        down_scales=stub,
    )
    with pytest.raises(ValueError, match=r"4096.*8192|8192.*4096"):
        fused.routed_experts_prepared(routes_4k, stub_weights, reduce_tables=stub)

    with pytest.raises(TypeError):
        cast("Callable[..., object]", fused.routed_experts_prepared)(
            routes_4k,
            stub_weights,
            reduce_tables=stub,
            reduce_dtype=jnp.bfloat16,
        )

    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    monkeypatch.setattr(config, "active", lambda: cfg)
    topology = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    mesh = Mesh(topology.devices, (AXIS_SP,))
    sharded = NamedSharding(mesh, P(AXIS_SP))
    replicated = NamedSharding(mesh, P())
    families = _build_production_bf16_moe_families(mesh, depth=_BF16_TEST_DEPTH)
    length = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)
    slot = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)

    for window in _RETIRED_BF16_WINDOWS:
        local_tokens = window // TP_SIZE
        hidden = jax.ShapeDtypeStruct(
            (TP_SIZE, local_tokens, HIDDEN_SIZE), jnp.bfloat16, sharding=sharded
        )
        prefix = jax.ShapeDtypeStruct(
            (TP_SIZE, local_tokens, HIDDEN_SIZE), jnp.bfloat16, sharding=sharded
        )
        moe_fn = kernels.make_indexed_moe_sp_ring(mesh, seq_len=window)
        lowered = moe_fn.lower(hidden, prefix, families, length, slot)
        assert lowered.out_info.shape == (TP_SIZE, local_tokens, HIDDEN_SIZE)
        assert lowered.out_info.dtype == jnp.bfloat16
        stablehlo_text = lowered.as_text()
        assert "AllocateBuffer" not in stablehlo_text
        custom_call_count = stablehlo_text.count("tpu_custom_call")
        assert custom_call_count == _BF16_REFERENCE_CUSTOM_CALLS, (
            f"Expected {_BF16_REFERENCE_CUSTOM_CALLS} tpu_custom_call ops at "
            f"seq_len={window}, got {custom_call_count}"
        )
        hlo_comp = lowered.compiler_ir(dialect="hlo")
        assert hlo_comp is not None
        options = xla_hlo.HloPrintOptions.canonical()
        options.print_backend_config = True
        options.print_metadata = False
        canonical_text = hlo_comp.as_hlo_module().to_string(options)
        assert "partition-id" not in canonical_text


def test_indexed_ragged_gmm_guards_report_offending_values() -> None:
    """Verify indexed_ragged_gmm guards report offending runtime shapes and sizes."""
    packed = indexed_ragged_gmm.PackedRows(
        cast(
            "PallasRefProtocol",
            jnp.zeros((_SUBLANE_ALIGN_ROWS, 2, 8), dtype=jnp.bfloat16),
        )
    )
    with pytest.raises(ValueError, match=r"16.*\(1, 16\)"):
        packed.reshape(1, _SUBLANE_ALIGN_ROWS)
    with pytest.raises(ValueError, match=r"16.*\(-1, 8\)"):
        packed.reshape(-1, 8)

    devices = topologies.get_topology_desc(
        topology_name="tpu7x_2x2x4", platform="tpu"
    ).devices
    mesh = Mesh(devices[:1], ("x",))
    replicated = NamedSharding(mesh, P())
    latent_bf16 = jax.ShapeDtypeStruct(
        (_NON_8K_WINDOW, 2, _LATENT_HALF_WIDTH), jnp.bfloat16, sharding=replicated
    )
    rhs = jax.ShapeDtypeStruct(
        (3, _LOCAL_EXPERTS, _PACKED_DOWN_OUT, _PACKED_GU_OUT),
        jnp.uint4,
        sharding=replicated,
    )
    scales = jax.ShapeDtypeStruct(
        (3, _LOCAL_EXPERTS, _GU_SCALE_BLOCKS, _PACKED_GU_OUT),
        jnp.uint8,
        sharding=replicated,
    )
    sizes = jax.ShapeDtypeStruct((_LOCAL_EXPERTS,), jnp.int32, sharding=replicated)
    layer = jax.ShapeDtypeStruct((1,), jnp.int32, sharding=replicated)

    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        with pytest.raises(
            ValueError,
            match=rf"{_UNALIGNED_ROWS}.*{_SUBLANE_ALIGN_ROWS}|{_SUBLANE_ALIGN_ROWS}.*{_UNALIGNED_ROWS}",
        ):
            jax.eval_shape(
                indexed_ragged_gmm.indexed_gate_up,
                latent_bf16,
                jax.ShapeDtypeStruct(
                    (_UNALIGNED_ROWS,), jnp.int32, sharding=replicated
                ),
                rhs,
                sizes,
                scales,
                layer,
            )

        with pytest.raises(ValueError, match="float32"):
            jax.eval_shape(
                indexed_ragged_gmm.indexed_gate_up,
                jax.ShapeDtypeStruct(
                    (_NON_8K_WINDOW, 2, _LATENT_HALF_WIDTH),
                    jnp.float32,
                    sharding=replicated,
                ),
                jax.ShapeDtypeStruct((_ALIGNED_ROWS,), jnp.int32, sharding=replicated),
                rhs,
                sizes,
                scales,
                layer,
            )

        with pytest.raises(ValueError, match="uint32"):
            jax.eval_shape(
                indexed_ragged_gmm.indexed_gate_up,
                latent_bf16,
                jax.ShapeDtypeStruct((_ALIGNED_ROWS,), jnp.uint32, sharding=replicated),
                rhs,
                sizes,
                scales,
                layer,
            )

        with pytest.raises(ValueError, match=rf"{_SMEM_CAPACITY_TPU7X}"):
            jax.eval_shape(
                indexed_ragged_gmm.indexed_gate_up,
                latent_bf16,
                jax.ShapeDtypeStruct(
                    (_SMEM_OVERFLOW_ROUTES,), jnp.int32, sharding=replicated
                ),
                rhs,
                sizes,
                scales,
                layer,
            )
