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

import hashlib
import re
import struct
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.xla_client import hlo as xla_hlo

from vllm_torchtpu.models.kimi_k3.megakernel import config
from vllm_torchtpu.models.kimi_k3.megakernel.common.constants import (
    AXIS_SP,
    DP2_GROUPS,
    OVERFLOW_WINDOW_TOKENS,
    PREFILL_WINDOWS,
    ROUTED_EXPERTS,
    TP_SIZE,
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
_EXPECTED_OVERFLOW_HLO_SHA256_U64 = (
    0x03EF631B95F1192F,
    0x404DC534B728AA94,
    0xDF79F9AB379DE0A3,
    0x8731CF7FCF0033F5,
)
_EXPECTED_OVERFLOW_HLO_BYTES = 34758
_EXPECTED_OVERFLOW_CUSTOM_CALL_COUNT = 8
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
    results = [
        down_valu_compact(
            activations,
            None,
            None,
            jnp.array([local_count]),
            routes,
            routes,
            None,
            metadata_prefix_rows=_COMPACT_PREFIX_CAPACITY,
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


def test_prefill_moe_bf16_reduce_dispatches_to_reference_with_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify BF16 at 4K/8K logs via warning_once and routes to reference MoE."""
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    warned_windows: list[int] = []
    dummy_mesh = cast("Mesh", None)
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda _mesh, *, seq_len: ("reference", seq_len),
    )
    monkeypatch.setattr(
        kernels.logger,
        "warning_once",
        lambda _msg, seq_len: warned_windows.append(int(seq_len)),
    )
    for silent_window in _SILENT_BF16_WINDOWS:
        assert kernels.make_indexed_moe_sp_ring(dummy_mesh, seq_len=silent_window) == (
            "reference",
            silent_window,
        )
    assert not warned_windows
    for retired_window in (*_RETIRED_BF16_WINDOWS, *_RETIRED_BF16_WINDOWS):
        assert kernels.make_indexed_moe_sp_ring(dummy_mesh, seq_len=retired_window) == (
            "reference",
            retired_window,
        )
    assert warned_windows == [4096, 8192, 4096, 8192]


def test_prefill_bf16_warning_uses_logger_warning_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify BF16 fallback delegates deduplication to logger.warning_once."""
    assert "_WARNED_BF16_WINDOWS" not in vars(kernels)
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    once_windows: list[int] = []
    raw_warnings: list[object] = []
    dummy_mesh = cast("Mesh", None)
    monkeypatch.setattr(config, "active", lambda: cfg)
    monkeypatch.setattr(
        kernels,
        "_make_indexed_moe_sp_ring_reference",
        lambda _mesh, *, seq_len: ("reference", seq_len),
    )
    monkeypatch.setattr(
        kernels.logger,
        "warning_once",
        lambda _msg, seq_len: once_windows.append(int(seq_len)),
    )
    monkeypatch.setattr(
        kernels.logger,
        "warning",
        lambda *args: raw_warnings.append(args),
    )
    assert kernels.make_indexed_moe_sp_ring(dummy_mesh, seq_len=4096) == (
        "reference",
        4096,
    )
    assert kernels.make_indexed_moe_sp_ring(dummy_mesh, seq_len=8192) == (
        "reference",
        8192,
    )
    assert once_windows == list(_RETIRED_BF16_WINDOWS)
    assert not raw_warnings


def test_routed_experts_prepared_forwards_reduce_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify routed_experts_prepared forwards explicit BF16 reduce_dtype at 4K/8K."""
    seen: list[tuple[int, jnp.dtype]] = []
    stub_arr = jnp.zeros((1,), dtype=jnp.int32)
    monkeypatch.setattr(
        fused,
        "indexed_gate_up",
        lambda latent, *_args: jnp.zeros((latent.shape[0], 1), dtype=jnp.bfloat16),
    )
    monkeypatch.setattr(
        fused,
        "down_valu",
        lambda *_args, **kw: (
            seen.append((int(kw["tokens"]), jnp.dtype(kw["ring_dtype"]))),
            stub_arr,
        )[1],
    )
    monkeypatch.setattr(
        fused,
        "down_valu_compact",
        lambda *_args, **kw: (
            seen.append((int(kw["tokens"]), jnp.dtype(kw["ring_dtype"]))),
            stub_arr,
        )[1],
    )
    for window in _RETIRED_BF16_WINDOWS:
        fused.routed_experts_prepared(
            fused.PreparedRoutes(
                latent=jnp.zeros((window, 2, 1), dtype=jnp.bfloat16),
                sorted_tokens=stub_arr,
                sorted_probs=stub_arr,
                counts=stub_arr,
            ),
            fused.RoutedExpertWeights(
                gate_up=stub_arr,
                gate_up_scales=stub_arr,
                down=stub_arr,
                down_scales=stub_arr,
            ),
            reduce_tables=stub_arr,
            reduce_dtype=jnp.bfloat16,
        )
    assert seen == [(4096, jnp.dtype(jnp.bfloat16)), (8192, jnp.dtype(jnp.bfloat16))]


def test_prefill_moe_8192_overflow_canonical_hlo_sha256() -> None:
    """Verify 8192-token MoE overflow lowering matches canonical HLO SHA256."""
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
            (TP_SIZE, OVERFLOW_WINDOW_TOKENS, 2, 1792), jnp.bfloat16, sharding=sharded
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
                (TP_SIZE, 3, _LOCAL_EXPERTS, 3584, 3072), jnp.uint4, sharding=sharded
            ),
            "gus": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, 28, 3072), jnp.uint8, sharding=sharded
            ),
            "down": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, 1536, 3584), jnp.uint4, sharding=sharded
            ),
            "ds": jax.ShapeDtypeStruct(
                (TP_SIZE, 3, _LOCAL_EXPERTS, 12, 3584), jnp.uint8, sharding=sharded
            ),
        },
        jax.ShapeDtypeStruct((2,), jnp.int32, sharding=replicated),
    )
    hlo_comp = lowered.compiler_ir(dialect="hlo")
    assert hlo_comp is not None
    hlo_module = hlo_comp.get_hlo_module()
    options = xla_hlo.HloPrintOptions.canonical()
    options.print_backend_config = True
    options.print_metadata = False
    canonical_text = hlo_module.to_string(options)
    custom_calls = [
        line.strip()
        for line in canonical_text.splitlines()
        if "custom-call(" in line and "tpu_custom_call" in line
    ]
    assert len(custom_calls) == _EXPECTED_OVERFLOW_CUSTOM_CALL_COUNT
    normalized_hlo = re.sub(r'"body"\s*:\s*"[^"]*"', '"body": ""', canonical_text)
    canonical_bytes = normalized_hlo.encode("utf-8")
    assert len(canonical_bytes) == _EXPECTED_OVERFLOW_HLO_BYTES
    assert (
        struct.unpack(">4Q", hashlib.sha256(canonical_bytes).digest())
        == _EXPECTED_OVERFLOW_HLO_SHA256_U64
    )
