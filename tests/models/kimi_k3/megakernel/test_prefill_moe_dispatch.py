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
"""Prefill MoE dispatch, overflow HLO lowering, and metadata capacity checks."""

import hashlib
import re
import struct
import unittest
from pathlib import Path
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import topologies
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.xla_client import hlo as xla_hlo

from vllm_torchtpu.models.kimi_k3.megakernel import config
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

_WINDOWS = (512, 1024, 2048, 4096, 8192)
_COMPACT_PREFIX_CAPACITY = 32768
_OVERFLOW_ROUTES = 131072
_OVERFLOW_WINDOW_TOKENS = 8192
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


def test_dispatch_by_window_and_reduce_dtype() -> None:
    """Verify FP32 dispatches to fused MoE and BF16 dispatches to reference MoE."""
    for window in _WINDOWS:
        for reduce_dtype in ("float32", "bfloat16"):
            cfg = config.KernelConfig(moe_reduce_dtype=reduce_dtype)
            with (
                mock.patch.object(config, "active", return_value=cfg),
                mock.patch.object(
                    kernels,
                    "_make_indexed_moe_sp_ring_reference",
                    side_effect=lambda _mesh, *, seq_len: ("reference", seq_len),
                ),
                mock.patch.object(
                    fused,
                    "make_indexed_moe_sp_ring",
                    side_effect=lambda _mesh, *, seq_len: ("fused", seq_len),
                ),
            ):
                result = kernels.make_indexed_moe_sp_ring(None, seq_len=window)
                expected = "fused" if reduce_dtype == "float32" else "reference"
                assert result == (expected, window)


def test_metadata_overflow_is_uniform_and_uncapped() -> None:
    """Verify all ranks take the same compact or overflow branch from global peak."""

    def consume(
        activations: jax.Array,
        _rhs: object,
        _scales: object,
        _counts: jax.Array,
        destinations: jax.Array,
        probabilities: jax.Array,
        _tables: object,
        *,
        stream_metadata: bool = False,
        metadata_row_capacity: int = 0,
        **_extra: object,
    ) -> jax.Array:
        """Report input capacity, metadata row slices, and streaming flag."""
        return jnp.array(
            [
                activations.shape[0],
                destinations.shape[0],
                probabilities.shape[0],
                int(stream_metadata),
                metadata_row_capacity,
            ]
        )

    activations = jnp.zeros((_OVERFLOW_ROUTES, 1), dtype=jnp.bfloat16)
    routes = jnp.arange(_OVERFLOW_ROUTES, dtype=jnp.int32)
    with mock.patch.object(moe_down_valu, "down_valu", side_effect=consume):
        for peak in (0, 32768, 32769, 131072):
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


def test_direct_fused_factory_rejects_bf16() -> None:
    """Verify the direct fused prefill MoE factory rejects BF16 reduction."""
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    case = unittest.TestCase()
    with (
        mock.patch.object(config, "active", return_value=cfg),
        case.assertRaisesRegex(ValueError, "FP32"),
    ):
        fused.make_indexed_moe_sp_ring(None, seq_len=4096)


def test_fused_validity_matches_sequence_parallel_ownership() -> None:
    """Verify stripe validity mask matches sequence-parallel token ownership."""
    for tokens in _WINDOWS:
        rows = tokens // 32
        stripe = min(rows, 64)
        for live in (0, 1, 17, tokens - 17, tokens):
            for rank in (0, 1, 15, 31):
                owned = np.arange(rank * stripe, tokens, 32 * stripe)[:, None]
                owned = (owned + np.arange(stripe)[None, :]).reshape(rows, 1)
                actual = fused.stripe_validity(rows, jnp.int32(live), jnp.int32(rank))
                np.testing.assert_array_equal(actual, owned < live)


def test_prefill_moe_bf16_reduce_dispatches_to_reference_with_warning() -> None:
    """Verify BF16 at 4K/8K logs a one-time notice and routes to reference MoE."""
    cfg = config.KernelConfig(moe_reduce_dtype="bfloat16")
    warned: list[int] = []
    with (
        mock.patch.object(config, "active", return_value=cfg),
        mock.patch.object(
            kernels,
            "_make_indexed_moe_sp_ring_reference",
            side_effect=lambda _mesh, *, seq_len: ("reference", seq_len),
        ),
        mock.patch.object(kernels, "_WARNED_BF16_WINDOWS", set()),
        mock.patch.object(
            kernels.logger,
            "warning",
            side_effect=lambda _msg, seq_len: warned.append(seq_len),
        ),
    ):
        for silent_window in (512, 1024, 2048):
            assert kernels.make_indexed_moe_sp_ring(None, seq_len=silent_window) == (
                "reference",
                silent_window,
            )
        assert not warned
        for retired_window in (4096, 8192, 4096, 8192):
            assert kernels.make_indexed_moe_sp_ring(None, seq_len=retired_window) == (
                "reference",
                retired_window,
            )
        assert warned == [4096, 8192]


def test_routed_experts_prepared_forwards_reduce_dtype() -> None:
    """Verify routed_experts_prepared forwards explicit BF16 reduce_dtype at 4K/8K."""
    seen: list[tuple[int, jnp.dtype]] = []
    stub_arr = jnp.zeros((1,), dtype=jnp.int32)
    with (
        mock.patch.object(
            fused,
            "indexed_gate_up",
            side_effect=lambda latent, *_args: jnp.zeros(
                (latent.shape[0], 1), dtype=jnp.bfloat16
            ),
        ),
        mock.patch.object(
            fused,
            "down_valu",
            side_effect=lambda *_args, **kw: (
                seen.append((int(kw["tokens"]), jnp.dtype(kw["ring_dtype"]))),
                stub_arr,
            )[1],
        ),
        mock.patch.object(
            fused,
            "down_valu_compact",
            side_effect=lambda *_args, **kw: (
                seen.append((int(kw["tokens"]), jnp.dtype(kw["ring_dtype"]))),
                stub_arr,
            )[1],
        ),
    ):
        for window in (4096, 8192):
            fused.routed_experts_prepared(
                jnp.zeros((window, 2, 1), dtype=jnp.bfloat16),
                stub_arr,
                stub_arr,
                stub_arr,
                stub_arr,
                stub_arr,
                stub_arr,
                stub_arr,
                reduce_tables=stub_arr,
                reduce_dtype=jnp.bfloat16,
            )
    assert seen == [(4096, jnp.dtype(jnp.bfloat16)), (8192, jnp.dtype(jnp.bfloat16))]


def test_prefill_moe_8192_overflow_canonical_hlo_sha256() -> None:
    """Verify 8192-token MoE overflow lowering matches canonical HLO SHA256."""
    moe_dir = Path(fused.__file__).resolve().parent
    for retired in _RETIRED_MOE_MODULES:
        assert not (moe_dir / retired).exists()
    assert hasattr(indexed_ragged_gmm, "PackedRows")
    assert hasattr(fused, "routed_experts_prepared")

    topology = topologies.get_topology_desc(topology_name="tpu7x_2x2x4", platform="tpu")
    mesh = Mesh(topology.devices, ("k3_sp",))
    tables = kernels.ring_tables(list(mesh.devices.flat), tuple(range(32)))

    @jax.jit
    @jax.shard_map(
        mesh=mesh,
        in_specs=(P("k3_sp"), P("k3_sp"), P("k3_sp"), P()),
        out_specs=P("k3_sp"),
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
            latent[0],
            routing["token_ids"][0],
            routing["probs"][0],
            routing["counts"][0],
            expert_weights["gu"][0],
            expert_weights["gus"][0],
            expert_weights["down"][0],
            expert_weights["ds"][0],
            reduce_tables=tables,
            expert_layer=scalars[:1],
            global_route_max=scalars[1],
        )[None]

    sharded = NamedSharding(mesh, P("k3_sp"))
    replicated = NamedSharding(mesh, P())
    lowered = overflow_fn.lower(
        jax.ShapeDtypeStruct(
            (32, _OVERFLOW_WINDOW_TOKENS, 2, 1792), jnp.bfloat16, sharding=sharded
        ),
        {
            "token_ids": jax.ShapeDtypeStruct(
                (32, _OVERFLOW_WINDOW_TOKENS * 16), jnp.int32, sharding=sharded
            ),
            "probs": jax.ShapeDtypeStruct(
                (32, _OVERFLOW_WINDOW_TOKENS * 16), jnp.float32, sharding=sharded
            ),
            "counts": jax.ShapeDtypeStruct((32, 56), jnp.int32, sharding=sharded),
        },
        {
            "gu": jax.ShapeDtypeStruct(
                (32, 3, 56, 3584, 3072), jnp.uint4, sharding=sharded
            ),
            "gus": jax.ShapeDtypeStruct(
                (32, 3, 56, 28, 3072), jnp.uint8, sharding=sharded
            ),
            "down": jax.ShapeDtypeStruct(
                (32, 3, 56, 1536, 3584), jnp.uint4, sharding=sharded
            ),
            "ds": jax.ShapeDtypeStruct(
                (32, 3, 56, 12, 3584), jnp.uint8, sharding=sharded
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
