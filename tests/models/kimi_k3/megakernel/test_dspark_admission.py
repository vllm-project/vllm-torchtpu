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
"""Admission rules of the fused MLA DSpark draft kernel."""

from __future__ import annotations

import importlib
from typing import Any

import jax.numpy as jnp

from vllm_torchtpu.models.kimi_k3.megakernel.kernels import dspark, dspark_mla_kernel
from vllm_torchtpu.models.kimi_k3.megakernel.kernels.dspark_mla import MlaDraftConfig

pytest: Any = importlib.import_module("pytest")
PAGES = 16


@pytest.mark.parametrize(
    ("geometry", "error"),
    [
        ({"requests": 1}, None),
        ({"requests": 8}, None),
        ({"requests": 16, "rows": 4, "block": 4}, None),
        ({"requests": 1, "rows": 6}, "4 or 8 rows"),
        ({"requests": 1, "block": 1}, "block must be"),
        ({"requests": 9}, "up to 8 requests"),
        ({"requests": 3, "attention_dp": 2}, "attention DP2"),
    ],
)
def test_fused_draft_admission(geometry: dict[str, int], error: str | None) -> None:
    """Supported geometries build a kernel; others raise naming the limit."""
    if error is None:
        kernel = dspark_mla_kernel.fused_draft_local(
            MlaDraftConfig(), pages=PAGES, **geometry
        )
        assert callable(kernel)
    else:
        with pytest.raises(ValueError, match=error):
            dspark_mla_kernel.fused_draft_local(
                MlaDraftConfig(), pages=PAGES, **geometry
            )


_RING_BLOCKS = 15
_RING_TOTAL_LENGTH = 20
_RING_FIRST_VALID = 12
_SMALL_RING_WINDOW = 16
_SMALL_RING_LAYERS = 2
_SMALL_RING_HEAD_DIM = 8


def test_dspark_ring_pack_unpack_and_positions_round_trip() -> None:
    """Verify DSpark ring packing, magic validation, slot positions, and overflow."""
    cfg = dspark.DraftConfig(
        layers=_SMALL_RING_LAYERS,
        window=_SMALL_RING_WINDOW,
        head_dim=_SMALL_RING_HEAD_DIM,
    )
    ring_shape = (1, cfg.layers, cfg.window, 2 * cfg.head_dim)
    ring = jnp.arange(
        cfg.layers * cfg.window * 2 * cfg.head_dim, dtype=jnp.bfloat16
    ).reshape(ring_shape)
    meta = jnp.array([[_RING_TOTAL_LENGTH, _RING_FIRST_VALID]], dtype=jnp.int32)

    packed = dspark.pack_ring(ring, meta, blocks=_RING_BLOCKS)
    unpacked_ring, unpacked_meta = dspark.unpack_ring(
        cfg, packed[0], length=_RING_TOTAL_LENGTH
    )
    assert jnp.array_equal(unpacked_ring, ring[0])
    assert tuple(int(x) for x in unpacked_meta) == (
        _RING_TOTAL_LENGTH,
        _RING_FIRST_VALID,
    )

    _, mismatch_meta = dspark.unpack_ring(cfg, packed[0], length=_RING_TOTAL_LENGTH + 1)
    assert tuple(int(x) for x in mismatch_meta) == (
        _RING_TOTAL_LENGTH + 1,
        _RING_TOTAL_LENGTH + 1,
    )

    positions = dspark.ring_positions(cfg, unpacked_meta)
    valid_positions = [int(p) for p in positions if int(p) >= 0]
    assert sorted(valid_positions) == list(range(_RING_FIRST_VALID, _RING_TOTAL_LENGTH))

    full_cfg = dspark.DraftConfig()
    huge_ring = jnp.zeros(
        (1, full_cfg.layers, full_cfg.window, 2 * full_cfg.head_dim),
        dtype=jnp.bfloat16,
    )
    with pytest.raises(ValueError, match="a ring needs"):
        dspark.pack_ring(huge_ring, meta, blocks=1)
