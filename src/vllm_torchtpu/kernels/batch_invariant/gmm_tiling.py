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
"""Batch-invariant tile selection for gmm_v2 grouped matmuls.

A gmm_v2 output row depends only on tile_k: tile_m, tile_n, the group size and
the row offset leave it bit identical. calculate_tiling shrinks tile_k once
tile_m grows with the row count, so a token's expert output changed with the
batch. This picks tile_k as calculate_tiling would for an unbounded row count
and keeps the M dependent tile_m and tile_n for speed.
"""

from __future__ import annotations

import dataclasses

from vllm_torchtpu.kernels.megablox.gmm_v2 import (
    Dimensions,
    InputConfigs,
    TileSizes,
    calculate_tiling,
)

# Larger than any step's routed rows; tile_m saturates at 4 buckets far below it.
_UNBOUNDED_M = 1 << 20
# Prefill retune of tile_m and bucket_base; both leave every row bit identical.
_PREFILL_M = 1024
_SMALL_TILE_MAX_M = 16384
_MIN_BUCKET = 128


def batch_invariant_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """calculate_tiling with tile_k fixed by the weight shape alone."""
    actual = calculate_tiling(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
    big = calculate_tiling(
        dataclasses.replace(dims, size_m=_UNBOUNDED_M),
        lhs_cfgs,
        rhs_cfgs,
        vmem_limit_bytes,
        fuse_act,
    )
    # big.tile_k <= actual.tile_k, so the VMEM estimate only drops.
    tiles = dataclasses.replace(actual, tile_k=big.tile_k)
    if dims.size_m > _PREFILL_M and tiles.tile_m >= _MIN_BUCKET:
        # Prefill: EP leaves about 1/8 of the routed rows valid, so per expert
        # groups are small. 128 row buckets (and 128 row tiles up to 16K rows)
        # waste fewer MXU passes than 64 row buckets of a 256 row tile. Round
        # tile_m down to a multiple of _MIN_BUCKET so e.g. 192 never pairs with
        # bucket_base=128.
        tm = (
            _MIN_BUCKET
            if dims.size_m <= _SMALL_TILE_MAX_M
            else (tiles.tile_m // _MIN_BUCKET) * _MIN_BUCKET
        )
        tiles = dataclasses.replace(tiles, tile_m=tm, bucket_base=_MIN_BUCKET)
    return tiles
