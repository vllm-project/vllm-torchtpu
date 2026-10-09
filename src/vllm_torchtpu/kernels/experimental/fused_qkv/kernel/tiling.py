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

"""Tile sizes for the fused kernels.

choose_tiling() picks the largest token tile that fits VMEM and still leaves
MIN_TILES tiles: the first tile's loads and the last tile's writeback can't
overlap compute, and with 4 or more tiles they stay a small share of the run.

The rule is simple on purpose; this sweep shows its gap to the fastest tile.
Kernel us on TPU7x (benchmark.py test_tile_sweep, 2026-10-06; -: OOM):

  model                  T     128     256     512  best  picked
  qwen3_coder_tp8     1024    28.0    26.2       -   256     256
  qwen3_coder_tp8    16384   333.3   277.7       -   256     256
  qwen3_5_tp8         1024    24.7    23.9    24.9   256     256
  qwen3_5_tp8        16384   289.5   249.5   238.1   512     512
  qwen3_5_tp2         1024    62.7    63.6       -   128     128
  qwen3_5_tp2        16384   764.5   759.2       -   256     128  +0.7%
  gemma4_sliding_tp4  1024    40.3    40.3    40.9   256     256
  gemma4_sliding_tp4 16384   488.6   470.8   463.3   512     512
  gemma4_full_tp4     1024    48.0    48.6    49.5   128     256  +1.4%
  gemma4_full_tp4    16384   594.9   592.6   580.2   512     512
  gpt_oss_tp4         1024    21.9    20.7    19.7   512     256  +5.3%
  gpt_oss_tp4        16384   250.8   217.6   195.6   512     512
  qwen3_5_gdn_tp8     1024    27.2    25.9    27.8   256     256
  qwen3_5_gdn_tp8    16384   318.2   274.2   266.2   512     256  +3.0%

GDN's 512 compiles, but vmem_bytes overestimates it, so it isn't a candidate.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.experimental.fused_qkv.kernel import configs
from vllm_torchtpu.kernels.experimental.fused_qkv.kernel.library import layouts

Config = configs.AttentionConfig | configs.GdnConfig
# QkvWeights or GdnWeights, of arrays, ShapeDtypeStructs or refs.
Weights = Any

MiB = 2**20
# VMEM holds the weights, the kernel's tiles and scratch, and XLA's register
# spill slots. The spill slots are in no config: up to ~9 MiB at tile 512.
VMEM_MARGIN_BYTES = 9 * MiB
# vmem_bytes is within -2%..+3% of what Mosaic needs.
VMEM_MODEL_MARGIN_BYTES = MiB // 2
MIN_TILES = 4


@dataclasses.dataclass(frozen=True)
class Tiling:
    """Every tile size the fused kernels use."""

    tile_size: int  # Tokens per pipeline step.
    matmul_rows: int  # Rows per Step 1 group.
    prep_rows: int  # Rows per Step 0 chunk.
    norm_cols: int  # bf16: columns per input-norm block.
    vmem_limit_bytes: int


def column_block(config: Config) -> int:
    """Columns per MXU call: whole heads, at least one full MXU width."""
    tpu_info = pltpu.get_tpu_info()
    block = max(config.head_dim, tpu_info.mxu_column_size)
    if config.num_cols % block:
        block = max(config.head_dim, tpu_info.num_lanes)
    return block


def input_bytes_per_row(config: Config, hidden_size: int) -> dict[str, int]:
    """The input stage's {term: VMEM bytes per token row}.

    Pipelined blocks are double-buffered; scratch is not. Every block's last
    dim is padded to the lane count; second-minor padding is ignored.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    k = layouts.align(hidden_size, num_lanes)
    bf16 = 2
    residual = k * bf16 if config.residual else 0
    terms = {
        "x": 2 * k * bf16,
        "residual": 2 * residual,
        "residual_out": 2 * residual,
        "x_q": k * (1 if config.fp8 else bf16),
    }
    if config.fp8:
        terms["x_scale"] = hidden_size // config.act_quant_block * num_lanes * 4
        terms["row_scale"] = num_lanes * 4
    return terms


def vmem_bytes(bytes_per_row: dict[str, int], tile_size: int) -> dict[str, int]:
    """Predicted scoped VMEM, {term: bytes}, of a kernel with several tiles."""
    return {name: per_row * tile_size for name, per_row in bytes_per_row.items()}


def choose_tiling(
    config: Config,
    num_tokens: int,
    weights: Weights,
    bytes_per_row: dict[str, int],
    *,
    tile_size: int | None = None,
) -> Tiling:
    """Picks the token tile (see the module docstring) and the sub-tile sizes.

    Args:
      weights: the kernel's prepared weights; only their sizes are used.
      bytes_per_row: the kernel's {term: VMEM bytes per token row}.
      tile_size: overrides the token tile.
    """
    tpu_info = pltpu.get_tpu_info()
    num_lanes = tpu_info.num_lanes
    # Not .nbytes: the weights may be ShapeDtypeStructs.
    weights_bytes = sum(a.size * a.dtype.itemsize for a in jax.tree.leaves(weights))
    vmem_limit = tpu_info.vmem_capacity_bytes - weights_bytes - VMEM_MARGIN_BYTES
    if tile_size is None:
        # Tiles of a multi-tile grid must be lane-aligned (SEQ_ALONG_LANE KV
        # blocks put tokens on lanes); a single tile can be any size.
        fits = [
            tile
            for tile in sorted({*range(num_lanes, num_tokens, num_lanes), num_tokens})
            if num_tokens % tile == 0
            and sum(vmem_bytes(bytes_per_row, tile).values()) + VMEM_MODEL_MARGIN_BYTES
            <= vmem_limit
        ]
        assert fits, "no tile fits in VMEM"
        tile_size = max(
            [tile for tile in fits if num_tokens // tile >= MIN_TILES] or [min(fits)]
        )
    # One fp8 tile tall, so every x_q store is whole tiles.
    prep_rows = tpu_info.get_sublane_tiling(jnp.float8_e4m3fn)
    return Tiling(
        tile_size=tile_size,
        # Measured: fewer rows push W to the MXU more often (fp8 at 64 rows is
        # 50% slower than 128; bf16 needs 256).
        matmul_rows=math.gcd(128 if config.fp8 else 256, tile_size),
        prep_rows=prep_rows,
        # bf16 input norm: prep_rows x norm_cols f32 is 16 vregs.
        norm_cols=16 * tpu_info.num_sublanes * num_lanes // prep_rows,
        vmem_limit_bytes=vmem_limit,
    )
