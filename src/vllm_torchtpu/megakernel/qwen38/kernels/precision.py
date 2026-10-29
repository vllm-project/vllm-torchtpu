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
"""Explicit runtime precision recipes and bounded native NVFP4 conversion.

The checkpoint remains native NVFP4. The opt-in TorchTPU profile reproduces
its pinned loader's block-64 E2M1 requantization in bounded host scratch.
"""

from __future__ import annotations

import numpy as np

NATIVE = "native"
TORCHTPU_W4A8_B64 = "torchtpu_w4a8_b64"
PROFILES = (NATIVE, TORCHTPU_W4A8_B64)
TORCHTPU_REVISION = "8511a38f70e763a55f6a11d89183f60c1c6a3436"
CONVERSION_SCRATCH_BYTES = 96 << 20
PROVENANCE = {
    "revision": TORCHTPU_REVISION,
    "requantization_source_sha256": "784de791cfb9977455c97bf1c98818a918defffe3c673f3341f2917c87364c77",
    "expert_source_sha256": "5b9b051072238c1bf89aaaf1ea0884c86026950c66484a6b75dc64b0ae94e5f9",
    "adapter_source_sha256": "163d7f0cca089e798acee95c32b6425f3194aeaf8f646404efd1f3b87628a2c0",
    "weight_rounding": "native E2M1*FP8 block16*FP32 global -> FP32 block64 amax/6 -> E2M1 RNE",
    "activation_rounding": "BF16 -> dynamic per-row FP8, BF16 multiply by reciprocal",
    "intermediate_rounding": "SiLU(FP32 gate)*FP32 up -> BF16 -> dynamic per-row FP8",
    "expert_output_rounding": "FP32 down -> BF16 -> dynamic FP8; FP32 scaled routed combine -> BF16",
    "communication_dtype": "bfloat16 residual partials and host sums, float32 sums",
}
_LEVELS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


def profile_metadata(profile):
    if profile not in PROFILES:
        raise ValueError(f"Unknown runtime precision {profile!r}")
    return {"name": profile, **(PROVENANCE if profile != NATIVE else {})}


def _encode_e2m1(value):
    """Finite saturating E2M1 RNE, including the sign of zero."""
    # Seven ordered comparisons avoid a per-value binary search and retain
    # the alternating even-code midpoint ties of the pinned FP4 conversion.
    absolute = np.abs(value)
    code = np.greater(absolute, np.float32(0.25)).astype(np.uint8)
    scratch = np.empty(absolute.shape, np.bool_)
    for threshold, inclusive in (
        (0.75, True),
        (1.25, False),
        (1.75, True),
        (2.5, False),
        (3.5, True),
        (5.0, False),
    ):
        compare = np.greater_equal if inclusive else np.greater
        compare(absolute, np.float32(threshold), out=scratch)
        np.add(code, scratch, out=code)
    # Match searchsorted for NaN intermediates as well as finite values.
    np.isnan(absolute, out=scratch)
    np.copyto(code, np.uint8(7), where=scratch)
    # Preserve the original sign bit, including -0.
    return code | (np.signbit(value).astype(np.uint8) << 3)


def requantize_kmajor(packed, block_scales, weight_globals):
    """Convert [E,K/2,N] native bytes into [E,K/8,N] U32 block64.

    Work on one expert and at most 128 output channels at once. Gate and up
    must be called independently with their own checkpoint global scales.
    FP32 operation order matches pinned ``requant_unpack_kmajor``.
    """
    packed, block_scales = np.asarray(packed), np.asarray(block_scales)
    globals_ = np.asarray(weight_globals, np.float32)
    if packed.ndim != 3 or packed.dtype != np.uint8:
        raise ValueError("Expected native [E,K/2,N] uint8 weights")
    experts, half_k, outputs = packed.shape
    k = half_k * 2
    if k % 64 or block_scales.shape != (experts, k // 16, outputs):
        raise ValueError("Block64 conversion requires complete native block16 groups")
    if globals_.shape != (experts,) or not np.all(np.isfinite(globals_) & (globals_ > 0)):
        raise ValueError("Each expert needs its own finite positive FP32 global scale")
    if not np.all(np.isfinite(block_scales.astype(np.float32)) & (block_scales >= 0)):
        raise ValueError("Native block scales must be finite and nonnegative")
    out = np.empty((experts, k // 8, outputs), np.uint32)
    scales = np.empty((experts, k // 64, outputs), np.float32)
    for expert in range(experts):
        for first in range(0, outputs, 128):
            last = min(first + 128, outputs)
            source = packed[expert, :, first:last]
            codes = np.empty((k, last - first), np.uint8)
            codes[0::2] = source & 15
            codes[1::2] = source >> 4
            native = _LEVELS[codes & 7] * np.where(codes & 8, np.float32(-1), np.float32(1))
            fused = block_scales[expert, :, first:last].astype(np.float32) * globals_[expert]
            dequant = (native.reshape(k // 16, 16, -1) * fused[:, None, :]).reshape(k // 64, 64, -1)
            scale = np.max(np.abs(dequant), axis=1) * np.float32(1.0 / 6.0)
            inv = np.zeros_like(scale)
            np.divide(np.float32(1), scale, out=inv, where=scale != 0)
            encoded = _encode_e2m1(np.clip(dequant * inv[:, None, :], -6, 6)).reshape(k // 8, 8, -1)
            words = np.zeros((k // 8, last - first), np.uint32)
            for nibble in range(8):
                words |= encoded[:, nibble, :].astype(np.uint32) << (4 * nibble)
            out[expert, :, first:last] = words
            scales[expert, :, first:last] = scale
    return out, scales


def fast_tile_updates(tile, *, rank, expert_tp, config, include_params=True):
    """Convert a rank-owned native atom directly into final fast-profile slices."""
    from .preshard import _geometry, _validate_tile, owner_rank

    local_experts, width = _geometry(config, expert_tp)
    _validate_tile(tile, config)
    owner = owner_rank(tile.expert_start, tile.channel_start, expert_tp, config=config)
    last = owner_rank(
        tile.expert_start + tile.expert_count - 1,
        tile.channel_start + tile.channel_width - 1,
        expert_tp,
        config=config,
    )
    if owner != last:
        raise ValueError("Split the native atom at rank boundaries before conversion")
    if owner != rank:
        return
    e = tile.expert_start - rank // expert_tp * local_experts
    j = tile.channel_start - rank % expert_tp * width
    for half, name in enumerate(("gate", "up")):
        weight, scale = requantize_kmajor(
            tile.arrays["egu_" + name],
            tile.arrays["egu_" + name + "_scale"],
            tile.arrays["eparams"][:, half * 2 + 1],
        )
        yield "egu", (tile.layer, e, 0, half * width + j), weight[None]
        yield "egu_scale", (tile.layer, e, 0, half * width + j), scale[None]
    weight, scale = requantize_kmajor(
        tile.arrays["edw"], tile.arrays["edw_scale"], tile.arrays["eparams"][:, 5]
    )
    yield "edw", (tile.layer, e, j // 8, 0), weight[None]
    yield "edw_scale", (tile.layer, e, j // 64, 0), scale[None]
    if include_params:
        params = np.zeros((1, tile.expert_count, 8, 128), np.float32)
        params[0, :, 0, :6] = tile.arrays["eparams"]
        yield "eparams", (tile.layer, e, 0, 0), params
