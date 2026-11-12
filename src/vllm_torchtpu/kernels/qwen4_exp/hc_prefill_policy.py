# SPDX-License-Identifier: Apache-2.0
"""HC prefill tile defaults and token-aligned full-mix selection."""

from collections.abc import Iterable
from typing import Literal

HcPrefillMode = int | Literal["auto"]
AUTO_MIX_TILES = (256, 128)
AUTO_COMBINE_TILE = 128
AUTO_MIN_TOKENS = 1024


def select_mix_tile(tokens: int, tiles: Iterable[int]) -> int | None:
    return next((tile for tile in tiles if tokens >= tile and tokens % tile == 0), None)


def validate_hc_prefill_mode(value: object, *, full_mix: bool) -> HcPrefillMode:
    if isinstance(value, str) and value == "auto":
        return "auto"
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        message = "HC prefill mode must be 'auto' or a nonnegative integer"
        if full_mix:
            message += "; full mix must be a multiple of 128"
        raise ValueError(message)
    if full_mix and value and value % 128:
        raise ValueError("HC prefill full mix tile must be a multiple of 128")
    return value
