# SPDX-License-Identifier: Apache-2.0
"""Measured tiles for grouped output projections."""

from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class WoATuningKey:
    device_kind: str
    activation_shape: tuple[int, ...]
    weight_shape: tuple[int, ...]
    activation_dtype: str
    weight_dtype: str
    scale_dtype: str
    tile_t: int
    tile_r: int
    has_rope: bool
    quantize_activations: bool


# TPU7x, JAX 0.11.2 / libtpu 0.0.50, Falcon exp-1yxqyt6ap6.
# One 288-row contraction replaces three 96-row contractions. Two input seeds
# were bit-exact; device-module medians fell from 25.45 to 23.86 us.
WO_A_SUB_T: dict[WoATuningKey, int] = {
    WoATuningKey(
        "TPU7x",
        (288, 64, 512),
        (4096, 8192),
        "bfloat16",
        "float8_e4m3fn",
        "float32",
        288,
        1024,
        False,
        True,
    ): 288,
}


def get_wo_a_sub_t(key: WoATuningKey, default: int) -> int:
    tile = WO_A_SUB_T.get(key)
    if tile is None:
        logger.warning_once("No measured WO_A sub-tile for %s; using %d", key, default)
        return default
    return tile
