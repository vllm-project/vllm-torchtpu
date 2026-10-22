# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured static tile-height policies for block-scaled FP4 experts."""

from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
DEFAULT_HEIGHT_RUNGS = 4


@dataclass(frozen=True)
class HeightTuningKey:
    generation: int
    hidden: int
    inter: int
    rhs_qb: int
    tile_m: int
    g_local: int
    ep: int
    rows_alloc: int
    act_fn: str
    logical_intermediate_size: int | None
    has_bias: bool
    padded_scales: bool


# TPU7x, JAX 0.11.2 / libtpu 0.0.50, 288 local rows and top-k 6.
# Two bodies reduce static program size; the final rung still covers a full tile.
FP4_HEIGHT_RUNGS: dict[HeightTuningKey, int] = {
    HeightTuningKey(
        generation=7,
        hidden=5120,
        inter=2560,
        rhs_qb=512,
        tile_m=128,
        g_local=48,
        ep=8,
        rows_alloc=14336,
        act_fn="silu_and_mul_with_clamp",
        logical_intermediate_size=2304,
        has_bias=False,
        padded_scales=True,
    ): 2,
}


def get_fp4_height_rungs(key: HeightTuningKey) -> int:
    """Keep the existing height ladder outside measured workloads."""
    rungs = FP4_HEIGHT_RUNGS.get(key)
    if rungs is None:
        logger.warning_once(
            "[fused_ep_moe_v2] No tuned height ladder for %s; using %d rungs",
            key,
            DEFAULT_HEIGHT_RUNGS,
        )
        return DEFAULT_HEIGHT_RUNGS
    return rungs
