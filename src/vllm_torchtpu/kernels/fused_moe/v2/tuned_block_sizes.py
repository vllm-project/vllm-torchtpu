# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured static tiles for expert-parallel routing and FP4 experts."""

from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)
DEFAULT_HEIGHT_RUNGS = 4


@dataclass(frozen=True)
class RouterTuningKey:
    device_kind: str
    rows: int
    experts: int
    topk: int
    separate_weights: bool


# TPU7x, JAX 0.11.2 / libtpu 0.0.50: one block removes eight grid iterations.
# Both one-table and noaux two-table selection were bit-exact in EP8 runs.
ROUTER_BLOCK_ROWS: dict[RouterTuningKey, int] = {
    RouterTuningKey("TPU7x", 288, 384, 6, False): 288,
    RouterTuningKey("TPU7x", 288, 384, 6, True): 288,
}


# TPU7x / JAX 0.11.2: the masked ID sum removes a separate HBM gather.
# Complete EP8 execution excluding entry barriers: 467.58 -> 450.57 us
# and 491.58 -> 473.43 us, with bit-exact outputs at both seeds.
FUSED_EXPERT_ID_MAP_KEYS = frozenset({RouterTuningKey("TPU7x", 288, 384, 6, True)})


def use_fused_expert_id_map(key: RouterTuningKey) -> bool:
    """Fuse the ID lookup only for the measured decode selector geometry."""
    if key in FUSED_EXPERT_ID_MAP_KEYS:
        return True
    logger.warning_once(
        "[fused_ep_moe_v2] No fused expert-map tuning for %s; using separate lookup",
        key,
    )
    return False


def get_router_block_rows(key: RouterTuningKey, max_default_rows: int) -> int:
    """Use a measured selector tile or the existing power-of-two divisor."""
    tile = ROUTER_BLOCK_ROWS.get(key)
    if tile is not None:
        return tile
    tile = max_default_rows
    while key.rows % tile:
        tile //= 2
    logger.warning_once(
        "[fused_ep_moe_v2] No tuned router tile for %s; using %d rows", key, tile
    )
    return tile


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
