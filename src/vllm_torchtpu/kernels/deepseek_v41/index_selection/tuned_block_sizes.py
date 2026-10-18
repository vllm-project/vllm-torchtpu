# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass

from vllm_torchtpu.logger import init_logger

# At Bq=128, a 32K-key FP32 output window occupies 16 MiB. Mosaic double
# buffers grid-window outputs, so long contexts need 32 MiB plus scorer
# scratch. A single <=64K window retains the original single-buffer layout.
SCORE_WINDOW = 32768
QUERY_TILE = 128
KEY_TILE = 2048


# Decode tables are calibrated with decode_indexer's default score/top-k options.
# The dimensions describe one logical device, not global DP concurrency.
logger = init_logger(__name__)
DEFAULT_DECODE_REQUEST_TILE = 16


@dataclass(frozen=True)
class DecodeTuningKey:
    device_kind: str
    requests: int
    queries: int
    heads: int
    score_capacity: int
    page_records: int
    key_tile: int
    k: int
    banks: int


# Calibrated on TPU7x, JAX 0.10.2 / libtpu 0.0.47; DSpark K5.
# Seeds 517/519, effective lengths 49408/98816; all other options default.
DECODE_REQUEST_TILES: dict[DecodeTuningKey, int] = {
    DecodeTuningKey("TPU7x", 48, 6, 32, 65536, 128, 4096, 512, 2): 12,
    DecodeTuningKey("TPU7x", 48, 6, 32, 131072, 128, 4096, 512, 2): 16,
}


def get_decode_request_tile(key: DecodeTuningKey) -> int:
    """Resolve a static request group; unknown shapes keep a bounded divisor."""
    if key.requests < 1:
        raise ValueError("decode requests must be positive")
    tile = DECODE_REQUEST_TILES.get(key)
    if tile is None:
        tile = next(
            n
            for n in range(min(DEFAULT_DECODE_REQUEST_TILE, key.requests), 0, -1)
            if key.requests % n == 0
        )
        logger.warning_once(
            "[decode_indexer] No tuned request_tile for %s; using %d", key, tile
        )
    if type(tile) is not int or tile < 1 or key.requests % tile:
        raise ValueError(f"Invalid decode request_tile={tile} for {key}")
    return tile


@dataclass(frozen=True)
class CandidateTuningKey:
    device_kind: str
    requests: int
    queries: int
    heads: int
    capacity: int
    key_tile: int
    k: int
    packed: bool = False
    halfplane: bool = False


# Native two-bank preparation, BF16 reusable union, seeds 824/825, JAX 0.10.2
# and libtpu 0.0.47. At 40K capacity, smaller groups lose to SC launch overhead.
CANDIDATE_REQUEST_TILES: dict[CandidateTuningKey, int] = {
    CandidateTuningKey("TPU7x", 48, 6, 32, 40960, 4096, 512): 48,
    # Captured C384 candidates, synthetic Q/weights/pages, seeds 825/826.
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 4096, 512): 48,
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 8192, 512): 48,
    CandidateTuningKey("TPU7x", 48, 6, 32, 49152, 8192, 512, True, True): 48,
}
# JAX 0.10.2 / libtpu 0.0.47, seeds 825/826 plus tied scores: preparation
# and four serial reindexes improve 5.8-7.1% with key/request tiles 8192/48.
# Key: device, request-major Q shape, union capacity, top-k, packed keys.
CANDIDATE_KEY_TILES: dict[tuple[str, tuple[int, ...], int, int, bool, bool], int] = {
    ("TPU7x", (48, 6, 32, 128), 49152, 512, False, False): 8192,
    # Halfplane prepare + four consumers, captured candidate IDs, synthetic Q/K.
    ("TPU7x", (48, 6, 32, 128), 49152, 512, True, True): 8192,
}
# 40K overflowed in 12/48 profiled rank-steps. The 48K bucket covers the
# captured maximum of 5138 blocks with headroom, at 576 MiB per 48 requests.
CANDIDATE_KEY_CAPACITY = 49152
# Generalize the measured 8192/48 configuration without increasing its score
# buffer or per-request dot-product footprint. These defaults are heuristics;
# only exact table entries have measured performance evidence.
DEFAULT_CANDIDATE_KEY_TILE = 8192
DEFAULT_CANDIDATE_REQUEST_TILE = 48
CANDIDATE_SCORE_ELEMENTS = DEFAULT_CANDIDATE_REQUEST_TILE * 6 * CANDIDATE_KEY_CAPACITY


def get_candidate_key_tile(
    device_kind: str,
    query_shape: tuple[int, ...],
    capacity: int,
    k: int,
    packed: bool,
    halfplane: bool = False,
) -> int:
    """Prefer measured tiles, then use an aligned default for related workloads."""
    if capacity < 1:
        raise ValueError("candidate capacity must be positive")
    key = (device_kind, query_shape, capacity, k, packed, packed and halfplane)
    tile = CANDIDATE_KEY_TILES.get(key)
    if tile is None:
        related = (
            device_kind == "TPU7x"
            and not packed
            and k == 512
            and len(query_shape) == 4
            and 1 <= query_shape[1] <= 6
            and query_shape[2:] == (32, 128)
        )
        tile = math.gcd(DEFAULT_CANDIDATE_KEY_TILE if related else 4096, capacity)
        logger.warning_once(
            "[candidate_indexer] No tuned key_tile for %s; using %d", key, tile
        )
    if type(tile) is not int or tile < 1 or capacity % tile:
        raise ValueError(f"Invalid candidate key_tile={tile} for {key}")
    return tile


def get_candidate_request_tile(key: CandidateTuningKey) -> int:
    """Use whole requests and bound untuned groups by the measured score volume."""
    if key.requests < 1:
        raise ValueError("candidate requests must be positive")
    if key.queries < 1 or key.capacity < 1:
        raise ValueError("candidate queries and capacity must be positive")
    tile = CANDIDATE_REQUEST_TILES.get(key)
    if tile is None:
        related = (
            key.device_kind == "TPU7x"
            and not key.packed
            and 1 <= key.queries <= 6
            and key.heads == 32
            and key.key_tile <= DEFAULT_CANDIDATE_KEY_TILE
            and key.k == 512
        )
        limit = (
            min(
                DEFAULT_CANDIDATE_REQUEST_TILE,
                max(1, CANDIDATE_SCORE_ELEMENTS // (key.queries * key.capacity)),
            )
            if related
            else 16
        )
        tile = next(
            n for n in range(min(limit, key.requests), 0, -1) if key.requests % n == 0
        )
        logger.warning_once(
            "[candidate_indexer] No tuned request_tile for %s; using %d", key, tile
        )
    if type(tile) is not int or tile < 1 or key.requests % tile:
        raise ValueError(f"Invalid candidate request_tile={tile} for {key}")
    return tile
