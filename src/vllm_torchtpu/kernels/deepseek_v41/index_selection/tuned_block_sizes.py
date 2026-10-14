# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
