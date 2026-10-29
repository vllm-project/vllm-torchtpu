# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from dataclasses import dataclass, replace

from vllm_torchtpu.logger import init_logger

# At Bq=128, a 32K-key FP32 output window occupies 16 MiB. Mosaic double
# buffers grid-window outputs, so long contexts need 32 MiB plus scorer
# scratch. A single <=64K window retains the original single-buffer layout.
SCORE_WINDOW = 32768
QUERY_TILE = 128
KEY_TILE = 2048

# Full-context paged indexer: (key columns, query rows). Two 32K SparseCore
# slices cover up to 128K uncompressed tokens at compression ratio 2. On v7x,
# 128 and 256 query rows have comparable latency; 128 uses half the score HBM
# (32 MiB). Longer contexts retain the bounded 32-row pipeline.
PAGED_INDEXER_PREFILL_BUCKET = (65536, 128)

# Measured candidate-union and gather policies. Keep these separate from
# hardware lane widths and the packed-key format.
CANDIDATE_DENSE_UNION_MAX_BLOCKS = 16384
CANDIDATE_GATHER_ROWS_PER_SUBCORE = 512
CANDIDATE_GATHER_MAX_READS = 8
# (TPU generation, score width, candidate count, records per candidate block).
PUBLISH_CANDIDATE_SHAPES = frozenset({(7, 16384, 2048, 8)})


@dataclass(frozen=True)
class PersistentScoreConfig:
    key_tile: int
    compute_tile: int
    head_group: int

    def __post_init__(self):
        if any(
            type(value) is not int or value < 1
            for value in (self.key_tile, self.compute_tile, self.head_group)
        ):
            raise ValueError("Score tiles and head_group must be positive integers")
        if self.key_tile % self.compute_tile:
            raise ValueError("key_tile must be a multiple of compute_tile")


PERSISTENT_SCORE_CONFIGS: dict[tuple[int, ...], PersistentScoreConfig] = {
    (6, 32, 128, 65536, 1): PersistentScoreConfig(99328, 1024, 32),
    (6, 32, 128, 131072, 1): PersistentScoreConfig(99328, 1024, 32),
    (6, 32, 128, 65536, 2): PersistentScoreConfig(99328, 1024, 32),
    (6, 32, 128, 131072, 2): PersistentScoreConfig(99328, 1024, 32),
}


# Decode tables are calibrated with decode_indexer's default score/top-k options.
# Request tiles bound per-group buffers; total batch size is not a lookup key.
logger = init_logger(__name__)
DEFAULT_DECODE_REQUEST_TILE = 16


@dataclass(frozen=True)
class DecodeTuningKey:
    device_kind: str
    queries: int
    heads: int
    score_capacity: int
    page_records: int
    k: int
    banks: int


# Calibrated at R48 on TPU7x, JAX 0.10.2 / libtpu 0.0.47; DSpark K5.
# Seeds 517/519, effective lengths 49408/98816; all other options default.
DECODE_REQUEST_TILES: dict[DecodeTuningKey, int] = {
    DecodeTuningKey("TPU7x", 6, 32, 65536, 128, 512, 1): 12,
    DecodeTuningKey("TPU7x", 6, 32, 131072, 128, 512, 1): 16,
    DecodeTuningKey("TPU7x", 6, 32, 65536, 128, 512, 2): 12,
    DecodeTuningKey("TPU7x", 6, 32, 131072, 128, 512, 2): 16,
}


def get_persistent_score_config(key: tuple[int, ...]) -> PersistentScoreConfig:
    """Choose persistent tiles; table misses keep the same kernel."""
    config = PERSISTENT_SCORE_CONFIGS.get(key)
    if config is None:
        # The native ragged-query suite exercises 4K DMA / 1K compute fragments.
        config = PersistentScoreConfig(4096, 1024, 32)
        logger.warning_once(
            "[decode_scores] No tuned config for %s; using %s", key, config
        )
    return config


def get_decode_request_tile(key: DecodeTuningKey, requests: int) -> int:
    """Bound each score group independently of the total batch size."""
    if requests < 1:
        raise ValueError("decode requests must be positive")
    tile = DECODE_REQUEST_TILES.get(key)
    if tile is None:
        tile = min(DEFAULT_DECODE_REQUEST_TILE, requests)
        logger.warning_once(
            "[decode_indexer] No tuned request_tile for %s; using %d", key, tile
        )
    if type(tile) is not int or tile < 1:
        raise ValueError(f"Invalid decode request_tile={tile} for {key}")
    return min(tile, requests)


@dataclass(frozen=True)
class CandidateTuningKey:
    """Workload key for canonical halfplane int32 candidate caches."""

    device_kind: str
    queries: int
    heads: int
    capacity: int
    key_tile: int | None
    k: int


@dataclass(frozen=True)
class CandidateBlockSizes:
    key_tile: int
    request_tile: int
    topk_block_size: int = 8192
    reconstruct_threshold: int = 8192
    compute_tile: int | None = None
    head_stripe: int | None = None
    fuse_topk_id_mapping: bool = False
    dma_tile: int | None = None


_TPU7X_CACHED_DECODE = CandidateBlockSizes(
    key_tile=49152,
    request_tile=16,
    topk_block_size=16384,
    reconstruct_threshold=1024,
    compute_tile=8192,
    head_stripe=8,
    fuse_topk_id_mapping=True,
    dma_tile=24576,
)


# Exact configurations take precedence over the bounded upstream fallback.
CANDIDATE_BLOCK_SIZES: dict[CandidateTuningKey, CandidateBlockSizes] = {
    CandidateTuningKey("TPU7x", 6, 32, 40960, None, 512): CandidateBlockSizes(4096, 48),
    CandidateTuningKey("TPU7x", 6, 32, 40960, 4096, 512): CandidateBlockSizes(4096, 48),
    CandidateTuningKey("TPU7x", 6, 32, 49152, 4096, 512): CandidateBlockSizes(4096, 48),
    CandidateTuningKey("TPU7x", 6, 32, 49152, 8192, 512): CandidateBlockSizes(8192, 48),
    CandidateTuningKey("TPU7x", 6, 32, 49152, 16384, 512): CandidateBlockSizes(
        16384, 24, 16384, 1024
    ),
    CandidateTuningKey("TPU7x", 6, 32, 49152, 24576, 512): CandidateBlockSizes(
        24576, 24, 16384, 1024, compute_tile=4096
    ),
    CandidateTuningKey("TPU7x", 6, 32, 49152, 49152, 512): _TPU7X_CACHED_DECODE,
    CandidateTuningKey("TPU7x", 6, 32, 49152, None, 512): _TPU7X_CACHED_DECODE,
}
# 40K overflowed in 12/48 profiled rank-steps. The 48K bucket covers the
# captured maximum of 5138 blocks with headroom, at 576 MiB per 48 requests.
CANDIDATE_KEY_CAPACITY = 49152
# Basic config for table misses: bounded groups and a smaller MXU key tile.
# Keep the score-buffer budget below the measured 48 x 6 x 48K configuration.
DEFAULT_CANDIDATE_KEY_TILE = 4096
DEFAULT_CANDIDATE_REQUEST_TILE = 16
CANDIDATE_SCORE_ELEMENTS = 48 * 6 * CANDIDATE_KEY_CAPACITY


def get_candidate_block_sizes(
    key: CandidateTuningKey, requests: int
) -> CandidateBlockSizes:
    """Resolve scorer and selector tiles together for the complete pipeline."""
    if requests < 1:
        raise ValueError("candidate requests must be positive")
    if key.queries < 1 or key.capacity < 1:
        raise ValueError("candidate queries and capacity must be positive")
    config = CANDIDATE_BLOCK_SIZES.get(key)
    if config is None:
        key_tile = key.key_tile
        if key_tile is None:
            key_tile = math.gcd(DEFAULT_CANDIDATE_KEY_TILE, key.capacity)
        limit = min(
            DEFAULT_CANDIDATE_REQUEST_TILE,
            max(1, CANDIDATE_SCORE_ELEMENTS // (key.queries * key.capacity)),
        )
        config = CandidateBlockSizes(key_tile, limit)
        logger.warning_once(
            "[candidate_indexer] No tuned block sizes for %s; using %s", key, config
        )
    if config.request_tile < 1:
        raise ValueError(
            f"Invalid candidate request_tile={config.request_tile} for {key}"
        )
    return replace(config, request_tile=min(config.request_tile, requests))
