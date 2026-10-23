# SPDX-License-Identifier: Apache-2.0
"""QSA prefill tile choices."""

import jax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Query block -> scoped VMEM MiB. Q256/KV256 compiles to 61.13 MiB on v7;
# Q128 is retained for environments where the full graph cannot spare 62 MiB.
PAIRED_PAGE_VMEM_MIB = {128: 61, 256: 62}

# (chip, tokens, query heads, KV heads, head dim, compression ratio,
#  physical cache page shape, query dtype, cache dtype) -> (query block, KV tile).
# T8192, contexts 0/8192, uniform/recent selections: Q256/KV256 reduced isolated
# latency 22-37% relative to Q256/KV128 with FP32 attention accumulation.
# The unified serving pool uses 1664-token pages (13 mask segments);
# dedicated attention caches use 128. Both require segment-wise addressing.
SPARSE_PREFILL_TILES = {
    (
        pltpu.ChipVersion.TPU_7X,
        8192,
        24,
        2,
        256,
        4,
        (page_size, 1, 4, 256),
        "bfloat16",
        "float8_e4m3fn",
    ): (256, 256)
    for page_size in (128, 1664)
}

# Tensor-parallel ranks hold one KV head and 24 / TP query heads (TP8 Q3,
# TP4 Q6, TP2 Q12), which missed the table above and fell back to Q256/KV128.
# (chip, query heads, KV heads, head dim, compression ratio, cache row shape,
#  query dtype, cache dtype) -> (query block, KV tile, query subtile), for any
# token count and any 128-aligned page size (the KV tile DMA splits at
# 128-token mask segments; TP8 serving uses 512-token pages). Longer KV
# tiles and query blocks amortize the per-tile FP32 accumulator rescale, KV
# decode and loop overhead over more keys and rows. v7 isolated latency,
# T8192 at a 94k-token prefix, uniform top-512 selections, Q256/KV128 ->
# tuned (ms): Q3 32.3 -> 9.0, Q6 43.7 -> 12.7, Q12 69.2 -> 23.6 (KV1024
# exceeds VMEM at Q12). Q1024 was about 8% faster still at T8192, but a short
# prefill pays for a whole query block over its full context, so the query
# block stays at 512.
SHARD_PREFILL_TILES = {
    (
        pltpu.ChipVersion.TPU_7X,
        query_heads,
        1,
        256,
        4,
        (1, 4, 256),
        "bfloat16",
        "float8_e4m3fn",
    ): tiles
    for query_heads, tiles in (
        (3, (512, 1024, 256)),
        (6, (512, 1024, 128)),
        (12, (512, 512, 128)),
    )
}


def get_sparse_prefill_tiles(
    query: jax.Array,
    cache: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
    with_query_tile: bool = False,
) -> tuple[int, ...]:
    """Select measured tiles, otherwise use one 32-group selection word.

    ``with_query_tile`` appends the query subtile, or None for the kernel's
    default.
    """
    tiles = _get_sparse_prefill_tiles(query, cache, kv_heads=kv_heads, ratio=ratio)
    if len(tiles) == 2:
        tiles = (*tiles, None)
    return tiles if with_query_tile else tiles[:2]


def _get_sparse_prefill_tiles(query, cache, *, kv_heads, ratio):
    tokens, query_heads, head_dim = query.shape
    key = (
        pltpu.get_tpu_info().chip_version,
        tokens,
        query_heads,
        kv_heads,
        head_dim,
        ratio,
        cache.shape[1:],
        str(query.dtype),
        str(cache.dtype),
    )
    shard = SHARD_PREFILL_TILES.get((*key[:1], *key[2:6], cache.shape[2:], *key[7:]))
    if (
        key not in SPARSE_PREFILL_TILES
        and shard is not None
        and not (cache.shape[1] % 128)
    ):
        query_block, kv_tile, query_tile = shard
        # Small steps: do not pad a few hundred tokens to a long query block.
        while query_block > 256 and query_block >= 2 * tokens:
            query_block //= 2
        return query_block, kv_tile, min(query_tile, query_block)
    if key not in SPARSE_PREFILL_TILES:
        logger.warning_once("QSA uses default prefill tiles for key: %s", key)
        # Keep query/state windows within the measured Q24/D256 footprint
        # when fewer TP shards give this rank more attention heads.
        rows = max(8, 256 * 24 * 256 // (query_heads * cache.shape[-1]))
        query_block = min(256, 1 << (rows.bit_length() - 1))
        return query_block, ratio * 32
    return SPARSE_PREFILL_TILES[key]
