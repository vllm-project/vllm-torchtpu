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


def get_sparse_prefill_tiles(
    query: jax.Array, cache: jax.Array, *, kv_heads: int, ratio: int
) -> tuple[int, int]:
    """Select measured tiles, otherwise use one 32-group selection word."""
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
    if key not in SPARSE_PREFILL_TILES:
        logger.warning_once("QSA uses default prefill tiles for key: %s", key)
        # Keep query/state windows within the measured Q24/D256 footprint
        # when fewer TP shards give this rank more attention heads.
        rows = max(8, 256 * 24 * 256 // (query_heads * cache.shape[-1]))
        query_block = min(256, 1 << (rows.bit_length() - 1))
        return query_block, ratio * 32
    return SPARSE_PREFILL_TILES[key]
