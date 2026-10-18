# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# Full-context paged indexer: (key columns, query rows). Two 32K SparseCore
# slices cover up to 128K uncompressed tokens at compression ratio 2. On v7x,
# 128 and 256 query rows have comparable latency; 128 uses half the score HBM
# (32 MiB). Longer contexts retain the bounded 32-row pipeline.
PAGED_INDEXER_PREFILL_BUCKET = (65536, 128)

# (chip, query tokens, heads, request capacity, SWA page size, noncausal).
# BQ=32 uses four BF16 KV slots: 2 MiB instead of BQ=16's 8 MiB.
# The request dimension matters: grouped prefill amortizes SWA QK, whereas
# one-token requests still execute a full BQ group each.
_CSA2_ATTENTION_BQ = {
    (pltpu.ChipVersion.TPU_7X, 2048, 64, 8, 32, False): 32,
    # Keep the SC/TC gather pipeline's existing tile for 128-token chunks.
    (pltpu.ChipVersion.TPU_7X, 128, 64, 8, 32, False): 16,
    (pltpu.ChipVersion.TPU_7X, 48, 64, 1, 32, False): 16,
    # Eight six-token MTP requests prefer BQ=6. The same capacity can also
    # hold one 48-token request; do not infer its active count from capacity.
    (pltpu.ChipVersion.TPU_7X, 48, 64, 8, 32, False): 6,
    (pltpu.ChipVersion.TPU_7X, 48, 64, 48, 32, False): 1,
}


def csa2_attention_bq(
    num_tokens: int,
    num_heads: int,
    request_capacity: int,
    page_size: int,
    non_causal: bool,
    *,
    default_bq: int,
) -> int:
    """Select for the actual attention call, including gather pipeline chunks."""
    key = (
        pltpu.get_tpu_info().chip_version,
        num_tokens,
        num_heads,
        request_capacity,
        page_size,
        non_causal,
    )
    if key in _CSA2_ATTENTION_BQ:
        return _CSA2_ATTENTION_BQ[key]
    logger.warning_once(
        "CSA2 attention tuning miss: key=%s; using bq=%d", key, default_bq
    )
    return default_bq


def csa2_gather_chunk_size(num_tokens: int) -> int:
    # On v7x TP1 with 64 heads, 128-row chunks overlap SC gather with TC
    # attention while retaining full gathered KV for subsequent layers.
    # Up to 512 queries, launch and concatenation costs erase the stage-level
    # benefit; use one gather and one attention without extra staging HBM.
    return num_tokens if num_tokens <= 512 else 128
