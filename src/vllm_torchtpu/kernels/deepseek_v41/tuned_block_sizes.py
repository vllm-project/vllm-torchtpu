# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Full-context paged indexer: (key columns, query rows). Two 32K SparseCore
# slices cover up to 128K uncompressed tokens at compression ratio 2. On v7x,
# 128 and 256 query rows have comparable latency; 128 uses half the score HBM
# (32 MiB). Longer contexts retain the bounded 32-row pipeline.
PAGED_INDEXER_PREFILL_BUCKET = (65536, 128)


def csa2_gather_chunk_size(num_tokens: int) -> int:
    # On v7x TP1 with 64 heads, 128-row chunks overlap SC gather with TC
    # attention while retaining full gathered KV for subsequent layers.
    # Up to 512 queries, launch and concatenation costs erase the stage-level
    # benefit; use one gather and one attention without extra staging HBM.
    return num_tokens if num_tokens <= 512 else 128
