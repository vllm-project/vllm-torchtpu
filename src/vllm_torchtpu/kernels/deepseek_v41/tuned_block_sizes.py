# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Full-context paged indexer: (key columns, query rows). Two 32K SparseCore
# slices cover up to 128K uncompressed tokens at compression ratio 2. On v7x,
# 128 and 256 query rows have comparable latency; 128 uses half the score HBM
# (32 MiB). Longer contexts retain the bounded 32-row pipeline.
PAGED_INDEXER_PREFILL_BUCKET = (65536, 128)

# At Bq=128, a 32K-key FP32 output window occupies 16 MiB. Mosaic double
# buffers grid-window outputs, so long contexts need 32 MiB plus scorer
# scratch. A single <=64K window retains the original single-buffer layout.
PREFILL_SCORE_WINDOW = 32768
PREFILL_QUERY_TILE = 128
PREFILL_KEY_TILE = 2048
