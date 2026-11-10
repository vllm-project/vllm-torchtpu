# SPDX-License-Identifier: Apache-2.0
"""Bounds for the SparseCore prefill staging window and metadata."""

# XLA rejects larger gather sources: "sparseCore only supports 16 GiB of HBM
# per tensor" (a per-tensor gather-source limit, distinct from total device
# HBM and from VMEM; the TPU capability interface does not expose it). The
# gather source is a bounded staging window, independent of the pool size.
SC_GATHER_MAX_CACHE_BYTES = 16 * 1024**3

# Queries per SparseCore gather / TensorCore attention microchunk.
SC_CHUNK_QUERIES = 64
# Queries sharing one attention program (batched QK/PV owner axis).
SC_QUERY_GROUP = 8
# Selected-KV rows per attention tile.
SC_TILE_K = 256
