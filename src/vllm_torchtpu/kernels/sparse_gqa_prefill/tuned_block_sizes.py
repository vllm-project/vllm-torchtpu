# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Bounded prefill dispatch and metadata schedules."""

# Query-row window used by decode indexer scoring.
PREFILL_DISPATCH_CHUNK_QUERIES = 8192

# XLA rejects larger gather sources: "sparseCore only supports 16 GiB of HBM
# per tensor" (a per-tensor gather-source limit, distinct from total device
# HBM and from VMEM; the TPU capability interface does not expose it -- jax
# SparseCoreInfo publishes only cores/lanes/VMEM/dma-granule). The gather source
# is a bounded staging window, independent of the persistent owner size.
SC_GATHER_MAX_CACHE_BYTES = 16 * 1024**3

# Bound the word-axis batch used by the exact-union reduction.
UNION_WORD_BATCH_SIZE = 16
