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

# (unpadded tokens, hidden size, KV + gate width, compression ratio).
# Single-token Indexer decode shares the input pipeline and state DMA across
# KV and gate. Other shapes retain the existing separate-field tiling.
PROJECTION_OUTPUT_TILES: dict[tuple[int, int, int, int], int] = {
    (1, 4096, 512, 4): 512,
}

# Single-query HCA metadata rows validated with max_model_len=66560. Keep
# longer tables on the general path; the state row alone occupies 8320 B SMEM.
HCA_DECODE_METADATA_PAGES: dict[
    tuple[str, int, int, int, int, int], tuple[int, int]
] = {
    ("TPU v7", 4096, 1024, 128, 32, 256): (2080, 65),
}

# (token count, request capacity, compression ratio). The compact metadata
# removes long prefix scans at 256 slots; 1/4-slot consumers regress.
COMPACT_BOUNDARY_BATCHES: frozenset[tuple[int, int, int]] = frozenset({(1, 256, 4)})
