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
"""Measured workload selections for DeepSeek-V4 core attention."""

# (device, topk, sequence capacity, pages per sequence, page size) -> query
# buckets. One metadata program handles one query: scalar lookup and pipeline
# costs grow with query count. TPU v7 complete SWA+CSA A/B benefits at 1/2/4;
# eight queries regresses, so larger batches retain the original bulk path.
FUSED_CSA_METADATA_BUCKETS = {
    ("TPU v7", 512, 256, 65, 256): (1, 2, 4),
}

# (device, query heads, head dimension, SWA page size, HCA page size) ->
# HCA pages per tile. Preserve the generic HCA kernel's 32-page reduction order.
FUSED_HCA_DECODE_KV_PAGES = {
    ("TPU v7", 64, 512, 128, 8): 32,
}
