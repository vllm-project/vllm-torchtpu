# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
"""Shared-window tiles measured on TPU v7x with JAX 0.10.2."""

# (query heads, head dim, window width): queries per compute tile.
# Sharing a 256-key KV tile across 64 queries avoids reloading the tile;
# the fused kernel stays below the default 32-MiB scoped VMEM budget.
QUERY_BLOCKS: dict[tuple[int, int, int], int] = {(64, 512, 128): 64}
