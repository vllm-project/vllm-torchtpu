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
"""Tiles for BF16 sliding-window attention on v7x."""

# 128 query/head rows use four 32-entry MRB regions (two QK, two PV),
# leaving room for the 64-entry KV permutation. Eight windows amortize DMA
# pipeline prologue while keeping enough independent grids for both cores.
SWA_QUERY_ROWS = 128
SWA_PIPELINE_GROUPS = 8

# Amortize the four plane DMAs across independent decode windows.
SWA_DECODE_GROUPS = 8
