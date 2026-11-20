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

TPU_TILE_ROWS = 8
_BF16_ITEMSIZE = 2


def fused_qkvz_pcp_dma_row_alignment(hidden_itemsize: int = _BF16_ITEMSIZE) -> int:
    """Row alignment required by ``fused_qkvz_projection_pcp_gdn`` VMEM DMA."""
    return TPU_TILE_ROWS * (4 // hidden_itemsize)


__all__ = ["TPU_TILE_ROWS", "fused_qkvz_pcp_dma_row_alignment"]
