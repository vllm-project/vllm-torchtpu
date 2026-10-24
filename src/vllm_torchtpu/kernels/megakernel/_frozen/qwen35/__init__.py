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
"""Frozen snapshot of Qwen3.5-397B-A17B-FP8 decode megakernel (@ c196d2500)."""

from . import megakernel, pool_alias, pool_io
from .megakernel import (
    DEFAULT_TILES,
    FULL,
    FULL_EP,
    TINY,
    TINY_EP,
    TINY_TILES,
    VERIFY_TILES,
    WIDE_TILES,
    MKConfig,
    Tiles,
    build,
    default_tiles,
    make_step,
)
from .pool_io import (
    gather_gdn_states,
    make_flat_attn_step,
    make_flat_pooled_step,
    make_pooled_step,
    scatter_pool_states,
)

__all__ = [
    "DEFAULT_TILES",
    "FULL",
    "FULL_EP",
    "MKConfig",
    "TINY",
    "TINY_EP",
    "TINY_TILES",
    "Tiles",
    "VERIFY_TILES",
    "WIDE_TILES",
    "build",
    "default_tiles",
    "gather_gdn_states",
    "make_flat_attn_step",
    "make_flat_pooled_step",
    "make_pooled_step",
    "make_step",
    "megakernel",
    "pool_alias",
    "pool_io",
    "scatter_pool_states",
]
