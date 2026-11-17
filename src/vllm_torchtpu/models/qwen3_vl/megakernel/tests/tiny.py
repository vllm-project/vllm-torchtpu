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
"""Tiny Qwen3-VL text geometries and tiles for this folder's tests."""

from __future__ import annotations

import dataclasses
from typing import Any

from vllm_torchtpu.megakernel.core.exec_spec import Tiles

from .. import model, profiles


def tiny_geom(R: int = 1, L: int = 2) -> Any:
    """A 2-layer geometry with one KV head per rank, as the checkpoint has at
    TP=8 (`R=1`: 4 query heads; `R=8`: 2 per rank)."""
    return model.Qwen3VLText.derive_geometry(
        {
            **model.COSMOS3_SUPER_TEXT_CONFIG,
            "vocab_size": 64,
            "hidden_size": 1024 if R > 1 else 256,
            "intermediate_size": 512 * R,
            "num_hidden_layers": L,
            "num_attention_heads": 4 if R == 1 else 2 * R,
            "num_key_value_heads": R,
        }
    )


def tiny_tiles(kv_blk: int) -> Tiles:
    """The profile's decode tiles with small banks for tiny geometries."""
    return dataclasses.replace(
        profiles.DECODE_TILES, bank=512 << 10, nbd=8, vmem_mib=32
    ).with_modules({"gqa": {"kv_blk": kv_blk, "nkv": 3}})
