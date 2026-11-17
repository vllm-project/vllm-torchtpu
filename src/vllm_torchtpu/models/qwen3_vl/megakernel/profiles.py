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
"""`qwen3_vl-baseline`: the Qwen3-VL text decoder's execution profile.

One single-request decode bucket (`B=1`, `T=1`). Every layer streams ~122 MB
of weights per rank, so the tiles keep HBM streaming: equal ~8 MiB dense
tiles (slot count = byte count), no unrolled matmul (`dmm_unroll=8` lets one
matmul pin every ring slot) and no lazy refill.
"""

from __future__ import annotations

from vllm_torchtpu.megakernel.core.exec_spec import Tiles
from vllm_torchtpu.megakernel.core.modules.gqa import GQAModule
from vllm_torchtpu.megakernel.core.profile import Profile, TileContext

# `ModelSpec.name` of the Qwen3-VL text decoder.
MODEL_NAME = "qwen3_vl_text"

# vLLM rotates q / k in float32 (`layers/adapter/rotary_embedding.py`). The KV
# blocks share the dense ring, so its 7 banks (58 MiB of scratch, under the
# 62 MiB Mosaic limit at B=1) hold the lookahead of both: 3459 us vs 3654 at
# 6 banks for a 64k-token context, the same at 8k.
DECODE_TILES = Tiles(
    bank=8 << 20, nbd=7, hier=1, mb8=2, lazy=False, dmm_unroll=1
).with_modules({GQAModule.name: {"rope_fp32": True}})


def decode_tiles(tc: TileContext | int) -> Tiles:
    """The decode tiles, for every bucket."""
    del tc
    return DECODE_TILES


PROFILE = Profile(
    name="qwen3_vl-baseline",
    model=MODEL_NAME,
    numerics="exact",
    model_default=True,
    tiles=decode_tiles,
    buckets=(1,),
    verify_buckets=(),
    description="Qwen3-VL text decoder: single-request decode, 8 MiB dense tiles.",
)
PROFILES: dict[str, Profile] = {PROFILE.name: PROFILE}
DEFAULT_PROFILE = PROFILE
