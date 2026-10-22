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
"""Decode lanes of the batched RPA kernel, from a table keyed on layer shape.

The decode kernel works on ``decode_batch_size`` sequences (lanes) at once;
the wrapper's default is 8. Fewer lanes leave VMEM for a larger KV block. With
one KV head per device, head_dim 256 and an fp8 KV cache, 8 lanes cap the KV
block at 2,816 tokens and 4 lanes allow 4,096, so a ~98K-token sequence takes
25 blocks instead of 36 and pays the fixed per-block cost (masks, stitch,
lane-chained softmax) less often: the attention call ran 15.1% faster at 64
rows and 14.6% at 112 rows (tpu7x, standalone, j-023feef8), and the decode
step 4.3% / 3.8% faster in the engine (j-014e7742 vs j-b647e38d). The result
is not bit-exact against 8 lanes (max abs 4.3e-4, bf16 rounding scale).

Only that measured shape is in the table, and only for long-context
deployments (``max_model_len`` of at least the measured ~98K tokens); every
other layer keeps the wrapper's default.
"""

from typing import Any

import torch

# (kv heads per device, head_dim, fp8 KV cache) -> decode lanes.
_DECODE_LANES: dict[tuple[int, int, bool], int] = {
    (1, 256, True): 4,
}
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
# The context the table entries were measured at (tokens per sequence).
DECODE_LANES_MIN_MODEL_LEN = 98304


def decode_lanes(
    *,
    num_kv_heads: int,
    head_dim: int,
    kv_cache_dtype: torch.dtype | None,
    max_model_len: int,
) -> int | None:
    """Decode lanes for a layer of this shape, or None for the default."""
    if max_model_len < DECODE_LANES_MIN_MODEL_LEN:
        return None
    return _DECODE_LANES.get((num_kv_heads, head_dim, kv_cache_dtype in _FP8_DTYPES))


def kernel_kwargs(lanes: int | None) -> dict[str, Any]:
    """Keyword arguments that give every batched RPA program ``lanes`` lanes."""
    return {} if lanes is None else {"decode_batch_size": lanes}
