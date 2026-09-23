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
"""Sort-free top-k for MoE routing, in plain torch ops.

The same algorithm as the Pallas kernel in ``kernels/router_topk.py``: ``k``
passes of (row max -> lowest matching column -> mask that column out). Used
off-TPU, and as the numerics oracle the kernel tests compare against.
"""

import torch

# Clamp floor. Below any real routing score, and strictly above the dtype
# minimum used to mask a selected column -- see ``kernels/router_topk.py``.
NEG = -3.0e38


def rowmax_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``torch.topk(scores, k, dim=-1)`` without the sort.

    Returns ``(values, indices)``, values descending and indices int32. Ties
    resolve to the lowest expert id; the sort path leaves them undefined,
    since torch_tpu emits ``is_stable=false``.

    Scores at or below ``NEG`` are clamped up to it, so every returned index
    is a real expert. A row made entirely of such scores keeps NaN weights and
    still names k distinct experts, as the sort path does.
    """
    n = scores.shape[-1]
    if k > n:
        raise ValueError(f"topk k={k} exceeds the expert count {n}")
    iota = torch.arange(n, device=scores.device, dtype=torch.int32)
    masked = torch.finfo(scores.dtype).min

    cur = torch.where(scores > NEG, scores, NEG)
    values, indices = [], []
    for _ in range(k):
        m = torch.amax(cur, dim=-1, keepdim=True)
        # Lowest column attaining the max; the rest fill with the last expert
        # so the min never selects a phantom.
        idx = torch.amin(torch.where(cur == m, iota, n - 1), dim=-1, keepdim=True)
        values.append(m)
        indices.append(idx)
        cur = torch.where(iota == idx, masked, cur)

    weights = torch.cat(values, dim=-1)
    ids = torch.cat(indices, dim=-1)
    # No score above the sentinel means no real maximum: keep the row
    # poisonous so the caller's renormalization propagates it.
    weights = torch.where(weights[..., :1] <= NEG, torch.nan, weights)
    return weights, ids
