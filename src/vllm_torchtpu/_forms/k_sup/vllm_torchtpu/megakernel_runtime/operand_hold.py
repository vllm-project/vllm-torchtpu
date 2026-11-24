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
"""Megakernel operands torch_tpu passes to a compiled call without per-call re-materialisation.

torch_tpu runs one eager `execute_as_strided` program per compiled call for every device buffer
whose history holds an in-place slice write or a view / dtype reinterpretation; `clone()` keeps
that history, a host round-trip does not, and fresh cat / pad / arithmetic results are clean
(2d's single-host microbench, segment MK 3e0de2206: the 23-program burst, ~1.1-1.4 ms/step;
whole-step w08r c1: 18 programs, +0.85 ms/step). Shared by the segment and whole-step packers.
"""

from __future__ import annotations

import torch

SMALL_OPERAND_BYTES = 32 << 20  # small packed operands (smalls, scales, norms, tables)


def host_round_trip_small(ops: list[torch.Tensor], limit: int = SMALL_OPERAND_BYTES) -> int:
    """Replace, in place, every operand of at most `limit` bytes by a host round-trip copy (no
    history). Larger ones (expert weights: parameter views, or fresh stacks) are kept. Returns
    how many were replaced."""
    n = 0
    for i, t in enumerate(ops):
        if t.numel() * t.element_size() <= limit:
            ops[i] = t.cpu().to(t.device)
            n += 1
    return n


def hbm_summary(tensors, model) -> str:
    """`"<total> GiB, <own> GiB not shared with model parameters"` of packed kernel operands: what they
    add to HBM before the KV cache is sized (zero-copy parameter views cost nothing)."""
    try:
        params = {int(p.untyped_storage().data_ptr()) for p in model.parameters()}
        total = own = 0
        seen: set[int] = set()
        for t in tensors:
            ptr = int(t.untyped_storage().data_ptr())
            if ptr in seen:
                continue
            seen.add(ptr)
            n = t.untyped_storage().nbytes()
            total += n
            if ptr not in params:
                own += n
    except Exception as e:  # noqa: BLE001 - logging only, never fail the driver
        return f"unknown ({type(e).__name__})"
    return f"{total / 2**30:.2f} GiB, {own / 2**30:.2f} GiB not shared with model parameters"
