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
"""XLA optimization-barrier fences for token-window micro-batching.

Micro-batching splits the token dimension into ``N`` windows so that XLA can
overlap one window's collective communication (TP All-Reduce, DP All-Gather,
EP All-to-All / Reduce-Scatter) with another window's TensorCore/SparseCore
compute. Without explicit HLO data-dependency barriers, however, XLA's post-order
scheduler and horizontal fusion passes can:

1. Horizontally fuse sibling window operators (e.g., ``abs_reduce_fusion`` across
   Window 0 and Window 1 in GDN linear attention, which caused a 3.3x VPU launch
   regression on TPU v7x).
2. Reorder or batch window compute and collectives across windows, defeating the
   intended compute/communication overlap.

This module registers two custom ops backed by ``jax.lax.optimization_barrier``:
- ``torch.ops.pallas.microbatch_fence(x) -> Tensor``: 1-tensor barrier.
- ``torch.ops.pallas.microbatch_fence_pair(x, y) -> (Tensor, Tensor)``: 2-tensor
  coupled barrier ``optimization_barrier((x, y))``.

In XLA HLO, ``optimization_barrier((a_done, b_in))`` creates a directed path
from ``a_done`` to every consumer of ``b_in`` while preserving zero runtime copy
cost. Placing ``microbatch_fence_pair`` between Window ``w-1``'s completed
compute output and Window ``w``'s next compute input:
- Makes cross-window horizontal fusion mathematically illegal in XLA's
  ``InstructionFusion::ShouldFuse`` reachability check (fusing them would form a
  cycle through the ``optimization-barrier`` HLO instruction).
- Serializes Window ``w``'s compute after Window ``w-1``'s compute while leaving
  Window ``w-1``'s collective and Window ``w``'s compute as parallel,
  concurrently schedulable sibling branches between consecutive barriers.
"""

from enum import StrEnum

import jax
import torch
from torch_tpu._internal import pallas


class FenceMode(StrEnum):
    """Controls where ``jax.lax.optimization_barrier`` ops are inserted."""

    NONE = "none"
    WINDOW = "window"
    STAGE = "stage"
    SUBSTAGE = "substage"


def parse_fence_mode(value: str | FenceMode | None) -> FenceMode:
    """Parse a string or ``FenceMode`` into a validated ``FenceMode``."""
    if isinstance(value, FenceMode):
        return value
    if value is None or value.strip() == "":
        return FenceMode.SUBSTAGE
    normalized = value.strip().lower()
    try:
        return FenceMode(normalized)
    except ValueError as exc:
        valid = [m.value for m in FenceMode]
        raise ValueError(
            f"Invalid microbatch fence mode {value!r}. Valid options: {valid}"
        ) from exc


def _jax_microbatch_fence(x: jax.Array) -> jax.Array:
    return jax.lax.optimization_barrier(x)


def _jax_microbatch_fence_pair(
    x: jax.Array,
    y: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return jax.lax.optimization_barrier((x, y))


def _register_fence_ops() -> None:
    if not hasattr(torch.ops.pallas, "microbatch_fence"):
        op_single = pallas.jax_op(
            "pallas::microbatch_fence",
            _jax_microbatch_fence,
        )
        op_single.register_fake(lambda x: torch.empty_like(x))
        op_single.register_kernel("cpu")(lambda x: x.clone())

    if not hasattr(torch.ops.pallas, "microbatch_fence_pair"):
        op_pair = pallas.jax_op(
            "pallas::microbatch_fence_pair",
            _jax_microbatch_fence_pair,
        )
        op_pair.register_fake(lambda x, y: (torch.empty_like(x), torch.empty_like(y)))
        op_pair.register_kernel("cpu")(lambda x, y: (x.clone(), y.clone()))


_register_fence_ops()


def microbatch_fence(x: torch.Tensor) -> torch.Tensor:
    """Apply a single-tensor XLA ``optimization_barrier``."""
    return torch.ops.pallas.microbatch_fence(x)


def microbatch_fence_pair(
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply a coupled 2-tensor XLA ``optimization_barrier((x, y))``.

    In HLO, consumers of ``y`` become reachable from producers of ``x`` and
    vice-versa, preventing horizontal fusion between producer(x) and consumer(y)
    and enforcing that producer(x) completes before consumer(y) starts.
    """
    return torch.ops.pallas.microbatch_fence_pair(x, y)


def fence_windows(
    windows: list[torch.Tensor],
    mode: FenceMode,
) -> list[torch.Tensor]:
    """Apply fencing across a list of window tensors according to ``mode``.

    - ``NONE``: returns ``windows`` unchanged.
    - ``WINDOW``: applies ``microbatch_fence`` to each window independently.
    - ``STAGE`` / ``SUBSTAGE``: chains ``microbatch_fence_pair`` across adjacent
      windows so the stage boundary acts as a coupled synchronization barrier
      across all windows.
    """
    if mode == FenceMode.NONE or not windows:
        return windows
    if mode == FenceMode.WINDOW:
        return [microbatch_fence(w) for w in windows]
    if len(windows) == 1:
        return [microbatch_fence(windows[0])]
    out = list(windows)
    for i in range(1, len(out)):
        out[i - 1], out[i] = microbatch_fence_pair(out[i - 1], out[i])
    return out
