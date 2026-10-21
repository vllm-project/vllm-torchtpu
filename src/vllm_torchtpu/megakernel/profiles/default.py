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
"""Model-agnostic default tile rule: one schedule for every bucket.

`default_tiles` returns the same tiles for every bucket: library field defaults,
the flat-ring all-reduce (`hier=0`) and the chip's VMEM / SMEM budgets. It does
not branch on the batch size, the rank count or the model; ring depths may be
refined at build time to fit the VMEM budget when `ExecSpec.tiles` is left
`None`.

`default_tiles` sizes for `chip` (default: TPU v7x, the only chip the
megakernel is validated on) and never queries devices: profiles evaluate it at
import time, before any JAX backend may be initialized. When unset tiles are
fitted at build time, the budgets come from `chip_budget()`: `CHIP_BUDGETS[kind]`
for a supported TPU, `MegakernelConfigError` for an unsupported TPU kind, or
`DEFAULT_CHIP` for non-TPU kinds (including CPU interpret runs).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import jax

from ..errors import MegakernelConfigError
from ..exec_spec import Tiles


@dataclass(frozen=True)
class ChipBudget:
    """On-chip memory budgets the default tiles are sized for."""

    name: str
    vmem_mib: int  # VMEM limit handed to Mosaic (MiB)
    smem_mib: int  # SMEM limit (MiB)


# TPU v7x: 64 MiB VMEM per core, 2 MiB kept free for compiler-internal scratch.
TPU_V7X = ChipBudget(name="tpu_v7x", vmem_mib=62, smem_mib=1)
DEFAULT_CHIP = TPU_V7X
# `jax.Device.device_kind` -> budget, for the chips the megakernel runs on.
CHIP_BUDGETS: dict[str, ChipBudget] = {"TPU7x": TPU_V7X}


@functools.cache
def _first_device_kind() -> str:
    return jax.devices()[0].device_kind


def chip_budget(device_kind: str | None = None) -> ChipBudget:
    """Budget of `device_kind` (default: the first JAX device's).

    Returns `CHIP_BUDGETS[kind]` if known; raises `MegakernelConfigError` if
    `kind` is a TPU kind (`kind.startswith("TPU")`) not in `CHIP_BUDGETS`;
    otherwise (CPU interpret or any non-TPU device) returns `DEFAULT_CHIP`.
    Without `device_kind` this initializes the JAX backend; call it only where
    the kernel is about to be built.
    """
    kind = _first_device_kind() if device_kind is None else device_kind
    if kind in CHIP_BUDGETS:
        return CHIP_BUDGETS[kind]
    if kind.startswith("TPU"):
        supported = ", ".join(sorted(CHIP_BUDGETS))
        raise MegakernelConfigError(
            f"Unsupported TPU device_kind {kind!r}; supported TPU kinds: {supported}"
        )
    return DEFAULT_CHIP


def default_tiles(
    B: int, *, moe_tp: bool = False, R: int = 8, chip: ChipBudget = DEFAULT_CHIP
) -> Tiles:
    """Model-agnostic default tiles (identical for every bucket).

    `B`, `moe_tp` and `R` are accepted for tile-rule call-site compatibility
    and ignored: field defaults everywhere, flat-ring all-reduce (`hier=0`),
    and `chip`'s VMEM / SMEM budgets.
    """
    del B, moe_tp, R
    return Tiles(hier=0, vmem_mib=chip.vmem_mib, smem_mib=chip.smem_mib)


__all__ = [
    "CHIP_BUDGETS",
    "DEFAULT_CHIP",
    "TPU_V7X",
    "ChipBudget",
    "chip_budget",
    "default_tiles",
]
