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
"""Scoped-VMEM budgets for Pallas kernels, resolved from the live device.

Physical VMEM per TensorCore differs sharply across TPU generations --
v6e has 128 MiB, v7x has ~64 MiB -- so a hard-coded byte budget is never
portable. Kernels across this repo historically declared

    DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024

and handed it straight to ``pltpu.CompilerParams``. On v7x that claims
36 MiB more than the chip has, so instead of degrading to a smaller block
the compiler plans a program that cannot fit and fails outright, e.g.

    Ran out of memory in memory space vmem.
    Used 77.69M of 63.94M vmem. Exceeded vmem capacity by 13.75M.

Two different budgets are needed, and conflating them is what bites:

``compiler_vmem_limit_bytes()``
    What to hand ``pltpu.CompilerParams``. This should be the truth about
    the hardware, clamped by the historic 100 MiB convention so that
    large-VMEM generations keep their previously benchmarked behaviour.

``planning_vmem_budget_bytes()``
    What a kernel's own ``vmem_need`` estimate should be compared against
    when choosing a block size. This leaves headroom below the ceiling,
    because analytic window arithmetic cannot see VREG spills or Mosaic
    bookkeeping.

Both are overridable with ``TPU_KERNEL_VMEM_LIMIT_BYTES`` for sweeps.
"""

import functools

from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu import envs

MIB = 1024 * 1024

# Historic repo-wide budget, retained as an upper bound so that generations
# with plenty of VMEM keep the behaviour they were benchmarked with.
LEGACY_VMEM_LIMIT_BYTES = 100 * MIB

# Headroom left below the physical ceiling when a kernel sizes its own
# blocks from an analytic estimate.
DEFAULT_VMEM_SAFETY_FRACTION = 0.94

# Used when the device kind is unknown (CPU test hosts, or a generation
# whose info lookup is unsupported). Deliberately conservative.
FALLBACK_VMEM_CAPACITY_BYTES = 64 * MIB


@functools.cache
def physical_vmem_bytes() -> int:
    """VMEM capacity per TensorCore, probed once from the live device."""
    try:
        return int(pltpu.get_tpu_info().vmem_capacity_bytes)
    except (ValueError, AttributeError):
        # get_tpu_info raises for unsupported device kinds (e.g. CPU hosts).
        return FALLBACK_VMEM_CAPACITY_BYTES


def _env_override() -> int | None:
    return envs.TPU_KERNEL_VMEM_LIMIT_BYTES


def compiler_vmem_limit_bytes() -> int:
    """Budget to pass to ``pltpu.CompilerParams(vmem_limit_bytes=...)``.

    ``min(LEGACY_VMEM_LIMIT_BYTES, physical capacity)``:

    * v6e (128 MiB) -> 100 MiB, i.e. unchanged from the old constant.
    * v7x (~64 MiB) -> ~64 MiB, i.e. the truth instead of a 36 MiB lie.

    Because the result is never below the physical capacity, no program
    that fits the chip today can be rejected by this change.
    """
    override = _env_override()
    if override is not None:
        return override
    return min(LEGACY_VMEM_LIMIT_BYTES, physical_vmem_bytes())


def planning_vmem_budget_bytes() -> int:
    """Budget for comparing against a kernel's own ``vmem_need`` estimate.

    Same clamp as :func:`compiler_vmem_limit_bytes` but scaled by
    :data:`DEFAULT_VMEM_SAFETY_FRACTION`, so a kernel shrinks its block
    before the program reaches the hardware ceiling.
    """
    override = _env_override()
    if override is not None:
        return override
    derived = int(physical_vmem_bytes() * DEFAULT_VMEM_SAFETY_FRACTION)
    return min(LEGACY_VMEM_LIMIT_BYTES, derived)


def resolve_compiler_limit(explicit: int | None) -> int:
    """``explicit`` when a caller pinned a budget, else the host default."""
    return compiler_vmem_limit_bytes() if explicit is None else explicit
