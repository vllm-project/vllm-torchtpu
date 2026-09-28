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
"""Per-TPU-generation tuning knobs for the DeepSeek-V4 mHC Pallas kernels.

This module is the single place to edit mHC kernel defaults.

Developers tuning a new generation should edit ``_TUNED_BY_TPU_VERSION``
below. Everything can also be overridden at run time without a code change,
via the environment variables named in each field's docstring — useful for
sweeps and for A/B benchmarking a candidate before making it the default.

Physical VMEM per TensorCore differs sharply across generations (v6e has
128 MiB, v7x has ~64 MiB), so a hard-coded byte budget is never portable.
``vmem_limit_bytes`` is therefore derived from the *live* device via
``pltpu.get_tpu_info().vmem_capacity_bytes`` — the same idiom used by
``ragged_paged_attention/v3`` and ``flash_attention``.
"""

import dataclasses
import functools

from vllm_torchtpu import envs
from vllm_torchtpu.kernels import vmem_budget
from vllm_torchtpu.kernels.ragged_paged_attention.v3.util import get_tpu_version

# The VMEM constants and the live-device probe are shared with every other
# Pallas kernel in this repo; see ``vllm_torchtpu.kernels.vmem_budget``.
MIB = vmem_budget.MIB
LEGACY_VMEM_LIMIT_BYTES = vmem_budget.LEGACY_VMEM_LIMIT_BYTES
DEFAULT_VMEM_SAFETY_FRACTION = vmem_budget.DEFAULT_VMEM_SAFETY_FRACTION
FALLBACK_VMEM_CAPACITY_BYTES = vmem_budget.FALLBACK_VMEM_CAPACITY_BYTES


@dataclasses.dataclass(frozen=True)
class MhcTuning:
    """Tunable mHC kernel defaults for one TPU generation."""

    token_block_size: int
    """Tokens per Pallas grid step. Override: ``TPU_DSV4_MHC_TOKEN_BLOCK``."""

    adaptive_grid: bool = False
    """Cap the block so ``grid >= 2`` and Pallas can double-buffer its DMAs.
    Default ``False``: on 256x256 MXU TPUs (v6e/v7x), ``fn`` is VMEM-resident
    (``BlockSpec(..., lambda i: (0, 0))``), so halving ``tb`` at ``T <= 128``
    doubles the K=16384 MXU passes and 20-step Sinkhorn VPU loop (+31% slower
    at T=64, +18% slower at T=128 on hardware). With ``adaptive_grid=False``,
    ``tb = min(token_block_size, padded_n)`` naturally selects ``tb=32`` at
    T=32, ``tb=64`` at T=64, ``tb=128`` at T=128, and ``grid >= 2`` for all
    ``T >= 256``. Override: ``TPU_DSV4_MHC_ADAPTIVE_GRID`` (``1``/``0``)."""

    vmem_limit_bytes: int | None = None
    """Scoped-VMEM budget, or ``None`` to derive it from the live device.
    Override: ``TPU_DSV4_MHC_VMEM_LIMIT_BYTES``."""


# ---------------------------------------------------------------------------
# Edit this table to retune a TPU generation.
# ---------------------------------------------------------------------------
_TUNED_BY_TPU_VERSION: dict[int, MhcTuning] = {
    # v6e (Trillium): 128 MiB VMEM, 256x256 MXU. Hardware sweep on rish-tpu-v6e
    # measured tb=128 at 1.68x faster than tb=32 at T=256 (47.36 us vs 79.38 us)
    # and 1.60x faster at T=2048 (426.05 us vs 680.87 us).
    6: MhcTuning(token_block_size=128),
    # v7x (Ironwood): ~64 MiB VMEM, 256x256 MXU. tb=128 fills half the MXU
    # and still fits (~38 MiB); tb=256 does not (77.69M > 63.94M, measured
    # as an XLA compile failure on rish-tpu7x-8).
    7: MhcTuning(token_block_size=128),
}

_FALLBACK_TUNING = MhcTuning(
    token_block_size=32, vmem_limit_bytes=LEGACY_VMEM_LIMIT_BYTES
)


@functools.cache
def _tuning() -> MhcTuning:
    """The tuning entry for this host's TPU generation."""
    try:
        version = get_tpu_version()
    except (AssertionError, IndexError, RuntimeError):
        return _FALLBACK_TUNING
    return _TUNED_BY_TPU_VERSION.get(version, _FALLBACK_TUNING)


def vmem_limit_bytes() -> int:
    """Scoped-VMEM budget for the mHC kernels on this host.

    Resolution order: ``TPU_DSV4_MHC_VMEM_LIMIT_BYTES`` env override, then
    the generation's table entry, then
    :func:`vmem_budget.planning_vmem_budget_bytes`, i.e.
    ``min(LEGACY_VMEM_LIMIT_BYTES, capacity * DEFAULT_VMEM_SAFETY_FRACTION)``.

    The ``min`` keeps large-VMEM generations on their previously benchmarked
    budget while clamping small-VMEM ones (v7x) below the hardware ceiling.
    The mHC kernels size their own blocks from an analytic ``vmem_need``
    estimate, which is why this is the *planning* budget and not the raw
    compiler ceiling.
    """
    if envs.TPU_DSV4_MHC_VMEM_LIMIT_BYTES is not None:
        return envs.TPU_DSV4_MHC_VMEM_LIMIT_BYTES
    configured = _tuning().vmem_limit_bytes
    if configured is not None:
        return configured
    return vmem_budget.planning_vmem_budget_bytes()


def resolve_vmem_limit_bytes(explicit: int | None) -> int:
    """``explicit`` when a caller pinned a budget, else this host's default."""
    return vmem_limit_bytes() if explicit is None else explicit


def default_token_block_size() -> int:
    """Default tokens per grid step, before any adaptive-grid capping."""
    if envs.TPU_DSV4_MHC_TOKEN_BLOCK is not None:
        return envs.TPU_DSV4_MHC_TOKEN_BLOCK
    return _tuning().token_block_size


def adaptive_grid_enabled() -> bool:
    """Whether to cap the token block so Pallas keeps ``grid >= 2``."""
    if envs.TPU_DSV4_MHC_ADAPTIVE_GRID is not None:
        return envs.TPU_DSV4_MHC_ADAPTIVE_GRID
    return _tuning().adaptive_grid
