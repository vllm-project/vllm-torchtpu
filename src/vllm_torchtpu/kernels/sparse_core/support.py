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
"""Which TPUs the SparseCore kernels in this package can target."""

from jax.experimental.pallas import tpu as pltpu

# Chips the SparseCore kernels in this package are validated on. Every other
# chip (no SparseCore, or TPU8 until the kernels are ported) takes the
# TensorCore fallback.
_SC_KERNEL_CHIPS = frozenset(
    {
        pltpu.ChipVersion.TPU_V5P,
        pltpu.ChipVersion.TPU_V6E,
        pltpu.ChipVersion.TPU_7,
        pltpu.ChipVersion.TPU_7X,
    }
)


def get_sparse_core_info(tpu_info: pltpu.TpuInfo | None = None):
    """Returns the SparseCore info if these kernels support this chip, else None."""
    if tpu_info is not None:
        info = tpu_info
    else:
        try:
            info = pltpu.get_tpu_info()
        except ValueError:  # Non-TPU host.
            return None
    if info.chip_version not in _SC_KERNEL_CHIPS:
        return None
    return info.sparse_core
