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
"""The chip this package runs on, and the jax releases behind it."""

import jax
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.fused_moe.v3.layout import SUBLANES, align_up

# The chip generations the package has been built, tested and timed on.
# Everything else is read off the device record at build time, so another
# generation would compile and run untested, so it raises an error instead.
SUPPORTED_GENERATIONS = (7,)
# The share of the chip's VMEM a kernel may claim.
VMEM_FRACTION = 0.98


def chip_generation(info=None):
    """The TPU generation of the chip this process builds for."""
    if info is None:
        info = pltpu.get_tpu_info()
    return info.generation


def check_generation(info=None):
    """Raises ValueError on a generation the package has not
    been run on."""
    generation = chip_generation(info)
    if generation not in SUPPORTED_GENERATIONS:
        raise ValueError(
            f"this chip is TPU generation {generation}. The package has been "
            f"built and tested on generations {SUPPORTED_GENERATIONS} only"
        )


def supported_on(device):
    """Whether the package runs on `device` (a jax.Device)."""
    return (
        device.platform == "tpu"
        and pltpu.get_tpu_info().generation in SUPPORTED_GENERATIONS
    )


def vmem_limit(fraction=VMEM_FRACTION):
    """The VMEM budget of one kernel: `fraction` of the chip's capacity
    (Config.vmem_fraction)."""
    return int(pltpu.get_tpu_info().vmem_capacity_bytes * fraction)


def array_vmem_bytes(shape, dtype, info):
    """Bytes a VMEM array of this shape and dtype occupies: the minor
    dimension padded to the lane count, the second-minor to the dtype's
    sublane tiling, both read off the device record."""
    itemsize = jnp.dtype(dtype).itemsize
    if len(shape) == 1:
        return align_up(shape[0], info.num_lanes) * itemsize
    lanes = align_up(shape[-1], info.num_lanes)
    sublanes = align_up(shape[-2], info.get_sublane_tiling(dtype))
    rows = 1
    for dim in shape[:-2]:
        rows *= dim
    return rows * sublanes * lanes * itemsize


def check_u32_sublane_tile(info=None):
    """Raises ValueError on a chip whose 32-bit sublane tiling
    is not layout.SUBLANES, the tile the packed four-bit weight layout is
    written for."""
    if info is None:
        info = pltpu.get_tpu_info()
    queried = info.get_sublane_tiling(jnp.uint32)
    if queried != SUBLANES:
        raise ValueError(
            f"this chip tiles 32-bit words to {queried} sublanes. The packed "
            f"four-bit weight layout is written for {SUBLANES}"
        )


def jax_version():
    """The installed JAX version string."""
    return jax.__version__
