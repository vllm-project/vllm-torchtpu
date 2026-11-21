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
"""Chip-generation checks and VMEM queries for Fused EP MoE v3."""

import jax
from jax.experimental.pallas import tpu as pltpu

# Supported TPU generations. Device properties are read at build time,
# and check_generation rejects generations outside this set.
SUPPORTED_GENERATIONS = (7,)
# The share of the chip's VMEM a kernel may claim.
VMEM_FRACTION = 0.98


def chip_generation(info=None):
    """The TPU generation of the chip this process builds for."""
    if info is None:
        info = pltpu.get_tpu_info()
    return info.generation


def check_generation(info=None):
    """Raise ValueError for an unsupported TPU generation."""
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


def jax_version():
    """The installed JAX version string."""
    return jax.__version__
