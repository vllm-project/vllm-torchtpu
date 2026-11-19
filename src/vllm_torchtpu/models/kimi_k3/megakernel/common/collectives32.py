# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

# Source provenance: inferact/k3-megakernel@30c9ba966b
# src/vllm_torchtpu/megakernel/common/collectives32.py
"""TP32 collectives: a barrier across the 32 ranks of the ``tp`` mesh axis."""

from __future__ import annotations

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

from .constants import AXIS_TP, TP_PEERS, TP_SIZE


def barrier() -> None:
    """Synchronize all 32 ranks across the ``tp`` mesh axis via barrier semaphore."""
    rank = jax.lax.axis_index(AXIS_TP)
    sem = tpu.get_barrier_semaphore()
    for offset in range(1, TP_SIZE):
        pl.semaphore_signal(
            sem,
            1,
            device_id=(rank ^ offset,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(sem, TP_PEERS)
