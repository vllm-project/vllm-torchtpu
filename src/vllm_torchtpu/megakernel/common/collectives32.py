# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP32 collectives: a barrier across the 32 ranks of the ``tp`` mesh axis."""

import jax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu


def barrier():
    rank = jax.lax.axis_index("tp")
    sem = tpu.get_barrier_semaphore()
    for offset in range(1, 32):
        pl.semaphore_signal(sem, 1, device_id=(rank ^ offset,), device_id_type=pl.DeviceIdType.MESH)
    pl.semaphore_wait(sem, 31)
