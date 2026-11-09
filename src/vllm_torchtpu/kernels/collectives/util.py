# Copyright 2025 Google LLC
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
"""utilities for collective kernels."""

import functools

from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def local_barrier(left_neighbor, right_neighbor, double_barrier=True):
    """Performs a barrier with neighbors on the global barrier semaphore.

    Optionally performs a second barrier, which prevents a potential race
    when reusing the same collective_id across kernel invocations.

    Args:
      left_neighbor: Left neighbor device id.
      right_neighbor: Right neighbor device id.
      double_barrier: Whether to perform a second barrier.
    """
    barrier_sem = pltpu.get_barrier_semaphore()
    for neighbor in [left_neighbor, right_neighbor]:
        pl.semaphore_signal(
            barrier_sem,
            inc=1,
            device_id=(neighbor,),
            device_id_type=pl.DeviceIdType.MESH,
        )
    pl.semaphore_wait(barrier_sem, 2)
    if double_barrier:
        # The double-barrier prevents a race condition where one neighbor can
        # re-enter the kernel again on a subsequent call and increment the
        # barrier semaphore a second time. This would unblock the current device
        # even if the other neighbor is not ready yet.
        # To implement a double-barrier, we stack-allocate a second REGULAR
        # semaphore using run_scoped.
        @functools.partial(pl.run_scoped, second_barrier=pltpu.SemaphoreType.REGULAR)
        def _(second_barrier):
            for neighbor in [left_neighbor, right_neighbor]:
                pl.semaphore_signal(
                    second_barrier,
                    inc=1,
                    device_id=(neighbor,),
                    device_id_type=pl.DeviceIdType.MESH,
                )
            pl.semaphore_wait(second_barrier, 2)
