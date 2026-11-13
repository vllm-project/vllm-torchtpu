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
"""Input checks of the small all-reduce that raise before any kernel is built,
so they run on the CPU lane without a TPU."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar

from .oracle import AXIS


def test_refuses_a_dtype_other_than_bf16():
    with pytest.raises(ValueError, match="float32"):
        ar.small_all_reduce(jnp.zeros((4, 4096), jnp.float32), num_cores=8, axis=AXIS)


def test_refuses_forced_two_shot_at_a_width_that_does_not_split():
    """Hidden 1920 is 15 lane tiles, which do not split over 4 cores."""
    with pytest.raises(ValueError, match="two-shot"):
        ar.small_all_reduce(
            jnp.zeros((4, 1920), jnp.bfloat16), num_cores=4, axis=AXIS, form="twoshot"
        )
