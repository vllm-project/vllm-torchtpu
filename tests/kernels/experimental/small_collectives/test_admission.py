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
"""Input checks and the default form choice of the small all-reduce. Both are
decided before any kernel is built, so they run on the CPU lane without a TPU."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

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


# (cores, hidden, rows, form): the bf16 blocks on each side of every limit of
# ``form_for``. Each pair of rows straddles one limit; the comment gives the
# block bytes per core and the quantity the limit compares.
_FORMS = [
    # 8 cores. One-shot while 7 x block <= 336 KiB, that is, block <= 48 KiB.
    (8, 4096, 6, "oneshot"),  # 48 KiB
    (8, 4096, 7, "twoshot"),  # 56 KiB
    # 8 cores. Two-shot while block x 3 <= 2560 KiB, that is, <= 853 KiB.
    (8, 4096, 106, "twoshot"),  # 848 KiB
    (8, 4096, 107, "xla"),  # 856 KiB
    # 16 cores. One-shot while 15 x block <= 336 KiB, that is, <= 22.4 KiB.
    (16, 2048, 5, "oneshot"),  # 20 KiB
    (16, 2048, 6, "twoshot"),  # 24 KiB
    # 16 cores. Two-shot while block x 4 <= 2560 KiB, that is, <= 640 KiB.
    (16, 4096, 80, "twoshot"),  # 640 KiB
    (16, 4096, 81, "xla"),  # 648 KiB
    # A width two-shot cannot split: one-shot while (n - 1) x block <= 630 KiB.
    (16, 7168, 3, "oneshot"),  # 42 KiB, 16 x 128 does not divide 7168
    (16, 7168, 4, "xla"),  # 56 KiB
    (8, 1920, 24, "oneshot"),  # 90 KiB, 8 x 128 does not divide 1920
    (8, 1920, 25, "xla"),  # 93.75 KiB
]


@pytest.mark.parametrize("cores,hidden,rows,form", _FORMS)
def test_form_for_around_each_limit(cores, hidden, rows, form):
    assert ar.form_for(rows, hidden, 2, cores) == form


@pytest.mark.parametrize(
    "cores,rows,form",
    [
        (8, 1, "oneshot"),
        (8, 4, "oneshot"),
        (8, 8, "twoshot"),
        (8, 64, "twoshot"),
        (8, 128, "xla"),
        (16, 1, "oneshot"),
        (16, 2, "twoshot"),
        (16, 32, "twoshot"),
        (16, 64, "xla"),
    ],
)
def test_form_for_follows_the_measured_fastest_at_hidden_6144(cores, rows, form):
    """The fastest method measured at each of these rows, hidden 6144, bf16
    (microbenchmark sweep on TPU v7x)."""
    assert ar.form_for(rows, 6144, 2, cores) == form


def _default_call_jaxpr(rows, hidden, num_cores):
    """The jaxpr of a default-form call inside a shard_map over a one-device
    mesh. ``num_cores`` only steers the form choice, so the trace needs no
    TPU; tracing the XLA form never reaches the kernel builder."""
    mesh = Mesh(np.array(jax.devices()[:1]), (AXIS,))

    def body(x):
        return ar.small_all_reduce(x, num_cores=num_cores, axis=AXIS)

    fn = jax.shard_map(body, mesh=mesh, in_specs=P(AXIS), out_specs=P(AXIS))
    return str(jax.make_jaxpr(fn)(jnp.zeros((rows, hidden), jnp.bfloat16)))


@pytest.mark.parametrize(
    "rows,hidden,num_cores",
    [(32, 7168, 16), (4, 7168, 16), (64, 6144, 16), (128, 6144, 8)],
    ids=[
        "no_split_32_rows",
        "no_split_4_rows",
        "above_two_shot_16",
        "above_two_shot_8",
    ],
)
def test_default_form_takes_xla_all_reduce(rows, hidden, num_cores):
    """At a width two-shot cannot split, one-shot loses to XLA's all-reduce
    above about 42 KiB per core on 16 cores. Two-shot loses to it above about
    640 KiB per core on 16 cores and about 850 KiB per core on 8 cores. The
    default form then calls ``jax.lax.psum``."""
    jaxpr = _default_call_jaxpr(rows, hidden, num_cores)
    assert "psum" in jaxpr
    assert "pallas_call" not in jaxpr
