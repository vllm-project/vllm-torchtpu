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
"""Which blocks and forms the small-collective kernels accept, and which form
the all-reduce picks; checked before any kernel is built, so on the host."""

import jax.numpy as jnp
import pytest

from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar
from vllm_torchtpu.kernels.experimental.small_collectives import (
    reduce_scatter as rs,
)

pytestmark = pytest.mark.cpu_test

AXIS = "d"


def _block(rows, hidden=4096):
    return jnp.zeros((rows, hidden), jnp.bfloat16)


@pytest.mark.parametrize(
    "rows,cores,form",
    [
        (1, 8, "oneshot_xor"),
        (4, 8, "oneshot_xor"),
        (5, 8, "twoshot"),
        (8, 8, "hier_halves"),
        (16, 8, "twoshot"),
        (23, 8, "twoshot"),
        (24, 8, "hier_2stage"),
        (128, 8, "hier_2stage"),
        (4, 4, "oneshot"),
        (8, 4, "twoshot"),
        (24, 4, "twoshot"),
    ],
)
def test_all_reduce_form_by_bytes_and_rows(rows, cores, form):
    """One-shot up to 32 KiB per core (4 rows of 4096 bf16); above that the
    8-core forms by row count, two-shot on other core counts."""
    assert ar.dispatch_form_for(rows, 4096, 2, cores) == form


def test_all_reduce_refuses_a_width_that_does_not_split():
    with pytest.raises(ValueError, match="lanes"):
        ar.small_all_reduce(_block(16, 4000), num_cores=8, axis=AXIS)


@pytest.mark.parametrize(
    "form,cores,match",
    [
        ("hier_halves", 4, "8 cores"),
        ("hier_2stage", 16, "8 cores"),
        ("oneshot_xor", 6, "power-of-two"),
        ("threeshot", 8, "unknown"),
    ],
)
def test_all_reduce_refuses_a_form_the_cores_cannot_run(form, cores, match):
    with pytest.raises(ValueError, match=match):
        ar.small_all_reduce(
            _block(8, cores * 512), num_cores=cores, axis=AXIS, form=form
        )


def test_reduce_scatter_refuses_a_width_off_the_lane_tile():
    with pytest.raises(ValueError, match="lanes"):
        rs.small_reduce_scatter(_block(8, 4000), num_cores=8, axis=AXIS)


def test_reduce_scatter_refuses_rows_that_do_not_split():
    with pytest.raises(ValueError, match="split"):
        rs.small_reduce_scatter(_block(12), num_cores=8, axis=AXIS)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_cores": 4, "form": "hier"}, "8 cores"),
        ({"num_cores": 8, "form": "ring"}, "unknown"),
        ({"num_cores": 8, "block_order": (0, 1, 2, 3, 4, 5, 6, 6)}, "permutation"),
        ({"num_cores": 4, "block_order": (0, 1, 2, 3, 4, 5, 6, 7)}, "permutation"),
    ],
)
def test_reduce_scatter_refuses_a_form_or_order_it_cannot_run(kwargs, match):
    with pytest.raises(ValueError, match=match):
        rs.small_reduce_scatter(_block(16), axis=AXIS, **kwargs)
