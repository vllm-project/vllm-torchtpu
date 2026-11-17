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
"""The small all-reduce over the TP pairs of a (dp, tp) = (4, 2) mesh, on the
8 cores of one host: each core's result against the float64 sum of its pair's
blocks, with the core's TP index handed in as data (the torch bridge's way,
``small_all_reduce_op._build_tp_all_reduce``). The interpret-mode checks of
the same call, with the race detector, are in test_tp_pair_interpret.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.experimental.small_collectives import all_reduce as ar
from vllm_torchtpu.kernels.experimental.small_collectives import common

from .oracle import N, blocks, exact_sum, sum_tolerance, within

pytestmark = pytest.mark.multichip

DP, TP = 4, 2
AXES = ("dp_", "tp_")


@pytest.fixture(scope="module")
def pair_mesh():
    """The eight cores as (dp, tp). A fixture, not a `skipif`: asking JAX for
    devices at collection would claim the chip for the pytest parent."""
    if jax.default_backend() != "tpu" or jax.device_count() < N:
        pytest.skip("needs the 8 TPU cores of one host")
    return Mesh(np.array(jax.devices()[:N]).reshape(DP, TP), AXES)


def _pair_all_reduce(mesh, per_core, form):
    n, rows, hidden = per_core.shape
    spec = P(AXES)
    sharding = NamedSharding(mesh, spec)
    x = jax.device_put(jnp.asarray(per_core).reshape(n * rows, hidden), sharding)
    rank = jax.device_put(jnp.asarray([i % TP for i in range(n)], jnp.int32), sharding)

    def body(block, r):
        with common.mesh_axes(AXES):
            return ar.small_all_reduce(
                block, num_cores=TP, axis="tp_", rank=r, form=form
            )

    fn = jax.jit(
        jax.shard_map(
            body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
        )
    )
    out = np.asarray(fn(x, rank))
    return out.reshape(n, rows, hidden)


@pytest.mark.parametrize("rows", [8, 16, 48, 64, 128])
@pytest.mark.parametrize("form", ["oneshot", None])
def test_each_core_holds_its_pairs_sum(pair_mesh, rows, form):
    per_core = blocks((N, rows, 4096), seed=rows)
    got = _pair_all_reduce(pair_mesh, per_core, form)
    for core in range(N):
        first = TP * (core // TP)
        pair = per_core[first : first + TP]
        ok = within(got[core], exact_sum(pair), sum_tolerance(pair))
        assert ok.all(), f"core {core}: {np.count_nonzero(~ok)} elements off"
