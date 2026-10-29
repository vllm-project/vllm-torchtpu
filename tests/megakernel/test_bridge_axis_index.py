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
"""The SPMD-safe axis_index: XLA-level calls over any shard_map mesh go through
Pallas; calls under axes bound beyond it (a core_map kernel body) stay stock."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import core as jax_core
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.megakernel import bridge

pytestmark = pytest.mark.cpu_test


def _traced(body):
    # A mesh that is not a bridge kernel mesh, as a pipeline stage's.
    mesh = Mesh(np.asarray(jax.devices()[:1]), axis_names=("stage_sp",))
    mapped = jax.shard_map(
        body,
        mesh=mesh,
        in_specs=P("stage_sp"),
        out_specs=P("stage_sp"),
        check_vma=False,
    )
    with bridge.spmd_safe_axis_index():
        return str(jax.make_jaxpr(mapped)(jnp.zeros((1,), jnp.int32)))


def test_axis_index_over_a_stage_mesh_goes_through_pallas():
    jaxpr = _traced(lambda x: x + jax.lax.axis_index("stage_sp"))
    assert jaxpr.count("spmd_safe_axis_index") == 1


def test_axis_index_under_axes_beyond_the_mesh_stays_stock():
    # jax 0.11 traces a core_map kernel body with its core mesh axes bound
    # next to the enclosing shard_map's.
    def body(x):
        with jax_core.extend_axis_env_nd([("count_core", 2)]):
            jax.lax.axis_index("count_core")
        return x

    assert "spmd_safe_axis_index" not in _traced(body)
