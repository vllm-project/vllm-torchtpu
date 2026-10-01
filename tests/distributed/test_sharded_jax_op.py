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
"""sharded_jax_op: a sharded output is sized for this rank's shard."""

import inspect
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec
from torch_tpu._internal.pallas import pallas as pallas_impl

from vllm_torchtpu.distributed import sharded_jax_op

pytestmark = pytest.mark.cpu_test


def _rows(x: jax.Array) -> jax.Array:
    return x


@pytest.mark.skipif(
    "output_partition_specs"
    not in inspect.signature(pallas_impl.JaxCallable).parameters,
    reason="this torch_tpu's JaxCallable takes no output_partition_specs",
)
def test_the_callable_sizes_a_sharded_output_per_rank(monkeypatch):
    made = {}

    class Recording(pallas_impl.JaxCallable):
        def __init__(self, **kwargs):
            made.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sharded_jax_op, "_SHARDED_CALLABLE_CLS", Recording)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("ep",))
    sharded_jax_op.sharded_jax_op(
        "vllm_torchtpu_test::sharded_rows",
        _rows,
        mesh=mesh,
        input_partition_specs=[PartitionSpec("ep")],
        output_partition_specs=PartitionSpec("ep"),
    )

    # The specs stock sizes the output with, on an 8-way mesh: 96 global
    # rows are 12 on each rank, not 96.
    (spec,) = pallas_impl._normalize_output_partition_specs(
        made.get("output_partition_specs"), 1
    ) or [None]
    eight_ranks = SimpleNamespace(axis_names=("ep",), devices=np.empty(8))
    local = pallas_impl.get_local_shape((96, 1024), eight_ranks, spec)
    assert tuple(local) == (12, 1024)
