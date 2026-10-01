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
"""A hand-written fake keeps the caller's dynamic batch dimension, and a
sharded output is sized for this rank's shard."""

import types

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.distributed import sharded_jax_op as sjo
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

pytestmark = pytest.mark.cpu_test


def _halve(x: jax.Array) -> jax.Array:
    return x[: x.shape[0] // 2]


def _op(name, fake):
    mesh = Mesh(np.array(jax.devices("cpu")[:1]), ("d",))
    return sharded_jax_op(
        f"sharded_jax_op_test::{name}",
        _halve,
        mesh=mesh,
        input_partition_specs=(P("d"),),
        output_partition_specs=P("d"),
        fake=fake,
    )


def _export(op):
    x = torch.zeros(64, 128, dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x, 0)
    # mark_dynamic makes export raise if anything specializes the dimension.
    torch._dynamo.export(lambda t: op(t) * 2)(x)


def test_hand_written_fake_keeps_the_dynamic_dimension():
    seen = []

    def fake(x):
        seen.append(x.shape)
        return x.new_empty((x.shape[0] // 2, x.shape[1]))

    _export(_op("halve", fake))
    assert seen and isinstance(seen[0][0], torch.SymInt)


def test_a_specializing_fake_is_caught():
    """Positive control: a fake that reads the dimension as an int fails."""

    def fake(x):
        return x.new_empty((int(x.shape[0]) // 2, x.shape[1]))

    with pytest.raises(Exception, match="(?i)constraint|specializ|dynamic"):
        _export(_op("halve_specialized", fake))


@pytest.mark.skipif(
    not sjo._stock_takes_output_specs(),
    reason="the installed torch_tpu sizes outputs through the derived class",
)
def test_stock_callable_gets_the_output_specs(monkeypatch):
    """A torch_tpu whose JaxCallable sizes outputs from its own specs must be
    given them: without, a P("d") output is sized for the global array, eight
    times this rank's rows on an 8-core mesh."""
    built = []
    real = torch.library.custom_op

    def capture(name, fn, **kwargs):
        built.append(fn)
        return real(name, fn, **kwargs)

    monkeypatch.setattr(torch.library, "custom_op", capture)
    _op("output_specs", lambda x: x)
    (wrapped,) = built
    (spec,) = sjo.pallas_impl._normalize_output_partition_specs(
        wrapped.output_partition_specs, 1
    )
    mesh8 = types.SimpleNamespace(axis_names=("d",), devices=np.empty((8,)))
    assert sjo.pallas_impl.get_local_shape((8 * 48, 4096), mesh8, spec) == (48, 4096)
