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
"""`build_jax_step_op` with TPU tensors under `torch.compile(backend="tpu",
fullgraph=True)`: a donated output updates its input in place, and the
outputs of a sharded spec have the shapes the fake kernel gives them."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
import torch
from jax.sharding import PartitionSpec as P
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_torchtpu.megakernel_runtime.jax_bridge import build_jax_step_op


def _tpu_mesh() -> jax.sharding.Mesh:
    pytest.importorskip("torch_tpu")
    from torch_tpu._internal.pallas import get_pallas_mesh

    return get_pallas_mesh(axis_names=("tp",))


@pytest.mark.tpu_test
def test_compiled_step_donates_and_shards_on_tpu() -> None:
    mesh = _tpu_mesh()
    tp = int(mesh.devices.size)

    def fn(x: jax.Array, state: jax.Array) -> tuple[Any, Any]:
        return x * 2.0, state + jnp.sum(x).astype(state.dtype)

    # x is this rank's shard of a ("tp",)-sharded array, the output is
    # replicated; the state is replicated and donated.
    step = build_jax_step_op(
        fn,
        mesh=mesh,
        in_specs=(P("tp"), P()),
        out_specs=(P(), P()),
        donate={1: 1},
        name="mk_tests::jax_step_tpu",
    )
    compiled = torch.compile(step, backend="tpu", fullgraph=True, dynamic=False)
    x = torch.arange(8, dtype=torch.float32).to("tpu")
    state = torch.zeros(4, dtype=torch.float32).to("tpu")
    ptr = state.data_ptr()
    with FakeTensorMode():
        fake = step.op([torch.empty(8, device="meta"), torch.empty(4, device="meta")])
    want_shapes = [tuple(t.shape) for t in fake]
    assert want_shapes == [(8 * tp,), (4,)]

    y, new_state = compiled(x, state)
    assert [tuple(y.shape), tuple(new_state.shape)] == want_shapes
    assert state.data_ptr() == ptr and torch.equal(new_state.cpu(), state.cpu())
    # Every rank holds the same shard here, so the global x is x repeated.
    full = torch.arange(8, dtype=torch.float32).repeat(tp)
    assert torch.equal(y.cpu(), full * 2.0)
    assert torch.equal(state.cpu(), torch.full((4,), float(full.sum())))
    compiled(x, state)
    assert torch.equal(state.cpu(), torch.full((4,), 2.0 * float(full.sum())))
