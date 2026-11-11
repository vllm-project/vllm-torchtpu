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
"""Unit tests for sharded_jax_op."""

import inspect
import uuid

import jax
import numpy as np
import torch
from jax.sharding import Mesh, PartitionSpec

from vllm_torchtpu.distributed import sharded_jax_op as sjo


def _cpu_mesh() -> Mesh:
    devs = np.array(jax.devices("cpu")[:1])
    return Mesh(devs, ("x",))


def test_sharded_jax_op_uses_custom_fake_impl(monkeypatch):
    captured: dict[str, object] = {}

    class FakeStockCallable:
        def __init__(self, *, output_partition_specs=None, **kwargs):
            def _proto(x: torch.Tensor) -> torch.Tensor:
                raise AssertionError("FakeTensorMode must dispatch to fake=")

            self.__globals__ = _proto.__globals__
            self.__name__ = str(kwargs["name"]).split("::")[-1]
            self.__qualname__ = self.__name__
            self.__annotations__ = _proto.__annotations__
            self.__signature__ = inspect.signature(_proto)
            captured["output_partition_specs"] = output_partition_specs
            captured["kwargs"] = kwargs

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            raise AssertionError("FakeTensorMode must dispatch to fake=")

    monkeypatch.setattr(sjo.pallas_impl, "JaxCallable", FakeStockCallable)

    mesh = _cpu_mesh()
    fake_calls: list[tuple[int, ...]] = []
    op_name = f"tpu_test::scale_{uuid.uuid4().hex[:8]}"

    def fn(x: jax.Array) -> jax.Array:
        return x * 2

    def custom_fake(x: torch.Tensor) -> torch.Tensor:
        fake_calls.append(tuple(x.shape))
        return x.new_empty(x.shape)

    op = sjo.sharded_jax_op(
        op_name,
        fn,
        mesh=mesh,
        input_partition_specs=(PartitionSpec(None, None),),
        output_partition_specs=PartitionSpec(None, None),
        fake=custom_fake,
    )
    with torch._subclasses.FakeTensorMode():
        x = torch.empty((5, 8), dtype=torch.float32)
        out = op(x)
    assert tuple(out.shape) == (5, 8)
    assert out.dtype == torch.float32
    assert fake_calls == [(5, 8)]
    assert captured["output_partition_specs"] == [PartitionSpec(None, None)]
    kw = captured["kwargs"]
    assert isinstance(kw, dict) and kw["static_argnums"] == ()
