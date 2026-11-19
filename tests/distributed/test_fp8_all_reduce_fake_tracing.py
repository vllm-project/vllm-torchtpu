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
"""The TP all-reduce jax_op built with the fp8 wire
(`distributed.pallas_all_reduce._make_op(..., fp8)`, TPU_TP_FP8_ALL_REDUCE):
its fake fits the op schema with a symbolic token count, under
``FakeTensorMode`` and under ``torch.compile(fullgraph=True)`` over meta
tensors, as Dynamo runs it in the runner's warmup; and `all_reduce` hands
every size to the op, so the fp8 switch adds no graph branch."""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

pytest.importorskip("torch_tpu")

from vllm_torchtpu.distributed import pallas_all_reduce as par  # noqa: E402

H = 6144
TOKENS = 4096  # hint of the symbolic token count


def _mesh(n=8):
    devs = jax.devices("cpu")
    if len(devs) < n:
        pytest.skip(f"needs {n} CPU devices (XLA_FLAGS set too late)")
    return Mesh(np.asarray(devs[:n]), axis_names=("d",))


@pytest.fixture(params=["a2a", "rs", "a2a16"])
def op(request):
    fp8 = par.Fp8Config(min_tokens=2048, mode=request.param, granularity="g128")
    return par._make_op(_mesh(), 589824, fp8)


def test_fake_symbolic(op):
    assert len(op._opoverload._schema.returns) == 1
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        num = seed.shape[0]
        assert isinstance(num, torch.SymInt)
        x = torch.empty((num, H), dtype=torch.bfloat16)
        rank = torch.zeros((1, 1), dtype=torch.int32)
        out = op(x, rank)
    assert out.dtype == torch.bfloat16
    assert out.shape[0] == num and out.shape[1] == H


def test_compile_with_a_dynamic_token_dim(op, monkeypatch):
    rank = torch.zeros((1, 1), dtype=torch.int32, device="meta")
    armed = par._Armed(
        ranks=(0, 1, 2, 3, 4, 5, 6, 7),
        op=op,
        rank=rank,
        fp8=par.Fp8Config(2048, "a2a", "g128"),
    )
    monkeypatch.setattr(par, "_ARMED", armed)

    def fn(x):
        return par.all_reduce(armed.ranks, x) + 1

    x = torch.zeros(TOKENS, H, dtype=torch.bfloat16).to("meta")
    torch._dynamo.mark_dynamic(x, 0)
    torch._dynamo.reset()
    out = torch.compile(fn, backend="eager", fullgraph=True)(x)
    assert tuple(out.shape) == (TOKENS, H)
    assert out.dtype == torch.bfloat16
