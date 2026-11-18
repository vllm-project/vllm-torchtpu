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
"""The MiniMax M3 SP collectives (`models/minimax_m3/sp.py`) and the hierrs_tc
reduce-scatter op (`distributed/hier_reduce_scatter.py`) as torch_tpu
``jax_op``s: their fakes fit the op schema (derived from the JAX function's
return annotation) with a symbolic token count, under ``FakeTensorMode`` and
under ``torch.compile(fullgraph=True)`` over meta tensors, as Dynamo runs
them in the runner's warmup. The JAX bodies are checked against a reference
on simulated CPU devices (when XLA_FLAGS reaches jax first).
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import (
    DimDynamic,
    ShapeEnv,
    StatelessSymbolicContext,
)

from vllm_torchtpu.models.minimax_m3 import sp

pytest.importorskip("torch_tpu")

H = 6144
TOKENS = 2048  # hint of the symbolic token count


def _mesh(n=None):
    devs = jax.devices("cpu")
    n = len(devs) if n is None else n
    if len(devs) < n:
        pytest.skip(f"needs {n} CPU devices (XLA_FLAGS set too late)")
    return Mesh(np.asarray(devs[:n]), axis_names=("d",))


def _returns(op) -> int:
    return len(op._opoverload._schema.returns)


def _fake_call(op, make_args):
    mode = FakeTensorMode(shape_env=ShapeEnv())
    seed = mode.from_tensor(
        torch.empty(TOKENS),
        symbolic_context=StatelessSymbolicContext(dynamic_sizes=[DimDynamic.DYNAMIC]),
    )
    with mode:
        num = seed.shape[0]
        assert isinstance(num, torch.SymInt)
        args = [
            mode.from_tensor(a) if not mode.is_our_fake(a) else a
            for a in make_args(num, mode)
        ]
        return num, op(*args)


def _compile_call(fn, *args, dynamic=(0,)):
    meta = []
    for i, a in enumerate(args):
        a = a.to("meta")
        if i in dynamic:
            torch._dynamo.mark_dynamic(a, 0)
        meta.append(a)
    torch._dynamo.reset()
    return torch.compile(fn, backend="eager", fullgraph=True)(*meta)


@pytest.fixture(params=[False, True], ids=["no_rank", "rank"])
def ops(request):
    mesh = _mesh()
    rank = torch.zeros(1, 1, dtype=torch.int32) if request.param else None
    return sp.make_ops(mesh, rank)


def test_reduce_scatter_fake_symbolic(ops):
    n = ops.n
    rank = () if ops.rank is None else (ops.rank,)
    assert _returns(ops.reduce_scatter) == 1

    def make(num, mode):
        return (torch.empty((num * n, H), dtype=torch.bfloat16), *rank)

    num, out = _fake_call(ops.reduce_scatter, make)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.bfloat16
    assert out.shape[1] == H
    assert out.shape[0] == num


@pytest.mark.parametrize("fp8", [False, True])
def test_all_gather_fake_symbolic(ops, fp8):
    op = ops.all_gather_fp8 if fp8 else ops.all_gather
    assert _returns(op) == 1
    n = ops.n

    def make(num, mode):
        return (torch.empty((num, H), dtype=torch.bfloat16),)

    num, out = _fake_call(op, make)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.bfloat16  # dequantized on arrival
    assert tuple(out.shape) == (num * n, H)


def test_sp_wrappers_compile_with_a_dynamic_token_dim(monkeypatch, ops):
    """`sp.reduce_scatter` then `sp.all_gather` (viewed with the step's token
    symbol) under Dynamo with the token dim dynamic, fullgraph."""
    monkeypatch.setattr(sp, "_OPS", ops)

    def fn(ids, x):
        shard = sp.reduce_scatter(x)
        a = sp.all_gather(shard, False, ids.shape[0])
        b = sp.all_gather(shard, True, ids.shape[0])
        return a + b

    n = ops.n
    ids = torch.zeros(TOKENS, dtype=torch.long)
    x = torch.zeros(TOKENS, H, dtype=torch.bfloat16)
    out = _compile_call(fn, ids, x, dynamic=(0, 1))
    assert tuple(out.shape) == (TOKENS, H)
    assert n >= 1


def test_hier_reduce_scatter_op_fake_symbolic():
    from vllm_torchtpu.distributed import hier_reduce_scatter as hrs

    mesh = _mesh(8)
    op = hrs._build_op(mesh, tuple(range(8)), "d", chip_perm=(0, 3, 1, 2))
    assert _returns(op) == 1

    def make(num, mode):
        return (torch.empty((num * 8, H), dtype=torch.bfloat16),)

    num, out = _fake_call(op, make)
    assert tuple(out.shape) == (num, H)
    out = _compile_call(
        lambda x: op(x), torch.zeros(TOKENS * 8, H, dtype=torch.bfloat16)
    )
    assert tuple(out.shape) == (TOKENS, H)


# ---------------------------------------------------------------------------
# JAX bodies on simulated devices
# ---------------------------------------------------------------------------


def _run(body, x, mesh):
    f = jax.shard_map(
        body, mesh=mesh, in_specs=P("d"), out_specs=P("d"), check_vma=False
    )
    return np.asarray(jax.jit(f)(x).astype(jnp.float32))


def test_bodies_match_the_reference():
    n = 4
    mesh = _mesh(n)
    rs, ag, ag8 = sp.make_bodies("d")
    t, h = 32, 256
    rng = np.random.default_rng(0)
    # Each rank's [t, h] partial, stacked along rows.
    parts = rng.standard_normal((n, t, h)).astype(np.float32) * 4
    x = jnp.asarray(parts.reshape(n * t, h), dtype=jnp.bfloat16)
    want_rs = parts.astype(np.float32).sum(0)  # rank i keeps rows i*t/n..
    got_rs = _run(rs, x, mesh)
    np.testing.assert_allclose(got_rs, want_rs, rtol=2e-2, atol=5e-2)
    # all_gather: every rank's [t/n... ] shard back in mesh order.
    shards = jnp.asarray(parts[0], dtype=jnp.bfloat16)  # [t, h], t/n rows each
    got_ag = _run(ag, shards, mesh).reshape(n, t, h)
    ref = np.asarray(shards.astype(jnp.float32))
    for r in range(n):
        np.testing.assert_array_equal(got_ag[r], ref)
    got8 = _run(ag8, shards, mesh).reshape(n, t, h)
    amax = np.abs(ref).max(-1, keepdims=True)
    tol = np.abs(ref) * (2**-4 + 2**-8) + amax * 2**-9
    for r in range(n):
        assert np.all(np.abs(got8[r] - ref) <= tol)
    assert not np.array_equal(got8[0], ref)  # it really went through e4m3


def test_fp8_rows_roundtrip_is_exact_on_e4m3_values():
    x = jnp.asarray([[448.0, -2.0, 0.5, 0.0], [1.0, 1.0, -1.0, 0.25]], jnp.bfloat16)
    q, scale = sp.fp8_rows(x)
    assert q.dtype == jnp.float8_e4m3fn and scale.shape == (2, 1)
    back = sp.from_fp8_rows(q, scale, jnp.bfloat16)
    np.testing.assert_array_equal(np.asarray(back), np.asarray(x))
