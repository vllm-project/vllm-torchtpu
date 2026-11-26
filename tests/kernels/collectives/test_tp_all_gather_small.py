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
"""TPU_TP_PALLAS_SMALL_ALL_GATHER: the Pallas small all-gather (interpret
mode over 8 CPU devices) and its dispatch."""

import functools
import os

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")

from jax.experimental import pallas as pl  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402
from jax.sharding import Mesh  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from vllm_torchtpu.kernels.collectives import tp_all_reduce as kernel  # noqa: E402

N = 8


@pytest.fixture
def interpret(monkeypatch):
    if len(jax.devices()) < N:
        pytest.skip("needs XLA_FLAGS=--xla_force_host_platform_device_count=8")
    monkeypatch.setattr(
        kernel.pl,
        "pallas_call",
        functools.partial(pl.pallas_call, interpret=pltpu.InterpretParams()),
    )
    kernel._gather_call.cache_clear()
    yield
    kernel._gather_call.cache_clear()


@pytest.mark.parametrize("shape", [(16, 1), (5, 3), (64, 2)])
def test_all_gather_small_is_exact_in_rank_order(interpret, shape):
    mesh = Mesh(np.array(jax.devices()[:N]), ("x",))
    xs = np.arange(N * np.prod(shape), dtype=np.float32).reshape(N * shape[0], *shape[1:])
    ranks = np.arange(N, dtype=np.int32).reshape(N, 1)
    f = jax.shard_map(
        lambda x, r: kernel.all_gather_small(x, r, N),
        mesh=mesh,
        in_specs=(P("x"), P("x")),
        out_specs=P("x"),
        check_vma=False,
    )
    out = np.asarray(jax.jit(f)(xs, ranks)).reshape(N, N * shape[0], *shape[1:])
    for i in range(N):
        np.testing.assert_array_equal(out[i], xs)


def test_gather_view():
    assert kernel.gather_view(16, 4) == (8, 128)
    assert kernel.gather_view(1025, 4) == (16, 128)
    assert kernel.gather_view(16, 2) is None


class _FakeGather:
    """Stands in for the armed op: concatenates N copies (rank-order blocks
    tagged by rank) the way the kernel lays them out."""

    def __init__(self):
        self.calls = []

    def __call__(self, x, rank):
        self.calls.append(tuple(x.shape))
        return torch.cat([x + 0 * r for r in range(N)])


@pytest.fixture
def armed(monkeypatch):
    par = pytest.importorskip("vllm_torchtpu.distributed.pallas_all_reduce")
    op = _FakeGather()
    rec = par._Armed(
        ranks=tuple(range(N)),
        op=None,
        rank=torch.zeros(1, 1, dtype=torch.int32),
        rank_ordered=True,
        gather_op=op,
    )
    monkeypatch.setattr(par, "_ARMED", rec)
    return par, op, rec


def test_all_gather_small_dispatch(armed, monkeypatch):
    par, op, rec = armed
    ranks = tuple(range(N))
    x = torch.randn(16, 1)
    assert par.all_gather_small(ranks, x).shape == (N * 16, 1)
    # Other groups, 16-bit tensors, too-large tensors, no op: caller's path.
    assert par.all_gather_small((0, 1), x) is None
    assert par.all_gather_small(ranks, x.bfloat16()) is None
    assert par.all_gather_small(ranks, torch.randn(70000, 1)) is None
    monkeypatch.setattr(par, "_ARMED", par._Armed(**{**rec.__dict__, "gather_op": None}))
    assert par.all_gather_small(ranks, x) is None
    monkeypatch.setattr(
        par, "_ARMED", par._Armed(**{**rec.__dict__, "rank_ordered": False})
    )
    assert par.all_gather_small(ranks, x) is None
    assert op.calls == [(16, 1)]


def test_tp_all_gather_matches_concat(armed, monkeypatch):
    """bf16 pairs travel as int32 words; any dim comes back concatenated in
    rank order, as vLLM's all-gather returns it."""
    par, op, _ = armed
    import vllm.distributed as vd

    monkeypatch.setattr(vd, "get_tp_group", lambda: type("G", (), {"ranks": list(range(N))})())
    x = torch.randn(3, 8).bfloat16()
    for dim in (0, -1):
        got = par.tp_all_gather(x, dim=dim)
        want = torch.cat([x] * N, dim=dim)
        assert torch.equal(got, want)
    assert op.calls == [(3, 4), (3, 4)]


def test_flag_default_off(monkeypatch):
    from vllm_torchtpu import envs

    monkeypatch.delenv("TPU_TP_PALLAS_SMALL_ALL_GATHER", raising=False)
    assert envs.TPU_TP_PALLAS_SMALL_ALL_GATHER is False
