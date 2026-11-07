# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3's SP slice hands each device its rows without baking the rank in.

`TokenShardCollectives.local_rows` must be the inverse of the token
all-gather on the same axis_index_groups, for both the TP32 group and the TP2
chip pairs, and its program must not contain `partition-id` (torch_tpu's
jax.export bridge rejects it, and nothing short of a TPU compile would
otherwise catch it). The JAX functions run on a 32-device CPU mesh whose
device order is scrambled against the group order.

The JAX cases need 32 CPU devices; without them (as in CI, which runs JAX on
one CPU device) a child process with
XLA_FLAGS=--xla_force_host_platform_device_count=32 runs them instead.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

jax = pytest.importorskip("jax")

N, T, D = 32, 256, 8
# Group position of mesh index i; a fixed scramble so mesh order != group order.
OWNERS = [(7 * i + 3) % N for i in range(N)]


def _groups(world_size):
    if world_size == N:
        return (tuple(sorted(range(N), key=lambda i: OWNERS[i])),)
    return tuple(
        tuple(sorted(pair, key=lambda i: OWNERS[i]))
        for pair in zip(range(0, N, 2), range(1, N, 2), strict=True)
    )


@pytest.fixture
def k3_ops(monkeypatch):
    if jax.device_count() < N:
        pytest.skip(
            f"needs {N} devices (XLA_FLAGS=--xla_force_host_platform_device_count={N})"
        )
    from jax.sharding import Mesh

    from vllm_torchtpu.models.vllm.kimi_k3 import collective_ops as ops

    mesh = Mesh(np.array(jax.devices()[:N]), ("k3_sp",))
    monkeypatch.setattr(ops, "_token_groups", lambda w: (mesh, _groups(w)))
    captured = {}

    def fake_sharded_jax_op(name, fn, **kw):
        captured[name] = fn
        return SimpleNamespace(register_fake=lambda f: None)

    monkeypatch.setattr(ops, "sharded_jax_op", fake_sharded_jax_op)
    ops._build_local_rows_op.cache_clear()
    ops._build_token_op.cache_clear()

    def build(world_size):
        ops._build_local_rows_op(world_size)
        ops._build_token_op(world_size, True)
        return (
            captured[f"pallas::kimi_token_{world_size}_local"],
            captured[f"pallas::kimi_token_{world_size}_ag"],
        )

    yield mesh, build
    ops._build_local_rows_op.cache_clear()
    ops._build_token_op.cache_clear()


@pytest.mark.parametrize("world_size", [2, N])
def test_local_rows_is_the_inverse_of_the_token_all_gather(k3_ops, world_size):
    mesh, build = k3_ops
    local_rows, all_gather = build(world_size)
    x = np.arange(T * D, dtype=np.float32).reshape(T, D)
    shards = jax.jit(local_rows)(x)
    position = {i: g.index(i) for g in _groups(world_size) for i in g}
    for shard in shards.addressable_shards:
        i = mesh.devices.tolist().index(shard.device)
        want = x.reshape(world_size, -1, D)[position[i]]
        np.testing.assert_array_equal(np.asarray(shard.data), want, f"mesh index {i}")
    gathered = jax.jit(all_gather)(shards)
    for shard in gathered.addressable_shards:
        np.testing.assert_array_equal(np.asarray(shard.data), x)


def test_local_rows_has_no_partition_id(k3_ops):
    _, build = k3_ops
    local_rows, _ = build(N)
    lowered = jax.jit(local_rows).lower(jax.ShapeDtypeStruct((T, D), np.float32))
    assert "partition_id" not in lowered.as_text()
    assert "partition-id" not in lowered.compile().as_text()


def _trace_sp_forward(rank):
    from vllm_torchtpu.models.vllm.kimi_k3.model import KimiModel

    model = KimiModel.__new__(KimiModel)
    nn.Module.__init__(model)
    model.sp_prefill = True
    model.sp_group = SimpleNamespace(
        rank_in_group=rank,
        world_size=N,
        local_rows=lambda x: x[: x.shape[0] // N] * 1,
        all_gather=lambda x, dim: x.repeat(N, 1),
    )
    model.aux_hidden_state_layers = ()
    model.attn_res_block_size = None
    model.layers = nn.ModuleList()
    model.norm = nn.Identity()
    graphs = []

    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    fn = torch.compile(model.forward, backend=capture, fullgraph=True, dynamic=False)
    hidden = torch.zeros(512, 16)
    fn(None, torch.arange(512), inputs_embeds=hidden)
    return graphs[0].code


def test_sp_graph_does_not_depend_on_the_rank():
    # Every rank must trace the same SP graph, or under TPU_PARALLEL_PRECOMPILE
    # each one compiles every SP bucket itself instead of loading the program
    # another rank put in the shared compile cache. Indexing with the Python
    # rank_in_group makes the rank a constant of the graph.
    assert _trace_sp_forward(3) == _trace_sp_forward(5)


def test_jax_cases_run_with_32_host_devices():
    flag = f"--xla_force_host_platform_device_count={N}"
    if flag in os.environ.get("XLA_FLAGS", ""):
        pytest.skip("this process already has the devices; the cases ran above")
    env = {**os.environ, "XLA_FLAGS": flag, "JAX_PLATFORMS": "cpu"}
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", __file__]
        + ["-k", "inverse or partition_id"],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = result.stdout[-4000:] + result.stderr[-2000:]
    assert result.returncode == 0, out
    assert "3 passed" in result.stdout, out
