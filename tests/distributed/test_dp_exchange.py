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

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_torchtpu.distributed import dp_exchange

# Runs on 8 host devices, in its own process so the device count can be set.
_WORKER = """
import jax, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from vllm_torchtpu.distributed.dp_exchange import exchange_fn

mesh = Mesh(np.array(jax.devices()), ("d",))
# Pairs in mesh coordinates, DP rank 0 first, not in mesh order.
groups = ((0, 6), (5, 1), (2, 4), (7, 3))
rows, cols = 16, 128
for dtype in (np.int16, np.int32):
    x = np.random.default_rng(0).integers(-30000, 30000, (8 * rows, cols)).astype(dtype)
    x = jax.device_put(x, NamedSharding(mesh, PartitionSpec("d")))
    blocks = np.asarray(x).reshape(8, rows, cols)
    # Each DP rank's op is only run for its own devices: check those.
    for dp_rank in (0, 1):
        out = jax.jit(exchange_fn(mesh, groups, dp_rank))(x)
        out = np.asarray(out).reshape(8, 2 * rows, cols)
        for group in groups:
            want = np.concatenate([blocks[group[0]], blocks[group[1]]])
            assert np.array_equal(out[group[dp_rank]], want)
print("OK")
"""


def test_exchange_orders_pair_rows_like_dp_all_gather():
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        XLA_FLAGS=f"{os.environ.get('XLA_FLAGS', '')} "
        "--xla_force_host_platform_device_count=8",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _WORKER],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-3000:]


@pytest.mark.parametrize(
    "tensor",
    [
        torch.randint(-(2**15), 2**15, (4, 8), dtype=torch.int16).view(torch.bfloat16),
        torch.randint(-(2**31), 2**31, (4, 8), dtype=torch.int64)
        .to(torch.int32)
        .view(torch.float32),
    ],
)
def test_all_gather_moves_rows_as_integers(monkeypatch, tensor):
    seen = []

    def op(bits):
        seen.append(bits.dtype)
        return torch.cat([bits, bits.flip(0)])

    monkeypatch.setattr(dp_exchange, "_OP", op)
    out = dp_exchange.dp_pair_all_gather(tensor)
    assert seen == [{2: torch.int16, 4: torch.int32}[tensor.element_size()]]
    assert out.dtype == tensor.dtype
    bits = {2: torch.int16, 4: torch.int32}[tensor.element_size()]
    assert torch.equal(out.view(bits), torch.cat([tensor, tensor.flip(0)]).view(bits))


@pytest.mark.parametrize(
    "config",
    [
        {"enable_expert_parallel": False},
        {"data_parallel_size": 4},
        {"pipeline_parallel_size": 2},
    ],
)
def test_prepare_builds_nothing_outside_an_ep_dp_pair(monkeypatch, config):
    parallel_config = SimpleNamespace(
        **{
            "enable_expert_parallel": True,
            "data_parallel_size": 2,
            "pipeline_parallel_size": 1,
            **config,
        }
    )
    monkeypatch.setattr(dp_exchange, "_OP", None)
    monkeypatch.setattr(dp_exchange, "get_ep_group", lambda: object())
    build = Mock()
    monkeypatch.setattr(dp_exchange, "build_ep_mesh", build)
    dp_exchange.prepare_dp_pair_exchange(parallel_config)
    build.assert_not_called()
    assert not dp_exchange.dp_pair_exchange_ready()


def test_each_dp_rank_registers_its_own_op(monkeypatch):
    # Peer replay shares compiled graphs whose code matches, and each rank's
    # op bakes its row order in, so the op names must differ.
    parallel_config = SimpleNamespace(
        enable_expert_parallel=True, data_parallel_size=2, pipeline_parallel_size=1
    )
    monkeypatch.setattr(dp_exchange, "get_ep_group", lambda: object())
    monkeypatch.setattr(dp_exchange, "build_ep_mesh", lambda: SimpleNamespace(size=4))
    monkeypatch.setattr(dp_exchange, "ep_dp_groups", lambda: ((0, 2), (1, 3)))
    jax_op = Mock()
    monkeypatch.setattr(dp_exchange, "jax_op", jax_op)
    for dp_rank in (0, 1):
        monkeypatch.setattr(dp_exchange, "_OP", None)
        group = SimpleNamespace(rank_in_group=dp_rank)
        monkeypatch.setattr(dp_exchange, "get_dp_group", lambda g=group: g)
        dp_exchange.prepare_dp_pair_exchange(parallel_config)
        assert dp_exchange.dp_pair_exchange_ready()
    names = [c.args[0] for c in jax_op.call_args_list]
    assert len(names) == 2 and names[0] != names[1]
