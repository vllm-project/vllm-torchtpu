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
"""hierrs_tc over a mesh whose chips are not in hypercube order (chip_perm).

Runs both kernels in TPU interpret mode over 8 simulated CPU devices: a
rank-ordered tpu7x mesh (device ids 0, 1, 4, 5, 6, 7, 2, 3) with the chip
permutation must give the same reduce-scatter as the reference, for the
identity and the tpu7x rank order. Interpret mode lets any device reach any
other, so this checks the chunk bookkeeping; `test_neighbours_*` checks that
with the permutation every inter-chip exchange is between ICI neighbours.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.hierrs_tc import dispatch, topology

N = 8
# Physical chips (ICI coordinate order) as device ids, as on one tpu7x host.
CHIPS = [[0, 1], [2, 3], [4, 5], [6, 7]]
# A vLLM rank-ordered mesh on tpu7x (rank r on device id RANK_DEVICES[r]).
RANK_DEVICES = (0, 1, 4, 5, 6, 7, 2, 3)
# 2x2 grid: chip index k = 2 * x + y, so neighbours differ in one bit.
COORDS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def _cpu_devices():
    devs = jax.devices("cpu")
    if len(devs) < N:
        pytest.skip(f"needs {N} CPU devices (XLA_FLAGS set too late)")
    return {int(d.id): d for d in devs[:N]}


def _chip_perm_for(device_ids):
    """What pallas_all_reduce._chip_perm derives for this mesh."""
    from vllm_torchtpu.distributed import pallas_all_reduce as par

    devices = [par._ChipCore(i, (*COORDS[i // 2], 0), i % 2) for i in device_ids]
    return par._chip_perm(devices)


@pytest.fixture
def interpret(monkeypatch):
    real = pl.pallas_call

    def call(*args, **kwargs):
        kwargs.pop("compiler_params", None)
        kwargs["interpret"] = pltpu.InterpretParams()
        return real(*args, **kwargs)

    monkeypatch.setattr(pl, "pallas_call", call)
    # The kernels and Pallas' pipeline read tpu7x's TPU info at trace time.
    from jax._src import tpu_info

    monkeypatch.setattr(tpu_info, "get_device_kind", lambda: "TPU7x")
    tpu_info.get_tpu_info.cache_clear()
    yield
    tpu_info.get_tpu_info.cache_clear()


def test_chip_perm_of_the_rank_ordered_mesh():
    assert _chip_perm_for(range(N)) is None
    # Mesh chips hold physical chips 0, 2, 3, 1; perm[k] = mesh chip of chip k.
    assert _chip_perm_for(RANK_DEVICES) == (0, 3, 1, 2)


def test_logical_order():
    assert topology.logical_order(None, None, N) is None
    order = (3, 1, 4, 0, 5, 2, 7, 6)
    assert topology.logical_order(order, None, N) == order
    perm = (0, 3, 1, 2)
    got = topology.logical_order(order, perm, N)
    # Logical device d sits at mesh index perm[d // 2] * 2 + d % 2.
    assert got == tuple(order[perm[d // 2] * 2 + d % 2] for d in range(N))
    with pytest.raises(ValueError, match="permutation"):
        topology.check_chip_perm((0, 1, 1, 2), 4)


@pytest.mark.parametrize("perm", [None, (0, 3, 1, 2), (2, 0, 3, 1)])
def test_neighbours_are_ici_neighbours(perm):
    """Every hypercube exchange of logical chip k (k ^ 1, k ^ 2) lands on a
    mesh chip whose physical chip is one grid step away."""
    mesh_ids = RANK_DEVICES if perm == (0, 3, 1, 2) else tuple(range(N))
    if perm == (2, 0, 3, 1):
        # Some other chip order: mesh chip m holds physical chip phys[m].
        phys = [perm.index(m) for m in range(4)]
        mesh_ids = tuple(i for m in range(4) for i in CHIPS[phys[m]])
    derived = _chip_perm_for(mesh_ids)
    assert derived == perm
    p = perm or (0, 1, 2, 3)
    mesh_chip_phys = {m: CHIPS.index(sorted(mesh_ids[2 * m : 2 * m + 2])) for m in range(4)}
    for k in range(4):
        me = mesh_chip_phys[p[k]]
        assert me == k
        for dim in range(2):
            other = mesh_chip_phys[p[k ^ (1 << dim)]]
            (x0, y0), (x1, y1) = COORDS[me], COORDS[other]
            assert abs(x0 - x1) + abs(y0 - y1) == 1


def _run(mesh_ids, order, perm, rows_per_device, hidden, seed):
    devs = _cpu_devices()
    mesh = Mesh(np.asarray([devs[i] for i in mesh_ids]), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, P("x", None))
    fn = jax.jit(
        dispatch.make_reduce_scatter_fn(mesh, order, axis_name="x", chip_perm=perm)
    )
    rng = np.random.default_rng(seed)
    block = rows_per_device // N
    # per_rank[r]: rank r's input; mesh index i holds rank order[i]'s.
    per_rank = [
        np.asarray(
            jnp.asarray(rng.standard_normal((rows_per_device, hidden)), jnp.bfloat16),
            np.float32,
        )
        for _ in range(N)
    ]
    x = np.concatenate([per_rank[r] for r in order], axis=0)
    out = np.asarray(fn(jax.device_put(jnp.asarray(x, jnp.bfloat16), sharding)), np.float32)
    summed = sum(per_rank)
    want = np.concatenate([summed[r * block : (r + 1) * block] for r in order], axis=0)
    return out, want


@pytest.mark.parametrize(
    "rows_per_device, hidden, stream",
    [
        pytest.param(64, 512, False, id="micro_batched"),
        pytest.param(1024, 512, True, id="row_streaming"),
    ],
)
@pytest.mark.parametrize(
    "mesh_ids, order",
    [
        pytest.param(tuple(range(N)), tuple(range(N)), id="sorted_mesh"),
        pytest.param(RANK_DEVICES, tuple(range(N)), id="rank_ordered_mesh"),
        pytest.param(RANK_DEVICES, (7, 6, 5, 4, 3, 2, 1, 0), id="rank_mesh_reversed"),
    ],
)
def test_chip_perm_matches_reference(
    interpret, monkeypatch, rows_per_device, hidden, stream, mesh_ids, order
):
    if stream:
        monkeypatch.setattr(dispatch, "STREAM_MIN_BYTES", 0)
        monkeypatch.setattr(
            dispatch, "supports_stream", lambda rows, hidden, itemsize: True
        )
    else:
        monkeypatch.setattr(dispatch, "STREAM_MIN_BYTES", 1 << 40)
    perm = _chip_perm_for(mesh_ids)
    out, want = _run(mesh_ids, order, perm, rows_per_device, hidden, seed=sum(order))
    err = np.abs(out - want).max() / max(np.abs(want).max(), 1e-6)
    # bf16 sums in another order than the f32 reference: a few bf16 ulps.
    assert err < 2e-2, err
