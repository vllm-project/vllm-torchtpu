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
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.fused_moe.v2 import direct_all_gather as gather
from vllm_torchtpu.kernels.fused_moe.v2 import layer

DECODE_ROW_BYTES = 4096
AXIS = "ep"
# DeepSeek-V4.1-Flash routing rows: a decode step and a 2,048-token step.
ROW_WIDTHS = [640, 14720]


@pytest.fixture
def tpu7x(monkeypatch):
    monkeypatch.setattr(gather.jax, "default_backend", lambda: "tpu")
    monkeypatch.setattr(
        gather.pltpu, "get_tpu_info", lambda: SimpleNamespace(generation=7)
    )


def _supports(ep=8, local_bytes=DECODE_ROW_BYTES, single_host=True):
    return gather.supports_direct_all_gather(ep, local_bytes, single_host)


def test_serves_the_decode_exchange_on_8_shards(tpu7x):
    assert _supports()


def test_serves_the_largest_measured_routing_row(tpu7x):
    width = layer._exchange_layout(2048, 6, 384, True)[2]
    assert _supports(local_bytes=width * 4)


def test_needs_a_peer(tpu7x):
    assert not _supports(ep=1)


def test_stays_within_the_measured_shard_count(tpu7x):
    assert _supports(ep=gather.MAX_SHARDS)
    assert not _supports(ep=16)


def test_needs_every_shard_on_one_host(tpu7x):
    assert not _supports(single_host=False)


def test_stays_within_the_measured_row_size(tpu7x):
    assert _supports(local_bytes=gather.MAX_LOCAL_BYTES)
    assert not _supports(
        local_bytes=gather.MAX_LOCAL_BYTES + gather.LOCAL_BYTES_ALIGNMENT
    )


def test_needs_whole_128_lane_int32_rows(tpu7x):
    assert not _supports(local_bytes=DECODE_ROW_BYTES + 4)


def test_every_admitted_exchange_fits_the_shared_memory():
    assert gather.MAX_SHARDS * gather.MAX_LOCAL_BYTES <= gather.SHARED_MEMORY_BYTES


@pytest.mark.parametrize("generation", [6, 8])
def test_needs_a_tpu7x_sparse_core(tpu7x, monkeypatch, generation):
    monkeypatch.setattr(
        gather.pltpu, "get_tpu_info", lambda: SimpleNamespace(generation=generation)
    )
    assert not _supports()


def test_off_tpu_the_exchange_stays_on_xla(monkeypatch):
    monkeypatch.setattr(gather.jax, "default_backend", lambda: "cpu")
    assert not _supports()


def _mesh_or_skip(ep):
    if jax.default_backend() != "tpu":
        pytest.skip("the direct all-gather is a SparseCore kernel; needs a TPU")
    if gather.pltpu.get_tpu_info().generation != 7:
        pytest.skip("the direct all-gather is built for TPU7x")
    if jax.device_count() < ep:
        pytest.skip(f"needs {ep} devices")
    return Mesh(np.asarray(jax.devices()[:ep]), axis_names=(AXIS,))


def _gathered(mesh, x, calls):
    """Each call's direct and lax.all_gather results, as every shard sees
    them, from `calls` back-to-back gathers of distinct payloads."""
    ep = mesh.shape[AXIS]

    def body(local):
        offsets = [i * x.size for i in range(calls)]
        direct = tuple(gather.direct_all_gather(local + o, AXIS, ep) for o in offsets)
        xla = tuple(
            lax.all_gather(local + o, AXIS, axis=0, tiled=True) for o in offsets
        )
        return direct, xla

    return jax.jit(
        jax.shard_map(
            body,
            mesh=mesh,
            in_specs=P(AXIS),
            out_specs=P(AXIS),
            check_vma=False,
        )
    )(x)


def _check(mesh, local_shape, calls):
    ep = mesh.shape[AXIS]
    rows = (ep * local_shape[0], *local_shape[1:])
    x = jnp.arange(np.prod(rows), dtype=jnp.int32).reshape(rows)
    direct, xla = _gathered(mesh, x, calls)
    for i, (got, want) in enumerate(zip(direct, xla)):
        expected = np.concatenate([np.asarray(x) + i * x.size] * ep)
        np.testing.assert_array_equal(np.asarray(want), expected)
        np.testing.assert_array_equal(np.asarray(got), expected, err_msg=f"call {i}")


@pytest.mark.tpu_test
@pytest.mark.parametrize("width", ROW_WIDTHS)
@pytest.mark.parametrize("stage", ["fused", "compute"])
def test_matches_lax_all_gather_on_the_two_cores_of_one_chip(width, stage):
    """The fused path gathers a 1-D row, the compute stage a [1, width] one."""
    local_shape = (width,) if stage == "fused" else (1, width)
    _check(_mesh_or_skip(2), local_shape, calls=1)


@pytest.mark.multichip
@pytest.mark.parametrize("width", ROW_WIDTHS)
def test_matches_lax_all_gather_on_8_shards_call_after_call(width):
    """Several gathers back to back, so a shard can enter the next call while
    a peer is still draining the last one."""
    _check(_mesh_or_skip(8), (width,), calls=4)
