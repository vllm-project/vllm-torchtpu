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
"""Hierarchical TP4 all-reduce kernel against its fixed summation order."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from vllm_torchtpu.kernels.collectives.tp4_all_reduce import supports, tp4_all_reduce


def _bf16(a):
    return np.asarray(
        jnp.asarray(a, jnp.float32).astype(jnp.bfloat16).astype(jnp.float32)
    )


def _reference(x):
    """x [dp, 4, R, H] (bf16 values) -> bf16(bf16(P0+P1) + bf16(P2+P3)) per dp."""
    return _bf16(_bf16(x[:, 0] + x[:, 1]) + _bf16(x[:, 2] + x[:, 3]))


def _run(x, num_blocks, max_rows, interpret):
    devices = sorted(jax.devices(), key=lambda d: d.id)[:8]
    mesh = Mesh(np.array(devices).reshape(2, 4), ("dp", "tp"))
    xs = jax.device_put(
        jnp.asarray(x, jnp.bfloat16), NamedSharding(mesh, P("dp", "tp"))
    )
    out = jax.jit(
        lambda a: tp4_all_reduce(
            a,
            mesh=mesh,
            dp_axis="dp",
            tp_axis="tp",
            num_blocks=num_blocks,
            max_rows=max_rows,
            interpret=interpret,
        )
    )(xs)
    return np.asarray(jax.device_get(out)).astype(np.float32)


@pytest.mark.cpu_test
def test_supports():
    assert supports(92, 8192, 4, 256)
    assert not supports(300, 8192, 4, 256)
    assert not supports(92, 8192 + 128, 4, 256)
    assert supports(23, 2048, 4, 256) and not supports(23, 2048, 8, 256)


@pytest.mark.parametrize(
    "rows,hidden,num_blocks", [(16, 2048, 1), (23, 2048, 2), (92, 2048, 4)]
)
def test_kernel_matches_fixed_order(rows, hidden, num_blocks):
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices")
    interpret = jax.devices()[0].platform == "cpu"
    rng = np.random.default_rng(rows + num_blocks)
    x = _bf16(rng.standard_normal((2, 4, rows, hidden)).astype(np.float32))
    out = _run(x, num_blocks, 256, interpret)
    ref = _reference(x)
    assert np.array_equal(out, np.broadcast_to(ref[:, None], out.shape))


def test_fallback_above_max_rows_is_xla_sum():
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices")
    interpret = jax.devices()[0].platform == "cpu"
    rng = np.random.default_rng(3)
    x = _bf16(rng.standard_normal((2, 4, 32, 2048)).astype(np.float32))
    out = _run(x, 4, 16, interpret)
    # every TP rank holds the same sum, within bf16 rounding of the exact one
    assert np.array_equal(out, np.broadcast_to(out[:, :1], out.shape))
    np.testing.assert_allclose(out[:, 0], x.sum(axis=1), rtol=0, atol=0.1)
