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
"""Fused decode MLP kernel vs the torch path's math (`reference`).

``interpret=True`` runs the kernel body on the host; it does not gate Mosaic
lowering.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh

from vllm_torchtpu.kernels import dspark_mlp as kernel


@pytest.fixture(autouse=True)
def _tpu7x():
    """get_tpu_info() on the host answers for the abstract device."""
    mesh = AbstractMesh((), (), (), abstract_device=AbstractDevice("TPU7x", 1, "tpu"))
    with use_abstract_mesh(mesh):
        yield


def _operands(m: int, h: int, i: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    fp8 = jnp.float8_e4m3fn
    x = jnp.asarray(rng.standard_normal((m, h)), jnp.bfloat16)
    w_gu = jnp.asarray(rng.standard_normal((h, 2 * i)) * 4, fp8)
    w_d = jnp.asarray(rng.standard_normal((i, h)) * 4, fp8)
    # Per-channel powers of two, as requantize_mxfp8_per_channel makes them.
    s_gu = jnp.asarray(2.0 ** rng.integers(-9, -6, (1, 2 * i)), jnp.float32)
    s_d = jnp.asarray(2.0 ** rng.integers(-9, -6, (1, h)), jnp.float32)
    return x, w_gu, s_gu, w_d, s_d


def _err(got, want) -> float:
    """Max abs difference over the reference's max abs value."""
    got, want = np.asarray(got, np.float32), np.asarray(want, np.float32)
    return float(np.abs(got - want).max() / np.abs(want).max())


# Rows padded to a sublane tile (7, 21), a half bf16 tile, a tile, the cap; a
# multi-chunk K (h = 2048 rows at 2i = 2048: 3 gate_up and 2 down chunks).
@pytest.mark.parametrize("m", [7, 8, 16, 21, 32])
@pytest.mark.parametrize("h,i", [(256, 128), (2048, 1024)])
def test_matches_reference(m: int, h: int, i: int) -> None:
    ops = _operands(m, h, i, seed=m + h)
    got = kernel.fused_mlp(*ops, interpret=True)
    want = kernel.reference(*ops)
    assert got.shape == (m, h) and got.dtype == jnp.bfloat16
    # Summation order only: a few bf16 ulps of the output's scale.
    assert _err(got, want) < 2e-2


def test_faults_are_caught() -> None:
    """The bar above rejects a dropped K chunk and swapped gate / up."""
    m, h, i = 16, 2048, 1024
    x, w_gu, s_gu, w_d, s_d = _operands(m, h, i, seed=3)
    got = kernel.fused_mlp(x, w_gu, s_gu, w_d, s_d, interpret=True)
    dropped = w_gu.at[-128:].set(0)
    swapped = jnp.concatenate([w_gu[:, i:], w_gu[:, :i]], 1)
    for w in (dropped, swapped):
        assert _err(got, kernel.reference(x, w, s_gu, w_d, s_d)) > 2e-2


def test_supported_rows() -> None:
    assert kernel.supported(32, 6144, 1536)
    assert not kernel.supported(40, 6144, 1536)
    assert kernel.supported(7, 6144, 1536)
    assert not kernel.supported(32, 6144, 1000)
