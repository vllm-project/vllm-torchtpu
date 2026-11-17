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
"""The fused gated norm + fp8 quantization against the two-step path: in the
Pallas interpreter on the host (bit for bit), and compiled on a TPU at the
Qwen3.5-397B GDN widths and step sizes (within one rounding step)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen3_5.kernels import gated_norm_quant as gnq
from vllm_torchtpu.models.qwen3_5.kernels.gated_norm_quant import kernel


@pytest.fixture(autouse=True)
def _on_the_host(request):
    # The interpreter tests' reference must round as the interpreter does,
    # even in a session that has brought up the TPU backend. The compiled
    # kernel test runs on the TPU.
    if request.node.get_closest_marker("tpu_test"):
        yield
        return
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _inputs(tokens=64, heads=4, dim=128, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    o = (jax.random.normal(k[0], (tokens, heads, dim)) * 3).astype(jnp.bfloat16)
    z = jax.random.normal(k[1], (tokens, heads, dim)).astype(jnp.bfloat16)
    w = (1.0 + 0.1 * jax.random.normal(k[2], (dim,))).astype(jnp.bfloat16)
    return o, z, w


def _assert_matches_reference(o, z, w, tile):
    q, s = gnq.gated_norm_quant(o, z, w, 1e-6, tile=tile, interpret=True)
    q_ref, s_ref = gnq.reference(o, z, w, 1e-6)
    np.testing.assert_array_equal(np.asarray(s), np.asarray(s_ref))
    np.testing.assert_array_equal(
        np.asarray(q, np.float32), np.asarray(q_ref, np.float32)
    )


@pytest.mark.cpu_test
def test_kernel_matches_the_two_step_path():
    _assert_matches_reference(*_inputs(), tile=16)


@pytest.mark.cpu_test
def test_kernel_2d_z_matches_the_two_step_path():
    o, z, w = _inputs(tokens=128, heads=4, dim=128)
    _assert_matches_reference(o, z.reshape(128, -1), w, tile=64)


# 36, 44 and 52 have no multiple-of-8 divisor and run as one block.
@pytest.mark.cpu_test
@pytest.mark.parametrize("tokens", [4, 12, 20, 36, 40, 44, 48, 52, 80])
def test_row_counts_the_tile_does_not_divide(tokens):
    o, z, w = _inputs(tokens=tokens, heads=2)
    _assert_matches_reference(o, z, w, tile=32)
    _assert_matches_reference(o, z.reshape(tokens, -1), w, tile=32)


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "rows, cap, tile", [(64, 32, 32), (48, 32, 24), (12, 32, 12), (36, 32, 36)]
)
def test_legal_tile(rows, cap, tile):
    assert kernel.legal_tile(rows, cap) == tile


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "tokens, cols, cap",
    [
        # TP8 GDN width: the 512-row prefill cap and the grid >= 6 decode caps.
        (2048, 1024, 512),
        (8192, 1024, 512),
        (96, 1024, 16),
        # TP1 width (a pipeline stage): 16 MiB over 10 bytes per element.
        (2048, 8192, 200),
        (8192, 8192, 200),
        (96, 8192, 16),
    ],
)
def test_default_tile_cap_stays_under_the_vmem_budget(tokens, cols, cap):
    assert kernel.default_tile_cap(tokens, cols) == cap
    assert cap * cols * kernel.TILE_BYTES_PER_ELEMENT <= kernel.TILE_VMEM_BUDGET_BYTES


@pytest.mark.cpu_test
def test_full_width_2d_z_matches_the_two_step_path():
    # 64 heads, the width of an unsharded GDN layer; the 2D-z kernel gathers
    # one strided slice per head.
    o, z, w = _inputs(tokens=32, heads=64, dim=128)
    _assert_matches_reference(o, z.reshape(32, -1), w, tile=16)


@pytest.mark.cpu_test
def test_zero_rows_get_unit_scale():
    o, z, w = _inputs(tokens=16, heads=2)
    o = o.at[3].set(0)
    q, s = gnq.gated_norm_quant(o, z, w, 1e-6, tile=16, interpret=True)
    assert float(s[3, 0]) == 1.0
    assert float(jnp.abs(q[3].astype(jnp.float32)).max()) == 0.0


def _e4m3_steps(got, want):
    """|got - want| in units of the float8_e4m3fn spacing at that magnitude
    (2**(e - 3) for normal values in [2**e, 2**(e + 1)), 2**-9 below 2**-6)."""
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    mag = np.maximum(np.abs(got), np.abs(want))
    spacing = np.where(
        mag >= 2.0**-6,
        2.0 ** (np.floor(np.log2(np.maximum(mag, 2.0**-6))) - 3),
        2.0**-9,
    )
    return np.abs(got - want) / spacing


@pytest.mark.tpu_test
# The 397B GDN widths per device: 8 value heads of 128 at TP8 (decode legs) and
# 64 at TP1 (pipeline-parallel prefill legs). 96 rows is a decode step (16-row
# blocks, grid of 6), 8192 a prefill step (512- and 128-row blocks).
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("tokens", [96, 8192])
@pytest.mark.parametrize("z_2d", [False, True], ids=["3d_z", "2d_z"])
def test_compiled_kernel_matches_the_two_step_path_on_tpu(heads, tokens, z_2d):
    """The Mosaic-compiled kernel at its default tiles: the 3D-z kernel, the
    2D-z kernel with its strided scratch, the 512-row VMEM blocks and the 16 MiB
    budget (the 2D-z kernel serves the 8192-row TP8 step; the other 2D-z cases
    take the 3D kernel on the reshaped gate). The TPU may round one scale a
    bf16 step away from the reference (an f32 reduction order), and the fp8
    values then move by at most one float8_e4m3fn step."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    o, z, w = _inputs(tokens=tokens, heads=heads, dim=128, seed=tokens + heads)
    if z_2d:
        z = z.reshape(tokens, -1)
    q, s = jax.block_until_ready(gnq.gated_norm_quant(o, z, w, 1e-6))
    q_ref, s_ref = jax.block_until_ready(gnq.reference(o, z, w, 1e-6))
    # The fp8 rows take the gate's shape, 3D or 2D.
    assert q.shape == q_ref.shape == z.shape
    assert q.dtype == q_ref.dtype == jnp.float8_e4m3fn
    assert s.shape == s_ref.shape == (tokens, 1)
    np.testing.assert_allclose(
        np.asarray(s, np.float32), np.asarray(s_ref, np.float32), rtol=2.0**-7, atol=0
    )
    assert float(_e4m3_steps(q, q_ref).max()) <= 1.0
