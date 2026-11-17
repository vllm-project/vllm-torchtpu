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
"""The fused attention output sigmoid gate + per-token fp8 quantization
against the eager gate and quantization in the Pallas interpreter on the
host, and its tiling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.models.qwen3_5.kernels.sigmoid_gate_quant import (
    kernel,
    pallas_sigmoid_gate_quant,
    sigmoid_gate_quant_reference,
)


@pytest.fixture(autouse=True)
def _on_the_host(request):
    # The interpreter tests run on the host even in a session that has
    # brought up the TPU backend; the compiled kernel test runs on the TPU.
    if request.node.get_closest_marker("tpu_test"):
        yield
        return
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.cpu_test
def test_pal_gate_q_interpreter_parity():
    """Verify pallas_sigmoid_gate_quant in Pallas CPU interpreter."""
    rng = np.random.default_rng(42)
    for m in (4, 8, 16, 64):
        a = jnp.asarray(rng.standard_normal((m, 256)), dtype=jnp.bfloat16)
        g = jnp.asarray(rng.standard_normal((m, 256)), dtype=jnp.bfloat16)
        q_pal, s_pal = pallas_sigmoid_gate_quant(a, g, interpret=True)
        q_ref, s_ref = sigmoid_gate_quant_reference(a, g)
        assert q_pal.shape == (m, 256) and s_pal.shape == (m, 1)
        np.testing.assert_allclose(
            np.asarray(s_pal, np.float32),
            np.asarray(s_ref, np.float32),
            rtol=1e-2,
            atol=1e-3,
        )
        deq_pal = np.asarray(q_pal, np.float32) * np.asarray(s_pal, np.float32)
        deq_ref = np.asarray(q_ref, np.float32) * np.asarray(s_ref, np.float32)
        np.testing.assert_allclose(deq_pal, deq_ref, rtol=0.15, atol=0.15)
        assert np.mean(np.abs(deq_pal - deq_ref)) < 5e-3


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("rows", "cols", "expected_bm", "expected_step"),
    [
        (16, 1024, 16, 16),  # < 32: full shape
        (32, 1024, 32, 32),  # 32-aligned decode bucket (TP=8)
        (64, 1024, 32, 32),  # 32-aligned sub-block for float8_e4m3fn (32, 128) tiles
        (512, 1024, 512, 64),  # TP=8 prefill: 512-row block, 64-row in-kernel steps
        # TP=2 prefill: the 16-row VREG step is not 32-aligned, so each block
        # runs as one step; the 16 MiB tile budget holds a 4096-wide block to
        # 384 rows, and 256 is the largest 32-aligned divisor of 4096 under it.
        (4096, 4096, 256, 256),
        # TP=1 prefill (a pipeline stage): the budget holds an 8192-wide block
        # to 192 rows, so a 2176-row chunk and an 8192-row step tile at 128.
        (2176, 8192, 128, 128),
        (8192, 8192, 128, 128),
    ],
)
def test_sigmoid_gate_quant_tile_selection(rows, cols, expected_bm, expected_step):
    """The BlockSpec tile `bm` and the in-kernel row step that
    `pallas_sigmoid_gate_quant` derives from it."""
    bm = kernel.legal_tile(rows, kernel.select_tile(rows, cols))
    step = kernel.legal_tile(bm, kernel.select_vreg_step(cols))
    assert (bm, step) == (expected_bm, expected_step)
    assert bm == rows or (rows % bm == 0 and bm % 32 == 0)
    assert step == bm or (bm % step == 0 and step % 32 == 0)
    # Two bf16 inputs and one fp8 output, double buffered, within 16 MiB.
    assert bm * cols * 10 <= (16 << 20)

