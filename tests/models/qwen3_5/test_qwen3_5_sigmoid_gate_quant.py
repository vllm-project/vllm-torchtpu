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
against the eager gate and quantization: in the Pallas interpreter on the
host, its tiling, and compiled on a TPU at the Qwen3.5-397B ``o_proj`` input
widths and step sizes."""

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
# 4 rows pad to 8; 208 rows at 8192 columns pad to 224 (no legal tile of 208
# rows fits the VMEM budget).
@pytest.mark.parametrize(
    ("m", "cols"), [(4, 256), (8, 256), (16, 256), (64, 256), (208, 8192)]
)
def test_pal_gate_q_interpreter_parity(m, cols):
    """Verify pallas_sigmoid_gate_quant in Pallas CPU interpreter."""
    rng = np.random.default_rng(42)
    a = jnp.asarray(rng.standard_normal((m, cols)), dtype=jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((m, cols)), dtype=jnp.bfloat16)
    q_pal, s_pal = pallas_sigmoid_gate_quant(a, g, interpret=True)
    q_ref, s_ref = sigmoid_gate_quant_reference(a, g)
    assert q_pal.shape == (m, cols) and s_pal.shape == (m, 1)
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


# Two bf16 inputs and one fp8 output, double buffered, per element of a block.
_BLOCK_BYTES = 10
_VMEM_BUDGET = 16 << 20


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    ("rows", "cols", "expected"),
    [
        (4, 1024, (8, 8, 8)),  # fewer than 8 rows pad to 8
        (16, 1024, (16, 16, 16)),  # < 32: full shape
        (32, 1024, (32, 32, 32)),  # 32-aligned decode bucket (TP=8)
        # 32-aligned sub-block for float8_e4m3fn (32, 128) tiles
        (64, 1024, (64, 32, 32)),
        # TP=8 prefill: 512-row block, 64-row in-kernel steps
        (512, 1024, (512, 512, 64)),
        # Wider than 2048 columns a step of up to 256 rows runs as one block
        # where that block fits the budget: TP=2 decode (4096 columns) and
        # TP=1 steps (8192 columns) of up to 204 rows...
        (96, 4096, (96, 96, 96)),
        (256, 4096, (256, 256, 256)),
        (204, 8192, (204, 204, 204)),
        # ...and past it tiles at the largest legal tile within the budget.
        (256, 8192, (256, 128, 128)),
        # No legal tile of 208 or 1000 rows fits the budget at 8192 columns,
        # so the step pads to the next multiple of 32.
        (208, 8192, (224, 32, 32)),
        (1000, 8192, (1024, 128, 128)),
        # TP=2 prefill: the 16-row VREG step is not 32-aligned, so each block
        # runs as one step; the 16 MiB tile budget holds a 4096-wide block to
        # 384 rows, and 256 is the largest 32-aligned divisor of 4096 under it.
        (4096, 4096, (4096, 256, 256)),
        # TP=1 prefill (a pipeline stage): the budget holds an 8192-wide block
        # to 192 rows, so a 2176-row chunk and an 8192-row step tile at 128.
        (2176, 8192, (2176, 128, 128)),
        (8192, 8192, (8192, 128, 128)),
    ],
)
def test_sigmoid_gate_quant_tile_selection(rows, cols, expected):
    """The padded rows, BlockSpec tile and in-kernel row step that
    `pallas_sigmoid_gate_quant` runs a step with."""
    plan = kernel.plan_blocks(rows, cols)
    assert tuple(plan) == expected
    # The cap is itself a legal tile: a multiple of 32, or the whole step.
    cap = kernel.select_tile(plan.rows, cols)
    assert cap == plan.rows or cap % 32 == 0


def _legal(tile: int, rows: int) -> bool:
    return tile == rows or (rows % tile == 0 and tile % 32 == 0)


@pytest.mark.cpu_test
@pytest.mark.parametrize("cols", [1024, 2048, 4096, 8192])
def test_every_block_fits_the_vmem_budget(cols):
    """At every step the Qwen3.5 model patch sends the kernel (up to 8192
    rows, at the o_proj input width of TP 1, 2, 4 and 8) the block is legal
    and fits the budget; a step of up to 256 rows wider than 2048 columns runs
    as one block where that fits, else at the largest legal tile that does."""
    for rows in range(1, 8193):
        padded, block, step = kernel.plan_blocks(rows, cols)
        least = max(rows, 8)
        assert padded == least or (padded % 32 == 0 and padded - least < 32)
        assert _legal(block, padded) and _legal(step, block)
        assert block * cols * _BLOCK_BYTES <= _VMEM_BUDGET, (rows, cols, block)
        if padded <= 256 and cols > 2048:
            fitting = [
                t
                for t in range(1, padded + 1)
                if _legal(t, padded) and t * cols * _BLOCK_BYTES <= _VMEM_BUDGET
            ]
            assert block == max(fitting), (rows, cols, block)


@pytest.mark.cpu_test
def test_a_width_past_the_budget_at_32_rows_is_refused():
    # 32 rows of 53248 columns take 17,039,360 bytes, over the 16 MiB budget.
    with pytest.raises(NotImplementedError, match="16 MiB VMEM budget"):
        kernel.plan_blocks(32, 53248)


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
# The 397B o_proj input per device: 32 query heads of 256 at TP1 (pipeline-
# parallel prefill legs), 16 at TP2 and 4 at TP8 (decode legs). 4 rows take
# the 8-row pad, 96 and 256 are decode steps (one block wider than 2048
# columns; at 8192 columns 256 rows run as two 128-row blocks), 208 pads to
# 224 rows at 8192 columns, 2176 and 8192 are prefill steps (128- to 512-row
# blocks).
@pytest.mark.parametrize("cols", [1024, 4096, 8192])
@pytest.mark.parametrize("rows", [4, 96, 208, 256, 2176, 8192])
def test_compiled_kernel_matches_the_eager_gate_on_tpu(rows, cols):
    """The Mosaic-compiled kernel at its default tiles against the reference.
    The TPU may round one scale a bf16 step away from the reference (an f32
    reduction order), and the fp8 values then move by at most one
    float8_e4m3fn step."""
    if jax.default_backend() != "tpu":
        pytest.skip("requires TPU")
    rng = np.random.default_rng(rows + cols)
    a = jnp.asarray(rng.standard_normal((rows, cols)), dtype=jnp.bfloat16)
    g = jnp.asarray(rng.standard_normal((rows, cols)), dtype=jnp.bfloat16)
    q, s = jax.block_until_ready(pallas_sigmoid_gate_quant(a, g))
    # The reference runs on the host, so it shares no TPU lowering with the
    # kernel.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        q_ref, s_ref = jax.block_until_ready(
            sigmoid_gate_quant_reference(*jax.device_put((a, g), cpu))
        )
    assert q.shape == q_ref.shape == (rows, cols)
    assert q.dtype == q_ref.dtype == jnp.float8_e4m3fn
    assert s.shape == s_ref.shape == (rows, 1)
    np.testing.assert_allclose(
        np.asarray(s, np.float32), np.asarray(s_ref, np.float32), rtol=2.0**-7, atol=0
    )
    assert float(_e4m3_steps(q, q_ref).max()) <= 1.0
