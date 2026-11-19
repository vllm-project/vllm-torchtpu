# SPDX-License-Identifier: Apache-2.0
"""`reduce_precision` BF16 rounding is bitwise identical to the integer RNE."""

from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from vllm_torchtpu import envs
from vllm_torchtpu.kernels.qwen4_exp import rounding


def _bf16_bits(x: np.ndarray, reduce_precision: bool) -> np.ndarray:
    with mock.patch.object(
        envs, "VLLM_TPU_QSA_BF16_REDUCE_PRECISION", reduce_precision
    ):
        # A fresh jit per call: the flag is read at trace time.
        out = jax.jit(lambda v: rounding.bf16(v))(jnp.asarray(x))
    return np.asarray(jax.lax.bitcast_convert_type(out, jnp.uint16))


def _inputs() -> np.ndarray:
    # Every finite BF16 value as an f32 bit pattern ...
    hi = np.arange(1 << 16, dtype=np.uint32) << 16
    finite = hi[np.isfinite(hi.view(np.float32))]
    # ... plus, around each, the exact tie, one ulp either side of it, and the
    # largest f32 below the next BF16 value: every branch of RNE.
    offsets = np.array([0, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFF], dtype=np.uint32)
    bits = (finite[:, None] + offsets[None, :]).reshape(-1)
    rng = np.random.default_rng(0)
    bits = np.concatenate([bits, rng.integers(0, 1 << 32, 1 << 20, dtype=np.uint32)])
    x = bits.view(np.float32)
    return x[np.isfinite(x)]


def test_reduce_precision_matches_integer_rne_on_all_finite_inputs():
    x = _inputs()
    expected = _bf16_bits(x, reduce_precision=False)
    got = _bf16_bits(x, reduce_precision=True)
    mismatch = np.flatnonzero(expected != got)
    assert mismatch.size == 0, (
        f"{mismatch.size} of {x.size} differ, first input bits "
        f"{x[mismatch[:4]].view(np.uint32)}"
    )
