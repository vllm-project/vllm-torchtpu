# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for sort-free compressed_write compaction and adaptive MoE rungs."""

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.experimental.adaptive_fused_moe import kernel as moe_kernel
from vllm_torchtpu.kernels.qwen4_exp import compressed_write


@pytest.mark.parametrize("n", [8, 32, 128, 1024])
@pytest.mark.parametrize("size", [8, 32, 128])
@pytest.mark.parametrize("fill_value", [0, 1024, 9999])
def test_compact_indices_matches_jnp_nonzero(n, size, fill_value):
    rng = np.random.default_rng(seed=n * 1000 + size * 10 + fill_value)
    for prob in (0.0, 0.05, 0.25, 0.75, 1.0):
        mask = jnp.asarray(rng.random(n) < prob, dtype=jnp.bool_)
        got = np.asarray(compressed_write._compact_indices(mask, size, fill_value))
        (expected,) = jnp.nonzero(mask, size=size, fill_value=fill_value)
        np.testing.assert_array_equal(got, np.asarray(expected, dtype=np.int32))
        assert got.dtype == np.int32


def test_nonzero_indices_flag_parity(monkeypatch):
    mask = jnp.asarray(
        [False, True, False, True, True, False, False, True], dtype=jnp.bool_
    )
    monkeypatch.delenv("VLLM_TPU_QSA_SORTFREE_COMPACT", raising=False)
    base = np.asarray(compressed_write._nonzero_indices(mask, 8, 99))
    monkeypatch.setenv("VLLM_TPU_QSA_SORTFREE_COMPACT", "1")
    opt = np.asarray(compressed_write._nonzero_indices(mask, 8, 99))
    np.testing.assert_array_equal(opt, base)


def test_moe_prefill_rungs(monkeypatch):
    monkeypatch.setenv("MOE_ADAPTIVE_PREFILL_RUNGS", "1")
    assert moe_kernel._prefill_rungs() is True
    assert moe_kernel.tile_height_rungs(128, 100000, 64, 8) == (32, 64, 96, 128)
    monkeypatch.setenv("MOE_ADAPTIVE_PREFILL_RUNGS", "0")
    assert moe_kernel._prefill_rungs() is False
    assert moe_kernel.tile_height_rungs(128, 100000, 64, 8) == (128,)
