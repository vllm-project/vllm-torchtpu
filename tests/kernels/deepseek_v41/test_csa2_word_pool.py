# SPDX-License-Identifier: Apache-2.0
"""SWA publication preserves the shared 128-word pool and unrelated records."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.paged_swa import swa_word_pages, write_swa_word_pool


@pytest.mark.parametrize("banked", [False, True])
def test_publish_shared_word_pool(banked):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore publication requires TPU")
    page_size, rows = 64, 16384
    rng = np.random.default_rng(17)
    original = rng.integers(-(1 << 31), 1 << 31, (rows, 128), dtype=np.int32)
    records = np.array([-1, 0, 63, 64, 65, 67, 70, rows // 4, rows // 2 - 1], np.int32)
    values = np.asarray(
        jnp.asarray(rng.standard_normal((len(records), 512)), jnp.bfloat16)
    )
    bits = values.view(np.uint16).astype(np.uint32)
    expected = original.copy().view(np.uint32).reshape(-1, 512)
    for record, word in zip(records, bits):
        if record >= page_size:
            shift = int(record % 2) * 16
            mask = np.uint32(65535 << shift)
            expected[record // 2] = (expected[record // 2] & ~mask) | (word << shift)
    expected = expected.view(np.int32).reshape(rows, 128)
    raw = jax.device_put(original)
    if banked:
        raw = (raw[: rows // 2], raw[rows // 2 :])
    compiled = write_swa_word_pool.lower(
        raw, jnp.asarray(records), jnp.asarray(values), page_size=page_size
    ).compile()
    memory = compiled.memory_analysis()
    assert memory.alias_size_in_bytes == original.nbytes
    assert memory.temp_size_in_bytes < original.nbytes
    updated = compiled(raw, jnp.asarray(records), jnp.asarray(values))
    actual = np.concatenate(jax.device_get(updated)) if banked else np.asarray(updated)
    np.testing.assert_array_equal(actual, expected)
    # Adding the page axis must be a bitcast, without a pool-sized HBM copy.
    view = jax.jit(lambda x: swa_word_pages(x, page_size)).lower(updated).compile()
    assert view.memory_analysis().temp_size_in_bytes == 0
    pages = view(updated)
    for bank in pages if banked else (pages,):
        assert bank.shape[1:] == (2 * page_size, 128)
