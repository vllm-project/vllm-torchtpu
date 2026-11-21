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


def _plan_records(case, capacity, rng):
    if case == "decode":  # one request's six consecutive rows, then padding
        return np.array([200, 201, 202, 203, 204, 205] + [-1] * 10, np.int32)
    if case == "null_page":  # rows of the null page are dropped at row offset 0 only
        return np.array(
            [3, 4, 5, 130, 131, -1, 64, 65, 2, 9, -1, 66, 1, 0, 70, 71], np.int32
        )
    if case == "bounds":  # rows a shift moves across the pool's end
        top = capacity - 64
        return np.array([top + 1, top, top + 63, 96, -1, 97, 1000, 1001] * 2, np.int32)
    if case == "duplicates":
        return np.array([300, 300, 301, 303, 303, -1, 64, 64] * 2, np.int32)
    records = 64 + rng.permutation(capacity - 128)[:44].astype(np.int32)  # not 8k
    records[::7] = -1
    return records


def _emulate_publish(pool, ids, words, masks):
    pool = pool.copy()
    for row, record in enumerate(np.asarray(ids)):
        if record >= 0:
            mask = np.asarray(masks)[row % len(masks)]
            pool[record] = (pool[record] & ~mask) | np.asarray(words)[row]
    return pool


@pytest.mark.parametrize(
    "case", ["decode", "null_page", "bounds", "duplicates", "random"]
)
@pytest.mark.parametrize("shift", [0, 64, 192, -64])
def test_publish_plan_matches_shifted_native(monkeypatch, case, shift):
    """A plan of base records publishes exactly what the native pack publishes
    for the same records shifted by whole pages, as attention shifts them."""
    from types import SimpleNamespace

    from vllm_torchtpu.kernels.mla import paged_swa

    sparse_core = SimpleNamespace(num_lanes=16, num_cores=2, num_subcores=16)
    monkeypatch.setattr(
        paged_swa.pltpu,
        "get_tpu_info",
        lambda: SimpleNamespace(sparse_core=sparse_core),
    )
    capacity, page_size = 4096, 64
    rng = np.random.default_rng([len(case), shift % 4096])
    records = _plan_records(case, capacity, rng)
    bits = rng.integers(0, 1 << 16, (records.size, 512), dtype=np.uint16)
    values = jnp.asarray(bits.view(jnp.bfloat16))
    pool = rng.integers(-(1 << 31), 1 << 31, (2 * capacity, 128), dtype=np.int32)

    shifted = jnp.asarray(np.where(records >= 0, records + shift, records))
    native = paged_swa._pack_native(shifted, values, capacity, page_size)
    # The plan's other group holds another table's records, as attention batches
    # every SWA group of a stage into one plan.
    other = np.where(records >= 0, (records + 640) % capacity, records)
    plan = paged_swa.swa_publish_plan(jnp.asarray(np.stack((other, records))))
    assert plan.lead.shape[-1] % 512 == 0
    ids, words, masks = paged_swa._pack_planned(
        values, plan, 1, shift, page_size, capacity, interpret=True
    )
    np.testing.assert_array_equal(
        _emulate_publish(pool, ids, words, masks), _emulate_publish(pool, *native)
    )


@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("shift", [0, 128])
@pytest.mark.parametrize("every_pattern", [False, True])
def test_planned_publish_matches_word_pool(banked, shift, every_pattern):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore publication requires TPU")
    from vllm_torchtpu.kernels.mla.paged_swa import (
        swa_publish_plan,
        write_swa_word_pool_planned,
    )

    page_size, rows = 64, 16384
    rng = np.random.default_rng(23)
    original = rng.integers(-(1 << 31), 1 << 31, (rows, 128), dtype=np.int32)
    records = np.array(
        [-1, 0, 63, 64, 65, 67, 70, rows // 4, rows // 2 - 65, 5, -1, 300, 301, 302],
        np.int32,
    )
    values = jnp.asarray(rng.standard_normal((len(records), 512)), jnp.bfloat16)
    if every_pattern:  # all 2^16 BF16 bit patterns, NaNs and denormals included
        records = np.arange(64, 192, dtype=np.int32)
        bits = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16)
        values = jnp.asarray(bits.reshape(128, 512).view(jnp.bfloat16))
    shifted = jnp.asarray(np.where(records >= 0, records + shift, records))

    def pool():
        raw = jax.device_put(original)
        return (raw[: rows // 2], raw[rows // 2 :]) if banked else raw

    expected = write_swa_word_pool(pool(), shifted, values, page_size=page_size)
    # Group 1 of a two-group plan, as attention plans a stage's SWA groups.
    other = np.where(records >= 0, rows // 2 - 1 - records, records)
    plan = jax.jit(swa_publish_plan)(jnp.asarray(np.stack((other, records))))
    actual = write_swa_word_pool_planned(
        pool(), shifted, values, plan, group=1, shift=shift, page_size=page_size
    )
    if banked:
        expected, actual = np.concatenate(expected), np.concatenate(actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
