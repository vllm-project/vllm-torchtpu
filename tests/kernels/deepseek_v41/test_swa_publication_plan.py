# SPDX-License-Identifier: Apache-2.0
"""A step-level SWA publication plan packs values exactly like the inline path."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.mla.paged_swa import (
    apply_swa_bank_plan,
    plan_swa_bank,
    plan_swa_pages,
)


def _inline(records, values, *, base, capacity, page_size, padded):
    """The original per-layer packing, kept verbatim as the reference."""
    valid = (records >= max(base, page_size)) & (records < base + capacity)
    local = jnp.where(valid, records - base, capacity)
    order = jnp.argsort(local, stable=True)
    ids = local[order]
    next_ids = jnp.concatenate((ids[1:], jnp.array([capacity], jnp.int32)))
    pairs = ids // 2
    boundary = (ids < capacity) & jnp.concatenate(
        (jnp.ones(1, jnp.bool_), pairs[1:] != pairs[:-1])
    )
    even = ids % 2 == 0
    has_odd = (~even) | (next_ids // 2 == pairs)
    bits = jax.lax.bitcast_convert_type(values[order], jnp.uint16).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)))
    low = jnp.where(even[:, None], bits, jnp.uint32(0))
    high = jnp.where(even[:, None], next_bits, bits)
    high = jnp.where(has_odd[:, None], high, jnp.uint32(0))
    words = jax.lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        has_odd, jnp.int32(-65536), jnp.int32(0)
    )
    padding = padded - records.size
    return (
        jnp.pad(jnp.where(boundary, pairs, -1), (0, padding), constant_values=-1),
        jnp.broadcast_to(jnp.pad(masks, (0, padding))[:, None], (padded, 512)),
        jnp.pad(words, ((0, padding), (0, 0))),
    )


@pytest.mark.parametrize("tokens", [1, 7, 96, 288])
@pytest.mark.parametrize("banked", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "fragmented", "invalid"])
def test_plan_matches_inline_packing(tokens, banked, layout):
    rng = np.random.default_rng(tokens * 7 + banked)
    page_size, pages = 64, 64
    capacities = (pages * page_size // 2,) * 2 if banked else (pages * page_size,)
    total = sum(capacities)
    if layout == "contiguous":
        records = np.arange(tokens) + 3 * page_size + 1
    elif layout == "fragmented":
        records = rng.choice(np.arange(page_size, total), tokens, replace=False)
    else:
        # Null page, negative and out-of-range records are all dropped.
        records = rng.choice(np.arange(page_size, total), tokens, replace=False)
        records[::3] = rng.choice([-1, 0, page_size - 1, total, total + 5])
    records = jnp.asarray(records, jnp.int32)
    values = jax.lax.bitcast_convert_type(
        jnp.asarray(rng.integers(0, 1 << 16, (tokens, 512)), jnp.uint16),
        jnp.bfloat16,
    )
    padded = -(-tokens // 256) * 256
    plans = plan_swa_pages(records, capacities=capacities, page_size=page_size)
    base = 0
    for capacity, plan in zip(capacities, plans, strict=True):
        assert all(p.shape == (tokens,) and p.dtype == jnp.int32 for p in plan)
        expected = _inline(
            records,
            values,
            base=base,
            capacity=capacity,
            page_size=page_size,
            padded=padded,
        )
        for got, want in zip(
            apply_swa_bank_plan(plan, values, padded), expected, strict=True
        ):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        single = plan_swa_bank(
            records, base=base, capacity=capacity, page_size=page_size
        )
        for got, want in zip(single, plan, strict=True):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        base += capacity
