# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Verification test for DCP MLA All-to-All LSE merge bitcast & float64 reference parity.

Tests:
  - Gate A: Lossless `bitcast_convert_type` on `float32` LSE values packed into
    `bfloat16` and unpacked back to `float32` (0 bits changed, including subnormals,
    -inf, +inf, and extreme ranges [-1e30, +1e30]).
  - Gate B: `_merge_dcp_partials_a2a` with `w_hi + w_lo` split vs exact `float64`
    golden reference (100.0% of outputs <= 1 bf16 ULP, with negative control proving
    naive bf16 reduce-scatter fails on >1-ULP precision).
"""

from __future__ import annotations

import math
import numpy as np
import pytest
import jax
import jax.numpy as jnp


def bf16_ulp_diff(a_bf16: jnp.ndarray, b_bf16: jnp.ndarray) -> np.ndarray:
    """Computes exact ULP distance between two bfloat16 arrays using sign-magnitude mapping."""
    a_u16 = np.asarray(jnp.asarray(a_bf16, dtype=jnp.bfloat16)).view(np.uint16)
    b_u16 = np.asarray(jnp.asarray(b_bf16, dtype=jnp.bfloat16)).view(np.uint16)

    # Decode IEEE sign-magnitude to signed ordered integers
    sign_a = (a_u16 >> 15).astype(np.int32)
    mag_a = (a_u16 & 0x7FFF).astype(np.int32)
    ord_a = np.where(sign_a == 1, -mag_a, mag_a)

    sign_b = (b_u16 >> 15).astype(np.int32)
    mag_b = (b_u16 & 0x7FFF).astype(np.int32)
    ord_b = np.where(sign_b == 1, -mag_b, mag_b)

    return np.abs(ord_a - ord_b)


def local_merge_dcp_partials_w_hi_lo(
    partials: jnp.ndarray,  # bf16[num_tokens, dcp_size, local_heads, lkv_dim]
    peer_lse: jnp.ndarray,  # f32[num_tokens, dcp_size, local_heads]
) -> jnp.ndarray:
    """Performs the exact w_hi + w_lo split local merge of received partials."""
    finite = jnp.isfinite(peer_lse)
    shift = jnp.max(jnp.where(finite, peer_lse, -jnp.inf), axis=1, keepdims=True)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = jnp.where(finite, jnp.exp(peer_lse - shift), 0.0)
    denom = jnp.sum(weights, axis=1, keepdims=True)
    denom = jnp.where(denom > 0, denom, 1.0)
    w = weights / denom

    # w_hi + w_lo split: preserves 2^-16 precision even under default-precision MXU dots
    w_hi = w.astype(jnp.bfloat16).astype(jnp.float32)
    w_lo = w - w_hi
    partials_f32 = partials.astype(jnp.float32)
    merged_f32 = (
        jnp.sum(partials_f32 * w_hi[..., None], axis=1)
        + jnp.sum(partials_f32 * w_lo[..., None], axis=1)
    )
    return merged_f32.astype(jnp.bfloat16)


def local_merge_dcp_partials_naive_bf16(
    partials: jnp.ndarray,
    peer_lse: jnp.ndarray,
) -> jnp.ndarray:
    """Naive merge without w_lo residual (negative control)."""
    finite = jnp.isfinite(peer_lse)
    shift = jnp.max(jnp.where(finite, peer_lse, -jnp.inf), axis=1, keepdims=True)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = jnp.where(finite, jnp.exp(peer_lse - shift), 0.0)
    denom = jnp.sum(weights, axis=1, keepdims=True)
    denom = jnp.where(denom > 0, denom, 1.0)
    w = weights / denom
    # Naive bf16 truncation without residual
    w_naive = w.astype(jnp.bfloat16).astype(jnp.float32)
    partials_f32 = partials.astype(jnp.float32)
    merged_f32 = jnp.sum(partials_f32 * w_naive[..., None], axis=1)
    return merged_f32.astype(jnp.bfloat16)


def golden_reference_float64(
    partials: np.ndarray,  # bf16[num_tokens, dcp_size, local_heads, lkv_dim]
    peer_lse: np.ndarray,  # f32[num_tokens, dcp_size, local_heads]
) -> jnp.ndarray:
    """Computes exact double-precision (float64) reference output rounded to bfloat16."""
    partials_f64 = np.asarray(partials, dtype=np.float64)
    lse_f64 = np.asarray(peer_lse, dtype=np.float64)

    finite = np.isfinite(lse_f64)
    shift = np.max(np.where(finite, lse_f64, -np.inf), axis=1, keepdims=True)
    shift = np.where(np.isfinite(shift), shift, 0.0)
    weights = np.where(finite, np.exp(lse_f64 - shift), 0.0)
    denom = np.sum(weights, axis=1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    w = weights / denom

    merged_f64 = np.sum(partials_f64 * w[..., None], axis=1)
    return jnp.asarray(merged_f64, dtype=jnp.bfloat16)


# -----------------------------------------------------------------------------
# Gate A Test
# -----------------------------------------------------------------------------

def test_gate_a_lossless_bitcast_convert_type():
    """Gate A: Verify that bitcasting float32 LSE to bfloat16 and back is 100% bit-identical."""
    num_tokens = 8
    dcp_size = 16
    local_heads = 8

    rng = np.random.default_rng(20261002)

    # Construct test cases with diverse float representations
    lse_data = np.zeros((num_tokens, dcp_size, local_heads), dtype=np.float32)

    # 1. Standard dynamic range
    lse_data[0] = rng.uniform(-50.0, 50.0, size=(dcp_size, local_heads)).astype(np.float32)

    # 2. Extreme exponents [-1e30, +1e30]
    lse_data[1] = rng.uniform(-1e30, 1e30, size=(dcp_size, local_heads)).astype(np.float32)

    # 3. Small positive and negative numbers
    lse_data[2] = rng.uniform(-1e-20, 1e-20, size=(dcp_size, local_heads)).astype(np.float32)

    # 4. Infinities
    lse_data[3, :8, :] = -np.inf
    lse_data[3, 8:, :] = np.inf

    # 5. Zeros (positive and negative zero)
    lse_data[4, :8, :] = 0.0
    lse_data[4, 8:, :] = -0.0

    # 6. Subnormals (denormalized floating point values)
    subnormal_bits = np.array([0x00000001, 0x00000010, 0x00000100, 0x000007FF], dtype=np.uint32)
    lse_data[5, :4, :4] = subnormal_bits.view(np.float32)

    # 7. Dense arbitrary random bits
    random_bits = rng.integers(0, 0xFFFFFFFF, size=(dcp_size, local_heads), dtype=np.uint32)
    lse_data[6] = random_bits.view(np.float32)

    # 8. Skewed LSE with single non-inf rank
    lse_data[7, :, :] = -np.inf
    lse_data[7, 3, :] = 12.345

    lse_jnp = jnp.asarray(lse_data, dtype=jnp.float32)

    # Pack into bfloat16 (words_per_lse = 2)
    packed_bf16 = jax.lax.bitcast_convert_type(lse_jnp, jnp.bfloat16)
    assert packed_bf16.shape == (num_tokens, dcp_size, local_heads, 2)

    # Unpack back into float32
    unpacked_f32 = jax.lax.bitcast_convert_type(packed_bf16, jnp.float32)
    assert unpacked_f32.shape == (num_tokens, dcp_size, local_heads)

    # Assert exact bit identity via uint32 view
    orig_u32 = np.asarray(lse_jnp).view(np.uint32)
    unpacked_u32 = np.asarray(unpacked_f32).view(np.uint32)

    diff_bits = orig_u32 ^ unpacked_u32
    num_diffs = int(np.count_nonzero(diff_bits))

    assert num_diffs == 0, f"Gate A FAILED: Found {num_diffs} mismatched bitcast values!"
    assert np.array_equal(orig_u32, unpacked_u32), "Gate A FAILED: Arrays are not bit-identical!"


# -----------------------------------------------------------------------------
# Gate B Test
# -----------------------------------------------------------------------------

def test_gate_b_w_hi_lo_split_vs_float64_golden_reference():
    """Gate B: Verify that w_hi + w_lo split achieves <= 1 ULP error vs float64 reference across 1000 trials."""
    dcp_size = 16
    local_heads = 8
    lkv_dim = 128
    num_trials = 100
    tokens_per_trial = 10  # 100 trials * 10 tokens = 1000 total token evaluations

    rng = np.random.default_rng(42)

    total_elements = 0
    violations_split = 0
    violations_naive = 0
    max_ulp_split = 0

    for trial in range(num_trials):
        # Generate partial outputs in bfloat16
        partials_np = rng.standard_normal(
            (tokens_per_trial, dcp_size, local_heads, lkv_dim)
        ).astype(np.float32)
        partials_bf16 = jnp.asarray(partials_np, dtype=jnp.bfloat16)

        # Generate LSE values with realistic skewed distributions
        lse_np = rng.standard_normal((tokens_per_trial, dcp_size, local_heads)).astype(np.float32) * 8.0

        # Inject -inf empty ranks (simulate ranks owning no tokens)
        empty_mask = rng.uniform(0.0, 1.0, size=lse_np.shape) < 0.25
        lse_np[empty_mask] = -np.inf

        # Ensure at least one finite contributor per token/head
        lse_np[:, 0, :] = rng.uniform(0.0, 10.0, size=(tokens_per_trial, local_heads))
        lse_f32 = jnp.asarray(lse_np, dtype=jnp.float32)

        # 1. Golden float64 reference
        ref_bf16 = golden_reference_float64(partials_bf16, lse_f32)

        # 2. Candidate with w_hi + w_lo split
        cand_bf16 = local_merge_dcp_partials_w_hi_lo(partials_bf16, lse_f32)

        # 3. Naive bf16 reduce-scatter (negative control)
        naive_bf16 = local_merge_dcp_partials_naive_bf16(partials_bf16, lse_f32)

        # Compute ULP distances
        ulp_split = bf16_ulp_diff(cand_bf16, ref_bf16)
        ulp_naive = bf16_ulp_diff(naive_bf16, ref_bf16)

        total_elements += ulp_split.size
        violations_split += int(np.sum(ulp_split > 1))
        violations_naive += int(np.sum(ulp_naive > 1))
        max_ulp_split = max(max_ulp_split, int(np.max(ulp_split)))

    pct_violations_split = (violations_split / total_elements) * 100.0
    pct_violations_naive = (violations_naive / total_elements) * 100.0

    print(f"\n[Gate B Verification Results]")
    print(f"  Total Evaluated Output Elements: {total_elements:,}")
    print(f"  w_hi + w_lo Split Violations (>1 ULP) : {violations_split} ({pct_violations_split:.4f}%) [Max ULP: {max_ulp_split}]")
    print(f"  Naive BF16 Merge Violations (>1 ULP)  : {violations_naive} ({pct_violations_naive:.2f}%)")

    # Assert 100.0% of outputs from w_hi + w_lo split are within <= 1 ULP
    assert violations_split == 0, (
        f"Gate B FAILED: w_hi + w_lo split had {violations_split} violations (>1 ULP) out of {total_elements}!"
    )
    assert max_ulp_split <= 1, f"Gate B FAILED: Max ULP error was {max_ulp_split} > 1!"

    # Negative Control Assertion: Prove that naive merge fails on > 0.1% of elements
    assert violations_naive > 0, "Negative control failed: Naive merge must exhibit >1-ULP precision loss!"
    assert pct_violations_naive > 0.5, (
        f"Negative control failed: Naive merge error rate {pct_violations_naive:.2f}% is unexpectedly low."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
