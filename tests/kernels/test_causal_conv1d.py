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
"""Unit and TPU hardware tests for the Pallas ragged_causal_conv1d kernel."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.causal_conv1d import strided_ldst
from vllm_torchtpu.kernels.causal_conv1d.causal_conv1d import (
    ConvConfigs,
    _estimate_conv1d_vmem_bytes,
    _select_conv1d_tile_size_and_vmem_limit,
    preprocess_metadata,
    ragged_causal_conv1d,
)


def _ragged_causal_conv1d_ref(
    x: jax.Array,
    conv_state: jax.Array,
    conv_weight: jax.Array,
    conv_bias: jax.Array | None,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    distribution: jax.Array,
    has_initial_state: jax.Array,
    *,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Step-by-step float32 reference for ragged_causal_conv1d."""
    x_f32 = np.asarray(x.astype(jnp.float32), dtype=np.float32)
    orig_state_shape = conv_state.shape
    state_3d = conv_state[:, :, 0, :] if conv_state.ndim == 4 else conv_state
    expected_state = np.array(state_3d.astype(jnp.float32), dtype=np.float32, copy=True)
    w_f32 = np.asarray(conv_weight[:, 0, :].astype(jnp.float32), dtype=np.float32)
    b_f32 = (
        np.asarray(conv_bias.astype(jnp.float32), dtype=np.float32)
        if conv_bias is not None
        else None
    )

    dim = x_f32.shape[1]
    prev_k = kernel_size - 1
    out_f32 = np.zeros_like(x_f32)
    num_active_seqs = int(np.asarray(distribution)[2])
    qsl = np.asarray(query_start_loc)
    slots = np.asarray(state_indices)
    has_init = np.asarray(has_initial_state)

    for s in range(num_active_seqs):
        q_start = int(qsl[s])
        q_end = int(qsl[s + 1])
        q_len = q_end - q_start
        if q_len <= 0:
            continue

        slot = int(slots[s])
        prev_rows = (
            expected_state[slot].copy()
            if bool(has_init[s])
            else np.zeros((prev_k, dim), dtype=np.float32)
        )
        window = np.concatenate([prev_rows, x_f32[q_start:q_end]], axis=0)
        for t in range(q_len):
            acc = np.zeros((dim,), dtype=np.float32)
            for k in range(kernel_size):
                acc += window[t + k] * w_f32[:, k]
            if b_f32 is not None:
                acc += b_f32
            out_f32[q_start + t] = acc
        expected_state[slot] = window[q_len : q_len + prev_k]

    out_cast = np.asarray(jnp.asarray(out_f32).astype(x.dtype).astype(jnp.float32))
    state_cast = np.asarray(
        jnp.asarray(expected_state).astype(conv_state.dtype).astype(jnp.float32)
    ).reshape(orig_state_shape)
    return out_cast, state_cast


def test_conv1d_reference_and_preprocess_metadata_on_cpu():
    dim = 4
    kernel_size = 4
    x = jnp.arange(24 * dim, dtype=jnp.float32).reshape(24, dim) * 0.1
    conv_state = jnp.arange(6 * 3 * dim, dtype=jnp.float32).reshape(6, 3, dim) * 0.05
    conv_weight = jnp.ones((dim, 1, kernel_size), dtype=jnp.float32)
    conv_bias = jnp.full((dim,), 0.25, dtype=jnp.float32)

    # 3 active sequences (1 decode w/ init, 1 decode w/o init, 1 prefill len=17)
    # = 19 valid tokens inside a 32-row padded batch, plus 2 inactive seq slots
    # carrying -1 sentinels.
    query_start_loc = jnp.array([0, 1, 2, 19, 24, 24], dtype=jnp.int32)
    state_indices = jnp.array([2, 0, 4, -1, -1], dtype=jnp.int32)
    distribution = jnp.array([2, 2, 3], dtype=jnp.int32)
    has_initial_state = jnp.array([True, False, True, True, True], dtype=bool)

    out_ref, state_ref = _ragged_causal_conv1d_ref(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    # Seq 1 has has_initial_state=False and q_len=1: only x[1] * w[:, 3] + bias.
    np.testing.assert_allclose(
        out_ref[1], np.asarray(x[1]) + 0.25, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_array_equal(state_ref[1], np.asarray(conv_state[1]))
    np.testing.assert_array_equal(state_ref[3], np.asarray(conv_state[3]))
    np.testing.assert_array_equal(state_ref[5], np.asarray(conv_state[5]))

    cfgs = ConvConfigs(
        batch_size=32, dim_size=dim, kernel_size=kernel_size, tile_size=16
    )
    metadata = preprocess_metadata(
        cfgs=cfgs,
        query_start_loc=query_start_loc,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        num_seqs=distribution[-1],
    )
    assert int(metadata.num_tiles) == 2
    # Padded tail rows >= 19 must issue zero-length old-state reads and zero writes.
    np.testing.assert_array_equal(np.asarray(metadata.b_idx_to_sz_from_old[19:]), 0)
    np.testing.assert_array_equal(np.asarray(metadata.b_idx_should_write[19:]), 0)
    write_rows = np.flatnonzero(np.asarray(metadata.b_idx_should_write))
    np.testing.assert_array_equal(write_rows, np.array([0, 1, 18]))
    # Inactive -1 sentinels must be clamped to 0 in s_idx_to_state_idx.
    np.testing.assert_array_equal(
        np.asarray(metadata.s_idx_to_state_idx), np.array([2, 0, 4, 0, 0])
    )

    # Inter-tile boundary case: sequence 1 starts at row 15 (last row of tile 0
    # when tile_size=16) and continues into tile 1 (rows 16..39). Rows 16 and 17
    # at the start of tile 1 combine `sz_from_old` rows from `conv_state` with
    # earlier sequence-1 rows from `prev_x_scratch`.
    qsl_boundary = jnp.array([0, 15, 40], dtype=jnp.int32)
    cfgs_boundary = ConvConfigs(
        batch_size=48, dim_size=dim, kernel_size=kernel_size, tile_size=16
    )
    meta_boundary = preprocess_metadata(
        cfgs=cfgs_boundary,
        query_start_loc=qsl_boundary,
        state_indices=jnp.array([1, 3], dtype=jnp.int32),
        has_initial_state=jnp.array([True, True], dtype=bool),
        num_seqs=jnp.array(2, dtype=jnp.int32),
    )
    assert int(meta_boundary.num_tiles) == 3
    # Row 15 (seq 1, t=0) needs 3 rows from conv_state; row 16 (tile 1, idx=0,
    # seq 1, t=1) needs 2 rows from conv_state + 1 row (x[15]) from prev_x_scratch;
    # row 17 (tile 1, idx=1, seq 1, t=2) needs 1 row from conv_state + 2 rows
    # (x[15], x[16]) from prev_x_scratch/x_compact; row 18 (t=3) needs 0 from old.
    np.testing.assert_array_equal(
        np.asarray(meta_boundary.b_idx_to_sz_from_old[14:20]),
        np.array([0, 3, 2, 1, 0, 0]),
    )
    np.testing.assert_array_equal(
        np.flatnonzero(np.asarray(meta_boundary.b_idx_should_write)),
        np.array([14, 39]),
    )


@pytest.mark.parametrize(
    ("sublane_tiling", "x_itemsize"),
    [
        (16, 2),  # bfloat16
        (8, 4),  # float32
    ],
)
@pytest.mark.parametrize("vmem_mib", [16, 8])
def test_select_conv1d_tile_size_and_vmem_limit(
    sublane_tiling: int, x_itemsize: int, vmem_mib: int
):
    vmem_capacity_bytes = vmem_mib * 1024 * 1024
    tile_size, vmem_limit_bytes = _select_conv1d_tile_size_and_vmem_limit(
        batch_size=128,
        dim=8448,
        kernel_size=4,
        requested_tile_size=64,
        sublane_tiling=sublane_tiling,
        x_itemsize=x_itemsize,
        has_bias=True,
        vmem_capacity_bytes=vmem_capacity_bytes,
    )
    assert vmem_limit_bytes == int(0.90 * vmem_capacity_bytes)
    assert tile_size < 64
    assert tile_size >= sublane_tiling
    assert tile_size % sublane_tiling == 0
    if tile_size > sublane_tiling:
        est = _estimate_conv1d_vmem_bytes(
            tile_size=tile_size,
            dim=8448,
            kernel_size=4,
            x_itemsize=x_itemsize,
            has_bias=True,
        )
        assert est <= int(0.80 * vmem_capacity_bytes)


def test_select_conv1d_tile_size_rejects_unaligned_or_nonpositive_tile():
    with pytest.raises(ValueError, match="tile_size > 0"):
        _select_conv1d_tile_size_and_vmem_limit(
            batch_size=16,
            dim=128,
            kernel_size=4,
            requested_tile_size=0,
            sublane_tiling=16,
            x_itemsize=2,
            has_bias=False,
            vmem_capacity_bytes=16 * 1024 * 1024,
        )
    with pytest.raises(ValueError, match="divisible by sublane_tiling"):
        _select_conv1d_tile_size_and_vmem_limit(
            batch_size=16,
            dim=128,
            kernel_size=4,
            requested_tile_size=24,
            sublane_tiling=16,
            x_itemsize=2,
            has_bias=False,
            vmem_capacity_bytes=16 * 1024 * 1024,
        )


def test_ragged_causal_conv1d_validates_inputs_on_cpu():
    dim = 128
    kernel_size = 4
    num_seqs = 2
    x = jnp.zeros((8, dim), dtype=jnp.float32)
    conv_state = jnp.zeros((4, kernel_size - 1, dim), dtype=jnp.float32)
    conv_weight = jnp.zeros((dim, 1, kernel_size), dtype=jnp.float32)
    conv_bias = jnp.zeros((dim,), dtype=jnp.float32)
    qsl = jnp.array([0, 4, 8], dtype=jnp.int32)
    state_indices = jnp.array([0, 1], dtype=jnp.int32)
    dist = jnp.array([0, 2, 2], dtype=jnp.int32)
    has_init = jnp.array([False, True], dtype=bool)

    # Non-float32 conv_state must raise ValueError before TPU lookup.
    with pytest.raises(ValueError, match=r"Expected conv_state\.dtype == float32"):
        ragged_causal_conv1d(
            x,
            conv_state.astype(jnp.bfloat16),
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    # Dim-first (DS) conv_state layout must raise ValueError before TPU lookup.
    conv_state_ds = jnp.zeros((4, dim, kernel_size - 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="Expected conv_state shape"):
        ragged_causal_conv1d(
            x,
            conv_state_ds,
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match="Expected 2D x"):
        ragged_causal_conv1d(
            jnp.zeros((2, 4, dim), dtype=jnp.float32),
            conv_state,
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match=r"Expected x\.dtype"):
        ragged_causal_conv1d(
            jnp.zeros((8, dim), dtype=jnp.int32),
            conv_state,
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match="kernel_size >= 2"):
        ragged_causal_conv1d(
            x,
            jnp.zeros((4, 0, dim), dtype=jnp.float32),
            jnp.zeros((dim, 1, 1), dtype=jnp.float32),
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=1,
        )

    with pytest.raises(ValueError, match="tile_size > 0"):
        ragged_causal_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
            tile_size=0,
        )

    with pytest.raises(ValueError, match="Expected conv_weight shape"):
        ragged_causal_conv1d(
            x,
            conv_state,
            jnp.zeros((dim, kernel_size), dtype=jnp.float32),
            conv_bias,
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match="Expected conv_bias shape"):
        ragged_causal_conv1d(
            x,
            conv_state,
            conv_weight,
            jnp.zeros((dim + 1,), dtype=jnp.float32),
            qsl,
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match="Expected query_start_loc shape"):
        ragged_causal_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            jnp.zeros((num_seqs,), dtype=jnp.int32),
            state_indices,
            dist,
            has_init,
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match="Expected has_initial_state shape"):
        ragged_causal_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            dist,
            jnp.zeros((num_seqs + 1,), dtype=bool),
            kernel_size=kernel_size,
        )

    with pytest.raises(ValueError, match="Expected distribution shape"):
        ragged_causal_conv1d(
            x,
            conv_state,
            conv_weight,
            conv_bias,
            qsl,
            state_indices,
            jnp.zeros((2,), dtype=jnp.int32),
            has_init,
            kernel_size=kernel_size,
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("batch_size", [4, 19])
def test_ragged_causal_conv1d_tpu_decode_and_initial_state(dtype, batch_size: int):
    dim = 128
    kernel_size = 4
    num_slots = 8
    rng = jax.random.key(101 + batch_size)
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    conv_state = jax.random.normal(
        k2, (num_slots, kernel_size - 1, dim), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    conv_bias = jax.random.normal(k4, (dim,), dtype=dtype)

    query_start_loc = jnp.array([0, 1, 2, 3, 4, 4], dtype=jnp.int32)
    state_indices = jnp.array([5, 1, 7, 2, -1], dtype=jnp.int32)
    distribution = jnp.array([4, 4, 4], dtype=jnp.int32)
    has_initial_state = jnp.array([True, False, True, False, True], dtype=bool)

    expected_out, expected_state = _ragged_causal_conv1d_ref(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d(
        x.copy(),
        conv_state.copy(),
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )

    assert actual_out.shape == (batch_size, dim)
    assert actual_out.dtype == dtype
    tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual_out[:4].astype(jnp.float32)),
        expected_out[:4],
        rtol=tol,
        atol=tol,
    )
    for slot in (5, 1, 7, 2):
        np.testing.assert_allclose(
            np.asarray(actual_state[slot]),
            expected_state[slot],
            rtol=tol,
            atol=tol,
        )
    for untouched in (0, 3, 4, 6):
        np.testing.assert_array_equal(
            np.asarray(actual_state[untouched]), conv_state_orig[untouched]
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize(
    ("qsl_starts", "seq1_has_init", "use_4d_state"),
    [
        ((0, 21, 40), False, False),
        ((0, 15, 40), False, False),
        ((0, 15, 40), True, False),
        ((0, 15, 40), True, True),
    ],
)
def test_ragged_causal_conv1d_tpu_multi_tile_prefill(
    dtype,
    use_bias: bool,
    qsl_starts: tuple[int, int, int],
    seq1_has_init: bool,
    use_4d_state: bool,
):
    dim = 256
    kernel_size = 4
    tile_size = 16
    batch_size = 40
    num_slots = 4
    rng = jax.random.key(202 + int(use_bias) + qsl_starts[1] + int(seq1_has_init))
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    conv_state_shape = (
        (num_slots, kernel_size - 1, 1, dim)
        if use_4d_state
        else (num_slots, kernel_size - 1, dim)
    )
    conv_state = jax.random.normal(k2, conv_state_shape, dtype=jnp.float32)
    conv_weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    conv_bias = jax.random.normal(k4, (dim,), dtype=dtype) if use_bias else None

    query_start_loc = jnp.array(qsl_starts, dtype=jnp.int32)
    state_indices = jnp.array([1, 3], dtype=jnp.int32)
    distribution = jnp.array([0, 2, 2], dtype=jnp.int32)
    has_initial_state = jnp.array([True, seq1_has_init], dtype=bool)

    expected_out, expected_state = _ragged_causal_conv1d_ref(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d(
        x.copy(),
        conv_state.copy(),
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
        tile_size=tile_size,
    )

    tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual_out.astype(jnp.float32)), expected_out, rtol=tol, atol=tol
    )
    for slot in (1, 3):
        np.testing.assert_allclose(
            np.asarray(actual_state[slot]),
            expected_state[slot],
            rtol=tol,
            atol=tol,
        )
    for untouched in (0, 2):
        np.testing.assert_array_equal(
            np.asarray(actual_state[untouched]), conv_state_orig[untouched]
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("inactive_sentinel", [0, -1])
def test_ragged_causal_conv1d_tpu_mixed_ragged_padded_tail_and_inactive_slots(
    dtype, inactive_sentinel: int
):
    dim = 128
    kernel_size = 4
    tile_size = 16
    batch_size = 37  # Non-multiple of tile_size (37 -> 48), 31 active tokens.
    num_slots = 7
    rng = jax.random.key(303 + (10 if inactive_sentinel < 0 else 0))
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    conv_state = jax.random.normal(
        k2, (num_slots, kernel_size - 1, dim), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    conv_bias = jax.random.normal(k4, (dim,), dtype=dtype)

    # 2 decodes (len 1, 1) + 2 prefills (len 20, 9) = 31 valid tokens, plus 2
    # inactive sequence entries in metadata.
    query_start_loc = jnp.array([0, 1, 2, 22, 31, 37, 37], dtype=jnp.int32)
    state_indices = jnp.array(
        [0, 2, 5, 4, inactive_sentinel, inactive_sentinel], dtype=jnp.int32
    )
    distribution = jnp.array([2, 2, 4], dtype=jnp.int32)
    has_initial_state = jnp.array([True, False, True, False, True, True], dtype=bool)

    expected_out, expected_state = _ragged_causal_conv1d_ref(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d(
        x.copy(),
        conv_state.copy(),
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
        tile_size=tile_size,
    )

    assert actual_out.shape == (batch_size, dim)
    assert actual_out.dtype == dtype
    tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(actual_out[:31].astype(jnp.float32)),
        expected_out[:31],
        rtol=tol,
        atol=tol,
    )
    for slot in (0, 2, 5, 4):
        np.testing.assert_allclose(
            np.asarray(actual_state[slot]),
            expected_state[slot],
            rtol=tol,
            atol=tol,
        )
    for untouched in (1, 3, 6):
        np.testing.assert_array_equal(
            np.asarray(actual_state[untouched]), conv_state_orig[untouched]
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_ragged_causal_conv1d_tpu_large_dim_vmem_fit(dtype):
    dim = 8448
    kernel_size = 4
    batch_size = 50
    num_slots = 4
    rng = jax.random.key(404)
    k1, k2, k3, k4 = jax.random.split(rng, 4)

    x = jax.random.normal(k1, (batch_size, dim), dtype=dtype)
    conv_state = jax.random.normal(
        k2, (num_slots, kernel_size - 1, dim), dtype=jnp.float32
    )
    conv_weight = jax.random.normal(k3, (dim, 1, kernel_size), dtype=dtype)
    conv_bias = jax.random.normal(k4, (dim,), dtype=dtype)

    query_start_loc = jnp.array([0, 1, 26, 43, 50], dtype=jnp.int32)
    state_indices = jnp.array([0, 2, 3, -1], dtype=jnp.int32)
    distribution = jnp.array([1, 1, 3], dtype=jnp.int32)
    has_initial_state = jnp.array([True, True, False, True], dtype=bool)

    expected_out, expected_state = _ragged_causal_conv1d_ref(
        x,
        conv_state,
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
    )
    conv_state_orig = np.array(conv_state, copy=True)

    actual_out, actual_state = ragged_causal_conv1d(
        x.copy(),
        conv_state.copy(),
        conv_weight,
        conv_bias,
        query_start_loc,
        state_indices,
        distribution,
        has_initial_state,
        kernel_size=kernel_size,
        tile_size=64,
    )

    tol = 1e-2 if dtype == jnp.bfloat16 else 1e-4
    np.testing.assert_allclose(
        np.asarray(actual_out[:43].astype(jnp.float32)),
        expected_out[:43],
        rtol=tol,
        atol=tol,
    )
    for slot in (0, 2, 3):
        np.testing.assert_allclose(
            np.asarray(actual_state[slot]),
            expected_state[slot],
            rtol=tol,
            atol=tol,
        )
    np.testing.assert_array_equal(np.asarray(actual_state[1]), conv_state_orig[1])


def test_strided_ldst_load_and_store_roundtrip():
    class _FakeVmemRef:
        def __init__(self, arr: np.ndarray):
            self.arr = arr

        @property
        def ndim(self) -> int:
            return self.arr.ndim

        @property
        def shape(self) -> tuple[int, ...]:
            return self.arr.shape

        @property
        def dtype(self):
            return jnp.dtype(self.arr.dtype)

        def __getitem__(self, idx):
            return jnp.asarray(self.arr[idx])

        def __setitem__(self, idx, val):
            self.arr[idx] = np.asarray(val)

    src = np.arange(32, dtype=np.float32).reshape(4, 8)
    src_ref = _FakeVmemRef(src.copy())
    compact = strided_ldst.load_large_to_compact(src_ref, jnp.float32)
    assert compact.shape == (4, 1, 8)

    dst_ref = _FakeVmemRef(np.zeros_like(src))
    strided_ldst.store_compact_to_large(dst_ref, compact * 2.0)
    np.testing.assert_allclose(dst_ref.arr, src * 2.0)
