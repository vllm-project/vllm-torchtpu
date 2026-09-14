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

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.layers.core.utils import (
    inverse_reorder_for_sharding, reorder_concatenated_tensor_for_sharding,
    slice_sharded_tensor_for_concatenation)


def test_reorder_concatenated_tensor_1d():
    """
    Reordering a 1D concatenated tensor for sharding.
    This simulates splitting an array of elements (e.g., Q, K, V weights)
    so that when we distribute them across `n_shards` devices, each device
    gets a fair slice of Q, K, and V, concatenated together locally.
    """
    # 12 As, 8 Bs, 4 Cs
    a = jnp.full((12, ), 1)
    b = jnp.full((8, ), 2)
    c = jnp.full((4, ), 3)
    concatenated = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    # Each shard should have 3 As (12/4), 2 Bs (8/4), and 1 C (4/4)
    expected_shard = [1, 1, 1, 2, 2, 3]
    expected_full = jnp.array(expected_shard * n_shards)

    np.testing.assert_array_equal(reordered, expected_full)


def test_reorder_concatenated_tensor_2d_dim0():
    """
    Reordering a 2D tensor along dimension 0.
    This simulates when we have a 2D weight matrix concatenated along its row dimension,
    and we want to partition it across multiple chips such that each chip gets
    the right rows from each sub-tensor.
    """
    a = jnp.full((12, 2), 1)
    b = jnp.full((8, 2), 2)
    c = jnp.full((4, 2), 3)
    concatenated = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    # For each shard: 3x2 As, 2x2 Bs, 1x2 Cs -> total 6x2 per shard
    expected_shard = jnp.concatenate(
        [jnp.full((3, 2), 1),
         jnp.full((2, 2), 2),
         jnp.full((1, 2), 3)],
        axis=0)

    # 4 shards stacked together along dim 0
    expected_full = jnp.concatenate([expected_shard] * n_shards, axis=0)

    np.testing.assert_array_equal(reordered, expected_full)


def test_reorder_concatenated_tensor_2d_dim1():
    """
    Reordering a 2D tensor along dimension -1 (or 1).
    This simulates concatenating feature dimensions (like hidden_dim)
    and slicing them across devices.
    """
    a = jnp.full((5, 12), 1)
    b = jnp.full((5, 8), 2)
    c = jnp.full((5, 4), 3)
    concatenated = jnp.concatenate([a, b, c], axis=-1)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=-1)

    expected_shard = jnp.concatenate(
        [jnp.full((5, 3), 1),
         jnp.full((5, 2), 2),
         jnp.full((5, 1), 3)],
        axis=-1)

    expected_full = jnp.concatenate([expected_shard] * n_shards, axis=-1)

    np.testing.assert_array_equal(reordered, expected_full)


def test_inverse_reorder_1d():
    """
    Verifying the inverse reordering works properly for 1D.
    This guarantees that `inverse_reorder_for_sharding` reverses exactly what
    `reorder_concatenated_tensor_for_sharding` did, recovering the original tensor.
    """
    a = jnp.full((12, ), 1)
    b = jnp.full((8, ), 2)
    c = jnp.full((4, ), 3)
    original = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(original,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    recovered = inverse_reorder_for_sharding(reordered,
                                             split_sizes,
                                             n_shards,
                                             dim=0)

    np.testing.assert_array_equal(recovered, original)


def test_inverse_reorder_2d_dim1():
    """
    Verifying the inverse reordering works properly for 2D on last dim.
    Similar to Case 4, this ensures we can gather scattered slices back
    into their original contiguous 2D structure.
    """
    a = jnp.full((5, 12), 1)
    b = jnp.full((5, 8), 2)
    c = jnp.full((5, 4), 3)
    original = jnp.concatenate([a, b, c], axis=-1)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(original,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=-1)

    recovered = inverse_reorder_for_sharding(reordered,
                                             split_sizes,
                                             n_shards,
                                             dim=-1)

    np.testing.assert_array_equal(recovered, original)


def test_slice_sharded_tensor_for_concatenation():
    """
    Slicing an already sharded tensor into individual tensors.
    If a tensor is physically distributed and we want to conceptually split it
    into 3 individual sharded tensors without modifying the sharding axis.
    """
    # Create the sharded tensor structure (e.g., 4 shards concatenated together)
    # Shard pattern: AAABBC, AAABBC, AAABBC, AAABBC
    shard_pattern = [1, 1, 1, 2, 2, 3]
    sharded_tensor = jnp.array(shard_pattern * 4)

    split_sizes = [12, 8, 4]
    n_shards = 4

    sliced_tensors = slice_sharded_tensor_for_concatenation(
        sharded_tensor, split_sizes, n_shards)

    assert len(sliced_tensors) == 3

    # The first tensor should contain all '1's from all 4 shards (4 shards * 3 As = 12 As)
    # and its shape should represent the "global" tensor before physical sharding.
    expected_a = jnp.full((12, ), 1)
    expected_b = jnp.full((8, ), 2)
    expected_c = jnp.full((4, ), 3)

    np.testing.assert_array_equal(sliced_tensors[0], expected_a)
    np.testing.assert_array_equal(sliced_tensors[1], expected_b)
    np.testing.assert_array_equal(sliced_tensors[2], expected_c)


def test_slice_sharded_tensor_for_concatenation_2d():
    """
    Slicing a 2D sharded tensor on its last dim.
    This simulates when we have Q, K, V activations sharded along the hidden dim
    and we want to split them back into Q, K, V activations while retaining the
    distributed semantics.
    """
    n_shards = 4

    # 4 shards, each shard is 5x6
    # In each shard: 3 columns of As, 2 columns of Bs, 1 column of C
    shard_a = jnp.full((5, 3), 1)
    shard_b = jnp.full((5, 2), 2)
    shard_c = jnp.full((5, 1), 3)

    single_shard = jnp.concatenate([shard_a, shard_b, shard_c], axis=-1)

    # The full sharded_tensor contains 4 shards concatenated along the last dimension
    sharded_tensor = jnp.concatenate([single_shard] * n_shards, axis=-1)

    split_sizes = [12, 8, 4]

    sliced_tensors = slice_sharded_tensor_for_concatenation(
        sharded_tensor, split_sizes, n_shards)

    assert len(sliced_tensors) == 3

    # Expected tensors as if they were gathered globally
    expected_a = jnp.full((5, 12), 1)
    expected_b = jnp.full((5, 8), 2)
    expected_c = jnp.full((5, 4), 3)

    np.testing.assert_array_equal(sliced_tensors[0], expected_a)
    np.testing.assert_array_equal(sliced_tensors[1], expected_b)
    np.testing.assert_array_equal(sliced_tensors[2], expected_c)


def test_reorder_concatenated_tensor_1d_8_shards():
    """
    Reordering a 1D tensor across 8 shards.
    """
    # 24 As, 16 Bs, 8 Cs
    a = jnp.full((24, ), 1)
    b = jnp.full((16, ), 2)
    c = jnp.full((8, ), 3)
    concatenated = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [24, 16, 8]
    n_shards = 8

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    # Each shard should have 3 As (24/8), 2 Bs (16/8), and 1 C (8/8)
    expected_shard = [1, 1, 1, 2, 2, 3]
    expected_full = jnp.array(expected_shard * n_shards)

    np.testing.assert_array_equal(reordered, expected_full)


def test_reorder_concatenated_tensor_2d_dim0_2_shards():
    """
    Reordering a 2D tensor along dimension 0 across 2 shards.
    """
    a = jnp.full((12, 2), 1)
    b = jnp.full((8, 2), 2)
    c = jnp.full((4, 2), 3)
    concatenated = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [12, 8, 4]
    n_shards = 2

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    # For each shard: 6x2 As, 4x2 Bs, 2x2 Cs -> total 12x2 per shard
    expected_shard = jnp.concatenate(
        [jnp.full((6, 2), 1),
         jnp.full((4, 2), 2),
         jnp.full((2, 2), 3)],
        axis=0)

    # 2 shards stacked together along dim 0
    expected_full = jnp.concatenate([expected_shard] * n_shards, axis=0)

    np.testing.assert_array_equal(reordered, expected_full)


def test_reorder_concatenated_tensor_1d_64_shards():
    """
    Reordering a 1D tensor across 64 shards (Simulating a large TPU pod).
    """
    # 192 As, 128 Bs, 64 Cs
    a = jnp.full((192, ), 1)
    b = jnp.full((128, ), 2)
    c = jnp.full((64, ), 3)
    concatenated = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [192, 128, 64]
    n_shards = 64

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    # Each shard should have 3 As (192/64), 2 Bs (128/64), and 1 C (64/64)
    expected_shard = [1, 1, 1, 2, 2, 3]
    expected_full = jnp.array(expected_shard * n_shards)

    np.testing.assert_array_equal(reordered, expected_full)


def test_reorder_concatenated_tensor_3d_dim1():
    """
    Reordering a 3D tensor along dimension 1.
    This ensures that old_shape[:dim] and old_shape[dim+1:] are both non-empty.
    """
    # Shapes: (batch, seq, hidden) -> concatenate along seq (dim=1)
    a = jnp.full((2, 12, 4), 1)
    b = jnp.full((2, 8, 4), 2)
    c = jnp.full((2, 4, 4), 3)
    concatenated = jnp.concatenate([a, b, c], axis=1)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(concatenated,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=1)

    # For each shard: 3 As, 2 Bs, 1 Cs along dim 1
    expected_shard = jnp.concatenate([
        jnp.full((2, 3, 4), 1),
        jnp.full((2, 2, 4), 2),
        jnp.full((2, 1, 4), 3)
    ],
                                     axis=1)

    # 4 shards stacked together along dim 1
    expected_full = jnp.concatenate([expected_shard] * n_shards, axis=1)

    np.testing.assert_array_equal(reordered, expected_full)


def test_inverse_reorder_3d_dim1():
    """
    Inverse reordering a 3D tensor along dimension 1.
    """
    a = jnp.full((2, 12, 4), 1)
    b = jnp.full((2, 8, 4), 2)
    c = jnp.full((2, 4, 4), 3)
    original = jnp.concatenate([a, b, c], axis=1)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(original,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=1)

    recovered = inverse_reorder_for_sharding(reordered,
                                             split_sizes,
                                             n_shards,
                                             dim=1)

    np.testing.assert_array_equal(recovered, original)


def test_inverse_reorder_2d_dim0():
    """
    Verifying the inverse reordering works properly for 2D on the first dim (dim=0).
    """
    a = jnp.full((12, 2), 1)
    b = jnp.full((8, 2), 2)
    c = jnp.full((4, 2), 3)
    original = jnp.concatenate([a, b, c], axis=0)

    split_sizes = [12, 8, 4]
    n_shards = 4

    reordered = reorder_concatenated_tensor_for_sharding(original,
                                                         split_sizes,
                                                         n_shards,
                                                         dim=0)

    recovered = inverse_reorder_for_sharding(reordered,
                                             split_sizes,
                                             n_shards,
                                             dim=0)

    np.testing.assert_array_equal(recovered, original)


def test_sharding_utils_indivisible_errors():
    """
    Verifying that assertion errors are raised when split_sizes are not divisible by n_shards.
    """
    split_sizes = [13, 8, 4]  # 13 is not divisible by 4
    n_shards = 4

    # For inverse reorder
    dummy_reordered = jnp.zeros((25, 2))
    with pytest.raises(AssertionError):
        inverse_reorder_for_sharding(dummy_reordered,
                                     split_sizes,
                                     n_shards,
                                     dim=0)

    # For slice sharded tensor
    # The last dimension must be a multiple of n_shards (4) so the initial reshape succeeds,
    # allowing the function to reach the assert split_size % n_shards == 0 check.
    dummy_sharded = jnp.zeros((25, 4))
    with pytest.raises(AssertionError):
        slice_sharded_tensor_for_concatenation(dummy_sharded, split_sizes,
                                               n_shards)

    # For reorder (JAX raises a reshape error, usually TypeError or ValueError because size mismatches)
    a = jnp.full((13, 2), 1)
    b = jnp.full((8, 2), 2)
    c = jnp.full((4, 2), 3)
    concatenated = jnp.concatenate([a, b, c], axis=0)
    with pytest.raises((TypeError, ValueError)):
        reorder_concatenated_tensor_for_sharding(concatenated,
                                                 split_sizes,
                                                 n_shards,
                                                 dim=0)
