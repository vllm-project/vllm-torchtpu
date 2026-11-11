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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.gdn.v3.metadata import compute_batched_seq_metadata


@pytest.mark.parametrize(
    "capacity,tile,active",
    [(1, 2, 1), (3, 2, 3), (3, 2, 1), (3, 2, 0), (5, 4, 5), (8, 4, 3)],
)
def test_batched_metadata_pads_inactive_tile_lanes(capacity, tile, active):
    cfg = SimpleNamespace(tile_size=tile, seq_tile_size=tile)
    lengths = jnp.arange(capacity, dtype=jnp.int32) % 2 + 1
    starts = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(lengths)])
    states = jnp.arange(capacity, dtype=jnp.int32) * 3 + 1
    offsets = jnp.ones(capacity, jnp.int32)
    checkpoints = jnp.stack([states, states + 1], axis=1)
    meta = compute_batched_seq_metadata(
        cfg,
        lengths + 16,
        starts,
        states,
        offsets,
        jnp.asarray(active, jnp.int32),
        checkpoints,
    )
    size = ((capacity + tile - 1) // tile) * tile
    expected_ids = np.concatenate(
        [np.arange(active), np.zeros(size - active, dtype=np.int32)]
    )
    np.testing.assert_array_equal(np.asarray(meta.p_id_to_s_idx.data), expected_ids)
    np.testing.assert_array_equal(
        np.asarray(meta.p_id_to_r_base.data), np.asarray(starts)[expected_ids]
    )
    np.testing.assert_array_equal(
        np.asarray(meta.p_id_to_r_size.data),
        np.concatenate([np.asarray(lengths)[:active], np.zeros(size - active)]),
    )
    for predicate in (meta.p_id_is_first_tile, meta.p_id_is_last_tile):
        np.testing.assert_array_equal(
            np.asarray(predicate.data), np.arange(size) < active
        )
    assert int(meta.num_tiles) == (active + tile - 1) // tile
    # Padding program lanes must not change request allocation or read slots.
    np.testing.assert_array_equal(
        np.asarray(meta.s_idx_to_state_indices), np.asarray(states)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.s_idx_to_read_offset), np.asarray(offsets)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.s_idx_to_ckpt_indices), np.asarray(checkpoints)
    )
    np.testing.assert_array_equal(
        np.asarray(meta.s_idx_has_initial_state), np.ones(capacity, bool)
    )
