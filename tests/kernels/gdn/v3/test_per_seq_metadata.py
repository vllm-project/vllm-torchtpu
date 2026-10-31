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
"""CPU checks for the per-sequence (prefill) tile metadata."""

import jax.numpy as jnp
import numpy as np

from vllm_torchtpu.kernels.gdn.v3 import config, metadata


def _cfg(batch_size: int, tile_size: int = 64) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.bfloat16.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=384,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=1,
        num_v_heads=1,
        kq_head_dim=128,
        v_head_dim=128,
    )


def test_tile_metadata_is_sized_by_tiles_not_tokens():
    # The kernel copies these arrays to SMEM on every call: one entry per tile
    # the batch can hold (full tiles plus one partial tile per sequence).
    max_num_seqs, tile = 8, 64
    lengths = np.array([100, 64, 4000, 1], np.int32)
    batch_size = 8192
    query_start = np.zeros(max_num_seqs + 1, np.int32)
    query_start[1 : lengths.size + 1] = np.cumsum(lengths)
    query_start[lengths.size + 1 :] = query_start[lengths.size]
    seq_lens = np.zeros(max_num_seqs, np.int32)
    seq_lens[: lengths.size] = lengths
    meta = metadata.compute_per_seq_metadata(
        cfg=_cfg(batch_size, tile),
        seq_lens=jnp.asarray(seq_lens),
        query_start_loc=jnp.asarray(query_start),
        state_indices=jnp.arange(max_num_seqs, dtype=jnp.int32),
        read_offsets=jnp.zeros(max_num_seqs, jnp.int32),
        start_seq=jnp.int32(0),
        end_seq=jnp.int32(lengths.size),
    )

    assert meta.p_id_to_s_idx.data.shape == (batch_size // tile + max_num_seqs,)
    tiles = -(-lengths // tile)
    num_tiles = int(meta.num_tiles)
    assert num_tiles == tiles.sum()
    seq = np.repeat(np.arange(lengths.size), tiles)
    start = np.concatenate(([0], np.cumsum(tiles)[:-1]))
    tile_in_seq = np.arange(num_tiles) - start[seq]
    base = query_start[seq] + tile_in_seq * tile
    np.testing.assert_array_equal(meta.p_id_to_s_idx.data[:num_tiles], seq)
    np.testing.assert_array_equal(meta.p_id_to_r_base.data[:num_tiles], base)
    np.testing.assert_array_equal(
        meta.p_id_to_r_size.data[:num_tiles],
        np.minimum(query_start[seq + 1] - base, tile),
    )
    np.testing.assert_array_equal(
        meta.p_id_is_last_tile.data[:num_tiles], tile_in_seq == tiles[seq] - 1
    )
