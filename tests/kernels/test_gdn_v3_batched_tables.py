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
"""The GDN v3 batched pass reads its routing tables only in bounds.

A batched tile reads `seq_tile_size` (the decode tile) table entries, so
when the slot count is not a multiple of the tile, or is below it, the last
tile reads past the last sequence. Both table builders pad to whole tiles
with empty entries (no rows, no state copy). The kernel test runs the dense
entry in the Pallas TPU interpreter with out-of-bounds reads raising.
"""
import functools
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.pallas.mosaic import tpu_info
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import (config, metadata, tables_in_kernel,
                                          wrapper)

pytestmark = pytest.mark.cpu_test

N_KQ, N_V, D_K, D_V, KERNEL_SIZE = 2, 8, 128, 128, 4
DIM = N_KQ * D_K * 2 + N_V * D_V

# (slots, live decodes, decode tile): the last tile is partial...
PARTIAL_LAST_TILE = [(2, 2, 4), (3, 2, 4), (6, 5, 4), (6, 6, 4), (20, 17, 8)]
# ...or every tile is whole.
WHOLE_TILES = [(8, 8, 4), (8, 5, 4)]


def _batched_cfg(tile):
    f32 = jnp.float32.dtype
    return config.GDNConfig(mode=config.GDNMode.BATCHED,
                            dtypes=config.Dtypes(act_in=f32,
                                                 act_out=f32,
                                                 compute=f32,
                                                 recurrent_state=f32,
                                                 conv_state=f32),
                            batch_size=8,
                            dim_size=DIM,
                            kernel_size=KERNEL_SIZE,
                            tile_size=tile,
                            num_kq_heads=N_KQ,
                            num_v_heads=N_V,
                            kq_head_dim=D_K,
                            v_head_dim=D_V)


def _decode_step(num_slots, num_decodes):
    q_lens = [1] * num_decodes + [0] * (num_slots - num_decodes)
    query_start_loc = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens)]),
                                  jnp.int32)
    seq_lens = jnp.asarray([5] * num_decodes + [0] * (num_slots - num_decodes),
                           jnp.int32)
    return query_start_loc, seq_lens


@pytest.mark.parametrize("num_slots, num_decodes, tile",
                         PARTIAL_LAST_TILE + WHOLE_TILES)
def test_batched_tables_cover_every_tile(num_slots, num_decodes, tile):
    query_start_loc, seq_lens = _decode_step(num_slots, num_decodes)
    meta = metadata.compute_batched_seq_metadata(
        cfg=_batched_cfg(tile),
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        state_indices=jnp.arange(num_slots, dtype=jnp.int32),
        read_offsets=jnp.zeros((num_slots, ), jnp.int32),
        end_seq=jnp.int32(num_decodes))
    num_read = int(meta.num_tiles) * tile
    tables = (meta.p_id_to_s_idx, meta.p_id_to_r_base, meta.p_id_to_r_size,
              meta.p_id_is_first_tile, meta.p_id_is_last_tile)
    for table in tables:
        assert table.data.shape[0] >= num_read, (table.data.shape, num_read)
    # Live sequences keep their entries; every other entry is empty.
    r_size = np.asarray(meta.p_id_to_r_size.data)
    np.testing.assert_array_equal(r_size[:num_decodes], 1)
    for flags in (r_size, np.asarray(meta.p_id_is_first_tile.data),
                  np.asarray(meta.p_id_is_last_tile.data)):
        assert not flags[num_decodes:].any()
    assert np.asarray(meta.p_id_is_first_tile.data)[:num_decodes].all()


@pytest.mark.parametrize("num_slots, num_decodes, tile",
                         PARTIAL_LAST_TILE + WHOLE_TILES)
def test_in_kernel_tables_cover_every_tile(num_slots, num_decodes, tile):
    shapes = tables_in_kernel.scratch_shapes(num_slots, 8, tile)
    num_read = pl.cdiv(num_decodes, tile) * tile
    for name in tables_in_kernel.TABLE_NAMES:
        assert shapes[f"tb_{name}"].shape[0] >= num_read, name


@pytest.fixture
def interpret_on_cpu(monkeypatch):
    """Run the kernels in the Pallas TPU interpreter, out-of-bounds reads
    raising, with v6e's TPU info for the CPU device."""

    def v6e():
        return tpu_info.get_tpu_info_for_chip(tpu_info.ChipVersion.TPU_V6E, 1)

    params = pltpu.InterpretParams(detect_races=True,
                                   uninitialized_memory="nan",
                                   out_of_bounds_reads="raise")
    monkeypatch.setattr(pl, "pallas_call",
                        functools.partial(pl.pallas_call, interpret=params))
    tpu_info.get_tpu_info.cache_clear()
    with mock.patch.dict(tpu_info.registry, {"cpu": v6e}):
        yield
    tpu_info.get_tpu_info.cache_clear()


def _run_dense_decode(num_slots, num_decodes, num_rows=8):
    """`num_decodes` decodes over `num_slots` slots (the rest empty) in a
    step of `num_rows` token rows; the decode tile is 4."""
    keys = iter(jax.random.split(jax.random.key(0), 9))
    query_start_loc, seq_lens = _decode_step(num_slots, num_decodes)
    (conv, recurrent), out = wrapper.fused_conv1d_gdn(
        qkv=jax.random.normal(next(keys), (num_rows, DIM)),
        b=jax.random.normal(next(keys), (num_rows, N_V)),
        a=jax.random.normal(next(keys), (num_rows, N_V)),
        conv_state=jax.random.normal(next(keys), (9, KERNEL_SIZE - 1, DIM)),
        recurrent_state=jax.random.normal(next(keys), (9, N_V, D_K, D_V)),
        conv_weight=jax.random.normal(next(keys), (DIM, 1, KERNEL_SIZE)),
        conv_bias=jax.random.normal(next(keys), (DIM, )),
        a_log=jax.random.normal(next(keys), (N_V, )),
        dt_bias=jax.random.normal(next(keys), (N_V, )),
        query_start_loc=query_start_loc,
        state_indices=jnp.arange(1, num_slots + 1, dtype=jnp.int32),
        distribution=jnp.full((3, ), num_decodes, jnp.int32),
        seq_lens=seq_lens,
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
        decode_tile_size=4)
    return np.asarray(out)[:num_decodes], np.asarray(conv), np.asarray(
        recurrent)


@pytest.mark.parametrize("num_slots, num_decodes", [(2, 2), (6, 5)])
def test_dense_partial_last_tile_matches_whole_tiles(interpret_on_cpu,
                                                     num_slots, num_decodes):
    """Fewer slots than a tile (2) or not a multiple of it (6): the step
    runs in bounds and computes what the same decodes over 8 slots do."""
    got = _run_dense_decode(num_slots, num_decodes)
    want = _run_dense_decode(8, num_decodes)
    for g, w in zip(got, want):
        np.testing.assert_array_equal(g, w)
