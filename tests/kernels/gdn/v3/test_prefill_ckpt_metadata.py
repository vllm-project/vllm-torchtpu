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
"""CPU tests for non-PCP GDN v3 internal prefill checkpoint metadata and tracing."""

import dataclasses
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels import pool_adapters
from vllm_torchtpu.kernels.gdn.v3 import config, metadata, tiling, wrapper

TILE = 64

# Two decodes, then PER_SEQ rows: a fresh prefill checkpointing after its
# first tile, a continuation that names a block with offset 0, a fresh prefill
# checkpointing at its second-to-last tile, a continuation without a block,
# and one padded row whose stale offset and index must be ignored.
QUERY_LENS = [1, 1, 192, 128, 256, 64]
SEQ_LENS = [9, 9, 192, 200, 256, 300, 0, 0]
OFFSETS = [0, 0, 64, 0, 192, 0, 64, 0]
INDICES = [-1, -1, 6, 7, 8, -1, 9, -1]
DISTRIBUTION = [2, 6, 6]
MAX_SEQS = 8


def _cfg(batch_size: int, tile_size: int = TILE) -> config.GDNConfig:
    return config.GDNConfig(
        mode=config.GDNMode.PER_SEQ,
        dtypes=config.Dtypes(
            act_in=jnp.float32.dtype,
            act_out=jnp.float32.dtype,
            compute=jnp.float32.dtype,
            recurrent_state=jnp.float32.dtype,
            conv_state=jnp.float32.dtype,
        ),
        batch_size=batch_size,
        dim_size=1536,
        kernel_size=4,
        tile_size=tile_size,
        num_kq_heads=2,
        num_v_heads=8,
        kq_head_dim=128,
        v_head_dim=128,
    )


def _query_start_loc(query_lens, max_seqs):
    qsl = np.zeros((max_seqs + 1,), dtype=np.int32)
    qsl[1 : len(query_lens) + 1] = np.cumsum(query_lens)
    qsl[len(query_lens) + 1 :] = qsl[len(query_lens)]
    return qsl


def _per_seq_metadata(offsets=OFFSETS, indices=INDICES, with_ckpt=True, tile_size=TILE):
    qsl = _query_start_loc(QUERY_LENS, MAX_SEQS)
    kwargs = {}
    if with_ckpt:
        kwargs = dict(
            prefill_ckpt_offsets=jnp.asarray(offsets, dtype=jnp.int32),
            prefill_ckpt_indices=jnp.asarray(indices, dtype=jnp.int32),
        )
    md = metadata.compute_per_seq_metadata(
        cfg=_cfg(int(qsl[-1]), tile_size),
        seq_lens=jnp.asarray(SEQ_LENS, dtype=jnp.int32),
        query_start_loc=jnp.asarray(qsl),
        state_indices=jnp.arange(1, MAX_SEQS + 1, dtype=jnp.int32),
        read_offsets=jnp.zeros((MAX_SEQS,), dtype=jnp.int32),
        start_seq=jnp.int32(DISTRIBUTION[0]),
        end_seq=jnp.int32(DISTRIBUTION[-1]),
        **kwargs,
    )
    return md, qsl


def _ckpt_tiles(md):
    return [
        p_id
        for p_id in range(int(md.num_tiles))
        if bool(md.is_prefill_ckpt_tile(p_id, 0))
    ]


def test_rows_and_indices_follow_the_start_seq_roll():
    md, qsl = _per_seq_metadata()

    # Sequence slot s is batch row s + start_seq. Row 2 checkpoints after 64
    # of its tokens, row 4 after 192; every other slot has none.
    expected_rows = np.full((MAX_SEQS,), -1, dtype=np.int32)
    expected_rows[0] = qsl[2] + 64
    expected_rows[2] = qsl[4] + 192
    expected_indices = np.full((MAX_SEQS,), -1, dtype=np.int32)
    expected_indices[0] = 6
    expected_indices[2] = 8
    np.testing.assert_array_equal(
        np.asarray(md.s_idx_to_prefill_ckpt_row), expected_rows
    )
    np.testing.assert_array_equal(
        np.asarray(md.s_idx_to_prefill_ckpt_index), expected_indices
    )


def test_exactly_the_tile_ending_at_the_checkpoint_exports():
    md, qsl = _per_seq_metadata()

    # Tiles: row 2 -> p_id 0..2, row 3 -> 3..4, row 4 -> 5..8, row 5 -> 9.
    assert int(md.num_tiles) == 10
    assert _ckpt_tiles(md) == [0, 7]
    for p_id, row, offset in ((0, 2, 64), (7, 4, 192)):
        tile_end = int(md.p_id_to_r_base[p_id, 0]) + int(md.p_id_to_r_size[p_id, 0])
        assert tile_end == qsl[row] + offset
        # The final state still goes out on the sequence's last tile.
        assert not bool(md.p_id_is_last_tile[p_id, 0])


@pytest.mark.parametrize(
    "offset, index",
    [
        (0, 6),  # no checkpoint
        (192, 6),  # the whole query: that is the final state, not a checkpoint
        (256, 6),  # past the query
        (64, -1),  # no destination block
    ],
)
def test_invalid_requests_produce_no_checkpoint(offset, index):
    offsets = list(OFFSETS)
    indices = list(INDICES)
    offsets[2], indices[2] = offset, index
    md, _ = _per_seq_metadata(offsets, indices)

    assert int(md.s_idx_to_prefill_ckpt_row[0]) == -1
    assert int(md.s_idx_to_prefill_ckpt_index[0]) == -1
    assert _ckpt_tiles(md) == [7]


def _tiles(md):
    """(sequence slot, first row, row count, first, last) of each live tile."""
    return [
        (
            int(md.p_id_to_s_idx[p_id, 0]),
            int(md.p_id_to_r_base[p_id, 0]),
            int(md.p_id_to_r_size[p_id, 0]),
            bool(md.p_id_is_first_tile[p_id, 0]),
            bool(md.p_id_is_last_tile[p_id, 0]),
        )
        for p_id in range(int(md.num_tiles))
    ]


def test_offset_off_the_tile_grid_splits_the_tiles_at_the_checkpoint():
    offsets = list(OFFSETS)
    offsets[2] = 100
    md, qsl = _per_seq_metadata(offsets)

    # Row 2 runs [0, 64), [64, 100) up to its checkpoint, then [100, 164)
    # and [164, 192) from it; the other rows keep their tiles.
    row2 = int(qsl[2])
    assert _tiles(md)[:4] == [
        (0, row2, 64, True, False),
        (0, row2 + 64, 36, False, False),
        (0, row2 + 100, 64, False, False),
        (0, row2 + 164, 28, False, True),
    ]
    assert int(md.num_tiles) == 11
    assert _ckpt_tiles(md) == [1, 8]


# Every PER_SEQ tile size the wrapper can run: the dynamic tiling candidates
# and the static default, plus sizes off the power-of-two grid.
_WRAPPER_MIXED_TILE_SIZE = (
    inspect.signature(wrapper.fused_conv1d_gdn).parameters["mixed_tile_size"].default
)
_TILE_SIZES = sorted(set(tiling.MIXED_TILE_SIZES) | {_WRAPPER_MIXED_TILE_SIZE, 48, 96})


@pytest.mark.parametrize("tile_size", _TILE_SIZES)
@pytest.mark.parametrize(
    "row2_offset, row4_offset",
    [(64, 192), (128, 64), (100, 191), (1, 255), (191, 1)],
)
def test_every_tile_size_ends_a_tile_at_the_checkpoint(
    tile_size, row2_offset, row4_offset
):
    """Verifies that exactly one tile ends at each checkpoint row for any tile size."""
    offsets = list(OFFSETS)
    offsets[2], offsets[4] = row2_offset, row4_offset
    md, qsl = _per_seq_metadata(offsets, tile_size=tile_size)
    plain, _ = _per_seq_metadata(with_ckpt=False, tile_size=tile_size)
    tiles = _tiles(md)
    ckpt_rows = np.asarray(md.s_idx_to_prefill_ckpt_row)

    # Each live sequence's tiles cover its query rows in order, none empty
    # or wider than a tile, flagged first and last at the ends.
    num_seqs = DISTRIBUTION[-1] - DISTRIBUTION[0]
    for s_idx in range(num_seqs):
        row = s_idx + DISTRIBUTION[0]
        seq_tiles = [t for t in tiles if t[0] == s_idx]
        starts = [t[1] for t in seq_tiles]
        ends = [t[1] + t[2] for t in seq_tiles]
        assert starts[0] == qsl[row] and ends[-1] == qsl[row + 1]
        assert starts[1:] == ends[:-1]
        assert all(1 <= t[2] <= tile_size for t in seq_tiles)
        assert [t[3] for t in seq_tiles] == [True] + [False] * (len(seq_tiles) - 1)
        assert [t[4] for t in seq_tiles] == [False] * (len(seq_tiles) - 1) + [True]
    assert [t[0] for t in tiles] == sorted(t[0] for t in tiles)
    assert len(tiles) <= md.p_id_to_s_idx.data.size

    # Exactly one tile per checkpoint ends at its row, and it is not the
    # sequence's last tile, which writes the final state.
    expected_rows = {0: qsl[2] + row2_offset, 2: qsl[4] + row4_offset}
    assert sorted(ckpt_rows[ckpt_rows >= 0].tolist()) == sorted(expected_rows.values())
    exporting = [tiles[p_id] for p_id in _ckpt_tiles(md)]
    assert sorted((t[0], t[1] + t[2]) for t in exporting) == sorted(
        expected_rows.items()
    )
    assert not any(t[4] for t in exporting)

    # An offset on the tile grid keeps the tiles of the run without a
    # checkpoint.
    if row2_offset % tile_size == 0 and row4_offset % tile_size == 0:
        assert tiles == _tiles(plain)


def test_without_checkpoints_the_metadata_is_unchanged():
    plain, _ = _per_seq_metadata(with_ckpt=False)
    with_ckpt, _ = _per_seq_metadata()

    assert plain.s_idx_to_prefill_ckpt_row is None
    assert plain.s_idx_to_prefill_ckpt_index is None
    # Leaf count drives the pallas_call operand and alias indices.
    assert len(with_ckpt) == len(plain) + 2
    for field in dataclasses.fields(plain):
        if field.name.startswith("s_idx_to_prefill_ckpt"):
            continue
        a, b = getattr(plain, field.name), getattr(with_ckpt, field.name)
        for leaf_a, leaf_b in zip(jax.tree.leaves(a), jax.tree.leaves(b)):
            np.testing.assert_array_equal(np.asarray(leaf_a), np.asarray(leaf_b))


def test_offsets_and_indices_go_together():
    qsl = _query_start_loc(QUERY_LENS, MAX_SEQS)
    with pytest.raises(AssertionError):
        metadata.compute_per_seq_metadata(
            cfg=_cfg(int(qsl[-1])),
            seq_lens=jnp.asarray(SEQ_LENS, dtype=jnp.int32),
            query_start_loc=jnp.asarray(qsl),
            state_indices=jnp.arange(MAX_SEQS, dtype=jnp.int32),
            read_offsets=jnp.zeros((MAX_SEQS,), dtype=jnp.int32),
            start_seq=jnp.int32(0),
            end_seq=jnp.int32(6),
            prefill_ckpt_offsets=jnp.asarray(OFFSETS, dtype=jnp.int32),
        )


# --------------------------------------------------------------------
# Kernel entry point, traced on CPU against a static tpu7x description.
# --------------------------------------------------------------------

N_KQ, N_V, D_K, D_V = 2, 8, 128, 128
KERNEL_SIZE = 4
DIM = N_KQ * D_K * 2 + N_V * D_V
SPLIT, KBS, LANES = 3, 256, 256
PAYLOAD = (1, 2)
SSM_NTOK = N_V * D_K * D_V * 4 // (2 * LANES * 2)
NUM_MGR = 12
NUM_TOKENS = sum(QUERY_LENS)


@pytest.fixture
def static_tpu_info(monkeypatch):
    # The wrapper sizes lanes and VMEM from the chip; nothing here runs on it.
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


def _kernel_kwargs():
    num_seqs = len(QUERY_LENS)
    return dict(
        qkv=jnp.zeros((NUM_TOKENS, DIM), jnp.float32),
        b=jnp.zeros((NUM_TOKENS, N_V), jnp.float32),
        a=jnp.zeros((NUM_TOKENS, N_V), jnp.float32),
        conv_weight=jnp.zeros((DIM, 1, KERNEL_SIZE), jnp.float32),
        conv_bias=jnp.zeros((DIM,), jnp.float32),
        a_log=jnp.zeros((N_V,), jnp.float32),
        dt_bias=jnp.zeros((N_V,), jnp.float32),
        query_start_loc=jnp.asarray(_query_start_loc(QUERY_LENS, num_seqs)),
        # Distinct from the checkpoint blocks named in INDICES.
        state_indices=jnp.asarray([1, 2, 3, 4, 5, 10], dtype=jnp.int32),
        distribution=jnp.asarray(DISTRIBUTION, dtype=jnp.int32),
        seq_lens=jnp.asarray(SEQ_LENS[:num_seqs], dtype=jnp.int32),
        n_kq=N_KQ,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )


def _ckpt_kwargs():
    num_seqs = len(QUERY_LENS)
    return dict(
        prefill_ckpt_offsets=jnp.asarray(OFFSETS[:num_seqs], dtype=jnp.int32),
        prefill_ckpt_indices=jnp.asarray(INDICES[:num_seqs], dtype=jnp.int32),
    )


def _pooled_kwargs(whole_block_dma):
    pool = jnp.zeros((NUM_MGR * SPLIT, KBS) + PAYLOAD + (LANES,), jnp.bfloat16)
    plan = pool_adapters.v3_state_source(
        pool,
        split=SPLIT,
        ssm_ntok=SSM_NTOK,
        conv_tok0=SSM_NTOK,
        conv_ntok=16,
        conv_dim=DIM,
        n_v=N_V,
        d_k=D_K,
        d_v=D_V,
        kernel_size=KERNEL_SIZE,
    )
    plan = dataclasses.replace(plan, whole_block_dma=whole_block_dma)
    return dict(
        conv_state=None,
        recurrent_state=None,
        state_source=pool,
        state_plan=plan,
    )


@pytest.mark.usefixtures("static_tpu_info")
@pytest.mark.parametrize("whole_block_dma", [False, True], ids=["rows", "whole"])
def test_export_adds_one_dma_per_state_region(whole_block_dma):
    def trace(**extra):
        return str(
            jax.make_jaxpr(
                lambda: wrapper.fused_conv1d_gdn(
                    **_kernel_kwargs(), **_pooled_kwargs(whole_block_dma), **extra
                )
            )()
        )

    plain = trace()
    with_ckpt = trace(**_ckpt_kwargs())

    # One conv and one recurrent region copy, each waited inline; the
    # pipelined state copies are untouched.
    assert with_ckpt.count("dma_start") == plain.count("dma_start") + 2
    assert with_ckpt.count("dma_wait") == plain.count("dma_wait") + 2


def test_dense_path_rejects_checkpoints():
    num_seqs = len(QUERY_LENS)
    with pytest.raises(AssertionError, match="state source"):
        wrapper.fused_conv1d_gdn(
            conv_state=jnp.zeros((num_seqs + 1, KERNEL_SIZE - 1, DIM), jnp.float32),
            recurrent_state=jnp.zeros((num_seqs + 1, N_V, D_K, D_V), jnp.float32),
            **_kernel_kwargs(),
            **_ckpt_kwargs(),
        )


def test_pooled_path_needs_both_arrays():
    with pytest.raises(AssertionError, match="go together"):
        wrapper.fused_conv1d_gdn(
            **_kernel_kwargs(),
            **_pooled_kwargs(whole_block_dma=False),
            prefill_ckpt_offsets=_ckpt_kwargs()["prefill_ckpt_offsets"],
        )
