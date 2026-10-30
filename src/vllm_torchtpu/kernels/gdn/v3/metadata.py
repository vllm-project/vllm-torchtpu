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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref


def compute_batched_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    end_seq: jax.Array,
    ckpt_indices: jax.Array | None = None,
) -> memory_ref.MetadataRef:
    """Metadata for computing multiple sequences per tile.

    A sequence contributes exactly one tile holding all of its query tokens:
    a single decoded token, or a speculative verify window of up to
    `cfg.window_size` of them. The initial state is read from
    `state_indices[s] + read_offsets[s]` and one state checkpoint per window
    position is written back to `state_indices[s] + t`.
    """

    max_seqs = seq_lens.size
    all_seqs = jnp.arange(max_seqs)

    # NOTE: Only supports use case where query_lens[i] <= cfg.window_size where
    # i < end_seq. This must be guaranteed by the function caller.
    # TODO(kyuyeunk): Add error handling when above condition is not met.
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    is_valid_seqs = jnp.where(all_seqs < end_seq, True, False)
    has_initial_state = (seq_lens - query_lens) > 0
    all_valid_seqs = jnp.where(is_valid_seqs, all_seqs, 0)

    return memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=pl.cdiv(end_seq, cfg.tile_size),
        p_id_to_s_idx=all_valid_seqs,
        p_id_to_r_base=query_start_loc[all_valid_seqs],
        p_id_to_r_size=jnp.where(is_valid_seqs, query_lens, 0),
        p_id_is_first_tile=is_valid_seqs,
        p_id_is_last_tile=is_valid_seqs,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
        s_idx_to_read_offset=read_offsets,
        s_idx_to_ckpt_indices=ckpt_indices,
    )


def compute_per_seq_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
    ckpt_indices: jax.Array | None = None,
) -> memory_ref.MetadataRef:
    """Metadata for computing single sequence per tile.

    `read_offsets` selects the checkpoint a resuming sequence's initial state
    is read from. It is 0 in the common cases (a fresh or chunked prefill
    resumes from the single state its previous chunk wrote), but a
    prefix-cache resume on the unified pool may land on a boundary state
    block whose committed state is a non-zero checkpoint from a verify step.
    The final state is always written to checkpoint 0.
    """
    max_seqs = seq_lens.size
    max_tokens = cfg.batch_size
    all_seqs = jnp.arange(max_seqs)
    all_tokens = jnp.arange(max_tokens)

    # Shift to ensure first element is for start_seq.
    query_start_loc = jnp.roll(query_start_loc, shift=-start_seq)
    seq_lens = jnp.roll(seq_lens, shift=-start_seq)
    state_indices = jnp.roll(state_indices, shift=-start_seq)
    read_offsets = jnp.roll(read_offsets, shift=-start_seq)
    if ckpt_indices is not None:
        # Sequence-major axis only: the checkpoint axis must stay put or the
        # read offset would select another slot's block.
        ckpt_indices = jnp.roll(ckpt_indices, shift=-start_seq, axis=0)

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # NOTE: query_lens is used for calculating num_tiles. Defensive programming
    # that masks out all the other values (seq_lens, state_indices) are not needed
    # since they will not be visited as long as num_tiles is correct.
    num_seqs = end_seq - start_seq
    query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

    # Calculate number of tiles needed for each sequence.
    s_idx_to_num_tiles = pl.cdiv(query_lens, cfg.chunk_size)
    # Calculate starting p_id of each sequence.
    s_idx_to_start_p_id = jnp.cumulative_sum(s_idx_to_num_tiles, include_initial=True)
    # Map tile index to seq index.
    # Consider following case:
    # all_seqs = [0 1 2 3 4]
    # s_idx_to_num_tiles = [1 2 3 0 1]
    # jnp.repeat will return following results:
    # p_id_to_s_idx = [0 1 1 2 2 2 4]
    # This means p_id_to_s_idx[i] will point to its corresponding seq index.

    # NOTE: To make jnp.repeat jit compilable, we add total_repeat_length. This
    # introduces padding to p_id_to_s_idx[i] where i >= num_tiles. Since the
    # kernel only checks value up-to p_id_to_s_idx[num_tiles-1], padded value
    # will not impact kernel execution.
    p_id_to_s_idx = jnp.repeat(
        all_seqs, s_idx_to_num_tiles, total_repeat_length=max_tokens
    )
    # Map program id (p_id) to tile id of a sequence.
    p_id_to_t_id = all_tokens - s_idx_to_start_p_id[p_id_to_s_idx]
    # Map tile index to starting row of its activation.
    p_id_to_r_base = query_start_loc[p_id_to_s_idx] + p_id_to_t_id * cfg.chunk_size
    # Calculate number of rows to calculate / fetch for each tile.
    p_id_to_r_size = jnp.minimum(
        query_start_loc[p_id_to_s_idx + 1] - p_id_to_r_base,
        cfg.tile_size,
    )

    # Calculate predicate used for state DMA. State is read if program id (p_id)
    # is the first tile of a sequence and the sequence had been computed before
    # (chunked prefill, decode, etc). State is written if the program id is the
    # last tile of a sequence.
    has_initial_state = (seq_lens - query_lens) > 0
    p_id_is_first_tile = p_id_to_t_id == 0
    p_id_is_last_tile = p_id_to_t_id == (s_idx_to_num_tiles[p_id_to_s_idx] - 1)

    # NOTE: Since query_lens[i] = 0 where i >= num_seqs, s_idx_to_num_tiles[i]
    # where i >= num_seqs will also be 0. Therefore, s_idx_to_num_tiles.sum()
    # will contain number of tiles for valid sequence.
    num_tiles = s_idx_to_num_tiles.sum()

    return memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=num_tiles,
        p_id_to_s_idx=p_id_to_s_idx,
        p_id_to_r_base=p_id_to_r_base,
        p_id_to_r_size=p_id_to_r_size,
        p_id_is_first_tile=p_id_is_first_tile,
        p_id_is_last_tile=p_id_is_last_tile,
        s_idx_has_initial_state=has_initial_state,
        s_idx_to_state_indices=state_indices,
        s_idx_to_read_offset=read_offsets,
        s_idx_to_ckpt_indices=ckpt_indices,
    )


def compute_per_seq_segment_metadata(
    cfg: config.GDNConfig,
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    state_indices: jax.Array,
    read_offsets: jax.Array,
    start_seq: jax.Array,
    end_seq: jax.Array,
    seg_bounds: jax.Array,
    seg_state_indices: jax.Array,
    ckpt_indices: jax.Array | None = None,
) -> memory_ref.MetadataRef:
    """`compute_per_seq_metadata` with up to two in-chunk state checkpoints.

    Sequence `s` is split at the query offsets `seg_bounds[s] = (c1, c2)`
    (0, or outside `(0, query_len)`, means no cut) into three "virtual
    sequences" `[0, c1)`, `[c1, c2)`, `[c2, query_len)`. Each tiles from its
    own start, exactly like a separate prefill chunk would, and its last tile
    stores the state through the existing last-tile path: segment k < 2 to
    `seg_state_indices[s, k]`, the final segment to the sequence's own state
    slot (checkpoint 0). Only the first non-empty segment reads the initial
    state (from the sequence's usual read location); later segments continue
    from the inter-tile fp32 carry, so no state round-trips through HBM.

    Read and write locations differ for a non-final first segment, so they
    are encoded per virtual sequence:
      * checkpoint mode (`ckpt_indices` given): row `[write_block,
        read_block, ...]` with read offset 1 (row width >= 2);
      * dense mode: `state_indices = write_slot` and `read_offset =
        read_slot - write_slot` (the kernel reads `state + offset`).
    """
    max_seqs = seq_lens.size
    num_virtual = 3 * max_seqs
    max_tiles = min(
        cfg.batch_size,
        pl.cdiv(cfg.batch_size, cfg.chunk_size) + num_virtual,
    )

    query_start_loc = jnp.roll(query_start_loc, shift=-start_seq)
    seq_lens = jnp.roll(seq_lens, shift=-start_seq)
    state_indices = jnp.roll(state_indices, shift=-start_seq)
    read_offsets = jnp.roll(read_offsets, shift=-start_seq)
    seg_bounds = jnp.roll(seg_bounds.astype(jnp.int32), shift=-start_seq, axis=0)
    seg_state_indices = jnp.roll(
        seg_state_indices.astype(jnp.int32), shift=-start_seq, axis=0
    )
    if ckpt_indices is not None:
        ckpt_indices = jnp.roll(ckpt_indices, shift=-start_seq, axis=0)

    num_seqs = end_seq - start_seq
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    query_lens = jnp.where(jnp.arange(max_seqs) < num_seqs, query_lens, 0)
    has_initial_state = (seq_lens - query_lens) > 0

    # Normalized cut points: 0 <= c1 <= c2 <= query_len.
    e1, e2 = seg_bounds[:, 0], seg_bounds[:, 1]
    c1 = jnp.where((e1 > 0) & (e1 < query_lens), e1, 0)
    c2 = jnp.where((e2 > c1) & (e2 < query_lens), e2, c1)
    zeros = jnp.zeros_like(c1)
    seg_starts = jnp.stack([zeros, c1, c2], axis=1)
    seg_ends = jnp.stack([c1, c2, query_lens], axis=1)
    v_query_lens = (seg_ends - seg_starts).reshape(-1)
    base = query_start_loc[:-1, None]
    v_row_start = (base + seg_starts).reshape(-1)
    v_row_end = (base + seg_ends).reshape(-1)

    nonempty = (seg_ends - seg_starts) > 0
    first_seg = jnp.argmax(nonempty, axis=1)
    v_is_first_seg = (jnp.arange(3)[None, :] == first_seg[:, None]).reshape(-1)
    v_has_initial_state = jnp.repeat(has_initial_state, 3)

    if ckpt_indices is not None:
        width = max(ckpt_indices.shape[1], 2)
        ro = jnp.clip(read_offsets, 0, ckpt_indices.shape[1] - 1)
        read_block = jnp.take_along_axis(ckpt_indices, ro[:, None], axis=1)[:, 0]
        write_block = jnp.stack(
            [seg_state_indices[:, 0], seg_state_indices[:, 1], ckpt_indices[:, 0]],
            axis=1,
        ).reshape(-1)
        v_read_block = jnp.repeat(read_block, 3)
        v_ckpt_indices = jnp.concatenate(
            [
                write_block[:, None],
                jnp.broadcast_to(v_read_block[:, None], (num_virtual, width - 1)),
            ],
            axis=1,
        ).astype(jnp.int32)
        v_state_indices = write_block.astype(jnp.int32)
        v_read_offsets = jnp.ones((num_virtual,), jnp.int32)
    else:
        v_ckpt_indices = None
        write_slot = jnp.stack(
            [seg_state_indices[:, 0], seg_state_indices[:, 1], state_indices],
            axis=1,
        ).reshape(-1)
        v_read_slot = jnp.repeat(state_indices + read_offsets, 3)
        v_state_indices = write_slot.astype(jnp.int32)
        v_read_offsets = (v_read_slot - write_slot).astype(jnp.int32)

    all_virtual = jnp.arange(num_virtual)
    all_tiles = jnp.arange(max_tiles)
    v_num_tiles = pl.cdiv(v_query_lens, cfg.chunk_size)
    v_start_p_id = jnp.cumulative_sum(v_num_tiles, include_initial=True)
    p_id_to_s_idx = jnp.repeat(all_virtual, v_num_tiles, total_repeat_length=max_tiles)
    p_id_to_t_id = all_tiles - v_start_p_id[p_id_to_s_idx]
    p_id_to_r_base = v_row_start[p_id_to_s_idx] + p_id_to_t_id * cfg.chunk_size
    p_id_to_r_size = jnp.minimum(
        v_row_end[p_id_to_s_idx] - p_id_to_r_base, cfg.tile_size
    )
    p_id_is_first_tile = (p_id_to_t_id == 0) & v_is_first_seg[p_id_to_s_idx]
    p_id_is_last_tile = p_id_to_t_id == (v_num_tiles[p_id_to_s_idx] - 1)

    return memory_ref.MetadataRef.create(
        cfgs=cfg,
        num_tiles=v_num_tiles.sum(),
        p_id_to_s_idx=p_id_to_s_idx,
        p_id_to_r_base=p_id_to_r_base,
        p_id_to_r_size=p_id_to_r_size,
        p_id_is_first_tile=p_id_is_first_tile,
        p_id_is_last_tile=p_id_is_last_tile,
        s_idx_has_initial_state=v_has_initial_state,
        s_idx_to_state_indices=v_state_indices,
        s_idx_to_read_offset=v_read_offsets,
        s_idx_to_ckpt_indices=v_ckpt_indices,
    )
