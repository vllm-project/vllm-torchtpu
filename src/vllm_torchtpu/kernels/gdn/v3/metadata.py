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


def prefill_ckpt_offset_is_valid(
    offsets: jax.Array, query_lens: jax.Array
) -> jax.Array:
    """Returns whether offsets lies strictly within (0, query_lens)."""
    return jnp.logical_and(offsets > 0, offsets < query_lens)


def mask_prefill_ckpt_requests(
    offsets: jax.Array, indices: jax.Array, query_lens: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Normalizes invalid or unreserved prefill checkpoint entries to (0, -1)."""
    has_ckpt = jnp.logical_and(
        prefill_ckpt_offset_is_valid(offsets, query_lens), indices >= 0
    )
    return jnp.where(has_ckpt, offsets, 0), jnp.where(has_ckpt, indices, -1)


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
    prefill_ckpt_offsets: jax.Array | None = None,
    prefill_ckpt_indices: jax.Array | None = None,
) -> memory_ref.MetadataRef:
    """Metadata for computing single sequence per tile.

    `read_offsets` selects the checkpoint a resuming sequence's initial state
    is read from. It is 0 in the common cases (a fresh or chunked prefill
    resumes from the single state its previous chunk wrote), but a
    prefix-cache resume on the unified pool may land on a boundary state
    block whose committed state is a non-zero checkpoint from a verify step.
    The final state is always written to checkpoint 0.

    When prefill_ckpt_offsets and prefill_ckpt_indices are provided, sequences
    with a valid checkpoint split their tile grid at R = query_start + offset
    into [query_start, R) and [R, query_end), ensuring row R always lands on a
    tile boundary regardless of chunk_size.
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
    has_prefill_ckpt_input = prefill_ckpt_offsets is not None
    assert has_prefill_ckpt_input == (prefill_ckpt_indices is not None)
    if prefill_ckpt_offsets is not None:
        assert prefill_ckpt_indices is not None
        prefill_ckpt_offsets = jnp.roll(prefill_ckpt_offsets, shift=-start_seq)
        prefill_ckpt_indices = jnp.roll(prefill_ckpt_indices, shift=-start_seq)

    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    # NOTE: query_lens is used for calculating num_tiles. Defensive programming
    # that masks out all the other values (seq_lens, state_indices) are not needed
    # since they will not be visited as long as num_tiles is correct.
    num_seqs = end_seq - start_seq
    query_lens = jnp.where(all_seqs < num_seqs, query_lens, 0)

    s_idx_to_prefill_ckpt_row = s_idx_to_prefill_ckpt_index = None
    ckpt_split_row = s_idx_to_num_pre_ckpt_tiles = None
    if prefill_ckpt_offsets is not None:
        assert prefill_ckpt_indices is not None
        # query_lens is 0 past num_seqs, so padded slots get no checkpoint.
        prefill_ckpt_offsets, s_idx_to_prefill_ckpt_index = mask_prefill_ckpt_requests(
            prefill_ckpt_offsets, prefill_ckpt_indices, query_lens
        )
        ckpt_split_row = query_start_loc[:-1] + prefill_ckpt_offsets
        s_idx_to_prefill_ckpt_row = jnp.where(
            prefill_ckpt_offsets > 0, ckpt_split_row, -1
        )

    # Calculate number of tiles needed for each sequence.
    if prefill_ckpt_offsets is not None:
        assert prefill_ckpt_indices is not None
        s_idx_to_num_pre_ckpt_tiles = pl.cdiv(prefill_ckpt_offsets, cfg.chunk_size)
        s_idx_to_num_tiles = s_idx_to_num_pre_ckpt_tiles + pl.cdiv(
            query_lens - prefill_ckpt_offsets, cfg.chunk_size
        )
    else:
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
    if ckpt_split_row is not None and s_idx_to_num_pre_ckpt_tiles is not None:
        num_pre_ckpt_tiles = s_idx_to_num_pre_ckpt_tiles[p_id_to_s_idx]
        is_pre_ckpt = p_id_to_t_id < num_pre_ckpt_tiles
        segment_start = jnp.where(
            is_pre_ckpt,
            query_start_loc[p_id_to_s_idx],
            ckpt_split_row[p_id_to_s_idx],
        )
        segment_end = jnp.where(
            is_pre_ckpt,
            ckpt_split_row[p_id_to_s_idx],
            query_start_loc[p_id_to_s_idx + 1],
        )
        segment_t_id = jnp.where(
            is_pre_ckpt, p_id_to_t_id, p_id_to_t_id - num_pre_ckpt_tiles
        )
        p_id_to_r_base = segment_start + segment_t_id * cfg.chunk_size
        p_id_to_r_size = jnp.minimum(segment_end - p_id_to_r_base, cfg.tile_size)
    else:
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
        s_idx_to_prefill_ckpt_row=s_idx_to_prefill_ckpt_row,
        s_idx_to_prefill_ckpt_index=s_idx_to_prefill_ckpt_index,
    )
