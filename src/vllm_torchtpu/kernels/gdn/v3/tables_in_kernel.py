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
"""GDN v3 routing tables built inside the kernel for decode-sized steps.

`metadata.compute_batched_seq_metadata` and `compute_per_seq_metadata` build
the tables the pipeline reads (tile to sequence, row base, row count, first
and last tile flags, per-sequence state slots) as XLA ops in the wrapper,
around two Pallas calls. This module fills the same tables with scalar loops
in SMEM at kernel entry, so one Pallas call runs both passes: a loop over the
sequence count for the batched pass, and a loop over the per-sequence range
(empty on a pure decode step) for the other.

The tables live in SMEM scratch, so the wrapper uses this path only for steps
with at most `MAX_ROWS` padded rows and without speculative decoding.
"""

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref

MAX_ROWS = 256

_TILE_TABLE_NAMES = ("s_idx", "r_base", "r_size", "first", "last")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BoolSmemWrapper(memory_ref.SmemWrapper):
    """A flag table stored as int32 words, read back as predicates."""

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)] != 0


class _BoolRef:
    """A per-sequence flag table stored as int32 words, read as predicates."""

    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, index):
        return self.ref[index] != 0


class _ScalarRef:
    """A one-word SMEM scratch read like the scalar `num_tiles[...]` operand."""

    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, _):
        return self.ref[0]


def scratch_shapes(
    max_seqs: int, max_tokens: int, has_prefill_ckpt: bool
) -> dict[str, Any]:
    """SMEM scratch for both passes' tables.

    The batched pass has one tile entry per sequence, so its tile tables are
    sized by sequence. The per-sequence pass has at most one tile per token.
    """
    shapes = {
        "batched_num_tiles": pltpu.SMEM((1,), jnp.int32),
        "per_seq_num_tiles": pltpu.SMEM((1,), jnp.int32),
        "batched_has_init": pltpu.SMEM((max_seqs,), jnp.int32),
        "per_seq_has_init": pltpu.SMEM((max_seqs,), jnp.int32),
        "per_seq_state": pltpu.SMEM((max_seqs,), jnp.int32),
        "per_seq_read": pltpu.SMEM((max_seqs,), jnp.int32),
    }
    for name in _TILE_TABLE_NAMES:
        shapes[f"batched_{name}"] = pltpu.SMEM((max_seqs,), jnp.int32)
        shapes[f"per_seq_{name}"] = pltpu.SMEM((max_tokens,), jnp.int32)
    if has_prefill_ckpt:
        shapes["per_seq_prefill_ckpt_row"] = pltpu.SMEM((max_seqs,), jnp.int32)
        shapes["per_seq_prefill_ckpt_index"] = pltpu.SMEM((max_seqs,), jnp.int32)
    return shapes


def fill_and_wrap(
    query_start_loc_ref,
    seq_lens_ref,
    state_indices_ref,
    read_offsets_ref,
    distribution_ref,
    prefill_ckpt_offsets_ref,
    prefill_ckpt_indices_ref,
    tables: dict,
    batched_cfg: config.GDNConfig,
    per_seq_cfg: config.GDNConfig,
    max_seqs: int,
) -> tuple[memory_ref.MetadataRef, memory_ref.MetadataRef]:
    """Fills the tables from the raw arrays and returns both passes' metadata.

    `tables` maps the names of `scratch_shapes` to refs. The values match
    `metadata.compute_batched_seq_metadata` and
    `metadata.compute_per_seq_metadata` at every index the pipeline reads.
    distribution = [decode_end, prefill_end, mixed_end]: the batched pass
    takes sequences [0, decode_end), the per-sequence pass
    [decode_end, mixed_end).
    """
    batched_end = distribution_ref[0]
    per_seq_start = distribution_ref[0]
    per_seq_end = distribution_ref[2]
    has_prefill_ckpt = prefill_ckpt_offsets_ref is not None

    def fill_batched_seq(s, carry):
        valid = s < batched_end
        query_len = query_start_loc_ref[s + 1] - query_start_loc_ref[s]
        valid_s = jnp.where(valid, s, 0)
        tables["batched_s_idx"][s] = valid_s
        tables["batched_r_base"][s] = query_start_loc_ref[valid_s]
        tables["batched_r_size"][s] = jnp.where(valid, query_len, 0)
        tables["batched_first"][s] = valid.astype(jnp.int32)
        tables["batched_last"][s] = valid.astype(jnp.int32)
        tables["batched_has_init"][s] = (seq_lens_ref[s] - query_len > 0).astype(
            jnp.int32
        )
        return carry

    lax.fori_loop(0, max_seqs, fill_batched_seq, jnp.int32(0))
    tables["batched_num_tiles"][0] = pl.cdiv(batched_end, batched_cfg.tile_size)

    chunk_size = per_seq_cfg.chunk_size
    tile_size = per_seq_cfg.tile_size

    def fill_per_seq(rel_s, num_tiles):
        s = rel_s + per_seq_start
        query_start = query_start_loc_ref[s]
        query_end = query_start_loc_ref[s + 1]
        query_len = query_end - query_start
        tables["per_seq_has_init"][rel_s] = (seq_lens_ref[s] - query_len > 0).astype(
            jnp.int32
        )
        tables["per_seq_state"][rel_s] = state_indices_ref[s]
        tables["per_seq_read"][rel_s] = read_offsets_ref[s]

        # An internal prefill checkpoint splits the sequence's tiles at
        # `split_row` so that row lands on a tile boundary.
        num_pre_ckpt_tiles = jnp.int32(0)
        split_row = query_start
        if has_prefill_ckpt:
            offset = prefill_ckpt_offsets_ref[s]
            has_ckpt = jnp.logical_and(
                jnp.logical_and(offset > 0, offset < query_len),
                prefill_ckpt_indices_ref[s] >= 0,
            )
            offset = jnp.where(has_ckpt, offset, 0)
            split_row = query_start + offset
            tables["per_seq_prefill_ckpt_row"][rel_s] = jnp.where(
                offset > 0, split_row, -1
            )
            tables["per_seq_prefill_ckpt_index"][rel_s] = jnp.where(
                has_ckpt, prefill_ckpt_indices_ref[s], -1
            )
            num_pre_ckpt_tiles = pl.cdiv(offset, chunk_size)
        num_seq_tiles = num_pre_ckpt_tiles + pl.cdiv(query_end - split_row, chunk_size)

        def fill_tile(t, p_id):
            is_pre_ckpt = t < num_pre_ckpt_tiles
            segment_start = jnp.where(is_pre_ckpt, query_start, split_row)
            segment_end = jnp.where(is_pre_ckpt, split_row, query_end)
            segment_t = jnp.where(is_pre_ckpt, t, t - num_pre_ckpt_tiles)
            r_base = segment_start + segment_t * chunk_size
            tables["per_seq_s_idx"][p_id] = rel_s
            tables["per_seq_r_base"][p_id] = r_base
            tables["per_seq_r_size"][p_id] = jnp.minimum(
                segment_end - r_base, tile_size
            )
            tables["per_seq_first"][p_id] = (t == 0).astype(jnp.int32)
            tables["per_seq_last"][p_id] = (t == num_seq_tiles - 1).astype(jnp.int32)
            return p_id + 1

        return lax.fori_loop(0, num_seq_tiles, fill_tile, num_tiles)

    tables["per_seq_num_tiles"][0] = lax.fori_loop(
        0, per_seq_end - per_seq_start, fill_per_seq, jnp.int32(0)
    )

    def wrap(prefix, cfg, has_init, state_indices, read_offsets, **prefill_ckpt):
        shape = (1, cfg.seq_tile_size)
        tile_tables = {
            name: memory_ref.SmemWrapper(tables[f"{prefix}_{name}"], shape)
            for name in ("s_idx", "r_base", "r_size")
        }
        return memory_ref.MetadataRef(
            num_tiles=_ScalarRef(tables[f"{prefix}_num_tiles"]),
            p_id_to_s_idx=tile_tables["s_idx"],
            p_id_to_r_base=tile_tables["r_base"],
            p_id_to_r_size=tile_tables["r_size"],
            p_id_is_first_tile=BoolSmemWrapper(tables[f"{prefix}_first"], shape),
            p_id_is_last_tile=BoolSmemWrapper(tables[f"{prefix}_last"], shape),
            s_idx_has_initial_state=_BoolRef(has_init),
            s_idx_to_state_indices=state_indices,
            s_idx_to_read_offset=read_offsets,
            # Without speculative decoding the only checkpoint is the slot's
            # own state block, which `s_idx_to_state_indices` already names.
            s_idx_to_ckpt_indices=None,
            **prefill_ckpt,
        )

    batched_metadata = wrap(
        "batched",
        batched_cfg,
        tables["batched_has_init"],
        state_indices_ref,
        read_offsets_ref,
    )
    prefill_ckpt = {}
    if has_prefill_ckpt:
        prefill_ckpt = dict(
            s_idx_to_prefill_ckpt_row=tables["per_seq_prefill_ckpt_row"],
            s_idx_to_prefill_ckpt_index=tables["per_seq_prefill_ckpt_index"],
        )
    per_seq_metadata = wrap(
        "per_seq",
        per_seq_cfg,
        tables["per_seq_has_init"],
        tables["per_seq_state"],
        tables["per_seq_read"],
        **prefill_ckpt,
    )
    return batched_metadata, per_seq_metadata
