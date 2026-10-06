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
"""Pooled GDN routing tables built inside the kernel.

Fills the tables of `metadata.compute_batched_seq_metadata` and
`metadata.compute_per_seq_metadata` with scalar loops into SMEM scratch at
kernel entry, so one call can run both passes without XLA table ops. The
per-sequence tables are indexed by tile, so steps above `MAX_ROWS` rows keep
the wrapper's XLA tables.
"""

import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref

MAX_ROWS = 256

_TABLE_NAMES = ("s_idx", "r_base", "r_size", "first", "last")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class _BoolSmemWrapper(memory_ref.SmemWrapper):
    """Flag table stored as int32 words, read as predicates."""

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)] != 0


class _BoolRef:
    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, i):
        return self.ref[i] != 0


class _ScalarRef:
    """One-word SMEM scratch read as `metadata_ref.num_tiles[...]`."""

    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, _):
        return self.ref[0]


def batched_table_size(max_seqs: int, decode_tile_size: int) -> int:
    """Batched-pass entries: the pipeline reads whole tiles of sequences."""
    return pl.cdiv(max_seqs, decode_tile_size) * decode_tile_size


def scratch_shapes(
    max_seqs: int, max_tokens: int, decode_tile_size: int, num_ckpts: int = 0
) -> dict[str, Any]:
    """SMEM scratch for both passes' tables.

    The batched pass has one entry per sequence; the per-sequence pass one per
    tile, at most one per token. With `num_ckpts` checkpoint blocks per
    sequence (speculative decoding), the per-sequence pass also keeps its
    shifted copy of the checkpoint indices.
    """
    num_b = batched_table_size(max_seqs, decode_tile_size)
    shapes = {
        "tb_num_tiles": pltpu.SMEM((1,), jnp.int32),
        "tp_num_tiles": pltpu.SMEM((1,), jnp.int32),
        "tb_has_init": pltpu.SMEM((num_b,), jnp.int32),
        "tp_has_init": pltpu.SMEM((max_seqs,), jnp.int32),
        "tp_state": pltpu.SMEM((max_seqs,), jnp.int32),
        "tp_read": pltpu.SMEM((max_seqs,), jnp.int32),
    }
    for name in _TABLE_NAMES:
        shapes[f"tb_{name}"] = pltpu.SMEM((num_b,), jnp.int32)
        shapes[f"tp_{name}"] = pltpu.SMEM((max_tokens,), jnp.int32)
    if num_ckpts:
        shapes["tp_ckpt"] = pltpu.SMEM((max_seqs * num_ckpts,), jnp.int32)
    return shapes


def _vmem_bytes(ref: Any, info: Any) -> int:
    """Bytes of a VMEM scratch, its two minor dims padded to whole tiles."""
    *major, rows, cols = ref.shape
    sublanes = info.num_sublanes * 4 // jnp.dtype(ref.dtype).itemsize
    rows = pl.cdiv(rows, sublanes) * sublanes
    cols = pl.cdiv(cols, info.num_lanes) * info.num_lanes
    return math.prod(major) * rows * cols * jnp.dtype(ref.dtype).itemsize


def vmem_limit_bytes(cfg_b: config.GDNConfig, cfg_p: config.GDNConfig) -> int:
    """Scoped VMEM the one-call form needs, from each pass's own limit.

    Each pass's pipeline buffers are scoped to that pass, so the call peaks at
    one pass's limit plus what of the other pass stays live across it: the
    per-sequence carry scratch, held through the batched pass.

    Each pass's `cfg.split_state` is chosen against that pass's own limit
    (`memory_ref.split_state_fits`, which counts the per-sequence pass's
    carry), and the carry is added on top of the batched pass's limit here,
    so split buffers fit inside the one call wherever they fit in two.
    """
    info = pltpu.get_tpu_info()
    carry = sum(
        _vmem_bytes(ref, info)
        for ref in cfg_p.get_scratch_shape_dict().values()
        if ref is not None
    )
    return max(cfg_b.get_vmem_limit_bytes() + carry, cfg_p.get_vmem_limit_bytes())


def fill_and_wrap(
    query_start_loc_ref,
    seq_lens_ref,
    state_indices_ref,
    read_offsets_ref,
    distribution_ref,
    ckpt_indices_ref,
    sc: dict[str, Any],
    cfg_b: config.GDNConfig,
    cfg_p: config.GDNConfig,
    max_seqs: int,
) -> tuple[memory_ref.MetadataRef, memory_ref.MetadataRef]:
    """Fill the tables from the raw arrays; return both passes' MetadataRefs.

    `sc` maps the names of `scratch_shapes` to refs. The batched pass takes
    sequences `[0, distribution[0])`, the per-sequence pass
    `[distribution[0], distribution[2])`, whose per-sequence arrays are
    shifted to start at `distribution[0]`. `ckpt_indices_ref` is the
    `[max_seqs, num_ckpts]` checkpoint block table, or None.
    """
    qsl = query_start_loc_ref
    end_b = distribution_ref[0]
    start_p = distribution_ref[0]
    end_p = distribution_ref[2]

    def fill_batched(s, carry):
        valid = s < end_b
        s_clamped = jnp.minimum(s, max_seqs - 1)
        q_len = qsl[s_clamped + 1] - qsl[s_clamped]
        s_valid = jnp.where(valid, s, 0)
        sc["tb_s_idx"][s] = s_valid
        sc["tb_r_base"][s] = qsl[s_valid]
        sc["tb_r_size"][s] = jnp.where(valid, q_len, 0)
        sc["tb_first"][s] = valid.astype(jnp.int32)
        sc["tb_last"][s] = valid.astype(jnp.int32)
        has_init = (seq_lens_ref[s_clamped] - q_len) > 0
        sc["tb_has_init"][s] = has_init.astype(jnp.int32)
        return carry

    num_b = batched_table_size(max_seqs, cfg_b.tile_size)
    lax.fori_loop(0, num_b, fill_batched, 0)
    sc["tb_num_tiles"][0] = pl.cdiv(end_b, cfg_b.tile_size)

    def fill_seq(s_rel, p):
        s = s_rel + start_p
        q_len = qsl[s + 1] - qsl[s]
        num_tiles = pl.cdiv(q_len, cfg_p.chunk_size)
        has_init = (seq_lens_ref[s] - q_len) > 0
        sc["tp_has_init"][s_rel] = has_init.astype(jnp.int32)
        sc["tp_state"][s_rel] = state_indices_ref[s]
        sc["tp_read"][s_rel] = read_offsets_ref[s]
        if ckpt_indices_ref is not None:
            num_ckpts = ckpt_indices_ref.shape[1]
            for t in range(num_ckpts):
                sc["tp_ckpt"][s_rel * num_ckpts + t] = ckpt_indices_ref[s, t]

        def fill_tile(t, p):
            r_base = qsl[s] + t * cfg_p.chunk_size
            sc["tp_s_idx"][p] = s_rel
            sc["tp_r_base"][p] = r_base
            sc["tp_r_size"][p] = jnp.minimum(qsl[s + 1] - r_base, cfg_p.tile_size)
            sc["tp_first"][p] = (t == 0).astype(jnp.int32)
            sc["tp_last"][p] = (t == num_tiles - 1).astype(jnp.int32)
            return p + 1

        return lax.fori_loop(0, num_tiles, fill_tile, p)

    sc["tp_num_tiles"][0] = lax.fori_loop(0, end_p - start_p, fill_seq, 0)

    def wrap(prefix, cfg, state_indices, read_offsets, ckpt_indices):
        shape = (1, cfg.seq_tile_size)
        return memory_ref.MetadataRef(
            num_tiles=_ScalarRef(sc[f"{prefix}_num_tiles"]),
            p_id_to_s_idx=memory_ref.SmemWrapper(sc[f"{prefix}_s_idx"], shape),
            p_id_to_r_base=memory_ref.SmemWrapper(sc[f"{prefix}_r_base"], shape),
            p_id_to_r_size=memory_ref.SmemWrapper(sc[f"{prefix}_r_size"], shape),
            p_id_is_first_tile=_BoolSmemWrapper(sc[f"{prefix}_first"], shape),
            p_id_is_last_tile=_BoolSmemWrapper(sc[f"{prefix}_last"], shape),
            s_idx_has_initial_state=_BoolRef(sc[f"{prefix}_has_init"]),
            s_idx_to_state_indices=state_indices,
            s_idx_to_read_offset=read_offsets,
            s_idx_to_ckpt_indices=ckpt_indices,
        )

    ckpt_p = None
    if ckpt_indices_ref is not None:
        ckpt_p = memory_ref.SmemWrapper(sc["tp_ckpt"], ckpt_indices_ref.shape)
    meta_b = wrap("tb", cfg_b, state_indices_ref, read_offsets_ref, ckpt_indices_ref)
    meta_p = wrap("tp", cfg_p, sc["tp_state"], sc["tp_read"], ckpt_p)
    return meta_b, meta_p
