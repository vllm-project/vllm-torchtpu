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
"""The kernel's routing tables built inside the kernel, from the raw sequence
arrays, for decode-sized steps.

`metadata.compute_batched_seq_metadata` and `compute_per_seq_metadata` build
the tables the pipeline reads (tile -> sequence, row base, row count, first
and last tile flags, the per-sequence state slots) as XLA ops in the wrapper.
This path fills the same tables with scalar loops in SMEM at kernel entry:
a loop over the padded sequence count for the batched pass,
and a loop over the per-sequence range (empty on a decode step) for the other.
The tables live in SMEM scratch, so the path serves steps whose tables fit:
rows up to MAX_ROWS; larger steps keep the wrapper's XLA tables."""
import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.gdn.v3 import config, memory_ref

MAX_ROWS = 8192
I32 = jnp.int32


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class BoolSmemWrapper(memory_ref.SmemWrapper):
    """The flag tables as int32 words, read back as predicates."""

    def __getitem__(self, indices):
        return self.data[self._get_pos(indices)] != 0


class _BoolRef:

    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, i):
        return self.ref[i] != 0


class _ScalarRef:
    """A one-word SMEM scratch read as the scalar `metadata_ref.num_tiles[...]`."""

    def __init__(self, ref):
        self.ref = ref

    def __getitem__(self, _):
        return self.ref[0]


TABLE_NAMES = ("s_idx", "r_base", "r_size", "first", "last")


def scratch_shapes(max_seqs: int, max_tokens: int) -> dict[str, Any]:
    """SMEM scratch for both passes' tables: the batched pass indexes its
    tables by sequence (one tile entry per sequence), the per-sequence pass by
    tile (up to one per token)."""
    d = {
        "tb_num_tiles": pltpu.SMEM((1, ), I32),
        "tp_num_tiles": pltpu.SMEM((1, ), I32),
        "tb_has_init": pltpu.SMEM((max_seqs, ), I32),
        "tp_has_init": pltpu.SMEM((max_seqs, ), I32),
        "tp_state": pltpu.SMEM((max_seqs, ), I32),
        "tp_read": pltpu.SMEM((max_seqs, ), I32)
    }
    for n in TABLE_NAMES:
        d[f"tb_{n}"] = pltpu.SMEM((max_seqs, ), I32)
        d[f"tp_{n}"] = pltpu.SMEM((max_tokens, ), I32)
    return d


def fill_and_wrap(qsl_ref, seq_lens_ref, state_ref, read_ref, dist_ref,
                  sc: dict, cfg_b: config.GDNConfig, cfg_p: config.GDNConfig,
                  max_seqs: int):
    """Fill the tables from the raw arrays and return the two MetadataRefs
    over them. `sc` maps the scratch names of `scratch_shapes` to refs.
    distribution = [decode_end, prefill_end, mixed_end]: the batched pass takes
    sequences [0, decode_end), the per-sequence pass [decode_end, mixed_end)."""
    end_b = dist_ref[0]
    start_p = dist_ref[0]
    end_p = dist_ref[2]

    def body_b(s, carry):
        valid = s < end_b
        q_len = qsl_ref[s + 1] - qsl_ref[s]
        s_valid = jnp.where(valid, s, 0)
        sc["tb_s_idx"][s] = s_valid
        sc["tb_r_base"][s] = qsl_ref[s_valid]
        sc["tb_r_size"][s] = jnp.where(valid, q_len, 0)
        sc["tb_first"][s] = valid.astype(I32)
        sc["tb_last"][s] = valid.astype(I32)
        sc["tb_has_init"][s] = ((seq_lens_ref[s] - q_len) > 0).astype(I32)
        return carry

    lax.fori_loop(0, jnp.where(end_b > 0, max_seqs, 0), body_b, jnp.int32(0))
    sc["tb_num_tiles"][0] = pl.cdiv(end_b, cfg_b.tile_size)

    chunk = cfg_p.chunk_size
    tile = cfg_p.tile_size

    def body_s(s_rel, p):
        s = s_rel + start_p
        q_len = qsl_ref[s + 1] - qsl_ref[s]
        n = pl.cdiv(q_len, chunk)
        sc["tp_has_init"][s_rel] = ((seq_lens_ref[s] - q_len) > 0).astype(I32)
        sc["tp_state"][s_rel] = state_ref[s]
        sc["tp_read"][s_rel] = read_ref[s]

        def body_t(t, p):
            sc["tp_s_idx"][p] = s_rel
            r_base = qsl_ref[s] + t * chunk
            sc["tp_r_base"][p] = r_base
            sc["tp_r_size"][p] = jnp.minimum(qsl_ref[s + 1] - r_base, tile)
            sc["tp_first"][p] = (t == 0).astype(I32)
            sc["tp_last"][p] = (t == n - 1).astype(I32)
            return p + 1

        return lax.fori_loop(0, n, body_t, p)

    num_tiles_p = lax.fori_loop(0, end_p - start_p, body_s, jnp.int32(0))
    sc["tp_num_tiles"][0] = num_tiles_p

    def wrap(prefix, cfg, has_init, state, read):
        shape = (1, cfg.seq_tile_size)
        return memory_ref.MetadataRef(
            num_tiles=_ScalarRef(sc[f"{prefix}_num_tiles"]),
            p_id_to_s_idx=memory_ref.SmemWrapper(sc[f"{prefix}_s_idx"], shape),
            p_id_to_r_base=memory_ref.SmemWrapper(sc[f"{prefix}_r_base"],
                                                  shape),
            p_id_to_r_size=memory_ref.SmemWrapper(sc[f"{prefix}_r_size"],
                                                  shape),
            p_id_is_first_tile=BoolSmemWrapper(sc[f"{prefix}_first"], shape),
            p_id_is_last_tile=BoolSmemWrapper(sc[f"{prefix}_last"], shape),
            s_idx_has_initial_state=_BoolRef(has_init),
            s_idx_to_state_indices=state,
            s_idx_to_read_offset=read,
            s_idx_to_ckpt_indices=None,
        )

    meta_b = wrap("tb", cfg_b, sc["tb_has_init"], state_ref, read_ref)
    meta_p = wrap("tp", cfg_p, sc["tp_has_init"], sc["tp_state"],
                  sc["tp_read"])
    return meta_b, meta_p
