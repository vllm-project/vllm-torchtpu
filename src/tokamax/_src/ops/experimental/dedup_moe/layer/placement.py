# Copyright 2026 The Tokamax Authors.
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
# ==============================================================================
"""Host-side helpers to serve a static expert placement (EPLB) from a caller.

A placement is `slot_to_orig[S]` int32: physical expert slot `s` holds logical
expert `slot_to_orig[s]`. Slots are numbered EP-rank-major (slot `s` lives on EP
rank `s // (S // ep)`), which is the order `dedup_ep_moe`'s
`logical_to_physical_map` produces before it relabels ids to mesh order.
`S == e_logical` is a pure permutation. `S > e_logical` adds replicas, so every
logical expert needs at least one slot.

A caller that holds expert weights as a mesh-ordered global array
(`[e_logical, ...]`, sharded on the EP axis, mesh index `m` holding EP rank
`mesh_ep_ranks[m]`'s contiguous block) does two things:

  1. Once at load: `w_placed = jnp.take(w, source_rows(...), axis=0)` for every
     expert-major operand (weights and per-expert scales). This yields the
     `[S, ...]` mesh-ordered array that holds slot `s` where the kernel looks
     for it.
  2. On every call: pass `logical_to_physical_map(...)` to `dedup_ep_moe`.

The output is bit-identical to the unplaced layer, because the router still
scores logical experts and only the selected ids are relabeled.
"""

from collections.abc import Sequence
import os
import re

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from tokamax._src.ops.experimental.dedup_moe.dedup_plan import topology

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def validate(slot_to_orig, *, e_logical: int, ep: int) -> np.ndarray:
  """`slot_to_orig` as int32 [S] after checking it is servable; raises otherwise."""
  s2o = np.asarray(slot_to_orig)
  if s2o.ndim != 1 or not np.issubdtype(s2o.dtype, np.integer):
    raise ValueError(f"slot_to_orig must be a 1-D integer array; got {s2o.dtype}{s2o.shape}")
  s2o = s2o.astype(np.int32)
  n_slots = s2o.shape[0]
  if n_slots < e_logical or n_slots % ep:
    raise ValueError(
        f"{n_slots} slots must be >= {e_logical} logical experts and divide over ep={ep}"
    )
  if s2o.min() < 0 or s2o.max() >= e_logical:
    raise ValueError(f"slot_to_orig values must lie in [0, {e_logical})")
  missing = np.setdiff1d(np.arange(e_logical), s2o)
  if missing.size:
    raise ValueError(f"logical experts with no slot: {missing[:8].tolist()}...")
  return s2o


def _rank_maps(ep: int, mesh_ep_ranks: Sequence[int] | None):
  if mesh_ep_ranks is None:
    ident = np.arange(ep, dtype=np.int64)
    return ident, ident
  rank_at_mesh = np.asarray(mesh_ep_ranks, dtype=np.int64)
  if sorted(rank_at_mesh.tolist()) != list(range(ep)):
    raise ValueError(f"mesh_ep_ranks must permute range({ep})")
  mesh_of_rank = np.empty(ep, dtype=np.int64)
  mesh_of_rank[rank_at_mesh] = np.arange(ep)
  return rank_at_mesh, mesh_of_rank


def source_rows(
    slot_to_orig,
    *,
    e_logical: int,
    ep: int,
    mesh_ep_ranks: Sequence[int] | None = None,
) -> np.ndarray:
  """int32 [S]: placed global row g takes current global row `rows[g]`.

  Both arrays are mesh-ordered: mesh index m holds rows [m * local, (m + 1) *
  local). The current array holds EP rank r's contiguous logical block
  [r * e_logical // ep, ...), and the placed one holds EP rank r's slot block.
  """
  s2o = validate(slot_to_orig, e_logical=e_logical, ep=ep)
  rank_at_mesh, mesh_of_rank = _rank_maps(ep, mesh_ep_ranks)
  g_new = s2o.shape[0] // ep
  g_old = e_logical // ep
  if g_old * ep != e_logical:
    raise ValueError(f"{e_logical} logical experts do not divide over ep={ep}")
  row = np.arange(s2o.shape[0], dtype=np.int64)
  slot = rank_at_mesh[row // g_new] * g_new + row % g_new
  orig = s2o[slot].astype(np.int64)
  cur_row = mesh_of_rank[orig // g_old] * g_old + orig % g_old
  return cur_row.astype(np.int32)


def logical_to_physical_map(
    slot_to_orig, *, e_logical: int, ep: int, board_size: int | None = None
) -> np.ndarray:
  """The `dedup_ep_moe(logical_to_physical_map=...)` operand for a placement.

  For a pure permutation, this is 1-D `orig_to_slot[e_logical]`. With replicas,
  it is the 3-D `[e_logical, num_boards, REP_BUCKETS]` quota-balanced LUT.
  """
  s2o = validate(slot_to_orig, e_logical=e_logical, ep=ep)
  if s2o.shape[0] == e_logical:
    orig_to_slot = np.empty(e_logical, dtype=np.int32)
    orig_to_slot[s2o] = np.arange(e_logical, dtype=np.int32)
    return orig_to_slot
  return np.asarray(
      topology.build_rep_lut(
          s2o, ep=ep, e_total=e_logical, board_size=board_size
      ),
      dtype=np.int32,
  )


def place_rows(w, rows, *, mesh, axis: str):
  """`jnp.take(w, rows, axis=0)` for an EP-sharded `w`, with bounded memory.

  `w` is `[R, ...]`, sharded on `axis`, and `rows` is `[S]` (replicated). The
  result is `[S, ...]`, sharded on `axis`. A plain sharded gather can make
  XLA materialize the full `[S, ...]` gather on every device (16 GiB per
  layer at Qwen3.8 shapes). This version rotates the shards around the ring
  with `ppermute` and selects the rows each device needs, so the peak extra
  memory is one input shard plus one output shard. The result is a pure copy,
  so it is bit-exact.
  """
  pspec = jax.sharding.PartitionSpec
  ep = mesh.shape[axis]
  n_new = rows.shape[0]
  if n_new % ep or w.shape[0] % ep:
    raise ValueError(f"rows {n_new} and w {w.shape[0]} must divide over {ep}")
  g_new, g_old = n_new // ep, w.shape[0] // ep
  perm = [(i, (i + 1) % ep) for i in range(ep)]

  def body(w_l, rows_all):
    m = lax.axis_index(axis)
    mine = lax.dynamic_slice_in_dim(rows_all.astype(jnp.int32), m * g_new, g_new)
    bshape = (g_new,) + (1,) * (w_l.ndim - 1)

    def step(k, carry):
      out, buf = carry
      src = lax.rem(m - k + ep, ep)  # buf holds mesh index src's shard
      local = mine - src * g_old
      hit = (local >= 0) & (local < g_old)
      picked = jnp.take(buf, jnp.clip(local, 0, g_old - 1), axis=0)
      out = jnp.where(hit.reshape(bshape), picked, out)
      return out, lax.ppermute(buf, axis, perm)

    out0 = jnp.zeros((g_new,) + w_l.shape[1:], w_l.dtype)
    out, _ = lax.fori_loop(0, ep, step, (out0, w_l))
    return out

  return jax.shard_map(
      body,
      mesh=mesh,
      in_specs=(pspec(axis), pspec()),
      out_specs=pspec(axis),
      check_vma=False,
  )(w, rows)


def layer_index(layer_name: str) -> int | None:
  """Transformer layer index from a module path like `model.layers.7.mlp.experts`."""
  m = _LAYER_RE.search(layer_name or "")
  return int(m.group(1)) if m else None


def load_layer_placement(directory: str, layer_idx: int) -> np.ndarray | None:
  """`<directory>/layer_<idx:03d>.npy` as int32, or None if there is no file (layer unplaced)."""
  path = os.path.join(directory, f"layer_{layer_idx:03d}.npy")
  if not os.path.exists(path):
    return None
  return np.load(path).astype(np.int32)
