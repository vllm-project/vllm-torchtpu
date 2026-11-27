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
"""Sharding of the pooled step: `shard_map` specs of the step metadata, the
host-side per-rank checkpoint bounds check, and the sharded call of the
per-rank body (`cache_flat.run_local`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
from jax.sharding import PartitionSpec as P

from .cache_common import PooledBuild, default_act_mode, validate_page_table_shape
from .cache_flat import local_meta, run_local
from .pool import check_ckpt_bounds
from .pool.backend import resolve_active_seqs
from .pool.config import cache_exec


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class ModeTag:
    """Zero-leaf pytree wrapper for string modes like `'from_ckpt'` in `jax.jit`."""

    mode: str

    def tree_flatten(self) -> tuple[tuple[()], str]:
        return (), self.mode

    @classmethod
    def tree_unflatten(cls, aux_data: str, children: tuple[()]) -> ModeTag:
        del children
        return cls(aux_data)


def wrap_active_seqs(active_seqs: Any) -> Any:
    """Wrap an `active_seqs` mode string so it passes `jax.jit` as static data."""
    if isinstance(active_seqs, str):
        if active_seqs not in ("from_ckpt", "all"):
            raise ValueError(
                f"Unsupported active_seqs mode {active_seqs!r}; "
                "expected 'from_ckpt', 'all', None, or an int32/bool mask array"
            )
        return ModeTag(active_seqs)
    return active_seqs


def unwrap_active_seqs(active_seqs: Any) -> Any:
    return active_seqs.mode if isinstance(active_seqs, ModeTag) else active_seqs


def _meta_spec(b: PooledBuild, arr: Any, ns_dim: int, max_unsharded_ndim: int = 2) -> P:
    ex_paged = b.ex_paged
    if isinstance(arr, (ModeTag, str)) or arr is None or ex_paged.R <= 1:
        return P()
    if arr.ndim > max_unsharded_ndim:
        return P("tp") if arr.shape[0] == ex_paged.R else P()
    return P("tp") if arr.shape[0] == ex_paged.R * ns_dim else P()


def meta_specs(
    b: PooledBuild, page_table: Any, ckpt: Any, roff: Any, active: Any
) -> dict[str, P]:
    """`shard_map` spec of each step metadata array: `P("tp")` for rank-folded
    (leading `R * n` or rank-leading `R`) arrays, else replicated."""
    ns, slots = b.ex_paged.NS, b.rec_slots
    ckpt_2d = ckpt is not None and ckpt.ndim == 2
    roff_1d = roff is not None and roff.ndim == 1
    return {
        b.pt_key: _meta_spec(b, page_table, ns, max_unsharded_ndim=2),
        "ckpt_indices": _meta_spec(
            b, ckpt, ns if ckpt_2d else slots, max_unsharded_ndim=3
        ),
        "read_offsets": _meta_spec(
            b, roff, ns if roff_1d else slots, max_unsharded_ndim=2
        ),
        "active_seqs": _meta_spec(b, active, ns, max_unsharded_ndim=1),
    }


def check_ckpt_bounds_per_rank(
    b: PooledBuild, page_table: Any, ckpt: Any, roff: Any, active_seqs: Any
) -> None:
    """Host-side `check_ckpt_bounds` on the metadata each rank receives.

    Slices rank-folded arrays exactly like `raw_step` shards them
    (`meta_specs`), so `(R * NS, K)` / `(R, ...)` metadata is checked per rank.
    """
    ex, active_seqs = b.ex_paged, unwrap_active_seqs(active_seqs)
    specs = meta_specs(b, page_table, ckpt, roff, active_seqs)
    arrs = {
        b.pt_key: page_table,
        "ckpt_indices": ckpt,
        "read_offsets": roff,
        "active_seqs": active_seqs,
    }

    def _rank(k: str, r: int) -> Any:
        x = arrs[k]
        if specs[k] != P("tp"):
            return x
        n = x.shape[0] // ex.R
        return x[r * n : (r + 1) * n]

    for r in range(ex.R):
        _, ckpt_r, roff_r, act_r = local_meta(
            b,
            _rank(b.pt_key, r),
            _rank("ckpt_indices", r),
            _rank("read_offsets", r),
            _rank("active_seqs", r),
        )
        act_mode, mask = resolve_active_seqs(
            act_r, default_mode=default_act_mode(b), ns=ex.NS, has_recurrent=b.has_rec
        )
        check_ckpt_bounds(
            ckpt_r,
            cache_exec(ex).num_pages,
            b.pb.layout,
            act_mode=act_mode,
            active_mask=mask,
            read_offsets=roff_r,
            ns=ex.NS,
            min_k=ex.T,
        )


def raw_step(
    b: PooledBuild,
    pos: Any,
    hidden: Any,
    weights: dict[str, Any],
    raw_pools: tuple[jax.Array, ...],
    page_table: jax.Array,
    ckpt_indices: jax.Array | None,
    read_offsets: jax.Array | None,
    active_wrapped: Any,
    *extra_tuple: Any,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """Shard the pooled step's arguments and run `run_local` per rank."""
    pt_key = b.pt_key
    validate_page_table_shape(page_table, b.ex_paged)
    ws = {
        k: tuple(P("tp") for _ in weights[k])
        if isinstance(weights[k], tuple)
        else P("tp")
        for k in weights
    }
    raw_spec = tuple(P("tp") for _ in raw_pools)
    extra_specs = tuple(P("tp") if x.memory_space == "hbm" else P() for x in b.extras)

    all_specs = meta_specs(b, page_table, ckpt_indices, read_offsets, active_wrapped)
    meta: dict[str, Any] = {pt_key: page_table}
    if b.has_rec:
        meta["ckpt_indices"] = ckpt_indices
        meta["read_offsets"] = read_offsets
    if active_wrapped is not None:
        meta["active_seqs"] = active_wrapped
    m_specs = {k: all_specs[k] for k in meta}

    def local_pos(
        pos_l: Any,
        hid_l: Any,
        w_l: Any,
        raw_l: Any,
        meta_l: dict[str, Any],
        *extra_l: Any,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return run_local(
            b,
            pos_l,
            hid_l,
            w_l,
            raw_l,
            meta_l[pt_key],
            meta_l.get("ckpt_indices"),
            meta_l.get("read_offsets"),
            extra_l,
            unwrap_active_seqs(meta_l.get("active_seqs")),
        )

    return jax.shard_map(
        local_pos,
        mesh=b.mesh,
        in_specs=(P(), P("tp"), ws, raw_spec, m_specs) + extra_specs,
        out_specs=(P("tp"), raw_spec),
        check_vma=False,
    )(pos, hidden, weights, raw_pools, meta, *extra_tuple)
