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
"""Sharded body of the pooled step: `shard_map` specs and the per-rank
gather / megakernel / scatter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import kernel
from .cache_common import (
    ModeTag,
    resolve_active_seqs_mode,
    unwrap_active_seqs,
    validate_page_table_shape,
)
from .pool import check_ckpt_bounds
from .pool.config import cache_exec

if TYPE_CHECKING:
    from .cache_pooled import PooledBuild


def run_local(
    b: PooledBuild,
    pos: Any,
    hidden: Any,
    weights: dict[str, Any],
    raw_pools: tuple[jax.Array, ...],
    page_table: jax.Array,
    ckpt_indices: jax.Array | None,
    read_offsets: jax.Array | None,
    extra_tuple: tuple[Any, ...],
    active_seqs: Any,
) -> tuple[jax.Array, tuple[jax.Array, ...]]:
    """One rank's step: unpack weights, run the kernel, write the pools back."""
    pb, ex_paged, use_fused, use_gpr = b.pb, b.ex_paged, b.use_fused, b.use_gpr
    call, p, w_names, s_names, o_names = b.call, b.p, b.w_names, b.s_names, b.o_names
    wsh, has_rec, pool_key, pt_key = b.wsh, b.has_rec, b.pool_key, b.pt_key
    max_pt, _get_compiled = b.max_pt, b.get_compiled
    active_seqs = unwrap_active_seqs(active_seqs)
    raw_6d = raw_pools[0].ndim == 6
    raw_in = tuple(r.squeeze(0) for r in raw_pools) if raw_6d else tuple(raw_pools)
    if page_table.ndim == 3:
        page_table = page_table.squeeze(0)
    if page_table.shape[-1] > max_pt:
        page_table = page_table[..., :max_pt]
    if ckpt_indices is not None and ckpt_indices.ndim == 4:
        ckpt_indices = ckpt_indices.squeeze(0)
    if read_offsets is not None and read_offsets.ndim == 3:
        read_offsets = read_offsets.squeeze(0)
    if (
        active_seqs is not None
        and not isinstance(active_seqs, str)
        and active_seqs.ndim == 2
    ):
        active_seqs = active_seqs.squeeze(0)

    w = [
        tuple(
            kernel.unpack_local_weight(
                x,
                kernel.target_weight_shape(wsh[k]),
                name=k,
                allow_expert_reshape=k in p.dynamic_kinds,
            )
            for x in weights[k]
        )
        if isinstance(weights[k], tuple)
        else kernel.unpack_local_weight(
            weights[k],
            kernel.target_weight_shape(wsh[k]),
            name=k,
            allow_expert_reshape=k in p.dynamic_kinds,
        )
        for k in w_names
    ]

    if use_fused or use_gpr:
        act_mode_l, active_mask_l = resolve_active_seqs_mode(
            active_seqs, default_mode="from_ckpt" if has_rec else "all"
        )
        if has_rec and cache_exec(ex_paged).check_ckpt_bounds:
            check_ckpt_bounds(
                ckpt_indices,
                cache_exec(ex_paged).num_pages,
                pb.layout,
                act_mode=act_mode_l,
                active_mask=active_mask_l,
            )
        ckpt_2d_l = bool(ckpt_indices is not None and ckpt_indices.ndim == 2)
        ckpt_k_l = (
            int(ckpt_indices.shape[-1]) if ckpt_indices is not None else ex_paged.T
        )
        roff_1d_l = bool(read_offsets is not None and read_offsets.ndim == 1)
        call_l, info_l = _get_compiled(act_mode_l, ckpt_2d_l, ckpt_k_l, roff_1d_l)
        p_l = info_l["plan"]
        st_map_f: dict[str, Any] = {
            pool_key: raw_in,
            pt_key: page_table,
        }
        if has_rec:
            st_map_f["ckpt_indices"] = ckpt_indices
            st_map_f["read_offsets"] = read_offsets
            if act_mode_l == "ref":
                st_map_f["active_seqs"] = active_mask_l
        s_args_f = [st_map_f[k] for k in p_l.s_names]
        out_vals_f = call_l(pos, hidden, *w, *s_args_f, *extra_tuple)
        if not isinstance(out_vals_f, (tuple, list)):
            out_vals_f = (out_vals_f,)
        out_map_f = dict(zip(p_l.o_names, out_vals_f))
        hid = out_map_f["hidden"]
        if use_fused:
            raw_out = tuple(out_map_f[pool_key]) if pool_key in out_map_f else raw_in
        else:
            raw_out = pb.scatter_states(
                ex_paged,
                raw_in,
                pos,
                page_table,
                out_map_f,
                ckpt_indices if has_rec else None,
                active_seqs,
            )
        if raw_6d:
            raw_out = tuple(jnp.expand_dims(r, 0) for r in raw_out)
        return hid, raw_out

    gathered = (
        pb.gather_states(
            ex_paged,
            raw_in,
            ckpt_indices,
            read_offsets,
            active_seqs,
        )
        if (has_rec and pb.gather_states is not None)
        else {}
    )

    st_map: dict[str, Any] = {
        **gathered,
        pool_key: raw_in,
        pt_key: page_table,
    }
    s_args = [st_map[k] for k in s_names]
    out_vals = call(pos, hidden, *w, *s_args, *extra_tuple)
    if not isinstance(out_vals, (tuple, list)):
        out_vals = (out_vals,)
    out_map = dict(zip(o_names, out_vals))
    hid = out_map["hidden"]

    raw_out = pb.scatter_states(
        ex_paged,
        raw_in,
        pos,
        page_table,
        out_map,
        ckpt_indices if has_rec else None,
        active_seqs,
    )

    if raw_6d:
        raw_out = tuple(jnp.expand_dims(r, 0) for r in raw_out)
    return hid, raw_out


def _meta_spec(b: PooledBuild, arr: Any, ns_dim: int, max_unsharded_ndim: int = 2) -> P:
    ex_paged = b.ex_paged
    if isinstance(arr, ModeTag) or arr is None or ex_paged.R <= 1:
        return P()
    if arr.ndim > max_unsharded_ndim:
        return P("tp") if arr.shape[0] == ex_paged.R else P()
    return P("tp") if arr.shape[0] == ex_paged.R * ns_dim else P()


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
    mesh, ex_paged, extras, has_rec = b.mesh, b.ex_paged, b.extras, b.has_rec
    rec_slots, pt_key = b.rec_slots, b.pt_key
    validate_page_table_shape(page_table, ex_paged)
    ws = {
        k: tuple(P("tp") for _ in weights[k])
        if isinstance(weights[k], tuple)
        else P("tp")
        for k in weights
    }
    raw_spec = tuple(P("tp") for _ in raw_pools)
    extra_specs = tuple(P("tp") if x.memory_space == "hbm" else P() for x in extras)

    meta: dict[str, Any] = {pt_key: page_table}
    meta_specs: dict[str, Any] = {
        pt_key: _meta_spec(b, page_table, ex_paged.NS, max_unsharded_ndim=2)
    }
    if has_rec:
        meta["ckpt_indices"] = ckpt_indices
        meta["read_offsets"] = read_offsets
        meta_specs["ckpt_indices"] = _meta_spec(
            b,
            ckpt_indices,
            ex_paged.NS
            if (ckpt_indices is not None and ckpt_indices.ndim == 2)
            else rec_slots,
            max_unsharded_ndim=3,
        )
        meta_specs["read_offsets"] = _meta_spec(
            b,
            read_offsets,
            ex_paged.NS
            if (read_offsets is not None and read_offsets.ndim == 1)
            else rec_slots,
            max_unsharded_ndim=2,
        )
    if active_wrapped is not None:
        meta["active_seqs"] = active_wrapped
        meta_specs["active_seqs"] = _meta_spec(
            b, active_wrapped, ex_paged.NS, max_unsharded_ndim=1
        )

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
            meta_l.get("active_seqs"),
        )

    return jax.shard_map(
        local_pos,
        mesh=mesh,
        in_specs=(P(), P("tp"), ws, raw_spec, meta_specs) + extra_specs,
        out_specs=(P("tp"), raw_spec),
        check_vma=False,
    )(pos, hidden, weights, raw_pools, meta, *extra_tuple)
