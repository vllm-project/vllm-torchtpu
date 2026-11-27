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
"""Per-rank pooled step body (`run_local`) and the flat-signature pooled step
(`make_flat_pooled_step`) for bridge integration."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from . import kernel
from .cache_common import (
    PooledBuild,
    build_pooled,
    default_act_mode,
    validate_page_table_shape,
)
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .pool import check_ckpt_bounds, check_step_pool_metadata
from .pool.backend import resolve_active_seqs
from .pool.config import (
    FUSED_POOL_IO_MAX_B,
    FUSED_POOL_IO_MIN_B_HI,
    cache_exec,
    with_cache,
)


def _drop_lead(x: Any, ndim: int, name: str) -> Any:
    """Squeeze a rank-leading `ndim`-D `x` (its leading dim must be 1 per rank)."""
    if x is None or isinstance(x, str) or x.ndim != ndim:
        return x
    if x.shape[0] != 1:
        raise ValueError(
            f"{name} has shape {tuple(x.shape)}; a rank-leading {name} needs a "
            "leading dim of 1 per rank"
        )
    return x.squeeze(0)


def local_meta(
    b: PooledBuild, page_table: Any, ckpt: Any, roff: Any, active_seqs: Any
) -> tuple[Any, Any, Any, Any]:
    """One rank's `(page_table, ckpt_indices, read_offsets, active_seqs)` with
    the rank-leading axes dropped and the page table trimmed to `max_pt`."""
    page_table = _drop_lead(page_table, 3, "page_table")
    if page_table.shape[-1] > b.max_pt:
        page_table = page_table[..., : b.max_pt]
    return (
        page_table,
        _drop_lead(ckpt, 4, "ckpt_indices"),
        _drop_lead(roff, 3, "read_offsets"),
        _drop_lead(active_seqs, 2, "active_seqs"),
    )


def _unpack_weights(b: PooledBuild, weights: dict[str, Any]) -> list[Any]:
    def _one(k: str, x: Any) -> Any:
        t_sh = kernel.target_weight_shape(b.wsh[k])
        return kernel.unpack_local_weight(x, t_sh, name=k, allow_merged_tiles=True)

    return [
        tuple(_one(k, x) for x in weights[k])
        if isinstance(weights[k], tuple)
        else _one(k, weights[k])
        for k in b.w_names
    ]


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
    """One rank's step: unpack weights, run the kernel, write the pools back.

    `active_seqs` is None, a mode (`"from_ckpt"` / `"all"`) or a mask array.
    6D `(1, num_pages, ...)` pools come back 6D, so donated inputs and outputs
    keep the same shape.
    """
    pb, ex_paged, has_rec = b.pb, b.ex_paged, b.has_rec
    cx = cache_exec(ex_paged)
    raw_6d = raw_pools[0].ndim == 6
    raw_in = tuple(_drop_lead(r, 6, "raw_pools") for r in raw_pools)
    page_table, ckpt_indices, read_offsets, active_seqs = local_meta(
        b, page_table, ckpt_indices, read_offsets, active_seqs
    )
    w = _unpack_weights(b, weights)
    act_mode, act_mask = resolve_active_seqs(
        active_seqs,
        default_mode="from_ckpt" if has_rec else "all",
        ns=ex_paged.NS,
        has_recurrent=has_rec,
    )
    if has_rec:
        # Trace-time shape admission (no ops are emitted).
        check_ckpt_bounds(
            ckpt_indices,
            cx.num_pages,
            pb.layout,
            act_mode=act_mode,
            active_mask=act_mask,
            read_offsets=read_offsets,
            raw_pools=raw_in,
            ns=ex_paged.NS,
            min_k=ex_paged.T,
        )

    if b.use_fused or b.use_gpr:
        if cx.check_ckpt_bounds:
            # Fused I/O skips the scatter kernel and GPR the gather kernel, so
            # validate the step's pool metadata (read offsets included) here.
            # Under "all" the kernel treats every sequence as active, so check
            # every sequence (with no mask the check infers activity from
            # checkpoint / page-table column 0, as "from_ckpt" does).
            check_mask = (
                jnp.ones((ex_paged.NS,), jnp.bool_) if act_mode == "all" else act_mask
            )
            check_step_pool_metadata(
                pb.layout,
                ex_paged,
                page_table,
                pos,
                ckpt_indices=ckpt_indices if has_rec else None,
                read_offsets=read_offsets if has_rec else None,
                active_seqs=check_mask,
                check_bounds=True,
                force_under_jit=True,
            )
        call, info = b.get_compiled(
            act_mode,
            bool(ckpt_indices is not None and ckpt_indices.ndim == 2),
            int(ckpt_indices.shape[-1]) if ckpt_indices is not None else ex_paged.T,
            bool(read_offsets is not None and read_offsets.ndim == 1),
        )
        s_names, o_names = info["plan"].s_names, info["plan"].o_names
        st_map: dict[str, Any] = {b.pool_key: raw_in, b.pt_key: page_table}
        if has_rec:
            st_map["ckpt_indices"] = ckpt_indices
            st_map["read_offsets"] = read_offsets
        if act_mode == "ref":
            st_map["active_seqs"] = act_mask
    else:
        gathered = (
            pb.gather_states(ex_paged, raw_in, ckpt_indices, read_offsets, active_seqs)
            if (has_rec and pb.gather_states is not None)
            else {}
        )
        call, s_names, o_names = b.call, b.s_names, b.o_names
        st_map = {**gathered, b.pool_key: raw_in, b.pt_key: page_table}

    out_vals = call(pos, hidden, *w, *[st_map[k] for k in s_names], *extra_tuple)
    if not isinstance(out_vals, (tuple, list)):
        out_vals = (out_vals,)
    out_map = dict(zip(o_names, out_vals))
    if b.use_fused:
        raw_out = tuple(out_map.get(b.pool_key, raw_in))
    else:
        ex_scatter = (
            with_cache(ex_paged, pool_gather=False, fused_pool_io=False)
            if b.use_gpr and pb.layout.conv_itemsize == 4
            else ex_paged
        )
        raw_out = pb.scatter_states(
            ex_scatter,
            raw_in,
            pos,
            page_table,
            out_map,
            ckpt_indices if has_rec else None,
            active_seqs,
        )
    if raw_6d:
        raw_out = tuple(jnp.expand_dims(r, 0) for r in raw_out)
    return out_map["hidden"], tuple(raw_out)


def make_flat_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = True,
    fused_pool_io: bool | str | None = None,
    max_b_low: int = FUSED_POOL_IO_MAX_B,
    min_b_hi: int = FUSED_POOL_IO_MIN_B_HI,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    """Build a flat-signature `shard_map` step for `torch_tpu` bridge integration.

    Flat input signature:
      `(pos, hidden, *w_flat, *raw_pools, page_table, ckpt_indices, read_offsets)`
    Flat output signature:
      `(out_hidden, *raw_pools_out)`, each pool output shaped like its input.

    The metadata is replicated (`P()`): `page_table` is `(NS, K)` or
    `(1, NS, K)`; rank-folded metadata is not accepted. Sequence activity is
    `CacheExec.act_mode` (`"from_ckpt"`, read as `"all"` without recurrent
    pools, or `"all"`); `"ref"` needs an `active_seqs` input, which this
    signature does not have, so it raises `MegakernelConfigError` before any
    kernel is built. The per-rank body is `run_local`, shared with
    `make_pooled_step`.
    """
    if cache_exec(ex).act_mode == "ref":
        raise MegakernelConfigError(
            "make_flat_pooled_step: CacheExec.act_mode='ref' needs an active_seqs "
            "mask, which the flat signature does not take; use 'from_ckpt' or 'all'"
        )
    b = build_pooled(
        mesh,
        model_spec,
        ex,
        donate_pools=donate_pools,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )
    act_mode = default_act_mode(b)
    if b.n_extra:
        names = [x.name for x in b.extras]
        raise MegakernelConfigError(
            f"make_flat_pooled_step: the flat signature has no module extra "
            f"inputs, but ModelSpec {b.ms.name!r} declares {names}"
        )
    w_tuple_lens = [
        len(b.wsh[k]) if kernel.is_tupled_weight(b.wsh[k]) else 1 for k in b.w_names
    ]
    n_w_flat = sum(w_tuple_lens)
    num_pools = b.pb.num_pools
    pool_start = 2 + n_w_flat
    donate_argnums = (
        tuple(range(pool_start, pool_start + num_pools)) if donate_pools else ()
    )
    in_specs = (
        (P(), P("tp"))
        + (P("tp"),) * n_w_flat
        + (P("tp"),) * num_pools
        + (P(), P(), P())
    )
    out_specs = (P("tp"),) + (P("tp"),) * num_pools

    def _body(pos: Any, hidden: Any, *rest: Any) -> tuple[Any, ...]:
        weights: dict[str, Any] = {}
        i = 0
        for k, t_len in zip(b.w_names, w_tuple_lens):
            tupled = kernel.is_tupled_weight(b.wsh[k])
            weights[k] = tuple(rest[i : i + t_len]) if tupled else rest[i]
            i += t_len
        raw_pools = tuple(rest[n_w_flat : n_w_flat + num_pools])
        page_table, ckpt_indices, read_offsets = rest[
            n_w_flat + num_pools : n_w_flat + num_pools + 3
        ]
        validate_page_table_shape(page_table, b.ex_paged, rank_folded=False)
        hid, raw_out = run_local(
            b,
            pos,
            hidden,
            weights,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            (),
            act_mode,
        )
        return (hid, *raw_out)

    smap = jax.jit(
        jax.shard_map(
            _body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        ),
        donate_argnums=donate_argnums,
    )
    return smap, in_specs, out_specs, donate_argnums
