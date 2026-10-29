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
"""Flat-signature pooled step (`make_flat_pooled_step`) for bridge integration."""

from __future__ import annotations

from typing import Any

import jax
from jax.sharding import PartitionSpec as P

from . import kernel
from .cache_common import (
    require_fused_donation,
    require_pool_backend,
    resolve_model_spec,
    should_use_pool_gather,
    validate_page_table_shape,
)
from .cache_experimental import should_fuse_pool_io
from .exec_spec import ExecSpec
from .pool import check_ckpt_bounds
from .pool.config import (
    POOL_STATE_NAME,
    cache_exec,
    resolved_max_pages_per_seq,
    with_cache,
)


def make_flat_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = True,
    fused_pool_io: bool | str | None = None,
    max_b_low: int = 24,
    min_b_hi: int = 56,
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...], tuple[int, ...]]:
    """Build a flat-signature `shard_map` step for `torch_xla` bridge integration.

    Flat input signature:
      `(pos, hidden, *w_flat, *raw_pools, page_table, ckpt_indices, read_offsets)`
    Flat output signature:
      `(out_hidden, *raw_pools_out)`
    """
    ms = resolve_model_spec(model_spec, R=ex.R)
    pb = require_pool_backend(ms)
    pb.layout.validate(ex)
    use_fused = should_fuse_pool_io(
        ms,
        ex,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )
    use_gpr = should_use_pool_gather(ms, pb, ex, use_fused=use_fused)
    require_fused_donation(ms, use_fused, donate_pools)
    ex_paged = with_cache(
        ex,
        paged_kv=True,
        fused_pool_io=use_fused,
        pool_gather=use_gpr,
    )
    pb.layout.validate(ex_paged)
    _build_cache: dict[tuple[str, bool, int, bool], tuple[Any, dict[str, Any]]] = {}

    def _get_compiled(
        act_mode: str, ckpt_2d: bool, ckpt_k: int, roff_1d: bool
    ) -> tuple[Any, dict[str, Any]]:
        key = (act_mode, ckpt_2d, ckpt_k, roff_1d)
        if key not in _build_cache:
            ex_k = with_cache(
                ex_paged,
                act_mode=act_mode,
                ckpt_2d=ckpt_2d,
                ckpt_k=ckpt_k,
                roff_1d=roff_1d,
            )
            _build_cache[key] = kernel.build(ms, ex_k)
        return _build_cache[key]

    cx = cache_exec(ex_paged)
    default_ckpt_k = cx.ckpt_k if cx.ckpt_k > 0 else ex_paged.T
    call, info = _get_compiled(cx.act_mode, cx.ckpt_2d, default_ckpt_k, cx.roff_1d)
    p = info["plan"]
    w_names = p.w_names
    s_names = p.s_names
    o_names = p.o_names
    wsh = info["wshape"]
    num_pools = pb.num_pools
    has_rec = pb.has_recurrent_pool

    w_tuple_lens = [
        len(wsh[k]) if kernel.is_tupled_weight(wsh[k]) else 1 for k in w_names
    ]
    n_w_flat = sum(w_tuple_lens)
    pool_start = 2 + n_w_flat
    pool_end = pool_start + num_pools
    donate_argnums = tuple(range(pool_start, pool_end)) if donate_pools else ()

    in_specs = (
        (P(), P("tp"))
        + (P("tp"),) * n_w_flat
        + (P("tp"),) * num_pools
        + (P(), P(), P())
    )
    out_specs = (P("tp"),) + (P("tp"),) * num_pools
    max_pt = resolved_max_pages_per_seq(ex_paged)

    def _body(pos: Any, hidden: Any, *rest: Any) -> tuple[Any, ...]:
        w_flat = rest[:n_w_flat]
        raw_pools = tuple(
            r.squeeze(0) if r.ndim == 6 else r
            for r in rest[n_w_flat : n_w_flat + num_pools]
        )
        page_table, ckpt_indices, read_offsets = rest[
            n_w_flat + num_pools : n_w_flat + num_pools + 3
        ]

        w_args: list[Any] = []
        w_idx = 0
        for k, t_len in zip(w_names, w_tuple_lens):
            t_sh = kernel.target_weight_shape(wsh[k])
            allow_exp = k in p.dynamic_kinds
            if kernel.is_tupled_weight(wsh[k]):
                w_args.append(
                    tuple(
                        kernel.unpack_local_weight(
                            w_flat[w_idx + i],
                            t_sh,
                            name=k,
                            allow_expert_reshape=allow_exp,
                        )
                        for i in range(t_len)
                    )
                )
            else:
                w_args.append(
                    kernel.unpack_local_weight(
                        w_flat[w_idx],
                        t_sh,
                        name=k,
                        allow_expert_reshape=allow_exp,
                    )
                )
            w_idx += t_len

        validate_page_table_shape(page_table, ex_paged)
        if page_table.ndim == 3:
            page_table = page_table.squeeze(0)
        if page_table.shape[-1] > max_pt:
            page_table = page_table[..., :max_pt]
        if ckpt_indices is not None and ckpt_indices.ndim == 4:
            ckpt_indices = ckpt_indices.squeeze(0)
        if read_offsets is not None and read_offsets.ndim == 3:
            read_offsets = read_offsets.squeeze(0)

        if use_fused or use_gpr:
            act_mode_l = "from_ckpt" if has_rec else "all"
            if has_rec and cache_exec(ex_paged).check_ckpt_bounds:
                check_ckpt_bounds(
                    ckpt_indices,
                    cache_exec(ex_paged).num_pages,
                    pb.layout,
                    act_mode=act_mode_l,
                    read_offsets=read_offsets,
                )
            ckpt_2d_l = bool(ckpt_indices is not None and ckpt_indices.ndim == 2)
            ckpt_k_l = (
                int(ckpt_indices.shape[-1]) if ckpt_indices is not None else ex_paged.T
            )
            roff_1d_l = bool(read_offsets is not None and read_offsets.ndim == 1)
            call_l, info_l = _get_compiled(act_mode_l, ckpt_2d_l, ckpt_k_l, roff_1d_l)
            p_l = info_l["plan"]
            st_map_f: dict[str, Any] = {
                POOL_STATE_NAME: raw_pools,
                pb.page_table_state_name: page_table,
            }
            if has_rec:
                st_map_f["ckpt_indices"] = ckpt_indices
                st_map_f["read_offsets"] = read_offsets
            s_args_f = [st_map_f[k] for k in p_l.s_names]
            out_vals_f = call_l(pos, hidden, *w_args, *s_args_f)
            if not isinstance(out_vals_f, (tuple, list)):
                out_vals_f = (out_vals_f,)
            out_map_f = dict(zip(p_l.o_names, out_vals_f))
            hid = out_map_f["hidden"]
            if use_fused:
                raw_out = (
                    tuple(out_map_f[POOL_STATE_NAME])
                    if POOL_STATE_NAME in out_map_f
                    else tuple(raw_pools)
                )
                return (hid, *raw_out)
            raw_out = pb.scatter_states(
                ex_paged,
                raw_pools,
                pos,
                page_table,
                out_map_f,
                ckpt_indices if has_rec else None,
                act_mode_l if has_rec else None,
            )
            return (hid, *raw_out)

        gathered = (
            pb.gather_states(
                ex_paged,
                raw_pools,
                ckpt_indices,
                read_offsets,
                "from_ckpt",
            )
            if (has_rec and pb.gather_states is not None)
            else {}
        )

        st_map: dict[str, Any] = {
            **gathered,
            POOL_STATE_NAME: raw_pools,
            pb.page_table_state_name: page_table,
        }
        s_args = [st_map[k] for k in s_names]
        out_vals = call(pos, hidden, *w_args, *s_args)
        if not isinstance(out_vals, (tuple, list)):
            out_vals = (out_vals,)
        out_map = dict(zip(o_names, out_vals))
        hid = out_map["hidden"]

        raw_out = pb.scatter_states(
            ex_paged,
            raw_pools,
            pos,
            page_table,
            out_map,
            ckpt_indices if has_rec else None,
            "from_ckpt" if has_rec else None,
        )
        return (hid, *raw_out)

    smap = jax.jit(
        jax.shard_map(
            _body, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=False
        ),
        donate_argnums=donate_argnums,
    )
    return smap, in_specs, out_specs, donate_argnums
