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
"""Pooled step (`make_pooled_step`): `jax.jit` + `shard_map` over block pools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax

from . import kernel
from .cache_common import (
    PooledStepFn,
    require_fused_donation,
    require_pool_backend,
    resolve_active_seqs_mode,
    resolve_model_spec,
    should_use_pool_gather,
    unwrap_active_seqs,
    wrap_active_seqs,
)
from .cache_experimental import should_fuse_pool_io
from .cache_pooled_local import raw_step
from .exec_spec import ExecSpec
from .planner import extra_inputs
from .pool import check_ckpt_bounds
from .pool.config import (
    POOL_STATE_NAME,
    PoolBackend,
    cache_exec,
    resolved_max_pages_per_seq,
    with_cache,
)
from .spec import ModelSpec


@dataclass(frozen=True)
class PooledBuild:
    """What one `make_pooled_step` build fixes: the paged `ExecSpec`, the
    compiled kernel (`call`, `p` = its plan) and the pool argument names."""

    mesh: Any
    ms: ModelSpec
    pb: PoolBackend
    ex_paged: ExecSpec
    use_fused: bool
    use_gpr: bool
    extras: Any
    n_extra: int
    get_compiled: Any
    call: Any
    p: Any
    w_names: Any
    s_names: Any
    o_names: Any
    wsh: Any
    has_rec: bool
    rec_slots: int
    pool_key: str
    pt_key: str
    max_pt: int


def make_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = False,
    fused_pool_io: bool | str | None = None,
    max_b_low: int = 24,
    min_b_hi: int = 56,
) -> Any:
    """Build the `jax.jit` `shard_map` step over unified block pools.

    All calling conventions route through a single `jax.jit` function with
    `donate_argnums=(3,) if donate_pools else ()`, where argument 3 is always
    `raw_pools: tuple[jax.Array, ...]`:
      1. States dict (`PallasWriter.make_step` / `make_step` compatible):
         `hid, states = step(pos, hidden, weights, states, [*extra_inputs])`
      2. Recurrent + attention positional:
         `hid, raw_pools = step(pos, hidden, weights, raw_pools, page_table,
                                ckpt_indices, read_offsets, [*extra_inputs],
                                [active_seqs])`
      3. Attention-only (`not cache_backend.has_recurrent_pool`) positional:
         `hid, raw_pools = step(pos, hidden, weights, raw_pools, page_table,
                                [*extra_inputs], [active_seqs])`
    """
    ms = resolve_model_spec(model_spec)
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
    # Block extra inputs (`Block.extra_inputs`) follow the pool arguments.
    extras = extra_inputs(ms, ex_paged)
    n_extra = len(extras)
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
    info: dict[str, Any] = {**info, "use_fused": use_fused, "use_gpr": use_gpr}
    p: Any = info["plan"]
    w_names = p.w_names
    s_names = p.s_names
    o_names = p.o_names
    wsh = info["wshape"]
    has_rec = pb.has_recurrent_pool
    rec_slots = pb.recurrent_slots_per_pool
    pool_key = POOL_STATE_NAME
    pt_key = pb.page_table_state_name
    max_pt = resolved_max_pages_per_seq(ex_paged)
    b = PooledBuild(
        mesh=mesh,
        ms=ms,
        pb=pb,
        ex_paged=ex_paged,
        use_fused=use_fused,
        use_gpr=use_gpr,
        extras=extras,
        n_extra=n_extra,
        get_compiled=_get_compiled,
        call=call,
        p=p,
        w_names=w_names,
        s_names=s_names,
        o_names=o_names,
        wsh=wsh,
        has_rec=has_rec,
        rec_slots=rec_slots,
        pool_key=pool_key,
        pt_key=pt_key,
        max_pt=max_pt,
    )

    def _raw_step(
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
        return raw_step(
            b,
            pos,
            hidden,
            weights,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_wrapped,
            *extra_tuple,
        )

    _jitted_step = jax.jit(
        _raw_step,
        donate_argnums=(3,) if donate_pools else (),
    )

    def _call(
        pos: Any, hidden: Any, weights: dict[str, Any], states_or_raw: Any, *rest: Any
    ) -> Any:
        (
            is_dict,
            states_dict,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            extra_tuple,
        ) = parse_call_args(b, states_or_raw, rest)
        if has_rec and cache_exec(ex_paged).check_ckpt_bounds:
            act_m, act_arr = resolve_active_seqs_mode(
                unwrap_active_seqs(active_seqs),
                default_mode="from_ckpt",
            )
            check_ckpt_bounds(
                ckpt_indices,
                cache_exec(ex_paged).num_pages,
                pb.layout,
                act_mode=act_m,
                active_mask=act_arr,
            )
        hid, new_raw = _jitted_step(
            pos,
            hidden,
            weights,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            *extra_tuple,
        )
        if is_dict:
            assert states_dict is not None
            return hid, {**states_dict, pool_key: new_raw}
        return hid, new_raw

    def _lower(
        pos: Any, hidden: Any, weights: dict[str, Any], states_or_raw: Any, *rest: Any
    ) -> Any:
        (
            _,
            _,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            extra_tuple,
        ) = parse_call_args(b, states_or_raw, rest)
        return _jitted_step.lower(
            pos,
            hidden,
            weights,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            *extra_tuple,
        )

    return PooledStepFn(_call, _lower, info)


def parse_call_args(
    b: PooledBuild, states_or_raw: Any, rest: tuple[Any, ...]
) -> tuple[
    bool,
    dict[str, Any] | None,
    tuple[jax.Array, ...],
    jax.Array,
    jax.Array | None,
    jax.Array | None,
    Any,
    tuple[Any, ...],
]:
    """Split the `states`-dict or positional call forms into step arguments."""
    n_extra, has_rec, pool_key, pt_key = b.n_extra, b.has_rec, b.pool_key, b.pt_key
    if isinstance(states_or_raw, dict):
        states = states_or_raw
        raw_pools = tuple(states[pool_key])
        page_table = states[pt_key]
        ckpt_indices = states.get("ckpt_indices")
        read_offsets = states.get("read_offsets")
        active_seqs = wrap_active_seqs(states.get("active_seqs"))
        extra_tuple = tuple(rest[:n_extra])
        return (
            True,
            states,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            active_seqs,
            extra_tuple,
        )

    raw_pools = tuple(states_or_raw)
    page_table = rest[0]
    if has_rec:
        ckpt_indices = rest[1]
        read_offsets = rest[2]
        tail = rest[3:]
    else:
        ckpt_indices = None
        read_offsets = None
        tail = rest[1:]

    extra_tuple = tuple(tail[:n_extra])
    active_seqs = wrap_active_seqs(tail[n_extra] if len(tail) > n_extra else None)

    return (
        False,
        None,
        raw_pools,
        page_table,
        ckpt_indices,
        read_offsets,
        active_seqs,
        extra_tuple,
    )
