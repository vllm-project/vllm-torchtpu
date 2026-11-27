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

from typing import Any

import jax

from .cache_common import PooledBuild, build_pooled, default_act_mode
from .cache_pooled_local import (
    ModeTag,
    check_ckpt_bounds_per_rank,
    raw_step,
    wrap_active_seqs,
)
from .errors import MegakernelConfigError
from .exec_spec import ExecSpec
from .pool.config import FUSED_POOL_IO_MAX_B, FUSED_POOL_IO_MIN_B_HI, cache_exec


class PooledStepFn:
    """Callable step function that unpacks as `(step_fn, info)` and has `.lower()`."""

    def __init__(self, call_fn: Any, lower_fn: Any, info: dict[str, Any]):
        self._call_fn = call_fn
        self._lower_fn = lower_fn
        self.info = info

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._call_fn(*args, **kwargs)

    def lower(self, *args: Any, **kwargs: Any) -> Any:
        return self._lower_fn(*args, **kwargs)

    def __iter__(self) -> Any:
        yield self
        yield self.info


def make_pooled_step(
    mesh: Any,
    model_spec: Any,
    ex: ExecSpec,
    *,
    donate_pools: bool = False,
    fused_pool_io: bool | str | None = None,
    max_b_low: int = FUSED_POOL_IO_MAX_B,
    min_b_hi: int = FUSED_POOL_IO_MIN_B_HI,
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
    `active_seqs` is a mask array or a mode (`"from_ckpt"` / `"all"`); without
    one the step runs `CacheExec.act_mode`, and `"ref"` raises
    `MegakernelConfigError`. The per-rank body is `cache_flat.run_local`,
    shared with `make_flat_pooled_step`.
    """
    b = build_pooled(
        mesh,
        model_spec,
        ex,
        donate_pools=donate_pools,
        fused_pool_io=fused_pool_io,
        max_b_low=max_b_low,
        min_b_hi=min_b_hi,
    )

    def _raw_step(*args: Any) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return raw_step(b, *args)

    _jitted_step = jax.jit(_raw_step, donate_argnums=(3,) if donate_pools else ())

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
        if b.has_rec and cache_exec(b.ex_paged).check_ckpt_bounds:
            check_ckpt_bounds_per_rank(
                b, page_table, ckpt_indices, read_offsets, active_seqs
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
            return hid, {**states_dict, b.pool_key: new_raw}
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

    return PooledStepFn(_call, _lower, b.info)


def _default_active_seqs(b: PooledBuild) -> ModeTag:
    """`active_seqs` of a call that passes none: `CacheExec.act_mode`."""
    mode = default_act_mode(b)
    if mode == "ref":
        raise MegakernelConfigError(
            "pooled step: CacheExec.act_mode='ref' needs an active_seqs mask; "
            "pass one, or build with act_mode 'from_ckpt' or 'all'"
        )
    return ModeTag(mode)


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
    """Split the `states`-dict or positional call forms into step arguments.

    Without `active_seqs`, the step runs `CacheExec.act_mode` (see
    `default_act_mode`); `"ref"` then raises `MegakernelConfigError`.
    """
    n_extra, has_rec, pool_key, pt_key = b.n_extra, b.has_rec, b.pool_key, b.pt_key
    if isinstance(states_or_raw, dict):
        states = states_or_raw
        raw_pools = tuple(states[pool_key])
        page_table = states[pt_key]
        ckpt_indices = states.get("ckpt_indices")
        read_offsets = states.get("read_offsets")
        active_seqs = states.get("active_seqs")
        if len(rest) > n_extra:
            raise TypeError(
                f"pooled step (states dict form) takes {n_extra} extra inputs "
                f"after the states, got {len(rest)} trailing arguments"
            )
        extra_tuple = tuple(rest[:n_extra])
        return (
            True,
            states,
            raw_pools,
            page_table,
            ckpt_indices,
            read_offsets,
            _default_active_seqs(b)
            if active_seqs is None
            else wrap_active_seqs(active_seqs),
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

    if len(tail) > n_extra + 1:
        raise TypeError(
            f"pooled step takes {n_extra} extra inputs and an optional "
            f"active_seqs after the pool metadata, got {len(tail)} arguments"
        )
    extra_tuple = tuple(tail[:n_extra])
    active_seqs = (
        wrap_active_seqs(tail[n_extra])
        if len(tail) > n_extra and tail[n_extra] is not None
        else _default_active_seqs(b)
    )

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
