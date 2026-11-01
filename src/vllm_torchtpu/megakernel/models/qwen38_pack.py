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
"""Pack per-rank logical arrays into `kernel.make_step` inputs for any `ModelSpec`.

Inputs are keyed by the plan's own names, so the packer needs no per-block
knowledge:

- `weights[name]`: `[R, l, *lead, K, N]` for stacked weights, or a tuple of
  `l` per-layer `[R, *lead, K, N]` arrays; `K` is split into the plan's
  `(nt, tk)` tiles. `es` (expert scales) is passed already packed.
- `smalls[field]`: `[R, L, w]` or `[L, w]` (replicated), written at the
  field's offset in the `smalls[L, 1, S]` table; absent fields stay zero.
- `states[name]`: per-rank `[R, ...]` arrays, folded to `[R * N0, ...]`.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .. import kernel, planner
from ..exec_spec import ExecSpec
from ..ops import F32
from ..spec import ModelSpec

_PASSTHROUGH = ("es",)


def _tile(arr: jax.Array, nt: int, tk: int) -> jax.Array:
    *lead, k_in, n_in = arr.shape
    assert k_in <= nt * tk, (arr.shape, nt, tk)
    if k_in < nt * tk:
        pad = [(0, 0)] * len(lead) + [(0, nt * tk - k_in), (0, 0)]
        arr = jnp.pad(arr, pad)
    return arr.reshape((*lead, nt, tk, n_in))


def pack_smalls(
    spec: ModelSpec, ex: ExecSpec, smalls: dict[str, Any], p: Any = None
) -> jax.Array:
    """`[R, L, 1, S]` smalls table from `smalls[field]`."""
    p = planner.plan(spec, ex) if p is None else p
    R, L = ex.R, spec.config.L
    sm = jnp.zeros((R, L, 1, p.smalls_width), F32)
    for name, (o, w) in p.smalls_layout.items():
        if name not in smalls:
            continue
        val = jnp.asarray(smalls[name], F32)
        if val.ndim == 2:
            val = jnp.broadcast_to(val[None], (R, *val.shape))
        assert val.shape[:2] == (R, L), (name, val.shape, R, L)
        vw = min(w, val.shape[-1])
        sm = sm.at[:, :, 0, o : o + vw].set(val[:, :, :vw])
    return sm


def pack_weights(
    spec: ModelSpec,
    ex: ExecSpec,
    weights: dict[str, Any],
    smalls: dict[str, Any],
    final_norm: Any = None,
) -> dict[str, Any]:
    """Per-rank (unsharded, leading `R`) weight operands in `plan.w_names` order."""
    p = planner.plan(spec, ex)
    g = spec.config
    out: dict[str, Any] = {}
    for name in p.w_names:
        if name == "smalls":
            out[name] = pack_smalls(spec, ex, smalls, p)
        elif name == "final_norm":
            fn = jnp.zeros((g.H,), F32) if final_norm is None else final_norm
            out[name] = jnp.broadcast_to(
                jnp.asarray(fn, F32)[None, None, :], (ex.R, 1, g.H)
            )
        elif name in _PASSTHROUGH or name not in p.nt:
            out[name] = weights[name]
        else:
            nt, tk = p.nt[name], p.tk[name]
            dt = p.weight_decls[name].dtype
            w = weights[name]
            if isinstance(w, tuple):
                out[name] = tuple(_tile(x, nt, tk).astype(dt) for x in w)
            else:
                out[name] = _tile(w, nt, tk).astype(dt)
    return out


def shard_step_inputs(
    mesh: Any,
    ex: ExecSpec,
    pos: Any,
    hidden: Any,
    weights: dict[str, Any],
    states: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
    """Fold leading-`R` arrays and place them `P('tp')`; `pos` is replicated."""
    tp = NamedSharding(mesh, P("tp"))

    def put(x: Any) -> Any:
        if isinstance(x, tuple):
            return tuple(put(v) for v in x)
        return jax.device_put(kernel.fold(x), tp)

    M = ex.M
    hid = jnp.pad(hidden[: ex.B], ((0, M - ex.B), (0, 0)))
    hid = jnp.broadcast_to(hid[None], (ex.R, *hid.shape))
    return (
        jax.device_put(pos, NamedSharding(mesh, P())),
        put(hid),
        {k: put(v) for k, v in weights.items()},
        {k: put(v) for k, v in states.items()},
    )
