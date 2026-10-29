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
"""Declaration-driven synthetic weight and logical/packed input generators."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from .. import kernel, pool
from ..decls import HFParamDecl
from ..errors import MegakernelConfigError
from ..exec_spec import ExecSpec
from ..model_def import ModelDef
from ..ops import BF16, F32, FP8, I32
from ..pool.config import POOL_STATE_NAME, cache_exec, resolved_max_pages_per_seq
from ..spec import SwitchSlot
from .logical import (
    load_logical_weights,
    pack_logical_weights,
    resolve_spec_and_norms,
    switch_branch_for_layer,
)
from .spec_inputs import (
    pack_spec_inputs,
    random_spec_inputs,
    random_spec_logical_weights,
)


def _resolve_model_and_geom(
    g: Any, model_def: Any = None
) -> tuple[type[ModelDef], Any]:
    if isinstance(g, type) and issubclass(g, ModelDef):
        return g, g.derive_geometry()
    m_def = getattr(g, "_model_def", None) if model_def is None else model_def
    if not (isinstance(m_def, type) and issubclass(m_def, ModelDef)):
        raise MegakernelConfigError(
            "A ModelDef is required: pass `model_def=...` or a geometry derived "
            f"from a ModelDef (`g._model_def`), got {g!r}"
        )
    return m_def, g


def _module_layer_counts(ms: Any, L: int) -> dict[str, int]:
    counts: dict[str, int] = {m.name: 0 for m in ms.all_modules()}
    for s in ms.layer_type.slots:
        if isinstance(s, SwitchSlot):
            for lyr in range(L):
                b_mod, _ = switch_branch_for_layer(s, lyr)
                counts[b_mod.name] = counts.get(b_mod.name, 0) + 1
        else:
            counts[s.name] = counts.get(s.name, 0) + L
    return counts


def random_hf_weights(
    key: jax.Array, g: Any, *, model_def: Any = None
) -> dict[str, jax.Array]:
    """Generate unsharded HF weights (`[L, *d.hf_shape]`) from `hf_params()`.

    Draws every declared parameter at its `HFParamDecl.hf_shape` (sharing one
    drawn source tensor across declarations that slice the same `split_from`
    HF checkpoint key) and emits only declared parameter names (`d.name`).
    """
    m_def, geom = _resolve_model_and_geom(g, model_def)
    ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(m_def, None, g=geom)
    L = int(geom.L)
    counts = _module_layer_counts(ms, L)

    active_mods = [
        (m, counts[m.name]) for m in ms.all_modules() if counts.get(m.name, 0) > 0
    ]
    total_decls = 3 + sum(len(m.hf_params(g=geom)) for m, _ in active_mods)
    it = iter(jax.random.split(key, max(16, total_decls)))

    def _draw(d: HFParamDecl, lead: tuple[int, ...]) -> jax.Array:
        hf_sh = d.shape if d.hf_shape is None else d.hf_shape
        sh = (*lead, *hf_sh)
        if d.transform.scale_for is not None:
            return jnp.abs(jax.random.normal(next(it), sh, dtype=F32)) * 0.02 + 0.01
        if d.transform.quantize is not None or jnp.dtype(d.dtype) == jnp.dtype(FP8):
            return (jax.random.normal(next(it), sh, dtype=F32) * 0.25).astype(FP8)
        mean = 0.0 if d.init is None else float(d.init.mean)
        std = 0.05 if d.init is None else float(d.init.std)
        raw = jax.random.normal(next(it), sh, dtype=d.dtype) * std
        return jnp.ones(sh, dtype=d.dtype) + raw if mean == 1.0 else raw + mean

    hf: dict[str, jax.Array] = {}
    for nm in (pre_n, post_n):
        if nm is not None:
            nd = nm.hf_params(g=geom)[0]
            hf[nd.name] = _draw(nd, (L,))
    if final_n is not None:
        fd = final_n.hf_params(g=geom)[0]
        hf[fd.name] = _draw(fd, ())

    split_sources: dict[tuple[str, str], jax.Array] = {}
    for mod, n_l in active_mods:
        for d in mod.hf_params(g=geom):
            if d.name in hf:
                continue
            if d.transform.split_from is not None:
                src_key = (mod.name, d.transform.split_from)
                if src_key not in split_sources:
                    split_sources[src_key] = _draw(d, (n_l,))
                hf[d.name] = split_sources[src_key]
            else:
                hf[d.name] = _draw(d, (n_l,))
    return hf


def random_logical_inputs(
    key: jax.Array, g: Any, ex: ExecSpec, *, model_def: Any = None
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate declaration-driven logical inputs for `reference_step`."""
    m_def, geom = _resolve_model_and_geom(g, model_def)
    geom.check(ex)
    ms, geom, _, _, _ = resolve_spec_and_norms(m_def, ex, g=geom)
    counts = _module_layer_counts(ms, int(geom.L))
    mods = [m for m in ms.all_modules() if counts.get(m.name, 0) > 0]
    keys = jax.random.split(key, 2 + len(mods))
    k_hf, k_hid, *k_mods = keys

    base_pos = jnp.arange(1, ex.NS + 1, dtype=I32) * min(16, ex.ctx // 4)
    pos = (
        base_pos
        if ex.T == 1
        else (base_pos[:, None] + jnp.arange(ex.T, dtype=I32)[None, :]).reshape(ex.B)
    )
    hidden = jax.random.normal(k_hid, (ex.B, int(geom.H)), dtype=BF16) * 0.1
    hf = random_hf_weights(k_hf, geom, model_def=m_def)
    logical_weights = load_logical_weights(m_def, ex, hf, g=geom)

    logical_states: dict[str, Any] = {}
    for k_m, mod in zip(k_mods, mods):
        logical_states.update(mod.init_logical_states(k_m, geom, ex, counts[mod.name]))
    return pos, hidden, logical_weights, logical_states


def pack_inputs(
    pos: jax.Array,
    hidden: jax.Array,
    logical_weights: dict[str, Any],
    logical_states: dict[str, Any],
    g: Any,
    ex: ExecSpec,
    *,
    mesh: Any = None,
    build_default_pool: bool = False,
    model_def: Any = None,
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Pack logical model weights and states into the tiled/folded megakernel layout."""
    m_def, geom = _resolve_model_and_geom(g, model_def)
    ms, geom, _, _, _ = resolve_spec_and_norms(m_def, ex, g=geom)
    counts = _module_layer_counts(ms, int(geom.L))
    R, B, NS, M = ex.R, ex.B, ex.NS, ex.M
    hid_pad = jnp.pad(hidden[:B], ((0, M - B), (0, 0)))
    hid_mk = jnp.tile(hid_pad[None, :, :], (R, 1, 1)).reshape(R * M, int(geom.H))
    weights_mk = pack_logical_weights(m_def, ex, logical_weights, g=geom)

    states_mk: dict[str, Any] = {}
    for mod in ms.all_modules():
        if counts.get(mod.name, 0) > 0:
            states_mk.update(mod.pack_logical_states(logical_states, geom, ex))

    if cache_exec(ex).paged_kv:
        if POOL_STATE_NAME in logical_states and "page_table" in logical_states:
            states_mk[POOL_STATE_NAME] = tuple(
                kernel.fold(r) if r.ndim == 6 else r
                for r in logical_states[POOL_STATE_NAME]
            )
            pt_in = logical_states["page_table"]
            max_pt = resolved_max_pages_per_seq(ex)
            if pt_in.ndim == 2:
                if pt_in.shape[0] not in (NS, R * NS) or pt_in.shape[1] < max_pt:
                    raise ValueError(f"page_table has shape {pt_in.shape}")
                pt_s = pt_in[:, :max_pt]
                states_mk["page_table"] = (
                    pt_s if pt_s.shape[0] == R * NS else jnp.tile(pt_s, (R, 1))
                )
            elif pt_in.ndim == 3:
                if pt_in.shape[:2] != (R, NS) or pt_in.shape[2] < max_pt:
                    raise ValueError(f"page_table has shape {pt_in.shape}")
                states_mk["page_table"] = kernel.fold(pt_in[:, :, :max_pt])
            else:
                raise ValueError(f"page_table must be 2D or 3D, got {pt_in.shape}")
        elif build_default_pool:
            lay = m_def.pool_layout(geom, R=R)
            pt, ckpt, roff, _ = pool.allocate_test_layout(lay, ex)
            states_mk[POOL_STATE_NAME] = pool.states_to_pool(
                lay, ex, logical_states, pt, ckpt, roff
            )
            states_mk["page_table"] = jnp.tile(
                pt[:, : resolved_max_pages_per_seq(ex)], (R, 1)
            )
        else:
            raise ValueError(
                f"paged_kv=True requires {POOL_STATE_NAME!r} and 'page_table'"
            )

    if mesh is not None:
        rep, tp = NamedSharding(mesh, P()), NamedSharding(mesh, P("tp"))
        pos, hid_mk = jax.device_put(pos, rep), jax.device_put(hid_mk, tp)

        def _put(d: dict[str, Any]) -> dict[str, Any]:
            return {
                k: tuple(jax.device_put(x, tp) for x in v)
                if isinstance(v, tuple)
                else jax.device_put(v, tp)
                for k, v in d.items()
            }

        weights_mk, states_mk = _put(weights_mk), _put(states_mk)
    return pos, hid_mk, weights_mk, states_mk


def random_inputs(
    key: jax.Array, g: Any, ex: ExecSpec, *, mesh: Any = None, model_def: Any = None
) -> tuple[jax.Array, jax.Array, dict[str, Any], dict[str, Any]]:
    """Generate packed megakernel inputs `(pos, hidden, weights, states)`."""
    pos, hid_l, w_l, st_l = random_logical_inputs(key, g, ex, model_def=model_def)
    return pack_inputs(
        pos,
        hid_l,
        w_l,
        st_l,
        g,
        ex,
        mesh=mesh,
        build_default_pool=cache_exec(ex).paged_kv,
        model_def=model_def,
    )


__all__ = [
    "pack_inputs",
    "pack_spec_inputs",
    "random_hf_weights",
    "random_inputs",
    "random_logical_inputs",
    "random_spec_inputs",
    "random_spec_logical_weights",
]
