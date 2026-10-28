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
"""Per-rank generic weight loader (`load_weights`) over any `WeightSource`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .. import planner
from ..decls import HFParamDecl, VLLMAlias
from ..exec_spec import ExecSpec
from ..module import Module
from ..ops import F32, rup
from ..spec import SwitchSlot
from .eplb import resolve_mapping_ecopy
from .logical import resolve_spec_and_norms, switch_branch_for_layer
from .ops import ArrayOps, JaxArrayOps, WeightSource


def _has_rank_axis(d: HFParamDecl) -> bool:
    tf = d.transform
    return (
        tf.shard_axis is not None or tf.channel_splits is not None or tf.replicate_ranks
    )


class MappingWeightSource:
    """`WeightSource` backed by an in-memory `Mapping` of logical arrays."""

    def __init__(
        self,
        logical_weights: Mapping[str, Any],
        *,
        ops: ArrayOps | None = None,
    ) -> None:
        self._w = logical_weights
        self._ops: ArrayOps = JaxArrayOps() if ops is None else ops

    @property
    def ops(self) -> ArrayOps:
        return self._ops

    def has_layer_alias(self, lyr: int, ord_idx: int, source_name: str) -> bool:
        del lyr, ord_idx
        return source_name in self._w

    def resolve_alias(
        self,
        lyr: int,
        ord_idx: int,
        alias: VLLMAlias,
        target_decls: Sequence[HFParamDecl],
        scale_decls: Mapping[str, HFParamDecl],
        *,
        R: int,
        tp_rank: int,
        fuse_groups: Mapping[str, tuple[str, ...]] | None = None,
    ) -> dict[str, Any]:
        del lyr, alias, R, fuse_groups
        out: dict[str, Any] = {}
        for d in target_decls:
            key = d.name if d.name in self._w else (d.transform.fuse_into or d.name)
            arr = self._w[key]
            out[key] = arr[tp_rank, ord_idx] if _has_rank_axis(d) else arr[ord_idx]
        for sn, sd in scale_decls.items():
            key = sn if sn in self._w else (sd.transform.fuse_into or sn)
            if key in self._w:
                arr = self._w[key]
                out[key] = arr[tp_rank, ord_idx] if _has_rank_axis(sd) else arr[ord_idx]
        return out

    def read_model_param(self, decl: HFParamDecl, *, R: int, tp_rank: int) -> Any:
        del R
        arr = self._w[decl.name]
        return arr[tp_rank] if arr.ndim == len(decl.shape) + 1 else arr

    def read_layer_ecopy(
        self,
        lyr: int,
        g: Any,
        ex: ExecSpec,
        *,
        expert_placement: Any = None,
        experts_path: str = "mlp.experts",
    ) -> Any | None:
        del expert_placement, experts_path
        return resolve_mapping_ecopy(self._w, lyr, g, ex.R)

    def supports_passthrough(self) -> bool:
        return False


def load_weights(
    model_def: Any,
    ex: ExecSpec,
    source: WeightSource,
    *,
    g: Any = None,
    tp_rank: int = 0,
    expert_placement: Any = None,
    merged_tiles: bool = False,
) -> dict[str, Any]:
    """Generic per-rank weight loader over any `WeightSource` via `source.ops`."""
    ops = source.ops
    use_merged_tiles = bool(merged_tiles or getattr(source, "merged_tiles", False))
    ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(model_def, ex, g=g)
    p = planner.plan(ms, ex)
    R, L = ex.R, int(geom.L)
    w_shapes = planner.weight_shapes(ms, ex, p)

    assert final_n is not None, f"ModelSpec {ms.name!r} requires a final_norm module"
    fn_decl = final_n.hf_params(g=geom)[0]
    fn_raw = source.read_model_param(fn_decl, R=R, tp_rank=tp_rank)
    final_norm_w = ops.own(ops.reshape(ops.cast(fn_raw, F32), (1, int(geom.H))))

    per_layer_lists: dict[str, list[Any]] = {
        wd.name: [] for mod in ms.all_modules() for wd in mod.weights(geom, ex)
    }
    smalls_list: list[Any] = []

    for lyr in range(L):
        active_mods: list[tuple[Module, int]] = []
        for nm in (pre_n, post_n):
            if nm is not None:
                active_mods.append((nm, lyr))
        for slot in ms.layer_type.slots:
            if isinstance(slot, SwitchSlot):
                active_mods.append(switch_branch_for_layer(slot, lyr))
            else:
                active_mods.append((slot, lyr))

        small_fields: dict[str, Any] = {}
        experts_path = "mlp.experts"
        for mod, ord_idx in active_mods:
            decls = mod.hf_params(g=geom)
            decl_map = {d.name: d for d in decls}
            fuse_groups: dict[str, list[HFParamDecl]] = {}
            for d in decls:
                if d.transform.fuse_into is not None:
                    fuse_groups.setdefault(d.transform.fuse_into, []).append(d)
                if d.transform.expert_axis is not None and d.vllm_aliases:
                    src_name = d.vllm_aliases[0].source
                    if "." in src_name:
                        experts_path = src_name.rsplit(".", 1)[0]
            for grp in fuse_groups.values():
                grp.sort(key=lambda m: m.transform.fuse_order)
            fuse_names = {
                k: tuple(m.name for m in grp) for k, grp in fuse_groups.items()
            }

            resolved: dict[str, Any] = {}
            for d in decls:
                if d.name in resolved or (
                    d.transform.fuse_into and d.transform.fuse_into in resolved
                ):
                    continue
                chosen: VLLMAlias | None = None
                for al in d.vllm_aliases:
                    if source.has_layer_alias(lyr, ord_idx, al.source):
                        chosen = al
                        break
                if chosen is None:
                    f_in = d.transform.fuse_into
                    if source.has_layer_alias(lyr, ord_idx, d.name) or (
                        f_in is not None and source.has_layer_alias(lyr, ord_idx, f_in)
                    ):
                        chosen = VLLMAlias(source=d.name, targets=(d.name,))
                    else:
                        tried = [al.source for al in d.vllm_aliases] or [d.name]
                        raise ValueError(
                            f"Layer {lyr} ({mod.name}[{ord_idx}]): missing parameter "
                            f"{d.name!r}; tried aliases {tried}"
                        )
                t_decls = [decl_map[tn] for tn in chosen.targets]
                s_decls = {sn: decl_map[sn] for sn in chosen.scale_targets}
                resolved.update(
                    source.resolve_alias(
                        lyr,
                        ord_idx,
                        chosen,
                        t_decls,
                        s_decls,
                        R=R,
                        tp_rank=tp_rank,
                        fuse_groups=fuse_names,
                    )
                )

            for grp_name, members in fuse_groups.items():
                if grp_name not in resolved:
                    ax = members[0].transform.fuse_axis
                    resolved[grp_name] = ops.own(
                        ops.concat([resolved[m.name] for m in members], axis=ax)
                    )

            for d in decls:
                sf_name = d.transform.small_field
                if sf_name is not None:
                    if sf_name not in p.smalls_layout:
                        raise ValueError(
                            f"Module {mod.name!r} HFParamDecl {d.name!r} declares "
                            f"small_field={sf_name!r}, which is not in "
                            f"plan.smalls_layout {list(p.smalls_layout)}"
                        )
                    l_key = d.transform.fuse_into or d.name
                    if l_key not in resolved:
                        raise ValueError(
                            f"Layer {lyr} module {mod.name!r}: small_field "
                            f"{sf_name!r} source {l_key!r} was not resolved"
                        )
                    small_fields[sf_name] = resolved[l_key]

            for sf in mod.smalls(geom, ex):
                if sf.name != "ecopy" and sf.name not in small_fields:
                    raise ValueError(
                        f"Layer {lyr} module {mod.name!r} smalls() declared "
                        f"{sf.name!r}, but no HFParamDecl populated it"
                    )

            for wd in mod.weights(geom, ex):
                raw = (
                    ops.own(ops.concat([resolved[s] for s in wd.fuse_from], axis=-1))
                    if wd.fuse_from
                    else resolved[wd.name]
                )
                if wd.stream_ring in ("static", "dynamic"):
                    exp_raw = (*wd.lead_dims, wd.K, wd.N)
                    got_raw = tuple(int(dim) for dim in raw.shape)
                    if got_raw != exp_raw:
                        raise ValueError(
                            f"Layer {lyr} weight {wd.name!r} has shape {got_raw}, "
                            f"expected {exp_raw}"
                        )
                    is_passthrough = (
                        wd.layout == "per_layer"
                        and wd.stream_ring == "dynamic"
                        and not ex.stack_e
                        and source.supports_passthrough()
                    )
                    if is_passthrough:
                        per_layer_lists[wd.name].append(raw)
                    elif use_merged_tiles:
                        per_layer_lists[wd.name].append(
                            ops.own(raw) if wd.layout == "per_layer" else raw
                        )
                    else:
                        tiled = ops.reshape(
                            raw,
                            (*wd.lead_dims, p.nt[wd.name], p.tk[wd.name], wd.N),
                        )
                        per_layer_lists[wd.name].append(
                            ops.own(tiled) if wd.layout == "per_layer" else tiled
                        )
                else:
                    sh_el = (
                        (*wd.lead_dims, wd.K * wd.N)
                        if (use_merged_tiles and wd.K == 1 and len(wd.lead_dims) >= 1)
                        else (*wd.lead_dims, wd.K, wd.N)
                    )
                    per_layer_lists[wd.name].append(
                        ops.own(
                            ops.reshape(
                                ops.cast(raw, wd.dtype),
                                sh_el,
                            )
                        )
                    )

        if "ecopy" in p.smalls_layout:
            ec = source.read_layer_ecopy(
                lyr,
                geom,
                ex,
                expert_placement=expert_placement,
                experts_path=experts_path,
            )
            if ec is not None:
                small_fields["ecopy"] = ec

        parts: list[Any] = []
        for field_name, (_, width) in p.smalls_layout.items():
            padded_w = rup(width, 128)
            v = small_fields.get(field_name)
            if v is None:
                parts.append(ops.zeros((padded_w,), F32))
            else:
                v_1d = ops.cast(ops.reshape(v, (-1,)), F32)
                n_el = int(v_1d.shape[0])
                if n_el > width:
                    raise ValueError(f"smalls field {field_name!r}: {n_el} > {width}")
                if padded_w > n_el:
                    v_1d = ops.pad(v_1d, ((0, padded_w - n_el),))
                parts.append(v_1d)
        smalls_list.append(
            ops.own(ops.reshape(ops.concat(parts, axis=0), (1, p.smalls_width)))
        )

    weights: dict[str, Any] = {}
    for mod in ms.all_modules():
        for wd in mod.weights(geom, ex):
            xs = per_layer_lists[wd.name]
            if wd.layout == "per_layer":
                weights[wd.name] = tuple(xs)
            elif xs:
                weights[wd.name] = ops.stack_layers(xs)
            else:
                sh, _ = w_shapes[wd.name]
                weights[wd.name] = ops.own(ops.zeros(sh, wd.dtype))

    weights["smalls"] = ops.stack_layers(smalls_list)
    weights["final_norm"] = final_norm_w
    weights["plan"] = p
    w_flat: list[Any] = []
    for k in p.w_names:
        val = weights[k]
        if isinstance(val, (tuple, list)):
            w_flat.extend(val)
        else:
            w_flat.append(val)
    weights["w_flat"] = w_flat
    return weights


__all__ = ["MappingWeightSource", "load_weights"]
