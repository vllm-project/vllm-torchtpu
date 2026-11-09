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
"""Synthetic vLLM-layout `WeightSource` built directly from unsharded HF weights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from ..decls import HFParamDecl, VLLMAlias
from ..exec_spec import ExecSpec
from ..module import Module
from ..ops import BF16, F32, FP8, I32
from ..spec import SwitchSlot
from .eplb import default_eplb_cores, encode_eplb_ecopy
from .logical import (
    _coerce_canonical_hf_entry,
    resolve_spec_and_norms,
    switch_branch_for_layer,
)
from .ops import ArrayOps, JaxArrayOps, slice_alias_targets


def normalize_hf_for_vllm(
    model_def: Any,
    ex: ExecSpec,
    hf: Mapping[str, Any],
    *,
    g: Any = None,
) -> dict[str, jax.Array]:
    """Normalize `per_channel_fp8` HF pairs so on-load FP8 quantization is exact."""
    ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(model_def, ex, g=g)
    decls: list[HFParamDecl] = []
    for nm in (pre_n, post_n, final_n):
        if nm is not None:
            decls.extend(nm.hf_params(g=geom))
    for mod in ms.all_modules():
        decls.extend(mod.hf_params(g=geom))
    scale_for_map = {
        d.transform.scale_for: d for d in decls if d.transform.scale_for is not None
    }
    j_ops = JaxArrayOps()
    out = {k: jnp.asarray(v) for k, v in hf.items()}
    for d in decls:
        if d.transform.quantize == "per_channel_fp8" and d.name in out:
            sd = scale_for_map.get(d.name)
            if sd is not None and sd.name in out:
                w_c = _coerce_canonical_hf_entry(d, out, {}).astype(F32)
                s_f = _coerce_canonical_hf_entry(sd, out, {}).astype(F32)
                bf16 = (w_c * s_f[:, None, :]).astype(BF16)
                qs = [j_ops.quantize_fp8(bf16[i]) for i in range(bf16.shape[0])]
                out[d.name] = jnp.stack([q for q, _ in qs], axis=0)
                out[sd.name] = jnp.stack([s for _, s in qs], axis=0)
    return out


class SyntheticVLLMWeightSource:
    """`WeightSource` constructing vLLM-layout buffers directly from HF weights."""

    def __init__(
        self,
        model_def: Any,
        ex: ExecSpec,
        hf: Mapping[str, Any],
        *,
        g: Any = None,
        e_cores: list[list[list[int]]] | None = None,
        ops: ArrayOps | None = None,
    ) -> None:
        self._ops: ArrayOps = JaxArrayOps() if ops is None else ops
        ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(model_def, ex, g=g)
        self._geom = geom
        self._ex = ex
        R, L = ex.R, int(geom.L)
        er, ew = int(geom.ER(R)), int(geom.EW(R))
        moe_tp = bool(getattr(geom, "moe_tp", True))
        if not moe_tp and ew > er and e_cores is None:
            e_cores = default_eplb_cores(geom, R)
        self._e_cores = e_cores
        self._ecopy = (
            encode_eplb_ecopy(geom, R, e_cores)
            if (not moe_tp and ew > er and e_cores is not None)
            else None
        )

        hf_norm = normalize_hf_for_vllm(model_def, ex, hf, g=geom)
        self._model_params: dict[str, jax.Array] = {}
        if final_n is not None:
            fd = final_n.hf_params(g=geom)[0]
            self._model_params[fd.name] = jnp.asarray(hf_norm[fd.name])

        self._buffers: dict[tuple[int, int, int], dict[str, jax.Array]] = {}
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

            for mod, ord_idx in active_mods:
                self._populate_module_buffers(
                    mod, lyr, ord_idx, hf_norm, R=R, moe_tp=moe_tp, er=er, ew=ew
                )

    def _populate_module_buffers(
        self,
        mod: Module,
        lyr: int,
        ord_idx: int,
        hf_norm: Mapping[str, jax.Array],
        *,
        R: int,
        moe_tp: bool,
        er: int,
        ew: int,
    ) -> None:
        geom = self._geom
        decls = mod.hf_params(g=geom)
        decl_map = {d.name: d for d in decls}
        scale_for_map = {
            d.transform.scale_for: d for d in decls if d.transform.scale_for is not None
        }
        fuse_members: dict[str, list[HFParamDecl]] = {}
        for d in decls:
            if d.transform.fuse_into is not None:
                fuse_members.setdefault(d.transform.fuse_into, []).append(d)
        for grp in fuse_members.values():
            grp.sort(key=lambda m: m.transform.fuse_order)

        chosen_aliases: list[VLLMAlias] = []
        covered: set[str] = set()
        for d in decls:
            if d.name in covered or d.transform.scale_for is not None:
                continue
            chosen: VLLMAlias | None = None
            for al in d.vllm_aliases:
                if all(
                    decl_map[tn].transform.quantize == "per_channel_fp8"
                    or (jnp.dtype(decl_map[tn].dtype) == jnp.dtype(FP8))
                    == al.already_fp8
                    for tn in al.targets
                ):
                    chosen = al
                    break
            if chosen is None and d.vllm_aliases:
                chosen = d.vllm_aliases[-1]
            if chosen is None:
                sd = scale_for_map.get(d.name)
                chosen = VLLMAlias(
                    source=d.name,
                    targets=(d.name,),
                    scale_targets=(sd.name,) if sd is not None else (),
                    scale_source=f"{d.name}_scale" if sd is not None else None,
                    shard_axis=d.transform.shard_axis,
                    already_sharded=(
                        d.transform.shard_axis is not None
                        or d.transform.channel_splits is not None
                    ),
                    already_fp8=(jnp.dtype(d.dtype) == jnp.dtype(FP8)),
                    flipped=True,
                )
            chosen_aliases.append(chosen)
            covered.update(chosen.targets)
            covered.update(chosen.scale_targets)

        canon = {
            d.name: _coerce_canonical_hf_entry(d, hf_norm, fuse_members) for d in decls
        }
        for tp_rank in range(R):
            bufs = self._buffers.setdefault((lyr, ord_idx, tp_rank), {})
            for al in chosen_aliases:
                w_pieces: list[jax.Array] = []
                for tn in al.targets:
                    td = decl_map[tn]
                    arr_all = canon[tn]
                    arr = (
                        arr_all[ord_idx]
                        if arr_all.ndim == len(td.shape) + 1
                        else arr_all
                    )
                    if not al.already_fp8 and jnp.dtype(td.dtype) == jnp.dtype(FP8):
                        sd = scale_for_map[tn]
                        s_all = canon[sd.name]
                        s_arr = (
                            s_all[ord_idx] if s_all.ndim == len(sd.shape) + 1 else s_all
                        )
                        arr = (arr.astype(F32) * s_arr[None, :].astype(F32)).astype(
                            BF16
                        )
                    w_pieces.append(
                        self._build_target_piece(
                            arr,
                            td,
                            al,
                            lyr=lyr,
                            tp_rank=tp_rank,
                            R=R,
                            moe_tp=moe_tp,
                            er=er,
                            ew=ew,
                        )
                    )
                split_ax = al.shard_axis if al.shard_axis is not None else -1
                bufs[al.source] = (
                    jnp.concatenate(w_pieces, axis=split_ax)
                    if len(w_pieces) > 1
                    else w_pieces[0]
                )
                if al.scale_source is not None and al.scale_targets:
                    s_pieces = [
                        self._build_scale_piece(
                            canon[sn][ord_idx]
                            if canon[sn].ndim == len(decl_map[sn].shape) + 1
                            else canon[sn],
                            decl_map[sn],
                            al,
                            lyr=lyr,
                            tp_rank=tp_rank,
                            R=R,
                            moe_tp=moe_tp,
                            er=er,
                            ew=ew,
                        )
                        for sn in al.scale_targets
                    ]
                    bufs[al.scale_source] = (
                        jnp.concatenate(s_pieces, axis=-1)
                        if len(s_pieces) > 1
                        else s_pieces[0]
                    )

    def _build_target_piece(
        self,
        arr: jax.Array,
        td: HFParamDecl,
        al: VLLMAlias,
        *,
        lyr: int,
        tp_rank: int,
        R: int,
        moe_tp: bool,
        er: int,
        ew: int,
    ) -> jax.Array:
        tf = td.transform
        if tf.expert_axis is not None and not moe_tp and ew > er:
            assert self._e_cores is not None
            idx = jnp.maximum(jnp.asarray(self._e_cores[lyr][tp_rank], dtype=I32), 0)
            return arr[idx]
        if tf.channel_splits is not None:
            parts = []
            off = 0
            for width in tf.channel_splits:
                w_r = width // R
                start = off + tp_rank * w_r
                parts.append(arr[..., start : start + w_r])
                off += width
            cat = jnp.concatenate(parts, axis=-1)
            if tf.flatten_tail and not al.flipped:
                return jnp.transpose(cat, (1, 0))[:, None, :]
            return cat
        if tf.kv_repeat and len(td.shape) == 3:
            if al.already_sharded:
                if arr.shape[0] <= R:
                    return arr[tp_rank // (R // arr.shape[0])]
                k_r = arr.shape[0] // R
                sub = arr[tp_rank * k_r : (tp_rank + 1) * k_r]
                return jnp.swapaxes(sub, 0, 1).reshape(arr.shape[1], k_r * arr.shape[2])
            return jnp.swapaxes(arr, 0, 1).reshape(arr.shape[1], -1)
        if tf.head_reduce == "mean" and len(td.shape) == 2:
            h_r = arr.shape[0] // R
            return arr[tp_rank * h_r : (tp_rank + 1) * h_r].mean(axis=0)
        if tf.kv_repeat and len(td.shape) == 2 and not al.already_sharded:
            return arr[tp_rank // (R // arr.shape[0])]
        if al.already_sharded and tf.shard_axis is not None:
            ax = tf.shard_axis % arr.ndim
            w_r = arr.shape[ax] // R
            sl = [slice(None)] * arr.ndim
            sl[ax] = slice(tp_rank * w_r, (tp_rank + 1) * w_r)
            return arr[tuple(sl)]
        return arr

    def _build_scale_piece(
        self,
        s_arr: jax.Array,
        sd: HFParamDecl,
        al: VLLMAlias,
        *,
        lyr: int,
        tp_rank: int,
        R: int,
        moe_tp: bool,
        er: int,
        ew: int,
    ) -> jax.Array:
        tf = sd.transform
        if tf.expert_axis is not None and not moe_tp and ew > er:
            assert self._e_cores is not None
            idx = jnp.maximum(jnp.asarray(self._e_cores[lyr][tp_rank], dtype=I32), 0)
            return s_arr[idx]
        if tf.kv_repeat and len(sd.shape) == 2:
            if al.already_sharded:
                if s_arr.shape[0] <= R:
                    return s_arr[tp_rank // (R // s_arr.shape[0])]
                k_r = s_arr.shape[0] // R
                return s_arr[tp_rank * k_r : (tp_rank + 1) * k_r].reshape(-1)
            return s_arr.reshape(-1)
        if al.already_sharded and tf.shard_axis is not None:
            ax = tf.shard_axis % s_arr.ndim
            w_r = s_arr.shape[ax] // R
            sl = [slice(None)] * s_arr.ndim
            sl[ax] = slice(tp_rank * w_r, (tp_rank + 1) * w_r)
            return s_arr[tuple(sl)]
        return s_arr

    @property
    def ops(self) -> ArrayOps:
        return self._ops

    def has_layer_alias(self, lyr: int, ord_idx: int, source_name: str) -> bool:
        return source_name in self._buffers.get((lyr, ord_idx, 0), {})

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
        bufs = self._buffers[(lyr, ord_idx, tp_rank)]
        w_buf = bufs[alias.source]
        s_buf = bufs.get(alias.scale_source) if alias.scale_source is not None else None
        return slice_alias_targets(
            self._ops,
            alias,
            w_buf,
            s_buf,
            target_decls,
            scale_decls,
            R=R,
            tp_rank=tp_rank,
            fuse_groups=fuse_groups,
        )

    def read_model_param(self, decl: HFParamDecl, *, R: int, tp_rank: int) -> Any:
        del R, tp_rank
        return self._model_params[decl.name]

    def read_layer_ecopy(
        self,
        lyr: int,
        g: Any,
        ex: ExecSpec,
        *,
        expert_placement: Any = None,
        experts_path: str = "mlp.experts",
    ) -> Any | None:
        del g, ex, expert_placement, experts_path
        if self._ecopy is None:
            return None
        return self._ecopy[:, lyr] if self._ecopy.ndim == 3 else self._ecopy[lyr]

    def supports_passthrough(self) -> bool:
        return False


__all__ = ["SyntheticVLLMWeightSource", "normalize_hf_for_vllm"]
