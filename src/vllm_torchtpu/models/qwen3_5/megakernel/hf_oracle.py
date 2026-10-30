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
"""HuggingFace `transformers` oracle for Qwen3.5 (`qwen35_hf_oracle`).

`torch` and `transformers` are optional dependencies and are imported lazily
inside `qwen35_hf_oracle` only when invoked.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from vllm_torchtpu.megakernel.decls import HFParamDecl
from vllm_torchtpu.megakernel.derive import hf_config_for_geometry
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.geometry import ModelGeometry
from vllm_torchtpu.megakernel.ops import F32
from vllm_torchtpu.megakernel.spec import SwitchSlot
from vllm_torchtpu.megakernel.weights.logical import (
    coerce_canonical_hf_entry,
    resolve_spec_and_norms,
    switch_branch_for_layer,
)

QWEN35_HF_TEXT_MODEL_TYPE = "qwen3_5_moe_text"
QWEN35_HF_LAYER_PREFIX = "model.layers.{l}."


def _dequantize_decl_to_torch(
    d: HFParamDecl,
    by_name: Mapping[str, HFParamDecl],
    w_hf_q: Mapping[str, Any],
    torch_mod: Any,
    *,
    ord_idx: int | None = None,
) -> Any:
    has_l = ord_idx is not None
    arr_c = coerce_canonical_hf_entry(d, w_hf_q, {}, has_layer_dim=has_l)
    arr_f32 = np.asarray((arr_c[ord_idx] if has_l else arr_c).astype(F32))
    s_decl = next(
        (sd for sd in by_name.values() if sd.transform.scale_for == d.name), None
    )
    if s_decl is not None and s_decl.name in w_hf_q:
        s_c = coerce_canonical_hf_entry(s_decl, w_hf_q, {}, has_layer_dim=has_l)
        s_f32 = np.asarray((s_c[ord_idx] if has_l else s_c).astype(F32))
        if s_f32.shape != arr_f32.shape:
            s_f32 = np.expand_dims(s_f32, axis=-2)
        arr_f32 = arr_f32 * s_f32
    if d.transform.transpose is not None:
        arr_f32 = np.transpose(arr_f32, d.transform.transpose)
    hfs = d.hf_shape
    if d.transform.split_from is None and hfs is not None and arr_f32.shape != hfs:
        if arr_f32.ndim == 2 and len(hfs) == 3 and hfs[1] == 1:
            arr_f32 = arr_f32.T[:, None, :]
        elif arr_f32.ndim == 2 and len(hfs) == 1 and arr_f32.shape[-1] == hfs[0]:
            arr_f32 = arr_f32.mean(axis=0)
        else:
            arr_f32 = arr_f32.reshape(hfs)
    return torch_mod.from_numpy(np.array(arr_f32, copy=True)).to(torch_mod.bfloat16)


def _resolve_layer_hf_key(lyr: int, raw_name: str) -> str:
    formatted = raw_name.format(l=lyr)
    if "{l}" in raw_name:
        return formatted
    lyr_prefix = QWEN35_HF_LAYER_PREFIX.format(l=lyr)
    if formatted.startswith(lyr_prefix):
        return formatted
    return f"{lyr_prefix}{formatted}"


def populate_qwen35_hf_experts(
    lyr: int,
    e_decls: Sequence[HFParamDecl],
    by_name: Mapping[str, HFParamDecl],
    w_hf_q: Mapping[str, Any],
    sd: Mapping[str, Any],
    torch_mod: Any,
    dequant_fn: Any,
    *,
    ord_idx: int | None = None,
) -> set[str]:
    """Populate `Qwen3_5MoeForCausalLM` fused expert weights for decoder layer `lyr`."""
    prefix = QWEN35_HF_LAYER_PREFIX.format(l=lyr)
    gup_k = f"{prefix}mlp.experts.gate_up_proj"
    dn_k = f"{prefix}mlp.experts.down_proj"
    if gup_k not in sd or dn_k not in sd:
        raise KeyError(
            f"Missing Qwen3.5 HF expert keys ({gup_k!r}, {dn_k!r}) in state_dict"
        )
    d_gate = next(d for d in e_decls if d.hf_name.endswith(".gate_proj.weight"))
    d_up = next(d for d in e_decls if d.hf_name.endswith(".up_proj.weight"))
    d_down = next(d for d in e_decls if d.hf_name.endswith(".down_proj.weight"))
    gp = dequant_fn(d_gate, by_name, w_hf_q, torch_mod, ord_idx=ord_idx)
    up = dequant_fn(d_up, by_name, w_hf_q, torch_mod, ord_idx=ord_idx)
    dp = dequant_fn(d_down, by_name, w_hf_q, torch_mod, ord_idx=ord_idx)
    sd[gup_k].copy_(torch_mod.cat([gp, up], dim=1).to(sd[gup_k].dtype))
    sd[dn_k].copy_(dp.to(sd[dn_k].dtype))
    return {gup_k, dn_k}


def _populate_qwen35_hf_state_dict(
    m_def: Any,
    g: ModelGeometry,
    ex: ExecSpec,
    w_hf_q: Mapping[str, Any],
    hf_model: Any,
    torch_mod: Any,
) -> None:
    ms, geom, pre_n, post_n, final_n = resolve_spec_and_norms(m_def, ex, g=g)
    sd = hf_model.state_dict()
    populated: set[str] = set()

    for tok_mod in (m_def.embed, m_def.lm_head):
        if tok_mod is not None and getattr(tok_mod, "prefix", None):
            w_k = f"{tok_mod.prefix}.weight"
            if w_k in sd:
                sd[w_k].zero_()
                populated.add(w_k)

    if final_n is not None:
        fd = final_n.hf_params(g=geom)[0]
        if fd.hf_name not in sd:
            raise KeyError(
                f"HF state_dict missing declared final_norm key {fd.hf_name!r}"
            )
        t_w = _dequantize_decl_to_torch(fd, {fd.name: fd}, w_hf_q, torch_mod)
        sd[fd.hf_name].copy_(t_w.to(sd[fd.hf_name].dtype))
        populated.add(fd.hf_name)

    for lyr in range(int(g.L)):
        active: list[tuple[Any, int]] = []
        for nm in (pre_n, post_n):
            if nm is not None:
                active.append((nm, lyr))
        for slot in ms.layer_type.slots:
            if isinstance(slot, SwitchSlot):
                active.append(switch_branch_for_layer(slot, lyr))
            else:
                active.append((slot, lyr))
        for mod, ord_idx in active:
            decls = mod.hf_params(g=geom)
            by_name = {d.name: d for d in decls}
            split_groups: dict[str, list[HFParamDecl]] = {}
            for d in decls:
                if d.transform.scale_for is not None:
                    continue
                if d.transform.split_from is not None:
                    split_groups.setdefault(d.hf_name, []).append(d)
                elif d.transform.expert_axis is not None:
                    continue
                else:
                    hf_k = _resolve_layer_hf_key(lyr, d.hf_name)
                    if hf_k not in sd:
                        raise KeyError(f"HF state_dict missing declared key {hf_k!r}")
                    t_w = _dequantize_decl_to_torch(
                        d, by_name, w_hf_q, torch_mod, ord_idx=ord_idx
                    )
                    sd[hf_k].copy_(t_w.to(sd[hf_k].dtype))
                    populated.add(hf_k)
            for raw_k, s_decls in split_groups.items():
                hf_k = _resolve_layer_hf_key(lyr, raw_k)
                if hf_k not in sd:
                    raise KeyError(f"HF state_dict missing fused split key {hf_k!r}")
                s_sorted = sorted(s_decls, key=lambda x: x.transform.split_slice[0])
                parts = [
                    _dequantize_decl_to_torch(
                        d, by_name, w_hf_q, torch_mod, ord_idx=ord_idx
                    )
                    for d in s_sorted
                ]
                sd[hf_k].copy_(torch_mod.cat(parts, dim=0).to(sd[hf_k].dtype))
                populated.add(hf_k)
            e_decls = [
                d
                for d in decls
                if d.transform.expert_axis is not None and d.transform.scale_for is None
            ]
            if e_decls:
                populated.update(
                    populate_qwen35_hf_experts(
                        lyr,
                        e_decls,
                        by_name,
                        w_hf_q,
                        sd,
                        torch_mod,
                        _dequantize_decl_to_torch,
                        ord_idx=ord_idx,
                    )
                )

    unpopulated = sorted(set(sd.keys()) - populated)
    if unpopulated:
        raise RuntimeError(
            f"Unpopulated HF state_dict parameters ({len(unpopulated)}): "
            f"{unpopulated[:5]}"
        )
    hf_model.load_state_dict(sd, strict=True)


def _instantiate_qwen35_hf_model(
    m_def: Any, g: ModelGeometry, torch_mod: Any, tf_mod: Any
) -> Any:
    hf_cfg_dict = dict(hf_config_for_geometry(m_def, g, max_vocab=256))
    hf_cfg_dict.pop("text_config", None)
    cfg_kwargs = {
        k: v for k, v in hf_cfg_dict.items() if k not in ("model_type", "architectures")
    }
    cfg = tf_mod.AutoConfig.for_model(QWEN35_HF_TEXT_MODEL_TYPE, **cfg_kwargs)
    cfg._attn_implementation = "eager"

    @contextlib.contextmanager
    def _skip_init():
        init = torch_mod.nn.init
        saved = (init.kaiming_uniform_, init.normal_, init.uniform_)
        init.kaiming_uniform_ = init.normal_ = init.uniform_ = lambda t, *a, **kw: t
        try:
            yield
        finally:
            init.kaiming_uniform_, init.normal_, init.uniform_ = saved

    prev_dtype = torch_mod.get_default_dtype()
    try:
        torch_mod.set_default_dtype(torch_mod.bfloat16)
        with _skip_init():
            model = tf_mod.AutoModelForCausalLM.from_config(cfg)
    finally:
        torch_mod.set_default_dtype(prev_dtype)
    model.eval()
    return model


def qwen35_hf_oracle(
    weights: Mapping[str, Any], inputs: Any
) -> tuple[np.ndarray, dict[str, Any]]:
    """Injectable HF `transformers` oracle `(weights, inputs) -> (hidden, states)`."""
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")

    g: ModelGeometry = inputs["g"]
    ex: ExecSpec = inputs["ex"]
    pos = np.asarray(inputs["pos"])
    hidden = np.asarray(inputs["hidden"].astype(F32))
    states: Mapping[str, Any] = inputs["states"]

    backbone = states.get("_hf_backbone")
    if backbone is None:
        from .model import Qwen35

        m_def = getattr(g, "_model_def", None) or Qwen35
        hf_model = _instantiate_qwen35_hf_model(m_def, g, torch, transformers)
        _populate_qwen35_hf_state_dict(m_def, g, ex, weights, hf_model, torch)
        backbone = getattr(hf_model, "model", hf_model)

    xs_th = (
        torch.from_numpy(np.array(hidden, copy=True)).to(torch.bfloat16).unsqueeze(0)
    )
    with torch.no_grad():
        if int(ex.T) > 1 or int(hidden.shape[0]) > 1:
            out_h = (
                backbone(inputs_embeds=xs_th, use_cache=False)
                .last_hidden_state[0]
                .float()
                .numpy()
            )
            return out_h, {"_hf_backbone": backbone}

        past = None if int(pos[0]) == 0 else states.get("_hf_past")
        out_t = backbone(
            inputs_embeds=xs_th[:, :1, :],
            past_key_values=past,
            use_cache=True,
        )
        out_h = out_t.last_hidden_state[0].float().numpy()
        return out_h, {"_hf_backbone": backbone, "_hf_past": out_t.past_key_values}


__all__ = [
    "QWEN35_HF_LAYER_PREFIX",
    "QWEN35_HF_TEXT_MODEL_TYPE",
    "populate_qwen35_hf_experts",
    "qwen35_hf_oracle",
]
