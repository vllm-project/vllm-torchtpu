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
"""Qwen3.5 weight collector and packer for vLLM TPU models.

Extracts sharded `torch.Tensor` weights from vLLM Qwen3.5 layers and packs them
into the layout declared by `planner.plan(qwen35.qwen35_spec(g), ex)`:
- When `ex.stack_e` is `False` (default for serving), dynamic routed-expert
  weights (`e13`, `e2` — ~45.0 GiB / 48.3 GB per rank at `R=8` for 60 layers)
  are passed as their original 3D `torch.nn.Parameter` base buffers
  (`e13[l].data_ptr() == w13.data_ptr()`), avoiding duplicate expert HBM
  allocations; the 3D -> 4D reshape happens as a zero-copy XLA bitcast inside
  `cache.make_flat_pooled_step`.
- Non-expert layer weights (`lin_in`, `lin_out`, `att_in`, `att_out`, `router`,
  `sh13`, `sh2`, `es`, `smalls`, `final_norm` — ~1.5 GB per rank at `R=8` for 60
  layers) are stacked across layers into contiguous base buffers
  (`storage_offset() == 0`, `_is_view() is False`) required by the single-call
  megakernel so `torch_tpu` never emits `tt_jit_execute_as_strided` copies on
  compiled steps.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from typing import Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models import qwen35
from vllm_torchtpu.megakernel.ops import rup

WEIGHT_FLIPPED_ATTR = "_tpu_weight_flipped"


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(f"megakernel_runtime.weight_collector: {msg}")


def _own(x: torch.Tensor) -> torch.Tensor:
    """Return `x` as a contiguous base buffer (`storage_offset() == 0`, not a view)."""
    if not x._is_view() and x.storage_offset() == 0 and x.is_contiguous():
        return x
    return torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)


def _stack(xs: Sequence[torch.Tensor]) -> torch.Tensor:
    """`torch.stack(xs, 0)` that returns a base buffer instead of a view."""
    return torch.cat([x.unsqueeze(0) for x in xs], dim=0)


def _try_set(obj: Any, name: str, value: Any) -> None:
    """Set an existing attribute on `obj`, skipping read-only properties."""
    if hasattr(obj, name):
        with suppress(AttributeError):
            setattr(obj, name, value)


def find_backbone(model: Any) -> Any:
    """Locate the decoder backbone (`layers` + `norm`) inside a vLLM model wrapper."""
    for path in (
        ("model", "language_model", "model"),
        ("language_model", "model"),
        ("model", "model"),
        ("model",),
        (),
    ):
        cur = model
        for attr in path:
            cur = getattr(cur, attr, None)
            if cur is None:
                break
        if cur is not None and hasattr(cur, "layers") and hasattr(cur, "norm"):
            return cur
    raise ValueError(f"Cannot locate decoder backbone in {type(model).__name__}")


def get_routed_experts(mlp: Any) -> Any:
    """Return the underlying `FusedMoE` / `RoutedExperts` module on `mlp`."""
    experts = mlp.experts
    return getattr(experts, "routed_experts", experts)


def extract_kn_fp8_weight(
    proj: Any, hidden_dim: int, out_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(w_kn [hidden_dim, out_dim] fp8, s_n [out_dim] f32)` from `proj`."""
    w = proj.weight.data
    flipped = getattr(proj, WEIGHT_FLIPPED_ATTR, None)
    if flipped is True and tuple(w.shape) == (hidden_dim, out_dim):
        w_kn = w
    elif flipped is False and tuple(w.shape) == (out_dim, hidden_dim):
        w_kn = w.transpose(0, 1)
    elif tuple(w.shape) == (hidden_dim, out_dim) and hidden_dim != out_dim:
        w_kn = w
    elif tuple(w.shape) == (out_dim, hidden_dim):
        w_kn = w.transpose(0, 1)
    else:
        raise ValueError(
            f"Unexpected weight shape {tuple(w.shape)}; expected "
            f"({hidden_dim}, {out_dim}) or ({out_dim}, {hidden_dim})"
        )

    s = getattr(proj, "weight_scale", None)
    if s is None:
        s = getattr(proj, "weight_scale_inv", None)

    if w_kn.dtype == torch.float8_e4m3fn:
        _need(s is not None, "FP8 linear projection is missing weight_scale")
        assert s is not None
        s_1d = s.data.float().reshape(-1).contiguous()
        _need(
            s_1d.shape == (out_dim,),
            f"FP8 weight_scale shape {tuple(s_1d.shape)} != ({out_dim},)",
        )
        return w_kn.contiguous(), s_1d

    # Fallback: quantize BF16/FP32 weight [hidden_dim, out_dim] per output channel.
    w_nk_f32 = w_kn.transpose(0, 1).float()
    amax = w_nk_f32.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
    w_fp8_kn = (
        (w_nk_f32 / scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
    )
    return w_fp8_kn, scale.squeeze(-1).float().contiguous()


def quantize_ba_weight(
    in_proj_ba: Any, hidden_dim: int, vh: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 `in_proj_ba` (`2*vh` channels) per-channel to FP8."""
    out_dim = 2 * vh
    w = in_proj_ba.weight.data
    flipped = getattr(in_proj_ba, WEIGHT_FLIPPED_ATTR, None)
    if flipped is True and tuple(w.shape) == (hidden_dim, out_dim):
        w_nk = w.transpose(0, 1)
    elif tuple(w.shape) == (out_dim, hidden_dim):
        w_nk = w
    elif tuple(w.shape) == (hidden_dim, out_dim):
        w_nk = w.transpose(0, 1)
    else:
        raise ValueError(
            f"Unexpected in_proj_ba.weight shape {tuple(w.shape)}; "
            f"expected ({out_dim}, {hidden_dim}) or ({hidden_dim}, {out_dim})"
        )

    if w_nk.dtype == torch.float8_e4m3fn:
        s = getattr(in_proj_ba, "weight_scale", None)
        if s is None:
            s = getattr(in_proj_ba, "weight_scale_inv", None)
        _need(s is not None, "FP8 in_proj_ba missing weight_scale")
        assert s is not None
        return (
            w_nk.transpose(0, 1).contiguous(),
            s.data.float().reshape(out_dim).contiguous(),
        )

    w_f32 = w_nk.float()
    amax = w_f32.abs().amax(dim=-1, keepdim=True)
    ba_scale = torch.where(amax == 0.0, torch.ones_like(amax), amax * (1.0 / 448.0))
    ba_fp8_kn = (
        (w_f32 / ba_scale)
        .clamp(-448.0, 448.0)
        .to(torch.float8_e4m3fn)
        .transpose(0, 1)
        .contiguous()
    )
    return ba_fp8_kn, ba_scale.squeeze(-1).float().contiguous()


def pack_lin_in_layer(
    gdn_layer: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    tk_lin_in: int,
    in_s_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a GDN layer's `in_proj_qkvz` + `in_proj_ba` into `lin_in` and `in_s`."""
    lay = GDNBlock.lin_in_layout(g, ex)
    nai = g.NAI(ex.R)
    qkvz_dim = lay.QKV + lay.Z
    w_qkvz, s_qkvz = extract_kn_fp8_weight(gdn_layer.in_proj_qkvz, g.H, qkvz_dim)
    ba_fp8, ba_scale = quantize_ba_weight(gdn_layer.in_proj_ba, g.H, lay.vh)

    ba_cols = lay.NLI - lay.BA
    a_rel = lay.A_OFF - lay.BA
    ba_pad = torch.zeros(
        (g.H, ba_cols), dtype=torch.float8_e4m3fn, device=w_qkvz.device
    )
    ba_pad[:, 0 : lay.vh] = ba_fp8[:, 0 : lay.vh]
    ba_pad[:, a_rel : a_rel + lay.vh] = ba_fp8[:, lay.vh : 2 * lay.vh]

    w_cat = torch.cat([w_qkvz, ba_pad], dim=1).contiguous()
    w_tiled = w_cat.view(g.H // tk_lin_in, tk_lin_in, lay.NLI)

    s_width = in_s_width if in_s_width is not None else max(lay.NLI, nai)
    in_s = torch.zeros((s_width,), dtype=torch.float32, device=s_qkvz.device)
    in_s[0:qkvz_dim] = s_qkvz
    in_s[lay.BA : lay.BA + lay.vh] = ba_scale[0 : lay.vh]
    in_s[lay.A_OFF : lay.A_OFF + lay.vh] = ba_scale[lay.vh : 2 * lay.vh]
    return w_tiled, in_s


def pack_lin_out_layer(
    gdn_layer: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    tk_lin_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a GDN layer's `out_proj` into `[Z // tk_lin_out, tk_lin_out, H]`."""
    lay = GDNBlock.lin_in_layout(g, ex)
    w_kn, s_n = extract_kn_fp8_weight(gdn_layer.out_proj, lay.Z, g.H)
    return w_kn.view(lay.Z // tk_lin_out, tk_lin_out, g.H), s_n


def pack_att_in_layer(
    attn_layer: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    tk_att_in: int,
    in_s_width: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a full-attention layer's `qkv_proj` into `att_in` and `in_s`."""
    nh = g.nh(ex.R)
    nai = g.NAI(ex.R)
    gdn_lay = GDNBlock.lin_in_layout(g, ex)
    if hasattr(attn_layer, "qkv_proj"):
        w_kn, s_n = extract_kn_fp8_weight(attn_layer.qkv_proj, g.H, nai)
    else:
        q_dim = 2 * nh * g.HD
        kv_dim = (nai - q_dim) // 2
        wq, sq = extract_kn_fp8_weight(attn_layer.q_proj, g.H, q_dim)
        wk, sk = extract_kn_fp8_weight(attn_layer.k_proj, g.H, kv_dim)
        wv, sv = extract_kn_fp8_weight(attn_layer.v_proj, g.H, kv_dim)
        w_kn = torch.cat([wq, wk, wv], dim=1).contiguous()
        s_n = torch.cat([sq, sk, sv], dim=0).contiguous()
    s_width = in_s_width if in_s_width is not None else max(gdn_lay.NLI, nai)
    in_s = torch.zeros((s_width,), dtype=torch.float32, device=s_n.device)
    in_s[0:nai] = s_n
    return w_kn.view(g.H // tk_att_in, tk_att_in, nai), in_s


def pack_att_out_layer(
    attn_layer: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    tk_att_out: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a full-attention layer's `o_proj` into `att_out` and `out_s`."""
    k_dim = g.nh(ex.R) * g.HD
    w_kn, s_n = extract_kn_fp8_weight(attn_layer.o_proj, k_dim, g.H)
    return w_kn.view(k_dim // tk_att_out, tk_att_out, g.H), s_n


def pack_router_and_shared_layer(
    mlp: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    *,
    tp_rank: int = 0,
    tk_router: int,
    tk_sh13: int,
    tk_sh2: int,
) -> dict[str, torch.Tensor]:
    """Extract router (`bf16`), shared expert `sh13`/`sh2` (`fp8`), and scales."""
    isr = g.ISR(ex.R)
    sh_rank = tp_rank // g.sh_rep(ex.R)
    w_gu = getattr(mlp, "_fused_gate_gu_fp8_weight", None)
    if w_gu is not None:
        s_gu = mlp._fused_gate_gu_fp8_scale.data.float().reshape(-1)
        if hasattr(mlp, "_fused_se_down_fp8_weight"):
            w_down = mlp._fused_se_down_fp8_weight.data
            s_down = mlp._fused_se_down_fp8_scale.data.float().reshape(g.H).contiguous()
        else:
            down_proj = mlp.shared_expert.down_proj
            down_rows = int(
                down_proj.weight.shape[0]
                if getattr(down_proj, WEIGHT_FLIPPED_ATTR, False)
                or down_proj.weight.shape[1] == g.H
                else down_proj.weight.shape[1]
            )
            w_down, s_down = extract_kn_fp8_weight(down_proj, down_rows, g.H)
        if hasattr(mlp, "_fused_se_gate_weight"):
            sh_gate = mlp._fused_se_gate_weight.data.float().reshape(g.H).contiguous()
        else:
            sh_gate = (
                mlp.shared_expert_gate.weight.data.float().reshape(g.H).contiguous()
            )

        w_router_fp8 = w_gu.data[:, : g.E]
        s_router = s_gu[: g.E]
        w_router_bf16 = (
            (w_router_fp8.float() * s_router.unsqueeze(0))
            .to(torch.bfloat16)
            .contiguous()
            .view(g.H // tk_router, tk_router, g.E)
        )

        n_cols = int(w_gu.shape[1])
        if n_cols == g.E + 2 * g.IS:
            g0 = g.E + sh_rank * isr
            u0 = g.E + g.IS + sh_rank * isr
            w_sh13 = torch.cat(
                [w_gu.data[:, g0 : g0 + isr], w_gu.data[:, u0 : u0 + isr]], dim=1
            ).contiguous()
            s_sh13 = torch.cat(
                [s_gu[g0 : g0 + isr], s_gu[u0 : u0 + isr]], dim=0
            ).contiguous()
        elif n_cols == g.E + 2 * isr:
            w_sh13 = w_gu.data[:, g.E : g.E + 2 * isr].contiguous()
            s_sh13 = s_gu[g.E : g.E + 2 * isr].contiguous()
        else:
            raise ValueError(
                f"Unexpected _fused_gate_gu_fp8_weight shape {tuple(w_gu.shape)}"
            )

        if int(w_down.shape[0]) == g.IS:
            r0 = sh_rank * isr
            w_sh2 = w_down[r0 : r0 + isr, :].contiguous()
        elif int(w_down.shape[0]) == isr:
            w_sh2 = w_down.contiguous()
        else:
            raise ValueError(
                f"Unexpected _fused_se_down_fp8_weight shape {tuple(w_down.shape)}"
            )
    else:
        gate_w = mlp.gate.weight.data
        if tuple(gate_w.shape) == (g.E, g.H):
            gate_w = gate_w.transpose(0, 1)
        w_router_bf16 = (
            gate_w.to(torch.bfloat16)
            .contiguous()
            .view(g.H // tk_router, tk_router, g.E)
        )

        gu_proj = mlp.shared_expert.gate_up_proj
        down_proj = mlp.shared_expert.down_proj
        gu_cols = int(
            gu_proj.weight.shape[1]
            if getattr(gu_proj, WEIGHT_FLIPPED_ATTR, False)
            or gu_proj.weight.shape[0] == g.H
            else gu_proj.weight.shape[0]
        )
        w_gu_only, s_gu_only = extract_kn_fp8_weight(gu_proj, g.H, gu_cols)
        if gu_cols == 2 * g.IS:
            g0 = sh_rank * isr
            u0 = g.IS + sh_rank * isr
            w_sh13 = torch.cat(
                [w_gu_only[:, g0 : g0 + isr], w_gu_only[:, u0 : u0 + isr]], dim=1
            ).contiguous()
            s_sh13 = torch.cat(
                [s_gu_only[g0 : g0 + isr], s_gu_only[u0 : u0 + isr]], dim=0
            ).contiguous()
        elif gu_cols == 2 * isr:
            w_sh13 = w_gu_only.contiguous()
            s_sh13 = s_gu_only.contiguous()
        else:
            raise ValueError(f"Unexpected shared_expert.gate_up_proj cols {gu_cols}")

        down_rows = int(
            down_proj.weight.shape[0]
            if getattr(down_proj, WEIGHT_FLIPPED_ATTR, False)
            or down_proj.weight.shape[1] == g.H
            else down_proj.weight.shape[1]
        )
        w_down_only, s_down = extract_kn_fp8_weight(down_proj, down_rows, g.H)
        if down_rows == g.IS:
            r0 = sh_rank * isr
            w_sh2 = w_down_only[r0 : r0 + isr, :].contiguous()
        elif down_rows == isr:
            w_sh2 = w_down_only.contiguous()
        else:
            raise ValueError(f"Unexpected shared_expert.down_proj rows {down_rows}")

        sh_gate = mlp.shared_expert_gate.weight.data.float().reshape(g.H).contiguous()

    return {
        "router": w_router_bf16,
        "sh13": w_sh13.view(g.H // tk_sh13, tk_sh13, 2 * isr),
        "sh2": w_sh2.view(isr // tk_sh2, tk_sh2, g.H),
        "sh13_s": s_sh13,
        "sh2_s": s_down,
        "sh_gate": sh_gate,
    }


def extract_ecopy_layer(
    layer_idx: int | None,
    layer: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    dev: torch.device,
    *,
    expert_placement: Any = None,
) -> torch.Tensor | None:
    """Build the `[E]` `float32` `ecopy` descriptor vector for `smalls['ecopy']`.

    Returns `None` when no explicit `_ecopy`, `_ecopy_cores`, or matching
    `expert_placement` manifest is present on the module, so `smalls['ecopy']`
    remains all zeros (primary-only routing).
    """
    er = g.ER(ex.R)
    ew = g.EW(ex.R)
    if g.moe_tp or g.e_slots <= er:
        return None
    mlp = getattr(layer, "mlp", None)
    routed = (
        get_routed_experts(mlp) if mlp is not None and hasattr(mlp, "experts") else None
    )
    if routed is not None:
        ec_buf = getattr(routed, "_ecopy", None)
        if ec_buf is not None:
            return ec_buf.to(device=dev, dtype=torch.float32).reshape(g.E)
        ec_cores = getattr(routed, "_ecopy_cores", None)
        if ec_cores is not None:
            return torch.tensor(
                planner.encode_ecopy(g.E, er, ew, g.moe_tp, ec_cores),
                dtype=torch.float32,
                device=dev,
            )
        if expert_placement is None:
            expert_placement = getattr(routed, "_expert_placement", None)
    if expert_placement is not None:
        if layer_idx is None and routed is not None:
            prefix = getattr(routed, "layer_name", "") or ""
            layer_idx = expert_placement.layer_index(prefix)
        if (
            layer_idx is not None
            and expert_placement.has_copies()
            and int(expert_placement.slots()) == int(ew)
            and int(expert_placement.global_count()) == int(g.E)
            and int(expert_placement.shards()) == int(ex.R)
            and str(layer_idx) in expert_placement.data().get("layers", {})
        ):
            cores = expert_placement.cores(layer_idx)
            return torch.tensor(
                planner.encode_ecopy(g.E, er, ew, g.moe_tp, cores),
                dtype=torch.float32,
                device=dev,
            )
    return None


def pack_smalls_layer(
    layer: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    smalls_layout: dict[str, tuple[int, int]],
    smalls_stride: int,
    *,
    in_s: torch.Tensor,
    out_s: torch.Tensor,
    sh13_s: torch.Tensor,
    sh2_s: torch.Tensor,
    sh_gate: torch.Tensor,
    layer_idx: int | None = None,
    ecopy: torch.Tensor | None = None,
    expert_placement: Any = None,
) -> torch.Tensor:
    """Pack a single layer's small parameters into `[1, S] float32`."""
    dev = in_s.device
    isr = g.ISR(ex.R)
    gdn_lay = GDNBlock.lin_in_layout(g, ex)

    in_norm_w = layer.input_layernorm.weight.data.float().reshape(g.H)
    post_norm_w = layer.post_attention_layernorm.weight.data.float().reshape(g.H)

    fields: dict[str, torch.Tensor] = {
        "in_norm": in_norm_w,
        "post_norm": post_norm_w,
        "in_s": in_s.float().reshape(-1),
        "out_s": out_s.float().reshape(g.H),
        "sh13_s": sh13_s.float().reshape(2 * isr),
        "sh2_s": sh2_s.float().reshape(g.H),
        "sh_gate": sh_gate.float().reshape(g.H),
    }
    if "ecopy" in smalls_layout:
        if ecopy is None:
            ecopy = extract_ecopy_layer(
                layer_idx, layer, g, ex, dev, expert_placement=expert_placement
            )
        if ecopy is not None:
            fields["ecopy"] = ecopy.float().reshape(g.E)

    ltype = getattr(layer, "layer_type", None)
    if ltype is None and layer_idx is not None:
        ltype = (
            "full_attention"
            if (layer_idx % g.att_every == g.att_every - 1)
            else "linear_attention"
        )
    if ltype == "linear_attention":
        la = layer.linear_attn
        conv_w = (
            la.conv1d.weight.data.float()
            .reshape(gdn_lay.QKV, 4)
            .transpose(0, 1)
            .contiguous()
        )
        alog_w = la.A_log.data.float().reshape(gdn_lay.vh)
        dtb_w = la.dt_bias.data.float().reshape(gdn_lay.vh)
        gnorm_w = la.norm.weight.data.float().reshape(gdn_lay.VD)

        fields["conv"] = conv_w.reshape(4 * gdn_lay.QKV)
        fields["alog"] = F.pad(alog_w, (0, 128 - gdn_lay.vh))
        fields["dtb"] = F.pad(dtb_w, (0, 128 - gdn_lay.vh))
        fields["gnorm"] = gnorm_w
    else:
        sa = layer.self_attn
        fields["qn"] = sa.q_norm.weight.data.float().reshape(g.HD)
        fields["kn"] = sa.k_norm.weight.data.float().reshape(g.HD)

    parts = []
    for name, (_, width) in smalls_layout.items():
        v = fields.get(name)
        if v is None:
            v = torch.zeros((width,), dtype=torch.float32, device=dev)
        else:
            v = v.reshape(-1).float()
        _need(v.numel() == width, f"smalls field {name}: {v.numel()} != {width}")
        pad_len = rup(width, 128) - width
        if pad_len > 0:
            v = F.pad(v, (0, pad_len))
        parts.append(v)

    return torch.cat(parts, dim=0).view(1, smalls_stride).contiguous()


def pack_expert_views_layer(
    mlp: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    tk_e13: int,
    tk_e2: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract per-layer routed expert weights (`e13`, `e2`) and scales (`es_mk`).

    When `ex.stack_e` is `False`, returns the base 3D `Parameter` tensors
    `[EW, H, 2*IR]` and `[EW, IR, H]` directly without `.view(...)`, preserving
    `data_ptr()` and `storage_offset() == 0` with zero duplicate HBM.
    """
    experts = get_routed_experts(mlp)
    w13 = experts.w13_weight
    w2 = experts.w2_weight
    ew = g.EW(ex.R)
    ir = g.IR(ex.R)
    _need(
        int(w13.shape[0]) == ew,
        f"experts.w13_weight.shape[0]={w13.shape[0]} != EW={ew}",
    )
    _need(
        w13.is_contiguous() and w2.is_contiguous(),
        "expert weights must be contiguous for zero-copy views",
    )

    is_nvfp4 = (
        str(g.expert_quant) == "nvfp4"
        or getattr(experts, "w13_weight_scale_2", None) is not None
    )
    if is_nvfp4:

        def _pack_nvfp4_w(
            w_raw: torch.Tensor,
            bscale_raw: torch.Tensor,
            k_dim: int,
            n_dim: int,
            tk_raw: int,
        ) -> torch.Tensor:
            k_half = k_dim // 2
            n_raw = n_dim + rup(n_dim // 8, 128)
            w_u8 = w_raw.data.view(torch.uint8)
            if tuple(w_u8.shape) == (ew, k_half, n_raw):
                fused = w_u8.contiguous()
            else:
                if tuple(w_u8.shape) == (ew, n_dim, k_half):
                    w_kn = w_u8.transpose(1, 2).contiguous()
                else:
                    w_kn = w_u8.reshape(ew, k_half, n_dim).contiguous()
                bs_u8 = bscale_raw.data.view(torch.uint8)
                if tuple(bs_u8.shape) == (ew, n_dim, k_dim // 16):
                    bs_kn = bs_u8.transpose(1, 2).contiguous()
                else:
                    bs_kn = bs_u8.reshape(ew, k_dim // 16, n_dim).contiguous()
                bs_cols = bs_kn.reshape(ew, k_half, n_dim // 8)
                pad_cols = n_raw - n_dim - (n_dim // 8)
                parts = [w_kn, bs_cols]
                if pad_cols > 0:
                    parts.append(
                        torch.zeros(
                            (ew, k_half, pad_cols),
                            dtype=torch.uint8,
                            device=w_u8.device,
                        )
                    )
                fused = _own(torch.cat(parts, dim=-1))
            if not ex.stack_e:
                return fused
            return fused.view(ew, k_half // tk_raw, tk_raw, n_raw)

        e13_view = _pack_nvfp4_w(w13, experts.w13_weight_scale, g.H, 2 * ir, tk_e13)
        e2_view = _pack_nvfp4_w(w2, experts.w2_weight_scale, ir, g.H, tk_e2)
        g13 = getattr(experts, "w13_weight_scale_2", None)
        g2 = getattr(experts, "w2_weight_scale_2", None)
        s13 = (
            g13.data.float().reshape(ew, -1).expand(ew, 2 * ir).contiguous()
            if g13 is not None
            else torch.ones((ew, 2 * ir), dtype=torch.float32, device=w13.device)
        )
        s2 = (
            g2.data.float().reshape(ew, -1).expand(ew, g.H).contiguous()
            if g2 is not None
            else torch.ones((ew, g.H), dtype=torch.float32, device=w2.device)
        )
    else:
        if not ex.stack_e:
            e13_view = w13
            e2_view = w2
        else:
            t13 = g.H // tk_e13
            t2 = ir // tk_e2
            e13_view = w13.view(ew, t13, tk_e13, 2 * ir)
            e2_view = w2.view(ew, t2, tk_e2, g.H)
        _need(
            e13_view.data_ptr() == w13.data_ptr()
            and e2_view.data_ptr() == w2.data_ptr(),
            "expert views must share storage with w13_weight / w2_weight",
        )

        s13_raw = getattr(experts, "w13_weight_scale_inv", None)
        if s13_raw is None:
            s13_raw = experts.w13_weight_scale
        s2_raw = getattr(experts, "w2_weight_scale_inv", None)
        if s2_raw is None:
            s2_raw = experts.w2_weight_scale

        s13 = s13_raw.data.float().reshape(ew, 2 * ir)
        s2 = s2_raw.data.float().reshape(ew, g.H)
    es_cat = torch.cat([s13, s2], dim=-1).contiguous()
    es_mk = es_cat.unsqueeze(1).contiguous()
    return e13_view, e2_view, es_mk


def reshard_experts_ep_to_tp_local(
    w13_all: torch.Tensor,
    w2_all: torch.Tensor,
    s13_all: torch.Tensor,
    s2_all: torch.Tensor,
    tp_rank: int,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Slice global `E`-expert weights/scales into `tp_rank`'s TP-sharded tensors."""
    ir = g.I // ex.R
    g0 = tp_rank * ir
    u0 = g.I + tp_rank * ir

    w13_tp = _own(
        torch.cat([w13_all[:, :, g0 : g0 + ir], w13_all[:, :, u0 : u0 + ir]], dim=-1)
    )
    w2_tp = _own(w2_all[:, g0 : g0 + ir, :])

    s13_flat = s13_all.float().reshape(g.E, 2 * g.I)
    s2_flat = s2_all.float().reshape(g.E, g.H)
    s13_tp = _own(
        torch.cat(
            [s13_flat[:, g0 : g0 + ir], s13_flat[:, u0 : u0 + ir]], dim=-1
        ).reshape(g.E, 1, 1, 2 * ir)
    )
    s2_tp = _own(s2_flat.reshape(g.E, 1, 1, g.H))
    return w13_tp, w2_tp, s13_tp, s2_tp


def apply_tp_moe_flags(experts: Any, *, prebuild_moe_fn: Any = None) -> None:
    """Update a `FusedMoE` module's flags after in-place EP->TP resharding."""
    targets = [experts]
    routed = getattr(experts, "routed_experts", None)
    if routed is not None and routed is not experts:
        targets.append(routed)
    for tgt in targets:
        moe_cfg = getattr(tgt, "moe_config", None)
        if moe_cfg is not None:
            p_cfg = getattr(moe_cfg, "moe_parallel_config", None)
            if p_cfg is not None:
                _try_set(p_cfg, "use_ep", False)
                if (
                    getattr(p_cfg, "ep_size", 1) > 1
                    and getattr(p_cfg, "tp_size", 1) == 1
                ):
                    _try_set(p_cfg, "tp_size", p_cfg.ep_size)
                    _try_set(p_cfg, "tp_rank", getattr(p_cfg, "ep_rank", 0))
                    _try_set(p_cfg, "ep_size", 1)
                    _try_set(p_cfg, "ep_rank", 0)
                _try_set(p_cfg, "sp_size", 1)
                _try_set(p_cfg, "is_sequence_parallel", False)
            _try_set(moe_cfg, "use_ep", False)
            _try_set(moe_cfg, "is_sequence_parallel", False)
        _try_set(tgt, "_tpu_fused_ep_eligible", False)
        for buf in ("_experts_start", "_expert_map", "expert_mask"):
            if buf in getattr(tgt, "_buffers", {}):
                tgt.register_buffer(buf, None, persistent=False)
            else:
                _try_set(tgt, buf, None)
        _try_set(tgt, "_gmm_primary_experts", None)
        _try_set(tgt, "_lowtok_has_copies", False)
        if hasattr(tgt, "global_num_experts"):
            _try_set(tgt, "local_num_experts", tgt.global_num_experts)
        for qm_attr in ("quant_method", "_quant_method"):
            qm = getattr(tgt, qm_attr, None)
            if qm is not None:
                _try_set(qm, "_tpu_fused_ep_eligible", False)
                _try_set(qm, "_moe_lowtok_scores_op", None)
                _try_set(qm, "_fused_moe_ep_op", None)
                _try_set(qm, "_use_adaptive_fused_moe", False)
        if (
            prebuild_moe_fn is not None
            and moe_cfg is not None
            and hasattr(moe_cfg, "experts_per_token")
            and hasattr(tgt, "activation")
        ):
            act_val = getattr(tgt.activation, "value", tgt.activation)
            prebuild_moe_fn(
                topk=moe_cfg.experts_per_token,
                activation=act_val,
                use_ep=False,
            )


def reshard_layer_experts_ep_to_tp(
    mlp: Any,
    tp_rank: int,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    *,
    all_gather_fn: Any = None,
    prebuild_moe_fn: Any = None,
) -> None:
    """Reshard `mlp.experts` from EP (`[ER, H, 2*I]`) to TP (`[E, H, 2*IR]`)."""
    experts = get_routed_experts(mlp)
    er = g.ER(ex.R)
    w13_ep = experts.w13_weight.data[:er].contiguous()
    w2_ep = experts.w2_weight.data[:er].contiguous()

    s13_attr = (
        "w13_weight_scale_inv"
        if hasattr(experts, "w13_weight_scale_inv")
        else "w13_weight_scale"
    )
    s2_attr = (
        "w2_weight_scale_inv"
        if hasattr(experts, "w2_weight_scale_inv")
        else "w2_weight_scale"
    )
    s13_ep = getattr(experts, s13_attr).data[:er].reshape(er, 2 * g.I).contiguous()
    s2_ep = getattr(experts, s2_attr).data[:er].reshape(er, g.H).contiguous()

    if all_gather_fn is None:
        _need(
            ex.R == 1,
            "all_gather_fn must be provided to reshard_layer_experts_ep_to_tp "
            "when R > 1",
        )
        w13_all, w2_all, s13_all, s2_all = w13_ep, w2_ep, s13_ep, s2_ep
    else:
        w13_all = all_gather_fn(w13_ep)
        w2_all = all_gather_fn(w2_ep)
        s13_all = all_gather_fn(s13_ep)
        s2_all = all_gather_fn(s2_ep)

    w13_tp, w2_tp, s13_tp, s2_tp = reshard_experts_ep_to_tp_local(
        w13_all, w2_all, s13_all, s2_all, tp_rank=tp_rank, g=g, ex=ex
    )
    del w13_all, w2_all, s13_all, s2_all
    del w13_ep, w2_ep, s13_ep, s2_ep

    experts.w13_weight = torch.nn.Parameter(w13_tp, requires_grad=False)
    experts.w2_weight = torch.nn.Parameter(w2_tp, requires_grad=False)
    s13_new_param = torch.nn.Parameter(s13_tp, requires_grad=False)
    s2_new_param = torch.nn.Parameter(s2_tp, requires_grad=False)
    setattr(experts, s13_attr, s13_new_param)
    setattr(experts, s2_attr, s2_new_param)

    if w13_tp.device.type == "tpu":
        from torch_tpu._internal import sync

        sync.synchronize(
            tensors=[
                experts.w13_weight,
                experts.w2_weight,
                s13_new_param,
                s2_new_param,
            ],
            wait=True,
        )

    apply_tp_moe_flags(mlp.experts, prebuild_moe_fn=prebuild_moe_fn)


def pack_weights(
    backbone: Any,
    g: qwen35.Qwen35Geometry,
    ex: ExecSpec,
    *,
    tp_rank: int = 0,
    expert_placement: Any = None,
) -> dict[str, Any]:
    """Pack all `g.L` layers of `backbone` into the megakernel's weight dictionary.

    Driven by `planner.plan(qwen35.qwen35_spec(g), ex, fit_vmem=False)`:
    - Returns all keys in `plan.w_names` (`lin_in`, `lin_out`, `att_in`, `att_out`,
      `router`, `sh13`, `sh2`, `e13`, `e2`, `es`, `smalls`, `final_norm`).
    - Returns `"w_flat"`: flat list of `torch.Tensor` leaves in `plan.w_names` order
      (expanding `tuple`s for unstacked dynamic kinds `plan.dynamic_kinds`).
    - Returns `"plan"`: the unfitted `Plan` (`fit_vmem=False`, so packing never
      initializes a JAX backend).
    """
    spec = qwen35.qwen35_spec(g)
    p = planner.plan(spec, ex, fit_vmem=False)
    tk = p.tk
    in_s_width = p.smalls_layout["in_s"][1]

    layers = backbone.layers
    _need(len(layers) == g.L, f"expected {g.L} layers, got {len(layers)}")

    lin_in_list: list[torch.Tensor] = []
    lin_out_list: list[torch.Tensor] = []
    att_in_list: list[torch.Tensor] = []
    att_out_list: list[torch.Tensor] = []
    router_list: list[torch.Tensor] = []
    sh13_list: list[torch.Tensor] = []
    sh2_list: list[torch.Tensor] = []
    e13_list: list[torch.Tensor] = []
    e2_list: list[torch.Tensor] = []
    es_mk_list: list[torch.Tensor] = []
    smalls_mk_list: list[torch.Tensor] = []

    final_norm_w = _own(backbone.norm.weight.data.float().reshape(1, g.H))

    for l_idx in range(g.L):
        layer = layers[l_idx]
        ltype = getattr(layer, "layer_type", None)
        if ltype is None:
            ltype = (
                "full_attention"
                if (l_idx % g.att_every == g.att_every - 1)
                else "linear_attention"
            )

        if ltype == "linear_attention":
            w_in, in_s = pack_lin_in_layer(
                layer.linear_attn, g, ex, tk["lin_in"], in_s_width=in_s_width
            )
            w_out, out_s = pack_lin_out_layer(layer.linear_attn, g, ex, tk["lin_out"])
            lin_in_list.append(w_in)
            lin_out_list.append(w_out)
        elif ltype == "full_attention":
            w_in, in_s = pack_att_in_layer(
                layer.self_attn, g, ex, tk["att_in"], in_s_width=in_s_width
            )
            w_out, out_s = pack_att_out_layer(layer.self_attn, g, ex, tk["att_out"])
            att_in_list.append(w_in)
            att_out_list.append(w_out)
        else:
            raise ValueError(f"Layer {l_idx}: unknown layer_type={ltype!r}")

        rs = pack_router_and_shared_layer(
            layer.mlp,
            g=g,
            ex=ex,
            tp_rank=tp_rank,
            tk_router=tk["router"],
            tk_sh13=tk["sh13"],
            tk_sh2=tk["sh2"],
        )
        router_list.append(rs["router"])
        sh13_list.append(rs["sh13"])
        sh2_list.append(rs["sh2"])

        e13_v, e2_v, es_mk = pack_expert_views_layer(
            layer.mlp, g=g, ex=ex, tk_e13=tk["e13"], tk_e2=tk["e2"]
        )
        e13_list.append(e13_v)
        e2_list.append(e2_v)
        es_mk_list.append(es_mk)

        sm_mk = pack_smalls_layer(
            layer,
            g=g,
            ex=ex,
            smalls_layout=p.smalls_layout,
            smalls_stride=p.smalls_width,
            in_s=in_s,
            out_s=out_s,
            sh13_s=rs["sh13_s"],
            sh2_s=rs["sh2_s"],
            sh_gate=rs["sh_gate"],
            layer_idx=l_idx,
            expert_placement=expert_placement,
        )
        smalls_mk_list.append(sm_mk)

    e13_out: Any = _stack(e13_list) if ex.stack_e else tuple(e13_list)
    e2_out: Any = _stack(e2_list) if ex.stack_e else tuple(e2_list)
    w_shapes = planner.weight_shapes(spec, ex, p)

    def _stack_optional(xs: list[torch.Tensor], kind: str) -> torch.Tensor:
        if xs:
            return _stack(xs)
        shape, _ = w_shapes[kind]
        return torch.zeros(shape, dtype=torch.float8_e4m3fn, device=final_norm_w.device)

    weights: dict[str, Any] = {
        "lin_in": _stack_optional(lin_in_list, "lin_in"),
        "lin_out": _stack_optional(lin_out_list, "lin_out"),
        "att_in": _stack(att_in_list),
        "att_out": _stack(att_out_list),
        "router": _stack(router_list),
        "sh13": _stack(sh13_list),
        "sh2": _stack(sh2_list),
        "e13": e13_out,
        "e2": e2_out,
        "es": _stack(es_mk_list),
        "smalls": _stack(smalls_mk_list),
        "final_norm": final_norm_w,
        "plan": p,
    }

    w_flat: list[torch.Tensor] = []
    for k in p.w_names:
        val = weights[k]
        if isinstance(val, (tuple, list)):
            w_flat.extend(t.detach() for t in val)
        else:
            w_flat.append(val.detach())

    weights["w_flat"] = w_flat
    return weights
