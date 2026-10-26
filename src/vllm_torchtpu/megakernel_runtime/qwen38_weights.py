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
"""Per-rank Qwen3.8 segment-kernel operands from loaded vLLM `qwen4_exp` modules.

Segment `s` covers QSA layer `q = att_every*s + att_every-1` after its
attention core, then the linear-attention layers up to the next QSA layer
(none for the last segment). Kernel layer `j = 0` is `q`; `j >= 1` is
`q + j`. Dense weights stay BF16 (no requantization); routed experts are the
vLLM per-channel FP8 tensors, passed zero-copy. Every operand is this rank's
local shard of the kernel's `P('tp')` input, in `plan.w_names` order.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.layers.adapter.quantization.unquantized import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models.qwen38 import Qwen38Geometry
from vllm_torchtpu.megakernel.ops import rup
from vllm_torchtpu.megakernel.spec import ModelSpec

from .weight_collector import pack_expert_views_layer


def kn(proj: Any, k: int, n: int) -> torch.Tensor:
    """`[k, n]` BF16 view of a linear weight stored `[k, n]` (flipped) or `[n, k]`."""
    w = proj.weight.data if hasattr(proj, "weight") else proj.data
    flipped = getattr(proj, WEIGHT_FLIPPED_ATTR, None)
    if tuple(w.shape) == (k, n) and (flipped is not False or k != n):
        out = w
    elif tuple(w.shape) == (n, k):
        out = w.transpose(0, 1)
    else:
        raise ValueError(f"weight shape {tuple(w.shape)} is neither ({k},{n}) nor ({n},{k})")
    assert out.dtype == torch.bfloat16, out.dtype
    return out


def segment_layers(g: Qwen38Geometry, s: int) -> list[int]:
    """Model layer of each kernel layer of segment `s`."""
    q = g.att_every * s + g.att_every - 1
    n = 1 if s == g.L // g.att_every - 1 else g.att_every
    return [q + j for j in range(n)]


def _hc_mix_operands(hc: Any, g: Qwen38Geometry) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """`(hdn [W, n_down], [hup_b [k_up, H]])` of one mixing hyper-connection."""
    W, lr = g.NHC * g.H, g.LOWRANK
    down = hc.input_mix_weight_down.weight.data  # [LR, W]
    inj = hc.block_inject_weight.weight.data  # [NHC, W]
    up = hc.input_mix_weight_up.weight.data  # [W, LR]
    assert tuple(down.shape) == (lr, W) and tuple(up.shape) == (W, lr), (down.shape, up.shape)
    hdn = torch.cat([down.t(), inj.t()], dim=1)
    hdn = F.pad(hdn, (0, rup(lr + g.NHC, 128) - hdn.shape[1]))
    up_t = F.pad(up.t(), (0, 0, 0, rup(lr, 128) - lr))
    return hdn.contiguous(), [up_t[:, b * g.H : (b + 1) * g.H].contiguous() for b in range(g.NHC)]


def _lin_in(la: Any, g: Any, ex: ExecSpec) -> torch.Tensor:
    """`[H, NLI]` = `[qkv | z | b@BA | a@A_OFF]` from `in_proj_qkvz` + `in_proj_ba`."""
    lay = GDNBlock.lin_in_layout(g, ex)
    qkvz = kn(la.in_proj_qkvz, g.H, lay.QKV + lay.Z)
    ba = kn(la.in_proj_ba, g.H, 2 * lay.vh)
    pad = torch.zeros((g.H, lay.NLI - lay.BA), dtype=qkvz.dtype, device=qkvz.device)
    a_rel = lay.A_OFF - lay.BA
    pad[:, : lay.vh] = ba[:, : lay.vh]
    pad[:, a_rel : a_rel + lay.vh] = ba[:, lay.vh :]
    return torch.cat([qkvz, pad], dim=1).contiguous()


def _shared(mlp: Any, g: Any, ex: ExecSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared expert `(sh13 [H, 2*isr], sh2 [isr, H])`, each half zero-padded to `isr`."""
    isr = g.IS // ex.R
    gu, down = mlp.shared_expert.gate_up_proj, mlp.shared_expert.down_proj
    n_gu = gu.weight.shape[1] if getattr(gu, WEIGHT_FLIPPED_ATTR, False) else gu.weight.shape[0]
    half = int(n_gu) // 2
    w_gu = kn(gu, g.H, 2 * half)
    sh13 = torch.cat(
        [F.pad(w_gu[:, :half], (0, isr - half)), F.pad(w_gu[:, half:], (0, isr - half))], dim=1
    )
    w_dn = kn(down, half, g.H)
    sh2 = F.pad(w_dn, (0, 0, 0, isr - half))
    return sh13.contiguous(), sh2.contiguous()


def head_groups(model: Any, g: Qwen38Geometry) -> list[list[int]]:
    """Layers before the first QSA layer, split so a PLE layer starts a group."""
    groups: list[list[int]] = []
    for i in range(g.att_every - 1):
        if not groups or model.model.layers[i].ple is not None:
            groups.append([])
        groups[-1].append(i)
    return groups


def pack_segment_operands(
    model: Any,
    g: Qwen38Geometry,
    s: int,
    spec: ModelSpec,
    ex: ExecSpec,
    layer_ids: list[int] | None = None,
    head: bool = False,
) -> list[torch.Tensor]:
    """This rank's segment-`s` (or head group `layer_ids`) weight operands.

    Flattened in `plan.w_names` order.
    """
    p = planner.plan(spec, ex)
    L = spec.config.L
    ids = segment_layers(g, s) if layer_ids is None else layer_ids
    layers = [model.model.layers[i] for i in ids]
    assert len(layers) == L, (len(layers), L)
    gi = g.inner(ex.R, L, head)
    gdn = layers if head else layers[1:]
    lay = GDNBlock.lin_in_layout(gi, ex) if gdn else None
    q = layers[0]
    dev = q.mlp.gate.weight.device
    W = g.NHC * g.H

    w: dict[str, Any] = {}
    sm: dict[str, torch.Tensor] = {}

    def row(name: str, j: int, val: torch.Tensor) -> None:
        if name not in sm:
            sm[name] = torch.zeros((L, p.smalls_layout[name][1]), dtype=torch.float32, device=dev)
        sm[name][j, : val.numel()] = val.reshape(-1).float()

    if not head:
        inj = q.attn_hyper_connection.block_inject_weight.weight.data.t()
        w["qsa_out_hinj"] = F.pad(inj, (0, 128 - g.NHC))[None]
        w["o_proj"] = kn(q.self_attn.o_proj, g.NH // ex.R * g.HD, g.H)[None]
        row("qsa_out_hcn", 0, q.attn_hyper_connection.hc_norm.weight.data)

    if gdn:
        hcs = [_hc_mix_operands(layer.attn_hyper_connection, g) for layer in gdn]
        w["gdn_hdn"] = torch.stack([h[0] for h in hcs])
        for b in range(g.NHC):
            w[f"gdn_hup{b}"] = torch.stack([h[1][b] for h in hcs])
        w["lin_in"] = torch.stack([_lin_in(layer.linear_attn, gi, ex) for layer in gdn])
        w["lin_out"] = torch.stack([kn(layer.linear_attn.out_proj, lay.Z, g.H) for layer in gdn])
        for j, layer in enumerate(gdn, start=L - len(gdn)):
            la = layer.linear_attn
            row("gdn_hcn", j, layer.attn_hyper_connection.hc_norm.weight.data)
            row("in_s", j, torch.ones(lay.NLI, device=dev))
            row("out_s", j, torch.ones(g.H, device=dev))
            row("conv", j, la.conv1d.weight.data.float().reshape(lay.QKV, 4).t())
            row("alog", j, la.A_log.data)
            row("dtb", j, la.dt_bias.data)
            row("gnorm", j, la.norm.weight.data)

    hcs = [_hc_mix_operands(layer.mlp_hyper_connection, g) for layer in layers]
    w["moe_hdn"] = torch.stack([h[0] for h in hcs])
    for b in range(g.NHC):
        w[f"moe_hup{b}"] = torch.stack([h[1][b] for h in hcs])
    w["router"] = torch.stack([kn(layer.mlp.gate, g.H, g.E) for layer in layers])
    shared = [_shared(layer.mlp, gi, ex) for layer in layers]
    w["sh13"] = torch.stack([sh[0] for sh in shared])
    w["sh2"] = torch.stack([sh[1] for sh in shared])
    e13, e2, es = [], [], []
    for layer in layers:
        a, b_, c = pack_expert_views_layer(layer.mlp, gi, ex, p.tk["e13"], p.tk["e2"])
        e13.append(a)
        e2.append(b_)
        es.append(c)
    w["e13"], w["e2"], w["es"] = tuple(e13), tuple(e2), torch.stack(es)
    for j, layer in enumerate(layers):
        row("moe_hcn", j, layer.mlp_hyper_connection.hc_norm.weight.data)
        row("sh_gate", j, layer.mlp.shared_expert_gate.weight.data)
        row("sh13_s", j, torch.ones(2 * (gi.IS // ex.R), device=dev))
        row("sh2_s", j, torch.ones(g.H, device=dev))

    S = p.smalls_width
    smalls = torch.zeros((L, 1, S), dtype=torch.float32, device=dev)
    for name, val in sm.items():
        o, width = p.smalls_layout[name]
        smalls[:, 0, o : o + width] = val
    out: list[torch.Tensor] = []
    for name in p.w_names:
        if name == "smalls":
            out.append(smalls)
        elif name == "final_norm":
            out.append(torch.zeros((1, W), dtype=torch.float32, device=dev))
        elif name in ("e13", "e2"):
            out.extend(w[name])
        elif name == "es":
            # [L, EW, 1, X] -> [L, EW, X]: no rank-4 operand at the torch boundary
            # (see below); unpack_local_weight restores the singleton under jit.
            t = w[name]
            out.append(t.reshape(*t.shape[:-3], t.shape[-3] * t.shape[-2], t.shape[-1]))
        else:
            nt, tk = p.nt[name], p.tk[name]
            t = w[name]
            k_in = t.shape[-2]
            if k_in < nt * tk:
                t = F.pad(t, (0, 0, 0, nt * tk - k_in))
            # Merged tile dims ([..., nt * tk, N]); the kernel's unpack_local_weight
            # splits them under jit. A rank-4 operand is re-laid out by torch_tpu
            # on every call (q40 j-c2430942: ~1.8 ms/step of execute_as_strided).
            out.append(t.contiguous())
    return out


def own_operands(ops: list[torch.Tensor]) -> tuple[list[torch.Tensor], int]:
    """Kernel operands as non-view tensors, and how many were views.

    torch_tpu re-materializes a view operand with an eager as_strided program
    on every call of the compiled program that takes it. A contiguous view at
    storage offset 0 is detached (same storage, no copy); any other view is
    cloned (only small operands, such as the expert-scale reshape, are views).
    """
    out, fixed = [], 0
    for t in ops:
        if t._base is None:
            out.append(t)
            continue
        fixed += 1
        if t.is_contiguous() and t.storage_offset() == 0:
            out.append(t.detach())
        else:
            out.append(t.clone())
    return out, fixed
