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
"""Per-rank operands of the Qwen3.8 whole-step kernel from loaded vLLM `qwen4_exp` modules.

The whole-step spec (`wholestep.spec.wholestep_spec(first="qsa")`) runs model layers
`first_layer .. first_layer + L - 1` with kernel layer `j` a QSA layer when `j % att_every == 0`
and a GDN layer otherwise; every layer has the MoE sublayer. GDN / MoE / HC operands are the
segment packer's (`qwen38_weights`); the QSA layer adds `qkvg` (`[q_h | gate_h]` per local head
then the rank's K and V head, the stock `q_proj` / `k_proj` / `v_proj` views), `idx_w`
(`indexer.index_qk_proj`), `o_proj` and its norms.

Every operand is returned as its own freshly allocated base tensor unless it already is a
whole contiguous parameter (the routed experts): a view (a `reshape` of a stack, a transposed
weight) reaches the torch_tpu bridge through `execute_as_strided`, a full copy on every call
(q32: 1.8 ms/step).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from vllm_torchtpu.megakernel import planner
from vllm_torchtpu.megakernel.blocks.gdn import GDNBlock
from vllm_torchtpu.megakernel.exec_spec import ExecSpec
from vllm_torchtpu.megakernel.models.qwen38 import Qwen38Geometry
from vllm_torchtpu.megakernel.spec import ModelSpec

from .qwen38_weights import _hc_mix_operands, _lin_in, _shared, kn
from .weight_collector import pack_expert_views_layer


def own_operand(t: torch.Tensor) -> torch.Tensor:
    """`t` itself if the bridge can pass it as is, else a fresh contiguous base tensor.

    Passed as is: a contiguous, offset-0 tensor that is a base (e.g. a fresh stack) or a whole
    parameter (the routed-expert `w13` / `w2`). Any view — including a transposed parameter
    (`kn` of an `[n, k]` weight) — reaches torch_tpu through `execute_as_strided`, a copy on
    every call (agent-1a, q39: parameter views are part of the leftover copies).
    """
    plain = t.is_contiguous() and t.storage_offset() == 0
    if plain and (t._base is None or isinstance(t, torch.nn.Parameter)):
        return t
    return torch.empty_like(t, memory_format=torch.contiguous_format).copy_(t)


def qkvg_operand(attn: Any, g: Any, R: int) -> torch.Tensor:
    """`[H, nh * 2 * HD + 2 * HD]`: the rank's `q_proj` columns (per head `[q | gate]`), K, V."""
    nh, hd = g.NH // R, g.HD
    q = kn(attn.q_proj, g.H, nh * 2 * hd)
    k = kn(attn.k_proj, g.H, hd)
    v = kn(attn.v_proj, g.H, hd)
    return torch.cat([q, k, v], dim=1)


def pack_wholestep_operands(
    model: Any,
    g: Qwen38Geometry,
    spec: ModelSpec,
    ex: ExecSpec,
    layer_ids: list[int],
) -> list[torch.Tensor]:
    """This rank's whole-step operands, flattened in `plan.w_names` order."""
    p = planner.plan(spec, ex)
    L = spec.config.L
    assert len(layer_ids) == L, (len(layer_ids), L)
    layers = [model.model.layers[i] for i in layer_ids]
    att = [j for j in range(L) if j % g.att_every == 0]
    gdn_j = [j for j in range(L) if j % g.att_every != 0]
    for j in att:
        assert layers[j].layer_type == "full_attention", (layer_ids[j], layers[j].layer_type)
    for j in gdn_j:
        assert layers[j].layer_type == "linear_attention", (layer_ids[j], layers[j].layer_type)
    gi = g.inner(ex.R, L)
    lay = GDNBlock.lin_in_layout(gi, ex) if gdn_j else None
    dev = layers[0].mlp.gate.weight.device

    w: dict[str, Any] = {}
    sm: dict[str, torch.Tensor] = {}

    def row(name: str, j: int, val: torch.Tensor) -> None:
        if name not in sm:
            sm[name] = torch.zeros((L, p.smalls_layout[name][1]), dtype=torch.float32, device=dev)
        sm[name][j, : val.numel()] = val.reshape(-1).float()

    # QSA layers: HC(attention) mix, projections, indexer, o_proj, norms.
    hcs = [_hc_mix_operands(layers[j].attn_hyper_connection, g) for j in att]
    w["qsa_full_hdn"] = torch.stack([h[0] for h in hcs])
    for b in range(g.NHC):
        w[f"qsa_full_hup{b}"] = torch.stack([h[1][b] for h in hcs])
    w["qkvg"] = torch.stack([qkvg_operand(layers[j].self_attn, g, ex.R) for j in att])
    w["idx_w"] = torch.stack(
        [kn(layers[j].self_attn.indexer.index_qk_proj, g.H, 5 * 128) for j in att]
    )
    w["o_proj"] = torch.stack(
        [kn(layers[j].self_attn.o_proj, g.NH // ex.R * g.HD, g.H) for j in att]
    )
    for j in att:
        a = layers[j].self_attn
        row("qsa_full_hcn", j, layers[j].attn_hyper_connection.hc_norm.weight.data)
        row("qsa_qn", j, a.q_norm.weight.data)
        row("qsa_kn", j, a.k_norm.weight.data)
        row("qsa_iqn", j, a.indexer.q_layernorm.weight.data)
        row("qsa_ikn", j, a.indexer.k_layernorm.weight.data)

    # GDN layers (as the segment packer).
    if gdn_j:
        gl = [layers[j] for j in gdn_j]
        hcs = [_hc_mix_operands(layer.attn_hyper_connection, g) for layer in gl]
        w["gdn_hdn"] = torch.stack([h[0] for h in hcs])
        for b in range(g.NHC):
            w[f"gdn_hup{b}"] = torch.stack([h[1][b] for h in hcs])
        w["lin_in"] = torch.stack([_lin_in(layer.linear_attn, gi, ex) for layer in gl])
        w["lin_out"] = torch.stack([kn(layer.linear_attn.out_proj, lay.Z, g.H) for layer in gl])
        for j in gdn_j:
            la = layers[j].linear_attn
            row("gdn_hcn", j, layers[j].attn_hyper_connection.hc_norm.weight.data)
            row("in_s", j, torch.ones(lay.NLI, device=dev))
            row("out_s", j, torch.ones(g.H, device=dev))
            row("conv", j, la.conv1d.weight.data.float().reshape(lay.QKV, 4).t())
            row("alog", j, la.A_log.data)
            row("dtb", j, la.dt_bias.data)
            row("gnorm", j, la.norm.weight.data)

    # MoE (every layer).
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

    smalls = torch.zeros((L, 1, p.smalls_width), dtype=torch.float32, device=dev)
    for name, val in sm.items():
        o, width = p.smalls_layout[name]
        smalls[:, 0, o : o + width] = val
    out: list[torch.Tensor] = []
    for name in p.w_names:
        if name == "smalls":
            out.append(smalls)
        elif name == "final_norm":
            out.append(torch.zeros((1, g.NHC * g.H), dtype=torch.float32, device=dev))
        elif name in ("e13", "e2"):
            out.extend(w[name])
        elif name == "es":  # [L, EW, 1, X] -> 3-D [L*EW, 1, X]; the call reshapes it back
            out.append(w[name].reshape(-1, *w[name].shape[2:]))
        else:
            nt, tk = p.nt[name], p.tk[name]
            t = w[name]
            k_in = t.shape[-2]
            if k_in < nt * tk:
                t = F.pad(t, (0, 0, 0, nt * tk - k_in))
            # 3-D [L*nt, tk, N] (the call reshapes to [L, nt, tk, N]): torch_tpu re-lays out
            # 4-D operands through execute_as_strided on every call.
            out.append(t.reshape(-1, tk, t.shape[-1]))
    return [own_operand(t) for t in out]
