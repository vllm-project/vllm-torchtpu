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
"""Per-rank operands of the Qwen3.8 whole-step kernel from loaded vLLM `qwen4_exp`
modules.

The whole-step spec (`wholestep.spec.wholestep_spec(first="qsa")`) runs model layers
`first_layer .. first_layer + L - 1` with kernel layer `j` a QSA layer when `j %
att_every == 0`
and a GDN layer otherwise; every layer has the MoE sublayer. GDN / MoE / HC operands are
the
segment packer's (`qwen38_weights`); the QSA layer adds `qkvg` (`[q_h | gate_h]` per
local head
then the rank's K and V head, the stock `q_proj` / `k_proj` / `v_proj` views), `idx_w`
(`indexer.index_qk_proj`), `o_proj` and its norms.

Every operand is returned as its own freshly allocated base tensor unless it already is
a
whole contiguous parameter (the routed experts): a view (a `reshape` of a stack, a
transposed
weight) reaches the torch_tpu bridge through `execute_as_strided`, a full copy on every
call
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

    Passed as is: a contiguous, offset-0 tensor that is a base (e.g. a fresh stack) or a
    whole
    parameter (the routed-expert `w13` / `w2`). Any view — including a transposed
    parameter
    (`kn` of an `[n, k]` weight) — reaches torch_tpu through `execute_as_strided`, a
    copy on
    every call (agent-1a, q39: parameter views are part of the leftover copies).
    """
    plain = t.is_contiguous() and t.storage_offset() == 0
    if plain and (t._base is None or isinstance(t, torch.nn.Parameter)):
        return t
    return fresh_copy(t)


def fresh_copy(t: torch.Tensor) -> torch.Tensor:
    """A copy of `t` that is the fresh result of an op (`cat` of two halves along dim
    0), in
    `t`'s shape. `empty_like().copy_()` writes the new base in place, the history class
    torch_tpu
    re-materialises per call (w10 c1: the 32 MiB host round-trip covered 7 of 114
    operands; the
    ~18 big reshaped stacks still went through `copy_`). 0-d tensors are host
    round-tripped."""
    if t.dim() == 0:
        return t.cpu().to(t.device)
    if t.shape[0] < 2:
        return torch.cat([t, t[:0]], dim=0)
    return torch.cat(t.split((t.shape[0] + 1) // 2, dim=0), dim=0)


def qkvg_operand(attn: Any, g: Any, R: int) -> torch.Tensor:
    """`[H, nh * 2 * HD + 2 * HD]`: the rank's `q_proj` columns (per head `[q | gate]`),
    K, V."""
    nh, hd = g.NH // R, g.HD
    q = kn(attn.q_proj, g.H, nh * 2 * hd)
    k = kn(attn.k_proj, g.H, hd)
    v = kn(attn.v_proj, g.H, hd)
    return torch.cat([q, k, v], dim=1)


def _fp8_columns(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`w [..., K, N]` as FP8 E4M3 with per-column (`N`) f32 scales `[..., N]`."""
    wf = w.float()
    amax = wf.abs().amax(dim=-2)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    return (wf / scale.unsqueeze(-2)).to(torch.float8_e4m3fn), scale


def _hc_fp8(w: dict, row: Any, prefix: str, nhc: int, layer_js: list[int]) -> None:
    """Quantize `<prefix>_hdn` / `<prefix>_hup<b>` in `w` (FP8 HC weights, target layers)."""
    q, s = _fp8_columns(w[f"{prefix}_hdn"])
    w[f"{prefix}_hdn"] = q
    ups = []
    for b in range(nhc):
        qb, sb = _fp8_columns(w[f"{prefix}_hup{b}"])
        w[f"{prefix}_hup{b}"] = qb
        ups.append(sb)
    up_s = torch.cat(ups, dim=-1)
    for i, j in enumerate(layer_js):
        row(f"{prefix}_hdn_s", j, s[i])
        row(f"{prefix}_hup_s", j, up_s[i])


def pack_tail_operands(
    hc: Any, lm: Any, V: int, g: Qwen38Geometry, ex: ExecSpec
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The greedy tail's weights (`wholestep_spec(tail=...)`): the final HC mixer
    (`GatedResidual(use_combine=False)`: no injection, zero columns) and `lm_head` in
    bank-sized column blocks (`tail.vocab_chunks`, the last padded), both BF16 as
    loaded; returns `(weights, tail_hcn row)`."""
    from vllm_torchtpu.megakernel.wholestep.tail import vocab_chunk_width, vocab_chunks

    hdn, ups = _hc_mix_operands(hc, g)
    w = {"tail_hdn": hdn[None]}
    for b in range(g.NHC):
        w[f"tail_hup{b}"] = ups[b][None]
    lm_kn = kn(lm, g.H, V)
    for name, c0, width in vocab_chunks(V, vocab_chunk_width(ex)):
        blk = lm_kn[:, c0 : c0 + width]
        w[name] = F.pad(blk, (0, width - blk.shape[1]))[None]
    return w, hc.hc_norm.weight.data


def pack_wholestep_operands(
    model: Any,
    g: Qwen38Geometry,
    spec: ModelSpec,
    ex: ExecSpec,
    layer_ids: list[int],
    tail: tuple[Any, Any, int] | None = None,
) -> list[torch.Tensor]:
    """This rank's whole-step operands, flattened in `plan.w_names` order.

    `tail` = `(final HC mixer, lm_head, vocab shard)` for a spec with the greedy tail
    (`wholestep_spec(tail=...)`, the MTP draft step): its mixer and `lm_head` weights.
    """
    p = planner.plan(spec, ex)
    L = spec.config.L
    assert len(layer_ids) == L, (len(layer_ids), L)
    layers = [model.model.layers[i] for i in layer_ids]
    att = [j for j in range(L) if j % g.att_every == 0]
    gdn_j = [j for j in range(L) if j % g.att_every != 0]
    for j in att:
        assert layers[j].layer_type == "full_attention", (
            layer_ids[j],
            layers[j].layer_type,
        )
    for j in gdn_j:
        assert layers[j].layer_type == "linear_attention", (
            layer_ids[j],
            layers[j].layer_type,
        )
    gi = g.inner(ex.R, L)
    lay = GDNBlock.lin_in_layout(gi, ex) if gdn_j else None
    dev = layers[0].mlp.gate.weight.device

    w: dict[str, Any] = {}
    sm: dict[str, torch.Tensor] = {}

    def row(name: str, j: int, val: torch.Tensor) -> None:
        if name not in sm:
            sm[name] = torch.zeros(
                (L, p.smalls_layout[name][1]), dtype=torch.float32, device=dev
            )
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
        assert lay is not None
        gl = [layers[j] for j in gdn_j]
        hcs = [_hc_mix_operands(layer.attn_hyper_connection, g) for layer in gl]
        w["gdn_hdn"] = torch.stack([h[0] for h in hcs])
        for b in range(g.NHC):
            w[f"gdn_hup{b}"] = torch.stack([h[1][b] for h in hcs])
        w["lin_in"] = torch.stack([_lin_in(layer.linear_attn, gi, ex) for layer in gl])
        w["lin_out"] = torch.stack(
            [kn(layer.linear_attn.out_proj, lay.Z, g.H) for layer in gl]
        )
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

    if tail is not None:  # final HC mixer (last layer's slot) + lm_head [H, V shard]
        tw, hcn = pack_tail_operands(*tail, g, ex)
        w.update(tw)
        row("tail_hcn", L - 1, hcn)

    if "moe_hdn_s" in p.smalls_layout:  # VLLM_TPU_MK_HC_FP8 (spec hc_fp8)
        _hc_fp8(w, row, "qsa_full", g.NHC, att)
        if gdn_j:
            _hc_fp8(w, row, "gdn", g.NHC, gdn_j)
        _hc_fp8(w, row, "moe", g.NHC, list(range(L)))

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
        elif (
            name == "es"
        ):  # [L, EW, 1, X] -> 3-D [L*EW, 1, X]; the call reshapes it back
            out.append(w[name].reshape(-1, *w[name].shape[2:]))
        else:
            nt, tk = p.nt[name], p.tk[name]
            t = w[name]
            k_in = t.shape[-2]
            if k_in < nt * tk:
                t = F.pad(t, (0, 0, 0, nt * tk - k_in))
            # 3-D [L*nt, tk, N] (the call reshapes to [L, nt, tk, N]): torch_tpu re-lays
            # out
            # 4-D operands through execute_as_strided on every call.
            out.append(t.reshape(-1, tk, t.shape[-1]))
    return [own_operand(t) for t in out]
