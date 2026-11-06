# SPDX-License-Identifier: Apache-2.0
# Based on vLLM v0.32.0 nemotron_h.py.
"""NemotronHForCausalLM on vllm-torchtpu: vLLM's nemotron_h.py with the two CUDA-only blocks replaced.

* `M` layers -> TPUMambaMixer2: vLLM's MambaMixer2 (same parameters, TP sharding, MambaSpec state
  shapes and dtypes) with a pure-torch forward over the ragged token batch: causal depthwise conv1d (left
  context from the conv state) and chunked SSD in matmul form (chunks masked to one request, per-request
  initial state, final conv/SSM state written back to the request's slot). No Triton/CUDA ops.
* `E` layers -> TPUNemotronHMoE: fp32 sigmoid router + fp32 e_score_correction_bias (selection only),
  non-gated relu2 experts sharded over the TP group (expert parallel), shared relu2 MLP (vLLM's
  NemotronHMLP, reduce_results=False), one fp32 all-reduce. Bypasses FusedMoE, whose config rejects
  is_act_and_mul=False off CUDA/XPU/ROCm (fused_moe/config.py:1402).
* `*` NoPE GQA attention, `-` dense MLP, embedding, norms, lm_head: vLLM's own layers.

State: the TPU runner's per-layer Mamba pools (runner/kv_cache_manager.py: each MambaSpec layer gets
(conv [S, *conv_shape], ssm [S, H_l, P, N]) with S = max_num_reqs + 1 and slot 0 the null slot); the
request -> slot map arrives as AttentionMetadata.mamba_state_indices. Serving contract:
--no-enable-prefix-caching (a request's state is never shared), no speculative decoding, no ReplaySSM.
dt is not clamped (vLLM v0.32.0 never clamps).
"""

import math
import os
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models import nemotron_h as nh
from vllm.model_executor.models.utils import maybe_prefix

# Expert back-end for the routed experts: "dense" (every local expert on every token; static shapes,
# plain matmuls) or "grouped_mm" (sort by expert + torch._grouped_mm; needs backend support).
MOE_IMPL_ENV = "VLLM_TORCHTPU_NEMOTRON_MOE_IMPL"


# ----------------------------------------------------------------------------- Mamba2 on a ragged batch
def mamba2_ragged(xbc, dt_raw, conv_w, conv_b, A, D, dt_bias, conv_pool, ssm_pool, query_start_loc, seq_lens,
                  state_indices, *, num_heads, head_dim, n_groups, d_state, chunk_size, conv_dim_first):
    """Conv1d + SSD for one engine step; reads and writes the per-layer slot pools in place.

    xbc [T, C] (pre-conv x|B|C, activation dtype), dt_raw [T, H]; conv_w [C, 1, K], conv_b [C] | None;
    A [H] (= -exp(A_log), fp32), D, dt_bias [H]; conv_pool [S, C, K-1] (dim-first) or [S, K-1, C];
    ssm_pool [S, H, P, N]; query_start_loc [R+1], seq_lens [R] (computed + scheduled), state_indices [R]
    (slot per request; padded requests have q_len 0 and write only the null slot 0).
    Returns y [T, H, P] in promote(dtype, fp32), D skip included. Every shape is static in (T, R).
    """
    dev = xbc.device
    T, Cd = xbc.shape
    R = seq_lens.shape[0]
    K = conv_w.shape[-1]
    H, P, G, N = num_heads, head_dim, n_groups, d_state
    Rh = H // G
    ad = torch.promote_types(xbc.dtype, torch.float32)
    zero = torch.zeros((), dtype=ad, device=dev)

    qsl = query_start_loc[: R + 1].to(torch.long)
    starts, ends = qsl[:-1], qsl[1:]
    q_len = ends - starts
    active = q_len > 0
    has_init = active & ((seq_lens.to(torch.long) - q_len) > 0)  # first chunk of a (re)started request: zero state
    slot = torch.where(active, state_indices.to(torch.long), torch.zeros_like(q_len))
    tpos = torch.arange(T, device=dev)
    rid = (ends.unsqueeze(0) <= tpos.unsqueeze(1)).sum(-1)  # owning request per token; R for the padded tail
    valid = rid < R
    rid_c = rid.clamp(max=R - 1)
    p = tpos - starts.index_select(0, rid_c)  # position inside the request's chunk of this step

    # ---- conv1d: tap j multiplies input t - (K-1) + j; inputs before the chunk come from the conv state
    conv_rows = conv_pool.index_select(0, slot)
    if conv_dim_first:
        conv_rows = conv_rows.transpose(1, 2)  # -> [R, K-1, C]
    # where, not multiply: a reused slot may hold anything (0 * NaN = NaN)
    conv_init = torch.where(has_init.view(R, 1, 1), conv_rows.to(ad), zero)
    xf = xbc.to(ad)
    w = conv_w.reshape(Cd, K).to(ad)
    init_tok = conv_init.index_select(0, rid_c)  # [T, K-1, C]
    acc = xf * w[:, K - 1]
    for d in range(1, K):  # K static taps, not tokens
        shifted = F.pad(xf, (0, 0, d, 0))[:T]  # x[t - d]
        sidx = (K - 1 - d + p).clamp(0, K - 2).view(T, 1, 1).expand(T, 1, Cd)
        from_state = torch.gather(init_tok, 1, sidx).squeeze(1)
        acc = acc + w[:, K - 1 - d] * torch.where((p >= d).unsqueeze(1), shifted, from_state)
    if conv_b is not None:
        acc = acc + conv_b.to(ad)
    xconv = F.silu(acc).to(xbc.dtype)  # the scan reads the conv output in the activation dtype (vLLM, HF)

    # new conv state: the last K-1 inputs of [state | chunk] per request
    m = torch.arange(K - 1, device=dev)
    u = q_len.unsqueeze(1) - (K - 1) + m.unsqueeze(0)  # [R, K-1], offset from the chunk start
    tok = (starts.unsqueeze(1) + u).clamp(0, T - 1)
    from_x = xbc.index_select(0, tok.reshape(-1)).view(R, K - 1, Cd).to(ad)
    from_s = torch.gather(conv_init, 1, (K - 1 + u).clamp(0, K - 2).unsqueeze(-1).expand(R, K - 1, Cd))
    new_conv = torch.where((u >= 0).unsqueeze(-1), from_x, from_s)
    if conv_dim_first:
        new_conv = new_conv.transpose(1, 2)
    conv_pool.index_copy_(0, slot, new_conv.to(conv_pool.dtype))

    # ---- SSD inputs
    x = xconv[:, : H * P].to(ad).view(T, H, P)
    Bm = xconv[:, H * P: H * P + G * N].to(ad).view(T, G, N)
    Cm = xconv[:, H * P + G * N: H * P + 2 * G * N].to(ad).view(T, G, N)
    dt = F.softplus(dt_raw.to(ad) + dt_bias.to(ad))  # no clamp (vLLM v0.32.0 never clamps)
    dt = torch.where(valid.unsqueeze(1), dt, zero)  # padded tokens neither decay nor write any state
    a = dt * A.to(ad)
    h0 = torch.where(has_init.view(R, 1, 1, 1), ssm_pool.index_select(0, slot).to(ad), zero)  # [R, H, P, N]

    L = min(chunk_size, T)
    nc = -(-T // L)
    Tp = nc * L
    seg = rid
    if Tp != T:
        pad = Tp - T
        x, Bm, Cm = F.pad(x, (0, 0, 0, 0, 0, pad)), F.pad(Bm, (0, 0, 0, 0, 0, pad)), F.pad(Cm, (0, 0, 0, 0, 0, pad))
        dt, a = F.pad(dt, (0, 0, 0, pad)), F.pad(a, (0, 0, 0, pad))
        seg = torch.cat([rid, torch.full((pad,), R, dtype=rid.dtype, device=dev)])
    xc = x.view(nc, L, G, Rh, P)
    Bc = Bm.view(nc, L, G, N)
    Cc = Cm.view(nc, L, G, N)
    segc = seg.view(nc, L)
    cs = torch.cumsum(a.view(nc, L, G, Rh), dim=1)  # within-chunk cumulative log decay
    xdt = xc * dt.view(nc, L, G, Rh).unsqueeze(-1)

    # 1. intra-chunk, same request only: y_l = sum_{s<=l, seg(s)=seg(l)} (C_l.B_s) exp(cs_l - cs_s) dt_s x_s
    causal = torch.ones(L, L, dtype=torch.bool, device=dev).tril()
    same = (segc.unsqueeze(2) == segc.unsqueeze(1)) & causal  # [nc, L, L]
    csT = cs.permute(0, 2, 3, 1)  # [nc, G, Rh, L]
    decay = torch.exp((csT.unsqueeze(-1) - csT.unsqueeze(-2)).masked_fill(~same.view(nc, 1, 1, L, L), -math.inf))
    cb = torch.einsum("clgn,csgn->cgls", Cc, Bc)  # shared by the Rh heads of a group
    y = torch.einsum("cgrls,csgrp->clgrp", cb.unsqueeze(2) * decay, xdt)

    # a token's entering state: its request's h0 if the request starts inside this chunk, else the state
    # carried across the chunk boundary; decay_in is exp(sum of a from that entry point to the token)
    tok_start = qsl.index_select(0, seg)  # qsl[R] = total for padded tokens
    base = torch.arange(Tp, device=dev) // L * L
    starts_here = (tok_start >= base).view(nc, L)
    b = (tok_start - base).clamp(min=0).view(nc, L, 1, 1).expand(nc, L, G, Rh)
    cs_before = torch.gather(F.pad(cs, (0, 0, 0, 0, 1, 0)), 1, b)  # cs just before the entry point (0 at chunk start)
    decay_in = torch.exp(cs - cs_before)  # [nc, L, G, Rh]

    # 2. state carried into each chunk by the request that crosses its start boundary
    h0g = h0.view(R, G, Rh, P, N)
    h0_ext = torch.cat([h0g, h0g.new_zeros(1, G, Rh, P, N)])  # row R: padded tokens
    last_seg = segc[:, -1]
    w_end = torch.exp(cs[:, -1:] - cs) * (segc == last_seg.unsqueeze(1)).view(nc, L, 1, 1).to(ad)
    intra_end = torch.einsum("clgn,clgrp->cgrpn", Bc, xdt * w_end.unsqueeze(-1))  # [nc, G, Rh, P, N]
    h0_last = h0_ext.index_select(0, last_seg)
    s_in = [h0g.new_zeros(G, Rh, P, N)]
    for c in range(nc - 1):  # static, T / L chunks
        entering = torch.where(starts_here[c, -1], h0_last[c], s_in[c])
        s_in.append(intra_end[c] + entering * decay_in[c, -1].view(G, Rh, 1, 1))
    S_in = torch.stack(s_in)  # [nc, G, Rh, P, N]

    # 3. outputs from entering states: carried state (crossing requests) ...
    cross = (~starts_here).view(nc, L, 1, 1).to(ad)
    y = y + torch.einsum("clgn,cgrpn->clgrp", Cc, S_in) * (decay_in * cross).unsqueeze(-1)
    # ... and h0 for the tokens of each request inside the chunk where it starts (a window of <= L tokens)
    j = torch.arange(L, device=dev)
    win = starts.unsqueeze(1) + j  # [R, L]
    win_ok = (win < ends.unsqueeze(1)) & (win < (starts // L + 1).unsqueeze(1) * L)
    win_c = win.clamp(max=Tp - 1).reshape(-1)
    Cw = Cm.index_select(0, win_c).view(R, L, G, N)
    dw = decay_in.reshape(Tp, G, Rh).index_select(0, win_c).view(R, L, G, Rh) * win_ok.view(R, L, 1, 1).to(ad)
    y0 = torch.einsum("rlgn,rgkpn->rlgkp", Cw, h0g) * dw.unsqueeze(-1)
    y = y.reshape(Tp, H, P).index_add(0, win_c, y0.reshape(R * L, H, P))  # masked rows add 0
    y = y + x * D.to(ad).view(1, H, 1)

    # 4. final state per request: entering state decayed to its last token + its last-chunk tokens
    e = (ends - 1).clamp(min=0)
    ce = e // L
    win2 = e.unsqueeze(1) - (L - 1) + j
    win2_ok = (win2 >= torch.maximum(starts, ce * L).unsqueeze(1)) & active.unsqueeze(1)
    win2_c = win2.clamp(min=0).reshape(-1)
    cs_flat = cs.reshape(Tp, G, Rh)
    cs_w = cs_flat.index_select(0, win2_c).view(R, L, G, Rh)
    wgt = torch.exp((cs_flat.index_select(0, e).unsqueeze(1) - cs_w).masked_fill(~win2_ok.view(R, L, 1, 1), -math.inf))
    xdt_w = xdt.reshape(Tp, G, Rh, P).index_select(0, win2_c).view(R, L, G, Rh, P)
    Bw = Bc.reshape(Tp, G, N).index_select(0, win2_c).view(R, L, G, N)
    h_end = torch.einsum("rlgn,rlgkp->rgkpn", Bw, xdt_w * wgt.unsqueeze(-1))
    entering = torch.where((starts >= ce * L).view(R, 1, 1, 1, 1), h0g, S_in.index_select(0, ce))
    h_end = h_end + entering * decay_in.reshape(Tp, G, Rh).index_select(0, e).view(R, G, Rh, 1, 1)
    ssm_pool.index_copy_(0, slot, h_end.reshape(R, H, P, N).to(ssm_pool.dtype))
    return y[:T]


class TPUMambaMixer2(MambaMixer2):
    """vLLM MambaMixer2 parameters and state spec; forward = mamba2_ragged on the engine's slot pools."""

    def __init__(self, *args, chunk_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_spec or self.use_replayssm:
            raise NotImplementedError("TPUMambaMixer2: no speculative decoding / ReplaySSM state rings")
        if self.n_groups % self.tp_size or self.norm.n_groups % self.tp_size:
            raise NotImplementedError("TPUMambaMixer2: n_groups must divide by TP (group-local gated norm)")
        self.chunk_size = chunk_size
        self.conv_dim_first = is_conv_state_dim_first()
        self.local_heads = self.num_heads // self.tp_size
        self.local_groups = self.n_groups // self.tp_size

    def forward(self, hidden_states: torch.Tensor, mup_vector: torch.Tensor | None = None) -> torch.Tensor:
        projected, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected = projected * mup_vector
        il, cl, hl = self.tped_intermediate_size, self.tped_conv_size, self.tped_dt_size
        gate = projected[:, :il]
        xbc = projected[:, il: il + cl]
        dt_raw = projected[:, il + cl: il + cl + hl]
        num_tokens = hidden_states.shape[0]
        kv = self.kv_cache
        if kv is None or len(kv) == 0 or kv[0].numel() == 0:
            # Profiling / warm-up before the runner binds the state pools: nothing to read or write.
            y = torch.zeros(num_tokens, il, dtype=torch.float32, device=hidden_states.device)
        else:
            if len(kv) != 2:
                raise NotImplementedError("TPUMambaMixer2 reads the per-layer (conv, ssm) pools, not the unified pool")
            conv_pool, ssm_pool = kv
            md = get_forward_context().attn_metadata[self.prefix]
            slots = md.mamba_state_indices
            if slots is None:  # uniform layout: mamba shares the attention block pool, slot = first block
                slots = md.block_tables.reshape(md.seq_lens.shape[0], -1)[:, 0]
            y = mamba2_ragged(
                xbc, dt_raw, self.conv1d.weight, self.conv1d.bias, self.A, self.D, self.dt_bias,
                conv_pool, ssm_pool, md.query_start_loc, md.seq_lens, slots,
                num_heads=self.local_heads, head_dim=self.head_dim, n_groups=self.local_groups,
                d_state=self.ssm_state_size, chunk_size=self.chunk_size, conv_dim_first=self.conv_dim_first,
            ).reshape(num_tokens, il)
        # gated RMSNorm, group-local (HF NemotronHRMSNormGated / vLLM Mixer2RMSNormGated, norm_before_gate=False)
        ad = torch.promote_types(hidden_states.dtype, torch.float32)
        v = y.to(ad) * F.silu(gate.to(ad))
        vg = v.view(num_tokens, -1, self.norm.group_size)
        v = (vg * torch.rsqrt(vg.pow(2).mean(-1, keepdim=True) + self.norm.variance_epsilon)).view(num_tokens, il)
        out, _ = self.out_proj(self.norm.weight * v.to(hidden_states.dtype))
        return out


# ----------------------------------------------------------------------------- relu2 MoE
def _relu2(x):
    return torch.square(torch.relu(x))  # HF ReLUSquaredActivation: relu then square, in the input dtype


def route(x, gate_w, bias, *, top_k, n_group, topk_group, norm_topk_prob, routed_scaling_factor):
    """HF NemotronHTopkRouter: fp32 sigmoid scores; the bias picks experts, the unbiased scores weight them."""
    rdt = torch.promote_types(x.dtype, torch.float32)
    scores = F.linear(x.to(rdt), gate_w.to(rdt)).sigmoid()
    choice = scores + bias.to(rdt)
    if n_group > 1:  # group-limited routing; the released config has n_group = topk_group = 1
        t, ne = choice.shape
        gs = choice.view(t, n_group, ne // n_group).topk(2, dim=-1)[0].sum(-1)
        gmask = torch.zeros_like(gs).scatter_(1, gs.topk(topk_group, dim=-1)[1], 1.0)
        choice = choice.masked_fill(gmask.repeat_interleave(ne // n_group, dim=1) == 0, float("-inf"))
    idx = torch.topk(choice, top_k, dim=-1).indices
    w = scores.gather(1, idx)
    if norm_topk_prob:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * routed_scaling_factor, idx


def experts_dense(x, w, idx, up, down, expert_offset):
    """Every local expert on every token, weighted by a dense [T, E_l] routing matrix (fp32 partial [T, H])."""
    el = up.shape[0]
    onehot = (idx - expert_offset).unsqueeze(-1) == torch.arange(el, device=idx.device)
    wd = (onehot.to(w.dtype) * w.unsqueeze(-1)).sum(1)
    h = _relu2(torch.matmul(x.unsqueeze(0), up.transpose(1, 2)))  # [E_l, T, I]
    y = torch.matmul(h, down.transpose(1, 2))  # [E_l, T, H]
    return torch.einsum("eth,te->th", y.to(w.dtype), wd)


def experts_grouped_mm(x, w, idx, up, down, expert_offset):
    """(token, k) pairs sorted by local expert -> two torch._grouped_mm; non-local pairs masked with where."""
    t, k = idx.shape
    el = up.shape[0]
    loc = idx.reshape(-1) - expert_offset
    ok = (loc >= 0) & (loc < el)
    key = torch.where(ok, loc, torch.full_like(loc, el))
    order = torch.argsort(key, stable=True)
    revert = torch.argsort(order)
    offs = torch.cumsum((key.unsqueeze(1) == torch.arange(el, device=key.device)).sum(0), 0).to(torch.int32)
    h = _relu2(torch._grouped_mm(x.index_select(0, order // k), up.transpose(1, 2), offs=offs))
    y = torch._grouped_mm(h, down.transpose(1, 2), offs=offs).index_select(0, revert).view(t, k, -1)
    return torch.where(ok.view(t, k, 1), y.to(w.dtype) * w.unsqueeze(-1), 0.0).sum(1)


EXPERT_IMPLS = {"dense": experts_dense, "grouped_mm": experts_grouped_mm}


class TPUNemotronHRouter(nn.Module):
    """`mixer.gate`: router weight [E, H] and the fp32 e_score_correction_bias (plain parameters, so no
    platform linear method can re-layout or down-cast them)."""

    def __init__(self, hidden_size: int, n_experts: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_experts, hidden_size), requires_grad=False)
        self.e_score_correction_bias = nn.Parameter(torch.empty(n_experts, dtype=torch.float32), requires_grad=False)


class TPURelu2Experts(nn.Module):
    """This rank's experts [offset, offset + E/tp) in HF layout: up [E_l, I, H], down [E_l, H, I]."""

    def __init__(self, n_experts: int, hidden_size: int, intermediate_size: int, tp_rank: int, tp_size: int):
        super().__init__()
        if n_experts % tp_size:
            raise NotImplementedError(f"{n_experts} experts do not split over TP={tp_size}")
        self.num_local = n_experts // tp_size
        self.expert_offset = tp_rank * self.num_local
        self.up = nn.Parameter(torch.empty(self.num_local, intermediate_size, hidden_size), requires_grad=False)
        self.down = nn.Parameter(torch.empty(self.num_local, hidden_size, intermediate_size), requires_grad=False)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded = set()
        for name, w in weights:  # "<expert>.up_proj.weight" | "<expert>.down_proj.weight"
            e, proj, _ = name.split(".")
            local = int(e) - self.expert_offset
            if not 0 <= local < self.num_local:
                continue  # another EP rank's expert
            param = {"up_proj": self.up, "down_proj": self.down}[proj]
            param.data[local].copy_(w)
            loaded.add("up" if proj == "up_proj" else "down")
        return loaded


class TPUNemotronHMoE(nn.Module):
    """NemotronH MoE block (sigmoid+bias top-k, non-gated relu2 experts, shared relu2 MLP) on plain torch."""

    def __init__(self, config, quant_config=None, parallel_config=None, prefix: str = ""):
        super().__init__()
        if getattr(config, "moe_latent_size", None) is not None or config.mlp_hidden_act != "relu2":
            raise NotImplementedError("TPUNemotronHMoE: relu2 experts without latent projections only")
        if parallel_config is not None and (parallel_config.enable_eplb or parallel_config.use_sequence_parallel_moe):
            raise NotImplementedError("TPUNemotronHMoE: no EPLB / sequence-parallel MoE")
        self.tp_size = get_tensor_model_parallel_world_size()
        self.top_k = config.num_experts_per_tok
        self.n_group, self.topk_group = config.n_group, config.topk_group
        self.norm_topk_prob, self.routed_scaling_factor = config.norm_topk_prob, config.routed_scaling_factor
        self.impl = os.environ.get(MOE_IMPL_ENV, "dense")
        if self.impl not in EXPERT_IMPLS:
            raise ValueError(f"{MOE_IMPL_ENV}={self.impl!r} not in {sorted(EXPERT_IMPLS)}")
        self.gate = TPUNemotronHRouter(config.hidden_size, config.n_routed_experts)
        self.experts = TPURelu2Experts(config.n_routed_experts, config.hidden_size, config.moe_intermediate_size,
                                           get_tensor_model_parallel_rank(), self.tp_size)
        self.shared_experts = None
        if config.n_shared_experts:
            self.shared_experts = nh.NemotronHMLP(
                config=config, hidden_size=config.hidden_size,
                intermediate_size=config.moe_shared_expert_intermediate_size * config.n_shared_experts,
                quant_config=quant_config, reduce_results=False, prefix=f"{prefix}.shared_experts")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.gate.e_score_correction_bias.dtype != torch.float32:  # trace-time check, no device sync
            raise RuntimeError("e_score_correction_bias must stay fp32 (bf16 erases its per-expert part)")
        w, idx = route(hidden_states, self.gate.weight, self.gate.e_score_correction_bias, top_k=self.top_k,
                       n_group=self.n_group, topk_group=self.topk_group, norm_topk_prob=self.norm_topk_prob,
                       routed_scaling_factor=self.routed_scaling_factor)
        out = EXPERT_IMPLS[self.impl](hidden_states, w, idx, self.experts.up, self.experts.down,
                                      self.experts.expert_offset)  # fp32 partial over this rank's experts
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states).to(out.dtype)  # this rank's column/row shard
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)  # one reduce for routed + shared
        return out.to(hidden_states.dtype)


# ----------------------------------------------------------------------------- decoder layers + model
class TPUNemotronHMambaDecoderLayer(nh.NemotronHMambaDecoderLayer):
    def __init__(self, config, layer_idx, model_config=None, cache_config=None, quant_config=None,
                 parallel_config=None, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = config
        self.mixer = TPUMambaMixer2(
            hidden_size=config.hidden_size, ssm_state_size=config.ssm_state_size, conv_kernel_size=config.conv_kernel,
            intermediate_size=config.mamba_num_heads * config.mamba_head_dim, use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias, n_groups=config.n_groups, num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim, rms_norm_eps=config.layer_norm_epsilon,
            activation=config.mamba_hidden_act, model_config=model_config, cache_config=cache_config,
            quant_config=quant_config, prefix=f"{prefix}.mixer", chunk_size=config.chunk_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


class TPUNemotronHMoEDecoderLayer(nh.NemotronHMoEDecoderLayer):
    def __init__(self, config, layer_idx, model_config=None, cache_config=None, quant_config=None,
                 parallel_config=None, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = config
        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config
        self.mixer = TPUNemotronHMoE(layer_config, quant_config=quant_config, parallel_config=parallel_config,
                                         prefix=f"{prefix}.mixer")
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)


TPU_LAYER_TYPES = {**nh.ALL_DECODER_LAYER_TYPES, "M": TPUNemotronHMambaDecoderLayer,
                       "E": TPUNemotronHMoEDecoderLayer}


class TPUNemotronHForCausalLM(nh.NemotronHForCausalLM):
    """vLLM's NemotronHForCausalLM (weight mapper, Mamba state shape/dtype classmethods, IsHybrid) with the
    TPU `M` / `E` layers; TP only (no PP, LoRA, EAGLE, EPLB)."""

    def __init__(self, *, vllm_config, prefix: str = ""):
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        if cache_config is not None and cache_config.enable_prefix_caching:
            raise NotImplementedError("TPUNemotronHForCausalLM serves with --no-enable-prefix-caching")
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = vllm_config.quant_config
        self.config = config
        self.model = nh.NemotronHModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"),
                                       decoder_layer_types=TPU_LAYER_TYPES)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, quant_config=self.quant_config,
                                      prefix=maybe_prefix(prefix, "lm_head"))
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        # MixtureOfExperts metadata drives EPLB / FusedMoE expert maps only; this MoE has neither.
        self.num_moe_layers = 0
        self.moe_layers = []
        # Delivery evidence: one line per engine worker that builds this class.
        print(f"[armed] TPUNemotronHForCausalLM pid={os.getpid()} tp_rank={get_tensor_model_parallel_rank()} "
              f"layers={config.hybrid_override_pattern} moe_impl={os.environ.get(MOE_IMPL_ENV, 'dense')}", flush=True)

    def update_physical_experts_metadata(self, num_physical_experts: int, num_local_physical_experts: int) -> None:
        raise NotImplementedError("TPUNemotronHForCausalLM has no EPLB")

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # A smoke run truncates layers_block_type via --hf-overrides; skip checkpoint layers past that depth.
        depth = len(self.config.hybrid_override_pattern)
        kept = (wt for wt in weights
                if not wt[0].startswith("backbone.layers.") or int(wt[0].split(".")[2]) < depth)
        return super().load_weights(kept)
