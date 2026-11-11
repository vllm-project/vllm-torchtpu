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
"""Local-KV + LSE-merge video-to-audio cross-attention for LTX-2.

In LTX-2's ``video_to_audio_attn``, the short audio query ``(B, S_a, H, D)`` is
replicated on every sequence-parallel rank while the long video K and V
``(B, S_v / U, H_kv, D)`` arrive sequence-sharded across the Ulysses group.
Upstream redistributes all video K and V across heads with two ``all_to_all``
collectives and then all-gathers the audio output heads.

Instead, each Ulysses rank computes attention of the replicated audio queries
against its local video K/V shard, keeping per-row ``(m_r, l_r)`` softmax
statistics alongside the unnormalized output ``o_r`` in ``float32``. A single
``all_gather_into_tensor`` over the small audio tensor (packed along rows into
one ``(rows, D)`` buffer) merges the shard outputs with
exact online-softmax rescaling.
"""

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

DEFAULT_MAX_SCORE_ELEMENTS = 16_777_216


def ltx2_v2a_local_kv_decline_reason(
    attn_layer: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: Any | None,
    sp_group: Any,
) -> str | None:
    """Why ``_LTX2ParallelAttention`` cannot merge V2A over local K/V.

    Returns ``None`` when the local-KV merge can run. The caller has already
    checked that ``attn_layer.ltx_sp_mode == "video_to_audio"``.
    """
    if attn_layer._has_custom_attention:
        return "custom attention backend"
    if attn_layer.causal:
        return "causal=True"
    if attn_layer.is_paged_kv_active():
        return "paged KV cache active"
    if sp_group.ring_world_size != 1 or sp_group.ulysses_world_size <= 1:
        return (
            f"ring_world_size={sp_group.ring_world_size}, "
            f"ulysses_world_size={sp_group.ulysses_world_size} "
            "(requires ring 1 and Ulysses > 1)"
        )
    if (
        query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or key.shape != value.shape
        or query.dtype != key.dtype
        or query.dtype != value.dtype
        or query.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        return (
            f"query {tuple(query.shape)} {query.dtype}, key {tuple(key.shape)} "
            f"{key.dtype}, value {tuple(value.shape)} {value.dtype} (requires "
            "4-D q/k/v of one bf16/f16/f32 dtype and k/v of one shape)"
        )
    bsz, s_q, q_heads, head_dim = query.shape
    k_bsz, s_kv, kv_heads, k_dim = key.shape
    world_size = sp_group.ulysses_world_size
    if (
        bsz <= 0
        or bsz != k_bsz
        or s_q <= 0
        or s_kv <= 0
        or head_dim <= 0
        or head_dim != k_dim
        or kv_heads <= 0
        or q_heads % kv_heads != 0
        or q_heads % world_size != 0
        or kv_heads % world_size != 0
    ):
        return (
            f"query {tuple(query.shape)}, key {tuple(key.shape)} on "
            f"ulysses_world_size={world_size} (requires matching batch and head "
            "dim, and q/kv heads divisible by kv heads and by the Ulysses size)"
        )
    eff_meta = attn_layer._with_kv_cache_dtype(attn_metadata)
    if eff_meta is not None:
        if eff_meta.attn_mask is not None:
            return "attn_mask set"
        kv_cache_dtype = eff_meta.extra.get("kv_cache_dtype")
        if kv_cache_dtype not in (None, "float"):
            return f"kv_cache_dtype={kv_cache_dtype}"
    return None


def _local_unnormalized_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    prob_dtype: torch.dtype,
    max_score_elements: int = DEFAULT_MAX_SCORE_ELEMENTS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local ``(o_unnorm, m, l)`` in ``float32`` over ``(B, H, S_q, S_kv)``."""
    bsz, heads, s_q, dim = q.shape
    s_kv = k.shape[2]
    rows_per_kv = max(1, bsz * heads * s_q)
    kv_chunk = max(1, max_score_elements // rows_per_kv)
    if s_kv <= kv_chunk:
        scores = torch.matmul(q, k.float().transpose(-2, -1)) * scale
        m_local = scores.amax(dim=-1, keepdim=True)
        probs = torch.exp(scores - m_local)
        l_local = probs.sum(dim=-1, keepdim=True)
        p_cast = probs.to(prob_dtype).float() if prob_dtype != torch.float32 else probs
        o_local = torch.matmul(p_cast, v.float())
        return o_local, m_local, l_local

    m_run = torch.full(
        (bsz, heads, s_q, 1), -float("inf"), dtype=torch.float32, device=q.device
    )
    l_run = torch.zeros((bsz, heads, s_q, 1), dtype=torch.float32, device=q.device)
    o_run = torch.zeros((bsz, heads, s_q, dim), dtype=torch.float32, device=q.device)
    for start in range(0, s_kv, kv_chunk):
        end = min(start + kv_chunk, s_kv)
        scores = torch.matmul(q, k[:, :, start:end].float().transpose(-2, -1)) * scale
        m_curr = scores.amax(dim=-1, keepdim=True)
        m_next = torch.maximum(m_run, m_curr)
        probs = torch.exp(scores - m_next)
        l_curr = probs.sum(dim=-1, keepdim=True)
        alpha = torch.exp(m_run - m_next)
        p_cast = probs.to(prob_dtype).float() if prob_dtype != torch.float32 else probs
        o_curr = torch.matmul(p_cast, v[:, :, start:end].float())
        o_run = o_run * alpha + o_curr
        l_run = l_run * alpha + l_curr
        m_run = m_next
    return o_run, m_run, l_run


def ltx2_v2a_local_kv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float,
    ulysses_pg: dist.ProcessGroup,
    use_sync: bool = False,
    max_score_elements: int = DEFAULT_MAX_SCORE_ELEMENTS,
) -> torch.Tensor:
    """Compute ``video_to_audio`` cross-attention from replicated Q and sharded K/V."""
    bsz, s_q, q_heads, dim = query.shape
    kv_heads = key.shape[2]
    q = query.transpose(1, 2).float()
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)
    if q_heads != kv_heads:
        rep = q_heads // kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

    o_local, m_local, l_local = _local_unnormalized_attention(
        q,
        k,
        v,
        scale=softmax_scale,
        prob_dtype=query.dtype,
        max_score_elements=max_score_elements,
    )

    # Pack (o_local, m_local, l_local) along rows into one float32 buffer
    # whose trailing dimension is the head dim (64 for LTX-2's
    # video-to-audio attention), so a single all-gather moves all three.
    n_rows = bsz * q_heads * s_q
    o_flat = o_local.reshape(n_rows, dim)
    ml_flat = torch.cat((m_local.reshape(-1), l_local.reshape(-1)), dim=0)
    pad = (-ml_flat.numel()) % dim
    if pad:
        ml_flat = F.pad(ml_flat, (0, pad))
    ml_rows = ml_flat.reshape(-1, dim)
    send_buf = torch.cat((o_flat, ml_rows), dim=0).contiguous()

    world_size = dist.get_world_size(ulysses_pg)
    recv_buf = torch.empty(
        (world_size * send_buf.shape[0], dim),
        dtype=torch.float32,
        device=query.device,
    )
    dist.all_gather_into_tensor(recv_buf, send_buf, group=ulysses_pg)
    if use_sync:
        from vllm_omni.platforms import current_omni_platform

        current_omni_platform.synchronize()

    recv = recv_buf.reshape(world_size, send_buf.shape[0], dim)
    o_all = recv[:, :n_rows, :].reshape(world_size, bsz, q_heads, s_q, dim)
    ml_all = recv[:, n_rows:, :].reshape(world_size, -1)[:, : 2 * n_rows]
    m_all = ml_all[:, :n_rows].reshape(world_size, bsz, q_heads, s_q, 1)
    l_all = ml_all[:, n_rows:].reshape(world_size, bsz, q_heads, s_q, 1)

    m_max = m_all.amax(dim=0, keepdim=True)
    m_max = torch.where(torch.isfinite(m_max), m_max, torch.zeros_like(m_max))
    rescale = torch.exp(m_all - m_max)
    denom = (l_all * rescale).sum(dim=0)
    out = (o_all * rescale).sum(dim=0) / torch.where(
        denom > 0, denom, torch.ones_like(denom)
    )
    return out.transpose(1, 2).to(query.dtype).contiguous()
