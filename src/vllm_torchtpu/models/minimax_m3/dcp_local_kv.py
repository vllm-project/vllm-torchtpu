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
"""Every rank of a DCP group projects the K, V and index_q heads of the whole
group itself in decode-sized steps (TPU_MINIMAX_M3_DCP_LOCAL_KV).

A DCP group wider than a KV head's TP replicas spans several KV heads, and
every rank stores all of their rows for its context blocks, so each layer
all-gathered the group's new k / v / index_q (ops/msa.py `gather_group_kv`):
a latency-bound collective on the TensorCore at decode sizes. With the flag,
qkv_proj also holds the k / v / index_q columns of the group's other KV heads
(``group_weight`` / ``group_scale``, gathered once after loading). Steps of
fewer than `MAX_TOKENS` tokens, as bf16 rows or as an SP step's e4m3 rows,
project them too (`project`) and reorder all heads into the group's
(`group_heads`); larger steps keep the gather: in
mixed steps removing it moved the wait for the slowest DCP rank to a later
collective, where it cost more.
"""

import torch
from torch import nn

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads
from vllm_torchtpu.models.minimax_m3.sp_o_proj import all_gather_cpu
from vllm_torchtpu.utils import synchronize_tensors

logger = init_logger(__name__)

# Set on the qkv_proj modules to prepare: (query heads, head dim, DCP size,
# TP ranks per KV head).
PREPARE_ATTR = "_minimax_m3_dcp_local_kv"
PREPARED_ATTR = "_minimax_m3_dcp_local_kv_prepared"
# Steps below this many tokens project the group's heads (the decode-sized
# steps of `ops.msa.gather_group_kv`).
MAX_TOKENS = msa_dcp_heads.HEAD_PARALLEL_MIN_TOKENS
# Bar for the self-test, whose reference sums the same columns in a matmul of
# other width: roundings pass, a column block of the wrong head misses by O(1).
_SELFTEST_REL_TOL = 2**-6
_SELFTEST_ROWS = 16
_FP8_MAX = 448.0
_selftested = False


def enabled() -> bool:
    return envs.TPU_MINIMAX_M3_DCP_LOCAL_KV


def mark(
    qkv_proj: nn.Module, num_heads: int, head_dim: int, dcp: int, replicas: int
) -> None:
    setattr(qkv_proj, PREPARE_ATTR, (num_heads, head_dim, dcp, replicas))


def group_columns(
    ranks_kvi: torch.Tensor, ranks_kvi_scale: torch.Tensor, rank: int, replicas: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``k | v | index_q`` columns ``[K, 3 D (G - 1)]`` and scale of the
    group's KV heads other than this rank's, ascending, from every DCP rank's
    ``[dcp, K, 3 D]`` / ``[dcp, 3 D]`` in rank order (rank ``g * replicas``
    holds group head ``g``). Any dtype (8-bit floats as their bytes)."""
    holders = range(0, ranks_kvi.shape[0], replicas)  # one rank per head
    others = [r for r in holders if r // replicas != rank // replicas]
    return (
        torch.cat([ranks_kvi[r] for r in others], dim=1),
        torch.cat([ranks_kvi_scale[r] for r in others]),
    )


def group_order(
    qkv_proj: nn.Module, dcp_rank: torch.Tensor, heads: int, replicas: int
) -> torch.Tensor:
    """Per group head ``g``, its index in ``[own head, other heads
    ascending]`` (`group_heads`' stack), from this rank's DCP rank (int32
    ``[1]``, a tensor, so every rank traces the same program). Reads the
    group parameters too, so every trace of the model takes the same inputs
    (compilation/shape_variants hands buckets between traces only then)."""
    if not getattr(qkv_proj, PREPARED_ATTR, False):
        raise RuntimeError(
            "TPU_MINIMAX_M3_DCP_LOCAL_KV: qkv_proj was not prepared after loading "
            "(it needs MXFP8 weights kept e4m3, TPU_MINIMAX_M3_FP8_LINEARS)."
        )
    inputs = torch.zeros_like(qkv_proj.group_weight[:1, :1], dtype=torch.int32)
    inputs = inputs + torch.zeros_like(qkv_proj.group_scale[:1], dtype=torch.int32)
    own = torch.div(dcp_rank.view(1), replicas, rounding_mode="floor") + inputs.view(1)
    g = torch.arange(heads, dtype=own.dtype, device=own.device)
    return torch.where(g == own, 0, g + 1 - (g > own).to(own.dtype))


def project(
    qkv_proj: nn.Module, x: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """The group's other KV heads' ``k | v | index_q`` of bf16 rows ``x``, as
    MiniMaxM3Mxfp8LinearMethod projects decode rows; e4m3 rows with per-row
    f32 scales (an SP step's all-gather) contract as e4m3 x e4m3, as its
    `apply_e4m3_rows` projects them."""
    if isinstance(x, tuple):
        return _e4m3_rows_matmul_op()(*x, qkv_proj.group_weight, qkv_proj.group_scale)
    return torch.matmul(x, qkv_proj.group_weight.to(x.dtype)) * qkv_proj.group_scale


def group_heads(
    qkv: torch.Tensor,
    others: torch.Tensor,
    num_heads: int,
    head_dim: int,
    heads: int,
    order: torch.Tensor,
) -> torch.Tensor:
    """``[q | k (G) | v (G) | index_q (G) | index_k]``, the G group heads in
    DCP rank order (`group_order`), from qkv_proj's ``[q | k | v | index_q |
    index_k]`` of this rank and the other heads' ``(k | v | index_q) x (G - 1)``
    (`project`)."""
    d, q_end = head_dim, num_heads * head_dim
    rows = qkv.shape[0]
    own = qkv[:, q_end : q_end + 3 * d].reshape(rows, 1, 3, d)
    others = others.reshape(rows, heads - 1, 3, d)
    kvi = torch.cat([own, others], dim=1).index_select(1, order)
    return torch.cat(
        [
            qkv[:, :q_end],
            *(kvi[:, :, i].reshape(rows, heads * d) for i in range(3)),
            qkv[:, q_end + 3 * d :],
        ],
        dim=-1,
    )


def _e4m3_rows_matmul_op():
    # Imported on use: quant imports this module.
    from vllm_torchtpu.models.minimax_m3.quant import _e4m3_matmul

    return _e4m3_matmul


def _selftest(group, qkv_proj, nq, d, replicas) -> None:
    """The group heads `group_heads` assembles from the prepared projections
    against every rank's own k / v / index_q projection all-gathered over the
    DCP group (the path they replace), on rows shared by all ranks: bf16 rows,
    and the same rows as e4m3 with per-row scales (an SP step's all-gather).
    Collective over the group; every rank raises when any rank fails."""
    w, s = qkv_proj.weight, qkv_proj.weight_scale
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(_SELFTEST_ROWS, w.shape[0], generator=gen)
    x = x.to(torch.bfloat16).to(w.device)
    row_scale = x.float().abs().amax(dim=-1, keepdim=True) / _FP8_MAX
    rows8 = ((x.float() / row_scale).to(torch.float8_e4m3fn), row_scale)
    rank = torch.tensor([group.rank_in_group], dtype=torch.int32, device=x.device)
    max_abs, scales = {}, {}
    ok = True
    for name, rows, mine in (
        ("bf16", x, torch.matmul(x, w.to(x.dtype)) * s),
        ("e4m3", rows8, _e4m3_rows_matmul_op()(*rows8, w, s)),
    ):
        kvi = mine[:, nq * d : (nq + 3) * d]
        want = group.all_gather(kvi.unsqueeze(0).contiguous(), dim=0)[::replicas]
        heads = want.shape[0]
        got = group_heads(
            mine,
            project(qkv_proj, rows),
            nq,
            d,
            heads,
            group_order(qkv_proj, rank, heads, replicas),
        )
        got = got[:, nq * d : (nq + 3 * heads) * d].reshape(-1, 3, heads, d)
        want = want.reshape(heads, -1, 3, d).permute(1, 2, 0, 3)
        got, want = got.float().cpu(), want.float().cpu()
        max_abs[name] = float((got - want).abs().max())
        scales[name] = float(want.abs().max()) or 1.0
        ok = ok and bool(torch.isfinite(got).all())
        ok = ok and max_abs[name] <= _SELFTEST_REL_TOL * scales[name]
    seen = [None] * group.world_size
    torch.distributed.all_gather_object(seen, ok, group=group.cpu_group)
    if not all(seen):
        raise RuntimeError(
            f"TPU_MINIMAX_M3_DCP_LOCAL_KV self-test failed (per DCP rank {seen}); "
            f"here max_abs={max_abs} scale={scales}"
        )
    logger.info(
        "MiniMax-M3 DCP local K/V self-test: ok, %s (max_abs=%s)",
        "bit-exact" if not any(max_abs.values()) else "within roundings",
        max_abs,
    )


def prepare(qkv_proj: nn.Module) -> None:
    """After loading (collective over the DCP group): give this rank's e4m3
    [K, N] qkv_proj (per-channel scale) the group's other KV heads'
    k / v / index_q columns as ``group_weight`` / ``group_scale``
    (`group_columns`)."""
    from vllm_torchtpu.distributed.dcp import get_dcp_group

    global _selftested
    nq, d, dcp, replicas = getattr(qkv_proj, PREPARE_ATTR)
    w, s = qkv_proj.weight.data, qkv_proj.weight_scale.data
    if not (
        w.dtype == torch.float8_e4m3fn
        and getattr(qkv_proj, WEIGHT_FLIPPED_ATTR, False)
        and w.shape[1] == (nq + 4) * d
        and s.shape == (w.shape[1],)
    ):
        raise RuntimeError(
            "TPU_MINIMAX_M3_DCP_LOCAL_KV needs qkv_proj as an e4m3 [K, (q heads + "
            f"4) * {d}] weight with a per-channel scale; got {tuple(w.shape)} "
            f"{w.dtype}, scale {tuple(s.shape)}."
        )
    group = get_dcp_group()
    if group is None or group.world_size != dcp:
        raise RuntimeError(f"TPU_MINIMAX_M3_DCP_LOCAL_KV: no DCP group of {dcp}.")
    kvi = slice(nq * d, (nq + 3) * d)
    # gloo carries no 8-bit floats: the columns travel as int32 words.
    words = w.cpu().view(torch.uint8)[:, kvi].contiguous().view(torch.int32)
    group_w, group_s = group_columns(
        all_gather_cpu(words, group).view(torch.uint8),
        all_gather_cpu(s[kvi].float(), group),
        group.rank_in_group,
        replicas,
    )
    for name, value in (
        ("group_weight", group_w.view(torch.float8_e4m3fn)),
        ("group_scale", group_s.to(s.dtype)),
    ):
        qkv_proj.register_parameter(
            name, nn.Parameter(value.to(w.device), requires_grad=False)
        )
    if w.device.type == "tpu":
        synchronize_tensors([qkv_proj.group_weight, qkv_proj.group_scale])
    setattr(qkv_proj, PREPARED_ATTR, True)
    if not _selftested:
        _selftest(group, qkv_proj, nq, d, replicas)
        _selftested = True
