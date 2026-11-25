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
group itself (TPU_MINIMAX_M3_DCP_LOCAL_KV).

A DCP group wider than a KV head's TP replicas spans several KV heads, and
every rank stores all of their rows for its context blocks, so each layer
all-gathered the group's new k / v / index_q (ops/msa.py `gather_group_kv`):
a latency-bound collective on the TensorCore at decode sizes. With the flag,
qkv_proj's weight gets the k / v / index_q columns of the group's other KV
heads appended (gathered once after loading). Steps of fewer than
`MAX_TOKENS` tokens project all of them and reorder them into the group's
heads (`group_heads`); larger steps project only the original columns, a
prefix of the weight, and keep the gather: in mixed steps removing it moved
the wait for the slowest DCP rank to a later collective, where it cost more.
"""

import torch
from torch import nn
from vllm.model_executor.utils import replace_parameter

from vllm_torchtpu import envs
from vllm_torchtpu.layers.adapter.linear_common import WEIGHT_FLIPPED_ATTR
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.minimax_m3.kernels import msa_dcp_heads
from vllm_torchtpu.models.minimax_m3.sp_o_proj import all_gather_cpu

logger = init_logger(__name__)

# Set on the qkv_proj modules to prepare: (query heads, head dim, DCP size,
# TP ranks per KV head).
PREPARE_ATTR = "_minimax_m3_dcp_local_kv"
PREPARED_ATTR = "_minimax_m3_dcp_local_kv_prepared"
# Bar for the self-test, whose reference sums the same columns in a matmul of
# other width: roundings pass, a column block of the wrong head misses by O(1).
_SELFTEST_REL_TOL = 2**-6
_SELFTEST_ROWS = 16
_selftested = False
# Steps below this many tokens project the group's heads (the decode-sized
# steps of `ops.msa.gather_group_kv`).
MAX_TOKENS = msa_dcp_heads.HEAD_PARALLEL_MIN_TOKENS


def enabled() -> bool:
    return envs.TPU_MINIMAX_M3_DCP_LOCAL_KV


def mark(
    qkv_proj: nn.Module, num_heads: int, head_dim: int, dcp: int, replicas: int
) -> None:
    setattr(qkv_proj, PREPARE_ATTR, (num_heads, head_dim, dcp, replicas))


def group_columns(
    weight: torch.Tensor,
    scale: torch.Tensor,
    ranks_kvi: torch.Tensor,
    ranks_kvi_scale: torch.Tensor,
    rank: int,
    replicas: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's qkv_proj weight ``[K, N]`` and scale ``[N]`` followed by
    the ``k | v | index_q`` columns of the group's other KV heads, ascending,
    from every DCP rank's ``[dcp, K, 3 D]`` / ``[dcp, 3 D]`` in rank order
    (rank ``g * replicas`` holds group head ``g``). Any dtype (8-bit floats as
    their bytes)."""
    holders = range(0, ranks_kvi.shape[0], replicas)  # one rank per head
    others = [r for r in holders if r // replicas != rank // replicas]
    return (
        torch.cat([weight, *(ranks_kvi[r] for r in others)], dim=1),
        torch.cat([scale, *(ranks_kvi_scale[r] for r in others)]),
    )


def group_order(dcp_rank: torch.Tensor, heads: int, replicas: int) -> torch.Tensor:
    """Per group head ``g``, its index in ``[own head, other heads
    ascending]`` (`group_heads`' stack), from this rank's DCP rank (int32
    ``[1]``, a tensor, so every rank traces the same program)."""
    own = torch.div(dcp_rank.view(1), replicas, rounding_mode="floor")
    g = torch.arange(heads, dtype=own.dtype, device=own.device)
    return torch.where(g == own, 0, g + 1 - (g > own).to(own.dtype))


def group_heads(
    qkv: torch.Tensor, num_heads: int, head_dim: int, heads: int, order: torch.Tensor
) -> torch.Tensor:
    """``[q | k | v | index_q | index_k | (k | v | index_q) x (G - 1)]``, the
    prepared qkv_proj's output, as ``[q | k (G) | v (G) | index_q (G) |
    index_k]`` with the G group heads in DCP rank order (`group_order`)."""
    d, q_end = head_dim, num_heads * head_dim
    rows = qkv.shape[0]
    own = qkv[:, q_end : q_end + 3 * d].reshape(rows, 1, 3, d)
    others = qkv[:, q_end + 4 * d :].reshape(rows, heads - 1, 3, d)
    kvi = torch.cat([own, others], dim=1).index_select(1, order)
    return torch.cat(
        [
            qkv[:, :q_end],
            *(kvi[:, :, i].reshape(rows, heads * d) for i in range(3)),
            qkv[:, q_end + 3 * d : q_end + 4 * d],
        ],
        dim=-1,
    )


def _selftest(group, old_w, old_s, new_w, new_s, nq, d, replicas) -> None:
    """The group heads `group_heads` takes from the new weight's projection
    against every rank's own k / v / index_q projection all-gathered over the
    DCP group (the path they replace), on rows shared by all ranks.
    Collective over the group; every rank raises when any rank fails."""
    gen = torch.Generator().manual_seed(0)
    x = torch.randn(_SELFTEST_ROWS, old_w.shape[0], generator=gen)
    x = x.to(torch.bfloat16).to(old_w.device)
    kvi = slice(nq * d, (nq + 3) * d)
    mine = torch.matmul(x, old_w[:, kvi].to(x.dtype)) * old_s[kvi]
    want = group.all_gather(mine.unsqueeze(0).contiguous(), dim=0)[::replicas]
    heads = want.shape[0]
    rank = torch.tensor([group.rank_in_group], dtype=torch.int32, device=x.device)
    got = group_heads(
        torch.matmul(x, new_w.to(x.dtype)) * new_s,
        nq,
        d,
        heads,
        group_order(rank, heads, replicas),
    )
    got = got[:, nq * d : (nq + 3 * heads) * d].reshape(-1, 3, heads, d)
    want = want.reshape(heads, -1, 3, d).permute(1, 2, 0, 3)
    got, want = got.float().cpu(), want.float().cpu()
    max_abs = float((got - want).abs().max())
    scale = float(want.abs().max()) or 1.0
    ok = bool(torch.isfinite(got).all()) and max_abs <= _SELFTEST_REL_TOL * scale
    seen = [None] * group.world_size
    torch.distributed.all_gather_object(seen, ok, group=group.cpu_group)
    if not all(seen):
        raise RuntimeError(
            f"TPU_MINIMAX_M3_DCP_LOCAL_KV self-test failed (per DCP rank {seen}); "
            f"here max_abs={max_abs:.3g} scale={scale:.3g}"
        )
    logger.info(
        "MiniMax-M3 DCP local K/V self-test: ok, %s (max_abs=%.3g)",
        "bit-exact" if max_abs == 0 else "within roundings",
        max_abs,
    )


def prepare(qkv_proj: nn.Module) -> None:
    """After loading (collective over the DCP group): append the group's
    other KV heads' k / v / index_q columns to this rank's e4m3 [K, N]
    qkv_proj weight and its per-channel scale (`group_columns`); the
    original weight stays its first N columns."""
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
    w_bytes = w.cpu().view(torch.uint8)
    # gloo carries no 8-bit floats: the columns travel as int32 words.
    ranks_kvi = all_gather_cpu(w_bytes[:, kvi].contiguous().view(torch.int32), group)
    ranks_kvi_scale = all_gather_cpu(s[kvi].float(), group)
    new_w, new_s = group_columns(
        w_bytes,
        s.float().cpu(),
        ranks_kvi.view(torch.uint8),
        ranks_kvi_scale,
        group.rank_in_group,
        replicas,
    )
    new_w = new_w.view(torch.float8_e4m3fn).to(w.device)
    new_s = new_s.to(s.dtype).to(w.device)
    if not _selftested:
        _selftest(group, w, s, new_w, new_s, nq, d, replicas)
        _selftested = True
    replace_parameter(qkv_proj, "weight", nn.Parameter(new_w, requires_grad=False))
    replace_parameter(
        qkv_proj, "weight_scale", nn.Parameter(new_s, requires_grad=False)
    )
    setattr(qkv_proj, PREPARED_ATTR, True)
