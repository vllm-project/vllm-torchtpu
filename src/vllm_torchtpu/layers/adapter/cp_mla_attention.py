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
"""DCP (Decode Context Parallelism) attention for MLA: the head gather and
the log-sum-exp merge that bracket the sharded attention call.

The MLA counterpart of `cp_attention.py`.

Every collective runs on tensors already in torch's hands, outside
the exported op.

Each DCP rank stores a shard of the KV cache, so it can only produce a
*partial* attention result.

  1. `all_gather_heads` widens this rank's query head slice to the whole
     group's heads, so every rank's heads are attended against every rank's
     KV shard.
  2. the kernel runs against this rank's shard alone, returning a partial
     output and its log-sum-exp.
  3. `merge_lse_partials_scatter_heads` folds the partials into the exact
     whole-context result and narrows the head axis back to this rank's own.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu import envs, transforms
from vllm_torchtpu.distributed.dcp import get_dcp_group
from vllm_torchtpu.distributed.dcp_comm import cp_lse_ag_out_rs
from vllm_torchtpu.kernels.lse_rescale import lse_rescale_quantize
from vllm_torchtpu.layers.core.quantization import quantize_tensor

if envs.TPU_DCP_MERGE_VSPACE_FUSED:
    # Registers the fused V-space operand ops at import, never in a trace.
    from vllm_torchtpu.layers.adapter import dcp_vspace  # noqa: F401


def all_gather_heads(q: torch.Tensor) -> torch.Tensor:
    """Widen a rank's query head slice to the whole DCP group's heads.

    DCP doesn't add a new parallel dimension—it borrows from TP. It
    reallocates dcp ways from heads to sequence positions, reducing
    head sharding from tp to tp // dcp.

    Args:
      q: `[num_tokens, tp_local_num_heads, dim]`, this rank's head slice.

    Returns:
      `[num_tokens, tp_local_num_heads * dcp_size, dim]`, identical
      on every rank. Returned unchanged without a DCP group.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return q
    return group.all_gather(q.contiguous(), dim=1)


def all_gather_q_nope_pe_heads(
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused all-gather of `(ql_nope, q_pe)` across DCP ranks in a single call."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return ql_nope, q_pe
    q_dcp = group.all_gather(torch.cat([ql_nope, q_pe], dim=-1), dim=1)
    return (
        q_dcp[..., :kv_lora_rank].contiguous(),
        q_dcp[..., kv_lora_rank:].contiguous(),
    )


def project_uv_heads(
    partial: torch.Tensor,
    w_uv: torch.Tensor,
    w_uv_scale: torch.Tensor | None,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """`[T, N, kv_lora_rank]` partial -> `[T, N, v_head_dim]` in `out_dtype`,
    head n by W_UV[n].

    W_UV is linear and the merge weights are per (token, head), so projecting
    each partial before the merge gives the same result as projecting after.
    """
    if out_dtype != torch.float32 and w_uv_scale is None:
        # One MXU pass that writes the narrow dtype directly.
        out = torch.bmm(partial.transpose(0, 1), w_uv.to(partial.dtype))
        return out.transpose(0, 1).to(out_dtype)
    # f32 operands make bmm return f32 (out_dtype is ignored on TPU). bf16/fp8
    # convert exactly, so XLA still runs one bf16 MXU pass.
    out = torch.bmm(partial.transpose(0, 1).to(torch.float32), w_uv.to(torch.float32))
    if w_uv_scale is not None:
        out = out * w_uv_scale.to(torch.float32)
    return out.transpose(0, 1).to(out_dtype)


def merge_lse_partials_scatter_heads(
    output: torch.Tensor,
    lse: torch.Tensor,
    is_backbone: bool = True,
    wire_dtype: torch.dtype | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine this rank's partial attention output with its peers'.

    With per-rank normalized outputs `o_r` and `lse_r = m_r + log(l_r)`, the
    exact whole-context result is `o = sum_r softmax_r(lse) * o_r`. Correctness
    does not depend on the order the ranks are combined in.

    Two collectives:

      1. all-gather the log-sum-exps. They are `[num_tokens, num_heads]` f32 --
         Leaves every rank holding every `lse_r`.
      2. reduce-scatter the weighted outputs

    With TPU_DCP_LSE_MERGE_A2A set, the backbone's steps of at most
    TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS tokens instead take
    `merge_lse_partials_all_to_all`, which gives the same result. With
    TPU_DCP_MERGE_CHUNKED=n, the backbone's steps of at least
    TPU_DCP_MERGE_CHUNKED_MIN_TOKENS tokens run the reduce-scatter merge as n
    token chunks (`_merge_chunked`), which gives the same per-row result.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      is_backbone: True if the caller is the backbone, the only model that is
        traced separately per token bucket. Only then may the merge compare
        the token count to TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS (or
        TPU_DCP_MERGE_CHUNKED_MIN_TOKENS), since that comparison is fixed at
        trace time. Pass False from the MTP drafter,
        which is traced once for every bucket: it always takes the
        reduce-scatter, exactly as with TPU_DCP_LSE_MERGE_A2A unset.
      query_start_loc: the step's `[num_reqs + 1]` query starts, padded with
        the real token count (so its last entry is that count). With
        TPU_DCP_LSE_MERGE_A2A_REAL_ROWS the packed all-to-all sends only the
        real tokens' rows; None keeps every padded row.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    # This `if` runs while tracing, so its answer is baked into the graph.
    # Only the backbone is traced once per bucket group, so only it can pick
    # by size: the all-to-all wins at decode sizes, the reduce-scatter above
    # because it sums in the network. The MTP drafter has one graph for every
    # bucket; comparing its token count would pin that graph to the first
    # size traced. `is_backbone` is checked before the token count, so the
    # drafter never compares it.
    # A W_UV-projected f32 partial still crosses in the attention dtype.
    narrow = envs.TPU_DCP_RS_OUT_PRECISION and is_backbone
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and is_backbone
        and output.shape[0] <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return merge_lse_partials_all_to_all(
            output,
            lse,
            group,
            send_dtype=wire_dtype if narrow else None,
            query_start_loc=query_start_loc,
        )

    # The drafter cannot compare the token count in torch, but JAX exports
    # each of its buckets at a concrete shape: one op picks the exchange
    # there (TPU_MTP_DCP_MERGE_SIZED).
    if (
        envs.TPU_MTP_DCP_MERGE_SIZED
        and not is_backbone
        and output.element_size() == 2
        and int(group.world_size) in _SIZED_MERGE_OPS
    ):
        return _SIZED_MERGE_OPS[int(group.world_size)](output, lse)

    rs_dtype = (wire_dtype or output.dtype) if narrow else None
    # Same trace-time rule as above: only the backbone may split by size.
    chunks = _merge_chunks(output.shape[0]) if is_backbone else 1
    if chunks > 1:
        return _retain_rank(
            _merge_chunked(output, lse, group, chunks, rs_dtype=rs_dtype)
        )
    return _retain_rank(cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype))


def vspace_fused_applies(num_tokens: int, is_backbone: bool) -> bool:
    """TPU_DCP_MERGE_VSPACE_FUSED: whether this step's merge is
    the chunked reduce-scatter merge with the heads-major weighted operand,
    so the operand can be built in V space by `dcp_vspace` (#1935 trial 6's
    TPU_DCP_MERGE_VSPACE). Fixed at trace time; every comparison here is one
    `merge_lse_partials_scatter_heads` already makes for the same bucket. The
    MTP drafter never takes it."""
    if not envs.TPU_DCP_MERGE_VSPACE_FUSED or not is_backbone:
        return False
    if envs.TPU_DCP_MERGE_RS_BF16 == 2:
        return False
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return False
    if (
        envs.TPU_DCP_LSE_MERGE_A2A
        and num_tokens <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
    ):
        return False
    return _merge_chunks(num_tokens) > 1


def merge_lse_partials_scatter_heads_vspace(
    output: torch.Tensor,
    lse: torch.Tensor,
    w_uv: torch.Tensor,
    w_uv_scale: torch.Tensor | None,
    wire_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`merge_lse_partials_scatter_heads` on a bucket `vspace_fused_applies`
    takes, merging in V space: every chunk projects this rank's latent
    partial `[T, H, C]` through W_UV `[H, C, V]` for all `H` heads, weights it
    and reduce-scatters the V-wide result. Returns `[T, H / world, V]`, this
    rank's heads already through W_UV."""
    group = get_dcp_group()
    narrow = envs.TPU_DCP_RS_OUT_PRECISION
    rs_dtype = (wire_dtype or output.dtype) if narrow else None
    chunks = _merge_chunks(output.shape[0])
    merged = _merge_chunked(
        output, lse, group, chunks, rs_dtype=rs_dtype, vspace=(w_uv, w_uv_scale)
    )
    return _retain_rank(merged)


def _merge_chunks(num_tokens: int) -> int:
    """How many token chunks `TPU_DCP_MERGE_CHUNKED` splits this bucket's
    merge into: 1 (unsplit) unless the bucket reaches
    `TPU_DCP_MERGE_CHUNKED_MIN_TOKENS` and splits into equal chunks whose
    length is a multiple of 8 (the f32 sublane tile).

    TPU_DCP_MERGE_CHUNKED_SMALL=m (>= 2): the buckets that gather their lse
    once (TPU_DCP_MERGE_CHUNKED_LSE_AG, at most
    TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS tokens: c1k, c2k) split into m
    chunks instead. The comparison is the one `_lse_gathered_once` already
    makes for the same bucket, so no new shape guard; JAX (the kernel's merge
    pieces) calls this with the static row count and gets the same answer."""
    chunks = envs.TPU_DCP_MERGE_CHUNKED
    if chunks <= 1 or not (
        num_tokens > envs.bucket_gate_edge(envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS)
    ):
        return 1
    small = envs.TPU_DCP_MERGE_CHUNKED_SMALL
    if small >= 2 and _lse_gathered_once(num_tokens):
        chunks = small
    if num_tokens % (8 * chunks):
        return 1
    return chunks


def _merge_chunked(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    chunks: int,
    rs_dtype: torch.dtype | None = None,
    vspace: tuple[torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """The reduce-scatter merge (`cp_lse_ag_out_rs`) over `chunks`
    independent token chunks. `vspace` (TPU_DCP_MERGE_VSPACE_FUSED):
    `(W_UV, scale)` over every head; the result is `[T, H / world, V]`.

    Every token's merge depends on its own row alone (the lse all-gather,
    logsumexp, weight and reduce-scatter sum all run per token), so splitting
    the token axis changes no per-row arithmetic: each chunk gathers its own
    rows' lse, weights them and reduce-scatters them exactly as the unsplit
    merge does. What changes is the schedule. The f32 reduce-scatter is a
    SparseCore-offloaded async collective while the lse gather and the weight
    multiply are TensorCore work, so chunk `k`'s reduce-scatter can run under
    chunk `k + 1`'s gather and multiply instead of after all of them.
    """
    if _lse_gathered_once(output.shape[0]):
        return _merge_chunked_lse_gathered(
            output, lse, group, chunks, rs_dtype, vspace=vspace
        )
    use_barrier = not bool(envs.TPU_DCP_MLA_MERGE_PIECES)
    if envs.TPU_DCP_MERGE_RS_BF16 or envs.TPU_DCP_MERGE_WEIGHTED_HM or vspace:
        # `cp_lse_ag_out_rs` op for op, except the scatter-sum's wire dtype
        # (RS_BF16) and/or the heads-major operand (WEIGHTED_HM).
        lse_f32 = lse.to(torch.float32)
        with transforms.microbatch(
            min_len=envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS,
            groups=chunks,
            barrier=use_barrier,
        ):
            lses = _gather_lse(lse_f32, group)
            return _weighted_reduce_scatter(
                output,
                lse_f32,
                lses,
                group,
                rs_dtype,
                vspace=vspace,
            )
    with transforms.microbatch(
        min_len=envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS,
        groups=chunks,
        barrier=use_barrier,
    ):
        return cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype)


def _lse_gathered_once(num_tokens: int) -> bool:
    """Whether `TPU_DCP_MERGE_CHUNKED_LSE_AG` applies to this chunked bucket
    (fixed at trace time; only the backbone chunks)."""
    cap = envs.TPU_DCP_MERGE_CHUNKED_LSE_AG_MAX_TOKENS
    return bool(envs.TPU_DCP_MERGE_CHUNKED_LSE_AG) and (cap <= 0 or num_tokens <= cap)


def _gather_lse(lse_f32: torch.Tensor, group: Any) -> torch.Tensor:
    """`[T, H]` f32 -> `[world, T, H]`: every rank's lse rows (the gather
    `cp_lse_ag_out_rs` runs)."""
    return group.all_gather(lse_f32.unsqueeze(0).contiguous(), dim=0)


def _weighted_reduce_scatter(
    output: torch.Tensor,
    lse_f32: torch.Tensor,
    lses: torch.Tensor,
    group: Any,
    rs_dtype: torch.dtype | None = None,
    vspace: tuple[torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """`cp_lse_ag_out_rs` after its lse gather, op for op: `lses` is
    `[world, T, H]` for exactly `output`'s `T` rows. `vspace`
    (TPU_DCP_MERGE_VSPACE_FUSED): the operand is built in V space."""
    merged_lse = torch.logsumexp(lses, dim=0)
    if vspace is not None:
        weight = torch.exp(lse_f32 - merged_lse)
        return _weighted_reduce_scatter_vspace(output, weight, vspace, group, rs_dtype)
    # The heads-major operand feeds the f32 or the bf16 (RS_BF16=1)
    # reduce-scatter; RS_BF16=2's all-to-all keeps the tokens-major operand.
    if (
        envs.TPU_DCP_MERGE_WEIGHTED_HM
        and envs.TPU_DCP_MERGE_RS_BF16 != 2
        and output.shape[0] * max(int(envs.TPU_DCP_MERGE_CHUNKED), 1)
        > envs.bucket_gate_edge(envs.TPU_DCP_MERGE_WEIGHTED_HM_MIN_TOKENS)
    ):
        return _weighted_reduce_scatter_heads_major(
            output, lse_f32, merged_lse, group, rs_dtype
        )
    # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
    weight = torch.exp(lse_f32 - merged_lse).unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    return _scatter_sum(weighted, group, rs_dtype).to(output.dtype)


def _scatter_sum(
    weighted: torch.Tensor, group: Any, rs_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Sum `[T, H, V]` f32 `weighted` over the ranks and keep this rank's
    head block `[T, H / world, V]`: the merge's reduce-scatter.

    `TPU_DCP_MERGE_RS_BF16` (not bitwise) sends the weighted partial in bf16:
    1 reduce-scatters it in bf16 (the collective accumulates in bf16); 2
    exchanges the same bf16 bytes by all-to-all and sums the received ranks
    locally in f32. Either halves the bytes of the f32 reduce-scatter.
    Unset, the reduce-scatter runs in `rs_dtype` (TPU_DCP_RS_OUT_PRECISION),
    as `cp_lse_ag_out_rs` does; None keeps f32.
    """
    mode = envs.TPU_DCP_MERGE_RS_BF16
    if mode == 1:
        return group.reduce_scatter(weighted.to(torch.bfloat16).contiguous(), dim=1)
    if mode == 2:
        world = int(group.world_size)
        tokens, heads, vdim = weighted.shape
        send = (
            weighted.to(torch.bfloat16)
            .reshape(tokens, world, heads // world, vdim)
            .transpose(0, 1)
        )
        return _all_to_all(send, group).to(torch.float32).sum(dim=0)
    if rs_dtype is not None:
        weighted = weighted.to(rs_dtype)
    return group.reduce_scatter(weighted.contiguous(), dim=1)


def _weighted_reduce_scatter_vspace(
    output: torch.Tensor,
    weight: torch.Tensor,
    vspace: tuple[torch.Tensor, torch.Tensor | None],
    group: Any,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`[T, H, C]` partial, f32 `[T, H]` weights, `(W_UV [H, C, V], scale)`
    -> this rank's `[T, H / world, V]`: the sum over ranks of
    `where(w > 0, (o_h @ W_UV_h) * w, 0)`, reduce-scattered over heads, with
    the operand built heads-major in one computation (TPU_DCP_MERGE_VSPACE_FUSED).
    Wire dtype as `_weighted_reduce_scatter_heads_major`'s."""
    w_uv, w_uv_scale = vspace
    wire = torch.bfloat16 if envs.TPU_DCP_MERGE_RS_BF16 == 1 else rs_dtype
    wire = wire or torch.float32
    if output.device.type == "tpu":
        from vllm_torchtpu.layers.adapter.dcp_vspace import vspace_weighted_heads_major

        x = vspace_weighted_heads_major(output, weight, w_uv, w_uv_scale, wire)
    else:
        # The same arithmetic in torch (host tests): o's dtype operands, f32
        # accumulation, the weight on the f32 product, one rounding.
        v = torch.bmm(
            output.transpose(0, 1).to(torch.float32),
            w_uv.to(output.dtype).to(torch.float32),
        )
        if w_uv_scale is not None:
            v = v * w_uv_scale.to(torch.float32).reshape(v.shape[0], 1, -1)
        w = weight.to(torch.float32).transpose(0, 1).unsqueeze(-1)
        x = torch.where(w > 0, v * w, torch.zeros_like(v)).to(wire)
    return group.reduce_scatter(x.contiguous(), dim=0).transpose(0, 1).to(output.dtype)


def _weighted_reduce_scatter_heads_major(
    output: torch.Tensor,
    lse_f32: torch.Tensor,
    merged_lse: torch.Tensor,
    group: Any,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """`_weighted_reduce_scatter`'s weight multiply and reduce-scatter with the
    operand built heads-major (`TPU_DCP_MERGE_WEIGHTED_HM`) by the same
    Pallas kernel `cp_lse_ag_out_rs` uses (`kernels/lse_rescale.py`), so no
    relayout copy -- TensorCore or SparseCore data-format -- sits between the
    weights and the reduce-scatter.

    The wire dtype is `cp_lse_ag_out_rs`'s: `rs_dtype`
    (TPU_DCP_RS_OUT_PRECISION), else f32; `TPU_DCP_MERGE_RS_BF16=1` makes it
    bf16 (each f32 product rounded once, as `_scatter_sum`'s
    `.to(bfloat16)`), the RS_BF16 merge, heads-major.
    """
    wire = torch.bfloat16 if envs.TPU_DCP_MERGE_RS_BF16 == 1 else rs_dtype
    if output.device.type == "tpu":
        from vllm_torchtpu.distributed.dcp_comm import _lse_rescale_head_major

        name = str(wire or torch.float32).removeprefix("torch.")
        x = _lse_rescale_head_major(output, lse_f32, merged_lse, name)
    else:
        w = torch.exp(lse_f32 - merged_lse).transpose(0, 1).unsqueeze(-1)
        x = output.transpose(0, 1).to(torch.float32) * w
        x = torch.where(w > 0, x, torch.zeros_like(x))
        if wire is not None:
            x = x.to(wire)
    return group.reduce_scatter(x.contiguous(), dim=0).transpose(0, 1).to(output.dtype)


def _merge_chunked_lse_gathered(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    chunks: int,
    rs_dtype: torch.dtype | None = None,
    vspace: tuple[torch.Tensor, torch.Tensor | None] | None = None,
) -> torch.Tensor:
    """`_merge_chunked` with one lse all-gather for every chunk
    (`TPU_DCP_MERGE_CHUNKED_LSE_AG`).

    The per-chunk gathers are synchronous TensorCore collectives with no data
    edge to the earlier chunks, so the scheduler issues most of them before
    chunk 0's weight multiply and each one's fixed cost delays the first
    SparseCore reduce-scatter, the head of the SC-bound chain. One gather pays
    that cost once. It only moves rows, so each chunk's rows of it are bit for
    bit what the chunk's own gather returned, and the per-row merge arithmetic
    is unchanged.
    """
    lse_f32 = lse.to(torch.float32)
    use_barrier = not bool(envs.TPU_DCP_MLA_MERGE_PIECES)
    if envs.TPU_DCP_MLA_MERGE_PIECES >= 2:
        with transforms.microbatch(
            min_len=envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS,
            groups=chunks,
            barrier=use_barrier,
        ):
            lses = _gather_lse(lse_f32, group)
            return _weighted_reduce_scatter(
                output,
                lse_f32,
                lses,
                group,
                rs_dtype,
                vspace=vspace,
            )
    gathered_t = _gather_lse(lse_f32, group).transpose(0, 1)
    with transforms.microbatch(
        min_len=envs.TPU_DCP_MERGE_CHUNKED_MIN_TOKENS,
        groups=chunks,
        barrier=use_barrier,
    ):
        return _weighted_reduce_scatter(
            output,
            lse_f32,
            gathered_t.transpose(0, 1),
            group,
            rs_dtype,
            vspace=vspace,
        )


def _fake_lse_rescale_quantize(
    output: torch.Tensor,
    lse: torch.Tensor,
    merged_lse: torch.Tensor,
    start: int,
    out_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    heads, _, dim = output.shape
    rows = lse.shape[0]
    return (
        output.new_empty((rows, heads * dim), dtype=getattr(torch, out_dtype)),
        output.new_empty((rows, 1), dtype=torch.float32),
    )


# Registered at import: Dynamo cannot trace a lock-guarded registration.
_lse_rescale_quantize = pallas.jax_op(
    "pallas::lse_rescale_quantize", lse_rescale_quantize
)
_lse_rescale_quantize.register_fake(_fake_lse_rescale_quantize)


def weight_lse_partials(
    output: torch.Tensor, lse: torch.Tensor, quant_dtype: torch.dtype, chunk: int
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """`merge_lse_partials_scatter_heads` without the reduce-scatter: this
    rank's `[T, H, V]` partial for every head, scaled by its share of the
    exact merge and quantized per token for o_proj. Per chunk of `chunk`
    tokens (the last one shorter), `q` `[rows, H * V]` in `quant_dtype` and
    its `scale` `[rows, 1]` f32. The caller sums the ranks
    (TPU_DCP_MLA_OPROJ_ALL_REDUCE: in o_proj's all-reduce)."""
    group = get_dcp_group()
    assert group is not None
    lse = lse.to(torch.float32)
    lses = group.all_gather(lse.unsqueeze(0).contiguous(), dim=0)
    pieces = zip(lse.split(chunk), torch.logsumexp(lses, dim=0).split(chunk))
    if output.device.type == "tpu":
        # Head-major, as `project_uv_heads`' batched matmul writes it.
        head_major = output.transpose(0, 1)
        dtype = str(quant_dtype).removeprefix("torch.")
        return [
            _lse_rescale_quantize(head_major, own, merged, c * chunk, dtype)
            for c, (own, merged) in enumerate(pieces)
        ]
    out = []
    for x, (own, merged) in zip(output.split(chunk), pieces):
        # where, not *: a zero-weight partial may be NaN (NaN * 0 = NaN).
        weight = torch.exp(own - merged).unsqueeze(-1)
        weighted = torch.where(weight > 0, x.to(torch.float32) * weight, 0.0)
        out.append(quantize_tensor(weighted.flatten(1), quant_dtype))
    return out


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Block `d` of `x`'s leading axis goes to rank `d`; block `r` of the
    result came from rank `r`."""
    from torch.distributed import _functional_collectives as funcol

    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    return funcol.wait_tensor(out)


def merge_lse_partials_all_to_all(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    send_dtype: torch.dtype | None = None,
    query_start_loc: torch.Tensor | None = None,
) -> torch.Tensor:
    """Same result as the reduce-scatter merge, via all-to-all.

    Each rank sends head block `d` of its partial and lse to rank `d`, then
    merges its own heads locally in f32. Faster for small token counts.
    `send_dtype` is the dtype the partial crosses in; None keeps its own.
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    if send_dtype is not None:
        o_send = o_send.to(send_dtype)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    if envs.TPU_DCP_LSE_MERGE_A2A_PACK and output.element_size() == 2:
        real_tokens = None
        if envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS and query_start_loc is not None:
            # Padded with the real count, so the last entry is that count.
            real_tokens = query_start_loc[-1:].to(torch.int32)
        o_recv, l_recv = _all_to_all_packed(o_send, l_send, group, real_tokens)
    else:
        o_recv = _all_to_all(o_send, group)
        l_recv = _all_to_all(l_send, group)
    return _merge_received(o_recv, l_recv).to(output.dtype)


def _all_to_all_packed(
    o_send: torch.Tensor,
    l_send: torch.Tensor,
    group: Any,
    real_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """`(_all_to_all(o_send), _all_to_all(l_send))` in ONE all-to-all
    (`TPU_DCP_LSE_MERGE_A2A_PACK`).

    `o_send` is `[world, T, local, V]` in a 2-byte dtype and `l_send` the f32
    `[world, T, local]`. The exchange carries int16 words: the partial
    reinterpreted as is, then each destination's `T * local` lse bit patterns
    split into a block of high halves and a block of low halves (integer
    shifts and masks, so no lane interleave), zero padded to whole `V`-wide
    rows after its `T * local` partial rows. The words stay integers until
    they are unpacked: a float concat may be lowered to a select/max that
    canonicalizes the NaN bit patterns half an f32 can take. The exchange moves
    bits, so both halves come back bit for bit what the two separate exchanges
    return; only the f32 lse's own collective (latency-bound at a few KB) is
    gone, for `ceil(2 * T * local / V)` extra rows on the partial's.

    With `real_tokens` (`TPU_DCP_LSE_MERGE_A2A_REAL_ROWS`, an int32 `[1]` count
    of real tokens, the rest of the `T` rows padding), the lse rows lead each
    destination block and a ragged all-to-all sends only them plus the real
    tokens' `real_tokens * local` partial rows: at decode `b1` a 16-row bucket
    holds 4 real rows, so three quarters of the payload stays home. Received
    rows past that are zero. Real rows come back bit for bit; a padding row's
    partial is zero instead of whatever the kernel left there (its lse still
    arrives whole).
    """
    world, tokens, local, vdim = o_send.shape
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_words = o_send.reshape(world, rows, vdim).view(torch.int16)
    bits = l_send.contiguous().view(torch.int32).reshape(world, rows)
    hi = (bits >> 16).to(torch.int16)
    # The low half as a signed 16-bit value, so the narrowing is exact.
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).to(torch.int16)
    l_words = torch.nn.functional.pad(
        torch.cat([hi, lo], dim=1), (0, extra * vdim - words)
    )
    if real_tokens is not None:
        send = torch.cat([l_words.reshape(world, extra, vdim), o_words], dim=1)
        recv = _ragged_all_to_all(send, extra + real_tokens * local, group)
        o_recv = recv[:, extra:].reshape(world, tokens, local, vdim)
        o_recv = o_recv.view(o_send.dtype)
        l_recv_words = recv[:, :extra].reshape(world, extra * vdim)
    else:
        send = torch.cat([o_words, l_words.reshape(world, extra, vdim)], dim=1)
        recv = _all_to_all(send, group)
        o_recv = recv[:, :rows].reshape(world, tokens, local, vdim).view(o_send.dtype)
        l_recv_words = recv[:, rows:].reshape(world, extra * vdim)
    hi = l_recv_words[:, :rows].to(torch.int32)
    lo = l_recv_words[:, rows:words].to(torch.int32)
    l_recv = (
        ((hi << 16) | (lo & 0xFFFF)).view(torch.float32).reshape(world, tokens, local)
    )
    return o_recv, l_recv


def send_layout_applies(
    num_tokens: int,
    num_heads: int,
    vdim: int,
    dtype: torch.dtype,
    is_backbone: bool,
) -> bool:
    """Whether this step's DCP merge takes the kernel-written send buffer
    (`TPU_DCP_MLA_SEND_LAYOUT`): exactly where `merge_lse_partials_scatter_heads`
    would take the packed real-rows all-to-all, on a decode bucket the kernel
    owns the lse of, at a geometry it can lay out. Fixed at trace time (the
    backbone only, as for the all-to-all)."""
    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel

    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return False
    if not (
        envs.TPU_DCP_MLA_SEND_LAYOUT
        and envs.TPU_DCP_LSE_MERGE_A2A
        and envs.TPU_DCP_LSE_MERGE_A2A_PACK
        and envs.TPU_DCP_LSE_MERGE_A2A_REAL_ROWS
        and envs.TPU_MLA_SPARSE_SPLIT_Q
        and envs.TPU_DCP_MLA_MASK_IN_KERNEL
        and is_backbone
        and dtype == torch.bfloat16
        and num_tokens <= envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS
        and num_tokens <= sparse_kernel.FUSED_PREP_MAX_TOKENS
        and num_tokens > envs.bucket_gate_edge(envs.TPU_DCP_MLA_SEND_LAYOUT_MIN_TOKENS)
    ):
        return False
    # The kernel runs one chunk of `num_tokens` (the capped chunk is larger
    # than any decode bucket); its grid step is this many tokens.
    _, batch = sparse_kernel._chunk_kernel_batch_size(
        num_tokens,
        sparse_kernel.DEFAULT_ATTENTION_KERNEL_BATCH_SIZE,
        envs.TPU_DCP_MLA_SMALL_CHUNK_TOKENS,
        envs.TPU_DCP_MLA_SMALL_CHUNK_BATCH,
    )
    return sparse_kernel.send_layout_supported(
        num_tokens, num_heads, int(group.world_size), vdim, batch
    )


def merge_send_layout(
    send: torch.Tensor, query_start_loc: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """`merge_lse_partials_all_to_all` with pack + real rows, from the send
    buffer the kernel wrote (`TPU_DCP_MLA_SEND_LAYOUT`).

    `send` is this rank's int16 `[world, SEND_PAD_ROWS + T * local, V]`
    (`sparse_mla_kernel.SEND_PAD_ROWS`): per destination, the lse word row at
    `SEND_PAD_ROWS - 1` and the partial rows from `SEND_PAD_ROWS` on. One
    ragged all-to-all moves `[SEND_PAD_ROWS - 1, SEND_PAD_ROWS + real * local)`
    of each block -- the same `extra + real * local` rows, holding the same
    bits, as `_all_to_all_packed`'s -- into the same place of the receiving
    block, so the received partials and lse words are those of the packed
    exchange and the merge below is its merge.

    Returns `[T, local, V]` bf16, this rank's head slice of the result.
    """
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    group = get_dcp_group()
    world, block, vdim = send.shape
    local = (block - SEND_PAD_ROWS) // num_tokens
    rows = num_tokens * local
    # Padded with the real count, so the last entry is that count.
    real = query_start_loc[-1:].to(torch.int32)
    if send_small_a2a_applies(num_tokens):
        # One Pallas round of remote DMAs + the unpack, one JAX op; partial
        # rows past the real ones zeroed as the ragged exchange leaves them.
        # The merge below is the incumbent's, on the same bits.
        from vllm_torchtpu.distributed.dcp import dcp_lane_operand

        prebuild_send_small(world)
        o_recv, l_recv = _SEND_SMALL_OPS[world](
            send.reshape(world * block, vdim).contiguous(),
            1 + real * local,
            _DCP_RANK,
            dcp_lane_operand(),
        )
        o_recv = o_recv.reshape(world, num_tokens, local, vdim)
        l_recv = l_recv.reshape(world, num_tokens, local)
        return _merge_received(o_recv, l_recv).to(torch.bfloat16)
    group_rows = int(envs.TPU_DCP_MLA_SEND_GROUP_ROWS)
    if group_rows > 0 and send_group_rows_fit(block, group_rows):
        # Row-group ragged exchange + unpack, one JAX op (tiled layout kept);
        # the merge below is the incumbent's, on the same bits.
        prebuild_send_group(world, group_rows)
        o_recv, l_recv = _SEND_GROUP_OPS[(world, group_rows)](
            send.reshape(world * block, vdim).contiguous(), 1 + real * local, _DCP_RANK
        )
        o_recv = o_recv.reshape(world, num_tokens, local, vdim)
        l_recv = l_recv.reshape(world, num_tokens, local)
        return _merge_received(o_recv, l_recv).to(torch.bfloat16)
    if envs.TPU_DCP_MLA_SEND_LAYOUT_UNPACK:
        # The exchange and the unpack (row slices and bitcasts only: bits
        # move, nothing is computed) as one JAX op; the merge below is the
        # incumbent's, on the same partials and lse.
        prebuild_send_unpack(world)
        o_recv, l_recv = _SEND_UNPACK_OPS[world](
            send.reshape(world * block, vdim).contiguous(), 1 + real * local, _DCP_RANK
        )
        o_recv = o_recv.reshape(world, num_tokens, local, vdim)
        l_recv = l_recv.reshape(world, num_tokens, local)
        return _merge_received(o_recv, l_recv).to(torch.bfloat16)
    recv = _ragged_all_to_all(send, 1 + real * local, group, skip=SEND_PAD_ROWS - 1)
    o_recv = recv[:, SEND_PAD_ROWS:].reshape(world, num_tokens, local, vdim)
    o_recv = o_recv.view(torch.bfloat16)
    l_recv_words = recv[:, SEND_PAD_ROWS - 1]
    hi = l_recv_words[:, :rows].to(torch.int32)
    lo = l_recv_words[:, rows : 2 * rows].to(torch.int32)
    l_recv = (
        ((hi << 16) | (lo & 0xFFFF))
        .view(torch.float32)
        .reshape(world, num_tokens, local)
    )
    return _merge_received(o_recv, l_recv).to(torch.bfloat16)


_RAGGED_A2A_OPS: dict = {}
# This rank's DCP index as an int32 [1] device tensor: the exchange takes it
# as data (XLA's SPMD export refuses `axis_index`'s partition-id).
_DCP_RANK: torch.Tensor | None = None


def _ragged_all_to_all(
    x: torch.Tensor, rows: torch.Tensor, group: Any, skip: int = 0
) -> torch.Tensor:
    """`_all_to_all(x)` for `x` `[world, R, V]`, moving only rows `[skip, skip +
    rows)` (`rows` int32 `[1]`, the same on every rank) of each block into the
    same rows of the receiving block; the rest of each received block is zero.
    One `jax.lax.ragged_all_to_all` over the DCP mesh axis
    (`TPU_DCP_LSE_MERGE_A2A_REAL_ROWS`; `skip` for TPU_DCP_MLA_SEND_LAYOUT)."""
    world, block, vdim = x.shape
    prebuild_ragged_all_to_all(world, skip)
    op = _RAGGED_A2A_OPS[(world, skip)]
    return op(x.reshape(world * block, vdim).contiguous(), rows, _DCP_RANK).reshape(
        world, block, vdim
    )


def _ragged_fake(
    x: torch.Tensor, rows: torch.Tensor, rank: torch.Tensor
) -> torch.Tensor:
    return torch.empty_like(x)


def prebuild_ragged_all_to_all(world: int, skip: int = 0) -> None:
    """Build (register) the real-rows exchange op for a DCP group of `world`
    ranks (rows from `skip` on) at model build, not inside a trace."""
    global _DCP_RANK
    if (world, skip) not in _RAGGED_A2A_OPS:
        from vllm_torchtpu.distributed.dcp import dcp_rank_operand

        # The same per-device int32 [1] rank buffer TPU_DCP_RANK_OPERAND hands
        # the DCP attention and indexer ops: one buffer for every consumer.
        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _RAGGED_A2A_OPS[(world, skip)] = _build_ragged_all_to_all(world, skip)


def _retain_rank(out: torch.Tensor) -> torch.Tensor:
    """`out` plus a zero read from the rank buffer, on the merge branch that
    does not run the ragged exchange: the backbone warmup hands a bucket over
    only between traces that lift the same buffers (as
    `small_all_reduce_op.retain`). A dead read would be eliminated, so the
    zero rides on `out`."""
    from vllm_torchtpu.compilation import shape_variants

    if _DCP_RANK is None or not shape_variants.in_warmup():
        return out
    value = _DCP_RANK[0].to(out.dtype)
    return out + (value - value)


def _build_ragged_all_to_all(world: int, skip: int = 0):
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    # Each rank's own block, as the DCP attention op lays out its per-rank
    # partials (`_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS`); the exchange runs
    # within each lane's DCP group.
    rank_spec = P(DCP_AXIS_NAME)

    def body(x, rows, rank):
        block = x.shape[0] // world
        me = rank[0].astype(jnp.int32)
        n = jnp.clip(rows[0], 0, block - skip).astype(jnp.int32)
        sizes = jnp.full((world,), n, jnp.int32)
        return jax.lax.ragged_all_to_all(
            x,
            jnp.zeros_like(x),
            jnp.arange(world, dtype=jnp.int32) * block + skip,  # block d -> rank d
            sizes,
            jnp.full((world,), me * block + skip, jnp.int32),  # lands at block `me`
            sizes,  # every rank sends the same count
            axis_name=DCP_AXIS_NAME,
        )

    def fn(x: jax.Array, rows: jax.Array, rank: jax.Array) -> jax.Array:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(rank_spec, P(), rank_spec),
            out_specs=rank_spec,
            check_vma=False,
        )(x, rows, rank)

    return sharded_jax_op(
        f"pallas::dcp_merge_ragged_all_to_all_{world}"
        + (f"_skip{skip}" if skip else ""),
        fn,
        mesh=mesh,
        input_partition_specs=(rank_spec, P(), rank_spec),
        output_partition_specs=rank_spec,
        fake=_ragged_fake,
    )


# How `merge_received_jax` forms the two weighted sums over ranks; the first
# is the one the fused small all-to-all merge uses. "reduce_f32": f32 products summed by
# a reduce (the HLO torch's merge compiles to; measured within a few elements
# per 1e5 of it, not bitwise); "dot_default" / "dot_bf16": bf16-operand dots
# (far off -- torch's multiply-then-sum is not lowered to a dot here).
MERGE_JAX_VARIANTS = ("reduce_f32", "dot_default", "dot_bf16")


def merge_received_jax(o, lse, variant: str = MERGE_JAX_VARIANTS[0]):
    """`_merge_received` in jnp on `[R, n, V]` bf16 partials and `[R, n]` f32
    lse (rows flattened): the f32 `[n, V]` merge."""
    import jax.numpy as jnp
    from jax import lax

    finite = jnp.isfinite(lse)
    shift = jnp.where(finite, lse, -jnp.inf).max(axis=0)
    shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
    weights = jnp.where(finite, jnp.exp(lse - shift[None]), 0.0)
    denom = weights.sum(axis=0)
    weights = weights / jnp.where(denom > 0, denom, 1.0)
    of = o.astype(jnp.float32)
    of = jnp.where(jnp.isnan(of), 0.0, of)
    w_hi = weights.astype(jnp.bfloat16).astype(jnp.float32)
    w_lo = weights - w_hi
    if variant == "reduce_f32":
        return (of * w_hi[..., None]).sum(axis=0) + (of * w_lo[..., None]).sum(axis=0)
    if variant == "dot_bf16":
        of, w_hi, w_lo = (x.astype(jnp.bfloat16) for x in (of, w_hi, w_lo))
    # Contract ranks, batch rows: [R, n, V] x [R, n] -> [n, V].
    dims = (((0,), (0,)), ((1,), (1,)))

    def dot(a, w):
        return lax.dot_general(
            a,
            w,
            dims,
            precision=lax.Precision.DEFAULT,
            preferred_element_type=jnp.float32,
        )

    return dot(of, w_hi) + dot(of, w_lo)


_SEND_UNPACK_OPS: dict = {}


def prebuild_send_unpack(world: int) -> None:
    """Build (register) the send-layout exchange + unpack op
    (`TPU_DCP_MLA_SEND_LAYOUT_UNPACK`) for a DCP group of `world` ranks."""
    global _DCP_RANK
    if world not in _SEND_UNPACK_OPS:
        from vllm_torchtpu.distributed.dcp import dcp_rank_operand

        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _SEND_UNPACK_OPS[world] = _build_send_unpack(world)


def _send_unpack_fake(x: torch.Tensor, rows: torch.Tensor, rank: torch.Tensor):
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    group = get_dcp_group()
    assert group is not None
    world = int(group.world_size)
    tl = x.shape[0] // world - SEND_PAD_ROWS
    return (
        x.new_empty((world, tl, x.shape[1]), dtype=torch.bfloat16),
        x.new_empty((world, tl), dtype=torch.float32),
    )


def _build_send_unpack(world: int):
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    rank_spec = P(DCP_AXIS_NAME)
    skip = SEND_PAD_ROWS - 1

    def body(x, rows, rank):
        block = x.shape[0] // world
        vdim = x.shape[1]
        me = rank[0].astype(jnp.int32)
        n = jnp.clip(rows[0], 0, block - skip).astype(jnp.int32)
        sizes = jnp.full((world,), n, jnp.int32)
        recv = jax.lax.ragged_all_to_all(
            x,
            jnp.zeros_like(x),
            jnp.arange(world, dtype=jnp.int32) * block + skip,
            sizes,
            jnp.full((world,), me * block + skip, jnp.int32),
            sizes,
            axis_name=DCP_AXIS_NAME,
        ).reshape(world, block, vdim)
        tl = block - SEND_PAD_ROWS
        o = lax.bitcast_convert_type(recv[:, SEND_PAD_ROWS:], jnp.bfloat16)
        words = recv[:, SEND_PAD_ROWS - 1]
        hi = words[:, :tl].astype(jnp.int32)
        lo = words[:, tl : 2 * tl].astype(jnp.int32)
        lse = lax.bitcast_convert_type((hi << 16) | (lo & 0xFFFF), jnp.float32)
        return o, lse

    def fn(
        x: jax.Array, rows: jax.Array, rank: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(rank_spec, P(), rank_spec),
            out_specs=(rank_spec, rank_spec),
            check_vma=False,
        )(x, rows, rank)

    return sharded_jax_op(
        f"pallas::dcp_send_layout_unpack_{world}",
        fn,
        mesh=mesh,
        input_partition_specs=(rank_spec, P(), rank_spec),
        output_partition_specs=(rank_spec, rank_spec),
        fake=_send_unpack_fake,
    )


def send_group_rows_fit(block: int, group_rows: int) -> bool:
    """Whether a send block of `block` rows splits into whole groups of
    `group_rows` rows (static: `block` is the bucket's)."""
    return group_rows in (8, 16) and block % group_rows == 0


_SEND_GROUP_OPS: dict = {}
_SEND_SMALL_OPS: dict = {}


def send_small_a2a_applies(num_tokens: int) -> bool:
    """Whether `merge_send_layout` exchanges this bucket with the Pallas
    small all-to-all (`TPU_DCP_MLA_SEND_SMALL_A2A`): `T <= cap` on the static
    bucket size, and only where the indexer's small all-gather already lifts
    the lane operand on every bucket (so no bucket lifts a buffer another
    does not)."""
    return bool(
        envs.TPU_DCP_MLA_SEND_SMALL_A2A
        and envs.TPU_DCP_INDEXER_SMALL_AG
        and envs.TPU_SC_TOPK_DCP_MERGE_FLAT
        and envs.TPU_DCP_RANK_OPERAND
        and num_tokens <= envs.TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS
    )


def fused_small_a2a_applies(num_tokens: int) -> bool:
    """Whether the decode MLA kernel runs this bucket's small all-to-all
    itself (`TPU_MLA_FUSED_SMALL_A2A`): a bucket `send_small_a2a_applies`
    takes whose attention grid is one step -- the decode body lever "step8"
    on the capped fused-prep row-widths body. The only token-count gate is
    `send_small_a2a_applies`' `T <= cap` (a power of 2): the rest is static --
    the cap itself at most DECODE_STEP8_TOKENS (a power-of-2 bucket at or
    below it is its own step) and within the small-chunk path. Fixed at trace
    time on the static bucket size. The cap is the one that applies at this
    DCP size (`dcp_mla_topk_cap`: the auto default caps only at DCP >= 8)."""
    from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel
    from vllm_torchtpu.layers.core.attention_interface import dcp_mla_topk_cap

    if not (envs.TPU_MLA_FUSED_SMALL_A2A and send_small_a2a_applies(num_tokens)):
        return False
    group = get_dcp_group()
    dcp_size = 1 if group is None else int(group.world_size)
    value = envs.TPU_MLA_DECODE_BODY_LEVERS
    levers = (
        sparse_kernel.DECODE_BODY_DEFAULT
        if value == "1"
        else tuple(
            v.strip() for v in (value or "").split(",") if v.strip() and v != "0"
        )
    )
    return bool(
        "step8" in levers
        and dcp_mla_topk_cap(dcp_size)
        and envs.TPU_DCP_MLA_FUSED_PREP
        and envs.TPU_MLA_DECODE_ROW_WIDTHS
        and envs.TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS
        <= sparse_kernel.DECODE_STEP8_TOKENS
        and envs.TPU_DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS
        <= envs.TPU_DCP_MLA_SMALL_CHUNK_TOKENS
    )


def merge_fused_recv(
    o_recv: torch.Tensor, l_recv: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """`merge_send_layout`'s small all-to-all merge from the partials and lse
    the fused-exchange attention op unpacked (`TPU_MLA_FUSED_SMALL_A2A`):
    `o_recv` bf16 `[world, T * local, V]`, `l_recv` f32 `[world, T * local]`,
    real rows only (the rest zeroed as the small all-to-all's unpack leaves
    them). Returns `[T, local, V]` bf16."""
    world, rows, vdim = o_recv.shape
    local = rows // num_tokens
    o_recv = o_recv.reshape(world, num_tokens, local, vdim)
    l_recv = l_recv.reshape(world, num_tokens, local)
    return _merge_received(o_recv, l_recv).to(torch.bfloat16)


def prebuild_send_small(world: int) -> None:
    """Build (register) the send-layout small all-to-all + unpack op
    (`TPU_DCP_MLA_SEND_SMALL_A2A`) for a DCP group of `world` ranks."""
    global _DCP_RANK
    if world not in _SEND_SMALL_OPS:
        from vllm_torchtpu.distributed.dcp import dcp_rank_operand

        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _SEND_SMALL_OPS[world] = _build_send_small(world)


def _send_small_fake(x, rows, rank, lane):
    return _send_unpack_fake(x, rows, rank)


def send_small_unpack_local(
    x, rows, rank, lane, *, world: int, axis_name: str, lane_axis: str | None
):
    """One rank's `TPU_DCP_MLA_SEND_SMALL_A2A` exchange + unpack inside a
    `shard_map` over the (lane, DCP) mesh: `_build_send_unpack`'s result, bit
    for bit.

    The kernel moves every destination block from the (8,128) tile row that
    holds the lse word row (`SEND_PAD_ROWS - 1`) to the block's end; rows
    before it stay unwritten and are never read. Partial rows past the real
    ones (padding tokens) arrive as the sender left them and are zeroed by
    the unpack, as the row-group exchange does."""
    from vllm_torchtpu.kernels.experimental.small_collectives.all_to_all import (
        small_all_to_all,
    )
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    block = x.shape[0] // world
    vdim = x.shape[1]
    recv = small_all_to_all(
        x,
        num_cores=world,
        axis=axis_name,
        row0=(SEND_PAD_ROWS - 1) // 8 * 8,
        rank=jnp.reshape(rank, (1,)),
        lane_axis=lane_axis,
        lane=None if lane_axis is None else jnp.reshape(lane, (1,)),
        hbm_operand=envs.TPU_SMALL_COLL_HBM_OPERAND,
    ).reshape(world, block, vdim)
    return _mask_unpack(recv, rows, block - SEND_PAD_ROWS)


def _build_send_small(world: int):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    rank_spec = P(DCP_AXIS_NAME)
    lane_axes = tuple(a for a in mesh.axis_names if a != DCP_AXIS_NAME)
    assert len(lane_axes) <= 1, mesh.axis_names
    lane_axis = lane_axes[0] if lane_axes else None

    def body(x, rows, rank, lane):
        return send_small_unpack_local(
            x,
            rows,
            rank,
            lane,
            world=world,
            axis_name=DCP_AXIS_NAME,
            lane_axis=lane_axis,
        )

    def fn(
        x: jax.Array, rows: jax.Array, rank: jax.Array, lane: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(rank_spec, P(), rank_spec, rank_spec),
            out_specs=(rank_spec, rank_spec),
            check_vma=False,
        )(x, rows, rank, lane)

    return sharded_jax_op(
        f"pallas::dcp_send_layout_small_a2a_{world}",
        fn,
        mesh=mesh,
        input_partition_specs=(rank_spec, P(), rank_spec, rank_spec),
        output_partition_specs=(rank_spec, rank_spec),
        fake=_send_small_fake,
    )


def prebuild_send_group(world: int, group_rows: int) -> None:
    """Build (register) the send-layout row-group exchange + unpack op
    (`TPU_DCP_MLA_SEND_GROUP_ROWS`) for a DCP group of `world` ranks."""
    global _DCP_RANK
    if (world, group_rows) not in _SEND_GROUP_OPS:
        from vllm_torchtpu.distributed.dcp import dcp_rank_operand

        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _SEND_GROUP_OPS[(world, group_rows)] = _build_send_group(world, group_rows)


def _mask_unpack(recv, rows, tl: int):
    """`_build_send_unpack`'s unpack of received blocks `[world, block, V]`
    with partial rows at or past `rows - 1` (the real ones) zeroed."""
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    world, _, vdim = recv.shape
    live = jnp.clip(rows[0] - 1, 0, tl).astype(jnp.int32)
    row = jax.lax.broadcasted_iota(jnp.int32, (world, tl, vdim), 1)
    part = recv[:, SEND_PAD_ROWS:]
    part = jnp.where(row < live, part, jnp.zeros_like(part))
    o = jax.lax.bitcast_convert_type(part, jnp.bfloat16)
    words = recv[:, SEND_PAD_ROWS - 1]
    hi = words[:, :tl].astype(jnp.int32)
    lo = words[:, tl : 2 * tl].astype(jnp.int32)
    lse = jax.lax.bitcast_convert_type((hi << 16) | (lo & 0xFFFF), jnp.float32)
    return o, lse


def send_group_unpack_local(
    x, rows, rank, *, world: int, group_rows: int, axis_name: str
):
    """One rank's `TPU_DCP_MLA_SEND_GROUP_ROWS` exchange + unpack inside a
    `shard_map` over the DCP axis: `_build_send_unpack`'s result, bit for bit.

    `x` is this rank's int16 send buffer `[world * block, V]`, `rows` int32
    `[1]` the row-ragged exchange's count `1 + real * local` from row
    `SEND_PAD_ROWS - 1`, `rank` this rank's int32 `[1]` DCP index. The ragged
    all-to-all moves whole `group_rows`-row groups `[first, last]` of each
    block, `first` holding the lse word row and `last` the last real partial
    row; on the (8,128)-tiled int16 buffer a group is one contiguous tile row,
    so XLA exchanges it in place. The extra rows it carries (pad rows before
    the lse row, padding tokens' partial rows) are never read or are zeroed."""
    from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS

    g = group_rows
    block = x.shape[0] // world
    vdim = x.shape[1]
    gb = block // g
    first = (SEND_PAD_ROWS - 1) // g
    me = rank[0].astype(jnp.int32)
    end = SEND_PAD_ROWS - 1 + jnp.clip(rows[0], 0, block - SEND_PAD_ROWS + 1)
    n = jnp.clip((end + g - 1) // g - first, 0, gb - first).astype(jnp.int32)
    sizes = jnp.full((world,), n, jnp.int32)
    xg = x.reshape(world * gb, g, vdim)
    recv = jax.lax.ragged_all_to_all(
        xg,
        jnp.zeros_like(xg),
        jnp.arange(world, dtype=jnp.int32) * gb + first,  # block d -> rank d
        sizes,
        jnp.full((world,), me * gb + first, jnp.int32),  # lands at block `me`
        sizes,
        axis_name=axis_name,
    ).reshape(world, block, vdim)
    return _mask_unpack(recv, rows, block - SEND_PAD_ROWS)


def _build_send_group(world: int, group_rows: int):
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    rank_spec = P(DCP_AXIS_NAME)

    def body(x, rows, rank):
        return send_group_unpack_local(
            x, rows, rank, world=world, group_rows=group_rows, axis_name=DCP_AXIS_NAME
        )

    def fn(
        x: jax.Array, rows: jax.Array, rank: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return jax.shard_map(
            body,
            mesh=mesh,
            in_specs=(rank_spec, P(), rank_spec),
            out_specs=(rank_spec, rank_spec),
            check_vma=False,
        )(x, rows, rank)

    return sharded_jax_op(
        f"pallas::dcp_send_layout_group{group_rows}_{world}",
        fn,
        mesh=mesh,
        input_partition_specs=(rank_spec, P(), rank_spec),
        output_partition_specs=(rank_spec, rank_spec),
        fake=_send_unpack_fake,
    )


def _zero_nan(o_f32: torch.Tensor) -> torch.Tensor:
    """Zero NaN entries of the received partials.

    A rank that owns no position of a row ran its kernel on a dummy key and may
    return NaN there; the merge gives it weight 0, but NaN * 0 = NaN. Selecting
    on the partial's own value (not on a weight or lse mask) keeps the select
    inside the merge's existing fusion: the decode HLO gains no op. Finite
    partials are untouched, bit for bit. A NaN partial of nonzero weight (never
    produced by an owning rank) would also be zeroed rather than propagated.
    """
    return torch.where(torch.isnan(o_f32), torch.zeros_like(o_f32), o_f32)


def _merge_received(o_recv: torch.Tensor, l_recv: torch.Tensor) -> torch.Tensor:
    """`[R, T, h, V]` partials and `[R, T, h]` lse -> the f32 `[T, h, V]` merge."""
    finite = torch.isfinite(l_recv)
    shift = torch.where(finite, l_recv, torch.full_like(l_recv, float("-inf"))).amax(
        dim=0
    )
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    weights = torch.where(
        finite, torch.exp(l_recv - shift.unsqueeze(0)), torch.zeros_like(l_recv)
    )
    denom = weights.sum(dim=0)
    weights = weights / torch.where(denom > 0, denom, torch.ones_like(denom))
    # The compiler lowers multiply-then-sum to a dot that rounds f32 operands to
    # bf16. Splitting the weights into a bf16 part and a residual keeps it exact.
    o_f32 = _zero_nan(o_recv.to(torch.float32))
    w_hi = weights.to(torch.bfloat16).to(torch.float32)
    w_lo = weights - w_hi
    return (o_f32 * w_hi.unsqueeze(-1)).sum(dim=0) + (o_f32 * w_lo.unsqueeze(-1)).sum(
        dim=0
    )


# ===================================================================== #
# The MTP drafter's sized merge (TPU_MTP_DCP_MERGE_SIZED).
# ===================================================================== #

# world -> the torch op over the DCP mesh; built at model build
# (`prebuild_sized_merge`), never inside a trace.
_SIZED_MERGE_OPS: dict = {}


def _rs_weighted(o: jax.Array, own_lse: jax.Array, lses: jax.Array) -> jax.Array:
    """`cp_lse_ag_out_rs`'s per-element arithmetic before its sum: `o` `[..., V]`
    a partial, `own_lse` `[...]` its lse, `lses` `[world, ...]` every rank's
    lse for the same rows. f32 `o * exp(own_lse - logsumexp(lses))`, zero where
    the weight is not positive (a zero-weight partial may be NaN)."""
    merged = jax.nn.logsumexp(lses, axis=0)
    weight = jnp.exp(own_lse - merged)[..., None]
    weighted = o.astype(jnp.float32) * weight
    return jnp.where(weight > 0, weighted, jnp.zeros_like(weighted))


def _pack_words(o_send: jax.Array, l_send: jax.Array) -> tuple[jax.Array, int]:
    """`[world, T, local, V]` 2-byte partial and f32 `[world, T, local]` lse ->
    int16 `[world, T*local + extra, V]` (`_all_to_all_packed`'s layout): the
    partial's words, then each destination's lse high halves and low halves,
    zero padded to whole V-wide rows. Integer ops only, so every bit moves."""
    world, tokens, local, vdim = o_send.shape
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_words = jax.lax.bitcast_convert_type(o_send, jnp.int16).reshape(world, rows, vdim)
    bits = jax.lax.bitcast_convert_type(l_send, jnp.int32).reshape(world, rows)
    hi = (bits >> 16).astype(jnp.int16)
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).astype(jnp.int16)
    l_words = jnp.pad(
        jnp.concatenate([hi, lo], axis=1), ((0, 0), (0, extra * vdim - words))
    )
    return jnp.concatenate(
        [o_words, l_words.reshape(world, extra, vdim)], axis=1
    ), extra


def _unpack_words(
    recv: jax.Array, tokens: int, local: int, dtype
) -> tuple[jax.Array, jax.Array]:
    """Inverse of `_pack_words` on the received blocks: `[world, T, local, V]`
    partials in `dtype` and the f32 `[world, T, local]` lse, bit for bit."""
    world, _, vdim = recv.shape
    rows = tokens * local
    o_recv = jax.lax.bitcast_convert_type(recv[:, :rows], dtype).reshape(
        world, tokens, local, vdim
    )
    l_words = recv[:, rows:].reshape(world, -1)
    hi = l_words[:, :rows].astype(jnp.int32)
    lo = l_words[:, rows : 2 * rows].astype(jnp.int32)
    l_recv = jax.lax.bitcast_convert_type((hi << 16) | (lo & 0xFFFF), jnp.float32)
    return o_recv, l_recv.reshape(world, tokens, local)


def sized_merge_local(
    o: jax.Array,
    lse: jax.Array,
    *,
    world: int,
    a2a_max_tokens: int,
    axis_name: str,
    small: dict | None = None,
) -> jax.Array:
    """One rank's merge inside a `shard_map` over the DCP axis.

    `o` `[T, H, V]` (2-byte dtype) and `lse` `[T, H]` are this rank's partial
    and lse over all `H = world * local` heads; returns this rank's merged head
    block `[T, local, V]` in `o.dtype`. `T` is concrete here, so the exchange is
    chosen per bucket:

    - `T <= a2a_max_tokens`: head block `d` of the partial and lse goes to rank
      `d` in one packed all-to-all; each rank weights the received partials as
      the reduce-scatter merge would (same lse gather values, logsumexp, exp,
      multiply, zero select) and adds them in rank order 0..world-1.
    - otherwise: the reduce-scatter merge, op for op (`cp_lse_ag_out_rs`): lse
      all-gather, weight, f32 reduce-scatter over heads.

    The two differ only in the order the `world` weighted f32 terms are summed.

    With `small` (TPU_MTP_DCP_MERGE_SMALL_A2A: `max_tokens`, `rank`, `lane`,
    `lane_axis`), buckets of `T <= small["max_tokens"]` move the packed words
    with the Pallas small all-to-all: each destination block padded to whole
    8-row tiles, every row moved (row0 = 0), so the received words the unpack
    reads are `lax.all_to_all`'s bit for bit.
    """
    tokens, heads, vdim = o.shape
    local = heads // world
    lse = lse.astype(jnp.float32)
    if tokens > a2a_max_tokens:
        lses = jax.lax.all_gather(lse, axis_name, axis=0)
        weighted = _rs_weighted(o, lse, lses)
        merged = jax.lax.psum_scatter(
            weighted, axis_name, scatter_dimension=1, tiled=True
        )
        return merged.astype(o.dtype)
    o_send = o.reshape(tokens, world, local, vdim).transpose(1, 0, 2, 3)
    l_send = lse.reshape(tokens, world, local).transpose(1, 0, 2)
    send, _ = _pack_words(o_send, l_send)
    if small is not None and tokens <= small["max_tokens"]:
        recv = _small_exchange(send, world=world, axis_name=axis_name, **small)
    else:
        recv = jax.lax.all_to_all(send, axis_name, 0, 0, tiled=True)
    o_recv, l_recv = _unpack_words(recv, tokens, local, o.dtype)
    # Rank r's weight reads every rank's lse for this rank's heads, which is
    # exactly l_recv `[world, T, local]`.
    terms = _rs_weighted(o_recv, l_recv, l_recv)
    # An explicit chain of adds (a reduce over the rank axis may be lowered to
    # a default-precision dot that rounds its f32 operands to bf16).
    acc = terms[0]
    for r in range(1, world):
        acc = acc + terms[r]
    return acc.astype(o.dtype)


def _small_exchange(
    send: jax.Array,
    *,
    world: int,
    axis_name: str,
    max_tokens: int,
    rank: jax.Array,
    lane: jax.Array,
    lane_axis: str | None,
) -> jax.Array:
    """`lax.all_to_all(send, axis_name, 0, 0, tiled=True)` of the packed
    `[world, R, V]` words through `small_all_to_all`: blocks padded to
    `ceil(R / 8) * 8` rows, all rows moved, the padding sliced off."""
    from vllm_torchtpu.kernels.experimental.small_collectives.all_to_all import (
        small_all_to_all,
    )

    del max_tokens
    _, rows, vdim = send.shape
    block = -(-rows // 8) * 8
    padded = jnp.pad(send, ((0, 0), (0, block - rows), (0, 0)))
    recv = small_all_to_all(
        padded.reshape(world * block, vdim),
        num_cores=world,
        axis=axis_name,
        row0=0,
        rank=jnp.reshape(rank, (1,)),
        lane_axis=lane_axis,
        lane=None if lane_axis is None else jnp.reshape(lane, (1,)),
        hbm_operand=envs.TPU_SMALL_COLL_HBM_OPERAND,
    ).reshape(world, block, vdim)
    return recv[:, :rows]


def prebuild_sized_merge(world: int) -> None:
    """Register the drafter's sized merge op for a DCP group of `world` ranks
    (TPU_MTP_DCP_MERGE_SIZED). Call at model build, outside any trace."""
    if not envs.TPU_MTP_DCP_MERGE_SIZED or world <= 1 or world in _SIZED_MERGE_OPS:
        return
    from jax.sharding import PartitionSpec as P

    from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
    from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op

    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    cap = int(envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS)
    # Each rank's own rows, as the DCP attention op lays out its partials
    # (`_SPARSE_MLA_DCP_OUTPUT_PARTITION_SPECS`).
    spec = P(DCP_AXIS_NAME)

    def fake(output: torch.Tensor, lse: torch.Tensor, *_) -> torch.Tensor:
        tokens, heads, vdim = output.shape
        return output.new_empty((tokens, heads // world, vdim))

    if not (envs.TPU_MTP_DCP_MERGE_SMALL_A2A and envs.TPU_DCP_RANK_OPERAND):

        def body(o, lse):
            return sized_merge_local(
                o, lse, world=world, a2a_max_tokens=cap, axis_name=DCP_AXIS_NAME
            )

        def fn(output: jax.Array, lse: jax.Array) -> jax.Array:
            return jax.shard_map(
                body, mesh=mesh, in_specs=(spec, spec), out_specs=spec, check_vma=False
            )(output, lse)

        _SIZED_MERGE_OPS[world] = sharded_jax_op(
            f"pallas::mtp_dcp_merge_sized_{world}",
            fn,
            mesh=mesh,
            input_partition_specs=(spec, spec),
            output_partition_specs=spec,
            fake=fake,
        )
        return

    # TPU_MTP_DCP_MERGE_SMALL_A2A: the op takes this rank's DCP index and lane
    # as data on every bucket (the small all-to-all addresses peers by them);
    # buckets above the small cap ignore them.
    from vllm_torchtpu.distributed.dcp import dcp_lane_operand, dcp_rank_operand

    small_cap = int(envs.TPU_MTP_DCP_MERGE_SMALL_A2A_MAX_TOKENS)
    lane_axes = tuple(a for a in mesh.axis_names if a != DCP_AXIS_NAME)
    assert len(lane_axes) <= 1, mesh.axis_names
    lane_axis = lane_axes[0] if lane_axes else None
    rank_t = dcp_rank_operand(torch.device("tpu"))
    lane_t = dcp_lane_operand(torch.device("tpu"))

    def body_small(o, lse, rank, lane):
        small = {
            "max_tokens": small_cap,
            "rank": rank,
            "lane": lane,
            "lane_axis": lane_axis,
        }
        return sized_merge_local(
            o,
            lse,
            world=world,
            a2a_max_tokens=cap,
            axis_name=DCP_AXIS_NAME,
            small=small,
        )

    def fn_small(
        output: jax.Array, lse: jax.Array, rank: jax.Array, lane: jax.Array
    ) -> jax.Array:
        return jax.shard_map(
            body_small,
            mesh=mesh,
            in_specs=(spec, spec, spec, spec),
            out_specs=spec,
            check_vma=False,
        )(output, lse, rank, lane)

    op = sharded_jax_op(
        f"pallas::mtp_dcp_merge_sized_small_{world}",
        fn_small,
        mesh=mesh,
        input_partition_specs=(spec, spec, spec, spec),
        output_partition_specs=spec,
        fake=fake,
    )

    def merge(output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        return op(output, lse, rank_t, lane_t)

    _SIZED_MERGE_OPS[world] = merge
