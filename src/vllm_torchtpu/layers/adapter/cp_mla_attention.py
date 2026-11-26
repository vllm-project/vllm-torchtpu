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
"""DCP (Decode Context Parallelism) head all-gather and LSE partial merge for MLA."""

from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from torch.distributed import _functional_collectives as funcol

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import (
    DCP_AXIS_NAME,
    dcp_lane_operand,
    dcp_rank_operand,
    get_dcp_group,
    get_or_create_dcp_mesh,
)
from vllm_torchtpu.distributed.dcp_comm import (
    cp_lse_ag_out_rs,
    lse_rescale_head_major_op,
)
from vllm_torchtpu.distributed.sharded_jax_op import sharded_jax_op
from vllm_torchtpu.kernels.experimental.small_collectives.all_to_all import (
    small_all_to_all,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_kernel
from vllm_torchtpu.kernels.mla.sparse.kernel import SEND_PAD_ROWS
from vllm_torchtpu.kernels.quantized_matmul.fp8_window_all_gather import (
    fp8_window_all_gather,
)
from vllm_torchtpu.layers.core.attention_interface import (
    _lse_gathered_once,
    _mask_unpack,
    _merge_chunks,
)
from vllm_torchtpu.layers.core.attention_interface import (
    merge_received_jax as merge_received_jax,
)
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    DCP_MERGE_CHUNKED_PREFILL_SPLITS,
    DCP_MERGE_WEIGHTED_HM_MIN_TOKENS,
    DCP_MLA_SEND_GROUP_ROWS,
    DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS,
    DCP_MLA_SMALL_CHUNK_BATCH,
    DCP_MLA_SMALL_CHUNK_TOKENS,
    bucket_gate_edge,
    glm_dsa_v7x_active,
    in_shape_variants_warmup,
    resolve_dcp_lse_merge_a2a_max_tokens,
    resolve_dcp_mla_topk_cap,
)


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


_LOCAL_Q_ALL_GATHER_OPS: dict[int, Any] = {}
_LOCAL_Q_NORM_ALL_GATHER_OPS: dict[tuple[int, float], Any] = {}


def local_q_norm_all_gather_op(eps: float) -> Any:
    """Map `(x, w_q, w_s, rank, lane, norm_w)` -> `[m, n_out]` with fused RMSNorm."""
    eps_f = float(eps)
    mesh = get_or_create_dcp_mesh()
    world = int(mesh.shape[DCP_AXIS_NAME])
    key = (world, eps_f)
    op = _LOCAL_Q_NORM_ALL_GATHER_OPS.get(key)
    if op is not None:
        return op
    lane_axes = tuple(a for a in mesh.axis_names if a != DCP_AXIS_NAME)
    assert len(lane_axes) <= 1, mesh.axis_names
    lane_axis = lane_axes[0] if lane_axes else None
    spec = P(DCP_AXIS_NAME)

    def body(x, w_q, w_s, rank, lane, norm_w):
        return fp8_window_all_gather(
            x,
            w_q,
            w_s,
            rank,
            num_cores=world,
            axis=DCP_AXIS_NAME,
            lane_axis=lane_axis,
            lane=None if lane_axis is None else lane,
            norm_w=norm_w,
            eps=eps_f,
        )

    def fn(
        x: jax.Array,
        w_q: jax.Array,
        w_s: jax.Array,
        rank: jax.Array,
        lane: jax.Array,
        norm_w: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 6, out_specs=spec, check_vma=False
        )(x, w_q, w_s, rank, lane, norm_w)

    def fake(x: torch.Tensor, w_q: torch.Tensor, *_) -> torch.Tensor:
        return x.new_empty((x.shape[0], w_q.shape[1]))

    eps_tag = f"{eps_f:.12g}".replace("-", "m").replace("+", "p").replace(".", "_")
    op = sharded_jax_op(
        f"pallas::fp8_window_all_gather_rn_{world}_{eps_tag}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 6,
        output_partition_specs=spec,
        fake=fake,
    )
    _LOCAL_Q_NORM_ALL_GATHER_OPS[key] = op
    return op


def local_q_all_gather_op() -> Any | None:
    """Build or fetch the fused FP8 column-window matmul + DCP all-gather op."""
    mesh = get_or_create_dcp_mesh()
    world = int(mesh.shape[DCP_AXIS_NAME])
    op = _LOCAL_Q_ALL_GATHER_OPS.get(world)
    if op is not None:
        return op
    lane_axes = tuple(a for a in mesh.axis_names if a != DCP_AXIS_NAME)
    assert len(lane_axes) <= 1, mesh.axis_names
    lane_axis = lane_axes[0] if lane_axes else None
    spec = P(DCP_AXIS_NAME)

    def body(x, w_q, w_s, rank, lane):
        return fp8_window_all_gather(
            x,
            w_q,
            w_s,
            rank,
            num_cores=world,
            axis=DCP_AXIS_NAME,
            lane_axis=lane_axis,
            lane=None if lane_axis is None else lane,
        )

    def fn(
        x: jax.Array,
        w_q: jax.Array,
        w_s: jax.Array,
        rank: jax.Array,
        lane: jax.Array,
    ) -> jax.Array:
        return jax.shard_map(
            body, mesh=mesh, in_specs=(spec,) * 5, out_specs=spec, check_vma=False
        )(x, w_q, w_s, rank, lane)

    def fake(x: torch.Tensor, w_q: torch.Tensor, *_args: Any) -> torch.Tensor:
        return x.new_empty((x.shape[0], w_q.shape[1]))

    op = sharded_jax_op(
        f"pallas::fp8_window_all_gather_{world}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 5,
        output_partition_specs=spec,
        fake=fake,
    )
    _LOCAL_Q_ALL_GATHER_OPS[world] = op
    return op


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
    `merge_lse_partials_all_to_all`, which gives the same result.

    Args:
      output: `[num_tokens, tp_local_num_heads * dcp_size, kv_lora_rank]`,
        this rank's partial.
      lse: `[num_tokens, tp_local_num_heads * dcp_size]`, the matching log-sum-exps.
        Entries are `-inf` for tokens this rank owns none of the top-k for;
        such a rank was still fed a dummy key (the kernel requires
        `kv_len >= 1`) and its output must not be weighted in.
      is_backbone: True if the caller is the backbone, the only model that is
        traced separately per token bucket. Only then may the merge compare
        the token count to TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS, since that
        comparison is fixed at trace time. Pass False from the MTP drafter,
        which is traced once for every bucket: it always takes the
        reduce-scatter, exactly as with TPU_DCP_LSE_MERGE_A2A unset.

    Returns:
      `[num_tokens, tp_local_num_heads, kv_lora_rank]` in `output.dtype` --
      this rank's head slice of the whole-context result.
    """
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return output

    glm_v7x = glm_dsa_v7x_active()
    narrow = envs.TPU_DCP_RS_OUT_PRECISION and is_backbone
    use_a2a = bool(envs.TPU_DCP_LSE_MERGE_A2A or glm_v7x)
    if (
        use_a2a
        and is_backbone
        and output.shape[0]
        <= resolve_dcp_lse_merge_a2a_max_tokens(envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS)
    ):
        a2a_kwargs: dict[str, Any] = {}
        if narrow and wire_dtype is not None:
            a2a_kwargs["send_dtype"] = wire_dtype
        if query_start_loc is not None:
            a2a_kwargs["query_start_loc"] = query_start_loc
        return merge_lse_partials_all_to_all(
            output,
            lse,
            group,
            **a2a_kwargs,
        )

    rs_dtype = (wire_dtype or output.dtype) if narrow else None
    chunks = (
        _merge_chunks(output.shape[0], glm_active=glm_v7x)
        if (is_backbone and glm_v7x)
        else 1
    )
    if chunks > 1:
        return _retain_rank(
            _merge_chunked(output, lse, group, chunks, rs_dtype=rs_dtype)
        )
    return _retain_rank(cp_lse_ag_out_rs(output, lse, group, rs_dtype=rs_dtype))


def _gather_lse(lse_f32: torch.Tensor, group: Any) -> torch.Tensor:
    return group.all_gather(lse_f32.unsqueeze(0).contiguous(), dim=0)


def _weighted_reduce_scatter_heads_major(
    output: torch.Tensor,
    lse_f32: torch.Tensor,
    merged_lse: torch.Tensor,
    group: Any,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    glm_active = glm_dsa_v7x_active()
    wire = rs_dtype
    rescale_op = None
    if output.device.type == "tpu" and glm_active:
        rescale_op = lse_rescale_head_major_op(wire)
    if rescale_op is not None:
        x = rescale_op(output, lse_f32, merged_lse)
    else:
        w = torch.exp(lse_f32 - merged_lse).transpose(0, 1).unsqueeze(-1)
        x = output.transpose(0, 1).to(torch.float32) * w
        x = torch.where(w > 0, x, torch.zeros_like(x))
        if wire is not None:
            x = x.to(wire)
    return group.reduce_scatter(x.contiguous(), dim=0).transpose(0, 1).to(output.dtype)


def _weighted_reduce_scatter(
    output: torch.Tensor,
    lse_f32: torch.Tensor,
    lses: torch.Tensor,
    group: Any,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    merged_lse = torch.logsumexp(lses, dim=0)
    if glm_dsa_v7x_active() and output.shape[0] * max(
        DCP_MERGE_CHUNKED_PREFILL_SPLITS, 1
    ) > bucket_gate_edge(DCP_MERGE_WEIGHTED_HM_MIN_TOKENS):
        return _weighted_reduce_scatter_heads_major(
            output, lse_f32, merged_lse, group, rs_dtype
        )
    weight = torch.exp(lse_f32 - merged_lse).unsqueeze(-1)
    weighted = output.to(torch.float32) * weight
    weighted = torch.where(weight > 0, weighted, torch.zeros_like(weighted))
    wire = rs_dtype
    if wire is not None:
        weighted = weighted.to(wire)
    return group.reduce_scatter(weighted.contiguous(), dim=1).to(output.dtype)


def _merge_chunked_lse_gathered(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    chunks: int,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    step = output.shape[0] // chunks
    lse_f32 = lse.to(torch.float32)
    head = _gather_lse(lse_f32[:step], group)
    rest = _gather_lse(lse_f32[step:], group)
    rows = [head] + [rest[:, (c - 1) * step : c * step] for c in range(1, chunks)]
    pieces = [
        _weighted_reduce_scatter(
            output[c * step : (c + 1) * step],
            lse_f32[c * step : (c + 1) * step],
            rows[c],
            group,
            rs_dtype,
        )
        for c in range(chunks)
    ]
    return torch.cat(pieces, dim=0)


def _merge_chunked(
    output: torch.Tensor,
    lse: torch.Tensor,
    group: Any,
    chunks: int,
    rs_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if _lse_gathered_once(output.shape[0]):
        return _merge_chunked_lse_gathered(output, lse, group, chunks, rs_dtype)
    step = output.shape[0] // chunks
    lse_f32 = lse.to(torch.float32)
    pieces = [
        _weighted_reduce_scatter(
            output[c * step : (c + 1) * step],
            lse_f32[c * step : (c + 1) * step],
            _gather_lse(lse_f32[c * step : (c + 1) * step], group),
            group,
            rs_dtype,
        )
        for c in range(chunks)
    ]
    return torch.cat(pieces, dim=0)


def _all_to_all(x: torch.Tensor, group: Any) -> torch.Tensor:
    out = funcol.all_to_all_single(x.contiguous(), None, None, group.device_group)
    if isinstance(out, funcol.AsyncCollectiveTensor):
        return funcol.wait_tensor(out)
    return out


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
    """
    world = int(group.world_size)
    num_tokens, num_heads, vdim = output.shape
    local = num_heads // world
    # Destination-major: [world, T, local, V] and [world, T, local].
    o_send = output.reshape(num_tokens, world, local, vdim).transpose(0, 1)
    if send_dtype is not None:
        o_send = o_send.to(send_dtype)
    l_send = lse.to(torch.float32).reshape(num_tokens, world, local).transpose(0, 1)
    if glm_dsa_v7x_active() and o_send.element_size() == 2:
        real_tokens = (
            query_start_loc[-1:].to(torch.int32)
            if query_start_loc is not None
            else None
        )
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
    world, tokens, local, vdim = o_send.shape
    rows = tokens * local
    words = rows * 2
    extra = -(-words // vdim)
    o_words = o_send.contiguous().reshape(world, rows, vdim).view(torch.int16)
    bits = l_send.contiguous().view(torch.int32).reshape(world, rows)
    hi = (bits >> 16).to(torch.int16)
    lo = (((bits & 0xFFFF) ^ 0x8000) - 0x8000).to(torch.int16)
    l_words = torch.nn.functional.pad(
        torch.cat([hi, lo], dim=1), (0, extra * vdim - words)
    )
    if real_tokens is not None and o_send.device.type == "tpu":
        send = torch.cat([l_words.reshape(world, extra, vdim), o_words], dim=1)
        recv = _ragged_all_to_all(send, extra + real_tokens * local, group)
        o_recv = recv[:, extra:].reshape(world, tokens, local, vdim).view(o_send.dtype)
        l_recv_words = recv[:, :extra].reshape(world, extra * vdim)
    else:
        send = torch.cat([o_words, l_words.reshape(world, extra, vdim)], dim=1)
        recv = _all_to_all(send, group)
        o_recv = recv[:, :rows].reshape(world, tokens, local, vdim).view(o_send.dtype)
        l_recv_words = recv[:, rows:].reshape(world, extra * vdim)
    hi_r = l_recv_words[:, :rows].to(torch.int32)
    lo_r = l_recv_words[:, rows:words].to(torch.int32)
    l_recv = (
        ((hi_r << 16) | (lo_r & 0xFFFF))
        .view(torch.float32)
        .reshape(world, tokens, local)
    )
    return o_recv, l_recv


def send_layout_applies(
    num_tokens: int,
    num_heads: int,
    vdim: int,
    dtype: torch.dtype,
    is_backbone: bool,
) -> bool:
    """Whether the decode kernel writes the packed A2A send buffer directly."""
    group = get_dcp_group()
    if group is None or int(group.world_size) == 1:
        return False
    if not (
        glm_dsa_v7x_active()
        and is_backbone
        and dtype == torch.bfloat16
        and 0
        < num_tokens
        <= resolve_dcp_lse_merge_a2a_max_tokens(envs.TPU_DCP_LSE_MERGE_A2A_MAX_TOKENS)
        and num_tokens <= sparse_kernel.FUSED_PREP_MAX_TOKENS
    ):
        return False
    _, batch = sparse_kernel._chunk_kernel_batch_size(
        num_tokens,
        sparse_kernel.DEFAULT_ATTENTION_KERNEL_BATCH_SIZE,
        DCP_MLA_SMALL_CHUNK_TOKENS,
        DCP_MLA_SMALL_CHUNK_BATCH,
    )
    return sparse_kernel.send_layout_supported(
        num_tokens, num_heads, int(group.world_size), vdim, batch
    )


def send_group_rows_fit(block: int, group_rows: int) -> bool:
    """True when ``block`` divides evenly into ``group_rows``-row tiles."""
    return group_rows in (8, 16) and block % group_rows == 0


def _send_small_a2a_max_tokens(dcp_size: int) -> int:
    if dcp_size >= 16:
        return 32
    return DCP_MLA_SEND_SMALL_A2A_MAX_TOKENS


def send_small_a2a_applies(num_tokens: int) -> bool:
    """True when standalone small-collectives all-to-all serves ``num_tokens``."""
    group = get_dcp_group()
    dcp_size = 1 if group is None else int(group.world_size)
    return bool(
        glm_dsa_v7x_active() and 0 < num_tokens <= _send_small_a2a_max_tokens(dcp_size)
    )


def fused_small_a2a_applies(num_tokens: int) -> bool:
    """True when in-kernel send_exchange serves ``num_tokens`` on GLM-5 DSA v7x."""
    group = get_dcp_group()
    dcp_size = 1 if group is None else int(group.world_size)
    max_tokens = _send_small_a2a_max_tokens(dcp_size)
    if not (glm_dsa_v7x_active() and 0 < num_tokens <= max_tokens):
        return False
    return bool(
        "step8" in sparse_kernel.DECODE_BODY_DEFAULT
        and resolve_dcp_mla_topk_cap(envs.TPU_DCP_MLA_TOPK_CAP, dcp_size)
        and max_tokens <= sparse_kernel.DECODE_STEP8_TOKENS
        and max_tokens <= DCP_MLA_SMALL_CHUNK_TOKENS
    )


def merge_send_layout(
    send: torch.Tensor, query_start_loc: torch.Tensor, num_tokens: int
) -> torch.Tensor:
    """Exchange and LSE-merge a decode kernel ``send_layout`` buffer."""
    world, block, vdim = send.shape
    local = (block - SEND_PAD_ROWS) // num_tokens
    real = query_start_loc[-1:].to(torch.int32)
    if send_small_a2a_applies(num_tokens):
        prebuild_send_small(world)
        lane = _DCP_LANE if _DCP_LANE is not None else dcp_lane_operand()
        o_recv, l_recv = _SEND_SMALL_OPS[world](
            send.reshape(world * block, vdim).contiguous(),
            1 + real * local,
            _DCP_RANK,
            lane,
        )
        o_recv = o_recv.reshape(world, num_tokens, local, vdim)
        l_recv = l_recv.reshape(world, num_tokens, local)
        return _merge_received(o_recv, l_recv).to(torch.bfloat16)
    group_rows = DCP_MLA_SEND_GROUP_ROWS
    if group_rows > 0 and send_group_rows_fit(block, group_rows):
        prebuild_send_group(world, group_rows)
        o_recv, l_recv = _SEND_GROUP_OPS[(world, group_rows)](
            send.reshape(world * block, vdim).contiguous(),
            1 + real * local,
            _DCP_RANK,
        )
        o_recv = o_recv.reshape(world, num_tokens, local, vdim)
        l_recv = l_recv.reshape(world, num_tokens, local)
        return _merge_received(o_recv, l_recv).to(torch.bfloat16)
    prebuild_send_unpack(world)
    o_recv, l_recv = _SEND_UNPACK_OPS[world](
        send.reshape(world * block, vdim).contiguous(),
        1 + real * local,
        _DCP_RANK,
    )
    o_recv = o_recv.reshape(world, num_tokens, local, vdim)
    l_recv = l_recv.reshape(world, num_tokens, local)
    return _merge_received(o_recv, l_recv).to(torch.bfloat16)


_RAGGED_A2A_OPS: dict[tuple[int, int], Any] = {}
_SEND_UNPACK_OPS: dict[int, Any] = {}
_SEND_GROUP_OPS: dict[tuple[int, int], Any] = {}
_SEND_SMALL_OPS: dict[int, Any] = {}
_DCP_RANK: torch.Tensor | None = None
_DCP_LANE: torch.Tensor | None = None


def _ragged_fake(
    x: torch.Tensor, rows: torch.Tensor, rank: torch.Tensor
) -> torch.Tensor:
    del rows, rank
    return torch.empty_like(x)


def _retain_rank(out: torch.Tensor) -> torch.Tensor:
    if _DCP_RANK is None or not in_shape_variants_warmup():
        return out
    value = _DCP_RANK[0].to(out.dtype)
    return out + (value - value)


def _ragged_all_to_all(
    x: torch.Tensor, rows: torch.Tensor, group: Any, skip: int = 0
) -> torch.Tensor:
    del group
    world, block, vdim = x.shape
    prebuild_ragged_all_to_all(world, skip)
    op = _RAGGED_A2A_OPS[(world, skip)]
    return op(x.reshape(world * block, vdim).contiguous(), rows, _DCP_RANK).reshape(
        world, block, vdim
    )


def prebuild_ragged_all_to_all(world: int, skip: int = 0) -> None:
    """Pre-register the real-rows ragged A2A custom op outside Dynamo."""
    global _DCP_RANK
    if (world, skip) not in _RAGGED_A2A_OPS:
        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _RAGGED_A2A_OPS[(world, skip)] = _build_ragged_all_to_all(world, skip)


def _build_ragged_all_to_all(world: int, skip: int = 0) -> Any:
    mesh = get_or_create_dcp_mesh()
    assert mesh.shape[DCP_AXIS_NAME] == world, (mesh.shape, world)
    rank_spec = P(DCP_AXIS_NAME)

    def body(x, rows, rank):
        block = x.shape[0] // world
        me = rank[0].astype(jnp.int32)
        n = jnp.clip(rows[0], 0, block - skip).astype(jnp.int32)
        sizes = jnp.full((world,), n, jnp.int32)
        return jax.lax.ragged_all_to_all(
            x,
            jnp.zeros_like(x),
            jnp.arange(world, dtype=jnp.int32) * block + skip,
            sizes,
            jnp.full((world,), me * block + skip, jnp.int32),
            sizes,
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


def _send_unpack_fake(
    x: torch.Tensor, rows: torch.Tensor, rank: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del rows, rank
    group = get_dcp_group()
    assert group is not None
    world = int(group.world_size)
    tl = x.shape[0] // world - SEND_PAD_ROWS
    return (
        x.new_empty((world, tl, x.shape[1]), dtype=torch.bfloat16),
        x.new_empty((world, tl), dtype=torch.float32),
    )


def prebuild_send_unpack(world: int) -> None:
    """Pre-register the ragged send-unpack custom op outside Dynamo."""
    global _DCP_RANK
    if world not in _SEND_UNPACK_OPS:
        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _SEND_UNPACK_OPS[world] = _build_send_unpack(world)


def _build_send_unpack(world: int) -> Any:
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
        return _mask_unpack(recv, rows, block - SEND_PAD_ROWS)

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


def prebuild_send_group(world: int, group_rows: int) -> None:
    """Pre-register the tile-aligned send-group custom op outside Dynamo."""
    global _DCP_RANK
    if (world, group_rows) not in _SEND_GROUP_OPS:
        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _SEND_GROUP_OPS[(world, group_rows)] = _build_send_group(world, group_rows)


def send_group_unpack_local(
    x: jax.Array,
    rows: jax.Array,
    rank: jax.Array,
    *,
    world: int,
    group_rows: int,
    axis_name: str,
) -> tuple[jax.Array, jax.Array]:
    """Run one rank's ``group_rows``-aligned ragged all-to-all and unpack."""
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
        jnp.arange(world, dtype=jnp.int32) * gb + first,
        sizes,
        jnp.full((world,), me * gb + first, jnp.int32),
        sizes,
        axis_name=axis_name,
    ).reshape(world, block, vdim)
    return _mask_unpack(recv, rows, block - SEND_PAD_ROWS)


def _build_send_group(world: int, group_rows: int) -> Any:
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


def _send_small_fake(
    x: torch.Tensor, rows: torch.Tensor, rank: torch.Tensor, lane: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    del lane
    return _send_unpack_fake(x, rows, rank)


def send_small_unpack_local(
    x: jax.Array,
    rows: jax.Array,
    rank: jax.Array,
    lane: jax.Array,
    *,
    world: int,
    axis_name: str,
    lane_axis: str | None,
) -> tuple[jax.Array, jax.Array]:
    """Run ``small_all_to_all`` on the send layout and unpack."""
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
        hbm_operand=True,
    ).reshape(world, block, vdim)
    return _mask_unpack(recv, rows, block - SEND_PAD_ROWS)


def prebuild_send_small(world: int) -> None:
    """Pre-register the small-collectives send-unpack custom op outside Dynamo."""
    global _DCP_RANK, _DCP_LANE
    if world not in _SEND_SMALL_OPS:
        _DCP_RANK = dcp_rank_operand(torch.device("tpu"))
        _DCP_LANE = dcp_lane_operand(torch.device("tpu"))
        _SEND_SMALL_OPS[world] = _build_send_small(world)


def _build_send_small(world: int) -> Any:
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


def _zero_nan(o_f32: torch.Tensor) -> torch.Tensor:
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
    o_f32 = _zero_nan(o_recv.to(torch.float32))
    w_hi = weights.to(torch.bfloat16).to(torch.float32)
    w_lo = weights - w_hi
    return (o_f32 * w_hi.unsqueeze(-1)).sum(dim=0) + (o_f32 * w_lo.unsqueeze(-1)).sum(
        dim=0
    )
