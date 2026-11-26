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
"""Per-bucket sequence-parallel MoE and dense prefill for GLM-5 DSA on TPU v7x."""

import functools
import inspect
import types
from collections.abc import Callable
from itertools import islice
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from vllm.compilation import backends as vllm_compilation_backends
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_reduce_scatter,
)
from vllm.model_executor.layers import vocab_parallel_embedding as vpe
from vllm.model_executor.models import deepseek_v2

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.ep_mesh import (
    EP_AXIS_NAME,
    build_ep_mesh,
    ep_mesh_index,
    ep_rank_order,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.kv_cache_append import (
    owned_rows_bound_from_local,
)
from vllm_torchtpu.kernels.fused_moe.grouped_shared_expert import (
    grouped_shared_expert_sharded,
)
from vllm_torchtpu.layers.adapter import (
    dcp_token_parallel_op,
    fused_moe_ep,
    replicated_embedding,
    rmsnorm,
    small_all_reduce_op,
    token_padding,
)
from vllm_torchtpu.layers.adapter.dcp_token_parallel_op import MIN_TOKENS
from vllm_torchtpu.layers.adapter.linear_common import (
    KEEP_VLLM_LAYOUT_ATTR,
    WEIGHT_FLIPPED_ATTR,
    retain_input,
)
from vllm_torchtpu.layers.adapter.moe_hierarchical import ep_mesh_is_rank_ordered
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    bucket_gate_edge,
    glm_dsa_v7x_active,
    in_shape_variants_warmup,
    select_sp_shared_group,
)
from vllm_torchtpu.utils import fresh_buffer, synchronize_tensors

pallas: Any
try:
    from torch_tpu._internal import pallas
except ModuleNotFoundError:
    pallas = None

logger = init_logger(__name__)

# On an armed decoder layer: every tensor either branch reads, in a fixed order.
KEEP_ATTR = "_tpu_sp_keep"
# On an armed decoder layer's mlp: the replicated shared expert and the op.
# On the backbone: every MoE layer's shared-expert slice, stacked on dim 0
# (w13, s13, w2, s2). Four model-wide tensors rather than four per layer: the
# smaller buckets' programs lift them too (bucket hand-over), and each lifted
# tensor is one more argument of every decode step.
SHARED_STACK = (
    "_tpu_sp_shared_w13",
    "_tpu_sp_shared_s13",
    "_tpu_sp_shared_w2",
    "_tpu_sp_shared_s2",
)
DENSE_STACK = (
    "_tpu_sp_dense_w13",
    "_tpu_sp_dense_s13",
    "_tpu_sp_dense_w2",
    "_tpu_sp_dense_s2",
)
# On an armed decoder layer's mlp: the stacked tensors, its row in them, the op.
SHARED_ATTR = "_tpu_sp_shared"
SHARED_INDEX = "_tpu_sp_shared_index"
SHARED_OP = "_tpu_sp_shared_op"
DENSE_ATTR = "_tpu_sp_dense"
DENSE_INDEX = "_tpu_sp_dense_index"
DENSE_OP = "_tpu_sp_dense_op"
EMBED_RS_ATTR = "_tpu_sp_embed_rs"
DEFER_FINAL_AG_ATTR = "_tpu_sp_defer_final_ag"
OP_ATTR = "_tpu_sp_moe_op"
SCALE_ATTR = "_tpu_sp_routed_scale"

UPSTREAM_DECODER_LAYER_FORWARD: Any = deepseek_v2.DeepseekV2DecoderLayer.forward
UPSTREAM_MODEL_FORWARD: Any = getattr(
    deepseek_v2.DeepseekV2Model.forward,
    "__wrapped__",
    deepseek_v2.DeepseekV2Model.forward,
)

_SP_ENABLED = False


@torch.compiler.assume_constant_result
def _skip_padded_tokens() -> bool:
    return bool(envs.TPU_MOE_SKIP_PADDED_TOKENS or glm_dsa_v7x_active())


def _retain_buffers(out: torch.Tensor, *buffers: torch.Tensor | None) -> torch.Tensor:
    if not in_shape_variants_warmup():
        return out
    for buf in buffers:
        if buf is not None:
            out = retain_input(out, buf)
    return out


def applies(num_tokens: int) -> bool:
    """True when static backbone bucket ``num_tokens`` runs sequence-parallel."""
    return (
        _SP_ENABLED
        and vllm_compilation_backends.model_tag == "backbone"
        and num_tokens > bucket_gate_edge(MIN_TOKENS)
    )


# ---------------------------------------------------------------------------
# The shared expert, re-sharded from TP over all ranks to TP within groups
# ---------------------------------------------------------------------------


def merge_gate_up(
    weight: torch.Tensor, scale: torch.Tensor, shards: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """`shards` merged gate/up TP shards side by side (`[K, shards * 2i]`, each
    `[gate_r | up_r]`) as one `[gate | up]` projection, with their per-channel
    scales (`[shards * 2i]`). Each shard's scales already cover the whole
    contraction, so this is a column permutation: no value changes."""
    k, n = weight.shape
    i = n // (2 * shards)
    if n != 2 * shards * i or scale.shape != (n,):
        raise ValueError(f"gate_up shards {tuple(weight.shape)} / {tuple(scale.shape)}")
    order = torch.arange(n, device=weight.device).view(shards, 2, i).transpose(0, 1)
    order = order.reshape(n)
    return weight.index_select(1, order), scale.index_select(0, order)


def member_slice(
    w13: torch.Tensor,
    s13: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    *,
    tp: int,
    group: int,
    member: int,
) -> tuple[torch.Tensor, ...]:
    """Group member `member`'s share of a shared expert, from all `tp` TP
    shards gathered: `w13 [K, tp * 2i]` / `s13 [tp * 2i]` (shard r's columns
    `[gate_r | up_r]`), `w2 [tp * i, N]` / `s2 [tp, N]` (shard r's rows, scaled
    per shard). The member takes `tp / group` consecutive shards: its merged
    `[gate | up]` columns and the matching down rows, block-scaled with one
    block per shard, so every fp8 value and scale is kept as it was."""
    per = tp // group
    if per * group != tp or not 0 <= member < group:
        raise ValueError(f"tp={tp} does not split into {group} groups ({member})")
    cols = w13.shape[1] // tp
    lo, hi = member * per, (member + 1) * per
    gu, gu_s = merge_gate_up(
        w13[:, lo * cols : hi * cols], s13[lo * cols : hi * cols], per
    )
    rows = w2.shape[0] // tp
    return gu, gu_s, w2[lo * rows : hi * rows], s2[lo:hi]


def _gather(t: torch.Tensor, dim: int) -> torch.Tensor:
    """All-gather over the TP group at load. fp8 is gathered (and sliced by
    the caller) as bf16, which holds every e4m3 value exactly."""
    if t.dtype == torch.float8_e4m3fn:
        t = t.to(torch.bfloat16)
    return get_tp_group().all_gather(t, dim)


def _is_per_channel_fp8(linear: Any) -> bool:
    w = getattr(linear, "weight", None)
    s = getattr(linear, "weight_scale", None)
    return (
        w is not None
        and w.dtype == torch.float8_e4m3fn
        and s is not None
        and s.dim() == 1
        and bool(getattr(linear, WEIGHT_FLIPPED_ATTR, False))
        and not bool(getattr(linear, KEEP_VLLM_LAYOUT_ATTR, False))
        and getattr(linear, "bias", None) is None
    )


def _per_channel_fp8(linear: Any) -> tuple[torch.Tensor, torch.Tensor]:
    w, s = linear.weight, getattr(linear, "weight_scale", None)
    if not _is_per_channel_fp8(linear):
        raise NotImplementedError(
            "Sequence-parallel MoE prefill re-shards a shared expert held as "
            "per-channel fp8 (k, n) linears without bias; got "
            f"{w.dtype} weight {tuple(w.shape)}, scale "
            f"{None if s is None else tuple(s.shape)}"
        )
    assert s is not None
    return w.data, s.data


def _group_mlp_block(
    mlp_block: Any, tp: int, rank: int, group: int
) -> tuple[torch.Tensor, ...]:
    """This rank's member slice of a `DeepseekV2MLP` block (`w13, s13, w2, s2`)."""
    w13, s13 = _per_channel_fp8(mlp_block.gate_up_proj)
    w2, s2 = _per_channel_fp8(mlp_block.down_proj)
    return member_slice(
        _gather(w13, 1),
        _gather(s13, 0),
        _gather(w2, 0),
        _gather(s2[None], 0),
        tp=tp,
        group=group,
        member=rank % group,
    )


def _group_shared_expert(
    mlp: Any, tp: int, rank: int, group: int
) -> tuple[torch.Tensor, ...]:
    """Return this rank's member slice of the layer's shared expert."""
    return _group_mlp_block(mlp.shared_experts, tp, rank, group)


def _stack_mlp_parts(
    backbone: Any,
    mlps: list[Any],
    parts: list[tuple[torch.Tensor, ...]],
    *,
    stack_names: tuple[str, str, str, str],
    attr: str,
    index_attr: str,
) -> None:
    stacked = []
    dtypes = (torch.float8_e4m3fn, torch.float32, torch.float8_e4m3fn, torch.float32)
    for name, column, dtype in zip(stack_names, zip(*parts), dtypes):
        # A fresh buffer, not a view: torch_tpu hands the executable the buffer
        # behind a tensor, and peer-replayed programs then disagree with this
        # rank's argument shapes. Stacked before the fp8 cast (exact).
        t = fresh_buffer(torch.stack(column).to(dtype))
        backbone.register_buffer(name, t, persistent=False)
        stacked.append(t)
    for i, mlp in enumerate(mlps):
        setattr(mlp, attr, tuple(stacked))
        setattr(mlp, index_attr, i)


def _stack_shared_experts(
    backbone: Any, mlps: list[Any], parts: list[tuple[torch.Tensor, ...]]
) -> None:
    _stack_mlp_parts(
        backbone,
        mlps,
        parts,
        stack_names=SHARED_STACK,
        attr=SHARED_ATTR,
        index_attr=SHARED_INDEX,
    )


_SHARED_OPS: dict[tuple, Any] = {}


def _build_shared_op(group: int) -> Any:
    """The grouped shared expert as one SPMD op over the EP mesh (built at
    load, outside Dynamo)."""
    key = (group,)
    if key in _SHARED_OPS:
        return _SHARED_OPS[key]
    mesh = build_ep_mesh()
    # jax_op hands fn the global arrays: the per-rank body goes under shard_map.
    sharded = grouped_shared_expert_sharded(mesh, EP_AXIS_NAME, group)

    def fn(
        x: jax.Array, w13: jax.Array, s13: jax.Array, w2: jax.Array, s2: jax.Array
    ) -> jax.Array:
        return sharded(x, w13, s13, w2, s2)

    spec = PartitionSpec(EP_AXIS_NAME)
    assert pallas is not None
    # pyrefly: ignore[no-matching-overload]
    op = pallas.jax_op(
        f"pallas::moe_prefill_sp_shared_g{group}",
        fn,
        mesh=mesh,
        input_partition_specs=(spec,) * 5,
        output_partition_specs=spec,
    )
    op.register_fake(lambda x, *_args, **_kw: torch.empty_like(x))
    _SHARED_OPS[key] = op
    return op


def _shared_expert(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    i = getattr(mlp, SHARED_INDEX)
    return getattr(mlp, SHARED_OP)(x, *(t[i] for t in getattr(mlp, SHARED_ATTR)))


def _dense_mlp(mlp: Any, x: torch.Tensor) -> torch.Tensor:
    i = getattr(mlp, DENSE_INDEX)
    return getattr(mlp, DENSE_OP)(x, *(t[i] for t in getattr(mlp, DENSE_ATTR)))


def _embed_reduce_scatter(embed: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """Vocab-parallel embedding reduced onto sequence-parallel rows `[T / tp, H]`.

    Reads the same shard-bound buffers and weight in the same order as
    `VocabParallelEmbedding.forward` so decode and prefill buckets lift
    identical arguments for bucket hand-over.
    """
    masked_input, input_mask = vpe.get_masked_input_and_mask(
        input_ids,
        getattr(
            embed,
            "org_vocab_start_index",
            embed.shard_indices.org_vocab_start_index,
        ),
        getattr(
            embed,
            "org_vocab_end_index",
            embed.shard_indices.org_vocab_end_index,
        ),
        getattr(
            embed,
            "num_org_vocab_padding",
            embed.shard_indices.num_org_vocab_padding,
        ),
        getattr(
            embed,
            "added_vocab_start_index",
            embed.shard_indices.added_vocab_start_index,
        ),
        getattr(
            embed,
            "added_vocab_end_index",
            embed.shard_indices.added_vocab_end_index,
        ),
    )
    output_parallel = embed.quant_method.embedding(embed, masked_input.long())
    output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    return tensor_model_parallel_reduce_scatter(output_parallel, 0)


# ---------------------------------------------------------------------------
# Forwards
# ---------------------------------------------------------------------------


class _OProjReduceScatter:
    """The MLA op's `prefill_group` for an input of all rows: o_proj's partial
    sums, reduce-scattered to each rank's rows instead of all-reduced."""

    def __init__(self, zigzag_bands: int = 1) -> None:
        self.zigzag_bands = zigzag_bands

    def project_reduce_scatter(self, x: torch.Tensor, o_proj: Any) -> torch.Tensor:
        partial = o_proj.quant_method.apply(o_proj, x)
        if self.zigzag_bands > 1:
            tp = get_tensor_model_parallel_world_size()
            partial = _to_zigzag_rank_major(partial, tp, self.zigzag_bands)
        return tensor_model_parallel_reduce_scatter(partial, 0)


class _GatherProjections(_OProjReduceScatter):
    """The MLA op's `prefill_group` for an input of this rank's rows: the
    row-local input projections (fused_qkv_a and, on an indexer layer, the
    indexer's wk/weights) run on the rank's rows and their outputs, 2624 (+160)
    columns, are all-gathered in place of the 6144-column hidden rows."""

    @staticmethod
    def project_gather(
        mla: Any,
        x: torch.Tensor,
        num_tokens: int,
        *,
        positions_local: torch.Tensor | None = None,
        zigzag_bands: int = 1,
        owned_k_resolver: Callable[
            [Any, torch.Tensor, int], tuple[torch.Tensor, torch.Tensor] | None
        ]
        | None = None,
    ) -> tuple[
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        qkv = mla.fused_qkv_a_proj(x)[0]
        indexer = mla.indexer if not mla.skip_topk else None
        tp_op = getattr(getattr(mla, "mla_attn", None), "sparse_mla_tp_op", None)
        idx_qown = indexer is None or getattr(
            getattr(getattr(indexer, "indexer_op", None), "topk_op_tp", None),
            "q_own_rows",
            False,
        )
        rows_local = x.shape[0]
        if (
            tp_op is not None
            and getattr(tp_op, "q_replicated", False)
            and getattr(tp_op, "rank", None) is not None
            and dcp_token_parallel_op.applies(num_tokens)
            and not getattr(mla, "gate_is_fused", False)
            and getattr(mla, "kimi_preprocess", None) is None
            and idx_qown
            and mla.q_lora_rank is not None
        ):
            tp = tp_op.layout.dcp_size
            s = tp_op.layout.split_rows
            q_lora = int(mla.q_lora_rank)
            if (
                num_tokens == tp * rows_local
                and get_tensor_model_parallel_world_size() == tp
                and 0 < s <= rows_local
                and (s * q_lora) % rows_local == 0
            ):
                kv_dim = qkv.shape[-1] - q_lora
                q_c_local, kv_lora_local = qkv.split([q_lora, kv_dim], dim=-1)
                cols_qc = (s * q_lora) // rows_local
                use_sharded_kv = (
                    glm_dsa_v7x_active()
                    and positions_local is not None
                    and getattr(mla, "is_sparse", False)
                    and getattr(mla, "g_proj", None) is None
                    and getattr(tp_op, "serves_sharded_kv", False)
                    and getattr(mla, "o_proj", None) is not None
                    and dcp_token_parallel_op.serves_sequence_row_oproj(mla.o_proj)
                )
                qc_head_shift_op = getattr(tp_op, "qc_head_shift_op", None)
                if use_sharded_kv and qc_head_shift_op is not None:
                    assert positions_local is not None
                    if indexer is None:
                        q_c_qrows = qc_head_shift_op(q_c_local, tp_op.rank)
                        return (q_c_qrows, kv_lora_local, positions_local), None
                    gather_owned_k_op = getattr(tp_op, "gather_owned_k_op", None)
                    if gather_owned_k_op is not None and owned_k_resolver is not None:
                        kw_local = indexer.wk_weights_proj(x)[0]
                        k_dim = int(indexer.head_dim)
                        k_local, w_local = kw_local.split(
                            [k_dim, kw_local.shape[-1] - k_dim], dim=-1
                        )
                        owned_plan = owned_k_resolver(mla, k_local, num_tokens)
                        if owned_plan is not None:
                            raw_own, pos_own = owned_plan
                            qcw_qrows = qc_head_shift_op(
                                torch.cat([q_c_local, w_local], dim=-1), tp_op.rank
                            )
                            q_c_qrows, w_qrows = qcw_qrows.split(
                                [q_lora, w_local.shape[-1]], dim=-1
                            )
                            k_own = gather_owned_k_op(k_local, raw_own)
                            return (
                                (q_c_qrows, kv_lora_local, positions_local),
                                (k_own, w_qrows, pos_own),
                            )
                qc_head_packed = q_c_local[:s].reshape(rows_local, cols_qc)
                parts = (
                    [qc_head_packed]
                    if use_sharded_kv
                    else [kv_lora_local, qc_head_packed]
                )
                if indexer is not None:
                    parts.append(indexer.wk_weights_proj(x)[0])
                gathered = _all_rows(
                    parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1),
                    num_tokens,
                )
                kv_lora = kv_lora_local
                if use_sharded_kv:
                    all_heads = gathered[:, :cols_qc].reshape(tp, s, q_lora)
                    kw = (
                        _from_zigzag_rank_major(
                            gathered[:, cols_qc:], num_tokens, tp, zigzag_bands
                        )
                        if indexer is not None
                        else None
                    )
                else:
                    kv_lora = _from_zigzag_rank_major(
                        gathered[:, :kv_dim], num_tokens, tp, zigzag_bands
                    )
                    all_heads = gathered[:, kv_dim : kv_dim + cols_qc].reshape(
                        tp, s, q_lora
                    )
                    kw = (
                        _from_zigzag_rank_major(
                            gathered[:, kv_dim + cols_qc :],
                            num_tokens,
                            tp,
                            zigzag_bands,
                        )
                        if indexer is not None
                        else None
                    )
                rank = tp_op.rank.view(1)
                if zigzag_bands > 1:
                    pad = q_c_local[-1:].expand(s, -1)
                    r0_tail = torch.cat([q_c_local[s:], pad], dim=0)
                    tail = torch.where(rank.view(1, 1) == 0, r0_tail, q_c_local)
                    q_c_qrows = torch.cat([all_heads[0], tail], dim=0)
                else:
                    next_rank = torch.clamp(rank + 1, max=tp - 1)
                    from_next = all_heads.index_select(0, next_rank).squeeze(0)
                    tail_s = torch.where(
                        rank.view(1, 1) < tp - 1,
                        from_next,
                        q_c_local[-1:].expand(s, -1),
                    )
                    q_c_qrows = torch.cat([all_heads[0], q_c_local[s:], tail_s], dim=0)
                if use_sharded_kv:
                    assert positions_local is not None
                    return (q_c_qrows, kv_lora_local, positions_local), kw
                return (q_c_qrows, kv_lora), kw

        parts = [qkv]
        if indexer is not None:
            parts.append(indexer.wk_weights_proj(x)[0])
        rows = _all_rows(torch.cat(parts, dim=-1), num_tokens)
        if zigzag_bands > 1:
            rows = _from_zigzag_rank_major(
                rows, num_tokens, get_tensor_model_parallel_world_size(), zigzag_bands
            )
        if indexer is None:
            return rows, None
        width = parts[0].shape[-1]
        return rows[:, :width], rows[:, width:]


class _ZigzagGatherProjections(_OProjReduceScatter):
    """Bound `_GatherProjections` carrying a non-unit `zigzag_bands` count."""

    def project_gather(
        self,
        mla: Any,
        x: torch.Tensor,
        num_tokens: int,
        *,
        positions_local: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        return _GatherProjections.project_gather(
            mla,
            x,
            num_tokens,
            positions_local=positions_local,
            zigzag_bands=self.zigzag_bands,
        )


class _ShardedKVGatherProjections(_OProjReduceScatter):
    """Per-step projection gatherer that slices local positions lazily."""

    def __init__(
        self, positions: torch.Tensor, tp: int, *, zigzag_bands: int = 1
    ) -> None:
        super().__init__(zigzag_bands=zigzag_bands)
        self._positions = positions
        self._tp = tp
        self._positions_local: torch.Tensor | None = None
        self._owned_k_plans: dict[
            tuple[int, int, int],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}

    def _resolve_owned_k_plan(
        self, mla: Any, k_local: torch.Tensor, num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        mla_attn = getattr(mla, "mla_attn", None)
        tp_op = getattr(mla_attn, "sparse_mla_tp_op", None)
        if tp_op is None:
            return None
        owned_rows_op = getattr(tp_op, "indexer_owned_rows_op", None)
        if owned_rows_op is None or getattr(tp_op, "rank", None) is None:
            return None
        indexer_op = getattr(getattr(mla, "indexer", None), "indexer_op", None)
        topk_op_tp = getattr(indexer_op, "topk_op_tp", None)
        if not getattr(topk_op_tp, "serves_sharded_k", False):
            return None
        unified_qb_op = getattr(mla_attn, "sparse_mla_unified_idx_qb_op", None)
        if unified_qb_op is not None and not getattr(
            unified_qb_op, "serves_sharded_k", False
        ):
            return None
        unified_tp_op = getattr(mla_attn, "sparse_mla_unified_tp_op", None)
        if unified_tp_op is not None and not getattr(
            unified_tp_op, "serves_sharded_k", False
        ):
            return None
        step_meta_fn = getattr(indexer_op, "live_cache_step_metadata", None)
        if not callable(step_meta_fn):
            return None
        step_meta = step_meta_fn()
        if step_meta is None:
            return None
        seq_lens, block_tables, query_start_loc = step_meta
        bound = owned_rows_bound_from_local(
            k_local.shape[0],
            num_seqs=seq_lens.shape[0],
            interleave_size=int(tp_op.layout.interleave_size),
        )
        if bound >= num_tokens:
            return None
        # Avoid int(num_tokens) in plan_key: num_tokens is symbolic under AOT
        # dynamic tracing, and each _ShardedKVGatherProjections instance serves
        # one step token count.
        plan_key = (
            id(seq_lens),
            id(block_tables),
            id(query_start_loc),
        )
        cached = self._owned_k_plans.get(plan_key)
        if cached is not None:
            return cached
        own_idx, raw_own = owned_rows_op(
            seq_lens,
            block_tables,
            query_start_loc,
            k_local[:, :1],
            tp_op.rank,
        )
        pos_own = self._positions.index_select(0, own_idx)
        plan = (raw_own, pos_own)
        self._owned_k_plans[plan_key] = plan
        return plan

    def project_gather(
        self,
        mla: Any,
        x: torch.Tensor,
        num_tokens: int,
    ) -> tuple[
        torch.Tensor
        | tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None,
    ]:
        # Avoid eager slicing in patched_model_forward: calling
        # fused_moe_ep.rank_operand() before layer 0's _retain_buffers(*keep)
        # binds the rank tensor to the module global instead of
        # layers[0]._tpu_sp_keep, causing ShapeSpecializationError on bucket handover.
        if self._positions_local is None:
            rank = fused_moe_ep.rank_operand()
            self._positions_local = (
                own_rows_zigzag(self._positions, rank, self._tp, self.zigzag_bands)
                if self.zigzag_bands > 1
                else own_rows(self._positions, rank, self._tp)
            )
        extra_kw: dict[str, Any] = {
            "owned_k_resolver": self._resolve_owned_k_plan,
        }
        if self.zigzag_bands > 1:
            extra_kw["zigzag_bands"] = self.zigzag_bands
        return _GatherProjections.project_gather(
            mla,
            x,
            num_tokens,
            positions_local=self._positions_local,
            **extra_kw,
        )


_O_PROJ_RS = _OProjReduceScatter()
_GATHER_PROJECTIONS = _GatherProjections()


def _check_divides(num_tokens: int) -> int:
    tp = get_tensor_model_parallel_world_size()
    torch._check(num_tokens % tp == 0)
    return tp


def _to_zigzag_rank_major(x: torch.Tensor, tp: int, num_bands: int) -> torch.Tensor:
    if num_bands <= 1:
        return x
    num_tokens = x.shape[0]
    band = num_tokens // (tp * num_bands)
    x_bands = x.view(num_bands, tp, band, *x.shape[1:])
    slices = [
        x_bands[k].flip(0) if k % 2 == 1 else x_bands[k] for k in range(num_bands)
    ]
    return torch.stack(slices, dim=1).reshape(num_tokens, *x.shape[1:])


def _from_zigzag_rank_major(
    gathered: torch.Tensor, num_tokens: int, tp: int, num_bands: int
) -> torch.Tensor:
    if num_bands <= 1:
        return gathered
    band = num_tokens // (tp * num_bands)
    g = gathered.view(tp, num_bands, band, *gathered.shape[1:])
    parts = [g[:, k].flip(0) if k % 2 == 1 else g[:, k] for k in range(num_bands)]
    return torch.cat(parts, dim=0).reshape(num_tokens, *gathered.shape[1:])


def zigzag_global_to_owner_and_local(
    indices: torch.Tensor, num_tokens: int, tp: int, num_bands: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global indices `[M]` to `(owner_rank, local_row)`."""
    safe_idx = indices.to(torch.int32).clamp(0, num_tokens - 1)
    rows_per_rank = num_tokens // tp
    if num_bands <= 1:
        owner_rank = torch.div(safe_idx, rows_per_rank, rounding_mode="floor")
        local_row = torch.remainder(safe_idx, rows_per_rank)
        return owner_rank.to(torch.int32), local_row.to(torch.int32)
    band = rows_per_rank // num_bands
    blk = torch.div(safe_idx, band, rounding_mode="floor")
    off = torch.remainder(safe_idx, band)
    band_idx = torch.div(blk, tp, rounding_mode="floor")
    slot = torch.remainder(blk, tp)
    odd_band = (band_idx % 2) == 1
    owner_rank = torch.where(odd_band, tp - 1 - slot, slot)
    local_row = band_idx * band + off
    return owner_rank.to(torch.int32), local_row.to(torch.int32)


def select_sp_hidden_states(
    local_hidden: torch.Tensor,
    indices: torch.Tensor,
    *,
    num_tokens: int,
    tp: int,
    zigzag_bands: int,
    rank_tensor: torch.Tensor | None = None,
    all_reduce_fn: Any = None,
) -> torch.Tensor:
    """Gather `[M, H]` rows from rank-local SP `local_hidden` `[T/tp, H]`.

    Falls back to a full all-gather + un-zigzag when `M >= T/tp` (e.g.
    prompt_logprobs over a whole prompt chunk). For typical sampling indices
    (`M <= max_num_reqs`), gathers local rows and runs a masked all-reduce so
    non-owning ranks contribute +0 and the single owner contributes x.
    """
    local_rows = local_hidden.shape[0]
    if indices.shape[0] >= local_rows:
        full = _all_rows(local_hidden, num_tokens)
        if zigzag_bands > 1:
            full = _from_zigzag_rank_major(full, num_tokens, tp, zigzag_bands)
        return torch.index_select(full, 0, indices)
    owner_rank, local_row = zigzag_global_to_owner_and_local(
        indices, num_tokens, tp, zigzag_bands
    )
    picked = torch.index_select(local_hidden, 0, local_row)
    rk = (
        (fused_moe_ep.rank_operand() if rank_tensor is None else rank_tensor)
        .reshape(())
        .to(owner_rank.dtype)
    )
    mask = (owner_rank == rk).unsqueeze(-1)
    masked = torch.where(mask, picked, torch.zeros_like(picked))
    reduce_fn = (
        tensor_model_parallel_all_reduce if all_reduce_fn is None else all_reduce_fn
    )
    return reduce_fn(masked)


def step_zigzag_bands_for_model(model: torch.nn.Module, num_tokens: int) -> int:
    backbone = getattr(model, "model", None)
    if backbone is None or not getattr(backbone, KEEP_ATTR, False):
        return 1
    tp = get_tensor_model_parallel_world_size()
    if num_tokens % tp != 0:
        return 1
    active_layers = list(
        islice(backbone.layers, backbone.start_layer, backbone.end_layer)
    )
    return _step_zigzag_bands(active_layers, num_tokens, tp, check_model_tag=False)


def is_final_ag_deferred_for_bucket(model: torch.nn.Module, num_tokens: int) -> bool:
    # Avoid calling applies(): callers run outside model_tag == "backbone"
    # during precompile and post-forward logits selection.
    if not (_SP_ENABLED and num_tokens > bucket_gate_edge(MIN_TOKENS)):
        return False
    backbone = getattr(model, "model", None)
    if backbone is None:
        return False
    return bool(
        getattr(backbone, KEEP_ATTR, False)
        and getattr(backbone, DEFER_FINAL_AG_ATTR, False)
        and get_tensor_model_parallel_world_size() > 1
    )


def _step_zigzag_bands(
    active_layers: list[Any],
    num_tokens: int,
    tp: int,
    *,
    check_model_tag: bool = True,
) -> int:
    if check_model_tag:
        if not dcp_token_parallel_op.applies(num_tokens):
            return 1
    elif not (
        dcp_token_parallel_op._state.enabled
        and not dcp_token_parallel_op._state.over_capacity
        and num_tokens > bucket_gate_edge(dcp_token_parallel_op.MIN_TOKENS)
    ):
        return 1
    found: int | None = None
    for layer in active_layers:
        if getattr(layer, KEEP_ATTR, None) is None:
            continue
        mla = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
        tp_op = getattr(getattr(mla, "mla_attn", None), "sparse_mla_tp_op", None)
        if tp_op is None:
            return 1
        layout = getattr(tp_op, "layout", None)
        if layout is None or layout.dcp_size != tp:
            return 1
        nb = int(layout.effective_zigzag_bands(num_tokens))
        if nb <= 1:
            return 1
        if found is None:
            found = nb
        elif found != nb:
            raise ValueError(
                f"inconsistent zigzag_bands across armed layers: {found} vs {nb}"
            )
    return found or 1


def own_rows_zigzag(
    x: torch.Tensor, rank: torch.Tensor, tp: int, num_bands: int
) -> torch.Tensor:
    """Rank `rank`'s `num_bands` alternating forward/reverse zigzag bands."""
    if num_bands <= 1:
        return own_rows(x, rank, tp)
    rows_per_rank = x.shape[0] // tp
    band = rows_per_rank // num_bands
    r = rank.reshape(())
    blocks = x.view(num_bands * tp, band, *x.shape[1:])
    idx = torch.stack(
        [k * tp + r if k % 2 == 0 else (k + 1) * tp - 1 - r for k in range(num_bands)]
    )
    return blocks.index_select(0, idx).reshape(rows_per_rank, *x.shape[1:])


def own_rows(x: torch.Tensor, rank: torch.Tensor, tp: int) -> torch.Tensor:
    """Rank `rank`'s contiguous `1/tp` of the rows of `x`. The rank is data
    (int32 `[1, 1]`), so every rank traces the same program."""
    rows_per_rank = x.shape[0] // tp
    rows = x.view(tp, rows_per_rank, *x.shape[1:])
    return rows.index_select(0, rank.view(1)).view(rows_per_rank, *x.shape[1:])


def _all_rows(x: torch.Tensor, num_tokens: int) -> torch.Tensor:
    # narrow, not [:n]: keeps the token dim the traced symbol itself.
    return tensor_model_parallel_all_gather(x, 0).narrow(0, 0, num_tokens)


def _zero_rank_dependency(tensor: torch.Tensor, rank: torch.Tensor) -> torch.Tensor:
    # Avoid omitting this zero-delta dependency: without tying rank to the
    # shared-expert output, XLA schedules shared_expert_blocked_down after
    # fused_ep_moe_v2 instead of overlapping it inside MoE's async FP8
    # all-gather. Reinterpreting low bits as int16->float32 keeps f - f == 0
    # for all bit patterns (including NaN/Inf) while surviving simplification.
    bits_f32 = tensor[:1, :1].reshape(1).view(torch.int16)[:1].to(torch.float32)
    return rank + (bits_f32 - bits_f32).reshape(rank.shape).to(rank.dtype)


def _moe(
    mlp: Any, x: torch.Tensor, num_tokens: int, tp: int, *, num_bands: int = 1
) -> torch.Tensor:
    """The MoE block on this rank's rows: routed (fused EP, own tokens) times
    the routed scale, plus the shared expert -- the runner's order."""
    experts = mlp.experts.routed_experts
    rank = fused_moe_ep.rank_operand()
    padding = token_padding._padding_state  # noqa: SLF001 -- runner singleton
    if _skip_padded_tokens() and padding is not None:
        raw_mask = padding.local_padding_mask.narrow(0, 0, num_tokens)
        pad_mask = (
            own_rows_zigzag(raw_mask, rank, tp, num_bands)
            if num_bands > 1
            else own_rows(raw_mask, rank, tp)
        )
        x = torch.where(pad_mask[:, None], torch.zeros_like(x), x)
        router_logits, _ = mlp.gate(x)
        router_logits = torch.where(pad_mask[:, None], float("nan"), router_logits)
    else:
        router_logits, _ = mlp.gate(x)
    shared = _shared_expert(mlp, x)
    routed_rank = _zero_rank_dependency(shared, rank)
    routed = fused_moe_ep.run_sequence_parallel_op(
        getattr(mlp, OP_ATTR),
        x,
        experts.w13_weight,
        experts.w2_weight,
        experts.w13_weight_scale_inv,
        experts.w2_weight_scale_inv,
        router_logits,
        routed_rank,
        experts.e_score_correction_bias,
    )
    scale = getattr(mlp, SCALE_ATTR)
    if scale != 1.0:
        routed = routed * scale
    return shared + routed


def _sequence_parallel_layer_forward(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None,
    gather_group: Any = _GATHER_PROJECTIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = positions.shape[0]
    tp = _check_divides(num_tokens)
    sharded_in = hidden_states.shape[0] != num_tokens
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    zigzag_bands = int(getattr(gather_group, "zigzag_bands", 1))
    group = (
        gather_group
        if sharded_in
        else (
            _O_PROJ_RS
            if zigzag_bands <= 1
            else _OProjReduceScatter(zigzag_bands=zigzag_bands)
        )
    )
    hidden_states = layer.self_attn.mla_attn(
        positions, hidden_states, llama_4_scaling, prefill_group=group
    )
    if not sharded_in:
        rk = fused_moe_ep.rank_operand()
        residual = (
            own_rows_zigzag(residual, rk, tp, zigzag_bands)
            if zigzag_bands > 1
            else own_rows(residual, rk, tp)
        )
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    if hasattr(layer.mlp, DENSE_OP):
        return _dense_mlp(layer.mlp, hidden_states), residual
    return _moe(
        layer.mlp, hidden_states, num_tokens, tp, num_bands=zigzag_bands
    ), residual


def _can_fuse_decode_post_attn_norm(
    layer: Any,
    hidden_states: torch.Tensor,
) -> bool:
    if (
        not glm_dsa_v7x_active()
        or getattr(layer, "use_sequence_parallel_moe", False)
        or getattr(layer, "use_mha", False)
        or hidden_states.dim() != 2
        or hidden_states.dtype != torch.bfloat16
    ):
        return False
    mla = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
    if (
        mla is None
        or getattr(getattr(mla, "mla_attn", None), "W_O_dcp_qrep", None) is not None
    ):
        return False
    o_proj = getattr(mla, "o_proj", None)
    if o_proj is None:
        return False
    return small_all_reduce_op.can_fuse_rmsnorm(
        hidden_states.shape[0],
        o_proj,
        getattr(layer, "post_attention_layernorm", None),
    )


def _decode_fused_post_attn_layer_forward(
    layer: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual)
    norm = layer.post_attention_layernorm
    hidden_states, residual = layer.self_attn.mla_attn(
        positions,
        hidden_states,
        llama_4_scaling,
        post_attn_norm=(residual, norm.weight, float(norm.variance_epsilon)),
    )
    hidden_states = layer.mlp(hidden_states)
    return hidden_states, residual


def patched_decoder_layer_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    llama_4_scaling: torch.Tensor | None = None,
    *,
    gather_group: Any = _GATHER_PROJECTIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    keep = getattr(self, KEEP_ATTR, None)
    if keep is None:
        return UPSTREAM_DECODER_LAYER_FORWARD(
            self, positions, hidden_states, residual, llama_4_scaling
        )
    hidden_states = _retain_buffers(hidden_states, *keep)
    if not applies(positions.shape[0]):
        if _can_fuse_decode_post_attn_norm(self, hidden_states):
            return _decode_fused_post_attn_layer_forward(
                self, positions, hidden_states, residual, llama_4_scaling
            )
        return UPSTREAM_DECODER_LAYER_FORWARD(
            self, positions, hidden_states, residual, llama_4_scaling
        )
    return _sequence_parallel_layer_forward(
        self, positions, hidden_states, residual, llama_4_scaling, gather_group
    )


def patched_model_forward(
    self,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: Any = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = positions.shape[0]
    if not applies(num_tokens) or not getattr(self, KEEP_ATTR, False):
        return UPSTREAM_MODEL_FORWARD(
            self, input_ids, positions, intermediate_tensors, inputs_embeds
        )
    # Avoid leaving the lifted `num_tokens` SymInt symbolic inside AOT export:
    # when all layers take sharded KV/indexer paths, no later op equates the
    # lifted token symbol to a concrete tensor dim, leaving `_embed_reduce_scatter`'s
    # `aten.empty([floordiv, H])` symbolic and failing static lowering.
    positions = positions.view(num_tokens, *positions.shape[1:])
    active_layers = list(islice(self.layers, self.start_layer, self.end_layer))
    tp = _check_divides(num_tokens)
    zigzag_bands = _step_zigzag_bands(active_layers, num_tokens, tp)
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    elif (
        getattr(self, EMBED_RS_ATTR, False)
        and input_ids is not None
        and input_ids.ndim == 1
    ):
        input_ids = input_ids.view(num_tokens)
        full_table = replicated_embedding.full_weight(self.embed_tokens)
        if full_table is None:
            embed_ids = (
                _to_zigzag_rank_major(input_ids, tp, zigzag_bands)
                if zigzag_bands > 1
                else input_ids
            )
            hidden_states = _embed_reduce_scatter(self.embed_tokens, embed_ids)
        else:
            # Avoid calling fused_moe_ep.rank_operand() before layer 0's
            # _retain_buffers(*keep): it binds the rank tensor to the module
            # global instead of layers[0]._tpu_sp_keep, and the decode trace
            # then names it differently (ShapeSpecializationError on handover).
            first_keep = getattr(self.layers[self.start_layer], KEEP_ATTR)
            input_ids = _retain_buffers(input_ids, *first_keep)
            rk = fused_moe_ep.rank_operand()
            own_ids = (
                own_rows_zigzag(input_ids, rk, tp, zigzag_bands)
                if zigzag_bands > 1
                else own_rows(input_ids, rk, tp)
            )
            hidden_states = replicated_embedding.lookup(full_table, own_ids)
            # The decode buckets keep the vocab-parallel lookup: lift what it
            # reads so both traces hand buckets over with the same inputs.
            hidden_states = _retain_buffers(
                hidden_states, *replicated_embedding.shard_operands(self.embed_tokens)
            )
    else:
        hidden_states = self.embed_input_ids(input_ids)
    # Avoid ShapeSpecializationError on warmup bucket handover: when
    # embed_tokens and all leading dense layers take SP collectives, no
    # all-reduce runs on the prefill branch to lift small_all_reduce_op._RANK.
    hidden_states = small_all_reduce_op.retain(hidden_states)
    gather_group: Any = (
        _GATHER_PROJECTIONS
        if zigzag_bands <= 1
        else _ZigzagGatherProjections(zigzag_bands=zigzag_bands)
    )
    if glm_dsa_v7x_active() and any(
        getattr(layer, KEEP_ATTR, None) is not None for layer in active_layers
    ):
        gather_group = _ShardedKVGatherProjections(
            positions, tp, zigzag_bands=zigzag_bands
        )
    residual = None
    for layer in active_layers:
        if getattr(layer, KEEP_ATTR, None) is not None:
            hidden_states, residual = layer(
                positions, hidden_states, residual, None, gather_group=gather_group
            )
        else:
            hidden_states, residual = layer(positions, hidden_states, residual, None)
    hidden_states, _ = self.norm(hidden_states, residual)
    if hidden_states.shape[0] != num_tokens and not getattr(
        self, DEFER_FINAL_AG_ATTR, False
    ):
        hidden_states = _all_rows(hidden_states, num_tokens)
        if zigzag_bands > 1:
            hidden_states = _from_zigzag_rank_major(
                hidden_states, num_tokens, tp, zigzag_bands
            )
    return hidden_states


def _config_refusal_reasons(vllm_config: Any) -> list[str]:
    parallel = vllm_config.parallel_config
    tp = int(parallel.tensor_parallel_size)
    comp_cfg = getattr(vllm_config, "compilation_config", None)
    sizes = getattr(comp_cfg, "compile_sizes", None) if comp_cfg is not None else None
    gate = bucket_gate_edge(MIN_TOKENS)
    bad_sizes = bool(
        sizes and tp > 1 and any(int(s) > gate and int(s) % tp != 0 for s in sizes)
    )
    refusals = {
        "needs --enable-expert-parallel": not parallel.enable_expert_parallel,
        "needs data_parallel_size 1": parallel.data_parallel_size != 1,
        "needs pipeline_parallel_size 1": parallel.pipeline_parallel_size != 1,
        "needs prefill_context_parallel_size 1": (
            parallel.prefill_context_parallel_size != 1
        ),
        "needs tensor_parallel_size > 1": tp < 2,
        "needs compile_sizes > 512 divisible by tp": bad_sizes,
        "needs a rank-ordered EP mesh": not ep_mesh_is_rank_ordered(parallel),
        "needs a bfloat16 model": vllm_config.model_config.dtype != torch.bfloat16,
        "does not take speculative decoding": vllm_config.speculative_config
        is not None,
        "does not take TPU_MOE_SEQUENCE_PARALLEL": bool(envs.TPU_MOE_SEQUENCE_PARALLEL),
        "does not take USE_MOE_FUSED_EP_KERNEL": bool(envs.USE_MOE_FUSED_EP_KERNEL),
    }
    return [why for why, hit in refusals.items() if hit]


def check_config(vllm_config: Any) -> None:
    """Raise ``NotImplementedError`` if ``vllm_config`` cannot be armed."""
    bad = _config_refusal_reasons(vllm_config)
    if bad:
        raise NotImplementedError("moe_prefill_sp " + "; ".join(bad) + ".")


def _backbone(model: torch.nn.Module) -> Any:
    inner = getattr(model, "model", None)
    if inner is None or not isinstance(inner, deepseek_v2.DeepseekV2Model):
        raise NotImplementedError(
            "moe_prefill_sp requires DeepseekV2Model backbone; "
            f"got {type(inner).__name__}"
        )
    if getattr(inner.config, "llama_4_scaling", None) is not None:
        raise NotImplementedError("llama_4_scaling is not threaded through the split")
    if inner.aux_hidden_state_layers:
        raise NotImplementedError(
            "aux_hidden_state_layers is not threaded through the split"
        )
    return inner


def _keep_list(mlp: torch.nn.Module) -> list[torch.Tensor]:
    seen: dict[int, torch.Tensor] = {}
    pad_mask = (
        token_padding._padding_state.local_padding_mask
        if token_padding._padding_state is not None
        else None
    )
    extra = (fused_moe_ep.rank_operand(), pad_mask)
    for t in (*mlp.parameters(), *mlp.buffers(), *getattr(mlp, SHARED_ATTR), *extra):
        if t is not None:
            seen.setdefault(id(t), t)
    return list(seen.values())


def _keep_list_dense(mlp: torch.nn.Module) -> list[torch.Tensor]:
    """Buffers lifted by either branch of an armed dense decoder layer."""
    seen: dict[int, torch.Tensor] = {}
    extra = (fused_moe_ep.rank_operand(),)
    for t in (*mlp.parameters(), *mlp.buffers(), *getattr(mlp, DENSE_ATTR), *extra):
        if t is not None:
            seen.setdefault(id(t), t)
    return list(seen.values())


def _dense_group_size(tp: int, shared_group: int) -> int:
    return 2 if (shared_group >= 2 and tp % 2 == 0) else 1


def _can_arm_dense_layer(layer: Any) -> bool:
    if not isinstance(layer.mlp, deepseek_v2.DeepseekV2MLP):
        return False
    mla = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
    if (
        mla is None
        or "prefill_group" not in inspect.signature(type(mla).forward).parameters
        or mla.kimi_preprocess is not None
        or mla.q_lora_rank is None
        or mla.g_proj is not None
    ):
        return False
    return _is_per_channel_fp8(layer.mlp.gate_up_proj) and _is_per_channel_fp8(
        layer.mlp.down_proj
    )


def _can_embed_reduce_scatter(backbone: Any, tp: int) -> bool:
    embed = getattr(backbone, "embed_tokens", None)
    return (
        tp > 1
        and isinstance(embed, vpe.VocabParallelEmbedding)
        and getattr(embed, "tp_size", 1) == tp
        and bool(
            getattr(
                vpe.VocabParallelEmbedding, "_tpu_vocab_parallel_embedding_patch", False
            )
        )
        and not bool(getattr(embed, "use_fused_embedding", False))
        and hasattr(embed, "quant_method")
    )


def _query_min_free_hbm_per_core(vllm_config: Any) -> int:
    local_free: int | None = None
    try:
        free_b, total_b = torch.accelerator.get_memory_info()
        cache_cfg = vllm_config.cache_config
        raw_util = cache_cfg.gpu_memory_utilization if cache_cfg is not None else None
        util = 1.0 if raw_util is None else float(raw_util)
        reserve = max(0, int(total_b * (1.0 - util)))
        local_free = max(0, int(free_b) - reserve)
    except (RuntimeError, AttributeError, TypeError, ValueError):
        local_free = None
    try:
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)
    except (RuntimeError, AssertionError, AttributeError):
        return 0
    if tp_size <= 1:
        return 0 if local_free is None else local_free
    cpu_group = getattr(tp_group, "cpu_group", None)
    if cpu_group is None:
        return 0
    try:
        seen: list[int | None] = [None] * tp_size
        torch.distributed.all_gather_object(seen, local_free, group=cpu_group)
    except RuntimeError:
        return 0
    if any(v is None for v in seen):
        return 0
    return min(int(v) for v in seen if v is not None)


def _bind_instance_forwards(
    backbone: Any, armed_layers: list[Any], vllm_config: Any
) -> None:
    for layer in armed_layers:
        layer.forward = types.MethodType(patched_decoder_layer_forward, layer)
    backbone.forward = types.MethodType(patched_model_forward, backbone)
    from vllm.compilation.wrapper import (
        TorchCompileWithNoGuardsWrapper,
        reset_compile_wrapper,
    )
    from vllm.config import set_current_vllm_config

    if isinstance(backbone, TorchCompileWithNoGuardsWrapper) and not getattr(
        backbone, "do_not_compile", True
    ):
        backbone.cleanup()
        comp_cfg = vllm_config.compilation_config
        saved = (comp_cfg.cache_dir, comp_cfg.local_cache_dir)
        with set_current_vllm_config(vllm_config):
            reset_compile_wrapper(backbone)  # type: ignore[arg-type]
        comp_cfg.cache_dir, comp_cfg.local_cache_dir = saved


def _allocate_mlp_buffers(
    backbone: Any,
    first_mlp_block: Any,
    n_layers: int,
    tp: int,
    group: int,
    stack_names: tuple[str, str, str, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w13, _ = _per_channel_fp8(first_mlp_block.gate_up_proj)
    w2, _ = _per_channel_fp8(first_mlp_block.down_proj)
    per = tp // group
    hidden, two_fs = w13.shape
    fs = w2.shape[0]
    device = w13.device
    stacked = (
        torch.zeros((n_layers, hidden, per * two_fs), dtype=w13.dtype, device=device),
        torch.zeros((n_layers, per * two_fs), dtype=torch.float32, device=device),
        torch.zeros((n_layers, per * fs, hidden), dtype=w2.dtype, device=device),
        torch.zeros((n_layers, per, hidden), dtype=torch.float32, device=device),
    )
    for name, buf in zip(stack_names, stacked):
        backbone.register_buffer(name, buf, persistent=False)
    return stacked


def _populate_mlp_layer(
    stacked: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    layer_idx: int,
    mlp_owner: Any,
    mlp_block: Any,
    tp: int,
    rank: int,
    group: int,
    *,
    attr: str,
    index_attr: str,
) -> None:
    parts = _group_mlp_block(mlp_block, tp, rank, group)
    for dst, part in zip(stacked, parts):
        dst[layer_idx].copy_(part.to(dst.dtype))
    del parts
    if stacked[0].device.type == "tpu":
        synchronize_tensors(stacked)
    setattr(mlp_owner, attr, stacked)
    setattr(mlp_owner, index_attr, layer_idx)


def arm(
    model: torch.nn.Module,
    vllm_config: Any,
    *,
    shared_group: int | None = None,
    defer_final_ag: bool = False,
) -> int:
    """Arm MoE and eligible leading dense layers for sequence-parallel prefill."""
    global _SP_ENABLED
    check_config(vllm_config)
    backbone = _backbone(model)
    tp = get_tensor_model_parallel_world_size()
    if torch.distributed.get_world_size() != tp:
        raise NotImplementedError("the TP group must cover the whole world")
    layers = list(islice(backbone.layers, backbone.start_layer, backbone.end_layer))
    moe = [isinstance(layer.mlp, deepseek_v2.DeepseekV2MoE) for layer in layers]
    if not any(moe):
        raise NotImplementedError("backbone contains no DeepseekV2MoE layers")
    first = moe.index(True)
    if not all(moe[first:]):
        raise NotImplementedError(
            "moe_prefill_sp expects every layer after the first MoE layer to be MoE"
        )
    group = (
        int(shared_group)
        if shared_group is not None
        else select_sp_shared_group(
            tp_size=tp,
            max_model_len=int(vllm_config.model_config.max_model_len),
            free_hbm_per_core_bytes=_query_min_free_hbm_per_core(vllm_config),
        )
    )
    if group < 1 or tp % group != 0:
        raise ValueError(f"shared_group={group} must divide tp={tp}")
    moe_layers = layers[first:]
    for layer in moe_layers:
        mla = getattr(layer.self_attn, "mla_attn", None)
        if (
            mla is None
            or "prefill_group" not in inspect.signature(type(mla).forward).parameters
        ):
            raise NotImplementedError(
                f"moe_prefill_sp needs the TPU MLA prefill_group hook; got {type(mla)}"
            )
        if (
            mla.kimi_preprocess is not None
            or mla.q_lora_rank is None
            or mla.g_proj is not None
        ):
            raise NotImplementedError(
                "moe_prefill_sp requires the q_lora MLA path without fused "
                "preprocessing or g_proj"
            )
        if layer.mlp.shared_experts is None:
            raise NotImplementedError("moe_prefill_sp expects a separate shared expert")
        if float(layer.mlp.experts.routed_experts.routed_scaling_factor) != 1.0:
            raise NotImplementedError("moe_prefill_sp applies the routed scale itself")

    stacked = _allocate_mlp_buffers(
        backbone,
        moe_layers[0].mlp.shared_experts,
        len(moe_layers),
        tp,
        group,
        SHARED_STACK,
    )
    tp_rank = get_tp_group().rank_in_group
    shared_op = _build_shared_op(group)
    for layer_idx, layer in enumerate(moe_layers):
        mlp = layer.mlp
        experts = mlp.experts.routed_experts
        op = fused_moe_ep.build_sequence_parallel_op(
            experts,
            topk=experts.moe_config.experts_per_token,
            renormalize=experts.renormalize,
            activation=experts.quant_method._tpu_activation_str,
        )
        if ep_mesh_index() != tp_rank:
            raise NotImplementedError(
                "EP mesh index must match TP rank so the rank operand aligns with "
                "residual row slices"
            )
        setattr(mlp, OP_ATTR, op)
        setattr(mlp, SCALE_ATTR, float(mlp.routed_scaling_factor))
        _populate_mlp_layer(
            stacked,
            layer_idx,
            mlp,
            mlp.shared_experts,
            tp,
            tp_rank,
            group,
            attr=SHARED_ATTR,
            index_attr=SHARED_INDEX,
        )
        setattr(mlp, SHARED_OP, shared_op)

    stacked = tuple(fresh_buffer(buf) for buf in stacked)  # type: ignore[assignment]
    for name, buf in zip(SHARED_STACK, stacked):
        backbone.register_buffer(name, buf, persistent=False)
    for layer in moe_layers:
        setattr(layer.mlp, SHARED_ATTR, stacked)
        setattr(layer, KEEP_ATTR, _keep_list(layer.mlp))
        if getattr(layer, "post_attention_layernorm", None) is not None:
            small_all_reduce_op.prepare_rmsnorm_op(
                float(layer.post_attention_layernorm.variance_epsilon)
            )

    dense_armed = 0
    dense_group = _dense_group_size(tp, group)
    dense_layers = layers[:first]
    armed_layers: list[Any] = []
    if dense_layers and all(_can_arm_dense_layer(layer) for layer in dense_layers):
        dense_stacked = _allocate_mlp_buffers(
            backbone,
            dense_layers[0].mlp,
            len(dense_layers),
            tp,
            dense_group,
            DENSE_STACK,
        )
        dense_op = _build_shared_op(dense_group)
        for layer_idx, layer in enumerate(dense_layers):
            _populate_mlp_layer(
                dense_stacked,
                layer_idx,
                layer.mlp,
                layer.mlp,
                tp,
                tp_rank,
                dense_group,
                attr=DENSE_ATTR,
                index_attr=DENSE_INDEX,
            )
            rmsnorm.disarm_post_attention_prequant(layer)
            setattr(layer.mlp, DENSE_OP, dense_op)
        dense_stacked = tuple(fresh_buffer(buf) for buf in dense_stacked)  # type: ignore[assignment]
        for name, buf in zip(DENSE_STACK, dense_stacked):
            backbone.register_buffer(name, buf, persistent=False)
        for layer in dense_layers:
            setattr(layer.mlp, DENSE_ATTR, dense_stacked)
            setattr(layer, KEEP_ATTR, _keep_list_dense(layer.mlp))
            if getattr(layer, "post_attention_layernorm", None) is not None:
                small_all_reduce_op.prepare_rmsnorm_op(
                    float(layer.post_attention_layernorm.variance_epsilon)
                )
        dense_armed = len(dense_layers)
        armed_layers.extend(dense_layers)
    armed_layers.extend(moe_layers)

    embed_rs = (first == 0 or dense_armed == first) and _can_embed_reduce_scatter(
        backbone, tp
    )
    setattr(backbone, EMBED_RS_ATTR, embed_rs)
    embed_replicated = (
        embed_rs
        and glm_dsa_v7x_active()
        and replicated_embedding.arm_if_eligible(
            backbone.embed_tokens,
            tp,
            functools.partial(_gather, dim=0),
            decode=True,
        )
    )
    setattr(backbone, DEFER_FINAL_AG_ATTR, bool(defer_final_ag))
    setattr(backbone, KEEP_ATTR, True)
    _bind_instance_forwards(backbone, armed_layers, vllm_config)
    _SP_ENABLED = True
    n = len(moe_layers)
    logger.info(
        "moe_prefill_sp: armed %d MoE layers (layers %d..%d) and %d dense layers "
        "(shared_group=%d, dense_group=%d, embed_rs=%s, embed_replicated=%s, "
        "defer_final_ag=%s), buckets > %d tokens, tp=%d",
        n,
        backbone.start_layer + first,
        backbone.end_layer - 1,
        dense_armed,
        group,
        dense_group,
        embed_rs,
        embed_replicated,
        bool(defer_final_ag),
        bucket_gate_edge(MIN_TOKENS),
        tp,
    )
    return n


def maybe_arm(
    model: torch.nn.Module,
    vllm_config: Any,
    *,
    defer_final_ag: bool = False,
) -> int:
    """Arm sequence-parallel MoE prefill strictly for eligible GLM-5 DSA runs."""
    if not glm_dsa_v7x_active(vllm_config.model_config):
        return 0
    bad = _config_refusal_reasons(vllm_config)
    if bad:
        logger.info_once("Skipping moe_prefill_sp (%s).", "; ".join(bad))
        return 0
    if not torch.distributed.is_initialized():
        return 0
    if torch.distributed.get_world_size() != get_tensor_model_parallel_world_size():
        return 0
    order = ep_rank_order()
    if order is not None and tuple(order) != tuple(range(len(order))):
        logger.info_once(
            "Skipping moe_prefill_sp (EP mesh is not rank-ordered: %s).", order
        )
        return 0
    inner = getattr(model, "model", None)
    if not isinstance(inner, deepseek_v2.DeepseekV2Model):
        return 0
    return arm(model, vllm_config, defer_final_ag=defer_final_ag)
