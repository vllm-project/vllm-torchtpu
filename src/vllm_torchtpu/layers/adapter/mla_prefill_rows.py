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
"""TPU_MLA_PREFILL_SEQUENCE_ROWS: the MLA attention block of a split prefill
bucket on each rank's own residual rows.

Under TPU_MOE_PREFILL_SEQUENCE_PARALLEL rank `r` holds the residual rows
`[r * R, (r + 1) * R)`, `R = T / tp`. With TPU_DCP_PREFILL_TOKEN_PARALLEL each
MoE layer's attention then all-gathers the input projections of every row,
attends rows `split_rows + r * R + [0, R)`, returns the outputs to head
sharding by an all-to-all and reduce-scatters o_proj's partial sums back to
the own rows. On a bucket `applies` takes, each MoE layer instead (`forward`):

  * runs every per-row op on its own rows and all-gathers only what the other
    ranks write into their cache shards: the fp8 latent, rope and index-K rows;
  * hands the DCP sub-batch, rows `[0, split_rows)` (rank 0's), their q_c and
    raw index weights from rank 0;
  * attends its own rows with every head (`TokenParallelLayout.rows_aligned`);
  * runs W_UV and o_proj on them with every rank's o_proj / W_UV shard,
    gathered while the layer before runs (`prefetch`,
    `kernels/mla/rows_oproj.py`).

No activation all-to-all or reduce-scatter is left in the block. The shard
partials of o_proj are summed in f32 instead of a bf16 reduce-scatter.
"""

import dataclasses
import functools
from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec as P
from torch_tpu._internal import pallas
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.attention.attention import get_attention_context

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import (
    DCP_AXIS_NAME,
    get_dcp_rank,
    get_or_create_dcp_mesh,
)
from vllm_torchtpu.kernels.mla import rows_oproj
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.layers.adapter.linear_common import (
    FP8_PALLAS_SCALE,
    FP8_PALLAS_WEIGHT,
    WEIGHT_FLIPPED_ATTR,
)
from vllm_torchtpu.layers.core.quantization import quantize_kv

# Smallest bucket that takes the rows block (a power of two:
# `bucket_gate_edge`). The shard gather is the same size at every bucket; the
# collectives it replaces scale with the bucket.
MIN_TOKENS = 2048

# On an armed MLA wrapper: its `_Rows`.
ROWS_ATTR = "_tpu_mla_rows"


def applies(num_tokens: int) -> bool:
    """Whether this bucket (static at trace time) takes the rows block; the
    MoE split and the token-parallel split must take it too."""
    from vllm_torchtpu.layers.adapter import moe_prefill_sp

    return (
        envs.TPU_MLA_PREFILL_SEQUENCE_ROWS
        and num_tokens > envs.bucket_gate_edge(MIN_TOKENS)
        and moe_prefill_sp.applies(num_tokens)
        and dcp_token_parallel_op.applies(num_tokens)
    )


@dataclasses.dataclass(frozen=True)
class _Rows:
    """What one MLA layer's rows block uses, built at arming."""

    prefetch_op: Any
    oproj_op: Any


def check_config(vllm_config: Any) -> None:
    """Raise for a server config the rows block does not serve (at model
    build, with TPU_DCP_PREFILL_TOKEN_PARALLEL's checks)."""
    parallel = vllm_config.parallel_config
    tp = int(parallel.tensor_parallel_size)
    refusals = {
        "needs TPU_MOE_PREFILL_SEQUENCE_PARALLEL": (
            not envs.TPU_MOE_PREFILL_SEQUENCE_PARALLEL
        ),
        "needs TPU_DCP_PREFILL_TOKEN_PARALLEL": not envs.TPU_DCP_PREFILL_TOKEN_PARALLEL,
        "needs decode_context_parallel_size == tensor_parallel_size > 1 (every "
        "rank holds a DCP shard and a residual row block)": (
            parallel.decode_context_parallel_size != tp or tp < 2
        ),
    }
    if not any(refusals.values()):
        split_rows = dcp_token_parallel_op.layout_for(vllm_config).split_rows
        refusals[
            f"needs every bucket it takes to give rank 0 at least the {split_rows} "
            "DCP rows"
        ] = (envs.bucket_gate_edge(MIN_TOKENS) + 1) // tp < split_rows
    bad = [why for why, hit in refusals.items() if hit]
    if bad:
        raise NotImplementedError("TPU_MLA_PREFILL_SEQUENCE_ROWS " + "; ".join(bad))


def _check_layer(mla: Any) -> None:
    attn = mla.mla_attn
    # The q_b overlap, the decode prologue and the local-q window act only on
    # the buckets the prefill splits leave (the wrapper's forward), never here.
    problems = {
        "the q_lora MLA path without fused preprocessing, gates or g_proj": (
            mla.q_lora_rank is None
            or mla.kimi_preprocess is not None
            or mla.gate_is_fused
            or mla.g_proj is not None
        ),
        "sparse MLA under query replication with the rows op built": (
            not mla.is_sparse
            or not mla.dcp_q_replicate
            or getattr(attn, "sparse_mla_tp_rows_op", None) is None
        ),
        "bf16 W_UK / W_UV (not quantized on load)": (
            getattr(attn, "W_UK_T_dcp_qrep_scale", None) is not None
            or hasattr(attn, "W_UV_scale")
            or attn.W_UV.dtype != torch.bfloat16
        ),
        "an fp8 KV cache": attn.kv_cache_quantized_dtype != torch.float8_e4m3fn,
        "an fp8 per-channel (k, n) o_proj": (
            mla.o_proj.weight.dtype != torch.float8_e4m3fn
            or not getattr(mla.o_proj, WEIGHT_FLIPPED_ATTR, False)
            or mla.o_proj.weight_scale.dim() != 1
            or getattr(mla.o_proj, "bias", None) is not None
        ),
    }
    indexer = mla.indexer if not mla.skip_topk else None
    if indexer is not None:
        problems["the indexer's rows op"] = (
            getattr(indexer.indexer_op, "topk_op_tp_rows", None) is None
        )
    bad = [what for what, hit in problems.items() if hit]
    if bad:
        raise NotImplementedError(
            "TPU_MLA_PREFILL_SEQUENCE_ROWS needs " + "; ".join(bad) + f" ({mla.prefix})"
        )


# Built once: both ops take every shape from their operands.
_OPS: list[_Rows] = []


def _build_ops() -> _Rows:
    """The shard gather and the W_UV + o_proj ops (shared by every layer)."""
    if _OPS:
        return _OPS[0]
    mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
    sharded = P(DCP_AXIS_NAME)

    def gather_fn(
        w_o: jax.Array, w_uv: jax.Array, w_scale: jax.Array, after: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # Each rank's gathered shards stacked along dim 0 of the outputs.
        return jax.shard_map(
            functools.partial(rows_oproj.gather_shards, axis_name=DCP_AXIS_NAME),
            mesh=mesh,
            in_specs=(sharded,) * 4,
            out_specs=(sharded,) * 3,
            check_vma=False,
        )(w_o, w_uv, w_scale, after)

    prefetch_op = pallas.jax_op(
        "pallas::mla_rows_gather",
        gather_fn,
        mesh=mesh,
        input_partition_specs=(sharded,) * 4,
        output_partition_specs=(sharded,) * 3,
    )
    tp = mesh.devices.size

    def _gather_fake(w_o, w_uv, w_scale, *_a, **_k):
        return (
            w_o.new_empty((tp, *w_o.shape)),
            w_uv.new_empty((tp * w_uv.shape[0], *w_uv.shape[1:])),
            w_scale.new_empty((tp, *w_scale.shape), dtype=torch.float32),
        )

    prefetch_op.register_fake(_gather_fake)

    def oproj_fn(
        out_latent: jax.Array, w_uv: jax.Array, w_o: jax.Array, w_scale: jax.Array
    ) -> jax.Array:
        return rows_oproj.rows_oproj(out_latent, w_uv, w_o, w_scale)

    oproj_op = pallas.jax_op("pallas::mla_rows_oproj", oproj_fn)
    oproj_op.register_fake(
        lambda out_latent, w_uv, w_o, w_scale: out_latent.new_empty(
            (out_latent.shape[0], w_o.shape[2])
        )
    )
    _OPS.append(_Rows(prefetch_op, oproj_op))
    return _OPS[0]


def arm(mla: Any) -> None:
    """Prepare one MoE layer's MLA wrapper (at load, after weights are
    processed). Raises for anything the rows block does not serve."""
    _check_layer(mla)
    attn = mla.mla_attn
    tp = get_tensor_model_parallel_world_size()
    if get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME).devices.size != tp:
        raise NotImplementedError("the DCP mesh must be the whole TP group")
    # The residual rows follow the MoE split's rank (its EP mesh index, which
    # moe_prefill_sp.arm checks is the TP rank); the aligned ops and the DCP
    # merge follow the DCP rank. They must be one index.
    if get_dcp_rank() != get_tp_group().rank_in_group:
        raise NotImplementedError(
            f"DCP rank {get_dcp_rank()} != TP rank {get_tp_group().rank_in_group}"
        )
    heads, _, v_head_dim = attn.W_UV.shape
    if mla.o_proj.weight.shape[0] != heads * v_head_dim:
        raise NotImplementedError(
            f"o_proj rows {mla.o_proj.weight.shape[0]} != the rank's heads x "
            f"v_head_dim {heads * v_head_dim} ({mla.prefix})"
        )
    setattr(mla, ROWS_ATTR, _build_ops())


def _prefetched(mla: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """What `prefetch` gathers: o_proj's weight, W_UV and o_proj's scale."""
    return mla.o_proj.weight, mla.mla_attn.W_UV, mla.o_proj.weight_scale


def keep_list(mla: Any) -> list[torch.Tensor]:
    """Every tensor the layer's attention reads on any bucket, for the layer's
    keep list (`moe_prefill_sp.KEEP_ATTR`): the warmup hands a bucket over
    only between traces that lift the same buffers under the same names, and
    the rows block reads neither the wrapper's operands (o_proj's kernel copy,
    the token-parallel rank, the local-q window) nor the decode buckets' W_UV /
    W_UK / W_O copies, while it reads o_proj's weight and W_UV, which the decode
    buckets need not. `_prefetched(mla)` comes first: `prefetch` reads it from
    the keep list (a layer early), so every trace names it as the keep list
    does."""
    attn = mla.mla_attn
    tp_op = getattr(attn, "sparse_mla_tp_op", None)
    rotary = getattr(mla, "rotary_emb", None)
    operands = (
        getattr(mla.o_proj, FP8_PALLAS_WEIGHT, None),
        getattr(mla.o_proj, FP8_PALLAS_SCALE, None),
        getattr(tp_op, "rank", None),
        getattr(mla, "_dcp_rank_keep", None),
        getattr(mla, "_dcp_lane_keep", None),
        getattr(mla, "local_q_window", None),
        getattr(mla, "local_q_lane", None),
        getattr(rotary, "cos_sin_cache", None),
        *(
            getattr(attn, name, None)
            for name in (
                "W_UK_T",
                "W_UK_T_scale",
                "W_UK_T_dcp_qrep",
                "W_UV_dcp_qrep",
                "W_UV_dcp_qrep_scale",
                "W_O_dcp_qrep",
                "W_O_dcp_qrep_scale",
            )
        ),
    )
    seen: dict[int, torch.Tensor] = {}
    for t in (*_prefetched(mla), *mla.parameters(), *mla.buffers(), *operands):
        if isinstance(t, torch.Tensor):
            seen.setdefault(id(t), t)
    return list(seen.values())


def check_keep(mla: Any, keep: list[torch.Tensor]) -> None:
    """Raise unless a layer's keep list starts with `prefetch`'s operands."""
    want = _prefetched(mla)
    if len(keep) < len(want) or any(a is not b for a, b in zip(keep, want)):
        raise RuntimeError(f"the keep list of {mla.prefix} must start with keep_list")


def rows_layers(skip_topk: list[bool]) -> list[bool]:
    """Which of a model's MoE layers (in order; `skip_topk[i]`: layer i reuses
    the top-k rows of the indexer layer before it) take the rows block: a
    layer that reuses top-k rows only after one that took it (the rows are in
    its order), so the first is an indexer layer."""
    out: list[bool] = []
    for reuses in skip_topk:
        out.append(not reuses or bool(out and out[-1]))
    return out


def armed(layer: Any) -> bool:
    mla = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
    return getattr(mla, ROWS_ATTR, None) is not None


def prefetch(
    layer: Any, keep: list[torch.Tensor], after: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Every rank's o_proj, W_UV and o_proj scale shard of `layer`
    (`rows_oproj.gather_shards`), gathered once `after` (this rank's rows of
    the layer input before) exists: the gather has no activation input, so it
    can run under the layers in between. `keep` is the layer's keep list,
    `keep_list` first: the shards are read from it, not from the modules."""
    rows: _Rows = getattr(layer.self_attn.mla_attn, ROWS_ATTR)
    return rows.prefetch_op(keep[0], keep[1], keep[2], after)


def _rope(mla: Any):
    """The wrapper's rope: NeoX on half-split columns once permuted."""
    from vllm_torchtpu.layers.adapter.custom_ops.mla_attention_op import (
        rope_half_split,
    )

    if getattr(mla, "_rope_neox_permuted", False):
        return functools.partial(rope_half_split, mla.rotary_emb)
    return mla.rotary_emb


def _queries(mla: Any, q_c: torch.Tensor, positions: torch.Tensor):
    """Every head's (ql_nope, q_pe) of rows `q_c` at `positions`: q_b, rope
    and the absorbed W_UK, as the wrapper and backend compute them."""
    from vllm_torchtpu.layers.adapter.attention import _absorb_q

    attn = mla.mla_attn
    heads = mla.num_heads * mla.dcp_group_size
    q = mla.q_b_proj(q_c)[0].view(-1, heads, mla.qk_head_dim)
    q_nope, q_pe = q.split([mla.qk_nope_head_dim, mla.qk_rope_head_dim], dim=-1)
    # The rope takes a key too; this one's result is dropped.
    q_pe = _rope(mla)(positions, q_pe, q_pe[:, :1])[0].to(q.dtype)
    ql_nope = _absorb_q(q_nope, attn.W_UK_T_dcp_qrep.to(q_nope.dtype)).to(q.dtype)
    return ql_nope, q_pe


def forward(
    mla: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    gathered: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rank: torch.Tensor,
) -> torch.Tensor:
    """The attention block of `mla` (an armed MLA wrapper) on this rank's rows
    of the input-normed `hidden_states` `[R, hidden]`; returns this rank's
    rows of its output `[R, hidden]`. `gathered` is `prefetch`'s shards of
    this layer, `rank` this rank's TP index as an int32 `[1, 1]` operand."""
    from vllm_torchtpu.layers.adapter import moe_prefill_sp

    rows: _Rows = getattr(mla, ROWS_ATTR)
    attn = mla.mla_attn
    tp = get_tensor_model_parallel_world_size()
    split = attn.sparse_mla_tp_rows_op.layout.split_rows
    pos_own = moe_prefill_sp.own_rows(positions, rank, tp)

    qkv = mla.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv.split(
        [mla.q_lora_rank, mla.kv_lora_rank + mla.qk_rope_head_dim], dim=-1
    )
    q_c = mla.q_a_layernorm(q_c)
    kv_c, k_pe = kv_lora.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
    kv_c = mla.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)
    k_pe = _rope(mla)(pos_own, k_pe, k_pe)[1].to(k_pe.dtype)
    _, k_scale, _ = attn.impl._get_kv_scales(attn)
    kv_dtype = attn.kv_cache_quantized_dtype
    kv_q, _ = quantize_kv(kv_dtype, kv_c, value=None, k_scale=k_scale)
    kpe_q, _ = quantize_kv(kv_dtype, k_pe.flatten(1), value=None, k_scale=k_scale)

    indexer = mla.indexer if not mla.skip_topk else None
    parts = [kv_q.view(torch.uint8), kpe_q.view(torch.uint8)]
    src = [q_c[:split]]
    if indexer is not None:
        kw = indexer.wk_weights_proj(hidden_states)[0]
        idx_q, idx_w, k_packed = indexer.rows_inputs(
            q_c, pos_own, mla.indexer_rope_emb, kw
        )
        parts.append(k_packed)
        src.append(kw[:split])
    # What every rank writes into its cache shard: every row's cache-ready
    # latent, rope and index K.
    payload = moe_prefill_sp.all_rows(torch.cat(parts, dim=-1), positions.shape[0])
    kv_all = payload[:, : mla.kv_lora_rank].view(kv_q.dtype)
    kpe_end = mla.kv_lora_rank + mla.qk_rope_head_dim
    kpe_all = payload[:, mla.kv_lora_rank : kpe_end].view(kpe_q.dtype)
    # The DCP sub-batch's rows are rank 0's own first rows. The rows buckets
    # carry more than split_rows real tokens, so the TPU all-reduce's
    # valid-row count (from the step's padding mask) covers all of them.
    src = torch.cat(src, dim=-1)
    src = tensor_model_parallel_all_reduce(
        torch.where(rank.reshape(1, 1) == 0, src, torch.zeros_like(src))
    )
    q_c_dcp = src[:, : mla.q_lora_rank]
    pos_dcp = positions[:split]

    # The ops below take the DCP sub-batch's rows, then this rank's own.
    ql_dcp, pe_dcp = _queries(mla, q_c_dcp, pos_dcp)
    ql_own, pe_own = _queries(mla, q_c, pos_own)
    ql_nope, q_pe = torch.cat([ql_dcp, ql_own]), torch.cat([pe_dcp, pe_own])
    if indexer is not None:
        idx_q_dcp, idx_w_dcp, _ = indexer.rows_inputs(
            q_c_dcp, pos_dcp, mla.indexer_rope_emb, src[:, mla.q_lora_rank :]
        )
        topk_rows = indexer.indexer_op.rows_topk(
            torch.cat([idx_q_dcp, idx_q]),
            torch.cat([idx_w_dcp, idx_w]),
            payload[:, kpe_end:],
        )
    else:
        topk_rows = mla.topk_indices_buffer

    attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
    if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
        # The memory probe before the caches exist (as the backend's forward).
        latent = ql_nope.new_zeros(ql_nope[split:].shape)
    else:
        latent = dcp_token_parallel_op.attention_rows(
            attn,
            (kv_cache[0], kv_cache[1]),
            ql_nope,
            q_pe,
            kv_all,
            kpe_all,
            topk_rows,
            attn_metadata,
        )
    w_o, w_uv, w_scale = gathered
    return rows.oproj_op(latent, w_uv, w_o, w_scale)
