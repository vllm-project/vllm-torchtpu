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
MoE layer's attention then all-gathers fused_qkv_a's output of every row,
runs the per-row work (norms, rope, q, index q) on every row, attends rows
`split_rows + r * R + [0, R)`, returns the outputs to head sharding by an
all-to-all and reduce-scatters o_proj's partial sums back to the own rows. On
a bucket `applies` takes, each MoE layer instead (`forward`):

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
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# On an armed MLA wrapper: its `_Rows`.
ROWS_ATTR = "_tpu_mla_rows"
# Row tile of the W_UV + o_proj kernel: every shard's o_proj is read once per
# tile, so a tile of all R rows reads it once.
OPROJ_BLOCK_ROWS = 512


def applies(num_tokens: int) -> bool:
    """Whether this bucket (static at trace time) takes the rows block; the
    MoE split and the token-parallel split must take it too."""
    from vllm_torchtpu.layers.adapter import moe_prefill_sp

    return (
        envs.TPU_MLA_PREFILL_SEQUENCE_ROWS
        and num_tokens > envs.bucket_gate_edge(envs.TPU_MLA_PREFILL_ROWS_MIN_TOKENS)
        and moe_prefill_sp.applies(num_tokens)
        and dcp_token_parallel_op.applies(num_tokens)
    )


@dataclasses.dataclass(frozen=True)
class _Rows:
    """What one MLA layer's rows block uses, built at arming."""

    geom: rows_oproj.PackedOProj
    prefetch_op: Any
    oproj_op: Any


def check_config(vllm_config: Any) -> None:
    """Raise for a server config the rows block does not serve (at model
    build, with TPU_DCP_PREFILL_TOKEN_PARALLEL's checks)."""
    parallel = vllm_config.parallel_config
    tp = int(parallel.tensor_parallel_size)
    split_rows = dcp_token_parallel_op.split_rows_for(vllm_config)
    refusals = {
        "needs TPU_MOE_PREFILL_SEQUENCE_PARALLEL": (
            not envs.TPU_MOE_PREFILL_SEQUENCE_PARALLEL
        ),
        "needs TPU_DCP_PREFILL_TOKEN_PARALLEL": not envs.TPU_DCP_PREFILL_TOKEN_PARALLEL,
        "needs decode_context_parallel_size == tensor_parallel_size (every rank "
        "holds a DCP shard and a residual row block)": (
            parallel.decode_context_parallel_size != parallel.tensor_parallel_size
        ),
        f"needs every bucket it takes to give rank 0 at least the {split_rows} "
        "DCP rows (TPU_MLA_PREFILL_ROWS_MIN_TOKENS)": (
            (envs.bucket_gate_edge(envs.TPU_MLA_PREFILL_ROWS_MIN_TOKENS) + 1) // tp
            < split_rows
        ),
    }
    bad = [why for why, hit in refusals.items() if hit]
    if bad:
        raise NotImplementedError("TPU_MLA_PREFILL_SEQUENCE_ROWS " + "; ".join(bad))


def _check_layer(mla: Any) -> None:
    attn = mla.mla_attn
    problems = {
        "the q_lora MLA path without fused preprocessing, gates or g_proj": (
            mla.q_lora_rank is None
            or mla.kimi_preprocess is not None
            or mla.gate_is_fused
            or mla.g_proj is not None
            or getattr(mla, "qb_overlap", False)
            or getattr(mla, "decode_prologue", False)
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


_OPS: dict[tuple, tuple[Any, Any]] = {}


def _build_ops(geom: rows_oproj.PackedOProj) -> tuple[Any, Any]:
    """The shard gather and the W_UV + o_proj ops (shared by every layer)."""
    gather = envs.TPU_MLA_PREFILL_ROWS_GATHER
    key = (geom, gather)
    if key in _OPS:
        return _OPS[key]
    mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
    ring = rows_oproj.ring_order(mesh.devices.reshape(-1)) if gather == "ring" else None
    sharded = P(DCP_AXIS_NAME)

    def local(w_o, w_uv, w_scale, after):
        return rows_oproj.gather_shards(
            geom, w_o, w_uv, w_scale, after, DCP_AXIS_NAME, ring
        )

    def gather_fn(w_o, w_uv, w_scale, after):
        # Each rank's [shards, rows, n_out] stacked along dim 0.
        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(sharded,) * 4,
            out_specs=sharded,
            check_vma=False,
        )(w_o, w_uv, w_scale, after)

    prefetch_op = pallas.jax_op(
        f"pallas::mla_rows_gather_{gather}",
        gather_fn,
        mesh=mesh,
        input_partition_specs=(sharded,) * 4,
        output_partition_specs=sharded,
    )
    prefetch_op.register_fake(
        lambda w_o, *_a, **_k: w_o.new_empty((geom.shards, geom.rows, geom.n_out))
    )

    def oproj_fn(out_latent, gathered):
        w_o, w_uv, w_scale = rows_oproj.unpack(geom, gathered, out_latent[0, 0])
        return rows_oproj.rows_oproj(
            out_latent,
            w_uv,
            w_o,
            w_scale,
            k_shard=geom.k_shard,
            block_rows=OPROJ_BLOCK_ROWS,
        )

    oproj_op = pallas.jax_op(f"pallas::mla_rows_oproj_s{geom.shards}", oproj_fn)
    oproj_op.register_fake(
        lambda out_latent, gathered: out_latent.new_empty(
            (out_latent.shape[0], geom.n_out)
        )
    )
    _OPS[key] = (prefetch_op, oproj_op)
    return _OPS[key]


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
    heads, kv_lora_rank, v_head_dim = attn.W_UV.shape
    geom = rows_oproj.PackedOProj(
        shards=tp,
        k_shard=heads * v_head_dim,
        n_out=mla.o_proj.weight.shape[1],
        heads=heads,
        kv_lora_rank=kv_lora_rank,
        v_head_dim=v_head_dim,
    )
    prefetch_op, oproj_op = _build_ops(geom)
    setattr(mla, ROWS_ATTR, _Rows(geom, prefetch_op, oproj_op))


def keep_list(mla: Any) -> list[torch.Tensor]:
    """What the other buckets' attention block reads and the rows block does
    not, for the layer's keep list (`moe_prefill_sp.KEEP_ATTR`): the warmup
    hands a bucket over only between traces that lift the same buffers. The
    rows block reads o_proj's weight and scale but not o_proj's kernel copy,
    and the rows from the MoE split's rank operand, not the token-parallel
    op's."""
    o_proj = mla.o_proj
    tensors = (
        getattr(o_proj, FP8_PALLAS_WEIGHT, None),
        getattr(o_proj, FP8_PALLAS_SCALE, None),
        mla.mla_attn.sparse_mla_tp_op.rank,
    )
    return [t for t in tensors if t is not None]


def armed(layer: Any) -> bool:
    mla = getattr(getattr(layer, "self_attn", None), "mla_attn", None)
    return getattr(mla, ROWS_ATTR, None) is not None


def prefetch(layer: Any, after: torch.Tensor) -> torch.Tensor:
    """Every rank's o_proj / W_UV shard of `layer`, gathered once `after`
    (this rank's rows of the layer input before) exists: the gather has no
    activation input, so it can run under the layers in between."""
    mla = layer.self_attn.mla_attn
    rows: _Rows = getattr(mla, ROWS_ATTR)
    return rows.prefetch_op(
        mla.o_proj.weight,
        mla.mla_attn.W_UV,
        mla.o_proj.weight_scale,
        after,
    )


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
    w_uk = attn.W_UK_T_dcp_qrep
    ql_nope = _absorb_q(q_nope, w_uk.to(q_nope.dtype)).to(q.dtype)
    return ql_nope, q_pe


def forward(
    mla: Any,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    gathered: torch.Tensor,
    rank: torch.Tensor,
) -> torch.Tensor:
    """The attention block of `mla` (an armed MLA wrapper) on this rank's rows
    of the input-normed `hidden_states` `[R, hidden]`; returns this rank's
    rows of its output `[R, hidden]`. `gathered` is `prefetch`'s buffer of
    this layer, `rank` this rank's TP index as an int32 `[1, 1]` operand."""
    from vllm_torchtpu.layers.adapter import moe_prefill_sp

    rows: _Rows = getattr(mla, ROWS_ATTR)
    attn = mla.mla_attn
    tp = get_tensor_model_parallel_world_size()
    split = attn.sparse_mla_tp_rows_op.layout.split_rows
    pos_own = moe_prefill_sp.own_rows(positions, rank, tp)
    rope = _rope(mla)

    qkv = mla.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv.split(
        [mla.q_lora_rank, mla.kv_lora_rank + mla.qk_rope_head_dim], dim=-1
    )
    q_c = mla.q_a_layernorm(q_c)
    kv_c, k_pe = kv_lora.split([mla.kv_lora_rank, mla.qk_rope_head_dim], dim=-1)
    kv_c = mla.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(1)
    k_pe = rope(pos_own, k_pe, k_pe)[1].to(k_pe.dtype)
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
    # The DCP sub-batch's rows are rank 0's own first rows.
    src = torch.cat(src, dim=-1)
    src = tensor_model_parallel_all_reduce(
        torch.where(rank.reshape(1, 1) == 0, src, torch.zeros_like(src))
    )
    q_c_dcp = src[:, : mla.q_lora_rank]
    pos_dcp = positions[:split]

    ql_nope, q_pe = _queries(mla, q_c, pos_own)
    ql_nope_dcp, q_pe_dcp = _queries(mla, q_c_dcp, pos_dcp)

    num_rows = split + hidden_states.shape[0]
    if indexer is not None:
        idx_q_dcp, idx_w_dcp, _ = indexer.rows_inputs(
            q_c_dcp, pos_dcp, mla.indexer_rope_emb, src[:, mla.q_lora_rank :]
        )
        topk_rows = indexer.indexer_op.rows_topk(
            idx_q, idx_w, idx_q_dcp, idx_w_dcp, payload[:, kpe_end:]
        )
    else:
        topk_rows = mla.topk_indices_buffer
    topk_rows = topk_rows[:num_rows]

    attn_metadata, _, kv_cache, _ = get_attention_context(attn.layer_name)
    if isinstance(kv_cache, torch.Tensor) and kv_cache.numel() == 0:
        # The memory probe before the caches exist (as the backend's forward).
        latent = ql_nope.new_zeros(ql_nope.shape)
    else:
        latent = dcp_token_parallel_op.attention_rows(
            attn,
            kv_cache,
            ql_nope,
            q_pe,
            ql_nope_dcp,
            q_pe_dcp,
            kv_all,
            kpe_all,
            topk_rows,
            attn_metadata,
            rank,
        )
    return rows.oproj_op(latent, gathered)
