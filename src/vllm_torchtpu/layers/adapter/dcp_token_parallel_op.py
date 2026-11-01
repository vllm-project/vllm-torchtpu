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
"""TPU_DCP_PREFILL_TOKEN_PARALLEL: the torch side of the token-parallel split
of large DCP sparse MLA buckets (`kernels/mla/dcp_token_parallel.py`).

On a backbone bucket `applies` takes, every sparse MLA layer runs

  * rows `[0, split_rows)` through today's DCP ops on a sub-batch
    (`dcp_rows_metadata`): the indexer's rank-local top-k, the head
    all-gather, the sharded attention and the log-sum-exp merge;
  * rows `[split_rows, T)` through the token-parallel ops built here.

The indexer leaves both halves in the shared top-k buffer, one after the
other: rows `[0, split_rows)` hold the DCP rank-local indices as before, rows
`[split_rows, split_rows + T / dcp)` this rank's token-parallel rows as flat
rows of the step's gather buffer. Every layer of the step derives the same
buffer coordinates from the same metadata, so IndexShare layers read them
unchanged.

The choice is static per compiled bucket; a step the gather buffer cannot
hold raises in the runner (`check_step_capacity`) instead of running.
"""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import lax
from jax.sharding import PartitionSpec as P
from vllm.compilation import backends as vllm_compilation_backends

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import DCP_AXIS_NAME, get_or_create_dcp_mesh
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    streamindex_topk,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    pcp_streaming_jax_op,
)
from vllm_torchtpu.kernels.mla import dcp_token_parallel, page_gather
from vllm_torchtpu.kernels.mla.kv_cache_utils import update_sparse_mla_kv_cache_dcp
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel

# Smallest bucket that takes the split (a power of two: `bucket_gate_edge`).
MIN_TOKENS = 1024
# Virtual pages the gather buffer holds past one max-model-len request: the
# other requests a chunked-prefill step can carry next to it.
EXTRA_PAGES = 4
# Rows [0, split_rows) stay on the DCP path; a multiple of this.
SPLIT_ROWS_ALIGN = 16
# Gather sizes (virtual pages) below the buffer's capacity a step may take.
GATHER_CAPS = (2, 4, 8, 12, 16, 24, 32)


def applies(num_tokens: int) -> bool:
    """Whether this bucket (static at trace time) takes the split. The
    backbone only: it is traced once per bucket."""
    return (
        envs.TPU_DCP_PREFILL_TOKEN_PARALLEL
        and vllm_compilation_backends.model_tag == "backbone"
        and num_tokens > envs.bucket_gate_edge(MIN_TOKENS)
    )


def layout_for(vllm_config: Any) -> dcp_token_parallel.TokenParallelLayout:
    """The split's static shape for this server config."""
    parallel = vllm_config.parallel_config
    dcp = int(parallel.decode_context_parallel_size)
    page_size = int(vllm_config.cache_config.block_size)
    max_num_seqs = int(vllm_config.scheduler_config.max_num_seqs)
    vps = page_size * dcp
    max_model_len = int(vllm_config.model_config.max_model_len)
    max_pages = -(-max_model_len // vps) + EXTRA_PAGES
    return dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=int(parallel.cp_kv_cache_interleave_size),
        page_size=page_size,
        split_rows=-(-max_num_seqs // SPLIT_ROWS_ALIGN) * SPLIT_ROWS_ALIGN,
        max_pages=max_pages,
        gather_caps=tuple(c for c in GATHER_CAPS if c < max_pages) + (max_pages,),
    )


def check_config(vllm_config: Any, *, q_replicate: bool) -> None:
    """Raise for a config the split does not serve."""
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL: speculative decoding is not "
            "supported (the drafter is traced once for every bucket)."
        )
    if q_replicate and envs.TPU_MLA_DECODE_PROLOGUE:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL does not take TPU_MLA_DECODE_PROLOGUE "
            "(an un-roped q under query replication)."
        )
    if envs.TPU_DCP_MLA_UV_BEFORE_MERGE or envs.TPU_DCP_MLA_OPROJ_ALL_REDUCE:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL returns merged latent outputs; it "
            "does not take TPU_DCP_MLA_UV_BEFORE_MERGE / _OPROJ_ALL_REDUCE."
        )
    if envs.TPU_INDEXER_QB_OVERLAP:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL does not take TPU_INDEXER_QB_OVERLAP."
        )
    if vllm_config.cache_config.enable_prefix_caching:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL does not take prefix caching: the "
            "gather buffer is sized for the scheduled rows (check_step_capacity)."
        )
    if vllm_config.scheduler_config.max_num_seqs > MIN_TOKENS // 2:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL: max_num_seqs must leave rows for "
            f"the token-parallel side of a {MIN_TOKENS}-token bucket."
        )


def check_step_capacity(
    vllm_config: Any,
    padded_num_tokens: int,
    query_start_loc: np.ndarray,
    seq_lens: np.ndarray,
    num_reqs: int,
) -> None:
    """Raise if a step that takes the split needs more gather pages than the
    buffer holds (host side, before the step runs)."""
    if not envs.TPU_DCP_PREFILL_TOKEN_PARALLEL:
        return
    if int(vllm_config.parallel_config.decode_context_parallel_size) <= 1:
        return
    if padded_num_tokens <= envs.bucket_gate_edge(MIN_TOKENS):
        return
    layout = layout_for(vllm_config)
    q_start = query_start_loc[:num_reqs]
    q_end = query_start_loc[1 : num_reqs + 1]
    listed = (q_end > layout.split_rows) & (q_end > q_start)
    vps = layout.virtual_page_size
    need = int(np.sum((seq_lens[:num_reqs][listed] + vps - 1) // vps))
    if need > layout.max_pages:
        raise RuntimeError(
            f"TPU_DCP_PREFILL_TOKEN_PARALLEL: this step's prefill requests "
            f"need {need} gather pages, the buffer holds {layout.max_pages} "
            f"(seq_lens {seq_lens[:num_reqs][listed].tolist()})."
        )


def dcp_rows_metadata(
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    distribution: torch.Tensor,
    split_rows: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The step cut after row `split_rows`: a request that straddles the cut
    loses its rows past it (its `seq_len` with them), later ones drop out.
    `distribution`'s first two counts are clipped to the requests left."""
    qsl = torch.clamp(query_start_loc, max=split_rows)
    cut = (query_start_loc[1:] - query_start_loc[:-1]) - (qsl[1:] - qsl[:-1])
    req = torch.arange(
        seq_lens.shape[0], device=seq_lens.device, dtype=distribution.dtype
    )
    n = ((query_start_loc[:-1] < split_rows) & (req < distribution[2])).sum()
    n = n.to(distribution.dtype)
    dist = torch.stack(
        [torch.minimum(distribution[0], n), torch.minimum(distribution[1], n), n]
    )
    return seq_lens - cut, qsl, dist


def _sharded_rank_ids() -> torch.Tensor | None:
    if not envs.TPU_DCP_RANK_OPERAND:
        return None
    from vllm_torchtpu.distributed.dcp import dcp_rank_operand

    return dcp_rank_operand()


def make_attend(
    *,
    sm_scale: float,
    k_scale: float,
    cache_layout: str,
    packed_rope: bool,
    lkv_dim: int,
) -> dcp_token_parallel.AttendFn:
    """The single-device sparse MLA kernel as the split's `attend`: one ragged
    "request" of `num_rows` rows; with no page table the kernel reads the
    top-k as flat rows of the gathered buffer. Exact softmax over the row's
    whole top-k, so no log-sum-exp is returned."""
    from vllm_torchtpu.layers.core.attention_interface import sparse_mla_chunk_kwargs

    packed = {"packed_rope": True} if packed_rope else {}

    def attend(q, g_nope, g_rope, topk, num_rows):
        out = sparse_mla_kernel.sparse_ragged_paged_attention(
            q,
            g_nope,
            g_rope,
            topk,
            None,
            jnp.stack([jnp.int32(0), num_rows.astype(jnp.int32)]),
            jnp.array([0, 0, 1], jnp.int32),
            sm_scale=sm_scale,
            k_scale=k_scale,
            cache_layout=cache_layout,
            **sparse_mla_chunk_kwargs(),
            **packed,
        )
        return out[..., :lkv_dim]

    return attend


def make_select_topk(
    topk: int,
    kv_layout: KVLayout,
    *,
    chunk_tokens: int | None,
    small_bucket: tuple,
    skip_empty_decode: bool,
    skip_empty_decode_chunked: bool,
    mixed_bkvp: int | None,
    decode_pack: bool,
    shared_meta: bool,
) -> dcp_token_parallel.SelectTopkFn:
    """The single-device StreamIndex top-k as the split's `select_topk`, with
    the block sizes and knobs the DCP indexer op uses (`cp_size=1`: exact
    top-k, global positions)."""
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

    def select(q_r, w_r, natural, rr, table):
        bkvp, bq = mla_op._streamidx_block_sizes(
            q_r.shape[0], *small_bucket, mixed_bkvp, decode_pack
        )
        zero = jnp.int32(0)
        return streamindex_topk(
            q_r,
            w_r,
            natural,
            rr.seq_lens,
            table,
            rr.cu_q_lens,
            jnp.stack([zero, zero, rr.num_seqs]),
            k=topk,
            compression_ratio=1,
            num_kv_pages_per_block=bkvp,
            num_queries_per_block=bq,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            enable_early_exit=mla_op.STREAMIDX_ENABLE_EARLY_EXIT,
            kv_layout=kv_layout,
            chunk_tokens=chunk_tokens,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
        )

    return select


def _after_reads(*caches: jax.Array, after: jax.Array) -> tuple[jax.Array, ...]:
    """`caches`, ordered after `after` (a result of the step's DCP sub-batch,
    which reads the same caches). Without the edge XLA may schedule this op's
    in-place cache write first and then has to copy the whole cache to keep
    the value the sub-batch reads."""
    return lax.optimization_barrier((*caches, after))[:-1]


def build_attention_op(
    layer: Any,
    impl: Any,
    layout: dcp_token_parallel.TokenParallelLayout,
    q_replicated: bool,
):
    """The token-parallel sparse MLA op of `layer`: `(kv_cache, q, kv_c_normed,
    k_pe, flat_topk, seq_lens, block_tables, query_start_loc,
    request_distribution) -> [T, heads, kv_lora_rank]`, row `i` being step row
    `split_rows + i`. `q` is this rank's heads of every row, latent nope and
    rope concatenated -- or, with `q_replicated` (query replication), every
    head of this rank's own rows (`q_rows`); the op writes every row's
    latents into the caches."""
    mesh = impl._dcp_mesh
    assert mesh is not None
    dcp, il = impl.dcp_size, impl.dcp_interleave_size
    if (dcp, il) != (layout.dcp_size, layout.interleave_size):
        raise ValueError(f"backend DCP ({dcp}, {il}) does not match {layout}")
    k_scale = None
    if layer.kv_cache_quantized_dtype is not None:
        _, k_scale, _ = impl._get_kv_scales(layer)
    dequant = float(k_scale) if k_scale is not None else 1.0
    rank_tensor = _sharded_rank_ids()

    def _run(
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q: jax.Array,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        flat_topk: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        after: jax.Array,
        rank_ids: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        kv_cache_nope, kv_cache_rope = _after_reads(
            kv_cache_nope, kv_cache_rope, after=after
        )
        nope_spec, rope_spec = layer.mla_kv_spec
        if nope_spec.page_size != layout.page_size:
            raise ValueError(
                f"sparse MLA cache page {nope_spec.page_size} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )
        if nope_spec.layout is not rope_spec.layout:
            raise ValueError("sparse MLA needs matching NoPE and RoPE layouts")

        def write_kv(nope, rope, kv_c, k_pe, seq_lens, bt, qsl, rank):
            return update_sparse_mla_kv_cache_dcp(
                nope,
                rope,
                kv_c,
                k_pe,
                seq_lens,
                bt,
                qsl,
                rank,
                nope_spec=nope_spec,
                rope_spec=rope_spec,
                dcp_size=dcp,
                interleave_size=il,
            )

        attend = make_attend(
            sm_scale=layer.scale,
            k_scale=dequant,
            cache_layout=nope_spec.layout.value,
            packed_rope=rope_spec.packed_rope,
            lkv_dim=kv_c_normed.shape[-1],
        )

        return dcp_token_parallel.sparse_mla_attention(
            layout,
            mesh,
            DCP_AXIS_NAME,
            kv_cache_nope,
            kv_cache_rope,
            q,
            kv_c_normed,
            k_pe,
            flat_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            write_kv=write_kv,
            attend=attend,
            rank_ids=rank_ids,
            q_own_rows=q_replicated,
            read_pages=(
                functools.partial(page_gather.gather_pages, row_width=128),
                functools.partial(
                    page_gather.gather_pages, row_width=rope_spec.token_bytes // 4
                ),
            ),
        )

    # Fixed arity per variant: torch custom-op schemas take no varargs.
    def sparse_mla_tp(
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q: jax.Array,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        flat_topk: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        after: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _run(
            kv_cache_nope,
            kv_cache_rope,
            q,
            kv_c_normed,
            k_pe,
            flat_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            after,
            None,
        )

    def sparse_mla_tp_rank(
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q: jax.Array,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        flat_topk: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        after: jax.Array,
        rank_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return _run(
            kv_cache_nope,
            kv_cache_rope,
            q,
            kv_c_normed,
            k_pe,
            flat_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            after,
            rank_ids,
        )

    sharded, rep = P(DCP_AXIS_NAME), P()
    in_specs = (sharded, sharded, sharded, rep, rep, sharded, rep, rep, rep, rep)
    in_specs += (sharded,)
    if rank_tensor is not None:
        in_specs += (sharded,)
    op = pcp_streaming_jax_op(
        f"pallas::sparse_mla_tp_{layer.layer_name.replace('.', '_')}"
        + ("_qrep" if q_replicated else ""),
        sparse_mla_tp if rank_tensor is None else sparse_mla_tp_rank,
        donate_argnums=(0, 1),
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=(sharded, sharded, sharded),
    )
    num_heads, kv_lora_rank = layer.num_heads, layer.kv_lora_rank

    def _fake(kv_cache_nope, kv_cache_rope, q, kv_c_normed, *rest, **kwargs):
        del rest, kwargs
        return (
            torch.empty_like(kv_cache_nope),
            torch.empty_like(kv_cache_rope),
            q.new_empty((kv_c_normed.shape[0], num_heads, kv_lora_rank)),
        )

    op.register_fake(_fake)

    def run(
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        flat_topk: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_distribution: torch.Tensor,
        after: torch.Tensor,
    ) -> torch.Tensor:
        nope_cache, rope_cache = kv_cache
        new_nope, new_rope, out = op(
            nope_cache,
            rope_cache,
            q,
            kv_c_normed,
            k_pe,
            flat_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            after,
            *(() if rank_tensor is None else (rank_tensor,)),
        )
        nope_cache.copy_(new_nope)
        rope_cache.copy_(new_rope)
        return out

    run.layout = layout  # type: ignore[attr-defined]
    run.q_replicated = q_replicated  # type: ignore[attr-defined]
    # This rank's DCP index as data, for `q_rows` (built here, outside Dynamo).
    from vllm_torchtpu.distributed.dcp import dcp_rank_operand

    run.rank = dcp_rank_operand() if q_replicated else None  # type: ignore[attr-defined]
    return run


def q_rows(tp_op: Any, num_tokens: int) -> torch.Tensor:
    """i32[split_rows + T / dcp]: the rows a rank needs every head of under
    query replication -- the DCP sub-batch's, then its own token-parallel
    rows (rows past `T` repeat row `T - 1`, padding). Same graph on every
    rank; only the rank operand's value differs."""
    layout = tp_op.layout
    rows = layout.rows_per_rank(num_tokens)
    rank = tp_op.rank
    ar = torch.arange(layout.split_rows + rows, dtype=rank.dtype, device=rank.device)
    own = ar + torch.where(ar >= layout.split_rows, rank * rows, 0)
    return torch.clamp(own, max=num_tokens - 1)


def attention(
    layer: Any,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    ql_nope: torch.Tensor,
    q_pe: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    topk_rows: torch.Tensor,
    attn_metadata: Any,
    q_replicated: bool,
) -> torch.Tensor:
    """Sparse MLA of a split bucket: `[T, heads, kv_lora_rank]`, this rank's
    heads, merged; what the DCP path returns for the same step. `ql_nope` /
    `q_pe` hold this rank's heads of every row, or, under query replication,
    every head of the `q_rows` rows."""
    from vllm_torchtpu.layers.adapter.cp_mla_attention import (
        all_gather_q_nope_pe_heads,
        merge_lse_partials_scatter_heads,
    )

    tp_op = layer.sparse_mla_tp_op
    if tp_op.q_replicated != q_replicated:
        raise ValueError(
            f"token-parallel op built for q_replicated={tp_op.q_replicated}, "
            f"called with {q_replicated}"
        )
    layout = tp_op.layout
    s = layout.split_rows
    num_tokens = kv_c_normed.shape[0]
    rows = layout.rows_per_rank(num_tokens)
    if q_replicated and ql_nope.shape[0] != s + rows:
        raise ValueError(f"q has {ql_nope.shape[0]} rows, not the {s + rows} q_rows")
    md = attn_metadata
    seq_s, qsl_s, dist_s = dcp_rows_metadata(
        md.seq_lens, md.query_start_loc, md.request_distribution, s
    )
    if q_replicated:
        q_dcp, pe_dcp = ql_nope[:s], q_pe[:s]
    else:
        q_dcp, pe_dcp = all_gather_q_nope_pe_heads(
            ql_nope[:s], q_pe[:s], layer.kv_lora_rank
        )
    partial, lse = layer.sparse_mla_dcp_op(
        kv_cache,
        q_dcp,
        pe_dcp,
        kv_c_normed[:s],
        k_pe[:s],
        topk_rows[:s],
        seq_s,
        md.block_tables,
        qsl_s,
        dist_s,
    )
    head = merge_lse_partials_scatter_heads(
        partial, lse, is_backbone=True, wire_dtype=partial.dtype, query_start_loc=qsl_s
    )
    # The token-parallel op rewrites the caches in place; it must not start
    # before the sub-batch has read them (`_after_reads`).
    q_tp = torch.cat([ql_nope, q_pe], dim=-1)
    tail = tp_op(
        kv_cache,
        q_tp[s:] if q_replicated else q_tp,
        kv_c_normed,
        k_pe,
        topk_rows[s : s + rows],
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        lse,
    )
    return torch.cat([head, tail[: num_tokens - s]])


def build_indexer_op(indexer: Any, layout: dcp_token_parallel.TokenParallelLayout):
    """The token-parallel top-k op of a DCP indexer: `(q_bytes, weights,
    kv_cache, k_packed, seq_lens, page_indices, cu_q_lens, distribution) ->
    [T / dcp, topk]` this rank's rows as flat gather-buffer rows. Writes every
    row's K into the cache first, as the DCP op does."""
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

    dcp, il = layout.dcp_size, layout.interleave_size
    mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
    topk = indexer.topk_tokens
    chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
    small_bucket = (
        envs.TPU_STREAMIDX_SMALL_BUCKET_BQ,
        envs.TPU_STREAMIDX_SMALL_BUCKET_BKVP,
        envs.TPU_STREAMIDX_SMALL_BUCKET_TOKENS,
    )
    kv_cache_mode = (
        envs.VLLM_TPU_INDEXER_KV_CACHE_MODE
        if envs.is_indexer_kv_cache_mode_explicit()
        else None
    )
    skip_empty_decode = envs.TPU_STREAMIDX_SKIP_EMPTY_DECODE
    skip_empty_decode_chunked = envs.TPU_STREAMIDX_SKIP_EMPTY_DECODE_CHUNKED
    mixed_bkvp = envs.TPU_STREAMIDX_MIXED_KV_PAGES
    decode_pack = envs.TPU_STREAMIDX_DECODE_PACK
    shared_meta = indexer._shared_meta
    k_write_compact = envs.TPU_INDEXER_K_WRITE_COMPACT
    rank_tensor = _sharded_rank_ids()

    def _run(
        q_bytes: jax.Array,
        weights: jax.Array,
        cache_kv: jax.Array,
        k_packed: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        after: jax.Array,
        rank_ids: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        (cache_kv,) = _after_reads(cache_kv, after=after)
        writer, kv_layout = mla_op._resolve_indexer_kv_cache_mode(
            cache_kv, k_packed, mode=kv_cache_mode
        )
        if kv_layout != KVLayout.SEQ_ALONG_LANE:
            raise NotImplementedError(
                "TPU_DCP_PREFILL_TOKEN_PARALLEL needs the token-along-lane "
                f"index-K cache, got {kv_layout}"
            )
        if cache_kv.shape[-1] != layout.page_size:
            raise ValueError(
                f"index-K page {cache_kv.shape[-1]} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )
        cache_kv = mla_op._cp_scatter_indexer_k(
            cache_kv,
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            mesh=mesh,
            cp_size=dcp,
            interleave_size=il,
            axis_name=DCP_AXIS_NAME,
            writer=writer,
            kv_layout=kv_layout,
            rank_ids=rank_ids,
            compact=k_write_compact,
        )
        q = jax.lax.bitcast_convert_type(q_bytes, jnp.float8_e4m3fn)

        select = make_select_topk(
            topk,
            kv_layout,
            chunk_tokens=chunk_tokens,
            small_bucket=small_bucket,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            mixed_bkvp=mixed_bkvp,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
        )

        flat = dcp_token_parallel.indexer_topk(
            layout,
            mesh,
            DCP_AXIS_NAME,
            cache_kv,
            q,
            weights,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            select_topk=select,
            rank_ids=rank_ids,
        )
        return cache_kv, flat

    def streamidx_topk_tp(
        q_bytes: jax.Array,
        weights: jax.Array,
        cache_kv: jax.Array,
        k_packed: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        after: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _run(
            q_bytes,
            weights,
            cache_kv,
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            after,
            None,
        )

    def streamidx_topk_tp_rank(
        q_bytes: jax.Array,
        weights: jax.Array,
        cache_kv: jax.Array,
        k_packed: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        after: jax.Array,
        rank_ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return _run(
            q_bytes,
            weights,
            cache_kv,
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            after,
            rank_ids,
        )

    sharded, rep = P(DCP_AXIS_NAME), P()
    in_specs = (rep, rep, sharded, rep, rep, rep, rep, rep, sharded)
    if rank_tensor is not None:
        in_specs += (sharded,)
    op = pcp_streaming_jax_op(
        f"pallas::streamidx_topk_tp_{indexer.k_cache.prefix.replace('.', '_')}",
        streamidx_topk_tp if rank_tensor is None else streamidx_topk_tp_rank,
        donate_argnums=(2,),
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=(sharded, sharded),
    )

    def _fake(q_bytes, weights, cache_kv, *rest, **kwargs):
        del weights, rest, kwargs
        return (
            torch.empty_like(cache_kv),
            q_bytes.new_empty((q_bytes.shape[0] // dcp, topk), dtype=torch.int32),
        )

    op.register_fake(_fake)

    def run(*args: torch.Tensor) -> torch.Tensor:
        if rank_tensor is not None:
            args = (*args, rank_tensor)
        new_cache, flat = op(*args)
        args[2].copy_(new_cache)
        return flat

    run.layout = layout  # type: ignore[attr-defined]
    return run


def indexer_rows(
    indexer: Any,
    q_bytes: torch.Tensor,
    weights: torch.Tensor,
    kv_cache: torch.Tensor,
    k_packed: torch.Tensor,
    seq_lens: torch.Tensor,
    page_indices: torch.Tensor,
    cu_q_lens: torch.Tensor,
    distribution: torch.Tensor,
) -> torch.Tensor:
    """`[split_rows + T / dcp, topk]`: the DCP sub-batch's rank-local top-k,
    then this rank's token-parallel rows (see the module docstring)."""
    s = indexer.topk_op_tp.layout.split_rows
    seq_s, qsl_s, dist_s = dcp_rows_metadata(seq_lens, cu_q_lens, distribution, s)
    head = indexer.topk_op_dcp(
        q_bytes[:s],
        weights[:s],
        kv_cache,
        k_packed[:s],
        seq_s,
        page_indices,
        qsl_s,
        dist_s,
    )
    tail = indexer.topk_op_tp(
        q_bytes,
        weights,
        kv_cache,
        k_packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        head,
    )
    return torch.cat([head, tail])
