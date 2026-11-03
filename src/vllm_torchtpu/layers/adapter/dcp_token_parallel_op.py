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
from collections.abc import Callable
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
GATHER_CAPS = (2, 4, 8, 12, 16, 20, 24, 32)


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
    caps_env = envs.TPU_DCP_PREFILL_GATHER_CAPS.strip()
    base_caps = (
        tuple(int(x) for x in caps_env.split(",") if x.strip())
        if caps_env
        else GATHER_CAPS
    )
    return dcp_token_parallel.TokenParallelLayout(
        dcp_size=dcp,
        interleave_size=int(parallel.cp_kv_cache_interleave_size),
        page_size=page_size,
        split_rows=-(-max_num_seqs // SPLIT_ROWS_ALIGN) * SPLIT_ROWS_ALIGN,
        max_pages=max_pages,
        gather_caps=tuple(c for c in base_caps if c < max_pages) + (max_pages,),
    )


def check_config(vllm_config: Any, *, q_replicate: bool) -> None:
    """Raise for a config the split does not serve."""
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL: speculative decoding is not "
            "supported (the drafter is traced once for every bucket)."
        )
    if envs.TPU_DCP_MLA_UV_BEFORE_MERGE or envs.TPU_DCP_MLA_OPROJ_ALL_REDUCE:
        raise NotImplementedError(
            "TPU_DCP_PREFILL_TOKEN_PARALLEL returns merged latent outputs; it "
            "does not take TPU_DCP_MLA_UV_BEFORE_MERGE / _OPROJ_ALL_REDUCE."
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


def _dcp_rows_metadata_jax(
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """JAX counterpart of `dcp_rows_metadata` for fused sub-batch execution."""
    qsl = jnp.minimum(query_start_loc, split_rows)
    cut = (query_start_loc[1:] - query_start_loc[:-1]) - (qsl[1:] - qsl[:-1])
    req = jnp.arange(seq_lens.shape[0], dtype=distribution.dtype)
    n = jnp.sum((query_start_loc[:-1] < split_rows) & (req < distribution[2])).astype(
        distribution.dtype
    )
    dist = jnp.stack(
        [jnp.minimum(distribution[0], n), jnp.minimum(distribution[1], n), n]
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
    split_q: bool = False,
    mixed_body_levers: tuple[str, ...] = (),
) -> Callable[..., jax.Array]:
    """The single-device sparse MLA kernel as the split's `attend`: one ragged
    "request" of `num_rows` rows; with no page table the kernel reads the
    top-k as flat rows of the gathered buffer. Exact softmax over the row's
    whole top-k, so no log-sum-exp is returned."""
    from vllm_torchtpu.layers.core.attention_interface import sparse_mla_chunk_kwargs

    packed = {"packed_rope": True} if packed_rope else {}

    def attend(q, g_nope, g_rope, topk, num_rows, q_pe=None):
        split_kw = {"q_pe": q_pe} if split_q and q_pe is not None else {}
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
            mixed_body_levers=mixed_body_levers,
            **sparse_mla_chunk_kwargs(),
            **packed,
            **split_kw,
        )
        return out if out.shape[-1] == lkv_dim else out[..., :lkv_dim]

    return attend


def make_select_topk(
    topk: int,
    kv_layout: KVLayout,
    *,
    chunk_tokens: int | None,
    small_bucket: tuple[int | None, int | None, int],
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
    from vllm_torchtpu.layers.core.attention_interface import (
        _mixed_body_levers,
        sparse_mla_attention_dcp,
    )

    rank_tensor = _sharded_rank_ids()
    uv_in_tp = envs.TPU_DCP_PREFILL_TP_UV_BEFORE_A2A and not hasattr(
        layer, "W_UV_scale"
    )
    split_q = envs.TPU_MLA_SPARSE_SPLIT_Q
    mixed_levers = _mixed_body_levers(
        envs.TPU_DCP_MLA_MIXED_BODY, envs.TPU_DCP_MLA_MIXED_KERNEL_V2
    )
    fused_subbatch = envs.TPU_DCP_PREFILL_FUSED_SUBBATCH

    def _run(
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q: jax.Array,
        q_pe: jax.Array | None,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        topk_in: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        after: jax.Array | None,
        rank_ids: jax.Array | None,
        w_uv: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if after is not None:
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

        def write_kv(nope, rope, kv_c, k_pe_in, sl, bt, qsl, rank):
            return update_sparse_mla_kv_cache_dcp(
                nope,
                rope,
                kv_c,
                k_pe_in,
                sl,
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
            split_q=q_pe is not None,
            mixed_body_levers=mixed_levers,
        )
        read_pages = (
            functools.partial(page_gather.gather_pages, row_width=128),
            functools.partial(
                page_gather.gather_pages, row_width=rope_spec.token_bytes // 4
            ),
        )

        if not fused_subbatch:
            return dcp_token_parallel.sparse_mla_attention(
                layout,
                mesh,
                DCP_AXIS_NAME,
                kv_cache_nope,
                kv_cache_rope,
                q,
                kv_c_normed,
                k_pe,
                topk_in,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                write_kv=write_kv,
                attend=attend,
                rank_ids=rank_ids,
                q_own_rows=q_replicated,
                read_pages=read_pages,
                w_uv=w_uv,
                q_pe=q_pe,
            )

        assert q_pe is not None
        s = layout.split_rows
        num_tokens = kv_c_normed.shape[0]
        rows = layout.rows_per_rank(num_tokens)
        shd, rp = P(DCP_AXIS_NAME), P()

        def _write_once(nope, rope, kv_c, k_pe_in, sl, bt, qsl, *r_arg):
            rank = (
                r_arg[0][0]
                if r_arg
                else dcp_token_parallel.rank_as_data(DCP_AXIS_NAME, dcp)
            )
            return write_kv(nope, rope, kv_c, k_pe_in, sl, bt, qsl, rank)

        r_extra = () if rank_ids is None else (rank_ids,)
        kv_cache_nope, kv_cache_rope = jax.shard_map(
            _write_once,
            mesh=mesh,
            in_specs=(shd, shd, rp, rp, rp, rp, rp) + (shd,) * len(r_extra),
            out_specs=(shd, shd),
            check_vma=False,
        )(
            kv_cache_nope,
            kv_cache_rope,
            kv_c_normed,
            k_pe,
            seq_lens,
            block_tables,
            query_start_loc,
            *r_extra,
        )

        seq_s, qsl_s, dist_s = _dcp_rows_metadata_jax(
            seq_lens, query_start_loc, request_distribution, s
        )
        if q_replicated:
            q_dcp = jax.shard_map(
                lambda x: x[:s], mesh=mesh, in_specs=shd, out_specs=rp, check_vma=False
            )(q)
            pe_dcp = jax.shard_map(
                lambda x: x[:s], mesh=mesh, in_specs=shd, out_specs=rp, check_vma=False
            )(q_pe)
            q_tp = jax.shard_map(
                lambda x: x[s:], mesh=mesh, in_specs=shd, out_specs=shd, check_vma=False
            )(q)
            pe_tp = jax.shard_map(
                lambda x: x[s:], mesh=mesh, in_specs=shd, out_specs=shd, check_vma=False
            )(q_pe)
        else:
            q_dcp = jax.shard_map(
                lambda x: jax.lax.all_gather(x[:s], DCP_AXIS_NAME, axis=1, tiled=True),
                mesh=mesh,
                in_specs=shd,
                out_specs=rp,
                check_vma=False,
            )(q)
            pe_dcp = jax.shard_map(
                lambda x: jax.lax.all_gather(x[:s], DCP_AXIS_NAME, axis=1, tiled=True),
                mesh=mesh,
                in_specs=shd,
                out_specs=rp,
                check_vma=False,
            )(q_pe)
            q_tp, pe_tp = q, q_pe

        head_topk = jax.shard_map(
            lambda x: x[:s], mesh=mesh, in_specs=shd, out_specs=shd, check_vma=False
        )(topk_in)
        tail_topk = jax.shard_map(
            lambda x: x[s : s + rows],
            mesh=mesh,
            in_specs=shd,
            out_specs=shd,
            check_vma=False,
        )(topk_in)

        _, _, partial, lse = sparse_mla_attention_dcp(
            q_dcp,
            pe_dcp,
            kv_c_normed[:s],
            k_pe[:s],
            kv_cache_nope,
            kv_cache_rope,
            head_topk,
            seq_s,
            block_tables,
            qsl_s,
            dist_s,
            mesh,
            nope_spec,
            rope_spec,
            sm_scale=layer.scale,
            k_scale=dequant,
            dcp_size=dcp,
            interleave_size=il,
            dcp_rank_ids=rank_ids,
            merge_pieces=False,
            skip_kv_write=True,
        )
        _, _, tail_out = dcp_token_parallel.sparse_mla_attention(
            layout,
            mesh,
            DCP_AXIS_NAME,
            kv_cache_nope,
            kv_cache_rope,
            q_tp,
            kv_c_normed,
            k_pe,
            tail_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            write_kv=None,
            attend=attend,
            rank_ids=rank_ids,
            q_own_rows=q_replicated,
            read_pages=read_pages,
            w_uv=w_uv,
            q_pe=pe_tp,
        )

        def _merge_and_join(part, lse_part, tail_local, *w_uv_arg):
            lses = jax.lax.all_gather(lse_part, DCP_AXIS_NAME, axis=0, tiled=False)
            m = jax.nn.logsumexp(lses, axis=0)
            safe_m = jnp.where(jnp.isfinite(m), m, 0.0)
            w = jnp.where(jnp.isfinite(lse_part), jnp.exp(lse_part - safe_m), 0.0)
            weighted = (part.astype(jnp.float32) * w[..., None]).astype(part.dtype)
            head_local = jax.lax.psum_scatter(
                weighted, DCP_AXIS_NAME, scatter_dimension=1, tiled=True
            )
            if w_uv_arg:
                head_local = (
                    jnp.einsum(
                        "hrl,hlv->hrv",
                        jnp.swapaxes(head_local, 0, 1),
                        w_uv_arg[0].astype(head_local.dtype),
                    )
                    .swapaxes(0, 1)
                    .astype(tail_local.dtype)
                )
            return jnp.concatenate(
                [head_local.astype(tail_local.dtype), tail_local[: num_tokens - s]],
                axis=0,
            )

        uv_extra = () if w_uv is None else (w_uv,)
        full_out = jax.shard_map(
            _merge_and_join,
            mesh=mesh,
            in_specs=(shd, shd, shd) + (shd,) * len(uv_extra),
            out_specs=shd,
            check_vma=False,
        )(partial, lse, tail_out, *uv_extra)
        return kv_cache_nope, kv_cache_rope, full_out

    use_split_op = split_q or fused_subbatch
    sharded, rep = P(DCP_AXIS_NAME), P()

    def _build_wrapper(has_rank: bool, has_uv: bool):
        if fused_subbatch:

            def _fn_fused(
                kv_cache_nope: jax.Array,
                kv_cache_rope: jax.Array,
                ql_nope: jax.Array,
                q_pe: jax.Array,
                kv_c_normed: jax.Array,
                k_pe: jax.Array,
                topk_rows: jax.Array,
                seq_lens: jax.Array,
                block_tables: jax.Array,
                query_start_loc: jax.Array,
                request_distribution: jax.Array,
                *extra: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                r_id = extra[0] if has_rank else None
                uv_w = extra[-1] if has_uv else None
                return _run(
                    kv_cache_nope,
                    kv_cache_rope,
                    ql_nope,
                    q_pe,
                    kv_c_normed,
                    k_pe,
                    topk_rows,
                    seq_lens,
                    block_tables,
                    query_start_loc,
                    request_distribution,
                    None,
                    r_id,
                    uv_w,
                )

            if (has_rank, has_uv) == (False, False):

                def w0(
                    n: jax.Array,
                    r: jax.Array,
                    qn: jax.Array,
                    qp: jax.Array,
                    kc: jax.Array,
                    kp: jax.Array,
                    tk: jax.Array,
                    sl: jax.Array,
                    bt: jax.Array,
                    qsl: jax.Array,
                    rd: jax.Array,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _fn_fused(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd)

                return w0
            if (has_rank, has_uv) == (False, True):

                def w1(
                    n: jax.Array,
                    r: jax.Array,
                    qn: jax.Array,
                    qp: jax.Array,
                    kc: jax.Array,
                    kp: jax.Array,
                    tk: jax.Array,
                    sl: jax.Array,
                    bt: jax.Array,
                    qsl: jax.Array,
                    rd: jax.Array,
                    uv: jax.Array,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _fn_fused(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, uv)

                return w1
            if (has_rank, has_uv) == (True, False):

                def w2(
                    n: jax.Array,
                    r: jax.Array,
                    qn: jax.Array,
                    qp: jax.Array,
                    kc: jax.Array,
                    kp: jax.Array,
                    tk: jax.Array,
                    sl: jax.Array,
                    bt: jax.Array,
                    qsl: jax.Array,
                    rd: jax.Array,
                    rk: jax.Array,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _fn_fused(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, rk)

                return w2

            def w3(
                n: jax.Array,
                r: jax.Array,
                qn: jax.Array,
                qp: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                rd: jax.Array,
                rk: jax.Array,
                uv: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _fn_fused(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, rk, uv)

            return w3

        if use_split_op:
            if (has_rank, has_uv) == (False, False):

                def s0(
                    n: jax.Array,
                    r: jax.Array,
                    qn: jax.Array,
                    qp: jax.Array,
                    kc: jax.Array,
                    kp: jax.Array,
                    tk: jax.Array,
                    sl: jax.Array,
                    bt: jax.Array,
                    qsl: jax.Array,
                    rd: jax.Array,
                    af: jax.Array,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _run(
                        n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, af, None, None
                    )

                return s0
            if (has_rank, has_uv) == (False, True):

                def s1(
                    n: jax.Array,
                    r: jax.Array,
                    qn: jax.Array,
                    qp: jax.Array,
                    kc: jax.Array,
                    kp: jax.Array,
                    tk: jax.Array,
                    sl: jax.Array,
                    bt: jax.Array,
                    qsl: jax.Array,
                    rd: jax.Array,
                    af: jax.Array,
                    uv: jax.Array,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _run(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, af, None, uv)

                return s1
            if (has_rank, has_uv) == (True, False):

                def s2(
                    n: jax.Array,
                    r: jax.Array,
                    qn: jax.Array,
                    qp: jax.Array,
                    kc: jax.Array,
                    kp: jax.Array,
                    tk: jax.Array,
                    sl: jax.Array,
                    bt: jax.Array,
                    qsl: jax.Array,
                    rd: jax.Array,
                    af: jax.Array,
                    rk: jax.Array,
                ) -> tuple[jax.Array, jax.Array, jax.Array]:
                    return _run(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, af, rk, None)

                return s2

            def s3(
                n: jax.Array,
                r: jax.Array,
                qn: jax.Array,
                qp: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                rd: jax.Array,
                af: jax.Array,
                rk: jax.Array,
                uv: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(n, r, qn, qp, kc, kp, tk, sl, bt, qsl, rd, af, rk, uv)

            return s3

        if (has_rank, has_uv) == (False, False):

            def c0(
                n: jax.Array,
                r: jax.Array,
                q: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                rd: jax.Array,
                af: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(n, r, q, None, kc, kp, tk, sl, bt, qsl, rd, af, None, None)

            return c0
        if (has_rank, has_uv) == (False, True):

            def c1(
                n: jax.Array,
                r: jax.Array,
                q: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                rd: jax.Array,
                af: jax.Array,
                uv: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(n, r, q, None, kc, kp, tk, sl, bt, qsl, rd, af, None, uv)

            return c1
        if (has_rank, has_uv) == (True, False):

            def c2(
                n: jax.Array,
                r: jax.Array,
                q: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                rd: jax.Array,
                af: jax.Array,
                rk: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(n, r, q, None, kc, kp, tk, sl, bt, qsl, rd, af, rk, None)

            return c2

        def c3(
            n: jax.Array,
            r: jax.Array,
            q: jax.Array,
            kc: jax.Array,
            kp: jax.Array,
            tk: jax.Array,
            sl: jax.Array,
            bt: jax.Array,
            qsl: jax.Array,
            rd: jax.Array,
            af: jax.Array,
            rk: jax.Array,
            uv: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _run(n, r, q, None, kc, kp, tk, sl, bt, qsl, rd, af, rk, uv)

        return c3

    in_specs = (sharded, sharded, sharded)
    if use_split_op:
        in_specs += (sharded,)
    in_specs += (rep, rep, sharded, rep, rep, rep, rep)
    if not fused_subbatch:
        in_specs += (sharded,)
    if rank_tensor is not None:
        in_specs += (sharded,)
    if uv_in_tp:
        in_specs += (sharded,)
    fn = _build_wrapper(rank_tensor is not None, uv_in_tp)
    op = pcp_streaming_jax_op(
        f"pallas::sparse_mla_tp_{layer.layer_name.replace('.', '_')}"
        + ("_qrep" if q_replicated else "")
        + ("_uv" if uv_in_tp else "")
        + ("_sq" if use_split_op else "")
        + ("_fsb" if fused_subbatch else ""),
        fn,
        donate_argnums=(0, 1),
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=(sharded, sharded, sharded),
    )
    num_heads = layer.num_heads
    out_dim = layer.v_head_dim if uv_in_tp else layer.kv_lora_rank

    def _fake(kv_cache_nope, kv_cache_rope, q, *rest, **kwargs):
        del kwargs
        kv_c_normed = rest[1] if use_split_op else rest[0]
        return (
            torch.empty_like(kv_cache_nope),
            torch.empty_like(kv_cache_rope),
            q.new_empty((kv_c_normed.shape[0], num_heads, out_dim)),
        )

    op.register_fake(_fake)

    def run(
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        q: torch.Tensor,
        q_pe: torch.Tensor | None,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        topk_in: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_distribution: torch.Tensor,
        after: torch.Tensor | None,
        w_uv: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nope_cache, rope_cache = kv_cache
        q_args = (q, q_pe) if use_split_op else (q,)
        after_args = () if fused_subbatch else (after,)
        new_nope, new_rope, out = op(
            nope_cache,
            rope_cache,
            *q_args,
            kv_c_normed,
            k_pe,
            topk_in,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            *after_args,
            *(() if rank_tensor is None else (rank_tensor,)),
            *(() if w_uv is None else (w_uv,)),
        )
        nope_cache.copy_(new_nope)
        rope_cache.copy_(new_rope)
        return out

    run.layout = layout  # type: ignore[attr-defined]
    run.q_replicated = q_replicated  # type: ignore[attr-defined]
    run.uv_in_tp = uv_in_tp  # type: ignore[attr-defined]
    run.split_q = use_split_op  # type: ignore[attr-defined]
    run.fused_subbatch = fused_subbatch  # type: ignore[attr-defined]
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
        _retain_rank,
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
    if getattr(tp_op, "fused_subbatch", False):
        w_uv_arg = None
        if getattr(tp_op, "uv_in_tp", False):
            w_uv_arg = (
                layer.W_UV.to(ql_nope.dtype)
                if layer.W_UV.dtype != ql_nope.dtype
                else layer.W_UV
            )
        out = tp_op(
            kv_cache,
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            topk_rows,
            md.seq_lens,
            md.block_tables,
            md.query_start_loc,
            md.request_distribution,
            None,
            w_uv_arg,
        )
        return _retain_rank(out)

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
    w_uv_arg = None
    if getattr(tp_op, "uv_in_tp", False):
        w_uv_arg = (
            layer.W_UV.to(head.dtype) if layer.W_UV.dtype != head.dtype else layer.W_UV
        )
        head = (
            torch.bmm(head.transpose(0, 1), w_uv_arg).transpose(0, 1).to(ql_nope.dtype)
        )
    # The token-parallel op rewrites the caches in place; it must not start
    # before the sub-batch has read them (`_after_reads`).
    if getattr(tp_op, "split_q", False):
        q_tp = ql_nope[s:] if q_replicated else ql_nope
        pe_tp = q_pe[s:] if q_replicated else q_pe
    else:
        q_cat = torch.cat([ql_nope, q_pe], dim=-1)
        q_tp = q_cat[s:] if q_replicated else q_cat
        pe_tp = None
    tail = tp_op(
        kv_cache,
        q_tp,
        pe_tp,
        kv_c_normed,
        k_pe,
        topk_rows[s : s + rows],
        md.seq_lens,
        md.block_tables,
        md.query_start_loc,
        md.request_distribution,
        lse,
        w_uv_arg,
    )
    return torch.cat([head, tail[: num_tokens - s]])


def build_indexer_op(
    indexer: Any,
    layout: dcp_token_parallel.TokenParallelLayout,
    q_own_rows: bool = False,
):
    """The token-parallel top-k op of a DCP indexer: `(q_bytes, weights,
    kv_cache, k_packed, seq_lens, page_indices, cu_q_lens, distribution) ->
    [T / dcp, topk]` this rank's rows as flat gather-buffer rows. Writes every
    row's K into the cache first, as the DCP op does."""
    from vllm_torchtpu.distributed.dcp import dcp_lane_operand
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk_dcp
    from vllm_torchtpu.layers.adapter.custom_ops import mla_attention_op as mla_op

    dcp, il = layout.dcp_size, layout.interleave_size
    s = layout.split_rows
    mesh = get_or_create_dcp_mesh(axis_name=DCP_AXIS_NAME)
    topk = indexer.topk_tokens
    chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
    hist_merge = envs.TPU_DCP_INDEXER_HIST_MERGE
    a2a_merge = envs.TPU_DCP_INDEXER_A2A_MERGE
    a2a_min_tokens = envs.TPU_DCP_INDEXER_A2A_MIN_TOKENS
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
    fused_subbatch = envs.TPU_DCP_PREFILL_FUSED_SUBBATCH
    global_group = (
        mla_op.register_indexer_layer(indexer.k_cache.prefix)
        if (fused_subbatch and envs.TPU_IDX_TOPK_SCHED_GROUPS)
        else None
    )
    rank_tensor = _sharded_rank_ids()
    small_ag = bool(
        fused_subbatch
        and envs.TPU_DCP_INDEXER_SMALL_AG
        and envs.TPU_SC_TOPK_DCP_MERGE_FLAT
        and rank_tensor is not None
    )
    lane_tensor = dcp_lane_operand() if small_ag else None
    sharded, rep = P(DCP_AXIS_NAME), P()

    def _run(
        q_bytes: jax.Array,
        weights: jax.Array,
        cache_kv: jax.Array,
        k_packed: jax.Array,
        seq_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        after: jax.Array | None,
        rank_ids: jax.Array | None,
        lane_ids: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        if after is not None:
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

        if not fused_subbatch:
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
                q_own_rows=q_own_rows,
            )
            return cache_kv, flat

        if q_own_rows:
            q_dcp = jax.shard_map(
                lambda x: x[:s],
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(q)
            w_dcp = jax.shard_map(
                lambda x: x[:s],
                mesh=mesh,
                in_specs=sharded,
                out_specs=rep,
                check_vma=False,
            )(weights)
            q_tp = jax.shard_map(
                lambda x: x[s:],
                mesh=mesh,
                in_specs=sharded,
                out_specs=sharded,
                check_vma=False,
            )(q)
            w_tp = jax.shard_map(
                lambda x: x[s:],
                mesh=mesh,
                in_specs=sharded,
                out_specs=sharded,
                check_vma=False,
            )(weights)
        else:
            q_dcp, w_dcp = q[:s], weights[:s]
            q_tp, w_tp = q, weights

        seq_s, qsl_s, dist_s = _dcp_rows_metadata_jax(
            seq_lens, cu_q_lens, distribution, s
        )
        bkvp_dcp, bq_dcp = mla_op._streamidx_block_sizes(
            s, *small_bucket, mixed_bkvp, decode_pack
        )
        large = mla_op._dcp_indexer_a2a_for_bucket(s, True, a2a_min_tokens)
        head = streamindex_topk_dcp(
            q_dcp,
            w_dcp,
            cache_kv,
            seq_s,
            page_indices,
            qsl_s,
            dist_s,
            mesh=mesh,
            k=topk,
            compression_ratio=1,
            dcp_size=dcp,
            interleave_size=il,
            num_kv_pages_per_block=bkvp_dcp,
            num_queries_per_block=bq_dcp,
            kv_layout=kv_layout,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            chunk_tokens=chunk_tokens,
            a2a_merge=large and a2a_merge and not hist_merge,
            hist_merge=large and hist_merge,
            global_scheduling_group_id=global_group,
            dcp_rank_ids=rank_ids,
            dcp_lane_ids=lane_ids,
            small_all_gather=small_ag,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
        )
        tail = dcp_token_parallel.indexer_topk(
            layout,
            mesh,
            DCP_AXIS_NAME,
            cache_kv,
            q_tp,
            w_tp,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
            select_topk=select,
            rank_ids=rank_ids,
            q_own_rows=q_own_rows,
        )
        full = jax.shard_map(
            lambda h, t: jnp.concatenate([h, t], axis=0),
            mesh=mesh,
            in_specs=(sharded, sharded),
            out_specs=sharded,
            check_vma=False,
        )(head, tail)
        return cache_kv, full

    def _build_idx_wrapper():
        if fused_subbatch:
            if rank_tensor is None:

                def f0(
                    qb: jax.Array,
                    w: jax.Array,
                    kv: jax.Array,
                    kp: jax.Array,
                    sl: jax.Array,
                    pi: jax.Array,
                    cql: jax.Array,
                    dist: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _run(qb, w, kv, kp, sl, pi, cql, dist, None, None, None)

                return f0
            if lane_tensor is None:

                def f1(
                    qb: jax.Array,
                    w: jax.Array,
                    kv: jax.Array,
                    kp: jax.Array,
                    sl: jax.Array,
                    pi: jax.Array,
                    cql: jax.Array,
                    dist: jax.Array,
                    rk: jax.Array,
                ) -> tuple[jax.Array, jax.Array]:
                    return _run(qb, w, kv, kp, sl, pi, cql, dist, None, rk, None)

                return f1

            def f2(
                qb: jax.Array,
                w: jax.Array,
                kv: jax.Array,
                kp: jax.Array,
                sl: jax.Array,
                pi: jax.Array,
                cql: jax.Array,
                dist: jax.Array,
                rk: jax.Array,
                ln: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return _run(qb, w, kv, kp, sl, pi, cql, dist, None, rk, ln)

            return f2

        if rank_tensor is None:

            def u0(
                qb: jax.Array,
                w: jax.Array,
                kv: jax.Array,
                kp: jax.Array,
                sl: jax.Array,
                pi: jax.Array,
                cql: jax.Array,
                dist: jax.Array,
                af: jax.Array,
            ) -> tuple[jax.Array, jax.Array]:
                return _run(qb, w, kv, kp, sl, pi, cql, dist, af, None, None)

            return u0

        def u1(
            qb: jax.Array,
            w: jax.Array,
            kv: jax.Array,
            kp: jax.Array,
            sl: jax.Array,
            pi: jax.Array,
            cql: jax.Array,
            dist: jax.Array,
            af: jax.Array,
            rk: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            return _run(qb, w, kv, kp, sl, pi, cql, dist, af, rk, None)

        return u1

    q_spec = sharded if q_own_rows else rep
    in_specs = (q_spec, q_spec, sharded, rep, rep, rep, rep, rep)
    if not fused_subbatch:
        in_specs += (sharded,)
    if rank_tensor is not None:
        in_specs += (sharded,)
    if lane_tensor is not None:
        in_specs += (sharded,)
    op = pcp_streaming_jax_op(
        f"pallas::streamidx_topk_tp_{indexer.k_cache.prefix.replace('.', '_')}"
        + ("_qown" if q_own_rows else "")
        + ("_fsb" if fused_subbatch else ""),
        _build_idx_wrapper(),
        donate_argnums=(2,),
        mesh=mesh,
        input_partition_specs=in_specs,
        output_partition_specs=(sharded, sharded),
    )

    def _fake(q_bytes, weights, cache_kv, *rest, **kwargs):
        del weights, rest, kwargs
        if fused_subbatch:
            out_rows = (
                q_bytes.shape[0]
                if q_own_rows
                else layout.split_rows + q_bytes.shape[0] // dcp
            )
        else:
            out_rows = q_bytes.shape[0] if q_own_rows else q_bytes.shape[0] // dcp
        return (
            torch.empty_like(cache_kv),
            q_bytes.new_empty((out_rows, topk), dtype=torch.int32),
        )

    op.register_fake(_fake)

    def run(*args: torch.Tensor) -> torch.Tensor:
        extra = ()
        if rank_tensor is not None:
            extra = (*extra, rank_tensor)
        if lane_tensor is not None:
            extra = (*extra, lane_tensor)
        new_cache, flat = op(*args, *extra)
        args[2].copy_(new_cache)
        return flat

    run.layout = layout  # type: ignore[attr-defined]
    run.q_own_rows = q_own_rows  # type: ignore[attr-defined]
    run.fused_subbatch = fused_subbatch  # type: ignore[attr-defined]
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
    if getattr(indexer.topk_op_tp, "fused_subbatch", False):
        return indexer.topk_op_tp(
            q_bytes,
            weights,
            kv_cache,
            k_packed,
            seq_lens,
            page_indices,
            cu_q_lens,
            distribution,
        )
    s = indexer.topk_op_tp.layout.split_rows
    q_own = getattr(indexer.topk_op_tp, "q_own_rows", False)
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
        q_bytes[s:] if q_own else q_bytes,
        weights[s:] if q_own else weights,
        kv_cache,
        k_packed,
        seq_lens,
        page_indices,
        cu_q_lens,
        distribution,
        head,
    )
    return torch.cat([head, tail])
