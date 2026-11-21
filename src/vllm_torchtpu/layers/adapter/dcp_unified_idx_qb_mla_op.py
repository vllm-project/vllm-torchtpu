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
"""Unified token-parallel indexer + in-branch q_b + sparse MLA op for DCP prefill.

Extends `dcp_unified_indexer_mla_op` by deferring `fused_qb_rope_absorb_split_qpe`
and `lead_mla` inside the shared `_switch_on_cap` branch on active indexer layers,
so pure-TensorCore `q_b` overlaps final `sc_topk_direct` chunks and SparseCore
context KV gathers while keeping `0` extra `lax.switch` outputs.
"""

# Avoid `from __future__ import annotations`: `torch_tpu` `pallas._verify_signature`
# checks `annotation is jax.Array` at op registration time and rejects PEP 563 strings.

import functools
import weakref
from typing import Any

import jax
import jax.numpy as jnp
import torch
from jax.sharding import PartitionSpec as P
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_torchtpu import envs
from vllm_torchtpu.distributed.dcp import (
    DCP_AXIS_NAME,
    dcp_lane_operand,
    dcp_rank_operand,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import (
    KVLayout,
    streamindex_topk_dcp,
)
from vllm_torchtpu.kernels.experimental.pcp_streaming_rpa.vllm_adapter import (
    pcp_streaming_jax_op,
)
from vllm_torchtpu.kernels.mla import dcp_token_parallel, page_gather
from vllm_torchtpu.kernels.mla.fused_qb_absorb import fused_qb_rope_absorb_split_qpe
from vllm_torchtpu.kernels.mla.kv_cache_utils import (
    update_sparse_mla_kv_cache_dcp,
    update_sparse_mla_kv_cache_dcp_sharded,
)
from vllm_torchtpu.kernels.mla.sparse import kernel as sparse_mla_kernel
from vllm_torchtpu.layers.adapter import (
    dcp_nonidx_qb_mla_op,
    dcp_token_parallel_op,
    dcp_unified_indexer_mla_op,
)
from vllm_torchtpu.layers.adapter.cp_mla_attention import (
    _retain_rank,  # noqa: SLF001 -- existing internal rank-retention helper
)
from vllm_torchtpu.layers.core.attention_interface import (
    _decode_body_levers,  # noqa: SLF001 -- existing internal decode-body lever resolver
    _mixed_body_levers,  # noqa: SLF001 -- existing internal mixed-body lever resolver
    dcp_mla_topk_cap,
    sparse_mla_chunk_kwargs,
)
from vllm_torchtpu.utils import get_shared_jax_op


def _dcp_rows_metadata_jax(
    seq_lens: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    split_rows: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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


def build_unified_indexer_in_branch_qb_op(
    layer: Any,
    impl: Any,
    indexer_op: Any,
    layout: dcp_token_parallel.TokenParallelLayout,
    *,
    q_replicated: bool,
    callbacks: dcp_unified_indexer_mla_op.UnifiedIndexerCallbacks,
):
    """Build a shared unified indexer + in-branch q_b + sparse MLA + o_proj op."""
    if not (
        envs.TPU_DCP_TP_UNIFIED_INDEXER_MLA
        and envs.TPU_DCP_TP_UNIFIED_INDEXER_IN_BRANCH_QB
        and envs.TPU_DCP_TP_PREFILL_LOCAL_OPROJ
        and envs.TPU_DCP_RANK_OPERAND
        and q_replicated
        and layout.dcp_size == get_tensor_model_parallel_world_size()
        and not hasattr(layer, "W_UV_scale")
    ):
        return None

    mesh = impl._dcp_mesh  # noqa: SLF001 -- backend-owned DCP mesh attribute
    assert mesh is not None
    dcp, il = impl.dcp_size, impl.dcp_interleave_size
    if (dcp, il) != (layout.dcp_size, layout.interleave_size):
        raise ValueError(f"backend DCP ({dcp}, {il}) does not match {layout}")

    k_scale = None
    if layer.kv_cache_quantized_dtype is not None:
        _, k_scale, _ = impl._get_kv_scales(layer)  # noqa: SLF001 -- backend KV scale lookup
    dequant = float(k_scale) if k_scale is not None else 1.0
    sm_scale = float(layer.scale)
    mixed_levers = _mixed_body_levers(
        envs.TPU_DCP_MLA_MIXED_BODY, envs.TPU_DCP_MLA_MIXED_KERNEL_V2
    )
    decode_levers = _decode_body_levers(envs.TPU_MLA_DECODE_BODY_LEVERS)
    mla_chunk_tokens = envs.TPU_SPARSE_MLA_CHUNK_TOKENS
    tie_after_idx_max_cap = envs.TPU_DCP_TP_UNIFIED_TIE_AFTER_IDX_MAX_CAP
    epilogue_3d = envs.TPU_DCP_TP_EPILOGUE_3D_SHIFT_QUANT

    topk = int(indexer_op.topk_tokens)
    idx_chunk_tokens = envs.TPU_STREAMIDX_CHUNK_TOKENS
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
    shared_meta = bool(getattr(indexer_op, "_shared_meta", False))
    k_write_compact = envs.TPU_INDEXER_K_WRITE_COMPACT
    global_group = (
        callbacks.register_indexer_layer(indexer_op.k_cache.prefix)
        if envs.TPU_IDX_TOPK_SCHED_GROUPS
        else None
    )
    rank_tensor = dcp_rank_operand()
    small_ag = bool(envs.TPU_DCP_INDEXER_SMALL_AG and envs.TPU_SC_TOPK_DCP_MERGE_FLAT)
    lane_tensor = dcp_lane_operand() if small_ag else None
    has_lane = lane_tensor is not None

    target_layer = layer
    num_heads = int(layer.num_heads)
    out_dim = int(layer.v_head_dim)
    del layer, impl, indexer_op

    layers: list[weakref.ref] = []
    shd, rep = P(DCP_AXIS_NAME), P()

    def _run(
        index_cache: jax.Array,
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q_idx_bytes: jax.Array,
        w_idx: jax.Array,
        k_packed: jax.Array,
        q_c_own: jax.Array,
        w_qb: jax.Array,
        s_qb: jax.Array,
        w_uk_t: jax.Array,
        cos_sin: jax.Array,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        seq_lens: jax.Array,
        idx_block_tables: jax.Array,
        mla_block_tables: jax.Array,
        query_start_loc: jax.Array,
        idx_distribution: jax.Array,
        mla_distribution: jax.Array,
        rank_ids: jax.Array,
        w_uv: jax.Array,
        w_o: jax.Array,
        s_o: jax.Array,
        lane_ids: jax.Array | None,
        *,
        kv_sharded: bool,
        k_sharded: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        live = [
            sharer
            for ref in layers
            if (sharer := ref()) is not None and hasattr(sharer, "mla_kv_spec")
        ]
        nope_spec, rope_spec = live[0].mla_kv_spec
        for other in live[1:]:
            if other.mla_kv_spec != (nope_spec, rope_spec):
                raise NotImplementedError(
                    f"{getattr(other, 'layer_name', other)} shares a unified "
                    f"indexer+in-branch-qb op with "
                    f"{getattr(live[0], 'layer_name', live[0])} "
                    f"but recorded different KV cache specs."
                )
        if nope_spec.page_size != layout.page_size:
            raise ValueError(
                f"sparse MLA cache page {nope_spec.page_size} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )
        if nope_spec.layout is not rope_spec.layout:
            raise ValueError("sparse MLA needs matching NoPE and RoPE layouts")

        writer, kv_layout = callbacks.resolve_kv_cache_mode(
            index_cache, k_packed, mode=kv_cache_mode
        )
        if kv_layout != KVLayout.SEQ_ALONG_LANE:
            raise NotImplementedError(
                "TPU_DCP_TP_UNIFIED_INDEXER_IN_BRANCH_QB needs the "
                f"token-along-lane index-K cache, got {kv_layout}"
            )
        if index_cache.shape[-1] != layout.page_size:
            raise ValueError(
                f"index-K page {index_cache.shape[-1]} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )

        idx_zigzag_kw = (
            {"zigzag_bands": layout.effective_zigzag_bands(k_packed.shape[0])}
            if k_sharded and layout.effective_zigzag_bands(k_packed.shape[0]) > 1
            else {}
        )
        index_cache = callbacks.scatter_indexer_k(
            index_cache,
            k_packed,
            seq_lens,
            idx_block_tables,
            query_start_loc,
            mesh=mesh,
            cp_size=dcp,
            interleave_size=il,
            axis_name=DCP_AXIS_NAME,
            writer=writer,
            kv_layout=kv_layout,
            rank_ids=rank_ids,
            compact=k_write_compact,
            k_sharded=k_sharded,
            **idx_zigzag_kw,
        )

        def _write_kv_once(nope, rope, kv_c, k_pe_in, sl, bt, qsl, rk):
            rank = rk[0]
            if kv_sharded:
                return update_sparse_mla_kv_cache_dcp_sharded(
                    nope,
                    rope,
                    kv_c,
                    k_pe_in,
                    sl,
                    bt,
                    qsl,
                    rank,
                    axis_name=DCP_AXIS_NAME,
                    nope_spec=nope_spec,
                    rope_spec=rope_spec,
                    dcp_size=dcp,
                    interleave_size=il,
                    zigzag_bands=layout.effective_zigzag_bands(kv_c.shape[0] * dcp),
                )
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

        kv_spec = shd if kv_sharded else rep
        kv_cache_nope, kv_cache_rope = jax.shard_map(
            _write_kv_once,
            mesh=mesh,
            in_specs=(shd, shd, kv_spec, kv_spec, rep, rep, rep, shd),
            out_specs=(shd, shd),
            check_vma=False,
        )(
            kv_cache_nope,
            kv_cache_rope,
            kv_c_normed,
            k_pe,
            seq_lens,
            mla_block_tables,
            query_start_loc,
            rank_ids,
        )

        s = layout.split_rows
        num_tokens = kv_c_normed.shape[0]
        q_idx = jax.lax.bitcast_convert_type(q_idx_bytes, jnp.float8_e4m3fn)
        q_idx_dcp = jax.shard_map(
            lambda x: x[:s], mesh=mesh, in_specs=shd, out_specs=rep, check_vma=False
        )(q_idx)
        w_idx_dcp = jax.shard_map(
            lambda x: x[:s], mesh=mesh, in_specs=shd, out_specs=rep, check_vma=False
        )(w_idx)
        q_idx_tp = jax.shard_map(
            lambda x: x[s:], mesh=mesh, in_specs=shd, out_specs=shd, check_vma=False
        )(q_idx)
        w_idx_tp = jax.shard_map(
            lambda x: x[s:], mesh=mesh, in_specs=shd, out_specs=shd, check_vma=False
        )(w_idx)

        seq_s, qsl_s, idx_dist_s = _dcp_rows_metadata_jax(
            seq_lens, query_start_loc, idx_distribution, s
        )
        bkvp_dcp, bq_dcp = callbacks.streamidx_block_sizes(
            s, *small_bucket, mixed_bkvp, decode_pack
        )
        bkvp_dcp, bq_dcp, lead_decode_as_mixed = (
            dcp_token_parallel_op.lead_streamidx_block_sizes(s, bkvp_dcp, bq_dcp)
        )
        large = callbacks.dcp_indexer_a2a_for_bucket(s, True, a2a_min_tokens)
        head_topk = streamindex_topk_dcp(
            q_idx_dcp,
            w_idx_dcp,
            index_cache,
            seq_s,
            idx_block_tables,
            qsl_s,
            idx_dist_s,
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
            chunk_tokens=idx_chunk_tokens,
            a2a_merge=large and a2a_merge and not hist_merge,
            hist_merge=large and hist_merge,
            global_scheduling_group_id=global_group,
            dcp_rank_ids=rank_ids,
            dcp_lane_ids=lane_ids,
            small_all_gather=small_ag,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
            decode_as_mixed=lead_decode_as_mixed,
        )

        cache_layout = nope_spec.layout.value
        lkv_dim = kv_c_normed.shape[-1]
        packed_rope_kwargs = {"packed_rope": True} if rope_spec.packed_rope else {}
        topk_cap = dcp_mla_topk_cap(dcp)
        if not (
            topk_cap
            and cache_layout == sparse_mla_kernel.SPARSECORE_CACHE_LAYOUT
            and head_topk.shape[-1] > topk_cap
        ):
            topk_cap = None
        lead_chunk_kwargs: dict[str, Any] = (
            {
                "gather_and_attention_chunk_size": envs.TPU_DCP_MLA_TOPK_CAP_CHUNK,
                "capped_live_rows": envs.TPU_SPARSE_MLA_LIVE_GATHER,
            }
            if topk_cap
            else dict(sparse_mla_chunk_kwargs())
        )
        if s <= 64 and s % 8 == 0:
            lead_chunk_kwargs["attention_kernel_batch_size"] = 8
        lead_live_widths_raw = (
            (128, 256, 384)
            if envs.TPU_DCP_MLA_MIXED_LIVE_WIDTHS == "256,384"
            else tuple(
                int(w) for w in envs.TPU_DCP_MLA_MIXED_LIVE_WIDTHS.split(",") if w
            )
        )
        lead_live_widths = tuple(
            w for w in lead_live_widths_raw if not topk_cap or w < topk_cap
        )
        decode_row_widths = (
            tuple(int(w) for w in envs.TPU_MLA_DECODE_ROW_WIDTHS.split(",") if w)
            if topk_cap
            else ()
        )
        decode_bucket = s <= sparse_mla_kernel.FUSED_PREP_MAX_TOKENS
        mask_in_kernel = envs.TPU_DCP_MLA_MASK_IN_KERNEL or (
            envs.TPU_DCP_MLA_MIXED_MASK_IN_KERNEL and not decode_bucket
        )
        split_q = envs.TPU_MLA_SPARSE_SPLIT_Q
        small_chunk_batch = envs.TPU_DCP_MLA_SMALL_CHUNK_BATCH
        small_chunk_tokens = envs.TPU_DCP_MLA_SMALL_CHUNK_TOKENS
        mixed_live = envs.TPU_DCP_MLA_MIXED_LIVE
        mixed_fused_prep = envs.TPU_DCP_MLA_MIXED_FUSED_PREP
        fused_prep = envs.TPU_DCP_MLA_FUSED_PREP
        num_pages = int(nope_spec.num_pages)

        def _lead_attend(
            ql_head: jax.Array,
            qpe_head: jax.Array,
            nope: jax.Array,
            rope: jax.Array,
            h_topk: jax.Array,
            bt: jax.Array,
            sl: jax.Array,
            qsl: jax.Array,
            dist: jax.Array,
        ) -> tuple[jax.Array, jax.Array]:
            _, qsl_m, dist_m = _dcp_rows_metadata_jax(sl, qsl, dist, s)
            local_bt = jnp.mod(bt, jnp.int32(num_pages))
            owns = None
            if not mask_in_kernel:
                owns = jnp.any(h_topk >= 0, axis=-1)
                h_topk = h_topk.at[:, 0].set(
                    jnp.where(owns, h_topk[:, 0], jnp.int32(0))
                )
            if split_q:
                q_arg: jax.Array = ql_head
                split_kwargs: dict[str, Any] = {
                    "q_pe": qpe_head,
                    "lse_lane_dense": True,
                }
            else:
                q_arg = jnp.concatenate([ql_head, qpe_head], axis=-1)
                split_kwargs = {}
            out, lse = sparse_mla_kernel.sparse_ragged_paged_attention(
                q_arg,
                nope,
                rope,
                h_topk,
                local_bt,
                qsl_m,
                dist_m,
                sm_scale=sm_scale,
                k_scale=dequant,
                return_lse=True,
                is_dcp_mode=True,
                cache_layout=cache_layout,
                topk_cap=topk_cap,
                small_chunk_batch_size=small_chunk_batch,
                small_chunk_tokens=small_chunk_tokens,
                small_chunk_gather_atoms=4 if ql_head.shape[0] >= 16 else 1,
                lse_mask_empty=mask_in_kernel,
                fused_window_prep=fused_prep,
                mixed_live_gather=mixed_live,
                mixed_live_widths=lead_live_widths,
                mixed_body_levers=mixed_levers,
                mixed_fused_prep=mixed_fused_prep,
                decode_row_widths=decode_row_widths,
                decode_body_levers=decode_levers,
                **lead_chunk_kwargs,
                **split_kwargs,
                **packed_rope_kwargs,
            )
            if not split_q and lse.ndim == 3:
                lse = lse[..., 0]
            if not mask_in_kernel:
                assert owns is not None
                lse = jnp.where(owns[:, None], lse, -jnp.inf)
            return out[..., :lkv_dim], lse

        select = dcp_token_parallel_op.make_select_topk(
            topk,
            kv_layout,
            chunk_tokens=idx_chunk_tokens,
            small_bucket=small_bucket,
            skip_empty_decode=skip_empty_decode,
            skip_empty_decode_chunked=skip_empty_decode_chunked,
            mixed_bkvp=mixed_bkvp,
            decode_pack=decode_pack,
            shared_meta=shared_meta,
        )
        tail_attend = dcp_token_parallel_op.make_attend(
            sm_scale=sm_scale,
            k_scale=dequant,
            cache_layout=cache_layout,
            packed_rope=rope_spec.packed_rope,
            lkv_dim=lkv_dim,
            split_q=True,
            mixed_body_levers=mixed_levers,
        )
        read_pages = (
            functools.partial(page_gather.gather_pages, row_width=128),
            functools.partial(
                page_gather.gather_pages, row_width=rope_spec.token_bytes // 4
            ),
        )
        attn_out, tail_flat = dcp_token_parallel.unified_indexer_in_branch_qb_mla_oproj(
            layout,
            mesh,
            DCP_AXIS_NAME,
            index_cache,
            q_idx_tp,
            w_idx_tp,
            idx_block_tables,
            kv_cache_nope,
            kv_cache_rope,
            q_c_own,
            w_qb,
            s_qb,
            w_uk_t,
            cos_sin,
            head_topk,
            mla_block_tables,
            seq_lens,
            query_start_loc,
            mla_distribution,
            select_topk=select,
            fused_qb=fused_qb_rope_absorb_split_qpe,
            lead_attend=_lead_attend,
            tail_attend=tail_attend,
            w_uv=w_uv,
            w_o=w_o,
            w_o_scale=s_o,
            num_tokens=num_tokens,
            rank_ids=rank_ids,
            read_pages=read_pages,
            w_uv_replicated=True,
            tie_after_idx_max_cap=tie_after_idx_max_cap,
            tie_lead_min_cap=0,
            epilogue_3d=epilogue_3d,
            index_k_packed_u32=True,
        )
        full_topk = jax.shard_map(
            lambda h, t: jnp.concatenate([h, t], axis=0),
            mesh=mesh,
            in_specs=(shd, shd),
            out_specs=shd,
            check_vma=False,
        )(head_topk, tail_flat)
        return index_cache, kv_cache_nope, kv_cache_rope, full_topk, attn_out

    def _make_wrapper(kv_sharded: bool, k_sharded: bool = False):
        if has_lane:

            def _w_lane(
                idx_c: jax.Array,
                nope: jax.Array,
                rope: jax.Array,
                qb_idx: jax.Array,
                wi: jax.Array,
                kp_idx: jax.Array,
                qc: jax.Array,
                wqb: jax.Array,
                sqb: jax.Array,
                wuk: jax.Array,
                cs: jax.Array,
                kc: jax.Array,
                kp_mla: jax.Array,
                sl: jax.Array,
                idx_bt: jax.Array,
                mla_bt: jax.Array,
                qsl: jax.Array,
                idx_dist: jax.Array,
                mla_dist: jax.Array,
                rk: jax.Array,
                uv: jax.Array,
                wo: jax.Array,
                so: jax.Array,
                ln: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
                return _run(
                    idx_c,
                    nope,
                    rope,
                    qb_idx,
                    wi,
                    kp_idx,
                    qc,
                    wqb,
                    sqb,
                    wuk,
                    cs,
                    kc,
                    kp_mla,
                    sl,
                    idx_bt,
                    mla_bt,
                    qsl,
                    idx_dist,
                    mla_dist,
                    rk,
                    uv,
                    wo,
                    so,
                    ln,
                    kv_sharded=kv_sharded,
                    k_sharded=k_sharded,
                )

            return _w_lane

        def _w_nolane(
            idx_c: jax.Array,
            nope: jax.Array,
            rope: jax.Array,
            qb_idx: jax.Array,
            wi: jax.Array,
            kp_idx: jax.Array,
            qc: jax.Array,
            wqb: jax.Array,
            sqb: jax.Array,
            wuk: jax.Array,
            cs: jax.Array,
            kc: jax.Array,
            kp_mla: jax.Array,
            sl: jax.Array,
            idx_bt: jax.Array,
            mla_bt: jax.Array,
            qsl: jax.Array,
            idx_dist: jax.Array,
            mla_dist: jax.Array,
            rk: jax.Array,
            uv: jax.Array,
            wo: jax.Array,
            so: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return _run(
                idx_c,
                nope,
                rope,
                qb_idx,
                wi,
                kp_idx,
                qc,
                wqb,
                sqb,
                wuk,
                cs,
                kc,
                kp_mla,
                sl,
                idx_bt,
                mla_bt,
                qsl,
                idx_dist,
                mla_dist,
                rk,
                uv,
                wo,
                so,
                None,
                kv_sharded=kv_sharded,
                k_sharded=k_sharded,
            )

        return _w_nolane

    def _make_fake(kv_sharded: bool):
        def _fake(
            index_cache,
            kv_cache_nope,
            kv_cache_rope,
            q_idx_bytes,
            w_idx,
            k_packed,
            q_c_own,
            w_qb,
            s_qb,
            w_uk_t,
            cos_sin,
            kv_c_normed,
            k_pe,
            seq_lens,
            idx_bt,
            mla_bt,
            qsl,
            idx_dist,
            mla_dist,
            rank_ids,
            w_uv,
            w_o,
            s_o,
            *lane_arg,
        ):
            _ = (
                w_idx,
                k_packed,
                w_qb,
                s_qb,
                w_uk_t,
                cos_sin,
                k_pe,
                seq_lens,
                idx_bt,
                mla_bt,
                qsl,
                idx_dist,
                mla_dist,
                rank_ids,
                w_uv,
                s_o,
                lane_arg,
            )
            out_rows = (
                kv_c_normed.shape[0]
                if kv_sharded
                else kv_c_normed.shape[0] // layout.dcp_size
            )
            return (
                torch.empty_like(index_cache),
                torch.empty_like(kv_cache_nope),
                torch.empty_like(kv_cache_rope),
                q_idx_bytes.new_empty((q_idx_bytes.shape[0], topk), dtype=torch.int32),
                q_c_own.new_empty((out_rows, w_o.shape[1])),
            )

        return _fake

    def _in_specs(kv_sharded: bool, k_sharded: bool = False) -> tuple[P, ...]:
        kv_spec = shd if kv_sharded else rep
        kp_spec = shd if k_sharded else rep
        base = (
            shd,  # index_cache
            shd,  # kv_cache_nope
            shd,  # kv_cache_rope
            shd,  # q_idx_bytes
            shd,  # w_idx
            kp_spec,  # k_packed
            shd,  # q_c_own
            rep,  # w_qb
            rep,  # s_qb
            rep,  # w_uk_t
            shd,  # cos_sin
            kv_spec,  # kv_c_normed
            kv_spec,  # k_pe
            rep,  # seq_lens
            rep,  # idx_block_tables
            rep,  # mla_block_tables
            rep,  # query_start_loc
            rep,  # idx_distribution
            rep,  # mla_distribution
            shd,  # rank_ids
            rep,  # w_uv
            shd,  # w_o
            shd,  # s_o
        )
        return base + ((shd,) if has_lane else ())

    out_specs = (shd, shd, shd, shd, shd)
    cache_key = (
        id(mesh),
        layout,
        sm_scale,
        dequant,
        mixed_levers,
        decode_levers,
        mla_chunk_tokens,
        tie_after_idx_max_cap,
        epilogue_3d,
        num_heads,
        out_dim,
        topk,
        idx_chunk_tokens,
        bool(hist_merge),
        bool(a2a_merge),
        int(a2a_min_tokens),
        small_bucket,
        kv_cache_mode,
        bool(skip_empty_decode),
        bool(skip_empty_decode_chunked),
        mixed_bkvp,
        bool(decode_pack),
        shared_meta,
        bool(k_write_compact),
        global_group,
        small_ag,
        has_lane,
    )

    def _build_plain(op_name: str):
        built = pcp_streaming_jax_op(
            op_name,
            _make_wrapper(kv_sharded=False),
            donate_argnums=(0, 1, 2),
            mesh=mesh,
            input_partition_specs=_in_specs(kv_sharded=False),
            output_partition_specs=out_specs,
        )
        built.register_fake(_make_fake(kv_sharded=False))
        return built, layers

    plain_op, sharers = get_shared_jax_op(
        "unified_indexer_in_branch_qb_tp_oproj", cache_key, _build_plain
    )
    if getattr(target_layer, "use_sparse", True) and not any(
        ref() is target_layer for ref in sharers
    ):
        sharers.append(weakref.ref(target_layer))

    sharded_kv_op = None
    sharded_kv_k_op = None
    if envs.TPU_DCP_TP_SHARDED_KV_WRITE:

        def _build_sharded(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _make_wrapper(kv_sharded=True),
                donate_argnums=(0, 1, 2),
                mesh=mesh,
                input_partition_specs=_in_specs(kv_sharded=True),
                output_partition_specs=out_specs,
            )
            built.register_fake(_make_fake(kv_sharded=True))
            return built, layers

        sharded_kv_op, _ = get_shared_jax_op(
            "unified_indexer_in_branch_qb_tp_oproj_sharded_kv",
            cache_key,
            _build_sharded,
        )

        def _build_sharded_k(op_name: str):
            built = pcp_streaming_jax_op(
                op_name,
                _make_wrapper(kv_sharded=True, k_sharded=True),
                donate_argnums=(0, 1, 2),
                mesh=mesh,
                input_partition_specs=_in_specs(kv_sharded=True, k_sharded=True),
                output_partition_specs=out_specs,
            )
            built.register_fake(_make_fake(kv_sharded=True))
            return built, layers

        sharded_kv_k_op, _ = get_shared_jax_op(
            "unified_indexer_in_branch_qb_tp_oproj_sharded_kv_k",
            cache_key,
            _build_sharded_k,
        )

    def run(
        index_cache: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        q_idx_bytes: torch.Tensor,
        w_idx: torch.Tensor,
        k_packed: torch.Tensor,
        qb_inputs: dcp_nonidx_qb_mla_op.InBranchQbInputs,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        seq_lens: torch.Tensor,
        idx_block_tables: torch.Tensor,
        mla_block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        idx_distribution: torch.Tensor,
        mla_distribution: torch.Tensor,
        w_uv: torch.Tensor,
        w_o: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_sharded: bool = False,
        k_sharded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nope_cache, rope_cache = kv_cache
        if k_sharded:
            if not kv_sharded:
                raise RuntimeError(
                    "unified indexer+in-branch-qb sharded K requires kv_sharded=True"
                )
            selected = sharded_kv_k_op
        else:
            selected = sharded_kv_op if kv_sharded else plain_op
        if selected is None:
            raise RuntimeError(
                "unified indexer+in-branch-qb sharded KV op requested but not built"
            )
        lane_args = () if lane_tensor is None else (lane_tensor,)
        new_idx, new_nope, new_rope, full_topk, attn_out = selected(
            index_cache,
            nope_cache,
            rope_cache,
            q_idx_bytes,
            w_idx,
            k_packed,
            qb_inputs.q_c,
            qb_inputs.w_qb,
            qb_inputs.s_qb,
            qb_inputs.w_uk_t,
            qb_inputs.cos_sin,
            kv_c_normed,
            k_pe,
            seq_lens,
            idx_block_tables,
            mla_block_tables,
            query_start_loc,
            idx_distribution,
            mla_distribution,
            rank_tensor,
            w_uv,
            w_o[0],
            w_o[1],
            *lane_args,
        )
        index_cache.copy_(new_idx)
        nope_cache.copy_(new_nope)
        rope_cache.copy_(new_rope)
        return full_topk, attn_out

    run.layout = layout  # type: ignore[attr-defined]
    run.serves_sharded_kv = sharded_kv_op is not None  # type: ignore[attr-defined]
    run.serves_sharded_k = sharded_kv_k_op is not None  # type: ignore[attr-defined]
    return run


def unified_indexer_in_branch_qb_attention(
    layer: Any,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    qb_inputs: dcp_nonidx_qb_mla_op.InBranchQbInputs,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    unified_inputs: dcp_unified_indexer_mla_op.UnifiedIndexerInputs,
    attn_metadata: Any,
    w_o: tuple[torch.Tensor, torch.Tensor],
    *,
    kv_sharded: bool = False,
) -> torch.Tensor:
    """Run unified indexer + in-branch q_b + sparse MLA + o_proj and update top-k."""
    unified_qb_op = layer.sparse_mla_unified_idx_qb_op
    if unified_qb_op is None:
        raise RuntimeError("sparse_mla_unified_idx_qb_op was not built on layer")
    q_dtype = qb_inputs.q_c.dtype
    w_uv_src = layer.W_UV_tp_all
    w_uv_arg = w_uv_src.to(q_dtype) if w_uv_src.dtype != q_dtype else w_uv_src

    idx_md = unified_inputs.idx_metadata
    idx_rd = idx_md.request_distribution
    idx_dist = torch.stack([idx_rd[0], idx_rd[0], idx_rd[2]])
    index_cache = unified_inputs.indexer_op.k_cache.kv_cache

    full_topk, attn_out = unified_qb_op(
        index_cache,
        kv_cache,
        unified_inputs.q_bytes,
        unified_inputs.weights,
        unified_inputs.k_packed,
        qb_inputs,
        kv_c_normed,
        k_pe,
        attn_metadata.seq_lens,
        idx_md.block_tables,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        idx_dist,
        attn_metadata.request_distribution,
        w_uv_arg,
        w_o,
        kv_sharded=kv_sharded,
        k_sharded=unified_inputs.k_sharded,
    )
    unified_inputs.indexer_op._write_topk_rows(full_topk.shape[0], full_topk)  # noqa: SLF001 -- updates shared topk buffer in place
    return _retain_rank(attn_out)
