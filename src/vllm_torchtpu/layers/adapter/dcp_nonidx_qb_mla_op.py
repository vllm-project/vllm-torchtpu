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
"""In-branch fused q_b + lead MLA + token-parallel sparse MLA op for non-indexer layers.

Defers `fused_qb_rope_absorb_split_qpe` and `lead_mla` inside `_switch_on_cap`
on non-indexer token-parallel prefill layers so pure-TensorCore `q_b` projection
overlaps Chunk-0 SparseCore KV gather (small caps) and SparseCore-offloaded
`gather_latent` all-gathers (larger caps) with zero extra HBM.
"""

# Avoid `from __future__ import annotations`: `torch_tpu` `pallas._verify_signature`
# checks `annotation is jax.Array` at op registration time and rejects PEP 563 strings.

import dataclasses
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
    dcp_rank_operand,
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
from vllm_torchtpu.layers.adapter import dcp_token_parallel_op
from vllm_torchtpu.layers.adapter.cp_mla_attention import (
    _retain_rank,  # noqa: SLF001 -- existing internal rank-retention helper
)
from vllm_torchtpu.layers.core.attention_interface import (
    _decode_body_levers,  # noqa: SLF001 -- existing internal decode-body lever resolver
    _mixed_body_levers,  # noqa: SLF001 -- existing internal mixed-body lever resolver
    dcp_mla_topk_cap,
    sparse_mla_chunk_kwargs,
)
from vllm_torchtpu.models.vllm.glm_dsa_dispatch import (
    DCP_MLA_SMALL_CHUNK_BATCH,
    DCP_MLA_SMALL_CHUNK_TOKENS,
    MLA_DECODE_ROW_WIDTHS,
    MLA_MIXED_LIVE_WIDTHS,
    get_shared_jax_op,
    glm_dsa_v7x_active,
    resolve_dcp_mla_topk_cap_chunk,
)


@dataclasses.dataclass(frozen=True)
class InBranchQbInputs:
    """Deferred q_b_proj + RoPE + absorb_q operands for non-indexer prefill layers."""

    q_c: torch.Tensor
    w_qb: torch.Tensor
    s_qb: torch.Tensor
    w_uk_t: torch.Tensor
    cos_sin: torch.Tensor
    w_uk_t_hi: torch.Tensor | None = None
    q_ln_weight: torch.Tensor | None = None
    q_ln_eps: float | None = None


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


def build_in_branch_qb_attention_op(
    layer: Any,
    impl: Any,
    layout: dcp_token_parallel.TokenParallelLayout,
    *,
    q_replicated: bool,
):
    """Build a shared in-branch q_b + sparse MLA + local o_proj op."""
    if not (
        glm_dsa_v7x_active()
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
    mixed_levers = _mixed_body_levers(enabled=True)
    decode_levers = _decode_body_levers(enabled=True)
    tp_attend_chunk_tokens = envs.TPU_SPARSE_MLA_CHUNK_TOKENS
    free_nobar_max_cap = 0
    rank_tensor = dcp_rank_operand()
    assert rank_tensor is not None

    target_layer = layer
    num_heads = int(layer.num_heads)
    out_dim = int(layer.v_head_dim)
    del layer, impl

    layers: list[weakref.ref] = []
    shd, rep = P(DCP_AXIS_NAME), P()

    def _run(
        kv_cache_nope: jax.Array,
        kv_cache_rope: jax.Array,
        q_c_own: jax.Array,
        w_qb: jax.Array,
        s_qb: jax.Array,
        w_uk_t: jax.Array | tuple[jax.Array, jax.Array],
        cos_sin: jax.Array,
        kv_c_normed: jax.Array,
        k_pe: jax.Array,
        topk_indices: jax.Array,
        seq_lens: jax.Array,
        block_tables: jax.Array,
        query_start_loc: jax.Array,
        request_distribution: jax.Array,
        rank_ids: jax.Array,
        w_uv: jax.Array,
        w_o: jax.Array,
        s_o: jax.Array,
        *,
        kv_sharded: bool,
        q_ln_w: jax.Array | None = None,
        q_ln_eps: float = 1e-6,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        live = [
            sharer
            for ref in layers
            if (sharer := ref()) is not None and hasattr(sharer, "mla_kv_spec")
        ]
        nope_spec, rope_spec = live[0].mla_kv_spec
        for other in live[1:]:
            if other.mla_kv_spec != (nope_spec, rope_spec):
                raise NotImplementedError(
                    f"{getattr(other, 'layer_name', other)} shares an in-branch "
                    f"q_b MLA op with {getattr(live[0], 'layer_name', live[0])} "
                    f"but recorded different KV cache specs."
                )
        if nope_spec.page_size != layout.page_size:
            raise ValueError(
                f"sparse MLA cache page {nope_spec.page_size} != block size "
                f"{layout.page_size} the token-parallel layout was built for"
            )
        if nope_spec.layout is not rope_spec.layout:
            raise ValueError("sparse MLA needs matching NoPE and RoPE layouts")

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
            block_tables,
            query_start_loc,
            rank_ids,
        )

        s = layout.split_rows
        num_tokens = kv_c_normed.shape[0]
        rows = layout.rows_per_rank(num_tokens)
        head_topk = jax.shard_map(
            lambda x: x[:s], mesh=mesh, in_specs=shd, out_specs=shd, check_vma=False
        )(topk_indices)
        tail_topk = jax.shard_map(
            lambda x: x[s : s + rows],
            mesh=mesh,
            in_specs=shd,
            out_specs=shd,
            check_vma=False,
        )(topk_indices)

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
                "gather_and_attention_chunk_size": resolve_dcp_mla_topk_cap_chunk(
                    envs.TPU_DCP_MLA_TOPK_CAP_CHUNK
                ),
                "capped_live_rows": True,
            }
            if topk_cap
            else dict(sparse_mla_chunk_kwargs())
        )
        if s <= 64 and s % 8 == 0:
            lead_chunk_kwargs["attention_kernel_batch_size"] = 8
        lead_live_widths_raw = (128, *MLA_MIXED_LIVE_WIDTHS)
        lead_live_widths = tuple(
            w for w in lead_live_widths_raw if not topk_cap or w < topk_cap
        )
        decode_row_widths = (
            tuple(w for w in MLA_DECODE_ROW_WIDTHS if w < topk_cap) if topk_cap else ()
        )
        mask_in_kernel = True
        split_q = bool(envs.TPU_MLA_SPARSE_SPLIT_Q or glm_dsa_v7x_active())
        small_chunk_batch = DCP_MLA_SMALL_CHUNK_BATCH
        small_chunk_tokens = DCP_MLA_SMALL_CHUNK_TOKENS
        mixed_live = True
        fused_prep = True
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
            _, qsl_s, dist_s = _dcp_rows_metadata_jax(sl, qsl, dist, s)
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
                qsl_s,
                dist_s,
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
        attn_out = dcp_token_parallel.sparse_mla_attention_with_in_branch_qb_oproj(
            layout,
            mesh,
            DCP_AXIS_NAME,
            kv_cache_nope,
            kv_cache_rope,
            q_c_own,
            w_qb,
            s_qb,
            w_uk_t,
            cos_sin,
            head_topk,
            tail_topk,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
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
            flat_topk_cap_strided=True,
            free_nobar_max_cap=free_nobar_max_cap,
            epilogue_3d=True,
            q_ln_w=q_ln_w,
            q_ln_eps=q_ln_eps,
        )
        return kv_cache_nope, kv_cache_rope, attn_out

    split_w_uk = (
        getattr(target_layer, "W_UK_T_dcp_qrep_lo", None) is not None
        and getattr(target_layer, "W_UK_T_dcp_qrep_hi", None) is not None
    )

    def _make_wrapper(kv_sharded: bool, q_ln_eps: float | None = None):
        if q_ln_eps is not None and split_w_uk:

            def _w_split_qln(
                nope: jax.Array,
                rope: jax.Array,
                qc: jax.Array,
                wqb: jax.Array,
                sqb: jax.Array,
                wuk_lo: jax.Array,
                cs: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                dist: jax.Array,
                rk: jax.Array,
                uv: jax.Array,
                wo: jax.Array,
                so: jax.Array,
                wuk_hi: jax.Array,
                qln_w: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(
                    nope,
                    rope,
                    qc,
                    wqb,
                    sqb,
                    (wuk_lo, wuk_hi),
                    cs,
                    kc,
                    kp,
                    tk,
                    sl,
                    bt,
                    qsl,
                    dist,
                    rk,
                    uv,
                    wo,
                    so,
                    kv_sharded=kv_sharded,
                    q_ln_w=qln_w,
                    q_ln_eps=q_ln_eps,
                )

            return _w_split_qln

        if q_ln_eps is not None:

            def _w_qln(
                nope: jax.Array,
                rope: jax.Array,
                qc: jax.Array,
                wqb: jax.Array,
                sqb: jax.Array,
                wuk: jax.Array,
                cs: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                dist: jax.Array,
                rk: jax.Array,
                uv: jax.Array,
                wo: jax.Array,
                so: jax.Array,
                qln_w: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(
                    nope,
                    rope,
                    qc,
                    wqb,
                    sqb,
                    wuk,
                    cs,
                    kc,
                    kp,
                    tk,
                    sl,
                    bt,
                    qsl,
                    dist,
                    rk,
                    uv,
                    wo,
                    so,
                    kv_sharded=kv_sharded,
                    q_ln_w=qln_w,
                    q_ln_eps=q_ln_eps,
                )

            return _w_qln

        if split_w_uk:

            def _w_split(
                nope: jax.Array,
                rope: jax.Array,
                qc: jax.Array,
                wqb: jax.Array,
                sqb: jax.Array,
                wuk_lo: jax.Array,
                cs: jax.Array,
                kc: jax.Array,
                kp: jax.Array,
                tk: jax.Array,
                sl: jax.Array,
                bt: jax.Array,
                qsl: jax.Array,
                dist: jax.Array,
                rk: jax.Array,
                uv: jax.Array,
                wo: jax.Array,
                so: jax.Array,
                wuk_hi: jax.Array,
            ) -> tuple[jax.Array, jax.Array, jax.Array]:
                return _run(
                    nope,
                    rope,
                    qc,
                    wqb,
                    sqb,
                    (wuk_lo, wuk_hi),
                    cs,
                    kc,
                    kp,
                    tk,
                    sl,
                    bt,
                    qsl,
                    dist,
                    rk,
                    uv,
                    wo,
                    so,
                    kv_sharded=kv_sharded,
                )

            return _w_split

        def _w(
            nope: jax.Array,
            rope: jax.Array,
            qc: jax.Array,
            wqb: jax.Array,
            sqb: jax.Array,
            wuk: jax.Array,
            cs: jax.Array,
            kc: jax.Array,
            kp: jax.Array,
            tk: jax.Array,
            sl: jax.Array,
            bt: jax.Array,
            qsl: jax.Array,
            dist: jax.Array,
            rk: jax.Array,
            uv: jax.Array,
            wo: jax.Array,
            so: jax.Array,
        ) -> tuple[jax.Array, jax.Array, jax.Array]:
            return _run(
                nope,
                rope,
                qc,
                wqb,
                sqb,
                wuk,
                cs,
                kc,
                kp,
                tk,
                sl,
                bt,
                qsl,
                dist,
                rk,
                uv,
                wo,
                so,
                kv_sharded=kv_sharded,
            )

        return _w

    def _make_fake(kv_sharded: bool):
        def _fake(
            kv_cache_nope: torch.Tensor,
            kv_cache_rope: torch.Tensor,
            q_c_own: torch.Tensor,
            w_qb: torch.Tensor,
            s_qb: torch.Tensor,
            w_uk_t: torch.Tensor,
            cos_sin: torch.Tensor,
            kv_c_normed: torch.Tensor,
            k_pe: torch.Tensor,
            topk_indices: torch.Tensor,
            seq_lens: torch.Tensor,
            block_tables: torch.Tensor,
            query_start_loc: torch.Tensor,
            request_distribution: torch.Tensor,
            rank_ids: torch.Tensor,
            w_uv: torch.Tensor,
            w_o: torch.Tensor,
            s_o: torch.Tensor,
            *w_uk_extra: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _ = (
                w_qb,
                s_qb,
                w_uk_t,
                cos_sin,
                k_pe,
                topk_indices,
                seq_lens,
                block_tables,
                query_start_loc,
                request_distribution,
                rank_ids,
                w_uv,
                s_o,
                w_uk_extra,
            )
            out_rows = (
                kv_c_normed.shape[0]
                if kv_sharded
                else kv_c_normed.shape[0] // layout.dcp_size
            )
            return (
                torch.empty_like(kv_cache_nope),
                torch.empty_like(kv_cache_rope),
                q_c_own.new_empty((out_rows, w_o.shape[1])),
            )

        return _fake

    def _in_specs(kv_sharded: bool, *, with_q_ln: bool = False) -> tuple[P, ...]:
        kv_spec = shd if kv_sharded else rep
        base = (
            shd,  # kv_cache_nope
            shd,  # kv_cache_rope
            shd,  # q_c_own
            rep,  # w_qb
            rep,  # s_qb
            rep,  # w_uk_t
            shd,  # cos_sin
            kv_spec,  # kv_c_normed
            kv_spec,  # k_pe
            shd,  # topk_indices
            rep,  # seq_lens
            rep,  # block_tables
            rep,  # query_start_loc
            rep,  # request_distribution
            shd,  # rank_ids
            rep,  # w_uv
            shd,  # w_o
            shd,  # s_o
        )
        if split_w_uk:
            base = (*base, rep)
        if with_q_ln:
            base = (*base, rep)
        return base

    out_specs = (shd, shd, shd)
    cache_key = (
        id(mesh),
        layout,
        sm_scale,
        dequant,
        mixed_levers,
        decode_levers,
        tp_attend_chunk_tokens,
        int(free_nobar_max_cap),
        num_heads,
        out_dim,
        split_w_uk,
    )

    def _build_plain(op_name: str):
        built = pcp_streaming_jax_op(
            op_name,
            _make_wrapper(kv_sharded=False),
            donate_argnums=(0, 1),
            mesh=mesh,
            input_partition_specs=_in_specs(kv_sharded=False),
            output_partition_specs=out_specs,
        )
        built.register_fake(_make_fake(kv_sharded=False))
        return built, layers

    plain_op, sharers = get_shared_jax_op(
        "sparse_mla_in_branch_qb_oproj", cache_key, _build_plain
    )
    if getattr(target_layer, "use_sparse", True) and not any(
        ref() is target_layer for ref in sharers
    ):
        sharers.append(weakref.ref(target_layer))

    def _build_sharded(op_name: str):
        built = pcp_streaming_jax_op(
            op_name,
            _make_wrapper(kv_sharded=True),
            donate_argnums=(0, 1),
            mesh=mesh,
            input_partition_specs=_in_specs(kv_sharded=True),
            output_partition_specs=out_specs,
        )
        built.register_fake(_make_fake(kv_sharded=True))
        return built, layers

    sharded_kv_op, sharded_sharers = get_shared_jax_op(
        "sparse_mla_in_branch_qb_oproj_sharded_kv", cache_key, _build_sharded
    )
    if getattr(target_layer, "use_sparse", True) and not any(
        ref() is target_layer for ref in sharded_sharers
    ):
        sharded_sharers.append(weakref.ref(target_layer))

    qln_op_cache: dict[tuple[bool, float], Any] = {}

    def _prepare_qln_op(eps: float) -> None:
        eps_f = float(eps)
        for kv_sharded in (False, True) if sharded_kv_op is not None else (False,):
            key = (kv_sharded, eps_f)
            if key in qln_op_cache:
                continue
            prefix = (
                "sparse_mla_in_branch_qb_qln_oproj_sharded_kv"
                if kv_sharded
                else "sparse_mla_in_branch_qb_qln_oproj"
            )

            def _build_qln(
                op_name: str, _kv_sharded: bool = kv_sharded, _eps: float = eps_f
            ):
                built = pcp_streaming_jax_op(
                    op_name,
                    _make_wrapper(kv_sharded=_kv_sharded, q_ln_eps=_eps),
                    donate_argnums=(0, 1),
                    mesh=mesh,
                    input_partition_specs=_in_specs(
                        kv_sharded=_kv_sharded, with_q_ln=True
                    ),
                    output_partition_specs=out_specs,
                )
                built.register_fake(_make_fake(kv_sharded=_kv_sharded))
                return built, layers

            op, op_sharers = get_shared_jax_op(prefix, (*cache_key, eps_f), _build_qln)
            if getattr(target_layer, "use_sparse", True) and not any(
                ref() is target_layer for ref in op_sharers
            ):
                op_sharers.append(weakref.ref(target_layer))
            qln_op_cache[key] = op

    eager_q_ln_eps = getattr(target_layer, "in_branch_q_ln_eps", None)
    if eager_q_ln_eps is not None:
        _prepare_qln_op(float(eager_q_ln_eps))

    def run(
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        qb_inputs: InBranchQbInputs,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        topk_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        query_start_loc: torch.Tensor,
        request_distribution: torch.Tensor,
        w_uv: torch.Tensor,
        w_o: tuple[torch.Tensor, torch.Tensor],
        *,
        kv_sharded: bool = False,
    ) -> torch.Tensor:
        nope_cache, rope_cache = kv_cache
        use_q_ln = qb_inputs.q_ln_weight is not None and qb_inputs.q_ln_eps is not None
        if use_q_ln:
            assert qb_inputs.q_ln_eps is not None
            key = (kv_sharded, float(qb_inputs.q_ln_eps))
            selected = qln_op_cache.get(key)
            if selected is None:
                raise RuntimeError(
                    "in-branch q_b q_ln op was not registered for "
                    f"kv_sharded={kv_sharded}, eps={qb_inputs.q_ln_eps}"
                )
        else:
            selected = sharded_kv_op if kv_sharded else plain_op
            if selected is None:
                raise RuntimeError(
                    "in-branch q_b sharded KV op requested but not built"
                )
        extra_args: list[torch.Tensor] = []
        if split_w_uk:
            assert qb_inputs.w_uk_t_hi is not None
            extra_args.append(qb_inputs.w_uk_t_hi)
        if use_q_ln:
            assert qb_inputs.q_ln_weight is not None
            extra_args.append(qb_inputs.q_ln_weight)
        new_nope, new_rope, attn_out = selected(
            nope_cache,
            rope_cache,
            qb_inputs.q_c,
            qb_inputs.w_qb,
            qb_inputs.s_qb,
            qb_inputs.w_uk_t,
            qb_inputs.cos_sin,
            kv_c_normed,
            k_pe,
            topk_indices,
            seq_lens,
            block_tables,
            query_start_loc,
            request_distribution,
            rank_tensor,
            w_uv,
            w_o[0],
            w_o[1],
            *extra_args,
        )
        nope_cache.copy_(new_nope)
        rope_cache.copy_(new_rope)
        return attn_out

    def _has_q_ln(eps: float, kv_sharded: bool = False) -> bool:
        return (kv_sharded, float(eps)) in qln_op_cache

    run.layout = layout  # type: ignore[attr-defined]
    run.serves_sharded_kv = sharded_kv_op is not None  # type: ignore[attr-defined]
    run.prepare_q_ln = _prepare_qln_op  # type: ignore[attr-defined]
    run.has_q_ln = _has_q_ln  # type: ignore[attr-defined]
    return run


def in_branch_qb_attention(
    layer: Any,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    qb_inputs: InBranchQbInputs,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    topk_indices: torch.Tensor,
    attn_metadata: Any,
    w_o: tuple[torch.Tensor, torch.Tensor],
    *,
    kv_sharded: bool = False,
) -> torch.Tensor:
    """Run deferred in-branch q_b + sparse MLA + o_proj and update KV caches."""
    qb_op = layer.sparse_mla_in_branch_qb_op
    if qb_op is None:
        raise RuntimeError("sparse_mla_in_branch_qb_op was not built on layer")
    q_dtype = qb_inputs.q_c.dtype
    w_uv_src = layer.W_UV_tp_all
    w_uv_arg = w_uv_src.to(q_dtype) if w_uv_src.dtype != q_dtype else w_uv_src
    attn_out = qb_op(
        kv_cache,
        qb_inputs,
        kv_c_normed,
        k_pe,
        topk_indices,
        attn_metadata.seq_lens,
        attn_metadata.block_tables,
        attn_metadata.query_start_loc,
        attn_metadata.request_distribution,
        w_uv_arg,
        w_o,
        kv_sharded=kv_sharded,
    )
    return _retain_rank(attn_out)
